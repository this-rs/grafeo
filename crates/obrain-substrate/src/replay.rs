//! WAL replay — re-apply records to the mmap after a crash.
//!
//! Replay is called at boot from `SubstrateStore::open` (T4). It reads the
//! WAL from `last_checkpoint_offset` up to the first decode error (the crash
//! boundary) and re-applies each record to the zone mmaps.
//!
//! ## Idempotence
//!
//! WAL payloads carry full state, never deltas (see `wal-spec.md` §2), so
//! replaying an already-durable record is a no-op:
//!
//! * `NodeInsert` — overwrites the fixed-offset slot with identical bytes.
//! * `EdgeInsert` — same.
//! * `Checkpoint` — advisory only; used by the caller to trim the WAL.
//!
//! Records that cannot yet be applied (property pages, string heap, cognitive
//! ops) are accepted but counted as `skipped` for now — they'll be wired in
//! as T6/T7/T8 land on top of this replay primitive.

use crate::error::SubstrateResult;
use crate::file::{SubstrateFile, Zone, ZoneFile};
use crate::record::{EdgeRecord, NodeRecord, U48};
use crate::wal::{WalPayload, WalRecord};
use crate::wal_io::WalReader;
use crate::writer::{
    apply_coact_decay_to_zone, apply_energy_decay_to_zone, apply_synapse_decay_to_zone,
    ensure_room,
};
use tracing::{debug, warn};

/// Outcome of a replay pass.
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq)]
pub struct ReplayStats {
    /// Records decoded and applied cleanly.
    pub applied: u64,
    /// Records decoded but whose kind this replay version cannot apply yet
    /// (property / string / cognitive payloads — added in later tasks).
    pub skipped_unsupported: u64,
    /// Records that failed to decode (CRC mismatch, short read). The iterator
    /// stops at the first such error. Expected 0 after a clean shutdown, 1 if
    /// the tail of the WAL was torn by a crash.
    pub decode_errors: u64,
    /// Byte offset at which replay stopped. Points to the crash boundary.
    pub stopped_at: u64,
    /// Highest LSN seen during replay (or 0 if no records).
    pub last_lsn: u64,
}

/// Replay the WAL of `substrate` starting at `from_offset`.
///
/// The function opens the substrate's WAL file, decodes records in order,
/// and applies each to the corresponding zone mmap. It stops at the first
/// decode error.
pub fn replay_from(substrate: &SubstrateFile, from_offset: u64) -> SubstrateResult<ReplayStats> {
    let wal_path = substrate.wal_path();
    let reader = WalReader::open(&wal_path)?;
    let mut stats = ReplayStats {
        stopped_at: from_offset,
        ..Default::default()
    };
    if reader.is_empty() {
        return Ok(stats);
    }

    // Lazily open zones we need.
    let mut nodes: Option<ZoneFile> = None;
    let mut edges: Option<ZoneFile> = None;
    let mut engram_members: Option<ZoneFile> = None;
    let mut engram_bitset: Option<ZoneFile> = None;
    let mut hilbert: Option<ZoneFile> = None;
    let mut community: Option<ZoneFile> = None;

    for item in reader.iter_from(from_offset) {
        match item {
            Ok((rec, off, len)) => {
                stats.last_lsn = stats.last_lsn.max(rec.lsn);
                match apply(
                    substrate,
                    &rec,
                    &mut nodes,
                    &mut edges,
                    &mut engram_members,
                    &mut engram_bitset,
                    &mut hilbert,
                    &mut community,
                )? {
                    Applied::Yes => stats.applied += 1,
                    Applied::Skipped => stats.skipped_unsupported += 1,
                }
                stats.stopped_at = off + len as u64;
            }
            Err(e) => {
                warn!(
                    error = %e,
                    stopped_at = stats.stopped_at,
                    "wal replay halted at crash boundary"
                );
                stats.decode_errors += 1;
                break;
            }
        }
    }

    // Msync + fsync whatever zones we touched so the replay state is durable.
    if let Some(zf) = nodes.as_ref() {
        zf.msync()?;
        zf.fsync()?;
    }
    if let Some(zf) = edges.as_ref() {
        zf.msync()?;
        zf.fsync()?;
    }
    if let Some(zf) = engram_members.as_ref() {
        zf.msync()?;
        zf.fsync()?;
    }
    if let Some(zf) = engram_bitset.as_ref() {
        zf.msync()?;
        zf.fsync()?;
    }
    if let Some(zf) = hilbert.as_ref() {
        zf.msync()?;
        zf.fsync()?;
    }
    if let Some(zf) = community.as_ref() {
        zf.msync()?;
        zf.fsync()?;
    }

    debug!(?stats, "wal replay complete");
    Ok(stats)
}

enum Applied {
    Yes,
    Skipped,
}

fn apply(
    substrate: &SubstrateFile,
    rec: &WalRecord,
    nodes: &mut Option<ZoneFile>,
    edges: &mut Option<ZoneFile>,
    engram_members: &mut Option<ZoneFile>,
    engram_bitset: &mut Option<ZoneFile>,
    hilbert: &mut Option<ZoneFile>,
    community: &mut Option<ZoneFile>,
) -> SubstrateResult<Applied> {
    match &rec.payload {
        WalPayload::NodeInsert {
            node_id,
            label_bitset,
        } => {
            if nodes.is_none() {
                *nodes = Some(substrate.open_zone(Zone::Nodes)?);
            }
            let zf = nodes.as_mut().unwrap();
            let offset = (*node_id as usize) * NodeRecord::SIZE;
            ensure_room(zf, offset + NodeRecord::SIZE, 1 << 20)?;
            // The WAL records the *label_bitset* specifically for NodeInsert;
            // the rest of the NodeRecord is filled in later by subsequent
            // records (NodeUpdate, EnergyReinforce, etc.). Idempotent: writing
            // the same label_bitset twice is a no-op.
            let existing: &NodeRecord = bytemuck::from_bytes(
                &zf.as_slice()[offset..offset + NodeRecord::SIZE],
            );
            let mut updated = *existing;
            updated.label_bitset = *label_bitset;
            zf.as_slice_mut()[offset..offset + NodeRecord::SIZE]
                .copy_from_slice(bytemuck::bytes_of(&updated));
            Ok(Applied::Yes)
        }

        WalPayload::NodeUpdate {
            node_id,
            label_bitset,
            energy,
            scar_util_affinity,
            centrality_cached,
            flags,
            community_id,
        } => {
            if nodes.is_none() {
                *nodes = Some(substrate.open_zone(Zone::Nodes)?);
            }
            let zf = nodes.as_mut().unwrap();
            let offset = (*node_id as usize) * NodeRecord::SIZE;
            ensure_room(zf, offset + NodeRecord::SIZE, 1 << 20)?;
            let existing: &NodeRecord = bytemuck::from_bytes(
                &zf.as_slice()[offset..offset + NodeRecord::SIZE],
            );
            let mut updated = *existing;
            updated.label_bitset = *label_bitset;
            updated.energy = *energy;
            updated.scar_util_affinity = *scar_util_affinity;
            updated.centrality_cached = *centrality_cached;
            updated.flags = *flags;
            updated.community_id = *community_id;
            zf.as_slice_mut()[offset..offset + NodeRecord::SIZE]
                .copy_from_slice(bytemuck::bytes_of(&updated));
            Ok(Applied::Yes)
        }

        WalPayload::EdgeInsert {
            edge_id,
            src,
            dst,
            edge_type,
            weight_u16,
        } => {
            if edges.is_none() {
                *edges = Some(substrate.open_zone(Zone::Edges)?);
            }
            let zf = edges.as_mut().unwrap();
            let offset = (*edge_id as usize) * EdgeRecord::SIZE;
            ensure_room(zf, offset + EdgeRecord::SIZE, 1 << 20)?;
            let existing: &EdgeRecord = bytemuck::from_bytes(
                &zf.as_slice()[offset..offset + EdgeRecord::SIZE],
            );
            let mut updated = *existing;
            updated.src = *src;
            updated.dst = *dst;
            updated.edge_type = *edge_type;
            updated.weight_u16 = *weight_u16;
            zf.as_slice_mut()[offset..offset + EdgeRecord::SIZE]
                .copy_from_slice(bytemuck::bytes_of(&updated));
            Ok(Applied::Yes)
        }

        WalPayload::NodeDelete { node_id } => {
            if nodes.is_none() {
                *nodes = Some(substrate.open_zone(Zone::Nodes)?);
            }
            let zf = nodes.as_mut().unwrap();
            let offset = (*node_id as usize) * NodeRecord::SIZE;
            if offset + NodeRecord::SIZE > zf.as_slice().len() {
                // Slot never materialised — tombstone is implicit.
                return Ok(Applied::Yes);
            }
            let existing: &NodeRecord = bytemuck::from_bytes(
                &zf.as_slice()[offset..offset + NodeRecord::SIZE],
            );
            let mut updated = *existing;
            updated.set_tombstoned();
            zf.as_slice_mut()[offset..offset + NodeRecord::SIZE]
                .copy_from_slice(bytemuck::bytes_of(&updated));
            Ok(Applied::Yes)
        }

        WalPayload::EdgeUpdate {
            edge_id,
            weight_u16,
            ricci_u8,
            flags,
            engram_tag,
        } => {
            if edges.is_none() {
                *edges = Some(substrate.open_zone(Zone::Edges)?);
            }
            let zf = edges.as_mut().unwrap();
            let offset = (*edge_id as usize) * EdgeRecord::SIZE;
            if offset + EdgeRecord::SIZE > zf.as_slice().len() {
                // Slot never materialised; skip silently.
                return Ok(Applied::Yes);
            }
            let existing: &EdgeRecord = bytemuck::from_bytes(
                &zf.as_slice()[offset..offset + EdgeRecord::SIZE],
            );
            let mut updated = *existing;
            updated.weight_u16 = *weight_u16;
            updated.ricci_u8 = *ricci_u8;
            updated.flags = *flags;
            updated.engram_tag = *engram_tag;
            zf.as_slice_mut()[offset..offset + EdgeRecord::SIZE]
                .copy_from_slice(bytemuck::bytes_of(&updated));
            Ok(Applied::Yes)
        }

        WalPayload::EdgeDelete { edge_id } => {
            if edges.is_none() {
                *edges = Some(substrate.open_zone(Zone::Edges)?);
            }
            let zf = edges.as_mut().unwrap();
            let offset = (*edge_id as usize) * EdgeRecord::SIZE;
            if offset + EdgeRecord::SIZE > zf.as_slice().len() {
                return Ok(Applied::Yes);
            }
            let existing: &EdgeRecord = bytemuck::from_bytes(
                &zf.as_slice()[offset..offset + EdgeRecord::SIZE],
            );
            let mut updated = *existing;
            updated.flags |= crate::record::edge_flags::TOMBSTONED;
            zf.as_slice_mut()[offset..offset + EdgeRecord::SIZE]
                .copy_from_slice(bytemuck::bytes_of(&updated));
            Ok(Applied::Yes)
        }

        // ---- Cognitive column payloads (T6) --------------------------------

        WalPayload::EnergyReinforce {
            node_id,
            new_energy,
        } => {
            if nodes.is_none() {
                *nodes = Some(substrate.open_zone(Zone::Nodes)?);
            }
            let zf = nodes.as_mut().unwrap();
            let offset = (*node_id as usize) * NodeRecord::SIZE;
            ensure_room(zf, offset + NodeRecord::SIZE, 1 << 20)?;
            let existing: &NodeRecord = bytemuck::from_bytes(
                &zf.as_slice()[offset..offset + NodeRecord::SIZE],
            );
            let mut updated = *existing;
            updated.energy = *new_energy;
            zf.as_slice_mut()[offset..offset + NodeRecord::SIZE]
                .copy_from_slice(bytemuck::bytes_of(&updated));
            Ok(Applied::Yes)
        }

        WalPayload::EnergyDecay { factor_q16 } => {
            if nodes.is_none() {
                *nodes = Some(substrate.open_zone(Zone::Nodes)?);
            }
            let zf = nodes.as_mut().unwrap();
            // Decay every live slot in the zone. Use u32::MAX so the
            // function walks to end-of-zone — unknown high-water at replay
            // time, but beyond zone length the loop breaks. Tombstoned
            // slots are skipped internally.
            let slots = (zf.len() as usize / NodeRecord::SIZE) as u32;
            apply_energy_decay_to_zone(zf, *factor_q16, slots);
            Ok(Applied::Yes)
        }

        WalPayload::SynapseReinforce {
            edge_id,
            new_weight,
        } => {
            if edges.is_none() {
                *edges = Some(substrate.open_zone(Zone::Edges)?);
            }
            let zf = edges.as_mut().unwrap();
            let offset = (*edge_id as usize) * EdgeRecord::SIZE;
            ensure_room(zf, offset + EdgeRecord::SIZE, 1 << 20)?;
            let existing: &EdgeRecord = bytemuck::from_bytes(
                &zf.as_slice()[offset..offset + EdgeRecord::SIZE],
            );
            let mut updated = *existing;
            updated.weight_u16 = *new_weight;
            zf.as_slice_mut()[offset..offset + EdgeRecord::SIZE]
                .copy_from_slice(bytemuck::bytes_of(&updated));
            Ok(Applied::Yes)
        }

        WalPayload::SynapseDecay { factor_q16 } => {
            if edges.is_none() {
                *edges = Some(substrate.open_zone(Zone::Edges)?);
            }
            let zf = edges.as_mut().unwrap();
            // Walk every live slot to end-of-zone. Unknown high-water at
            // replay time — the loop breaks when offset+stride exceeds
            // zone length. Tombstoned slots are skipped internally.
            let slots = (zf.len() as usize / EdgeRecord::SIZE) as u64;
            apply_synapse_decay_to_zone(zf, *factor_q16, slots);
            Ok(Applied::Yes)
        }

        WalPayload::CoactDecay {
            factor_q16,
            coact_type_id,
        } => {
            if edges.is_none() {
                *edges = Some(substrate.open_zone(Zone::Edges)?);
            }
            let zf = edges.as_mut().unwrap();
            let slots = (zf.len() as usize / EdgeRecord::SIZE) as u64;
            apply_coact_decay_to_zone(zf, *factor_q16, slots, *coact_type_id);
            Ok(Applied::Yes)
        }

        WalPayload::ScarUtilAffinitySet { node_id, packed } => {
            if nodes.is_none() {
                *nodes = Some(substrate.open_zone(Zone::Nodes)?);
            }
            let zf = nodes.as_mut().unwrap();
            let offset = (*node_id as usize) * NodeRecord::SIZE;
            ensure_room(zf, offset + NodeRecord::SIZE, 1 << 20)?;
            let existing: &NodeRecord = bytemuck::from_bytes(
                &zf.as_slice()[offset..offset + NodeRecord::SIZE],
            );
            let mut updated = *existing;
            updated.scar_util_affinity = *packed;
            zf.as_slice_mut()[offset..offset + NodeRecord::SIZE]
                .copy_from_slice(bytemuck::bytes_of(&updated));
            Ok(Applied::Yes)
        }

        WalPayload::CentralityUpdate { updates } => {
            if nodes.is_none() {
                *nodes = Some(substrate.open_zone(Zone::Nodes)?);
            }
            let zf = nodes.as_mut().unwrap();
            for (node_id, centrality) in updates {
                let offset = (*node_id as usize) * NodeRecord::SIZE;
                ensure_room(zf, offset + NodeRecord::SIZE, 1 << 20)?;
                let existing: &NodeRecord = bytemuck::from_bytes(
                    &zf.as_slice()[offset..offset + NodeRecord::SIZE],
                );
                let mut updated = *existing;
                updated.centrality_cached = *centrality;
                updated.flags &= !crate::record::node_flags::CENTRALITY_STALE;
                zf.as_slice_mut()[offset..offset + NodeRecord::SIZE]
                    .copy_from_slice(bytemuck::bytes_of(&updated));
            }
            Ok(Applied::Yes)
        }

        // Community drive — T10. LDleiden persists node→community assignments
        // as a batch of `(node_id, community_id)` tuples. Replay writes the
        // column idempotently: re-applying the same batch overwrites the slot
        // with the same bytes.
        WalPayload::CommunityAssign { updates } => {
            if nodes.is_none() {
                *nodes = Some(substrate.open_zone(Zone::Nodes)?);
            }
            let zf = nodes.as_mut().unwrap();
            for (node_id, community_id) in updates {
                let offset = (*node_id as usize) * NodeRecord::SIZE;
                ensure_room(zf, offset + NodeRecord::SIZE, 1 << 20)?;
                let existing: &NodeRecord = bytemuck::from_bytes(
                    &zf.as_slice()[offset..offset + NodeRecord::SIZE],
                );
                let mut updated = *existing;
                updated.community_id = *community_id;
                zf.as_slice_mut()[offset..offset + NodeRecord::SIZE]
                    .copy_from_slice(bytemuck::bytes_of(&updated));
            }
            Ok(Applied::Yes)
        }

        // Engram-membership side-table (T7).
        WalPayload::EngramMembersSet {
            engram_id,
            members,
        } => {
            if engram_members.is_none() {
                *engram_members = Some(substrate.open_zone(Zone::EngramMembers)?);
            }
            let zf = engram_members.as_mut().unwrap();
            crate::engram::EngramZone::set_members_raw(zf, *engram_id, members)?;
            Ok(Applied::Yes)
        }

        // Per-node engram bitset column (T7 Step 1).
        WalPayload::EngramBitsetSet { node_id, bitset } => {
            if engram_bitset.is_none() {
                *engram_bitset = Some(substrate.open_zone(Zone::EngramBitset)?);
            }
            let zf = engram_bitset.as_mut().unwrap();
            crate::engram_bitset::EngramBitsetColumn::set_raw(zf, *node_id, *bitset)?;
            Ok(Applied::Yes)
        }

        // Hilbert page ordering (T11 Step 2).
        //
        // Idempotency story: the payload carries the `old_to_new`
        // permutation that was applied at first-sort time. We only
        // re-apply it if the substrate's meta header has NOT yet set
        // [`meta_flags::HILBERT_SORTED`] — i.e. we are replaying a WAL
        // segment that includes the sort event but the mmap state still
        // reflects the pre-sort layout (the case after a crash between
        // WAL-append and the subsequent meta-flag write).
        //
        // If the flag is already set, the mmap is already sorted (or
        // post-sort mutations landed on top) — re-applying would corrupt.
        // Silent skip is the correct behaviour. This is the Step 2
        // contract; the full transactional variant (T11 Step 4) uses an
        // out-of-place journal so the replay handler becomes a true
        // "re-run the whole thing" without pre-state assumptions.
        WalPayload::HilbertRepermute { permutation } => {
            let already_sorted = substrate.meta_header().flags
                & crate::meta::meta_flags::HILBERT_SORTED
                != 0;
            if already_sorted {
                return Ok(Applied::Skipped);
            }
            let n = permutation.len();

            // Permute nodes out-of-place.
            if nodes.is_none() {
                *nodes = Some(substrate.open_zone(Zone::Nodes)?);
            }
            let zf = nodes.as_mut().unwrap();
            let total = n * NodeRecord::SIZE;
            ensure_room(zf, total, 1 << 20)?;
            let src: Vec<NodeRecord> = bytemuck::cast_slice::<u8, NodeRecord>(
                &zf.as_slice()[..total],
            )
            .to_vec();
            let mut dst = vec![NodeRecord::default(); n];
            for old in 0..n {
                let new = permutation[old] as usize;
                if new < n {
                    dst[new] = src[old];
                }
            }
            zf.as_slice_mut()[..total].copy_from_slice(bytemuck::cast_slice(&dst));

            // Rewrite edge.src/dst in place.
            if edges.is_none() {
                *edges = Some(substrate.open_zone(Zone::Edges)?);
            }
            let ezf = edges.as_mut().unwrap();
            let edge_slots = ezf.len() as usize / EdgeRecord::SIZE;
            for idx in 0..edge_slots {
                let offset = idx * EdgeRecord::SIZE;
                let rec: EdgeRecord =
                    *bytemuck::from_bytes(&ezf.as_slice()[offset..offset + EdgeRecord::SIZE]);
                if rec.is_tombstoned() {
                    continue;
                }
                let mut updated = rec;
                if (rec.src as usize) < n {
                    updated.src = permutation[rec.src as usize];
                }
                if (rec.dst as usize) < n {
                    updated.dst = permutation[rec.dst as usize];
                }
                ezf.as_slice_mut()[offset..offset + EdgeRecord::SIZE]
                    .copy_from_slice(bytemuck::bytes_of(&updated));
            }
            // Set the flag so a subsequent replay skips this record.
            // (The writer also sets it at the end of the sort operation;
            // we duplicate the write here so replay converges regardless
            // of which side crashed.)
            let mut h = substrate.meta_header();
            h.flags |= crate::meta::meta_flags::HILBERT_SORTED;
            // We cannot call `write_meta_header` on the immutable
            // `substrate` borrow here — so callers of `replay_from` that
            // care about the flag persisting to disk must re-open the
            // substrate as `&mut` or rely on the writer's own flag
            // write at end-of-sort. For Step 2, the flag persistence is
            // the writer's responsibility; replay only ensures the in-
            // memory state matches.
            Ok(Applied::Yes)
        }

        // Transactional per-community compaction (T11 Step 4).
        //
        // Mirrors [`Writer::compact_community`] phases 6-8 (node relocate, edge
        // endpoint remap, side-column rewrite). The WAL-first ordering in the
        // writer (record appended + fsync'd *before* any mmap mutation) lets
        // replay recover any crash between the log append and the completion
        // of phase 8 or the subsequent `msync` in phase 10.
        //
        // ## Idempotency contract
        //
        // We must handle three crash-window states uniformly:
        //
        //   * **A — pre-apply**: no relocation has landed. `node[old]` still
        //     carries `community_id == target`. Replay applies every `(old,
        //     new)` move.
        //   * **B — partial**: some relocations have landed (node copied to
        //     `new`, old slot zero-filled); others have not. Replay applies
        //     only the still-pending moves.
        //   * **C — post-apply**: every relocation has already been mirrored to
        //     the mmap. All old slots are zero-filled, no more edges reference
        //     the old domain. Replay is a no-op.
        //
        // The test `rec.community_id == target` at the *old* slot is the
        // discriminator: it holds in (A) and partially in (B), and fails in
        // (C) because the old slot was zero-filled on first apply (so its
        // `community_id` is 0, never equal to a live target ≥ 1). The writer
        // rejects `community_id == 0` at call time, so target ≥ 1 is invariant.
        //
        // Edges are naturally idempotent: `old_slots` is a closed set, and
        // once a `(src, dst)` pair has been remapped the new values live
        // outside that set, so `old_slots.contains(&rec.src)` is false on
        // subsequent passes and the remap becomes a no-op.
        //
        // ## Side-column caveats
        //
        // * `community_id` column — deterministic from the WAL payload
        //   (`target` cid at every new slot, 0 at every old slot). Always
        //   rewritten by replay.
        // * `hilbert_key` column — not recoverable from the WAL payload (the
        //   keys are a function of (centrality, degree) sampled at compaction
        //   time by `Writer::compact_community` phase 2). We zero the old
        //   entries for cleanliness but leave the new-slot entries untouched:
        //   if the writer's phase 8 completed before the crash, they carry
        //   the correct hkey; if not, they stay zero and the next
        //   `CommunityWarden` tick recomputes them. Hilbert is an
        //   *optimisation hint* (locality prefetch), not a correctness
        //   invariant — degraded locality is tolerable, incorrect topology
        //   is not.
        WalPayload::CompactCommunity {
            community_id,
            relocations,
            page_range: _,
        } => {
            // target cid 0 is meaningless — the writer rejects it, and letting
            // it through would break the "zero-filled slot ⇒ not-this-cid"
            // idempotency invariant.
            if *community_id == 0 || relocations.is_empty() {
                return Ok(Applied::Skipped);
            }

            // Open the four zones we touch (lazy, cached across records).
            if nodes.is_none() {
                *nodes = Some(substrate.open_zone(Zone::Nodes)?);
            }
            if edges.is_none() {
                *edges = Some(substrate.open_zone(Zone::Edges)?);
            }
            if hilbert.is_none() {
                *hilbert = Some(substrate.open_zone(Zone::Hilbert)?);
            }
            if community.is_none() {
                *community = Some(substrate.open_zone(Zone::Community)?);
            }

            // Size-up every zone to hold the highest new slot.
            let max_new = relocations.iter().map(|(_, new)| *new).max().unwrap_or(0) as usize;
            ensure_room(
                nodes.as_mut().unwrap(),
                (max_new + 1) * NodeRecord::SIZE,
                1 << 20,
            )?;
            ensure_room(hilbert.as_mut().unwrap(), (max_new + 1) * 4, 1 << 20)?;
            ensure_room(community.as_mut().unwrap(), (max_new + 1) * 4, 1 << 20)?;

            // Lookup structures for the edge remap.
            let old_to_new: std::collections::HashMap<u32, u32> =
                relocations.iter().copied().collect();
            let old_slots: std::collections::HashSet<u32> =
                relocations.iter().map(|(old, _)| *old).collect();

            // ---- Phase 6 — node relocate (idempotent via cid-match) -------
            {
                let zf = nodes.as_mut().unwrap();
                // Snapshot still-pending moves so we can then mutate the zone
                // without aliasing the immutable read borrow.
                let mut pending: Vec<(u32, u32, NodeRecord)> =
                    Vec::with_capacity(relocations.len());
                {
                    let src = zf.as_slice();
                    for (old, new) in relocations.iter() {
                        let off = (*old as usize) * NodeRecord::SIZE;
                        if off + NodeRecord::SIZE > src.len() {
                            // Old slot never materialised — nothing to move.
                            continue;
                        }
                        let rec: &NodeRecord =
                            bytemuck::from_bytes(&src[off..off + NodeRecord::SIZE]);
                        if rec.community_id == *community_id {
                            pending.push((*old, *new, *rec));
                        }
                    }
                }
                let dst = zf.as_slice_mut();
                let zero = NodeRecord::default();
                for (old, new, rec) in &pending {
                    let new_off = (*new as usize) * NodeRecord::SIZE;
                    dst[new_off..new_off + NodeRecord::SIZE]
                        .copy_from_slice(bytemuck::bytes_of(rec));
                    let old_off = (*old as usize) * NodeRecord::SIZE;
                    dst[old_off..old_off + NodeRecord::SIZE]
                        .copy_from_slice(bytemuck::bytes_of(&zero));
                }
            }

            // ---- Phase 7 — edge endpoint remap (idempotent via old_slots) -
            {
                let zf = edges.as_mut().unwrap();
                let edge_slots = zf.as_slice().len() / EdgeRecord::SIZE;
                let dst = zf.as_slice_mut();
                for idx in 0..edge_slots {
                    let off = idx * EdgeRecord::SIZE;
                    let rec: EdgeRecord =
                        *bytemuck::from_bytes(&dst[off..off + EdgeRecord::SIZE]);
                    if rec.is_tombstoned() {
                        continue;
                    }
                    let src_in = old_slots.contains(&rec.src);
                    let dst_in = old_slots.contains(&rec.dst);
                    if !src_in && !dst_in {
                        continue;
                    }
                    let mut updated = rec;
                    if src_in {
                        if let Some(&ns) = old_to_new.get(&rec.src) {
                            updated.src = ns;
                        }
                    }
                    if dst_in {
                        if let Some(&nd) = old_to_new.get(&rec.dst) {
                            updated.dst = nd;
                        }
                    }
                    dst[off..off + EdgeRecord::SIZE]
                        .copy_from_slice(bytemuck::bytes_of(&updated));
                }
            }

            // ---- Phase 8a — community side-column (deterministic) ---------
            {
                let zf = community.as_mut().unwrap();
                let bytes = zf.as_slice_mut();
                let cid_le = community_id.to_le_bytes();
                let zero_le = 0u32.to_le_bytes();
                for (_, new) in relocations.iter() {
                    let off = (*new as usize) * 4;
                    if off + 4 <= bytes.len() {
                        bytes[off..off + 4].copy_from_slice(&cid_le);
                    }
                }
                for (old, _) in relocations.iter() {
                    let off = (*old as usize) * 4;
                    if off + 4 <= bytes.len() {
                        bytes[off..off + 4].copy_from_slice(&zero_le);
                    }
                }
            }

            // ---- Phase 8b — Hilbert side-column (zero old slots only) -----
            //
            // We cannot recompute hilbert_keys at replay time (would need a
            // full centrality+degree pass). The writer's phase 8 has already
            // written the correct keys for new slots if it completed; if it
            // crashed before phase 8, the new-slot hilbert entries stay zero
            // and the next CommunityWarden tick will refresh them. Either way
            // the topology of nodes+edges is correct; only retrieval locality
            // is temporarily degraded.
            {
                let zf = hilbert.as_mut().unwrap();
                let bytes = zf.as_slice_mut();
                let zero_le = 0u32.to_le_bytes();
                for (old, _) in relocations.iter() {
                    let off = (*old as usize) * 4;
                    if off + 4 <= bytes.len() {
                        bytes[off..off + 4].copy_from_slice(&zero_le);
                    }
                }
            }

            // Meta-header flag (HILBERT_SORTED clear) is the writer's
            // responsibility — same rationale as `HilbertRepermute` above:
            // the immutable `substrate` borrow here cannot rewrite the meta
            // header, and on next `open` the writer's phase 9 will either
            // have landed (flag clear) or replay will have reconstructed the
            // correct topology and the next compaction will clear the flag.

            Ok(Applied::Yes)
        }

        WalPayload::NodePropHeadUpdate {
            node_id,
            first_prop_off,
        } => {
            if nodes.is_none() {
                *nodes = Some(substrate.open_zone(Zone::Nodes)?);
            }
            let zf = nodes.as_mut().unwrap();
            let record_offset = (*node_id as usize) * NodeRecord::SIZE;
            let field_offset = record_offset + 14; // first_prop_off byte offset
            if field_offset + 6 > zf.as_slice().len() {
                // Slot not yet materialised — the matching NodeInsert has
                // not been replayed yet, so there is nothing to anchor this
                // head pointer to. The next iteration of the replay loop
                // will typically visit the NodeInsert first (records are
                // replayed in LSN order), at which point a later
                // NodePropHeadUpdate re-applies on the now-materialised
                // slot. Tolerated as a no-op here.
                return Ok(Applied::Yes);
            }
            let u48 = U48::from_u64(*first_prop_off);
            zf.as_slice_mut()[field_offset..field_offset + 6]
                .copy_from_slice(&u48.0);
            Ok(Applied::Yes)
        }

        // Markers — no mutation.
        WalPayload::NoOp | WalPayload::EndOfLog | WalPayload::Checkpoint { .. } => {
            Ok(Applied::Yes)
        }

        // Payload variants that depend on subsystems not yet implemented.
        // Property pages / string heap land in their own tasks; community /
        // hilbert / ricci / tier-0/1/2 payloads land in T10–T12. Until
        // those come online, replay counts them as skipped but does not
        // error out — they're tolerated but inert.
        _ => Ok(Applied::Skipped),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::file::SubstrateFile;
    use crate::record::{f32_to_q1_15, EdgeRecord, NodeRecord, PackedScarUtilAff, U48};
    use crate::wal_io::SyncMode;
    use crate::writer::Writer;
    use tempfile::tempdir;

    fn n(i: u32, label: u64) -> NodeRecord {
        NodeRecord {
            label_bitset: label,
            first_edge_off: U48::default(),
            first_prop_off: U48::default(),
            community_id: 0,
            energy: f32_to_q1_15(0.5),
            scar_util_affinity: PackedScarUtilAff::new(0, 0, 0, false).pack(),
            centrality_cached: 0,
            flags: 0,
        }
    }

    fn e(i: u64, src: u32, dst: u32) -> EdgeRecord {
        EdgeRecord {
            src,
            dst,
            edge_type: 1,
            weight_u16: f32_to_q1_15(0.8),
            next_from: U48::default(),
            next_to: U48::default(),
            ricci_u8: 0,
            flags: 0,
            engram_tag: 0,
            _pad: [0; 4],
        }
    }

    #[test]
    fn replay_empty_wal_is_noop() {
        let dir = tempdir().unwrap();
        let sub = SubstrateFile::create(dir.path().join("kb")).unwrap();
        let stats = replay_from(&sub, 0).unwrap();
        assert_eq!(stats.applied, 0);
        assert_eq!(stats.decode_errors, 0);
    }

    #[test]
    fn replay_reconstructs_nodes_and_edges() {
        let dir = tempdir().unwrap();
        let sub_path = dir.path().join("kb");

        // Round 1: create substrate, write some records, commit, drop without
        // flushing the mmap — simulate a crash that left the WAL durable but
        // the mmap writes may or may not have reached disk.
        {
            let sub = SubstrateFile::create(&sub_path).unwrap();
            let w = Writer::new(sub, SyncMode::EveryCommit).unwrap();
            for i in 0..10u32 {
                w.write_node(i, n(i, 1 << (i % 8))).unwrap();
            }
            for i in 0..5u64 {
                w.write_edge(i, e(i, i as u32, (i + 1) as u32)).unwrap();
            }
            w.commit().unwrap();
            // no msync_zones() — pretend we crashed before flushing.
        }

        // Round 2: wipe the zone files to prove replay reconstructs from WAL.
        std::fs::write(sub_path.join(crate::file::zone::NODES), &[]).unwrap();
        std::fs::write(sub_path.join(crate::file::zone::EDGES), &[]).unwrap();

        // Round 3: reopen + replay.
        let sub2 = SubstrateFile::open(&sub_path).unwrap();
        let stats = replay_from(&sub2, 0).unwrap();
        assert!(stats.applied >= 15); // 10 nodes + 5 edges + 1 commit marker
        assert_eq!(stats.decode_errors, 0);

        // Verify the nodes zone now reflects the inserts.
        let nz = sub2.open_zone(Zone::Nodes).unwrap();
        assert!(nz.len() >= 10 * NodeRecord::SIZE as u64);
        let slice: &[NodeRecord] = bytemuck::cast_slice(
            &nz.as_slice()[..10 * NodeRecord::SIZE],
        );
        for i in 0..10u32 {
            assert_eq!(slice[i as usize].label_bitset, 1u64 << (i % 8));
        }
        let ez = sub2.open_zone(Zone::Edges).unwrap();
        let eslice: &[EdgeRecord] = bytemuck::cast_slice(
            &ez.as_slice()[..5 * EdgeRecord::SIZE],
        );
        for i in 0..5u64 {
            assert_eq!(eslice[i as usize].src, i as u32);
            assert_eq!(eslice[i as usize].dst, (i + 1) as u32);
        }
    }

    #[test]
    fn replay_is_idempotent() {
        let dir = tempdir().unwrap();
        let sub_path = dir.path().join("kb");
        {
            let sub = SubstrateFile::create(&sub_path).unwrap();
            let w = Writer::new(sub, SyncMode::EveryCommit).unwrap();
            for i in 0..5u32 {
                w.write_node(i, n(i, i as u64 + 1)).unwrap();
            }
            w.commit().unwrap();
        }

        let sub = SubstrateFile::open(&sub_path).unwrap();
        let s1 = replay_from(&sub, 0).unwrap();
        let s2 = replay_from(&sub, 0).unwrap();
        // Applied counts match — same records, applied the same way.
        assert_eq!(s1.applied, s2.applied);
        assert_eq!(s1.decode_errors, 0);
        assert_eq!(s2.decode_errors, 0);

        // Bytes still correct.
        let nz = sub.open_zone(Zone::Nodes).unwrap();
        let slice: &[NodeRecord] = bytemuck::cast_slice(
            &nz.as_slice()[..5 * NodeRecord::SIZE],
        );
        for i in 0..5u32 {
            assert_eq!(slice[i as usize].label_bitset, i as u64 + 1);
        }
    }

    #[test]
    fn replay_stops_at_crc_boundary() {
        let dir = tempdir().unwrap();
        let sub_path = dir.path().join("kb");
        let wal_path;
        {
            let sub = SubstrateFile::create(&sub_path).unwrap();
            wal_path = sub.wal_path();
            let w = Writer::new(sub, SyncMode::Never).unwrap();
            for i in 0..10u32 {
                w.write_node(i, n(i, i as u64 + 1)).unwrap();
            }
            w.commit().unwrap();
            w.wal().fsync().unwrap();
        }

        // Tamper the middle of the WAL to simulate a torn write.
        let mut bytes = std::fs::read(&wal_path).unwrap();
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0xFF;
        std::fs::write(&wal_path, bytes).unwrap();

        let sub = SubstrateFile::open(&sub_path).unwrap();
        let stats = replay_from(&sub, 0).unwrap();
        assert_eq!(stats.decode_errors, 1);
        assert!(stats.applied > 0);
        assert!(stats.applied < 11); // stopped before all records
    }

    // =====================================================================
    // T6 cognitive-column replay
    // =====================================================================

    #[test]
    fn replay_reapplies_energy_reinforce_and_decay() {
        let dir = tempdir().unwrap();
        let sub_path = dir.path().join("kb");

        // Round 1: write 3 nodes, seed their energy columns via EnergyReinforce
        // (NodeInsert intentionally carries only label_bitset — cognitive
        // columns always transit via dedicated payloads, that's the T6
        // invariant), then apply decay ×0.5, tombstone one of them.
        {
            let sub = SubstrateFile::create(&sub_path).unwrap();
            let w = Writer::new(sub, SyncMode::EveryCommit).unwrap();
            for i in 1..=3u32 {
                w.write_node(i, n(i, 1)).unwrap();
                // Seed energy to 0.4 via EnergyReinforce.
                w.reinforce_energy(i, crate::record::f32_to_q1_15(0.4))
                    .unwrap();
            }
            // Reinforce node 2 to 0.9.
            w.reinforce_energy(2, crate::record::f32_to_q1_15(0.9))
                .unwrap();
            // Decay all by ×0.5.
            w.decay_all_energy(32768, 4).unwrap();
            // Tombstone node 3 after decay.
            w.tombstone_node(3).unwrap();
            w.commit().unwrap();
            w.wal().fsync().unwrap();
        }

        // Round 2: wipe the nodes zone.
        std::fs::write(sub_path.join(crate::file::zone::NODES), &[]).unwrap();

        // Round 3: replay rebuilds the zone AND reapplies column mutations.
        let sub2 = SubstrateFile::open(&sub_path).unwrap();
        let stats = replay_from(&sub2, 0).unwrap();
        assert_eq!(stats.decode_errors, 0);
        assert!(stats.applied > 0);

        let nz = sub2.open_zone(Zone::Nodes).unwrap();
        let slice: &[NodeRecord] = bytemuck::cast_slice(
            &nz.as_slice()[..4 * NodeRecord::SIZE],
        );

        // Node 1: seed 0.4 → decay ×0.5 → 0.2 (±1 ULP).
        let e1 = crate::record::q1_15_to_f32(slice[1].energy);
        assert!((e1 - 0.2).abs() < 1e-3, "node 1 energy after replay: {e1}");
        // Node 2: seed 0.4 → reinforce 0.9 → decay ×0.5 → 0.45.
        let e2 = crate::record::q1_15_to_f32(slice[2].energy);
        assert!((e2 - 0.45).abs() < 1e-3, "node 2 energy after replay: {e2}");
        // Node 3: tombstoned after decay — energy ≈ 0.2, TOMBSTONED flag set.
        assert!(slice[3].is_tombstoned(), "node 3 must be tombstoned");
    }

    #[test]
    fn replay_reapplies_synapse_reinforce_and_scar_util() {
        let dir = tempdir().unwrap();
        let sub_path = dir.path().join("kb");
        {
            let sub = SubstrateFile::create(&sub_path).unwrap();
            let w = Writer::new(sub, SyncMode::EveryCommit).unwrap();
            w.write_node(1, n(1, 1)).unwrap();
            w.write_edge(1, e(1, 1, 2)).unwrap();
            // Reinforce synapse weight to 0.77.
            let weight = (0.77_f32 * 65535.0).round() as u16;
            w.reinforce_synapse(1, weight).unwrap();
            // Set packed scar/util/affinity.
            let packed =
                crate::record::PackedScarUtilAff::new(9, 13, 25, false).pack();
            w.set_scar_util_affinity(1, packed).unwrap();
            // Centrality batch.
            w.update_centrality_batch(vec![(1, 0xABCD)]).unwrap();
            w.commit().unwrap();
            w.wal().fsync().unwrap();
        }
        // Wipe both zones.
        std::fs::write(sub_path.join(crate::file::zone::NODES), &[]).unwrap();
        std::fs::write(sub_path.join(crate::file::zone::EDGES), &[]).unwrap();

        let sub2 = SubstrateFile::open(&sub_path).unwrap();
        let stats = replay_from(&sub2, 0).unwrap();
        assert_eq!(stats.decode_errors, 0);

        let nz = sub2.open_zone(Zone::Nodes).unwrap();
        let n_slice: &[NodeRecord] =
            bytemuck::cast_slice(&nz.as_slice()[..2 * NodeRecord::SIZE]);
        let unpacked =
            crate::record::PackedScarUtilAff::unpack(n_slice[1].scar_util_affinity);
        assert_eq!(unpacked.scar, 9);
        assert_eq!(unpacked.utility, 13);
        assert_eq!(unpacked.affinity, 25);
        assert_eq!(n_slice[1].centrality_cached, 0xABCD);

        let ez = sub2.open_zone(Zone::Edges).unwrap();
        let e_slice: &[EdgeRecord] =
            bytemuck::cast_slice(&ez.as_slice()[..2 * EdgeRecord::SIZE]);
        let expected_weight = (0.77_f32 * 65535.0).round() as u16;
        assert_eq!(e_slice[1].weight_u16, expected_weight);
    }

    // =====================================================================
    // T11 Step 4 — transactional community compaction replay
    //
    // Acceptance criterion (plan): "Test crash mid-compaction → reopen →
    // replay → état cohérent (soit pré-, soit post-compaction, jamais
    // partiel)." We test three crash windows that together cover the full
    // state space of the write protocol:
    //
    //   1. pre-apply   — WAL has the record, mmap is unchanged.
    //   2. partial     — WAL has the record, mmap reflects some but not all
    //                    of phases 6-8.
    //   3. post-apply  — writer ran end to end, replay must not corrupt the
    //                    post-compaction state.
    //
    // All three must converge to the *same* post-compaction state.
    // =====================================================================

    /// Helper: node record pre-tagged with a community_id.
    fn n_cid(i: u32, label: u64, cid: u32) -> NodeRecord {
        let mut rec = n(i, label);
        rec.community_id = cid;
        rec
    }

    /// Helper: seed a substrate with a 3-node, 3-edge community tagged
    /// `cid`, plus one external node (slot 42) with one cross-community
    /// edge. Used by the three compaction tests.
    ///
    /// After this returns, the WAL contains the full pre-compaction history
    /// (NodeInsert + NodeUpdate + EdgeInsert records) and the mmap is
    /// fsync'd to match. The caller can then append a CompactCommunity
    /// record (or run `compact_community`) to set up the specific crash
    /// window under test.
    fn seed_community(
        sub_path: &std::path::Path,
        old_slots: &[u32; 3],
        cid: u32,
    ) -> Writer {
        let sub = SubstrateFile::create(sub_path).unwrap();
        let w = Writer::new(sub, SyncMode::EveryCommit).unwrap();
        for (i, &slot) in old_slots.iter().enumerate() {
            // NodeInsert only records label_bitset; NodeUpdate threads the
            // community_id through the WAL so replay can reconstruct the
            // pre-compaction cid assignment.
            w.write_node(slot, n_cid(slot, (i as u64) + 1, cid))
                .unwrap();
            w.update_node(slot, n_cid(slot, (i as u64) + 1, cid))
                .unwrap();
        }
        // Intra-community edges.
        w.write_edge(0, e(0, old_slots[0], old_slots[1])).unwrap();
        w.write_edge(1, e(1, old_slots[1], old_slots[2])).unwrap();
        // External node + cross-community edge.
        w.write_node(42, n(42, 0xFF)).unwrap();
        w.write_edge(2, e(2, old_slots[0], 42)).unwrap();
        w.commit().unwrap();
        // Flush the pre-compaction state so it survives a reopen — this is
        // the mmap state "just before the crash" for tests that inject the
        // CompactCommunity record manually.
        w.msync_zones().unwrap();
        w
    }

    /// Shared assertion: after replay, verify the post-compaction state
    /// (old slots zeroed, new slots filled, edges remapped, community col
    /// coherent).
    fn assert_post_compaction_state(
        sub: &SubstrateFile,
        old_slots: &[u32; 3],
        new_slots: &[u32; 3],
        cid: u32,
    ) {
        // Nodes.
        let nz = sub.open_zone(Zone::Nodes).unwrap();
        let need = (new_slots[2] as usize + 1) * NodeRecord::SIZE;
        let nodes: &[NodeRecord] = bytemuck::cast_slice(&nz.as_slice()[..need]);
        for &old in old_slots {
            let rec = &nodes[old as usize];
            assert_eq!(
                rec.community_id, 0,
                "old slot {old} community_id must be 0 after compaction"
            );
            assert_eq!(
                rec.label_bitset, 0,
                "old slot {old} label_bitset must be 0 after compaction"
            );
        }
        for (i, &new) in new_slots.iter().enumerate() {
            let rec = &nodes[new as usize];
            assert_eq!(
                rec.community_id, cid,
                "new slot {new} must carry cid={cid}"
            );
            assert_eq!(
                rec.label_bitset,
                (i as u64) + 1,
                "new slot {new} must carry the label from old slot {}",
                old_slots[i]
            );
        }
        // External node untouched.
        assert_eq!(nodes[42].label_bitset, 0xFF);
        assert_eq!(nodes[42].community_id, 0);

        // Edges.
        let ez = sub.open_zone(Zone::Edges).unwrap();
        let edges: &[EdgeRecord] =
            bytemuck::cast_slice(&ez.as_slice()[..3 * EdgeRecord::SIZE]);
        assert_eq!(edges[0].src, new_slots[0], "edge 0 src remapped");
        assert_eq!(edges[0].dst, new_slots[1], "edge 0 dst remapped");
        assert_eq!(edges[1].src, new_slots[1], "edge 1 src remapped");
        assert_eq!(edges[1].dst, new_slots[2], "edge 1 dst remapped");
        // Cross-community edge: only the in-community endpoint is remapped.
        assert_eq!(edges[2].src, new_slots[0], "edge 2 src remapped");
        assert_eq!(edges[2].dst, 42, "edge 2 dst (external) untouched");

        // Community side-column.
        let cz = sub.open_zone(Zone::Community).unwrap();
        let cbytes = cz.as_slice();
        for &old in old_slots {
            let off = (old as usize) * 4;
            if off + 4 <= cbytes.len() {
                let v =
                    u32::from_le_bytes(cbytes[off..off + 4].try_into().unwrap());
                assert_eq!(v, 0, "community col at old slot {old} must be 0");
            }
        }
        for &new in new_slots {
            let off = (new as usize) * 4;
            assert!(
                off + 4 <= cbytes.len(),
                "community zone must be sized up to cover new slot {new}"
            );
            let v =
                u32::from_le_bytes(cbytes[off..off + 4].try_into().unwrap());
            assert_eq!(v, cid, "community col at new slot {new} must be {cid}");
        }
    }

    /// Crash window **A — pre-apply**: the CompactCommunity record is
    /// durable in the WAL but none of phases 6-8 have touched the mmap.
    /// Replay must apply the full compaction.
    #[test]
    fn replay_compact_community_pre_apply() {
        let dir = tempdir().unwrap();
        let sub_path = dir.path().join("kb");
        let cid = 7u32;
        let old_slots = [1u32, 2, 3];
        let new_slots = [100u32, 101, 102];

        {
            let w = seed_community(&sub_path, &old_slots, cid);

            // Manually append CompactCommunity to simulate a crash between
            // WAL-append (phase 5) and the first mmap mutation (phase 6).
            let relocations: Vec<(u32, u32)> = old_slots
                .iter()
                .zip(new_slots.iter())
                .map(|(&o, &nn)| (o, nn))
                .collect();
            let rec = WalRecord {
                lsn: 0,
                timestamp: 0,
                flags: 0,
                payload: WalPayload::CompactCommunity {
                    community_id: cid,
                    relocations,
                    page_range: (0, 1),
                },
            };
            w.wal().append(rec).unwrap();
            w.commit().unwrap();
            w.wal().fsync().unwrap();
            // NOTE: no msync_zones after the compact record — the mmap
            // reflects the pre-compaction state, exactly as it would after
            // a crash mid-transaction.
        }

        let sub = SubstrateFile::open(&sub_path).unwrap();
        let stats = replay_from(&sub, 0).unwrap();
        assert_eq!(stats.decode_errors, 0);
        // 3 NodeInsert + 3 NodeUpdate + 1 NodeInsert(42) + 3 EdgeInsert
        // + 1 CompactCommunity + commit markers — all applied cleanly.
        assert!(stats.applied > 0);

        assert_post_compaction_state(&sub, &old_slots, &new_slots, cid);
    }

    /// Crash window **B — partial-apply**: the writer got past phase 6
    /// for *some* relocations but not all. The mmap is in an inconsistent
    /// intermediate state (some nodes moved, some still at old slots; some
    /// community-col entries rewritten, some not). Replay must complete
    /// the compaction.
    ///
    /// We simulate the partial state by opening the zones directly and
    /// applying the first two relocations (1→100, 2→101) while leaving
    /// the third (3→102) untouched.
    #[test]
    fn replay_compact_community_partial_apply() {
        let dir = tempdir().unwrap();
        let sub_path = dir.path().join("kb");
        let cid = 7u32;
        let old_slots = [1u32, 2, 3];
        let new_slots = [100u32, 101, 102];

        {
            let w = seed_community(&sub_path, &old_slots, cid);

            let relocations: Vec<(u32, u32)> = old_slots
                .iter()
                .zip(new_slots.iter())
                .map(|(&o, &nn)| (o, nn))
                .collect();
            let rec = WalRecord {
                lsn: 0,
                timestamp: 0,
                flags: 0,
                payload: WalPayload::CompactCommunity {
                    community_id: cid,
                    relocations: relocations.clone(),
                    page_range: (0, 1),
                },
            };
            w.wal().append(rec).unwrap();
            w.commit().unwrap();
            w.wal().fsync().unwrap();

            // --- Simulate partial phase 6 (first 2 of 3 relocations) -----
            let sub_arc = w.substrate();
            let sub = sub_arc.lock();
            let mut nz = sub.open_zone(Zone::Nodes).unwrap();
            let mut cz = sub.open_zone(Zone::Community).unwrap();
            crate::writer::ensure_room(
                &mut nz,
                (new_slots[2] as usize + 1) * NodeRecord::SIZE,
                1 << 20,
            )
            .unwrap();
            crate::writer::ensure_room(
                &mut cz,
                (new_slots[2] as usize + 1) * 4,
                1 << 20,
            )
            .unwrap();
            // Partial relocation: first two pairs.
            for (old, new) in relocations.iter().take(2) {
                let old_off = (*old as usize) * NodeRecord::SIZE;
                let new_off = (*new as usize) * NodeRecord::SIZE;
                let rec_bytes: [u8; NodeRecord::SIZE] = {
                    let slice = &nz.as_slice()[old_off..old_off + NodeRecord::SIZE];
                    let mut out = [0u8; NodeRecord::SIZE];
                    out.copy_from_slice(slice);
                    out
                };
                let zero = NodeRecord::default();
                nz.as_slice_mut()[new_off..new_off + NodeRecord::SIZE]
                    .copy_from_slice(&rec_bytes);
                nz.as_slice_mut()[old_off..old_off + NodeRecord::SIZE]
                    .copy_from_slice(bytemuck::bytes_of(&zero));
                // Community col partial rewrite.
                let cid_le = cid.to_le_bytes();
                let zero_le = 0u32.to_le_bytes();
                cz.as_slice_mut()[(*new as usize) * 4..(*new as usize) * 4 + 4]
                    .copy_from_slice(&cid_le);
                cz.as_slice_mut()[(*old as usize) * 4..(*old as usize) * 4 + 4]
                    .copy_from_slice(&zero_le);
            }
            nz.msync().unwrap();
            nz.fsync().unwrap();
            cz.msync().unwrap();
            cz.fsync().unwrap();
            drop(sub);
        }

        let sub = SubstrateFile::open(&sub_path).unwrap();
        let stats = replay_from(&sub, 0).unwrap();
        assert_eq!(stats.decode_errors, 0);

        assert_post_compaction_state(&sub, &old_slots, &new_slots, cid);
    }

    /// Crash window **C — post-apply**: the writer completed all phases
    /// successfully. The mmap is in the post-compaction state and the WAL
    /// carries the CompactCommunity record. A subsequent replay must be a
    /// true no-op (or at worst rewrite identical bytes).
    ///
    /// We drive the writer's full `compact_community` pipeline, then
    /// reopen and replay twice to prove convergence is idempotent across
    /// arbitrarily many restarts.
    #[test]
    fn replay_compact_community_post_apply_is_idempotent() {
        let dir = tempdir().unwrap();
        let sub_path = dir.path().join("kb");
        let cid = 7u32;
        let old_slots = [1u32, 2, 3];

        // Run the full writer flow. The new slots are chosen by
        // `compact_community` itself (page-aligned at the current tail):
        // node_hw = 43 (we wrote slots 1..3 and 42), NODES_PER_PAGE rounds
        // up, so new_start = the first aligned slot ≥ 43.
        let new_start;
        {
            let w = seed_community(&sub_path, &old_slots, cid);
            // node_hw = max slot written + 1 = 43 ; edge_hw = 3.
            let result = w
                .compact_community(cid, 43, 3, /*order=*/ 8, /*max_degree=*/ 64)
                .unwrap();
            assert_eq!(result.community_id, cid);
            assert_eq!(result.relocations.len(), 3);
            new_start = result.relocations[0].1;
            // New slots are contiguous.
            assert_eq!(result.relocations[1].1, new_start + 1);
            assert_eq!(result.relocations[2].1, new_start + 2);
            w.commit().unwrap();
            w.wal().fsync().unwrap();
        }

        // The writer's internal sort is by (hilbert_key, old_slot). With all
        // energies/centralities identical the hilbert_keys tie, so the
        // secondary key (old_slot ascending) drives the permutation and
        // the mapping is 1→new_start, 2→new_start+1, 3→new_start+2.
        let new_slots = [new_start, new_start + 1, new_start + 2];

        // First replay pass — mmap is already post-compaction.
        let sub = SubstrateFile::open(&sub_path).unwrap();
        let stats1 = replay_from(&sub, 0).unwrap();
        assert_eq!(stats1.decode_errors, 0);
        assert_post_compaction_state(&sub, &old_slots, &new_slots, cid);

        // Second replay pass — convergence must be stable.
        drop(sub);
        let sub = SubstrateFile::open(&sub_path).unwrap();
        let stats2 = replay_from(&sub, 0).unwrap();
        assert_eq!(stats2.decode_errors, 0);
        assert_eq!(
            stats1.applied, stats2.applied,
            "replay must apply the same number of records on every pass"
        );
        assert_post_compaction_state(&sub, &old_slots, &new_slots, cid);
    }
}
