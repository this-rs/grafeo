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
use crate::record::{EdgeRecord, NodeRecord};
use crate::wal::{WalPayload, WalRecord};
use crate::wal_io::WalReader;
use crate::writer::{apply_energy_decay_to_zone, apply_synapse_decay_to_zone, ensure_room};
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

    for item in reader.iter_from(from_offset) {
        match item {
            Ok((rec, off, len)) => {
                stats.last_lsn = stats.last_lsn.max(rec.lsn);
                match apply(substrate, &rec, &mut nodes, &mut edges, &mut engram_members)? {
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
}
