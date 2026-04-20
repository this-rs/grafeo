//! Write-through path: WAL first, then mmap mutation.
//!
//! This is the minimal primitive used by `SubstrateStore` (T4) to guarantee
//! that every on-disk mutation is preceded by a durable WAL record. The
//! invariant is:
//!
//! ```text
//!   durable WAL record committed
//!             ↓
//!     mmap mutation visible
//!             ↓
//!   msync dirty range (lazy, batched)
//! ```
//!
//! A crash between (1) and (2) is recovered by replay (see
//! [`crate::replay`]). A crash between (2) and (3) is harmless — the mmap
//! edits are reapplied at replay from the WAL.
//!
//! ## Scope
//!
//! This module intentionally exposes only a small surface:
//!
//! * [`Writer::write_node`] / [`Writer::update_node`] / [`Writer::tombstone_node`]
//!   / [`Writer::read_node`] — node slot primitives.
//! * [`Writer::write_edge`] / [`Writer::update_edge`] / [`Writer::tombstone_edge`]
//!   / [`Writer::read_edge`] — edge slot primitives.
//! * [`Writer::commit`] — mark a transaction boundary + fsync WAL.
//! * [`Writer::msync_zones`] — force mmap flush (end-of-checkpoint only).
//!
//! The full property / string / cognitive write paths are wired in T4 on top
//! of these primitives.

use crate::error::SubstrateResult;
use crate::file::{SubstrateFile, Zone, ZoneFile};
use crate::record::{EdgeRecord, NodeRecord};
use crate::wal::{WalPayload, WalRecord};
use crate::wal_io::{SyncMode, WalWriter};
use parking_lot::Mutex;
use std::sync::Arc;

/// High-level write-through primitive for a substrate.
///
/// The writer keeps a single [`WalWriter`] and lazily materializes a
/// [`ZoneFile`] handle per zone. It grows zones exponentially (×1.5) to
/// limit the number of remaps during bulk inserts.
pub struct Writer {
    substrate: Arc<Mutex<SubstrateFile>>,
    wal: Arc<WalWriter>,
    zones: Mutex<ZoneCache>,
    /// Growth policy: minimum bytes of headroom kept in each zone.
    min_headroom: u64,
}

#[derive(Default)]
struct ZoneCache {
    nodes: Option<ZoneFile>,
    edges: Option<ZoneFile>,
    engram_members: Option<ZoneFile>,
    engram_bitset: Option<ZoneFile>,
}

impl Writer {
    /// Construct a writer. The WAL is opened at `substrate.wal_path()`.
    pub fn new(substrate: SubstrateFile, sync_mode: SyncMode) -> SubstrateResult<Self> {
        let wal_path = substrate.wal_path();
        let next_lsn = substrate.meta_header().last_wal_offset.saturating_add(1);
        let wal = Arc::new(WalWriter::open(&wal_path, sync_mode, next_lsn)?);
        Ok(Self {
            substrate: Arc::new(Mutex::new(substrate)),
            wal,
            zones: Mutex::new(ZoneCache::default()),
            min_headroom: 1 << 20, // 1 MiB default
        })
    }

    /// Access the shared [`SubstrateFile`] behind this writer.
    pub fn substrate(&self) -> Arc<Mutex<SubstrateFile>> {
        self.substrate.clone()
    }

    /// Access the underlying WAL writer (for advanced callers — group-commit
    /// timers, checkpoint orchestrator).
    pub fn wal(&self) -> Arc<WalWriter> {
        self.wal.clone()
    }

    /// Write a [`NodeRecord`] at `idx` and log a `NodeInsert` record.
    ///
    /// Protocol:
    /// 1. append `WalPayload::NodeInsert` to WAL (not yet committed),
    /// 2. mutate the mmap,
    /// 3. caller must call [`Writer::commit`] to seal the transaction.
    #[tracing::instrument(level = "trace", skip(self, node))]
    pub fn write_node(&self, idx: u32, node: NodeRecord) -> SubstrateResult<()> {
        // (1) WAL first.
        let rec = WalRecord {
            lsn: 0, // assigned by WalWriter::append
            timestamp: unix_micros(),
            flags: 0,
            payload: WalPayload::NodeInsert {
                node_id: idx,
                label_bitset: node.label_bitset,
            },
        };
        self.wal.append(rec)?;

        // (2) mmap mutation.
        self.with_node_zone(|zf| {
            let offset = idx as usize * NodeRecord::SIZE;
            ensure_room(zf, offset + NodeRecord::SIZE, self.min_headroom)?;
            let bytes = bytemuck::bytes_of(&node);
            zf.as_slice_mut()[offset..offset + NodeRecord::SIZE].copy_from_slice(bytes);
            Ok(())
        })
    }

    /// Rewrite an existing [`NodeRecord`] at `idx` and log a `NodeUpdate`.
    ///
    /// Unlike [`Writer::write_node`] this logs the richer `NodeUpdate` payload
    /// that carries every mutable cognitive field — replay restores the slot
    /// exactly as it was at commit time.
    #[tracing::instrument(level = "trace", skip(self, node))]
    pub fn update_node(&self, idx: u32, node: NodeRecord) -> SubstrateResult<()> {
        let rec = WalRecord {
            lsn: 0,
            timestamp: unix_micros(),
            flags: 0,
            payload: WalPayload::NodeUpdate {
                node_id: idx,
                label_bitset: node.label_bitset,
                energy: node.energy,
                scar_util_affinity: node.scar_util_affinity,
                centrality_cached: node.centrality_cached,
                flags: node.flags,
                community_id: node.community_id,
            },
        };
        self.wal.append(rec)?;

        self.with_node_zone(|zf| {
            let offset = idx as usize * NodeRecord::SIZE;
            ensure_room(zf, offset + NodeRecord::SIZE, self.min_headroom)?;
            let bytes = bytemuck::bytes_of(&node);
            zf.as_slice_mut()[offset..offset + NodeRecord::SIZE].copy_from_slice(bytes);
            Ok(())
        })
    }

    /// Tombstone the node at `idx`: logs `NodeDelete` and flips the
    /// [`node_flags::TOMBSTONED`](crate::record::node_flags::TOMBSTONED) bit
    /// in place without touching the rest of the record.
    ///
    /// Silently no-ops if the slot is beyond the current zone length (replay
    /// may request a delete for a node that was never materialised past a
    /// crash — the tombstone is then implicit).
    #[tracing::instrument(level = "trace", skip(self))]
    pub fn tombstone_node(&self, idx: u32) -> SubstrateResult<()> {
        let rec = WalRecord {
            lsn: 0,
            timestamp: unix_micros(),
            flags: 0,
            payload: WalPayload::NodeDelete { node_id: idx },
        };
        self.wal.append(rec)?;

        self.with_node_zone(|zf| {
            let offset = idx as usize * NodeRecord::SIZE;
            if offset + NodeRecord::SIZE > zf.as_slice().len() {
                return Ok(());
            }
            // Read-modify-write with bytemuck's safe Pod API.
            let cur: NodeRecord = {
                let bytes = &zf.as_slice()[offset..offset + NodeRecord::SIZE];
                *bytemuck::from_bytes::<NodeRecord>(bytes)
            };
            let mut updated = cur;
            updated.set_tombstoned();
            let bytes = bytemuck::bytes_of(&updated);
            zf.as_slice_mut()[offset..offset + NodeRecord::SIZE].copy_from_slice(bytes);
            Ok(())
        })
    }

    /// Read the [`NodeRecord`] at slot `idx`. Returns `None` if the slot is
    /// beyond the current zone length (never allocated).
    pub fn read_node(&self, idx: u32) -> SubstrateResult<Option<NodeRecord>> {
        self.with_node_zone(|zf| {
            let offset = idx as usize * NodeRecord::SIZE;
            if offset + NodeRecord::SIZE > zf.as_slice().len() {
                return Ok(None);
            }
            let bytes = &zf.as_slice()[offset..offset + NodeRecord::SIZE];
            Ok(Some(*bytemuck::from_bytes::<NodeRecord>(bytes)))
        })
    }

    /// Write an [`EdgeRecord`] at `idx` and log an `EdgeInsert` record.
    #[tracing::instrument(level = "trace", skip(self, edge))]
    pub fn write_edge(&self, idx: u64, edge: EdgeRecord) -> SubstrateResult<()> {
        let rec = WalRecord {
            lsn: 0,
            timestamp: unix_micros(),
            flags: 0,
            payload: WalPayload::EdgeInsert {
                edge_id: idx,
                src: edge.src,
                dst: edge.dst,
                edge_type: edge.edge_type,
                weight_u16: edge.weight_u16,
            },
        };
        self.wal.append(rec)?;

        self.with_edge_zone(|zf| {
            let offset = idx as usize * EdgeRecord::SIZE;
            ensure_room(zf, offset + EdgeRecord::SIZE, self.min_headroom)?;
            let bytes = bytemuck::bytes_of(&edge);
            zf.as_slice_mut()[offset..offset + EdgeRecord::SIZE].copy_from_slice(bytes);
            Ok(())
        })
    }

    /// Rewrite an existing [`EdgeRecord`] at `idx` and log an `EdgeUpdate`.
    ///
    /// Unlike [`Writer::write_edge`] this logs the richer `EdgeUpdate` payload
    /// that carries the mutable cognitive fields (weight, ricci, flags, engram
    /// tag). The chain pointers (`next_from`/`next_to`) and the immutable
    /// `src`/`dst`/`edge_type` are taken from the full `edge` argument —
    /// callers MUST read the record first, mutate, then pass the complete
    /// updated version.
    ///
    /// Replay restores the slot exactly as it was at commit time.
    #[tracing::instrument(level = "trace", skip(self, edge))]
    pub fn update_edge(&self, idx: u64, edge: EdgeRecord) -> SubstrateResult<()> {
        let rec = WalRecord {
            lsn: 0,
            timestamp: unix_micros(),
            flags: 0,
            payload: WalPayload::EdgeUpdate {
                edge_id: idx,
                weight_u16: edge.weight_u16,
                ricci_u8: edge.ricci_u8,
                flags: edge.flags,
                engram_tag: edge.engram_tag,
            },
        };
        self.wal.append(rec)?;

        self.with_edge_zone(|zf| {
            let offset = idx as usize * EdgeRecord::SIZE;
            ensure_room(zf, offset + EdgeRecord::SIZE, self.min_headroom)?;
            let bytes = bytemuck::bytes_of(&edge);
            zf.as_slice_mut()[offset..offset + EdgeRecord::SIZE].copy_from_slice(bytes);
            Ok(())
        })
    }

    /// Tombstone the edge at `idx`: logs `EdgeDelete` and flips the
    /// [`edge_flags::TOMBSTONED`](crate::record::edge_flags::TOMBSTONED) bit
    /// in place without touching the rest of the record.
    ///
    /// Silently no-ops if the slot is beyond the current zone length.
    #[tracing::instrument(level = "trace", skip(self))]
    pub fn tombstone_edge(&self, idx: u64) -> SubstrateResult<()> {
        let rec = WalRecord {
            lsn: 0,
            timestamp: unix_micros(),
            flags: 0,
            payload: WalPayload::EdgeDelete { edge_id: idx },
        };
        self.wal.append(rec)?;

        self.with_edge_zone(|zf| {
            let offset = idx as usize * EdgeRecord::SIZE;
            if offset + EdgeRecord::SIZE > zf.as_slice().len() {
                return Ok(());
            }
            let cur: EdgeRecord = {
                let bytes = &zf.as_slice()[offset..offset + EdgeRecord::SIZE];
                *bytemuck::from_bytes::<EdgeRecord>(bytes)
            };
            let mut updated = cur;
            updated.flags |= crate::record::edge_flags::TOMBSTONED;
            let bytes = bytemuck::bytes_of(&updated);
            zf.as_slice_mut()[offset..offset + EdgeRecord::SIZE].copy_from_slice(bytes);
            Ok(())
        })
    }

    /// Read the [`EdgeRecord`] at slot `idx`. Returns `None` if the slot is
    /// beyond the current zone length (never allocated).
    pub fn read_edge(&self, idx: u64) -> SubstrateResult<Option<EdgeRecord>> {
        self.with_edge_zone(|zf| {
            let offset = idx as usize * EdgeRecord::SIZE;
            if offset + EdgeRecord::SIZE > zf.as_slice().len() {
                return Ok(None);
            }
            let bytes = &zf.as_slice()[offset..offset + EdgeRecord::SIZE];
            Ok(Some(*bytemuck::from_bytes::<EdgeRecord>(bytes)))
        })
    }

    // ==================================================================
    // Cognitive column primitives (T6)
    // ==================================================================
    //
    // These methods write directly to the typed columns carried inside
    // `NodeRecord` / `EdgeRecord` — energy, scar/util/affinity, synapse
    // weight, centrality — using dedicated WAL payloads. They preserve
    // the "WAL-first, then mmap" invariant but avoid the full NodeUpdate
    // payload overhead for per-cognition mutations.
    //
    // All methods are idempotent under replay (payloads carry absolute
    // post-state, never deltas) except [`decay_all_energy`] whose semantics
    // are "multiply all live slots by factor_q16"; replay re-applies the
    // same multiplicative step in order (see `replay.rs`).

    /// Reinforce a single node's energy to `new_energy_u16` (Q1.15).
    ///
    /// WAL payload: [`WalPayload::EnergyReinforce`].
    ///
    /// Idempotent: replaying the same record writes the same u16 into the
    /// same slot.
    #[tracing::instrument(level = "trace", skip(self))]
    pub fn reinforce_energy(&self, idx: u32, new_energy_u16: u16) -> SubstrateResult<()> {
        let rec = WalRecord {
            lsn: 0,
            timestamp: unix_micros(),
            flags: 0,
            payload: WalPayload::EnergyReinforce {
                node_id: idx,
                new_energy: new_energy_u16,
            },
        };
        self.wal.append(rec)?;

        self.with_node_zone(|zf| {
            let offset = idx as usize * NodeRecord::SIZE;
            ensure_room(zf, offset + NodeRecord::SIZE, self.min_headroom)?;
            let cur: NodeRecord = {
                let bytes = &zf.as_slice()[offset..offset + NodeRecord::SIZE];
                *bytemuck::from_bytes::<NodeRecord>(bytes)
            };
            let mut updated = cur;
            updated.energy = new_energy_u16;
            zf.as_slice_mut()[offset..offset + NodeRecord::SIZE]
                .copy_from_slice(bytemuck::bytes_of(&updated));
            Ok(())
        })
    }

    /// Apply an eager decay to every live node's energy column:
    /// `energy ← energy × factor / 65536` (Q0.16 multiplier).
    ///
    /// This is the "eager-periodic-batch" replacement for the previous
    /// lazy-per-read decay that cached a per-node `last_activated`
    /// timestamp. The Consolidator Thinker (T13) owns the scheduling.
    ///
    /// WAL payload: [`WalPayload::EnergyDecay`] (single compact record).
    ///
    /// `high_water` is the exclusive upper bound of live node slots (i.e.
    /// `SubstrateStore::slot_high_water()`). Tombstoned slots are skipped.
    #[tracing::instrument(level = "trace", skip(self))]
    pub fn decay_all_energy(
        &self,
        factor_q16: u16,
        high_water: u32,
    ) -> SubstrateResult<()> {
        let rec = WalRecord {
            lsn: 0,
            timestamp: unix_micros(),
            flags: 0,
            payload: WalPayload::EnergyDecay { factor_q16 },
        };
        self.wal.append(rec)?;

        self.with_node_zone(|zf| {
            apply_energy_decay_to_zone(zf, factor_q16, high_water);
            Ok(())
        })
    }

    /// Overwrite the packed scar/utility/affinity byte-pair of a node.
    ///
    /// WAL payload: [`WalPayload::ScarUtilAffinitySet`].
    #[tracing::instrument(level = "trace", skip(self))]
    pub fn set_scar_util_affinity(&self, idx: u32, packed: u16) -> SubstrateResult<()> {
        let rec = WalRecord {
            lsn: 0,
            timestamp: unix_micros(),
            flags: 0,
            payload: WalPayload::ScarUtilAffinitySet {
                node_id: idx,
                packed,
            },
        };
        self.wal.append(rec)?;

        self.with_node_zone(|zf| {
            let offset = idx as usize * NodeRecord::SIZE;
            ensure_room(zf, offset + NodeRecord::SIZE, self.min_headroom)?;
            let cur: NodeRecord = {
                let bytes = &zf.as_slice()[offset..offset + NodeRecord::SIZE];
                *bytemuck::from_bytes::<NodeRecord>(bytes)
            };
            let mut updated = cur;
            updated.scar_util_affinity = packed;
            zf.as_slice_mut()[offset..offset + NodeRecord::SIZE]
                .copy_from_slice(bytemuck::bytes_of(&updated));
            Ok(())
        })
    }

    /// Update the 5-bit scar sub-field of a node's packed `scar_util_affinity`
    /// column under the zone lock. Utility and affinity bits are preserved;
    /// the dirty flag is forced to 1. Emits an absolute
    /// [`WalPayload::ScarUtilAffinitySet`] (idempotent replay).
    #[tracing::instrument(level = "trace", skip(self))]
    pub fn update_scar_field(&self, idx: u32, scar_q5: u8) -> SubstrateResult<()> {
        self.update_scar_util_aff_field(idx, |p| {
            crate::record::PackedScarUtilAff::new(scar_q5, p.utility, p.affinity, true)
        })
    }

    /// Update the 5-bit utility sub-field of the packed column.
    #[tracing::instrument(level = "trace", skip(self))]
    pub fn update_utility_field(&self, idx: u32, util_q5: u8) -> SubstrateResult<()> {
        self.update_scar_util_aff_field(idx, |p| {
            crate::record::PackedScarUtilAff::new(p.scar, util_q5, p.affinity, true)
        })
    }

    /// Update the 5-bit affinity sub-field of the packed column.
    #[tracing::instrument(level = "trace", skip(self))]
    pub fn update_affinity_field(&self, idx: u32, aff_q5: u8) -> SubstrateResult<()> {
        self.update_scar_util_aff_field(idx, |p| {
            crate::record::PackedScarUtilAff::new(p.scar, p.utility, aff_q5, true)
        })
    }

    /// Internal read-modify-write primitive for the `scar_util_affinity` u16
    /// column. The whole read + transform + write + WAL append happens under
    /// the zone lock so concurrent sub-field updates from different stores
    /// cannot clobber each other.
    fn update_scar_util_aff_field<F>(&self, idx: u32, transform: F) -> SubstrateResult<()>
    where
        F: FnOnce(crate::record::PackedScarUtilAff) -> crate::record::PackedScarUtilAff,
    {
        self.with_node_zone(|zf| {
            let offset = idx as usize * NodeRecord::SIZE;
            ensure_room(zf, offset + NodeRecord::SIZE, self.min_headroom)?;
            let cur: NodeRecord = {
                let bytes = &zf.as_slice()[offset..offset + NodeRecord::SIZE];
                *bytemuck::from_bytes::<NodeRecord>(bytes)
            };
            let current_packed = crate::record::PackedScarUtilAff::unpack(cur.scar_util_affinity);
            let new_packed = transform(current_packed).pack();

            let rec = WalRecord {
                lsn: 0,
                timestamp: unix_micros(),
                flags: 0,
                payload: WalPayload::ScarUtilAffinitySet {
                    node_id: idx,
                    packed: new_packed,
                },
            };
            self.wal.append(rec)?;

            let mut updated = cur;
            updated.scar_util_affinity = new_packed;
            zf.as_slice_mut()[offset..offset + NodeRecord::SIZE]
                .copy_from_slice(bytemuck::bytes_of(&updated));
            Ok(())
        })
    }

    /// Apply an eager decay to every live edge's synapse weight column:
    /// `weight ← weight × factor / 65536` (Q0.16 multiplier).
    ///
    /// Sibling to [`Self::decay_all_energy`] but operating on [`EdgeRecord`]
    /// rather than [`NodeRecord`]. The Consolidator Thinker (T13) schedules
    /// both in the same batch tick, so a single transaction covers node-
    /// and edge-side cognitive decay.
    ///
    /// WAL payload: [`WalPayload::SynapseDecay`] (single compact record).
    ///
    /// `edge_high_water` is the exclusive upper bound of live edge slots
    /// (`SubstrateStore::edge_slot_high_water()`). Tombstoned slots are
    /// skipped. Slot 0 is reserved and never touched.
    #[tracing::instrument(level = "trace", skip(self))]
    pub fn decay_all_synapse(
        &self,
        factor_q16: u16,
        edge_high_water: u64,
    ) -> SubstrateResult<()> {
        let rec = WalRecord {
            lsn: 0,
            timestamp: unix_micros(),
            flags: 0,
            payload: WalPayload::SynapseDecay { factor_q16 },
        };
        self.wal.append(rec)?;

        self.with_edge_zone(|zf| {
            apply_synapse_decay_to_zone(zf, factor_q16, edge_high_water);
            Ok(())
        })
    }

    /// Reinforce a synapse (edge weight column) to `new_weight_u16` (Q0.16).
    ///
    /// WAL payload: [`WalPayload::SynapseReinforce`].
    #[tracing::instrument(level = "trace", skip(self))]
    pub fn reinforce_synapse(&self, idx: u64, new_weight_u16: u16) -> SubstrateResult<()> {
        let rec = WalRecord {
            lsn: 0,
            timestamp: unix_micros(),
            flags: 0,
            payload: WalPayload::SynapseReinforce {
                edge_id: idx,
                new_weight: new_weight_u16,
            },
        };
        self.wal.append(rec)?;

        self.with_edge_zone(|zf| {
            let offset = idx as usize * EdgeRecord::SIZE;
            ensure_room(zf, offset + EdgeRecord::SIZE, self.min_headroom)?;
            let cur: EdgeRecord = {
                let bytes = &zf.as_slice()[offset..offset + EdgeRecord::SIZE];
                *bytemuck::from_bytes::<EdgeRecord>(bytes)
            };
            let mut updated = cur;
            updated.weight_u16 = new_weight_u16;
            zf.as_slice_mut()[offset..offset + EdgeRecord::SIZE]
                .copy_from_slice(bytemuck::bytes_of(&updated));
            Ok(())
        })
    }

    /// Batch-update centrality for a list of nodes.
    ///
    /// WAL payload: [`WalPayload::CentralityUpdate`].
    #[tracing::instrument(level = "trace", skip(self, updates))]
    pub fn update_centrality_batch(
        &self,
        updates: Vec<(u32, u16)>,
    ) -> SubstrateResult<()> {
        if updates.is_empty() {
            return Ok(());
        }
        let rec = WalRecord {
            lsn: 0,
            timestamp: unix_micros(),
            flags: 0,
            payload: WalPayload::CentralityUpdate {
                updates: updates.clone(),
            },
        };
        self.wal.append(rec)?;

        self.with_node_zone(|zf| {
            for (idx, value) in &updates {
                let offset = (*idx as usize) * NodeRecord::SIZE;
                ensure_room(zf, offset + NodeRecord::SIZE, self.min_headroom)?;
                let cur: NodeRecord = {
                    let bytes = &zf.as_slice()[offset..offset + NodeRecord::SIZE];
                    *bytemuck::from_bytes::<NodeRecord>(bytes)
                };
                let mut updated = cur;
                updated.centrality_cached = *value;
                // Clear the CENTRALITY_STALE flag if set.
                updated.flags &= !crate::record::node_flags::CENTRALITY_STALE;
                zf.as_slice_mut()[offset..offset + NodeRecord::SIZE]
                    .copy_from_slice(bytemuck::bytes_of(&updated));
            }
            Ok(())
        })
    }

    // ==================================================================
    // Engram membership side-table (T7 Step 0)
    // ==================================================================

    /// Write the full membership list for `engram_id` through the WAL-first
    /// path: append a [`WalPayload::EngramMembersSet`] record, then update
    /// the mmap'd side-table.
    ///
    /// Passing an empty `members` slice clears the engram's directory entry
    /// (no payload bytes written). `engram_id = 0` is reserved and rejected.
    #[tracing::instrument(level = "trace", skip(self, members))]
    pub fn set_engram_members(
        &self,
        engram_id: u16,
        members: Vec<u32>,
    ) -> SubstrateResult<()> {
        // (1) WAL first — payload carries full state for idempotent replay.
        let rec = WalRecord {
            lsn: 0,
            timestamp: unix_micros(),
            flags: 0,
            payload: WalPayload::EngramMembersSet {
                engram_id,
                members: members.clone(),
            },
        };
        self.wal.append(rec)?;

        // (2) Side-table mutation.
        self.with_engram_zone(|zf| {
            crate::engram::EngramZone::set_members_raw(zf, engram_id, &members)
        })
    }

    /// Read the current membership list for `engram_id`.
    pub fn engram_members(&self, engram_id: u16) -> SubstrateResult<Option<Vec<u32>>> {
        self.with_engram_zone(|zf| crate::engram::EngramZone::members(zf, engram_id))
    }

    // ==================================================================
    // Engram bitset column (T7 Step 1)
    // ==================================================================

    /// Overwrite the 64-bit engram signature for `node_id` through the
    /// WAL-first path. Absolute semantics — the caller is responsible for
    /// folding in every engram the node still belongs to.
    #[tracing::instrument(level = "trace", skip(self))]
    pub fn set_engram_bitset(&self, node_id: u32, bitset: u64) -> SubstrateResult<()> {
        let rec = WalRecord {
            lsn: 0,
            timestamp: unix_micros(),
            flags: 0,
            payload: WalPayload::EngramBitsetSet { node_id, bitset },
        };
        self.wal.append(rec)?;
        self.with_engram_bitset_zone(|zf| {
            crate::engram_bitset::EngramBitsetColumn::set_raw(zf, node_id, bitset)
        })
    }

    /// Convenience RMW: OR the mask for `engram_id` into `node_id`'s bitset.
    /// Reads the current bitset under the zone lock, ORs the new bit, logs
    /// the absolute post-state, then writes back.
    ///
    /// Does NOT remove stale bits — use [`Self::set_engram_bitset`] with a
    /// freshly-recomputed value for leave/remove semantics.
    #[tracing::instrument(level = "trace", skip(self))]
    pub fn add_engram_bit(&self, node_id: u32, engram_id: u16) -> SubstrateResult<()> {
        crate::engram_bitset::check_nonzero_engram_id(engram_id)?;
        let mask = crate::engram_bitset::engram_bit_mask(engram_id);
        self.with_engram_bitset_zone(|zf| {
            let current = crate::engram_bitset::EngramBitsetColumn::get(zf, node_id);
            let new_bits = current | mask;
            let rec = WalRecord {
                lsn: 0,
                timestamp: unix_micros(),
                flags: 0,
                payload: WalPayload::EngramBitsetSet {
                    node_id,
                    bitset: new_bits,
                },
            };
            self.wal.append(rec)?;
            crate::engram_bitset::EngramBitsetColumn::set_raw(zf, node_id, new_bits)
        })
    }

    /// Read the 64-bit engram signature for `node_id`.
    pub fn engram_bitset(&self, node_id: u32) -> SubstrateResult<u64> {
        self.with_engram_bitset_zone(|zf| {
            Ok(crate::engram_bitset::EngramBitsetColumn::get(zf, node_id))
        })
    }

    /// Seal the transaction: append a commit-flagged no-op and fsync.
    ///
    /// Under [`SyncMode::EveryCommit`] the `FLAG_COMMIT` bit triggers fsync
    /// automatically; under [`SyncMode::Group`] we still append the marker but
    /// the caller's group-commit timer owns the actual fsync.
    pub fn commit(&self) -> SubstrateResult<u64> {
        let rec = WalRecord {
            lsn: 0,
            timestamp: unix_micros(),
            flags: WalRecord::FLAG_COMMIT,
            payload: WalPayload::NoOp,
        };
        let (off, _) = self.wal.append(rec)?;
        Ok(off)
    }

    /// Msync all live zone mmaps. Call at end-of-checkpoint.
    pub fn msync_zones(&self) -> SubstrateResult<()> {
        let zones = self.zones.lock();
        if let Some(z) = zones.nodes.as_ref() {
            z.msync()?;
            z.fsync()?;
        }
        if let Some(z) = zones.edges.as_ref() {
            z.msync()?;
            z.fsync()?;
        }
        if let Some(z) = zones.engram_members.as_ref() {
            z.msync()?;
            z.fsync()?;
        }
        if let Some(z) = zones.engram_bitset.as_ref() {
            z.msync()?;
            z.fsync()?;
        }
        Ok(())
    }

    // ----- internal helpers ------------------------------------------------

    fn with_node_zone<R>(
        &self,
        f: impl FnOnce(&mut ZoneFile) -> SubstrateResult<R>,
    ) -> SubstrateResult<R> {
        let mut zones = self.zones.lock();
        if zones.nodes.is_none() {
            let sub = self.substrate.lock();
            zones.nodes = Some(sub.open_zone(Zone::Nodes)?);
        }
        f(zones.nodes.as_mut().unwrap())
    }

    fn with_edge_zone<R>(
        &self,
        f: impl FnOnce(&mut ZoneFile) -> SubstrateResult<R>,
    ) -> SubstrateResult<R> {
        let mut zones = self.zones.lock();
        if zones.edges.is_none() {
            let sub = self.substrate.lock();
            zones.edges = Some(sub.open_zone(Zone::Edges)?);
        }
        f(zones.edges.as_mut().unwrap())
    }

    fn with_engram_zone<R>(
        &self,
        f: impl FnOnce(&mut ZoneFile) -> SubstrateResult<R>,
    ) -> SubstrateResult<R> {
        let mut zones = self.zones.lock();
        if zones.engram_members.is_none() {
            let sub = self.substrate.lock();
            zones.engram_members = Some(sub.open_zone(Zone::EngramMembers)?);
        }
        f(zones.engram_members.as_mut().unwrap())
    }

    fn with_engram_bitset_zone<R>(
        &self,
        f: impl FnOnce(&mut ZoneFile) -> SubstrateResult<R>,
    ) -> SubstrateResult<R> {
        let mut zones = self.zones.lock();
        if zones.engram_bitset.is_none() {
            let sub = self.substrate.lock();
            zones.engram_bitset = Some(sub.open_zone(Zone::EngramBitset)?);
        }
        f(zones.engram_bitset.as_mut().unwrap())
    }
}

/// Apply an energy-decay step in place to every live node slot in the zone.
///
/// `factor_q16` is a Q0.16 multiplier: `factor_q16 = 65535` is ≈ ×1.0,
/// `factor_q16 = 32768` halves. The operation is:
///
/// ```text
///   energy ← (energy × factor_q16) / 65536
/// ```
///
/// Tombstoned slots are skipped. `high_water` is the exclusive upper bound
/// of allocated slots (slot 0 is reserved). The function is side-effect
/// only — no WAL logging happens here; callers (`Writer::decay_all_energy`,
/// `replay::apply`) own the WAL logic.
///
/// The hot path processes 8 live slots at a time through
/// [`crate::simd::decay_u16x8`] (SSE2 on x86_64, NEON on aarch64, scalar
/// otherwise). The batch gathers 8 strided `energy` u16s into a stack
/// buffer, applies the SIMD mul-high, then scatters the results back —
/// skipping tombstoned slots via a per-slot flag test so the column keeps
/// its exact scalar semantics (bit-for-bit equivalent, property-tested in
/// [`crate::simd`]).
pub fn apply_energy_decay_to_zone(zf: &mut ZoneFile, factor_q16: u16, high_water: u32) {
    let zone_slice = zf.as_slice_mut();
    let stride = NodeRecord::SIZE;
    // high_water is exclusive; slot 0 is reserved.
    let first_slot: u32 = 1;
    let last_slot_exclusive: u32 = high_water;
    let total_capacity = zone_slice.len() / stride;
    let cap = (last_slot_exclusive as usize).min(total_capacity);
    let mut slot = first_slot as usize;

    // --- Batched SIMD body: 8 slots per iteration ------------------------
    while slot + 8 <= cap {
        let base = slot * stride;
        // Gather 8 u16 energies + 8 flags. We go through bytemuck on each
        // slot to stay strict-aliasing-safe under the unsafe_code ban at
        // the crate root — the inner `decay_u16x8` is the only unsafe site.
        let mut lanes = [0u16; 8];
        let mut alive = [false; 8];
        for i in 0..8 {
            let off = base + i * stride;
            let rec: &NodeRecord = bytemuck::from_bytes(&zone_slice[off..off + stride]);
            lanes[i] = rec.energy;
            alive[i] = rec.flags & crate::record::node_flags::TOMBSTONED == 0;
        }
        crate::simd::decay_u16x8(&mut lanes, factor_q16);
        for i in 0..8 {
            if !alive[i] {
                continue;
            }
            let off = base + i * stride;
            let rec: &mut NodeRecord =
                bytemuck::from_bytes_mut(&mut zone_slice[off..off + stride]);
            rec.energy = lanes[i];
        }
        slot += 8;
    }

    // --- Scalar tail for the last (<8) slots -----------------------------
    let factor = factor_q16 as u32;
    while slot < cap {
        let offset = slot * stride;
        let rec: &mut NodeRecord =
            bytemuck::from_bytes_mut(&mut zone_slice[offset..offset + stride]);
        if rec.flags & crate::record::node_flags::TOMBSTONED == 0 {
            rec.energy = (((rec.energy as u32) * factor) >> 16) as u16;
        }
        slot += 1;
    }
}

/// Apply a synapse-decay step in place to every live edge slot in the zone.
///
/// `factor_q16` is a Q0.16 multiplier; the operation is:
///
/// ```text
///   weight_u16 ← (weight_u16 × factor_q16) / 65536
/// ```
///
/// Tombstoned slots (`edge_flags::TOMBSTONED`) are skipped. `high_water` is the
/// exclusive upper bound of allocated edge slots (slot 0 is reserved). The
/// function is side-effect only — no WAL logging; callers own the WAL.
///
/// Design note: the column-view model treats every edge as a potential
/// synapse (the weight column lives on `EdgeRecord`), so this decays *all*
/// live edges rather than filtering on a specific edge_type. Non-synapse
/// edges with weight 0 decay to 0 — a no-op. If a future iteration wants
/// to restrict decay to `edge_flags::SYNAPSE_ACTIVE`, that's an O(1) flag
/// test inside this loop.
///
/// The hot path batches 8 edge slots per iteration through
/// [`crate::simd::decay_u16x8`], gathering/scattering `weight_u16` through a
/// stack buffer and masking tombstoned slots on write-back. Bit-for-bit
/// equivalent to the scalar fallback — property-tested in [`crate::simd`].
pub fn apply_synapse_decay_to_zone(zf: &mut ZoneFile, factor_q16: u16, high_water: u64) {
    let zone_slice = zf.as_slice_mut();
    let stride = EdgeRecord::SIZE;
    let total_capacity = zone_slice.len() / stride;
    let cap = (high_water as usize).min(total_capacity);
    let mut slot: usize = 1;

    // --- Batched SIMD body: 8 edges per iteration ------------------------
    while slot + 8 <= cap {
        let base = slot * stride;
        let mut lanes = [0u16; 8];
        let mut alive = [false; 8];
        for i in 0..8 {
            let off = base + i * stride;
            let rec: &EdgeRecord = bytemuck::from_bytes(&zone_slice[off..off + stride]);
            lanes[i] = rec.weight_u16;
            alive[i] = rec.flags & crate::record::edge_flags::TOMBSTONED == 0;
        }
        crate::simd::decay_u16x8(&mut lanes, factor_q16);
        for i in 0..8 {
            if !alive[i] {
                continue;
            }
            let off = base + i * stride;
            let rec: &mut EdgeRecord =
                bytemuck::from_bytes_mut(&mut zone_slice[off..off + stride]);
            rec.weight_u16 = lanes[i];
        }
        slot += 8;
    }

    // --- Scalar tail for the last (<8) slots -----------------------------
    let factor = factor_q16 as u32;
    while slot < cap {
        let offset = slot * stride;
        let rec: &mut EdgeRecord =
            bytemuck::from_bytes_mut(&mut zone_slice[offset..offset + stride]);
        if rec.flags & crate::record::edge_flags::TOMBSTONED == 0 {
            rec.weight_u16 = (((rec.weight_u16 as u32) * factor) >> 16) as u16;
        }
        slot += 1;
    }
}

/// Ensure `zf` is at least `needed_end` bytes long, with exponential pre-alloc.
///
/// Growth policy: target = max(needed_end + min_headroom, current × 3/2),
/// rounded up to 4 KiB. This yields O(log N) remaps for N-byte bulk inserts.
pub fn ensure_room(zf: &mut ZoneFile, needed_end: usize, min_headroom: u64) -> SubstrateResult<()> {
    let need = needed_end as u64;
    if zf.len() >= need {
        return Ok(());
    }
    let grown = (zf.len() * 3 / 2).max(need + min_headroom);
    let aligned = grown.div_ceil(4096) * 4096;
    zf.grow_to(aligned)?;
    Ok(())
}

fn unix_micros() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::{f32_to_q1_15, PackedScarUtilAff, U48};

    fn sample_node(i: u32) -> NodeRecord {
        NodeRecord {
            label_bitset: 1 << (i % 64),
            first_edge_off: U48::from_u64(i as u64 * 32),
            first_prop_off: U48::from_u64(i as u64 * 16),
            community_id: i / 100,
            energy: f32_to_q1_15(0.5),
            scar_util_affinity: PackedScarUtilAff::new(1, 2, 3, true).pack(),
            centrality_cached: i as u16,
            flags: 0,
        }
    }

    fn sample_edge(i: u64) -> EdgeRecord {
        EdgeRecord {
            src: i as u32,
            dst: (i + 1) as u32,
            edge_type: (i as u16) % 10,
            weight_u16: f32_to_q1_15(0.3),
            next_from: U48::from_u64(i * 32),
            next_to: U48::from_u64(i * 64),
            ricci_u8: (i % 256) as u8,
            flags: 0,
            engram_tag: i as u16,
            _pad: [0; 4],
        }
    }

    #[test]
    fn write_node_then_read_back_from_mmap() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let w = Writer::new(sub, SyncMode::Never).unwrap();
        for i in 0..10u32 {
            w.write_node(i, sample_node(i)).unwrap();
        }
        w.commit().unwrap();
        w.msync_zones().unwrap();

        // Read back via the mmap view.
        let zones = w.zones.lock();
        let zf = zones.nodes.as_ref().unwrap();
        let bytes = zf.as_slice();
        let slice: &[NodeRecord] = bytemuck::cast_slice(&bytes[..10 * NodeRecord::SIZE]);
        for i in 0..10u32 {
            assert_eq!(slice[i as usize], sample_node(i));
        }
    }

    #[test]
    fn write_edge_persists_via_zone_file() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let w = Writer::new(sub, SyncMode::Never).unwrap();
        for i in 0..20u64 {
            w.write_edge(i, sample_edge(i)).unwrap();
        }
        w.commit().unwrap();
        w.msync_zones().unwrap();

        let zones = w.zones.lock();
        let zf = zones.edges.as_ref().unwrap();
        let bytes = zf.as_slice();
        let slice: &[EdgeRecord] = bytemuck::cast_slice(&bytes[..20 * EdgeRecord::SIZE]);
        for i in 0..20u64 {
            assert_eq!(slice[i as usize], sample_edge(i));
        }
    }

    #[test]
    fn wal_contains_every_mutation() {
        // Use a persistent tempdir (kept alive until end-of-test) so the WAL
        // file survives the Writer's drop — open_tempfile() deletes the dir
        // when the SubstrateFile inside the Writer drops.
        let td = tempfile::tempdir().unwrap();
        let sub = SubstrateFile::create(td.path().join("kb")).unwrap();
        let wal_path = sub.wal_path();
        let w = Writer::new(sub, SyncMode::Never).unwrap();
        for i in 0..5u32 {
            w.write_node(i, sample_node(i)).unwrap();
        }
        w.commit().unwrap();
        w.wal().fsync().unwrap();
        drop(w);

        let r = crate::wal_io::WalReader::open(&wal_path).unwrap();
        let items: Vec<_> = r.iter_from(0).collect::<Result<Vec<_>, _>>().unwrap();
        // 5 NodeInsert + 1 NoOp commit
        assert_eq!(items.len(), 6);
        assert!(matches!(
            items[0].0.payload,
            WalPayload::NodeInsert { .. }
        ));
        assert!(items.last().unwrap().0.is_commit());
    }

    #[test]
    fn ensure_room_grows_exponentially() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let mut zf = sub.open_zone(Zone::Nodes).unwrap();
        let start = zf.len();
        ensure_room(&mut zf, 1, 4096).unwrap();
        let after_small = zf.len();
        assert!(after_small >= 4096);
        ensure_room(&mut zf, (after_small as usize) + 1, 4096).unwrap();
        let after_grow = zf.len();
        assert!(after_grow > after_small);
        assert!(
            after_grow >= after_small * 3 / 2,
            "grow should be exponential: {start} → {after_small} → {after_grow}"
        );
    }

    #[test]
    fn sync_mode_every_commit_fsyncs_on_commit() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let w = Writer::new(sub, SyncMode::EveryCommit).unwrap();
        for i in 0..3u32 {
            w.write_node(i, sample_node(i)).unwrap();
        }
        // commit() writes a FLAG_COMMIT record; EveryCommit fsyncs it.
        w.commit().unwrap();
    }

    // =====================================================================
    // T6 cognitive-column primitives
    // =====================================================================

    #[test]
    fn reinforce_energy_mutates_column_and_logs_wal() {
        let td = tempfile::tempdir().unwrap();
        let sub = SubstrateFile::create(td.path().join("kb")).unwrap();
        let wal_path = sub.wal_path();
        let w = Writer::new(sub, SyncMode::Never).unwrap();
        // Seed slot 1 as a real node so the column has a known prior state.
        w.write_node(1, sample_node(1)).unwrap();
        // Reinforce to 0.95.
        let new_q = f32_to_q1_15(0.95);
        w.reinforce_energy(1, new_q).unwrap();
        w.commit().unwrap();
        w.wal().fsync().unwrap();

        // Mmap reflects the new value.
        let got = w.read_node(1).unwrap().unwrap();
        assert_eq!(got.energy, new_q);

        // WAL stream contains the dedicated record.
        drop(w);
        let r = crate::wal_io::WalReader::open(&wal_path).unwrap();
        let items: Vec<_> = r.iter_from(0).collect::<Result<Vec<_>, _>>().unwrap();
        let has_reinforce = items
            .iter()
            .any(|(rec, _, _)| matches!(rec.payload, WalPayload::EnergyReinforce { .. }));
        assert!(has_reinforce, "EnergyReinforce record missing from WAL");
    }

    #[test]
    fn decay_all_energy_multiplies_every_live_slot() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let w = Writer::new(sub, SyncMode::Never).unwrap();
        // Write 5 nodes with known energy = 0.8.
        for i in 1..=5u32 {
            let mut n = sample_node(i);
            n.set_energy_f32(0.8);
            w.write_node(i, n).unwrap();
        }
        // Tombstone slot 3 — must NOT be touched by decay.
        w.tombstone_node(3).unwrap();
        w.commit().unwrap();

        // Apply ×0.5 decay.
        w.decay_all_energy(32768, 6).unwrap();
        w.commit().unwrap();

        for i in 1..=5u32 {
            let rec = w.read_node(i).unwrap().unwrap();
            if i == 3 {
                // Tombstoned: energy preserved, flags flipped.
                assert!(rec.is_tombstoned());
            } else {
                let expected = f32_to_q1_15(0.4); // 0.8 × 0.5
                // Q0.16 factor 32768 / 65536 is 0.5 exactly; accept ±1 ULP.
                let delta = (rec.energy as i32 - expected as i32).abs();
                assert!(delta <= 1, "slot {i}: energy={} expected≈{expected}", rec.energy);
            }
        }
    }

    #[test]
    fn set_scar_util_affinity_roundtrip() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let w = Writer::new(sub, SyncMode::Never).unwrap();
        w.write_node(7, sample_node(7)).unwrap();
        let packed = PackedScarUtilAff::new(11, 22, 30, true).pack();
        w.set_scar_util_affinity(7, packed).unwrap();
        w.commit().unwrap();
        let rec = w.read_node(7).unwrap().unwrap();
        let u = PackedScarUtilAff::unpack(rec.scar_util_affinity);
        assert_eq!(u.scar, 11);
        assert_eq!(u.utility, 22);
        assert_eq!(u.affinity, 30);
        assert!(u.dirty);
    }

    #[test]
    fn decay_all_synapse_multiplies_every_live_edge() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let w = Writer::new(sub, SyncMode::Never).unwrap();
        // Seed 5 edges with known weight = 0.8 (in Q0.16).
        let seed = (0.8_f32 * 65535.0).round() as u16;
        for i in 1..=5u64 {
            let mut e = sample_edge(i);
            e.weight_u16 = seed;
            w.write_edge(i, e).unwrap();
        }
        // Tombstone slot 3 — must NOT be touched.
        w.tombstone_edge(3).unwrap();
        w.commit().unwrap();

        // Apply ×0.5 decay.
        w.decay_all_synapse(32768, 6).unwrap();
        w.commit().unwrap();

        for i in 1..=5u64 {
            let rec = w.read_edge(i).unwrap().unwrap();
            if i == 3 {
                assert!(rec.is_tombstoned());
            } else {
                let expected = (seed as u32 * 32768) >> 16;
                let delta = (rec.weight_u16 as i32 - expected as i32).abs();
                assert!(
                    delta <= 1,
                    "slot {i}: weight={} expected≈{expected}",
                    rec.weight_u16
                );
            }
        }
    }

    #[test]
    fn reinforce_synapse_updates_weight_column() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let w = Writer::new(sub, SyncMode::Never).unwrap();
        w.write_edge(4, sample_edge(4)).unwrap();
        let new_w = (0.77_f32.clamp(0.0, 1.0) * 65535.0).round() as u16;
        w.reinforce_synapse(4, new_w).unwrap();
        w.commit().unwrap();
        let rec = w.read_edge(4).unwrap().unwrap();
        assert_eq!(rec.weight_u16, new_w);
    }

    #[test]
    fn update_scar_field_preserves_other_fields() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let w = Writer::new(sub, SyncMode::Never).unwrap();
        w.write_node(1, sample_node(1)).unwrap();
        // Seed all three sub-fields via whole-column write.
        w.set_scar_util_affinity(1, PackedScarUtilAff::new(3, 7, 15, false).pack())
            .unwrap();
        // Update only the scar field.
        w.update_scar_field(1, 25).unwrap();
        w.commit().unwrap();
        let rec = w.read_node(1).unwrap().unwrap();
        let u = PackedScarUtilAff::unpack(rec.scar_util_affinity);
        assert_eq!(u.scar, 25, "scar updated");
        assert_eq!(u.utility, 7, "utility preserved");
        assert_eq!(u.affinity, 15, "affinity preserved");
        assert!(u.dirty, "dirty forced to 1");
    }

    #[test]
    fn update_utility_and_affinity_fields_are_orthogonal() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let w = Writer::new(sub, SyncMode::Never).unwrap();
        w.write_node(2, sample_node(2)).unwrap();
        // Zero-seed scar to distinguish from the sample_node default.
        w.set_scar_util_affinity(2, PackedScarUtilAff::new(0, 0, 0, false).pack())
            .unwrap();
        w.update_utility_field(2, 19).unwrap();
        w.update_affinity_field(2, 11).unwrap();
        w.commit().unwrap();
        let rec = w.read_node(2).unwrap().unwrap();
        let u = PackedScarUtilAff::unpack(rec.scar_util_affinity);
        assert_eq!(u.scar, 0, "scar not touched by util/affinity updates");
        assert_eq!(u.utility, 19);
        assert_eq!(u.affinity, 11);
        assert!(u.dirty);
    }

    // =====================================================================
    // T7 Step 0 — Engram membership side-table
    // =====================================================================

    #[test]
    fn set_engram_members_walks_and_mutates_zone() {
        let td = tempfile::tempdir().unwrap();
        let sub = SubstrateFile::create(td.path().join("kb")).unwrap();
        let wal_path = sub.wal_path();
        let w = Writer::new(sub, SyncMode::Never).unwrap();

        let members: Vec<u32> = (1..=100).map(|i| i * 3).collect();
        w.set_engram_members(42, members.clone()).unwrap();
        w.commit().unwrap();
        w.wal().fsync().unwrap();

        // (1) Direct readback from the side-table via the Writer.
        let got = w.engram_members(42).unwrap().unwrap();
        assert_eq!(got, members);

        // (2) WAL stream contains the EngramMembersSet payload.
        drop(w);
        let r = crate::wal_io::WalReader::open(&wal_path).unwrap();
        let items: Vec<_> = r.iter_from(0).collect::<Result<Vec<_>, _>>().unwrap();
        let found = items.iter().any(|(rec, _, _)| {
            matches!(
                &rec.payload,
                WalPayload::EngramMembersSet { engram_id: 42, members: m } if m == &members
            )
        });
        assert!(found, "EngramMembersSet record missing from WAL");
    }

    #[test]
    fn set_engram_members_close_reopen_roundtrip() {
        // T7 Step 0 acceptance: create engram with 100 members → close →
        // reopen → members() returns the same list, rebuilt from the WAL.
        let td = tempfile::tempdir().unwrap();
        let sub_path = td.path().join("kb");
        let members: Vec<u32> = (1..=100).map(|i| i * 7 + 1).collect();
        {
            let sub = SubstrateFile::create(&sub_path).unwrap();
            let w = Writer::new(sub, SyncMode::EveryCommit).unwrap();
            w.set_engram_members(999, members.clone()).unwrap();
            w.commit().unwrap();
            w.msync_zones().unwrap();
            w.wal().fsync().unwrap();
        }
        // Re-open the substrate — mmap picks up the on-disk zone directly.
        let sub2 = SubstrateFile::open(&sub_path).unwrap();
        let w2 = Writer::new(sub2, SyncMode::Never).unwrap();
        let got = w2.engram_members(999).unwrap().unwrap();
        assert_eq!(got, members);
    }

    #[test]
    fn engram_members_rebuild_from_wal_after_zone_wipe() {
        // Simulate a crash that destroys the side-table zone file but keeps
        // the WAL durable. Replay must reconstruct the membership exactly.
        let td = tempfile::tempdir().unwrap();
        let sub_path = td.path().join("kb");
        let members: Vec<u32> = (1..=100).map(|i| i + 10_000).collect();
        {
            let sub = SubstrateFile::create(&sub_path).unwrap();
            let w = Writer::new(sub, SyncMode::EveryCommit).unwrap();
            w.set_engram_members(1234, members.clone()).unwrap();
            // Also set a second engram to check replay handles multiple records.
            w.set_engram_members(1, vec![1u32, 2, 3, 4, 5]).unwrap();
            w.commit().unwrap();
            w.wal().fsync().unwrap();
            // Intentionally do NOT msync_zones — pretend the mmap writes
            // never hit disk.
        }
        // Wipe the side-table zone.
        std::fs::write(
            sub_path.join(crate::file::zone::ENGRAM_MEMBERS),
            Vec::<u8>::new(),
        )
        .unwrap();

        // Replay reconstructs the side-table from the WAL.
        let sub2 = SubstrateFile::open(&sub_path).unwrap();
        let stats = crate::replay::replay_from(&sub2, 0).unwrap();
        assert_eq!(stats.decode_errors, 0);

        let w2 = Writer::new(sub2, SyncMode::Never).unwrap();
        let got = w2.engram_members(1234).unwrap().unwrap();
        assert_eq!(got, members);
        let got1 = w2.engram_members(1).unwrap().unwrap();
        assert_eq!(got1, vec![1u32, 2, 3, 4, 5]);
    }

    // =====================================================================
    // T7 Step 1 — Engram bitset column
    // =====================================================================

    #[test]
    fn set_engram_bitset_roundtrip_via_writer() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let w = Writer::new(sub, SyncMode::Never).unwrap();
        w.set_engram_bitset(7, 0xFACE_B00C_0000_0001).unwrap();
        w.commit().unwrap();
        let got = w.engram_bitset(7).unwrap();
        assert_eq!(got, 0xFACE_B00C_0000_0001);
        // Unset slots return 0.
        assert_eq!(w.engram_bitset(8).unwrap(), 0);
    }

    #[test]
    fn add_engram_bit_ors_in_mask() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let w = Writer::new(sub, SyncMode::Never).unwrap();
        w.add_engram_bit(1, 5).unwrap();
        w.add_engram_bit(1, 10).unwrap();
        w.add_engram_bit(1, 70).unwrap(); // 70 % 64 = 6
        w.commit().unwrap();
        let got = w.engram_bitset(1).unwrap();
        assert_eq!(got, (1u64 << 5) | (1u64 << 10) | (1u64 << 6));
    }

    #[test]
    fn add_engram_bit_rejects_zero_engram_id() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let w = Writer::new(sub, SyncMode::Never).unwrap();
        let err = w.add_engram_bit(1, 0).unwrap_err();
        assert!(matches!(err, crate::SubstrateError::WalBadFrame(_)));
    }

    #[test]
    fn engram_bitset_wal_rebuild_after_zone_wipe() {
        // Crash-safety: wipe the bitset column file, replay rebuilds it.
        let td = tempfile::tempdir().unwrap();
        let sub_path = td.path().join("kb");
        {
            let sub = SubstrateFile::create(&sub_path).unwrap();
            let w = Writer::new(sub, SyncMode::EveryCommit).unwrap();
            // Mix of set_engram_bitset (absolute) and add_engram_bit (RMW).
            w.set_engram_bitset(1, 0xAAAA_BBBB).unwrap();
            w.add_engram_bit(2, 3).unwrap();
            w.add_engram_bit(2, 3).unwrap(); // idempotent
            w.add_engram_bit(2, 67).unwrap(); // 67 % 64 = 3 (collides)
            w.add_engram_bit(2, 7).unwrap();
            w.set_engram_bitset(100, 0xFFFF_FFFF_FFFF_FFFF).unwrap();
            w.commit().unwrap();
            w.wal().fsync().unwrap();
        }
        // Wipe the bitset zone.
        std::fs::write(
            sub_path.join(crate::file::zone::ENGRAM_BITSET),
            Vec::<u8>::new(),
        )
        .unwrap();

        let sub2 = SubstrateFile::open(&sub_path).unwrap();
        let stats = crate::replay::replay_from(&sub2, 0).unwrap();
        assert_eq!(stats.decode_errors, 0);

        let w2 = Writer::new(sub2, SyncMode::Never).unwrap();
        assert_eq!(w2.engram_bitset(1).unwrap(), 0xAAAA_BBBB);
        assert_eq!(w2.engram_bitset(2).unwrap(), (1u64 << 3) | (1u64 << 7));
        assert_eq!(w2.engram_bitset(100).unwrap(), 0xFFFF_FFFF_FFFF_FFFF);
        assert_eq!(w2.engram_bitset(3).unwrap(), 0); // untouched
    }

    #[test]
    fn centrality_batch_updates_and_clears_stale_flag() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let w = Writer::new(sub, SyncMode::Never).unwrap();
        for i in 1..=3u32 {
            let mut n = sample_node(i);
            n.flags |= crate::record::node_flags::CENTRALITY_STALE;
            w.write_node(i, n).unwrap();
        }
        w.update_centrality_batch(vec![(1, 100), (2, 200), (3, 300)])
            .unwrap();
        w.commit().unwrap();
        for (slot, expected) in [(1u32, 100u16), (2, 200), (3, 300)] {
            let rec = w.read_node(slot).unwrap().unwrap();
            assert_eq!(rec.centrality_cached, expected);
            assert_eq!(
                rec.flags & crate::record::node_flags::CENTRALITY_STALE,
                0,
                "CENTRALITY_STALE flag must be cleared"
            );
        }
    }
}
