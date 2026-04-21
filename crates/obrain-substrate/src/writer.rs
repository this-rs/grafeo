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

use crate::error::{SubstrateError, SubstrateResult};
use crate::file::{SubstrateFile, Zone, ZoneFile};
use crate::hilbert::{compute_hilbert_permutation_page_aligned, hilbert_key_from_features};
use crate::meta::meta_flags;
use crate::record::{EdgeRecord, NodeRecord, NODES_PER_PAGE};
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

    // ==================================================================
    // Coactivation column primitives (T7 Step 5)
    // ==================================================================
    //
    // COACT edges share `EdgeRecord.weight_u16` with synapse edges but
    // are distinguished by `edge_type == coact_type_id` (interned via
    // [`crate::record::COACT_EDGE_TYPE_NAME`]). The decay schedule is
    // independent — COACT decays slowly, synapse decays fast — so we
    // ship a dedicated WAL payload [`WalPayload::CoactDecay`] that
    // carries the type id alongside the multiplicative factor.
    //
    // Reinforcement uses [`WalPayload::SynapseReinforce`] (which carries
    // the absolute post-state and is therefore agnostic to edge type) so
    // we don't grow the payload set unnecessarily — the coact_type_id
    // is implicit in the edge slot itself.

    /// Saturating-add a Q0.16 delta to the `weight_u16` column of edge
    /// slot `idx`, then log the **absolute** post-state via
    /// [`WalPayload::SynapseReinforce`] (so replay is idempotent).
    ///
    /// Designed for COACT edges (each coactivation adds a small δ); also
    /// usable for any other Q0.16 weight column whose semantics are
    /// "accumulate evidence with saturation at 1.0".
    ///
    /// Returns the new weight in Q0.16 so the caller can monitor
    /// saturation without a follow-up read.
    ///
    /// ⚠ Caller must verify that `idx` resolves to a live COACT edge —
    /// this primitive does **not** check `edge_type` (it's a column
    /// operation, not a typed-edge operation). The store-level
    /// [`crate::store::SubstrateStore::coact_reinforce`] enforces typing.
    #[tracing::instrument(level = "trace", skip(self))]
    pub fn coact_reinforce_at(&self, idx: u64, delta_q16: u16) -> SubstrateResult<u16> {
        self.with_edge_zone(|zf| {
            let offset = idx as usize * EdgeRecord::SIZE;
            ensure_room(zf, offset + EdgeRecord::SIZE, self.min_headroom)?;
            let cur: EdgeRecord = {
                let bytes = &zf.as_slice()[offset..offset + EdgeRecord::SIZE];
                *bytemuck::from_bytes::<EdgeRecord>(bytes)
            };
            let new_weight = cur.weight_u16.saturating_add(delta_q16);

            // (1) WAL first — log absolute post-state for idempotent replay.
            let rec = WalRecord {
                lsn: 0,
                timestamp: unix_micros(),
                flags: 0,
                payload: WalPayload::SynapseReinforce {
                    edge_id: idx,
                    new_weight,
                },
            };
            self.wal.append(rec)?;

            // (2) mmap mutation under the same zone lock — no reader can
            //     observe an interleaved column / WAL state.
            let mut updated = cur;
            updated.weight_u16 = new_weight;
            zf.as_slice_mut()[offset..offset + EdgeRecord::SIZE]
                .copy_from_slice(bytemuck::bytes_of(&updated));
            Ok(new_weight)
        })
    }

    /// Apply a Q0.16 multiplicative decay to every live edge whose
    /// `edge_type` equals `coact_type_id`. Synapse edges and other types
    /// are untouched.
    ///
    /// WAL payload: [`WalPayload::CoactDecay { factor_q16, coact_type_id }`].
    ///
    /// `edge_high_water` is the exclusive upper bound of allocated edge
    /// slots (`SubstrateStore::edge_slot_high_water()`). Slot 0 is
    /// reserved.
    #[tracing::instrument(level = "trace", skip(self))]
    pub fn decay_all_coact(
        &self,
        factor_q16: u16,
        edge_high_water: u64,
        coact_type_id: u16,
    ) -> SubstrateResult<()> {
        let rec = WalRecord {
            lsn: 0,
            timestamp: unix_micros(),
            flags: 0,
            payload: WalPayload::CoactDecay {
                factor_q16,
                coact_type_id,
            },
        };
        self.wal.append(rec)?;

        self.with_edge_zone(|zf| {
            apply_coact_decay_to_zone(zf, factor_q16, edge_high_water, coact_type_id);
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

    /// Batch-update community_id for a list of nodes.
    ///
    /// Used by LDleiden (T10) to persist node→community assignments after an
    /// incremental update. Just like centrality, the WAL payload carries the
    /// full `(node_id, community_id)` tuples so replay is idempotent.
    ///
    /// WAL payload: [`WalPayload::CommunityAssign`].
    #[tracing::instrument(level = "trace", skip(self, updates))]
    pub fn update_community_batch(
        &self,
        updates: Vec<(u32, u32)>,
    ) -> SubstrateResult<()> {
        if updates.is_empty() {
            return Ok(());
        }
        let rec = WalRecord {
            lsn: 0,
            timestamp: unix_micros(),
            flags: 0,
            payload: WalPayload::CommunityAssign {
                updates: updates.clone(),
            },
        };
        self.wal.append(rec)?;

        self.with_node_zone(|zf| {
            for (idx, community_id) in &updates {
                let offset = (*idx as usize) * NodeRecord::SIZE;
                ensure_room(zf, offset + NodeRecord::SIZE, self.min_headroom)?;
                let cur: NodeRecord = {
                    let bytes = &zf.as_slice()[offset..offset + NodeRecord::SIZE];
                    *bytemuck::from_bytes::<NodeRecord>(bytes)
                };
                let mut updated = cur;
                updated.community_id = *community_id;
                zf.as_slice_mut()[offset..offset + NodeRecord::SIZE]
                    .copy_from_slice(bytemuck::bytes_of(&updated));
            }
            Ok(())
        })
    }

    // ==================================================================
    // Hilbert page ordering (T11 Step 2)
    // ==================================================================

    /// Bulk-sort node slots by `(community_id, hilbert_key(centrality, degree))`.
    ///
    /// This is the one-shot reorder invoked by the caller (typically at
    /// first-open after community detection has run at least once, or
    /// after a significant mutation burst). Semantics:
    ///
    /// 1. Scan `Zone::Nodes` to collect `(community_id, centrality_cached,
    ///    tombstoned)` per slot.
    /// 2. Scan `Zone::Edges` once to compute out-degree per source slot.
    /// 3. Compute the `old_to_new` permutation via
    ///    [`compute_hilbert_permutation`]. Nodes with the same community
    ///    become contiguous; within a community they're ordered by a 2D
    ///    Hilbert curve on `(centrality, degree)`; tombstoned slots move
    ///    to the tail.
    /// 4. Emit a [`WalPayload::HilbertRepermute`] WAL record carrying the
    ///    permutation — WAL-first discipline.
    /// 5. Apply the permutation **out-of-place** to `Zone::Nodes` (build
    ///    a fresh buffer, swap at the end).
    /// 6. Rewrite every `EdgeRecord.src` / `dst` using the permutation.
    /// 7. Rebuild `Zone::Hilbert` and `Zone::Community` side-columns so
    ///    each new slot carries its hilbert_key / community_id — these
    ///    columns are the fast-path for spreading activation prefetch
    ///    (`madvise(WILLNEED)` range computation).
    /// 8. Set [`meta_flags::HILBERT_SORTED`] and persist the meta header.
    /// 9. `msync` + `fsync` touched zones + commit WAL.
    ///
    /// ### Parameters
    ///
    /// * `node_high_water` — exclusive upper bound on allocated node slots
    ///   (the caller usually tracks this via its graph manager).
    /// * `edge_high_water` — exclusive upper bound on allocated edge slots.
    /// * `order` — Hilbert resolution in bits per axis. Typical values:
    ///   6 (64×64 grid) for KBs up to 10⁵ nodes, 8 (256×256) for 10⁶+.
    ///   Persisted into [`crate::meta::MetaHeader::hilbert_order`].
    /// * `max_degree` — saturation cap for the degree axis.
    ///
    /// ### Crash safety (Step 2 scope)
    ///
    /// This Step 2 implementation is **not** transactionally crash-safe:
    /// a crash between (5) and (6) leaves the substrate in an intermediate
    /// state where the node zone is permuted but edges still reference the
    /// old slot IDs. Callers MUST restrict `bulk_sort_by_hilbert` to a
    /// quiet window (no concurrent writes, no expected mid-operation
    /// shutdown). The full transactional variant with out-of-place journal
    /// + atomic swap lands in T11 Step 4 (`CompactCommunity` pass).
    ///
    /// ### Idempotency
    ///
    /// Early-returns if `HILBERT_SORTED` is already set in the meta flags,
    /// so accidental re-invocation is safe. Callers that want to re-sort
    /// after major mutations must first clear the flag (a privileged op
    /// exposed via `SubstrateFile::write_meta_header`) — this explicit gate
    /// is deliberate to prevent pointless re-permutations.
    #[tracing::instrument(level = "info", skip(self))]
    pub fn bulk_sort_by_hilbert(
        &self,
        node_high_water: u32,
        edge_high_water: u64,
        order: u32,
        max_degree: u32,
    ) -> SubstrateResult<()> {
        // Early-out idempotency check.
        {
            let sub = self.substrate.lock();
            let h = sub.meta_header();
            if h.flags & meta_flags::HILBERT_SORTED != 0 {
                tracing::debug!("substrate already Hilbert-sorted, skipping");
                return Ok(());
            }
        }

        if node_high_water == 0 {
            return Ok(());
        }

        // ---- Phase 1 — read node columns + compute degrees -------------
        //
        // A NodeRecord is treated as `tombstoned` for sort purposes when:
        //   - its TOMBSTONED flag is set (explicit delete), OR
        //   - it is a zero-sentinel (all fields zero) left behind by
        //     slow-path alignment in
        //     `SubstrateStore::allocate_node_id_in_community` (T11 Step 3).
        //     Skipped-slot zero-fill from `ZoneFile::grow_to` has no
        //     community membership; classifying it as tombstoned pushes it
        //     to the tail post-sort instead of silently inflating the
        //     zero-community run.
        //
        // Slot 0 is always the null-sentinel and is likewise treated as
        // tombstoned so the sort never needs to spare it from the tail.
        let mut communities = vec![0u32; node_high_water as usize];
        let mut centrality = vec![0u16; node_high_water as usize];
        let mut tombstoned = vec![false; node_high_water as usize];
        self.with_node_zone(|zf| {
            for idx in 0..node_high_water {
                let offset = (idx as usize) * NodeRecord::SIZE;
                if offset + NodeRecord::SIZE > zf.as_slice().len() {
                    break;
                }
                let rec: &NodeRecord =
                    bytemuck::from_bytes(&zf.as_slice()[offset..offset + NodeRecord::SIZE]);
                let is_zero_sentinel = rec.label_bitset == 0
                    && rec.community_id == 0
                    && rec.centrality_cached == 0
                    && rec.energy == 0
                    && rec.scar_util_affinity == 0
                    && rec.flags == 0
                    && rec.first_edge_off == crate::record::U48::ZERO
                    && rec.first_prop_off == crate::record::U48::ZERO;
                communities[idx as usize] = rec.community_id;
                centrality[idx as usize] = rec.centrality_cached;
                tombstoned[idx as usize] =
                    rec.is_tombstoned() || is_zero_sentinel || idx == 0;
            }
            Ok(())
        })?;

        let mut degrees = vec![0u32; node_high_water as usize];
        self.with_edge_zone(|zf| {
            for idx in 0..edge_high_water {
                let offset = (idx as usize) * EdgeRecord::SIZE;
                if offset + EdgeRecord::SIZE > zf.as_slice().len() {
                    break;
                }
                let rec: &EdgeRecord =
                    bytemuck::from_bytes(&zf.as_slice()[offset..offset + EdgeRecord::SIZE]);
                if rec.is_tombstoned() {
                    continue;
                }
                if (rec.src as usize) < degrees.len() {
                    degrees[rec.src as usize] = degrees[rec.src as usize].saturating_add(1);
                }
            }
            Ok(())
        })?;

        // ---- Phase 2 — compute permutation ----------------------------
        //
        // Page-aligned: each community's run starts on a multiple of
        // `NODES_PER_PAGE` so a single community never straddles a page
        // boundary post-sort. Tomb/zero-sentinel slots (plentiful after
        // slow-path community alignment on insert) absorb the padding
        // holes between community runs.
        let old_to_new = compute_hilbert_permutation_page_aligned(
            &communities,
            &centrality,
            &degrees,
            &tombstoned,
            order,
            max_degree,
            NODES_PER_PAGE,
        );

        // ---- Phase 3 — WAL-first: log the permutation -----------------
        let rec = WalRecord {
            lsn: 0,
            timestamp: unix_micros(),
            flags: 0,
            payload: WalPayload::HilbertRepermute {
                permutation: old_to_new.clone(),
            },
        };
        self.wal.append(rec)?;

        // ---- Phase 4 — apply permutation out-of-place to Zone::Nodes --
        self.with_node_zone(|zf| {
            let n = node_high_water as usize;
            let total = n * NodeRecord::SIZE;
            ensure_room(zf, total, self.min_headroom)?;
            let mut buf = vec![NodeRecord::default(); n];
            let src = &zf.as_slice()[..total];
            for old in 0..n {
                let new = old_to_new[old] as usize;
                let rec: &NodeRecord =
                    bytemuck::from_bytes(&src[old * NodeRecord::SIZE..(old + 1) * NodeRecord::SIZE]);
                buf[new] = *rec;
            }
            zf.as_slice_mut()[..total].copy_from_slice(bytemuck::cast_slice(&buf));
            Ok(())
        })?;

        // ---- Phase 5 — rewrite edge.src/dst in place ------------------
        self.with_edge_zone(|zf| {
            for idx in 0..edge_high_water {
                let offset = (idx as usize) * EdgeRecord::SIZE;
                if offset + EdgeRecord::SIZE > zf.as_slice().len() {
                    break;
                }
                let rec: EdgeRecord =
                    *bytemuck::from_bytes(&zf.as_slice()[offset..offset + EdgeRecord::SIZE]);
                // Silent tombstones keep their (now stale) src/dst — their payload
                // is ignored by every reader anyway.
                if rec.is_tombstoned() {
                    continue;
                }
                let mut updated = rec;
                if (rec.src as usize) < old_to_new.len() {
                    updated.src = old_to_new[rec.src as usize];
                }
                if (rec.dst as usize) < old_to_new.len() {
                    updated.dst = old_to_new[rec.dst as usize];
                }
                zf.as_slice_mut()[offset..offset + EdgeRecord::SIZE]
                    .copy_from_slice(bytemuck::bytes_of(&updated));
            }
            Ok(())
        })?;

        // ---- Phase 6 — rebuild side-columns (Hilbert, Community) ------
        // Hilbert side-column: u32 per new slot (the recomputed hilbert_key).
        // Community side-column: u32 per new slot (the community_id).
        {
            let sub = self.substrate.lock();
            let mut hilbert_zone = sub.open_zone(Zone::Hilbert)?;
            let need = (node_high_water as u64) * 4;
            ensure_room(&mut hilbert_zone, need as usize, self.min_headroom)?;
            let mut hbuf = vec![0u32; node_high_water as usize];
            let mut cbuf = vec![0u32; node_high_water as usize];
            for old in 0..node_high_water as usize {
                let new = old_to_new[old] as usize;
                hbuf[new] = hilbert_key_from_features(
                    centrality[old],
                    degrees[old],
                    order,
                    max_degree,
                );
                cbuf[new] = communities[old];
            }
            hilbert_zone.as_slice_mut()[..(need as usize)]
                .copy_from_slice(bytemuck::cast_slice(&hbuf));
            hilbert_zone.msync()?;
            hilbert_zone.fsync()?;

            let mut community_zone = sub.open_zone(Zone::Community)?;
            ensure_room(&mut community_zone, need as usize, self.min_headroom)?;
            community_zone.as_slice_mut()[..(need as usize)]
                .copy_from_slice(bytemuck::cast_slice(&cbuf));
            community_zone.msync()?;
            community_zone.fsync()?;
        }

        // ---- Phase 7 — set HILBERT_SORTED in meta header ---------------
        {
            let mut sub = self.substrate.lock();
            let mut h = sub.meta_header();
            h.flags |= meta_flags::HILBERT_SORTED;
            h.hilbert_order = order;
            h.node_count = node_high_water as u64;
            sub.write_meta_header(&h)?;
        }

        // ---- Phase 8 — msync touched node/edge zones + WAL commit -----
        self.msync_zones()?;
        let _ = self.commit()?;
        Ok(())
    }

    /// T11 Step 5 — transactional per-community compaction.
    ///
    /// Relocates every live node of `community_id` from its current scattered
    /// slots to a fresh page-aligned contiguous range appended at the node
    /// zone's tail. Returns the list of `(old_slot, new_slot)` mappings and
    /// the new high-water mark the caller must publish back to the store's
    /// slot allocator.
    ///
    /// # Durability protocol
    ///
    /// ```text
    ///   1. scan Nodes zone for live nodes with community_id == target
    ///   2. sort by hilbert_key (intra-community locality)
    ///   3. pick new_range = [aligned(current_hw), aligned(current_hw)+count)
    ///   4. WAL-append  CompactCommunity { cid, relocations, page_range }
    ///   5. fsync WAL   (durable commit boundary)
    ///   6. copy each old → new, zero-fill old
    ///   7. remap every edge whose (src or dst) is in old_slot set
    ///   8. rebuild Hilbert + Community side-columns for the affected slots
    ///   9. msync touched zones + WAL commit marker
    /// ```
    ///
    /// A crash between steps 4 and 9 is recovered idempotently from the WAL:
    /// the replay engine re-runs step 6–8 using the logged mapping.
    ///
    /// # Idempotency
    ///
    /// * Node relocation — only moves `old → new` when the node at `old`
    ///   still carries `community_id == target`. If replay finds the node
    ///   already at `new`, the record at `old` will be a zero-sentinel (we
    ///   zeroed it on first apply), the condition fails, and the move is
    ///   skipped.
    /// * Edge remap — only rewrites `src` / `dst` values that still live in
    ///   the `old_slot` domain. After the first pass, those values have
    ///   moved to the new-slot range (outside the keyset), so the lookup
    ///   misses and the second application is a no-op.
    ///
    /// Returns a struct holding the transaction summary so the caller can
    /// update its in-memory state (store slot allocator, community
    /// placements map) to match the new layout.
    #[tracing::instrument(level = "info", skip(self))]
    pub fn compact_community(
        &self,
        community_id: u32,
        current_node_hw: u32,
        current_edge_hw: u64,
        order: u32,
        max_degree: u32,
    ) -> SubstrateResult<CompactionResult> {
        if current_node_hw == 0 {
            return Ok(CompactionResult::empty());
        }

        // ---- Phase 1 — gather live members of the target community ----
        let mut members: Vec<(u32, u32)> = Vec::new(); // (old_slot, hilbert_key)
        self.with_node_zone(|zf| {
            for slot in 0..current_node_hw {
                let offset = (slot as usize) * NodeRecord::SIZE;
                if offset + NodeRecord::SIZE > zf.as_slice().len() {
                    break;
                }
                let rec: &NodeRecord = bytemuck::from_bytes(
                    &zf.as_slice()[offset..offset + NodeRecord::SIZE],
                );
                if rec.is_tombstoned() {
                    continue;
                }
                if rec.community_id != community_id {
                    continue;
                }
                // Zero-sentinel guard (slow-path padding left by Step 3).
                if rec.label_bitset == 0
                    && rec.community_id == 0
                    && rec.centrality_cached == 0
                    && rec.energy == 0
                    && rec.scar_util_affinity == 0
                    && rec.flags == 0
                    && rec.first_edge_off == crate::record::U48::ZERO
                    && rec.first_prop_off == crate::record::U48::ZERO
                {
                    continue;
                }
                let hkey = hilbert_key_from_features(
                    rec.centrality_cached,
                    0, // degree unknown here — Step 7 will plumb it; Hilbert locality already suffices
                    order,
                    max_degree,
                );
                members.push((slot, hkey));
            }
            Ok(())
        })?;

        if members.is_empty() {
            return Ok(CompactionResult::empty());
        }

        // ---- Phase 2 — sort by (hilbert_key, old_slot) for determinism --
        members.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));

        // ---- Phase 3 — pick new page-aligned range at the tail --------
        let ppg = NODES_PER_PAGE;
        let new_start = current_node_hw.div_ceil(ppg) * ppg;
        let count = members.len() as u32;
        let new_end = new_start
            .checked_add(count)
            .ok_or_else(|| SubstrateError::Internal("compact_community: slot-id overflow".into()))?;
        let first_page = new_start / ppg;
        let last_page = new_end.div_ceil(ppg);

        let relocations: Vec<(u32, u32)> = members
            .iter()
            .enumerate()
            .map(|(i, (old, _))| (*old, new_start + i as u32))
            .collect();

        // ---- Phase 4/5 — WAL-first (fsync happens at commit()) --------
        let rec = WalRecord {
            lsn: 0,
            timestamp: unix_micros(),
            flags: 0,
            payload: WalPayload::CompactCommunity {
                community_id,
                relocations: relocations.clone(),
                page_range: (first_page, last_page),
            },
        };
        self.wal.append(rec)?;

        // Old → New maps for fast lookup during edge remap.
        let old_slots: std::collections::HashSet<u32> =
            relocations.iter().map(|(old, _)| *old).collect();
        let old_to_new_map: std::collections::HashMap<u32, u32> =
            relocations.iter().copied().collect();

        // ---- Phase 6 — copy old → new + zero old ----------------------
        self.with_node_zone(|zf| {
            let needed = (new_end as usize) * NodeRecord::SIZE;
            ensure_room(zf, needed, self.min_headroom)?;
            // Grab a snapshot of the old records first (borrowing rules).
            let mut copies: Vec<(u32, NodeRecord)> = Vec::with_capacity(relocations.len());
            {
                let src = zf.as_slice();
                for (old, new) in &relocations {
                    let off = (*old as usize) * NodeRecord::SIZE;
                    let rec: &NodeRecord =
                        bytemuck::from_bytes(&src[off..off + NodeRecord::SIZE]);
                    copies.push((*new, *rec));
                }
            }
            // Write each to its new slot.
            let zone_mut = zf.as_slice_mut();
            for (new, rec) in &copies {
                let off = (*new as usize) * NodeRecord::SIZE;
                zone_mut[off..off + NodeRecord::SIZE]
                    .copy_from_slice(bytemuck::bytes_of(rec));
            }
            // Zero each old slot (turn into zero-sentinel — safe because we
            // set community_id back to 0, so the warden & bulk_sort will skip
            // it).
            let zero = NodeRecord::default();
            for (old, _) in &relocations {
                let off = (*old as usize) * NodeRecord::SIZE;
                zone_mut[off..off + NodeRecord::SIZE]
                    .copy_from_slice(bytemuck::bytes_of(&zero));
            }
            Ok(())
        })?;

        // ---- Phase 7 — remap edges whose endpoint is in old_slots -----
        self.with_edge_zone(|zf| {
            for idx in 0..current_edge_hw {
                let offset = (idx as usize) * EdgeRecord::SIZE;
                if offset + EdgeRecord::SIZE > zf.as_slice().len() {
                    break;
                }
                let rec: EdgeRecord =
                    *bytemuck::from_bytes(&zf.as_slice()[offset..offset + EdgeRecord::SIZE]);
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
                    if let Some(&ns) = old_to_new_map.get(&rec.src) {
                        updated.src = ns;
                    }
                }
                if dst_in {
                    if let Some(&nd) = old_to_new_map.get(&rec.dst) {
                        updated.dst = nd;
                    }
                }
                zf.as_slice_mut()[offset..offset + EdgeRecord::SIZE]
                    .copy_from_slice(bytemuck::bytes_of(&updated));
            }
            Ok(())
        })?;

        // ---- Phase 8 — rebuild affected slots in side-columns ---------
        {
            let sub = self.substrate.lock();
            let mut hilbert_zone = sub.open_zone(Zone::Hilbert)?;
            let needed = (new_end as usize) * 4;
            ensure_room(&mut hilbert_zone, needed, self.min_headroom)?;
            let hbytes = hilbert_zone.as_slice_mut();
            for (i, (old, new)) in relocations.iter().enumerate() {
                let _ = old; // unused here, only hkey matters — recompute from sorted list
                let hkey = members[i].1;
                let off = (*new as usize) * 4;
                hbytes[off..off + 4].copy_from_slice(&hkey.to_le_bytes());
            }
            // Zero side-column entries at old slots.
            for (old, _) in &relocations {
                let off = (*old as usize) * 4;
                hbytes[off..off + 4].copy_from_slice(&0u32.to_le_bytes());
            }
            hilbert_zone.msync()?;
            hilbert_zone.fsync()?;

            let mut community_zone = sub.open_zone(Zone::Community)?;
            ensure_room(&mut community_zone, needed, self.min_headroom)?;
            let cbytes = community_zone.as_slice_mut();
            for (_, new) in &relocations {
                let off = (*new as usize) * 4;
                cbytes[off..off + 4].copy_from_slice(&community_id.to_le_bytes());
            }
            for (old, _) in &relocations {
                let off = (*old as usize) * 4;
                cbytes[off..off + 4].copy_from_slice(&0u32.to_le_bytes());
            }
            community_zone.msync()?;
            community_zone.fsync()?;
        }

        // ---- Phase 9 — meta header: clear HILBERT_SORTED (compaction
        //                invalidates global sort), bump node_count ------
        {
            let mut sub = self.substrate.lock();
            let mut h = sub.meta_header();
            h.flags &= !meta_flags::HILBERT_SORTED;
            h.node_count = new_end as u64;
            sub.write_meta_header(&h)?;
        }

        // ---- Phase 10 — flush mmap + commit WAL ----------------------
        self.msync_zones()?;
        let _ = self.commit()?;

        Ok(CompactionResult {
            community_id,
            relocations,
            new_node_hw: new_end,
            page_range: (first_page, last_page),
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

    // ==================================================================
    // Hopfield recall (T7 Step 3)
    // ==================================================================

    /// Compute the top-`k` nodes by bitset overlap with `query_nid`'s
    /// signature. The scan runs under the engram-bitset zone lock through
    /// [`crate::engram_recall::top_k_by_overlap`], which dispatches to a
    /// SIMD kernel on aarch64 and to hardware POPCNT on x86_64.
    ///
    /// Returns `Vec<(node_id, overlap_bits)>` sorted by descending overlap
    /// (ties broken by ascending node id). Nodes with zero overlap are
    /// skipped.
    ///
    /// An all-zero query bitset (the node belongs to no engram) returns
    /// the empty vector without touching the column.
    ///
    /// **First tier only.** Callers must resolve Bloom collisions via
    /// [`Self::engram_members`] before using the candidates as actual
    /// co-members.
    #[tracing::instrument(level = "trace", skip(self))]
    pub fn hopfield_recall(
        &self,
        query_nid: u32,
        k: usize,
    ) -> SubstrateResult<Vec<(u32, u32)>> {
        let query = self.engram_bitset(query_nid)?;
        if query == 0 || k == 0 {
            return Ok(Vec::new());
        }
        self.with_engram_bitset_zone(|zf| {
            let bytes = zf.as_slice();
            // Zone capacity is always a multiple of BITSET_ENTRY_SIZE (8 B);
            // the bitset column is pre-allocated, so trailing slots read
            // back as 0 and contribute overlap=0 (naturally skipped).
            let entry = crate::engram_bitset::BITSET_ENTRY_SIZE;
            let n = bytes.len() / entry;
            // SAFETY: the zone is byte-aligned on mmap; u64 requires 8-byte
            // alignment which every mmapped page provides. We still use
            // bytemuck's checked cast to make the alignment promise explicit.
            let column: &[u64] = bytemuck::cast_slice(&bytes[..n * entry]);
            Ok(crate::engram_recall::top_k_by_overlap(column, query, k))
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

    /// Issue `madvise(WILLNEED)` on a byte range of a zone (T11 Step 5
    /// prefetch hook). Best-effort — safe to call on cold zones (returns
    /// `Ok(())` silently).
    ///
    /// Zones covered:
    /// * [`Zone::Nodes`] / [`Zone::Edges`] / [`Zone::EngramMembers`] /
    ///   [`Zone::EngramBitset`] — go through the writer's zone cache.
    /// * [`Zone::Hilbert`] / [`Zone::Community`] / [`Zone::Props`] /
    ///   [`Zone::Strings`] / [`Zone::Tier0`] / [`Zone::Tier1`] /
    ///   [`Zone::Tier2`] — opened on-demand through the SubstrateFile.
    ///   A fresh `ZoneFile` is created per call and dropped; `madvise`
    ///   hits the kernel on the underlying file descriptor, so no
    ///   durability ordering is perturbed.
    pub fn advise_zone_willneed(
        &self,
        zone: Zone,
        offset: usize,
        len: usize,
    ) -> SubstrateResult<()> {
        match zone {
            Zone::Nodes => self.with_node_zone(|zf| zf.advise_willneed(offset, len)),
            Zone::Edges => self.with_edge_zone(|zf| zf.advise_willneed(offset, len)),
            Zone::EngramMembers => self.with_engram_zone(|zf| zf.advise_willneed(offset, len)),
            Zone::EngramBitset => {
                self.with_engram_bitset_zone(|zf| zf.advise_willneed(offset, len))
            }
            // One-shot zones: open a fresh handle, advise, drop. These
            // zones are not writer-managed in the ZoneCache because
            // mutations happen through dedicated paths (bulk_sort,
            // compact_community, etc.), not the normal hot-write loop.
            Zone::Hilbert
            | Zone::Community
            | Zone::Props
            | Zone::Strings
            | Zone::Tier0
            | Zone::Tier1
            | Zone::Tier2 => {
                let sub = self.substrate.lock();
                let zf = sub.open_zone(zone)?;
                drop(sub);
                zf.advise_willneed(offset, len)
            }
        }
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

/// Apply a Q0.16 multiplicative decay to every live edge whose `edge_type`
/// equals `coact_type_id`.
///
/// Sibling to [`apply_synapse_decay_to_zone`] but type-filtered: synapse
/// edges and any other non-COACT edge type are skipped (their `weight_u16`
/// column is left untouched). The math itself is identical to the
/// synapse decay — `weight ← (weight × factor_q16) >> 16` — but applied
/// only to slots whose `edge_type` matches.
///
/// Tombstoned slots are skipped. Slot 0 is reserved.
///
/// The hot path batches 8 edge slots per iteration, gathering eight
/// `weight_u16`s and eight `(alive ∧ is_coact)` masks, running the
/// shared SIMD `decay_u16x8` kernel, then scattering only the masked
/// lanes back. Bit-for-bit equivalent to the scalar fallback for the
/// COACT-typed slots; non-COACT slots are bit-for-bit untouched.
pub fn apply_coact_decay_to_zone(
    zf: &mut ZoneFile,
    factor_q16: u16,
    high_water: u64,
    coact_type_id: u16,
) {
    let zone_slice = zf.as_slice_mut();
    let stride = EdgeRecord::SIZE;
    let total_capacity = zone_slice.len() / stride;
    let cap = (high_water as usize).min(total_capacity);
    let mut slot: usize = 1;

    // --- Batched SIMD body: 8 edges per iteration ------------------------
    while slot + 8 <= cap {
        let base = slot * stride;
        let mut lanes = [0u16; 8];
        let mut decay_lane = [false; 8];
        for i in 0..8 {
            let off = base + i * stride;
            let rec: &EdgeRecord = bytemuck::from_bytes(&zone_slice[off..off + stride]);
            lanes[i] = rec.weight_u16;
            decay_lane[i] = rec.flags & crate::record::edge_flags::TOMBSTONED == 0
                && rec.edge_type == coact_type_id;
        }
        crate::simd::decay_u16x8(&mut lanes, factor_q16);
        for i in 0..8 {
            if !decay_lane[i] {
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
        if rec.flags & crate::record::edge_flags::TOMBSTONED == 0
            && rec.edge_type == coact_type_id
        {
            rec.weight_u16 = (((rec.weight_u16 as u32) * factor) >> 16) as u16;
        }
        slot += 1;
    }
}

/// Summary of a single [`Writer::compact_community`] transaction.
///
/// * `community_id` — the community that was relocated.
/// * `relocations` — `(old_slot, new_slot)` pairs describing every live node
///   that was moved. Length = number of live members of the community at the
///   time of the call.
/// * `new_node_hw` — post-compaction slot high-water mark the caller must
///   publish to the store's slot allocator so subsequent inserts land past
///   the relocated range.
/// * `page_range` — the `(first_page, last_page)` half-open interval the
///   relocated nodes now occupy, for verification / debugging.
#[derive(Debug, Clone)]
pub struct CompactionResult {
    pub community_id: u32,
    pub relocations: Vec<(u32, u32)>,
    pub new_node_hw: u32,
    pub page_range: (u32, u32),
}

impl CompactionResult {
    /// Empty / no-op outcome (used when the target community has no live
    /// members, so there's nothing to relocate).
    pub fn empty() -> Self {
        Self {
            community_id: 0,
            relocations: Vec::new(),
            new_node_hw: 0,
            page_range: (0, 0),
        }
    }

    /// `true` when no nodes were moved.
    pub fn is_empty(&self) -> bool {
        self.relocations.is_empty()
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
    fn hopfield_recall_returns_top_k_by_bitset_overlap() {
        // T7 Step 3 — Writer end-to-end. Three co-engram clusters:
        //   - nodes 1,2,3,4 all belong to engrams 1 and 2 (bits 1 and 2 set)
        //   - nodes 5,6 belong to engrams 1 and 3 (bits 1 and 3 set)
        //   - nodes 7,8 belong to engram 4 only (bit 4 set)
        let sub = SubstrateFile::open_tempfile().unwrap();
        let w = Writer::new(sub, SyncMode::Never).unwrap();
        for nid in [1u32, 2, 3, 4] {
            w.add_engram_bit(nid, 1).unwrap();
            w.add_engram_bit(nid, 2).unwrap();
        }
        for nid in [5u32, 6] {
            w.add_engram_bit(nid, 1).unwrap();
            w.add_engram_bit(nid, 3).unwrap();
        }
        for nid in [7u32, 8] {
            w.add_engram_bit(nid, 4).unwrap();
        }
        w.commit().unwrap();

        // Query from node 1 (bitset = bit1 | bit2). Expect 2, 3, 4 first
        // (overlap=2 — bits 1 and 2), then 5, 6 (overlap=1 — bit 1 only),
        // then nothing from 7, 8 (overlap=0).
        let got = w.hopfield_recall(1, 10).unwrap();
        let ids: Vec<u32> = got.iter().map(|(nid, _)| *nid).collect();
        // 1 itself is returned (full self-overlap = 2).
        assert_eq!(ids[0], 1);
        // Next three must be 2, 3, 4 (overlap=2, ordered by ascending idx).
        assert_eq!(&ids[1..4], &[2, 3, 4]);
        // Then 5, 6 (overlap=1).
        assert_eq!(&ids[4..6], &[5, 6]);
        // Nothing from 7, 8.
        for (nid, _) in got.iter() {
            assert!(*nid != 7 && *nid != 8, "non-overlapping nodes leaked");
        }
        // Overlap values are monotonically non-increasing.
        for w in got.windows(2) {
            assert!(w[0].1 >= w[1].1);
        }
    }

    #[test]
    fn hopfield_recall_empty_query_yields_empty() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let w = Writer::new(sub, SyncMode::Never).unwrap();
        // Node 99 has never been given any engram bit.
        let got = w.hopfield_recall(99, 10).unwrap();
        assert!(got.is_empty());
    }

    #[test]
    fn hopfield_recall_k_zero_yields_empty() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let w = Writer::new(sub, SyncMode::Never).unwrap();
        w.add_engram_bit(1, 1).unwrap();
        w.add_engram_bit(2, 1).unwrap();
        let got = w.hopfield_recall(1, 0).unwrap();
        assert!(got.is_empty());
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

    // T10 Step 3 — CommunityAssign batch writer + in-memory read-back.
    //
    // Writes three nodes with community_id = 0, calls
    // `update_community_batch` to assign ids (11, 22, 33), commits, then
    // verifies the mmap zone reflects the new assignments.
    #[test]
    fn community_batch_assigns_and_persists() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let w = Writer::new(sub, SyncMode::EveryCommit).unwrap();
        for i in 1..=3u32 {
            let mut n = sample_node(i);
            n.community_id = 0;
            w.write_node(i, n).unwrap();
        }
        w.update_community_batch(vec![(1, 11), (2, 22), (3, 33)])
            .unwrap();
        w.commit().unwrap();
        for (slot, expected) in [(1u32, 11u32), (2, 22), (3, 33)] {
            assert_eq!(w.read_node(slot).unwrap().unwrap().community_id, expected);
        }
    }

    // T10 Step 3 — WAL replay round-trip for CommunityAssign. After the
    // zone file is wiped, a reopen + replay must restore every assignment.
    // Exercises the replay handler added in `replay.rs`.
    #[test]
    fn community_batch_wal_replay_round_trip() {
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let sub_path = dir.path().join("kb");
        // Round 1: create substrate, write nodes, assign communities, commit.
        {
            let sub = SubstrateFile::create(&sub_path).unwrap();
            let w = Writer::new(sub, SyncMode::EveryCommit).unwrap();
            for i in 1..=3u32 {
                let mut n = sample_node(i);
                n.community_id = 0;
                w.write_node(i, n).unwrap();
            }
            w.update_community_batch(vec![(1, 11), (2, 22), (3, 33)])
                .unwrap();
            w.commit().unwrap();
        }

        // Round 2: wipe the nodes zone (keep the WAL).
        std::fs::write(sub_path.join(crate::file::zone::NODES), &[]).unwrap();

        // Round 3: reopen + replay. community_id must be restored.
        let sub2 = SubstrateFile::open(&sub_path).unwrap();
        let stats = crate::replay::replay_from(&sub2, 0).unwrap();
        assert_eq!(stats.decode_errors, 0);

        let nz = sub2.open_zone(crate::file::Zone::Nodes).unwrap();
        let slice: &[NodeRecord] = bytemuck::cast_slice(
            &nz.as_slice()[..4 * NodeRecord::SIZE],
        );
        assert_eq!(slice[1].community_id, 11);
        assert_eq!(slice[2].community_id, 22);
        assert_eq!(slice[3].community_id, 33);
    }

    // T10 Step 3 — empty batch is a no-op (no WAL record, no mmap writes).
    #[test]
    fn community_batch_empty_is_noop() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let w = Writer::new(sub, SyncMode::Never).unwrap();
        w.write_node(1, sample_node(1)).unwrap();
        let before = w.read_node(1).unwrap().unwrap().community_id;
        w.update_community_batch(Vec::new()).unwrap();
        w.commit().unwrap();
        assert_eq!(w.read_node(1).unwrap().unwrap().community_id, before);
    }

    // =====================================================================
    // T7 Step 5 — COACT typed-edge column primitives
    // =====================================================================

    /// Helper: write an edge slot with a specific edge_type and weight,
    /// using `Writer::write_edge` so the WAL records it. Other fields are
    /// the bog-standard `sample_edge` defaults.
    fn write_typed_edge(
        w: &Writer,
        slot: u64,
        edge_type: u16,
        weight_u16: u16,
    ) {
        let mut e = sample_edge(slot);
        e.edge_type = edge_type;
        e.weight_u16 = weight_u16;
        w.write_edge(slot, e).unwrap();
    }

    #[test]
    fn coact_reinforce_at_saturates_and_persists() {
        // T7 Step 5 — `coact_reinforce_at` is a saturating-add on the
        // weight column, returning the new weight, with the absolute
        // post-state logged via SynapseReinforce so replay is idempotent.
        let sub = SubstrateFile::open_tempfile().unwrap();
        let w = Writer::new(sub, SyncMode::Never).unwrap();

        // Seed slot 1 as a COACT edge (type id 7) with weight 0.
        write_typed_edge(&w, 1, 7, 0);

        // Reinforce by 0x4000 four times: 0 → 0x4000 → 0x8000 → 0xC000 → 0xFFFF (sat).
        let new = w.coact_reinforce_at(1, 0x4000).unwrap();
        assert_eq!(new, 0x4000);
        let new = w.coact_reinforce_at(1, 0x4000).unwrap();
        assert_eq!(new, 0x8000);
        let new = w.coact_reinforce_at(1, 0x4000).unwrap();
        assert_eq!(new, 0xC000);
        // Saturation: 0xC000 + 0x4000 = 0x10000 → clamped to 0xFFFF.
        let new = w.coact_reinforce_at(1, 0x4000).unwrap();
        assert_eq!(new, 0xFFFF);
        // One more — already at ceiling.
        let new = w.coact_reinforce_at(1, 0x1234).unwrap();
        assert_eq!(new, 0xFFFF);
        w.commit().unwrap();

        // Read back: the column reflects the saturated value.
        let rec = w.read_edge(1).unwrap().unwrap();
        assert_eq!(rec.weight_u16, 0xFFFF);
        // Type and other fields are preserved.
        assert_eq!(rec.edge_type, 7);
    }

    #[test]
    fn decay_all_coact_only_touches_coact_typed_edges() {
        // T7 Step 5 — `decay_all_coact(factor, hw, type_id)` filters by
        // edge_type so synapse and other types are bit-for-bit
        // untouched. Mix three edge types in the same zone and assert.
        let sub = SubstrateFile::open_tempfile().unwrap();
        let w = Writer::new(sub, SyncMode::Never).unwrap();
        const COACT_ID: u16 = 7;
        const SYNAPSE_ID: u16 = 3;
        const OTHER_ID: u16 = 11;

        // Slots 1..=12 alternate COACT / SYNAPSE / OTHER. All start at
        // weight 0x8000 (≈ 0.5).
        for slot in 1..=12u64 {
            let ty = match slot % 3 {
                1 => COACT_ID,
                2 => SYNAPSE_ID,
                _ => OTHER_ID,
            };
            write_typed_edge(&w, slot, ty, 0x8000);
        }
        w.commit().unwrap();

        // Decay factor 0.5 (Q0.16 = 32768): COACT slots should halve, others stay.
        w.decay_all_coact(32768, 13, COACT_ID).unwrap();
        w.commit().unwrap();

        for slot in 1..=12u64 {
            let rec = w.read_edge(slot).unwrap().unwrap();
            let expected = match slot % 3 {
                1 => 0x4000, // COACT decayed (0x8000 * 0.5)
                _ => 0x8000, // others untouched
            };
            assert_eq!(
                rec.weight_u16, expected,
                "slot {slot} (type {}) — expected {expected:#x}, got {:#x}",
                rec.edge_type, rec.weight_u16,
            );
        }
    }

    #[test]
    fn decay_all_coact_skips_tombstoned() {
        // Tombstoned COACT edges must NOT be touched.
        let sub = SubstrateFile::open_tempfile().unwrap();
        let w = Writer::new(sub, SyncMode::Never).unwrap();
        const COACT_ID: u16 = 7;

        for slot in 1..=4u64 {
            write_typed_edge(&w, slot, COACT_ID, 0x8000);
        }
        // Tombstone slot 2.
        w.tombstone_edge(2).unwrap();
        w.commit().unwrap();

        w.decay_all_coact(32768, 5, COACT_ID).unwrap();
        w.commit().unwrap();

        // 1, 3, 4 decayed to 0x4000; 2 still 0x8000 (tombstoned, skipped).
        assert_eq!(w.read_edge(1).unwrap().unwrap().weight_u16, 0x4000);
        let rec2 = w.read_edge(2).unwrap().unwrap();
        assert_eq!(rec2.weight_u16, 0x8000);
        assert!(rec2.flags & crate::record::edge_flags::TOMBSTONED != 0);
        assert_eq!(w.read_edge(3).unwrap().unwrap().weight_u16, 0x4000);
        assert_eq!(w.read_edge(4).unwrap().unwrap().weight_u16, 0x4000);
    }

    #[test]
    fn coact_decay_wal_replay_roundtrip() {
        // T7 Step 5 crash-safety — the new `CoactDecay` payload must
        // round-trip through WAL replay. Setup: write 4 COACT edges +
        // 2 synapse edges, log a single coact decay, then replay from
        // scratch and assert the column matches the post-decay state.
        use crate::file::SubstrateFile;
        let td = tempfile::tempdir().unwrap();
        let sub = SubstrateFile::create(td.path().join("kb")).unwrap();
        let wal_path = sub.wal_path();
        let w = Writer::new(sub, SyncMode::Never).unwrap();

        const COACT_ID: u16 = 7;
        const SYNAPSE_ID: u16 = 3;
        for slot in 1..=4u64 {
            write_typed_edge(&w, slot, COACT_ID, 0x8000);
        }
        for slot in 5..=6u64 {
            write_typed_edge(&w, slot, SYNAPSE_ID, 0x8000);
        }
        w.decay_all_coact(32768, 7, COACT_ID).unwrap();
        w.commit().unwrap();
        w.wal().fsync().unwrap();
        // Drop the writer so msync fully flushes — but we explicitly want
        // replay to reapply the decay, so do NOT msync_zones here.
        drop(w);

        // Replay into a freshly-opened substrate (same dir).
        let sub2 = SubstrateFile::open(td.path().join("kb")).unwrap();
        let _ = wal_path; // silence
        let stats = crate::replay::replay_from(&sub2, 0).unwrap();
        // We logged 6 EdgeInsert + 1 CoactDecay + 1 NoOp commit = 8 records.
        assert!(
            stats.applied >= 7,
            "replay should apply our 6 inserts + decay (got {})",
            stats.applied
        );

        // Re-open through the writer to validate the column.
        let w2 = Writer::new(sub2, SyncMode::Never).unwrap();
        for slot in 1..=4u64 {
            assert_eq!(
                w2.read_edge(slot).unwrap().unwrap().weight_u16,
                0x4000,
                "COACT slot {slot} should be decayed"
            );
        }
        for slot in 5..=6u64 {
            assert_eq!(
                w2.read_edge(slot).unwrap().unwrap().weight_u16,
                0x8000,
                "SYNAPSE slot {slot} must NOT be decayed by CoactDecay"
            );
        }
    }
}
