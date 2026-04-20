//! `SubstrateStore` — the concrete `GraphStore` / `GraphStoreMut` implementation
//! backed by the topology-as-storage substrate.
//!
//! ## Role in the stack
//!
//! ```text
//! obrain-core (LpgStore → GraphStore trait)
//!      ▲
//!      │ impl GraphStore for SubstrateStore
//!      │
//! obrain-substrate (this file)   ← you are here
//!      │
//!      ▼ wraps
//! SubstrateFile { mmap zones, WAL, meta }
//! ```
//!
//! ## T4 step progression
//!
//! | Step | What this file gains                                             |
//! |------|------------------------------------------------------------------|
//! | 0    | Skeleton — every method `unimplemented!()`, compiles.            |
//! | 1 ✨ | Real node ops: create/get/delete, properties, iter, by-label.   |
//! | 2 ✨ | Edges: create_edge, edges_from, traversal chains, edge props.   |
//! | 3 ✨ | Persistent registries (label bitset, edge-type, property keys). |
//! | 4    | LpgStore parity suite (port of tests, 100% green).              |
//! | 5    | Randomised 10k-op parity (100 seeds, 0 divergence).             |
//! | 6    | Criterion CRUD bench (≤ 2× LpgStore on each op).                |
//!
//! ## Step 3 design notes
//!
//! The three in-memory registries (labels, edge types, property keys)
//! plus the two slot high-water counters (`next_node_id`, `next_edge_id`)
//! are persisted as a single atomic sidecar file —
//! `<substrate-dir>/substrate.dict`. See [`crate::dict`] for the
//! bit-level layout and CRC scheme.
//!
//! **Why a separate file?** The format spec (§9) reserves the tail of
//! `substrate.meta` for the dict, but that slot is 4 KiB and we want
//! headroom without a v1→v2 format bump. A dedicated file is easier to
//! grow and avoids contention with the mmap'd meta header.
//!
//! **Why persist the counters here?** mmap-grow zeros the tail, so a
//! fresh zero NodeRecord on disk is indistinguishable from "allocated
//! slot that happens to carry zero data". The counters give us the
//! true high-water mark — without them the rebuild scan would have to
//! fall back to WAL replay to disambiguate.
//!
//! **Rebuild on open.** `from_substrate` loads the dict, repopulates
//! the three registries, sets the two atomic counters, then scans
//! `[1, next_*_id)` in each zone to restock:
//!
//! * `nodes` DashMap — labels from bitset (via `LabelRegistry`), empty
//!   property maps (step 4+ wires the property-page reader).
//! * `edges` DashMap — edge_type ArcStr (via `EdgeTypeRegistry`), empty
//!   property maps.
//! * `incoming_heads` — head per `dst` is the live edge with the highest
//!   `EdgeId` referencing it (splice-at-head invariant: newer ids sit at
//!   the front; middle removals never promote a smaller id above the
//!   current head).
//!
//! ## Step 2 design notes
//!
//! Edges live in two places:
//!
//! * **Durable slot** — the 32 B `EdgeRecord` in the `Edges` zone carries
//!   `src`, `dst`, `edge_type` (interned u16), `weight_u16`, `next_from`
//!   (same-src chain pointer), `next_to` (same-dst chain pointer), and the
//!   cognitive fields (`ricci_u8`, `flags`, `engram_tag`).
//! * **In-memory side** — `edge_type` name (ArcStr) and the property map
//!   live in [`EdgeInMem`], indexed by [`EdgeId`]. Step 3 persists these
//!   via the dedicated edge-type registry and property pages.
//!
//! **Outgoing chain**: entered at O(1) via `NodeRecord.first_edge_off`,
//! walked via `EdgeRecord.next_from`.
//!
//! **Incoming chain**: the `EdgeRecord.next_to` chain exists on-disk but
//! the entry point lives in memory only. [`SubstrateStore::incoming_heads`]
//! maps `NodeId → EdgeId` for the first incoming edge. This is **rebuilt on
//! open** by a single O(E) scan over the Edges zone — step 3 may promote
//! it to a sidecar (`substrate.in_edges`, 6 B × node_count) if the rebuild
//! cost matters at scale.
//!
//! Rationale: keeping `NodeRecord = 32 B` as invariant'd by the format
//! spec means we cannot add a `first_in_edge_off` field without a v1→v2
//! format bump. The in-memory head index is cheap, symmetric with
//! step 1's label/property pattern, and one `O(E)` scan at open time is
//! a small price compared to the cost of a format revision.
//!
//! ## Step 1 design notes (carried forward)
//!
//! The 32-byte `NodeRecord` in the `Nodes` zone is the **durable** slot —
//! `label_bitset`, `flags`, `community_id`, `energy`, etc. are all there.
//! What's *not* yet durable at step 1 or 2:
//!
//! * **Label names** (ArcStr → bit position) live in an in-memory
//!   [`LabelRegistry`]. Step 3 persists this into the dedicated zone.
//! * **Properties** (PropertyKey/Value pairs) for both nodes and edges
//!   live in in-memory `DashMap`s. Step 3 wires them onto the `Props` +
//!   `Strings` zones.
//! * **Edge-type names** (ArcStr ↔ u16) live in an in-memory
//!   [`EdgeTypeRegistry`]. Step 3 persists this alongside the label
//!   registry.
//!
//! Close/reopen parity for labels/properties/edge-types lives in step 3;
//! step 2's test suite therefore exercises only single-process lifetimes.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use arcstr::ArcStr;
use dashmap::DashMap;
use obrain_common::types::{
    EdgeId, EpochId, NodeId, OrderableValue, PropertyKey, PropertyMap, TransactionId, Value,
};
use obrain_common::utils::hash::FxHashMap;
use obrain_core::graph::Direction;
use obrain_core::graph::lpg::{CompareOp, Edge, Node};
use obrain_core::graph::traits::{GraphStore, GraphStoreMut};
use obrain_core::statistics::Statistics;
use parking_lot::{Mutex, RwLock};
use smallvec::SmallVec;

use crate::error::{SubstrateError, SubstrateResult};
use crate::file::SubstrateFile;
use crate::record::{
    EdgeRecord, NodeRecord, PackedScarUtilAff, U48, edge_flags, f32_to_q1_15,
};
use crate::wal_io::SyncMode;
use crate::writer::Writer;

/// Dict side-car filename relative to the substrate directory.
const DICT_FILENAME: &str = "substrate.dict";

// ---------------------------------------------------------------------------
// Label registry (step 1 in-memory version)
//
// 64-bit `label_bitset` on NodeRecord = at most 64 distinct labels at a time.
// In step 3 this grows to a spill-page structure; for step 1 we error out on
// overflow so tests that exercise > 64 labels fail loudly instead of silently
// collapsing different labels onto the same bit.
// ---------------------------------------------------------------------------

#[derive(Default)]
struct LabelRegistry {
    /// `name → bit_index` (0..64).
    name_to_bit: FxHashMap<ArcStr, u8>,
    /// `bit_index → name`, dense.
    bit_to_name: Vec<ArcStr>,
}

impl LabelRegistry {
    /// Register a label name if absent, return its bit index in [0, 63].
    fn intern(&mut self, name: &str) -> SubstrateResult<u8> {
        if let Some(&bit) = self.name_to_bit.get(name) {
            return Ok(bit);
        }
        let bit = self.bit_to_name.len();
        if bit >= 64 {
            return Err(SubstrateError::Internal(format!(
                "LabelRegistry overflow: more than 64 distinct labels (requested {name:?}); \
                 step 3 adds spill-page support"
            )));
        }
        let name: ArcStr = name.into();
        self.name_to_bit.insert(name.clone(), bit as u8);
        self.bit_to_name.push(name);
        Ok(bit as u8)
    }

    fn bit_for(&self, name: &str) -> Option<u8> {
        self.name_to_bit.get(name).copied()
    }

    fn labels_for_bitset(&self, bitset: u64) -> SmallVec<[ArcStr; 2]> {
        let mut out = SmallVec::new();
        for bit in 0..64u8 {
            if bitset & (1u64 << bit) != 0
                && let Some(name) = self.bit_to_name.get(bit as usize)
            {
                out.push(name.clone());
            }
        }
        out
    }

    /// Dump the label names in dense order for persistence via
    /// [`crate::dict::DictSnapshot`].
    fn names(&self) -> Vec<String> {
        self.bit_to_name.iter().map(|s| s.to_string()).collect()
    }

    /// Rehydrate the registry from a dict snapshot. Replaces any
    /// existing content.
    fn load_from(&mut self, names: &[String]) -> SubstrateResult<()> {
        self.name_to_bit.clear();
        self.bit_to_name.clear();
        if names.len() > 64 {
            return Err(SubstrateError::Internal(format!(
                "LabelRegistry::load_from: {} labels exceeds 64-bit bitset",
                names.len()
            )));
        }
        for (bit, name) in names.iter().enumerate() {
            let arc: ArcStr = name.as_str().into();
            self.name_to_bit.insert(arc.clone(), bit as u8);
            self.bit_to_name.push(arc);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Edge-type registry (persistent from step 3)
//
// Each edge's `edge_type: u16` is an interned id. The mapping persists
// via `substrate.dict` alongside the label + property-key registries.
// ---------------------------------------------------------------------------

#[derive(Default)]
struct EdgeTypeRegistry {
    name_to_id: FxHashMap<ArcStr, u16>,
    id_to_name: Vec<ArcStr>, // indexed by id (u16), len == number of interned types
}

impl EdgeTypeRegistry {
    /// Register an edge-type name if absent, return its interned id.
    fn intern(&mut self, name: &str) -> SubstrateResult<u16> {
        if let Some(&id) = self.name_to_id.get(name) {
            return Ok(id);
        }
        let id = self.id_to_name.len();
        if id >= u16::MAX as usize {
            return Err(SubstrateError::Internal(format!(
                "EdgeTypeRegistry overflow: more than {} distinct edge types \
                 (requested {name:?}); EdgeRecord.edge_type is u16",
                u16::MAX
            )));
        }
        let name: ArcStr = name.into();
        self.name_to_id.insert(name.clone(), id as u16);
        self.id_to_name.push(name);
        Ok(id as u16)
    }

    fn name_for(&self, id: u16) -> Option<ArcStr> {
        self.id_to_name.get(id as usize).cloned()
    }

    #[allow(dead_code)] // reserved for future dedup fast-paths
    fn id_for(&self, name: &str) -> Option<u16> {
        self.name_to_id.get(name).copied()
    }

    /// Dump the edge-type names in id order for persistence via
    /// [`crate::dict::DictSnapshot`].
    fn names(&self) -> Vec<String> {
        self.id_to_name.iter().map(|s| s.to_string()).collect()
    }

    /// Rehydrate the registry from a dict snapshot. Replaces any
    /// existing content.
    fn load_from(&mut self, names: &[String]) -> SubstrateResult<()> {
        self.name_to_id.clear();
        self.id_to_name.clear();
        if names.len() > u16::MAX as usize {
            return Err(SubstrateError::Internal(format!(
                "EdgeTypeRegistry::load_from: {} names exceeds u16 id space",
                names.len()
            )));
        }
        for (id, name) in names.iter().enumerate() {
            let arc: ArcStr = name.as_str().into();
            self.name_to_id.insert(arc.clone(), id as u16);
            self.id_to_name.push(arc);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Property-key registry (persistent from step 3)
//
// Property-key names are interned to u16 ids that index into the on-disk
// `Props` zone (step 4+ will wire the actual page-based storage). For step
// 3 the registry is only used for persistence parity with labels and edge
// types; nobody interns through this registry yet.
// ---------------------------------------------------------------------------

#[derive(Default)]
struct PropertyKeyRegistry {
    name_to_id: FxHashMap<ArcStr, u16>,
    id_to_name: Vec<ArcStr>,
}

impl PropertyKeyRegistry {
    /// Register a property-key name if absent, return its interned id.
    #[allow(dead_code)] // wired by step 4 (property page writer)
    fn intern(&mut self, name: &str) -> SubstrateResult<u16> {
        if let Some(&id) = self.name_to_id.get(name) {
            return Ok(id);
        }
        let id = self.id_to_name.len();
        if id >= u16::MAX as usize {
            return Err(SubstrateError::Internal(format!(
                "PropertyKeyRegistry overflow: more than {} distinct property keys \
                 (requested {name:?})",
                u16::MAX
            )));
        }
        let name: ArcStr = name.into();
        self.name_to_id.insert(name.clone(), id as u16);
        self.id_to_name.push(name);
        Ok(id as u16)
    }

    /// Dump the property-key names in id order for persistence via
    /// [`crate::dict::DictSnapshot`].
    fn names(&self) -> Vec<String> {
        self.id_to_name.iter().map(|s| s.to_string()).collect()
    }

    /// Rehydrate the registry from a dict snapshot. Replaces any
    /// existing content.
    fn load_from(&mut self, names: &[String]) -> SubstrateResult<()> {
        self.name_to_id.clear();
        self.id_to_name.clear();
        if names.len() > u16::MAX as usize {
            return Err(SubstrateError::Internal(format!(
                "PropertyKeyRegistry::load_from: {} names exceeds u16 id space",
                names.len()
            )));
        }
        for (id, name) in names.iter().enumerate() {
            let arc: ArcStr = name.as_str().into();
            self.name_to_id.insert(arc.clone(), id as u16);
            self.id_to_name.push(arc);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// SubstrateStore
// ---------------------------------------------------------------------------

/// Graph store backed by a `SubstrateFile` (mmap zones + WAL).
///
/// Construction goes through [`SubstrateStore::open`] or
/// [`SubstrateStore::create`]; you do not build one from an already-opened
/// `SubstrateFile` because the store owns the `Writer` that journals
/// mutations.
pub struct SubstrateStore {
    /// The underlying substrate file. Held through `Writer::substrate()`
    /// as well; this handle lets read-only methods skip the writer entirely.
    #[allow(dead_code)]
    substrate: Arc<Mutex<SubstrateFile>>,
    /// Writer-side state (mutex-guarded WAL writer + zone cache).
    writer: Arc<Writer>,
    /// Next node slot to allocate. Slot 0 is the "null sentinel" — we start
    /// at 1 so `NodeId::default()` (which is 0) can never collide with a
    /// real node.
    next_node_id: AtomicU32,
    /// Next edge slot to allocate. Slot 0 is the null-edge sentinel per the
    /// format spec (EdgeRecord at index 0 is never a real edge); allocation
    /// starts at 1.
    next_edge_id: AtomicU64,
    /// In-memory node state (labels + properties). The durable skeleton
    /// lives in the Nodes zone; the fields here become persistent in later
    /// T4 steps.
    nodes: DashMap<NodeId, NodeInMem>,
    /// In-memory edge state (edge-type name ArcStr + property map). The
    /// durable skeleton lives in the Edges zone; these fields become
    /// persistent in step 3.
    edges: DashMap<EdgeId, EdgeInMem>,
    /// First incoming edge head per destination node. The `next_to` chain
    /// on `EdgeRecord` is durable, but the entry point is kept in memory
    /// only — step 3 may promote it to a sidecar file. Rebuilt from an
    /// O(E) zone scan on open.
    incoming_heads: DashMap<NodeId, EdgeId>,
    /// Label name ↔ bit index, keyed by `ArcStr`. Persisted via
    /// `substrate.dict` (step 3).
    labels: RwLock<LabelRegistry>,
    /// Edge-type name ↔ u16 id registry. Persisted via `substrate.dict`
    /// (step 3).
    edge_types: RwLock<EdgeTypeRegistry>,
    /// Property-key name ↔ u16 id registry. Persisted via
    /// `substrate.dict` (step 3); actual property-page writes land in
    /// step 4+.
    prop_keys: RwLock<PropertyKeyRegistry>,
    /// Cached statistics snapshot (cost-based optimizer). Step 4 wires this
    /// to real stats; for now a fresh empty snapshot is returned.
    stats: Arc<Statistics>,
}

#[derive(Clone, Default)]
struct NodeInMem {
    labels: SmallVec<[ArcStr; 2]>,
    properties: PropertyMap,
}

#[derive(Clone, Default)]
struct EdgeInMem {
    edge_type: ArcStr,
    properties: PropertyMap,
}

impl SubstrateStore {
    /// Open an existing substrate at `path`, journaling in `EveryCommit` mode.
    pub fn open(path: impl AsRef<Path>) -> SubstrateResult<Self> {
        Self::open_with_mode(path, SyncMode::EveryCommit)
    }

    /// Create a new substrate at `path` and open it in `EveryCommit` mode.
    pub fn create(path: impl AsRef<Path>) -> SubstrateResult<Self> {
        let sub = SubstrateFile::create(path.as_ref())?;
        Self::from_substrate(sub, SyncMode::EveryCommit)
    }

    /// Open with an explicit sync mode — useful in benches.
    pub fn open_with_mode(
        path: impl AsRef<Path>,
        sync_mode: SyncMode,
    ) -> SubstrateResult<Self> {
        let sub = SubstrateFile::open(path.as_ref())?;
        Self::from_substrate(sub, sync_mode)
    }

    fn from_substrate(
        sub: SubstrateFile,
        sync_mode: SyncMode,
    ) -> SubstrateResult<Self> {
        // (1) Load the persisted dict (registries + slot allocator state).
        // Missing on fresh create → DictSnapshot::default() (empty
        // registries, counters = 1).
        let dict_path = sub.path().join(DICT_FILENAME);
        let snapshot = crate::dict::DictSnapshot::load(&dict_path)?;

        let mut labels = LabelRegistry::default();
        labels.load_from(&snapshot.labels)?;
        let mut edge_types = EdgeTypeRegistry::default();
        edge_types.load_from(&snapshot.edge_types)?;
        let mut prop_keys = PropertyKeyRegistry::default();
        prop_keys.load_from(&snapshot.prop_keys)?;

        let writer = Writer::new(sub, sync_mode)?;
        let substrate = writer.substrate();

        let store = Self {
            substrate,
            writer: Arc::new(writer),
            next_node_id: AtomicU32::new(snapshot.next_node_id as u32),
            next_edge_id: AtomicU64::new(snapshot.next_edge_id),
            nodes: DashMap::new(),
            edges: DashMap::new(),
            incoming_heads: DashMap::new(),
            labels: RwLock::new(labels),
            edge_types: RwLock::new(edge_types),
            prop_keys: RwLock::new(prop_keys),
            stats: Arc::new(Statistics::new()),
        };

        // (2) Rebuild in-memory side-cars from the on-disk zones. This is
        //     an O(N + E) scan bounded by the just-loaded high-water marks
        //     — zero-filled slots past the marks are never read.
        store.rebuild_from_zones()?;

        Ok(store)
    }

    /// Walk the Nodes + Edges zones and rebuild:
    ///
    /// * the in-memory `nodes` DashMap (labels from bitset; properties
    ///   are lost at step 3 since property pages land in step 4+);
    /// * the in-memory `edges` DashMap (edge-type ArcStr via the
    ///   registry; properties are lost at step 3 for the same reason);
    /// * the `incoming_heads` map — the first live edge on each
    ///   destination's `next_to` chain, identified as the live edge with
    ///   the highest `EdgeId` per `dst` (splice-at-head invariant: newer
    ///   edges always sit at the front).
    ///
    /// Slots 1..`next_node_id` and 1..`next_edge_id` are considered
    /// allocated; anything further is zero-initialised mmap padding and
    /// ignored.
    fn rebuild_from_zones(&self) -> SubstrateResult<()> {
        // ---- Nodes: rebuild labels view ----
        let node_hw = self.next_node_id.load(Ordering::Acquire);
        let labels_guard = self.labels.read();
        for slot in 1..node_hw {
            let Some(rec) = self.writer.read_node(slot)? else {
                // Zone shorter than the persisted high-water mark. This
                // would indicate corruption; loudly ignore and continue
                // — replay may still fill the gap.
                continue;
            };
            if rec.flags & crate::record::node_flags::TOMBSTONED != 0 {
                continue;
            }
            let labels = labels_guard.labels_for_bitset(rec.label_bitset);
            self.nodes.insert(
                NodeId(slot as u64),
                NodeInMem {
                    labels,
                    properties: PropertyMap::new(),
                },
            );
        }
        drop(labels_guard);

        // ---- Edges: rebuild edges DashMap + incoming_heads ----
        let edge_hw = self.next_edge_id.load(Ordering::Acquire);
        let edge_types_guard = self.edge_types.read();
        // For each dst, track the highest live EdgeId seen so far —
        // that's the head of the incoming chain (splice-at-head
        // invariant: newer ids are spliced at the front, and edges only
        // drop out of middle positions; head is always the max live id).
        let mut heads: FxHashMap<NodeId, EdgeId> = FxHashMap::default();
        for slot in 1..edge_hw {
            let Some(rec) = self.writer.read_edge(slot)? else {
                continue;
            };
            if rec.flags & edge_flags::TOMBSTONED != 0 {
                continue;
            }
            let edge_id = EdgeId(slot);
            let edge_type = edge_types_guard
                .name_for(rec.edge_type)
                .unwrap_or_else(|| {
                    // Should never happen: every edge_type id was interned
                    // before the edge was written. If it does, fall back to
                    // a synthetic ArcStr so the map entry is still usable.
                    tracing::error!(
                        target: "substrate",
                        "rebuild: edge slot {slot} references unknown edge_type id {} — \
                         dict/zone drift?",
                        rec.edge_type
                    );
                    ArcStr::from(format!("__UNKNOWN_{}", rec.edge_type))
                });
            self.edges.insert(
                edge_id,
                EdgeInMem {
                    edge_type,
                    properties: PropertyMap::new(),
                },
            );
            let dst = NodeId(rec.dst as u64);
            heads
                .entry(dst)
                .and_modify(|cur| {
                    if cur.0 < edge_id.0 {
                        *cur = edge_id;
                    }
                })
                .or_insert(edge_id);
        }
        drop(edge_types_guard);
        for (dst, head) in heads {
            self.incoming_heads.insert(dst, head);
        }

        Ok(())
    }

    /// Build a [`DictSnapshot`] capturing all three registries + slot
    /// allocator state. Called by [`Self::flush`] and tests.
    fn build_dict_snapshot(&self) -> crate::dict::DictSnapshot {
        crate::dict::DictSnapshot {
            labels: self.labels.read().names(),
            edge_types: self.edge_types.read().names(),
            prop_keys: self.prop_keys.read().names(),
            next_node_id: self.next_node_id.load(Ordering::Acquire) as u64,
            next_edge_id: self.next_edge_id.load(Ordering::Acquire),
        }
    }

    /// Persist the dict to `<substrate-dir>/substrate.dict`.
    fn persist_dict(&self) -> SubstrateResult<()> {
        let snapshot = self.build_dict_snapshot();
        let path = {
            let sub = self.substrate.lock();
            sub.path().join(DICT_FILENAME)
        };
        snapshot.persist(&path)
    }

    /// Force-flush pending mutations (fsync WAL + msync zones + persist dict).
    ///
    /// Order matters:
    /// 1. `writer.commit()` — seals the WAL with a commit marker and
    ///    fsyncs under `SyncMode::EveryCommit`.
    /// 2. `writer.msync_zones()` — flushes dirty mmap pages so the
    ///    Nodes/Edges zones on disk agree with the WAL.
    /// 3. `persist_dict()` — atomically rewrites `substrate.dict` with
    ///    the current registries and slot counters. Done last so any
    ///    crash before here leaves the WAL as source of truth (replay
    ///    will reconstruct both).
    pub fn flush(&self) -> SubstrateResult<()> {
        self.writer.commit()?;
        self.writer.msync_zones()?;
        self.persist_dict()?;
        Ok(())
    }

    /// Number of node slots handed out so far (including tombstoned).
    pub fn slot_high_water(&self) -> u32 {
        self.next_node_id.load(Ordering::Acquire)
    }

    /// Number of edge slots handed out so far (including tombstoned).
    pub fn edge_slot_high_water(&self) -> u64 {
        self.next_edge_id.load(Ordering::Acquire)
    }

    // ==================================================================
    // Cognitive column API (T6)
    // ==================================================================
    //
    // Typed accessors for the cognitive columns of NodeRecord / EdgeRecord.
    // The f32 variants are ergonomic wrappers around Q1.15 / Q0.16 quantization
    // helpers in `crate::record`. Callers can work in normalized [0, 1] floats
    // and the store transparently handles packing + WAL logging.

    /// Read a node's energy column (`NodeRecord.energy`) as `f32` in `[0.0, 2.0)`.
    /// Returns `None` if the slot is unallocated or beyond the zone.
    pub fn get_node_energy_f32(&self, id: NodeId) -> SubstrateResult<Option<f32>> {
        if id.0 == 0 || id.0 as u32 >= self.slot_high_water() {
            return Ok(None);
        }
        Ok(self
            .writer
            .read_node(id.0 as u32)?
            .filter(|r| !r.is_tombstoned())
            .map(|r| r.energy_q15()))
    }

    /// Overwrite a node's energy column with `value` (clamped to `[0.0, 1.0]`).
    /// The WAL record is an [`WalPayload::EnergyReinforce`] (absolute).
    pub fn set_node_energy_f32(&self, id: NodeId, value: f32) -> SubstrateResult<()> {
        let q = f32_to_q1_15(value);
        self.writer.reinforce_energy(id.0 as u32, q)?;
        Ok(())
    }

    /// Additive-reinforce the node's energy column by `amount`, clamped at
    /// `max`. Returns the new energy in `f32`. If the node is missing or
    /// tombstoned, returns `None`.
    pub fn boost_node_energy_f32(
        &self,
        id: NodeId,
        amount: f32,
        max: f32,
    ) -> SubstrateResult<Option<f32>> {
        let Some(rec) = self.writer.read_node(id.0 as u32)? else {
            return Ok(None);
        };
        if rec.is_tombstoned() {
            return Ok(None);
        }
        let cur = rec.energy_q15();
        let new = (cur + amount).clamp(0.0, max);
        self.set_node_energy_f32(id, new)?;
        Ok(Some(new))
    }

    /// Apply a multiplicative decay to every live node's energy column
    /// (`energy ← energy × factor`). A single `EnergyDecay` WAL record is
    /// written, then every live slot is rewritten in one pass. `factor` is
    /// clamped to `[0.0, 1.0]`.
    pub fn decay_all_energy(&self, factor: f32) -> SubstrateResult<()> {
        let factor_q16 =
            (factor.clamp(0.0, 1.0) * 65536.0).round().min(65535.0) as u16;
        self.writer
            .decay_all_energy(factor_q16, self.slot_high_water())
    }

    /// Read a node's packed `scar_util_affinity` column.
    pub fn get_node_scar_util_affinity(
        &self,
        id: NodeId,
    ) -> SubstrateResult<Option<PackedScarUtilAff>> {
        if id.0 == 0 || id.0 as u32 >= self.slot_high_water() {
            return Ok(None);
        }
        Ok(self
            .writer
            .read_node(id.0 as u32)?
            .filter(|r| !r.is_tombstoned())
            .map(|r| r.unpack_scar_util_aff()))
    }

    /// Overwrite the packed scar/utility/affinity byte-pair.
    pub fn set_node_scar_util_affinity(
        &self,
        id: NodeId,
        packed: PackedScarUtilAff,
    ) -> SubstrateResult<()> {
        self.writer
            .set_scar_util_affinity(id.0 as u32, packed.pack())?;
        Ok(())
    }

    /// Overwrite only the scar sub-field of the packed column. Utility and
    /// affinity bits are preserved (read-modify-write under the zone lock).
    /// `intensity` is clamped to `[0, SCAR_MAX_INTENSITY_Q5]` before
    /// quantization to 5 bits.
    pub fn set_node_scar_field_f32(&self, id: NodeId, intensity: f32) -> SubstrateResult<()> {
        self.writer
            .update_scar_field(id.0 as u32, crate::record::scar_to_q5(intensity))
    }

    /// Overwrite only the utility sub-field. `score` is clamped to
    /// `[0, UTILITY_MAX_SCORE_Q5]` before quantization.
    pub fn set_node_utility_field_f32(&self, id: NodeId, score: f32) -> SubstrateResult<()> {
        self.writer
            .update_utility_field(id.0 as u32, crate::record::utility_to_q5(score))
    }

    /// Overwrite only the affinity sub-field. `score` is clamped to `[0, 1]`.
    pub fn set_node_affinity_field_f32(&self, id: NodeId, score: f32) -> SubstrateResult<()> {
        self.writer
            .update_affinity_field(id.0 as u32, crate::record::affinity_to_q5(score))
    }

    /// Read the scar sub-field as `f32` (dequantized to
    /// `[0, SCAR_MAX_INTENSITY_Q5]`). Returns `None` if slot missing.
    pub fn get_node_scar_field_f32(&self, id: NodeId) -> SubstrateResult<Option<f32>> {
        Ok(self
            .get_node_scar_util_affinity(id)?
            .map(|p| crate::record::q5_to_scar(p.scar)))
    }

    /// Read the utility sub-field as `f32` (dequantized to
    /// `[0, UTILITY_MAX_SCORE_Q5]`).
    pub fn get_node_utility_field_f32(&self, id: NodeId) -> SubstrateResult<Option<f32>> {
        Ok(self
            .get_node_scar_util_affinity(id)?
            .map(|p| crate::record::q5_to_utility(p.utility)))
    }

    /// Read the affinity sub-field as `f32` (dequantized to `[0, 1]`).
    pub fn get_node_affinity_field_f32(&self, id: NodeId) -> SubstrateResult<Option<f32>> {
        Ok(self
            .get_node_scar_util_affinity(id)?
            .map(|p| crate::record::q5_to_affinity(p.affinity)))
    }

    /// Read an edge's synapse weight column (`EdgeRecord.weight_u16`) as `f32`.
    pub fn get_edge_synapse_weight_f32(
        &self,
        id: EdgeId,
    ) -> SubstrateResult<Option<f32>> {
        if id.0 == 0 || id.0 >= self.edge_slot_high_water() {
            return Ok(None);
        }
        Ok(self
            .writer
            .read_edge(id.0)?
            .filter(|r| r.flags & edge_flags::TOMBSTONED == 0)
            .map(|r| r.weight_f32()))
    }

    /// Reinforce an edge's synapse weight column. Clamps to `[0.0, 1.0]`.
    pub fn reinforce_edge_synapse_f32(
        &self,
        id: EdgeId,
        new_weight: f32,
    ) -> SubstrateResult<()> {
        let q = (new_weight.clamp(0.0, 1.0) * 65535.0).round() as u16;
        self.writer.reinforce_synapse(id.0, q)?;
        Ok(())
    }

    /// Additive-reinforce an edge's synapse weight by `amount`, clamped at
    /// `max`. Mirror of [`Self::boost_node_energy_f32`]. Returns the new
    /// weight in `f32`; returns `None` if the edge is missing or tombstoned.
    pub fn boost_edge_synapse_f32(
        &self,
        id: EdgeId,
        amount: f32,
        max: f32,
    ) -> SubstrateResult<Option<f32>> {
        let Some(rec) = self.writer.read_edge(id.0)? else {
            return Ok(None);
        };
        if rec.flags & edge_flags::TOMBSTONED != 0 {
            return Ok(None);
        }
        let cur = rec.weight_f32();
        let new = (cur + amount).clamp(0.0, max);
        self.reinforce_edge_synapse_f32(id, new)?;
        Ok(Some(new))
    }

    /// Apply a multiplicative decay to every live edge's synapse weight
    /// column (`weight ← weight × factor`). A single `SynapseDecay` WAL
    /// record is written, then every live slot is rewritten in one pass.
    /// `factor` is clamped to `[0.0, 1.0]`. Mirror of
    /// [`Self::decay_all_energy`].
    pub fn decay_all_edge_synapse(&self, factor: f32) -> SubstrateResult<()> {
        let factor_q16 =
            (factor.clamp(0.0, 1.0) * 65536.0).round().min(65535.0) as u16;
        self.writer
            .decay_all_synapse(factor_q16, self.edge_slot_high_water())
    }

    /// Iterate live edges (non-tombstoned, slot > 0, slot < high-water) and
    /// produce a `Vec<(EdgeId, f32)>` of their current synapse weights. Used
    /// by `SynapseStore::snapshot()` in substrate-view mode.
    ///
    /// O(N) pass over the edges mmap.
    pub fn iter_live_synapse_weights(&self) -> SubstrateResult<Vec<(EdgeId, f32)>> {
        let hw = self.edge_slot_high_water();
        let mut out = Vec::with_capacity(hw.saturating_sub(1) as usize);
        for slot in 1..hw {
            let Some(rec) = self.writer.read_edge(slot)? else {
                continue;
            };
            if rec.flags & edge_flags::TOMBSTONED != 0 {
                continue;
            }
            out.push((EdgeId(slot), rec.weight_f32()));
        }
        Ok(out)
    }

    /// Iterate live nodes (non-tombstoned, slot > 0, slot < high-water) and
    /// produce a `Vec<(NodeId, f32)>` of their current energy columns. Used
    /// by cognitive stores' `snapshot()` / `list_low_energy()`.
    ///
    /// This is an O(N) pass over the nodes mmap — fine for megalaw-scale
    /// snapshots on the scale of 10⁶ nodes (≈ 32 MB sequential read).
    pub fn iter_live_node_energies(&self) -> SubstrateResult<Vec<(NodeId, f32)>> {
        let hw = self.slot_high_water();
        let mut out = Vec::with_capacity(hw.saturating_sub(1) as usize);
        for slot in 1..hw {
            let Some(rec) = self.writer.read_node(slot)? else {
                continue;
            };
            if rec.is_tombstoned() {
                continue;
            }
            out.push((NodeId(slot as u64), rec.energy_q15()));
        }
        Ok(out)
    }

    /// Iterate live nodes and produce `(NodeId, PackedScarUtilAff)` pairs.
    pub fn iter_live_scar_util_affinity(
        &self,
    ) -> SubstrateResult<Vec<(NodeId, PackedScarUtilAff)>> {
        let hw = self.slot_high_water();
        let mut out = Vec::with_capacity(hw.saturating_sub(1) as usize);
        for slot in 1..hw {
            let Some(rec) = self.writer.read_node(slot)? else {
                continue;
            };
            if rec.is_tombstoned() {
                continue;
            }
            out.push((NodeId(slot as u64), rec.unpack_scar_util_aff()));
        }
        Ok(out)
    }

    /// Batch-update centrality for a list of nodes (single WAL record).
    pub fn update_centrality_batch_f32(
        &self,
        updates: Vec<(NodeId, f32)>,
    ) -> SubstrateResult<()> {
        let encoded: Vec<(u32, u16)> = updates
            .into_iter()
            .map(|(id, v)| {
                let q = (v.clamp(0.0, 1.0) * 65535.0).round() as u16;
                (id.0 as u32, q)
            })
            .collect();
        self.writer.update_centrality_batch(encoded)
    }

    // ---- internal helpers ----

    fn intern_labels_to_bitset(&self, labels: &[&str]) -> SubstrateResult<u64> {
        let mut reg = self.labels.write();
        let mut bitset = 0u64;
        for l in labels {
            let bit = reg.intern(l)?;
            bitset |= 1u64 << bit;
        }
        Ok(bitset)
    }

    fn is_live_on_disk(&self, id: NodeId) -> bool {
        // Slot 0 is reserved; ids above the high-water mark are unallocated.
        if id.0 == 0 || id.0 as u32 >= self.next_node_id.load(Ordering::Acquire) {
            return false;
        }
        match self.writer.read_node(id.0 as u32) {
            Ok(Some(rec)) => !rec.is_tombstoned(),
            _ => false,
        }
    }

    /// Allocate the next node id (slot index).
    fn allocate_node_id(&self) -> NodeId {
        let raw = self.next_node_id.fetch_add(1, Ordering::AcqRel);
        NodeId(raw as u64)
    }

    /// Allocate the next edge slot (index into `substrate.edges`). Slot 0
    /// is the null-edge sentinel, so allocation starts at 1.
    fn allocate_edge_id(&self) -> EdgeId {
        let raw = self.next_edge_id.fetch_add(1, Ordering::AcqRel);
        EdgeId(raw)
    }

    /// Is the edge slot live (allocated, within high-water mark, not
    /// tombstoned on disk, and has the in-memory side-car)?
    fn is_live_edge_on_disk(&self, id: EdgeId) -> bool {
        if id.0 == 0 || id.0 >= self.next_edge_id.load(Ordering::Acquire) {
            return false;
        }
        match self.writer.read_edge(id.0) {
            Ok(Some(rec)) => rec.flags & edge_flags::TOMBSTONED == 0,
            _ => false,
        }
    }

    /// Convert an `EdgeId` slot to a byte offset in the Edges zone for
    /// use in `U48` chain pointers (`first_edge_off`, `next_from`,
    /// `next_to`). Slot 0 → offset 0 (sentinel).
    #[inline]
    fn edge_slot_to_offset(id: EdgeId) -> U48 {
        U48::from_u64(id.0 * EdgeRecord::SIZE as u64)
    }

    /// Convert a `U48` byte offset back to an `EdgeId` slot. Offset 0 →
    /// `EdgeId(0)` (sentinel).
    #[inline]
    fn offset_to_edge_id(off: U48) -> EdgeId {
        EdgeId(off.to_u64() / EdgeRecord::SIZE as u64)
    }

    /// Intern an edge-type name. Returns the interned u16 id.
    fn intern_edge_type(&self, name: &str) -> SubstrateResult<u16> {
        self.edge_types.write().intern(name)
    }

    /// Walk the outgoing edge chain of `src`, skipping tombstoned records.
    /// Entry point: `NodeRecord.first_edge_off`; link field: `next_from`.
    fn walk_outgoing_chain(
        &self,
        src: NodeId,
        mut visit: impl FnMut(&EdgeRecord, EdgeId),
    ) {
        let Some(src_rec) = self
            .writer
            .read_node(src.0 as u32)
            .ok()
            .flatten()
        else {
            return;
        };
        let mut cur = Self::offset_to_edge_id(src_rec.first_edge_off);
        let mut guard = 0u64;
        while cur.0 != 0 {
            // Cycle guard: no real graph should ever chain beyond the
            // slot-high-water mark.
            guard += 1;
            if guard > self.next_edge_id.load(Ordering::Acquire) + 1 {
                tracing::error!(
                    target: "substrate",
                    "walk_outgoing_chain cycle detected at src={}, cur={}",
                    src.0, cur.0
                );
                return;
            }
            let Some(rec) = self.writer.read_edge(cur.0).ok().flatten() else {
                return;
            };
            let next = Self::offset_to_edge_id(rec.next_from);
            if rec.flags & edge_flags::TOMBSTONED == 0 {
                visit(&rec, cur);
            }
            cur = next;
        }
    }

    /// Walk the incoming edge chain of `dst`, skipping tombstoned records.
    /// Entry point: `incoming_heads[dst]`; link field: `next_to`.
    fn walk_incoming_chain(
        &self,
        dst: NodeId,
        mut visit: impl FnMut(&EdgeRecord, EdgeId),
    ) {
        let Some(head_ref) = self.incoming_heads.get(&dst) else {
            return;
        };
        let mut cur = *head_ref;
        drop(head_ref);
        let mut guard = 0u64;
        while cur.0 != 0 {
            guard += 1;
            if guard > self.next_edge_id.load(Ordering::Acquire) + 1 {
                tracing::error!(
                    target: "substrate",
                    "walk_incoming_chain cycle detected at dst={}, cur={}",
                    dst.0, cur.0
                );
                return;
            }
            let Some(rec) = self.writer.read_edge(cur.0).ok().flatten() else {
                return;
            };
            let next = Self::offset_to_edge_id(rec.next_to);
            if rec.flags & edge_flags::TOMBSTONED == 0 {
                visit(&rec, cur);
            }
            cur = next;
        }
    }

    /// Splice a freshly-allocated edge at the head of both chains.
    ///
    /// Protocol:
    /// 1. Read `NodeRecord[src]` — capture old `first_edge_off`. Set
    ///    new edge's `next_from` to it.
    /// 2. Read `incoming_heads[dst]` — capture old head. Set new edge's
    ///    `next_to` to its byte offset.
    /// 3. Update `NodeRecord[src].first_edge_off` to point to the new edge
    ///    (via `update_node`).
    /// 4. Update `incoming_heads[dst]` in memory.
    /// 5. Write the new `EdgeRecord` via `write_edge`.
    fn splice_edge_at_head(
        &self,
        edge_id: EdgeId,
        src: NodeId,
        dst: NodeId,
        edge_type_id: u16,
    ) -> EdgeRecord {
        // (1) fetch current outgoing head of src
        let mut src_rec = self
            .writer
            .read_node(src.0 as u32)
            .ok()
            .flatten()
            .unwrap_or_default();
        let prev_out_head = src_rec.first_edge_off;

        // (2) fetch current incoming head of dst
        let prev_in_head = self
            .incoming_heads
            .get(&dst)
            .map(|e| *e)
            .unwrap_or(EdgeId(0));

        // Build the EdgeRecord with both chain pointers set.
        let edge = EdgeRecord {
            src: src.0 as u32,
            dst: dst.0 as u32,
            edge_type: edge_type_id,
            weight_u16: 0,
            next_from: prev_out_head,
            next_to: Self::edge_slot_to_offset(prev_in_head),
            ricci_u8: 0,
            flags: 0,
            engram_tag: 0,
            _pad: [0; 4],
        };

        // (3) update src's first_edge_off
        src_rec.first_edge_off = Self::edge_slot_to_offset(edge_id);
        self.writer
            .update_node(src.0 as u32, src_rec)
            .expect("update_node (splice src head) failed");

        // (4) update incoming_heads (in-memory)
        self.incoming_heads.insert(dst, edge_id);

        // (5) write the edge slot
        self.writer
            .write_edge(edge_id.0, edge)
            .expect("write_edge failed — WAL append or mmap grow");

        edge
    }

    /// Unlink `edge_id` from the outgoing chain of `src` and the incoming
    /// chain of `dst`. Called before tombstoning an edge.
    ///
    /// If `edge_id` is the head of `src`'s outgoing chain, updates
    /// `NodeRecord.first_edge_off`. Otherwise walks the chain to find the
    /// predecessor and patches its `next_from`.
    ///
    /// If `edge_id` is the head of `dst`'s incoming chain, updates the
    /// `incoming_heads` map. Otherwise walks and patches the predecessor's
    /// `next_to`.
    fn unlink_edge_from_chains(&self, edge_id: EdgeId, rec: &EdgeRecord) {
        let src_id = NodeId(rec.src as u64);
        let dst_id = NodeId(rec.dst as u64);

        // --- Outgoing chain ---
        let mut src_rec = self
            .writer
            .read_node(src_id.0 as u32)
            .ok()
            .flatten()
            .unwrap_or_default();
        if Self::offset_to_edge_id(src_rec.first_edge_off) == edge_id {
            // Was head — drop to next_from.
            src_rec.first_edge_off = rec.next_from;
            self.writer
                .update_node(src_id.0 as u32, src_rec)
                .expect("update_node (unlink src head) failed");
        } else {
            // Walk until we find the predecessor pointing at edge_id.
            let target_off = Self::edge_slot_to_offset(edge_id);
            let mut cur = Self::offset_to_edge_id(src_rec.first_edge_off);
            let mut guard = 0u64;
            while cur.0 != 0 {
                guard += 1;
                if guard > self.next_edge_id.load(Ordering::Acquire) + 1 {
                    tracing::error!(
                        target: "substrate",
                        "unlink_edge_from_chains cycle on out-chain src={}",
                        src_id.0
                    );
                    break;
                }
                let Some(mut prev_rec) = self.writer.read_edge(cur.0).ok().flatten() else {
                    break;
                };
                if prev_rec.next_from == target_off {
                    prev_rec.next_from = rec.next_from;
                    self.writer
                        .update_edge(cur.0, prev_rec)
                        .expect("update_edge (unlink src mid) failed");
                    break;
                }
                cur = Self::offset_to_edge_id(prev_rec.next_from);
            }
        }

        // --- Incoming chain ---
        let head = self
            .incoming_heads
            .get(&dst_id)
            .map(|e| *e)
            .unwrap_or(EdgeId(0));
        if head == edge_id {
            let next_id = Self::offset_to_edge_id(rec.next_to);
            if next_id.0 == 0 {
                self.incoming_heads.remove(&dst_id);
            } else {
                self.incoming_heads.insert(dst_id, next_id);
            }
        } else if head.0 != 0 {
            let target_off = Self::edge_slot_to_offset(edge_id);
            let mut cur = head;
            let mut guard = 0u64;
            while cur.0 != 0 {
                guard += 1;
                if guard > self.next_edge_id.load(Ordering::Acquire) + 1 {
                    tracing::error!(
                        target: "substrate",
                        "unlink_edge_from_chains cycle on in-chain dst={}",
                        dst_id.0
                    );
                    break;
                }
                let Some(mut prev_rec) = self.writer.read_edge(cur.0).ok().flatten() else {
                    break;
                };
                if prev_rec.next_to == target_off {
                    prev_rec.next_to = rec.next_to;
                    self.writer
                        .update_edge(cur.0, prev_rec)
                        .expect("update_edge (unlink dst mid) failed");
                    break;
                }
                cur = Self::offset_to_edge_id(prev_rec.next_to);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GraphStore — read side
// ---------------------------------------------------------------------------

impl GraphStore for SubstrateStore {
    // -- Point lookups --
    fn get_node(&self, id: NodeId) -> Option<Node> {
        if !self.is_live_on_disk(id) {
            return None;
        }
        let entry = self.nodes.get(&id)?;
        let mut n = Node::new(id);
        n.labels = entry.labels.clone();
        n.properties = entry.properties.clone();
        Some(n)
    }

    fn get_edge(&self, id: EdgeId) -> Option<Edge> {
        if !self.is_live_edge_on_disk(id) {
            return None;
        }
        let rec = self.writer.read_edge(id.0).ok().flatten()?;
        let mem = self.edges.get(&id)?;
        let mut e = Edge::new(
            id,
            NodeId(rec.src as u64),
            NodeId(rec.dst as u64),
            mem.edge_type.clone(),
        );
        e.properties = mem.properties.clone();
        Some(e)
    }

    fn get_node_versioned(
        &self,
        id: NodeId,
        _epoch: EpochId,
        _transaction_id: TransactionId,
    ) -> Option<Node> {
        // MVCC deferred to T5; for step 1 we treat every version as current.
        self.get_node(id)
    }

    fn get_edge_versioned(
        &self,
        id: EdgeId,
        _epoch: EpochId,
        _transaction_id: TransactionId,
    ) -> Option<Edge> {
        // MVCC deferred to T5; step 2 treats every version as current.
        self.get_edge(id)
    }

    fn get_node_at_epoch(&self, id: NodeId, _epoch: EpochId) -> Option<Node> {
        self.get_node(id)
    }

    fn get_edge_at_epoch(&self, id: EdgeId, _epoch: EpochId) -> Option<Edge> {
        self.get_edge(id)
    }

    // -- Property access --
    fn get_node_property(&self, id: NodeId, key: &PropertyKey) -> Option<Value> {
        if !self.is_live_on_disk(id) {
            return None;
        }
        let entry = self.nodes.get(&id)?;
        entry.properties.get(key).cloned()
    }

    fn get_edge_property(&self, id: EdgeId, key: &PropertyKey) -> Option<Value> {
        if !self.is_live_edge_on_disk(id) {
            return None;
        }
        let entry = self.edges.get(&id)?;
        entry.properties.get(key).cloned()
    }

    fn get_node_property_batch(
        &self,
        ids: &[NodeId],
        key: &PropertyKey,
    ) -> Vec<Option<Value>> {
        ids.iter().map(|id| self.get_node_property(*id, key)).collect()
    }

    fn get_nodes_properties_batch(
        &self,
        ids: &[NodeId],
    ) -> Vec<FxHashMap<PropertyKey, Value>> {
        ids.iter()
            .map(|id| {
                let mut out = FxHashMap::default();
                if !self.is_live_on_disk(*id) {
                    return out;
                }
                if let Some(entry) = self.nodes.get(id) {
                    for (k, v) in entry.properties.iter() {
                        out.insert(k.clone(), v.clone());
                    }
                }
                out
            })
            .collect()
    }

    fn get_nodes_properties_selective_batch(
        &self,
        ids: &[NodeId],
        keys: &[PropertyKey],
    ) -> Vec<FxHashMap<PropertyKey, Value>> {
        ids.iter()
            .map(|id| {
                let mut out = FxHashMap::default();
                if !self.is_live_on_disk(*id) {
                    return out;
                }
                if let Some(entry) = self.nodes.get(id) {
                    for k in keys {
                        if let Some(v) = entry.properties.get(k) {
                            out.insert(k.clone(), v.clone());
                        }
                    }
                }
                out
            })
            .collect()
    }

    fn get_edges_properties_selective_batch(
        &self,
        ids: &[EdgeId],
        keys: &[PropertyKey],
    ) -> Vec<FxHashMap<PropertyKey, Value>> {
        ids.iter()
            .map(|id| {
                let mut out = FxHashMap::default();
                if !self.is_live_edge_on_disk(*id) {
                    return out;
                }
                if let Some(entry) = self.edges.get(id) {
                    for k in keys {
                        if let Some(v) = entry.properties.get(k) {
                            out.insert(k.clone(), v.clone());
                        }
                    }
                }
                out
            })
            .collect()
    }

    // -- Traversal --
    //
    // Walk the intrusive chains on `EdgeRecord`:
    // * Outgoing: head at `NodeRecord.first_edge_off`, follow `next_from`.
    // * Incoming: head at `SubstrateStore.incoming_heads[dst]`, follow
    //   `next_to`.
    //
    // Tombstoned edges are skipped. The chain walker is cycle-safe by
    // construction (head insertion always writes the new edge with the
    // old head in its `next_*` field).
    fn neighbors(&self, node: NodeId, direction: Direction) -> Vec<NodeId> {
        self.edges_from(node, direction)
            .into_iter()
            .map(|(target, _)| target)
            .collect()
    }

    fn edges_from(
        &self,
        node: NodeId,
        direction: Direction,
    ) -> Vec<(NodeId, EdgeId)> {
        if !self.is_live_on_disk(node) {
            return Vec::new();
        }
        let mut out = Vec::new();
        match direction {
            Direction::Outgoing => {
                self.walk_outgoing_chain(node, |rec, edge_id| {
                    out.push((NodeId(rec.dst as u64), edge_id));
                });
            }
            Direction::Incoming => {
                self.walk_incoming_chain(node, |rec, edge_id| {
                    out.push((NodeId(rec.src as u64), edge_id));
                });
            }
            Direction::Both => {
                self.walk_outgoing_chain(node, |rec, edge_id| {
                    out.push((NodeId(rec.dst as u64), edge_id));
                });
                self.walk_incoming_chain(node, |rec, edge_id| {
                    out.push((NodeId(rec.src as u64), edge_id));
                });
            }
        }
        out
    }

    fn out_degree(&self, node: NodeId) -> usize {
        if !self.is_live_on_disk(node) {
            return 0;
        }
        let mut count = 0usize;
        self.walk_outgoing_chain(node, |_, _| count += 1);
        count
    }

    fn in_degree(&self, node: NodeId) -> usize {
        if !self.is_live_on_disk(node) {
            return 0;
        }
        let mut count = 0usize;
        self.walk_incoming_chain(node, |_, _| count += 1);
        count
    }

    fn has_backward_adjacency(&self) -> bool {
        // `EdgeRecord` keeps both `next_from` (src chain) and `next_to` (dst
        // chain). The concrete read paths light up in step 2; the answer is
        // already `true` structurally.
        true
    }

    // -- Scans --
    fn node_ids(&self) -> Vec<NodeId> {
        let mut out: Vec<NodeId> = self
            .nodes
            .iter()
            .filter(|e| self.is_live_on_disk(*e.key()))
            .map(|e| *e.key())
            .collect();
        out.sort_by_key(|n| n.0);
        out
    }

    fn nodes_by_label(&self, label: &str) -> Vec<NodeId> {
        let reg = self.labels.read();
        let Some(bit) = reg.bit_for(label) else {
            return Vec::new();
        };
        let mask = 1u64 << bit;
        drop(reg);

        let mut out: Vec<NodeId> = self
            .nodes
            .iter()
            .filter_map(|e| {
                let id = *e.key();
                if !self.is_live_on_disk(id) {
                    return None;
                }
                let rec = self.writer.read_node(id.0 as u32).ok()??;
                if rec.label_bitset & mask != 0 {
                    Some(id)
                } else {
                    None
                }
            })
            .collect();
        out.sort_by_key(|n| n.0);
        out
    }

    fn node_count(&self) -> usize {
        self.nodes
            .iter()
            .filter(|e| self.is_live_on_disk(*e.key()))
            .count()
    }

    fn edge_count(&self) -> usize {
        self.edges
            .iter()
            .filter(|e| self.is_live_edge_on_disk(*e.key()))
            .count()
    }

    // -- Entity metadata --
    fn edge_type(&self, id: EdgeId) -> Option<ArcStr> {
        if !self.is_live_edge_on_disk(id) {
            return None;
        }
        self.edges.get(&id).map(|e| e.edge_type.clone())
    }

    // -- Filtered search --
    fn find_nodes_by_property(&self, property: &str, value: &Value) -> Vec<NodeId> {
        let key = PropertyKey::new(property);
        let mut out: Vec<NodeId> = self
            .nodes
            .iter()
            .filter_map(|e| {
                let id = *e.key();
                if !self.is_live_on_disk(id) {
                    return None;
                }
                match e.value().properties.get(&key) {
                    Some(v) if v == value => Some(id),
                    _ => None,
                }
            })
            .collect();
        out.sort_by_key(|n| n.0);
        out
    }

    fn find_nodes_by_properties(&self, conditions: &[(&str, Value)]) -> Vec<NodeId> {
        let keys: Vec<PropertyKey> = conditions.iter().map(|(k, _)| PropertyKey::new(*k)).collect();
        let mut out: Vec<NodeId> = self
            .nodes
            .iter()
            .filter_map(|e| {
                let id = *e.key();
                if !self.is_live_on_disk(id) {
                    return None;
                }
                let props = &e.value().properties;
                for (key, (_, expected)) in keys.iter().zip(conditions.iter()) {
                    match props.get(key) {
                        Some(v) if v == expected => continue,
                        _ => return None,
                    }
                }
                Some(id)
            })
            .collect();
        out.sort_by_key(|n| n.0);
        out
    }

    fn find_nodes_in_range(
        &self,
        property: &str,
        min: Option<&Value>,
        max: Option<&Value>,
        min_inclusive: bool,
        max_inclusive: bool,
    ) -> Vec<NodeId> {
        // `Value` is not `Ord` — GQL has non-comparable variants (Null, Map,
        // List, Bytes, Vector). The `OrderableValue` wrapper carves out the
        // naturally-ordered subset; values outside that subset are filtered
        // out (matching LpgStore's behaviour).
        let min_o = min.and_then(|v| OrderableValue::try_from(v).ok());
        let max_o = max.and_then(|v| OrderableValue::try_from(v).ok());
        let key = PropertyKey::new(property);

        let mut out: Vec<NodeId> = self
            .nodes
            .iter()
            .filter_map(|e| {
                let id = *e.key();
                if !self.is_live_on_disk(id) {
                    return None;
                }
                let v = e.value().properties.get(&key)?.clone();
                let ov = OrderableValue::try_from(&v).ok()?;
                if let Some(m) = &min_o {
                    let cmp = ov.partial_cmp(m)?;
                    match (cmp, min_inclusive) {
                        (std::cmp::Ordering::Greater, _) => {}
                        (std::cmp::Ordering::Equal, true) => {}
                        _ => return None,
                    }
                }
                if let Some(m) = &max_o {
                    let cmp = ov.partial_cmp(m)?;
                    match (cmp, max_inclusive) {
                        (std::cmp::Ordering::Less, _) => {}
                        (std::cmp::Ordering::Equal, true) => {}
                        _ => return None,
                    }
                }
                Some(id)
            })
            .collect();
        out.sort_by_key(|n| n.0);
        out
    }

    // -- Zone maps --
    fn node_property_might_match(
        &self,
        _property: &PropertyKey,
        _op: CompareOp,
        _value: &Value,
    ) -> bool {
        // Pessimistic: no zone maps yet → always say "might match". Query
        // engine falls back to scan. Step 8 wires up real zone maps.
        true
    }

    fn edge_property_might_match(
        &self,
        _property: &PropertyKey,
        _op: CompareOp,
        _value: &Value,
    ) -> bool {
        true
    }

    // -- Statistics --
    fn statistics(&self) -> Arc<Statistics> {
        self.stats.clone()
    }

    fn estimate_label_cardinality(&self, label: &str) -> f64 {
        self.nodes_by_label(label).len() as f64
    }

    fn estimate_avg_degree(&self, _edge_type: &str, _outgoing: bool) -> f64 {
        0.0 // step 2 wires real degree stats
    }

    // -- Epoch --
    fn current_epoch(&self) -> EpochId {
        // No MVCC pre-T5 — a constant zero epoch is visible to everyone.
        EpochId(0)
    }
}

// ---------------------------------------------------------------------------
// GraphStoreMut — write side
// ---------------------------------------------------------------------------

impl GraphStoreMut for SubstrateStore {
    // -- Node creation --
    fn create_node(&self, labels: &[&str]) -> NodeId {
        let id = self.allocate_node_id();
        let bitset = self
            .intern_labels_to_bitset(labels)
            .expect("label registry overflow (>64 labels); step 3 lifts this");
        let rec = NodeRecord {
            label_bitset: bitset,
            ..Default::default()
        };
        self.writer
            .write_node(id.0 as u32, rec)
            .expect("write_node failed — WAL append or mmap grow");
        let label_vec = {
            let reg = self.labels.read();
            reg.labels_for_bitset(bitset)
        };
        self.nodes.insert(
            id,
            NodeInMem {
                labels: label_vec,
                properties: PropertyMap::new(),
            },
        );
        id
    }

    fn create_node_versioned(
        &self,
        labels: &[&str],
        _epoch: EpochId,
        _transaction_id: TransactionId,
    ) -> NodeId {
        self.create_node(labels)
    }

    // -- Edge creation (step 2) --
    fn create_edge(&self, src: NodeId, dst: NodeId, edge_type: &str) -> EdgeId {
        // LpgStore is permissive: it does NOT reject edges on missing
        // endpoints — they just become dangling. We match that contract;
        // the caller is responsible for endpoint liveness.
        let edge_type_id = self
            .intern_edge_type(edge_type)
            .expect("edge-type registry overflow (>65535 types); lifted in step 3");
        let id = self.allocate_edge_id();
        let _rec = self.splice_edge_at_head(id, src, dst, edge_type_id);

        // In-memory side — edge_type name + empty property map.
        self.edges.insert(
            id,
            EdgeInMem {
                edge_type: edge_type.into(),
                properties: PropertyMap::new(),
            },
        );
        id
    }

    fn create_edge_versioned(
        &self,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
        _epoch: EpochId,
        _transaction_id: TransactionId,
    ) -> EdgeId {
        self.create_edge(src, dst, edge_type)
    }

    fn batch_create_edges(&self, edges: &[(NodeId, NodeId, &str)]) -> Vec<EdgeId> {
        edges
            .iter()
            .map(|(s, d, t)| self.create_edge(*s, *d, t))
            .collect()
    }

    // -- Deletion --
    fn delete_node(&self, id: NodeId) -> bool {
        if !self.is_live_on_disk(id) {
            return false;
        }
        // Flip the TOMBSTONED flag in the on-disk slot + journal NodeDelete.
        self.writer
            .tombstone_node(id.0 as u32)
            .expect("tombstone_node failed");
        // Drop from the in-memory side-table — get_node will now return None.
        self.nodes.remove(&id);
        true
    }

    fn delete_node_versioned(
        &self,
        id: NodeId,
        _epoch: EpochId,
        _transaction_id: TransactionId,
    ) -> bool {
        self.delete_node(id)
    }

    fn delete_node_edges(&self, node_id: NodeId) {
        // DETACH DELETE — collect both directions first, then tombstone.
        // We cannot mutate the chains while walking them.
        if !self.is_live_on_disk(node_id) {
            return;
        }
        let mut to_delete: Vec<EdgeId> = Vec::new();
        self.walk_outgoing_chain(node_id, |_, id| to_delete.push(id));
        self.walk_incoming_chain(node_id, |_, id| to_delete.push(id));
        // Deduplicate — self-loops appear in both chains.
        to_delete.sort_by_key(|e| e.0);
        to_delete.dedup();
        for id in to_delete {
            self.delete_edge(id);
        }
    }

    fn delete_edge(&self, id: EdgeId) -> bool {
        if !self.is_live_edge_on_disk(id) {
            return false;
        }
        let Some(rec) = self.writer.read_edge(id.0).ok().flatten() else {
            return false;
        };
        if rec.flags & edge_flags::TOMBSTONED != 0 {
            return false;
        }
        // Splice the edge out of both chains before tombstoning the slot
        // — future walks must never encounter the dead edge.
        self.unlink_edge_from_chains(id, &rec);
        self.writer
            .tombstone_edge(id.0)
            .expect("tombstone_edge failed");
        self.edges.remove(&id);
        true
    }

    fn delete_edge_versioned(
        &self,
        id: EdgeId,
        _epoch: EpochId,
        _transaction_id: TransactionId,
    ) -> bool {
        self.delete_edge(id)
    }

    // -- Property mutation --
    fn set_node_property(&self, id: NodeId, key: &str, value: Value) {
        if !self.is_live_on_disk(id) {
            // LpgStore silently no-ops on missing nodes; match that contract.
            return;
        }
        self.nodes
            .entry(id)
            .or_default()
            .properties
            .insert(PropertyKey::new(key), value);
    }

    fn set_edge_property(&self, id: EdgeId, key: &str, value: Value) {
        if !self.is_live_edge_on_disk(id) {
            // LpgStore silently no-ops on missing edges; match that contract.
            return;
        }
        self.edges
            .entry(id)
            .or_default()
            .properties
            .insert(PropertyKey::new(key), value);
    }

    fn remove_node_property(&self, id: NodeId, key: &str) -> Option<Value> {
        if !self.is_live_on_disk(id) {
            return None;
        }
        self.nodes
            .get_mut(&id)?
            .properties
            .remove(&PropertyKey::new(key))
    }

    fn remove_edge_property(&self, id: EdgeId, key: &str) -> Option<Value> {
        if !self.is_live_edge_on_disk(id) {
            return None;
        }
        self.edges
            .get_mut(&id)?
            .properties
            .remove(&PropertyKey::new(key))
    }

    // -- Label mutation --
    fn add_label(&self, node_id: NodeId, label: &str) -> bool {
        if !self.is_live_on_disk(node_id) {
            return false;
        }
        // Update the in-memory side and recompute the bitset.
        let mut entry = self.nodes.entry(node_id).or_default();
        if entry.labels.iter().any(|l| l.as_str() == label) {
            return false;
        }
        entry.labels.push(label.into());
        let new_bitset = {
            let mut reg = self.labels.write();
            let mut b = 0u64;
            for l in &entry.labels {
                let bit = reg.intern(l.as_str()).expect("label registry overflow");
                b |= 1u64 << bit;
            }
            b
        };
        let old = self
            .writer
            .read_node(node_id.0 as u32)
            .ok()
            .flatten()
            .unwrap_or_default();
        let updated = NodeRecord {
            label_bitset: new_bitset,
            ..old
        };
        self.writer
            .update_node(node_id.0 as u32, updated)
            .expect("update_node failed");
        true
    }

    fn remove_label(&self, node_id: NodeId, label: &str) -> bool {
        if !self.is_live_on_disk(node_id) {
            return false;
        }
        let mut entry = self.nodes.entry(node_id).or_default();
        let before = entry.labels.len();
        entry.labels.retain(|l| l.as_str() != label);
        if entry.labels.len() == before {
            return false;
        }
        let new_bitset = {
            let reg = self.labels.read();
            let mut b = 0u64;
            for l in &entry.labels {
                if let Some(bit) = reg.bit_for(l.as_str()) {
                    b |= 1u64 << bit;
                }
            }
            b
        };
        let old = self
            .writer
            .read_node(node_id.0 as u32)
            .ok()
            .flatten()
            .unwrap_or_default();
        let updated = NodeRecord {
            label_bitset: new_bitset,
            ..old
        };
        self.writer
            .update_node(node_id.0 as u32, updated)
            .expect("update_node failed");
        true
    }
}

// ---------------------------------------------------------------------------
// Tests — node-ops surface (step 1 verification subset).
//
// These tests mirror the LpgStore node-ops suite. Step 4 ports the full
// suite verbatim; step 1 establishes the behaviour class-by-class.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> (tempfile::TempDir, SubstrateStore) {
        let td = tempfile::tempdir().unwrap();
        let s = SubstrateStore::create(td.path().join("kb")).unwrap();
        (td, s)
    }

    #[test]
    fn create_node_returns_nonzero_increasing_ids() {
        let (_td, s) = store();
        let a = s.create_node(&["Person"]);
        let b = s.create_node(&["Person"]);
        let c = s.create_node(&["Company"]);
        assert_ne!(a.0, 0);
        assert_ne!(b.0, 0);
        assert_ne!(c.0, 0);
        assert!(a.0 < b.0 && b.0 < c.0);
    }

    #[test]
    fn create_node_no_labels() {
        let (_td, s) = store();
        let id = s.create_node(&[]);
        let n = s.get_node(id).unwrap();
        assert!(n.labels.is_empty());
        assert_eq!(n.id, id);
    }

    #[test]
    fn get_node_returns_labels_and_empty_props() {
        let (_td, s) = store();
        let id = s.create_node(&["Person", "Employee"]);
        let n = s.get_node(id).unwrap();
        assert_eq!(n.id, id);
        assert_eq!(n.labels.len(), 2);
        let names: Vec<&str> = n.labels.iter().map(|l| l.as_str()).collect();
        assert!(names.contains(&"Person") && names.contains(&"Employee"));
        assert!(n.properties.is_empty());
    }

    #[test]
    fn set_and_get_node_property() {
        let (_td, s) = store();
        let id = s.create_node(&["Person"]);
        s.set_node_property(id, "name", Value::from("Alix"));
        s.set_node_property(id, "age", Value::Int64(30));

        let name = s.get_node_property(id, &PropertyKey::new("name")).unwrap();
        let age = s.get_node_property(id, &PropertyKey::new("age")).unwrap();
        assert_eq!(name, Value::from("Alix"));
        assert_eq!(age, Value::Int64(30));

        // Full-object read carries all props.
        let n = s.get_node(id).unwrap();
        assert_eq!(n.properties.len(), 2);
    }

    #[test]
    fn set_overwrites_existing_property() {
        let (_td, s) = store();
        let id = s.create_node(&["Person"]);
        s.set_node_property(id, "age", Value::Int64(30));
        s.set_node_property(id, "age", Value::Int64(31));
        let age = s.get_node_property(id, &PropertyKey::new("age")).unwrap();
        assert_eq!(age, Value::Int64(31));
    }

    #[test]
    fn remove_node_property_returns_previous_value() {
        let (_td, s) = store();
        let id = s.create_node(&["Person"]);
        s.set_node_property(id, "age", Value::Int64(30));
        let prev = s.remove_node_property(id, "age").unwrap();
        assert_eq!(prev, Value::Int64(30));
        assert!(s.get_node_property(id, &PropertyKey::new("age")).is_none());
    }

    #[test]
    fn remove_missing_property_returns_none() {
        let (_td, s) = store();
        let id = s.create_node(&["Person"]);
        assert!(s.remove_node_property(id, "nope").is_none());
    }

    #[test]
    fn delete_node_is_idempotent_and_hides_it() {
        let (_td, s) = store();
        let id = s.create_node(&["Person"]);
        assert!(s.delete_node(id));
        assert!(s.get_node(id).is_none());
        // Second delete is a no-op.
        assert!(!s.delete_node(id));
    }

    #[test]
    fn set_property_on_deleted_node_is_ignored() {
        let (_td, s) = store();
        let id = s.create_node(&["Person"]);
        s.delete_node(id);
        // LpgStore silently no-ops; we match.
        s.set_node_property(id, "name", Value::from("ghost"));
        assert!(s.get_node_property(id, &PropertyKey::new("name")).is_none());
    }

    #[test]
    fn node_count_tracks_live_nodes() {
        let (_td, s) = store();
        assert_eq!(s.node_count(), 0);
        let a = s.create_node(&["A"]);
        let _b = s.create_node(&["B"]);
        let _c = s.create_node(&["C"]);
        assert_eq!(s.node_count(), 3);
        s.delete_node(a);
        assert_eq!(s.node_count(), 2);
    }

    #[test]
    fn node_ids_is_sorted_and_excludes_tombstones() {
        let (_td, s) = store();
        let a = s.create_node(&["A"]);
        let b = s.create_node(&["A"]);
        let c = s.create_node(&["A"]);
        s.delete_node(b);
        let ids = s.node_ids();
        assert_eq!(ids, vec![a, c]);
    }

    #[test]
    fn nodes_by_label_filters_correctly() {
        let (_td, s) = store();
        let alice = s.create_node(&["Person"]);
        let bob = s.create_node(&["Person", "Engineer"]);
        let acme = s.create_node(&["Company"]);

        let persons = s.nodes_by_label("Person");
        assert_eq!(persons.len(), 2);
        assert!(persons.contains(&alice) && persons.contains(&bob));
        assert!(!persons.contains(&acme));

        assert_eq!(s.nodes_by_label("Company"), vec![acme]);
        assert_eq!(s.nodes_by_label("Missing"), Vec::<NodeId>::new());
    }

    #[test]
    fn add_label_adds_only_new_labels() {
        let (_td, s) = store();
        let id = s.create_node(&["Person"]);
        assert!(!s.add_label(id, "Person")); // already there
        assert!(s.add_label(id, "Engineer"));
        let n = s.get_node(id).unwrap();
        assert_eq!(n.labels.len(), 2);
        // nodes_by_label now finds it under "Engineer" too.
        assert_eq!(s.nodes_by_label("Engineer"), vec![id]);
    }

    #[test]
    fn remove_label_returns_true_when_present() {
        let (_td, s) = store();
        let id = s.create_node(&["Person", "Engineer"]);
        assert!(s.remove_label(id, "Engineer"));
        let n = s.get_node(id).unwrap();
        assert_eq!(n.labels.len(), 1);
        assert!(!s.remove_label(id, "Engineer"));
    }

    #[test]
    fn find_nodes_by_property_returns_matches() {
        let (_td, s) = store();
        let alice = s.create_node(&["Person"]);
        let bob = s.create_node(&["Person"]);
        s.set_node_property(alice, "dept", Value::from("Eng"));
        s.set_node_property(bob, "dept", Value::from("HR"));

        let eng = s.find_nodes_by_property("dept", &Value::from("Eng"));
        assert_eq!(eng, vec![alice]);
        let hr = s.find_nodes_by_property("dept", &Value::from("HR"));
        assert_eq!(hr, vec![bob]);
    }

    #[test]
    fn find_nodes_by_properties_matches_all_conditions() {
        let (_td, s) = store();
        let alice = s.create_node(&["Person"]);
        let bob = s.create_node(&["Person"]);
        s.set_node_property(alice, "dept", Value::from("Eng"));
        s.set_node_property(alice, "active", Value::Bool(true));
        s.set_node_property(bob, "dept", Value::from("Eng"));
        s.set_node_property(bob, "active", Value::Bool(false));

        let hits = s.find_nodes_by_properties(&[
            ("dept", Value::from("Eng")),
            ("active", Value::Bool(true)),
        ]);
        assert_eq!(hits, vec![alice]);
    }

    #[test]
    fn find_nodes_in_range_uses_inclusive_bounds() {
        let (_td, s) = store();
        let a = s.create_node(&["N"]);
        let b = s.create_node(&["N"]);
        let c = s.create_node(&["N"]);
        s.set_node_property(a, "age", Value::Int64(10));
        s.set_node_property(b, "age", Value::Int64(20));
        s.set_node_property(c, "age", Value::Int64(30));

        let mid = s.find_nodes_in_range(
            "age",
            Some(&Value::Int64(10)),
            Some(&Value::Int64(25)),
            true,
            true,
        );
        assert!(mid.contains(&a));
        assert!(mid.contains(&b));
        assert!(!mid.contains(&c));
    }

    #[test]
    fn get_node_property_batch_returns_in_order() {
        let (_td, s) = store();
        let a = s.create_node(&["N"]);
        let b = s.create_node(&["N"]);
        let c = s.create_node(&["N"]);
        s.set_node_property(a, "age", Value::Int64(10));
        s.set_node_property(c, "age", Value::Int64(30));
        // b intentionally has no `age`.

        let key = PropertyKey::new("age");
        let res = s.get_node_property_batch(&[a, b, c], &key);
        assert_eq!(res.len(), 3);
        assert_eq!(res[0], Some(Value::Int64(10)));
        assert_eq!(res[1], None);
        assert_eq!(res[2], Some(Value::Int64(30)));
    }

    #[test]
    fn store_is_usable_as_dyn_graphstore() {
        let (_td, s) = store();
        let _id = s.create_node(&["Person"]);
        let as_trait: Arc<dyn GraphStore> = Arc::new(s);
        assert_eq!(as_trait.node_count(), 1);
    }

    // -----------------------------------------------------------------------
    // Step 2 — Edge-ops tests. These mirror the core LpgStore edge contract
    // (create/get/delete, chain traversal, properties, detach delete).
    // Step 4 ports the verbatim LpgStore suite.
    // -----------------------------------------------------------------------

    fn make_pair(s: &SubstrateStore) -> (NodeId, NodeId) {
        (s.create_node(&["Person"]), s.create_node(&["Company"]))
    }

    #[test]
    fn create_edge_returns_nonzero_id_and_roundtrips() {
        let (_td, s) = store();
        let (a, b) = make_pair(&s);
        let e = s.create_edge(a, b, "WORKS_AT");
        assert_ne!(e.0, 0);
        let edge = s.get_edge(e).unwrap();
        assert_eq!(edge.id, e);
        assert_eq!(edge.src, a);
        assert_eq!(edge.dst, b);
        assert_eq!(edge.edge_type.as_str(), "WORKS_AT");
        assert!(edge.properties.is_empty());
    }

    #[test]
    fn create_edge_increments_edge_count() {
        let (_td, s) = store();
        let (a, b) = make_pair(&s);
        assert_eq!(s.edge_count(), 0);
        let _ = s.create_edge(a, b, "R");
        let _ = s.create_edge(a, b, "R");
        let _ = s.create_edge(a, b, "R");
        assert_eq!(s.edge_count(), 3);
    }

    #[test]
    fn edges_from_outgoing_returns_all_targets() {
        let (_td, s) = store();
        let a = s.create_node(&["A"]);
        let b = s.create_node(&["B"]);
        let c = s.create_node(&["C"]);
        let d = s.create_node(&["D"]);
        let eab = s.create_edge(a, b, "R");
        let eac = s.create_edge(a, c, "R");
        let ead = s.create_edge(a, d, "R");
        let mut neigh: Vec<(NodeId, EdgeId)> = s.edges_from(a, Direction::Outgoing);
        neigh.sort_by_key(|p| p.0.0);
        let mut expected = vec![(b, eab), (c, eac), (d, ead)];
        expected.sort_by_key(|p| p.0.0);
        assert_eq!(neigh, expected);
    }

    #[test]
    fn edges_from_incoming_returns_all_sources() {
        let (_td, s) = store();
        let t = s.create_node(&["Target"]);
        let a = s.create_node(&["A"]);
        let b = s.create_node(&["B"]);
        let c = s.create_node(&["C"]);
        let eat = s.create_edge(a, t, "R");
        let ebt = s.create_edge(b, t, "R");
        let ect = s.create_edge(c, t, "R");
        let mut neigh = s.edges_from(t, Direction::Incoming);
        neigh.sort_by_key(|p| p.0.0);
        let mut expected = vec![(a, eat), (b, ebt), (c, ect)];
        expected.sort_by_key(|p| p.0.0);
        assert_eq!(neigh, expected);
    }

    #[test]
    fn edges_from_both_returns_union() {
        let (_td, s) = store();
        let a = s.create_node(&["A"]);
        let b = s.create_node(&["B"]);
        let c = s.create_node(&["C"]);
        let _ab = s.create_edge(a, b, "R");
        let _ca = s.create_edge(c, a, "R");
        let both = s.edges_from(a, Direction::Both);
        assert_eq!(both.len(), 2);
        let ids: Vec<NodeId> = both.iter().map(|p| p.0).collect();
        assert!(ids.contains(&b) && ids.contains(&c));
    }

    #[test]
    fn out_degree_counts_outgoing_only() {
        let (_td, s) = store();
        let a = s.create_node(&["A"]);
        let b = s.create_node(&["B"]);
        let c = s.create_node(&["C"]);
        let _ = s.create_edge(a, b, "R");
        let _ = s.create_edge(a, c, "R");
        let _ = s.create_edge(c, a, "R"); // incoming on a
        assert_eq!(s.out_degree(a), 2);
        assert_eq!(s.out_degree(b), 0);
        assert_eq!(s.in_degree(a), 1);
    }

    #[test]
    fn neighbors_is_edges_from_projected() {
        let (_td, s) = store();
        let a = s.create_node(&["A"]);
        let b = s.create_node(&["B"]);
        let c = s.create_node(&["C"]);
        let _ = s.create_edge(a, b, "R");
        let _ = s.create_edge(a, c, "R");
        let mut n = s.neighbors(a, Direction::Outgoing);
        n.sort_by_key(|nid| nid.0);
        let mut exp = vec![b, c];
        exp.sort_by_key(|nid| nid.0);
        assert_eq!(n, exp);
    }

    #[test]
    fn delete_edge_unlinks_from_outgoing_chain_head() {
        let (_td, s) = store();
        let a = s.create_node(&["A"]);
        let b = s.create_node(&["B"]);
        let c = s.create_node(&["C"]);
        let _e_ab = s.create_edge(a, b, "R");
        let e_ac = s.create_edge(a, c, "R");
        // Latest is head → delete it and assert the chain now has only ab.
        assert!(s.delete_edge(e_ac));
        let neigh = s.edges_from(a, Direction::Outgoing);
        assert_eq!(neigh.len(), 1);
        assert_eq!(neigh[0].0, b);
    }

    #[test]
    fn delete_edge_unlinks_from_outgoing_chain_middle() {
        let (_td, s) = store();
        let a = s.create_node(&["A"]);
        let b = s.create_node(&["B"]);
        let c = s.create_node(&["C"]);
        let d = s.create_node(&["D"]);
        let _e_ab = s.create_edge(a, b, "R"); // oldest → tail
        let e_ac = s.create_edge(a, c, "R"); // middle
        let _e_ad = s.create_edge(a, d, "R"); // newest → head
        assert!(s.delete_edge(e_ac));
        let mut n = s.edges_from(a, Direction::Outgoing);
        n.sort_by_key(|p| p.0.0);
        let mut exp = vec![b, d];
        exp.sort_by_key(|nid| nid.0);
        assert_eq!(n.into_iter().map(|p| p.0).collect::<Vec<_>>(), exp);
    }

    #[test]
    fn delete_edge_unlinks_from_incoming_chain() {
        let (_td, s) = store();
        let t = s.create_node(&["T"]);
        let a = s.create_node(&["A"]);
        let b = s.create_node(&["B"]);
        let e_at = s.create_edge(a, t, "R");
        let _e_bt = s.create_edge(b, t, "R");
        assert!(s.delete_edge(e_at));
        let incoming = s.edges_from(t, Direction::Incoming);
        assert_eq!(incoming.len(), 1);
        assert_eq!(incoming[0].0, b);
    }

    #[test]
    fn delete_edge_is_idempotent_and_missing_returns_false() {
        let (_td, s) = store();
        let (a, b) = make_pair(&s);
        let e = s.create_edge(a, b, "R");
        assert!(s.delete_edge(e));
        assert!(!s.delete_edge(e)); // second delete on same id
        assert!(!s.delete_edge(EdgeId(99999))); // never allocated
        assert!(s.get_edge(e).is_none());
    }

    #[test]
    fn set_and_get_edge_property() {
        let (_td, s) = store();
        let (a, b) = make_pair(&s);
        let e = s.create_edge(a, b, "WORKS_AT");
        s.set_edge_property(e, "since", Value::Int64(2020));
        s.set_edge_property(e, "role", Value::from("Engineer"));
        assert_eq!(
            s.get_edge_property(e, &PropertyKey::new("since")),
            Some(Value::Int64(2020))
        );
        assert_eq!(
            s.get_edge_property(e, &PropertyKey::new("role")),
            Some(Value::from("Engineer"))
        );
        let edge = s.get_edge(e).unwrap();
        assert_eq!(edge.properties.len(), 2);
    }

    #[test]
    fn remove_edge_property_returns_previous_value() {
        let (_td, s) = store();
        let (a, b) = make_pair(&s);
        let e = s.create_edge(a, b, "R");
        s.set_edge_property(e, "w", Value::Int64(42));
        assert_eq!(
            s.remove_edge_property(e, "w"),
            Some(Value::Int64(42))
        );
        assert!(s.get_edge_property(e, &PropertyKey::new("w")).is_none());
    }

    #[test]
    fn set_property_on_deleted_edge_is_ignored() {
        let (_td, s) = store();
        let (a, b) = make_pair(&s);
        let e = s.create_edge(a, b, "R");
        s.delete_edge(e);
        s.set_edge_property(e, "ghost", Value::Int64(1));
        assert!(s.get_edge_property(e, &PropertyKey::new("ghost")).is_none());
    }

    #[test]
    fn delete_node_edges_removes_both_directions() {
        let (_td, s) = store();
        let hub = s.create_node(&["Hub"]);
        let p1 = s.create_node(&["P"]);
        let p2 = s.create_node(&["P"]);
        let p3 = s.create_node(&["P"]);
        let _ = s.create_edge(hub, p1, "R");
        let _ = s.create_edge(hub, p2, "R");
        let _ = s.create_edge(p3, hub, "R");
        assert_eq!(s.out_degree(hub), 2);
        assert_eq!(s.in_degree(hub), 1);
        s.delete_node_edges(hub);
        assert_eq!(s.out_degree(hub), 0);
        assert_eq!(s.in_degree(hub), 0);
        // The other endpoints lose the edge from their side too.
        assert_eq!(s.in_degree(p1), 0);
        assert_eq!(s.in_degree(p2), 0);
        assert_eq!(s.out_degree(p3), 0);
    }

    #[test]
    fn detach_delete_handles_self_loop() {
        // Self-loop appears in both outgoing and incoming chains. Ensure
        // delete_node_edges deduplicates and does not double-free.
        let (_td, s) = store();
        let n = s.create_node(&["N"]);
        let _ = s.create_edge(n, n, "LOOP");
        assert_eq!(s.out_degree(n), 1);
        assert_eq!(s.in_degree(n), 1);
        s.delete_node_edges(n);
        assert_eq!(s.out_degree(n), 0);
        assert_eq!(s.in_degree(n), 0);
        assert_eq!(s.edge_count(), 0);
    }

    #[test]
    fn edge_type_is_interned_but_observable() {
        let (_td, s) = store();
        let (a, b) = make_pair(&s);
        let e = s.create_edge(a, b, "FOLLOWS");
        assert_eq!(s.edge_type(e).unwrap().as_str(), "FOLLOWS");
    }

    #[test]
    fn multiple_edges_between_same_pair_are_independent() {
        let (_td, s) = store();
        let (a, b) = make_pair(&s);
        let e1 = s.create_edge(a, b, "R");
        let e2 = s.create_edge(a, b, "R");
        assert_ne!(e1, e2);
        s.set_edge_property(e1, "w", Value::Int64(1));
        s.set_edge_property(e2, "w", Value::Int64(2));
        assert_eq!(
            s.get_edge_property(e1, &PropertyKey::new("w")),
            Some(Value::Int64(1))
        );
        assert_eq!(
            s.get_edge_property(e2, &PropertyKey::new("w")),
            Some(Value::Int64(2))
        );
        assert_eq!(s.out_degree(a), 2);
    }

    #[test]
    fn batch_create_edges_preserves_input_order() {
        let (_td, s) = store();
        let a = s.create_node(&["A"]);
        let b = s.create_node(&["B"]);
        let c = s.create_node(&["C"]);
        let ids = s.batch_create_edges(&[(a, b, "R"), (b, c, "R"), (a, c, "R")]);
        assert_eq!(ids.len(), 3);
        let e0 = s.get_edge(ids[0]).unwrap();
        let e1 = s.get_edge(ids[1]).unwrap();
        let e2 = s.get_edge(ids[2]).unwrap();
        assert_eq!((e0.src, e0.dst), (a, b));
        assert_eq!((e1.src, e1.dst), (b, c));
        assert_eq!((e2.src, e2.dst), (a, c));
    }

    #[test]
    fn get_edges_properties_selective_batch_returns_in_order() {
        let (_td, s) = store();
        let (a, b) = make_pair(&s);
        let e0 = s.create_edge(a, b, "R");
        let e1 = s.create_edge(a, b, "R");
        let e2 = s.create_edge(a, b, "R");
        s.set_edge_property(e0, "w", Value::Int64(10));
        s.set_edge_property(e2, "w", Value::Int64(30));
        // e1 intentionally has no `w`.
        let res = s.get_edges_properties_selective_batch(
            &[e0, e1, e2],
            &[PropertyKey::new("w")],
        );
        assert_eq!(res.len(), 3);
        assert_eq!(res[0].get(&PropertyKey::new("w")), Some(&Value::Int64(10)));
        assert!(res[1].is_empty());
        assert_eq!(res[2].get(&PropertyKey::new("w")), Some(&Value::Int64(30)));
    }

    // ----- step 3 : close/reopen roundtrip (dict persistence) --------------

    #[test]
    fn reopen_preserves_label_ids() {
        // Same label created in two sessions must land on the same bit
        // index, so existing NodeRecord.label_bitset values remain valid.
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("kb");

        let a_bit = {
            let s = SubstrateStore::create(&path).unwrap();
            // Interleave to force a deterministic bit assignment.
            let a = s.create_node(&["A"]);
            let _b = s.create_node(&["B"]);
            let _c = s.create_node(&["C"]);
            s.flush().unwrap();
            // Grab the raw bitset to check later.
            s.writer.read_node(a.0 as u32).unwrap().unwrap().label_bitset
        };

        let s2 = SubstrateStore::open(&path).unwrap();
        // Create a new node with the same label — must reuse the bit.
        let a2 = s2.create_node(&["A"]);
        let new_bitset =
            s2.writer.read_node(a2.0 as u32).unwrap().unwrap().label_bitset;
        assert_eq!(
            a_bit, new_bitset,
            "label 'A' must occupy the same bit across sessions"
        );

        // And a previously-persisted node retains its labels on reopen.
        let node_a1 = s2.get_node(NodeId(1)).unwrap();
        assert!(node_a1.labels.iter().any(|l| l.as_str() == "A"));
    }

    #[test]
    fn reopen_preserves_edge_type_ids() {
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("kb");

        let (a_id, b_id, knows_type_id) = {
            let s = SubstrateStore::create(&path).unwrap();
            let a = s.create_node(&["P"]);
            let b = s.create_node(&["P"]);
            let _works_at = s.create_edge(a, b, "WORKS_AT");
            let knows = s.create_edge(a, b, "KNOWS");
            s.flush().unwrap();
            let rec = s.writer.read_edge(knows.0).unwrap().unwrap();
            (a, b, rec.edge_type)
        };

        let s2 = SubstrateStore::open(&path).unwrap();
        // Interning the same name in the reopened store must return
        // exactly the same u16 id.
        let reinterned = s2.intern_edge_type("KNOWS").unwrap();
        assert_eq!(
            reinterned, knows_type_id,
            "edge-type 'KNOWS' must re-intern to the same u16 id"
        );
        // And a previously-persisted edge exposes its type name via
        // get_edge after reopen.
        let edges: Vec<(NodeId, EdgeId)> =
            s2.edges_from(a_id, Direction::Outgoing);
        assert_eq!(edges.len(), 2);
        let types: Vec<String> = edges
            .iter()
            .map(|(_, id)| s2.get_edge(*id).unwrap().edge_type.to_string())
            .collect();
        assert!(types.iter().any(|t| t == "WORKS_AT"));
        assert!(types.iter().any(|t| t == "KNOWS"));
        // Silence unused binding warning — b is the shared destination.
        let _ = b_id;
    }

    #[test]
    fn reopen_preserves_slot_allocator() {
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("kb");

        let (node_hw, edge_hw) = {
            let s = SubstrateStore::create(&path).unwrap();
            let a = s.create_node(&["P"]);
            let b = s.create_node(&["P"]);
            let _ = s.create_edge(a, b, "R");
            let _ = s.create_edge(b, a, "R");
            s.flush().unwrap();
            (
                s.next_node_id.load(Ordering::Acquire),
                s.next_edge_id.load(Ordering::Acquire),
            )
        };

        let s2 = SubstrateStore::open(&path).unwrap();
        assert_eq!(
            s2.next_node_id.load(Ordering::Acquire),
            node_hw,
            "next_node_id must round-trip via substrate.dict"
        );
        assert_eq!(
            s2.next_edge_id.load(Ordering::Acquire),
            edge_hw,
            "next_edge_id must round-trip via substrate.dict"
        );

        // A subsequent create must not collide with any previously
        // allocated slot.
        let c = s2.create_node(&["P"]);
        assert_eq!(c.0 as u32, node_hw, "new slot continues past high-water");
    }

    #[test]
    fn reopen_rebuilds_nodes_and_edges_views() {
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("kb");

        let (a, b, c, e_ab, e_bc) = {
            let s = SubstrateStore::create(&path).unwrap();
            let a = s.create_node(&["P"]);
            let b = s.create_node(&["P", "M"]);
            let c = s.create_node(&["Q"]);
            let e_ab = s.create_edge(a, b, "R1");
            let e_bc = s.create_edge(b, c, "R2");
            s.flush().unwrap();
            (a, b, c, e_ab, e_bc)
        };

        let s2 = SubstrateStore::open(&path).unwrap();

        // Nodes: labels must be recovered from NodeRecord.label_bitset
        // via the persisted LabelRegistry.
        let n_a = s2.get_node(a).unwrap();
        assert_eq!(n_a.labels.len(), 1);
        assert_eq!(n_a.labels[0].as_str(), "P");

        let n_b = s2.get_node(b).unwrap();
        assert_eq!(n_b.labels.len(), 2);
        let b_names: Vec<&str> =
            n_b.labels.iter().map(|l| l.as_str()).collect();
        assert!(b_names.contains(&"P") && b_names.contains(&"M"));

        // Edges: edge_type must be recovered via the persisted
        // EdgeTypeRegistry.
        let e = s2.get_edge(e_ab).unwrap();
        assert_eq!(e.edge_type.as_str(), "R1");
        let e = s2.get_edge(e_bc).unwrap();
        assert_eq!(e.edge_type.as_str(), "R2");

        // Outgoing chains work because src's first_edge_off is durable
        // on NodeRecord.
        let out_a = s2.edges_from(a, Direction::Outgoing);
        assert_eq!(out_a.len(), 1);
        assert_eq!(out_a[0].0, b); // peer = dst for Outgoing
        assert_eq!(out_a[0].1, e_ab);

        // Incoming chains work because incoming_heads was rebuilt from
        // the Edges-zone scan.
        let in_c = s2.edges_from(c, Direction::Incoming);
        assert_eq!(in_c.len(), 1);
        assert_eq!(in_c[0].0, b); // peer = src for Incoming
        assert_eq!(in_c[0].1, e_bc);

        let in_b = s2.edges_from(b, Direction::Incoming);
        assert_eq!(in_b.len(), 1);
        assert_eq!(in_b[0].0, a);
        assert_eq!(in_b[0].1, e_ab);
    }

    #[test]
    fn reopen_skips_tombstoned_slots() {
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("kb");

        let (a, b, e_live) = {
            let s = SubstrateStore::create(&path).unwrap();
            let a = s.create_node(&["P"]);
            let b = s.create_node(&["P"]);
            let zombie = s.create_node(&["P"]);
            let e_dead = s.create_edge(a, b, "R");
            let e_live = s.create_edge(a, b, "R");
            // Tombstone a node and an edge before flush.
            assert!(s.delete_node(zombie));
            assert!(s.delete_edge(e_dead));
            s.flush().unwrap();
            (a, b, e_live)
        };

        let s2 = SubstrateStore::open(&path).unwrap();
        // Tombstoned node must not appear in any view.
        assert!(s2.get_node(NodeId(3)).is_none());
        assert_eq!(s2.node_count(), 2);
        // Tombstoned edge must be gone; the live one remains.
        assert!(s2.get_edge(EdgeId(1)).is_none());
        assert!(s2.get_edge(e_live).is_some());
        let out_a = s2.edges_from(a, Direction::Outgoing);
        assert_eq!(out_a.len(), 1);
        assert_eq!(out_a[0].1, e_live);
        assert_eq!(out_a[0].0, b);
    }

    #[test]
    fn reopen_preserves_incoming_head_order() {
        // Incoming-heads rebuild uses "max live EdgeId per dst" — verify
        // that matches the splice-at-head invariant (newest at front,
        // oldest at back).
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("kb");

        let (a, b, c, e1, e2, e3) = {
            let s = SubstrateStore::create(&path).unwrap();
            let a = s.create_node(&["P"]);
            let b = s.create_node(&["P"]);
            let c = s.create_node(&["Q"]); // common dst
            let e1 = s.create_edge(a, c, "R");
            let e2 = s.create_edge(b, c, "R");
            let e3 = s.create_edge(a, c, "R");
            s.flush().unwrap();
            (a, b, c, e1, e2, e3)
        };

        let s2 = SubstrateStore::open(&path).unwrap();
        let in_c: Vec<EdgeId> = s2
            .edges_from(c, Direction::Incoming)
            .into_iter()
            .map(|(_, id)| id)
            .collect();
        assert_eq!(in_c.len(), 3);
        // Exact order depends on next_to linking; the invariant we check
        // is that all three show up and the highest id is the head.
        assert_eq!(in_c[0], e3, "head of incoming chain is newest edge");
        let set: std::collections::BTreeSet<EdgeId> =
            in_c.iter().copied().collect();
        assert!(set.contains(&e1) && set.contains(&e2) && set.contains(&e3));
        // Silence unused bindings: a, b serve only to make the edges.
        let _ = (a, b);
    }

    #[test]
    fn dict_is_written_on_flush() {
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("kb");
        let s = SubstrateStore::create(&path).unwrap();
        let _ = s.create_node(&["X"]);
        assert!(
            !path.join(DICT_FILENAME).exists(),
            "dict must not appear until flush"
        );
        s.flush().unwrap();
        assert!(
            path.join(DICT_FILENAME).exists(),
            "flush must atomically rewrite substrate.dict"
        );

        // Reload the snapshot directly and confirm its contents.
        let snap =
            crate::dict::DictSnapshot::load(&path.join(DICT_FILENAME)).unwrap();
        assert_eq!(snap.labels, vec!["X".to_string()]);
        assert!(snap.edge_types.is_empty());
        assert!(snap.prop_keys.is_empty());
        assert_eq!(snap.next_node_id, 2);
        assert_eq!(snap.next_edge_id, 1);
    }
}
