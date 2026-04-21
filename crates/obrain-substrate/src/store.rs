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
use std::sync::atomic::{AtomicU16, AtomicU32, AtomicU64, Ordering};

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
use crate::meta::meta_flags;
use crate::record::{
    EdgeRecord, NodeRecord, NODES_PER_PAGE, PackedScarUtilAff, U48, edge_flags, f32_to_q1_15,
};
use crate::props_snapshot::{
    entries_to_map, map_to_entries, PropEntry, PropertiesSnapshotV1,
    PropertiesStreamingWriter, PROPS_FILENAME,
};
use crate::vec_column::EntityKind;
use crate::vec_column_registry::VecColumnRegistry;
use crate::wal_io::SyncMode;
use crate::writer::Writer;

/// Dict side-car filename relative to the substrate directory.
const DICT_FILENAME: &str = "substrate.dict";

/// Property keys that are NOT rehydrated into the in-memory `DashMap`
/// during [`SubstrateStore::load_properties`].
///
/// Rationale (T16 / T17 staging): these properties already have a
/// mmappable on-disk representation elsewhere and exist in
/// `substrate.props` only as a legacy artefact of the migration
/// pipeline. Loading them into the heap-backed `DashMap` inflates
/// anonymous RSS by a factor of 5-8× on large corpora (wiki: 7.19 GiB
/// props → 54.85 GiB anon). Since the runtime accessors for these keys
/// are the typed columns (tiers / future dedicated zones), nothing
/// observable changes for legitimate callers — only migration-time
/// QA tools (`obrain-migrate`, `rebuild_tiers`, `inspect`) still read
/// them via `get_node_property`, and they open the store once and
/// exit, so they can afford a slower path if needed later.
///
/// Extend this list cautiously: every entry here is an implicit
/// assertion that no runtime code in the tree reads that key via
/// `get_node_property` / `get_properties`. Audit with
/// `rg 'get_node_property.*"<key>"' | rg -v 'examples/|tests/|migrate|neo4j2obrain'`
/// before adding.
///
/// **T16.7** — extended to `_kernel_embedding` (80-dim Φ₀ projection) and
/// `_hilbert_features` (64-dim topology-derived signature). Both are
/// derived by the `kernel_manager` warden from the graph itself, not
/// persistent user data — loading them inflates anon RSS for nothing.
/// `obrain-migrate` (T16.6) already drops them at migration time, so
/// only legacy bases still carry them; this list handles that case too.
///
/// Audit trail for each entry:
/// - `_st_embedding` (384-dim SentenceTransformer): tier zones are the
///   runtime accessor; the copy in props is a migration artefact.
///   Added: T16 anon-RSS post-mortem.
/// - `_kernel_embedding` (80-dim Φ₀): no runtime caller outside
///   `obrain-migrate`; kernel warden regenerates it in-memory on demand.
///   Added: T16.7 (runtime filter sync with T16.6 migrate filter).
/// - `_hilbert_features` (64-dim topology-derived): same story —
///   recomputed by the hilbert warden; never read by the hub runtime.
///   Added: T16.7.
pub const SKIP_ON_LOAD_PROP_KEYS: &[&str] =
    &["_st_embedding", "_kernel_embedding", "_hilbert_features"];

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
    ///
    /// Wired by T16.7: every `Value::Vector` write routes through
    /// [`VecColumnRegistry`], which needs a stable `prop_key_id` to
    /// name the vec-column zone file. Scalar properties still don't
    /// touch this registry — they live in the DashMap and are
    /// persisted via the props sidecar.
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
    /// Next engram id to allocate. Id 0 is the null-engram sentinel
    /// (see [`crate::engram::MAX_ENGRAM_ID`]); allocation starts at 1.
    /// Persisted via `substrate.dict` v2.
    next_engram_id: AtomicU16,
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
    /// Per-community slot allocator state (T11 Step 3).
    ///
    /// Maps `community_id → last_slot_allocated_for_this_community`. The
    /// allocator uses this to decide:
    ///
    /// * **Fast path** — if the community's last slot is the global tail
    ///   AND lies in the same 4 KiB page as the next allocation would,
    ///   extend in place (`next_node_id += 1`). Preserves community
    ///   contiguity at zero cost.
    /// * **Slow path** — otherwise, round `next_node_id` up to the next
    ///   page boundary and open a fresh page for this community. Any
    ///   trailing slots in the previous page are skipped (bounded waste
    ///   ≤ NODES_PER_PAGE - 1 slots per community transition).
    ///
    /// Rebuilt on open by scanning the Nodes zone (one entry per live
    /// `community_id` observed, pointing at its highest allocated slot).
    community_placements: DashMap<u32, u32>,
    /// Per-community **first** allocated slot (T11 Step 5 — prefetch
    /// companion to `community_placements`).
    ///
    /// Maps `community_id → first_slot_ever_allocated_for_this_community`
    /// that is still live in the current generation of the zone. Together
    /// with `community_placements` (last slot), this pair forms the
    /// bounding slot range used by [`Self::prefetch_community`] to issue
    /// `madvise(WILLNEED)` over the whole community's page range.
    ///
    /// * **Rebuild on open** — populated by the ascending-slot scan in
    ///   [`Self::rebuild_from_zones`] via `entry(cid).or_insert(slot)`.
    ///   The first live slot observed for a given cid is the oldest page
    ///   that still holds a node of that community.
    /// * **Online allocation** — `allocate_node_id_in_community` sets the
    ///   entry on the slow path for a fresh community (entry was
    ///   `u32::MAX`). Subsequent slow-path allocations (opening a new
    ///   page for an existing community) leave it untouched: the
    ///   community's earliest pages still live on disk, so the bounding
    ///   range starts there.
    /// * **Post-compaction** — [`Self::refresh_community_ranges`] is
    ///   called by the CommunityWarden after a `CompactCommunity` cycle
    ///   relocates nodes to a new page. This rewrites both maps from the
    ///   current Nodes zone so the prefetch range tracks the new layout.
    ///
    /// The range `[first_slot, last_slot]` is a **superset** of the
    /// community's actual pages (other communities may have slots
    /// interleaved mid-range on pre-compaction data). That over-advise is
    /// bounded — post-compaction the range is tight, pre-compaction it
    /// is the same bound that `CommunityFragmentation::fragmentation`
    /// reports. `madvise(WILLNEED)` is a kernel hint, so a loose bound
    /// wastes no correctness; it just dilutes the page-fault savings.
    community_first_slots: DashMap<u32, u32>,
    /// Routing table + open-writer cache for `Value::Vector` properties.
    ///
    /// Every `set_node_property(_, _, Value::Vector(_))` and
    /// `set_edge_property(_, _, Value::Vector(_))` call is intercepted
    /// here and routed to a dense mmap'd `substrate.veccol.*` zone
    /// instead of the per-entity `PropertyMap` + bincode `substrate.props`
    /// sidecar. This closes the T16 anon-RSS gate (≤ 1 GiB) — vector
    /// payloads never touch the anonymous heap.
    ///
    /// Hydrated on open from `DictSnapshot.vec_columns` (v3) so a
    /// fresh `get_node_property` resolves the right zone even before
    /// the first write in this session. Durability is batched to
    /// [`Self::flush`] via `sync_all()` — same cadence as the dict and
    /// the props sidecar.
    vec_columns: VecColumnRegistry,
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
            next_engram_id: AtomicU16::new(snapshot.next_engram_id),
            nodes: DashMap::new(),
            edges: DashMap::new(),
            incoming_heads: DashMap::new(),
            labels: RwLock::new(labels),
            edge_types: RwLock::new(edge_types),
            prop_keys: RwLock::new(prop_keys),
            stats: Arc::new(Statistics::new()),
            community_placements: DashMap::new(),
            community_first_slots: DashMap::new(),
            vec_columns: VecColumnRegistry::new(),
        };

        // (2) Rebuild in-memory side-cars from the on-disk zones. This is
        //     an O(N + E) scan bounded by the just-loaded high-water marks
        //     — zero-filled slots past the marks are never read.
        store.rebuild_from_zones()?;

        // (3) Load properties from the sidecar snapshot, if present.
        //     Fresh stores have no `.props` file and this is a no-op.
        //     A corrupt snapshot logs a warning and degrades to empty
        //     properties rather than refusing to open — the store is
        //     still usable structurally.
        store.load_properties()?;

        // (4) Hydrate the vec-column registry from the persisted specs
        //     (dict v3). For each `(prop_key_id, entity_kind, dim, dtype)`
        //     tuple we reopen the corresponding `substrate.veccol.*`
        //     zone so subsequent reads and writes in this session route
        //     to the right file without a cold `ZoneFile::open`.
        //
        //     Hydration runs after `prop_keys` is loaded so the registry
        //     can resolve `prop_key_id → PropertyKey` names.
        {
            let sub = store.substrate.lock();
            let names = store.prop_keys.read().names();
            store
                .vec_columns
                .hydrate_from_dict(&sub, &names, &snapshot.vec_columns)?;
        }

        Ok(store)
    }

    /// Load the properties sidecar and populate `self.nodes[*].properties`
    /// and `self.edges[*].properties` maps. See
    /// [`Self::persist_properties`] for the symmetric write path.
    fn load_properties(&self) -> SubstrateResult<()> {
        let path = {
            let sub = self.substrate.lock();
            sub.path().join(PROPS_FILENAME)
        };
        let snap = PropertiesSnapshotV1::load(&path)?;
        let mut nodes_loaded = 0usize;
        let mut edges_loaded = 0usize;
        let mut node_props_skipped = 0usize;
        let mut edge_props_skipped = 0usize;
        // Hot loop: a small contains check over <10 short strings. A linear
        // scan is faster than any hash-set overhead at this size.
        let skip = |k: &str| SKIP_ON_LOAD_PROP_KEYS.iter().any(|s| *s == k);
        for e in snap.nodes {
            let nid = obrain_common::types::NodeId(e.id);
            let before = e.props.len();
            let filtered: Vec<_> = e.props.into_iter().filter(|(k, _)| !skip(k)).collect();
            node_props_skipped += before - filtered.len();
            let map = entries_to_map(filtered);
            // Merge into the DashMap entry built by rebuild_from_zones.
            // If the node was tombstoned between the last flush and the
            // reopen it will not be present here; we silently skip.
            if let Some(mut in_mem) = self.nodes.get_mut(&nid) {
                in_mem.properties = map;
                nodes_loaded += 1;
            }
        }
        for e in snap.edges {
            let eid = obrain_common::types::EdgeId(e.id);
            let before = e.props.len();
            let filtered: Vec<_> = e.props.into_iter().filter(|(k, _)| !skip(k)).collect();
            edge_props_skipped += before - filtered.len();
            let map = entries_to_map(filtered);
            if let Some(mut in_mem) = self.edges.get_mut(&eid) {
                in_mem.properties = map;
                edges_loaded += 1;
            }
        }
        if nodes_loaded > 0 || edges_loaded > 0 {
            tracing::info!(
                "props snapshot: loaded {} node-property maps, {} edge-property maps \
                 (skipped-on-load: {} node-props, {} edge-props — see SKIP_ON_LOAD_PROP_KEYS)",
                nodes_loaded,
                edges_loaded,
                node_props_skipped,
                edge_props_skipped,
            );
        }
        Ok(())
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
        // ---- Nodes: rebuild labels view + community_placements ----
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
            // T11 Step 3: community_placements = max live slot per cid.
            // The scan order is ascending so a simple overwrite trails the
            // highest-slot-wins invariant without an explicit compare.
            self.community_placements.insert(rec.community_id, slot);
            // T11 Step 5: community_first_slots = min live slot per cid.
            // `entry.or_insert` keeps the first observation — which is the
            // lowest slot in an ascending scan.
            self.community_first_slots
                .entry(rec.community_id)
                .or_insert(slot);
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
    ///
    /// `vec_columns` is populated from the live [`VecColumnRegistry`]
    /// so every open zone has a persisted spec entry — on next open,
    /// `hydrate_from_dict` can reopen each zone by filename without
    /// relying on a directory scan.
    fn build_dict_snapshot(&self) -> crate::dict::DictSnapshot {
        crate::dict::DictSnapshot {
            labels: self.labels.read().names(),
            edge_types: self.edge_types.read().names(),
            prop_keys: self.prop_keys.read().names(),
            next_node_id: self.next_node_id.load(Ordering::Acquire) as u64,
            next_edge_id: self.next_edge_id.load(Ordering::Acquire),
            next_engram_id: self.next_engram_id.load(Ordering::Acquire),
            vec_columns: self.vec_columns.specs_snapshot(),
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
    /// 3. `vec_columns.sync_all()` — for each open `substrate.veccol.*`
    ///    zone, recompute the CRC, overwrite the header, and
    ///    `msync + fsync`. Done **before** the dict is persisted so
    ///    the dict can't reference a zone whose header is stale from
    ///    a previous session (CRC mismatch on next open would silently
    ///    demote the column to "missing" and lose data).
    /// 4. `persist_dict()` — atomically rewrites `substrate.dict` with
    ///    the current registries, slot counters, and vec-column specs.
    ///    Done last so any crash before here leaves the WAL as source
    ///    of truth (replay will reconstruct both).
    pub fn flush(&self) -> SubstrateResult<()> {
        self.writer.commit()?;
        self.writer.msync_zones()?;
        self.vec_columns.sync_all()?;
        self.persist_dict()?;
        self.persist_properties()?;
        Ok(())
    }

    /// Open a [`PropertiesStreamingWriter`] targeting this store's
    /// `substrate.props` sidecar.
    ///
    /// This is the DashMap-bypass path used by `obrain-migrate` on
    /// bulk runs (megalaw, wikipedia…) where routing every property
    /// through `set_node_property` would blow the DashMap past
    /// available RAM before the first `flush()`. See
    /// [`crate::props_snapshot`] for the coexistence contract with
    /// `flush()` / `persist_properties()`.
    ///
    /// Runtime code MUST NOT use this — there's no synchronisation
    /// with the in-memory property maps, so reads through
    /// `get_node_property` won't see the streamed values until the
    /// store is reopened.
    pub fn open_streaming_props_writer(
        &self,
    ) -> SubstrateResult<PropertiesStreamingWriter> {
        let path = {
            let sub = self.substrate.lock();
            sub.path().join(PROPS_FILENAME)
        };
        PropertiesStreamingWriter::open(path)
    }

    /// Persist the in-memory `self.nodes[*].properties` and
    /// `self.edges[*].properties` maps to a sidecar `substrate.props`
    /// file. See [`crate::props_snapshot`] for rationale and format.
    ///
    /// This is a **full rewrite** every call — there is no delta
    /// encoding. Callers that churn properties should prefer larger
    /// flush intervals. The T17 property-pages subsystem will replace
    /// this with O(Δ) mmap writes.
    fn persist_properties(&self) -> SubstrateResult<()> {
        let mut snap = PropertiesSnapshotV1::default();
        // Defensive filter: `Value::Vector` writes are routed to
        // `vec_columns` and never inserted into the DashMap by
        // `set_node_property`, but older bases migrated before T16.7
        // may still carry vector entries loaded via non-trait paths.
        // Dropping them here prevents re-persistence into the bincode
        // sidecar (which is the whole point of T16.7).
        let is_vec = |(_, v): &(String, Value)| matches!(v, Value::Vector(_));
        // Nodes
        snap.nodes.reserve(self.nodes.len());
        for entry in self.nodes.iter() {
            // Skip entries with no properties — they're just label
            // views, no need to persist.
            if entry.value().properties.is_empty() {
                continue;
            }
            let props: Vec<(String, Value)> = map_to_entries(&entry.value().properties)
                .into_iter()
                .filter(|p| !is_vec(p))
                .collect();
            if props.is_empty() {
                continue;
            }
            snap.nodes.push(PropEntry {
                id: entry.key().as_u64(),
                props,
            });
        }
        // Edges
        snap.edges.reserve(self.edges.len());
        for entry in self.edges.iter() {
            if entry.value().properties.is_empty() {
                continue;
            }
            let props: Vec<(String, Value)> = map_to_entries(&entry.value().properties)
                .into_iter()
                .filter(|p| !is_vec(p))
                .collect();
            if props.is_empty() {
                continue;
            }
            snap.edges.push(PropEntry {
                id: entry.key().as_u64(),
                props,
            });
        }
        let path = {
            let sub = self.substrate.lock();
            sub.path().join(PROPS_FILENAME)
        };
        snap.persist(&path)
    }

    /// Number of node slots handed out so far (including tombstoned).
    pub fn slot_high_water(&self) -> u32 {
        self.next_node_id.load(Ordering::Acquire)
    }

    /// Access the underlying [`Writer`] (used by tests and CommunityWarden
    /// for direct slot-level reads).
    pub fn writer(&self) -> &Writer {
        &self.writer
    }

    /// Load the on-disk tier index (L0 / L1 / L2) from `substrate.tier0/1/2`
    /// zones, if present and valid.
    ///
    /// Returns:
    /// * `Ok(Some(index))` — all three zones present, CRC-valid, consistent
    ///   `n_slots`. Ready for `search_topk`.
    /// * `Ok(None)` — any zone missing, corrupted, wrong version, or slot
    ///   counts disagree. Caller should fall back to a rebuild (e.g.
    ///   `SubstrateTieredIndex::rebuild` from `_st_embedding` properties).
    /// * `Err(_)` — only on I/O failure that prevented even opening a zone.
    ///
    /// Corruption of a tier zone is **never** fatal here; the fallback-to-
    /// rebuild path lets the store stay open for callers that don't need
    /// retrieval (geometric activation, tooling, tests).
    ///
    /// The `dim` argument is the tier2 / original embedding dimension
    /// (currently always `L2_DIM = 384`). It feeds `Tier0Builder` /
    /// `Tier1Builder` with the seed so subsequent inserts use the same
    /// projection as the persisted zones.
    ///
    /// See `docs/rfc/substrate/tier-persistence.md` for the on-disk format.
    pub fn load_tier_index(
        &self,
        dim: usize,
    ) -> SubstrateResult<Option<crate::retrieval::SubstrateTieredIndex>> {
        let sub = self.substrate.lock();
        crate::retrieval::SubstrateTieredIndex::load_from_zones(&sub, dim)
    }

    /// Create a node whose `community_id` is known at insert time
    /// (T11 Step 3 — online community-local allocation).
    ///
    /// The slot is chosen by [`Self::allocate_node_id_in_community`] so
    /// successive inserts into the same community land in the same 4 KiB
    /// page whenever possible, preserving community contiguity without a
    /// full bulk-sort pass.
    ///
    /// Contract vs `create_node(labels)` (trait method):
    ///
    /// | aspect                  | `create_node`            | `create_node_in_community` |
    /// |-------------------------|--------------------------|----------------------------|
    /// | slot policy             | tail-append              | community-local page       |
    /// | NodeRecord.community_id | 0 (uncategorized)        | caller-supplied            |
    /// | HILBERT_SORTED flag     | untouched                | cleared                    |
    /// | page alignment on 1st   | —                        | rounds up to page boundary |
    ///
    /// Returns the fresh [`NodeId`] (always non-zero; slot 0 is reserved).
    pub fn create_node_in_community(
        &self,
        labels: &[&str],
        community_id: u32,
    ) -> NodeId {
        let id = self.allocate_node_id_in_community(community_id);
        let bitset = self
            .intern_labels_to_bitset(labels)
            .expect("label registry overflow (>64 labels); step 3 lifts this");
        let rec = NodeRecord {
            label_bitset: bitset,
            community_id,
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

    /// Look up the most recently allocated slot for `community_id` (or
    /// `None` if that community has never been used in this store).
    /// Exposed for tests and for the CommunityWarden (Step 4).
    pub fn last_slot_for_community(&self, community_id: u32) -> Option<u32> {
        self.community_placements
            .get(&community_id)
            .map(|r| *r.value())
            .filter(|s| *s != u32::MAX)
    }

    /// Look up the earliest live slot for `community_id` (or `None` if
    /// that community has never been used in this store).
    ///
    /// Companion to [`Self::last_slot_for_community`] — the pair
    /// `(first, last)` is the bounding slot range used by the prefetch
    /// hook (T11 Step 5). See the docs on `community_first_slots` for
    /// how this is maintained across online allocation and compaction.
    pub fn first_slot_for_community(&self, community_id: u32) -> Option<u32> {
        self.community_first_slots
            .get(&community_id)
            .map(|r| *r.value())
            .filter(|s| *s != u32::MAX)
    }

    /// Bounding slot range `[first, last]` (inclusive) for a community,
    /// or `None` if the community has no live nodes. Both bounds are
    /// live slots; the range may contain slots that belong to other
    /// communities (pre-compaction) — the CommunityWarden tightens it by
    /// running compaction.
    pub fn community_slot_range(&self, community_id: u32) -> Option<(u32, u32)> {
        let first = self.first_slot_for_community(community_id)?;
        let last = self.last_slot_for_community(community_id)?;
        if first > last {
            // Out-of-order bounds mean the maps drifted — treat as empty
            // and let the next rebuild/refresh repair. Logging kept
            // cheap: this is a hot path for prefetch.
            tracing::debug!(
                target: "substrate.prefetch",
                "community {community_id}: inverted range first={first} > last={last} — \
                 skipping prefetch; call refresh_community_ranges() after compaction"
            );
            return None;
        }
        Some((first, last))
    }

    /// Issue `madvise(WILLNEED)` over the bounding page range of
    /// `community_id` on the **Nodes**, **Community**, and **Hilbert**
    /// zones (T11 Step 5).
    ///
    /// The hook is designed to be called on activation points where a
    /// node of `community_id` is about to be traversed — e.g. a
    /// retrieval hit, a spreading-activation seed, or a user message
    /// citing a node. By the time the hot path reads the surrounding
    /// community pages, the kernel has already scheduled readahead for
    /// them, shaving most of the cold-cache page-fault latency.
    ///
    /// Returns `Ok(())` regardless of whether advise was actually issued
    /// (zones may be unmapped, the range may be empty) — the hook is
    /// best-effort by contract.
    ///
    /// ### Complexity
    /// Three syscalls (one per advised zone), each O(1) kernel-side.
    /// Safe to call on every hot event.
    ///
    /// ### What it does *not* do
    /// * Does not block — `madvise` is asynchronous.
    /// * Does not guarantee residency — under memory pressure the
    ///   kernel may drop the pages before the app reads them.
    /// * Does not prefetch property / string / tier zones — those are
    ///   accessed via distinct locality patterns and are not keyed by
    ///   community_id. Callers that need them should issue their own
    ///   targeted advise.
    pub fn prefetch_community(&self, community_id: u32) -> SubstrateResult<()> {
        let Some((first, last)) = self.community_slot_range(community_id) else {
            return Ok(());
        };
        // Byte ranges on each zone. `last` is inclusive so we add one to
        // cover the last slot's full record.
        let slot_count = (last - first + 1) as usize;

        // Nodes zone: 32 B per slot.
        let nodes_off = (first as usize) * NodeRecord::SIZE;
        let nodes_len = slot_count * NodeRecord::SIZE;
        self.writer
            .advise_zone_willneed(crate::file::Zone::Nodes, nodes_off, nodes_len)?;

        // Community side-column: 4 B per slot (u32).
        let cid_off = (first as usize) * core::mem::size_of::<u32>();
        let cid_len = slot_count * core::mem::size_of::<u32>();
        self.writer
            .advise_zone_willneed(crate::file::Zone::Community, cid_off, cid_len)?;

        // Hilbert side-column: 4 B per slot (u32).
        self.writer
            .advise_zone_willneed(crate::file::Zone::Hilbert, cid_off, cid_len)?;

        Ok(())
    }

    /// Convenience: read the `community_id` of `node_id` and issue a
    /// prefetch for its whole community. Returns `Ok(())` silently if
    /// the node is tombstoned, out of range, or uncategorized
    /// (`community_id == 0`).
    ///
    /// This is the one-call hook to drop at retrieval hits / citations /
    /// spreading-activation seeds.
    pub fn on_node_activated(&self, node_id: NodeId) -> SubstrateResult<()> {
        let slot = node_id.0 as u32;
        if slot == 0 || slot >= self.slot_high_water() {
            return Ok(());
        }
        let Some(rec) = self.writer.read_node(slot)? else {
            return Ok(());
        };
        if rec.is_tombstoned() {
            return Ok(());
        }
        // Community 0 = "uncategorized". Advising the entire
        // uncategorized slice is rarely useful (it's the default bucket)
        // and would spam advise on cold zones. Skip.
        if rec.community_id == 0 {
            return Ok(());
        }
        self.prefetch_community(rec.community_id)
    }

    /// Rebuild the `(first_slot, last_slot)` maps from the current Nodes
    /// zone. Called after compaction / bulk re-sort to track the new
    /// layout — online allocation maintains the maps incrementally, but
    /// a `CompactCommunity` cycle moves nodes to fresh slots that the
    /// allocator didn't see. Cost: O(node_high_water) Nodes-zone scan.
    pub fn refresh_community_ranges(&self) -> SubstrateResult<()> {
        self.community_placements.clear();
        self.community_first_slots.clear();
        let hw = self.next_node_id.load(Ordering::Acquire);
        for slot in 1..hw {
            let Some(rec) = self.writer.read_node(slot)? else {
                continue;
            };
            if rec.is_tombstoned() {
                continue;
            }
            // Ascending scan → first_slot takes the min, last_slot the max.
            self.community_first_slots
                .entry(rec.community_id)
                .or_insert(slot);
            self.community_placements.insert(rec.community_id, slot);
        }
        Ok(())
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

    // ==================================================================
    // Coactivation (COACT) typed-edge column API (T7 Step 5)
    // ==================================================================
    //
    // Hub-side semantics (count, last_seen_ts, reward) live in
    // `obrain-hub::memory::CoactivationMap`. The substrate side stores a
    // single Q0.16 cumulative weight per (a, b) pair as the `weight_u16`
    // column of an EdgeRecord whose `edge_type` is the dict-interned
    // `"COACT"` id. The Hub keeps a `(NodeId, NodeId) → EdgeId` cache so
    // it can reinforce a known slot in O(1) without scanning chains.

    /// Return the dict-interned id of the canonical COACT edge type,
    /// registering it on demand if this is the first call on a fresh
    /// substrate. Persisted to `substrate.dict` at the next [`Self::flush`].
    ///
    /// Cheap to call repeatedly (single `RwLock::read` fast-path on the
    /// already-interned name).
    pub fn coact_type_id(&self) -> SubstrateResult<u16> {
        // Fast-path: already interned.
        if let Some(id) = self
            .edge_types
            .read()
            .id_for(crate::record::COACT_EDGE_TYPE_NAME)
        {
            return Ok(id);
        }
        // Slow-path: take the write lock and intern.
        self.edge_types
            .write()
            .intern(crate::record::COACT_EDGE_TYPE_NAME)
    }

    /// Saturating-add `delta` (clamped to `[0.0, 1.0]`) to the COACT
    /// edge slot's weight column, returning the new weight in `f32`.
    ///
    /// The caller is responsible for ensuring `id` resolves to a live
    /// edge whose `edge_type` is the COACT id (typically because the
    /// Hub allocated the slot itself when first observing the pair).
    pub fn coact_reinforce_f32(
        &self,
        id: EdgeId,
        delta: f32,
    ) -> SubstrateResult<f32> {
        let delta_q16 = (delta.clamp(0.0, 1.0) * 65535.0).round() as u16;
        let new = self.writer.coact_reinforce_at(id.0, delta_q16)?;
        Ok(new as f32 / 65535.0)
    }

    /// Apply a Q0.16 multiplicative decay to every live COACT edge's
    /// weight column. Synapse and other edge types are untouched.
    /// `factor` is clamped to `[0.0, 1.0]`.
    pub fn decay_all_coact(&self, factor: f32) -> SubstrateResult<()> {
        let factor_q16 =
            (factor.clamp(0.0, 1.0) * 65536.0).round().min(65535.0) as u16;
        let coact_id = self.coact_type_id()?;
        self.writer.decay_all_coact(
            factor_q16,
            self.edge_slot_high_water(),
            coact_id,
        )
    }

    /// Read the current weight (`f32` in `[0, 1]`) of a COACT edge slot.
    /// Returns `None` if the slot is unallocated, tombstoned, or not a
    /// COACT-typed edge.
    pub fn coact_weight_f32(&self, id: EdgeId) -> SubstrateResult<Option<f32>> {
        if id.0 == 0 || id.0 >= self.edge_slot_high_water() {
            return Ok(None);
        }
        let coact_id = self.coact_type_id()?;
        Ok(self
            .writer
            .read_edge(id.0)?
            .filter(|r| {
                r.flags & edge_flags::TOMBSTONED == 0 && r.edge_type == coact_id
            })
            .map(|r| r.weight_f32()))
    }

    // ---- Engram seeding (T7 Step 6) ----------------------------------
    //
    // The substrate stores two pieces of state per engram:
    //
    //   * an id-keyed membership list in the EngramZone side-table
    //     (`Writer::set_engram_members`), and
    //   * a 64-bit Bloom signature on every member node in the
    //     EngramBitset column (`Writer::add_engram_bit`).
    //
    // Hub-level callers (CoactivationMap → cluster detection → engram
    // formation) hand us a list of node-id clusters; we allocate fresh
    // engram ids out of the dict-persisted `next_engram_id` counter,
    // write the membership list, and OR the bit into each member's
    // signature. Every step is WAL-first (the underlying `Writer`
    // primitives all log absolute payloads), so a crash mid-batch leaves
    // a consistent durable state.

    /// Allocate the next engram id atomically, advancing the
    /// dict-persisted counter. Returns an error once the u16 space is
    /// exhausted (max 65535 engrams per substrate — the directory zone
    /// is sized for exactly that).
    ///
    /// O(1) on the fast-path; loops on contention.
    fn alloc_engram_id(&self) -> SubstrateResult<u16> {
        loop {
            let cur = self.next_engram_id.load(Ordering::Acquire);
            if cur == 0 {
                // Previously wrapped past u16::MAX — allocator poisoned.
                return Err(SubstrateError::Internal(
                    "engram id allocator exhausted (>65535 engrams)".into(),
                ));
            }
            // After allocating `cur`, the counter advances to `cur + 1`.
            // When `cur == u16::MAX`, the next slot wraps to 0 — that's
            // the poison sentinel observed on the next call.
            let new_counter = cur.checked_add(1).unwrap_or(0);
            match self.next_engram_id.compare_exchange_weak(
                cur,
                new_counter,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Ok(cur),
                Err(_) => continue,
            }
        }
    }

    /// Inspect the current engram-id high-water mark (next id to allocate).
    /// Useful for tests and for checkpoint/snapshot machinery.
    pub fn next_engram_id(&self) -> u16 {
        self.next_engram_id.load(Ordering::Acquire)
    }

    /// Seed a single engram from the given member node ids. Allocates a
    /// fresh engram id, writes the membership list to the EngramZone
    /// side-table, then OR's the engram's bit into every member's
    /// 64-bit signature in the EngramBitset column.
    ///
    /// Returns the allocated engram id.
    ///
    /// **Crash semantics**: all WAL writes happen synchronously through
    /// the underlying `Writer` primitives. After a crash, replay
    /// reconstructs the membership list and bitset deterministically;
    /// however, no checkpoint marker is emitted at the end of the seed,
    /// so an interrupted multi-member seed may leave only the prefix of
    /// bitset OR's that completed before the crash applied. Callers that
    /// need atomic-batch semantics across many engrams should use
    /// [`Self::seed_engrams_batch`] which wraps the whole batch in a
    /// single commit/flush and rejects the entire batch on the first
    /// failure.
    ///
    /// Empty `members` is allowed and only allocates the id + clears the
    /// directory entry; no member bits are touched.
    pub fn seed_engram(&self, members: &[NodeId]) -> SubstrateResult<u16> {
        // Validate node ids before consuming an engram id, so a bad
        // batch can't leak counter values.
        let nid_high = self.next_node_id.load(Ordering::Acquire) as u64;
        for nid in members {
            if nid.0 == 0 || nid.0 >= nid_high {
                return Err(SubstrateError::Internal(format!(
                    "seed_engram: member NodeId({}) out of range \
                     (high-water = {nid_high})",
                    nid.0
                )));
            }
            if nid.0 > u32::MAX as u64 {
                return Err(SubstrateError::Internal(format!(
                    "seed_engram: NodeId({}) exceeds u32 (substrate slot ids \
                     are u32)",
                    nid.0
                )));
            }
        }

        let engram_id = self.alloc_engram_id()?;

        // 1) Write the membership directory entry.
        //    We hand the Vec to the writer (consumes) and keep our own
        //    copy via `members.iter()` for the bit-OR loop below.
        let member_u32: Vec<u32> =
            members.iter().map(|n| n.0 as u32).collect();
        self.writer
            .set_engram_members(engram_id, member_u32.clone())?;

        // 2) OR the engram bit into each member's signature.
        for nid in &member_u32 {
            self.writer.add_engram_bit(*nid, engram_id)?;
        }

        Ok(engram_id)
    }

    /// Seed many engrams in one go. Each cluster becomes a fresh engram
    /// with a freshly allocated id; the returned `Vec<u16>` is in
    /// 1-to-1 order with `clusters`.
    ///
    /// On failure, IDs already allocated to earlier clusters in the
    /// batch are NOT released (the dict counter is monotonic), but the
    /// partial state is durable — a subsequent reopen sees exactly the
    /// engrams whose `set_engram_members` had completed. Callers that
    /// want all-or-nothing semantics should pre-validate inputs.
    pub fn seed_engrams_batch(
        &self,
        clusters: &[Vec<NodeId>],
    ) -> SubstrateResult<Vec<u16>> {
        let mut out = Vec::with_capacity(clusters.len());
        for cluster in clusters {
            let id = self.seed_engram(cluster)?;
            out.push(id);
        }
        Ok(out)
    }

    /// Read the 64-bit engram-membership signature for `node_id`. Returns
    /// `0` for the null sentinel, for slots past the high-water mark, and
    /// for nodes that belong to no engram.
    pub fn engram_bitset(&self, node_id: NodeId) -> SubstrateResult<u64> {
        if node_id.0 == 0
            || node_id.0 as u32 >= self.next_node_id.load(Ordering::Acquire)
        {
            return Ok(0);
        }
        self.writer.engram_bitset(node_id.0 as u32)
    }

    /// OR `engram_id`'s bit into `node_id`'s 64-bit signature. Used by
    /// Hub-side semantic routing to assign a freshly observed node (e.g.
    /// an identity question) into an existing engram without rewriting
    /// the membership directory.
    ///
    /// **Note**: this does NOT update the engram's membership directory
    /// — the bitset is a Bloom signature for fast recall, the directory
    /// is the authoritative member list. Callers that want both should
    /// pair this with [`Self::set_engram_members`] on the resolved set.
    pub fn add_engram_bit(
        &self,
        node_id: NodeId,
        engram_id: u16,
    ) -> SubstrateResult<()> {
        if node_id.0 == 0
            || node_id.0 as u32 >= self.next_node_id.load(Ordering::Acquire)
        {
            return Err(SubstrateError::Internal(format!(
                "add_engram_bit: NodeId({}) out of range",
                node_id.0
            )));
        }
        self.writer
            .add_engram_bit(node_id.0 as u32, engram_id)
    }

    /// Read the membership directory entry for `engram_id`. Returns
    /// `Ok(None)` for unknown / cleared / null engrams; otherwise the
    /// list of member node-id slots.
    pub fn engram_members(
        &self,
        engram_id: u16,
    ) -> SubstrateResult<Option<Vec<NodeId>>> {
        Ok(self
            .writer
            .engram_members(engram_id)?
            .map(|v| v.into_iter().map(|n| NodeId(n as u64)).collect()))
    }

    /// Replace the membership directory for `engram_id` with `members`.
    /// Empty `members` clears the slot. Bitset bits on the affected
    /// nodes are NOT updated by this call — pair with
    /// [`Self::add_engram_bit`] per member if needed.
    pub fn set_engram_members(
        &self,
        engram_id: u16,
        members: &[NodeId],
    ) -> SubstrateResult<()> {
        let raw: Vec<u32> = members.iter().map(|n| n.0 as u32).collect();
        self.writer.set_engram_members(engram_id, raw)
    }

    /// Compute the top-`k` nodes by engram-bitset overlap with
    /// `query_nid`'s 64-bit signature. Returns `Vec<(NodeId, overlap_bits)>`
    /// sorted by descending overlap. Nodes with zero overlap are skipped.
    /// SIMD-accelerated on aarch64 / x86_64.
    ///
    /// This is the substrate-level Hopfield recall primitive used by
    /// cognitive recall (engram-mediated cross-session retrieval).
    pub fn hopfield_recall(
        &self,
        query_nid: NodeId,
        k: usize,
    ) -> SubstrateResult<Vec<(NodeId, u32)>> {
        if query_nid.0 == 0
            || query_nid.0 as u32 >= self.next_node_id.load(Ordering::Acquire)
        {
            return Ok(Vec::new());
        }
        Ok(self
            .writer
            .hopfield_recall(query_nid.0 as u32, k)?
            .into_iter()
            .map(|(nid, score)| (NodeId(nid as u64), score))
            .collect())
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

    /// Interior helper: route a `Value::Vector` write to the
    /// [`VecColumnRegistry`]. Interns the property-key, locks the
    /// substrate for the duration of `write_slot` (which may `ensure_room`
    /// the zone), and swallows errors with a log — the `GraphStoreMut`
    /// trait's setters are infallible, so the historical contract is to
    /// silently drop a bad write and let the next flush surface the
    /// problem via the props sidecar. For vectors, a bad write generally
    /// means the spec drifted (different dim for the same key) or the
    /// zone could not be grown; both are programming bugs and land in
    /// the `error!` log.
    fn route_vector_write(
        &self,
        entity_kind: EntityKind,
        slot: u32,
        key: &str,
        vector: &[f32],
    ) {
        let prop_key_id = match self.prop_keys.write().intern(key) {
            Ok(id) => id,
            Err(e) => {
                tracing::error!(
                    target: "substrate::vec_columns",
                    "prop_keys.intern({key:?}) failed: {e} — dropping Value::Vector write"
                );
                return;
            }
        };
        let pk = PropertyKey::new(key);
        let sub = self.substrate.lock();
        if let Err(e) = self
            .vec_columns
            .write(&sub, &pk, entity_kind, prop_key_id, slot, vector)
        {
            tracing::error!(
                target: "substrate::vec_columns",
                "vec_columns.write(key={key:?}, ek={entity_kind:?}, \
                 slot={slot}, dim={}) failed: {e} — dropping Value::Vector write",
                vector.len()
            );
        }
    }

    /// Allocate the next node id (slot index).
    fn allocate_node_id(&self) -> NodeId {
        let raw = self.next_node_id.fetch_add(1, Ordering::AcqRel);
        NodeId(raw as u64)
    }

    /// Allocate a node slot inside the given community's current page,
    /// opening a new 4 KiB-aligned page if necessary (T11 Step 3).
    ///
    /// Policy:
    ///
    /// * **Fast path** — if this community already owns the *global tail*
    ///   (`last_slot == next_node_id - 1`) and the next slot is still in
    ///   the same page, append at `next_node_id`. No waste.
    /// * **Slow path** — round `next_node_id` up to the next page
    ///   boundary and claim that slot. Any trailing slots in the previous
    ///   page are left as zero-initialised padding (tombstoned-by-absence
    ///   — `is_live_on_disk` filters them out by the zero label_bitset
    ///   never matching a real node's bitset OR by explicit
    ///   `TOMBSTONED` flag if a padding slot is later recycled).
    ///
    /// Concurrent-safe: the compare_exchange loop ensures no two threads
    /// pick the same slot under a `next_node_id` CAS race. The
    /// per-community DashMap entry is then updated under its write lock.
    ///
    /// Also clears `meta_flags::HILBERT_SORTED` because any online
    /// allocation breaks the invariant "file is globally Hilbert-sorted".
    /// Full resort requires another call to
    /// [`Writer::bulk_sort_by_hilbert`].
    fn allocate_node_id_in_community(&self, community_id: u32) -> NodeId {
        // Acquire the per-community entry BEFORE reading next_node_id so
        // the fast-path check is consistent.
        let mut entry = self.community_placements.entry(community_id).or_insert(u32::MAX);

        loop {
            let hw = self.next_node_id.load(Ordering::Acquire);
            let last_slot = *entry;

            // Fast path: extend in place.
            //
            // Preconditions:
            //   - community has a prior allocation (last_slot != u32::MAX),
            //   - that allocation is the global tail (last_slot == hw - 1),
            //   - the next slot (hw) lies in the same 4 KiB page as
            //     last_slot (i.e. the page isn't already full).
            let can_fast_path = last_slot != u32::MAX
                && hw > 0
                && last_slot == hw - 1
                && (last_slot / NODES_PER_PAGE) == (hw / NODES_PER_PAGE);

            if can_fast_path {
                // Single-step bump. Fails only on CAS race with another
                // thread; on failure we re-read hw and try again.
                if self
                    .next_node_id
                    .compare_exchange(hw, hw + 1, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    *entry = hw;
                    self.clear_hilbert_sorted_flag();
                    return NodeId(hw as u64);
                }
                continue;
            }

            // Slow path: open a new page for this community.
            //
            // If hw is already at a page boundary (or at the reserved
            // slot-1 head), use it directly; otherwise round up.
            let aligned = if hw <= 1 || hw % NODES_PER_PAGE == 0 {
                hw.max(1)
            } else {
                // ceil(hw / NODES_PER_PAGE) * NODES_PER_PAGE
                ((hw + NODES_PER_PAGE - 1) / NODES_PER_PAGE) * NODES_PER_PAGE
            };
            if self
                .next_node_id
                .compare_exchange(hw, aligned + 1, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                let was_fresh = last_slot == u32::MAX;
                *entry = aligned;
                // T11 Step 5: track the community's earliest slot so the
                // prefetch hook can compute a bounding range. Fresh
                // community → first_slot = this allocation. Existing
                // community opening a second page → leave first_slot
                // untouched; it still points at the community's oldest
                // live page, which is where the prefetch range must
                // start.
                if was_fresh {
                    self.community_first_slots.insert(community_id, aligned);
                }
                self.clear_hilbert_sorted_flag();
                return NodeId(aligned as u64);
            }
            // CAS lost — retry with a fresh read of hw.
        }
    }

    /// Public, idempotent handle to [`Self::clear_hilbert_sorted_flag`]
    /// used by the [`crate::warden::CommunityWarden`] to force a
    /// `bulk_sort_by_hilbert` re-run after it decides a community is too
    /// fragmented. A no-op if the flag is already clear.
    pub fn invalidate_layout_flag(&self) {
        self.clear_hilbert_sorted_flag();
    }

    /// Clear `meta_flags::HILBERT_SORTED` on the in-memory meta header.
    ///
    /// Called on every online allocation because any new node breaks the
    /// post-sort invariant that the file is globally Hilbert-ordered.
    /// The flag is persisted to disk on the next checkpoint / flush.
    fn clear_hilbert_sorted_flag(&self) {
        let mut sub = self.substrate.lock();
        let mut h = sub.meta_header();
        if h.flags & meta_flags::HILBERT_SORTED != 0 {
            h.flags &= !meta_flags::HILBERT_SORTED;
            // Best-effort in-memory rewrite; persistence lands via the
            // checkpoint path. An I/O error here would mean msync/fsync
            // failure which the caller will surface on flush.
            let _ = sub.write_meta_header(&h);
        }
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
    ///
    /// `pub(crate)` so tight-loop consumers inside the crate (T12 heat
    /// kernel, spreading activation, Ricci refresh) can iterate
    /// outgoing edges without paying the `Vec` allocation of the
    /// public trait method `edges_from`. The callback receives the
    /// full `EdgeRecord` by reference so callers can read `dst`,
    /// `weight_u16`, `ricci_u8`, etc. in one pass.
    pub(crate) fn walk_outgoing_chain(
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
        // T16.7: vector-typed properties live in their own mmap zone.
        // Check the registry first so reads after a fresh open resolve
        // without waiting for a `set_node_property` in this session.
        if let Some(arc) = self.vec_columns.read(key, EntityKind::Node, id.0 as u32) {
            return Some(Value::Vector(arc));
        }
        let entry = self.nodes.get(&id)?;
        entry.properties.get(key).cloned()
    }

    fn get_edge_property(&self, id: EdgeId, key: &PropertyKey) -> Option<Value> {
        if !self.is_live_edge_on_disk(id) {
            return None;
        }
        if let Some(arc) = self.vec_columns.read(key, EntityKind::Edge, id.0 as u32) {
            return Some(Value::Vector(arc));
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
        // T16.7: `Value::Vector` writes bypass the DashMap + props
        // sidecar and go straight to a dense mmap'd column. This keeps
        // embeddings off the anon heap. Scalar values still take the
        // in-memory path.
        if let Value::Vector(ref v) = value {
            self.route_vector_write(EntityKind::Node, id.0 as u32, key, v);
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
        if let Value::Vector(ref v) = value {
            // EdgeId is u64 in the trait, but `VecColumnWriter::write_slot`
            // takes u32. In practice no substrate deployment has come close
            // to 2^32 live edges; if that ever changes, the vec-column
            // schema needs a v2 bump to widen the slot field.
            let slot = id.0 as u32;
            self.route_vector_write(EntityKind::Edge, slot, key, v);
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
    fn properties_survive_reopen() {
        use obrain_core::graph::traits::GraphStore;
        use std::sync::Arc;

        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("kb");

        // --- Write phase ---
        // `generic_vec` is a user-visible vector property that MUST roundtrip.
        // We intentionally avoid `_st_embedding` here because that key is
        // filtered during `load_properties` (see `SKIP_ON_LOAD_PROP_KEYS`);
        // its roundtrip semantics are covered by
        // `skip_on_load_keys_are_dropped_after_reopen` below.
        let (node_id, edge_id, generic_vec) = {
            let s = SubstrateStore::create(&path).unwrap();
            let a = s.create_node(&["Doc"]);
            let b = s.create_node(&["Doc"]);
            let e = s.create_edge(a, b, "LINKS_TO");

            let vec: Arc<[f32]> = Arc::from(vec![0.1_f32, 0.2, -0.3, 0.5].into_boxed_slice());
            s.set_node_property(a, "title", Value::String("hello".into()));
            s.set_node_property(a, "generic_vec", Value::Vector(vec.clone()));
            s.set_node_property(a, "count", Value::Int64(42));
            s.set_edge_property(e, "weight", Value::Float64(0.75));
            s.flush().unwrap();
            (a, e, vec)
        };

        // --- Reopen phase ---
        let s = SubstrateStore::open(&path).unwrap();

        // Node properties
        let title = s
            .get_node_property(node_id, &PropertyKey::new("title"))
            .expect("title should persist");
        assert!(matches!(title, Value::String(ref s) if s.as_str() == "hello"));

        let count = s
            .get_node_property(node_id, &PropertyKey::new("count"))
            .expect("count should persist");
        assert!(matches!(count, Value::Int64(42)));

        let emb = s
            .get_node_property(node_id, &PropertyKey::new("generic_vec"))
            .expect("generic_vec should persist");
        match emb {
            Value::Vector(v) => {
                assert_eq!(&*v, &*generic_vec, "vector value must roundtrip byte-exact");
            }
            other => panic!("expected Vector, got {other:?}"),
        }

        // Edge property
        let w = s
            .get_edge_property(edge_id, &PropertyKey::new("weight"))
            .expect("edge weight should persist");
        assert!(matches!(w, Value::Float64(f) if (f - 0.75).abs() < 1e-12));
    }

    /// T16.7 contract: `Value::Vector` writes — including those named
    /// in `SKIP_ON_LOAD_PROP_KEYS` — are routed to dense mmap'd
    /// `substrate.veccol.*` zones and roundtrip byte-exactly through
    /// a reopen, without going near the bincode `substrate.props`
    /// sidecar. This is the replacement for the pre-T16.7 contract
    /// where `_st_embedding` was silently dropped at load time to
    /// dodge the anon-RSS blow-out.
    ///
    /// Regression guard: if vector routing stops triggering (e.g.
    /// someone removes the `Value::Vector` arm in `set_node_property`),
    /// this test goes red — the vector ends up in the DashMap, hits
    /// `SKIP_ON_LOAD_PROP_KEYS`, and the reopen returns `None` again,
    /// re-introducing the anon-RSS problem.
    #[test]
    fn vector_properties_roundtrip_via_vec_columns_after_reopen() {
        use obrain_core::graph::traits::GraphStore;
        use std::sync::Arc;

        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("kb");

        let (node_id, expected) = {
            let s = SubstrateStore::create(&path).unwrap();
            let a = s.create_node(&["Doc"]);
            let v: Vec<f32> = (0..8).map(|i| (i as f32) * 0.125).collect();
            let arc: Arc<[f32]> = Arc::from(v.clone().into_boxed_slice());
            s.set_node_property(a, "title", Value::String("persistent".into()));
            s.set_node_property(a, "_st_embedding", Value::Vector(arc));
            s.flush().unwrap();
            (a, v)
        };

        let s = SubstrateStore::open(&path).unwrap();

        // Non-vector key roundtrips through the props sidecar as usual.
        assert!(
            s.get_node_property(node_id, &PropertyKey::new("title")).is_some(),
            "scalar key should roundtrip via the props sidecar"
        );

        // Vector roundtrip: post-T16.7 this is stored in the vec-column
        // zone and hydrated back on open — bytes-exact.
        let got = s
            .get_node_property(node_id, &PropertyKey::new("_st_embedding"))
            .expect("_st_embedding should roundtrip via vec_columns after T16.7");
        match got {
            Value::Vector(arc) => {
                assert_eq!(
                    arc.as_ref(),
                    &expected[..],
                    "vector payload must roundtrip byte-exact"
                );
            }
            other => panic!("expected Value::Vector, got {other:?}"),
        }

        // Every name in `SKIP_ON_LOAD_PROP_KEYS` is still a valid lookup
        // key — the list now only affects legacy bincode-sidecar bases
        // (pre-T16.7 migrations); freshly-written vectors are untouched
        // by it. Exercise the lookup path so a future regression that
        // panics on these keys is caught.
        for key in SKIP_ON_LOAD_PROP_KEYS {
            let _ = s.get_node_property(node_id, &PropertyKey::new(*key));
        }
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

    // ---------------------------------------------------------------------
    // T7 Step 5 — COACT typed-edge column API on the store
    // ---------------------------------------------------------------------

    #[test]
    fn coact_type_id_is_interned_lazily_and_persisted() {
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("kb");
        let s = SubstrateStore::create(&path).unwrap();

        // First call → registers "COACT" in the edge-type dict.
        let id1 = s.coact_type_id().unwrap();
        // Second call → fast-path read returns the same id.
        let id2 = s.coact_type_id().unwrap();
        assert_eq!(id1, id2);

        s.flush().unwrap();
        // The persisted dict carries the name.
        let snap =
            crate::dict::DictSnapshot::load(&path.join(DICT_FILENAME)).unwrap();
        assert!(
            snap.edge_types
                .iter()
                .any(|n| n == crate::record::COACT_EDGE_TYPE_NAME),
            "dict must carry COACT after flush, got {:?}",
            snap.edge_types
        );

        // Reopen → the id is recovered (might or might not be the same
        // numeric id depending on registry order, but the registry must
        // resolve the name).
        drop(s);
        let s2 = SubstrateStore::open(&path).unwrap();
        let id3 = s2.coact_type_id().unwrap();
        assert_eq!(id3, id1, "id must be stable across reopen");
    }

    #[test]
    fn coact_reinforce_f32_saturates_and_reads_back() {
        let (_td, s) = store();
        let coact_id = s.coact_type_id().unwrap();

        // Manually allocate a COACT edge slot via the writer (we don't
        // expose a typed-create helper at the store level yet — Hub-side
        // CoactivationMap will do this through the higher API in Step 5
        // wiring).
        let mut e = EdgeRecord::default();
        e.src = 1;
        e.dst = 2;
        e.edge_type = coact_id;
        e.weight_u16 = 0;
        s.writer.write_edge(1, e).unwrap();
        // Important: bump the edge high-water so coact_weight_f32 will
        // even consider slot 1 (the API guards on edge_slot_high_water).
        s.next_edge_id.store(2, Ordering::Release);

        // Reinforce three times by 0.4 → saturates at 1.0.
        let new = s.coact_reinforce_f32(EdgeId(1), 0.4).unwrap();
        assert!((new - 0.4).abs() < 1e-3, "got {new}");
        let new = s.coact_reinforce_f32(EdgeId(1), 0.4).unwrap();
        assert!((new - 0.8).abs() < 1e-3, "got {new}");
        let new = s.coact_reinforce_f32(EdgeId(1), 0.4).unwrap();
        // Saturated at 1.0 (allow ±1 ULP for Q0.16 quantization).
        assert!(new >= 0.999, "got {new}");

        // Read-back via typed accessor: only returns Some for COACT-typed
        // edges; returns None for synapse / wrong type.
        let w = s.coact_weight_f32(EdgeId(1)).unwrap();
        assert!(w.is_some());
        assert!(w.unwrap() >= 0.999);

        // A non-COACT edge slot must NOT be readable as a COACT weight.
        let mut e2 = EdgeRecord::default();
        e2.src = 3;
        e2.dst = 4;
        e2.edge_type = coact_id.wrapping_add(1); // anything but COACT
        e2.weight_u16 = 0xFFFF;
        s.writer.write_edge(2, e2).unwrap();
        s.next_edge_id.store(3, Ordering::Release);
        assert_eq!(s.coact_weight_f32(EdgeId(2)).unwrap(), None);
    }

    #[test]
    fn decay_all_coact_only_touches_coact_typed_edges() {
        let (_td, s) = store();
        let coact_id = s.coact_type_id().unwrap();
        let other_id = coact_id.wrapping_add(1);

        // Slots 1..=4: COACT, slots 5..=6: other type. All weight=0x8000.
        for slot in 1..=6u64 {
            let mut e = EdgeRecord::default();
            e.src = slot as u32;
            e.dst = (slot as u32) + 100;
            e.edge_type = if slot <= 4 { coact_id } else { other_id };
            e.weight_u16 = 0x8000;
            s.writer.write_edge(slot, e).unwrap();
        }
        s.next_edge_id.store(7, Ordering::Release);

        // Decay COACT by 0.5 — only COACT slots halve.
        s.decay_all_coact(0.5).unwrap();

        for slot in 1..=4u64 {
            let rec = s.writer.read_edge(slot).unwrap().unwrap();
            assert_eq!(rec.weight_u16, 0x4000, "COACT slot {slot} must be halved");
        }
        for slot in 5..=6u64 {
            let rec = s.writer.read_edge(slot).unwrap().unwrap();
            assert_eq!(
                rec.weight_u16, 0x8000,
                "non-COACT slot {slot} must NOT be touched"
            );
        }
    }

    // ---------------------------------------------------------------------
    // T7 Step 6 — Engram seed batch operation + id allocator persistence
    // ---------------------------------------------------------------------

    #[test]
    fn alloc_engram_id_starts_at_one_and_increments() {
        let (_td, s) = store();
        assert_eq!(s.next_engram_id(), 1);
        let a = s.alloc_engram_id().unwrap();
        let b = s.alloc_engram_id().unwrap();
        let c = s.alloc_engram_id().unwrap();
        assert_eq!((a, b, c), (1, 2, 3));
        assert_eq!(s.next_engram_id(), 4);
    }

    #[test]
    fn alloc_engram_id_rejects_after_exhaustion() {
        let (_td, s) = store();
        // Pre-poison the counter to simulate a previously-exhausted
        // allocator (this avoids burning 65k allocations in a unit test).
        s.next_engram_id.store(0, Ordering::Release);
        let err = s.alloc_engram_id().unwrap_err();
        let msg = format!("{err:?}");
        assert!(msg.contains("exhausted"), "got: {msg}");
    }

    #[test]
    fn alloc_engram_id_last_slot_then_poisoned() {
        let (_td, s) = store();
        // Seed the counter at u16::MAX — one allocation must succeed
        // (returns 65535 = MAX_ENGRAM_ID), the next must fail.
        s.next_engram_id.store(u16::MAX, Ordering::Release);
        let last = s.alloc_engram_id().unwrap();
        assert_eq!(last, u16::MAX);
        assert_eq!(s.next_engram_id(), 0, "counter wraps to poison sentinel");
        let err = s.alloc_engram_id().unwrap_err();
        assert!(format!("{err:?}").contains("exhausted"));
    }

    #[test]
    fn seed_engram_writes_members_and_bitset() {
        let (_td, s) = store();
        let n1 = s.create_node(&["A"]);
        let n2 = s.create_node(&["B"]);
        let n3 = s.create_node(&["C"]);

        let eid = s.seed_engram(&[n1, n2, n3]).unwrap();
        assert_eq!(eid, 1, "first engram id is 1");

        // Members must round-trip through the EngramZone directory.
        let members = s.writer.engram_members(eid).unwrap().unwrap();
        let mut expected: Vec<u32> =
            [n1, n2, n3].iter().map(|n| n.0 as u32).collect();
        let mut got = members.clone();
        expected.sort();
        got.sort();
        assert_eq!(got, expected);

        // The corresponding bit must be set in every member's signature.
        let mask = crate::engram_bitset::engram_bit_mask(eid);
        for nid in [n1, n2, n3] {
            let bits = s.writer.engram_bitset(nid.0 as u32).unwrap();
            assert!(
                bits & mask == mask,
                "node {nid:?} missing engram bit; bits=0x{bits:016x} mask=0x{mask:016x}"
            );
        }
    }

    #[test]
    fn seed_engram_rejects_out_of_range_node() {
        let (_td, s) = store();
        let _n1 = s.create_node(&["A"]);
        // NodeId(99) is past the high-water mark.
        let err = s.seed_engram(&[NodeId(99)]).unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.contains("out of range") || msg.contains("high-water"),
            "got: {msg}"
        );
        // The id allocator must NOT have advanced.
        assert_eq!(s.next_engram_id(), 1);
    }

    #[test]
    fn seed_engram_rejects_null_node_zero() {
        let (_td, s) = store();
        let _n1 = s.create_node(&["A"]);
        let err = s.seed_engram(&[NodeId(0)]).unwrap_err();
        assert!(format!("{err:?}").contains("out of range"));
    }

    #[test]
    fn seed_engram_empty_members_allocates_id_only() {
        let (_td, s) = store();
        let eid = s.seed_engram(&[]).unwrap();
        assert_eq!(eid, 1);
        // Empty membership clears the directory slot per
        // `EngramZone::set_members_raw` semantics → reads back as None.
        // The id is still consumed (allocator advanced), so a follow-up
        // seed gets id=2 — the empty seed is observable only through the
        // counter, not the directory.
        let m = s.writer.engram_members(eid).unwrap();
        assert!(m.is_none(), "cleared directory slot must read as None");
        assert_eq!(s.next_engram_id(), 2);
    }

    #[test]
    fn seed_engrams_batch_allocates_distinct_ids_in_order() {
        let (_td, s) = store();
        let n1 = s.create_node(&["A"]);
        let n2 = s.create_node(&["B"]);
        let n3 = s.create_node(&["C"]);
        let n4 = s.create_node(&["D"]);

        let clusters = vec![
            vec![n1, n2],
            vec![n3, n4],
            vec![n1, n3], // overlap allowed — bit ORs accumulate
        ];
        let ids = s.seed_engrams_batch(&clusters).unwrap();
        assert_eq!(ids, vec![1, 2, 3]);

        // n1 belongs to engrams {1, 3}; n3 belongs to {2, 3}.
        let m1 = crate::engram_bitset::engram_bit_mask(1);
        let m2 = crate::engram_bitset::engram_bit_mask(2);
        let m3 = crate::engram_bitset::engram_bit_mask(3);
        let bits_n1 = s.writer.engram_bitset(n1.0 as u32).unwrap();
        let bits_n3 = s.writer.engram_bitset(n3.0 as u32).unwrap();
        assert_eq!(bits_n1 & m1, m1);
        assert_eq!(bits_n1 & m3, m3);
        assert_eq!(bits_n3 & m2, m2);
        assert_eq!(bits_n3 & m3, m3);
        // n1 should NOT carry engram-2's bit (it wasn't in cluster 1).
        // (Only meaningful when the masks differ — engram-1 and engram-2
        // hash to different bits modulo 64 for these small ids.)
        if m1 != m2 && m2 != m3 {
            assert_eq!(bits_n1 & m2, 0, "n1 must not carry engram-2 bit");
        }
    }

    #[test]
    fn seed_engram_persists_allocator_across_reopen() {
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("kb");
        let s = SubstrateStore::create(&path).unwrap();
        let n1 = s.create_node(&["A"]);
        let n2 = s.create_node(&["B"]);
        let _ = s.seed_engram(&[n1, n2]).unwrap();
        let _ = s.seed_engram(&[n1]).unwrap();
        assert_eq!(s.next_engram_id(), 3);
        s.flush().unwrap();
        drop(s);

        // Reopen — the counter must come back from substrate.dict.
        let s2 = SubstrateStore::open(&path).unwrap();
        assert_eq!(
            s2.next_engram_id(),
            3,
            "next_engram_id must round-trip via substrate.dict v2"
        );
        // Allocating again gives 3, not 1 — ids are monotonic across reopens.
        let next = s2.alloc_engram_id().unwrap();
        assert_eq!(next, 3);
    }

    #[test]
    fn seed_engram_membership_persists_across_reopen() {
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("kb");
        let s = SubstrateStore::create(&path).unwrap();
        let n1 = s.create_node(&["A"]);
        let n2 = s.create_node(&["B"]);
        let n3 = s.create_node(&["C"]);
        let eid = s.seed_engram(&[n1, n2, n3]).unwrap();
        s.flush().unwrap();
        drop(s);

        let s2 = SubstrateStore::open(&path).unwrap();
        let mut got = s2.writer.engram_members(eid).unwrap().unwrap();
        let mut expected: Vec<u32> =
            [n1, n2, n3].iter().map(|n| n.0 as u32).collect();
        got.sort();
        expected.sort();
        assert_eq!(got, expected);

        // Bitsets must also survive.
        let mask = crate::engram_bitset::engram_bit_mask(eid);
        for nid in [n1, n2, n3] {
            let bits = s2.writer.engram_bitset(nid.0 as u32).unwrap();
            assert!(
                bits & mask == mask,
                "after reopen, node {nid:?} missing engram bit"
            );
        }
    }

    // -----------------------------------------------------------------
    // T11 Step 5 — madvise(WILLNEED) prefetch hook
    // -----------------------------------------------------------------
    //
    // These tests exercise the observable state of the prefetch API
    // (range tracking + idempotency + rebuild parity + live-node
    // reach). madvise is a best-effort kernel hint, so the tests
    // assert that the calls succeed and do not perturb the data; the
    // actual page-fault savings are measured by the bench harness
    // that consumes this API (Step 6).

    #[test]
    fn community_range_tracks_first_and_last_slot_online() {
        let (_td, s) = store();
        // Community 42: three consecutive inserts through the
        // community-local allocator. Fast-path extends the same page
        // so first = allocation #1's slot, last = allocation #3's.
        let a = s.create_node_in_community(&["L"], 42);
        let b = s.create_node_in_community(&["L"], 42);
        let c = s.create_node_in_community(&["L"], 42);
        let first = s.first_slot_for_community(42).expect("first must exist");
        let last = s.last_slot_for_community(42).expect("last must exist");
        assert_eq!(first, a.0 as u32, "first slot = first insertion");
        assert_eq!(last, c.0 as u32, "last slot = last insertion");
        assert!(
            b.0 as u32 >= first && (b.0 as u32) <= last,
            "middle insertion inside bounding range"
        );
        let (rf, rl) = s.community_slot_range(42).unwrap();
        assert_eq!((rf, rl), (first, last));
    }

    #[test]
    fn community_range_separates_communities() {
        let (_td, s) = store();
        let a1 = s.create_node_in_community(&["L"], 1);
        let a2 = s.create_node_in_community(&["L"], 1);
        let b1 = s.create_node_in_community(&["L"], 2);
        let a3 = s.create_node_in_community(&["L"], 1);
        // Community 2 opens its own page (slot alignment forces it on the
        // slow path); its range is just the single slot it owns.
        let (c1_lo, c1_hi) = s.community_slot_range(1).unwrap();
        let (c2_lo, c2_hi) = s.community_slot_range(2).unwrap();
        assert_eq!(c1_lo, a1.0 as u32);
        assert_eq!(c1_hi, a3.0 as u32);
        assert_eq!(c2_lo, b1.0 as u32);
        assert_eq!(c2_hi, b1.0 as u32);
        // Verify c1 actually spans multiple slots, proving the range
        // covers the bucket rather than a single slot.
        assert!(a3.0 as u32 > a2.0 as u32 && a2.0 as u32 > a1.0 as u32);
    }

    #[test]
    fn community_range_none_for_unknown_cid() {
        let (_td, s) = store();
        assert!(s.community_slot_range(999).is_none());
        assert!(s.first_slot_for_community(999).is_none());
        assert!(s.last_slot_for_community(999).is_none());
    }

    #[test]
    fn prefetch_on_empty_community_is_noop() {
        let (_td, s) = store();
        // No community 42 has been created — prefetch must swallow
        // silently (best-effort contract).
        s.prefetch_community(42).expect("empty prefetch must not error");
    }

    #[test]
    fn prefetch_populated_community_succeeds() {
        let (_td, s) = store();
        for _ in 0..NODES_PER_PAGE as usize + 3 {
            // Force the slow path to fire at least once by pushing the
            // community across a 4 KiB page boundary.
            s.create_node_in_community(&["L"], 7);
        }
        let (lo, hi) = s.community_slot_range(7).unwrap();
        assert!(hi >= lo + NODES_PER_PAGE);
        // Call is idempotent and never errors.
        s.prefetch_community(7).unwrap();
        s.prefetch_community(7).unwrap();
    }

    #[test]
    fn on_node_activated_resolves_cid_and_prefetches() {
        let (_td, s) = store();
        let a = s.create_node_in_community(&["L"], 3);
        let _b = s.create_node_in_community(&["L"], 3);
        // Known node → prefetch returns Ok, no panic.
        s.on_node_activated(a).unwrap();
        // Out-of-range slot → silent Ok.
        s.on_node_activated(NodeId(u32::MAX as u64)).unwrap();
        // Null-sentinel slot → silent Ok.
        s.on_node_activated(NodeId(0)).unwrap();
    }

    #[test]
    fn on_node_activated_skips_uncategorized() {
        let (_td, s) = store();
        let a = s.create_node(&["Uncat"]); // community_id == 0
        // No community 0 range is tracked in the hot path; the call
        // still succeeds as a silent no-op.
        s.on_node_activated(a).unwrap();
    }

    #[test]
    fn rebuild_populates_first_and_last_slot_maps() {
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("kb");
        let s = SubstrateStore::create(&path).unwrap();
        let a = s.create_node_in_community(&["L"], 11);
        let _b = s.create_node_in_community(&["L"], 11);
        let c = s.create_node_in_community(&["L"], 11);
        let d = s.create_node_in_community(&["L"], 22);
        s.flush().unwrap();
        drop(s);

        let s2 = SubstrateStore::open(&path).unwrap();
        let (lo11, hi11) = s2.community_slot_range(11).unwrap();
        assert_eq!(lo11, a.0 as u32);
        assert_eq!(hi11, c.0 as u32);
        let (lo22, hi22) = s2.community_slot_range(22).unwrap();
        assert_eq!(lo22, d.0 as u32);
        assert_eq!(hi22, d.0 as u32);
        // Prefetch after reopen must still work — proves the rebuild
        // populated everything the hot path needs.
        s2.prefetch_community(11).unwrap();
        s2.prefetch_community(22).unwrap();
    }

    #[test]
    fn refresh_community_ranges_repairs_after_compaction() {
        let (_td, s) = store();
        // Seed three distinct communities so refresh has something to
        // rewrite.
        let a1 = s.create_node_in_community(&["L"], 1);
        let a2 = s.create_node_in_community(&["L"], 1);
        let b1 = s.create_node_in_community(&["L"], 2);
        let c1 = s.create_node_in_community(&["L"], 3);
        // Manual invariant poisoning to simulate drift (e.g. the maps
        // fell out of sync with disk after a compaction cycle that
        // bypassed the allocator).
        s.community_first_slots.insert(1, 9999);
        s.community_placements.insert(1, 1);
        // After refresh, the maps rebuild from the actual Nodes zone.
        s.refresh_community_ranges().unwrap();
        let (lo1, hi1) = s.community_slot_range(1).unwrap();
        let (lo2, hi2) = s.community_slot_range(2).unwrap();
        let (lo3, hi3) = s.community_slot_range(3).unwrap();
        assert_eq!(lo1, a1.0 as u32);
        assert_eq!(hi1, a2.0 as u32);
        assert_eq!(lo2, b1.0 as u32);
        assert_eq!(hi2, b1.0 as u32);
        assert_eq!(lo3, c1.0 as u32);
        assert_eq!(hi3, c1.0 as u32);
    }

    #[test]
    fn prefetch_does_not_mutate_data() {
        // The prefetch hook is a madvise hint — it must not touch
        // bytes. Verify the Nodes zone content is bit-identical before
        // and after the call.
        let (_td, s) = store();
        for _ in 0..NODES_PER_PAGE as usize {
            s.create_node_in_community(&["L"], 5);
        }
        let before: Vec<u8> = s
            .writer
            .substrate()
            .lock()
            .open_zone(crate::file::Zone::Nodes)
            .unwrap()
            .as_slice()
            .to_vec();
        s.prefetch_community(5).unwrap();
        let after: Vec<u8> = s
            .writer
            .substrate()
            .lock()
            .open_zone(crate::file::Zone::Nodes)
            .unwrap()
            .as_slice()
            .to_vec();
        assert_eq!(before, after, "prefetch must be read-only");
    }

    #[test]
    fn advise_zone_out_of_bounds_is_clamped() {
        // Direct ZoneFile::advise_willneed smoke — the wrapper must
        // tolerate offset/len overrunning the mapped region (the
        // prefetch path relies on this for the last partial page).
        let (_td, s) = store();
        // Make sure the zone is non-empty so ZoneFile::map is Some.
        let _ = s.create_node_in_community(&["L"], 1);
        // Huge offset/len: silent Ok.
        s.writer
            .advise_zone_willneed(crate::file::Zone::Nodes, 0, usize::MAX / 2)
            .unwrap();
        s.writer
            .advise_zone_willneed(crate::file::Zone::Nodes, usize::MAX / 2, 4096)
            .unwrap();
    }

    #[test]
    fn advise_empty_zone_is_noop() {
        // Fresh substrate: Hilbert / Community zones haven't been
        // touched, so their ZoneFile::map is None. Advise must swallow.
        let (_td, s) = store();
        s.writer
            .advise_zone_willneed(crate::file::Zone::Hilbert, 0, 4096)
            .unwrap();
        s.writer
            .advise_zone_willneed(crate::file::Zone::Community, 0, 4096)
            .unwrap();
    }
}
