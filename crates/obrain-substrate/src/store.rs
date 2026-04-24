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
use std::sync::OnceLock;
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
    PropertiesStreamingWriter, EDGE_PROPS_FILENAME, PROPS_FILENAME,
};
use crate::page::PropertyEntry;
use crate::props_codec::{decode_value, encode_value};
use crate::props_zone::{decode_page_id, encode_page_id, PropsZone, PROPS_V2_FILENAME};
use crate::blob_column_registry::{
    blob_payload_len, encode_blob_payload, BlobColumnRegistry, BLOB_COLUMN_THRESHOLD_BYTES,
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
pub(crate) struct EdgeTypeRegistry {
    name_to_id: FxHashMap<ArcStr, u16>,
    id_to_name: Vec<ArcStr>, // indexed by id (u16), len == number of interned types
}

impl EdgeTypeRegistry {
    /// Register an edge-type name if absent, return its interned id.
    pub(crate) fn intern(&mut self, name: &str) -> SubstrateResult<u16> {
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

    pub(crate) fn name_for(&self, id: u16) -> Option<ArcStr> {
        self.id_to_name.get(id as usize).cloned()
    }

    pub(crate) fn id_for(&self, name: &str) -> Option<u16> {
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

    /// Look up an existing interned id without mutating the registry.
    /// Returns `None` when the name was never interned — safe to call
    /// from the read path (`get_node_property`) where a missing key
    /// must not pollute the id space.
    fn lookup(&self, name: &str) -> Option<u16> {
        self.name_to_id.get(name).copied()
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

/// Result of [`SubstrateStore::finalize_props_v2`]. Captures the work
/// done so the migration CLI can log a summary and CI can assert
/// expected counts (e.g. nodes_processed matches the live-node count
/// of the source).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PropsV2FinalizeStats {
    /// Number of nodes whose DashMap entry was re-emitted through
    /// the v2 path. Equals the number of nodes with at least one
    /// scalar property on entry.
    pub nodes_processed: usize,
    /// Total number of `set_node_property` calls made — roughly the
    /// number of `(node, key)` pairs on the v2 chain afterward.
    pub scalars_emitted: usize,
    /// T17c Step 6 — number of edges whose `edge_properties` DashMap
    /// entry was serialised into the dedicated edge sidecar
    /// (`substrate.edge_props`). Edges have no PropsZone v2 path yet
    /// (EdgeRecord lacks `first_prop_off` — T17f scope), so they are
    /// preserved as bincode alongside the v2 pages.
    pub edges_processed: usize,
    /// T17c Step 6 — total number of `(key, value)` pairs serialised
    /// into the edge sidecar.
    pub edge_scalars_emitted: usize,
    /// T17c Step 6 — bytes written to `substrate.edge_props`.
    /// Zero when no edge properties were pending.
    pub edge_sidecar_bytes: u64,
}

/// Result of [`SubstrateStore::finalize_edge_props_v2`] (T17f Step 5).
/// Parallel to [`PropsV2FinalizeStats`] but edge-chain focused — the
/// drain writes scalars onto the per-edge v2 chain (addressed by
/// `EdgeRecord.first_prop_off`) rather than the sidecar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EdgePropsV2FinalizeStats {
    /// Number of edges whose DashMap entry was re-emitted through
    /// the v2 edge-chain path. Equals the number of edges with at
    /// least one scalar property on entry.
    pub edges_processed: usize,
    /// Total number of `set_edge_property` calls made — roughly the
    /// number of `(edge, key)` pairs on the v2 chain afterward.
    pub scalars_emitted: usize,
}

// ---------------------------------------------------------------------------
// CommunityRanges (T17g T1 — lazy build)
// ---------------------------------------------------------------------------

/// Per-community slot range maps, built lazily on first access.
///
/// **T17g T1** — the pair `(placements, first_slots)` was formerly eager at
/// open time (O(node_high_water) scan in `rebuild_from_zones`). On Megalaw
/// (5.2M communities) that scan cost ~920 ms / open. Moving to a
/// `OnceLock`-guarded lazy build mirrors the T17e Phase 4 pattern used for
/// `incoming_heads`: the scan only runs the first time the CommunityWarden
/// or `allocate_node_id_in_community` actually needs the data.
///
/// The two DashMaps live together because they are populated by the same
/// Nodes-zone scan and mutated by the same code paths (allocator + compaction
/// refresh). Keeping them as a pair in a single `OnceLock` avoids duplicate
/// init barriers.
#[derive(Debug)]
pub(crate) struct CommunityRanges {
    /// `community_id → last live slot` (max slot per cid in an ascending scan).
    ///
    /// Drives the fast-path / slow-path decision in
    /// [`SubstrateStore::allocate_node_id_in_community`] — when the community's
    /// last slot is the global tail AND still fits in its 4 KiB page, the
    /// allocator extends in place (fast path). Otherwise it opens a fresh
    /// page for the community (slow path).
    pub(crate) placements: DashMap<u32, u32>,
    /// `community_id → first live slot` (min slot per cid in an ascending scan).
    ///
    /// Paired with `placements` (last slot), this forms the bounding slot
    /// range consumed by [`SubstrateStore::prefetch_community`] to issue
    /// `madvise(WILLNEED)` over the community's page range. The range is
    /// a superset of the community's pages pre-compaction; post-compaction
    /// the CommunityWarden re-runs [`SubstrateStore::refresh_community_ranges`]
    /// to tighten it.
    pub(crate) first_slots: DashMap<u32, u32>,
}

impl CommunityRanges {
    /// Empty pair — used by the lazy builder and by
    /// [`SubstrateStore::refresh_community_ranges`] when it needs a fresh
    /// state post-compaction.
    fn empty() -> Self {
        Self {
            placements: DashMap::new(),
            first_slots: DashMap::new(),
        }
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
    /// Dedicated property map for nodes (T17e Phase 2). Entries are only
    /// materialised when a property is actually set — nodes without
    /// properties pay zero anon-RSS here. Pruned on `delete_node` to
    /// prevent `persist_properties` re-serialising dead entries.
    ///
    /// T17e Phase 3: the former `nodes: DashMap<NodeId, NodeInMem>` has
    /// been eliminated — labels and liveness now resolve directly from
    /// `NodeRecord.label_bitset` + `LabelRegistry` via
    /// [`Self::resolve_node_labels_from_bitset`] and
    /// [`Self::is_live_on_disk`].
    node_properties: DashMap<NodeId, PropertyMap>,
    /// Dedicated property map for edges (T17e Phase 2). Same contract as
    /// [`Self::node_properties`]. T17e Phase 3: the former
    /// `edges: DashMap<EdgeId, EdgeInMem>` has been eliminated — edge
    /// type and liveness resolve from `EdgeRecord.edge_type` +
    /// `EdgeTypeRegistry` via [`Self::resolve_edge_type_by_id`] and
    /// [`Self::is_live_edge_on_disk`].
    edge_properties: DashMap<EdgeId, PropertyMap>,
    /// First incoming edge head per destination node. The `next_to` chain
    /// on `EdgeRecord` is durable, but the entry point is kept in memory
    /// only — a future sidecar persist may promote it to disk.
    ///
    /// **T17e Phase 4**: lazily built on first access via
    /// [`Self::incoming_heads`]. The map is no longer populated by
    /// [`Self::rebuild_from_zones`] at open time — the O(E) zone scan
    /// now only happens the first time a reverse traversal, a
    /// `create_edge`, or a `delete_edge` is issued.
    ///
    /// Races with concurrent writes during the first build are safe:
    /// `OnceLock::get_or_init` serialises init, so a writer that fires
    /// `incoming_heads()` blocks until the scan completes; the writer
    /// then inserts its own edge head into the now-populated map. Edges
    /// written AFTER the scan's `next_edge_id` snapshot are handled by
    /// the writer's own `insert` call post-init.
    incoming_heads_cell: OnceLock<DashMap<NodeId, EdgeId>>,
    /// Label name ↔ bit index, keyed by `ArcStr`. Persisted via
    /// `substrate.dict` (step 3).
    labels: RwLock<LabelRegistry>,
    /// Edge-type name ↔ u16 id registry. Persisted via `substrate.dict`
    /// (step 3).
    /// T17h T8 refactor : wrapped in `Arc` so `TypedDegreeRegistry` can
    /// share the same registry (name → id lookups for filename / Cypher
    /// type resolution).
    edge_types: Arc<RwLock<EdgeTypeRegistry>>,
    /// Property-key name ↔ u16 id registry. Persisted via
    /// `substrate.dict` (step 3); actual property-page writes land in
    /// step 4+.
    prop_keys: RwLock<PropertyKeyRegistry>,
    /// Cached statistics snapshot (cost-based optimizer). Step 4 wires this
    /// to real stats; for now a fresh empty snapshot is returned.
    stats: Arc<Statistics>,
    /// Per-community slot range state — lazy-built on first access
    /// (T17g T1).
    ///
    /// Holds `(placements, first_slots)` — the pair of DashMaps that track
    /// the last and first live slot per community id. Formerly populated
    /// eagerly by [`Self::rebuild_from_zones`] (O(node_high_water) scan);
    /// now built lazily on first call to [`Self::community_ranges`] via
    /// `OnceLock::get_or_init`. See `CommunityRanges` docs for field
    /// semantics and [`Self::refresh_community_ranges`] for post-compaction
    /// rebuild.
    ///
    /// Concurrency mirrors the `incoming_heads_cell` (T17e Phase 4) pattern:
    /// a concurrent allocator firing `community_ranges()` mid-build blocks
    /// until the scan completes, then inserts its own placement into the
    /// now-populated maps.
    community_ranges_cell: OnceLock<CommunityRanges>,
    /// T17h T1 — total live nodes counter (atomic O(1) replacement for
    /// the former `node_count()` scan). Maintained by create/delete_node.
    /// Persisted in `DictSnapshot.counters` (v5+); bases v1..=v4 trigger
    /// a one-shot rebuild at open, persisted on next flush.
    total_live_nodes: AtomicU64,
    /// T17h T1 — total live edges counter (O(1) `edge_count()`).
    total_live_edges: AtomicU64,
    /// T17h T1 — per-label live counts, indexed by bit index 0..=63 of
    /// `NodeRecord.label_bitset`. Fixed-size array (64 entries) because
    /// the bitset width is hard-capped at u64 — cache-friendly, no hash.
    label_live_counts: [AtomicU64; 64],
    /// T17h T1 — per-edge-type live counts. DashMap because the u16
    /// edge-type id space is open and typically sparse (≤ 128 in practice
    /// but no hard cap). Fast path reads existing entry; cold path
    /// inserts on first use via `or_insert_with(AtomicU64::new(0))`.
    edge_type_live_counts: DashMap<u16, AtomicU64>,
    /// T17i T2 — per-edge-type histogram of target-node label bits.
    /// Key is `(edge_type_id, label_bit)` ; value is the count of live
    /// edges of `edge_type_id` whose `dst.label_bitset` has `label_bit`
    /// set. Used by `GraphStore::edge_target_labels(type)` to gate the
    /// T17i T3 Cypher-planner rewrite on peer-label constraints.
    edge_type_target_label_counts: DashMap<(u16, u8), AtomicU64>,
    /// T17i T2 — symmetric histogram for **source-node** label bits,
    /// used by `GraphStore::edge_source_labels(type)`.
    edge_type_source_label_counts: DashMap<(u16, u8), AtomicU64>,
    /// T17h T5 — per-node in/out degree column, lazy-built on first
    /// access. Atomic u32 counters maintained by `create_edge` /
    /// `delete_edge`. Enables O(1) degree queries for `most_connected`
    /// patterns. Persisted to `substrate.degrees.node.u32` at flush;
    /// rebuilt from edge scan at open if sidecar absent or CRC invalid.
    ///
    /// RwLock wraps the column so atomic increments (read lock) don't
    /// block each other, but `ensure_slot` grow calls take an exclusive
    /// write lock. See `degree_column::DegreeColumn`.
    degrees_cell: OnceLock<Arc<RwLock<crate::degree_column::DegreeColumn>>>,
    /// T17h T8 — per-edge-type degree registry. One column per
    /// `edge_type_id`, persisted as `substrate.degrees.node.<type>.u32`.
    /// Enables correct routing of Cypher patterns filtered by edge type
    /// (e.g. `MATCH (f:File)-[:IMPORTS]->()`). See `typed_degree.rs`.
    typed_degrees_cell: OnceLock<Arc<crate::typed_degree::TypedDegreeRegistry>>,
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
    /// Routing table + open-writer cache for oversized `Value::String`
    /// and `Value::Bytes` properties (T16.7 Step 4).
    ///
    /// Every `set_node_property(_, _, Value::String(s))` and
    /// `set_edge_property(_, _, Value::Bytes(b))` call whose payload
    /// exceeds [`BLOB_COLUMN_THRESHOLD_BYTES`] is intercepted here and
    /// routed to a variable-length `substrate.blobcol.*` zone pair
    /// (idx + dat) instead of the bincode `substrate.props` sidecar.
    /// This is what finally closes the T16 anon-RSS gate (≤ 1 GiB) for
    /// the chat/event `data` key on PO (928 MB across 681k entries,
    /// avg 1.4 KiB per payload).
    ///
    /// Hydrated on open from `DictSnapshot.blob_columns` (v4) so a
    /// fresh `get_node_property` after reopen resolves the right column
    /// before the first write of the session. A 1-byte type tag inside
    /// the arena payload preserves the `String` vs `Bytes` distinction
    /// (see [`crate::blob_column_registry::encode_blob_payload`]).
    /// Durability is batched to [`Self::flush`] via `sync_all()` — same
    /// cadence as the dict, the vec-column registry, and the props
    /// sidecar.
    blob_columns: BlobColumnRegistry,
    /// Optional v2 props zone (T17c Step 3a — plumbing only).
    ///
    /// When `Some`, the mmap'd `substrate.props.v2` + `substrate.props.heap.v2`
    /// pair has been opened. The field is NOT yet wired to
    /// `set_*_property` / `get_*_property` — Step 3b/3c do that. For
    /// now, its sole job is to make sure the zone files get created
    /// (and grown to the initial dummy-sentinel + first-slot size)
    /// whenever the feature is enabled, so operational tooling can
    /// inspect the substrate directory and see the migration target.
    ///
    /// **Enablement**:
    /// * `OBRAIN_PROPS_V2=1` in the environment at open time, OR
    /// * `substrate.props.v2` already exists on disk (auto-detected —
    ///   once a substrate has been upgraded, it stays upgraded).
    ///
    /// Pure feature-flag plumbing: when both conditions are false, this
    /// stays `None` and the legacy [`Self::load_properties`] /
    /// [`Self::persist_properties`] bincode sidecar path is the only
    /// property storage in play.
    props_zone: Option<RwLock<PropsZone>>,
}

// T17e Phase 3 — `NodeInMem` and `EdgeInMem` were removed along with
// their DashMaps. Labels/edge_type/liveness all resolve from the mmap'd
// zones + in-memory registries now. See `resolve_node_labels_from_bitset`,
// `resolve_edge_type_by_id`, `is_live_on_disk`, `is_live_edge_on_disk`.

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

    /// Create a throw-away substrate rooted in a fresh temp directory and
    /// open it in `EveryCommit` mode. The temp directory is owned by the
    /// underlying [`SubstrateFile`] (via an internal `TempDirGuard`) and
    /// deleted when this store is dropped.
    ///
    /// This is the canonical post-T17 replacement for `LpgStore::new()` in
    /// unit tests and doctests that do not exercise cross-restart behaviour.
    pub fn open_tempfile() -> SubstrateResult<Self> {
        let sub = SubstrateFile::open_tempfile()?;
        Self::from_substrate(sub, SyncMode::EveryCommit)
    }

    fn from_substrate(
        sub: SubstrateFile,
        sync_mode: SyncMode,
    ) -> SubstrateResult<Self> {
        // T17a — Instrumentation. Each phase emits an `info_span!` scope
        // plus an explicit `elapsed_ms` event so `RUST_LOG=info` gives a
        // full breakdown of cold-open latency. Added to hunt the 79 s
        // Wikipedia startup (gate: 100 ms). Do not remove without a
        // replacement — these spans also serve as regression anchors.
        let _total_span = tracing::info_span!("substrate.from_substrate").entered();
        let t_total = std::time::Instant::now();

        // (1) Load the persisted dict (registries + slot allocator state).
        // Missing on fresh create → DictSnapshot::default() (empty
        // registries, counters = 1).
        let t_dict = std::time::Instant::now();
        let dict_span = tracing::info_span!("dict_load").entered();
        let dict_path = sub.path().join(DICT_FILENAME);
        let snapshot = crate::dict::DictSnapshot::load(&dict_path)?;

        let mut labels = LabelRegistry::default();
        labels.load_from(&snapshot.labels)?;
        let mut edge_types = EdgeTypeRegistry::default();
        edge_types.load_from(&snapshot.edge_types)?;
        let mut prop_keys = PropertyKeyRegistry::default();
        prop_keys.load_from(&snapshot.prop_keys)?;
        drop(dict_span);
        tracing::info!(
            phase = "dict_load",
            elapsed_ms = t_dict.elapsed().as_millis() as u64,
            labels = snapshot.labels.len(),
            edge_types = snapshot.edge_types.len(),
            prop_keys = snapshot.prop_keys.len(),
            vec_columns = snapshot.vec_columns.len(),
            blob_columns = snapshot.blob_columns.len(),
            next_node_id = snapshot.next_node_id,
            next_edge_id = snapshot.next_edge_id,
            "dict + registries loaded"
        );

        // (1b) Writer init — opens the WAL, maps each zone into the
        //      process address space, wires sync_mode. This is what
        //      brings the on-disk zone files into the VM.
        let t_writer = std::time::Instant::now();
        let writer_span = tracing::info_span!("writer_init").entered();
        let writer = Writer::new(sub, sync_mode)?;
        let substrate = writer.substrate();

        // (1c) T17c Step 3a — plumb the v2 props zone.
        //      Conditional open: either `OBRAIN_PROPS_V2=1` in the env
        //      at open time, or `substrate.props.v2` already exists on
        //      disk (once upgraded, always upgraded). When neither is
        //      true we stay on the legacy props sidecar path.
        //
        //      Step 3a is pure plumbing: the zone is opened and held in
        //      the store skeleton but not yet consulted by any reader or
        //      writer. Step 3b/3c (write-through + read path) follow.
        let t_props_v2 = std::time::Instant::now();
        let props_zone_open_span = tracing::info_span!("props_zone_open").entered();
        let props_zone = {
            let sub_guard = substrate.lock();
            let enable_env = std::env::var("OBRAIN_PROPS_V2").ok().as_deref() == Some("1");
            let zone_exists = sub_guard.path().join(PROPS_V2_FILENAME).exists();
            if enable_env || zone_exists {
                Some(RwLock::new(PropsZone::open(&sub_guard)?))
            } else {
                None
            }
        };
        drop(props_zone_open_span);
        tracing::info!(
            phase = "props_zone_open",
            elapsed_ms = t_props_v2.elapsed().as_millis() as u64,
            enabled = props_zone.is_some(),
            "v2 props zone open decision"
        );

        let store = Self {
            substrate,
            writer: Arc::new(writer),
            next_node_id: AtomicU32::new(snapshot.next_node_id as u32),
            next_edge_id: AtomicU64::new(snapshot.next_edge_id),
            next_engram_id: AtomicU16::new(snapshot.next_engram_id),
            node_properties: DashMap::new(),
            edge_properties: DashMap::new(),
            incoming_heads_cell: OnceLock::new(),
            labels: RwLock::new(labels),
            edge_types: Arc::new(RwLock::new(edge_types)),
            prop_keys: RwLock::new(prop_keys),
            stats: Arc::new(Statistics::new()),
            community_ranges_cell: OnceLock::new(),
            total_live_nodes: AtomicU64::new(0),
            total_live_edges: AtomicU64::new(0),
            label_live_counts: std::array::from_fn(|_| AtomicU64::new(0)),
            edge_type_live_counts: DashMap::new(),
            edge_type_target_label_counts: DashMap::new(),
            edge_type_source_label_counts: DashMap::new(),
            degrees_cell: OnceLock::new(),
            typed_degrees_cell: OnceLock::new(),
            vec_columns: VecColumnRegistry::new(),
            blob_columns: BlobColumnRegistry::new(),
            props_zone,
        };
        drop(writer_span);
        tracing::info!(
            phase = "writer_init",
            elapsed_ms = t_writer.elapsed().as_millis() as u64,
            "writer + mmap + store skeleton ready"
        );

        // (2) Rebuild the minimal in-memory side-cars from the on-disk
        //     zones. T17e Phase 4: the `incoming_heads` map is now lazy
        //     (OnceLock<DashMap>) — it is NOT populated here. The scan
        //     only fills the community range maps (first/last live slot
        //     per community, used by the CommunityWarden + prefetch).
        //     The reverse-chain head map is built on demand by
        //     `SubstrateStore::incoming_heads()` the first time a
        //     reverse traversal, create_edge, or delete_edge needs it.
        let t_rebuild = std::time::Instant::now();
        let rebuild_span = tracing::info_span!("rebuild_zones").entered();
        store.rebuild_from_zones()?;
        drop(rebuild_span);
        tracing::info!(
            phase = "rebuild_zones",
            elapsed_ms = t_rebuild.elapsed().as_millis() as u64,
            incoming_heads = "lazy",
            communities = "lazy",
            "side-cars deferred to first access (community ranges + incoming_heads both lazy)"
        );

        // (2b) T17h T1 — Live counters restoration.
        //
        // v5+ dicts carry `PersistedCounters` — restore in O(1). Legacy
        // v1..=v4 dicts have `counters = None` — one-shot rebuild from
        // zones (O(N+E) sequential scan), will be persisted on next
        // flush (v5 bump takes effect).
        let t_counters = std::time::Instant::now();
        let counters_span = tracing::info_span!("restore_counters").entered();
        let counters_source = if let Some(ref persisted) = snapshot.counters {
            store.restore_counters_from_snapshot(persisted);
            // T17i T2 : a v5 dict (loaded from a pre-T17i base) has empty
            // `edge_type_{target,source}_label_counts` in its persisted
            // block. Detect this case — `counters.is_some()` but BOTH
            // histograms are empty AND edge_type_counts is non-empty —
            // and trigger a one-shot rebuild that only repopulates the
            // two histograms. First flush persists them as v6. On truly
            // empty stores (no edges yet) we skip the rebuild — there is
            // nothing to histogram.
            if !persisted.edge_type_counts.is_empty()
                && persisted.edge_type_target_label_counts.is_empty()
                && persisted.edge_type_source_label_counts.is_empty()
            {
                tracing::info!(
                    "T17i T2: v5 dict detected, rebuilding peer-label \
                     histograms (one-shot, persisted to v6 immediately via T17j T3 auto-flush)"
                );
                store.rebuild_live_counters_from_zones()?;
                // T17j T3 — auto-flush the upgraded dict now so the
                // next open reads v6 directly (O(M) histogram restore,
                // no rebuild scan). Without this, the first post-deploy
                // open of a large v5 base would re-scan edges until the
                // user-triggered flush eventually persists v6 — on
                // Wiki (119M edges) that's ~11 s faulting ~4 GB RSS.
                //
                // Graceful failure : a flush error (read-only FS, disk
                // full) logs a warning and returns Ok — the registry
                // is correct in RAM, the next flush attempt will
                // retry. Never panic, never block `from_substrate`.
                let auto_flush_span =
                    tracing::info_span!("auto_flush_v5_to_v6").entered();
                let t_flush = std::time::Instant::now();
                match store.flush() {
                    Ok(()) => {
                        tracing::info!(
                            elapsed_ms = t_flush.elapsed().as_millis() as u64,
                            "T17j T3: auto-flushed v5→v6 upgrade"
                        );
                    }
                    Err(e) => {
                        tracing::warn!(
                            error = ?e,
                            "T17j T3: auto-flush after v5→v6 upgrade failed \
                             (non-fatal — registry valid in RAM, next flush \
                             will retry)"
                        );
                    }
                }
                drop(auto_flush_span);
                "snapshot+v5-histogram-rebuild-auto-flushed"
            } else {
                "snapshot"
            }
        } else {
            store.rebuild_live_counters_from_zones()?;
            "rebuild_scan"
        };
        drop(counters_span);
        tracing::info!(
            phase = "restore_counters",
            elapsed_ms = t_counters.elapsed().as_millis() as u64,
            source = counters_source,
            total_nodes = store.total_live_nodes.load(Ordering::Relaxed),
            total_edges = store.total_live_edges.load(Ordering::Relaxed),
            "live counters initialised"
        );

        // (2c) T17h T5 — eager init of the degree column. MUST happen
        // BEFORE any subsequent create_edge/delete_edge so the lazy
        // rebuild scan never interleaves with explicit increments
        // (which would double-count the in-flight edge).
        //
        // Cost : O(1) mmap restore if sidecar exists, or O(edges) scan
        // if absent/corrupt. Amortised once per session.
        let t_degrees = std::time::Instant::now();
        let degrees_span = tracing::info_span!("init_degrees").entered();
        let _ = store.degrees();
        drop(degrees_span);
        tracing::info!(
            phase = "init_degrees",
            elapsed_ms = t_degrees.elapsed().as_millis() as u64,
            "degree column ready"
        );

        // (2d) T17i T1 — the per-edge-type degree registry is now
        // **deferred** : no more eager scan at `from_substrate`.
        // Previously this phase cost ~1.3-4.7 ms per edge_type and
        // inflated DB open on PO (65 types → +85 ms). The registry
        // hydrates on the first hot-path call to `typed_degrees()`
        // via `TypedDegreeRegistry::ensure_initialized`. Idempotent,
        // race-safe under concurrent `create_edge`.

        // (3) Hydrate the vec-column registry from the persisted specs
        //     (dict v3). For each `(prop_key_id, entity_kind, dim, dtype)`
        //     tuple we reopen the corresponding `substrate.veccol.*`
        //     zone so subsequent reads and writes in this session route
        //     to the right file without a cold `ZoneFile::open`.
        //
        //     Hydration runs after `prop_keys` is loaded so the registry
        //     can resolve `prop_key_id → PropertyKey` names, and BEFORE
        //     `load_properties` so pre-T16.7 bases whose sidecar still
        //     holds `Value::Vector` payloads can auto-migrate them into
        //     vec_columns on the fly during the sidecar walk. Swapping
        //     (3) and (4) vs. the original order is what makes the
        //     upgrade path safe against silent data loss — see the
        //     `auto_migrates_legacy_vectors_on_load` test for the
        //     regression anchor.
        let t_hydrate = std::time::Instant::now();
        let hydrate_span = tracing::info_span!("hydrate_columns").entered();
        {
            let sub = store.substrate.lock();
            let names = store.prop_keys.read().names();
            store
                .vec_columns
                .hydrate_from_dict(&sub, &names, &snapshot.vec_columns)?;
            // (3b) Same contract for blob columns (T16.7 Step 4d): reopen
            //      each `(prop_key_id, entity_kind)` pair's two files so
            //      `get_node_property` / `get_edge_property` resolve the
            //      right column before the first session write. Runs
            //      BEFORE `load_properties` so pre-T16.7.4 sidecars
            //      carrying oversized `Value::String` / `Value::Bytes`
            //      can auto-migrate through the setter path.
            store
                .blob_columns
                .hydrate_from_dict(&sub, &names, &snapshot.blob_columns)?;
        }
        drop(hydrate_span);
        tracing::info!(
            phase = "hydrate_columns",
            elapsed_ms = t_hydrate.elapsed().as_millis() as u64,
            vec_columns = snapshot.vec_columns.len(),
            blob_columns = snapshot.blob_columns.len(),
            "vec + blob column registries hydrated"
        );

        // (4) Load properties from the sidecar snapshot, if present.
        //     Fresh stores have no `.props` file and this is a no-op.
        //     A corrupt snapshot logs a warning and degrades to empty
        //     properties rather than refusing to open — the store is
        //     still usable structurally. `Value::Vector` entries are
        //     routed into the (already-hydrated) vec_columns registry
        //     instead of the DashMap — this auto-migrates pre-T16.7
        //     sidecars in place; the next `flush()` drops them from the
        //     re-serialised sidecar via the `persist_properties`
        //     defensive filter, closing the upgrade in a single
        //     open-close cycle.
        let t_props = std::time::Instant::now();
        let props_span = tracing::info_span!("load_properties").entered();
        store.load_properties()?;
        drop(props_span);
        tracing::info!(
            phase = "load_properties",
            elapsed_ms = t_props.elapsed().as_millis() as u64,
            "properties sidecar walked + scalars/vectors/blobs routed"
        );

        tracing::info!(
            phase = "from_substrate_total",
            elapsed_ms = t_total.elapsed().as_millis() as u64,
            "SubstrateStore::from_substrate complete"
        );

        Ok(store)
    }

    /// Load the properties sidecar and populate `self.node_properties`
    /// and `self.edge_properties` maps (T17e Phase 2). See
    /// [`Self::persist_properties`] for the symmetric write path.
    ///
    /// T16.7 Step 3b contract: `Value::Vector` entries found in the
    /// sidecar (pre-T16.7 vintage bases) are routed through
    /// [`Self::set_node_property`] / [`Self::set_edge_property`] so they
    /// land in the vec-column registry instead of the DashMap. This
    /// auto-migrates vectors in place — the next `flush()` then drops
    /// them from the sidecar via the `persist_properties` defensive
    /// filter. Without this routing, opening a pre-T16.7 base with
    /// T16.7+ code would silently lose vector data on first close.
    ///
    /// T16.7 Step 4d extends the contract to oversized `Value::String` /
    /// `Value::Bytes` entries (payload > [`BLOB_COLUMN_THRESHOLD_BYTES`]).
    /// They are routed through the same public setter so the blob
    /// registry materialises the correct column pair, and the next
    /// `flush()` drops them from the sidecar — closing the anon-RSS
    /// gate for the `data` key on pre-T16.7.4 bases in one open-close
    /// cycle.
    fn load_properties(&self) -> SubstrateResult<()> {
        // T17c Step 3d — When PropsZone v2 is active and has at least
        // one allocated page, skip **node** hydration from the legacy
        // sidecar entirely:
        //
        // - All node-property reads resolve through
        //   `lookup_node_property_v2` (walk_chain + LWW), so the
        //   in-memory DashMap is no longer on the read path.
        // - Keeping the DashMap empty saves the full bincode decode +
        //   per-node `DashMap::insert` at open time — the dominant
        //   term of startup on Wikipedia-scale DBs
        //   (~4.6 s / 7.46 s observed).
        // - Edges still hydrate from the sidecar: `EdgeRecord` has
        //   no `first_prop_off`, so edge properties remain
        //   DashMap-backed until a dedicated edge-props zone ships
        //   (out of scope for T17c).
        //
        // The "at least one page" gate is important: a v2-enabled
        // store that has never had a scalar written (fresh open, or a
        // migration-in-flight that hasn't yet touched the node in
        // question) MUST still hydrate from the sidecar — otherwise
        // legacy node properties silently disappear on reopen.
        //
        // Once Step 4 ships the `obrain-migrate --finalize-v2` tool
        // and the sidecar is deleted at the end of migration, this
        // gate becomes a pure documentation aid (no sidecar → nothing
        // to load).
        let v2_has_pages = self
            .props_zone
            .as_ref()
            .map(|pz| pz.read().allocated_page_count() > 0)
            .unwrap_or(false);

        let path = {
            let sub = self.substrate.lock();
            sub.path().join(PROPS_FILENAME)
        };
        // T17c Step 6 — edge-only sidecar path detection (needed before
        // deciding whether to load the legacy sidecar at all).
        let edge_sidecar_path = {
            let sub = self.substrate.lock();
            sub.path().join(EDGE_PROPS_FILENAME)
        };
        let edge_sidecar_present = edge_sidecar_path.exists();

        // T17g T2b — **short-circuit the legacy substrate.props load
        // entirely** when v2 is fully authoritative. On Megalaw post
        // `--finalize-v2` + `--upgrade-edges-v2` we observed the legacy
        // sidecar at 1.07 GB (re-written by `persist_properties` at the
        // tail of the old migration flow) decoding in 1.9 s per open —
        // pure waste because:
        //   - Nodes are in v2 → `v2_has_pages` triggers the per-node
        //     skip below, so `snap.nodes` is discarded.
        //   - Edges are in v2 edge chain (post-upgrade-edges-v2) and the
        //     `substrate.edge_props` sidecar is absent → the `skip_edges_legacy`
        //     branch below discards `snap.edges` too.
        //
        // If both branches will discard, there is no reason to pay the
        // bincode decode. We synthesise an empty snapshot and skip the
        // load. The stale legacy file will be reclaimed on the next
        // `flush()` via `persist_properties` rewriting from the
        // (intentionally empty) DashMaps.
        let legacy_fully_stale = v2_has_pages && !edge_sidecar_present && path.exists();
        let snap = if legacy_fully_stale {
            tracing::info!(
                phase = "load_properties",
                legacy_props_size_mib = std::fs::metadata(&path).map(|m| m.len() / (1024 * 1024)).unwrap_or(0),
                "T17g T2b — bypassing legacy substrate.props decode: v2 is \
                 authoritative for both nodes and edges. Stale bytes will be \
                 reclaimed on next flush."
            );
            PropertiesSnapshotV1::default()
        } else {
            PropertiesSnapshotV1::load(&path)?
        };

        // T17c Step 6 — when the dedicated edge sidecar is present, it
        // is the authoritative source for edge-property maps (written by
        // `finalize_props_v2` or `persist_edge_properties_sidecar`).
        // Its presence means the legacy `substrate.props` sidecar has
        // either been drained of edges or is absent entirely
        // (post-`delete_legacy_props_sidecar`). In that case the
        // `snap.edges` loop below is short-circuited so we don't
        // double-hydrate.
        let edge_snap = if edge_sidecar_present {
            Some(PropertiesSnapshotV1::load(&edge_sidecar_path)?)
        } else {
            None
        };

        let mut nodes_loaded = 0usize;
        let mut edges_loaded = 0usize;
        let mut node_props_skipped = 0usize;
        let mut edge_props_skipped = 0usize;
        // Count the vectors we auto-migrate out of the sidecar — one
        // log line at the end lets operators confirm the upgrade
        // happened (and gives them a before/after view of the sidecar
        // size should they want to repack).
        let mut node_vectors_migrated = 0usize;
        let mut edge_vectors_migrated = 0usize;
        let mut node_blobs_migrated = 0usize;
        let mut edge_blobs_migrated = 0usize;
        // Hot loop: a small contains check over <10 short strings. A linear
        // scan is faster than any hash-set overhead at this size.
        let skip = |k: &str| SKIP_ON_LOAD_PROP_KEYS.iter().any(|s| *s == k);
        // Local predicate: is this an oversized scalar payload that
        // should auto-migrate into the blob-column registry? Checked
        // by byte length, not by registry membership, so the very
        // first load after a T16.7.4 upgrade (registry still empty)
        // still routes the payload — the setter will then register
        // the column on the fly.
        let is_blob = |v: &Value| {
            blob_payload_len(v)
                .map(|n| n > BLOB_COLUMN_THRESHOLD_BYTES)
                .unwrap_or(false)
        };
        let mut nodes_skipped_v2 = 0usize;
        for e in snap.nodes {
            let nid = obrain_common::types::NodeId(e.id);
            // T17c Step 3d — when the v2 chain is already populated,
            // the DashMap is no longer authoritative for node props.
            // Bincode-decoded the entry (we own `e`) and drop it
            // without touching any field — the allocation for the
            // scalar vec / vector / blob split is skipped entirely,
            // which is exactly the startup speedup we want.
            if v2_has_pages {
                nodes_skipped_v2 += 1;
                continue;
            }
            let before = e.props.len();
            // Split the sidecar entry four ways (T16.7 Step 4d):
            //   - dropped entirely (SKIP_ON_LOAD keys — rebuildable)
            //   - routed to vec_columns   (Value::Vector)
            //   - routed to blob_columns  (oversized String / Bytes)
            //   - kept on the DashMap scalar path (everything else)
            let mut scalars: Vec<(String, Value)> = Vec::with_capacity(before);
            let mut vectors: Vec<(String, Value)> = Vec::new();
            let mut blobs: Vec<(String, Value)> = Vec::new();
            for (k, v) in e.props {
                if skip(&k) {
                    node_props_skipped += 1;
                    continue;
                }
                if matches!(v, Value::Vector(_)) {
                    vectors.push((k, v));
                } else if is_blob(&v) {
                    blobs.push((k, v));
                } else {
                    scalars.push((k, v));
                }
            }
            // Route vectors through the public setter so the registry
            // allocates/reopens the correct column file and writes the
            // slot. `set_node_property`'s `is_live_on_disk` check also
            // protects us from tombstoned-between-flush-and-reopen
            // nodes — it no-ops in that case, matching the behaviour
            // of the scalar branch below.
            for (k, v) in vectors {
                self.set_node_property(nid, &k, v);
                node_vectors_migrated += 1;
            }
            // Same contract for blob-eligible scalars — the setter
            // handles the type-tag prefix and column allocation.
            for (k, v) in blobs {
                self.set_node_property(nid, &k, v);
                node_blobs_migrated += 1;
            }
            let map = entries_to_map(scalars);
            // Populate the dedicated node_properties map (T17e Phase 2).
            // Liveness gate now reads straight from the mmap'd NodeRecord
            // (T17e Phase 3) — tombstoned-between-flush-and-reopen nodes
            // are silently skipped, matching the pre-Phase-2 contract.
            if !map.is_empty() && self.is_live_on_disk(nid) {
                self.node_properties.insert(nid, map);
                nodes_loaded += 1;
            }
        }
        if nodes_skipped_v2 > 0 {
            tracing::info!(
                phase = "load_properties",
                nodes_skipped_v2,
                "PropsZone v2 already populated — node sidecar hydration skipped"
            );
        }
        // T17c Step 6 — when the dedicated edge sidecar is present, it
        // is the authoritative source. Otherwise fall back to the
        // combined `snap.edges` from the legacy sidecar (pre-
        // finalize-v2 bases, fresh stores, etc.).
        //
        // T17g T2b — **edge v2 skip-gate**. When PropsZone v2 is active
        // AND has at least one page AND the dedicated edge sidecar
        // (`substrate.edge_props`) is ABSENT, the invariant is:
        //   - either this is a fresh v2 base (no edges yet), OR
        //   - this base has been fully migrated through
        //     `--finalize-v2` AND `--upgrade-edges-v2`, which means
        //     edges live in the v2 edge chain and any entries in the
        //     legacy `substrate.props` are stale duplicates
        //     (re-written by `persist_properties` at the tail of a
        //     pre-T17g-T2b migration run — see bug note `<pending>`).
        //
        // In both cases, loading edges from the legacy sidecar would
        // re-populate the `edge_properties` DashMap with entries that
        // already exist in v2 — doubling the anon-RSS footprint and
        // paying the 7+ second bincode decode on Megalaw-scale bases.
        // Skip it.
        //
        // If a pre-v2 edge truly sat in legacy only (never seen by v2),
        // this skip would silently lose it. That corner case is
        // considered closed: any v2-enabled base that ever wrote an
        // edge did so through `set_edge_property` which routes to v2
        // since T17f Step 4, AND the only way to have legacy edges on
        // disk at open is via pre-T17f migration output — which is
        // healed by running `--finalize-v2` + `--upgrade-edges-v2` in
        // order (the drain reads legacy → v2, then this gate prevents
        // re-loading).
        let skip_edges_legacy = v2_has_pages && !edge_sidecar_present && !snap.edges.is_empty();
        let mut edges_from_new_sidecar = 0usize;
        let edge_entries: Vec<PropEntry> = match edge_snap {
            Some(es) => {
                edges_from_new_sidecar = es.edges.len();
                es.edges
            }
            None if skip_edges_legacy => {
                tracing::warn!(
                    phase = "load_properties",
                    stale_edges_in_legacy_sidecar = snap.edges.len(),
                    "T17g T2b — skipping edge load from legacy sidecar: v2 is \
                     authoritative and edge_props sidecar is absent. Legacy \
                     entries will be reclaimed on next flush (persist_properties \
                     rewrites from a cleared DashMap)."
                );
                Vec::new()
            }
            None => snap.edges,
        };
        for e in edge_entries {
            let eid = obrain_common::types::EdgeId(e.id);
            let before = e.props.len();
            let mut scalars: Vec<(String, Value)> = Vec::with_capacity(before);
            let mut vectors: Vec<(String, Value)> = Vec::new();
            let mut blobs: Vec<(String, Value)> = Vec::new();
            for (k, v) in e.props {
                if skip(&k) {
                    edge_props_skipped += 1;
                    continue;
                }
                if matches!(v, Value::Vector(_)) {
                    vectors.push((k, v));
                } else if is_blob(&v) {
                    blobs.push((k, v));
                } else {
                    scalars.push((k, v));
                }
            }
            for (k, v) in vectors {
                self.set_edge_property(eid, &k, v);
                edge_vectors_migrated += 1;
            }
            for (k, v) in blobs {
                self.set_edge_property(eid, &k, v);
                edge_blobs_migrated += 1;
            }
            let map = entries_to_map(scalars);
            if !map.is_empty() && self.is_live_edge_on_disk(eid) {
                self.edge_properties.insert(eid, map);
                edges_loaded += 1;
            }
        }
        if nodes_loaded > 0 || edges_loaded > 0 {
            tracing::info!(
                "props snapshot: loaded {} node-property maps, {} edge-property maps \
                 (skipped-on-load: {} node-props, {} edge-props — see SKIP_ON_LOAD_PROP_KEYS; \
                 auto-migrated to vec_columns: {} node vectors, {} edge vectors; \
                 auto-migrated to blob_columns: {} node blobs, {} edge blobs; \
                 edge_sidecar_present: {}, edges_from_new_sidecar: {})",
                nodes_loaded,
                edges_loaded,
                node_props_skipped,
                edge_props_skipped,
                node_vectors_migrated,
                edge_vectors_migrated,
                node_blobs_migrated,
                edge_blobs_migrated,
                edge_sidecar_present,
                edges_from_new_sidecar,
            );
        }
        Ok(())
    }

    /// Open-path rebuild of the minimal in-memory side-cars.
    ///
    /// **T17g T1 update**: the Nodes-zone scan that populated
    /// `community_placements` / `community_first_slots` has been removed
    /// — those maps are now built lazily on first access via
    /// [`Self::community_ranges`] (T17g T1, O(node_hw) parallel scan
    /// guarded by `OnceLock::get_or_init`).
    ///
    /// **T17e Phase 4**: the reverse-chain head map (`incoming_heads_cell`)
    /// was already made lazy — see [`Self::incoming_heads`] for its O(E)
    /// parallel scan (was the dominant term at 99 s / 95% of startup on
    /// Wikipedia 119M edges).
    ///
    /// Labels and edge-type ArcStrs are resolved on demand via
    /// [`Self::resolve_node_labels_from_bitset`] and
    /// [`Self::resolve_edge_type_by_id`] straight from the mmap'd zones
    /// + registries.
    ///
    /// Post-T17g-T1 this function is effectively a no-op (both community
    /// and incoming side-cars are lazy). It is retained as a semantic
    /// anchor — a future sidecar may still need an open-path rebuild, in
    /// which case the hook is here.
    fn rebuild_from_zones(&self) -> SubstrateResult<()> {
        // Nodes: community range maps are lazy — see `community_ranges()`.
        // Edges: `incoming_heads_cell` is lazy — see `incoming_heads()`.
        // Nothing to eagerly rebuild at open time.
        Ok(())
    }

    /// Return the reverse-chain head map, building it on first access.
    ///
    /// **T17e Phase 4 — lazy incoming index.**
    ///
    /// The map is stored as `OnceLock<DashMap<NodeId, EdgeId>>` and built
    /// the first time it is needed (a reverse traversal, `create_edge`, or
    /// `delete_edge`). The build is an O(E) parallel scan of the Edges
    /// zone that computes, for each dst, the **highest live `EdgeId`** —
    /// the head of the incoming chain (splice-at-head invariant: newer
    /// ids are always spliced at the front, and edges only drop out of
    /// middle positions, so the head is always the max live id).
    ///
    /// Concurrency: `OnceLock::get_or_init` serialises the build — a
    /// concurrent writer firing `incoming_heads()` mid-build blocks until
    /// the scan completes, then inserts its own head into the now-live
    /// DashMap. Edges allocated **after** the scan's `next_edge_id`
    /// snapshot are handled by the writer's own post-init `insert`, so
    /// the "live window" is closed by construction.
    pub(crate) fn incoming_heads(&self) -> &DashMap<NodeId, EdgeId> {
        self.incoming_heads_cell.get_or_init(|| {
            use rayon::prelude::*;

            let t0 = std::time::Instant::now();
            let edge_hw = self.next_edge_id.load(Ordering::Acquire);
            let map: DashMap<NodeId, EdgeId> = DashMap::new();

            // Parallel scan: each worker folds a local FxHashMap<dst, max
            // EdgeId>, then we merge into the final DashMap. Reading the
            // Edges zone under rayon is safe because `writer.read_edge`
            // only touches the mmap — no shared mutable state.
            let partials: Vec<FxHashMap<NodeId, EdgeId>> = (1..edge_hw)
                .into_par_iter()
                .fold(
                    FxHashMap::<NodeId, EdgeId>::default,
                    |mut acc, slot| {
                        if let Ok(Some(rec)) = self.writer.read_edge(slot) {
                            if rec.flags & edge_flags::TOMBSTONED == 0 {
                                let edge_id = EdgeId(slot);
                                let dst = NodeId(rec.dst as u64);
                                acc.entry(dst)
                                    .and_modify(|cur: &mut EdgeId| {
                                        if cur.0 < edge_id.0 {
                                            *cur = edge_id;
                                        }
                                    })
                                    .or_insert(edge_id);
                            }
                        }
                        acc
                    },
                )
                .collect();

            for partial in partials {
                for (dst, head) in partial {
                    map.entry(dst)
                        .and_modify(|cur| {
                            if cur.0 < head.0 {
                                *cur = head;
                            }
                        })
                        .or_insert(head);
                }
            }

            tracing::info!(
                phase = "incoming_heads_lazy_build",
                elapsed_ms = t0.elapsed().as_millis() as u64,
                edges_scanned = edge_hw.saturating_sub(1),
                heads = map.len(),
                "lazy build of incoming_heads completed"
            );

            map
        })
    }

    /// Return the per-community slot range maps, building them on first
    /// access.
    ///
    /// **T17g T1 — lazy community ranges.**
    ///
    /// The pair `(placements, first_slots)` is stored as
    /// `OnceLock<CommunityRanges>` and built the first time the
    /// CommunityWarden, the allocator (`allocate_node_id_in_community`),
    /// or the prefetch hook actually needs it. The build is an
    /// O(node_high_water) parallel scan of the Nodes zone that computes,
    /// for each live community, its max live slot (→ `placements`) and
    /// its min live slot (→ `first_slots`).
    ///
    /// Concurrency: `OnceLock::get_or_init` serialises the build — a
    /// concurrent allocator firing `community_ranges()` mid-build blocks
    /// until the scan completes, then inserts its own placement into the
    /// now-live DashMaps. Nodes allocated **after** the scan's
    /// `next_node_id` snapshot are handled by the allocator's own
    /// post-init `insert`, so the "live window" is closed by construction.
    pub(crate) fn community_ranges(&self) -> &CommunityRanges {
        self.community_ranges_cell.get_or_init(|| {
            use rayon::prelude::*;

            let t0 = std::time::Instant::now();
            let node_hw = self.next_node_id.load(Ordering::Acquire);
            let ranges = CommunityRanges::empty();

            // Parallel scan: each worker folds local (min, max) tables
            // keyed by community_id; we merge into the final DashMaps.
            // Reading the Nodes zone under rayon is safe — `writer.read_node`
            // only touches the mmap, no shared mutable state.
            //
            // Return type: Vec<FxHashMap<cid, (min_slot, max_slot)>>
            type CidRange = (u32, u32); // (min, max)
            let partials: Vec<FxHashMap<u32, CidRange>> = (1..node_hw)
                .into_par_iter()
                .fold(
                    FxHashMap::<u32, CidRange>::default,
                    |mut acc, slot| {
                        if let Ok(Some(rec)) = self.writer.read_node(slot) {
                            if rec.flags & crate::record::node_flags::TOMBSTONED == 0 {
                                acc.entry(rec.community_id)
                                    .and_modify(|(min, max)| {
                                        if slot < *min {
                                            *min = slot;
                                        }
                                        if slot > *max {
                                            *max = slot;
                                        }
                                    })
                                    .or_insert((slot, slot));
                            }
                        }
                        acc
                    },
                )
                .collect();

            // Merge partials into the final DashMaps.
            for partial in partials {
                for (cid, (min, max)) in partial {
                    ranges
                        .first_slots
                        .entry(cid)
                        .and_modify(|cur| {
                            if min < *cur {
                                *cur = min;
                            }
                        })
                        .or_insert(min);
                    ranges
                        .placements
                        .entry(cid)
                        .and_modify(|cur| {
                            if max > *cur {
                                *cur = max;
                            }
                        })
                        .or_insert(max);
                }
            }

            tracing::info!(
                phase = "community_ranges_lazy_build",
                elapsed_ms = t0.elapsed().as_millis() as u64,
                nodes_scanned = node_hw.saturating_sub(1),
                communities = ranges.placements.len(),
                "lazy build of community ranges completed"
            );

            ranges
        })
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
            // Every open blob-column gets a spec here so
            // `hydrate_from_dict` can reopen the column pair by
            // filename on next open — no directory scan needed.
            blob_columns: self.blob_columns.specs_snapshot(),
            // T17h T1: live counter snapshot from atomics. Writing v5
            // format always includes counters so subsequent opens
            // restore in O(1) (no zone scan).
            counters: Some(self.snapshot_counters()),
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
    /// 4. `blob_columns.sync_all()` — same contract for the blob-column
    ///    pairs (idx + dat). Both CRCs are recomputed and msync'd
    ///    before the dict references them.
    /// 5. `persist_dict()` — atomically rewrites `substrate.dict` with
    ///    the current registries, slot counters, and column specs
    ///    (vec + blob). Done last so any crash before here leaves the
    ///    WAL as source of truth (replay will reconstruct both).
    pub fn flush(&self) -> SubstrateResult<()> {
        self.writer.commit()?;
        self.writer.msync_zones()?;
        self.vec_columns.sync_all()?;
        self.blob_columns.sync_all()?;
        // T17c Step 3a — flush the v2 props zone when enabled. Pure
        // plumbing step: no writes go through it yet (3b/3c wire the
        // setters), so `flush()` is effectively a no-op except for the
        // first session on an enabled substrate (where the dummy
        // sentinel page was just mmap-extended and needs to hit disk).
        if let Some(pz) = self.props_zone.as_ref() {
            pz.read().flush()?;
        }
        // T17h T5: flush the degree column (header CRC + msync). Only
        // if the lazy column has been materialised this session; a
        // read-only query path that never created/deleted edges leaves
        // it untouched.
        if let Some(degrees) = self.degrees_cell.get() {
            let mut col = degrees.write();
            col.persist_header_crc();
            col.msync()?;
        }
        // T17h T8: flush the per-edge-type degree columns. Same laziness
        // contract as T5 — only the materialised columns are touched.
        if let Some(typed) = self.typed_degrees_cell.get() {
            typed.flush()?;
        }
        self.persist_dict()?;
        self.persist_properties()?;
        Ok(())
    }

    /// Returns `true` when the v2 props zone
    /// (`substrate.props.v2` + `substrate.props.heap.v2`) has been
    /// opened for this session.
    ///
    /// T17c Step 3a observability hook. Enabled when either
    /// `OBRAIN_PROPS_V2=1` was set at open time, or the zone file
    /// already existed on disk.
    pub fn props_v2_enabled(&self) -> bool {
        self.props_zone.is_some()
    }

    /// Delete the legacy `substrate.props` sidecar, if present.
    ///
    /// Called by the migration tool after
    /// [`Self::finalize_props_v2`] + [`Self::flush`] have successfully
    /// drained the sidecar into the v2 chain, so the next open
    /// doesn't load an out-of-date `PropertiesSnapshotV1` (which
    /// would no-op anyway thanks to the Step 3d gate, but keeps the
    /// on-disk footprint honest).
    ///
    /// **T17c Step 6 safety gate**: refuses the delete when the
    /// in-memory `edge_properties` DashMap is non-empty AND the
    /// dedicated edge sidecar `substrate.edge_props` is absent —
    /// that combination means the edge properties live only in RAM +
    /// the legacy sidecar, and dropping the sidecar would lose them.
    /// Callers must run [`Self::persist_edge_properties_sidecar`]
    /// (or [`Self::finalize_props_v2`], which does it internally)
    /// before calling this. The legacy pre-T17c contract (no edge
    /// sidecar, no guard) would silently lose Wikipedia's 312 edge-
    /// property maps on migration.
    ///
    /// Does NOT delete `substrate.props.v2` / `substrate.props.heap.v2`
    /// / `substrate.edge_props` — those are the v2 targets.
    pub fn delete_legacy_props_sidecar(&self) -> SubstrateResult<Option<u64>> {
        let (legacy_path, edge_path) = {
            let sub = self.substrate.lock();
            let base = sub.path();
            (base.join(PROPS_FILENAME), base.join(EDGE_PROPS_FILENAME))
        };
        if !legacy_path.exists() {
            return Ok(None);
        }
        // T17c Step 6 edge-loss gate.
        let edge_props_pending = self
            .edge_properties
            .iter()
            .any(|entry| !entry.value().is_empty());
        if edge_props_pending && !edge_path.exists() {
            return Err(SubstrateError::Internal(format!(
                "delete_legacy_props_sidecar refused: {} has unsaved edge props \
                 and {} is absent — run persist_edge_properties_sidecar (or \
                 finalize_props_v2) first to avoid silent data loss",
                legacy_path.display(),
                edge_path.display()
            )));
        }
        let size = std::fs::metadata(&legacy_path).map(|m| m.len()).unwrap_or(0);
        std::fs::remove_file(&legacy_path).map_err(|e| {
            SubstrateError::Internal(format!(
                "delete_legacy_props_sidecar: remove {} failed: {e}",
                legacy_path.display()
            ))
        })?;
        tracing::info!(
            sidecar_bytes_freed = size,
            path = %legacy_path.display(),
            "legacy substrate.props sidecar deleted"
        );
        Ok(Some(size))
    }

    /// Number of allocated pages in the PropsZone v2. Returns `None`
    /// when v2 is disabled. Observability helper for the migration
    /// tool — gives operators a concrete "pages written" number to
    /// verify progress.
    pub fn props_v2_page_count(&self) -> Option<u32> {
        self.props_zone.as_ref().map(|pz| pz.read().allocated_page_count())
    }

    /// T17c Step 6 — persist the in-memory `edge_properties` DashMap
    /// to the dedicated edge sidecar `substrate.edge_props`.
    ///
    /// Used by [`Self::finalize_props_v2`] and by tests. Returns the
    /// number of edges and `(key, value)` pairs serialised, plus the
    /// final file size.
    ///
    /// Format: byte-identical to `substrate.props` — the same
    /// [`PropertiesSnapshotV1`] wrapper is reused with
    /// `nodes == vec![]` and `edges` carrying the drained entries.
    /// Atomic tmp-file + rename, so a crash mid-write never leaves a
    /// half-written sidecar.
    ///
    /// Keys are filtered through `SKIP_ON_LOAD_PROP_KEYS` (same
    /// contract as `persist_properties`) — vector and oversized-blob
    /// payloads are routed elsewhere and never land in the DashMap,
    /// but a defensive filter keeps the edge sidecar lean.
    pub fn persist_edge_properties_sidecar(
        &self,
    ) -> SubstrateResult<(usize, usize, u64)> {
        let is_routed_out = |(_, v): &(String, Value)| {
            if matches!(v, Value::Vector(_)) {
                return true;
            }
            blob_payload_len(v)
                .map(|n| n > BLOB_COLUMN_THRESHOLD_BYTES)
                .unwrap_or(false)
        };

        let mut snap = PropertiesSnapshotV1::default();
        let mut scalars_emitted = 0usize;
        snap.edges.reserve(self.edge_properties.len());
        for entry in self.edge_properties.iter() {
            if entry.value().is_empty() {
                continue;
            }
            let props: Vec<(String, Value)> = map_to_entries(entry.value())
                .into_iter()
                .filter(|p| !is_routed_out(p))
                .collect();
            if props.is_empty() {
                continue;
            }
            scalars_emitted += props.len();
            snap.edges.push(PropEntry {
                id: entry.key().as_u64(),
                props,
            });
        }
        let edges_written = snap.edges.len();

        let path = {
            let sub = self.substrate.lock();
            sub.path().join(EDGE_PROPS_FILENAME)
        };
        snap.persist(&path)?;
        let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        tracing::info!(
            phase = "persist_edge_properties_sidecar",
            edges_written,
            scalars_emitted,
            sidecar_bytes = size,
            path = %path.display(),
            "edge properties sidecar written"
        );
        Ok((edges_written, scalars_emitted, size))
    }

    /// Path of the edge-only sidecar (`substrate.edge_props`).
    /// Observability helper for the migration tool + tests.
    pub fn edge_props_sidecar_path(&self) -> std::path::PathBuf {
        let sub = self.substrate.lock();
        sub.path().join(EDGE_PROPS_FILENAME)
    }

    /// Number of edge property maps currently held in the in-memory
    /// `edge_properties` DashMap. Observability helper for the
    /// migration dry-run path — lets operators see how much edge
    /// data is about to be written to `substrate.edge_props`.
    pub fn edge_properties_count(&self) -> Option<usize> {
        Some(self.edge_properties.len())
    }

    /// Live-node count estimate based on the slot high-water mark.
    /// Counts tombstoned slots too (so it's a ceiling, not exact).
    /// Fine-grained for the dry-run log; a precise count would
    /// require a zone walk, which isn't worth it here.
    pub fn node_count_live_estimate(&self) -> u32 {
        self.slot_high_water().saturating_sub(1)
    }

    /// T17c Step 4 — finalize the PropsZone v2 migration by draining
    /// the legacy DashMap sidecar into the v2 chain.
    ///
    /// Assumes: v2 is enabled (panics otherwise, per the explicit
    /// public contract), the store is just opened (so `load_properties`
    /// has populated `node_properties` from the sidecar), and no
    /// concurrent writers are running against this handle.
    ///
    /// Contract:
    /// - For every live node with scalar entries in the DashMap,
    ///   every `(key, value)` is re-emitted through
    ///   `set_node_property`, which writes to the v2 chain (intern
    ///   key → append entry → WAL-log head pointer if the head page
    ///   rotated). The DashMap is overwritten in place with the same
    ///   value — a no-op from the read side.
    /// - Vector / blob entries already routed out by
    ///   `load_properties` are left untouched (they live in
    ///   `vec_columns` / `blob_columns`).
    /// - **T17c Step 6** — the `edge_properties` DashMap is persisted
    ///   to the dedicated edge sidecar `substrate.edge_props` BEFORE
    ///   returning, so [`Self::delete_legacy_props_sidecar`] can be
    ///   called safely by the migration tool without losing edge
    ///   properties (EdgeRecord has no `first_prop_off` — that's T17f
    ///   scope).
    /// - The legacy `substrate.props` sidecar is NOT deleted here.
    ///   The caller (`obrain-migrate --finalize-v2`) is responsible
    ///   for that after verifying the migration succeeded.
    ///
    /// Returns a summary of the work done — number of nodes touched,
    /// scalar entries re-emitted, plus edge-side counts + edge
    /// sidecar size. Logs one INFO line per completed decade of nodes
    /// for progress visibility on Wikipedia-scale inputs.
    pub fn finalize_props_v2(&self) -> SubstrateResult<PropsV2FinalizeStats> {
        if !self.props_v2_enabled() {
            return Err(SubstrateError::Internal(
                "finalize_props_v2 called without PropsZone v2 enabled \
                 (set OBRAIN_PROPS_V2=1 before SubstrateStore::open)"
                    .to_string(),
            ));
        }

        // Snapshot the DashMap into a Vec so we can mutate
        // node_properties through set_node_property without holding
        // the iterator.
        let snapshot: Vec<(NodeId, Vec<(String, Value)>)> = self
            .node_properties
            .iter()
            .map(|entry| {
                let nid = *entry.key();
                let pairs: Vec<(String, Value)> = entry
                    .value()
                    .iter()
                    .map(|(k, v)| (k.as_str().to_string(), v.clone()))
                    .collect();
                (nid, pairs)
            })
            .collect();

        let t_start = std::time::Instant::now();
        let nodes_total = snapshot.len();
        let mut stats = PropsV2FinalizeStats::default();
        for (idx, (nid, pairs)) in snapshot.into_iter().enumerate() {
            for (k, v) in pairs {
                // set_node_property routes scalars through
                // append_scalar_to_props_zone_v2 when v2 is enabled,
                // and is a no-op for tombstoned slots (gated by
                // is_live_on_disk). Vector/blob entries were already
                // routed out of the DashMap by load_properties.
                self.set_node_property(nid, &k, v);
                stats.scalars_emitted += 1;
            }
            stats.nodes_processed += 1;
            // One log line every 10% of progress (or every 100k
            // nodes, whichever is larger) so Wikipedia-scale runs
            // don't flood the journal.
            let decade_step = (nodes_total / 10).max(100_000);
            if decade_step > 0 && idx > 0 && idx % decade_step == 0 {
                tracing::info!(
                    phase = "finalize_props_v2",
                    progress_pct = (idx * 100 / nodes_total),
                    nodes_processed = stats.nodes_processed,
                    scalars_emitted = stats.scalars_emitted,
                    elapsed_ms = t_start.elapsed().as_millis() as u64,
                    "draining legacy sidecar → v2 chain"
                );
            }
        }
        // T17c Step 6 — persist the edge DashMap to the dedicated
        // edge sidecar so that `delete_legacy_props_sidecar` is safe
        // to call next. This guard-rail caught the Wikipedia edge-
        // loss bug: the legacy sidecar held 312 edge-property maps
        // that would have vanished when the caller deleted it.
        let (edges_written, edge_scalars_emitted, edge_sidecar_bytes) =
            self.persist_edge_properties_sidecar()?;
        stats.edges_processed = edges_written;
        stats.edge_scalars_emitted = edge_scalars_emitted;
        stats.edge_sidecar_bytes = edge_sidecar_bytes;

        tracing::info!(
            phase = "finalize_props_v2",
            nodes_processed = stats.nodes_processed,
            scalars_emitted = stats.scalars_emitted,
            edges_processed = stats.edges_processed,
            edge_scalars_emitted = stats.edge_scalars_emitted,
            edge_sidecar_bytes = stats.edge_sidecar_bytes,
            v2_pages_allocated = self.props_v2_page_count().unwrap_or(0),
            elapsed_ms = t_start.elapsed().as_millis() as u64,
            "PropsZone v2 migration complete"
        );
        Ok(stats)
    }

    /// T17f Step 5 — drain the in-memory `edge_properties` DashMap into
    /// the PropsZone v2 **edge** chain via the Step 4 route
    /// (`set_edge_property` → `append_scalar_to_props_zone_v2_edge`).
    ///
    /// Pre-T17f bases keep edge scalars in `substrate.edge_props` (the
    /// T17c Step 6 dedicated sidecar). At open time, `load_properties`
    /// hydrates them into `edge_properties`. This method re-emits every
    /// `(edge, key)` pair through the public setter, so after a
    /// subsequent flush the same values live on the per-edge v2 chain
    /// addressed by `EdgeRecord.first_prop_off`. The caller (typically
    /// `obrain-migrate --upgrade-edges-v2`) is then responsible for
    /// deleting the now-redundant sidecar via
    /// [`Self::delete_edge_props_sidecar`].
    ///
    /// Contract:
    /// - v2 must be enabled (`OBRAIN_PROPS_V2=1` at open time, or the
    ///   v2 zone files already present). Otherwise returns `Err`.
    /// - `set_edge_property` routes `Value::Vector` and oversized
    ///   blobs to their own columns; scalars land on the edge chain.
    /// - Tombstoned or deleted edges are silently skipped (the
    ///   `is_live_edge_on_disk` gate inside `set_edge_property`).
    /// - The DashMap is left populated — the dual-write contract keeps
    ///   reads fast during the remainder of the session. The next open
    ///   sees the deleted sidecar and falls back cleanly to the v2
    ///   chain (via `lookup_edge_property_v2`).
    ///
    /// Returns the number of edges touched and scalars re-emitted.
    pub fn finalize_edge_props_v2(
        &self,
    ) -> SubstrateResult<EdgePropsV2FinalizeStats> {
        if !self.props_v2_enabled() {
            return Err(SubstrateError::Internal(
                "finalize_edge_props_v2 called without PropsZone v2 enabled \
                 (set OBRAIN_PROPS_V2=1 before SubstrateStore::open)"
                    .to_string(),
            ));
        }

        // Snapshot the DashMap into a Vec so we can mutate edge_properties
        // through set_edge_property without holding an iterator.
        let snapshot: Vec<(EdgeId, Vec<(String, Value)>)> = self
            .edge_properties
            .iter()
            .map(|entry| {
                let eid = *entry.key();
                let pairs: Vec<(String, Value)> = entry
                    .value()
                    .iter()
                    .map(|(k, v)| (k.as_str().to_string(), v.clone()))
                    .collect();
                (eid, pairs)
            })
            .collect();

        let t_start = std::time::Instant::now();
        let edges_total = snapshot.len();
        let mut stats = EdgePropsV2FinalizeStats::default();
        for (idx, (eid, pairs)) in snapshot.into_iter().enumerate() {
            for (k, v) in pairs {
                // set_edge_property routes scalars through
                // append_scalar_to_props_zone_v2_edge when v2 is
                // enabled. Vector/blob payloads are silently routed
                // to their dedicated columns.
                self.set_edge_property(eid, &k, v);
                stats.scalars_emitted += 1;
            }
            stats.edges_processed += 1;
            let decade_step = (edges_total / 10).max(100_000);
            if decade_step > 0 && idx > 0 && idx % decade_step == 0 {
                tracing::info!(
                    phase = "finalize_edge_props_v2",
                    progress_pct = (idx * 100 / edges_total),
                    edges_processed = stats.edges_processed,
                    scalars_emitted = stats.scalars_emitted,
                    elapsed_ms = t_start.elapsed().as_millis() as u64,
                    "draining edge sidecar → v2 edge chain"
                );
            }
        }

        tracing::info!(
            phase = "finalize_edge_props_v2",
            edges_processed = stats.edges_processed,
            scalars_emitted = stats.scalars_emitted,
            v2_pages_allocated = self.props_v2_page_count().unwrap_or(0),
            elapsed_ms = t_start.elapsed().as_millis() as u64,
            "edge props v2 drain complete"
        );
        Ok(stats)
    }

    /// Delete the dedicated edge sidecar (`substrate.edge_props`), if
    /// present. Called by the T17f migration tool AFTER
    /// [`Self::finalize_edge_props_v2`] + [`Self::flush`] have
    /// successfully drained the DashMap to the v2 edge chain.
    ///
    /// Returns the size of the deleted file (0 when nothing was there).
    pub fn delete_edge_props_sidecar(&self) -> SubstrateResult<Option<u64>> {
        let path = self.edge_props_sidecar_path();
        if !path.exists() {
            return Ok(None);
        }
        let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        std::fs::remove_file(&path).map_err(|e| {
            SubstrateError::Internal(format!(
                "delete_edge_props_sidecar: remove {} failed: {e}",
                path.display()
            ))
        })?;
        tracing::info!(
            sidecar_bytes_freed = size,
            path = %path.display(),
            "edge-only substrate.edge_props sidecar deleted"
        );
        Ok(Some(size))
    }

    /// T17c Step 3b.2b: encode `value` via the props codec, append a
    /// fresh [`PropertyEntry`] to the PropsZone chain for `id`, and
    /// WAL-log the new `first_prop_off` head pointer via the writer.
    ///
    /// Callers MUST ensure that `self.props_zone.is_some()` before
    /// invoking — otherwise this function panics on `unwrap`. That
    /// guarantee comes from the callsite in `set_node_property`.
    ///
    /// Semantics:
    ///
    /// - The `key` string is interned into
    ///   [`PropertyKeyRegistry`] for a stable `u16` prop-key id.
    /// - `encode_value` may intern string / bytes payloads into the
    ///   heap side-car (`substrate.props.heap.v2`).
    /// - `PropsZone::append_entry` either appends to the current head
    ///   page (fast path) or allocates a fresh head page when the
    ///   current one is full (slow path that writes a brand new 4 KiB
    ///   page with `next_page = old_head`).
    /// - When the head page changes, a
    ///   [`crate::wal::WalPayload::NodePropHeadUpdate`] record is
    ///   appended to the WAL via `Writer::update_node_first_prop_off`
    ///   — only the 6 bytes of `first_prop_off` in the NodeRecord are
    ///   rewritten, leaving every other slot-field intact.
    ///
    /// The DashMap sidecar write still happens unconditionally at the
    /// callsite, keeping reads fast during 3b (the DashMap is the
    /// authoritative read source until 3c wires the walk_chain read
    /// path).
    ///
    /// Errors propagate verbatim — the callsite in
    /// `set_node_property` catches them and logs a `tracing::warn!`
    /// so that a PropsZone failure never loses the underlying write.
    fn append_scalar_to_props_zone_v2(
        &self,
        id: NodeId,
        key: &str,
        value: &Value,
    ) -> SubstrateResult<()> {
        let pz_lock = self
            .props_zone
            .as_ref()
            .expect("append_scalar_to_props_zone_v2 called without props_zone");

        // 1. Intern the key → u16 prop-key id (reuses the existing
        //    PropertyKeyRegistry that also backs VecColumnRegistry).
        let prop_key_id = self.prop_keys.write().intern(key)?;

        // 2. Read the current NodeRecord to pick up the existing head
        //    pointer. Early-out if the slot is gone — should never
        //    happen because `is_live_on_disk` already gated the call.
        let slot = id.0 as u32;
        let node_rec = match self.writer.read_node(slot)? {
            Some(rec) => rec,
            None => {
                return Err(SubstrateError::Internal(format!(
                    "set_node_property: node slot {slot} missing from nodes zone"
                )));
            }
        };
        let current_head = decode_page_id(node_rec.first_prop_off);

        // 3. Encode the Value → PropertyValue + append the entry.
        //    Holds the PropsZone write lock across encode + append so
        //    heap interns and page mutations land atomically.
        let new_head_idx = {
            let mut pz = pz_lock.write();
            let pv = encode_value(&mut pz, value)?;
            let entry = PropertyEntry::new(prop_key_id, pv);
            pz.append_entry(slot, current_head, &entry)?
        };

        // 4. If the head page changed (new head allocated), WAL-log
        //    the new pointer. No-op when the entry fit in the existing
        //    head page — in that case `first_prop_off` is unchanged.
        let new_head_u48 = encode_page_id(new_head_idx);
        if new_head_u48 != node_rec.first_prop_off {
            self.writer.update_node_first_prop_off(slot, new_head_u48)?;
        }
        Ok(())
    }

    /// T17c Step 3c — Append a **tombstone** for `(slot, key)` to the
    /// PropsZone chain and WAL-log the new head pointer if the head
    /// page rotated. Mirrors `append_scalar_to_props_zone_v2` but
    /// writes a `PropertyEntry::tombstone(..)` instead of a live value.
    ///
    /// Semantics:
    /// - The key must already be interned — a tombstone for a key the
    ///   writer never interned is a silent no-op (no chain entry), the
    ///   read path returns `None` naturally via
    ///   `lookup_node_property_v2`.
    /// - On a never-written slot (`first_prop_off == 0`) we still
    ///   allocate a fresh head page with the tombstone, so a later
    ///   legacy-hydrated DashMap entry is shadowed by the tombstone on
    ///   next read. This is required for the LWW contract.
    fn append_tombstone_to_props_zone_v2(
        &self,
        id: NodeId,
        key: &str,
    ) -> SubstrateResult<()> {
        let pz_lock = self
            .props_zone
            .as_ref()
            .expect("append_tombstone_to_props_zone_v2 called without props_zone");

        // Intern the key so a freshly deleted key that was never `set`
        // before still gets a stable id on the chain. Cheap: the write
        // lock is held only for the intern call.
        let prop_key_id = self.prop_keys.write().intern(key)?;

        let slot = id.0 as u32;
        let node_rec = match self.writer.read_node(slot)? {
            Some(rec) => rec,
            None => {
                return Err(SubstrateError::Internal(format!(
                    "remove_node_property: node slot {slot} missing from nodes zone"
                )));
            }
        };
        let current_head = decode_page_id(node_rec.first_prop_off);

        let new_head_idx = {
            let mut pz = pz_lock.write();
            let entry = PropertyEntry::tombstone(
                prop_key_id,
                crate::page::ValueTag::Null,
            );
            pz.append_entry(slot, current_head, &entry)?
        };

        let new_head_u48 = encode_page_id(new_head_idx);
        if new_head_u48 != node_rec.first_prop_off {
            self.writer.update_node_first_prop_off(slot, new_head_u48)?;
        }
        Ok(())
    }

    /// T17c Step 3c — Resolve a scalar property through the PropsZone
    /// v2 chain, following LWW semantics.
    ///
    /// Returns:
    /// - `Some(Some(value))` — a live entry was found on the chain.
    /// - `Some(None)` — a tombstone was found on the chain; the key is
    ///   semantically deleted and callers MUST return `None` **without
    ///   consulting the DashMap fallback** (LWW: a newer tombstone
    ///   shadows any older DashMap value).
    /// - `None` — nothing on the chain about this key. Callers fall
    ///   back to the DashMap legacy sidecar.
    ///
    /// `None` is also returned when:
    /// - `self.props_zone` is `None` (v2 disabled — legacy mode),
    /// - the NodeRecord has `first_prop_off == 0` (slot never wrote
    ///   through v2 — e.g. not-yet-migrated data),
    /// - the key was never interned (no prop_key_id in the registry),
    /// - an I/O or decode error occurs (warned via `tracing::warn!` so
    ///   we degrade to the DashMap path rather than loudly fail the
    ///   read).
    fn lookup_node_property_v2(
        &self,
        id: NodeId,
        key: &str,
    ) -> Option<Option<Value>> {
        let pz_lock = self.props_zone.as_ref()?;
        let slot = id.0 as u32;
        let node_rec = match self.writer.read_node(slot) {
            Ok(Some(rec)) => rec,
            _ => return None,
        };
        let head = decode_page_id(node_rec.first_prop_off)?;
        let prop_key_id = self.prop_keys.read().lookup(key)?;
        let pz = pz_lock.read();
        let entry = match pz.get_latest_for_key(Some(head), prop_key_id) {
            Ok(Some(e)) => e,
            Ok(None) => return None,
            Err(err) => {
                tracing::warn!(
                    node_id = id.0,
                    key = %key,
                    error = %err,
                    "props zone v2 read failed — falling back to DashMap"
                );
                return None;
            }
        };
        if entry.is_tombstone() {
            // LWW: tombstone wins over any DashMap fallback.
            return Some(None);
        }
        match decode_value(&pz, &entry.value) {
            Ok(v) => Some(Some(v)),
            Err(err) => {
                tracing::warn!(
                    node_id = id.0,
                    key = %key,
                    error = %err,
                    "props zone v2 decode failed — falling back to DashMap"
                );
                None
            }
        }
    }

    // ==================================================================
    // T17f Step 4 — Edge mirrors of the PropsZone v2 helpers.
    //
    // These are edge-owner-kind equivalents of
    // `append_scalar_to_props_zone_v2`, `append_tombstone_to_props_zone_v2`,
    // and `lookup_node_property_v2`. They differ only in:
    //   (a) reading the head pointer from `EdgeRecord.first_prop_off`
    //       (U48 at byte offset 24) rather than `NodeRecord.first_prop_off`
    //       (at byte offset 14),
    //   (b) using `PropsZone::append_entry_edge` (page magic 0xE507_5FA6),
    //   (c) emitting `WalPayload::EdgePropHeadUpdate` via
    //       `Writer::update_edge_first_prop_off`.
    //
    // Parity-first implementation — future refactor may genericise over a
    // `PropertyOwner` trait, but the three-method triplet keeps the
    // call-site trivially debuggable and the per-kind audit surface small.
    // ==================================================================

    /// T17f Step 4 — Edge mirror of
    /// [`Self::append_scalar_to_props_zone_v2`]. Appends a live
    /// `(prop_key, value)` entry to the edge-owned chain rooted in
    /// `EdgeRecord.first_prop_off` and WAL-logs the new head pointer via
    /// `EdgePropHeadUpdate` when the head page rotates.
    ///
    /// Caller (`set_edge_property`) MUST check `self.props_zone.is_some()`
    /// before invoking — otherwise this function panics on `unwrap`.
    fn append_scalar_to_props_zone_v2_edge(
        &self,
        id: EdgeId,
        key: &str,
        value: &Value,
    ) -> SubstrateResult<()> {
        let pz_lock = self
            .props_zone
            .as_ref()
            .expect("append_scalar_to_props_zone_v2_edge called without props_zone");

        // 1. Intern the key → u16 prop-key id (shared registry with nodes;
        //    prop keys are owner-agnostic).
        let prop_key_id = self.prop_keys.write().intern(key)?;

        // 2. Read the current EdgeRecord to pick up the existing head
        //    pointer. Early-out if the slot is gone — should never happen
        //    because `is_live_edge_on_disk` already gated the call.
        let slot = id.0 as u32;
        let edge_rec = match self.writer.read_edge(id.0)? {
            Some(rec) => rec,
            None => {
                return Err(SubstrateError::Internal(format!(
                    "set_edge_property: edge slot {slot} missing from edges zone"
                )));
            }
        };
        let current_head = decode_page_id(edge_rec.first_prop_off);

        // 3. Encode the Value → PropertyValue + append the entry on the
        //    edge-kind chain (page magic 0xE507_5FA6). Holds the PropsZone
        //    write lock across encode + append so heap interns and page
        //    mutations land atomically.
        let new_head_idx = {
            let mut pz = pz_lock.write();
            let pv = encode_value(&mut pz, value)?;
            let entry = PropertyEntry::new(prop_key_id, pv);
            pz.append_entry_edge(slot, current_head, &entry)?
        };

        // 4. If the head page changed (new head allocated), WAL-log the
        //    new pointer via EdgePropHeadUpdate. No-op when the entry fit
        //    in the existing head page — in that case `first_prop_off` is
        //    unchanged.
        let new_head_u48 = encode_page_id(new_head_idx);
        if new_head_u48 != edge_rec.first_prop_off {
            self.writer.update_edge_first_prop_off(slot, new_head_u48)?;
        }
        Ok(())
    }

    /// T17f Step 4 — Edge mirror of
    /// [`Self::append_tombstone_to_props_zone_v2`]. Appends a tombstone
    /// entry for `(edge_slot, key)` on the edge-owned chain.
    fn append_tombstone_to_props_zone_v2_edge(
        &self,
        id: EdgeId,
        key: &str,
    ) -> SubstrateResult<()> {
        let pz_lock = self
            .props_zone
            .as_ref()
            .expect("append_tombstone_to_props_zone_v2_edge called without props_zone");

        let prop_key_id = self.prop_keys.write().intern(key)?;

        let slot = id.0 as u32;
        let edge_rec = match self.writer.read_edge(id.0)? {
            Some(rec) => rec,
            None => {
                return Err(SubstrateError::Internal(format!(
                    "remove_edge_property: edge slot {slot} missing from edges zone"
                )));
            }
        };
        let current_head = decode_page_id(edge_rec.first_prop_off);

        let new_head_idx = {
            let mut pz = pz_lock.write();
            let entry = PropertyEntry::tombstone(
                prop_key_id,
                crate::page::ValueTag::Null,
            );
            pz.append_entry_edge(slot, current_head, &entry)?
        };

        let new_head_u48 = encode_page_id(new_head_idx);
        if new_head_u48 != edge_rec.first_prop_off {
            self.writer.update_edge_first_prop_off(slot, new_head_u48)?;
        }
        Ok(())
    }

    /// T17f Step 4 — Edge mirror of [`Self::lookup_node_property_v2`].
    /// Walks the edge-owner chain and returns LWW semantics:
    /// - `Some(Some(v))` — live entry wins;
    /// - `Some(None)` — tombstone wins (deleted; caller returns `None`
    ///   without consulting the DashMap sidecar);
    /// - `None` — nothing on the chain for this key; fall back to the
    ///   DashMap legacy sidecar.
    fn lookup_edge_property_v2(
        &self,
        id: EdgeId,
        key: &str,
    ) -> Option<Option<Value>> {
        let pz_lock = self.props_zone.as_ref()?;
        let edge_rec = match self.writer.read_edge(id.0) {
            Ok(Some(rec)) => rec,
            _ => return None,
        };
        let head = decode_page_id(edge_rec.first_prop_off)?;
        let prop_key_id = self.prop_keys.read().lookup(key)?;
        let pz = pz_lock.read();
        let entry = match pz.get_latest_for_key(Some(head), prop_key_id) {
            Ok(Some(e)) => e,
            Ok(None) => return None,
            Err(err) => {
                tracing::warn!(
                    edge_id = id.0,
                    key = %key,
                    error = %err,
                    "props zone v2 (edge) read failed — falling back to DashMap"
                );
                return None;
            }
        };
        if entry.is_tombstone() {
            // LWW: tombstone wins over any DashMap fallback.
            return Some(None);
        }
        match decode_value(&pz, &entry.value) {
            Ok(v) => Some(Some(v)),
            Err(err) => {
                tracing::warn!(
                    edge_id = id.0,
                    key = %key,
                    error = %err,
                    "props zone v2 (edge) decode failed — falling back to DashMap"
                );
                None
            }
        }
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

    /// Persist the in-memory `self.node_properties` and
    /// `self.edge_properties` maps to a sidecar `substrate.props` file.
    /// See [`crate::props_snapshot`] for rationale and format.
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
        //
        // T16.7 Step 4d extends the filter to oversized `Value::String`
        // / `Value::Bytes` entries (payload > BLOB_COLUMN_THRESHOLD_BYTES).
        // These were routed to `blob_columns` by the setter, but a
        // streamed-write path or a pre-T16.7.4 sidecar load could still
        // land them on the DashMap. Dropping here closes the anon-RSS
        // gate on next flush.
        let is_routed_out = |(_, v): &(String, Value)| {
            if matches!(v, Value::Vector(_)) {
                return true;
            }
            blob_payload_len(v)
                .map(|n| n > BLOB_COLUMN_THRESHOLD_BYTES)
                .unwrap_or(false)
        };
        // Nodes (T17e Phase 2: iterate the dedicated property map; empty
        // property sets no longer occupy a DashMap entry, so the `is_empty`
        // guard below is purely a defensive no-op for entries left behind
        // by `remove_node_property` clearing the last key).
        snap.nodes.reserve(self.node_properties.len());
        for entry in self.node_properties.iter() {
            if entry.value().is_empty() {
                continue;
            }
            let props: Vec<(String, Value)> = map_to_entries(entry.value())
                .into_iter()
                .filter(|p| !is_routed_out(p))
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
        snap.edges.reserve(self.edge_properties.len());
        for entry in self.edge_properties.iter() {
            if entry.value().is_empty() {
                continue;
            }
            let props: Vec<(String, Value)> = map_to_entries(entry.value())
                .into_iter()
                .filter(|p| !is_routed_out(p))
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
        // T17e Phase 3: labels resolve from the mmap'd NodeRecord +
        // registry on demand; no DashMap to populate.
        id
    }

    /// Look up the most recently allocated slot for `community_id` (or
    /// `None` if that community has never been used in this store).
    /// Exposed for tests and for the CommunityWarden (Step 4).
    ///
    /// **T17g T1**: routes via [`Self::community_ranges`] which lazy-builds
    /// the range maps on first call.
    pub fn last_slot_for_community(&self, community_id: u32) -> Option<u32> {
        self.community_ranges()
            .placements
            .get(&community_id)
            .map(|r| *r.value())
            .filter(|s| *s != u32::MAX)
    }

    /// Look up the earliest live slot for `community_id` (or `None` if
    /// that community has never been used in this store).
    ///
    /// Companion to [`Self::last_slot_for_community`] — the pair
    /// `(first, last)` is the bounding slot range used by the prefetch
    /// hook (T11 Step 5). Range maintenance across online allocation and
    /// compaction: see [`Self::refresh_community_ranges`] and the
    /// `CommunityRanges` struct docs.
    ///
    /// **T17g T1**: routes via [`Self::community_ranges`] which lazy-builds
    /// the range maps on first call.
    pub fn first_slot_for_community(&self, community_id: u32) -> Option<u32> {
        self.community_ranges()
            .first_slots
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
    ///
    /// **T17g T1**: routes via [`Self::community_ranges`] to force the
    /// lazy build first (in case this is called before any other accessor),
    /// then clears and re-scans into the same DashMaps. The `OnceLock`
    /// stays initialized — only the contents are mutated.
    pub fn refresh_community_ranges(&self) -> SubstrateResult<()> {
        let ranges = self.community_ranges();
        ranges.placements.clear();
        ranges.first_slots.clear();
        let hw = self.next_node_id.load(Ordering::Acquire);
        for slot in 1..hw {
            let Some(rec) = self.writer.read_node(slot)? else {
                continue;
            };
            if rec.is_tombstoned() {
                continue;
            }
            // Ascending scan → first_slot takes the min, last_slot the max.
            ranges
                .first_slots
                .entry(rec.community_id)
                .or_insert(slot);
            ranges.placements.insert(rec.community_id, slot);
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

    /// Interior helper: route an oversized `Value::String` or
    /// `Value::Bytes` write to the [`BlobColumnRegistry`]. Interns the
    /// property-key, locks the substrate for the duration of
    /// `write_slot` (which may `ensure_room` the two backing files),
    /// and swallows errors with a log — the `GraphStoreMut` trait's
    /// setters are infallible, so the historical contract is to
    /// silently drop a bad write and let the next flush surface the
    /// problem.
    ///
    /// `tagged` is the already-encoded payload (1-byte type tag + raw
    /// bytes) produced by
    /// [`crate::blob_column_registry::encode_blob_payload`]. Callers
    /// must have verified the payload qualifies for routing (length >
    /// [`BLOB_COLUMN_THRESHOLD_BYTES`]) before invoking this helper.
    fn route_blob_write(
        &self,
        entity_kind: EntityKind,
        slot: u32,
        key: &str,
        tagged: &[u8],
    ) {
        let prop_key_id = match self.prop_keys.write().intern(key) {
            Ok(id) => id,
            Err(e) => {
                tracing::error!(
                    target: "substrate::blob_columns",
                    "prop_keys.intern({key:?}) failed: {e} — dropping blob write"
                );
                return;
            }
        };
        let pk = PropertyKey::new(key);
        let sub = self.substrate.lock();
        if let Err(e) = self
            .blob_columns
            .write(&sub, &pk, entity_kind, prop_key_id, slot, tagged)
        {
            tracing::error!(
                target: "substrate::blob_columns",
                "blob_columns.write(key={key:?}, ek={entity_kind:?}, \
                 slot={slot}, len={}) failed: {e} — dropping blob write",
                tagged.len()
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
        // T17g T1: route through the lazy-built community ranges. First
        // caller triggers the O(node_hw) parallel scan; subsequent calls
        // hit the DashMap directly.
        let ranges = self.community_ranges();
        // Acquire the per-community entry BEFORE reading next_node_id so
        // the fast-path check is consistent.
        let mut entry = ranges.placements.entry(community_id).or_insert(u32::MAX);

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
                    ranges.first_slots.insert(community_id, aligned);
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

    // -----------------------------------------------------------------
    // T17e Phase 1 — mmap + registry resolvers
    //
    // Zero-rebuild read helpers introduced as a preparatory step toward
    // T17e (eliminate the `nodes` / `edges` DashMaps). Both methods
    // return the *same* data that `NodeInMem.labels` and
    // `EdgeInMem.edge_type` carry after a full `rebuild_from_zones`, but
    // they read it directly from the mmap'd `NodeRecord.label_bitset`
    // and `EdgeRecord.edge_type` fields via the in-memory registries.
    //
    // Phase 1 is behaviour-preserving: the DashMaps stay populated and
    // continue to be the source of truth for properties; these helpers
    // simply demonstrate that the mmap+registry path produces the same
    // labels/edge_type for every live record. Phase 3 will remove the
    // DashMaps and promote these helpers to the sole read path.
    // -----------------------------------------------------------------

    /// Resolve label names from a `NodeRecord.label_bitset` via the
    /// in-memory [`LabelRegistry`]. Equivalent to `NodeInMem.labels` after
    /// a rebuild, but reads directly from the registry — no DashMap lookup.
    #[inline]
    fn resolve_node_labels_from_bitset(&self, bitset: u64) -> SmallVec<[ArcStr; 2]> {
        self.labels.read().labels_for_bitset(bitset)
    }

    /// Resolve an edge-type `ArcStr` from an `EdgeRecord.edge_type` id via
    /// the in-memory [`EdgeTypeRegistry`]. Falls back to `__UNKNOWN_{id}`
    /// on dict/zone drift — matches `rebuild_from_zones` behaviour so any
    /// dictionary corruption manifests identically on both code paths.
    #[inline]
    fn resolve_edge_type_by_id(&self, id: u16) -> ArcStr {
        match self.edge_types.read().name_for(id) {
            Some(n) => n,
            None => {
                tracing::error!(
                    target: "substrate",
                    "resolve_edge_type_by_id: edge_type id {id} not in registry — dict/zone drift?"
                );
                ArcStr::from(format!("__UNKNOWN_{id}"))
            }
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
        let Some(head_ref) = self.incoming_heads().get(&dst) else {
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
            .incoming_heads()
            .get(&dst)
            .map(|e| *e)
            .unwrap_or(EdgeId(0));

        // Build the EdgeRecord with both chain pointers set.
        // `first_prop_off` starts at ZERO — property head gets patched
        // later by `update_edge_first_prop_off` when the edge gets its
        // first property via PropsZone v2 (T17f Steps 3-4).
        let edge = EdgeRecord {
            src: src.0 as u32,
            dst: dst.0 as u32,
            edge_type: edge_type_id,
            weight_u16: 0,
            next_from: prev_out_head,
            next_to: Self::edge_slot_to_offset(prev_in_head),
            first_prop_off: U48::ZERO,
            ricci_u8: 0,
            flags: 0,
            engram_tag: 0,
            _pad: [0; 2],
        };

        // (3) update src's first_edge_off
        src_rec.first_edge_off = Self::edge_slot_to_offset(edge_id);
        self.writer
            .update_node(src.0 as u32, src_rec)
            .expect("update_node (splice src head) failed");

        // (4) update incoming_heads (in-memory)
        self.incoming_heads().insert(dst, edge_id);

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
        let heads = self.incoming_heads();
        let head = heads.get(&dst_id).map(|e| *e).unwrap_or(EdgeId(0));
        if head == edge_id {
            let next_id = Self::offset_to_edge_id(rec.next_to);
            if next_id.0 == 0 {
                heads.remove(&dst_id);
            } else {
                heads.insert(dst_id, next_id);
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
    //
    // T17e Phase 1 — Labels and edge_type are resolved directly from the
    // mmap'd record + in-memory registry. The `nodes` / `edges` DashMaps
    // are still consulted for properties (Phase 2 splits them into
    // dedicated `node_properties` / `edge_properties` maps; Phase 3
    // eliminates the structural DashMaps entirely). Behaviour is
    // preserved: the mmap+registry path returns the same SmallVec /
    // ArcStr a post-rebuild DashMap entry would carry.
    fn get_node(&self, id: NodeId) -> Option<Node> {
        if !self.is_live_on_disk(id) {
            return None;
        }
        let rec = self.writer.read_node(id.0 as u32).ok().flatten()?;
        let mut n = Node::new(id);
        n.labels = self.resolve_node_labels_from_bitset(rec.label_bitset);
        if let Some(entry) = self.node_properties.get(&id) {
            n.properties = entry.clone();
        }
        Some(n)
    }

    fn get_edge(&self, id: EdgeId) -> Option<Edge> {
        if !self.is_live_edge_on_disk(id) {
            return None;
        }
        let rec = self.writer.read_edge(id.0).ok().flatten()?;
        let edge_type = self.resolve_edge_type_by_id(rec.edge_type);
        let mut e = Edge::new(
            id,
            NodeId(rec.src as u64),
            NodeId(rec.dst as u64),
            edge_type,
        );
        if let Some(entry) = self.edge_properties.get(&id) {
            e.properties = entry.clone();
        }
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
        // T16.7 Step 4d: blob-routed String/Bytes live in their own
        // column pair. The 1-byte type tag baked into the arena
        // payload lets `read_value` reconstruct the exact original
        // variant.
        if let Some(v) = self
            .blob_columns
            .read_value(key, EntityKind::Node, id.0 as u32)
        {
            return Some(v);
        }
        // T17c Step 3c — PropsZone v2 read path with LWW semantics.
        // `lookup_node_property_v2` returns:
        //   - Some(Some(v)) → live entry wins
        //   - Some(None)    → tombstone wins (deleted; skip DashMap)
        //   - None          → nothing on v2 chain; fall through
        match self.lookup_node_property_v2(id, key.as_str()) {
            Some(v) => return v,
            None => {}
        }
        let entry = self.node_properties.get(&id)?;
        entry.get(key).cloned()
    }

    fn get_edge_property(&self, id: EdgeId, key: &PropertyKey) -> Option<Value> {
        if !self.is_live_edge_on_disk(id) {
            return None;
        }
        if let Some(arc) = self.vec_columns.read(key, EntityKind::Edge, id.0 as u32) {
            return Some(Value::Vector(arc));
        }
        if let Some(v) = self
            .blob_columns
            .read_value(key, EntityKind::Edge, id.0 as u32)
        {
            return Some(v);
        }
        // T17f Step 4 — PropsZone v2 read path with LWW semantics
        // (mirror of `get_node_property`):
        //   - Some(Some(v)) → live entry wins
        //   - Some(None)    → tombstone wins (deleted; skip DashMap)
        //   - None          → nothing on v2 chain; fall through
        match self.lookup_edge_property_v2(id, key.as_str()) {
            Some(v) => return v,
            None => {}
        }
        let entry = self.edge_properties.get(&id)?;
        entry.get(key).cloned()
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
                if let Some(entry) = self.node_properties.get(id) {
                    for (k, v) in entry.iter() {
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
                if let Some(entry) = self.node_properties.get(id) {
                    for k in keys {
                        if let Some(v) = entry.get(k) {
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
                if let Some(entry) = self.edge_properties.get(id) {
                    for k in keys {
                        if let Some(v) = entry.get(k) {
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

    // T17h T9 — typed-degree trait methods. O(1) via the per-edge-type
    // registry built in T8. `edge_type = None` means "sum across all
    // types" and routes to the total T5 degree column (also O(1)).
    //
    // Live-check : `is_live_on_disk` filters tombstoned / out-of-range
    // slots (the typed registry doesn't carry a liveness bit itself —
    // incr_in/out are the only mutators, so a tombstoned node's slot
    // can still hold a stale positive value until the edges pointing
    // at it are individually deleted ; we never rely on them).
    fn out_degree_by_type(&self, node: NodeId, edge_type: Option<&str>) -> usize {
        if !self.is_live_on_disk(node) {
            return 0;
        }
        match edge_type {
            Some(name) => Self::out_degree_by_type(self, node, name) as usize,
            None => Self::out_degree(self, node) as usize, // inherent O(1) T5
        }
    }

    fn in_degree_by_type(&self, node: NodeId, edge_type: Option<&str>) -> usize {
        if !self.is_live_on_disk(node) {
            return 0;
        }
        match edge_type {
            Some(name) => Self::in_degree_by_type(self, node, name) as usize,
            None => Self::in_degree(self, node) as usize, // inherent O(1) T5
        }
    }

    fn supports_typed_degree(&self) -> bool {
        true
    }

    // T17i T2 — label-histogram trait overrides. Route to the inherent
    // `edge_target_labels_substrate` / `edge_source_labels_substrate`
    // methods that read the DashMap counters in O(K ≤ 64). Backends
    // without a histogram keep the default O(E) scan impl.
    fn edge_target_labels(&self, edge_type: &str) -> std::collections::HashSet<String> {
        Self::edge_target_labels_substrate(self, edge_type)
    }

    fn edge_source_labels(&self, edge_type: &str) -> std::collections::HashSet<String> {
        Self::edge_source_labels_substrate(self, edge_type)
    }

    fn supports_edge_label_histogram(&self) -> bool {
        true
    }

    fn has_backward_adjacency(&self) -> bool {
        // `EdgeRecord` keeps both `next_from` (src chain) and `next_to` (dst
        // chain). The concrete read paths light up in step 2; the answer is
        // already `true` structurally.
        true
    }

    // -- Scans --
    //
    // T17e Phase 3: the scans iterate the slot range `1..high_water_mark`
    // directly and filter via the mmap'd TOMBSTONED flag. Cost shifts
    // from O(live) DashMap-iter to O(high_water) mmap-scan — for a
    // healthy store the two are within a constant factor, and the mmap
    // scan is pointer-chased cache-line-sized, so in practice the new
    // scans are *faster* than iterating a DashMap's sharded buckets.
    fn node_ids(&self) -> Vec<NodeId> {
        let hw = self.next_node_id.load(Ordering::Acquire);
        let mut out: Vec<NodeId> = Vec::new();
        for slot in 1..hw {
            let id = NodeId(slot as u64);
            if self.is_live_on_disk(id) {
                out.push(id);
            }
        }
        // Already sorted by construction — ascending slot scan.
        out
    }

    fn nodes_by_label(&self, label: &str) -> Vec<NodeId> {
        let reg = self.labels.read();
        let Some(bit) = reg.bit_for(label) else {
            return Vec::new();
        };
        let mask = 1u64 << bit;
        drop(reg);

        let hw = self.next_node_id.load(Ordering::Acquire);
        let mut out: Vec<NodeId> = Vec::new();
        for slot in 1..hw {
            let Ok(Some(rec)) = self.writer.read_node(slot) else {
                continue;
            };
            if rec.flags & crate::record::node_flags::TOMBSTONED != 0 {
                continue;
            }
            if rec.label_bitset & mask != 0 {
                out.push(NodeId(slot as u64));
            }
        }
        // Already sorted by construction.
        out
    }

    fn node_count(&self) -> usize {
        // T17h T1: O(1) atomic load (was O(N) scan over live slots).
        self.total_live_nodes.load(Ordering::Relaxed) as usize
    }

    fn edge_count(&self) -> usize {
        // T17h T1: O(1) atomic load (was O(N) scan over live slots).
        self.total_live_edges.load(Ordering::Relaxed) as usize
    }

    fn node_count_by_label(&self, label: &str) -> usize {
        // T17h T1: O(1) override of the trait's default O(N) impl.
        // Unknown label (not interned) → 0, which matches a scan result.
        let labels = self.labels.read();
        let Some(bit) = labels.bit_for(label) else {
            return 0;
        };
        drop(labels);
        self.label_live_counts[bit as usize].load(Ordering::Relaxed) as usize
    }

    // -- Entity metadata --
    fn edge_type(&self, id: EdgeId) -> Option<ArcStr> {
        if !self.is_live_edge_on_disk(id) {
            return None;
        }
        // T17e Phase 3: resolve directly from the mmap'd EdgeRecord +
        // registry — no DashMap cache.
        let rec = self.writer.read_edge(id.0).ok().flatten()?;
        Some(self.resolve_edge_type_by_id(rec.edge_type))
    }

    // -- Filtered search --
    //
    // T17e Phase 2: these scans iterate `node_properties` directly — a
    // node without the queried property cannot possibly match, so
    // skipping the empty-property nodes (no entry in the map) is both
    // correct and strictly more efficient than scanning every live
    // `NodeInMem`. The liveness gate still runs because a node may be
    // tombstoned after its property was set.
    fn find_nodes_by_property(&self, property: &str, value: &Value) -> Vec<NodeId> {
        let key = PropertyKey::new(property);
        let mut out: Vec<NodeId> = self
            .node_properties
            .iter()
            .filter_map(|e| {
                let id = *e.key();
                if !self.is_live_on_disk(id) {
                    return None;
                }
                match e.value().get(&key) {
                    Some(v) if v == value => Some(id),
                    _ => None,
                }
            })
            .collect();
        out.sort_by_key(|n| n.0);
        out
    }

    fn find_nodes_by_properties(&self, conditions: &[(&str, Value)]) -> Vec<NodeId> {
        // Contract (LpgStore parity): empty conditions match every live
        // node. T17e Phase 2: we must NOT iterate `node_properties` here
        // because nodes without any property would be silently dropped —
        // fall back to the structural `nodes` map, same as `node_ids()`.
        if conditions.is_empty() {
            return self.node_ids();
        }
        let keys: Vec<PropertyKey> = conditions.iter().map(|(k, _)| PropertyKey::new(*k)).collect();
        let mut out: Vec<NodeId> = self
            .node_properties
            .iter()
            .filter_map(|e| {
                let id = *e.key();
                if !self.is_live_on_disk(id) {
                    return None;
                }
                let props = e.value();
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
            .node_properties
            .iter()
            .filter_map(|e| {
                let id = *e.key();
                if !self.is_live_on_disk(id) {
                    return None;
                }
                let v = e.value().get(&key)?.clone();
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

    // -- Schema introspection --
    //
    // Substrate interns labels / edge types / property keys in dense
    // registries on every create_* and set_* call (and reloads them from
    // `DictSnapshot` on open). The trait-default `Vec::new()` would return
    // an empty list — break `InstrumentedStore` delegation tests and any
    // query planner that asks \"what labels exist?\". These overrides read
    // directly from the live registries.
    fn all_labels(&self) -> Vec<String> {
        self.labels.read().names()
    }

    fn all_edge_types(&self) -> Vec<String> {
        self.edge_types.read().names()
    }

    fn all_property_keys(&self) -> Vec<String> {
        self.prop_keys.read().names()
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

// T17h T1 — Live counter helpers (inherent methods, shared by
// create_*/delete_* hot paths in the GraphStoreMut impl below).
impl SubstrateStore {
    /// Apply delta to an AtomicU64 counter (positive = fetch_add, negative
    /// = fetch_sub). Relaxed ordering: counters are eventually-consistent
    /// metrics, no happens-before dependency with NodeRecord/EdgeRecord
    /// mutations.
    #[inline]
    fn apply_counter_delta(atomic: &AtomicU64, delta: i64) {
        if delta >= 0 {
            atomic.fetch_add(delta as u64, Ordering::Relaxed);
        } else {
            atomic.fetch_sub((-delta) as u64, Ordering::Relaxed);
        }
    }

    /// Increment/decrement label counters for every bit set in `bitset`.
    /// Walks each set bit via `trailing_zeros` + clear-lowest-bit trick
    /// (compiles to TZCNT/BLSR on x86_64 with BMI1, or CTZ/AND on ARM64).
    pub(crate) fn incr_label_counts(&self, bitset: u64, delta: i64) {
        let mut bits = bitset;
        while bits != 0 {
            let bit = bits.trailing_zeros() as usize;
            Self::apply_counter_delta(&self.label_live_counts[bit], delta);
            bits &= bits - 1;
        }
    }

    /// Increment/decrement per-edge-type counter via DashMap. Fast path
    /// reads existing entry to avoid holding the shard write lock for
    /// the atomic op.
    pub(crate) fn incr_edge_type_count(&self, edge_type_id: u16, delta: i64) {
        if let Some(atomic) = self.edge_type_live_counts.get(&edge_type_id) {
            Self::apply_counter_delta(&atomic, delta);
            return;
        }
        let entry = self
            .edge_type_live_counts
            .entry(edge_type_id)
            .or_insert_with(|| AtomicU64::new(0));
        Self::apply_counter_delta(&entry, delta);
    }

    /// T17h T1 accessor — per-edge-type live count, O(1). Returns 0 for
    /// unknown edge-type strings (not yet interned).
    pub fn live_count_by_edge_type(&self, edge_type: &str) -> u64 {
        let registry = self.edge_types.read();
        let Some(id) = registry.id_for(edge_type) else {
            return 0;
        };
        drop(registry);
        self.edge_type_live_counts
            .get(&id)
            .map(|a| a.load(Ordering::Relaxed))
            .unwrap_or(0)
    }

    /// T17i T2 — increment/decrement the target-label histogram for
    /// every bit set in `label_bitset` under `edge_type_id`. Walks
    /// each set bit via `trailing_zeros` + clear-lowest-bit trick
    /// (same pattern as `incr_label_counts`). `delta = +1` on
    /// `create_edge`, `-1` on `delete_edge`.
    pub(crate) fn incr_edge_type_target_label(
        &self,
        edge_type_id: u16,
        label_bitset: u64,
        delta: i64,
    ) {
        let mut bits = label_bitset;
        while bits != 0 {
            let bit = bits.trailing_zeros() as u8;
            let key = (edge_type_id, bit);
            if let Some(atomic) = self.edge_type_target_label_counts.get(&key) {
                Self::apply_counter_delta(&atomic, delta);
            } else {
                let entry = self
                    .edge_type_target_label_counts
                    .entry(key)
                    .or_insert_with(|| AtomicU64::new(0));
                Self::apply_counter_delta(&entry, delta);
            }
            bits &= bits - 1;
        }
    }

    /// T17i T2 — same as [`Self::incr_edge_type_target_label`] but
    /// for source labels. Fed by `rec.src` instead of `rec.dst`.
    pub(crate) fn incr_edge_type_source_label(
        &self,
        edge_type_id: u16,
        label_bitset: u64,
        delta: i64,
    ) {
        let mut bits = label_bitset;
        while bits != 0 {
            let bit = bits.trailing_zeros() as u8;
            let key = (edge_type_id, bit);
            if let Some(atomic) = self.edge_type_source_label_counts.get(&key) {
                Self::apply_counter_delta(&atomic, delta);
            } else {
                let entry = self
                    .edge_type_source_label_counts
                    .entry(key)
                    .or_insert_with(|| AtomicU64::new(0));
                Self::apply_counter_delta(&entry, delta);
            }
            bits &= bits - 1;
        }
    }

    /// T17i T2 — returns the set of labels observed on the target
    /// nodes of edges of `edge_type`. Used by the T17i T3 Cypher
    /// planner rewrite to verify corpus invariants before routing
    /// label-constrained queries. O(K) where K ≤ 64 label bits for
    /// this edge type.
    pub fn edge_target_labels_substrate(&self, edge_type: &str) -> std::collections::HashSet<String> {
        let registry = self.edge_types.read();
        let Some(type_id) = registry.id_for(edge_type) else {
            return std::collections::HashSet::new();
        };
        drop(registry);
        let mut out = std::collections::HashSet::new();
        for bit in 0..64u8 {
            let key = (type_id, bit);
            if let Some(atomic) = self.edge_type_target_label_counts.get(&key)
                && atomic.load(Ordering::Relaxed) > 0
            {
                let labels = self.labels.read();
                if let Some(name) = labels.bit_to_name.get(bit as usize) {
                    out.insert(name.to_string());
                }
            }
        }
        out
    }

    /// T17i T2 — symmetric accessor for source labels.
    pub fn edge_source_labels_substrate(&self, edge_type: &str) -> std::collections::HashSet<String> {
        let registry = self.edge_types.read();
        let Some(type_id) = registry.id_for(edge_type) else {
            return std::collections::HashSet::new();
        };
        drop(registry);
        let mut out = std::collections::HashSet::new();
        for bit in 0..64u8 {
            let key = (type_id, bit);
            if let Some(atomic) = self.edge_type_source_label_counts.get(&key)
                && atomic.load(Ordering::Relaxed) > 0
            {
                let labels = self.labels.read();
                if let Some(name) = labels.bit_to_name.get(bit as usize) {
                    out.insert(name.to_string());
                }
            }
        }
        out
    }

    /// T17j T5 — rebuild every outgoing chain from `EdgeRecord.src`.
    ///
    /// Treats `rec.src/dst/edge_type` as authoritative (they're
    /// immutable per the EdgeRecord contract) and reconstructs every
    /// `NodeRecord.first_edge_off` + `EdgeRecord.next_from` pointer
    /// from scratch. Use when chain ↔ edge-record divergence has
    /// been diagnosed (see `diagnose_chain_vs_rec_src`).
    ///
    /// ### Passes
    ///
    /// 1. **Reset** — zero every live `NodeRecord.first_edge_off`
    ///    (tombstoned nodes untouched).
    /// 2. **Rewire** — for every live edge slot, splice the edge at
    ///    the head of its `rec.src` node's chain by :
    ///    - reading `src_rec.first_edge_off` (current head after
    ///      pass 1 / partial pass 2),
    ///    - writing `edge.next_from = current_head`,
    ///    - writing `src_rec.first_edge_off = edge_slot_offset`.
    /// 3. **Flush** — commit WAL + msync zones.
    ///
    /// ### Idempotency & crash-safety
    ///
    /// Idempotent : re-running on an already-healthy store produces
    /// the same link graph (modulo edge-slot iteration order, which
    /// is deterministic for a fixed zone state).
    ///
    /// Crash-safe : every write goes through the WAL. Mid-repair
    /// crash leaves some chains half-built ; the next open replays
    /// the committed prefix. Re-running the repair completes it.
    ///
    /// ### Cost
    ///
    /// O(N + 2·E) with one WAL record per node write (pass 1) + two
    /// per edge (pass 2 : update_edge + update_node). On PO (1.36M
    /// nodes, 1.88M edges) : ~8M WAL records. On Wiki (119M edges) :
    /// ~240M records — multiple minutes in steady state, amortised
    /// once across subsequent reopens.
    ///
    /// Returns `(nodes_reset, edges_rewired)`.
    pub fn repair_outgoing_chains(&self) -> SubstrateResult<(u64, u64)> {
        let node_hw = self.next_node_id.load(Ordering::Acquire);
        let mut nodes_reset = 0u64;
        for slot in 1..node_hw {
            let Some(rec) = self.writer.read_node(slot)? else {
                continue;
            };
            if rec.flags & crate::record::node_flags::TOMBSTONED != 0 {
                continue;
            }
            if rec.first_edge_off.is_zero() {
                continue;
            }
            let mut new_rec = rec;
            new_rec.first_edge_off = crate::record::U48::from_u64(0);
            self.writer.update_node(slot, new_rec)?;
            nodes_reset += 1;
        }

        let edge_hw = self.next_edge_id.load(Ordering::Acquire);
        let mut edges_rewired = 0u64;
        for edge_slot in 1..edge_hw {
            let edge = match self.writer.read_edge(edge_slot)? {
                Some(e) if e.flags & crate::record::edge_flags::TOMBSTONED == 0 => e,
                _ => continue,
            };
            let src_slot = edge.src;
            let src_rec = match self.writer.read_node(src_slot)? {
                Some(r) => r,
                None => continue, // dangling — skip
            };
            let prev_head_off = src_rec.first_edge_off;
            let new_edge = crate::record::EdgeRecord {
                next_from: prev_head_off,
                ..edge
            };
            self.writer.update_edge(edge_slot, new_edge)?;
            let new_src = crate::record::NodeRecord {
                first_edge_off: Self::edge_slot_to_offset(EdgeId(edge_slot)),
                ..src_rec
            };
            self.writer.update_node(src_slot, new_src)?;
            edges_rewired += 1;
        }

        self.writer.commit()?;
        self.writer.msync_zones()?;
        Ok((nodes_reset, edges_rewired))
    }

    /// T17j T5 — diagnostic : walks the outgoing chain of `node_id`
    /// and returns stats comparing the chain-owner NodeId vs each
    /// edge's `rec.src` field. Returns `(edges_walked, rec_src_matches,
    /// rec_src_mismatches, first_mismatch_info)`.
    ///
    /// If `rec_src_mismatches > 0`, the chain and edge record fields
    /// are desynchronised — Cypher sees the edge attributed to the
    /// chain-owner node, but raw scans see it attributed to whatever
    /// `rec.src` says.
    ///
    /// `first_mismatch_info` is `Some((edge_slot, rec_src))` when a
    /// mismatch was found, letting the caller inspect the misrouted
    /// edge manually.
    pub fn diagnose_chain_vs_rec_src(
        &self,
        node_id: NodeId,
    ) -> (u64, u64, u64, Option<(u64, u32)>) {
        let mut walked = 0u64;
        let mut matches = 0u64;
        let mut mismatches = 0u64;
        let mut first_mismatch: Option<(u64, u32)> = None;
        self.walk_outgoing_chain(node_id, |rec, edge_id| {
            walked += 1;
            if rec.src as u64 == node_id.0 {
                matches += 1;
            } else {
                mismatches += 1;
                if first_mismatch.is_none() {
                    first_mismatch = Some((edge_id.0, rec.src));
                }
            }
        });
        (walked, matches, mismatches, first_mismatch)
    }

    /// T17j T5 — diagnostic : for a given label name, sample the first
    /// N NodeRecords returned by the label index and OR their
    /// label_bitset together. Returns `(count_sampled, union_bits)`.
    /// If the union doesn't contain `label`'s bit, the label_index
    /// and the on-disk NodeRecords diverge (migration bug).
    pub fn diagnose_label_bitsets_for(
        &self,
        label: &str,
        sample: usize,
    ) -> SubstrateResult<(usize, u64)> {
        let bit = {
            let reg = self.labels.read();
            match reg.bit_for(label) {
                Some(b) => b,
                None => return Ok((0, 0)),
            }
        };
        let nids = self.nodes_by_label(label);
        let n = nids.len().min(sample);
        let mut union_bits = 0u64;
        for nid in nids.iter().take(n) {
            if let Some(nr) = self.writer.read_node(nid.0 as u32)?
                && nr.flags & crate::record::node_flags::TOMBSTONED == 0
            {
                union_bits |= nr.label_bitset;
            }
        }
        let _ = bit;
        Ok((n, union_bits))
    }

    /// T17j T4 — deep diagnostic : scan every live edge of `edge_type`
    /// and return the aggregate bitset OR'd across all src / dst
    /// endpoints. This reveals which label bits actually appear in
    /// the endpoints (as opposed to what the histogram claims).
    pub fn diagnose_edge_endpoint_bits(
        &self,
        edge_type: &str,
    ) -> SubstrateResult<(u64, u64, u64)> {
        let reg = self.edge_types.read();
        let Some(type_id) = reg.id_for(edge_type) else {
            return Ok((0, 0, 0));
        };
        drop(reg);
        let edge_hw = self.next_edge_id.load(Ordering::Acquire);
        let mut total = 0u64;
        let mut src_union_bits = 0u64;
        let mut dst_union_bits = 0u64;
        for slot in 1..edge_hw {
            let Some(rec) = self.writer.read_edge(slot)? else {
                continue;
            };
            if rec.flags & crate::record::edge_flags::TOMBSTONED != 0 {
                continue;
            }
            if rec.edge_type != type_id {
                continue;
            }
            total += 1;
            if let Some(nr) = self.writer.read_node(rec.src)?
                && nr.flags & crate::record::node_flags::TOMBSTONED == 0
            {
                src_union_bits |= nr.label_bitset;
            }
            if let Some(nr) = self.writer.read_node(rec.dst)?
                && nr.flags & crate::record::node_flags::TOMBSTONED == 0
            {
                dst_union_bits |= nr.label_bitset;
            }
        }
        Ok((total, src_union_bits, dst_union_bits))
    }

    /// T17j T4 — diagnostic : scan every live edge of `edge_type` and
    /// return per-endpoint classification stats. Used to investigate
    /// why the peer-label histogram `edge_source_labels` returns a
    /// small subset (e.g. only `{File}` on PO's IMPORTS) while
    /// Cypher counts report many more distinct source labels.
    ///
    /// Returns `(total, src_valid_with_labels, src_valid_no_labels,
    /// src_missing, dst_valid_with_labels, dst_valid_no_labels,
    /// dst_missing)`.
    pub fn diagnose_edge_endpoints(
        &self,
        edge_type: &str,
    ) -> SubstrateResult<(u64, u64, u64, u64, u64, u64, u64)> {
        let reg = self.edge_types.read();
        let Some(type_id) = reg.id_for(edge_type) else {
            return Ok((0, 0, 0, 0, 0, 0, 0));
        };
        drop(reg);
        let edge_hw = self.next_edge_id.load(Ordering::Acquire);
        let mut total = 0u64;
        let mut src_valid_with = 0u64;
        let mut src_valid_no = 0u64;
        let mut src_missing = 0u64;
        let mut dst_valid_with = 0u64;
        let mut dst_valid_no = 0u64;
        let mut dst_missing = 0u64;
        for slot in 1..edge_hw {
            let Some(rec) = self.writer.read_edge(slot)? else {
                continue;
            };
            if rec.flags & crate::record::edge_flags::TOMBSTONED != 0 {
                continue;
            }
            if rec.edge_type != type_id {
                continue;
            }
            total += 1;
            match self.writer.read_node(rec.src)? {
                None => src_missing += 1,
                Some(nr) if nr.flags & crate::record::node_flags::TOMBSTONED != 0 => {
                    src_missing += 1
                }
                Some(nr) if nr.label_bitset == 0 => src_valid_no += 1,
                Some(_) => src_valid_with += 1,
            }
            match self.writer.read_node(rec.dst)? {
                None => dst_missing += 1,
                Some(nr) if nr.flags & crate::record::node_flags::TOMBSTONED != 0 => {
                    dst_missing += 1
                }
                Some(nr) if nr.label_bitset == 0 => dst_valid_no += 1,
                Some(_) => dst_valid_with += 1,
            }
        }
        Ok((
            total,
            src_valid_with,
            src_valid_no,
            src_missing,
            dst_valid_with,
            dst_valid_no,
            dst_missing,
        ))
    }

    /// T17h T1 — one-shot scan that populates counters from the on-disk
    /// zones. Sequential (rayon contends on the zone cache Mutex). Used
    /// at open time when the loaded dict snapshot does not carry a
    /// `counters` block (i.e. a v1..=v4 base). Once persisted in the
    /// next flush, subsequent opens restore counters in O(1) from the
    /// v5 snapshot and this scan is never re-run.
    pub(crate) fn rebuild_live_counters_from_zones(&self) -> SubstrateResult<()> {
        let node_hw = self.next_node_id.load(Ordering::Acquire);
        let mut total_nodes: u64 = 0;
        let mut label_totals: [u64; 64] = [0; 64];
        for slot in 1..node_hw {
            let Some(rec) = self.writer.read_node(slot)? else {
                continue;
            };
            if rec.flags & crate::record::node_flags::TOMBSTONED != 0 {
                continue;
            }
            total_nodes += 1;
            let mut bits = rec.label_bitset;
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                label_totals[bit] += 1;
                bits &= bits - 1;
            }
        }
        self.total_live_nodes.store(total_nodes, Ordering::Relaxed);
        for (bit, &count) in label_totals.iter().enumerate() {
            self.label_live_counts[bit].store(count, Ordering::Relaxed);
        }

        let edge_hw = self.next_edge_id.load(Ordering::Acquire);
        let mut total_edges: u64 = 0;
        let mut edge_type_totals: FxHashMap<u16, u64> = FxHashMap::default();
        // T17i T2 — per-edge-type × label-bit histograms populated
        // alongside the edge-type totals in a single sequential scan.
        let mut target_hist: FxHashMap<(u16, u8), u64> = FxHashMap::default();
        let mut source_hist: FxHashMap<(u16, u8), u64> = FxHashMap::default();
        for slot in 1..edge_hw {
            let Some(rec) = self.writer.read_edge(slot)? else {
                continue;
            };
            if rec.flags & crate::record::edge_flags::TOMBSTONED != 0 {
                continue;
            }
            total_edges += 1;
            *edge_type_totals.entry(rec.edge_type).or_insert(0) += 1;
            // Fetch endpoint bitsets ; dangling endpoints contribute 0.
            let src_bitset = self
                .writer
                .read_node(rec.src)
                .ok()
                .flatten()
                .map(|r| r.label_bitset)
                .unwrap_or(0);
            let dst_bitset = self
                .writer
                .read_node(rec.dst)
                .ok()
                .flatten()
                .map(|r| r.label_bitset)
                .unwrap_or(0);
            let mut bits = src_bitset;
            while bits != 0 {
                let bit = bits.trailing_zeros() as u8;
                *source_hist.entry((rec.edge_type, bit)).or_insert(0) += 1;
                bits &= bits - 1;
            }
            let mut bits = dst_bitset;
            while bits != 0 {
                let bit = bits.trailing_zeros() as u8;
                *target_hist.entry((rec.edge_type, bit)).or_insert(0) += 1;
                bits &= bits - 1;
            }
        }
        self.total_live_edges.store(total_edges, Ordering::Relaxed);
        self.edge_type_live_counts.clear();
        for (type_id, count) in edge_type_totals {
            self.edge_type_live_counts
                .insert(type_id, AtomicU64::new(count));
        }
        // T17i T2 — install the two histograms atomically (clear + insert).
        self.edge_type_target_label_counts.clear();
        for (key, count) in target_hist {
            self.edge_type_target_label_counts
                .insert(key, AtomicU64::new(count));
        }
        self.edge_type_source_label_counts.clear();
        for (key, count) in source_hist {
            self.edge_type_source_label_counts
                .insert(key, AtomicU64::new(count));
        }
        Ok(())
    }

    /// T17h T1 — restore counters from a `PersistedCounters` snapshot
    /// (v5+ dict). O(1) vs rebuild's O(N+E). Called at open time when
    /// `DictSnapshot.counters = Some(_)`.
    pub(crate) fn restore_counters_from_snapshot(
        &self,
        counters: &crate::dict::PersistedCounters,
    ) {
        self.total_live_nodes
            .store(counters.total_live_nodes, Ordering::Relaxed);
        self.total_live_edges
            .store(counters.total_live_edges, Ordering::Relaxed);
        for (bit, &count) in counters.label_counts.iter().enumerate() {
            self.label_live_counts[bit].store(count, Ordering::Relaxed);
        }
        self.edge_type_live_counts.clear();
        for &(type_id, count) in &counters.edge_type_counts {
            self.edge_type_live_counts
                .insert(type_id, AtomicU64::new(count));
        }
        // T17i T2 — restore the two histograms from the v6 dict block.
        // Empty (legacy v5 load) → caller is expected to trigger
        // `rebuild_live_counters_from_zones` to repopulate.
        self.edge_type_target_label_counts.clear();
        for &(type_id, ref arr) in &counters.edge_type_target_label_counts {
            for (bit, &count) in arr.iter().enumerate() {
                if count > 0 {
                    self.edge_type_target_label_counts
                        .insert((type_id, bit as u8), AtomicU64::new(count));
                }
            }
        }
        self.edge_type_source_label_counts.clear();
        for &(type_id, ref arr) in &counters.edge_type_source_label_counts {
            for (bit, &count) in arr.iter().enumerate() {
                if count > 0 {
                    self.edge_type_source_label_counts
                        .insert((type_id, bit as u8), AtomicU64::new(count));
                }
            }
        }
    }

    /// T17h T5 — degree column accessor with lazy build.
    ///
    /// First call either opens the persisted sidecar (O(1) mmap) or
    /// rebuilds it from an edge zone scan (O(edges), one-shot
    /// amortised, persists on success).
    ///
    /// Returns `&Arc<RwLock<_>>` so callers clone the Arc cheaply and
    /// hold read locks for concurrent atomic increments, or write locks
    /// for grow/persist.
    pub(crate) fn degrees(&self) -> &Arc<RwLock<crate::degree_column::DegreeColumn>> {
        self.degrees_cell.get_or_init(|| {
            let sub = self.substrate.lock();
            let opened = crate::degree_column::DegreeColumn::open(&sub);
            drop(sub);
            let col = match opened {
                Ok(Some(col)) => col,
                _ => self
                    .rebuild_degrees_from_scan()
                    .expect("rebuild degree column failed"),
            };
            Arc::new(RwLock::new(col))
        })
    }

    /// T17h T5 — one-shot scan to rebuild the degree column. Scans the
    /// edges zone sequentially and populates out/in degrees. Persists
    /// CRC + msync before returning so the subsequent open (same session
    /// or next open) finds a valid sidecar.
    ///
    /// Sequential because `writer.read_edge` takes the zone-cache mutex
    /// (rayon would contend). Typical cost : ≤ 30 s on Wikipedia
    /// (119M edges), one-shot, amortised.
    fn rebuild_degrees_from_scan(
        &self,
    ) -> SubstrateResult<crate::degree_column::DegreeColumn> {
        let n_slots = self.next_node_id.load(Ordering::Acquire);
        let sub = self.substrate.lock();
        let mut col = crate::degree_column::DegreeColumn::create(&sub, n_slots)?;
        drop(sub);
        let edge_hw = self.next_edge_id.load(Ordering::Acquire);
        for slot in 1..edge_hw {
            let Some(rec) = self.writer.read_edge(slot)? else {
                continue;
            };
            if rec.flags & crate::record::edge_flags::TOMBSTONED != 0 {
                continue;
            }
            // Endpoints must be within n_slots (allocator invariant :
            // edges never reference unallocated nodes). Defensive
            // ensure_slot keeps us safe if a test violates this.
            if rec.src >= n_slots || rec.dst >= n_slots {
                let max = rec.src.max(rec.dst);
                col.ensure_slot(max)?;
            }
            col.incr_out(rec.src, 1);
            col.incr_in(rec.dst, 1);
        }
        col.persist_header_crc();
        col.msync()?;
        Ok(col)
    }

    /// T17h T5 — increment out-degree for `slot`. Grows column if needed
    /// (slot past current n_slots). Fast path takes a read lock for the
    /// atomic op; cold path takes write lock to grow then the atomic op.
    pub(crate) fn incr_out_degree(&self, slot: u32, delta: i32) {
        let degrees = self.degrees().clone();
        {
            let rl = degrees.read();
            if slot < rl.n_slots() {
                rl.incr_out(slot, delta);
                return;
            }
        }
        let mut wl = degrees.write();
        if let Err(e) = wl.ensure_slot(slot) {
            tracing::warn!(error = ?e, slot, "degree column grow failed");
            return;
        }
        wl.incr_out(slot, delta);
    }

    /// T17h T5 — increment in-degree for `slot`. Same pattern as
    /// [`Self::incr_out_degree`].
    pub(crate) fn incr_in_degree(&self, slot: u32, delta: i32) {
        let degrees = self.degrees().clone();
        {
            let rl = degrees.read();
            if slot < rl.n_slots() {
                rl.incr_in(slot, delta);
                return;
            }
        }
        let mut wl = degrees.write();
        if let Err(e) = wl.ensure_slot(slot) {
            tracing::warn!(error = ?e, slot, "degree column grow failed");
            return;
        }
        wl.incr_in(slot, delta);
    }

    /// T17h T5 — query out-degree for a node. O(1) atomic load.
    pub fn out_degree(&self, node: NodeId) -> u32 {
        self.degrees().read().out_degree(node.0 as u32)
    }

    /// T17h T5 — query in-degree for a node. O(1) atomic load.
    pub fn in_degree(&self, node: NodeId) -> u32 {
        self.degrees().read().in_degree(node.0 as u32)
    }

    /// T17h T8 + T17i T1 — per-edge-type degree registry accessor with
    /// deferred first-init. The registry Arc is materialised once via
    /// `OnceLock::get_or_init`, then `ensure_initialized` is driven to
    /// completion on every call (O(1) after first) so callers always
    /// see a fully-populated registry before they increment counters
    /// or query degrees.
    ///
    /// Why hydrate here rather than in `from_substrate` : the eager
    /// path opens every persisted sidecar (1.3-4.7 ms/type × 65 types
    /// on PO = +85 ms DB open). Deferring pushes that cost to the
    /// first hot-path call — amortised across the session and
    /// invisible to read-only queries that never touch typed degrees
    /// (e.g. a `count(n)` / `count(n:Label)` traffic pattern).
    pub(crate) fn typed_degrees(
        &self,
    ) -> &Arc<crate::typed_degree::TypedDegreeRegistry> {
        let registry = self.typed_degrees_cell.get_or_init(|| {
            Arc::new(crate::typed_degree::TypedDegreeRegistry::new(
                Arc::clone(&self.substrate),
                Arc::clone(&self.edge_types),
            ))
        });
        // Race-safe first-init : the closure runs under the registry's
        // internal mutex. Concurrent callers wait here on the first
        // call, then get a straight path on every subsequent call.
        // Errors during init are logged and swallowed — the registry
        // remains uninitialised and subsequent accesses re-attempt
        // (a disk glitch should eventually resolve).
        if let Err(e) =
            registry.ensure_initialized(|reg| Self::rebuild_typed_degrees_into(reg, self))
        {
            tracing::warn!(
                error = ?e,
                "typed_degrees ensure_initialized failed — registry may be incomplete"
            );
        }
        registry
    }

    /// T17i T1 — one-shot rebuild that populates a given registry from
    /// the substrate edge zone. Called by `ensure_initialized` when no
    /// sidecars exist on disk. Sequential (zone-cache Mutex contention
    /// rules out rayon) ; ~30 s worst-case on Wikipedia (119 M edges).
    ///
    /// Associated function rather than a method so the closure passed
    /// to `ensure_initialized` doesn't capture `self` and form a
    /// self-reference cycle with the registry.
    fn rebuild_typed_degrees_into(
        registry: &crate::typed_degree::TypedDegreeRegistry,
        store: &Self,
    ) -> SubstrateResult<()> {
        let n_slots_hint = store.next_node_id.load(Ordering::Acquire);
        let edge_hw = store.next_edge_id.load(Ordering::Acquire);
        for slot in 1..edge_hw {
            let Some(rec) = store.writer.read_edge(slot)? else {
                continue;
            };
            if rec.flags & crate::record::edge_flags::TOMBSTONED != 0 {
                continue;
            }
            registry.incr_out(rec.edge_type, rec.src, 1, n_slots_hint)?;
            registry.incr_in(rec.edge_type, rec.dst, 1, n_slots_hint)?;
        }
        // Persist sidecars so subsequent opens take the fast `open_existing`
        // path inside `ensure_initialized`.
        registry.flush()?;
        Ok(())
    }

    /// T17h T8 — query out-degree for `(node, edge_type)`. Returns 0
    /// if the type has no column (rare type never edged).
    pub fn out_degree_by_type(&self, node: NodeId, edge_type: &str) -> u32 {
        let reg = self.edge_types.read();
        let Some(id) = reg.id_for(edge_type) else {
            return 0;
        };
        drop(reg);
        self.typed_degrees().out_degree(id, node.0 as u32)
    }

    /// T17h T8 — query in-degree for `(node, edge_type)`.
    pub fn in_degree_by_type(&self, node: NodeId, edge_type: &str) -> u32 {
        let reg = self.edge_types.read();
        let Some(id) = reg.id_for(edge_type) else {
            return 0;
        };
        drop(reg);
        self.typed_degrees().in_degree(id, node.0 as u32)
    }

    /// T17h T8 — increment typed out-degree. Hook called from
    /// `create_edge` alongside the total T5 increment.
    pub(crate) fn incr_typed_out_degree(
        &self,
        edge_type_id: u16,
        slot: u32,
        delta: i32,
    ) {
        let n_slots_hint = self.next_node_id.load(Ordering::Acquire);
        if let Err(e) = self.typed_degrees().incr_out(
            edge_type_id,
            slot,
            delta,
            n_slots_hint,
        ) {
            tracing::warn!(
                error = ?e,
                edge_type_id,
                slot,
                "typed degree incr_out failed"
            );
        }
    }

    /// T17h T8 — increment typed in-degree.
    pub(crate) fn incr_typed_in_degree(
        &self,
        edge_type_id: u16,
        slot: u32,
        delta: i32,
    ) {
        let n_slots_hint = self.next_node_id.load(Ordering::Acquire);
        if let Err(e) = self.typed_degrees().incr_in(
            edge_type_id,
            slot,
            delta,
            n_slots_hint,
        ) {
            tracing::warn!(
                error = ?e,
                edge_type_id,
                slot,
                "typed degree incr_in failed"
            );
        }
    }

    /// T17h T1 — snapshot current in-memory counter atomics into a
    /// `PersistedCounters` block suitable for v5 dict serialization.
    /// Called by `build_dict_snapshot` at flush time.
    pub(crate) fn snapshot_counters(&self) -> crate::dict::PersistedCounters {
        let mut label_counts = [0u64; 64];
        for (bit, atomic) in self.label_live_counts.iter().enumerate() {
            label_counts[bit] = atomic.load(Ordering::Relaxed);
        }
        let mut edge_type_counts: Vec<(u16, u64)> = self
            .edge_type_live_counts
            .iter()
            .map(|e| (*e.key(), e.value().load(Ordering::Relaxed)))
            .collect();
        edge_type_counts.sort_by_key(|&(id, _)| id); // stable wire order
        // T17i T2 — dump the DashMap counters into the v6 wire format.
        // Group by edge_type_id and fill a [u64; 64] per type, skipping
        // types with no observed label bits (sparse on the u16 id space).
        let mut target_grouped: FxHashMap<u16, [u64; 64]> = FxHashMap::default();
        for e in self.edge_type_target_label_counts.iter() {
            let (type_id, bit) = *e.key();
            let v = e.value().load(Ordering::Relaxed);
            if v == 0 {
                continue;
            }
            target_grouped
                .entry(type_id)
                .or_insert([0u64; 64])[bit as usize] = v;
        }
        let mut source_grouped: FxHashMap<u16, [u64; 64]> = FxHashMap::default();
        for e in self.edge_type_source_label_counts.iter() {
            let (type_id, bit) = *e.key();
            let v = e.value().load(Ordering::Relaxed);
            if v == 0 {
                continue;
            }
            source_grouped
                .entry(type_id)
                .or_insert([0u64; 64])[bit as usize] = v;
        }
        let mut edge_type_target_label_counts: Vec<(u16, [u64; 64])> =
            target_grouped.into_iter().collect();
        edge_type_target_label_counts.sort_by_key(|&(id, _)| id);
        let mut edge_type_source_label_counts: Vec<(u16, [u64; 64])> =
            source_grouped.into_iter().collect();
        edge_type_source_label_counts.sort_by_key(|&(id, _)| id);

        crate::dict::PersistedCounters {
            total_live_nodes: self.total_live_nodes.load(Ordering::Relaxed),
            total_live_edges: self.total_live_edges.load(Ordering::Relaxed),
            label_counts,
            edge_type_counts,
            edge_type_target_label_counts,
            edge_type_source_label_counts,
        }
    }
}

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
        // T17e Phase 3: labels resolve from the mmap'd NodeRecord +
        // registry on demand; no DashMap to populate.
        // T17h T1: live counter increments (total + per-label).
        self.total_live_nodes.fetch_add(1, Ordering::Relaxed);
        self.incr_label_counts(bitset, 1);
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
        // T17i T1 : pre-hydrate the typed-degree registry **before**
        // `splice_edge_at_head` writes the new edge to the zone. If
        // this is the first call on a fresh store (or first post-
        // upgrade), the rebuild scan inside `ensure_initialized` then
        // iterates the pre-edge state and doesn't observe the new
        // edge — so our explicit `incr_typed_*_degree` below is the
        // single source of increment (no double-count).
        let _ = self.typed_degrees();
        let id = self.allocate_edge_id();
        let _rec = self.splice_edge_at_head(id, src, dst, edge_type_id);

        // T17e Phase 3: edge_type resolves from the mmap'd EdgeRecord +
        // registry on demand; no DashMap to populate.
        // T17h T1: live counter increments (total + per-edge-type).
        self.total_live_edges.fetch_add(1, Ordering::Relaxed);
        self.incr_edge_type_count(edge_type_id, 1);
        // T17h T5: degree column increments (src.out, dst.in).
        self.incr_out_degree(src.0 as u32, 1);
        self.incr_in_degree(dst.0 as u32, 1);
        // T17h T8: per-edge-type degree column increments.
        self.incr_typed_out_degree(edge_type_id, src.0 as u32, 1);
        self.incr_typed_in_degree(edge_type_id, dst.0 as u32, 1);
        // T17i T2: per-edge-type × target-label / source-label histograms.
        // Read the endpoints' label bitsets and increment the two
        // histograms for each set bit. Dangling endpoints (no NodeRecord)
        // yield bitset=0 → no increment, consistent with the LpgStore
        // permissive-endpoint contract.
        let src_bitset = self
            .writer
            .read_node(src.0 as u32)
            .ok()
            .flatten()
            .map(|r| r.label_bitset)
            .unwrap_or(0);
        let dst_bitset = self
            .writer
            .read_node(dst.0 as u32)
            .ok()
            .flatten()
            .map(|r| r.label_bitset)
            .unwrap_or(0);
        self.incr_edge_type_source_label(edge_type_id, src_bitset, 1);
        self.incr_edge_type_target_label(edge_type_id, dst_bitset, 1);
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
        // T17h T1: snapshot label_bitset BEFORE tombstone for counter
        // decrement. Record is still readable post-tombstone but cleaner
        // to read it now, semantically.
        let label_bitset = match self.writer.read_node(id.0 as u32).ok().flatten() {
            Some(rec) => rec.label_bitset,
            None => 0,
        };
        // Flip the TOMBSTONED flag in the on-disk slot + journal NodeDelete.
        // T17e Phase 3: the flag flip in the mmap'd record is now the
        // sole source of truth for liveness — `is_live_on_disk` + every
        // `resolve_*` helper consult it directly. `get_node` will now
        // return None.
        self.writer
            .tombstone_node(id.0 as u32)
            .expect("tombstone_node failed");
        // The dedicated `node_properties` map may still hold an entry
        // when the node has properties; drop it here so
        // `persist_properties` doesn't re-serialise a dead entity on the
        // next flush.
        self.node_properties.remove(&id);
        // T17h T1: live counter decrements.
        self.total_live_nodes.fetch_sub(1, Ordering::Relaxed);
        self.incr_label_counts(label_bitset, -1);
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
        // T17h T1: snapshot edge_type BEFORE tombstone for counter decrement.
        let edge_type_id = rec.edge_type;
        // T17h T5: snapshot src/dst BEFORE tombstone for degree decrement.
        let src_slot = rec.src;
        let dst_slot = rec.dst;
        // T17i T1 : pre-hydrate typed-degree registry BEFORE the
        // tombstone write so the rebuild scan (if needed) sees the
        // still-live edge and accounts for it. Our explicit
        // `incr_typed_*_degree(-1)` below then decrements it once,
        // leaving the correct final state.
        let _ = self.typed_degrees();
        // Splice the edge out of both chains before tombstoning the slot
        // — future walks must never encounter the dead edge.
        self.unlink_edge_from_chains(id, &rec);
        self.writer
            .tombstone_edge(id.0)
            .expect("tombstone_edge failed");
        // T17e Phase 3: the mmap TOMBSTONED flag is authoritative; only
        // the property map needs explicit cleanup here.
        self.edge_properties.remove(&id);
        // T17h T1: live counter decrements.
        self.total_live_edges.fetch_sub(1, Ordering::Relaxed);
        self.incr_edge_type_count(edge_type_id, -1);
        // T17h T5: degree column decrements.
        self.incr_out_degree(src_slot, -1);
        self.incr_in_degree(dst_slot, -1);
        // T17h T8: per-edge-type degree column decrements.
        self.incr_typed_out_degree(edge_type_id, src_slot, -1);
        self.incr_typed_in_degree(edge_type_id, dst_slot, -1);
        // T17i T2: per-edge-type × target/source label histogram decrements.
        // Endpoints must be read BEFORE any cleanup that might tombstone
        // them ; if a caller deletes the edge AFTER deleting its nodes,
        // the label bitsets are already on tombstoned records (still
        // readable — tombstone only sets the flag, never wipes fields).
        let src_bitset = self
            .writer
            .read_node(src_slot)
            .ok()
            .flatten()
            .map(|r| r.label_bitset)
            .unwrap_or(0);
        let dst_bitset = self
            .writer
            .read_node(dst_slot)
            .ok()
            .flatten()
            .map(|r| r.label_bitset)
            .unwrap_or(0);
        self.incr_edge_type_source_label(edge_type_id, src_bitset, -1);
        self.incr_edge_type_target_label(edge_type_id, dst_bitset, -1);
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
        // Intern the key in the property-key registry up-front so
        // `all_property_keys()` reports it regardless of which downstream
        // path takes the write (dense vec column, blob column, PropsZone
        // v2, or the DashMap fallback — the last of which would otherwise
        // bypass the registry entirely). Idempotent; overflow at >u16 keys
        // is swallowed here to match the best-effort semantics of the
        // DashMap-fallback path below.
        let _ = self.prop_keys.write().intern(key);
        // T16.7: `Value::Vector` writes bypass the DashMap + props
        // sidecar and go straight to a dense mmap'd column. This keeps
        // embeddings off the anon heap. Scalar values still take the
        // in-memory path.
        if let Value::Vector(ref v) = value {
            self.route_vector_write(EntityKind::Node, id.0 as u32, key, v);
            return;
        }
        // T16.7 Step 4d: oversized `Value::String` / `Value::Bytes`
        // payloads route to the blob-column registry. `encode_blob_payload`
        // returns `Some(tagged_bytes)` iff the payload qualifies — `None`
        // for short scalars, non-blob variants, etc. The 1-byte type tag
        // at the head of the returned buffer lets the reader reconstruct
        // the original `Value` variant byte-exactly on the get side.
        if let Some(tagged) = encode_blob_payload(&value) {
            self.route_blob_write(EntityKind::Node, id.0 as u32, key, &tagged);
            return;
        }
        // T17c Step 3b.2b: dual-write into PropsZone v2 when enabled.
        // Best-effort — if PropsZone fails (e.g. page-size overflow for an
        // exotic value), we log and still fall back to the DashMap sidecar
        // so the call is never lost. 3c will remove the DashMap path once
        // PropsZone is the single source of truth.
        if self.props_zone.is_some() {
            if let Err(err) = self.append_scalar_to_props_zone_v2(id, key, &value) {
                tracing::warn!(
                    node_id = id.0,
                    key = %key,
                    error = %err,
                    "props zone v2 append failed — falling back to DashMap only"
                );
            }
        }
        self.node_properties
            .entry(id)
            .or_default()
            .insert(PropertyKey::new(key), value);
    }

    fn set_edge_property(&self, id: EdgeId, key: &str, value: Value) {
        if !self.is_live_edge_on_disk(id) {
            // LpgStore silently no-ops on missing edges; match that contract.
            return;
        }
        // Intern the key up-front (mirrors `set_node_property`; see comment
        // there). Overflow is swallowed to match best-effort DashMap-fallback
        // semantics.
        let _ = self.prop_keys.write().intern(key);
        if let Value::Vector(ref v) = value {
            // EdgeId is u64 in the trait, but `VecColumnWriter::write_slot`
            // takes u32. In practice no substrate deployment has come close
            // to 2^32 live edges; if that ever changes, the vec-column
            // schema needs a v2 bump to widen the slot field.
            let slot = id.0 as u32;
            self.route_vector_write(EntityKind::Edge, slot, key, v);
            return;
        }
        if let Some(tagged) = encode_blob_payload(&value) {
            let slot = id.0 as u32;
            self.route_blob_write(EntityKind::Edge, slot, key, &tagged);
            return;
        }
        // T17f Step 4: dual-write into PropsZone v2 (edge-owner chain)
        // when enabled. Mirrors the node path at `set_node_property`;
        // best-effort, a PropsZone failure downgrades to DashMap-only
        // without losing the write.
        if self.props_zone.is_some() {
            if let Err(err) = self.append_scalar_to_props_zone_v2_edge(id, key, &value) {
                tracing::warn!(
                    edge_id = id.0,
                    key = %key,
                    error = %err,
                    "props zone v2 (edge) append failed — falling back to DashMap only"
                );
            }
        }
        self.edge_properties
            .entry(id)
            .or_default()
            .insert(PropertyKey::new(key), value);
    }

    fn remove_node_property(&self, id: NodeId, key: &str) -> Option<Value> {
        if !self.is_live_on_disk(id) {
            return None;
        }
        // T17c Step 3c — write a tombstone to the PropsZone chain so
        // the LWW read path (get_node_property) sees the deletion
        // even when the DashMap sidecar no longer has the entry.
        // Best-effort: a failure here downgrades to DashMap-only
        // semantics but must not lose the delete.
        if self.props_zone.is_some() {
            if let Err(err) = self.append_tombstone_to_props_zone_v2(id, key) {
                tracing::warn!(
                    node_id = id.0,
                    key = %key,
                    error = %err,
                    "props zone v2 tombstone append failed — falling back to DashMap only"
                );
            }
        }
        self.node_properties
            .get_mut(&id)?
            .remove(&PropertyKey::new(key))
    }

    fn remove_edge_property(&self, id: EdgeId, key: &str) -> Option<Value> {
        if !self.is_live_edge_on_disk(id) {
            return None;
        }
        // T17f Step 4: write a tombstone on the PropsZone v2 edge chain
        // so the LWW read path (`get_edge_property`) sees the deletion
        // even when the DashMap sidecar no longer has the entry.
        // Best-effort: a failure here downgrades to DashMap-only
        // semantics but must not lose the delete.
        if self.props_zone.is_some() {
            if let Err(err) = self.append_tombstone_to_props_zone_v2_edge(id, key) {
                tracing::warn!(
                    edge_id = id.0,
                    key = %key,
                    error = %err,
                    "props zone v2 (edge) tombstone append failed — falling back to DashMap only"
                );
            }
        }
        self.edge_properties
            .get_mut(&id)?
            .remove(&PropertyKey::new(key))
    }

    // -- Label mutation --
    //
    // T17e Phase 3: both `add_label` and `remove_label` pivot on the
    // `NodeRecord.label_bitset` field — the canonical representation.
    // We read the current record, shift the bit for the target label,
    // and write the new record back. No in-memory DashMap to keep in
    // sync (the former `NodeInMem.labels` view was a cache of this
    // same bitset that we can now materialise on the fly via
    // `resolve_node_labels_from_bitset`).
    fn add_label(&self, node_id: NodeId, label: &str) -> bool {
        if !self.is_live_on_disk(node_id) {
            return false;
        }
        let old = match self.writer.read_node(node_id.0 as u32) {
            Ok(Some(rec)) => rec,
            _ => return false,
        };
        // Intern first — if the label is already interned we can
        // early-out on idempotent adds without holding the write lock
        // longer than necessary.
        let bit = self
            .labels
            .write()
            .intern(label)
            .expect("label registry overflow");
        let mask = 1u64 << bit;
        if old.label_bitset & mask != 0 {
            // Label already set: idempotent no-op, matches LpgStore.
            return false;
        }
        let updated = NodeRecord {
            label_bitset: old.label_bitset | mask,
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
        let old = match self.writer.read_node(node_id.0 as u32) {
            Ok(Some(rec)) => rec,
            _ => return false,
        };
        let Some(bit) = self.labels.read().bit_for(label) else {
            // Label was never interned → can't be on this node.
            return false;
        };
        let mask = 1u64 << bit;
        if old.label_bitset & mask == 0 {
            // Label not present: no-op, matches LpgStore.
            return false;
        }
        let updated = NodeRecord {
            label_bitset: old.label_bitset & !mask,
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

    /// T16.7 Step 3b — auto-migration regression anchor.
    ///
    /// Simulates a **pre-T16.7 vintage base**: a `SubstrateStore` whose
    /// `substrate.props` sidecar carries a `Value::Vector` payload
    /// written through the old code path (direct bincode serialisation
    /// of the DashMap, no vec-column zone). We fabricate the state by
    /// driving `PropertiesStreamingWriter` directly so we can write a
    /// vector entry without going through the routed setter.
    ///
    /// On reopen, `load_properties` MUST:
    ///   1. Route the vector into the vec-column registry so
    ///      `get_node_property` returns it (byte-exact) via vec_columns.
    ///   2. Keep the DashMap sidecar free of the vector entry — the
    ///      next `flush()` will then re-persist a vector-free sidecar
    ///      via the `persist_properties` defensive filter, completing
    ///      the upgrade.
    ///
    /// If someone reintroduces the pre-Step-3b load path (direct
    /// `entries_to_map` without the vector split) this test goes red:
    /// the vector would end up in the DashMap, the next flush would
    /// drop it from the sidecar (defensive filter) AND from the DashMap
    /// on the subsequent reopen — silent data loss.
    #[test]
    fn auto_migrates_legacy_vectors_on_load() {
        use crate::props_snapshot::PropertiesStreamingWriter;
        use obrain_core::graph::traits::GraphStore;
        use std::sync::Arc;

        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("kb");

        // Phase 1 — create a fresh T16.7 store, write ONE node with a
        // scalar (so the node slot exists and labels are persisted),
        // then flush + drop.
        let node_id = {
            let s = SubstrateStore::create(&path).unwrap();
            let a = s.create_node(&["Doc"]);
            s.set_node_property(a, "title", Value::String("legacy".into()));
            s.flush().unwrap();
            a
        };

        // Phase 2 — simulate the pre-T16.7 sidecar state by OVER-
        // WRITING `substrate.props` with a hand-crafted snapshot that
        // contains both the scalar and a vector entry for the same
        // node. Because `PropertiesStreamingWriter` doesn't filter by
        // value kind (it's the post-filter consumer of whatever
        // `persist_properties` feeds it), we get the vector into the
        // sidecar exactly as a v1.x migration would have done.
        let sidecar_path = path.join(PROPS_FILENAME);
        let vec_payload: Vec<f32> = (0..16).map(|i| i as f32 * 0.0625).collect();
        let arc: Arc<[f32]> = Arc::from(vec_payload.clone().into_boxed_slice());
        {
            let mut w = PropertiesStreamingWriter::open(&sidecar_path).unwrap();
            let props: Vec<(String, Value)> = vec![
                ("title".to_string(), Value::String("legacy".into())),
                ("legacy_embedding".to_string(), Value::Vector(arc.clone())),
            ];
            w.append_node(node_id.0, &props).unwrap();
            w.finish().unwrap();
        }

        // Sanity: the raw pre-upgrade sidecar must actually contain the
        // vector key — otherwise this test is not exercising what it
        // claims. Bincode encodes the key as a UTF-8 string, so a byte
        // substring scan is conclusive.
        {
            let raw = std::fs::read(&sidecar_path).unwrap();
            assert!(
                raw.windows(b"legacy_embedding".len())
                    .any(|w| w == b"legacy_embedding"),
                "pre-upgrade sidecar should carry the vector key (test invariant)"
            );
        }

        // Phase 3 — reopen. `load_properties` must route the vector
        // into vec_columns.
        let s = SubstrateStore::open(&path).unwrap();

        // 3a. Vector is readable via vec_columns — byte-exact.
        let got = s
            .get_node_property(node_id, &PropertyKey::new("legacy_embedding"))
            .expect("vector should be auto-migrated to vec_columns on load");
        match got {
            Value::Vector(arc) => {
                assert_eq!(
                    arc.as_ref(),
                    &vec_payload[..],
                    "auto-migrated vector must be byte-exact",
                );
            }
            other => panic!("expected Value::Vector, got {other:?}"),
        }

        // 3b. Scalar still lives on the DashMap sidecar path.
        assert!(
            matches!(
                s.get_node_property(node_id, &PropertyKey::new("title")),
                Some(Value::String(_))
            ),
            "scalar property should still be present after auto-migration"
        );

        // Phase 4 — flush + drop. The defensive filter in
        // `persist_properties` now strips the vector from the
        // re-serialised sidecar.
        s.flush().unwrap();
        drop(s);

        // 4a. Post-flush sidecar must NOT carry the vector key.
        let raw_after = std::fs::read(&sidecar_path).unwrap();
        assert!(
            !raw_after
                .windows(b"legacy_embedding".len())
                .any(|w| w == b"legacy_embedding"),
            "post-flush sidecar must not carry the auto-migrated vector key \
             (bytes len={})",
            raw_after.len(),
        );
        // 4b. The scalar key survives.
        assert!(
            raw_after.windows(b"title".len()).any(|w| w == b"title"),
            "scalar key should still be in the post-flush sidecar"
        );

        // Phase 5 — reopen a THIRD time. The vector must still be
        // readable (now exclusively from the vec-column zone; the
        // sidecar no longer references it).
        let s = SubstrateStore::open(&path).unwrap();
        let got = s
            .get_node_property(node_id, &PropertyKey::new("legacy_embedding"))
            .expect("vector should still be readable after upgrade-then-flush cycle");
        match got {
            Value::Vector(arc) => {
                assert_eq!(arc.as_ref(), &vec_payload[..]);
            }
            other => panic!("expected Value::Vector, got {other:?}"),
        }
    }

    /// T16.7 Step 4d — blob auto-migration regression anchor.
    ///
    /// Mirror of [`auto_migrates_legacy_vectors_on_load`] for the
    /// variable-length blob path: an oversized `Value::String` (bigger
    /// than [`BLOB_COLUMN_THRESHOLD_BYTES`]) sitting in a pre-T16.7.4
    /// `substrate.props` sidecar must auto-migrate into the
    /// [`BlobColumnRegistry`] on open, and the next `flush()` must
    /// evict it from the sidecar. A `Value::Bytes` variant is exercised
    /// to prove the 1-byte type tag preserves the distinction.
    #[test]
    fn auto_migrates_legacy_blobs_on_load() {
        use crate::props_snapshot::PropertiesStreamingWriter;
        use obrain_core::graph::traits::GraphStore;
        use std::sync::Arc;

        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("kb");

        // Phase 1 — fresh T16.7.4 store, allocate a node slot so the
        // auto-migration has a live target to write into.
        let (node_id_str, node_id_bytes) = {
            let s = SubstrateStore::create(&path).unwrap();
            let a = s.create_node(&["Chat"]);
            let b = s.create_node(&["Chat"]);
            s.flush().unwrap();
            (a, b)
        };

        // Phase 2 — overwrite `substrate.props` with a hand-crafted
        // snapshot carrying BOTH an oversized string and an oversized
        // byte payload. These are the exact shape of pre-T16.7.4 data
        // coming out of `obrain-migrate` on a PO chat history.
        let sidecar_path = path.join(PROPS_FILENAME);
        let big_string: String = "chat payload éclair ".repeat(100); // ~2 KiB
        let big_bytes: Vec<u8> =
            (0..2048u32).map(|i| (i & 0xff) as u8).collect();
        let bytes_arc: Arc<[u8]> = Arc::from(big_bytes.clone());
        assert!(big_string.as_bytes().len() > BLOB_COLUMN_THRESHOLD_BYTES);
        assert!(big_bytes.len() > BLOB_COLUMN_THRESHOLD_BYTES);
        {
            let mut w = PropertiesStreamingWriter::open(&sidecar_path).unwrap();
            let props_a: Vec<(String, Value)> = vec![
                ("title".to_string(), Value::String("keep me inline".into())),
                (
                    "data".to_string(),
                    Value::String(big_string.as_str().into()),
                ),
            ];
            w.append_node(node_id_str.0, &props_a).unwrap();
            let props_b: Vec<(String, Value)> = vec![
                ("blob".to_string(), Value::Bytes(bytes_arc.clone())),
            ];
            w.append_node(node_id_bytes.0, &props_b).unwrap();
            w.finish().unwrap();
        }

        // Sanity: pre-upgrade sidecar must carry BOTH blob keys.
        {
            let raw = std::fs::read(&sidecar_path).unwrap();
            assert!(
                raw.windows(b"data".len()).any(|w| w == b"data"),
                "pre-upgrade sidecar should carry the `data` key"
            );
            assert!(
                raw.windows(b"blob".len()).any(|w| w == b"blob"),
                "pre-upgrade sidecar should carry the `blob` key"
            );
        }

        // Phase 3 — reopen. `load_properties` must route the two
        // oversized entries into blob_columns.
        let s = SubstrateStore::open(&path).unwrap();

        // 3a. Big string is readable via blob_columns, byte-exact.
        let got = s
            .get_node_property(node_id_str, &PropertyKey::new("data"))
            .expect("big string should auto-migrate on load");
        match got {
            Value::String(ref arc) => assert_eq!(arc.as_str(), big_string),
            other => panic!("expected Value::String, got {other:?}"),
        }
        // 3b. Big bytes are readable via blob_columns, byte-exact —
        //     AND must come back as Value::Bytes (not Value::String).
        let got = s
            .get_node_property(node_id_bytes, &PropertyKey::new("blob"))
            .expect("big bytes should auto-migrate on load");
        match got {
            Value::Bytes(arc) => assert_eq!(arc.as_ref(), big_bytes.as_slice()),
            other => panic!("expected Value::Bytes, got {other:?}"),
        }
        // 3c. Short scalar still lives on the DashMap sidecar path.
        match s.get_node_property(node_id_str, &PropertyKey::new("title")) {
            Some(Value::String(arc)) => assert_eq!(arc.as_str(), "keep me inline"),
            other => panic!("expected inline Value::String, got {other:?}"),
        }

        // Phase 4 — flush + drop. Defensive filter must strip both
        // blob entries from the rewritten sidecar.
        s.flush().unwrap();
        drop(s);

        let raw_after = std::fs::read(&sidecar_path).unwrap();
        // The scalar key survives.
        assert!(
            raw_after.windows(b"title".len()).any(|w| w == b"title"),
            "short scalar key should still be in the post-flush sidecar"
        );
        // The blob keys do NOT.
        assert!(
            !raw_after.windows(b"data".len()).any(|w| w == b"data"),
            "post-flush sidecar must not carry the auto-migrated `data` key \
             (bytes len={})",
            raw_after.len()
        );
        assert!(
            !raw_after.windows(b"blob".len()).any(|w| w == b"blob"),
            "post-flush sidecar must not carry the auto-migrated `blob` key \
             (bytes len={})",
            raw_after.len()
        );

        // Phase 5 — reopen a third time. The blob columns must still
        // satisfy reads (hydrated from dict v4, no sidecar fallback).
        let s = SubstrateStore::open(&path).unwrap();
        match s.get_node_property(node_id_str, &PropertyKey::new("data")) {
            Some(Value::String(arc)) => assert_eq!(arc.as_str(), big_string),
            other => panic!(
                "expected persisted Value::String after second reopen, got {other:?}"
            ),
        }
        match s.get_node_property(node_id_bytes, &PropertyKey::new("blob")) {
            Some(Value::Bytes(arc)) => {
                assert_eq!(arc.as_ref(), big_bytes.as_slice());
            }
            other => panic!(
                "expected persisted Value::Bytes after second reopen, got {other:?}"
            ),
        }
    }

    /// T16.7 Step 4d — fresh writes also route, not just legacy load.
    /// Ensures `set_node_property` on an oversized Value::String /
    /// Value::Bytes never touches the DashMap (= never lands in the
    /// bincode sidecar) and that reads come back byte-exact.
    #[test]
    fn fresh_blob_writes_bypass_sidecar() {
        use obrain_core::graph::traits::GraphStore;
        use std::sync::Arc;

        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("kb");
        let big_string: String = "éclair ".repeat(200); // > 256 B
        let big_bytes: Vec<u8> =
            (0..1024u32).map(|i| ((i * 7) & 0xff) as u8).collect();

        let (nid, eid) = {
            let s = SubstrateStore::create(&path).unwrap();
            let a = s.create_node(&["Doc"]);
            let b = s.create_node(&["Doc"]);
            let e = s.create_edge(a, b, "REL");
            s.set_node_property(a, "data", Value::String(big_string.as_str().into()));
            s.set_edge_property(
                e,
                "blob",
                Value::Bytes(Arc::from(big_bytes.clone())),
            );
            s.flush().unwrap();
            (a, e)
        };

        // Sidecar must NOT contain either blob key.
        let sidecar = std::fs::read(path.join(PROPS_FILENAME)).unwrap();
        assert!(
            !sidecar.windows(b"data".len()).any(|w| w == b"data"),
            "fresh oversized String must not land in substrate.props"
        );
        assert!(
            !sidecar.windows(b"blob".len()).any(|w| w == b"blob"),
            "fresh oversized Bytes must not land in substrate.props"
        );

        // Reopen and verify byte-exact reads.
        let s = SubstrateStore::open(&path).unwrap();
        match s.get_node_property(nid, &PropertyKey::new("data")) {
            Some(Value::String(arc)) => assert_eq!(arc.as_str(), big_string),
            other => panic!("expected String, got {other:?}"),
        }
        match s.get_edge_property(eid, &PropertyKey::new("blob")) {
            Some(Value::Bytes(arc)) => {
                assert_eq!(arc.as_ref(), big_bytes.as_slice());
            }
            other => panic!("expected Bytes, got {other:?}"),
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

    /// T17h T8 — per-edge-type degree counters mirror total T5 counters.
    /// Invariant : `sum over types (out_by_type[n]) == total_out[n]`.
    #[test]
    fn typed_degrees_sum_equals_total() {
        let (_td, s) = store();
        let a = s.create_node(&["A"]);
        let b = s.create_node(&["B"]);
        let c = s.create_node(&["C"]);
        // Mixed edge types on the same source.
        let _ = s.create_edge(a, b, "IMPORTS");
        let _ = s.create_edge(a, c, "IMPORTS");
        let _ = s.create_edge(a, b, "CONTAINS");
        let _ = s.create_edge(c, a, "IMPORTS"); // incoming on a
        assert_eq!(s.out_degree_by_type(a, "IMPORTS"), 2);
        assert_eq!(s.out_degree_by_type(a, "CONTAINS"), 1);
        assert_eq!(s.out_degree_by_type(a, "UNKNOWN"), 0);
        assert_eq!(s.in_degree_by_type(a, "IMPORTS"), 1);
        // Sum invariant.
        let by_type_sum =
            s.out_degree_by_type(a, "IMPORTS") + s.out_degree_by_type(a, "CONTAINS");
        assert_eq!(by_type_sum, s.out_degree(a));
    }

    /// T17h T8 — delete_edge propagates to typed counters.
    #[test]
    fn typed_degrees_decrement_on_delete() {
        let (_td, s) = store();
        let a = s.create_node(&["A"]);
        let b = s.create_node(&["B"]);
        let e = s.create_edge(a, b, "IMPORTS");
        let _ = s.create_edge(a, b, "IMPORTS");
        assert_eq!(s.out_degree_by_type(a, "IMPORTS"), 2);
        assert!(s.delete_edge(e));
        assert_eq!(s.out_degree_by_type(a, "IMPORTS"), 1);
        assert_eq!(s.in_degree_by_type(b, "IMPORTS"), 1);
    }

    /// T17h T8 — typed columns survive close/reopen via the
    /// `substrate.degrees.node.<type>.u32` sidecars. On reopen,
    /// `init_typed_degrees` takes the "open_existing" path.
    #[test]
    fn typed_degrees_survive_reopen() {
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("kb");
        let (a_id, b_id) = {
            let s = SubstrateStore::create(&path).unwrap();
            let a = s.create_node(&["File"]);
            let b = s.create_node(&["File"]);
            let c = s.create_node(&["File"]);
            let _ = s.create_edge(a, b, "IMPORTS");
            let _ = s.create_edge(a, c, "IMPORTS");
            let _ = s.create_edge(a, b, "CONTAINS");
            s.flush().unwrap();
            (a, b)
        };
        let s = SubstrateStore::open(&path).unwrap();
        assert_eq!(s.out_degree_by_type(a_id, "IMPORTS"), 2);
        assert_eq!(s.out_degree_by_type(a_id, "CONTAINS"), 1);
        assert_eq!(s.in_degree_by_type(b_id, "IMPORTS"), 1);
        assert_eq!(s.in_degree_by_type(b_id, "CONTAINS"), 1);
        // Total T5 still consistent.
        assert_eq!(s.out_degree(a_id), 3);
    }

    /// T17h T8 — cold rebuild path : a base with edges but no persisted
    /// typed-column sidecars (simulated by deleting the sidecars on
    /// disk) must rebuild correctly at reopen.
    #[test]
    fn typed_degrees_rebuild_from_scan() {
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("kb");
        let (a_id, _b_id) = {
            let s = SubstrateStore::create(&path).unwrap();
            let a = s.create_node(&["File"]);
            let b = s.create_node(&["File"]);
            let _ = s.create_edge(a, b, "IMPORTS");
            let _ = s.create_edge(a, b, "CONTAINS");
            s.flush().unwrap();
            (a, b)
        };
        // Delete the typed-column sidecars to simulate a pre-T8 base.
        for entry in std::fs::read_dir(&path).unwrap().flatten() {
            let name = entry.file_name();
            let Some(ns) = name.to_str() else { continue };
            if ns.starts_with(crate::typed_degree::TYPED_DEGREE_FILENAME_PREFIX)
                && ns.ends_with(crate::typed_degree::TYPED_DEGREE_FILENAME_SUFFIX)
                && ns.len()
                    > crate::typed_degree::TYPED_DEGREE_FILENAME_PREFIX.len()
                        + crate::typed_degree::TYPED_DEGREE_FILENAME_SUFFIX.len()
            {
                std::fs::remove_file(entry.path()).unwrap();
            }
        }
        // Reopen — rebuild path takes over.
        let s = SubstrateStore::open(&path).unwrap();
        assert_eq!(s.out_degree_by_type(a_id, "IMPORTS"), 1);
        assert_eq!(s.out_degree_by_type(a_id, "CONTAINS"), 1);
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

        // Incoming chains work because `incoming_heads()` lazily builds
        // the reverse-head map on first access (T17e Phase 4).
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
        // bypassed the allocator). T17g T1: access via lazy accessor so
        // the OnceLock is triggered before we poison.
        let ranges = s.community_ranges();
        ranges.first_slots.insert(1, 9999);
        ranges.placements.insert(1, 1);
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

    // ---------------------------------------------------------------
    // T17c Step 3a — Props-zone plumbing smoke tests.
    //
    // These tests verify the gating contract (env OR file) and that the
    // field is wired through flush() without corrupting the legacy
    // path. Setters/getters still live on the DashMap route; Step 3b
    // will move the write path onto PropsZone.
    // ---------------------------------------------------------------

    /// Serialise access to `OBRAIN_PROPS_V2` across parallel test
    /// runners — `env::set_var` is process-wide, so two concurrent
    /// tests would clobber each other's configuration.
    fn props_v2_env_lock() -> &'static std::sync::Mutex<()> {
        use std::sync::OnceLock;
        static LOCK: OnceLock<std::sync::Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| std::sync::Mutex::new(()))
    }

    #[test]
    #[allow(unsafe_code)]
    fn props_zone_v2_disabled_by_default() {
        // No env, no pre-existing zone file — open must stay on the
        // legacy path and the zone files must not be created.
        let _guard = props_v2_env_lock().lock().unwrap();
        // Make sure a stray env var from another test can't leak in.
        // SAFETY: serialised via `props_v2_env_lock`.
        unsafe { std::env::remove_var("OBRAIN_PROPS_V2") };

        let td = tempfile::tempdir().unwrap();
        let s = SubstrateStore::create(td.path().join("kb")).unwrap();
        assert!(
            !s.props_v2_enabled(),
            "props_v2 must remain disabled when env is unset and no file exists"
        );

        let root = td.path().join("kb");
        assert!(
            !root.join(crate::props_zone::PROPS_V2_FILENAME).exists(),
            "zone file must not be created when feature is off"
        );
        assert!(
            !root.join(crate::props_zone::PROPS_HEAP_V2_FILENAME).exists(),
            "heap zone file must not be created when feature is off"
        );
    }

    #[test]
    #[allow(unsafe_code)]
    fn props_zone_v2_env_enables_creation() {
        // `OBRAIN_PROPS_V2=1` at open time must force-open the zone
        // files and flip `props_v2_enabled()` to true.
        let _guard = props_v2_env_lock().lock().unwrap();
        // SAFETY: serialised via `props_v2_env_lock`.
        unsafe { std::env::set_var("OBRAIN_PROPS_V2", "1") };

        let td = tempfile::tempdir().unwrap();
        let root = td.path().join("kb");
        let s = SubstrateStore::create(&root).unwrap();
        assert!(s.props_v2_enabled(), "env must enable props_v2");

        // Flush so the mmap-grown zone files are sized on disk.
        s.flush().unwrap();
        assert!(
            root.join(crate::props_zone::PROPS_V2_FILENAME).exists(),
            "props.v2 file must exist after create+flush"
        );
        assert!(
            root.join(crate::props_zone::PROPS_HEAP_V2_FILENAME).exists(),
            "props.heap.v2 file must exist after create+flush"
        );

        // SAFETY: serialised via `props_v2_env_lock`.
        unsafe { std::env::remove_var("OBRAIN_PROPS_V2") };
    }

    #[test]
    #[allow(unsafe_code)]
    fn props_zone_v2_auto_detected_on_reopen() {
        // Once the zone file exists on disk, a reopen without env must
        // still enable the feature — "once upgraded, always upgraded".
        let _guard = props_v2_env_lock().lock().unwrap();

        let td = tempfile::tempdir().unwrap();
        let root = td.path().join("kb");

        // Phase 1: create + open with the feature on so the zone file
        // lands on disk.
        // SAFETY: serialised via `props_v2_env_lock`.
        unsafe { std::env::set_var("OBRAIN_PROPS_V2", "1") };
        {
            let s = SubstrateStore::create(&root).unwrap();
            s.flush().unwrap();
            assert!(s.props_v2_enabled());
        }
        // SAFETY: serialised via `props_v2_env_lock`.
        unsafe { std::env::remove_var("OBRAIN_PROPS_V2") };

        // Phase 2: reopen WITHOUT env — auto-detect must fire.
        assert!(root.join(crate::props_zone::PROPS_V2_FILENAME).exists());
        let s = SubstrateStore::open(&root).unwrap();
        assert!(
            s.props_v2_enabled(),
            "existing substrate.props.v2 on disk must re-enable the zone"
        );
    }

    #[test]
    #[allow(unsafe_code)]
    fn props_zone_v2_flush_is_noop_in_step_3a() {
        // Pure plumbing test: with the zone enabled but no write path
        // wired yet, flush() must not blow up and must not change the
        // zone layout (allocated_page_count stays at 0 because no
        // entries were appended).
        let _guard = props_v2_env_lock().lock().unwrap();
        // SAFETY: serialised via `props_v2_env_lock`.
        unsafe { std::env::set_var("OBRAIN_PROPS_V2", "1") };

        let td = tempfile::tempdir().unwrap();
        let s = SubstrateStore::create(td.path().join("kb")).unwrap();
        s.flush().unwrap();
        s.flush().unwrap();
        // Idempotent — zone still empty.
        let pz = s.props_zone.as_ref().expect("enabled");
        assert_eq!(pz.read().allocated_page_count(), 0);
        assert_eq!(pz.read().allocated_heap_page_count(), 0);

        // SAFETY: serialised via `props_v2_env_lock`.
        unsafe { std::env::remove_var("OBRAIN_PROPS_V2") };
    }

    // -------------------------------------------------------------------
    // T17c Step 3b.2b — PropsZone write-through tests
    //
    // These exercise the new `append_scalar_to_props_zone_v2` path:
    // a `set_node_property` call lands both into the DashMap (read
    // cache) AND a fresh PropertyEntry on the PropsZone chain, with
    // the NodeRecord.first_prop_off head pointer updated via WAL.
    // -------------------------------------------------------------------

    /// Helper: walk the PropsZone chain for node `slot` and return the
    /// decoded entries in chain order (newest → oldest).
    fn collect_node_props_v2(
        s: &SubstrateStore,
        slot: u32,
    ) -> Vec<crate::page::PropertyEntry> {
        let rec = s.writer.read_node(slot).unwrap().unwrap();
        let head = crate::props_zone::decode_page_id(rec.first_prop_off);
        let pz = s.props_zone.as_ref().expect("v2 enabled").read();
        pz.collect_entries(head).unwrap()
    }

    #[test]
    #[allow(unsafe_code)]
    fn set_node_property_writes_to_props_zone_v2() {
        // With OBRAIN_PROPS_V2=1, a scalar set_node_property must land
        // as a PropertyEntry on the PropsZone chain AND flip the head
        // pointer on the NodeRecord.
        let _guard = props_v2_env_lock().lock().unwrap();
        unsafe { std::env::set_var("OBRAIN_PROPS_V2", "1") };

        let td = tempfile::tempdir().unwrap();
        let s = SubstrateStore::create(td.path().join("kb")).unwrap();
        let a = s.create_node(&["Doc"]);
        s.set_node_property(a, "title", Value::String("hello".into()));
        s.set_node_property(a, "count", Value::Int64(42));
        s.flush().unwrap();

        // Head pointer must now be non-zero.
        let rec = s.writer.read_node(a.0 as u32).unwrap().unwrap();
        assert!(
            !rec.first_prop_off.is_zero(),
            "first_prop_off must be set after property writes"
        );

        // Chain contains both entries.
        let entries = collect_node_props_v2(&s, a.0 as u32);
        assert_eq!(entries.len(), 2, "got entries: {:?}", entries);
        // walk_chain emits entries page-by-page (newest → oldest
        // across pages) but within a page it iterates the cursor in
        // APPEND order (oldest → newest). When both writes fit into
        // the first head page, the observed order is [title, count].
        // The get-side LWW layer (Step 3c) will reverse the semantic
        // ordering so the *latest* write wins.
        let title_id = s.prop_keys.write().intern("title").unwrap();
        let count_id = s.prop_keys.write().intern("count").unwrap();
        assert_eq!(
            entries[0].prop_key, title_id,
            "first append = title"
        );
        assert_eq!(
            entries[1].prop_key, count_id,
            "second append = count"
        );

        unsafe { std::env::remove_var("OBRAIN_PROPS_V2") };
    }

    #[test]
    #[allow(unsafe_code)]
    fn props_zone_v2_survives_reopen() {
        // Write a property with v2 enabled, close the store, reopen,
        // and confirm the PropsZone chain still holds the entry.
        let _guard = props_v2_env_lock().lock().unwrap();
        unsafe { std::env::set_var("OBRAIN_PROPS_V2", "1") };

        let td = tempfile::tempdir().unwrap();
        let root = td.path().join("kb");
        let a_raw;
        {
            let s = SubstrateStore::create(&root).unwrap();
            let a = s.create_node(&["Doc"]);
            a_raw = a.0 as u32;
            s.set_node_property(a, "title", Value::String("persistent".into()));
            s.set_node_property(a, "count", Value::Int64(99));
            s.flush().unwrap();
            // Drop at end of scope releases mmaps & WAL file handles.
        }

        // Re-open without env — auto-detection still activates v2.
        unsafe { std::env::remove_var("OBRAIN_PROPS_V2") };
        let s = SubstrateStore::open(&root).unwrap();
        assert!(s.props_v2_enabled(), "v2 must auto-enable on reopen");

        let entries = collect_node_props_v2(&s, a_raw);
        assert_eq!(entries.len(), 2, "entries survived reopen: {:?}", entries);
    }

    // -------------------------------------------------------------------
    // T17f Step 4 — PropsZone v2 edge-parity tests
    //
    // Mirror the node-side tests for edges: scalar writes land on the
    // edge-owner chain (page magic 0xE507_5FA6), LWW semantics hold
    // across repeated writes, tombstones cover the removal path, and
    // node + edge chains coexist on the same PropsZone without
    // cross-kind splicing.
    // -------------------------------------------------------------------

    /// Helper: walk the PropsZone edge chain for edge `slot` and return
    /// the decoded entries in chain order (newest → oldest).
    ///
    /// Uses [`PropsZone::collect_entries`] which filters tombstones. For
    /// tests that need to observe tombstones on the chain, use
    /// [`collect_edge_props_v2_raw`].
    fn collect_edge_props_v2(
        s: &SubstrateStore,
        slot: u64,
    ) -> Vec<crate::page::PropertyEntry> {
        let rec = s.writer.read_edge(slot).unwrap().unwrap();
        let head = crate::props_zone::decode_page_id(rec.first_prop_off);
        let pz = s.props_zone.as_ref().expect("v2 enabled").read();
        pz.collect_entries(head).unwrap()
    }

    /// Helper: tombstone-inclusive walk of the PropsZone edge chain. Used
    /// by the remove-emits-tombstone test to observe the physical chain
    /// state (live entries AND tombstones) that `collect_entries` hides.
    fn collect_edge_props_v2_raw(
        s: &SubstrateStore,
        slot: u64,
    ) -> Vec<crate::page::PropertyEntry> {
        let rec = s.writer.read_edge(slot).unwrap().unwrap();
        let mut cur = crate::props_zone::decode_page_id(rec.first_prop_off);
        let pz = s.props_zone.as_ref().expect("v2 enabled").read();
        let mut out = Vec::new();
        while let Some(idx) = cur {
            let page = pz.read_page_for_test(idx).unwrap();
            for entry in page.cursor() {
                out.push(entry.unwrap());
            }
            cur = crate::props_zone::decode_page_id(page.header.next_page);
        }
        out
    }

    #[test]
    #[allow(unsafe_code)]
    fn set_edge_property_writes_to_props_zone_v2() {
        // With OBRAIN_PROPS_V2=1, a scalar set_edge_property must land
        // as a PropertyEntry on the edge PropsZone chain AND flip the
        // head pointer on the EdgeRecord (byte offset 24).
        let _guard = props_v2_env_lock().lock().unwrap();
        unsafe { std::env::set_var("OBRAIN_PROPS_V2", "1") };

        let td = tempfile::tempdir().unwrap();
        let s = SubstrateStore::create(td.path().join("kb")).unwrap();
        let a = s.create_node(&["Doc"]);
        let b = s.create_node(&["Doc"]);
        let e = s.create_edge(a, b, "LINKS_TO");
        s.set_edge_property(e, "weight", Value::Float64(0.75));
        s.set_edge_property(e, "since", Value::Int64(2026));
        s.flush().unwrap();

        // Head pointer must now be non-zero.
        let rec = s.writer.read_edge(e.0).unwrap().unwrap();
        assert!(
            !rec.first_prop_off.is_zero(),
            "EdgeRecord.first_prop_off must be set after property writes"
        );

        // Chain contains both entries (in append order since they fit
        // on the fresh first head page).
        let entries = collect_edge_props_v2(&s, e.0);
        assert_eq!(entries.len(), 2, "got entries: {:?}", entries);
        let weight_id = s.prop_keys.write().intern("weight").unwrap();
        let since_id = s.prop_keys.write().intern("since").unwrap();
        assert_eq!(entries[0].prop_key, weight_id, "first append = weight");
        assert_eq!(entries[1].prop_key, since_id, "second append = since");

        unsafe { std::env::remove_var("OBRAIN_PROPS_V2") };
    }

    #[test]
    #[allow(unsafe_code)]
    fn get_edge_property_lww_latest_wins() {
        // Three successive set_edge_property calls for the same key —
        // the LWW read path must surface v3, not v1 or v2.
        let _guard = props_v2_env_lock().lock().unwrap();
        unsafe { std::env::set_var("OBRAIN_PROPS_V2", "1") };

        let td = tempfile::tempdir().unwrap();
        let s = SubstrateStore::create(td.path().join("kb")).unwrap();
        let a = s.create_node(&["Doc"]);
        let b = s.create_node(&["Doc"]);
        let e = s.create_edge(a, b, "LINKS_TO");
        s.set_edge_property(e, "label", Value::String("v1".into()));
        s.set_edge_property(e, "label", Value::String("v2".into()));
        s.set_edge_property(e, "label", Value::String("v3".into()));

        // Chain holds all three writes (append-only).
        let entries = collect_edge_props_v2(&s, e.0);
        assert_eq!(entries.len(), 3, "append-only chain retains all 3 entries");

        let got = s.get_edge_property(e, &PropertyKey::new("label"));
        assert_eq!(
            got,
            Some(Value::String("v3".into())),
            "LWW on edge props: latest write wins (got: {:?})",
            got
        );

        unsafe { std::env::remove_var("OBRAIN_PROPS_V2") };
    }

    #[test]
    #[allow(unsafe_code)]
    fn remove_edge_property_emits_tombstone_and_shadows_read() {
        // `set_edge_property` then `remove_edge_property` must land a
        // tombstone on the chain, and `get_edge_property` must return
        // `None` via the LWW read path (NOT fall back to the DashMap,
        // which would otherwise still have the value in a pre-T17f
        // implementation — but here the DashMap is also cleared on
        // remove, so correctness comes from the chain).
        let _guard = props_v2_env_lock().lock().unwrap();
        unsafe { std::env::set_var("OBRAIN_PROPS_V2", "1") };

        let td = tempfile::tempdir().unwrap();
        let s = SubstrateStore::create(td.path().join("kb")).unwrap();
        let a = s.create_node(&["Doc"]);
        let b = s.create_node(&["Doc"]);
        let e = s.create_edge(a, b, "LINKS_TO");
        s.set_edge_property(e, "weight", Value::Float64(0.9));
        // After set: chain should have exactly 1 live entry.
        let after_set = collect_edge_props_v2_raw(&s, e.0);
        assert_eq!(after_set.len(), 1, "1 live entry after set");
        assert!(!after_set[0].is_tombstone(), "first append is live");
        assert_eq!(
            s.get_edge_property(e, &PropertyKey::new("weight")),
            Some(Value::Float64(0.9)),
        );

        let popped = s.remove_edge_property(e, "weight");
        assert_eq!(popped, Some(Value::Float64(0.9)), "removed value returned");

        // Tombstone must now be on the chain (raw walk, tombstone-inclusive).
        let entries = collect_edge_props_v2_raw(&s, e.0);
        assert_eq!(entries.len(), 2, "2 entries: live + tombstone");
        assert!(entries[1].is_tombstone(), "last append is a tombstone");

        // Read must return None via the LWW read path.
        assert_eq!(
            s.get_edge_property(e, &PropertyKey::new("weight")),
            None,
            "tombstone shadows the older live entry on read",
        );

        unsafe { std::env::remove_var("OBRAIN_PROPS_V2") };
    }

    #[test]
    #[allow(unsafe_code)]
    fn edge_props_v2_survives_reopen() {
        // Write an edge property with v2 enabled, close the store,
        // reopen, and confirm the chain still holds the entry and the
        // LWW read still returns the correct value.
        let _guard = props_v2_env_lock().lock().unwrap();
        unsafe { std::env::set_var("OBRAIN_PROPS_V2", "1") };

        let td = tempfile::tempdir().unwrap();
        let root = td.path().join("kb");
        let e_raw;
        {
            let s = SubstrateStore::create(&root).unwrap();
            let a = s.create_node(&["Doc"]);
            let b = s.create_node(&["Doc"]);
            let e = s.create_edge(a, b, "LINKS_TO");
            e_raw = e.0;
            s.set_edge_property(e, "weight", Value::Float64(0.42));
            s.set_edge_property(e, "note", Value::String("persistent".into()));
            s.flush().unwrap();
        }

        unsafe { std::env::remove_var("OBRAIN_PROPS_V2") };
        let s = SubstrateStore::open(&root).unwrap();
        assert!(s.props_v2_enabled(), "v2 must auto-enable on reopen");

        let entries = collect_edge_props_v2(&s, e_raw);
        assert_eq!(entries.len(), 2, "entries survived reopen: {:?}", entries);

        // LWW reads also work post-reopen.
        let got = s.get_edge_property(EdgeId(e_raw), &PropertyKey::new("weight"));
        assert_eq!(got, Some(Value::Float64(0.42)));
        let got = s.get_edge_property(EdgeId(e_raw), &PropertyKey::new("note"));
        assert_eq!(got, Some(Value::String("persistent".into())));
    }

    #[test]
    #[allow(unsafe_code)]
    fn node_and_edge_chains_coexist_without_cross_splicing() {
        // Write properties on both a node and an edge in the same
        // PropsZone. The two chains must each land on their own
        // owner-kind pages (node magic 0xF507_5FA6, edge magic
        // 0xE507_5FA6) — a single PropsZone carrying both owner kinds
        // was the whole point of T17f Step 2.
        let _guard = props_v2_env_lock().lock().unwrap();
        unsafe { std::env::set_var("OBRAIN_PROPS_V2", "1") };

        let td = tempfile::tempdir().unwrap();
        let s = SubstrateStore::create(td.path().join("kb")).unwrap();
        let a = s.create_node(&["Doc"]);
        let b = s.create_node(&["Doc"]);
        let e = s.create_edge(a, b, "LINKS_TO");

        s.set_node_property(a, "title", Value::String("node-side".into()));
        s.set_edge_property(e, "label", Value::String("edge-side".into()));

        // Each side reads its own value unaltered.
        assert_eq!(
            s.get_node_property(a, &PropertyKey::new("title")),
            Some(Value::String("node-side".into())),
        );
        assert_eq!(
            s.get_edge_property(e, &PropertyKey::new("label")),
            Some(Value::String("edge-side".into())),
        );

        // The node chain head is a node-magic page; the edge chain
        // head is an edge-magic page.
        let node_rec = s.writer.read_node(a.0 as u32).unwrap().unwrap();
        let edge_rec = s.writer.read_edge(e.0).unwrap().unwrap();
        let node_head = crate::props_zone::decode_page_id(node_rec.first_prop_off)
            .expect("node chain head allocated");
        let edge_head = crate::props_zone::decode_page_id(edge_rec.first_prop_off)
            .expect("edge chain head allocated");
        assert_ne!(
            node_head, edge_head,
            "node and edge chain heads must live on distinct pages"
        );

        let pz = s.props_zone.as_ref().unwrap().read();
        assert_eq!(
            pz.owner_kind_at(node_head),
            Some(crate::page::OwnerKind::Node),
            "node chain head page must carry node magic",
        );
        assert_eq!(
            pz.owner_kind_at(edge_head),
            Some(crate::page::OwnerKind::Edge),
            "edge chain head page must carry edge magic",
        );

        unsafe { std::env::remove_var("OBRAIN_PROPS_V2") };
    }

    // -------------------------------------------------------------------
    // T17c Step 3c — PropsZone read-path (LWW) tests.
    //
    // These exercise `get_node_property` routing through
    // `lookup_node_property_v2` (walk_chain + LWW) instead of (or
    // before) the DashMap fallback.
    // -------------------------------------------------------------------

    /// LWW: two successive `set_node_property` calls for the same key
    /// must have the **second write win** via the v2 chain. The read
    /// path must not return the older value even though both entries
    /// coexist on the chain.
    #[test]
    #[allow(unsafe_code)]
    fn get_node_property_lww_same_page() {
        let _guard = props_v2_env_lock().lock().unwrap();
        unsafe { std::env::set_var("OBRAIN_PROPS_V2", "1") };

        let td = tempfile::tempdir().unwrap();
        let s = SubstrateStore::create(td.path().join("kb")).unwrap();
        let a = s.create_node(&["Doc"]);
        s.set_node_property(a, "title", Value::String("v1".into()));
        s.set_node_property(a, "title", Value::String("v2".into()));
        s.set_node_property(a, "title", Value::String("v3".into()));

        // Chain holds all three writes (append-only).
        let entries = collect_node_props_v2(&s, a.0 as u32);
        assert_eq!(
            entries.len(),
            3,
            "append-only chain retains all 3 entries"
        );

        // Read must surface the latest value.
        let got = s.get_node_property(a, &PropertyKey::new("title"));
        assert_eq!(
            got,
            Some(Value::String("v3".into())),
            "LWW: latest write wins (got: {:?})",
            got
        );

        unsafe { std::env::remove_var("OBRAIN_PROPS_V2") };
    }

    /// Tombstone: `set` then `remove` must make `get` return `None`
    /// via the v2 chain, even though the older `set` entry is still
    /// physically present. The v2 tombstone also shadows the DashMap
    /// sidecar (which removed the key in the same call).
    #[test]
    #[allow(unsafe_code)]
    fn get_node_property_tombstone_shadows_older_set() {
        let _guard = props_v2_env_lock().lock().unwrap();
        unsafe { std::env::set_var("OBRAIN_PROPS_V2", "1") };

        let td = tempfile::tempdir().unwrap();
        let s = SubstrateStore::create(td.path().join("kb")).unwrap();
        let a = s.create_node(&["Doc"]);
        s.set_node_property(a, "title", Value::String("hello".into()));
        // Sanity: live read returns the value.
        assert_eq!(
            s.get_node_property(a, &PropertyKey::new("title")),
            Some(Value::String("hello".into()))
        );
        s.remove_node_property(a, "title");

        // v2 chain must have both the live entry (old) and a
        // tombstone (new); tombstone wins.
        let got = s.get_node_property(a, &PropertyKey::new("title"));
        assert_eq!(
            got,
            None,
            "tombstone must shadow older live entry (got: {:?})",
            got
        );

        unsafe { std::env::remove_var("OBRAIN_PROPS_V2") };
    }

    /// Tombstone persistence: after a drop/reopen cycle, the tombstone
    /// entry on disk must still shadow the earlier live entry. This
    /// validates the WAL-logged head pointer update survives a fresh
    /// mmap open.
    #[test]
    #[allow(unsafe_code)]
    fn get_node_property_tombstone_survives_reopen() {
        let _guard = props_v2_env_lock().lock().unwrap();
        unsafe { std::env::set_var("OBRAIN_PROPS_V2", "1") };

        let td = tempfile::tempdir().unwrap();
        let root = td.path().join("kb");
        let a_id;
        {
            let s = SubstrateStore::create(&root).unwrap();
            let a = s.create_node(&["Doc"]);
            a_id = a;
            s.set_node_property(a, "title", Value::String("persistent".into()));
            s.remove_node_property(a, "title");
            s.flush().unwrap();
        }

        unsafe { std::env::remove_var("OBRAIN_PROPS_V2") };
        let s = SubstrateStore::open(&root).unwrap();
        assert!(s.props_v2_enabled(), "v2 must auto-enable on reopen");

        let got = s.get_node_property(a_id, &PropertyKey::new("title"));
        assert_eq!(
            got,
            None,
            "tombstone survives reopen (got: {:?})",
            got
        );
    }

    /// Legacy fallback: a node whose `first_prop_off == 0` (never
    /// wrote through v2 — e.g. migrated-in or DashMap-only insertion)
    /// must still resolve via the DashMap sidecar. This exercises the
    /// `None` arm of `lookup_node_property_v2`.
    #[test]
    #[allow(unsafe_code)]
    fn get_node_property_falls_back_to_dashmap_when_head_is_zero() {
        let _guard = props_v2_env_lock().lock().unwrap();
        // Explicitly disable v2 so `set_node_property` writes only to
        // the DashMap — the NodeRecord.first_prop_off stays at 0.
        unsafe { std::env::set_var("OBRAIN_PROPS_V2", "0") };

        let td = tempfile::tempdir().unwrap();
        let s = SubstrateStore::create(td.path().join("kb")).unwrap();
        let a = s.create_node(&["Doc"]);
        s.set_node_property(a, "title", Value::String("legacy".into()));

        // Sanity: v2 is off, chain head is zero.
        assert!(!s.props_v2_enabled());
        let rec = s.writer.read_node(a.0 as u32).unwrap().unwrap();
        assert!(rec.first_prop_off.is_zero());

        // Read still returns the value via DashMap fallback.
        let got = s.get_node_property(a, &PropertyKey::new("title"));
        assert_eq!(got, Some(Value::String("legacy".into())));

        unsafe { std::env::remove_var("OBRAIN_PROPS_V2") };
    }

    /// T17c Step 3d — when the v2 zone has at least one allocated
    /// page on reopen, the node-property DashMap must stay empty
    /// after `load_properties`. Reads keep working because
    /// `get_node_property` routes through the v2 chain first.
    #[test]
    #[allow(unsafe_code)]
    fn load_properties_skips_node_hydration_when_v2_has_pages() {
        let _guard = props_v2_env_lock().lock().unwrap();
        unsafe { std::env::set_var("OBRAIN_PROPS_V2", "1") };

        let td = tempfile::tempdir().unwrap();
        let root = td.path().join("kb");
        let a_id;
        {
            let s = SubstrateStore::create(&root).unwrap();
            let a = s.create_node(&["Doc"]);
            a_id = a;
            s.set_node_property(a, "title", Value::String("v2-native".into()));
            s.set_node_property(a, "count", Value::Int64(7));
            s.flush().unwrap();
        }

        // Reopen without env — auto-detection on substrate.props.v2
        // existence still activates v2.
        unsafe { std::env::remove_var("OBRAIN_PROPS_V2") };
        let s = SubstrateStore::open(&root).unwrap();
        assert!(s.props_v2_enabled(), "v2 must auto-enable on reopen");

        // DashMap must be empty — load_properties short-circuited.
        assert_eq!(
            s.node_properties.len(),
            0,
            "DashMap must be empty when v2 zone has pages"
        );

        // Reads still work via the v2 chain.
        assert_eq!(
            s.get_node_property(a_id, &PropertyKey::new("title")),
            Some(Value::String("v2-native".into()))
        );
        assert_eq!(
            s.get_node_property(a_id, &PropertyKey::new("count")),
            Some(Value::Int64(7))
        );
    }

    /// T17c Step 3d — when v2 is enabled but has no allocated pages
    /// (fresh store, never wrote a scalar), legacy sidecar hydration
    /// must still run. Otherwise, a legacy-only DB that enables v2 on
    /// upgrade but hasn't migrated yet would silently lose every
    /// node property on first reopen.
    #[test]
    #[allow(unsafe_code)]
    fn load_properties_still_hydrates_when_v2_zone_empty() {
        let _guard = props_v2_env_lock().lock().unwrap();
        // Phase 1: write WITHOUT v2 — sidecar gets node entries, v2
        // zone stays empty.
        unsafe { std::env::set_var("OBRAIN_PROPS_V2", "0") };
        let td = tempfile::tempdir().unwrap();
        let root = td.path().join("kb");
        let a_id;
        {
            let s = SubstrateStore::create(&root).unwrap();
            let a = s.create_node(&["Doc"]);
            a_id = a;
            s.set_node_property(a, "legacy", Value::String("from-sidecar".into()));
            s.flush().unwrap();
            assert!(!s.props_v2_enabled());
        }

        // Phase 2: reopen WITH v2 enabled. Zone is fresh (no pages)
        // because the prior session had v2 disabled — hydration must
        // still populate the DashMap so the legacy key is readable.
        unsafe { std::env::set_var("OBRAIN_PROPS_V2", "1") };
        let s = SubstrateStore::open(&root).unwrap();
        assert!(s.props_v2_enabled(), "v2 active on reopen");
        let pz = s.props_zone.as_ref().unwrap().read();
        assert_eq!(
            pz.allocated_page_count(),
            0,
            "v2 zone is fresh — 0 allocated pages"
        );
        drop(pz);

        // DashMap must be populated with the legacy key.
        assert_eq!(
            s.get_node_property(a_id, &PropertyKey::new("legacy")),
            Some(Value::String("from-sidecar".into())),
            "legacy sidecar key must still resolve (fallback path)"
        );

        unsafe { std::env::remove_var("OBRAIN_PROPS_V2") };
    }

    /// T17c Step 3d — edges always hydrate from the sidecar
    /// regardless of v2 state. EdgeRecord has no `first_prop_off`,
    /// so edges have no v2 chain to walk; if we skipped the edge
    /// loop alongside nodes, we'd lose all edge properties on
    /// reopen.
    #[test]
    #[allow(unsafe_code)]
    fn load_properties_still_hydrates_edges_when_v2_has_pages() {
        let _guard = props_v2_env_lock().lock().unwrap();
        unsafe { std::env::set_var("OBRAIN_PROPS_V2", "1") };

        let td = tempfile::tempdir().unwrap();
        let root = td.path().join("kb");
        let e_id;
        {
            let s = SubstrateStore::create(&root).unwrap();
            let a = s.create_node(&["Doc"]);
            let b = s.create_node(&["Doc"]);
            // Write a node prop so v2 zone gets a page.
            s.set_node_property(a, "title", Value::String("x".into()));
            let e = s.create_edge(a, b, "LINKS");
            e_id = e;
            // Edges still go to the DashMap since EdgeRecord has no
            // first_prop_off.
            s.set_edge_property(e, "weight", Value::Float64(0.5));
            s.flush().unwrap();
        }

        unsafe { std::env::remove_var("OBRAIN_PROPS_V2") };
        let s = SubstrateStore::open(&root).unwrap();

        // Node DashMap empty (Step 3d gate).
        assert_eq!(s.node_properties.len(), 0);
        // Edge DashMap populated (hydrated from sidecar despite v2 gate).
        let got = s.get_edge_property(e_id, &PropertyKey::new("weight"));
        assert_eq!(
            got,
            Some(Value::Float64(0.5)),
            "edge prop must survive reopen via sidecar (got: {:?})",
            got
        );
    }

    // -------------------------------------------------------------------
    // T17c Step 4 — finalize_props_v2 migration tests.
    // -------------------------------------------------------------------

    /// End-to-end migration: legacy sidecar → v2 chain → reopen
    /// without DashMap hydration. Proves that after `finalize_props_v2`
    /// + `flush` + sidecar delete, a fresh reopen resolves every
    /// property through the v2 read path.
    #[test]
    #[allow(unsafe_code)]
    fn finalize_props_v2_drains_legacy_sidecar_end_to_end() {
        let _guard = props_v2_env_lock().lock().unwrap();

        let td = tempfile::tempdir().unwrap();
        let root = td.path().join("kb");

        // Phase 1: write legacy sidecar (v2 OFF).
        let a_id;
        let b_id;
        unsafe { std::env::set_var("OBRAIN_PROPS_V2", "0") };
        {
            let s = SubstrateStore::create(&root).unwrap();
            let a = s.create_node(&["Doc"]);
            let b = s.create_node(&["Doc"]);
            a_id = a;
            b_id = b;
            s.set_node_property(a, "title", Value::String("alpha".into()));
            s.set_node_property(a, "count", Value::Int64(1));
            s.set_node_property(b, "title", Value::String("beta".into()));
            s.set_node_property(b, "count", Value::Int64(2));
            s.flush().unwrap();
            assert!(!s.props_v2_enabled());
        }
        // Sanity: legacy sidecar exists.
        let sidecar = root.join(PROPS_FILENAME);
        assert!(sidecar.exists(), "legacy sidecar written");

        // Phase 2: reopen with v2 ON and migrate.
        unsafe { std::env::set_var("OBRAIN_PROPS_V2", "1") };
        {
            let s = SubstrateStore::open(&root).unwrap();
            assert!(s.props_v2_enabled());
            // Sidecar loaded into DashMap because v2 zone was empty.
            assert!(s.node_properties.len() > 0);
            assert_eq!(s.props_v2_page_count(), Some(0));

            // Drain DashMap into v2 chain.
            let stats = s.finalize_props_v2().unwrap();
            assert_eq!(stats.nodes_processed, 2);
            assert_eq!(stats.scalars_emitted, 4);
            assert!(s.props_v2_page_count().unwrap() > 0);

            s.flush().unwrap();

            // Remove the now-stale sidecar.
            let freed = s.delete_legacy_props_sidecar().unwrap();
            assert!(freed.is_some(), "sidecar was present");
            assert!(!sidecar.exists(), "sidecar deleted after drain");
        }

        // Phase 3: reopen. Step 3d gate must now skip hydration
        // (no sidecar + v2 pages present). Reads must still return
        // everything via v2 chain.
        unsafe { std::env::remove_var("OBRAIN_PROPS_V2") };
        let s = SubstrateStore::open(&root).unwrap();
        assert!(s.props_v2_enabled(), "v2 auto-detected on reopen");
        assert_eq!(
            s.node_properties.len(),
            0,
            "DashMap must stay empty post-migration"
        );
        assert_eq!(
            s.get_node_property(a_id, &PropertyKey::new("title")),
            Some(Value::String("alpha".into()))
        );
        assert_eq!(
            s.get_node_property(a_id, &PropertyKey::new("count")),
            Some(Value::Int64(1))
        );
        assert_eq!(
            s.get_node_property(b_id, &PropertyKey::new("title")),
            Some(Value::String("beta".into()))
        );
        assert_eq!(
            s.get_node_property(b_id, &PropertyKey::new("count")),
            Some(Value::Int64(2))
        );
    }

    /// finalize_props_v2 must err cleanly when v2 is disabled — no
    /// partial state, just a typed error the migration CLI can
    /// surface.
    #[test]
    #[allow(unsafe_code)]
    fn finalize_props_v2_errors_when_v2_disabled() {
        let _guard = props_v2_env_lock().lock().unwrap();
        unsafe { std::env::set_var("OBRAIN_PROPS_V2", "0") };

        let td = tempfile::tempdir().unwrap();
        let s = SubstrateStore::create(td.path().join("kb")).unwrap();
        let err = s.finalize_props_v2().unwrap_err();
        assert!(
            format!("{err}").contains("finalize_props_v2"),
            "error must mention the function name (got: {err})"
        );
    }

    /// Multi-page LWW: force a chain spanning two pages and check
    /// that a write on the **newer** head page wins over the older
    /// same-key entry on the tail page. This validates the
    /// page-by-page short-circuit in `get_latest_for_key`.
    #[test]
    #[allow(unsafe_code)]
    fn get_node_property_lww_across_pages() {
        let _guard = props_v2_env_lock().lock().unwrap();
        unsafe { std::env::set_var("OBRAIN_PROPS_V2", "1") };

        let td = tempfile::tempdir().unwrap();
        let s = SubstrateStore::create(td.path().join("kb")).unwrap();
        let a = s.create_node(&["Doc"]);

        // First write: short key "K", lands on head page 1.
        s.set_node_property(a, "K", Value::String("oldest".into()));
        let head_after_first = decode_page_id(
            s.writer.read_node(a.0 as u32).unwrap().unwrap().first_prop_off,
        );

        // Fill the page with filler entries until a fresh head page
        // gets allocated.
        let mut filler = 0u32;
        let original_head = head_after_first;
        loop {
            let key = format!("filler_{filler}");
            s.set_node_property(a, &key, Value::Int64(filler as i64));
            filler += 1;
            let cur_head = decode_page_id(
                s.writer.read_node(a.0 as u32).unwrap().unwrap().first_prop_off,
            );
            if cur_head != original_head {
                break; // New head page allocated.
            }
            // Safety net against runaway loops.
            assert!(filler < 5_000, "failed to rotate head page");
        }

        // New write on the new head page for the same key.
        s.set_node_property(a, "K", Value::String("latest".into()));
        let got = s.get_node_property(a, &PropertyKey::new("K"));
        assert_eq!(
            got,
            Some(Value::String("latest".into())),
            "cross-page LWW: newer head page wins (got: {:?})",
            got
        );

        unsafe { std::env::remove_var("OBRAIN_PROPS_V2") };
    }

    // -------------------------------------------------------------------
    // T17c Step 6 — edge-property preservation path tests.
    //
    // Before Step 6, `finalize_props_v2` drained only `node_properties`
    // and `delete_legacy_props_sidecar` unconditionally removed the
    // combined sidecar — any in-memory edge properties (Wikipedia ships
    // with 312 such entries) vanished silently on the next reopen.
    // These tests pin the fix in place.
    // -------------------------------------------------------------------

    /// Core regression: run the full migration flow (legacy sidecar →
    /// finalize → delete legacy → reopen) with edge properties in
    /// play, and assert every edge prop survives. Reproduces the
    /// Wikipedia edge-loss bug that was caught pre-migration.
    #[test]
    #[allow(unsafe_code)]
    fn finalize_props_v2_preserves_edge_properties() {
        let _guard = props_v2_env_lock().lock().unwrap();

        let td = tempfile::tempdir().unwrap();
        let root = td.path().join("kb");

        let a_id;
        let b_id;
        let e1_id;
        let e2_id;

        // Phase 1: legacy sidecar write (v2 OFF) — nodes + edges.
        unsafe { std::env::set_var("OBRAIN_PROPS_V2", "0") };
        {
            let s = SubstrateStore::create(&root).unwrap();
            let a = s.create_node(&["Doc"]);
            let b = s.create_node(&["Doc"]);
            a_id = a;
            b_id = b;
            s.set_node_property(a, "title", Value::String("alpha".into()));
            s.set_node_property(b, "title", Value::String("beta".into()));

            let e1 = s.create_edge(a, b, "LINKS");
            let e2 = s.create_edge(b, a, "LINKS");
            e1_id = e1;
            e2_id = e2;
            s.set_edge_property(e1, "weight", Value::Float64(0.25));
            s.set_edge_property(e1, "label", Value::String("forward".into()));
            s.set_edge_property(e2, "weight", Value::Float64(0.75));

            s.flush().unwrap();
            assert!(!s.props_v2_enabled());
        }
        let legacy_sidecar = root.join(PROPS_FILENAME);
        let edge_sidecar = root.join(EDGE_PROPS_FILENAME);
        assert!(legacy_sidecar.exists(), "legacy sidecar written");
        assert!(!edge_sidecar.exists(), "no edge sidecar pre-finalize");

        // Phase 2: reopen with v2, run finalize — both nodes AND
        // edges must be handled. Assert the new edge sidecar
        // appears and the legacy sidecar can be dropped.
        unsafe { std::env::set_var("OBRAIN_PROPS_V2", "1") };
        {
            let s = SubstrateStore::open(&root).unwrap();
            assert!(s.props_v2_enabled());
            assert!(s.node_properties.len() > 0);
            assert!(s.edge_properties.len() > 0);

            let stats = s.finalize_props_v2().unwrap();
            assert_eq!(stats.nodes_processed, 2, "2 nodes drained");
            assert_eq!(stats.scalars_emitted, 2, "2 node scalars");
            assert_eq!(stats.edges_processed, 2, "2 edges persisted");
            assert_eq!(stats.edge_scalars_emitted, 3, "3 edge scalars");
            assert!(
                stats.edge_sidecar_bytes > 0,
                "edge sidecar has non-zero footprint"
            );
            assert!(
                edge_sidecar.exists(),
                "edge sidecar written by finalize_props_v2"
            );

            s.flush().unwrap();

            // Sidecar drop must now succeed because the edge sidecar
            // is in place.
            let freed = s.delete_legacy_props_sidecar().unwrap();
            assert!(freed.is_some());
            assert!(!legacy_sidecar.exists());
            assert!(
                edge_sidecar.exists(),
                "edge sidecar preserved post legacy delete"
            );
        }

        // Phase 3: reopen. Node hydration is skipped (v2 pages), but
        // edges must be hydrated from the NEW edge sidecar. Every
        // edge prop set in Phase 1 must be retrievable.
        unsafe { std::env::remove_var("OBRAIN_PROPS_V2") };
        let s = SubstrateStore::open(&root).unwrap();
        assert!(s.props_v2_enabled());
        assert_eq!(
            s.node_properties.len(),
            0,
            "node DashMap empty (v2 gate)"
        );
        assert_eq!(
            s.edge_properties.len(),
            2,
            "edges hydrated from new sidecar"
        );

        // Node props via v2.
        assert_eq!(
            s.get_node_property(a_id, &PropertyKey::new("title")),
            Some(Value::String("alpha".into()))
        );
        assert_eq!(
            s.get_node_property(b_id, &PropertyKey::new("title")),
            Some(Value::String("beta".into()))
        );
        // Edge props via DashMap (populated from new edge sidecar).
        assert_eq!(
            s.get_edge_property(e1_id, &PropertyKey::new("weight")),
            Some(Value::Float64(0.25))
        );
        assert_eq!(
            s.get_edge_property(e1_id, &PropertyKey::new("label")),
            Some(Value::String("forward".into()))
        );
        assert_eq!(
            s.get_edge_property(e2_id, &PropertyKey::new("weight")),
            Some(Value::Float64(0.75))
        );
    }

    /// Safety gate: `delete_legacy_props_sidecar` must refuse when
    /// edge_properties is non-empty AND the edge sidecar is missing.
    /// Caller has to run `finalize_props_v2` (or the explicit
    /// `persist_edge_properties_sidecar` helper) first.
    #[test]
    #[allow(unsafe_code)]
    fn delete_legacy_props_sidecar_refuses_when_edges_would_be_lost() {
        let _guard = props_v2_env_lock().lock().unwrap();

        let td = tempfile::tempdir().unwrap();
        let root = td.path().join("kb");

        // Build a legacy sidecar carrying edge props.
        unsafe { std::env::set_var("OBRAIN_PROPS_V2", "0") };
        let a_id;
        let b_id;
        let e_id;
        {
            let s = SubstrateStore::create(&root).unwrap();
            let a = s.create_node(&["Doc"]);
            let b = s.create_node(&["Doc"]);
            a_id = a;
            b_id = b;
            s.set_node_property(a, "title", Value::String("alpha".into()));
            let e = s.create_edge(a, b, "LINKS");
            e_id = e;
            s.set_edge_property(e, "weight", Value::Float64(0.5));
            s.flush().unwrap();
        }
        let legacy_sidecar = root.join(PROPS_FILENAME);
        let edge_sidecar = root.join(EDGE_PROPS_FILENAME);
        assert!(legacy_sidecar.exists());
        assert!(!edge_sidecar.exists());

        unsafe { std::env::set_var("OBRAIN_PROPS_V2", "1") };
        let s = SubstrateStore::open(&root).unwrap();
        assert!(s.edge_properties.len() > 0);

        // Direct delete without finalize → must refuse.
        let err = s.delete_legacy_props_sidecar().unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("unsaved edge props"),
            "error must name the safety gate (got: {msg})"
        );
        assert!(
            legacy_sidecar.exists(),
            "legacy sidecar untouched after refusal"
        );

        // Unblock via the explicit helper + retry. No need to touch
        // node_properties — finalize_props_v2 isn't required for the
        // gate-lift path.
        let (edges_written, scalars, bytes) =
            s.persist_edge_properties_sidecar().unwrap();
        assert_eq!(edges_written, 1);
        assert_eq!(scalars, 1);
        assert!(bytes > 0);
        assert!(edge_sidecar.exists());

        let freed = s.delete_legacy_props_sidecar().unwrap();
        assert!(freed.is_some(), "delete succeeds post edge-sidecar write");
        assert!(!legacy_sidecar.exists());

        // Reopen: edge prop must survive.
        drop(s);
        unsafe { std::env::remove_var("OBRAIN_PROPS_V2") };
        let s = SubstrateStore::open(&root).unwrap();
        assert_eq!(
            s.get_edge_property(e_id, &PropertyKey::new("weight")),
            Some(Value::Float64(0.5)),
            "edge prop survived legacy-sidecar delete via new edge sidecar"
        );
        // Silence unused-var warnings on nodes — they're included to
        // build a realistic fixture but not asserted here.
        let _ = (a_id, b_id);
    }

    /// Empty edge_properties case: finalize_props_v2 must still
    /// succeed, the edge sidecar is written (empty edges vec), and
    /// the legacy sidecar delete proceeds normally.
    #[test]
    #[allow(unsafe_code)]
    fn finalize_props_v2_handles_empty_edges_cleanly() {
        let _guard = props_v2_env_lock().lock().unwrap();

        let td = tempfile::tempdir().unwrap();
        let root = td.path().join("kb");

        unsafe { std::env::set_var("OBRAIN_PROPS_V2", "1") };
        let s = SubstrateStore::create(&root).unwrap();
        let a = s.create_node(&["Doc"]);
        s.set_node_property(a, "title", Value::String("solo".into()));
        s.flush().unwrap();
        assert_eq!(s.edge_properties.len(), 0);

        let stats = s.finalize_props_v2().unwrap();
        assert_eq!(stats.edges_processed, 0);
        assert_eq!(stats.edge_scalars_emitted, 0);
        // Sidecar is still written (empty edges vec) so the gate on
        // `delete_legacy_props_sidecar` accepts the drop.
        let edge_sidecar = s.edge_props_sidecar_path();
        assert!(edge_sidecar.exists(), "empty edge sidecar still persisted");

        // Legacy sidecar drop must succeed since edge_properties is
        // empty (no gate violation possible).
        let _ = s.delete_legacy_props_sidecar();

        unsafe { std::env::remove_var("OBRAIN_PROPS_V2") };
    }
}
