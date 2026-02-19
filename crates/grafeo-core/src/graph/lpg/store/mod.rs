//! The in-memory LPG graph store.
//!
//! This is where your nodes and edges actually live. Most users interact
//! through [`GrafeoDB`](grafeo_engine::GrafeoDB), but algorithm implementers
//! sometimes need the raw [`LpgStore`] for direct adjacency traversal.
//!
//! Key features:
//! - MVCC versioning - concurrent readers don't block each other
//! - Columnar properties with zone maps for fast filtering
//! - Forward and backward adjacency indexes

mod edge_ops;
mod index;
mod node_ops;
mod property_ops;
mod schema;
mod search;
mod statistics;
mod traversal;
mod versioning;

#[cfg(test)]
mod tests;

use super::PropertyStorage;
#[cfg(not(feature = "tiered-storage"))]
use super::{EdgeRecord, NodeRecord};
use crate::index::adjacency::ChunkedAdjacency;
use crate::statistics::Statistics;
use arcstr::ArcStr;
use dashmap::DashMap;
#[cfg(not(feature = "tiered-storage"))]
use grafeo_common::mvcc::VersionChain;
use grafeo_common::types::{EdgeId, EpochId, HashableValue, NodeId, PropertyKey, Value};
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};
use parking_lot::RwLock;
use std::cmp::Ordering as CmpOrdering;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

#[cfg(feature = "vector-index")]
use crate::index::vector::HnswIndex;

#[cfg(feature = "tiered-storage")]
use crate::storage::EpochStore;
#[cfg(feature = "tiered-storage")]
use grafeo_common::memory::arena::ArenaAllocator;
#[cfg(feature = "tiered-storage")]
use grafeo_common::mvcc::VersionIndex;

/// Compares two values for ordering (used for range checks).
pub(super) fn compare_values_for_range(a: &Value, b: &Value) -> Option<CmpOrdering> {
    match (a, b) {
        (Value::Int64(a), Value::Int64(b)) => Some(a.cmp(b)),
        (Value::Float64(a), Value::Float64(b)) => a.partial_cmp(b),
        (Value::String(a), Value::String(b)) => Some(a.cmp(b)),
        (Value::Bool(a), Value::Bool(b)) => Some(a.cmp(b)),
        _ => None,
    }
}

/// Checks if a value is within a range.
pub(super) fn value_in_range(
    value: &Value,
    min: Option<&Value>,
    max: Option<&Value>,
    min_inclusive: bool,
    max_inclusive: bool,
) -> bool {
    // Check lower bound
    if let Some(min_val) = min {
        match compare_values_for_range(value, min_val) {
            Some(CmpOrdering::Less) => return false,
            Some(CmpOrdering::Equal) if !min_inclusive => return false,
            None => return false, // Can't compare
            _ => {}
        }
    }

    // Check upper bound
    if let Some(max_val) = max {
        match compare_values_for_range(value, max_val) {
            Some(CmpOrdering::Greater) => return false,
            Some(CmpOrdering::Equal) if !max_inclusive => return false,
            None => return false,
            _ => {}
        }
    }

    true
}

/// Configuration for the LPG store.
///
/// The defaults work well for most cases. Tune `backward_edges` if you only
/// traverse in one direction (saves memory), or adjust capacities if you know
/// your graph size upfront (avoids reallocations).
#[derive(Debug, Clone)]
pub struct LpgStoreConfig {
    /// Maintain backward adjacency for incoming edge queries. Turn off if
    /// you only traverse outgoing edges - saves ~50% adjacency memory.
    pub backward_edges: bool,
    /// Initial capacity for nodes (avoids early reallocations).
    pub initial_node_capacity: usize,
    /// Initial capacity for edges (avoids early reallocations).
    pub initial_edge_capacity: usize,
}

impl Default for LpgStoreConfig {
    fn default() -> Self {
        Self {
            backward_edges: true,
            initial_node_capacity: 1024,
            initial_edge_capacity: 4096,
        }
    }
}

/// The core in-memory graph storage.
///
/// Everything lives here: nodes, edges, properties, adjacency indexes, and
/// version chains for MVCC. Concurrent reads never block each other.
///
/// Most users should go through `GrafeoDB` (from the `grafeo_engine` crate) which
/// adds transaction management and query execution. Use `LpgStore` directly
/// when you need raw performance for algorithm implementations.
///
/// # Example
///
/// ```
/// use grafeo_core::graph::lpg::LpgStore;
/// use grafeo_core::graph::Direction;
///
/// let store = LpgStore::new();
///
/// // Create a small social network
/// let alice = store.create_node(&["Person"]);
/// let bob = store.create_node(&["Person"]);
/// store.create_edge(alice, bob, "KNOWS");
///
/// // Traverse outgoing edges
/// for neighbor in store.neighbors(alice, Direction::Outgoing) {
///     println!("Alice knows node {:?}", neighbor);
/// }
/// ```
///
/// # Lock Ordering
///
/// `LpgStore` contains multiple `RwLock` fields that must be acquired in a
/// consistent order to prevent deadlocks. Always acquire locks in this order:
///
/// ## Level 1 - Entity Storage (mutually exclusive via feature flag)
/// 1. `nodes` / `node_versions`
/// 2. `edges` / `edge_versions`
///
/// ## Level 2 - Catalogs (acquire as pairs when writing)
/// 3. `label_to_id` + `id_to_label`
/// 4. `edge_type_to_id` + `id_to_edge_type`
///
/// ## Level 3 - Indexes
/// 5. `label_index`
/// 6. `node_labels`
/// 7. `property_indexes`
///
/// ## Level 4 - Statistics
/// 8. `statistics`
///
/// ## Level 5 - Nested Locks (internal to other structs)
/// 9. `PropertyStorage::columns` (via `node_properties`/`edge_properties`)
/// 10. `ChunkedAdjacency::lists` (via `forward_adj`/`backward_adj`)
///
/// ## Rules
/// - Catalog pairs must be acquired together when writing.
/// - Never hold entity locks while acquiring catalog locks in a different scope.
/// - Statistics lock is always last.
/// - Read locks are generally safe, but avoid read-to-write upgrades.
pub struct LpgStore {
    /// Configuration.
    #[allow(dead_code)]
    pub(super) config: LpgStoreConfig,

    /// Node records indexed by NodeId, with version chains for MVCC.
    /// Used when `tiered-storage` feature is disabled.
    /// Lock order: 1
    #[cfg(not(feature = "tiered-storage"))]
    pub(super) nodes: RwLock<FxHashMap<NodeId, VersionChain<NodeRecord>>>,

    /// Edge records indexed by EdgeId, with version chains for MVCC.
    /// Used when `tiered-storage` feature is disabled.
    /// Lock order: 2
    #[cfg(not(feature = "tiered-storage"))]
    pub(super) edges: RwLock<FxHashMap<EdgeId, VersionChain<EdgeRecord>>>,

    // === Tiered Storage Fields (feature-gated) ===
    //
    // Lock ordering for arena access:
    //   version_lock (read/write) → arena read lock (via arena_allocator.arena())
    //
    // Rules:
    // - Acquire arena read lock *after* version locks, never before.
    // - Multiple threads may call arena.read_at() concurrently (shared refs only).
    // - Never acquire arena write lock (alloc_new_chunk) while holding version locks.
    // - freeze_epoch order: node_versions.read() → arena.read_at(),
    //   then edge_versions.read() → arena.read_at().
    /// Arena allocator for hot data storage.
    /// Data is stored in per-epoch arenas for fast allocation and bulk deallocation.
    #[cfg(feature = "tiered-storage")]
    pub(super) arena_allocator: Arc<ArenaAllocator>,

    /// Node version indexes - store metadata and arena offsets.
    /// The actual NodeRecord data is stored in the arena.
    /// Lock order: 1
    #[cfg(feature = "tiered-storage")]
    pub(super) node_versions: RwLock<FxHashMap<NodeId, VersionIndex>>,

    /// Edge version indexes - store metadata and arena offsets.
    /// The actual EdgeRecord data is stored in the arena.
    /// Lock order: 2
    #[cfg(feature = "tiered-storage")]
    pub(super) edge_versions: RwLock<FxHashMap<EdgeId, VersionIndex>>,

    /// Cold storage for frozen epochs.
    /// Contains compressed epoch blocks for historical data.
    #[cfg(feature = "tiered-storage")]
    pub(super) epoch_store: Arc<EpochStore>,

    /// Property storage for nodes.
    pub(super) node_properties: PropertyStorage<NodeId>,

    /// Property storage for edges.
    pub(super) edge_properties: PropertyStorage<EdgeId>,

    /// Label name to ID mapping.
    /// Lock order: 3 (acquire with id_to_label)
    pub(super) label_to_id: RwLock<FxHashMap<ArcStr, u32>>,

    /// Label ID to name mapping.
    /// Lock order: 3 (acquire with label_to_id)
    pub(super) id_to_label: RwLock<Vec<ArcStr>>,

    /// Edge type name to ID mapping.
    /// Lock order: 4 (acquire with id_to_edge_type)
    pub(super) edge_type_to_id: RwLock<FxHashMap<ArcStr, u32>>,

    /// Edge type ID to name mapping.
    /// Lock order: 4 (acquire with edge_type_to_id)
    pub(super) id_to_edge_type: RwLock<Vec<ArcStr>>,

    /// Forward adjacency lists (outgoing edges).
    pub(super) forward_adj: ChunkedAdjacency,

    /// Backward adjacency lists (incoming edges).
    /// Only populated if config.backward_edges is true.
    pub(super) backward_adj: Option<ChunkedAdjacency>,

    /// Label index: label_id -> set of node IDs.
    /// Lock order: 5
    pub(super) label_index: RwLock<Vec<FxHashMap<NodeId, ()>>>,

    /// Node labels: node_id -> set of label IDs.
    /// Reverse mapping to efficiently get labels for a node.
    /// Lock order: 6
    pub(super) node_labels: RwLock<FxHashMap<NodeId, FxHashSet<u32>>>,

    /// Property indexes: property_key -> (value -> set of node IDs).
    ///
    /// When a property is indexed, lookups by value are O(1) instead of O(n).
    /// Use [`create_property_index`] to enable indexing for a property.
    /// Lock order: 7
    pub(super) property_indexes:
        RwLock<FxHashMap<PropertyKey, DashMap<HashableValue, FxHashSet<NodeId>>>>,

    /// Vector indexes: "label:property" -> HNSW index.
    ///
    /// Created via [`GrafeoDB::create_vector_index`](grafeo_engine::GrafeoDB::create_vector_index).
    /// Lock order: 7 (same level as property_indexes, disjoint keys)
    #[cfg(feature = "vector-index")]
    pub(super) vector_indexes: RwLock<FxHashMap<String, Arc<HnswIndex>>>,

    /// Text indexes: "label:property" -> inverted index with BM25 scoring.
    ///
    /// Created via [`GrafeoDB::create_text_index`](grafeo_engine::GrafeoDB::create_text_index).
    /// Lock order: 7 (same level as property_indexes, disjoint keys)
    #[cfg(feature = "text-index")]
    pub(super) text_indexes:
        RwLock<FxHashMap<String, Arc<RwLock<crate::index::text::InvertedIndex>>>>,

    /// Next node ID.
    pub(super) next_node_id: AtomicU64,

    /// Next edge ID.
    pub(super) next_edge_id: AtomicU64,

    /// Current epoch.
    pub(super) current_epoch: AtomicU64,

    /// Statistics for cost-based optimization.
    /// Lock order: 8 (always last)
    pub(super) statistics: RwLock<Arc<Statistics>>,

    /// Whether statistics need recomputation after mutations.
    pub(super) needs_stats_recompute: AtomicBool,
}

impl LpgStore {
    /// Creates a new LPG store with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(LpgStoreConfig::default())
    }

    /// Creates a new LPG store with custom configuration.
    #[must_use]
    pub fn with_config(config: LpgStoreConfig) -> Self {
        let backward_adj = if config.backward_edges {
            Some(ChunkedAdjacency::new())
        } else {
            None
        };

        Self {
            #[cfg(not(feature = "tiered-storage"))]
            nodes: RwLock::new(FxHashMap::default()),
            #[cfg(not(feature = "tiered-storage"))]
            edges: RwLock::new(FxHashMap::default()),
            #[cfg(feature = "tiered-storage")]
            arena_allocator: Arc::new(ArenaAllocator::new()),
            #[cfg(feature = "tiered-storage")]
            node_versions: RwLock::new(FxHashMap::default()),
            #[cfg(feature = "tiered-storage")]
            edge_versions: RwLock::new(FxHashMap::default()),
            #[cfg(feature = "tiered-storage")]
            epoch_store: Arc::new(EpochStore::new()),
            node_properties: PropertyStorage::new(),
            edge_properties: PropertyStorage::new(),
            label_to_id: RwLock::new(FxHashMap::default()),
            id_to_label: RwLock::new(Vec::new()),
            edge_type_to_id: RwLock::new(FxHashMap::default()),
            id_to_edge_type: RwLock::new(Vec::new()),
            forward_adj: ChunkedAdjacency::new(),
            backward_adj,
            label_index: RwLock::new(Vec::new()),
            node_labels: RwLock::new(FxHashMap::default()),
            property_indexes: RwLock::new(FxHashMap::default()),
            #[cfg(feature = "vector-index")]
            vector_indexes: RwLock::new(FxHashMap::default()),
            #[cfg(feature = "text-index")]
            text_indexes: RwLock::new(FxHashMap::default()),
            next_node_id: AtomicU64::new(0),
            next_edge_id: AtomicU64::new(0),
            current_epoch: AtomicU64::new(0),
            statistics: RwLock::new(Arc::new(Statistics::new())),
            needs_stats_recompute: AtomicBool::new(true),
            config,
        }
    }

    /// Returns the current epoch.
    #[must_use]
    pub fn current_epoch(&self) -> EpochId {
        EpochId::new(self.current_epoch.load(Ordering::Acquire))
    }

    /// Creates a new epoch.
    pub fn new_epoch(&self) -> EpochId {
        let id = self.current_epoch.fetch_add(1, Ordering::AcqRel) + 1;
        EpochId::new(id)
    }

    /// Syncs the store epoch to match an external epoch counter.
    ///
    /// Used by the transaction manager to keep the store's epoch in step
    /// after a transaction commit advances the global epoch.
    pub fn sync_epoch(&self, epoch: EpochId) {
        self.current_epoch
            .fetch_max(epoch.as_u64(), Ordering::AcqRel);
    }

    // === Internal Helpers ===

    pub(super) fn get_or_create_label_id(&self, label: &str) -> u32 {
        {
            let label_to_id = self.label_to_id.read();
            if let Some(&id) = label_to_id.get(label) {
                return id;
            }
        }

        let mut label_to_id = self.label_to_id.write();
        let mut id_to_label = self.id_to_label.write();

        // Double-check after acquiring write lock
        if let Some(&id) = label_to_id.get(label) {
            return id;
        }

        let id = id_to_label.len() as u32;

        let label: ArcStr = label.into();
        label_to_id.insert(label.clone(), id);
        id_to_label.push(label);

        id
    }

    pub(super) fn get_or_create_edge_type_id(&self, edge_type: &str) -> u32 {
        {
            let type_to_id = self.edge_type_to_id.read();
            if let Some(&id) = type_to_id.get(edge_type) {
                return id;
            }
        }

        let mut type_to_id = self.edge_type_to_id.write();
        let mut id_to_type = self.id_to_edge_type.write();

        // Double-check
        if let Some(&id) = type_to_id.get(edge_type) {
            return id;
        }

        let id = id_to_type.len() as u32;
        let edge_type: ArcStr = edge_type.into();
        type_to_id.insert(edge_type.clone(), id);
        id_to_type.push(edge_type);

        id
    }
}

impl Default for LpgStore {
    fn default() -> Self {
        Self::new()
    }
}
