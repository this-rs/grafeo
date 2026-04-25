//! Storage traits for the graph engine.
//!
//! These traits capture the minimal surface that query operators need from
//! the graph store. The split is intentional:
//!
//! - [`GraphStore`]: Read-only operations (scans, lookups, traversal, statistics)
//! - [`GraphStoreMut`]: Write operations (create, delete, mutate)
//!
//! Admin operations (index management, MVCC internals, schema introspection,
//! statistics recomputation, WAL recovery) stay on the concrete store
//! (post-T17: `obrain_substrate::SubstrateStore`) and are not part of these
//! traits.
//!
//! ## Design rationale
//!
//! The traits work with typed graph objects (`Node`, `Edge`, `Value`) rather
//! than raw bytes. Since T17 cutover, `SubstrateStore` (in the
//! `obrain-substrate` crate) is the canonical backend (mmap + WAL-native).
//! The traits are kept generic so legacy shims (e.g. the projection adapter)
//! and future backends can implement the same interface. Intra-doc links
//! cannot resolve `SubstrateStore` from here because `obrain-core` does not
//! depend on `obrain-substrate`.

use crate::graph::Direction;
use crate::graph::lpg::CompareOp;
use crate::graph::lpg::{Edge, Node};
use crate::statistics::Statistics;
use arcstr::ArcStr;
use obrain_common::memory::arena::AllocError;
use obrain_common::types::{EdgeId, EpochId, NodeId, PropertyKey, TransactionId, Value};
use obrain_common::utils::hash::FxHashMap;
use std::sync::Arc;

/// Read-only graph operations used by the query engine.
///
/// This trait captures the minimal surface that scan, expand, filter,
/// project, and shortest-path operators need. Implementations may serve
/// data from memory, disk, or a hybrid of both.
///
/// # Object safety
///
/// This trait is object-safe: you can use `Arc<dyn GraphStore>` for dynamic
/// dispatch. Traversal methods return `Vec` instead of `impl Iterator` to
/// enable this.
pub trait GraphStore: Send + Sync {
    // --- Point lookups ---

    /// Returns a node by ID (latest visible version at current epoch).
    fn get_node(&self, id: NodeId) -> Option<Node>;

    /// Returns an edge by ID (latest visible version at current epoch).
    fn get_edge(&self, id: EdgeId) -> Option<Edge>;

    /// Returns a node visible to a specific transaction.
    fn get_node_versioned(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Option<Node>;

    /// Returns an edge visible to a specific transaction.
    fn get_edge_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Option<Edge>;

    /// Returns a node using pure epoch-based visibility (no transaction context).
    ///
    /// The node is visible if `created_epoch <= epoch` and not deleted at or
    /// before `epoch`. Used for time-travel queries where transaction ownership
    /// must not bypass the epoch check.
    fn get_node_at_epoch(&self, id: NodeId, epoch: EpochId) -> Option<Node>;

    /// Returns an edge using pure epoch-based visibility (no transaction context).
    fn get_edge_at_epoch(&self, id: EdgeId, epoch: EpochId) -> Option<Edge>;

    // --- Property access (fast path, avoids loading full entity) ---

    /// Gets a single property from a node without loading all properties.
    fn get_node_property(&self, id: NodeId, key: &PropertyKey) -> Option<Value>;

    /// Gets a single property from an edge without loading all properties.
    fn get_edge_property(&self, id: EdgeId, key: &PropertyKey) -> Option<Value>;

    /// Gets a property for multiple nodes in a single batch operation.
    fn get_node_property_batch(&self, ids: &[NodeId], key: &PropertyKey) -> Vec<Option<Value>>;

    /// Gets all properties for multiple nodes in a single batch operation.
    fn get_nodes_properties_batch(&self, ids: &[NodeId]) -> Vec<FxHashMap<PropertyKey, Value>>;

    /// Gets selected properties for multiple nodes (projection pushdown).
    fn get_nodes_properties_selective_batch(
        &self,
        ids: &[NodeId],
        keys: &[PropertyKey],
    ) -> Vec<FxHashMap<PropertyKey, Value>>;

    /// Gets selected properties for multiple edges (projection pushdown).
    fn get_edges_properties_selective_batch(
        &self,
        ids: &[EdgeId],
        keys: &[PropertyKey],
    ) -> Vec<FxHashMap<PropertyKey, Value>>;

    // --- Traversal ---

    /// Returns neighbor node IDs in the specified direction.
    ///
    /// Returns `Vec` instead of an iterator for object safety. The underlying
    /// `ChunkedAdjacency` already produces a `Vec` internally.
    fn neighbors(&self, node: NodeId, direction: Direction) -> Vec<NodeId>;

    /// Returns (target_node, edge_id) pairs for edges from a node.
    fn edges_from(&self, node: NodeId, direction: Direction) -> Vec<(NodeId, EdgeId)>;

    /// Returns the out-degree of a node (number of outgoing edges).
    fn out_degree(&self, node: NodeId) -> usize;

    /// Returns the in-degree of a node (number of incoming edges).
    fn in_degree(&self, node: NodeId) -> usize;

    /// Returns the out-degree of a node filtered by edge type. O(1) on
    /// backends with a per-edge-type degree index (SubstrateStore
    /// T17h T8) ; default fallback returns 0 for backends that haven't
    /// wired the index, which pushes the query planner to its legacy
    /// expand+count path rather than silently dropping edges.
    ///
    /// `edge_type = None` means "sum across all types" and is
    /// equivalent to `out_degree(node)` ; it exists so planner rewrites
    /// can uniformly route through this method.
    fn out_degree_by_type(&self, node: NodeId, edge_type: Option<&str>) -> usize {
        let _ = (node, edge_type);
        0
    }

    /// Returns the in-degree of a node filtered by edge type. See
    /// [`Self::out_degree_by_type`] for contract.
    fn in_degree_by_type(&self, node: NodeId, edge_type: Option<&str>) -> usize {
        let _ = (node, edge_type);
        0
    }

    /// Returns `true` when this backend supports O(1) typed-degree
    /// lookups via [`Self::out_degree_by_type`] / [`Self::in_degree_by_type`].
    /// Query planners use this to gate the typed-degree TopK rewrite —
    /// backends that return `false` keep the slow expand+count path
    /// for correctness (the default-0 fallback would otherwise silently
    /// return empty top-K results).
    fn supports_typed_degree(&self) -> bool {
        false
    }

    /// Returns the set of labels observed on the **target** nodes of
    /// every live edge whose type is `edge_type`. Backends with a
    /// persistent histogram (SubstrateStore T17i T2) answer in O(K)
    /// where K ≤ 64 ; the default impl scans every edge once per
    /// call (O(E)) via `all_edge_ids` + `edge_type` + `edge_endpoints`
    /// + `labels(node)`, which is correct but expensive — planners
    /// should gate their use of this method on [`Self::supports_edge_label_histogram`].
    ///
    /// Used by the T17i T3 Cypher planner rewrite to decide whether
    /// a peer-label constraint like `(imported:File)` can be safely
    /// routed through the typed-degree top-K operator : the rewrite
    /// is only accepted when the histogram contains exactly one
    /// entry matching the anchor's label.
    fn edge_target_labels(&self, edge_type: &str) -> std::collections::HashSet<String> {
        let _ = edge_type;
        std::collections::HashSet::new()
    }

    /// Symmetric accessor for **source** node labels per edge type.
    /// Same contract as [`Self::edge_target_labels`].
    fn edge_source_labels(&self, edge_type: &str) -> std::collections::HashSet<String> {
        let _ = edge_type;
        std::collections::HashSet::new()
    }

    /// `true` iff the backend maintains a persistent histogram of
    /// peer labels per edge type — i.e. `edge_target_labels` /
    /// `edge_source_labels` are O(1)/O(K) rather than O(E). Planners
    /// can use the histogram cheaply when this is `true` and should
    /// skip the feature otherwise.
    fn supports_edge_label_histogram(&self) -> bool {
        false
    }

    /// Whether backward adjacency is available for incoming edge queries.
    fn has_backward_adjacency(&self) -> bool;

    // --- Scans ---

    /// Returns all non-deleted node IDs, sorted by ID.
    fn node_ids(&self) -> Vec<NodeId>;

    /// Returns all node IDs including uncommitted/PENDING versions.
    ///
    /// Unlike `node_ids()` which pre-filters by current epoch, this method
    /// returns every node that has a version chain entry. Used by scan operators
    /// that perform their own MVCC visibility filtering (e.g. with transaction context).
    fn all_node_ids(&self) -> Vec<NodeId> {
        // Default: fall back to node_ids() for stores without MVCC
        self.node_ids()
    }

    /// Returns node IDs with a specific label.
    fn nodes_by_label(&self, label: &str) -> Vec<NodeId>;

    /// Returns the total number of non-deleted nodes.
    fn node_count(&self) -> usize;

    /// Returns the count of nodes with a specific label (O(1)).
    fn node_count_by_label(&self, label: &str) -> usize {
        // Default fallback (O(n) via nodes_by_label); concrete backends
        // (SubstrateStore) override with an O(1) label-index lookup.
        self.nodes_by_label(label).len()
    }

    /// Returns the total number of non-deleted edges.
    fn edge_count(&self) -> usize;

    // --- Entity metadata ---

    /// Returns the type string of an edge.
    fn edge_type(&self, id: EdgeId) -> Option<ArcStr>;

    /// Returns the type string of an edge visible to a specific transaction.
    ///
    /// Falls back to epoch-based `edge_type` if not overridden.
    fn edge_type_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Option<ArcStr> {
        let _ = (epoch, transaction_id);
        self.edge_type(id)
    }

    // --- Index introspection ---

    /// Returns `true` if a property index exists for the given property.
    ///
    /// The default returns `false`, which is correct for stores without indexes.
    fn has_property_index(&self, _property: &str) -> bool {
        false
    }

    /// Returns only the labels of a node without loading its properties.
    ///
    /// The default uses `get_node` and collects labels into a `Vec`; concrete
    /// backends may override with an O(1) lookup when labels live in a
    /// separate index.
    fn get_node_labels(&self, id: NodeId) -> Option<Vec<ArcStr>> {
        self.get_node(id).map(|n| n.labels.into_iter().collect())
    }

    // --- Filtered search ---

    /// Finds all nodes with a specific property value. Uses indexes when available.
    fn find_nodes_by_property(&self, property: &str, value: &Value) -> Vec<NodeId>;

    /// Finds nodes matching multiple property equality conditions.
    fn find_nodes_by_properties(&self, conditions: &[(&str, Value)]) -> Vec<NodeId>;

    /// Finds nodes whose property value falls within a range.
    fn find_nodes_in_range(
        &self,
        property: &str,
        min: Option<&Value>,
        max: Option<&Value>,
        min_inclusive: bool,
        max_inclusive: bool,
    ) -> Vec<NodeId>;

    /// Finds nodes with a substring match on a property, optionally filtered by label.
    ///
    /// The search is case-insensitive. `max_scan` caps the total number of nodes
    /// inspected to prevent worst-case O(N) full scans. Returns early when either
    /// `limit` results are found **or** `max_scan` nodes have been inspected.
    ///
    /// Default implementation returns empty — this is LpgStore-inherent for now.
    /// SubstrateStore exposes substring search natively through its L2 inverted
    /// index; this default keeps the trait object-safe while letting concrete
    /// backends override.
    fn find_nodes_by_label_property_contains_bounded(
        &self,
        _label: Option<&str>,
        _property: &str,
        _substring: &str,
        _limit: usize,
        _max_scan: usize,
    ) -> Vec<NodeId> {
        Vec::new()
    }

    // --- Auxiliary index registries (text/vector) ---

    /// Returns all registered text (BM25) index entries as `(key, index)` pairs.
    ///
    /// The key format is `"label:property"`. Default returns empty for stores
    /// that do not maintain a BM25 registry. `LpgStore` overrides; in substrate
    /// mode this stays empty because retrieval is served by the substrate L2
    /// inverted index + SYNAPSE layer natively.
    #[cfg(feature = "text-index")]
    fn text_index_entries(
        &self,
    ) -> Vec<(
        String,
        Arc<parking_lot::RwLock<crate::index::text::InvertedIndex>>,
    )> {
        Vec::new()
    }

    /// Returns all registered vector (HNSW) index entries as `(key, index)` pairs.
    ///
    /// The key format is `"label:property"`. Default returns empty. `LpgStore`
    /// overrides; substrate-backed stores expose vector retrieval through their
    /// native L1 vector store instead.
    #[cfg(feature = "vector-index")]
    fn vector_index_entries(&self) -> Vec<(String, Arc<crate::index::vector::HnswIndex>)> {
        Vec::new()
    }

    /// Retrieves the vector index for a label+property pair, if any.
    ///
    /// Default returns `None`. LpgStore overrides to look up its HNSW registry.
    #[cfg(feature = "vector-index")]
    fn get_vector_index(
        &self,
        _label: &str,
        _property: &str,
    ) -> Option<Arc<crate::index::vector::HnswIndex>> {
        None
    }

    /// Retrieves the text (BM25) index for a label+property pair, if any.
    ///
    /// Default returns `None`. LpgStore overrides to look up its text-index
    /// registry. Substrate surfaces equivalent functionality through its
    /// tiered-storage inverted index — callers interested in full-text
    /// retrieval should dispatch on `Arc<SubstrateStore>` directly in
    /// substrate-mode paths.
    #[cfg(feature = "text-index")]
    fn get_text_index(
        &self,
        _label: &str,
        _property: &str,
    ) -> Option<Arc<parking_lot::RwLock<crate::index::text::InvertedIndex>>> {
        None
    }

    /// Registers a pre-built vector (HNSW) index for a label+property pair.
    /// Default no-op — backends without named vector indexes ignore the
    /// registration.
    #[cfg(feature = "vector-index")]
    fn add_vector_index(
        &self,
        _label: &str,
        _property: &str,
        _index: Arc<crate::index::vector::HnswIndex>,
    ) {
    }

    /// Removes the vector index for a label+property pair. Returns `false`
    /// if none existed. Default returns `false`.
    #[cfg(feature = "vector-index")]
    fn remove_vector_index(&self, _label: &str, _property: &str) -> bool {
        false
    }

    /// Registers a pre-built text (BM25) index for a label+property pair.
    /// Default no-op.
    #[cfg(feature = "text-index")]
    fn add_text_index(
        &self,
        _label: &str,
        _property: &str,
        _index: Arc<parking_lot::RwLock<crate::index::text::InvertedIndex>>,
    ) {
    }

    /// Removes the text index for a label+property pair. Returns `false`
    /// if none existed. Default returns `false`.
    #[cfg(feature = "text-index")]
    fn remove_text_index(&self, _label: &str, _property: &str) -> bool {
        false
    }

    /// Returns node IDs matching an operator-style filter value on a
    /// given property (`{$gt: 5}`, `{$in: [..]}`, ...). Used by the
    /// query engine's server-side filter path.
    /// Default returns empty — backends without operator-filter support
    /// should fall back to `node_ids()` + manual scan.
    fn find_nodes_matching_filter(&self, _property: &str, _filter_value: &Value) -> Vec<NodeId> {
        Vec::new()
    }

    // --- Zone maps (skip pruning) ---

    /// Returns `true` if a node property predicate might match any nodes.
    /// Uses zone maps for early filtering.
    fn node_property_might_match(
        &self,
        property: &PropertyKey,
        op: CompareOp,
        value: &Value,
    ) -> bool;

    /// Returns `true` if an edge property predicate might match any edges.
    fn edge_property_might_match(
        &self,
        property: &PropertyKey,
        op: CompareOp,
        value: &Value,
    ) -> bool;

    // --- Statistics (for cost-based optimizer) ---

    /// Returns the current statistics snapshot (cheap Arc clone).
    fn statistics(&self) -> Arc<Statistics>;

    /// Estimates cardinality for a label scan.
    fn estimate_label_cardinality(&self, label: &str) -> f64;

    /// Estimates average degree for an edge type.
    fn estimate_avg_degree(&self, edge_type: &str, outgoing: bool) -> f64;

    // --- Epoch ---

    /// Returns the current MVCC epoch.
    fn current_epoch(&self) -> EpochId;

    // --- Schema introspection ---

    /// Returns all label names in the database.
    fn all_labels(&self) -> Vec<String> {
        Vec::new()
    }

    /// Returns all edge type names in the database.
    fn all_edge_types(&self) -> Vec<String> {
        Vec::new()
    }

    /// Returns all property key names used in the database.
    fn all_property_keys(&self) -> Vec<String> {
        Vec::new()
    }

    // --- Visibility checks (fast path, avoids building full entities) ---

    /// Checks if a node is visible at the given epoch without building the full Node.
    ///
    /// More efficient than `get_node_at_epoch(...).is_some()` because it skips
    /// label and property loading. Override in concrete stores for optimal
    /// performance.
    fn is_node_visible_at_epoch(&self, id: NodeId, epoch: EpochId) -> bool {
        self.get_node_at_epoch(id, epoch).is_some()
    }

    /// Checks if a node is visible to a specific transaction without building
    /// the full Node.
    fn is_node_visible_versioned(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        self.get_node_versioned(id, epoch, transaction_id).is_some()
    }

    /// Checks if an edge is visible at the given epoch without building the full Edge.
    ///
    /// More efficient than `get_edge_at_epoch(...).is_some()` because it skips
    /// type name resolution and property loading. Override in concrete stores
    /// for optimal performance.
    fn is_edge_visible_at_epoch(&self, id: EdgeId, epoch: EpochId) -> bool {
        self.get_edge_at_epoch(id, epoch).is_some()
    }

    /// Checks if an edge is visible to a specific transaction without building
    /// the full Edge.
    fn is_edge_visible_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        self.get_edge_versioned(id, epoch, transaction_id).is_some()
    }

    /// Filters node IDs to only those visible at the given epoch (batch).
    ///
    /// More efficient than per-node calls because implementations can hold
    /// a single lock for the entire batch.
    fn filter_visible_node_ids(&self, ids: &[NodeId], epoch: EpochId) -> Vec<NodeId> {
        ids.iter()
            .copied()
            .filter(|id| self.is_node_visible_at_epoch(*id, epoch))
            .collect()
    }

    /// Filters node IDs to only those visible to a transaction (batch).
    fn filter_visible_node_ids_versioned(
        &self,
        ids: &[NodeId],
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Vec<NodeId> {
        ids.iter()
            .copied()
            .filter(|id| self.is_node_visible_versioned(*id, epoch, transaction_id))
            .collect()
    }

    // --- History ---

    /// Returns all versions of a node with their creation/deletion epochs, newest first.
    ///
    /// Each entry is `(created_epoch, deleted_epoch, Node)`. Properties and labels
    /// reflect the current state (they are not versioned per-epoch).
    ///
    /// Default returns empty (not all backends track version history).
    fn get_node_history(&self, _id: NodeId) -> Vec<(EpochId, Option<EpochId>, Node)> {
        Vec::new()
    }

    /// Returns all versions of an edge with their creation/deletion epochs, newest first.
    ///
    /// Each entry is `(created_epoch, deleted_epoch, Edge)`. Properties reflect
    /// the current state (they are not versioned per-epoch).
    ///
    /// Default returns empty (not all backends track version history).
    fn get_edge_history(&self, _id: EdgeId) -> Vec<(EpochId, Option<EpochId>, Edge)> {
        Vec::new()
    }
}

/// Write operations for graph mutation.
///
/// Separated from [`GraphStore`] so read-only wrappers (snapshots, read
/// replicas) can implement only `GraphStore`. Any mutable store is also
/// readable via the supertrait bound.
pub trait GraphStoreMut: GraphStore {
    // --- Node creation ---

    /// Creates a new node with the given labels.
    fn create_node(&self, labels: &[&str]) -> NodeId;

    /// Creates a new node within a transaction context.
    fn create_node_versioned(
        &self,
        labels: &[&str],
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> NodeId;

    // --- Edge creation ---

    /// Creates a new edge between two nodes.
    fn create_edge(&self, src: NodeId, dst: NodeId, edge_type: &str) -> EdgeId;

    /// Creates a new edge within a transaction context.
    fn create_edge_versioned(
        &self,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> EdgeId;

    /// Creates multiple edges in batch (single lock acquisition).
    fn batch_create_edges(&self, edges: &[(NodeId, NodeId, &str)]) -> Vec<EdgeId>;

    // --- Deletion ---

    /// Deletes a node. Returns `true` if the node existed.
    fn delete_node(&self, id: NodeId) -> bool;

    /// Deletes a node within a transaction context. Returns `true` if the node existed.
    fn delete_node_versioned(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool;

    /// Deletes all edges connected to a node (DETACH DELETE).
    fn delete_node_edges(&self, node_id: NodeId);

    /// Deletes an edge. Returns `true` if the edge existed.
    fn delete_edge(&self, id: EdgeId) -> bool;

    /// Deletes an edge within a transaction context. Returns `true` if the edge existed.
    fn delete_edge_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool;

    // --- Property mutation ---

    /// Sets a property on a node.
    fn set_node_property(&self, id: NodeId, key: &str, value: Value);

    /// Sets a property on an edge.
    fn set_edge_property(&self, id: EdgeId, key: &str, value: Value);

    /// Sets a node property within a transaction, recording the previous value
    /// so it can be restored on rollback.
    ///
    /// Default delegates to [`set_node_property`](Self::set_node_property).
    fn set_node_property_versioned(
        &self,
        id: NodeId,
        key: &str,
        value: Value,
        _transaction_id: TransactionId,
    ) {
        self.set_node_property(id, key, value);
    }

    /// Sets an edge property within a transaction, recording the previous value
    /// so it can be restored on rollback.
    ///
    /// Default delegates to [`set_edge_property`](Self::set_edge_property).
    fn set_edge_property_versioned(
        &self,
        id: EdgeId,
        key: &str,
        value: Value,
        _transaction_id: TransactionId,
    ) {
        self.set_edge_property(id, key, value);
    }

    /// Removes a property from a node. Returns the previous value if it existed.
    fn remove_node_property(&self, id: NodeId, key: &str) -> Option<Value>;

    /// Removes a property from an edge. Returns the previous value if it existed.
    fn remove_edge_property(&self, id: EdgeId, key: &str) -> Option<Value>;

    /// Removes a node property within a transaction, recording the previous value
    /// so it can be restored on rollback.
    ///
    /// Default delegates to [`remove_node_property`](Self::remove_node_property).
    fn remove_node_property_versioned(
        &self,
        id: NodeId,
        key: &str,
        _transaction_id: TransactionId,
    ) -> Option<Value> {
        self.remove_node_property(id, key)
    }

    /// Removes an edge property within a transaction, recording the previous value
    /// so it can be restored on rollback.
    ///
    /// Default delegates to [`remove_edge_property`](Self::remove_edge_property).
    fn remove_edge_property_versioned(
        &self,
        id: EdgeId,
        key: &str,
        _transaction_id: TransactionId,
    ) -> Option<Value> {
        self.remove_edge_property(id, key)
    }

    // --- Label mutation ---

    /// Adds a label to a node. Returns `true` if the label was new.
    fn add_label(&self, node_id: NodeId, label: &str) -> bool;

    /// Removes a label from a node. Returns `true` if the label existed.
    fn remove_label(&self, node_id: NodeId, label: &str) -> bool;

    /// Adds a label within a transaction, recording the change for rollback.
    ///
    /// Default delegates to [`add_label`](Self::add_label).
    fn add_label_versioned(
        &self,
        node_id: NodeId,
        label: &str,
        _transaction_id: TransactionId,
    ) -> bool {
        self.add_label(node_id, label)
    }

    /// Removes a label within a transaction, recording the change for rollback.
    ///
    /// Default delegates to [`remove_label`](Self::remove_label).
    fn remove_label_versioned(
        &self,
        node_id: NodeId,
        label: &str,
        _transaction_id: TransactionId,
    ) -> bool {
        self.remove_label(node_id, label)
    }

    // --- Convenience (with default implementations) ---

    /// Creates a new node with labels and properties in one call.
    ///
    /// The default implementation calls [`create_node`](Self::create_node)
    /// followed by [`set_node_property`](Self::set_node_property) for each
    /// property. Implementations may override for atomicity or performance.
    fn create_node_with_props(
        &self,
        labels: &[&str],
        properties: &[(PropertyKey, Value)],
    ) -> NodeId {
        let id = self.create_node(labels);
        for (key, value) in properties {
            self.set_node_property(id, key.as_str(), value.clone());
        }
        id
    }

    /// Creates a new edge with properties in one call.
    ///
    /// The default implementation calls [`create_edge`](Self::create_edge)
    /// followed by [`set_edge_property`](Self::set_edge_property) for each
    /// property. Implementations may override for atomicity or performance.
    fn create_edge_with_props(
        &self,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
        properties: &[(PropertyKey, Value)],
    ) -> EdgeId {
        let id = self.create_edge(src, dst, edge_type);
        for (key, value) in properties {
            self.set_edge_property(id, key.as_str(), value.clone());
        }
        id
    }

    // --- MVCC hooks (T17 Wave B: 2026-04-23) ---
    //
    // These methods were LpgStore-inherent in the legacy design. With
    // substrate as the single production backend, substrate's
    // topology-as-storage model no longer tracks per-transaction
    // epochs, undo logs, or next-id-peek snapshots. The methods below
    // therefore default to no-ops: substrate inherits the defaults,
    // LpgStore overrides them (legacy behavior retained for the
    // archival `legacy-read` code path in obrain-migrate, and for
    // cognitive sidecars that still persist via the in-memory
    // LpgStore shadow).
    //
    // Moving these onto the trait lets `Session::resolve_store` and
    // `ObrainDB::store` hold `Arc<dyn GraphStoreMut>` instead of a
    // concrete `Arc<LpgStore>`, which is what finally unblocks the T17
    // cutover (the last concrete-type leak on the production path).

    /// Makes all PENDING version records from this transaction visible at
    /// the given commit epoch. No-op for backends without MVCC.
    fn finalize_version_epochs(&self, _transaction_id: TransactionId, _commit_epoch: EpochId) {}

    /// Commits property changes from this transaction (discards undo logs,
    /// making the mutations permanent). No-op for backends without MVCC.
    fn commit_transaction_properties(&self, _transaction_id: TransactionId) {}

    /// Rolls back property changes from this transaction (replays the undo
    /// log). No-op for backends without MVCC.
    fn rollback_transaction_properties(&self, _transaction_id: TransactionId) {}

    /// Rolls back property changes from this transaction back to the given
    /// undo-log position (savepoint-level rollback). No-op for backends
    /// without MVCC.
    fn rollback_transaction_properties_to(&self, _transaction_id: TransactionId, _since: usize) {}

    /// Publishes the given epoch as the store's "current" read epoch so that
    /// convenience lookups surface committed versions immediately. No-op for
    /// backends without MVCC.
    fn sync_epoch(&self, _epoch: EpochId) {}

    /// Prunes version records strictly older than `min_epoch` when the
    /// backend maintains a version history. No-op for backends without MVCC.
    fn gc_versions(&self, _min_epoch: EpochId) {}

    /// Returns the next node ID that would be allocated by `create_node`.
    /// Used by savepoint bookkeeping to rewind a pending allocator. Backends
    /// without a stable next-id peek return `0` — callers must treat a zero
    /// result as "savepoints unsupported" and skip the rewind path.
    fn peek_next_node_id(&self) -> u64 {
        0
    }

    /// Returns the current position of the property undo log for the given
    /// transaction. Used by savepoints. Backends without undo logs return
    /// `0` — callers interpret zero as "no undo log available".
    fn property_undo_log_position(&self, _transaction_id: TransactionId) -> usize {
        0
    }

    /// Discards all uncommitted (PENDING) version records for the given
    /// transaction. Used when aborting a transaction. No-op for backends
    /// without per-transaction version staging.
    fn discard_uncommitted_versions(&self, _transaction_id: TransactionId) {}

    /// Returns the next edge ID that would be allocated by `create_edge`.
    /// Used by savepoint bookkeeping. Backends without a stable peek
    /// return `0` — callers interpret zero as "savepoints unsupported".
    fn peek_next_edge_id(&self) -> u64 {
        0
    }

    /// Discards nodes and edges with the given ID lists for the current
    /// transaction. Used by savepoint rollback to cull entities allocated
    /// after the savepoint. No-op for backends without savepoint-level
    /// discard.
    fn discard_entities_by_id(
        &self,
        _transaction_id: TransactionId,
        _node_ids: &[NodeId],
        _edge_ids: &[EdgeId],
    ) {
    }

    // --- Named-graph hooks (T17 Step 24: 2026-04-23) ---
    //
    // LpgStore supports named sub-graphs (CREATE GRAPH g / DROP GRAPH g /
    // USE GRAPH g). The substrate backend does not — one database = one
    // graph in the topology-as-storage model. Named-graph methods default
    // to "not supported" values so that the Session layer can drive both
    // backends through `Arc<dyn GraphStoreMut>`; callers are already
    // defensive about `None` / empty results (fall back to the root
    // store when a named graph is missing). A substrate-native
    // multi-graph story is a separate RFC and is not required for T17.

    /// Returns a named graph by name, or `None` if it does not exist /
    /// named graphs are not supported by this backend.
    fn named_graph(&self, _name: &str) -> Option<Arc<dyn GraphStoreMut>> {
        None
    }

    /// Creates a named graph. Returns `Ok(true)` on success, `Ok(false)`
    /// if it already exists. The default implementation returns
    /// `Ok(false)` — backends without named-graph support treat every
    /// name as a no-op.
    fn create_named_graph(&self, _name: &str) -> Result<bool, AllocError> {
        Ok(false)
    }

    /// Drops a named graph. Returns `false` if it did not exist / named
    /// graphs are not supported.
    fn drop_named_graph(&self, _name: &str) -> bool {
        false
    }

    /// Returns the names of all named graphs. Empty for backends without
    /// named-graph support.
    fn named_graph_names(&self) -> Vec<String> {
        Vec::new()
    }

    /// Creates a node with a caller-specified `NodeId`. Used by
    /// reversible contraction (see `expand_supernode`) to restore
    /// original IDs after a round-trip. Default returns an error —
    /// substrate allocates monotonically and does not support
    /// reusing a specific slot; the legacy `LpgStore` backend
    /// overrides this with its inherent slot-reclamation routine.
    fn create_node_with_id(&self, _id: NodeId, _labels: &[&str]) -> Result<(), AllocError> {
        Err(AllocError::OutOfMemory)
    }

    /// Copies all nodes and edges from the `source` named graph into the
    /// `dest` named graph (both `None` = root graph). Used by
    /// `CREATE GRAPH g AS COPY OF h`. The default returns an error for
    /// backends without named-graph support.
    fn copy_named_graph(&self, _source: Option<&str>, _dest: Option<&str>) -> Result<(), String> {
        Err("named graphs are not supported by this backend".to_string())
    }

    // --- LpgStore-legacy inspection hooks (T17 Step 24: 2026-04-23) ---
    //
    // These methods back the `SHOW DATABASE` / `SHOW INDEXES` admin
    // paths and the GQL index-management statements that were
    // LpgStore-inherent. Substrate publishes the same information
    // through its tiered-storage zone headers + registry, so the
    // defaults return empty / not-supported values — sufficient for
    // the admin.rs paths to compile against `Arc<dyn GraphStoreMut>`.
    // LpgStore overrides each method by delegating to its inherent
    // implementation (legacy-read path + in-memory shadow).

    /// Returns every node in the store. Default iterates through
    /// `node_ids()` + `get_node()` (hydrating each node on demand).
    /// Implementations with a faster path (e.g. DashMap iteration)
    /// may override.
    fn all_nodes(&self) -> Vec<Node> {
        self.node_ids()
            .into_iter()
            .filter_map(|id| self.get_node(id))
            .collect()
    }

    /// Returns all nodes carrying the given label. Used by index
    /// builders. Default returns empty — backends that do not maintain
    /// a label→nodes reverse index should populate on demand via
    /// `nodes_by_label` (IDs) + `get_node` (hydration).
    fn nodes_with_label(&self, _label: &str) -> Vec<Node> {
        Vec::new()
    }

    /// Returns all edges carrying the given edge-type. Used by schema
    /// introspection. Default returns empty.
    fn edges_with_type(&self, _edge_type: &str) -> Vec<Edge> {
        Vec::new()
    }

    /// Returns every edge in the store. Used by schema introspection
    /// + legacy snapshot paths. Default returns empty — substrate
    /// surfaces edges through `edges_from(node, direction)` which
    /// scales to live-graph sizes.
    fn all_edges(&self) -> Vec<Edge> {
        Vec::new()
    }

    /// Creates a property index over the given property name. Default
    /// no-op — substrate does not maintain property-level side
    /// indexes; the tiered-storage column layout provides equivalent
    /// functionality.
    fn create_property_index(&self, _property: &str) {}

    /// Drops a property index. Default returns `false` ("nothing was
    /// dropped") — substrate does not maintain named property
    /// indexes.
    fn drop_property_index(&self, _property: &str) -> bool {
        false
    }

    /// Returns a detailed RAM memory breakdown of the store.
    ///
    /// Default returns all-zero totals: this category of
    /// introspection is LpgStore-specific (VersionChain maps + MVCC
    /// chains + property DashMaps). Substrate exposes zone footprints
    /// through a different API (future
    /// `SubstrateStore::memory_breakdown`) that reports mmap zones
    /// instead of heap allocations.
    fn memory_breakdown(
        &self,
    ) -> (
        obrain_common::memory::usage::StoreMemory,
        obrain_common::memory::usage::IndexMemory,
        obrain_common::memory::usage::MvccMemory,
        obrain_common::memory::usage::StringPoolMemory,
    ) {
        (
            obrain_common::memory::usage::StoreMemory::default(),
            obrain_common::memory::usage::IndexMemory::default(),
            obrain_common::memory::usage::MvccMemory::default(),
            obrain_common::memory::usage::StringPoolMemory::default(),
        )
    }
}
