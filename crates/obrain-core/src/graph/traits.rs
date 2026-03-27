//! Storage traits for the graph engine.
//!
//! These traits capture the minimal surface that query operators need from
//! the graph store. The split is intentional:
//!
//! - [`GraphStore`]: Read-only operations (scans, lookups, traversal, statistics)
//! - [`GraphStoreMut`]: Write operations (create, delete, mutate)
//!
//! Admin operations (index management, MVCC internals, schema introspection,
//! statistics recomputation, WAL recovery) stay on the concrete [`LpgStore`]
//! and are not part of these traits.
//!
//! ## Design rationale
//!
//! The traits work with typed graph objects (`Node`, `Edge`, `Value`) rather
//! than raw bytes. This preserves zero-overhead access for in-memory storage
//! while allowing future backends (SpilloverStore, disk-backed) to implement
//! the same interface with transparent serialization where needed.
//!
//! [`LpgStore`]: crate::graph::lpg::LpgStore

use crate::graph::Direction;
use crate::graph::lpg::CompareOp;
use crate::graph::lpg::{Edge, Node};
use crate::statistics::Statistics;
use arcstr::ArcStr;
use grafeo_common::types::{EdgeId, EpochId, NodeId, PropertyKey, TransactionId, Value};
use grafeo_common::utils::hash::FxHashMap;
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
}
