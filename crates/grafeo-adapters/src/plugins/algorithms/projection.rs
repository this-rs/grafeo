//! Graph projections — virtual filtered views over a graph store.
//!
//! A projection wraps an existing [`GraphStore`] and applies filters on node
//! labels, edge types, and property predicates. The resulting view implements
//! [`GraphStore`] itself, so every existing algorithm (PageRank, Louvain,
//! HITS, etc.) runs on it **without modification**.
//!
//! This is the Rust equivalent of Neo4j GDS `graph.project()`.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────┐
//! │        GraphProjection          │
//! │  ┌───────────┐ ┌─────────────┐ │
//! │  │ NodeFilter │ │ EdgeFilter  │ │
//! │  └───────────┘ └─────────────┘ │
//! │         │              │        │
//! │         ▼              ▼        │
//! │   ┌─────────────────────────┐   │
//! │   │   &dyn GraphStore       │   │
//! │   └─────────────────────────┘   │
//! └─────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```no_run
//! use grafeo_adapters::plugins::algorithms::projection::ProjectionBuilder;
//! use grafeo_core::graph::lpg::LpgStore;
//! use grafeo_core::graph::GraphStore;
//!
//! let store = LpgStore::new().unwrap();
//! // ... populate store ...
//!
//! // Project only File and Function nodes with IMPORTS edges
//! let projection = ProjectionBuilder::new(&store)
//!     .with_node_labels(&["File", "Function"])
//!     .with_edge_types(&["IMPORTS"])
//!     .build();
//!
//! // Use projection as any GraphStore — run PageRank, Louvain, etc.
//! let node_ids = projection.node_ids();
//! ```

use arcstr::ArcStr;
use grafeo_common::types::{EdgeId, EpochId, NodeId, PropertyKey, TransactionId, Value};
use grafeo_common::utils::hash::FxHashMap;
use grafeo_core::graph::lpg::{CompareOp, Edge, Node};
use grafeo_core::graph::{Direction, GraphStore};
use grafeo_core::statistics::Statistics;
use std::collections::HashSet;
use std::sync::Arc;

// ============================================================================
// Property Predicates
// ============================================================================

/// A predicate that tests a single property value.
///
/// Used by [`NodeFilter`] and [`EdgeFilter`] to filter entities based on
/// their property values. The predicate is a boxed closure for maximum
/// flexibility.
///
/// # Example
///
/// ```no_run
/// use grafeo_adapters::plugins::algorithms::projection::PropertyPredicate;
/// use grafeo_common::types::Value;
///
/// let pred = PropertyPredicate::new("weight", |v| {
///     matches!(v, Value::Float64(w) if *w > 0.5)
/// });
/// ```
pub struct PropertyPredicate {
    /// The property key to test.
    pub key: PropertyKey,
    /// The predicate function. Returns `true` if the entity passes the filter.
    predicate: Box<dyn Fn(&Value) -> bool + Send + Sync>,
}

impl PropertyPredicate {
    /// Creates a new property predicate.
    ///
    /// # Arguments
    ///
    /// * `key` - The property key to test
    /// * `predicate` - A closure that receives the property value and returns
    ///   `true` if the entity should be included in the projection
    ///
    /// # Complexity
    ///
    /// O(1) — construction only stores the closure.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use grafeo_adapters::plugins::algorithms::projection::PropertyPredicate;
    /// use grafeo_common::types::Value;
    ///
    /// let pred = PropertyPredicate::new("status", |v| {
    ///     matches!(v, Value::String(s) if s.as_str() == "active")
    /// });
    /// ```
    pub fn new<F>(key: impl Into<PropertyKey>, predicate: F) -> Self
    where
        F: Fn(&Value) -> bool + Send + Sync + 'static,
    {
        Self {
            key: key.into(),
            predicate: Box::new(predicate),
        }
    }

    /// Tests a value against this predicate.
    ///
    /// # Arguments
    ///
    /// * `value` - The value to test
    ///
    /// # Returns
    ///
    /// `true` if the value satisfies the predicate.
    ///
    /// # Complexity
    ///
    /// O(1) — single closure invocation.
    #[inline]
    pub fn test(&self, value: &Value) -> bool {
        (self.predicate)(value)
    }
}

impl std::fmt::Debug for PropertyPredicate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PropertyPredicate")
            .field("key", &self.key)
            .field("predicate", &"<fn>")
            .finish()
    }
}

// ============================================================================
// Node Filter
// ============================================================================

/// Filter configuration for nodes in a graph projection.
///
/// A node passes the filter if:
/// 1. It has at least one of the specified labels (or no label filter is set), **AND**
/// 2. All property predicates evaluate to `true` on the node's properties.
///
/// # Complexity
///
/// Applying the filter to a single node is O(L + P) where L is the number
/// of labels on the node and P is the number of property predicates.
///
/// # Example
///
/// ```no_run
/// use grafeo_adapters::plugins::algorithms::projection::{NodeFilter, PropertyPredicate};
/// use grafeo_common::types::Value;
///
/// let filter = NodeFilter::new()
///     .labels(vec!["File".into(), "Function".into()])
///     .property(PropertyPredicate::new("active", |v| {
///         matches!(v, Value::Bool(true))
///     }));
/// ```
#[derive(Debug, Default)]
pub struct NodeFilter {
    /// If non-empty, only nodes with at least one of these labels pass.
    pub labels: HashSet<String>,
    /// Property predicates that must all be satisfied.
    pub property_predicates: Vec<PropertyPredicate>,
}

impl NodeFilter {
    /// Creates a new empty node filter (accepts all nodes).
    ///
    /// # Returns
    ///
    /// A [`NodeFilter`] with no constraints.
    ///
    /// # Complexity
    ///
    /// O(1)
    ///
    /// # Example
    ///
    /// ```no_run
    /// use grafeo_adapters::plugins::algorithms::projection::NodeFilter;
    ///
    /// let filter = NodeFilter::new(); // accepts everything
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the allowed label set (builder pattern).
    ///
    /// # Arguments
    ///
    /// * `labels` - Labels to allow. A node must have at least one.
    ///
    /// # Returns
    ///
    /// `self` for chaining.
    ///
    /// # Complexity
    ///
    /// O(L) where L is the number of labels.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use grafeo_adapters::plugins::algorithms::projection::NodeFilter;
    ///
    /// let filter = NodeFilter::new().labels(vec!["Person".into()]);
    /// ```
    pub fn labels(mut self, labels: Vec<String>) -> Self {
        self.labels = labels.into_iter().collect();
        self
    }

    /// Adds a property predicate (builder pattern).
    ///
    /// # Arguments
    ///
    /// * `predicate` - A [`PropertyPredicate`] that must evaluate to `true`.
    ///
    /// # Returns
    ///
    /// `self` for chaining.
    ///
    /// # Complexity
    ///
    /// O(1)
    ///
    /// # Example
    ///
    /// ```no_run
    /// use grafeo_adapters::plugins::algorithms::projection::{NodeFilter, PropertyPredicate};
    /// use grafeo_common::types::Value;
    ///
    /// let filter = NodeFilter::new()
    ///     .property(PropertyPredicate::new("weight", |v| {
    ///         matches!(v, Value::Float64(w) if *w > 0.5)
    ///     }));
    /// ```
    pub fn property(mut self, predicate: PropertyPredicate) -> Self {
        self.property_predicates.push(predicate);
        self
    }

    /// Tests whether a node passes this filter.
    ///
    /// # Arguments
    ///
    /// * `node` - The node to test
    /// * `store` - The backing graph store (for property lookups)
    ///
    /// # Returns
    ///
    /// `true` if the node satisfies all filter criteria.
    ///
    /// # Complexity
    ///
    /// O(L + P) where L = node label count, P = predicate count.
    pub fn accepts(&self, node: &Node, store: &dyn GraphStore) -> bool {
        // Label filter
        if !self.labels.is_empty() {
            let has_label = node.labels.iter().any(|l| self.labels.contains(l.as_str()));
            if !has_label {
                return false;
            }
        }

        // Property predicates
        for pred in &self.property_predicates {
            match store.get_node_property(node.id, &pred.key) {
                Some(ref val) => {
                    if !pred.test(val) {
                        return false;
                    }
                }
                None => return false,
            }
        }

        true
    }

    /// Returns `true` if this filter has no constraints (accepts everything).
    ///
    /// # Returns
    ///
    /// `true` if both labels and predicates are empty.
    ///
    /// # Complexity
    ///
    /// O(1)
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.labels.is_empty() && self.property_predicates.is_empty()
    }
}

// ============================================================================
// Edge Filter
// ============================================================================

/// Filter configuration for edges in a graph projection.
///
/// An edge passes the filter if:
/// 1. Its type is in the allowed set (or no type filter is set), **AND**
/// 2. All property predicates evaluate to `true` on the edge's properties.
///
/// # Complexity
///
/// Applying the filter to a single edge is O(P) where P is the number
/// of property predicates (type check is O(1) via `HashSet`).
///
/// # Example
///
/// ```no_run
/// use grafeo_adapters::plugins::algorithms::projection::EdgeFilter;
///
/// let filter = EdgeFilter::new()
///     .types(vec!["IMPORTS".into(), "CALLS".into()]);
/// ```
#[derive(Debug, Default)]
pub struct EdgeFilter {
    /// If non-empty, only edges with one of these types pass.
    pub edge_types: HashSet<String>,
    /// Property predicates that must all be satisfied.
    pub property_predicates: Vec<PropertyPredicate>,
}

impl EdgeFilter {
    /// Creates a new empty edge filter (accepts all edges).
    ///
    /// # Returns
    ///
    /// An [`EdgeFilter`] with no constraints.
    ///
    /// # Complexity
    ///
    /// O(1)
    ///
    /// # Example
    ///
    /// ```no_run
    /// use grafeo_adapters::plugins::algorithms::projection::EdgeFilter;
    ///
    /// let filter = EdgeFilter::new(); // accepts everything
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the allowed edge type set (builder pattern).
    ///
    /// # Arguments
    ///
    /// * `types` - Edge types to allow.
    ///
    /// # Returns
    ///
    /// `self` for chaining.
    ///
    /// # Complexity
    ///
    /// O(T) where T is the number of types.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use grafeo_adapters::plugins::algorithms::projection::EdgeFilter;
    ///
    /// let filter = EdgeFilter::new().types(vec!["KNOWS".into()]);
    /// ```
    pub fn types(mut self, types: Vec<String>) -> Self {
        self.edge_types = types.into_iter().collect();
        self
    }

    /// Adds a property predicate (builder pattern).
    ///
    /// # Arguments
    ///
    /// * `predicate` - A [`PropertyPredicate`] that must evaluate to `true`.
    ///
    /// # Returns
    ///
    /// `self` for chaining.
    ///
    /// # Complexity
    ///
    /// O(1)
    ///
    /// # Example
    ///
    /// ```no_run
    /// use grafeo_adapters::plugins::algorithms::projection::{EdgeFilter, PropertyPredicate};
    /// use grafeo_common::types::Value;
    ///
    /// let filter = EdgeFilter::new()
    ///     .property(PropertyPredicate::new("weight", |v| {
    ///         matches!(v, Value::Float64(w) if *w > 0.1)
    ///     }));
    /// ```
    pub fn property(mut self, predicate: PropertyPredicate) -> Self {
        self.property_predicates.push(predicate);
        self
    }

    /// Tests whether an edge passes this filter.
    ///
    /// # Arguments
    ///
    /// * `edge` - The edge to test
    /// * `store` - The backing graph store (for property lookups)
    ///
    /// # Returns
    ///
    /// `true` if the edge satisfies all filter criteria.
    ///
    /// # Complexity
    ///
    /// O(P) where P = predicate count (type check is O(1)).
    pub fn accepts(&self, edge: &Edge, store: &dyn GraphStore) -> bool {
        // Edge type filter
        if !self.edge_types.is_empty() && !self.edge_types.contains(edge.edge_type.as_str()) {
            return false;
        }

        // Property predicates
        for pred in &self.property_predicates {
            match store.get_edge_property(edge.id, &pred.key) {
                Some(ref val) => {
                    if !pred.test(val) {
                        return false;
                    }
                }
                None => return false,
            }
        }

        true
    }

    /// Tests whether an edge passes this filter using only its type string.
    ///
    /// This is a fast path for when only edge type filtering is needed
    /// (no property predicates). Returns `None` if property predicates
    /// exist and a full edge check is required.
    ///
    /// # Arguments
    ///
    /// * `edge_type` - The edge type string to test
    ///
    /// # Returns
    ///
    /// `Some(true)` if the edge type passes (and no property predicates),
    /// `Some(false)` if the edge type fails, `None` if property predicates
    /// need checking.
    ///
    /// # Complexity
    ///
    /// O(1) for type-only check.
    #[inline]
    pub fn accepts_type(&self, edge_type: &str) -> Option<bool> {
        if !self.edge_types.is_empty() && !self.edge_types.contains(edge_type) {
            return Some(false);
        }
        if self.property_predicates.is_empty() {
            Some(true)
        } else {
            None // Need full edge check
        }
    }

    /// Returns `true` if this filter has no constraints (accepts everything).
    ///
    /// # Returns
    ///
    /// `true` if both types and predicates are empty.
    ///
    /// # Complexity
    ///
    /// O(1)
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.edge_types.is_empty() && self.property_predicates.is_empty()
    }
}

// ============================================================================
// GraphProjection
// ============================================================================

/// A virtual filtered view over an existing graph store.
///
/// `GraphProjection` wraps a reference to any [`GraphStore`] and applies
/// [`NodeFilter`] + [`EdgeFilter`] lazily on every read operation. It
/// implements [`GraphStore`] itself, so all existing algorithms work on
/// it without modification.
///
/// **No data is copied** — filtering happens at query time. This makes
/// projection creation O(1) and memory usage O(F) where F is the filter
/// configuration size.
///
/// # Performance
///
/// The overhead of filtering is typically <5% compared to running directly
/// on the underlying store, because:
/// - Label checks use `HashSet::contains` (O(1) amortized)
/// - Edge type checks use `HashSet::contains` (O(1) amortized)
/// - Property predicates are only evaluated when present
///
/// # Example
///
/// ```no_run
/// use grafeo_adapters::plugins::algorithms::projection::ProjectionBuilder;
/// use grafeo_core::graph::lpg::LpgStore;
/// use grafeo_core::graph::GraphStore;
///
/// let store = LpgStore::new().unwrap();
/// // ... populate the store with nodes and edges ...
///
/// // Create a projection for code analysis
/// let code_view = ProjectionBuilder::new(&store)
///     .with_node_labels(&["File", "Function", "Class"])
///     .with_edge_types(&["IMPORTS", "CALLS", "EXTENDS"])
///     .build();
///
/// // Run any algorithm on the filtered view
/// let nodes = code_view.node_ids();
/// ```
pub struct GraphProjection<'a> {
    /// The underlying graph store.
    store: &'a dyn GraphStore,
    /// Node filter configuration.
    node_filter: NodeFilter,
    /// Edge filter configuration.
    edge_filter: EdgeFilter,
}

impl<'a> GraphProjection<'a> {
    /// Creates a new graph projection with the given filters.
    ///
    /// Prefer using [`ProjectionBuilder`] for a more ergonomic API.
    ///
    /// # Arguments
    ///
    /// * `store` - The underlying graph store to project over
    /// * `node_filter` - Filter for nodes
    /// * `edge_filter` - Filter for edges
    ///
    /// # Returns
    ///
    /// A new `GraphProjection` wrapping the store.
    ///
    /// # Complexity
    ///
    /// O(1) — no data is copied or pre-computed.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use grafeo_adapters::plugins::algorithms::projection::{
    ///     GraphProjection, NodeFilter, EdgeFilter,
    /// };
    /// use grafeo_core::graph::lpg::LpgStore;
    ///
    /// let store = LpgStore::new().unwrap();
    /// let projection = GraphProjection::new(
    ///     &store,
    ///     NodeFilter::new().labels(vec!["Person".into()]),
    ///     EdgeFilter::new().types(vec!["KNOWS".into()]),
    /// );
    /// ```
    pub fn new(
        store: &'a dyn GraphStore,
        node_filter: NodeFilter,
        edge_filter: EdgeFilter,
    ) -> Self {
        Self {
            store,
            node_filter,
            edge_filter,
        }
    }

    /// Returns a reference to the underlying store.
    ///
    /// # Returns
    ///
    /// The wrapped `&dyn GraphStore`.
    ///
    /// # Complexity
    ///
    /// O(1)
    #[inline]
    pub fn inner_store(&self) -> &dyn GraphStore {
        self.store
    }

    /// Returns a reference to the node filter.
    ///
    /// # Returns
    ///
    /// The current [`NodeFilter`] configuration.
    ///
    /// # Complexity
    ///
    /// O(1)
    #[inline]
    pub fn node_filter(&self) -> &NodeFilter {
        &self.node_filter
    }

    /// Returns a reference to the edge filter.
    ///
    /// # Returns
    ///
    /// The current [`EdgeFilter`] configuration.
    ///
    /// # Complexity
    ///
    /// O(1)
    #[inline]
    pub fn edge_filter(&self) -> &EdgeFilter {
        &self.edge_filter
    }

    /// Checks if a node ID passes the node filter.
    ///
    /// # Arguments
    ///
    /// * `id` - The node ID to check
    ///
    /// # Returns
    ///
    /// `true` if the node exists and passes all filter criteria.
    ///
    /// # Complexity
    ///
    /// O(L + P) where L = label count, P = predicate count.
    #[inline]
    fn node_passes(&self, id: NodeId) -> bool {
        if self.node_filter.is_empty() {
            return self.store.get_node(id).is_some();
        }
        match self.store.get_node(id) {
            Some(node) => self.node_filter.accepts(&node, self.store),
            None => false,
        }
    }

    /// Checks if an edge ID passes the edge filter (and both endpoints pass node filter).
    ///
    /// # Arguments
    ///
    /// * `edge_id` - The edge ID to check
    ///
    /// # Returns
    ///
    /// `true` if the edge exists, passes the edge filter, and both
    /// source and destination nodes pass the node filter.
    ///
    /// # Complexity
    ///
    /// O(L + P + E) where L/P are node filter costs and E is edge filter cost.
    fn edge_passes(&self, edge_id: EdgeId) -> bool {
        let Some(edge) = self.store.get_edge(edge_id) else {
            return false;
        };

        // Check edge filter
        if !self.edge_filter.is_empty() && !self.edge_filter.accepts(&edge, self.store) {
            return false;
        }

        // Check that both endpoints pass the node filter
        self.node_passes(edge.src) && self.node_passes(edge.dst)
    }
}

// ============================================================================
// GraphStore Implementation
// ============================================================================

impl GraphStore for GraphProjection<'_> {
    // --- Point lookups ---

    fn get_node(&self, id: NodeId) -> Option<Node> {
        let node = self.store.get_node(id)?;
        if self.node_filter.is_empty() || self.node_filter.accepts(&node, self.store) {
            Some(node)
        } else {
            None
        }
    }

    fn get_edge(&self, id: EdgeId) -> Option<Edge> {
        let edge = self.store.get_edge(id)?;
        if !self.edge_filter.is_empty() && !self.edge_filter.accepts(&edge, self.store) {
            return None;
        }
        // Both endpoints must pass node filter
        if !self.node_passes(edge.src) || !self.node_passes(edge.dst) {
            return None;
        }
        Some(edge)
    }

    fn get_node_versioned(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Option<Node> {
        let node = self.store.get_node_versioned(id, epoch, transaction_id)?;
        if self.node_filter.is_empty() || self.node_filter.accepts(&node, self.store) {
            Some(node)
        } else {
            None
        }
    }

    fn get_edge_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Option<Edge> {
        let edge = self.store.get_edge_versioned(id, epoch, transaction_id)?;
        if !self.edge_filter.is_empty() && !self.edge_filter.accepts(&edge, self.store) {
            return None;
        }
        if !self.node_passes(edge.src) || !self.node_passes(edge.dst) {
            return None;
        }
        Some(edge)
    }

    fn get_node_at_epoch(&self, id: NodeId, epoch: EpochId) -> Option<Node> {
        let node = self.store.get_node_at_epoch(id, epoch)?;
        if self.node_filter.is_empty() || self.node_filter.accepts(&node, self.store) {
            Some(node)
        } else {
            None
        }
    }

    fn get_edge_at_epoch(&self, id: EdgeId, epoch: EpochId) -> Option<Edge> {
        let edge = self.store.get_edge_at_epoch(id, epoch)?;
        if !self.edge_filter.is_empty() && !self.edge_filter.accepts(&edge, self.store) {
            return None;
        }
        if !self.node_passes(edge.src) || !self.node_passes(edge.dst) {
            return None;
        }
        Some(edge)
    }

    // --- Property access ---

    fn get_node_property(&self, id: NodeId, key: &PropertyKey) -> Option<Value> {
        if !self.node_passes(id) {
            return None;
        }
        self.store.get_node_property(id, key)
    }

    fn get_edge_property(&self, id: EdgeId, key: &PropertyKey) -> Option<Value> {
        if !self.edge_passes(id) {
            return None;
        }
        self.store.get_edge_property(id, key)
    }

    fn get_node_property_batch(&self, ids: &[NodeId], key: &PropertyKey) -> Vec<Option<Value>> {
        ids.iter()
            .map(|&id| {
                if self.node_passes(id) {
                    self.store.get_node_property(id, key)
                } else {
                    None
                }
            })
            .collect()
    }

    fn get_nodes_properties_batch(&self, ids: &[NodeId]) -> Vec<FxHashMap<PropertyKey, Value>> {
        ids.iter()
            .map(|&id| {
                if self.node_passes(id) {
                    self.store
                        .get_nodes_properties_batch(&[id])
                        .into_iter()
                        .next()
                        .unwrap_or_default()
                } else {
                    FxHashMap::default()
                }
            })
            .collect()
    }

    fn get_nodes_properties_selective_batch(
        &self,
        ids: &[NodeId],
        keys: &[PropertyKey],
    ) -> Vec<FxHashMap<PropertyKey, Value>> {
        ids.iter()
            .map(|&id| {
                if self.node_passes(id) {
                    self.store
                        .get_nodes_properties_selective_batch(&[id], keys)
                        .into_iter()
                        .next()
                        .unwrap_or_default()
                } else {
                    FxHashMap::default()
                }
            })
            .collect()
    }

    fn get_edges_properties_selective_batch(
        &self,
        ids: &[EdgeId],
        keys: &[PropertyKey],
    ) -> Vec<FxHashMap<PropertyKey, Value>> {
        ids.iter()
            .map(|&id| {
                if self.edge_passes(id) {
                    self.store
                        .get_edges_properties_selective_batch(&[id], keys)
                        .into_iter()
                        .next()
                        .unwrap_or_default()
                } else {
                    FxHashMap::default()
                }
            })
            .collect()
    }

    // --- Traversal ---

    fn neighbors(&self, node: NodeId, direction: Direction) -> Vec<NodeId> {
        if !self.node_passes(node) {
            return Vec::new();
        }

        // If we have edge filters, we need to check edges
        if self.edge_filter.is_empty() && self.node_filter.is_empty() {
            return self.store.neighbors(node, direction);
        }

        // Get edges and filter
        let edges = self.store.edges_from(node, direction);
        let mut result = Vec::with_capacity(edges.len());
        for (target, edge_id) in edges {
            if !self.node_passes(target) {
                continue;
            }
            if self.edge_filter.is_empty() {
                result.push(target);
                continue;
            }
            // Check edge type first (fast path)
            if let Some(edge_type) = self.store.edge_type(edge_id) {
                match self.edge_filter.accepts_type(edge_type.as_str()) {
                    Some(true) => result.push(target),
                    Some(false) => {}
                    None => {
                        // Need full edge check (property predicates)
                        if let Some(edge) = self.store.get_edge(edge_id)
                            && self.edge_filter.accepts(&edge, self.store)
                        {
                            result.push(target);
                        }
                    }
                }
            }
        }
        result
    }

    fn edges_from(&self, node: NodeId, direction: Direction) -> Vec<(NodeId, EdgeId)> {
        if !self.node_passes(node) {
            return Vec::new();
        }

        let edges = self.store.edges_from(node, direction);

        if self.edge_filter.is_empty() && self.node_filter.is_empty() {
            return edges;
        }

        edges
            .into_iter()
            .filter(|&(target, edge_id)| {
                if !self.node_passes(target) {
                    return false;
                }
                if self.edge_filter.is_empty() {
                    return true;
                }
                if let Some(edge_type) = self.store.edge_type(edge_id) {
                    match self.edge_filter.accepts_type(edge_type.as_str()) {
                        Some(accepted) => accepted,
                        None => {
                            if let Some(edge) = self.store.get_edge(edge_id) {
                                self.edge_filter.accepts(&edge, self.store)
                            } else {
                                false
                            }
                        }
                    }
                } else {
                    false
                }
            })
            .collect()
    }

    fn out_degree(&self, node: NodeId) -> usize {
        self.edges_from(node, Direction::Outgoing).len()
    }

    fn in_degree(&self, node: NodeId) -> usize {
        self.edges_from(node, Direction::Incoming).len()
    }

    fn has_backward_adjacency(&self) -> bool {
        self.store.has_backward_adjacency()
    }

    // --- Scans ---

    fn node_ids(&self) -> Vec<NodeId> {
        if self.node_filter.is_empty() {
            return self.store.node_ids();
        }

        // Optimization: if only label filter, use nodes_by_label
        if self.node_filter.property_predicates.is_empty() && !self.node_filter.labels.is_empty() {
            let mut ids: HashSet<NodeId> = HashSet::new();
            for label in &self.node_filter.labels {
                for id in self.store.nodes_by_label(label) {
                    ids.insert(id);
                }
            }
            let mut result: Vec<NodeId> = ids.into_iter().collect();
            result.sort();
            return result;
        }

        // General case: scan all nodes and filter
        self.store
            .node_ids()
            .into_iter()
            .filter(|&id| {
                if let Some(node) = self.store.get_node(id) {
                    self.node_filter.accepts(&node, self.store)
                } else {
                    false
                }
            })
            .collect()
    }

    fn all_node_ids(&self) -> Vec<NodeId> {
        if self.node_filter.is_empty() {
            return self.store.all_node_ids();
        }
        // Filter through standard node_ids for projections
        self.node_ids()
    }

    fn nodes_by_label(&self, label: &str) -> Vec<NodeId> {
        // If label filter is set and doesn't include this label, return empty
        if !self.node_filter.labels.is_empty() && !self.node_filter.labels.contains(label) {
            return Vec::new();
        }

        let ids = self.store.nodes_by_label(label);

        if self.node_filter.property_predicates.is_empty() {
            return ids;
        }

        // Apply property predicates
        ids.into_iter()
            .filter(|&id| {
                if let Some(node) = self.store.get_node(id) {
                    self.node_filter.accepts(&node, self.store)
                } else {
                    false
                }
            })
            .collect()
    }

    fn node_count(&self) -> usize {
        if self.node_filter.is_empty() {
            self.store.node_count()
        } else {
            self.node_ids().len()
        }
    }

    fn edge_count(&self) -> usize {
        if self.edge_filter.is_empty() && self.node_filter.is_empty() {
            return self.store.edge_count();
        }
        // Count edges by iterating (expensive but correct)
        let nodes = self.node_ids();
        let mut count = 0;
        for &node in &nodes {
            count += self.edges_from(node, Direction::Outgoing).len();
        }
        count
    }

    // --- Entity metadata ---

    fn edge_type(&self, id: EdgeId) -> Option<ArcStr> {
        // Delegate — the edge filter is applied at traversal time
        self.store.edge_type(id)
    }

    fn edge_type_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Option<ArcStr> {
        self.store.edge_type_versioned(id, epoch, transaction_id)
    }

    // --- Index introspection ---

    fn has_property_index(&self, property: &str) -> bool {
        self.store.has_property_index(property)
    }

    // --- Filtered search ---

    fn find_nodes_by_property(&self, property: &str, value: &Value) -> Vec<NodeId> {
        let ids = self.store.find_nodes_by_property(property, value);
        if self.node_filter.is_empty() {
            return ids;
        }
        ids.into_iter().filter(|&id| self.node_passes(id)).collect()
    }

    fn find_nodes_by_properties(&self, conditions: &[(&str, Value)]) -> Vec<NodeId> {
        let ids = self.store.find_nodes_by_properties(conditions);
        if self.node_filter.is_empty() {
            return ids;
        }
        ids.into_iter().filter(|&id| self.node_passes(id)).collect()
    }

    fn find_nodes_in_range(
        &self,
        property: &str,
        min: Option<&Value>,
        max: Option<&Value>,
        min_inclusive: bool,
        max_inclusive: bool,
    ) -> Vec<NodeId> {
        let ids = self
            .store
            .find_nodes_in_range(property, min, max, min_inclusive, max_inclusive);
        if self.node_filter.is_empty() {
            return ids;
        }
        ids.into_iter().filter(|&id| self.node_passes(id)).collect()
    }

    // --- Zone maps ---

    fn node_property_might_match(
        &self,
        property: &PropertyKey,
        op: CompareOp,
        value: &Value,
    ) -> bool {
        self.store.node_property_might_match(property, op, value)
    }

    fn edge_property_might_match(
        &self,
        property: &PropertyKey,
        op: CompareOp,
        value: &Value,
    ) -> bool {
        self.store.edge_property_might_match(property, op, value)
    }

    // --- Statistics ---
    // Delegate to underlying store (approximation acceptable per spec)

    fn statistics(&self) -> Arc<Statistics> {
        self.store.statistics()
    }

    fn estimate_label_cardinality(&self, label: &str) -> f64 {
        self.store.estimate_label_cardinality(label)
    }

    fn estimate_avg_degree(&self, edge_type: &str, outgoing: bool) -> f64 {
        self.store.estimate_avg_degree(edge_type, outgoing)
    }

    // --- Epoch ---

    fn current_epoch(&self) -> EpochId {
        self.store.current_epoch()
    }

    // --- Schema introspection ---

    fn all_labels(&self) -> Vec<String> {
        if self.node_filter.labels.is_empty() {
            self.store.all_labels()
        } else {
            self.node_filter.labels.iter().cloned().collect()
        }
    }

    fn all_edge_types(&self) -> Vec<String> {
        if self.edge_filter.edge_types.is_empty() {
            self.store.all_edge_types()
        } else {
            self.edge_filter.edge_types.iter().cloned().collect()
        }
    }

    fn all_property_keys(&self) -> Vec<String> {
        self.store.all_property_keys()
    }

    // --- Visibility checks ---

    fn is_node_visible_at_epoch(&self, id: NodeId, epoch: EpochId) -> bool {
        if !self.store.is_node_visible_at_epoch(id, epoch) {
            return false;
        }
        self.node_passes(id)
    }

    fn is_node_visible_versioned(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        if !self
            .store
            .is_node_visible_versioned(id, epoch, transaction_id)
        {
            return false;
        }
        self.node_passes(id)
    }

    fn is_edge_visible_at_epoch(&self, id: EdgeId, epoch: EpochId) -> bool {
        if !self.store.is_edge_visible_at_epoch(id, epoch) {
            return false;
        }
        self.edge_passes(id)
    }

    fn is_edge_visible_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        if !self
            .store
            .is_edge_visible_versioned(id, epoch, transaction_id)
        {
            return false;
        }
        self.edge_passes(id)
    }

    fn filter_visible_node_ids(&self, ids: &[NodeId], epoch: EpochId) -> Vec<NodeId> {
        let visible = self.store.filter_visible_node_ids(ids, epoch);
        if self.node_filter.is_empty() {
            return visible;
        }
        visible
            .into_iter()
            .filter(|&id| self.node_passes(id))
            .collect()
    }

    fn filter_visible_node_ids_versioned(
        &self,
        ids: &[NodeId],
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Vec<NodeId> {
        let visible = self
            .store
            .filter_visible_node_ids_versioned(ids, epoch, transaction_id);
        if self.node_filter.is_empty() {
            return visible;
        }
        visible
            .into_iter()
            .filter(|&id| self.node_passes(id))
            .collect()
    }

    // --- History ---

    fn get_node_history(&self, id: NodeId) -> Vec<(EpochId, Option<EpochId>, Node)> {
        if !self.node_passes(id) {
            return Vec::new();
        }
        self.store.get_node_history(id)
    }

    fn get_edge_history(&self, id: EdgeId) -> Vec<(EpochId, Option<EpochId>, Edge)> {
        if !self.edge_passes(id) {
            return Vec::new();
        }
        self.store.get_edge_history(id)
    }
}

// ============================================================================
// ProjectionBuilder
// ============================================================================

/// Fluent builder for constructing [`GraphProjection`] instances.
///
/// Provides a convenient API for defining node and edge filters
/// without manually constructing filter structs.
///
/// # Example
///
/// ```no_run
/// use grafeo_adapters::plugins::algorithms::projection::ProjectionBuilder;
/// use grafeo_core::graph::lpg::LpgStore;
/// use grafeo_core::graph::GraphStore;
/// use grafeo_common::types::Value;
///
/// let store = LpgStore::new().unwrap();
///
/// let projection = ProjectionBuilder::new(&store)
///     .with_node_labels(&["File", "Function"])
///     .with_edge_types(&["IMPORTS", "CALLS"])
///     .with_node_property("active", |v: &Value| matches!(v, Value::Bool(true)))
///     .build();
///
/// assert_eq!(projection.node_ids().len(), 0); // empty store
/// ```
pub struct ProjectionBuilder<'a> {
    store: &'a dyn GraphStore,
    node_filter: NodeFilter,
    edge_filter: EdgeFilter,
}

impl<'a> ProjectionBuilder<'a> {
    /// Creates a new builder targeting the given graph store.
    ///
    /// # Arguments
    ///
    /// * `store` - The graph store to create a projection over
    ///
    /// # Returns
    ///
    /// A new builder with no filters (equivalent to the full graph).
    ///
    /// # Complexity
    ///
    /// O(1)
    ///
    /// # Example
    ///
    /// ```no_run
    /// use grafeo_adapters::plugins::algorithms::projection::ProjectionBuilder;
    /// use grafeo_core::graph::lpg::LpgStore;
    ///
    /// let store = LpgStore::new().unwrap();
    /// let builder = ProjectionBuilder::new(&store);
    /// ```
    pub fn new(store: &'a dyn GraphStore) -> Self {
        Self {
            store,
            node_filter: NodeFilter::new(),
            edge_filter: EdgeFilter::new(),
        }
    }

    /// Restricts nodes to those with at least one of the given labels.
    ///
    /// # Arguments
    ///
    /// * `labels` - Slice of label strings. A node must have at least one.
    ///
    /// # Returns
    ///
    /// `self` for chaining.
    ///
    /// # Complexity
    ///
    /// O(L) where L is the number of labels.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use grafeo_adapters::plugins::algorithms::projection::ProjectionBuilder;
    /// use grafeo_core::graph::lpg::LpgStore;
    ///
    /// let store = LpgStore::new().unwrap();
    /// let builder = ProjectionBuilder::new(&store)
    ///     .with_node_labels(&["Person", "Company"]);
    /// ```
    pub fn with_node_labels(mut self, labels: &[&str]) -> Self {
        self.node_filter.labels = labels.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Restricts edges to those with one of the given types.
    ///
    /// # Arguments
    ///
    /// * `types` - Slice of edge type strings.
    ///
    /// # Returns
    ///
    /// `self` for chaining.
    ///
    /// # Complexity
    ///
    /// O(T) where T is the number of types.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use grafeo_adapters::plugins::algorithms::projection::ProjectionBuilder;
    /// use grafeo_core::graph::lpg::LpgStore;
    ///
    /// let store = LpgStore::new().unwrap();
    /// let builder = ProjectionBuilder::new(&store)
    ///     .with_edge_types(&["KNOWS", "WORKS_AT"]);
    /// ```
    pub fn with_edge_types(mut self, types: &[&str]) -> Self {
        self.edge_filter.edge_types = types.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Adds a property predicate for nodes.
    ///
    /// The predicate must return `true` for the node to be included.
    /// Multiple predicates are ANDed together.
    ///
    /// # Arguments
    ///
    /// * `key` - The property key to test
    /// * `predicate` - A closure testing the property value
    ///
    /// # Returns
    ///
    /// `self` for chaining.
    ///
    /// # Complexity
    ///
    /// O(1) — stores the closure.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use grafeo_adapters::plugins::algorithms::projection::ProjectionBuilder;
    /// use grafeo_core::graph::lpg::LpgStore;
    /// use grafeo_common::types::Value;
    ///
    /// let store = LpgStore::new().unwrap();
    /// let builder = ProjectionBuilder::new(&store)
    ///     .with_node_property("score", |v: &Value| {
    ///         matches!(v, Value::Float64(s) if *s > 0.5)
    ///     });
    /// ```
    pub fn with_node_property<F>(mut self, key: &str, predicate: F) -> Self
    where
        F: Fn(&Value) -> bool + Send + Sync + 'static,
    {
        self.node_filter
            .property_predicates
            .push(PropertyPredicate::new(key, predicate));
        self
    }

    /// Adds a property predicate for edges.
    ///
    /// The predicate must return `true` for the edge to be included.
    /// Multiple predicates are ANDed together.
    ///
    /// # Arguments
    ///
    /// * `key` - The property key to test
    /// * `predicate` - A closure testing the property value
    ///
    /// # Returns
    ///
    /// `self` for chaining.
    ///
    /// # Complexity
    ///
    /// O(1) — stores the closure.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use grafeo_adapters::plugins::algorithms::projection::ProjectionBuilder;
    /// use grafeo_core::graph::lpg::LpgStore;
    /// use grafeo_common::types::Value;
    ///
    /// let store = LpgStore::new().unwrap();
    /// let builder = ProjectionBuilder::new(&store)
    ///     .with_edge_property("weight", |v: &Value| {
    ///         matches!(v, Value::Float64(w) if *w > 0.1)
    ///     });
    /// ```
    pub fn with_edge_property<F>(mut self, key: &str, predicate: F) -> Self
    where
        F: Fn(&Value) -> bool + Send + Sync + 'static,
    {
        self.edge_filter
            .property_predicates
            .push(PropertyPredicate::new(key, predicate));
        self
    }

    /// Builds the projection with the configured filters.
    ///
    /// # Returns
    ///
    /// A [`GraphProjection`] that lazily applies all configured filters.
    ///
    /// # Complexity
    ///
    /// O(1) — no data is pre-computed.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use grafeo_adapters::plugins::algorithms::projection::ProjectionBuilder;
    /// use grafeo_core::graph::lpg::LpgStore;
    /// use grafeo_core::graph::GraphStore;
    ///
    /// let store = LpgStore::new().unwrap();
    /// let projection = ProjectionBuilder::new(&store)
    ///     .with_node_labels(&["Person"])
    ///     .build();
    /// ```
    pub fn build(self) -> GraphProjection<'a> {
        GraphProjection::new(self.store, self.node_filter, self.edge_filter)
    }
}

// ============================================================================
// ProjectionConfig (serializable configuration for named projections)
// ============================================================================

/// Serializable configuration for a named graph projection.
///
/// Unlike [`GraphProjection`] which holds a reference to a store,
/// `ProjectionConfig` stores only the filter **specification** — label
/// names and edge type names — so it can live in a registry without
/// lifetime constraints.
///
/// Property predicates are not supported in named projections because
/// closures cannot be serialized. Use label and type filters instead.
///
/// # Example
///
/// ```no_run
/// use grafeo_adapters::plugins::algorithms::projection::ProjectionConfig;
///
/// let config = ProjectionConfig::new("code_view")
///     .node_labels(vec!["File".into(), "Function".into()])
///     .edge_types(vec!["IMPORTS".into(), "CALLS".into()]);
///
/// assert_eq!(config.name(), "code_view");
/// ```
#[derive(Debug, Clone)]
pub struct ProjectionConfig {
    name: String,
    node_labels: Vec<String>,
    edge_types: Vec<String>,
}

impl ProjectionConfig {
    /// Creates a new named projection configuration.
    ///
    /// # Arguments
    ///
    /// * `name` - The projection name (used as lookup key in the registry)
    ///
    /// # Returns
    ///
    /// A new config with no filters.
    ///
    /// # Complexity
    ///
    /// O(1)
    ///
    /// # Example
    ///
    /// ```no_run
    /// use grafeo_adapters::plugins::algorithms::projection::ProjectionConfig;
    ///
    /// let config = ProjectionConfig::new("my_view");
    /// ```
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            node_labels: Vec::new(),
            edge_types: Vec::new(),
        }
    }

    /// Sets the allowed node labels (builder pattern).
    ///
    /// # Arguments
    ///
    /// * `labels` - Label strings to allow
    ///
    /// # Returns
    ///
    /// `self` for chaining.
    ///
    /// # Complexity
    ///
    /// O(1)
    pub fn node_labels(mut self, labels: Vec<String>) -> Self {
        self.node_labels = labels;
        self
    }

    /// Sets the allowed edge types (builder pattern).
    ///
    /// # Arguments
    ///
    /// * `types` - Edge type strings to allow
    ///
    /// # Returns
    ///
    /// `self` for chaining.
    ///
    /// # Complexity
    ///
    /// O(1)
    pub fn edge_types(mut self, types: Vec<String>) -> Self {
        self.edge_types = types;
        self
    }

    /// Returns the projection name.
    ///
    /// # Returns
    ///
    /// The name string.
    ///
    /// # Complexity
    ///
    /// O(1)
    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the configured node labels.
    ///
    /// # Returns
    ///
    /// Slice of label strings.
    ///
    /// # Complexity
    ///
    /// O(1)
    #[inline]
    pub fn node_labels_ref(&self) -> &[String] {
        &self.node_labels
    }

    /// Returns the configured edge types.
    ///
    /// # Returns
    ///
    /// Slice of edge type strings.
    ///
    /// # Complexity
    ///
    /// O(1)
    #[inline]
    pub fn edge_types_ref(&self) -> &[String] {
        &self.edge_types
    }

    /// Creates a [`GraphProjection`] from this config over the given store.
    ///
    /// # Arguments
    ///
    /// * `store` - The graph store to project over
    ///
    /// # Returns
    ///
    /// A new [`GraphProjection`] with filters derived from this config.
    ///
    /// # Complexity
    ///
    /// O(L + T) where L = label count, T = edge type count.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use grafeo_adapters::plugins::algorithms::projection::ProjectionConfig;
    /// use grafeo_core::graph::lpg::LpgStore;
    /// use grafeo_core::graph::GraphStore;
    ///
    /// let store = LpgStore::new().unwrap();
    /// let config = ProjectionConfig::new("view")
    ///     .node_labels(vec!["Person".into()]);
    /// let projection = config.to_projection(&store);
    /// ```
    pub fn to_projection<'a>(&self, store: &'a dyn GraphStore) -> GraphProjection<'a> {
        let node_filter = if self.node_labels.is_empty() {
            NodeFilter::new()
        } else {
            NodeFilter::new().labels(self.node_labels.clone())
        };

        let edge_filter = if self.edge_types.is_empty() {
            EdgeFilter::new()
        } else {
            EdgeFilter::new().types(self.edge_types.clone())
        };

        GraphProjection::new(store, node_filter, edge_filter)
    }
}

// ============================================================================
// ProjectionRegistry
// ============================================================================

/// Registry of named graph projections.
///
/// Stores [`ProjectionConfig`] entries that can be instantiated on demand
/// over any [`GraphStore`]. Thread-safe via `parking_lot::RwLock`.
///
/// Named projections are created/dropped via:
/// - `CALL grafeo.projection.create('name', {node_labels: [...], edge_types: [...]})`
/// - `CALL grafeo.projection.drop('name')`
/// - `CALL grafeo.projection.list()`
///
/// # Example
///
/// ```no_run
/// use grafeo_adapters::plugins::algorithms::projection::{ProjectionConfig, ProjectionRegistry};
///
/// let registry = ProjectionRegistry::new();
/// let config = ProjectionConfig::new("code_view")
///     .node_labels(vec!["File".into()])
///     .edge_types(vec!["IMPORTS".into()]);
///
/// registry.create(config).unwrap();
/// assert!(registry.get("code_view").is_some());
/// registry.drop_projection("code_view");
/// assert!(registry.get("code_view").is_none());
/// ```
pub struct ProjectionRegistry {
    projections: parking_lot::RwLock<FxHashMap<String, ProjectionConfig>>,
}

impl ProjectionRegistry {
    /// Creates a new empty registry.
    ///
    /// # Returns
    ///
    /// An empty `ProjectionRegistry`.
    ///
    /// # Complexity
    ///
    /// O(1)
    ///
    /// # Example
    ///
    /// ```no_run
    /// use grafeo_adapters::plugins::algorithms::projection::ProjectionRegistry;
    ///
    /// let registry = ProjectionRegistry::new();
    /// ```
    pub fn new() -> Self {
        Self {
            projections: parking_lot::RwLock::new(FxHashMap::default()),
        }
    }

    /// Registers a named projection.
    ///
    /// # Arguments
    ///
    /// * `config` - The projection configuration to register
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, `Err` if a projection with the same name already exists.
    ///
    /// # Complexity
    ///
    /// O(1) amortized (hash map insert).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use grafeo_adapters::plugins::algorithms::projection::{ProjectionConfig, ProjectionRegistry};
    ///
    /// let registry = ProjectionRegistry::new();
    /// registry.create(ProjectionConfig::new("view")).unwrap();
    /// ```
    pub fn create(&self, config: ProjectionConfig) -> std::result::Result<(), String> {
        let mut projections = self.projections.write();
        if projections.contains_key(&config.name) {
            return Err(format!(
                "Projection '{}' already exists. Drop it first.",
                config.name
            ));
        }
        projections.insert(config.name.clone(), config);
        Ok(())
    }

    /// Removes a named projection.
    ///
    /// # Arguments
    ///
    /// * `name` - The projection name to remove
    ///
    /// # Returns
    ///
    /// `true` if the projection existed and was removed.
    ///
    /// # Complexity
    ///
    /// O(1) amortized.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use grafeo_adapters::plugins::algorithms::projection::{ProjectionConfig, ProjectionRegistry};
    ///
    /// let registry = ProjectionRegistry::new();
    /// registry.create(ProjectionConfig::new("view")).unwrap();
    /// assert!(registry.drop_projection("view"));
    /// assert!(!registry.drop_projection("view")); // already gone
    /// ```
    pub fn drop_projection(&self, name: &str) -> bool {
        self.projections.write().remove(name).is_some()
    }

    /// Retrieves a projection configuration by name.
    ///
    /// # Arguments
    ///
    /// * `name` - The projection name to look up
    ///
    /// # Returns
    ///
    /// `Some(config)` if found, `None` otherwise.
    ///
    /// # Complexity
    ///
    /// O(1) amortized.
    pub fn get(&self, name: &str) -> Option<ProjectionConfig> {
        self.projections.read().get(name).cloned()
    }

    /// Lists all registered projection names and their configurations.
    ///
    /// # Returns
    ///
    /// A vector of `(name, config)` pairs, sorted by name.
    ///
    /// # Complexity
    ///
    /// O(N log N) where N is the number of projections (for sorting).
    pub fn list(&self) -> Vec<ProjectionConfig> {
        let projections = self.projections.read();
        let mut result: Vec<ProjectionConfig> = projections.values().cloned().collect();
        result.sort_by(|a, b| a.name.cmp(&b.name));
        result
    }

    /// Returns the number of registered projections.
    ///
    /// # Returns
    ///
    /// The count of registered projections.
    ///
    /// # Complexity
    ///
    /// O(1)
    pub fn len(&self) -> usize {
        self.projections.read().len()
    }

    /// Returns `true` if no projections are registered.
    ///
    /// # Returns
    ///
    /// `true` if the registry is empty.
    ///
    /// # Complexity
    ///
    /// O(1)
    pub fn is_empty(&self) -> bool {
        self.projections.read().is_empty()
    }
}

impl Default for ProjectionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use grafeo_core::graph::lpg::LpgStore;
    /// Creates a test graph with multiple labels and edge types:
    ///
    /// (File:f1) --IMPORTS--> (Function:fn1) --CALLS--> (Function:fn2)
    /// (Note:n1) --SYNAPSE--> (Note:n2)
    /// (File:f1) --CONTAINS--> (Function:fn1)
    fn create_test_graph() -> LpgStore {
        let store = LpgStore::new().unwrap();

        let f1 = store.create_node(&["File"]);
        store.set_node_property(f1, "name", Value::from("main.rs"));
        store.set_node_property(f1, "active", Value::Bool(true));

        let fn1 = store.create_node(&["Function"]);
        store.set_node_property(fn1, "name", Value::from("parse"));
        store.set_node_property(fn1, "active", Value::Bool(true));

        let fn2 = store.create_node(&["Function"]);
        store.set_node_property(fn2, "name", Value::from("compile"));
        store.set_node_property(fn2, "active", Value::Bool(false));

        let n1 = store.create_node(&["Note"]);
        store.set_node_property(n1, "name", Value::from("design"));

        let n2 = store.create_node(&["Note"]);
        store.set_node_property(n2, "name", Value::from("review"));

        store.create_edge(f1, fn1, "IMPORTS");
        store.create_edge(fn1, fn2, "CALLS");
        store.create_edge(n1, n2, "SYNAPSE");
        store.create_edge(f1, fn1, "CONTAINS");

        store
    }

    #[test]
    fn test_empty_filter_returns_all() {
        let store = create_test_graph();
        let projection = ProjectionBuilder::new(&store).build();
        assert_eq!(projection.node_ids().len(), store.node_ids().len());
        assert_eq!(projection.node_count(), store.node_count());
    }

    #[test]
    fn test_node_label_filter() {
        let store = create_test_graph();
        let projection = ProjectionBuilder::new(&store)
            .with_node_labels(&["File", "Function"])
            .build();

        let ids = projection.node_ids();
        assert_eq!(ids.len(), 3); // f1, fn1, fn2

        // Verify all returned nodes have File or Function label
        for id in &ids {
            let node = projection.get_node(*id).unwrap();
            assert!(
                node.labels
                    .iter()
                    .any(|l| l.as_str() == "File" || l.as_str() == "Function"),
                "Node {:?} should have File or Function label",
                id
            );
        }

        // Note nodes should not be visible
        let all_ids = store.node_ids();
        for id in &all_ids {
            if let Some(node) = store.get_node(*id)
                && node.labels.iter().any(|l| l.as_str() == "Note")
            {
                assert!(projection.get_node(*id).is_none());
            }
        }
    }

    #[test]
    fn test_edge_type_filter() {
        let store = create_test_graph();
        let projection = ProjectionBuilder::new(&store)
            .with_edge_types(&["IMPORTS", "CALLS"])
            .build();

        // Get all outgoing edges from all nodes
        let mut edge_count = 0;
        for id in store.node_ids() {
            let edges = projection.edges_from(id, Direction::Outgoing);
            edge_count += edges.len();
        }
        assert_eq!(edge_count, 2); // IMPORTS + CALLS (not SYNAPSE, not CONTAINS)
    }

    #[test]
    fn test_combined_node_and_edge_filter() {
        let store = create_test_graph();
        let projection = ProjectionBuilder::new(&store)
            .with_node_labels(&["Note"])
            .with_edge_types(&["SYNAPSE"])
            .build();

        let ids = projection.node_ids();
        assert_eq!(ids.len(), 2); // n1, n2

        // Should see SYNAPSE edge
        let mut found_synapse = false;
        for &id in &ids {
            for (target, edge_id) in projection.edges_from(id, Direction::Outgoing) {
                if let Some(et) = store.edge_type(edge_id)
                    && et.as_str() == "SYNAPSE"
                {
                    found_synapse = true;
                    assert!(ids.contains(&target));
                }
            }
        }
        assert!(found_synapse, "Should find SYNAPSE edge");
    }

    #[test]
    fn test_node_property_predicate() {
        let store = create_test_graph();
        let projection = ProjectionBuilder::new(&store)
            .with_node_property("active", |v: &Value| matches!(v, Value::Bool(true)))
            .build();

        let ids = projection.node_ids();
        // f1 (active=true), fn1 (active=true). fn2 has active=false, notes have no active prop
        assert_eq!(ids.len(), 2);
    }

    #[test]
    fn test_neighbors_filtered() {
        let store = create_test_graph();
        let f1 = store.node_ids()[0]; // First node (File)

        // Without filter: f1 has 2 outgoing neighbors (fn1 via IMPORTS, fn1 via CONTAINS)
        let all_neighbors: Vec<NodeId> = store.neighbors(f1, Direction::Outgoing).collect();
        assert!(!all_neighbors.is_empty());

        // With edge type filter: only IMPORTS
        let projection = ProjectionBuilder::new(&store)
            .with_edge_types(&["IMPORTS"])
            .build();
        let filtered_neighbors = projection.neighbors(f1, Direction::Outgoing);
        assert_eq!(filtered_neighbors.len(), 1);
    }

    #[test]
    fn test_get_node_filtered_out() {
        let store = create_test_graph();
        let projection = ProjectionBuilder::new(&store)
            .with_node_labels(&["File"])
            .build();

        // Function nodes should not be visible
        for id in store.node_ids() {
            if let Some(node) = store.get_node(id)
                && node.labels.iter().any(|l| l.as_str() == "Function")
            {
                assert!(projection.get_node(id).is_none());
            }
        }
    }

    #[test]
    fn test_node_count_filtered() {
        let store = create_test_graph();
        let projection = ProjectionBuilder::new(&store)
            .with_node_labels(&["Function"])
            .build();
        assert_eq!(projection.node_count(), 2);
    }

    #[test]
    fn test_edge_count_filtered() {
        let store = create_test_graph();
        let projection = ProjectionBuilder::new(&store)
            .with_edge_types(&["CALLS"])
            .build();
        // Only CALLS edges between Function nodes
        assert_eq!(projection.edge_count(), 1);
    }

    #[test]
    fn test_nodes_by_label_respects_filter() {
        let store = create_test_graph();
        let projection = ProjectionBuilder::new(&store)
            .with_node_labels(&["File"])
            .build();

        // Querying for Function label should return empty since it's not in the filter
        assert_eq!(projection.nodes_by_label("Function").len(), 0);
        assert_eq!(projection.nodes_by_label("File").len(), 1);
    }

    #[test]
    fn test_projection_with_property_and_label_filter() {
        let store = create_test_graph();
        let projection = ProjectionBuilder::new(&store)
            .with_node_labels(&["File", "Function"])
            .with_node_property("active", |v: &Value| matches!(v, Value::Bool(true)))
            .build();

        let ids = projection.node_ids();
        // Only f1 and fn1 (both File/Function with active=true)
        assert_eq!(ids.len(), 2);
    }

    #[test]
    fn test_find_nodes_by_property_filtered() {
        let store = create_test_graph();
        let projection = ProjectionBuilder::new(&store)
            .with_node_labels(&["Function"])
            .build();

        // Find active=true nodes, but only Functions should come back
        let found = projection.find_nodes_by_property("active", &Value::Bool(true));
        assert_eq!(found.len(), 1); // Only fn1 (Function + active=true)
    }

    #[test]
    fn test_all_labels_with_filter() {
        let store = create_test_graph();
        let projection = ProjectionBuilder::new(&store)
            .with_node_labels(&["File", "Function"])
            .build();

        let labels = projection.all_labels();
        assert_eq!(labels.len(), 2);
        assert!(labels.contains(&"File".to_string()));
        assert!(labels.contains(&"Function".to_string()));
    }

    #[test]
    fn test_all_edge_types_with_filter() {
        let store = create_test_graph();
        let projection = ProjectionBuilder::new(&store)
            .with_edge_types(&["IMPORTS"])
            .build();

        let types = projection.all_edge_types();
        assert_eq!(types.len(), 1);
        assert!(types.contains(&"IMPORTS".to_string()));
    }

    #[test]
    fn test_get_node_property_filtered() {
        let store = create_test_graph();
        let projection = ProjectionBuilder::new(&store)
            .with_node_labels(&["Function"])
            .build();

        let file_id = store.node_ids()[0]; // File node
        // File node property should not be accessible through projection
        assert!(
            projection
                .get_node_property(file_id, &PropertyKey::from("name"))
                .is_none()
        );
    }

    #[test]
    fn test_degree_filtered() {
        let store = create_test_graph();
        let fn1 = store.node_ids()[1]; // Function node (fn1)

        // fn1 has CALLS outgoing edge
        let projection = ProjectionBuilder::new(&store)
            .with_edge_types(&["CALLS"])
            .build();
        assert_eq!(projection.out_degree(fn1), 1);

        // With SYNAPSE filter, fn1 should have no outgoing edges
        let projection2 = ProjectionBuilder::new(&store)
            .with_edge_types(&["SYNAPSE"])
            .build();
        assert_eq!(projection2.out_degree(fn1), 0);
    }

    #[test]
    fn test_backward_adjacency_delegated() {
        let store = create_test_graph();
        let projection = ProjectionBuilder::new(&store)
            .with_node_labels(&["File"])
            .build();
        assert_eq!(
            projection.has_backward_adjacency(),
            store.has_backward_adjacency()
        );
    }

    #[test]
    fn test_current_epoch_delegated() {
        let store = create_test_graph();
        let projection = ProjectionBuilder::new(&store).build();
        assert_eq!(projection.current_epoch(), store.current_epoch());
    }

    #[test]
    fn test_statistics_delegated() {
        let store = create_test_graph();
        let projection = ProjectionBuilder::new(&store).build();
        // Just verify it doesn't panic — statistics are delegated
        let _stats = projection.statistics();
    }

    #[test]
    fn test_node_filter_is_empty() {
        let filter = NodeFilter::new();
        assert!(filter.is_empty());

        let filter = NodeFilter::new().labels(vec!["A".into()]);
        assert!(!filter.is_empty());
    }

    #[test]
    fn test_edge_filter_is_empty() {
        let filter = EdgeFilter::new();
        assert!(filter.is_empty());

        let filter = EdgeFilter::new().types(vec!["X".into()]);
        assert!(!filter.is_empty());
    }

    #[test]
    fn test_edge_filter_accepts_type_fast_path() {
        let filter = EdgeFilter::new().types(vec!["IMPORTS".into()]);
        assert_eq!(filter.accepts_type("IMPORTS"), Some(true));
        assert_eq!(filter.accepts_type("CALLS"), Some(false));

        // Empty filter accepts everything
        let filter = EdgeFilter::new();
        assert_eq!(filter.accepts_type("ANYTHING"), Some(true));
    }

    #[test]
    fn test_property_predicate() {
        let pred = PropertyPredicate::new(
            "weight",
            |v: &Value| matches!(v, Value::Float64(w) if *w > 0.5),
        );
        assert!(pred.test(&Value::Float64(0.8)));
        assert!(!pred.test(&Value::Float64(0.2)));
        assert!(!pred.test(&Value::Int64(1)));
    }

    #[test]
    fn test_inner_store_accessor() {
        let store = create_test_graph();
        let projection = ProjectionBuilder::new(&store).build();
        assert_eq!(projection.inner_store().node_count(), store.node_count());
    }

    #[test]
    fn test_filter_accessors() {
        let store = create_test_graph();
        let projection = ProjectionBuilder::new(&store)
            .with_node_labels(&["File"])
            .with_edge_types(&["IMPORTS"])
            .build();

        assert!(projection.node_filter().labels.contains("File"));
        assert!(projection.edge_filter().edge_types.contains("IMPORTS"));
    }

    // ========================================================================
    // ProjectionConfig tests
    // ========================================================================

    #[test]
    fn test_projection_config_new() {
        let config = ProjectionConfig::new("test_view");
        assert_eq!(config.name(), "test_view");
        assert!(config.node_labels_ref().is_empty());
        assert!(config.edge_types_ref().is_empty());
    }

    #[test]
    fn test_projection_config_with_labels_and_types() {
        let config = ProjectionConfig::new("code")
            .node_labels(vec!["File".into(), "Function".into()])
            .edge_types(vec!["IMPORTS".into()]);
        assert_eq!(config.node_labels_ref(), &["File", "Function"]);
        assert_eq!(config.edge_types_ref(), &["IMPORTS"]);
    }

    #[test]
    fn test_projection_config_to_projection() {
        let store = create_test_graph();
        let config = ProjectionConfig::new("funcs").node_labels(vec!["Function".into()]);
        let projection = config.to_projection(&store);
        assert_eq!(projection.node_count(), 2);
    }

    // ========================================================================
    // ProjectionRegistry tests
    // ========================================================================

    #[test]
    fn test_registry_create_and_get() {
        let registry = ProjectionRegistry::new();
        let config = ProjectionConfig::new("view1").node_labels(vec!["File".into()]);
        registry.create(config).unwrap();

        let retrieved = registry.get("view1").unwrap();
        assert_eq!(retrieved.name(), "view1");
        assert_eq!(retrieved.node_labels_ref(), &["File"]);
    }

    #[test]
    fn test_registry_duplicate_name_error() {
        let registry = ProjectionRegistry::new();
        registry.create(ProjectionConfig::new("dup")).unwrap();
        let result = registry.create(ProjectionConfig::new("dup"));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("already exists"));
    }

    #[test]
    fn test_registry_drop() {
        let registry = ProjectionRegistry::new();
        registry.create(ProjectionConfig::new("temp")).unwrap();
        assert!(registry.drop_projection("temp"));
        assert!(!registry.drop_projection("temp"));
        assert!(registry.get("temp").is_none());
    }

    #[test]
    fn test_registry_list() {
        let registry = ProjectionRegistry::new();
        registry.create(ProjectionConfig::new("b_view")).unwrap();
        registry.create(ProjectionConfig::new("a_view")).unwrap();
        let list = registry.list();
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].name(), "a_view"); // sorted
        assert_eq!(list[1].name(), "b_view");
    }

    #[test]
    fn test_registry_len_and_is_empty() {
        let registry = ProjectionRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);

        registry.create(ProjectionConfig::new("v")).unwrap();
        assert!(!registry.is_empty());
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn test_registry_get_nonexistent() {
        let registry = ProjectionRegistry::new();
        assert!(registry.get("nope").is_none());
    }

    #[test]
    fn test_config_to_projection_with_edge_types() {
        let store = create_test_graph();
        let config = ProjectionConfig::new("synapse_view")
            .node_labels(vec!["Note".into()])
            .edge_types(vec!["SYNAPSE".into()]);
        let projection = config.to_projection(&store);
        assert_eq!(projection.node_count(), 2);
        assert_eq!(projection.edge_count(), 1);
    }

    // ========================================================================
    // Integration tests with algorithms
    // ========================================================================

    /// Creates a larger test graph for algorithm integration tests:
    ///
    /// Code layer: File --IMPORTS--> Function --CALLS--> Function (4 nodes, 4 edges)
    /// Knowledge layer: Note --SYNAPSE--> Note (3 nodes, 3 edges)
    /// Cross-layer: File --DOCUMENTS--> Note (1 edge)
    fn create_algo_test_graph() -> LpgStore {
        let store = LpgStore::new().unwrap();

        // Code layer
        let f1 = store.create_node(&["File"]);
        let f2 = store.create_node(&["File"]);
        let fn1 = store.create_node(&["Function"]);
        let fn2 = store.create_node(&["Function"]);

        store.create_edge(f1, fn1, "IMPORTS");
        store.create_edge(f2, fn2, "IMPORTS");
        store.create_edge(fn1, fn2, "CALLS");
        store.create_edge(fn2, fn1, "CALLS");

        // Knowledge layer
        let n1 = store.create_node(&["Note"]);
        let n2 = store.create_node(&["Note"]);
        let n3 = store.create_node(&["Note"]);

        store.create_edge(n1, n2, "SYNAPSE");
        store.create_edge(n2, n3, "SYNAPSE");
        store.create_edge(n3, n1, "SYNAPSE");

        // Cross-layer
        store.create_edge(f1, n1, "DOCUMENTS");

        store
    }

    #[test]
    fn test_pagerank_on_projection_vs_full() {
        use crate::plugins::algorithms::pagerank;

        let store = create_algo_test_graph();

        // PageRank on full graph
        let full_scores = pagerank(&store, 0.85, 100, 1e-6);

        // PageRank on code-only projection
        let projection = ProjectionBuilder::new(&store)
            .with_node_labels(&["File", "Function"])
            .with_edge_types(&["IMPORTS", "CALLS"])
            .build();
        let projected_scores = pagerank(&projection, 0.85, 100, 1e-6);

        // Results must be different (different graph structure)
        assert_ne!(full_scores.len(), projected_scores.len());
        assert_eq!(projected_scores.len(), 4); // 2 Files + 2 Functions
        assert_eq!(full_scores.len(), 7); // All 7 nodes

        // Verify projected scores sum ≈ 1.0
        let sum: f64 = projected_scores.values().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "PageRank scores should sum to ~1.0, got {sum}"
        );
    }

    #[test]
    fn test_louvain_on_projection() {
        use crate::plugins::algorithms::louvain;

        let store = create_algo_test_graph();

        // Louvain on knowledge-only projection (Notes with SYNAPSE)
        let projection = ProjectionBuilder::new(&store)
            .with_node_labels(&["Note"])
            .with_edge_types(&["SYNAPSE"])
            .build();

        let result = louvain(&projection, 1.0);
        assert_eq!(result.communities.len(), 3); // 3 Note nodes

        // All Notes should be in the same community (strongly connected triangle)
        let communities: HashSet<u64> = result.communities.values().copied().collect();
        assert_eq!(
            communities.len(),
            1,
            "Triangle of Notes should form one community"
        );
    }

    #[test]
    fn test_louvain_full_vs_projected_differ() {
        use crate::plugins::algorithms::louvain;

        let store = create_algo_test_graph();

        // Louvain on full graph
        let full_result = louvain(&store, 1.0);

        // Louvain on code-only projection
        let projection = ProjectionBuilder::new(&store)
            .with_node_labels(&["File", "Function"])
            .with_edge_types(&["IMPORTS", "CALLS"])
            .build();
        let proj_result = louvain(&projection, 1.0);

        // Different number of nodes in the results
        assert_eq!(full_result.communities.len(), 7);
        assert_eq!(proj_result.communities.len(), 4);
    }

    #[test]
    fn test_hits_on_projection() {
        use crate::plugins::algorithms::hits;

        let store = create_algo_test_graph();

        // HITS on knowledge layer only
        let projection = ProjectionBuilder::new(&store)
            .with_node_labels(&["Note"])
            .with_edge_types(&["SYNAPSE"])
            .build();

        let result = hits(&projection, 100, 1e-6);
        assert_eq!(result.hub_scores.len(), 3);
        assert_eq!(result.authority_scores.len(), 3);

        // All scores should be non-negative
        for score in result.hub_scores.values() {
            assert!(*score >= 0.0);
        }
        for score in result.authority_scores.values() {
            assert!(*score >= 0.0);
        }
    }

    #[test]
    fn test_projection_overhead_minimal() {
        // Benchmark: projection with no filters should have minimal overhead
        let store = LpgStore::new().unwrap();

        // Create a graph with 100 nodes
        let mut nodes = Vec::new();
        for _ in 0..100 {
            nodes.push(store.create_node(&["Node"]));
        }
        // Create edges in a ring
        for i in 0..100 {
            store.create_edge(nodes[i], nodes[(i + 1) % 100], "NEXT");
        }

        // Empty projection (no filters) should return same results
        let projection = ProjectionBuilder::new(&store).build();

        let direct_ids = GraphStore::node_ids(&store);
        let projected_ids = projection.node_ids();
        assert_eq!(direct_ids.len(), projected_ids.len());
        assert_eq!(direct_ids, projected_ids);

        // Neighbors should match
        for &node in &nodes[..10] {
            let direct_n = GraphStore::neighbors(&store, node, Direction::Outgoing);
            let proj_n = projection.neighbors(node, Direction::Outgoing);
            assert_eq!(direct_n, proj_n);
        }
    }

    #[test]
    fn test_projection_get_node_property() {
        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["File"]);
        let n1 = store.create_node(&["Note"]);
        store.set_node_property(n0, "path", Value::from("main.rs"));
        store.set_node_property(n1, "content", Value::from("hello"));

        let projection = ProjectionBuilder::new(&store)
            .with_node_labels(&["File"])
            .build();

        // File node should have property accessible
        let prop = projection.get_node_property(n0, &"path".into());
        assert!(prop.is_some());

        // Note node should be filtered out
        let prop = projection.get_node_property(n1, &"content".into());
        assert!(prop.is_none());
    }

    #[test]
    fn test_projection_get_node_and_edge() {
        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["File"]);
        let n1 = store.create_node(&["Function"]);
        let n2 = store.create_node(&["Note"]);
        let e0 = store.create_edge(n0, n1, "IMPORTS");
        let e1 = store.create_edge(n0, n2, "SYNAPSE");

        let projection = ProjectionBuilder::new(&store)
            .with_node_labels(&["File", "Function"])
            .with_edge_types(&["IMPORTS"])
            .build();

        // get_node: File passes, Note doesn't
        assert!(projection.get_node(n0).is_some());
        assert!(projection.get_node(n1).is_some());
        assert!(projection.get_node(n2).is_none());

        // get_edge: IMPORTS passes, SYNAPSE doesn't
        assert!(projection.get_edge(e0).is_some());
        assert!(projection.get_edge(e1).is_none());
    }

    #[test]
    fn test_projection_node_count_edge_count() {
        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["File"]);
        let n1 = store.create_node(&["File"]);
        let n2 = store.create_node(&["Note"]);
        store.create_edge(n0, n1, "IMPORTS");
        store.create_edge(n0, n2, "SYNAPSE");

        let projection = ProjectionBuilder::new(&store)
            .with_node_labels(&["File"])
            .with_edge_types(&["IMPORTS"])
            .build();

        assert_eq!(projection.node_count(), 2);
        assert_eq!(projection.edge_count(), 1);
    }

    #[test]
    fn test_projection_edges_from_filtered_node() {
        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["Note"]);
        let n1 = store.create_node(&["File"]);
        store.create_edge(n0, n1, "IMPORTS");

        let projection = ProjectionBuilder::new(&store)
            .with_node_labels(&["File"])
            .build();

        // n0 is filtered out — edges_from should return empty
        let edges = projection.edges_from(n0, Direction::Outgoing);
        assert!(edges.is_empty());

        // neighbors from filtered node should be empty
        let neighbors = projection.neighbors(n0, Direction::Both);
        assert!(neighbors.is_empty());
    }

    #[test]
    fn test_projection_out_degree_in_degree() {
        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["File"]);
        let n1 = store.create_node(&["File"]);
        let n2 = store.create_node(&["File"]);
        store.create_edge(n0, n1, "IMPORTS");
        store.create_edge(n0, n2, "IMPORTS");

        let projection = ProjectionBuilder::new(&store)
            .with_node_labels(&["File"])
            .with_edge_types(&["IMPORTS"])
            .build();

        assert_eq!(projection.out_degree(n0), 2);
        assert_eq!(projection.in_degree(n1), 1);
    }

    #[test]
    fn test_projection_all_labels_and_edge_types() {
        let store = LpgStore::new().unwrap();
        store.create_node(&["File"]);
        store.create_node(&["Note"]);

        let projection = ProjectionBuilder::new(&store)
            .with_node_labels(&["File"])
            .with_edge_types(&["IMPORTS"])
            .build();

        let labels = projection.all_labels();
        assert!(labels.contains(&"File".to_string()));
        assert!(!labels.contains(&"Note".to_string()));

        let edge_types = projection.all_edge_types();
        assert!(edge_types.contains(&"IMPORTS".to_string()));
    }

    #[test]
    fn test_projection_nodes_by_label() {
        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["File"]);
        let _n1 = store.create_node(&["Note"]);
        let n2 = store.create_node(&["File"]);

        let projection = ProjectionBuilder::new(&store)
            .with_node_labels(&["File"])
            .build();

        let files = projection.nodes_by_label("File");
        assert_eq!(files.len(), 2);
        assert!(files.contains(&n0));
        assert!(files.contains(&n2));

        // Label not in filter should return empty
        let notes = projection.nodes_by_label("Note");
        assert!(notes.is_empty());
    }

    #[test]
    fn test_projection_all_node_ids() {
        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["File"]);
        let _n1 = store.create_node(&["Note"]);

        let projection = ProjectionBuilder::new(&store)
            .with_node_labels(&["File"])
            .build();

        let all_ids = projection.all_node_ids();
        assert_eq!(all_ids.len(), 1);
        assert!(all_ids.contains(&n0));
    }

    #[test]
    fn test_projection_edge_type() {
        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["File"]);
        let n1 = store.create_node(&["File"]);
        let e0 = store.create_edge(n0, n1, "IMPORTS");

        let projection = ProjectionBuilder::new(&store)
            .with_edge_types(&["IMPORTS"])
            .build();

        let et = projection.edge_type(e0);
        assert_eq!(et.as_deref(), Some("IMPORTS"));
    }

    #[test]
    fn test_projection_get_node_property_batch() {
        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["File"]);
        let n1 = store.create_node(&["Note"]);
        store.set_node_property(n0, "path", Value::from("a.rs"));
        store.set_node_property(n1, "path", Value::from("b.rs"));

        let projection = ProjectionBuilder::new(&store)
            .with_node_labels(&["File"])
            .build();

        let batch = projection.get_node_property_batch(&[n0, n1], &"path".into());
        assert!(batch[0].is_some());
        assert!(batch[1].is_none()); // Note filtered out
    }

    #[test]
    fn test_projection_get_nodes_properties_batch() {
        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["File"]);
        let n1 = store.create_node(&["Note"]);
        store.set_node_property(n0, "path", Value::from("a.rs"));
        store.set_node_property(n1, "content", Value::from("note"));

        let projection = ProjectionBuilder::new(&store)
            .with_node_labels(&["File"])
            .build();

        let batch = projection.get_nodes_properties_batch(&[n0, n1]);
        assert!(!batch[0].is_empty());
        assert!(batch[1].is_empty()); // Note filtered out
    }

    #[test]
    fn test_projection_find_nodes_by_property() {
        let store = LpgStore::new().unwrap();
        let _n0 = store.create_node(&["File"]);
        let n1 = store.create_node(&["File"]);
        store.set_node_property(n1, "path", Value::from("main.rs"));

        let projection = ProjectionBuilder::new(&store)
            .with_node_labels(&["File"])
            .build();

        let found = projection.find_nodes_by_property("path", &Value::from("main.rs"));
        assert!(found.contains(&n1));
    }

    #[test]
    fn test_projection_statistics_and_schema() {
        let store = LpgStore::new().unwrap();
        store.create_node(&["File"]);

        let projection = ProjectionBuilder::new(&store)
            .with_node_labels(&["File"])
            .build();

        // These delegate to underlying store
        let _stats = projection.statistics();
        let _card = projection.estimate_label_cardinality("File");
        let _keys = projection.all_property_keys();
        let _has_idx = projection.has_property_index("path");
        let _has_back = projection.has_backward_adjacency();
        let _epoch = projection.current_epoch();
    }

    #[test]
    fn test_projection_visibility_checks() {
        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["File"]);
        let n1 = store.create_node(&["Note"]);
        let e0 = store.create_edge(n0, n1, "SYNAPSE");

        let projection = ProjectionBuilder::new(&store)
            .with_node_labels(&["File"])
            .build();

        let epoch = projection.current_epoch();
        assert!(projection.is_node_visible_at_epoch(n0, epoch));
        // Note is filtered out
        assert!(!projection.is_node_visible_at_epoch(n1, epoch));
        // Edge to Note is filtered out
        assert!(!projection.is_edge_visible_at_epoch(e0, epoch));
    }

    #[test]
    fn test_projection_filter_visible_node_ids() {
        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["File"]);
        let n1 = store.create_node(&["Note"]);

        let projection = ProjectionBuilder::new(&store)
            .with_node_labels(&["File"])
            .build();

        let epoch = projection.current_epoch();
        let visible = projection.filter_visible_node_ids(&[n0, n1], epoch);
        assert!(visible.contains(&n0));
        assert!(!visible.contains(&n1));
    }

    #[test]
    fn test_projection_get_node_history() {
        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["File"]);
        let n1 = store.create_node(&["Note"]);

        let projection = ProjectionBuilder::new(&store)
            .with_node_labels(&["File"])
            .build();

        let history = projection.get_node_history(n0);
        assert!(!history.is_empty());

        let history = projection.get_node_history(n1);
        assert!(history.is_empty()); // Note filtered out
    }

    #[test]
    fn test_projection_get_edge_history() {
        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["File"]);
        let n1 = store.create_node(&["File"]);
        let n2 = store.create_node(&["Note"]);
        let e0 = store.create_edge(n0, n1, "IMPORTS");
        let e1 = store.create_edge(n0, n2, "SYNAPSE");

        let projection = ProjectionBuilder::new(&store)
            .with_node_labels(&["File"])
            .with_edge_types(&["IMPORTS"])
            .build();

        let history = projection.get_edge_history(e0);
        assert!(!history.is_empty());

        // Edge to Note with SYNAPSE type should be filtered
        let history = projection.get_edge_history(e1);
        assert!(history.is_empty());
    }

    #[test]
    fn test_projection_get_edge_property() {
        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["File"]);
        let n1 = store.create_node(&["File"]);
        let e0 = store.create_edge(n0, n1, "IMPORTS");
        store.set_edge_property(e0, "weight", Value::Float64(1.5));

        let projection = ProjectionBuilder::new(&store)
            .with_edge_types(&["IMPORTS"])
            .build();

        let prop = projection.get_edge_property(e0, &"weight".into());
        assert_eq!(prop, Some(Value::Float64(1.5)));
    }

    #[test]
    fn test_projection_node_property_predicates() {
        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["File"]);
        let n1 = store.create_node(&["File"]);
        store.set_node_property(n0, "active", Value::Bool(true));
        store.set_node_property(n1, "active", Value::Bool(false));

        let projection = ProjectionBuilder::new(&store)
            .with_node_labels(&["File"])
            .with_node_property("active", |v: &Value| matches!(v, Value::Bool(true)))
            .build();

        let ids = projection.node_ids();
        assert_eq!(ids.len(), 1);
        assert!(ids.contains(&n0));
    }

    #[test]
    fn test_projection_nodes_by_label_with_property_predicates() {
        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["File"]);
        let n1 = store.create_node(&["File"]);
        store.set_node_property(n0, "size", Value::Int64(100));
        store.set_node_property(n1, "size", Value::Int64(5));

        let projection = ProjectionBuilder::new(&store)
            .with_node_labels(&["File"])
            .with_node_property("size", |v: &Value| matches!(v, Value::Int64(x) if *x > 50))
            .build();

        let files = projection.nodes_by_label("File");
        assert_eq!(files.len(), 1);
        assert!(files.contains(&n0));
    }

    #[test]
    fn test_projection_get_nodes_properties_selective_batch() {
        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["File"]);
        let n1 = store.create_node(&["Note"]);
        store.set_node_property(n0, "path", Value::from("a.rs"));
        store.set_node_property(n0, "size", Value::Int64(42));
        store.set_node_property(n1, "content", Value::from("note"));

        let projection = ProjectionBuilder::new(&store)
            .with_node_labels(&["File"])
            .build();

        let keys = vec!["path".into(), "size".into()];
        let batch = projection.get_nodes_properties_selective_batch(&[n0, n1], &keys);
        assert!(!batch[0].is_empty());
        assert!(batch[1].is_empty()); // Note filtered
    }

    #[test]
    fn test_projection_get_edges_properties_selective_batch() {
        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["File"]);
        let n1 = store.create_node(&["File"]);
        let e0 = store.create_edge(n0, n1, "IMPORTS");
        let e1 = store.create_edge(n0, n1, "SYNAPSE");
        store.set_edge_property(e0, "weight", Value::Float64(1.0));
        store.set_edge_property(e1, "weight", Value::Float64(2.0));

        let projection = ProjectionBuilder::new(&store)
            .with_edge_types(&["IMPORTS"])
            .build();

        let keys = vec!["weight".into()];
        let batch = projection.get_edges_properties_selective_batch(&[e0, e1], &keys);
        assert!(!batch[0].is_empty());
        assert!(batch[1].is_empty()); // SYNAPSE filtered
    }

    #[test]
    fn test_projection_find_nodes_by_properties() {
        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["File"]);
        let _n1 = store.create_node(&["Note"]);
        store.set_node_property(n0, "lang", Value::from("rust"));

        let projection = ProjectionBuilder::new(&store)
            .with_node_labels(&["File"])
            .build();

        let found = projection.find_nodes_by_properties(&[("lang", Value::from("rust"))]);
        assert!(found.contains(&n0));
    }

    #[test]
    fn test_projection_find_nodes_in_range() {
        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["File"]);
        let n1 = store.create_node(&["Note"]);
        store.set_node_property(n0, "size", Value::Int64(50));
        store.set_node_property(n1, "size", Value::Int64(100));

        let projection = ProjectionBuilder::new(&store)
            .with_node_labels(&["File"])
            .build();

        let min_val = Value::Int64(10);
        let max_val = Value::Int64(200);
        let found =
            projection.find_nodes_in_range("size", Some(&min_val), Some(&max_val), true, true);
        assert!(found.contains(&n0));
        assert!(!found.contains(&n1)); // Note filtered
    }

    #[test]
    fn test_projection_zone_map_delegates() {
        use grafeo_core::graph::lpg::CompareOp;

        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["File"]);
        store.set_node_property(n0, "size", Value::Int64(42));

        let projection = ProjectionBuilder::new(&store)
            .with_node_labels(&["File"])
            .build();

        let _might_match_node =
            projection.node_property_might_match(&"size".into(), CompareOp::Eq, &Value::Int64(42));
        let _might_match_edge = projection.edge_property_might_match(
            &"weight".into(),
            CompareOp::Gt,
            &Value::Float64(0.0),
        );
    }

    #[test]
    fn test_projection_estimate_avg_degree() {
        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["File"]);
        let n1 = store.create_node(&["File"]);
        store.create_edge(n0, n1, "IMPORTS");

        let projection = ProjectionBuilder::new(&store)
            .with_edge_types(&["IMPORTS"])
            .build();

        let _avg = projection.estimate_avg_degree("IMPORTS", true);
    }

    #[test]
    fn test_projection_edge_only_filter() {
        // Node filter empty, edge filter set
        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["File"]);
        let n1 = store.create_node(&["File"]);
        store.create_edge(n0, n1, "IMPORTS");
        store.create_edge(n0, n1, "CALLS");

        let projection = ProjectionBuilder::new(&store)
            .with_edge_types(&["IMPORTS"])
            .build();

        let neighbors = projection.neighbors(n0, Direction::Outgoing);
        assert_eq!(neighbors.len(), 1);
    }

    #[test]
    fn test_projection_node_only_filter() {
        // Node filter set, edge filter empty
        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["File"]);
        let n1 = store.create_node(&["Note"]);
        store.create_edge(n0, n1, "IMPORTS");

        let projection = ProjectionBuilder::new(&store)
            .with_node_labels(&["File"])
            .build();

        // n1 is filtered — neighbor should be empty
        let neighbors = projection.neighbors(n0, Direction::Outgoing);
        assert!(neighbors.is_empty());
    }

    #[test]
    fn test_projection_versioned_methods() {
        use grafeo_common::types::{EpochId, TransactionId};

        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["File"]);
        let n1 = store.create_node(&["Note"]);
        let e0 = store.create_edge(n0, n1, "SYNAPSE");

        let projection = ProjectionBuilder::new(&store)
            .with_node_labels(&["File"])
            .build();

        let epoch = EpochId(0);
        let txn = TransactionId(0);

        // Versioned get_node
        let node = projection.get_node_versioned(n0, epoch, txn);
        assert!(node.is_some());
        let node = projection.get_node_versioned(n1, epoch, txn);
        assert!(node.is_none()); // Note filtered

        // Versioned get_edge
        let edge = projection.get_edge_versioned(e0, epoch, txn);
        assert!(edge.is_none()); // dst is Note, filtered

        // get_node_at_epoch
        let node = projection.get_node_at_epoch(n0, epoch);
        assert!(node.is_some());
        let node = projection.get_node_at_epoch(n1, epoch);
        assert!(node.is_none());

        // get_edge_at_epoch
        let edge = projection.get_edge_at_epoch(e0, epoch);
        assert!(edge.is_none()); // dst is Note

        // edge_type_versioned
        let _et = projection.edge_type_versioned(e0, epoch, txn);

        // is_node_visible_versioned
        assert!(projection.is_node_visible_versioned(n0, epoch, txn));
        assert!(!projection.is_node_visible_versioned(n1, epoch, txn));

        // is_edge_visible_versioned
        assert!(!projection.is_edge_visible_versioned(e0, epoch, txn));

        // filter_visible_node_ids_versioned
        let visible = projection.filter_visible_node_ids_versioned(&[n0, n1], epoch, txn);
        assert!(visible.contains(&n0));
        assert!(!visible.contains(&n1));
    }
}
