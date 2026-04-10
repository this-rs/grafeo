//! [`InstrumentedStore`] — a transparent wrapper around any [`GraphStoreMut`]
//! that records mutations for later publication via the [`MutationBus`].
//!
//! The store does NOT publish events immediately. Instead, it accumulates
//! [`MutationEvent`]s in a buffer. The commit hook calls
//! [`drain_pending`](InstrumentedStore::drain_pending) to collect all buffered
//! events and publish them as a single [`MutationBatch`].

use crate::event::{EdgeSnapshot, EventContext, MutationEvent, NodeSnapshot};
use arcstr::ArcStr;
use obrain_common::types::{EdgeId, EpochId, NodeId, PropertyKey, TransactionId, Value};
use obrain_common::utils::hash::FxHashMap;
use obrain_core::graph::Direction;
use obrain_core::graph::lpg::{CompareOp, Edge, Node};
use obrain_core::graph::traits::{GraphStore, GraphStoreMut};
use obrain_core::statistics::Statistics;
use parking_lot::Mutex;
use std::sync::Arc;

/// A wrapper around a [`GraphStoreMut`] that intercepts write operations
/// and records mutation events for the reactive substrate.
///
/// The inner store is fully delegated to — `InstrumentedStore` adds zero
/// overhead to read operations and only a snapshot capture on writes.
pub struct InstrumentedStore<S: GraphStoreMut> {
    /// The wrapped store.
    inner: S,
    /// Pending mutation events accumulated during the current transaction.
    pending: Mutex<Vec<MutationEvent>>,
    /// Optional event context for the current session/tenant.
    event_context: Mutex<Option<Arc<EventContext>>>,
}

impl<S: GraphStoreMut> InstrumentedStore<S> {
    /// Wraps an existing store with mutation tracking.
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            pending: Mutex::new(Vec::new()),
            event_context: Mutex::new(None),
        }
    }

    /// Returns a reference to the inner store.
    pub fn inner(&self) -> &S {
        &self.inner
    }

    /// Returns a mutable reference to the inner store.
    pub fn inner_mut(&mut self) -> &mut S {
        &mut self.inner
    }

    /// Drains all pending mutation events (called by the commit hook).
    pub fn drain_pending(&self) -> Vec<MutationEvent> {
        std::mem::take(&mut *self.pending.lock())
    }

    /// Clears pending events without returning them (called on rollback).
    pub fn clear_pending(&self) {
        self.pending.lock().clear();
    }

    /// Returns the number of pending events.
    pub fn pending_count(&self) -> usize {
        self.pending.lock().len()
    }

    /// Sets the event context for this store's pending mutations.
    ///
    /// When [`drain_pending`](Self::drain_pending) is called, the caller can
    /// use this context to create a [`MutationBatch::with_context`].
    pub fn set_event_context(&self, context: Arc<EventContext>) {
        *self.event_context.lock() = Some(context);
    }

    /// Clears the event context.
    pub fn clear_event_context(&self) {
        *self.event_context.lock() = None;
    }

    /// Returns the current event context, if any.
    pub fn event_context(&self) -> Option<Arc<EventContext>> {
        self.event_context.lock().clone()
    }

    fn snapshot_node(node: &Node) -> NodeSnapshot {
        NodeSnapshot {
            id: node.id,
            labels: node.labels.clone(),
            properties: node
                .properties
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        }
    }

    fn snapshot_edge(edge: &Edge) -> EdgeSnapshot {
        EdgeSnapshot {
            id: edge.id,
            src: edge.src,
            dst: edge.dst,
            edge_type: edge.edge_type.clone(),
            properties: edge
                .properties
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        }
    }

    fn push_event(&self, event: MutationEvent) {
        self.pending.lock().push(event);
    }
}

// --- GraphStore delegation (read-only, zero overhead) ---
// We delegate every method to inner, preserving the exact signatures.

impl<S: GraphStoreMut> GraphStore for InstrumentedStore<S> {
    fn get_node(&self, id: NodeId) -> Option<Node> {
        self.inner.get_node(id)
    }

    fn get_edge(&self, id: EdgeId) -> Option<Edge> {
        self.inner.get_edge(id)
    }

    fn get_node_versioned(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Option<Node> {
        self.inner.get_node_versioned(id, epoch, transaction_id)
    }

    fn get_edge_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Option<Edge> {
        self.inner.get_edge_versioned(id, epoch, transaction_id)
    }

    fn get_node_at_epoch(&self, id: NodeId, epoch: EpochId) -> Option<Node> {
        self.inner.get_node_at_epoch(id, epoch)
    }

    fn get_edge_at_epoch(&self, id: EdgeId, epoch: EpochId) -> Option<Edge> {
        self.inner.get_edge_at_epoch(id, epoch)
    }

    fn get_node_property(&self, id: NodeId, key: &PropertyKey) -> Option<Value> {
        self.inner.get_node_property(id, key)
    }

    fn get_edge_property(&self, id: EdgeId, key: &PropertyKey) -> Option<Value> {
        self.inner.get_edge_property(id, key)
    }

    fn get_node_property_batch(&self, ids: &[NodeId], key: &PropertyKey) -> Vec<Option<Value>> {
        self.inner.get_node_property_batch(ids, key)
    }

    fn get_nodes_properties_batch(&self, ids: &[NodeId]) -> Vec<FxHashMap<PropertyKey, Value>> {
        self.inner.get_nodes_properties_batch(ids)
    }

    fn get_nodes_properties_selective_batch(
        &self,
        ids: &[NodeId],
        keys: &[PropertyKey],
    ) -> Vec<FxHashMap<PropertyKey, Value>> {
        self.inner.get_nodes_properties_selective_batch(ids, keys)
    }

    fn get_edges_properties_selective_batch(
        &self,
        ids: &[EdgeId],
        keys: &[PropertyKey],
    ) -> Vec<FxHashMap<PropertyKey, Value>> {
        self.inner.get_edges_properties_selective_batch(ids, keys)
    }

    fn neighbors(&self, node: NodeId, direction: Direction) -> Vec<NodeId> {
        self.inner.neighbors(node, direction)
    }

    fn edges_from(&self, node: NodeId, direction: Direction) -> Vec<(NodeId, EdgeId)> {
        self.inner.edges_from(node, direction)
    }

    fn out_degree(&self, node: NodeId) -> usize {
        self.inner.out_degree(node)
    }

    fn in_degree(&self, node: NodeId) -> usize {
        self.inner.in_degree(node)
    }

    fn has_backward_adjacency(&self) -> bool {
        self.inner.has_backward_adjacency()
    }

    fn node_ids(&self) -> Vec<NodeId> {
        self.inner.node_ids()
    }

    fn all_node_ids(&self) -> Vec<NodeId> {
        self.inner.all_node_ids()
    }

    fn nodes_by_label(&self, label: &str) -> Vec<NodeId> {
        self.inner.nodes_by_label(label)
    }

    fn node_count(&self) -> usize {
        self.inner.node_count()
    }

    fn edge_count(&self) -> usize {
        self.inner.edge_count()
    }

    fn edge_type(&self, id: EdgeId) -> Option<ArcStr> {
        self.inner.edge_type(id)
    }

    fn edge_type_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Option<ArcStr> {
        self.inner.edge_type_versioned(id, epoch, transaction_id)
    }

    fn has_property_index(&self, property: &str) -> bool {
        self.inner.has_property_index(property)
    }

    fn find_nodes_by_property(&self, property: &str, value: &Value) -> Vec<NodeId> {
        self.inner.find_nodes_by_property(property, value)
    }

    fn find_nodes_by_properties(&self, conditions: &[(&str, Value)]) -> Vec<NodeId> {
        self.inner.find_nodes_by_properties(conditions)
    }

    fn find_nodes_in_range(
        &self,
        property: &str,
        min: Option<&Value>,
        max: Option<&Value>,
        min_inclusive: bool,
        max_inclusive: bool,
    ) -> Vec<NodeId> {
        self.inner
            .find_nodes_in_range(property, min, max, min_inclusive, max_inclusive)
    }

    fn node_property_might_match(
        &self,
        property: &PropertyKey,
        op: CompareOp,
        value: &Value,
    ) -> bool {
        self.inner.node_property_might_match(property, op, value)
    }

    fn edge_property_might_match(
        &self,
        property: &PropertyKey,
        op: CompareOp,
        value: &Value,
    ) -> bool {
        self.inner.edge_property_might_match(property, op, value)
    }

    fn statistics(&self) -> Arc<Statistics> {
        self.inner.statistics()
    }

    fn estimate_label_cardinality(&self, label: &str) -> f64 {
        self.inner.estimate_label_cardinality(label)
    }

    fn estimate_avg_degree(&self, edge_type: &str, outgoing: bool) -> f64 {
        self.inner.estimate_avg_degree(edge_type, outgoing)
    }

    fn current_epoch(&self) -> EpochId {
        self.inner.current_epoch()
    }

    fn all_labels(&self) -> Vec<String> {
        self.inner.all_labels()
    }

    fn all_edge_types(&self) -> Vec<String> {
        self.inner.all_edge_types()
    }

    fn all_property_keys(&self) -> Vec<String> {
        self.inner.all_property_keys()
    }

    fn is_node_visible_at_epoch(&self, id: NodeId, epoch: EpochId) -> bool {
        self.inner.is_node_visible_at_epoch(id, epoch)
    }

    fn is_node_visible_versioned(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        self.inner
            .is_node_visible_versioned(id, epoch, transaction_id)
    }

    fn is_edge_visible_at_epoch(&self, id: EdgeId, epoch: EpochId) -> bool {
        self.inner.is_edge_visible_at_epoch(id, epoch)
    }

    fn is_edge_visible_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        self.inner
            .is_edge_visible_versioned(id, epoch, transaction_id)
    }

    fn filter_visible_node_ids(&self, ids: &[NodeId], epoch: EpochId) -> Vec<NodeId> {
        self.inner.filter_visible_node_ids(ids, epoch)
    }

    fn filter_visible_node_ids_versioned(
        &self,
        ids: &[NodeId],
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Vec<NodeId> {
        self.inner
            .filter_visible_node_ids_versioned(ids, epoch, transaction_id)
    }

    fn get_node_history(&self, id: NodeId) -> Vec<(EpochId, Option<EpochId>, Node)> {
        self.inner.get_node_history(id)
    }

    fn get_edge_history(&self, id: EdgeId) -> Vec<(EpochId, Option<EpochId>, Edge)> {
        self.inner.get_edge_history(id)
    }
}

// --- GraphStoreMut: intercept writes and record events ---

impl<S: GraphStoreMut> GraphStoreMut for InstrumentedStore<S> {
    fn create_node(&self, labels: &[&str]) -> NodeId {
        let id = self.inner.create_node(labels);
        if let Some(node) = self.inner.get_node(id) {
            self.push_event(MutationEvent::NodeCreated {
                node: Self::snapshot_node(&node),
            });
        }
        id
    }

    fn create_node_versioned(
        &self,
        labels: &[&str],
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> NodeId {
        let id = self
            .inner
            .create_node_versioned(labels, epoch, transaction_id);
        if let Some(node) = self.inner.get_node_versioned(id, epoch, transaction_id) {
            self.push_event(MutationEvent::NodeCreated {
                node: Self::snapshot_node(&node),
            });
        }
        id
    }

    fn create_edge(&self, src: NodeId, dst: NodeId, edge_type: &str) -> EdgeId {
        let id = self.inner.create_edge(src, dst, edge_type);
        if let Some(edge) = self.inner.get_edge(id) {
            self.push_event(MutationEvent::EdgeCreated {
                edge: Self::snapshot_edge(&edge),
            });
        }
        id
    }

    fn create_edge_versioned(
        &self,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> EdgeId {
        let id = self
            .inner
            .create_edge_versioned(src, dst, edge_type, epoch, transaction_id);
        if let Some(edge) = self.inner.get_edge_versioned(id, epoch, transaction_id) {
            self.push_event(MutationEvent::EdgeCreated {
                edge: Self::snapshot_edge(&edge),
            });
        }
        id
    }

    fn batch_create_edges(&self, edges: &[(NodeId, NodeId, &str)]) -> Vec<EdgeId> {
        let ids = self.inner.batch_create_edges(edges);
        for &id in &ids {
            if let Some(edge) = self.inner.get_edge(id) {
                self.push_event(MutationEvent::EdgeCreated {
                    edge: Self::snapshot_edge(&edge),
                });
            }
        }
        ids
    }

    fn delete_node(&self, id: NodeId) -> bool {
        let before = self.inner.get_node(id).map(|n| Self::snapshot_node(&n));
        let deleted = self.inner.delete_node(id);
        if deleted && let Some(node) = before {
            self.push_event(MutationEvent::NodeDeleted { node });
        }
        deleted
    }

    fn delete_node_versioned(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        let before = self
            .inner
            .get_node_versioned(id, epoch, transaction_id)
            .map(|n| Self::snapshot_node(&n));
        let deleted = self.inner.delete_node_versioned(id, epoch, transaction_id);
        if deleted && let Some(node) = before {
            self.push_event(MutationEvent::NodeDeleted { node });
        }
        deleted
    }

    fn delete_node_edges(&self, node_id: NodeId) {
        // Capture edges before deletion
        let edges_before: Vec<_> = self
            .inner
            .edges_from(node_id, Direction::Both)
            .iter()
            .filter_map(|(_target, eid)| self.inner.get_edge(*eid).map(|e| Self::snapshot_edge(&e)))
            .collect();

        self.inner.delete_node_edges(node_id);

        for edge in edges_before {
            self.push_event(MutationEvent::EdgeDeleted { edge });
        }
    }

    fn delete_edge(&self, id: EdgeId) -> bool {
        let before = self.inner.get_edge(id).map(|e| Self::snapshot_edge(&e));
        let deleted = self.inner.delete_edge(id);
        if deleted && let Some(edge) = before {
            self.push_event(MutationEvent::EdgeDeleted { edge });
        }
        deleted
    }

    fn delete_edge_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        let before = self
            .inner
            .get_edge_versioned(id, epoch, transaction_id)
            .map(|e| Self::snapshot_edge(&e));
        let deleted = self.inner.delete_edge_versioned(id, epoch, transaction_id);
        if deleted && let Some(edge) = before {
            self.push_event(MutationEvent::EdgeDeleted { edge });
        }
        deleted
    }

    fn set_node_property(&self, id: NodeId, key: &str, value: Value) {
        let before = self.inner.get_node(id).map(|n| Self::snapshot_node(&n));
        self.inner.set_node_property(id, key, value);
        let after = self.inner.get_node(id).map(|n| Self::snapshot_node(&n));
        if let (Some(before), Some(after)) = (before, after) {
            self.push_event(MutationEvent::NodeUpdated { before, after });
        }
    }

    fn set_edge_property(&self, id: EdgeId, key: &str, value: Value) {
        let before = self.inner.get_edge(id).map(|e| Self::snapshot_edge(&e));
        self.inner.set_edge_property(id, key, value);
        let after = self.inner.get_edge(id).map(|e| Self::snapshot_edge(&e));
        if let (Some(before), Some(after)) = (before, after) {
            self.push_event(MutationEvent::EdgeUpdated { before, after });
        }
    }

    fn set_node_property_versioned(
        &self,
        id: NodeId,
        key: &str,
        value: Value,
        transaction_id: TransactionId,
    ) {
        let before = self.inner.get_node(id).map(|n| Self::snapshot_node(&n));
        self.inner
            .set_node_property_versioned(id, key, value, transaction_id);
        let after = self.inner.get_node(id).map(|n| Self::snapshot_node(&n));
        if let (Some(before), Some(after)) = (before, after) {
            self.push_event(MutationEvent::NodeUpdated { before, after });
        }
    }

    fn set_edge_property_versioned(
        &self,
        id: EdgeId,
        key: &str,
        value: Value,
        transaction_id: TransactionId,
    ) {
        let before = self.inner.get_edge(id).map(|e| Self::snapshot_edge(&e));
        self.inner
            .set_edge_property_versioned(id, key, value, transaction_id);
        let after = self.inner.get_edge(id).map(|e| Self::snapshot_edge(&e));
        if let (Some(before), Some(after)) = (before, after) {
            self.push_event(MutationEvent::EdgeUpdated { before, after });
        }
    }

    fn remove_node_property(&self, id: NodeId, key: &str) -> Option<Value> {
        let before = self.inner.get_node(id).map(|n| Self::snapshot_node(&n));
        let result = self.inner.remove_node_property(id, key);
        if result.is_some() {
            let after = self.inner.get_node(id).map(|n| Self::snapshot_node(&n));
            if let (Some(before), Some(after)) = (before, after) {
                self.push_event(MutationEvent::NodeUpdated { before, after });
            }
        }
        result
    }

    fn remove_edge_property(&self, id: EdgeId, key: &str) -> Option<Value> {
        let before = self.inner.get_edge(id).map(|e| Self::snapshot_edge(&e));
        let result = self.inner.remove_edge_property(id, key);
        if result.is_some() {
            let after = self.inner.get_edge(id).map(|e| Self::snapshot_edge(&e));
            if let (Some(before), Some(after)) = (before, after) {
                self.push_event(MutationEvent::EdgeUpdated { before, after });
            }
        }
        result
    }

    fn remove_node_property_versioned(
        &self,
        id: NodeId,
        key: &str,
        transaction_id: TransactionId,
    ) -> Option<Value> {
        let before = self.inner.get_node(id).map(|n| Self::snapshot_node(&n));
        let result = self
            .inner
            .remove_node_property_versioned(id, key, transaction_id);
        if result.is_some() {
            let after = self.inner.get_node(id).map(|n| Self::snapshot_node(&n));
            if let (Some(before), Some(after)) = (before, after) {
                self.push_event(MutationEvent::NodeUpdated { before, after });
            }
        }
        result
    }

    fn remove_edge_property_versioned(
        &self,
        id: EdgeId,
        key: &str,
        transaction_id: TransactionId,
    ) -> Option<Value> {
        let before = self.inner.get_edge(id).map(|e| Self::snapshot_edge(&e));
        let result = self
            .inner
            .remove_edge_property_versioned(id, key, transaction_id);
        if result.is_some() {
            let after = self.inner.get_edge(id).map(|e| Self::snapshot_edge(&e));
            if let (Some(before), Some(after)) = (before, after) {
                self.push_event(MutationEvent::EdgeUpdated { before, after });
            }
        }
        result
    }

    fn add_label(&self, node_id: NodeId, label: &str) -> bool {
        let before = self
            .inner
            .get_node(node_id)
            .map(|n| Self::snapshot_node(&n));
        let result = self.inner.add_label(node_id, label);
        if result {
            let after = self
                .inner
                .get_node(node_id)
                .map(|n| Self::snapshot_node(&n));
            if let (Some(before), Some(after)) = (before, after) {
                self.push_event(MutationEvent::NodeUpdated { before, after });
            }
        }
        result
    }

    fn remove_label(&self, node_id: NodeId, label: &str) -> bool {
        let before = self
            .inner
            .get_node(node_id)
            .map(|n| Self::snapshot_node(&n));
        let result = self.inner.remove_label(node_id, label);
        if result {
            let after = self
                .inner
                .get_node(node_id)
                .map(|n| Self::snapshot_node(&n));
            if let (Some(before), Some(after)) = (before, after) {
                self.push_event(MutationEvent::NodeUpdated { before, after });
            }
        }
        result
    }

    fn add_label_versioned(
        &self,
        node_id: NodeId,
        label: &str,
        transaction_id: TransactionId,
    ) -> bool {
        let before = self
            .inner
            .get_node(node_id)
            .map(|n| Self::snapshot_node(&n));
        let result = self
            .inner
            .add_label_versioned(node_id, label, transaction_id);
        if result {
            let after = self
                .inner
                .get_node(node_id)
                .map(|n| Self::snapshot_node(&n));
            if let (Some(before), Some(after)) = (before, after) {
                self.push_event(MutationEvent::NodeUpdated { before, after });
            }
        }
        result
    }

    fn remove_label_versioned(
        &self,
        node_id: NodeId,
        label: &str,
        transaction_id: TransactionId,
    ) -> bool {
        let before = self
            .inner
            .get_node(node_id)
            .map(|n| Self::snapshot_node(&n));
        let result = self
            .inner
            .remove_label_versioned(node_id, label, transaction_id);
        if result {
            let after = self
                .inner
                .get_node(node_id)
                .map(|n| Self::snapshot_node(&n));
            if let (Some(before), Some(after)) = (before, after) {
                self.push_event(MutationEvent::NodeUpdated { before, after });
            }
        }
        result
    }

    fn create_node_with_props(
        &self,
        labels: &[&str],
        properties: &[(PropertyKey, Value)],
    ) -> NodeId {
        let id = self.inner.create_node_with_props(labels, properties);
        if let Some(node) = self.inner.get_node(id) {
            self.push_event(MutationEvent::NodeCreated {
                node: Self::snapshot_node(&node),
            });
        }
        id
    }

    fn create_edge_with_props(
        &self,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
        properties: &[(PropertyKey, Value)],
    ) -> EdgeId {
        let id = self
            .inner
            .create_edge_with_props(src, dst, edge_type, properties);
        if let Some(edge) = self.inner.get_edge(id) {
            self.push_event(MutationEvent::EdgeCreated {
                edge: Self::snapshot_edge(&edge),
            });
        }
        id
    }
}

impl<S: GraphStoreMut + std::fmt::Debug> std::fmt::Debug for InstrumentedStore<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InstrumentedStore")
            .field("inner", &self.inner)
            .field("pending_events", &self.pending_count())
            .finish()
    }
}
