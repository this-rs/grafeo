//! `GraphStore` and `GraphStoreMut` trait implementations for `LpgStore`.
//!
//! Every method here is pure delegation to the existing `LpgStore` method.
//! The only adapters are `neighbors()` and `edges_from()`, which collect
//! the `impl Iterator` return into `Vec` for trait object safety.

use super::LpgStore;
use crate::graph::Direction;
use crate::graph::lpg::CompareOp;
use crate::graph::lpg::{Edge, Node};
use crate::graph::traits::{GraphStore, GraphStoreMut};
use crate::statistics::Statistics;
use arcstr::ArcStr;
use obrain_common::types::{EdgeId, EpochId, NodeId, PropertyKey, TransactionId, Value};
use obrain_common::utils::hash::FxHashMap;
use std::sync::Arc;

impl GraphStore for LpgStore {
    fn get_node(&self, id: NodeId) -> Option<Node> {
        LpgStore::get_node(self, id)
    }

    fn get_edge(&self, id: EdgeId) -> Option<Edge> {
        LpgStore::get_edge(self, id)
    }

    fn get_node_versioned(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Option<Node> {
        LpgStore::get_node_versioned(self, id, epoch, transaction_id)
    }

    fn get_edge_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Option<Edge> {
        LpgStore::get_edge_versioned(self, id, epoch, transaction_id)
    }

    fn get_node_at_epoch(&self, id: NodeId, epoch: EpochId) -> Option<Node> {
        LpgStore::get_node_at_epoch(self, id, epoch)
    }

    fn get_edge_at_epoch(&self, id: EdgeId, epoch: EpochId) -> Option<Edge> {
        LpgStore::get_edge_at_epoch(self, id, epoch)
    }

    fn get_node_property(&self, id: NodeId, key: &PropertyKey) -> Option<Value> {
        LpgStore::get_node_property(self, id, key)
    }

    fn get_edge_property(&self, id: EdgeId, key: &PropertyKey) -> Option<Value> {
        LpgStore::get_edge_property(self, id, key)
    }

    fn get_node_property_batch(&self, ids: &[NodeId], key: &PropertyKey) -> Vec<Option<Value>> {
        LpgStore::get_node_property_batch(self, ids, key)
    }

    fn get_nodes_properties_batch(&self, ids: &[NodeId]) -> Vec<FxHashMap<PropertyKey, Value>> {
        LpgStore::get_nodes_properties_batch(self, ids)
    }

    fn get_nodes_properties_selective_batch(
        &self,
        ids: &[NodeId],
        keys: &[PropertyKey],
    ) -> Vec<FxHashMap<PropertyKey, Value>> {
        LpgStore::get_nodes_properties_selective_batch(self, ids, keys)
    }

    fn get_edges_properties_selective_batch(
        &self,
        ids: &[EdgeId],
        keys: &[PropertyKey],
    ) -> Vec<FxHashMap<PropertyKey, Value>> {
        LpgStore::get_edges_properties_selective_batch(self, ids, keys)
    }

    fn neighbors(&self, node: NodeId, direction: Direction) -> Vec<NodeId> {
        LpgStore::neighbors(self, node, direction).collect()
    }

    fn edges_from(&self, node: NodeId, direction: Direction) -> Vec<(NodeId, EdgeId)> {
        LpgStore::edges_from(self, node, direction).collect()
    }

    fn out_degree(&self, node: NodeId) -> usize {
        LpgStore::out_degree(self, node)
    }

    fn in_degree(&self, node: NodeId) -> usize {
        LpgStore::in_degree(self, node)
    }

    fn has_backward_adjacency(&self) -> bool {
        LpgStore::has_backward_adjacency(self)
    }

    fn node_ids(&self) -> Vec<NodeId> {
        LpgStore::node_ids(self)
    }

    fn all_node_ids(&self) -> Vec<NodeId> {
        LpgStore::all_node_ids(self)
    }

    fn nodes_by_label(&self, label: &str) -> Vec<NodeId> {
        LpgStore::nodes_by_label(self, label)
    }

    fn node_count(&self) -> usize {
        LpgStore::node_count(self)
    }

    fn node_count_by_label(&self, label: &str) -> usize {
        LpgStore::node_count_by_label(self, label)
    }

    fn edge_count(&self) -> usize {
        LpgStore::edge_count(self)
    }

    fn edge_type(&self, id: EdgeId) -> Option<ArcStr> {
        LpgStore::edge_type(self, id)
    }

    fn edge_type_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Option<ArcStr> {
        LpgStore::edge_type_versioned(self, id, epoch, transaction_id)
    }

    fn has_property_index(&self, property: &str) -> bool {
        LpgStore::has_property_index(self, property)
    }

    fn find_nodes_by_property(&self, property: &str, value: &Value) -> Vec<NodeId> {
        LpgStore::find_nodes_by_property(self, property, value)
    }

    fn find_nodes_by_properties(&self, conditions: &[(&str, Value)]) -> Vec<NodeId> {
        LpgStore::find_nodes_by_properties(self, conditions)
    }

    fn find_nodes_in_range(
        &self,
        property: &str,
        min: Option<&Value>,
        max: Option<&Value>,
        min_inclusive: bool,
        max_inclusive: bool,
    ) -> Vec<NodeId> {
        LpgStore::find_nodes_in_range(self, property, min, max, min_inclusive, max_inclusive)
    }

    fn node_property_might_match(
        &self,
        property: &PropertyKey,
        op: CompareOp,
        value: &Value,
    ) -> bool {
        LpgStore::node_property_might_match(self, property, op, value)
    }

    fn edge_property_might_match(
        &self,
        property: &PropertyKey,
        op: CompareOp,
        value: &Value,
    ) -> bool {
        LpgStore::edge_property_might_match(self, property, op, value)
    }

    fn statistics(&self) -> Arc<Statistics> {
        LpgStore::statistics(self)
    }

    fn estimate_label_cardinality(&self, label: &str) -> f64 {
        LpgStore::estimate_label_cardinality(self, label)
    }

    fn estimate_avg_degree(&self, edge_type: &str, outgoing: bool) -> f64 {
        LpgStore::estimate_avg_degree(self, edge_type, outgoing)
    }

    fn current_epoch(&self) -> EpochId {
        LpgStore::current_epoch(self)
    }

    fn all_labels(&self) -> Vec<String> {
        LpgStore::all_labels(self)
    }

    fn all_edge_types(&self) -> Vec<String> {
        LpgStore::all_edge_types(self)
    }

    fn all_property_keys(&self) -> Vec<String> {
        LpgStore::all_property_keys(self)
    }

    fn is_node_visible_at_epoch(&self, id: NodeId, epoch: EpochId) -> bool {
        LpgStore::is_node_visible_at_epoch(self, id, epoch)
    }

    fn is_node_visible_versioned(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        LpgStore::is_node_visible_versioned(self, id, epoch, transaction_id)
    }

    fn is_edge_visible_at_epoch(&self, id: EdgeId, epoch: EpochId) -> bool {
        LpgStore::is_edge_visible_at_epoch(self, id, epoch)
    }

    fn is_edge_visible_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        LpgStore::is_edge_visible_versioned(self, id, epoch, transaction_id)
    }

    fn filter_visible_node_ids(&self, ids: &[NodeId], epoch: EpochId) -> Vec<NodeId> {
        LpgStore::filter_visible_node_ids(self, ids, epoch)
    }

    fn filter_visible_node_ids_versioned(
        &self,
        ids: &[NodeId],
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Vec<NodeId> {
        LpgStore::filter_visible_node_ids_versioned(self, ids, epoch, transaction_id)
    }

    fn get_node_history(&self, id: NodeId) -> Vec<(EpochId, Option<EpochId>, Node)> {
        LpgStore::get_node_history(self, id)
    }

    fn get_edge_history(&self, id: EdgeId) -> Vec<(EpochId, Option<EpochId>, Edge)> {
        LpgStore::get_edge_history(self, id)
    }

    fn find_nodes_by_label_property_contains_bounded(
        &self,
        label: Option<&str>,
        property: &str,
        substring: &str,
        limit: usize,
        max_scan: usize,
    ) -> Vec<NodeId> {
        LpgStore::find_nodes_by_label_property_contains_bounded(
            self, label, property, substring, limit, max_scan,
        )
    }

    fn get_node_labels(&self, id: NodeId) -> Option<Vec<ArcStr>> {
        LpgStore::get_node_labels(self, id)
    }

    #[cfg(feature = "text-index")]
    fn text_index_entries(
        &self,
    ) -> Vec<(
        String,
        Arc<parking_lot::RwLock<crate::index::text::InvertedIndex>>,
    )> {
        LpgStore::text_index_entries(self)
    }

    #[cfg(feature = "vector-index")]
    fn vector_index_entries(
        &self,
    ) -> Vec<(String, Arc<crate::index::vector::HnswIndex>)> {
        LpgStore::vector_index_entries(self)
    }

    #[cfg(feature = "vector-index")]
    fn get_vector_index(
        &self,
        label: &str,
        property: &str,
    ) -> Option<Arc<crate::index::vector::HnswIndex>> {
        LpgStore::get_vector_index(self, label, property)
    }

    #[cfg(feature = "text-index")]
    fn get_text_index(
        &self,
        label: &str,
        property: &str,
    ) -> Option<Arc<parking_lot::RwLock<crate::index::text::InvertedIndex>>> {
        LpgStore::get_text_index(self, label, property)
    }

    #[cfg(feature = "vector-index")]
    fn add_vector_index(
        &self,
        label: &str,
        property: &str,
        index: Arc<crate::index::vector::HnswIndex>,
    ) {
        LpgStore::add_vector_index(self, label, property, index)
    }

    #[cfg(feature = "vector-index")]
    fn remove_vector_index(&self, label: &str, property: &str) -> bool {
        LpgStore::remove_vector_index(self, label, property)
    }

    #[cfg(feature = "text-index")]
    fn add_text_index(
        &self,
        label: &str,
        property: &str,
        index: Arc<parking_lot::RwLock<crate::index::text::InvertedIndex>>,
    ) {
        LpgStore::add_text_index(self, label, property, index)
    }

    #[cfg(feature = "text-index")]
    fn remove_text_index(&self, label: &str, property: &str) -> bool {
        LpgStore::remove_text_index(self, label, property)
    }

    fn find_nodes_matching_filter(
        &self,
        property: &str,
        filter_value: &Value,
    ) -> Vec<NodeId> {
        LpgStore::find_nodes_matching_filter(self, property, filter_value)
    }
}

impl GraphStoreMut for LpgStore {
    // --- T17 Wave B (2026-04-23): MVCC trait hooks ---
    //
    // These delegate to the LpgStore inherent implementations so that
    // callers holding an `Arc<dyn GraphStoreMut>` still get LpgStore's
    // real MVCC semantics when the concrete backend is LpgStore. On
    // substrate these methods fall back to the trait defaults (no-ops)
    // since topology-as-storage has no per-transaction epoch tracking.
    fn finalize_version_epochs(&self, transaction_id: TransactionId, commit_epoch: EpochId) {
        LpgStore::finalize_version_epochs(self, transaction_id, commit_epoch)
    }

    fn commit_transaction_properties(&self, transaction_id: TransactionId) {
        LpgStore::commit_transaction_properties(self, transaction_id)
    }

    fn rollback_transaction_properties(&self, transaction_id: TransactionId) {
        LpgStore::rollback_transaction_properties(self, transaction_id)
    }

    fn rollback_transaction_properties_to(
        &self,
        transaction_id: TransactionId,
        since: usize,
    ) {
        LpgStore::rollback_transaction_properties_to(self, transaction_id, since)
    }

    fn sync_epoch(&self, epoch: EpochId) {
        LpgStore::sync_epoch(self, epoch)
    }

    fn gc_versions(&self, min_epoch: EpochId) {
        LpgStore::gc_versions(self, min_epoch)
    }

    fn peek_next_node_id(&self) -> u64 {
        LpgStore::peek_next_node_id(self)
    }

    fn property_undo_log_position(&self, transaction_id: TransactionId) -> usize {
        LpgStore::property_undo_log_position(self, transaction_id)
    }

    fn discard_uncommitted_versions(&self, transaction_id: TransactionId) {
        LpgStore::discard_uncommitted_versions(self, transaction_id)
    }

    fn peek_next_edge_id(&self) -> u64 {
        LpgStore::peek_next_edge_id(self)
    }

    fn discard_entities_by_id(
        &self,
        transaction_id: TransactionId,
        node_ids: &[NodeId],
        edge_ids: &[EdgeId],
    ) {
        LpgStore::discard_entities_by_id(self, transaction_id, node_ids, edge_ids)
    }

    // --- Named-graph overrides ---
    //
    // LpgStore supports named sub-graphs; substrate does not yet. These
    // overrides keep LpgStore's native API reachable through the trait
    // so Session / other callers can hold `Arc<dyn GraphStoreMut>` and
    // still use `CREATE GRAPH g` / `DROP GRAPH g` / `USE GRAPH g` on
    // the LpgStore code path (legacy-read feature + tests).
    fn named_graph(&self, name: &str) -> Option<std::sync::Arc<dyn GraphStoreMut>> {
        LpgStore::graph(self, name).map(|g| g as std::sync::Arc<dyn GraphStoreMut>)
    }

    fn create_named_graph(
        &self,
        name: &str,
    ) -> Result<bool, obrain_common::memory::arena::AllocError> {
        LpgStore::create_graph(self, name)
    }

    fn drop_named_graph(&self, name: &str) -> bool {
        LpgStore::drop_graph(self, name)
    }

    fn named_graph_names(&self) -> Vec<String> {
        LpgStore::graph_names(self)
    }

    fn copy_named_graph(
        &self,
        source: Option<&str>,
        dest: Option<&str>,
    ) -> Result<(), String> {
        LpgStore::copy_graph(self, source, dest).map_err(|e| e.to_string())
    }

    // --- LpgStore-legacy inspection overrides ---
    fn nodes_with_label(&self, label: &str) -> Vec<Node> {
        LpgStore::nodes_with_label(self, label).collect()
    }

    fn edges_with_type(&self, edge_type: &str) -> Vec<Edge> {
        LpgStore::edges_with_type(self, edge_type).collect()
    }

    fn all_edges(&self) -> Vec<Edge> {
        LpgStore::all_edges(self).collect()
    }

    fn create_property_index(&self, property: &str) {
        LpgStore::create_property_index(self, property)
    }

    fn drop_property_index(&self, property: &str) -> bool {
        LpgStore::drop_property_index(self, property)
    }

    fn memory_breakdown(
        &self,
    ) -> (
        obrain_common::memory::usage::StoreMemory,
        obrain_common::memory::usage::IndexMemory,
        obrain_common::memory::usage::MvccMemory,
        obrain_common::memory::usage::StringPoolMemory,
    ) {
        LpgStore::memory_breakdown(self)
    }

    fn create_node(&self, labels: &[&str]) -> NodeId {
        LpgStore::create_node(self, labels)
    }

    fn create_node_versioned(
        &self,
        labels: &[&str],
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> NodeId {
        LpgStore::create_node_versioned(self, labels, epoch, transaction_id)
    }

    fn create_edge(&self, src: NodeId, dst: NodeId, edge_type: &str) -> EdgeId {
        LpgStore::create_edge(self, src, dst, edge_type)
    }

    fn create_edge_versioned(
        &self,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> EdgeId {
        LpgStore::create_edge_versioned(self, src, dst, edge_type, epoch, transaction_id)
    }

    fn batch_create_edges(&self, edges: &[(NodeId, NodeId, &str)]) -> Vec<EdgeId> {
        LpgStore::batch_create_edges(self, edges)
    }

    fn delete_node(&self, id: NodeId) -> bool {
        LpgStore::delete_node(self, id)
    }

    fn delete_node_versioned(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        if transaction_id == TransactionId::SYSTEM {
            LpgStore::delete_node_at_epoch(self, id, epoch)
        } else {
            LpgStore::delete_node_transactional(self, id, epoch, transaction_id)
        }
    }

    fn delete_node_edges(&self, node_id: NodeId) {
        LpgStore::delete_node_edges(self, node_id);
    }

    fn delete_edge(&self, id: EdgeId) -> bool {
        LpgStore::delete_edge(self, id)
    }

    fn delete_edge_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        if transaction_id == TransactionId::SYSTEM {
            LpgStore::delete_edge_at_epoch(self, id, epoch)
        } else {
            LpgStore::delete_edge_transactional(self, id, epoch, transaction_id)
        }
    }

    fn set_node_property(&self, id: NodeId, key: &str, value: Value) {
        LpgStore::set_node_property(self, id, key, value);
    }

    fn set_edge_property(&self, id: EdgeId, key: &str, value: Value) {
        LpgStore::set_edge_property(self, id, key, value);
    }

    fn set_node_property_versioned(
        &self,
        id: NodeId,
        key: &str,
        value: Value,
        transaction_id: TransactionId,
    ) {
        LpgStore::set_node_property_versioned(self, id, key, value, transaction_id);
    }

    fn set_edge_property_versioned(
        &self,
        id: EdgeId,
        key: &str,
        value: Value,
        transaction_id: TransactionId,
    ) {
        LpgStore::set_edge_property_versioned(self, id, key, value, transaction_id);
    }

    fn remove_node_property(&self, id: NodeId, key: &str) -> Option<Value> {
        LpgStore::remove_node_property(self, id, key)
    }

    fn remove_edge_property(&self, id: EdgeId, key: &str) -> Option<Value> {
        LpgStore::remove_edge_property(self, id, key)
    }

    fn remove_node_property_versioned(
        &self,
        id: NodeId,
        key: &str,
        transaction_id: TransactionId,
    ) -> Option<Value> {
        LpgStore::remove_node_property_versioned(self, id, key, transaction_id)
    }

    fn remove_edge_property_versioned(
        &self,
        id: EdgeId,
        key: &str,
        transaction_id: TransactionId,
    ) -> Option<Value> {
        LpgStore::remove_edge_property_versioned(self, id, key, transaction_id)
    }

    fn add_label(&self, node_id: NodeId, label: &str) -> bool {
        LpgStore::add_label(self, node_id, label)
    }

    fn remove_label(&self, node_id: NodeId, label: &str) -> bool {
        LpgStore::remove_label(self, node_id, label)
    }

    fn add_label_versioned(
        &self,
        node_id: NodeId,
        label: &str,
        transaction_id: TransactionId,
    ) -> bool {
        LpgStore::add_label_versioned(self, node_id, label, transaction_id)
    }

    fn remove_label_versioned(
        &self,
        node_id: NodeId,
        label: &str,
        transaction_id: TransactionId,
    ) -> bool {
        LpgStore::remove_label_versioned(self, node_id, label, transaction_id)
    }

    fn create_node_with_props(
        &self,
        labels: &[&str],
        properties: &[(PropertyKey, Value)],
    ) -> NodeId {
        // Delegate to LpgStore's optimized version that sets props under a single lock.
        LpgStore::create_node_with_props(
            self,
            labels,
            properties.iter().map(|(k, v)| (k.clone(), v.clone())),
        )
    }

    fn create_edge_with_props(
        &self,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
        properties: &[(PropertyKey, Value)],
    ) -> EdgeId {
        LpgStore::create_edge_with_props(
            self,
            src,
            dst,
            edge_type,
            properties.iter().map(|(k, v)| (k.clone(), v.clone())),
        )
    }
}
