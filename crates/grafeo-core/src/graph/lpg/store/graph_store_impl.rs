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
use grafeo_common::types::{EdgeId, EpochId, NodeId, PropertyKey, TxId, Value};
use grafeo_common::utils::hash::FxHashMap;
use std::sync::Arc;

impl GraphStore for LpgStore {
    fn get_node(&self, id: NodeId) -> Option<Node> {
        LpgStore::get_node(self, id)
    }

    fn get_edge(&self, id: EdgeId) -> Option<Edge> {
        LpgStore::get_edge(self, id)
    }

    fn get_node_versioned(&self, id: NodeId, epoch: EpochId, tx_id: TxId) -> Option<Node> {
        LpgStore::get_node_versioned(self, id, epoch, tx_id)
    }

    fn get_edge_versioned(&self, id: EdgeId, epoch: EpochId, tx_id: TxId) -> Option<Edge> {
        LpgStore::get_edge_versioned(self, id, epoch, tx_id)
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

    fn nodes_by_label(&self, label: &str) -> Vec<NodeId> {
        LpgStore::nodes_by_label(self, label)
    }

    fn node_count(&self) -> usize {
        LpgStore::node_count(self)
    }

    fn edge_count(&self) -> usize {
        LpgStore::edge_count(self)
    }

    fn edge_type(&self, id: EdgeId) -> Option<ArcStr> {
        LpgStore::edge_type(self, id)
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

    fn get_node_history(&self, id: NodeId) -> Vec<(EpochId, Option<EpochId>, Node)> {
        LpgStore::get_node_history(self, id)
    }

    fn get_edge_history(&self, id: EdgeId) -> Vec<(EpochId, Option<EpochId>, Edge)> {
        LpgStore::get_edge_history(self, id)
    }
}

impl GraphStoreMut for LpgStore {
    fn create_node(&self, labels: &[&str]) -> NodeId {
        LpgStore::create_node(self, labels)
    }

    fn create_node_versioned(&self, labels: &[&str], epoch: EpochId, tx_id: TxId) -> NodeId {
        LpgStore::create_node_versioned(self, labels, epoch, tx_id)
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
        tx_id: TxId,
    ) -> EdgeId {
        LpgStore::create_edge_versioned(self, src, dst, edge_type, epoch, tx_id)
    }

    fn batch_create_edges(&self, edges: &[(NodeId, NodeId, &str)]) -> Vec<EdgeId> {
        LpgStore::batch_create_edges(self, edges)
    }

    fn delete_node(&self, id: NodeId) -> bool {
        LpgStore::delete_node(self, id)
    }

    fn delete_node_versioned(&self, id: NodeId, epoch: EpochId, _tx_id: TxId) -> bool {
        LpgStore::delete_node_at_epoch(self, id, epoch)
    }

    fn delete_node_edges(&self, node_id: NodeId) {
        LpgStore::delete_node_edges(self, node_id);
    }

    fn delete_edge(&self, id: EdgeId) -> bool {
        LpgStore::delete_edge(self, id)
    }

    fn delete_edge_versioned(&self, id: EdgeId, epoch: EpochId, _tx_id: TxId) -> bool {
        LpgStore::delete_edge_at_epoch(self, id, epoch)
    }

    fn set_node_property(&self, id: NodeId, key: &str, value: Value) {
        LpgStore::set_node_property(self, id, key, value);
    }

    fn set_edge_property(&self, id: EdgeId, key: &str, value: Value) {
        LpgStore::set_edge_property(self, id, key, value);
    }

    fn remove_node_property(&self, id: NodeId, key: &str) -> Option<Value> {
        LpgStore::remove_node_property(self, id, key)
    }

    fn remove_edge_property(&self, id: EdgeId, key: &str) -> Option<Value> {
        LpgStore::remove_edge_property(self, id, key)
    }

    fn add_label(&self, node_id: NodeId, label: &str) -> bool {
        LpgStore::add_label(self, node_id, label)
    }

    fn remove_label(&self, node_id: NodeId, label: &str) -> bool {
        LpgStore::remove_label(self, node_id, label)
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
