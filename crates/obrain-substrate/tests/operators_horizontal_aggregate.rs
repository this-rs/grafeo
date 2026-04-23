//! Integration tests for `obrain_core::execution::operators::HorizontalAggregateOperator`
//! against the substrate backend.
//!
//! ## Why this lives in obrain-substrate and not in obrain-core
//!
//! Relocated as part of T17 Step 3 W2/Class-2 follow-up (decision
//! `b1dfe229`). `obrain-core` cannot take `obrain-substrate` as a
//! dev-dependency without creating two distinct compilation units of
//! `obrain-core` (gotcha `598dda40`). The forward direction
//! (`obrain-substrate → obrain-core`) has no cycle, so moving operator
//! tests into an integration test of `obrain-substrate` is the
//! post-T17-cutover home for the LPG-era fixtures.
//!
//! Run with:
//!
//! ```bash
//! cargo test -p obrain-substrate --test operators_horizontal_aggregate
//! ```

use obrain_common::types::{LogicalType, Value};
use obrain_core::execution::chunk::{DataChunk, DataChunkBuilder};
use obrain_core::execution::operators::{
    AggregateFunction, EntityKind, HorizontalAggregateOperator, Operator, OperatorResult,
};
use obrain_core::graph::GraphStore;
use obrain_core::graph::traits::GraphStoreMut;
use obrain_substrate::SubstrateStore;
use std::sync::Arc;

struct MockOperator {
    chunks: Vec<DataChunk>,
    position: usize,
}

impl MockOperator {
    fn new(chunks: Vec<DataChunk>) -> Self {
        Self {
            chunks,
            position: 0,
        }
    }
}

impl Operator for MockOperator {
    fn next(&mut self) -> OperatorResult {
        if self.position < self.chunks.len() {
            let chunk = std::mem::replace(&mut self.chunks[self.position], DataChunk::empty());
            self.position += 1;
            Ok(Some(chunk))
        } else {
            Ok(None)
        }
    }

    fn reset(&mut self) {
        self.position = 0;
    }

    fn name(&self) -> &'static str {
        "Mock"
    }
}

/// Substrate-backed fixture — three edges with float weights, paired
/// node IDs returned for downstream use in list chunks.
fn setup_store_with_edges() -> (Arc<dyn GraphStore>, Vec<Value>) {
    let store = SubstrateStore::open_tempfile().unwrap();
    let n1 = store.create_node(&[]);
    let n2 = store.create_node(&[]);

    let e1 = store.create_edge(n1, n2, "ROAD");
    let e2 = store.create_edge(n1, n2, "ROAD");
    let e3 = store.create_edge(n1, n2, "ROAD");

    store.set_edge_property(e1, "weight", Value::Float64(1.5));
    store.set_edge_property(e2, "weight", Value::Float64(2.5));
    store.set_edge_property(e3, "weight", Value::Float64(3.0));

    let edge_ids: Vec<Value> = vec![
        Value::Int64(e1.0 as i64),
        Value::Int64(e2.0 as i64),
        Value::Int64(e3.0 as i64),
    ];
    (Arc::new(store), edge_ids)
}

/// Substrate-backed fixture — three City nodes with float `pop` property.
fn setup_store_with_nodes() -> (Arc<dyn GraphStore>, Vec<Value>) {
    let store = SubstrateStore::open_tempfile().unwrap();
    // Use Float64 properties since the result column is Float64-typed
    // (Int64 values from SumInt finalize would be silently dropped by the Float64 column)
    let n1 = store.create_node(&["City"]);
    store.set_node_property(n1, "pop", Value::Float64(100.0));
    let n2 = store.create_node(&["City"]);
    store.set_node_property(n2, "pop", Value::Float64(200.0));
    let n3 = store.create_node(&["City"]);
    store.set_node_property(n3, "pop", Value::Float64(300.0));

    let node_ids: Vec<Value> = vec![
        Value::Int64(n1.0 as i64),
        Value::Int64(n2.0 as i64),
        Value::Int64(n3.0 as i64),
    ];
    (Arc::new(store), node_ids)
}

#[test]
fn test_horizontal_sum_over_edges() {
    let (store, edge_ids) = setup_store_with_edges();

    // Build a chunk with one row: column 0 = some label, column 1 = list of edge IDs
    let mut builder = DataChunkBuilder::new(&[LogicalType::String, LogicalType::Any]);
    builder
        .column_mut(0)
        .unwrap()
        .push_value(Value::String("path1".into()));
    builder
        .column_mut(1)
        .unwrap()
        .push_value(Value::List(edge_ids.into()));
    builder.advance_row();
    let chunk = builder.finish();

    let mock = MockOperator::new(vec![chunk]);
    let mut op = HorizontalAggregateOperator::new(
        Box::new(mock),
        1, // list_column_idx
        EntityKind::Edge,
        AggregateFunction::Sum,
        "weight".to_string(),
        store,
        2, // input_column_count
    );

    let result = op.next().unwrap().unwrap();
    assert_eq!(result.row_count(), 1);
    // Sum of weights: 1.5 + 2.5 + 3.0 = 7.0
    let agg_val = result.column(2).unwrap().get_float64(0).unwrap();
    assert!((agg_val - 7.0).abs() < 0.001);

    // Should be done
    assert!(op.next().unwrap().is_none());
}

#[test]
fn test_horizontal_sum_over_nodes() {
    let (store, node_ids) = setup_store_with_nodes();

    let mut builder = DataChunkBuilder::new(&[LogicalType::Any]);
    builder
        .column_mut(0)
        .unwrap()
        .push_value(Value::List(node_ids.into()));
    builder.advance_row();
    let chunk = builder.finish();

    let mock = MockOperator::new(vec![chunk]);
    let mut op = HorizontalAggregateOperator::new(
        Box::new(mock),
        0,
        EntityKind::Node,
        AggregateFunction::Sum,
        "pop".to_string(),
        store,
        1,
    );

    let result = op.next().unwrap().unwrap();
    assert_eq!(result.row_count(), 1);
    // Sum of node populations: 100.0 + 200.0 + 300.0 = 600.0
    let agg_val = result.column(1).unwrap().get_float64(0).unwrap();
    assert!((agg_val - 600.0).abs() < 0.001);
}

#[test]
fn test_horizontal_avg_over_edges() {
    let (store, edge_ids) = setup_store_with_edges();

    let mut builder = DataChunkBuilder::new(&[LogicalType::Any]);
    builder
        .column_mut(0)
        .unwrap()
        .push_value(Value::List(edge_ids.into()));
    builder.advance_row();
    let chunk = builder.finish();

    let mock = MockOperator::new(vec![chunk]);
    let mut op = HorizontalAggregateOperator::new(
        Box::new(mock),
        0,
        EntityKind::Edge,
        AggregateFunction::Avg,
        "weight".to_string(),
        store,
        1,
    );

    let result = op.next().unwrap().unwrap();
    // Avg: (1.5 + 2.5 + 3.0) / 3 = 2.333...
    let agg_val = result.column(1).unwrap().get_float64(0).unwrap();
    assert!((agg_val - 7.0 / 3.0).abs() < 0.001);
}

#[test]
fn test_horizontal_min_max_over_edges() {
    let (store, edge_ids) = setup_store_with_edges();

    // Test MIN
    let mut builder = DataChunkBuilder::new(&[LogicalType::Any]);
    builder
        .column_mut(0)
        .unwrap()
        .push_value(Value::List(edge_ids.clone().into()));
    builder.advance_row();
    let chunk = builder.finish();

    let mock = MockOperator::new(vec![chunk]);
    let mut op = HorizontalAggregateOperator::new(
        Box::new(mock),
        0,
        EntityKind::Edge,
        AggregateFunction::Min,
        "weight".to_string(),
        Arc::clone(&store),
        1,
    );

    let result = op.next().unwrap().unwrap();
    let min_val = result.column(1).unwrap().get_float64(0).unwrap();
    assert!((min_val - 1.5).abs() < 0.001);

    // Test MAX
    let mut builder = DataChunkBuilder::new(&[LogicalType::Any]);
    builder
        .column_mut(0)
        .unwrap()
        .push_value(Value::List(edge_ids.into()));
    builder.advance_row();
    let chunk = builder.finish();

    let mock = MockOperator::new(vec![chunk]);
    let mut op = HorizontalAggregateOperator::new(
        Box::new(mock),
        0,
        EntityKind::Edge,
        AggregateFunction::Max,
        "weight".to_string(),
        store,
        1,
    );

    let result = op.next().unwrap().unwrap();
    let max_val = result.column(1).unwrap().get_float64(0).unwrap();
    assert!((max_val - 3.0).abs() < 0.001);
}

#[test]
fn test_horizontal_empty_list_returns_null() {
    let store: Arc<dyn GraphStore> = Arc::new(SubstrateStore::open_tempfile().unwrap());

    let mut builder = DataChunkBuilder::new(&[LogicalType::Any]);
    builder
        .column_mut(0)
        .unwrap()
        .push_value(Value::List(vec![].into()));
    builder.advance_row();
    let chunk = builder.finish();

    let mock = MockOperator::new(vec![chunk]);
    let mut op = HorizontalAggregateOperator::new(
        Box::new(mock),
        0,
        EntityKind::Edge,
        AggregateFunction::Sum,
        "weight".to_string(),
        store,
        1,
    );

    let result = op.next().unwrap().unwrap();
    assert_eq!(result.row_count(), 1);
    // Empty list sum should finalize to NULL (ISO/IEC 39075 Section 20.9)
    let agg_val = result.column(1).unwrap().get_value(0);
    assert!(
        matches!(agg_val, Some(Value::Null)),
        "Expected Null, got {agg_val:?}"
    );
}

#[test]
fn test_horizontal_non_list_column_returns_null() {
    let store: Arc<dyn GraphStore> = Arc::new(SubstrateStore::open_tempfile().unwrap());

    // Put a non-list value in the list column
    let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
    builder.column_mut(0).unwrap().push_int64(42);
    builder.advance_row();
    let chunk = builder.finish();

    let mock = MockOperator::new(vec![chunk]);
    let mut op = HorizontalAggregateOperator::new(
        Box::new(mock),
        0,
        EntityKind::Edge,
        AggregateFunction::Sum,
        "weight".to_string(),
        store,
        1,
    );

    let result = op.next().unwrap().unwrap();
    // Non-list value should produce Null
    let agg_val = result.column(1).unwrap().get_value(0);
    assert_eq!(agg_val, Some(Value::Null));
}

#[test]
fn test_horizontal_multiple_rows() {
    let (store, edge_ids) = setup_store_with_edges();

    let mut builder = DataChunkBuilder::new(&[LogicalType::String, LogicalType::Any]);
    // Row 0: all three edges
    builder
        .column_mut(0)
        .unwrap()
        .push_value(Value::String("path_all".into()));
    builder
        .column_mut(1)
        .unwrap()
        .push_value(Value::List(edge_ids.clone().into()));
    builder.advance_row();
    // Row 1: only first edge
    builder
        .column_mut(0)
        .unwrap()
        .push_value(Value::String("path_one".into()));
    builder
        .column_mut(1)
        .unwrap()
        .push_value(Value::List(vec![edge_ids[0].clone()].into()));
    builder.advance_row();
    let chunk = builder.finish();

    let mock = MockOperator::new(vec![chunk]);
    let mut op = HorizontalAggregateOperator::new(
        Box::new(mock),
        1,
        EntityKind::Edge,
        AggregateFunction::Sum,
        "weight".to_string(),
        store,
        2,
    );

    let result = op.next().unwrap().unwrap();
    assert_eq!(result.row_count(), 2);

    // Row 0: sum = 7.0
    let val0 = result.column(2).unwrap().get_float64(0).unwrap();
    assert!((val0 - 7.0).abs() < 0.001);
    // Row 1: sum = 1.5
    let val1 = result.column(2).unwrap().get_float64(1).unwrap();
    assert!((val1 - 1.5).abs() < 0.001);
}

#[test]
fn test_horizontal_reset() {
    let (store, edge_ids) = setup_store_with_edges();

    // Build two identical chunks so after reset the second is still available
    let mut builder1 = DataChunkBuilder::new(&[LogicalType::Any]);
    builder1
        .column_mut(0)
        .unwrap()
        .push_value(Value::List(edge_ids.clone().into()));
    builder1.advance_row();

    let mut builder2 = DataChunkBuilder::new(&[LogicalType::Any]);
    builder2
        .column_mut(0)
        .unwrap()
        .push_value(Value::List(edge_ids.into()));
    builder2.advance_row();

    let mock = MockOperator::new(vec![builder1.finish(), builder2.finish()]);
    let mut op = HorizontalAggregateOperator::new(
        Box::new(mock),
        0,
        EntityKind::Edge,
        AggregateFunction::Sum,
        "weight".to_string(),
        store,
        1,
    );

    // First chunk
    let result = op.next().unwrap().unwrap();
    assert_eq!(result.row_count(), 1);

    // Second chunk
    let result = op.next().unwrap().unwrap();
    assert_eq!(result.row_count(), 1);

    // Done
    assert!(op.next().unwrap().is_none());

    // After reset, position goes back to 0 but chunks are consumed
    // This verifies reset() propagates to the child
    op.reset();
}

#[test]
fn test_horizontal_name() {
    let store: Arc<dyn GraphStore> = Arc::new(SubstrateStore::open_tempfile().unwrap());
    let mock = MockOperator::new(vec![]);
    let op = HorizontalAggregateOperator::new(
        Box::new(mock),
        0,
        EntityKind::Edge,
        AggregateFunction::Sum,
        "weight".to_string(),
        store,
        1,
    );
    assert_eq!(op.name(), "HorizontalAggregate");
}

#[test]
fn test_horizontal_child_returns_none() {
    let store: Arc<dyn GraphStore> = Arc::new(SubstrateStore::open_tempfile().unwrap());
    let mock = MockOperator::new(vec![]); // No chunks
    let mut op = HorizontalAggregateOperator::new(
        Box::new(mock),
        0,
        EntityKind::Edge,
        AggregateFunction::Sum,
        "weight".to_string(),
        store,
        1,
    );

    assert!(op.next().unwrap().is_none());
}
