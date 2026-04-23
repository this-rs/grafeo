//! Integration tests for `obrain_core::execution::operators::mutation::*`
//! (CreateNode, CreateEdge, DeleteNode, DeleteEdge, AddLabel, RemoveLabel,
//! SetProperty, ConstraintValidator) against the substrate backend.
//!
//! Relocated from `crates/obrain-core/src/execution/operators/mutation.rs`'s
//! in-crate `#[cfg(test)] mod tests` block as part of T17 W4.p4. See note
//! tagged `t17 w4.p4 migration-pattern` for rationale — briefly, a
//! substrate-backed fixture cannot live inside `obrain-core` because the
//! dev-dep cycle (gotcha `598dda40-a186-4be3-97f3-c75053af4e6e`) produces
//! two distinct compilation units of `obrain-core`, breaking the
//! `Arc<SubstrateStore> as Arc<dyn GraphStoreMut>` trait cast.
//!
//! Run with:
//!
//! ```bash
//! cargo test -p obrain-substrate --test operators_mutation
//! ```

use obrain_common::types::{EdgeId, LogicalType, NodeId, PropertyKey, Value};
use obrain_core::execution::DataChunk;
use obrain_core::execution::chunk::DataChunkBuilder;
use obrain_core::execution::operators::{
    AddLabelOperator, ConstraintValidator, CreateEdgeOperator, CreateNodeOperator,
    DeleteEdgeOperator, DeleteNodeOperator, Operator, OperatorError, OperatorResult,
    PropertySource, RemoveLabelOperator, SetPropertyOperator,
};
use obrain_core::graph::traits::GraphStoreMut;
use obrain_substrate::SubstrateStore;
use std::sync::Arc;

// ── Helpers ────────────────────────────────────────────────────

fn create_test_store() -> Arc<dyn GraphStoreMut> {
    Arc::new(SubstrateStore::open_tempfile().unwrap())
}

struct MockInput {
    chunk: Option<DataChunk>,
}

impl MockInput {
    fn boxed(chunk: DataChunk) -> Box<Self> {
        Box::new(Self { chunk: Some(chunk) })
    }
}

impl Operator for MockInput {
    fn next(&mut self) -> OperatorResult {
        Ok(self.chunk.take())
    }
    fn reset(&mut self) {}
    fn name(&self) -> &'static str {
        "MockInput"
    }
}

struct EmptyInput;
impl Operator for EmptyInput {
    fn next(&mut self) -> OperatorResult {
        Ok(None)
    }
    fn reset(&mut self) {}
    fn name(&self) -> &'static str {
        "EmptyInput"
    }
}

fn node_id_chunk(ids: &[NodeId]) -> DataChunk {
    let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
    for id in ids {
        builder.column_mut(0).unwrap().push_int64(id.0 as i64);
        builder.advance_row();
    }
    builder.finish()
}

fn edge_id_chunk(ids: &[EdgeId]) -> DataChunk {
    let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
    for id in ids {
        builder.column_mut(0).unwrap().push_int64(id.0 as i64);
        builder.advance_row();
    }
    builder.finish()
}

// ── CreateNodeOperator ──────────────────────────────────────

#[test]
fn test_create_node_standalone() {
    let store = create_test_store();

    let mut op = CreateNodeOperator::new(
        Arc::clone(&store),
        None,
        vec!["Person".to_string()],
        vec![(
            "name".to_string(),
            PropertySource::Constant(Value::String("Alix".into())),
        )],
        vec![LogicalType::Int64],
        0,
    );

    let chunk = op.next().unwrap().unwrap();
    assert_eq!(chunk.row_count(), 1);

    // Second call should return None (standalone executes once)
    assert!(op.next().unwrap().is_none());

    assert_eq!(store.node_count(), 1);
}

#[test]
fn test_create_edge() {
    let store = create_test_store();

    let node1 = store.create_node(&["Person"]);
    let node2 = store.create_node(&["Person"]);

    let mut builder = DataChunkBuilder::new(&[LogicalType::Int64, LogicalType::Int64]);
    builder.column_mut(0).unwrap().push_int64(node1.0 as i64);
    builder.column_mut(1).unwrap().push_int64(node2.0 as i64);
    builder.advance_row();

    let mut op = CreateEdgeOperator::new(
        Arc::clone(&store),
        MockInput::boxed(builder.finish()),
        0,
        1,
        "KNOWS".to_string(),
        vec![LogicalType::Int64, LogicalType::Int64],
    );

    let _chunk = op.next().unwrap().unwrap();
    assert_eq!(store.edge_count(), 1);
}

#[test]
fn test_delete_node() {
    let store = create_test_store();

    let node_id = store.create_node(&["Person"]);
    assert_eq!(store.node_count(), 1);

    let mut op = DeleteNodeOperator::new(
        Arc::clone(&store),
        MockInput::boxed(node_id_chunk(&[node_id])),
        0,
        vec![LogicalType::Node],
        false,
    );

    let chunk = op.next().unwrap().unwrap();
    // Pass-through: output row contains the original node ID
    assert_eq!(chunk.row_count(), 1);
    assert_eq!(store.node_count(), 0);
}

// ── DeleteEdgeOperator ───────────────────────────────────────

#[test]
fn test_delete_edge() {
    let store = create_test_store();

    let n1 = store.create_node(&["Person"]);
    let n2 = store.create_node(&["Person"]);
    let eid = store.create_edge(n1, n2, "KNOWS");
    assert_eq!(store.edge_count(), 1);

    let mut op = DeleteEdgeOperator::new(
        Arc::clone(&store),
        MockInput::boxed(edge_id_chunk(&[eid])),
        0,
        vec![LogicalType::Node],
    );

    let chunk = op.next().unwrap().unwrap();
    assert_eq!(chunk.row_count(), 1);
    assert_eq!(store.edge_count(), 0);
}

#[test]
fn test_delete_edge_no_input_returns_none() {
    let store = create_test_store();

    let mut op = DeleteEdgeOperator::new(
        Arc::clone(&store),
        Box::new(EmptyInput),
        0,
        vec![LogicalType::Int64],
    );

    assert!(op.next().unwrap().is_none());
}

#[test]
fn test_delete_multiple_edges() {
    let store = create_test_store();

    let n1 = store.create_node(&["N"]);
    let n2 = store.create_node(&["N"]);
    let e1 = store.create_edge(n1, n2, "R");
    let e2 = store.create_edge(n2, n1, "S");
    assert_eq!(store.edge_count(), 2);

    let mut op = DeleteEdgeOperator::new(
        Arc::clone(&store),
        MockInput::boxed(edge_id_chunk(&[e1, e2])),
        0,
        vec![LogicalType::Node],
    );

    let chunk = op.next().unwrap().unwrap();
    assert_eq!(chunk.row_count(), 2);
    assert_eq!(store.edge_count(), 0);
}

// ── DeleteNodeOperator with DETACH ───────────────────────────

#[test]
fn test_delete_node_detach() {
    let store = create_test_store();

    let n1 = store.create_node(&["Person"]);
    let n2 = store.create_node(&["Person"]);
    store.create_edge(n1, n2, "KNOWS");
    store.create_edge(n2, n1, "FOLLOWS");
    assert_eq!(store.edge_count(), 2);

    let mut op = DeleteNodeOperator::new(
        Arc::clone(&store),
        MockInput::boxed(node_id_chunk(&[n1])),
        0,
        vec![LogicalType::Node],
        true, // detach = true
    );

    let chunk = op.next().unwrap().unwrap();
    assert_eq!(chunk.row_count(), 1);
    assert_eq!(store.node_count(), 1);
    assert_eq!(store.edge_count(), 0); // edges detached
}

// ── AddLabelOperator ─────────────────────────────────────────

#[test]
fn test_add_label() {
    let store = create_test_store();

    let node = store.create_node(&["Person"]);

    let mut op = AddLabelOperator::new(
        Arc::clone(&store),
        MockInput::boxed(node_id_chunk(&[node])),
        0,
        vec!["Employee".to_string()],
        vec![LogicalType::Int64],
    );

    let chunk = op.next().unwrap().unwrap();
    // After fix: operator passes through input columns (node ID), not a count
    assert_eq!(chunk.row_count(), 1);
    let node_id_val = chunk.column(0).unwrap().get_int64(0).unwrap();
    assert_eq!(node_id_val, node.0 as i64);

    // Verify label was added
    let node_data = store.get_node(node).unwrap();
    let labels: Vec<&str> = node_data.labels.iter().map(|l| l.as_ref()).collect();
    assert!(labels.contains(&"Person"));
    assert!(labels.contains(&"Employee"));
}

#[test]
fn test_add_multiple_labels() {
    let store = create_test_store();

    let node = store.create_node(&["Base"]);

    let mut op = AddLabelOperator::new(
        Arc::clone(&store),
        MockInput::boxed(node_id_chunk(&[node])),
        0,
        vec!["LabelA".to_string(), "LabelB".to_string()],
        vec![LogicalType::Int64],
    );

    let chunk = op.next().unwrap().unwrap();
    // After fix: operator passes through input rows, not a count
    assert_eq!(chunk.row_count(), 1);
    let node_id_val = chunk.column(0).unwrap().get_int64(0).unwrap();
    assert_eq!(node_id_val, node.0 as i64);

    let node_data = store.get_node(node).unwrap();
    let labels: Vec<&str> = node_data.labels.iter().map(|l| l.as_ref()).collect();
    assert!(labels.contains(&"LabelA"));
    assert!(labels.contains(&"LabelB"));
}

#[test]
fn test_add_label_no_input_returns_none() {
    let store = create_test_store();

    let mut op = AddLabelOperator::new(
        Arc::clone(&store),
        Box::new(EmptyInput),
        0,
        vec!["Foo".to_string()],
        vec![LogicalType::Int64],
    );

    assert!(op.next().unwrap().is_none());
}

// ── RemoveLabelOperator ──────────────────────────────────────

#[test]
fn test_remove_label() {
    let store = create_test_store();

    let node = store.create_node(&["Person", "Employee"]);

    let mut op = RemoveLabelOperator::new(
        Arc::clone(&store),
        MockInput::boxed(node_id_chunk(&[node])),
        0,
        vec!["Employee".to_string()],
        vec![LogicalType::Int64],
    );

    let chunk = op.next().unwrap().unwrap();
    // After fix: operator passes through input rows, not a count
    assert_eq!(chunk.row_count(), 1);
    let node_id_val = chunk.column(0).unwrap().get_int64(0).unwrap();
    assert_eq!(node_id_val, node.0 as i64);

    // Verify label was removed
    let node_data = store.get_node(node).unwrap();
    let labels: Vec<&str> = node_data.labels.iter().map(|l| l.as_ref()).collect();
    assert!(labels.contains(&"Person"));
    assert!(!labels.contains(&"Employee"));
}

#[test]
fn test_remove_nonexistent_label() {
    let store = create_test_store();

    let node = store.create_node(&["Person"]);

    let mut op = RemoveLabelOperator::new(
        Arc::clone(&store),
        MockInput::boxed(node_id_chunk(&[node])),
        0,
        vec!["NonExistent".to_string()],
        vec![LogicalType::Int64],
    );

    let chunk = op.next().unwrap().unwrap();
    // After fix: operator passes through input rows regardless of whether label existed
    assert_eq!(chunk.row_count(), 1);
    let node_id_val = chunk.column(0).unwrap().get_int64(0).unwrap();
    assert_eq!(node_id_val, node.0 as i64);
}

#[test]
fn test_add_label_type_mismatch() {
    // Covers the TypeMismatch error path when input column is not Int64
    let store = create_test_store();

    // Create a chunk with a String value instead of Int64
    let mut builder = DataChunkBuilder::new(&[LogicalType::String]);
    builder
        .column_mut(0)
        .unwrap()
        .push_value(Value::String("not_a_node_id".into()));
    builder.advance_row();

    let mut op = AddLabelOperator::new(
        Arc::clone(&store),
        MockInput::boxed(builder.finish()),
        0,
        vec!["Employee".to_string()],
        vec![LogicalType::String],
    );

    let result = op.next();
    assert!(result.is_err(), "Should fail with TypeMismatch");
    let err = result.unwrap_err();
    assert!(
        format!("{err:?}").contains("TypeMismatch"),
        "Error should be TypeMismatch, got: {err:?}"
    );
}

#[test]
fn test_remove_label_type_mismatch() {
    // Covers the TypeMismatch error path when input column is not Int64
    let store = create_test_store();

    let mut builder = DataChunkBuilder::new(&[LogicalType::String]);
    builder
        .column_mut(0)
        .unwrap()
        .push_value(Value::String("not_a_node_id".into()));
    builder.advance_row();

    let mut op = RemoveLabelOperator::new(
        Arc::clone(&store),
        MockInput::boxed(builder.finish()),
        0,
        vec!["Employee".to_string()],
        vec![LogicalType::String],
    );

    let result = op.next();
    assert!(result.is_err(), "Should fail with TypeMismatch");
    let err = result.unwrap_err();
    assert!(
        format!("{err:?}").contains("TypeMismatch"),
        "Error should be TypeMismatch, got: {err:?}"
    );
}

#[test]
fn test_add_label_multiple_rows_with_null() {
    // Covers the push_value(Value::Null) branch during column copy
    // when a multi-column input has null values
    let store = create_test_store();

    let node = store.create_node(&["Person"]);

    // Create a 2-column chunk: [node_id, null_property]
    let mut builder = DataChunkBuilder::new(&[LogicalType::Int64, LogicalType::String]);
    builder.column_mut(0).unwrap().push_int64(node.0 as i64);
    builder.column_mut(1).unwrap().push_value(Value::Null);
    builder.advance_row();

    let mut op = AddLabelOperator::new(
        Arc::clone(&store),
        MockInput::boxed(builder.finish()),
        0,
        vec!["Tagged".to_string()],
        vec![LogicalType::Int64, LogicalType::String],
    );

    let chunk = op.next().unwrap().unwrap();
    assert_eq!(chunk.row_count(), 1);
    // Second column should be Null (passed through)
    let val = chunk.column(1).unwrap().get_value(0);
    assert!(
        val.is_none() || val == Some(Value::Null),
        "Null property should be preserved"
    );
}

// ── SetPropertyOperator ──────────────────────────────────────

#[test]
fn test_set_node_property_constant() {
    let store = create_test_store();

    let node = store.create_node(&["Person"]);

    let mut op = SetPropertyOperator::new_for_node(
        Arc::clone(&store),
        MockInput::boxed(node_id_chunk(&[node])),
        0,
        vec![(
            "name".to_string(),
            PropertySource::Constant(Value::String("Alix".into())),
        )],
        vec![LogicalType::Int64],
    );

    let chunk = op.next().unwrap().unwrap();
    assert_eq!(chunk.row_count(), 1);

    // Verify property was set
    let node_data = store.get_node(node).unwrap();
    assert_eq!(
        node_data
            .properties
            .get(&obrain_common::types::PropertyKey::new("name")),
        Some(&Value::String("Alix".into()))
    );
}

#[test]
fn test_set_node_property_from_column() {
    let store = create_test_store();

    let node = store.create_node(&["Person"]);

    // Input: column 0 = node ID, column 1 = property value
    let mut builder = DataChunkBuilder::new(&[LogicalType::Int64, LogicalType::String]);
    builder.column_mut(0).unwrap().push_int64(node.0 as i64);
    builder
        .column_mut(1)
        .unwrap()
        .push_value(Value::String("Gus".into()));
    builder.advance_row();

    let mut op = SetPropertyOperator::new_for_node(
        Arc::clone(&store),
        MockInput::boxed(builder.finish()),
        0,
        vec![("name".to_string(), PropertySource::Column(1))],
        vec![LogicalType::Int64, LogicalType::String],
    );

    let chunk = op.next().unwrap().unwrap();
    assert_eq!(chunk.row_count(), 1);

    let node_data = store.get_node(node).unwrap();
    assert_eq!(
        node_data
            .properties
            .get(&obrain_common::types::PropertyKey::new("name")),
        Some(&Value::String("Gus".into()))
    );
}

#[test]
fn test_set_edge_property() {
    let store = create_test_store();

    let n1 = store.create_node(&["N"]);
    let n2 = store.create_node(&["N"]);
    let eid = store.create_edge(n1, n2, "KNOWS");

    let mut op = SetPropertyOperator::new_for_edge(
        Arc::clone(&store),
        MockInput::boxed(edge_id_chunk(&[eid])),
        0,
        vec![(
            "weight".to_string(),
            PropertySource::Constant(Value::Float64(0.75)),
        )],
        vec![LogicalType::Int64],
    );

    let chunk = op.next().unwrap().unwrap();
    assert_eq!(chunk.row_count(), 1);

    let edge_data = store.get_edge(eid).unwrap();
    assert_eq!(
        edge_data
            .properties
            .get(&obrain_common::types::PropertyKey::new("weight")),
        Some(&Value::Float64(0.75))
    );
}

#[test]
fn test_set_multiple_properties() {
    let store = create_test_store();

    let node = store.create_node(&["Person"]);

    let mut op = SetPropertyOperator::new_for_node(
        Arc::clone(&store),
        MockInput::boxed(node_id_chunk(&[node])),
        0,
        vec![
            (
                "name".to_string(),
                PropertySource::Constant(Value::String("Alix".into())),
            ),
            (
                "age".to_string(),
                PropertySource::Constant(Value::Int64(30)),
            ),
        ],
        vec![LogicalType::Int64],
    );

    op.next().unwrap().unwrap();

    let node_data = store.get_node(node).unwrap();
    assert_eq!(
        node_data
            .properties
            .get(&obrain_common::types::PropertyKey::new("name")),
        Some(&Value::String("Alix".into()))
    );
    assert_eq!(
        node_data
            .properties
            .get(&obrain_common::types::PropertyKey::new("age")),
        Some(&Value::Int64(30))
    );
}

#[test]
fn test_set_property_no_input_returns_none() {
    let store = create_test_store();

    let mut op = SetPropertyOperator::new_for_node(
        Arc::clone(&store),
        Box::new(EmptyInput),
        0,
        vec![("x".to_string(), PropertySource::Constant(Value::Int64(1)))],
        vec![LogicalType::Int64],
    );

    assert!(op.next().unwrap().is_none());
}

// ── Error paths ──────────────────────────────────────────────

#[test]
fn test_delete_node_without_detach_errors_when_edges_exist() {
    let store = create_test_store();

    let n1 = store.create_node(&["Person"]);
    let n2 = store.create_node(&["Person"]);
    store.create_edge(n1, n2, "KNOWS");

    let mut op = DeleteNodeOperator::new(
        Arc::clone(&store),
        MockInput::boxed(node_id_chunk(&[n1])),
        0,
        vec![LogicalType::Int64],
        false, // no detach
    );

    let err = op.next().unwrap_err();
    match err {
        OperatorError::ConstraintViolation(msg) => {
            assert!(msg.contains("connected edge"), "unexpected message: {msg}");
        }
        other => panic!("expected ConstraintViolation, got {other:?}"),
    }
    // Node should still exist
    assert_eq!(store.node_count(), 2);
}

// ── CreateNodeOperator with input ───────────────────────────

#[test]
fn test_create_node_with_input_operator() {
    let store = create_test_store();

    // Seed node to provide input rows
    let existing = store.create_node(&["Seed"]);

    let mut op = CreateNodeOperator::new(
        Arc::clone(&store),
        Some(MockInput::boxed(node_id_chunk(&[existing]))),
        vec!["Created".to_string()],
        vec![(
            "source".to_string(),
            PropertySource::Constant(Value::String("from_input".into())),
        )],
        vec![LogicalType::Int64, LogicalType::Int64], // input col + output col
        1,                                            // output column for new node ID
    );

    let chunk = op.next().unwrap().unwrap();
    assert_eq!(chunk.row_count(), 1);

    // Should have created one new node (2 total: Seed + Created)
    assert_eq!(store.node_count(), 2);

    // Exhausted
    assert!(op.next().unwrap().is_none());
}

// ── CreateEdgeOperator with properties and output column ────

#[test]
fn test_create_edge_with_properties_and_output_column() {
    let store = create_test_store();

    let n1 = store.create_node(&["Person"]);
    let n2 = store.create_node(&["Person"]);

    let mut builder = DataChunkBuilder::new(&[LogicalType::Int64, LogicalType::Int64]);
    builder.column_mut(0).unwrap().push_int64(n1.0 as i64);
    builder.column_mut(1).unwrap().push_int64(n2.0 as i64);
    builder.advance_row();

    let mut op = CreateEdgeOperator::new(
        Arc::clone(&store),
        MockInput::boxed(builder.finish()),
        0,
        1,
        "KNOWS".to_string(),
        vec![LogicalType::Int64, LogicalType::Int64, LogicalType::Int64],
    )
    .with_properties(vec![(
        "since".to_string(),
        PropertySource::Constant(Value::Int64(2024)),
    )])
    .with_output_column(2);

    let chunk = op.next().unwrap().unwrap();
    assert_eq!(chunk.row_count(), 1);
    assert_eq!(store.edge_count(), 1);

    // Verify the output chunk contains the edge ID in column 2
    let edge_id_raw = chunk
        .column(2)
        .and_then(|c| c.get_int64(0))
        .expect("edge ID should be in output column 2");
    let edge_id = EdgeId(edge_id_raw as u64);

    // Verify the edge has the property
    let edge = store.get_edge(edge_id).expect("edge should exist");
    assert_eq!(
        edge.properties
            .get(&obrain_common::types::PropertyKey::new("since")),
        Some(&Value::Int64(2024))
    );
}

// ── SetPropertyOperator with map replacement ────────────────

#[test]
fn test_set_property_map_replace() {
    use std::collections::BTreeMap;

    let store = create_test_store();

    let node = store.create_node(&["Person"]);
    store.set_node_property(node, "old_prop", Value::String("should_be_removed".into()));

    let mut map = BTreeMap::new();
    map.insert(PropertyKey::new("new_key"), Value::String("new_val".into()));

    let mut op = SetPropertyOperator::new_for_node(
        Arc::clone(&store),
        MockInput::boxed(node_id_chunk(&[node])),
        0,
        vec![(
            "*".to_string(),
            PropertySource::Constant(Value::Map(Arc::new(map))),
        )],
        vec![LogicalType::Int64],
    )
    .with_replace(true);

    op.next().unwrap().unwrap();

    let node_data = store.get_node(node).unwrap();
    // Old property should be gone
    assert!(
        node_data
            .properties
            .get(&PropertyKey::new("old_prop"))
            .is_none()
    );
    // New property should exist
    assert_eq!(
        node_data.properties.get(&PropertyKey::new("new_key")),
        Some(&Value::String("new_val".into()))
    );
}

// ── SetPropertyOperator with map merge (no replace) ─────────

#[test]
fn test_set_property_map_merge() {
    use std::collections::BTreeMap;

    let store = create_test_store();

    let node = store.create_node(&["Person"]);
    store.set_node_property(node, "existing", Value::Int64(42));

    let mut map = BTreeMap::new();
    map.insert(PropertyKey::new("added"), Value::String("hello".into()));

    let mut op = SetPropertyOperator::new_for_node(
        Arc::clone(&store),
        MockInput::boxed(node_id_chunk(&[node])),
        0,
        vec![(
            "*".to_string(),
            PropertySource::Constant(Value::Map(Arc::new(map))),
        )],
        vec![LogicalType::Int64],
    ); // replace defaults to false

    op.next().unwrap().unwrap();

    let node_data = store.get_node(node).unwrap();
    // Existing property should still be there
    assert_eq!(
        node_data.properties.get(&PropertyKey::new("existing")),
        Some(&Value::Int64(42))
    );
    // New property should also exist
    assert_eq!(
        node_data.properties.get(&PropertyKey::new("added")),
        Some(&Value::String("hello".into()))
    );
}

// ── PropertySource::PropertyAccess ──────────────────────────

#[test]
fn test_property_source_property_access() {
    let store = create_test_store();

    let source_node = store.create_node(&["Source"]);
    store.set_node_property(source_node, "name", Value::String("Alix".into()));

    let target_node = store.create_node(&["Target"]);

    // Build chunk: col 0 = source node ID (Node type for PropertyAccess), col 1 = target node ID
    let mut builder = DataChunkBuilder::new(&[LogicalType::Node, LogicalType::Int64]);
    builder.column_mut(0).unwrap().push_node_id(source_node);
    builder
        .column_mut(1)
        .unwrap()
        .push_int64(target_node.0 as i64);
    builder.advance_row();

    let mut op = SetPropertyOperator::new_for_node(
        Arc::clone(&store),
        MockInput::boxed(builder.finish()),
        1, // entity column = target node
        vec![(
            "copied_name".to_string(),
            PropertySource::PropertyAccess {
                column: 0,
                property: "name".to_string(),
            },
        )],
        vec![LogicalType::Node, LogicalType::Int64],
    );

    op.next().unwrap().unwrap();

    let target_data = store.get_node(target_node).unwrap();
    assert_eq!(
        target_data.properties.get(&PropertyKey::new("copied_name")),
        Some(&Value::String("Alix".into()))
    );
}

// ── ConstraintValidator integration ─────────────────────────

#[test]
fn test_create_node_with_constraint_validator() {
    let store = create_test_store();

    struct RejectAgeValidator;
    impl ConstraintValidator for RejectAgeValidator {
        fn validate_node_property(
            &self,
            _labels: &[String],
            key: &str,
            _value: &Value,
        ) -> Result<(), OperatorError> {
            if key == "forbidden" {
                return Err(OperatorError::ConstraintViolation(
                    "property 'forbidden' is not allowed".to_string(),
                ));
            }
            Ok(())
        }
        fn validate_node_complete(
            &self,
            _labels: &[String],
            _properties: &[(String, Value)],
        ) -> Result<(), OperatorError> {
            Ok(())
        }
        fn check_unique_node_property(
            &self,
            _labels: &[String],
            _key: &str,
            _value: &Value,
        ) -> Result<(), OperatorError> {
            Ok(())
        }
        fn validate_edge_property(
            &self,
            _edge_type: &str,
            _key: &str,
            _value: &Value,
        ) -> Result<(), OperatorError> {
            Ok(())
        }
        fn validate_edge_complete(
            &self,
            _edge_type: &str,
            _properties: &[(String, Value)],
        ) -> Result<(), OperatorError> {
            Ok(())
        }
    }

    // Valid property should succeed
    let mut op = CreateNodeOperator::new(
        Arc::clone(&store),
        None,
        vec!["Thing".to_string()],
        vec![(
            "name".to_string(),
            PropertySource::Constant(Value::String("ok".into())),
        )],
        vec![LogicalType::Int64],
        0,
    )
    .with_validator(Arc::new(RejectAgeValidator));

    assert!(op.next().is_ok());
    assert_eq!(store.node_count(), 1);

    // Forbidden property should fail
    let mut op = CreateNodeOperator::new(
        Arc::clone(&store),
        None,
        vec!["Thing".to_string()],
        vec![(
            "forbidden".to_string(),
            PropertySource::Constant(Value::Int64(1)),
        )],
        vec![LogicalType::Int64],
        0,
    )
    .with_validator(Arc::new(RejectAgeValidator));

    let err = op.next().unwrap_err();
    assert!(matches!(err, OperatorError::ConstraintViolation(_)));
    // Node count should still be 2 (the node is created before validation, but the error
    // propagates - this tests the validation logic fires)
}

// ── Reset behavior ──────────────────────────────────────────

#[test]
fn test_create_node_reset_allows_re_execution() {
    let store = create_test_store();

    let mut op = CreateNodeOperator::new(
        Arc::clone(&store),
        None,
        vec!["Person".to_string()],
        vec![],
        vec![LogicalType::Int64],
        0,
    );

    // First execution
    assert!(op.next().unwrap().is_some());
    assert!(op.next().unwrap().is_none());

    // Reset and re-execute
    op.reset();
    assert!(op.next().unwrap().is_some());

    assert_eq!(store.node_count(), 2);
}

// ── Operator name() ──────────────────────────────────────────

#[test]
fn test_operator_names() {
    let store = create_test_store();

    let op = CreateNodeOperator::new(
        Arc::clone(&store),
        None,
        vec![],
        vec![],
        vec![LogicalType::Int64],
        0,
    );
    assert_eq!(op.name(), "CreateNode");

    let op = CreateEdgeOperator::new(
        Arc::clone(&store),
        Box::new(EmptyInput),
        0,
        1,
        "R".to_string(),
        vec![LogicalType::Int64],
    );
    assert_eq!(op.name(), "CreateEdge");

    let op = DeleteNodeOperator::new(
        Arc::clone(&store),
        Box::new(EmptyInput),
        0,
        vec![LogicalType::Int64],
        false,
    );
    assert_eq!(op.name(), "DeleteNode");

    let op = DeleteEdgeOperator::new(
        Arc::clone(&store),
        Box::new(EmptyInput),
        0,
        vec![LogicalType::Int64],
    );
    assert_eq!(op.name(), "DeleteEdge");

    let op = AddLabelOperator::new(
        Arc::clone(&store),
        Box::new(EmptyInput),
        0,
        vec!["L".to_string()],
        vec![LogicalType::Int64],
    );
    assert_eq!(op.name(), "AddLabel");

    let op = RemoveLabelOperator::new(
        Arc::clone(&store),
        Box::new(EmptyInput),
        0,
        vec!["L".to_string()],
        vec![LogicalType::Int64],
    );
    assert_eq!(op.name(), "RemoveLabel");

    let op = SetPropertyOperator::new_for_node(
        Arc::clone(&store),
        Box::new(EmptyInput),
        0,
        vec![],
        vec![LogicalType::Int64],
    );
    assert_eq!(op.name(), "SetProperty");
}
