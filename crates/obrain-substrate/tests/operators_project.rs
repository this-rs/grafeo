//! Integration tests for `obrain_core::execution::operators::ProjectOperator`
//! against the substrate backend.
//!
//! Relocated from `crates/obrain-core/src/execution/operators/project.rs`'s
//! in-crate `#[cfg(test)] mod tests` block as part of T17 W4.p4. See note
//! tagged `t17 w4.p4 migration-pattern` for rationale — briefly, a
//! substrate-backed fixture cannot live inside `obrain-core` because the
//! dev-dep cycle (gotcha `598dda40-a186-4be3-97f3-c75053af4e6e`) produces
//! two distinct compilation units of `obrain-core`, breaking the
//! `Arc<SubstrateStore> as Arc<dyn GraphStore>` trait cast.
//!
//! Run with:
//!
//! ```bash
//! cargo test -p obrain-substrate --test operators_project
//! ```

use obrain_common::types::{LogicalType, NodeId, PropertyKey, Value};
use obrain_core::execution::chunk::{DataChunk, DataChunkBuilder};
use obrain_core::execution::operators::{Operator, OperatorResult, ProjectExpr, ProjectOperator};
use obrain_core::graph::traits::GraphStoreMut;
use obrain_substrate::SubstrateStore;
use std::sync::Arc;

struct MockScanOperator {
    chunks: Vec<DataChunk>,
    position: usize,
}

impl Operator for MockScanOperator {
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
        "MockScan"
    }
}

#[test]
fn test_project_select_columns() {
    // Create input with 3 columns: [int, string, int]
    let mut builder =
        DataChunkBuilder::new(&[LogicalType::Int64, LogicalType::String, LogicalType::Int64]);

    builder.column_mut(0).unwrap().push_int64(1);
    builder.column_mut(1).unwrap().push_string("hello");
    builder.column_mut(2).unwrap().push_int64(100);
    builder.advance_row();

    builder.column_mut(0).unwrap().push_int64(2);
    builder.column_mut(1).unwrap().push_string("world");
    builder.column_mut(2).unwrap().push_int64(200);
    builder.advance_row();

    let chunk = builder.finish();

    let mock_scan = MockScanOperator {
        chunks: vec![chunk],
        position: 0,
    };

    // Project to select columns 2 and 0 (reordering)
    let mut project = ProjectOperator::select_columns(
        Box::new(mock_scan),
        vec![2, 0],
        vec![LogicalType::Int64, LogicalType::Int64],
    );

    let result = project.next().unwrap().unwrap();

    assert_eq!(result.column_count(), 2);
    assert_eq!(result.row_count(), 2);

    // Check values are reordered
    assert_eq!(result.column(0).unwrap().get_int64(0), Some(100));
    assert_eq!(result.column(1).unwrap().get_int64(0), Some(1));
}

#[test]
fn test_project_constant() {
    let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
    builder.column_mut(0).unwrap().push_int64(1);
    builder.advance_row();
    builder.column_mut(0).unwrap().push_int64(2);
    builder.advance_row();

    let chunk = builder.finish();

    let mock_scan = MockScanOperator {
        chunks: vec![chunk],
        position: 0,
    };

    // Project with a constant
    let mut project = ProjectOperator::new(
        Box::new(mock_scan),
        vec![
            ProjectExpr::Column(0),
            ProjectExpr::Constant(Value::String("constant".into())),
        ],
        vec![LogicalType::Int64, LogicalType::String],
    );

    let result = project.next().unwrap().unwrap();

    assert_eq!(result.column_count(), 2);
    assert_eq!(result.column(1).unwrap().get_string(0), Some("constant"));
    assert_eq!(result.column(1).unwrap().get_string(1), Some("constant"));
}

#[test]
fn test_project_empty_input() {
    let mock_scan = MockScanOperator {
        chunks: vec![],
        position: 0,
    };

    let mut project =
        ProjectOperator::select_columns(Box::new(mock_scan), vec![0], vec![LogicalType::Int64]);

    assert!(project.next().unwrap().is_none());
}

#[test]
fn test_project_column_not_found() {
    let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
    builder.column_mut(0).unwrap().push_int64(1);
    builder.advance_row();
    let chunk = builder.finish();

    let mock_scan = MockScanOperator {
        chunks: vec![chunk],
        position: 0,
    };

    // Reference column index 5 which doesn't exist
    let mut project = ProjectOperator::new(
        Box::new(mock_scan),
        vec![ProjectExpr::Column(5)],
        vec![LogicalType::Int64],
    );

    let result = project.next();
    assert!(result.is_err(), "Should fail with ColumnNotFound");
}

#[test]
fn test_project_multiple_constants() {
    let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
    builder.column_mut(0).unwrap().push_int64(1);
    builder.advance_row();
    let chunk = builder.finish();

    let mock_scan = MockScanOperator {
        chunks: vec![chunk],
        position: 0,
    };

    let mut project = ProjectOperator::new(
        Box::new(mock_scan),
        vec![
            ProjectExpr::Constant(Value::Int64(42)),
            ProjectExpr::Constant(Value::String("fixed".into())),
            ProjectExpr::Constant(Value::Bool(true)),
        ],
        vec![LogicalType::Int64, LogicalType::String, LogicalType::Bool],
    );

    let result = project.next().unwrap().unwrap();
    assert_eq!(result.column_count(), 3);
    assert_eq!(result.column(0).unwrap().get_int64(0), Some(42));
    assert_eq!(result.column(1).unwrap().get_string(0), Some("fixed"));
    assert_eq!(
        result.column(2).unwrap().get_value(0),
        Some(Value::Bool(true))
    );
}

#[test]
fn test_project_identity() {
    // Select all columns in original order
    let mut builder = DataChunkBuilder::new(&[LogicalType::Int64, LogicalType::String]);
    builder.column_mut(0).unwrap().push_int64(10);
    builder.column_mut(1).unwrap().push_string("test");
    builder.advance_row();
    let chunk = builder.finish();

    let mock_scan = MockScanOperator {
        chunks: vec![chunk],
        position: 0,
    };

    let mut project = ProjectOperator::select_columns(
        Box::new(mock_scan),
        vec![0, 1],
        vec![LogicalType::Int64, LogicalType::String],
    );

    let result = project.next().unwrap().unwrap();
    assert_eq!(result.column(0).unwrap().get_int64(0), Some(10));
    assert_eq!(result.column(1).unwrap().get_string(0), Some("test"));
}

#[test]
fn test_project_name() {
    let mock_scan = MockScanOperator {
        chunks: vec![],
        position: 0,
    };
    let project =
        ProjectOperator::select_columns(Box::new(mock_scan), vec![0], vec![LogicalType::Int64]);
    assert_eq!(project.name(), "Project");
}

#[test]
fn test_project_node_resolve() {
    // Create a store with a test node
    let store = SubstrateStore::open_tempfile().unwrap();
    let node_id = store.create_node(&["Person"]);
    store.set_node_property(node_id, "name", Value::String("Alix".into()));
    store.set_node_property(node_id, "age", Value::Int64(30));

    // Create input chunk with a NodeId column
    let mut builder = DataChunkBuilder::new(&[LogicalType::Node]);
    builder.column_mut(0).unwrap().push_node_id(node_id);
    builder.advance_row();
    let chunk = builder.finish();

    let mock_scan = MockScanOperator {
        chunks: vec![chunk],
        position: 0,
    };

    let mut project = ProjectOperator::with_store(
        Box::new(mock_scan),
        vec![ProjectExpr::NodeResolve { column: 0 }],
        vec![LogicalType::Any],
        Arc::new(store),
    );

    let result = project.next().unwrap().unwrap();
    assert_eq!(result.column_count(), 1);

    let value = result.column(0).unwrap().get_value(0).unwrap();
    if let Value::Map(map) = value {
        assert_eq!(
            map.get(&PropertyKey::new("_id")),
            Some(&Value::Int64(node_id.as_u64() as i64))
        );
        assert!(map.get(&PropertyKey::new("_labels")).is_some());
        assert_eq!(
            map.get(&PropertyKey::new("name")),
            Some(&Value::String("Alix".into()))
        );
        assert_eq!(map.get(&PropertyKey::new("age")), Some(&Value::Int64(30)));
    } else {
        panic!("Expected Value::Map, got {:?}", value);
    }
}

#[test]
fn test_project_edge_resolve() {
    let store = SubstrateStore::open_tempfile().unwrap();
    let src = store.create_node(&["Person"]);
    let dst = store.create_node(&["Company"]);
    let edge_id = store.create_edge(src, dst, "WORKS_AT");
    store.set_edge_property(edge_id, "since", Value::Int64(2020));

    // Create input chunk with an EdgeId column
    let mut builder = DataChunkBuilder::new(&[LogicalType::Edge]);
    builder.column_mut(0).unwrap().push_edge_id(edge_id);
    builder.advance_row();
    let chunk = builder.finish();

    let mock_scan = MockScanOperator {
        chunks: vec![chunk],
        position: 0,
    };

    let mut project = ProjectOperator::with_store(
        Box::new(mock_scan),
        vec![ProjectExpr::EdgeResolve { column: 0 }],
        vec![LogicalType::Any],
        Arc::new(store),
    );

    let result = project.next().unwrap().unwrap();
    let value = result.column(0).unwrap().get_value(0).unwrap();
    if let Value::Map(map) = value {
        assert_eq!(
            map.get(&PropertyKey::new("_id")),
            Some(&Value::Int64(edge_id.as_u64() as i64))
        );
        assert_eq!(
            map.get(&PropertyKey::new("_type")),
            Some(&Value::String("WORKS_AT".into()))
        );
        assert_eq!(
            map.get(&PropertyKey::new("_source")),
            Some(&Value::Int64(src.as_u64() as i64))
        );
        assert_eq!(
            map.get(&PropertyKey::new("_target")),
            Some(&Value::Int64(dst.as_u64() as i64))
        );
        assert_eq!(
            map.get(&PropertyKey::new("since")),
            Some(&Value::Int64(2020))
        );
    } else {
        panic!("Expected Value::Map, got {:?}", value);
    }
}

#[test]
fn test_project_resolve_missing_entity() {
    // NodeId(999) does not exist in a freshly-opened substrate; the resolver
    // must surface Value::Null without panicking. (Substrate reserves NodeId(0)
    // as a sentinel and allocates from NodeId(1) upward, so any id ≥ a few is
    // guaranteed unassigned on an empty store.)
    let store = SubstrateStore::open_tempfile().unwrap();

    // Create input chunk with a NodeId that doesn't exist in the store
    let mut builder = DataChunkBuilder::new(&[LogicalType::Node]);
    builder
        .column_mut(0)
        .unwrap()
        .push_node_id(NodeId::new(999));
    builder.advance_row();
    let chunk = builder.finish();

    let mock_scan = MockScanOperator {
        chunks: vec![chunk],
        position: 0,
    };

    let mut project = ProjectOperator::with_store(
        Box::new(mock_scan),
        vec![ProjectExpr::NodeResolve { column: 0 }],
        vec![LogicalType::Any],
        Arc::new(store),
    );

    let result = project.next().unwrap().unwrap();
    assert_eq!(result.column(0).unwrap().get_value(0), Some(Value::Null));
}
