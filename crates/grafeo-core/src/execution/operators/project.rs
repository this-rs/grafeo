//! Project operator for selecting and transforming columns.

use super::filter::{ExpressionPredicate, FilterExpression};
use super::{Operator, OperatorError, OperatorResult};
use crate::execution::DataChunk;
use crate::graph::lpg::{Edge, LpgStore, Node};
use grafeo_common::types::{LogicalType, PropertyKey, Value};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

/// A projection expression.
pub enum ProjectExpr {
    /// Reference to an input column.
    Column(usize),
    /// A constant value.
    Constant(Value),
    /// Property access on a node/edge column.
    PropertyAccess {
        /// The column containing the node or edge ID.
        column: usize,
        /// The property name to access.
        property: String,
    },
    /// Edge type accessor (for type(r) function).
    EdgeType {
        /// The column containing the edge ID.
        column: usize,
    },
    /// Full expression evaluation (for CASE WHEN, etc.).
    Expression {
        /// The filter expression to evaluate.
        expr: FilterExpression,
        /// Variable name to column index mapping.
        variable_columns: HashMap<String, usize>,
    },
    /// Resolve a node ID column to a full node map with metadata and properties.
    NodeResolve {
        /// The column containing the node ID.
        column: usize,
    },
    /// Resolve an edge ID column to a full edge map with metadata and properties.
    EdgeResolve {
        /// The column containing the edge ID.
        column: usize,
    },
}

/// A project operator that selects and transforms columns.
pub struct ProjectOperator {
    /// Child operator to read from.
    child: Box<dyn Operator>,
    /// Projection expressions.
    projections: Vec<ProjectExpr>,
    /// Output column types.
    output_types: Vec<LogicalType>,
    /// Optional store for property access.
    store: Option<Arc<LpgStore>>,
}

impl ProjectOperator {
    /// Creates a new project operator.
    pub fn new(
        child: Box<dyn Operator>,
        projections: Vec<ProjectExpr>,
        output_types: Vec<LogicalType>,
    ) -> Self {
        assert_eq!(projections.len(), output_types.len());
        Self {
            child,
            projections,
            output_types,
            store: None,
        }
    }

    /// Creates a new project operator with store access for property lookups.
    pub fn with_store(
        child: Box<dyn Operator>,
        projections: Vec<ProjectExpr>,
        output_types: Vec<LogicalType>,
        store: Arc<LpgStore>,
    ) -> Self {
        assert_eq!(projections.len(), output_types.len());
        Self {
            child,
            projections,
            output_types,
            store: Some(store),
        }
    }

    /// Creates a project operator that selects specific columns.
    pub fn select_columns(
        child: Box<dyn Operator>,
        columns: Vec<usize>,
        types: Vec<LogicalType>,
    ) -> Self {
        let projections = columns.into_iter().map(ProjectExpr::Column).collect();
        Self::new(child, projections, types)
    }
}

impl Operator for ProjectOperator {
    fn next(&mut self) -> OperatorResult {
        // Get next chunk from child
        let Some(input) = self.child.next()? else {
            return Ok(None);
        };

        // Create output chunk
        let mut output = DataChunk::with_capacity(&self.output_types, input.row_count());

        // Evaluate each projection
        for (i, proj) in self.projections.iter().enumerate() {
            match proj {
                ProjectExpr::Column(col_idx) => {
                    // Copy column from input to output
                    let input_col = input.column(*col_idx).ok_or_else(|| {
                        OperatorError::ColumnNotFound(format!("Column {col_idx}"))
                    })?;

                    let output_col = output
                        .column_mut(i)
                        .expect("column exists: index matches projection schema");

                    // Copy selected rows
                    for row in input.selected_indices() {
                        if let Some(value) = input_col.get_value(row) {
                            output_col.push_value(value);
                        }
                    }
                }
                ProjectExpr::Constant(value) => {
                    // Push constant for each row
                    let output_col = output
                        .column_mut(i)
                        .expect("column exists: index matches projection schema");
                    for _ in input.selected_indices() {
                        output_col.push_value(value.clone());
                    }
                }
                ProjectExpr::PropertyAccess { column, property } => {
                    // Access property from node/edge in the specified column
                    let input_col = input
                        .column(*column)
                        .ok_or_else(|| OperatorError::ColumnNotFound(format!("Column {column}")))?;

                    let output_col = output
                        .column_mut(i)
                        .expect("column exists: index matches projection schema");

                    let store = self.store.as_ref().ok_or_else(|| {
                        OperatorError::Execution("Store required for property access".to_string())
                    })?;

                    // Extract property for each row
                    let prop_key = PropertyKey::new(property);
                    for row in input.selected_indices() {
                        // Try node ID, then edge ID, then map value (for UNWIND maps)
                        let value = if let Some(node_id) = input_col.get_node_id(row) {
                            store
                                .get_node(node_id)
                                .and_then(|node| node.get_property(property).cloned())
                                .unwrap_or(Value::Null)
                        } else if let Some(edge_id) = input_col.get_edge_id(row) {
                            store
                                .get_edge(edge_id)
                                .and_then(|edge| edge.get_property(property).cloned())
                                .unwrap_or(Value::Null)
                        } else if let Some(Value::Map(map)) = input_col.get_value(row) {
                            map.get(&prop_key).cloned().unwrap_or(Value::Null)
                        } else {
                            Value::Null
                        };
                        output_col.push_value(value);
                    }
                }
                ProjectExpr::EdgeType { column } => {
                    // Get edge type string from an edge column
                    let input_col = input
                        .column(*column)
                        .ok_or_else(|| OperatorError::ColumnNotFound(format!("Column {column}")))?;

                    let output_col = output
                        .column_mut(i)
                        .expect("column exists: index matches projection schema");

                    let store = self.store.as_ref().ok_or_else(|| {
                        OperatorError::Execution("Store required for edge type access".to_string())
                    })?;

                    for row in input.selected_indices() {
                        let value = if let Some(edge_id) = input_col.get_edge_id(row) {
                            store.edge_type(edge_id).map_or(Value::Null, Value::String)
                        } else {
                            Value::Null
                        };
                        output_col.push_value(value);
                    }
                }
                ProjectExpr::Expression {
                    expr,
                    variable_columns,
                } => {
                    let output_col = output
                        .column_mut(i)
                        .expect("column exists: index matches projection schema");

                    let store = self.store.as_ref().ok_or_else(|| {
                        OperatorError::Execution(
                            "Store required for expression evaluation".to_string(),
                        )
                    })?;

                    // Use the ExpressionPredicate for expression evaluation
                    let evaluator = ExpressionPredicate::new(
                        expr.clone(),
                        variable_columns.clone(),
                        Arc::clone(store),
                    );

                    for row in input.selected_indices() {
                        let value = evaluator.eval_at(&input, row).unwrap_or(Value::Null);
                        output_col.push_value(value);
                    }
                }
                ProjectExpr::NodeResolve { column } => {
                    let input_col = input
                        .column(*column)
                        .ok_or_else(|| OperatorError::ColumnNotFound(format!("Column {column}")))?;

                    let output_col = output
                        .column_mut(i)
                        .expect("column exists: index matches projection schema");

                    let store = self.store.as_ref().ok_or_else(|| {
                        OperatorError::Execution("Store required for node resolution".to_string())
                    })?;

                    for row in input.selected_indices() {
                        let value = if let Some(node_id) = input_col.get_node_id(row) {
                            store
                                .get_node(node_id)
                                .map_or(Value::Null, |n| node_to_map(&n))
                        } else {
                            Value::Null
                        };
                        output_col.push_value(value);
                    }
                }
                ProjectExpr::EdgeResolve { column } => {
                    let input_col = input
                        .column(*column)
                        .ok_or_else(|| OperatorError::ColumnNotFound(format!("Column {column}")))?;

                    let output_col = output
                        .column_mut(i)
                        .expect("column exists: index matches projection schema");

                    let store = self.store.as_ref().ok_or_else(|| {
                        OperatorError::Execution("Store required for edge resolution".to_string())
                    })?;

                    for row in input.selected_indices() {
                        let value = if let Some(edge_id) = input_col.get_edge_id(row) {
                            store
                                .get_edge(edge_id)
                                .map_or(Value::Null, |e| edge_to_map(&e))
                        } else {
                            Value::Null
                        };
                        output_col.push_value(value);
                    }
                }
            }
        }

        output.set_count(input.row_count());
        Ok(Some(output))
    }

    fn reset(&mut self) {
        self.child.reset();
    }

    fn name(&self) -> &'static str {
        "Project"
    }
}

/// Converts a [`Node`] to a `Value::Map` with metadata and properties.
///
/// The map contains `_id` (integer), `_labels` (list of strings), and
/// all node properties at the top level.
fn node_to_map(node: &Node) -> Value {
    let mut map = BTreeMap::new();
    map.insert(
        PropertyKey::new("_id"),
        Value::Int64(node.id.as_u64() as i64),
    );
    let labels: Vec<Value> = node
        .labels
        .iter()
        .map(|l| Value::String(l.clone()))
        .collect();
    map.insert(PropertyKey::new("_labels"), Value::List(labels.into()));
    for (key, value) in &node.properties {
        map.insert(key.clone(), value.clone());
    }
    Value::Map(Arc::new(map))
}

/// Converts an [`Edge`] to a `Value::Map` with metadata and properties.
///
/// The map contains `_id`, `_type`, `_source`, `_target`, and all edge
/// properties at the top level.
fn edge_to_map(edge: &Edge) -> Value {
    let mut map = BTreeMap::new();
    map.insert(
        PropertyKey::new("_id"),
        Value::Int64(edge.id.as_u64() as i64),
    );
    map.insert(
        PropertyKey::new("_type"),
        Value::String(edge.edge_type.clone()),
    );
    map.insert(
        PropertyKey::new("_source"),
        Value::Int64(edge.src.as_u64() as i64),
    );
    map.insert(
        PropertyKey::new("_target"),
        Value::Int64(edge.dst.as_u64() as i64),
    );
    for (key, value) in &edge.properties {
        map.insert(key.clone(), value.clone());
    }
    Value::Map(Arc::new(map))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::chunk::DataChunkBuilder;
    use grafeo_common::types::Value;

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
        let store = LpgStore::new();
        let node_id = store.create_node(&["Person"]);
        store.set_node_property(node_id, "name", Value::String("Alice".into()));
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
                Some(&Value::String("Alice".into()))
            );
            assert_eq!(map.get(&PropertyKey::new("age")), Some(&Value::Int64(30)));
        } else {
            panic!("Expected Value::Map, got {:?}", value);
        }
    }

    #[test]
    fn test_project_edge_resolve() {
        let store = LpgStore::new();
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
        use grafeo_common::types::NodeId;

        let store = LpgStore::new();

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
}
