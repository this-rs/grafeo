//! Project operator for selecting and transforming columns.

use super::filter::{ExpressionPredicate, FilterExpression, SessionContext};
use super::{Operator, OperatorError, OperatorResult};
use crate::execution::DataChunk;
use crate::graph::GraphStore;
use crate::graph::lpg::{Edge, Node};
use obrain_common::types::{EpochId, LogicalType, PropertyKey, TransactionId, Value};
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
    store: Option<Arc<dyn GraphStore>>,
    /// Transaction ID for MVCC-aware property lookups.
    transaction_id: Option<TransactionId>,
    /// Viewing epoch for MVCC-aware property lookups.
    viewing_epoch: Option<EpochId>,
    /// Session context for introspection functions in expression evaluation.
    session_context: SessionContext,
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
            transaction_id: None,
            viewing_epoch: None,
            session_context: SessionContext::default(),
        }
    }

    /// Creates a new project operator with store access for property lookups.
    pub fn with_store(
        child: Box<dyn Operator>,
        projections: Vec<ProjectExpr>,
        output_types: Vec<LogicalType>,
        store: Arc<dyn GraphStore>,
    ) -> Self {
        assert_eq!(projections.len(), output_types.len());
        Self {
            child,
            projections,
            output_types,
            store: Some(store),
            transaction_id: None,
            viewing_epoch: None,
            session_context: SessionContext::default(),
        }
    }

    /// Sets the transaction context for MVCC-aware property lookups.
    pub fn with_transaction_context(
        mut self,
        epoch: EpochId,
        transaction_id: Option<TransactionId>,
    ) -> Self {
        self.viewing_epoch = Some(epoch);
        self.transaction_id = transaction_id;
        self
    }

    /// Sets the session context for introspection functions.
    pub fn with_session_context(mut self, context: SessionContext) -> Self {
        self.session_context = context;
        self
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

                    // Extract property for each row.
                    // For typed columns (VectorData::NodeId / EdgeId) there is
                    // no ambiguity. For Generic/Any columns (e.g. after a hash
                    // join), both get_node_id and get_edge_id can succeed on the
                    // same Int64 value, so we verify against the store to resolve
                    // the entity type.
                    let prop_key = PropertyKey::new(property);
                    let epoch = self.viewing_epoch;
                    let tx_id = self.transaction_id;
                    // T17k: single-property projection. Use the scalar
                    // `store.get_node_property` / `get_edge_property`
                    // directly — they do a targeted per-zone lookup
                    // (O(1)) rather than the O(N) bulk hydrate that
                    // `get_node` performs (needed for bulk consumers
                    // like merge / RAG but wasteful for N-row × 1-prop
                    // projections). MVCC-aware paths still route
                    // through `get_node_at_epoch` for historical reads.
                    let has_mvcc = epoch.is_some();
                    for row in input.selected_indices() {
                        let value = if let Some(node_id) = input_col.get_node_id(row) {
                            let prop = if has_mvcc {
                                let node = if let (Some(ep), Some(tx)) = (epoch, tx_id) {
                                    store.get_node_versioned(node_id, ep, tx)
                                } else if let Some(ep) = epoch {
                                    store.get_node_at_epoch(node_id, ep)
                                } else {
                                    None
                                };
                                node.and_then(|n| n.get_property(property).cloned())
                            } else {
                                store.get_node_property(node_id, &prop_key)
                            };
                            if let Some(v) = prop {
                                v
                            } else if let Some(edge_id) = input_col.get_edge_id(row) {
                                if has_mvcc {
                                    let edge = if let (Some(ep), Some(tx)) = (epoch, tx_id) {
                                        store.get_edge_versioned(edge_id, ep, tx)
                                    } else if let Some(ep) = epoch {
                                        store.get_edge_at_epoch(edge_id, ep)
                                    } else {
                                        None
                                    };
                                    edge.and_then(|e| e.get_property(property).cloned())
                                        .unwrap_or(Value::Null)
                                } else {
                                    store
                                        .get_edge_property(edge_id, &prop_key)
                                        .unwrap_or(Value::Null)
                                }
                            } else {
                                Value::Null
                            }
                        } else if let Some(edge_id) = input_col.get_edge_id(row) {
                            if has_mvcc {
                                let edge = if let (Some(ep), Some(tx)) = (epoch, tx_id) {
                                    store.get_edge_versioned(edge_id, ep, tx)
                                } else if let Some(ep) = epoch {
                                    store.get_edge_at_epoch(edge_id, ep)
                                } else {
                                    None
                                };
                                edge.and_then(|e| e.get_property(property).cloned())
                                    .unwrap_or(Value::Null)
                            } else {
                                store
                                    .get_edge_property(edge_id, &prop_key)
                                    .unwrap_or(Value::Null)
                            }
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

                    let epoch = self.viewing_epoch;
                    let tx_id = self.transaction_id;
                    for row in input.selected_indices() {
                        let value = if let Some(edge_id) = input_col.get_edge_id(row) {
                            let etype = if let (Some(ep), Some(tx)) = (epoch, tx_id) {
                                store.edge_type_versioned(edge_id, ep, tx)
                            } else {
                                store.edge_type(edge_id)
                            };
                            etype.map_or(Value::Null, Value::String)
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
                    let mut evaluator = ExpressionPredicate::new(
                        expr.clone(),
                        variable_columns.clone(),
                        Arc::clone(store),
                    )
                    .with_session_context(self.session_context.clone());
                    if let (Some(ep), tx_id) = (self.viewing_epoch, self.transaction_id) {
                        evaluator = evaluator.with_transaction_context(ep, tx_id);
                    }

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

                    let epoch = self.viewing_epoch;
                    let tx_id = self.transaction_id;
                    for row in input.selected_indices() {
                        let value = if let Some(node_id) = input_col.get_node_id(row) {
                            let node = if let (Some(ep), Some(tx)) = (epoch, tx_id) {
                                store.get_node_versioned(node_id, ep, tx)
                            } else if let Some(ep) = epoch {
                                store.get_node_at_epoch(node_id, ep)
                            } else {
                                store.get_node(node_id)
                            };
                            node.map_or(Value::Null, |n| node_to_map(&n))
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

                    let epoch = self.viewing_epoch;
                    let tx_id = self.transaction_id;
                    for row in input.selected_indices() {
                        let value = if let Some(edge_id) = input_col.get_edge_id(row) {
                            let edge = if let (Some(ep), Some(tx)) = (epoch, tx_id) {
                                store.get_edge_versioned(edge_id, ep, tx)
                            } else if let Some(ep) = epoch {
                                store.get_edge_at_epoch(edge_id, ep)
                            } else {
                                store.get_edge(edge_id)
                            };
                            edge.map_or(Value::Null, |e| edge_to_map(&e))
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

// NOTE (T17 W4.p4 closure, 2026-04-23): the in-crate `#[cfg(test)] mod tests`
// block that used to live here has been relocated to
// `crates/obrain-substrate/tests/operators_project.rs`. See gotcha
// `598dda40-a186-4be3-97f3-c75053af4e6e` and decision
// `0a84cf59-b7d7-4a60-9d62-aa00eec820a3` for the class-2 migration path.
