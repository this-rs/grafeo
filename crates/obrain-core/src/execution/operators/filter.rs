//! Filter operator for applying predicates.

use super::{Operator, OperatorResult};
use crate::execution::{ChunkZoneHints, DataChunk, SelectionVector};
use crate::graph::Direction;
use crate::graph::GraphStore;
use crate::graph::lpg::{Edge, Node};
use obrain_common::types::{EpochId, HashableValue, NodeId, PropertyKey, TransactionId, Value};
#[cfg(feature = "regex")]
use regex::Regex;
#[cfg(all(feature = "regex-lite", not(feature = "regex")))]
use regex_lite::Regex;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

/// Extracts a required integer field from a temporal constructor map.
fn map_int(m: &BTreeMap<PropertyKey, Value>, key: &str) -> Option<i64> {
    match m.get(&PropertyKey::from(key))? {
        Value::Int64(v) => Some(*v),
        Value::Float64(f) => Some(*f as i64),
        _ => None,
    }
}

/// Extracts an optional integer field from a temporal constructor map,
/// returning a default value when the key is absent.
fn map_int_or(m: &BTreeMap<PropertyKey, Value>, key: &str, default: i64) -> i64 {
    map_int(m, key).unwrap_or(default)
}

/// A predicate for filtering rows.
pub trait Predicate: Send + Sync {
    /// Evaluates the predicate for a single row.
    fn evaluate(&self, chunk: &DataChunk, row: usize) -> bool;

    /// Returns `false` if zone map proves no rows in this chunk can match.
    ///
    /// This method enables chunk-level filtering optimization. When a chunk
    /// has zone map hints attached, the filter operator calls this method
    /// first. If it returns `false`, the entire chunk is skipped without
    /// evaluating any rows.
    ///
    /// The default implementation is conservative and returns `true` (might match).
    /// Predicates that support zone map checking should override this.
    fn might_match_chunk(&self, _hints: &ChunkZoneHints) -> bool {
        true
    }
}

// The `CompareOp` enum and `ComparisonPredicate` struct that used to live
// here (as `#[cfg(test)] pub(crate)` helpers) have been relocated to
// `crates/obrain-substrate/tests/operators_filter.rs` as part of T17 W4.p4
// — substrate-backed fixtures cannot live in `obrain-core` due to the
// dev-dep cycle. They were never part of the crate's public or internal API.

/// An expression-based predicate that evaluates logical expressions.
///
/// This predicate can evaluate complex expressions involving variables,
/// properties, and operators.
pub struct ExpressionPredicate {
    /// The expression to evaluate.
    expression: FilterExpression,
    /// Map from variable name to column index.
    variable_columns: HashMap<String, usize>,
    /// The graph store for property lookups.
    store: Arc<dyn GraphStore>,
    /// Transaction ID for MVCC-aware lookups.
    transaction_id: Option<TransactionId>,
    /// Viewing epoch for MVCC-aware lookups.
    viewing_epoch: Option<EpochId>,
    /// Session context for introspection functions (info, schema, current_schema, etc.).
    session_context: SessionContext,
}

/// A lazily-computed, cloneable value.
///
/// The factory runs at most once (via `OnceLock`). Cloning is cheap because
/// both the lock and the factory are behind `Arc`.
#[derive(Clone)]
pub struct LazyValue {
    cell: Arc<std::sync::OnceLock<Value>>,
    factory: Arc<dyn Fn() -> Value + Send + Sync>,
}

impl LazyValue {
    /// Creates a new lazy value with the given factory.
    pub fn new(factory: impl Fn() -> Value + Send + Sync + 'static) -> Self {
        Self {
            cell: Arc::new(std::sync::OnceLock::new()),
            factory: Arc::new(factory),
        }
    }

    /// Returns the value, computing it on first access.
    pub fn get(&self) -> &Value {
        self.cell.get_or_init(|| (self.factory)())
    }
}

impl std::fmt::Debug for LazyValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.cell.get() {
            Some(v) => write!(f, "LazyValue({v:?})"),
            None => write!(f, "LazyValue(<not yet computed>)"),
        }
    }
}

impl Default for LazyValue {
    fn default() -> Self {
        Self::new(|| Value::Null)
    }
}

/// Session-level context passed to the filter evaluator for introspection functions.
///
/// Lightweight strings (`current_schema`, `current_graph`) are stored directly.
/// Expensive introspection maps (`db_info`, `schema_info`) are lazily computed
/// on first access, so queries that never call `info()` or `schema()` pay zero cost.
#[derive(Debug, Clone, Default)]
pub struct SessionContext {
    /// Current session schema name (for `CURRENT_SCHEMA`).
    pub current_schema: Option<String>,
    /// Current session graph name (for `CURRENT_GRAPH`).
    pub current_graph: Option<String>,
    /// Lazily-computed `info()` result.
    pub db_info: LazyValue,
    /// Lazily-computed `schema()` result.
    pub schema_info: LazyValue,
}

/// A filter expression that can be evaluated.
#[derive(Debug, Clone)]
pub enum FilterExpression {
    /// A literal value.
    Literal(Value),
    /// A variable reference (column index).
    Variable(String),
    /// Property access on a variable.
    Property {
        /// The variable name.
        variable: String,
        /// The property name.
        property: String,
    },
    /// Binary operation.
    Binary {
        /// Left operand.
        left: Box<FilterExpression>,
        /// Operator.
        op: BinaryFilterOp,
        /// Right operand.
        right: Box<FilterExpression>,
    },
    /// Unary operation.
    Unary {
        /// Operator.
        op: UnaryFilterOp,
        /// Operand.
        operand: Box<FilterExpression>,
    },
    /// Function call.
    FunctionCall {
        /// Function name (e.g., "id", "labels", "type", "size", "coalesce", "exists").
        name: String,
        /// Arguments.
        args: Vec<FilterExpression>,
    },
    /// List literal.
    List(Vec<FilterExpression>),
    /// Map literal (e.g., {name: 'Alix', age: 30}).
    Map(Vec<(String, FilterExpression)>),
    /// Index access (e.g., `list[0]`).
    IndexAccess {
        /// The base expression.
        base: Box<FilterExpression>,
        /// The index expression.
        index: Box<FilterExpression>,
    },
    /// Slice access (e.g., list[1..3]).
    SliceAccess {
        /// The base expression.
        base: Box<FilterExpression>,
        /// Start index (None means from beginning).
        start: Option<Box<FilterExpression>>,
        /// End index (None means to end).
        end: Option<Box<FilterExpression>>,
    },
    /// CASE expression.
    Case {
        /// Test expression (for simple CASE).
        operand: Option<Box<FilterExpression>>,
        /// WHEN clauses (condition, result).
        when_clauses: Vec<(FilterExpression, FilterExpression)>,
        /// ELSE clause.
        else_clause: Option<Box<FilterExpression>>,
    },
    /// Entity ID access.
    Id(String),
    /// Node labels access.
    Labels(String),
    /// Edge type access.
    Type(String),
    /// List comprehension: [x IN list WHERE predicate | expression]
    ListComprehension {
        /// Variable name for each element.
        variable: String,
        /// The source list expression.
        list_expr: Box<FilterExpression>,
        /// Optional filter predicate.
        filter_expr: Option<Box<FilterExpression>>,
        /// The mapping expression for each element.
        map_expr: Box<FilterExpression>,
    },
    /// List predicate: all/any/none/single(x IN list WHERE pred).
    ListPredicate {
        /// The kind of list predicate.
        kind: ListPredicateKind,
        /// The iteration variable name.
        variable: String,
        /// The source list expression.
        list_expr: Box<FilterExpression>,
        /// The predicate to test for each element.
        predicate: Box<FilterExpression>,
    },
    /// EXISTS subquery: evaluates inner plan and returns true if results exist.
    ExistsSubquery {
        /// The start node variable from outer query.
        start_var: String,
        /// Direction of edge traversal.
        direction: Direction,
        /// Edge type filter (empty = match all types, multiple = match any).
        edge_types: Vec<String>,
        /// Optional end node labels filter.
        end_labels: Option<Vec<String>>,
        /// Minimum number of hops (for variable-length patterns).
        min_hops: Option<u32>,
        /// Maximum number of hops (for variable-length patterns).
        max_hops: Option<u32>,
    },
    /// COUNT subquery: counts matching edges from a node (fast path).
    CountSubquery {
        /// The start node variable from outer query.
        start_var: String,
        /// Direction of edge traversal.
        direction: Direction,
        /// Edge type filter (empty = match all types, multiple = match any).
        edge_types: Vec<String>,
    },
    /// reduce() accumulator: `reduce(acc = init, x IN list | expr)`.
    Reduce {
        /// Accumulator variable name.
        accumulator: String,
        /// Initial value for the accumulator.
        initial: Box<FilterExpression>,
        /// Iteration variable name.
        variable: String,
        /// List to iterate over.
        list: Box<FilterExpression>,
        /// Body expression (references both accumulator and variable).
        expression: Box<FilterExpression>,
    },
}

/// The kind of list predicate function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ListPredicateKind {
    /// all(x IN list WHERE pred): true if pred holds for every element.
    All,
    /// any(x IN list WHERE pred): true if pred holds for at least one element.
    Any,
    /// none(x IN list WHERE pred): true if pred holds for no element.
    None,
    /// single(x IN list WHERE pred): true if pred holds for exactly one element.
    Single,
}

/// Binary operators for filter expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryFilterOp {
    /// Equal.
    Eq,
    /// Not equal.
    Ne,
    /// Less than.
    Lt,
    /// Less than or equal.
    Le,
    /// Greater than.
    Gt,
    /// Greater than or equal.
    Ge,
    /// Logical AND.
    And,
    /// Logical OR.
    Or,
    /// Logical XOR.
    Xor,
    /// Addition.
    Add,
    /// Subtraction.
    Sub,
    /// Multiplication.
    Mul,
    /// Division.
    Div,
    /// Modulo.
    Mod,
    /// String starts with.
    StartsWith,
    /// String ends with.
    EndsWith,
    /// String contains.
    Contains,
    /// List membership.
    In,
    /// Regex match (=~).
    Regex,
    /// Power/exponentiation (^).
    Pow,
    /// SQL LIKE pattern matching (% = any chars, _ = single char).
    Like,
    /// String concatenation (||).
    Concat,
}

/// Unary operators for filter expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryFilterOp {
    /// Logical NOT.
    Not,
    /// IS NULL.
    IsNull,
    /// IS NOT NULL.
    IsNotNull,
    /// Numeric negation.
    Neg,
}

impl ExpressionPredicate {
    /// Creates a new expression predicate.
    pub fn new(
        expression: FilterExpression,
        variable_columns: HashMap<String, usize>,
        store: Arc<dyn GraphStore>,
    ) -> Self {
        Self {
            expression,
            variable_columns,
            store,
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

    /// Resolves a node using transaction-aware access when available.
    fn resolve_node(&self, node_id: NodeId) -> Option<Node> {
        if let (Some(ep), Some(tx)) = (self.viewing_epoch, self.transaction_id) {
            self.store.get_node_versioned(node_id, ep, tx)
        } else if let Some(ep) = self.viewing_epoch {
            self.store.get_node_at_epoch(node_id, ep)
        } else {
            self.store.get_node(node_id)
        }
    }

    /// Resolves an edge using transaction-aware access when available.
    fn resolve_edge(&self, edge_id: obrain_common::types::EdgeId) -> Option<Edge> {
        if let (Some(ep), Some(tx)) = (self.viewing_epoch, self.transaction_id) {
            self.store.get_edge_versioned(edge_id, ep, tx)
        } else if let Some(ep) = self.viewing_epoch {
            self.store.get_edge_at_epoch(edge_id, ep)
        } else {
            self.store.get_edge(edge_id)
        }
    }

    /// Evaluates the expression for a specific row in a chunk, returning the result value.
    /// This is useful for evaluating expressions in contexts like RETURN clauses.
    pub fn eval_at(&self, chunk: &DataChunk, row: usize) -> Option<Value> {
        self.eval_expr(&self.expression, chunk, row)
    }

    /// Evaluates the expression for a row, returning the result value.
    fn eval(&self, chunk: &DataChunk, row: usize) -> Option<Value> {
        self.eval_expr(&self.expression, chunk, row)
    }

    fn eval_expr(&self, expr: &FilterExpression, chunk: &DataChunk, row: usize) -> Option<Value> {
        match expr {
            FilterExpression::Literal(v) => Some(v.clone()),
            FilterExpression::Variable(name) => {
                let col_idx = *self.variable_columns.get(name)?;
                chunk.column(col_idx)?.get_value(row)
            }
            FilterExpression::Property { variable, property } => {
                let col_idx = *self.variable_columns.get(variable)?;
                let col = chunk.column(col_idx)?;
                // T17k : single-property access. Route directly through
                // `store.get_node_property` / `get_edge_property` which
                // do a targeted read (O(1) mmap lookup per zone) instead
                // of going through `get_node` which bulk-hydrates all
                // ~N properties of the entity (O(N) lock-per-column).
                // On wide entities (File with 57 cols), bulk hydration
                // per-row turns a 21 ms query into 1.4 s. Scalar route
                // is ~1 μs per cell.
                let key = obrain_common::types::PropertyKey::new(property);
                if let Some(node_id) = col.get_node_id(row) {
                    // Fast path : direct scalar lookup on the store.
                    if let Some(v) = self.store.get_node_property(node_id, &key) {
                        return Some(v);
                    }
                    // Fall through to MVCC-aware path only when the
                    // scalar store misses (lets `get_node_at_epoch` kick
                    // in for historical reads).
                    if self.viewing_epoch.is_some()
                        && let Some(node) = self.resolve_node(node_id)
                    {
                        return node.get_property(property).cloned();
                    }
                    return None;
                }
                if let Some(edge_id) = col.get_edge_id(row) {
                    if let Some(v) = self.store.get_edge_property(edge_id, &key) {
                        return Some(v);
                    }
                    if self.viewing_epoch.is_some()
                        && let Some(edge) = self.resolve_edge(edge_id)
                    {
                        return edge.get_property(property).cloned();
                    }
                    return None;
                }
                // Try as map value (e.g. from UNWIND with map elements)
                if let Some(Value::Map(map)) = col.get_value(row) {
                    return map.get(&key).cloned();
                }
                None
            }
            FilterExpression::Binary { left, op, right } => {
                // For IN operator, right side is a list that we evaluate specially
                if *op == BinaryFilterOp::In {
                    let left_val = self.eval_expr(left, chunk, row)?;
                    return self.eval_in_operator(&left_val, right, chunk, row);
                }
                let left_val = self.eval_expr(left, chunk, row)?;
                let right_val = self.eval_expr(right, chunk, row)?;
                self.eval_binary_op(&left_val, *op, &right_val)
            }
            FilterExpression::Unary { op, operand } => {
                let val = self.eval_expr(operand, chunk, row);
                self.eval_unary_op(*op, val)
            }
            FilterExpression::FunctionCall { name, args } => {
                self.eval_function(name, args, chunk, row)
            }
            FilterExpression::List(items) => {
                let values: Vec<Value> = items
                    .iter()
                    .filter_map(|item| self.eval_expr(item, chunk, row))
                    .collect();
                Some(Value::List(values.into()))
            }
            FilterExpression::Map(pairs) => {
                let mut map = BTreeMap::new();
                for (k, v) in pairs {
                    if let Some(val) = self.eval_expr(v, chunk, row) {
                        if k == "*" {
                            // AllProperties marker: flatten the inner map into the result
                            if let Value::Map(inner) = val {
                                map.extend(inner.iter().map(|(pk, pv)| (pk.clone(), pv.clone())));
                            }
                        } else {
                            map.insert(PropertyKey::new(k.as_str()), val);
                        }
                    }
                }
                Some(Value::Map(Arc::new(map)))
            }
            FilterExpression::IndexAccess { base, index } => {
                let base_val = self.eval_expr(base, chunk, row)?;
                let index_val = self.eval_expr(index, chunk, row)?;
                match (&base_val, &index_val) {
                    (Value::List(items), Value::Int64(i)) => {
                        let idx = if *i < 0 {
                            // Negative indexing from end
                            let len = items.len() as i64;
                            (len + i) as usize
                        } else {
                            *i as usize
                        };
                        items.get(idx).cloned()
                    }
                    (Value::String(s), Value::Int64(i)) => {
                        let idx = if *i < 0 {
                            let len = s.len() as i64;
                            (len + i) as usize
                        } else {
                            *i as usize
                        };
                        s.chars()
                            .nth(idx)
                            .map(|c| Value::String(c.to_string().into()))
                    }
                    (Value::Map(m), Value::String(key)) => {
                        let prop_key = PropertyKey::new(key.as_str());
                        m.get(&prop_key).cloned()
                    }
                    (_, Value::String(key)) => {
                        // Node/edge bracket access: n['name']. T17k :
                        // use the scalar store accessor (targeted zone
                        // lookup) instead of bulk-hydrating via
                        // `resolve_node` — same rationale as the
                        // FilterExpression::Property branch above.
                        if let FilterExpression::Variable(var) = base.as_ref()
                            && let Some(&col_idx) = self.variable_columns.get(var)
                            && let Some(col) = chunk.column(col_idx)
                        {
                            let prop_key = PropertyKey::new(key.as_str());
                            if let Some(node_id) = col.get_node_id(row) {
                                return self.store.get_node_property(node_id, &prop_key);
                            }
                            if let Some(edge_id) = col.get_edge_id(row) {
                                return self.store.get_edge_property(edge_id, &prop_key);
                            }
                        }
                        None
                    }
                    _ => None,
                }
            }
            FilterExpression::SliceAccess { base, start, end } => {
                let base_val = self.eval_expr(base, chunk, row)?;
                let start_idx = start
                    .as_ref()
                    .and_then(|s| self.eval_expr(s, chunk, row))
                    .and_then(|v| {
                        if let Value::Int64(i) = v {
                            Some(i as usize)
                        } else {
                            None
                        }
                    })
                    .unwrap_or(0);

                match &base_val {
                    Value::List(items) => {
                        let end_idx = end
                            .as_ref()
                            .and_then(|e| self.eval_expr(e, chunk, row))
                            .and_then(|v| {
                                if let Value::Int64(i) = v {
                                    Some(i as usize)
                                } else {
                                    None
                                }
                            })
                            .unwrap_or(items.len());
                        let sliced: Vec<Value> = items
                            .get(start_idx..end_idx.min(items.len()))
                            .unwrap_or(&[])
                            .to_vec();
                        Some(Value::List(sliced.into()))
                    }
                    Value::String(s) => {
                        let chars: Vec<char> = s.chars().collect();
                        let end_idx = end
                            .as_ref()
                            .and_then(|e| self.eval_expr(e, chunk, row))
                            .and_then(|v| {
                                if let Value::Int64(i) = v {
                                    Some(i as usize)
                                } else {
                                    None
                                }
                            })
                            .unwrap_or(chars.len());
                        let sliced: String = chars
                            .get(start_idx..end_idx.min(chars.len()))
                            .unwrap_or(&[])
                            .iter()
                            .collect();
                        Some(Value::String(sliced.into()))
                    }
                    _ => None,
                }
            }
            FilterExpression::Case {
                operand,
                when_clauses,
                else_clause,
            } => self.eval_case(
                operand.as_deref(),
                when_clauses,
                else_clause.as_deref(),
                chunk,
                row,
            ),
            FilterExpression::Id(variable) => {
                let col_idx = *self.variable_columns.get(variable)?;
                let col = chunk.column(col_idx)?;
                // Try as node first, then as edge
                if let Some(node_id) = col.get_node_id(row) {
                    Some(Value::Int64(node_id.0 as i64))
                } else {
                    col.get_edge_id(row)
                        .map(|edge_id| Value::Int64(edge_id.0 as i64))
                }
            }
            FilterExpression::Labels(variable) => {
                let col_idx = *self.variable_columns.get(variable)?;
                let col = chunk.column(col_idx)?;
                let node_id = col.get_node_id(row)?;
                let node = self.resolve_node(node_id)?;
                let labels: Vec<Value> = node
                    .labels
                    .iter()
                    .map(|l| Value::String(l.clone()))
                    .collect();
                Some(Value::List(labels.into()))
            }
            FilterExpression::Type(variable) => {
                let col_idx = *self.variable_columns.get(variable)?;
                let col = chunk.column(col_idx)?;
                let edge_id = col.get_edge_id(row)?;
                let edge = self.resolve_edge(edge_id)?;
                Some(Value::String(edge.edge_type.clone()))
            }
            FilterExpression::ListComprehension {
                variable,
                list_expr,
                filter_expr,
                map_expr,
            } => {
                // Evaluate the source list (accept both List and Vector)
                let list_val = self.eval_expr(list_expr, chunk, row)?;
                let owned_items: Vec<Value>;
                let items: &[Value] = match &list_val {
                    Value::List(list) => list,
                    Value::Vector(vec) => {
                        owned_items = vec.iter().map(|&f| Value::Float64(f64::from(f))).collect();
                        &owned_items
                    }
                    _ => return None,
                };

                // Build the result list by iterating over source items
                let mut result = Vec::new();
                for item in items {
                    // Create a temporary context with the iteration variable bound
                    // For now, we'll do a simplified version that works for literals
                    // A full implementation would need to create a sub-evaluator

                    // Check filter predicate if present
                    let passes_filter = if let Some(filter) = filter_expr {
                        // Simplified: evaluate filter with item as context
                        // This works for simple cases like x > 5
                        matches!(
                            self.eval_comprehension_expr(filter, item, variable),
                            Some(Value::Bool(true))
                        )
                    } else {
                        true
                    };

                    if passes_filter {
                        // Apply the mapping expression
                        if let Some(mapped) = self.eval_comprehension_expr(map_expr, item, variable)
                        {
                            result.push(mapped);
                        }
                    }
                }

                Some(Value::List(result.into()))
            }
            FilterExpression::ListPredicate {
                kind,
                variable,
                list_expr,
                predicate,
            } => {
                let list_val = self.eval_expr(list_expr, chunk, row)?;
                // Accept both List and Vector as iterable sequences
                let items: Vec<&Value>;
                let vec_items: Vec<Value>;
                match &list_val {
                    Value::List(list) => {
                        items = list.iter().collect();
                    }
                    Value::Vector(vec) => {
                        vec_items = vec.iter().map(|&f| Value::Float64(f64::from(f))).collect();
                        items = vec_items.iter().collect();
                    }
                    _ => return None,
                }

                let mut match_count: u32 = 0;
                for item in &items {
                    let result = self.eval_comprehension_expr(predicate, item, variable);
                    if matches!(result, Some(Value::Bool(true))) {
                        match_count += 1;
                    }
                }

                let result = match kind {
                    ListPredicateKind::All => match_count == items.len() as u32,
                    ListPredicateKind::Any => match_count > 0,
                    ListPredicateKind::None => match_count == 0,
                    ListPredicateKind::Single => match_count == 1,
                };

                Some(Value::Bool(result))
            }
            FilterExpression::ExistsSubquery {
                start_var,
                direction,
                edge_types,
                ..
            } => {
                // Get the start node ID from the current row
                let col_idx = *self.variable_columns.get(start_var)?;
                let col = chunk.column(col_idx)?;
                let start_node_id = col.get_node_id(row)?;

                // Check if any matching edges exist
                let exists = self
                    .store
                    .edges_from(start_node_id, *direction)
                    .into_iter()
                    .any(|(_, edge_id)| {
                        // Check edge type if specified
                        if !edge_types.is_empty() {
                            if let Some(actual_type) = self.store.edge_type(edge_id) {
                                edge_types
                                    .iter()
                                    .any(|t| actual_type.as_str().eq_ignore_ascii_case(t.as_str()))
                            } else {
                                false
                            }
                        } else {
                            true
                        }
                    });

                Some(Value::Bool(exists))
            }
            FilterExpression::CountSubquery {
                start_var,
                direction,
                edge_types,
            } => {
                let col_idx = *self.variable_columns.get(start_var)?;
                let col = chunk.column(col_idx)?;
                let start_node_id = col.get_node_id(row)?;

                let count = self
                    .store
                    .edges_from(start_node_id, *direction)
                    .into_iter()
                    .filter(|(_, edge_id)| {
                        if !edge_types.is_empty() {
                            if let Some(actual_type) = self.store.edge_type(*edge_id) {
                                edge_types
                                    .iter()
                                    .any(|t| actual_type.as_str().eq_ignore_ascii_case(t.as_str()))
                            } else {
                                false
                            }
                        } else {
                            true
                        }
                    })
                    .count();

                Some(Value::Int64(count as i64))
            }
            FilterExpression::Reduce {
                accumulator,
                initial,
                variable,
                list,
                expression,
            } => {
                let init_val = self.eval_expr(initial, chunk, row)?;
                let list_val = self.eval_expr(list, chunk, row)?;
                let owned_items: Vec<Value>;
                let items: &[Value] = match &list_val {
                    Value::List(list) => list,
                    Value::Vector(vec) => {
                        owned_items = vec.iter().map(|&f| Value::Float64(f64::from(f))).collect();
                        &owned_items
                    }
                    _ => return None,
                };
                let mut acc = init_val;
                for item in items {
                    acc = self.eval_reduce_expr(
                        expression,
                        &acc,
                        accumulator,
                        item,
                        variable,
                        (chunk, row),
                    )?;
                }
                Some(acc)
            }
        }
    }

    /// Evaluates an expression in the context of a reduce() call.
    ///
    /// Both the accumulator variable and the iteration variable are bound.
    /// The `ctx` parameter provides chunk context `(chunk, row)` for resolving
    /// outer-scope variables (variables not bound by reduce).
    fn eval_reduce_expr(
        &self,
        expr: &FilterExpression,
        acc_val: &Value,
        acc_name: &str,
        item_val: &Value,
        item_name: &str,
        ctx: (&DataChunk, usize),
    ) -> Option<Value> {
        // Closure for recursive calls with all bindings
        let recurse = |e| self.eval_reduce_expr(e, acc_val, acc_name, item_val, item_name, ctx);
        match expr {
            FilterExpression::Variable(name) if name == acc_name => Some(acc_val.clone()),
            FilterExpression::Variable(name) if name == item_name => Some(item_val.clone()),
            FilterExpression::Literal(v) => Some(v.clone()),
            FilterExpression::Binary { left, op, right } => {
                // IN operator needs special handling: right side is a list
                if *op == BinaryFilterOp::In {
                    let l = recurse(left)?;
                    let r = recurse(right)?;
                    return match r {
                        Value::List(items) => {
                            if l.is_null() {
                                return Some(Value::Null);
                            }
                            let mut has_null = false;
                            for v in items.iter() {
                                if v.is_null() {
                                    has_null = true;
                                } else if Self::values_equal(&l, v) {
                                    return Some(Value::Bool(true));
                                }
                            }
                            if has_null {
                                Some(Value::Null)
                            } else {
                                Some(Value::Bool(false))
                            }
                        }
                        _ => None,
                    };
                }
                let l = recurse(left)?;
                let r = recurse(right)?;
                self.eval_binary_op(&l, *op, &r)
            }
            FilterExpression::Unary { op, operand } => {
                let val = recurse(operand);
                self.eval_unary_op(*op, val)
            }
            FilterExpression::Property {
                variable: var,
                property,
            } if var == item_name => {
                if let Value::Map(map) = item_val {
                    Some(
                        map.iter()
                            .find(|(k, _)| k.as_str() == property)
                            .map_or(Value::Null, |(_, v)| v.clone()),
                    )
                } else {
                    None
                }
            }
            FilterExpression::Property {
                variable: var,
                property,
            } if var == acc_name => {
                if let Value::Map(map) = acc_val {
                    Some(
                        map.iter()
                            .find(|(k, _)| k.as_str() == property)
                            .map_or(Value::Null, |(_, v)| v.clone()),
                    )
                } else {
                    None
                }
            }
            FilterExpression::List(items) => {
                let values: Vec<Value> = items.iter().filter_map(&recurse).collect();
                Some(Value::List(values.into()))
            }
            FilterExpression::Case {
                operand,
                when_clauses,
                else_clause,
            } => {
                if let Some(test_expr) = operand.as_deref() {
                    let test_val = recurse(test_expr)?;
                    for (when_expr, then_expr) in when_clauses {
                        let when_val = recurse(when_expr)?;
                        if Self::values_equal(&test_val, &when_val) {
                            return recurse(then_expr);
                        }
                    }
                } else {
                    for (when_expr, then_expr) in when_clauses {
                        let when_val = recurse(when_expr)?;
                        if when_val.as_bool() == Some(true) {
                            return recurse(then_expr);
                        }
                    }
                }
                if let Some(else_expr) = else_clause.as_deref() {
                    recurse(else_expr)
                } else {
                    Some(Value::Null)
                }
            }
            FilterExpression::IndexAccess { base, index } => {
                let base_val = recurse(base)?;
                let index_val = recurse(index)?;
                match (&base_val, &index_val) {
                    (Value::List(items), Value::Int64(i)) => {
                        let idx = if *i < 0 {
                            let len = items.len() as i64;
                            (len + i) as usize
                        } else {
                            *i as usize
                        };
                        items.get(idx).cloned()
                    }
                    (Value::String(s), Value::Int64(i)) => {
                        let idx = if *i < 0 {
                            let len = s.len() as i64;
                            (len + i) as usize
                        } else {
                            *i as usize
                        };
                        s.chars()
                            .nth(idx)
                            .map(|c| Value::String(c.to_string().into()))
                    }
                    (Value::Map(m), Value::String(key)) => {
                        let prop_key = PropertyKey::new(key.as_str());
                        m.get(&prop_key).cloned()
                    }
                    _ => None,
                }
            }
            // For expressions not referencing the local variables, resolve
            // from the outer scope (chunk/row)
            _ => self.eval_expr(expr, ctx.0, ctx.1),
        }
    }

    /// Evaluates an expression in the context of a list comprehension.
    /// The `item` is the current iteration value bound to `variable`.
    fn eval_comprehension_expr(
        &self,
        expr: &FilterExpression,
        item: &Value,
        variable: &str,
    ) -> Option<Value> {
        match expr {
            FilterExpression::Variable(name) if name == variable => Some(item.clone()),
            FilterExpression::Literal(v) => Some(v.clone()),
            FilterExpression::Binary { left, op, right } => {
                // IN operator needs special handling: right side is a list
                if *op == BinaryFilterOp::In {
                    let left_val = self.eval_comprehension_expr(left, item, variable)?;
                    let right_val = self.eval_comprehension_expr(right, item, variable)?;
                    return match right_val {
                        Value::List(items) => {
                            if left_val.is_null() {
                                return Some(Value::Null);
                            }
                            let mut has_null = false;
                            for v in items.iter() {
                                if v.is_null() {
                                    has_null = true;
                                } else if Self::values_equal(&left_val, v) {
                                    return Some(Value::Bool(true));
                                }
                            }
                            if has_null {
                                Some(Value::Null)
                            } else {
                                Some(Value::Bool(false))
                            }
                        }
                        _ => None,
                    };
                }
                let left_val = self.eval_comprehension_expr(left, item, variable)?;
                let right_val = self.eval_comprehension_expr(right, item, variable)?;
                self.eval_binary_op(&left_val, *op, &right_val)
            }
            FilterExpression::Unary { op, operand } => {
                let val = self.eval_comprehension_expr(operand, item, variable);
                self.eval_unary_op(*op, val)
            }
            FilterExpression::Property {
                variable: var,
                property,
            } if var == variable => {
                // Property access on the iteration variable
                if let Value::Map(m) = item {
                    let key = PropertyKey::new(property.as_str());
                    m.get(&key).cloned()
                } else {
                    None
                }
            }
            FilterExpression::List(items) => {
                let values: Vec<Value> = items
                    .iter()
                    .filter_map(|i| self.eval_comprehension_expr(i, item, variable))
                    .collect();
                Some(Value::List(values.into()))
            }
            FilterExpression::Case {
                operand,
                when_clauses,
                else_clause,
            } => self.eval_case_in_comprehension(
                operand.as_deref(),
                when_clauses,
                else_clause.as_deref(),
                item,
                variable,
            ),
            // For other expression types, return None (unsupported in comprehension)
            _ => None,
        }
    }

    /// Evaluates a CASE expression inside a list comprehension or predicate context.
    fn eval_case_in_comprehension(
        &self,
        operand: Option<&FilterExpression>,
        when_clauses: &[(FilterExpression, FilterExpression)],
        else_clause: Option<&FilterExpression>,
        item: &Value,
        variable: &str,
    ) -> Option<Value> {
        if let Some(test_expr) = operand {
            let test_val = self
                .eval_comprehension_expr(test_expr, item, variable)
                .unwrap_or(Value::Null);
            for (when_expr, then_expr) in when_clauses {
                let when_val = self
                    .eval_comprehension_expr(when_expr, item, variable)
                    .unwrap_or(Value::Null);
                if !test_val.is_null()
                    && !when_val.is_null()
                    && Self::values_equal(&test_val, &when_val)
                {
                    return self.eval_comprehension_expr(then_expr, item, variable);
                }
            }
        } else {
            for (when_expr, then_expr) in when_clauses {
                let when_val = self.eval_comprehension_expr(when_expr, item, variable)?;
                if when_val.as_bool() == Some(true) {
                    return self.eval_comprehension_expr(then_expr, item, variable);
                }
            }
        }
        if let Some(else_expr) = else_clause {
            self.eval_comprehension_expr(else_expr, item, variable)
        } else {
            Some(Value::Null)
        }
    }

    /// Evaluates a binary operator with ISO three-valued logic.
    ///
    /// NULL propagation: `NULL = x`, `x = NULL`, `NULL <> x` all yield
    /// `Value::Null` (UNKNOWN), not `Value::Bool`. AND/OR/XOR follow the
    /// standard truth tables where FALSE AND UNKNOWN = FALSE, etc.
    ///
    /// For structural equality (DISTINCT, GROUP BY), use [`values_equal`]
    /// directly, which treats NULL == NULL as true.
    fn eval_binary_op(&self, left: &Value, op: BinaryFilterOp, right: &Value) -> Option<Value> {
        match op {
            // Three-valued logic for AND/OR/XOR (ISO/IEC 39075 Section 21)
            BinaryFilterOp::And => match (left.as_bool(), right.as_bool()) {
                (Some(false), _) | (_, Some(false)) => Some(Value::Bool(false)),
                (Some(true), Some(true)) => Some(Value::Bool(true)),
                _ => Some(Value::Null), // UNKNOWN
            },
            BinaryFilterOp::Or => match (left.as_bool(), right.as_bool()) {
                (Some(true), _) | (_, Some(true)) => Some(Value::Bool(true)),
                (Some(false), Some(false)) => Some(Value::Bool(false)),
                _ => Some(Value::Null), // UNKNOWN
            },
            BinaryFilterOp::Xor => match (left.as_bool(), right.as_bool()) {
                (Some(l), Some(r)) => Some(Value::Bool(l ^ r)),
                _ => Some(Value::Null), // UNKNOWN
            },
            // NULL = anything or anything = NULL is UNKNOWN (three-valued logic).
            // values_equal is preserved for structural equality (DISTINCT, GROUP BY).
            BinaryFilterOp::Eq => {
                if left.is_null() || right.is_null() {
                    Some(Value::Null)
                } else {
                    Some(Value::Bool(Self::values_equal(left, right)))
                }
            }
            BinaryFilterOp::Ne => {
                if left.is_null() || right.is_null() {
                    Some(Value::Null)
                } else {
                    Some(Value::Bool(!Self::values_equal(left, right)))
                }
            }
            BinaryFilterOp::Lt => self.compare_values(left, right).map(|c| Value::Bool(c < 0)),
            BinaryFilterOp::Le => self
                .compare_values(left, right)
                .map(|c| Value::Bool(c <= 0)),
            BinaryFilterOp::Gt => self.compare_values(left, right).map(|c| Value::Bool(c > 0)),
            BinaryFilterOp::Ge => self
                .compare_values(left, right)
                .map(|c| Value::Bool(c >= 0)),
            // Arithmetic operators
            BinaryFilterOp::Add => {
                // String concatenation: string + string, or string + other
                match (left, right) {
                    (Value::String(a), Value::String(b)) => {
                        let mut s = String::with_capacity(a.len() + b.len());
                        s.push_str(a);
                        s.push_str(b);
                        Some(Value::String(s.into()))
                    }
                    (Value::String(a), other) => {
                        let b = match other {
                            Value::Int64(i) => i.to_string(),
                            Value::Float64(f) => f.to_string(),
                            Value::Bool(b) => b.to_string(),
                            Value::Null => return Some(Value::Null),
                            _ => return None,
                        };
                        let mut s = String::with_capacity(a.len() + b.len());
                        s.push_str(a);
                        s.push_str(&b);
                        Some(Value::String(s.into()))
                    }
                    // Temporal addition
                    (Value::Date(d), Value::Duration(dur))
                    | (Value::Duration(dur), Value::Date(d)) => {
                        Some(Value::Date(d.add_duration(dur)))
                    }
                    (Value::Time(t), Value::Duration(dur))
                    | (Value::Duration(dur), Value::Time(t)) => {
                        Some(Value::Time(t.add_duration(dur)))
                    }
                    (Value::Timestamp(ts), Value::Duration(dur))
                    | (Value::Duration(dur), Value::Timestamp(ts)) => {
                        Some(Value::Timestamp(ts.add_duration(dur)))
                    }
                    (Value::Duration(a), Value::Duration(b)) => Some(Value::Duration(a.add(*b))),
                    _ => self.eval_arithmetic(left, right, |a, b| a + b, |a, b| a + b),
                }
            }
            BinaryFilterOp::Sub => match (left, right) {
                // Temporal subtraction
                (Value::Date(a), Value::Duration(dur)) => Some(Value::Date(a.sub_duration(dur))),
                (Value::Time(a), Value::Duration(dur)) => {
                    Some(Value::Time(a.add_duration(&dur.neg())))
                }
                (Value::Timestamp(a), Value::Duration(dur)) => {
                    Some(Value::Timestamp(a.add_duration(&dur.neg())))
                }
                (Value::Date(a), Value::Date(b)) => {
                    let days = a.as_days() as i64 - b.as_days() as i64;
                    Some(Value::Duration(obrain_common::types::Duration::from_days(
                        days,
                    )))
                }
                (Value::Time(a), Value::Time(b)) => {
                    let nanos = a.as_nanos() as i64 - b.as_nanos() as i64;
                    Some(Value::Duration(obrain_common::types::Duration::from_nanos(
                        nanos,
                    )))
                }
                (Value::Timestamp(a), Value::Timestamp(b)) => {
                    let micros = a.duration_since(*b);
                    Some(Value::Duration(obrain_common::types::Duration::from_nanos(
                        micros * 1000,
                    )))
                }
                (Value::Duration(a), Value::Duration(b)) => Some(Value::Duration(a.sub(*b))),
                _ => self.eval_arithmetic(left, right, |a, b| a - b, |a, b| a - b),
            },
            BinaryFilterOp::Mul => match (left, right) {
                (Value::Duration(d), Value::Int64(n)) | (Value::Int64(n), Value::Duration(d)) => {
                    Some(Value::Duration(d.mul(*n)))
                }
                _ => self.eval_arithmetic(left, right, |a, b| a * b, |a, b| a * b),
            },
            BinaryFilterOp::Div => match (left, right) {
                (Value::Duration(d), Value::Int64(n)) if *n != 0 => {
                    Some(Value::Duration(d.div(*n)))
                }
                _ => self.eval_arithmetic(left, right, |a, b| a / b, |a, b| a / b),
            },
            BinaryFilterOp::Mod => self.eval_modulo(left, right),
            // String operators
            BinaryFilterOp::StartsWith => {
                let l = left.as_str()?;
                let r = right.as_str()?;
                Some(Value::Bool(l.starts_with(r)))
            }
            BinaryFilterOp::EndsWith => {
                let l = left.as_str()?;
                let r = right.as_str()?;
                Some(Value::Bool(l.ends_with(r)))
            }
            BinaryFilterOp::Contains => {
                let l = left.as_str()?;
                let r = right.as_str()?;
                Some(Value::Bool(l.contains(r)))
            }
            // IN is handled separately
            BinaryFilterOp::In => None,
            // Regex match (=~)
            BinaryFilterOp::Regex => {
                #[cfg(any(feature = "regex", feature = "regex-lite"))]
                match (left, right) {
                    (Value::String(s), Value::String(pattern)) => match Regex::new(pattern) {
                        Ok(re) => Some(Value::Bool(re.is_match(s))),
                        Err(_) => None,
                    },
                    _ => None,
                }
                #[cfg(not(any(feature = "regex", feature = "regex-lite")))]
                {
                    let _ = (left, right);
                    None
                }
            }
            // Power/exponentiation (^)
            BinaryFilterOp::Pow => {
                match (left, right) {
                    (Value::Int64(base), Value::Int64(exp)) => {
                        Some(Value::Float64((*base as f64).powf(*exp as f64)))
                    }
                    (Value::Float64(base), Value::Float64(exp)) => {
                        Some(Value::Float64(base.powf(*exp)))
                    }
                    (Value::Int64(base), Value::Float64(exp)) => {
                        Some(Value::Float64((*base as f64).powf(*exp)))
                    }
                    (Value::Float64(base), Value::Int64(exp)) => {
                        Some(Value::Float64(base.powf(*exp as f64)))
                    }
                    _ => None, // Type mismatch
                }
            }
            // SQL LIKE pattern matching
            BinaryFilterOp::Like => {
                #[cfg(any(feature = "regex", feature = "regex-lite"))]
                match (left, right) {
                    (Value::String(s), Value::String(pattern)) => {
                        let mut regex_pattern = String::with_capacity(pattern.len() + 4);
                        regex_pattern.push('^');
                        let mut chars = pattern.chars().peekable();
                        while let Some(ch) = chars.next() {
                            match ch {
                                '%' => regex_pattern.push_str(".*"),
                                '_' => regex_pattern.push('.'),
                                '\\' => {
                                    if let Some(next) = chars.next() {
                                        regex_escape_char(next, &mut regex_pattern);
                                    }
                                }
                                _ => regex_escape_char(ch, &mut regex_pattern),
                            }
                        }
                        regex_pattern.push('$');
                        match Regex::new(&regex_pattern) {
                            Ok(re) => Some(Value::Bool(re.is_match(s))),
                            Err(_) => None,
                        }
                    }
                    (Value::Null, _) | (_, Value::Null) => Some(Value::Null),
                    _ => None,
                }
                #[cfg(not(any(feature = "regex", feature = "regex-lite")))]
                {
                    let _ = (left, right);
                    None
                }
            }
            // String concatenation (||)
            BinaryFilterOp::Concat => match (left, right) {
                (Value::String(a), Value::String(b)) => {
                    let mut s = String::with_capacity(a.len() + b.len());
                    s.push_str(a);
                    s.push_str(b);
                    Some(Value::String(s.into()))
                }
                (Value::String(a), other) => {
                    let b = value_to_string(other)?;
                    let mut s = String::with_capacity(a.len() + b.len());
                    s.push_str(a);
                    s.push_str(&b);
                    Some(Value::String(s.into()))
                }
                (other, Value::String(b)) => {
                    let a = value_to_string(other)?;
                    let mut s = String::with_capacity(a.len() + b.len());
                    s.push_str(&a);
                    s.push_str(b);
                    Some(Value::String(s.into()))
                }
                (Value::Null, _) | (_, Value::Null) => Some(Value::Null),
                _ => None,
            },
        }
    }

    fn eval_arithmetic<F1, F2>(
        &self,
        left: &Value,
        right: &Value,
        int_op: F1,
        float_op: F2,
    ) -> Option<Value>
    where
        F1: Fn(i64, i64) -> i64,
        F2: Fn(f64, f64) -> f64,
    {
        match (left, right) {
            (Value::Int64(a), Value::Int64(b)) => Some(Value::Int64(int_op(*a, *b))),
            (Value::Float64(a), Value::Float64(b)) => Some(Value::Float64(float_op(*a, *b))),
            (Value::Int64(a), Value::Float64(b)) => Some(Value::Float64(float_op(*a as f64, *b))),
            (Value::Float64(a), Value::Int64(b)) => Some(Value::Float64(float_op(*a, *b as f64))),
            _ => None,
        }
    }

    fn eval_modulo(&self, left: &Value, right: &Value) -> Option<Value> {
        match (left, right) {
            (Value::Int64(a), Value::Int64(b)) if *b != 0 => Some(Value::Int64(a % b)),
            (Value::Float64(a), Value::Float64(b)) if *b != 0.0 => Some(Value::Float64(a % b)),
            (Value::Int64(a), Value::Float64(b)) if *b != 0.0 => {
                Some(Value::Float64(*a as f64 % b))
            }
            (Value::Float64(a), Value::Int64(b)) if *b != 0 => Some(Value::Float64(a % *b as f64)),
            _ => None,
        }
    }

    /// Evaluates `left IN right` with three-valued NULL semantics.
    ///
    /// - `NULL IN [...]` yields UNKNOWN.
    /// - If no element matches but the list contains NULLs, yields UNKNOWN.
    /// - Otherwise yields `true` on first match, `false` if none match.
    fn eval_in_operator(
        &self,
        left: &Value,
        right: &FilterExpression,
        chunk: &DataChunk,
        row: usize,
    ) -> Option<Value> {
        let right_val = self.eval_expr(right, chunk, row)?;
        match right_val {
            Value::List(items) => {
                // Three-valued IN: NULL IN (...) is UNKNOWN
                if left.is_null() {
                    return Some(Value::Null);
                }
                let mut has_null = false;
                for item in items.iter() {
                    if item.is_null() {
                        has_null = true;
                    } else if Self::values_equal(left, item) {
                        return Some(Value::Bool(true));
                    }
                }
                if has_null {
                    Some(Value::Null) // no match but NULLs present: UNKNOWN
                } else {
                    Some(Value::Bool(false))
                }
            }
            _ => None,
        }
    }

    fn eval_function(
        &self,
        name: &str,
        args: &[FilterExpression],
        chunk: &DataChunk,
        row: usize,
    ) -> Option<Value> {
        match name.to_lowercase().as_str() {
            "id" => {
                if args.len() != 1 {
                    return None;
                }
                if let FilterExpression::Variable(var) = &args[0] {
                    let col_idx = *self.variable_columns.get(var)?;
                    let col = chunk.column(col_idx)?;
                    if let Some(node_id) = col.get_node_id(row) {
                        return Some(Value::Int64(node_id.0 as i64));
                    } else if let Some(edge_id) = col.get_edge_id(row) {
                        return Some(Value::Int64(edge_id.0 as i64));
                    }
                }
                None
            }
            "element_id" | "elementid" => {
                if args.len() != 1 {
                    return None;
                }
                if let FilterExpression::Variable(var) = &args[0] {
                    let col_idx = *self.variable_columns.get(var)?;
                    let col = chunk.column(col_idx)?;
                    // Resolve ambiguity between node/edge by verifying against the
                    // store. VectorData::Generic stores raw Int64 values that both
                    // get_node_id and get_edge_id accept, so we must check which
                    // entity actually exists.
                    if let Some(edge_id) = col.get_edge_id(row)
                        && self.resolve_edge(edge_id).is_some()
                    {
                        return Some(Value::String(format!("e:{}", edge_id.0).into()));
                    }
                    if let Some(node_id) = col.get_node_id(row)
                        && self.resolve_node(node_id).is_some()
                    {
                        return Some(Value::String(format!("n:{}", node_id.0).into()));
                    }
                }
                None
            }
            "labels" => {
                if args.len() != 1 {
                    return None;
                }
                if let FilterExpression::Variable(var) = &args[0] {
                    let col_idx = *self.variable_columns.get(var)?;
                    let col = chunk.column(col_idx)?;
                    let node_id = col.get_node_id(row)?;
                    let node = self.resolve_node(node_id)?;
                    let labels: Vec<Value> = node
                        .labels
                        .iter()
                        .map(|l| Value::String(l.clone()))
                        .collect();
                    return Some(Value::List(labels.into()));
                }
                None
            }
            "type" => {
                if args.len() != 1 {
                    return None;
                }
                if let FilterExpression::Variable(var) = &args[0] {
                    let col_idx = *self.variable_columns.get(var)?;
                    let col = chunk.column(col_idx)?;
                    let edge_id = col.get_edge_id(row)?;
                    let edge = self.resolve_edge(edge_id)?;
                    return Some(Value::String(edge.edge_type.clone()));
                }
                None
            }
            "size" | "length" | "cardinality" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::List(items) => Some(Value::Int64(items.len() as i64)),
                    Value::String(s) => Some(Value::Int64(s.len() as i64)),
                    Value::Path { edges, .. } => Some(Value::Int64(edges.len() as i64)),
                    _ => None,
                }
            }
            "coalesce" => {
                for arg in args {
                    if let Some(val) = self.eval_expr(arg, chunk, row)
                        && !matches!(val, Value::Null)
                    {
                        return Some(val);
                    }
                }
                Some(Value::Null)
            }
            "exists" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row);
                Some(Value::Bool(
                    val.is_some() && !matches!(val, Some(Value::Null)),
                ))
            }
            "tostring" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                let s = match &val {
                    Value::String(s) => s.to_string(),
                    Value::Int64(i) => i.to_string(),
                    Value::Float64(f) => f.to_string(),
                    Value::Bool(b) => b.to_string(),
                    Value::Null => return Some(Value::Null),
                    _ => format!("{val:?}"),
                };
                Some(Value::String(s.into()))
            }
            "tointeger" | "toint" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Int64(i)),
                    Value::Float64(f) => Some(Value::Int64(f as i64)),
                    Value::String(s) => s.parse::<i64>().ok().map(Value::Int64),
                    _ => None,
                }
            }
            "tofloat" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Float64(i as f64)),
                    Value::Float64(f) => Some(Value::Float64(f)),
                    Value::String(s) => s.parse::<f64>().ok().map(Value::Float64),
                    _ => None,
                }
            }
            "toboolean" | "tobool" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Bool(b) => Some(Value::Bool(b)),
                    Value::String(s) => match s.to_lowercase().as_str() {
                        "true" => Some(Value::Bool(true)),
                        "false" => Some(Value::Bool(false)),
                        _ => None,
                    },
                    _ => None,
                }
            }
            "haslabel" => {
                // hasLabel(node, label) - checks if a node has a specific label
                if args.len() != 2 {
                    return None;
                }
                // First arg is the node variable
                let node_id = if let FilterExpression::Variable(var) = &args[0] {
                    let col_idx = *self.variable_columns.get(var)?;
                    let col = chunk.column(col_idx)?;
                    col.get_node_id(row)?
                } else {
                    return None;
                };
                // Second arg is the label to check
                let Value::String(label) = self.eval_expr(&args[1], chunk, row)? else {
                    return None;
                };
                // Check if the node has this label
                let node = self.resolve_node(node_id)?;
                let has_label = node.labels.iter().any(|l| l.as_str() == label.as_str());
                Some(Value::Bool(has_label))
            }
            "istyped" => {
                // isTyped(value, type_name) - checks if a value has a specific GQL type
                if args.len() != 2 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                let Value::String(type_name) = self.eval_expr(&args[1], chunk, row)? else {
                    return None;
                };
                let matches = match type_name.to_uppercase().as_str() {
                    "BOOLEAN" | "BOOL" => matches!(val, Value::Bool(_)),
                    "INTEGER" | "INT" | "INT64" => matches!(val, Value::Int64(_)),
                    "FLOAT" | "FLOAT64" | "DOUBLE" => matches!(val, Value::Float64(_)),
                    "STRING" => matches!(val, Value::String(_)),
                    "LIST" => matches!(val, Value::List(_)),
                    "MAP" | "RECORD" => matches!(val, Value::Map(_)),
                    "NULL" => matches!(val, Value::Null),
                    "DATE" => matches!(val, Value::Date(_)),
                    "TIME" => matches!(val, Value::Time(_)),
                    "DATETIME" | "TIMESTAMP" => matches!(val, Value::Timestamp(_)),
                    "DURATION" => matches!(val, Value::Duration(_)),
                    "PATH" => matches!(val, Value::Path { .. }),
                    "NODE" | "EDGE" | "GRAPH" => false,
                    s if s.starts_with("LIST<") && s.ends_with('>') => {
                        let elem_type = &s[5..s.len() - 1];
                        match &val {
                            Value::List(items) => {
                                items.iter().all(|v| Self::value_matches_type(v, elem_type))
                            }
                            _ => false,
                        }
                    }
                    _ => false,
                };
                Some(Value::Bool(matches))
            }
            "isdirected" => {
                // isDirected(edge) - checks if an edge is directed (always true in LPG)
                if args.len() != 1 {
                    return None;
                }
                // In LPG, all edges are directed
                if let FilterExpression::Variable(var) = &args[0] {
                    let col_idx = *self.variable_columns.get(var)?;
                    let col = chunk.column(col_idx)?;
                    // If the column contains an edge ID, it's directed
                    if col.get_edge_id(row).is_some() {
                        return Some(Value::Bool(true));
                    }
                }
                Some(Value::Bool(false))
            }
            "issource" => {
                // isSource(node, edge) - checks if node is the source of edge
                if args.len() != 2 {
                    return None;
                }
                let node_id = if let FilterExpression::Variable(var) = &args[0] {
                    let col_idx = *self.variable_columns.get(var)?;
                    let col = chunk.column(col_idx)?;
                    col.get_node_id(row)?
                } else {
                    return None;
                };
                let edge_id = if let FilterExpression::Variable(var) = &args[1] {
                    let col_idx = *self.variable_columns.get(var)?;
                    let col = chunk.column(col_idx)?;
                    col.get_edge_id(row)?
                } else {
                    return None;
                };
                let edge = self.resolve_edge(edge_id)?;
                Some(Value::Bool(edge.src == node_id))
            }
            "isdestination" => {
                // isDestination(node, edge) - checks if node is the destination of edge
                if args.len() != 2 {
                    return None;
                }
                let node_id = if let FilterExpression::Variable(var) = &args[0] {
                    let col_idx = *self.variable_columns.get(var)?;
                    let col = chunk.column(col_idx)?;
                    col.get_node_id(row)?
                } else {
                    return None;
                };
                let edge_id = if let FilterExpression::Variable(var) = &args[1] {
                    let col_idx = *self.variable_columns.get(var)?;
                    let col = chunk.column(col_idx)?;
                    col.get_edge_id(row)?
                } else {
                    return None;
                };
                let edge = self.resolve_edge(edge_id)?;
                Some(Value::Bool(edge.dst == node_id))
            }
            "all_different" => {
                // ALL_DIFFERENT - ISO GQL predicate (G113)
                // Two calling conventions:
                //   all_different(list)          - check list elements are distinct
                //   ALL_DIFFERENT(var1, var2, ..) - check graph elements are distinct
                if args.is_empty() {
                    return Some(Value::Bool(true));
                }
                if args.len() == 1 {
                    // Single-argument: treat as list check
                    let val = self.eval_expr(&args[0], chunk, row)?;
                    return match val {
                        Value::List(items) => {
                            let mut seen = std::collections::HashSet::new();
                            let all_diff = items.iter().all(|item| {
                                let key = format!("{item:?}");
                                seen.insert(key)
                            });
                            Some(Value::Bool(all_diff))
                        }
                        _ => Some(Value::Bool(true)),
                    };
                }
                // Multi-argument: compare element IDs
                let mut ids: Vec<u64> = Vec::with_capacity(args.len());
                for arg in args {
                    if let FilterExpression::Variable(var) = arg {
                        let col_idx = *self.variable_columns.get(var)?;
                        let col = chunk.column(col_idx)?;
                        if let Some(nid) = col.get_node_id(row) {
                            ids.push(nid.0);
                        } else if let Some(eid) = col.get_edge_id(row) {
                            ids.push(eid.0);
                        } else {
                            return None;
                        }
                    } else {
                        return None;
                    }
                }
                let length = ids.len();
                ids.sort_unstable();
                ids.dedup();
                Some(Value::Bool(ids.len() == length))
            }
            "same" => {
                // SAME - ISO GQL predicate (G114)
                // Two calling conventions:
                //   same(list)          - check list elements are equal
                //   SAME(var1, var2, ..) - check graph elements are identical
                if args.is_empty() {
                    return Some(Value::Bool(true));
                }
                if args.len() == 1 {
                    // Single-argument: treat as list check
                    let val = self.eval_expr(&args[0], chunk, row)?;
                    return match val {
                        Value::List(items) => {
                            let all_same = if items.is_empty() {
                                true
                            } else {
                                items.iter().all(|item| item == &items[0])
                            };
                            Some(Value::Bool(all_same))
                        }
                        _ => Some(Value::Bool(true)),
                    };
                }
                // Multi-argument: compare element IDs
                let mut first_id: Option<u64> = None;
                for arg in args {
                    if let FilterExpression::Variable(var) = arg {
                        let col_idx = *self.variable_columns.get(var)?;
                        let col = chunk.column(col_idx)?;
                        let current_id = if let Some(nid) = col.get_node_id(row) {
                            nid.0
                        } else if let Some(eid) = col.get_edge_id(row) {
                            eid.0
                        } else {
                            return None;
                        };
                        match first_id {
                            None => first_id = Some(current_id),
                            Some(fid) if fid != current_id => return Some(Value::Bool(false)),
                            _ => {}
                        }
                    } else {
                        return None;
                    }
                }
                Some(Value::Bool(true))
            }
            "property_exists" => {
                // property_exists(entity, key) - checks if a property key exists on an entity
                if args.len() != 2 {
                    return None;
                }
                let Value::String(key) = self.eval_expr(&args[1], chunk, row)? else {
                    return None;
                };
                // Try node first, then edge
                if let FilterExpression::Variable(var) = &args[0] {
                    let col_idx = *self.variable_columns.get(var)?;
                    let col = chunk.column(col_idx)?;
                    if let Some(nid) = col.get_node_id(row)
                        && let Some(node) = self.resolve_node(nid)
                    {
                        let exists = node
                            .properties
                            .iter()
                            .any(|(k, _)| k.as_str() == key.as_str());
                        return Some(Value::Bool(exists));
                    }
                    if let Some(eid) = col.get_edge_id(row)
                        && let Some(edge) = self.resolve_edge(eid)
                    {
                        let exists = edge
                            .properties
                            .iter()
                            .any(|(k, _)| k.as_str() == key.as_str());
                        return Some(Value::Bool(exists));
                    }
                }
                Some(Value::Bool(false))
            }
            "head" => {
                // head(list) - returns the first element of a list
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::List(items) => items.first().cloned(),
                    _ => None,
                }
            }
            "tail" => {
                // tail(list) - returns all elements except the first
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::List(items) => {
                        if items.is_empty() {
                            Some(Value::List(vec![].into()))
                        } else {
                            Some(Value::List(items[1..].to_vec().into()))
                        }
                    }
                    _ => None,
                }
            }
            "last" => {
                // last(list) - returns the last element of a list
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::List(items) => items.last().cloned(),
                    _ => None,
                }
            }
            "reverse" => {
                // reverse(list) - returns the list in reverse order
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::List(items) => {
                        let reversed: Vec<Value> = items.iter().rev().cloned().collect();
                        Some(Value::List(reversed.into()))
                    }
                    Value::String(s) => {
                        let reversed: String = s.chars().rev().collect();
                        Some(Value::String(reversed.into()))
                    }
                    _ => None,
                }
            }
            // vector(list) - converts a list of numbers to a Vector
            "vector" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::List(items) => {
                        let floats: Vec<f32> = items
                            .iter()
                            .filter_map(|v| match v {
                                Value::Float64(f) => Some(*f as f32),
                                Value::Int64(i) => Some(*i as f32),
                                _ => None,
                            })
                            .collect();
                        if floats.len() == items.len() {
                            Some(Value::Vector(floats.into()))
                        } else {
                            None
                        }
                    }
                    Value::Vector(v) => Some(Value::Vector(v)),
                    _ => None,
                }
            }
            // Vector distance / similarity functions (SIMD-accelerated)
            "cosine_similarity" => {
                if args.len() != 2 {
                    return None;
                }
                let a_val = self.eval_expr(&args[0], chunk, row)?;
                let b_val = self.eval_expr(&args[1], chunk, row)?;
                let a = a_val.as_vector()?;
                let b = b_val.as_vector()?;
                if a.len() != b.len() {
                    return None;
                }
                Some(Value::Float64(
                    crate::index::vector::cosine_similarity(a, b) as f64,
                ))
            }
            "euclidean_distance" => {
                if args.len() != 2 {
                    return None;
                }
                let a_val = self.eval_expr(&args[0], chunk, row)?;
                let b_val = self.eval_expr(&args[1], chunk, row)?;
                let a = a_val.as_vector()?;
                let b = b_val.as_vector()?;
                if a.len() != b.len() {
                    return None;
                }
                Some(Value::Float64(
                    crate::index::vector::euclidean_distance(a, b) as f64,
                ))
            }
            "dot_product" => {
                if args.len() != 2 {
                    return None;
                }
                let a_val = self.eval_expr(&args[0], chunk, row)?;
                let b_val = self.eval_expr(&args[1], chunk, row)?;
                let a = a_val.as_vector()?;
                let b = b_val.as_vector()?;
                if a.len() != b.len() {
                    return None;
                }
                Some(Value::Float64(
                    crate::index::vector::dot_product(a, b) as f64
                ))
            }
            "manhattan_distance" => {
                if args.len() != 2 {
                    return None;
                }
                let a_val = self.eval_expr(&args[0], chunk, row)?;
                let b_val = self.eval_expr(&args[1], chunk, row)?;
                let a = a_val.as_vector()?;
                let b = b_val.as_vector()?;
                if a.len() != b.len() {
                    return None;
                }
                Some(Value::Float64(
                    crate::index::vector::manhattan_distance(a, b) as f64,
                ))
            }
            // --- String functions ---
            "keys" => {
                if args.len() != 1 {
                    return None;
                }
                // keys(n) on a node variable: get property keys from the store
                if let FilterExpression::Variable(var) = &args[0] {
                    let col_idx = *self.variable_columns.get(var)?;
                    let col = chunk.column(col_idx)?;
                    if let Some(node_id) = col.get_node_id(row) {
                        let node = self.resolve_node(node_id)?;
                        let keys: Vec<Value> = node
                            .properties
                            .iter()
                            .map(|(k, _)| Value::String(k.as_str().into()))
                            .collect();
                        return Some(Value::List(keys.into()));
                    }
                }
                // keys(map) on a map value
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Map(map) => {
                        let keys: Vec<Value> = map
                            .keys()
                            .map(|k| Value::String(k.as_str().into()))
                            .collect();
                        Some(Value::List(keys.into()))
                    }
                    _ => None,
                }
            }
            "properties" => {
                if args.len() != 1 {
                    return None;
                }
                if let FilterExpression::Variable(var) = &args[0] {
                    let col_idx = *self.variable_columns.get(var)?;
                    let col = chunk.column(col_idx)?;
                    if let Some(node_id) = col.get_node_id(row) {
                        let node = self.resolve_node(node_id)?;
                        let map: std::collections::BTreeMap<PropertyKey, Value> = node
                            .properties
                            .iter()
                            .map(|(k, v)| (k.clone(), v.clone()))
                            .collect();
                        return Some(Value::Map(Arc::new(map)));
                    } else if let Some(edge_id) = col.get_edge_id(row) {
                        let edge = self.resolve_edge(edge_id)?;
                        let map: std::collections::BTreeMap<PropertyKey, Value> = edge
                            .properties
                            .iter()
                            .map(|(k, v)| (k.clone(), v.clone()))
                            .collect();
                        return Some(Value::Map(Arc::new(map)));
                    }
                }
                None
            }
            "trim" => {
                if args.len() == 1 {
                    // Simple trim(string) - trim whitespace
                    let val = self.eval_expr(&args[0], chunk, row)?;
                    return match val {
                        Value::String(s) => Some(Value::String(s.trim().to_string().into())),
                        _ => None,
                    };
                }
                if args.len() == 3 {
                    // Extended trim(string, chars, mode)
                    // mode: 0=both, 1=leading, 2=trailing
                    let val = self.eval_expr(&args[0], chunk, row)?;
                    let chars_val = self.eval_expr(&args[1], chunk, row)?;
                    let mode_val = self.eval_expr(&args[2], chunk, row)?;
                    let Value::String(s) = val else { return None };
                    let Value::String(chars) = chars_val else {
                        return None;
                    };
                    let Value::Int64(mode) = mode_val else {
                        return None;
                    };
                    let char_set: Vec<char> = chars.chars().collect();
                    let result = match mode {
                        0 => s.trim_matches(|c| char_set.contains(&c)).to_string(),
                        1 => s.trim_start_matches(|c| char_set.contains(&c)).to_string(),
                        2 => s.trim_end_matches(|c| char_set.contains(&c)).to_string(),
                        _ => return None,
                    };
                    return Some(Value::String(result.into()));
                }
                None
            }
            "ltrim" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::String(s) => Some(Value::String(s.trim_start().to_string().into())),
                    _ => None,
                }
            }
            "rtrim" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::String(s) => Some(Value::String(s.trim_end().to_string().into())),
                    _ => None,
                }
            }
            "replace" => {
                if args.len() != 3 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                let search = self.eval_expr(&args[1], chunk, row)?;
                let replacement = self.eval_expr(&args[2], chunk, row)?;
                match (&val, &search, &replacement) {
                    (Value::String(s), Value::String(from), Value::String(to)) => {
                        Some(Value::String(s.replace(from.as_str(), to.as_str()).into()))
                    }
                    _ => None,
                }
            }
            "substring" => {
                if args.len() < 2 || args.len() > 3 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                let start = self.eval_expr(&args[1], chunk, row)?;
                let Value::String(s) = val else {
                    return None;
                };
                let Value::Int64(start_idx) = start else {
                    return None;
                };
                let start_idx = start_idx.max(0) as usize;
                if args.len() == 3 {
                    let length = self.eval_expr(&args[2], chunk, row)?;
                    let Value::Int64(len) = length else {
                        return None;
                    };
                    let len = len.max(0) as usize;
                    let chars: String = s.chars().skip(start_idx).take(len).collect();
                    Some(Value::String(chars.into()))
                } else {
                    let chars: String = s.chars().skip(start_idx).collect();
                    Some(Value::String(chars.into()))
                }
            }
            "split" => {
                if args.len() != 2 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                let delim = self.eval_expr(&args[1], chunk, row)?;
                match (&val, &delim) {
                    (Value::String(s), Value::String(d)) => {
                        let parts: Vec<Value> = s
                            .split(d.as_str())
                            .map(|p| Value::String(p.to_string().into()))
                            .collect();
                        Some(Value::List(parts.into()))
                    }
                    _ => None,
                }
            }
            "toupper" | "upper" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::String(s) => Some(Value::String(s.to_uppercase().into())),
                    _ => None,
                }
            }
            "tolower" | "lower" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::String(s) => Some(Value::String(s.to_lowercase().into())),
                    _ => None,
                }
            }
            "char_length" | "charlength" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::String(s) => Some(Value::Int64(s.chars().count() as i64)),
                    _ => None,
                }
            }
            // --- Numeric functions ---
            "abs" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Int64(i.abs())),
                    Value::Float64(f) => Some(Value::Float64(f.abs())),
                    _ => None,
                }
            }
            "ceil" | "ceiling" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Int64(i)),
                    Value::Float64(f) => Some(Value::Int64(f.ceil() as i64)),
                    _ => None,
                }
            }
            "floor" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Int64(i)),
                    Value::Float64(f) => Some(Value::Int64(f.floor() as i64)),
                    _ => None,
                }
            }
            "round" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Int64(i)),
                    Value::Float64(f) => Some(Value::Int64(f.round() as i64)),
                    _ => None,
                }
            }
            "sqrt" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Float64((i as f64).sqrt())),
                    Value::Float64(f) => Some(Value::Float64(f.sqrt())),
                    _ => None,
                }
            }
            "rand" | "random" => {
                use std::hash::{Hash, Hasher};
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                row.hash(&mut hasher);
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .ok()?
                    .as_nanos()
                    .hash(&mut hasher);
                let hash = hasher.finish();
                let random = (hash as f64) / (u64::MAX as f64);
                Some(Value::Float64(random))
            }
            // --- Collection functions ---
            "range" => {
                if args.len() < 2 || args.len() > 3 {
                    return None;
                }
                let start = self.eval_expr(&args[0], chunk, row)?;
                let stop = self.eval_expr(&args[1], chunk, row)?;
                let Value::Int64(start_val) = start else {
                    return None;
                };
                let Value::Int64(end_val) = stop else {
                    return None;
                };
                let step = if args.len() == 3 {
                    let s = self.eval_expr(&args[2], chunk, row)?;
                    let Value::Int64(sv) = s else {
                        return None;
                    };
                    if sv == 0 {
                        return None;
                    }
                    sv
                } else {
                    1
                };
                let mut result = Vec::new();
                let mut current = start_val;
                if step > 0 {
                    while current <= end_val {
                        result.push(Value::Int64(current));
                        current += step;
                    }
                } else {
                    while current >= end_val {
                        result.push(Value::Int64(current));
                        current += step;
                    }
                }
                Some(Value::List(result.into()))
            }
            // --- String functions (left, right) ---
            "left" => {
                if args.len() != 2 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                let len = self.eval_expr(&args[1], chunk, row)?;
                match (&val, &len) {
                    (Value::String(s), Value::Int64(n)) => {
                        let n = (*n).max(0) as usize;
                        let result: String = s.chars().take(n).collect();
                        Some(Value::String(result.into()))
                    }
                    _ => None,
                }
            }
            "right" => {
                if args.len() != 2 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                let len = self.eval_expr(&args[1], chunk, row)?;
                match (&val, &len) {
                    (Value::String(s), Value::Int64(n)) => {
                        let n = (*n).max(0) as usize;
                        let char_count = s.chars().count();
                        let skip = char_count.saturating_sub(n);
                        let result: String = s.chars().skip(skip).collect();
                        Some(Value::String(result.into()))
                    }
                    _ => None,
                }
            }
            // --- Numeric functions (sign, log, log10, exp, e, pi) ---
            "sign" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Int64(i.signum())),
                    Value::Float64(f) => {
                        if f > 0.0 {
                            Some(Value::Int64(1))
                        } else if f < 0.0 {
                            Some(Value::Int64(-1))
                        } else {
                            Some(Value::Int64(0))
                        }
                    }
                    _ => None,
                }
            }
            "power" | "pow" => {
                if args.len() != 2 {
                    return None;
                }
                let base_val = self.eval_expr(&args[0], chunk, row)?;
                let exp_val = self.eval_expr(&args[1], chunk, row)?;
                let base = match base_val {
                    Value::Int64(i) => i as f64,
                    Value::Float64(f) => f,
                    _ => return None,
                };
                let exponent = match exp_val {
                    Value::Int64(i) => i as f64,
                    Value::Float64(f) => f,
                    _ => return None,
                };
                Some(Value::Float64(base.powf(exponent)))
            }
            "log" | "ln" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Float64((i as f64).ln())),
                    Value::Float64(f) => Some(Value::Float64(f.ln())),
                    _ => None,
                }
            }
            "log10" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Float64((i as f64).log10())),
                    Value::Float64(f) => Some(Value::Float64(f.log10())),
                    _ => None,
                }
            }
            "log2" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Float64((i as f64).log2())),
                    Value::Float64(f) => Some(Value::Float64(f.log2())),
                    _ => None,
                }
            }
            "exp" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Float64((i as f64).exp())),
                    Value::Float64(f) => Some(Value::Float64(f.exp())),
                    _ => None,
                }
            }
            "e" => Some(Value::Float64(std::f64::consts::E)),
            "pi" => Some(Value::Float64(std::f64::consts::PI)),
            // --- Trigonometric functions ---
            "sin" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Float64((i as f64).sin())),
                    Value::Float64(f) => Some(Value::Float64(f.sin())),
                    _ => None,
                }
            }
            "cos" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Float64((i as f64).cos())),
                    Value::Float64(f) => Some(Value::Float64(f.cos())),
                    _ => None,
                }
            }
            "tan" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Float64((i as f64).tan())),
                    Value::Float64(f) => Some(Value::Float64(f.tan())),
                    _ => None,
                }
            }
            "asin" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Float64((i as f64).asin())),
                    Value::Float64(f) => Some(Value::Float64(f.asin())),
                    _ => None,
                }
            }
            "acos" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Float64((i as f64).acos())),
                    Value::Float64(f) => Some(Value::Float64(f.acos())),
                    _ => None,
                }
            }
            "atan" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Float64((i as f64).atan())),
                    Value::Float64(f) => Some(Value::Float64(f.atan())),
                    _ => None,
                }
            }
            "atan2" => {
                if args.len() != 2 {
                    return None;
                }
                let y_val = self.eval_expr(&args[0], chunk, row)?;
                let x_val = self.eval_expr(&args[1], chunk, row)?;
                let y = match y_val {
                    Value::Int64(i) => i as f64,
                    Value::Float64(f) => f,
                    _ => return None,
                };
                let x = match x_val {
                    Value::Int64(i) => i as f64,
                    Value::Float64(f) => f,
                    _ => return None,
                };
                Some(Value::Float64(y.atan2(x)))
            }
            "degrees" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Float64((i as f64).to_degrees())),
                    Value::Float64(f) => Some(Value::Float64(f.to_degrees())),
                    _ => None,
                }
            }
            "radians" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Float64((i as f64).to_radians())),
                    Value::Float64(f) => Some(Value::Float64(f.to_radians())),
                    _ => None,
                }
            }
            // Temporal constructors and accessors
            "date" | "todate" => {
                if args.is_empty() {
                    return Some(Value::Date(obrain_common::types::Date::today()));
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::String(s) => obrain_common::types::Date::parse(&s).map(Value::Date),
                    Value::Timestamp(ts) => Some(Value::Date(ts.to_date())),
                    Value::Date(_) => Some(val),
                    Value::Map(m) => {
                        let year = map_int(&m, "year")? as i32;
                        let month = map_int_or(&m, "month", 1) as u32;
                        let day = map_int_or(&m, "day", 1) as u32;
                        obrain_common::types::Date::from_ymd(year, month, day).map(Value::Date)
                    }
                    _ => None,
                }
            }
            "time" | "totime" | "local_time" => {
                if args.is_empty() {
                    return Some(Value::Time(obrain_common::types::Time::now()));
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::String(s) => obrain_common::types::Time::parse(&s).map(Value::Time),
                    Value::Timestamp(ts) => Some(Value::Time(ts.to_time())),
                    Value::Time(_) => Some(val),
                    Value::Map(m) => {
                        let hour = map_int_or(&m, "hour", 0) as u32;
                        let minute = map_int_or(&m, "minute", 0) as u32;
                        let second = map_int_or(&m, "second", 0) as u32;
                        let nanosecond = map_int_or(&m, "nanosecond", 0) as u32;
                        obrain_common::types::Time::from_hms_nano(hour, minute, second, nanosecond)
                            .map(Value::Time)
                    }
                    _ => None,
                }
            }
            "datetime" | "localdatetime" | "local_datetime" | "todatetime" => {
                if args.is_empty() {
                    return Some(Value::Timestamp(obrain_common::types::Timestamp::now()));
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::String(s) => {
                        // Parse ISO datetime: try Date first, then full timestamp
                        if let Some(d) = obrain_common::types::Date::parse(&s) {
                            return Some(Value::Timestamp(d.to_timestamp()));
                        }
                        // Try full ISO format: YYYY-MM-DDTHH:MM:SS[.fff][Z|+HH:MM]
                        if let Some(pos) = s.find('T') {
                            let date_part = &s[..pos];
                            let time_part = &s[pos + 1..];
                            if let (Some(d), Some(t)) = (
                                obrain_common::types::Date::parse(date_part),
                                obrain_common::types::Time::parse(time_part),
                            ) {
                                return Some(Value::Timestamp(
                                    obrain_common::types::Timestamp::from_date_time(d, t),
                                ));
                            }
                        }
                        None
                    }
                    Value::Timestamp(_) => Some(val),
                    Value::Map(m) => {
                        let year = map_int(&m, "year")? as i32;
                        let month = map_int_or(&m, "month", 1) as u32;
                        let day = map_int_or(&m, "day", 1) as u32;
                        let hour = map_int_or(&m, "hour", 0) as u32;
                        let minute = map_int_or(&m, "minute", 0) as u32;
                        let second = map_int_or(&m, "second", 0) as u32;
                        let nanosecond = map_int_or(&m, "nanosecond", 0) as u32;
                        let date = obrain_common::types::Date::from_ymd(year, month, day)?;
                        let time = obrain_common::types::Time::from_hms_nano(
                            hour, minute, second, nanosecond,
                        )?;
                        Some(Value::Timestamp(
                            obrain_common::types::Timestamp::from_date_time(date, time),
                        ))
                    }
                    _ => None,
                }
            }
            "duration" | "toduration" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::String(s) => {
                        obrain_common::types::Duration::parse(&s).map(Value::Duration)
                    }
                    Value::Duration(_) => Some(val),
                    Value::Map(m) => {
                        let years = map_int_or(&m, "years", 0);
                        let months = map_int_or(&m, "months", 0);
                        let weeks = map_int_or(&m, "weeks", 0);
                        let days = map_int_or(&m, "days", 0);
                        let hours = map_int_or(&m, "hours", 0);
                        let minutes = map_int_or(&m, "minutes", 0);
                        let seconds = map_int_or(&m, "seconds", 0);
                        let nanoseconds = map_int_or(&m, "nanoseconds", 0);
                        let total_months = years * 12 + months;
                        let total_days = weeks * 7 + days;
                        let total_nanos = hours * 3_600_000_000_000
                            + minutes * 60_000_000_000
                            + seconds * 1_000_000_000
                            + nanoseconds;
                        Some(Value::Duration(obrain_common::types::Duration::new(
                            total_months,
                            total_days,
                            total_nanos,
                        )))
                    }
                    _ => None,
                }
            }
            "tozoneddatetime" | "zoneddatetime" | "zoned_datetime" => {
                if args.is_empty() {
                    return Some(Value::ZonedDatetime(
                        obrain_common::types::ZonedDatetime::from_timestamp_offset(
                            obrain_common::types::Timestamp::now(),
                            0,
                        ),
                    ));
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::String(s) => {
                        obrain_common::types::ZonedDatetime::parse(&s).map(Value::ZonedDatetime)
                    }
                    Value::Timestamp(ts) => Some(Value::ZonedDatetime(
                        obrain_common::types::ZonedDatetime::from_timestamp_offset(ts, 0),
                    )),
                    Value::ZonedDatetime(_) => Some(val),
                    _ => None,
                }
            }
            "tozonedtime" | "zonedtime" => {
                let val = self.eval_expr(args.first()?, chunk, row)?;
                match val {
                    Value::String(s) => {
                        let t = obrain_common::types::Time::parse(&s)?;
                        if t.offset_seconds().is_some() {
                            Some(Value::Time(t))
                        } else {
                            None
                        }
                    }
                    Value::Time(t) if t.offset_seconds().is_some() => Some(val),
                    _ => None,
                }
            }
            "current_date" | "currentdate" => {
                Some(Value::Date(obrain_common::types::Date::today()))
            }
            "current_time" | "currenttime" => Some(Value::Time(obrain_common::types::Time::now())),
            "now" | "current_timestamp" | "currenttimestamp" => {
                Some(Value::Timestamp(obrain_common::types::Timestamp::now()))
            }
            "timestamp" => {
                // Neo4j-compatible: returns milliseconds since Unix epoch as Int64
                let duration = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or(std::time::Duration::ZERO);
                Some(Value::Int64(duration.as_millis() as i64))
            }
            // Component extraction
            "year" => {
                let val = self.eval_expr(args.first()?, chunk, row)?;
                match val {
                    Value::Date(d) => Some(Value::Int64(i64::from(d.year()))),
                    Value::Timestamp(ts) => Some(Value::Int64(i64::from(ts.to_date().year()))),
                    Value::ZonedDatetime(zdt) => {
                        Some(Value::Int64(i64::from(zdt.to_local_date().year())))
                    }
                    _ => None,
                }
            }
            "month" => {
                let val = self.eval_expr(args.first()?, chunk, row)?;
                match val {
                    Value::Date(d) => Some(Value::Int64(i64::from(d.month()))),
                    Value::Timestamp(ts) => Some(Value::Int64(i64::from(ts.to_date().month()))),
                    Value::ZonedDatetime(zdt) => {
                        Some(Value::Int64(i64::from(zdt.to_local_date().month())))
                    }
                    _ => None,
                }
            }
            "day" => {
                let val = self.eval_expr(args.first()?, chunk, row)?;
                match val {
                    Value::Date(d) => Some(Value::Int64(i64::from(d.day()))),
                    Value::Timestamp(ts) => Some(Value::Int64(i64::from(ts.to_date().day()))),
                    Value::ZonedDatetime(zdt) => {
                        Some(Value::Int64(i64::from(zdt.to_local_date().day())))
                    }
                    _ => None,
                }
            }
            "hour" => {
                let val = self.eval_expr(args.first()?, chunk, row)?;
                match val {
                    Value::Time(t) => Some(Value::Int64(i64::from(t.hour()))),
                    Value::Timestamp(ts) => Some(Value::Int64(i64::from(ts.to_time().hour()))),
                    Value::ZonedDatetime(zdt) => {
                        Some(Value::Int64(i64::from(zdt.to_local_time().hour())))
                    }
                    _ => None,
                }
            }
            "minute" => {
                let val = self.eval_expr(args.first()?, chunk, row)?;
                match val {
                    Value::Time(t) => Some(Value::Int64(i64::from(t.minute()))),
                    Value::Timestamp(ts) => Some(Value::Int64(i64::from(ts.to_time().minute()))),
                    Value::ZonedDatetime(zdt) => {
                        Some(Value::Int64(i64::from(zdt.to_local_time().minute())))
                    }
                    _ => None,
                }
            }
            "second" => {
                let val = self.eval_expr(args.first()?, chunk, row)?;
                match val {
                    Value::Time(t) => Some(Value::Int64(i64::from(t.second()))),
                    Value::Timestamp(ts) => Some(Value::Int64(i64::from(ts.to_time().second()))),
                    Value::ZonedDatetime(zdt) => {
                        Some(Value::Int64(i64::from(zdt.to_local_time().second())))
                    }
                    _ => None,
                }
            }
            // --- Temporal truncation ---
            "date_trunc" | "truncate" => {
                if args.len() < 2 {
                    return None;
                }
                let unit = match self.eval_expr(&args[0], chunk, row)? {
                    Value::String(s) => s.to_lowercase(),
                    _ => return None,
                };
                let val = self.eval_expr(&args[1], chunk, row)?;
                match val {
                    Value::Date(d) => Some(Value::Date(d.truncate(&unit)?)),
                    Value::Time(t) => Some(Value::Time(t.truncate(&unit)?)),
                    Value::Timestamp(ts) => Some(Value::Timestamp(ts.truncate(&unit)?)),
                    Value::ZonedDatetime(zdt) => Some(Value::ZonedDatetime(zdt.truncate(&unit)?)),
                    _ => None,
                }
            }
            // --- Path constructor ---
            "path" => {
                if args.len() == 2 {
                    // path(nodes_list, edges_list) - construct from component lists
                    let nodes_val = self.eval_expr(&args[0], chunk, row)?;
                    let edges_val = self.eval_expr(&args[1], chunk, row)?;
                    match (&nodes_val, &edges_val) {
                        (Value::Null, _) | (_, Value::Null) => Some(Value::Null),
                        (Value::List(nodes), Value::List(edges)) => {
                            if nodes.is_empty() || edges.len() != nodes.len() - 1 {
                                return None;
                            }
                            Some(Value::Path {
                                nodes: Arc::from(nodes.as_ref()),
                                edges: Arc::from(edges.as_ref()),
                            })
                        }
                        _ => None,
                    }
                } else {
                    // path(node1, edge1, node2, ...) - alternating nodes and edges
                    if args.is_empty() || args.len().is_multiple_of(2) {
                        return None;
                    }
                    let mut nodes = Vec::with_capacity(args.len() / 2 + 1);
                    let mut edges = Vec::with_capacity(args.len() / 2);
                    for (i, arg) in args.iter().enumerate() {
                        let val = self.eval_expr(arg, chunk, row)?;
                        if i % 2 == 0 {
                            nodes.push(val);
                        } else {
                            edges.push(val);
                        }
                    }
                    Some(Value::Path {
                        nodes: nodes.into(),
                        edges: edges.into(),
                    })
                }
            }
            // --- Path decomposition functions ---
            "nodes" => {
                // nodes(path) - extracts nodes from a path value
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Path { nodes, .. } => Some(Value::List(nodes)),
                    Value::Map(map) => map.get(&PropertyKey::from("nodes")).cloned(),
                    Value::List(items) => {
                        // Legacy: alternating node, edge, node, edge, ...
                        let nodes: Vec<Value> = items.iter().step_by(2).cloned().collect();
                        Some(Value::List(nodes.into()))
                    }
                    _ => None,
                }
            }
            "edges" | "relationships" => {
                // edges(path) / relationships(path) - extracts edges from a path value
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Path { edges, .. } => Some(Value::List(edges)),
                    Value::Map(map) => map.get(&PropertyKey::from("edges")).cloned(),
                    Value::List(items) => {
                        // Legacy: alternating node, edge, node, edge, ...
                        let edges: Vec<Value> = items.iter().skip(1).step_by(2).cloned().collect();
                        Some(Value::List(edges.into()))
                    }
                    _ => None,
                }
            }
            // --- Path predicate functions (ISO GQL) ---
            "isacyclic" => {
                // isAcyclic(path) - true if no node appears more than once
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Path { nodes, .. } => {
                        let mut seen = std::collections::HashSet::new();
                        let acyclic = nodes
                            .iter()
                            .all(|n| seen.insert(HashableValue::new(n.clone())));
                        Some(Value::Bool(acyclic))
                    }
                    Value::Null => Some(Value::Null),
                    _ => None,
                }
            }
            "issimple" => {
                // isSimple(path) - true if no node repeats except possibly first == last
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Path { nodes, .. } => {
                        if nodes.is_empty() {
                            return Some(Value::Bool(true));
                        }
                        let mut seen = std::collections::HashSet::new();
                        let simple = nodes.iter().enumerate().all(|(i, n)| {
                            let hv = HashableValue::new(n.clone());
                            if !seen.insert(hv) {
                                // Duplicate allowed only if last node == first node
                                i == nodes.len() - 1 && n == &nodes[0]
                            } else {
                                true
                            }
                        });
                        Some(Value::Bool(simple))
                    }
                    Value::Null => Some(Value::Null),
                    _ => None,
                }
            }
            "istrail" => {
                // isTrail(path) - true if no edge repeats
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Path { edges, .. } => {
                        let mut seen = std::collections::HashSet::new();
                        let trail = edges
                            .iter()
                            .all(|e| seen.insert(HashableValue::new(e.clone())));
                        Some(Value::Bool(trail))
                    }
                    Value::Null => Some(Value::Null),
                    _ => None,
                }
            }
            // --- GQL ISO standard functions ---
            "normalize" => {
                // normalize(string) - returns the string as-is (NFC normalization).
                // Rust strings are valid UTF-8; full NFC normalization requires the
                // unicode-normalization crate, which is deferred to Phase 2.
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::String(s) => Some(Value::String(s)),
                    Value::Null => Some(Value::Null),
                    _ => None,
                }
            }
            "isnormalized" => {
                // IS [NFC|NFD|NFKC|NFKD] NORMALIZED - check Unicode normalization form.
                // Args: (string_value) or (string_value, form_name)
                // Default form is NFC when not specified.
                if args.is_empty() || args.len() > 2 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                let form = if args.len() == 2 {
                    match self.eval_expr(&args[1], chunk, row)? {
                        Value::String(s) => s.to_uppercase(),
                        _ => return None,
                    }
                } else {
                    "NFC".to_string()
                };
                match val {
                    Value::String(ref s) => {
                        use unicode_normalization::UnicodeNormalization;
                        let normalized = match form.as_str() {
                            "NFC" => s.nfc().collect::<String>() == s.as_ref(),
                            "NFD" => s.nfd().collect::<String>() == s.as_ref(),
                            "NFKC" => s.nfkc().collect::<String>() == s.as_ref(),
                            "NFKD" => s.nfkd().collect::<String>() == s.as_ref(),
                            _ => return None,
                        };
                        Some(Value::Bool(normalized))
                    }
                    Value::Null => Some(Value::Null),
                    _ => None,
                }
            }
            "string_join" => {
                // string_join(list, separator) - join list elements with separator
                if args.len() != 2 {
                    return None;
                }
                let list_val = self.eval_expr(&args[0], chunk, row)?;
                let sep_val = self.eval_expr(&args[1], chunk, row)?;
                match (list_val, sep_val) {
                    (Value::List(items), Value::String(sep)) => {
                        let sep_str: &str = &sep;
                        let joined: String = items
                            .iter()
                            .filter_map(|v| match v {
                                Value::String(s) => Some(s.to_string()),
                                Value::Int64(i) => Some(i.to_string()),
                                Value::Float64(f) => Some(f.to_string()),
                                Value::Bool(b) => Some(b.to_string()),
                                Value::Null => None,
                                other => Some(format!("{other}")),
                            })
                            .collect::<Vec<String>>()
                            .join(sep_str);
                        Some(Value::String(joined.into()))
                    }
                    (Value::Null, _) | (_, Value::Null) => Some(Value::Null),
                    _ => None,
                }
            }
            "session_user" => {
                // session_user() - returns the current session user
                // For embedded databases, returns a default user string
                Some(Value::String("default".into()))
            }
            // ISO/IEC 39075 Section 17.1 / Section 21: session schema/graph references
            "current_schema" => Some(self.session_context.current_schema.as_ref().map_or_else(
                || Value::String("default".into()),
                |s| Value::String(s.clone().into()),
            )),
            "current_graph" => Some(self.session_context.current_graph.as_ref().map_or_else(
                || Value::String("default".into()),
                |g| Value::String(g.clone().into()),
            )),
            "home_schema" | "home_graph" => {
                // Home schema/graph: not configurable yet, returns null
                Some(Value::Null)
            }
            // Obrain extension: info() returns database metadata as a map
            "info" => Some(self.session_context.db_info.get().clone()),
            // Obrain extension: schema() returns schema metadata as a map
            "schema" => Some(self.session_context.schema_info.get().clone()),
            "octet_length" | "byte_length" => {
                // octet_length(string) - byte length
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::String(s) => Some(Value::Int64(s.len() as i64)),
                    Value::Null => Some(Value::Null),
                    _ => None,
                }
            }
            "tolist" => {
                // toList(value) - wraps a scalar in a single-element list, or returns list as-is
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::List(_) => Some(val),
                    Value::Null => Some(Value::Null),
                    other => Some(Value::List(vec![other].into())),
                }
            }
            "totypedlist" => {
                // toTypedList(value, element_type) - coerces value to a list with typed elements
                if args.len() != 2 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                let Value::String(elem_type) = self.eval_expr(&args[1], chunk, row)? else {
                    return None;
                };
                if matches!(val, Value::Null) {
                    return Some(Value::Null);
                }
                // Wrap scalar in a list first
                let items = match val {
                    Value::List(items) => items.to_vec(),
                    other => vec![other],
                };
                // Coerce each element to the target type
                let coerced: Option<Vec<Value>> = items
                    .into_iter()
                    .map(|v| Self::coerce_to_type(v, &elem_type))
                    .collect();
                coerced.map(|v| Value::List(v.into()))
            }
            "nullif" => {
                // NULLIF(expr1, expr2) - returns NULL if expr1 = expr2, else expr1
                if args.len() != 2 {
                    return None;
                }
                let val1 = self.eval_expr(&args[0], chunk, row)?;
                let val2 = self.eval_expr(&args[1], chunk, row)?;
                // Three-valued: NULLIF(NULL, x) = NULL; NULLIF(x, NULL) = x
                if val1.is_null() || val2.is_null() {
                    Some(val1)
                } else if Self::values_equal(&val1, &val2) {
                    Some(Value::Null)
                } else {
                    Some(val1)
                }
            }
            "startnode" => {
                // startNode(r) — returns the source node of relationship r
                if args.len() != 1 {
                    return None;
                }
                let edge_id = if let FilterExpression::Variable(var) = &args[0] {
                    let col_idx = *self.variable_columns.get(var)?;
                    let col = chunk.column(col_idx)?;
                    col.get_edge_id(row)?
                } else {
                    let val = self.eval_expr(&args[0], chunk, row)?;
                    match val {
                        Value::Int64(id) => obrain_common::types::EdgeId(id as u64),
                        _ => return None,
                    }
                };
                let edge = self.resolve_edge(edge_id)?;
                let node = self.resolve_node(edge.src)?;
                Some(Value::Int64(node.id.0 as i64))
            }
            "endnode" => {
                // endNode(r) — returns the destination node of relationship r
                if args.len() != 1 {
                    return None;
                }
                let edge_id = if let FilterExpression::Variable(var) = &args[0] {
                    let col_idx = *self.variable_columns.get(var)?;
                    let col = chunk.column(col_idx)?;
                    col.get_edge_id(row)?
                } else {
                    let val = self.eval_expr(&args[0], chunk, row)?;
                    match val {
                        Value::Int64(id) => obrain_common::types::EdgeId(id as u64),
                        _ => return None,
                    }
                };
                let edge = self.resolve_edge(edge_id)?;
                let node = self.resolve_node(edge.dst)?;
                Some(Value::Int64(node.id.0 as i64))
            }
            _ => None, // Unknown function
        }
    }

    fn eval_case(
        &self,
        operand: Option<&FilterExpression>,
        when_clauses: &[(FilterExpression, FilterExpression)],
        else_clause: Option<&FilterExpression>,
        chunk: &DataChunk,
        row: usize,
    ) -> Option<Value> {
        if let Some(test_expr) = operand {
            // Simple CASE: CASE expr WHEN val1 THEN res1 ...
            // Use unwrap_or(Null) so a NULL test expression falls through to ELSE
            // rather than short-circuiting the entire CASE (NULL != anything).
            let test_val = self.eval_expr(test_expr, chunk, row).unwrap_or(Value::Null);
            for (when_expr, then_expr) in when_clauses {
                let when_val = self.eval_expr(when_expr, chunk, row).unwrap_or(Value::Null);
                // Three-valued logic: NULL never matches anything in simple CASE
                if !test_val.is_null()
                    && !when_val.is_null()
                    && Self::values_equal(&test_val, &when_val)
                {
                    return self.eval_expr(then_expr, chunk, row);
                }
            }
        } else {
            // Searched CASE: CASE WHEN cond1 THEN res1 ...
            // Use unwrap_or(Null) so a NULL/UNKNOWN condition falls through
            // to the next WHEN or ELSE (three-valued logic: only TRUE matches).
            for (when_expr, then_expr) in when_clauses {
                let when_val = self.eval_expr(when_expr, chunk, row).unwrap_or(Value::Null);
                if when_val.as_bool() == Some(true) {
                    return self.eval_expr(then_expr, chunk, row);
                }
            }
        }
        // No match - return ELSE or NULL
        if let Some(else_expr) = else_clause {
            self.eval_expr(else_expr, chunk, row)
        } else {
            Some(Value::Null)
        }
    }

    fn eval_unary_op(&self, op: UnaryFilterOp, val: Option<Value>) -> Option<Value> {
        match op {
            UnaryFilterOp::Not => {
                let v = val?.as_bool()?;
                Some(Value::Bool(!v))
            }
            UnaryFilterOp::IsNull => Some(Value::Bool(
                val.is_none() || matches!(val, Some(Value::Null)),
            )),
            UnaryFilterOp::IsNotNull => Some(Value::Bool(
                val.is_some() && !matches!(val, Some(Value::Null)),
            )),
            UnaryFilterOp::Neg => match val? {
                Value::Int64(i) => Some(Value::Int64(-i)),
                Value::Float64(f) => Some(Value::Float64(-f)),
                _ => None,
            },
        }
    }

    /// Structural equality for DISTINCT, GROUP BY, and list/map comparison.
    ///
    /// Treats NULL == NULL as `true` (grouping semantics). For SQL/GQL
    /// comparison operators, use [`eval_binary_op`] which returns UNKNOWN
    /// when either operand is NULL.
    fn values_equal(left: &Value, right: &Value) -> bool {
        match (left, right) {
            (Value::Null, Value::Null) => true,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::Int64(a), Value::Int64(b)) => a == b,
            (Value::Float64(a), Value::Float64(b)) => (a - b).abs() < f64::EPSILON,
            (Value::String(a), Value::String(b)) => a == b,
            (Value::Int64(a), Value::Float64(b)) | (Value::Float64(b), Value::Int64(a)) => {
                (*a as f64 - b).abs() < f64::EPSILON
            }
            // RDF stores numeric literals as strings; allow cross-type equality
            (Value::String(s), Value::Int64(i)) | (Value::Int64(i), Value::String(s)) => {
                s.parse::<i64>().is_ok_and(|n| n == *i)
            }
            (Value::String(s), Value::Float64(f)) | (Value::Float64(f), Value::String(s)) => {
                s.parse::<f64>().is_ok_and(|n| (n - f).abs() < f64::EPSILON)
            }
            (Value::List(a), Value::List(b)) => {
                a.len() == b.len()
                    && a.iter()
                        .zip(b.iter())
                        .all(|(x, y)| Self::values_equal(x, y))
            }
            (Value::Map(a), Value::Map(b)) => {
                a.len() == b.len()
                    && a.iter()
                        .zip(b.iter())
                        .all(|((k1, v1), (k2, v2))| k1 == k2 && Self::values_equal(v1, v2))
            }
            (
                Value::Path {
                    nodes: n1,
                    edges: e1,
                },
                Value::Path {
                    nodes: n2,
                    edges: e2,
                },
            ) => {
                n1.len() == n2.len()
                    && e1.len() == e2.len()
                    && n1
                        .iter()
                        .zip(n2.iter())
                        .all(|(a, b)| Self::values_equal(a, b))
                    && e1
                        .iter()
                        .zip(e2.iter())
                        .all(|(a, b)| Self::values_equal(a, b))
            }
            _ => false,
        }
    }

    /// Checks if a value matches a GQL type name (used by IS TYPED).
    fn value_matches_type(val: &Value, type_name: &str) -> bool {
        match type_name {
            "BOOLEAN" | "BOOL" => matches!(val, Value::Bool(_)),
            "INTEGER" | "INT" | "INT64" => matches!(val, Value::Int64(_)),
            "FLOAT" | "FLOAT64" | "DOUBLE" => matches!(val, Value::Float64(_)),
            "STRING" => matches!(val, Value::String(_)),
            "LIST" => matches!(val, Value::List(_)),
            "MAP" | "RECORD" => matches!(val, Value::Map(_)),
            "NULL" => matches!(val, Value::Null),
            "DATE" => matches!(val, Value::Date(_)),
            "TIME" => matches!(val, Value::Time(_)),
            "DATETIME" | "TIMESTAMP" => matches!(val, Value::Timestamp(_)),
            "DURATION" => matches!(val, Value::Duration(_)),
            "PATH" => matches!(val, Value::Path { .. }),
            "NODE" | "EDGE" | "GRAPH" => false, // element refs not stored as values
            _ => false,
        }
    }

    /// Coerces a value to a target GQL type. Returns None if coercion is impossible.
    fn coerce_to_type(val: Value, type_name: &str) -> Option<Value> {
        match type_name.to_uppercase().as_str() {
            "INTEGER" | "INT" | "INT64" => match val {
                Value::Int64(_) => Some(val),
                Value::Float64(f) => Some(Value::Int64(f as i64)),
                Value::String(ref s) => s.parse::<i64>().ok().map(Value::Int64),
                Value::Bool(b) => Some(Value::Int64(i64::from(b))),
                _ => None,
            },
            "FLOAT" | "FLOAT64" | "DOUBLE" => match val {
                Value::Float64(_) => Some(val),
                Value::Int64(i) => Some(Value::Float64(i as f64)),
                Value::String(ref s) => s.parse::<f64>().ok().map(Value::Float64),
                _ => None,
            },
            "STRING" => match val {
                Value::String(_) => Some(val),
                other => Some(Value::String(other.to_string().into())),
            },
            "BOOLEAN" | "BOOL" => match val {
                Value::Bool(_) => Some(val),
                Value::String(ref s) => match s.to_lowercase().as_str() {
                    "true" => Some(Value::Bool(true)),
                    "false" => Some(Value::Bool(false)),
                    _ => None,
                },
                _ => None,
            },
            _ => {
                // For unknown types, keep value as-is if it already matches
                if Self::value_matches_type(&val, &type_name.to_uppercase()) {
                    Some(val)
                } else {
                    None
                }
            }
        }
    }

    fn compare_values(&self, left: &Value, right: &Value) -> Option<i32> {
        match (left, right) {
            (Value::Int64(a), Value::Int64(b)) => Some(a.cmp(b) as i32),
            (Value::Float64(a), Value::Float64(b)) => {
                if a < b {
                    Some(-1)
                } else if a > b {
                    Some(1)
                } else {
                    Some(0)
                }
            }
            (Value::String(a), Value::String(b)) => Some(a.cmp(b) as i32),
            (Value::Int64(a), Value::Float64(b)) => (*a as f64).partial_cmp(b).map(|o| o as i32),
            (Value::Float64(a), Value::Int64(b)) => a.partial_cmp(&(*b as f64)).map(|o| o as i32),
            // RDF stores numeric literals as strings; allow cross-type comparison
            (Value::String(s), Value::Int64(i)) => s
                .parse::<f64>()
                .ok()
                .and_then(|n| n.partial_cmp(&(*i as f64)).map(|o| o as i32)),
            (Value::Int64(i), Value::String(s)) => s
                .parse::<f64>()
                .ok()
                .and_then(|n| (*i as f64).partial_cmp(&n).map(|o| o as i32)),
            (Value::String(s), Value::Float64(f)) => s
                .parse::<f64>()
                .ok()
                .and_then(|n| n.partial_cmp(f).map(|o| o as i32)),
            (Value::Float64(f), Value::String(s)) => s
                .parse::<f64>()
                .ok()
                .and_then(|n| f.partial_cmp(&n).map(|o| o as i32)),
            // Temporal comparisons
            (Value::Timestamp(a), Value::Timestamp(b)) => Some(a.cmp(b) as i32),
            (Value::Date(a), Value::Date(b)) => Some(a.cmp(b) as i32),
            (Value::Time(a), Value::Time(b)) => Some(a.cmp(b) as i32),
            _ => None,
        }
    }
}

impl Predicate for ExpressionPredicate {
    fn evaluate(&self, chunk: &DataChunk, row: usize) -> bool {
        match self.eval(chunk, row) {
            Some(Value::Bool(b)) => b,
            _ => false,
        }
    }
}

/// A filter operator that applies a predicate to filter rows.
pub struct FilterOperator {
    /// Child operator to read from.
    child: Box<dyn Operator>,
    /// Predicate to apply.
    predicate: Box<dyn Predicate>,
}

impl FilterOperator {
    /// Creates a new filter operator.
    pub fn new(child: Box<dyn Operator>, predicate: Box<dyn Predicate>) -> Self {
        Self { child, predicate }
    }
}

impl Operator for FilterOperator {
    fn next(&mut self) -> OperatorResult {
        loop {
            // Get next chunk from child
            let Some(mut chunk) = self.child.next()? else {
                return Ok(None);
            };

            // Zone map check: skip entire chunk if no rows can match
            if let Some(hints) = chunk.zone_hints()
                && !self.predicate.might_match_chunk(hints)
            {
                continue; // Skip entire chunk - zone map proves no matches
            }

            // Apply predicate to create selection vector, respecting any
            // existing selection from child operators (stacked filters).
            let selection = if let Some(existing) = chunk.selection() {
                let mut sel = SelectionVector::new_empty();
                for pos in 0..existing.len() {
                    if let Some(row) = existing.get(pos)
                        && self.predicate.evaluate(&chunk, row)
                    {
                        sel.push(row);
                    }
                }
                sel
            } else {
                let count = chunk.total_row_count();
                SelectionVector::from_predicate(count, |row| self.predicate.evaluate(&chunk, row))
            };

            // If nothing passes, skip to next chunk
            if selection.is_empty() {
                continue;
            }

            chunk.set_selection(selection);
            return Ok(Some(chunk));
        }
    }

    fn reset(&mut self) {
        self.child.reset();
    }

    fn name(&self) -> &'static str {
        "Filter"
    }
}

/// Escapes a character for use in a regex pattern.
#[cfg(any(feature = "regex", feature = "regex-lite"))]
fn regex_escape_char(ch: char, out: &mut String) {
    if ".+*?^${}()|[]\\".contains(ch) {
        out.push('\\');
    }
    out.push(ch);
}

/// Converts a `Value` to its string representation for concatenation.
fn value_to_string(val: &Value) -> Option<String> {
    match val {
        Value::Int64(i) => Some(i.to_string()),
        Value::Float64(f) => Some(f.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        Value::String(s) => Some(s.to_string()),
        Value::Null => None,
        _ => Some(format!("{val}")),
    }
}
