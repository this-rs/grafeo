//! T17h T9c — Cypher planner rewrite : `try_plan_typed_degree_topk`.
//!
//! Recognises the canonical `most_connected_files` pattern (and its
//! single-direction variant) and substitutes the slow
//! `LabelScan → LeftJoin(Expand[:TYPE]) → Aggregate(count DISTINCT) →
//! Sort → Limit` pipeline with a direct [`TypedDegreeTopKOperator`]
//! that pulls per-node counts from the T17h T8 per-edge-type degree
//! column (O(1) atomic load).
//!
//! ### Invariants
//!
//! - `supports_typed_degree()` is true on the store (checked first)
//! - `OBRAIN_DISABLE_TYPED_DEGREE_TOPK` env var unset (kill switch)
//! - Sort key is DESC and references the sum-alias (single-direction
//!   case : the count alias itself ; dual-direction : the alias of
//!   the `Binary(Add, alias1, alias2)` projection)
//! - Every Aggregate function is `CountNonNull` with `distinct: true`
//! - Aggregate `group_by` is `[Variable(group_var)]`
//! - NodeScan anchor variable matches `group_var` and carries a label
//! - Both Expands (dual case) share the same edge-type and are
//!   anchored on `group_var` in opposite directions
//!
//! ### Counter
//!
//! [`TYPED_DEGREE_REWRITE_COUNTER`](obrain_core::execution::operators::TYPED_DEGREE_REWRITE_COUNTER)
//! is incremented on every successful match — T9b snapshot tests
//! read it to verify the rewrite fired.

use std::sync::atomic::Ordering;

use obrain_core::execution::operators::{
    BinaryFilterOp, FilterExpression, Operator, ProjectExpr, ProjectOperator,
    TYPED_DEGREE_REWRITE_COUNTER, TypedDegreeDirection, TypedDegreeTopKOperator,
};
use obrain_common::types::LogicalType;

use super::{
    AggregateOp, BinaryOp, ExpandDirection, ExpandOp, LimitOp, LogicalAggregateFunction,
    LogicalExpression, LogicalOperator, ReturnOp, SortOrder,
};
use crate::query::plan::AggregateExpr;

/// Extracted shape of the matched pattern.
enum PatternShape {
    /// `OPTIONAL MATCH (f)-[:T]->(x)` or `()-[:T]->(f)` — one branch.
    Single {
        /// `Outgoing` or `Incoming` — **never** `Both` / `Separate`
        /// for single-branch patterns.
        direction: TypedDegreeDirection,
        /// Alias produced by the Aggregate (e.g. `"imports"`).
        count_alias: String,
    },
    /// Dual-direction : two OPTIONAL MATCH branches sharing the same
    /// edge type, one outgoing from `f`, one incoming to `f`.
    Dual {
        out_alias: String,
        in_alias: String,
        /// Alias of the `Binary(Add, out_alias, in_alias)` projection
        /// in the `Return` items (e.g. `"connections"`).
        connections_alias: String,
    },
}

/// Fully-extracted pattern, enough to build the replacement operator.
struct TypedDegreePattern {
    shape: PatternShape,
    label: String,
    edge_type: String,
    /// Variable bound by the NodeScan anchor (e.g. `"f"`).
    group_var: String,
    /// Variable referenced in the Return for property access (usually
    /// the same as `group_var`).
    prop_var: String,
    /// Property name accessed in the Return (e.g. `"path"`).
    prop_name: String,
    /// Top-K size.
    k: usize,
}

/// `true` iff `expr` is `Binary(Add, Variable(left), Variable(right))`
/// (commutative) and the alias pair matches.
fn is_binary_add_of(expr: &LogicalExpression, a: &str, b: &str) -> bool {
    let LogicalExpression::Binary { left, op, right } = expr else {
        return false;
    };
    if !matches!(op, BinaryOp::Add) {
        return false;
    }
    let lv = match left.as_ref() {
        LogicalExpression::Variable(v) => v.as_str(),
        _ => return false,
    };
    let rv = match right.as_ref() {
        LogicalExpression::Variable(v) => v.as_str(),
        _ => return false,
    };
    (lv == a && rv == b) || (lv == b && rv == a)
}

fn is_count_distinct(a: &AggregateExpr) -> bool {
    matches!(a.function, LogicalAggregateFunction::CountNonNull) && a.distinct
}

/// Returns the `Expand` directly at the root, **provided** there is
/// no label constraint on the peer variable — either via a `Filter`
/// wrapper (e.g. auto-inserted `hasLabel(peer, L)`) or via a
/// `NodeScan(peer:L)` as the Expand's input.
///
/// Rationale : the T8 `out_degree_by_type(node, edge_type)` counts
/// **every** outgoing edge of that type regardless of the peer's
/// label. If the Cypher source restricts the peer via `(x:Label)`,
/// the slow path filters to edges whose peer has that label — the
/// two semantics diverge whenever the corpus has mixed-label peers
/// for the given edge type. Reject → fall back to the slow path, so
/// the matcher never silently returns wrong rows.
fn extract_expand_underneath(op: &LogicalOperator) -> Option<&ExpandOp> {
    let expand = match op {
        LogicalOperator::Expand(e) => e,
        _ => return None,
    };
    // Validate the Expand's input : must be a NodeScan with no label,
    // or a NodeScan whose label matches the anchor (handled by the
    // caller via `group_var` check). We accept `NodeScan(var:*)` —
    // no label — as the placeholder under an outgoing Expand. We
    // reject a `NodeScan(var:L)` with a non-empty label because the
    // caller cannot check from here whether L matches the anchor
    // semantically. For safety we only allow the anchor's
    // `NodeScan` under the top-level LeftJoin branch, never under
    // an Expand.
    match expand.input.as_ref() {
        LogicalOperator::NodeScan(inner) if inner.label.is_none() => Some(expand),
        _ => None, // any non-bare NodeScan, or any Filter / other wrapper
    }
}

/// Given an Expand anchored on `group_var`, returns the
/// direction this should be counted as from `group_var`'s perspective :
/// - `(group)-[:T]->(x)` with direction Outgoing → `Outgoing`
/// - `(x)-[:T]->(group)` with direction Outgoing (source ≠ group) →
///   `Incoming` (it's an incoming edge from `group`'s POV)
/// - Any other arrangement : `None`
fn classify_direction(expand: &ExpandOp, group_var: &str) -> Option<TypedDegreeDirection> {
    match (expand.from_variable == group_var, expand.to_variable == group_var) {
        (true, false) => match expand.direction {
            ExpandDirection::Outgoing => Some(TypedDegreeDirection::Outgoing),
            ExpandDirection::Incoming => Some(TypedDegreeDirection::Incoming),
            ExpandDirection::Both => None,
        },
        (false, true) => match expand.direction {
            ExpandDirection::Outgoing => Some(TypedDegreeDirection::Incoming),
            ExpandDirection::Incoming => Some(TypedDegreeDirection::Outgoing),
            ExpandDirection::Both => None,
        },
        _ => None, // expand doesn't anchor on group_var
    }
}

/// Extracts the top-K pattern rooted at a `LimitOp`, returning the
/// fully-populated [`TypedDegreePattern`] or `None` if any constraint
/// is violated. Silent fallback — never panics.
fn extract_typed_degree_pattern(limit: &LimitOp) -> Option<TypedDegreePattern> {
    // Limit → Sort
    let sort = match limit.input.as_ref() {
        LogicalOperator::Sort(s) => s,
        _ => return None,
    };
    if sort.keys.len() != 1 {
        return None;
    }
    if sort.keys[0].order != SortOrder::Descending {
        return None;
    }
    // Sort → Return
    let ret: &ReturnOp = match sort.input.as_ref() {
        LogicalOperator::Return(r) => r,
        _ => return None,
    };
    if ret.distinct {
        return None;
    }
    // Return → Aggregate
    let agg: &AggregateOp = match ret.input.as_ref() {
        LogicalOperator::Aggregate(a) => a,
        _ => return None,
    };
    if agg.having.is_some() {
        return None;
    }
    if agg.group_by.len() != 1 {
        return None;
    }
    let group_var = match &agg.group_by[0] {
        LogicalExpression::Variable(v) => v.clone(),
        _ => return None,
    };
    // All aggregates must be count(DISTINCT Variable(...)) with an alias.
    if agg.aggregates.is_empty() || agg.aggregates.len() > 2 {
        return None;
    }
    // Ordered (alias, variable) pairs as they appear in the aggregate.
    let mut alias_vars: Vec<(String, String)> = Vec::new();
    for a in &agg.aggregates {
        if !is_count_distinct(a) {
            return None;
        }
        let v = match &a.expression {
            Some(LogicalExpression::Variable(v)) => v.clone(),
            _ => return None,
        };
        let alias = a.alias.clone()?;
        alias_vars.push((alias, v));
    }

    // Pattern classification + LeftJoin traversal.
    let pattern = if agg.aggregates.len() == 1 {
        // Single : Aggregate → LeftJoin { left: NodeScan, right: Expand }
        let join = match agg.input.as_ref() {
            LogicalOperator::LeftJoin(j) => j,
            _ => return None,
        };
        let scan = match join.left.as_ref() {
            LogicalOperator::NodeScan(n) => n,
            _ => return None,
        };
        if scan.variable != group_var {
            return None;
        }
        let label = scan.label.as_ref()?.clone();
        let expand = extract_expand_underneath(join.right.as_ref())?;
        if expand.edge_types.len() != 1 {
            return None;
        }
        let edge_type = expand.edge_types[0].clone();
        let direction = classify_direction(expand, &group_var)?;
        // Sort key must reference the single count alias.
        let sort_var = match &sort.keys[0].expression {
            LogicalExpression::Variable(v) => v.clone(),
            _ => return None,
        };
        if sort_var != alias_vars[0].0 {
            return None;
        }
        // Return items : expect at least [<f.prop>, Variable(alias)].
        // For the single variant we accept any Return shape that at
        // minimum surfaces the alias ; the downstream Project will
        // rebuild the exact columns.
        let (prop_var, prop_name) = extract_single_property_from_return(ret, &group_var)?;
        TypedDegreePattern {
            shape: PatternShape::Single {
                direction,
                count_alias: alias_vars[0].0.clone(),
            },
            label,
            edge_type,
            group_var,
            prop_var,
            prop_name,
            k: limit.count.try_value().ok()?,
        }
    } else {
        // Dual : Aggregate → LeftJoin { left: LeftJoin { left: NodeScan, right: Expand1 }, right: Expand2 }
        let outer = match agg.input.as_ref() {
            LogicalOperator::LeftJoin(j) => j,
            _ => return None,
        };
        let inner = match outer.left.as_ref() {
            LogicalOperator::LeftJoin(j) => j,
            _ => return None,
        };
        let scan = match inner.left.as_ref() {
            LogicalOperator::NodeScan(n) => n,
            _ => return None,
        };
        if scan.variable != group_var {
            return None;
        }
        let label = scan.label.as_ref()?.clone();
        let expand1 = extract_expand_underneath(inner.right.as_ref())?;
        let expand2 = extract_expand_underneath(outer.right.as_ref())?;
        // Both expands must share exactly one edge type.
        if expand1.edge_types.len() != 1 || expand2.edge_types.len() != 1 {
            return None;
        }
        if expand1.edge_types[0] != expand2.edge_types[0] {
            return None;
        }
        let edge_type = expand1.edge_types[0].clone();
        // One should classify as Outgoing from group_var, the other Incoming.
        let dir1 = classify_direction(expand1, &group_var)?;
        let dir2 = classify_direction(expand2, &group_var)?;
        if dir1 == dir2 {
            return None;
        }
        // Map the two aggregate aliases to out / in depending on which
        // expand contributed to each. The aliases are ordered in
        // `alias_vars` by position in the Aggregate — we need to
        // identify which alias came from expand1 vs expand2.
        //
        // Convention : the inner LeftJoin's expand feeds the first
        // aggregate (see the canonical query's translation). We rely
        // on `alias_vars[0]` <> expand1 and `alias_vars[1]` <> expand2.
        // The variable referenced by each aggregate's CountNonNull
        // should match the `to_variable` or `from_variable` of its
        // Expand (whichever is not group_var).
        let expand1_target = if expand1.from_variable == group_var {
            &expand1.to_variable
        } else {
            &expand1.from_variable
        };
        let expand2_target = if expand2.from_variable == group_var {
            &expand2.to_variable
        } else {
            &expand2.from_variable
        };
        let (alias1_matches_expand1, alias2_matches_expand2) = (
            &alias_vars[0].1 == expand1_target,
            &alias_vars[1].1 == expand2_target,
        );
        let (alias1_matches_expand2, alias2_matches_expand1) = (
            &alias_vars[0].1 == expand2_target,
            &alias_vars[1].1 == expand1_target,
        );
        let (alias_for_dir1, alias_for_dir2) =
            if alias1_matches_expand1 && alias2_matches_expand2 {
                (alias_vars[0].0.clone(), alias_vars[1].0.clone())
            } else if alias1_matches_expand2 && alias2_matches_expand1 {
                (alias_vars[1].0.clone(), alias_vars[0].0.clone())
            } else {
                return None;
            };
        let (out_alias, in_alias) = match (dir1, dir2) {
            (TypedDegreeDirection::Outgoing, TypedDegreeDirection::Incoming) => {
                (alias_for_dir1, alias_for_dir2)
            }
            (TypedDegreeDirection::Incoming, TypedDegreeDirection::Outgoing) => {
                (alias_for_dir2, alias_for_dir1)
            }
            _ => return None,
        };
        // Return items must include a Binary(Add, out_alias, in_alias) AS X
        // and the sort key must reference X.
        let connections_alias = find_add_alias(ret, &out_alias, &in_alias)?;
        let sort_var = match &sort.keys[0].expression {
            LogicalExpression::Variable(v) => v.clone(),
            _ => return None,
        };
        if sort_var != connections_alias {
            return None;
        }
        let (prop_var, prop_name) = extract_single_property_from_return(ret, &group_var)?;
        TypedDegreePattern {
            shape: PatternShape::Dual {
                out_alias,
                in_alias,
                connections_alias,
            },
            label,
            edge_type,
            group_var,
            prop_var,
            prop_name,
            k: limit.count.try_value().ok()?,
        }
    };

    Some(pattern)
}

/// Scans `ret.items` for a single `LogicalExpression::Property
/// { variable = group_var, property: X }` item and returns `(var,
/// property)`. Returns `None` if more than one property on group_var
/// is projected or if none is found.
fn extract_single_property_from_return(
    ret: &ReturnOp,
    group_var: &str,
) -> Option<(String, String)> {
    let mut found: Option<(String, String)> = None;
    for item in &ret.items {
        if let LogicalExpression::Property { variable, property } = &item.expression
            && variable == group_var
        {
            if found.is_some() {
                return None; // multiple properties — out of narrow scope
            }
            found = Some((variable.clone(), property.clone()));
        }
    }
    found
}

/// Scans `ret.items` for a `Binary(Add, Variable(a), Variable(b))` item
/// (commutative) where {a, b} = {`out_alias`, `in_alias`}. Returns the
/// item's alias on success.
fn find_add_alias(
    ret: &ReturnOp,
    out_alias: &str,
    in_alias: &str,
) -> Option<String> {
    for item in &ret.items {
        if is_binary_add_of(&item.expression, out_alias, in_alias) {
            return item.alias.clone();
        }
    }
    None
}

/// Entry point — called from `plan_limit`. Returns `Some((operator,
/// output_columns))` when the rewrite fires, `None` otherwise.
pub(super) fn try_plan_typed_degree_topk(
    planner: &super::Planner,
    limit: &LimitOp,
) -> Option<(Box<dyn Operator>, Vec<String>)> {
    // Kill switch — tests can force the slow path via env var.
    if std::env::var("OBRAIN_DISABLE_TYPED_DEGREE_TOPK").is_ok() {
        return None;
    }
    // Backend must support T8 per-edge-type degree lookups.
    if !planner.store.supports_typed_degree() {
        return None;
    }

    let pattern = extract_typed_degree_pattern(limit)?;

    // Build the physical pipeline.
    let store = planner.store.clone() as std::sync::Arc<dyn obrain_core::graph::GraphStore>;
    let (operator, output_columns) = build_replacement(&pattern, store);

    TYPED_DEGREE_REWRITE_COUNTER.fetch_add(1, Ordering::Relaxed);
    Some((operator, output_columns))
}

/// Constructs the replacement operator tree :
/// - For `Single` : `TypedDegreeTopK(direction)` → `Project(<prop>,
///   <count_alias>)`.
/// - For `Dual` : `TypedDegreeTopK(Separate)` → `Project(<prop>,
///   <out_alias>, <in_alias>, <connections_alias>)`.
fn build_replacement(
    pattern: &TypedDegreePattern,
    store: std::sync::Arc<dyn obrain_core::graph::GraphStore>,
) -> (Box<dyn Operator>, Vec<String>) {
    match &pattern.shape {
        PatternShape::Single {
            direction,
            count_alias,
        } => {
            let scan = TypedDegreeTopKOperator::new(
                store.clone(),
                Some(pattern.label.clone()),
                Some(pattern.edge_type.clone()),
                *direction,
                pattern.k,
            );
            // Projections : [<prop>, <count_alias>]
            let projections = vec![
                ProjectExpr::PropertyAccess {
                    column: 0,
                    property: pattern.prop_name.clone(),
                },
                ProjectExpr::Column(1),
            ];
            let output_types = vec![LogicalType::Any, LogicalType::Int64];
            let project = ProjectOperator::with_store(
                Box::new(scan),
                projections,
                output_types,
                store,
            );
            let cols = vec![
                format!("{}_{}", pattern.prop_var, pattern.prop_name),
                count_alias.clone(),
            ];
            (Box::new(project), cols)
        }
        PatternShape::Dual {
            out_alias,
            in_alias,
            connections_alias,
        } => {
            let scan = TypedDegreeTopKOperator::new(
                store.clone(),
                Some(pattern.label.clone()),
                Some(pattern.edge_type.clone()),
                TypedDegreeDirection::Separate,
                pattern.k,
            );
            // Separate outputs [Node, Int64 out, Int64 in].
            // Projections : [<prop>, <out>, <in>, <out+in> AS connections]
            let mut var_cols = std::collections::HashMap::new();
            var_cols.insert(out_alias.clone(), 1usize);
            var_cols.insert(in_alias.clone(), 2usize);
            let add_expr = FilterExpression::Binary {
                left: Box::new(FilterExpression::Variable(out_alias.clone())),
                op: BinaryFilterOp::Add,
                right: Box::new(FilterExpression::Variable(in_alias.clone())),
            };
            let projections = vec![
                ProjectExpr::PropertyAccess {
                    column: 0,
                    property: pattern.prop_name.clone(),
                },
                ProjectExpr::Column(1),
                ProjectExpr::Column(2),
                ProjectExpr::Expression {
                    expr: add_expr,
                    variable_columns: var_cols,
                },
            ];
            let output_types = vec![
                LogicalType::Any,
                LogicalType::Int64,
                LogicalType::Int64,
                LogicalType::Int64,
            ];
            let project = ProjectOperator::with_store(
                Box::new(scan),
                projections,
                output_types,
                store,
            );
            let cols = vec![
                format!("{}_{}", pattern.prop_var, pattern.prop_name),
                out_alias.clone(),
                in_alias.clone(),
                connections_alias.clone(),
            ];
            (Box::new(project), cols)
        }
    }
}
