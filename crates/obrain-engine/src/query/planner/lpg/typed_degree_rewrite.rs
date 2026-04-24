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

/// T17i T3 — gate a peer-label constraint against the store's
/// T17i T2 histogram. Returns `true` iff the rewrite can safely
/// substitute an expand+count pipeline that filters the peer on
/// `peer_label` with a typed-degree lookup that is label-agnostic.
///
/// The invariant checked : every live edge of `edge_type` in the
/// given direction lands on a node whose label set is exactly
/// `{peer_label}`. When the histogram shows a singleton matching,
/// the count is identical across the two paths.
///
/// `direction` here is interpreted **from the anchor's POV** :
/// Outgoing means the anchor is the source (`edge_target_labels`
/// applies to the peer) ; Incoming means the anchor is the target
/// (`edge_source_labels` applies to the peer). `Both` / `Separate`
/// require BOTH histograms to be singleton matches.
fn validate_peer_label(
    store: &dyn obrain_core::graph::GraphStore,
    edge_type: &str,
    direction: TypedDegreeDirection,
    peer_label: &str,
) -> bool {
    if !store.supports_edge_label_histogram() {
        return false;
    }
    let singleton_match = |set: std::collections::HashSet<String>| -> bool {
        set.len() == 1 && set.contains(peer_label)
    };
    match direction {
        TypedDegreeDirection::Outgoing => {
            singleton_match(store.edge_target_labels(edge_type))
        }
        TypedDegreeDirection::Incoming => {
            singleton_match(store.edge_source_labels(edge_type))
        }
        TypedDegreeDirection::Both | TypedDegreeDirection::Separate => {
            singleton_match(store.edge_target_labels(edge_type))
                && singleton_match(store.edge_source_labels(edge_type))
        }
    }
}

/// Extracts `hasLabel(<peer_var>, "<label>")` from a FunctionCall
/// expression. Returns the peer variable name and the label string
/// if the expression has exactly that shape.
fn extract_has_label(expr: &LogicalExpression) -> Option<(String, String)> {
    let LogicalExpression::FunctionCall { name, args, .. } = expr else {
        return None;
    };
    if !name.eq_ignore_ascii_case("hasLabel") || args.len() != 2 {
        return None;
    }
    let var = match &args[0] {
        LogicalExpression::Variable(v) => v.clone(),
        _ => return None,
    };
    let label = match &args[1] {
        LogicalExpression::Literal(obrain_common::types::Value::String(s)) => s.to_string(),
        _ => return None,
    };
    Some((var, label))
}

/// Captured `(expand, peer_label_constraint)` pair. The peer label
/// constraint is `Some((var, L))` when the source expresses `(x:L)`
/// via a `Filter(hasLabel(x, L))` wrapper or a `NodeScan(x:L)`
/// input. The matcher must then check the corpus invariant via
/// `validate_peer_label` before accepting the rewrite.
type ExpandWithPeerLabel<'a> = (&'a ExpandOp, Option<(String, String)>);

/// Returns the `Expand` at the root of the branch plus — when
/// present — the peer-label constraint observed either in a
/// `Filter(hasLabel(peer, L))` wrapper above the Expand or in a
/// labelled `NodeScan(peer:L)` that the Expand reads from.
///
/// The T17i T3 rewrite passes the constraint (if any) to
/// `validate_peer_label` which gates the routing decision on the
/// store's `edge_target_labels` / `edge_source_labels` histograms.
/// In T17h T9c this function rejected any label constraint outright
/// for safety ; T17i T3 lifts the restriction, but only when the
/// T17i T2 histogram proves the constraint matches the corpus.
fn extract_expand_underneath(op: &LogicalOperator) -> Option<ExpandWithPeerLabel<'_>> {
    // Case 1 : `Filter(hasLabel(peer, L)) → Expand(...)`.
    if let LogicalOperator::Filter(filter) = op
        && let Some((var, label)) = extract_has_label(&filter.predicate)
        && let LogicalOperator::Expand(expand) = filter.input.as_ref()
    {
        return Some((expand, Some((var, label))));
    }
    // Case 2 : bare `Expand(...)` with a labelled NodeScan as input
    // (happens for the inverse direction in dual patterns :
    // `NodeScan(dependent:File) → Expand(dependent → f)`). The label
    // on the source node is the peer-label constraint from that
    // branch's perspective.
    if let LogicalOperator::Expand(expand) = op {
        if let LogicalOperator::NodeScan(scan) = expand.input.as_ref() {
            let label = scan.label.clone();
            return Some((
                expand,
                label.map(|l| (scan.variable.clone(), l)),
            ));
        }
    }
    None
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
fn extract_typed_degree_pattern(
    store: &dyn obrain_core::graph::GraphStore,
    limit: &LimitOp,
) -> Option<TypedDegreePattern> {
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
        let (expand, peer_label) = extract_expand_underneath(join.right.as_ref())?;
        if expand.edge_types.len() != 1 {
            return None;
        }
        let edge_type = expand.edge_types[0].clone();
        let direction = classify_direction(expand, &group_var)?;
        // T17i T3 : if the source carries a peer-label constraint,
        // gate the rewrite on the store's histogram. Only accept when
        // the histogram proves every edge of this type lands on the
        // requested label.
        if let Some((_, pl)) = peer_label {
            if pl != label {
                return None; // anchor label and peer label must agree
            }
            if !validate_peer_label(store, &edge_type, direction, &pl) {
                return None;
            }
        }
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
        let (expand1, peer_label1) = extract_expand_underneath(inner.right.as_ref())?;
        let (expand2, peer_label2) = extract_expand_underneath(outer.right.as_ref())?;
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
        // T17i T3 : validate each peer-label constraint against the
        // histogram for its own branch direction. Both branches must
        // pass (or have no constraint).
        for (pl_opt, dir) in [(&peer_label1, dir1), (&peer_label2, dir2)] {
            if let Some((_, pl)) = pl_opt {
                if *pl != label {
                    return None;
                }
                if !validate_peer_label(store, &edge_type, dir, pl) {
                    return None;
                }
            }
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

    // T17i T3 : pass the store to the extractor so peer-label
    // constraints can be gated on the `edge_target_labels` /
    // `edge_source_labels` histograms.
    let store_ref: &dyn obrain_core::graph::GraphStore =
        planner.store.as_ref() as &dyn obrain_core::graph::GraphStore;
    let pattern = extract_typed_degree_pattern(store_ref, limit)?;

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
