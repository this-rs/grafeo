//! Shared planning utilities for LPG and RDF planners.
//!
//! These free functions build physical operators from pre-planned children,
//! eliminating duplication between `Planner` (LPG) and `RdfPlanner`.
//! Each function takes already-planned input operators and column lists,
//! plus a schema derivation function to handle LPG vs RDF type differences.

use crate::query::plan::LogicalExpression;
use grafeo_common::types::LogicalType;
use grafeo_common::utils::error::{Error, Result};
use grafeo_core::execution::operators::{
    DistinctOperator, ExceptOperator, HashJoinOperator, IntersectOperator,
    JoinType as PhysicalJoinType, LimitOperator, Operator, OtherwiseOperator, ProjectExpr,
    ProjectOperator, SkipOperator, UnionOperator,
};

/// Builds a LIMIT physical operator.
pub(crate) fn build_limit(
    input: Box<dyn Operator>,
    columns: Vec<String>,
    count: usize,
    schema: Vec<LogicalType>,
) -> (Box<dyn Operator>, Vec<String>) {
    let operator = Box::new(LimitOperator::new(input, count, schema));
    (operator, columns)
}

/// Builds a SKIP physical operator.
pub(crate) fn build_skip(
    input: Box<dyn Operator>,
    columns: Vec<String>,
    count: usize,
    schema: Vec<LogicalType>,
) -> (Box<dyn Operator>, Vec<String>) {
    let operator = Box::new(SkipOperator::new(input, count, schema));
    (operator, columns)
}

/// Builds a DISTINCT physical operator.
///
/// Handles both full-row dedup and column-specific dedup (when `distinct.columns` is set).
pub(crate) fn build_distinct(
    input: Box<dyn Operator>,
    columns: Vec<String>,
    distinct_columns: Option<&[String]>,
    schema: Vec<LogicalType>,
) -> (Box<dyn Operator>, Vec<String>) {
    let operator: Box<dyn Operator> = if let Some(dist_cols) = distinct_columns {
        let col_indices: Vec<usize> = dist_cols
            .iter()
            .filter_map(|name| columns.iter().position(|c| c == name))
            .collect();
        if col_indices.is_empty() {
            Box::new(DistinctOperator::new(input, schema))
        } else {
            Box::new(DistinctOperator::on_columns(input, col_indices, schema))
        }
    } else {
        Box::new(DistinctOperator::new(input, schema))
    };
    (operator, columns)
}

/// Builds a UNION physical operator from multiple pre-planned inputs.
pub(crate) fn build_union(
    inputs: Vec<Box<dyn Operator>>,
    columns: Vec<String>,
    schema: Vec<LogicalType>,
) -> Result<(Box<dyn Operator>, Vec<String>)> {
    if inputs.is_empty() {
        return Err(Error::Internal(
            "Union requires at least one input".to_string(),
        ));
    }
    let operator = Box::new(UnionOperator::new(inputs, schema));
    Ok((operator, columns))
}

/// Builds an EXCEPT physical operator.
pub(crate) fn build_except(
    left: Box<dyn Operator>,
    right: Box<dyn Operator>,
    columns: Vec<String>,
    all: bool,
    schema: Vec<LogicalType>,
) -> (Box<dyn Operator>, Vec<String>) {
    let operator = Box::new(ExceptOperator::new(left, right, all, schema));
    (operator, columns)
}

/// Builds an INTERSECT physical operator.
pub(crate) fn build_intersect(
    left: Box<dyn Operator>,
    right: Box<dyn Operator>,
    columns: Vec<String>,
    all: bool,
    schema: Vec<LogicalType>,
) -> (Box<dyn Operator>, Vec<String>) {
    let operator = Box::new(IntersectOperator::new(left, right, all, schema));
    (operator, columns)
}

/// Builds an OTHERWISE physical operator.
pub(crate) fn build_otherwise(
    left: Box<dyn Operator>,
    right: Box<dyn Operator>,
    columns: Vec<String>,
) -> (Box<dyn Operator>, Vec<String>) {
    let operator = Box::new(OtherwiseOperator::new(left, right));
    (operator, columns)
}

/// Builds an INNER JOIN physical operator.
///
/// Finds shared variables between left and right column lists for join keys,
/// then creates a hash join with inner semantics. Deduplicates shared columns
/// by projecting away right-side columns that already appear on the left.
/// Falls back to cross join when no shared variables exist.
///
/// When `cardinalities` is provided as `(left_card, right_card)`, the smaller
/// side is placed as the build side for better memory and cache performance.
#[allow(dead_code)]
pub(crate) fn build_inner_join(
    left: Box<dyn Operator>,
    right: Box<dyn Operator>,
    left_columns: &[String],
    right_columns: &[String],
    schema_fn: impl Fn(&[String]) -> Vec<LogicalType>,
    cardinalities: Option<(f64, f64)>,
) -> (Box<dyn Operator>, Vec<String>) {
    let (probe_keys, build_keys) = find_shared_join_keys(left_columns, right_columns);

    let join_type = if probe_keys.is_empty() {
        PhysicalJoinType::Cross
    } else {
        PhysicalJoinType::Inner
    };

    // Decide whether to swap sides: build on the smaller input.
    // Only swap for equi-joins (not cross joins) and when left is significantly larger.
    let swap_sides = matches!(join_type, PhysicalJoinType::Inner)
        && cardinalities.is_some_and(|(left_card, right_card)| right_card < left_card * 0.8);

    if swap_sides {
        // Swap: right becomes probe, left becomes build
        // Output order is right+left from the join, then we project back to left+right
        let mut join_columns: Vec<String> = right_columns.to_vec();
        join_columns.extend(left_columns.iter().cloned());
        let join_schema = schema_fn(&join_columns);

        let join_op: Box<dyn Operator> = Box::new(HashJoinOperator::new(
            right,      // probe (larger)
            left,       // build (smaller, materialized)
            build_keys, // swapped: right keys become probe keys
            probe_keys, // swapped: left keys become build keys
            join_type,
            join_schema,
        ));

        // Remap to logical left+right order and deduplicate shared columns
        let right_count = right_columns.len();
        let left_set: std::collections::HashSet<&str> =
            left_columns.iter().map(String::as_str).collect();

        // Build projection: first map left columns (which are at offset right_count in physical output)
        let mut proj_indices: Vec<usize> =
            (0..left_columns.len()).map(|i| right_count + i).collect();
        let mut output_columns: Vec<String> = left_columns.to_vec();
        // Then add right columns not already in left
        for (right_idx, right_col) in right_columns.iter().enumerate() {
            if !left_set.contains(right_col.as_str()) {
                proj_indices.push(right_idx);
                output_columns.push(right_col.clone());
            }
        }

        let proj_exprs: Vec<ProjectExpr> = proj_indices
            .iter()
            .map(|&i| ProjectExpr::Column(i))
            .collect();
        let proj_types: Vec<LogicalType> = proj_indices.iter().map(|_| LogicalType::Any).collect();
        let operator = Box::new(ProjectOperator::new(join_op, proj_exprs, proj_types));
        (operator, output_columns)
    } else {
        // Normal order: left = probe, right = build
        let mut join_columns: Vec<String> = left_columns.to_vec();
        join_columns.extend(right_columns.iter().cloned());
        let join_schema = schema_fn(&join_columns);

        let join_op: Box<dyn Operator> = Box::new(HashJoinOperator::new(
            left,
            right,
            probe_keys,
            build_keys,
            join_type,
            join_schema,
        ));

        // Deduplicate: keep left columns, then only right columns not already on the left
        let left_set: std::collections::HashSet<&str> =
            left_columns.iter().map(String::as_str).collect();
        let mut keep_indices: Vec<usize> = (0..left_columns.len()).collect();
        let mut output_columns: Vec<String> = left_columns.to_vec();
        for (right_idx, right_col) in right_columns.iter().enumerate() {
            if !left_set.contains(right_col.as_str()) {
                keep_indices.push(left_columns.len() + right_idx);
                output_columns.push(right_col.clone());
            }
        }

        // If there are duplicates, add a ProjectOperator to strip them
        if keep_indices.len() < join_columns.len() {
            let proj_exprs: Vec<ProjectExpr> = keep_indices
                .iter()
                .map(|&i| ProjectExpr::Column(i))
                .collect();
            let proj_types: Vec<LogicalType> =
                keep_indices.iter().map(|_| LogicalType::Any).collect();
            let operator = Box::new(ProjectOperator::new(join_op, proj_exprs, proj_types));
            (operator, output_columns)
        } else {
            (join_op, output_columns)
        }
    }
}

/// Builds an ANTI JOIN physical operator.
///
/// Finds shared variables between left and right column lists for join keys,
/// then creates a hash join with anti semantics (only left rows with no match).
pub(crate) fn build_anti_join(
    left: Box<dyn Operator>,
    right: Box<dyn Operator>,
    left_columns: Vec<String>,
    right_columns: &[String],
    schema: Vec<LogicalType>,
) -> (Box<dyn Operator>, Vec<String>) {
    let (probe_keys, build_keys) = find_shared_join_keys(&left_columns, right_columns);

    let operator: Box<dyn Operator> = Box::new(HashJoinOperator::new(
        left,
        right,
        probe_keys,
        build_keys,
        PhysicalJoinType::Anti,
        schema,
    ));
    (operator, left_columns)
}

/// Builds a SEMI JOIN physical operator.
///
/// Finds shared variables between left and right column lists for join keys,
/// then creates a hash join with semi semantics (only left rows with a match).
#[allow(dead_code)]
pub(crate) fn build_semi_join(
    left: Box<dyn Operator>,
    right: Box<dyn Operator>,
    left_columns: Vec<String>,
    right_columns: &[String],
    schema: Vec<LogicalType>,
) -> (Box<dyn Operator>, Vec<String>) {
    let (probe_keys, build_keys) = find_shared_join_keys(&left_columns, right_columns);

    let operator: Box<dyn Operator> = Box::new(HashJoinOperator::new(
        left,
        right,
        probe_keys,
        build_keys,
        PhysicalJoinType::Semi,
        schema,
    ));
    (operator, left_columns)
}

/// Builds a LEFT JOIN physical operator.
///
/// Joins left and right sides, deduplicates shared columns by projecting away
/// right-side columns that already appear on the left.
pub(crate) fn build_left_join(
    left: Box<dyn Operator>,
    right: Box<dyn Operator>,
    left_columns: &[String],
    right_columns: &[String],
    schema_fn: impl Fn(&[String]) -> Vec<LogicalType>,
) -> (Box<dyn Operator>, Vec<String>) {
    let (probe_keys, build_keys) = find_shared_join_keys(left_columns, right_columns);

    // Full join outputs all left + all right columns
    let mut join_columns: Vec<String> = left_columns.to_vec();
    join_columns.extend(right_columns.iter().cloned());
    let join_schema = schema_fn(&join_columns);

    let join_op: Box<dyn Operator> = Box::new(HashJoinOperator::new(
        left,
        right,
        probe_keys,
        build_keys,
        PhysicalJoinType::Left,
        join_schema,
    ));

    // Deduplicate: keep left columns, then only right columns not already on the left
    let left_set: std::collections::HashSet<&str> =
        left_columns.iter().map(String::as_str).collect();
    let mut keep_indices: Vec<usize> = (0..left_columns.len()).collect();
    let mut output_columns: Vec<String> = left_columns.to_vec();
    for (right_idx, right_col) in right_columns.iter().enumerate() {
        if !left_set.contains(right_col.as_str()) {
            keep_indices.push(left_columns.len() + right_idx);
            output_columns.push(right_col.clone());
        }
    }

    // If there are duplicates, add a ProjectOperator to strip them
    if keep_indices.len() < join_columns.len() {
        let proj_exprs: Vec<ProjectExpr> = keep_indices
            .iter()
            .map(|&i| ProjectExpr::Column(i))
            .collect();
        let proj_types: Vec<LogicalType> = keep_indices.iter().map(|_| LogicalType::Any).collect();
        let operator = Box::new(ProjectOperator::new(join_op, proj_exprs, proj_types));
        (operator, output_columns)
    } else {
        (join_op, output_columns)
    }
}

/// Finds shared variable names between two column lists and returns
/// `(left_indices, right_indices)` for use as join keys.
fn find_shared_join_keys(left: &[String], right: &[String]) -> (Vec<usize>, Vec<usize>) {
    let mut probe_keys = Vec::new();
    let mut build_keys = Vec::new();
    for (right_idx, right_col) in right.iter().enumerate() {
        if let Some(left_idx) = left.iter().position(|c| c == right_col) {
            probe_keys.push(left_idx);
            build_keys.push(right_idx);
        }
    }
    (probe_keys, build_keys)
}

/// Converts a logical expression to a human-readable string for column naming.
pub(crate) fn expression_to_string(expr: &LogicalExpression) -> String {
    match expr {
        LogicalExpression::Variable(name) => name.clone(),
        LogicalExpression::Property { variable, property } => {
            format!("{variable}.{property}")
        }
        LogicalExpression::Literal(value) => format!("{value:?}"),
        LogicalExpression::FunctionCall { name, .. } => format!("{name}(...)"),
        _ => "expr".to_string(),
    }
}
