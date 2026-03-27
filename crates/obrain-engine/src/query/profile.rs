//! PROFILE statement: per-operator execution metrics.
//!
//! After a profiled query executes, the results are collected into a
//! [`ProfileNode`] tree that mirrors the physical operator tree, annotated
//! with actual row counts, timing, and call counts.

use std::fmt::Write;
use std::sync::Arc;

use grafeo_common::types::{LogicalType, Value};
use grafeo_core::execution::profile::{ProfileStats, SharedProfileStats};
use parking_lot::Mutex;

use super::plan::LogicalOperator;
use crate::database::QueryResult;

/// A node in the profile output tree, corresponding to one physical operator.
#[derive(Debug)]
pub struct ProfileNode {
    /// Operator name (e.g., "NodeScan", "Filter", "Expand").
    pub name: String,
    /// Display label (e.g., "(n:Person)", "(n.age > 25) [label-first]").
    pub label: String,
    /// Shared stats handle, populated during execution.
    pub stats: SharedProfileStats,
    /// Child nodes.
    pub children: Vec<ProfileNode>,
}

/// An entry collected during physical planning, used to build the profile tree.
pub struct ProfileEntry {
    /// Operator name from `Operator::name()`.
    pub name: String,
    /// Human-readable label from the logical operator.
    pub label: String,
    /// Shared stats handle passed to the `ProfiledOperator` wrapper.
    pub stats: SharedProfileStats,
}

impl ProfileEntry {
    /// Creates a new profile entry with fresh (empty) stats.
    pub fn new(name: &str, label: String) -> (Self, SharedProfileStats) {
        let stats = Arc::new(Mutex::new(ProfileStats::default()));
        let entry = Self {
            name: name.to_string(),
            label,
            stats: Arc::clone(&stats),
        };
        (entry, stats)
    }
}

/// Builds a `ProfileNode` tree from the logical plan and a list of
/// [`ProfileEntry`] items collected during physical planning.
///
/// The entries must be in **post-order** (children before parents),
/// matching the order in which `plan_operator()` processes operators.
pub fn build_profile_tree(
    logical: &LogicalOperator,
    entries: &mut impl Iterator<Item = ProfileEntry>,
) -> ProfileNode {
    // Recurse into children first (post-order)
    let children: Vec<ProfileNode> = logical
        .children()
        .into_iter()
        .map(|child| build_profile_tree(child, entries))
        .collect();

    // Consume the entry for this node
    let entry = entries
        .next()
        .expect("profile entry count must match logical operator count");

    ProfileNode {
        name: entry.name,
        label: entry.label,
        stats: entry.stats,
        children,
    }
}

/// Formats a `ProfileNode` tree into a human-readable text representation
/// and wraps it in a `QueryResult` with a single "profile" column.
pub fn profile_result(root: &ProfileNode, total_time_ms: f64) -> QueryResult {
    let mut output = String::new();
    format_node(&mut output, root, 0);
    let _ = writeln!(output);
    let _ = write!(output, "Total time: {total_time_ms:.2}ms");

    QueryResult {
        columns: vec!["profile".to_string()],
        column_types: vec![LogicalType::String],
        rows: vec![vec![Value::String(output.into())]],
        execution_time_ms: Some(total_time_ms),
        rows_scanned: None,
        status_message: None,
        gql_status: grafeo_common::utils::GqlStatus::SUCCESS,
    }
}

/// Recursively formats a profile node with indentation.
fn format_node(out: &mut String, node: &ProfileNode, depth: usize) {
    let indent = "  ".repeat(depth);

    // Compute self-time before locking stats (self_time_ns also locks).
    let self_time_ns = self_time_ns(node);
    let self_time_ms = self_time_ns as f64 / 1_000_000.0;

    let rows_out = node.stats.lock().rows_out;

    let _ = writeln!(
        out,
        "{indent}{name} ({label})  rows={rows}  time={time:.2}ms",
        name = node.name,
        label = node.label,
        rows = rows_out,
        time = self_time_ms,
    );

    for child in &node.children {
        format_node(out, child, depth + 1);
    }
}

/// Computes self-time: wall time minus children's wall time.
fn self_time_ns(node: &ProfileNode) -> u64 {
    let own_time = node.stats.lock().time_ns;
    let child_time: u64 = node.children.iter().map(|c| c.stats.lock().time_ns).sum();
    own_time.saturating_sub(child_time)
}
