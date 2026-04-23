//! Shortest path operator for finding paths between nodes.
//!
//! This operator computes shortest paths between source and target nodes
//! using BFS for unweighted graphs.

use super::{Operator, OperatorResult};
use crate::execution::chunk::DataChunkBuilder;
use crate::graph::Direction;
use crate::graph::GraphStore;
use obrain_common::types::{LogicalType, NodeId, Value};
use obrain_common::utils::hash::FxHashMap;
use std::collections::VecDeque;
use std::sync::Arc;

/// Operator that finds shortest paths between source and target nodes.
///
/// For each input row containing source and target nodes, this operator
/// computes the shortest path and outputs the path as a value.
pub struct ShortestPathOperator {
    /// The graph store.
    store: Arc<dyn GraphStore>,
    /// Input operator providing source/target node pairs.
    input: Box<dyn Operator>,
    /// Column index of the source node.
    source_column: usize,
    /// Column index of the target node.
    target_column: usize,
    /// Edge type filter (empty means all types).
    edge_types: Vec<String>,
    /// Direction of edge traversal.
    direction: Direction,
    /// Whether to find all shortest paths (vs. just one).
    all_paths: bool,
    /// Whether the operator has been exhausted.
    exhausted: bool,
}

impl ShortestPathOperator {
    /// Creates a new shortest path operator.
    pub fn new(
        store: Arc<dyn GraphStore>,
        input: Box<dyn Operator>,
        source_column: usize,
        target_column: usize,
        edge_types: Vec<String>,
        direction: Direction,
    ) -> Self {
        Self {
            store,
            input,
            source_column,
            target_column,
            edge_types,
            direction,
            all_paths: false,
            exhausted: false,
        }
    }

    /// Sets whether to find all shortest paths.
    pub fn with_all_paths(mut self, all_paths: bool) -> Self {
        self.all_paths = all_paths;
        self
    }

    /// Finds the shortest path between source and target using BFS.
    /// Returns the path length (number of edges).
    fn find_shortest_path(&self, source: NodeId, target: NodeId) -> Option<i64> {
        if source == target {
            return Some(0);
        }

        let mut visited: FxHashMap<NodeId, i64> = FxHashMap::default();
        let mut queue: VecDeque<(NodeId, i64)> = VecDeque::new();

        visited.insert(source, 0);
        queue.push_back((source, 0));

        while let Some((current, depth)) = queue.pop_front() {
            // Get neighbors based on direction
            let neighbors = self.get_neighbors(current);

            for neighbor in neighbors {
                if neighbor == target {
                    return Some(depth + 1);
                }

                if !visited.contains_key(&neighbor) {
                    visited.insert(neighbor, depth + 1);
                    queue.push_back((neighbor, depth + 1));
                }
            }
        }

        None // No path found
    }

    /// Finds all shortest paths between source and target using BFS.
    /// Returns a vector of path lengths (all will be the same minimum length).
    /// For allShortestPaths, we return the count of paths with minimum length.
    fn find_all_shortest_paths(&self, source: NodeId, target: NodeId) -> Vec<i64> {
        if source == target {
            return vec![0];
        }

        // BFS that tracks number of paths to each node at each depth
        let mut distances: FxHashMap<NodeId, i64> = FxHashMap::default();
        let mut path_counts: FxHashMap<NodeId, usize> = FxHashMap::default();
        let mut queue: VecDeque<NodeId> = VecDeque::new();

        distances.insert(source, 0);
        path_counts.insert(source, 1);
        queue.push_back(source);

        let mut target_depth: Option<i64> = None;
        let mut target_path_count = 0;

        while let Some(current) = queue.pop_front() {
            let current_depth = *distances
                .get(&current)
                .expect("BFS: node dequeued has distance");
            let current_paths = *path_counts
                .get(&current)
                .expect("BFS: node dequeued has path count");

            // If we've found target and we're past its depth, stop
            if let Some(td) = target_depth
                && current_depth >= td
            {
                continue;
            }

            for neighbor in self.get_neighbors(current) {
                let new_depth = current_depth + 1;

                if neighbor == target {
                    // Found target
                    if target_depth.is_none() {
                        target_depth = Some(new_depth);
                        target_path_count = current_paths;
                    } else if Some(new_depth) == target_depth {
                        target_path_count += current_paths;
                    }
                    continue;
                }

                // If not visited or same depth (for counting all paths)
                if let Some(&existing_depth) = distances.get(&neighbor) {
                    if existing_depth == new_depth {
                        // Same depth, add to path count
                        *path_counts
                            .get_mut(&neighbor)
                            .expect("BFS: neighbor has path count at same depth") += current_paths;
                    }
                    // If existing_depth < new_depth, skip (already processed at shorter distance)
                } else {
                    // New node
                    distances.insert(neighbor, new_depth);
                    path_counts.insert(neighbor, current_paths);
                    queue.push_back(neighbor);
                }
            }
        }

        // Return one entry per path
        if let Some(depth) = target_depth {
            vec![depth; target_path_count]
        } else {
            vec![]
        }
    }

    /// Gets neighbors of a node in a specific direction, respecting edge type filter.
    ///
    /// This is the direction-parameterized variant used by bidirectional BFS
    /// to traverse the forward and backward frontiers independently.
    fn get_neighbors_directed(&self, node: NodeId, direction: Direction) -> Vec<NodeId> {
        self.store
            .edges_from(node, direction)
            .into_iter()
            .filter(|(_target, edge_id)| {
                if self.edge_types.is_empty() {
                    true
                } else if let Some(actual_type) = self.store.edge_type(*edge_id) {
                    self.edge_types
                        .iter()
                        .any(|t| actual_type.as_str().eq_ignore_ascii_case(t.as_str()))
                } else {
                    false
                }
            })
            .map(|(target, _)| target)
            .collect()
    }

    /// Gets neighbors of a node respecting edge type filter and direction.
    fn get_neighbors(&self, node: NodeId) -> Vec<NodeId> {
        self.get_neighbors_directed(node, self.direction)
    }

    /// Finds shortest path using bidirectional BFS.
    ///
    /// Maintains forward and backward frontiers, alternating expansion of the
    /// smaller one. When a node is found in both visited sets, the shortest
    /// path is `forward_depth + backward_depth`. This reduces the search space
    /// from O(b^d) to O(b^(d/2)) where b is the branching factor and d is
    /// the path length.
    ///
    /// Falls back to unidirectional BFS if backward adjacency is unavailable.
    fn find_shortest_path_bidirectional(&self, source: NodeId, target: NodeId) -> Option<i64> {
        if source == target {
            return Some(0);
        }

        // Fall back to unidirectional if backward adjacency is unavailable
        if !self.store.has_backward_adjacency() {
            return self.find_shortest_path(source, target);
        }

        let reverse_dir = self.direction.reverse();

        // Forward BFS state
        let mut forward_visited: FxHashMap<NodeId, i64> = FxHashMap::default();
        let mut forward_queue: VecDeque<(NodeId, i64)> = VecDeque::new();
        forward_visited.insert(source, 0);
        forward_queue.push_back((source, 0));

        // Backward BFS state
        let mut backward_visited: FxHashMap<NodeId, i64> = FxHashMap::default();
        let mut backward_queue: VecDeque<(NodeId, i64)> = VecDeque::new();
        backward_visited.insert(target, 0);
        backward_queue.push_back((target, 0));

        // Best known path length (upper bound)
        let mut best: Option<i64> = None;

        loop {
            // Decide which frontier to expand, or stop
            let expand_forward = match (forward_queue.front(), backward_queue.front()) {
                (Some(_), Some(_)) => forward_queue.len() <= backward_queue.len(),
                (Some(_), None) => true,
                (None, Some(_)) => false,
                (None, None) => break,
            };

            if expand_forward {
                let Some((current, depth)) = forward_queue.pop_front() else {
                    break;
                };

                // If this depth alone exceeds best, this frontier is exhausted
                if let Some(b) = best
                    && depth + 1 > b
                {
                    // Clear the queue; no further expansion can improve
                    forward_queue.clear();
                    continue;
                }

                for neighbor in self.get_neighbors_directed(current, self.direction) {
                    let new_depth = depth + 1;

                    // Check if backward frontier already visited this node
                    if let Some(&backward_depth) = backward_visited.get(&neighbor) {
                        let total = new_depth + backward_depth;
                        best = Some(best.map_or(total, |b: i64| b.min(total)));
                    }

                    if !forward_visited.contains_key(&neighbor) {
                        forward_visited.insert(neighbor, new_depth);
                        if best.is_none_or(|b| new_depth < b) {
                            forward_queue.push_back((neighbor, new_depth));
                        }
                    }
                }
            } else {
                let Some((current, depth)) = backward_queue.pop_front() else {
                    break;
                };

                if let Some(b) = best
                    && depth + 1 > b
                {
                    backward_queue.clear();
                    continue;
                }

                for neighbor in self.get_neighbors_directed(current, reverse_dir) {
                    let new_depth = depth + 1;

                    // Check if forward frontier already visited this node
                    if let Some(&forward_depth) = forward_visited.get(&neighbor) {
                        let total = forward_depth + new_depth;
                        best = Some(best.map_or(total, |b: i64| b.min(total)));
                    }

                    if !backward_visited.contains_key(&neighbor) {
                        backward_visited.insert(neighbor, new_depth);
                        if best.is_none_or(|b| new_depth < b) {
                            backward_queue.push_back((neighbor, new_depth));
                        }
                    }
                }
            }
        }

        best
    }
}

impl Operator for ShortestPathOperator {
    fn next(&mut self) -> OperatorResult {
        if self.exhausted {
            return Ok(None);
        }

        // Get input chunk
        let Some(input_chunk) = self.input.next()? else {
            self.exhausted = true;
            return Ok(None);
        };

        // Build output: input columns + path length
        let num_input_cols = input_chunk.column_count();
        let mut output_schema: Vec<LogicalType> = (0..num_input_cols)
            .map(|i| {
                input_chunk
                    .column(i)
                    .map_or(LogicalType::Any, |c| c.data_type().clone())
            })
            .collect();
        output_schema.push(LogicalType::Any); // Path column (stores length as int)

        // For allShortestPaths, we may need more rows than input
        let initial_capacity = if self.all_paths {
            input_chunk.row_count() * 4 // Estimate 4x for multiple paths
        } else {
            input_chunk.row_count()
        };
        let mut builder = DataChunkBuilder::with_capacity(&output_schema, initial_capacity);

        for row in input_chunk.selected_indices() {
            // Get source and target nodes
            let source = input_chunk
                .column(self.source_column)
                .and_then(|c| c.get_node_id(row));
            let target = input_chunk
                .column(self.target_column)
                .and_then(|c| c.get_node_id(row));

            // Compute shortest path(s)
            let path_lengths: Vec<Option<i64>> = match (source, target) {
                (Some(s), Some(t)) => {
                    if self.all_paths {
                        let paths = self.find_all_shortest_paths(s, t);
                        if paths.is_empty() {
                            vec![None] // No path found, still output one row with null
                        } else {
                            paths.into_iter().map(Some).collect()
                        }
                    } else {
                        // Use bidirectional BFS when possible (single shortest path)
                        vec![self.find_shortest_path_bidirectional(s, t)]
                    }
                }
                _ => vec![None],
            };

            // Output one row per path
            for path_length in path_lengths {
                // Copy input columns
                for col_idx in 0..num_input_cols {
                    if let Some(in_col) = input_chunk.column(col_idx)
                        && let Some(out_col) = builder.column_mut(col_idx)
                    {
                        if let Some(node_id) = in_col.get_node_id(row) {
                            out_col.push_node_id(node_id);
                        } else if let Some(edge_id) = in_col.get_edge_id(row) {
                            out_col.push_edge_id(edge_id);
                        } else if let Some(value) = in_col.get_value(row) {
                            out_col.push_value(value);
                        } else {
                            out_col.push_value(Value::Null);
                        }
                    }
                }

                // Add path length column
                if let Some(out_col) = builder.column_mut(num_input_cols) {
                    match path_length {
                        Some(len) => out_col.push_value(Value::Int64(len)),
                        None => out_col.push_value(Value::Null),
                    }
                }

                builder.advance_row();
            }
        }

        let chunk = builder.finish();
        if chunk.row_count() > 0 {
            Ok(Some(chunk))
        } else {
            self.exhausted = true;
            Ok(None)
        }
    }

    fn reset(&mut self) {
        self.input.reset();
        self.exhausted = false;
    }

    fn name(&self) -> &'static str {
        "ShortestPath"
    }
}

// NOTE (T17 W4.p4 closure, 2026-04-23): the in-crate `#[cfg(test)] mod tests`
// block that used to live here has been relocated to
// `crates/obrain-substrate/tests/operators_shortest_path.rs`. Rationale:
// post-T17 cutover, the canonical test fixture is `SubstrateStore::open_tempfile()`
// (the documented replacement for `LpgStore::new()`). A substrate-backed fixture
// cannot live inside `obrain-core` because `obrain-core` cannot take
// `obrain-substrate` as a dev-dependency — that produces a dev-dep cycle
// (Cargo emits two distinct compilation units of `obrain-core`, breaking
// `Arc<SubstrateStore> as Arc<dyn GraphStore>` trait resolution). Empirically
// recorded in gotcha `598dda40-a186-4be3-97f3-c75053af4e6e`; decision
// `0a84cf59-b7d7-4a60-9d62-aa00eec820a3` documents the class-2 structural path.
// The integration-test home in `obrain-substrate/tests/` is the same pattern
// used by `graph_store_parity.rs` and works because the forward direction
// (`obrain-substrate → obrain-core`) has no cycle.
