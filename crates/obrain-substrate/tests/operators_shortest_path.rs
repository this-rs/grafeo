//! Integration tests for `obrain_core::execution::operators::ShortestPathOperator`
//! against the substrate backend.
//!
//! ## Why this lives in obrain-substrate and not in obrain-core
//!
//! `obrain-core` (the crate hosting `ShortestPathOperator`) cannot take
//! `obrain-substrate` as a dev-dependency without introducing a dev-dep
//! cycle (`obrain-substrate → obrain-core` forward + `obrain-core[dev] →
//! obrain-substrate` back). Cargo's cycle handling produces two distinct
//! compilation units of `obrain-core`, so `Arc<SubstrateStore> as
//! Arc<dyn GraphStore>` fails trait-resolution even when `cargo check`
//! reports no errors on the lib target alone. Documented in gotcha
//! `598dda40-a186-4be3-97f3-c75053af4e6e` and decision
//! `0a84cf59-b7d7-4a60-9d62-aa00eec820a3`.
//!
//! The forward direction (`obrain-substrate` depending on `obrain-core`)
//! has no cycle, so moving the operator tests into an integration test
//! of `obrain-substrate` is the post-T17-cutover home for the LPG-era
//! fixtures. This is the same pattern already used by
//! `graph_store_parity.rs` in this crate.
//!
//! Run with:
//!
//! ```bash
//! cargo test -p obrain-substrate --test operators_shortest_path
//! ```

use obrain_common::types::{LogicalType, NodeId, Value};
use obrain_core::execution::chunk::DataChunkBuilder;
use obrain_core::execution::operators::{Operator, OperatorResult, ShortestPathOperator};
use obrain_core::graph::traits::GraphStoreMut;
use obrain_core::graph::{Direction, GraphStore};
use obrain_substrate::SubstrateStore;
use std::sync::Arc;

/// A mock operator that returns a single chunk with source/target node pairs.
struct MockPairOperator {
    pairs: Vec<(NodeId, NodeId)>,
    exhausted: bool,
}

impl MockPairOperator {
    fn new(pairs: Vec<(NodeId, NodeId)>) -> Self {
        Self {
            pairs,
            exhausted: false,
        }
    }
}

impl Operator for MockPairOperator {
    fn next(&mut self) -> OperatorResult {
        if self.exhausted || self.pairs.is_empty() {
            return Ok(None);
        }
        self.exhausted = true;

        let schema = vec![LogicalType::Node, LogicalType::Node];
        let mut builder = DataChunkBuilder::with_capacity(&schema, self.pairs.len());

        for (source, target) in &self.pairs {
            builder.column_mut(0).unwrap().push_node_id(*source);
            builder.column_mut(1).unwrap().push_node_id(*target);
            builder.advance_row();
        }

        Ok(Some(builder.finish()))
    }

    fn reset(&mut self) {
        self.exhausted = false;
    }

    fn name(&self) -> &'static str {
        "MockPair"
    }
}

/// Canonical post-T17 fixture: a fresh substrate-backed store rooted in a
/// temp directory owned by the store itself. Drops auto-clean the dir.
fn fresh_store() -> Arc<SubstrateStore> {
    Arc::new(SubstrateStore::open_tempfile().unwrap())
}

#[test]
fn test_find_shortest_path_direct() {
    let store = fresh_store();

    // a -> b (1 hop)
    let a = store.create_node(&["Node"]);
    let b = store.create_node(&["Node"]);
    store.create_edge(a, b, "KNOWS");

    let input = Box::new(MockPairOperator::new(vec![(a, b)]));
    let mut op = ShortestPathOperator::new(
        store.clone() as Arc<dyn GraphStore>,
        input,
        0, // source column
        1, // target column
        vec![],
        Direction::Outgoing,
    );

    let chunk = op.next().unwrap().unwrap();
    assert_eq!(chunk.row_count(), 1);

    // Path length should be 1
    let path_col = chunk.column(2).unwrap();
    let path_len = path_col.get_value(0).unwrap();
    assert_eq!(path_len, Value::Int64(1));
}

#[test]
fn test_find_shortest_path_same_node() {
    let store = fresh_store();
    let a = store.create_node(&["Node"]);

    let input = Box::new(MockPairOperator::new(vec![(a, a)]));
    let mut op = ShortestPathOperator::new(
        store.clone() as Arc<dyn GraphStore>,
        input,
        0,
        1,
        vec![],
        Direction::Outgoing,
    );

    let chunk = op.next().unwrap().unwrap();
    assert_eq!(chunk.row_count(), 1);

    // Path length should be 0 (same node)
    let path_col = chunk.column(2).unwrap();
    let path_len = path_col.get_value(0).unwrap();
    assert_eq!(path_len, Value::Int64(0));
}

#[test]
fn test_find_shortest_path_two_hops() {
    let store = fresh_store();

    // a -> b -> c (2 hops from a to c)
    let a = store.create_node(&["Node"]);
    let b = store.create_node(&["Node"]);
    let c = store.create_node(&["Node"]);
    store.create_edge(a, b, "KNOWS");
    store.create_edge(b, c, "KNOWS");

    let input = Box::new(MockPairOperator::new(vec![(a, c)]));
    let mut op = ShortestPathOperator::new(
        store.clone() as Arc<dyn GraphStore>,
        input,
        0,
        1,
        vec![],
        Direction::Outgoing,
    );

    let chunk = op.next().unwrap().unwrap();
    assert_eq!(chunk.row_count(), 1);

    let path_col = chunk.column(2).unwrap();
    let path_len = path_col.get_value(0).unwrap();
    assert_eq!(path_len, Value::Int64(2));
}

#[test]
fn test_find_shortest_path_no_path() {
    let store = fresh_store();

    // a and b are disconnected
    let a = store.create_node(&["Node"]);
    let b = store.create_node(&["Node"]);

    let input = Box::new(MockPairOperator::new(vec![(a, b)]));
    let mut op = ShortestPathOperator::new(
        store.clone() as Arc<dyn GraphStore>,
        input,
        0,
        1,
        vec![],
        Direction::Outgoing,
    );

    let chunk = op.next().unwrap().unwrap();
    assert_eq!(chunk.row_count(), 1);

    // Path length should be null (no path)
    let path_col = chunk.column(2).unwrap();
    let path_len = path_col.get_value(0).unwrap();
    assert_eq!(path_len, Value::Null);
}

#[test]
fn test_find_shortest_path_prefers_shorter() {
    let store = fresh_store();

    // Create two paths: a -> d (1 hop) and a -> b -> c -> d (3 hops)
    let a = store.create_node(&["Node"]);
    let b = store.create_node(&["Node"]);
    let c = store.create_node(&["Node"]);
    let d = store.create_node(&["Node"]);

    // Long path
    store.create_edge(a, b, "KNOWS");
    store.create_edge(b, c, "KNOWS");
    store.create_edge(c, d, "KNOWS");

    // Short path (direct)
    store.create_edge(a, d, "KNOWS");

    let input = Box::new(MockPairOperator::new(vec![(a, d)]));
    let mut op = ShortestPathOperator::new(
        store.clone() as Arc<dyn GraphStore>,
        input,
        0,
        1,
        vec![],
        Direction::Outgoing,
    );

    let chunk = op.next().unwrap().unwrap();
    let path_col = chunk.column(2).unwrap();
    let path_len = path_col.get_value(0).unwrap();
    assert_eq!(path_len, Value::Int64(1)); // Should find direct path
}

#[test]
fn test_find_shortest_path_with_edge_type_filter() {
    let store = fresh_store();

    // a -KNOWS-> b -LIKES-> c
    let a = store.create_node(&["Node"]);
    let b = store.create_node(&["Node"]);
    let c = store.create_node(&["Node"]);
    store.create_edge(a, b, "KNOWS");
    store.create_edge(b, c, "LIKES");

    // Path with KNOWS filter should only reach b, not c
    let input = Box::new(MockPairOperator::new(vec![(a, c)]));
    let mut op = ShortestPathOperator::new(
        store.clone() as Arc<dyn GraphStore>,
        input,
        0,
        1,
        vec!["KNOWS".to_string()],
        Direction::Outgoing,
    );

    let chunk = op.next().unwrap().unwrap();
    let path_col = chunk.column(2).unwrap();
    let path_len = path_col.get_value(0).unwrap();
    assert_eq!(path_len, Value::Null); // Can't reach c via KNOWS only
}

#[test]
fn test_all_shortest_paths_single_path() {
    let store = fresh_store();

    // a -> b (single path)
    let a = store.create_node(&["Node"]);
    let b = store.create_node(&["Node"]);
    store.create_edge(a, b, "KNOWS");

    let input = Box::new(MockPairOperator::new(vec![(a, b)]));
    let mut op = ShortestPathOperator::new(
        store.clone() as Arc<dyn GraphStore>,
        input,
        0,
        1,
        vec![],
        Direction::Outgoing,
    )
    .with_all_paths(true);

    let chunk = op.next().unwrap().unwrap();
    assert_eq!(chunk.row_count(), 1); // Only one path exists
}

#[test]
fn test_all_shortest_paths_multiple_paths() {
    let store = fresh_store();

    // Create diamond: a -> b -> d and a -> c -> d (two paths of length 2)
    let a = store.create_node(&["Node"]);
    let b = store.create_node(&["Node"]);
    let c = store.create_node(&["Node"]);
    let d = store.create_node(&["Node"]);

    store.create_edge(a, b, "KNOWS");
    store.create_edge(a, c, "KNOWS");
    store.create_edge(b, d, "KNOWS");
    store.create_edge(c, d, "KNOWS");

    let input = Box::new(MockPairOperator::new(vec![(a, d)]));
    let mut op = ShortestPathOperator::new(
        store.clone() as Arc<dyn GraphStore>,
        input,
        0,
        1,
        vec![],
        Direction::Outgoing,
    )
    .with_all_paths(true);

    let chunk = op.next().unwrap().unwrap();
    // Should return 2 rows (two paths of length 2)
    assert_eq!(chunk.row_count(), 2);

    // Both should have length 2
    let path_col = chunk.column(2).unwrap();
    assert_eq!(path_col.get_value(0).unwrap(), Value::Int64(2));
    assert_eq!(path_col.get_value(1).unwrap(), Value::Int64(2));
}

#[test]
fn test_multiple_pairs_in_chunk() {
    let store = fresh_store();

    // Create: a -> b, c -> d
    let a = store.create_node(&["Node"]);
    let b = store.create_node(&["Node"]);
    let c = store.create_node(&["Node"]);
    let d = store.create_node(&["Node"]);

    store.create_edge(a, b, "KNOWS");
    store.create_edge(c, d, "KNOWS");

    // Test multiple pairs at once
    let input = Box::new(MockPairOperator::new(vec![(a, b), (c, d), (a, d)]));
    let mut op = ShortestPathOperator::new(
        store.clone() as Arc<dyn GraphStore>,
        input,
        0,
        1,
        vec![],
        Direction::Outgoing,
    );

    let chunk = op.next().unwrap().unwrap();
    assert_eq!(chunk.row_count(), 3);

    let path_col = chunk.column(2).unwrap();
    assert_eq!(path_col.get_value(0).unwrap(), Value::Int64(1)); // a->b = 1
    assert_eq!(path_col.get_value(1).unwrap(), Value::Int64(1)); // c->d = 1
    assert_eq!(path_col.get_value(2).unwrap(), Value::Null); // a->d = no path
}

#[test]
fn test_operator_reset() {
    let store = fresh_store();
    let a = store.create_node(&["Node"]);
    let b = store.create_node(&["Node"]);
    store.create_edge(a, b, "KNOWS");

    let input = Box::new(MockPairOperator::new(vec![(a, b)]));
    let mut op = ShortestPathOperator::new(
        store.clone() as Arc<dyn GraphStore>,
        input,
        0,
        1,
        vec![],
        Direction::Outgoing,
    );

    // First iteration
    let chunk = op.next().unwrap();
    assert!(chunk.is_some());
    let chunk = op.next().unwrap();
    assert!(chunk.is_none());

    // After reset
    op.reset();
    let chunk = op.next().unwrap();
    assert!(chunk.is_some());
}

#[test]
fn test_operator_name() {
    let store = fresh_store();
    let input = Box::new(MockPairOperator::new(vec![]));
    let op = ShortestPathOperator::new(
        store.clone() as Arc<dyn GraphStore>,
        input,
        0,
        1,
        vec![],
        Direction::Outgoing,
    );

    assert_eq!(op.name(), "ShortestPath");
}

#[test]
fn test_empty_input() {
    let store = fresh_store();
    let input = Box::new(MockPairOperator::new(vec![]));
    let mut op = ShortestPathOperator::new(
        store.clone() as Arc<dyn GraphStore>,
        input,
        0,
        1,
        vec![],
        Direction::Outgoing,
    );

    // Empty input should return None
    let chunk = op.next().unwrap();
    assert!(chunk.is_none());
}

#[test]
fn test_all_shortest_paths_no_path() {
    let store = fresh_store();

    // Disconnected nodes
    let a = store.create_node(&["Node"]);
    let b = store.create_node(&["Node"]);

    let input = Box::new(MockPairOperator::new(vec![(a, b)]));
    let mut op = ShortestPathOperator::new(
        store.clone() as Arc<dyn GraphStore>,
        input,
        0,
        1,
        vec![],
        Direction::Outgoing,
    )
    .with_all_paths(true);

    let chunk = op.next().unwrap().unwrap();
    assert_eq!(chunk.row_count(), 1); // Still returns one row with null

    let path_col = chunk.column(2).unwrap();
    assert_eq!(path_col.get_value(0).unwrap(), Value::Null);
}

#[test]
fn test_all_shortest_paths_same_node() {
    let store = fresh_store();
    let a = store.create_node(&["Node"]);

    let input = Box::new(MockPairOperator::new(vec![(a, a)]));
    let mut op = ShortestPathOperator::new(
        store.clone() as Arc<dyn GraphStore>,
        input,
        0,
        1,
        vec![],
        Direction::Outgoing,
    )
    .with_all_paths(true);

    let chunk = op.next().unwrap().unwrap();
    assert_eq!(chunk.row_count(), 1);

    let path_col = chunk.column(2).unwrap();
    assert_eq!(path_col.get_value(0).unwrap(), Value::Int64(0));
}

// === Bidirectional BFS Tests ===

#[test]
fn test_bidirectional_bfs_long_chain() {
    let store = fresh_store();

    // Chain: n0 -> n1 -> n2 -> ... -> n9 (9 hops)
    let nodes: Vec<NodeId> = (0..10).map(|_| store.create_node(&["Node"])).collect();
    for i in 0..9 {
        store.create_edge(nodes[i], nodes[i + 1], "NEXT");
    }

    let input = Box::new(MockPairOperator::new(vec![(nodes[0], nodes[9])]));
    let mut op = ShortestPathOperator::new(
        store.clone() as Arc<dyn GraphStore>,
        input,
        0,
        1,
        vec![],
        Direction::Outgoing,
    );

    let chunk = op.next().unwrap().unwrap();
    let path_col = chunk.column(2).unwrap();
    assert_eq!(path_col.get_value(0).unwrap(), Value::Int64(9));
}

#[test]
fn test_bidirectional_bfs_diamond() {
    let store = fresh_store();

    // Diamond: a -> b -> d, a -> c -> d (two paths of length 2)
    let a = store.create_node(&["Node"]);
    let b = store.create_node(&["Node"]);
    let c = store.create_node(&["Node"]);
    let d = store.create_node(&["Node"]);

    store.create_edge(a, b, "KNOWS");
    store.create_edge(a, c, "KNOWS");
    store.create_edge(b, d, "KNOWS");
    store.create_edge(c, d, "KNOWS");

    let input = Box::new(MockPairOperator::new(vec![(a, d)]));
    let mut op = ShortestPathOperator::new(
        store.clone() as Arc<dyn GraphStore>,
        input,
        0,
        1,
        vec![],
        Direction::Outgoing,
    );

    let chunk = op.next().unwrap().unwrap();
    let path_col = chunk.column(2).unwrap();
    assert_eq!(path_col.get_value(0).unwrap(), Value::Int64(2));
}

#[test]
fn test_bidirectional_bfs_no_path() {
    let store = fresh_store();

    let a = store.create_node(&["Node"]);
    let b = store.create_node(&["Node"]);
    // No edges between a and b

    let input = Box::new(MockPairOperator::new(vec![(a, b)]));
    let mut op = ShortestPathOperator::new(
        store.clone() as Arc<dyn GraphStore>,
        input,
        0,
        1,
        vec![],
        Direction::Outgoing,
    );

    let chunk = op.next().unwrap().unwrap();
    let path_col = chunk.column(2).unwrap();
    assert_eq!(path_col.get_value(0).unwrap(), Value::Null);
}

#[test]
fn test_bidirectional_bfs_same_node() {
    let store = fresh_store();
    let a = store.create_node(&["Node"]);

    let input = Box::new(MockPairOperator::new(vec![(a, a)]));
    let mut op = ShortestPathOperator::new(
        store.clone() as Arc<dyn GraphStore>,
        input,
        0,
        1,
        vec![],
        Direction::Outgoing,
    );

    let chunk = op.next().unwrap().unwrap();
    let path_col = chunk.column(2).unwrap();
    assert_eq!(path_col.get_value(0).unwrap(), Value::Int64(0));
}

#[test]
fn test_bidirectional_bfs_prefers_shorter() {
    let store = fresh_store();

    let a = store.create_node(&["Node"]);
    let b = store.create_node(&["Node"]);
    let c = store.create_node(&["Node"]);
    let d = store.create_node(&["Node"]);

    // Long path: a -> b -> c -> d (3 hops)
    store.create_edge(a, b, "KNOWS");
    store.create_edge(b, c, "KNOWS");
    store.create_edge(c, d, "KNOWS");
    // Short path: a -> d (1 hop)
    store.create_edge(a, d, "KNOWS");

    let input = Box::new(MockPairOperator::new(vec![(a, d)]));
    let mut op = ShortestPathOperator::new(
        store.clone() as Arc<dyn GraphStore>,
        input,
        0,
        1,
        vec![],
        Direction::Outgoing,
    );

    let chunk = op.next().unwrap().unwrap();
    let path_col = chunk.column(2).unwrap();
    assert_eq!(path_col.get_value(0).unwrap(), Value::Int64(1));
}

#[test]
fn test_bidirectional_bfs_with_edge_type_filter() {
    let store = fresh_store();

    // a -KNOWS-> b -LIKES-> c
    let a = store.create_node(&["Node"]);
    let b = store.create_node(&["Node"]);
    let c = store.create_node(&["Node"]);
    store.create_edge(a, b, "KNOWS");
    store.create_edge(b, c, "LIKES");

    // Only KNOWS edges: a can reach b but not c
    let input = Box::new(MockPairOperator::new(vec![(a, c)]));
    let mut op = ShortestPathOperator::new(
        store.clone() as Arc<dyn GraphStore>,
        input,
        0,
        1,
        vec!["KNOWS".to_string()],
        Direction::Outgoing,
    );

    let chunk = op.next().unwrap().unwrap();
    let path_col = chunk.column(2).unwrap();
    assert_eq!(path_col.get_value(0).unwrap(), Value::Null);
}

#[test]
fn test_bidirectional_bfs_has_backward_adjacency() {
    // Default substrate store has backward adjacency enabled, matching the
    // LpgStore contract the bidirectional BFS path expected.
    let store = fresh_store();
    assert!(store.has_backward_adjacency());

    let a = store.create_node(&["Node"]);
    let b = store.create_node(&["Node"]);
    let c = store.create_node(&["Node"]);
    store.create_edge(a, b, "KNOWS");
    store.create_edge(b, c, "KNOWS");

    // Bidirectional BFS should work and find shortest path
    let input = Box::new(MockPairOperator::new(vec![(a, c)]));
    let mut op = ShortestPathOperator::new(
        store.clone() as Arc<dyn GraphStore>,
        input,
        0,
        1,
        vec![],
        Direction::Outgoing,
    );

    let chunk = op.next().unwrap().unwrap();
    let path_col = chunk.column(2).unwrap();
    assert_eq!(path_col.get_value(0).unwrap(), Value::Int64(2));
}
