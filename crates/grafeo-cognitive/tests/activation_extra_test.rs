//! Additional coverage tests for the spreading activation engine.

#![cfg(feature = "synapse")]

use grafeo_cognitive::activation::{ActivationSource, SpreadConfig, spread, spread_single};
use grafeo_common::types::NodeId;
use std::collections::HashMap;

struct MockGraph {
    adjacency: HashMap<NodeId, Vec<(NodeId, f64)>>,
}

impl MockGraph {
    fn new() -> Self {
        Self {
            adjacency: HashMap::new(),
        }
    }

    fn add_edge(&mut self, a: NodeId, b: NodeId, weight: f64) {
        self.adjacency.entry(a).or_default().push((b, weight));
        self.adjacency.entry(b).or_default().push((a, weight));
    }
}

impl ActivationSource for MockGraph {
    fn neighbors(&self, node_id: NodeId) -> Vec<(NodeId, f64)> {
        self.adjacency.get(&node_id).cloned().unwrap_or_default()
    }
}

fn n(id: u64) -> NodeId {
    NodeId::new(id)
}

// ---------------------------------------------------------------------------
// SpreadConfig::default() defaults
// ---------------------------------------------------------------------------

#[test]
fn spread_config_default_values() {
    let config = SpreadConfig::default();
    assert_eq!(config.max_hops, 3);
    assert!((config.min_propagated_energy - 0.01).abs() < 1e-10);
    assert!((config.decay_factor - 0.5).abs() < 1e-10);
    assert!((config.activation_threshold - 0.0).abs() < 1e-10);
}

// ---------------------------------------------------------------------------
// Spread with zero max_hops
// ---------------------------------------------------------------------------

#[test]
fn spread_zero_hops_only_source() {
    let mut g = MockGraph::new();
    g.add_edge(n(1), n(2), 1.0);

    let config = SpreadConfig {
        max_hops: 0,
        decay_factor: 1.0,
        min_propagated_energy: 0.001,
        activation_threshold: 0.0,
    };

    let map = spread_single(n(1), 1.0, &g, &config);
    assert_eq!(map.len(), 1);
    assert!(map.contains_key(&n(1)));
    assert!(!map.contains_key(&n(2)));
}

// ---------------------------------------------------------------------------
// Spread: diamond graph (multiple paths to same node)
// ---------------------------------------------------------------------------

#[test]
fn spread_diamond_graph_superposition() {
    //     1
    //    / \
    //   2   3
    //    \ /
    //     4
    let mut g = MockGraph::new();
    g.add_edge(n(1), n(2), 1.0);
    g.add_edge(n(1), n(3), 1.0);
    g.add_edge(n(2), n(4), 1.0);
    g.add_edge(n(3), n(4), 1.0);

    let config = SpreadConfig {
        max_hops: 2,
        decay_factor: 0.5,
        min_propagated_energy: 0.001,
        activation_threshold: 0.0,
    };

    let map = spread_single(n(1), 1.0, &g, &config);
    // Node 4 should receive energy from both paths: via 2 and via 3
    assert!(map.contains_key(&n(4)));
    let e4 = map[&n(4)];
    // Each path: 1.0 * 1.0 * 0.5 * 1.0 * 0.5 = 0.25, two paths → 0.5
    assert!((e4 - 0.5).abs() < 0.1, "expected ~0.5, got {e4}");
}

// ---------------------------------------------------------------------------
// Spread: cycle graph doesn't infinite loop
// ---------------------------------------------------------------------------

#[test]
fn spread_cycle_terminates() {
    let mut g = MockGraph::new();
    g.add_edge(n(1), n(2), 1.0);
    g.add_edge(n(2), n(3), 1.0);
    g.add_edge(n(3), n(1), 1.0);

    let config = SpreadConfig {
        max_hops: 5,
        decay_factor: 0.5,
        min_propagated_energy: 0.001,
        activation_threshold: 0.0,
    };

    // Should terminate (not infinite loop) because of decay
    let map = spread_single(n(1), 1.0, &g, &config);
    assert!(!map.is_empty());
}

// ---------------------------------------------------------------------------
// Spread: multiple sources on same node
// ---------------------------------------------------------------------------

#[test]
fn spread_multiple_sources_same_node() {
    let g = MockGraph::new();
    let config = SpreadConfig::default();

    let map = spread(&[(n(1), 1.0), (n(1), 2.0)], &g, &config);
    // Energies should be summed
    assert!((map[&n(1)] - 3.0).abs() < 1e-10);
}

// ---------------------------------------------------------------------------
// Spread: activation threshold filters source node
// ---------------------------------------------------------------------------

#[test]
fn spread_threshold_filters_low_energy_source() {
    let g = MockGraph::new();
    let config = SpreadConfig {
        max_hops: 1,
        decay_factor: 0.5,
        min_propagated_energy: 0.001,
        activation_threshold: 2.0,
    };

    let map = spread_single(n(1), 1.0, &g, &config);
    // Source has energy 1.0 < threshold 2.0 → filtered out
    assert!(map.is_empty());
}

// ---------------------------------------------------------------------------
// SpreadConfig: builder chain preserves other fields
// ---------------------------------------------------------------------------

#[test]
fn config_builder_preserves_defaults() {
    let config = SpreadConfig::default().with_max_hops(10);
    assert_eq!(config.max_hops, 10);
    // Other fields unchanged
    assert!((config.decay_factor - 0.5).abs() < 1e-10);
    assert!((config.min_propagated_energy - 0.01).abs() < 1e-10);
    assert!((config.activation_threshold - 0.0).abs() < 1e-10);
}

// ---------------------------------------------------------------------------
// Spread: node with no neighbors only returns itself
// ---------------------------------------------------------------------------

#[test]
fn spread_disconnected_node() {
    let mut g = MockGraph::new();
    // Add some edges that don't involve node 100
    g.add_edge(n(1), n(2), 1.0);

    let config = SpreadConfig::default();
    let map = spread_single(n(100), 5.0, &g, &config);

    assert_eq!(map.len(), 1);
    assert!((map[&n(100)] - 5.0).abs() < 1e-10);
}
