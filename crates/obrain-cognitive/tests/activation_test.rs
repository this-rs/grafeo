//! Integration tests for the spreading activation engine.

#![cfg(feature = "synapse")]

use obrain_cognitive::activation::{ActivationSource, SpreadConfig, spread, spread_single};
use obrain_cognitive::{SynapseActivationSource, SynapseConfig, SynapseStore};
use obrain_common::types::NodeId;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

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

#[test]
fn spread_single_hop() {
    let mut g = MockGraph::new();
    g.add_edge(n(1), n(2), 0.8);

    let config = SpreadConfig {
        max_hops: 1,
        decay_factor: 0.5,
        min_propagated_energy: 0.001,
        activation_threshold: 0.0,
        ..Default::default()
    };

    let map = spread_single(n(1), 1.0, &g, &config);
    assert!((map[&n(1)] - 1.0).abs() < 1e-10);
    assert!((map[&n(2)] - 0.4).abs() < 1e-10); // 1.0 * 0.8 * 0.5
}

#[test]
fn spread_two_hops_chain() {
    let mut g = MockGraph::new();
    g.add_edge(n(1), n(2), 0.8);
    g.add_edge(n(2), n(3), 0.6);

    let config = SpreadConfig {
        max_hops: 3,
        decay_factor: 0.5,
        min_propagated_energy: 0.001,
        activation_threshold: 0.0,
        ..Default::default()
    };

    let map = spread_single(n(1), 1.0, &g, &config);
    let b = map[&n(2)];
    // B gets 0.4 direct + back-propagation bounces from bidirectional edges
    assert!(b > 0.39, "B expected > 0.39, got {b}");
    let c = map[&n(3)];
    // C: 0.4 * 0.6 * 0.5 = 0.12 from first hop
    assert!((c - 0.12).abs() < 0.02, "C expected ~0.12, got {c}");
    // Verify ordering: source > B > C
    assert!(map[&n(1)] > b && b > c, "energy should decrease with hops");
}

#[test]
fn spread_max_hops_cutoff() {
    let mut g = MockGraph::new();
    g.add_edge(n(1), n(2), 1.0);
    g.add_edge(n(2), n(3), 1.0);
    g.add_edge(n(3), n(4), 1.0);

    let config = SpreadConfig {
        max_hops: 1,
        decay_factor: 1.0,
        min_propagated_energy: 0.001,
        activation_threshold: 0.0,
        ..Default::default()
    };

    let map = spread_single(n(1), 1.0, &g, &config);
    assert!(map.contains_key(&n(2)));
    assert!(!map.contains_key(&n(3)));
    assert!(!map.contains_key(&n(4)));
}

#[test]
fn spread_min_energy_cutoff() {
    let mut g = MockGraph::new();
    g.add_edge(n(1), n(2), 0.01);
    g.add_edge(n(2), n(3), 0.01);

    let config = SpreadConfig {
        max_hops: 3,
        decay_factor: 0.5,
        min_propagated_energy: 0.001,
        activation_threshold: 0.0,
        ..Default::default()
    };

    let map = spread_single(n(1), 1.0, &g, &config);
    assert!(map.contains_key(&n(2))); // 1.0*0.01*0.5 = 0.005
    // C: 0.005*0.01*0.5 = 0.000025 < 0.001 cutoff
    assert!(!map.contains_key(&n(3)) || map[&n(3)] < 0.001);
}

#[test]
fn spread_multi_source_superposition() {
    let mut g = MockGraph::new();
    g.add_edge(n(1), n(2), 0.8);
    g.add_edge(n(3), n(2), 0.6);

    let config = SpreadConfig {
        max_hops: 1,
        decay_factor: 0.5,
        min_propagated_energy: 0.001,
        activation_threshold: 0.0,
        ..Default::default()
    };

    let map = spread(&[(n(1), 1.0), (n(3), 1.0)], &g, &config);
    let b = map[&n(2)];
    assert!((b - 0.7).abs() < 0.01, "B expected ~0.7, got {b}");
}

#[test]
fn spread_activation_threshold_filters() {
    let mut g = MockGraph::new();
    g.add_edge(n(1), n(2), 1.0);
    g.add_edge(n(1), n(3), 0.01);

    let config = SpreadConfig {
        max_hops: 1,
        decay_factor: 0.5,
        min_propagated_energy: 0.001,
        activation_threshold: 0.1,
        ..Default::default()
    };

    let map = spread_single(n(1), 1.0, &g, &config);
    assert!(map.contains_key(&n(2))); // 0.5 >= 0.1
    assert!(!map.contains_key(&n(3))); // 0.005 < 0.1
}

#[test]
fn spread_empty_sources() {
    let g = MockGraph::new();
    let map = spread(&[], &g, &SpreadConfig::default());
    assert!(map.is_empty());
}

#[test]
fn spread_isolated_node() {
    let g = MockGraph::new();
    let map = spread_single(n(1), 1.0, &g, &SpreadConfig::default());
    assert_eq!(map.len(), 1);
    assert!((map[&n(1)] - 1.0).abs() < 1e-10);
}

#[test]
fn spread_with_synapse_store() {
    let store = Arc::new(SynapseStore::new(SynapseConfig {
        initial_weight: 0.0,
        default_half_life: Duration::from_secs(3600 * 24 * 365),
        ..SynapseConfig::default()
    }));

    store.reinforce(n(1), n(2), 0.8);
    store.reinforce(n(2), n(3), 0.6);

    let source = SynapseActivationSource::new(store);
    let config = SpreadConfig {
        max_hops: 2,
        decay_factor: 0.5,
        min_propagated_energy: 0.001,
        activation_threshold: 0.0,
        ..Default::default()
    };

    let map = spread_single(n(1), 1.0, &source, &config);
    assert!(map.contains_key(&n(2)), "node 2 should be activated");
    assert!(map.contains_key(&n(3)), "node 3 should be activated via 2");
    assert!(map[&n(2)] > map[&n(3)], "node 2 > node 3 in energy");
}

#[test]
fn spread_benchmark_10k_nodes() {
    let mut g = MockGraph::new();
    let node_count = 10_000u64;

    let mut seed: u64 = 42;
    for _ in 0..50_000 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let a = (seed >> 32) % node_count;
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let b = (seed >> 32) % node_count;
        if a != b {
            let weight = ((seed % 100) as f64) / 100.0;
            g.add_edge(n(a), n(b), weight.max(0.01));
        }
    }

    let config = SpreadConfig {
        max_hops: 3,
        decay_factor: 0.5,
        min_propagated_energy: 0.001,
        activation_threshold: 0.0,
        ..Default::default()
    };

    // Warm up
    let _ = spread_single(n(0), 1.0, &g, &config);

    let start = std::time::Instant::now();
    let iterations = 10;
    for i in 0..iterations {
        let _ = spread_single(n(i % node_count), 1.0, &g, &config);
    }
    let per_spread = start.elapsed() / iterations as u32;

    assert!(
        per_spread.as_millis() < 50,
        "spread took {:?} per call, expected < 50ms",
        per_spread
    );
}

// ---------------------------------------------------------------------------
// SpreadConfig builder methods
// ---------------------------------------------------------------------------

#[test]
fn config_with_max_hops() {
    let config = SpreadConfig::default().with_max_hops(5);
    assert_eq!(config.max_hops, 5);
    // Other fields unchanged
    assert!((config.decay_factor - 0.5).abs() < 1e-10);
}

#[test]
fn config_with_decay_factor() {
    let config = SpreadConfig::default().with_decay_factor(0.8);
    assert!((config.decay_factor - 0.8).abs() < 1e-10);
    assert_eq!(config.max_hops, 3); // default unchanged
}

#[test]
fn config_with_min_energy() {
    let config = SpreadConfig::default().with_min_energy(0.1);
    assert!((config.min_propagated_energy - 0.1).abs() < 1e-10);
}

#[test]
fn config_with_activation_threshold() {
    let config = SpreadConfig::default().with_activation_threshold(0.5);
    assert!((config.activation_threshold - 0.5).abs() < 1e-10);
}

#[test]
fn config_chained_builders() {
    let config = SpreadConfig::default()
        .with_max_hops(10)
        .with_decay_factor(0.9)
        .with_min_energy(0.05)
        .with_activation_threshold(0.2);
    assert_eq!(config.max_hops, 10);
    assert!((config.decay_factor - 0.9).abs() < 1e-10);
    assert!((config.min_propagated_energy - 0.05).abs() < 1e-10);
    assert!((config.activation_threshold - 0.2).abs() < 1e-10);
}

#[test]
fn config_debug_formatting() {
    let config = SpreadConfig::default();
    let dbg = format!("{:?}", config);
    assert!(dbg.contains("SpreadConfig"), "got: {dbg}");
    assert!(dbg.contains("max_hops"), "got: {dbg}");
    assert!(dbg.contains("decay_factor"), "got: {dbg}");
}
