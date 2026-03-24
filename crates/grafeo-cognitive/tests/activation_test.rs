//! Integration tests for the spreading activation engine.

use grafeo_cognitive::activation::{ActivationSource, SpreadConfig, spread, spread_single};
use grafeo_cognitive::{SynapseConfig, SynapseStore};
use grafeo_common::types::NodeId;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Mock graph source for controlled testing
// ---------------------------------------------------------------------------

struct MockGraph {
    /// node → [(neighbor, weight)]
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
// Tests
// ---------------------------------------------------------------------------

#[test]
fn spread_single_hop() {
    // A --0.8--> B
    let mut g = MockGraph::new();
    g.add_edge(n(1), n(2), 0.8);

    let config = SpreadConfig {
        max_hops: 1,
        decay_factor: 0.5,
        min_propagated_energy: 0.001,
        activation_threshold: 0.0,
    };

    let map = spread_single(n(1), 1.0, &g, &config);

    // Source: 1.0
    assert!((map[&n(1)] - 1.0).abs() < 1e-10);
    // B: 1.0 * 0.8 * 0.5 = 0.4
    assert!((map[&n(2)] - 0.4).abs() < 1e-10);
}

#[test]
fn spread_two_hops_chain() {
    // A --0.8--> B --0.6--> C (bidirectional edges)
    // With bidirectional edges, energy can flow back: A→B→A→B etc.
    let mut g = MockGraph::new();
    g.add_edge(n(1), n(2), 0.8);
    g.add_edge(n(2), n(3), 0.6);

    // Use max_hops=1 first to verify the direct hop
    let config_1hop = SpreadConfig {
        max_hops: 1,
        decay_factor: 0.5,
        min_propagated_energy: 0.001,
        activation_threshold: 0.0,
    };

    let map = spread_single(n(1), 1.0, &g, &config_1hop);

    // B: 1.0 * 0.8 * 0.5 = 0.4 (direct, no backflow)
    let b = map[&n(2)];
    assert!((b - 0.4).abs() < 0.01, "B expected ~0.4, got {b}");

    // Now test 2 hops — C gets energy from B
    let config_2hop = SpreadConfig {
        max_hops: 2,
        decay_factor: 0.5,
        min_propagated_energy: 0.001,
        activation_threshold: 0.0,
    };

    let map = spread_single(n(1), 1.0, &g, &config_2hop);

    // C: 0.4 * 0.6 * 0.5 = 0.12 (only from B→C at hop 2)
    let c = map[&n(3)];
    assert!((c - 0.12).abs() < 0.01, "C expected ~0.12, got {c}");

    // B gets some energy back from A→B (hop 1) + C→B (hop 2, 0.12*0.6*0.5=0.036)
    // and from A via B→A→B (hop 2, 0.4*0.8*0.5*0.8*0.5=0.064)
    // Total B > 0.4
    let b = map[&n(2)];
    assert!(b >= 0.4, "B should be at least 0.4, got {b}");
}

#[test]
fn spread_max_hops_cutoff() {
    // A → B → C → D, max_hops = 1
    let mut g = MockGraph::new();
    g.add_edge(n(1), n(2), 1.0);
    g.add_edge(n(2), n(3), 1.0);
    g.add_edge(n(3), n(4), 1.0);

    let config = SpreadConfig {
        max_hops: 1,
        decay_factor: 1.0,
        min_propagated_energy: 0.001,
        activation_threshold: 0.0,
    };

    let map = spread_single(n(1), 1.0, &g, &config);

    assert!(map.contains_key(&n(2)), "B should be reached");
    assert!(!map.contains_key(&n(3)), "C should NOT be reached (2 hops)");
    assert!(!map.contains_key(&n(4)), "D should NOT be reached (3 hops)");
}

#[test]
fn spread_min_energy_cutoff() {
    // A → B → C with very small weights
    let mut g = MockGraph::new();
    g.add_edge(n(1), n(2), 0.01);
    g.add_edge(n(2), n(3), 0.01);

    let config = SpreadConfig {
        max_hops: 3,
        decay_factor: 0.5,
        min_propagated_energy: 0.001,
        activation_threshold: 0.0,
    };

    let map = spread_single(n(1), 1.0, &g, &config);

    // B: 1.0 * 0.01 * 0.5 = 0.005 (above 0.001)
    assert!(map.contains_key(&n(2)));
    // C: 0.005 * 0.01 * 0.5 = 0.000025 (below 0.001 cutoff)
    assert!(
        !map.contains_key(&n(3)) || map[&n(3)] < 0.001,
        "C should be below cutoff"
    );
}

#[test]
fn spread_multi_source_superposition() {
    // A → B ← C
    let mut g = MockGraph::new();
    g.add_edge(n(1), n(2), 0.8);
    g.add_edge(n(3), n(2), 0.6);

    let config = SpreadConfig {
        max_hops: 1,
        decay_factor: 0.5,
        min_propagated_energy: 0.001,
        activation_threshold: 0.0,
    };

    let map = spread(&[(n(1), 1.0), (n(3), 1.0)], &g, &config);

    // B from A: 1.0 * 0.8 * 0.5 = 0.4
    // B from C: 1.0 * 0.6 * 0.5 = 0.3
    // Total B: 0.7
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
        activation_threshold: 0.1, // Only include nodes with >= 0.1
    };

    let map = spread_single(n(1), 1.0, &g, &config);

    // B: 1.0 * 1.0 * 0.5 = 0.5 (above 0.1)
    assert!(map.contains_key(&n(2)));
    // C: 1.0 * 0.01 * 0.5 = 0.005 (below 0.1)
    assert!(!map.contains_key(&n(3)));
}

#[test]
fn spread_empty_sources() {
    let g = MockGraph::new();
    let config = SpreadConfig::default();
    let map = spread(&[], &g, &config);
    assert!(map.is_empty());
}

#[test]
fn spread_isolated_node() {
    let g = MockGraph::new();
    let config = SpreadConfig::default();
    let map = spread_single(n(1), 1.0, &g, &config);
    assert_eq!(map.len(), 1);
    assert!((map[&n(1)] - 1.0).abs() < 1e-10);
}

#[test]
fn spread_with_synapse_store() {
    use grafeo_cognitive::SynapseActivationSource;

    let store = Arc::new(SynapseStore::new(SynapseConfig {
        initial_weight: 0.0,
        default_half_life: Duration::from_secs(3600 * 24 * 365),
        ..SynapseConfig::default()
    }));

    // Create synapses: 1↔2 (weight ~0.8), 2↔3 (weight ~0.6)
    store.reinforce(n(1), n(2), 0.8);
    store.reinforce(n(2), n(3), 0.6);

    let source = SynapseActivationSource::new(store);
    let config = SpreadConfig {
        max_hops: 2,
        decay_factor: 0.5,
        min_propagated_energy: 0.001,
        activation_threshold: 0.0,
    };

    let map = spread_single(n(1), 1.0, &source, &config);
    assert!(map.contains_key(&n(2)), "node 2 should be activated");
    assert!(map.contains_key(&n(3)), "node 3 should be activated via 2");
    assert!(
        map[&n(2)] > map[&n(3)],
        "node 2 should have more energy than 3"
    );
}

// ---------------------------------------------------------------------------
// Performance benchmark (inline)
// ---------------------------------------------------------------------------

#[test]
fn spread_benchmark_10k_nodes_50k_edges() {
    use std::time::Instant;

    // Build a random-ish graph: 10K nodes, ~50K edges
    let mut g = MockGraph::new();
    let node_count = 10_000u64;
    let edge_count = 50_000;

    // Deterministic pseudo-random edges
    let mut seed: u64 = 42;
    for _ in 0..edge_count {
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
    };

    // Warm up
    let _ = spread_single(n(0), 1.0, &g, &config);

    // Measure
    let start = Instant::now();
    let iterations = 10;
    for i in 0..iterations {
        let _ = spread_single(n(i % node_count), 1.0, &g, &config);
    }
    let elapsed = start.elapsed();
    let per_spread = elapsed / iterations as u32;

    assert!(
        per_spread.as_millis() < 50,
        "spread took {:?} per call, expected < 50ms",
        per_spread
    );
}
