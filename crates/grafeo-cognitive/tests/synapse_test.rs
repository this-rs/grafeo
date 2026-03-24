//! Integration tests for the synapse subsystem.

use grafeo_cognitive::{Synapse, SynapseConfig, SynapseListener, SynapseStore};
use grafeo_common::types::NodeId;
use grafeo_reactive::{MutationEvent, MutationListener, NodeSnapshot};
use std::sync::Arc;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_node(id: u64) -> NodeSnapshot {
    NodeSnapshot {
        id: NodeId::new(id),
        labels: smallvec::smallvec![arcstr::literal!("Test")],
        properties: vec![],
    }
}

fn make_edge_snapshot(id: u64, src: u64, dst: u64) -> grafeo_reactive::EdgeSnapshot {
    grafeo_reactive::EdgeSnapshot {
        id: grafeo_common::types::EdgeId::new(id),
        src: NodeId::new(src),
        dst: NodeId::new(dst),
        edge_type: arcstr::literal!("KNOWS"),
        properties: vec![],
    }
}

// ---------------------------------------------------------------------------
// Synapse decay tests
// ---------------------------------------------------------------------------

#[test]
fn synapse_decay_half_life() {
    let half_life = Duration::from_secs(24 * 3600); // 24h
    let start = Instant::now();
    let syn = Synapse::new_at(NodeId::new(1), NodeId::new(2), 1.0, half_life, start);

    // After 1 half-life → ~0.5
    let at = start + Duration::from_secs(24 * 3600);
    let w = syn.weight_at(at);
    assert!((w - 0.5).abs() < 1e-10, "expected ~0.5, got {w}");

    // After 2 half-lives → ~0.25
    let at = start + Duration::from_secs(48 * 3600);
    let w = syn.weight_at(at);
    assert!((w - 0.25).abs() < 1e-10, "expected ~0.25, got {w}");
}

#[test]
fn synapse_at_t0_is_full() {
    let start = Instant::now();
    let syn = Synapse::new_at(
        NodeId::new(1),
        NodeId::new(2),
        2.0,
        Duration::from_secs(3600),
        start,
    );
    let w = syn.weight_at(start);
    assert!((w - 2.0).abs() < 1e-10);
}

#[test]
fn synapse_reinforce_applies_decay_then_adds() {
    let half_life = Duration::from_secs(3600);
    let start = Instant::now();
    let mut syn = Synapse::new_at(NodeId::new(1), NodeId::new(2), 1.0, half_life, start);

    let at_1h = start + Duration::from_secs(3600);
    syn.reinforce_at(0.3, at_1h);

    // After decay: 0.5, then +0.3 = 0.8
    let w = syn.weight_at(at_1h);
    assert!((w - 0.8).abs() < 1e-10, "expected ~0.8, got {w}");
    assert_eq!(syn.reinforcement_count, 2);
}

// ---------------------------------------------------------------------------
// SynapseStore tests
// ---------------------------------------------------------------------------

#[test]
fn store_reinforce_creates_synapse() {
    let store = SynapseStore::new(SynapseConfig::default());
    store.reinforce(NodeId::new(1), NodeId::new(2), 0.5);

    let s = store.get_synapse(NodeId::new(1), NodeId::new(2));
    assert!(s.is_some());
    let s = s.unwrap();
    assert!(s.current_weight() > 0.5); // initial_weight + amount
}

#[test]
fn store_reinforce_symmetric() {
    let store = SynapseStore::new(SynapseConfig::default());
    store.reinforce(NodeId::new(1), NodeId::new(2), 0.5);

    // Can retrieve in either direction
    assert!(store.get_synapse(NodeId::new(2), NodeId::new(1)).is_some());
}

#[test]
fn store_no_self_synapses() {
    let store = SynapseStore::new(SynapseConfig::default());
    store.reinforce(NodeId::new(1), NodeId::new(1), 1.0);
    assert!(store.is_empty());
}

#[test]
fn store_multiple_reinforcements() {
    let store = SynapseStore::new(SynapseConfig {
        initial_weight: 0.1,
        reinforce_amount: 0.2,
        ..SynapseConfig::default()
    });

    store.reinforce(NodeId::new(1), NodeId::new(2), 0.2);
    store.reinforce(NodeId::new(1), NodeId::new(2), 0.2);
    store.reinforce(NodeId::new(1), NodeId::new(2), 0.2);

    let s = store.get_synapse(NodeId::new(1), NodeId::new(2)).unwrap();
    // First: 0.1 + 0.2 = 0.3. Second: ~0.3 + 0.2 = ~0.5. Third: ~0.5 + 0.2 = ~0.7
    assert!(
        s.current_weight() > 0.69,
        "expected ~0.7, got {}",
        s.current_weight()
    );
    assert_eq!(s.reinforcement_count, 3); // 1 (creation) + 2 reinforce
}

#[test]
fn store_list_synapses_sorted_by_weight() {
    let store = SynapseStore::new(SynapseConfig::default());

    store.reinforce(NodeId::new(1), NodeId::new(2), 0.1);
    store.reinforce(NodeId::new(1), NodeId::new(3), 0.5);
    store.reinforce(NodeId::new(1), NodeId::new(4), 0.3);

    let synapses = store.list_synapses(NodeId::new(1));
    assert_eq!(synapses.len(), 3);

    // Should be sorted by weight descending
    let weights: Vec<f64> = synapses.iter().map(|s| s.current_weight()).collect();
    for i in 1..weights.len() {
        assert!(weights[i - 1] >= weights[i], "not sorted: {:?}", weights);
    }
}

#[test]
fn store_prune_removes_weak_synapses() {
    let config = SynapseConfig {
        initial_weight: 0.001,
        default_half_life: Duration::from_millis(1),
        ..SynapseConfig::default()
    };
    let store = SynapseStore::new(config);

    store.reinforce(NodeId::new(1), NodeId::new(2), 0.001);
    // Wait for decay
    std::thread::sleep(Duration::from_millis(50));

    let pruned = store.prune(0.01);
    assert_eq!(pruned, 1);
    assert!(store.is_empty());
}

#[test]
fn store_prune_keeps_strong_synapses() {
    let store = SynapseStore::new(SynapseConfig::default());
    store.reinforce(NodeId::new(1), NodeId::new(2), 5.0); // Strong

    let pruned = store.prune(0.01);
    assert_eq!(pruned, 0);
    assert_eq!(store.len(), 1);
}

// ---------------------------------------------------------------------------
// SynapseListener tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn listener_co_activation_creates_synapses() {
    let store = Arc::new(SynapseStore::new(SynapseConfig::default()));
    let listener = SynapseListener::new(Arc::clone(&store));

    // Batch with mutations on nodes 1, 2, and 3
    let events = vec![
        MutationEvent::NodeCreated { node: make_node(1) },
        MutationEvent::NodeCreated { node: make_node(2) },
        MutationEvent::NodeCreated { node: make_node(3) },
    ];
    listener.on_batch(&events).await;

    // All pairs should have synapses: (1,2), (1,3), (2,3)
    assert!(store.get_synapse(NodeId::new(1), NodeId::new(2)).is_some());
    assert!(store.get_synapse(NodeId::new(1), NodeId::new(3)).is_some());
    assert!(store.get_synapse(NodeId::new(2), NodeId::new(3)).is_some());
    assert_eq!(store.len(), 3);
}

#[tokio::test]
async fn listener_edge_coactivates_endpoints() {
    let store = Arc::new(SynapseStore::new(SynapseConfig::default()));
    let listener = SynapseListener::new(Arc::clone(&store));

    let events = vec![MutationEvent::EdgeCreated {
        edge: make_edge_snapshot(1, 10, 20),
    }];
    listener.on_batch(&events).await;

    // Edge creates co-activation between src and dst
    assert!(
        store
            .get_synapse(NodeId::new(10), NodeId::new(20))
            .is_some()
    );
}

#[tokio::test]
async fn listener_repeated_batches_reinforce() {
    let store = Arc::new(SynapseStore::new(SynapseConfig {
        reinforce_amount: 0.5,
        ..SynapseConfig::default()
    }));
    let listener = SynapseListener::new(Arc::clone(&store));

    let events = vec![
        MutationEvent::NodeCreated { node: make_node(1) },
        MutationEvent::NodeCreated { node: make_node(2) },
    ];

    listener.on_batch(&events).await;
    let w1 = store
        .get_synapse(NodeId::new(1), NodeId::new(2))
        .unwrap()
        .current_weight();

    listener.on_batch(&events).await;
    let w2 = store
        .get_synapse(NodeId::new(1), NodeId::new(2))
        .unwrap()
        .current_weight();

    assert!(
        w2 > w1,
        "weight should increase with reinforcement: {w1} -> {w2}"
    );
}

#[tokio::test]
async fn listener_single_node_batch_no_synapse() {
    let store = Arc::new(SynapseStore::new(SynapseConfig::default()));
    let listener = SynapseListener::new(Arc::clone(&store));

    let events = vec![MutationEvent::NodeCreated { node: make_node(1) }];
    listener.on_batch(&events).await;

    // Only one node — no pairs to co-activate
    assert!(store.is_empty());
}
