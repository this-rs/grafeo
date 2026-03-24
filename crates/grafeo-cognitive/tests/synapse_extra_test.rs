//! Additional coverage tests for the synapse subsystem.

#![cfg(feature = "synapse")]

use grafeo_cognitive::{Synapse, SynapseConfig, SynapseListener, SynapseStore};
use grafeo_common::types::NodeId;
use grafeo_reactive::{MutationEvent, MutationListener, NodeSnapshot};
use std::sync::Arc;
use std::time::Duration;

fn make_node(id: u64) -> NodeSnapshot {
    NodeSnapshot {
        id: NodeId::new(id),
        labels: smallvec::smallvec![arcstr::literal!("Test")],
        properties: vec![],
    }
}

// ---------------------------------------------------------------------------
// Synapse::new (non-_at variant)
// ---------------------------------------------------------------------------

#[test]
fn synapse_new_creates_with_current_time() {
    let syn = Synapse::new(
        NodeId::new(1),
        NodeId::new(2),
        1.0,
        Duration::from_secs(3600),
    );
    assert_eq!(syn.source, NodeId::new(1));
    assert_eq!(syn.target, NodeId::new(2));
    assert_eq!(syn.reinforcement_count, 1);
    // Weight should be ~1.0 since just created
    assert!((syn.current_weight() - 1.0).abs() < 0.01);
}

#[test]
fn synapse_reinforce_non_at_variant() {
    let mut syn = Synapse::new(
        NodeId::new(1),
        NodeId::new(2),
        1.0,
        Duration::from_secs(3600),
    );
    syn.reinforce(0.5);
    assert_eq!(syn.reinforcement_count, 2);
    // Weight should be ~1.5
    assert!((syn.current_weight() - 1.5).abs() < 0.01);
}

// ---------------------------------------------------------------------------
// SynapseStore: config() accessor
// ---------------------------------------------------------------------------

#[test]
fn store_config_accessor() {
    let config = SynapseConfig {
        initial_weight: 0.5,
        min_weight: 0.05,
        ..SynapseConfig::default()
    };
    let store = SynapseStore::new(config);
    assert_eq!(store.config().initial_weight, 0.5);
    assert_eq!(store.config().min_weight, 0.05);
}

// ---------------------------------------------------------------------------
// SynapseStore: get_synapse returns None for nonexistent
// ---------------------------------------------------------------------------

#[test]
fn store_get_synapse_nonexistent() {
    let store = SynapseStore::new(SynapseConfig::default());
    assert!(store.get_synapse(NodeId::new(1), NodeId::new(2)).is_none());
}

// ---------------------------------------------------------------------------
// SynapseStore: list_synapses empty for unconnected node
// ---------------------------------------------------------------------------

#[test]
fn store_list_synapses_unconnected_node() {
    let store = SynapseStore::new(SynapseConfig::default());
    store.reinforce(NodeId::new(1), NodeId::new(2), 0.5);
    let synapses = store.list_synapses(NodeId::new(99));
    assert!(synapses.is_empty());
}

// ---------------------------------------------------------------------------
// SynapseStore: prune with mixed strong/weak
// ---------------------------------------------------------------------------

#[test]
fn store_prune_mixed_strong_weak() {
    let store = SynapseStore::new(SynapseConfig {
        initial_weight: 0.0,
        default_half_life: Duration::from_secs(3600 * 24 * 365),
        ..SynapseConfig::default()
    });

    store.reinforce(NodeId::new(1), NodeId::new(2), 5.0); // strong
    store.reinforce(NodeId::new(3), NodeId::new(4), 0.001); // weak
    store.reinforce(NodeId::new(5), NodeId::new(6), 0.0005); // weaker

    let pruned = store.prune(0.01);
    assert_eq!(pruned, 2);
    assert_eq!(store.len(), 1);
    assert!(store.get_synapse(NodeId::new(1), NodeId::new(2)).is_some());
}

// ---------------------------------------------------------------------------
// SynapseListener: on_event is no-op for single events
// ---------------------------------------------------------------------------

#[tokio::test]
async fn listener_on_event_is_noop() {
    let store = Arc::new(SynapseStore::new(SynapseConfig::default()));
    let listener = SynapseListener::new(Arc::clone(&store));

    let event = MutationEvent::NodeCreated { node: make_node(1) };
    listener.on_event(&event).await;

    assert!(store.is_empty(), "single event should not create synapses");
}

// ---------------------------------------------------------------------------
// SynapseConfig: default values
// ---------------------------------------------------------------------------

#[test]
fn synapse_config_default_values() {
    let config = SynapseConfig::default();
    assert_eq!(config.initial_weight, 0.1);
    assert_eq!(config.reinforce_amount, 0.2);
    assert_eq!(config.default_half_life, Duration::from_secs(7 * 24 * 3600));
    assert_eq!(config.min_weight, 0.01);
}

// ---------------------------------------------------------------------------
// Synapse: created_at field
// ---------------------------------------------------------------------------

#[test]
fn synapse_created_at_is_preserved() {
    let start = std::time::Instant::now();
    let syn = Synapse::new_at(
        NodeId::new(1),
        NodeId::new(2),
        1.0,
        Duration::from_secs(3600),
        start,
    );
    assert_eq!(syn.created_at, start);

    // Reinforcing does not change created_at
    let mut syn = syn;
    let later = start + Duration::from_secs(100);
    syn.reinforce_at(0.5, later);
    assert_eq!(syn.created_at, start);
}

// ---------------------------------------------------------------------------
// SynapseStore: len after multiple operations
// ---------------------------------------------------------------------------

#[test]
fn store_len_tracks_unique_pairs() {
    let store = SynapseStore::new(SynapseConfig::default());
    assert_eq!(store.len(), 0);

    store.reinforce(NodeId::new(1), NodeId::new(2), 0.1);
    assert_eq!(store.len(), 1);

    // Same pair → no new synapse
    store.reinforce(NodeId::new(1), NodeId::new(2), 0.1);
    assert_eq!(store.len(), 1);

    // Reverse order → same pair
    store.reinforce(NodeId::new(2), NodeId::new(1), 0.1);
    assert_eq!(store.len(), 1);

    // New pair
    store.reinforce(NodeId::new(3), NodeId::new(4), 0.1);
    assert_eq!(store.len(), 2);
}
