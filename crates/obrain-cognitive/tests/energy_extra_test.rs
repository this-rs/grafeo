//! Additional coverage tests for the energy subsystem.

#![cfg(feature = "energy")]

use obrain_cognitive::{EnergyConfig, EnergyListener, EnergyStore, NodeEnergy};
use obrain_common::types::NodeId;
use obrain_reactive::{MutationEvent, MutationListener, NodeSnapshot};
use std::sync::Arc;
use std::time::{Duration, Instant};

fn make_node(id: u64) -> NodeSnapshot {
    NodeSnapshot {
        id: NodeId::new(id),
        labels: smallvec::smallvec![arcstr::literal!("Test")],
        properties: vec![],
    }
}

// ---------------------------------------------------------------------------
// NodeEnergy::new (non-_at variant)
// ---------------------------------------------------------------------------

#[test]
fn node_energy_new_creates_with_current_time() {
    let node = NodeEnergy::new(2.0, Duration::from_secs(3600));
    // Just created, energy should be very close to 2.0
    let e = node.current_energy();
    assert!((e - 2.0).abs() < 0.01, "expected ~2.0, got {e}");
}

#[test]
fn node_energy_boost_non_at_variant() {
    let mut node = NodeEnergy::new(1.0, Duration::from_secs(3600));
    node.boost(0.5);
    // Just boosted, should be ~1.5
    let e = node.current_energy();
    assert!((e - 1.5).abs() < 0.01, "expected ~1.5, got {e}");
}

// ---------------------------------------------------------------------------
// EnergyStore: boost existing vs new node
// ---------------------------------------------------------------------------

#[test]
fn store_boost_existing_node_adds_to_energy() {
    let store = EnergyStore::new(EnergyConfig::default());
    let id = NodeId::new(1);

    store.boost(id, 2.0);
    let e1 = store.get_energy(id);
    assert!(e1 > 1.99);

    store.boost(id, 3.0);
    let e2 = store.get_energy(id);
    // Should be ~5.0 (2.0 + 3.0 with minimal decay)
    assert!(e2 > 4.99, "expected ~5.0, got {e2}");
}

#[test]
fn store_boost_new_node_uses_amount_as_initial() {
    let store = EnergyStore::new(EnergyConfig {
        default_half_life: Duration::from_secs(3600),
        ..EnergyConfig::default()
    });
    let id = NodeId::new(99);
    store.boost(id, 7.5);
    let e = store.get_energy(id);
    assert!(e > 7.4, "expected ~7.5, got {e}");
}

// ---------------------------------------------------------------------------
// EnergyStore: snapshot returns all nodes with their decayed energy
// ---------------------------------------------------------------------------

#[test]
fn store_snapshot_includes_decayed_energy() {
    let store = EnergyStore::new(EnergyConfig {
        default_half_life: Duration::from_secs(3600 * 24),
        ..EnergyConfig::default()
    });
    store.boost(NodeId::new(1), 1.0);
    store.boost(NodeId::new(2), 2.0);
    store.boost(NodeId::new(3), 3.0);

    let snap = store.snapshot();
    assert_eq!(snap.len(), 3);

    // All should have positive energy
    for (_, energy) in &snap {
        assert!(*energy > 0.0);
    }
}

// ---------------------------------------------------------------------------
// EnergyStore: list_low_energy returns empty when no nodes are tracked
// ---------------------------------------------------------------------------

#[test]
fn store_list_low_energy_empty_store() {
    let store = EnergyStore::new(EnergyConfig::default());
    let low = store.list_low_energy(1.0);
    assert!(low.is_empty());
}

// ---------------------------------------------------------------------------
// EnergyStore: multiple distinct nodes
// ---------------------------------------------------------------------------

#[test]
fn store_tracks_multiple_nodes_independently() {
    let store = EnergyStore::new(EnergyConfig::default());

    store.boost(NodeId::new(1), 5.0);
    store.boost(NodeId::new(2), 0.5);
    store.boost(NodeId::new(3), 9.0);

    assert_eq!(store.len(), 3);

    // Each node has its own energy (capped at max_energy=10.0 by default)
    let e1 = store.get_energy(NodeId::new(1));
    let e2 = store.get_energy(NodeId::new(2));
    let e3 = store.get_energy(NodeId::new(3));

    assert!(e1 > 4.9, "e1={e1}");
    assert!(e2 > 0.4 && e2 < 1.0, "e2={e2}");
    assert!(e3 > 8.9, "e3={e3}");
}

// ---------------------------------------------------------------------------
// EnergyListener: on_event with NodeUpdated
// ---------------------------------------------------------------------------

#[tokio::test]
async fn listener_boosts_on_node_updated() {
    let store = Arc::new(EnergyStore::new(EnergyConfig {
        boost_on_mutation: 1.5,
        ..EnergyConfig::default()
    }));
    let listener = EnergyListener::new(Arc::clone(&store));

    let event = MutationEvent::NodeUpdated {
        before: make_node(10),
        after: make_node(10),
    };
    listener.on_event(&event).await;

    let e = store.get_energy(NodeId::new(10));
    assert!(e > 1.4, "expected ~1.5, got {e}");
}

// ---------------------------------------------------------------------------
// EnergyListener: on_event with NodeDeleted
// ---------------------------------------------------------------------------

#[tokio::test]
async fn listener_boosts_on_node_deleted() {
    let store = Arc::new(EnergyStore::new(EnergyConfig {
        boost_on_mutation: 1.0,
        ..EnergyConfig::default()
    }));
    let listener = EnergyListener::new(Arc::clone(&store));

    let event = MutationEvent::NodeDeleted {
        node: make_node(99),
    };
    listener.on_event(&event).await;

    let e = store.get_energy(NodeId::new(99));
    assert!(e > 0.99);
}

// ---------------------------------------------------------------------------
// NodeEnergy: energy_at far in future is near zero
// ---------------------------------------------------------------------------

#[test]
fn node_energy_at_far_future_near_zero() {
    let start = Instant::now();
    let node = NodeEnergy::new_at(1.0, Duration::from_secs(60), start);
    // 1000 half-lives later
    let far = start + Duration::from_secs(60_000);
    let e = node.energy_at(far);
    assert!(e < 1e-100, "expected near-zero, got {e}");
}

// ---------------------------------------------------------------------------
// EnergyConfig: default values
// ---------------------------------------------------------------------------

#[test]
fn energy_config_default_values() {
    let config = EnergyConfig::default();
    assert_eq!(config.boost_on_mutation, 1.0);
    assert_eq!(config.default_energy, 1.0);
    assert_eq!(config.default_half_life, Duration::from_secs(24 * 3600));
    assert_eq!(config.min_energy, 0.01);
}
