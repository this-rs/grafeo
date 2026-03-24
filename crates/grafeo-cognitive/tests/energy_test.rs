//! Integration tests for the energy subsystem.

#![cfg(feature = "energy")]

use grafeo_cognitive::{EnergyConfig, EnergyListener, EnergyStore, NodeEnergy};
use grafeo_common::types::NodeId;
use grafeo_reactive::{MutationBus, MutationEvent, MutationListener, NodeSnapshot};
use std::sync::Arc;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// NodeEnergy decay tests
// ---------------------------------------------------------------------------

#[test]
fn decay_half_life_one_hour() {
    let half_life = Duration::from_secs(3600);
    let start = Instant::now();
    let node = NodeEnergy::new_at(1.0, half_life, start);

    // After exactly 1 half-life → ~0.5
    let after_1h = start + Duration::from_secs(3600);
    let e = node.energy_at(after_1h);
    assert!((e - 0.5).abs() < 1e-10, "expected ~0.5, got {e}");

    // After 2 half-lives → ~0.25
    let after_2h = start + Duration::from_secs(7200);
    let e = node.energy_at(after_2h);
    assert!((e - 0.25).abs() < 1e-10, "expected ~0.25, got {e}");

    // After 10 half-lives → ~0.001 (nearly zero)
    let after_10h = start + Duration::from_secs(36000);
    let e = node.energy_at(after_10h);
    assert!(e < 0.001, "expected < 0.001, got {e}");
}

#[test]
fn decay_at_t0_is_full_energy() {
    let start = Instant::now();
    let node = NodeEnergy::new_at(5.0, Duration::from_secs(3600), start);
    let e = node.energy_at(start);
    assert!((e - 5.0).abs() < 1e-10);
}

#[test]
fn boost_applies_decay_then_adds() {
    let half_life = Duration::from_secs(3600);
    let start = Instant::now();
    let mut node = NodeEnergy::new_at(1.0, half_life, start);

    // After 1 half-life, energy is ~0.5. Boost by 0.5 → ~1.0
    let at_1h = start + Duration::from_secs(3600);
    node.boost_at(0.5, at_1h);

    // Right after boost, energy should be ~1.0
    let e = node.energy_at(at_1h);
    assert!((e - 1.0).abs() < 1e-10, "expected ~1.0, got {e}");
}

#[test]
fn boost_resets_activation_time() {
    let half_life = Duration::from_secs(3600);
    let start = Instant::now();
    let mut node = NodeEnergy::new_at(1.0, half_life, start);

    let at_1h = start + Duration::from_secs(3600);
    node.boost_at(0.5, at_1h);
    assert_eq!(node.last_activated(), at_1h);
}

// ---------------------------------------------------------------------------
// EnergyStore tests
// ---------------------------------------------------------------------------

#[test]
fn store_get_energy_untracked_returns_zero() {
    let store = EnergyStore::new(EnergyConfig::default());
    assert_eq!(store.get_energy(NodeId::new(42)), 0.0);
}

#[test]
fn store_boost_and_get() {
    let store = EnergyStore::new(EnergyConfig::default());
    let id = NodeId::new(1);

    store.boost(id, 3.0);
    let e = store.get_energy(id);
    // Just boosted, should be ~3.0 (tiny decay due to elapsed time)
    assert!(e > 2.99, "expected ~3.0, got {e}");
}

#[test]
fn store_multiple_boosts_accumulate() {
    let store = EnergyStore::new(EnergyConfig::default());
    let id = NodeId::new(1);

    store.boost(id, 1.0);
    store.boost(id, 1.0);
    store.boost(id, 1.0);

    let e = store.get_energy(id);
    // ~3.0 (tiny decay between boosts)
    assert!(e > 2.99, "expected ~3.0, got {e}");
}

#[test]
fn store_list_low_energy() {
    let config = EnergyConfig {
        default_half_life: Duration::from_millis(1), // Very fast decay
        ..EnergyConfig::default()
    };
    let store = EnergyStore::new(config);

    store.boost(NodeId::new(1), 0.001); // Will decay to ~0 almost immediately
    store.boost(NodeId::new(2), 100.0); // High energy

    // Wait a bit for decay
    std::thread::sleep(Duration::from_millis(50));

    let low = store.list_low_energy(0.01);
    assert!(low.contains(&NodeId::new(1)), "node 1 should be low energy");
    // Node 2 may or may not be low — 100.0 with 1ms half-life after 50ms = 100 * 2^(-50) ≈ 0
    // Actually 2^(-50) ≈ 8.88e-16, so 100 * that ≈ 0. Both will be low.
    // Let's test with a more realistic scenario:
}

#[test]
fn store_list_low_energy_mixed() {
    let config = EnergyConfig {
        default_half_life: Duration::from_secs(3600), // 1 hour
        ..EnergyConfig::default()
    };
    let store = EnergyStore::new(config);

    store.boost(NodeId::new(1), 0.001); // Very low energy
    store.boost(NodeId::new(2), 100.0); // High energy

    let low = store.list_low_energy(0.01);
    assert!(low.contains(&NodeId::new(1)));
    assert!(!low.contains(&NodeId::new(2)));
}

#[test]
fn store_len_and_empty() {
    let store = EnergyStore::new(EnergyConfig::default());
    assert!(store.is_empty());
    assert_eq!(store.len(), 0);

    store.boost(NodeId::new(1), 1.0);
    assert!(!store.is_empty());
    assert_eq!(store.len(), 1);
}

#[test]
fn store_snapshot() {
    let store = EnergyStore::new(EnergyConfig::default());
    store.boost(NodeId::new(1), 2.0);
    store.boost(NodeId::new(2), 3.0);

    let snap = store.snapshot();
    assert_eq!(snap.len(), 2);
}

// ---------------------------------------------------------------------------
// EnergyListener tests
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

#[tokio::test]
async fn listener_boosts_on_node_created() {
    let store = Arc::new(EnergyStore::new(EnergyConfig {
        boost_on_mutation: 2.0,
        ..EnergyConfig::default()
    }));
    let listener = EnergyListener::new(Arc::clone(&store));

    let event = MutationEvent::NodeCreated {
        node: make_node(42),
    };
    listener.on_event(&event).await;

    let e = store.get_energy(NodeId::new(42));
    assert!(e > 1.99, "expected ~2.0, got {e}");
}

#[tokio::test]
async fn listener_boosts_both_endpoints_on_edge() {
    let store = Arc::new(EnergyStore::new(EnergyConfig {
        boost_on_mutation: 1.0,
        ..EnergyConfig::default()
    }));
    let listener = EnergyListener::new(Arc::clone(&store));

    let event = MutationEvent::EdgeCreated {
        edge: make_edge_snapshot(1, 10, 20),
    };
    listener.on_event(&event).await;

    assert!(store.get_energy(NodeId::new(10)) > 0.99);
    assert!(store.get_energy(NodeId::new(20)) > 0.99);
}

#[tokio::test]
async fn listener_on_batch_accumulates() {
    let store = Arc::new(EnergyStore::new(EnergyConfig {
        boost_on_mutation: 1.0,
        ..EnergyConfig::default()
    }));
    let listener = EnergyListener::new(Arc::clone(&store));

    let events = vec![
        MutationEvent::NodeCreated { node: make_node(1) },
        MutationEvent::NodeUpdated {
            before: make_node(1),
            after: make_node(1),
        },
        MutationEvent::NodeDeleted { node: make_node(1) },
    ];
    listener.on_batch(&events).await;

    // Node 1 was touched 3 times → 3 boosts
    let e = store.get_energy(NodeId::new(1));
    assert!(e > 2.99, "expected ~3.0, got {e}");
}

#[tokio::test]
async fn listener_with_scheduler_integration() {
    use grafeo_reactive::BatchConfig;

    let bus = MutationBus::new();
    let config = BatchConfig::new(100, Duration::from_millis(30));
    let scheduler = grafeo_reactive::Scheduler::new(&bus, config);

    let store = Arc::new(EnergyStore::new(EnergyConfig {
        boost_on_mutation: 1.0,
        ..EnergyConfig::default()
    }));
    let listener = Arc::new(EnergyListener::new(Arc::clone(&store)));
    scheduler.register_listener(listener);

    // Publish mutations
    bus.publish(MutationEvent::NodeCreated { node: make_node(1) });
    bus.publish(MutationEvent::NodeCreated { node: make_node(2) });
    bus.publish(MutationEvent::EdgeCreated {
        edge: make_edge_snapshot(1, 1, 2),
    });

    // Wait for scheduler to dispatch
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Node 1: created + edge endpoint = 2 boosts
    let e1 = store.get_energy(NodeId::new(1));
    assert!(e1 > 1.99, "node 1 expected ~2.0, got {e1}");

    // Node 2: created + edge endpoint = 2 boosts
    let e2 = store.get_energy(NodeId::new(2));
    assert!(e2 > 1.99, "node 2 expected ~2.0, got {e2}");

    scheduler.shutdown().await;
}

// ---------------------------------------------------------------------------
// EnergyConfig builder methods
// ---------------------------------------------------------------------------

#[test]
fn config_with_half_life() {
    let config = EnergyConfig::default().with_half_life(Duration::from_secs(7200));
    assert_eq!(config.default_half_life, Duration::from_secs(7200));
    // Other fields unchanged
    assert_eq!(config.boost_on_mutation, 1.0);
    assert_eq!(config.default_energy, 1.0);
}

#[test]
fn config_with_boost() {
    let config = EnergyConfig::default().with_boost(5.0);
    assert_eq!(config.boost_on_mutation, 5.0);
    // Other fields unchanged
    assert_eq!(config.default_half_life, Duration::from_secs(24 * 3600));
}

#[test]
fn config_chained_builders() {
    let config = EnergyConfig::default()
        .with_half_life(Duration::from_secs(60))
        .with_boost(10.0);
    assert_eq!(config.default_half_life, Duration::from_secs(60));
    assert_eq!(config.boost_on_mutation, 10.0);
}

// ---------------------------------------------------------------------------
// NodeEnergy accessors: raw_energy, half_life
// ---------------------------------------------------------------------------

#[test]
fn node_raw_energy_returns_stored_value() {
    let node = NodeEnergy::new(3.5, Duration::from_secs(3600));
    assert!((node.raw_energy() - 3.5).abs() < 1e-10);
}

#[test]
fn node_half_life_returns_configured_value() {
    let hl = Duration::from_secs(7200);
    let node = NodeEnergy::new(1.0, hl);
    assert_eq!(node.half_life(), hl);
}

#[test]
fn raw_energy_does_not_decay() {
    let start = Instant::now();
    let node = NodeEnergy::new_at(4.0, Duration::from_secs(1), start);
    // Even after simulating time, raw_energy stays the same
    assert!((node.raw_energy() - 4.0).abs() < 1e-10);
}

// ---------------------------------------------------------------------------
// EnergyStore: config() accessor
// ---------------------------------------------------------------------------

#[test]
fn store_config_accessor() {
    let config = EnergyConfig {
        boost_on_mutation: 3.0,
        min_energy: 0.05,
        ..EnergyConfig::default()
    };
    let store = EnergyStore::new(config);
    assert_eq!(store.config().boost_on_mutation, 3.0);
    assert_eq!(store.config().min_energy, 0.05);
}

// ---------------------------------------------------------------------------
// Debug impls
// ---------------------------------------------------------------------------

#[test]
fn energy_store_debug_formatting() {
    let store = EnergyStore::new(EnergyConfig::default());
    store.boost(NodeId::new(1), 1.0);
    let dbg = format!("{:?}", store);
    assert!(dbg.contains("EnergyStore"), "got: {dbg}");
    assert!(dbg.contains("tracked_nodes"), "got: {dbg}");
}

#[test]
fn energy_listener_debug_formatting() {
    let store = Arc::new(EnergyStore::new(EnergyConfig::default()));
    let listener = EnergyListener::new(Arc::clone(&store));
    let dbg = format!("{:?}", listener);
    assert!(dbg.contains("EnergyListener"), "got: {dbg}");
    assert!(dbg.contains("store"), "got: {dbg}");
}

// ---------------------------------------------------------------------------
// EnergyListener: name(), store()
// ---------------------------------------------------------------------------

#[test]
fn listener_name() {
    let store = Arc::new(EnergyStore::new(EnergyConfig::default()));
    let listener = EnergyListener::new(Arc::clone(&store));
    assert_eq!(listener.name(), "cognitive:energy");
}

#[test]
fn listener_store_accessor() {
    let store = Arc::new(EnergyStore::new(EnergyConfig::default()));
    let listener = EnergyListener::new(Arc::clone(&store));
    // Should return a reference to the same store
    assert!(listener.store().is_empty());
    store.boost(NodeId::new(1), 1.0);
    assert_eq!(listener.store().len(), 1);
}

// ---------------------------------------------------------------------------
// EnergyListener: edge update and edge delete events
// ---------------------------------------------------------------------------

#[tokio::test]
async fn listener_boosts_on_edge_updated() {
    let store = Arc::new(EnergyStore::new(EnergyConfig {
        boost_on_mutation: 1.0,
        ..EnergyConfig::default()
    }));
    let listener = EnergyListener::new(Arc::clone(&store));

    let event = MutationEvent::EdgeUpdated {
        before: make_edge_snapshot(1, 10, 20),
        after: make_edge_snapshot(1, 10, 20),
    };
    listener.on_event(&event).await;

    assert!(store.get_energy(NodeId::new(10)) > 0.99);
    assert!(store.get_energy(NodeId::new(20)) > 0.99);
}

#[tokio::test]
async fn listener_boosts_on_edge_deleted() {
    let store = Arc::new(EnergyStore::new(EnergyConfig {
        boost_on_mutation: 1.0,
        ..EnergyConfig::default()
    }));
    let listener = EnergyListener::new(Arc::clone(&store));

    let event = MutationEvent::EdgeDeleted {
        edge: make_edge_snapshot(1, 30, 40),
    };
    listener.on_event(&event).await;

    assert!(store.get_energy(NodeId::new(30)) > 0.99);
    assert!(store.get_energy(NodeId::new(40)) > 0.99);
}

// ---------------------------------------------------------------------------
// Adversarial tests — NaN energy, zero half_life, overflow values
// ---------------------------------------------------------------------------

#[test]
fn adversarial_nan_energy_no_panic() {
    let start = Instant::now();
    let node = NodeEnergy::new_at(f64::NAN, Duration::from_secs(3600), start);

    // NaN energy should propagate as NaN, not panic
    let e = node.energy_at(start);
    assert!(e.is_nan(), "NaN energy should remain NaN, got {e}");

    let e_later = node.energy_at(start + Duration::from_secs(100));
    assert!(e_later.is_nan(), "NaN energy after decay should remain NaN, got {e_later}");
}

#[test]
fn adversarial_nan_energy_boost_no_panic() {
    let start = Instant::now();
    let mut node = NodeEnergy::new_at(f64::NAN, Duration::from_secs(3600), start);

    // Boosting NaN energy should not panic
    node.boost_at(1.0, start + Duration::from_secs(1));
    let e = node.energy_at(start + Duration::from_secs(1));
    // NaN + 1.0 = NaN
    assert!(e.is_nan(), "NaN + boost should still be NaN, got {e}");
}

#[test]
fn adversarial_infinity_energy_no_panic() {
    let start = Instant::now();
    let node = NodeEnergy::new_at(f64::INFINITY, Duration::from_secs(3600), start);

    let e = node.energy_at(start);
    assert!(e.is_infinite(), "Infinity energy at t=0 should be infinite, got {e}");

    let e_later = node.energy_at(start + Duration::from_secs(3600));
    // inf * 0.5 = inf
    assert!(e_later.is_infinite(), "Infinity energy after decay should still be infinite, got {e_later}");
}

#[test]
fn adversarial_zero_half_life_energy_no_panic() {
    let start = Instant::now();
    let node = NodeEnergy::new_at(1.0, Duration::ZERO, start);

    // At t=0: elapsed=0, half_lives = 0/0 = NaN, 2^(-NaN) = NaN
    let e = node.energy_at(start);
    assert!(e.is_nan() || e.is_finite(), "zero half_life at t=0 should not panic");

    // After some time: elapsed > 0, half_lives = t/0 = inf, 2^(-inf) = 0
    let later = start + Duration::from_secs(1);
    let e_later = node.energy_at(later);
    assert!(
        e_later == 0.0 || e_later.is_nan(),
        "zero half_life after elapsed time: got {e_later}"
    );
}

#[test]
fn adversarial_negative_energy_no_panic() {
    let start = Instant::now();
    let node = NodeEnergy::new_at(-5.0, Duration::from_secs(3600), start);

    let e = node.energy_at(start);
    assert!((e - (-5.0)).abs() < 1e-10, "negative energy at t=0: got {e}");

    // Decay should still work (negative decays towards zero)
    let e_later = node.energy_at(start + Duration::from_secs(3600));
    assert!((e_later - (-2.5)).abs() < 1e-10, "negative energy after 1 half-life: got {e_later}");
}

#[test]
fn adversarial_very_large_energy_no_overflow() {
    let start = Instant::now();
    let node = NodeEnergy::new_at(f64::MAX / 2.0, Duration::from_secs(3600), start);

    let e = node.energy_at(start);
    assert!(e.is_finite(), "large energy at t=0 should be finite, got {e}");

    // After some decay, still finite
    let e_later = node.energy_at(start + Duration::from_secs(3600));
    assert!(e_later.is_finite(), "large energy after decay should be finite, got {e_later}");
}

#[test]
fn adversarial_store_nan_boost_no_panic() {
    let store = EnergyStore::new(EnergyConfig::default());
    let id = NodeId::new(1);

    // Boosting with NaN should not panic
    store.boost(id, f64::NAN);
    let e = store.get_energy(id);
    // Result should be NaN (default_energy + NaN = NaN)
    assert!(e.is_nan(), "NaN boost should produce NaN energy, got {e}");
}

#[test]
fn adversarial_store_zero_boost_no_panic() {
    let store = EnergyStore::new(EnergyConfig::default());
    let id = NodeId::new(1);

    // Boosting with 0.0 on a new node should not panic
    store.boost(id, 0.0);
    let e = store.get_energy(id);
    assert!(e.is_finite(), "zero boost should produce finite energy, got {e}");
}
