//! Integration tests for the memory horizons subsystem.

#![cfg(all(feature = "memory", feature = "energy"))]

use grafeo_cognitive::memory::{
    ArchiveBackend, FileArchiveBackend, InMemoryArchiveBackend, MemoryConfig, MemoryHorizon,
    MemoryManager, MemoryStore, NodeMemoryState, SweepResult,
};
use grafeo_cognitive::{EnergyConfig, EnergyStore};
use grafeo_common::types::NodeId;
use std::sync::Arc;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// MemoryHorizon basics
// ---------------------------------------------------------------------------

#[test]
fn new_node_state_starts_operational() {
    let state = NodeMemoryState::new();
    assert_eq!(state.horizon, MemoryHorizon::Operational);
}

#[test]
fn horizon_display() {
    assert_eq!(format!("{}", MemoryHorizon::Operational), "operational");
    assert_eq!(format!("{}", MemoryHorizon::Consolidated), "consolidated");
    assert_eq!(format!("{}", MemoryHorizon::Archived), "archived");
}

// ---------------------------------------------------------------------------
// MemoryConfig defaults
// ---------------------------------------------------------------------------

#[test]
fn config_default_has_reasonable_values() {
    let config = MemoryConfig::default();
    assert_eq!(config.promotion_energy_threshold, 2.0);
    assert_eq!(config.promotion_min_age, Duration::from_secs(3600));
    assert_eq!(config.demotion_energy_threshold, 0.1);
    assert_eq!(config.demotion_max_idle, Duration::from_secs(7 * 24 * 3600));
    assert_eq!(config.sweep_interval, Duration::from_secs(3600));
}

// ---------------------------------------------------------------------------
// NodeMemoryState
// ---------------------------------------------------------------------------

#[test]
fn node_memory_state_starts_operational() {
    let state = NodeMemoryState::new();
    assert_eq!(state.horizon, MemoryHorizon::Operational);
    assert_eq!(state.transition_count, 0);
}

#[test]
fn node_memory_state_transition() {
    let mut state = NodeMemoryState::new();
    state.transition_to(MemoryHorizon::Consolidated);
    assert_eq!(state.horizon, MemoryHorizon::Consolidated);
    assert_eq!(state.transition_count, 1);

    // Same horizon → no-op
    state.transition_to(MemoryHorizon::Consolidated);
    assert_eq!(state.transition_count, 1);

    state.transition_to(MemoryHorizon::Archived);
    assert_eq!(state.horizon, MemoryHorizon::Archived);
    assert_eq!(state.transition_count, 2);
}

#[test]
fn node_memory_state_age() {
    let past = Instant::now()
        .checked_sub(Duration::from_secs(100))
        .unwrap();
    let state = NodeMemoryState::new_at(past);
    assert!(state.age() >= Duration::from_secs(99));
}

// ---------------------------------------------------------------------------
// MemoryStore
// ---------------------------------------------------------------------------

#[test]
fn store_default_horizon_is_operational_for_unknown_nodes() {
    let store = MemoryStore::new();
    assert_eq!(store.get_horizon(NodeId(999)), MemoryHorizon::Operational);
}

#[test]
fn store_is_thread_safe() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<MemoryStore>();
}

#[test]
fn store_track_and_get() {
    let store = MemoryStore::new();
    store.track(NodeId(1));
    assert_eq!(store.get_horizon(NodeId(1)), MemoryHorizon::Operational);
    assert_eq!(store.len(), 1);
}

#[test]
fn store_set_horizon() {
    let store = MemoryStore::new();
    store.track(NodeId(1));
    store.set_horizon(NodeId(1), MemoryHorizon::Consolidated);
    assert_eq!(store.get_horizon(NodeId(1)), MemoryHorizon::Consolidated);
}

#[test]
fn store_list_by_horizon() {
    let store = MemoryStore::new();
    store.track(NodeId(1));
    store.track(NodeId(2));
    store.track(NodeId(3));
    store.set_horizon(NodeId(2), MemoryHorizon::Consolidated);
    store.set_horizon(NodeId(3), MemoryHorizon::Archived);

    let operational = store.list_by_horizon(MemoryHorizon::Operational);
    assert_eq!(operational.len(), 1);
    assert!(operational.contains(&NodeId(1)));

    let consolidated = store.list_by_horizon(MemoryHorizon::Consolidated);
    assert_eq!(consolidated.len(), 1);
    assert!(consolidated.contains(&NodeId(2)));

    let archived = store.list_by_horizon(MemoryHorizon::Archived);
    assert_eq!(archived.len(), 1);
    assert!(archived.contains(&NodeId(3)));
}

#[test]
fn store_horizon_counts() {
    let store = MemoryStore::new();
    store.track(NodeId(1));
    store.track(NodeId(2));
    store.track(NodeId(3));
    store.set_horizon(NodeId(2), MemoryHorizon::Consolidated);
    store.set_horizon(NodeId(3), MemoryHorizon::Archived);

    let counts = store.horizon_counts();
    assert_eq!(*counts.get(&MemoryHorizon::Operational).unwrap_or(&0), 1);
    assert_eq!(*counts.get(&MemoryHorizon::Consolidated).unwrap_or(&0), 1);
    assert_eq!(*counts.get(&MemoryHorizon::Archived).unwrap_or(&0), 1);
}

#[test]
fn store_snapshot() {
    let store = MemoryStore::new();
    store.track(NodeId(10));
    store.set_horizon(NodeId(10), MemoryHorizon::Consolidated);

    let snap = store.snapshot();
    assert_eq!(snap.len(), 1);
    assert_eq!(snap[0].0, NodeId(10));
    assert_eq!(snap[0].1.horizon, MemoryHorizon::Consolidated);
}

// ---------------------------------------------------------------------------
// MemoryManager — sweep promotion
// ---------------------------------------------------------------------------

#[test]
fn sweep_promotes_high_energy_old_node() {
    let energy_config = EnergyConfig {
        boost_on_mutation: 1.0,
        default_energy: 1.0,
        default_half_life: Duration::from_secs(3600 * 24),
        min_energy: 0.01,
        max_energy: 10.0,
    };
    let energy_store = Arc::new(EnergyStore::new(energy_config));
    let memory_store = Arc::new(MemoryStore::new());
    let config = MemoryConfig {
        promotion_energy_threshold: 2.0,
        promotion_min_age: Duration::from_secs(60), // 1 min
        demotion_energy_threshold: 0.1,
        demotion_max_idle: Duration::from_secs(3600),
        sweep_interval: Duration::from_secs(60),
    };

    let node = NodeId(1);

    // Track node and boost energy above threshold
    memory_store.track(node);
    energy_store.boost(node, 3.0); // energy = 3.0 > threshold of 2.0

    // Sweep at a time far enough in the future that age > promotion_min_age
    let future = Instant::now() + Duration::from_secs(120);
    let manager = MemoryManager::new(memory_store.clone(), config, energy_store.clone());
    let result = manager.sweep_at(future);

    assert!(result.promoted.contains(&node), "node should be promoted");
    assert_eq!(memory_store.get_horizon(node), MemoryHorizon::Consolidated);
}

#[test]
fn sweep_does_not_promote_young_node() {
    let energy_config = EnergyConfig::default();
    let energy_store = Arc::new(EnergyStore::new(energy_config));
    let memory_store = Arc::new(MemoryStore::new());
    let config = MemoryConfig {
        promotion_energy_threshold: 2.0,
        promotion_min_age: Duration::from_secs(3600), // 1 hour — node won't be old enough
        demotion_energy_threshold: 0.1,
        demotion_max_idle: Duration::from_secs(86400),
        sweep_interval: Duration::from_secs(60),
    };

    let node = NodeId(2);
    memory_store.track(node);
    energy_store.boost(node, 5.0); // high energy, but too young

    let manager = MemoryManager::new(memory_store.clone(), config, energy_store);

    // Sweep at ~now — node was just created, age < promotion_min_age
    let result = manager.sweep_at(Instant::now());
    assert!(
        result.promoted.is_empty(),
        "young node should not be promoted"
    );
    assert_eq!(memory_store.get_horizon(node), MemoryHorizon::Operational);
}

// ---------------------------------------------------------------------------
// MemoryManager — sweep demotion
// ---------------------------------------------------------------------------

#[test]
fn sweep_demotes_low_energy_idle_node_from_consolidated() {
    let energy_config = EnergyConfig {
        default_half_life: Duration::from_secs(60), // fast decay
        ..Default::default()
    };
    let energy_store = Arc::new(EnergyStore::new(energy_config));
    let memory_store = Arc::new(MemoryStore::new());
    let config = MemoryConfig {
        promotion_energy_threshold: 2.0,
        promotion_min_age: Duration::from_secs(60),
        demotion_energy_threshold: 0.1,
        demotion_max_idle: Duration::from_secs(300), // 5 min
        sweep_interval: Duration::from_secs(60),
    };

    let node = NodeId(3);
    memory_store.track(node);
    memory_store.set_horizon(node, MemoryHorizon::Consolidated);
    // Don't boost energy — it will be 0.0 (never tracked in energy store)

    let manager = MemoryManager::new(memory_store.clone(), config, energy_store);

    // Sweep far in the future — idle time > demotion_max_idle
    let future = Instant::now() + Duration::from_secs(600);
    let result = manager.sweep_at(future);

    assert!(
        result.demoted.contains(&node),
        "idle consolidated node should be demoted"
    );
    assert_eq!(memory_store.get_horizon(node), MemoryHorizon::Archived);
}

#[test]
fn sweep_emergency_demotes_operational_node() {
    // Operational node with very low energy AND old enough → skip Consolidated, go to Archived
    let energy_config = EnergyConfig::default();
    let energy_store = Arc::new(EnergyStore::new(energy_config));
    let memory_store = Arc::new(MemoryStore::new());
    let config = MemoryConfig {
        promotion_energy_threshold: 2.0,
        promotion_min_age: Duration::from_secs(60),
        demotion_energy_threshold: 0.1,
        demotion_max_idle: Duration::from_secs(300),
        sweep_interval: Duration::from_secs(60),
    };

    let node = NodeId(4);
    memory_store.track(node);
    // energy is 0.0 (not tracked), which is < 0.1

    let manager = MemoryManager::new(memory_store.clone(), config, energy_store);

    // Sweep far enough that age > demotion_max_idle
    let future = Instant::now() + Duration::from_secs(600);
    let result = manager.sweep_at(future);

    assert!(
        result.demoted.contains(&node),
        "old low-energy operational node should be demoted"
    );
    assert_eq!(memory_store.get_horizon(node), MemoryHorizon::Archived);
}

// ---------------------------------------------------------------------------
// MemoryManager — sweep reactivation
// ---------------------------------------------------------------------------

#[test]
fn sweep_reactivates_archived_node_with_high_energy() {
    let energy_config = EnergyConfig::default();
    let energy_store = Arc::new(EnergyStore::new(energy_config));
    let memory_store = Arc::new(MemoryStore::new());
    let config = MemoryConfig {
        promotion_energy_threshold: 2.0,
        promotion_min_age: Duration::from_secs(60),
        demotion_energy_threshold: 0.1,
        demotion_max_idle: Duration::from_secs(300),
        sweep_interval: Duration::from_secs(60),
    };

    let node = NodeId(5);
    memory_store.track(node);
    memory_store.set_horizon(node, MemoryHorizon::Archived);

    // Boost energy above promotion threshold → should reactivate
    energy_store.boost(node, 3.0);

    let manager = MemoryManager::new(memory_store.clone(), config, energy_store);
    let result = manager.sweep();

    assert!(
        result.reactivated.contains(&node),
        "archived node with high energy should be reactivated"
    );
    assert_eq!(memory_store.get_horizon(node), MemoryHorizon::Operational);
}

#[test]
fn sweep_does_not_reactivate_low_energy_archived_node() {
    let energy_config = EnergyConfig::default();
    let energy_store = Arc::new(EnergyStore::new(energy_config));
    let memory_store = Arc::new(MemoryStore::new());
    let config = MemoryConfig::default();

    let node = NodeId(6);
    memory_store.track(node);
    memory_store.set_horizon(node, MemoryHorizon::Archived);
    // No energy boost — stays at 0.0

    let manager = MemoryManager::new(memory_store.clone(), config, energy_store);
    let result = manager.sweep();

    assert!(result.reactivated.is_empty());
    assert_eq!(memory_store.get_horizon(node), MemoryHorizon::Archived);
}

// ---------------------------------------------------------------------------
// SweepResult
// ---------------------------------------------------------------------------

#[test]
fn sweep_result_total_transitions() {
    let result = SweepResult {
        promoted: vec![NodeId(1)],
        demoted: vec![NodeId(2), NodeId(3)],
        reactivated: vec![],
    };
    assert_eq!(result.total_transitions(), 3);
}

// ---------------------------------------------------------------------------
// Periodic sweep (async)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn periodic_sweep_runs_and_can_shutdown() {
    let energy_config = EnergyConfig::default();
    let energy_store = Arc::new(EnergyStore::new(energy_config));
    let memory_store = Arc::new(MemoryStore::new());
    let config = MemoryConfig {
        sweep_interval: Duration::from_millis(50), // fast for testing
        ..Default::default()
    };

    let manager = Arc::new(MemoryManager::new(
        memory_store.clone(),
        config,
        energy_store,
    ));

    let handle = Arc::clone(&manager).start_periodic();

    // Let it run a couple ticks
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Shutdown
    manager.shutdown();
    let _ = tokio::time::timeout(Duration::from_secs(1), handle).await;
    // If we get here without hanging, the periodic sweep shut down correctly
}

// ---------------------------------------------------------------------------
// Archive backends
// ---------------------------------------------------------------------------

#[tokio::test]
async fn in_memory_archive_roundtrip() {
    let backend = InMemoryArchiveBackend::new();
    let node = NodeId(42);
    let data = b"hello archive";

    backend.archive(node, data).await.unwrap();
    assert!(backend.exists(node).await.unwrap());
    assert_eq!(backend.len(), 1);

    let restored = backend.restore(node).await.unwrap();
    assert_eq!(restored.as_deref(), Some(data.as_slice()));

    backend.remove(node).await.unwrap();
    assert!(!backend.exists(node).await.unwrap());
    assert!(backend.is_empty());
}

#[tokio::test]
async fn in_memory_archive_restore_missing_returns_none() {
    let backend = InMemoryArchiveBackend::new();
    let result = backend.restore(NodeId(999)).await.unwrap();
    assert!(result.is_none());
}

#[tokio::test]
async fn file_archive_roundtrip() {
    let tmp_dir = std::env::temp_dir().join(format!("grafeo_memory_test_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp_dir); // clean up from previous runs

    let backend = FileArchiveBackend::new(&tmp_dir);
    let node = NodeId(100);
    let data = b"{\"key\": \"value\"}";

    backend.archive(node, data).await.unwrap();
    assert!(backend.exists(node).await.unwrap());

    let restored = backend.restore(node).await.unwrap();
    assert_eq!(restored.as_deref(), Some(data.as_slice()));

    backend.remove(node).await.unwrap();
    assert!(!backend.exists(node).await.unwrap());

    // Restore of missing node returns None
    let result = backend.restore(NodeId(999)).await.unwrap();
    assert!(result.is_none());

    // Cleanup
    let _ = std::fs::remove_dir_all(&tmp_dir);
}

// ---------------------------------------------------------------------------
// Mixed scenario: full lifecycle
// ---------------------------------------------------------------------------

#[test]
fn full_lifecycle_operational_to_consolidated_to_archived_to_reactivated() {
    // Phase 1: Operational → Consolidated (via sweep with high energy + age)
    let energy_config = EnergyConfig {
        default_half_life: Duration::from_secs(3600),
        ..Default::default()
    };
    let energy_store = Arc::new(EnergyStore::new(energy_config));
    let memory_store = Arc::new(MemoryStore::new());
    let config = MemoryConfig {
        promotion_energy_threshold: 2.0,
        promotion_min_age: Duration::from_secs(60),
        demotion_energy_threshold: 0.1,
        demotion_max_idle: Duration::from_secs(300),
        sweep_interval: Duration::from_secs(60),
    };

    let node = NodeId(10);
    memory_store.track(node);
    energy_store.boost(node, 5.0); // energy = 5.0 > threshold 2.0

    let manager = MemoryManager::new(memory_store.clone(), config.clone(), energy_store.clone());

    // Sweep at future time so node age > promotion_min_age
    let t1 = Instant::now() + Duration::from_secs(120);
    let result = manager.sweep_at(t1);
    assert!(
        result.promoted.contains(&node),
        "should promote high-energy old node"
    );
    assert_eq!(memory_store.get_horizon(node), MemoryHorizon::Consolidated);

    // Phase 2: Consolidated → Archived
    // Manually set horizon back to consolidated to control the test
    // The energy is still high (no real time passed), so we use a separate
    // energy store with no energy for this node to simulate decay.
    let energy_config2 = EnergyConfig::default();
    let energy_store2 = Arc::new(EnergyStore::new(energy_config2));
    // Don't boost — energy is 0.0 < demotion threshold 0.1
    let manager2 = MemoryManager::new(memory_store.clone(), config.clone(), energy_store2);

    // Sweep far enough that time_in_horizon > demotion_max_idle
    let t2 = t1 + Duration::from_secs(600);
    let result = manager2.sweep_at(t2);
    assert!(
        result.demoted.contains(&node),
        "should demote low-energy idle node"
    );
    assert_eq!(memory_store.get_horizon(node), MemoryHorizon::Archived);

    // Phase 3: Archived → Operational (reactivation via energy boost)
    let energy_config3 = EnergyConfig::default();
    let energy_store3 = Arc::new(EnergyStore::new(energy_config3));
    energy_store3.boost(node, 5.0); // re-energize
    let manager3 = MemoryManager::new(memory_store.clone(), config, energy_store3);

    let result = manager3.sweep();
    assert!(
        result.reactivated.contains(&node),
        "should reactivate boosted archived node"
    );
    assert_eq!(memory_store.get_horizon(node), MemoryHorizon::Operational);
}

// ---------------------------------------------------------------------------
// Filtering queries by horizon
// ---------------------------------------------------------------------------

#[test]
fn filter_nodes_by_horizon_for_queries() {
    let store = MemoryStore::new();

    // Simulate a graph with nodes in different horizons
    for i in 1..=10 {
        store.track(NodeId(i));
    }
    // Consolidate nodes 3-5
    for i in 3..=5 {
        store.set_horizon(NodeId(i), MemoryHorizon::Consolidated);
    }
    // Archive nodes 8-10
    for i in 8..=10 {
        store.set_horizon(NodeId(i), MemoryHorizon::Archived);
    }

    let operational = store.list_by_horizon(MemoryHorizon::Operational);
    assert_eq!(operational.len(), 4); // 1, 2, 6, 7

    let consolidated = store.list_by_horizon(MemoryHorizon::Consolidated);
    assert_eq!(consolidated.len(), 3); // 3, 4, 5

    let archived = store.list_by_horizon(MemoryHorizon::Archived);
    assert_eq!(archived.len(), 3); // 8, 9, 10
}

// ---------------------------------------------------------------------------
// NodeMemoryState — time_in_current_horizon
// ---------------------------------------------------------------------------

#[test]
fn node_memory_state_time_in_current_horizon() {
    let past = Instant::now().checked_sub(Duration::from_secs(50)).unwrap();
    let state = NodeMemoryState::new_at(past);
    let time = state.time_in_current_horizon();
    assert!(time >= Duration::from_secs(49));
}

// ---------------------------------------------------------------------------
// MemoryHorizon Display (already tested, but ensure all three)
// ---------------------------------------------------------------------------

#[test]
fn horizon_display_all_variants() {
    assert_eq!(MemoryHorizon::Operational.to_string(), "operational");
    assert_eq!(MemoryHorizon::Consolidated.to_string(), "consolidated");
    assert_eq!(MemoryHorizon::Archived.to_string(), "archived");
}

// ---------------------------------------------------------------------------
// MemoryStore Debug impl
// ---------------------------------------------------------------------------

#[test]
fn memory_store_debug_impl() {
    let store = MemoryStore::new();
    store.track(NodeId(1));
    let s = format!("{:?}", store);
    assert!(s.contains("MemoryStore"));
    assert!(s.contains("tracked_nodes"));
}

// ---------------------------------------------------------------------------
// MemoryManager Debug impl
// ---------------------------------------------------------------------------

#[test]
fn memory_manager_debug_impl() {
    let energy_store = Arc::new(EnergyStore::new(EnergyConfig::default()));
    let memory_store = Arc::new(MemoryStore::new());
    let config = MemoryConfig::default();
    let manager = MemoryManager::new(memory_store, config, energy_store);
    let s = format!("{:?}", manager);
    assert!(s.contains("MemoryManager"));
    assert!(s.contains("has_archive"));
}

// ---------------------------------------------------------------------------
// MemoryManager::with_archive
// ---------------------------------------------------------------------------

#[test]
fn memory_manager_with_archive() {
    let energy_store = Arc::new(EnergyStore::new(EnergyConfig::default()));
    let memory_store = Arc::new(MemoryStore::new());
    let config = MemoryConfig::default();
    let archive = Arc::new(InMemoryArchiveBackend::new());
    let manager = MemoryManager::new(memory_store, config, energy_store)
        .with_archive(archive as Arc<dyn ArchiveBackend>);
    let s = format!("{:?}", manager);
    assert!(s.contains("has_archive: true"));
}

// ---------------------------------------------------------------------------
// MemoryManager::shutdown (sync path)
// ---------------------------------------------------------------------------

#[test]
fn memory_manager_shutdown_does_not_panic() {
    let energy_store = Arc::new(EnergyStore::new(EnergyConfig::default()));
    let memory_store = Arc::new(MemoryStore::new());
    let config = MemoryConfig::default();
    let manager = MemoryManager::new(memory_store, config, energy_store);
    // Calling shutdown without a periodic task should not panic
    manager.shutdown();
}

// ---------------------------------------------------------------------------
// SweepResult::total_transitions with all fields
// ---------------------------------------------------------------------------

#[test]
fn sweep_result_total_transitions_all_fields() {
    let result = SweepResult {
        promoted: vec![NodeId(1), NodeId(2)],
        demoted: vec![NodeId(3)],
        reactivated: vec![NodeId(4), NodeId(5), NodeId(6)],
    };
    assert_eq!(result.total_transitions(), 6);
}

#[test]
fn sweep_result_total_transitions_empty() {
    let result = SweepResult::default();
    assert_eq!(result.total_transitions(), 0);
}

// ---------------------------------------------------------------------------
// MemoryStore::Default impl
// ---------------------------------------------------------------------------

#[test]
fn memory_store_default_impl() {
    let store = MemoryStore::default();
    assert!(store.is_empty());
}
