//! Coverage-boosting tests for the memory horizons subsystem.

#![cfg(all(feature = "memory", feature = "energy"))]

use grafeo_cognitive::memory::{
    ArchiveBackend, FileArchiveBackend, InMemoryArchiveBackend, MemoryConfig, MemoryHorizon,
    MemoryManager, MemoryStore,
};
use grafeo_cognitive::{EnergyConfig, EnergyStore};
use grafeo_common::types::NodeId;
use std::sync::Arc;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// FileArchiveBackend — new, archive, restore, exists, remove
// ---------------------------------------------------------------------------

#[tokio::test]
async fn file_archive_backend_full_lifecycle() {
    let tmp_dir = std::env::temp_dir().join(format!("grafeo_fab_coverage_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp_dir);

    let backend = FileArchiveBackend::new(&tmp_dir);
    let node = NodeId(200);
    let payload = b"test-payload-data";

    // Initially does not exist
    assert!(!backend.exists(node).await.unwrap());

    // Restore of missing node returns None
    assert!(backend.restore(node).await.unwrap().is_none());

    // Archive
    backend.archive(node, payload).await.unwrap();
    assert!(backend.exists(node).await.unwrap());

    // Restore returns correct data
    let restored = backend.restore(node).await.unwrap();
    assert_eq!(restored.as_deref(), Some(payload.as_slice()));

    // Remove
    backend.remove(node).await.unwrap();
    assert!(!backend.exists(node).await.unwrap());

    // Remove of already-removed node is idempotent
    backend.remove(node).await.unwrap();

    // Restore after remove returns None
    assert!(backend.restore(node).await.unwrap().is_none());

    let _ = std::fs::remove_dir_all(&tmp_dir);
}

#[tokio::test]
async fn file_archive_backend_multiple_nodes() {
    let tmp_dir = std::env::temp_dir().join(format!("grafeo_fab_multi_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp_dir);

    let backend = FileArchiveBackend::new(&tmp_dir);

    for i in 1..=5 {
        let node = NodeId(i);
        let data = format!("data-for-node-{}", i);
        backend.archive(node, data.as_bytes()).await.unwrap();
    }

    // All should exist
    for i in 1..=5 {
        assert!(backend.exists(NodeId(i)).await.unwrap());
    }

    // Restore each and verify content
    for i in 1..=5 {
        let restored = backend.restore(NodeId(i)).await.unwrap().unwrap();
        let expected = format!("data-for-node-{}", i);
        assert_eq!(restored, expected.as_bytes());
    }

    let _ = std::fs::remove_dir_all(&tmp_dir);
}

// ---------------------------------------------------------------------------
// Helper: create a MemoryManager with short durations for testing
// ---------------------------------------------------------------------------

fn make_test_setup(
    promotion_min_age: Duration,
    demotion_max_idle: Duration,
) -> (Arc<MemoryStore>, Arc<EnergyStore>, MemoryConfig) {
    let energy_config = EnergyConfig {
        boost_on_mutation: 1.0,
        default_energy: 1.0,
        default_half_life: Duration::from_secs(3600 * 24),
        min_energy: 0.01,
    };
    let energy_store = Arc::new(EnergyStore::new(energy_config));
    let memory_store = Arc::new(MemoryStore::new());
    let config = MemoryConfig {
        promotion_energy_threshold: 2.0,
        promotion_min_age,
        demotion_energy_threshold: 0.1,
        demotion_max_idle,
        sweep_interval: Duration::from_millis(50),
    };
    (memory_store, energy_store, config)
}

// ---------------------------------------------------------------------------
// sweep_at: Operational → Consolidated promotion
// ---------------------------------------------------------------------------

#[test]
fn sweep_at_operational_to_consolidated() {
    let (mem, energy, config) = make_test_setup(Duration::from_secs(60), Duration::from_secs(300));

    let node = NodeId(10);
    mem.track(node);
    energy.boost(node, 5.0); // above promotion threshold of 2.0

    let manager = MemoryManager::new(mem.clone(), config, energy);

    // Sweep at a time well past promotion_min_age
    let future = Instant::now() + Duration::from_secs(120);
    let result = manager.sweep_at(future);

    assert!(result.promoted.contains(&node));
    assert_eq!(mem.get_horizon(node), MemoryHorizon::Consolidated);
    assert_eq!(result.total_transitions(), 1);
}

#[test]
fn sweep_at_operational_not_promoted_if_too_young() {
    let (mem, energy, config) = make_test_setup(
        Duration::from_secs(3600), // 1 hour min age
        Duration::from_secs(86400),
    );

    let node = NodeId(11);
    mem.track(node);
    energy.boost(node, 10.0); // energy is high, but node is too young

    let manager = MemoryManager::new(mem.clone(), config, energy);
    let result = manager.sweep_at(Instant::now() + Duration::from_secs(5));

    assert!(result.promoted.is_empty());
    assert_eq!(mem.get_horizon(node), MemoryHorizon::Operational);
}

// ---------------------------------------------------------------------------
// sweep_at: Consolidated → Archived demotion
// ---------------------------------------------------------------------------

#[test]
fn sweep_at_consolidated_to_archived() {
    let (mem, energy, config) = make_test_setup(Duration::from_secs(60), Duration::from_secs(300));

    let node = NodeId(20);
    mem.track(node);
    mem.set_horizon(node, MemoryHorizon::Consolidated);
    // energy is 0.0 (never boosted) → below demotion_energy_threshold of 0.1

    let manager = MemoryManager::new(mem.clone(), config, energy);

    // Sweep far enough that time_in_horizon > demotion_max_idle (300s)
    let future = Instant::now() + Duration::from_secs(600);
    let result = manager.sweep_at(future);

    assert!(result.demoted.contains(&node));
    assert_eq!(mem.get_horizon(node), MemoryHorizon::Archived);
}

#[test]
fn sweep_at_consolidated_not_demoted_if_recent() {
    let (mem, energy, config) = make_test_setup(Duration::from_secs(60), Duration::from_secs(300));

    let node = NodeId(21);
    mem.track(node);
    mem.set_horizon(node, MemoryHorizon::Consolidated);
    // energy is 0, but time_in_horizon is short

    let manager = MemoryManager::new(mem.clone(), config, energy);
    // Only 10 seconds have passed — below demotion_max_idle of 300s
    let result = manager.sweep_at(Instant::now() + Duration::from_secs(10));

    assert!(result.demoted.is_empty());
    assert_eq!(mem.get_horizon(node), MemoryHorizon::Consolidated);
}

// ---------------------------------------------------------------------------
// sweep_at: Emergency Operational → Archived
// ---------------------------------------------------------------------------

#[test]
fn sweep_at_emergency_operational_to_archived() {
    let (mem, energy, config) = make_test_setup(Duration::from_secs(60), Duration::from_secs(300));

    let node = NodeId(30);
    mem.track(node);
    // energy is 0.0, which is < 0.1 (demotion_energy_threshold)

    let manager = MemoryManager::new(mem.clone(), config, energy);

    // age > demotion_max_idle (300s) → emergency demotion
    let future = Instant::now() + Duration::from_secs(600);
    let result = manager.sweep_at(future);

    assert!(result.demoted.contains(&node));
    assert_eq!(mem.get_horizon(node), MemoryHorizon::Archived);
}

#[test]
fn sweep_at_operational_stays_if_energy_between_thresholds() {
    let (mem, energy, config) = make_test_setup(Duration::from_secs(60), Duration::from_secs(300));

    let node = NodeId(31);
    mem.track(node);
    // Energy between demotion (0.1) and promotion (2.0) thresholds
    energy.boost(node, 0.5);

    let manager = MemoryManager::new(mem.clone(), config, energy);
    let future = Instant::now() + Duration::from_secs(600);
    let result = manager.sweep_at(future);

    // Neither promoted nor demoted
    assert!(result.promoted.is_empty());
    assert!(result.demoted.is_empty());
    assert_eq!(mem.get_horizon(node), MemoryHorizon::Operational);
}

// ---------------------------------------------------------------------------
// sweep_at: Archived → Operational reactivation
// ---------------------------------------------------------------------------

#[test]
fn sweep_at_archived_to_operational() {
    let (mem, energy, config) = make_test_setup(Duration::from_secs(60), Duration::from_secs(300));

    let node = NodeId(40);
    mem.track(node);
    mem.set_horizon(node, MemoryHorizon::Archived);
    energy.boost(node, 5.0); // above promotion threshold

    let manager = MemoryManager::new(mem.clone(), config, energy);
    let result = manager.sweep_at(Instant::now());

    assert!(result.reactivated.contains(&node));
    assert_eq!(mem.get_horizon(node), MemoryHorizon::Operational);
}

#[test]
fn sweep_at_archived_stays_if_low_energy() {
    let (mem, energy, config) = make_test_setup(Duration::from_secs(60), Duration::from_secs(300));

    let node = NodeId(41);
    mem.track(node);
    mem.set_horizon(node, MemoryHorizon::Archived);
    // energy is 0.0 → below promotion threshold

    let manager = MemoryManager::new(mem.clone(), config, energy);
    let result = manager.sweep_at(Instant::now());

    assert!(result.reactivated.is_empty());
    assert_eq!(mem.get_horizon(node), MemoryHorizon::Archived);
}

// ---------------------------------------------------------------------------
// sweep_at: Mixed transitions in a single sweep
// ---------------------------------------------------------------------------

#[test]
fn sweep_at_multiple_transitions_in_one_sweep() {
    let (mem, energy, config) = make_test_setup(Duration::from_secs(60), Duration::from_secs(300));

    // Node A: Operational, high energy, old → promote
    let a = NodeId(50);
    mem.track(a);
    energy.boost(a, 5.0);

    // Node B: Consolidated, zero energy, idle → demote
    let b = NodeId(51);
    mem.track(b);
    mem.set_horizon(b, MemoryHorizon::Consolidated);

    // Node C: Archived, high energy → reactivate
    let c = NodeId(52);
    mem.track(c);
    mem.set_horizon(c, MemoryHorizon::Archived);
    energy.boost(c, 3.0);

    // Node D: Operational, zero energy, old → emergency demote
    let d = NodeId(53);
    mem.track(d);

    let manager = MemoryManager::new(mem.clone(), config, energy);
    let future = Instant::now() + Duration::from_secs(600);
    let result = manager.sweep_at(future);

    assert!(result.promoted.contains(&a));
    assert!(result.demoted.contains(&b));
    assert!(result.reactivated.contains(&c));
    assert!(result.demoted.contains(&d)); // emergency demotion
    assert_eq!(result.total_transitions(), 4);
}

// ---------------------------------------------------------------------------
// MemoryManager::with_archive — fluent API
// ---------------------------------------------------------------------------

#[test]
fn with_archive_fluent_api() {
    let (mem, energy, config) = make_test_setup(Duration::from_secs(60), Duration::from_secs(300));

    let archive = Arc::new(InMemoryArchiveBackend::new());
    let manager =
        MemoryManager::new(mem, config, energy).with_archive(archive as Arc<dyn ArchiveBackend>);

    let dbg = format!("{:?}", manager);
    assert!(dbg.contains("has_archive: true"));
}

#[test]
fn without_archive_shows_false() {
    let (mem, energy, config) = make_test_setup(Duration::from_secs(60), Duration::from_secs(300));

    let manager = MemoryManager::new(mem, config, energy);
    let dbg = format!("{:?}", manager);
    assert!(dbg.contains("has_archive: false"));
}

// ---------------------------------------------------------------------------
// MemoryManager::start_periodic + shutdown
// ---------------------------------------------------------------------------

#[tokio::test]
async fn start_periodic_and_shutdown() {
    let (mem, energy, config) = make_test_setup(Duration::from_secs(60), Duration::from_secs(300));

    // Track a node that will be promoted when sweep runs
    let node = NodeId(60);
    mem.track(node);
    energy.boost(node, 5.0);

    // Override sweep_interval to be very fast
    let config = MemoryConfig {
        sweep_interval: Duration::from_millis(30),
        promotion_min_age: Duration::from_millis(1), // very short so promotion triggers
        ..config
    };

    let manager = Arc::new(MemoryManager::new(mem.clone(), config, energy));
    let handle = Arc::clone(&manager).start_periodic();

    // Let a few ticks run
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Shutdown
    manager.shutdown();

    // Wait for the task to finish (with timeout so test doesn't hang)
    let join_result = tokio::time::timeout(Duration::from_secs(2), handle).await;
    assert!(
        join_result.is_ok(),
        "periodic task should finish after shutdown"
    );
}

#[tokio::test]
async fn shutdown_without_start_does_not_panic() {
    let (mem, energy, config) = make_test_setup(Duration::from_secs(60), Duration::from_secs(300));

    let manager = MemoryManager::new(mem, config, energy);
    // Calling shutdown without having started the periodic task
    manager.shutdown();
}

#[tokio::test]
async fn start_periodic_performs_sweep() {
    let (mem, energy, config) = make_test_setup(
        Duration::from_millis(1), // very short min age
        Duration::from_secs(300),
    );

    let node = NodeId(70);
    mem.track(node);
    energy.boost(node, 5.0); // above promotion threshold

    let config = MemoryConfig {
        sweep_interval: Duration::from_millis(20),
        ..config
    };

    let manager = Arc::new(MemoryManager::new(mem.clone(), config, energy));
    let handle = Arc::clone(&manager).start_periodic();

    // Wait enough time for at least one sweep to have run
    tokio::time::sleep(Duration::from_millis(200)).await;

    manager.shutdown();
    let _ = tokio::time::timeout(Duration::from_secs(2), handle).await;

    // Node should have been promoted by the periodic sweep
    assert_eq!(
        mem.get_horizon(node),
        MemoryHorizon::Consolidated,
        "periodic sweep should have promoted the node"
    );
}
