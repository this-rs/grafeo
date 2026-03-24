//! Additional coverage tests for the memory horizons subsystem.

#![cfg(all(feature = "memory", feature = "energy"))]

use grafeo_cognitive::memory::{
    InMemoryArchiveBackend, MemoryConfig, MemoryHorizon, MemoryManager, MemoryStore,
    NodeMemoryState, SweepResult,
};
use grafeo_cognitive::{EnergyConfig, EnergyStore};
use grafeo_common::types::NodeId;
use std::sync::Arc;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// NodeMemoryState::Default impl
// ---------------------------------------------------------------------------

#[test]
fn node_memory_state_default_is_operational() {
    let state = NodeMemoryState::default();
    assert_eq!(state.horizon, MemoryHorizon::Operational);
    assert_eq!(state.transition_count, 0);
}

// ---------------------------------------------------------------------------
// NodeMemoryState::transition_to_at
// ---------------------------------------------------------------------------

#[test]
fn node_memory_state_transition_to_at() {
    let start = Instant::now();
    let mut state = NodeMemoryState::new_at(start);

    let later = start + Duration::from_secs(100);
    state.transition_to_at(MemoryHorizon::Consolidated, later);

    assert_eq!(state.horizon, MemoryHorizon::Consolidated);
    assert_eq!(state.transition_count, 1);
    assert_eq!(state.entered_current_horizon, later);
}

#[test]
fn node_memory_state_transition_to_at_same_horizon_noop() {
    let start = Instant::now();
    let mut state = NodeMemoryState::new_at(start);

    let later = start + Duration::from_secs(100);
    state.transition_to_at(MemoryHorizon::Operational, later);

    // Same horizon → no transition
    assert_eq!(state.transition_count, 0);
    assert_eq!(state.entered_current_horizon, start);
}

// ---------------------------------------------------------------------------
// MemoryStore::get_state
// ---------------------------------------------------------------------------

#[test]
fn store_get_state_returns_some_for_tracked() {
    let store = MemoryStore::new();
    store.track(NodeId(1));
    let state = store.get_state(NodeId(1));
    assert!(state.is_some());
    assert_eq!(state.unwrap().horizon, MemoryHorizon::Operational);
}

#[test]
fn store_get_state_returns_none_for_untracked() {
    let store = MemoryStore::new();
    assert!(store.get_state(NodeId(999)).is_none());
}

// ---------------------------------------------------------------------------
// MemoryStore::set_horizon creates entry if absent
// ---------------------------------------------------------------------------

#[test]
fn store_set_horizon_creates_entry_if_absent() {
    let store = MemoryStore::new();
    assert!(store.is_empty());

    store.set_horizon(NodeId(1), MemoryHorizon::Archived);
    assert_eq!(store.len(), 1);
    assert_eq!(store.get_horizon(NodeId(1)), MemoryHorizon::Archived);
}

// ---------------------------------------------------------------------------
// MemoryStore::is_empty and len
// ---------------------------------------------------------------------------

#[test]
fn store_is_empty_and_len() {
    let store = MemoryStore::new();
    assert!(store.is_empty());
    assert_eq!(store.len(), 0);

    store.track(NodeId(1));
    store.track(NodeId(2));
    assert!(!store.is_empty());
    assert_eq!(store.len(), 2);
}

// ---------------------------------------------------------------------------
// MemoryManager::store() accessor
// ---------------------------------------------------------------------------

#[test]
fn memory_manager_store_accessor() {
    let energy_store = Arc::new(EnergyStore::new(EnergyConfig::default()));
    let memory_store = Arc::new(MemoryStore::new());
    let config = MemoryConfig::default();
    let manager = MemoryManager::new(memory_store.clone(), config, energy_store);
    assert!(Arc::ptr_eq(manager.store(), &memory_store));
}

// ---------------------------------------------------------------------------
// MemoryManager::config() accessor
// ---------------------------------------------------------------------------

#[test]
fn memory_manager_config_accessor() {
    let energy_store = Arc::new(EnergyStore::new(EnergyConfig::default()));
    let memory_store = Arc::new(MemoryStore::new());
    let config = MemoryConfig {
        promotion_energy_threshold: 5.0,
        ..Default::default()
    };
    let manager = MemoryManager::new(memory_store, config, energy_store);
    assert_eq!(manager.config().promotion_energy_threshold, 5.0);
}

// ---------------------------------------------------------------------------
// MemoryManager::with_archive and sweep
// ---------------------------------------------------------------------------

#[test]
fn memory_manager_with_archive_sets_backend() {
    let energy_store = Arc::new(EnergyStore::new(EnergyConfig::default()));
    let memory_store = Arc::new(MemoryStore::new());
    let config = MemoryConfig::default();
    let archive = Arc::new(InMemoryArchiveBackend::new());

    let manager = MemoryManager::new(memory_store, config, energy_store)
        .with_archive(archive as Arc<dyn grafeo_cognitive::memory::ArchiveBackend>);
    let dbg = format!("{:?}", manager);
    assert!(dbg.contains("has_archive: true"));
}

// ---------------------------------------------------------------------------
// SweepResult::default
// ---------------------------------------------------------------------------

#[test]
fn sweep_result_default_all_empty() {
    let result = SweepResult::default();
    assert!(result.promoted.is_empty());
    assert!(result.demoted.is_empty());
    assert!(result.reactivated.is_empty());
    assert_eq!(result.total_transitions(), 0);
}

// ---------------------------------------------------------------------------
// MemoryHorizon: Hash, Eq
// ---------------------------------------------------------------------------

#[test]
fn memory_horizon_hash_and_eq() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(MemoryHorizon::Operational);
    set.insert(MemoryHorizon::Operational);
    set.insert(MemoryHorizon::Consolidated);
    set.insert(MemoryHorizon::Archived);
    assert_eq!(set.len(), 3);
}

// ---------------------------------------------------------------------------
// MemoryHorizon: Clone and Copy
// ---------------------------------------------------------------------------

#[test]
fn memory_horizon_clone_copy() {
    let h = MemoryHorizon::Consolidated;
    let h2 = h;
    assert_eq!(h, h2);
}

// ---------------------------------------------------------------------------
// Sweep: no transitions when store is empty
// ---------------------------------------------------------------------------

#[test]
fn sweep_empty_store_no_transitions() {
    let energy_store = Arc::new(EnergyStore::new(EnergyConfig::default()));
    let memory_store = Arc::new(MemoryStore::new());
    let config = MemoryConfig::default();
    let manager = MemoryManager::new(memory_store, config, energy_store);
    let result = manager.sweep();
    assert_eq!(result.total_transitions(), 0);
}

// ---------------------------------------------------------------------------
// Sweep: consolidated node with energy above threshold stays consolidated
// ---------------------------------------------------------------------------

#[test]
fn sweep_consolidated_high_energy_stays() {
    let energy_store = Arc::new(EnergyStore::new(EnergyConfig::default()));
    let memory_store = Arc::new(MemoryStore::new());
    let config = MemoryConfig {
        demotion_energy_threshold: 0.1,
        demotion_max_idle: Duration::from_secs(300),
        ..Default::default()
    };

    let node = NodeId(1);
    memory_store.track(node);
    memory_store.set_horizon(node, MemoryHorizon::Consolidated);
    energy_store.boost(node, 5.0); // well above demotion threshold

    let manager = MemoryManager::new(memory_store.clone(), config, energy_store);
    let future = Instant::now() + Duration::from_secs(600);
    let result = manager.sweep_at(future);

    assert!(result.demoted.is_empty());
    assert_eq!(memory_store.get_horizon(node), MemoryHorizon::Consolidated);
}

// ---------------------------------------------------------------------------
// InMemoryArchiveBackend: len, is_empty
// ---------------------------------------------------------------------------

#[tokio::test]
async fn in_memory_archive_len_is_empty() {
    let backend = InMemoryArchiveBackend::new();
    assert_eq!(backend.len(), 0);
    assert!(backend.is_empty());

    use grafeo_cognitive::memory::ArchiveBackend;
    backend.archive(NodeId(1), b"data").await.unwrap();
    assert_eq!(backend.len(), 1);
    assert!(!backend.is_empty());
}

// ---------------------------------------------------------------------------
// InMemoryArchiveBackend: remove nonexistent is no-op
// ---------------------------------------------------------------------------

#[tokio::test]
async fn in_memory_archive_remove_nonexistent() {
    use grafeo_cognitive::memory::ArchiveBackend;
    let backend = InMemoryArchiveBackend::new();
    // Should not panic
    backend.remove(NodeId(999)).await.unwrap();
}

// ---------------------------------------------------------------------------
// InMemoryArchiveBackend: exists for nonexistent
// ---------------------------------------------------------------------------

#[tokio::test]
async fn in_memory_archive_exists_nonexistent() {
    use grafeo_cognitive::memory::ArchiveBackend;
    let backend = InMemoryArchiveBackend::new();
    assert!(!backend.exists(NodeId(999)).await.unwrap());
}
