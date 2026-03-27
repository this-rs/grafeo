//! Additional coverage tests for the scar subsystem.

#![cfg(feature = "scar")]

use grafeo_cognitive::scar::{Scar, ScarConfig, ScarId, ScarReason, ScarStore};
use grafeo_common::types::NodeId;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// ScarId Display
// ---------------------------------------------------------------------------

#[test]
fn scar_id_display() {
    let id = ScarId(42);
    assert_eq!(id.to_string(), "scar:42");
}

#[test]
fn scar_id_debug() {
    let id = ScarId(7);
    let dbg = format!("{:?}", id);
    assert!(dbg.contains('7'), "got: {dbg}");
}

#[test]
fn scar_id_equality() {
    assert_eq!(ScarId(1), ScarId(1));
    assert_ne!(ScarId(1), ScarId(2));
}

#[test]
fn scar_id_hash() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(ScarId(1));
    set.insert(ScarId(1));
    set.insert(ScarId(2));
    assert_eq!(set.len(), 2);
}

// ---------------------------------------------------------------------------
// ScarReason equality and clone
// ---------------------------------------------------------------------------

#[test]
fn scar_reason_equality() {
    assert_eq!(ScarReason::Rollback, ScarReason::Rollback);
    assert_eq!(ScarReason::Invalidation, ScarReason::Invalidation);
    assert_eq!(ScarReason::Error("a".into()), ScarReason::Error("a".into()));
    assert_ne!(ScarReason::Error("a".into()), ScarReason::Error("b".into()));
    assert_ne!(ScarReason::Rollback, ScarReason::Invalidation);
}

#[test]
fn scar_reason_clone() {
    let reason = ScarReason::ConstraintViolation("unique".into());
    let cloned = reason.clone();
    assert_eq!(reason, cloned);
}

// ---------------------------------------------------------------------------
// Scar: heal (non-_at variant)
// ---------------------------------------------------------------------------

#[test]
fn scar_heal_non_at_variant() {
    let now = Instant::now();
    let mut scar = Scar::new_at(
        NodeId(1),
        1.0,
        ScarReason::Rollback,
        Duration::from_secs(3600),
        now,
    );
    assert!(!scar.is_healed());
    scar.heal();
    assert!(scar.is_healed());
    assert!(scar.healed_at.is_some());
    // After healing, current_intensity is 0
    assert_eq!(scar.current_intensity(), 0.0);
}

// ---------------------------------------------------------------------------
// Scar: is_active after healing
// ---------------------------------------------------------------------------

#[test]
fn scar_is_active_false_when_healed() {
    let now = Instant::now();
    let mut scar = Scar::new_at(
        NodeId(1),
        1.0,
        ScarReason::Rollback,
        Duration::from_secs(3600),
        now,
    );
    scar.heal_at(now + Duration::from_secs(1));
    assert!(!scar.is_active(0.0)); // Even with threshold 0, healed scar is inactive
}

// ---------------------------------------------------------------------------
// ScarStore: default impl
// ---------------------------------------------------------------------------

#[test]
fn scar_store_default() {
    let store = ScarStore::default();
    assert_eq!(store.total_scars(), 0);
    assert_eq!(store.active_scar_count(), 0);
}

// ---------------------------------------------------------------------------
// ScarStore: max_scars_per_node pruning
// ---------------------------------------------------------------------------

#[test]
fn store_max_scars_per_node_pruning() {
    let config = ScarConfig {
        max_scars_per_node: 3,
        min_intensity: 0.001,
        ..ScarConfig::default()
    };
    let store = ScarStore::new(config);
    let node = NodeId(1);

    // Add 5 scars (exceeds max of 3)
    for i in 0..5 {
        store.add_scar(node, (i as f64) + 1.0, ScarReason::Rollback);
    }

    // Should be truncated to 3 (kept the highest intensity)
    let scars = store.get_scars(node);
    assert!(
        scars.len() <= 3,
        "expected at most 3 scars, got {}",
        scars.len()
    );
}

// ---------------------------------------------------------------------------
// ScarStore: cumulative_intensity for untracked node
// ---------------------------------------------------------------------------

#[test]
fn store_cumulative_intensity_untracked() {
    let store = ScarStore::new(ScarConfig::default());
    assert_eq!(store.cumulative_intensity(NodeId(999)), 0.0);
}

// ---------------------------------------------------------------------------
// ScarStore: nodes_with_active_scars empty
// ---------------------------------------------------------------------------

#[test]
fn store_nodes_with_active_scars_empty() {
    let store = ScarStore::new(ScarConfig::default());
    let result = store.nodes_with_active_scars(0.0);
    assert!(result.is_empty());
}

// ---------------------------------------------------------------------------
// ScarStore: prune empty store
// ---------------------------------------------------------------------------

#[test]
fn store_prune_empty_returns_zero() {
    let store = ScarStore::new(ScarConfig::default());
    assert_eq!(store.prune(), 0);
}

// ---------------------------------------------------------------------------
// ScarStore: prune removes healed scars
// ---------------------------------------------------------------------------

#[test]
fn store_prune_removes_healed() {
    let store = ScarStore::new(ScarConfig::default());
    let node = NodeId(1);

    let id1 = store.add_scar(node, 1.0, ScarReason::Rollback);
    store.add_scar(node, 1.0, ScarReason::Invalidation);

    store.heal(id1);
    assert_eq!(store.total_scars(), 2);

    let pruned = store.prune();
    assert_eq!(pruned, 1);
    assert_eq!(store.total_scars(), 1);
}

// ---------------------------------------------------------------------------
// ScarStore: config() accessor
// ---------------------------------------------------------------------------

#[test]
fn store_config_accessor() {
    let config = ScarConfig {
        min_intensity: 0.05,
        max_scars_per_node: 10,
        ..ScarConfig::default()
    };
    let store = ScarStore::new(config);
    assert_eq!(store.config().min_intensity, 0.05);
    assert_eq!(store.config().max_scars_per_node, 10);
}

// ---------------------------------------------------------------------------
// ScarConfig: default values
// ---------------------------------------------------------------------------

#[test]
fn scar_config_default_values() {
    let config = ScarConfig::default();
    assert_eq!(
        config.default_half_life,
        Duration::from_secs(30 * 24 * 3600)
    );
    assert_eq!(config.min_intensity, 0.01);
    assert_eq!(config.max_scars_per_node, 50);
}

// ---------------------------------------------------------------------------
// ScarStore: Debug impl
// ---------------------------------------------------------------------------

#[test]
fn scar_store_debug() {
    let store = ScarStore::new(ScarConfig::default());
    store.add_scar(NodeId(1), 1.0, ScarReason::Rollback);
    let dbg = format!("{:?}", store);
    assert!(dbg.contains("ScarStore"), "got: {dbg}");
    assert!(dbg.contains("total_scars"), "got: {dbg}");
}

// ---------------------------------------------------------------------------
// Scar: current_intensity vs intensity_at consistency
// ---------------------------------------------------------------------------

#[test]
fn scar_current_intensity_near_creation() {
    let scar = Scar::new_at(
        NodeId(1),
        2.0,
        ScarReason::Error("test".into()),
        Duration::from_secs(3600),
        Instant::now(),
    );
    let ci = scar.current_intensity();
    assert!((ci - 2.0).abs() < 0.01);
}

// ---------------------------------------------------------------------------
// ScarStore: multiple nodes with scars
// ---------------------------------------------------------------------------

#[test]
fn store_multiple_nodes() {
    let store = ScarStore::new(ScarConfig::default());

    store.add_scar(NodeId(1), 1.0, ScarReason::Rollback);
    store.add_scar(NodeId(2), 0.5, ScarReason::Invalidation);
    store.add_scar(NodeId(3), 2.0, ScarReason::Custom("test".into()));

    assert_eq!(store.total_scars(), 3);

    let scars1 = store.get_scars(NodeId(1));
    assert_eq!(scars1.len(), 1);

    let scars2 = store.get_scars(NodeId(2));
    assert_eq!(scars2.len(), 1);

    let active = store.nodes_with_active_scars(0.01);
    assert_eq!(active.len(), 3);
}
