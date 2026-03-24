//! Tests for the Scar System — error memory with exponential decay and healing.

#![cfg(feature = "scar")]

#[cfg(feature = "fabric")]
use grafeo_cognitive::fabric::{FabricScore, FabricStore};
use grafeo_cognitive::scar::{Scar, ScarConfig, ScarId, ScarReason, ScarStore};
use grafeo_common::types::NodeId;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Scar struct tests
// ---------------------------------------------------------------------------

#[test]
fn scar_new_initialises_correctly() {
    let target = NodeId(42);
    let half_life = Duration::from_secs(7 * 24 * 3600); // 7 days
    let scar = Scar::new_at(target, 1.0, ScarReason::Rollback, half_life, Instant::now());

    assert_eq!(scar.target, target);
    assert!((scar.raw_intensity() - 1.0).abs() < f64::EPSILON);
    assert!(scar.healed_at.is_none());
    assert!(!scar.is_healed());
}

#[test]
fn scar_negative_intensity_clamped_to_zero() {
    let scar = Scar::new_at(
        NodeId(1),
        -5.0,
        ScarReason::Custom("test".into()),
        Duration::from_secs(3600),
        Instant::now(),
    );
    assert!((scar.raw_intensity() - 0.0).abs() < f64::EPSILON);
}

#[test]
fn scar_current_intensity_at_creation_is_initial() {
    let now = Instant::now();
    let scar = Scar::new_at(
        NodeId(1),
        1.0,
        ScarReason::Rollback,
        Duration::from_secs(3600),
        now,
    );
    let intensity = scar.intensity_at(now);
    assert!((intensity - 1.0).abs() < 1e-10);
}

#[test]
fn scar_decay_after_one_half_life() {
    let half_life = Duration::from_secs(7 * 24 * 3600); // 7 days
    let now = Instant::now();
    let scar = Scar::new_at(NodeId(1), 1.0, ScarReason::Rollback, half_life, now);

    // After exactly one half-life, intensity should be ~0.5
    let after_half_life = now + half_life;
    let intensity = scar.intensity_at(after_half_life);
    assert!(
        (intensity - 0.5).abs() < 1e-10,
        "Expected ~0.5 after one half-life, got {}",
        intensity
    );
}

#[test]
fn scar_decay_after_two_half_lives() {
    let half_life = Duration::from_secs(3600);
    let now = Instant::now();
    let scar = Scar::new_at(NodeId(1), 1.0, ScarReason::Invalidation, half_life, now);

    let after_two = now + half_life * 2;
    let intensity = scar.intensity_at(after_two);
    assert!(
        (intensity - 0.25).abs() < 1e-10,
        "Expected ~0.25 after two half-lives, got {}",
        intensity
    );
}

#[test]
fn scar_heal_sets_intensity_to_zero() {
    let now = Instant::now();
    let mut scar = Scar::new_at(
        NodeId(1),
        1.0,
        ScarReason::Rollback,
        Duration::from_secs(3600),
        now,
    );

    assert!(!scar.is_healed());
    scar.heal_at(now + Duration::from_secs(10));
    assert!(scar.is_healed());
    assert!(scar.healed_at.is_some());

    // After healing, intensity is always 0 regardless of time
    assert!((scar.intensity_at(now + Duration::from_secs(10)) - 0.0).abs() < f64::EPSILON);
    assert!((scar.intensity_at(now + Duration::from_secs(100)) - 0.0).abs() < f64::EPSILON);
}

#[test]
fn scar_is_active_respects_threshold() {
    let now = Instant::now();
    let half_life = Duration::from_secs(3600);
    let scar = Scar::new_at(NodeId(1), 0.5, ScarReason::Rollback, half_life, now);

    // At creation: intensity = 0.5, threshold 0.4 → active
    assert!(scar.is_active(0.4));
    // At creation: intensity = 0.5, threshold 0.6 → NOT active
    assert!(!scar.is_active(0.6));
}

#[test]
fn scar_reason_display() {
    assert_eq!(ScarReason::Rollback.to_string(), "rollback");
    assert_eq!(
        ScarReason::Error("timeout".into()).to_string(),
        "error: timeout"
    );
    assert_eq!(ScarReason::Invalidation.to_string(), "invalidation");
    assert_eq!(
        ScarReason::ConstraintViolation("unique key".into()).to_string(),
        "constraint violation: unique key"
    );
    assert_eq!(ScarReason::Custom("oops".into()).to_string(), "oops");
}

// ---------------------------------------------------------------------------
// ScarStore tests
// ---------------------------------------------------------------------------

#[test]
fn store_add_and_get_scars() {
    let store = ScarStore::new(ScarConfig::default());
    let node = NodeId(10);

    let id1 = store.add_scar(node, 1.0, ScarReason::Rollback);
    let id2 = store.add_scar(node, 0.5, ScarReason::Error("test".into()));

    let scars = store.get_scars(node);
    assert_eq!(scars.len(), 2);
    assert!(scars.iter().any(|s| s.id == id1));
    assert!(scars.iter().any(|s| s.id == id2));
}

#[test]
fn store_get_scars_empty_node() {
    let store = ScarStore::new(ScarConfig::default());
    let scars = store.get_scars(NodeId(999));
    assert!(scars.is_empty());
}

#[test]
fn store_heal_by_id() {
    let store = ScarStore::new(ScarConfig::default());
    let node = NodeId(10);
    let scar_id = store.add_scar(node, 1.0, ScarReason::Rollback);

    assert!(store.heal(scar_id));
    let scars = store.get_scars(node);
    assert_eq!(scars.len(), 1);
    assert!(scars[0].is_healed());
}

#[test]
fn store_heal_nonexistent_returns_false() {
    let store = ScarStore::new(ScarConfig::default());
    assert!(!store.heal(ScarId(999)));
}

#[test]
fn store_get_active_scars_filters_healed() {
    let store = ScarStore::new(ScarConfig::default());
    let node = NodeId(10);

    let id1 = store.add_scar(node, 1.0, ScarReason::Rollback);
    let _id2 = store.add_scar(node, 0.8, ScarReason::Invalidation);

    // Heal one
    store.heal(id1);

    let active = store.get_active_scars(node);
    assert_eq!(active.len(), 1);
    assert!(!active[0].is_healed());
}

#[test]
fn store_cumulative_intensity() {
    let store = ScarStore::new(ScarConfig::default());
    let node = NodeId(10);

    store.add_scar(node, 1.0, ScarReason::Rollback);
    store.add_scar(node, 0.5, ScarReason::Invalidation);

    let cumulative = store.cumulative_intensity(node);
    assert!(
        (cumulative - 1.5).abs() < 0.01,
        "Expected ~1.5, got {}",
        cumulative
    );
}

#[test]
fn store_cumulative_intensity_excludes_healed() {
    let store = ScarStore::new(ScarConfig::default());
    let node = NodeId(10);

    let id1 = store.add_scar(node, 1.0, ScarReason::Rollback);
    store.add_scar(node, 0.5, ScarReason::Invalidation);

    store.heal(id1);
    let cumulative = store.cumulative_intensity(node);
    assert!(
        (cumulative - 0.5).abs() < 0.01,
        "Expected ~0.5, got {}",
        cumulative
    );
}

#[test]
fn store_nodes_with_active_scars() {
    let store = ScarStore::new(ScarConfig::default());

    store.add_scar(NodeId(1), 1.0, ScarReason::Rollback);
    store.add_scar(NodeId(2), 0.5, ScarReason::Invalidation);
    store.add_scar(NodeId(3), 0.001, ScarReason::Custom("tiny".into()));

    // min_intensity = 0.01 → node 3 should be excluded (0.001 < 0.01)
    let active = store.nodes_with_active_scars(0.01);
    assert_eq!(active.len(), 2);
    assert!(active.iter().any(|(id, _)| *id == NodeId(1)));
    assert!(active.iter().any(|(id, _)| *id == NodeId(2)));
}

#[test]
fn store_prune_removes_inactive() {
    let config = ScarConfig {
        min_intensity: 0.5,
        ..ScarConfig::default()
    };
    let store = ScarStore::new(config);
    let node = NodeId(10);

    // Add a scar that's below threshold
    store.add_scar(node, 0.1, ScarReason::Custom("weak".into()));
    // Add one above threshold
    store.add_scar(node, 1.0, ScarReason::Rollback);

    assert_eq!(store.total_scars(), 2);
    let pruned = store.prune();
    assert_eq!(pruned, 1);
    assert_eq!(store.total_scars(), 1);
}

#[test]
fn store_active_scar_count() {
    let store = ScarStore::new(ScarConfig::default());

    store.add_scar(NodeId(1), 1.0, ScarReason::Rollback);
    store.add_scar(NodeId(2), 1.0, ScarReason::Rollback);
    let id3 = store.add_scar(NodeId(3), 1.0, ScarReason::Rollback);

    store.heal(id3);
    assert_eq!(store.active_scar_count(), 2);
}

#[test]
fn store_is_thread_safe() {
    use std::sync::Arc;
    use std::thread;

    let store = Arc::new(ScarStore::new(ScarConfig::default()));
    let mut handles = vec![];

    for i in 0..10 {
        let store = Arc::clone(&store);
        handles.push(thread::spawn(move || {
            store.add_scar(NodeId(i), 1.0, ScarReason::Rollback);
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(store.total_scars(), 10);
}

// ---------------------------------------------------------------------------
// FabricScore scar integration tests
// ---------------------------------------------------------------------------

#[test]
#[cfg(feature = "fabric")]
fn fabric_scar_intensity_field_default() {
    let score = FabricScore::new();
    assert!((score.scar_intensity - 0.0).abs() < f64::EPSILON);
}

#[test]
#[cfg(feature = "fabric")]
fn fabric_set_scar_intensity() {
    let store = FabricStore::new();
    let node = NodeId(1);

    store.set_scar_intensity(node, 2.5);
    let score = store.get_fabric_score(node);
    assert!((score.scar_intensity - 2.5).abs() < f64::EPSILON);
}

#[test]
#[cfg(feature = "fabric")]
fn fabric_risk_score_increases_with_scar() {
    let store = FabricStore::new();
    let node_clean = NodeId(1);
    let node_scarred = NodeId(2);

    // Set modest metrics so base risk is well below 1.0
    store.update_churn(node_clean);
    store.update_churn(node_scarred);
    store.set_gds_metrics(node_clean, 0.3, 0.3, None);
    store.set_gds_metrics(node_scarred, 0.3, 0.3, None);
    // Set knowledge_density to 0.5 to reduce base risk further
    store.set_knowledge_density(node_clean, 0.5);
    store.set_knowledge_density(node_scarred, 0.5);

    // Add scar intensity only to node_scarred
    store.set_scar_intensity(node_scarred, 1.0);

    store.recalculate_all_risks();

    let risk_clean = store.get_fabric_score(node_clean).risk_score;
    let risk_scarred = store.get_fabric_score(node_scarred).risk_score;

    assert!(
        risk_scarred > risk_clean,
        "Scarred node risk ({}) should be higher than clean node risk ({})",
        risk_scarred,
        risk_clean
    );
}

#[test]
#[cfg(feature = "fabric")]
fn fabric_risk_score_proportional_to_scar_intensity() {
    let store = FabricStore::new();
    let low_scar = NodeId(1);
    let high_scar = NodeId(2);

    // Both nodes have zero base metrics → base risk = 0
    // Only scar intensity contributes
    store.set_scar_intensity(low_scar, 0.5);
    store.set_scar_intensity(high_scar, 2.0);

    store.recalculate_all_risks();

    let risk_low = store.get_fabric_score(low_scar).risk_score;
    let risk_high = store.get_fabric_score(high_scar).risk_score;

    assert!(
        risk_high > risk_low,
        "Higher scar intensity ({}) should yield higher risk ({}) vs ({}) for low ({})",
        2.0,
        risk_high,
        risk_low,
        0.5
    );
}

#[test]
#[cfg(feature = "fabric")]
fn fabric_risk_score_clamped_to_one() {
    let store = FabricStore::new();
    let node = NodeId(1);

    // Max out everything
    store.update_churn(node);
    store.set_gds_metrics(node, 100.0, 100.0, None);
    store.set_scar_intensity(node, 100.0);

    store.recalculate_all_risks();

    let risk = store.get_fabric_score(node).risk_score;
    assert!(
        risk <= 1.0,
        "Risk score should be clamped to 1.0, got {}",
        risk
    );
}
