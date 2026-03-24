//! Additional coverage tests for the stagnation detection subsystem.

#![cfg(all(feature = "stagnation", feature = "energy"))]

use grafeo_cognitive::energy::{EnergyConfig, EnergyStore};
use grafeo_cognitive::stagnation::{
    StagnationConfig, StagnationDetector, StagnationScore, StagnationStore, Trend,
};
use grafeo_common::types::NodeId;
use std::sync::Arc;
use std::time::Duration;

fn default_config() -> StagnationConfig {
    StagnationConfig::default()
}

fn make_energy_store(ids_and_energy: &[(u64, f64)]) -> Arc<EnergyStore> {
    let store = Arc::new(EnergyStore::new(EnergyConfig {
        default_half_life: Duration::from_secs(3600 * 24),
        ..EnergyConfig::default()
    }));
    for &(id, energy) in ids_and_energy {
        store.boost(NodeId::new(id), energy);
    }
    store
}

#[cfg(feature = "synapse")]
fn make_synapse_store(pairs: &[(u64, u64, f64)]) -> Arc<grafeo_cognitive::synapse::SynapseStore> {
    let store = Arc::new(grafeo_cognitive::synapse::SynapseStore::new(
        grafeo_cognitive::synapse::SynapseConfig {
            initial_weight: 0.0,
            reinforce_amount: 0.0,
            default_half_life: Duration::from_secs(7 * 24 * 3600),
            min_weight: 0.001,
            max_synapse_weight: 10.0,
            max_total_outgoing_weight: 100.0,
        },
    ));
    for &(a, b, weight) in pairs {
        store.reinforce(NodeId::new(a), NodeId::new(b), weight);
    }
    store
}

// ---------------------------------------------------------------------------
// StagnationConfig default values
// ---------------------------------------------------------------------------

#[test]
fn stagnation_config_default_values() {
    let config = StagnationConfig::default();
    assert_eq!(config.weight_energy, 0.4);
    assert_eq!(config.weight_mutation_age, 0.35);
    assert_eq!(config.weight_synapse_activity, 0.25);
    assert_eq!(config.max_mutation_age, Duration::from_secs(30 * 24 * 3600));
    assert_eq!(config.stagnation_threshold, 0.7);
    assert_eq!(
        config.synapse_recent_window,
        Duration::from_secs(7 * 24 * 3600)
    );
    assert_eq!(config.trend_window_size, 5);
    assert_eq!(config.trend_tolerance, 0.05);
    assert_eq!(config.scan_interval, Duration::from_secs(3600));
}

// ---------------------------------------------------------------------------
// Trend enum: Debug, Clone, Copy, PartialEq, Eq
// ---------------------------------------------------------------------------

#[test]
fn trend_debug() {
    assert_eq!(format!("{:?}", Trend::Improving), "Improving");
    assert_eq!(format!("{:?}", Trend::Degrading), "Degrading");
    assert_eq!(format!("{:?}", Trend::Stable), "Stable");
}

#[test]
fn trend_clone_and_copy() {
    let t = Trend::Improving;
    let cloned = t;
    assert_eq!(t, cloned);
}

#[test]
fn trend_equality() {
    assert_eq!(Trend::Stable, Trend::Stable);
    assert_ne!(Trend::Stable, Trend::Degrading);
    assert_ne!(Trend::Improving, Trend::Degrading);
}

// ---------------------------------------------------------------------------
// StagnationScore: Clone and Debug
// ---------------------------------------------------------------------------

#[test]
fn stagnation_score_clone_and_debug() {
    let score = StagnationScore {
        community_id: 5,
        avg_energy: 0.5,
        last_mutation_age: Duration::from_secs(100),
        synapse_activity: 0.3,
        stagnation_score: 0.4,
        trend: Trend::Stable,
    };
    let cloned = score.clone();
    assert_eq!(cloned.community_id, 5);
    assert_eq!(cloned.trend, Trend::Stable);

    let dbg = format!("{:?}", score);
    assert!(dbg.contains("StagnationScore"));
}

// ---------------------------------------------------------------------------
// StagnationDetector: analyze_community with single node (no synapse pairs)
// ---------------------------------------------------------------------------

#[test]
fn analyze_community_single_node() {
    let config = default_config();
    let stag_store = Arc::new(StagnationStore::new(config.clone()));
    let energy_store = make_energy_store(&[(1, 0.5)]);

    #[cfg(feature = "synapse")]
    let synapse_store = make_synapse_store(&[]);

    let detector = StagnationDetector::new(
        energy_store,
        #[cfg(feature = "synapse")]
        synapse_store,
        stag_store,
        config,
    );

    let node_ids = vec![NodeId::new(1)];
    let score = detector.analyze_community(1, &node_ids, Duration::from_secs(60));
    assert!(score.stagnation_score >= 0.0 && score.stagnation_score <= 1.0);
    // Single node can't have synapse pairs, so synapse_activity = 0
    assert_eq!(score.synapse_activity, 0.0);
}

// ---------------------------------------------------------------------------
// StagnationStore: multiple communities
// ---------------------------------------------------------------------------

#[test]
fn store_tracks_multiple_communities() {
    let store = StagnationStore::new(default_config());

    for id in 1..=5 {
        store.update(
            id,
            StagnationScore {
                community_id: id,
                avg_energy: 0.5,
                last_mutation_age: Duration::from_secs(100),
                synapse_activity: 0.3,
                stagnation_score: 0.4,
                trend: Trend::Stable,
            },
        );
    }

    assert_eq!(store.len(), 5);
    assert!(!store.is_empty());
}

// ---------------------------------------------------------------------------
// Trend detection: exactly at tolerance boundary
// ---------------------------------------------------------------------------

#[test]
fn trend_at_exact_tolerance_is_stable() {
    let config = StagnationConfig {
        trend_tolerance: 0.1,
        ..default_config()
    };
    let store = Arc::new(StagnationStore::new(config));

    store.update(
        1,
        StagnationScore {
            community_id: 1,
            avg_energy: 0.5,
            last_mutation_age: Duration::from_secs(100),
            synapse_activity: 0.3,
            stagnation_score: 0.5,
            trend: Trend::Stable,
        },
    );

    // Delta of exactly 0.1 = tolerance → Stable (not degrading, since
    // the check is > tolerance, not >=)
    store.update(
        1,
        StagnationScore {
            community_id: 1,
            avg_energy: 0.4,
            last_mutation_age: Duration::from_secs(200),
            synapse_activity: 0.2,
            stagnation_score: 0.6,
            trend: Trend::Stable,
        },
    );

    let vitality = store.get_community_vitality(1).unwrap();
    assert_eq!(vitality.trend, Trend::Stable);
}

// ---------------------------------------------------------------------------
// StagnationDetector: analyze with mutation age exceeding max
// ---------------------------------------------------------------------------

#[test]
fn analyze_community_mutation_age_exceeds_max() {
    let config = StagnationConfig {
        max_mutation_age: Duration::from_secs(100),
        ..default_config()
    };
    let stag_store = Arc::new(StagnationStore::new(config.clone()));
    let energy_store = make_energy_store(&[(1, 0.5)]);

    #[cfg(feature = "synapse")]
    let synapse_store = make_synapse_store(&[]);

    let detector = StagnationDetector::new(
        energy_store,
        #[cfg(feature = "synapse")]
        synapse_store,
        stag_store,
        config,
    );

    // Mutation age = 200 > max of 100 → clamped to 1.0
    let score = detector.analyze_community(1, &[NodeId::new(1)], Duration::from_secs(200));
    assert!(score.stagnation_score >= 0.0);
    assert!(score.stagnation_score <= 1.0);
}
