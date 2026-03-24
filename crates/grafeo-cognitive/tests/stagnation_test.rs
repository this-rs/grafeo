//! Integration tests for the stagnation detection subsystem.

#![cfg(all(feature = "stagnation", feature = "energy"))]

use grafeo_cognitive::energy::{EnergyConfig, EnergyStore};
use grafeo_cognitive::stagnation::{
    StagnationConfig, StagnationDetector, StagnationScore, StagnationStore, Trend,
};
use grafeo_common::types::NodeId;
use std::sync::Arc;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn default_config() -> StagnationConfig {
    StagnationConfig {
        weight_energy: 0.4,
        weight_mutation_age: 0.35,
        weight_synapse_activity: 0.25,
        max_mutation_age: Duration::from_secs(7 * 24 * 3600),
        stagnation_threshold: 0.6,
        synapse_recent_window: Duration::from_secs(7 * 24 * 3600),
        trend_window_size: 5,
        trend_tolerance: 0.05,
        scan_interval: Duration::from_secs(60),
    }
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
fn make_synapse_store(
    pairs: &[(u64, u64, f64)],
) -> Arc<grafeo_cognitive::synapse::SynapseStore> {
    let store = Arc::new(grafeo_cognitive::synapse::SynapseStore::new(
        grafeo_cognitive::synapse::SynapseConfig {
            initial_weight: 0.0,
            reinforce_amount: 0.0,
            default_half_life: Duration::from_secs(7 * 24 * 3600),
            min_weight: 0.001,
        },
    ));
    for &(a, b, weight) in pairs {
        store.reinforce(NodeId::new(a), NodeId::new(b), weight);
    }
    store
}

// ---------------------------------------------------------------------------
// StagnationScore calculation
// ---------------------------------------------------------------------------

#[test]
fn score_high_energy_low_age_active_synapses_means_low_stagnation() {
    let config = default_config();
    let stag_store = Arc::new(StagnationStore::new(config.clone()));
    let energy_store = make_energy_store(&[(1, 0.9), (2, 0.8), (3, 0.85)]);

    #[cfg(feature = "synapse")]
    let synapse_store = make_synapse_store(&[(1, 2, 0.7), (2, 3, 0.6), (1, 3, 0.5)]);

    let detector = StagnationDetector::new(
        energy_store,
        #[cfg(feature = "synapse")]
        synapse_store,
        stag_store,
        config,
    );

    let node_ids = vec![NodeId::new(1), NodeId::new(2), NodeId::new(3)];
    let score = detector.analyze_community(1, &node_ids, Duration::from_secs(60));

    // High energy, recent mutations, active synapses → low stagnation
    assert!(
        score.stagnation_score < 0.5,
        "expected low stagnation, got {}",
        score.stagnation_score
    );
    assert!(score.avg_energy > 0.5);
}

#[test]
fn score_zero_energy_max_age_no_synapses_means_high_stagnation() {
    let config = default_config();
    let stag_store = Arc::new(StagnationStore::new(config.clone()));
    // No energy boosts — all nodes at 0.0
    let energy_store = Arc::new(EnergyStore::new(EnergyConfig::default()));

    #[cfg(feature = "synapse")]
    let synapse_store = make_synapse_store(&[]);

    let detector = StagnationDetector::new(
        energy_store,
        #[cfg(feature = "synapse")]
        synapse_store,
        stag_store,
        config.clone(),
    );

    let node_ids = vec![NodeId::new(10), NodeId::new(11)];
    let age = config.max_mutation_age; // maximum age → normalized to 1.0
    let score = detector.analyze_community(2, &node_ids, age);

    // (1 - 0) * 0.4 + 1.0 * 0.3 + (1 - 0) * 0.3 = 1.0
    assert!(
        (score.stagnation_score - 1.0).abs() < 1e-6,
        "expected ~1.0, got {}",
        score.stagnation_score
    );
}

#[test]
fn score_is_clamped_to_unit_interval() {
    let config = default_config();
    let stag_store = Arc::new(StagnationStore::new(config.clone()));
    let energy_store = make_energy_store(&[(1, 5.0)]); // energy > 1, clamped to 1.0

    #[cfg(feature = "synapse")]
    let synapse_store = make_synapse_store(&[]);

    let detector = StagnationDetector::new(
        energy_store,
        #[cfg(feature = "synapse")]
        synapse_store,
        stag_store,
        config,
    );

    let score = detector.analyze_community(3, &[NodeId::new(1)], Duration::ZERO);
    assert!(score.stagnation_score >= 0.0);
    assert!(score.stagnation_score <= 1.0);
}

// ---------------------------------------------------------------------------
// Trend detection
// ---------------------------------------------------------------------------

#[test]
fn trend_stable_then_degrading_then_improving() {
    let config = StagnationConfig {
        trend_tolerance: 0.05,
        ..default_config()
    };
    let store = Arc::new(StagnationStore::new(config));

    // First snapshot — no previous, so trend is Stable.
    let s1 = StagnationScore {
        community_id: 1,
        avg_energy: 0.5,
        last_mutation_age: Duration::from_secs(100),
        synapse_activity: 0.3,
        stagnation_score: 0.4,
        trend: Trend::Stable,
    };
    store.update(1, s1);
    let vitality = store.get_community_vitality(1).unwrap();
    assert_eq!(vitality.trend, Trend::Stable);

    // Second snapshot — score increases by 0.2 → Degrading.
    let s2 = StagnationScore {
        community_id: 1,
        avg_energy: 0.3,
        last_mutation_age: Duration::from_secs(500),
        synapse_activity: 0.1,
        stagnation_score: 0.6,
        trend: Trend::Stable,
    };
    store.update(1, s2);
    let vitality = store.get_community_vitality(1).unwrap();
    assert_eq!(vitality.trend, Trend::Degrading);

    // Third snapshot — score drops by 0.3 → Improving.
    let s3 = StagnationScore {
        community_id: 1,
        avg_energy: 0.8,
        last_mutation_age: Duration::from_secs(10),
        synapse_activity: 0.7,
        stagnation_score: 0.3,
        trend: Trend::Stable,
    };
    store.update(1, s3);
    let vitality = store.get_community_vitality(1).unwrap();
    assert_eq!(vitality.trend, Trend::Improving);
}

#[test]
fn trend_remains_stable_within_threshold() {
    let config = StagnationConfig {
        trend_tolerance: 0.1,
        ..default_config()
    };
    let store = Arc::new(StagnationStore::new(config));

    let s1 = StagnationScore {
        community_id: 5,
        avg_energy: 0.5,
        last_mutation_age: Duration::from_secs(100),
        synapse_activity: 0.3,
        stagnation_score: 0.50,
        trend: Trend::Stable,
    };
    store.update(5, s1);

    // Delta of 0.05 < threshold of 0.1 → still Stable
    let s2 = StagnationScore {
        community_id: 5,
        avg_energy: 0.45,
        last_mutation_age: Duration::from_secs(120),
        synapse_activity: 0.28,
        stagnation_score: 0.55,
        trend: Trend::Stable,
    };
    store.update(5, s2);
    let vitality = store.get_community_vitality(5).unwrap();
    assert_eq!(vitality.trend, Trend::Stable);
}

// ---------------------------------------------------------------------------
// StagnationStore — get_stagnant_zones
// ---------------------------------------------------------------------------

#[test]
fn get_stagnant_zones_filters_and_sorts() {
    let config = default_config();
    let store = StagnationStore::new(config);

    // Insert communities with varying scores.
    store.update(
        1,
        StagnationScore {
            community_id: 1,
            avg_energy: 0.1,
            last_mutation_age: Duration::from_secs(1000),
            synapse_activity: 0.05,
            stagnation_score: 0.9,
            trend: Trend::Stable,
        },
    );
    store.update(
        2,
        StagnationScore {
            community_id: 2,
            avg_energy: 0.8,
            last_mutation_age: Duration::from_secs(10),
            synapse_activity: 0.7,
            stagnation_score: 0.2,
            trend: Trend::Stable,
        },
    );
    store.update(
        3,
        StagnationScore {
            community_id: 3,
            avg_energy: 0.3,
            last_mutation_age: Duration::from_secs(600),
            synapse_activity: 0.2,
            stagnation_score: 0.7,
            trend: Trend::Stable,
        },
    );

    let zones = store.get_stagnant_zones(0.5);
    assert_eq!(zones.len(), 2, "only communities 1 and 3 exceed 0.5");
    assert_eq!(zones[0].community_id, 1, "highest stagnation first");
    assert_eq!(zones[1].community_id, 3);
}

#[test]
fn get_stagnant_zones_empty_when_all_below_threshold() {
    let config = default_config();
    let store = StagnationStore::new(config);

    store.update(
        1,
        StagnationScore {
            community_id: 1,
            avg_energy: 0.9,
            last_mutation_age: Duration::from_secs(10),
            synapse_activity: 0.8,
            stagnation_score: 0.1,
            trend: Trend::Stable,
        },
    );

    let zones = store.get_stagnant_zones(0.5);
    assert!(zones.is_empty());
}

// ---------------------------------------------------------------------------
// StagnationStore — get_community_vitality
// ---------------------------------------------------------------------------

#[test]
fn get_community_vitality_returns_none_for_unknown() {
    let store = StagnationStore::new(default_config());
    assert!(store.get_community_vitality(999).is_none());
}

#[test]
fn get_community_vitality_returns_latest_score() {
    let store = StagnationStore::new(default_config());
    store.update(
        42,
        StagnationScore {
            community_id: 42,
            avg_energy: 0.5,
            last_mutation_age: Duration::from_secs(300),
            synapse_activity: 0.4,
            stagnation_score: 0.55,
            trend: Trend::Stable,
        },
    );

    let vitality = store.get_community_vitality(42).unwrap();
    assert_eq!(vitality.community_id, 42);
    assert!((vitality.stagnation_score - 0.55).abs() < 1e-10);
}

// ---------------------------------------------------------------------------
// Detector with analyze + record round-trip
// ---------------------------------------------------------------------------

#[test]
fn analyze_and_record_round_trip() {
    let config = default_config();
    let stag_store = Arc::new(StagnationStore::new(config.clone()));
    let energy_store = make_energy_store(&[(1, 0.6), (2, 0.4)]);

    #[cfg(feature = "synapse")]
    let synapse_store = make_synapse_store(&[(1, 2, 0.3)]);

    let detector = StagnationDetector::new(
        energy_store,
        #[cfg(feature = "synapse")]
        synapse_store,
        Arc::clone(&stag_store),
        config,
    );

    let node_ids = vec![NodeId::new(1), NodeId::new(2)];
    let score = detector.analyze_community(10, &node_ids, Duration::from_secs(3600));
    detector.record_snapshot(10, score);

    let vitality = stag_store.get_community_vitality(10);
    assert!(vitality.is_some());
    let v = vitality.unwrap();
    assert_eq!(v.community_id, 10);
    assert!(v.stagnation_score >= 0.0 && v.stagnation_score <= 1.0);
}
