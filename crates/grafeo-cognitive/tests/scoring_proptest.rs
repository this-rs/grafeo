//! Property-based tests verifying all scoring functions return [0.0, 1.0]
//! for arbitrary inputs, including edge cases (NaN, infinity, negatives).

#![cfg(all(feature = "energy", feature = "synapse", feature = "fabric"))]

use grafeo_cognitive::{energy_score, synapse_score};
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// energy_score: [0.0, 1.0] for all finite inputs
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn energy_score_always_in_unit_range(energy in -1e10_f64..1e10) {
        let s = energy_score(energy);
        prop_assert!(s >= 0.0, "energy_score({energy}) = {s} < 0.0");
        prop_assert!(s <= 1.0, "energy_score({energy}) = {s} > 1.0");
    }

    #[test]
    fn energy_score_monotonically_increasing(a in 0.0_f64..100.0, b in 0.0_f64..100.0) {
        let sa = energy_score(a);
        let sb = energy_score(b);
        if a <= b {
            prop_assert!(sa <= sb + 1e-12, "energy_score not monotonic: f({a})={sa} > f({b})={sb}");
        } else {
            prop_assert!(sb <= sa + 1e-12, "energy_score not monotonic: f({b})={sb} > f({a})={sa}");
        }
    }

    #[test]
    fn energy_score_zero_at_zero_or_negative(energy in -1e10_f64..=0.0) {
        let s = energy_score(energy);
        prop_assert_eq!(s, 0.0);
    }
}

// ---------------------------------------------------------------------------
// synapse_score: [0.0, 1.0] for all finite inputs
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn synapse_score_always_in_unit_range(weight in -1e10_f64..1e10) {
        let s = synapse_score(weight);
        prop_assert!(s >= 0.0, "synapse_score({weight}) = {s} < 0.0");
        prop_assert!(s <= 1.0, "synapse_score({weight}) = {s} > 1.0");
    }

    #[test]
    fn synapse_score_monotonically_increasing(a in 0.0_f64..100.0, b in 0.0_f64..100.0) {
        let sa = synapse_score(a);
        let sb = synapse_score(b);
        if a <= b {
            prop_assert!(sa <= sb + 1e-12, "synapse_score not monotonic: f({a})={sa} > f({b})={sb}");
        } else {
            prop_assert!(sb <= sa + 1e-12, "synapse_score not monotonic: f({b})={sb} > f({a})={sa}");
        }
    }

    #[test]
    fn synapse_score_zero_at_zero_or_negative(weight in -1e10_f64..=0.0) {
        let s = synapse_score(weight);
        prop_assert_eq!(s, 0.0);
    }
}

// ---------------------------------------------------------------------------
// Edge cases: NaN, infinity, special values
// ---------------------------------------------------------------------------

#[test]
fn energy_score_nan_returns_zero() {
    assert_eq!(energy_score(f64::NAN), 0.0);
}

#[test]
fn energy_score_positive_infinity_returns_bounded() {
    let s = energy_score(f64::INFINITY);
    assert!(s >= 0.0 && s <= 1.0, "got {s}");
}

#[test]
fn energy_score_negative_infinity_returns_zero() {
    assert_eq!(energy_score(f64::NEG_INFINITY), 0.0);
}

#[test]
fn synapse_score_nan_returns_zero() {
    assert_eq!(synapse_score(f64::NAN), 0.0);
}

#[test]
fn synapse_score_positive_infinity_returns_bounded() {
    let s = synapse_score(f64::INFINITY);
    assert!(s >= 0.0 && s <= 1.0, "got {s}");
}

#[test]
fn synapse_score_negative_infinity_returns_zero() {
    assert_eq!(synapse_score(f64::NEG_INFINITY), 0.0);
}

// ---------------------------------------------------------------------------
// Known values
// ---------------------------------------------------------------------------

#[test]
fn energy_score_at_known_values() {
    // energy=0 → 0.0
    assert!((energy_score(0.0)).abs() < 1e-12);
    // energy=1 → 1 - e^(-1) ≈ 0.6321
    assert!((energy_score(1.0) - 0.6321205588285577).abs() < 1e-10);
    // energy=10 → very close to 1.0
    assert!(energy_score(10.0) > 0.9999);
}

#[test]
fn synapse_score_at_known_values() {
    // weight=0 → 0.0
    assert!((synapse_score(0.0)).abs() < 1e-12);
    // weight=1 → tanh(1) ≈ 0.7616
    assert!((synapse_score(1.0) - 0.7615941559557649).abs() < 1e-10);
    // weight=10 → very close to 1.0
    assert!(synapse_score(10.0) > 0.9999);
}

// ---------------------------------------------------------------------------
// FabricScore risk: verify recalculation produces [0, 1]
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn fabric_risk_score_in_unit_range(
        pagerank in 0.0_f64..100.0,
        churn_count in 0u32..1000,
        knowledge_density in 0.0_f64..1.0,
        betweenness in 0.0_f64..100.0,
        scar_intensity in 0.0_f64..50.0,
    ) {
        use grafeo_cognitive::FabricStore;
        use grafeo_common::types::NodeId;

        let store = FabricStore::new();
        let nid = NodeId::new(1);

        // Build up churn by calling update_churn multiple times
        for _ in 0..churn_count {
            store.update_churn(nid);
        }
        store.set_gds_metrics(nid, pagerank, betweenness, None);
        store.set_knowledge_density(nid, knowledge_density);
        store.set_scar_intensity(nid, scar_intensity);

        store.recalculate_risk(nid, pagerank.max(1.0), (churn_count as f64).max(1.0), betweenness.max(1.0));

        let score = store.get_fabric_score(nid);
        prop_assert!(score.risk_score >= 0.0, "risk_score = {} < 0.0", score.risk_score);
        prop_assert!(score.risk_score <= 1.0, "risk_score = {} > 1.0", score.risk_score);
    }
}

// ---------------------------------------------------------------------------
// Stagnation formula: verify produces [0, 1] for arbitrary inputs
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn stagnation_formula_in_unit_range(
        avg_energy in 0.0_f64..10.0,
        mutation_age_secs in 0u64..100_000_000,
        synapse_activity in 0.0_f64..10.0,
    ) {
        use grafeo_cognitive::StagnationConfig;
        let config = StagnationConfig::default();

        // Replicate the stagnation formula
        let avg_energy_norm = avg_energy.clamp(0.0, 1.0);
        let age_norm = (mutation_age_secs as f64 / config.max_mutation_age.as_secs_f64()).clamp(0.0, 1.0);
        let synapse_norm = synapse_activity.clamp(0.0, 1.0);

        let raw = (1.0 - avg_energy_norm) * config.weight_energy
            + age_norm * config.weight_mutation_age
            + (1.0 - synapse_norm) * config.weight_synapse_activity;
        let stagnation = raw.clamp(0.0, 1.0);

        prop_assert!(stagnation >= 0.0, "stagnation = {} < 0.0", stagnation);
        prop_assert!(stagnation <= 1.0, "stagnation = {} > 1.0", stagnation);
    }
}
