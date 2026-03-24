//! Extra coverage tests for the distillation module.
//!
//! These test the private helper functions (`synapse_jaccard`, `energy_pearson`,
//! `compute_hub_coverage`, `compute_cemented`, `compute_cross_community`)
//! indirectly through the public `evaluate` / `evaluate_with_config` API.

#![cfg(feature = "distillation")]

use std::time::SystemTime;

use grafeo_cognitive::distillation::{
    ArtifactMetadata, DistillArtifact, EnergySnapshot, EvaluateConfig, ParityReport,
    SynapseSnapshot, evaluate, evaluate_with_config,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_artifact(synapses: Vec<(u64, u64, f64)>, energies: Vec<(u64, f64)>) -> DistillArtifact {
    DistillArtifact {
        version: String::from("1.0"),
        created_at: SystemTime::now(),
        synapses: synapses
            .into_iter()
            .map(|(s, t, w)| SynapseSnapshot {
                source: s,
                target: t,
                weight: w,
            })
            .collect(),
        energies: energies
            .into_iter()
            .map(|(id, e)| EnergySnapshot {
                node_id: id,
                energy: e,
            })
            .collect(),
        fingerprints: Vec::new(),
        metadata: ArtifactMetadata {
            source_instance: String::from("test"),
            total_nodes: 0,
            total_synapses: 0,
            total_communities: 0,
        },
    }
}

/// Helper that uses equal weights so each factor contributes 0.2.
fn equal_weights_config() -> EvaluateConfig {
    EvaluateConfig {
        threshold: 0.0,
        weights: [0.2, 0.2, 0.2, 0.2, 0.2],
    }
}

/// Evaluate and return each factor individually for inspection.
fn factors(a: &DistillArtifact, b: &DistillArtifact) -> ParityReport {
    evaluate_with_config(a, b, &equal_weights_config())
}

// =========================================================================
// 1. synapse_jaccard edge cases
// =========================================================================

#[test]
fn jaccard_both_empty_returns_one() {
    let a = make_artifact(vec![], vec![]);
    let b = make_artifact(vec![], vec![]);
    let r = factors(&a, &b);
    assert!(
        (r.evidence_coverage - 1.0).abs() < f64::EPSILON,
        "both empty => jaccard 1.0, got {}",
        r.evidence_coverage,
    );
}

#[test]
fn jaccard_one_empty_returns_zero() {
    let a = make_artifact(vec![(1, 2, 0.5)], vec![]);
    let b = make_artifact(vec![], vec![]);
    let r = factors(&a, &b);
    assert!(
        r.evidence_coverage.abs() < f64::EPSILON,
        "one empty => jaccard 0.0, got {}",
        r.evidence_coverage,
    );
}

#[test]
fn jaccard_partial_overlap() {
    // shared: (1,2), only-a: (3,4), only-b: (5,6) => intersection=1, union=3 => 1/3
    let a = make_artifact(vec![(1, 2, 0.5), (3, 4, 0.5)], vec![]);
    let b = make_artifact(vec![(1, 2, 0.5), (5, 6, 0.5)], vec![]);
    let r = factors(&a, &b);
    let expected = 1.0 / 3.0;
    assert!(
        (r.evidence_coverage - expected).abs() < 1e-9,
        "partial overlap => {expected}, got {}",
        r.evidence_coverage,
    );
}

// =========================================================================
// 2. energy_pearson edge cases
// =========================================================================

#[test]
fn pearson_both_empty_returns_one() {
    let a = make_artifact(vec![], vec![]);
    let b = make_artifact(vec![], vec![]);
    let r = factors(&a, &b);
    assert!(
        (r.community_overlap - 1.0).abs() < f64::EPSILON,
        "both empty => pearson 1.0, got {}",
        r.community_overlap,
    );
}

#[test]
fn pearson_no_shared_nodes_returns_zero() {
    let a = make_artifact(vec![], vec![(1, 1.0), (2, 2.0)]);
    let b = make_artifact(vec![], vec![(3, 3.0), (4, 4.0)]);
    let r = factors(&a, &b);
    assert!(
        r.community_overlap.abs() < f64::EPSILON,
        "no shared nodes => pearson 0.0, got {}",
        r.community_overlap,
    );
}

#[test]
fn pearson_single_pair_equal() {
    // single pair with equal values => 1.0
    let a = make_artifact(vec![], vec![(1, 5.0)]);
    let b = make_artifact(vec![], vec![(1, 5.0)]);
    let r = factors(&a, &b);
    assert!(
        (r.community_overlap - 1.0).abs() < f64::EPSILON,
        "single pair equal => pearson 1.0, got {}",
        r.community_overlap,
    );
}

#[test]
fn pearson_single_pair_unequal() {
    // single pair with different values => 0.0
    let a = make_artifact(vec![], vec![(1, 5.0)]);
    let b = make_artifact(vec![], vec![(1, 10.0)]);
    let r = factors(&a, &b);
    assert!(
        r.community_overlap.abs() < f64::EPSILON,
        "single pair unequal => pearson 0.0, got {}",
        r.community_overlap,
    );
}

#[test]
fn pearson_constant_values_both_sides() {
    // All identical values on both sides => zero variance on both => 1.0
    let a = make_artifact(vec![], vec![(1, 3.0), (2, 3.0), (3, 3.0)]);
    let b = make_artifact(vec![], vec![(1, 3.0), (2, 3.0), (3, 3.0)]);
    let r = factors(&a, &b);
    assert!(
        (r.community_overlap - 1.0).abs() < f64::EPSILON,
        "constant both => pearson 1.0, got {}",
        r.community_overlap,
    );
}

#[test]
fn pearson_constant_one_side_only() {
    // Constant on a, varying on b => zero variance on a only => 0.0
    let a = make_artifact(vec![], vec![(1, 5.0), (2, 5.0), (3, 5.0)]);
    let b = make_artifact(vec![], vec![(1, 1.0), (2, 2.0), (3, 3.0)]);
    let r = factors(&a, &b);
    assert!(
        r.community_overlap.abs() < f64::EPSILON,
        "constant one side => pearson 0.0, got {}",
        r.community_overlap,
    );
}

#[test]
fn pearson_negative_correlation() {
    // Perfect negative correlation: a increases, b decreases
    let a = make_artifact(vec![], vec![(1, 1.0), (2, 2.0), (3, 3.0)]);
    let b = make_artifact(vec![], vec![(1, 3.0), (2, 2.0), (3, 1.0)]);
    let r = factors(&a, &b);
    assert!(
        (r.community_overlap - (-1.0)).abs() < 1e-9,
        "negative correlation => pearson -1.0, got {}",
        r.community_overlap,
    );
}

// =========================================================================
// 3. compute_hub_coverage
// =========================================================================

#[test]
fn hub_coverage_before_empty_returns_one() {
    let a = make_artifact(vec![], vec![]);
    let b = make_artifact(vec![(1, 2, 0.5)], vec![]);
    let r = factors(&a, &b);
    assert!(
        (r.hub_coverage - 1.0).abs() < f64::EPSILON,
        "before empty => hub_coverage 1.0, got {}",
        r.hub_coverage,
    );
}

#[test]
fn hub_coverage_top20_selection() {
    // 10 distinct nodes from 5 synapses: hub_count = max(10/5, 1) = 2
    // Node 1 has degree 5 (source in all), node 2 has degree 1 (target of first).
    // Make node 1 a hub by connecting it to many targets.
    // before: 1->2, 1->3, 1->4, 1->5, 1->6 => degree(1)=5, degree(2..6)=1 each
    // 6 unique nodes => hub_count = max(6/5, 1) = max(1,1)=1 => top hub is node 1
    // after has node 1 => coverage = 1.0
    let a = make_artifact(
        vec![(1, 2, 0.5), (1, 3, 0.5), (1, 4, 0.5), (1, 5, 0.5), (1, 6, 0.5)],
        vec![],
    );
    let b = make_artifact(vec![(1, 7, 0.5)], vec![]);
    let r = factors(&a, &b);
    assert!(
        (r.hub_coverage - 1.0).abs() < f64::EPSILON,
        "hub node 1 present in after => 1.0, got {}",
        r.hub_coverage,
    );
}

#[test]
fn hub_coverage_hub_missing_in_after() {
    // Same as above but after does NOT contain node 1
    let a = make_artifact(
        vec![(1, 2, 0.5), (1, 3, 0.5), (1, 4, 0.5), (1, 5, 0.5), (1, 6, 0.5)],
        vec![],
    );
    let b = make_artifact(vec![(10, 11, 0.5)], vec![]);
    let r = factors(&a, &b);
    assert!(
        r.hub_coverage.abs() < f64::EPSILON,
        "hub node missing => 0.0, got {}",
        r.hub_coverage,
    );
}

#[test]
fn hub_coverage_fewer_than_5_nodes() {
    // 2 nodes only from 1 synapse => hub_count = max(2/5, 1) = 1
    let a = make_artifact(vec![(1, 2, 0.5)], vec![]);
    let b = make_artifact(vec![(1, 2, 0.5)], vec![]);
    let r = factors(&a, &b);
    // Top hub: either node 1 or 2 (both degree 1). One of them is in after => 1.0
    assert!(
        (r.hub_coverage - 1.0).abs() < f64::EPSILON,
        "few nodes, hub present => 1.0, got {}",
        r.hub_coverage,
    );
}

// =========================================================================
// 4. compute_cemented
// =========================================================================

#[test]
fn cemented_both_empty_returns_one() {
    let a = make_artifact(vec![], vec![]);
    let b = make_artifact(vec![], vec![]);
    let r = factors(&a, &b);
    assert!(
        (r.cemented - 1.0).abs() < f64::EPSILON,
        "both empty => cemented 1.0, got {}",
        r.cemented,
    );
}

#[test]
fn cemented_no_shared_synapses() {
    let a = make_artifact(vec![(1, 2, 0.5)], vec![]);
    let b = make_artifact(vec![(3, 4, 0.5)], vec![]);
    let r = factors(&a, &b);
    assert!(
        r.cemented.abs() < f64::EPSILON,
        "no shared => cemented 0.0, got {}",
        r.cemented,
    );
}

#[test]
fn cemented_weight_agreement() {
    // Shared synapse (1,2): before=0.8, after=0.6. min/max = 0.6/0.8 = 0.75 > 0.5 => cemented
    let a = make_artifact(vec![(1, 2, 0.8)], vec![]);
    let b = make_artifact(vec![(1, 2, 0.6)], vec![]);
    let r = factors(&a, &b);
    assert!(
        (r.cemented - 1.0).abs() < f64::EPSILON,
        "good agreement => cemented 1.0, got {}",
        r.cemented,
    );
}

#[test]
fn cemented_weight_disagreement() {
    // Shared synapse (1,2): before=1.0, after=0.1. min/max = 0.1/1.0 = 0.1 < 0.5 => not cemented
    let a = make_artifact(vec![(1, 2, 1.0)], vec![]);
    let b = make_artifact(vec![(1, 2, 0.1)], vec![]);
    let r = factors(&a, &b);
    assert!(
        r.cemented.abs() < f64::EPSILON,
        "bad agreement => cemented 0.0, got {}",
        r.cemented,
    );
}

#[test]
fn cemented_epsilon_case() {
    // Both weights ~0 (epsilon case: max < EPSILON => counted as cemented)
    let a = make_artifact(vec![(1, 2, 0.0)], vec![]);
    let b = make_artifact(vec![(1, 2, 0.0)], vec![]);
    let r = factors(&a, &b);
    assert!(
        (r.cemented - 1.0).abs() < f64::EPSILON,
        "epsilon case => cemented 1.0, got {}",
        r.cemented,
    );
}

#[test]
fn cemented_mixed_agreement() {
    // Two shared synapses: one agrees, one doesn't => 0.5
    let a = make_artifact(vec![(1, 2, 0.8), (3, 4, 1.0)], vec![]);
    let b = make_artifact(vec![(1, 2, 0.7), (3, 4, 0.1)], vec![]);
    let r = factors(&a, &b);
    // (1,2): min/max = 0.7/0.8 = 0.875 > 0.5 => cemented
    // (3,4): min/max = 0.1/1.0 = 0.1 < 0.5 => not cemented
    assert!(
        (r.cemented - 0.5).abs() < f64::EPSILON,
        "mixed => cemented 0.5, got {}",
        r.cemented,
    );
}

// =========================================================================
// 5. compute_cross_community
// =========================================================================

#[test]
fn cross_community_both_empty_returns_one() {
    let a = make_artifact(vec![], vec![]);
    let b = make_artifact(vec![], vec![]);
    let r = factors(&a, &b);
    assert!(
        (r.cross_community - 1.0).abs() < f64::EPSILON,
        "both empty => cross 1.0, got {}",
        r.cross_community,
    );
}

#[test]
fn cross_community_same_degree_class() {
    // All nodes degree 1 (class 0: d<=2). No cross-class edges => 0.0
    let a = make_artifact(vec![(1, 2, 0.5)], vec![]);
    let b = make_artifact(vec![], vec![]);
    let r = factors(&a, &b);
    // degree(1)=1, degree(2)=1 => both class 0 => cross = 0
    assert!(
        r.cross_community.abs() < f64::EPSILON,
        "same class => cross 0.0, got {}",
        r.cross_community,
    );
}

#[test]
fn cross_community_different_degree_classes() {
    // Node 1 connects to many targets => high degree (class 2: d>5)
    // Those targets have low degree (class 0: d<=2)
    // All synapses bridge class 2 <-> class 0 => cross = 1.0
    let a = make_artifact(
        vec![
            (1, 10, 0.5),
            (1, 11, 0.5),
            (1, 12, 0.5),
            (1, 13, 0.5),
            (1, 14, 0.5),
            (1, 15, 0.5),
        ],
        vec![],
    );
    let b = make_artifact(vec![], vec![]);
    let r = factors(&a, &b);
    // degree(1)=6 => class 2, degree(10..15)=1 => class 0
    // All 6 synapses are cross-class => cross = 6/6 = 1.0
    assert!(
        (r.cross_community - 1.0).abs() < f64::EPSILON,
        "all cross-class => 1.0, got {}",
        r.cross_community,
    );
}

#[test]
fn cross_community_mixed_classes() {
    // Create a scenario with both cross-class and same-class edges
    // Combine into before+after: degrees computed over all synapses from both
    // before: (1,2), (1,3), (1,4) => degree(1)=3 in before
    // after:  (1,5), (1,6), (1,7) => degree(1)=3 in after
    // Combined: degree(1) = 6 => class 2, degree(2..7) = 1 each => class 0
    // All 6 synapses are cross-class => cross = 1.0
    //
    // Let's also add a same-class edge to get a mix:
    // before: (1,2), (1,3), (1,4), (10,11)
    // after:  (1,5), (1,6), (1,7)
    // Combined degrees: 1=>6, 2..7=>1, 10=>1, 11=>1. All class 0 except node 1 (class 2).
    // Synapses involving node 1: 6 are cross-class. (10,11) is same-class.
    // Total = 7, cross = 6 => 6/7
    let a = make_artifact(
        vec![(1, 2, 0.5), (1, 3, 0.5), (1, 4, 0.5), (10, 11, 0.5)],
        vec![],
    );
    let b = make_artifact(vec![(1, 5, 0.5), (1, 6, 0.5), (1, 7, 0.5)], vec![]);
    let r = factors(&a, &b);
    let expected = 6.0 / 7.0;
    assert!(
        (r.cross_community - expected).abs() < 1e-9,
        "mixed => cross {expected}, got {}",
        r.cross_community,
    );
}

#[test]
fn cross_community_empty_synapses_in_one() {
    // Only one artifact has synapses
    let a = make_artifact(vec![], vec![]);
    let b = make_artifact(vec![(1, 2, 0.5)], vec![]);
    let r = factors(&a, &b);
    // degree(1)=1, degree(2)=1 => both class 0 => cross = 0/1 = 0.0
    assert!(
        r.cross_community.abs() < f64::EPSILON,
        "one empty synapses => cross 0.0, got {}",
        r.cross_community,
    );
}

// =========================================================================
// 6. Full integration: evaluate with shared/divergent state
// =========================================================================

#[test]
fn integration_shared_and_divergent_state() {
    // Epoch "before": rich graph with synapses and energies
    let before = make_artifact(
        vec![
            (1, 2, 0.8),
            (2, 3, 0.6),
            (3, 4, 0.7),
            (4, 5, 0.5),
            (5, 1, 0.9),
            (1, 3, 0.4),
            (1, 4, 0.3),
        ],
        vec![
            (1, 5.0),
            (2, 3.0),
            (3, 2.0),
            (4, 1.0),
            (5, 4.0),
        ],
    );

    // Epoch "after": some shared synapses, some new, some dropped
    let after = make_artifact(
        vec![
            (1, 2, 0.7),   // shared, similar weight
            (2, 3, 0.5),   // shared, similar weight
            (3, 4, 0.1),   // shared, divergent weight
            (6, 7, 0.5),   // new
            (7, 8, 0.3),   // new
        ],
        vec![
            (1, 4.5),  // shared node, similar
            (2, 3.5),  // shared node, similar
            (3, 2.5),  // shared node, similar
            (6, 1.0),  // new node
            (7, 2.0),  // new node
        ],
    );

    let report = evaluate(&before, &after);

    // evidence_coverage: intersection=3 {(1,2),(2,3),(3,4)}, union=9 => 3/9 = 1/3
    assert!(
        (report.evidence_coverage - 1.0 / 3.0).abs() < 1e-9,
        "evidence_coverage = 1/3, got {}",
        report.evidence_coverage,
    );

    // community_overlap: shared nodes {1,2,3} with correlated energies
    // before: (5.0, 3.0, 2.0), after: (4.5, 3.5, 2.5) => positive correlation
    assert!(
        report.community_overlap > 0.9,
        "community_overlap should be high positive, got {}",
        report.community_overlap,
    );

    // hub_coverage: node 1 is the hub in before (degree 4: source of 1->2,1->3,1->4 + target of 5->1)
    // After has node 1 (in synapse 1->2) => covered
    assert!(
        report.hub_coverage > 0.0,
        "hub_coverage > 0, got {}",
        report.hub_coverage,
    );

    // cemented: 3 shared synapses.
    // (1,2): 0.7/0.8 = 0.875 > 0.5 => cemented
    // (2,3): 0.5/0.6 = 0.833 > 0.5 => cemented
    // (3,4): 0.1/0.7 = 0.143 < 0.5 => not cemented
    // => 2/3
    assert!(
        (report.cemented - 2.0 / 3.0).abs() < 1e-9,
        "cemented = 2/3, got {}",
        report.cemented,
    );

    // cross_community: depends on degree classes of combined graph
    assert!(
        report.cross_community >= 0.0 && report.cross_community <= 1.0,
        "cross_community in [0,1], got {}",
        report.cross_community,
    );

    // composite_score: weighted sum should be meaningful
    assert!(
        report.composite_score > 0.0,
        "composite_score > 0, got {}",
        report.composite_score,
    );
}

#[test]
fn integration_identical_epochs_pass() {
    let epoch = make_artifact(
        vec![
            (1, 2, 0.5),
            (2, 3, 0.8),
            (3, 1, 0.6),
        ],
        vec![
            (1, 1.0),
            (2, 2.0),
            (3, 3.0),
        ],
    );

    let report = evaluate(&epoch, &epoch);

    assert!(
        (report.evidence_coverage - 1.0).abs() < f64::EPSILON,
        "identical => jaccard 1.0",
    );
    assert!(
        (report.community_overlap - 1.0).abs() < 1e-9,
        "identical => pearson 1.0",
    );
    assert!(
        (report.cemented - 1.0).abs() < f64::EPSILON,
        "identical => cemented 1.0",
    );
    assert!(
        report.passed || report.composite_score >= report.threshold * 0.99,
        "identical epochs should score high",
    );
}

#[test]
fn integration_completely_disjoint_epochs() {
    let before = make_artifact(
        vec![(1, 2, 0.5), (2, 3, 0.8)],
        vec![(1, 1.0), (2, 2.0), (3, 3.0)],
    );
    let after = make_artifact(
        vec![(10, 11, 0.5), (11, 12, 0.8)],
        vec![(10, 1.0), (11, 2.0), (12, 3.0)],
    );

    let report = evaluate(&before, &after);

    assert!(
        report.evidence_coverage.abs() < f64::EPSILON,
        "disjoint => jaccard 0.0",
    );
    assert!(
        report.community_overlap.abs() < f64::EPSILON,
        "disjoint => pearson 0.0",
    );
    assert!(
        report.cemented.abs() < f64::EPSILON,
        "disjoint => cemented 0.0",
    );
    assert!(
        report.composite_score < 0.3,
        "disjoint => low score, got {}",
        report.composite_score,
    );
}

#[test]
fn evaluate_custom_weights_affect_composite() {
    let a = make_artifact(vec![(1, 2, 0.5)], vec![(1, 1.0), (2, 2.0)]);
    let b = make_artifact(vec![(1, 2, 0.5)], vec![(1, 1.0), (2, 2.0)]);

    // Weight everything on evidence_coverage (which is 1.0 for identical synapses)
    let config = EvaluateConfig {
        threshold: 0.99,
        weights: [1.0, 0.0, 0.0, 0.0, 0.0],
    };
    let r = evaluate_with_config(&a, &b, &config);
    assert!(
        (r.composite_score - 1.0).abs() < f64::EPSILON,
        "100% weight on jaccard=1.0 => composite 1.0, got {}",
        r.composite_score,
    );
    assert!(r.passed, "should pass with threshold 0.99");
}
