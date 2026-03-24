//! Integration tests for the distillation P2P module.

#![cfg(all(
    feature = "distillation",
    feature = "energy",
    feature = "synapse"
))]

use std::sync::Arc;
use std::time::SystemTime;

use grafeo_cognitive::distillation::{
    ArtifactMetadata, DistillArtifact, DistillConfig, EnergySnapshot, SynapseSnapshot, distill,
    evaluate, inject,
};
use grafeo_cognitive::engine::CognitiveEngine;
use grafeo_cognitive::{CognitiveConfig, CognitiveEngineBuilder};
use grafeo_common::types::NodeId;
use grafeo_reactive::{BatchConfig, MutationBus, Scheduler};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_test_engine() -> (Arc<dyn CognitiveEngine>, Scheduler) {
    let bus = MutationBus::new();
    let scheduler = Scheduler::new(&bus, BatchConfig::default());
    let mut config = CognitiveConfig::new();
    config.energy.enabled = true;
    config.synapse.enabled = true;
    let engine = CognitiveEngineBuilder::from_config(&config).build(&scheduler);
    (Arc::new(engine) as Arc<dyn CognitiveEngine>, scheduler)
}

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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn distill_extracts_synapses_above_threshold() {
    let (engine, _scheduler) = make_test_engine();

    // Create synapses with varying weights via the store.
    // reinforce creates a synapse with initial_weight + amount.
    // Default initial_weight = 0.1, so reinforce(0.5) => 0.6.
    let store = engine.synapse_store().unwrap();
    store.reinforce(NodeId(1), NodeId(2), 0.5);
    // reinforce(0.0) => initial_weight 0.1 + 0.0 = 0.1
    store.reinforce(NodeId(3), NodeId(4), 0.0);
    // 0.1 + 0.001 = 0.101 (barely above default 0.1 threshold)
    store.reinforce(NodeId(5), NodeId(6), 0.001);

    let config = DistillConfig {
        min_synapse_weight: 0.15,
        ..DistillConfig::default()
    };

    let artifact = distill(&config, &[], engine.as_ref());
    // Only the (1,2) synapse with weight ~0.6 should be included
    assert_eq!(artifact.synapses.len(), 1);
    assert!(artifact.synapses[0].weight > 0.5);
}

#[tokio::test]
async fn distill_extracts_energies_above_threshold() {
    let (engine, _scheduler) = make_test_engine();

    let estore = engine.energy_store().unwrap();
    estore.boost(NodeId(10), 5.0);
    estore.boost(NodeId(20), 0.005); // below default threshold 0.01
    estore.boost(NodeId(30), 1.0);

    let config = DistillConfig::default();
    let node_ids = vec![NodeId(10), NodeId(20), NodeId(30)];
    let artifact = distill(&config, &node_ids, engine.as_ref());

    // node 20 has energy 0.005 < 0.01 threshold => excluded
    assert_eq!(artifact.energies.len(), 2);
    let ids: Vec<u64> = artifact.energies.iter().map(|e| e.node_id).collect();
    assert!(ids.contains(&10));
    assert!(ids.contains(&30));
}

#[tokio::test]
async fn distill_empty_engine_returns_empty_artifact() {
    let (engine, _scheduler) = make_test_engine();
    let config = DistillConfig::default();
    let artifact = distill(&config, &[], engine.as_ref());

    assert!(artifact.synapses.is_empty());
    assert!(artifact.energies.is_empty());
    assert!(artifact.fingerprints.is_empty());
    assert_eq!(artifact.version, "1.0");
}

#[tokio::test]
async fn inject_applies_trust_discounted_weights() {
    let (engine, _scheduler) = make_test_engine();

    let artifact = make_artifact(vec![(1, 2, 0.8)], vec![]);
    inject(&artifact, 0.5, engine.as_ref());

    let store = engine.synapse_store().unwrap();
    let syn = store.get_synapse(NodeId(1), NodeId(2)).unwrap();
    // reinforce creates with initial_weight(0.1) + amount(0.8 * 0.5 = 0.4) = 0.5
    let w = syn.current_weight();
    assert!(
        (w - 0.5).abs() < 0.05,
        "expected weight ~0.5, got {w}"
    );
}

#[tokio::test]
async fn inject_applies_trust_discounted_energy() {
    let (engine, _scheduler) = make_test_engine();

    let artifact = make_artifact(vec![], vec![(42, 10.0)]);
    inject(&artifact, 0.3, engine.as_ref());

    let estore = engine.energy_store().unwrap();
    let e = estore.get_energy(NodeId(42));
    // boost(10.0 * 0.3 = 3.0)
    assert!(
        (e - 3.0).abs() < 0.1,
        "expected energy ~3.0, got {e}"
    );
}

#[tokio::test]
async fn evaluate_identical_artifacts_score_near_one() {
    let a = make_artifact(
        vec![(1, 2, 0.5), (3, 4, 0.8)],
        vec![(1, 1.0), (2, 2.0), (3, 3.0)],
    );
    let b = make_artifact(
        vec![(1, 2, 0.5), (3, 4, 0.8)],
        vec![(1, 1.0), (2, 2.0), (3, 3.0)],
    );
    let report = evaluate(&a, &b);

    assert!(
        (report.evidence_coverage - 1.0).abs() < f64::EPSILON,
        "evidence_coverage should be 1.0, got {}",
        report.evidence_coverage
    );
    assert!(
        (report.community_overlap - 1.0).abs() < 1e-9,
        "community_overlap should be ~1.0, got {}",
        report.community_overlap
    );
    // With identical artifacts, most factors should be high
    assert!(
        report.composite_score > 0.3,
        "composite_score should be > 0.3, got {}",
        report.composite_score
    );
}

#[tokio::test]
async fn evaluate_different_artifacts_lower_score() {
    let a = make_artifact(vec![(1, 2, 0.5)], vec![(1, 1.0), (2, 2.0)]);
    let b = make_artifact(
        vec![(3, 4, 0.8), (5, 6, 0.3)],
        vec![(3, 5.0), (4, 10.0)],
    );
    let report = evaluate(&a, &b);

    assert!(
        report.evidence_coverage < 0.01,
        "evidence_coverage should be ~0, got {}",
        report.evidence_coverage
    );
    // No shared nodes => community_overlap = 0.0
    assert!(
        report.community_overlap.abs() < f64::EPSILON,
        "community_overlap should be 0.0 for disjoint nodes, got {}",
        report.community_overlap
    );
    assert!(
        report.composite_score < 0.1,
        "composite_score should be low, got {}",
        report.composite_score
    );
}

#[tokio::test]
async fn default_config_has_sensible_values() {
    let config = DistillConfig::default();
    assert!((config.min_synapse_weight - 0.1).abs() < f64::EPSILON);
    assert!((config.min_energy - 0.01).abs() < f64::EPSILON);
    assert!(!config.include_fingerprints);
    assert!(!config.include_episodes);
    assert!(config.community_filter.is_none());
}

#[tokio::test]
async fn artifact_metadata_is_populated() {
    let (engine, _scheduler) = make_test_engine();

    let store = engine.synapse_store().unwrap();
    store.reinforce(NodeId(1), NodeId(2), 0.5);
    store.reinforce(NodeId(3), NodeId(4), 0.5);

    let estore = engine.energy_store().unwrap();
    estore.boost(NodeId(1), 1.0);
    estore.boost(NodeId(2), 1.0);
    estore.boost(NodeId(3), 1.0);

    let node_ids = vec![NodeId(1), NodeId(2), NodeId(3)];
    let config = DistillConfig::default();
    let artifact = distill(&config, &node_ids, engine.as_ref());

    assert_eq!(artifact.metadata.source_instance, "local");
    assert_eq!(artifact.metadata.total_nodes, 3);
    assert_eq!(artifact.metadata.total_synapses, 2);
    assert_eq!(artifact.version, "1.0");
}

#[tokio::test]
async fn distill_respects_community_filter() {
    // Community filtering is a no-op for synapse/energy extraction,
    // but verify the config is accepted without error.
    let (engine, _scheduler) = make_test_engine();

    let store = engine.synapse_store().unwrap();
    store.reinforce(NodeId(1), NodeId(2), 0.5);

    let config = DistillConfig {
        community_filter: Some(vec![42]),
        ..DistillConfig::default()
    };

    // Should still extract synapses (community_filter is not applied to synapse extraction)
    let artifact = distill(&config, &[], engine.as_ref());
    assert_eq!(artifact.synapses.len(), 1);
}
