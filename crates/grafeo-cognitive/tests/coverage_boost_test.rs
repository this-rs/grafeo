#![cfg(all(
    feature = "energy",
    feature = "synapse",
    feature = "fabric",
    feature = "scar",
    feature = "stagnation",
    feature = "consolidation",
    feature = "distillation",
    feature = "episodic",
    feature = "memory"
))]

//! Coverage-boost tests targeting low-coverage modules:
//! - search.rs: pipeline builders, signal computation, Debug/Default, sort, truncate
//! - tenant.rs: store accessors, validate_name edge cases, Debug/Default
//! - store_trait.rs: load/persist helpers for node/edge f64 properties
//! - provenance.rs: CognitiveEventId Display, convenience recorders, Debug/Default
//! - scar.rs: with_graph_store, persist_scar_summary

use std::sync::Arc;

use grafeo_cognitive::energy::{EnergyConfig, EnergyStore};
use grafeo_cognitive::fabric::FabricStore;
use grafeo_cognitive::provenance::{CognitiveEventId, CognitiveEventType, ProvenanceRecorder};
use grafeo_cognitive::scar::{ScarConfig, ScarReason, ScarStore};
use grafeo_cognitive::search::{
    NoopReranker, Reranker, SearchConfig, SearchPipeline, SearchResult, SearchWeights,
};
use grafeo_cognitive::synapse::{SynapseConfig, SynapseStore};
use grafeo_cognitive::tenant::{TenantError, TenantManager};
use grafeo_common::types::NodeId;

fn nid(id: u64) -> NodeId {
    NodeId(id)
}

// =========================================================================
// search.rs — coverage boost
// =========================================================================

#[test]
fn search_pipeline_with_energy_store_builder() {
    let store = Arc::new(EnergyStore::new(EnergyConfig::default()));
    store.boost(nid(1), 10.0);

    let pipeline = SearchPipeline::new().with_energy_store(Arc::clone(&store));

    let config = SearchConfig {
        weights: SearchWeights::new(0.0, 1.0, 0.0, 0.0),
        limit: 5,
        ..Default::default()
    };

    let candidates = vec![(nid(1), 0.5), (nid(2), 0.5)];
    let results = pipeline.search(&candidates, &config);
    assert_eq!(results.len(), 2);

    // Node 1 has energy, node 2 does not — node 1 should rank higher
    let r1 = results.iter().find(|r| r.node_id == nid(1)).unwrap();
    let r2 = results.iter().find(|r| r.node_id == nid(2)).unwrap();
    assert!(r1.signal_energy > r2.signal_energy);
}

#[test]
fn search_pipeline_with_fabric_store_builder() {
    let store = Arc::new(FabricStore::new());
    store.set_gds_metrics(nid(1), 0.9, 0.7, None);
    store.set_gds_metrics(nid(2), 0.1, 0.05, None);

    let pipeline = SearchPipeline::new().with_fabric_store(Arc::clone(&store));

    let config = SearchConfig {
        weights: SearchWeights::new(0.0, 0.0, 1.0, 0.0),
        limit: 5,
        ..Default::default()
    };

    let candidates = vec![(nid(1), 0.5), (nid(2), 0.5)];
    let results = pipeline.search(&candidates, &config);

    // Node 1 has much higher topology metrics
    assert_eq!(results[0].node_id, nid(1));
    assert!(results[0].signal_topology > results[1].signal_topology);
}

#[test]
fn search_pipeline_with_synapse_store_builder() {
    let store = Arc::new(SynapseStore::new(SynapseConfig::default()));
    store.reinforce(nid(1), nid(3), 1.0);
    store.reinforce(nid(3), nid(4), 0.5);

    let pipeline = SearchPipeline::new().with_synapse_store(Arc::clone(&store));

    let config = SearchConfig {
        weights: SearchWeights::new(0.0, 0.0, 0.0, 1.0),
        limit: 10,
        synapse_max_hops: 2,
        synapse_decay_factor: 0.5,
        ..Default::default()
    };

    let candidates = vec![(nid(1), 0.9)];
    let results = pipeline.search(&candidates, &config);

    // The synapse traversal should discover connected nodes
    assert!(!results.is_empty());
    // Node 1 should be in results
    assert!(results.iter().any(|r| r.node_id == nid(1)));
}

#[test]
fn search_pipeline_with_reranker() {
    struct DoubleScoreReranker;
    impl Reranker for DoubleScoreReranker {
        fn rerank(&self, results: Vec<SearchResult>) -> Vec<SearchResult> {
            results
                .into_iter()
                .map(|mut r| {
                    r.score = (r.score * 2.0).min(1.0);
                    r
                })
                .collect()
        }
    }

    let pipeline = SearchPipeline::new().with_reranker(Box::new(DoubleScoreReranker));
    let config = SearchConfig {
        weights: SearchWeights::new(1.0, 0.0, 0.0, 0.0),
        limit: 5,
        ..Default::default()
    };

    let candidates = vec![(nid(1), 0.3)];
    let results = pipeline.search(&candidates, &config);
    assert_eq!(results.len(), 1);
    // Score should be doubled (0.3 * 2.0 = 0.6)
    assert!((results[0].score - 0.6).abs() < 1e-10);
}

#[test]
fn search_pipeline_topology_signal_zero_metrics() {
    // Test topology computation when all metrics are zero
    let store = Arc::new(FabricStore::new());
    // Don't set any metrics — all nodes have zero pagerank/betweenness

    let pipeline = SearchPipeline::new().with_fabric_store(store);
    let config = SearchConfig {
        weights: SearchWeights::new(0.0, 0.0, 1.0, 0.0),
        limit: 5,
        ..Default::default()
    };

    let candidates = vec![(nid(1), 0.5), (nid(2), 0.3)];
    let results = pipeline.search(&candidates, &config);
    assert_eq!(results.len(), 2);
    // With zero topology, all topology signals should be 0
    for r in &results {
        assert!((r.signal_topology - 0.0).abs() < 1e-10);
    }
}

#[test]
fn search_pipeline_synapse_signal_no_connections() {
    // Synapse store has no connections — synapse signal may still be non-zero
    // because spreading activation includes the source nodes themselves.
    let store = Arc::new(SynapseStore::new(SynapseConfig::default()));
    let pipeline = SearchPipeline::new().with_synapse_store(store);

    let config = SearchConfig {
        weights: SearchWeights::new(0.0, 0.0, 0.0, 1.0),
        limit: 5,
        ..Default::default()
    };

    let candidates = vec![(nid(1), 0.9)];
    let results = pipeline.search(&candidates, &config);
    assert_eq!(results.len(), 1);
    // Signal should be bounded [0, 1]
    assert!(results[0].signal_synapse >= 0.0 && results[0].signal_synapse <= 1.0);
}

#[test]
fn search_pipeline_debug_impl() {
    let pipeline = SearchPipeline::new();
    let debug_str = format!("{:?}", pipeline);
    assert!(debug_str.contains("SearchPipeline"));
    assert!(debug_str.contains("energy"));
    assert!(debug_str.contains("synapse"));
    assert!(debug_str.contains("fabric"));
}

#[test]
fn search_pipeline_debug_impl_with_stores() {
    let pipeline = SearchPipeline::new()
        .with_energy_store(Arc::new(EnergyStore::new(EnergyConfig::default())))
        .with_synapse_store(Arc::new(SynapseStore::new(SynapseConfig::default())))
        .with_fabric_store(Arc::new(FabricStore::new()));

    let debug_str = format!("{:?}", pipeline);
    assert!(debug_str.contains("SearchPipeline"));
    // With stores set, the booleans should be true
    assert!(debug_str.contains("true"));
}

#[test]
fn search_pipeline_default_impl() {
    let pipeline = SearchPipeline::default();
    let config = SearchConfig::default();
    let candidates = vec![(nid(1), 0.5)];
    let results = pipeline.search(&candidates, &config);
    assert_eq!(results.len(), 1);
}

#[test]
fn search_sort_by_score_descending() {
    let pipeline = SearchPipeline::new();
    let config = SearchConfig {
        weights: SearchWeights::new(1.0, 0.0, 0.0, 0.0),
        limit: 100,
        ..Default::default()
    };

    let candidates = vec![
        (nid(1), 0.1),
        (nid(2), 0.9),
        (nid(3), 0.5),
        (nid(4), 0.7),
        (nid(5), 0.3),
    ];

    let results = pipeline.search(&candidates, &config);
    // Should be sorted descending by score
    for i in 0..results.len() - 1 {
        assert!(
            results[i].score >= results[i + 1].score,
            "Results not sorted descending at index {}: {} < {}",
            i,
            results[i].score,
            results[i + 1].score
        );
    }
}

#[test]
fn search_truncate_respects_limit() {
    let pipeline = SearchPipeline::new();
    let config = SearchConfig {
        weights: SearchWeights::new(1.0, 0.0, 0.0, 0.0),
        limit: 3,
        ..Default::default()
    };

    let candidates: Vec<(NodeId, f64)> =
        (0..20).map(|i| (nid(i), (i as f64) / 20.0)).collect();

    let results = pipeline.search(&candidates, &config);
    assert_eq!(results.len(), 3);
    // Top 3 should have the highest scores
    assert!(results[0].score >= results[1].score);
    assert!(results[1].score >= results[2].score);
}

#[test]
fn search_all_four_signals_combined() {
    let energy_store = Arc::new(EnergyStore::new(EnergyConfig::default()));
    energy_store.boost(nid(1), 5.0);
    energy_store.boost(nid(2), 1.0);

    let fabric_store = Arc::new(FabricStore::new());
    fabric_store.set_gds_metrics(nid(1), 0.5, 0.3, None);
    fabric_store.set_gds_metrics(nid(2), 0.8, 0.9, None);

    let synapse_store = Arc::new(SynapseStore::new(SynapseConfig::default()));
    synapse_store.reinforce(nid(1), nid(2), 0.8);

    let pipeline = SearchPipeline::new()
        .with_energy_store(energy_store)
        .with_fabric_store(fabric_store)
        .with_synapse_store(synapse_store);

    let config = SearchConfig {
        weights: SearchWeights::new(0.25, 0.25, 0.25, 0.25),
        limit: 10,
        ..Default::default()
    };

    let candidates = vec![(nid(1), 0.7), (nid(2), 0.6)];
    let results = pipeline.search(&candidates, &config);

    // Both nodes should appear with all signals computed
    assert!(results.len() >= 2);
    for r in &results {
        assert!(r.score >= 0.0 && r.score <= 1.0);
        assert!(r.signal_similarity >= 0.0 && r.signal_similarity <= 1.0);
        assert!(r.signal_energy >= 0.0 && r.signal_energy <= 1.0);
        assert!(r.signal_topology >= 0.0 && r.signal_topology <= 1.0);
        assert!(r.signal_synapse >= 0.0 && r.signal_synapse <= 1.0);
    }
}

#[test]
fn search_noop_reranker_debug_and_default() {
    let reranker = NoopReranker;
    let debug_str = format!("{:?}", reranker);
    assert!(debug_str.contains("NoopReranker"));

    let default_reranker = NoopReranker;
    let results = vec![SearchResult {
        node_id: nid(1),
        score: 0.5,
        signal_similarity: 0.5,
        signal_energy: 0.0,
        signal_topology: 0.0,
        signal_synapse: 0.0,
    }];
    let reranked = default_reranker.rerank(results.clone());
    assert_eq!(reranked.len(), 1);
    assert_eq!(reranked[0].node_id, nid(1));
}

// =========================================================================
// tenant.rs — coverage boost
// =========================================================================

#[test]
fn tenant_manager_default_impl() {
    let tm = TenantManager::default();
    assert_eq!(tm.tenant_count(), 0);
    assert!(tm.active_tenant_name().is_none());
}

#[test]
fn tenant_manager_debug_impl() {
    let tm = TenantManager::new();
    tm.create_tenant("test-tenant").unwrap();
    tm.switch_tenant("test-tenant").unwrap();

    let debug_str = format!("{:?}", tm);
    assert!(debug_str.contains("TenantManager"));
    assert!(debug_str.contains("tenant_count"));
    assert!(debug_str.contains("active_tenant"));
}

#[test]
fn tenant_graph_debug_impl() {
    let tm = TenantManager::new();
    tm.create_tenant("dbg-tenant").unwrap();

    let graph = tm.get_tenant("dbg-tenant").unwrap();
    let debug_str = format!("{:?}", &*graph);
    assert!(debug_str.contains("TenantGraph"));
    assert!(debug_str.contains("dbg-tenant"));
    assert!(debug_str.contains("energy_nodes"));
    assert!(debug_str.contains("synapse_count"));
}

#[test]
fn tenant_store_accessors_energy() {
    let tm = TenantManager::new();
    // No active tenant — accessors return None
    assert!(tm.energy_store().is_none());

    tm.create_tenant("accessor-test").unwrap();
    tm.switch_tenant("accessor-test").unwrap();

    let energy = tm.energy_store();
    assert!(energy.is_some());
    let store = energy.unwrap();
    store.boost(nid(42), 3.0);
    assert!((store.get_energy(nid(42)) - 3.0).abs() < 0.1);
}

#[test]
fn tenant_store_accessors_synapse() {
    let tm = TenantManager::new();
    assert!(tm.synapse_store().is_none());

    tm.create_tenant("syn-test").unwrap();
    tm.switch_tenant("syn-test").unwrap();

    let synapse = tm.synapse_store();
    assert!(synapse.is_some());
    let store = synapse.unwrap();
    store.reinforce(nid(1), nid(2), 1.0);
    assert!(store.get_synapse(nid(1), nid(2)).is_some());
}

#[test]
fn tenant_store_accessors_fabric() {
    let tm = TenantManager::new();
    assert!(tm.fabric_store().is_none());

    tm.create_tenant("fab-test").unwrap();
    tm.switch_tenant("fab-test").unwrap();

    let fabric = tm.fabric_store();
    assert!(fabric.is_some());
}

#[test]
fn tenant_store_accessors_scar() {
    let tm = TenantManager::new();
    assert!(tm.scar_store().is_none());

    tm.create_tenant("scar-test").unwrap();
    tm.switch_tenant("scar-test").unwrap();

    let scar = tm.scar_store();
    assert!(scar.is_some());
    let store = scar.unwrap();
    let id = store.add_scar(nid(1), 0.8, ScarReason::Rollback);
    assert!(!store.get_scars(nid(1)).is_empty());
    let _ = id;
}

#[test]
fn tenant_validate_name_too_long() {
    let tm = TenantManager::new();
    let long_name = "a".repeat(257);
    let result = tm.create_tenant(&long_name);
    assert!(matches!(result, Err(TenantError::InvalidName(_))));
}

#[test]
fn tenant_validate_name_max_length_ok() {
    let tm = TenantManager::new();
    let name = "a".repeat(256);
    assert!(tm.create_tenant(&name).is_ok());
}

#[test]
fn tenant_validate_name_invalid_chars() {
    let tm = TenantManager::new();

    // Various invalid characters
    let invalid_names = vec!["hello world", "test@name", "a/b", "a:b", "a#b", "a$b"];
    for name in invalid_names {
        let result = tm.create_tenant(name);
        assert!(
            matches!(result, Err(TenantError::InvalidName(_))),
            "Expected InvalidName for '{}', got {:?}",
            name,
            result
        );
    }
}

#[test]
fn tenant_validate_name_valid_special_chars() {
    let tm = TenantManager::new();
    // Underscore, hyphen, dot are all valid
    assert!(tm.create_tenant("with_underscore").is_ok());
    assert!(tm.create_tenant("with-hyphen").is_ok());
    assert!(tm.create_tenant("with.dot").is_ok());
    assert!(tm.create_tenant("mix_ed-chars.123").is_ok());
}

#[test]
fn tenant_exists_and_count() {
    let tm = TenantManager::new();
    assert!(!tm.tenant_exists("nonexistent"));
    assert_eq!(tm.tenant_count(), 0);

    tm.create_tenant("exists-test").unwrap();
    assert!(tm.tenant_exists("exists-test"));
    assert_eq!(tm.tenant_count(), 1);
}

// =========================================================================
// store_trait.rs — coverage boost
// =========================================================================

#[test]
fn store_trait_load_persist_node_f64() {
    use grafeo_cognitive::store_trait::{load_node_f64, persist_node_f64};
    use grafeo_core::LpgStore;

    let store = LpgStore::new().expect("LpgStore::new should succeed");
    let node_id = store.create_node(&["TestNode"]);

    // Initially no property
    assert!(load_node_f64(&store, node_id, "_test_score").is_none());

    // Persist a value
    persist_node_f64(&store, node_id, "_test_score", 42.5);

    // Load it back
    let loaded = load_node_f64(&store, node_id, "_test_score");
    assert!(loaded.is_some());
    assert!((loaded.unwrap() - 42.5).abs() < 1e-10);
}

#[test]
fn store_trait_load_persist_edge_f64() {
    use grafeo_cognitive::store_trait::{load_edge_f64, persist_edge_f64};
    use grafeo_core::LpgStore;

    let store = LpgStore::new().expect("LpgStore::new should succeed");
    let n1 = store.create_node(&["A"]);
    let n2 = store.create_node(&["B"]);
    let edge_id = store.create_edge(n1, n2, "CONNECTS");

    // Initially no property
    assert!(load_edge_f64(&store, edge_id, "_test_weight").is_none());

    // Persist a value
    persist_edge_f64(&store, edge_id, "_test_weight", 0.75);

    // Load it back
    let loaded = load_edge_f64(&store, edge_id, "_test_weight");
    assert!(loaded.is_some());
    assert!((loaded.unwrap() - 0.75).abs() < 1e-10);
}

#[test]
fn store_trait_overwrite_node_f64() {
    use grafeo_cognitive::store_trait::{load_node_f64, persist_node_f64};
    use grafeo_core::LpgStore;

    let store = LpgStore::new().expect("LpgStore::new should succeed");
    let node_id = store.create_node(&["TestNode"]);

    persist_node_f64(&store, node_id, "_score", 1.0);
    assert!((load_node_f64(&store, node_id, "_score").unwrap() - 1.0).abs() < 1e-10);

    // Overwrite
    persist_node_f64(&store, node_id, "_score", 2.0);
    assert!((load_node_f64(&store, node_id, "_score").unwrap() - 2.0).abs() < 1e-10);
}

#[test]
fn store_trait_load_nonexistent_node() {
    use grafeo_cognitive::store_trait::load_node_f64;
    use grafeo_core::LpgStore;

    let store = LpgStore::new().expect("LpgStore::new should succeed");
    // Node 99999 does not exist
    let result = load_node_f64(&store, nid(99999), "_score");
    assert!(result.is_none());
}

// =========================================================================
// provenance.rs — coverage boost
// =========================================================================

#[test]
fn provenance_cognitive_event_id_display() {
    let id = CognitiveEventId(42);
    let display = format!("{}", id);
    assert_eq!(display, "cog_event:42");
}

#[test]
fn provenance_cognitive_event_id_display_zero() {
    let id = CognitiveEventId(0);
    assert_eq!(format!("{}", id), "cog_event:0");
}

#[test]
fn provenance_record_synapse_prune() {
    let recorder = ProvenanceRecorder::new();
    let n1 = nid(10);

    recorder.record_synapse_prune(n1, 0.05);

    let history = recorder.get_history(n1);
    assert_eq!(history.len(), 1);
    assert_eq!(history[0].event_type, CognitiveEventType::SynapsePrune);
    assert_eq!(history[0].old_value, Some(0.05));
    assert_eq!(history[0].new_value, None);
    assert_eq!(history[0].trigger_source, "synapse_prune");
    assert_eq!(history[0].target_node, n1);
}

#[test]
fn provenance_record_scar_creation() {
    let recorder = ProvenanceRecorder::new();
    let n1 = nid(20);

    recorder.record_scar_creation(n1, 0.9);

    let history = recorder.get_history(n1);
    assert_eq!(history.len(), 1);
    assert_eq!(history[0].event_type, CognitiveEventType::ScarCreation);
    assert_eq!(history[0].old_value, None);
    assert_eq!(history[0].new_value, Some(0.9));
    assert_eq!(history[0].trigger_source, "scar_creation");
}

#[test]
fn provenance_record_risk_recalc() {
    let recorder = ProvenanceRecorder::new();
    let n1 = nid(30);

    recorder.record_risk_recalc(n1, 0.2, 0.8);

    let history = recorder.get_history(n1);
    assert_eq!(history.len(), 1);
    assert_eq!(history[0].event_type, CognitiveEventType::RiskRecalc);
    assert_eq!(history[0].old_value, Some(0.2));
    assert_eq!(history[0].new_value, Some(0.8));
    assert_eq!(history[0].trigger_source, "risk_recalc");
}

#[test]
fn provenance_debug_impl() {
    let recorder = ProvenanceRecorder::new();
    recorder.record_energy_boost(nid(1), None, 1.0, "test");
    recorder.record_scar_creation(nid(2), 0.5);

    let debug_str = format!("{:?}", recorder);
    assert!(debug_str.contains("ProvenanceRecorder"));
    assert!(debug_str.contains("total_events"));
    assert!(debug_str.contains("nodes_tracked"));
}

#[test]
fn provenance_default_impl() {
    let recorder = ProvenanceRecorder::default();
    assert_eq!(recorder.total_events(), 0);
    assert_eq!(recorder.nodes_with_events(), 0);
}

#[test]
fn provenance_get_derived_from_edges() {
    let recorder = ProvenanceRecorder::new();
    let target = nid(100);
    let sources = vec![nid(1), nid(2), nid(3)];

    recorder.record_derivation(target, sources, "merge");

    let edges = recorder.get_derived_from_edges(target);
    assert_eq!(edges.len(), 3);
    for (t, _src) in &edges {
        assert_eq!(*t, target);
    }

    // Non-existent node should return empty
    let empty = recorder.get_derived_from_edges(nid(999));
    assert!(empty.is_empty());
}

#[test]
fn provenance_multiple_event_types_on_same_node() {
    let recorder = ProvenanceRecorder::new();
    let n1 = nid(50);

    recorder.record_energy_boost(n1, None, 1.0, "test");
    recorder.record_synapse_prune(n1, 0.01);
    recorder.record_scar_creation(n1, 0.7);
    recorder.record_risk_recalc(n1, 0.1, 0.9);

    assert_eq!(recorder.total_events(), 4);
    assert_eq!(recorder.nodes_with_events(), 1);

    let history = recorder.get_history(n1);
    assert_eq!(history.len(), 4);
}

// =========================================================================
// scar.rs — coverage boost
// =========================================================================

#[test]
fn scar_store_with_graph_store() {
    use grafeo_cognitive::store_trait::{PROP_SCAR_COUNT, PROP_SCAR_INTENSITY, load_node_f64};
    use grafeo_core::LpgStore;

    let gs = Arc::new(LpgStore::new().expect("LpgStore::new should succeed"));
    let node_id = gs.create_node(&["ScarNode"]);

    let store = ScarStore::with_graph_store(ScarConfig::default(), gs.clone());

    // Add a scar — should write-through to graph store
    store.add_scar(node_id, 0.8, ScarReason::Rollback);

    // Verify write-through: scar count and intensity should be in the graph store
    let count = load_node_f64(gs.as_ref(), node_id, PROP_SCAR_COUNT);
    assert!(count.is_some());
    assert!((count.unwrap() - 1.0).abs() < 1e-10);

    let intensity = load_node_f64(gs.as_ref(), node_id, PROP_SCAR_INTENSITY);
    assert!(intensity.is_some());
    assert!(intensity.unwrap() > 0.0);
}

#[test]
fn scar_store_persist_scar_summary_multiple_scars() {
    use grafeo_cognitive::store_trait::{PROP_SCAR_COUNT, load_node_f64};
    use grafeo_core::LpgStore;

    let gs = Arc::new(LpgStore::new().expect("LpgStore::new should succeed"));
    let node_id = gs.create_node(&["ScarNode"]);

    let store = ScarStore::with_graph_store(ScarConfig::default(), gs.clone());

    store.add_scar(node_id, 0.5, ScarReason::Invalidation);
    store.add_scar(node_id, 0.3, ScarReason::Error("test error".into()));
    store.add_scar(
        node_id,
        0.9,
        ScarReason::ConstraintViolation("unique".into()),
    );

    let count = load_node_f64(gs.as_ref(), node_id, PROP_SCAR_COUNT);
    assert!(count.is_some());
    assert!((count.unwrap() - 3.0).abs() < 1e-10);
}

#[test]
fn scar_store_persist_after_heal() {
    use grafeo_cognitive::store_trait::{PROP_SCAR_COUNT, load_node_f64};
    use grafeo_core::LpgStore;

    let gs = Arc::new(LpgStore::new().expect("LpgStore::new should succeed"));
    let node_id = gs.create_node(&["ScarNode"]);

    let store = ScarStore::with_graph_store(ScarConfig::default(), gs.clone());

    let scar_id = store.add_scar(node_id, 0.8, ScarReason::Rollback);

    // Verify count is 1
    let count_before = load_node_f64(gs.as_ref(), node_id, PROP_SCAR_COUNT);
    assert!((count_before.unwrap() - 1.0).abs() < 1e-10);

    // Heal the scar
    assert!(store.heal(scar_id));

    // After healing, active count should be 0
    let count_after = load_node_f64(gs.as_ref(), node_id, PROP_SCAR_COUNT);
    assert!((count_after.unwrap() - 0.0).abs() < 1e-10);
}

#[test]
fn scar_store_default_impl() {
    let store = ScarStore::default();
    assert_eq!(store.total_scars(), 0);
    assert_eq!(store.active_scar_count(), 0);
}

#[test]
fn scar_store_debug_impl() {
    let store = ScarStore::new(ScarConfig::default());
    store.add_scar(nid(1), 0.5, ScarReason::Rollback);

    let debug_str = format!("{:?}", store);
    assert!(debug_str.contains("ScarStore"));
    assert!(debug_str.contains("total_scars"));
    assert!(debug_str.contains("config"));
}

#[test]
fn scar_store_without_graph_store_no_panic() {
    // Without a graph store, persist_scar_summary should silently do nothing
    let store = ScarStore::new(ScarConfig::default());
    store.add_scar(nid(1), 0.8, ScarReason::Custom("test".into()));
    assert_eq!(store.total_scars(), 1);
}
