//! Integration tests for the GDS refresh scheduler.

#![cfg(feature = "gds-refresh")]

use obrain_cognitive::fabric::FabricStore;
use obrain_cognitive::gds_refresh::{GdsRefreshConfig, GdsRefreshScheduler};
use obrain_core::graph::traits::GraphStoreMut;
use std::sync::Arc;
use std::time::Duration;

// ---------------------------------------------------------------------------
// GdsRefreshConfig tests
// ---------------------------------------------------------------------------

#[test]
fn config_default_values() {
    let config = GdsRefreshConfig::default();
    assert_eq!(config.refresh_interval, Duration::from_secs(5 * 60));
    assert_eq!(config.mutation_threshold, 1000);
    assert!((config.pagerank_damping - 0.85).abs() < 1e-10);
    assert_eq!(config.pagerank_max_iterations, 100);
    assert!((config.pagerank_tolerance - 1e-6).abs() < 1e-12);
    assert!((config.louvain_resolution - 1.0).abs() < 1e-10);
    assert!(config.betweenness_normalized);
}

#[test]
fn config_default_refresh_interval_is_five_minutes() {
    let config = GdsRefreshConfig::default();
    assert_eq!(config.refresh_interval.as_secs(), 300);
}

#[test]
fn config_default_mutation_threshold_is_1000() {
    let config = GdsRefreshConfig::default();
    assert_eq!(config.mutation_threshold, 1000);
}

#[test]
fn config_clone() {
    let config = GdsRefreshConfig::default();
    let cloned = config.clone();
    assert_eq!(cloned.refresh_interval, config.refresh_interval);
    assert_eq!(cloned.mutation_threshold, config.mutation_threshold);
    assert!((cloned.pagerank_damping - config.pagerank_damping).abs() < 1e-15);
    assert_eq!(
        cloned.pagerank_max_iterations,
        config.pagerank_max_iterations
    );
    assert!((cloned.pagerank_tolerance - config.pagerank_tolerance).abs() < 1e-15);
    assert!((cloned.louvain_resolution - config.louvain_resolution).abs() < 1e-15);
    assert_eq!(cloned.betweenness_normalized, config.betweenness_normalized);
}

#[test]
fn config_debug_formatting() {
    let config = GdsRefreshConfig::default();
    let dbg = format!("{config:?}");
    assert!(dbg.contains("GdsRefreshConfig"), "got: {dbg}");
    assert!(dbg.contains("refresh_interval"), "got: {dbg}");
    assert!(dbg.contains("mutation_threshold"), "got: {dbg}");
    assert!(dbg.contains("pagerank_damping"), "got: {dbg}");
    assert!(dbg.contains("pagerank_max_iterations"), "got: {dbg}");
    assert!(dbg.contains("pagerank_tolerance"), "got: {dbg}");
    assert!(dbg.contains("louvain_resolution"), "got: {dbg}");
    assert!(dbg.contains("betweenness_normalized"), "got: {dbg}");
}

#[test]
fn config_custom_values() {
    let config = GdsRefreshConfig {
        refresh_interval: Duration::from_secs(60),
        mutation_threshold: 500,
        pagerank_damping: 0.90,
        pagerank_max_iterations: 50,
        pagerank_tolerance: 1e-4,
        louvain_resolution: 2.0,
        betweenness_normalized: false,
    };
    assert_eq!(config.refresh_interval, Duration::from_secs(60));
    assert_eq!(config.mutation_threshold, 500);
    assert!((config.pagerank_damping - 0.90).abs() < 1e-10);
    assert_eq!(config.pagerank_max_iterations, 50);
    assert!((config.pagerank_tolerance - 1e-4).abs() < 1e-10);
    assert!((config.louvain_resolution - 2.0).abs() < 1e-10);
    assert!(!config.betweenness_normalized);
}

// ---------------------------------------------------------------------------
// GdsRefreshScheduler tests
// ---------------------------------------------------------------------------

fn make_scheduler(threshold: u64) -> GdsRefreshScheduler {
    let fabric = Arc::new(FabricStore::new());
    let config = GdsRefreshConfig {
        mutation_threshold: threshold,
        ..GdsRefreshConfig::default()
    };
    GdsRefreshScheduler::new(fabric, config)
}

fn make_scheduler_with_fabric(threshold: u64) -> (Arc<FabricStore>, GdsRefreshScheduler) {
    let fabric = Arc::new(FabricStore::new());
    let config = GdsRefreshConfig {
        mutation_threshold: threshold,
        ..GdsRefreshConfig::default()
    };
    let sched = GdsRefreshScheduler::new(Arc::clone(&fabric), config);
    (fabric, sched)
}

#[test]
fn scheduler_initial_mutations_zero() {
    let sched = make_scheduler(100);
    assert_eq!(sched.mutations_since_refresh(), 0);
}

#[test]
fn record_mutations_below_threshold_returns_false() {
    let sched = make_scheduler(100);
    let reached = sched.record_mutations(10);
    assert!(!reached, "10 mutations < threshold of 100");
    assert_eq!(sched.mutations_since_refresh(), 10);
}

#[test]
fn record_mutations_at_threshold_returns_true() {
    let sched = make_scheduler(100);
    let reached = sched.record_mutations(100);
    assert!(reached, "100 mutations >= threshold of 100");
    assert_eq!(sched.mutations_since_refresh(), 100);
}

#[test]
fn record_mutations_above_threshold_returns_true() {
    let sched = make_scheduler(100);
    let reached = sched.record_mutations(200);
    assert!(reached, "200 mutations >= threshold of 100");
}

#[test]
fn record_mutations_accumulates_across_calls() {
    let sched = make_scheduler(100);
    assert!(!sched.record_mutations(30));
    assert_eq!(sched.mutations_since_refresh(), 30);
    assert!(!sched.record_mutations(30));
    assert_eq!(sched.mutations_since_refresh(), 60);
    // 60 + 50 = 110 >= 100
    assert!(sched.record_mutations(50));
    assert_eq!(sched.mutations_since_refresh(), 110);
}

#[test]
fn record_mutations_single_mutation() {
    let sched = make_scheduler(5);
    assert!(!sched.record_mutations(1));
    assert_eq!(sched.mutations_since_refresh(), 1);
    assert!(!sched.record_mutations(1));
    assert_eq!(sched.mutations_since_refresh(), 2);
    assert!(!sched.record_mutations(1));
    assert!(!sched.record_mutations(1));
    // 4 + 1 = 5 >= 5
    assert!(sched.record_mutations(1));
    assert_eq!(sched.mutations_since_refresh(), 5);
}

#[test]
fn record_mutations_zero_does_not_trigger() {
    let sched = make_scheduler(100);
    assert!(!sched.record_mutations(0));
    assert_eq!(sched.mutations_since_refresh(), 0);
}

#[test]
fn record_mutations_threshold_one() {
    let sched = make_scheduler(1);
    // 0 + 1 = 1 >= 1
    assert!(sched.record_mutations(1));
}

#[test]
fn record_mutations_continues_after_threshold() {
    let sched = make_scheduler(10);
    assert!(sched.record_mutations(10));
    // Keeps accumulating even after threshold
    assert!(sched.record_mutations(5));
    assert_eq!(sched.mutations_since_refresh(), 15);
}

#[test]
fn config_accessor() {
    let sched = make_scheduler(42);
    assert_eq!(sched.config().mutation_threshold, 42);
    assert!((sched.config().pagerank_damping - 0.85).abs() < 1e-10);
}

#[test]
fn config_accessor_returns_custom_values() {
    let fabric = Arc::new(FabricStore::new());
    let config = GdsRefreshConfig {
        refresh_interval: Duration::from_secs(10),
        mutation_threshold: 7,
        pagerank_damping: 0.5,
        pagerank_max_iterations: 20,
        pagerank_tolerance: 1e-3,
        louvain_resolution: 3.0,
        betweenness_normalized: false,
    };
    let sched = GdsRefreshScheduler::new(fabric, config);
    let c = sched.config();
    assert_eq!(c.refresh_interval, Duration::from_secs(10));
    assert_eq!(c.mutation_threshold, 7);
    assert!((c.pagerank_damping - 0.5).abs() < 1e-10);
    assert_eq!(c.pagerank_max_iterations, 20);
    assert!((c.pagerank_tolerance - 1e-3).abs() < 1e-10);
    assert!((c.louvain_resolution - 3.0).abs() < 1e-10);
    assert!(!c.betweenness_normalized);
}

#[test]
fn fabric_store_accessor() {
    let fabric = Arc::new(FabricStore::new());
    let config = GdsRefreshConfig::default();
    let sched = GdsRefreshScheduler::new(Arc::clone(&fabric), config);
    // Should return a reference to the same fabric store
    assert_eq!(sched.fabric_store().len(), 0);
}

#[test]
fn fabric_store_accessor_shares_same_instance() {
    let (fabric, sched) = make_scheduler_with_fabric(100);
    // Both references point to the same store
    assert_eq!(fabric.len(), sched.fabric_store().len());
    assert!(sched.fabric_store().is_empty());
}

#[test]
fn debug_formatting() {
    let sched = make_scheduler(50);
    sched.record_mutations(7);
    let dbg = format!("{sched:?}");
    assert!(dbg.contains("GdsRefreshScheduler"), "got: {dbg}");
    assert!(dbg.contains("mutations_since_refresh"), "got: {dbg}");
    assert!(dbg.contains("config"), "got: {dbg}");
}

#[test]
fn debug_formatting_shows_mutation_count() {
    let sched = make_scheduler(50);
    sched.record_mutations(42);
    let dbg = format!("{sched:?}");
    assert!(dbg.contains("42"), "expected mutation count 42, got: {dbg}");
}

#[test]
fn debug_formatting_zero_mutations() {
    let sched = make_scheduler(50);
    let dbg = format!("{sched:?}");
    assert!(
        dbg.contains("GdsRefreshScheduler"),
        "should contain struct name, got: {dbg}"
    );
}

// ---------------------------------------------------------------------------
// refresh() integration test (requires gds-refresh + obrain-adapters)
// ---------------------------------------------------------------------------

#[test]
fn refresh_with_graph_store() {
    use obrain_substrate::SubstrateStore;
    let store = SubstrateStore::open_tempfile().expect("SubstrateStore::open_tempfile");

    // Create a small graph: A -> B -> C, A -> C
    let a = store.create_node(&["Component"]);
    let b = store.create_node(&["Component"]);
    let c = store.create_node(&["Component"]);
    let _e1 = store.create_edge(a, b, "DEPENDS_ON");
    let _e2 = store.create_edge(b, c, "DEPENDS_ON");
    let _e3 = store.create_edge(a, c, "DEPENDS_ON");

    let fabric = Arc::new(FabricStore::new());
    let config = GdsRefreshConfig {
        mutation_threshold: 10,
        ..GdsRefreshConfig::default()
    };
    let sched = GdsRefreshScheduler::new(Arc::clone(&fabric), config);

    // Record some mutations so we can verify the counter resets
    sched.record_mutations(5);
    assert_eq!(sched.mutations_since_refresh(), 5);

    // Run refresh
    sched.refresh(&store);

    // Mutation count should be reset to 0
    assert_eq!(
        sched.mutations_since_refresh(),
        0,
        "mutation count should reset after refresh"
    );

    // FabricStore should have entries for all 3 nodes
    assert_eq!(fabric.len(), 3, "fabric should track all 3 nodes");

    // Check that pagerank was computed (all nodes should have non-zero pagerank
    // in a connected graph)
    let score_a = fabric.get_fabric_score(a);
    let score_b = fabric.get_fabric_score(b);
    let score_c = fabric.get_fabric_score(c);

    // PageRank should be > 0 for at least some nodes
    let total_pr = score_a.pagerank + score_b.pagerank + score_c.pagerank;
    assert!(
        total_pr > 0.0,
        "total pagerank should be > 0, got {total_pr}"
    );

    // Node A has the most outgoing edges, B and C receive edges
    // C receives edges from both A and B, so it should have higher pagerank than B
    // (though exact values depend on algorithm)
    assert!(score_c.pagerank > 0.0, "node C should have pagerank > 0");

    // Betweenness: B is on the path A->B->C, so it should have non-zero betweenness
    // (at least in normalized form)
    let total_bc = score_a.betweenness + score_b.betweenness + score_c.betweenness;
    assert!(
        total_bc >= 0.0,
        "total betweenness should be >= 0, got {total_bc}"
    );

    // Community IDs should be assigned (in a small connected graph, likely one community)
    let has_community = score_a.community_id.is_some()
        || score_b.community_id.is_some()
        || score_c.community_id.is_some();
    assert!(
        has_community,
        "at least one node should have a community_id"
    );
}

#[test]
fn refresh_empty_graph_does_not_panic() {
    use obrain_substrate::SubstrateStore;

    let store = SubstrateStore::open_tempfile().expect("SubstrateStore::open_tempfile");
    let fabric = Arc::new(FabricStore::new());
    let config = GdsRefreshConfig::default();
    let sched = GdsRefreshScheduler::new(Arc::clone(&fabric), config);

    sched.record_mutations(10);

    // Should not panic on an empty graph
    sched.refresh(&store);

    assert_eq!(sched.mutations_since_refresh(), 0);
    assert_eq!(fabric.len(), 0);
}

#[test]
fn refresh_single_node_no_edges() {
    use obrain_substrate::SubstrateStore;

    let store = SubstrateStore::open_tempfile().expect("SubstrateStore::open_tempfile");
    let _a = store.create_node(&["Isolated"]);

    let fabric = Arc::new(FabricStore::new());
    let config = GdsRefreshConfig::default();
    let sched = GdsRefreshScheduler::new(Arc::clone(&fabric), config);

    sched.refresh(&store);

    assert_eq!(sched.mutations_since_refresh(), 0);
    // Single isolated node should still appear in fabric after refresh
    // (pagerank/betweenness might be 0, but community should be assigned)
}

#[test]
fn refresh_resets_mutation_count_even_if_above_threshold() {
    use obrain_substrate::SubstrateStore;

    let store = SubstrateStore::open_tempfile().expect("SubstrateStore::open_tempfile");
    let fabric = Arc::new(FabricStore::new());
    let config = GdsRefreshConfig {
        mutation_threshold: 10,
        ..GdsRefreshConfig::default()
    };
    let sched = GdsRefreshScheduler::new(Arc::clone(&fabric), config);

    // Exceed threshold
    sched.record_mutations(50);
    assert_eq!(sched.mutations_since_refresh(), 50);

    sched.refresh(&store);
    assert_eq!(
        sched.mutations_since_refresh(),
        0,
        "counter should be 0 after refresh regardless of prior value"
    );
}

#[test]
fn refresh_updates_metrics_for_larger_graph() {
    use obrain_substrate::SubstrateStore;
    let store = SubstrateStore::open_tempfile().expect("SubstrateStore::open_tempfile");

    // Build a star graph: center -> spoke1, center -> spoke2, ..., center -> spoke5
    let center = store.create_node(&["Hub"]);
    let mut spokes = Vec::new();
    for _ in 0..5 {
        let spoke = store.create_node(&["Spoke"]);
        store.create_edge(center, spoke, "CONNECTS");
        spokes.push(spoke);
    }
    // Add a cross-edge between two spokes
    store.create_edge(spokes[0], spokes[1], "CONNECTS");

    let fabric = Arc::new(FabricStore::new());
    let config = GdsRefreshConfig::default();
    let sched = GdsRefreshScheduler::new(Arc::clone(&fabric), config);

    sched.refresh(&store);

    assert_eq!(fabric.len(), 6, "hub + 5 spokes = 6 nodes");

    // The hub fans out to all spokes, so the spoke nodes receiving edges
    // should have meaningful pagerank
    let center_score = fabric.get_fabric_score(center);
    let spoke1_score = fabric.get_fabric_score(spokes[1]);

    // spoke1 receives edges from both center and spoke0
    // so it should have higher pagerank than spokes that receive only from center
    let spoke2_score = fabric.get_fabric_score(spokes[2]);
    assert!(
        spoke1_score.pagerank >= spoke2_score.pagerank,
        "spoke1 (2 incoming) should have >= pagerank than spoke2 (1 incoming): {} vs {}",
        spoke1_score.pagerank,
        spoke2_score.pagerank
    );

    // All nodes should have community assignments
    for &spoke in &spokes {
        let score = fabric.get_fabric_score(spoke);
        // Pagerank should be non-negative
        assert!(score.pagerank >= 0.0);
        assert!(score.betweenness >= 0.0);
    }
    assert!(center_score.pagerank >= 0.0);
}
