//! Integration tests for the knowledge fabric subsystem.

#![cfg(feature = "fabric")]

use grafeo_cognitive::FabricListener;
use grafeo_cognitive::fabric::{FabricScore, FabricStore};
#[cfg(feature = "gds-refresh")]
use grafeo_cognitive::gds_refresh::{GdsRefreshConfig, GdsRefreshScheduler};
use grafeo_common::types::NodeId;
use grafeo_reactive::{MutationEvent, MutationListener, NodeSnapshot};
use smallvec::smallvec;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn node_snapshot(id: u64) -> NodeSnapshot {
    NodeSnapshot {
        id: NodeId(id),
        labels: smallvec![],
        properties: vec![],
    }
}

fn node_created(id: u64) -> MutationEvent {
    MutationEvent::NodeCreated {
        node: node_snapshot(id),
    }
}

fn node_updated(id: u64) -> MutationEvent {
    MutationEvent::NodeUpdated {
        before: node_snapshot(id),
        after: node_snapshot(id),
    }
}

// ---------------------------------------------------------------------------
// FabricScore default
// ---------------------------------------------------------------------------

#[test]
fn fabric_score_default_all_zeros() {
    let score = FabricScore::default();
    assert_eq!(score.churn_score, 0.0);
    assert_eq!(score.knowledge_density, 0.0);
    assert_eq!(score.staleness, 0.0);
    assert_eq!(score.risk_score, 0.0);
    assert_eq!(score.pagerank, 0.0);
    assert_eq!(score.betweenness, 0.0);
    assert_eq!(score.community_id, None);
}

#[test]
fn fabric_score_new_equals_default() {
    assert_eq!(FabricScore::new(), FabricScore::default());
}

// ---------------------------------------------------------------------------
// FabricStore — churn incremental
// ---------------------------------------------------------------------------

#[test]
fn update_churn_increments() {
    let store = FabricStore::new();
    let nid = NodeId(1);

    store.update_churn(nid);
    assert_eq!(store.get_fabric_score(nid).churn_score, 1.0);

    store.update_churn(nid);
    assert_eq!(store.get_fabric_score(nid).churn_score, 2.0);

    store.update_churn(nid);
    assert_eq!(store.get_fabric_score(nid).churn_score, 3.0);
}

#[test]
fn update_churn_resets_staleness() {
    let store = FabricStore::new();
    let nid = NodeId(1);

    store.update_churn(nid);
    let score = store.get_fabric_score(nid);
    // Staleness should be very close to 0 (just mutated)
    assert!(
        score.staleness < 1.0,
        "staleness should be ~0, got {}",
        score.staleness
    );
}

#[test]
fn untracked_node_returns_default() {
    let store = FabricStore::new();
    let score = store.get_fabric_score(NodeId(999));
    assert_eq!(score, FabricScore::default());
}

// ---------------------------------------------------------------------------
// FabricStore — hotspots
// ---------------------------------------------------------------------------

#[test]
fn get_hotspots_sorted_by_churn_desc() {
    let store = FabricStore::new();

    // Node 1: 3 mutations
    for _ in 0..3 {
        store.update_churn(NodeId(1));
    }
    // Node 2: 5 mutations
    for _ in 0..5 {
        store.update_churn(NodeId(2));
    }
    // Node 3: 1 mutation
    store.update_churn(NodeId(3));

    let hotspots = store.get_hotspots(10);
    assert_eq!(hotspots.len(), 3);
    assert_eq!(hotspots[0].0, NodeId(2)); // highest churn
    assert_eq!(hotspots[0].1.churn_score, 5.0);
    assert_eq!(hotspots[1].0, NodeId(1));
    assert_eq!(hotspots[1].1.churn_score, 3.0);
    assert_eq!(hotspots[2].0, NodeId(3));
    assert_eq!(hotspots[2].1.churn_score, 1.0);
}

#[test]
fn get_hotspots_truncates_to_top_n() {
    let store = FabricStore::new();
    for i in 0..10 {
        for _ in 0..(i + 1) {
            store.update_churn(NodeId(i));
        }
    }
    let hotspots = store.get_hotspots(3);
    assert_eq!(hotspots.len(), 3);
    // Top 3 should be nodes 9, 8, 7
    assert_eq!(hotspots[0].1.churn_score, 10.0);
    assert_eq!(hotspots[1].1.churn_score, 9.0);
    assert_eq!(hotspots[2].1.churn_score, 8.0);
}

// ---------------------------------------------------------------------------
// FabricStore — risk zones
// ---------------------------------------------------------------------------

#[test]
fn get_risk_zones_filters_by_min_risk() {
    let store = FabricStore::new();

    // Set up nodes with different risk profiles
    let n1 = NodeId(1);
    let n2 = NodeId(2);
    let n3 = NodeId(3);

    // Node 1: high risk scenario
    for _ in 0..10 {
        store.update_churn(n1);
    }
    store.set_gds_metrics(n1, 0.9, 0.8, Some(1));
    // Low knowledge density (default 0.0) → high knowledge gap

    // Node 2: low risk (good documentation)
    store.update_churn(n2);
    store.set_knowledge_density(n2, 0.95);
    store.set_gds_metrics(n2, 0.1, 0.1, Some(1));

    // Node 3: medium risk
    for _ in 0..5 {
        store.update_churn(n3);
    }
    store.set_gds_metrics(n3, 0.5, 0.4, Some(2));

    // Recalculate all risk scores
    store.recalculate_all_risks();

    // Node 1 should have the highest risk
    let s1 = store.get_fabric_score(n1);
    assert!(
        s1.risk_score > 0.0,
        "node 1 risk should be > 0, got {}",
        s1.risk_score
    );

    // Node 2 should have very low risk (high density = low knowledge gap)
    let s2 = store.get_fabric_score(n2);
    assert!(
        s2.risk_score < s1.risk_score,
        "node 2 risk ({}) should be < node 1 risk ({})",
        s2.risk_score,
        s1.risk_score
    );

    // Risk zones with high threshold should only include high-risk nodes
    let zones_high = store.get_risk_zones(s1.risk_score);
    assert!(
        zones_high.len() >= 1,
        "should have at least 1 high-risk node"
    );
    assert!(
        zones_high
            .iter()
            .all(|(_, s)| s.risk_score >= s1.risk_score)
    );
}

// ---------------------------------------------------------------------------
// Risk score — composite formula
// ---------------------------------------------------------------------------

#[test]
fn risk_score_high_pr_high_churn_low_density() {
    let store = FabricStore::new();
    let nid = NodeId(1);

    // High churn
    for _ in 0..100 {
        store.update_churn(nid);
    }
    // High pagerank, high betweenness, low density (default 0.0)
    store.set_gds_metrics(nid, 1.0, 1.0, Some(1));

    store.recalculate_all_risks();
    let score = store.get_fabric_score(nid);

    // With max values and density=0 → risk = 1.0 × 1.0 × (1-0) × 1.0 = 1.0
    assert!(
        (score.risk_score - 1.0).abs() < 1e-10,
        "expected risk ~1.0, got {}",
        score.risk_score
    );
}

#[test]
fn risk_score_zero_when_high_density() {
    let store = FabricStore::new();
    let nid = NodeId(1);

    for _ in 0..10 {
        store.update_churn(nid);
    }
    store.set_knowledge_density(nid, 1.0); // Fully documented
    store.set_gds_metrics(nid, 1.0, 1.0, Some(1));

    store.recalculate_all_risks();
    let score = store.get_fabric_score(nid);

    // density=1.0 → knowledge_gap = 0 → risk = 0
    assert!(
        score.risk_score.abs() < 1e-10,
        "expected risk ~0.0, got {}",
        score.risk_score
    );
}

#[test]
fn risk_score_zero_when_no_pagerank() {
    let store = FabricStore::new();
    let nid = NodeId(1);

    for _ in 0..10 {
        store.update_churn(nid);
    }
    // pagerank = 0 → normalized = 0 → risk = 0
    store.set_gds_metrics(nid, 0.0, 1.0, Some(1));

    store.recalculate_all_risks();
    let score = store.get_fabric_score(nid);

    assert!(
        score.risk_score.abs() < 1e-10,
        "expected risk ~0.0, got {}",
        score.risk_score
    );
}

// ---------------------------------------------------------------------------
// FabricListener — MutationListener integration
// ---------------------------------------------------------------------------

#[tokio::test]
async fn listener_on_event_increments_churn() {
    let store = Arc::new(FabricStore::new());
    let listener = FabricListener::new(Arc::clone(&store));

    listener.on_event(&node_created(1)).await;
    assert_eq!(store.get_fabric_score(NodeId(1)).churn_score, 1.0);

    listener.on_event(&node_updated(1)).await;
    assert_eq!(store.get_fabric_score(NodeId(1)).churn_score, 2.0);
}

#[tokio::test]
async fn listener_on_batch_updates_all_nodes() {
    let store = Arc::new(FabricStore::new());
    let listener = FabricListener::new(Arc::clone(&store));

    let events = vec![
        node_created(1),
        node_created(2),
        node_updated(1),
        node_created(3),
    ];

    listener.on_batch(&events).await;

    // Node 1 appears in 2 events but on_batch deduplicates → churn = 1
    // (only incremented once per batch due to HashSet dedup)
    assert_eq!(store.get_fabric_score(NodeId(1)).churn_score, 1.0);
    assert_eq!(store.get_fabric_score(NodeId(2)).churn_score, 1.0);
    assert_eq!(store.get_fabric_score(NodeId(3)).churn_score, 1.0);
}

#[tokio::test]
async fn listener_on_batch_resets_staleness() {
    let store = Arc::new(FabricStore::new());
    let listener = FabricListener::new(Arc::clone(&store));

    listener.on_batch(&[node_created(42)]).await;

    let score = store.get_fabric_score(NodeId(42));
    assert!(
        score.staleness < 1.0,
        "staleness should be ~0, got {}",
        score.staleness
    );
    assert!(score.last_mutated().is_some());
}

// ---------------------------------------------------------------------------
// GdsRefreshScheduler — configuration and mutation tracking
// ---------------------------------------------------------------------------

#[cfg(feature = "gds-refresh")]
#[test]
fn gds_refresh_config_defaults() {
    let config = GdsRefreshConfig::default();
    assert_eq!(config.refresh_interval, std::time::Duration::from_secs(300));
    assert_eq!(config.mutation_threshold, 1000);
}

#[cfg(feature = "gds-refresh")]
#[test]
fn gds_refresh_mutation_threshold() {
    let store = Arc::new(FabricStore::new());
    let scheduler = GdsRefreshScheduler::new(store, GdsRefreshConfig::default());

    // Should not trigger until threshold reached
    assert!(!scheduler.record_mutations(500));
    assert_eq!(scheduler.mutations_since_refresh(), 500);

    assert!(!scheduler.record_mutations(499));
    assert_eq!(scheduler.mutations_since_refresh(), 999);

    // 1000th mutation triggers refresh
    assert!(scheduler.record_mutations(1));
    assert_eq!(scheduler.mutations_since_refresh(), 1000);
}

#[cfg(feature = "gds-refresh")]
#[test]
fn gds_refresh_custom_threshold() {
    let store = Arc::new(FabricStore::new());
    let config = GdsRefreshConfig {
        mutation_threshold: 10,
        ..Default::default()
    };
    let scheduler = GdsRefreshScheduler::new(store, config);

    assert!(!scheduler.record_mutations(5));
    assert!(scheduler.record_mutations(5));
}

// ---------------------------------------------------------------------------
// FabricStore — thread safety (basic check)
// ---------------------------------------------------------------------------

#[test]
fn fabric_store_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<FabricStore>();
    assert_send_sync::<FabricListener>();
}

// ---------------------------------------------------------------------------
// FabricStore — len and is_empty
// ---------------------------------------------------------------------------

#[test]
fn store_len_and_is_empty() {
    let store = FabricStore::new();
    assert!(store.is_empty());
    assert_eq!(store.len(), 0);

    store.update_churn(NodeId(1));
    assert!(!store.is_empty());
    assert_eq!(store.len(), 1);

    store.update_churn(NodeId(2));
    assert_eq!(store.len(), 2);
}
