//! Additional coverage tests for the fabric subsystem.

#![cfg(feature = "fabric")]

use grafeo_cognitive::FabricListener;
use grafeo_cognitive::fabric::{FabricScore, FabricStore};
use grafeo_common::types::NodeId;
use grafeo_reactive::{MutationEvent, MutationListener, NodeSnapshot};
use smallvec::smallvec;
use std::sync::Arc;
use std::time::Instant;

fn node_snapshot(id: u64) -> NodeSnapshot {
    NodeSnapshot {
        id: NodeId(id),
        labels: smallvec![],
        properties: vec![],
    }
}

fn make_edge(id: u64, src: u64, dst: u64) -> grafeo_reactive::EdgeSnapshot {
    grafeo_reactive::EdgeSnapshot {
        id: grafeo_common::types::EdgeId::new(id),
        src: NodeId(src),
        dst: NodeId(dst),
        edge_type: arcstr::literal!("KNOWS"),
        properties: vec![],
    }
}

// ---------------------------------------------------------------------------
// FabricScore::last_mutated
// ---------------------------------------------------------------------------

#[test]
fn fabric_score_last_mutated_none_by_default() {
    let score = FabricScore::new();
    assert!(score.last_mutated().is_none());
}

#[test]
fn fabric_score_last_mutated_set_after_churn() {
    let store = FabricStore::new();
    store.update_churn(NodeId(1));
    let score = store.get_fabric_score(NodeId(1));
    assert!(score.last_mutated().is_some());
}

// ---------------------------------------------------------------------------
// FabricStore::update_staleness
// ---------------------------------------------------------------------------

#[test]
fn update_staleness_updates_existing_node() {
    let store = FabricStore::new();
    let nid = NodeId(1);

    // First mutate to create the entry
    store.update_churn(nid);
    // Then wait a tiny bit and update staleness
    std::thread::sleep(std::time::Duration::from_millis(10));
    store.update_staleness(nid);

    let score = store.get_fabric_score(nid);
    // Staleness should be > 0 since some time passed
    assert!(score.staleness > 0.0);
}

#[test]
fn update_staleness_noop_for_untracked_node() {
    let store = FabricStore::new();
    // Should not panic or create an entry
    store.update_staleness(NodeId(999));
    assert!(store.is_empty());
}

#[test]
fn update_staleness_noop_for_node_without_last_mutated() {
    let store = FabricStore::new();
    let nid = NodeId(1);
    // Create entry via knowledge density (no last_mutated set)
    store.set_knowledge_density(nid, 0.5);
    store.update_staleness(nid);
    let score = store.get_fabric_score(nid);
    // staleness should still be 0 (no last_mutated timestamp)
    assert_eq!(score.staleness, 0.0);
}

// ---------------------------------------------------------------------------
// FabricStore::recalculate_risk (single node)
// ---------------------------------------------------------------------------

#[test]
fn recalculate_risk_single_node() {
    let store = FabricStore::new();
    let nid = NodeId(1);

    for _ in 0..5 {
        store.update_churn(nid);
    }
    store.set_gds_metrics(nid, 0.8, 0.6, Some(1));

    // Risk should be 0 before recalculation
    assert_eq!(store.get_fabric_score(nid).risk_score, 0.0);

    store.recalculate_risk(nid, 1.0, 10.0, 1.0);

    let score = store.get_fabric_score(nid);
    assert!(
        score.risk_score > 0.0,
        "risk should be positive after recalculation"
    );
}

#[test]
fn recalculate_risk_nonexistent_node_is_noop() {
    let store = FabricStore::new();
    // Should not panic
    store.recalculate_risk(NodeId(999), 1.0, 1.0, 1.0);
    assert!(store.is_empty());
}

// ---------------------------------------------------------------------------
// FabricStore::update_churn_at (explicit instant)
// ---------------------------------------------------------------------------

#[test]
fn update_churn_at_uses_specified_time() {
    let store = FabricStore::new();
    let nid = NodeId(1);
    let now = Instant::now();

    store.update_churn_at(nid, now);
    let score = store.get_fabric_score(nid);
    assert_eq!(score.churn_score, 1.0);
    assert!(score.last_mutated().is_some());
}

// ---------------------------------------------------------------------------
// FabricStore::get_hotspots empty store
// ---------------------------------------------------------------------------

#[test]
fn get_hotspots_empty_store() {
    let store = FabricStore::new();
    let hotspots = store.get_hotspots(10);
    assert!(hotspots.is_empty());
}

// ---------------------------------------------------------------------------
// FabricStore::get_risk_zones empty and zero threshold
// ---------------------------------------------------------------------------

#[test]
fn get_risk_zones_empty_store() {
    let store = FabricStore::new();
    let zones = store.get_risk_zones(0.0);
    assert!(zones.is_empty());
}

#[test]
fn get_risk_zones_zero_threshold_returns_all_with_risk() {
    let store = FabricStore::new();
    for _ in 0..5 {
        store.update_churn(NodeId(1));
    }
    store.set_gds_metrics(NodeId(1), 1.0, 1.0, None);
    store.recalculate_all_risks();

    let zones = store.get_risk_zones(0.0);
    // Node has risk > 0 (churn > 0, pagerank > 0, betweenness > 0, density = 0)
    assert!(!zones.is_empty());
}

// ---------------------------------------------------------------------------
// FabricListener: name()
// ---------------------------------------------------------------------------

#[test]
fn listener_name() {
    let store = Arc::new(FabricStore::new());
    let listener = FabricListener::new(store);
    assert_eq!(listener.name(), "cognitive:fabric");
}

// ---------------------------------------------------------------------------
// FabricListener: store() accessor
// ---------------------------------------------------------------------------

#[test]
fn listener_store_accessor() {
    let store = Arc::new(FabricStore::new());
    let listener = FabricListener::new(Arc::clone(&store));
    assert!(Arc::ptr_eq(listener.store(), &store));
}

// ---------------------------------------------------------------------------
// FabricListener: on_batch with edge events
// ---------------------------------------------------------------------------

#[tokio::test]
async fn listener_on_batch_with_edge_created() {
    let store = Arc::new(FabricStore::new());
    let listener = FabricListener::new(Arc::clone(&store));

    let events = vec![MutationEvent::EdgeCreated {
        edge: make_edge(1, 10, 20),
    }];
    listener.on_batch(&events).await;

    assert_eq!(store.get_fabric_score(NodeId(10)).churn_score, 1.0);
    assert_eq!(store.get_fabric_score(NodeId(20)).churn_score, 1.0);
}

#[tokio::test]
async fn listener_on_batch_with_edge_deleted() {
    let store = Arc::new(FabricStore::new());
    let listener = FabricListener::new(Arc::clone(&store));

    let events = vec![MutationEvent::EdgeDeleted {
        edge: make_edge(1, 30, 40),
    }];
    listener.on_batch(&events).await;

    assert_eq!(store.get_fabric_score(NodeId(30)).churn_score, 1.0);
    assert_eq!(store.get_fabric_score(NodeId(40)).churn_score, 1.0);
}

#[tokio::test]
async fn listener_on_batch_with_node_deleted() {
    let store = Arc::new(FabricStore::new());
    let listener = FabricListener::new(Arc::clone(&store));

    let events = vec![MutationEvent::NodeDeleted {
        node: node_snapshot(55),
    }];
    listener.on_batch(&events).await;

    assert_eq!(store.get_fabric_score(NodeId(55)).churn_score, 1.0);
}

// ---------------------------------------------------------------------------
// FabricStore: risk with scar_intensity but zero base metrics
// ---------------------------------------------------------------------------

#[test]
fn risk_score_scar_only_no_base_risk() {
    let store = FabricStore::new();
    let nid = NodeId(1);
    // Only scar intensity, no churn/pagerank/betweenness
    store.set_scar_intensity(nid, 1.0);
    store.recalculate_all_risks();

    let score = store.get_fabric_score(nid);
    // base_risk = 0 (no churn/pagerank/betweenness), scar_boost = 0.5 * (1.0/1.0) = 0.5
    assert!(
        (score.risk_score - 0.5).abs() < 0.01,
        "expected ~0.5 from scar alone, got {}",
        score.risk_score
    );
}

// ---------------------------------------------------------------------------
// FabricStore: community_ids with duplicates
// ---------------------------------------------------------------------------

#[test]
fn community_ids_deduplicates() {
    let store = FabricStore::new();
    for i in 0..10 {
        store.set_gds_metrics(NodeId(i), 0.1, 0.1, Some(1));
    }
    let ids = store.community_ids();
    assert_eq!(ids, vec![1]);
}
