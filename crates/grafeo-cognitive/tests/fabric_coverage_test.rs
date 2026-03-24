//! Coverage-boosting tests for the knowledge fabric subsystem.

#![cfg(feature = "fabric")]

use grafeo_cognitive::fabric::FabricStore;
use grafeo_cognitive::FabricListener;
use grafeo_common::types::NodeId;
use grafeo_reactive::{MutationEvent, MutationListener, NodeSnapshot};
use smallvec::smallvec;
use std::sync::Arc;
use std::time::Instant;

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

fn make_edge(id: u64, src: u64, dst: u64) -> grafeo_reactive::EdgeSnapshot {
    grafeo_reactive::EdgeSnapshot {
        id: grafeo_common::types::EdgeId::new(id),
        src: NodeId(src),
        dst: NodeId(dst),
        edge_type: arcstr::literal!("RELATES_TO"),
        properties: vec![],
    }
}

// ---------------------------------------------------------------------------
// recalculate_risk — individual node with provided maxima
// ---------------------------------------------------------------------------

#[test]
fn recalculate_risk_with_custom_maxima() {
    let store = FabricStore::new();
    let n = NodeId(1);

    for _ in 0..8 {
        store.update_churn(n);
    }
    store.set_gds_metrics(n, 0.6, 0.4, Some(1));
    // knowledge_density = 0.0 (default) → knowledge_gap = 1.0

    // Provide custom global maxima for normalization
    store.recalculate_risk(n, 1.0, 10.0, 1.0);

    let score = store.get_fabric_score(n);
    // pr = 0.6/1.0 = 0.6, churn = 8/10 = 0.8, gap = 1.0, btwn = 0.4/1.0 = 0.4
    // base_risk = 0.6 * 0.8 * 1.0 * 0.4 = 0.192, scar_boost = 0 → 0.192
    assert!(
        (score.risk_score - 0.192).abs() < 0.01,
        "expected ~0.192, got {}",
        score.risk_score
    );
}

#[test]
fn recalculate_risk_with_different_maxima_changes_result() {
    let store = FabricStore::new();
    let n = NodeId(1);

    for _ in 0..4 {
        store.update_churn(n);
    }
    store.set_gds_metrics(n, 0.5, 0.5, None);

    // With max 1.0 for all
    store.recalculate_risk(n, 1.0, 10.0, 1.0);
    let risk_a = store.get_fabric_score(n).risk_score;

    // With max 0.5 → normalized values are higher (clamped to 1.0)
    store.recalculate_risk(n, 0.5, 4.0, 0.5);
    let risk_b = store.get_fabric_score(n).risk_score;

    assert!(
        risk_b > risk_a,
        "lower maxima should yield higher normalized risk: {} vs {}",
        risk_b,
        risk_a
    );
}

// ---------------------------------------------------------------------------
// recalculate_all_risks — batch across multiple nodes
// ---------------------------------------------------------------------------

#[test]
fn recalculate_all_risks_uses_global_maxima() {
    let store = FabricStore::new();

    // Node 1: highest churn and metrics
    for _ in 0..10 {
        store.update_churn(NodeId(1));
    }
    store.set_gds_metrics(NodeId(1), 1.0, 1.0, Some(1));

    // Node 2: half the metrics
    for _ in 0..5 {
        store.update_churn(NodeId(2));
    }
    store.set_gds_metrics(NodeId(2), 0.5, 0.5, Some(1));

    // Node 3: minimal metrics
    store.update_churn(NodeId(3));
    store.set_gds_metrics(NodeId(3), 0.1, 0.1, Some(2));

    store.recalculate_all_risks();

    let s1 = store.get_fabric_score(NodeId(1));
    let s2 = store.get_fabric_score(NodeId(2));
    let s3 = store.get_fabric_score(NodeId(3));

    // Node 1 should have maximum base risk (all normalized to 1.0)
    assert!(
        (s1.risk_score - 1.0).abs() < 0.01,
        "node 1 should have ~1.0 risk, got {}",
        s1.risk_score
    );

    // Node 2 risk should be between node 1 and node 3
    assert!(
        s2.risk_score < s1.risk_score && s2.risk_score > s3.risk_score,
        "expected s3 < s2 < s1: s1={}, s2={}, s3={}",
        s1.risk_score,
        s2.risk_score,
        s3.risk_score
    );
}

#[test]
fn recalculate_all_risks_on_empty_store_is_noop() {
    let store = FabricStore::new();
    store.recalculate_all_risks(); // should not panic
    assert!(store.is_empty());
}

// ---------------------------------------------------------------------------
// update_staleness — elapsed time and None case
// ---------------------------------------------------------------------------

#[test]
fn update_staleness_reflects_elapsed_time() {
    let store = FabricStore::new();
    let n = NodeId(1);

    // Set churn with a past instant
    let past = Instant::now() - std::time::Duration::from_millis(200);
    store.update_churn_at(n, past);

    store.update_staleness(n);
    let score = store.get_fabric_score(n);
    // Staleness should be at least 0.1s (200ms ago)
    assert!(
        score.staleness >= 0.1,
        "expected staleness >= 0.1, got {}",
        score.staleness
    );
}

#[test]
fn update_staleness_none_case_for_untracked() {
    let store = FabricStore::new();
    store.update_staleness(NodeId(42));
    // Store should remain empty — no entry created
    assert!(store.is_empty());
}

#[test]
fn update_staleness_none_case_for_node_without_timestamp() {
    let store = FabricStore::new();
    let n = NodeId(1);
    // Create entry via set_knowledge_density — no last_mutated
    store.set_knowledge_density(n, 0.5);
    store.update_staleness(n);
    let score = store.get_fabric_score(n);
    assert_eq!(score.staleness, 0.0, "no last_mutated → staleness stays 0");
}

// ---------------------------------------------------------------------------
// set_scar_intensity — cumulative scar and risk impact
// ---------------------------------------------------------------------------

#[test]
fn set_scar_intensity_cumulative_and_risk_impact() {
    let store = FabricStore::new();
    let n = NodeId(1);

    // Set progressively higher scar intensities
    store.set_scar_intensity(n, 1.0);
    assert!((store.get_fabric_score(n).scar_intensity - 1.0).abs() < 1e-10);

    store.set_scar_intensity(n, 3.0);
    assert!((store.get_fabric_score(n).scar_intensity - 3.0).abs() < 1e-10);

    // With scar alone (no churn/pagerank/betweenness), risk = scar_boost only
    store.recalculate_all_risks();
    let score = store.get_fabric_score(n);
    // scar_boost = normalize(3.0, 3.0) * 0.5 = 0.5
    assert!(
        (score.risk_score - 0.5).abs() < 0.01,
        "expected ~0.5 from scar alone, got {}",
        score.risk_score
    );
}

#[test]
fn scar_intensity_adds_to_base_risk() {
    let store = FabricStore::new();
    let n = NodeId(1);

    // Set up non-zero base risk
    for _ in 0..10 {
        store.update_churn(n);
    }
    store.set_gds_metrics(n, 1.0, 1.0, Some(1));
    // knowledge_density = 0 → knowledge_gap = 1.0

    // Without scar
    store.recalculate_all_risks();
    let risk_no_scar = store.get_fabric_score(n).risk_score;

    // With scar
    store.set_scar_intensity(n, 2.0);
    store.recalculate_all_risks();
    let risk_with_scar = store.get_fabric_score(n).risk_score;

    // Scar should increase risk (but clamped to 1.0)
    assert!(
        risk_with_scar >= risk_no_scar,
        "scar should increase risk: {} vs {}",
        risk_with_scar,
        risk_no_scar
    );
}

// ---------------------------------------------------------------------------
// community_ids & get_community_nodes
// ---------------------------------------------------------------------------

#[test]
fn community_ids_and_get_community_nodes_roundtrip() {
    let store = FabricStore::new();

    store.set_gds_metrics(NodeId(1), 0.1, 0.1, Some(10));
    store.set_gds_metrics(NodeId(2), 0.2, 0.2, Some(10));
    store.set_gds_metrics(NodeId(3), 0.3, 0.3, Some(20));
    store.set_gds_metrics(NodeId(4), 0.4, 0.4, Some(20));
    store.set_gds_metrics(NodeId(5), 0.5, 0.5, Some(30));
    store.set_gds_metrics(NodeId(6), 0.6, 0.6, None);

    let ids = store.community_ids();
    assert_eq!(ids, vec![10, 20, 30]);

    let mut c10 = store.get_community_nodes(10);
    c10.sort_by_key(|n| n.0);
    assert_eq!(c10, vec![NodeId(1), NodeId(2)]);

    let mut c20 = store.get_community_nodes(20);
    c20.sort_by_key(|n| n.0);
    assert_eq!(c20, vec![NodeId(3), NodeId(4)]);

    let c30 = store.get_community_nodes(30);
    assert_eq!(c30, vec![NodeId(5)]);

    // None community nodes not returned for any community
    let c_missing = store.get_community_nodes(999);
    assert!(c_missing.is_empty());
}

#[test]
fn community_ids_with_no_communities_assigned() {
    let store = FabricStore::new();
    store.update_churn(NodeId(1));
    store.set_knowledge_density(NodeId(2), 0.5);
    // No community_id set on any node
    assert!(store.community_ids().is_empty());
}

// ---------------------------------------------------------------------------
// Risk formula edge cases
// ---------------------------------------------------------------------------

#[test]
fn risk_edge_case_all_metrics_zero() {
    let store = FabricStore::new();
    let n = NodeId(1);
    // Create a node with all-zero metrics (via set_knowledge_density to create entry)
    store.set_knowledge_density(n, 0.0);

    store.recalculate_all_risks();
    let score = store.get_fabric_score(n);
    // All zero → normalize returns 0 for each → base_risk = 0, scar_boost = 0
    assert!(
        score.risk_score.abs() < 1e-10,
        "all-zero metrics should give risk 0, got {}",
        score.risk_score
    );
}

#[test]
fn risk_edge_case_max_scar_intensity_zero() {
    let store = FabricStore::new();
    let n = NodeId(1);

    for _ in 0..5 {
        store.update_churn(n);
    }
    store.set_gds_metrics(n, 1.0, 1.0, None);
    // scar_intensity = 0 (default), so max_scar = 0 → normalize(0, 0) = 0

    store.recalculate_all_risks();
    let score = store.get_fabric_score(n);
    // base_risk = 1.0 * 1.0 * 1.0 * 1.0 = 1.0 (single node → all normalized to 1.0)
    // scar_boost = 0 (max_scar = 0)
    assert!(
        (score.risk_score - 1.0).abs() < 0.01,
        "expected ~1.0 with zero scar, got {}",
        score.risk_score
    );
}

#[test]
fn risk_edge_case_knowledge_density_one() {
    let store = FabricStore::new();
    let n = NodeId(1);

    for _ in 0..10 {
        store.update_churn(n);
    }
    store.set_gds_metrics(n, 1.0, 1.0, None);
    store.set_knowledge_density(n, 1.0);

    store.recalculate_all_risks();
    let score = store.get_fabric_score(n);
    // knowledge_density = 1.0 → knowledge_gap = 0 → base_risk = 0
    assert!(
        score.risk_score.abs() < 1e-10,
        "density=1.0 should zero base risk, got {}",
        score.risk_score
    );
}

#[test]
fn risk_clamp_to_one_when_scar_pushes_above() {
    let store = FabricStore::new();

    // Node with max base risk
    let n1 = NodeId(1);
    for _ in 0..10 {
        store.update_churn(n1);
    }
    store.set_gds_metrics(n1, 1.0, 1.0, None);
    // density = 0 → base_risk = 1.0

    // Also give it a high scar
    store.set_scar_intensity(n1, 10.0);

    store.recalculate_all_risks();
    let score = store.get_fabric_score(n1);
    // base_risk = 1.0, scar_boost = 0.5 * (10/10) = 0.5 → 1.5 clamped to 1.0
    assert!(
        (score.risk_score - 1.0).abs() < 1e-10,
        "risk should be clamped to 1.0, got {}",
        score.risk_score
    );
}

// ---------------------------------------------------------------------------
// FabricListener::on_batch deduplication — HashSet behavior
// ---------------------------------------------------------------------------

#[tokio::test]
async fn on_batch_deduplication_same_node_multiple_events() {
    let store = Arc::new(FabricStore::new());
    let listener = FabricListener::new(Arc::clone(&store));

    // Same node appears in 5 different events
    let events = vec![
        node_created(1),
        node_updated(1),
        node_updated(1),
        node_updated(1),
        node_updated(1),
    ];

    listener.on_batch(&events).await;

    // HashSet deduplication → churn incremented only once
    assert_eq!(
        store.get_fabric_score(NodeId(1)).churn_score,
        1.0,
        "dedup should increment churn only once per batch"
    );
}

#[tokio::test]
async fn on_batch_dedup_edge_events_share_nodes() {
    let store = Arc::new(FabricStore::new());
    let listener = FabricListener::new(Arc::clone(&store));

    // Two edges share node 10
    let events = vec![
        MutationEvent::EdgeCreated {
            edge: make_edge(1, 10, 20),
        },
        MutationEvent::EdgeCreated {
            edge: make_edge(2, 10, 30),
        },
    ];

    listener.on_batch(&events).await;

    // Node 10 appears in both edges but should be deduped
    assert_eq!(
        store.get_fabric_score(NodeId(10)).churn_score,
        1.0,
        "shared node 10 should be deduped"
    );
    // Nodes 20 and 30 each appear once
    assert_eq!(store.get_fabric_score(NodeId(20)).churn_score, 1.0);
    assert_eq!(store.get_fabric_score(NodeId(30)).churn_score, 1.0);
}

#[tokio::test]
async fn on_batch_recalculates_risk_for_affected_nodes() {
    let store = Arc::new(FabricStore::new());

    // Pre-populate with GDS metrics so risk can be computed
    store.set_gds_metrics(NodeId(1), 0.8, 0.6, Some(1));
    // Give some initial churn
    for _ in 0..5 {
        store.update_churn(NodeId(1));
    }

    let listener = FabricListener::new(Arc::clone(&store));
    let events = vec![node_updated(1)];
    listener.on_batch(&events).await;

    let score = store.get_fabric_score(NodeId(1));
    // After batch, risk should be recalculated (churn is now 6)
    // With non-zero pagerank, churn, betweenness, and density=0, risk > 0
    assert!(
        score.risk_score > 0.0,
        "on_batch should recalculate risk, got {}",
        score.risk_score
    );
}

#[tokio::test]
async fn on_batch_mixed_node_and_edge_events() {
    let store = Arc::new(FabricStore::new());
    let listener = FabricListener::new(Arc::clone(&store));

    let events = vec![
        node_created(1),
        MutationEvent::EdgeCreated {
            edge: make_edge(1, 1, 2),
        },
        MutationEvent::EdgeDeleted {
            edge: make_edge(2, 2, 3),
        },
        MutationEvent::NodeDeleted {
            node: node_snapshot(3),
        },
    ];

    listener.on_batch(&events).await;

    // Node 1: from NodeCreated + EdgeCreated(src) → deduped to 1
    assert_eq!(store.get_fabric_score(NodeId(1)).churn_score, 1.0);
    // Node 2: from EdgeCreated(dst) + EdgeDeleted(src) → deduped to 1
    assert_eq!(store.get_fabric_score(NodeId(2)).churn_score, 1.0);
    // Node 3: from EdgeDeleted(dst) + NodeDeleted → deduped to 1
    assert_eq!(store.get_fabric_score(NodeId(3)).churn_score, 1.0);

    // Verify total tracked nodes
    assert_eq!(store.len(), 3);
}
