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
    assert_eq!(score.mutation_frequency, 0.0);
    assert_eq!(score.annotation_density, 0.0);
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
// FabricStore — mutation_frequency incremental
// ---------------------------------------------------------------------------

#[test]
fn record_mutation_increments() {
    let store = FabricStore::new();
    let nid = NodeId(1);

    store.record_mutation(nid);
    assert_eq!(store.get_fabric_score(nid).mutation_frequency, 1.0);

    store.record_mutation(nid);
    assert_eq!(store.get_fabric_score(nid).mutation_frequency, 2.0);

    store.record_mutation(nid);
    assert_eq!(store.get_fabric_score(nid).mutation_frequency, 3.0);
}

#[test]
fn record_mutation_resets_staleness() {
    let store = FabricStore::new();
    let nid = NodeId(1);

    store.record_mutation(nid);
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
fn get_hotspots_sorted_by_mutation_frequency_desc() {
    let store = FabricStore::new();

    // Node 1: 3 mutations
    for _ in 0..3 {
        store.record_mutation(NodeId(1));
    }
    // Node 2: 5 mutations
    for _ in 0..5 {
        store.record_mutation(NodeId(2));
    }
    // Node 3: 1 mutation
    store.record_mutation(NodeId(3));

    let hotspots = store.get_hotspots(10);
    assert_eq!(hotspots.len(), 3);
    assert_eq!(hotspots[0].0, NodeId(2)); // highest mutation_frequency
    assert_eq!(hotspots[0].1.mutation_frequency, 5.0);
    assert_eq!(hotspots[1].0, NodeId(1));
    assert_eq!(hotspots[1].1.mutation_frequency, 3.0);
    assert_eq!(hotspots[2].0, NodeId(3));
    assert_eq!(hotspots[2].1.mutation_frequency, 1.0);
}

#[test]
fn get_hotspots_truncates_to_top_n() {
    let store = FabricStore::new();
    for i in 0..10 {
        for _ in 0..=i {
            store.record_mutation(NodeId(i));
        }
    }
    let hotspots = store.get_hotspots(3);
    assert_eq!(hotspots.len(), 3);
    // Top 3 should be nodes 9, 8, 7
    assert_eq!(hotspots[0].1.mutation_frequency, 10.0);
    assert_eq!(hotspots[1].1.mutation_frequency, 9.0);
    assert_eq!(hotspots[2].1.mutation_frequency, 8.0);
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
        store.record_mutation(n1);
    }
    store.set_gds_metrics(n1, 0.9, 0.8, Some(1));
    // Low knowledge density (default 0.0) → high knowledge gap

    // Node 2: low risk (good documentation)
    store.record_mutation(n2);
    store.set_annotation_density(n2, 0.95);
    store.set_gds_metrics(n2, 0.1, 0.1, Some(1));

    // Node 3: medium risk
    for _ in 0..5 {
        store.record_mutation(n3);
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
        !zones_high.is_empty(),
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
fn risk_score_high_pr_high_mutation_frequency_low_density() {
    let store = FabricStore::new();
    let nid = NodeId(1);

    // High mutation_frequency
    for _ in 0..100 {
        store.record_mutation(nid);
    }
    // High pagerank, high betweenness, low density (default 0.0)
    store.set_gds_metrics(nid, 1.0, 1.0, Some(1));

    store.recalculate_all_risks();
    let score = store.get_fabric_score(nid);

    // Weighted additive: pr=1*0.25 + mutation_frequency=1*0.25 + gap=1*0.20 + btwn=1*0.15 + scar=0*0.15 = 0.85
    assert!(
        (score.risk_score - 0.85).abs() < 0.01,
        "expected risk ~0.85, got {}",
        score.risk_score
    );
}

#[test]
fn risk_score_zero_when_high_density() {
    let store = FabricStore::new();
    let nid = NodeId(1);

    for _ in 0..10 {
        store.record_mutation(nid);
    }
    store.set_annotation_density(nid, 1.0); // Fully documented
    store.set_gds_metrics(nid, 1.0, 1.0, Some(1));

    store.recalculate_all_risks();
    let score = store.get_fabric_score(nid);

    // density=1.0 → gap=0 → pr=1*0.25 + mutation_frequency=1*0.25 + gap=0 + btwn=1*0.15 + scar=0 = 0.65
    assert!(
        (score.risk_score - 0.65).abs() < 0.01,
        "expected risk ~0.65, got {}",
        score.risk_score
    );
}

#[test]
fn risk_score_zero_when_no_pagerank() {
    let store = FabricStore::new();
    let nid = NodeId(1);

    for _ in 0..10 {
        store.record_mutation(nid);
    }
    // pagerank=0 → pr=0, but mutation_frequency=1*0.25 + gap=1*0.20 + btwn=1*0.15 still contribute
    store.set_gds_metrics(nid, 0.0, 1.0, Some(1));

    store.recalculate_all_risks();
    let score = store.get_fabric_score(nid);

    // Weighted additive: pr=0 + mutation_frequency=1*0.25 + gap=1*0.20 + btwn=1*0.15 + scar=0 = 0.60
    assert!(
        (score.risk_score - 0.60).abs() < 0.01,
        "expected risk ~0.60, got {}",
        score.risk_score
    );
}

// ---------------------------------------------------------------------------
// FabricListener — MutationListener integration
// ---------------------------------------------------------------------------

#[tokio::test]
async fn listener_on_event_increments_mutation_frequency() {
    let store = Arc::new(FabricStore::new());
    let listener = FabricListener::new(Arc::clone(&store));

    listener.on_event(&node_created(1)).await;
    assert_eq!(store.get_fabric_score(NodeId(1)).mutation_frequency, 1.0);

    listener.on_event(&node_updated(1)).await;
    assert_eq!(store.get_fabric_score(NodeId(1)).mutation_frequency, 2.0);
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

    // Node 1 appears in 2 events but on_batch deduplicates → mutation_frequency = 1
    // (only incremented once per batch due to HashSet dedup)
    assert_eq!(store.get_fabric_score(NodeId(1)).mutation_frequency, 1.0);
    assert_eq!(store.get_fabric_score(NodeId(2)).mutation_frequency, 1.0);
    assert_eq!(store.get_fabric_score(NodeId(3)).mutation_frequency, 1.0);
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

    store.record_mutation(NodeId(1));
    assert!(!store.is_empty());
    assert_eq!(store.len(), 1);

    store.record_mutation(NodeId(2));
    assert_eq!(store.len(), 2);
}

// ---------------------------------------------------------------------------
// FabricStore — set_annotation_density (line 154)
// ---------------------------------------------------------------------------

#[test]
fn set_annotation_density_creates_entry_if_absent() {
    let store = FabricStore::new();
    let nid = NodeId(50);
    store.set_annotation_density(nid, 0.75);
    let score = store.get_fabric_score(nid);
    assert!((score.annotation_density - 0.75).abs() < 1e-10);
    // Other fields should be default
    assert_eq!(score.mutation_frequency, 0.0);
}

#[test]
fn set_annotation_density_updates_existing_entry() {
    let store = FabricStore::new();
    let nid = NodeId(51);
    store.record_mutation(nid);
    store.set_annotation_density(nid, 0.8);
    let score = store.get_fabric_score(nid);
    assert!((score.annotation_density - 0.8).abs() < 1e-10);
    assert_eq!(score.mutation_frequency, 1.0); // mutation_frequency preserved
}

// ---------------------------------------------------------------------------
// FabricStore — set_scar_intensity (line 167)
// ---------------------------------------------------------------------------

#[test]
fn set_scar_intensity_creates_entry_if_absent() {
    let store = FabricStore::new();
    let nid = NodeId(60);
    store.set_scar_intensity(nid, 3.5);
    let score = store.get_fabric_score(nid);
    assert!((score.scar_intensity - 3.5).abs() < 1e-10);
    assert_eq!(score.mutation_frequency, 0.0);
}

#[test]
fn set_scar_intensity_updates_existing_entry() {
    let store = FabricStore::new();
    let nid = NodeId(61);
    store.record_mutation(nid);
    store.set_scar_intensity(nid, 1.2);
    let score = store.get_fabric_score(nid);
    assert!((score.scar_intensity - 1.2).abs() < 1e-10);
    assert_eq!(score.mutation_frequency, 1.0);
}

// ---------------------------------------------------------------------------
// FabricStore — set_gds_metrics (line 178)
// ---------------------------------------------------------------------------

#[test]
fn set_gds_metrics_creates_entry_if_absent() {
    let store = FabricStore::new();
    let nid = NodeId(70);
    store.set_gds_metrics(nid, 0.5, 0.3, Some(7));
    let score = store.get_fabric_score(nid);
    assert!((score.pagerank - 0.5).abs() < 1e-10);
    assert!((score.betweenness - 0.3).abs() < 1e-10);
    assert_eq!(score.community_id, Some(7));
}

#[test]
fn set_gds_metrics_updates_existing_entry() {
    let store = FabricStore::new();
    let nid = NodeId(71);
    store.record_mutation(nid);
    store.set_gds_metrics(nid, 0.9, 0.7, None);
    let score = store.get_fabric_score(nid);
    assert!((score.pagerank - 0.9).abs() < 1e-10);
    assert!((score.betweenness - 0.7).abs() < 1e-10);
    assert_eq!(score.community_id, None);
    assert_eq!(score.mutation_frequency, 1.0);
}

// ---------------------------------------------------------------------------
// FabricStore — recalculate_all_risks (line 255)
// ---------------------------------------------------------------------------

#[test]
fn recalculate_all_risks_updates_every_node() {
    let store = FabricStore::new();

    for _ in 0..5 {
        store.record_mutation(NodeId(1));
    }
    store.set_gds_metrics(NodeId(1), 0.8, 0.6, Some(1));

    for _ in 0..3 {
        store.record_mutation(NodeId(2));
    }
    store.set_gds_metrics(NodeId(2), 0.4, 0.2, Some(1));
    store.set_annotation_density(NodeId(2), 1.0); // fully documented

    store.recalculate_all_risks();

    let s1 = store.get_fabric_score(NodeId(1));
    let s2 = store.get_fabric_score(NodeId(2));

    // Node 1 has density 0 → knowledge_gap = 1 → non-zero risk
    assert!(s1.risk_score > 0.0, "expected risk > 0 for node 1");
    // Node 2 has density 1.0 → knowledge_gap = 0 → base risk = 0
    assert!(
        s2.risk_score < s1.risk_score,
        "node 2 should have less risk than node 1"
    );
}

// ---------------------------------------------------------------------------
// FabricStore — community_ids (line 297)
// ---------------------------------------------------------------------------

#[test]
fn community_ids_returns_distinct_sorted() {
    let store = FabricStore::new();
    store.set_gds_metrics(NodeId(1), 0.1, 0.1, Some(3));
    store.set_gds_metrics(NodeId(2), 0.1, 0.1, Some(1));
    store.set_gds_metrics(NodeId(3), 0.1, 0.1, Some(3)); // duplicate
    store.set_gds_metrics(NodeId(4), 0.1, 0.1, Some(2));
    store.set_gds_metrics(NodeId(5), 0.1, 0.1, None); // no community

    let ids = store.community_ids();
    assert_eq!(ids, vec![1, 2, 3]);
}

#[test]
fn community_ids_empty_store() {
    let store = FabricStore::new();
    assert!(store.community_ids().is_empty());
}

// ---------------------------------------------------------------------------
// FabricStore — get_community_nodes (line 309)
// ---------------------------------------------------------------------------

#[test]
fn get_community_nodes_returns_matching() {
    let store = FabricStore::new();
    store.set_gds_metrics(NodeId(1), 0.1, 0.1, Some(5));
    store.set_gds_metrics(NodeId(2), 0.1, 0.1, Some(5));
    store.set_gds_metrics(NodeId(3), 0.1, 0.1, Some(6));
    store.set_gds_metrics(NodeId(4), 0.1, 0.1, None);

    let mut nodes = store.get_community_nodes(5);
    nodes.sort_by_key(|n| n.0);
    assert_eq!(nodes, vec![NodeId(1), NodeId(2)]);
}

#[test]
fn get_community_nodes_empty_for_unknown_community() {
    let store = FabricStore::new();
    store.set_gds_metrics(NodeId(1), 0.1, 0.1, Some(1));
    assert!(store.get_community_nodes(999).is_empty());
}

// ---------------------------------------------------------------------------
// FabricListener::on_event with edge events (line 431)
// ---------------------------------------------------------------------------

fn make_edge(id: u64, src: u64, dst: u64) -> grafeo_reactive::EdgeSnapshot {
    grafeo_reactive::EdgeSnapshot {
        id: grafeo_common::types::EdgeId::new(id),
        src: NodeId(src),
        dst: NodeId(dst),
        edge_type: arcstr::literal!("KNOWS"),
        properties: vec![],
    }
}

#[tokio::test]
async fn listener_on_event_edge_created() {
    let store = Arc::new(FabricStore::new());
    let listener = FabricListener::new(Arc::clone(&store));

    let event = MutationEvent::EdgeCreated {
        edge: make_edge(1, 10, 20),
    };
    listener.on_event(&event).await;

    assert_eq!(store.get_fabric_score(NodeId(10)).mutation_frequency, 1.0);
    assert_eq!(store.get_fabric_score(NodeId(20)).mutation_frequency, 1.0);
}

#[tokio::test]
async fn listener_on_event_edge_updated() {
    let store = Arc::new(FabricStore::new());
    let listener = FabricListener::new(Arc::clone(&store));

    let event = MutationEvent::EdgeUpdated {
        before: make_edge(1, 10, 20),
        after: make_edge(1, 10, 20),
    };
    listener.on_event(&event).await;

    assert_eq!(store.get_fabric_score(NodeId(10)).mutation_frequency, 1.0);
    assert_eq!(store.get_fabric_score(NodeId(20)).mutation_frequency, 1.0);
}

#[tokio::test]
async fn listener_on_event_edge_deleted() {
    let store = Arc::new(FabricStore::new());
    let listener = FabricListener::new(Arc::clone(&store));

    let event = MutationEvent::EdgeDeleted {
        edge: make_edge(1, 30, 40),
    };
    listener.on_event(&event).await;

    assert_eq!(store.get_fabric_score(NodeId(30)).mutation_frequency, 1.0);
    assert_eq!(store.get_fabric_score(NodeId(40)).mutation_frequency, 1.0);
}

#[tokio::test]
async fn listener_on_event_node_deleted() {
    let store = Arc::new(FabricStore::new());
    let listener = FabricListener::new(Arc::clone(&store));

    let event = MutationEvent::NodeDeleted {
        node: node_snapshot(99),
    };
    listener.on_event(&event).await;

    assert_eq!(store.get_fabric_score(NodeId(99)).mutation_frequency, 1.0);
}

// ---------------------------------------------------------------------------
// FabricStore::Default impl
// ---------------------------------------------------------------------------

#[test]
fn fabric_store_default_impl() {
    let store = FabricStore::default();
    assert!(store.is_empty());
    assert_eq!(store.len(), 0);
}

// ---------------------------------------------------------------------------
// Debug impls for FabricStore and FabricListener
// ---------------------------------------------------------------------------

#[test]
fn fabric_store_debug_impl() {
    let store = FabricStore::new();
    store.record_mutation(NodeId(1));
    let s = format!("{:?}", store);
    assert!(s.contains("FabricStore"));
    assert!(s.contains("tracked_nodes"));
}

#[test]
fn fabric_listener_debug_impl() {
    let store = Arc::new(FabricStore::new());
    let listener = FabricListener::new(store);
    let s = format!("{:?}", listener);
    assert!(s.contains("FabricListener"));
    assert!(s.contains("store"));
}
