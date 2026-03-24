//! Integration tests for the co-change detection subsystem.

#![cfg(feature = "co-change")]

use grafeo_cognitive::{CoChangeConfig, CoChangeDetector, CoChangeRelation, CoChangeStore};
use grafeo_common::types::NodeId;
use grafeo_reactive::{MutationEvent, MutationListener, NodeSnapshot};
use std::sync::Arc;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_node(id: u64) -> NodeSnapshot {
    NodeSnapshot {
        id: NodeId::new(id),
        labels: smallvec::smallvec![arcstr::literal!("Test")],
        properties: vec![],
    }
}

fn make_edge_snapshot(id: u64, src: u64, dst: u64) -> grafeo_reactive::EdgeSnapshot {
    grafeo_reactive::EdgeSnapshot {
        id: grafeo_common::types::EdgeId::new(id),
        src: NodeId::new(src),
        dst: NodeId::new(dst),
        edge_type: arcstr::literal!("RELATES_TO"),
        properties: vec![],
    }
}

// ---------------------------------------------------------------------------
// CoChangeRelation tests
// ---------------------------------------------------------------------------

#[test]
fn relation_new_initializes_count_1() {
    let rel = CoChangeRelation::new(NodeId::new(1), NodeId::new(2), Duration::from_secs(3600));
    assert_eq!(rel.count, 1);
    assert!(rel.strength() > 0.0);
}

#[test]
fn relation_strength_decays_over_time() {
    let half_life = Duration::from_secs(3600); // 1 hour
    let start = Instant::now();
    let rel = CoChangeRelation::new_at(NodeId::new(1), NodeId::new(2), half_life, start);

    // At t=0: strength = 1.0 (count=1)
    let s0 = rel.strength_at(start);
    assert!((s0 - 1.0).abs() < 1e-10, "expected 1.0, got {s0}");

    // After 1 half-life: strength ~= 0.5
    let s1 = rel.strength_at(start + Duration::from_secs(3600));
    assert!((s1 - 0.5).abs() < 1e-10, "expected ~0.5, got {s1}");

    // After 2 half-lives: strength ~= 0.25
    let s2 = rel.strength_at(start + Duration::from_secs(7200));
    assert!((s2 - 0.25).abs() < 1e-10, "expected ~0.25, got {s2}");
}

#[test]
fn relation_record_increments_count() {
    let start = Instant::now();
    let mut rel = CoChangeRelation::new_at(
        NodeId::new(1),
        NodeId::new(2),
        Duration::from_secs(3600),
        start,
    );
    assert_eq!(rel.count, 1);

    rel.record_at(start + Duration::from_millis(1));
    assert_eq!(rel.count, 2);

    rel.record_at(start + Duration::from_millis(2));
    assert_eq!(rel.count, 3);

    // Strength should be count * decay_factor. At ~0ms elapsed, ~= 3.0
    let s = rel.strength_at(start + Duration::from_millis(2));
    assert!(s > 2.9 && s <= 3.0, "expected ~3.0, got {s}");
}

// ---------------------------------------------------------------------------
// CoChangeStore tests
// ---------------------------------------------------------------------------

#[test]
fn store_record_creates_relation() {
    let store = CoChangeStore::new(CoChangeConfig::default());
    store.record_co_change(NodeId::new(1), NodeId::new(2));

    let rel = store.get_relation(NodeId::new(1), NodeId::new(2));
    assert!(rel.is_some());
    assert_eq!(rel.unwrap().count, 1);
}

#[test]
fn store_record_is_symmetric() {
    let store = CoChangeStore::new(CoChangeConfig::default());
    store.record_co_change(NodeId::new(1), NodeId::new(2));

    // Retrieve in reverse order
    assert!(store.get_relation(NodeId::new(2), NodeId::new(1)).is_some());
}

#[test]
fn store_no_self_co_change() {
    let store = CoChangeStore::new(CoChangeConfig::default());
    store.record_co_change(NodeId::new(1), NodeId::new(1));
    assert!(store.is_empty());
}

#[test]
fn store_record_increments_existing() {
    let store = CoChangeStore::new(CoChangeConfig::default());
    store.record_co_change(NodeId::new(1), NodeId::new(2));
    store.record_co_change(NodeId::new(1), NodeId::new(2));
    store.record_co_change(NodeId::new(2), NodeId::new(1)); // reverse order

    let rel = store.get_relation(NodeId::new(1), NodeId::new(2)).unwrap();
    assert_eq!(rel.count, 3);
    assert_eq!(store.len(), 1); // Still just one relation
}

#[test]
fn store_get_co_changed_sorted_by_strength() {
    let store = CoChangeStore::new(CoChangeConfig::default());

    // Node 1 co-changed with 2 once, with 3 three times, with 4 twice
    store.record_co_change(NodeId::new(1), NodeId::new(2));

    store.record_co_change(NodeId::new(1), NodeId::new(3));
    store.record_co_change(NodeId::new(1), NodeId::new(3));
    store.record_co_change(NodeId::new(1), NodeId::new(3));

    store.record_co_change(NodeId::new(1), NodeId::new(4));
    store.record_co_change(NodeId::new(1), NodeId::new(4));

    let co_changed = store.get_co_changed(NodeId::new(1));
    assert_eq!(co_changed.len(), 3);

    // Should be sorted: node3 (count=3) > node4 (count=2) > node2 (count=1)
    assert_eq!(co_changed[0].0, NodeId::new(3));
    assert_eq!(co_changed[1].0, NodeId::new(4));
    assert_eq!(co_changed[2].0, NodeId::new(2));
}

#[test]
fn store_get_co_changed_empty_for_unknown_node() {
    let store = CoChangeStore::new(CoChangeConfig::default());
    let result = store.get_co_changed(NodeId::new(999));
    assert!(result.is_empty());
}

// ---------------------------------------------------------------------------
// CoChangeDetector (MutationListener) tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn detector_basic_batch_creates_co_changes() {
    let store = Arc::new(CoChangeStore::new(CoChangeConfig::default()));
    let detector = CoChangeDetector::new(Arc::clone(&store));

    // Batch with mutations on nodes A(1), B(2), C(3)
    let events = vec![
        MutationEvent::NodeCreated { node: make_node(1) },
        MutationEvent::NodeCreated { node: make_node(2) },
        MutationEvent::NodeCreated { node: make_node(3) },
    ];
    detector.on_batch(&events).await;

    // Should create 3 co-changes: (1,2), (1,3), (2,3) = n(n-1)/2 = 3
    assert_eq!(store.len(), 3);
    assert!(store.get_relation(NodeId::new(1), NodeId::new(2)).is_some());
    assert!(store.get_relation(NodeId::new(1), NodeId::new(3)).is_some());
    assert!(store.get_relation(NodeId::new(2), NodeId::new(3)).is_some());
}

#[tokio::test]
async fn detector_single_node_no_co_change() {
    let store = Arc::new(CoChangeStore::new(CoChangeConfig::default()));
    let detector = CoChangeDetector::new(Arc::clone(&store));

    let events = vec![MutationEvent::NodeCreated { node: make_node(1) }];
    detector.on_batch(&events).await;

    assert!(store.is_empty());
}

#[tokio::test]
async fn detector_repeated_batches_increment_count() {
    let store = Arc::new(CoChangeStore::new(CoChangeConfig::default()));
    let detector = CoChangeDetector::new(Arc::clone(&store));

    let events = vec![
        MutationEvent::NodeCreated { node: make_node(1) },
        MutationEvent::NodeCreated { node: make_node(2) },
    ];

    detector.on_batch(&events).await;
    let c1 = store
        .get_relation(NodeId::new(1), NodeId::new(2))
        .unwrap()
        .count;
    assert_eq!(c1, 1);

    detector.on_batch(&events).await;
    let c2 = store
        .get_relation(NodeId::new(1), NodeId::new(2))
        .unwrap()
        .count;
    assert_eq!(c2, 2);

    detector.on_batch(&events).await;
    let c3 = store
        .get_relation(NodeId::new(1), NodeId::new(2))
        .unwrap()
        .count;
    assert_eq!(c3, 3);
}

#[tokio::test]
async fn detector_edge_events_co_change_endpoints() {
    let store = Arc::new(CoChangeStore::new(CoChangeConfig::default()));
    let detector = CoChangeDetector::new(Arc::clone(&store));

    let events = vec![MutationEvent::EdgeCreated {
        edge: make_edge_snapshot(1, 10, 20),
    }];
    detector.on_batch(&events).await;

    assert!(
        store
            .get_relation(NodeId::new(10), NodeId::new(20))
            .is_some()
    );
}

#[tokio::test]
async fn detector_large_batch_combinatorial() {
    let store = Arc::new(CoChangeStore::new(CoChangeConfig::default()));
    let detector = CoChangeDetector::new(Arc::clone(&store));

    // 10 nodes → n(n-1)/2 = 45 co-change pairs
    let events: Vec<MutationEvent> = (1..=10)
        .map(|id| MutationEvent::NodeCreated {
            node: make_node(id),
        })
        .collect();
    detector.on_batch(&events).await;

    assert_eq!(store.len(), 45);
}

#[tokio::test]
async fn detector_large_batch_skipped_when_over_max() {
    let config = CoChangeConfig {
        max_batch_nodes: 5,
        ..CoChangeConfig::default()
    };
    let store = Arc::new(CoChangeStore::new(config));
    let detector = CoChangeDetector::new(Arc::clone(&store));

    // 10 nodes > max_batch_nodes=5 → should skip
    let events: Vec<MutationEvent> = (1..=10)
        .map(|id| MutationEvent::NodeCreated {
            node: make_node(id),
        })
        .collect();
    detector.on_batch(&events).await;

    assert!(store.is_empty(), "should have skipped large batch");
}

#[tokio::test]
async fn detector_deduplicates_node_ids_in_batch() {
    let store = Arc::new(CoChangeStore::new(CoChangeConfig::default()));
    let detector = CoChangeDetector::new(Arc::clone(&store));

    // Same node appears multiple times in batch
    let events = vec![
        MutationEvent::NodeCreated { node: make_node(1) },
        MutationEvent::NodeUpdated {
            before: make_node(1),
            after: make_node(1),
        },
        MutationEvent::NodeCreated { node: make_node(2) },
    ];
    detector.on_batch(&events).await;

    // Only 1 co-change pair (1,2), count=1 (not duplicated)
    assert_eq!(store.len(), 1);
    let rel = store.get_relation(NodeId::new(1), NodeId::new(2)).unwrap();
    assert_eq!(rel.count, 1);
}

#[tokio::test]
async fn detector_mixed_node_and_edge_events() {
    let store = Arc::new(CoChangeStore::new(CoChangeConfig::default()));
    let detector = CoChangeDetector::new(Arc::clone(&store));

    // Node 1 created, edge between 2→3 created
    let events = vec![
        MutationEvent::NodeCreated { node: make_node(1) },
        MutationEvent::EdgeCreated {
            edge: make_edge_snapshot(1, 2, 3),
        },
    ];
    detector.on_batch(&events).await;

    // Nodes touched: {1, 2, 3} → 3 pairs
    assert_eq!(store.len(), 3);
}

// ---------------------------------------------------------------------------
// Debug impls
// ---------------------------------------------------------------------------

#[test]
fn store_debug_formatting() {
    let store = CoChangeStore::new(CoChangeConfig::default());
    store.record_co_change(NodeId::new(1), NodeId::new(2));
    let dbg = format!("{:?}", store);
    assert!(dbg.contains("CoChangeStore"), "got: {dbg}");
    assert!(dbg.contains("relation_count"), "got: {dbg}");
    assert!(dbg.contains("config"), "got: {dbg}");
}

#[test]
fn detector_debug_formatting() {
    let store = Arc::new(CoChangeStore::new(CoChangeConfig::default()));
    let detector = CoChangeDetector::new(Arc::clone(&store));
    let dbg = format!("{:?}", detector);
    assert!(dbg.contains("CoChangeDetector"), "got: {dbg}");
    assert!(dbg.contains("store"), "got: {dbg}");
}

// ---------------------------------------------------------------------------
// CoChangeDetector: store(), name()
// ---------------------------------------------------------------------------

#[test]
fn detector_store_accessor() {
    let store = Arc::new(CoChangeStore::new(CoChangeConfig::default()));
    let detector = CoChangeDetector::new(Arc::clone(&store));
    // store accessor returns a reference to the underlying store
    assert!(detector.store().is_empty());
    store.record_co_change(NodeId::new(1), NodeId::new(2));
    assert_eq!(detector.store().len(), 1);
}

#[test]
fn detector_name() {
    let store = Arc::new(CoChangeStore::new(CoChangeConfig::default()));
    let detector = CoChangeDetector::new(Arc::clone(&store));
    assert_eq!(detector.name(), "cognitive:co_change");
}

// ---------------------------------------------------------------------------
// CoChangeDetector: on_event is a no-op for single events
// ---------------------------------------------------------------------------

#[tokio::test]
async fn detector_on_event_is_noop() {
    let store = Arc::new(CoChangeStore::new(CoChangeConfig::default()));
    let detector = CoChangeDetector::new(Arc::clone(&store));

    let event = MutationEvent::NodeCreated { node: make_node(1) };
    detector.on_event(&event).await;
    assert!(store.is_empty(), "single event should not produce co-changes");
}

// ---------------------------------------------------------------------------
// Edge event variants in node_ids()
// ---------------------------------------------------------------------------

#[tokio::test]
async fn detector_edge_updated_co_changes_endpoints() {
    let store = Arc::new(CoChangeStore::new(CoChangeConfig::default()));
    let detector = CoChangeDetector::new(Arc::clone(&store));

    let events = vec![
        MutationEvent::EdgeUpdated {
            before: make_edge_snapshot(1, 10, 20),
            after: make_edge_snapshot(1, 10, 20),
        },
    ];
    detector.on_batch(&events).await;

    assert!(store.get_relation(NodeId::new(10), NodeId::new(20)).is_some());
}

#[tokio::test]
async fn detector_edge_deleted_co_changes_endpoints() {
    let store = Arc::new(CoChangeStore::new(CoChangeConfig::default()));
    let detector = CoChangeDetector::new(Arc::clone(&store));

    let events = vec![
        MutationEvent::EdgeDeleted {
            edge: make_edge_snapshot(1, 30, 40),
        },
    ];
    detector.on_batch(&events).await;

    assert!(store.get_relation(NodeId::new(30), NodeId::new(40)).is_some());
}

#[tokio::test]
async fn detector_node_deleted_extracts_id() {
    let store = Arc::new(CoChangeStore::new(CoChangeConfig::default()));
    let detector = CoChangeDetector::new(Arc::clone(&store));

    let events = vec![
        MutationEvent::NodeDeleted { node: make_node(5) },
        MutationEvent::NodeDeleted { node: make_node(6) },
    ];
    detector.on_batch(&events).await;

    assert!(store.get_relation(NodeId::new(5), NodeId::new(6)).is_some());
}
