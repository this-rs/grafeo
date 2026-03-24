//! Additional coverage tests for the co-change detection subsystem.

#![cfg(feature = "co-change")]

use grafeo_cognitive::{CoChangeConfig, CoChangeDetector, CoChangeRelation, CoChangeStore};
use grafeo_common::types::NodeId;
use grafeo_reactive::{MutationEvent, MutationListener, NodeSnapshot};
use std::sync::Arc;
use std::time::{Duration, Instant};

fn make_node(id: u64) -> NodeSnapshot {
    NodeSnapshot {
        id: NodeId::new(id),
        labels: smallvec::smallvec![arcstr::literal!("Test")],
        properties: vec![],
    }
}

// ---------------------------------------------------------------------------
// CoChangeConfig default values
// ---------------------------------------------------------------------------

#[test]
fn co_change_config_default_values() {
    let config = CoChangeConfig::default();
    assert_eq!(config.window_duration, Duration::ZERO);
    assert_eq!(
        config.strength_half_life,
        Duration::from_secs(30 * 24 * 3600)
    );
    assert_eq!(config.max_batch_nodes, 100);
}

// ---------------------------------------------------------------------------
// CoChangeStore: config() accessor
// ---------------------------------------------------------------------------

#[test]
fn store_config_accessor() {
    let config = CoChangeConfig {
        max_batch_nodes: 50,
        ..CoChangeConfig::default()
    };
    let store = CoChangeStore::new(config);
    assert_eq!(store.config().max_batch_nodes, 50);
}

// ---------------------------------------------------------------------------
// CoChangeRelation: record (non-_at variant)
// ---------------------------------------------------------------------------

#[test]
fn relation_record_non_at_variant() {
    let mut rel = CoChangeRelation::new(NodeId::new(1), NodeId::new(2), Duration::from_secs(3600));
    assert_eq!(rel.count, 1);

    rel.record();
    assert_eq!(rel.count, 2);

    // Strength should be ~2.0 (just recorded)
    let s = rel.strength();
    assert!((s - 2.0).abs() < 0.01, "expected ~2.0, got {s}");
}

// ---------------------------------------------------------------------------
// CoChangeRelation: created_at is preserved
// ---------------------------------------------------------------------------

#[test]
fn relation_created_at_preserved() {
    let start = Instant::now();
    let mut rel = CoChangeRelation::new_at(
        NodeId::new(1),
        NodeId::new(2),
        Duration::from_secs(3600),
        start,
    );
    assert_eq!(rel.created_at, start);

    let later = start + Duration::from_secs(100);
    rel.record_at(later);
    // created_at should not change
    assert_eq!(rel.created_at, start);
    // last_co_changed should update
    assert_eq!(rel.last_co_changed, later);
}

// ---------------------------------------------------------------------------
// CoChangeStore: get_relation nonexistent
// ---------------------------------------------------------------------------

#[test]
fn store_get_relation_nonexistent() {
    let store = CoChangeStore::new(CoChangeConfig::default());
    assert!(store.get_relation(NodeId::new(1), NodeId::new(2)).is_none());
}

// ---------------------------------------------------------------------------
// CoChangeStore: len and is_empty
// ---------------------------------------------------------------------------

#[test]
fn store_len_and_is_empty() {
    let store = CoChangeStore::new(CoChangeConfig::default());
    assert_eq!(store.len(), 0);
    assert!(store.is_empty());

    store.record_co_change(NodeId::new(1), NodeId::new(2));
    assert_eq!(store.len(), 1);
    assert!(!store.is_empty());
}

// ---------------------------------------------------------------------------
// CoChangeDetector: on_batch with NodeUpdated
// ---------------------------------------------------------------------------

#[tokio::test]
async fn detector_on_batch_with_node_updated() {
    let store = Arc::new(CoChangeStore::new(CoChangeConfig::default()));
    let detector = CoChangeDetector::new(Arc::clone(&store));

    let events = vec![
        MutationEvent::NodeUpdated {
            before: make_node(1),
            after: make_node(1),
        },
        MutationEvent::NodeCreated { node: make_node(2) },
    ];
    detector.on_batch(&events).await;

    assert!(store.get_relation(NodeId::new(1), NodeId::new(2)).is_some());
}

// ---------------------------------------------------------------------------
// CoChangeStore: get_co_changed returns correct other node
// ---------------------------------------------------------------------------

#[test]
fn store_get_co_changed_returns_other_node() {
    let store = CoChangeStore::new(CoChangeConfig::default());
    store.record_co_change(NodeId::new(5), NodeId::new(10));

    let co_changed = store.get_co_changed(NodeId::new(5));
    assert_eq!(co_changed.len(), 1);
    assert_eq!(co_changed[0].0, NodeId::new(10));

    let co_changed = store.get_co_changed(NodeId::new(10));
    assert_eq!(co_changed.len(), 1);
    assert_eq!(co_changed[0].0, NodeId::new(5));
}

// ---------------------------------------------------------------------------
// CoChangeRelation: strength decays to near zero over many half-lives
// ---------------------------------------------------------------------------

#[test]
fn relation_strength_near_zero_far_future() {
    let start = Instant::now();
    let rel = CoChangeRelation::new_at(
        NodeId::new(1),
        NodeId::new(2),
        Duration::from_secs(60),
        start,
    );
    let far = start + Duration::from_secs(60 * 100);
    let s = rel.strength_at(far);
    assert!(s < 1e-20, "expected near-zero, got {s}");
}

// ---------------------------------------------------------------------------
// CoChangeRelation: Debug and Clone
// ---------------------------------------------------------------------------

#[test]
fn relation_debug_and_clone() {
    let rel = CoChangeRelation::new(NodeId::new(1), NodeId::new(2), Duration::from_secs(3600));
    let dbg = format!("{:?}", rel);
    assert!(dbg.contains("CoChangeRelation"), "got: {dbg}");

    let cloned = rel.clone();
    assert_eq!(cloned.count, rel.count);
}

// ---------------------------------------------------------------------------
// CoChangeDetector: empty batch is no-op
// ---------------------------------------------------------------------------

#[tokio::test]
async fn detector_empty_batch() {
    let store = Arc::new(CoChangeStore::new(CoChangeConfig::default()));
    let detector = CoChangeDetector::new(Arc::clone(&store));

    detector.on_batch(&[]).await;
    assert!(store.is_empty());
}
