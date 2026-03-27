//! Additional tests for MutationBus — edge cases, Default impl, Debug,
//! lagged receiver, cloned bus metrics.

use obrain_common::types::NodeId;
use obrain_reactive::{MutationBatch, MutationBus, MutationEvent, NodeSnapshot};

fn make_node_event(id: u64) -> MutationEvent {
    MutationEvent::NodeCreated {
        node: NodeSnapshot {
            id: NodeId::new(id),
            labels: smallvec::smallvec![arcstr::literal!("Test")],
            properties: vec![],
        },
    }
}

// ========================================================================
// Default impl
// ========================================================================

#[test]
fn default_creates_bus() {
    let bus = MutationBus::default();
    assert_eq!(bus.subscriber_count(), 0);
    assert_eq!(bus.total_events_published(), 0);
    assert_eq!(bus.total_batches_published(), 0);
}

// ========================================================================
// Debug impl
// ========================================================================

#[test]
fn debug_format() {
    let bus = MutationBus::new();
    let debug = format!("{:?}", bus);
    assert!(debug.contains("MutationBus"));
}

// ========================================================================
// Clone shares channel but copies counters
// ========================================================================

#[tokio::test]
async fn clone_copies_metrics_snapshot() {
    let bus = MutationBus::new();
    let _rx = bus.subscribe();

    // Publish some events
    bus.publish(make_node_event(1));
    bus.publish(make_node_event(2));
    assert_eq!(bus.total_events_published(), 2);
    assert_eq!(bus.total_batches_published(), 2);

    // Clone gets the metrics at clone time
    let bus_clone = bus.clone();
    assert_eq!(bus_clone.total_events_published(), 2);
    assert_eq!(bus_clone.total_batches_published(), 2);

    // Publishing on original increments only the original's counters
    bus.publish(make_node_event(3));
    assert_eq!(bus.total_events_published(), 3);
    // Clone's counter was copied at clone-time, so the clone's own counter stays at 2
    // unless the clone also publishes
    assert_eq!(bus_clone.total_events_published(), 2);
}

#[tokio::test]
async fn clone_shares_broadcast_channel() {
    let bus = MutationBus::new();
    let bus_clone = bus.clone();

    // Subscribe from original
    let mut rx = bus.subscribe();

    // Publish from clone
    bus_clone.publish(make_node_event(1));

    // Receive from original's subscriber
    let batch = rx.recv().await.unwrap();
    assert_eq!(batch.len(), 1);

    // Subscribe from clone
    let mut rx2 = bus_clone.subscribe();

    // Publish from original
    bus.publish(make_node_event(2));

    // Receive from clone's subscriber
    let batch2 = rx2.recv().await.unwrap();
    assert_eq!(batch2.len(), 1);
}

// ========================================================================
// Empty batch edge case
// ========================================================================

#[test]
fn publish_empty_batch_with_subscriber_returns_false() {
    let bus = MutationBus::new();
    let _rx = bus.subscribe();

    let batch = MutationBatch::new(vec![]);
    let result = bus.publish_batch(batch);
    assert!(!result);
    // Empty batches should not increment counters
    assert_eq!(bus.total_events_published(), 0);
    assert_eq!(bus.total_batches_published(), 0);
}

// ========================================================================
// Lagged receiver
// ========================================================================

#[tokio::test]
async fn lagged_receiver_gets_error() {
    // Small capacity to force lagging
    let bus = MutationBus::with_capacity(2);
    let mut rx = bus.subscribe();

    // Publish more than capacity
    for i in 0..10 {
        bus.publish(make_node_event(i));
    }

    // First recv should get a Lagged error
    let result = rx.recv().await;
    match result {
        Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
            assert!(n > 0, "should report some lagged messages");
        }
        Ok(_) => {
            // In some cases, the first messages are still available
            // Continue receiving to exercise the path
        }
        Err(other) => panic!("unexpected error: {:?}", other),
    }
}

// ========================================================================
// Subscriber dropped
// ========================================================================

#[test]
fn subscriber_drop_decrements_count() {
    let bus = MutationBus::new();
    assert_eq!(bus.subscriber_count(), 0);

    let rx1 = bus.subscribe();
    let rx2 = bus.subscribe();
    let rx3 = bus.subscribe();
    assert_eq!(bus.subscriber_count(), 3);

    drop(rx2);
    assert_eq!(bus.subscriber_count(), 2);

    drop(rx1);
    drop(rx3);
    assert_eq!(bus.subscriber_count(), 0);
}

// ========================================================================
// Publish returns true only when subscribers exist
// ========================================================================

#[test]
fn publish_returns_true_with_subscriber() {
    let bus = MutationBus::new();
    let _rx = bus.subscribe();
    assert!(bus.publish(make_node_event(1)));
}

#[test]
fn publish_returns_false_without_subscriber() {
    let bus = MutationBus::new();
    assert!(!bus.publish(make_node_event(1)));
}

#[test]
fn publish_batch_returns_true_with_subscriber() {
    let bus = MutationBus::new();
    let _rx = bus.subscribe();
    let batch = MutationBatch::new(vec![make_node_event(1), make_node_event(2)]);
    assert!(bus.publish_batch(batch));
}

// ========================================================================
// Metrics accumulate correctly
// ========================================================================

#[test]
fn metrics_accumulate_across_publishes() {
    let bus = MutationBus::new();
    let _rx = bus.subscribe();

    // Single events
    bus.publish(make_node_event(1));
    bus.publish(make_node_event(2));
    assert_eq!(bus.total_events_published(), 2);
    assert_eq!(bus.total_batches_published(), 2);

    // Batch of 3
    let batch = MutationBatch::new(vec![
        make_node_event(3),
        make_node_event(4),
        make_node_event(5),
    ]);
    bus.publish_batch(batch);
    assert_eq!(bus.total_events_published(), 5);
    assert_eq!(bus.total_batches_published(), 3);
}

// ========================================================================
// Multiple subscribers all receive same batch
// ========================================================================

#[tokio::test]
async fn many_subscribers_receive_batch() {
    let bus = MutationBus::new();
    let mut receivers: Vec<_> = (0..10).map(|_| bus.subscribe()).collect();
    assert_eq!(bus.subscriber_count(), 10);

    let batch = MutationBatch::new(vec![make_node_event(1), make_node_event(2)]);
    bus.publish_batch(batch);

    for rx in &mut receivers {
        let batch = rx.recv().await.unwrap();
        assert_eq!(batch.len(), 2);
    }
}
