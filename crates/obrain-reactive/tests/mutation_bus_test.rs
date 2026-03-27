//! Integration tests for the MutationBus.

use grafeo_common::types::{EdgeId, NodeId};
use grafeo_reactive::{EdgeSnapshot, MutationBatch, MutationBus, MutationEvent, NodeSnapshot};

fn make_node(id: u64, labels: &[&str]) -> NodeSnapshot {
    NodeSnapshot {
        id: NodeId::new(id),
        labels: labels.iter().map(|l| arcstr::ArcStr::from(*l)).collect(),
        properties: vec![],
    }
}

fn make_edge(id: u64, src: u64, dst: u64, edge_type: &str) -> EdgeSnapshot {
    EdgeSnapshot {
        id: EdgeId::new(id),
        src: NodeId::new(src),
        dst: NodeId::new(dst),
        edge_type: arcstr::ArcStr::from(edge_type),
        properties: vec![],
    }
}

// --- subscribe → publish → receive ---

#[tokio::test]
async fn subscribe_publish_receive() {
    let bus = MutationBus::new();
    let mut rx = bus.subscribe();

    let event = MutationEvent::NodeCreated {
        node: make_node(1, &["Person"]),
    };
    assert!(bus.publish(event));

    let batch = rx.recv().await.unwrap();
    assert_eq!(batch.len(), 1);
    assert_eq!(batch.events[0].kind(), "node_created");
}

#[tokio::test]
async fn subscribe_publish_batch_receive() {
    let bus = MutationBus::new();
    let mut rx = bus.subscribe();

    let events = vec![
        MutationEvent::NodeCreated {
            node: make_node(1, &["Person"]),
        },
        MutationEvent::EdgeCreated {
            edge: make_edge(1, 1, 2, "KNOWS"),
        },
        MutationEvent::NodeUpdated {
            before: make_node(2, &["Person"]),
            after: make_node(2, &["Person", "Employee"]),
        },
        MutationEvent::NodeDeleted {
            node: make_node(3, &["Temp"]),
        },
        MutationEvent::EdgeUpdated {
            before: make_edge(2, 1, 3, "LIKES"),
            after: make_edge(2, 1, 3, "LIKES"),
        },
        MutationEvent::EdgeDeleted {
            edge: make_edge(3, 2, 3, "FOLLOWS"),
        },
    ];

    let batch = MutationBatch::new(events);
    assert!(bus.publish_batch(batch));

    let received = rx.recv().await.unwrap();
    assert_eq!(received.len(), 6);
    assert_eq!(received.events[0].kind(), "node_created");
    assert_eq!(received.events[1].kind(), "edge_created");
    assert_eq!(received.events[2].kind(), "node_updated");
    assert_eq!(received.events[3].kind(), "node_deleted");
    assert_eq!(received.events[4].kind(), "edge_updated");
    assert_eq!(received.events[5].kind(), "edge_deleted");
}

// --- multi-subscriber ---

#[tokio::test]
async fn multi_subscriber_all_receive() {
    let bus = MutationBus::new();
    let mut rx1 = bus.subscribe();
    let mut rx2 = bus.subscribe();
    let mut rx3 = bus.subscribe();

    let event = MutationEvent::NodeCreated {
        node: make_node(42, &["Document"]),
    };
    bus.publish(event);

    let b1 = rx1.recv().await.unwrap();
    let b2 = rx2.recv().await.unwrap();
    let b3 = rx3.recv().await.unwrap();

    assert_eq!(b1.len(), 1);
    assert_eq!(b2.len(), 1);
    assert_eq!(b3.len(), 1);
    assert_eq!(b1.events[0].kind(), "node_created");
    assert_eq!(b2.events[0].kind(), "node_created");
    assert_eq!(b3.events[0].kind(), "node_created");
}

#[tokio::test]
async fn multi_subscriber_independent_receive() {
    let bus = MutationBus::new();
    let mut rx1 = bus.subscribe();
    let mut rx2 = bus.subscribe();

    // Publish two batches
    bus.publish(MutationEvent::NodeCreated {
        node: make_node(1, &["A"]),
    });
    bus.publish(MutationEvent::NodeCreated {
        node: make_node(2, &["B"]),
    });

    // Both subscribers should get both batches
    let b1_1 = rx1.recv().await.unwrap();
    let b1_2 = rx1.recv().await.unwrap();
    let b2_1 = rx2.recv().await.unwrap();
    let b2_2 = rx2.recv().await.unwrap();

    assert_eq!(b1_1.len(), 1);
    assert_eq!(b1_2.len(), 1);
    assert_eq!(b2_1.len(), 1);
    assert_eq!(b2_2.len(), 1);
}

// --- publish without subscriber (no panic) ---

#[test]
fn publish_no_subscriber_no_panic() {
    let bus = MutationBus::new();

    // Should not panic, just return false
    let result = bus.publish(MutationEvent::NodeCreated {
        node: make_node(1, &["Person"]),
    });
    assert!(!result);
    assert_eq!(bus.total_events_published(), 0);
}

#[test]
fn publish_batch_no_subscriber_no_panic() {
    let bus = MutationBus::new();

    let batch = MutationBatch::new(vec![
        MutationEvent::NodeCreated {
            node: make_node(1, &["A"]),
        },
        MutationEvent::NodeCreated {
            node: make_node(2, &["B"]),
        },
    ]);
    let result = bus.publish_batch(batch);
    assert!(!result);
    assert_eq!(bus.total_events_published(), 0);
}

#[test]
fn publish_empty_batch_no_subscriber_no_panic() {
    let bus = MutationBus::new();
    let batch = MutationBatch::new(vec![]);
    let result = bus.publish_batch(batch);
    assert!(!result);
}

// --- overhead benchmark (inline, not criterion) ---

#[test]
fn publish_overhead_no_subscriber_under_5us() {
    let bus = MutationBus::new();

    // Warm up
    for _ in 0..100 {
        bus.publish(MutationEvent::NodeCreated {
            node: make_node(1, &["Person"]),
        });
    }

    // Measure
    let iterations = 10_000;
    let start = std::time::Instant::now();
    for i in 0..iterations {
        bus.publish(MutationEvent::NodeCreated {
            node: make_node(i, &["Person"]),
        });
    }
    let elapsed = start.elapsed();
    let per_op = elapsed / iterations as u32;

    // Assert < 5µs per operation
    assert!(
        per_op.as_micros() < 5,
        "MutationBus overhead per publish (no subscriber) was {:?}, expected < 5µs",
        per_op
    );
}

// --- subscriber count tracking ---

#[test]
fn subscriber_count_tracks_correctly() {
    let bus = MutationBus::new();
    assert_eq!(bus.subscriber_count(), 0);

    let rx1 = bus.subscribe();
    assert_eq!(bus.subscriber_count(), 1);

    let rx2 = bus.subscribe();
    assert_eq!(bus.subscriber_count(), 2);

    drop(rx1);
    assert_eq!(bus.subscriber_count(), 1);

    drop(rx2);
    assert_eq!(bus.subscriber_count(), 0);
}

// --- metrics tracking ---

#[tokio::test]
async fn metrics_track_published_events() {
    let bus = MutationBus::new();
    let mut _rx = bus.subscribe();

    assert_eq!(bus.total_events_published(), 0);
    assert_eq!(bus.total_batches_published(), 0);

    bus.publish(MutationEvent::NodeCreated {
        node: make_node(1, &["A"]),
    });
    assert_eq!(bus.total_events_published(), 1);
    assert_eq!(bus.total_batches_published(), 1);

    let batch = MutationBatch::new(vec![
        MutationEvent::NodeCreated {
            node: make_node(2, &["B"]),
        },
        MutationEvent::EdgeCreated {
            edge: make_edge(1, 1, 2, "KNOWS"),
        },
    ]);
    bus.publish_batch(batch);
    assert_eq!(bus.total_events_published(), 3);
    assert_eq!(bus.total_batches_published(), 2);
}

// --- custom capacity ---

#[test]
fn custom_capacity_bus() {
    let bus = MutationBus::with_capacity(4);
    let _rx = bus.subscribe();

    // Should work fine with small capacity
    for i in 0..4 {
        bus.publish(MutationEvent::NodeCreated {
            node: make_node(i, &["Test"]),
        });
    }
    assert_eq!(bus.total_events_published(), 4);
}

// --- clone shares channel ---

#[tokio::test]
async fn cloned_bus_shares_channel() {
    let bus = MutationBus::new();
    let bus_clone = bus.clone();
    let mut rx = bus.subscribe();

    // Publish from clone, receive from original's subscriber
    bus_clone.publish(MutationEvent::NodeCreated {
        node: make_node(1, &["Person"]),
    });

    let batch = rx.recv().await.unwrap();
    assert_eq!(batch.len(), 1);
}
