//! Integration tests for the Scheduler — batching and dispatch behavior.

use async_trait::async_trait;
use grafeo_common::types::NodeId;
use grafeo_reactive::{
    BatchConfig, MutationBus, MutationEvent, MutationListener, NodeSnapshot, Scheduler,
};
use parking_lot::Mutex;
use std::sync::Arc;
use std::time::Duration;

// --- Helper listener that records received batches and notifies via watch ---

struct RecordingListener {
    name: String,
    batches: Arc<Mutex<Vec<Vec<MutationEvent>>>>,
    event_count_tx: tokio::sync::watch::Sender<usize>,
}

impl RecordingListener {
    fn new(
        name: &str,
    ) -> (
        Arc<Self>,
        Arc<Mutex<Vec<Vec<MutationEvent>>>>,
        tokio::sync::watch::Receiver<usize>,
    ) {
        let batches = Arc::new(Mutex::new(Vec::new()));
        let (tx, rx) = tokio::sync::watch::channel(0usize);
        let listener = Arc::new(Self {
            name: name.to_string(),
            batches: Arc::clone(&batches),
            event_count_tx: tx,
        });
        (listener, batches, rx)
    }
}

#[async_trait]
impl MutationListener for RecordingListener {
    fn name(&self) -> &str {
        &self.name
    }

    async fn on_event(&self, _event: &MutationEvent) {
        // Not used — we override on_batch
    }

    async fn on_batch(&self, events: &[MutationEvent]) {
        let total = {
            let mut b = self.batches.lock();
            b.push(events.to_vec());
            b.iter().map(|batch| batch.len()).sum()
        };
        let _ = self.event_count_tx.send(total);
    }
}

fn make_node_event(id: u64) -> MutationEvent {
    MutationEvent::NodeCreated {
        node: NodeSnapshot {
            id: NodeId::new(id),
            labels: smallvec::smallvec![arcstr::literal!("Test")],
            properties: vec![],
        },
    }
}

/// Wait until at least `expected` events have been received, with a safety timeout.
async fn wait_for_events(rx: &mut tokio::sync::watch::Receiver<usize>, expected: usize) {
    tokio::time::timeout(
        Duration::from_secs(5),
        rx.wait_for(|count| *count >= expected),
    )
    .await
    .expect("timed out waiting for events")
    .expect("watch channel closed");
}

// --- Tests ---

#[tokio::test]
async fn listener_receives_batch_after_flush() {
    let bus = MutationBus::new();
    let config = BatchConfig::new(10, Duration::from_millis(50));
    let scheduler = Scheduler::new(&bus, config);

    let (listener, batches, mut rx) = RecordingListener::new("test");
    scheduler.register_listener(listener);

    // Publish 3 events — below max_batch_size, so they'll flush on timeout
    for i in 0..3 {
        bus.publish(make_node_event(i));
    }

    // Wait for all events via channel notification
    wait_for_events(&mut rx, 3).await;

    let received = batches.lock().clone();
    assert!(
        !received.is_empty(),
        "listener should have received at least one batch"
    );
    let total_events: usize = received.iter().map(|b| b.len()).sum();
    assert_eq!(total_events, 3, "should have received all 3 events");

    scheduler.shutdown().await;
}

#[tokio::test]
async fn batching_splits_by_max_batch_size() {
    let bus = MutationBus::new();
    // max_batch_size = 10, so 50 events → should get ~5 batches
    let config = BatchConfig::new(10, Duration::from_secs(10));
    let scheduler = Scheduler::new(&bus, config);

    let (listener, batches, mut rx) = RecordingListener::new("batcher");
    scheduler.register_listener(listener);

    // Publish 50 events
    for i in 0..50 {
        bus.publish(make_node_event(i));
    }

    // Wait for all events via channel notification
    wait_for_events(&mut rx, 50).await;

    let received = batches.lock().clone();
    let total_events: usize = received.iter().map(|b| b.len()).sum();
    assert_eq!(total_events, 50, "should have received all 50 events");

    // Each batch should have at most 10 events
    for (i, batch) in received.iter().enumerate() {
        assert!(
            batch.len() <= 10,
            "batch {i} had {} events, expected <= 10",
            batch.len()
        );
    }

    // Should have at least 5 batches (may be more if events arrive across multiple bus batches)
    assert!(
        received.len() >= 5,
        "expected at least 5 batches, got {}",
        received.len()
    );

    scheduler.shutdown().await;
}

#[tokio::test]
async fn timeout_flushes_incomplete_batch() {
    let bus = MutationBus::new();
    // Large batch size, short timeout — should flush on timeout
    let config = BatchConfig::new(1000, Duration::from_millis(30));
    let scheduler = Scheduler::new(&bus, config);

    let (listener, batches, mut rx) = RecordingListener::new("timeout-test");
    scheduler.register_listener(listener);

    // Publish just 5 events (well below max_batch_size=1000)
    for i in 0..5 {
        bus.publish(make_node_event(i));
    }

    // Wait for all events via channel notification (timeout flush will deliver them)
    wait_for_events(&mut rx, 5).await;

    let received = batches.lock().clone();
    assert!(
        !received.is_empty(),
        "timeout should have flushed the batch"
    );
    let total: usize = received.iter().map(|b| b.len()).sum();
    assert_eq!(total, 5);

    scheduler.shutdown().await;
}

#[tokio::test]
async fn multiple_listeners_all_receive() {
    let bus = MutationBus::new();
    let config = BatchConfig::new(100, Duration::from_millis(30));
    let scheduler = Scheduler::new(&bus, config);

    let (l1, batches1, mut rx1) = RecordingListener::new("listener-1");
    let (l2, batches2, mut rx2) = RecordingListener::new("listener-2");
    let (l3, batches3, mut rx3) = RecordingListener::new("listener-3");
    scheduler.register_listener(l1);
    scheduler.register_listener(l2);
    scheduler.register_listener(l3);

    bus.publish(make_node_event(1));
    bus.publish(make_node_event(2));

    // Wait for all listeners to receive events
    wait_for_events(&mut rx1, 2).await;
    wait_for_events(&mut rx2, 2).await;
    wait_for_events(&mut rx3, 2).await;

    for (name, batches) in [("l1", &batches1), ("l2", &batches2), ("l3", &batches3)] {
        let total: usize = batches.lock().iter().map(|b| b.len()).sum();
        assert_eq!(
            total, 2,
            "{name} should have received 2 events, got {total}"
        );
    }

    scheduler.shutdown().await;
}

#[tokio::test]
async fn shutdown_drains_pending_events() {
    let bus = MutationBus::new();
    let config = BatchConfig::new(1000, Duration::from_secs(60)); // Long timeout
    let scheduler = Scheduler::new(&bus, config);

    let (listener, batches, _rx) = RecordingListener::new("drain-test");
    scheduler.register_listener(listener);

    // Publish events
    for i in 0..10 {
        bus.publish(make_node_event(i));
    }

    // Shutdown drains all pending events (no sleep needed)
    scheduler.shutdown().await;

    let received = batches.lock().clone();
    let total: usize = received.iter().map(|b| b.len()).sum();
    assert_eq!(total, 10, "shutdown should drain all pending events");
}

#[tokio::test]
async fn accepts_filter_skips_unwanted_events() {
    struct NodeOnlyListener {
        batches: Arc<Mutex<Vec<Vec<MutationEvent>>>>,
        event_count_tx: tokio::sync::watch::Sender<usize>,
    }

    #[async_trait]
    impl MutationListener for NodeOnlyListener {
        fn name(&self) -> &str {
            "node-only"
        }

        async fn on_event(&self, _event: &MutationEvent) {}

        async fn on_batch(&self, events: &[MutationEvent]) {
            let total = {
                let mut b = self.batches.lock();
                b.push(events.to_vec());
                b.iter().map(|batch| batch.len()).sum()
            };
            let _ = self.event_count_tx.send(total);
        }

        fn accepts(&self, event: &MutationEvent) -> bool {
            matches!(
                event,
                MutationEvent::NodeCreated { .. }
                    | MutationEvent::NodeUpdated { .. }
                    | MutationEvent::NodeDeleted { .. }
            )
        }
    }

    let bus = MutationBus::new();
    let config = BatchConfig::new(100, Duration::from_millis(30));
    let scheduler = Scheduler::new(&bus, config);

    let batches = Arc::new(Mutex::new(Vec::new()));
    let (tx, mut rx) = tokio::sync::watch::channel(0usize);
    let listener = Arc::new(NodeOnlyListener {
        batches: Arc::clone(&batches),
        event_count_tx: tx,
    });
    scheduler.register_listener(listener);

    // Publish mix of node and edge events
    bus.publish(make_node_event(1));
    bus.publish(MutationEvent::EdgeCreated {
        edge: grafeo_reactive::EdgeSnapshot {
            id: grafeo_common::types::EdgeId::new(1),
            src: NodeId::new(1),
            dst: NodeId::new(2),
            edge_type: arcstr::literal!("KNOWS"),
            properties: vec![],
        },
    });
    bus.publish(make_node_event(2));

    // Wait for the 2 accepted node events
    wait_for_events(&mut rx, 2).await;

    let received = batches.lock().clone();
    let total: usize = received.iter().map(|b| b.len()).sum();
    assert_eq!(total, 2, "node-only listener should skip edge events");

    scheduler.shutdown().await;
}
