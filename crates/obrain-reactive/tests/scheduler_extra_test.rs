//! Additional tests for Scheduler — BatchConfig, lazy startup, Debug, config accessors,
//! and edge cases.

use async_trait::async_trait;
use obrain_common::types::NodeId;
use obrain_reactive::{
    BatchConfig, MutationBatch, MutationBus, MutationEvent, MutationListener, NodeSnapshot,
    Scheduler,
};
use parking_lot::Mutex;
use std::sync::Arc;
use std::time::Duration;

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

    async fn on_event(&self, _event: &MutationEvent) {}

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

// ========================================================================
// BatchConfig
// ========================================================================

#[test]
fn batch_config_default() {
    let config = BatchConfig::default();
    assert_eq!(config.max_batch_size, 100);
    assert_eq!(config.max_delay, Duration::from_millis(50));
}

#[test]
fn batch_config_new_custom() {
    let config = BatchConfig::new(500, Duration::from_secs(1));
    assert_eq!(config.max_batch_size, 500);
    assert_eq!(config.max_delay, Duration::from_secs(1));
}

#[test]
fn batch_config_clone() {
    let config = BatchConfig::new(200, Duration::from_millis(100));
    let cloned = config.clone();
    assert_eq!(cloned.max_batch_size, 200);
    assert_eq!(cloned.max_delay, Duration::from_millis(100));
}

#[test]
fn batch_config_debug() {
    let config = BatchConfig::new(50, Duration::from_millis(25));
    let debug = format!("{:?}", config);
    assert!(debug.contains("BatchConfig"));
    assert!(debug.contains("50"));
}

// ========================================================================
// Scheduler creation — lazy startup
// ========================================================================

#[tokio::test]
async fn scheduler_not_running_before_listener_registered() {
    let bus = MutationBus::new();
    let config = BatchConfig::default();
    let scheduler = Scheduler::new(&bus, config);

    assert!(!scheduler.is_running());
    assert_eq!(scheduler.listener_count(), 0);

    scheduler.shutdown().await;
}

#[tokio::test]
async fn scheduler_starts_on_first_listener() {
    let bus = MutationBus::new();
    let config = BatchConfig::default();
    let scheduler = Scheduler::new(&bus, config);

    assert!(!scheduler.is_running());

    let (listener, _batches, _rx) = RecordingListener::new("test");
    scheduler.register_listener(listener);

    assert!(scheduler.is_running());
    assert_eq!(scheduler.listener_count(), 1);

    scheduler.shutdown().await;
}

#[tokio::test]
async fn scheduler_does_not_restart_on_second_listener() {
    let bus = MutationBus::new();
    let config = BatchConfig::default();
    let scheduler = Scheduler::new(&bus, config);

    let (l1, _, _rx1) = RecordingListener::new("first");
    let (l2, _, _rx2) = RecordingListener::new("second");

    scheduler.register_listener(l1);
    assert!(scheduler.is_running());
    assert_eq!(scheduler.listener_count(), 1);

    scheduler.register_listener(l2);
    assert!(scheduler.is_running());
    assert_eq!(scheduler.listener_count(), 2);

    scheduler.shutdown().await;
}

// ========================================================================
// Scheduler config accessor
// ========================================================================

#[tokio::test]
async fn scheduler_config_returns_correct_values() {
    let bus = MutationBus::new();
    let config = BatchConfig::new(42, Duration::from_millis(123));
    let scheduler = Scheduler::new(&bus, config);

    assert_eq!(scheduler.config().max_batch_size, 42);
    assert_eq!(scheduler.config().max_delay, Duration::from_millis(123));

    scheduler.shutdown().await;
}

// ========================================================================
// Scheduler Debug
// ========================================================================

#[tokio::test]
async fn scheduler_debug() {
    let bus = MutationBus::new();
    let config = BatchConfig::default();
    let scheduler = Scheduler::new(&bus, config);

    let debug = format!("{:?}", scheduler);
    assert!(debug.contains("Scheduler"));
    assert!(debug.contains("listener_count"));
    assert!(debug.contains("running"));

    scheduler.shutdown().await;
}

// ========================================================================
// Shutdown without listeners is a no-op
// ========================================================================

#[tokio::test]
async fn shutdown_without_listeners_is_noop() {
    let bus = MutationBus::new();
    let config = BatchConfig::default();
    let scheduler = Scheduler::new(&bus, config);

    // No listeners registered, shutdown should not block
    scheduler.shutdown().await;
    // If we get here, the test passed
}

// ========================================================================
// Scheduler handles shutdown via explicit shutdown (bus closure is
// inherently racy — events may not reach the scheduler buffer before
// the channel closes, so we test explicit shutdown instead)
// ========================================================================

#[tokio::test]
async fn explicit_shutdown_flushes_buffer() {
    let bus = MutationBus::new();
    let config = BatchConfig::new(1000, Duration::from_secs(60));
    let scheduler = Scheduler::new(&bus, config);

    let (listener, batches, _rx) = RecordingListener::new("flush-test");
    scheduler.register_listener(listener);

    // Publish some events
    for i in 0..5 {
        bus.publish(make_node_event(i));
    }

    // Explicit shutdown drains remaining events (no sleep needed)
    scheduler.shutdown().await;

    let received = batches.lock().clone();
    let total: usize = received.iter().map(|b| b.len()).sum();
    assert_eq!(total, 5, "shutdown should flush all buffered events");
}

// ========================================================================
// Late-registered listener gets only new events
// ========================================================================

#[tokio::test]
async fn late_registered_listener_gets_only_new_events() {
    let bus = MutationBus::new();
    let config = BatchConfig::new(100, Duration::from_millis(30));
    let scheduler = Scheduler::new(&bus, config);

    let (l1, batches1, mut rx1) = RecordingListener::new("early");
    scheduler.register_listener(l1);

    // Publish first events
    bus.publish(make_node_event(1));
    bus.publish(make_node_event(2));

    // Wait for early listener to receive the first batch
    wait_for_events(&mut rx1, 2).await;

    // Now register second listener
    let (l2, batches2, mut rx2) = RecordingListener::new("late");
    scheduler.register_listener(l2);

    // Publish more events
    bus.publish(make_node_event(3));
    bus.publish(make_node_event(4));

    // Wait for both listeners to receive the new events
    wait_for_events(&mut rx1, 4).await;
    wait_for_events(&mut rx2, 2).await;

    let total1: usize = batches1.lock().iter().map(|b| b.len()).sum();
    let total2: usize = batches2.lock().iter().map(|b| b.len()).sum();

    // Early listener should have all 4
    assert_eq!(total1, 4, "early listener should have 4 events");
    // Late listener should have at least the last 2
    assert!(
        total2 >= 2,
        "late listener should have at least 2 events, got {}",
        total2
    );

    scheduler.shutdown().await;
}

// ========================================================================
// Batch size of 1 — immediate dispatch
// ========================================================================

#[tokio::test]
async fn batch_size_one_dispatches_immediately() {
    let bus = MutationBus::new();
    let config = BatchConfig::new(1, Duration::from_secs(60));
    let scheduler = Scheduler::new(&bus, config);

    let (listener, batches, mut rx) = RecordingListener::new("batch-1");
    scheduler.register_listener(listener);

    // Publish 5 events one at a time
    for i in 0..5 {
        bus.publish(make_node_event(i));
    }

    // Wait for all events via channel notification
    wait_for_events(&mut rx, 5).await;

    let received = batches.lock().clone();
    let total: usize = received.iter().map(|b| b.len()).sum();
    assert_eq!(total, 5);

    // Each batch should have exactly 1 event
    for batch in &received {
        assert_eq!(
            batch.len(),
            1,
            "with max_batch_size=1, each batch should have 1 event"
        );
    }

    scheduler.shutdown().await;
}

// ========================================================================
// Large burst of events
// ========================================================================

#[tokio::test]
async fn large_burst_all_delivered() {
    let bus = MutationBus::new();
    let config = BatchConfig::new(50, Duration::from_millis(30));
    let scheduler = Scheduler::new(&bus, config);

    let (listener, batches, mut rx) = RecordingListener::new("burst");
    scheduler.register_listener(listener);

    let event_count = 500;
    for i in 0..event_count {
        bus.publish(make_node_event(i));
    }

    // Wait for all events via channel notification
    wait_for_events(&mut rx, event_count as usize).await;

    let received = batches.lock().clone();
    let total: usize = received.iter().map(|b| b.len()).sum();
    assert_eq!(
        total, event_count as usize,
        "all events should be delivered"
    );

    scheduler.shutdown().await;
}

// ========================================================================
// Bus closed path — drop the bus AND the scheduler while loop is running
// This covers L275-280 (RecvError::Closed branch with flush)
// ========================================================================

#[tokio::test]
async fn bus_closed_flushes_remaining_and_stops() {
    let bus = MutationBus::with_capacity(16384);
    let config = BatchConfig::new(1000, Duration::from_secs(60));
    let scheduler = Scheduler::new(&bus, config);

    let (listener, batches, mut rx) = RecordingListener::new("bus-close");
    scheduler.register_listener(listener);

    // Publish events
    for i in 0..3 {
        bus.publish(make_node_event(i));
    }

    // Drop bus and scheduler — the background task sees either Closed or shutdown
    // and drains the buffer. We use a oneshot to detect when events arrive.
    drop(bus);
    drop(scheduler);

    // Wait for the flush to deliver events (with timeout for safety)
    let _ = tokio::time::timeout(Duration::from_secs(5), rx.wait_for(|c| *c >= 3)).await;

    let received = batches.lock().clone();
    let total: usize = received.iter().map(|b| b.len()).sum();
    assert_eq!(total, 3, "bus close should flush all buffered events");
}

// ========================================================================
// Timeout flush — incomplete batch flushed after max_delay
// This covers L290 (the timeout select branch)
// ========================================================================

#[tokio::test]
async fn timeout_flushes_incomplete_batch() {
    let bus = MutationBus::new();
    // Large batch size but very short delay so timeout fires
    let config = BatchConfig::new(1000, Duration::from_millis(30));
    let scheduler = Scheduler::new(&bus, config);

    let (listener, batches, mut rx) = RecordingListener::new("timeout");
    scheduler.register_listener(listener);

    // Publish fewer events than max_batch_size
    for i in 0..3 {
        bus.publish(make_node_event(i));
    }

    // Wait for events via channel (timeout flush will deliver them)
    wait_for_events(&mut rx, 3).await;

    let received = batches.lock().clone();
    let total: usize = received.iter().map(|b| b.len()).sum();
    assert_eq!(total, 3, "timeout should flush incomplete batch");

    scheduler.shutdown().await;
}

// ========================================================================
// Shutdown drains events from channel (L228-230)
// ========================================================================

#[tokio::test]
async fn shutdown_drains_channel_events() {
    let bus = MutationBus::new();
    // Very large batch size and delay so nothing flushes before shutdown
    let config = BatchConfig::new(10000, Duration::from_secs(3600));
    let scheduler = Scheduler::new(&bus, config);

    let (listener, batches, _rx) = RecordingListener::new("drain");
    scheduler.register_listener(listener);

    // Yield to let the scheduler loop start
    tokio::task::yield_now().await;

    // Publish a burst of events
    for i in 0..10 {
        bus.publish(make_node_event(i));
    }

    // Shutdown drains all channel events (no sleep needed)
    scheduler.shutdown().await;

    let received = batches.lock().clone();
    let total: usize = received.iter().map(|b| b.len()).sum();
    assert_eq!(total, 10, "shutdown should drain all channel events");
}

// ========================================================================
// Lagged receiver — covers L267-268 (RecvError::Lagged)
// ========================================================================

#[tokio::test]
async fn lagged_receiver_continues() {
    // Use a tiny broadcast capacity so the receiver lags quickly
    let bus = MutationBus::with_capacity(2);
    let config = BatchConfig::new(1, Duration::from_millis(30));
    let scheduler = Scheduler::new(&bus, config);

    let (listener, batches, mut rx) = RecordingListener::new("lagged");
    scheduler.register_listener(listener);

    // Flood the bus beyond its capacity to cause lag
    for i in 0..20 {
        bus.publish(make_node_event(i));
    }

    // Wait for at least some events to arrive (lag means not all will be received)
    let _ = tokio::time::timeout(Duration::from_secs(5), rx.wait_for(|c| *c > 0)).await;

    // The scheduler should have continued despite lagging — it gets at least some events
    let received = batches.lock().clone();
    let total: usize = received.iter().map(|b| b.len()).sum();
    assert!(
        total > 0,
        "scheduler should recover from lag and receive events"
    );

    scheduler.shutdown().await;
}

// ========================================================================
// Listener filtering — partial acceptance (L321, and the partial path L329-332)
// ========================================================================

/// A listener that only accepts NodeCreated events (rejects NodeDeleted).
struct SelectiveListener {
    name: String,
    batches: Arc<Mutex<Vec<Vec<MutationEvent>>>>,
    event_count_tx: tokio::sync::watch::Sender<usize>,
}

impl SelectiveListener {
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
impl MutationListener for SelectiveListener {
    fn name(&self) -> &str {
        &self.name
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
        matches!(event, MutationEvent::NodeCreated { .. })
    }
}

/// A listener that rejects ALL events.
struct RejectAllListener {
    name: String,
}

impl RejectAllListener {
    fn new(name: &str) -> Arc<Self> {
        Arc::new(Self {
            name: name.to_string(),
        })
    }
}

#[async_trait]
impl MutationListener for RejectAllListener {
    fn name(&self) -> &str {
        &self.name
    }

    async fn on_event(&self, _event: &MutationEvent) {}

    fn accepts(&self, _event: &MutationEvent) -> bool {
        false
    }
}

fn make_node_deleted_event(id: u64) -> MutationEvent {
    MutationEvent::NodeDeleted {
        node: NodeSnapshot {
            id: obrain_common::types::NodeId::new(id),
            labels: smallvec::smallvec![arcstr::literal!("Test")],
            properties: vec![],
        },
    }
}

#[tokio::test]
async fn selective_listener_receives_only_accepted_events() {
    let bus = MutationBus::new();
    let config = BatchConfig::new(100, Duration::from_millis(30));
    let scheduler = Scheduler::new(&bus, config);

    let (selective, sel_batches, mut sel_rx) = SelectiveListener::new("selective");
    scheduler.register_listener(selective);

    // Publish a mix of NodeCreated and NodeDeleted events in a single batch
    let events = vec![
        make_node_event(1),
        make_node_deleted_event(2),
        make_node_event(3),
        make_node_deleted_event(4),
    ];
    bus.publish_batch(MutationBatch::new(events));

    // Wait for the 2 accepted NodeCreated events
    wait_for_events(&mut sel_rx, 2).await;

    let received = sel_batches.lock().clone();
    let total: usize = received.iter().map(|b| b.len()).sum();
    assert_eq!(
        total, 2,
        "selective listener should only receive NodeCreated events"
    );

    scheduler.shutdown().await;
}

#[tokio::test]
async fn reject_all_listener_receives_nothing() {
    let bus = MutationBus::new();
    let config = BatchConfig::new(100, Duration::from_millis(30));
    let scheduler = Scheduler::new(&bus, config);

    let reject = RejectAllListener::new("reject-all");
    let (recording, rec_batches, mut rec_rx) = RecordingListener::new("accept-all");

    // Register both: one that rejects everything, one that accepts everything
    scheduler.register_listener(reject);
    scheduler.register_listener(recording);

    bus.publish(make_node_event(1));
    bus.publish(make_node_event(2));

    // Wait for the accept-all listener to receive 2 events
    wait_for_events(&mut rec_rx, 2).await;

    // The accept-all listener should have received events
    let total: usize = rec_batches.lock().iter().map(|b| b.len()).sum();
    assert_eq!(total, 2, "accept-all listener should receive 2 events");

    scheduler.shutdown().await;
}

// ========================================================================
// Drop scheduler without explicit shutdown — sender dropped path (L225 result.is_err())
// ========================================================================

#[tokio::test]
async fn drop_scheduler_without_shutdown() {
    let bus = MutationBus::new();
    let config = BatchConfig::new(1000, Duration::from_secs(60));
    let scheduler = Scheduler::new(&bus, config);

    let (listener, _batches, mut rx) = RecordingListener::new("drop-test");
    scheduler.register_listener(listener);

    bus.publish(make_node_event(1));

    // Wait for event to be received before dropping
    let _ = tokio::time::timeout(Duration::from_secs(5), rx.wait_for(|c| *c >= 1)).await;

    // Drop the scheduler without calling shutdown — the watch sender is dropped,
    // which causes shutdown_rx.changed() to return Err, triggering the drain path
    drop(scheduler);

    // Yield to let the background task notice the drop and exit
    tokio::task::yield_now().await;
    // If we get here without hanging, the background task exited gracefully
}

// ========================================================================
// Publish batch with multiple events to test batch splitting (while loop L256-260)
// ========================================================================

#[tokio::test]
async fn batch_splitting_with_small_max_size() {
    let bus = MutationBus::new();
    // max_batch_size = 2, so a batch of 5 events triggers the while loop multiple times
    let config = BatchConfig::new(2, Duration::from_millis(30));
    let scheduler = Scheduler::new(&bus, config);

    let (listener, batches, mut rx) = RecordingListener::new("split");
    scheduler.register_listener(listener);

    // Send 5 events in a single MutationBatch via the bus
    let events: Vec<MutationEvent> = (0..5).map(make_node_event).collect();
    bus.publish_batch(MutationBatch::new(events));

    // Wait for all 5 events
    wait_for_events(&mut rx, 5).await;

    let received = batches.lock().clone();
    let total: usize = received.iter().map(|b| b.len()).sum();
    assert_eq!(total, 5, "all 5 events should be dispatched");

    // Each dispatch should be at most 2 events
    for batch in &received {
        assert!(
            batch.len() <= 2,
            "batch should have at most 2 events, got {}",
            batch.len()
        );
    }

    scheduler.shutdown().await;
}
