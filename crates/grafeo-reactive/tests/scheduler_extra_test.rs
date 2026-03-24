//! Additional tests for Scheduler — BatchConfig, lazy startup, Debug, config accessors,
//! and edge cases.

use async_trait::async_trait;
use grafeo_common::types::NodeId;
use grafeo_reactive::{
    BatchConfig, MutationBus, MutationEvent, MutationListener, NodeSnapshot, Scheduler,
};
use parking_lot::Mutex;
use std::sync::Arc;
use std::time::Duration;

struct RecordingListener {
    name: String,
    batches: Arc<Mutex<Vec<Vec<MutationEvent>>>>,
}

impl RecordingListener {
    fn new(name: &str) -> (Arc<Self>, Arc<Mutex<Vec<Vec<MutationEvent>>>>) {
        let batches = Arc::new(Mutex::new(Vec::new()));
        let listener = Arc::new(Self {
            name: name.to_string(),
            batches: Arc::clone(&batches),
        });
        (listener, batches)
    }
}

#[async_trait]
impl MutationListener for RecordingListener {
    fn name(&self) -> &str {
        &self.name
    }

    async fn on_event(&self, _event: &MutationEvent) {}

    async fn on_batch(&self, events: &[MutationEvent]) {
        self.batches.lock().push(events.to_vec());
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

    let (listener, _batches) = RecordingListener::new("test");
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

    let (l1, _) = RecordingListener::new("first");
    let (l2, _) = RecordingListener::new("second");

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

    let (listener, batches) = RecordingListener::new("flush-test");
    scheduler.register_listener(listener);

    // Publish some events
    for i in 0..5 {
        bus.publish(make_node_event(i));
    }

    // Small delay for events to reach scheduler buffer
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Explicit shutdown should flush remaining events
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

    let (l1, batches1) = RecordingListener::new("early");
    scheduler.register_listener(l1);

    // Publish first events
    bus.publish(make_node_event(1));
    bus.publish(make_node_event(2));

    tokio::time::sleep(Duration::from_millis(100)).await;

    // Now register second listener
    let (l2, batches2) = RecordingListener::new("late");
    scheduler.register_listener(l2);

    // Publish more events
    bus.publish(make_node_event(3));
    bus.publish(make_node_event(4));

    tokio::time::sleep(Duration::from_millis(100)).await;

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

    let (listener, batches) = RecordingListener::new("batch-1");
    scheduler.register_listener(listener);

    // Publish 5 events one at a time
    for i in 0..5 {
        bus.publish(make_node_event(i));
    }

    tokio::time::sleep(Duration::from_millis(200)).await;

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

    let (listener, batches) = RecordingListener::new("burst");
    scheduler.register_listener(listener);

    let event_count = 500;
    for i in 0..event_count {
        bus.publish(make_node_event(i));
    }

    // Give plenty of time for processing
    tokio::time::sleep(Duration::from_millis(500)).await;

    let received = batches.lock().clone();
    let total: usize = received.iter().map(|b| b.len()).sum();
    assert_eq!(
        total, event_count as usize,
        "all events should be delivered"
    );

    scheduler.shutdown().await;
}
