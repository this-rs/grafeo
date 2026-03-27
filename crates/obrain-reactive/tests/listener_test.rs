//! Tests for the MutationListener trait — default implementations,
//! accepts filtering, and on_batch delegation.

use async_trait::async_trait;
use obrain_common::types::NodeId;
use obrain_reactive::{MutationEvent, MutationListener, NodeSnapshot};
use parking_lot::Mutex;
use std::sync::Arc;

fn make_node_event(id: u64) -> MutationEvent {
    MutationEvent::NodeCreated {
        node: NodeSnapshot {
            id: NodeId::new(id),
            labels: smallvec::smallvec![arcstr::literal!("Test")],
            properties: vec![],
        },
    }
}

fn make_edge_event(id: u64) -> MutationEvent {
    MutationEvent::EdgeCreated {
        edge: obrain_reactive::EdgeSnapshot {
            id: obrain_common::types::EdgeId::new(id),
            src: NodeId::new(1),
            dst: NodeId::new(2),
            edge_type: arcstr::literal!("KNOWS"),
            properties: vec![],
        },
    }
}

// --- A simple listener that records individual on_event calls ---

struct EventRecorder {
    events: Arc<Mutex<Vec<String>>>,
}

#[async_trait]
impl MutationListener for EventRecorder {
    fn name(&self) -> &str {
        "event-recorder"
    }

    async fn on_event(&self, event: &MutationEvent) {
        self.events.lock().push(event.kind().to_string());
    }
}

// --- A listener with custom accepts ---

struct NodeOnlyRecorder {
    events: Arc<Mutex<Vec<String>>>,
}

#[async_trait]
impl MutationListener for NodeOnlyRecorder {
    fn name(&self) -> &str {
        "node-only-recorder"
    }

    async fn on_event(&self, event: &MutationEvent) {
        self.events.lock().push(event.kind().to_string());
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

// --- A listener that overrides on_batch ---

struct BatchRecorder {
    batch_sizes: Arc<Mutex<Vec<usize>>>,
}

#[async_trait]
impl MutationListener for BatchRecorder {
    fn name(&self) -> &str {
        "batch-recorder"
    }

    async fn on_event(&self, _event: &MutationEvent) {
        // Should not be called when on_batch is overridden
        panic!("on_event should not be called directly");
    }

    async fn on_batch(&self, events: &[MutationEvent]) {
        self.batch_sizes.lock().push(events.len());
    }
}

// ========================================================================
// Default on_batch delegates to on_event
// ========================================================================

#[tokio::test]
async fn default_on_batch_delegates_to_on_event() {
    let events_log = Arc::new(Mutex::new(Vec::new()));
    let listener = EventRecorder {
        events: Arc::clone(&events_log),
    };

    let events = vec![make_node_event(1), make_node_event(2), make_node_event(3)];

    // Call on_batch — default impl should call on_event for each
    listener.on_batch(&events).await;

    let recorded = events_log.lock().clone();
    assert_eq!(recorded.len(), 3);
    assert!(recorded.iter().all(|k| k == "node_created"));
}

#[tokio::test]
async fn default_on_batch_empty_events() {
    let events_log = Arc::new(Mutex::new(Vec::new()));
    let listener = EventRecorder {
        events: Arc::clone(&events_log),
    };

    listener.on_batch(&[]).await;
    assert!(events_log.lock().is_empty());
}

// ========================================================================
// Default accepts returns true for all events
// ========================================================================

#[test]
fn default_accepts_returns_true_for_all_variants() {
    let events_log = Arc::new(Mutex::new(Vec::new()));
    let listener = EventRecorder { events: events_log };

    let events = vec![
        make_node_event(1),
        MutationEvent::NodeUpdated {
            before: NodeSnapshot {
                id: NodeId::new(1),
                labels: smallvec::smallvec![],
                properties: vec![],
            },
            after: NodeSnapshot {
                id: NodeId::new(1),
                labels: smallvec::smallvec![arcstr::literal!("A")],
                properties: vec![],
            },
        },
        MutationEvent::NodeDeleted {
            node: NodeSnapshot {
                id: NodeId::new(1),
                labels: smallvec::smallvec![],
                properties: vec![],
            },
        },
        make_edge_event(1),
        MutationEvent::EdgeUpdated {
            before: obrain_reactive::EdgeSnapshot {
                id: obrain_common::types::EdgeId::new(1),
                src: NodeId::new(1),
                dst: NodeId::new(2),
                edge_type: arcstr::literal!("KNOWS"),
                properties: vec![],
            },
            after: obrain_reactive::EdgeSnapshot {
                id: obrain_common::types::EdgeId::new(1),
                src: NodeId::new(1),
                dst: NodeId::new(2),
                edge_type: arcstr::literal!("KNOWS"),
                properties: vec![],
            },
        },
        MutationEvent::EdgeDeleted {
            edge: obrain_reactive::EdgeSnapshot {
                id: obrain_common::types::EdgeId::new(1),
                src: NodeId::new(1),
                dst: NodeId::new(2),
                edge_type: arcstr::literal!("KNOWS"),
                properties: vec![],
            },
        },
    ];

    for event in &events {
        assert!(
            listener.accepts(event),
            "accepts should return true for {}",
            event.kind()
        );
    }
}

// ========================================================================
// Custom accepts filters correctly
// ========================================================================

#[test]
fn custom_accepts_filters_edge_events() {
    let events_log = Arc::new(Mutex::new(Vec::new()));
    let listener = NodeOnlyRecorder { events: events_log };

    // Node events should be accepted
    assert!(listener.accepts(&make_node_event(1)));
    assert!(listener.accepts(&MutationEvent::NodeDeleted {
        node: NodeSnapshot {
            id: NodeId::new(1),
            labels: smallvec::smallvec![],
            properties: vec![],
        },
    }));

    // Edge events should be rejected
    assert!(!listener.accepts(&make_edge_event(1)));
    assert!(!listener.accepts(&MutationEvent::EdgeDeleted {
        edge: obrain_reactive::EdgeSnapshot {
            id: obrain_common::types::EdgeId::new(1),
            src: NodeId::new(1),
            dst: NodeId::new(2),
            edge_type: arcstr::literal!("KNOWS"),
            properties: vec![],
        },
    }));
}

// ========================================================================
// Custom on_batch override
// ========================================================================

#[tokio::test]
async fn custom_on_batch_receives_batch() {
    let sizes = Arc::new(Mutex::new(Vec::new()));
    let listener = BatchRecorder {
        batch_sizes: Arc::clone(&sizes),
    };

    let events = vec![make_node_event(1), make_node_event(2), make_node_event(3)];
    listener.on_batch(&events).await;

    let recorded = sizes.lock().clone();
    assert_eq!(recorded, vec![3]);
}

#[tokio::test]
async fn custom_on_batch_called_multiple_times() {
    let sizes = Arc::new(Mutex::new(Vec::new()));
    let listener = BatchRecorder {
        batch_sizes: Arc::clone(&sizes),
    };

    listener.on_batch(&[make_node_event(1)]).await;
    listener
        .on_batch(&[make_node_event(2), make_node_event(3)])
        .await;
    listener.on_batch(&[]).await;

    let recorded = sizes.lock().clone();
    assert_eq!(recorded, vec![1, 2, 0]);
}

// ========================================================================
// Listener name
// ========================================================================

#[test]
fn listener_name_returns_correct_value() {
    let events_log = Arc::new(Mutex::new(Vec::new()));
    let listener = EventRecorder { events: events_log };
    assert_eq!(listener.name(), "event-recorder");

    let events_log2 = Arc::new(Mutex::new(Vec::new()));
    let listener2 = NodeOnlyRecorder {
        events: events_log2,
    };
    assert_eq!(listener2.name(), "node-only-recorder");

    let sizes = Arc::new(Mutex::new(Vec::new()));
    let listener3 = BatchRecorder { batch_sizes: sizes };
    assert_eq!(listener3.name(), "batch-recorder");
}

// ========================================================================
// Listener is Send + Sync
// ========================================================================

#[test]
fn listener_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    // These compile if the trait objects are Send + Sync
    let events_log = Arc::new(Mutex::new(Vec::new()));
    let listener: Arc<dyn MutationListener> = Arc::new(EventRecorder { events: events_log });
    // Should be usable across threads
    assert_send_sync::<Arc<dyn MutationListener>>();
    drop(listener);
}
