//! Event subscription system for real-time graph mutation notifications.
//!
//! Provides a synchronous observer pattern where consumers can subscribe to
//! [`GraphEvent`]s with optional filtering. Callbacks are invoked inline during
//! mutations — **callbacks MUST NOT call mutation methods on the store** (this
//! would deadlock the `parking_lot::RwLock`).
//!
//! ## Architecture
//!
//! - **Sync callbacks** (`Box<dyn Fn(&GraphEvent) + Send + Sync>`) — no tokio dependency
//! - **Pre-call filtering** via [`EventFilter`] — skips callbacks that don't match
//! - **Zero overhead** when no subscribers are registered (`Option::is_none()`)
//! - **Panic isolation** — a panicking callback doesn't block other subscribers
//!
//! ## Usage (historical — LpgStore-coupled)
//!
//! The observer pattern below is exposed by the `LpgStore` backend via
//! `enable_tracking`, `enable_subscriptions`, and `subscribe` (inherent
//! methods). `SubstrateStore` does not currently implement this observer
//! surface — a stream-based replacement driven by the substrate WAL is
//! tracked under T17 W2c (kernel_manager / hilbert_manager degeneralisation).
//!
//! ```ignore
//! // LpgStore-only example — kept for illustration; does not run under
//! // `cargo test --doc` because `SubstrateStore` has no equivalent API yet.
//! use obrain_core::graph::lpg::LpgStore;
//! use obrain_core::subscription::{EventFilter, EventType};
//! use std::sync::Arc;
//! use std::sync::atomic::{AtomicUsize, Ordering};
//!
//! let mut store = LpgStore::new().unwrap();
//! store.enable_tracking(1000);
//! store.enable_subscriptions();
//!
//! let counter = Arc::new(AtomicUsize::new(0));
//! let counter_clone = counter.clone();
//!
//! let filter = EventFilter {
//!     event_types: Some([EventType::NodeCreated].into_iter().collect()),
//!     ..Default::default()
//! };
//!
//! store.subscribe(filter, Box::new(move |_event| {
//!     counter_clone.fetch_add(1, Ordering::Relaxed);
//! }));
//!
//! store.create_node(&["Person"]);
//! store.create_node(&["Document"]);
//!
//! assert_eq!(counter.load(Ordering::Relaxed), 2);
//! ```
//!
//! ## Lock ordering
//!
//! The `SubscriptionManager` is held at **lock order 12** (after `ChangeTracker` = 11).
//! The notify path acquires a read lock only.

use crate::change_tracker::GraphEvent;
use std::collections::HashSet;

// ============================================================================
// Types
// ============================================================================

/// Unique identifier for a subscription.
pub type SubscriptionId = u64;

/// Event type discriminant for filtering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EventType {
    /// A node was created.
    NodeCreated,
    /// A node was deleted.
    NodeDeleted,
    /// An edge was created.
    EdgeCreated,
    /// An edge was deleted.
    EdgeDeleted,
    /// A property was set on a node or edge.
    PropertySet,
}

impl EventType {
    /// Extract the event type from a [`GraphEvent`].
    #[must_use]
    pub fn from_event(event: &GraphEvent) -> Self {
        match event {
            GraphEvent::NodeCreated { .. } => Self::NodeCreated,
            GraphEvent::NodeDeleted { .. } => Self::NodeDeleted,
            GraphEvent::EdgeCreated { .. } => Self::EdgeCreated,
            GraphEvent::EdgeDeleted { .. } => Self::EdgeDeleted,
            GraphEvent::PropertySet { .. } => Self::PropertySet,
        }
    }
}

/// Filter for selecting which events a subscriber receives.
///
/// All fields are optional — `None` means "match all". When multiple fields
/// are set, they are ANDed together.
#[derive(Debug, Clone, Default)]
pub struct EventFilter {
    /// Only receive these event types. `None` = all types.
    pub event_types: Option<HashSet<EventType>>,
    /// Only receive events for nodes with these labels.
    /// Matches if the node has ANY of the specified labels.
    /// `None` = all labels.
    pub labels: Option<HashSet<String>>,
    /// Only receive events for edges with these types.
    /// `None` = all edge types.
    pub edge_types: Option<HashSet<String>>,
    /// Only receive PropertySet events for these property keys.
    /// `None` = all properties.
    pub properties: Option<HashSet<String>>,
}

impl EventFilter {
    /// Check if an event matches this filter.
    #[must_use]
    pub fn matches(&self, event: &GraphEvent) -> bool {
        // Check event type filter
        if let Some(ref types) = self.event_types
            && !types.contains(&EventType::from_event(event))
        {
            return false;
        }

        // Check label filter (only for node events)
        if let Some(ref labels) = self.labels {
            match event {
                GraphEvent::NodeCreated {
                    labels: node_labels,
                    ..
                } => {
                    if !node_labels.iter().any(|l| labels.contains(l.as_str())) {
                        return false;
                    }
                }
                // Non-node events pass label filter (OR semantics: label filter
                // only constrains node events)
                _ => {}
            }
        }

        // Check edge type filter
        if let Some(ref edge_types) = self.edge_types {
            match event {
                GraphEvent::EdgeCreated { edge_type, .. } => {
                    if !edge_types.contains(edge_type.as_str()) {
                        return false;
                    }
                }
                _ => {}
            }
        }

        // Check property key filter
        if let Some(ref props) = self.properties {
            match event {
                GraphEvent::PropertySet { key, .. } => {
                    if !props.contains(key.as_str()) {
                        return false;
                    }
                }
                // Non-property events pass this filter
                _ => {}
            }
        }

        true
    }
}

// ============================================================================
// SubscriptionManager
// ============================================================================

/// Manages event subscriptions and dispatches notifications.
///
/// Thread-safe: designed to be held in a `parking_lot::RwLock` on the host
/// graph store (currently `LpgStore`; substrate equivalent tracked under T17 W2c).
/// The `notify()` path uses a read lock; `subscribe()`/`unsubscribe()` use write.
pub struct SubscriptionManager {
    subscribers: Vec<(
        SubscriptionId,
        EventFilter,
        Box<dyn Fn(&GraphEvent) + Send + Sync>,
    )>,
    next_id: u64,
}

impl SubscriptionManager {
    /// Create a new empty subscription manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            subscribers: Vec::new(),
            next_id: 1,
        }
    }

    /// Register a new subscriber with the given filter and callback.
    ///
    /// Returns a [`SubscriptionId`] that can be used to [`unsubscribe`](Self::unsubscribe).
    ///
    /// # Warning
    ///
    /// The callback **MUST NOT** call mutation methods on the host graph store — doing
    /// so will deadlock (the notification path holds a read lock on the store's
    /// internal structures, and mutations require write locks).
    pub fn subscribe(
        &mut self,
        filter: EventFilter,
        callback: Box<dyn Fn(&GraphEvent) + Send + Sync>,
    ) -> SubscriptionId {
        let id = self.next_id;
        self.next_id += 1;
        self.subscribers.push((id, filter, callback));
        id
    }

    /// Remove a subscriber by ID.
    ///
    /// Returns `true` if the subscriber was found and removed.
    pub fn unsubscribe(&mut self, id: SubscriptionId) -> bool {
        let len_before = self.subscribers.len();
        self.subscribers.retain(|(sid, _, _)| *sid != id);
        self.subscribers.len() < len_before
    }

    /// Notify all matching subscribers of an event.
    ///
    /// Filters are checked before invoking callbacks. Panicking callbacks are
    /// caught via `std::panic::catch_unwind` — a panic in one callback does not
    /// prevent other subscribers from being notified.
    ///
    /// # Cost
    ///
    /// O(subscribers) — linear scan. For most use cases (<100 subscribers),
    /// this is negligible.
    pub fn notify(&self, event: &GraphEvent) {
        if self.subscribers.is_empty() {
            return;
        }

        for (_, filter, callback) in &self.subscribers {
            if filter.matches(event) {
                // Isolate panics: one bad callback shouldn't crash the store
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    callback(event);
                }));
            }
        }
    }

    /// Returns the number of active subscribers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.subscribers.len()
    }

    /// Returns `true` if there are no subscribers.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.subscribers.is_empty()
    }
}

impl Default for SubscriptionManager {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::change_tracker::EntityRef;
    use obrain_common::types::{NodeId, PropertyKey, Value};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn node_created_event(id: NodeId, labels: &[&str]) -> GraphEvent {
        GraphEvent::NodeCreated {
            id,
            labels: labels.iter().map(|s| s.to_string()).collect(),
            timestamp: 1,
        }
    }

    fn edge_created_event() -> GraphEvent {
        GraphEvent::EdgeCreated {
            id: obrain_common::types::EdgeId(100),
            src: NodeId(1),
            dst: NodeId(2),
            edge_type: "KNOWS".to_string(),
            timestamp: 1,
        }
    }

    fn property_set_event(key: &str) -> GraphEvent {
        GraphEvent::PropertySet {
            entity: EntityRef::Node(NodeId(1)),
            key: PropertyKey::new(key),
            old_value: None,
            new_value: Value::Int64(42),
            timestamp: 1,
        }
    }

    #[test]
    fn test_subscribe_all() {
        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();

        let mut mgr = SubscriptionManager::new();
        mgr.subscribe(
            EventFilter::default(),
            Box::new(move |_| {
                c.fetch_add(1, Ordering::Relaxed);
            }),
        );

        mgr.notify(&node_created_event(NodeId(1), &["Person"]));
        mgr.notify(&edge_created_event());
        mgr.notify(&property_set_event("name"));

        assert_eq!(counter.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn test_filtered_labels() {
        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();

        let mut mgr = SubscriptionManager::new();
        mgr.subscribe(
            EventFilter {
                labels: Some(["Person".to_string()].into_iter().collect()),
                ..Default::default()
            },
            Box::new(move |_| {
                c.fetch_add(1, Ordering::Relaxed);
            }),
        );

        mgr.notify(&node_created_event(NodeId(1), &["Person"]));
        mgr.notify(&node_created_event(NodeId(2), &["Document"]));
        mgr.notify(&node_created_event(NodeId(3), &["Person", "Admin"]));

        // Person and Person+Admin match, Document does not
        assert_eq!(counter.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_filtered_edge_type() {
        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();

        let mut mgr = SubscriptionManager::new();
        mgr.subscribe(
            EventFilter {
                edge_types: Some(["KNOWS".to_string()].into_iter().collect()),
                ..Default::default()
            },
            Box::new(move |_| {
                c.fetch_add(1, Ordering::Relaxed);
            }),
        );

        mgr.notify(&edge_created_event()); // KNOWS → matches
        mgr.notify(&GraphEvent::EdgeCreated {
            id: obrain_common::types::EdgeId(101),
            src: NodeId(1),
            dst: NodeId(2),
            edge_type: "FOLLOWS".to_string(),
            timestamp: 1,
        }); // FOLLOWS → no match

        // Non-edge events pass the edge_type filter
        mgr.notify(&node_created_event(NodeId(5), &["X"])); // passes

        assert_eq!(counter.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_multiple_subscribers() {
        let c1 = Arc::new(AtomicUsize::new(0));
        let c2 = Arc::new(AtomicUsize::new(0));
        let c1_clone = c1.clone();
        let c2_clone = c2.clone();

        let mut mgr = SubscriptionManager::new();
        mgr.subscribe(
            EventFilter {
                event_types: Some([EventType::NodeCreated].into_iter().collect()),
                ..Default::default()
            },
            Box::new(move |_| {
                c1_clone.fetch_add(1, Ordering::Relaxed);
            }),
        );
        mgr.subscribe(
            EventFilter {
                event_types: Some([EventType::EdgeCreated].into_iter().collect()),
                ..Default::default()
            },
            Box::new(move |_| {
                c2_clone.fetch_add(1, Ordering::Relaxed);
            }),
        );

        mgr.notify(&node_created_event(NodeId(1), &["A"]));
        mgr.notify(&edge_created_event());
        mgr.notify(&node_created_event(NodeId(2), &["B"]));

        assert_eq!(c1.load(Ordering::Relaxed), 2); // 2 node events
        assert_eq!(c2.load(Ordering::Relaxed), 1); // 1 edge event
    }

    #[test]
    fn test_unsubscribe() {
        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();

        let mut mgr = SubscriptionManager::new();
        let id = mgr.subscribe(
            EventFilter::default(),
            Box::new(move |_| {
                c.fetch_add(1, Ordering::Relaxed);
            }),
        );

        mgr.notify(&node_created_event(NodeId(1), &["A"]));
        assert_eq!(counter.load(Ordering::Relaxed), 1);

        assert!(mgr.unsubscribe(id));
        mgr.notify(&node_created_event(NodeId(2), &["A"]));
        assert_eq!(counter.load(Ordering::Relaxed), 1); // unchanged

        // Unsubscribing again returns false
        assert!(!mgr.unsubscribe(id));
    }

    #[test]
    fn test_no_subscriber_no_overhead() {
        let mgr = SubscriptionManager::new();
        // Should be essentially a no-op
        mgr.notify(&node_created_event(NodeId(1), &["A"]));
        mgr.notify(&edge_created_event());
        assert!(mgr.is_empty());
    }

    #[test]
    fn test_callback_panic_isolation() {
        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();

        let mut mgr = SubscriptionManager::new();

        // First subscriber panics
        mgr.subscribe(
            EventFilter::default(),
            Box::new(|_| {
                panic!("intentional test panic");
            }),
        );

        // Second subscriber should still be called
        mgr.subscribe(
            EventFilter::default(),
            Box::new(move |_| {
                c.fetch_add(1, Ordering::Relaxed);
            }),
        );

        mgr.notify(&node_created_event(NodeId(1), &["A"]));

        // The second callback was still invoked despite the first panicking
        assert_eq!(counter.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_property_filter() {
        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();

        let mut mgr = SubscriptionManager::new();
        mgr.subscribe(
            EventFilter {
                properties: Some(["name".to_string()].into_iter().collect()),
                ..Default::default()
            },
            Box::new(move |_| {
                c.fetch_add(1, Ordering::Relaxed);
            }),
        );

        mgr.notify(&property_set_event("name")); // matches
        mgr.notify(&property_set_event("age")); // no match
        mgr.notify(&node_created_event(NodeId(1), &["A"])); // passes (not a property event)

        assert_eq!(counter.load(Ordering::Relaxed), 2);
    }
}
