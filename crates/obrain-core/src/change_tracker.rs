//! In-memory change tracker for graph mutations.
//!
//! Records a bounded log of [`GraphEvent`]s in a ring buffer, enabling:
//!
//! - **Graph diff**: what changed between two points in time
//! - **Streaming subscriptions** (T4): notify listeners of mutations in real time
//! - **Incremental recomputation**: only recalculate features for changed nodes
//!
//! The tracker is **opt-in**: when disabled (`None` on `LpgStore`), there is
//! zero overhead — the `Option::is_none()` branch is eliminated by branch
//! prediction.
//!
//! # Ring buffer semantics
//!
//! Events are stored in a `VecDeque` with a fixed capacity. When the buffer
//! is full, the oldest events are evicted. Each event gets a monotonically
//! increasing sequence number, so consumers can request "events since sequence X".
//!
//! # Thread safety
//!
//! The `ChangeTracker` itself is **not** `Sync` — it is wrapped in a
//! `parking_lot::RwLock` on `LpgStore` (lock order **11**, after
//! `property_undo_log` at 10).
//!
//! # Example
//!
//! ```
//! use obrain_core::change_tracker::{ChangeTracker, GraphEvent, EntityRef};
//! use obrain_common::types::NodeId;
//!
//! let mut tracker = ChangeTracker::new(100);
//! tracker.record(GraphEvent::NodeCreated {
//!     id: NodeId(1),
//!     labels: vec!["Person".to_string()],
//!     timestamp: 0,
//! });
//! assert_eq!(tracker.len(), 1);
//! let events = tracker.since(0);
//! assert_eq!(events.len(), 1);
//! ```

use obrain_common::types::{EdgeId, NodeId, PropertyKey, Value};
use std::collections::VecDeque;

// ============================================================================
// Types
// ============================================================================

/// A mutation event recorded by the change tracker.
#[derive(Debug, Clone)]
pub enum GraphEvent {
    /// A node was created.
    NodeCreated {
        /// The new node's ID.
        id: NodeId,
        /// Labels assigned at creation.
        labels: Vec<String>,
        /// Logical timestamp (epoch or sequence).
        timestamp: u64,
    },
    /// A node was deleted.
    NodeDeleted {
        /// The deleted node's ID.
        id: NodeId,
        /// Logical timestamp.
        timestamp: u64,
    },
    /// An edge was created.
    EdgeCreated {
        /// The new edge's ID.
        id: EdgeId,
        /// Source node.
        src: NodeId,
        /// Destination node.
        dst: NodeId,
        /// Edge type label.
        edge_type: String,
        /// Logical timestamp.
        timestamp: u64,
    },
    /// An edge was deleted.
    EdgeDeleted {
        /// The deleted edge's ID.
        id: EdgeId,
        /// Logical timestamp.
        timestamp: u64,
    },
    /// A property was set on a node or edge.
    PropertySet {
        /// The entity (node or edge) that was modified.
        entity: EntityRef,
        /// Property key.
        key: PropertyKey,
        /// Previous value (`None` if the property was new).
        old_value: Option<Value>,
        /// New value.
        new_value: Value,
        /// Logical timestamp.
        timestamp: u64,
    },
}

/// Reference to either a node or an edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EntityRef {
    /// A node reference.
    Node(NodeId),
    /// An edge reference.
    Edge(EdgeId),
}

/// Summary of changes between two sequence numbers.
#[derive(Debug, Clone, Default)]
pub struct GraphDiff {
    /// Number of nodes added.
    pub nodes_added: usize,
    /// Number of nodes removed.
    pub nodes_removed: usize,
    /// Number of edges added.
    pub edges_added: usize,
    /// Number of edges removed.
    pub edges_removed: usize,
    /// Number of property changes.
    pub properties_changed: usize,
}

// ============================================================================
// ChangeTracker
// ============================================================================

/// Bounded in-memory log of graph mutation events.
///
/// Uses a ring buffer (`VecDeque`) with a fixed capacity. When full, the
/// oldest events are evicted. Each event is assigned a monotonically
/// increasing sequence number.
///
/// # Complexity
///
/// - `record()`: O(1) amortized
/// - `since()`: O(n) where n = events returned
/// - `diff()`: O(n) where n = events in range
pub struct ChangeTracker {
    events: VecDeque<(u64, GraphEvent)>,
    capacity: usize,
    sequence: u64,
}

impl ChangeTracker {
    /// Creates a new change tracker with the given ring buffer capacity.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Maximum number of events to retain (default recommendation: 10_000)
    pub fn new(capacity: usize) -> Self {
        Self {
            events: VecDeque::with_capacity(capacity.min(1024)),
            capacity,
            sequence: 0,
        }
    }

    /// Records a new graph event.
    ///
    /// Assigns a monotonically increasing sequence number. If the buffer is
    /// full, the oldest event is evicted.
    ///
    /// # Complexity
    ///
    /// O(1) amortized.
    pub fn record(&mut self, event: GraphEvent) {
        if self.events.len() >= self.capacity {
            self.events.pop_front();
        }
        self.events.push_back((self.sequence, event));
        self.sequence += 1;
    }

    /// Returns all events with sequence number >= `since_sequence`.
    ///
    /// # Arguments
    ///
    /// * `since_sequence` - Return events from this sequence number onwards
    ///
    /// # Returns
    ///
    /// Slice of `(sequence, event)` pairs, oldest first.
    pub fn since(&self, since_sequence: u64) -> Vec<&(u64, GraphEvent)> {
        self.events
            .iter()
            .filter(|(seq, _)| *seq >= since_sequence)
            .collect()
    }

    /// Computes a structured diff between two sequence numbers.
    ///
    /// # Arguments
    ///
    /// * `from` - Start sequence (inclusive)
    /// * `to` - End sequence (exclusive)
    ///
    /// # Returns
    ///
    /// A [`GraphDiff`] summarizing the changes.
    pub fn diff(&self, from: u64, to: u64) -> GraphDiff {
        let mut diff = GraphDiff::default();
        for (seq, event) in &self.events {
            if *seq >= from && *seq < to {
                match event {
                    GraphEvent::NodeCreated { .. } => diff.nodes_added += 1,
                    GraphEvent::NodeDeleted { .. } => diff.nodes_removed += 1,
                    GraphEvent::EdgeCreated { .. } => diff.edges_added += 1,
                    GraphEvent::EdgeDeleted { .. } => diff.edges_removed += 1,
                    GraphEvent::PropertySet { .. } => diff.properties_changed += 1,
                }
            }
        }
        diff
    }

    /// Returns the current sequence number (next event will get this number).
    pub fn current_sequence(&self) -> u64 {
        self.sequence
    }

    /// Returns the number of events currently in the buffer.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Returns `true` if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Returns the ring buffer capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_records_node_creation() {
        let mut tracker = ChangeTracker::new(100);
        tracker.record(GraphEvent::NodeCreated {
            id: NodeId(1),
            labels: vec!["Person".to_string()],
            timestamp: 0,
        });
        assert_eq!(tracker.len(), 1);
        let events = tracker.since(0);
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0].1, GraphEvent::NodeCreated { .. }));
    }

    #[test]
    fn test_records_edge_creation() {
        let mut tracker = ChangeTracker::new(100);
        tracker.record(GraphEvent::EdgeCreated {
            id: EdgeId(10),
            src: NodeId(1),
            dst: NodeId(2),
            edge_type: "KNOWS".to_string(),
            timestamp: 0,
        });
        assert_eq!(tracker.len(), 1);
        assert!(matches!(
            tracker.since(0)[0].1,
            GraphEvent::EdgeCreated { .. }
        ));
    }

    #[test]
    fn test_records_property_set() {
        let mut tracker = ChangeTracker::new(100);
        tracker.record(GraphEvent::PropertySet {
            entity: EntityRef::Node(NodeId(1)),
            key: "name".into(),
            old_value: None,
            new_value: Value::String("Alice".into()),
            timestamp: 0,
        });
        let events = tracker.since(0);
        assert_eq!(events.len(), 1);
        match &events[0].1 {
            GraphEvent::PropertySet {
                old_value,
                new_value,
                ..
            } => {
                assert!(old_value.is_none());
                assert_eq!(*new_value, Value::String("Alice".into()));
            }
            _ => panic!("expected PropertySet"),
        }
    }

    #[test]
    fn test_ring_buffer() {
        let mut tracker = ChangeTracker::new(3);
        for i in 0..5 {
            tracker.record(GraphEvent::NodeCreated {
                id: NodeId(i),
                labels: vec![],
                timestamp: i,
            });
        }
        // Capacity is 3, so only last 3 events remain
        assert_eq!(tracker.len(), 3);
        let events = tracker.since(0);
        // Oldest surviving event has sequence 2
        assert_eq!(events[0].0, 2);
        assert_eq!(events[2].0, 4);
    }

    #[test]
    fn test_diff() {
        let mut tracker = ChangeTracker::new(100);
        tracker.record(GraphEvent::NodeCreated {
            id: NodeId(1),
            labels: vec![],
            timestamp: 0,
        });
        tracker.record(GraphEvent::NodeCreated {
            id: NodeId(2),
            labels: vec![],
            timestamp: 1,
        });
        tracker.record(GraphEvent::EdgeCreated {
            id: EdgeId(1),
            src: NodeId(1),
            dst: NodeId(2),
            edge_type: "LINK".to_string(),
            timestamp: 2,
        });
        tracker.record(GraphEvent::NodeDeleted {
            id: NodeId(1),
            timestamp: 3,
        });
        tracker.record(GraphEvent::PropertySet {
            entity: EntityRef::Node(NodeId(2)),
            key: "x".into(),
            old_value: None,
            new_value: Value::Int64(42),
            timestamp: 4,
        });

        let diff = tracker.diff(0, 5);
        assert_eq!(diff.nodes_added, 2);
        assert_eq!(diff.nodes_removed, 1);
        assert_eq!(diff.edges_added, 1);
        assert_eq!(diff.edges_removed, 0);
        assert_eq!(diff.properties_changed, 1);

        // Partial diff
        let diff2 = tracker.diff(2, 4);
        assert_eq!(diff2.edges_added, 1);
        assert_eq!(diff2.nodes_removed, 1);
    }

    #[test]
    fn test_disabled() {
        // Simulates Option<ChangeTracker> = None
        let tracker: Option<ChangeTracker> = None;
        assert!(tracker.is_none());
        // Zero overhead — the None branch is never entered
    }

    #[test]
    fn test_since() {
        let mut tracker = ChangeTracker::new(100);
        for i in 0..10 {
            tracker.record(GraphEvent::NodeCreated {
                id: NodeId(i),
                labels: vec![],
                timestamp: i,
            });
        }
        let events = tracker.since(7);
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].0, 7);
        assert_eq!(events[1].0, 8);
        assert_eq!(events[2].0, 9);
    }

    #[test]
    fn test_concurrent_access() {
        use parking_lot::RwLock;
        use std::sync::Arc;

        let tracker = Arc::new(RwLock::new(ChangeTracker::new(1000)));

        let handles: Vec<_> = (0..4)
            .map(|t| {
                let tracker = Arc::clone(&tracker);
                std::thread::spawn(move || {
                    for i in 0..100 {
                        tracker.write().record(GraphEvent::NodeCreated {
                            id: NodeId(t * 100 + i),
                            labels: vec![],
                            timestamp: t * 100 + i,
                        });
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        let guard = tracker.read();
        assert_eq!(guard.len(), 400);
    }
}
