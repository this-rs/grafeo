//! Mutation event types published by the [`MutationBus`](crate::MutationBus).

use arcstr::ArcStr;
use grafeo_common::types::{EdgeId, NodeId, PropertyKey, Value};
use smallvec::SmallVec;
use web_time::Instant;

/// Snapshot of a node's state at a point in time.
#[derive(Debug, Clone)]
pub struct NodeSnapshot {
    /// Node identifier.
    pub id: NodeId,
    /// Labels on the node.
    pub labels: SmallVec<[ArcStr; 2]>,
    /// Properties on the node.
    pub properties: Vec<(PropertyKey, Value)>,
}

/// Snapshot of an edge's state at a point in time.
#[derive(Debug, Clone)]
pub struct EdgeSnapshot {
    /// Edge identifier.
    pub id: EdgeId,
    /// Source node.
    pub src: NodeId,
    /// Destination node.
    pub dst: NodeId,
    /// Edge type/label.
    pub edge_type: ArcStr,
    /// Properties on the edge.
    pub properties: Vec<(PropertyKey, Value)>,
}

/// A single mutation event published after a successful commit.
///
/// Each variant carries the complete data needed for downstream processing,
/// so listeners never need to query back into the store.
#[derive(Debug, Clone)]
pub enum MutationEvent {
    /// A new node was created.
    NodeCreated {
        /// The created node's snapshot.
        node: NodeSnapshot,
    },

    /// A node was updated (properties or labels changed).
    NodeUpdated {
        /// State before the mutation.
        before: NodeSnapshot,
        /// State after the mutation.
        after: NodeSnapshot,
    },

    /// A node was deleted.
    NodeDeleted {
        /// The deleted node's last known state.
        node: NodeSnapshot,
    },

    /// A new edge was created.
    EdgeCreated {
        /// The created edge's snapshot.
        edge: EdgeSnapshot,
    },

    /// An edge was updated (properties changed).
    EdgeUpdated {
        /// State before the mutation.
        before: EdgeSnapshot,
        /// State after the mutation.
        after: EdgeSnapshot,
    },

    /// An edge was deleted.
    EdgeDeleted {
        /// The deleted edge's last known state.
        edge: EdgeSnapshot,
    },
}

impl MutationEvent {
    /// Returns the node or edge ID involved in this event.
    pub fn entity_id(&self) -> EntityRef {
        match self {
            Self::NodeCreated { node } => EntityRef::Node(node.id),
            Self::NodeUpdated { after, .. } => EntityRef::Node(after.id),
            Self::NodeDeleted { node } => EntityRef::Node(node.id),
            Self::EdgeCreated { edge } => EntityRef::Edge(edge.id),
            Self::EdgeUpdated { after, .. } => EntityRef::Edge(after.id),
            Self::EdgeDeleted { edge } => EntityRef::Edge(edge.id),
        }
    }

    /// Returns a short description of the event kind.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::NodeCreated { .. } => "node_created",
            Self::NodeUpdated { .. } => "node_updated",
            Self::NodeDeleted { .. } => "node_deleted",
            Self::EdgeCreated { .. } => "edge_created",
            Self::EdgeUpdated { .. } => "edge_updated",
            Self::EdgeDeleted { .. } => "edge_deleted",
        }
    }
}

/// Reference to a graph entity (node or edge).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EntityRef {
    /// A node reference.
    Node(NodeId),
    /// An edge reference.
    Edge(EdgeId),
}

/// A batch of mutation events from a single transaction commit.
#[derive(Debug, Clone)]
pub struct MutationBatch {
    /// The events in commit order.
    pub events: Vec<MutationEvent>,
    /// When the batch was created (for latency tracking).
    pub timestamp: Instant,
}

impl MutationBatch {
    /// Creates a new batch from a list of events.
    pub fn new(events: Vec<MutationEvent>) -> Self {
        Self {
            events,
            timestamp: Instant::now(),
        }
    }

    /// Returns `true` if this batch has no events.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Returns the number of events in this batch.
    pub fn len(&self) -> usize {
        self.events.len()
    }
}
