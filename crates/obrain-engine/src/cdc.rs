//! Change Data Capture (CDC) for tracking entity mutations.
//!
//! When the `cdc` feature is enabled, the database records every mutation
//! (create, update, delete) with before/after property snapshots. This
//! enables audit trails, temporal queries, and downstream sync.
//!
//! # Example
//!
//! ```no_run
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use grafeo_engine::GrafeoDB;
//! use grafeo_common::types::Value;
//!
//! let db = GrafeoDB::new_in_memory();
//! let id = db.create_node(&["Person"]);
//! db.set_node_property(id, "name", Value::from("Alix"));
//! db.set_node_property(id, "name", Value::from("Gus"));
//!
//! let history = db.history(id)?;
//! assert_eq!(history.len(), 3); // create + 2 updates
//! # Ok(())
//! # }
//! ```

use grafeo_common::types::{EdgeId, EpochId, NodeId, Value};
use hashbrown::HashMap as HbHashMap;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

/// The kind of mutation that occurred.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChangeKind {
    /// A new entity was created.
    Create,
    /// An existing entity was updated (property set or removed).
    Update,
    /// An entity was deleted.
    Delete,
}

/// A unique identifier for a graph entity (node or edge).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EntityId {
    /// A node identifier.
    Node(NodeId),
    /// An edge identifier.
    Edge(EdgeId),
}

impl From<NodeId> for EntityId {
    fn from(id: NodeId) -> Self {
        Self::Node(id)
    }
}

impl From<EdgeId> for EntityId {
    fn from(id: EdgeId) -> Self {
        Self::Edge(id)
    }
}

impl EntityId {
    /// Returns the raw u64 value for binding layers.
    #[must_use]
    pub fn as_u64(&self) -> u64 {
        match self {
            Self::Node(id) => id.as_u64(),
            Self::Edge(id) => id.as_u64(),
        }
    }

    /// Returns `true` if this is a node identifier.
    #[must_use]
    pub fn is_node(&self) -> bool {
        matches!(self, Self::Node(_))
    }
}

/// A recorded change event with before/after property snapshots.
#[derive(Debug, Clone)]
pub struct ChangeEvent {
    /// The entity that was changed.
    pub entity_id: EntityId,
    /// The kind of change.
    pub kind: ChangeKind,
    /// MVCC epoch when the change occurred.
    pub epoch: EpochId,
    /// Wall-clock timestamp (milliseconds since Unix epoch).
    pub timestamp: u64,
    /// Properties before the change (None for Create).
    pub before: Option<HashMap<String, Value>>,
    /// Properties after the change (None for Delete).
    pub after: Option<HashMap<String, Value>>,
}

/// The CDC log that records entity mutations.
///
/// Thread-safe: uses `RwLock<HashMap>` for concurrent access.
#[derive(Debug)]
pub struct CdcLog {
    events: RwLock<HbHashMap<EntityId, Vec<ChangeEvent>>>,
}

impl CdcLog {
    /// Creates a new empty CDC log.
    #[must_use]
    pub fn new() -> Self {
        Self {
            events: RwLock::new(HbHashMap::new()),
        }
    }

    /// Records a change event.
    pub fn record(&self, event: ChangeEvent) {
        self.events
            .write()
            .entry(event.entity_id)
            .or_default()
            .push(event);
    }

    /// Records a node creation.
    pub fn record_create_node(
        &self,
        id: NodeId,
        epoch: EpochId,
        props: Option<HashMap<String, Value>>,
    ) {
        self.record(ChangeEvent {
            entity_id: EntityId::Node(id),
            kind: ChangeKind::Create,
            epoch,
            timestamp: now_millis(),
            before: None,
            after: props,
        });
    }

    /// Records an edge creation.
    pub fn record_create_edge(
        &self,
        id: EdgeId,
        epoch: EpochId,
        props: Option<HashMap<String, Value>>,
    ) {
        self.record(ChangeEvent {
            entity_id: EntityId::Edge(id),
            kind: ChangeKind::Create,
            epoch,
            timestamp: now_millis(),
            before: None,
            after: props,
        });
    }

    /// Records a property update.
    pub fn record_update(
        &self,
        entity_id: EntityId,
        epoch: EpochId,
        key: &str,
        old_value: Option<Value>,
        new_value: Value,
    ) {
        let before = old_value.map(|v| {
            let mut m = HashMap::new();
            m.insert(key.to_string(), v);
            m
        });
        let mut after_map = HashMap::new();
        after_map.insert(key.to_string(), new_value);

        self.record(ChangeEvent {
            entity_id,
            kind: ChangeKind::Update,
            epoch,
            timestamp: now_millis(),
            before,
            after: Some(after_map),
        });
    }

    /// Records an entity deletion.
    pub fn record_delete(
        &self,
        entity_id: EntityId,
        epoch: EpochId,
        props: Option<HashMap<String, Value>>,
    ) {
        self.record(ChangeEvent {
            entity_id,
            kind: ChangeKind::Delete,
            epoch,
            timestamp: now_millis(),
            before: props,
            after: None,
        });
    }

    /// Returns all change events for an entity, ordered by epoch.
    #[must_use]
    pub fn history(&self, entity_id: EntityId) -> Vec<ChangeEvent> {
        self.events
            .read()
            .get(&entity_id)
            .cloned()
            .unwrap_or_default()
    }

    /// Returns change events for an entity since the given epoch.
    #[must_use]
    pub fn history_since(&self, entity_id: EntityId, since_epoch: EpochId) -> Vec<ChangeEvent> {
        self.events
            .read()
            .get(&entity_id)
            .map(|events| {
                events
                    .iter()
                    .filter(|e| e.epoch >= since_epoch)
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Returns all change events across all entities in an epoch range.
    #[must_use]
    pub fn changes_between(&self, start_epoch: EpochId, end_epoch: EpochId) -> Vec<ChangeEvent> {
        let guard = self.events.read();
        let mut results = Vec::new();
        for events in guard.values() {
            for event in events {
                if event.epoch >= start_epoch && event.epoch <= end_epoch {
                    results.push(event.clone());
                }
            }
        }
        results.sort_by_key(|e| e.epoch);
        results
    }

    /// Returns the total number of recorded events.
    #[must_use]
    pub fn event_count(&self) -> usize {
        self.events.read().values().map(Vec::len).sum()
    }
}

impl Default for CdcLog {
    fn default() -> Self {
        Self::new()
    }
}

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_and_history() {
        let log = CdcLog::new();
        let node_id = NodeId::new(1);

        log.record_create_node(node_id, EpochId(1), None);
        log.record_update(
            EntityId::Node(node_id),
            EpochId(2),
            "name",
            None,
            Value::from("Alix"),
        );
        log.record_update(
            EntityId::Node(node_id),
            EpochId(3),
            "name",
            Some(Value::from("Alix")),
            Value::from("Gus"),
        );

        let history = log.history(EntityId::Node(node_id));
        assert_eq!(history.len(), 3);
        assert_eq!(history[0].kind, ChangeKind::Create);
        assert_eq!(history[1].kind, ChangeKind::Update);
        assert_eq!(history[2].kind, ChangeKind::Update);
    }

    #[test]
    fn test_history_since() {
        let log = CdcLog::new();
        let node_id = NodeId::new(1);

        log.record_create_node(node_id, EpochId(1), None);
        log.record_update(
            EntityId::Node(node_id),
            EpochId(5),
            "name",
            None,
            Value::from("Alix"),
        );
        log.record_update(
            EntityId::Node(node_id),
            EpochId(10),
            "name",
            Some(Value::from("Alix")),
            Value::from("Gus"),
        );

        let since_5 = log.history_since(EntityId::Node(node_id), EpochId(5));
        assert_eq!(since_5.len(), 2);
        assert_eq!(since_5[0].epoch, EpochId(5));
    }

    #[test]
    fn test_changes_between() {
        let log = CdcLog::new();

        log.record_create_node(NodeId::new(1), EpochId(1), None);
        log.record_create_node(NodeId::new(2), EpochId(3), None);
        log.record_update(
            EntityId::Node(NodeId::new(1)),
            EpochId(5),
            "x",
            None,
            Value::from(42),
        );

        let changes = log.changes_between(EpochId(2), EpochId(5));
        assert_eq!(changes.len(), 2); // epoch 3 and 5
    }

    #[test]
    fn test_delete_event() {
        let log = CdcLog::new();
        let node_id = NodeId::new(1);

        let mut props = HashMap::new();
        props.insert("name".to_string(), Value::from("Alix"));

        log.record_create_node(node_id, EpochId(1), Some(props.clone()));
        log.record_delete(EntityId::Node(node_id), EpochId(2), Some(props));

        let history = log.history(EntityId::Node(node_id));
        assert_eq!(history.len(), 2);
        assert_eq!(history[1].kind, ChangeKind::Delete);
        assert!(history[1].after.is_none());
        assert!(history[1].before.is_some());
    }

    #[test]
    fn test_empty_history() {
        let log = CdcLog::new();
        let history = log.history(EntityId::Node(NodeId::new(999)));
        assert!(history.is_empty());
    }

    #[test]
    fn test_event_count() {
        let log = CdcLog::new();
        assert_eq!(log.event_count(), 0);

        log.record_create_node(NodeId::new(1), EpochId(1), None);
        log.record_create_node(NodeId::new(2), EpochId(2), None);
        assert_eq!(log.event_count(), 2);
    }

    #[test]
    fn test_entity_id_conversions() {
        let node_id = NodeId::new(42);
        let entity: EntityId = node_id.into();
        assert!(entity.is_node());
        assert_eq!(entity.as_u64(), 42);

        let edge_id = EdgeId::new(7);
        let entity: EntityId = edge_id.into();
        assert!(!entity.is_node());
        assert_eq!(entity.as_u64(), 7);
    }
}
