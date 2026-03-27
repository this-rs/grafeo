//! Comprehensive tests for InstrumentedStore — mutation tracking, event capture,
//! drain/clear, and delegation of read operations.

use obrain_common::types::{EdgeId, NodeId, PropertyKey, TransactionId, Value};
use obrain_core::LpgStore;
use obrain_core::graph::traits::{GraphStore, GraphStoreMut};
use obrain_reactive::{InstrumentedStore, MutationEvent};

fn new_store() -> InstrumentedStore<LpgStore> {
    let inner = LpgStore::new().expect("LpgStore::new failed");
    InstrumentedStore::new(inner)
}

// ========================================================================
// Construction and basic accessors
// ========================================================================

#[test]
fn new_store_has_no_pending_events() {
    let store = new_store();
    assert_eq!(store.pending_count(), 0);
    assert!(store.drain_pending().is_empty());
}

#[test]
fn inner_returns_reference() {
    let store = new_store();
    // The inner store should have zero nodes initially
    assert_eq!(store.inner().node_count(), 0);
}

#[test]
fn inner_mut_returns_mutable_reference() {
    let mut store = new_store();
    // We can access the inner store mutably
    assert_eq!(store.inner_mut().node_count(), 0);
}

// ========================================================================
// Node creation tracking
// ========================================================================

#[test]
fn create_node_emits_node_created_event() {
    let store = new_store();
    let id = store.create_node(&["Person"]);

    assert_eq!(store.pending_count(), 1);
    let events = store.drain_pending();
    assert_eq!(events.len(), 1);
    match &events[0] {
        MutationEvent::NodeCreated { node } => {
            assert_eq!(node.id, id);
            assert_eq!(node.labels.len(), 1);
            assert_eq!(node.labels[0].as_str(), "Person");
        }
        other => panic!("expected NodeCreated, got {:?}", other),
    }
}

#[test]
fn create_node_multiple_labels() {
    let store = new_store();
    let id = store.create_node(&["Person", "Employee", "Active"]);

    let events = store.drain_pending();
    assert_eq!(events.len(), 1);
    if let MutationEvent::NodeCreated { node } = &events[0] {
        assert_eq!(node.id, id);
        assert_eq!(node.labels.len(), 3);
    } else {
        panic!("expected NodeCreated");
    }
}

#[test]
fn create_node_no_labels() {
    let store = new_store();
    let id = store.create_node(&[]);

    let events = store.drain_pending();
    assert_eq!(events.len(), 1);
    if let MutationEvent::NodeCreated { node } = &events[0] {
        assert_eq!(node.id, id);
        assert!(node.labels.is_empty());
    } else {
        panic!("expected NodeCreated");
    }
}

#[test]
fn create_node_with_props_emits_event_with_properties() {
    let store = new_store();
    let props = vec![
        (
            PropertyKey::from("name"),
            Value::String(arcstr::literal!("Alice")),
        ),
        (PropertyKey::from("age"), Value::Int64(30)),
    ];
    let id = store.create_node_with_props(&["Person"], &props);

    let events = store.drain_pending();
    assert_eq!(events.len(), 1);
    if let MutationEvent::NodeCreated { node } = &events[0] {
        assert_eq!(node.id, id);
        assert_eq!(node.labels.len(), 1);
        assert_eq!(node.properties.len(), 2);
    } else {
        panic!("expected NodeCreated");
    }
}

#[test]
fn create_multiple_nodes_emits_multiple_events() {
    let store = new_store();
    let _id1 = store.create_node(&["A"]);
    let _id2 = store.create_node(&["B"]);
    let _id3 = store.create_node(&["C"]);

    assert_eq!(store.pending_count(), 3);
    let events = store.drain_pending();
    assert_eq!(events.len(), 3);
    assert!(events.iter().all(|e| e.kind() == "node_created"));
}

// ========================================================================
// Edge creation tracking
// ========================================================================

#[test]
fn create_edge_emits_edge_created_event() {
    let store = new_store();
    let n1 = store.create_node(&["A"]);
    let n2 = store.create_node(&["B"]);
    store.drain_pending(); // clear node events

    let eid = store.create_edge(n1, n2, "KNOWS");

    let events = store.drain_pending();
    assert_eq!(events.len(), 1);
    match &events[0] {
        MutationEvent::EdgeCreated { edge } => {
            assert_eq!(edge.id, eid);
            assert_eq!(edge.src, n1);
            assert_eq!(edge.dst, n2);
            assert_eq!(edge.edge_type.as_str(), "KNOWS");
        }
        other => panic!("expected EdgeCreated, got {:?}", other),
    }
}

#[test]
fn create_edge_with_props_emits_event() {
    let store = new_store();
    let n1 = store.create_node(&["A"]);
    let n2 = store.create_node(&["B"]);
    store.drain_pending();

    let props = vec![(PropertyKey::from("weight"), Value::Float64(0.75))];
    let eid = store.create_edge_with_props(n1, n2, "LIKES", &props);

    let events = store.drain_pending();
    assert_eq!(events.len(), 1);
    if let MutationEvent::EdgeCreated { edge } = &events[0] {
        assert_eq!(edge.id, eid);
        assert_eq!(edge.edge_type.as_str(), "LIKES");
        assert_eq!(edge.properties.len(), 1);
    } else {
        panic!("expected EdgeCreated");
    }
}

#[test]
fn batch_create_edges_emits_multiple_events() {
    let store = new_store();
    let n1 = store.create_node(&["A"]);
    let n2 = store.create_node(&["B"]);
    let n3 = store.create_node(&["C"]);
    store.drain_pending();

    let edges = vec![(n1, n2, "KNOWS"), (n2, n3, "FOLLOWS"), (n1, n3, "LIKES")];
    let eids = store.batch_create_edges(&edges);

    assert_eq!(eids.len(), 3);
    let events = store.drain_pending();
    assert_eq!(events.len(), 3);
    assert!(events.iter().all(|e| e.kind() == "edge_created"));
}

// ========================================================================
// Node deletion tracking
// ========================================================================

#[test]
fn delete_node_emits_node_deleted_event() {
    let store = new_store();
    let id = store.create_node(&["Temp"]);
    store.drain_pending();

    let deleted = store.delete_node(id);
    assert!(deleted);

    let events = store.drain_pending();
    assert_eq!(events.len(), 1);
    match &events[0] {
        MutationEvent::NodeDeleted { node } => {
            assert_eq!(node.id, id);
            assert_eq!(node.labels[0].as_str(), "Temp");
        }
        other => panic!("expected NodeDeleted, got {:?}", other),
    }
}

#[test]
fn delete_nonexistent_node_no_event() {
    let store = new_store();
    let deleted = store.delete_node(NodeId::new(999));
    assert!(!deleted);
    assert_eq!(store.pending_count(), 0);
}

#[test]
fn delete_node_edges_emits_edge_deleted_events() {
    let store = new_store();
    let n1 = store.create_node(&["A"]);
    let n2 = store.create_node(&["B"]);
    let n3 = store.create_node(&["C"]);
    let _e1 = store.create_edge(n1, n2, "KNOWS");
    let _e2 = store.create_edge(n1, n3, "LIKES");
    store.drain_pending();

    store.delete_node_edges(n1);

    let events = store.drain_pending();
    // Should have EdgeDeleted events for both edges
    assert!(
        events.len() >= 2,
        "expected at least 2 EdgeDeleted events, got {}",
        events.len()
    );
    assert!(
        events.iter().all(|e| e.kind() == "edge_deleted"),
        "all events should be edge_deleted"
    );
}

// ========================================================================
// Edge deletion tracking
// ========================================================================

#[test]
fn delete_edge_emits_edge_deleted_event() {
    let store = new_store();
    let n1 = store.create_node(&["A"]);
    let n2 = store.create_node(&["B"]);
    let eid = store.create_edge(n1, n2, "KNOWS");
    store.drain_pending();

    let deleted = store.delete_edge(eid);
    assert!(deleted);

    let events = store.drain_pending();
    assert_eq!(events.len(), 1);
    if let MutationEvent::EdgeDeleted { edge } = &events[0] {
        assert_eq!(edge.id, eid);
        assert_eq!(edge.edge_type.as_str(), "KNOWS");
    } else {
        panic!("expected EdgeDeleted");
    }
}

#[test]
fn delete_nonexistent_edge_no_event() {
    let store = new_store();
    let deleted = store.delete_edge(obrain_common::types::EdgeId::new(999));
    assert!(!deleted);
    assert_eq!(store.pending_count(), 0);
}

// ========================================================================
// Node property mutation tracking
// ========================================================================

#[test]
fn set_node_property_emits_node_updated() {
    let store = new_store();
    let id = store.create_node(&["Person"]);
    store.drain_pending();

    store.set_node_property(id, "name", Value::String(arcstr::literal!("Alice")));

    let events = store.drain_pending();
    assert_eq!(events.len(), 1);
    match &events[0] {
        MutationEvent::NodeUpdated { before, after } => {
            assert_eq!(before.id, id);
            assert_eq!(after.id, id);
            // Before should have no properties, after should have one
            assert!(before.properties.is_empty());
            assert_eq!(after.properties.len(), 1);
        }
        other => panic!("expected NodeUpdated, got {:?}", other),
    }
}

#[test]
fn set_node_property_overwrite_emits_node_updated() {
    let store = new_store();
    let id = store.create_node(&["Person"]);
    store.set_node_property(id, "name", Value::String(arcstr::literal!("Alice")));
    store.drain_pending();

    store.set_node_property(id, "name", Value::String(arcstr::literal!("Bob")));

    let events = store.drain_pending();
    assert_eq!(events.len(), 1);
    if let MutationEvent::NodeUpdated { before, after } = &events[0] {
        assert_eq!(before.properties.len(), 1);
        assert_eq!(after.properties.len(), 1);
    } else {
        panic!("expected NodeUpdated");
    }
}

#[test]
fn remove_node_property_emits_node_updated() {
    let store = new_store();
    let id = store.create_node(&["Person"]);
    store.set_node_property(id, "name", Value::String(arcstr::literal!("Alice")));
    store.drain_pending();

    let removed = store.remove_node_property(id, "name");
    assert!(removed.is_some());

    let events = store.drain_pending();
    assert_eq!(events.len(), 1);
    if let MutationEvent::NodeUpdated { before, after } = &events[0] {
        assert_eq!(before.properties.len(), 1);
        assert!(after.properties.is_empty());
    } else {
        panic!("expected NodeUpdated");
    }
}

#[test]
fn remove_nonexistent_node_property_no_event() {
    let store = new_store();
    let id = store.create_node(&["Person"]);
    store.drain_pending();

    let removed = store.remove_node_property(id, "nonexistent");
    assert!(removed.is_none());
    assert_eq!(store.pending_count(), 0);
}

// ========================================================================
// Edge property mutation tracking
// ========================================================================

#[test]
fn set_edge_property_emits_edge_updated() {
    let store = new_store();
    let n1 = store.create_node(&["A"]);
    let n2 = store.create_node(&["B"]);
    let eid = store.create_edge(n1, n2, "KNOWS");
    store.drain_pending();

    store.set_edge_property(eid, "weight", Value::Float64(0.5));

    let events = store.drain_pending();
    assert_eq!(events.len(), 1);
    match &events[0] {
        MutationEvent::EdgeUpdated { before, after } => {
            assert_eq!(before.id, eid);
            assert_eq!(after.id, eid);
            assert!(before.properties.is_empty());
            assert_eq!(after.properties.len(), 1);
        }
        other => panic!("expected EdgeUpdated, got {:?}", other),
    }
}

#[test]
fn remove_edge_property_emits_edge_updated() {
    let store = new_store();
    let n1 = store.create_node(&["A"]);
    let n2 = store.create_node(&["B"]);
    let eid = store.create_edge(n1, n2, "KNOWS");
    store.set_edge_property(eid, "weight", Value::Float64(0.5));
    store.drain_pending();

    let removed = store.remove_edge_property(eid, "weight");
    assert!(removed.is_some());

    let events = store.drain_pending();
    assert_eq!(events.len(), 1);
    if let MutationEvent::EdgeUpdated { before, after } = &events[0] {
        assert_eq!(before.properties.len(), 1);
        assert!(after.properties.is_empty());
    } else {
        panic!("expected EdgeUpdated");
    }
}

#[test]
fn remove_nonexistent_edge_property_no_event() {
    let store = new_store();
    let n1 = store.create_node(&["A"]);
    let n2 = store.create_node(&["B"]);
    let eid = store.create_edge(n1, n2, "KNOWS");
    store.drain_pending();

    let removed = store.remove_edge_property(eid, "nonexistent");
    assert!(removed.is_none());
    assert_eq!(store.pending_count(), 0);
}

// ========================================================================
// Label mutation tracking
// ========================================================================

#[test]
fn add_label_emits_node_updated() {
    let store = new_store();
    let id = store.create_node(&["Person"]);
    store.drain_pending();

    let added = store.add_label(id, "Employee");
    assert!(added);

    let events = store.drain_pending();
    assert_eq!(events.len(), 1);
    if let MutationEvent::NodeUpdated { before, after } = &events[0] {
        assert_eq!(before.labels.len(), 1);
        assert_eq!(after.labels.len(), 2);
    } else {
        panic!("expected NodeUpdated");
    }
}

#[test]
fn add_duplicate_label_no_event() {
    let store = new_store();
    let id = store.create_node(&["Person"]);
    store.drain_pending();

    // Adding a label that already exists should return false and emit no event
    let added = store.add_label(id, "Person");
    assert!(!added);
    assert_eq!(store.pending_count(), 0);
}

#[test]
fn remove_label_emits_node_updated() {
    let store = new_store();
    let id = store.create_node(&["Person", "Employee"]);
    store.drain_pending();

    let removed = store.remove_label(id, "Employee");
    assert!(removed);

    let events = store.drain_pending();
    assert_eq!(events.len(), 1);
    if let MutationEvent::NodeUpdated { before, after } = &events[0] {
        assert_eq!(before.labels.len(), 2);
        assert_eq!(after.labels.len(), 1);
    } else {
        panic!("expected NodeUpdated");
    }
}

#[test]
fn remove_nonexistent_label_no_event() {
    let store = new_store();
    let id = store.create_node(&["Person"]);
    store.drain_pending();

    let removed = store.remove_label(id, "NonExistent");
    assert!(!removed);
    assert_eq!(store.pending_count(), 0);
}

// ========================================================================
// drain_pending / clear_pending behavior
// ========================================================================

#[test]
fn drain_pending_clears_buffer() {
    let store = new_store();
    store.create_node(&["A"]);
    store.create_node(&["B"]);

    assert_eq!(store.pending_count(), 2);
    let events = store.drain_pending();
    assert_eq!(events.len(), 2);

    // After drain, should be empty
    assert_eq!(store.pending_count(), 0);
    assert!(store.drain_pending().is_empty());
}

#[test]
fn clear_pending_discards_events() {
    let store = new_store();
    store.create_node(&["A"]);
    store.create_node(&["B"]);
    store.create_node(&["C"]);

    assert_eq!(store.pending_count(), 3);
    store.clear_pending();

    assert_eq!(store.pending_count(), 0);
    assert!(store.drain_pending().is_empty());
}

#[test]
fn drain_then_clear_no_panic() {
    let store = new_store();
    store.create_node(&["A"]);
    store.drain_pending();
    store.clear_pending(); // clearing already empty buffer
    assert_eq!(store.pending_count(), 0);
}

// ========================================================================
// Read delegation (GraphStore methods pass through)
// ========================================================================

#[test]
fn get_node_delegates_to_inner() {
    let store = new_store();
    let id = store.create_node(&["Person"]);
    store.drain_pending();

    let node = store.get_node(id);
    assert!(node.is_some());
    let node = node.unwrap();
    assert_eq!(node.id, id);
}

#[test]
fn get_node_nonexistent_returns_none() {
    let store = new_store();
    assert!(store.get_node(NodeId::new(999)).is_none());
}

#[test]
fn get_edge_delegates_to_inner() {
    let store = new_store();
    let n1 = store.create_node(&["A"]);
    let n2 = store.create_node(&["B"]);
    let eid = store.create_edge(n1, n2, "KNOWS");
    store.drain_pending();

    let edge = store.get_edge(eid);
    assert!(edge.is_some());
    let edge = edge.unwrap();
    assert_eq!(edge.id, eid);
    assert_eq!(edge.src, n1);
    assert_eq!(edge.dst, n2);
}

#[test]
fn node_count_delegates_to_inner() {
    let store = new_store();
    assert_eq!(store.node_count(), 0);
    store.create_node(&["A"]);
    assert_eq!(store.node_count(), 1);
    store.create_node(&["B"]);
    assert_eq!(store.node_count(), 2);
}

#[test]
fn edge_count_delegates_to_inner() {
    let store = new_store();
    let n1 = store.create_node(&["A"]);
    let n2 = store.create_node(&["B"]);
    assert_eq!(store.edge_count(), 0);
    store.create_edge(n1, n2, "KNOWS");
    assert_eq!(store.edge_count(), 1);
}

#[test]
fn node_ids_delegates_to_inner() {
    let store = new_store();
    let id1 = store.create_node(&["A"]);
    let id2 = store.create_node(&["B"]);
    let mut ids = store.node_ids();
    ids.sort();
    let mut expected = vec![id1, id2];
    expected.sort();
    assert_eq!(ids, expected);
}

#[test]
fn nodes_by_label_delegates_to_inner() {
    let store = new_store();
    let id1 = store.create_node(&["Person"]);
    let _id2 = store.create_node(&["Document"]);
    let id3 = store.create_node(&["Person"]);
    let mut persons = store.nodes_by_label("Person");
    persons.sort();
    let mut expected = vec![id1, id3];
    expected.sort();
    assert_eq!(persons, expected);
}

#[test]
fn get_node_property_delegates_to_inner() {
    let store = new_store();
    let id = store.create_node(&["Person"]);
    store.set_node_property(id, "name", Value::String(arcstr::literal!("Alice")));
    store.drain_pending();

    let val = store.get_node_property(id, &PropertyKey::from("name"));
    assert!(val.is_some());
}

#[test]
fn get_edge_property_delegates_to_inner() {
    let store = new_store();
    let n1 = store.create_node(&["A"]);
    let n2 = store.create_node(&["B"]);
    let eid = store.create_edge(n1, n2, "KNOWS");
    store.set_edge_property(eid, "weight", Value::Float64(0.5));
    store.drain_pending();

    let val = store.get_edge_property(eid, &PropertyKey::from("weight"));
    assert!(val.is_some());
}

#[test]
fn neighbors_delegates_to_inner() {
    use obrain_core::graph::Direction;

    let store = new_store();
    let n1 = store.create_node(&["A"]);
    let n2 = store.create_node(&["B"]);
    let n3 = store.create_node(&["C"]);
    store.create_edge(n1, n2, "KNOWS");
    store.create_edge(n1, n3, "LIKES");
    store.drain_pending();

    let neighbors = store.neighbors(n1, Direction::Outgoing);
    assert_eq!(neighbors.len(), 2);
}

#[test]
fn out_degree_delegates_to_inner() {
    let store = new_store();
    let n1 = store.create_node(&["A"]);
    let n2 = store.create_node(&["B"]);
    let n3 = store.create_node(&["C"]);
    store.create_edge(n1, n2, "KNOWS");
    store.create_edge(n1, n3, "LIKES");
    store.drain_pending();

    assert_eq!(store.out_degree(n1), 2);
    assert_eq!(store.out_degree(n2), 0);
}

#[test]
fn edge_type_delegates_to_inner() {
    let store = new_store();
    let n1 = store.create_node(&["A"]);
    let n2 = store.create_node(&["B"]);
    let eid = store.create_edge(n1, n2, "FOLLOWS");
    store.drain_pending();

    let etype = store.edge_type(eid);
    assert!(etype.is_some());
    assert_eq!(etype.unwrap().as_str(), "FOLLOWS");
}

// Note: Debug impl for InstrumentedStore requires S: Debug.
// LpgStore does not implement Debug, so we skip the Debug test here.

// ========================================================================
// Combined operations — event ordering
// ========================================================================

#[test]
fn events_are_in_operation_order() {
    let store = new_store();

    let n1 = store.create_node(&["Person"]); // event 0: NodeCreated
    store.set_node_property(n1, "name", Value::String(arcstr::literal!("Alice"))); // event 1: NodeUpdated
    let n2 = store.create_node(&["Document"]); // event 2: NodeCreated
    let _eid = store.create_edge(n1, n2, "AUTHORED"); // event 3: EdgeCreated
    store.add_label(n1, "Author"); // event 4: NodeUpdated

    let events = store.drain_pending();
    assert_eq!(events.len(), 5);
    assert_eq!(events[0].kind(), "node_created");
    assert_eq!(events[1].kind(), "node_updated");
    assert_eq!(events[2].kind(), "node_created");
    assert_eq!(events[3].kind(), "edge_created");
    assert_eq!(events[4].kind(), "node_updated");
}

#[test]
fn full_lifecycle_create_update_delete() {
    let store = new_store();

    // Create
    let id = store.create_node(&["Temp"]);
    // Update property
    store.set_node_property(id, "value", Value::Int64(42));
    // Delete
    store.delete_node(id);

    let events = store.drain_pending();
    assert_eq!(events.len(), 3);
    assert_eq!(events[0].kind(), "node_created");
    assert_eq!(events[1].kind(), "node_updated");
    assert_eq!(events[2].kind(), "node_deleted");
}

#[test]
fn edge_full_lifecycle() {
    let store = new_store();
    let n1 = store.create_node(&["A"]);
    let n2 = store.create_node(&["B"]);
    store.drain_pending();

    let eid = store.create_edge(n1, n2, "KNOWS");
    store.set_edge_property(eid, "since", Value::String(arcstr::literal!("2024")));
    store.delete_edge(eid);

    let events = store.drain_pending();
    assert_eq!(events.len(), 3);
    assert_eq!(events[0].kind(), "edge_created");
    assert_eq!(events[1].kind(), "edge_updated");
    assert_eq!(events[2].kind(), "edge_deleted");
}

// ========================================================================
// Snapshot correctness — before/after states
// ========================================================================

#[test]
fn node_update_captures_before_and_after_snapshots() {
    let store = new_store();
    let id = store.create_node(&["Person"]);
    store.set_node_property(id, "name", Value::String(arcstr::literal!("Alice")));
    store.drain_pending();

    store.set_node_property(id, "age", Value::Int64(30));

    let events = store.drain_pending();
    assert_eq!(events.len(), 1);
    if let MutationEvent::NodeUpdated { before, after } = &events[0] {
        // Before should have "name" but not "age"
        assert_eq!(before.properties.len(), 1);
        // After should have both "name" and "age"
        assert_eq!(after.properties.len(), 2);
    } else {
        panic!("expected NodeUpdated");
    }
}

#[test]
fn node_deleted_captures_last_state() {
    let store = new_store();
    let id = store.create_node(&["Person"]);
    store.set_node_property(id, "name", Value::String(arcstr::literal!("Alice")));
    store.add_label(id, "Employee");
    store.drain_pending();

    store.delete_node(id);

    let events = store.drain_pending();
    assert_eq!(events.len(), 1);
    if let MutationEvent::NodeDeleted { node } = &events[0] {
        assert_eq!(node.id, id);
        assert_eq!(node.labels.len(), 2); // Person + Employee
        assert_eq!(node.properties.len(), 1); // name
    } else {
        panic!("expected NodeDeleted");
    }
}

#[test]
fn edge_deleted_captures_last_state() {
    let store = new_store();
    let n1 = store.create_node(&["A"]);
    let n2 = store.create_node(&["B"]);
    let eid = store.create_edge(n1, n2, "KNOWS");
    store.set_edge_property(eid, "weight", Value::Float64(0.8));
    store.drain_pending();

    store.delete_edge(eid);

    let events = store.drain_pending();
    assert_eq!(events.len(), 1);
    if let MutationEvent::EdgeDeleted { edge } = &events[0] {
        assert_eq!(edge.id, eid);
        assert_eq!(edge.src, n1);
        assert_eq!(edge.dst, n2);
        assert_eq!(edge.edge_type.as_str(), "KNOWS");
        assert_eq!(edge.properties.len(), 1);
    } else {
        panic!("expected EdgeDeleted");
    }
}

// ========================================================================
// Read delegation — versioned / epoch methods
// ========================================================================

#[test]
fn get_node_versioned_delegates_to_inner() {
    let store = new_store();
    let id = store.create_node(&["Person"]);
    store.drain_pending();

    let epoch = store.current_epoch();
    let txn = TransactionId::SYSTEM;
    let node = store.get_node_versioned(id, epoch, txn);
    assert!(node.is_some());
    assert_eq!(node.unwrap().id, id);

    // Non-existent node returns None
    assert!(
        store
            .get_node_versioned(NodeId::new(999), epoch, txn)
            .is_none()
    );
}

#[test]
fn get_edge_versioned_delegates_to_inner() {
    let store = new_store();
    let n1 = store.create_node(&["A"]);
    let n2 = store.create_node(&["B"]);
    let eid = store.create_edge(n1, n2, "KNOWS");
    store.drain_pending();

    let epoch = store.current_epoch();
    let txn = TransactionId::SYSTEM;
    let edge = store.get_edge_versioned(eid, epoch, txn);
    assert!(edge.is_some());
    assert_eq!(edge.unwrap().id, eid);

    assert!(
        store
            .get_edge_versioned(EdgeId::new(999), epoch, txn)
            .is_none()
    );
}

#[test]
fn get_node_at_epoch_delegates_to_inner() {
    let store = new_store();
    let id = store.create_node(&["Person"]);
    store.drain_pending();

    let epoch = store.current_epoch();
    let node = store.get_node_at_epoch(id, epoch);
    assert!(node.is_some());
    assert_eq!(node.unwrap().id, id);

    // Non-existent node
    assert!(store.get_node_at_epoch(NodeId::new(999), epoch).is_none());
}

#[test]
fn get_edge_at_epoch_delegates_to_inner() {
    let store = new_store();
    let n1 = store.create_node(&["A"]);
    let n2 = store.create_node(&["B"]);
    let eid = store.create_edge(n1, n2, "KNOWS");
    store.drain_pending();

    let epoch = store.current_epoch();
    let edge = store.get_edge_at_epoch(eid, epoch);
    assert!(edge.is_some());
    assert_eq!(edge.unwrap().id, eid);

    assert!(store.get_edge_at_epoch(EdgeId::new(999), epoch).is_none());
}

// ========================================================================
// Read delegation — batch property methods
// ========================================================================

#[test]
fn get_node_property_batch_delegates_to_inner() {
    let store = new_store();
    let n1 = store.create_node(&["A"]);
    let n2 = store.create_node(&["B"]);
    store.set_node_property(n1, "name", Value::String(arcstr::literal!("Alice")));
    store.drain_pending();

    let key = PropertyKey::from("name");
    let results = store.get_node_property_batch(&[n1, n2], &key);
    assert_eq!(results.len(), 2);
    assert!(results[0].is_some());
    assert!(results[1].is_none());
}

#[test]
fn get_nodes_properties_batch_delegates_to_inner() {
    let store = new_store();
    let n1 = store.create_node(&["A"]);
    let n2 = store.create_node(&["B"]);
    store.set_node_property(n1, "name", Value::String(arcstr::literal!("Alice")));
    store.set_node_property(n2, "age", Value::Int64(30));
    store.drain_pending();

    let results = store.get_nodes_properties_batch(&[n1, n2]);
    assert_eq!(results.len(), 2);
    assert!(results[0].contains_key(&PropertyKey::from("name")));
    assert!(results[1].contains_key(&PropertyKey::from("age")));
}

#[test]
fn get_nodes_properties_selective_batch_delegates_to_inner() {
    let store = new_store();
    let n1 = store.create_node(&["A"]);
    store.set_node_property(n1, "name", Value::String(arcstr::literal!("Alice")));
    store.set_node_property(n1, "age", Value::Int64(30));
    store.drain_pending();

    let keys = vec![PropertyKey::from("name")];
    let results = store.get_nodes_properties_selective_batch(&[n1], &keys);
    assert_eq!(results.len(), 1);
    assert!(results[0].contains_key(&PropertyKey::from("name")));
    // "age" was not requested
    assert!(!results[0].contains_key(&PropertyKey::from("age")));
}

#[test]
fn get_edges_properties_selective_batch_delegates_to_inner() {
    let store = new_store();
    let n1 = store.create_node(&["A"]);
    let n2 = store.create_node(&["B"]);
    let eid = store.create_edge(n1, n2, "KNOWS");
    store.set_edge_property(eid, "weight", Value::Float64(0.5));
    store.set_edge_property(eid, "since", Value::Int64(2024));
    store.drain_pending();

    let keys = vec![PropertyKey::from("weight")];
    let results = store.get_edges_properties_selective_batch(&[eid], &keys);
    assert_eq!(results.len(), 1);
    assert!(results[0].contains_key(&PropertyKey::from("weight")));
    assert!(!results[0].contains_key(&PropertyKey::from("since")));
}

// ========================================================================
// Read delegation — adjacency, degree, backward adjacency
// ========================================================================

#[test]
fn edges_from_delegates_to_inner() {
    use obrain_core::graph::Direction;

    let store = new_store();
    let n1 = store.create_node(&["A"]);
    let n2 = store.create_node(&["B"]);
    let eid = store.create_edge(n1, n2, "KNOWS");
    store.drain_pending();

    let edges = store.edges_from(n1, Direction::Outgoing);
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0], (n2, eid));
}

#[test]
fn in_degree_delegates_to_inner() {
    let store = new_store();
    let n1 = store.create_node(&["A"]);
    let n2 = store.create_node(&["B"]);
    store.create_edge(n1, n2, "KNOWS");
    store.drain_pending();

    assert_eq!(store.in_degree(n2), 1);
    assert_eq!(store.in_degree(n1), 0);
}

#[test]
fn has_backward_adjacency_delegates_to_inner() {
    let store = new_store();
    // LpgStore supports backward adjacency
    let result = store.has_backward_adjacency();
    // Just verify it returns a bool (delegation works)
    let _ = result;
}

#[test]
fn all_node_ids_delegates_to_inner() {
    let store = new_store();
    let id1 = store.create_node(&["A"]);
    let id2 = store.create_node(&["B"]);
    store.drain_pending();

    let mut ids = store.all_node_ids();
    ids.sort();
    let mut expected = vec![id1, id2];
    expected.sort();
    assert_eq!(ids, expected);
}

// ========================================================================
// Read delegation — edge type versioned, property index, find methods
// ========================================================================

#[test]
fn edge_type_versioned_delegates_to_inner() {
    let store = new_store();
    let n1 = store.create_node(&["A"]);
    let n2 = store.create_node(&["B"]);
    let eid = store.create_edge(n1, n2, "FOLLOWS");
    store.drain_pending();

    let epoch = store.current_epoch();
    let txn = TransactionId::SYSTEM;
    let etype = store.edge_type_versioned(eid, epoch, txn);
    assert!(etype.is_some());
    assert_eq!(etype.unwrap().as_str(), "FOLLOWS");

    assert!(
        store
            .edge_type_versioned(EdgeId::new(999), epoch, txn)
            .is_none()
    );
}

#[test]
fn has_property_index_delegates_to_inner() {
    let store = new_store();
    // LpgStore typically returns false for unindexed properties
    let result = store.has_property_index("nonexistent");
    assert!(!result);
}

#[test]
fn find_nodes_by_property_delegates_to_inner() {
    let store = new_store();
    let n1 = store.create_node(&["A"]);
    store.set_node_property(n1, "color", Value::String(arcstr::literal!("red")));
    let _n2 = store.create_node(&["B"]);
    store.drain_pending();

    let results = store.find_nodes_by_property("color", &Value::String(arcstr::literal!("red")));
    assert!(results.contains(&n1));
}

#[test]
fn find_nodes_by_properties_delegates_to_inner() {
    let store = new_store();
    let n1 = store.create_node(&["A"]);
    store.set_node_property(n1, "color", Value::String(arcstr::literal!("red")));
    store.set_node_property(n1, "size", Value::Int64(10));
    store.drain_pending();

    let conditions: Vec<(&str, Value)> = vec![
        ("color", Value::String(arcstr::literal!("red"))),
        ("size", Value::Int64(10)),
    ];
    let results = store.find_nodes_by_properties(&conditions);
    assert!(results.contains(&n1));
}

#[test]
fn find_nodes_in_range_delegates_to_inner() {
    let store = new_store();
    let n1 = store.create_node(&["A"]);
    store.set_node_property(n1, "score", Value::Int64(50));
    let n2 = store.create_node(&["B"]);
    store.set_node_property(n2, "score", Value::Int64(150));
    store.drain_pending();

    let min = Value::Int64(0);
    let max = Value::Int64(100);
    let results = store.find_nodes_in_range("score", Some(&min), Some(&max), true, true);
    // n1 should be in range, n2 should not
    assert!(results.contains(&n1));
    assert!(!results.contains(&n2));
}

// ========================================================================
// Read delegation — bloom filter / might-match methods
// ========================================================================

#[test]
fn node_property_might_match_delegates_to_inner() {
    use obrain_core::graph::lpg::CompareOp;

    let store = new_store();
    let _n1 = store.create_node(&["A"]);
    store.drain_pending();

    let key = PropertyKey::from("name");
    let val = Value::String(arcstr::literal!("test"));
    // Just verifying it delegates without panic; result depends on implementation
    let _result = store.node_property_might_match(&key, CompareOp::Eq, &val);
}

#[test]
fn edge_property_might_match_delegates_to_inner() {
    use obrain_core::graph::lpg::CompareOp;

    let store = new_store();
    let key = PropertyKey::from("weight");
    let val = Value::Float64(1.0);
    let _result = store.edge_property_might_match(&key, CompareOp::Eq, &val);
}

// ========================================================================
// Read delegation — statistics, cardinality, degree estimates
// ========================================================================

#[test]
fn statistics_delegates_to_inner() {
    let store = new_store();
    store.create_node(&["Person"]);
    store.drain_pending();

    let stats = store.statistics();
    // statistics() returns Arc<Statistics>, just verify it does not panic
    let _ = stats;
}

#[test]
fn estimate_label_cardinality_delegates_to_inner() {
    let store = new_store();
    store.create_node(&["Person"]);
    store.drain_pending();

    let cardinality = store.estimate_label_cardinality("Person");
    // Should be >= 0
    assert!(cardinality >= 0.0);
}

#[test]
fn estimate_avg_degree_delegates_to_inner() {
    let store = new_store();
    let n1 = store.create_node(&["A"]);
    let n2 = store.create_node(&["B"]);
    store.create_edge(n1, n2, "KNOWS");
    store.drain_pending();

    let avg_out = store.estimate_avg_degree("KNOWS", true);
    let avg_in = store.estimate_avg_degree("KNOWS", false);
    assert!(avg_out >= 0.0);
    assert!(avg_in >= 0.0);
}

#[test]
fn current_epoch_delegates_to_inner() {
    let store = new_store();
    let epoch = store.current_epoch();
    // Should return a valid epoch (>= INITIAL)
    // Should return a valid epoch
    let _ = epoch;
}

// ========================================================================
// Read delegation — schema introspection
// ========================================================================

#[test]
fn all_labels_delegates_to_inner() {
    let store = new_store();
    store.create_node(&["Person"]);
    store.create_node(&["Document"]);
    store.drain_pending();

    let labels = store.all_labels();
    assert!(labels.contains(&"Person".to_string()));
    assert!(labels.contains(&"Document".to_string()));
}

#[test]
fn all_edge_types_delegates_to_inner() {
    let store = new_store();
    let n1 = store.create_node(&["A"]);
    let n2 = store.create_node(&["B"]);
    store.create_edge(n1, n2, "KNOWS");
    store.create_edge(n1, n2, "LIKES");
    store.drain_pending();

    let types = store.all_edge_types();
    assert!(types.contains(&"KNOWS".to_string()));
    assert!(types.contains(&"LIKES".to_string()));
}

#[test]
fn all_property_keys_delegates_to_inner() {
    let store = new_store();
    let n1 = store.create_node(&["A"]);
    store.set_node_property(n1, "name", Value::String(arcstr::literal!("Alice")));
    store.drain_pending();

    let keys = store.all_property_keys();
    assert!(keys.contains(&"name".to_string()));
}

// ========================================================================
// Read delegation — visibility methods
// ========================================================================

#[test]
fn is_node_visible_at_epoch_delegates_to_inner() {
    let store = new_store();
    let id = store.create_node(&["A"]);
    store.drain_pending();

    let epoch = store.current_epoch();
    assert!(store.is_node_visible_at_epoch(id, epoch));
    assert!(!store.is_node_visible_at_epoch(NodeId::new(999), epoch));
}

#[test]
fn is_node_visible_versioned_delegates_to_inner() {
    let store = new_store();
    let id = store.create_node(&["A"]);
    store.drain_pending();

    let epoch = store.current_epoch();
    let txn = TransactionId::SYSTEM;
    assert!(store.is_node_visible_versioned(id, epoch, txn));
    assert!(!store.is_node_visible_versioned(NodeId::new(999), epoch, txn));
}

#[test]
fn is_edge_visible_at_epoch_delegates_to_inner() {
    let store = new_store();
    let n1 = store.create_node(&["A"]);
    let n2 = store.create_node(&["B"]);
    let eid = store.create_edge(n1, n2, "KNOWS");
    store.drain_pending();

    let epoch = store.current_epoch();
    assert!(store.is_edge_visible_at_epoch(eid, epoch));
    assert!(!store.is_edge_visible_at_epoch(EdgeId::new(999), epoch));
}

#[test]
fn is_edge_visible_versioned_delegates_to_inner() {
    let store = new_store();
    let n1 = store.create_node(&["A"]);
    let n2 = store.create_node(&["B"]);
    let eid = store.create_edge(n1, n2, "KNOWS");
    store.drain_pending();

    let epoch = store.current_epoch();
    let txn = TransactionId::SYSTEM;
    assert!(store.is_edge_visible_versioned(eid, epoch, txn));
    assert!(!store.is_edge_visible_versioned(EdgeId::new(999), epoch, txn));
}

#[test]
fn filter_visible_node_ids_delegates_to_inner() {
    let store = new_store();
    let n1 = store.create_node(&["A"]);
    let n2 = store.create_node(&["B"]);
    store.drain_pending();

    let epoch = store.current_epoch();
    let visible = store.filter_visible_node_ids(&[n1, n2, NodeId::new(999)], epoch);
    assert!(visible.contains(&n1));
    assert!(visible.contains(&n2));
    assert!(!visible.contains(&NodeId::new(999)));
}

#[test]
fn filter_visible_node_ids_versioned_delegates_to_inner() {
    let store = new_store();
    let n1 = store.create_node(&["A"]);
    let n2 = store.create_node(&["B"]);
    store.drain_pending();

    let epoch = store.current_epoch();
    let txn = TransactionId::SYSTEM;
    let visible = store.filter_visible_node_ids_versioned(&[n1, n2, NodeId::new(999)], epoch, txn);
    assert!(visible.contains(&n1));
    assert!(visible.contains(&n2));
    assert!(!visible.contains(&NodeId::new(999)));
}

// ========================================================================
// Read delegation — history methods
// ========================================================================

#[test]
fn get_node_history_delegates_to_inner() {
    let store = new_store();
    let id = store.create_node(&["A"]);
    store.drain_pending();

    let history = store.get_node_history(id);
    // At least one version should exist
    assert!(!history.is_empty());
    assert_eq!(history[0].2.id, id);
}

#[test]
fn get_edge_history_delegates_to_inner() {
    let store = new_store();
    let n1 = store.create_node(&["A"]);
    let n2 = store.create_node(&["B"]);
    let eid = store.create_edge(n1, n2, "KNOWS");
    store.drain_pending();

    let history = store.get_edge_history(eid);
    assert!(!history.is_empty());
    assert_eq!(history[0].2.id, eid);
}

// ========================================================================
// GraphStoreMut — versioned write delegation
// ========================================================================

#[test]
fn create_node_versioned_delegates_and_emits_event() {
    let store = new_store();
    let epoch = store.current_epoch();
    let txn = TransactionId::SYSTEM;

    let id = store.create_node_versioned(&["Person"], epoch, txn);
    assert!(store.get_node(id).is_some());

    let events = store.drain_pending();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].kind(), "node_created");
}

#[test]
fn create_edge_versioned_delegates_and_emits_event() {
    let store = new_store();
    let n1 = store.create_node(&["A"]);
    let n2 = store.create_node(&["B"]);
    store.drain_pending();

    let epoch = store.current_epoch();
    let txn = TransactionId::SYSTEM;

    let eid = store.create_edge_versioned(n1, n2, "KNOWS", epoch, txn);
    assert!(store.get_edge(eid).is_some());

    let events = store.drain_pending();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].kind(), "edge_created");
}

#[test]
fn delete_node_versioned_delegates_and_emits_event() {
    let store = new_store();
    let id = store.create_node(&["Temp"]);
    store.drain_pending();

    let epoch = store.current_epoch();
    let txn = TransactionId::SYSTEM;

    let deleted = store.delete_node_versioned(id, epoch, txn);
    assert!(deleted);

    let events = store.drain_pending();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].kind(), "node_deleted");
}

#[test]
fn delete_node_versioned_nonexistent_no_event() {
    let store = new_store();
    let epoch = store.current_epoch();
    let txn = TransactionId::SYSTEM;

    let deleted = store.delete_node_versioned(NodeId::new(999), epoch, txn);
    assert!(!deleted);
    assert_eq!(store.pending_count(), 0);
}

#[test]
fn delete_edge_versioned_delegates_and_emits_event() {
    let store = new_store();
    let n1 = store.create_node(&["A"]);
    let n2 = store.create_node(&["B"]);
    let eid = store.create_edge(n1, n2, "KNOWS");
    store.drain_pending();

    let epoch = store.current_epoch();
    let txn = TransactionId::SYSTEM;

    let deleted = store.delete_edge_versioned(eid, epoch, txn);
    assert!(deleted);

    let events = store.drain_pending();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].kind(), "edge_deleted");
}

#[test]
fn delete_edge_versioned_nonexistent_no_event() {
    let store = new_store();
    let epoch = store.current_epoch();
    let txn = TransactionId::SYSTEM;

    let deleted = store.delete_edge_versioned(EdgeId::new(999), epoch, txn);
    assert!(!deleted);
    assert_eq!(store.pending_count(), 0);
}

#[test]
fn set_node_property_versioned_delegates_to_inner() {
    let store = new_store();
    let id = store.create_node(&["Person"]);
    store.drain_pending();

    let txn = TransactionId::SYSTEM;
    store.set_node_property_versioned(id, "name", Value::String(arcstr::literal!("Alice")), txn);

    // Verify the property was set
    let val = store.get_node_property(id, &PropertyKey::from("name"));
    assert!(val.is_some());
}

#[test]
fn set_edge_property_versioned_delegates_to_inner() {
    let store = new_store();
    let n1 = store.create_node(&["A"]);
    let n2 = store.create_node(&["B"]);
    let eid = store.create_edge(n1, n2, "KNOWS");
    store.drain_pending();

    let txn = TransactionId::SYSTEM;
    store.set_edge_property_versioned(eid, "weight", Value::Float64(0.8), txn);

    let val = store.get_edge_property(eid, &PropertyKey::from("weight"));
    assert!(val.is_some());
}

#[test]
fn remove_node_property_versioned_delegates_to_inner() {
    let store = new_store();
    let id = store.create_node(&["Person"]);
    store.set_node_property(id, "name", Value::String(arcstr::literal!("Alice")));
    store.drain_pending();

    let txn = TransactionId::SYSTEM;
    let removed = store.remove_node_property_versioned(id, "name", txn);
    assert!(removed.is_some());

    // Property should be gone
    let val = store.get_node_property(id, &PropertyKey::from("name"));
    assert!(val.is_none());
}

#[test]
fn remove_node_property_versioned_nonexistent_returns_none() {
    let store = new_store();
    let id = store.create_node(&["Person"]);
    store.drain_pending();

    let txn = TransactionId::SYSTEM;
    let removed = store.remove_node_property_versioned(id, "nonexistent", txn);
    assert!(removed.is_none());
}

#[test]
fn remove_edge_property_versioned_delegates_to_inner() {
    let store = new_store();
    let n1 = store.create_node(&["A"]);
    let n2 = store.create_node(&["B"]);
    let eid = store.create_edge(n1, n2, "KNOWS");
    store.set_edge_property(eid, "weight", Value::Float64(0.5));
    store.drain_pending();

    let txn = TransactionId::SYSTEM;
    let removed = store.remove_edge_property_versioned(eid, "weight", txn);
    assert!(removed.is_some());

    let val = store.get_edge_property(eid, &PropertyKey::from("weight"));
    assert!(val.is_none());
}

#[test]
fn remove_edge_property_versioned_nonexistent_returns_none() {
    let store = new_store();
    let n1 = store.create_node(&["A"]);
    let n2 = store.create_node(&["B"]);
    let eid = store.create_edge(n1, n2, "KNOWS");
    store.drain_pending();

    let txn = TransactionId::SYSTEM;
    let removed = store.remove_edge_property_versioned(eid, "nonexistent", txn);
    assert!(removed.is_none());
}

// Under the `temporal` feature, versioned writes are deferred (EpochId::PENDING)
// and not visible via get_node() until the transaction commits. These tests verify
// immediate visibility which only applies to the non-temporal code path.
#[cfg(not(feature = "temporal"))]
#[test]
fn add_label_versioned_delegates_to_inner() {
    let store = new_store();
    let id = store.create_node(&["Person"]);
    store.drain_pending();

    let txn = TransactionId::SYSTEM;
    let added = store.add_label_versioned(id, "Employee", txn);
    assert!(added);

    // Verify label was added
    let node = store.get_node(id).unwrap();
    assert!(node.labels.iter().any(|l| l.as_str() == "Employee"));
}

#[test]
fn add_label_versioned_duplicate_returns_false() {
    let store = new_store();
    let id = store.create_node(&["Person"]);
    store.drain_pending();

    let txn = TransactionId::SYSTEM;
    let added = store.add_label_versioned(id, "Person", txn);
    assert!(!added);
}

#[cfg(not(feature = "temporal"))]
#[test]
fn remove_label_versioned_delegates_to_inner() {
    let store = new_store();
    let id = store.create_node(&["Person", "Employee"]);
    store.drain_pending();

    let txn = TransactionId::SYSTEM;
    let removed = store.remove_label_versioned(id, "Employee", txn);
    assert!(removed);

    let node = store.get_node(id).unwrap();
    assert!(!node.labels.iter().any(|l| l.as_str() == "Employee"));
}

#[test]
fn remove_label_versioned_nonexistent_returns_false() {
    let store = new_store();
    let id = store.create_node(&["Person"]);
    store.drain_pending();

    let txn = TransactionId::SYSTEM;
    let removed = store.remove_label_versioned(id, "NonExistent", txn);
    assert!(!removed);
}

// Note: Debug impl for InstrumentedStore<S> requires S: Debug.
// LpgStore does not implement Debug, so we cannot test the Debug impl
// with the standard test store. The Debug impl is a trivial delegation
// (lines 654-659 of store.rs).

// ========================================================================
// Versioned writes emit MutationEvent (regression tests for silent writes)
// ========================================================================

// Under the `temporal` feature, versioned writes are deferred (EpochId::PENDING)
// so get_node()/get_edge() returns the pre-mutation snapshot → before == after → no event.
#[cfg(not(feature = "temporal"))]
#[test]
fn set_node_property_versioned_emits_node_updated() {
    let store = new_store();
    let id = store.create_node(&["Person"]);
    store.drain_pending();

    let txn = TransactionId::SYSTEM;
    store.set_node_property_versioned(id, "name", Value::String(arcstr::literal!("Alice")), txn);

    let events = store.drain_pending();
    assert_eq!(
        events.len(),
        1,
        "set_node_property_versioned must emit a MutationEvent"
    );
    match &events[0] {
        MutationEvent::NodeUpdated { before, after } => {
            assert_eq!(before.id, id);
            assert_eq!(after.id, id);
            assert!(before.properties.is_empty());
            assert_eq!(after.properties.len(), 1);
        }
        other => panic!("expected NodeUpdated, got {:?}", other),
    }
}

#[cfg(not(feature = "temporal"))]
#[test]
fn set_edge_property_versioned_emits_edge_updated() {
    let store = new_store();
    let n1 = store.create_node(&["A"]);
    let n2 = store.create_node(&["B"]);
    let eid = store.create_edge(n1, n2, "KNOWS");
    store.drain_pending();

    let txn = TransactionId::SYSTEM;
    store.set_edge_property_versioned(eid, "weight", Value::Float64(0.5), txn);

    let events = store.drain_pending();
    assert_eq!(
        events.len(),
        1,
        "set_edge_property_versioned must emit a MutationEvent"
    );
    match &events[0] {
        MutationEvent::EdgeUpdated { before, after } => {
            assert_eq!(before.id, eid);
            assert_eq!(after.id, eid);
            assert!(before.properties.is_empty());
            assert_eq!(after.properties.len(), 1);
        }
        other => panic!("expected EdgeUpdated, got {:?}", other),
    }
}

#[cfg(not(feature = "temporal"))]
#[test]
fn add_label_versioned_emits_node_updated() {
    let store = new_store();
    let id = store.create_node(&["Person"]);
    store.drain_pending();

    let txn = TransactionId::SYSTEM;
    let added = store.add_label_versioned(id, "Employee", txn);
    assert!(added);

    let events = store.drain_pending();
    assert_eq!(
        events.len(),
        1,
        "add_label_versioned must emit a MutationEvent"
    );
    match &events[0] {
        MutationEvent::NodeUpdated { before, after } => {
            assert_eq!(before.labels.len(), 1);
            assert_eq!(after.labels.len(), 2);
        }
        other => panic!("expected NodeUpdated, got {:?}", other),
    }
}

#[test]
fn add_label_versioned_duplicate_emits_no_event() {
    let store = new_store();
    let id = store.create_node(&["Person"]);
    store.drain_pending();

    let txn = TransactionId::SYSTEM;
    let added = store.add_label_versioned(id, "Person", txn);
    assert!(!added);
    assert_eq!(
        store.pending_count(),
        0,
        "duplicate add_label_versioned must not emit event"
    );
}

#[cfg(not(feature = "temporal"))]
#[test]
fn remove_label_versioned_emits_node_updated() {
    let store = new_store();
    let id = store.create_node(&["Person", "Employee"]);
    store.drain_pending();

    let txn = TransactionId::SYSTEM;
    let removed = store.remove_label_versioned(id, "Employee", txn);
    assert!(removed);

    let events = store.drain_pending();
    assert_eq!(
        events.len(),
        1,
        "remove_label_versioned must emit a MutationEvent"
    );
    match &events[0] {
        MutationEvent::NodeUpdated { before, after } => {
            assert_eq!(before.labels.len(), 2);
            assert_eq!(after.labels.len(), 1);
        }
        other => panic!("expected NodeUpdated, got {:?}", other),
    }
}

#[test]
fn remove_label_versioned_nonexistent_emits_no_event() {
    let store = new_store();
    let id = store.create_node(&["Person"]);
    store.drain_pending();

    let txn = TransactionId::SYSTEM;
    let removed = store.remove_label_versioned(id, "Ghost", txn);
    assert!(!removed);
    assert_eq!(
        store.pending_count(),
        0,
        "nonexistent remove_label_versioned must not emit event"
    );
}

#[test]
fn remove_node_property_versioned_emits_node_updated() {
    let store = new_store();
    let id = store.create_node(&["Person"]);
    store.set_node_property(id, "name", Value::String(arcstr::literal!("Alice")));
    store.drain_pending();

    let txn = TransactionId::SYSTEM;
    let removed = store.remove_node_property_versioned(id, "name", txn);
    assert!(removed.is_some());

    let events = store.drain_pending();
    assert_eq!(
        events.len(),
        1,
        "remove_node_property_versioned must emit a MutationEvent"
    );
    match &events[0] {
        MutationEvent::NodeUpdated { before, after } => {
            assert_eq!(before.properties.len(), 1);
            assert!(after.properties.is_empty());
        }
        other => panic!("expected NodeUpdated, got {:?}", other),
    }
}

#[test]
fn remove_node_property_versioned_nonexistent_emits_no_event() {
    let store = new_store();
    let id = store.create_node(&["Person"]);
    store.drain_pending();

    let txn = TransactionId::SYSTEM;
    let removed = store.remove_node_property_versioned(id, "ghost", txn);
    assert!(removed.is_none());
    assert_eq!(store.pending_count(), 0);
}

#[test]
fn remove_edge_property_versioned_emits_edge_updated() {
    let store = new_store();
    let n1 = store.create_node(&["A"]);
    let n2 = store.create_node(&["B"]);
    let eid = store.create_edge(n1, n2, "KNOWS");
    store.set_edge_property(eid, "weight", Value::Float64(0.5));
    store.drain_pending();

    let txn = TransactionId::SYSTEM;
    let removed = store.remove_edge_property_versioned(eid, "weight", txn);
    assert!(removed.is_some());

    let events = store.drain_pending();
    assert_eq!(
        events.len(),
        1,
        "remove_edge_property_versioned must emit a MutationEvent"
    );
    match &events[0] {
        MutationEvent::EdgeUpdated { before, after } => {
            assert_eq!(before.properties.len(), 1);
            assert!(after.properties.is_empty());
        }
        other => panic!("expected EdgeUpdated, got {:?}", other),
    }
}

#[test]
fn remove_edge_property_versioned_nonexistent_emits_no_event() {
    let store = new_store();
    let n1 = store.create_node(&["A"]);
    let n2 = store.create_node(&["B"]);
    let eid = store.create_edge(n1, n2, "KNOWS");
    store.drain_pending();

    let txn = TransactionId::SYSTEM;
    let removed = store.remove_edge_property_versioned(eid, "ghost", txn);
    assert!(removed.is_none());
    assert_eq!(store.pending_count(), 0);
}

// ========================================================================
// Concurrency tests — DashMap-backed store with tokio::spawn × N tasks
// ========================================================================

#[tokio::test]
async fn concurrent_node_creation_no_data_loss() {
    let store = std::sync::Arc::new(new_store());
    let n = 100;
    let mut handles = Vec::with_capacity(n);

    for _ in 0..n {
        let s = store.clone();
        handles.push(tokio::spawn(async move {
            s.create_node(&["Concurrent"]);
        }));
    }

    for h in handles {
        h.await.unwrap();
    }

    assert_eq!(store.node_count(), n);
    let events = store.drain_pending();
    assert_eq!(
        events.len(),
        n,
        "every concurrent create_node must emit exactly one event"
    );
}

#[tokio::test]
async fn concurrent_edge_creation_no_data_loss() {
    let store = std::sync::Arc::new(new_store());
    // Pre-create nodes
    let n1 = store.create_node(&["A"]);
    let n2 = store.create_node(&["B"]);
    store.drain_pending();

    let n = 50;
    let mut handles = Vec::with_capacity(n);
    for i in 0..n {
        let s = store.clone();
        handles.push(tokio::spawn(async move {
            s.create_edge(n1, n2, &format!("REL_{}", i));
        }));
    }

    for h in handles {
        h.await.unwrap();
    }

    assert_eq!(store.edge_count(), n);
    let events = store.drain_pending();
    assert_eq!(
        events.len(),
        n,
        "every concurrent create_edge must emit exactly one event"
    );
}

#[tokio::test]
async fn concurrent_property_writes_no_panic() {
    let store = std::sync::Arc::new(new_store());
    let id = store.create_node(&["Target"]);
    store.drain_pending();

    let n = 100;
    let mut handles = Vec::with_capacity(n);
    for i in 0..n {
        let s = store.clone();
        handles.push(tokio::spawn(async move {
            s.set_node_property(id, "counter", Value::Int64(i as i64));
        }));
    }

    for h in handles {
        h.await.unwrap();
    }

    // All writes should have produced events (no lost events)
    let events = store.drain_pending();
    assert_eq!(
        events.len(),
        n,
        "every concurrent set_node_property must emit an event"
    );
    assert!(events.iter().all(|e| e.kind() == "node_updated"));
}

#[tokio::test]
async fn concurrent_label_add_remove_no_panic() {
    let store = std::sync::Arc::new(new_store());
    let id = store.create_node(&["Base"]);
    store.drain_pending();

    let n = 50;
    let mut handles = Vec::with_capacity(n * 2);
    for i in 0..n {
        let s = store.clone();
        let label = format!("Label_{}", i);
        handles.push(tokio::spawn(async move {
            s.add_label(id, &label);
        }));
    }

    for h in handles {
        h.await.unwrap();
    }

    // All unique labels should have been added
    let node = store.get_node(id).unwrap();
    // Base + n unique labels
    assert_eq!(node.labels.len(), n + 1);
    let events = store.drain_pending();
    assert_eq!(
        events.len(),
        n,
        "every successful add_label must emit an event"
    );
}

#[tokio::test]
async fn concurrent_mixed_operations_no_panic() {
    use std::sync::Arc;

    let store = Arc::new(new_store());
    let n = 40;
    let mut handles = Vec::new();

    // Spawn node creators
    for _ in 0..n {
        let s = store.clone();
        handles.push(tokio::spawn(async move {
            let id = s.create_node(&["Mixed"]);
            s.set_node_property(id, "key", Value::Int64(42));
            id
        }));
    }

    let mut node_ids = Vec::new();
    for h in handles {
        node_ids.push(h.await.unwrap());
    }

    // Now spawn edge creators between random pairs
    let mut edge_handles = Vec::new();
    for i in 0..n {
        let s = store.clone();
        let src = node_ids[i];
        let dst = node_ids[(i + 1) % n];
        edge_handles.push(tokio::spawn(async move {
            s.create_edge(src, dst, "LINKED");
        }));
    }

    for h in edge_handles {
        h.await.unwrap();
    }

    assert_eq!(store.node_count(), n);
    assert_eq!(store.edge_count(), n);

    let events = store.drain_pending();
    // n create_node + n set_property + n create_edge = 3n events
    assert_eq!(
        events.len(),
        3 * n,
        "all concurrent operations must emit events"
    );
}

#[tokio::test]
async fn concurrent_versioned_property_writes_emit_events() {
    let store = std::sync::Arc::new(new_store());
    let id = store.create_node(&["Versioned"]);
    store.drain_pending();

    let n = 50;
    let mut handles = Vec::with_capacity(n);
    for i in 0..n {
        let s = store.clone();
        handles.push(tokio::spawn(async move {
            let txn = TransactionId(i as u64 + 1);
            s.set_node_property_versioned(id, "val", Value::Int64(i as i64), txn);
        }));
    }

    for h in handles {
        h.await.unwrap();
    }

    let events = store.drain_pending();
    assert_eq!(
        events.len(),
        n,
        "every concurrent versioned write must emit an event"
    );
    assert!(events.iter().all(|e| e.kind() == "node_updated"));
}
