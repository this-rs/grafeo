//! Comprehensive tests for InstrumentedStore — mutation tracking, event capture,
//! drain/clear, and delegation of read operations.

use grafeo_common::types::{NodeId, PropertyKey, Value};
use grafeo_core::LpgStore;
use grafeo_core::graph::traits::{GraphStore, GraphStoreMut};
use grafeo_reactive::{InstrumentedStore, MutationEvent};

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
    let deleted = store.delete_edge(grafeo_common::types::EdgeId::new(999));
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
    use grafeo_core::graph::Direction;

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
