//! Integration tests for snapshot export/import.

use grafeo_common::types::{EdgeId, NodeId, Value};
use grafeo_engine::GrafeoDB;

/// Mirror of the private Snapshot struct for crafting test payloads.
#[derive(serde::Serialize, serde::Deserialize)]
struct TestSnapshot {
    version: u8,
    nodes: Vec<TestNode>,
    edges: Vec<TestEdge>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct TestNode {
    id: NodeId,
    labels: Vec<String>,
    properties: Vec<(String, Value)>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct TestEdge {
    id: EdgeId,
    src: NodeId,
    dst: NodeId,
    edge_type: String,
    properties: Vec<(String, Value)>,
}

fn encode_snapshot(snap: &TestSnapshot) -> Vec<u8> {
    bincode::serde::encode_to_vec(snap, bincode::config::standard()).unwrap()
}

#[test]
fn export_import_empty_database() {
    let db = GrafeoDB::new_in_memory();
    let bytes = db.export_snapshot().unwrap();
    let restored = GrafeoDB::import_snapshot(&bytes).unwrap();
    assert_eq!(restored.node_count(), 0);
    assert_eq!(restored.edge_count(), 0);
}

#[test]
fn export_import_preserves_nodes() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("INSERT (:Person {name: 'Alix', age: 30})")
        .unwrap();
    session
        .execute("INSERT (:Person {name: 'Gus', age: 25})")
        .unwrap();

    let bytes = db.export_snapshot().unwrap();
    let restored = GrafeoDB::import_snapshot(&bytes).unwrap();

    assert_eq!(restored.node_count(), 2);

    let session2 = restored.session();
    let result = session2
        .execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
        .unwrap();
    assert_eq!(result.rows.len(), 2);
}

#[test]
fn export_import_preserves_edges() {
    let db = GrafeoDB::new_in_memory();
    let alix = db.create_node(&["Person"]);
    db.set_node_property(alix, "name", "Alix".into());
    let gus = db.create_node(&["Person"]);
    db.set_node_property(gus, "name", "Gus".into());
    db.create_edge(alix, gus, "KNOWS");

    let bytes = db.export_snapshot().unwrap();
    let restored = GrafeoDB::import_snapshot(&bytes).unwrap();

    assert_eq!(restored.node_count(), 2);
    assert_eq!(restored.edge_count(), 1);

    let session2 = restored.session();
    let result = session2
        .execute("MATCH (a)-[:KNOWS]->(b) RETURN a.name, b.name")
        .unwrap();
    assert_eq!(result.rows.len(), 1);
}

#[test]
fn export_import_preserves_properties() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("INSERT (:Item {name: 'Widget', price: 9.99, active: true})")
        .unwrap();

    let bytes = db.export_snapshot().unwrap();
    let restored = GrafeoDB::import_snapshot(&bytes).unwrap();

    let session2 = restored.session();
    let result = session2
        .execute("MATCH (i:Item) RETURN i.name, i.price, i.active")
        .unwrap();
    assert_eq!(result.rows.len(), 1);
}

#[test]
fn import_rejects_invalid_data() {
    let result = GrafeoDB::import_snapshot(b"not a valid snapshot");
    assert!(result.is_err());
}

#[test]
fn snapshot_round_trip_schema() {
    let db = GrafeoDB::new_in_memory();
    let alix = db.create_node(&["Person"]);
    db.set_node_property(alix, "name", "Alix".into());
    let gus = db.create_node(&["Person"]);
    db.set_node_property(gus, "name", "Gus".into());
    db.create_edge(alix, gus, "KNOWS");

    let schema_before = db.schema();
    let bytes = db.export_snapshot().unwrap();
    let restored = GrafeoDB::import_snapshot(&bytes).unwrap();
    let schema_after = restored.schema();

    // Both schemas should report the same label/edge info
    let fmt_before = format!("{schema_before:?}");
    let fmt_after = format!("{schema_after:?}");
    assert_eq!(fmt_before, fmt_after);
}

// --- Edge property round-trip ---

#[test]
fn export_import_preserves_edge_properties() {
    let db = GrafeoDB::new_in_memory();
    let a = db.create_node(&["Person"]);
    let b = db.create_node(&["Person"]);
    let edge = db.create_edge(a, b, "KNOWS");
    db.set_edge_property(edge, "since", Value::Int64(2020));
    db.set_edge_property(edge, "strength", Value::Float64(0.95));

    let bytes = db.export_snapshot().unwrap();
    let restored = GrafeoDB::import_snapshot(&bytes).unwrap();

    let session = restored.session();
    let result = session
        .execute("MATCH ()-[e:KNOWS]->() RETURN e.since, e.strength")
        .unwrap();
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][0], Value::Int64(2020));
    assert_eq!(result.rows[0][1], Value::Float64(0.95));
}

// --- Multi-label nodes ---

#[test]
fn export_import_preserves_multiple_labels() {
    let db = GrafeoDB::new_in_memory();
    db.create_node(&["Person", "Employee"]);
    db.create_node(&["Person", "Manager"]);
    db.create_node(&["Animal"]);

    let bytes = db.export_snapshot().unwrap();
    let restored = GrafeoDB::import_snapshot(&bytes).unwrap();

    assert_eq!(restored.node_count(), 3);

    let session = restored.session();
    let persons = session.execute("MATCH (p:Person) RETURN p").unwrap();
    assert_eq!(persons.rows.len(), 2);

    let employees = session.execute("MATCH (e:Employee) RETURN e").unwrap();
    assert_eq!(employees.rows.len(), 1);

    let managers = session.execute("MATCH (m:Manager) RETURN m").unwrap();
    assert_eq!(managers.rows.len(), 1);

    let animals = session.execute("MATCH (a:Animal) RETURN a").unwrap();
    assert_eq!(animals.rows.len(), 1);
}

// --- All scalar Value types ---

#[test]
fn export_import_preserves_all_value_types() {
    let db = GrafeoDB::new_in_memory();
    let id = db.create_node(&["Test"]);
    db.set_node_property(id, "str_val", Value::String("hello".into()));
    db.set_node_property(id, "int_val", Value::Int64(42));
    db.set_node_property(id, "float_val", Value::Float64(9.81));
    db.set_node_property(id, "bool_val", Value::Bool(true));
    db.set_node_property(id, "null_val", Value::Null);
    db.set_node_property(
        id,
        "bytes_val",
        Value::Bytes(vec![0xDE, 0xAD, 0xBE, 0xEF].into()),
    );

    let bytes = db.export_snapshot().unwrap();
    let restored = GrafeoDB::import_snapshot(&bytes).unwrap();

    let session = restored.session();
    let result = session
        .execute("MATCH (t:Test) RETURN t.str_val, t.int_val, t.float_val, t.bool_val, t.null_val, t.bytes_val")
        .unwrap();
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][0], Value::String("hello".into()));
    assert_eq!(result.rows[0][1], Value::Int64(42));
    assert_eq!(result.rows[0][2], Value::Float64(9.81));
    assert_eq!(result.rows[0][3], Value::Bool(true));
    assert_eq!(result.rows[0][4], Value::Null);
    assert_eq!(
        result.rows[0][5],
        Value::Bytes(vec![0xDE, 0xAD, 0xBE, 0xEF].into())
    );
}

// --- List and Map values ---

#[test]
fn export_import_preserves_collection_values() {
    let db = GrafeoDB::new_in_memory();
    let id = db.create_node(&["Test"]);
    db.set_node_property(
        id,
        "tags",
        Value::List(vec![Value::String("a".into()), Value::String("b".into())].into()),
    );

    let bytes = db.export_snapshot().unwrap();
    let restored = GrafeoDB::import_snapshot(&bytes).unwrap();

    let session = restored.session();
    let result = session.execute("MATCH (t:Test) RETURN t.tags").unwrap();
    assert_eq!(result.rows.len(), 1);
    assert_eq!(
        result.rows[0][0],
        Value::List(vec![Value::String("a".into()), Value::String("b".into()),].into())
    );
}

// --- Multiple edge types ---

#[test]
fn export_import_preserves_multiple_edge_types() {
    let db = GrafeoDB::new_in_memory();
    let a = db.create_node(&["Person"]);
    let b = db.create_node(&["Person"]);
    let c = db.create_node(&["Company"]);

    db.create_edge(a, b, "KNOWS");
    db.create_edge(a, c, "WORKS_AT");
    db.create_edge(b, c, "WORKS_AT");

    let bytes = db.export_snapshot().unwrap();
    let restored = GrafeoDB::import_snapshot(&bytes).unwrap();

    assert_eq!(restored.node_count(), 3);
    assert_eq!(restored.edge_count(), 3);

    let session = restored.session();
    let knows = session.execute("MATCH ()-[e:KNOWS]->() RETURN e").unwrap();
    assert_eq!(knows.rows.len(), 1);

    let works = session
        .execute("MATCH ()-[e:WORKS_AT]->() RETURN e")
        .unwrap();
    assert_eq!(works.rows.len(), 2);
}

// --- Nodes with no properties ---

#[test]
fn export_import_preserves_empty_property_nodes() {
    let db = GrafeoDB::new_in_memory();
    db.create_node(&["Empty"]);
    db.create_node(&["Empty"]);

    let bytes = db.export_snapshot().unwrap();
    let restored = GrafeoDB::import_snapshot(&bytes).unwrap();

    assert_eq!(restored.node_count(), 2);

    let session = restored.session();
    let result = session.execute("MATCH (e:Empty) RETURN e").unwrap();
    assert_eq!(result.rows.len(), 2);
}

// --- Moderate dataset ---

#[test]
fn export_import_moderate_dataset() {
    let db = GrafeoDB::new_in_memory();

    // Create 100 nodes with properties
    let mut ids = Vec::new();
    for i in 0..100 {
        let id = db.create_node(&["Item"]);
        db.set_node_property(id, "index", Value::Int64(i));
        db.set_node_property(id, "name", Value::String(format!("item_{i}").into()));
        ids.push(id);
    }

    // Create 50 edges
    for i in 0..50 {
        db.create_edge(ids[i], ids[i + 50], "LINKS_TO");
    }

    let bytes = db.export_snapshot().unwrap();
    assert!(!bytes.is_empty());

    let restored = GrafeoDB::import_snapshot(&bytes).unwrap();

    assert_eq!(restored.node_count(), 100);
    assert_eq!(restored.edge_count(), 50);

    // Verify a sample property survived
    let session = restored.session();
    let result = session
        .execute("MATCH (i:Item) WHERE i.index = 42 RETURN i.name")
        .unwrap();
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][0], Value::String("item_42".into()));
}

// --- Import empty bytes ---

#[test]
fn import_rejects_empty_bytes() {
    let result = GrafeoDB::import_snapshot(&[]);
    assert!(result.is_err());
}

// --- Snapshot version mismatch ---

#[test]
fn import_rejects_unsupported_version() {
    // Export a valid snapshot, then tamper with the version byte to trigger the
    // "unsupported snapshot version" error path.
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

    let mut bytes = db.export_snapshot().unwrap();
    // The first byte in bincode standard encoding for a struct starting with
    // a u8 field is the version byte itself.
    bytes[0] = 99; // Set to invalid version

    let result = GrafeoDB::import_snapshot(&bytes);
    match result {
        Ok(_) => panic!("Expected error for tampered snapshot"),
        Err(e) => {
            let err_msg = e.to_string();
            assert!(
                err_msg.contains("snapshot")
                    || err_msg.contains("unsupported")
                    || err_msg.contains("import"),
                "Expected snapshot error, got: {err_msg}"
            );
        }
    }
}

// --- Double export produces identical bytes ---

#[test]
fn double_export_is_deterministic() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

    let bytes1 = db.export_snapshot().unwrap();
    let bytes2 = db.export_snapshot().unwrap();
    assert_eq!(bytes1, bytes2);
}

// --- Edge reference validation ---

#[test]
fn import_rejects_dangling_edge_source() {
    // Create a valid snapshot, then re-export after deleting the source node
    // to create a snapshot with a dangling edge reference.
    let db = GrafeoDB::new_in_memory();
    let a = db.create_node(&["Person"]);
    let b = db.create_node(&["Person"]);
    db.create_edge(a, b, "KNOWS");

    // Export valid snapshot
    let bytes = db.export_snapshot().unwrap();
    let restored = GrafeoDB::import_snapshot(&bytes).unwrap();
    assert_eq!(restored.edge_count(), 1);

    // Now create a corrupted snapshot by exporting nodes only from a db
    // that has 1 node, then manually crafting a snapshot with an edge
    // that references a non-existent source node.
    // The simplest way: delete a node, re-export (edge still exists in store)
    // Actually, let's tamper with binary data at a higher level:
    // Build a snapshot where edge references node ID 999 which doesn't exist.
    let db2 = GrafeoDB::new_in_memory();
    let n = db2.create_node(&["Person"]);
    // Create an edge referencing a non-existent source node
    // We need to use the direct store API since create_edge validates at a higher level
    db2.store()
        .create_edge_with_id(
            grafeo_common::types::EdgeId::new(0),
            grafeo_common::types::NodeId::new(999), // doesn't exist
            n,
            "KNOWS",
        )
        .unwrap();

    let bytes = db2.export_snapshot().unwrap();
    let result = GrafeoDB::import_snapshot(&bytes);
    match result {
        Ok(_) => panic!("Expected error for dangling source node"),
        Err(e) => {
            let err = e.to_string();
            assert!(
                err.contains("non-existent source node"),
                "Expected dangling source error, got: {err}"
            );
        }
    }
}

#[test]
fn import_rejects_dangling_edge_destination() {
    let db = GrafeoDB::new_in_memory();
    let n = db.create_node(&["Person"]);
    db.store()
        .create_edge_with_id(
            grafeo_common::types::EdgeId::new(0),
            n,
            grafeo_common::types::NodeId::new(999), // doesn't exist
            "KNOWS",
        )
        .unwrap();

    let bytes = db.export_snapshot().unwrap();
    let result = GrafeoDB::import_snapshot(&bytes);
    match result {
        Ok(_) => panic!("Expected error for dangling destination node"),
        Err(e) => {
            let err = e.to_string();
            assert!(
                err.contains("non-existent destination node"),
                "Expected dangling destination error, got: {err}"
            );
        }
    }
}

#[test]
fn import_rejects_duplicate_node_ids() {
    let snap = TestSnapshot {
        version: 1,
        nodes: vec![
            TestNode {
                id: NodeId::new(0),
                labels: vec!["A".into()],
                properties: vec![],
            },
            TestNode {
                id: NodeId::new(0), // duplicate
                labels: vec!["B".into()],
                properties: vec![],
            },
        ],
        edges: vec![],
    };
    let bytes = encode_snapshot(&snap);
    let result = GrafeoDB::import_snapshot(&bytes);
    match result {
        Ok(_) => panic!("Expected error for duplicate node ID"),
        Err(e) => {
            let err = e.to_string();
            assert!(
                err.contains("duplicate node ID"),
                "Expected duplicate node error, got: {err}"
            );
        }
    }
}

#[test]
fn import_rejects_duplicate_edge_ids() {
    let snap = TestSnapshot {
        version: 1,
        nodes: vec![
            TestNode {
                id: NodeId::new(0),
                labels: vec![],
                properties: vec![],
            },
            TestNode {
                id: NodeId::new(1),
                labels: vec![],
                properties: vec![],
            },
        ],
        edges: vec![
            TestEdge {
                id: EdgeId::new(0),
                src: NodeId::new(0),
                dst: NodeId::new(1),
                edge_type: "REL".into(),
                properties: vec![],
            },
            TestEdge {
                id: EdgeId::new(0), // duplicate
                src: NodeId::new(0),
                dst: NodeId::new(1),
                edge_type: "REL".into(),
                properties: vec![],
            },
        ],
    };
    let bytes = encode_snapshot(&snap);
    let result = GrafeoDB::import_snapshot(&bytes);
    match result {
        Ok(_) => panic!("Expected error for duplicate edge ID"),
        Err(e) => {
            let err = e.to_string();
            assert!(
                err.contains("duplicate edge ID"),
                "Expected duplicate edge error, got: {err}"
            );
        }
    }
}
