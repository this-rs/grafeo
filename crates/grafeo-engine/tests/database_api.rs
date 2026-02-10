//! Integration tests for GrafeoDB public API methods that lacked coverage.
//!
//! Covers: edge operations, property indexes, label management,
//! node/edge iteration, validation, info/stats, to_memory, and
//! remove_property operations.

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

// ── Edge operations ──────────────────────────────────────────────

#[test]
fn test_create_edge_with_props() {
    let db = GrafeoDB::new_in_memory();
    let alice = db.create_node(&["Person"]);
    let bob = db.create_node(&["Person"]);

    let eid = db.create_edge_with_props(
        alice,
        bob,
        "KNOWS",
        [
            ("since", Value::Int64(2020)),
            ("weight", Value::Float64(0.9)),
        ],
    );

    let edge = db.get_edge(eid).expect("edge should exist");
    assert_eq!(edge.edge_type.as_str(), "KNOWS");
    assert_eq!(edge.src, alice);
    assert_eq!(edge.dst, bob);
}

#[test]
fn test_get_edge_returns_none_for_invalid_id() {
    let db = GrafeoDB::new_in_memory();
    assert!(
        db.get_edge(grafeo_common::types::EdgeId::new(999))
            .is_none()
    );
}

#[test]
fn test_set_and_remove_edge_property() {
    let db = GrafeoDB::new_in_memory();
    let a = db.create_node(&["N"]);
    let b = db.create_node(&["N"]);
    let eid = db.create_edge(a, b, "R");

    db.set_edge_property(eid, "weight", Value::Float64(1.5));
    let edge = db.get_edge(eid).unwrap();
    assert_eq!(
        edge.properties
            .get(&grafeo_common::types::PropertyKey::new("weight")),
        Some(&Value::Float64(1.5))
    );

    // Remove
    assert!(db.remove_edge_property(eid, "weight"));
    let edge = db.get_edge(eid).unwrap();
    assert!(
        !edge
            .properties
            .contains_key(&grafeo_common::types::PropertyKey::new("weight"))
    );

    // Removing again returns false
    assert!(!db.remove_edge_property(eid, "weight"));
}

#[test]
fn test_delete_edge() {
    let db = GrafeoDB::new_in_memory();
    let a = db.create_node(&["N"]);
    let b = db.create_node(&["N"]);
    let eid = db.create_edge(a, b, "R");

    assert_eq!(db.edge_count(), 1);
    assert!(db.delete_edge(eid));
    assert_eq!(db.edge_count(), 0);
    assert!(!db.delete_edge(eid)); // second delete returns false
}

// ── Label management ─────────────────────────────────────────────

#[test]
fn test_add_and_remove_node_label() {
    let db = GrafeoDB::new_in_memory();
    let n = db.create_node(&["Person"]);

    // Add label
    assert!(db.add_node_label(n, "Employee"));
    let labels = db.get_node_labels(n).unwrap();
    assert!(labels.contains(&"Person".to_string()));
    assert!(labels.contains(&"Employee".to_string()));

    // Adding same label again returns false
    assert!(!db.add_node_label(n, "Employee"));

    // Remove label
    assert!(db.remove_node_label(n, "Employee"));
    let labels = db.get_node_labels(n).unwrap();
    assert!(!labels.contains(&"Employee".to_string()));
    assert!(labels.contains(&"Person".to_string()));

    // Removing again returns false
    assert!(!db.remove_node_label(n, "Employee"));
}

#[test]
fn test_get_node_labels_returns_none_for_invalid_id() {
    let db = GrafeoDB::new_in_memory();
    assert!(
        db.get_node_labels(grafeo_common::types::NodeId::new(999))
            .is_none()
    );
}

// ── Node property removal ────────────────────────────────────────

#[test]
fn test_remove_node_property() {
    let db = GrafeoDB::new_in_memory();
    let n = db.create_node(&["Person"]);
    db.set_node_property(n, "name", Value::String("Alice".into()));

    assert!(db.remove_node_property(n, "name"));
    let node = db.get_node(n).unwrap();
    assert!(
        !node
            .properties
            .contains_key(&grafeo_common::types::PropertyKey::new("name"))
    );

    // Removing again returns false
    assert!(!db.remove_node_property(n, "name"));
}

// ── Property indexes ─────────────────────────────────────────────

#[test]
fn test_property_index_lifecycle() {
    let db = GrafeoDB::new_in_memory();

    // No index initially
    assert!(!db.has_property_index("name"));

    // Create index
    db.create_property_index("name");
    assert!(db.has_property_index("name"));

    // Create nodes with the property
    let n1 = db.create_node(&["Person"]);
    db.set_node_property(n1, "name", Value::String("Alice".into()));
    let n2 = db.create_node(&["Person"]);
    db.set_node_property(n2, "name", Value::String("Bob".into()));
    let n3 = db.create_node(&["Person"]);
    db.set_node_property(n3, "name", Value::String("Alice".into()));

    // Find by property
    let results = db.find_nodes_by_property("name", &Value::String("Alice".into()));
    assert_eq!(results.len(), 2);
    assert!(results.contains(&n1));
    assert!(results.contains(&n3));

    let results = db.find_nodes_by_property("name", &Value::String("Bob".into()));
    assert_eq!(results.len(), 1);
    assert!(results.contains(&n2));

    // No matches
    let results = db.find_nodes_by_property("name", &Value::String("Carol".into()));
    assert!(results.is_empty());

    // Drop index
    assert!(db.drop_property_index("name"));
    assert!(!db.has_property_index("name"));
    assert!(!db.drop_property_index("name")); // second drop returns false
}

// ── Iteration ────────────────────────────────────────────────────

#[test]
fn test_iter_nodes() {
    let db = GrafeoDB::new_in_memory();
    let _n1 = db.create_node(&["Person"]);
    let _n2 = db.create_node(&["Company"]);
    let _n3 = db.create_node(&["Person"]);

    let nodes: Vec<_> = db.iter_nodes().collect();
    assert_eq!(nodes.len(), 3);
}

#[test]
fn test_iter_edges() {
    let db = GrafeoDB::new_in_memory();
    let a = db.create_node(&["N"]);
    let b = db.create_node(&["N"]);
    let c = db.create_node(&["N"]);

    let _e1 = db.create_edge(a, b, "R");
    let _e2 = db.create_edge(b, c, "S");

    let edges: Vec<_> = db.iter_edges().collect();
    assert_eq!(edges.len(), 2);
}

#[test]
fn test_iter_empty_database() {
    let db = GrafeoDB::new_in_memory();
    assert_eq!(db.iter_nodes().count(), 0);
    assert_eq!(db.iter_edges().count(), 0);
}

// ── Validation ───────────────────────────────────────────────────

#[test]
fn test_validate_healthy_database() {
    let db = GrafeoDB::new_in_memory();
    let a = db.create_node(&["Person"]);
    let b = db.create_node(&["Person"]);
    db.create_edge(a, b, "KNOWS");

    let result = db.validate();
    assert!(result.is_valid(), "healthy database should validate");
}

#[test]
fn test_validate_empty_database() {
    let db = GrafeoDB::new_in_memory();
    let result = db.validate();
    assert!(result.is_valid());
}

// ── Info and stats ───────────────────────────────────────────────

#[test]
fn test_info() {
    let db = GrafeoDB::new_in_memory();
    let _n = db.create_node(&["Person"]);
    db.set_node_property(_n, "name", Value::String("Alice".into()));

    let info = db.info();
    assert_eq!(info.node_count, 1);
    assert_eq!(info.edge_count, 0);
    assert!(!info.is_persistent);
    assert!(info.path.is_none());
}

#[test]
fn test_detailed_stats() {
    let db = GrafeoDB::new_in_memory();
    let a = db.create_node(&["Person"]);
    db.set_node_property(a, "name", Value::String("Alice".into()));
    let b = db.create_node(&["Company"]);
    db.set_node_property(b, "name", Value::String("Acme".into()));
    db.create_edge(a, b, "WORKS_AT");

    let stats = db.detailed_stats();
    assert_eq!(stats.node_count, 2);
    assert_eq!(stats.edge_count, 1);
    assert_eq!(stats.label_count, 2); // Person, Company
    assert_eq!(stats.edge_type_count, 1); // WORKS_AT
    assert!(stats.property_key_count >= 1); // at least "name"
    // memory_bytes may be 0 for in-memory database without buffer manager allocation
    assert!(stats.disk_bytes.is_none()); // in-memory
}

#[test]
fn test_graph_model_default() {
    let db = GrafeoDB::new_in_memory();
    let model = db.graph_model();
    // GraphModel::Lpg displays as "LPG"
    assert_eq!(
        format!("{model}"),
        "LPG",
        "default graph model should be LPG"
    );
}

// ── to_memory (clone) ────────────────────────────────────────────

#[test]
fn test_to_memory_clones_data() {
    let db = GrafeoDB::new_in_memory();
    let a = db.create_node(&["Person"]);
    db.set_node_property(a, "name", Value::String("Alice".into()));
    let b = db.create_node(&["Person"]);
    db.set_node_property(b, "name", Value::String("Bob".into()));
    db.create_edge(a, b, "KNOWS");

    let clone = db.to_memory().expect("to_memory should succeed");
    assert_eq!(clone.node_count(), 2);
    assert_eq!(clone.edge_count(), 1);

    // Original unaffected by clone modifications
    clone.create_node(&["Extra"]);
    assert_eq!(db.node_count(), 2); // original unchanged
    assert_eq!(clone.node_count(), 3);
}

// ── Multiple edge types and complex graph ────────────────────────

#[test]
fn test_complex_graph_with_multiple_edge_types() {
    let db = GrafeoDB::new_in_memory();
    let alice = db.create_node(&["Person"]);
    db.set_node_property(alice, "name", Value::String("Alice".into()));

    let bob = db.create_node(&["Person"]);
    db.set_node_property(bob, "name", Value::String("Bob".into()));

    let acme = db.create_node(&["Company"]);
    db.set_node_property(acme, "name", Value::String("Acme".into()));

    db.create_edge(alice, bob, "KNOWS");
    db.create_edge(alice, acme, "WORKS_AT");
    db.create_edge(bob, acme, "WORKS_AT");

    assert_eq!(db.node_count(), 3);
    assert_eq!(db.edge_count(), 3);

    // Validate integrity
    let result = db.validate();
    assert!(result.is_valid());

    // Stats reflect structure
    let stats = db.detailed_stats();
    assert_eq!(stats.label_count, 2);
    assert_eq!(stats.edge_type_count, 2);
}
