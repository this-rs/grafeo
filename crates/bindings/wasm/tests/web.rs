//! WASM binding integration tests.
//!
//! Run via: `wasm-pack test crates/bindings/wasm --node`

#![cfg(target_arch = "wasm32")]

use wasm_bindgen::prelude::*;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_node);

use grafeo_wasm::Database;

#[wasm_bindgen_test]
fn test_database_creation() {
    let db = Database::new().expect("should create in-memory database");
    assert_eq!(db.node_count(), 0);
    assert_eq!(db.edge_count(), 0);
}

#[wasm_bindgen_test]
fn test_insert_and_query() {
    let db = Database::new().expect("create db");

    db.execute("CREATE (:Person {name: 'Alice', age: 30})")
        .expect("create node");
    assert_eq!(db.node_count(), 1);

    let result = db
        .execute("MATCH (n:Person) RETURN n.name, n.age")
        .expect("query");

    // Result is a JsValue (array of objects)
    assert!(!result.is_null());
    assert!(!result.is_undefined());
}

#[wasm_bindgen_test]
fn test_execute_raw_structure() {
    let db = Database::new().expect("create db");

    db.execute("CREATE (:N {x: 1}), (:N {x: 2})")
        .expect("create");

    let raw = db.execute_raw("MATCH (n:N) RETURN n.x").expect("raw query");

    // execute_raw returns { columns, rows, executionTimeMs }
    assert!(!raw.is_null());
    assert!(!raw.is_undefined());
}

#[wasm_bindgen_test]
fn test_execute_with_language_gql() {
    let db = Database::new().expect("create db");

    db.execute("CREATE (:Person {name: 'Alice'})")
        .expect("create");

    let result = db
        .execute_with_language("MATCH (n:Person) RETURN n.name", "gql")
        .expect("gql query");

    assert!(!result.is_null());
}

#[wasm_bindgen_test]
fn test_execute_with_unknown_language_error() {
    let db = Database::new().expect("create db");

    let result = db.execute_with_language("SELECT 1", "unknown_lang");
    assert!(result.is_err(), "unknown language should error");
}

#[wasm_bindgen_test]
fn test_snapshot_roundtrip() {
    let db = Database::new().expect("create db");

    db.execute("CREATE (:Person {name: 'Alice'})")
        .expect("create");
    assert_eq!(db.node_count(), 1);

    // Export snapshot
    let snapshot = db.export_snapshot().expect("export");
    assert!(!snapshot.is_empty());

    // Import into new database
    let restored = Database::import_snapshot(&snapshot).expect("import");
    assert_eq!(restored.node_count(), 1);
}

#[wasm_bindgen_test]
fn test_schema() {
    let db = Database::new().expect("create db");

    db.execute("CREATE (:Person {name: 'Alice'})-[:KNOWS]->(:Person {name: 'Bob'})")
        .expect("create");

    let schema = db.schema().expect("schema");
    assert!(!schema.is_null());
    assert!(!schema.is_undefined());
}

#[wasm_bindgen_test]
fn test_version() {
    let v = grafeo_wasm::version();
    assert!(!v.is_empty());
}
