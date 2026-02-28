//! One-shot helper to generate the v1 snapshot fixture.
//!
//! Run with: cargo test --all-features -p grafeo-engine --test generate_fixture -- --ignored
//! Then commit the generated file. This test is ignored by default.

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

#[test]
#[ignore = "one-shot fixture generator, not a regular test"]
fn generate_v1_snapshot_fixture() {
    let db = GrafeoDB::new_in_memory();

    // 3 nodes with different labels and properties
    let alice = db.create_node(&["Person"]);
    db.set_node_property(alice, "name", Value::String("Alice".into()));
    db.set_node_property(alice, "age", Value::Int64(30));

    let bob = db.create_node(&["Person", "Employee"]);
    db.set_node_property(bob, "name", Value::String("Bob".into()));
    db.set_node_property(bob, "age", Value::Int64(25));

    let acme = db.create_node(&["Company"]);
    db.set_node_property(acme, "name", Value::String("Acme Corp".into()));

    // 2 edges with properties
    let knows = db.create_edge(alice, bob, "KNOWS");
    db.set_edge_property(knows, "since", Value::Int64(2020));

    let works = db.create_edge(bob, acme, "WORKS_AT");
    db.set_edge_property(works, "role", Value::String("Engineer".into()));

    let bytes = db.export_snapshot().unwrap();

    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/snapshot_v1.bin"
    );
    std::fs::write(path, &bytes).unwrap();
    println!("Wrote {} bytes to {path}", bytes.len());
}
