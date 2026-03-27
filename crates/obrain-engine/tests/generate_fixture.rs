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
    let alix = db.create_node(&["Person"]);
    db.set_node_property(alix, "name", Value::String("Alix".into()));
    db.set_node_property(alix, "age", Value::Int64(30));

    let gus = db.create_node(&["Person", "Employee"]);
    db.set_node_property(gus, "name", Value::String("Gus".into()));
    db.set_node_property(gus, "age", Value::Int64(25));

    let acme = db.create_node(&["Company"]);
    db.set_node_property(acme, "name", Value::String("Acme Corp".into()));

    // 2 edges with properties
    let knows = db.create_edge(alix, gus, "KNOWS");
    db.set_edge_property(knows, "since", Value::Int64(2020));

    let works = db.create_edge(gus, acme, "WORKS_AT");
    db.set_edge_property(works, "role", Value::String("Engineer".into()));

    let bytes = db.export_snapshot().unwrap();

    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/snapshot_v1.bin"
    );
    std::fs::write(path, &bytes).unwrap();
    println!("Wrote {} bytes to {path}", bytes.len());
}
