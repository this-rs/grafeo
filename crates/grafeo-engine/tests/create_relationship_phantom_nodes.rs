//! Regression tests for issue #181: phantom node creation on MATCH...CREATE
//! relationship patterns.
//!
//! When a Cypher query uses MATCH to bind existing nodes and then CREATE to
//! add a relationship between them, the engine must NOT create new (phantom)
//! empty nodes for the already-bound variables.
//!
//! Run with:
//! ```bash
//! cargo test -p grafeo-engine --features full --test create_relationship_phantom_nodes
//! ```

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Returns the total node count via `MATCH (n) RETURN count(n)`.
fn node_count(db: &GrafeoDB) -> i64 {
    let session = db.session();
    let result = session
        .execute("MATCH (n) RETURN count(n)")
        .expect("node count query failed");
    match &result.rows[0][0] {
        Value::Int64(n) => *n,
        other => panic!("expected Int64, got {:?}", other),
    }
}

/// Returns the total relationship count via `MATCH ()-[r]->() RETURN count(r)`.
fn edge_count(db: &GrafeoDB) -> i64 {
    let session = db.session();
    let result = session
        .execute("MATCH ()-[r]->() RETURN count(r)")
        .expect("edge count query failed");
    match &result.rows[0][0] {
        Value::Int64(n) => *n,
        other => panic!("expected Int64, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// 1. Basic case — two nodes, one relationship, no phantoms
// ---------------------------------------------------------------------------

#[test]
fn test_match_match_create_rel_no_phantom_nodes() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session
        .execute("CREATE (:A {id: '1'}), (:A {id: '2'})")
        .unwrap();
    assert_eq!(node_count(&db), 2);

    session
        .execute(
            "MATCH (a:A {id: '1'}) MATCH (b:A {id: '2'}) CREATE (a)-[:REL]->(b)",
        )
        .unwrap();

    assert_eq!(node_count(&db), 2, "phantom nodes were created");
    assert_eq!(edge_count(&db), 1, "relationship was not created");
}

// ---------------------------------------------------------------------------
// 2. Three nodes, two relationships
// ---------------------------------------------------------------------------

#[test]
fn test_match_match_create_rel_multiple() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session
        .execute("CREATE (:A {id: '1'}), (:A {id: '2'}), (:A {id: '3'})")
        .unwrap();

    session
        .execute(
            "MATCH (a:A {id: '1'}) MATCH (b:A {id: '2'}) CREATE (a)-[:REL]->(b)",
        )
        .unwrap();
    session
        .execute(
            "MATCH (a:A {id: '2'}) MATCH (b:A {id: '3'}) CREATE (a)-[:REL]->(b)",
        )
        .unwrap();

    assert_eq!(node_count(&db), 3, "phantom nodes were created");
    assert_eq!(edge_count(&db), 2, "wrong number of relationships");
}

// ---------------------------------------------------------------------------
// 3. Scale — 100 nodes, 99 chained relationships
// ---------------------------------------------------------------------------

#[test]
fn test_match_match_create_rel_scale_100() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    for i in 0..100 {
        session
            .execute(&format!("CREATE (:N {{id: '{}'}})", i))
            .unwrap();
    }
    assert_eq!(node_count(&db), 100);

    for i in 0..99 {
        session
            .execute(&format!(
                "MATCH (a:N {{id: '{}'}}) MATCH (b:N {{id: '{}'}}) CREATE (a)-[:NEXT]->(b)",
                i,
                i + 1
            ))
            .unwrap();
    }

    assert_eq!(node_count(&db), 100, "phantom nodes were created at scale");
    assert_eq!(edge_count(&db), 99, "wrong number of relationships at scale");
}

// ---------------------------------------------------------------------------
// 4. Inline CREATE — must still create nodes
// ---------------------------------------------------------------------------

#[test]
fn test_create_inline_rel_still_works() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session.execute("CREATE (a:X)-[:REL]->(b:Y)").unwrap();

    assert_eq!(node_count(&db), 2, "inline CREATE should produce 2 nodes");
    assert_eq!(edge_count(&db), 1, "inline CREATE should produce 1 edge");
}

// ---------------------------------------------------------------------------
// 5. Inline CREATE chain — 3 nodes, 2 edges
// ---------------------------------------------------------------------------

#[test]
fn test_create_inline_chain_still_works() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session
        .execute("CREATE (a:X)-[:R1]->(b:Y)-[:R2]->(c:Z)")
        .unwrap();

    assert_eq!(node_count(&db), 3, "inline chain should produce 3 nodes");
    assert_eq!(edge_count(&db), 2, "inline chain should produce 2 edges");
}

// ---------------------------------------------------------------------------
// 6. MATCH...CREATE rel with properties on the edge
// ---------------------------------------------------------------------------

#[test]
fn test_match_create_rel_with_properties() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session
        .execute("CREATE (:P {id: 'x'}), (:P {id: 'y'})")
        .unwrap();

    session
        .execute(
            "MATCH (a:P {id: 'x'}) MATCH (b:P {id: 'y'}) \
             CREATE (a)-[:WEIGHTED {weight: 42}]->(b)",
        )
        .unwrap();

    assert_eq!(node_count(&db), 2, "phantom nodes from rel with props");
    assert_eq!(edge_count(&db), 1);

    // Verify the property was stored on the edge.
    let result = session
        .execute("MATCH ()-[r:WEIGHTED]->() RETURN r.weight")
        .unwrap();
    match &result.rows[0][0] {
        Value::Int64(w) => assert_eq!(*w, 42, "edge property not preserved"),
        other => panic!("expected Int64 for weight, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// 7. Edge count correctness (not doubled)
// ---------------------------------------------------------------------------

#[test]
fn test_match_create_rel_edge_count_correct() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session
        .execute("CREATE (:T {id: '1'}), (:T {id: '2'})")
        .unwrap();

    session
        .execute(
            "MATCH (a:T {id: '1'}) MATCH (b:T {id: '2'}) CREATE (a)-[:E]->(b)",
        )
        .unwrap();

    // There should be exactly 1 relationship, not 2.
    let result = session
        .execute("MATCH (a:T {id: '1'})-[r:E]->(b:T {id: '2'}) RETURN count(r)")
        .unwrap();
    match &result.rows[0][0] {
        Value::Int64(count) => assert_eq!(*count, 1, "edge count should be exactly 1"),
        other => panic!("expected Int64, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// 8. MERGE still works (no regression)
// ---------------------------------------------------------------------------

#[test]
fn test_merge_still_no_phantoms() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session
        .execute("CREATE (:M {id: '1'}), (:M {id: '2'})")
        .unwrap();

    session
        .execute(
            "MATCH (a:M {id: '1'}) MATCH (b:M {id: '2'}) MERGE (a)-[:LINK]->(b)",
        )
        .unwrap();

    assert_eq!(node_count(&db), 2, "MERGE should not create phantom nodes");
    assert_eq!(edge_count(&db), 1, "MERGE should create 1 edge");
}

// ---------------------------------------------------------------------------
// 9. Mixed — one node from MATCH, one inline
// ---------------------------------------------------------------------------

#[test]
fn test_create_rel_mixed_inline_and_reference() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session.execute("CREATE (:A {id: 'existing'})").unwrap();
    assert_eq!(node_count(&db), 1);

    session
        .execute(
            "MATCH (a:A {id: 'existing'}) CREATE (a)-[:REL]->(b:B {id: 'new'})",
        )
        .unwrap();

    assert_eq!(
        node_count(&db),
        2,
        "should create only 1 new node (b), not phantom for a"
    );
    assert_eq!(edge_count(&db), 1);
}

// ---------------------------------------------------------------------------
// 10. Bidirectional relationships
// ---------------------------------------------------------------------------

#[test]
fn test_match_create_rel_bidirectional() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session
        .execute("CREATE (:D {id: '1'}), (:D {id: '2'})")
        .unwrap();

    session
        .execute(
            "MATCH (a:D {id: '1'}) MATCH (b:D {id: '2'}) CREATE (a)-[:FWD]->(b)",
        )
        .unwrap();
    session
        .execute(
            "MATCH (a:D {id: '1'}) MATCH (b:D {id: '2'}) CREATE (b)-[:BWD]->(a)",
        )
        .unwrap();

    assert_eq!(node_count(&db), 2, "bidirectional should not create phantoms");
    assert_eq!(edge_count(&db), 2, "should have 2 directed edges");
}

// ---------------------------------------------------------------------------
// 11. Self-loop
// ---------------------------------------------------------------------------

#[test]
fn test_match_create_rel_self_loop() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session.execute("CREATE (:S {id: 'loop'})").unwrap();

    session
        .execute("MATCH (a:S {id: 'loop'}) CREATE (a)-[:SELF]->(a)")
        .unwrap();

    assert_eq!(node_count(&db), 1, "self-loop must not create phantom nodes");
    assert_eq!(edge_count(&db), 1, "self-loop should produce 1 edge");
}

// ---------------------------------------------------------------------------
// 12. Multiple CREATE clauses between same pair
// ---------------------------------------------------------------------------

#[test]
fn test_match_create_multiple_rels_same_pair() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session
        .execute("CREATE (:Q {id: '1'}), (:Q {id: '2'})")
        .unwrap();

    session
        .execute(
            "MATCH (a:Q {id: '1'}) MATCH (b:Q {id: '2'}) \
             CREATE (a)-[:R1]->(b) \
             CREATE (a)-[:R2]->(b) \
             CREATE (a)-[:R3]->(b)",
        )
        .unwrap();

    assert_eq!(
        node_count(&db),
        2,
        "multiple CREATEs must not create phantoms"
    );
    assert_eq!(edge_count(&db), 3, "should have 3 edges");
}

// ---------------------------------------------------------------------------
// 13. Labels preserved after CREATE rel
// ---------------------------------------------------------------------------

#[test]
fn test_labels_preserved_after_create_rel() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session
        .execute("CREATE (:Foo {id: '1'}), (:Bar {id: '2'})")
        .unwrap();

    session
        .execute(
            "MATCH (a:Foo {id: '1'}) MATCH (b:Bar {id: '2'}) CREATE (a)-[:LINK]->(b)",
        )
        .unwrap();

    let result = session
        .execute("MATCH (n:Foo) RETURN count(n)")
        .unwrap();
    match &result.rows[0][0] {
        Value::Int64(n) => assert_eq!(*n, 1, "Foo label should still have 1 node"),
        other => panic!("expected Int64, got {:?}", other),
    }

    let result = session
        .execute("MATCH (n:Bar) RETURN count(n)")
        .unwrap();
    match &result.rows[0][0] {
        Value::Int64(n) => assert_eq!(*n, 1, "Bar label should still have 1 node"),
        other => panic!("expected Int64, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// 14. Properties preserved after CREATE rel
// ---------------------------------------------------------------------------

#[test]
fn test_properties_preserved_after_create_rel() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session
        .execute("CREATE (:W {name: 'alpha', val: 10}), (:W {name: 'beta', val: 20})")
        .unwrap();

    session
        .execute(
            "MATCH (a:W {name: 'alpha'}) MATCH (b:W {name: 'beta'}) CREATE (a)-[:CONN]->(b)",
        )
        .unwrap();

    let result = session
        .execute("MATCH (n:W) RETURN n.name, n.val ORDER BY n.val")
        .unwrap();
    assert_eq!(result.row_count(), 2);

    // First row: alpha, 10
    match &result.rows[0][0] {
        Value::String(s) => assert_eq!(s.as_str(), "alpha"),
        other => panic!("expected String, got {:?}", other),
    }
    match &result.rows[0][1] {
        Value::Int64(n) => assert_eq!(*n, 10),
        other => panic!("expected Int64, got {:?}", other),
    }

    // Second row: beta, 20
    match &result.rows[1][0] {
        Value::String(s) => assert_eq!(s.as_str(), "beta"),
        other => panic!("expected String, got {:?}", other),
    }
    match &result.rows[1][1] {
        Value::Int64(n) => assert_eq!(*n, 20),
        other => panic!("expected Int64, got {:?}", other),
    }
}
