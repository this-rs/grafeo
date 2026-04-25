#![cfg(feature = "cypher")]
//! Regression tests for issue #181: phantom node creation on MATCH...CREATE
//! relationship patterns.
//!
//! When a Cypher query uses MATCH to bind existing nodes and then CREATE to
//! add a relationship between them, the engine must NOT create new (phantom)
//! empty nodes for the already-bound variables.
//!
//! Run with:
//! ```bash
//! cargo test -p obrain-engine --features full --test create_relationship_phantom_nodes
//! ```

use obrain_common::types::Value;
use obrain_engine::ObrainDB;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Returns the total node count via `MATCH (n) RETURN count(n)`.
fn node_count(db: &ObrainDB) -> i64 {
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
fn edge_count(db: &ObrainDB) -> i64 {
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
    let db = ObrainDB::new_in_memory();
    let session = db.session();

    session
        .execute("CREATE (:A {id: '1'}), (:A {id: '2'})")
        .unwrap();
    assert_eq!(node_count(&db), 2);

    session
        .execute("MATCH (a:A {id: '1'}) MATCH (b:A {id: '2'}) CREATE (a)-[:REL]->(b)")
        .unwrap();

    assert_eq!(node_count(&db), 2, "phantom nodes were created");
    assert_eq!(edge_count(&db), 1, "relationship was not created");
}

// ---------------------------------------------------------------------------
// 2. Three nodes, two relationships
// ---------------------------------------------------------------------------

#[test]
fn test_match_match_create_rel_multiple() {
    let db = ObrainDB::new_in_memory();
    let session = db.session();

    session
        .execute("CREATE (:A {id: '1'}), (:A {id: '2'}), (:A {id: '3'})")
        .unwrap();

    session
        .execute("MATCH (a:A {id: '1'}) MATCH (b:A {id: '2'}) CREATE (a)-[:REL]->(b)")
        .unwrap();
    session
        .execute("MATCH (a:A {id: '2'}) MATCH (b:A {id: '3'}) CREATE (a)-[:REL]->(b)")
        .unwrap();

    assert_eq!(node_count(&db), 3, "phantom nodes were created");
    assert_eq!(edge_count(&db), 2, "wrong number of relationships");
}

// ---------------------------------------------------------------------------
// 3. Scale — 100 nodes, 99 chained relationships
// ---------------------------------------------------------------------------

#[test]
fn test_match_match_create_rel_scale_100() {
    let db = ObrainDB::new_in_memory();
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
    assert_eq!(
        edge_count(&db),
        99,
        "wrong number of relationships at scale"
    );
}

// ---------------------------------------------------------------------------
// 4. Inline CREATE — must still create nodes
// ---------------------------------------------------------------------------

#[test]
fn test_create_inline_rel_still_works() {
    let db = ObrainDB::new_in_memory();
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
    let db = ObrainDB::new_in_memory();
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
    let db = ObrainDB::new_in_memory();
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
    let db = ObrainDB::new_in_memory();
    let session = db.session();

    session
        .execute("CREATE (:T {id: '1'}), (:T {id: '2'})")
        .unwrap();

    session
        .execute("MATCH (a:T {id: '1'}) MATCH (b:T {id: '2'}) CREATE (a)-[:E]->(b)")
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
    let db = ObrainDB::new_in_memory();
    let session = db.session();

    session
        .execute("CREATE (:M {id: '1'}), (:M {id: '2'})")
        .unwrap();

    session
        .execute("MATCH (a:M {id: '1'}) MATCH (b:M {id: '2'}) MERGE (a)-[:LINK]->(b)")
        .unwrap();

    assert_eq!(node_count(&db), 2, "MERGE should not create phantom nodes");
    assert_eq!(edge_count(&db), 1, "MERGE should create 1 edge");
}

// ---------------------------------------------------------------------------
// 9. Mixed — one node from MATCH, one inline
// ---------------------------------------------------------------------------

#[test]
fn test_create_rel_mixed_inline_and_reference() {
    let db = ObrainDB::new_in_memory();
    let session = db.session();

    session.execute("CREATE (:A {id: 'existing'})").unwrap();
    assert_eq!(node_count(&db), 1);

    session
        .execute("MATCH (a:A {id: 'existing'}) CREATE (a)-[:REL]->(b:B {id: 'new'})")
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
    let db = ObrainDB::new_in_memory();
    let session = db.session();

    session
        .execute("CREATE (:D {id: '1'}), (:D {id: '2'})")
        .unwrap();

    session
        .execute("MATCH (a:D {id: '1'}) MATCH (b:D {id: '2'}) CREATE (a)-[:FWD]->(b)")
        .unwrap();
    session
        .execute("MATCH (a:D {id: '1'}) MATCH (b:D {id: '2'}) CREATE (b)-[:BWD]->(a)")
        .unwrap();

    assert_eq!(
        node_count(&db),
        2,
        "bidirectional should not create phantoms"
    );
    assert_eq!(edge_count(&db), 2, "should have 2 directed edges");
}

// ---------------------------------------------------------------------------
// 11. Self-loop
// ---------------------------------------------------------------------------

#[test]
fn test_match_create_rel_self_loop() {
    let db = ObrainDB::new_in_memory();
    let session = db.session();

    session.execute("CREATE (:S {id: 'loop'})").unwrap();

    session
        .execute("MATCH (a:S {id: 'loop'}) CREATE (a)-[:SELF]->(a)")
        .unwrap();

    assert_eq!(
        node_count(&db),
        1,
        "self-loop must not create phantom nodes"
    );
    assert_eq!(edge_count(&db), 1, "self-loop should produce 1 edge");
}

// ---------------------------------------------------------------------------
// 12. Multiple CREATE clauses between same pair
// ---------------------------------------------------------------------------

#[test]
fn test_match_create_multiple_rels_same_pair() {
    let db = ObrainDB::new_in_memory();
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
    let db = ObrainDB::new_in_memory();
    let session = db.session();

    session
        .execute("CREATE (:Foo {id: '1'}), (:Bar {id: '2'})")
        .unwrap();

    session
        .execute("MATCH (a:Foo {id: '1'}) MATCH (b:Bar {id: '2'}) CREATE (a)-[:LINK]->(b)")
        .unwrap();

    let result = session.execute("MATCH (n:Foo) RETURN count(n)").unwrap();
    match &result.rows[0][0] {
        Value::Int64(n) => assert_eq!(*n, 1, "Foo label should still have 1 node"),
        other => panic!("expected Int64, got {:?}", other),
    }

    let result = session.execute("MATCH (n:Bar) RETURN count(n)").unwrap();
    match &result.rows[0][0] {
        Value::Int64(n) => assert_eq!(*n, 1, "Bar label should still have 1 node"),
        other => panic!("expected Int64, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// 14. Properties preserved after CREATE rel
// ---------------------------------------------------------------------------

#[test]
fn test_create_path_bare_variable_no_input() {
    // Covers the path where path.start has no labels/props AND no input operator.
    // Bare variables in CREATE without labels trigger the fallback node creation,
    // but edge creation then fails with a semantic error because the variable
    // isn't registered in the scope. We verify the error is raised gracefully.
    let db = ObrainDB::new_in_memory();
    let session = db.session();

    // CREATE (a)-[:REL]->(b) with no prior MATCH and no labels — semantic error expected
    let result = session.execute("CREATE (a)-[:REL]->(b)");
    assert!(
        result.is_err(),
        "Bare variables without labels in CREATE path should produce a semantic error"
    );

    // Same test via Cypher translator to cover cypher.rs bare-variable-no-input path
    let result_cypher = session.execute_cypher("CREATE (x)-[:REL]->(y)");
    assert!(
        result_cypher.is_err(),
        "Cypher: bare variables in CREATE path should produce an error"
    );
}

#[test]
fn test_match_create_rel_via_cypher_translator() {
    // Ensures the cypher.rs code path for bare-variable-with-input is exercised
    let db = ObrainDB::new_in_memory();
    let session = db.session();

    session
        .execute_cypher("CREATE (:CY {id: '1'}), (:CY {id: '2'})")
        .unwrap();
    assert_eq!(node_count(&db), 2);

    session
        .execute_cypher("MATCH (a:CY {id: '1'}) MATCH (b:CY {id: '2'}) CREATE (a)-[:CREL]->(b)")
        .unwrap();

    assert_eq!(node_count(&db), 2, "Cypher: phantom nodes created");
    assert_eq!(edge_count(&db), 1, "Cypher: relationship not created");
}

#[test]
fn test_create_chained_path_via_cypher() {
    // Covers the last_node_var tracker in cypher.rs translate_create_pattern
    let db = ObrainDB::new_in_memory();
    let session = db.session();

    session
        .execute_cypher("CREATE (a:CX {id: '1'})-[:R1]->(b:CY {id: '2'})-[:R2]->(c:CZ {id: '3'})")
        .unwrap();

    assert_eq!(node_count(&db), 3);
    assert_eq!(edge_count(&db), 2);

    // Now MATCH + CREATE between existing nodes via Cypher
    session
        .execute_cypher("MATCH (a:CX {id: '1'}) MATCH (c:CZ {id: '3'}) CREATE (a)-[:DIRECT]->(c)")
        .unwrap();

    assert_eq!(node_count(&db), 3, "Cypher chained: no phantom nodes");
    assert_eq!(edge_count(&db), 3);
}

#[test]
fn test_create_rel_with_single_label_start() {
    // Covers the non-bare path: start node has labels, target node has labels
    // Exercises the standard CREATE path pattern through translate_create_path
    let db = ObrainDB::new_in_memory();
    let session = db.session();

    session
        .execute("CREATE (a:Start {id: 1})-[:LINK]->(b:End {id: 2})")
        .unwrap();
    assert_eq!(node_count(&db), 2);
    assert_eq!(edge_count(&db), 1);

    // Now MATCH only the start node, CREATE edge to a new labeled node
    session
        .execute("MATCH (a:Start {id: 1}) CREATE (a)-[:NEW]->(c:Extra {id: 3})")
        .unwrap();
    assert_eq!(node_count(&db), 3, "Should create only 1 new node (c)");
    assert_eq!(edge_count(&db), 2);
}

#[test]
fn test_create_chained_path_mixed() {
    // Covers the last_node_var tracker across iterations in a chained path
    // where some nodes are inline and some are bare references
    let db = ObrainDB::new_in_memory();
    let session = db.session();

    session
        .execute("CREATE (a:X {id: '1'})-[:R1]->(b:Y {id: '2'})-[:R2]->(c:Z {id: '3'})")
        .unwrap();

    assert_eq!(node_count(&db), 3);
    assert_eq!(edge_count(&db), 2);

    // Now use MATCH + CREATE with the chained pattern
    session
        .execute("MATCH (a:X {id: '1'}) MATCH (c:Z {id: '3'}) CREATE (a)-[:DIRECT]->(c)")
        .unwrap();

    // Still 3 nodes, now 3 edges
    assert_eq!(node_count(&db), 3);
    assert_eq!(edge_count(&db), 3);
}

#[test]
fn test_properties_preserved_after_create_rel() {
    let db = ObrainDB::new_in_memory();
    let session = db.session();

    session
        .execute("CREATE (:W {name: 'alpha', val: 10}), (:W {name: 'beta', val: 20})")
        .unwrap();

    session
        .execute("MATCH (a:W {name: 'alpha'}) MATCH (b:W {name: 'beta'}) CREATE (a)-[:CONN]->(b)")
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
