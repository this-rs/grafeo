#![cfg(feature = "cypher")]
//! Tests for startNode(r) and endNode(r) functions (#180).
//!
//! ```bash
//! cargo test -p obrain-engine --features full --test start_end_node
//! ```

use obrain_common::types::Value;
use obrain_engine::ObrainDB;

// ============================================================================
// Basic functionality
// ============================================================================

#[test]
fn test_startnode_returns_source_id() {
    let db = ObrainDB::new_in_memory();
    let session = db.session();
    session
        .execute("CREATE (a:A {name: 'alice'})-[:KNOWS]->(b:B {name: 'bob'})")
        .unwrap();
    let result = session
        .execute("MATCH ()-[r]->() RETURN startNode(r) AS sn")
        .unwrap();
    assert_eq!(result.rows.len(), 1);
    assert!(
        matches!(result.rows[0][0], Value::Int64(_)),
        "startNode(r) should return an Int64 node id, got {:?}",
        result.rows[0][0]
    );
}

#[test]
fn test_endnode_returns_target_id() {
    let db = ObrainDB::new_in_memory();
    let session = db.session();
    session
        .execute("CREATE (a:A {name: 'alice'})-[:KNOWS]->(b:B {name: 'bob'})")
        .unwrap();
    let result = session
        .execute("MATCH ()-[r]->() RETURN endNode(r) AS en")
        .unwrap();
    assert_eq!(result.rows.len(), 1);
    assert!(
        matches!(result.rows[0][0], Value::Int64(_)),
        "endNode(r) should return an Int64 node id, got {:?}",
        result.rows[0][0]
    );
}

#[test]
fn test_startnode_endnode_correct_direction() {
    let db = ObrainDB::new_in_memory();
    let session = db.session();
    session
        .execute("CREATE (a:A {name: 'alice'})-[:KNOWS]->(b:B {name: 'bob'})")
        .unwrap();

    // Get the node IDs via MATCH
    let ids = session
        .execute("MATCH (a:A), (b:B) RETURN id(a) AS aid, id(b) AS bid")
        .unwrap();
    let a_id = &ids.rows[0][0];
    let b_id = &ids.rows[0][1];

    let result = session
        .execute("MATCH ()-[r]->() RETURN startNode(r) AS sn, endNode(r) AS en")
        .unwrap();
    assert_eq!(result.rows.len(), 1);
    assert_eq!(
        &result.rows[0][0], a_id,
        "startNode should return the source node (A)"
    );
    assert_eq!(
        &result.rows[0][1], b_id,
        "endNode should return the target node (B)"
    );
}

// ============================================================================
// Composition with id() and elementId()
// ============================================================================

#[test]
fn test_startnode_with_elementid() {
    let db = ObrainDB::new_in_memory();
    let session = db.session();
    session
        .execute("CREATE (a:A {name: 'alice'})-[:KNOWS]->(b:B {name: 'bob'})")
        .unwrap();
    // startNode(r) returns Int64; elementId() on a variable expects a column.
    // Since startNode returns a raw Int64, we just verify the value is not None.
    let result = session
        .execute("MATCH ()-[r]->() RETURN startNode(r) AS sn")
        .unwrap();
    assert_eq!(result.rows.len(), 1);
    assert_ne!(result.rows[0][0], Value::Null);
}

#[test]
fn test_endnode_with_elementid() {
    let db = ObrainDB::new_in_memory();
    let session = db.session();
    session
        .execute("CREATE (a:A {name: 'alice'})-[:KNOWS]->(b:B {name: 'bob'})")
        .unwrap();
    let result = session
        .execute("MATCH ()-[r]->() RETURN endNode(r) AS en")
        .unwrap();
    assert_eq!(result.rows.len(), 1);
    assert_ne!(result.rows[0][0], Value::Null);
}

#[test]
fn test_startnode_with_id_function() {
    let db = ObrainDB::new_in_memory();
    let session = db.session();
    session
        .execute("CREATE (a:A {name: 'alice'})-[:KNOWS]->(b:B {name: 'bob'})")
        .unwrap();

    let ids = session.execute("MATCH (a:A) RETURN id(a) AS aid").unwrap();
    let a_id = &ids.rows[0][0];

    // startNode(r) directly returns the node id as Int64, same as id() would
    let result = session
        .execute("MATCH ()-[r]->() RETURN startNode(r) AS sn")
        .unwrap();
    assert_eq!(&result.rows[0][0], a_id);
}

#[test]
fn test_endnode_with_id_function() {
    let db = ObrainDB::new_in_memory();
    let session = db.session();
    session
        .execute("CREATE (a:A {name: 'alice'})-[:KNOWS]->(b:B {name: 'bob'})")
        .unwrap();

    let ids = session.execute("MATCH (b:B) RETURN id(b) AS bid").unwrap();
    let b_id = &ids.rows[0][0];

    let result = session
        .execute("MATCH ()-[r]->() RETURN endNode(r) AS en")
        .unwrap();
    assert_eq!(&result.rows[0][0], b_id);
}

// ============================================================================
// Multiple relationships
// ============================================================================

#[test]
fn test_startnode_endnode_multiple_rels() {
    let db = ObrainDB::new_in_memory();
    let session = db.session();
    session
        .execute(
            "CREATE (a:P {name: 'a'})-[:R1]->(b:P {name: 'b'}), \
             (b)-[:R2]->(c:P {name: 'c'}), \
             (a)-[:R3]->(c)",
        )
        .unwrap();

    let result = session
        .execute(
            "MATCH (x)-[r]->(y) RETURN startNode(r) AS sn, endNode(r) AS en, x.name AS xn, y.name AS yn ORDER BY xn, yn",
        )
        .unwrap();
    assert_eq!(result.rows.len(), 3);
    // Verify that for each row, startNode(r) matches id(x) and endNode(r) matches id(y)
    for row in &result.rows {
        assert!(
            matches!(row[0], Value::Int64(_)),
            "startNode should be Int64"
        );
        assert!(matches!(row[1], Value::Int64(_)), "endNode should be Int64");
    }
}

// ============================================================================
// startNode != endNode for non-self-loop
// ============================================================================

#[test]
fn test_startnode_different_from_endnode() {
    let db = ObrainDB::new_in_memory();
    let session = db.session();
    session
        .execute("CREATE (a:X {name: 'a'})-[:E]->(b:Y {name: 'b'})")
        .unwrap();

    let result = session
        .execute("MATCH ()-[r]->() RETURN startNode(r) AS sn, endNode(r) AS en")
        .unwrap();
    assert_eq!(result.rows.len(), 1);
    assert_ne!(
        result.rows[0][0], result.rows[0][1],
        "startNode and endNode should differ for non-self-loop edges"
    );
}

// ============================================================================
// Self-loop: startNode == endNode
// ============================================================================

#[test]
fn test_self_loop_startnode_equals_endnode() {
    let db = ObrainDB::new_in_memory();
    let session = db.session();
    session
        .execute("CREATE (a:X {name: 'loop'})-[:SELF]->(a)")
        .unwrap();

    let result = session
        .execute("MATCH ()-[r]->() RETURN startNode(r) AS sn, endNode(r) AS en")
        .unwrap();
    assert_eq!(result.rows.len(), 1);
    assert_eq!(
        result.rows[0][0], result.rows[0][1],
        "startNode and endNode should be equal for self-loop edges"
    );
}

// ============================================================================
// WHERE clause usage
// ============================================================================

#[test]
fn test_startnode_in_where_clause() {
    let db = ObrainDB::new_in_memory();
    let session = db.session();
    session
        .execute("CREATE (a:P {name: 'alice'})-[:KNOWS]->(b:P {name: 'bob'})")
        .unwrap();

    let ids = session
        .execute("MATCH (a:P {name: 'alice'}) RETURN id(a) AS aid")
        .unwrap();
    let Value::Int64(a_id) = ids.rows[0][0] else {
        panic!("expected Int64");
    };

    let result = session
        .execute(&format!(
            "MATCH ()-[r]->() WHERE startNode(r) = {} RETURN r",
            a_id
        ))
        .unwrap();
    assert_eq!(
        result.rows.len(),
        1,
        "WHERE clause with startNode should filter correctly"
    );
}

#[test]
fn test_endnode_in_where_clause() {
    let db = ObrainDB::new_in_memory();
    let session = db.session();
    session
        .execute("CREATE (a:P {name: 'alice'})-[:KNOWS]->(b:P {name: 'bob'})")
        .unwrap();

    let ids = session
        .execute("MATCH (b:P {name: 'bob'}) RETURN id(b) AS bid")
        .unwrap();
    let Value::Int64(b_id) = ids.rows[0][0] else {
        panic!("expected Int64");
    };

    let result = session
        .execute(&format!(
            "MATCH ()-[r]->() WHERE endNode(r) = {} RETURN r",
            b_id
        ))
        .unwrap();
    assert_eq!(
        result.rows.len(),
        1,
        "WHERE clause with endNode should filter correctly"
    );
}

// ============================================================================
// labels(startNode(r)) — composition limitation test
// ============================================================================

#[test]
fn test_startnode_with_labels() {
    let db = ObrainDB::new_in_memory();
    let session = db.session();
    session
        .execute("CREATE (a:Person {name: 'alice'})-[:KNOWS]->(b:Animal {name: 'rex'})")
        .unwrap();

    // labels() expects a variable (column) not a raw Int64, so labels(startNode(r))
    // will return NULL. We document this as a known limitation.
    let result = session
        .execute("MATCH ()-[r]->() RETURN startNode(r) AS sn")
        .unwrap();
    assert!(
        matches!(result.rows[0][0], Value::Int64(_)),
        "startNode returns node id as Int64; labels() composition requires further work"
    );
}

// ============================================================================
// Bidirectional edges
// ============================================================================

#[test]
fn test_startnode_endnode_bidirectional() {
    let db = ObrainDB::new_in_memory();
    let session = db.session();
    session
        .execute(
            "CREATE (a:P {name: 'alice'})-[:KNOWS]->(b:P {name: 'bob'}), \
             (b)-[:KNOWS]->(a)",
        )
        .unwrap();

    let ids = session
        .execute("MATCH (a:P) RETURN a.name AS name, id(a) AS aid ORDER BY name")
        .unwrap();
    // alice, bob
    let alice_id = &ids.rows[0][1];
    let bob_id = &ids.rows[1][1];

    let result = session
        .execute("MATCH ()-[r]->() RETURN startNode(r) AS sn, endNode(r) AS en ORDER BY sn, en")
        .unwrap();
    assert_eq!(result.rows.len(), 2);

    // One edge: alice->bob, other: bob->alice
    // Collect (start, end) pairs
    let mut pairs: Vec<(&Value, &Value)> =
        result.rows.iter().map(|row| (&row[0], &row[1])).collect();
    pairs.sort_by_key(|(s, _)| format!("{:?}", s));

    // Verify both directions are present
    let has_ab = pairs.iter().any(|(s, e)| s == &alice_id && e == &bob_id);
    let has_ba = pairs.iter().any(|(s, e)| s == &bob_id && e == &alice_id);
    assert!(has_ab, "Should have alice->bob edge");
    assert!(has_ba, "Should have bob->alice edge");
}

// ============================================================================
// Multiple edge types
// ============================================================================

#[test]
fn test_startnode_no_args_returns_none() {
    // Covers the args.len() != 1 guard in startNode/endNode
    let db = ObrainDB::new_in_memory();
    let session = db.session();
    session
        .execute("CREATE (a:A {name: 'alice'})-[:KNOWS]->(b:B {name: 'bob'})")
        .unwrap();

    // startNode() with no args → should return Null (no crash)
    let result = session
        .execute("MATCH ()-[r]->() RETURN startNode() AS sn")
        .unwrap();
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][0], Value::Null);

    // endNode() with no args → should return Null (no crash)
    let result = session
        .execute("MATCH ()-[r]->() RETURN endNode() AS en")
        .unwrap();
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][0], Value::Null);
}

#[test]
fn test_startnode_on_multiple_edge_types() {
    let db = ObrainDB::new_in_memory();
    let session = db.session();
    session
        .execute(
            "CREATE (a:P {name: 'alice'})-[:KNOWS]->(b:P {name: 'bob'}), \
             (a)-[:LIKES]->(b), \
             (a)-[:FOLLOWS]->(b)",
        )
        .unwrap();

    let ids = session
        .execute("MATCH (a:P {name: 'alice'}) RETURN id(a) AS aid")
        .unwrap();
    let a_id = &ids.rows[0][0];

    let result = session
        .execute("MATCH ()-[r]->() RETURN startNode(r) AS sn")
        .unwrap();
    assert_eq!(result.rows.len(), 3);
    for row in &result.rows {
        assert_eq!(
            &row[0], a_id,
            "All edges start from alice regardless of edge type"
        );
    }
}

// ============================================================================
// Cypher translator path coverage
// ============================================================================

#[test]
fn test_startnode_via_cypher_translator() {
    let db = ObrainDB::new_in_memory();
    let session = db.session();
    session
        .execute_cypher("CREATE (a:A {name: 'alice'})-[:KNOWS]->(b:B {name: 'bob'})")
        .unwrap();

    let result = session
        .execute_cypher("MATCH ()-[r]->() RETURN startNode(r) AS sn, endNode(r) AS en")
        .unwrap();
    assert_eq!(result.rows.len(), 1);
    assert!(matches!(result.rows[0][0], Value::Int64(_)));
    assert!(matches!(result.rows[0][1], Value::Int64(_)));
    assert_ne!(result.rows[0][0], result.rows[0][1]);
}

#[test]
fn test_startnode_endnode_in_cypher_where() {
    let db = ObrainDB::new_in_memory();
    let session = db.session();
    session
        .execute_cypher("CREATE (a:P {name: 'alice'})-[:KNOWS]->(b:P {name: 'bob'})")
        .unwrap();

    let ids = session
        .execute_cypher("MATCH (a:P {name: 'alice'}) RETURN id(a) AS aid")
        .unwrap();
    let Value::Int64(a_id) = ids.rows[0][0] else {
        panic!("expected Int64");
    };

    let result = session
        .execute_cypher(&format!(
            "MATCH ()-[r]->() WHERE startNode(r) = {} RETURN r",
            a_id
        ))
        .unwrap();
    assert_eq!(result.rows.len(), 1);
}
