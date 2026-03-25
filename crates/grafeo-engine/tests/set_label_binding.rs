//! Regression tests for SET/REMOVE label variable binding preservation.
//!
//! Covers issues #178 and #182: SET n:Label and REMOVE n:Label operators
//! must preserve all input variable bindings so that subsequent clauses
//! (RETURN, SET property, REMOVE property) can still reference them.
//!
//! Run with:
//! ```bash
//! cargo test -p grafeo-engine --features full --test set_label_binding
//! ```

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn db_with_nodes(count: usize) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for i in 0..count {
        session
            .execute(&format!(
                "CREATE (n:Node {{name: 'node{}', age: {}}})",
                i,
                20 + i * 5
            ))
            .unwrap();
    }
    db
}

// ---------------------------------------------------------------------------
// 1. SET label + count(*) returns correct count
// ---------------------------------------------------------------------------

#[test]
fn test_set_label_count_star_returns_correct_count() {
    let db = db_with_nodes(3);
    let session = db.session();
    let result = session
        .execute("MATCH (n:Node) SET n:Tagged RETURN count(*) AS cnt")
        .unwrap();
    assert_eq!(result.row_count(), 1, "count(*) should return 1 row");
    if let Value::Int64(cnt) = &result.rows[0][0] {
        assert_eq!(*cnt, 3, "count(*) should be 3");
    } else {
        panic!("Expected Int64, got {:?}", result.rows[0][0]);
    }
}

// ---------------------------------------------------------------------------
// 2. SET label + count(n) returns correct count
// ---------------------------------------------------------------------------

#[test]
fn test_set_label_count_variable_returns_correct_count() {
    let db = db_with_nodes(3);
    let session = db.session();
    let result = session
        .execute("MATCH (n:Node) SET n:Tagged RETURN count(n) AS cnt")
        .unwrap();
    assert_eq!(result.row_count(), 1);
    if let Value::Int64(cnt) = &result.rows[0][0] {
        assert_eq!(*cnt, 3, "count(n) should be 3");
    } else {
        panic!("Expected Int64, got {:?}", result.rows[0][0]);
    }
}

// ---------------------------------------------------------------------------
// 3. SET label then RETURN property — variable still accessible
// ---------------------------------------------------------------------------

#[test]
fn test_set_label_then_return_property() {
    let db = db_with_nodes(3);
    let session = db.session();
    let result = session
        .execute("MATCH (n:Node) SET n:Tagged RETURN n.name")
        .unwrap();
    assert_eq!(result.row_count(), 3, "Should return all 3 rows");
    // All values must be strings (names)
    for row in &result.rows {
        assert!(
            matches!(&row[0], Value::String(_)),
            "Expected string name, got {:?}",
            row[0]
        );
    }
}

// ---------------------------------------------------------------------------
// 4. SET label then SET property — no crash, property set
// ---------------------------------------------------------------------------

#[test]
fn test_set_label_then_set_property() {
    let db = db_with_nodes(3);
    let session = db.session();
    session
        .execute("MATCH (n:Node) SET n:Tagged SET n.flag = true")
        .unwrap();
    let result = session.execute("MATCH (n:Node) RETURN n.flag").unwrap();
    assert_eq!(result.row_count(), 3);
    for row in &result.rows {
        assert_eq!(row[0], Value::Bool(true), "flag should be true");
    }
}

// ---------------------------------------------------------------------------
// 5. SET label then REMOVE property — no crash
// ---------------------------------------------------------------------------

#[test]
fn test_set_label_then_remove_property() {
    let db = db_with_nodes(3);
    let session = db.session();
    // Should not crash — the variable n must survive the SET label operator
    session
        .execute("MATCH (n:Node) SET n:Tagged REMOVE n.age")
        .unwrap();
    let result = session.execute("MATCH (n:Node) RETURN n.age").unwrap();
    assert_eq!(result.row_count(), 3);
    for row in &result.rows {
        assert_eq!(row[0], Value::Null, "age should have been removed");
    }
}

// ---------------------------------------------------------------------------
// 6. SET label then REMOVE label — both operations work
// ---------------------------------------------------------------------------

#[test]
fn test_set_label_then_remove_label() {
    let db = db_with_nodes(2);
    let session = db.session();
    session
        .execute("MATCH (n:Node) SET n:Tagged REMOVE n:Node")
        .unwrap();
    // Nodes should now have Tagged but not Node
    let result = session
        .execute("MATCH (n:Tagged) RETURN count(*) AS cnt")
        .unwrap();
    assert_eq!(result.row_count(), 1);
    if let Value::Int64(cnt) = &result.rows[0][0] {
        assert_eq!(*cnt, 2);
    } else {
        panic!("Expected Int64");
    }
}

// ---------------------------------------------------------------------------
// 7. REMOVE label then SET label — reverse order works
// ---------------------------------------------------------------------------

#[test]
fn test_remove_label_then_set_label() {
    let db = db_with_nodes(2);
    let session = db.session();
    // REMOVE then SET in separate statements to avoid ordering ambiguity
    session.execute("MATCH (n:Node) REMOVE n:Node").unwrap();
    session.execute("MATCH (n) SET n:Replacement").unwrap();
    let result = session
        .execute("MATCH (n:Replacement) RETURN count(*) AS cnt")
        .unwrap();
    if let Value::Int64(cnt) = &result.rows[0][0] {
        assert_eq!(*cnt, 2);
    } else {
        panic!("Expected Int64");
    }
}

// ---------------------------------------------------------------------------
// 8. REMOVE label + count(*) returns correct count
// ---------------------------------------------------------------------------

#[test]
fn test_remove_label_count_star() {
    let db = db_with_nodes(3);
    let session = db.session();
    let result = session
        .execute("MATCH (n:Node) REMOVE n:Node RETURN count(*) AS cnt")
        .unwrap();
    assert_eq!(result.row_count(), 1);
    if let Value::Int64(cnt) = &result.rows[0][0] {
        assert_eq!(*cnt, 3, "All 3 rows should pass through REMOVE label");
    } else {
        panic!("Expected Int64, got {:?}", result.rows[0][0]);
    }
}

// ---------------------------------------------------------------------------
// 9. SET label preserves all columns (multi-variable pattern)
// ---------------------------------------------------------------------------

#[test]
fn test_set_label_preserves_all_columns() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("CREATE (a:Person {name: 'Alice'})-[:KNOWS]->(b:Person {name: 'Bob'})")
        .unwrap();
    let result = session
        .execute("MATCH (a:Person)-[r:KNOWS]->(b:Person) SET a:Tagged RETURN a.name, b.name")
        .unwrap();
    assert_eq!(result.row_count(), 1, "Should return 1 relationship row");
    assert_eq!(result.rows[0].len(), 2, "Should have 2 columns");
}

// ---------------------------------------------------------------------------
// 10. SET label with WHERE clause — correct filtered count
// ---------------------------------------------------------------------------

#[test]
fn test_set_label_with_where_clause() {
    let db = db_with_nodes(3); // ages: 20, 25, 30
    let session = db.session();
    let result = session
        .execute("MATCH (n:Node) WHERE n.age > 20 SET n:Adult RETURN count(*) AS cnt")
        .unwrap();
    assert_eq!(result.row_count(), 1);
    if let Value::Int64(cnt) = &result.rows[0][0] {
        assert_eq!(*cnt, 2, "Only 2 nodes have age > 20");
    } else {
        panic!("Expected Int64");
    }
}

// ---------------------------------------------------------------------------
// 11. MERGE + SET label + REMOVE property — exact #178 repro
// ---------------------------------------------------------------------------

#[test]
fn test_merge_set_label_remove_property() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("CREATE (n:X {id: 1, foreign: 'yes'})")
        .unwrap();
    // This was the crash scenario from #178
    session
        .execute("MERGE (n:X {id: 1}) SET n:Tenant REMOVE n.foreign")
        .unwrap();
    let result = session.execute("MATCH (n:X) RETURN n.foreign").unwrap();
    assert_eq!(result.row_count(), 1);
    assert_eq!(
        result.rows[0][0],
        Value::Null,
        "foreign property should be removed"
    );

    // Verify label was added
    let result = session
        .execute("MATCH (n:Tenant) RETURN count(*) AS cnt")
        .unwrap();
    if let Value::Int64(cnt) = &result.rows[0][0] {
        assert_eq!(*cnt, 1);
    } else {
        panic!("Expected Int64");
    }
}

// ---------------------------------------------------------------------------
// 12. SET multiple labels
// ---------------------------------------------------------------------------

#[test]
fn test_set_multiple_labels() {
    let db = db_with_nodes(2);
    let session = db.session();
    let result = session
        .execute("MATCH (n:Node) SET n:A:B RETURN count(*) AS cnt")
        .unwrap();
    assert_eq!(result.row_count(), 1);
    if let Value::Int64(cnt) = &result.rows[0][0] {
        assert_eq!(*cnt, 2);
    } else {
        panic!("Expected Int64");
    }
}

// ---------------------------------------------------------------------------
// 13. SET label on single node
// ---------------------------------------------------------------------------

#[test]
fn test_set_label_on_single_node() {
    let db = db_with_nodes(1);
    let session = db.session();
    let result = session
        .execute("MATCH (n:Node) SET n:Tagged RETURN count(*) AS cnt")
        .unwrap();
    assert_eq!(result.row_count(), 1);
    if let Value::Int64(cnt) = &result.rows[0][0] {
        assert_eq!(*cnt, 1, "Single node should yield count 1");
    } else {
        panic!("Expected Int64");
    }
}

// ---------------------------------------------------------------------------
// 14. SET label on empty match — 0 rows
// ---------------------------------------------------------------------------

#[test]
fn test_set_label_on_empty_match() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let result = session
        .execute("MATCH (n:NonExistent) SET n:Tagged RETURN count(*) AS cnt")
        .unwrap();
    assert_eq!(result.row_count(), 1);
    if let Value::Int64(cnt) = &result.rows[0][0] {
        assert_eq!(*cnt, 0, "No nodes matched, count should be 0");
    } else {
        panic!("Expected Int64");
    }
}

// ---------------------------------------------------------------------------
// 15. SET same label twice — idempotent, no crash
// ---------------------------------------------------------------------------

#[test]
fn test_set_label_idempotent() {
    let db = db_with_nodes(2);
    let session = db.session();
    // First SET
    session.execute("MATCH (n:Node) SET n:Tagged").unwrap();
    // Second SET — same label again
    let result = session
        .execute("MATCH (n:Node) SET n:Tagged RETURN count(*) AS cnt")
        .unwrap();
    assert_eq!(result.row_count(), 1);
    if let Value::Int64(cnt) = &result.rows[0][0] {
        assert_eq!(*cnt, 2, "Idempotent SET should still pass through all rows");
    } else {
        panic!("Expected Int64");
    }
}

// ---------------------------------------------------------------------------
// 16. REMOVE label preserves variable — RETURN n.name works
// ---------------------------------------------------------------------------

#[test]
fn test_set_label_preserves_null_properties() {
    // Covers the push_value(Value::Null) branch in column copy when a property is null
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("CREATE (:Node {name: 'a'}), (:Node {name: 'b', extra: 'yes'})")
        .unwrap();

    // extra is null for node 'a', non-null for node 'b'
    let result = session
        .execute("MATCH (n:Node) SET n:Tagged RETURN n.name, n.extra ORDER BY n.name")
        .unwrap();
    assert_eq!(result.row_count(), 2);
    // Node 'a' has null extra
    assert_eq!(result.rows[0][1], Value::Null);
    // Node 'b' has extra = 'yes'
    match &result.rows[1][1] {
        Value::String(s) => assert_eq!(s.as_str(), "yes"),
        other => panic!("Expected String, got {:?}", other),
    }
}

#[test]
fn test_set_label_in_transaction() {
    // Covers the versioned label path (add_label_versioned / remove_label_versioned)
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.execute("CREATE (:Node {name: 'x'})").unwrap();

    session.begin_transaction().unwrap();
    let result = session
        .execute("MATCH (n:Node) SET n:InTx RETURN count(*) AS cnt")
        .unwrap();
    assert_eq!(result.row_count(), 1);
    if let Value::Int64(cnt) = &result.rows[0][0] {
        assert_eq!(*cnt, 1);
    }
    session.commit().unwrap();

    // Verify label persists after commit
    let result = session
        .execute("MATCH (n:InTx) RETURN count(n) AS cnt")
        .unwrap();
    if let Value::Int64(cnt) = &result.rows[0][0] {
        assert_eq!(*cnt, 1, "Label should persist after commit");
    }
}

#[test]
fn test_remove_label_in_transaction() {
    // Covers remove_label_versioned path
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.execute("CREATE (:Node:Temp {name: 'y'})").unwrap();

    session.begin_transaction().unwrap();
    let result = session
        .execute("MATCH (n:Temp) REMOVE n:Temp RETURN count(*) AS cnt")
        .unwrap();
    assert_eq!(result.row_count(), 1);
    session.commit().unwrap();

    let result = session
        .execute("MATCH (n:Temp) RETURN count(n) AS cnt")
        .unwrap();
    if let Value::Int64(cnt) = &result.rows[0][0] {
        assert_eq!(*cnt, 0, "Temp label should be removed after commit");
    }
}

#[test]
fn test_remove_label_preserves_variable() {
    let db = db_with_nodes(3);
    let session = db.session();
    let result = session
        .execute("MATCH (n:Node) REMOVE n:Node RETURN n.name")
        .unwrap();
    assert_eq!(result.row_count(), 3, "All 3 rows should be returned");
    for row in &result.rows {
        assert!(
            matches!(&row[0], Value::String(_)),
            "Expected string name, got {:?}",
            row[0]
        );
    }
}
