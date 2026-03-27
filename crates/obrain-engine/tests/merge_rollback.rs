//! Tests for MERGE operator rollback behavior.
//!
//! Verifies that ON MATCH SET properties written by MERGE are correctly
//! undone when a transaction is rolled back.

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

#[test]
fn test_merge_on_match_set_rollback() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    // Create existing node
    session
        .execute("INSERT (:Person {name: 'Alix', status: 'active'})")
        .unwrap();

    // MERGE should match the existing node and apply ON MATCH SET
    session.begin_transaction().unwrap();
    session
        .execute("MERGE (p:Person {name: 'Alix'}) ON MATCH SET p.status = 'inactive'")
        .unwrap();

    // Verify the property was updated within the transaction
    let result = session
        .execute("MATCH (p:Person {name: 'Alix'}) RETURN p.status")
        .unwrap();
    assert_eq!(result.rows[0][0], Value::String("inactive".into()));

    session.rollback().unwrap();

    // Property should be restored to original value
    let result = session
        .execute("MATCH (p:Person {name: 'Alix'}) RETURN p.status")
        .unwrap();
    assert_eq!(
        result.rows[0][0],
        Value::String("active".into()),
        "status should be restored to 'active' after rollback"
    );
}

#[test]
fn test_merge_on_match_set_new_property_rollback() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    // Create existing node without 'updated' property
    session.execute("INSERT (:Person {name: 'Gus'})").unwrap();

    // MERGE adds a new property via ON MATCH SET
    session.begin_transaction().unwrap();
    session
        .execute("MERGE (p:Person {name: 'Gus'}) ON MATCH SET p.updated = true")
        .unwrap();
    session.rollback().unwrap();

    // New property should not exist after rollback
    let result = session
        .execute("MATCH (p:Person {name: 'Gus'}) RETURN p.updated")
        .unwrap();
    assert_eq!(
        result.rows[0][0],
        Value::Null,
        "'updated' property should not exist after rollback"
    );
}

#[test]
fn test_merge_on_match_committed_stays() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    session
        .execute("INSERT (:Person {name: 'Vincent', score: 0})")
        .unwrap();

    // MERGE with commit: property should persist
    session.begin_transaction().unwrap();
    session
        .execute("MERGE (p:Person {name: 'Vincent'}) ON MATCH SET p.score = 100")
        .unwrap();
    session.commit().unwrap();

    let result = session
        .execute("MATCH (p:Person {name: 'Vincent'}) RETURN p.score")
        .unwrap();
    assert_eq!(
        result.rows[0][0],
        Value::Int64(100),
        "score should retain the committed value"
    );
}
