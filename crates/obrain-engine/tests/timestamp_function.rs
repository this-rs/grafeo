//! Integration tests for the `timestamp()` function.
//!
//! `timestamp()` follows Neo4j semantics: returns the current time as
//! milliseconds since the Unix epoch, typed as `Int64`.

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn current_millis() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64
}

// ---------------------------------------------------------------------------
// 1. Basic RETURN
// ---------------------------------------------------------------------------

#[test]
fn test_timestamp_return_basic() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let result = session.execute("RETURN timestamp() AS ts").unwrap();

    assert_eq!(result.rows.len(), 1);
    match &result.rows[0][0] {
        Value::Int64(v) => assert!(*v > 0, "timestamp should be positive"),
        other => panic!("expected Int64, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// 2. Reasonable value (within 10 seconds of current time)
// ---------------------------------------------------------------------------

#[test]
fn test_timestamp_return_reasonable_value() {
    let before = current_millis();
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let result = session.execute("RETURN timestamp() AS ts").unwrap();
    let after = current_millis();

    match &result.rows[0][0] {
        Value::Int64(v) => {
            assert!(
                *v >= before - 10_000 && *v <= after + 10_000,
                "timestamp {} should be within 10s of [{}, {}]",
                v,
                before,
                after
            );
        }
        other => panic!("expected Int64, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// 3. SET clause
// ---------------------------------------------------------------------------

#[test]
fn test_timestamp_in_set_clause() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute("CREATE (:T {id: 'x'})").unwrap();
    session
        .execute("MATCH (n:T {id: 'x'}) SET n.ts = timestamp()")
        .unwrap();

    let result = session
        .execute("MATCH (n:T {id: 'x'}) RETURN n.ts")
        .unwrap();
    assert_eq!(result.rows.len(), 1);
    match &result.rows[0][0] {
        Value::Int64(v) => assert!(*v > 0),
        other => panic!("expected Int64 for SET timestamp, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// 4. ON CREATE SET
// ---------------------------------------------------------------------------

#[test]
fn test_timestamp_in_on_create_set() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("MERGE (n:T {id: 'y'}) ON CREATE SET n.created = timestamp()")
        .unwrap();

    let result = session
        .execute("MATCH (n:T {id: 'y'}) RETURN n.created")
        .unwrap();
    assert_eq!(result.rows.len(), 1);
    match &result.rows[0][0] {
        Value::Int64(v) => assert!(*v > 0),
        other => panic!("expected Int64 for ON CREATE SET, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// 5. ON MATCH SET
// ---------------------------------------------------------------------------

#[test]
fn test_timestamp_in_on_match_set() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    // First create the node
    session.execute("CREATE (:T {id: 'y'})").unwrap();
    // Then MERGE with ON MATCH SET
    session
        .execute("MERGE (n:T {id: 'y'}) ON MATCH SET n.updated = timestamp()")
        .unwrap();

    let result = session
        .execute("MATCH (n:T {id: 'y'}) RETURN n.updated")
        .unwrap();
    assert_eq!(result.rows.len(), 1);
    match &result.rows[0][0] {
        Value::Int64(v) => assert!(*v > 0),
        other => panic!("expected Int64 for ON MATCH SET, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// 6. Multiple calls close together
// ---------------------------------------------------------------------------

#[test]
fn test_timestamp_multiple_calls_close() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let result = session
        .execute("RETURN timestamp() AS t1, timestamp() AS t2")
        .unwrap();

    match (&result.rows[0][0], &result.rows[0][1]) {
        (Value::Int64(t1), Value::Int64(t2)) => {
            let diff = (t1 - t2).unsigned_abs();
            assert!(
                diff <= 1000,
                "two timestamp() calls in same query should be within 1000ms, got diff={}",
                diff
            );
        }
        other => panic!("expected (Int64, Int64), got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// 7. WHERE clause
// ---------------------------------------------------------------------------

#[test]
fn test_timestamp_in_where_clause() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute("CREATE (:T {id: 'w'})").unwrap();

    let result = session
        .execute("MATCH (n:T) WHERE timestamp() > 0 RETURN n.id")
        .unwrap();
    assert!(
        !result.rows.is_empty(),
        "WHERE timestamp() > 0 should not filter out rows"
    );
}

// ---------------------------------------------------------------------------
// 8. Arithmetic
// ---------------------------------------------------------------------------

#[test]
fn test_timestamp_arithmetic() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let result = session
        .execute("RETURN timestamp() - timestamp() AS diff")
        .unwrap();

    match &result.rows[0][0] {
        Value::Int64(v) => {
            assert!(
                v.unsigned_abs() <= 100,
                "timestamp() - timestamp() should be near 0, got {}",
                v
            );
        }
        other => panic!("expected Int64, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// 9. With alias (multiple aliases)
// ---------------------------------------------------------------------------

#[test]
fn test_timestamp_with_alias() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let result = session
        .execute("RETURN timestamp() AS t1, timestamp() AS t2")
        .unwrap();

    assert_eq!(result.rows.len(), 1);
    assert!(matches!(&result.rows[0][0], Value::Int64(_)));
    assert!(matches!(&result.rows[0][1], Value::Int64(_)));
}

// ---------------------------------------------------------------------------
// 10. No arguments only
// ---------------------------------------------------------------------------

#[test]
fn test_timestamp_no_args_only() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    // timestamp() with no args should work
    let result = session.execute("RETURN timestamp() AS ts").unwrap();
    assert!(matches!(&result.rows[0][0], Value::Int64(_)));
}

// ---------------------------------------------------------------------------
// 11. Stored and retrieved
// ---------------------------------------------------------------------------

#[test]
fn test_timestamp_stored_and_retrieved() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute("CREATE (:Ev {ts: timestamp()})").unwrap();

    let result = session.execute("MATCH (e:Ev) RETURN e.ts").unwrap();
    assert_eq!(result.rows.len(), 1);
    match &result.rows[0][0] {
        Value::Int64(v) => {
            assert!(
                *v > 1_000_000_000_000,
                "stored timestamp should be a realistic epoch-millis value, got {}",
                v
            );
        }
        other => panic!("expected Int64, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// 12. timestamp() vs now() — different types
// ---------------------------------------------------------------------------

#[test]
fn test_timestamp_vs_now_different_types() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let result = session
        .execute("RETURN timestamp() AS ts, now() AS n")
        .unwrap();

    assert_eq!(result.rows.len(), 1);
    assert!(
        matches!(&result.rows[0][0], Value::Int64(_)),
        "timestamp() should be Int64, got {:?}",
        &result.rows[0][0]
    );
    assert!(
        matches!(&result.rows[0][1], Value::Timestamp(_)),
        "now() should be Timestamp, got {:?}",
        &result.rows[0][1]
    );
}

// ---------------------------------------------------------------------------
// 13. CASE expression
// ---------------------------------------------------------------------------

#[test]
fn test_timestamp_in_case_expression() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let result = session
        .execute("RETURN CASE WHEN true THEN timestamp() ELSE 0 END AS ts")
        .unwrap();

    assert_eq!(result.rows.len(), 1);
    match &result.rows[0][0] {
        Value::Int64(v) => assert!(*v > 0, "CASE result should be positive timestamp"),
        other => panic!("expected Int64 from CASE, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// 14. In list
// ---------------------------------------------------------------------------

#[test]
fn test_timestamp_in_list() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let result = session
        .execute("RETURN [timestamp(), timestamp()] AS ts_list")
        .unwrap();

    assert_eq!(result.rows.len(), 1);
    match &result.rows[0][0] {
        Value::List(items) => {
            assert_eq!(items.len(), 2, "list should have 2 elements");
            for (i, item) in items.iter().enumerate() {
                assert!(
                    matches!(item, Value::Int64(v) if *v > 0),
                    "list element {} should be a positive Int64, got {:?}",
                    i,
                    item
                );
            }
        }
        other => panic!("expected List, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// 15. timestamp() in Cypher WHERE — forces runtime eval in filter.rs
// ---------------------------------------------------------------------------

#[test]
fn test_timestamp_in_cypher_where_clause() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute("CREATE (:TW {ts: 0})").unwrap();

    // Use a comparison with a property to prevent constant folding
    let result = session
        .execute("MATCH (n:TW) WHERE timestamp() > n.ts RETURN n.ts")
        .unwrap();
    assert_eq!(
        result.rows.len(),
        1,
        "timestamp() > 0 should match the node"
    );
}

// ---------------------------------------------------------------------------
// 16. timestamp() in Cypher RETURN via execute_cypher
// ---------------------------------------------------------------------------

#[test]
fn test_timestamp_via_execute_cypher() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let result = session.execute_cypher("RETURN timestamp() AS ts").unwrap();
    assert_eq!(result.rows.len(), 1);
    match &result.rows[0][0] {
        Value::Int64(v) => assert!(*v > 0, "Cypher timestamp() should be positive"),
        other => panic!("expected Int64, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// 17. timestamp() in Cypher SET via execute_cypher
// ---------------------------------------------------------------------------

#[test]
fn test_timestamp_in_cypher_set() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute_cypher("CREATE (:CTS {id: 1})").unwrap();
    session
        .execute_cypher("MATCH (n:CTS) SET n.ts = timestamp()")
        .unwrap();
    let result = session.execute_cypher("MATCH (n:CTS) RETURN n.ts").unwrap();
    assert_eq!(result.rows.len(), 1);
    match &result.rows[0][0] {
        Value::Int64(v) => assert!(*v > 0),
        other => panic!("expected Int64, got {:?}", other),
    }
}
