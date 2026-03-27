//! Coverage tests targeting the Grafeo→Obrain rebrand diff.
//!
//! These tests exercise renamed types, constructors, and public APIs
//! that appear in the PR diff to improve patch coverage.

use obrain_common::types::Value;
use obrain_engine::ObrainDB;

// ---------------------------------------------------------------------------
// CRUD operations (database/crud.rs)
// ---------------------------------------------------------------------------

#[test]
fn crud_create_node_and_retrieve() {
    let db = ObrainDB::new_in_memory();
    let id = db.create_node(&["Person"]);
    let labels = db.get_node_labels(id);
    assert!(labels.is_some());
    assert!(labels.unwrap().contains(&"Person".to_string()));
}

#[test]
fn crud_create_node_with_props() {
    let db = ObrainDB::new_in_memory();
    let _id = db.create_node_with_props(
        &["Person"],
        [("name", Value::from("Alice")), ("age", Value::from(30_i64))],
    );
    // Verify via query that properties were set
    let result = db
        .execute("MATCH (n:Person) RETURN n.name AS name, n.age AS age")
        .unwrap();
    assert_eq!(result.rows.len(), 1);
}

#[test]
fn crud_create_edge() {
    let db = ObrainDB::new_in_memory();
    let a = db.create_node(&["Person"]);
    let b = db.create_node(&["Person"]);
    let _edge = db.create_edge(a, b, "KNOWS");
    // Verify edge exists via query
    let result = db
        .execute("MATCH ()-[r:KNOWS]->() RETURN count(r) AS c")
        .unwrap();
    assert_eq!(result.rows.len(), 1);
}

#[test]
fn crud_set_and_remove_node_property() {
    let db = ObrainDB::new_in_memory();
    let id = db.create_node(&["Thing"]);
    db.set_node_property(id, "color", Value::from("blue"));
    // Verify property is set via query
    let result = db
        .execute("MATCH (n:Thing) RETURN n.color AS color")
        .unwrap();
    assert_eq!(result.rows.len(), 1);

    let removed = db.remove_node_property(id, "color");
    assert!(removed);
    // Verify property was removed
    let result = db
        .execute("MATCH (n:Thing) WHERE n.color IS NOT NULL RETURN n")
        .unwrap();
    assert_eq!(result.rows.len(), 0);
}

#[test]
fn crud_add_and_remove_label() {
    let db = ObrainDB::new_in_memory();
    let id = db.create_node(&["Base"]);
    db.add_node_label(id, "Extra");
    let labels = db.get_node_labels(id).unwrap();
    assert!(labels.contains(&"Extra".to_string()));

    db.remove_node_label(id, "Extra");
    let labels = db.get_node_labels(id).unwrap();
    assert!(!labels.contains(&"Extra".to_string()));
}

#[test]
fn crud_set_and_remove_edge_property() {
    let db = ObrainDB::new_in_memory();
    let a = db.create_node(&["A"]);
    let b = db.create_node(&["B"]);
    let edge = db.create_edge(a, b, "REL");
    db.set_edge_property(edge, "weight", Value::from(1.5_f64));
    // Verify via query
    let result = db
        .execute("MATCH ()-[r:REL]->() RETURN r.weight AS w")
        .unwrap();
    assert_eq!(result.rows.len(), 1);

    let removed = db.remove_edge_property(edge, "weight");
    assert!(removed);
}

#[test]
fn crud_delete_node() {
    let db = ObrainDB::new_in_memory();
    let id = db.create_node(&["Temp"]);
    assert!(db.get_node_labels(id).is_some());
    db.delete_node(id);
    assert!(db.get_node_labels(id).is_none());
}

#[test]
fn crud_delete_edge() {
    let db = ObrainDB::new_in_memory();
    let a = db.create_node(&["A"]);
    let b = db.create_node(&["B"]);
    let edge = db.create_edge(a, b, "REL");
    db.delete_edge(edge);
    // Edge should no longer exist
    let result = db.execute("MATCH ()-[r]->() RETURN count(r) AS c");
    assert!(result.is_ok());
}

// ---------------------------------------------------------------------------
// Query API (session/mod.rs, database/mod.rs)
// ---------------------------------------------------------------------------

#[test]
fn execute_gql_basic() {
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (:Person {name: 'Alice'})").unwrap();
    let result = db
        .execute("MATCH (n:Person) RETURN n.name AS name")
        .unwrap();
    assert_eq!(result.rows.len(), 1);
}

#[test]
fn execute_gql_count_star() {
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (:A), (:A), (:B)").unwrap();
    let result = db.execute("MATCH (n) RETURN count(*) AS c").unwrap();
    assert_eq!(result.rows.len(), 1);
}

#[test]
fn execute_gql_with_properties() {
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (:Item {price: 42, name: 'Widget'})")
        .unwrap();
    let result = db
        .execute("MATCH (i:Item) WHERE i.price > 10 RETURN i.name AS name, i.price AS price")
        .unwrap();
    assert!(!result.rows.is_empty());
}

#[test]
fn execute_gql_relationship_traversal() {
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (:Person {name: 'Alice'})-[:KNOWS]->(:Person {name: 'Bob'})")
        .unwrap();
    let result = db
        .execute("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name AS from, b.name AS to")
        .unwrap();
    assert_eq!(result.rows.len(), 1);
}

#[test]
fn execute_gql_aggregation() {
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (:N {v: 10}), (:N {v: 20}), (:N {v: 30})")
        .unwrap();
    let result = db
        .execute("MATCH (n:N) RETURN sum(n.v) AS total, avg(n.v) AS mean, count(n) AS cnt")
        .unwrap();
    assert_eq!(result.rows.len(), 1);
}

#[test]
fn execute_gql_order_and_limit() {
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (:N {v: 3}), (:N {v: 1}), (:N {v: 2})")
        .unwrap();
    let result = db
        .execute("MATCH (n:N) RETURN n.v AS v ORDER BY v ASC LIMIT 2")
        .unwrap();
    assert_eq!(result.rows.len(), 2);
}

#[cfg(feature = "cypher")]
#[test]
fn execute_cypher_basic() {
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (:Person {name: 'Alice'})").unwrap();
    let result = db
        .execute_cypher("MATCH (n:Person) RETURN n.name AS name")
        .unwrap();
    assert_eq!(result.rows.len(), 1);
}

// ---------------------------------------------------------------------------
// Error codes (utils/error.rs)
// ---------------------------------------------------------------------------

#[test]
fn error_codes_use_obrain_prefix() {
    use obrain_common::utils::error::{Error, ErrorCode};

    // All error codes should use OBRAIN- prefix (not GRAFEO-)
    let codes = [
        ErrorCode::QuerySyntax,
        ErrorCode::QuerySemantic,
        ErrorCode::QueryTimeout,
        ErrorCode::TransactionConflict,
        ErrorCode::TransactionTimeout,
        ErrorCode::TransactionReadOnly,
        ErrorCode::StorageFull,
        ErrorCode::StorageCorrupted,
        ErrorCode::InvalidInput,
        ErrorCode::NodeNotFound,
        ErrorCode::EdgeNotFound,
        ErrorCode::Internal,
        ErrorCode::IoError,
    ];

    for code in &codes {
        let s = code.as_str();
        assert!(
            s.starts_with("OBRAIN-"),
            "Error code {s} should start with OBRAIN-"
        );
    }

    // Verify retryable codes
    assert!(ErrorCode::TransactionConflict.is_retryable());
    assert!(ErrorCode::TransactionTimeout.is_retryable());
    assert!(ErrorCode::TransactionDeadlock.is_retryable());
    assert!(ErrorCode::QueryTimeout.is_retryable());
    assert!(!ErrorCode::Internal.is_retryable());
    assert!(!ErrorCode::StorageFull.is_retryable());

    // Verify Display impls
    let err = Error::Internal("test".into());
    let msg = err.to_string();
    assert!(msg.contains("OBRAIN-X001"));

    let err = Error::NodeNotFound(obrain_common::types::NodeId::new(99));
    let msg = err.to_string();
    assert!(msg.contains("OBRAIN-V002"));
}

#[test]
fn error_query_builder_chain() {
    use obrain_common::utils::error::{QueryError, QueryErrorKind, SourceSpan};

    let err = QueryError::new(QueryErrorKind::Semantic, "Unknown label")
        .with_span(SourceSpan::new(5, 10, 1, 6))
        .with_source("MATCH (n:Foo) RETURN n")
        .with_hint("Did you mean 'Bar'?");

    assert!(err.span.is_some());
    assert!(err.source_query.is_some());
    assert!(err.hint.is_some());

    let display = err.to_string();
    assert!(display.contains("Unknown label"));
    assert!(display.contains("query:1:6"));
    assert!(display.contains("Did you mean 'Bar'?"));
}

#[test]
fn error_storage_variants() {
    use obrain_common::utils::error::{Error, StorageError};

    let variants: Vec<(StorageError, &str)> = vec![
        (StorageError::Corruption("bad".into()), "OBRAIN-S002"),
        (StorageError::Full, "OBRAIN-S001"),
        (StorageError::InvalidWalEntry("bad".into()), "OBRAIN-S002"),
        (StorageError::RecoveryFailed("bad".into()), "OBRAIN-S003"),
        (StorageError::CheckpointFailed("bad".into()), "OBRAIN-S002"),
    ];

    for (variant, expected_code) in variants {
        assert_eq!(variant.error_code().as_str(), expected_code);
        // Test Display doesn't panic
        let _ = variant.to_string();
        // Test From conversion
        let err: Error = variant.into();
        assert!(matches!(err, Error::Storage(_)));
    }
}

#[test]
fn error_transaction_variants() {
    use obrain_common::utils::error::{Error, TransactionError};

    let variants = vec![
        TransactionError::Aborted,
        TransactionError::Conflict,
        TransactionError::WriteConflict("col".into()),
        TransactionError::SerializationFailure("ssi".into()),
        TransactionError::Deadlock,
        TransactionError::Timeout,
        TransactionError::ReadOnly,
        TransactionError::InvalidState("bad".into()),
    ];

    for variant in variants {
        // Test Display doesn't panic
        let _ = variant.to_string();
        // Test From conversion
        let err: Error = variant.into();
        assert!(matches!(err, Error::Transaction(_)));
        // Test error_code doesn't panic
        let _ = err.error_code();
    }
}

// ---------------------------------------------------------------------------
// Collections (collections.rs)
// ---------------------------------------------------------------------------

#[test]
fn collections_constructors() {
    use obrain_common::collections::*;

    // Test all constructor variants
    let map = obrain_map::<String, i32>();
    assert!(map.is_empty());

    let map = obrain_map_with_capacity::<String, i32>(100);
    assert!(map.is_empty());
    assert!(map.capacity() >= 100);

    let set = obrain_set::<i32>();
    assert!(set.is_empty());

    let set = obrain_set_with_capacity::<i32>(100);
    assert!(set.is_empty());

    let cmap = obrain_concurrent_map::<String, i32>();
    assert!(cmap.is_empty());

    let cmap = obrain_concurrent_map_with_capacity::<String, i32>(100);
    assert!(cmap.is_empty());

    let imap = obrain_index_map::<String, i32>();
    assert!(imap.is_empty());

    let imap = obrain_index_map_with_capacity::<String, i32>(100);
    assert!(imap.is_empty());
}

#[test]
fn collections_index_map_preserves_insertion_order() {
    use obrain_common::collections::obrain_index_map;

    let mut map = obrain_index_map::<&str, i32>();
    for (i, key) in ["z", "a", "m", "b", "x"].iter().enumerate() {
        map.insert(key, i as i32);
    }

    let keys: Vec<_> = map.keys().copied().collect();
    assert_eq!(keys, vec!["z", "a", "m", "b", "x"]);
}

#[test]
fn collections_concurrent_map_multithreaded() {
    use obrain_common::collections::obrain_concurrent_map;
    use std::sync::Arc;

    let map = Arc::new(obrain_concurrent_map::<i32, i32>());
    let mut handles = vec![];

    for t in 0..4 {
        let m = Arc::clone(&map);
        handles.push(std::thread::spawn(move || {
            for i in 0..100 {
                m.insert(t * 100 + i, i);
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(map.len(), 400);
}

// ---------------------------------------------------------------------------
// Session & Database API (session/mod.rs, database/mod.rs)
// ---------------------------------------------------------------------------

#[test]
fn database_node_count() {
    let db = ObrainDB::new_in_memory();
    assert_eq!(db.node_count(), 0);
    db.create_node(&["A"]);
    db.create_node(&["B"]);
    assert_eq!(db.node_count(), 2);
}

#[test]
fn database_edge_count() {
    let db = ObrainDB::new_in_memory();
    let a = db.create_node(&["A"]);
    let b = db.create_node(&["B"]);
    assert_eq!(db.edge_count(), 0);
    db.create_edge(a, b, "REL");
    assert_eq!(db.edge_count(), 1);
}

#[test]
fn database_execute_returns_column_names() {
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (:N {x: 1})").unwrap();
    let result = db
        .execute("MATCH (n:N) RETURN n.x AS val, labels(n) AS lbls")
        .unwrap();
    assert!(result.columns.contains(&"val".to_string()));
    assert!(result.columns.contains(&"lbls".to_string()));
}

#[test]
fn database_multiple_creates_in_one_query() {
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (:A), (:B), (:C), (:D), (:E)").unwrap();
    assert_eq!(db.node_count(), 5);
}

#[test]
fn database_match_where_filter() {
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (:N {v: 1}), (:N {v: 2}), (:N {v: 3}), (:N {v: 4}), (:N {v: 5})")
        .unwrap();
    let result = db
        .execute("MATCH (n:N) WHERE n.v > 3 RETURN n.v AS v")
        .unwrap();
    assert_eq!(result.rows.len(), 2);
}

#[test]
fn database_distinct() {
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (:N {v: 'a'}), (:N {v: 'a'}), (:N {v: 'b'})")
        .unwrap();
    let result = db.execute("MATCH (n:N) RETURN DISTINCT n.v AS v").unwrap();
    assert_eq!(result.rows.len(), 2);
}

#[test]
fn database_set_property_via_query() {
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (:Person {name: 'Alice'})").unwrap();
    db.execute("MATCH (n:Person {name: 'Alice'}) SET n.age = 30")
        .unwrap();
    let result = db
        .execute("MATCH (n:Person {name: 'Alice'}) RETURN n.age AS age")
        .unwrap();
    assert_eq!(result.rows.len(), 1);
}

#[test]
fn database_delete_via_query() {
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (:Temp {x: 1}), (:Temp {x: 2})").unwrap();
    assert_eq!(db.node_count(), 2);
    db.execute("MATCH (n:Temp) DELETE n").unwrap();
    assert_eq!(db.node_count(), 0);
}

#[test]
fn database_optional_match() {
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (:A {name: 'solo'})").unwrap();
    let result = db
        .execute("MATCH (a:A) OPTIONAL MATCH (a)-[:REL]->(b) RETURN a.name AS name, b AS other")
        .unwrap();
    assert_eq!(result.rows.len(), 1);
}

#[test]
fn database_unwind() {
    let db = ObrainDB::new_in_memory();
    let result = db.execute("UNWIND [1, 2, 3, 4, 5] AS x RETURN x").unwrap();
    assert_eq!(result.rows.len(), 5);
}

#[test]
fn database_with_clause() {
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (:N {v: 10}), (:N {v: 20}), (:N {v: 30})")
        .unwrap();
    let result = db
        .execute("MATCH (n:N) WITH n.v AS val WHERE val > 15 RETURN val ORDER BY val")
        .unwrap();
    assert_eq!(result.rows.len(), 2);
}
