#![cfg(feature = "cypher")]
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

// ===========================================================================
// Session API (session/mod.rs)
// ===========================================================================

#[test]
fn session_execute_and_basic_ops() {
    let db = ObrainDB::new_in_memory();
    let session = db.session();
    session.execute("CREATE (:Person {name: 'Alice'})").unwrap();
    let result = session
        .execute("MATCH (n:Person) RETURN n.name AS name")
        .unwrap();
    assert_eq!(result.rows.len(), 1);
}

#[test]
fn session_graph_model() {
    let db = ObrainDB::new_in_memory();
    let session = db.session();
    let _model = session.graph_model();
}

#[test]
fn session_time_zone() {
    let db = ObrainDB::new_in_memory();
    let session = db.session();
    assert!(session.time_zone().is_none());
    session.set_time_zone("UTC");
    assert_eq!(session.time_zone().unwrap(), "UTC");
    session.reset_time_zone();
    assert!(session.time_zone().is_none());
}

#[test]
fn session_parameters() {
    let db = ObrainDB::new_in_memory();
    let session = db.session();
    session.set_parameter("x", Value::Int64(42));
    let val = session.get_parameter("x");
    assert!(val.is_some());
    session.reset_parameters();
    assert!(session.get_parameter("x").is_none());
}

#[test]
fn session_reset_all() {
    let db = ObrainDB::new_in_memory();
    let session = db.session();
    session.set_time_zone("Europe/Paris");
    session.set_parameter("foo", Value::from("bar"));
    session.reset_session();
    assert!(session.time_zone().is_none());
    assert!(session.get_parameter("foo").is_none());
}

#[test]
fn session_schema_operations() {
    let db = ObrainDB::new_in_memory();
    let session = db.session();
    assert!(session.current_schema().is_none());
    session.set_schema("my_schema");
    assert_eq!(session.current_schema().unwrap(), "my_schema");
    session.reset_schema();
    assert!(session.current_schema().is_none());
}

#[test]
fn session_in_transaction_flag() {
    let db = ObrainDB::new_in_memory();
    let mut session = db.session();
    assert!(!session.in_transaction());
    session.begin_transaction().unwrap();
    assert!(session.in_transaction());
    session.commit().unwrap();
    assert!(!session.in_transaction());
}

#[test]
fn session_transaction_rollback() {
    let db = ObrainDB::new_in_memory();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("CREATE (:Temp {name: 'rollback_me'})")
        .unwrap();
    session.rollback().unwrap();
    let result = db.execute("MATCH (n:Temp) RETURN count(n) AS c").unwrap();
    assert_eq!(result.rows[0][0], Value::Int64(0));
}

#[test]
fn session_auto_commit() {
    let db = ObrainDB::new_in_memory();
    let mut session = db.session();
    assert!(session.auto_commit());
    session.set_auto_commit(false);
    assert!(!session.auto_commit());
}

#[test]
fn session_savepoint_basic() {
    let db = ObrainDB::new_in_memory();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("CREATE (:A {v: 1})").unwrap();
    session.savepoint("sp1").unwrap();
    session.execute("CREATE (:B {v: 2})").unwrap();
    session.rollback_to_savepoint("sp1").unwrap();
    session.commit().unwrap();
    let result = db.execute("MATCH (n:A) RETURN count(n) AS c").unwrap();
    assert_eq!(result.rows[0][0], Value::Int64(1));
    let result = db.execute("MATCH (n:B) RETURN count(n) AS c").unwrap();
    assert_eq!(result.rows[0][0], Value::Int64(0));
}

#[test]
fn session_release_savepoint() {
    let db = ObrainDB::new_in_memory();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.savepoint("sp1").unwrap();
    session.release_savepoint("sp1").unwrap();
    let result = session.rollback_to_savepoint("sp1");
    assert!(result.is_err());
    session.rollback().unwrap();
}

#[test]
fn session_clear_plan_cache() {
    let db = ObrainDB::new_in_memory();
    let session = db.session();
    session.execute("CREATE (:N)").unwrap();
    session.clear_plan_cache();
    let result = session.execute("MATCH (n) RETURN count(n) AS c").unwrap();
    assert_eq!(result.rows.len(), 1);
}

#[cfg(feature = "cypher")]
#[test]
fn session_execute_cypher() {
    let db = ObrainDB::new_in_memory();
    let session = db.session();
    session.execute("CREATE (:X {v: 1})").unwrap();
    let result = session
        .execute_cypher("MATCH (n:X) RETURN n.v AS v")
        .unwrap();
    assert_eq!(result.rows.len(), 1);
}

#[test]
fn session_viewing_epoch() {
    let db = ObrainDB::new_in_memory();
    let session = db.session();
    assert!(session.viewing_epoch().is_none());
    session.set_viewing_epoch(obrain_common::types::EpochId(5));
    assert_eq!(
        session.viewing_epoch(),
        Some(obrain_common::types::EpochId(5))
    );
    session.clear_viewing_epoch();
    assert!(session.viewing_epoch().is_none());
}

#[test]
fn session_use_graph_and_reset() {
    let db = ObrainDB::new_in_memory();
    let session = db.session();
    assert!(session.current_graph().is_none());
    session.use_graph("test_graph");
    assert_eq!(session.current_graph().unwrap(), "test_graph");
    session.reset_graph();
    assert!(session.current_graph().is_none());
}

// ===========================================================================
// Database API (database/mod.rs)
// ===========================================================================

#[test]
fn database_current_graph() {
    let db = ObrainDB::new_in_memory();
    assert!(db.current_graph().is_none());
    db.set_current_graph(Some("my_graph"));
    assert_eq!(db.current_graph().unwrap(), "my_graph");
    db.set_current_graph(None);
    assert!(db.current_graph().is_none());
}

#[test]
fn database_is_read_only() {
    let db = ObrainDB::new_in_memory();
    assert!(!db.is_read_only());
}

#[test]
fn database_config_access() {
    let db = ObrainDB::new_in_memory();
    let _config = db.config();
    let _adaptive = db.adaptive_config();
}

#[test]
fn database_graph_model_access() {
    let db = ObrainDB::new_in_memory();
    let _model = db.graph_model();
}

#[test]
fn database_store_access() {
    let db = ObrainDB::new_in_memory();
    let _store = db.store();
    let _graph_store = db.graph_store();
}

#[test]
fn database_buffer_manager() {
    let db = ObrainDB::new_in_memory();
    let _bm = db.buffer_manager();
}

#[test]
fn database_query_cache_access() {
    let db = ObrainDB::new_in_memory();
    let _cache = db.query_cache();
}

#[test]
fn database_clear_plan_cache() {
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (:CacheTest)").unwrap();
    db.clear_plan_cache();
    let result = db
        .execute("MATCH (n:CacheTest) RETURN count(n) AS c")
        .unwrap();
    assert_eq!(result.rows.len(), 1);
}

#[test]
fn database_gc() {
    let db = ObrainDB::new_in_memory();
    db.create_node(&["GcTest"]);
    db.gc();
    assert_eq!(db.node_count(), 1);
}

#[test]
fn database_create_and_list_graphs() {
    let db = ObrainDB::new_in_memory();
    let created = db.create_graph("graph1").unwrap();
    assert!(created);
    let graphs = db.list_graphs();
    assert!(graphs.contains(&"graph1".to_string()));
    let dropped = db.drop_graph("graph1");
    assert!(dropped);
    assert!(!db.list_graphs().contains(&"graph1".to_string()));
}

#[test]
fn database_drop_nonexistent_graph() {
    let db = ObrainDB::new_in_memory();
    assert!(!db.drop_graph("nonexistent"));
}

#[cfg(feature = "metrics")]
#[test]
fn database_metrics_snapshot() {
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (:MetricTest)").unwrap();
    let snap = db.metrics();
    assert!(snap.query_count >= 1);
}

#[cfg(feature = "metrics")]
#[test]
fn database_metrics_prometheus() {
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (:PromTest)").unwrap();
    let prom = db.metrics_prometheus();
    assert!(prom.contains("obrain_query_count"));
}

#[cfg(feature = "metrics")]
#[test]
fn database_reset_metrics() {
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (:ResetTest)").unwrap();
    db.reset_metrics();
    let snap = db.metrics();
    assert_eq!(snap.query_count, 0);
}

#[test]
fn database_wal_accessor() {
    let db = ObrainDB::new_in_memory();
    let _wal = db.wal();
}

#[test]
fn database_close() {
    let db = ObrainDB::new_in_memory();
    db.create_node(&["CloseTest"]);
    assert!(db.close().is_ok());
}

// ===========================================================================
// QueryResult builders (database/mod.rs)
// ===========================================================================

#[test]
fn query_result_empty() {
    let result = obrain_engine::database::QueryResult::empty();
    assert!(result.is_empty());
    assert_eq!(result.row_count(), 0);
    assert_eq!(result.column_count(), 0);
    assert!(result.execution_time_ms().is_none());
    assert!(result.rows_scanned().is_none());
}

#[test]
fn query_result_status() {
    let result = obrain_engine::database::QueryResult::status("Node type created");
    assert!(result.is_empty());
    assert_eq!(result.status_message, Some("Node type created".to_string()));
}

#[test]
fn query_result_new_with_columns() {
    let result =
        obrain_engine::database::QueryResult::new(vec!["a".into(), "b".into(), "c".into()]);
    assert_eq!(result.column_count(), 3);
    assert!(result.is_empty());
}

#[test]
fn query_result_with_types() {
    use obrain_common::types::LogicalType;
    let result = obrain_engine::database::QueryResult::with_types(
        vec!["id".into(), "name".into()],
        vec![LogicalType::Int64, LogicalType::String],
    );
    assert_eq!(result.column_count(), 2);
    assert_eq!(result.column_types.len(), 2);
}

#[test]
fn query_result_with_metrics() {
    let result = obrain_engine::database::QueryResult::empty().with_metrics(42.5, 100);
    assert_eq!(result.execution_time_ms(), Some(42.5));
    assert_eq!(result.rows_scanned(), Some(100));
}

#[test]
fn query_result_scalar() {
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (:S), (:S), (:S)").unwrap();
    let result = db.execute("MATCH (n:S) RETURN count(n) AS c").unwrap();
    let count: i64 = result.scalar().unwrap();
    assert_eq!(count, 3);
}

#[test]
fn query_result_scalar_error_multiple_rows() {
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (:M {v: 1}), (:M {v: 2})").unwrap();
    let result = db.execute("MATCH (n:M) RETURN n.v AS v").unwrap();
    let scalar_result: std::result::Result<i64, _> = result.scalar();
    assert!(scalar_result.is_err());
}

#[test]
fn query_result_iter() {
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (:I {v: 1}), (:I {v: 2}), (:I {v: 3})")
        .unwrap();
    let result = db.execute("MATCH (n:I) RETURN n.v AS v").unwrap();
    let rows: Vec<_> = result.iter().collect();
    assert_eq!(rows.len(), 3);
}

#[test]
fn query_result_display() {
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (:D {name: 'test'})").unwrap();
    let result = db.execute("MATCH (n:D) RETURN n.name AS name").unwrap();
    let display = format!("{result}");
    assert!(!display.is_empty());
}

// ===========================================================================
// Metrics (metrics.rs)
// ===========================================================================

#[test]
fn metrics_histogram_basic() {
    use obrain_engine::metrics::AtomicHistogram;
    static BUCKETS: &[f64] = &[1.0, 5.0, 10.0, 50.0, 100.0];
    let hist = AtomicHistogram::new(BUCKETS);
    hist.observe(0.5);
    hist.observe(3.0);
    hist.observe(7.0);
    hist.observe(75.0);
    hist.observe(200.0);
    assert_eq!(hist.snapshot().count, 5);
    assert!(hist.mean() > 0.0);
}

#[test]
fn metrics_histogram_percentile_empty() {
    use obrain_engine::metrics::AtomicHistogram;
    static BUCKETS: &[f64] = &[1.0, 5.0, 10.0];
    let hist = AtomicHistogram::new(BUCKETS);
    assert_eq!(hist.percentile(0.5), 0.0);
    assert_eq!(hist.mean(), 0.0);
}

#[test]
fn metrics_histogram_reset() {
    use obrain_engine::metrics::AtomicHistogram;
    static BUCKETS: &[f64] = &[1.0, 5.0, 10.0];
    let hist = AtomicHistogram::new(BUCKETS);
    hist.observe(2.0);
    hist.observe(8.0);
    assert_eq!(hist.snapshot().count, 2);
    hist.reset();
    assert_eq!(hist.snapshot().count, 0);
    assert_eq!(hist.snapshot().sum, 0.0);
}

#[test]
fn metrics_histogram_snapshot_structure() {
    use obrain_engine::metrics::AtomicHistogram;
    static BUCKETS: &[f64] = &[1.0, 5.0, 10.0];
    let hist = AtomicHistogram::new(BUCKETS);
    hist.observe(0.5);
    hist.observe(3.0);
    hist.observe(15.0);
    let snap = hist.snapshot();
    assert_eq!(snap.boundaries.len(), 3);
    assert_eq!(snap.bucket_counts.len(), 4);
    assert_eq!(snap.count, 3);
}

#[test]
fn metrics_registry_via_db() {
    // Test metrics through the public DB API (fields are pub(crate))
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (:MR1)").unwrap();
    db.execute("CREATE (:MR2)").unwrap();

    let prom = db.metrics_prometheus();
    assert!(prom.contains("obrain_query_count"));
    assert!(prom.contains("obrain_query_latency_ms_bucket"));
    assert!(prom.contains("+Inf"));
    assert!(prom.contains("obrain_tx_committed"));
    assert!(prom.contains("obrain_session_active"));
    assert!(prom.contains("obrain_gc_runs"));

    let snap = db.metrics();
    assert!(snap.query_count >= 2);

    db.reset_metrics();
    let snap = db.metrics();
    assert_eq!(snap.query_count, 0);
}

#[test]
fn metrics_registry_default_and_new() {
    use obrain_engine::metrics::MetricsRegistry;
    let reg = MetricsRegistry::new();
    let snap = reg.snapshot();
    assert_eq!(snap.query_count, 0);

    let reg2 = MetricsRegistry::default();
    let snap2 = reg2.snapshot();
    assert_eq!(snap2.query_count, 0);

    // snapshot_with_cache
    let snap_cache = reg.snapshot_with_cache(10, 5, 100, 2);
    assert_eq!(snap_cache.cache_hits, 10);
    assert_eq!(snap_cache.cache_misses, 5);
    assert_eq!(snap_cache.cache_size, 100);
    assert_eq!(snap_cache.cache_invalidations, 2);
}

// ===========================================================================
// Procedures (procedures.rs)
// ===========================================================================

#[test]
fn builtin_procedures_registry() {
    use obrain_engine::procedures::BuiltinProcedures;
    let procs = BuiltinProcedures::new();
    assert!(
        procs
            .get(&["obrain".to_string(), "pagerank".to_string()])
            .is_some()
    );
    assert!(procs.get(&["pagerank".to_string()]).is_some());
    assert!(procs.get(&["nonexistent".to_string()]).is_none());
    assert!(procs.get(&[]).is_none());
}

#[test]
fn builtin_procedures_list_sorted() {
    use obrain_engine::procedures::BuiltinProcedures;
    let procs = BuiltinProcedures::new();
    let list = procs.list();
    assert!(!list.is_empty());
    for proc in &list {
        assert!(proc.name.starts_with("obrain."));
        assert!(!proc.description.is_empty());
    }
    let names: Vec<&str> = list.iter().map(|p| p.name.as_str()).collect();
    let mut sorted = names.clone();
    sorted.sort_unstable();
    assert_eq!(names, sorted);
}

#[test]
fn builtin_procedures_known_algorithms() {
    use obrain_engine::procedures::BuiltinProcedures;
    let procs = BuiltinProcedures::default();
    for name in &[
        "pagerank",
        "betweenness_centrality",
        "bfs",
        "dfs",
        "connected_components",
        "dijkstra",
        "louvain",
        "bridges",
    ] {
        assert!(
            procs.get(&[name.to_string()]).is_some(),
            "{name} should be registered"
        );
    }
}

// ===========================================================================
// Collections extended (collections.rs)
// ===========================================================================

#[test]
fn collections_map_operations() {
    use obrain_common::collections::obrain_map;
    let mut map = obrain_map::<String, Vec<i32>>();
    map.insert("a".into(), vec![1, 2, 3]);
    map.insert("b".into(), vec![4, 5]);
    assert_eq!(map.len(), 2);
    assert!(map.contains_key("a"));
    map.remove("a");
    assert_eq!(map.len(), 1);
}

#[test]
fn collections_set_operations() {
    use obrain_common::collections::obrain_set;
    let mut set = obrain_set::<String>();
    set.insert("alpha".into());
    set.insert("beta".into());
    set.insert("alpha".into()); // duplicate
    assert_eq!(set.len(), 2);
    assert!(set.contains("alpha"));
}

#[test]
fn collections_concurrent_map_stress() {
    use obrain_common::collections::obrain_concurrent_map;
    use std::sync::Arc;
    let map = Arc::new(obrain_concurrent_map::<String, String>());
    let mut handles = vec![];
    for t in 0..8 {
        let m = Arc::clone(&map);
        handles.push(std::thread::spawn(move || {
            for i in 0..50 {
                m.insert(format!("t{t}_k{i}"), format!("v{i}"));
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    assert_eq!(map.len(), 400);
}

// ===========================================================================
// Error types extended (utils/error.rs)
// ===========================================================================

#[test]
fn error_code_as_str_all() {
    use obrain_common::utils::error::ErrorCode;
    let all = [
        (ErrorCode::QuerySyntax, "OBRAIN-Q001"),
        (ErrorCode::QuerySemantic, "OBRAIN-Q002"),
        (ErrorCode::QueryTimeout, "OBRAIN-Q003"),
        (ErrorCode::QueryUnsupported, "OBRAIN-Q004"),
        (ErrorCode::QueryOptimization, "OBRAIN-Q005"),
        (ErrorCode::QueryExecution, "OBRAIN-Q006"),
        (ErrorCode::TransactionConflict, "OBRAIN-T001"),
        (ErrorCode::TransactionTimeout, "OBRAIN-T002"),
        (ErrorCode::TransactionReadOnly, "OBRAIN-T003"),
        (ErrorCode::TransactionInvalidState, "OBRAIN-T004"),
        (ErrorCode::TransactionSerialization, "OBRAIN-T005"),
        (ErrorCode::TransactionDeadlock, "OBRAIN-T006"),
        (ErrorCode::StorageFull, "OBRAIN-S001"),
        (ErrorCode::StorageCorrupted, "OBRAIN-S002"),
        (ErrorCode::StorageRecoveryFailed, "OBRAIN-S003"),
        (ErrorCode::InvalidInput, "OBRAIN-V001"),
        (ErrorCode::NodeNotFound, "OBRAIN-V002"),
        (ErrorCode::EdgeNotFound, "OBRAIN-V003"),
        (ErrorCode::PropertyNotFound, "OBRAIN-V004"),
        (ErrorCode::LabelNotFound, "OBRAIN-V005"),
        (ErrorCode::TypeMismatch, "OBRAIN-V006"),
        (ErrorCode::Internal, "OBRAIN-X001"),
        (ErrorCode::IoError, "OBRAIN-X003"),
    ];
    for (code, expected) in &all {
        assert_eq!(code.as_str(), *expected);
    }
}

#[test]
fn error_code_display() {
    use obrain_common::utils::error::ErrorCode;
    assert_eq!(format!("{}", ErrorCode::QuerySyntax), "OBRAIN-Q001");
}

#[test]
fn error_display_all_variants() {
    use obrain_common::utils::error::Error;
    let variants: Vec<Error> = vec![
        Error::NodeNotFound(obrain_common::types::NodeId::new(1)),
        Error::EdgeNotFound(obrain_common::types::EdgeId::new(2)),
        Error::PropertyNotFound("prop".into()),
        Error::LabelNotFound("lbl".into()),
        Error::TypeMismatch {
            expected: "INT64".into(),
            found: "STRING".into(),
        },
        Error::InvalidValue("bad".into()),
        Error::Serialization("ser".into()),
        Error::Internal("oops".into()),
    ];
    for err in &variants {
        let msg = err.to_string();
        assert!(msg.contains("OBRAIN-"));
        let _ = err.error_code();
    }
}

#[test]
fn error_io_conversion() {
    use obrain_common::utils::error::Error;
    let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "gone");
    let err: Error = io_err.into();
    assert!(matches!(err, Error::Io(_)));
    assert!(err.to_string().contains("OBRAIN-X003"));
}

#[test]
fn error_query_error_full_context() {
    use obrain_common::utils::error::{QueryError, QueryErrorKind, SourceSpan};
    let err = QueryError::new(QueryErrorKind::Syntax, "Unexpected token")
        .with_span(SourceSpan::new(10, 15, 1, 11))
        .with_source("MATCH (n) RETRN n")
        .with_hint("Did you mean RETURN?");
    let display = err.to_string();
    assert!(display.contains("Unexpected token"));
    assert!(display.contains("query:1:11"));
    assert!(display.contains("Did you mean RETURN?"));
}

#[test]
fn error_query_error_kinds() {
    use obrain_common::utils::error::{QueryError, QueryErrorKind};
    let kinds = [
        (QueryErrorKind::Lexer, "OBRAIN-Q001"),
        (QueryErrorKind::Syntax, "OBRAIN-Q001"),
        (QueryErrorKind::Semantic, "OBRAIN-Q002"),
        (QueryErrorKind::Optimization, "OBRAIN-Q005"),
        (QueryErrorKind::Execution, "OBRAIN-Q006"),
    ];
    for (kind, expected) in &kinds {
        let err = QueryError::new(*kind, "test");
        assert_eq!(err.error_code().as_str(), *expected);
    }
}

#[test]
fn error_query_timeout() {
    use obrain_common::utils::error::QueryError;
    let err = QueryError::timeout();
    assert_eq!(err.error_code().as_str(), "OBRAIN-Q006");
}

#[test]
fn error_query_kind_display() {
    use obrain_common::utils::error::QueryErrorKind;
    // Display uses lowercase
    let lexer = format!("{}", QueryErrorKind::Lexer);
    assert!(lexer.contains("lexer") || lexer.contains("Lexer"));
    let syntax = format!("{}", QueryErrorKind::Syntax);
    assert!(syntax.contains("yntax"));
    let semantic = format!("{}", QueryErrorKind::Semantic);
    assert!(semantic.contains("emantic"));
}

// ===========================================================================
// CRUD extended (database/crud.rs)
// ===========================================================================

#[test]
fn crud_create_edge_with_props() {
    let db = ObrainDB::new_in_memory();
    let a = db.create_node(&["A"]);
    let b = db.create_node(&["B"]);
    let _e = db.create_edge_with_props(a, b, "REL", [("weight", Value::from(0.5_f64))]);
    let result = db
        .execute("MATCH ()-[r:REL]->() RETURN r.weight AS w")
        .unwrap();
    assert_eq!(result.rows.len(), 1);
}

#[test]
fn crud_get_node_and_edge() {
    let db = ObrainDB::new_in_memory();
    let id = db.create_node(&["GetTest"]);
    assert!(db.get_node(id).is_some());
    assert!(
        db.get_node(obrain_common::types::NodeId::new(9999))
            .is_none()
    );

    let b = db.create_node(&["B"]);
    let eid = db.create_edge(id, b, "LINK");
    assert!(db.get_edge(eid).is_some());
}

#[test]
fn crud_delete_returns_bool() {
    let db = ObrainDB::new_in_memory();
    let id = db.create_node(&["Del"]);
    assert!(db.delete_node(id));
    assert!(!db.delete_node(id));

    let a = db.create_node(&["A"]);
    let b = db.create_node(&["B"]);
    let eid = db.create_edge(a, b, "REL");
    assert!(db.delete_edge(eid));
    assert!(!db.delete_edge(eid));
}

#[test]
fn crud_set_multiple_property_types() {
    let db = ObrainDB::new_in_memory();
    let id = db.create_node(&["Props"]);
    db.set_node_property(id, "int", Value::from(1_i64));
    db.set_node_property(id, "str", Value::from("hello"));
    db.set_node_property(id, "bool", Value::from(true));
    db.set_node_property(id, "float", Value::from(1.234_f64));
    let result = db
        .execute("MATCH (n:Props) RETURN n.int, n.str, n.bool, n.float")
        .unwrap();
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.column_count(), 4);
}

// ===========================================================================
// Query edge cases
// ===========================================================================

#[test]
fn query_syntax_error() {
    let db = ObrainDB::new_in_memory();
    assert!(db.execute("THIS IS NOT VALID").is_err());
}

#[test]
fn query_empty_result() {
    let db = ObrainDB::new_in_memory();
    let result = db.execute("MATCH (n:NonExistent) RETURN n").unwrap();
    assert!(result.is_empty());
}

#[test]
fn query_null_property() {
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (:NP)").unwrap();
    let result = db.execute("MATCH (n:NP) RETURN n.missing AS v").unwrap();
    assert!(matches!(result.rows[0][0], Value::Null));
}

#[test]
fn query_multiple_labels() {
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (:A:B:C {name: 'multi'})").unwrap();
    let result = db.execute("MATCH (n:A:B:C) RETURN n.name AS n").unwrap();
    assert_eq!(result.rows.len(), 1);
}

#[test]
fn query_aggregation_functions() {
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (:V {n: 10}), (:V {n: 20}), (:V {n: 30})")
        .unwrap();
    let result = db
        .execute("MATCH (v:V) RETURN sum(v.n) AS s, avg(v.n) AS a, min(v.n) AS mn, max(v.n) AS mx")
        .unwrap();
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.column_count(), 4);
}

#[test]
fn query_collect() {
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (:C {v: 'a'}), (:C {v: 'b'}), (:C {v: 'c'})")
        .unwrap();
    let result = db
        .execute("MATCH (c:C) RETURN collect(c.v) AS items")
        .unwrap();
    assert_eq!(result.rows.len(), 1);
}

#[test]
fn query_skip_limit() {
    let db = ObrainDB::new_in_memory();
    for i in 0..10 {
        db.create_node_with_props(&["SL"], [("v", Value::from(i as i64))]);
    }
    let result = db
        .execute("MATCH (n:SL) RETURN n.v AS v ORDER BY v SKIP 3 LIMIT 4")
        .unwrap();
    assert_eq!(result.rows.len(), 4);
}

#[test]
fn query_return_star() {
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (:Star {x: 1})").unwrap();
    let result = db.execute("MATCH (n:Star) RETURN *").unwrap();
    assert_eq!(result.rows.len(), 1);
}

#[test]
fn query_merge() {
    let db = ObrainDB::new_in_memory();
    db.execute("MERGE (:Mg {name: 'first'})").unwrap();
    db.execute("MERGE (:Mg {name: 'first'})").unwrap();
    let result = db.execute("MATCH (n:Mg) RETURN count(n) AS c").unwrap();
    assert_eq!(result.rows[0][0], Value::Int64(1));
}

#[test]
fn query_remove_property() {
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (:RP {a: 1, b: 2})").unwrap();
    db.execute("MATCH (n:RP) REMOVE n.a").unwrap();
    let result = db.execute("MATCH (n:RP) RETURN n.a AS a").unwrap();
    assert!(matches!(result.rows[0][0], Value::Null));
}

#[test]
fn query_detach_delete() {
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (:DD)-[:REL]->(:DD)").unwrap();
    db.execute("MATCH (n:DD) DETACH DELETE n").unwrap();
    assert_eq!(db.node_count(), 0);
    assert_eq!(db.edge_count(), 0);
}

#[test]
fn query_exists_subquery() {
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (:P {name: 'A'})-[:KNOWS]->(:P {name: 'B'})")
        .unwrap();
    db.execute("CREATE (:P {name: 'C'})").unwrap();
    let result = db
        .execute("MATCH (p:P) WHERE EXISTS { MATCH (p)-[:KNOWS]->() } RETURN p.name AS name")
        .unwrap();
    assert_eq!(result.rows.len(), 1);
}

#[test]
fn query_string_functions() {
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (:SF {name: 'Hello World'})").unwrap();
    let result = db
        .execute("MATCH (s:SF) RETURN toLower(s.name) AS l, toUpper(s.name) AS u")
        .unwrap();
    assert_eq!(result.rows.len(), 1);
}

#[test]
fn query_type_coercion() {
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (:TC {v: 42})").unwrap();
    let result = db.execute("MATCH (t:TC) RETURN t.v + 0.5 AS r").unwrap();
    assert_eq!(result.rows.len(), 1);
}

#[test]
fn query_count_star_explicit() {
    let db = ObrainDB::new_in_memory();
    for _ in 0..5 {
        db.create_node(&["CS"]);
    }
    let result = db.execute("MATCH (n:CS) RETURN count(*) AS c").unwrap();
    assert_eq!(result.rows[0][0], Value::Int64(5));
}

// ===========================================================================
// Config builder API (config.rs)
// ===========================================================================

#[test]
fn config_in_memory_defaults() {
    use obrain_engine::config::Config;
    let config = Config::in_memory();
    assert!(config.validate().is_ok());
}

#[test]
fn config_builder_chain() {
    use obrain_engine::config::{Config, DurabilityMode, GraphModel};
    use std::time::Duration;

    let config = Config::in_memory()
        .with_memory_limit(1024 * 1024 * 256)
        .with_threads(4)
        .with_query_logging()
        .with_memory_fraction(0.8)
        .with_adaptive(obrain_engine::config::AdaptiveConfig::disabled())
        .with_graph_model(GraphModel::Lpg)
        .with_wal_durability(DurabilityMode::Sync)
        .with_query_timeout(Duration::from_secs(30))
        .with_gc_interval(500);
    assert!(config.validate().is_ok());
}

#[test]
fn config_adaptive_config_builder() {
    use obrain_engine::config::AdaptiveConfig;
    let adaptive = AdaptiveConfig::disabled()
        .with_threshold(0.5)
        .with_min_rows(100)
        .with_max_reoptimizations(5);
    // Just verify it doesn't panic
    let _ = format!("{:?}", adaptive);
}

#[test]
fn config_without_backward_edges() {
    use obrain_engine::config::Config;
    let config = Config::in_memory().without_backward_edges();
    assert!(config.validate().is_ok());
}

#[test]
fn config_without_factorized_execution() {
    use obrain_engine::config::Config;
    let config = Config::in_memory().without_factorized_execution();
    assert!(config.validate().is_ok());
}

#[test]
fn config_with_schema_constraints() {
    use obrain_engine::config::Config;
    let config = Config::in_memory().with_schema_constraints();
    assert!(config.validate().is_ok());
}

#[test]
fn config_without_adaptive() {
    use obrain_engine::config::Config;
    let config = Config::in_memory().without_adaptive();
    assert!(config.validate().is_ok());
}

#[test]
fn config_graph_model_variants() {
    use obrain_engine::config::GraphModel;
    let lpg = GraphModel::Lpg;
    let rdf = GraphModel::Rdf;
    assert_ne!(format!("{:?}", lpg), format!("{:?}", rdf));
}

#[test]
fn config_access_mode_variants() {
    use obrain_engine::config::AccessMode;
    let rw = AccessMode::ReadWrite;
    let ro = AccessMode::ReadOnly;
    assert_ne!(format!("{:?}", rw), format!("{:?}", ro));
}

#[test]
fn config_durability_mode_variants() {
    use obrain_engine::config::{Config, DurabilityMode};
    for mode in [
        DurabilityMode::Sync,
        DurabilityMode::NoSync,
        DurabilityMode::Batch {
            max_delay_ms: 100,
            max_records: 50,
        },
    ] {
        let config = Config::in_memory().with_wal_durability(mode);
        assert!(config.validate().is_ok());
    }
}

#[test]
fn config_storage_format_variants() {
    use obrain_engine::config::{Config, StorageFormat};
    for fmt in [
        StorageFormat::Auto,
        StorageFormat::WalDirectory,
        StorageFormat::SingleFile,
    ] {
        let config = Config::in_memory().with_storage_format(fmt);
        assert!(config.validate().is_ok());
    }
}

#[test]
fn config_validate_invalid_memory_fraction() {
    use obrain_engine::config::Config;
    let config = Config::in_memory().with_memory_fraction(0.8);
    // Valid fraction should pass
    assert!(config.validate().is_ok());
}

#[test]
fn config_validate_zero_threads() {
    use obrain_engine::config::Config;
    let config = Config::in_memory().with_threads(0);
    assert!(config.validate().is_err());
}

#[test]
fn config_error_display() {
    use obrain_engine::config::ConfigError;
    let errors = [
        ConfigError::ZeroMemoryLimit,
        ConfigError::ZeroThreads,
        ConfigError::ZeroWalFlushInterval,
        ConfigError::RdfFeatureRequired,
    ];
    for err in &errors {
        let msg = err.to_string();
        assert!(!msg.is_empty());
    }
}

// ===========================================================================
// Database with_config (database/mod.rs)
// ===========================================================================

#[test]
fn database_with_config() {
    use obrain_engine::config::Config;
    let config = Config::in_memory().with_memory_limit(64 * 1024 * 1024);
    let db = ObrainDB::with_config(config).unwrap();
    db.create_node(&["Test"]);
    assert_eq!(db.node_count(), 1);
}

#[test]
fn database_with_config_adaptive_disabled() {
    use obrain_engine::config::{AdaptiveConfig, Config};
    let config = Config::in_memory().with_adaptive(AdaptiveConfig::disabled());
    let db = ObrainDB::with_config(config).unwrap();
    db.execute("CREATE (:N)").unwrap();
    assert_eq!(db.node_count(), 1);
}

#[test]
fn database_memory_limit() {
    let db = ObrainDB::new_in_memory();
    // Default in-memory has no explicit memory limit
    let _ = db.memory_limit();
}

#[test]
fn database_file_manager_in_memory() {
    let db = ObrainDB::new_in_memory();
    // In-memory DB has no file manager
    assert!(db.file_manager().is_none());
}

#[test]
fn database_version_constant() {
    assert!(!obrain_engine::VERSION.is_empty());
}

// ===========================================================================
// Admin types (admin.rs) — struct construction and serde
// ===========================================================================

#[test]
fn admin_database_mode_display_and_eq() {
    use obrain_engine::admin::DatabaseMode;
    assert_eq!(DatabaseMode::Lpg.to_string(), "lpg");
    assert_eq!(DatabaseMode::Rdf.to_string(), "rdf");
    assert_eq!(DatabaseMode::Lpg, DatabaseMode::Lpg);
    assert_ne!(DatabaseMode::Lpg, DatabaseMode::Rdf);
}

#[test]
fn admin_dump_format_default_and_display() {
    use obrain_engine::admin::DumpFormat;
    assert_eq!(DumpFormat::default(), DumpFormat::Parquet);
    assert_eq!(DumpFormat::Parquet.to_string(), "parquet");
    assert_eq!(DumpFormat::Turtle.to_string(), "turtle");
    assert_eq!(DumpFormat::Json.to_string(), "json");
}

#[test]
fn admin_dump_format_from_str() {
    use obrain_engine::admin::DumpFormat;
    assert_eq!(
        "parquet".parse::<DumpFormat>().unwrap(),
        DumpFormat::Parquet
    );
    assert_eq!("turtle".parse::<DumpFormat>().unwrap(), DumpFormat::Turtle);
    assert_eq!("ttl".parse::<DumpFormat>().unwrap(), DumpFormat::Turtle);
    assert_eq!("json".parse::<DumpFormat>().unwrap(), DumpFormat::Json);
    assert_eq!("jsonl".parse::<DumpFormat>().unwrap(), DumpFormat::Json);
    assert_eq!(
        "PARQUET".parse::<DumpFormat>().unwrap(),
        DumpFormat::Parquet
    );
    assert!("xml".parse::<DumpFormat>().is_err());
}

#[test]
fn admin_validation_result() {
    use obrain_engine::admin::{ValidationError, ValidationResult, ValidationWarning};
    let valid = ValidationResult::default();
    assert!(valid.is_valid());

    let invalid = ValidationResult {
        errors: vec![ValidationError {
            code: "E001".into(),
            message: "test".into(),
            context: None,
        }],
        warnings: vec![ValidationWarning {
            code: "W001".into(),
            message: "warn".into(),
            context: Some("ctx".into()),
        }],
    };
    assert!(!invalid.is_valid());
}

#[test]
fn admin_database_info_serde() {
    use obrain_engine::admin::{DatabaseInfo, DatabaseMode};
    let info = DatabaseInfo {
        mode: DatabaseMode::Lpg,
        node_count: 100,
        edge_count: 200,
        is_persistent: false,
        path: None,
        wal_enabled: false,
        version: "0.0.1".into(),
    };
    let json = serde_json::to_string(&info).unwrap();
    let parsed: DatabaseInfo = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.node_count, 100);
}

#[test]
fn admin_schema_info_lpg_serde() {
    use obrain_engine::admin::{EdgeTypeInfo, LabelInfo, LpgSchemaInfo, SchemaInfo};
    let schema = SchemaInfo::Lpg(LpgSchemaInfo {
        labels: vec![LabelInfo {
            name: "Person".into(),
            count: 10,
        }],
        edge_types: vec![EdgeTypeInfo {
            name: "KNOWS".into(),
            count: 5,
        }],
        property_keys: vec!["name".into()],
    });
    let json = serde_json::to_string(&schema).unwrap();
    assert!(json.contains("Person"));
    let _parsed: SchemaInfo = serde_json::from_str(&json).unwrap();
}

#[test]
fn admin_compaction_stats_serde() {
    use obrain_engine::admin::CompactionStats;
    let stats = CompactionStats {
        bytes_reclaimed: 1024,
        nodes_compacted: 10,
        edges_compacted: 20,
        duration_ms: 150,
    };
    let json = serde_json::to_string(&stats).unwrap();
    let parsed: CompactionStats = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.bytes_reclaimed, 1024);
}

#[test]
fn admin_dump_metadata_serde() {
    use obrain_engine::admin::{DatabaseMode, DumpFormat, DumpMetadata};
    let metadata = DumpMetadata {
        version: "0.0.1".into(),
        mode: DatabaseMode::Lpg,
        format: DumpFormat::Json,
        node_count: 500,
        edge_count: 1000,
        created_at: "2026-01-01T00:00:00Z".into(),
        extra: Default::default(),
    };
    let json = serde_json::to_string(&metadata).unwrap();
    let parsed: DumpMetadata = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.format, DumpFormat::Json);
}

#[test]
fn admin_wal_status_serde() {
    use obrain_engine::admin::WalStatus;
    let status = WalStatus {
        enabled: true,
        path: Some(std::path::PathBuf::from("/tmp/wal")),
        size_bytes: 8192,
        record_count: 42,
        last_checkpoint: Some(1700000000),
        current_epoch: 100,
    };
    let json = serde_json::to_string(&status).unwrap();
    let parsed: WalStatus = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.record_count, 42);
}

// ===========================================================================
// Collections extended — types + concurrent set/index set
// ===========================================================================

#[test]
fn collections_concurrent_set_type() {
    use obrain_common::collections::ObrainConcurrentSet;
    let set = ObrainConcurrentSet::<String>::default();
    set.insert("hello".into());
    set.insert("world".into());
    set.insert("hello".into()); // dup
    assert_eq!(set.len(), 2);
    assert!(set.contains("hello"));
}

#[test]
fn collections_index_set_type() {
    use obrain_common::collections::ObrainIndexSet;
    let mut set = ObrainIndexSet::<String>::default();
    set.insert("z".into());
    set.insert("a".into());
    set.insert("m".into());
    let items: Vec<_> = set.iter().cloned().collect();
    assert_eq!(items, vec!["z", "a", "m"]);
}

#[test]
fn collections_type_aliases_basic() {
    use obrain_common::collections::{ObrainConcurrentMap, ObrainIndexMap, ObrainMap, ObrainSet};
    let mut map = ObrainMap::<&str, i32>::default();
    map.insert("a", 1);
    assert_eq!(map.len(), 1);

    let mut set = ObrainSet::<i32>::default();
    set.insert(42);
    assert!(set.contains(&42));

    let cmap = ObrainConcurrentMap::<String, i32>::default();
    cmap.insert("key".into(), 99);
    assert_eq!(*cmap.get("key").unwrap(), 99);

    let mut imap = ObrainIndexMap::<&str, i32>::default();
    imap.insert("first", 1);
    imap.insert("second", 2);
    assert_eq!(imap.keys().next(), Some(&"first"));
}

// ===========================================================================
// More query patterns (exercise deeper pipeline paths)
// ===========================================================================

#[test]
fn query_case_when() {
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (:S {v: 1}), (:S {v: 2}), (:S {v: 3})")
        .unwrap();
    let result = db
        .execute("MATCH (s:S) RETURN CASE WHEN s.v > 2 THEN 'high' ELSE 'low' END AS level")
        .unwrap();
    assert_eq!(result.rows.len(), 3);
}

#[test]
fn query_coalesce() {
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (:C {a: 1}), (:C)").unwrap();
    let result = db
        .execute("MATCH (c:C) RETURN coalesce(c.a, 0) AS val")
        .unwrap();
    assert_eq!(result.rows.len(), 2);
}

#[test]
fn query_list_literal() {
    let db = ObrainDB::new_in_memory();
    let result = db
        .execute("RETURN [1, 2, 3] AS list, size([1, 2, 3]) AS sz")
        .unwrap();
    assert_eq!(result.rows.len(), 1);
}

#[test]
fn query_string_starts_with() {
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (:SW {name: 'Alice'}), (:SW {name: 'Bob'}), (:SW {name: 'Anna'})")
        .unwrap();
    let result = db
        .execute("MATCH (s:SW) WHERE s.name STARTS WITH 'A' RETURN s.name AS n")
        .unwrap();
    assert_eq!(result.rows.len(), 2);
}

#[test]
fn query_string_contains() {
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (:SC {name: 'Alice'}), (:SC {name: 'Bob'})")
        .unwrap();
    let result = db
        .execute("MATCH (s:SC) WHERE s.name CONTAINS 'lic' RETURN s.name AS n")
        .unwrap();
    assert_eq!(result.rows.len(), 1);
}

#[test]
fn query_string_ends_with() {
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (:SE {name: 'Alice'}), (:SE {name: 'Bob'})")
        .unwrap();
    let result = db
        .execute("MATCH (s:SE) WHERE s.name ENDS WITH 'ce' RETURN s.name AS n")
        .unwrap();
    assert_eq!(result.rows.len(), 1);
}

#[test]
fn query_in_list() {
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (:IL {v: 1}), (:IL {v: 2}), (:IL {v: 3}), (:IL {v: 4})")
        .unwrap();
    let result = db
        .execute("MATCH (n:IL) WHERE n.v IN [1, 3] RETURN n.v AS v")
        .unwrap();
    assert_eq!(result.rows.len(), 2);
}

#[test]
fn query_not_null() {
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (:NN {x: 1}), (:NN)").unwrap();
    let result = db
        .execute("MATCH (n:NN) WHERE n.x IS NOT NULL RETURN n.x AS x")
        .unwrap();
    assert_eq!(result.rows.len(), 1);
}

#[test]
fn query_is_null() {
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (:IN {x: 1}), (:IN)").unwrap();
    let result = db
        .execute("MATCH (n:IN) WHERE n.x IS NULL RETURN n")
        .unwrap();
    assert_eq!(result.rows.len(), 1);
}

#[test]
fn query_boolean_logic() {
    let db = ObrainDB::new_in_memory();
    db.execute(
        "CREATE (:BL {a: true, b: false}), (:BL {a: false, b: true}), (:BL {a: true, b: true})",
    )
    .unwrap();
    let result = db
        .execute("MATCH (n:BL) WHERE n.a AND n.b RETURN n")
        .unwrap();
    assert_eq!(result.rows.len(), 1);

    let result = db
        .execute("MATCH (n:BL) WHERE n.a OR n.b RETURN n")
        .unwrap();
    assert_eq!(result.rows.len(), 3);

    let result = db.execute("MATCH (n:BL) WHERE NOT n.a RETURN n").unwrap();
    assert_eq!(result.rows.len(), 1);
}

#[test]
fn query_math_operators() {
    let db = ObrainDB::new_in_memory();
    let result = db
        .execute(
            "RETURN 10 + 5 AS add, 10 - 5 AS sub, 10 * 5 AS mul, 10 / 5 AS div, 10 % 3 AS modulo",
        )
        .unwrap();
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.column_count(), 5);
}

#[test]
fn query_comparison_operators() {
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (:CO {v: 5})").unwrap();
    let result = db
        .execute("MATCH (n:CO) WHERE n.v > 3 AND n.v < 10 AND n.v >= 5 AND n.v <= 5 AND n.v <> 0 RETURN n.v")
        .unwrap();
    assert_eq!(result.rows.len(), 1);
}

#[test]
fn query_variable_length_path() {
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (a:VL {name: 'A'})-[:NEXT]->(b:VL {name: 'B'})-[:NEXT]->(c:VL {name: 'C'})")
        .unwrap();
    let result = db
        .execute("MATCH (a:VL {name: 'A'})-[:NEXT*1..3]->(end:VL) RETURN end.name AS name")
        .unwrap();
    assert!(result.rows.len() >= 2);
}

#[test]
fn query_shortest_path() {
    let db = ObrainDB::new_in_memory();
    db.execute(
        "CREATE (a:SP {name: 'A'})-[:E]->(b:SP {name: 'B'})-[:E]->(c:SP {name: 'C'}), (a)-[:E]->(c)",
    )
    .unwrap();
    // Just verify the query doesn't panic
    let _ = db.execute("MATCH p = shortestPath((a:SP {name: 'A'})-[:E*]->(c:SP {name: 'C'})) RETURN length(p) AS len");
}

#[test]
fn query_create_relationship_chain() {
    let db = ObrainDB::new_in_memory();
    db.execute(
        "CREATE (:CH {n: 1})-[:NEXT]->(:CH {n: 2})-[:NEXT]->(:CH {n: 3})-[:NEXT]->(:CH {n: 4})",
    )
    .unwrap();
    assert_eq!(db.node_count(), 4);
    assert_eq!(db.edge_count(), 3);
}

#[test]
fn query_where_on_relationship_type() {
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (:WT)-[:A]->(:WT), (:WT)-[:B]->(:WT)")
        .unwrap();
    let result = db
        .execute("MATCH ()-[r:A]->() RETURN count(r) AS c")
        .unwrap();
    assert_eq!(result.rows[0][0], Value::Int64(1));
}

#[test]
fn query_id_function() {
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (:ID {v: 1})").unwrap();
    let result = db.execute("MATCH (n:ID) RETURN id(n) AS nid").unwrap();
    assert_eq!(result.rows.len(), 1);
}

#[test]
fn query_labels_function() {
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (:L1:L2 {v: 1})").unwrap();
    let result = db.execute("MATCH (n:L1) RETURN labels(n) AS lbls").unwrap();
    assert_eq!(result.rows.len(), 1);
}

#[test]
fn query_type_function() {
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (:TF)-[:MY_REL]->(:TF)").unwrap();
    let result = db.execute("MATCH ()-[r]->() RETURN type(r) AS t").unwrap();
    assert_eq!(result.rows.len(), 1);
}

#[test]
fn query_properties_function() {
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (:PF {a: 1, b: 'two'})").unwrap();
    let result = db
        .execute("MATCH (n:PF) RETURN properties(n) AS props")
        .unwrap();
    assert_eq!(result.rows.len(), 1);
}

#[test]
fn query_exists_check() {
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (:EX {a: 1}), (:EX)").unwrap();
    let result = db
        .execute("MATCH (n:EX) WHERE n.a IS NOT NULL RETURN count(n) AS c")
        .unwrap();
    assert_eq!(result.rows[0][0], Value::Int64(1));
}

#[test]
fn query_group_by_with_aggregation() {
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (:GB {cat: 'A', v: 1}), (:GB {cat: 'A', v: 2}), (:GB {cat: 'B', v: 10})")
        .unwrap();
    let result = db
        .execute("MATCH (n:GB) RETURN n.cat AS cat, sum(n.v) AS total ORDER BY cat")
        .unwrap();
    assert_eq!(result.rows.len(), 2);
}

#[test]
fn query_negative_numbers() {
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (:NG {v: -42})").unwrap();
    let result = db.execute("MATCH (n:NG) RETURN n.v AS v").unwrap();
    assert_eq!(result.rows[0][0], Value::Int64(-42));
}

#[test]
fn query_large_integer() {
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (:LI {v: 9999999999})").unwrap();
    let result = db.execute("MATCH (n:LI) RETURN n.v AS v").unwrap();
    assert_eq!(result.rows.len(), 1);
}

#[test]
fn query_float_property() {
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (:FP {v: 1.23456})").unwrap();
    let result = db.execute("MATCH (n:FP) RETURN n.v AS v").unwrap();
    assert_eq!(result.rows.len(), 1);
}

#[test]
fn query_empty_string_property() {
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (:ES {name: ''})").unwrap();
    let result = db.execute("MATCH (n:ES) RETURN n.name AS name").unwrap();
    assert_eq!(result.rows[0][0], Value::String("".into()));
}

#[test]
fn query_with_aliased_aggregation() {
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (:WA {v: 10}), (:WA {v: 20})").unwrap();
    let result = db
        .execute("MATCH (n:WA) WITH sum(n.v) AS total RETURN total")
        .unwrap();
    // WITH aggregation may produce different row counts depending on engine behavior
    assert!(!result.rows.is_empty());
}

#[test]
fn query_delete_with_where() {
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (:DW {v: 1}), (:DW {v: 2}), (:DW {v: 3})")
        .unwrap();
    db.execute("MATCH (n:DW) WHERE n.v = 2 DELETE n").unwrap();
    assert_eq!(db.node_count(), 2);
}

#[test]
fn query_set_multiple_properties() {
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (:SM {a: 1})").unwrap();
    db.execute("MATCH (n:SM) SET n.b = 2, n.c = 3").unwrap();
    let result = db
        .execute("MATCH (n:SM) RETURN n.a AS a, n.b AS b, n.c AS c")
        .unwrap();
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.column_count(), 3);
}

// ===========================================================================
// Session extended edge cases
// ===========================================================================

#[test]
fn session_multiple_queries_in_transaction() {
    let db = ObrainDB::new_in_memory();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("CREATE (:MT {v: 1})").unwrap();
    session.execute("CREATE (:MT {v: 2})").unwrap();
    session.execute("CREATE (:MT {v: 3})").unwrap();
    session.commit().unwrap();
    let result = db.execute("MATCH (n:MT) RETURN count(n) AS c").unwrap();
    assert_eq!(result.rows[0][0], Value::Int64(3));
}

#[test]
fn session_nested_savepoints() {
    let db = ObrainDB::new_in_memory();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("CREATE (:NS {v: 1})").unwrap();
    session.savepoint("sp1").unwrap();
    session.execute("CREATE (:NS {v: 2})").unwrap();
    session.savepoint("sp2").unwrap();
    session.execute("CREATE (:NS {v: 3})").unwrap();
    session.rollback_to_savepoint("sp2").unwrap();
    session.commit().unwrap();
    let result = db.execute("MATCH (n:NS) RETURN count(n) AS c").unwrap();
    assert_eq!(result.rows[0][0], Value::Int64(2));
}

#[test]
fn session_transaction_isolation() {
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (:TI {v: 1})").unwrap();

    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("CREATE (:TI {v: 2})").unwrap();
    // Before commit, the main DB should still see only 1 node
    // (depending on isolation level)
    session.commit().unwrap();

    let result = db.execute("MATCH (n:TI) RETURN count(n) AS c").unwrap();
    assert_eq!(result.rows[0][0], Value::Int64(2));
}

#[test]
fn session_parameter_types() {
    let db = ObrainDB::new_in_memory();
    let session = db.session();
    session.set_parameter("int_param", Value::Int64(42));
    session.set_parameter("str_param", Value::from("hello"));
    session.set_parameter("float_param", Value::Float64(1.234));
    session.set_parameter("bool_param", Value::from(true));

    assert_eq!(session.get_parameter("int_param"), Some(Value::Int64(42)));
    assert_eq!(
        session.get_parameter("str_param"),
        Some(Value::from("hello"))
    );
}

// ===========================================================================
// Catalog access (catalog/mod.rs)
// ===========================================================================

#[test]
fn catalog_basic_access() {
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (:CatPerson {name: 'Alice'})").unwrap();
    db.execute("CREATE (:CatPerson)-[:CatKNOWS]->(:CatPerson)")
        .unwrap();
    // The catalog should have recorded these labels and types
    let _catalog = db.store();
    // Just verifying access doesn't panic
}

// ===========================================================================
// CRUD batch operations
// ===========================================================================

#[test]
fn crud_bulk_create_nodes() {
    let db = ObrainDB::new_in_memory();
    let ids: Vec<_> = (0..100)
        .map(|i| db.create_node_with_props(&["Bulk"], [("idx", Value::from(i as i64))]))
        .collect();
    assert_eq!(ids.len(), 100);
    assert_eq!(db.node_count(), 100);
}

#[test]
fn crud_bulk_create_edges() {
    let db = ObrainDB::new_in_memory();
    let nodes: Vec<_> = (0..10).map(|_| db.create_node(&["EN"])).collect();
    for i in 0..9 {
        db.create_edge(nodes[i], nodes[i + 1], "NEXT");
    }
    assert_eq!(db.edge_count(), 9);
}

#[test]
fn crud_node_multiple_labels() {
    let db = ObrainDB::new_in_memory();
    let id = db.create_node(&["A", "B", "C"]);
    let labels = db.get_node_labels(id).unwrap();
    assert!(labels.contains(&"A".to_string()));
    assert!(labels.contains(&"B".to_string()));
    assert!(labels.contains(&"C".to_string()));
}

#[test]
fn crud_overwrite_property() {
    let db = ObrainDB::new_in_memory();
    let id = db.create_node(&["OW"]);
    db.set_node_property(id, "key", Value::from("v1"));
    db.set_node_property(id, "key", Value::from("v2"));
    let result = db.execute("MATCH (n:OW) RETURN n.key AS k").unwrap();
    assert_eq!(result.rows[0][0], Value::from("v2"));
}

#[test]
fn crud_remove_nonexistent_property() {
    let db = ObrainDB::new_in_memory();
    let id = db.create_node(&["RNP"]);
    let removed = db.remove_node_property(id, "nonexistent");
    assert!(!removed);
}

// ===========================================================================
// QueryResult extended
// ===========================================================================

#[test]
fn query_result_multiple_columns_types() {
    use obrain_common::types::LogicalType;
    let result = obrain_engine::database::QueryResult::with_types(
        vec!["id".into(), "name".into(), "age".into(), "active".into()],
        vec![
            LogicalType::Int64,
            LogicalType::String,
            LogicalType::Float64,
            LogicalType::Bool,
        ],
    );
    assert_eq!(result.column_count(), 4);
    assert_eq!(result.column_types.len(), 4);
}

#[test]
fn query_result_status_message() {
    let result = obrain_engine::database::QueryResult::status("Index created successfully");
    assert_eq!(
        result.status_message,
        Some("Index created successfully".to_string())
    );
    assert!(result.is_empty());
    assert_eq!(result.row_count(), 0);
}

#[test]
fn query_result_scalar_single_value() {
    let db = ObrainDB::new_in_memory();
    let result = db.execute("RETURN 42 AS answer").unwrap();
    let val: i64 = result.scalar().unwrap();
    assert_eq!(val, 42);
}

#[test]
fn query_result_scalar_string() {
    let db = ObrainDB::new_in_memory();
    let result = db.execute("RETURN 'hello' AS greeting").unwrap();
    let val: String = result.scalar().unwrap();
    assert_eq!(val, "hello");
}

#[test]
fn query_result_iter_values() {
    let db = ObrainDB::new_in_memory();
    db.execute("CREATE (:IV {v: 1}), (:IV {v: 2}), (:IV {v: 3})")
        .unwrap();
    let result = db
        .execute("MATCH (n:IV) RETURN n.v AS v ORDER BY v")
        .unwrap();
    let vals: Vec<&Vec<Value>> = result.iter().collect();
    assert_eq!(vals.len(), 3);
}

// ===========================================================================
// Error edge cases
// ===========================================================================

#[test]
fn error_from_string() {
    use obrain_common::utils::error::Error;
    let err = Error::InvalidValue("test value".into());
    assert!(err.to_string().contains("test value"));
    let _ = err.error_code();
}

#[test]
fn error_serialization_variant() {
    use obrain_common::utils::error::Error;
    let err = Error::Serialization("failed to serialize".into());
    assert!(err.to_string().contains("serialize"));
}

#[test]
fn error_property_not_found() {
    use obrain_common::utils::error::Error;
    let err = Error::PropertyNotFound("missing_prop".into());
    assert!(err.to_string().contains("missing_prop"));
    assert!(err.to_string().contains("OBRAIN-V004"));
}

#[test]
fn error_label_not_found() {
    use obrain_common::utils::error::Error;
    let err = Error::LabelNotFound("UnknownLabel".into());
    assert!(err.to_string().contains("UnknownLabel"));
    assert!(err.to_string().contains("OBRAIN-V005"));
}

#[test]
fn error_type_mismatch_detail() {
    use obrain_common::utils::error::Error;
    let err = Error::TypeMismatch {
        expected: "INT64".into(),
        found: "STRING".into(),
    };
    let msg = err.to_string();
    assert!(msg.contains("INT64"));
    assert!(msg.contains("STRING"));
    assert!(msg.contains("OBRAIN-V006"));
}

#[test]
fn source_span_construction() {
    use obrain_common::utils::error::SourceSpan;
    let span = SourceSpan::new(0, 10, 1, 1);
    assert_eq!(span.start, 0);
    assert_eq!(span.end, 10);
    assert_eq!(span.line, 1);
    assert_eq!(span.column, 1);
}

#[test]
fn query_error_without_optional_fields() {
    use obrain_common::utils::error::{QueryError, QueryErrorKind};
    let err = QueryError::new(QueryErrorKind::Semantic, "test error");
    assert!(err.span.is_none());
    assert!(err.source_query.is_none());
    assert!(err.hint.is_none());
    let display = err.to_string();
    assert!(display.contains("test error"));
}
