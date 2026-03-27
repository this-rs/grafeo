//! Integration tests for the single-file `.grafeo` database format.

#![cfg(feature = "grafeo-file")]

use grafeo_common::types::Value;
use grafeo_engine::{Config, GrafeoDB};

/// Helper: extract string values from column 0 of query result rows.
fn extract_strings(rows: &[Vec<Value>]) -> Vec<String> {
    let mut names: Vec<String> = rows
        .iter()
        .filter_map(|r| match &r[0] {
            Value::String(s) => Some(s.to_string()),
            _ => None,
        })
        .collect();
    names.sort();
    names
}

/// Helper: compute sidecar WAL path for a .grafeo file.
fn sidecar_wal_path(db_path: &std::path::Path) -> std::path::PathBuf {
    let mut p = db_path.as_os_str().to_owned();
    p.push(".wal");
    std::path::PathBuf::from(p)
}

// =========================================================================
// Basic create, open, and reopen
// =========================================================================

#[test]
fn create_new_grafeo_file() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("test.grafeo");

    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    assert_eq!(db.node_count(), 0);
    assert_eq!(db.edge_count(), 0);

    // File should exist
    assert!(path.exists());
    assert!(path.is_file());

    db.close().unwrap();
}

#[test]
fn insert_close_reopen_persists_data() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("persist.grafeo");

    // Create and populate
    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let session = db.session();
        session
            .execute("INSERT (:Person {name: 'Alix', age: 30})")
            .unwrap();
        session
            .execute("INSERT (:Person {name: 'Gus', age: 25})")
            .unwrap();
        session
            .execute(
                "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) \
                 INSERT (a)-[:KNOWS]->(b)",
            )
            .unwrap();
        assert_eq!(db.node_count(), 2);
        assert_eq!(db.edge_count(), 1);
        db.close().unwrap();
    }

    // Sidecar WAL should be gone after close
    assert!(
        !sidecar_wal_path(&path).exists(),
        "sidecar WAL should be removed after close"
    );

    // Reopen and verify
    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        assert_eq!(db.node_count(), 2);
        assert_eq!(db.edge_count(), 1);

        // Verify data is queryable
        let session = db.session();
        let result = session
            .execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
            .unwrap();
        assert_eq!(extract_strings(&result.rows), vec!["Alix", "Gus"]);
        db.close().unwrap();
    }
}

#[test]
fn save_as_grafeo_file_from_in_memory() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("exported.grafeo");

    // Create in-memory DB and populate
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("INSERT (:City {name: 'Amsterdam'})")
        .unwrap();
    session.execute("INSERT (:City {name: 'Berlin'})").unwrap();
    assert_eq!(db.node_count(), 2);

    // Save as .grafeo file
    db.save(&path).unwrap();

    // Open the file and verify
    let db2 = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    assert_eq!(db2.node_count(), 2);

    let session2 = db2.session();
    let result = session2
        .execute("MATCH (c:City) RETURN c.name ORDER BY c.name")
        .unwrap();
    assert_eq!(extract_strings(&result.rows), vec!["Amsterdam", "Berlin"]);
    db2.close().unwrap();
}

#[test]
fn wal_checkpoint_writes_to_file() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("checkpoint.grafeo");

    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    let session = db.session();
    session
        .execute("INSERT (:Person {name: 'Vincent'})")
        .unwrap();

    // Checkpoint should write snapshot to file
    db.wal_checkpoint().unwrap();

    // Verify the file manager has a non-empty header
    let fm = db.file_manager().expect("should have file manager");
    let header = fm.active_header();
    assert!(header.snapshot_length > 0);
    assert_eq!(header.node_count, 1);
    assert_eq!(header.edge_count, 0);

    db.close().unwrap();
}

#[test]
fn multiple_checkpoints_alternate_headers() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("multi.grafeo");

    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    let session = db.session();

    session.execute("INSERT (:Person {name: 'Jules'})").unwrap();
    db.wal_checkpoint().unwrap();

    let fm = db.file_manager().unwrap();
    assert_eq!(fm.active_header().iteration, 1);

    session.execute("INSERT (:Person {name: 'Mia'})").unwrap();
    db.wal_checkpoint().unwrap();
    assert_eq!(fm.active_header().iteration, 2);
    assert_eq!(fm.active_header().node_count, 2);

    db.close().unwrap();

    // Reopen and verify both nodes are there
    let db2 = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    assert_eq!(db2.node_count(), 2);
    db2.close().unwrap();
}

#[test]
fn auto_detect_does_not_use_grafeo_file_for_directory_path() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("test_legacy");

    // Without .grafeo extension, should use WAL directory format
    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();

    #[cfg(feature = "grafeo-file")]
    assert!(
        db.file_manager().is_none(),
        "directory path should not use single-file format"
    );

    let session = db.session();
    session.execute("INSERT (:Person {name: 'Butch'})").unwrap();
    db.close().unwrap();

    // Path should be a directory (WAL format)
    assert!(path.is_dir());
}

#[test]
fn info_reports_persistence() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("info.grafeo");

    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    let info = db.info();
    assert!(info.is_persistent);
    assert!(info.path.is_some());
    db.close().unwrap();
}

// =========================================================================
// Checkpoint merging: data inserted between checkpoints is preserved
// =========================================================================

#[test]
fn checkpoint_merges_incremental_writes() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("merge.grafeo");

    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    let session = db.session();

    // Batch 1: insert 3 nodes, checkpoint
    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
    session
        .execute("INSERT (:Person {name: 'Vincent'})")
        .unwrap();
    db.wal_checkpoint().unwrap();
    assert_eq!(db.file_manager().unwrap().active_header().node_count, 3);

    // Batch 2: insert 2 more nodes, modify one, checkpoint again
    session.execute("INSERT (:Person {name: 'Jules'})").unwrap();
    session.execute("INSERT (:Person {name: 'Mia'})").unwrap();
    session
        .execute("MATCH (p:Person {name: 'Alix'}) SET p.age = 31")
        .unwrap();
    db.wal_checkpoint().unwrap();
    assert_eq!(db.file_manager().unwrap().active_header().node_count, 5);

    // Batch 3: delete a node, add an edge, checkpoint
    session
        .execute("MATCH (p:Person {name: 'Gus'}) DELETE p")
        .unwrap();
    session
        .execute(
            "MATCH (a:Person {name: 'Vincent'}), (b:Person {name: 'Jules'}) \
             INSERT (a)-[:KNOWS]->(b)",
        )
        .unwrap();
    db.wal_checkpoint().unwrap();

    let header = db.file_manager().unwrap().active_header();
    assert_eq!(header.node_count, 4);
    assert_eq!(header.edge_count, 1);
    assert_eq!(header.iteration, 3); // 3 checkpoints = iteration 3

    db.close().unwrap();

    // Reopen and verify final state
    let db2 = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    assert_eq!(db2.node_count(), 4);
    assert_eq!(db2.edge_count(), 1);

    let session2 = db2.session();
    let result = session2
        .execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
        .unwrap();
    assert_eq!(
        extract_strings(&result.rows),
        vec!["Alix", "Jules", "Mia", "Vincent"]
    );

    // Verify the property survived
    let result = session2
        .execute("MATCH (p:Person {name: 'Alix'}) RETURN p.age")
        .unwrap();
    assert_eq!(result.rows[0][0], Value::Int64(31));

    db2.close().unwrap();
}

#[test]
fn writes_after_checkpoint_survive_reopen() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("post_checkpoint.grafeo");

    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    let session = db.session();

    session
        .execute("INSERT (:City {name: 'Amsterdam'})")
        .unwrap();
    db.wal_checkpoint().unwrap();

    // Write MORE data after checkpoint, then close (without explicit checkpoint)
    session.execute("INSERT (:City {name: 'Berlin'})").unwrap();
    session.execute("INSERT (:City {name: 'Prague'})").unwrap();
    db.close().unwrap(); // close() does its own checkpoint

    // Reopen: all 3 cities should be present
    let db2 = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    assert_eq!(db2.node_count(), 3);

    let session2 = db2.session();
    let result = session2
        .execute("MATCH (c:City) RETURN c.name ORDER BY c.name")
        .unwrap();
    assert_eq!(
        extract_strings(&result.rows),
        vec!["Amsterdam", "Berlin", "Prague"]
    );
    db2.close().unwrap();
}

// =========================================================================
// Edge cases
// =========================================================================

#[test]
fn empty_database_roundtrip() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("empty.grafeo");

    // Create, close immediately, reopen
    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        db.close().unwrap();
    }
    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        assert_eq!(db.node_count(), 0);
        assert_eq!(db.edge_count(), 0);
        db.close().unwrap();
    }
}

#[test]
fn multiple_reopen_cycles() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("cycles.grafeo");

    // Cycle 1: create with data
    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        db.session()
            .execute("INSERT (:Person {name: 'Alix'})")
            .unwrap();
        db.close().unwrap();
    }

    // Cycle 2: add more data
    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        assert_eq!(db.node_count(), 1);
        db.session()
            .execute("INSERT (:Person {name: 'Gus'})")
            .unwrap();
        db.close().unwrap();
    }

    // Cycle 3: add more data
    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        assert_eq!(db.node_count(), 2);
        db.session()
            .execute("INSERT (:Person {name: 'Vincent'})")
            .unwrap();
        db.close().unwrap();
    }

    // Cycle 4: verify all data present
    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        assert_eq!(db.node_count(), 3);
        let session = db.session();
        let result = session
            .execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
            .unwrap();
        assert_eq!(
            extract_strings(&result.rows),
            vec!["Alix", "Gus", "Vincent"]
        );
        db.close().unwrap();
    }
}

#[test]
fn large_property_values_roundtrip() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("large_props.grafeo");

    let big_string = "x".repeat(100_000);

    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let session = db.session();
        session
            .execute_with_params(
                "INSERT (:Doc {content: $text})",
                [("text".to_string(), Value::String(big_string.clone().into()))]
                    .into_iter()
                    .collect(),
            )
            .unwrap();
        db.close().unwrap();
    }

    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let session = db.session();
        let result = session.execute("MATCH (d:Doc) RETURN d.content").unwrap();
        match &result.rows[0][0] {
            Value::String(s) => assert_eq!(s.len(), 100_000),
            other => panic!("expected String, got {other:?}"),
        }
        db.close().unwrap();
    }
}

#[test]
fn diverse_property_types_roundtrip() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("types.grafeo");

    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let session = db.session();
        session
            .execute(
                "INSERT (:Thing { \
                    str_val: 'hello', \
                    int_val: 42, \
                    float_val: 3.14, \
                    bool_val: true, \
                    list_val: [1, 2, 3] \
                 })",
            )
            .unwrap();
        db.close().unwrap();
    }

    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (t:Thing) RETURN t.str_val, t.int_val, t.float_val, t.bool_val, t.list_val",
            )
            .unwrap();
        let row = &result.rows[0];
        assert_eq!(row[0], Value::String("hello".into()));
        assert_eq!(row[1], Value::Int64(42));
        assert!(matches!(row[2], Value::Float64(_)));
        assert_eq!(row[3], Value::Bool(true));
        assert!(matches!(row[4], Value::List(_)));
        db.close().unwrap();
    }
}

#[test]
fn open_nonexistent_creates_new() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("new.grafeo");

    assert!(!path.exists());
    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    assert!(path.exists());
    assert_eq!(db.node_count(), 0);
    db.close().unwrap();
}

#[test]
fn file_grows_and_shrinks_with_data() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("size.grafeo");

    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    let fm = db.file_manager().unwrap();
    let initial_size = fm.file_size().unwrap();

    // Add substantial data
    let session = db.session();
    for i in 0..100 {
        session
            .execute(&format!(
                "INSERT (:Node {{idx: {i}, data: '{}'}})",
                "x".repeat(1000)
            ))
            .unwrap();
    }
    db.wal_checkpoint().unwrap();
    let large_size = fm.file_size().unwrap();
    assert!(large_size > initial_size, "file should grow with data");

    // Delete most data
    session
        .execute("MATCH (n:Node) WHERE n.idx > 5 DELETE n")
        .unwrap();
    db.wal_checkpoint().unwrap();
    let small_size = fm.file_size().unwrap();
    assert!(
        small_size < large_size,
        "file should shrink after deleting data: {small_size} >= {large_size}"
    );

    db.close().unwrap();
}

#[test]
fn sidecar_wal_exists_during_operation() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("wal_lifecycle.grafeo");

    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    let session = db.session();
    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

    // Sidecar WAL should exist while DB is open
    let wal = sidecar_wal_path(&path);
    assert!(wal.exists(), "sidecar WAL should exist during operation");
    assert!(wal.is_dir(), "sidecar WAL should be a directory");

    db.close().unwrap();

    // After close, sidecar should be cleaned up
    assert!(!wal.exists(), "sidecar WAL should be removed after close");
}

#[test]
fn checkpoint_idempotent() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("idempotent.grafeo");

    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    let session = db.session();
    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

    // Multiple checkpoints without intervening writes should be safe
    db.wal_checkpoint().unwrap();
    let iter1 = db.file_manager().unwrap().active_header().iteration;

    db.wal_checkpoint().unwrap();
    let iter2 = db.file_manager().unwrap().active_header().iteration;

    db.wal_checkpoint().unwrap();
    let iter3 = db.file_manager().unwrap().active_header().iteration;

    // Each checkpoint increments the iteration counter
    assert_eq!(iter2, iter1 + 1);
    assert_eq!(iter3, iter2 + 1);

    // Data is still consistent
    assert_eq!(db.node_count(), 1);

    db.close().unwrap();
}

#[test]
fn concurrent_sessions_before_close() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("sessions.grafeo");

    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();

        // Multiple sessions writing data
        let s1 = db.session();
        let s2 = db.session();

        s1.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        s2.execute("INSERT (:Person {name: 'Gus'})").unwrap();
        s1.execute("INSERT (:Person {name: 'Vincent'})").unwrap();

        assert_eq!(db.node_count(), 3);
        db.close().unwrap();
    }

    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        assert_eq!(db.node_count(), 3);
        db.close().unwrap();
    }
}

#[test]
fn named_graphs_persist() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("named_graphs.grafeo");

    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let session = db.session();
        session.execute("CREATE GRAPH social").unwrap();
        session.execute("USE GRAPH social").unwrap();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        session.execute("USE GRAPH DEFAULT").unwrap();
        session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
        db.close().unwrap();
    }

    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        // Default graph should have Gus
        let session = db.session();
        let result = session.execute("MATCH (p:Person) RETURN p.name").unwrap();
        assert_eq!(extract_strings(&result.rows), vec!["Gus"]);

        // Social graph should have Alix
        session.execute("USE GRAPH social").unwrap();
        let result = session.execute("MATCH (p:Person) RETURN p.name").unwrap();
        assert_eq!(extract_strings(&result.rows), vec!["Alix"]);
        db.close().unwrap();
    }
}

#[test]
fn edges_with_properties_persist() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("edge_props.grafeo");

    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
        session
            .execute(
                "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) \
                 INSERT (a)-[:KNOWS {since: 2020, strength: 0.95}]->(b)",
            )
            .unwrap();
        db.close().unwrap();
    }

    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let session = db.session();
        let result = session
            .execute("MATCH ()-[e:KNOWS]->() RETURN e.since, e.strength")
            .unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0][0], Value::Int64(2020));
        assert!(matches!(result.rows[0][1], Value::Float64(_)));
        db.close().unwrap();
    }
}

// =========================================================================
// Corruption and validation
// =========================================================================

#[test]
fn corrupt_snapshot_detected_on_open() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("corrupt.grafeo");

    // Write valid data
    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        db.close().unwrap();
    }

    // Corrupt the snapshot data region (offset 12288+)
    {
        use std::io::{Seek, SeekFrom, Write};
        let mut file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.seek(SeekFrom::Start(12288)).unwrap();
        file.write_all(b"CORRUPTED DATA HERE!!!").unwrap();
    }

    // Opening should fail with a checksum error
    let result = GrafeoDB::with_config(Config::persistent(&path));
    assert!(result.is_err());
    let err_msg = result.err().unwrap().to_string();
    assert!(
        err_msg.contains("checksum"),
        "expected checksum error, got: {err_msg}"
    );
}

#[test]
fn validate_reports_clean_state() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("valid.grafeo");

    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    let session = db.session();
    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
    session
        .execute(
            "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) \
             INSERT (a)-[:KNOWS]->(b)",
        )
        .unwrap();

    let validation = db.validate();
    assert!(validation.errors.is_empty(), "should have no errors");
    db.close().unwrap();
}

// =========================================================================
// WAL status and detailed stats
// =========================================================================

#[test]
fn wal_status_reflects_single_file() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("wal_status.grafeo");

    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    let session = db.session();
    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

    let status = db.wal_status();
    assert!(status.enabled);

    db.close().unwrap();
}

#[test]
fn detailed_stats_with_grafeo_file() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("stats.grafeo");

    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    let session = db.session();

    for i in 0..10 {
        session
            .execute(&format!("INSERT (:Node {{idx: {i}}})"))
            .unwrap();
    }

    let stats = db.detailed_stats();
    assert_eq!(stats.node_count, 10);
    assert_eq!(stats.edge_count, 0);

    db.close().unwrap();
}

// =========================================================================
// File locking
// =========================================================================

#[test]
fn second_open_of_same_file_is_rejected() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("locked.grafeo");

    let db1 = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    let session = db1.session();
    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

    // Second open should fail because the file is locked
    let result = GrafeoDB::open(&path);
    assert!(result.is_err(), "second open should fail due to file lock");

    db1.close().unwrap();

    // After close, open should succeed
    let db2 = GrafeoDB::open(&path).unwrap();
    assert_eq!(db2.node_count(), 1);
    db2.close().unwrap();
}

#[test]
fn lock_released_on_drop() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("drop_lock.grafeo");

    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        db.session().execute("INSERT (:X {v: 1})").unwrap();
        // Drop without explicit close: lock should still be released
    }

    // Should be able to open after drop
    let db2 = GrafeoDB::open(&path).unwrap();
    // Data may or may not persist (no explicit close/checkpoint), but open should succeed
    db2.close().unwrap();
}

// =========================================================================
// Schema (DDL) persistence
// =========================================================================

#[test]
fn node_type_definitions_persist() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("schema.grafeo");

    // Create DB and define node types
    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let session = db.session();
        session
            .execute("CREATE NODE TYPE Person (name STRING NOT NULL, age INT64)")
            .unwrap();
        session
            .execute("CREATE NODE TYPE Company (name STRING NOT NULL)")
            .unwrap();
        session
            .execute("INSERT (:Person {name: 'Alix', age: 30})")
            .unwrap();
        db.close().unwrap();
    }

    // Reopen and verify types survived
    {
        let db = GrafeoDB::open(&path).unwrap();
        let session = db.session();

        // Verify data
        let result = session.execute("MATCH (p:Person) RETURN p.name").unwrap();
        assert_eq!(extract_strings(&result.rows), vec!["Alix"]);

        // Verify node type definitions survived via SHOW NODE TYPES
        let result = session.execute("SHOW NODE TYPES").unwrap();
        let type_names = extract_strings(&result.rows);
        assert!(
            type_names.contains(&"Person".to_string()),
            "Person type missing: {type_names:?}"
        );
        assert!(
            type_names.contains(&"Company".to_string()),
            "Company type missing: {type_names:?}"
        );

        db.close().unwrap();
    }
}

#[test]
fn edge_type_definitions_persist() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("edge_types.grafeo");

    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let session = db.session();
        session
            .execute("CREATE EDGE TYPE KNOWS (since INT64)")
            .unwrap();
        session
            .execute("CREATE EDGE TYPE WORKS_AT (role STRING)")
            .unwrap();
        db.close().unwrap();
    }

    {
        let db = GrafeoDB::open(&path).unwrap();
        let session = db.session();

        let result = session.execute("SHOW EDGE TYPES").unwrap();
        let type_names = extract_strings(&result.rows);
        assert!(
            type_names.contains(&"KNOWS".to_string()),
            "KNOWS type missing: {type_names:?}"
        );
        assert!(
            type_names.contains(&"WORKS_AT".to_string()),
            "WORKS_AT type missing: {type_names:?}"
        );

        db.close().unwrap();
    }
}

#[test]
fn graph_type_definitions_persist() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("graph_types.grafeo");

    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let session = db.session();
        session
            .execute("CREATE NODE TYPE Person (name STRING)")
            .unwrap();
        session
            .execute("CREATE EDGE TYPE KNOWS (since INT64)")
            .unwrap();
        session
            .execute(
                "CREATE GRAPH TYPE SocialGraph (\
                 NODE TYPE Person (name STRING),\
                 EDGE TYPE KNOWS (since INT64)\
                 )",
            )
            .unwrap();
        db.close().unwrap();
    }

    {
        let db = GrafeoDB::open(&path).unwrap();
        let session = db.session();

        let result = session.execute("SHOW GRAPH TYPES").unwrap();
        let type_names = extract_strings(&result.rows);
        assert!(
            type_names.contains(&"SocialGraph".to_string()),
            "SocialGraph type missing: {type_names:?}"
        );

        db.close().unwrap();
    }
}

#[test]
fn schema_survives_export_import_roundtrip() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session
        .execute("CREATE NODE TYPE Person (name STRING NOT NULL, age INT64)")
        .unwrap();
    session
        .execute("INSERT (:Person {name: 'Alix', age: 30})")
        .unwrap();

    // Export and import
    let snapshot = db.export_snapshot().unwrap();
    let db2 = GrafeoDB::import_snapshot(&snapshot).unwrap();
    let session2 = db2.session();

    // Verify data
    let result = session2.execute("MATCH (p:Person) RETURN p.name").unwrap();
    assert_eq!(extract_strings(&result.rows), vec!["Alix"]);

    // Verify schema
    let result = session2.execute("SHOW NODE TYPES").unwrap();
    let type_names = extract_strings(&result.rows);
    assert!(
        type_names.contains(&"Person".to_string()),
        "Person type missing after import: {type_names:?}"
    );
}

#[test]
fn stored_procedures_persist() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("procedures.grafeo");

    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let session = db.session();
        session
            .execute(
                "CREATE PROCEDURE get_people() RETURNS (name STRING) \
                 AS { MATCH (p:Person) RETURN p.name AS name }",
            )
            .unwrap();
        db.close().unwrap();
    }

    {
        let db = GrafeoDB::open(&path).unwrap();
        let session = db.session();
        // Insert data so the procedure has something to return
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        // CALL the procedure to verify it survived the roundtrip
        let result = session.execute("CALL get_people()").unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(extract_strings(&result.rows), vec!["Alix"]);
        db.close().unwrap();
    }
}

#[test]
fn schema_with_data_across_multiple_cycles() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("schema_cycles.grafeo");

    // Cycle 1: Create type + insert
    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let session = db.session();
        session
            .execute("CREATE NODE TYPE Person (name STRING NOT NULL)")
            .unwrap();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        db.close().unwrap();
    }

    // Cycle 2: Add another type + more data
    {
        let db = GrafeoDB::open(&path).unwrap();
        let session = db.session();

        // Verify first type survived
        let result = session.execute("SHOW NODE TYPES").unwrap();
        assert!(extract_strings(&result.rows).contains(&"Person".to_string()));

        session
            .execute("CREATE NODE TYPE City (name STRING NOT NULL)")
            .unwrap();
        session
            .execute("INSERT (:City {name: 'Amsterdam'})")
            .unwrap();
        db.close().unwrap();
    }

    // Cycle 3: Verify everything
    {
        let db = GrafeoDB::open(&path).unwrap();
        let session = db.session();

        let result = session.execute("SHOW NODE TYPES").unwrap();
        let types = extract_strings(&result.rows);
        assert!(types.contains(&"Person".to_string()), "Person missing");
        assert!(types.contains(&"City".to_string()), "City missing");

        assert_eq!(db.node_count(), 2);
        db.close().unwrap();
    }
}
