//! Integration tests for WAL-based persistence and recovery.
//!
//! Covers: save/restore cycle, WAL recovery of nodes/edges/properties/labels,
//! and checkpoint operations.
//!
//! ```bash
//! cargo test -p grafeo-engine --features full --test wal_recovery
//! ```

#[cfg(feature = "wal")]
mod wal {
    use grafeo_common::types::Value;
    use grafeo_engine::GrafeoDB;

    #[test]
    fn test_persistent_db_roundtrip() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("test.grafeo");

        // Create and populate
        {
            let db = GrafeoDB::open(&path).expect("open for write");
            let a = db.create_node(&["Person"]);
            db.set_node_property(a, "name", Value::String("Alice".into()));
            db.set_node_property(a, "age", Value::Int64(30));
            let b = db.create_node(&["Person"]);
            db.set_node_property(b, "name", Value::String("Bob".into()));
            db.create_edge(a, b, "KNOWS");
            db.close().expect("close");
        }

        // Reopen and verify
        {
            let db = GrafeoDB::open(&path).expect("reopen");
            assert_eq!(db.node_count(), 2);
            assert_eq!(db.edge_count(), 1);

            let session = db.session();
            let result = session
                .execute("MATCH (n:Person {name: 'Alice'}) RETURN n.age")
                .unwrap();
            assert_eq!(result.rows.len(), 1);
            assert_eq!(result.rows[0][0], Value::Int64(30));

            db.close().expect("close");
        }
    }

    #[test]
    fn test_wal_recovery_labels() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("labels.grafeo");

        {
            let db = GrafeoDB::open(&path).expect("open");
            let n = db.create_node(&["Person"]);
            db.add_node_label(n, "Employee");
            db.close().expect("close");
        }

        {
            let db = GrafeoDB::open(&path).expect("reopen");
            let session = db.session();
            let result = session.execute("MATCH (n:Employee) RETURN n").unwrap();
            assert_eq!(result.rows.len(), 1, "label should survive WAL recovery");
            db.close().expect("close");
        }
    }

    #[test]
    fn test_wal_recovery_deletes() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("deletes.grafeo");

        {
            let db = GrafeoDB::open(&path).expect("open");
            let a = db.create_node(&["Person"]);
            db.set_node_property(a, "name", Value::String("Alice".into()));
            let b = db.create_node(&["Person"]);
            db.set_node_property(b, "name", Value::String("Bob".into()));
            let c = db.create_node(&["Person"]);
            db.set_node_property(c, "name", Value::String("Carol".into()));
            db.create_edge(a, b, "KNOWS");

            // Delete Carol (no edges)
            db.delete_node(c);
            db.close().expect("close");
        }

        {
            let db = GrafeoDB::open(&path).expect("reopen");
            assert_eq!(
                db.node_count(),
                2,
                "deleted node should not survive recovery"
            );
            assert_eq!(db.edge_count(), 1);
            db.close().expect("close");
        }
    }

    #[test]
    fn test_save_to_new_location() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let save_path = dir.path().join("saved.grafeo");

        let db = GrafeoDB::new_in_memory();
        let a = db.create_node(&["Person"]);
        db.set_node_property(a, "name", Value::String("Alice".into()));
        let b = db.create_node(&["Person"]);
        db.set_node_property(b, "name", Value::String("Bob".into()));
        db.create_edge(a, b, "KNOWS");

        // Save in-memory database to disk
        db.save(&save_path).expect("save should succeed");

        // Open the saved database
        let restored = GrafeoDB::open(&save_path).expect("open saved");
        assert_eq!(restored.node_count(), 2);
        assert_eq!(restored.edge_count(), 1);
        restored.close().expect("close");
    }

    #[test]
    fn test_open_in_memory_from_file() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("source.grafeo");

        // Create persistent database
        {
            let db = GrafeoDB::open(&path).expect("open");
            let n = db.create_node(&["Person"]);
            db.set_node_property(n, "name", Value::String("Alice".into()));
            db.close().expect("close");
        }

        // Load into memory (detached from file)
        let mem_db = GrafeoDB::open_in_memory(&path).expect("open_in_memory");
        assert_eq!(mem_db.node_count(), 1);
        assert!(
            !mem_db.is_persistent(),
            "should be in-memory after open_in_memory"
        );

        // Modifications to the in-memory copy don't affect the file
        mem_db.create_node(&["Extra"]);
        assert_eq!(mem_db.node_count(), 2);

        // Reopening the file should still show 1 node
        let file_db = GrafeoDB::open(&path).expect("reopen");
        assert_eq!(file_db.node_count(), 1);
        file_db.close().expect("close");
    }

    #[test]
    fn test_wal_status_persistent() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("status.grafeo");

        let db = GrafeoDB::open(&path).expect("open");
        db.create_node(&["N"]);

        let status = db.wal_status();
        assert!(status.enabled, "persistent db should have WAL enabled");

        db.close().expect("close");
    }

    #[test]
    fn test_wal_checkpoint_succeeds() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("checkpoint.grafeo");

        let db = GrafeoDB::open(&path).expect("open");
        let n = db.create_node(&["Person"]);
        db.set_node_property(n, "name", Value::String("Alice".into()));

        // Explicit checkpoint should not error
        db.wal_checkpoint().expect("checkpoint should succeed");

        // Verify data is still accessible after checkpoint
        assert_eq!(db.node_count(), 1);

        db.close().expect("close");
    }
}
