//! Crash injection tests for the single-file `.grafeo` format.
//!
//! These tests simulate crashes at deterministic points during checkpoint and
//! verify that the database recovers correctly on the next open.
//!
//! Requires both `grafeo-file` and `testing-crash-injection` features.

#![cfg(all(feature = "grafeo-file", feature = "testing-crash-injection"))]

use std::panic::AssertUnwindSafe;

use grafeo_common::types::Value;
use grafeo_core::testing::crash::{CrashResult, with_crash_at};
use grafeo_engine::{Config, GrafeoDB};

/// Helper: extract sorted string values from column 0 of query result rows.
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

// =========================================================================
// Crash during checkpoint_to_file (close path)
// =========================================================================

#[test]
fn crash_during_close_checkpoint_preserves_data_via_sidecar_wal() {
    // There are 8 crash points in the checkpoint path:
    //   checkpoint_to_file: before_export, after_export, after_write_snapshot
    //   write_snapshot: before_data_write, after_data_write, after_truncate,
    //                   after_header_write, after_fsync
    //
    // For each crash point, we:
    // 1. Create a DB, insert data
    // 2. Crash during close
    // 3. Reopen and verify data survived (via sidecar WAL replay)

    for crash_point in 1..=8 {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("crash_test.grafeo");

        // Phase 1: Create and populate
        {
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

            // Crash during close
            let db = AssertUnwindSafe(db);
            let result = with_crash_at(crash_point, move || {
                let _ = db.close();
            });

            // Some crash points may complete normally (if the crash counter
            // exceeds the number of maybe_crash calls before close finishes)
            match result {
                CrashResult::Crashed => {
                    // Sidecar WAL should exist (crash prevented cleanup)
                    // The file may or may not have a valid snapshot
                }
                CrashResult::Completed(()) => {
                    // Close completed: the crash point was past all crash
                    // injection calls. Data should be in the .grafeo file.
                }
            }
        }

        // Phase 2: Reopen and verify data survived
        if path.exists() {
            let db = GrafeoDB::open(&path).unwrap();
            let session = db.session();

            let result = session.execute("MATCH (p:Person) RETURN p.name").unwrap();
            let names = extract_strings(&result.rows);
            assert_eq!(
                names,
                vec!["Alix", "Gus"],
                "crash_point={crash_point}: data lost after crash"
            );

            assert_eq!(
                db.edge_count(),
                1,
                "crash_point={crash_point}: edge lost after crash"
            );

            db.close().unwrap();
        }
    }
}

// =========================================================================
// Crash during explicit wal_checkpoint
// =========================================================================

#[test]
fn crash_during_wal_checkpoint_leaves_db_usable() {
    for crash_point in 1..=8 {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("wal_crash.grafeo");

        // Create and populate
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let session = db.session();
        session
            .execute("INSERT (:City {name: 'Amsterdam'})")
            .unwrap();
        session.execute("INSERT (:City {name: 'Berlin'})").unwrap();

        // Crash during explicit checkpoint (db stays open)
        let db_ref = AssertUnwindSafe(&db);
        let result = with_crash_at(crash_point, move || {
            let _ = db_ref.wal_checkpoint();
        });

        match result {
            CrashResult::Crashed => {
                // DB may be in an inconsistent internal state after panic,
                // but the in-memory data should still be present
            }
            CrashResult::Completed(()) => {
                // Checkpoint completed successfully
            }
        }

        // Drop without close to simulate process exit
        drop(db);

        // Reopen: data should survive via sidecar WAL replay
        if path.exists() {
            let db2 = GrafeoDB::open(&path).unwrap();
            let session2 = db2.session();

            let result = session2.execute("MATCH (c:City) RETURN c.name").unwrap();
            let names = extract_strings(&result.rows);
            assert_eq!(
                names,
                vec!["Amsterdam", "Berlin"],
                "crash_point={crash_point}: data lost after checkpoint crash"
            );

            db2.close().unwrap();
        }
    }
}

// =========================================================================
// Crash after first checkpoint, then more writes
// =========================================================================

#[test]
fn crash_after_successful_checkpoint_with_new_writes() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("incremental.grafeo");

    // Phase 1: Create, populate, and successfully checkpoint
    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        db.wal_checkpoint().unwrap();

        // Phase 2: Add more data
        session.execute("INSERT (:Person {name: 'Gus'})").unwrap();

        // Phase 3: Crash during close (after more writes)
        let db = AssertUnwindSafe(db);
        let _result = with_crash_at(1, move || {
            let _ = db.close();
        });
    }

    // Reopen: at minimum, pre-checkpoint data should survive.
    // If sidecar WAL was written before crash, post-checkpoint data may also survive.
    let db = GrafeoDB::open(&path).unwrap();
    let session = db.session();

    let result = session.execute("MATCH (p:Person) RETURN p.name").unwrap();
    let names = extract_strings(&result.rows);

    // Pre-checkpoint data must survive
    assert!(
        names.contains(&"Alix".to_string()),
        "pre-checkpoint data lost"
    );

    db.close().unwrap();
}

// =========================================================================
// Multiple checkpoint-crash-recover cycles
// =========================================================================

#[test]
fn repeated_crash_recovery_cycles() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("cycles.grafeo");

    let people = ["Alix", "Gus", "Vincent", "Jules", "Mia"];

    for (i, name) in people.iter().enumerate() {
        // Open (or create on first iteration)
        let db = if i == 0 {
            GrafeoDB::with_config(Config::persistent(&path)).unwrap()
        } else {
            GrafeoDB::open(&path).unwrap()
        };

        let session = db.session();
        session
            .execute(&format!("INSERT (:Person {{name: '{name}'}})"))
            .unwrap();

        // Alternate between clean close and crash
        if i % 2 == 0 {
            db.close().unwrap();
        } else {
            let db = AssertUnwindSafe(db);
            let _result = with_crash_at(2, move || {
                let _ = db.close();
            });
        }
    }

    // Final verification
    let db = GrafeoDB::open(&path).unwrap();
    let count = db.node_count();
    // At minimum the cleanly-closed sessions' data should persist
    assert!(count >= 3, "expected at least 3 nodes, got {count}");
    db.close().unwrap();
}
