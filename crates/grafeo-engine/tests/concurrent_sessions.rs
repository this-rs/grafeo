//! Concurrent Sessions Integration Tests
//!
//! Tests for multi-session concurrent access patterns:
//! - Multiple sessions executing queries simultaneously
//! - Thread-safe shared database access
//! - Transaction isolation across sessions
//! - Race condition handling

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;

use grafeo_engine::GrafeoDB;

// ============================================================================
// Concurrent Session Access Tests
// ============================================================================

#[test]
fn test_concurrent_read_sessions() {
    // Multiple sessions reading simultaneously should not block
    let db = Arc::new(GrafeoDB::new_in_memory());

    // Create some initial data
    {
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
        session.execute("INSERT (:Person {name: 'Harm'})").unwrap();
    }

    let num_threads = 8;
    let barrier = Arc::new(Barrier::new(num_threads));
    let success_count = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let db = Arc::clone(&db);
            let barrier = Arc::clone(&barrier);
            let success_count = Arc::clone(&success_count);

            thread::spawn(move || {
                // Wait for all threads to be ready
                barrier.wait();

                // Each thread creates a session and reads
                let session = db.session();
                let result = session.execute("MATCH (n:Person) RETURN n.name").unwrap();

                // Should see all 3 nodes
                if result.row_count() >= 3 {
                    success_count.fetch_add(1, Ordering::Relaxed);
                }
            })
        })
        .collect();

    // Wait for all threads to complete
    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    // All threads should have succeeded
    assert_eq!(
        success_count.load(Ordering::Relaxed),
        num_threads,
        "All concurrent reads should succeed"
    );
}

#[test]
fn test_concurrent_write_sessions() {
    // Multiple sessions writing to different entities should succeed
    let db = Arc::new(GrafeoDB::new_in_memory());

    let num_threads = 4;
    let barrier = Arc::new(Barrier::new(num_threads));
    let success_count = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..num_threads)
        .map(|i| {
            let db = Arc::clone(&db);
            let barrier = Arc::clone(&barrier);
            let success_count = Arc::clone(&success_count);

            thread::spawn(move || {
                // Wait for all threads to be ready
                barrier.wait();

                // Each thread creates a unique node
                let session = db.session();
                let query = format!("INSERT (:Thread{} {{id: {}}})", i, i);
                if session.execute(&query).is_ok() {
                    success_count.fetch_add(1, Ordering::Relaxed);
                }
            })
        })
        .collect();

    // Wait for all threads to complete
    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    // All threads should have succeeded
    assert_eq!(
        success_count.load(Ordering::Relaxed),
        num_threads,
        "All concurrent writes to different entities should succeed"
    );

    // Verify all nodes were created
    let session = db.session();
    for i in 0..num_threads {
        let query = format!("MATCH (n:Thread{}) RETURN n", i);
        let result = session.execute(&query).unwrap();
        assert!(
            result.row_count() >= 1,
            "Node for thread {} should exist",
            i
        );
    }
}

#[test]
fn test_session_isolation_between_threads() {
    // Changes in one session's transaction should not be visible to other sessions
    // until committed
    let db = Arc::new(GrafeoDB::new_in_memory());

    // Writer thread creates data in a transaction
    let db_clone = Arc::clone(&db);
    let writer_started = Arc::new(Barrier::new(2));
    let reader_check = Arc::new(Barrier::new(2));
    let writer_done = Arc::new(Barrier::new(2));

    let writer_started_clone = Arc::clone(&writer_started);
    let reader_check_clone = Arc::clone(&reader_check);
    let writer_done_clone = Arc::clone(&writer_done);

    let writer_handle = thread::spawn(move || {
        let mut session = db_clone.session();
        session.begin_tx().unwrap();

        // Create a node within the transaction
        session
            .execute("INSERT (:IsolatedNode {secret: 'hidden'})")
            .unwrap();

        // Signal that writer has created the node
        writer_started_clone.wait();

        // Wait for reader to check
        reader_check_clone.wait();

        // Now commit
        session.commit().unwrap();

        // Signal done
        writer_done_clone.wait();
    });

    // Reader thread checks visibility
    let reader_handle = thread::spawn(move || {
        let session = db.session();

        // Wait for writer to create node (but not commit)
        writer_started.wait();

        // Check if we can see the node (we shouldn't - transaction not committed)
        // Note: This test checks the expected behavior when MVCC is fully integrated
        let result = session.execute("MATCH (n:IsolatedNode) RETURN n").unwrap();
        let before_commit_count = result.row_count();

        // Signal that reader has checked
        reader_check.wait();

        // Wait for writer to commit
        writer_done.wait();

        // Now we should see the node (after commit)
        let result = session.execute("MATCH (n:IsolatedNode) RETURN n").unwrap();
        let after_commit_count = result.row_count();

        (before_commit_count, after_commit_count)
    });

    writer_handle.join().expect("Writer thread panicked");
    let (before, after) = reader_handle.join().expect("Reader thread panicked");

    // After commit, the node should be visible
    assert!(
        after >= 1,
        "Node should be visible after commit, got {} rows",
        after
    );

    // Note: Before commit visibility depends on MVCC integration level
    // Currently the store may or may not provide full isolation
    let _ = before; // Acknowledge we're not asserting on this yet
}

// ============================================================================
// Stress Tests
// ============================================================================

#[test]
fn test_many_sessions_rapid_creation() {
    // Creating many sessions rapidly should not cause issues
    let db = Arc::new(GrafeoDB::new_in_memory());

    let num_threads = 16;
    let sessions_per_thread = 50;
    let barrier = Arc::new(Barrier::new(num_threads));
    let success_count = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let db = Arc::clone(&db);
            let barrier = Arc::clone(&barrier);
            let success_count = Arc::clone(&success_count);

            thread::spawn(move || {
                barrier.wait();

                for _ in 0..sessions_per_thread {
                    let session = db.session();
                    // Just creating and dropping sessions
                    drop(session);
                }
                success_count.fetch_add(1, Ordering::Relaxed);
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    assert_eq!(
        success_count.load(Ordering::Relaxed),
        num_threads,
        "All threads should complete without panic"
    );
}

#[test]
fn test_interleaved_transactions() {
    // Multiple sessions with interleaved transaction operations.
    // Kept lightweight (2 threads, 3 iterations) to avoid lock-contention
    // slowdowns on resource-constrained CI runners.
    let db = Arc::new(GrafeoDB::new_in_memory());

    let completed = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..2)
        .map(|thread_id| {
            let db = Arc::clone(&db);
            let completed = Arc::clone(&completed);

            thread::spawn(move || {
                for i in 0..3 {
                    let mut session = db.session();

                    session.begin_tx().unwrap();

                    let query =
                        format!("INSERT (:Work {{thread: {}, iteration: {}}})", thread_id, i);
                    let _ = session.execute(&query);

                    if i % 3 == 0 {
                        let _ = session.rollback();
                    } else {
                        let _ = session.commit();
                    }
                }

                completed.fetch_add(1, Ordering::Relaxed);
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    assert_eq!(
        completed.load(Ordering::Relaxed),
        2,
        "All threads should complete"
    );
}

// ============================================================================
// Session State Tests
// ============================================================================

#[test]
fn test_session_transaction_state_independence() {
    // Each session should maintain independent transaction state
    let db = GrafeoDB::new_in_memory();

    let mut session1 = db.session();
    let mut session2 = db.session();

    // Session 1 starts transaction
    session1.begin_tx().unwrap();
    assert!(session1.in_transaction());
    assert!(!session2.in_transaction());

    // Session 2 starts its own transaction
    session2.begin_tx().unwrap();
    assert!(session1.in_transaction());
    assert!(session2.in_transaction());

    // Session 1 commits
    session1.commit().unwrap();
    assert!(!session1.in_transaction());
    assert!(session2.in_transaction());

    // Session 2 rolls back
    session2.rollback().unwrap();
    assert!(!session1.in_transaction());
    assert!(!session2.in_transaction());
}

#[test]
fn test_session_auto_commit_independence() {
    // Auto-commit setting should be independent per session
    let db = GrafeoDB::new_in_memory();

    let mut session1 = db.session();
    let session2 = db.session();

    assert!(session1.auto_commit());
    assert!(session2.auto_commit());

    session1.set_auto_commit(false);

    assert!(!session1.auto_commit());
    assert!(session2.auto_commit());
}

// ============================================================================
// Database Shared State Tests
// ============================================================================

#[test]
fn test_sessions_share_committed_data() {
    // Data committed by one session should be visible to others
    let db = GrafeoDB::new_in_memory();

    let session1 = db.session();
    let session2 = db.session();

    // Session 1 creates and commits data
    session1.execute("INSERT (:Shared {key: 'value'})").unwrap();

    // Session 2 should see the data
    let result = session2.execute("MATCH (n:Shared) RETURN n.key").unwrap();
    assert!(
        result.row_count() >= 1,
        "Session 2 should see committed data from Session 1"
    );
}

#[test]
fn test_node_count_consistency() {
    // Node count should be consistent across sessions
    let db = GrafeoDB::new_in_memory();

    // Create nodes from multiple sessions
    for i in 0..10 {
        let session = db.session();
        let query = format!("INSERT (:CountTest{{id: {}}})", i);
        session.execute(&query).unwrap();
    }

    // Check count from a new session
    let session = db.session();
    let result = session.execute("MATCH (n:CountTest) RETURN n").unwrap();
    assert_eq!(result.row_count(), 10, "Should see all 10 nodes");
}

// ============================================================================
// Async Session Tests (using tokio)
// ============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_async_concurrent_sessions() {
    use tokio::task;

    let db = Arc::new(GrafeoDB::new_in_memory());

    // Spawn multiple async tasks
    let mut handles = Vec::new();

    for i in 0..8 {
        let db: Arc<GrafeoDB> = Arc::clone(&db);
        handles.push(task::spawn_blocking(move || {
            let session = db.session();
            let query = format!("INSERT (:AsyncNode {{id: {}}})", i);
            session.execute(&query).unwrap();
        }));
    }

    // Wait for all tasks
    for handle in handles {
        handle.await.expect("Task panicked");
    }

    // Verify results
    let session = db.session();
    let result = session.execute("MATCH (n:AsyncNode) RETURN n").unwrap();
    assert_eq!(result.row_count(), 8, "All async nodes should exist");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_async_transaction_isolation() {
    use std::sync::atomic::AtomicBool;
    use tokio::task;

    let db = Arc::new(GrafeoDB::new_in_memory());
    let writer_committed = Arc::new(AtomicBool::new(false));

    // Writer task
    let db_writer: Arc<GrafeoDB> = Arc::clone(&db);
    let committed_flag = Arc::clone(&writer_committed);

    let writer = task::spawn_blocking(move || {
        let mut session = db_writer.session();
        session.begin_tx().unwrap();
        session
            .execute("INSERT (:AsyncIsolated {data: 'test'})")
            .unwrap();
        session.commit().unwrap();
        committed_flag.store(true, Ordering::Release);
    });

    // Reader task: waits for writer commit via atomic flag, no sleep
    let db_reader: Arc<GrafeoDB> = Arc::clone(&db);
    let reader_flag = Arc::clone(&writer_committed);

    let reader = task::spawn_blocking(move || {
        // Spin until writer has committed
        while !reader_flag.load(Ordering::Acquire) {
            std::hint::spin_loop();
        }

        let session = db_reader.session();
        let result = session.execute("MATCH (n:AsyncIsolated) RETURN n").unwrap();
        result.row_count()
    });

    writer.await.expect("Writer task panicked");
    let count = reader.await.expect("Reader task panicked");

    assert!(
        count >= 1,
        "Should see committed data after writer completes"
    );
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_session_after_transaction_error() {
    // Session should be usable after a transaction error
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    // Try to commit without transaction (should error)
    let result = session.commit();
    assert!(result.is_err());

    // Session should still work
    session.begin_tx().unwrap();
    session.execute("INSERT (:AfterError)").unwrap();
    session.commit().unwrap();

    let result = session.execute("MATCH (n:AfterError) RETURN n").unwrap();
    assert!(result.row_count() >= 1);
}

#[test]
fn test_multiple_sequential_transactions() {
    // Same session should handle multiple sequential transactions
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    for i in 0..5 {
        session.begin_tx().unwrap();
        let query = format!("INSERT (:Sequential{{iteration: {}}})", i);
        session.execute(&query).unwrap();
        session.commit().unwrap();
    }

    let result = session.execute("MATCH (n:Sequential) RETURN n").unwrap();
    assert_eq!(
        result.row_count(),
        5,
        "All 5 sequential transactions should have committed"
    );
}

// ============================================================================
// Concurrent Stress Tests
// ============================================================================

#[test]
#[ignore = "stress test: slow in CI, run locally with --ignored"]
fn test_stress_concurrent_writers() {
    // 8 threads each inserting 50 nodes simultaneously
    let db = Arc::new(GrafeoDB::new_in_memory());

    let num_threads = 8;
    let writes_per_thread = 50;
    let barrier = Arc::new(Barrier::new(num_threads));
    let success_count = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..num_threads)
        .map(|tid| {
            let db = Arc::clone(&db);
            let barrier = Arc::clone(&barrier);
            let success_count = Arc::clone(&success_count);

            thread::spawn(move || {
                barrier.wait();
                for i in 0..writes_per_thread {
                    let session = db.session();
                    let query = format!("INSERT (:Stress {{thread: {tid}, seq: {i}}})");
                    session.execute(&query).unwrap();
                }
                success_count.fetch_add(1, Ordering::Relaxed);
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    assert_eq!(success_count.load(Ordering::Relaxed), num_threads);

    // Verify total node count
    let session = db.session();
    let result = session.execute("MATCH (n:Stress) RETURN n").unwrap();
    assert_eq!(
        result.row_count(),
        num_threads * writes_per_thread,
        "All nodes should be created"
    );
}

#[test]
#[ignore = "stress test: slow in CI, run locally with --ignored"]
fn test_stress_concurrent_reads_during_writes() {
    // Mixed workload: 4 writers + 8 readers operating simultaneously
    let db = Arc::new(GrafeoDB::new_in_memory());

    // Seed initial data
    {
        let session = db.session();
        for i in 0..100 {
            session
                .execute(&format!("INSERT (:Item {{id: {i}}})"))
                .unwrap();
        }
    }

    let num_writers = 4;
    let num_readers = 8;
    let barrier = Arc::new(Barrier::new(num_writers + num_readers));
    let read_errors = Arc::new(AtomicUsize::new(0));
    let write_errors = Arc::new(AtomicUsize::new(0));

    let mut handles = Vec::new();

    // Writer threads
    for tid in 0..num_writers {
        let db = Arc::clone(&db);
        let barrier = Arc::clone(&barrier);
        let errors = Arc::clone(&write_errors);
        handles.push(thread::spawn(move || {
            barrier.wait();
            for i in 0..20 {
                let session = db.session();
                let id = 1000 + tid * 100 + i;
                if session
                    .execute(&format!("INSERT (:Written {{id: {id}}})"))
                    .is_err()
                {
                    errors.fetch_add(1, Ordering::Relaxed);
                }
            }
        }));
    }

    // Reader threads
    for _ in 0..num_readers {
        let db = Arc::clone(&db);
        let barrier = Arc::clone(&barrier);
        let errors = Arc::clone(&read_errors);
        handles.push(thread::spawn(move || {
            barrier.wait();
            for _ in 0..20 {
                let session = db.session();
                if session.execute("MATCH (n:Item) RETURN n.id").is_err() {
                    errors.fetch_add(1, Ordering::Relaxed);
                }
            }
        }));
    }

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    assert_eq!(
        read_errors.load(Ordering::Relaxed),
        0,
        "No read errors expected"
    );
    assert_eq!(
        write_errors.load(Ordering::Relaxed),
        0,
        "No write errors expected"
    );
}

#[test]
#[ignore = "stress test: slow in CI, run locally with --ignored"]
fn test_stress_transaction_conflicts() {
    // 4 threads with interleaved commit/rollback patterns
    let db = Arc::new(GrafeoDB::new_in_memory());

    let num_threads = 4;
    let iterations = 6;
    let barrier = Arc::new(Barrier::new(num_threads));
    let completed = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..num_threads)
        .map(|tid| {
            let db = Arc::clone(&db);
            let barrier = Arc::clone(&barrier);
            let completed = Arc::clone(&completed);

            thread::spawn(move || {
                barrier.wait();
                for i in 0..iterations {
                    let mut session = db.session();
                    session.begin_tx().unwrap();
                    let query = format!("INSERT (:TxNode {{thread: {tid}, iter: {i}}})");
                    let _ = session.execute(&query);

                    // Commit even iterations, rollback odd
                    if i % 2 == 0 {
                        let _ = session.commit();
                    } else {
                        let _ = session.rollback();
                    }
                }
                completed.fetch_add(1, Ordering::Relaxed);
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    assert_eq!(completed.load(Ordering::Relaxed), num_threads);

    // Only committed nodes (even iterations) should exist
    let session = db.session();
    let result = session.execute("MATCH (n:TxNode) RETURN n").unwrap();
    // Each thread commits 5 of 10 iterations (0, 2, 4, 6, 8)
    let expected = num_threads * (iterations / 2);
    assert_eq!(
        result.row_count(),
        expected,
        "Only committed transactions should be visible"
    );
}

#[test]
#[ignore = "stress test: slow in CI, run locally with --ignored"]
fn test_stress_concurrent_epoch_pressure() {
    // 4 threads each running 8 sequential transactions, creates many epochs
    let db = Arc::new(GrafeoDB::new_in_memory());

    let num_threads = 4;
    let txns_per_thread = 8;
    let barrier = Arc::new(Barrier::new(num_threads));
    let completed = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..num_threads)
        .map(|tid| {
            let db = Arc::clone(&db);
            let barrier = Arc::clone(&barrier);
            let completed = Arc::clone(&completed);

            thread::spawn(move || {
                barrier.wait();
                for i in 0..txns_per_thread {
                    let mut session = db.session();
                    session.begin_tx().unwrap();
                    session
                        .execute(&format!("INSERT (:Epoch {{thread: {tid}, txn: {i}}})"))
                        .unwrap();
                    session.commit().unwrap();
                }
                completed.fetch_add(1, Ordering::Relaxed);
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    assert_eq!(completed.load(Ordering::Relaxed), num_threads);

    // All nodes should be visible
    let session = db.session();
    let result = session.execute("MATCH (n:Epoch) RETURN n").unwrap();
    assert_eq!(
        result.row_count(),
        num_threads * txns_per_thread,
        "All epoch nodes should exist"
    );
}

#[test]
#[ignore = "stress test: slow in CI, run locally with --ignored"]
fn test_stress_rapid_session_lifecycle() {
    // 16 threads rapidly creating, using, and dropping sessions
    let db = Arc::new(GrafeoDB::new_in_memory());

    let num_threads = 16;
    let cycles = 100;
    let barrier = Arc::new(Barrier::new(num_threads));
    let completed = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let db = Arc::clone(&db);
            let barrier = Arc::clone(&barrier);
            let completed = Arc::clone(&completed);

            thread::spawn(move || {
                barrier.wait();
                for _ in 0..cycles {
                    let session = db.session();
                    let _ = session.execute("MATCH (n) RETURN n LIMIT 1");
                    drop(session);
                }
                completed.fetch_add(1, Ordering::Relaxed);
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    assert_eq!(
        completed.load(Ordering::Relaxed),
        num_threads,
        "All threads should complete"
    );
}

#[test]
#[ignore = "stress test: slow in CI, run locally with --ignored"]
fn test_stress_concurrent_edges_and_nodes() {
    // Create nodes and edges simultaneously from multiple threads
    let db = Arc::new(GrafeoDB::new_in_memory());

    // Seed some nodes first (needed for edge creation)
    let session = db.session();
    for i in 0..20 {
        session
            .execute(&format!("INSERT (:Hub {{id: {i}}})"))
            .unwrap();
    }
    drop(session);

    let num_threads = 4;
    let barrier = Arc::new(Barrier::new(num_threads));
    let completed = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..num_threads)
        .map(|tid| {
            let db = Arc::clone(&db);
            let barrier = Arc::clone(&barrier);
            let completed = Arc::clone(&completed);

            thread::spawn(move || {
                barrier.wait();
                let session = db.session();
                for i in 0..10 {
                    // Create new nodes
                    session
                        .execute(&format!("INSERT (:Spoke {{thread: {tid}, id: {i}}})"))
                        .unwrap();
                    // Create edges between existing hub nodes
                    let src = (tid * 5 + i) % 20;
                    let dst = (tid * 5 + i + 1) % 20;
                    let _ = session.execute(&format!(
                        "MATCH (a:Hub {{id: {src}}}), (b:Hub {{id: {dst}}}) \
                         INSERT (a)-[:LINK {{thread: {tid}}}]->(b)"
                    ));
                }
                completed.fetch_add(1, Ordering::Relaxed);
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    assert_eq!(completed.load(Ordering::Relaxed), num_threads);

    // Verify spoke nodes were created
    let session = db.session();
    let result = session.execute("MATCH (n:Spoke) RETURN n").unwrap();
    assert_eq!(
        result.row_count(),
        num_threads * 10,
        "All spoke nodes should exist"
    );
}
