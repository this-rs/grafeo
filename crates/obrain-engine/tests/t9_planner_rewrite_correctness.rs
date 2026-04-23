//! T17h T9b — row correctness and bench gate tests for the Cypher
//! planner rewrite.
//!
//! Compares the output of the canonical `most_connected_files` query
//! with and without the T9 rewrite active. Correctness is asserted
//! row-by-row : the 50 rows produced by the rewrite MUST match the
//! slow-path output exactly (order + values).
//!
//! The kill switch `OBRAIN_DISABLE_TYPED_DEGREE_TOPK=1` is set inside
//! the test to force the slow path on the reference run. T9c reads
//! that env var as the first thing in `try_plan_typed_degree_topk`.
//!
//! Tri-state expectation :
//!
//! - Before T9c : both runs are identical (the env var has no effect
//!   because no rewrite exists), so the `assert_eq` trivially passes.
//!   The bench test also passes trivially because the slow-path
//!   latency is measured without comparison to a rewrite.
//! - After T9c : the rewrite run returns in ≤ 30 ms (bench gate), the
//!   slow run takes ~245 ms on PO, and the 50 rows match exactly.
//!
//! Run with :
//!
//! ```bash
//! cargo test -p obrain-engine --release --features cypher \
//!   --test t9_planner_rewrite_correctness -- --nocapture
//! ```

#![cfg(feature = "cypher")]

use obrain_common::types::Value;
use obrain_engine::ObrainDB;
use std::time::Instant;

const CANONICAL_QUERY: &str = "MATCH (f:File) \
    OPTIONAL MATCH (f)-[:IMPORTS]->(imported:File) \
    OPTIONAL MATCH (dependent:File)-[:IMPORTS]->(f) \
    WITH f, count(DISTINCT imported) AS imports, count(DISTINCT dependent) AS dependents \
    RETURN f.path, imports, dependents, imports + dependents AS connections \
    ORDER BY connections DESC \
    LIMIT 50";

fn open_po_or_skip() -> Option<ObrainDB> {
    let home = std::env::var("HOME").unwrap_or_default();
    let po_path = std::path::PathBuf::from(&home).join(".obrain/db/po");
    if !po_path.exists() {
        eprintln!("⏭  ~/.obrain/db/po not present — skipping T9b correctness tests");
        return None;
    }
    Some(ObrainDB::open(&po_path).expect("open PO db"))
}

/// Fingerprint a row as a tuple of `(path, imports, dependents,
/// connections)` so `assert_eq!` prints legible diffs.
fn fingerprint(row: &[Value]) -> (String, i64, i64, i64) {
    let path = match row.first() {
        Some(Value::String(s)) => s.to_string(),
        other => format!("{other:?}"),
    };
    let imports = match row.get(1) {
        Some(Value::Int64(v)) => *v,
        _ => -1,
    };
    let dependents = match row.get(2) {
        Some(Value::Int64(v)) => *v,
        _ => -1,
    };
    let connections = match row.get(3) {
        Some(Value::Int64(v)) => *v,
        _ => -1,
    };
    (path, imports, dependents, connections)
}

#[test]
#[ignore = "T9c-gated: activates once try_plan_typed_degree_topk + OBRAIN_DISABLE_TYPED_DEGREE_TOPK land"]
fn row_correctness_rewrite_vs_slow_path() {
    let Some(db) = open_po_or_skip() else { return };
    let session = db.session();

    // Run 1 — slow path forced via env var.
    // SAFETY : set_var / remove_var are unsafe in rust 2024 edition (race-
    // hostile for multi-threaded runtimes). Cargo runs integration tests
    // per-binary ; we don't spawn threads here.
    // SAFETY: cargo runs integration tests single-binary — no concurrent env
    // access in this test scope.
    unsafe {
        std::env::set_var("OBRAIN_DISABLE_TYPED_DEGREE_TOPK", "1");
    }
    let slow = session
        .execute_cypher(CANONICAL_QUERY)
        .expect("slow-path query runs");
    unsafe {
        std::env::remove_var("OBRAIN_DISABLE_TYPED_DEGREE_TOPK");
    }

    // Run 2 — rewrite path.
    let fast = session
        .execute_cypher(CANONICAL_QUERY)
        .expect("rewrite-path query runs");

    assert_eq!(slow.rows.len(), fast.rows.len(), "row count must match");
    assert_eq!(slow.rows.len(), 50, "PO should return 50 rows");

    let slow_fp: Vec<_> = slow.rows.iter().map(|r| fingerprint(r)).collect();
    let fast_fp: Vec<_> = fast.rows.iter().map(|r| fingerprint(r)).collect();

    for (i, (s, f)) in slow_fp.iter().zip(fast_fp.iter()).enumerate() {
        assert_eq!(
            s, f,
            "row {i} differs : slow={s:?} fast={f:?} — rewrite produced a \
             semantically different result, audit extract_typed_degree_pattern"
        );
    }
}

#[test]
#[ignore = "T9c-gated: activates once try_plan_typed_degree_topk lands and hits the gate"]
fn bench_gate_most_connected_under_30ms() {
    let Some(db) = open_po_or_skip() else { return };
    let session = db.session();

    // 5 warmup iterations.
    for _ in 0..5 {
        let _ = session
            .execute_cypher(CANONICAL_QUERY)
            .expect("warmup run");
    }

    // 10 measured iterations. Compute p50 (median) for the assertion.
    let mut samples_ms: Vec<f64> = Vec::with_capacity(10);
    for _ in 0..10 {
        let t0 = Instant::now();
        let res = session
            .execute_cypher(CANONICAL_QUERY)
            .expect("bench run");
        samples_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
        assert_eq!(res.rows.len(), 50);
    }
    samples_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = samples_ms[samples_ms.len() / 2];
    let min = samples_ms.first().copied().unwrap_or(f64::INFINITY);
    let max = samples_ms.last().copied().unwrap_or(0.0);
    eprintln!(
        "T9 bench gate : most_connected_files TOP 50 — p50={:.2}ms min={:.2}ms max={:.2}ms",
        p50, min, max
    );
    assert!(
        p50 <= 30.0,
        "T7 gate violated : most_connected_files p50 = {p50:.2} ms > 30 ms (Neo4j 31 ms). \
         Check that the T9 rewrite is actually routing this query."
    );
}
