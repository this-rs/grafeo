//! T17h T9b — variance tests for the Cypher planner rewrite.
//!
//! Each test runs a query that is **close to** the canonical
//! `most_connected_files` pattern but deviates in exactly one aspect.
//! The matcher must FALL BACK (counter == 0) and the slow path must
//! produce correct results (no panic, reasonable row count).
//!
//! Tri-state expectation :
//!
//! - Before T9c : counter never increments on anything (no matcher).
//!   All 5 tests trivially green since the fallback is the current
//!   behaviour.
//! - After T9c : counter must still stay at 0 on every variant here
//!   (the matcher rejects them). All 5 tests stay green — this is the
//!   regression anchor against false-positive matching.
//!
//! Run with :
//!
//! ```bash
//! cargo test -p obrain-engine --release --features cypher \
//!   --test t9_planner_rewrite_variance -- --nocapture
//! ```

#![cfg(feature = "cypher")]

use obrain_core::execution::operators::{
    reset_typed_degree_rewrite_counter, typed_degree_rewrite_counter,
};
use obrain_engine::ObrainDB;

fn open_po_or_skip() -> Option<ObrainDB> {
    let home = std::env::var("HOME").unwrap_or_default();
    let po_path = std::path::PathBuf::from(&home).join(".obrain/db/po");
    if !po_path.exists() {
        eprintln!("⏭  ~/.obrain/db/po not present — skipping T9b variance tests");
        return None;
    }
    Some(ObrainDB::open(&po_path).expect("open PO db"))
}

/// Variant 1 — intermediate `WHERE` between OPTIONAL MATCH and WITH.
/// A `Filter` node gets injected in the plan ; the matcher must
/// decline to avoid stripping the predicate.
#[test]
fn variant_intermediate_where_falls_back() {
    let Some(db) = open_po_or_skip() else { return };
    let session = db.session();
    reset_typed_degree_rewrite_counter();
    let res = session
        .execute_cypher(
            "MATCH (f:File) \
             OPTIONAL MATCH (f)-[:IMPORTS]->(imported:File) \
             WHERE imported.path STARTS WITH 'crates/' \
             WITH f, count(DISTINCT imported) AS imports \
             RETURN f.path, imports ORDER BY imports DESC LIMIT 10",
        )
        .expect("query succeeds");
    let _ = res.rows.len();
    assert_eq!(
        typed_degree_rewrite_counter(),
        0,
        "intermediate WHERE MUST prevent the rewrite"
    );
}

/// Variant 2 — three OPTIONAL MATCH branches.  The matcher supports
/// at most 2 (Separate direction) ; a third branch is out of scope
/// and must fall back.
#[test]
fn variant_three_optional_matches_falls_back() {
    let Some(db) = open_po_or_skip() else { return };
    let session = db.session();
    reset_typed_degree_rewrite_counter();
    let res = session
        .execute_cypher(
            "MATCH (f:File) \
             OPTIONAL MATCH (f)-[:IMPORTS]->(imp:File) \
             OPTIONAL MATCH (dep:File)-[:IMPORTS]->(f) \
             OPTIONAL MATCH (f)-[:CONTAINS]->(fn:Function) \
             WITH f, count(DISTINCT imp) AS i, count(DISTINCT dep) AS d, count(DISTINCT fn) AS c \
             RETURN f.path, i, d, c ORDER BY i DESC LIMIT 10",
        )
        .expect("query succeeds");
    let _ = res.rows.len();
    assert_eq!(
        typed_degree_rewrite_counter(),
        0,
        "three OPTIONAL MATCH branches MUST fall back"
    );
}

/// Variant 3 — aggregate function other than `count` (e.g. `sum`).
/// The matcher only accepts `CountNonNull` ; any other agg must
/// decline.
#[test]
fn variant_sum_aggregate_falls_back() {
    let Some(db) = open_po_or_skip() else { return };
    let session = db.session();
    reset_typed_degree_rewrite_counter();
    // `sum(imp.line_count)` — an arbitrary property. We only care
    // about the agg function being non-count ; row semantics are
    // irrelevant here.
    let res = session
        .execute_cypher(
            "MATCH (f:File) \
             OPTIONAL MATCH (f)-[:IMPORTS]->(imp:File) \
             WITH f, sum(imp.line_count) AS total \
             RETURN f.path, total ORDER BY total DESC LIMIT 10",
        )
        .expect("query succeeds");
    let _ = res.rows.len();
    assert_eq!(
        typed_degree_rewrite_counter(),
        0,
        "sum() aggregate MUST prevent the rewrite"
    );
}

/// Variant 4 — `ORDER BY ASC` instead of `DESC`. Top-K with smallest
/// degree at the head is semantically a different operation — the
/// matcher must reject it to avoid emitting a reversed result set.
#[test]
fn variant_order_by_asc_falls_back() {
    let Some(db) = open_po_or_skip() else { return };
    let session = db.session();
    reset_typed_degree_rewrite_counter();
    let res = session
        .execute_cypher(
            "MATCH (f:File) \
             OPTIONAL MATCH (f)-[:IMPORTS]->(imp:File) \
             WITH f, count(DISTINCT imp) AS c \
             RETURN f.path, c ORDER BY c ASC LIMIT 10",
        )
        .expect("query succeeds");
    let _ = res.rows.len();
    assert_eq!(
        typed_degree_rewrite_counter(),
        0,
        "ORDER BY ASC MUST prevent the rewrite"
    );
}

/// Variant 5 — `count(*)` (without DISTINCT). The matcher requires
/// `count(DISTINCT var)` since that's what compiles to
/// `CountNonNull + distinct: true` which equals the typed-degree
/// semantics only under the unicity-per-pair invariant. `count(*)`
/// counts rows including duplicates from the LeftJoin expansion and
/// MUST stay on the slow path.
#[test]
fn variant_count_star_falls_back() {
    let Some(db) = open_po_or_skip() else { return };
    let session = db.session();
    reset_typed_degree_rewrite_counter();
    let res = session
        .execute_cypher(
            "MATCH (f:File) \
             OPTIONAL MATCH (f)-[:IMPORTS]->(imp:File) \
             WITH f, count(*) AS c \
             RETURN f.path, c ORDER BY c DESC LIMIT 10",
        )
        .expect("query succeeds");
    let _ = res.rows.len();
    assert_eq!(
        typed_degree_rewrite_counter(),
        0,
        "count(*) (non-DISTINCT) MUST prevent the rewrite"
    );
}
