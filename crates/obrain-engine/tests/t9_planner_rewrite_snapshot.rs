//! T17h T9b — snapshot tests for the Cypher planner rewrite
//! (`try_plan_typed_degree_topk`).
//!
//! ### Why counter-based rather than EXPLAIN-based ?
//!
//! `plan.rs::explain_tree` formats the **logical** plan tree. The T9
//! rewrite replaces a physical operator subtree at plan-time without
//! mutating the logical plan, so EXPLAIN output is unchanged before
//! and after the rewrite. We use an atomic counter
//! (`TYPED_DEGREE_REWRITE_COUNTER`) incremented by the matcher each
//! time it fires. The tests reset + run + read.
//!
//! ### Tri-state expectation
//!
//! - Before T9c (matcher not yet implemented) : counter stays at 0 on
//!   every query. The two positive tests (`canonical_`, `single_`) go
//!   red because they assert `counter == 1`. The negative test
//!   (`match_only_`) stays green (0 == 0).
//! - After T9c : counter fires exactly once on the canonical patterns
//!   and not at all on the MATCH-only variant. All three tests green.
//!
//! Run with :
//!
//! ```bash
//! cargo test -p obrain-engine --release --features cypher \
//!   --test t9_planner_rewrite_snapshot -- --nocapture
//! ```

#![cfg(feature = "cypher")]

use obrain_core::execution::operators::{
    reset_typed_degree_rewrite_counter, typed_degree_rewrite_counter,
};
use obrain_engine::ObrainDB;

/// Opens the PO database (or skips the test with an explanatory
/// message if the corpus is absent). Every T9b test needs a real,
/// populated graph because the matcher fires on the physical plan
/// shape that the optimizer emits, which depends on node labels and
/// edge types resolving to real registry ids.
fn open_po_or_skip() -> Option<ObrainDB> {
    let home = std::env::var("HOME").unwrap_or_default();
    let po_path = std::path::PathBuf::from(&home).join(".obrain/db/po");
    if !po_path.exists() {
        eprintln!("⏭  ~/.obrain/db/po not present — skipping T9b snapshot tests");
        return None;
    }
    Some(ObrainDB::open(&po_path).expect("open PO db"))
}

#[test]
#[ignore = "T9c-gated: remove #[ignore] once try_plan_typed_degree_topk lands"]
fn canonical_routes_to_typed_degree_topk() {
    let Some(db) = open_po_or_skip() else { return };
    let session = db.session();

    reset_typed_degree_rewrite_counter();
    let result = session
        .execute_cypher(
            "MATCH (f:File) \
             OPTIONAL MATCH (f)-[:IMPORTS]->(imported:File) \
             OPTIONAL MATCH (dependent:File)-[:IMPORTS]->(f) \
             WITH f, count(DISTINCT imported) AS imports, count(DISTINCT dependent) AS dependents \
             RETURN f.path, imports, dependents, imports + dependents AS connections \
             ORDER BY connections DESC \
             LIMIT 50",
        )
        .expect("canonical query runs");
    assert_eq!(
        result.rows.len(),
        50,
        "canonical query should produce 50 rows on PO"
    );
    assert_eq!(
        typed_degree_rewrite_counter(),
        1,
        "canonical pattern MUST be routed through the T9 rewrite exactly once"
    );
}

#[test]
#[ignore = "T9c-gated: remove #[ignore] once try_plan_typed_degree_topk lands"]
fn single_direction_routes_to_typed_degree_topk() {
    let Some(db) = open_po_or_skip() else { return };
    let session = db.session();

    reset_typed_degree_rewrite_counter();
    let result = session
        .execute_cypher(
            "MATCH (f:File) \
             OPTIONAL MATCH (f)-[:IMPORTS]->(imported:File) \
             WITH f, count(DISTINCT imported) AS imports \
             RETURN f.path, imports \
             ORDER BY imports DESC \
             LIMIT 50",
        )
        .expect("single-direction query runs");
    assert_eq!(
        result.rows.len(),
        50,
        "single-direction query should produce 50 rows on PO"
    );
    assert_eq!(
        typed_degree_rewrite_counter(),
        1,
        "single-direction pattern MUST be routed through the T9 rewrite exactly once"
    );
}

#[test]
fn match_only_keeps_slow_path() {
    let Some(db) = open_po_or_skip() else { return };
    let session = db.session();

    reset_typed_degree_rewrite_counter();
    let result = session
        .execute_cypher(
            "MATCH (f:File)-[:IMPORTS]->(t:File) \
             WITH f, count(t) AS c \
             RETURN f.path, c \
             ORDER BY c DESC \
             LIMIT 10",
        )
        .expect("match-only query runs");
    // Cardinality depends on corpus ; just assert the query succeeds.
    let _ = result.rows.len();
    assert_eq!(
        typed_degree_rewrite_counter(),
        0,
        "MATCH-only (no OPTIONAL MATCH → no LeftJoin) MUST fall back to the slow path"
    );
}
