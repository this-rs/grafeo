//! T17h T9b — snapshot tests for the Cypher planner rewrite
//! (`try_plan_typed_degree_topk`).
//!
//! ### Semantic constraint discovered during T9c implementation
//!
//! The T8 `out_degree_by_type(node, edge_type)` counter is
//! **label-agnostic on the peer** — it counts every outgoing edge of
//! the given type regardless of the target node's label. The
//! canonical bench query `MATCH (f:File) OPTIONAL MATCH
//! (f)-[:IMPORTS]->(imported:File)` filters edges on `(imported:File)`
//! (via an auto-inserted `Filter(hasLabel(imported, "File"))` or a
//! `NodeScan(peer:Label)` on the Expand's input). On corpora where
//! some IMPORTS edges leave File nodes towards non-File nodes (or
//! vice-versa) the slow-path count is **strictly smaller** than the
//! T8 count.
//!
//! The rewrite therefore **refuses to fire** whenever the Cypher
//! source carries a label constraint on any peer variable. On the PO
//! corpus this means the canonical `most_connected_files` query (both
//! `imported:File` and `dependent:File` peer labels) stays on the
//! slow path. The single-direction variant without the `:File` peer
//! constraint is routed correctly.
//!
//! A future T17i follow-up would add store introspection — e.g.
//! `store.edge_target_labels(edge_type)` — to verify the corpus
//! invariant upfront and allow the rewrite on label-constrained
//! queries when the constraint is degenerate (only-File → only-File).
//!
//! ### Counter-based test strategy
//!
//! `plan.rs::explain_tree` formats the logical plan unchanged after
//! rewrite, so we cannot snapshot-match on physical operator names.
//! We use an atomic counter that the matcher increments on every
//! successful match. Tests reset + run + read.
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
use std::sync::Mutex;

/// Serializes access to the global `TYPED_DEGREE_REWRITE_COUNTER` to
/// avoid race between parallel cargo-test threads.
static TEST_LOCK: Mutex<()> = Mutex::new(());

fn open_po_or_skip() -> Option<ObrainDB> {
    let home = std::env::var("HOME").unwrap_or_default();
    let po_path = std::path::PathBuf::from(&home).join(".obrain/db/po");
    if !po_path.exists() {
        eprintln!("⏭  ~/.obrain/db/po not present — skipping T9b snapshot tests");
        return None;
    }
    Some(ObrainDB::open(&po_path).expect("open PO db"))
}

/// T17i T3 — canonical bench query with homogeneous peer labels
/// (`:File` on both `imported` and `dependent`) MUST route through
/// the typed-degree rewrite. The T2 histogram proves that every
/// IMPORTS edge on PO lands on a File node (both sides), so the
/// constraint is semantically redundant and the O(1) lookup
/// produces the same rows as the slow expand+count path.
#[test]
fn canonical_query_with_peer_labels_routes_to_typed_degree_topk() {
    let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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
        .expect("canonical query routes to typed-degree TopK");
    assert_eq!(result.rows.len(), 50);
    assert_eq!(
        typed_degree_rewrite_counter(),
        1,
        "canonical dual-direction with homogeneous peer labels MUST \
         fire the T17i T3 rewrite exactly once"
    );
}

/// Dual-direction query WITHOUT peer label constraints — the rewrite
/// MUST fire. This is the "practical" canonical shape that T9 is
/// designed to accelerate : `MATCH (f:File) OPTIONAL MATCH
/// (f)-[:IMPORTS]->() OPTIONAL MATCH ()-[:IMPORTS]->(f) …`.
#[test]
fn canonical_without_peer_labels_routes_to_typed_degree_topk() {
    let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(db) = open_po_or_skip() else { return };
    let session = db.session();

    reset_typed_degree_rewrite_counter();
    let result = session
        .execute_cypher(
            "MATCH (f:File) \
             OPTIONAL MATCH (f)-[:IMPORTS]->(imported) \
             OPTIONAL MATCH (dependent)-[:IMPORTS]->(f) \
             WITH f, count(DISTINCT imported) AS imports, count(DISTINCT dependent) AS dependents \
             RETURN f.path, imports, dependents, imports + dependents AS connections \
             ORDER BY connections DESC \
             LIMIT 50",
        )
        .expect("dual-direction query without peer labels runs");
    assert_eq!(result.rows.len(), 50);
    assert_eq!(
        typed_degree_rewrite_counter(),
        1,
        "dual-direction pattern without peer labels MUST be routed through the T9 rewrite exactly once"
    );
}

/// Single-direction query WITHOUT peer label — must route through the
/// rewrite.
#[test]
fn single_direction_without_peer_label_routes_to_typed_degree_topk() {
    let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(db) = open_po_or_skip() else { return };
    let session = db.session();

    reset_typed_degree_rewrite_counter();
    let result = session
        .execute_cypher(
            "MATCH (f:File) \
             OPTIONAL MATCH (f)-[:IMPORTS]->(imported) \
             WITH f, count(DISTINCT imported) AS imports \
             RETURN f.path, imports \
             ORDER BY imports DESC \
             LIMIT 50",
        )
        .expect("single-direction query without peer label runs");
    assert_eq!(result.rows.len(), 50);
    assert_eq!(
        typed_degree_rewrite_counter(),
        1,
        "single-direction pattern without peer label MUST be routed through the T9 rewrite exactly once"
    );
}

/// T17i T3 — single-direction WITH peer label also routes through
/// the rewrite once the T2 histogram gate is in place.
#[test]
fn single_direction_with_peer_label_routes_to_typed_degree_topk() {
    let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(db) = open_po_or_skip() else { return };
    let session = db.session();

    reset_typed_degree_rewrite_counter();
    let _ = session
        .execute_cypher(
            "MATCH (f:File) \
             OPTIONAL MATCH (f)-[:IMPORTS]->(imported:File) \
             WITH f, count(DISTINCT imported) AS imports \
             RETURN f.path, imports \
             ORDER BY imports DESC \
             LIMIT 50",
        )
        .expect("single-direction query with peer label routes");
    assert_eq!(
        typed_degree_rewrite_counter(),
        1,
        "single-direction with homogeneous peer label :File MUST fire the T17i T3 rewrite"
    );
}

/// Non-OPTIONAL MATCH variant produces no LeftJoin → matcher must
/// NOT fire. Active regression anchor for the "no false positives"
/// guarantee.
#[test]
fn match_only_keeps_slow_path() {
    let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(db) = open_po_or_skip() else { return };
    let session = db.session();

    reset_typed_degree_rewrite_counter();
    let _ = session
        .execute_cypher(
            "MATCH (f:File)-[:IMPORTS]->(t:File) \
             WITH f, count(t) AS c \
             RETURN f.path, c \
             ORDER BY c DESC \
             LIMIT 10",
        )
        .expect("match-only query runs");
    assert_eq!(
        typed_degree_rewrite_counter(),
        0,
        "MATCH-only (no OPTIONAL MATCH → no LeftJoin) MUST fall back to the slow path"
    );
}
