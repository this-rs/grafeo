//! T17h T9 step 3 Gap A — dump the post-optimizer LogicalPlan for the
//! canonical `most_connected_files` bench query so the planner rewrite
//! matches exactly what the optimizer emits (not the Cypher-literal
//! tree).
//!
//! Run with :
//!
//! ```bash
//! cargo test -p obrain-engine --release --features cypher \
//!   --test t9_explain_dump -- --nocapture
//! ```

#![cfg(feature = "cypher")]

use obrain_engine::ObrainDB;

#[test]
fn explain_most_connected_files() {
    let home = std::env::var("HOME").unwrap_or_default();
    let po_path = std::path::PathBuf::from(&home).join(".obrain/db/po");
    if !po_path.exists() {
        println!("⏭ PO database not present; skipping");
        return;
    }

    let db = ObrainDB::open(&po_path).expect("open PO db");
    let session = db.session();

    let result = session
        .execute_cypher(
            "EXPLAIN \
             MATCH (f:File) \
             OPTIONAL MATCH (f)-[:IMPORTS]->(imported:File) \
             OPTIONAL MATCH (dependent:File)-[:IMPORTS]->(f) \
             WITH f, count(DISTINCT imported) AS imports, count(DISTINCT dependent) AS dependents \
             RETURN f.path, imports, dependents, imports + dependents AS connections \
             ORDER BY connections DESC \
             LIMIT 50",
        )
        .expect("explain execute");

    println!("=== EXPLAIN OUTPUT (most_connected_files) ===");
    println!("columns: {:?}", result.columns);
    println!("row_count: {}", result.rows.len());
    for (ri, row) in result.rows.iter().enumerate() {
        for (ci, val) in row.iter().enumerate() {
            let col = result.columns.get(ci).map(String::as_str).unwrap_or("?");
            match val {
                obrain_common::types::Value::String(s) => {
                    println!("--- row {ri} col {ci} [{col}] ---\n{s}");
                }
                other => {
                    println!("--- row {ri} col {ci} [{col}] ---\n{other:?}");
                }
            }
        }
    }
    println!("=== END EXPLAIN ===");
}

#[test]
fn explain_single_direction_variant() {
    let home = std::env::var("HOME").unwrap_or_default();
    let po_path = std::path::PathBuf::from(&home).join(".obrain/db/po");
    if !po_path.exists() {
        return;
    }

    let db = ObrainDB::open(&po_path).expect("open PO db");
    let session = db.session();

    // Single-direction variant — simpler shape, useful baseline for
    // comparing the complexity of the canonical dual-OPTIONAL-MATCH.
    let result = session
        .execute_cypher(
            "EXPLAIN \
             MATCH (f:File) \
             OPTIONAL MATCH (f)-[:IMPORTS]->(imported:File) \
             WITH f, count(DISTINCT imported) AS imports \
             RETURN f.path, imports \
             ORDER BY imports DESC \
             LIMIT 50",
        )
        .expect("explain execute");

    println!("\n=== EXPLAIN OUTPUT (single-direction variant) ===");
    for (ri, row) in result.rows.iter().enumerate() {
        for (ci, val) in row.iter().enumerate() {
            if let obrain_common::types::Value::String(s) = val {
                println!("--- row {ri} col {ci} ---\n{s}");
            }
        }
    }
    println!("=== END EXPLAIN ===");
}

#[test]
fn explain_simplest_count_pattern() {
    let home = std::env::var("HOME").unwrap_or_default();
    let po_path = std::path::PathBuf::from(&home).join(".obrain/db/po");
    if !po_path.exists() {
        return;
    }

    let db = ObrainDB::open(&po_path).expect("open PO db");
    let session = db.session();

    // Minimal TopK-ish pattern — no OPTIONAL MATCH.
    let result = session
        .execute_cypher(
            "EXPLAIN \
             MATCH (f:File)-[:IMPORTS]->(t:File) \
             WITH f, count(t) AS c \
             RETURN f.path, c \
             ORDER BY c DESC \
             LIMIT 10",
        )
        .expect("explain execute");

    println!("\n=== EXPLAIN OUTPUT (MATCH-only count) ===");
    for (ri, row) in result.rows.iter().enumerate() {
        for (ci, val) in row.iter().enumerate() {
            if let obrain_common::types::Value::String(s) = val {
                println!("--- row {ri} col {ci} ---\n{s}");
            }
        }
    }
    println!("=== END EXPLAIN ===");
}
