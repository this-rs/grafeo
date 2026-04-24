//! T17j T2 — audit the PO "chain corruption" gotcha (c9aabd7b).
//!
//! Hypotheses :
//! - (A) Corpus legit : most of the 8300 IMPORTS edges come from
//!   non-File sources (Module, Function, ...). `MATCH (f:File)
//!   -[:IMPORTS]->()` correctly returns only the 515 File-sourced
//!   ones ; `walk_outgoing_chain` on File nodes returns 0 for those
//!   that have no outgoing IMPORTS (the vast majority).
//! - (B) Chain corruption : the neo4j2obrain migration initialised
//!   `NodeRecord.first_out_off` incorrectly on some File nodes, so
//!   `walk_outgoing_chain` misses edges that DO exist in the zone
//!   with src == that File.
//!
//! Run with :
//!
//! ```bash
//! cargo test -p obrain-engine --release --features cypher \
//!   --test t17j_chain_audit -- --nocapture
//! ```

#![cfg(feature = "cypher")]

use obrain_core::graph::GraphStore;
use obrain_engine::ObrainDB;

#[test]
fn audit_imports_source_labels_on_po() {
    let home = std::env::var("HOME").unwrap_or_default();
    let po_path = std::path::PathBuf::from(&home).join(".obrain/db/po");
    if !po_path.exists() {
        eprintln!("⏭ PO database not present; skipping");
        return;
    }
    let db = ObrainDB::open(&po_path).expect("open PO db");
    let session = db.session();

    println!("\n=== T17j T2 — IMPORTS source labels audit on PO ===");

    // 1. Total IMPORTS.
    let r = session
        .execute_cypher("MATCH ()-[r:IMPORTS]->() RETURN count(r) AS total")
        .unwrap();
    println!("Total IMPORTS edges                : {:?}", r.rows);

    // 2. IMPORTS with File source.
    let r = session
        .execute_cypher("MATCH (s:File)-[r:IMPORTS]->() RETURN count(r) AS from_file")
        .unwrap();
    println!("IMPORTS sourced on :File           : {:?}", r.rows);

    // 3. IMPORTS where source is NOT a File.
    let r = session
        .execute_cypher(
            "MATCH (s)-[r:IMPORTS]->() WHERE NOT (s:File) RETURN count(r) AS non_file_source",
        )
        .unwrap();
    println!("IMPORTS NOT sourced on :File       : {:?}", r.rows);

    // 4. If (3) > 0, what are those source labels ? Sample count per-label.
    //    Obrain has ~64 possible labels ; we only enumerate the top few
    //    to avoid a giant output.
    let known_labels = [
        "File", "Module", "Function", "Struct", "Trait", "Enum", "Project",
        "Package", "Impl", "Method", "Class", "Variable",
    ];
    println!("\n  Source-label histogram (per known label) :");
    for label in known_labels {
        let r = session
            .execute_cypher(&format!(
                "MATCH (s:{label})-[r:IMPORTS]->() RETURN count(r) AS n"
            ))
            .unwrap();
        if let Some(row) = r.rows.first()
            && let Some(obrain_common::types::Value::Int64(n)) = row.first()
            && *n > 0
        {
            println!("    :{:<20} → {} edges", label, n);
        }
    }

    // 5. Sanity : via store API, `edge_target_labels("IMPORTS")` should
    //    contain the target-side labels of EVERY live IMPORTS edge.
    let store = db.store();
    let targets = store.edge_target_labels("IMPORTS");
    let sources = store.edge_source_labels("IMPORTS");
    println!("\n  store.edge_target_labels(IMPORTS) : {:?}", targets);
    println!("  store.edge_source_labels(IMPORTS) : {:?}", sources);
}
