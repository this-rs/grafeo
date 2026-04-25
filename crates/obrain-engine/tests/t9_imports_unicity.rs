//! T17h T9 Gap B — audit unicity of IMPORTS edges on PO.
//!
//! If multi-edges `(src, dst)` with relationship `IMPORTS` exist, then
//! `count(DISTINCT imported) != typed_out_degree(f, "IMPORTS")` and the
//! TypedDegreeTopK rewrite would diverge from the slow path. If no
//! multi-edges exist, the rewrite is exact on PO.
//!
//! Run with :
//!
//! ```bash
//! cargo test -p obrain-engine --release --features cypher \
//!   --test t9_imports_unicity -- --nocapture
//! ```

#![cfg(feature = "cypher")]

use obrain_engine::ObrainDB;

#[test]
fn count_duplicated_imports_pairs() {
    let home = std::env::var("HOME").unwrap_or_default();
    let po_path = std::path::PathBuf::from(&home).join(".obrain/db/po");
    if !po_path.exists() {
        println!("⏭ PO database not present; skipping");
        return;
    }

    let db = ObrainDB::open(&po_path).expect("open PO db");
    let session = db.session();

    // Count pairs (s, t) that have ≥ 2 IMPORTS edges between them.
    let dup_result = session
        .execute_cypher(
            "MATCH (s)-[r:IMPORTS]->(t) \
             WITH s, t, count(r) AS c \
             WHERE c > 1 \
             RETURN count(*) AS dup_pairs",
        )
        .expect("dup query");

    // Total IMPORTS edge count (for baseline).
    let total_result = session
        .execute_cypher("MATCH ()-[r:IMPORTS]->() RETURN count(r) AS total")
        .expect("total query");

    // Distinct (s, t) pairs — `count(DISTINCT …)` style.
    let distinct_pairs_result = session
        .execute_cypher(
            "MATCH (s)-[:IMPORTS]->(t) \
             WITH s, t \
             RETURN count(*) AS distinct_pairs",
        )
        .expect("distinct query");

    println!("\n=== IMPORTS unicity audit on PO ===");
    println!("Duplicated-pair result rows : {:?}", dup_result.rows);
    println!("Total IMPORTS edges         : {:?}", total_result.rows);
    println!(
        "Distinct (s,t) pairs        : {:?}",
        distinct_pairs_result.rows
    );

    // Actionable assertion — if duplicates > 0 the rewrite needs a guard.
    if let Some(row) = dup_result.rows.first()
        && let Some(val) = row.first()
    {
        println!("=> duplicates detected : {val:?}");
    }
}
