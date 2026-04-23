#![cfg(feature = "cypher")]
//! Profiling bench to identify exact bottlenecks in slow PO queries.
//!
//! cargo test -p obrain-engine --release --features "cypher,wal,tiered-storage" \
//!   --test po_profiling_bench -- --nocapture

use obrain_engine::ObrainDB;
use std::time::Instant;

fn measure(name: &str, session: &obrain_engine::Session, query: &str, iters: usize) {
    // Warmup
    for _ in 0..2 {
        let _ = session.execute_cypher(query);
    }
    let mut times = Vec::with_capacity(iters);
    let mut rows = 0;
    for _ in 0..iters {
        let start = Instant::now();
        let r = session.execute_cypher(query).unwrap();
        times.push(start.elapsed().as_secs_f64() * 1000.0);
        rows = r.row_count();
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let p50 = times[times.len() / 2];
    println!(
        "  {:<65} {:>8.1}ms (p50: {:.1})  [{} rows]",
        name, mean, p50, rows
    );
}

#[test]
fn profile_list_project_symbols() {
    let home = std::env::var("HOME").unwrap_or_default();
    let po_path = std::path::PathBuf::from(&home).join(".obrain/db/po");
    if !po_path.exists() {
        println!("  Skipping — ~/.obrain/db/po not found");
        return;
    }

    let db = ObrainDB::open(&po_path).expect("open PO db");
    let session = db.session();

    println!("\n{}", "═".repeat(80));
    println!("  PROFILING: list_project_symbols breakdown");
    println!("{}\n", "═".repeat(80));

    let n = 3;

    // Baseline: the slow query (1060ms)
    measure(
        "FULL: Proj→File→Fn/Struct/Trait/Enum + CASE + ORDER + LIMIT",
        &session,
        "MATCH (p:Project)-[:CONTAINS]->(f:File)-[:CONTAINS]->(s) \
         WHERE s:Function OR s:Struct OR s:Trait OR s:Enum \
         RETURN s.name, f.path, s.visibility \
         ORDER BY f.path, s.line_start LIMIT 5000",
        n,
    );

    // Step 2: With WHERE filter, count only
    measure(
        "Step 2: traverse + WHERE filter, count only",
        &session,
        "MATCH (p:Project)-[:CONTAINS]->(f:File)-[:CONTAINS]->(s) \
         WHERE s:Function OR s:Struct OR s:Trait OR s:Enum \
         RETURN count(s)",
        n,
    );

    // Step 3: With RETURN properties, no ORDER BY
    measure(
        "Step 3: + RETURN s.name, f.path (no ORDER BY, LIMIT 5000)",
        &session,
        "MATCH (p:Project)-[:CONTAINS]->(f:File)-[:CONTAINS]->(s) \
         WHERE s:Function OR s:Struct OR s:Trait OR s:Enum \
         RETURN s.name, f.path LIMIT 5000",
        n,
    );

    // Step 4: With ORDER BY but no CASE
    measure(
        "Step 4: + ORDER BY f.path, s.line_start LIMIT 5000",
        &session,
        "MATCH (p:Project)-[:CONTAINS]->(f:File)-[:CONTAINS]->(s) \
         WHERE s:Function OR s:Struct OR s:Trait OR s:Enum \
         RETURN s.name, f.path, s.line_start \
         ORDER BY f.path, s.line_start LIMIT 5000",
        n,
    );

    // Alternative: Single-label directly (should be fast)
    measure(
        "ALT: Single-label MATCH (s:Function) directly (LIMIT 5000)",
        &session,
        "MATCH (p:Project)-[:CONTAINS]->(f:File)-[:CONTAINS]->(s:Function) \
         RETURN s.name, f.path LIMIT 5000",
        n,
    );

    // Just count how many nodes are of each type
    for label in &["Function", "Struct", "Trait", "Enum"] {
        let r = session
            .execute_cypher(&format!("MATCH (n:{label}) RETURN count(n)"))
            .unwrap();
        println!("    {label}: {}", r.scalar::<i64>().unwrap());
    }

    println!("\n{}", "═".repeat(80));
    println!("  PROFILING: extends/implements edges breakdown");
    println!("{}\n", "═".repeat(80));

    // Extends: the 4-hop pattern
    measure(
        "FULL extends: Proj→File→Struct→EXTENDS→Struct←File←Proj",
        &session,
        "MATCH (p:Project)-[:CONTAINS]->(f1:File)-[:CONTAINS]->(s1:Struct)-[:EXTENDS]->(s2:Struct)<-[:CONTAINS]-(f2:File)<-[:CONTAINS]-(p) \
         WHERE f1 <> f2 \
         RETURN f1.path, f2.path",
        n,
    );

    // Start from EXTENDS (should be faster — fewer edges)
    measure(
        "ALT extends: Start from EXTENDS edge directly",
        &session,
        "MATCH (s1:Struct)-[:EXTENDS]->(s2:Struct) \
         RETURN count(*)",
        n,
    );

    measure(
        "ALT extends: EXTENDS + file lookup",
        &session,
        "MATCH (s1:Struct)-[:EXTENDS]->(s2:Struct) \
         MATCH (f1:File)-[:CONTAINS]->(s1) \
         MATCH (f2:File)-[:CONTAINS]->(s2) \
         WHERE f1 <> f2 \
         RETURN f1.path, f2.path",
        n,
    );

    // Implements
    measure(
        "FULL implements: Proj→File→Struct→IMPL→Trait←File←Proj",
        &session,
        "MATCH (p:Project)-[:CONTAINS]->(f1:File)-[:CONTAINS]->(s:Struct)-[:IMPLEMENTS]->(t:Trait)<-[:CONTAINS]-(f2:File)<-[:CONTAINS]-(p) \
         WHERE f1 <> f2 \
         RETURN f1.path, f2.path",
        n,
    );

    measure(
        "ALT implements: Start from IMPLEMENTS",
        &session,
        "MATCH (s:Struct)-[:IMPLEMENTS]->(t:Trait) \
         MATCH (f1:File)-[:CONTAINS]->(s) \
         MATCH (f2:File)-[:CONTAINS]->(t) \
         WHERE f1 <> f2 \
         RETURN f1.path, f2.path",
        n,
    );

    println!("\n{}", "═".repeat(80));
    println!("  PROFILING: most_connected breakdown");
    println!("{}\n", "═".repeat(80));

    measure(
        "FULL most_connected (OPTIONAL MATCH + count + ORDER + LIMIT)",
        &session,
        "MATCH (f:File) \
         OPTIONAL MATCH (f)-[:IMPORTS]->(imported:File) \
         OPTIONAL MATCH (dependent:File)-[:IMPORTS]->(f) \
         WITH f, count(DISTINCT imported) AS imports, count(DISTINCT dependent) AS dependents \
         RETURN f.path, imports, dependents, imports + dependents AS connections \
         ORDER BY connections DESC LIMIT 50",
        n,
    );

    // Alternative: Two separate counts via subquery
    measure(
        "ALT: Just MATCH imports count",
        &session,
        "MATCH (f:File)-[:IMPORTS]->(imported:File) \
         RETURN f.path, count(imported) AS imports \
         ORDER BY imports DESC LIMIT 50",
        n,
    );

    measure(
        "ALT: Just MATCH dependents count",
        &session,
        "MATCH (dependent:File)-[:IMPORTS]->(f:File) \
         RETURN f.path, count(dependent) AS deps \
         ORDER BY deps DESC LIMIT 50",
        n,
    );

    println!("\n  Profiling complete.");
}
