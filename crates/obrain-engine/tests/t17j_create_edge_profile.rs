//! T17j T1 profile — compare Cypher MATCH+CREATE loop vs direct-API
//! create_edge loop. The `create 200 relationships` bench runs 200
//! queries of the form `MATCH (a:_BenchSrc {id: i}), (b:_BenchDst
//! {id: i}) CREATE (a)-[:_BENCH_REL]->(b)` — this tests Cypher query
//! round-trip (parser + planner + executor) much more than the
//! substrate write path itself.
//!
//! Run with :
//!
//! ```bash
//! cargo test -p obrain-engine --release --features cypher \
//!   --test t17j_create_edge_profile -- --nocapture
//! ```

#![cfg(feature = "cypher")]

use obrain_core::graph::traits::GraphStoreMut;
use obrain_engine::ObrainDB;
use std::time::Instant;

const N: usize = 200;

fn fresh_db() -> (tempfile::TempDir, ObrainDB) {
    let td = tempfile::tempdir().unwrap();
    let path = td.path().join("kb");
    let db = ObrainDB::open(&path).unwrap();
    (td, db)
}

fn seed_bench_nodes(db: &ObrainDB) -> Vec<(obrain_common::types::NodeId, obrain_common::types::NodeId)> {
    let store = db.store();
    (0..N)
        .map(|_| {
            let a = store.create_node(&["_BenchSrc"]);
            let b = store.create_node(&["_BenchDst"]);
            (a, b)
        })
        .collect()
}

#[test]
fn profile_cypher_match_create_loop() {
    let (_td, db) = fresh_db();
    let session = db.session();
    // Seed 200 src + 200 dst nodes via direct API.
    for i in 0..N {
        let _ = session
            .execute_cypher(&format!("CREATE (:_BenchSrc {{id: {i}}})"))
            .unwrap();
        let _ = session
            .execute_cypher(&format!("CREATE (:_BenchDst {{id: {i}}})"))
            .unwrap();
    }
    // Measure the same pattern as the po_vs_neo4j_bench.
    let t0 = Instant::now();
    for i in 0..N {
        let _ = session
            .execute_cypher(&format!(
                "MATCH (a:_BenchSrc {{id: {i}}}), (b:_BenchDst {{id: {i}}}) \
                 CREATE (a)-[:_BENCH_REL]->(b)"
            ))
            .unwrap();
    }
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let per_call = elapsed_ms / N as f64;
    println!(
        "\n  Cypher MATCH+CREATE loop : {:.1} ms total  ({:.2} ms/call)",
        elapsed_ms, per_call
    );
}

#[test]
fn profile_direct_api_create_edge_loop() {
    let (_td, db) = fresh_db();
    let pairs = seed_bench_nodes(&db);
    let store = db.store();
    let t0 = Instant::now();
    for (a, b) in &pairs {
        store.create_edge(*a, *b, "_BENCH_REL");
    }
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let per_call = elapsed_ms / N as f64;
    println!(
        "\n  Direct-API create_edge loop : {:.1} ms total  ({:.2} ms/call)",
        elapsed_ms, per_call
    );
}

/// Profile each Cypher stage separately : parse, plan, execute.
#[test]
fn profile_cypher_stages() {
    let (_td, db) = fresh_db();
    let session = db.session();
    for i in 0..N {
        let _ = session
            .execute_cypher(&format!("CREATE (:_BenchSrc {{id: {i}}})"))
            .unwrap();
        let _ = session
            .execute_cypher(&format!("CREATE (:_BenchDst {{id: {i}}})"))
            .unwrap();
    }
    let query_str = "MATCH (a:_BenchSrc {id: 0}), (b:_BenchDst {id: 0}) CREATE (a)-[:_BENCH_REL]->(b)";
    let iterations = 50;
    let t0 = Instant::now();
    for _ in 0..iterations {
        let _ = session.execute_cypher(query_str).unwrap();
    }
    let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
    println!(
        "\n  Same query × {} iterations : {:.1} ms total  ({:.2} ms/call)",
        iterations,
        elapsed,
        elapsed / iterations as f64
    );
    println!(
        "  (same query string — difference vs unique strings \
         shows parse+plan amortisation potential via prepared statements)"
    );
}
