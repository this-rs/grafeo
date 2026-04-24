//! T17i T1 — benchmark DB open time across PO, Wikipedia, Megalaw
//! to quantify the regression introduced by the T8 eager-init
//! (init_typed_degrees in from_substrate).
//!
//! Baseline T17g : PO 8 ms, Wiki 24 ms, Megalaw 42 ms (from the
//! commit log of T17h T5).
//! Post T17h : PO 119 ms (measured by the po_vs_neo4j_bench run this
//! session). Wiki / Megalaw not yet measured — this test captures
//! them.
//!
//! Run with :
//!
//! ```bash
//! cargo test -p obrain-engine --release --features cypher \
//!   --test t17i_db_open_bench -- --nocapture
//! ```

#![cfg(feature = "cypher")]

use obrain_engine::ObrainDB;
use std::time::Instant;

fn measure_open(name: &str, path: std::path::PathBuf) {
    if !path.exists() {
        eprintln!("⏭  {name} : {} not found — skipping", path.display());
        return;
    }
    // 3 warmup opens to prime OS page cache, then 5 measured opens.
    for _ in 0..3 {
        let db = ObrainDB::open(&path).expect("open warmup");
        drop(db);
    }
    let mut samples_ms: Vec<f64> = Vec::with_capacity(5);
    for _ in 0..5 {
        let t0 = Instant::now();
        let db = ObrainDB::open(&path).expect("open measured");
        samples_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
        drop(db);
    }
    samples_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min = samples_ms.first().copied().unwrap();
    let max = samples_ms.last().copied().unwrap();
    let p50 = samples_ms[samples_ms.len() / 2];
    let mean = samples_ms.iter().sum::<f64>() / samples_ms.len() as f64;
    println!(
        "  {:<12} : p50={:>7.1}ms  min={:>7.1}ms  max={:>7.1}ms  mean={:>7.1}ms",
        name, p50, min, max, mean
    );
}

#[test]
fn bench_db_open_three_corpora() {
    let home = std::env::var("HOME").unwrap_or_default();
    println!("\n=== T17i T1 — DB open bench (post T17h eager init) ===");
    println!("Baselines T17g : PO 8ms, Wiki 24ms, Megalaw 42ms");
    measure_open(
        "PO",
        std::path::PathBuf::from(&home).join(".obrain/db/po"),
    );
    measure_open(
        "Wikipedia",
        std::path::PathBuf::from(&home).join(".obrain/db/wikipedia"),
    );
    measure_open(
        "Megalaw",
        std::path::PathBuf::from(&home).join(".obrain/db/megalaw"),
    );
    println!("=== END ===");
}
