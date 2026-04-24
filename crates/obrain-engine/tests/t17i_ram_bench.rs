//! T17i T4 — RAM consumption benchmark for DB open on PO,
//! Wikipedia, Megalaw. Reports process RSS before and after each
//! open so the T17i T1 lazy-init design's memory profile is
//! visible.
//!
//! Portable RSS read via `ps -o rss= -p $PID` (KB, works on macOS +
//! Linux). Not the most precise — /proc/self/statm would be finer-
//! grained on Linux — but good enough for order-of-magnitude
//! reporting across the 3 corpora.
//!
//! Run with :
//!
//! ```bash
//! cargo test -p obrain-engine --release --features cypher \
//!   --test t17i_ram_bench -- --nocapture
//! ```

#![cfg(feature = "cypher")]

use obrain_engine::ObrainDB;
use std::time::Instant;

fn current_rss_mb() -> f64 {
    let pid = std::process::id();
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output()
        .ok();
    let Some(o) = out else {
        return -1.0;
    };
    let s = String::from_utf8_lossy(&o.stdout);
    s.trim()
        .parse::<f64>()
        .map(|kb| kb / 1024.0)
        .unwrap_or(-1.0)
}

fn measure(name: &str, path: std::path::PathBuf) {
    if !path.exists() {
        eprintln!("⏭  {name} : {} not found", path.display());
        return;
    }
    let rss_before = current_rss_mb();
    let t0 = Instant::now();
    let db = ObrainDB::open(&path).expect("open");
    let open_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let rss_after_open = current_rss_mb();

    // Touch the store a bit so mmap pages get faulted in (typed
    // degrees, label histograms). This simulates a first query.
    let session = db.session();
    let _ = session.execute_cypher("MATCH (n) RETURN count(n)").ok();
    let _ = session
        .execute_cypher("MATCH (f:File)-[:IMPORTS]->() RETURN count(*)")
        .ok();
    let rss_after_touch = current_rss_mb();

    println!(
        "  {:<12} : open={:>6.1}ms  rss_before={:>7.1}MB  rss_after_open={:>7.1}MB  \
         delta_open={:>+6.1}MB  rss_after_touch={:>7.1}MB  delta_touch={:>+6.1}MB",
        name,
        open_ms,
        rss_before,
        rss_after_open,
        rss_after_open - rss_before,
        rss_after_touch,
        rss_after_touch - rss_after_open,
    );
    drop(db);
}

#[test]
fn bench_ram_three_corpora() {
    let home = std::env::var("HOME").unwrap_or_default();
    println!("\n=== T17i T4 — RAM bench : DB open + first queries ===");
    println!("(RSS values in MB, process-wide ; delta = increment during that phase)");
    measure(
        "PO",
        std::path::PathBuf::from(&home).join(".obrain/db/po"),
    );
    measure(
        "Wikipedia",
        std::path::PathBuf::from(&home).join(".obrain/db/wikipedia"),
    );
    measure(
        "Megalaw",
        std::path::PathBuf::from(&home).join(".obrain/db/megalaw"),
    );
    println!("=== END ===");
}
