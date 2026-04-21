//! # T17e Phase 4 — Reverse-traversal lazy-build latency bench
//!
//! Opens a substrate DB, measures the latency of the **first** reverse
//! traversal (which triggers the lazy build of `incoming_heads` via
//! `OnceLock::get_or_init`), then does N more reverse traversals to
//! measure steady-state latency (cache warm, DashMap lookups only).
//!
//! Gate (Phase 4 acceptance):
//! * first reverse traversal < 20 s (blocking, one-shot cold build)
//! * subsequent traversals: O(1) per call
//!
//! Usage:
//! ```text
//! cargo run --release --bin reverse-traversal-bench -- \
//!     --db /Users/triviere/.obrain/db/wikipedia
//! ```

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use obrain_core::graph::Direction;
use obrain_core::graph::traits::GraphStore;
use obrain_substrate::SubstrateStore;

#[derive(Parser, Debug)]
#[command(name = "reverse-traversal-bench", about = "T17e Phase 4 lazy-build latency")]
struct Args {
    /// Substrate directory (flat or nested layout — auto-detected).
    #[arg(long)]
    db: PathBuf,

    /// Number of additional reverse traversals to run after the first one
    /// (for steady-state measurement).
    #[arg(long, default_value_t = 1000)]
    steady_n: usize,
}

fn resolve_sub_dir(db: &Path) -> PathBuf {
    if db.join("substrate.obrain").is_dir() {
        db.join("substrate.obrain")
    } else {
        db.to_path_buf()
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();
    let sub_dir = resolve_sub_dir(&args.db);
    println!("db     : {}", sub_dir.display());

    // --- Step 1: open ---
    let t_open = Instant::now();
    let store = SubstrateStore::open(&sub_dir).context("SubstrateStore::open")?;
    let startup_ms = t_open.elapsed().as_secs_f64() * 1000.0;
    println!("startup: {:.2} ms", startup_ms);

    // --- Step 2: find a node with incoming edges to guarantee the lazy
    //            build is exercised. We scan node_ids for a high-slot one
    //            (older nodes tend to have more inbound edges in real
    //            graphs). If none has an inbound, we still trigger the
    //            build because `edges_from(_, Incoming)` still calls
    //            `incoming_heads()`.
    let ids = store.node_ids();
    let pivot = *ids
        .get(ids.len() / 2)
        .or_else(|| ids.first())
        .context("no nodes in DB")?;
    println!("pivot  : node {:?} (live_nodes = {})", pivot, ids.len());

    // --- Step 3: first reverse traversal — this triggers the lazy build.
    let t_first = Instant::now();
    let neighbors_first = store.edges_from(pivot, Direction::Incoming);
    let first_ms = t_first.elapsed().as_secs_f64() * 1000.0;
    println!(
        "first reverse traversal: {:.2} ms  ({} inbound edges found)",
        first_ms,
        neighbors_first.len()
    );

    // --- Step 4: steady-state — N more reverse traversals across
    //            different nodes.
    let steady_n = args.steady_n.min(ids.len());
    let step = (ids.len() / steady_n.max(1)).max(1);
    let mut total_found = 0usize;
    let t_steady = Instant::now();
    for i in 0..steady_n {
        let idx = (i * step) % ids.len();
        let nid = ids[idx];
        let nbrs = store.edges_from(nid, Direction::Incoming);
        total_found += nbrs.len();
    }
    let steady_total_us = t_steady.elapsed().as_micros() as f64;
    let per_call_us = steady_total_us / steady_n as f64;
    println!(
        "steady-state: {} calls, {:.2} µs/call, total {} inbound edges",
        steady_n, per_call_us, total_found
    );

    // --- Verdict ---
    let gate_first = 20_000.0; // 20 s
    let gate_first_pass = first_ms <= gate_first;
    println!();
    println!(
        "gate  first traversal ≤ 20 s : {}",
        if gate_first_pass { "✅ pass" } else { "❌ fail" }
    );

    if gate_first_pass {
        std::process::exit(0);
    } else {
        std::process::exit(1);
    }
}
