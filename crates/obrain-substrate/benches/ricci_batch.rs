//! T12 Step 1 — batch Ricci refresh latency bench.
//!
//! # Acceptance criterion
//!
//! A full-graph `refresh_all_ricci` on a 10⁶-node substrate with
//! `AVG_DEGREE ≈ 8` edges per node (≈ 8·10⁶ directed edges total)
//! must complete in **≤ 30 s on 8 cores** — the plan's explicit gate
//! for triggering on import (T15) and on periodic Dreamer ticks.
//!
//! # Shape of the bench
//!
//! One Criterion micro-bench group:
//!
//! 1. `refresh_all_ricci` — calls the batch pipeline end-to-end:
//!    neighbourhood snapshot (phase 1) + edge snapshot (phase 2) +
//!    rayon-parallel Jost-Liu compute (phase 3) + sequential WAL-
//!    logged apply (phase 4).
//!
//! # Scale control
//!
//! The bench is a one-shot measurement — Criterion's warm-up and
//! sample machinery would rerun the batch refresh many times, and
//! each rerun after the first is a no-op (idempotent: every edge's
//! `ricci_u8` already matches the computed value, so phase 4
//! short-circuits). We disable warm-up and set a sample_size of 10
//! with a tiny measurement_time so Criterion doesn't try to
//! amortise the fixed-cost setup.
//!
//! Set `OBRAIN_BENCH_N=1000000` at invocation time to run the full
//! 10⁶-node acceptance scale. Default `100_000` nodes (≈ 800 k
//! directed edges) keeps the bench below a minute of setup and is
//! enough to spot a regression in the compute phase.
//!
//! # Run
//!
//! ```bash
//! # Default (100k nodes)
//! cargo bench -p obrain-substrate --bench ricci_batch
//!
//! # Full acceptance scale (10⁶ nodes)
//! OBRAIN_BENCH_N=1000000 cargo bench -p obrain-substrate --bench ricci_batch
//! ```
//!
//! The setup pattern (burst community allocation + bucket-based
//! edge wiring) mirrors `benches/spreading_activation.rs` so the
//! two benches share a workload shape and regressions compare
//! like-for-like.

use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use obrain_common::types::NodeId;
use obrain_core::graph::traits::{GraphStore, GraphStoreMut};
use obrain_substrate::store::SubstrateStore;
use obrain_substrate::{RicciRefreshStats, refresh_all_ricci};
use tempfile::TempDir;

// -----------------------------------------------------------------------
// Workload parameters — match `spreading_activation.rs` so bench
// regressions compose cleanly.
// -----------------------------------------------------------------------

fn workload_n() -> usize {
    std::env::var("OBRAIN_BENCH_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000)
}

const NODES_PER_COMMUNITY: usize = 512;
const AVG_DEGREE: usize = 8;
const CROSS_COMMUNITY_PROB_PPM: u64 = 100_000; // 10 %

// -----------------------------------------------------------------------
// Xorshift RNG — zero-dep at bench time.
// -----------------------------------------------------------------------

struct Xorshift64(u64);
impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn range(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
}

struct Workload {
    _td: TempDir,
    store: SubstrateStore,
}

fn build_workload(n: usize) -> Workload {
    let td = tempfile::tempdir().expect("tempdir");
    let path = td.path().join("ricci-bench");
    let store = SubstrateStore::create(&path).expect("create store");

    let community_count = (n / NODES_PER_COMMUNITY).max(1) as u32;
    let mut nodes: Vec<NodeId> = Vec::with_capacity(n);
    let mut by_community: Vec<Vec<NodeId>> = (0..=community_count).map(|_| Vec::new()).collect();
    'outer: for cid in 1..=community_count {
        for _ in 0..NODES_PER_COMMUNITY {
            if nodes.len() >= n {
                break 'outer;
            }
            let id = store.create_node_in_community(&["N"], cid);
            nodes.push(id);
            by_community[cid as usize].push(id);
        }
    }

    let mut rng = Xorshift64::new(0xC0DE_F00D_F00D_BABE);
    for i in 0..nodes.len() {
        let src = nodes[i];
        let src_cid = ((i / NODES_PER_COMMUNITY) as u32 + 1).min(community_count);
        for _ in 0..AVG_DEGREE {
            let cross = rng.range(1_000_000) < CROSS_COMMUNITY_PROB_PPM;
            let dst = if cross {
                let idx = rng.range(nodes.len() as u64) as usize;
                nodes[idx]
            } else {
                let bucket = &by_community[src_cid as usize];
                if bucket.is_empty() {
                    continue;
                }
                bucket[rng.range(bucket.len() as u64) as usize]
            };
            if dst == src {
                continue;
            }
            let _ = store.create_edge(src, dst, "ACT");
        }
    }

    store.flush().expect("flush");

    Workload { _td: td, store }
}

// -----------------------------------------------------------------------
// Criterion entry — one iteration per sample. Phase 4 is idempotent
// so re-running measures steady-state overhead only; the first
// sample includes the initial-quantisation cost.
// -----------------------------------------------------------------------

fn bench_ricci_batch(c: &mut Criterion) {
    let n = workload_n();
    eprintln!("[ricci_batch] building workload: n = {n}");
    let w = build_workload(n);
    eprintln!(
        "[ricci_batch] workload ready: {} live nodes, {} live edges",
        GraphStore::node_count(&w.store),
        GraphStore::edge_count(&w.store)
    );

    let mut g = c.benchmark_group(format!("refresh_all_ricci_{n}"));
    g.measurement_time(Duration::from_secs(30));
    g.sample_size(10);

    g.bench_function("end_to_end", |b| {
        b.iter(|| {
            let stats: RicciRefreshStats = refresh_all_ricci(&w.store).expect("refresh");
            black_box(stats);
        });
    });

    g.finish();
}

criterion_group!(benches, bench_ricci_batch);
criterion_main!(benches);
