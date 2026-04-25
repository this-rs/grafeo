//! T12 Step 9 — Effective resistance latency bench.
//!
//! # Acceptance criterion
//!
//! Per-pair effective resistance via conjugate-gradient on a 10⁶-node
//! substrate with `AVG_DEGREE ≈ 8` must complete in **≤ 1 ms amortised
//! per pair** (gate: 1000 pairs ≤ 1 s on a prebuilt CSR snapshot).
//!
//! Each CG call is bounded by `max_iter = 200` iterations and a
//! relative residual tolerance of `1e-5`. On a connected graph the
//! iterations needed are typically ≪ 200 because the Laplacian is
//! diagonal-dominant and CG converges in ~√κ(L) steps where κ is the
//! condition number.
//!
//! # Shape of the bench
//!
//! One benchmark: `effective_resistance_random_pairs` — per iter, 10
//! random (u, v) pairs → 10 CG solves. Dividing the Criterion output
//! by 10 gives per-pair latency.
//!
//! # Run
//!
//! ```bash
//! cargo bench -p obrain-substrate --bench effective_resistance
//! OBRAIN_BENCH_N=1000000 cargo bench -p obrain-substrate --bench effective_resistance
//! ```

use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use obrain_common::types::NodeId;
use obrain_core::graph::traits::{GraphStore, GraphStoreMut};
use obrain_substrate::store::SubstrateStore;
use obrain_substrate::{CsrAdjacency, effective_resistance_csr};
use tempfile::TempDir;

fn workload_n() -> usize {
    std::env::var("OBRAIN_BENCH_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000)
}

const NODES_PER_COMMUNITY: usize = 512;
const AVG_DEGREE: usize = 8;
const CROSS_COMMUNITY_PROB_PPM: u64 = 100_000;
const PAIRS_PER_ITER: usize = 10;
const CG_MAX_ITER: usize = 200;
const CG_TOL: f32 = 1e-5;

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
    nodes: Vec<NodeId>,
}

fn build_workload(n: usize) -> Workload {
    let td = tempfile::tempdir().expect("tempdir");
    let path = td.path().join("er-bench");
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

    let mut rng = Xorshift64::new(0xE77E_EDEC_F00D_CAFE);
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
    Workload {
        _td: td,
        store,
        nodes,
    }
}

fn bench_effective_resistance(c: &mut Criterion) {
    let n = workload_n();
    eprintln!("[effective_resistance] building workload: n = {n}");
    let w = build_workload(n);
    eprintln!(
        "[effective_resistance] workload ready: {} live nodes, {} live edges",
        GraphStore::node_count(&w.store),
        GraphStore::edge_count(&w.store)
    );
    let csr = CsrAdjacency::build(&w.store);
    eprintln!(
        "[effective_resistance] CSR ready: {} row_offsets, {} edges",
        csr.row_offsets.len(),
        csr.edge_count()
    );

    let mut rng = Xorshift64::new(0x1337_BABE_B00B_CAFE);
    let pairs: Vec<(u32, u32)> = (0..PAIRS_PER_ITER)
        .map(|_| {
            let a = w.nodes[rng.range(w.nodes.len() as u64) as usize];
            let b = loop {
                let cand = w.nodes[rng.range(w.nodes.len() as u64) as usize];
                if cand != a {
                    break cand;
                }
            };
            (a.0 as u32, b.0 as u32)
        })
        .collect();

    let mut g = c.benchmark_group(format!("effective_resistance_{n}"));
    g.measurement_time(Duration::from_secs(15));
    g.sample_size(10);

    g.bench_function("random_pairs_10", |b| {
        b.iter(|| {
            let mut acc: f32 = 0.0;
            for (u, v) in pairs.iter() {
                if let Some(r) = effective_resistance_csr(&csr, *u, *v, CG_MAX_ITER, CG_TOL) {
                    acc += r;
                }
            }
            black_box(acc);
        });
    });

    g.finish();
}

criterion_group!(benches, bench_effective_resistance);
criterion_main!(benches);
