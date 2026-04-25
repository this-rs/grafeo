//! T12 Step 10 — Heat Kernel Signature refresh latency bench.
//!
//! # Acceptance criterion
//!
//! A full 3-scale HKS refresh on a 10⁶-node substrate with
//! `AVG_DEGREE ≈ 8` must complete in **≤ 150 ms** (gate per scale is
//! 50 ms, three scales = 150 ms total), running the Hutchinson
//! stochastic diagonal estimator with K=32 Rademacher probes.
//!
//! # Shape of the bench
//!
//! Three benchmark functions:
//!
//! 1. `hks_local` — single-scale HKS at `t=0.1` (≈ 1 heat step per probe).
//! 2. `hks_meso` — single-scale HKS at `t=1.0` (≈ 8 heat steps per probe).
//! 3. `hks_global` — single-scale HKS at `t=10.0` (≈ 80 heat steps per probe).
//! 4. `hks_all_scales` — [`compute_hks_descriptors`] end-to-end.
//!
//! The per-scale benches help pinpoint where the time goes: HKS cost
//! is dominated by `heat_step_unweighted_csr` × `HKS_HUTCHINSON_PROBES`
//! × `n_steps(t)`, so local is cheapest and global is dominant.
//!
//! # Run
//!
//! ```bash
//! cargo bench -p obrain-substrate --bench hks_refresh
//! OBRAIN_BENCH_N=1000000 cargo bench -p obrain-substrate --bench hks_refresh
//! ```

use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use obrain_common::types::NodeId;
use obrain_core::graph::traits::{GraphStore, GraphStoreMut};
use obrain_substrate::store::SubstrateStore;
use obrain_substrate::{
    CsrAdjacency, HKS_T_GLOBAL, HKS_T_LOCAL, HKS_T_MESO, compute_hks_descriptors,
    heat_kernel_signature_with_csr,
};
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
    let path = td.path().join("hks-bench");
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

    let mut rng = Xorshift64::new(0xCAFE_0017_BEEF_DEAD);
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

fn bench_hks(c: &mut Criterion) {
    let n = workload_n();
    eprintln!("[hks_refresh] building workload: n = {n}");
    let w = build_workload(n);
    eprintln!(
        "[hks_refresh] workload ready: {} live nodes, {} live edges",
        GraphStore::node_count(&w.store),
        GraphStore::edge_count(&w.store)
    );

    // One CSR snapshot per run — realistic Dreamer usage.
    let csr = CsrAdjacency::build(&w.store);
    eprintln!(
        "[hks_refresh] CSR ready: {} row_offsets, {} edges",
        csr.row_offsets.len(),
        csr.edge_count()
    );

    let mut g = c.benchmark_group(format!("hks_refresh_{n}"));
    g.measurement_time(Duration::from_secs(15));
    g.sample_size(10);

    g.bench_function("hks_local", |b| {
        b.iter(|| {
            let v = heat_kernel_signature_with_csr(&csr, HKS_T_LOCAL).expect("hks");
            black_box(v);
        });
    });
    g.bench_function("hks_meso", |b| {
        b.iter(|| {
            let v = heat_kernel_signature_with_csr(&csr, HKS_T_MESO).expect("hks");
            black_box(v);
        });
    });
    g.bench_function("hks_global", |b| {
        b.iter(|| {
            let v = heat_kernel_signature_with_csr(&csr, HKS_T_GLOBAL).expect("hks");
            black_box(v);
        });
    });
    g.bench_function("hks_all_scales", |b| {
        b.iter(|| {
            let v = compute_hks_descriptors(&w.store).expect("hks");
            black_box(v);
        });
    });

    g.finish();
}

criterion_group!(benches, bench_hks);
criterion_main!(benches);
