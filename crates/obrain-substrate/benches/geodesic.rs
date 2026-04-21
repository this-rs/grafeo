//! T12 Step 5 — Dijkstra geodesic distance latency bench.
//!
//! # Acceptance criterion
//!
//! For two nodes that are L2-connected (i.e. present in the same
//! weakly-connected component of the CSR snapshot), one call to
//! [`geodesic_distance_csr`] on a 10⁶-node substrate must complete
//! in **≤ 1 ms** at the p50.
//!
//! Dijkstra's worst case is O((V + E) log V); early-exit when the
//! target is dequeued keeps the amortised cost much lower for
//! semantically close pairs (the frontier expands only until the
//! target is popped).
//!
//! # Shape of the bench
//!
//! One benchmark: `geodesic_random_pairs` — for each iteration, pick
//! 100 random (src, dst) slot pairs and call `geodesic_distance_csr`
//! on each. Reporting is per-100 pairs, so divide the Criterion
//! output by 100 to get per-pair latency.
//!
//! # Run
//!
//! ```bash
//! cargo bench -p obrain-substrate --bench geodesic
//! OBRAIN_BENCH_N=1000000 cargo bench -p obrain-substrate --bench geodesic
//! ```

use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use obrain_common::types::NodeId;
use obrain_core::graph::traits::{GraphStore, GraphStoreMut};
use obrain_substrate::store::SubstrateStore;
use obrain_substrate::{CsrAdjacency, geodesic_distance_csr};
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
const PAIRS_PER_ITER: usize = 100;

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
    let path = td.path().join("geodesic-bench");
    let store = SubstrateStore::create(&path).expect("create store");

    let community_count = (n / NODES_PER_COMMUNITY).max(1) as u32;
    let mut nodes: Vec<NodeId> = Vec::with_capacity(n);
    let mut by_community: Vec<Vec<NodeId>> =
        (0..=community_count).map(|_| Vec::new()).collect();
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

    let mut rng = Xorshift64::new(0xDEAD_BEE7_F00D_ABBA);
    for i in 0..nodes.len() {
        let src = nodes[i];
        let src_cid =
            ((i / NODES_PER_COMMUNITY) as u32 + 1).min(community_count);
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
    Workload { _td: td, store, nodes }
}

fn bench_geodesic(c: &mut Criterion) {
    let n = workload_n();
    eprintln!("[geodesic] building workload: n = {n}");
    let w = build_workload(n);
    eprintln!(
        "[geodesic] workload ready: {} live nodes, {} live edges",
        GraphStore::node_count(&w.store),
        GraphStore::edge_count(&w.store)
    );
    let csr = CsrAdjacency::build(&w.store);
    eprintln!(
        "[geodesic] CSR ready: {} row_offsets, {} edges",
        csr.row_offsets.len(),
        csr.edge_count()
    );

    // Pre-generate a fixed set of src/dst pairs so the bench is
    // reproducible and the compiler can't hoist anything.
    let mut rng = Xorshift64::new(0x5EED_CAFE_1234_5678);
    let pairs: Vec<(u32, u32)> = (0..PAIRS_PER_ITER)
        .map(|_| {
            let a = w.nodes[rng.range(w.nodes.len() as u64) as usize];
            let b = w.nodes[rng.range(w.nodes.len() as u64) as usize];
            (a.0 as u32, b.0 as u32)
        })
        .collect();

    let mut g = c.benchmark_group(format!("geodesic_{n}"));
    g.measurement_time(Duration::from_secs(10));
    g.sample_size(20);

    g.bench_function("random_pairs_100", |b| {
        b.iter(|| {
            let mut acc: f32 = 0.0;
            for (s, d) in pairs.iter() {
                if let Some(x) = geodesic_distance_csr(&csr, *s, *d) {
                    acc += x;
                }
            }
            black_box(acc);
        });
    });

    g.finish();
}

criterion_group!(benches, bench_geodesic);
criterion_main!(benches);
