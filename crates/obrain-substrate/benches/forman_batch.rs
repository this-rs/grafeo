//! T12 Step 8 — Forman-Ricci batch compute latency bench.
//!
//! # Acceptance criterion
//!
//! A full-graph [`compute_all_forman`] on a 10⁶-node substrate with
//! `AVG_DEGREE ≈ 8` (≈ 8·10⁶ directed edges total) must complete in
//! **≤ 200 ms on 8 cores** — the plan's explicit gate for
//! substituting Forman for Ollivier on the hot path. Forman is
//! closed-form `2 − deg(u) − deg(v) + 3·triangles`, so it's about
//! 10× faster than Jost-Liu Ollivier (which needs the W1 sampling).
//!
//! # Shape of the bench
//!
//! One Criterion group, single function `compute_all_forman` —
//! end-to-end batch pass. Setup mirrors `ricci_batch.rs` so a
//! regression on one composes like-for-like on the other.
//!
//! # Run
//!
//! ```bash
//! cargo bench -p obrain-substrate --bench forman_batch
//! OBRAIN_BENCH_N=1000000 cargo bench -p obrain-substrate --bench forman_batch
//! ```

use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use obrain_common::types::NodeId;
use obrain_core::graph::traits::{GraphStore, GraphStoreMut};
use obrain_substrate::compute_all_forman;
use obrain_substrate::store::SubstrateStore;
use tempfile::TempDir;

fn workload_n() -> usize {
    std::env::var("OBRAIN_BENCH_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000)
}

const NODES_PER_COMMUNITY: usize = 512;
const AVG_DEGREE: usize = 8;
const CROSS_COMMUNITY_PROB_PPM: u64 = 100_000; // 10 %

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
    let path = td.path().join("forman-bench");
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

    let mut rng = Xorshift64::new(0xF0FA_F0FA_BEEF_BEEF);
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

fn bench_forman_batch(c: &mut Criterion) {
    let n = workload_n();
    eprintln!("[forman_batch] building workload: n = {n}");
    let w = build_workload(n);
    eprintln!(
        "[forman_batch] workload ready: {} live nodes, {} live edges",
        GraphStore::node_count(&w.store),
        GraphStore::edge_count(&w.store)
    );

    let mut g = c.benchmark_group(format!("compute_all_forman_{n}"));
    g.measurement_time(Duration::from_secs(10));
    g.sample_size(10);

    g.bench_function("end_to_end", |b| {
        b.iter(|| {
            let map = compute_all_forman(&w.store).expect("forman");
            black_box(map);
        });
    });
    g.finish();
}

criterion_group!(benches, bench_forman_batch);
criterion_main!(benches);
