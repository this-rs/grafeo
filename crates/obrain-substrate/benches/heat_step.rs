//! T12 Step 4 — heat kernel step latency bench.
//!
//! # Acceptance criterion
//!
//! One explicit-Euler heat step on a 10⁶-node community-partitioned
//! graph with `AVG_DEGREE ≈ 8` (≈ 8·10⁶ directed edges total) must
//! complete in **≤ 5 ms** — the plan's gate for letting Thinkers
//! spread activation at interactive latency inside the 100 ms chat
//! budget.
//!
//! # Shape of the bench
//!
//! Two Criterion groups:
//!
//! 1. `heat_step_unweighted` — pure topology pass. Every live
//!    directed edge contributes weight = 1 to the Laplacian. Isolates
//!    the memory-bandwidth-bound code path (state-vector + adjacency
//!    walk, no per-edge weight fetch).
//!
//! 2. `heat_step_weighted` — realistic pass. Every live directed
//!    edge's `weight_u16` is fetched from the EdgeRecord (32 B mmap
//!    read). This is the code path Thinkers will actually drive.
//!    The extra per-edge fetch pushes the working set from ~4 MB
//!    (state only) to ~40 MB (state + edge records), which is
//!    safely outside the M2 L3 and the bench measures DRAM
//!    bandwidth directly.
//!
//! # Scale control
//!
//! Default `OBRAIN_BENCH_N=100_000` keeps workload setup below a
//! minute. Set `OBRAIN_BENCH_N=1000000` to validate the ≤ 5 ms gate.
//!
//! ```bash
//! cargo bench -p obrain-substrate --bench heat_step
//! OBRAIN_BENCH_N=1000000 cargo bench -p obrain-substrate --bench heat_step
//! ```
//!
//! The workload construction (burst-allocated communities, uniform
//! AVG_DEGREE, CROSS_COMMUNITY 10 %) mirrors `ricci_batch.rs` and
//! `spreading_activation.rs` so regressions compose like-for-like.

use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use obrain_common::types::NodeId;
use obrain_core::graph::traits::{GraphStore, GraphStoreMut};
use obrain_substrate::record::EdgeRecord;
use obrain_substrate::store::SubstrateStore;
use obrain_substrate::{
    heat_step_unweighted, heat_step_unweighted_csr, heat_step_weighted,
    heat_step_weighted_csr, CsrAdjacency,
};
use tempfile::TempDir;

// -----------------------------------------------------------------------
// Workload parameters — shared with the other T11/T12 benches so a
// regression table can quote one `OBRAIN_BENCH_N` across the suite.
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
    state: Vec<f32>,
}

fn build_workload(n: usize, seed_with_weights: bool) -> Workload {
    let td = tempfile::tempdir().expect("tempdir");
    let path = td.path().join("heat-bench");
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

    let mut rng = Xorshift64::new(0xDEAD_F00D_CAFE_BEEF);
    let mut edge_ids = Vec::new();
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
            let eid = store.create_edge(src, dst, "ACT");
            edge_ids.push(eid);
        }
    }

    // If the weighted bench path is being set up, stamp every edge's
    // weight to a non-zero value (0.5) so the weighted step does
    // actual work rather than short-circuiting on weight == 0.
    if seed_with_weights {
        for eid in &edge_ids {
            if let Ok(Some(mut rec)) = store.writer().read_edge(eid.0) {
                if rec.is_tombstoned() {
                    continue;
                }
                rec.set_weight_f32(0.5);
                let _ = store.writer().update_edge(eid.0, rec);
            }
        }
    }
    // Silence unused-var warning on the else branch.
    let _ = EdgeRecord::default();

    store.flush().expect("flush");

    // Preallocate state buffer sized to slot_high_water (+ 1 to
    // guarantee >= exclusive bound). Place a single Dirac at the
    // first live slot so the first step does non-trivial work.
    let hw = store.slot_high_water() as usize + 1;
    let mut state = vec![0.0_f32; hw];
    if let Some(n0) = nodes.first() {
        state[n0.0 as usize] = 1.0;
    }

    Workload { _td: td, store, state }
}

// -----------------------------------------------------------------------
// Criterion entry point — one group, two functions.
// -----------------------------------------------------------------------

fn bench_heat_step(c: &mut Criterion) {
    let n = workload_n();

    // Each path gets its own workload: the weighted path needs every
    // edge's weight_u16 set to non-zero (the unweighted path ignores
    // weights entirely, so sharing would just waste setup time).
    eprintln!("[heat_step] building unweighted workload: n = {n}");
    let w_unweighted = build_workload(n, /* seed_with_weights */ false);
    eprintln!(
        "[heat_step] unweighted ready: {} live nodes, {} live edges",
        GraphStore::node_count(&w_unweighted.store),
        GraphStore::edge_count(&w_unweighted.store)
    );

    // Build the CSR once per workload — the realistic Thinker usage:
    // the snapshot is taken on a Dreamer tick and reused across K
    // diffusion steps. The benches measure STEP time; the build
    // cost is documented via the `csr_build` bench below.
    let csr_unweighted = CsrAdjacency::build(&w_unweighted.store);
    eprintln!(
        "[heat_step] unweighted CSR: {} row_offsets, {} edges",
        csr_unweighted.row_offsets.len(),
        csr_unweighted.edge_count()
    );

    let mut g = c.benchmark_group(format!("heat_step_unweighted_{n}"));
    g.measurement_time(Duration::from_secs(10));
    g.sample_size(20);
    g.bench_function("one_step_no_csr", |b| {
        // Legacy path — rebuilds CSR on every step. Useful for
        // small workloads or one-off diagnostics.
        let mut s = w_unweighted.state.clone();
        b.iter(|| {
            heat_step_unweighted(&w_unweighted.store, &mut s, 0.125).expect("step");
            black_box(&s);
        });
    });
    g.bench_function("one_step_csr", |b| {
        // Fast path — the Thinker / Predictor usage. CSR built
        // once above, reused on every step.
        let mut s = w_unweighted.state.clone();
        b.iter(|| {
            heat_step_unweighted_csr(&csr_unweighted, &mut s, 0.125).expect("step");
            black_box(&s);
        });
    });
    g.finish();

    eprintln!("[heat_step] building weighted workload: n = {n}");
    let w_weighted = build_workload(n, /* seed_with_weights */ true);
    eprintln!(
        "[heat_step] weighted ready: {} live nodes, {} live edges",
        GraphStore::node_count(&w_weighted.store),
        GraphStore::edge_count(&w_weighted.store)
    );
    let csr_weighted = CsrAdjacency::build(&w_weighted.store);
    eprintln!(
        "[heat_step] weighted CSR: {} row_offsets, {} edges",
        csr_weighted.row_offsets.len(),
        csr_weighted.edge_count()
    );

    let mut g = c.benchmark_group(format!("heat_step_weighted_{n}"));
    g.measurement_time(Duration::from_secs(10));
    g.sample_size(20);
    g.bench_function("one_step_no_csr", |b| {
        let mut s = w_weighted.state.clone();
        b.iter(|| {
            heat_step_weighted(&w_weighted.store, &mut s, 0.125).expect("step");
            black_box(&s);
        });
    });
    g.bench_function("one_step_csr", |b| {
        let mut s = w_weighted.state.clone();
        b.iter(|| {
            heat_step_weighted_csr(&csr_weighted, &mut s, 0.125).expect("step");
            black_box(&s);
        });
    });

    // Also measure the CSR build cost — this is how much setup the
    // caller pays once per Dreamer tick.
    g.bench_function("csr_build", |b| {
        b.iter(|| {
            let csr = CsrAdjacency::build(&w_weighted.store);
            black_box(csr);
        });
    });
    g.finish();
}

criterion_group!(benches, bench_heat_step);
criterion_main!(benches);
