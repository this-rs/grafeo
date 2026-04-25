//! T11 Step 6 — spreading activation latency bench on a community-partitioned graph.
//!
//! # Acceptance criterion
//!
//! Depth-3 BFS spreading activation from a single seed on a 10⁶-node
//! graph must complete in ≤ 1 ms when the `on_node_activated` prefetch
//! hook is wired on seed entry (plan: "depth 3 sur 10⁶ nodes ≤ 1 ms,
//! ratio ≥ 5× speedup vs 5-20 ms baseline").
//!
//! # Shape of the bench
//!
//! A Criterion micro-bench with three groups:
//! 1. `baseline_no_prefetch` — call BFS directly, no prefetch hint.
//! 2. `with_prefetch` — call `store.on_node_activated(seed)` before
//!    the BFS, same traversal.
//! 3. `prefetch_alone` — only the prefetch call itself, to isolate
//!    the hook's per-activation overhead.
//!
//! # Scale control
//!
//! Setting `OBRAIN_BENCH_N=1000000` at invocation time runs the full
//! 10⁶-node verification. Default `100_000` nodes keeps the bench
//! below 30 s of setup and is still large enough to exceed L3 cache,
//! so the madvise effect is visible on a cold cache.
//!
//! # Run
//!
//! ```bash
//! # Default (100k nodes)
//! cargo bench -p obrain-substrate --bench spreading_activation
//!
//! # Full acceptance scale (10⁶ nodes)
//! OBRAIN_BENCH_N=1000000 cargo bench -p obrain-substrate --bench spreading_activation
//! ```
//!
//! # Why we reimplement the BFS inline
//!
//! `obrain-cognitive::activation::spread` depends on `SynapseStore`
//! and a cognitive-layer `ActivationSource` trait, which would pull
//! the whole cognitive crate into obrain-substrate's dev graph — a
//! layering violation. The inline BFS mirrors the same algorithm
//! (energy × weight × decay, visited-at-depth HashMap, max_hops cap)
//! reading directly from the substrate's durable EdgeRecord chain,
//! so what we benchmark is the topology-as-storage traversal — not
//! the cognitive crate's overhead.

use std::collections::HashMap;
use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use obrain_common::types::NodeId;
use obrain_core::graph::Direction;
use obrain_core::graph::traits::{GraphStore, GraphStoreMut};
use obrain_substrate::store::SubstrateStore;
use tempfile::TempDir;

// -----------------------------------------------------------------------
// Workload parameters
// -----------------------------------------------------------------------

fn workload_n() -> usize {
    std::env::var("OBRAIN_BENCH_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000)
}

const NODES_PER_COMMUNITY: usize = 512;
const AVG_DEGREE: usize = 8;
/// Chance of a cross-community edge (vs intra-community).
const CROSS_COMMUNITY_PROB_PPM: u64 = 100_000; // 10 %

// -----------------------------------------------------------------------
// Xorshift RNG — deterministic, zero-dep at bench time.
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

// -----------------------------------------------------------------------
// Workload construction
// -----------------------------------------------------------------------

struct Workload {
    _td: TempDir, // keep the dir alive for the bench lifetime
    store: SubstrateStore,
    seeds: Vec<NodeId>,
}

fn build_workload(n: usize) -> Workload {
    let td = tempfile::tempdir().expect("tempdir");
    let path = td.path().join("spread-bench");
    let store = SubstrateStore::create(&path).expect("create store");

    // Step 1: create nodes, BURST-allocated per community — all of
    // community 1, then all of community 2, etc. This matches the
    // post-Hilbert-sort layout (communities physically contiguous on
    // disk) and is what the prefetch hook is designed to exploit.
    //
    // Interleaved allocation (one node per community per round) was
    // tried first; it produces a worst-case fragmentation where each
    // community's bounding range covers the whole file, so
    // `prefetch_community` advises every page and becomes useless.
    // Step 3's bulk_sort_by_hilbert rearranges to this burst layout in
    // production; we skip the sort here since the allocator already
    // gives us the target layout when we allocate in bursts.
    let community_count = (n / NODES_PER_COMMUNITY).max(1) as u32;
    let mut nodes: Vec<NodeId> = Vec::with_capacity(n);
    // Per-community bucket, so edge generation can pick same-community
    // neighbors by lookup into the bucket (O(1)) rather than slot
    // arithmetic that races the allocator's community-offset math.
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

    // Step 2: wire edges. Each node gets AVG_DEGREE edges to neighbors,
    // mostly intra-community (picked from the community's own bucket
    // for true locality), with a configurable cross-community rate.
    let mut rng = Xorshift64::new(0x5EED_BEEF_C0FFEE);
    for i in 0..nodes.len() {
        let src = nodes[i];
        // Burst allocation => node[i] is in community `1 + (i / NODES_PER_COMMUNITY)`
        // as long as i < community_count * NODES_PER_COMMUNITY.
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

    // Step 3: choose a deterministic pool of seeds across different
    // communities so the bench doesn't only exercise one page range.
    let seed_count = 64.min(nodes.len());
    let mut seeds = Vec::with_capacity(seed_count);
    for i in 0..seed_count {
        seeds.push(nodes[(i * 1013) % nodes.len()]);
    }

    store.flush().expect("flush");

    // Sanity: every seed must have real outgoing edges, otherwise the
    // bench measures an empty BFS (108 ns HashMap-alloc artefact seen
    // during initial wiring).
    use obrain_core::graph::Direction as _Dir;
    let sample_out = GraphStore::edges_from(&store, seeds[0], _Dir::Outgoing).len();
    assert!(
        sample_out > 0,
        "workload bug: seed has no outgoing edges — BFS would be empty",
    );

    Workload {
        _td: td,
        store,
        seeds,
    }
}

// -----------------------------------------------------------------------
// Inline spreading activation — substrate-direct, no cognitive crate.
// -----------------------------------------------------------------------

/// BFS spreading activation with hop-depth cap, energy cutoff and
/// per-hop decay. Returns the activation map. Mirrors the algorithm in
/// `obrain-cognitive::activation::spread` but reads the neighbor list
/// straight from the substrate's outgoing-chain view.
fn spread_depth3(store: &SubstrateStore, seed: NodeId) -> HashMap<NodeId, f64> {
    const MAX_HOPS: u32 = 3;
    const DECAY: f64 = 0.5;
    const MIN_ENERGY: f64 = 0.01;
    const MAX_NODES: usize = 1024;

    let mut activation: HashMap<NodeId, f64> = HashMap::with_capacity(256);
    let mut queue: std::collections::VecDeque<(NodeId, f64, u32)> =
        std::collections::VecDeque::with_capacity(256);
    activation.insert(seed, 1.0);
    queue.push_back((seed, 1.0, 0));

    while let Some((node, energy, hop)) = queue.pop_front() {
        if hop >= MAX_HOPS {
            continue;
        }
        if activation.len() >= MAX_NODES {
            break;
        }
        for (neighbor, _edge_id) in store.edges_from(node, Direction::Outgoing) {
            // Constant weight 1.0: the substrate's EdgeRecord carries
            // weight_u16, but plumbing the Q0.16 quantization adds
            // cycles unrelated to topology traversal. We're measuring
            // locality, not numerics.
            let propagated = energy * 1.0 * DECAY;
            if propagated < MIN_ENERGY {
                continue;
            }
            let slot = activation.entry(neighbor).or_insert(0.0);
            *slot += propagated;
            let next_hop = hop + 1;
            if next_hop < MAX_HOPS {
                queue.push_back((neighbor, propagated, next_hop));
            }
        }
    }
    activation
}

// -----------------------------------------------------------------------
// Criterion entry points
// -----------------------------------------------------------------------

fn bench_spreading_activation(c: &mut Criterion) {
    let n = workload_n();
    eprintln!("[spreading_activation] building workload: n = {n}");
    let w = build_workload(n);
    eprintln!(
        "[spreading_activation] workload ready: {} seeds",
        w.seeds.len()
    );

    let mut g = c.benchmark_group(format!("spreading_activation_depth3_{n}"));
    // 1ms target — Criterion's default warm-up and 100-sample config
    // is fine; the setup is paid once per bench run.
    g.measurement_time(Duration::from_secs(5));

    let mut idx = 0usize;
    g.bench_function("baseline_no_prefetch", |b| {
        b.iter(|| {
            let seed = w.seeds[idx % w.seeds.len()];
            idx = idx.wrapping_add(1);
            let act = spread_depth3(&w.store, black_box(seed));
            black_box(act)
        });
    });

    let mut idx = 0usize;
    g.bench_function("with_prefetch", |b| {
        b.iter(|| {
            let seed = w.seeds[idx % w.seeds.len()];
            idx = idx.wrapping_add(1);
            let _ = w.store.on_node_activated(black_box(seed));
            let act = spread_depth3(&w.store, black_box(seed));
            black_box(act)
        });
    });

    let mut idx = 0usize;
    g.bench_function("prefetch_alone", |b| {
        b.iter(|| {
            let seed = w.seeds[idx % w.seeds.len()];
            idx = idx.wrapping_add(1);
            let _ = w.store.on_node_activated(black_box(seed));
        });
    });

    g.finish();
}

criterion_group!(benches, bench_spreading_activation);
criterion_main!(benches);
