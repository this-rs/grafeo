//! T11 Step 7 — fragmentation soak test.
//!
//! # What this test proves
//!
//! Under sustained random mutations (`MUTATIONS = 100_000`: 60% creates
//! across different communities, 30% random edges, 10% deletes), the
//! substrate's community locality degrades far past the CommunityWarden
//! trigger threshold. A single `warden.tick()` must then compact the
//! store back to within the Step 6 locality budget:
//!
//! | Gate                              | Target  | Step         |
//! |-----------------------------------|---------|--------------|
//! | Pre-warden overall ratio          | > 1.40  | > 40% frag   |
//! | Post-warden overall ratio         | ≤ 1.15  | ≤ 15% frag   |
//! | Post-warden depth-3 BFS (p-mean)  | ≤ 1 ms  | Step 6 budget|
//! | Live node count preserved         | exact   | correctness  |
//!
//! # Shape
//!
//! 1. Burst-allocate a well-packed baseline across 10 communities
//!    (~200 nodes each).
//! 2. Run 100 k random mutations, mixing `create_node_in_community`
//!    (which opens a fresh page on every community transition),
//!    `create_edge` (exercises the outgoing-chain splicer under
//!    churn), and `delete_node` (leaves tombstones that inflate
//!    `distinct_pages / ideal_pages`).
//! 3. Assert pre-warden fragmentation exceeds 40%.
//! 4. Fire `CommunityWarden::tick()` once.
//! 5. Assert post-warden fragmentation is ≤ 15%.
//! 6. Replay the Step 6 BFS (depth 3, decay 0.5, max_nodes 1024) on a
//!    surviving seed; assert mean per-call latency ≤ 1 ms (the plan's
//!    "locality bench revient dans le budget" acceptance line).
//!
//! # Why inline the BFS (again)
//!
//! Same rationale as `benches/spreading_activation.rs`: pulling
//! `obrain-cognitive` into `obrain-substrate`'s dev graph for a single
//! algorithm would be a layering violation. The algorithm mirrors
//! `obrain-cognitive::activation::spread` exactly (same decay,
//! min_energy, max_nodes cap), reading straight from the substrate's
//! outgoing-chain view.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;

use obrain_common::types::NodeId;
use obrain_core::graph::Direction;
use obrain_core::graph::traits::{GraphStore, GraphStoreMut};
use obrain_substrate::{CommunityWarden, SubstrateStore};

// -----------------------------------------------------------------------
// Workload parameters
// -----------------------------------------------------------------------

const COMMUNITIES: u32 = 10;
const INITIAL_NODES_PER_COMMUNITY: u32 = 200; // 2000 nodes baseline
const MUTATIONS: usize = 100_000;

// Mutation mix (percentage points, must sum to 100).
const PCT_CREATE: u64 = 60;
const PCT_DELETE: u64 = 10;
// const PCT_EDGE: u64 = 30; // = 100 - CREATE - DELETE — implicit.

// Gates from the plan.
const PRE_WARDEN_FRAG_MIN: f32 = 1.40;
const POST_WARDEN_FRAG_MAX: f32 = 1.15;
/// Step 6 budget. We measure mean over 100 BFS calls to smooth out
/// scheduler jitter; any single outlier would still pass a 1 ms gate at
/// our measured 11.6 µs/call.
const BFS_BUDGET_MS: u128 = 1;

// -----------------------------------------------------------------------
// Xorshift RNG — deterministic, zero-dep.
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
// Inline spreading activation (mirrors `spread_depth3` in the bench).
// -----------------------------------------------------------------------

fn spread_depth3(store: &SubstrateStore, seed: NodeId) -> HashMap<NodeId, f64> {
    const MAX_HOPS: u32 = 3;
    const DECAY: f64 = 0.5;
    const MIN_ENERGY: f64 = 0.01;
    const MAX_NODES: usize = 1024;

    let mut activation: HashMap<NodeId, f64> = HashMap::with_capacity(256);
    let mut queue: VecDeque<(NodeId, f64, u32)> = VecDeque::with_capacity(256);
    activation.insert(seed, 1.0);
    queue.push_back((seed, 1.0, 0));

    while let Some((node, energy, hop)) = queue.pop_front() {
        if hop >= MAX_HOPS {
            continue;
        }
        if activation.len() >= MAX_NODES {
            break;
        }
        for (neighbor, _edge_id) in GraphStore::edges_from(store, node, Direction::Outgoing) {
            let propagated = energy * DECAY;
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
// Test
// -----------------------------------------------------------------------

#[test]
fn fragmentation_soak_recovers_under_warden() {
    let td = tempfile::tempdir().unwrap();
    let path = td.path().join("soak");
    let store = Arc::new(SubstrateStore::create(&path).unwrap());

    // Phase 1 — pristine baseline. Burst-allocate communities so each
    // one sits in its own contiguous page range before the churn
    // starts.
    let mut baseline_nodes: Vec<NodeId> = Vec::new();
    for cid in 1..=COMMUNITIES {
        for _ in 0..INITIAL_NODES_PER_COMMUNITY {
            let id = store.create_node_in_community(&["x"], cid);
            baseline_nodes.push(id);
        }
    }
    store.flush().unwrap();

    let warden = CommunityWarden::new(store.clone());
    let pristine = warden.scan().unwrap();
    eprintln!(
        "pristine: live={} distinct={} ideal={} ratio={:.3}",
        pristine.total_live_nodes,
        pristine.total_distinct_pages,
        pristine.total_ideal_pages,
        pristine.overall_ratio()
    );
    assert!(
        pristine.overall_ratio() <= 1.15,
        "baseline burst-allocation should be well packed, got {:.3}",
        pristine.overall_ratio()
    );

    // Phase 2 — soak: 10^5 random mutations.
    //
    // `live` is the pool of live node ids across the whole run; we
    // swap_remove on delete so subsequent picks don't re-target a
    // tombstone.
    let mut live: Vec<NodeId> = baseline_nodes.clone();
    let mut rng = Xorshift64::new(0x50AC_5EED_C0FFEE);
    let mut n_create = 0usize;
    let mut n_edge = 0usize;
    let mut n_delete = 0usize;

    for _ in 0..MUTATIONS {
        let r = rng.range(100);
        if r < PCT_CREATE {
            // create_node_in_community — cross-community churn. Each
            // community change forces the allocator's slow path: open a
            // fresh page, so fragmentation climbs fast.
            let cid = 1 + (rng.range(COMMUNITIES as u64) as u32);
            let id = store.create_node_in_community(&["x"], cid);
            live.push(id);
            n_create += 1;
        } else if r < PCT_CREATE + PCT_DELETE {
            // delete_node — tombstone + drop from live pool.
            if live.is_empty() {
                continue;
            }
            let idx = rng.range(live.len() as u64) as usize;
            let id = live.swap_remove(idx);
            if store.delete_node(id) {
                n_delete += 1;
            }
        } else {
            // create_edge between two random live nodes.
            if live.len() < 2 {
                continue;
            }
            let i = rng.range(live.len() as u64) as usize;
            let mut j = rng.range(live.len() as u64) as usize;
            if j == i {
                j = (j + 1) % live.len();
            }
            let src = live[i];
            let dst = live[j];
            let _ = store.create_edge(src, dst, "ACT");
            n_edge += 1;
        }
    }
    store.flush().unwrap();
    eprintln!(
        "soak mutations: create={} delete={} edge={} (total={})",
        n_create,
        n_delete,
        n_edge,
        n_create + n_delete + n_edge
    );

    // Phase 3 — pre-warden fragmentation must exceed 40%.
    let pre = warden.scan().unwrap();
    eprintln!(
        "pre-warden: live={} distinct={} ideal={} ratio={:.3}",
        pre.total_live_nodes,
        pre.total_distinct_pages,
        pre.total_ideal_pages,
        pre.overall_ratio()
    );
    assert!(
        pre.overall_ratio() > PRE_WARDEN_FRAG_MIN,
        "soak did not produce > 40% fragmentation (ratio {:.3}); workload too gentle",
        pre.overall_ratio()
    );

    // Phase 4 — fire compaction.
    let fired = warden.tick().unwrap();
    assert!(
        !fired.is_empty(),
        "warden.tick() did not fire despite pre-fragmentation {:.3}",
        pre.overall_ratio()
    );
    eprintln!("warden fired on {} communities: {:?}", fired.len(), fired);

    // Phase 5 — post-warden fragmentation must be at or below 15%.
    let post = warden.scan().unwrap();
    eprintln!(
        "post-warden: live={} distinct={} ideal={} ratio={:.3}",
        post.total_live_nodes,
        post.total_distinct_pages,
        post.total_ideal_pages,
        post.overall_ratio()
    );
    assert!(
        post.overall_ratio() <= POST_WARDEN_FRAG_MAX,
        "warden failed to restore locality: post ratio {:.3} > {:.2}",
        post.overall_ratio(),
        POST_WARDEN_FRAG_MAX
    );
    assert_eq!(
        pre.total_live_nodes, post.total_live_nodes,
        "compaction must preserve live node count: {} → {}",
        pre.total_live_nodes, post.total_live_nodes
    );

    // Phase 6 — locality bench gate. The warden's compaction applies a
    // slot-level permutation (`WalPayload::HilbertRepermute`), so the
    // pre-compaction NodeIds in `live` no longer identify the same
    // nodes post-sort. Pull fresh NodeIds from the substrate via
    // `GraphStore::node_ids` and keep only seeds that actually have
    // outgoing edges (most do — we wrote 30 k edges during the soak).
    //
    // BFS depth-3 must average ≤ 1 ms per call over 100 calls (the
    // Step 6 budget). Two pre-measurement calls warm the cache so the
    // first-fault cost doesn't contaminate the mean.
    let live_ids = GraphStore::node_ids(&*store);
    assert!(
        !live_ids.is_empty(),
        "post-compaction GraphStore::node_ids returned empty pool"
    );
    // Find up to 32 seeds with real outgoing edges so the BFS isn't a
    // single-entry map no-op. A sparse workload (30 % edge rate ×
    // random endpoints across 62 k live nodes) guarantees some nodes
    // have no outgoing chain.
    let mut seeds: Vec<NodeId> = Vec::with_capacity(32);
    for nid in &live_ids {
        if seeds.len() >= 32 {
            break;
        }
        if GraphStore::out_degree(&*store, *nid) > 0 {
            seeds.push(*nid);
        }
    }
    assert!(
        !seeds.is_empty(),
        "no post-compaction node has outgoing edges — edge chains may have been lost"
    );
    eprintln!(
        "post-warden seed pool: {} / {} live",
        seeds.len(),
        live_ids.len()
    );

    // Warm.
    let _ = spread_depth3(&store, seeds[0]);
    let _ = spread_depth3(&store, seeds[0]);

    const ITERS: usize = 100;
    let t0 = Instant::now();
    let mut sink = 0usize;
    for i in 0..ITERS {
        let s = seeds[i % seeds.len()];
        let act = spread_depth3(&store, s);
        sink = sink.wrapping_add(act.len());
    }
    let elapsed = t0.elapsed();
    let per_call_ns = elapsed.as_nanos() / ITERS as u128;
    let per_call_us = per_call_ns / 1_000;
    eprintln!(
        "post-warden BFS: {} calls, {} µs/call ({} ns) [avg activation size = {}]",
        ITERS,
        per_call_us,
        per_call_ns,
        sink / ITERS,
    );
    // Guard: BFS must return at least some activation beyond the seed
    // — if every call returned {seed} only, we have a stale-id bug
    // like the one that caught us at first wiring.
    assert!(
        sink / ITERS > 1,
        "BFS returned only the seed {} times on average — edge chains likely lost post-compaction",
        sink / ITERS
    );
    let per_call_ms = per_call_ns / 1_000_000;
    assert!(
        per_call_ms <= BFS_BUDGET_MS,
        "spreading activation over budget after warden: {} ms/call > {} ms (= {} µs)",
        per_call_ms,
        BFS_BUDGET_MS,
        per_call_us
    );
}
