//! T10 Step 5 ORIG — performance gate at n = 10⁶.
//!
//! The acceptance criterion:
//!
//!   > 10⁶ nœuds + 10% Δ edges → update ≤ 100 ms
//!   > per-delta ≤ 1 µs at n = 1e6
//!
//! This is a slow test (≈ 30-60 s of bootstrap + a few seconds of
//! delta streaming). It is gated by `#[ignore]` so `cargo test` doesn't
//! run it by default. Invoke explicitly:
//!
//!     cargo test -p obrain-cognitive --features community \
//!         --release --test ldleiden_perf_gate -- --ignored --nocapture
//!
//! Run in `--release` — debug builds are ≈ 20× slower on this workload
//! and would give a false picture of production perf.
//!
//! ## Why a dedicated integration test, not a criterion bench
//!
//! Criterion at n=10⁶ budgets 10 samples × (bootstrap + delta stream)
//! ≈ 10 × 45 s = 7.5 min per run. That kills iteration speed on a
//! perf-regression hunt. This file instead runs one bootstrap, times
//! the delta stream in a single pass, and reports ns/delta + the
//! projected 10% full-Δ cost — enough signal to gate without burning
//! wall-clock on statistical repetition.

#![cfg(feature = "community")]

use obrain_cognitive::community::{Graph, LDleiden, LeidenConfig};
use std::time::Instant;

fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Sparse SBM: p_in chosen so expected intra-block degree ≈ 15;
/// p_out = target_inter / n so inter-block degree is O(1) independent
/// of n (matches the `bench_sparse_scaling` regime from
/// `bench/substrate/benches/ldleiden.rs`). Returns `(n, edges)` with
/// O(n · avg_deg) edges — usable at n = 10⁶ without drowning in memory.
fn sparse_sbm(
    n_blocks: u32,
    block_size: u32,
    target_intra_deg: f64,
    target_inter_deg: f64,
    seed: u64,
) -> (u32, Vec<(u32, u32, f64)>) {
    let n = n_blocks * block_size;
    let p_in = target_intra_deg / (block_size as f64 - 1.0).max(1.0);
    let p_out = target_inter_deg / (n as f64 - block_size as f64).max(1.0);

    let mut state = seed;
    let mut edges = Vec::with_capacity(((n as f64) * (target_intra_deg + target_inter_deg) / 2.0) as usize);

    // Intra-block edges: iterate only inside each block — O(n · block_size).
    for block in 0..n_blocks {
        let base = block * block_size;
        for u in base..(base + block_size) {
            for v in (u + 1)..(base + block_size) {
                let r = (xorshift64(&mut state) as f64) / (u64::MAX as f64);
                if r < p_in {
                    edges.push((u, v, 1.0));
                }
            }
        }
    }

    // Inter-block edges via Bernoulli sampling with the very low p_out.
    // Direct sampling of all n²/2 pairs is infeasible at n=10⁶; use
    // the geometric-skip trick: sample indices with expected gap 1/p_out.
    //
    // We iterate pairs (u, v) with u < v and bu != bv, and emit an edge
    // if `rand < p_out`. For n=10⁶, total inter-block pairs ≈ 5e11 —
    // still infeasible to scan linearly. We use the skip approach:
    // advance the pair index by a geometric jump and emit at each stop.
    let total_inter_pairs =
        (n as u64) * (n as u64 - block_size as u64) / 2;
    let expected_inter = (total_inter_pairs as f64) * p_out;
    // Safety: cap inter-edge count at 5× expected to bound worst case.
    let inter_cap = ((expected_inter * 5.0) as usize).max(1024);

    let mut pair_idx: u64 = 0;
    let mut inter_emitted = 0usize;
    while pair_idx < total_inter_pairs && inter_emitted < inter_cap {
        // Geometric jump: next_pair - current_pair ~ Geom(p_out).
        let r = (xorshift64(&mut state) as f64 / u64::MAX as f64).max(1e-18);
        let skip = (r.ln() / (1.0 - p_out).ln()).ceil() as u64;
        pair_idx += skip.max(1);
        if pair_idx >= total_inter_pairs {
            break;
        }
        // Decode (u, v) from pair_idx. Use a compact scheme: iterate
        // blocks and local ids. We need only approximate distribution
        // — every inter-block pair has the same p_out so the exact
        // mapping doesn't affect statistical properties.
        let u_raw = (pair_idx % n as u64) as u32;
        let v_raw = ((pair_idx / n as u64) as u32) % n;
        let (u, v) = if u_raw < v_raw {
            (u_raw, v_raw)
        } else if u_raw > v_raw {
            (v_raw, u_raw)
        } else {
            continue;
        };
        let bu = u / block_size;
        let bv = v / block_size;
        if bu != bv {
            edges.push((u, v, 1.0));
            inter_emitted += 1;
        }
    }
    (n, edges)
}

/// **T10 Step 5 ORIG gate** — per-delta ≤ 1 µs at n = 10⁶.
///
/// Ignored by default (takes ≈ 30-90 s in release mode depending on
/// hardware). Run with:
///
/// ```sh
/// cargo test -p obrain-cognitive --features community --release \
///     --test ldleiden_perf_gate -- --ignored --nocapture
/// ```
#[test]
#[ignore = "slow perf gate — run explicitly with --ignored in --release"]
fn perf_gate_1e6_per_delta_under_1us() {
    // Match the bench harness's sparse-scaling regime:
    // n_blocks × block_size = 20_000 × 50 = 1_000_000.
    let n_blocks = 20_000u32;
    let block_size = 50u32;
    let n = n_blocks * block_size;

    let t0 = Instant::now();
    let (_, edges) = sparse_sbm(n_blocks, block_size, 15.0, 5.0, 0xC0DE_D00D);
    let edges_built = t0.elapsed();
    eprintln!(
        "[setup] n={n}, |E|={}, avg_deg={:.2}, generator {:?}",
        edges.len(),
        2.0 * edges.len() as f64 / n as f64,
        edges_built,
    );

    let t1 = Instant::now();
    let graph = Graph::from_edges(n, edges.clone());
    let graph_built = t1.elapsed();
    eprintln!("[setup] graph built in {:?}", graph_built);

    let t2 = Instant::now();
    let mut driver = LDleiden::bootstrap(graph, LeidenConfig::default());
    let bootstrap_time = t2.elapsed();
    eprintln!(
        "[setup] bootstrap in {:?}, {} communities",
        bootstrap_time,
        driver.num_communities()
    );

    // 10% of the edge count as Δ stream (intra-block to match realistic
    // reinforcement patterns — same as the bench harness).
    let delta_count = edges.len() / 10;
    let mut deltas: Vec<(u32, u32)> = Vec::with_capacity(delta_count);
    let mut state = 0xF00D_F00D_u64;
    for _ in 0..delta_count {
        let block = (xorshift64(&mut state) % n_blocks as u64) as u32;
        let base = block * block_size;
        let u = base + (xorshift64(&mut state) % block_size as u64) as u32;
        let v = base + (xorshift64(&mut state) % block_size as u64) as u32;
        if u != v {
            deltas.push((u, v));
        }
    }
    eprintln!("[setup] {} deltas prepared", deltas.len());

    // Measure.
    let t3 = Instant::now();
    for (u, v) in &deltas {
        driver.on_edge_add(*u, *v, 1.0);
    }
    let stream_time = t3.elapsed();

    let per_delta_ns = stream_time.as_nanos() as f64 / deltas.len() as f64;
    let full_10pct_ms = (stream_time.as_nanos() as f64 / 1e6)
        * (deltas.len() as f64 / delta_count as f64);

    eprintln!(
        "\n[result] n=10⁶  |E|={}M  {} deltas  stream={:?}  per-delta={:.1} ns  full-10%={:.1} ms",
        edges.len() / 1_000_000,
        deltas.len(),
        stream_time,
        per_delta_ns,
        full_10pct_ms,
    );
    eprintln!(
        "[result] bootstrap (one-shot): {:?}",
        bootstrap_time
    );
    eprintln!(
        "[gate  ] target: ≤ 1000 ns/delta  →  observed: {:.1} ns  ({})",
        per_delta_ns,
        if per_delta_ns <= 1000.0 {
            "PASS"
        } else {
            "FAIL"
        }
    );

    assert!(
        per_delta_ns <= 1000.0,
        "per-delta {per_delta_ns:.1} ns > 1 µs gate at n=10⁶"
    );
}
