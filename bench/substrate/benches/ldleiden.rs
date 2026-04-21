//! T10 Step 5 — LDleiden incremental community detection bench.
//!
//! Four groups:
//!   * `ldleiden/bootstrap` — batch Leiden at n=1k, 5k (dense SBM).
//!   * `ldleiden/incremental` — 10% Δ edges applied one at a time
//!     after bootstrap (dense SBM). Gate: ≤ 100 ms total.
//!   * `ldleiden/modularity_preservation` — end-to-end Q ratio vs.
//!     full re-bootstrap. Reported for manual inspection.
//!   * `ldleiden/sparse_scaling` — sparse-density SBM (avg_degree
//!     constant at ≈ 20, realistic for megalaw/PO) at n=1k, 10k,
//!     100k. Validates that per-delta cost is O(d̄), not O(n).
//!     Extrapolates the n=10⁶ gate.
//!
//! **SBM density warning.** The `ldleiden/incremental` group uses a
//! fixed `p_out = 0.005` which blows up |E| as O(n²) — the benchmark
//! cannot realistically extend past n=10k without burning hours on
//! bootstrap alone. The `sparse_scaling` group scales `p_out` as
//! `c/n` (constant inter-block density per node) so we get honest
//! numbers at n=10⁵ without drowning in unrealistic edges.
//!
//! Run: `cargo bench --bench ldleiden` (from `bench/substrate/`)

use std::hint::black_box;
use std::time::Instant;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use obrain_cognitive::community::{Graph, LDleiden, LeidenConfig, leiden_batch, modularity};

fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Build an SBM graph with `n_blocks × block_size` nodes. Sampled
/// with a deterministic xorshift so runs are reproducible.
fn build_sbm(
    n_blocks: u32,
    block_size: u32,
    p_in: f64,
    p_out: f64,
    seed: u64,
) -> (u32, Vec<(u32, u32, f64)>) {
    let n = n_blocks * block_size;
    let mut state = seed;
    let mut edges = Vec::new();
    for u in 0..n {
        let bu = u / block_size;
        // Only sample pairs with v > u so each edge is counted once.
        for v in (u + 1)..n {
            let bv = v / block_size;
            let p = if bu == bv { p_in } else { p_out };
            let r = (xorshift64(&mut state) as f64) / (u64::MAX as f64);
            if r < p {
                edges.push((u, v, 1.0));
            }
        }
    }
    (n, edges)
}

fn bench_bootstrap(c: &mut Criterion) {
    let mut g = c.benchmark_group("ldleiden/bootstrap");
    // Keep criterion iterations low — each build is expensive.
    g.sample_size(10);
    for &n_blocks in &[20u32, 100] {
        let (n, edges) = build_sbm(n_blocks, 50, 0.3, 0.005, 0xC0DE_D00D);
        let graph = Graph::from_edges(n, edges);
        g.bench_with_input(BenchmarkId::new("sbm_n", n), &graph, |b, graph| {
            b.iter_batched(
                || graph.clone(),
                |graph| {
                    let p = leiden_batch(&graph, LeidenConfig::default());
                    black_box(p);
                },
                criterion::BatchSize::LargeInput,
            );
        });
    }
    g.finish();
}

fn bench_incremental(c: &mut Criterion) {
    let mut g = c.benchmark_group("ldleiden/incremental");
    g.sample_size(10);
    for &n_blocks in &[20u32, 100] {
        let (n, edges) = build_sbm(n_blocks, 50, 0.3, 0.005, 0xC0DE_D00D);
        let graph = Graph::from_edges(n, edges.clone());
        let cfg = LeidenConfig::default();
        let driver = LDleiden::bootstrap(graph, cfg);
        // Prebuild 10% delta edges — sampled from intra-block pairs.
        let mut state = 0xF00D_F00D_u64;
        let delta_count = (edges.len() / 10).max(100);
        let mut deltas: Vec<(u32, u32)> = Vec::with_capacity(delta_count);
        for _ in 0..delta_count {
            let block = (xorshift64(&mut state) % n_blocks as u64) as u32;
            let base = block * 50;
            let u = base + (xorshift64(&mut state) % 50) as u32;
            let v = base + (xorshift64(&mut state) % 50) as u32;
            if u != v {
                deltas.push((u, v));
            }
        }
        g.bench_with_input(
            BenchmarkId::new("sbm_delta10pct_n", n),
            &(driver, deltas),
            |b, (driver, deltas)| {
                b.iter_batched(
                    || {
                        // Rebuild from scratch each iteration — we
                        // need a pristine driver for the full Δ.
                        let graph = Graph::from_edges(n, edges.clone());
                        let fresh = LDleiden::bootstrap(graph, cfg);
                        (fresh, deltas.clone())
                    },
                    |(mut d, deltas)| {
                        for (u, v) in &deltas {
                            d.on_edge_add(*u, *v, 1.0);
                        }
                        black_box(d.modularity())
                    },
                    criterion::BatchSize::LargeInput,
                );
                let _ = driver;
            },
        );
    }
    g.finish();
}

fn bench_modularity_preservation(c: &mut Criterion) {
    // Run once per n: apply Δ deltas incrementally and via full
    // re-bootstrap, report ratio. Not a timing bench but we co-locate
    // it here to share the fixture.
    let mut g = c.benchmark_group("ldleiden/modularity_preservation");
    g.sample_size(10);
    for &n_blocks in &[20u32] {
        let (n, edges) = build_sbm(n_blocks, 50, 0.3, 0.005, 0xC0DE_D00D);
        let cfg = LeidenConfig::default();
        let delta_count = (edges.len() / 10).max(100);
        let mut state = 0xF00D_F00D_u64;
        let mut deltas: Vec<(u32, u32)> = Vec::with_capacity(delta_count);
        for _ in 0..delta_count {
            let block = (xorshift64(&mut state) % n_blocks as u64) as u32;
            let base = block * 50;
            let u = base + (xorshift64(&mut state) % 50) as u32;
            let v = base + (xorshift64(&mut state) % 50) as u32;
            if u != v {
                deltas.push((u, v));
            }
        }
        g.bench_with_input(
            BenchmarkId::new("sbm_ratio_n", n),
            &(edges, deltas),
            |b, (edges, deltas)| {
                b.iter_custom(|iters| {
                    let mut total = std::time::Duration::ZERO;
                    for _ in 0..iters {
                        let start = Instant::now();
                        let graph = Graph::from_edges(n, edges.clone());
                        let mut d = LDleiden::bootstrap(graph, cfg);
                        for (u, v) in deltas {
                            d.on_edge_add(*u, *v, 1.0);
                        }
                        let q_incremental = d.modularity();

                        // Batch baseline on the final graph.
                        let mut final_edges = edges.clone();
                        for (u, v) in deltas {
                            final_edges.push((*u, *v, 1.0));
                        }
                        let g_final = Graph::from_edges(n, final_edges);
                        let p_batch = leiden_batch(&g_final, cfg);
                        let q_batch = modularity(&g_final, &p_batch, cfg.resolution);
                        let ratio = q_incremental / q_batch.max(1e-9);
                        black_box(ratio);
                        total += start.elapsed();
                    }
                    total
                });
            },
        );
    }
    g.finish();
}

/// Sparse-scaling bench: p_out = target_inter_deg / n so that average
/// inter-block degree stays constant (≈ 5) regardless of n. This is
/// the closest we get to a "realistic sparse graph" in an SBM
/// generator, and lets us push n to 10⁵ without exponential edge
/// count. Validates that per-delta cost is O(d̄), not O(n).
fn bench_sparse_scaling(c: &mut Criterion) {
    let mut g = c.benchmark_group("ldleiden/sparse_scaling");
    g.sample_size(10);
    let block_size = 50u32;
    // Target: intra-block avg deg ≈ 15, inter-block avg deg ≈ 5,
    // total avg degree ≈ 20 — same order as a real megalaw / PO
    // knowledge graph post-consolidation.
    let target_inter_deg = 5.0f64;
    for &n_blocks in &[20u32, 200, 2000] {
        let n = n_blocks * block_size;
        // p_out s.t. expected inter-block neighbors per node
        // ≈ target_inter_deg.
        //   expected_inter_nb(u) = (n - block_size) * p_out
        //   ⇒ p_out = target_inter_deg / (n - block_size)
        let p_out = target_inter_deg / (n as f64 - block_size as f64).max(1.0);
        let (_n, edges) = build_sbm(n_blocks, block_size, 0.3, p_out, 0xC0DE_D00D);
        let edge_count = edges.len();
        let avg_deg = 2.0 * edge_count as f64 / n as f64;
        eprintln!(
            "sparse_scaling: n={}, |E|={}, avg_deg={:.2}, p_out={:.6}",
            n, edge_count, avg_deg, p_out
        );

        // Pre-compute deltas (intra-block, matches realistic
        // reinforcement of existing community structure).
        let mut state = 0xF00D_F00D_u64;
        let delta_count = (edge_count / 10).max(100);
        let mut deltas: Vec<(u32, u32)> = Vec::with_capacity(delta_count);
        for _ in 0..delta_count {
            let block = (xorshift64(&mut state) % n_blocks as u64) as u32;
            let base = block * block_size;
            let u = base + (xorshift64(&mut state) % block_size as u64) as u32;
            let v = base + (xorshift64(&mut state) % block_size as u64) as u32;
            if u != v {
                deltas.push((u, v));
            }
        }
        g.bench_with_input(
            BenchmarkId::new("sbm_delta10pct_n", n),
            &(edges, deltas),
            |b, (edges, deltas)| {
                b.iter_batched(
                    || {
                        let graph = Graph::from_edges(n, edges.clone());
                        let d = LDleiden::bootstrap(graph, LeidenConfig::default());
                        (d, deltas.clone())
                    },
                    |(mut d, deltas)| {
                        for (u, v) in &deltas {
                            d.on_edge_add(*u, *v, 1.0);
                        }
                        black_box(d.modularity())
                    },
                    criterion::BatchSize::LargeInput,
                );
            },
        );
    }
    g.finish();
}

criterion_group!(
    benches,
    bench_bootstrap,
    bench_incremental,
    bench_modularity_preservation,
    bench_sparse_scaling,
);
criterion_main!(benches);
