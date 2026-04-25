//! Criterion bench — retrieval latency on the L0 / L1 / L2 tier cascade.
//!
//! T8 Step 7 delivery. Companion to the acceptance bench
//! [`crates/obrain-substrate/benches/tiered_scan.rs`], which only exercises
//! the full pipeline at K = 10. This suite gives per-stage characterization
//! — it's what T9's retrieval gate uses to attribute latency to individual
//! tiers when a regression shows up.
//!
//! ## Benchmark groups
//!
//! | Group              | What it measures                                           |
//! |--------------------|------------------------------------------------------------|
//! | `L0_scan`          | Hamming 128-bit brute force, top-K over all nodes          |
//! | `L1_rerank`        | Hamming 512-bit re-rank on a pre-selected subset (K0)      |
//! | `L2_cosine`        | f16 384-dim cosine exact, on a pre-selected subset (K1)    |
//! | `pipeline_topk`    | Full L0 → L1 → L2 cascade, K ∈ {1, 10, 100}                |
//! | `pipeline_threads` | Full cascade parallel vs serial on the target K            |
//!
//! ## Corpus size
//!
//! Defaults to 10⁶ nodes (the T8 / T9 acceptance size). Override via env:
//!
//! ```bash
//! TIERED_BENCH_N=100000 cargo bench -p bench-substrate --bench retrieval
//! ```
//!
//! Smaller corpora are fine for per-stage tuning (L0 scales linearly in N;
//! L1 / L2 are fixed-size at K0 / K1).
//!
//! ## Why a separate group per stage?
//!
//! The L0 scan is ~95% of the wall clock on a full cascade — L1 and L2 are
//! sub-millisecond even at 10⁶ corpus because they only ever see K0 = 1000
//! and K1 = 100 candidates. If you benchmark only the full pipeline, any
//! regression shows up as "pipeline got slower" without telling you which
//! tier caused it. The per-stage groups make the attribution mechanical.

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use obrain_substrate::tiered_scan::{
    DEFAULT_K0, DEFAULT_K1, ScanConfig, TieredQuery, scan_tiered,
};
use obrain_substrate::tiers::{
    L2_DIM, Tier0, Tier0Builder, Tier1, Tier1Builder, Tier2, Tier2Builder, tier0_topk,
    tier1_topk, tier2_cosine,
};

// 384-dim is the canonical embedding width for substrate corpora (matches
// sentence-transformers / MiniLM family used by obrain-rag).
const DIM: usize = L2_DIM;

/// Deterministic xorshift64 — no `rand` dependency at bench time.
///
/// Duplicated across bench files on purpose: each bench file is compiled
/// as an independent criterion harness, and we want the seed logic to be
/// self-contained so a bench run in isolation is reproducible from the
/// source.
struct Xs64(u64);
impl Xs64 {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn next_unit(&mut self) -> f32 {
        let s = self.next() as i64 as f64;
        (s / (i64::MAX as f64)) as f32
    }
}

/// Build `n` synthetic 384-dim embeddings and project them through the three
/// tier builders with canonical default seeds. Returns the triple of tier
/// vectors + the f32 backing (kept for sanity-check ranking if needed).
fn build_corpus(
    n: usize,
    seed: u64,
) -> (Vec<Tier0>, Vec<Tier1>, Vec<Tier2>, Vec<f32>) {
    let mut prng = Xs64::new(seed);
    let raw: Vec<f32> = (0..n * DIM).map(|_| prng.next_unit()).collect();
    let b0 = Tier0Builder::with_default_seed(DIM);
    let b1 = Tier1Builder::with_default_seed(DIM);
    let b2 = Tier2Builder::new();
    let l0 = b0.project_batch(&raw);
    let l1 = b1.project_batch(&raw);
    let l2 = b2.project_batch(&raw);
    (l0, l1, l2, raw)
}

/// Pick a query index offset-centered so self-recall doesn't sit on the
/// boundary of a wave shard (matters for the parallel path).
fn pick_query_idx(n: usize) -> usize {
    (n / 3) + 7
}

fn corpus_size() -> usize {
    std::env::var("TIERED_BENCH_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000)
}

// ---------------------------------------------------------------------------
// Stage 1 — L0 scan
// ---------------------------------------------------------------------------

fn bench_l0_scan(c: &mut Criterion) {
    let n = corpus_size();
    let (l0, _l1, _l2, _) = build_corpus(n, 0x5EED_C0FEE);
    let q_idx = pick_query_idx(n);
    let q0 = l0[q_idx];

    let mut g = c.benchmark_group("L0_scan");
    g.throughput(Throughput::Elements(n as u64));
    // L0 is the long tail — the sample count can stay modest because a
    // single iteration is 200 µs – 1 ms on 10⁶ nodes.
    g.sample_size(40);
    g.measurement_time(Duration::from_secs(8));

    for &k in &[10usize, DEFAULT_K0] {
        g.bench_with_input(
            BenchmarkId::new("tier0_topk", format!("N={n}/K={k}")),
            &k,
            |b, &k| {
                b.iter(|| {
                    let hits = tier0_topk(black_box(&l0), black_box(q0), k);
                    black_box(hits)
                });
            },
        );
    }
    g.finish();
}

// ---------------------------------------------------------------------------
// Stage 2 — L1 re-rank
// ---------------------------------------------------------------------------

/// Re-rank inputs are the top-K0 from L0 in the real cascade. We simulate
/// that by taking the first K0 nodes of the corpus — the exact ordering
/// doesn't matter for latency, only the count and the access pattern do.
fn bench_l1_rerank(c: &mut Criterion) {
    let n = corpus_size();
    let (_l0, l1, _l2, _) = build_corpus(n, 0x5EED_C0FEE);
    let q_idx = pick_query_idx(n);
    let q1 = l1[q_idx];

    let mut g = c.benchmark_group("L1_rerank");
    // K0 candidates — the fixed re-rank window.
    g.throughput(Throughput::Elements(DEFAULT_K0 as u64));
    g.sample_size(100);
    g.measurement_time(Duration::from_secs(4));

    // Simulate the L0 top-K0 output by taking a contiguous slice.
    let candidates: Vec<Tier1> = l1.iter().take(DEFAULT_K0).copied().collect();

    for &k in &[10usize, DEFAULT_K1] {
        g.bench_with_input(
            BenchmarkId::new("tier1_topk", format!("K0={}/K={k}", DEFAULT_K0)),
            &k,
            |b, &k| {
                b.iter(|| {
                    let hits = tier1_topk(black_box(&candidates), black_box(&q1), k);
                    black_box(hits)
                });
            },
        );
    }
    g.finish();
}

// ---------------------------------------------------------------------------
// Stage 3 — L2 cosine (f16 exact)
// ---------------------------------------------------------------------------

fn bench_l2_cosine(c: &mut Criterion) {
    let n = corpus_size();
    let (_l0, _l1, l2, _) = build_corpus(n, 0x5EED_C0FEE);
    let q_idx = pick_query_idx(n);
    let q2 = l2[q_idx];

    let mut g = c.benchmark_group("L2_cosine");
    // K1 candidates — the final re-rank window.
    g.throughput(Throughput::Elements(DEFAULT_K1 as u64));
    g.sample_size(100);
    g.measurement_time(Duration::from_secs(4));

    // Simulate the L1 top-K1 output.
    let candidates: Vec<Tier2> = l2.iter().take(DEFAULT_K1).cloned().collect();

    // Pairwise cosine over the candidate pool — this is what the cascade
    // does at stage 3 after re-rank narrows the field to K1.
    g.bench_function("tier2_cosine_batch", |b| {
        b.iter(|| {
            let mut out = Vec::with_capacity(candidates.len());
            for c in &candidates {
                out.push(tier2_cosine(black_box(&q2), black_box(c)));
            }
            black_box(out)
        });
    });
    g.finish();
}

// ---------------------------------------------------------------------------
// Stage 4 — Full cascade, K ∈ {1, 10, 100}
// ---------------------------------------------------------------------------

fn bench_pipeline_topk(c: &mut Criterion) {
    let n = corpus_size();
    let (l0, l1, l2, _) = build_corpus(n, 0x5EED_C0FEE);
    let q_idx = pick_query_idx(n);
    let q = TieredQuery {
        l0: l0[q_idx],
        l1: l1[q_idx],
        l2: l2[q_idx],
    };

    let mut g = c.benchmark_group("pipeline_topk");
    g.throughput(Throughput::Elements(n as u64));
    g.sample_size(30);
    g.measurement_time(Duration::from_secs(10));

    // Default config — rayon kicks in at PARALLEL_THRESHOLD (64 K).
    let cfg = ScanConfig::default();

    for &k in &[1usize, 10, 100] {
        g.bench_with_input(
            BenchmarkId::new("full_cascade", format!("N={n}/K={k}")),
            &k,
            |b, &k| {
                b.iter(|| {
                    let hits = scan_tiered(
                        black_box(&q),
                        black_box(&l0),
                        black_box(&l1),
                        black_box(&l2),
                        k,
                        cfg,
                    );
                    black_box(hits)
                });
            },
        );
    }
    g.finish();
}

// ---------------------------------------------------------------------------
// Stage 5 — Thread scaling on the target K (10)
// ---------------------------------------------------------------------------

fn bench_pipeline_threads(c: &mut Criterion) {
    let n = corpus_size();
    let (l0, l1, l2, _) = build_corpus(n, 0x5EED_C0FEE);
    let q_idx = pick_query_idx(n);
    let q = TieredQuery {
        l0: l0[q_idx],
        l1: l1[q_idx],
        l2: l2[q_idx],
    };

    let mut g = c.benchmark_group("pipeline_threads");
    g.throughput(Throughput::Elements(n as u64));
    g.sample_size(30);
    g.measurement_time(Duration::from_secs(10));

    // Serial: force the L0 scan to stay single-threaded by pushing the
    // threshold above the corpus size. Useful as a hardware-relative gate
    // on machines without AVX-512 (e.g. Apple M2 — ~800 µs single-thread).
    let serial = ScanConfig {
        parallel_threshold: usize::MAX,
        ..ScanConfig::default()
    };
    g.bench_function(BenchmarkId::new("serial", format!("N={n}/K=10")), |b| {
        b.iter(|| {
            let hits = scan_tiered(
                black_box(&q),
                black_box(&l0),
                black_box(&l1),
                black_box(&l2),
                10,
                serial,
            );
            black_box(hits)
        });
    });

    let parallel = ScanConfig::default();
    g.bench_function(BenchmarkId::new("rayon", format!("N={n}/K=10")), |b| {
        b.iter(|| {
            let hits = scan_tiered(
                black_box(&q),
                black_box(&l0),
                black_box(&l1),
                black_box(&l2),
                10,
                parallel,
            );
            black_box(hits)
        });
    });
    g.finish();
}

criterion_group!(
    benches,
    bench_l0_scan,
    bench_l1_rerank,
    bench_l2_cosine,
    bench_pipeline_topk,
    bench_pipeline_threads,
);
criterion_main!(benches);
