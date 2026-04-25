//! T8 Step 4 acceptance bench — hierarchical L0 → L1 → L2 cascade on 10⁶ nodes.
//!
//! Gate: single-thread top-10 on 10⁶ nodes ≤ 300 µs,
//!       multi-thread 8-cores top-10 on 10⁶ nodes ≤ 60 µs.
//!
//! The cascade widths default to 1 000 / 100 / k — chosen so the L0
//! scan (the only O(n) stage) dominates the wall clock. L1 re-rank
//! (1 000 × 64 B = 64 KB) and L2 cosine (100 × 768 B = 76 KB) fit in
//! L2 cache and cost ≤ 5 µs together on any modern core.
//!
//! Run:
//!
//! ```bash
//! cargo bench -p obrain-substrate --bench tiered_scan
//! ```
//!
//! To measure a smaller corpus (fast iteration), pass an env var:
//!
//! ```bash
//! TIERED_BENCH_N=100000 cargo bench -p obrain-substrate --bench tiered_scan
//! ```

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use obrain_substrate::tiered_scan::{ScanConfig, TieredQuery, scan_tiered};
use obrain_substrate::tiers::{Tier0Builder, Tier1Builder, Tier2Builder};

const DIM: usize = 384;

/// Deterministic xorshift64 — no `rand` dependency at bench time.
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

/// Build `n` synthetic 384-dim embeddings projected into the three tiers
/// using the canonical seeds.
fn build_corpus(
    n: usize,
    seed: u64,
) -> (
    Vec<obrain_substrate::tiers::Tier0>,
    Vec<obrain_substrate::tiers::Tier1>,
    Vec<obrain_substrate::tiers::Tier2>,
    Vec<f32>,
) {
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

fn bench_cascade(c: &mut Criterion) {
    // Default to 10⁶ nodes — the T8 Step 4 acceptance corpus. Override
    // via env for quicker iterations while tuning.
    let n: usize = std::env::var("TIERED_BENCH_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000);

    let mut g = c.benchmark_group("tiered_scan_top10");
    g.throughput(Throughput::Elements(n as u64));
    // Large corpora need a longer warm-up to let the OS warm the pages
    // and for criterion to collect stable samples.
    g.sample_size(20);
    g.measurement_time(std::time::Duration::from_secs(8));

    let (l0, l1, l2, _) = build_corpus(n, 0x5EED_C0FEE);
    // Pick a query from the corpus so the self-recall path hits the
    // happy case (top-1 is node q_idx).
    let q_idx = (n / 3) + 7; // off-centre to avoid boundary artefacts
    let q = TieredQuery {
        l0: l0[q_idx],
        l1: l1[q_idx],
        l2: l2[q_idx],
    };

    // Single-threaded: force the serial L0 path by setting
    // `parallel_threshold` above the corpus size.
    let cfg_serial = ScanConfig {
        parallel_threshold: usize::MAX,
        ..ScanConfig::default()
    };
    g.bench_with_input(BenchmarkId::new("single_thread", n), &n, |b, _| {
        b.iter(|| {
            let hits = scan_tiered(
                black_box(&q),
                black_box(&l0),
                black_box(&l1),
                black_box(&l2),
                10,
                cfg_serial,
            );
            black_box(hits)
        });
    });

    // Multi-threaded: rayon's global pool (≈ num_cpus). The parallel
    // threshold stays at the default 64 K so rayon kicks in as soon
    // as the corpus is worth sharding.
    let cfg_parallel = ScanConfig::default();
    g.bench_with_input(BenchmarkId::new("rayon_pool", n), &n, |b, _| {
        b.iter(|| {
            let hits = scan_tiered(
                black_box(&q),
                black_box(&l0),
                black_box(&l1),
                black_box(&l2),
                10,
                cfg_parallel,
            );
            black_box(hits)
        });
    });

    g.finish();
}

criterion_group!(benches, bench_cascade);
criterion_main!(benches);
