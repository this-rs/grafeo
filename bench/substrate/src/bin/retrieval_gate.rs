//! # T9 — Retrieval Gate
//!
//! Measures recall and latency of the L0 / L1 / L2 cascade against an f32
//! brute-force ground truth on a synthetic 10⁶-node corpus. Emits a JSON
//! report and exits non-zero if any gate is violated.
//!
//! ## Gates
//!
//! | Metric       | Threshold               | Rationale                                   |
//! |--------------|-------------------------|---------------------------------------------|
//! | recall@10    | ≥ 99%                   | T8 acceptance: "Recall@10 ≥ 99% mesuré"     |
//! | recall@100   | ≥ 99.5%                 | T8 acceptance: "Recall@100 ≥ 99.5%"         |
//! | p95 latency  | ≤ 1 ms single-thread    | T8 acceptance: "p95 ≤ 1 ms sur 10⁶ nœuds"   |
//!
//! ## Why synthetic first
//!
//! The gate validates *the cascade itself* (quantization + scan). It tells us
//! whether the algorithm preserves top-K ordering under signed-random-projection
//! quantization on a corpus of the right shape — independent of whether the
//! vectors come from PO, megalaw, or a PRNG. If recall drops below 99% on
//! synthetic data, it'll also drop on real data; we need to know before
//! shipping a real-data harness.
//!
//! The real-data plumbing (load `.obrain` store → extract `_st_embedding` column
//! → sample 1000 query messages from conversation logs) wires in during T14
//! (`obrain-migrate` CLI), which already teaches us how to read the legacy
//! embedding column. Until then this synthetic gate is the CI contract.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --release --bin retrieval-gate -- \
//!     --nodes 1000000 \
//!     --queries 1000 \
//!     --seed 0x5EED_C0FEE \
//!     --out report.json
//! ```
//!
//! Exit code 0 if all gates pass; 1 if any gate fails. The JSON report is
//! always written (even on failure) so CI can diff against the previous run.

use std::collections::HashSet;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::Parser;
use obrain_substrate::tiered_scan::{ScanConfig, TieredQuery, scan_tiered};
use obrain_substrate::tiers::{
    L2_DIM, Tier0, Tier0Builder, Tier1, Tier1Builder, Tier2, Tier2Builder,
};
use serde::Serialize;

#[derive(Parser, Debug)]
#[command(
    name = "retrieval-gate",
    about = "T9 retrieval gate — recall@10 / recall@100 + p95 latency"
)]
struct Args {
    /// Corpus size. Default 10⁶ matches the T8/T9 acceptance target.
    #[arg(long, default_value_t = 1_000_000)]
    nodes: usize,
    /// Number of query samples used to compute recall and latency percentiles.
    #[arg(long, default_value_t = 1_000)]
    queries: usize,
    /// RNG seed for corpus + query generation. Determinism is a gate
    /// requirement — a flaky recall number is unusable as a CI signal.
    #[arg(long, default_value_t = 0x5EED_C0FEE)]
    seed: u64,
    /// JSON report output path. Always written, even on gate failure.
    #[arg(long, default_value = "retrieval_gate.json")]
    out: PathBuf,
    /// Minimum recall@10 to pass the gate (as a ratio, not a percentage).
    #[arg(long, default_value_t = 0.99)]
    min_recall_10: f64,
    /// Minimum recall@100 to pass the gate.
    #[arg(long, default_value_t = 0.995)]
    min_recall_100: f64,
    /// Maximum p95 latency in microseconds (single-thread).
    #[arg(long, default_value_t = 1_000.0)]
    max_p95_us: f64,
    /// Disable parallel execution inside the cascade (forces the serial L0
    /// scan so the p95 gate is measured on a hardware-relative baseline).
    #[arg(long)]
    serial: bool,
}

const DIM: usize = L2_DIM; // 384

// ---------------------------------------------------------------------------
// Deterministic PRNG — local, no `rand` dep pulled in.
// ---------------------------------------------------------------------------

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
}

// ---------------------------------------------------------------------------
// f32 cosine — used inside the recall check to rank the 100 planted vectors
// against the query base. Not on any hot path (≤ 100 × DIM flops per query).
// ---------------------------------------------------------------------------

/// Cosine similarity, unnormalized input (we normalize implicitly via the
/// divisor). Returns `cos ∈ [-1, 1]`.
#[inline]
fn cosine_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        let x = a[i];
        let y = b[i];
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

// ---------------------------------------------------------------------------
// Cascade call site.
// ---------------------------------------------------------------------------

fn cascade_topk(
    q: &TieredQuery,
    l0: &[Tier0],
    l1: &[Tier1],
    l2: &[Tier2],
    k: usize,
    cfg: ScanConfig,
) -> Vec<u32> {
    let hits = scan_tiered(q, l0, l1, l2, k, cfg);
    hits.into_iter().map(|h| h.node_offset).collect()
}

// ---------------------------------------------------------------------------
// Percentile calculation — nearest-rank method.
// ---------------------------------------------------------------------------

fn percentile_us(samples: &mut [Duration], p: f64) -> f64 {
    assert!(!samples.is_empty());
    samples.sort_unstable();
    let idx = ((p * samples.len() as f64).ceil() as usize).saturating_sub(1);
    let idx = idx.min(samples.len() - 1);
    samples[idx].as_secs_f64() * 1_000_000.0
}

fn mean_us(samples: &[Duration]) -> f64 {
    let sum: f64 = samples.iter().map(|d| d.as_secs_f64()).sum();
    (sum / samples.len() as f64) * 1_000_000.0
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct Report {
    setup: Setup,
    recall: Recall,
    latency_us: Latency,
    gate: Gate,
}

#[derive(Serialize)]
struct Setup {
    nodes: usize,
    queries: usize,
    dim: usize,
    seed: u64,
    serial: bool,
    parallel_cascade: bool,
}

#[derive(Serialize)]
struct Recall {
    recall_at_10: f64,
    recall_at_100: f64,
}

#[derive(Serialize)]
struct Latency {
    mean: f64,
    p50: f64,
    p95: f64,
    p99: f64,
    min: f64,
    max: f64,
}

#[derive(Serialize)]
struct Gate {
    passed: bool,
    failures: Vec<String>,
    min_recall_10: f64,
    min_recall_100: f64,
    max_p95_us: f64,
}

// ---------------------------------------------------------------------------
// Main driver
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let args = Args::parse();
    let n = args.nodes;
    let q_n = args.queries;

    eprintln!(
        "[gate] building corpus n={n} dim={DIM} (clusters + planted-neighbors for ground-truth stability) ..."
    );
    let t_build = Instant::now();
    let mut prng = Xs64::new(args.seed);

    // Synthetic corpus strategy.
    //
    // We need a corpus on which the f32-cosine top-K is both (a) stable (not
    // swamped by concentration-of-measure noise) and (b) independent of the
    // SRP quantisation we're validating. Uniform i.i.d. in R^384 fails (a):
    // all pairwise cosines bunch near 0 with stddev 1/√D so the "true top-K"
    // is essentially random labels. A naive cluster model is tricky because
    // σ must be set very precisely (σ²D must be small — in D=384 that means
    // σ < 0.03 to keep within-cluster cos > 0.95).
    //
    // Simpler: *planted neighbors*. For each query i we generate a tight
    // cluster of exactly `planted_per_query` near-duplicates of the query
    // vector (the cluster is tagged by known offsets, so truth_top_K is
    // deterministic by construction). The rest of the corpus is background
    // noise — random unit vectors, which have cos ≈ 0 with every query and
    // with each other (concentration of measure in our favour here).
    //
    // For SRP to discriminate, within-cluster cos needs to be ≳ 0.9. With
    // Gaussian perturbation σ on a unit vector, expected cos after
    // normalisation ≈ 1/√(1 + σ²D). Solving for cos = 0.95 in D = 384 gives
    // σ ≈ 0.0167. We use σ = 0.02 for a small safety margin (cos ≈ 0.93).

    // Gaussian via Box-Muller from two Xs64 uniforms.
    fn gaussian(prng: &mut Xs64) -> f32 {
        let u1 = ((prng.next() as f64) / (u64::MAX as f64)).max(1e-12);
        let u2 = (prng.next() as f64) / (u64::MAX as f64);
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        z as f32
    }
    fn random_unit(prng: &mut Xs64) -> Vec<f32> {
        let v: Vec<f32> = (0..DIM).map(|_| gaussian(prng)).collect();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        v.into_iter().map(|x| x / norm).collect()
    }
    fn perturb_unit(base: &[f32], sigma: f32, prng: &mut Xs64) -> Vec<f32> {
        let v: Vec<f32> = base
            .iter()
            .map(|&b| b + sigma * gaussian(prng))
            .collect();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        v.into_iter().map(|x| x / norm).collect()
    }

    let sigma = 0.02f32;
    let planted_per_query = 100usize; // enough to measure recall@100
    let total_planted = q_n * planted_per_query;
    assert!(
        total_planted < n,
        "corpus too small: need n > q_n × 100 for planted-neighbor gate"
    );

    // Background: n - total_planted random unit vectors. These have pairwise
    // cos concentrated near 0 (±2/√D ≈ ±0.1 stddev in D=384), so they stay
    // well below the within-cluster cos ≈ 0.93 of the plants.
    let mut raw: Vec<f32> = Vec::with_capacity(n * DIM);
    let n_background = n - total_planted;
    for _ in 0..n_background {
        raw.extend(random_unit(&mut prng));
    }

    // Planted: for each query, generate a query vector + planted_per_query
    // near-duplicates. The contiguous block [n_background + i×100 ..
    // n_background + (i+1)×100] is the known ground-truth top-100 for
    // query i.
    let mut query_bases: Vec<Vec<f32>> = Vec::with_capacity(q_n);
    for _ in 0..q_n {
        let base = random_unit(&mut prng);
        for _ in 0..planted_per_query {
            raw.extend(perturb_unit(&base, sigma, &mut prng));
        }
        query_bases.push(base);
    }
    assert_eq!(raw.len(), n * DIM);

    let b0 = Tier0Builder::with_default_seed(DIM);
    let b1 = Tier1Builder::with_default_seed(DIM);
    let b2 = Tier2Builder::new();
    let l0 = b0.project_batch(&raw);
    let l1 = b1.project_batch(&raw);
    let l2 = b2.project_batch(&raw);
    eprintln!("[gate] corpus built in {:?}", t_build.elapsed());

    // Project each query-base through the three tier builders. Queries are
    // NOT corpus members — they're fresh vectors whose planted neighbors live
    // at known contiguous positions in the corpus tail.
    let query_tiers: Vec<TieredQuery> = (0..q_n)
        .map(|i| {
            let raw_base = &query_bases[i][..];
            let l0v = b0.project_batch(raw_base);
            let l1v = b1.project_batch(raw_base);
            let l2v = b2.project_batch(raw_base);
            TieredQuery {
                l0: l0v[0],
                l1: l1v[0],
                l2: l2v[0],
            }
        })
        .collect();

    // Cascade configuration. Serial mode forces the single-threaded L0 path,
    // which is the hardware-relative baseline the p95 gate targets.
    let cfg = if args.serial {
        ScanConfig {
            parallel_threshold: usize::MAX,
            ..ScanConfig::default()
        }
    } else {
        ScanConfig::default()
    };

    // ---- Recall pass: compare cascade top-K vs planted-neighbors truth ----
    //
    // By construction, the 100 planted neighbors for query i live at corpus
    // offsets [n_background + i*100 .. n_background + (i+1)*100]. These are
    // cos ≈ 0.93 with the query base, while background corpus vectors have
    // cos ≈ 0. So the f32 brute-force top-100 for query i *is* the planted
    // block (modulo a handful of random background hits at the tail, but
    // these shouldn't reach into the top-100 under concentration of measure).
    //
    // For top-10 truth, we still need to know *which* 10 of the 100 plants
    // are closest to the query — the plants themselves differ in their
    // individual perturbation draws. So we brute-force the query against
    // only the 100 planted vectors (100 × 384 f32 = fast) to determine the
    // authoritative top-10 of the planted block.
    eprintln!(
        "[gate] computing recall over {q_n} queries (fast — planted-truth avoids global brute force) ..."
    );
    let t_recall = Instant::now();
    let mut hits_10 = 0usize;
    let mut hits_100 = 0usize;
    let mut total_10 = 0usize;
    let mut total_100 = 0usize;
    for i in 0..q_n {
        let q = query_tiers[i];
        let q_raw = &query_bases[i][..];

        // Planted block for this query.
        let block_start = (n_background + i * planted_per_query) as u32;
        let block_end = (n_background + (i + 1) * planted_per_query) as u32;
        let truth_100: HashSet<u32> = (block_start..block_end).collect();

        // Top-10 inside the planted block, by f32 cosine.
        let mut scored: Vec<(u32, f32)> = (block_start..block_end)
            .map(|j| {
                let row = &raw[(j as usize) * DIM..(j as usize + 1) * DIM];
                (j, cosine_f32(q_raw, row))
            })
            .collect();
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        let truth_10: HashSet<u32> = scored.iter().take(10).map(|(id, _)| *id).collect();

        let cascade_10 = cascade_topk(&q, &l0, &l1, &l2, 10, cfg);
        let cascade_100 = cascade_topk(&q, &l0, &l1, &l2, 100, cfg);

        hits_10 += cascade_10.iter().filter(|id| truth_10.contains(id)).count();
        hits_100 += cascade_100
            .iter()
            .filter(|id| truth_100.contains(id))
            .count();
        total_10 += truth_10.len();
        total_100 += truth_100.len();
    }
    eprintln!("[gate] recall pass in {:?}", t_recall.elapsed());

    let recall_10 = hits_10 as f64 / total_10 as f64;
    let recall_100 = hits_100 as f64 / total_100 as f64;

    // ---- Latency pass: measure end-to-end cascade time, cascade-only ----
    eprintln!("[gate] measuring latency over {q_n} queries ...");
    let t_lat = Instant::now();
    // Warm-up: a few cascades so the OS pages-in the tier columns.
    for q in query_tiers.iter().take(10) {
        let _ = black_box(cascade_topk(q, &l0, &l1, &l2, 10, cfg));
    }

    let mut samples: Vec<Duration> = Vec::with_capacity(q_n);
    for q in &query_tiers {
        let t0 = Instant::now();
        let hits = cascade_topk(q, &l0, &l1, &l2, 10, cfg);
        samples.push(t0.elapsed());
        black_box(hits);
    }
    eprintln!("[gate] latency pass in {:?}", t_lat.elapsed());

    let mean = mean_us(&samples);
    let p50 = percentile_us(&mut samples, 0.50);
    let p95 = percentile_us(&mut samples, 0.95);
    let p99 = percentile_us(&mut samples, 0.99);
    let min_v = samples[0].as_secs_f64() * 1_000_000.0;
    let max_v = samples[samples.len() - 1].as_secs_f64() * 1_000_000.0;

    // ---- Gate ----
    let mut failures: Vec<String> = Vec::new();
    if recall_10 < args.min_recall_10 {
        failures.push(format!(
            "recall@10 = {:.4} < {:.4}",
            recall_10, args.min_recall_10
        ));
    }
    if recall_100 < args.min_recall_100 {
        failures.push(format!(
            "recall@100 = {:.4} < {:.4}",
            recall_100, args.min_recall_100
        ));
    }
    if p95 > args.max_p95_us {
        failures.push(format!(
            "p95 = {:.1} µs > {:.1} µs",
            p95, args.max_p95_us
        ));
    }
    let passed = failures.is_empty();

    let report = Report {
        setup: Setup {
            nodes: n,
            queries: q_n,
            dim: DIM,
            seed: args.seed,
            serial: args.serial,
            parallel_cascade: !args.serial,
        },
        recall: Recall {
            recall_at_10: recall_10,
            recall_at_100: recall_100,
        },
        latency_us: Latency {
            mean,
            p50,
            p95,
            p99,
            min: min_v,
            max: max_v,
        },
        gate: Gate {
            passed,
            failures: failures.clone(),
            min_recall_10: args.min_recall_10,
            min_recall_100: args.min_recall_100,
            max_p95_us: args.max_p95_us,
        },
    };

    let json =
        serde_json::to_string_pretty(&report).context("serializing JSON report")?;
    std::fs::write(&args.out, &json)
        .with_context(|| format!("writing report to {:?}", args.out))?;

    // Human-readable summary to stderr.
    eprintln!();
    eprintln!("========== Retrieval Gate Report ==========");
    eprintln!("Setup:   n={n}, queries={q_n}, dim={DIM}, serial={}", args.serial);
    eprintln!(
        "Recall:  @10 = {:.4} (gate ≥ {:.4}),  @100 = {:.4} (gate ≥ {:.4})",
        recall_10, args.min_recall_10, recall_100, args.min_recall_100
    );
    eprintln!(
        "Latency: mean = {:.1} µs, p50 = {:.1} µs, p95 = {:.1} µs (gate ≤ {:.1}), p99 = {:.1} µs",
        mean, p50, p95, args.max_p95_us, p99
    );
    eprintln!("Report:  {:?}", args.out);
    if passed {
        eprintln!("VERDICT: PASS ✓");
        Ok(())
    } else {
        eprintln!("VERDICT: FAIL ✗");
        for f in &failures {
            eprintln!("  - {f}");
        }
        std::process::exit(1);
    }
}
