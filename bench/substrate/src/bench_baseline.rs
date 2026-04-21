//! Baseline benchmark runner — measures the current LpgStore engine on a target database.
//!
//! Usage:
//!   cargo run --release --bin bench-baseline -- --target po --db /path/to/po.obrain --out baseline-po.json
//!   cargo run --release --bin bench-baseline -- --target megalaw --db /path/to/megalaw.obrain --out baseline-megalaw.json

use anyhow::Result;
use bench_substrate::{Backend, BenchResult, HostInfo, Measurement, RssSampler, Target, time_it};
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(about = "Baseline (LpgStore) benchmark runner")]
struct Args {
    /// Target database tag (po / megalaw / custom).
    #[arg(long, value_enum)]
    target: TargetArg,

    /// Path to the database directory or .obrain file.
    #[arg(long)]
    db: PathBuf,

    /// Output JSON path.
    #[arg(long)]
    out: PathBuf,

    /// Skip slow scenarios (for smoke test).
    #[arg(long, default_value_t = false)]
    smoke: bool,
}

#[derive(clap::ValueEnum, Debug, Clone, Copy)]
enum TargetArg {
    Po,
    Megalaw,
    Custom,
}

impl From<TargetArg> for Target {
    fn from(t: TargetArg) -> Self {
        match t {
            TargetArg::Po => Target::Po,
            TargetArg::Megalaw => Target::Megalaw,
            TargetArg::Custom => Target::Custom,
        }
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args = Args::parse();
    let target: Target = args.target.into();
    let mut sampler = RssSampler::new();
    sampler.sample();

    tracing::info!("baseline benchmark: target={:?} db={}", target, args.db.display());

    // ---------- Scenario 1: open + warmup ----------
    let (_open_result, open_duration) = time_it(|| {
        // TODO(T1-step3): wire ObrainDB::open with LpgStore on the provided path.
        // Placeholder — returns () so the harness is buildable standalone.
        tracing::warn!("open(): stub — wire ObrainDB::open in T1-step3");
        std::thread::sleep(std::time::Duration::from_millis(10));
    });
    let (anon_after_open, total_after_open) = sampler.sample();

    // ---------- Scenario 2: cold retrieval ----------
    let retrieval_latencies_us: Vec<f64> = Vec::new();
    // TODO(T1-step3): run 100 cold retrievals, record p50/p95/p99 latencies.

    // ---------- Scenario 3: activation spreading ----------
    let activation_latencies_us: Vec<f64> = Vec::new();
    // TODO(T1-step3): run 100 activation spreads from random seeds.

    // ---------- Scenario 4: CRUD mutations ----------
    let crud_latencies_us: Vec<f64> = Vec::new();
    // TODO(T1-step3): run 1000 insert/update/delete cycles.

    let (anon_peak, total_peak) = sampler.peaks();

    let mut measurements = vec![
        Measurement::new("open_duration", "ms", open_duration.as_secs_f64() * 1000.0),
        Measurement::new("rss_anon_after_open", "bytes", anon_after_open as f64),
        Measurement::new("rss_total_after_open", "bytes", total_after_open as f64),
        Measurement::new("rss_anon_peak", "bytes", anon_peak as f64),
        Measurement::new("rss_total_peak", "bytes", total_peak as f64),
    ];

    if !args.smoke {
        measurements.extend(stat_measurements(
            "retrieval_latency",
            "us",
            &retrieval_latencies_us,
        ));
        measurements.extend(stat_measurements(
            "activation_latency",
            "us",
            &activation_latencies_us,
        ));
        measurements.extend(stat_measurements("crud_latency", "us", &crud_latencies_us));
    }

    let result = BenchResult {
        schema_version: 1,
        backend: Backend::Lpg,
        target,
        db_path: args.db.display().to_string(),
        timestamp: chrono_now_unix_seconds(),
        host: HostInfo::collect(),
        measurements,
    };

    let json = serde_json::to_string_pretty(&result)?;
    std::fs::write(&args.out, json)?;
    tracing::info!("wrote {}", args.out.display());
    Ok(())
}

/// Compute simple stats over a slice of latencies.
fn stat_measurements(name_prefix: &str, unit: &str, xs: &[f64]) -> Vec<Measurement> {
    if xs.is_empty() {
        return vec![];
    }
    let mut sorted = xs.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = xs.iter().sum::<f64>() / xs.len() as f64;
    let p = |q: f64| {
        let idx = ((q * (sorted.len() as f64)) as usize).min(sorted.len() - 1);
        sorted[idx]
    };
    vec![
        Measurement::new(name_prefix, unit, mean).with_stat("mean"),
        Measurement::new(name_prefix, unit, p(0.50)).with_stat("p50"),
        Measurement::new(name_prefix, unit, p(0.95)).with_stat("p95"),
        Measurement::new(name_prefix, unit, p(0.99)).with_stat("p99"),
        Measurement::new(name_prefix, unit, *sorted.last().unwrap()).with_stat("max"),
    ]
}

fn chrono_now_unix_seconds() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}
