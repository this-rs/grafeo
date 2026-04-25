//! Substrate benchmark runner — measures the new SubstrateStore engine.
//!
//! Wired as a stub until T2 (records) + T3 (WAL) + T8 (tier retrieval) + T10 (LDleiden) land.
//!
//! Usage:
//!   cargo run --release --bin bench-substrate -- --target po --db /tmp/po-substrate --out substrate-po.json

use anyhow::Result;
use bench_substrate::{Backend, BenchResult, HostInfo, Measurement, RssSampler, Target, time_it};
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(about = "Substrate (mmap + WAL + tiers) benchmark runner — stub until T3")]
struct Args {
    #[arg(long, value_enum)]
    target: TargetArg,

    #[arg(long)]
    db: PathBuf,

    #[arg(long)]
    out: PathBuf,

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

    tracing::info!(
        "substrate benchmark: target={:?} db={}",
        target,
        args.db.display()
    );

    let (_, open_duration) = time_it(|| {
        // TODO(T3): SubstrateStore::open(args.db)
        tracing::warn!("SubstrateStore::open(): stub — wired in T3");
    });
    let (anon_after_open, total_after_open) = sampler.sample();

    let measurements = vec![
        Measurement::new("open_duration", "ms", open_duration.as_secs_f64() * 1000.0),
        Measurement::new("rss_anon_after_open", "bytes", anon_after_open as f64),
        Measurement::new("rss_total_after_open", "bytes", total_after_open as f64),
    ];

    let result = BenchResult {
        schema_version: 1,
        backend: Backend::Substrate,
        target,
        db_path: args.db.display().to_string(),
        timestamp: now_unix_seconds(),
        host: HostInfo::collect(),
        measurements,
    };

    std::fs::write(&args.out, serde_json::to_string_pretty(&result)?)?;
    tracing::info!("wrote {}", args.out.display());
    let _ = args.smoke;
    Ok(())
}

fn now_unix_seconds() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}
