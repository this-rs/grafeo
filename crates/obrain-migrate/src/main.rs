//! # `obrain-migrate` — one-shot converter from legacy `.obrain` to substrate
//!
//! Reads a legacy `.obrain` input in either of its on-disk shapes and
//! writes the new substrate format via [`obrain_substrate::SubstrateStore`]:
//!
//! * **Directory layout** (LpgStore epoch files, `--features legacy-read`).
//! * **Single-file layout** (`GRAF`-magic `.obrain` file, v1 bincode or v2
//!   native mmap, `--features single-file-read`). This is the path used
//!   for e.g. `wikipedia.obrain`.
//!
//! Both features are on by default. The reader dispatches on filesystem
//! type — see [`legacy_reader`] for the full contract.
//!
//! Usage:
//!
//! ```text
//! obrain-migrate \
//!     --in  path/to/old.obrain        \
//!     --out path/to/new.obrain/       \
//!     [--workers 4]                   \
//!     [--resume]                      \
//!     [--with-cognitive-init]         \
//!     [--with-tiers]
//! ```
//!
//! Design invariants:
//! - **Idempotent**: migrating the same input twice yields byte-equal output.
//! - **Resume-safe**: a checkpoint file is written every 10% of progress,
//!   and `--resume` picks up at the last completed phase.
//! - **Parity-preserving**: node / edge / label / property counts match
//!   the source exactly; cognitive state is mapped column-for-column.
//! - **Offline**: auto-start thinkers are suppressed on both source
//!   (read-only) and destination (silent substrate) — the migration is
//!   the only writer.

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

mod checkpoint;
mod converter;
#[cfg(feature = "legacy-read")]
mod legacy_reader;

/// One-shot migrator from legacy `.obrain` (LpgStore) to substrate.
#[derive(Debug, Parser)]
#[command(name = "obrain-migrate", version, about)]
struct Cli {
    /// Input `.obrain` database (either a directory of epoch files or a
    /// single `GRAF`-magic file). Auto-detected at open time.
    #[arg(long = "in", value_name = "PATH")]
    input: PathBuf,

    /// Output `.obrain` directory (substrate format). Created if absent.
    #[arg(long = "out", value_name = "PATH")]
    output: PathBuf,

    /// Number of worker threads for edge / property replay phases.
    /// (0 = rayon default = physical core count).
    #[arg(long, default_value_t = 0)]
    workers: usize,

    /// Resume from the latest checkpoint if one exists under `<out>/.migrate-checkpoint`.
    #[arg(long)]
    resume: bool,

    /// After structural migration, run LDleiden + PageRank + Ricci so the
    /// destination is cognitively initialised from the first open. Disabled
    /// by default — wire this on for production bases.
    #[arg(long = "with-cognitive-init")]
    with_cognitive_init: bool,

    /// Build L0 / L1 / L2 retrieval tiers from `_st_embedding` properties
    /// that survived the migration. Disabled by default.
    #[arg(long = "with-tiers")]
    with_tiers: bool,

    /// Skip a specific property key during `phase_nodes`. Repeatable.
    /// Useful when a source has one or two "fat" keys (e.g. megalaw's
    /// full legal article text) that would bloat the in-memory cache
    /// beyond available RAM. Cognitive keys and `_st_embedding` are
    /// already skipped unconditionally.
    #[arg(long = "skip-prop", value_name = "KEY")]
    skip_prop: Vec<String>,

    /// Skip any property value whose serialized size hint exceeds this
    /// many bytes. Scalar types (bool, int, float, date, …) always fall
    /// below the threshold. Example: `--max-prop-bytes 16384` drops any
    /// string, bytes, list, map, or vector > 16 KB.
    #[arg(long = "max-prop-bytes", value_name = "BYTES")]
    max_prop_bytes: Option<usize>,

    /// Copy **no** node properties at all — labels + structure only.
    /// Nuclear option for producing a bounded-RSS skeleton you can
    /// enrich later out-of-band.
    #[arg(long = "skip-all-props")]
    skip_all_props: bool,

    /// Verbose tracing (debug level).
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Tracing: keep quiet by default; crate consumers (CI, hub) drive verbosity
    // through their own subscribers.
    let level = if cli.verbose {
        tracing::Level::DEBUG
    } else {
        tracing::Level::INFO
    };
    tracing_subscriber::fmt()
        .with_max_level(level)
        .with_target(false)
        .init();

    tracing::info!(
        "obrain-migrate starting: in={} out={} workers={} resume={}",
        cli.input.display(),
        cli.output.display(),
        cli.workers,
        cli.resume
    );

    converter::migrate(&converter::MigrateOptions {
        input: cli.input,
        output: cli.output,
        workers: cli.workers,
        resume: cli.resume,
        with_cognitive_init: cli.with_cognitive_init,
        with_tiers: cli.with_tiers,
        skip_prop_keys: cli.skip_prop,
        max_prop_bytes: cli.max_prop_bytes,
        skip_all_props: cli.skip_all_props,
    })?;

    tracing::info!("obrain-migrate: done");
    Ok(())
}
