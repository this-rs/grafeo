//! # `neo4j2obrain` — streaming Neo4j → substrate importer
//!
//! Reads a Neo4j database over Bolt and writes an obrain
//! [`SubstrateStore`](obrain_substrate::SubstrateStore) directly, running
//! the 8-step cognitive pipeline inline so the resulting `.obrain` file
//! is cognitively alive the moment it is opened:
//!
//! | Stage | Responsibility                                         |
//! |------:|---------------------------------------------------------|
//! |     1 | Stream nodes + edges from Neo4j (bolt, batched 10k)     |
//! |     2 | Fill missing `_st_embedding` via SentenceTransformer    |
//! |     3 | Build L0 / L1 / L2 retrieval tiers                      |
//! |     4 | LDleiden batch → `community_id`                         |
//! |     5 | PageRank batch → `centrality_cached`                    |
//! |     6 | Ricci-Ollivier batch → `edge.ricci` + `node.curvature`  |
//! |     7 | Derive COACT edges from co-occurrence metadata          |
//! |     8 | Seed engrams over high-density × high-centrality clusters |
//!
//! Replaces the previous `neo4j2obrain` pipeline, which wrote an
//! intermediate in-memory graph and left cognitive init for the caller.
//! Since T17 cutover, the output is a substrate-format `.obrain/` directory.
//!
//! Usage:
//!
//! ```text
//! neo4j2obrain \
//!     --neo4j-url bolt://localhost:7687 \
//!     --user neo4j --password ****      \
//!     --out path/to/out.obrain          \
//!     --cognitive                       \
//!     [--batch-size 10000]              \
//!     [--embeddings-model all-MiniLM-L6-v2]
//! ```

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

mod pipeline;
mod cognitive_init;
#[cfg(feature = "neo4j-bolt")]
mod bolt_reader;
#[cfg(feature = "embed")]
mod embedder;
#[cfg(feature = "embed")]
mod sentence_transformer;

/// Streaming Neo4j → substrate importer with inline cognitive
/// bootstrapping.
#[derive(Debug, Parser)]
#[command(name = "neo4j2obrain", version, about)]
struct Cli {
    /// Neo4j bolt URL, e.g. `bolt://localhost:7687`. Required unless
    /// `--enrich-only` is passed.
    #[arg(long, default_value = "")]
    neo4j_url: String,

    /// Neo4j username. Required unless `--enrich-only` is passed.
    #[arg(long, default_value = "")]
    user: String,

    /// Neo4j password. Prefer env var + `--password-env` in CI.
    /// Required unless `--enrich-only` is passed.
    #[arg(long, env = "NEO4J_PASSWORD", default_value = "")]
    password: String,

    /// Output path for the substrate store. Created if absent.
    #[arg(long, value_name = "PATH")]
    out: PathBuf,

    /// Run the 8-step cognitive pipeline during import (LDleiden,
    /// PageRank, Ricci, COACT, engram seeding). Default on — pass
    /// `--no-cognitive` to skip.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    cognitive: bool,

    /// Batch size for the bolt streaming reader. Larger = more memory,
    /// better throughput.
    #[arg(long, default_value_t = 10_000)]
    batch_size: usize,

    /// Embeddings model label. Kept as a string for log lines / stats
    /// tagging; the actual model is resolved from `--st-model-path` +
    /// `--st-vocab-path`. Pass `stub` / `none` / empty to skip the
    /// embedding stage entirely (useful for dry runs and tests).
    #[arg(long, default_value = "all-MiniLM-L6-v2")]
    embeddings_model: String,

    /// Path to the `.onnx` sentence-transformer used by stage 2. Default
    /// points at the obrain-hub runtime model so embedded nodes are
    /// immediately query-compatible with the prod retrieval stack — no
    /// HuggingFace download, no model-name resolution, no drift between
    /// importer and runtime vector spaces.
    #[arg(
        long,
        default_value = "/Users/triviere/projects/obrain/obrain-chat/models/sentence-transformer/model.onnx"
    )]
    st_model_path: PathBuf,

    /// Path to the WordPiece `vocab.txt` matching `--st-model-path`
    /// (one token per line; line index = token id). Must be the exact
    /// vocab the ONNX model was exported with, or tokenisation will
    /// silently diverge and embeddings will be garbage.
    #[arg(
        long,
        default_value = "/Users/triviere/projects/obrain/obrain-chat/models/sentence-transformer/vocab.txt"
    )]
    st_vocab_path: PathBuf,

    /// Comma-separated property keys to inspect for text content when
    /// computing `_st_embedding` (stage 2). The first non-empty match
    /// per node (in the given order) is embedded. Override to tune for
    /// a given source schema — the default covers the common shapes
    /// (content / docstring / message / name …) but any concrete DB
    /// will benefit from pinning the one or two keys that actually
    /// carry semantic text.
    ///
    /// Example: `--text-keys docstring,message,content`
    #[arg(long, value_delimiter = ',')]
    text_keys: Option<Vec<String>>,

    /// Minimum character length of a text property before it is
    /// embedded. Defaults to 10 — short tags / ids / single-word names
    /// rarely carry enough signal to produce a meaningful embedding.
    #[arg(long, default_value_t = 10)]
    min_text_len: usize,

    /// Number of rayon worker threads for stage 2 (embedding
    /// inference). `0` = auto = `rayon::current_num_threads()` which
    /// defaults to the physical core count. Each worker owns its own
    /// `SentenceTransformer` session loaded with `intra_threads = 1`,
    /// so `N workers × 1 intra-thread` = N threads actually running
    /// ONNX — aligning the parallelism budget with physical cores.
    ///
    /// Override when: the machine is RAM-constrained (each session is
    /// ~23 MB of model weights), or when hyperthreading is wanted (set
    /// to 2× physical cores for SMT machines, though ORT typically
    /// gets no win from SMT at MiniLM sizes).
    #[arg(long, default_value_t = 0)]
    workers: usize,

    /// Verbose tracing (debug level).
    #[arg(short, long)]
    verbose: bool,

    /// Enrich-only mode: open an existing substrate, probe which
    /// cognitive columns are missing, and run only those. Stage 1
    /// (bolt stream) is always skipped; Stage 2 stays idempotent per-
    /// node; stages 3-8 consult a coverage probe and skip themselves
    /// when already populated.
    ///
    /// Use case: a base migrated via `obrain-migrate` (legacy .obrain
    /// → substrate) has the graph structure but no embeddings, tiers,
    /// communities, or engrams. `--enrich-only` adds those without
    /// re-streaming from Neo4j.
    #[arg(long, default_value_t = false)]
    enrich_only: bool,

    /// Force re-run of stages 3-8 even when the probe detects that the
    /// target column is already populated. Stage 2 remains idempotent
    /// (per-node `_st_embedding@L2_DIM` check) regardless.
    #[arg(long, default_value_t = false)]
    force_rerun_cognitive: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    tracing_subscriber::fmt()
        .with_max_level(if cli.verbose {
            tracing::Level::DEBUG
        } else {
            tracing::Level::INFO
        })
        .with_target(false)
        .init();

    // Wire the rayon global thread pool once. `cli.workers == 0`
    // means "rayon default" (physical core count). A second
    // initialisation attempt is silently ignored (the pool is global
    // and can only be set once per process).
    if cli.workers > 0 {
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(cli.workers)
            .build_global();
    }

    // Enforce neo4j connectivity args when we're in full-pipeline mode.
    // In --enrich-only mode, bolt is skipped entirely, so the strings
    // can stay empty.
    if !cli.enrich_only {
        if cli.neo4j_url.is_empty() || cli.user.is_empty() || cli.password.is_empty() {
            anyhow::bail!(
                "Missing Neo4j connectivity args (--neo4j-url, --user, --password) \
                 required when not using --enrich-only. Pass --enrich-only to run \
                 stages 2-8 on an existing substrate without touching Neo4j."
            );
        }
    }

    tracing::info!(
        "neo4j2obrain: mode={} out={} (cognitive={}, batch={}, workers={}, force_rerun={})",
        if cli.enrich_only { "enrich-only" } else { "full-import" },
        cli.out.display(),
        cli.cognitive,
        cli.batch_size,
        if cli.workers == 0 {
            format!("auto={}", rayon::current_num_threads())
        } else {
            cli.workers.to_string()
        },
        cli.force_rerun_cognitive,
    );

    let opts = pipeline::PipelineOptions {
        neo4j_url: cli.neo4j_url,
        neo4j_user: cli.user,
        neo4j_password: cli.password,
        output: cli.out,
        run_cognitive: cli.cognitive,
        batch_size: cli.batch_size,
        embeddings_model: cli.embeddings_model,
        st_model_path: cli.st_model_path,
        st_vocab_path: cli.st_vocab_path,
        text_keys: cli.text_keys.unwrap_or_else(|| {
            pipeline::DEFAULT_TEXT_KEYS
                .iter()
                .map(|s| s.to_string())
                .collect()
        }),
        min_text_len: cli.min_text_len,
        enrich_only: cli.enrich_only,
        force_rerun_cognitive: cli.force_rerun_cognitive,
    };

    pipeline::run(&opts)?;

    tracing::info!("neo4j2obrain: done");
    Ok(())
}
