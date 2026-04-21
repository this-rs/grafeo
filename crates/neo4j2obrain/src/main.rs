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
//! intermediate LpgStore and left cognitive init for the caller.
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

/// Streaming Neo4j → substrate importer with inline cognitive
/// bootstrapping.
#[derive(Debug, Parser)]
#[command(name = "neo4j2obrain", version, about)]
struct Cli {
    /// Neo4j bolt URL, e.g. `bolt://localhost:7687`.
    #[arg(long)]
    neo4j_url: String,

    /// Neo4j username.
    #[arg(long)]
    user: String,

    /// Neo4j password. Prefer env var + `--password-env` in CI.
    #[arg(long, env = "NEO4J_PASSWORD")]
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

    /// Embeddings model name used when nodes are missing `_st_embedding`
    /// but carry a textual property. The actual ONNX resolution happens
    /// in the embeddings crate — this flag is a label passed through.
    #[arg(long, default_value = "all-MiniLM-L6-v2")]
    embeddings_model: String,

    /// Verbose tracing (debug level).
    #[arg(short, long)]
    verbose: bool,
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

    tracing::info!(
        "neo4j2obrain: {} → {} (cognitive={}, batch={})",
        cli.neo4j_url,
        cli.out.display(),
        cli.cognitive,
        cli.batch_size
    );

    let opts = pipeline::PipelineOptions {
        neo4j_url: cli.neo4j_url,
        neo4j_user: cli.user,
        neo4j_password: cli.password,
        output: cli.out,
        run_cognitive: cli.cognitive,
        batch_size: cli.batch_size,
        embeddings_model: cli.embeddings_model,
    };

    pipeline::run(&opts)?;

    tracing::info!("neo4j2obrain: done");
    Ok(())
}
