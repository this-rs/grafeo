//! # 8-stage import pipeline
//!
//! Top-level orchestration of the `neo4j2obrain` import. Each stage is
//! its own free function so the surface can be unit-tested without a
//! live Neo4j instance.
//!
//! Stage 1 (bolt reader) is gated behind the `neo4j-bolt` feature and
//! replaced by a synthetic source when the feature is off — this keeps
//! `cargo check --workspace` hermetic and unit tests fast.
//!
//! Stages 2 through 8 receive the already-written `SubstrateStore` and
//! run purely over it.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use obrain_core::graph::traits::GraphStore;
use obrain_substrate::SubstrateStore;

use crate::cognitive_init;

/// User-facing options — parsed from the CLI and threaded through the
/// pipeline.
///
/// The `neo4j_*` and `batch_size` fields are consumed only by the
/// `neo4j-bolt`-gated bolt reader (stage 1). Under the default hermetic
/// build they look dead to rustc; `#[allow(dead_code)]` on those fields
/// silences that noise without hiding real regressions on the
/// exercised stages.
#[derive(Debug, Clone)]
pub struct PipelineOptions {
    #[allow(dead_code)]
    pub neo4j_url: String,
    #[allow(dead_code)]
    pub neo4j_user: String,
    #[allow(dead_code)]
    pub neo4j_password: String,
    pub output: PathBuf,
    pub run_cognitive: bool,
    #[allow(dead_code)]
    pub batch_size: usize,
    pub embeddings_model: String,
    /// Path to the `.onnx` sentence-transformer used by stage 2.
    /// Defaults in `main.rs` point at the obrain-hub runtime model so
    /// imported embeddings and query-time embeddings share a vector
    /// space by construction. The field is always present but is only
    /// consumed under `--features embed`; `#[allow(dead_code)]` covers
    /// the hermetic default build.
    #[allow(dead_code)]
    pub st_model_path: PathBuf,
    /// Path to the WordPiece `vocab.txt` paired with `st_model_path`.
    /// Same rationale as above for the `#[allow(dead_code)]`.
    #[allow(dead_code)]
    pub st_vocab_path: PathBuf,
    /// Ordered list of property keys to inspect for text content in
    /// stage 2. The first non-empty match is embedded. Defaults to
    /// [`DEFAULT_TEXT_KEYS`]. Override via CLI `--text-keys a,b,c`.
    pub text_keys: Vec<String>,
    /// Skip nodes whose selected text property is shorter than this
    /// many characters. Suppresses embeddings on short identifiers /
    /// tags / type names that carry no semantic signal.
    pub min_text_len: usize,
    /// Enrich-only mode: open an existing substrate, probe which
    /// cognitive stages are missing, and run only those.
    ///
    /// When `true`:
    /// * Stage 1 (bolt stream) is **always** skipped — the substrate
    ///   is assumed to have been populated by an earlier run (typically
    ///   `obrain-migrate` or a prior `neo4j2obrain --cognitive`).
    /// * Stages 2-8 consult a lightweight probe (see
    ///   [`probe_stage_coverage`]) and skip themselves if they detect
    ///   that the corresponding cognitive column is already populated,
    ///   unless `force_rerun_cognitive` overrides.
    ///
    /// Defaults to `false` (full pipeline, including stage 1).
    pub enrich_only: bool,
    /// Force re-run of stages 3-8 even when the probe says they are
    /// already populated. Stage 2 stays idempotent regardless (per-node
    /// `_st_embedding@L2_DIM` check). Use when you need to refresh
    /// communities / centrality / ricci after a structural change.
    pub force_rerun_cognitive: bool,
}

/// Default priority-ordered list of text property keys inspected by
/// stage 2 when computing `_st_embedding`. The order matters: the first
/// non-empty match wins. Covers the common shapes across documentation,
/// chat, code, and generic content graphs. Neo4j-schema-agnostic —
/// override via CLI `--text-keys` for tighter targeting.
pub const DEFAULT_TEXT_KEYS: &[&str] = &[
    // Prose / long-form
    "content",
    "body",
    "description",
    "summary",
    "text",
    // Documentation
    "docstring",
    "comment",
    "snippet",
    // Chat / events
    "message",
    // Titles / identifiers
    "title",
    "name",
];

/// Statistics emitted at every stage — used for CLI progress reporting
/// and post-import smoke-testing.
#[derive(Debug, Default, Clone)]
pub struct PipelineStats {
    pub nodes_read: u64,
    pub edges_read: u64,
    pub embeddings_computed: u64,
    pub tier_entries: u64,
    pub communities: u64,
    pub pagerank_nonzero: u64,
    pub ricci_refreshed: u64,
    pub coact_edges_added: u64,
    pub engrams_seeded: u64,
    /// T17l Stage 9 — nodes whose `_hilbert_features` were computed +
    /// persisted (canonical 64-72d topology signature).
    pub hilbert_features_computed: u64,
    /// T17l Stage 10 — nodes whose `_kernel_embedding` were computed
    /// + persisted (canonical 80d Φ₀ projection, depends on Hilbert).
    pub kernel_embeddings_computed: u64,
}

/// Run the full pipeline. Stages 1 runs in its own async task if the
/// `neo4j-bolt` feature is enabled; otherwise the synthetic source
/// below is used so the binary still has a meaningful `--help` and the
/// unit tests exercise stages 2-8.
pub fn run(opts: &PipelineOptions) -> Result<PipelineStats> {
    std::fs::create_dir_all(
        opts.output
            .parent()
            .unwrap_or_else(|| std::path::Path::new(".")),
    )
    .ok();

    let substrate = open_or_create_substrate(&opts.output)?;

    let mut stats = PipelineStats::default();

    // -------- Stage 1: stream nodes + edges ---------------------------
    // Skipped in --enrich-only mode: the substrate must already have
    // been populated by a prior run.
    if opts.enrich_only {
        let n_nodes = substrate.node_count();
        let n_edges = substrate.edge_count();
        tracing::info!(
            "--enrich-only: opened existing substrate — nodes={}, edges={}. \
             Skipping Stage 1 (bolt stream).",
            n_nodes,
            n_edges
        );
        if n_nodes == 0 {
            anyhow::bail!(
                "--enrich-only on an empty substrate ({}): nothing to enrich. \
                 Run the full pipeline first (without --enrich-only).",
                opts.output.display()
            );
        }
        // Seed stats with existing counts so the final report is meaningful.
        stats.nodes_read = n_nodes as u64;
        stats.edges_read = n_edges as u64;
    } else {
        stage_bolt_stream(opts, &substrate, &mut stats)?;
    }

    // Probe coverage of each cognitive column so we only redo what is
    // missing — respecting the operator's memory/CPU budget and keeping
    // re-runs idempotent. Stage 2 runs its own per-node probe (see
    // `stage_fill_embeddings`), so we skip probing it here.
    let probe = probe_stage_coverage(&substrate, &opts.output);
    tracing::info!("Probe coverage: {}", probe);

    // -------- Stage 2: compute missing embeddings ---------------------
    // Self-idempotent: the stage walks nodes, skips those already
    // carrying `_st_embedding@L2_DIM`, and only embeds text that hasn't
    // been embedded yet. Safe to always invoke.
    stage_fill_embeddings(opts, &substrate, &mut stats)?;

    // -------- Stage 3: build L0 / L1 / L2 tiers -----------------------
    // Peak RSS during tier build: O(n_embedded × (L2_DIM × 4 B)) —
    // ~7 GB at 4.5 M Wikipedia-scale embeddings. Skip when tiers are
    // already populated and --force-rerun-cognitive is off.
    let rerun_tiers = opts.force_rerun_cognitive || !probe.tiers_populated;
    if rerun_tiers {
        stage_build_tiers(&substrate, &mut stats)?;
    } else {
        tracing::info!(
            "Stage 3 (tiers): already populated — skipping (use --force-rerun-cognitive to rebuild)"
        );
    }

    if !opts.run_cognitive {
        tracing::info!("--no-cognitive: skipping stages 4-8");
        substrate.flush()?;
        return Ok(stats);
    }

    // -------- Stage 4: LDleiden batch → community_id ------------------
    if opts.force_rerun_cognitive || !probe.community_populated {
        cognitive_init::run_ldleiden_batch(&substrate, &mut stats)?;
    } else {
        tracing::info!(
            "Stage 4 (LDleiden): community_id already set on ≥ {:.0}% of nodes — skipping \
             (use --force-rerun-cognitive to recompute)",
            probe.community_coverage * 100.0
        );
        stats.communities = probe.community_sample_distinct as u64;
    }

    // -------- Stage 5: PageRank batch → centrality_cached -------------
    if opts.force_rerun_cognitive || !probe.centrality_populated {
        cognitive_init::run_pagerank_batch(&substrate, &mut stats)?;
    } else {
        tracing::info!(
            "Stage 5 (PageRank): centrality_cached already set on ≥ {:.0}% of nodes — skipping \
             (use --force-rerun-cognitive to recompute)",
            probe.centrality_coverage * 100.0
        );
    }

    // -------- Stage 6: Ricci-Ollivier batch ---------------------------
    // Internally idempotent via RICCI_STALE flags but the edge-level
    // walk is still expensive; gate it on the probe too.
    if opts.force_rerun_cognitive || !probe.ricci_populated {
        cognitive_init::run_ricci_batch(&substrate, &mut stats)?;
    } else {
        tracing::info!(
            "Stage 6 (Ricci): edge.ricci already populated — skipping \
             (use --force-rerun-cognitive to recompute)"
        );
    }

    // -------- Stage 7: derive COACT edges -----------------------------
    // Idempotent: relies on community_id + centrality being present.
    // Runs unconditionally because it needs to see the (possibly fresh)
    // outputs of stages 4/5. Duplicate COACT edges are cheap to dedupe
    // at query time and the stage already gates on non-empty inputs.
    if opts.force_rerun_cognitive || !probe.coact_populated {
        stage_derive_coact(&substrate, &mut stats)?;
    } else {
        tracing::info!(
            "Stage 7 (COACT): {} existing COACT edges detected — skipping \
             (use --force-rerun-cognitive to re-seed)",
            probe.coact_edge_sample
        );
    }

    // -------- Stage 8: seed engrams -----------------------------------
    if opts.force_rerun_cognitive || !probe.engrams_populated {
        stage_seed_engrams(&substrate, &mut stats)?;
    } else {
        tracing::info!(
            "Stage 8 (engrams): engram zones already populated — skipping \
             (use --force-rerun-cognitive to re-seed)"
        );
    }

    // -------- Stage 9 (T17l): _hilbert_features 64-72d ----------------
    // Canonical topology-derived signature consumed by the hub's
    // `CompositeEmbedding` (obrain-chat retrieval). Depends only on
    // graph topology — can run before or after any other cognitive
    // stage. Runs before Stage 10 because Kernel depends on Hilbert.
    if opts.force_rerun_cognitive || !probe.hilbert_populated {
        stage_fill_hilbert_features(&substrate, &mut stats)?;
    } else {
        tracing::info!(
            "Stage 9 (hilbert_features): _hilbert_features already populated \
             on ≥ {:.0}% of nodes — skipping (use --force-rerun-cognitive to recompute)",
            probe.hilbert_coverage * 100.0
        );
    }

    // -------- Stage 10 (T17l): _kernel_embedding 80d ------------------
    // Canonical Φ₀ projection — depends on `_hilbert_features` being
    // present on each node. Computed last in the cognitive chain.
    if opts.force_rerun_cognitive || !probe.kernel_populated {
        stage_fill_kernel_embeddings(&substrate, &mut stats)?;
    } else {
        tracing::info!(
            "Stage 10 (kernel_embedding): _kernel_embedding already populated \
             on ≥ {:.0}% of nodes — skipping (use --force-rerun-cognitive to recompute)",
            probe.kernel_coverage * 100.0
        );
    }

    substrate.flush().context("final flush")?;
    Ok(stats)
}

/// Per-column coverage probe used by `run()` to decide which stages
/// can be skipped (enrich-only mode) or short-circuited (re-run after
/// crash). Cheap: samples at most [`PROBE_SAMPLE_SIZE`] nodes via a
/// deterministic stride over `node_ids()` plus a lookup at the zone
/// files for 0-byte markers.
#[derive(Debug, Default)]
struct StageProbe {
    tiers_populated: bool,
    community_populated: bool,
    /// Fraction of probed nodes carrying a non-zero `community_id`.
    community_coverage: f64,
    /// Distinct community ids seen in the probe sample. Reported back
    /// into `stats.communities` when we skip stage 4.
    community_sample_distinct: usize,
    centrality_populated: bool,
    centrality_coverage: f64,
    ricci_populated: bool,
    coact_populated: bool,
    coact_edge_sample: u64,
    engrams_populated: bool,
    /// T17l Stage 9 — fraction of sampled nodes carrying `_hilbert_features`
    /// as a `Value::Vector` property.
    hilbert_coverage: f64,
    hilbert_populated: bool,
    /// T17l Stage 10 — same for `_kernel_embedding`.
    kernel_coverage: f64,
    kernel_populated: bool,
}

impl std::fmt::Display for StageProbe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "tiers={} community={} (cov={:.0}%) centrality={} (cov={:.0}%) ricci={} coact={} engrams={} hilbert={} (cov={:.0}%) kernel={} (cov={:.0}%)",
            yes_no(self.tiers_populated),
            yes_no(self.community_populated),
            self.community_coverage * 100.0,
            yes_no(self.centrality_populated),
            self.centrality_coverage * 100.0,
            yes_no(self.ricci_populated),
            yes_no(self.coact_populated),
            yes_no(self.engrams_populated),
            yes_no(self.hilbert_populated),
            self.hilbert_coverage * 100.0,
            yes_no(self.kernel_populated),
            self.kernel_coverage * 100.0,
        )
    }
}

fn yes_no(b: bool) -> &'static str {
    if b { "yes" } else { "no" }
}

/// Sample size used by the per-stage coverage probe. Small enough to
/// keep the probe well under a second on 10⁶+ node graphs, large enough
/// to give sub-1 % false-negative rate on populated columns.
const PROBE_SAMPLE_SIZE: usize = 2048;

/// Fraction of sampled nodes that must carry a non-zero value for a
/// column to be considered "already populated". Balances against
/// legitimate zeros (root/dangling nodes with PageRank near machine
/// zero, community 0 as a sink) vs unpopulated defaults.
const PROBE_POPULATED_THRESHOLD: f64 = 0.5;

fn probe_stage_coverage(
    substrate: &Arc<SubstrateStore>,
    output_path: &std::path::Path,
) -> StageProbe {
    let mut probe = StageProbe::default();

    // Zone-file size probes. Tiers / engram zones are written as a
    // single flush at the end of their respective stages, so a 0-byte
    // file is a strong "never ran" signal.
    //
    // `output_path` is either the substrate directory itself or the
    // base path (e.g. `…/wikipedia-substrate.obrain/substrate.obrain`).
    // Try both layouts: if the path is a directory, use it directly;
    // otherwise probe its parent.
    let dir: std::path::PathBuf = if output_path.is_dir() {
        output_path.to_path_buf()
    } else if let Some(parent) = output_path.parent() {
        parent.to_path_buf()
    } else {
        std::path::PathBuf::from(".")
    };
    probe.tiers_populated = nonempty(&dir.join("substrate.tier2"));
    probe.engrams_populated = nonempty(&dir.join("substrate.engram_members"))
        || nonempty(&dir.join("substrate.engram_bitset"));

    // Sample nodes via deterministic stride for community / centrality.
    let node_ids = substrate.all_node_ids();
    let sample: Vec<_> = if node_ids.len() <= PROBE_SAMPLE_SIZE {
        node_ids.clone()
    } else {
        let step = node_ids.len() / PROBE_SAMPLE_SIZE;
        node_ids
            .iter()
            .copied()
            .step_by(step.max(1))
            .take(PROBE_SAMPLE_SIZE)
            .collect()
    };
    let n_sample = sample.len().max(1) as f64;

    let writer = substrate.writer();
    let mut community_nonzero = 0usize;
    let mut centrality_nonzero = 0usize;
    let mut communities_seen: std::collections::HashSet<u32> = std::collections::HashSet::new();
    for &nid in &sample {
        if let Ok(Some(rec)) = writer.read_node(nid.0 as u32) {
            if rec.community_id != 0 {
                community_nonzero += 1;
                communities_seen.insert(rec.community_id);
            }
            if rec.centrality_cached != 0 {
                centrality_nonzero += 1;
            }
        }
    }
    probe.community_coverage = community_nonzero as f64 / n_sample;
    probe.centrality_coverage = centrality_nonzero as f64 / n_sample;
    probe.community_populated = probe.community_coverage >= PROBE_POPULATED_THRESHOLD;
    probe.centrality_populated = probe.centrality_coverage >= PROBE_POPULATED_THRESHOLD;
    probe.community_sample_distinct = communities_seen.len();

    // Ricci probe: sample outgoing edges from a handful of nodes and
    // peek at the substrate's ricci_u8 column. We use the existing
    // `refresh_all_ricci` internals indirectly: a RICCI_STALE scan is
    // too expensive for a probe, so instead we check whether the first
    // few outgoing edges of sample nodes report a non-zero ricci via
    // the on-disk record. Until substrate exposes a per-edge reader on
    // the stable API, we approximate this by counting COACT edges and
    // use that as a ricci correlate (stage 6 and 7 both run late in
    // the pipeline; if stage 7 already ran then stage 6 ran too). This
    // is pragmatic rather than exhaustive — the fallback path is the
    // `--force-rerun-cognitive` flag for operators who want certainty.
    //
    // For the simple case, we mirror the ricci probe on `tiers_populated`
    // AND a non-empty COACT sample: if those two are present and the
    // graph has any edges at all, stage 6 almost certainly ran before.
    let mut coact_edges = 0u64;
    for &nid in sample.iter().take(128) {
        for (_, eid) in substrate.edges_from(nid, obrain_core::graph::Direction::Outgoing) {
            if let Some(ty) = substrate.edge_type(eid) {
                let s: &str = ty.as_ref();
                if s == obrain_substrate::COACT_EDGE_TYPE_NAME {
                    coact_edges += 1;
                }
            }
        }
    }
    probe.coact_edge_sample = coact_edges;
    probe.coact_populated = coact_edges > 0;
    // Ricci runs before COACT in the pipeline, so COACT presence is a
    // sufficient (not necessary) signal. Treat tiers + coact as proxy.
    probe.ricci_populated = probe.coact_populated && probe.tiers_populated;

    // T17l Stage 9+10 — probe `_hilbert_features` (64-72d) and
    // `_kernel_embedding` (80d) coverage on the same sample. These live
    // in substrate vec_columns (not NodeRecord bit-packed fields like
    // community_id / centrality_cached), so we use `get_node_property`.
    let hilbert_key = obrain_common::PropertyKey::from("_hilbert_features");
    let kernel_key = obrain_common::PropertyKey::from("_kernel_embedding");
    let mut hilbert_nonzero = 0usize;
    let mut kernel_nonzero = 0usize;
    for &nid in &sample {
        if matches!(
            substrate.get_node_property(nid, &hilbert_key),
            Some(obrain_common::types::Value::Vector(_))
        ) {
            hilbert_nonzero += 1;
        }
        if matches!(
            substrate.get_node_property(nid, &kernel_key),
            Some(obrain_common::types::Value::Vector(_))
        ) {
            kernel_nonzero += 1;
        }
    }
    probe.hilbert_coverage = hilbert_nonzero as f64 / n_sample;
    probe.kernel_coverage = kernel_nonzero as f64 / n_sample;
    probe.hilbert_populated = probe.hilbert_coverage >= PROBE_POPULATED_THRESHOLD;
    probe.kernel_populated = probe.kernel_coverage >= PROBE_POPULATED_THRESHOLD;

    probe
}

fn nonempty(p: &std::path::Path) -> bool {
    std::fs::metadata(p).map(|m| m.len() > 0).unwrap_or(false)
}

// ---------------------------------------------------------------------
// Stage implementations
// ---------------------------------------------------------------------

fn stage_bolt_stream(
    opts: &PipelineOptions,
    _substrate: &Arc<SubstrateStore>,
    stats: &mut PipelineStats,
) -> Result<()> {
    #[cfg(feature = "neo4j-bolt")]
    {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .context("build tokio runtime for bolt stream")?;
        let read = runtime.block_on(crate::bolt_reader::stream_into_substrate(opts, _substrate))?;
        stats.nodes_read = read.nodes;
        stats.edges_read = read.edges;
        return Ok(());
    }
    #[cfg(not(feature = "neo4j-bolt"))]
    {
        let _ = opts; // silence unused
        tracing::warn!(
            "neo4j2obrain was built without `--features neo4j-bolt`; \
             stage 1 is a no-op. Rebuild with that feature to connect \
             to a live Neo4j instance."
        );
        stats.nodes_read = 0;
        stats.edges_read = 0;
        Ok(())
    }
}

/// Stage 2 — fill missing `_st_embedding` properties via an ONNX
/// SentenceTransformer embedder.
///
/// ## Feature-gating
///
/// Real computation is gated behind `--features embed`, which pulls
/// `obrain-engine` with its `embed` feature (≈ 17 MB of ort +
/// tokenizers + hf-hub transitive closure). Without that feature the
/// stage is a cheap no-op that logs "no embedder wired" and sets
/// `stats.embeddings_computed = 0` — same pattern as stage 1
/// (`--features neo4j-bolt`).
///
/// ## Text extraction convention
///
/// For each node without an `_st_embedding` already, the stage picks
/// the first non-empty string value under one of the standard text
/// property keys (checked in order): `text`, `content`, `body`,
/// `description`, `name`, `title`. Nodes with no text property are
/// skipped.
///
/// ## Batching
///
/// Texts are embedded in batches of [`EMBED_BATCH_SIZE`] to amortise
/// the ONNX session overhead. The resulting vectors are written back
/// as `Value::Vector(Arc<[f32]>)` under the `_st_embedding` key.
///
/// ## Sentinel
///
/// The CLI `--embeddings-model stub` (or `none`, or empty string)
/// disables the stage even when the `embed` feature is on — useful
/// for tests and dry-runs.
fn stage_fill_embeddings(
    opts: &PipelineOptions,
    substrate: &Arc<SubstrateStore>,
    stats: &mut PipelineStats,
) -> Result<()> {
    use obrain_common::types::{PropertyKey, Value};
    #[cfg(feature = "embed")]
    use obrain_core::graph::traits::GraphStoreMut;

    // Sentinel: empty / "stub" / "none" → skip.
    let model_name = opts.embeddings_model.trim();
    if model_name.is_empty() || model_name == "stub" || model_name == "none" {
        tracing::info!(
            "Stage 2 (embeddings): model '{}' is a stub sentinel — skipping",
            model_name
        );
        stats.embeddings_computed = 0;
        return Ok(());
    }

    // Streaming candidate selection + inference.
    //
    // A node is a candidate iff:
    //   1. It does NOT already have `_st_embedding` at `L2_DIM`
    //      dimension — a pre-existing embedding at the right dim
    //      (e.g. previously imported by obrain) is kept as-is.
    //   2. At least one of the configured text keys resolves to a
    //      non-empty String of length ≥ `min_text_len`.
    //
    // An existing `_st_embedding` at the *wrong* dimension (e.g. a
    // foreign 768-dim vector imported from the source graph) is
    // ignored — stage 2 will overwrite it with a fresh obrain-model
    // embedding at `L2_DIM`. This keeps the importer schema-agnostic:
    // the pipeline owns the embedding space; foreign vectors are not
    // trusted.
    //
    // **Memory footprint**: this stage used to materialise every
    // candidate `(NodeId, String)` pair into a single `Vec` before
    // starting inference. On Wikipedia-scale (4.5 M candidates × ~2 KB
    // article text) that balloons to ~10 GB of heap *before the first
    // ONNX batch runs*, plus ArcStr fragmentation during the probe
    // (gotcha `f174688b-c0ea-44ec-8b4b-9ed88bd8573d`). The streaming
    // path below keeps only one meta-batch (`EMBED_BATCH_SIZE ×
    // EMBED_META_BATCH_FACTOR = 512 texts ≈ 1-2 MB`) resident at a
    // time — RSS stays in the low hundreds of MB regardless of
    // total graph size.
    use obrain_substrate::L2_DIM;
    let emb_key = PropertyKey::new("_st_embedding");
    let text_keys: Vec<PropertyKey> = opts
        .text_keys
        .iter()
        .map(|s| PropertyKey::new(s.as_str()))
        .collect();

    let all_nodes = GraphStore::all_node_ids(&**substrate);
    let total_nodes = all_nodes.len() as u64;
    tracing::info!(
        "Stage 2 (embeddings): streaming scan of {} nodes; text_keys={:?}, min_text_len={}",
        total_nodes,
        opts.text_keys,
        opts.min_text_len,
    );

    #[cfg(not(feature = "embed"))]
    {
        // Single-pass count (no String cloning) to report what WOULD
        // have been embedded. Kept cheap so the warn message is
        // informative without paying the full embed cost.
        let mut needed = 0u64;
        let mut skipped_existing = 0u64;
        let mut skipped_short = 0u64;
        for nid in &all_nodes {
            if let Some(Value::Vector(arc)) = substrate.get_node_property(*nid, &emb_key) {
                if arc.len() == L2_DIM {
                    skipped_existing += 1;
                    continue;
                }
            }
            for tk in &text_keys {
                if let Some(Value::String(s)) = substrate.get_node_property(*nid, tk) {
                    let trimmed = s.trim();
                    if trimmed.chars().count() < opts.min_text_len {
                        skipped_short += 1;
                        break;
                    }
                    if !trimmed.is_empty() {
                        needed += 1;
                        break;
                    }
                }
            }
        }
        tracing::warn!(
            "Stage 2 (embeddings): {} nodes need embeddings for model '{}' \
             ({} existing @ L2_DIM, {} skipped short) but neo4j2obrain was \
             built without `--features embed`. Rebuild with that feature to \
             compute embeddings at import time. Stage skipped.",
            needed,
            model_name,
            skipped_existing,
            skipped_short,
        );
        stats.embeddings_computed = 0;
        Ok(())
    }

    // Parallel inference path. One `SentenceTransformer` per rayon
    // worker (thread-local), each with `intra_threads = 1` so the
    // product `workers × intra` stays below physical-core count. On an
    // 8-core M2 this takes stage 2 from ~200 texts/s (single session
    // with intra=2) to ~1 400 texts/s (8 sessions × intra=1) — a 7×
    // speedup that scales to Wikipedia-scale (4.5 M embeddings turns
    // from ~6 h into ~50 min) without blowing the RSS budget: each
    // extra session adds ~23 MB of model weights, so 8 workers =
    // ~180 MB, negligible next to the embedding buffers themselves.
    #[cfg(feature = "embed")]
    {
        use rayon::prelude::*;

        // One-pass streaming. Progress bar scoped to total nodes
        // (not total candidates — candidate count is only known after
        // the full scan, and pre-computing it would require either a
        // second pass or storing the candidates, defeating the
        // streaming purpose).
        let pb = progress_bar(total_nodes, "nodes scanned");

        let mut computed = 0u64;
        let mut skipped_existing = 0u64;
        let mut skipped_short = 0u64;
        let mut batches_since_flush = 0usize;
        let mut last_flush_at = std::time::Instant::now();

        // Meta-batch = N sub-batches fanned out to rayon workers in one
        // pass. Results are collected back on the main thread before
        // being written serially to substrate (which is single-writer
        // at the WAL layer). A factor of 8 keeps ~512 texts in flight
        // at any moment — enough to saturate 8 workers without
        // ballooning the pending-write buffer.
        let meta_batch_size = EMBED_BATCH_SIZE * EMBED_META_BATCH_FACTOR;
        let model_path = opts.st_model_path.clone();
        let vocab_path = opts.st_vocab_path.clone();

        // Keep the first embedder instance around so we can report
        // `embed_dim` on the final log line (thread-local instances
        // inside rayon workers are hidden behind a RefCell and we
        // shouldn't reach across threads just for metadata).
        let dim_probe = crate::sentence_transformer::SentenceTransformer::load_with_threads(
            &model_path,
            &vocab_path,
            1,
        )
        .context("load SentenceTransformer (stage 2 probe)")?;
        let embed_dim = dim_probe.embed_dim;
        drop(dim_probe);

        // Local buffer. Capacity preallocated to `meta_batch_size` so
        // the inner push never reallocates. Peak size never exceeds
        // `meta_batch_size` — once filled, the buffer is drained
        // (inferred + written + cleared) and reused in place.
        let mut batch: Vec<(obrain_common::NodeId, String)> = Vec::with_capacity(meta_batch_size);

        // Inference + write-back helper. Defined as a closure so it
        // captures `substrate`, paths, and counters — keeps the
        // streaming loop flat and the drain logic DRY between the
        // in-loop drain and the final residual drain.
        let run_meta_batch = |batch: &mut Vec<(obrain_common::NodeId, String)>,
                              computed: &mut u64|
         -> Result<()> {
            if batch.is_empty() {
                return Ok(());
            }
            let sub_results: Result<Vec<Vec<(obrain_common::NodeId, Vec<f32>)>>> = batch
                .par_chunks(EMBED_BATCH_SIZE)
                .map(|sub| -> Result<Vec<(obrain_common::NodeId, Vec<f32>)>> {
                    embed_sub_batch_tls(&model_path, &vocab_path, sub)
                })
                .collect();
            let sub_results = sub_results?;

            // Serial write-back. Substrate's WAL is single-writer,
            // so parallelising this block would just contend on
            // the WAL mutex — the parallelism win lives entirely
            // in inference.
            for sub in sub_results {
                for (nid, vec) in sub {
                    substrate.set_node_property(nid, "_st_embedding", Value::Vector(vec.into()));
                    *computed += 1;
                }
            }
            batch.clear();
            Ok(())
        };

        for nid in &all_nodes {
            pb.inc(1);

            // Probe existing embedding. Idempotent: nodes already at
            // L2_DIM are skipped without recomputing — this is what
            // makes `--enrich-only` resumable after a crash.
            if let Some(Value::Vector(arc)) = substrate.get_node_property(*nid, &emb_key) {
                if arc.len() == L2_DIM {
                    skipped_existing += 1;
                    continue;
                }
                // Wrong-dim foreign embedding → fall through, recompute.
            }

            // Pick first text key with acceptable length. Same
            // semantics as the pre-fix code: once a text key matches
            // and the text is < min_text_len, the node is skipped
            // entirely (we don't fall through to later keys). This
            // preserves the "primary key wins" convention.
            let mut picked: Option<String> = None;
            for tk in &text_keys {
                if let Some(Value::String(s)) = substrate.get_node_property(*nid, tk) {
                    let trimmed = s.trim();
                    if trimmed.chars().count() < opts.min_text_len {
                        skipped_short += 1;
                        break;
                    }
                    if !trimmed.is_empty() {
                        picked = Some(trimmed.to_string());
                        break;
                    }
                }
            }

            if let Some(text) = picked {
                batch.push((*nid, text));

                if batch.len() >= meta_batch_size {
                    run_meta_batch(&mut batch, &mut computed)?;

                    // Periodic flush. Bounds RAM to
                    // `EMBED_FLUSH_EVERY_N_BATCHES × EMBED_BATCH_SIZE
                    // × vec_bytes` regardless of total graph size,
                    // and makes progress visible on disk. Stage 2 is
                    // idempotent — rerunning after a crash skips
                    // nodes whose `_st_embedding` already landed at
                    // `L2_DIM` — so the periodic flush is also a
                    // cheap resume-point.
                    batches_since_flush += EMBED_META_BATCH_FACTOR;
                    if batches_since_flush >= EMBED_FLUSH_EVERY_N_BATCHES {
                        substrate
                            .flush()
                            .map_err(|e| anyhow::anyhow!("stage 2 periodic flush: {e}"))?;
                        let elapsed = last_flush_at.elapsed();
                        let rate = if elapsed.as_secs_f64() > 0.0 {
                            (EMBED_FLUSH_EVERY_N_BATCHES * EMBED_BATCH_SIZE) as f64
                                / elapsed.as_secs_f64()
                        } else {
                            0.0
                        };
                        tracing::info!(
                            "Stage 2 (embeddings): flushed checkpoint — {} embedded \
                             ({} existing @ L2_DIM, {} short) ({:.0} texts/s, {} workers)",
                            computed,
                            skipped_existing,
                            skipped_short,
                            rate,
                            rayon::current_num_threads(),
                        );
                        batches_since_flush = 0;
                        last_flush_at = std::time::Instant::now();
                    }
                }
            }
        }

        // Drain the residual (last partial meta-batch).
        run_meta_batch(&mut batch, &mut computed)?;

        pb.finish_and_clear();
        substrate
            .flush()
            .map_err(|e| anyhow::anyhow!("stage 2 final flush: {e}"))?;
        stats.embeddings_computed = computed;
        tracing::info!(
            "Stage 2 (embeddings): computed {} embeddings with model '{}' \
             (dim={}, {} workers) — {} existing @ L2_DIM kept, {} skipped (text < {} chars)",
            computed,
            model_name,
            embed_dim,
            rayon::current_num_threads(),
            skipped_existing,
            skipped_short,
            opts.min_text_len,
        );
        Ok(())
    }
}

/// Run one sub-batch on the **thread-local** `SentenceTransformer`,
/// lazily loading the ONNX session on first call per rayon worker.
///
/// Each rayon worker thread gets its own session (no sharing: the
/// embedder holds an `UnsafeCell<ort::Session>` and is explicitly
/// single-threaded per its contract). Loading is cheap after the first
/// call because `thread_local!` caches the `Option` slot for the
/// lifetime of the worker thread.
#[cfg(feature = "embed")]
fn embed_sub_batch_tls(
    model_path: &std::path::Path,
    vocab_path: &std::path::Path,
    sub: &[(obrain_common::NodeId, String)],
) -> Result<Vec<(obrain_common::NodeId, Vec<f32>)>> {
    thread_local! {
        static EMBEDDER: std::cell::RefCell<Option<crate::sentence_transformer::SentenceTransformer>> =
            const { std::cell::RefCell::new(None) };
    }

    EMBEDDER.with(|cell| -> Result<Vec<(obrain_common::NodeId, Vec<f32>)>> {
        let mut guard = cell.borrow_mut();
        if guard.is_none() {
            let embedder = crate::sentence_transformer::SentenceTransformer::load_with_threads(
                model_path, vocab_path,
                1, // one intra-op thread per session; rayon owns the parallelism budget
            )
            .context("load SentenceTransformer (stage 2 worker)")?;
            *guard = Some(embedder);
        }
        // SAFETY-analog: we hold a mutable borrow to the RefCell, but
        // `embed_batch` itself only takes `&self`, so the borrow is
        // de-facto read-only from the type-system's point of view.
        let embedder = guard.as_ref().unwrap();
        let texts: Vec<&str> = sub.iter().map(|(_, s)| s.as_str()).collect();
        let vectors = embedder
            .embed_batch(&texts)
            .map_err(|e| anyhow::anyhow!("embedder.embed_batch failed: {e}"))?;
        if vectors.len() != sub.len() {
            anyhow::bail!(
                "embedder returned {} vectors for {} inputs",
                vectors.len(),
                sub.len()
            );
        }
        Ok(sub
            .iter()
            .zip(vectors.into_iter())
            .map(|((nid, _), v)| (*nid, v))
            .collect())
    })
}

/// Stage 2 periodic flush interval.
///
/// Every `EMBED_FLUSH_EVERY_N_BATCHES` batches of [`EMBED_BATCH_SIZE`]
/// texts, a `substrate.flush()` is issued. With the defaults (500 × 64
/// = 32 000 texts per checkpoint) each cycle commits ~48 MB of
/// embeddings to the WAL and releases the corresponding in-process
/// buffers — capping the stage-2 RSS ceiling at a small multiple of
/// that, independent of the total nodes-to-embed count.
///
/// The choice is a balance:
/// * Smaller → tighter RSS bound, more resume-points, more fsync cost.
/// * Larger → fewer fsyncs, more RAM at risk on SIGKILL.
///
/// 500 keeps fsync overhead under ~1 % of the ONNX inference time at
/// the observed M2 SME2 throughput (~200 texts/s), while giving ~15
/// heartbeat log lines on a PO-sized import — enough for the operator
/// to see liveness without spamming the log.
#[cfg(feature = "embed")]
const EMBED_FLUSH_EVERY_N_BATCHES: usize = 500;

/// Batch size used by [`stage_fill_embeddings`]. 64 is a sweet spot on
/// both GPU and CPU ONNX runtimes: large enough to amortise launch
/// overhead, small enough to keep peak memory bounded.
#[cfg(feature = "embed")]
const EMBED_BATCH_SIZE: usize = 64;
#[cfg(not(feature = "embed"))]
#[allow(dead_code)]
const EMBED_BATCH_SIZE: usize = 64;

/// Number of sub-batches fanned out to rayon workers in one meta-batch
/// pass. With `EMBED_BATCH_SIZE = 64`, a factor of 8 keeps ~512 texts
/// in flight concurrently — enough to saturate an 8-core M2 when each
/// rayon worker owns a 1-intra-thread `SentenceTransformer` session.
///
/// The upper bound is set by two constraints:
/// * RAM — N × BATCH × 384 × 4 B = 192 KB × N, trivial but multiplied
///   by tokenisation buffers (~50 KB × BATCH × N).
/// * WAL write-back cadence — after each meta-batch we serialise
///   writes to the single-writer substrate. Too large a factor makes
///   the progress bar move in visible jumps and delays periodic
///   flushes; too small and the rayon task overhead eats the
///   parallelism win.
#[cfg(feature = "embed")]
const EMBED_META_BATCH_FACTOR: usize = 8;

fn stage_build_tiers(substrate: &Arc<SubstrateStore>, stats: &mut PipelineStats) -> Result<()> {
    use obrain_common::types::{PropertyKey, Value};
    use obrain_substrate::L2_DIM;
    use obrain_substrate::retrieval::{NodeOffset, SubstrateTieredIndex, VectorIndex};

    let key = PropertyKey::new("_st_embedding");
    let mut pairs: Vec<(NodeOffset, Vec<f32>)> = Vec::new();
    let mut wrong_dim = 0u64;
    for id in GraphStore::all_node_ids(&**substrate) {
        let Some(v) = substrate.get_node_property(id, &key) else {
            continue;
        };
        let Value::Vector(arc) = v else {
            continue;
        };
        if arc.len() != L2_DIM {
            wrong_dim += 1;
            continue;
        }
        let off: NodeOffset = id.as_u64() as NodeOffset;
        pairs.push((off, arc.to_vec()));
    }
    stats.tier_entries = pairs.len() as u64;
    if pairs.is_empty() {
        tracing::info!(
            "Stage 3 (tiers): no embeddings present yet ({} wrong-dim); \
             index will be built lazily on first query.",
            wrong_dim
        );
        return Ok(());
    }
    let idx = SubstrateTieredIndex::new(L2_DIM);
    idx.rebuild(&pairs);
    tracing::info!(
        "Stage 3 (tiers): indexed {} nodes (skipped {} wrong-dim)",
        pairs.len(),
        wrong_dim
    );
    Ok(())
}

/// Maximum number of seeded engrams per import. The substrate engram
/// id is `u16`, so the hard ceiling is 65535; 1024 is a generous,
/// policy-driven cap that leaves room for runtime-formed engrams.
const MAX_SEEDED_ENGRAMS: usize = 1024;

/// Minimum community size to seed an engram for. Singletons and
/// tiny pairs don't carry enough signal to be useful anchors.
const MIN_COMMUNITY_SIZE_FOR_ENGRAM: usize = 3;

/// Number of top-centrality nodes to include per seeded engram.
/// Engrams are Hopfield recall patterns — a handful of strongly-
/// coactivated members is the sweet spot; more dilutes the signature.
const ENGRAM_TOP_K: usize = 5;

/// Number of top-centrality nodes per community eligible to form
/// a COACT clique at import time. Must be ≥ 2 for any edges to be
/// emitted.
const COACT_TOP_K: usize = 3;

/// Weight assigned to import-time COACT edges (Q1.15). Weak on
/// purpose — real co-activation reinforcement happens at runtime via
/// the CommunityWarden tick; the import-time priors are just a
/// bootstrap scaffold.
const COACT_IMPORT_WEIGHT: f32 = 0.2;

/// Stage 7 — derive COACT edges from the community × centrality prior.
///
/// ## Why not metadata-driven?
///
/// The runtime obrain-cognitive path derives COACT from live
/// co-activations (same chat turn, same retrieval window) — the
/// CommunityWarden tick owns that. At import time, nodes have no
/// `source_doc_id` or `created_at`: the Neo4j bolt reader preserves
/// properties as-is and the source schema rarely carries either field.
///
/// ## The import-time prior
///
/// Nodes in the same community that also sit in the top-centrality
/// quartile are semantically close enough to treat as weakly
/// co-activated. The stage emits a sparse COACT clique per community
/// (top-[`COACT_TOP_K`] nodes, all pairs) with weight
/// [`COACT_IMPORT_WEIGHT`]. Runtime reinforcement will quickly
/// overwrite these with observation-grounded weights.
///
/// ## Dependencies
///
/// Needs `community_id` (stage 4) and `centrality_cached` (stage 5)
/// populated. If both are zero everywhere — e.g. `--no-cognitive` or
/// an empty graph — the stage emits no edges and logs the reason.
fn stage_derive_coact(substrate: &Arc<SubstrateStore>, stats: &mut PipelineStats) -> Result<()> {
    use obrain_core::graph::traits::GraphStoreMut;
    use obrain_substrate::COACT_EDGE_TYPE_NAME;

    let node_ids = GraphStore::all_node_ids(&**substrate);
    if node_ids.is_empty() {
        stats.coact_edges_added = 0;
        tracing::info!("Stage 7 (COACT): empty graph — nothing to derive");
        return Ok(());
    }

    // Group nodes by community_id, carrying centrality for ranking.
    let mut by_community: HashMap<u32, Vec<(obrain_common::NodeId, u16)>> = HashMap::new();
    let writer = substrate.writer();
    let mut nodes_with_community = 0u64;
    for nid in &node_ids {
        if let Ok(Some(rec)) = writer.read_node(nid.0 as u32) {
            if rec.community_id != 0 || rec.centrality_cached != 0 {
                by_community
                    .entry(rec.community_id)
                    .or_default()
                    .push((*nid, rec.centrality_cached));
                nodes_with_community += 1;
            }
        }
    }

    if nodes_with_community == 0 {
        tracing::info!(
            "Stage 7 (COACT): no nodes with community_id/centrality populated \
             — stages 4/5 likely did not run (--no-cognitive?). No edges added."
        );
        stats.coact_edges_added = 0;
        return Ok(());
    }

    // Emit COACT cliques over top-k centrality per community.
    let mut edges_added = 0u64;
    let mut communities_seeded = 0u64;
    for (_cid, mut members) in by_community.into_iter() {
        if members.len() < 2 {
            continue;
        }
        // Sort by centrality DESC, take top-k.
        members.sort_by(|a, b| b.1.cmp(&a.1));
        let top: Vec<obrain_common::NodeId> = members
            .into_iter()
            .take(COACT_TOP_K)
            .map(|(n, _)| n)
            .collect();
        if top.len() < 2 {
            continue;
        }
        communities_seeded += 1;
        // All pairs (i, j) with i < j — undirected clique. Substrate
        // edges are directed, so we create both directions to keep the
        // runtime warden happy (it expects COACT reinforcement to
        // observe both endpoints).
        for i in 0..top.len() {
            for j in 0..top.len() {
                if i == j {
                    continue;
                }
                let props = vec![(
                    obrain_common::types::PropertyKey::new("weight"),
                    obrain_common::types::Value::Float64(COACT_IMPORT_WEIGHT as f64),
                )];
                let _ =
                    substrate.create_edge_with_props(top[i], top[j], COACT_EDGE_TYPE_NAME, &props);
                edges_added += 1;
            }
        }
    }

    substrate.flush().ok();
    stats.coact_edges_added = edges_added;
    tracing::info!(
        "Stage 7 (COACT): seeded {} edges across {} communities (top-{} × top-{} clique, weight={})",
        edges_added,
        communities_seeded,
        COACT_TOP_K,
        COACT_TOP_K,
        COACT_IMPORT_WEIGHT,
    );
    Ok(())
}

/// Stage 8 — seed engrams from community × centrality.
///
/// For each community whose size is at least
/// [`MIN_COMMUNITY_SIZE_FOR_ENGRAM`], pick the top
/// [`ENGRAM_TOP_K`] nodes by centrality and form a fresh engram via
/// [`SubstrateStore::seed_engrams_batch`]. The total number of
/// seeded engrams is capped at [`MAX_SEEDED_ENGRAMS`] so that the
/// per-node `engram_bitset` (64-bit Bloom signature) does not saturate
/// on very large imports — runtime-formed engrams need room to fit.
///
/// ## Dependencies
///
/// Needs `community_id` (stage 4) and `centrality_cached` (stage 5).
/// Falls back to zero seeded engrams when those are absent.
fn stage_seed_engrams(substrate: &Arc<SubstrateStore>, stats: &mut PipelineStats) -> Result<()> {
    let node_ids = GraphStore::all_node_ids(&**substrate);
    if node_ids.is_empty() {
        stats.engrams_seeded = 0;
        tracing::info!("Stage 8 (engrams): empty graph — no seeds");
        return Ok(());
    }

    // Group by community with centrality.
    let mut by_community: HashMap<u32, Vec<(obrain_common::NodeId, u16)>> = HashMap::new();
    let writer = substrate.writer();
    for nid in &node_ids {
        if let Ok(Some(rec)) = writer.read_node(nid.0 as u32) {
            // Include even centrality=0 nodes — small communities need
            // them, and the top-k sort handles the order regardless.
            if rec.community_id != 0 || rec.centrality_cached != 0 {
                by_community
                    .entry(rec.community_id)
                    .or_default()
                    .push((*nid, rec.centrality_cached));
            }
        }
    }

    if by_community.is_empty() {
        tracing::info!(
            "Stage 8 (engrams): no nodes with community_id populated \
             — stages 4/5 likely did not run. No engrams seeded."
        );
        stats.engrams_seeded = 0;
        return Ok(());
    }

    // Build the cluster list: top-k centrality per eligible community.
    // Sort communities by size DESC so that the first MAX_SEEDED_ENGRAMS
    // slots go to the most densely populated ones — those carry the
    // strongest semantic signal.
    let mut communities: Vec<(u32, Vec<(obrain_common::NodeId, u16)>)> =
        by_community.into_iter().collect();
    communities.sort_by(|a, b| b.1.len().cmp(&a.1.len()));

    let mut clusters: Vec<Vec<obrain_common::NodeId>> = Vec::new();
    for (_cid, mut members) in communities.into_iter() {
        if members.len() < MIN_COMMUNITY_SIZE_FOR_ENGRAM {
            continue;
        }
        members.sort_by(|a, b| b.1.cmp(&a.1));
        let cluster: Vec<obrain_common::NodeId> = members
            .into_iter()
            .take(ENGRAM_TOP_K)
            .map(|(n, _)| n)
            .collect();
        clusters.push(cluster);
        if clusters.len() >= MAX_SEEDED_ENGRAMS {
            break;
        }
    }

    if clusters.is_empty() {
        tracing::info!(
            "Stage 8 (engrams): no community ≥ {} members — nothing to seed",
            MIN_COMMUNITY_SIZE_FOR_ENGRAM
        );
        stats.engrams_seeded = 0;
        return Ok(());
    }

    let engram_ids = substrate
        .seed_engrams_batch(&clusters)
        .context("seed_engrams_batch")?;
    stats.engrams_seeded = engram_ids.len() as u64;
    tracing::info!(
        "Stage 8 (engrams): seeded {} engrams (top-{} centrality × community size ≥ {}, cap {})",
        engram_ids.len(),
        ENGRAM_TOP_K,
        MIN_COMMUNITY_SIZE_FOR_ENGRAM,
        MAX_SEEDED_ENGRAMS,
    );
    Ok(())
}

// ---------------------------------------------------------------------
// T17l Stage 9 — `_hilbert_features` (canonical 64-72d topology signature)
// ---------------------------------------------------------------------
//
// Delegates to `obrain-adapters::plugins::algorithms::hilbert_features`
// which is the single canonical producer of this key (same function the
// runtime `HilbertEnricher` thinker uses as a fallback in the hub fleet).
// The result map is persisted one-node-at-a-time via
// `set_node_property` — each write flows through the substrate WAL, so
// a partial run survives a crash and the next `--enrich-only` resumes
// from the coverage probe.

fn stage_fill_hilbert_features(
    substrate: &Arc<SubstrateStore>,
    stats: &mut PipelineStats,
) -> Result<()> {
    use obrain_adapters::plugins::algorithms::hilbert_features::{
        HilbertFeaturesConfig, hilbert_features,
    };
    use obrain_common::types::Value;
    use obrain_core::graph::{GraphStore, GraphStoreMut};

    let t0 = std::time::Instant::now();
    let node_count = substrate.node_count();
    if node_count == 0 {
        tracing::info!("Stage 9 (hilbert_features): empty graph — skipping");
        return Ok(());
    }
    tracing::info!(
        "Stage 9 (hilbert_features): computing on {} nodes (canonical 64d topology signature)",
        node_count
    );

    let config = HilbertFeaturesConfig::default();
    let store_ref: &dyn GraphStore = substrate.as_ref();
    let result = hilbert_features(store_ref, &config);

    let pb = progress_bar(result.features.len() as u64, "hilbert_features");
    let mut nodes_written = 0u64;
    for (nid, features) in result.features.iter() {
        let arc: std::sync::Arc<[f32]> = features.as_slice().into();
        substrate.set_node_property(*nid, "_hilbert_features", Value::Vector(arc));
        nodes_written += 1;
        pb.inc(1);
    }
    pb.finish_with_message("hilbert_features done");

    stats.hilbert_features_computed = nodes_written;
    tracing::info!(
        "Stage 9 (hilbert_features): {} nodes written, dim {}, elapsed {:.2}s",
        nodes_written,
        result.dimensions,
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

// ---------------------------------------------------------------------
// T17l Stage 10 — `_kernel_embedding` (canonical 80d Φ₀ projection)
// ---------------------------------------------------------------------
//
// Depends on `_hilbert_features` being present on every node's
// neighborhood (kernel reads Hilbert features from each neighbor). Stage
// 9 produces them ; if a user runs Stage 10 in isolation without
// Stage 9 having populated the features, `KernelManager::compute_all`
// returns a partial map (only nodes whose neighborhood has Hilbert
// features). The dependency ordering in `run()` prevents this from
// happening in practice.

fn stage_fill_kernel_embeddings(
    substrate: &Arc<SubstrateStore>,
    stats: &mut PipelineStats,
) -> Result<()> {
    use obrain_adapters::plugins::algorithms::kernel_manager::KernelManager;
    use obrain_core::graph::GraphStore;

    let t0 = std::time::Instant::now();
    let node_count = substrate.node_count();
    if node_count == 0 {
        tracing::info!("Stage 10 (kernel_embedding): empty graph — skipping");
        return Ok(());
    }
    tracing::info!(
        "Stage 10 (kernel_embedding): computing on {} nodes (canonical 80d Φ₀ projection, depends on Hilbert)",
        node_count
    );

    // `new_untrained` seeds MultiHeadPhi0 with a deterministic RNG — the
    // resulting projection is reproducible across runs on the same
    // graph topology. `compute_all` writes `_kernel_embedding` on every
    // node whose neighborhood has `_hilbert_features` populated.
    let store_mut: std::sync::Arc<dyn obrain_core::graph::GraphStoreMut> = substrate.clone();
    let seed: u64 = 42;
    let mgr = KernelManager::new_untrained(store_mut, seed);
    mgr.compute_all();
    let count = mgr.embedding_count() as u64;

    stats.kernel_embeddings_computed = count;
    tracing::info!(
        "Stage 10 (kernel_embedding): {} nodes written, elapsed {:.2}s",
        count,
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------

fn open_or_create_substrate(path: &std::path::Path) -> Result<Arc<SubstrateStore>> {
    if path.exists() {
        Ok(Arc::new(
            SubstrateStore::open(path).with_context(|| format!("open {}", path.display()))?,
        ))
    } else {
        Ok(Arc::new(
            SubstrateStore::create(path).with_context(|| format!("create {}", path.display()))?,
        ))
    }
}

#[cfg_attr(not(feature = "embed"), allow(dead_code))]
fn progress_bar(total: u64, label: &str) -> ProgressBar {
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] \
             {pos:>7}/{len:7} {msg}",
        )
        .unwrap_or_else(|_| ProgressStyle::default_bar())
        .progress_chars("=>-"),
    );
    pb.set_message(format!("neo4j2obrain:{label}"));
    pb
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// Build a `PipelineOptions` with sensible test defaults. The
    /// single place tests specify overrides; keeps new `PipelineOptions`
    /// fields from ripping through every test.
    fn test_opts(out: PathBuf, run_cognitive: bool) -> PipelineOptions {
        PipelineOptions {
            neo4j_url: "bolt://example".into(),
            neo4j_user: "test".into(),
            neo4j_password: "test".into(),
            output: out,
            run_cognitive,
            batch_size: 100,
            embeddings_model: "stub".into(),
            // Tests always use the sentinel, so the paths are never
            // read. Placeholder values keep the struct total.
            st_model_path: PathBuf::from("/dev/null"),
            st_vocab_path: PathBuf::from("/dev/null"),
            text_keys: DEFAULT_TEXT_KEYS.iter().map(|s| s.to_string()).collect(),
            min_text_len: 10,
            enrich_only: false,
            force_rerun_cognitive: false,
        }
    }

    #[test]
    fn pipeline_runs_on_empty_substrate() {
        // The stages must be no-ops on an empty store and return
        // default-valued stats (nodes/edges = 0).
        let td = TempDir::new().unwrap();
        let out = td.path().join("n2o.obrain");
        let opts = test_opts(out, false);
        let stats = run(&opts).unwrap();
        assert_eq!(stats.nodes_read, 0);
        assert_eq!(stats.edges_read, 0);
        assert_eq!(stats.tier_entries, 0);
    }

    #[test]
    fn pipeline_runs_with_cognitive_on_empty_substrate() {
        let td = TempDir::new().unwrap();
        let out = td.path().join("n2o-c.obrain");
        let opts = test_opts(out, true);
        // Cognitive stages must degrade gracefully on an empty graph
        // (no communities, no centrality, no ricci to refresh).
        let stats = run(&opts).unwrap();
        assert_eq!(stats.nodes_read, 0);
    }

    /// Stages 7 (COACT) and 8 (engrams) on a populated graph.
    ///
    /// Build the canonical two-triangle graph (same topology the
    /// cognitive-init test uses), drop a pre-existing substrate at a
    /// temp path, then call `run()` with `run_cognitive = true`. After
    /// the full pipeline:
    ///   * `stats.communities == 2` (LDleiden finds both triangles),
    ///   * `stats.coact_edges_added > 0` (stage 7 seeds at least one
    ///     COACT clique — two triangles of 3 nodes each = 2 communities
    ///     × 3*2 directed pairs = 12 edges),
    ///   * `stats.engrams_seeded == 2` (two communities of size 3).
    #[test]
    fn pipeline_seeds_coact_and_engrams_on_two_triangles() {
        use obrain_core::graph::traits::GraphStoreMut;

        let td = TempDir::new().unwrap();
        let out = td.path().join("n2o-populated.obrain");

        // Pre-populate the substrate at the output path before calling run().
        {
            let sub = Arc::new(SubstrateStore::create(&out).unwrap());
            let a = sub.create_node_with_props(&["N"], &[]);
            let b = sub.create_node_with_props(&["N"], &[]);
            let c = sub.create_node_with_props(&["N"], &[]);
            sub.create_edge_with_props(a, b, "E", &[]);
            sub.create_edge_with_props(b, c, "E", &[]);
            sub.create_edge_with_props(c, a, "E", &[]);

            let d = sub.create_node_with_props(&["N"], &[]);
            let e = sub.create_node_with_props(&["N"], &[]);
            let f = sub.create_node_with_props(&["N"], &[]);
            sub.create_edge_with_props(d, e, "E", &[]);
            sub.create_edge_with_props(e, f, "E", &[]);
            sub.create_edge_with_props(f, d, "E", &[]);

            sub.create_edge_with_props(a, d, "BRIDGE", &[]);
            sub.flush().unwrap();
        }

        let opts = test_opts(out, true);
        let stats = run(&opts).unwrap();

        assert_eq!(
            stats.communities, 2,
            "expected 2 communities on two-triangle graph, got {}",
            stats.communities
        );
        assert!(
            stats.coact_edges_added > 0,
            "expected > 0 COACT edges on two populated communities, got {}",
            stats.coact_edges_added
        );
        assert_eq!(
            stats.engrams_seeded, 2,
            "expected exactly 2 engrams (one per triangle community), got {}",
            stats.engrams_seeded
        );
    }

    /// The "stub" / "none" / "" sentinel disables stage 2 even when the
    /// `embed` feature is compiled in. This keeps CI hermetic: no
    /// network access, no ORT session instantiation.
    #[test]
    fn stage2_embeddings_sentinel_is_noop() {
        let td = TempDir::new().unwrap();
        let out = td.path().join("n2o-stub.obrain");
        for sentinel in ["stub", "none", ""] {
            let mut opts = test_opts(out.clone(), false);
            opts.embeddings_model = sentinel.into();
            let stats = run(&opts).unwrap();
            assert_eq!(
                stats.embeddings_computed, 0,
                "sentinel '{}' must leave embeddings_computed = 0",
                sentinel
            );
        }
    }
}
