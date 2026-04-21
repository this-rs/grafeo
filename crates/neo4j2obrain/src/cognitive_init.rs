//! # Cognitive initialisation batches (stages 4-6 of the pipeline)
//!
//! Each batch runs over a freshly written [`SubstrateStore`] and fills
//! a cognitive column (`community_id`, `centrality_cached`, `edge.ricci`,
//! `node.curvature`). The goal is "cognitively alive on open" — nothing
//! lazy, no cold-start latency on the first query after import.
//!
//! ## Implementation status (plan T15 → T17)
//!
//! | Stage | Status                                                        |
//! |------:|----------------------------------------------------------------|
//! |     4 | Structural stub — LDleiden batch API not yet public (T17).     |
//! |     5 | Structural stub — PageRank batch API not yet public (T17).     |
//! |     6 | Real — wraps [`obrain_substrate::refresh_all_ricci`] and       |
//! |       | [`obrain_substrate::compute_all_node_curvatures`].             |
//!
//! Stubs emit an INFO trace and set their stats counter to zero; they
//! do not fail. This keeps the end-to-end pipeline callable today while
//! the batch surfaces in `obrain-cognitive` are being exposed.

use std::sync::Arc;

use anyhow::{Context, Result};
use obrain_substrate::SubstrateStore;

use crate::pipeline::PipelineStats;

/// Stage 4 — LDleiden community detection batch.
///
/// **Stub**: the Leiden implementation currently lives inside
/// `obrain-cognitive` and is driven per-query rather than as a public
/// batch entry point. Exposing `ldleiden::run_batch(store)` is tracked
/// in the T17 follow-up (cut-over plan).
///
/// When the batch API lands, the implementation replaces the `info!`
/// below by:
///
/// ```ignore
/// let run = obrain_cognitive::ldleiden::run_batch(substrate)?;
/// stats.communities = run.num_communities as u64;
/// ```
pub fn run_ldleiden_batch(
    _substrate: &Arc<SubstrateStore>,
    stats: &mut PipelineStats,
) -> Result<()> {
    tracing::info!(
        "Stage 4 (LDleiden): structural placeholder — public batch API \
         lands with T17 cut-over. Nodes will retain whatever `community_id` \
         they already carry from Neo4j."
    );
    stats.communities = 0;
    Ok(())
}

/// Stage 5 — PageRank batch.
///
/// **Stub**: same status as stage 4 — the runtime PageRank lives in
/// `obrain-cognitive` and is not yet exposed as a public batch. When it
/// is, this becomes:
///
/// ```ignore
/// let run = obrain_cognitive::pagerank::run_batch(substrate, ..)?;
/// stats.pagerank_nonzero = run.nonzero_count as u64;
/// ```
pub fn run_pagerank_batch(
    _substrate: &Arc<SubstrateStore>,
    stats: &mut PipelineStats,
) -> Result<()> {
    tracing::info!(
        "Stage 5 (PageRank): structural placeholder — public batch API \
         lands with T17 cut-over. `centrality_cached` will be populated \
         lazily on first query."
    );
    stats.pagerank_nonzero = 0;
    Ok(())
}

/// Stage 6 — Ricci-Ollivier batch + node curvature aggregation.
///
/// Real implementation: calls into [`obrain_substrate`]'s public batch
/// surfaces which were exposed in T12 (geometric inference).
///
/// * [`obrain_substrate::refresh_all_ricci`] — walks every live edge,
///   computes Ollivier-Ricci, writes the quantised `ricci_u8` back via
///   the WAL-logged `update_edge` path, clears `RICCI_STALE`.
/// * [`obrain_substrate::compute_all_node_curvatures`] — aggregates the
///   per-edge Ricci into a node curvature property (degree-weighted
///   mean) for nodes with at least one live incident edge.
///
/// The node-curvature map is currently discarded because the substrate
/// column write-back path lives inside obrain-cognitive; stage 6 writes
/// the edge values (authoritative) and leaves node curvature propagation
/// to the first cognitive query. This matches the runtime warden tick
/// behaviour.
pub fn run_ricci_batch(
    substrate: &Arc<SubstrateStore>,
    stats: &mut PipelineStats,
) -> Result<()> {
    // ---- Edge-level Ricci (authoritative, WAL-logged) ------------------
    let edge_stats = obrain_substrate::refresh_all_ricci(substrate)
        .context("refresh_all_ricci")?;

    // The substrate reports: edges_visited, edges_updated,
    // edges_skipped_dangling (+ curvature summary stats). For pipeline
    // metrics we count "refreshed" as edges_updated (the ones whose
    // persisted ricci_u8 actually changed).
    stats.ricci_refreshed = edge_stats.edges_updated;
    tracing::info!(
        "Stage 6 (Ricci): visited={}, updated={}, skipped_dangling={}, \
         |κ|_max={:.3}, frac(κ<0)={:.3}, frac(κ>0)={:.3}",
        edge_stats.edges_visited,
        edge_stats.edges_updated,
        edge_stats.edges_skipped_dangling,
        edge_stats.max_abs_curvature,
        edge_stats.negative_fraction,
        edge_stats.positive_fraction,
    );

    // ---- Node-level curvature aggregation ------------------------------
    let node_curv = obrain_substrate::compute_all_node_curvatures(substrate)
        .context("compute_all_node_curvatures")?;
    tracing::info!(
        "Stage 6 (Ricci): aggregated node curvature for {} live nodes",
        node_curv.len(),
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn empty_substrate() -> Arc<SubstrateStore> {
        let td = TempDir::new().unwrap();
        let path = td.path().join("cog.obrain");
        let store = SubstrateStore::create(&path).unwrap();
        // keep td alive by leaking — test lifetime is tiny
        std::mem::forget(td);
        Arc::new(store)
    }

    #[test]
    fn ldleiden_stub_is_idempotent_no_op() {
        let sub = empty_substrate();
        let mut stats = PipelineStats::default();
        run_ldleiden_batch(&sub, &mut stats).unwrap();
        run_ldleiden_batch(&sub, &mut stats).unwrap();
        assert_eq!(stats.communities, 0);
    }

    #[test]
    fn pagerank_stub_is_idempotent_no_op() {
        let sub = empty_substrate();
        let mut stats = PipelineStats::default();
        run_pagerank_batch(&sub, &mut stats).unwrap();
        run_pagerank_batch(&sub, &mut stats).unwrap();
        assert_eq!(stats.pagerank_nonzero, 0);
    }

    #[test]
    fn ricci_batch_on_empty_substrate_is_zero() {
        let sub = empty_substrate();
        let mut stats = PipelineStats::default();
        run_ricci_batch(&sub, &mut stats).unwrap();
        assert_eq!(stats.ricci_refreshed, 0);
    }
}
