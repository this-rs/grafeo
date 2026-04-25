//! # Cognitive initialisation batches (stages 4-6 of the pipeline)
//!
//! Each batch runs over a freshly written [`SubstrateStore`] and fills
//! a cognitive column (`community_id`, `centrality_cached`, `edge.ricci`,
//! `node.curvature`). The goal is "cognitively alive on open" — nothing
//! lazy, no cold-start latency on the first query after import.
//!
//! ## Implementation status (plan T15)
//!
//! | Stage | Status                                                        |
//! |------:|----------------------------------------------------------------|
//! |     4 | Real — wraps [`obrain_cognitive::community::leiden_batch`]      |
//! |       | and persists via [`Writer::update_community_batch`].            |
//! |     5 | Real — wraps [`obrain_adapters::plugins::algorithms::pagerank`] |
//! |       | and persists via [`SubstrateStore::update_centrality_batch_f32`]. |
//! |     6 | Real — wraps [`obrain_substrate::refresh_all_ricci`] and        |
//! |       | [`obrain_substrate::compute_all_node_curvatures`].              |
//!
//! All three stages degrade gracefully on an empty graph (zero counts,
//! no errors).

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use obrain_adapters::plugins::algorithms::pagerank;
use obrain_cognitive::community::{Graph, LeidenConfig, leiden_batch};
use obrain_common::NodeId;
use obrain_core::graph::Direction;
use obrain_core::graph::traits::GraphStore;
use obrain_substrate::SubstrateStore;

use crate::pipeline::PipelineStats;

/// Damping factor used by the PageRank batch (industry-standard 0.85).
const PAGERANK_DAMPING: f64 = 0.85;
/// Upper bound on power-iteration steps. 50 is more than enough for the
/// graph sizes targeted by obrain imports (≤ 10⁷ nodes typically
/// converges within 30 iterations at tol = 1e-6).
const PAGERANK_MAX_ITER: usize = 50;
/// Convergence tolerance (L1 norm of the PageRank vector delta).
const PAGERANK_TOL: f64 = 1.0e-6;

/// Stage 4 — LDleiden community detection batch.
///
/// Walks the substrate, builds a dense `(u32, u32, f64)` edge list, runs
/// [`leiden_batch`] with [`LeidenConfig::default`], then persists the
/// resulting partition back into substrate via
/// [`Writer::update_community_batch`] (WAL-logged, crash-safe).
///
/// ## Dense indexing
///
/// The substrate's `NodeId` is a 64-bit slot index that can be sparse
/// (tombstoned / padding slots). The leiden API needs a contiguous
/// `0..n` indexing, so we build two maps:
/// * `dense_of: NodeId -> u32` (substrate slot → dense idx)
/// * `slot_of:  u32 -> u32`    (dense idx → substrate slot, used to
///   write back the partition with the correct slot index)
///
/// ## Edge weights
///
/// Edges are currently fed to the leiden solver with unit weight. The
/// substrate does carry a `weight_u16` per edge, but exposing it via
/// `GraphStore` requires either a specialised path (substrate-aware
/// bypass of the trait) or a `weight` property lookup, both of which
/// are tracked separately. Unit weights give a well-posed Newman
/// modularity and keep the batch deterministic.
pub fn run_ldleiden_batch(
    substrate: &Arc<SubstrateStore>,
    stats: &mut PipelineStats,
) -> Result<()> {
    let store: &dyn GraphStore = &**substrate;

    // ---- Dense re-indexing --------------------------------------------------
    let node_ids = store.all_node_ids();
    let n = node_ids.len();
    if n == 0 {
        tracing::info!("Stage 4 (LDleiden): empty graph — no communities to compute");
        stats.communities = 0;
        return Ok(());
    }
    if n > u32::MAX as usize {
        anyhow::bail!(
            "Stage 4 (LDleiden): {} nodes exceeds u32::MAX — leiden_batch \
             addresses nodes with u32 indices",
            n
        );
    }
    let mut dense_of: HashMap<NodeId, u32> = HashMap::with_capacity(n);
    let mut slot_of: Vec<u32> = Vec::with_capacity(n);
    for (dense_idx, nid) in node_ids.iter().enumerate() {
        dense_of.insert(*nid, dense_idx as u32);
        slot_of.push(nid.0 as u32);
    }

    // ---- Edge list (unit-weighted, undirected collapse) ---------------------
    let mut edges: Vec<(u32, u32, f64)> = Vec::new();
    let mut dangling = 0u64;
    for src in &node_ids {
        let Some(&src_dense) = dense_of.get(src) else {
            continue;
        };
        for (dst, _eid) in store.edges_from(*src, Direction::Outgoing) {
            let Some(&dst_dense) = dense_of.get(&dst) else {
                dangling += 1;
                continue;
            };
            edges.push((src_dense, dst_dense, 1.0));
        }
    }
    let edge_count = edges.len();
    let graph = Graph::from_edges(n as u32, edges);

    // ---- Solve --------------------------------------------------------------
    let config = LeidenConfig::default();
    let partition = leiden_batch(&graph, config);

    // ---- Persist ------------------------------------------------------------
    // `update_community_batch` takes `(node_slot_idx, community_id)`. The
    // partition is indexed by dense idx, so we zip with `slot_of`.
    let updates: Vec<(u32, u32)> = partition
        .iter()
        .enumerate()
        .map(|(dense_idx, community_id)| (slot_of[dense_idx], *community_id))
        .collect();
    substrate
        .writer()
        .update_community_batch(updates)
        .context("update_community_batch")?;

    // ---- Stats --------------------------------------------------------------
    // Partition communities are 0-based dense ids (no gaps) per leiden
    // contract — num_communities = max + 1.
    let num_communities = partition.iter().copied().max().map(|m| m + 1).unwrap_or(0);
    stats.communities = num_communities as u64;
    tracing::info!(
        "Stage 4 (LDleiden): n={}, edges={}, dangling={}, communities={}",
        n,
        edge_count,
        dangling,
        num_communities,
    );
    Ok(())
}

/// Stage 5 — PageRank batch.
///
/// Wraps [`obrain_adapters::plugins::algorithms::pagerank`] (damping =
/// [`PAGERANK_DAMPING`], max_iter = [`PAGERANK_MAX_ITER`], tol =
/// [`PAGERANK_TOL`]) and persists the result via
/// [`SubstrateStore::update_centrality_batch_f32`] which quantises to
/// the `centrality_cached` u16 column (Q0.16) under a WAL record.
///
/// The `update_centrality_batch_f32` path also clears
/// `CENTRALITY_STALE`, so after this stage every node's cached
/// centrality matches the graph state at import time.
pub fn run_pagerank_batch(
    substrate: &Arc<SubstrateStore>,
    stats: &mut PipelineStats,
) -> Result<()> {
    let store: &dyn GraphStore = &**substrate;
    if store.node_count() == 0 {
        tracing::info!("Stage 5 (PageRank): empty graph — no scores to compute");
        stats.pagerank_nonzero = 0;
        return Ok(());
    }

    let scores = pagerank(store, PAGERANK_DAMPING, PAGERANK_MAX_ITER, PAGERANK_TOL);

    // Filter: only persist non-zero scores. Zero is the default column
    // value — writing zero is a no-op on a fresh store but still emits
    // a WAL record, so skipping keeps the replay stream lean.
    let mut updates: Vec<(NodeId, f32)> = Vec::with_capacity(scores.len());
    let mut nonzero = 0u64;
    for (nid, score) in scores {
        if score > 0.0 {
            nonzero += 1;
            updates.push((nid, score as f32));
        }
    }
    let total = updates.len() as u64;
    substrate
        .update_centrality_batch_f32(updates)
        .context("update_centrality_batch_f32")?;

    stats.pagerank_nonzero = nonzero;
    tracing::info!(
        "Stage 5 (PageRank): damping={}, max_iter={}, tol={:.0e}, persisted={}, nonzero={}",
        PAGERANK_DAMPING,
        PAGERANK_MAX_ITER,
        PAGERANK_TOL,
        total,
        nonzero,
    );
    Ok(())
}

/// Stage 6 — Ricci-Ollivier batch + node curvature aggregation.
///
/// Real implementation: calls into [`obrain_substrate`]'s public batch
/// surfaces exposed in T12 (geometric inference).
///
/// * [`obrain_substrate::refresh_all_ricci`] — walks every live edge,
///   computes Ollivier-Ricci, writes the quantised `ricci_u8` back via
///   the WAL-logged `update_edge` path, clears `RICCI_STALE`.
/// * [`obrain_substrate::compute_all_node_curvatures`] — aggregates the
///   per-edge Ricci into a node curvature property (degree-weighted
///   mean) for nodes with at least one live incident edge.
pub fn run_ricci_batch(substrate: &Arc<SubstrateStore>, stats: &mut PipelineStats) -> Result<()> {
    // ---- Edge-level Ricci (authoritative, WAL-logged) ------------------
    let edge_stats = obrain_substrate::refresh_all_ricci(substrate).context("refresh_all_ricci")?;

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
    fn ldleiden_on_empty_substrate_is_zero() {
        let sub = empty_substrate();
        let mut stats = PipelineStats::default();
        run_ldleiden_batch(&sub, &mut stats).unwrap();
        assert_eq!(stats.communities, 0);
    }

    #[test]
    fn pagerank_on_empty_substrate_is_zero() {
        let sub = empty_substrate();
        let mut stats = PipelineStats::default();
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

    /// End-to-end: populate a small substrate with two weakly-connected
    /// triangles, run LDleiden + PageRank, verify that:
    ///   * community detection yields 2 communities,
    ///   * PageRank assigns non-zero scores to at least the bridge
    ///     nodes (in-degree > 0).
    #[test]
    fn ldleiden_and_pagerank_on_two_triangles() {
        use obrain_core::graph::traits::GraphStoreMut;

        let td = TempDir::new().unwrap();
        let path = td.path().join("cog.obrain");
        let sub = Arc::new(SubstrateStore::create(&path).unwrap());
        // keep td alive
        std::mem::forget(td);

        // Triangle A: a-b-c-a
        let a = sub.create_node_with_props(&["N"], &[]);
        let b = sub.create_node_with_props(&["N"], &[]);
        let c = sub.create_node_with_props(&["N"], &[]);
        sub.create_edge_with_props(a, b, "E", &[]);
        sub.create_edge_with_props(b, c, "E", &[]);
        sub.create_edge_with_props(c, a, "E", &[]);

        // Triangle B: d-e-f-d
        let d = sub.create_node_with_props(&["N"], &[]);
        let e = sub.create_node_with_props(&["N"], &[]);
        let f = sub.create_node_with_props(&["N"], &[]);
        sub.create_edge_with_props(d, e, "E", &[]);
        sub.create_edge_with_props(e, f, "E", &[]);
        sub.create_edge_with_props(f, d, "E", &[]);

        // Single weak bridge a-d
        sub.create_edge_with_props(a, d, "BRIDGE", &[]);

        sub.flush().unwrap();

        let mut stats = PipelineStats::default();
        run_ldleiden_batch(&sub, &mut stats).unwrap();
        // A well-defined modularity optimum on this graph has exactly
        // two communities (one per triangle); the single bridge edge
        // is not strong enough to merge them.
        assert_eq!(
            stats.communities, 2,
            "expected 2 communities on two-triangle graph, got {}",
            stats.communities
        );

        let mut stats2 = PipelineStats::default();
        run_pagerank_batch(&sub, &mut stats2).unwrap();
        // Every node has at least one in-edge on this graph → every
        // PageRank score is non-zero. We do NOT assert == 6 here
        // because the quantisation floor (1 / 65535) may drop the
        // smallest scores in the `persisted` count; `nonzero` counts
        // f64 values > 0 pre-quantisation, so it must be ≥ 6.
        assert!(
            stats2.pagerank_nonzero >= 6,
            "expected ≥ 6 non-zero PageRank scores, got {}",
            stats2.pagerank_nonzero
        );
    }
}
