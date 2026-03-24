//! Knowledge Fabric — auto-computed metrics for graph nodes.
//!
//! Each node in the graph has a [`FabricScore`] that tracks:
//! - **churn_score**: frequency of mutations (how often the node changes)
//! - **knowledge_density**: number of notes/decisions linked (richness of context)
//! - **staleness**: time since last mutation (freshness)
//! - **risk_score**: composite metric combining pagerank × churn × knowledge_gap × betweenness
//! - **pagerank**: link-structure importance (from GDS refresh)
//! - **betweenness**: path-involvement centrality (from GDS refresh)
//! - **community_id**: Louvain community assignment (from GDS refresh)
//!
//! The [`FabricListener`] implements [`MutationListener`] and incrementally
//! updates churn and staleness on each mutation batch. Global graph metrics
//! (pagerank, betweenness, Louvain) are refreshed in batch by the
//! [`GdsRefreshScheduler`](super::gds_refresh::GdsRefreshScheduler).

use async_trait::async_trait;
use dashmap::DashMap;
use grafeo_common::types::NodeId;
use grafeo_reactive::{MutationEvent, MutationListener};
use smallvec::SmallVec;
use std::sync::Arc;
use std::time::Instant;

use crate::store_trait::{
    OptionalGraphStore, PROP_FABRIC_CHURN, PROP_FABRIC_DENSITY, PROP_FABRIC_RISK, persist_node_f64,
};

// ---------------------------------------------------------------------------
// FabricScore
// ---------------------------------------------------------------------------

/// Aggregated knowledge-fabric metrics for a single node.
///
/// All scores default to `0.0` (or `None` for community_id).
#[derive(Debug, Clone, PartialEq)]
pub struct FabricScore {
    /// Frequency of mutations — incremented each time the node is mutated.
    pub churn_score: f64,
    /// Richness of linked context (notes, decisions, etc.). Range [0.0, 1.0].
    pub knowledge_density: f64,
    /// Seconds since last mutation. `0.0` means "just mutated".
    pub staleness: f64,
    /// Composite risk: weighted sum of normalized pagerank, churn, knowledge_gap, betweenness, scar. Range [0.0, 1.0].
    pub risk_score: f64,
    /// Link-structure importance (PageRank). Set by GDS refresh.
    pub pagerank: f64,
    /// Path-involvement centrality. Set by GDS refresh.
    pub betweenness: f64,
    /// Cumulative scar intensity on this node. Set by scar system integration.
    /// Higher values indicate more past problems (rollbacks, errors, etc.).
    pub scar_intensity: f64,
    /// Louvain community assignment. Set by GDS refresh.
    pub community_id: Option<u64>,
    /// Timestamp of last mutation for staleness computation.
    last_mutated: Option<Instant>,
}

impl Default for FabricScore {
    fn default() -> Self {
        Self {
            churn_score: 0.0,
            knowledge_density: 0.0,
            staleness: 0.0,
            risk_score: 0.0,
            pagerank: 0.0,
            betweenness: 0.0,
            scar_intensity: 0.0,
            community_id: None,
            last_mutated: None,
        }
    }
}

impl FabricScore {
    /// Creates a new `FabricScore` with all metrics at zero.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the instant of the last mutation, if any.
    pub fn last_mutated(&self) -> Option<Instant> {
        self.last_mutated
    }
}

// ---------------------------------------------------------------------------
// FabricStore
// ---------------------------------------------------------------------------

/// Thread-safe store for per-node [`FabricScore`]s.
///
/// Uses [`DashMap`] for concurrent read/write without global locks.
/// When a backing [`GraphStoreMut`](grafeo_core::graph::GraphStoreMut) is
/// provided, risk_score, churn_score, and knowledge_density are persisted
/// as node properties prefixed with `_cog_` (write-through).
pub struct FabricStore {
    /// Per-node fabric scores.
    scores: DashMap<NodeId, FabricScore>,
    /// Optional backing graph store for write-through persistence.
    graph_store: OptionalGraphStore,
}

impl FabricStore {
    /// Creates a new, empty fabric store (in-memory only).
    pub fn new() -> Self {
        Self {
            scores: DashMap::new(),
            graph_store: None,
        }
    }

    /// Creates a new fabric store with write-through persistence.
    pub fn with_graph_store(graph_store: Arc<dyn grafeo_core::graph::GraphStoreMut>) -> Self {
        Self {
            scores: DashMap::new(),
            graph_store: Some(graph_store),
        }
    }

    /// Persists the key fabric metrics for a node to the graph store.
    fn persist_fabric(&self, node_id: NodeId) {
        if let Some(gs) = &self.graph_store {
            if let Some(entry) = self.scores.get(&node_id) {
                persist_node_f64(gs.as_ref(), node_id, PROP_FABRIC_RISK, entry.risk_score);
                persist_node_f64(gs.as_ref(), node_id, PROP_FABRIC_CHURN, entry.churn_score);
                persist_node_f64(
                    gs.as_ref(),
                    node_id,
                    PROP_FABRIC_DENSITY,
                    entry.knowledge_density,
                );
            }
        }
    }

    /// Returns the fabric score for a node, computing staleness on the fly.
    ///
    /// Returns a default (all-zero) score if the node has never been tracked.
    pub fn get_fabric_score(&self, node_id: NodeId) -> FabricScore {
        match self.scores.get(&node_id) {
            Some(entry) => {
                let mut score = entry.clone();
                // Recompute staleness from last_mutated
                if let Some(last) = score.last_mutated {
                    score.staleness = last.elapsed().as_secs_f64();
                }
                score
            }
            None => FabricScore::default(),
        }
    }

    /// Increments the churn score for a node and resets its staleness.
    ///
    /// If the node is not yet tracked, it is created with churn=1.
    pub fn update_churn(&self, node_id: NodeId) {
        self.update_churn_at(node_id, Instant::now());
    }

    /// Increments churn at a specific instant (for testing).
    pub fn update_churn_at(&self, node_id: NodeId, now: Instant) {
        self.scores
            .entry(node_id)
            .and_modify(|s| {
                s.churn_score += 1.0;
                s.staleness = 0.0;
                s.last_mutated = Some(now);
            })
            .or_insert_with(|| FabricScore {
                churn_score: 1.0,
                staleness: 0.0,
                last_mutated: Some(now),
                ..FabricScore::default()
            });
        self.persist_fabric(node_id);
    }

    /// Updates the staleness for a node based on elapsed time since last mutation.
    pub fn update_staleness(&self, node_id: NodeId) {
        if let Some(mut entry) = self.scores.get_mut(&node_id)
            && let Some(last) = entry.last_mutated
        {
            entry.staleness = last.elapsed().as_secs_f64();
        }
    }

    /// Sets the knowledge density for a node.
    pub fn set_knowledge_density(&self, node_id: NodeId, density: f64) {
        self.scores
            .entry(node_id)
            .and_modify(|s| s.knowledge_density = density)
            .or_insert_with(|| FabricScore {
                knowledge_density: density,
                ..FabricScore::default()
            });
        self.persist_fabric(node_id);
    }

    /// Sets the cumulative scar intensity for a node.
    ///
    /// This value is typically provided by the scar store's cumulative intensity method.
    pub fn set_scar_intensity(&self, node_id: NodeId, scar_intensity: f64) {
        self.scores
            .entry(node_id)
            .and_modify(|s| s.scar_intensity = scar_intensity)
            .or_insert_with(|| FabricScore {
                scar_intensity,
                ..FabricScore::default()
            });
        self.persist_fabric(node_id);
    }

    /// Sets GDS-computed metrics (pagerank, betweenness, community_id) for a node.
    pub fn set_gds_metrics(
        &self,
        node_id: NodeId,
        pagerank: f64,
        betweenness: f64,
        community_id: Option<u64>,
    ) {
        self.scores
            .entry(node_id)
            .and_modify(|s| {
                s.pagerank = pagerank;
                s.betweenness = betweenness;
                s.community_id = community_id;
            })
            .or_insert_with(|| FabricScore {
                pagerank,
                betweenness,
                community_id,
                ..FabricScore::default()
            });
        self.persist_fabric(node_id);
    }

    /// Returns the top-N hotspots sorted by churn score descending.
    pub fn get_hotspots(&self, top_n: usize) -> Vec<(NodeId, FabricScore)> {
        let mut entries: Vec<(NodeId, FabricScore)> = self
            .scores
            .iter()
            .map(|entry| {
                let mut score = entry.value().clone();
                if let Some(last) = score.last_mutated {
                    score.staleness = last.elapsed().as_secs_f64();
                }
                (*entry.key(), score)
            })
            .collect();
        entries.sort_by(|a, b| {
            b.1.churn_score
                .partial_cmp(&a.1.churn_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        entries.truncate(top_n);
        entries
    }

    /// Returns all nodes whose risk_score is ≥ `min_risk`.
    pub fn get_risk_zones(&self, min_risk: f64) -> Vec<(NodeId, FabricScore)> {
        self.scores
            .iter()
            .filter(|entry| entry.value().risk_score >= min_risk)
            .map(|entry| {
                let mut score = entry.value().clone();
                if let Some(last) = score.last_mutated {
                    score.staleness = last.elapsed().as_secs_f64();
                }
                (*entry.key(), score)
            })
            .collect()
    }

    /// Recalculates the composite risk score for a specific node.
    ///
    /// The `max_*` parameters are the current global maximums used for normalization.
    pub fn recalculate_risk(
        &self,
        node_id: NodeId,
        max_pagerank: f64,
        max_churn: f64,
        max_betweenness: f64,
    ) {
        let max_scar = self.max_scar_intensity();
        if let Some(mut entry) = self.scores.get_mut(&node_id) {
            entry.risk_score =
                compute_risk_score(&entry, max_pagerank, max_churn, max_betweenness, max_scar);
        }
    }

    /// Recalculates risk scores for ALL tracked nodes.
    pub fn recalculate_all_risks(&self) {
        let (max_pagerank, max_churn, max_betweenness) = self.global_maxima();
        let max_scar = self.max_scar_intensity();

        for mut entry in self.scores.iter_mut() {
            entry.risk_score =
                compute_risk_score(&entry, max_pagerank, max_churn, max_betweenness, max_scar);
        }
    }

    /// Returns the number of tracked nodes.
    pub fn len(&self) -> usize {
        self.scores.len()
    }

    /// Returns `true` if no nodes are tracked.
    pub fn is_empty(&self) -> bool {
        self.scores.is_empty()
    }

    /// Computes the global maximum values for normalization.
    fn global_maxima(&self) -> (f64, f64, f64) {
        let mut max_pr = 0.0_f64;
        let mut max_churn = 0.0_f64;
        let mut max_btwn = 0.0_f64;
        for entry in &self.scores {
            max_pr = max_pr.max(entry.pagerank);
            max_churn = max_churn.max(entry.churn_score);
            max_btwn = max_btwn.max(entry.betweenness);
        }
        (max_pr, max_churn, max_btwn)
    }

    /// Returns the maximum scar intensity across all tracked nodes.
    fn max_scar_intensity(&self) -> f64 {
        self.scores
            .iter()
            .map(|e| e.scar_intensity)
            .fold(0.0_f64, f64::max)
    }

    /// Returns all distinct community IDs assigned to tracked nodes.
    pub fn community_ids(&self) -> Vec<u64> {
        let mut ids: Vec<u64> = self
            .scores
            .iter()
            .filter_map(|entry| entry.community_id)
            .collect();
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    /// Returns all node IDs assigned to a specific community.
    pub fn get_community_nodes(&self, community_id: u64) -> Vec<NodeId> {
        self.scores
            .iter()
            .filter(|entry| entry.community_id == Some(community_id))
            .map(|entry| *entry.key())
            .collect()
    }
}

impl Default for FabricStore {
    fn default() -> Self {
        Self {
            scores: DashMap::new(),
            graph_store: None,
        }
    }
}

impl std::fmt::Debug for FabricStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FabricStore")
            .field("tracked_nodes", &self.scores.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Risk score computation
// ---------------------------------------------------------------------------

/// Normalizes a value to [0.0, 1.0] given a maximum.
/// Returns 0.0 if max is 0 (avoids division by zero).
#[inline]
fn normalize(value: f64, max: f64) -> f64 {
    if max <= 0.0 {
        0.0
    } else {
        (value / max).clamp(0.0, 1.0)
    }
}

/// Computes the composite risk score for a node.
///
/// `risk = base_risk + scar_boost`
///
/// Where:
/// - `base_risk = normalize(pagerank) × normalize(churn) × (1 - knowledge_density) × normalize(betweenness)`
/// - `scar_boost = normalize(scar_intensity)` (additive — scars always increase risk)
///
/// The final value is clamped to [0.0, 1.0].
///
/// A node is high-risk when it is:
/// - Important (high pagerank)
/// - Volatile (high churn)
/// - Poorly documented (low knowledge_density)
/// - Central in paths (high betweenness)
/// - Scarred by past errors (high scar_intensity)
fn compute_risk_score(
    score: &FabricScore,
    max_pagerank: f64,
    max_churn: f64,
    max_betweenness: f64,
    max_scar_intensity: f64,
) -> f64 {
    let pr = normalize(score.pagerank, max_pagerank);
    let churn = normalize(score.churn_score, max_churn);
    let knowledge_gap = 1.0 - score.knowledge_density.clamp(0.0, 1.0);
    let btwn = normalize(score.betweenness, max_betweenness);
    let scar = normalize(score.scar_intensity, max_scar_intensity);

    // Weighted additive formula — avoids the multiplicative collapse to zero
    // that the old `pr * churn * knowledge_gap * btwn` formula suffered from.
    //
    // Weights sum to 1.0 so the result is inherently in [0, 1]:
    //   pagerank:       25% — importance
    //   churn:          25% — volatility
    //   knowledge_gap:  20% — missing documentation
    //   betweenness:    15% — centrality
    //   scar:           15% — past failures
    let risk = pr * 0.25 + churn * 0.25 + knowledge_gap * 0.20 + btwn * 0.15 + scar * 0.15;

    risk.clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// FabricListener — MutationListener
// ---------------------------------------------------------------------------

/// A [`MutationListener`] that incrementally updates fabric metrics
/// on each mutation batch.
///
/// On each batch:
/// 1. Increments churn_score for all mutated nodes
/// 2. Resets staleness to 0 for mutated nodes
/// 3. Recalculates risk_score for affected nodes
pub struct FabricListener {
    /// Shared fabric store.
    store: Arc<FabricStore>,
}

impl FabricListener {
    /// Creates a new fabric listener backed by the given store.
    pub fn new(store: Arc<FabricStore>) -> Self {
        Self { store }
    }

    /// Returns a reference to the underlying store.
    pub fn store(&self) -> &Arc<FabricStore> {
        &self.store
    }

    /// Extracts all node IDs affected by a mutation event.
    fn affected_nodes(event: &MutationEvent) -> SmallVec<[NodeId; 2]> {
        use smallvec::smallvec;
        match event {
            MutationEvent::NodeCreated { node } => smallvec![node.id],
            MutationEvent::NodeUpdated { after, .. } => smallvec![after.id],
            MutationEvent::NodeDeleted { node } => smallvec![node.id],
            MutationEvent::EdgeCreated { edge } => smallvec![edge.src, edge.dst],
            MutationEvent::EdgeUpdated { after, .. } => smallvec![after.src, after.dst],
            MutationEvent::EdgeDeleted { edge } => smallvec![edge.src, edge.dst],
        }
    }
}

#[async_trait]
impl MutationListener for FabricListener {
    fn name(&self) -> &str {
        "cognitive:fabric"
    }

    async fn on_event(&self, event: &MutationEvent) {
        let now = Instant::now();
        for node_id in Self::affected_nodes(event) {
            self.store.update_churn_at(node_id, now);
        }
    }

    async fn on_batch(&self, events: &[MutationEvent]) {
        let now = Instant::now();
        // Collect all unique affected nodes
        let mut touched = std::collections::HashSet::new();
        for event in events {
            for node_id in Self::affected_nodes(event) {
                if touched.insert(node_id) {
                    self.store.update_churn_at(node_id, now);
                }
            }
        }

        // Recalculate risk for all affected nodes
        let (max_pr, max_churn, max_btwn) = self.store.global_maxima();
        for node_id in &touched {
            self.store
                .recalculate_risk(*node_id, max_pr, max_churn, max_btwn);
        }
    }
}

impl std::fmt::Debug for FabricListener {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FabricListener")
            .field("store", &self.store)
            .finish()
    }
}
