//! Multi-signal search pipeline — combines vector similarity, energy, topology,
//! and synapse traversal into a single composite score bounded [0, 1].
//!
//! # Architecture
//!
//! The pipeline evaluates 4 independent signals for each candidate node:
//!
//! 1. **Vector similarity** — cosine/euclidean distance from the HNSW index
//! 2. **Energy-weighted** — recency via exponential decay (`energy_score`)
//! 3. **Topology scoring** — PageRank + betweenness from the fabric store
//! 4. **Synapse traversal** — graph expansion via spreading activation
//!
//! Each signal is normalized to [0, 1] independently, then combined via a
//! weighted sum with configurable weights (also summing to 1).
//!
//! # Reranker
//!
//! An optional [`Reranker`] trait allows post-processing by an external hook
//! (e.g. LLM reranking, custom business logic). The default implementation
//! is a no-op passthrough.
//!
//! # Usage
//!
//! ```ignore
//! CALL cognitive.search({
//!     query_embedding: [...],
//!     weights: {energy: 0.3, topology: 0.3, similarity: 0.3, synapse: 0.1},
//!     limit: 10
//! })
//! YIELD node_id, score, signal_energy, signal_topology, signal_similarity, signal_synapse
//! ```

use grafeo_common::types::NodeId;
use std::collections::HashMap;
#[allow(unused_imports)]
use std::sync::Arc;

#[cfg(feature = "energy")]
use crate::energy::{EnergyStore, energy_score};

#[cfg(feature = "synapse")]
use crate::activation::{SpreadConfig, SynapseActivationSource, spread};
#[cfg(feature = "synapse")]
use crate::synapse::SynapseStore;

#[cfg(feature = "fabric")]
use crate::fabric::FabricStore;

// ---------------------------------------------------------------------------
// Signal weights
// ---------------------------------------------------------------------------

/// Configurable weights for each signal in the composite score.
///
/// Weights are auto-normalized to sum to 1.0 when the pipeline executes.
/// Setting a weight to 0.0 excludes that signal from computation entirely.
#[derive(Debug, Clone)]
pub struct SearchWeights {
    /// Weight for vector similarity signal.
    pub similarity: f64,
    /// Weight for energy-weighted (recency) signal.
    pub energy: f64,
    /// Weight for topology (PageRank + betweenness) signal.
    pub topology: f64,
    /// Weight for synapse traversal (graph expansion) signal.
    pub synapse: f64,
}

impl Default for SearchWeights {
    fn default() -> Self {
        Self {
            similarity: 0.3,
            energy: 0.3,
            topology: 0.3,
            synapse: 0.1,
        }
    }
}

impl SearchWeights {
    /// Creates new weights. Values are normalized to sum to 1.0.
    pub fn new(similarity: f64, energy: f64, topology: f64, synapse: f64) -> Self {
        Self {
            similarity,
            energy,
            topology,
            synapse,
        }
    }

    /// Returns normalized weights (summing to 1.0).
    /// If all weights are zero, returns equal weights.
    fn normalized(&self) -> (f64, f64, f64, f64) {
        let total = self.similarity + self.energy + self.topology + self.synapse;
        if total <= 0.0 || total.is_nan() {
            return (0.25, 0.25, 0.25, 0.25);
        }
        (
            self.similarity / total,
            self.energy / total,
            self.topology / total,
            self.synapse / total,
        )
    }
}

// ---------------------------------------------------------------------------
// Search configuration
// ---------------------------------------------------------------------------

/// Configuration for the search pipeline.
#[derive(Debug, Clone)]
pub struct SearchConfig {
    /// Signal weights.
    pub weights: SearchWeights,
    /// Maximum number of results to return.
    pub limit: usize,
    /// Reference energy for normalization (passed to `energy_score`).
    pub ref_energy: f64,
    /// Maximum hops for synapse traversal expansion.
    pub synapse_max_hops: u32,
    /// Decay factor for synapse traversal.
    pub synapse_decay_factor: f64,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            weights: SearchWeights::default(),
            limit: 10,
            ref_energy: 1.0,
            synapse_max_hops: 2,
            synapse_decay_factor: 0.5,
        }
    }
}

// ---------------------------------------------------------------------------
// Search result
// ---------------------------------------------------------------------------

/// A single result from the search pipeline, with per-signal breakdown.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    /// The matched node.
    pub node_id: NodeId,
    /// Composite score in [0.0, 1.0].
    pub score: f64,
    /// Vector similarity signal in [0.0, 1.0].
    pub signal_similarity: f64,
    /// Energy (recency) signal in [0.0, 1.0].
    pub signal_energy: f64,
    /// Topology (PageRank + betweenness) signal in [0.0, 1.0].
    pub signal_topology: f64,
    /// Synapse traversal signal in [0.0, 1.0].
    pub signal_synapse: f64,
}

// ---------------------------------------------------------------------------
// Reranker trait
// ---------------------------------------------------------------------------

/// Optional post-processing hook for search results.
///
/// Implementations can reorder, filter, or augment scores.
/// The default [`NoopReranker`] returns results unchanged.
pub trait Reranker: Send + Sync {
    /// Reranks search results. May reorder, filter, or adjust scores.
    fn rerank(&self, results: Vec<SearchResult>) -> Vec<SearchResult>;
}

/// No-op reranker — returns results unchanged.
#[derive(Debug, Default)]
pub struct NoopReranker;

impl Reranker for NoopReranker {
    fn rerank(&self, results: Vec<SearchResult>) -> Vec<SearchResult> {
        results
    }
}

// ---------------------------------------------------------------------------
// SearchPipeline
// ---------------------------------------------------------------------------

/// Multi-signal search pipeline combining 4 cognitive signals.
///
/// Delegates to:
/// - HNSW index results (provided as pre-computed similarity candidates)
/// - `EnergyStore` for recency scoring
/// - `FabricStore` for topology (PageRank + betweenness)
/// - `SynapseStore` + spreading activation for graph expansion
pub struct SearchPipeline {
    /// Energy store for recency-based scoring.
    #[cfg(feature = "energy")]
    energy_store: Option<Arc<EnergyStore>>,

    /// Synapse store for graph expansion via spreading activation.
    #[cfg(feature = "synapse")]
    synapse_store: Option<Arc<SynapseStore>>,

    /// Fabric store for topology scoring (PageRank + betweenness).
    #[cfg(feature = "fabric")]
    fabric_store: Option<Arc<FabricStore>>,

    /// Optional reranker for post-processing.
    reranker: Box<dyn Reranker>,
}

impl SearchPipeline {
    /// Creates a new search pipeline with no stores (all signals disabled).
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "energy")]
            energy_store: None,
            #[cfg(feature = "synapse")]
            synapse_store: None,
            #[cfg(feature = "fabric")]
            fabric_store: None,
            reranker: Box::new(NoopReranker),
        }
    }

    /// Sets the energy store for recency scoring.
    #[cfg(feature = "energy")]
    pub fn with_energy_store(mut self, store: Arc<EnergyStore>) -> Self {
        self.energy_store = Some(store);
        self
    }

    /// Sets the synapse store for graph expansion.
    #[cfg(feature = "synapse")]
    pub fn with_synapse_store(mut self, store: Arc<SynapseStore>) -> Self {
        self.synapse_store = Some(store);
        self
    }

    /// Sets the fabric store for topology scoring.
    #[cfg(feature = "fabric")]
    pub fn with_fabric_store(mut self, store: Arc<FabricStore>) -> Self {
        self.fabric_store = Some(store);
        self
    }

    /// Sets a custom reranker.
    pub fn with_reranker(mut self, reranker: Box<dyn Reranker>) -> Self {
        self.reranker = reranker;
        self
    }

    /// Executes the search pipeline.
    ///
    /// # Arguments
    /// * `vector_candidates` — Pre-computed (node_id, similarity_score) pairs from HNSW.
    ///   Similarity scores should already be in [0, 1] (higher = more similar).
    /// * `config` — Search configuration (weights, limits, etc.).
    ///
    /// # Returns
    /// Sorted `Vec<SearchResult>` with composite scores in [0, 1].
    pub fn search(
        &self,
        vector_candidates: &[(NodeId, f64)],
        config: &SearchConfig,
    ) -> Vec<SearchResult> {
        let (w_sim, w_energy, w_topo, w_syn) = config.weights.normalized();

        // Collect all candidate node IDs from vector search
        let mut candidates: HashMap<NodeId, (f64, f64, f64, f64)> = HashMap::new();
        for &(node_id, sim_score) in vector_candidates {
            let sim = sim_score.clamp(0.0, 1.0);
            candidates.insert(node_id, (sim, 0.0, 0.0, 0.0));
        }

        // --- Signal 2: Energy scoring ---
        if w_energy > 0.0 {
            self.compute_energy_signal(&mut candidates, config.ref_energy);
        }

        // --- Signal 3: Topology scoring ---
        if w_topo > 0.0 {
            self.compute_topology_signal(&mut candidates);
        }

        // --- Signal 4: Synapse traversal (graph expansion) ---
        if w_syn > 0.0 {
            self.compute_synapse_signal(&mut candidates, vector_candidates, config);
        }

        // --- Composite scoring ---
        let mut results: Vec<SearchResult> = candidates
            .into_iter()
            .map(|(node_id, (sim, energy, topo, syn))| {
                let score =
                    (w_sim * sim + w_energy * energy + w_topo * topo + w_syn * syn).clamp(0.0, 1.0);
                SearchResult {
                    node_id,
                    score,
                    signal_similarity: sim,
                    signal_energy: energy,
                    signal_topology: topo,
                    signal_synapse: syn,
                }
            })
            .collect();

        // Sort by score descending
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Apply limit
        results.truncate(config.limit);

        // Apply reranker
        self.reranker.rerank(results)
    }

    /// Computes energy signal for all candidates.
    fn compute_energy_signal(
        &self,
        #[allow(unused_variables)] candidates: &mut HashMap<NodeId, (f64, f64, f64, f64)>,
        #[allow(unused_variables)] ref_energy: f64,
    ) {
        #[cfg(feature = "energy")]
        if let Some(store) = &self.energy_store {
            for (node_id, signals) in candidates.iter_mut() {
                let raw = store.get_energy(*node_id);
                signals.1 = energy_score(raw, ref_energy);
            }
        }
    }

    /// Computes topology signal (PageRank + betweenness) for all candidates.
    fn compute_topology_signal(
        &self,
        #[allow(unused_variables)] candidates: &mut HashMap<NodeId, (f64, f64, f64, f64)>,
    ) {
        #[cfg(feature = "fabric")]
        if let Some(store) = &self.fabric_store {
            // First pass: find max values for normalization
            let mut max_pr = 0.0_f64;
            let mut max_btwn = 0.0_f64;
            for node_id in candidates.keys() {
                let fabric = store.get_fabric_score(*node_id);
                max_pr = max_pr.max(fabric.pagerank);
                max_btwn = max_btwn.max(fabric.betweenness);
            }

            // Second pass: compute normalized topology score
            for (node_id, signals) in candidates.iter_mut() {
                let fabric = store.get_fabric_score(*node_id);
                let pr_norm = if max_pr > 0.0 {
                    (fabric.pagerank / max_pr).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let btwn_norm = if max_btwn > 0.0 {
                    (fabric.betweenness / max_btwn).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                // Combine PageRank and betweenness equally
                signals.2 = (0.5 * pr_norm + 0.5 * btwn_norm).clamp(0.0, 1.0);
            }
        }
    }

    /// Computes synapse signal via spreading activation from top candidates.
    fn compute_synapse_signal(
        &self,
        #[allow(unused_variables)] candidates: &mut HashMap<NodeId, (f64, f64, f64, f64)>,
        #[allow(unused_variables)] vector_candidates: &[(NodeId, f64)],
        #[allow(unused_variables)] config: &SearchConfig,
    ) {
        #[cfg(feature = "synapse")]
        if let Some(store) = &self.synapse_store {
            let source = SynapseActivationSource::new(Arc::clone(store));
            let spread_config = SpreadConfig::default()
                .with_max_hops(config.synapse_max_hops)
                .with_decay_factor(config.synapse_decay_factor);

            // Use top-k vector candidates as activation sources
            let sources: Vec<(NodeId, f64)> = vector_candidates
                .iter()
                .take(config.limit.max(5))
                .map(|&(nid, sim)| (nid, sim.clamp(0.0, 1.0)))
                .collect();

            if sources.is_empty() {
                return;
            }

            let activation_map = spread(&sources, &source, &spread_config);

            // Find max activation for normalization
            let max_activation = activation_map.values().copied().fold(0.0_f64, f64::max);

            if max_activation <= 0.0 {
                return;
            }

            // Apply synapse scores to existing candidates AND add newly discovered nodes
            for (node_id, activation) in &activation_map {
                let syn_score = (*activation / max_activation).clamp(0.0, 1.0);
                candidates
                    .entry(*node_id)
                    .and_modify(|signals| signals.3 = syn_score)
                    .or_insert((0.0, 0.0, 0.0, syn_score));
            }
        }
    }
}

impl Default for SearchPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for SearchPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut d = f.debug_struct("SearchPipeline");
        #[cfg(feature = "energy")]
        d.field("energy", &self.energy_store.is_some());
        #[cfg(feature = "synapse")]
        d.field("synapse", &self.synapse_store.is_some());
        #[cfg(feature = "fabric")]
        d.field("fabric", &self.fabric_store.is_some());
        d.finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn nid(id: u64) -> NodeId {
        NodeId(id)
    }

    #[test]
    fn search_weights_normalization() {
        let weights = SearchWeights::new(1.0, 1.0, 1.0, 1.0);
        let (sim, energy, topo, syn) = weights.normalized();
        assert!((sim - 0.25).abs() < 1e-10);
        assert!((energy - 0.25).abs() < 1e-10);
        assert!((topo - 0.25).abs() < 1e-10);
        assert!((syn - 0.25).abs() < 1e-10);
    }

    #[test]
    fn search_weights_zero_fallback() {
        let weights = SearchWeights::new(0.0, 0.0, 0.0, 0.0);
        let (sim, energy, topo, syn) = weights.normalized();
        assert!((sim - 0.25).abs() < 1e-10);
        assert!((energy + topo + syn - 0.75).abs() < 1e-10);
    }

    #[test]
    fn search_composite_similarity_only() {
        let pipeline = SearchPipeline::new();
        let config = SearchConfig {
            weights: SearchWeights::new(1.0, 0.0, 0.0, 0.0),
            limit: 5,
            ..Default::default()
        };

        let candidates = vec![(nid(1), 0.9), (nid(2), 0.5), (nid(3), 0.7)];

        let results = pipeline.search(&candidates, &config);

        assert_eq!(results.len(), 3);
        // Should be sorted by similarity descending
        assert_eq!(results[0].node_id, nid(1));
        assert_eq!(results[1].node_id, nid(3));
        assert_eq!(results[2].node_id, nid(2));

        // All scores in [0, 1]
        for r in &results {
            assert!(
                r.score >= 0.0 && r.score <= 1.0,
                "score out of range: {}",
                r.score
            );
            assert!(r.signal_similarity >= 0.0 && r.signal_similarity <= 1.0);
            assert!(r.signal_energy >= 0.0 && r.signal_energy <= 1.0);
            assert!(r.signal_topology >= 0.0 && r.signal_topology <= 1.0);
            assert!(r.signal_synapse >= 0.0 && r.signal_synapse <= 1.0);
        }
    }

    #[test]
    fn search_composite_scores_bounded() {
        let pipeline = SearchPipeline::new();
        let config = SearchConfig::default();

        // Even with extreme inputs, scores should be bounded
        let candidates = vec![
            (nid(1), 1.5),  // Over 1.0 — should be clamped
            (nid(2), -0.5), // Negative — should be clamped
            (nid(3), 0.5),
        ];

        let results = pipeline.search(&candidates, &config);
        for r in &results {
            assert!(
                r.score >= 0.0 && r.score <= 1.0,
                "score out of range: {}",
                r.score
            );
        }
    }

    #[test]
    fn search_composite_limit_applied() {
        let pipeline = SearchPipeline::new();
        let config = SearchConfig {
            limit: 2,
            ..Default::default()
        };

        let candidates: Vec<(NodeId, f64)> =
            (0..10).map(|i| (nid(i), 1.0 - (i as f64 * 0.1))).collect();

        let results = pipeline.search(&candidates, &config);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn search_composite_disabled_signal_excluded() {
        // When a weight is 0, that signal should not contribute
        let pipeline = SearchPipeline::new();
        let config = SearchConfig {
            weights: SearchWeights::new(1.0, 0.0, 0.0, 0.0),
            ..Default::default()
        };

        let candidates = vec![(nid(1), 0.8)];
        let results = pipeline.search(&candidates, &config);

        assert_eq!(results.len(), 1);
        // With only similarity enabled, score should equal similarity
        assert!((results[0].score - 0.8).abs() < 1e-10);
        assert!((results[0].signal_energy - 0.0).abs() < 1e-10);
    }

    #[test]
    fn search_composite_custom_reranker() {
        struct ReverseReranker;
        impl Reranker for ReverseReranker {
            fn rerank(&self, mut results: Vec<SearchResult>) -> Vec<SearchResult> {
                results.reverse();
                results
            }
        }

        let pipeline = SearchPipeline::new().with_reranker(Box::new(ReverseReranker));

        let config = SearchConfig {
            weights: SearchWeights::new(1.0, 0.0, 0.0, 0.0),
            limit: 10,
            ..Default::default()
        };

        let candidates = vec![(nid(1), 0.9), (nid(2), 0.5)];

        let results = pipeline.search(&candidates, &config);
        // Default sort: nid(1) first, but reverse reranker should flip
        assert_eq!(results[0].node_id, nid(2));
        assert_eq!(results[1].node_id, nid(1));
    }

    #[cfg(feature = "energy")]
    #[test]
    fn search_composite_with_energy() {
        use crate::energy::{EnergyConfig, EnergyStore};

        let store = Arc::new(EnergyStore::new(EnergyConfig::default()));
        store.boost(nid(1), 5.0);
        store.boost(nid(2), 0.1);

        let pipeline = SearchPipeline::new().with_energy_store(store);
        let config = SearchConfig {
            weights: SearchWeights::new(0.5, 0.5, 0.0, 0.0),
            ..Default::default()
        };

        let candidates = vec![(nid(1), 0.5), (nid(2), 0.9)];

        let results = pipeline.search(&candidates, &config);
        assert_eq!(results.len(), 2);

        // Node 1 has higher energy, node 2 has higher similarity
        // Both scores should be in [0, 1]
        for r in &results {
            assert!(r.score >= 0.0 && r.score <= 1.0);
            assert!(r.signal_energy >= 0.0 && r.signal_energy <= 1.0);
        }

        // Node 1 should have higher energy signal than node 2
        let r1 = results.iter().find(|r| r.node_id == nid(1)).unwrap();
        let r2 = results.iter().find(|r| r.node_id == nid(2)).unwrap();
        assert!(r1.signal_energy > r2.signal_energy);
    }

    #[cfg(feature = "fabric")]
    #[test]
    fn search_composite_with_topology() {
        use crate::fabric::FabricStore;

        let store = Arc::new(FabricStore::new());
        store.set_gds_metrics(nid(1), 0.9, 0.8, None);
        store.set_gds_metrics(nid(2), 0.1, 0.1, None);

        let pipeline = SearchPipeline::new().with_fabric_store(store);
        let config = SearchConfig {
            weights: SearchWeights::new(0.0, 0.0, 1.0, 0.0),
            ..Default::default()
        };

        let candidates = vec![(nid(1), 0.5), (nid(2), 0.5)];

        let results = pipeline.search(&candidates, &config);
        assert_eq!(results.len(), 2);

        // Node 1 has higher topology (pagerank + betweenness)
        assert_eq!(results[0].node_id, nid(1));
        assert!(results[0].signal_topology > results[1].signal_topology);
        for r in &results {
            assert!(r.score >= 0.0 && r.score <= 1.0);
        }
    }

    #[test]
    fn search_benchmark_latency() {
        // Verify that search on 10K candidates completes quickly
        let pipeline = SearchPipeline::new();
        let config = SearchConfig {
            weights: SearchWeights::new(1.0, 0.0, 0.0, 0.0),
            limit: 10,
            ..Default::default()
        };

        let candidates: Vec<(NodeId, f64)> = (0..10_000)
            .map(|i| (nid(i), (i as f64 / 10_000.0)))
            .collect();

        let start = std::time::Instant::now();
        let results = pipeline.search(&candidates, &config);
        let elapsed = start.elapsed();

        assert_eq!(results.len(), 10);
        assert!(
            elapsed.as_millis() < 5,
            "Search took {}ms, expected < 5ms",
            elapsed.as_millis()
        );
    }
}
