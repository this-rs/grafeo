//! Query Affinity Store — graph-native prediction of future relevance.
//!
//! Each node accumulates an EMA (Exponential Moving Average) of the cosine
//! similarity between the query and the node's embedding at retrieval time.
//! Nodes that are consistently retrieved for similar queries develop high
//! affinity — they become predictors of future relevance.
//!
//! This replaces a separate MLP/neural network: the graph IS the world model.
//! The signal is:
//! - Computed from ONNX embeddings already calculated during retrieval (zero extra cost)
//! - Persisted as a node property (survives restarts, no re-training)
//! - Converges in ~5-10 interactions (not 50-100 rounds of training)
//!
//! ```text
//! affinity_new = β × cosine_sim + (1 - β) × affinity_old
//! ```

use dashmap::DashMap;
use obrain_common::types::NodeId;
use std::fmt;
use std::sync::Arc;

use crate::store_trait::{
    OptionalGraphStore, PROP_QUERY_AFFINITY, PROP_QUERY_AFFINITY_COUNT, load_node_f64,
    persist_node_f64,
};

// ---------------------------------------------------------------------------
// AffinityConfig
// ---------------------------------------------------------------------------

/// Configuration for the query affinity system.
#[derive(Debug, Clone)]
pub struct AffinityConfig {
    /// EMA blending factor (default: 0.2).
    /// Higher = faster adaptation, lower = more stable.
    pub ema_alpha: f64,
    /// Minimum affinity to keep in cache (below this, considered zero).
    pub min_affinity: f64,
    /// Number of top-K nodes to prefetch at session start.
    pub prefetch_top_k: usize,
}

impl Default for AffinityConfig {
    fn default() -> Self {
        Self {
            ema_alpha: 0.2,
            min_affinity: 0.01,
            prefetch_top_k: 10,
        }
    }
}

// ---------------------------------------------------------------------------
// NodeAffinity
// ---------------------------------------------------------------------------

/// Per-node query affinity state.
#[derive(Debug, Clone)]
struct NodeAffinity {
    /// EMA of cosine similarity with recent queries.
    score: f64,
    /// Number of times this affinity was updated.
    count: u32,
}

impl NodeAffinity {
    fn new(score: f64) -> Self {
        Self { score, count: 1 }
    }
}

// ---------------------------------------------------------------------------
// AffinityStore
// ---------------------------------------------------------------------------

/// Thread-safe store for query affinity scores.
///
/// Write-through to the backing graph store. On read, lazily loads from
/// graph properties if not in the hot cache.
pub struct AffinityStore {
    /// Hot cache: node_id → affinity state.
    nodes: DashMap<NodeId, NodeAffinity>,
    /// Configuration.
    config: AffinityConfig,
    /// Optional backing graph store for write-through persistence.
    graph_store: OptionalGraphStore,
    /// Optional substrate backend — writes through to the 5-bit `affinity`
    /// sub-field of `NodeRecord.scar_util_affinity`.
    #[cfg(feature = "substrate")]
    substrate: Option<Arc<obrain_substrate::SubstrateStore>>,
}

impl AffinityStore {
    /// Creates a new affinity store (in-memory only).
    pub fn new(config: AffinityConfig) -> Self {
        Self {
            nodes: DashMap::new(),
            config,
            graph_store: None,
            #[cfg(feature = "substrate")]
            substrate: None,
        }
    }

    /// Creates a new affinity store with write-through persistence.
    pub fn with_graph_store(
        config: AffinityConfig,
        graph_store: Arc<dyn obrain_core::graph::GraphStoreMut>,
    ) -> Self {
        Self {
            nodes: DashMap::new(),
            config,
            graph_store: Some(graph_store),
            #[cfg(feature = "substrate")]
            substrate: None,
        }
    }

    /// Creates an affinity store backed by a substrate column view (T6).
    #[cfg(feature = "substrate")]
    pub fn with_substrate(
        config: AffinityConfig,
        substrate: Arc<obrain_substrate::SubstrateStore>,
    ) -> Self {
        Self {
            nodes: DashMap::new(),
            config,
            graph_store: None,
            substrate: Some(substrate),
        }
    }

    /// Returns `true` if this store routes through a substrate column view.
    #[cfg(feature = "substrate")]
    pub fn is_substrate_backed(&self) -> bool {
        self.substrate.is_some()
    }

    /// Persists affinity state for a node.
    fn persist(&self, node_id: NodeId, affinity: &NodeAffinity) {
        #[cfg(feature = "substrate")]
        if let Some(sub) = &self.substrate {
            let _ = sub.set_node_affinity_field_f32(node_id, affinity.score as f32);
            return;
        }

        if let Some(gs) = &self.graph_store {
            persist_node_f64(gs.as_ref(), node_id, PROP_QUERY_AFFINITY, affinity.score);
            persist_node_f64(
                gs.as_ref(),
                node_id,
                PROP_QUERY_AFFINITY_COUNT,
                affinity.count as f64,
            );
        }
    }

    /// Lazy-load from graph store if not in hot cache.
    fn lazy_load(&self, node_id: NodeId) -> Option<f64> {
        if let Some(gs) = &self.graph_store {
            let score = load_node_f64(gs.as_ref(), node_id, PROP_QUERY_AFFINITY)?;
            let count = load_node_f64(gs.as_ref(), node_id, PROP_QUERY_AFFINITY_COUNT)
                .unwrap_or(1.0) as u32;
            let affinity = NodeAffinity { score, count };
            let val = affinity.score;
            self.nodes.insert(node_id, affinity);
            Some(val)
        } else {
            None
        }
    }

    /// Update the query affinity for a node with a new cosine similarity.
    ///
    /// Uses EMA: `affinity = α × cosine_sim + (1 - α) × old_affinity`
    ///
    /// For the first update (cold start), affinity is set directly to
    /// `cosine_sim` — no smoothing delay.
    ///
    /// Returns the new affinity score.
    pub fn update(&self, node_id: NodeId, cosine_sim: f64) -> f64 {
        let alpha = self.config.ema_alpha;
        let cosine_sim = cosine_sim.clamp(0.0, 1.0);

        let mut entry = self.nodes.entry(node_id).or_insert_with(|| {
            // Cold start: check graph store first
            if let Some(gs) = &self.graph_store
                && let Some(existing) = load_node_f64(gs.as_ref(), node_id, PROP_QUERY_AFFINITY)
            {
                let count = load_node_f64(gs.as_ref(), node_id, PROP_QUERY_AFFINITY_COUNT)
                    .unwrap_or(1.0) as u32;
                return NodeAffinity {
                    score: existing,
                    count,
                };
            }
            // True cold start — first interaction, no EMA needed
            NodeAffinity::new(cosine_sim)
        });

        let affinity = entry.value_mut();
        if affinity.count == 1 && affinity.score == cosine_sim {
            // Just created with cold start value, persist and return
            self.persist(node_id, affinity);
            return affinity.score;
        }

        // EMA update
        affinity.score = alpha * cosine_sim + (1.0 - alpha) * affinity.score;
        affinity.count += 1;

        let result = affinity.score;
        self.persist(node_id, affinity);
        result
    }

    /// Reduce the query affinity by a ratio (anti-prediction signal).
    ///
    /// Called when feedback is negative — the node was in context but
    /// led to a bad response, so we reduce its predicted relevance.
    ///
    /// Returns the new affinity score.
    pub fn penalize(&self, node_id: NodeId, ratio: f64) -> f64 {
        let ratio = ratio.clamp(0.0, 1.0);
        if let Some(mut entry) = self.nodes.get_mut(&node_id) {
            entry.score *= 1.0 - ratio;
            let result = entry.score;
            let snapshot = entry.clone();
            drop(entry);
            self.persist(node_id, &snapshot);
            result
        } else {
            0.0
        }
    }

    /// Get the current query affinity for a node.
    ///
    /// Returns 0.0 for nodes with no affinity history.
    ///
    /// **Substrate mode**: reads the 5-bit affinity sub-field directly.
    /// **Legacy mode**: Lazily loads from graph store if not in hot cache.
    pub fn get_affinity(&self, node_id: NodeId) -> f64 {
        #[cfg(feature = "substrate")]
        if let Some(sub) = &self.substrate {
            return sub
                .get_node_affinity_field_f32(node_id)
                .ok()
                .flatten()
                .map(|v| v as f64)
                .map(|v| if v < self.config.min_affinity { 0.0 } else { v })
                .unwrap_or(0.0);
        }

        if let Some(entry) = self.nodes.get(&node_id) {
            let val = entry.score;
            if val < self.config.min_affinity {
                0.0
            } else {
                val
            }
        } else {
            self.lazy_load(node_id)
                .map_or(0.0, |v| if v < self.config.min_affinity { 0.0 } else { v })
        }
    }

    /// Get the update count for a node.
    pub fn get_count(&self, node_id: NodeId) -> u32 {
        self.nodes.get(&node_id).map_or(0, |e| e.count)
    }

    /// Get the top-K nodes by affinity score.
    ///
    /// Used for predictive prefetch at session start.
    ///
    /// **Substrate mode**: O(N) scan over the authoritative column.
    pub fn top_k(&self, k: usize) -> Vec<(NodeId, f64)> {
        #[cfg(feature = "substrate")]
        if let Some(sub) = &self.substrate {
            let mut entries: Vec<(NodeId, f64)> = match sub.iter_live_scar_util_affinity() {
                Ok(pairs) => pairs
                    .into_iter()
                    .map(|(id, p)| (id, obrain_substrate::q5_to_affinity(p.affinity) as f64))
                    .filter(|(_, v)| *v >= self.config.min_affinity)
                    .collect(),
                Err(_) => return Vec::new(),
            };
            entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            entries.truncate(k);
            return entries;
        }

        let mut entries: Vec<(NodeId, f64)> = self
            .nodes
            .iter()
            .filter(|e| e.score >= self.config.min_affinity)
            .map(|entry| (*entry.key(), entry.score))
            .collect();
        entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        entries.truncate(k);
        entries
    }

    /// Rehydrates the affinity cache from the backing graph store.
    ///
    /// **Substrate mode**: no-op — the 5-bit affinity sub-field is already
    /// crash-consistent on open; reads go through the column directly.
    ///
    /// **Legacy mode**: iterates every node and loads `PROP_QUERY_AFFINITY`
    /// (+ count) into the hot cache. Required on brain open so `len()`,
    /// `top_k()`, and predictive prefetch reflect historical affinity
    /// rather than starting empty.
    ///
    /// Returns the number of nodes loaded.
    pub fn load_from_graph(&self) -> usize {
        #[cfg(feature = "substrate")]
        if self.substrate.is_some() {
            tracing::trace!("affinity::load_from_graph: no-op (state is column-resident)");
            return 0;
        }

        let Some(gs) = self.graph_store.as_ref() else {
            return 0;
        };
        let mut loaded = 0usize;
        for nid in gs.node_ids() {
            if self.nodes.contains_key(&nid) {
                continue;
            }
            let Some(score) = load_node_f64(gs.as_ref(), nid, PROP_QUERY_AFFINITY) else {
                continue;
            };
            if score <= 0.0 {
                continue;
            }
            let count =
                load_node_f64(gs.as_ref(), nid, PROP_QUERY_AFFINITY_COUNT).unwrap_or(1.0) as u32;
            self.nodes.insert(nid, NodeAffinity { score, count });
            loaded += 1;
        }
        loaded
    }

    /// Total number of tracked nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Returns a reference to the config.
    pub fn config(&self) -> &AffinityConfig {
        &self.config
    }
}

impl Default for AffinityStore {
    fn default() -> Self {
        Self::new(AffinityConfig::default())
    }
}

impl fmt::Debug for AffinityStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AffinityStore")
            .field("nodes", &self.nodes.len())
            .field("config", &self.config)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn nid(n: u64) -> NodeId {
        NodeId(n)
    }

    #[test]
    fn test_cold_start() {
        let store = AffinityStore::default();
        let a = store.update(nid(1), 0.8);
        // First update = cold start, should be cosine_sim directly
        assert!(a > 0.79 && a < 0.81, "cold start should be ~0.8, got {a}");
    }

    #[test]
    fn test_ema_convergence() {
        let store = AffinityStore::default(); // alpha = 0.2
        // First: cold start at 0.5
        store.update(nid(1), 0.5);
        // Second: EMA(0.2 * 0.9 + 0.8 * 0.5) = 0.18 + 0.4 = 0.58
        let a = store.update(nid(1), 0.9);
        assert!(a > 0.57 && a < 0.59, "EMA should be ~0.58, got {a}");
        // Third: EMA(0.2 * 0.9 + 0.8 * 0.58) = 0.18 + 0.464 = 0.644
        let a = store.update(nid(1), 0.9);
        assert!(
            a > 0.63 && a < 0.66,
            "EMA should converge toward 0.9, got {a}"
        );
    }

    #[test]
    fn test_multiple_topics_converge_independently() {
        let store = AffinityStore::default();
        store.update(nid(1), 0.9); // Node 1: high affinity topic
        store.update(nid(2), 0.2); // Node 2: low affinity topic

        let a1 = store.get_affinity(nid(1));
        let a2 = store.get_affinity(nid(2));
        assert!(a1 > a2, "node 1 should have higher affinity");
    }

    #[test]
    fn test_penalize_reduces_affinity() {
        let store = AffinityStore::default();
        store.update(nid(1), 0.8);
        let before = store.get_affinity(nid(1));
        let after = store.penalize(nid(1), 0.3); // Reduce by 30%
        assert!(after < before, "penalize should reduce affinity");
        assert!(
            after > 0.55 && after < 0.57,
            "0.8 * 0.7 = 0.56, got {after}"
        );
    }

    #[test]
    fn test_unknown_node_returns_zero() {
        let store = AffinityStore::default();
        assert_eq!(store.get_affinity(nid(999)), 0.0);
    }

    #[test]
    fn test_top_k() {
        let store = AffinityStore::default();
        store.update(nid(1), 0.3);
        store.update(nid(2), 0.9);
        store.update(nid(3), 0.6);

        let top = store.top_k(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, nid(2)); // Highest
        assert_eq!(top[1].0, nid(3)); // Second
    }

    #[test]
    fn test_cosine_clamped() {
        let store = AffinityStore::default();
        let a = store.update(nid(1), 1.5); // Above 1.0
        assert!(a <= 1.01, "should be clamped to 1.0, got {a}");
        let a = store.update(nid(2), -0.5); // Below 0.0
        assert!(a >= -0.01, "should be clamped to 0.0, got {a}");
    }
}

// ---------------------------------------------------------------------------
// Substrate-backed integration tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "substrate"))]
mod substrate_tests {
    use super::*;
    use obrain_core::graph::traits::GraphStoreMut;
    use obrain_substrate::SubstrateStore;

    fn make_substrate(n: usize) -> (Arc<SubstrateStore>, Vec<NodeId>, tempfile::TempDir) {
        let td = tempfile::tempdir().unwrap();
        let sub = SubstrateStore::create(td.path().join("kb")).unwrap();
        let mut ids = Vec::with_capacity(n);
        for _ in 0..n {
            ids.push(sub.create_node(&["n"]));
        }
        sub.flush().unwrap();
        (Arc::new(sub), ids, td)
    }

    #[test]
    fn update_writes_through_column() {
        let (sub, ids, _td) = make_substrate(3);
        let store = AffinityStore::with_substrate(AffinityConfig::default(), sub.clone());
        assert!(store.is_substrate_backed());
        let id = ids[0];
        store.update(id, 0.8);
        let packed = sub.get_node_scar_util_affinity(id).unwrap().unwrap();
        // affinity_to_q5(0.8) = (0.8 * 31).round() = 25
        assert!(
            (24..=26).contains(&packed.affinity),
            "got {}",
            packed.affinity
        );
        assert!(packed.dirty);
    }

    #[test]
    fn get_affinity_reads_column() {
        let (sub, ids, _td) = make_substrate(2);
        let store = AffinityStore::with_substrate(AffinityConfig::default(), sub.clone());
        let id = ids[0];
        store.update(id, 0.6);
        store.nodes.clear();
        let a = store.get_affinity(id);
        assert!((a - 0.6).abs() < 0.05, "got {a}");
    }

    #[test]
    fn update_preserves_scar_and_utility() {
        let (sub, ids, _td) = make_substrate(1);
        let store = AffinityStore::with_substrate(AffinityConfig::default(), sub.clone());
        let id = ids[0];
        sub.set_node_scar_field_f32(id, 1.0).unwrap();
        sub.set_node_utility_field_f32(id, 2.5).unwrap();
        store.update(id, 0.9);
        let packed = sub.get_node_scar_util_affinity(id).unwrap().unwrap();
        assert_eq!(packed.scar, 8); // 1.0 / 4 * 31 ≈ 8
        assert!((15..=17).contains(&packed.utility));
        assert!((27..=29).contains(&packed.affinity));
    }

    #[test]
    fn load_from_graph_is_noop_in_substrate_mode() {
        let (sub, _ids, _td) = make_substrate(1);
        let store = AffinityStore::with_substrate(AffinityConfig::default(), sub.clone());
        assert_eq!(store.load_from_graph(), 0);
    }

    #[test]
    fn top_k_reads_from_column() {
        let (sub, ids, _td) = make_substrate(5);
        let store = AffinityStore::with_substrate(AffinityConfig::default(), sub.clone());
        store.update(ids[0], 0.2);
        store.update(ids[1], 0.9);
        store.update(ids[2], 0.5);
        let top = store.top_k(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, ids[1]);
        assert_eq!(top[1].0, ids[2]);
    }
}
