//! Utility Store — tracks long-term usefulness of nodes.
//!
//! Unlike energy (24h half-life, "recently activated") the utility score
//! captures "this node has been USEFUL when it appeared in context" —
//! a much slower signal with a 30-day half-life.
//!
//! Scars handle the negative signal; utility handles the positive signal.
//! Together they provide a durable memory of what works and what doesn't.
//!
//! ```text
//! utility(t) = U0 × 2^(-Δt / half_life)
//! ```

use dashmap::DashMap;
use obrain_common::types::NodeId;
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::store_trait::{
    OptionalGraphStore, PROP_UTILITY_COUNT, PROP_UTILITY_LAST_UPDATED_EPOCH, PROP_UTILITY_SCORE,
    load_node_f64, now_epoch_secs, persist_node_f64,
};

// ---------------------------------------------------------------------------
// UtilityConfig
// ---------------------------------------------------------------------------

/// Configuration for the utility system.
#[derive(Debug, Clone)]
pub struct UtilityConfig {
    /// Default half-life for utility decay (default: 30 days).
    pub default_half_life: Duration,
    /// Amount to boost utility when a node is mentioned in a good response.
    pub boost_amount: f64,
    /// Minimum utility threshold — below this, considered zero.
    pub min_utility: f64,
    /// Maximum utility cap.
    pub max_utility: f64,
}

impl Default for UtilityConfig {
    fn default() -> Self {
        Self {
            default_half_life: Duration::from_secs(30 * 24 * 3600), // 30 days
            boost_amount: 0.1,
            min_utility: 0.001,
            max_utility: 5.0,
        }
    }
}

// ---------------------------------------------------------------------------
// NodeUtility
// ---------------------------------------------------------------------------

/// Per-node utility state.
#[derive(Debug, Clone)]
struct NodeUtility {
    /// Raw (non-decayed) utility score.
    score: f64,
    /// Number of times this node was boosted.
    count: u32,
    /// When utility was last boosted (for decay calculation).
    last_updated: Instant,
    /// Half-life for decay.
    half_life: Duration,
}

impl NodeUtility {
    fn new(score: f64, half_life: Duration) -> Self {
        Self {
            score,
            count: 1,
            last_updated: Instant::now(),
            half_life,
        }
    }

    fn new_at(score: f64, half_life: Duration, at: Instant) -> Self {
        Self {
            score,
            count: 1,
            last_updated: at,
            half_life,
        }
    }

    /// Returns the current utility after decay.
    fn current(&self) -> f64 {
        self.at(Instant::now())
    }

    /// Returns the utility at a specific instant.
    fn at(&self, now: Instant) -> f64 {
        let elapsed = now.duration_since(self.last_updated);
        let half_lives = elapsed.as_secs_f64() / self.half_life.as_secs_f64();
        self.score * 2.0_f64.powf(-half_lives)
    }
}

// ---------------------------------------------------------------------------
// UtilityStore
// ---------------------------------------------------------------------------

/// Thread-safe store for node utility scores.
///
/// Write-through to the backing graph store: every boost persists
/// `_cog_utility_score`, `_cog_utility_count`, and `_cog_utility_last_updated_epoch`.
/// On read, lazily loads from graph properties if not in the hot cache.
pub struct UtilityStore {
    /// Hot cache: node_id → utility state.
    nodes: DashMap<NodeId, NodeUtility>,
    /// Configuration.
    config: UtilityConfig,
    /// Optional backing graph store for write-through persistence.
    graph_store: OptionalGraphStore,
}

impl UtilityStore {
    /// Creates a new utility store (in-memory only).
    pub fn new(config: UtilityConfig) -> Self {
        Self {
            nodes: DashMap::new(),
            config,
            graph_store: None,
        }
    }

    /// Creates a new utility store with write-through persistence.
    pub fn with_graph_store(
        config: UtilityConfig,
        graph_store: Arc<dyn obrain_core::graph::GraphStoreMut>,
    ) -> Self {
        Self {
            nodes: DashMap::new(),
            config,
            graph_store: Some(graph_store),
        }
    }

    /// Persists utility state for a node to the graph store.
    fn persist(&self, node_id: NodeId, utility: &NodeUtility) {
        if let Some(gs) = &self.graph_store {
            persist_node_f64(gs.as_ref(), node_id, PROP_UTILITY_SCORE, utility.current());
            persist_node_f64(
                gs.as_ref(),
                node_id,
                PROP_UTILITY_COUNT,
                utility.count as f64,
            );
            persist_node_f64(
                gs.as_ref(),
                node_id,
                PROP_UTILITY_LAST_UPDATED_EPOCH,
                now_epoch_secs(),
            );
        }
    }

    /// Lazy-load from graph store if not in hot cache.
    fn lazy_load(&self, node_id: NodeId) -> Option<f64> {
        if let Some(gs) = &self.graph_store {
            let score = load_node_f64(gs.as_ref(), node_id, PROP_UTILITY_SCORE)?;
            let count =
                load_node_f64(gs.as_ref(), node_id, PROP_UTILITY_COUNT).unwrap_or(1.0) as u32;
            let epoch = load_node_f64(gs.as_ref(), node_id, PROP_UTILITY_LAST_UPDATED_EPOCH);
            let last_updated = crate::store_trait::epoch_to_instant(epoch);

            let utility = NodeUtility {
                score,
                count,
                last_updated,
                half_life: self.config.default_half_life,
            };
            let current = utility.current();
            self.nodes.insert(node_id, utility);
            Some(current)
        } else {
            None
        }
    }

    /// Boost the utility of a node (positive feedback signal).
    ///
    /// Called when a node is mentioned in a good response.
    /// Returns the new utility score.
    pub fn boost(&self, node_id: NodeId, amount: f64) -> f64 {
        let amount = amount.max(0.0);
        let mut entry = self
            .nodes
            .entry(node_id)
            .or_insert_with(|| NodeUtility::new(0.0, self.config.default_half_life));

        let utility = entry.value_mut();
        // Add to current (decayed) value, then re-anchor at now
        let current = utility.current();
        utility.score = (current + amount).min(self.config.max_utility);
        utility.count += 1;
        utility.last_updated = Instant::now();

        let result = utility.score;
        self.persist(node_id, utility);
        result
    }

    /// Get the current utility score for a node.
    ///
    /// Returns 0.0 for nodes with no utility history.
    /// Lazily loads from graph store if not in hot cache.
    pub fn get_utility(&self, node_id: NodeId) -> f64 {
        if let Some(entry) = self.nodes.get(&node_id) {
            let val = entry.current();
            if val < self.config.min_utility {
                0.0
            } else {
                val
            }
        } else {
            // Try lazy load from graph
            self.lazy_load(node_id)
                .map(|v| if v < self.config.min_utility { 0.0 } else { v })
                .unwrap_or(0.0)
        }
    }

    /// Get the access count for a node.
    pub fn get_count(&self, node_id: NodeId) -> u32 {
        self.nodes.get(&node_id).map(|e| e.count).unwrap_or(0)
    }

    /// Prune nodes with utility below the minimum threshold.
    /// Returns the number of entries pruned.
    pub fn prune(&self) -> usize {
        let min = self.config.min_utility;
        let before = self.nodes.len();
        self.nodes.retain(|_, v| v.current() >= min);
        before - self.nodes.len()
    }

    /// Get all nodes with utility above a threshold.
    pub fn nodes_above(&self, threshold: f64) -> Vec<(NodeId, f64)> {
        self.nodes
            .iter()
            .filter_map(|entry| {
                let val = entry.current();
                if val >= threshold {
                    Some((*entry.key(), val))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get the top-K nodes by utility score.
    pub fn top_k(&self, k: usize) -> Vec<(NodeId, f64)> {
        let mut entries: Vec<(NodeId, f64)> = self
            .nodes
            .iter()
            .map(|entry| (*entry.key(), entry.current()))
            .collect();
        entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        entries.truncate(k);
        entries
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
    pub fn config(&self) -> &UtilityConfig {
        &self.config
    }
}

impl Default for UtilityStore {
    fn default() -> Self {
        Self::new(UtilityConfig::default())
    }
}

impl fmt::Debug for UtilityStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UtilityStore")
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
    fn test_boost_and_get() {
        let store = UtilityStore::default();
        store.boost(nid(1), 0.5);
        let u = store.get_utility(nid(1));
        assert!(u > 0.49 && u < 0.51, "got {u}");
    }

    #[test]
    fn test_boost_accumulates() {
        let store = UtilityStore::default();
        store.boost(nid(1), 0.3);
        store.boost(nid(1), 0.2);
        let u = store.get_utility(nid(1));
        assert!(u > 0.49 && u < 0.51, "got {u}");
        assert_eq!(store.get_count(nid(1)), 3); // initial(1) + 2 boosts
    }

    #[test]
    fn test_capped_at_max() {
        let config = UtilityConfig {
            max_utility: 1.0,
            ..Default::default()
        };
        let store = UtilityStore::new(config);
        store.boost(nid(1), 0.8);
        store.boost(nid(1), 0.8);
        let u = store.get_utility(nid(1));
        assert!(u <= 1.01, "should be capped at max_utility, got {u}");
    }

    #[test]
    fn test_decay_over_time() {
        let config = UtilityConfig {
            default_half_life: Duration::from_secs(3600), // 1 hour
            ..Default::default()
        };
        let store = UtilityStore::new(config);

        // Insert a utility "in the past"
        let past = Instant::now() - Duration::from_secs(3600);
        store.nodes.insert(
            nid(1),
            NodeUtility::new_at(0.8, Duration::from_secs(3600), past),
        );

        let u = store.get_utility(nid(1));
        // After 1 half-life: 0.8 * 0.5 = 0.4
        assert!(u > 0.35 && u < 0.45, "expected ~0.4, got {u}");
    }

    #[test]
    fn test_unknown_node_returns_zero() {
        let store = UtilityStore::default();
        assert_eq!(store.get_utility(nid(999)), 0.0);
    }

    #[test]
    fn test_prune() {
        let config = UtilityConfig {
            default_half_life: Duration::from_secs(1), // Very short for test
            min_utility: 0.01,
            ..Default::default()
        };
        let store = UtilityStore::new(config);

        // Insert an old entry that should have decayed below threshold
        let past = Instant::now() - Duration::from_secs(100); // ~100 half-lives
        store.nodes.insert(
            nid(1),
            NodeUtility::new_at(0.5, Duration::from_secs(1), past),
        );
        store.boost(nid(2), 0.5); // Fresh entry

        let pruned = store.prune();
        assert_eq!(pruned, 1, "old entry should be pruned");
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_top_k() {
        let store = UtilityStore::default();
        store.boost(nid(1), 0.1);
        store.boost(nid(2), 0.5);
        store.boost(nid(3), 0.3);

        let top = store.top_k(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, nid(2)); // Highest
        assert_eq!(top[1].0, nid(3)); // Second
    }
}
