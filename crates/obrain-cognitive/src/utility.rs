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

    #[allow(dead_code)]
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
    /// Optional substrate backend — writes through to the 5-bit `utility`
    /// sub-field of `NodeRecord.scar_util_affinity`. Scores are clamped to
    /// `[0, UTILITY_MAX_SCORE_Q5]` (5.0 by default) before quantization.
    #[cfg(feature = "substrate")]
    substrate: Option<Arc<obrain_substrate::SubstrateStore>>,
}

impl UtilityStore {
    /// Creates a new utility store (in-memory only).
    pub fn new(config: UtilityConfig) -> Self {
        Self {
            nodes: DashMap::new(),
            config,
            graph_store: None,
            #[cfg(feature = "substrate")]
            substrate: None,
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
            #[cfg(feature = "substrate")]
            substrate: None,
        }
    }

    /// Creates a utility store backed by a substrate column view (T6).
    ///
    /// Every `boost` updates the 5-bit utility sub-field atomically under
    /// the zone lock, preserving scar / affinity bits via
    /// [`obrain_substrate::writer::Writer::update_utility_field`].
    #[cfg(feature = "substrate")]
    pub fn with_substrate(
        config: UtilityConfig,
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

    /// Persists utility state for a node to the graph store.
    fn persist(&self, node_id: NodeId, utility: &NodeUtility) {
        // Substrate path: write-through quantized score to the 5-bit utility
        // sub-field.
        #[cfg(feature = "substrate")]
        if let Some(sub) = &self.substrate {
            let _ = sub.set_node_utility_field_f32(node_id, utility.current() as f32);
            return;
        }

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
    ///
    /// **Substrate mode**: reads the 5-bit utility sub-field directly
    /// (column is the source of truth); the hot cache is bypassed.
    pub fn get_utility(&self, node_id: NodeId) -> f64 {
        #[cfg(feature = "substrate")]
        if let Some(sub) = &self.substrate {
            return sub
                .get_node_utility_field_f32(node_id)
                .ok()
                .flatten()
                .map(|v| v as f64)
                .map(|v| if v < self.config.min_utility { 0.0 } else { v })
                .unwrap_or(0.0);
        }

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
                .map_or(0.0, |v| if v < self.config.min_utility { 0.0 } else { v })
        }
    }

    /// Get the access count for a node.
    pub fn get_count(&self, node_id: NodeId) -> u32 {
        self.nodes.get(&node_id).map_or(0, |e| e.count)
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
    ///
    /// **Substrate mode**: iterates the authoritative column.
    pub fn nodes_above(&self, threshold: f64) -> Vec<(NodeId, f64)> {
        #[cfg(feature = "substrate")]
        if let Some(sub) = &self.substrate {
            return match sub.iter_live_scar_util_affinity() {
                Ok(pairs) => pairs
                    .into_iter()
                    .filter_map(|(id, p)| {
                        let v = obrain_substrate::q5_to_utility(p.utility) as f64;
                        (v >= threshold).then_some((id, v))
                    })
                    .collect(),
                Err(_) => Vec::new(),
            };
        }

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
    ///
    /// **Substrate mode**: O(N) scan over the column.
    pub fn top_k(&self, k: usize) -> Vec<(NodeId, f64)> {
        #[cfg(feature = "substrate")]
        if let Some(sub) = &self.substrate {
            let mut entries: Vec<(NodeId, f64)> = match sub.iter_live_scar_util_affinity() {
                Ok(pairs) => pairs
                    .into_iter()
                    .map(|(id, p)| (id, obrain_substrate::q5_to_utility(p.utility) as f64))
                    .filter(|(_, v)| *v >= self.config.min_utility)
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
            .map(|entry| (*entry.key(), entry.current()))
            .collect();
        entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        entries.truncate(k);
        entries
    }

    /// Rehydrates the utility cache from the backing graph store.
    ///
    /// **Substrate mode**: no-op — the 5-bit utility sub-field is already
    /// crash-consistent on open; `get_utility` reads it directly.
    ///
    /// **Legacy mode**: iterates every node and loads `PROP_UTILITY_SCORE`
    /// (+ count + epoch) into the hot cache. Required on brain open so
    /// `len()` and `top_k()` reflect historical utility rather than
    /// starting empty.
    ///
    /// Returns the number of nodes loaded.
    pub fn load_from_graph(&self) -> usize {
        #[cfg(feature = "substrate")]
        if self.substrate.is_some() {
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
            let Some(score) = load_node_f64(gs.as_ref(), nid, PROP_UTILITY_SCORE) else {
                continue;
            };
            if score <= 0.0 {
                continue;
            }
            let count =
                load_node_f64(gs.as_ref(), nid, PROP_UTILITY_COUNT).unwrap_or(1.0) as u32;
            let epoch = load_node_f64(gs.as_ref(), nid, PROP_UTILITY_LAST_UPDATED_EPOCH);
            let last_updated = crate::store_trait::epoch_to_instant(epoch);
            let utility = NodeUtility {
                score,
                count,
                last_updated,
                half_life: self.config.default_half_life,
            };
            self.nodes.insert(nid, utility);
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
        let past = Instant::now()
            .checked_sub(Duration::from_secs(3600))
            .unwrap();
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
        let past = Instant::now()
            .checked_sub(Duration::from_secs(100))
            .unwrap(); // ~100 half-lives
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
    fn boost_writes_through_column() {
        let (sub, ids, _td) = make_substrate(3);
        let store = UtilityStore::with_substrate(UtilityConfig::default(), sub.clone());
        assert!(store.is_substrate_backed());
        let id = ids[0];
        store.boost(id, 2.5);
        let packed = sub.get_node_scar_util_affinity(id).unwrap().unwrap();
        // utility_to_q5(2.5) = (2.5 / 5 * 31).round() = 16
        assert!((15..=17).contains(&packed.utility), "got {}", packed.utility);
        assert!(packed.dirty);
    }

    #[test]
    fn get_utility_reads_column() {
        let (sub, ids, _td) = make_substrate(2);
        let store = UtilityStore::with_substrate(UtilityConfig::default(), sub.clone());
        let id = ids[0];
        store.boost(id, 3.0);
        store.nodes.clear();
        let u = store.get_utility(id);
        assert!((u - 3.0).abs() < 0.25, "got {u}");
    }

    #[test]
    fn boost_preserves_scar_and_affinity() {
        let (sub, ids, _td) = make_substrate(1);
        let store = UtilityStore::with_substrate(UtilityConfig::default(), sub.clone());
        let id = ids[0];
        sub.set_node_scar_field_f32(id, 2.0).unwrap();
        sub.set_node_affinity_field_f32(id, 0.5).unwrap();
        store.boost(id, 4.0);
        let packed = sub.get_node_scar_util_affinity(id).unwrap().unwrap();
        assert_eq!(packed.scar, 16); // 2/4 * 31 = 15.5 → 16
        assert_eq!(packed.affinity, 16); // 0.5 * 31 = 15.5 → 16
        assert!(packed.utility >= 24); // 4/5 * 31 ≈ 25
    }

    #[test]
    fn load_from_graph_is_noop_in_substrate_mode() {
        let (sub, _ids, _td) = make_substrate(1);
        let store = UtilityStore::with_substrate(UtilityConfig::default(), sub.clone());
        assert_eq!(store.load_from_graph(), 0);
    }

    #[test]
    fn top_k_reads_from_column() {
        let (sub, ids, _td) = make_substrate(5);
        let store = UtilityStore::with_substrate(UtilityConfig::default(), sub.clone());
        store.boost(ids[0], 0.5);
        store.boost(ids[1], 3.0);
        store.boost(ids[2], 1.5);
        let top = store.top_k(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, ids[1]);
        assert_eq!(top[1].0, ids[2]);
    }
}
