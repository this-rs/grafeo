//! # KernelManager — Event-driven incremental kernel embedding management
//!
//! Follows the [`HilbertFeatureManager`] pattern: lazy recalculation
//! triggered by accumulated graph mutations, with debounce.
//!
//! ## Architecture
//!
//! ```text
//! LpgStore mutations
//!       │
//!       ▼
//!  SubscriptionManager ──callback──▶ KernelManager.pending_changes
//!       │                                       │
//!       │                                       ▼ (lazy, on get_embedding())
//!       │                            compute_incremental()
//!       │                                       │
//!       ▼                                       ▼
//!  GraphEvent log              Updated embeddings (HashMap cache)
//! ```
//!
//! ## Lifecycle
//!
//! 1. `KernelManager::new(store, phi)` — create with frozen Phi_0
//! 2. `enable()` — subscribe to graph events
//! 3. `compute_all()` — initial batch computation for all nodes
//! 4. Graph mutations accumulate in `pending_changes`
//! 5. `flush()` or `get_embedding()` triggers incremental recomputation

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use obrain_common::types::{NodeId, PropertyKey, Value};
use obrain_core::change_tracker::{EntityRef, GraphEvent};
use obrain_core::graph::lpg::LpgStore;
use obrain_core::graph::{Direction, GraphStore};
use obrain_core::subscription::{EventFilter, EventType, SubscriptionId};
use parking_lot::RwLock;

use super::kernel::{
    DEFAULT_ALPHA, DEFAULT_MAX_NEIGHBORS, KERNEL_EMBEDDING_KEY, MultiHeadPhi0,
    compute_batch_parallel, compute_node_embedding,
};
use super::kernel_math::Rng;
use super::kernel_train::{deserialize_phi0, serialize_phi0};

/// Label for the system config node storing Phi_0 weights.
const KERNEL_CONFIG_LABEL: &str = "_KernelConfig";
/// Property key for serialized Phi_0 weights on the config node.
const PHI_WEIGHTS_KEY: &str = "_phi_weights";
/// Property key for Phi state on the config node.
const PHI_STATE_KEY: &str = "_phi_state";

// ============================================================================
// Phi0 lifecycle state
// ============================================================================

/// State of Phi_0 weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhiState {
    /// Random initialization, not yet trained.
    Untrained,
    /// Training in progress.
    Training,
    /// Trained and frozen — ready for production.
    Frozen,
}

// ============================================================================
// KernelManager
// ============================================================================

/// Manages kernel embeddings with event-driven incremental updates.
///
/// Holds a frozen (or untrained) Phi_0, maintains an in-memory cache
/// of all node embeddings, and subscribes to graph events to keep
/// embeddings up-to-date incrementally.
///
/// # Thread safety
///
/// All mutable state is behind `parking_lot::RwLock`.
pub struct KernelManager {
    /// Reference to the graph store.
    store: Arc<LpgStore>,
    /// The kernel weights.
    phi: RwLock<MultiHeadPhi0>,
    /// Current state of Phi_0.
    phi_state: RwLock<PhiState>,
    /// APPNP anchoring factor.
    alpha: f64,
    /// Max neighbors before Fisher-Yates sampling.
    max_neighbors: usize,
    /// Number of threads for batch computation.
    n_threads: usize,
    /// PRNG seed.
    seed: u64,
    /// In-memory embedding cache: NodeId -> 80d f32 vector.
    embeddings: RwLock<HashMap<NodeId, Vec<f32>>>,
    /// Accumulated changed node IDs since last flush.
    pending_changes: Arc<RwLock<HashSet<NodeId>>>,
    /// Minimum pending changes before auto-flush in get_embedding().
    pub debounce_threshold: usize,
    /// Active subscription ID.
    subscription_id: RwLock<Option<SubscriptionId>>,
}

impl KernelManager {
    /// Create a new KernelManager with given Phi_0.
    ///
    /// Does NOT subscribe to events or compute embeddings.
    /// Call [`enable()`](Self::enable) then [`compute_all()`](Self::compute_all).
    pub fn new(store: Arc<LpgStore>, phi: MultiHeadPhi0, phi_state: PhiState) -> Self {
        Self {
            store,
            phi: RwLock::new(phi),
            phi_state: RwLock::new(phi_state),
            alpha: DEFAULT_ALPHA,
            max_neighbors: DEFAULT_MAX_NEIGHBORS,
            n_threads: num_cpus(),
            seed: 42,
            embeddings: RwLock::new(HashMap::new()),
            pending_changes: Arc::new(RwLock::new(HashSet::new())),
            debounce_threshold: 1,
            subscription_id: RwLock::new(None),
        }
    }

    /// Create with default random Phi_0 (untrained).
    pub fn new_untrained(store: Arc<LpgStore>, seed: u64) -> Self {
        Self::new(
            store,
            MultiHeadPhi0::default_with_seed(seed),
            PhiState::Untrained,
        )
    }

    // ── Configuration ──

    /// Set the APPNP anchoring factor.
    pub fn set_alpha(&mut self, alpha: f64) {
        self.alpha = alpha;
    }

    /// Set max neighbors for hub capping.
    pub fn set_max_neighbors(&mut self, max: usize) {
        self.max_neighbors = max;
    }

    /// Set number of threads for parallel computation.
    pub fn set_threads(&mut self, n: usize) {
        self.n_threads = n;
    }

    /// Set PRNG seed.
    pub fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
    }

    /// Replace Phi_0 weights (e.g. after training or deserialization).
    pub fn set_phi(&self, phi: MultiHeadPhi0, state: PhiState) {
        *self.phi.write() = phi;
        *self.phi_state.write() = state;
    }

    /// Get the current Phi state.
    pub fn phi_state(&self) -> PhiState {
        *self.phi_state.read()
    }

    /// Get a clone of the current Phi_0 weights.
    pub fn phi(&self) -> MultiHeadPhi0 {
        self.phi.read().clone()
    }

    // ── Persistence ──

    /// Persist Phi_0 weights to the graph store.
    ///
    /// Creates (or updates) a system node with label `_KernelConfig`
    /// containing the serialized weights as `Value::Bytes` and the
    /// Phi state as `Value::String`. This ensures Phi_0 survives
    /// snapshot + WAL round-trips.
    ///
    /// Call this after training or whenever you want to checkpoint weights.
    pub fn save_phi(&self) -> NodeId {
        let phi = self.phi.read().clone();
        let state = *self.phi_state.read();
        let bytes = serialize_phi0(&phi);
        let state_str = match state {
            PhiState::Untrained => "untrained",
            PhiState::Training => "training",
            PhiState::Frozen => "frozen",
        };

        // Find existing config node or create one
        let config_node = self
            .find_config_node()
            .unwrap_or_else(|| self.store.create_node(&[KERNEL_CONFIG_LABEL]));

        let arc_bytes: Arc<[u8]> = bytes.into();
        self.store
            .set_node_property(config_node, PHI_WEIGHTS_KEY, Value::Bytes(arc_bytes));
        self.store
            .set_node_property(config_node, PHI_STATE_KEY, Value::String(state_str.into()));

        config_node
    }

    /// Load Phi_0 weights from the graph store.
    ///
    /// Searches for the `_KernelConfig` system node and deserializes
    /// the stored weights. Returns `true` if weights were found and loaded.
    ///
    /// Call this on startup after loading a snapshot to restore Phi_0.
    pub fn load_phi(&self) -> bool {
        let Some(config_node) = self.find_config_node() else {
            return false;
        };

        let prop_key = PropertyKey::from(PHI_WEIGHTS_KEY);
        let Some(Value::Bytes(bytes)) = self.store.get_node_property(config_node, &prop_key) else {
            return false;
        };

        let Some(phi) = deserialize_phi0(&bytes) else {
            return false;
        };

        // Read state
        let state_key = PropertyKey::from(PHI_STATE_KEY);
        let state = match self.store.get_node_property(config_node, &state_key) {
            Some(Value::String(s)) => match s.as_str() {
                "frozen" => PhiState::Frozen,
                "training" => PhiState::Training,
                _ => PhiState::Untrained,
            },
            _ => PhiState::Untrained,
        };

        *self.phi.write() = phi;
        *self.phi_state.write() = state;
        true
    }

    /// Find the existing `_KernelConfig` system node, if any.
    fn find_config_node(&self) -> Option<NodeId> {
        self.store.node_ids().into_iter().find(|&nid| {
            self.store
                .get_node(nid)
                .is_some_and(|node| node.has_label(KERNEL_CONFIG_LABEL))
        })
    }

    // ── Event subscription ──

    /// Subscribe to graph mutation events.
    ///
    /// Accumulates affected node IDs in `pending_changes`.
    /// Safe to call multiple times (subsequent calls are no-ops).
    pub fn enable(&self) {
        let mut sub_id = self.subscription_id.write();
        if sub_id.is_some() {
            return;
        }

        let filter = EventFilter {
            event_types: Some(
                [
                    EventType::NodeCreated,
                    EventType::NodeDeleted,
                    EventType::EdgeCreated,
                    EventType::EdgeDeleted,
                    EventType::PropertySet,
                ]
                .into_iter()
                .collect(),
            ),
            ..Default::default()
        };

        let pending = Arc::clone(&self.pending_changes);
        let store_ref = Arc::clone(&self.store);
        let id = self.store.subscribe(
            filter,
            Box::new(move |event| {
                let affected = affected_nodes(event, &*store_ref);
                let mut changes = pending.write();
                for nid in affected {
                    changes.insert(nid);
                }
            }),
        );

        *sub_id = id;
    }

    /// Unsubscribe from graph events.
    pub fn disable(&self) {
        let mut sub_id = self.subscription_id.write();
        if let Some(id) = sub_id.take() {
            self.store.unsubscribe(id);
        }
    }

    /// Returns `true` if subscribed to events.
    pub fn is_enabled(&self) -> bool {
        self.subscription_id.read().is_some()
    }

    // ── Computation ──

    /// Compute embeddings for ALL nodes (initial batch).
    ///
    /// This is the initial computation — call once after setup.
    /// Writes `_kernel_embedding` on each node and populates the cache.
    pub fn compute_all(&self) {
        let phi = self.phi.read().clone();
        let node_ids = self.store.node_ids();

        let results = compute_batch_parallel(
            &phi,
            &*self.store,
            &node_ids,
            self.alpha,
            self.max_neighbors,
            self.n_threads,
            self.seed,
        );

        // Write embeddings to store and cache
        let mut cache = self.embeddings.write();
        for (nid, emb) in &results {
            let arc_vec: Arc<[f32]> = emb.as_slice().into();
            self.store
                .set_node_property(*nid, KERNEL_EMBEDDING_KEY, Value::Vector(arc_vec));
            cache.insert(*nid, emb.clone());
        }

        // Clear any pending changes accumulated during computation
        self.pending_changes.write().clear();
    }

    /// Compute embeddings incrementally for affected nodes.
    ///
    /// For each affected node + its 1-hop neighbors, recomputes the
    /// kernel embedding. Updates both the store and the cache.
    pub fn compute_incremental(&self, affected: &HashSet<NodeId>) {
        if affected.is_empty() {
            return;
        }

        let phi = self.phi.read().clone();
        let mut rng = Rng::new(self.seed);
        let mut cache = self.embeddings.write();

        for &nid in affected {
            if let Some(emb) = compute_node_embedding(
                &phi,
                &*self.store,
                nid,
                self.alpha,
                self.max_neighbors,
                &mut rng,
            ) {
                let arc_vec: Arc<[f32]> = emb.as_slice().into();
                self.store
                    .set_node_property(nid, KERNEL_EMBEDDING_KEY, Value::Vector(arc_vec));
                cache.insert(nid, emb);
            }
        }
    }

    /// Flush pending changes: expand to 1-hop neighborhoods, then recompute.
    ///
    /// This is the main incremental update entry point.
    pub fn flush(&self) {
        let changed: HashSet<NodeId> = self.pending_changes.write().drain().collect();
        if changed.is_empty() {
            return;
        }

        // Expand to 1-hop: each changed node + its neighbors need recomputation
        let mut affected = HashSet::new();
        for &nid in &changed {
            affected.insert(nid);
            for neighbor in self.store.neighbors(nid, Direction::Both) {
                affected.insert(neighbor);
            }
        }

        self.compute_incremental(&affected);

        // Clear any events that fired during recomputation (set_node_property triggers PropertySet)
        self.pending_changes.write().clear();
    }

    /// Recompute embeddings for a set of changed nodes.
    ///
    /// Expands each node to its 1-hop neighborhood, deduplicates, then
    /// recomputes all affected embeddings. This is the main entry point
    /// for external callers (e.g. `KernelListener` in `obrain-cognitive`).
    pub fn recompute_affected(&self, changed: &[NodeId]) {
        if changed.is_empty() {
            return;
        }

        let mut affected = HashSet::new();
        for &nid in changed {
            affected.insert(nid);
            for neighbor in self.store.neighbors(nid, Direction::Both) {
                affected.insert(neighbor);
            }
        }

        self.compute_incremental(&affected);

        // Clear any events that fired during recomputation
        self.pending_changes.write().clear();
    }

    // ── Query ──

    /// Get embedding for a node, triggering flush if debounce threshold reached.
    ///
    /// Returns `None` if the node has no embedding (not yet computed or no features).
    pub fn get_embedding(&self, node_id: NodeId) -> Option<Vec<f32>> {
        // Check if we need to flush
        let pending_count = self.pending_changes.read().len();
        if pending_count >= self.debounce_threshold {
            self.flush();
        }

        self.embeddings.read().get(&node_id).cloned()
    }

    /// Get embeddings for multiple nodes.
    pub fn get_embeddings(&self, node_ids: &[NodeId]) -> HashMap<NodeId, Vec<f32>> {
        let pending_count = self.pending_changes.read().len();
        if pending_count >= self.debounce_threshold {
            self.flush();
        }

        let cache = self.embeddings.read();
        node_ids
            .iter()
            .filter_map(|&nid| cache.get(&nid).map(|emb| (nid, emb.clone())))
            .collect()
    }

    /// Returns the number of cached embeddings.
    pub fn embedding_count(&self) -> usize {
        self.embeddings.read().len()
    }

    /// Returns the number of pending changes.
    pub fn pending_count(&self) -> usize {
        self.pending_changes.read().len()
    }

    /// Returns `true` if embeddings have been computed at least once.
    pub fn has_embeddings(&self) -> bool {
        !self.embeddings.read().is_empty()
    }

    /// Remove embedding for a deleted node.
    pub fn remove_embedding(&self, node_id: NodeId) {
        self.embeddings.write().remove(&node_id);
    }
}

// ============================================================================
// Event → affected nodes expansion
// ============================================================================

/// Extract affected node IDs from a graph event, including 1-hop neighbors.
///
/// - `NodeCreated/Updated`: the node itself
/// - `NodeDeleted`: the node itself (neighbors will be recomputed on flush)
/// - `EdgeCreated(src, dst)`: src and dst
/// - `EdgeDeleted`: edge ID only — limited info
/// - `PropertySet(Node)`: the node
fn affected_nodes(event: &GraphEvent, _store: &dyn GraphStore) -> Vec<NodeId> {
    match event {
        GraphEvent::NodeCreated { id, .. } => vec![*id],
        GraphEvent::NodeDeleted { id, .. } => vec![*id],
        GraphEvent::EdgeCreated { src, dst, .. } => vec![*src, *dst],
        GraphEvent::EdgeDeleted { .. } => {
            // EdgeDeleted doesn't carry src/dst in the current GraphEvent design.
            // The affected nodes will be picked up indirectly when their
            // properties or neighbors change.
            vec![]
        }
        GraphEvent::PropertySet { entity, .. } => match entity {
            EntityRef::Node(id) => vec![*id],
            EntityRef::Edge(_) => vec![],
        },
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Get approximate CPU count (fallback to 4 if unavailable).
fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::super::kernel::{D_MODEL, HILBERT_FEATURES_KEY};
    use super::*;
    use obrain_common::types::PropertyKey;

    /// Create a test store with tracking and subscriptions.
    fn test_store() -> Arc<LpgStore> {
        let mut store = LpgStore::new().unwrap();
        store.enable_tracking(1000);
        store.enable_subscriptions();
        Arc::new(store)
    }

    /// Create a simple graph with Hilbert features on each node.
    fn populate_graph_with_features(store: &LpgStore, n: usize) -> Vec<NodeId> {
        let mut rng = Rng::new(42);
        let mut nodes = Vec::with_capacity(n);

        for _ in 0..n {
            let nid = store.create_node(&["Node"]);
            // Generate random 80d features
            let features: Vec<f32> = (0..D_MODEL).map(|_| rng.next_f64() as f32).collect();
            let arc_vec: Arc<[f32]> = features.into();
            store.set_node_property(nid, HILBERT_FEATURES_KEY, Value::Vector(arc_vec));
            nodes.push(nid);
        }

        // Create edges: chain + some cross-links
        for i in 0..n.saturating_sub(1) {
            store.create_edge(nodes[i], nodes[i + 1], "LINK");
        }
        if n > 3 {
            store.create_edge(nodes[0], nodes[n - 1], "LINK");
        }

        nodes
    }

    #[test]
    fn test_new_untrained() {
        let store = test_store();
        let manager = KernelManager::new_untrained(store, 42);
        assert_eq!(manager.phi_state(), PhiState::Untrained);
        assert!(!manager.has_embeddings());
        assert_eq!(manager.embedding_count(), 0);
    }

    #[test]
    fn test_compute_all() {
        let store = test_store();
        let nodes = populate_graph_with_features(&store, 10);
        let manager = KernelManager::new_untrained(Arc::clone(&store), 42);

        manager.compute_all();

        assert!(manager.has_embeddings());
        assert_eq!(manager.embedding_count(), 10);

        // Check each node has an embedding
        for &nid in &nodes {
            let emb = manager.get_embedding(nid);
            assert!(emb.is_some(), "node {:?} should have embedding", nid);
            assert_eq!(emb.unwrap().len(), D_MODEL);
        }

        // Check embeddings are also persisted on the store
        for &nid in &nodes {
            let prop = store.get_node_property(nid, &PropertyKey::from(KERNEL_EMBEDDING_KEY));
            assert!(
                prop.is_some(),
                "node {:?} should have _kernel_embedding property",
                nid
            );
        }
    }

    #[test]
    fn test_compute_all_deterministic() {
        let store = test_store();
        populate_graph_with_features(&store, 8);

        let m1 = KernelManager::new_untrained(Arc::clone(&store), 42);
        m1.compute_all();

        let m2 = KernelManager::new_untrained(Arc::clone(&store), 42);
        m2.compute_all();

        let cache1 = m1.embeddings.read();
        let cache2 = m2.embeddings.read();
        assert_eq!(cache1.len(), cache2.len());
        for (nid, emb1) in cache1.iter() {
            let emb2 = cache2.get(nid).expect("same nodes");
            assert_eq!(emb1, emb2, "embeddings should be deterministic");
        }
    }

    #[test]
    fn test_compute_incremental() {
        let store = test_store();
        let nodes = populate_graph_with_features(&store, 10);
        let manager = KernelManager::new_untrained(Arc::clone(&store), 42);
        manager.compute_all();

        // Save original embeddings
        let original: HashMap<NodeId, Vec<f32>> = manager.embeddings.read().clone();

        // Recompute only nodes[0] and its neighbors
        let mut affected = HashSet::new();
        affected.insert(nodes[0]);
        for neighbor in store.neighbors(nodes[0], Direction::Both) {
            affected.insert(neighbor);
        }

        manager.compute_incremental(&affected);

        // Affected nodes may have same or different embeddings
        // (same because features didn't change, but the point is they were recomputed)
        assert_eq!(manager.embedding_count(), 10); // count unchanged

        // Non-affected nodes should be unchanged
        let updated = manager.embeddings.read();
        for &nid in &nodes {
            if !affected.contains(&nid) {
                assert_eq!(
                    original.get(&nid),
                    updated.get(&nid),
                    "non-affected node embedding should be unchanged"
                );
            }
        }
    }

    #[test]
    fn test_enable_subscribe() {
        let store = test_store();
        populate_graph_with_features(&store, 5);
        let manager = KernelManager::new_untrained(Arc::clone(&store), 42);
        manager.compute_all();

        manager.enable();
        assert!(manager.is_enabled());
        assert_eq!(manager.pending_count(), 0);

        // Add a new node — should trigger event
        let new_node = store.create_node(&["New"]);
        let features: Vec<f32> = vec![0.1; D_MODEL];
        let arc_vec: Arc<[f32]> = features.into();
        store.set_node_property(new_node, HILBERT_FEATURES_KEY, Value::Vector(arc_vec));

        // Pending changes should have accumulated
        // (NodeCreated + PropertySet = 2 events, but same node deduplicated)
        assert!(manager.pending_count() > 0);

        manager.disable();
        assert!(!manager.is_enabled());
    }

    #[test]
    fn test_flush() {
        let store = test_store();
        let nodes = populate_graph_with_features(&store, 8);
        let manager = KernelManager::new_untrained(Arc::clone(&store), 42);
        manager.compute_all();
        manager.enable();

        // Mutate: add a new edge
        store.create_edge(nodes[0], nodes[5], "NEW_LINK");

        // Flush pending changes
        manager.flush();
        assert_eq!(manager.pending_count(), 0);
    }

    #[test]
    fn test_get_embedding_auto_flush() {
        let store = test_store();
        let nodes = populate_graph_with_features(&store, 6);
        let mut manager = KernelManager::new_untrained(Arc::clone(&store), 42);
        manager.debounce_threshold = 1;
        manager.compute_all();
        manager.enable();

        // Add a property change
        let features: Vec<f32> = vec![0.5; D_MODEL];
        let arc_vec: Arc<[f32]> = features.into();
        store.set_node_property(nodes[0], HILBERT_FEATURES_KEY, Value::Vector(arc_vec));

        // get_embedding should auto-flush because pending >= threshold
        let emb = manager.get_embedding(nodes[0]);
        assert!(emb.is_some());
        assert_eq!(manager.pending_count(), 0); // flushed
    }

    #[test]
    fn test_set_phi() {
        let store = test_store();
        let manager = KernelManager::new_untrained(Arc::clone(&store), 42);
        assert_eq!(manager.phi_state(), PhiState::Untrained);

        let trained_phi = MultiHeadPhi0::default_with_seed(123);
        manager.set_phi(trained_phi, PhiState::Frozen);
        assert_eq!(manager.phi_state(), PhiState::Frozen);
    }

    #[test]
    fn test_get_embeddings_batch() {
        let store = test_store();
        let nodes = populate_graph_with_features(&store, 5);
        let manager = KernelManager::new_untrained(Arc::clone(&store), 42);
        manager.compute_all();

        let batch = manager.get_embeddings(&nodes[0..3]);
        assert_eq!(batch.len(), 3);
    }

    #[test]
    fn test_remove_embedding() {
        let store = test_store();
        let nodes = populate_graph_with_features(&store, 5);
        let manager = KernelManager::new_untrained(Arc::clone(&store), 42);
        manager.compute_all();
        assert_eq!(manager.embedding_count(), 5);

        manager.remove_embedding(nodes[0]);
        assert_eq!(manager.embedding_count(), 4);
        assert!(manager.get_embedding(nodes[0]).is_none());
    }

    #[test]
    fn test_node_without_features_skipped() {
        let store = test_store();
        // Create nodes without Hilbert features
        let n0 = store.create_node(&["NoFeatures"]);
        let n1 = store.create_node(&["NoFeatures"]);
        store.create_edge(n0, n1, "LINK");

        let manager = KernelManager::new_untrained(Arc::clone(&store), 42);
        manager.compute_all();

        // No embeddings because no Hilbert features
        assert_eq!(manager.embedding_count(), 0);
    }

    #[test]
    fn test_affected_nodes_edge_created() {
        let store = test_store();
        let n0 = store.create_node(&["A"]);
        let n1 = store.create_node(&["B"]);

        let event = GraphEvent::EdgeCreated {
            src: n0,
            dst: n1,
            id: obrain_common::types::EdgeId(0),
            edge_type: "LINK".into(),
            timestamp: 0,
        };

        let affected = affected_nodes(&event, &*store);
        assert!(affected.contains(&n0));
        assert!(affected.contains(&n1));
    }

    #[test]
    fn test_affected_nodes_property_set() {
        let store = test_store();
        let n0 = store.create_node(&["A"]);

        let event = GraphEvent::PropertySet {
            entity: EntityRef::Node(n0),
            key: "test".into(),
            old_value: None,
            new_value: Value::Int64(42),
            timestamp: 0,
        };

        let affected = affected_nodes(&event, &*store);
        assert_eq!(affected, vec![n0]);
    }

    #[test]
    fn test_save_phi_creates_config_node() {
        let store = test_store();
        populate_graph_with_features(&store, 5);
        let manager = KernelManager::new_untrained(Arc::clone(&store), 42);

        let config_node = manager.save_phi();

        // Config node should exist with correct label
        let node = store.get_node(config_node).unwrap();
        assert!(node.has_label("_KernelConfig"));

        // Should have the weights property
        let key = PropertyKey::from("_phi_weights");
        let prop = store.get_node_property(config_node, &key);
        assert!(matches!(prop, Some(Value::Bytes(_))));

        // Should have the state property
        let state_key = PropertyKey::from("_phi_state");
        let state_prop = store.get_node_property(config_node, &state_key);
        assert!(matches!(state_prop, Some(Value::String(s)) if s.as_str() == "untrained"));
    }

    #[test]
    fn test_save_phi_idempotent() {
        let store = test_store();
        populate_graph_with_features(&store, 3);
        let manager = KernelManager::new_untrained(Arc::clone(&store), 42);

        let node1 = manager.save_phi();
        let node2 = manager.save_phi();

        // Should reuse the same config node
        assert_eq!(node1, node2);
    }

    #[test]
    fn test_load_phi_roundtrip() {
        let store = test_store();
        populate_graph_with_features(&store, 5);

        // Create manager with frozen Phi and save
        let phi_original = MultiHeadPhi0::default_with_seed(99);
        let m1 = KernelManager::new(Arc::clone(&store), phi_original.clone(), PhiState::Frozen);
        m1.save_phi();

        // New manager — loads from store
        let m2 = KernelManager::new_untrained(Arc::clone(&store), 42);
        assert_eq!(m2.phi_state(), PhiState::Untrained);
        assert!(m2.load_phi());
        assert_eq!(m2.phi_state(), PhiState::Frozen);

        // Weights should match
        let phi_loaded = m2.phi();
        assert_eq!(
            phi_original.serialize_weights(),
            phi_loaded.serialize_weights(),
            "round-tripped Phi_0 weights must be identical"
        );
    }

    #[test]
    fn test_load_phi_no_config_node() {
        let store = test_store();
        let manager = KernelManager::new_untrained(Arc::clone(&store), 42);

        // No config node exists
        assert!(!manager.load_phi());
        assert_eq!(manager.phi_state(), PhiState::Untrained);
    }

    #[test]
    fn test_save_load_compute_embeddings_match() {
        let store = test_store();
        populate_graph_with_features(&store, 8);

        // Train (untrained but with fixed seed), compute, save
        let m1 = KernelManager::new_untrained(Arc::clone(&store), 42);
        m1.compute_all();
        m1.save_phi();
        let emb1: HashMap<NodeId, Vec<f32>> = m1.embeddings.read().clone();

        // New manager, load phi, compute
        let m2 = KernelManager::new_untrained(Arc::clone(&store), 42);
        assert!(m2.load_phi());
        m2.compute_all();
        let emb2: HashMap<NodeId, Vec<f32>> = m2.embeddings.read().clone();

        // Same phi + same graph = same embeddings
        assert_eq!(emb1.len(), emb2.len());
        for (nid, e1) in &emb1 {
            let e2 = emb2.get(nid).expect("same nodes");
            assert_eq!(e1, e2, "embeddings should match after phi roundtrip");
        }
    }
}
