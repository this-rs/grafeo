//! Incremental Hilbert feature manager with event-driven recalculation.
//!
//! Bridges the [`ChangeTracker`](obrain_core::change_tracker::ChangeTracker) /
//! [`SubscriptionManager`](obrain_core::subscription::SubscriptionManager) with
//! [`hilbert_features_incremental()`](super::hilbert_features::hilbert_features_incremental)
//! to automatically keep Hilbert features up-to-date as the graph mutates.
//!
//! ## Architecture
//!
//! ```text
//! LpgStore mutations
//!       │
//!       ▼
//!  SubscriptionManager ──callback──▶ HilbertFeatureManager.pending_changes
//!       │                                       │
//!       │                                       ▼ (lazy, on get_features())
//!       │                            hilbert_features_incremental()
//!       │                                       │
//!       ▼                                       ▼
//!  GraphEvent log              Updated HilbertFeaturesResult (cached)
//! ```
//!
//! The manager uses a **lazy recalculation** pattern with debounce:
//!
//! - Events are accumulated in `pending_changes` (deduplicated `HashSet<NodeId>`)
//! - `get_features()` triggers an incremental recalc only if `pending_changes.len() >= debounce_threshold`
//! - `force_full_recalc()` does a full recomputation and clears `dirty_global`
//!
//! ## Usage
//!
//! ```no_run
//! use obrain_core::graph::lpg::LpgStore;
//! use obrain_adapters::plugins::algorithms::hilbert_manager::HilbertFeatureManager;
//! use obrain_adapters::plugins::algorithms::hilbert_features::HilbertFeaturesConfig;
//! use std::sync::Arc;
//!
//! let mut store = LpgStore::new().unwrap();
//! store.enable_tracking(1000);
//! store.enable_subscriptions();
//!
//! let config = HilbertFeaturesConfig::default();
//! let manager = HilbertFeatureManager::new(Arc::new(store), config);
//! manager.enable();
//!
//! // ... mutations on the store ...
//! // manager.get_features() will auto-recalculate when threshold is reached
//! ```

use std::collections::HashSet;
use std::sync::Arc;

use obrain_common::types::NodeId;
use obrain_core::change_tracker::{EntityRef, GraphEvent};
use obrain_core::graph::lpg::LpgStore;
use obrain_core::subscription::{EventFilter, EventType, SubscriptionId};
use parking_lot::RwLock;

use super::hilbert_features::{
    HilbertFeaturesConfig, HilbertFeaturesResult, hilbert_features, hilbert_features_incremental,
};

// ============================================================================
// HilbertFeatureManager
// ============================================================================

/// Manages incremental Hilbert feature recalculation driven by graph events.
///
/// Subscribes to [`LpgStore`] mutations via the [`SubscriptionManager`] and
/// accumulates changed node IDs. On [`get_features()`](Self::get_features),
/// triggers a lazy incremental recalculation if the number of pending changes
/// exceeds the [`debounce_threshold`](Self::debounce_threshold).
///
/// # Thread safety
///
/// All internal state is behind `parking_lot::RwLock`. The subscription callback
/// only acquires a write lock on `pending_changes` (fast path: `HashSet::insert`).
pub struct HilbertFeatureManager {
    /// Reference to the graph store.
    store: Arc<LpgStore>,
    /// Configuration for Hilbert feature computation.
    config: HilbertFeaturesConfig,
    /// Cached feature result (None = never computed).
    cached: RwLock<Option<HilbertFeaturesResult>>,
    /// Accumulated changed node IDs since last recalculation.
    pending_changes: Arc<RwLock<HashSet<NodeId>>>,
    /// Minimum number of pending changes before triggering a recalc.
    ///
    /// Default: 1 (recalculate on any change). Set higher to batch changes.
    pub debounce_threshold: usize,
    /// Active subscription ID (None if not enabled).
    subscription_id: RwLock<Option<SubscriptionId>>,
}

impl HilbertFeatureManager {
    /// Creates a new manager.
    ///
    /// Does **not** subscribe to events or compute features — call [`enable()`](Self::enable)
    /// to start receiving events, and [`get_features()`](Self::get_features) to trigger
    /// the first computation.
    ///
    /// # Arguments
    ///
    /// * `store` - Arc to the LPG store (must have tracking and subscriptions enabled)
    /// * `config` - Hilbert features configuration
    pub fn new(store: Arc<LpgStore>, config: HilbertFeaturesConfig) -> Self {
        Self {
            store,
            config,
            cached: RwLock::new(None),
            pending_changes: Arc::new(RwLock::new(HashSet::new())),
            debounce_threshold: 1,
            subscription_id: RwLock::new(None),
        }
    }

    /// Subscribe to graph mutation events.
    ///
    /// Registers a callback that accumulates changed [`NodeId`]s in the
    /// `pending_changes` buffer. Events filtered to: `NodeCreated`, `NodeDeleted`,
    /// `EdgeCreated`, `EdgeDeleted`, `PropertySet`.
    ///
    /// Calling `enable()` multiple times is safe — subsequent calls are no-ops.
    pub fn enable(&self) {
        let mut sub_id = self.subscription_id.write();
        if sub_id.is_some() {
            return; // Already enabled
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
        let id = self.store.subscribe(
            filter,
            Box::new(move |event| {
                let mut changes = pending.write();
                for node_id in extract_changed_nodes(event) {
                    changes.insert(node_id);
                }
            }),
        );

        *sub_id = id;
    }

    /// Unsubscribe from graph mutation events.
    ///
    /// After calling `disable()`, mutations are no longer tracked. Pending changes
    /// are preserved — a subsequent `get_features()` will still process them.
    pub fn disable(&self) {
        let mut sub_id = self.subscription_id.write();
        if let Some(id) = sub_id.take() {
            self.store.unsubscribe(id);
        }
    }

    /// Returns `true` if the manager is currently subscribed to events.
    pub fn is_enabled(&self) -> bool {
        self.subscription_id.read().is_some()
    }

    /// Get the current Hilbert features, triggering an incremental recalculation
    /// if pending changes exceed the debounce threshold.
    ///
    /// - **First call**: performs a full `hilbert_features()` computation.
    /// - **Subsequent calls**: if `pending_changes.len() >= debounce_threshold`,
    ///   runs `hilbert_features_incremental()` with the accumulated changed nodes.
    /// - **No pending changes**: returns the cached result immediately.
    ///
    /// # Returns
    ///
    /// A clone of the cached [`HilbertFeaturesResult`]. The `dirty_global` flag
    /// is `true` after an incremental update (global facettes 0-3 are stale).
    pub fn get_features(&self) -> HilbertFeaturesResult {
        // Fast path: check if we need to recalculate
        let pending_count = self.pending_changes.read().len();

        let mut cached = self.cached.write();

        match cached.as_ref() {
            None => {
                // First call: full computation
                let result = hilbert_features(&*self.store, &self.config);
                self.pending_changes.write().clear();
                let cloned = result.clone();
                *cached = Some(result);
                cloned
            }
            Some(previous) if pending_count >= self.debounce_threshold => {
                // Incremental recalculation
                let changed: Vec<NodeId> = self.pending_changes.write().drain().collect();
                let result =
                    hilbert_features_incremental(&*self.store, &self.config, previous, &changed);
                let cloned = result.clone();
                *cached = Some(result);
                cloned
            }
            Some(existing) => {
                // No changes or below threshold: return cached
                existing.clone()
            }
        }
    }

    /// Force a full recalculation of all features.
    ///
    /// Clears `dirty_global` and recomputes all 8 facettes (global + local).
    /// Use this after an incremental update has set `dirty_global = true` and
    /// you need accurate global facettes (spectral, community, centrality).
    pub fn force_full_recalc(&self) {
        let result = hilbert_features(&*self.store, &self.config);
        self.pending_changes.write().clear();
        *self.cached.write() = Some(result);
    }

    /// Returns the number of pending (unprocessed) changes.
    pub fn pending_count(&self) -> usize {
        self.pending_changes.read().len()
    }

    /// Returns `true` if features have been computed at least once.
    pub fn has_features(&self) -> bool {
        self.cached.read().is_some()
    }
}

// ============================================================================
// Event → NodeId extraction
// ============================================================================

/// Extract all affected [`NodeId`]s from a [`GraphEvent`].
///
/// - `NodeCreated` / `NodeDeleted` → the node itself
/// - `EdgeCreated` / `EdgeDeleted` → both source and destination nodes
/// - `PropertySet` on a `Node` → the node; on an `Edge` → ignored (edge
///   property changes don't affect Hilbert node features directly)
fn extract_changed_nodes(event: &GraphEvent) -> Vec<NodeId> {
    match event {
        GraphEvent::NodeCreated { id, .. } => vec![*id],
        GraphEvent::NodeDeleted { id, .. } => vec![*id],
        GraphEvent::EdgeCreated { src, dst, .. } => vec![*src, *dst],
        GraphEvent::EdgeDeleted { .. } => {
            // EdgeDeleted only has the edge ID, not src/dst.
            // The incremental recalc will pick up structural changes via
            // the next NodeCreated/Deleted or PropertySet event, or via
            // force_full_recalc(). This is a known limitation of the
            // current GraphEvent design.
            vec![]
        }
        GraphEvent::PropertySet { entity, .. } => match entity {
            EntityRef::Node(id) => vec![*id],
            EntityRef::Edge(_) => vec![], // Edge properties don't affect node features
        },
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    /// Helper: create a store with tracking and subscriptions enabled.
    fn test_store() -> Arc<LpgStore> {
        let mut store = LpgStore::new().unwrap();
        store.enable_tracking(1000);
        store.enable_subscriptions();
        Arc::new(store)
    }

    /// Helper: build a small triangle graph.
    fn populate_triangle(store: &LpgStore) -> (NodeId, NodeId, NodeId) {
        let n0 = store.create_node(&["A"]);
        let n1 = store.create_node(&["B"]);
        let n2 = store.create_node(&["C"]);
        store.create_edge(n0, n1, "LINK");
        store.create_edge(n1, n2, "LINK");
        store.create_edge(n2, n0, "LINK");
        (n0, n1, n2)
    }

    #[test]
    fn test_no_initial_features() {
        let store = test_store();
        populate_triangle(&store);

        let manager = HilbertFeatureManager::new(store, HilbertFeaturesConfig::default());

        assert!(!manager.has_features());

        // First get_features() does a full computation
        let features = manager.get_features();
        assert!(manager.has_features());
        assert_eq!(features.features.len(), 3); // 3 nodes
        assert_eq!(features.dimensions, 64); // 8 facettes × 8 levels
        assert!(!features.dirty_global);
    }

    #[test]
    fn test_auto_recalc() {
        let store = test_store();
        populate_triangle(&store);

        let manager =
            HilbertFeatureManager::new(Arc::clone(&store), HilbertFeaturesConfig::default());
        manager.enable();

        // Initial computation
        let features1 = manager.get_features();
        assert_eq!(features1.features.len(), 3);

        // Mutate the graph — add a new node + edge
        let n3 = store.create_node(&["D"]);
        store.create_edge(n3, NodeId(0), "LINK");

        // Pending changes should be accumulated
        assert!(manager.pending_count() > 0);

        // get_features() triggers incremental recalc
        let features2 = manager.get_features();
        assert_eq!(features2.features.len(), 4); // Now 4 nodes
        assert!(features2.dirty_global); // Incremental → global stale
        assert_eq!(manager.pending_count(), 0); // Changes consumed
    }

    #[test]
    fn test_debounce() {
        let store = test_store();
        populate_triangle(&store);

        let mut manager =
            HilbertFeatureManager::new(Arc::clone(&store), HilbertFeaturesConfig::default());
        manager.debounce_threshold = 5; // High threshold
        manager.enable();

        // Initial computation
        let _ = manager.get_features();

        // Add 1 node → below threshold
        store.create_node(&["X"]);
        assert!(manager.pending_count() > 0);
        assert!(manager.pending_count() < 5);

        // get_features() should return cached (no recalc)
        let features = manager.get_features();
        assert_eq!(features.features.len(), 3); // Still 3 — not recalculated
        assert!(!features.dirty_global);
    }

    #[test]
    fn test_force_full_recalc() {
        let store = test_store();
        populate_triangle(&store);

        let manager =
            HilbertFeatureManager::new(Arc::clone(&store), HilbertFeaturesConfig::default());
        manager.enable();

        // Initial + mutation
        let _ = manager.get_features();
        store.create_node(&["D"]);
        let features = manager.get_features();
        assert!(features.dirty_global);

        // Force full recalc clears dirty_global
        manager.force_full_recalc();
        let features = manager.get_features();
        assert!(!features.dirty_global);
    }

    #[test]
    fn test_disable() {
        let store = test_store();
        populate_triangle(&store);

        let manager =
            HilbertFeatureManager::new(Arc::clone(&store), HilbertFeaturesConfig::default());
        manager.enable();
        assert!(manager.is_enabled());

        let _ = manager.get_features();

        // Disable subscriptions
        manager.disable();
        assert!(!manager.is_enabled());

        // Mutations after disable are NOT tracked
        store.create_node(&["E"]);
        assert_eq!(manager.pending_count(), 0); // No pending changes

        // Features stay at 3 nodes
        let features = manager.get_features();
        assert_eq!(features.features.len(), 3);
    }

    #[test]
    fn test_extract_changed_nodes() {
        // NodeCreated
        let nodes = extract_changed_nodes(&GraphEvent::NodeCreated {
            id: NodeId(1),
            labels: vec!["A".to_string()],
            timestamp: 0,
        });
        assert_eq!(nodes, vec![NodeId(1)]);

        // NodeDeleted
        let nodes = extract_changed_nodes(&GraphEvent::NodeDeleted {
            id: NodeId(2),
            timestamp: 0,
        });
        assert_eq!(nodes, vec![NodeId(2)]);

        // EdgeCreated → both src and dst
        let nodes = extract_changed_nodes(&GraphEvent::EdgeCreated {
            id: obrain_common::types::EdgeId(10),
            src: NodeId(1),
            dst: NodeId(2),
            edge_type: "LINK".to_string(),
            timestamp: 0,
        });
        assert_eq!(nodes, vec![NodeId(1), NodeId(2)]);

        // EdgeDeleted → empty (no src/dst in event)
        let nodes = extract_changed_nodes(&GraphEvent::EdgeDeleted {
            id: obrain_common::types::EdgeId(10),
            timestamp: 0,
        });
        assert!(nodes.is_empty());

        // PropertySet on Node
        let nodes = extract_changed_nodes(&GraphEvent::PropertySet {
            entity: EntityRef::Node(NodeId(5)),
            key: "name".into(),
            old_value: None,
            new_value: obrain_common::types::Value::Int64(42),
            timestamp: 0,
        });
        assert_eq!(nodes, vec![NodeId(5)]);

        // PropertySet on Edge → empty
        let nodes = extract_changed_nodes(&GraphEvent::PropertySet {
            entity: EntityRef::Edge(obrain_common::types::EdgeId(99)),
            key: "weight".into(),
            old_value: None,
            new_value: obrain_common::types::Value::Float64(1.0),
            timestamp: 0,
        });
        assert!(nodes.is_empty());
    }

    #[test]
    fn test_enable_idempotent() {
        let store = test_store();
        let manager = HilbertFeatureManager::new(store, HilbertFeaturesConfig::default());

        manager.enable();
        assert!(manager.is_enabled());

        // Second enable is a no-op
        manager.enable();
        assert!(manager.is_enabled());
    }
}
