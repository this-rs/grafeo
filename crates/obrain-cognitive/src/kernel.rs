//! # KernelListener — MutationBus integration for kernel embeddings
//!
//! Bridges the reactive mutation pipeline (`MutationBus`)
//! to the [`KernelManager`]
//! for incremental kernel embedding updates.
//!
//! ## Architecture
//!
//! ```text
//! MutationBus  ──▶  Scheduler  ──▶  KernelListener.on_batch()
//!                                          │
//!                                          ▼
//!                                   KernelManager.recompute_affected(nodes)
//!                                          │
//!                                          ├── expand to 1-hop neighborhoods
//!                                          ├── single_pass_attention() per node
//!                                          └── write _kernel_embedding on store
//! ```
//!
//! ## Usage
//!
//! ```rust,no_run
//! use obrain_cognitive::kernel::KernelListener;
//! use obrain_adapters::plugins::algorithms::{KernelManager, PhiState};
//! use std::sync::Arc;
//!
//! // manager already created with Arc<LpgStore> + Phi0
//! # fn example(manager: Arc<KernelManager>) {
//! let listener = Arc::new(KernelListener::new(Arc::clone(&manager)));
//! // Register with scheduler: scheduler.register_listener(listener);
//! # }
//! ```

use std::collections::HashSet;
use std::sync::Arc;

use async_trait::async_trait;
use obrain_adapters::plugins::algorithms::KernelManager;
use obrain_common::types::NodeId;
use obrain_reactive::{MutationEvent, MutationListener};
use smallvec::SmallVec;

// ============================================================================
// KernelListener
// ============================================================================

/// A [`MutationListener`] that triggers incremental kernel embedding updates.
///
/// On each batch of mutations, extracts the affected node IDs (deduped),
/// and delegates to [`KernelManager::recompute_affected`] which expands
/// to 1-hop neighborhoods and recomputes embeddings via single-pass attention.
///
/// # Performance
///
/// Batch processing is efficient: all events are scanned once to collect
/// unique node IDs, then a single `recompute_affected` call handles the
/// 1-hop expansion and recomputation (~0.14ms per node).
pub struct KernelListener {
    /// The kernel manager that owns Phi_0 and the embedding cache.
    manager: Arc<KernelManager>,
}

impl KernelListener {
    /// Creates a new kernel listener backed by the given manager.
    ///
    /// The manager must already be initialized (with Phi_0 weights and
    /// optionally a full `compute_all()` pass).
    pub fn new(manager: Arc<KernelManager>) -> Self {
        Self { manager }
    }

    /// Returns a reference to the underlying manager.
    pub fn manager(&self) -> &Arc<KernelManager> {
        &self.manager
    }

    /// Extracts all node IDs affected by a mutation event.
    ///
    /// - `NodeCreated/Updated/Deleted`: the node itself
    /// - `EdgeCreated/Updated/Deleted`: both src and dst
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
impl MutationListener for KernelListener {
    fn name(&self) -> &str {
        "cognitive:kernel"
    }

    async fn on_event(&self, event: &MutationEvent) {
        let affected: Vec<NodeId> = Self::affected_nodes(event).to_vec();
        self.manager.recompute_affected(&affected);
    }

    async fn on_batch(&self, events: &[MutationEvent]) {
        // Deduplicate all affected node IDs across the batch
        let mut affected_set = HashSet::new();
        for event in events {
            for nid in Self::affected_nodes(event) {
                affected_set.insert(nid);
            }
        }

        if affected_set.is_empty() {
            return;
        }

        let affected: Vec<NodeId> = affected_set.into_iter().collect();
        self.manager.recompute_affected(&affected);
    }

    fn accepts(&self, event: &MutationEvent) -> bool {
        // Accept all node and edge mutations — they all potentially
        // affect neighborhood structure and thus kernel embeddings.
        matches!(
            event,
            MutationEvent::NodeCreated { .. }
                | MutationEvent::NodeUpdated { .. }
                | MutationEvent::NodeDeleted { .. }
                | MutationEvent::EdgeCreated { .. }
                | MutationEvent::EdgeUpdated { .. }
                | MutationEvent::EdgeDeleted { .. }
        )
    }
}

impl std::fmt::Debug for KernelListener {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KernelListener")
            .field("has_embeddings", &self.manager.has_embeddings())
            .field("embedding_count", &self.manager.embedding_count())
            .finish()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use obrain_adapters::plugins::algorithms::kernel::{D_MODEL, HILBERT_FEATURES_KEY};
    use obrain_adapters::plugins::algorithms::kernel_math::Rng;
    use obrain_common::types::Value;
    use obrain_core::graph::GraphStoreMut;
    use obrain_reactive::{EdgeSnapshot, NodeSnapshot};
    use obrain_substrate::SubstrateStore;

    /// Create a test store (SubstrateStore — T17 W2c migration from LpgStore).
    fn test_store() -> Arc<dyn GraphStoreMut> {
        Arc::new(SubstrateStore::open_tempfile().unwrap())
    }

    /// Create a graph with Hilbert features.
    fn populate_graph(store: &Arc<dyn GraphStoreMut>, n: usize) -> Vec<NodeId> {
        let mut rng = Rng::new(42);
        let mut nodes = Vec::with_capacity(n);

        for _ in 0..n {
            let nid = store.create_node(&["Node"]);
            let features: Vec<f32> = (0..D_MODEL).map(|_| rng.next_f64() as f32).collect();
            let arc_vec: Arc<[f32]> = features.into();
            store.set_node_property(nid, HILBERT_FEATURES_KEY, Value::Vector(arc_vec));
            nodes.push(nid);
        }

        for i in 0..n.saturating_sub(1) {
            store.create_edge(nodes[i], nodes[i + 1], "LINK");
        }
        if n > 3 {
            store.create_edge(nodes[0], nodes[n - 1], "LINK");
        }

        nodes
    }

    fn make_manager(store: Arc<dyn GraphStoreMut>) -> Arc<KernelManager> {
        let manager = KernelManager::new_untrained(store, 42);
        manager.compute_all();
        Arc::new(manager)
    }

    #[tokio::test]
    async fn test_listener_name() {
        let store = test_store();
        let manager = Arc::new(KernelManager::new_untrained(store, 42));
        let listener = KernelListener::new(manager);
        assert_eq!(listener.name(), "cognitive:kernel");
    }

    #[tokio::test]
    async fn test_on_event_node_created() {
        let store = test_store();
        let nodes = populate_graph(&store, 6);
        let manager = make_manager(Arc::clone(&store));

        let _emb_before = manager.get_embedding(nodes[0]).unwrap();

        let listener = KernelListener::new(Arc::clone(&manager));

        // Simulate a new node created adjacent to nodes[0]
        let new_node = store.create_node(&["New"]);
        let features: Vec<f32> = vec![0.5; D_MODEL];
        let arc_vec: Arc<[f32]> = features.into();
        store.set_node_property(new_node, HILBERT_FEATURES_KEY, Value::Vector(arc_vec));
        store.create_edge(new_node, nodes[0], "LINK");

        // Fire event for the new node
        let event = MutationEvent::NodeCreated {
            node: NodeSnapshot {
                id: new_node,
                labels: smallvec::smallvec!["New".into()],
                properties: vec![],
            },
        };

        listener.on_event(&event).await;

        // The new node should now have an embedding
        let new_emb = manager.get_embedding(new_node);
        assert!(
            new_emb.is_some(),
            "new node should have embedding after on_event"
        );
    }

    #[tokio::test]
    async fn test_on_batch_deduplicates() {
        let store = test_store();
        let nodes = populate_graph(&store, 8);
        let manager = make_manager(Arc::clone(&store));

        let listener = KernelListener::new(Arc::clone(&manager));

        // Create a batch with duplicate node references
        let events = vec![
            MutationEvent::NodeUpdated {
                before: NodeSnapshot {
                    id: nodes[0],
                    labels: smallvec::smallvec!["Node".into()],
                    properties: vec![],
                },
                after: NodeSnapshot {
                    id: nodes[0],
                    labels: smallvec::smallvec!["Node".into()],
                    properties: vec![],
                },
            },
            MutationEvent::EdgeCreated {
                edge: EdgeSnapshot {
                    id: obrain_common::types::EdgeId(999),
                    src: nodes[0],
                    dst: nodes[3],
                    edge_type: "NEW".into(),
                    properties: vec![],
                },
            },
        ];

        listener.on_batch(&events).await;

        // Should still have all embeddings (no crash, no corruption)
        assert_eq!(manager.embedding_count(), 8);
    }

    #[tokio::test]
    async fn test_accepts_all_mutations() {
        let store = test_store();
        let manager = Arc::new(KernelManager::new_untrained(store, 42));
        let listener = KernelListener::new(manager);

        let node_event = MutationEvent::NodeCreated {
            node: NodeSnapshot {
                id: NodeId(0),
                labels: smallvec::smallvec!["A".into()],
                properties: vec![],
            },
        };
        assert!(listener.accepts(&node_event));

        let edge_event = MutationEvent::EdgeDeleted {
            edge: EdgeSnapshot {
                id: obrain_common::types::EdgeId(0),
                src: NodeId(0),
                dst: NodeId(1),
                edge_type: "X".into(),
                properties: vec![],
            },
        };
        assert!(listener.accepts(&edge_event));
    }

    #[tokio::test]
    async fn test_affected_nodes_edge() {
        let event = MutationEvent::EdgeCreated {
            edge: EdgeSnapshot {
                id: obrain_common::types::EdgeId(0),
                src: NodeId(10),
                dst: NodeId(20),
                edge_type: "LINK".into(),
                properties: vec![],
            },
        };
        let affected = KernelListener::affected_nodes(&event);
        assert_eq!(affected.len(), 2);
        assert!(affected.contains(&NodeId(10)));
        assert!(affected.contains(&NodeId(20)));
    }

    #[tokio::test]
    async fn test_empty_batch_noop() {
        let store = test_store();
        let _nodes = populate_graph(&store, 5);
        let manager = make_manager(Arc::clone(&store));
        let listener = KernelListener::new(Arc::clone(&manager));

        let count_before = manager.embedding_count();
        listener.on_batch(&[]).await;
        assert_eq!(manager.embedding_count(), count_before);
    }

    #[tokio::test]
    async fn test_debug_impl() {
        let store = test_store();
        populate_graph(&store, 3);
        let manager = make_manager(Arc::clone(&store));
        let listener = KernelListener::new(manager);
        let debug = format!("{:?}", listener);
        assert!(debug.contains("KernelListener"));
        assert!(debug.contains("has_embeddings"));
    }
}
