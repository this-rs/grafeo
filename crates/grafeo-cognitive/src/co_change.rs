//! Co-change detection — temporal coupling between graph nodes.
//!
//! When two nodes are mutated in the same batch (transaction), they are
//! considered co-changed. This is different from synapses (semantic
//! co-activation): co-change relations are strictly temporal signals
//! of coupling.
//!
//! The [`CoChangeDetector`] implements [`MutationListener`] and extracts
//! all distinct node pairs from each batch, recording them in the
//! [`CoChangeStore`].

use async_trait::async_trait;
use dashmap::DashMap;
use grafeo_common::types::{EdgeId, NodeId};
use grafeo_reactive::{MutationEvent, MutationListener};
use smallvec::SmallVec;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::store_trait::{OptionalGraphStore, PROP_CO_CHANGE_COUNT, persist_edge_f64};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the co-change detection subsystem.
#[derive(Debug, Clone)]
pub struct CoChangeConfig {
    /// Window duration for grouping mutations as co-changes.
    /// Default: `Duration::ZERO` (same batch = same transaction only).
    /// When > 0, mutations within this window are also considered co-changed.
    pub window_duration: Duration,

    /// Decay half-life for strength calculation.
    /// Strength decays over time if not reinforced.
    pub strength_half_life: Duration,

    /// Maximum number of nodes in a batch before we skip combinatorial
    /// pair generation (to avoid O(n²) explosion).
    pub max_batch_nodes: usize,
}

impl Default for CoChangeConfig {
    fn default() -> Self {
        Self {
            window_duration: Duration::ZERO,
            strength_half_life: Duration::from_secs(30 * 24 * 3600), // 30 days
            max_batch_nodes: 100,
        }
    }
}

// ---------------------------------------------------------------------------
// CoChangeRelation
// ---------------------------------------------------------------------------

/// A temporal co-change relation between two nodes.
///
/// Tracks how many times two nodes were mutated together and computes
/// a decaying strength signal.
#[derive(Debug, Clone)]
pub struct CoChangeRelation {
    /// Source node (always the smaller NodeId for canonical ordering).
    pub source: NodeId,
    /// Target node (always the larger NodeId for canonical ordering).
    pub target: NodeId,
    /// Number of times these nodes were co-changed.
    pub count: u32,
    /// When the last co-change was recorded.
    pub last_co_changed: Instant,
    /// When this relation was first created.
    pub created_at: Instant,
    /// Half-life for strength decay.
    half_life: Duration,
}

impl CoChangeRelation {
    /// Creates a new co-change relation with count=1.
    pub fn new(source: NodeId, target: NodeId, half_life: Duration) -> Self {
        let now = Instant::now();
        Self {
            source,
            target,
            count: 1,
            last_co_changed: now,
            created_at: now,
            half_life,
        }
    }

    /// Creates a co-change relation with explicit timestamp (for testing).
    pub fn new_at(source: NodeId, target: NodeId, half_life: Duration, at: Instant) -> Self {
        Self {
            source,
            target,
            count: 1,
            last_co_changed: at,
            created_at: at,
            half_life,
        }
    }

    /// Returns the current strength after applying time decay.
    ///
    /// `strength = count × 2^(-Δt / half_life)`
    pub fn strength(&self) -> f64 {
        self.strength_at(Instant::now())
    }

    /// Returns the strength at a specific instant.
    pub fn strength_at(&self, now: Instant) -> f64 {
        let elapsed = now.duration_since(self.last_co_changed);
        let half_lives = elapsed.as_secs_f64() / self.half_life.as_secs_f64();
        f64::from(self.count) * 2.0_f64.powf(-half_lives)
    }

    /// Records another co-change, incrementing count and resetting the timestamp.
    pub fn record(&mut self) {
        self.record_at(Instant::now());
    }

    /// Records another co-change at a specific instant (for testing).
    pub fn record_at(&mut self, now: Instant) {
        self.count += 1;
        self.last_co_changed = now;
    }
}

// ---------------------------------------------------------------------------
// CoChangeKey — canonical (src, tgt) pair
// ---------------------------------------------------------------------------

/// Canonical key for an undirected co-change relation (always min, max).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct CoChangeKey(NodeId, NodeId);

impl CoChangeKey {
    fn new(a: NodeId, b: NodeId) -> Self {
        if a.0 <= b.0 { Self(a, b) } else { Self(b, a) }
    }
}

// ---------------------------------------------------------------------------
// CoChangeStore
// ---------------------------------------------------------------------------

/// Edge type used for co-change edges in the graph store.
const CO_CHANGED_EDGE_TYPE: &str = "CO_CHANGED";

/// Thread-safe store for co-change relations between nodes.
///
/// When a backing [`GraphStoreMut`](grafeo_core::graph::GraphStoreMut) is
/// provided, co-change counts are persisted as edge properties on
/// `CO_CHANGED`-typed edges (write-through).
pub struct CoChangeStore {
    /// Co-change relations indexed by canonical key.
    relations: DashMap<CoChangeKey, CoChangeRelation>,
    /// Mapping from co-change key to graph edge ID (for persistence).
    edge_ids: DashMap<CoChangeKey, EdgeId>,
    /// Configuration.
    config: CoChangeConfig,
    /// Optional backing graph store for write-through persistence.
    graph_store: OptionalGraphStore,
}

impl CoChangeStore {
    /// Creates a new, empty co-change store (in-memory only).
    pub fn new(config: CoChangeConfig) -> Self {
        Self {
            relations: DashMap::new(),
            edge_ids: DashMap::new(),
            config,
            graph_store: None,
        }
    }

    /// Creates a new co-change store with write-through persistence.
    pub fn with_graph_store(
        config: CoChangeConfig,
        graph_store: Arc<dyn grafeo_core::graph::GraphStoreMut>,
    ) -> Self {
        Self {
            relations: DashMap::new(),
            edge_ids: DashMap::new(),
            config,
            graph_store: Some(graph_store),
        }
    }

    /// Ensures a graph edge exists for the co-change relation and returns its EdgeId.
    fn ensure_edge(&self, key: CoChangeKey) -> Option<EdgeId> {
        let gs = self.graph_store.as_ref()?;
        if let Some(eid) = self.edge_ids.get(&key) {
            return Some(*eid);
        }
        let eid = gs.create_edge(key.0, key.1, CO_CHANGED_EDGE_TYPE);
        self.edge_ids.insert(key, eid);
        Some(eid)
    }

    /// Records a co-change between two nodes.
    ///
    /// If a relation already exists, increments count and updates timestamp.
    /// Otherwise creates a new relation with count=1.
    pub fn record_co_change(&self, src: NodeId, tgt: NodeId) {
        if src == tgt {
            return; // No self co-changes
        }
        let key = CoChangeKey::new(src, tgt);
        self.relations
            .entry(key)
            .and_modify(|rel| rel.record())
            .or_insert_with(|| CoChangeRelation::new(key.0, key.1, self.config.strength_half_life));
        // Write-through
        if let Some(eid) = self.ensure_edge(key)
            && let Some(gs) = &self.graph_store
            && let Some(rel) = self.relations.get(&key)
        {
            persist_edge_f64(gs.as_ref(), eid, PROP_CO_CHANGE_COUNT, f64::from(rel.count));
        }
    }

    /// Returns all co-change relations for a given node, sorted by strength descending.
    pub fn get_co_changed(&self, node_id: NodeId) -> Vec<(NodeId, CoChangeRelation)> {
        let mut result: Vec<(NodeId, CoChangeRelation)> = self
            .relations
            .iter()
            .filter(|entry| {
                let k = entry.key();
                k.0 == node_id || k.1 == node_id
            })
            .map(|entry| {
                let k = entry.key();
                let other = if k.0 == node_id { k.1 } else { k.0 };
                (other, entry.value().clone())
            })
            .collect();
        result.sort_by(|a, b| {
            b.1.strength()
                .partial_cmp(&a.1.strength())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        result
    }

    /// Returns the co-change relation between two specific nodes, if it exists.
    pub fn get_relation(&self, a: NodeId, b: NodeId) -> Option<CoChangeRelation> {
        let key = CoChangeKey::new(a, b);
        self.relations.get(&key).map(|r| r.clone())
    }

    /// Returns the total number of co-change relations.
    pub fn len(&self) -> usize {
        self.relations.len()
    }

    /// Returns `true` if there are no co-change relations.
    pub fn is_empty(&self) -> bool {
        self.relations.is_empty()
    }

    /// Returns a reference to the configuration.
    pub fn config(&self) -> &CoChangeConfig {
        &self.config
    }
}

impl std::fmt::Debug for CoChangeStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CoChangeStore")
            .field("relation_count", &self.relations.len())
            .field("config", &self.config)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// CoChangeDetector — MutationListener
// ---------------------------------------------------------------------------

/// A [`MutationListener`] that detects co-changed nodes in each batch
/// and records their temporal coupling in the [`CoChangeStore`].
///
/// Co-change: when two nodes appear in the same mutation batch (transaction),
/// they are considered temporally coupled. This is different from synapses
/// which represent semantic co-activation.
///
/// When `window_duration > 0`, nodes from recent batches (within the window)
/// are also paired with nodes in the current batch, enabling cross-batch
/// co-change detection for temporally close mutations.
pub struct CoChangeDetector {
    store: Arc<CoChangeStore>,
    /// Recent node sets from past batches, with their timestamps.
    /// Used for time-windowed co-change detection.
    recent_batches: parking_lot::Mutex<Vec<(Instant, Vec<NodeId>)>>,
}

impl CoChangeDetector {
    /// Creates a new co-change detector backed by the given store.
    pub fn new(store: Arc<CoChangeStore>) -> Self {
        Self {
            store,
            recent_batches: parking_lot::Mutex::new(Vec::new()),
        }
    }

    /// Returns a reference to the underlying store.
    pub fn store(&self) -> &Arc<CoChangeStore> {
        &self.store
    }

    /// Extracts all unique node IDs from an event.
    fn node_ids(event: &MutationEvent) -> SmallVec<[NodeId; 2]> {
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
impl MutationListener for CoChangeDetector {
    fn name(&self) -> &str {
        "cognitive:co_change"
    }

    async fn on_event(&self, _event: &MutationEvent) {
        // Single events can't produce co-changes — need a batch
    }

    async fn on_batch(&self, events: &[MutationEvent]) {
        // Collect all unique node IDs touched in this batch
        let mut touched: HashSet<NodeId> = HashSet::new();
        for event in events {
            for nid in Self::node_ids(event) {
                touched.insert(nid);
            }
        }

        // Guard against combinatorial explosion
        if touched.len() > self.store.config().max_batch_nodes {
            tracing::warn!(
                nodes = touched.len(),
                max = self.store.config().max_batch_nodes,
                "co-change batch too large, skipping"
            );
            return;
        }

        let window = self.store.config().window_duration;
        let now = Instant::now();

        // Collect nodes within the time window from recent batches
        let mut windowed_nodes: HashSet<NodeId> = touched.clone();
        if !window.is_zero() {
            let mut recent = self.recent_batches.lock();
            // Evict expired entries
            recent.retain(|(ts, _)| now.duration_since(*ts) <= window);
            // Merge recent nodes into the pairing set
            for (_ts, nodes) in recent.iter() {
                for &nid in nodes {
                    windowed_nodes.insert(nid);
                }
            }
            // Store current batch for future window lookups
            recent.push((now, touched.iter().copied().collect()));
        }

        // Guard combined set against explosion
        if windowed_nodes.len() > self.store.config().max_batch_nodes {
            tracing::warn!(
                nodes = windowed_nodes.len(),
                max = self.store.config().max_batch_nodes,
                "co-change windowed set too large, skipping"
            );
            return;
        }

        // Record co-change for all distinct pairs in the combined set
        let nodes: Vec<NodeId> = windowed_nodes.into_iter().collect();
        for i in 0..nodes.len() {
            for j in (i + 1)..nodes.len() {
                self.store.record_co_change(nodes[i], nodes[j]);
            }
        }
    }
}

impl std::fmt::Debug for CoChangeDetector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CoChangeDetector")
            .field("store", &self.store)
            .finish()
    }
}
