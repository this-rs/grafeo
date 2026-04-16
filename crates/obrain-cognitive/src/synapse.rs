//! Hebbian synapses — co-activation learning between graph nodes.
//!
//! A synapse connects two nodes with a `weight` that decays exponentially
//! and is reinforced whenever the nodes are co-activated (mutated in the
//! same batch). This implements a simplified Hebbian learning rule:
//! "nodes that fire together, wire together."

use async_trait::async_trait;
use dashmap::DashMap;
use obrain_common::types::{EdgeId, NodeId};
use obrain_reactive::{MutationEvent, MutationListener};
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::store_trait::{
    OptionalGraphStore, PROP_SYNAPSE_LAST_REINFORCED_EPOCH, PROP_SYNAPSE_REINFORCEMENT_COUNT,
    PROP_SYNAPSE_WEIGHT, epoch_to_instant, load_edge_f64, now_epoch_secs, persist_edge_f64,
};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the synapse subsystem.
#[derive(Debug, Clone)]
pub struct SynapseConfig {
    /// Default initial weight for a newly created synapse.
    pub initial_weight: f64,
    /// Reinforcement amount when nodes are co-activated.
    pub reinforce_amount: f64,
    /// Default half-life for weight decay.
    pub default_half_life: Duration,
    /// Minimum weight threshold for pruning.
    pub min_weight: f64,
    /// Maximum individual synapse weight cap.
    pub max_synapse_weight: f64,
    /// Maximum total outgoing weight from a single node.
    /// When exceeded, all outgoing weights are normalized proportionally (competitive Hebbian).
    pub max_total_outgoing_weight: f64,
}

impl Default for SynapseConfig {
    fn default() -> Self {
        Self {
            initial_weight: 0.1,
            reinforce_amount: 0.2,
            default_half_life: Duration::from_secs(7 * 24 * 3600), // 7 days
            min_weight: 0.01,
            max_synapse_weight: 10.0,
            max_total_outgoing_weight: 100.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Synapse
// ---------------------------------------------------------------------------

/// A weighted connection between two nodes, learned via co-activation.
#[derive(Debug, Clone)]
pub struct Synapse {
    /// Source node.
    pub source: NodeId,
    /// Target node.
    pub target: NodeId,
    /// Current weight (as of `last_reinforced`).
    weight: f64,
    /// Number of times this synapse has been reinforced.
    pub reinforcement_count: u32,
    /// When the weight was last set/reinforced.
    last_reinforced: Instant,
    /// When this synapse was first created.
    pub created_at: Instant,
    /// Half-life for exponential weight decay.
    half_life: Duration,
}

impl Synapse {
    /// Creates a new synapse with the given initial weight.
    pub fn new(source: NodeId, target: NodeId, weight: f64, half_life: Duration) -> Self {
        let now = Instant::now();
        Self {
            source,
            target,
            weight,
            reinforcement_count: 1,
            last_reinforced: now,
            created_at: now,
            half_life,
        }
    }

    /// Creates a synapse with explicit timestamps (for testing).
    pub fn new_at(
        source: NodeId,
        target: NodeId,
        weight: f64,
        half_life: Duration,
        at: Instant,
    ) -> Self {
        Self {
            source,
            target,
            weight,
            reinforcement_count: 1,
            last_reinforced: at,
            created_at: at,
            half_life,
        }
    }

    /// Returns the current weight after applying decay.
    ///
    /// `W(t) = W0 × 2^(-Δt / half_life)`
    pub fn current_weight(&self) -> f64 {
        self.weight_at(Instant::now())
    }

    /// Returns the weight at a specific instant.
    pub fn weight_at(&self, now: Instant) -> f64 {
        let elapsed = now.duration_since(self.last_reinforced);
        let half_lives = elapsed.as_secs_f64() / self.half_life.as_secs_f64();
        self.weight * 2.0_f64.powf(-half_lives)
    }

    /// Reinforces this synapse: applies decay then adds `amount`.
    pub fn reinforce(&mut self, amount: f64) {
        self.reinforce_at(amount, Instant::now());
    }

    /// Reinforces at a specific instant (for testing).
    pub fn reinforce_at(&mut self, amount: f64, now: Instant) {
        let current = self.weight_at(now);
        self.weight = current + amount;
        self.reinforcement_count += 1;
        self.last_reinforced = now;
    }

    /// Returns the raw (non-decayed) weight value.
    pub fn raw_weight(&self) -> f64 {
        self.weight
    }

    /// Returns when this synapse was last reinforced.
    pub fn last_reinforced(&self) -> Instant {
        self.last_reinforced
    }

    /// Clamps the stored weight to `[0.0, max]`.
    pub fn clamp_weight(&mut self, max: f64) {
        self.weight = self.weight.clamp(0.0, max);
    }

    /// Scales the stored weight by `factor` (for competitive normalization).
    pub fn scale_weight(&mut self, factor: f64) {
        self.weight *= factor;
    }
}

// ---------------------------------------------------------------------------
// CrossBaseNodeId — node identifier spanning multiple databases
// ---------------------------------------------------------------------------

/// Identifies a node across different databases by combining a database
/// identifier with a [`NodeId`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct CrossBaseNodeId {
    /// The database this node belongs to.
    pub db_id: String,
    /// The node identifier within that database.
    pub node_id: NodeId,
}

impl CrossBaseNodeId {
    /// Creates a new cross-base node identifier.
    pub fn new(db_id: String, node_id: NodeId) -> Self {
        Self { db_id, node_id }
    }
}

/// A snapshot of a cross-base synapse — a weighted connection between two
/// nodes from potentially different databases.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CrossBaseSynapse {
    /// Source node.
    pub source: CrossBaseNodeId,
    /// Target node.
    pub target: CrossBaseNodeId,
    /// Current weight of the synapse.
    weight: f64,
    /// Number of times this synapse has been reinforced.
    pub reinforcement_count: u32,
}

impl CrossBaseSynapse {
    /// Returns the current weight of this cross-base synapse.
    pub fn current_weight(&self) -> f64 {
        self.weight
    }
}

// ---------------------------------------------------------------------------
// SynapseKey — canonical (src, tgt) pair with min/max ordering
// ---------------------------------------------------------------------------

/// Canonical key for an undirected synapse (always stored as min, max).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct SynapseKey(NodeId, NodeId);

impl SynapseKey {
    fn new(a: NodeId, b: NodeId) -> Self {
        if a.0 <= b.0 { Self(a, b) } else { Self(b, a) }
    }
}

// ---------------------------------------------------------------------------
// SynapseStore
// ---------------------------------------------------------------------------

/// Edge type used for synapse edges in the graph store.
const SYNAPSE_EDGE_TYPE: &str = "SYNAPSE";

/// Thread-safe store for synapses between nodes.
///
/// When a backing [`GraphStoreMut`](obrain_core::graph::GraphStoreMut) is
/// provided, synapse weights are persisted as edge properties on
/// `SYNAPSE`-typed edges. On read, if the synapse is not in the hot cache,
/// the store attempts to load the weight lazily from the graph property.
pub struct SynapseStore {
    /// Synapses indexed by canonical (source, target) key.
    synapses: DashMap<SynapseKey, Synapse>,
    /// Mapping from synapse key to graph edge ID (for persistence).
    edge_ids: DashMap<SynapseKey, EdgeId>,
    /// LRU access order — maps SynapseKey to monotonic access counter.
    access_order: DashMap<SynapseKey, u64>,
    /// Monotonic counter for LRU tracking.
    access_counter: AtomicU64,
    /// Maximum cache entries (0 = unlimited).
    max_cache_entries: usize,
    /// Configuration.
    config: SynapseConfig,
    /// Optional backing graph store for write-through persistence.
    graph_store: OptionalGraphStore,
    /// Cross-base synapses between nodes from different databases.
    cross_base: DashMap<(CrossBaseNodeId, CrossBaseNodeId), (f64, u32)>,
}

impl SynapseStore {
    /// Creates a new, empty synapse store (in-memory only).
    pub fn new(config: SynapseConfig) -> Self {
        Self {
            synapses: DashMap::new(),
            edge_ids: DashMap::new(),
            access_order: DashMap::new(),
            access_counter: AtomicU64::new(0),
            max_cache_entries: 0,
            config,
            graph_store: None,
            cross_base: DashMap::new(),
        }
    }

    /// Creates a new synapse store with write-through persistence.
    pub fn with_graph_store(
        config: SynapseConfig,
        graph_store: Arc<dyn obrain_core::graph::GraphStoreMut>,
    ) -> Self {
        Self {
            synapses: DashMap::new(),
            edge_ids: DashMap::new(),
            access_order: DashMap::new(),
            access_counter: AtomicU64::new(0),
            max_cache_entries: 0,
            config,
            graph_store: Some(graph_store),
            cross_base: DashMap::new(),
        }
    }

    /// Sets the maximum number of cache entries. When exceeded, LRU eviction kicks in.
    pub fn with_max_cache_entries(mut self, max: usize) -> Self {
        self.max_cache_entries = max;
        self
    }

    /// Records an access for LRU tracking.
    fn touch(&self, key: SynapseKey) {
        let order = self.access_counter.fetch_add(1, Ordering::Relaxed);
        self.access_order.insert(key, order);
    }

    /// Evicts least-recently-used entries if cache exceeds max_cache_entries.
    fn maybe_evict(&self) {
        if self.max_cache_entries == 0 {
            return;
        }
        let current_len = self.synapses.len();
        if current_len <= self.max_cache_entries {
            return;
        }
        let to_evict = current_len - self.max_cache_entries;
        let mut entries: Vec<(SynapseKey, u64)> = self
            .access_order
            .iter()
            .map(|e| (*e.key(), *e.value()))
            .collect();
        entries.sort_by_key(|(_, order)| *order);
        for (key, _) in entries.into_iter().take(to_evict) {
            self.synapses.remove(&key);
            self.access_order.remove(&key);
        }
    }

    /// Ensures a graph edge exists for the synapse and returns its EdgeId.
    fn ensure_edge(&self, key: SynapseKey) -> Option<EdgeId> {
        let gs = self.graph_store.as_ref()?;
        if let Some(eid) = self.edge_ids.get(&key) {
            return Some(*eid);
        }
        let eid = gs.create_edge(key.0, key.1, SYNAPSE_EDGE_TYPE);
        self.edge_ids.insert(key, eid);
        Some(eid)
    }

    /// Reinforces (or creates) a synapse between two nodes.
    ///
    /// After reinforcement, the individual weight is clamped to `max_synapse_weight`.
    /// Then, if the total outgoing weight from `source` (or `target`) exceeds
    /// `max_total_outgoing_weight`, all outgoing weights from that node are
    /// normalized proportionally (competitive Hebbian normalization).
    pub fn reinforce(&self, source: NodeId, target: NodeId, amount: f64) {
        if source == target {
            return; // No self-synapses
        }
        let key = SynapseKey::new(source, target);
        let max_w = self.config.max_synapse_weight;
        self.synapses
            .entry(key)
            .and_modify(|s| {
                s.reinforce(amount);
                s.clamp_weight(max_w);
            })
            .or_insert_with(|| {
                let w = (self.config.initial_weight + amount).min(max_w);
                Synapse::new(key.0, key.1, w, self.config.default_half_life)
            });

        // Competitive normalization for both endpoints
        self.normalize_outgoing(source);
        self.normalize_outgoing(target);

        self.touch(key);
        self.maybe_evict();

        // Write-through: persist weight, epoch timestamp, and reinforcement count
        if let Some(eid) = self.ensure_edge(key)
            && let Some(gs) = &self.graph_store
            && let Some(entry) = self.synapses.get(&key)
        {
            // Persist the raw weight (not decayed) — decay will be recomputed
            // from the epoch timestamp on reload.
            persist_edge_f64(gs.as_ref(), eid, PROP_SYNAPSE_WEIGHT, entry.raw_weight());
            // Persist epoch so cross-session decay works correctly.
            let epoch_secs = now_epoch_secs();
            persist_edge_f64(
                gs.as_ref(),
                eid,
                PROP_SYNAPSE_LAST_REINFORCED_EPOCH,
                epoch_secs,
            );
            persist_edge_f64(
                gs.as_ref(),
                eid,
                PROP_SYNAPSE_REINFORCEMENT_COUNT,
                entry.reinforcement_count as f64,
            );
        }
    }

    /// Normalizes all outgoing synapse weights from `node_id` if their sum
    /// exceeds `max_total_outgoing_weight`. Uses proportional scaling so
    /// relative weights are preserved (competitive Hebbian normalization).
    fn normalize_outgoing(&self, node_id: NodeId) {
        let max_total = self.config.max_total_outgoing_weight;
        // Collect all synapse keys involving this node and their current weights
        let entries: Vec<(SynapseKey, f64)> = self
            .synapses
            .iter()
            .filter(|entry| {
                let k = entry.key();
                k.0 == node_id || k.1 == node_id
            })
            .map(|entry| (*entry.key(), entry.value().current_weight()))
            .collect();

        let total: f64 = entries.iter().map(|(_, w)| *w).sum();
        if total <= max_total {
            return;
        }

        let scale = max_total / total;
        for (key, _) in &entries {
            if let Some(mut syn) = self.synapses.get_mut(key) {
                syn.scale_weight(scale);
            }
        }
    }

    /// Returns the synapse between two nodes, if it exists.
    ///
    /// If not in the hot cache, attempts lazy load from the graph store.
    /// When `edge_ids` is empty (e.g. after process restart), uses graph
    /// traversal to locate the SYNAPSE edge between the two nodes.
    pub fn get_synapse(&self, a: NodeId, b: NodeId) -> Option<Synapse> {
        let key = SynapseKey::new(a, b);
        if let Some(s) = self.synapses.get(&key) {
            self.touch(key);
            return Some(s.clone());
        }
        // Try known edge_id first, then fall back to graph traversal
        let gs = self.graph_store.as_ref()?;
        let eid = if let Some(eid) = self.edge_ids.get(&key) {
            *eid
        } else {
            // Traverse outgoing edges from the lower-id node to find the SYNAPSE edge
            self.find_synapse_edge(gs.as_ref(), key)?
        };
        let raw_weight = load_edge_f64(gs.as_ref(), eid, PROP_SYNAPSE_WEIGHT)?;

        // Reconstruct the Instant from the persisted epoch timestamp.
        // If epoch is missing, treat as "just now" (no cross-session decay).
        let last_reinforced = epoch_to_instant(
            load_edge_f64(gs.as_ref(), eid, PROP_SYNAPSE_LAST_REINFORCED_EPOCH),
        );
        let reinforcement_count = load_edge_f64(
            gs.as_ref(),
            eid,
            PROP_SYNAPSE_REINFORCEMENT_COUNT,
        )
        .map(|v| v as u32)
        .unwrap_or(1);

        let syn = Synapse {
            source: key.0,
            target: key.1,
            weight: raw_weight,
            reinforcement_count,
            last_reinforced,
            created_at: last_reinforced, // best approximation
            half_life: self.config.default_half_life,
        };
        self.synapses.insert(key, syn.clone());
        self.touch(key);
        self.maybe_evict();
        Some(syn)
    }

    /// Searches the graph for a SYNAPSE edge between the two nodes of a key.
    /// Populates `edge_ids` on hit for subsequent fast lookups.
    fn find_synapse_edge(
        &self,
        gs: &dyn obrain_core::graph::GraphStoreMut,
        key: SynapseKey,
    ) -> Option<EdgeId> {
        use obrain_core::graph::Direction;
        for (target, eid) in gs.edges_from(key.0, Direction::Outgoing) {
            if target == key.1 {
                if let Some(etype) = gs.edge_type(eid) {
                    if etype.as_str() == SYNAPSE_EDGE_TYPE {
                        self.edge_ids.insert(key, eid);
                        return Some(eid);
                    }
                }
            }
        }
        None
    }

    /// Returns all synapses for a given node, sorted by current weight descending.
    pub fn list_synapses(&self, node_id: NodeId) -> Vec<Synapse> {
        let mut result: Vec<Synapse> = self
            .synapses
            .iter()
            .filter(|entry| {
                let k = entry.key();
                k.0 == node_id || k.1 == node_id
            })
            .map(|entry| entry.value().clone())
            .collect();
        result.sort_by(|a, b| {
            b.current_weight()
                .partial_cmp(&a.current_weight())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        result
    }

    /// Removes all synapses whose current weight is below `min_weight`.
    ///
    /// Returns the number of pruned synapses.
    pub fn prune(&self, min_weight: f64) -> usize {
        let to_remove: Vec<SynapseKey> = self
            .synapses
            .iter()
            .filter(|entry| entry.value().current_weight() < min_weight)
            .map(|entry| *entry.key())
            .collect();
        let count = to_remove.len();
        for key in to_remove {
            self.synapses.remove(&key);
        }
        count
    }

    /// Returns the total number of synapses.
    pub fn len(&self) -> usize {
        self.synapses.len()
    }

    /// Returns `true` if there are no synapses.
    pub fn is_empty(&self) -> bool {
        self.synapses.is_empty()
    }

    /// Returns a reference to the configuration.
    pub fn config(&self) -> &SynapseConfig {
        &self.config
    }

    /// Returns a snapshot of all synapses.
    pub fn snapshot(&self) -> Vec<Synapse> {
        self.synapses
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    // -- cross-base synapse methods ------------------------------------------

    /// Reinforces (or creates) a cross-base synapse between two nodes from
    /// potentially different databases.
    ///
    /// The weight is clamped to `max_synapse_weight` after reinforcement.
    pub fn reinforce_cross_base(
        &self,
        source: CrossBaseNodeId,
        target: CrossBaseNodeId,
        amount: f64,
    ) {
        let max_w = self.config.max_synapse_weight;
        let initial = self.config.initial_weight;
        let key = (source, target);
        self.cross_base
            .entry(key)
            .and_modify(|(w, count)| {
                *w = (*w + amount).min(max_w);
                *count += 1;
            })
            .or_insert_with(|| ((initial + amount).min(max_w), 1));
    }

    /// Removes all cross-base synapses whose weight is below `threshold`.
    ///
    /// Returns the number of pruned entries.
    pub fn prune_cross_base(&self, threshold: f64) -> usize {
        let to_remove: Vec<(CrossBaseNodeId, CrossBaseNodeId)> = self
            .cross_base
            .iter()
            .filter(|entry| entry.value().0 < threshold)
            .map(|entry| entry.key().clone())
            .collect();
        let count = to_remove.len();
        for key in to_remove {
            self.cross_base.remove(&key);
        }
        count
    }

    /// Returns the total number of cross-base synapses.
    pub fn cross_base_len(&self) -> usize {
        self.cross_base.len()
    }

    /// Returns a snapshot of all cross-base synapses.
    pub fn snapshot_cross_base(&self) -> Vec<CrossBaseSynapse> {
        self.cross_base
            .iter()
            .map(|entry| {
                let (src, tgt) = entry.key().clone();
                let (weight, count) = *entry.value();
                CrossBaseSynapse {
                    source: src,
                    target: tgt,
                    weight,
                    reinforcement_count: count,
                }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Normalized scoring functions
// ---------------------------------------------------------------------------

/// Converts a raw synapse weight to a normalized score in `[0.0, 1.0]`.
///
/// Uses the formula: `score = tanh(weight / ref_weight)`.
///
/// - `weight = 0` → score = `0.0`
/// - `weight → ∞` → score → `1.0`
/// - Negative weight is clamped to `0.0`.
/// - `ref_weight` controls the curve spread (higher = slower saturation).
///   If `ref_weight <= 0` or NaN, falls back to `ref_weight = 1.0`.
///
/// This provides a smooth, bounded mapping from the unbounded weight
/// domain to a [0, 1] range suitable for cross-metric comparison.
#[inline]
pub fn synapse_score(weight: f64, ref_weight: f64) -> f64 {
    if weight <= 0.0 || weight.is_nan() {
        return 0.0;
    }
    let r = if ref_weight <= 0.0 || ref_weight.is_nan() || ref_weight.is_infinite() {
        1.0
    } else {
        ref_weight
    };
    (weight / r).tanh().clamp(0.0, 1.0)
}

/// Converts a mutation frequency count to a normalized score in `[0.0, 1.0]`.
///
/// Uses the formula: `score = min(1, log(1 + count) / log(1 + ref_count))`.
///
/// - `count = 0` → score = `0.0`
/// - `count = ref_count` → score ≈ `1.0`
/// - `count > ref_count` → score = `1.0` (clamped)
/// - `ref_count` is the reference count at which the score saturates.
///   If `ref_count <= 0` or NaN, falls back to `ref_count = 100.0`.
#[inline]
pub fn mutation_frequency_score(count: f64, ref_count: f64) -> f64 {
    if count <= 0.0 || count.is_nan() {
        return 0.0;
    }
    let r = if ref_count <= 0.0 || ref_count.is_nan() || ref_count.is_infinite() {
        100.0
    } else {
        ref_count
    };
    let log_num = (1.0 + count).ln();
    let log_den = (1.0 + r).ln();
    if log_den <= 0.0 {
        return 1.0;
    }
    (log_num / log_den).clamp(0.0, 1.0)
}

impl std::fmt::Debug for SynapseStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SynapseStore")
            .field("synapse_count", &self.synapses.len())
            .field("cross_base_count", &self.cross_base.len())
            .field("config", &self.config)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// SynapseListener
// ---------------------------------------------------------------------------

/// A [`MutationListener`] that detects co-activated nodes in each batch
/// and reinforces their synapses.
///
/// Co-activation: when two nodes appear in the same mutation batch,
/// they are considered co-activated and their synapse is reinforced.
pub struct SynapseListener {
    store: Arc<SynapseStore>,
}

impl SynapseListener {
    /// Creates a new synapse listener backed by the given store.
    pub fn new(store: Arc<SynapseStore>) -> Self {
        Self { store }
    }

    /// Returns a reference to the underlying store.
    pub fn store(&self) -> &Arc<SynapseStore> {
        &self.store
    }

    /// Extracts all unique node IDs from an event.
    fn node_ids(event: &MutationEvent) -> smallvec::SmallVec<[NodeId; 2]> {
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
impl MutationListener for SynapseListener {
    fn name(&self) -> &str {
        "cognitive:synapse"
    }

    async fn on_event(&self, _event: &MutationEvent) {
        // Single events can't produce co-activation — need a batch
        // (edge events do co-activate src/dst though)
        // We handle everything in on_batch for consistency
    }

    async fn on_batch(&self, events: &[MutationEvent]) {
        // Collect all unique node IDs touched in this batch
        let mut touched: HashSet<NodeId> = HashSet::new();
        for event in events {
            for nid in Self::node_ids(event) {
                touched.insert(nid);
            }
        }

        // Create/reinforce synapses for all pairs of co-activated nodes
        let nodes: Vec<NodeId> = touched.into_iter().collect();
        let amount = self.store.config.reinforce_amount;

        for i in 0..nodes.len() {
            for j in (i + 1)..nodes.len() {
                self.store.reinforce(nodes[i], nodes[j], amount);
            }
        }
    }
}

impl std::fmt::Debug for SynapseListener {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SynapseListener")
            .field("store", &self.store)
            .finish()
    }
}
