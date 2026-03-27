//! Node energy system — exponential decay and activation boosting.
//!
//! Each node in the graph has an associated energy level that decays
//! exponentially over time: `E(t) = E0 × 2^(-Δt / half_life)`.
//!
//! Nodes are boosted (energy increased) whenever they are mutated,
//! queried, or otherwise "activated". Nodes whose energy falls below
//! a configurable threshold are candidates for archival or eviction.

use async_trait::async_trait;
use dashmap::DashMap;
use obrain_common::types::NodeId;
use obrain_reactive::{MutationEvent, MutationListener};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::store_trait::{OptionalGraphStore, PROP_ENERGY, load_node_f64, persist_node_f64};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the energy subsystem.
#[derive(Debug, Clone)]
pub struct EnergyConfig {
    /// Energy boost applied to nodes touched by a mutation.
    pub boost_on_mutation: f64,
    /// Default energy for newly tracked nodes.
    pub default_energy: f64,
    /// Default half-life for energy decay.
    pub default_half_life: Duration,
    /// Minimum energy threshold — nodes below this are "low energy".
    pub min_energy: f64,
    /// Maximum energy cap — energy is clamped to `[0.0, max_energy]` after every operation.
    pub max_energy: f64,
    /// Reference energy for normalization: `energy_score = 1 - exp(-energy / ref_energy)`.
    /// Higher values spread the curve, lower values compress it.
    pub ref_energy: f64,
    /// Structural reinforcement coefficient (α).
    /// Modulates half-life by node degree: `effective_half_life = base * (1 + α * ln(1 + degree))`.
    /// Set to 0.0 to disable structural reinforcement.
    pub structural_reinforcement_alpha: f64,
}

impl Default for EnergyConfig {
    fn default() -> Self {
        Self {
            boost_on_mutation: 1.0,
            default_energy: 1.0,
            default_half_life: Duration::from_secs(24 * 3600), // 24 hours
            min_energy: 0.01,
            max_energy: 10.0,
            ref_energy: 1.0,
            structural_reinforcement_alpha: 0.0,
        }
    }
}

impl EnergyConfig {
    /// Creates a config with custom half-life.
    pub fn with_half_life(mut self, half_life: Duration) -> Self {
        self.default_half_life = half_life;
        self
    }

    /// Creates a config with custom mutation boost.
    pub fn with_boost(mut self, boost: f64) -> Self {
        self.boost_on_mutation = boost;
        self
    }
}

// ---------------------------------------------------------------------------
// NodeEnergy
// ---------------------------------------------------------------------------

/// Energy state for a single node.
///
/// Energy decays exponentially: `E(t) = E0 × 2^(-Δt / half_life)`.
/// Calling [`boost`](NodeEnergy::boost) adds energy and resets the
/// activation timestamp.
#[derive(Debug, Clone)]
pub struct NodeEnergy {
    /// Stored energy level (as of `last_activated`).
    energy: f64,
    /// When the energy was last set/boosted.
    last_activated: Instant,
    /// Half-life for exponential decay.
    half_life: Duration,
    /// Monotonic access counter for LRU eviction (updated on every read/write).
    pub(crate) last_access: u64,
}

impl NodeEnergy {
    /// Creates a new `NodeEnergy` with the given initial energy and half-life.
    pub fn new(energy: f64, half_life: Duration) -> Self {
        Self {
            energy,
            last_activated: Instant::now(),
            half_life,
            last_access: 0,
        }
    }

    /// Creates a `NodeEnergy` with a specific activation time.
    ///
    /// Useful for testing and simulation where you need to control the
    /// starting timestamp.
    pub fn new_at(energy: f64, half_life: Duration, last_activated: Instant) -> Self {
        Self {
            energy,
            last_activated,
            half_life,
            last_access: 0,
        }
    }

    /// Returns the current energy after applying decay.
    ///
    /// `E(t) = E0 × 2^(-Δt / half_life)`
    pub fn current_energy(&self) -> f64 {
        self.energy_at(Instant::now())
    }

    /// Returns the energy at a specific instant (for testing/simulation).
    pub fn energy_at(&self, now: Instant) -> f64 {
        let elapsed = now.duration_since(self.last_activated);
        let half_lives = elapsed.as_secs_f64() / self.half_life.as_secs_f64();
        self.energy * 2.0_f64.powf(-half_lives)
    }

    /// Boosts the node's energy by `amount`.
    ///
    /// First applies the current decay, then adds `amount`, and resets
    /// the activation timestamp.
    pub fn boost(&mut self, amount: f64) {
        self.boost_at(amount, Instant::now());
    }

    /// Boosts at a specific instant.
    ///
    /// Useful for testing and simulation.
    pub fn boost_at(&mut self, amount: f64, now: Instant) {
        let current = self.energy_at(now);
        self.energy = current + amount;
        self.last_activated = now;
    }

    /// Returns the stored (non-decayed) energy value.
    pub fn raw_energy(&self) -> f64 {
        self.energy
    }

    /// Returns the last activation instant.
    pub fn last_activated(&self) -> Instant {
        self.last_activated
    }

    /// Returns the configured half-life.
    pub fn half_life(&self) -> Duration {
        self.half_life
    }

    /// Clamps the stored energy to `[min, max]`.
    pub fn clamp(&mut self, min: f64, max: f64) {
        self.energy = self.energy.clamp(min, max);
    }
}

// ---------------------------------------------------------------------------
// EnergyStore
// ---------------------------------------------------------------------------

/// Thread-safe store for node energy states.
///
/// Uses `DashMap` for concurrent read/write without global locks.
/// When a backing [`GraphStoreMut`](obrain_core::graph::GraphStoreMut) is
/// provided, every write is also persisted as a node property (write-through).
/// On read, if the node is not in the hot cache, the store attempts to load
/// the value lazily from the graph property.
pub struct EnergyStore {
    /// Per-node energy entries (includes inline LRU access counter).
    nodes: DashMap<NodeId, NodeEnergy>,
    /// Monotonic counter for LRU tracking.
    access_counter: AtomicU64,
    /// Maximum cache entries (0 = unlimited).
    max_cache_entries: usize,
    /// Configuration.
    config: EnergyConfig,
    /// Optional backing graph store for write-through persistence.
    graph_store: OptionalGraphStore,
}

impl EnergyStore {
    /// Creates a new, empty energy store (in-memory only, no persistence).
    pub fn new(config: EnergyConfig) -> Self {
        Self {
            nodes: DashMap::new(),
            access_counter: AtomicU64::new(0),
            max_cache_entries: 0,
            config,
            graph_store: None,
        }
    }

    /// Creates a new energy store with write-through persistence.
    pub fn with_graph_store(
        config: EnergyConfig,
        graph_store: Arc<dyn obrain_core::graph::GraphStoreMut>,
    ) -> Self {
        Self {
            nodes: DashMap::new(),
            access_counter: AtomicU64::new(0),
            max_cache_entries: 0,
            config,
            graph_store: Some(graph_store),
        }
    }

    /// Sets the maximum number of cache entries. When exceeded, LRU eviction kicks in.
    /// Evicted entries remain in the graph store and are reloaded on-demand.
    pub fn with_max_cache_entries(mut self, max: usize) -> Self {
        self.max_cache_entries = max;
        self
    }

    /// Records an access for LRU tracking — updates the inline counter
    /// in the already-locked DashMap entry (no extra DashMap lookup).
    fn touch(&self, node_id: NodeId) {
        let order = self.access_counter.fetch_add(1, Ordering::Relaxed);
        if let Some(mut entry) = self.nodes.get_mut(&node_id) {
            entry.last_access = order;
        }
    }

    /// Evicts least-recently-used entries if cache exceeds max_cache_entries.
    fn maybe_evict(&self) {
        if self.max_cache_entries == 0 {
            return;
        }
        let current_len = self.nodes.len();
        if current_len <= self.max_cache_entries {
            return;
        }
        let to_evict = current_len - self.max_cache_entries;
        // Collect entries sorted by access order (ascending = oldest first)
        let mut entries: Vec<(NodeId, u64)> = self
            .nodes
            .iter()
            .map(|e| (*e.key(), e.value().last_access))
            .collect();
        entries.sort_by_key(|(_, order)| *order);
        for (node_id, _) in entries.into_iter().take(to_evict) {
            self.nodes.remove(&node_id);
        }
    }

    /// Returns the current energy for a node (with decay applied).
    ///
    /// Returns `0.0` if the node has never been tracked.
    /// If the node is not in the hot cache but exists in the graph store,
    /// the value is loaded lazily.
    pub fn get_energy(&self, node_id: NodeId) -> f64 {
        if let Some(entry) = self.nodes.get(&node_id) {
            let energy = entry.current_energy();
            drop(entry);
            // Only update LRU access order when eviction is configured —
            // avoids a DashMap write on every read in the default (no-eviction) path.
            if self.max_cache_entries > 0 {
                self.touch(node_id);
            }
            return energy;
        }
        // Lazy load from graph store
        if let Some(gs) = &self.graph_store
            && let Some(val) = load_node_f64(gs.as_ref(), node_id, PROP_ENERGY)
        {
            let mut ne = NodeEnergy::new(val, self.config.default_half_life);
            if self.max_cache_entries > 0 {
                ne.last_access = self.access_counter.fetch_add(1, Ordering::Relaxed);
            }
            self.nodes.insert(node_id, ne);
            self.maybe_evict();
            return val;
        }
        0.0
    }

    /// Boosts a node's energy by `amount`.
    ///
    /// If the node is not yet tracked, it is created with the boost as
    /// initial energy. The new energy value is written through to the
    /// backing graph store (if configured).
    pub fn boost(&self, node_id: NodeId, amount: f64) {
        let max_energy = self.config.max_energy;
        self.nodes
            .entry(node_id)
            .and_modify(|e| {
                e.boost(amount);
                e.clamp(0.0, max_energy);
            })
            .or_insert_with(|| {
                let clamped = amount.clamp(0.0, max_energy);
                NodeEnergy::new(clamped, self.config.default_half_life)
            });
        self.touch(node_id);
        self.maybe_evict();
        // Write-through
        if let Some(gs) = &self.graph_store
            && let Some(entry) = self.nodes.get(&node_id)
        {
            persist_node_f64(gs.as_ref(), node_id, PROP_ENERGY, entry.current_energy());
        }
    }

    /// Returns all node IDs whose current energy is below `threshold`.
    pub fn list_low_energy(&self, threshold: f64) -> Vec<NodeId> {
        self.nodes
            .iter()
            .filter(|entry| entry.value().current_energy() < threshold)
            .map(|entry| *entry.key())
            .collect()
    }

    /// Returns the number of tracked nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns `true` if no nodes are tracked.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Returns a reference to the configuration.
    pub fn config(&self) -> &EnergyConfig {
        &self.config
    }

    /// Returns all tracked node IDs with their current energy.
    pub fn snapshot(&self) -> Vec<(NodeId, f64)> {
        self.nodes
            .iter()
            .map(|entry| (*entry.key(), entry.value().current_energy()))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Normalized scoring functions
// ---------------------------------------------------------------------------

/// Converts a raw energy value to a normalized score in `[0.0, 1.0]`.
///
/// Uses the formula: `score = 1 - exp(-energy / ref_energy)`.
///
/// - `energy = 0` → score = `0.0`
/// - `energy → ∞` → score → `1.0`
/// - Negative energy is clamped to `0.0`.
/// - `ref_energy` controls the curve spread (higher = slower saturation).
///   If `ref_energy <= 0` or NaN, falls back to `ref_energy = 1.0`.
///
/// This provides a smooth, bounded mapping from the unbounded energy
/// domain to a [0, 1] range suitable for cross-metric comparison.
#[inline]
pub fn energy_score(energy: f64, ref_energy: f64) -> f64 {
    if energy <= 0.0 || energy.is_nan() {
        return 0.0;
    }
    let r = if ref_energy <= 0.0 || ref_energy.is_nan() || ref_energy.is_infinite() {
        1.0
    } else {
        ref_energy
    };
    (1.0 - (-energy / r).exp()).clamp(0.0, 1.0)
}

/// Computes the effective half-life modulated by structural degree.
///
/// `effective_half_life = base_half_life * (1 + α * ln(1 + degree))`
///
/// Nodes with higher degree (more connections) retain energy longer,
/// reflecting their structural importance in the graph.
///
/// - `degree = 0` → effective = base (no reinforcement)
/// - Higher α → stronger reinforcement effect
/// - α = 0 → no reinforcement (returns base)
#[inline]
pub fn effective_half_life(base: Duration, degree: usize, alpha: f64) -> Duration {
    if alpha <= 0.0 || alpha.is_nan() {
        return base;
    }
    let factor = 1.0 + alpha * (1.0 + degree as f64).ln();
    let secs = base.as_secs_f64() * factor;
    if secs.is_finite() && secs > 0.0 {
        Duration::from_secs_f64(secs)
    } else {
        base
    }
}

impl std::fmt::Debug for EnergyStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EnergyStore")
            .field("tracked_nodes", &self.nodes.len())
            .field("config", &self.config)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// EnergyListener
// ---------------------------------------------------------------------------

/// A [`MutationListener`] that automatically boosts energy for mutated nodes.
///
/// When a node or edge is created/updated/deleted, the involved node(s)
/// receive an energy boost of `config.boost_on_mutation`.
pub struct EnergyListener {
    /// Shared energy store.
    store: Arc<EnergyStore>,
}

impl EnergyListener {
    /// Creates a new energy listener backed by the given store.
    pub fn new(store: Arc<EnergyStore>) -> Self {
        Self { store }
    }

    /// Returns a reference to the underlying store.
    pub fn store(&self) -> &Arc<EnergyStore> {
        &self.store
    }

    /// Extracts all node IDs affected by a mutation event.
    fn affected_nodes(event: &MutationEvent) -> smallvec::SmallVec<[NodeId; 2]> {
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
impl MutationListener for EnergyListener {
    fn name(&self) -> &str {
        "cognitive:energy"
    }

    async fn on_event(&self, event: &MutationEvent) {
        let boost = self.store.config.boost_on_mutation;
        for node_id in Self::affected_nodes(event) {
            self.store.boost(node_id, boost);
        }
    }

    async fn on_batch(&self, events: &[MutationEvent]) {
        let boost = self.store.config.boost_on_mutation;
        for event in events {
            for node_id in Self::affected_nodes(event) {
                self.store.boost(node_id, boost);
            }
        }
    }
}

impl std::fmt::Debug for EnergyListener {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EnergyListener")
            .field("store", &self.store)
            .finish()
    }
}
