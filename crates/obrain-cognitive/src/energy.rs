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

use crate::store_trait::{
    OptionalGraphStore, PROP_ENERGY, PROP_ENERGY_LAST_ACTIVATED_EPOCH, epoch_to_instant,
    load_node_f64, now_epoch_secs, persist_node_f64,
};

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
    ///
    /// When the store is backed by a `SubstrateStore` (via [`Self::with_substrate`]),
    /// this cache is a best-effort write-back layer around the authoritative
    /// on-disk column; reads first probe the cache, then fall through to the
    /// substrate column. In the legacy graph-store mode it is the primary
    /// working set, write-through'd to the backing `GraphStoreMut` properties.
    nodes: DashMap<NodeId, NodeEnergy>,
    /// Cross-base energy map — tracks energy for nodes identified by (db_id, node_id) pairs.
    #[cfg(feature = "synapse")]
    cross_base: DashMap<crate::synapse::CrossBaseNodeId, f64>,
    /// Monotonic counter for LRU tracking.
    access_counter: AtomicU64,
    /// Maximum cache entries (0 = unlimited).
    max_cache_entries: usize,
    /// Configuration.
    config: EnergyConfig,
    /// Optional backing graph store for write-through persistence (legacy mode).
    graph_store: OptionalGraphStore,
    /// Optional backing substrate store — when set, the cognitive column of
    /// `NodeRecord.energy` (u16 Q1.15) is the source of truth and cache is a
    /// thin accelerator.
    #[cfg(feature = "substrate")]
    substrate: Option<Arc<obrain_substrate::SubstrateStore>>,
}

impl EnergyStore {
    /// Creates a new, empty energy store (in-memory only, no persistence).
    pub fn new(config: EnergyConfig) -> Self {
        Self {
            nodes: DashMap::new(),
            #[cfg(feature = "synapse")]
            cross_base: DashMap::new(),
            access_counter: AtomicU64::new(0),
            max_cache_entries: 0,
            config,
            graph_store: None,
            #[cfg(feature = "substrate")]
            substrate: None,
        }
    }

    /// Creates a new energy store with write-through persistence.
    pub fn with_graph_store(
        config: EnergyConfig,
        graph_store: Arc<dyn obrain_core::graph::GraphStoreMut>,
    ) -> Self {
        Self {
            nodes: DashMap::new(),
            #[cfg(feature = "synapse")]
            cross_base: DashMap::new(),
            access_counter: AtomicU64::new(0),
            max_cache_entries: 0,
            config,
            graph_store: Some(graph_store),
            #[cfg(feature = "substrate")]
            substrate: None,
        }
    }

    /// Creates a new energy store backed by a substrate column view (T6).
    ///
    /// The `NodeRecord.energy` u16 Q1.15 column is the source of truth —
    /// every `boost` / `set` translates to a dedicated `EnergyReinforce`
    /// WAL record + mmap column mutation via
    /// [`SubstrateStore::boost_node_energy_f32`] /
    /// [`SubstrateStore::set_node_energy_f32`]. The in-memory `DashMap`
    /// cache is retained as a warm-read accelerator.
    ///
    /// Decay semantics shift from lazy-per-read (legacy) to eager-periodic-
    /// batch — callers must invoke [`Self::decay_all`] on a schedule (the
    /// Consolidator Thinker from T13 owns this cadence at runtime).
    #[cfg(feature = "substrate")]
    pub fn with_substrate(
        config: EnergyConfig,
        substrate: Arc<obrain_substrate::SubstrateStore>,
    ) -> Self {
        Self {
            nodes: DashMap::new(),
            #[cfg(feature = "synapse")]
            cross_base: DashMap::new(),
            access_counter: AtomicU64::new(0),
            max_cache_entries: 0,
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

    /// Returns the current energy for a node.
    ///
    /// In **legacy mode**, decay is applied on read via `NodeEnergy::current_energy()`.
    /// In **substrate-backed mode**, the raw column value is returned — decay
    /// is eager/batched (see [`Self::decay_all`]). Returns `0.0` if the node
    /// has never been tracked (or has tombstoned / missing column).
    pub fn get_energy(&self, node_id: NodeId) -> f64 {
        // Substrate-backed: column is source of truth.
        #[cfg(feature = "substrate")]
        if let Some(sub) = self.substrate.as_ref() {
            match sub.get_node_energy_f32(node_id) {
                Ok(Some(v)) => return v as f64,
                Ok(None) => return 0.0,
                Err(_) => return 0.0,
            }
        }

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
        // Lazy load from graph store — reconstruct Instant from epoch
        if let Some(gs) = &self.graph_store
            && let Some(raw_energy) = load_node_f64(gs.as_ref(), node_id, PROP_ENERGY)
        {
            let last_activated = epoch_to_instant(load_node_f64(
                gs.as_ref(),
                node_id,
                PROP_ENERGY_LAST_ACTIVATED_EPOCH,
            ));
            let mut ne =
                NodeEnergy::new_at(raw_energy, self.config.default_half_life, last_activated);
            if self.max_cache_entries > 0 {
                ne.last_access = self.access_counter.fetch_add(1, Ordering::Relaxed);
            }
            let current = ne.current_energy();
            self.nodes.insert(node_id, ne);
            self.maybe_evict();
            return current;
        }
        0.0
    }

    /// Boosts a node's energy by `amount`.
    ///
    /// **Substrate mode**: dispatches to
    /// [`SubstrateStore::boost_node_energy_f32`] which logs an
    /// `EnergyReinforce` WAL record + mutates the u16 Q1.15 column.
    /// The column saturates at `config.max_energy` clamped to the Q1.15
    /// range (`[0, 1.0]` — values > 1.0 in the legacy config are silently
    /// clamped by the fixed-point encoding).
    ///
    /// **Legacy mode**: if the node is not yet tracked, it is created with
    /// the boost as initial energy. The new energy value is written through
    /// to the backing graph store (if configured).
    pub fn boost(&self, node_id: NodeId, amount: f64) {
        let max_energy = self.config.max_energy;

        // Substrate-backed: the column is the source of truth. We still
        // maintain the cache for legacy callers that inspect `snapshot()` /
        // `len()` fast-path, but the durable write is a WAL-first column
        // mutation.
        #[cfg(feature = "substrate")]
        if let Some(sub) = self.substrate.as_ref() {
            // Q1.15 tops out at ~1.0; clamp max_energy into that range.
            let max_q = max_energy.min(1.0) as f32;
            let amt = amount as f32;
            if let Ok(Some(new_v)) = sub.boost_node_energy_f32(node_id, amt, max_q) {
                // Sync cache to reflect the durable value (for snapshot/len parity).
                self.nodes
                    .entry(node_id)
                    .and_modify(|e| {
                        e.energy = new_v as f64;
                        e.last_activated = Instant::now();
                    })
                    .or_insert_with(|| {
                        NodeEnergy::new(new_v as f64, self.config.default_half_life)
                    });
                self.touch(node_id);
                self.maybe_evict();
            }
            return;
        }

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
        // Write-through: persist raw energy + epoch timestamp
        if let Some(gs) = &self.graph_store
            && let Some(entry) = self.nodes.get(&node_id)
        {
            persist_node_f64(gs.as_ref(), node_id, PROP_ENERGY, entry.raw_energy());
            persist_node_f64(
                gs.as_ref(),
                node_id,
                PROP_ENERGY_LAST_ACTIVATED_EPOCH,
                now_epoch_secs(),
            );
        }
    }

    /// Apply an eager multiplicative decay to every live node's energy
    /// column. `factor` is in `[0.0, 1.0]` (`0.5` halves all energies).
    ///
    /// **Substrate mode**: dispatches to [`SubstrateStore::decay_all_energy`]
    /// which logs a single `EnergyDecay` WAL record and rewrites every live
    /// slot in one pass. This is the replacement for the legacy
    /// per-read exponential decay.
    ///
    /// **Legacy mode**: iterates the hot cache and multiplies each
    /// `raw_energy`, resetting `last_activated` to `now`. The backing
    /// graph-store (if any) is NOT re-persisted in this path — callers that
    /// rely on legacy `PROP_ENERGY` durability should stick with lazy decay.
    pub fn decay_all(&self, factor: f64) {
        #[cfg(feature = "substrate")]
        if let Some(sub) = self.substrate.as_ref() {
            let _ = sub.decay_all_energy(factor as f32);
            // Invalidate cache — cheapest path is to clear, next read re-reads
            // the column. The cache is a best-effort accelerator in
            // substrate mode, not a source of truth.
            self.nodes.clear();
            return;
        }

        let f = factor.clamp(0.0, 1.0);
        let now = Instant::now();
        for mut entry in self.nodes.iter_mut() {
            entry.energy *= f;
            entry.last_activated = now;
        }
    }

    /// Returns all node IDs whose current energy is below `threshold`.
    pub fn list_low_energy(&self, threshold: f64) -> Vec<NodeId> {
        #[cfg(feature = "substrate")]
        if let Some(sub) = self.substrate.as_ref() {
            return match sub.iter_live_node_energies() {
                Ok(pairs) => pairs
                    .into_iter()
                    .filter(|(_, e)| (*e as f64) < threshold)
                    .map(|(id, _)| id)
                    .collect(),
                Err(_) => Vec::new(),
            };
        }

        self.nodes
            .iter()
            .filter(|entry| entry.value().current_energy() < threshold)
            .map(|entry| *entry.key())
            .collect()
    }

    /// Rehydrates the energy cache from the backing graph store.
    ///
    /// **Substrate mode**: this is a no-op returning `0`. The on-disk
    /// u16 Q1.15 column is the source of truth; there is nothing to
    /// "rehydrate" — the substrate's own mmap + WAL-replay already leaves
    /// the column in its crash-consistent state on open. The in-memory
    /// DashMap cache is populated lazily on demand (next `get_energy` /
    /// `boost`).
    ///
    /// **Legacy mode**: iterates every node and loads `PROP_ENERGY` (+
    /// epoch) into the hot cache. Required on brain open — otherwise
    /// `len()`, `snapshot()`, and aggregate health metrics (total energy,
    /// activation count) report zero until a `boost()` call repopulates the
    /// DashMap.
    ///
    /// Returns the number of nodes loaded.
    pub fn load_from_graph(&self) -> usize {
        #[cfg(feature = "substrate")]
        if self.substrate.is_some() {
            // The substrate column is the source of truth. The cache is a
            // warm-read accelerator populated lazily — no eager fill here.
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
            let Some(raw_energy) = load_node_f64(gs.as_ref(), nid, PROP_ENERGY) else {
                continue;
            };
            let last_activated = epoch_to_instant(load_node_f64(
                gs.as_ref(),
                nid,
                PROP_ENERGY_LAST_ACTIVATED_EPOCH,
            ));
            let ne = NodeEnergy::new_at(raw_energy, self.config.default_half_life, last_activated);
            self.nodes.insert(nid, ne);
            loaded += 1;
        }
        loaded
    }

    /// Returns the number of tracked nodes.
    ///
    /// **Substrate mode**: this reports the cache size (eagerly-boosted or
    /// previously-read nodes). Use [`Self::snapshot`] for an authoritative
    /// count over the on-disk column.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns `true` if no nodes are tracked.
    ///
    /// **Substrate mode**: reflects the cache only — see [`Self::len`].
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Returns a reference to the configuration.
    pub fn config(&self) -> &EnergyConfig {
        &self.config
    }

    /// Returns all tracked node IDs with their current energy.
    ///
    /// **Substrate mode**: iterates the authoritative on-disk column
    /// (O(N) mmap scan of live non-tombstoned slots). Tombstoned nodes and
    /// slots beyond the high-water mark are excluded.
    ///
    /// **Legacy mode**: snapshots the DashMap cache, applying decay on read.
    pub fn snapshot(&self) -> Vec<(NodeId, f64)> {
        #[cfg(feature = "substrate")]
        if let Some(sub) = self.substrate.as_ref() {
            return match sub.iter_live_node_energies() {
                Ok(pairs) => pairs.into_iter().map(|(id, e)| (id, e as f64)).collect(),
                Err(_) => Vec::new(),
            };
        }

        self.nodes
            .iter()
            .map(|entry| (*entry.key(), entry.value().current_energy()))
            .collect()
    }

    // -----------------------------------------------------------------------
    // Cross-base energy operations
    // -----------------------------------------------------------------------

    /// Returns a snapshot of all cross-base energy entries.
    #[cfg(feature = "synapse")]
    pub fn snapshot_cross_base(&self) -> Vec<(crate::synapse::CrossBaseNodeId, f64)> {
        use crate::synapse::CrossBaseNodeId;
        self.cross_base
            .iter()
            .map(
                |entry: dashmap::mapref::multiple::RefMulti<'_, CrossBaseNodeId, f64>| {
                    (entry.key().clone(), *entry.value())
                },
            )
            .collect()
    }

    /// Returns the top `limit` nodes for a specific database, sorted by
    /// energy descending.
    #[cfg(feature = "synapse")]
    pub fn get_top_nodes_for_base(&self, db_id: &str, limit: usize) -> Vec<(NodeId, f64)> {
        use crate::synapse::CrossBaseNodeId;
        let mut entries: Vec<(NodeId, f64)> = self
            .cross_base
            .iter()
            .filter(
                |entry: &dashmap::mapref::multiple::RefMulti<'_, CrossBaseNodeId, f64>| {
                    entry.key().db_id == db_id
                },
            )
            .map(
                |entry: dashmap::mapref::multiple::RefMulti<'_, CrossBaseNodeId, f64>| {
                    (entry.key().node_id, *entry.value())
                },
            )
            .collect();
        entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        entries.truncate(limit);
        entries
    }

    /// Boosts (inserts or adds to) the cross-base energy for a given node.
    #[cfg(feature = "synapse")]
    pub fn boost_cross_base(&self, xbid: crate::synapse::CrossBaseNodeId, amount: f64) {
        self.cross_base
            .entry(xbid)
            .and_modify(|e| *e += amount)
            .or_insert(amount);
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

// ---------------------------------------------------------------------------
// Substrate-backed tests (T6)
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "substrate"))]
mod substrate_tests {
    use super::*;
    use obrain_substrate::SubstrateStore;
    use std::sync::Arc;

    /// Helper: create a substrate store in a tempdir with `n` node slots
    /// allocated via the `GraphStoreMut` trait (each node gets a single
    /// label "n") so they have legitimate `label_bitset` entries. Returns
    /// the store and the allocated `NodeId`s.
    fn make_substrate(n: usize) -> (Arc<SubstrateStore>, Vec<NodeId>, tempfile::TempDir) {
        use obrain_core::graph::traits::GraphStoreMut;
        let td = tempfile::tempdir().unwrap();
        let sub = SubstrateStore::create(td.path().join("kb")).unwrap();
        let mut ids = Vec::with_capacity(n);
        for _ in 0..n {
            let id = sub.create_node(&["n"]);
            ids.push(id);
        }
        sub.flush().unwrap();
        (Arc::new(sub), ids, td)
    }

    #[test]
    fn substrate_energy_boost_reads_back_through_column() {
        let (sub, ids, _td) = make_substrate(3);
        let cfg = EnergyConfig::default();
        let store = EnergyStore::with_substrate(cfg, sub.clone());
        assert!(store.is_substrate_backed());

        // Boost slot 1 by 0.5 — column should hold ≈ 0.5.
        store.boost(ids[0], 0.5);
        let e0 = store.get_energy(ids[0]);
        assert!((e0 - 0.5).abs() < 1e-3, "energy after boost: {e0}");

        // Another boost — saturates at max_energy (clamped to 1.0 in Q1.15).
        store.boost(ids[0], 0.8);
        let e1 = store.get_energy(ids[0]);
        assert!(e1 <= 1.0_f64 + 1e-3, "energy exceeded 1.0: {e1}");
        assert!(e1 >= 0.9_f64, "energy should approach max: {e1}");
    }

    #[test]
    fn substrate_decay_all_halves_every_node() {
        let (sub, ids, _td) = make_substrate(5);
        let cfg = EnergyConfig::default();
        let store = EnergyStore::with_substrate(cfg, sub.clone());

        // Seed every node at 0.8.
        for id in &ids {
            store.boost(*id, 0.8);
        }
        // Decay ×0.5.
        store.decay_all(0.5);
        // Every node's column should now be ≈ 0.4 (±1 ULP of Q1.15).
        for id in &ids {
            let e = store.get_energy(*id);
            assert!((e - 0.4).abs() < 5e-3, "node {id:?} energy: {e} (expected ≈ 0.4)");
        }
    }

    #[test]
    fn substrate_load_from_graph_is_noop_but_returns_zero() {
        let (sub, ids, _td) = make_substrate(2);
        let cfg = EnergyConfig::default();
        let store = EnergyStore::with_substrate(cfg, sub.clone());
        // Seed one node so the column is non-zero.
        store.boost(ids[0], 0.3);
        // load_from_graph is a no-op in substrate mode (column is
        // already authoritative on disk).
        assert_eq!(store.load_from_graph(), 0);
        // But the value is still readable.
        assert!(store.get_energy(ids[0]) > 0.0);
    }

    #[test]
    fn substrate_snapshot_iterates_live_column() {
        let (sub, ids, _td) = make_substrate(4);
        let cfg = EnergyConfig::default();
        let store = EnergyStore::with_substrate(cfg, sub.clone());
        store.boost(ids[0], 0.2);
        store.boost(ids[2], 0.7);
        // Nodes 1 & 3 were never boosted — they have zero energy in the
        // column but are still "live" (allocated slots). snapshot() iterates
        // every live slot so it returns 4 entries.
        let snap = store.snapshot();
        assert_eq!(snap.len(), 4);
        let sum: f64 = snap.iter().map(|(_, e)| *e).sum();
        assert!((sum - 0.9).abs() < 5e-3, "sum of energies: {sum}");
    }

    #[test]
    fn substrate_list_low_energy_filters_via_column() {
        let (sub, ids, _td) = make_substrate(3);
        let cfg = EnergyConfig::default();
        let store = EnergyStore::with_substrate(cfg, sub.clone());
        store.boost(ids[0], 0.1);
        store.boost(ids[1], 0.5);
        store.boost(ids[2], 0.9);
        let low = store.list_low_energy(0.3);
        assert_eq!(low.len(), 1);
        assert_eq!(low[0], ids[0]);
    }
}
