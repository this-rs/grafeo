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
///
/// **Substrate-backed mode (T6):** When constructed via
/// [`Self::with_substrate`], the authoritative weight lives on the
/// `EdgeRecord.weight_u16` column of the `SubstrateStore` — every
/// reinforcement translates to a `SynapseReinforce` WAL record + mmap
/// column mutation. Decay is batched via [`Self::decay_all`] (the
/// Consolidator Thinker owns its cadence at runtime). The DashMap becomes
/// a warm-read accelerator. The `graph_store` is still required in this
/// mode because edge creation (structural shape) goes through the graph
/// store; only the weight column is substrate-routed.
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
    /// Optional backing substrate store — when set, the cognitive column
    /// `EdgeRecord.weight_u16` (Q0.16) is the source of truth for synapse
    /// weights and the DashMap is a thin warm-read accelerator.
    #[cfg(feature = "substrate")]
    substrate: Option<Arc<obrain_substrate::SubstrateStore>>,
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
            #[cfg(feature = "substrate")]
            substrate: None,
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
            #[cfg(feature = "substrate")]
            substrate: None,
            cross_base: DashMap::new(),
        }
    }

    /// Creates a new synapse store backed by a substrate column view (T6).
    ///
    /// The `EdgeRecord.weight_u16` Q0.16 column is the source of truth for
    /// synapse weights — every `reinforce` translates to a dedicated
    /// `SynapseReinforce` WAL record + mmap column mutation via
    /// [`SubstrateStore::reinforce_edge_synapse_f32`] /
    /// [`SubstrateStore::boost_edge_synapse_f32`]. The in-memory `DashMap`
    /// cache is retained as a warm-read accelerator and for cross-session
    /// bookkeeping (`reinforcement_count`, `last_reinforced` timestamps —
    /// neither currently fits on the 30 B `EdgeRecord`).
    ///
    /// Decay semantics shift from lazy-per-read (legacy) to eager-periodic-
    /// batch — callers must invoke [`Self::decay_all`] on a schedule (the
    /// Consolidator Thinker from T13 owns this cadence at runtime).
    ///
    /// `graph_store` is still required for edge creation (structural
    /// shape); only the weight column is substrate-routed. When the graph
    /// store is itself substrate-backed (via `HubWalStore`/`WalGraphStore`),
    /// the structural + cognitive paths converge on the same WAL.
    #[cfg(feature = "substrate")]
    pub fn with_substrate(
        config: SynapseConfig,
        graph_store: Arc<dyn obrain_core::graph::GraphStoreMut>,
        substrate: Arc<obrain_substrate::SubstrateStore>,
    ) -> Self {
        Self {
            synapses: DashMap::new(),
            edge_ids: DashMap::new(),
            access_order: DashMap::new(),
            access_counter: AtomicU64::new(0),
            max_cache_entries: 0,
            config,
            graph_store: Some(graph_store),
            substrate: Some(substrate),
            cross_base: DashMap::new(),
        }
    }

    /// Returns `true` if this store routes weight mutations through a
    /// substrate column view.
    #[cfg(feature = "substrate")]
    pub fn is_substrate_backed(&self) -> bool {
        self.substrate.is_some()
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
    ///
    /// In **substrate-backed mode**, the authoritative weight column
    /// (`EdgeRecord.weight_u16`, Q0.16 in `[0, 1]`) is mutated via
    /// [`SubstrateStore::reinforce_edge_synapse_f32`] — WAL-logged
    /// synchronously before the mmap write. The in-memory cache is updated
    /// as a mirror for reinforcement-count and timestamp bookkeeping.
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
            // In substrate mode, route the weight through the EdgeRecord
            // column — this is the sole durable write path. The graph
            // store's edge properties become metadata (reinforcement count
            // + epoch) that stay on the LPG side until T10 expands the
            // edge record.
            #[cfg(feature = "substrate")]
            if let Some(sub) = self.substrate.as_ref() {
                // Clamp to [0, 1] for the Q0.16 column. Values above 1.0
                // saturate — expected in high-activity regimes because
                // `max_synapse_weight` defaults to 10.0 while the column
                // tops at 1.0. The source-of-truth raw weight stays in
                // the DashMap + LPG property for cross-session fidelity;
                // the column is a fast-path gradient used by the
                // Consolidator and spreading-activation code.
                let normalized = (entry.raw_weight() / max_w).clamp(0.0, 1.0);
                let _ = sub.reinforce_edge_synapse_f32(eid, normalized as f32);
            }

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

    /// Apply a multiplicative decay to every live synapse weight column
    /// (`weight ← weight × factor`) in a single WAL batch.
    ///
    /// This is the eager-periodic-batch counterpart of the lazy per-read
    /// decay baked into [`Synapse::current_weight`]. In **substrate-backed
    /// mode** it issues a single `SynapseDecay` WAL record followed by a
    /// zone-wide mmap scan (O(live_edges), tombstones skipped).
    ///
    /// In legacy mode (no substrate), it iterates the DashMap and scales
    /// every cached synapse weight in place — useful for test parity but
    /// not as efficient.
    ///
    /// `factor` is clamped to `[0.0, 1.0]`. The Consolidator Thinker (T13)
    /// owns the cadence at runtime.
    pub fn decay_all(&self, factor: f64) {
        let f = factor.clamp(0.0, 1.0);
        #[cfg(feature = "substrate")]
        if let Some(sub) = self.substrate.as_ref() {
            let _ = sub.decay_all_edge_synapse(f as f32);
        }
        // Always also scale the in-memory cache so reads reflect the new
        // weight even before a cache reload. In substrate-only mode the
        // cache is a mirror; in legacy mode the cache is authoritative.
        for mut entry in self.synapses.iter_mut() {
            entry.scale_weight(f);
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

        // Load the raw weight. In substrate mode, the column (EdgeRecord.weight_u16)
        // is authoritative — the LPG property path may not be persisted at all
        // (substrate edge properties land in a later milestone). In legacy mode,
        // the LPG property is required; missing means no synapse.
        #[cfg(feature = "substrate")]
        let substrate_weight_raw: Option<f64> = self
            .substrate
            .as_ref()
            .and_then(|sub| match sub.get_edge_synapse_weight_f32(eid) {
                Ok(Some(w)) => Some((w as f64) * self.config.max_synapse_weight),
                _ => None,
            });
        #[cfg(not(feature = "substrate"))]
        let substrate_weight_raw: Option<f64> = None;

        let raw_weight = match substrate_weight_raw {
            Some(w) => w,
            None => load_edge_f64(gs.as_ref(), eid, PROP_SYNAPSE_WEIGHT)?,
        };

        // Reconstruct the Instant from the persisted epoch timestamp.
        // If epoch is missing, treat as "just now" (no cross-session decay).
        let last_reinforced = epoch_to_instant(load_edge_f64(
            gs.as_ref(),
            eid,
            PROP_SYNAPSE_LAST_REINFORCED_EPOCH,
        ));
        let reinforcement_count = load_edge_f64(gs.as_ref(), eid, PROP_SYNAPSE_REINFORCEMENT_COUNT)
            .map_or(1, |v| v as u32);

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
            if target == key.1
                && let Some(etype) = gs.edge_type(eid)
                && etype.as_str() == SYNAPSE_EDGE_TYPE
            {
                self.edge_ids.insert(key, eid);
                return Some(eid);
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

    /// Rehydrates the synapse cache from the backing graph store.
    ///
    /// Iterates over all nodes, scans their outgoing edges, and rebuilds the
    /// in-memory `synapses` DashMap from any edge typed `SYNAPSE` carrying
    /// the `PROP_SYNAPSE_WEIGHT` property. Must be called once on brain open
    /// — otherwise `len()`, `snapshot()`, and any other "list all" surface
    /// will report zero until a `reinforce()` call populates the cache
    /// (which only happens on fresh co-activations).
    ///
    /// Lazy per-key lookup via `get_synapse()` already works for individual
    /// reads, but health metrics and outbound enumeration rely on the cache
    /// being fully populated.
    ///
    /// **Substrate-backed mode:** this still walks the graph store because
    /// the `SynapseKey → EdgeId` mapping + `reinforcement_count` /
    /// `last_reinforced` metadata live on the LPG side. The weight column
    /// itself is read from the substrate (`EdgeRecord.weight_u16`) and
    /// overrides the LPG value on hit — the substrate is authoritative.
    ///
    /// Returns the number of synapses loaded into the cache.
    pub fn load_from_graph(&self) -> usize {
        use obrain_core::graph::Direction;
        let Some(gs) = self.graph_store.as_ref() else {
            return 0;
        };
        let mut loaded = 0usize;
        for nid in gs.node_ids() {
            for (target, eid) in gs.edges_from(nid, Direction::Outgoing) {
                let Some(etype) = gs.edge_type(eid) else {
                    continue;
                };
                if etype.as_str() != SYNAPSE_EDGE_TYPE {
                    continue;
                }
                // Canonical (min, max) key — skip the mirror iteration.
                let key = SynapseKey::new(nid, target);
                if key.0 != nid {
                    continue;
                }
                if self.synapses.contains_key(&key) {
                    continue;
                }
                let Some(raw_weight) = load_edge_f64(gs.as_ref(), eid, PROP_SYNAPSE_WEIGHT) else {
                    continue;
                };
                let last_reinforced = epoch_to_instant(load_edge_f64(
                    gs.as_ref(),
                    eid,
                    PROP_SYNAPSE_LAST_REINFORCED_EPOCH,
                ));
                let reinforcement_count =
                    load_edge_f64(gs.as_ref(), eid, PROP_SYNAPSE_REINFORCEMENT_COUNT)
                        .map_or(1, |v| v as u32);

                // Substrate is authoritative for weight when set — override
                // the LPG-persisted value with the column read. Denormalize
                // from Q0.16 `[0, 1]` back to raw `[0, max_synapse_weight]`.
                #[cfg(feature = "substrate")]
                let raw_weight = if let Some(sub) = self.substrate.as_ref() {
                    match sub.get_edge_synapse_weight_f32(eid) {
                        Ok(Some(w)) => (w as f64) * self.config.max_synapse_weight,
                        _ => raw_weight,
                    }
                } else {
                    raw_weight
                };

                let syn = Synapse {
                    source: key.0,
                    target: key.1,
                    weight: raw_weight,
                    reinforcement_count,
                    last_reinforced,
                    created_at: last_reinforced,
                    half_life: self.config.default_half_life,
                };
                self.synapses.insert(key, syn);
                self.edge_ids.insert(key, eid);
                loaded += 1;
            }
        }
        loaded
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

// ---------------------------------------------------------------------------
// Substrate-backed tests (T6 Step 2b)
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "substrate"))]
mod substrate_tests {
    use super::*;
    use obrain_core::graph::traits::GraphStoreMut;
    use obrain_substrate::SubstrateStore;

    /// Seed `n` nodes in a fresh substrate store (labels are single letter
    /// "n") and return the store + NodeIds + TempDir (kept alive so mmap
    /// survives for the duration of the test).
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
    fn substrate_reinforce_writes_through_edge_column() {
        let (sub, ids, _td) = make_substrate(2);
        let cfg = SynapseConfig::default();
        let store = SynapseStore::with_substrate(
            cfg.clone(),
            sub.clone() as Arc<dyn GraphStoreMut>,
            sub.clone(),
        );
        assert!(store.is_substrate_backed());

        // Reinforce — edge is created via the graph store (= substrate),
        // weight column is bumped via SynapseReinforce WAL.
        store.reinforce(ids[0], ids[1], 0.5);

        // Look up the EdgeId via the cache populated in ensure_edge.
        let key = SynapseKey::new(ids[0], ids[1]);
        let eid = *store
            .edge_ids
            .get(&key)
            .expect("edge_id populated after reinforce");

        // Column must hold the normalized weight (raw / max_synapse_weight).
        let col = sub
            .get_edge_synapse_weight_f32(eid)
            .unwrap()
            .expect("weight column readable");
        // raw = initial + 0.5 = 0.6; max = 10.0 → normalized ≈ 0.06.
        let expected = (0.6_f32 / cfg.max_synapse_weight as f32).clamp(0.0, 1.0);
        assert!(
            (col - expected).abs() < 1e-3,
            "column weight {col}, expected {expected}"
        );
    }

    #[test]
    fn substrate_decay_all_halves_every_edge_weight() {
        let (sub, ids, _td) = make_substrate(3);
        let cfg = SynapseConfig::default();
        let store = SynapseStore::with_substrate(
            cfg,
            sub.clone() as Arc<dyn GraphStoreMut>,
            sub.clone(),
        );
        // Seed three pairwise synapses.
        store.reinforce(ids[0], ids[1], 0.4);
        store.reinforce(ids[0], ids[2], 0.4);
        store.reinforce(ids[1], ids[2], 0.4);

        // Snapshot column values before decay.
        let before: Vec<(EdgeId, f32)> = sub
            .iter_live_synapse_weights()
            .unwrap()
            .into_iter()
            .filter(|(_, w)| *w > 0.0)
            .collect();
        assert_eq!(before.len(), 3, "three synapse edges have non-zero weight");

        // Apply ×0.5 decay.
        store.decay_all(0.5);

        for (eid, before_w) in &before {
            let after = sub
                .get_edge_synapse_weight_f32(*eid)
                .unwrap()
                .unwrap();
            let expected = before_w * 0.5;
            assert!(
                (after - expected).abs() < 1e-3,
                "edge {eid:?}: before={before_w}, after={after}, expected≈{expected}"
            );
        }
    }

    #[test]
    fn substrate_get_synapse_reads_column_weight() {
        let (sub, ids, _td) = make_substrate(2);
        let cfg = SynapseConfig::default();
        let store = SynapseStore::with_substrate(
            cfg.clone(),
            sub.clone() as Arc<dyn GraphStoreMut>,
            sub.clone(),
        );
        store.reinforce(ids[0], ids[1], 0.5);

        // Flush the cache to force a reload path.
        store.synapses.clear();

        let syn = store
            .get_synapse(ids[0], ids[1])
            .expect("synapse recovered via column read");
        // `raw_weight` stored is col × max_synapse_weight; col was 0.6/10 = 0.06
        // → raw ≈ 0.6. Decay is zero (just set).
        assert!(
            (syn.current_weight() - 0.6).abs() < 0.1,
            "recovered weight {} not close to 0.6",
            syn.current_weight()
        );
    }

    #[test]
    fn substrate_load_from_graph_rehydrates_from_column() {
        let (sub, ids, _td) = make_substrate(2);
        let cfg = SynapseConfig::default();
        {
            // First session: reinforce, then drop the SynapseStore — the
            // weight is already in substrate WAL + column.
            let store = SynapseStore::with_substrate(
                cfg.clone(),
                sub.clone() as Arc<dyn GraphStoreMut>,
                sub.clone(),
            );
            store.reinforce(ids[0], ids[1], 0.5);
        }
        // Second session: brand-new SynapseStore over the same substrate.
        let store2 = SynapseStore::with_substrate(
            cfg.clone(),
            sub.clone() as Arc<dyn GraphStoreMut>,
            sub.clone(),
        );
        assert_eq!(store2.len(), 0, "cache starts empty");
        let loaded = store2.load_from_graph();
        assert_eq!(loaded, 1, "one synapse rehydrated from graph + column");
        let syn = store2
            .get_synapse(ids[0], ids[1])
            .expect("synapse visible after load_from_graph");
        assert!(
            syn.raw_weight() > 0.4,
            "weight restored from column, got {}",
            syn.raw_weight()
        );
    }
}
