//! Hebbian synapses — co-activation learning between graph nodes.
//!
//! A synapse connects two nodes with a `weight` that decays exponentially
//! and is reinforced whenever the nodes are co-activated (mutated in the
//! same batch). This implements a simplified Hebbian learning rule:
//! "nodes that fire together, wire together."

use async_trait::async_trait;
use dashmap::DashMap;
use grafeo_common::types::NodeId;
use grafeo_reactive::{MutationEvent, MutationListener};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

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
}

impl Default for SynapseConfig {
    fn default() -> Self {
        Self {
            initial_weight: 0.1,
            reinforce_amount: 0.2,
            default_half_life: Duration::from_secs(7 * 24 * 3600), // 7 days
            min_weight: 0.01,
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

/// Thread-safe store for synapses between nodes.
pub struct SynapseStore {
    /// Synapses indexed by canonical (source, target) key.
    synapses: DashMap<SynapseKey, Synapse>,
    /// Configuration.
    config: SynapseConfig,
}

impl SynapseStore {
    /// Creates a new, empty synapse store.
    pub fn new(config: SynapseConfig) -> Self {
        Self {
            synapses: DashMap::new(),
            config,
        }
    }

    /// Reinforces (or creates) a synapse between two nodes.
    pub fn reinforce(&self, source: NodeId, target: NodeId, amount: f64) {
        if source == target {
            return; // No self-synapses
        }
        let key = SynapseKey::new(source, target);
        self.synapses
            .entry(key)
            .and_modify(|s| s.reinforce(amount))
            .or_insert_with(|| {
                Synapse::new(
                    key.0,
                    key.1,
                    self.config.initial_weight + amount,
                    self.config.default_half_life,
                )
            });
    }

    /// Returns the synapse between two nodes, if it exists.
    pub fn get_synapse(&self, a: NodeId, b: NodeId) -> Option<Synapse> {
        let key = SynapseKey::new(a, b);
        self.synapses.get(&key).map(|s| s.clone())
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
        self.synapses.iter().map(|entry| entry.value().clone()).collect()
    }
}

impl std::fmt::Debug for SynapseStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SynapseStore")
            .field("synapse_count", &self.synapses.len())
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
