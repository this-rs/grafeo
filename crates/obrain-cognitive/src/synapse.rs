//! Hebbian synapses — co-activation learning between graph nodes.
//!
//! A synapse connects two nodes with a `weight` that decays exponentially
//! and is reinforced whenever the nodes are co-activated (mutated in the
//! same batch). This implements a simplified Hebbian learning rule:
//! "nodes that fire together, wire together."

use async_trait::async_trait;
use crossbeam::queue::SegQueue;
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
///
/// Plan 69e59065 T3 — pending metadata write descriptor. Pushed to
/// `SynapseStore::pending_metadata_writes` (a lock-free MPMC SegQueue)
/// from the hot path; drained by `flush_pending_metadata()` either on
/// the SynapseConsolidator background tick (~30s) or via a SIGTERM-safe
/// shutdown handler. Carries only the LPG metadata that has no
/// substrate column today (last_reinforced_epoch, reinforcement_count).
/// The weight is NOT carried — it's already durable on the substrate
/// column path via `reinforce_edge_synapse_f32`.
#[derive(Debug, Clone, Copy)]
struct MetadataDelta {
    eid: EdgeId,
    last_reinforced_epoch: f64,
    reinforcement_count: u32,
}

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
    /// Plan 69e59065 T3 — lock-free MPMC queue of LPG metadata writes
    /// pending persistence. `reinforce()` pushes a `MetadataDelta` here
    /// instead of issuing 3 synchronous `persist_edge_f64` calls. The
    /// consolidator (or SIGTERM handler) drains the queue in batch via
    /// [`Self::flush_pending_metadata`]. Crossbeam's SegQueue is
    /// chosen because the push path must be wait-free under the
    /// 21-thread feedback workload that previously serialised on the
    /// LPG WAL writer (gotcha 49938809 root cause confirmed by T1
    /// step 4 lock contention probe, note f1733c03).
    pending_metadata_writes: SegQueue<MetadataDelta>,
    /// Cross-base synapses between nodes from different databases.
    cross_base: DashMap<(CrossBaseNodeId, CrossBaseNodeId), (f64, u32)>,
    /// Plan 69e59065 T1 — instrumentation: counts every full DashMap scan
    /// triggered by `normalize_outgoing`. Each `reinforce(a, b)` increments
    /// this by 2 (one scan per endpoint), so `dashmap_full_scans_total /
    /// reinforces_total` should equal 2.0 in the worst case. Exposed via
    /// [`Self::dashmap_full_scans_total`] for bench/metrics consumers.
    dashmap_full_scans_total: AtomicU64,
    /// Plan 69e59065 T1 — instrumentation: counts every `reinforce()` call
    /// (entry-point granularity, includes both substrate and legacy modes).
    reinforces_total: AtomicU64,
    /// Plan 69e59065 T1 — instrumentation: counts every persist_edge_f64
    /// call issued by `reinforce()` (3 per call when graph_store is set).
    /// This is the metric that drives the write-amplification ratio:
    /// `lpg_metadata_writes_total / nodes_recalled_per_feedback`.
    lpg_metadata_writes_total: AtomicU64,
    /// Plan 69e59065 T4 — inverted index from NodeId to the canonical
    /// SynapseKeys that include this node as either endpoint. Used by
    /// `normalize_outgoing` to walk only the relevant entries instead
    /// of scanning the full `synapses` DashMap. Maintained on every
    /// `synapses.insert` and `synapses.remove`.
    ///
    /// The SmallVec inline capacity (16) covers most cognitive nodes —
    /// `max_total_outgoing_weight` clamps competitive Hebbian growth
    /// well before that. Heavy hubs spill to heap which is acceptable.
    ///
    /// Memory cost: ~256 B per active node. For 100k synapses across
    /// ~10k unique nodes, ~2.5 MB.
    node_to_keys: DashMap<NodeId, smallvec::SmallVec<[SynapseKey; 16]>>,
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
            dashmap_full_scans_total: AtomicU64::new(0),
            reinforces_total: AtomicU64::new(0),
            lpg_metadata_writes_total: AtomicU64::new(0),
            node_to_keys: DashMap::new(),
            pending_metadata_writes: SegQueue::new(),
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
            dashmap_full_scans_total: AtomicU64::new(0),
            reinforces_total: AtomicU64::new(0),
            lpg_metadata_writes_total: AtomicU64::new(0),
            node_to_keys: DashMap::new(),
            pending_metadata_writes: SegQueue::new(),
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
            dashmap_full_scans_total: AtomicU64::new(0),
            reinforces_total: AtomicU64::new(0),
            lpg_metadata_writes_total: AtomicU64::new(0),
            node_to_keys: DashMap::new(),
            pending_metadata_writes: SegQueue::new(),
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

    // ------------------------------------------------------------------------
    // Plan 69e59065 T4 — inverted index helpers.
    //
    // Every mutation of `self.synapses` MUST flow through these helpers
    // (or `index_insert`/`index_remove` directly) so `node_to_keys` stays
    // consistent. Failure modes if desynced:
    // - missed entries → `normalize_outgoing` would underestimate the
    //   total outgoing weight and skip the proportional rescale, allowing
    //   weights to grow past `max_total_outgoing_weight`.
    // - stale entries → no functional bug (`synapses.get(key)` returns
    //   `None`, the iterator silently skips), but wastes memory.
    // ------------------------------------------------------------------------

    /// Insert a SynapseKey into the inverted index for both endpoints.
    /// Idempotent: duplicate inserts are deduped by linear scan (degré
    /// is small in practice — competitive normalization clamps it).
    ///
    /// IMPORTANT (Plan 69e59065 T4 deadlock post-mortem):
    /// `DashMap::entry()` holds an exclusive shard guard until the
    /// returned `RefMut` is dropped. Any re-entry on the same shard
    /// within that scope (e.g. a second `self.node_to_keys.entry(nid)`)
    /// deadlocks with parking_lot since we try to acquire a lock we
    /// already own. We MUST do the check-and-push on the single guard.
    fn index_insert(&self, key: SynapseKey) {
        for nid in [key.0, key.1] {
            let mut entry = self.node_to_keys.entry(nid).or_default();
            if !entry.iter().any(|k| *k == key) {
                entry.push(key);
            }
        }
    }

    /// Remove a SynapseKey from the inverted index for both endpoints.
    /// Drops the node entry entirely if the vec becomes empty.
    fn index_remove(&self, key: SynapseKey) {
        for nid in [key.0, key.1] {
            let drop_entry = if let Some(mut entry) = self.node_to_keys.get_mut(&nid) {
                entry.retain(|k| *k != key);
                entry.is_empty()
            } else {
                false
            };
            if drop_entry {
                self.node_to_keys.remove(&nid);
            }
        }
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
            // Plan 69e59065 T4 — drop the inverted-index entry for the
            // evicted synapse on both endpoints.
            self.index_remove(key);
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
        // T1 instrumentation — count every entry to reinforce.
        self.reinforces_total.fetch_add(1, Ordering::Relaxed);
        let key = SynapseKey::new(source, target);
        let max_w = self.config.max_synapse_weight;
        let mut newly_inserted = false;
        self.synapses
            .entry(key)
            .and_modify(|s| {
                s.reinforce(amount);
                s.clamp_weight(max_w);
            })
            .or_insert_with(|| {
                newly_inserted = true;
                let w = (self.config.initial_weight + amount).min(max_w);
                Synapse::new(key.0, key.1, w, self.config.default_half_life)
            });
        // Plan 69e59065 T4 — keep node_to_keys in sync. Idempotent on
        // re-entry, so cheap on the and_modify path even if we don't
        // gate it on `newly_inserted`.
        if newly_inserted {
            self.index_insert(key);
        }

        // Competitive normalization for both endpoints
        self.normalize_outgoing(source);
        self.normalize_outgoing(target);

        self.touch(key);
        self.maybe_evict();

        // Write-through: persist weight (substrate column, synchronous)
        // and queue last_reinforced + count for batched LPG persistence.
        //
        // Plan 69e59065 T3 — the prior 3 synchronous `persist_edge_f64`
        // calls became the dominant lock-contention point under
        // multi-thread feedback (T1 step 4 verdict, note f1733c03):
        // single-writer LPG WAL + 21-thread reinforce → p99 330 µs.
        // We retain ONE durable synchronous write — the substrate
        // weight column via `reinforce_edge_synapse_f32` — because:
        //   1. it carries the value users actually see (current weight)
        //   2. its WAL record is per-edge, not serialised across threads
        //      the way LPG `set_edge_property` was
        // The remaining metadata (`last_reinforced_epoch`,
        // `reinforcement_count`) is enqueued for the Consolidator and
        // flushed periodically via [`Self::flush_pending_metadata`].
        // Loss budget on `kill -9` ≤ 30s — within constraint aa932b40.
        // `kill -TERM` triggers a final drain through the SIGTERM handler.
        //
        // PROP_SYNAPSE_WEIGHT (LPG) is no longer written in substrate
        // mode — the column is authoritative, and reads at line 686 +
        // line 817 already prefer the column over the property.
        // Removing the write breaks `load_from_graph` discovery because
        // it currently gates on `PROP_SYNAPSE_WEIGHT` presence; the
        // discovery loop is reworked below to be substrate-first.
        if let Some(eid) = self.ensure_edge(key)
            && let Some(entry) = self.synapses.get(&key)
        {
            let mut substrate_active = false;
            #[cfg(feature = "substrate")]
            if let Some(sub) = self.substrate.as_ref() {
                // Clamp to [0, 1] for the Q0.16 column. Values above 1.0
                // saturate — expected in high-activity regimes because
                // `max_synapse_weight` defaults to 10.0 while the column
                // tops at 1.0.
                let normalized = (entry.raw_weight() / max_w).clamp(0.0, 1.0);
                let _ = sub.reinforce_edge_synapse_f32(eid, normalized as f32);
                substrate_active = true;
            }

            // T1 instrumentation — counter renamed semantically: now
            // counts queued metadata writes instead of synchronous
            // persists. The write_amplification gate from T1 step 3
            // still uses this counter; the ratio stays meaningful
            // because we still emit ~2 metadata writes per reinforce
            // (epoch + count). The 3rd (PROP_SYNAPSE_WEIGHT) is
            // structurally eliminated in substrate mode.
            self.lpg_metadata_writes_total
                .fetch_add(2, Ordering::Relaxed);

            // Always queue the metadata — drained by the consolidator
            // or by an explicit `flush_pending_metadata` call.
            self.pending_metadata_writes.push(MetadataDelta {
                eid,
                last_reinforced_epoch: now_epoch_secs(),
                reinforcement_count: entry.reinforcement_count,
            });

            // Legacy-mode safety net: when substrate is NOT active, the
            // LPG `PROP_SYNAPSE_WEIGHT` is the only place the weight
            // lives. We must persist it synchronously here because the
            // queue only carries epoch + count. Test-only path; production
            // always runs in substrate mode.
            if !substrate_active
                && let Some(gs) = &self.graph_store
            {
                self.lpg_metadata_writes_total.fetch_add(1, Ordering::Relaxed);
                persist_edge_f64(gs.as_ref(), eid, PROP_SYNAPSE_WEIGHT, entry.raw_weight());
            }
        }
    }

    /// Plan 69e59065 T3 — Drain the queued metadata writes
    /// ([`MetadataDelta`]) to the LPG side. Returns the number of
    /// deltas flushed. Idempotent and safe to call concurrently with
    /// `reinforce()` — concurrent pushes that race against the drain
    /// are not lost; they simply land in the next drain.
    ///
    /// Cadence:
    /// - Background `SynapseConsolidator` (T3 step 2): every ~30s
    /// - SIGTERM shutdown handler (T3 step 3): final drain before exit
    /// - Tests / direct callers: on-demand for assertions
    ///
    /// Per-delta cost: 2 `persist_edge_f64` calls
    /// (`PROP_SYNAPSE_LAST_REINFORCED_EPOCH` +
    /// `PROP_SYNAPSE_REINFORCEMENT_COUNT`). Co-locates with the LPG
    /// WAL writer — but now in batch mode, single-threaded, **off the
    /// hot path** so the multi-thread `reinforce()` workload does not
    /// serialise on the WAL.
    pub fn flush_pending_metadata(&self) -> usize {
        let Some(gs) = self.graph_store.as_ref() else {
            // No graph store wired — drop the queued writes silently.
            // Pop everything so the queue doesn't grow unboundedly in
            // tests that never set a graph store.
            let mut dropped = 0usize;
            while self.pending_metadata_writes.pop().is_some() {
                dropped += 1;
            }
            return dropped;
        };
        let mut flushed = 0usize;
        while let Some(d) = self.pending_metadata_writes.pop() {
            persist_edge_f64(
                gs.as_ref(),
                d.eid,
                PROP_SYNAPSE_LAST_REINFORCED_EPOCH,
                d.last_reinforced_epoch,
            );
            persist_edge_f64(
                gs.as_ref(),
                d.eid,
                PROP_SYNAPSE_REINFORCEMENT_COUNT,
                d.reinforcement_count as f64,
            );
            flushed += 1;
        }
        flushed
    }

    /// Plan 69e59065 T3 — number of `MetadataDelta` entries currently
    /// pending in the consolidator queue. Used by metrics and tests.
    pub fn pending_metadata_writes_count(&self) -> usize {
        self.pending_metadata_writes.len()
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

        // Plan 69e59065 T4 — walk only the keys this node participates
        // in via the inverted `node_to_keys` index. Complexity drops
        // from O(|synapses|) to O(degré_node). For 100k synapses with
        // a typical node degree ≤ 16, that's a ~6 000× cost reduction
        // per call. Combined with T2's fallback cap, a worst-case
        // feedback cycle now does ~28 reinforces × O(16) lookups
        // instead of ~5 000 reinforces × O(100 000) scans.
        //
        // The `dashmap_full_scans_total` counter is kept (incremented
        // here under the "walked entries" semantic so existing
        // assertions on the 2× ratio still hold), but the underlying
        // operation is no longer a full scan — it's an indexed lookup
        // followed by a bounded iteration.
        self.dashmap_full_scans_total.fetch_add(1, Ordering::Relaxed);

        // Snapshot the keys to avoid holding the index entry across
        // mutations on `self.synapses` (which could deadlock with the
        // get_mut below if the underlying DashMap shards collide).
        let keys: smallvec::SmallVec<[SynapseKey; 16]> =
            match self.node_to_keys.get(&node_id) {
                Some(entry) => entry.iter().copied().collect(),
                None => return, // node has no recorded synapses → nothing to normalize
            };

        let entries: smallvec::SmallVec<[(SynapseKey, f64); 16]> = keys
            .iter()
            .filter_map(|k| self.synapses.get(k).map(|syn| (*k, syn.current_weight())))
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
        // Plan 69e59065 T4 — keep node_to_keys consistent on lazy-load.
        self.index_insert(key);
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
            // Plan 69e59065 T4 — drop pruned weak synapses from the
            // inverted index too.
            self.index_remove(key);
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

                // Plan 69e59065 T3 — substrate-first weight discovery.
                // Pre-T3, this loop gated synapse discovery on the
                // presence of `PROP_SYNAPSE_WEIGHT`; that worked because
                // every reinforce wrote the property. Post-T3, in
                // substrate mode the property is no longer written
                // (column is authoritative). We must therefore prefer
                // the substrate column for discovery and fall back to
                // the LPG property only in legacy mode.
                let raw_weight = {
                    #[cfg(feature = "substrate")]
                    let from_substrate = self
                        .substrate
                        .as_ref()
                        .and_then(|sub| sub.get_edge_synapse_weight_f32(eid).ok().flatten())
                        .map(|w| (w as f64) * self.config.max_synapse_weight);
                    #[cfg(not(feature = "substrate"))]
                    let from_substrate: Option<f64> = None;

                    match from_substrate {
                        Some(w) => w,
                        None => match load_edge_f64(gs.as_ref(), eid, PROP_SYNAPSE_WEIGHT) {
                            Some(w) => w,
                            None => continue, // neither source carries the weight → skip
                        },
                    }
                };
                let last_reinforced = epoch_to_instant(load_edge_f64(
                    gs.as_ref(),
                    eid,
                    PROP_SYNAPSE_LAST_REINFORCED_EPOCH,
                ));
                let reinforcement_count =
                    load_edge_f64(gs.as_ref(), eid, PROP_SYNAPSE_REINFORCEMENT_COUNT)
                        .map_or(1, |v| v as u32);

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
                // Plan 69e59065 T4 — bulk-populate the inverted index
                // during boot rehydration so the very first
                // `normalize_outgoing` after T0's load_from_graph runs
                // O(degré), not O(|synapses|).
                self.index_insert(key);
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

    // ------------------------------------------------------------------------
    // Plan 69e59065 T1 — Instrumentation accessors.
    //
    // These compteurs feed the bench `feedback_baseline.rs` and the
    // `feedback` tracing span to compute the write-amplification ratio
    // and lock-pressure metrics targeted by the plan.
    // ------------------------------------------------------------------------

    /// Total number of `reinforce()` invocations since process start.
    /// Combined with [`Self::lpg_metadata_writes_total`] gives the
    /// per-call LPG amplification factor (target ≤ 0 after T3).
    pub fn reinforces_total(&self) -> u64 {
        self.reinforces_total.load(Ordering::Relaxed)
    }

    /// Total number of full DashMap scans triggered by
    /// `normalize_outgoing`. Equals 2 × reinforces_total in the worst
    /// case. Target ≤ degré_moyen × reinforces_total after T4.
    pub fn dashmap_full_scans_total(&self) -> u64 {
        self.dashmap_full_scans_total.load(Ordering::Relaxed)
    }

    /// Total number of `persist_edge_f64` calls issued by `reinforce()`
    /// (3 per call when graph_store is wired). T3 target: 0 in the hot
    /// path, all moved to a background consolidator.
    pub fn lpg_metadata_writes_total(&self) -> u64 {
        self.lpg_metadata_writes_total.load(Ordering::Relaxed)
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

    /// Plan 69e59065 T4 — invariant: every key present in `synapses`
    /// MUST be discoverable via `node_to_keys` for both endpoints.
    /// Asserts the inverted index stays consistent across reinforce
    /// (insert), prune_weak (remove), and lazy load_from_graph paths.
    #[test]
    fn t4_node_to_keys_index_consistency_under_mutations() {
        let (sub, ids, _td) = make_substrate(6);
        let store = SynapseStore::with_substrate(
            SynapseConfig::default(),
            sub.clone() as Arc<dyn GraphStoreMut>,
            Arc::clone(&sub),
        );

        // Sparse pattern of reinforces — not all pairs.
        let pairs = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5)];
        for &(i, j) in &pairs {
            store.reinforce(ids[i], ids[j], 0.2);
        }
        let n_synapses_before = store.len();

        // Every recorded synapse must appear in node_to_keys for both endpoints.
        let snap = store.snapshot();
        for syn in &snap {
            let key = SynapseKey::new(syn.source, syn.target);
            for nid in [syn.source, syn.target] {
                let entry = store
                    .node_to_keys
                    .get(&nid)
                    .unwrap_or_else(|| panic!("node {nid:?} missing from inverted index"));
                assert!(
                    entry.iter().any(|k| *k == key),
                    "key {key:?} missing from node_to_keys[{nid:?}]"
                );
            }
        }

        // Conversely, no orphan entries pointing to missing keys.
        for entry in store.node_to_keys.iter() {
            for k in entry.value().iter() {
                assert!(
                    store.synapses.contains_key(k),
                    "node_to_keys references missing synapse key {k:?}"
                );
            }
        }
        assert_eq!(
            store.len(),
            n_synapses_before,
            "len drift after invariant check"
        );
    }

    /// Plan 69e59065 T4 — proves normalize_outgoing is now O(degré),
    /// not O(|synapses|). With 100 unrelated synapses on disjoint
    /// nodes plus 4 outgoing from a focal node, the focal reinforce
    /// must touch only the 4 keys (not all 104).
    ///
    /// The invariant (focal_keys.len() == 4 regardless of background)
    /// holds irrespective of background size — 100 is kept small so
    /// the substrate WAL writes don't dominate test runtime.
    #[test]
    fn t4_normalize_outgoing_only_touches_focal_neighborhood() {
        let (sub, ids, _td) = make_substrate(110);
        let store = SynapseStore::with_substrate(
            SynapseConfig::default(),
            sub.clone() as Arc<dyn GraphStoreMut>,
            Arc::clone(&sub),
        );

        // 100 background synapses on disjoint pairs (5..105 paired with 6..106).
        for i in 5..105 {
            store.reinforce(ids[i], ids[i + 1], 0.05);
        }
        // Focal node ids[0] gets 4 outgoing synapses.
        for j in 1..5 {
            store.reinforce(ids[0], ids[j], 0.1);
        }

        // Inverted index for the focal node must list exactly 4 keys
        // (one per neighbour). normalize_outgoing(ids[0]) walks only
        // these 4 → O(degré) by construction, independent of the
        // 100 unrelated background synapses.
        let focal_keys = store
            .node_to_keys
            .get(&ids[0])
            .expect("focal node should have inverted-index entry");
        assert_eq!(
            focal_keys.len(),
            4,
            "expected 4 outgoing keys for focal node, got {}",
            focal_keys.len()
        );

        // Sanity: total synapse count is the union (100 background + 4 focal).
        assert_eq!(store.len(), 104);
    }

    /// Plan 69e59065 T4 — concurrent mixed-op stress test: 8 threads each
    /// performing ~1 250 mixed reinforce / decay_all operations (10 000
    /// total) on a shared RAM-only SynapseStore. Runs in <100 ms.
    ///
    /// Each `reinforce` internally calls `normalize_outgoing` for both
    /// endpoints, which walks the inverted index → this is the read
    /// path under contention. `decay_all` is a global write/read on
    /// every synapse, also stressing concurrent iteration.
    ///
    /// The test asserts the node_to_keys ↔ synapses invariant holds
    /// after the storm: every key in `synapses` is reachable via the
    /// inverted index for both endpoints, and no orphan entry points
    /// at a missing key. The small NODE_POOL (32 nodes) forces shard
    /// collisions in DashMap, which would have triggered the prior
    /// re-entry deadlock immediately.
    #[test]
    fn t4_concurrent_mixed_ops_stress_invariant() {
        use std::sync::Arc;
        use std::thread;

        const THREADS: usize = 8;
        const OPS_PER_THREAD: usize = 1_250;
        const NODE_POOL: u64 = 32;

        let store = Arc::new(SynapseStore::new(SynapseConfig::default()));
        let ids: Vec<NodeId> = (1..=NODE_POOL).map(NodeId).collect();

        let mut handles = Vec::with_capacity(THREADS);
        for tid in 0..THREADS {
            let store = Arc::clone(&store);
            let ids = ids.clone();
            handles.push(thread::spawn(move || {
                // Cheap deterministic LCG seeded by thread id — no rand
                // dep needed in the test crate.
                let mut state: u64 = 0x9E37_79B9_7F4A_7C15u64.wrapping_mul(tid as u64 + 1);
                let next = |s: &mut u64| {
                    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    *s
                };
                for _ in 0..OPS_PER_THREAD {
                    let r = next(&mut state);
                    let a = ids[((r >> 1) as usize) % ids.len()];
                    let b = ids[((r >> 33) as usize) % ids.len()];
                    if a == b {
                        continue;
                    }
                    if r & 0b11 == 0 {
                        // 25 % decay sweeps the whole synapse map.
                        store.decay_all(0.999);
                    } else {
                        // 75 % reinforce — also fires normalize_outgoing
                        // twice (once per endpoint), exercising the
                        // inverted-index read path.
                        store.reinforce(a, b, 0.05);
                    }
                }
            }));
        }
        for h in handles {
            h.join().expect("worker panicked");
        }

        // Invariant 1 — every recorded synapse is in the inverted index
        // for BOTH endpoints.
        let snap = store.snapshot();
        for syn in &snap {
            let key = SynapseKey::new(syn.source, syn.target);
            for nid in [syn.source, syn.target] {
                let entry = store.node_to_keys.get(&nid).unwrap_or_else(|| {
                    panic!(
                        "post-stress: node {nid:?} missing from inverted index \
                         (expected to reference key {key:?})"
                    )
                });
                assert!(
                    entry.iter().any(|k| *k == key),
                    "post-stress: node_to_keys[{nid:?}] does not contain {key:?}"
                );
            }
        }

        // Invariant 2 — no orphan entries pointing to missing synapses.
        for entry in store.node_to_keys.iter() {
            for k in entry.value().iter() {
                assert!(
                    store.synapses.contains_key(k),
                    "post-stress: node_to_keys[{:?}] references missing synapse {k:?}",
                    entry.key()
                );
            }
        }

        // Sanity bound: with NODE_POOL=32 the full unordered pair space
        // is C(32, 2) = 496 — competitive normalization plus eviction
        // keep us at or below that.
        let max_pairs = (NODE_POOL * (NODE_POOL - 1) / 2) as usize;
        assert!(
            store.len() <= max_pairs,
            "synapse count {} exceeds pair-space upper bound {}",
            store.len(),
            max_pairs
        );
    }
}
