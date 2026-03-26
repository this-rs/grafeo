//! # Stigmergy — Diffuse memory on graph edges (Layer 2)
//!
//! Inspired by swarm intelligence. Memory lives in the **edges** of the graph,
//! not in separate nodes. Pheromone traces are deposited on edges during
//! traversals, mutations, errors, and surprise events.
//!
//! ## Concurrency model
//!
//! Pheromones are `AtomicU64`-encoded `f64` values (lock-free). A lost pheromone
//! deposit in a race condition is not critical — the signal will reinforce on
//! the next deposit. This tolerance to races is an intrinsic property of
//! biological stigmergy (ants don't synchronize).
//!
//! The flush to [`EdgeAnnotator`] is a periodic batch operation, NOT synchronous
//! per deposit.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use dashmap::DashMap;
use grafeo_common::types::EdgeId;

use crate::engram::traits::EdgeAnnotator;

// ---------------------------------------------------------------------------
// TrailType — 4 pheromone categories
// ---------------------------------------------------------------------------

/// The four types of stigmergic pheromone trails.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TrailType {
    /// Traversal during read queries.
    Query,
    /// Edge involved in a mutation (create/update/delete).
    Mutation,
    /// Edge associated with an error context.
    Error,
    /// High prediction-error event on this edge.
    Surprise,
}

impl TrailType {
    /// Returns the annotation key used for [`EdgeAnnotator`] persistence.
    #[inline]
    pub fn annotation_key(self) -> &'static str {
        match self {
            TrailType::Query => "pheromone_query",
            TrailType::Mutation => "pheromone_mutation",
            TrailType::Error => "pheromone_error",
            TrailType::Surprise => "pheromone_surprise",
        }
    }

    /// All trail types for iteration.
    pub const ALL: [TrailType; 4] = [
        TrailType::Query,
        TrailType::Mutation,
        TrailType::Error,
        TrailType::Surprise,
    ];
}

// ---------------------------------------------------------------------------
// StigmergicTrace — snapshot view of pheromones on a single edge
// ---------------------------------------------------------------------------

/// A snapshot of all pheromone values on a single edge.
#[derive(Debug, Clone, PartialEq)]
pub struct StigmergicTrace {
    /// The edge these traces belong to.
    pub edge: EdgeId,
    /// Frequency of traversal during read queries.
    pub pheromone_query: f64,
    /// Frequency of modification.
    pub pheromone_mutation: f64,
    /// Association with error contexts.
    pub pheromone_error: f64,
    /// High prediction-error events.
    pub pheromone_surprise: f64,
}

impl StigmergicTrace {
    /// Total pheromone intensity across all trail types.
    #[inline]
    pub fn total_intensity(&self) -> f64 {
        self.pheromone_query
            + self.pheromone_mutation
            + self.pheromone_error
            + self.pheromone_surprise
    }
}

// ---------------------------------------------------------------------------
// AtomicF64 — lock-free f64 via AtomicU64 bit-punning
// ---------------------------------------------------------------------------

/// A lock-free atomic f64 implemented via `AtomicU64` bit-punning.
///
/// Uses `Relaxed` ordering for reads and `fetch_add`-equivalent operations.
/// The tolerance to stale reads is a design property of stigmergic pheromones.
#[derive(Debug)]
pub struct AtomicF64 {
    bits: AtomicU64,
}

impl Default for AtomicF64 {
    fn default() -> Self {
        Self::new(0.0)
    }
}

impl AtomicF64 {
    /// Creates a new `AtomicF64` with the given initial value.
    #[inline]
    pub fn new(val: f64) -> Self {
        Self {
            bits: AtomicU64::new(val.to_bits()),
        }
    }

    /// Loads the current value with `Relaxed` ordering.
    #[inline]
    pub fn load(&self) -> f64 {
        f64::from_bits(self.bits.load(Ordering::Relaxed))
    }

    /// Stores a value with `Relaxed` ordering.
    #[inline]
    pub fn store(&self, val: f64) {
        self.bits.store(val.to_bits(), Ordering::Relaxed);
    }

    /// Atomically adds `delta` to the current value using CAS loop.
    ///
    /// Returns the *new* value after addition. Under contention, the CAS
    /// may retry a few times — this is acceptable for pheromone deposits
    /// where exact precision is not required.
    #[inline]
    pub fn fetch_add(&self, delta: f64) -> f64 {
        loop {
            let old_bits = self.bits.load(Ordering::Relaxed);
            let old_val = f64::from_bits(old_bits);
            let new_val = old_val + delta;
            let new_bits = new_val.to_bits();
            match self.bits.compare_exchange_weak(
                old_bits,
                new_bits,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return new_val,
                Err(_) => continue, // CAS failed, retry
            }
        }
    }
}

// ---------------------------------------------------------------------------
// PheromoneMap — DashMap<(EdgeId, TrailType), AtomicF64> lock-free
// ---------------------------------------------------------------------------

/// Lock-free pheromone map storing f64 intensities per (edge, trail type) pair.
///
/// Uses `DashMap` for concurrent shard-level access and `AtomicF64` for
/// lock-free value updates. No mutex, no RwLock on pheromone values.
pub struct PheromoneMap {
    annotations: DashMap<(EdgeId, TrailType), AtomicF64>,
}

impl std::fmt::Debug for PheromoneMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PheromoneMap")
            .field("len", &self.annotations.len())
            .finish()
    }
}

impl Default for PheromoneMap {
    fn default() -> Self {
        Self::new()
    }
}

impl PheromoneMap {
    /// Creates a new empty pheromone map.
    pub fn new() -> Self {
        Self {
            annotations: DashMap::new(),
        }
    }

    /// Deposits pheromone `delta` on the given (edge, trail) pair.
    ///
    /// Lock-free: uses `AtomicF64::fetch_add`. If the entry doesn't exist
    /// yet, it is created with `delta` as the initial value.
    pub fn deposit(&self, edge: EdgeId, trail: TrailType, delta: f64) {
        // Fast path: entry already exists → atomic add, no lock on the map shard
        if let Some(entry) = self.annotations.get(&(edge, trail)) {
            entry.value().fetch_add(delta);
            return;
        }
        // Slow path: insert new entry. DashMap locks the shard briefly for insert.
        // Use `entry` API to avoid TOCTOU race.
        self.annotations
            .entry((edge, trail))
            .and_modify(|v| {
                v.fetch_add(delta);
            })
            .or_insert_with(|| AtomicF64::new(delta));
    }

    /// Reads the current pheromone value for the given (edge, trail) pair.
    ///
    /// Returns `0.0` if no pheromone has been deposited for this pair.
    /// Uses `Relaxed` ordering — the value may be slightly stale under
    /// concurrent writes, which is acceptable for stigmergic reads.
    pub fn read(&self, edge: EdgeId, trail: TrailType) -> f64 {
        self.annotations
            .get(&(edge, trail))
            .map_or(0.0, |entry| entry.value().load())
    }

    /// Returns a snapshot of all pheromone traces for a specific edge.
    pub fn trace(&self, edge: EdgeId) -> StigmergicTrace {
        StigmergicTrace {
            edge,
            pheromone_query: self.read(edge, TrailType::Query),
            pheromone_mutation: self.read(edge, TrailType::Mutation),
            pheromone_error: self.read(edge, TrailType::Error),
            pheromone_surprise: self.read(edge, TrailType::Surprise),
        }
    }

    /// Returns the number of (edge, trail) entries in the map.
    pub fn len(&self) -> usize {
        self.annotations.len()
    }

    /// Returns true if the map contains no entries.
    pub fn is_empty(&self) -> bool {
        self.annotations.is_empty()
    }

    /// Iterates over all entries and applies an evaporation factor.
    ///
    /// `factor` should be in `(0.0, 1.0)` — each pheromone value is multiplied
    /// by this factor. Values below `threshold` are removed entirely.
    pub fn evaporate(&self, factor: f64, threshold: f64) {
        let mut to_remove = Vec::new();
        for entry in &self.annotations {
            let old = entry.value().load();
            let new = old * factor;
            if new < threshold {
                to_remove.push(*entry.key());
            } else {
                entry.value().store(new);
            }
        }
        for key in to_remove {
            self.annotations.remove(&key);
        }
    }

    /// Applies temporal decay to ALL pheromones by multiplying each by `decay_rate`.
    ///
    /// Default `decay_rate` is `0.95`. After 100 cycles without deposit, a
    /// pheromone of initial value 1.0 decays to ~0.006 (0.95^100 ≈ 0.00592).
    ///
    /// Entries that fall below `1e-6` are pruned to avoid map bloat.
    /// This is the batch async lock-free evaporation called by the homeostasis
    /// scheduler.
    pub fn evaporate_all(&self, decay_rate: f64) {
        self.evaporate(decay_rate, 1e-6);
    }

    /// Returns the maximum pheromone value across all entries, or `0.0` if empty.
    pub fn max_value(&self) -> f64 {
        let mut max = 0.0_f64;
        for entry in &self.annotations {
            let v = entry.value().load();
            if v > max {
                max = v;
            }
        }
        max
    }

    /// Anti-lock-in: adds noise to pheromones that exceed `ratio` of the max value.
    ///
    /// For each pheromone value > `ratio * max`, a deterministic perturbation of
    /// ±`noise_pct` is applied. This breaks dominant paths and encourages exploration.
    ///
    /// `ratio` — threshold as fraction of max (e.g., 0.9 = 90%).
    /// `noise_pct` — noise magnitude as fraction (e.g., 0.05 = ±5%).
    /// `seed` — seed for reproducible noise (use timestamp or counter in prod).
    pub fn inject_noise(&self, ratio: f64, noise_pct: f64, seed: u64) {
        let max = self.max_value();
        if max <= 0.0 {
            return;
        }
        let threshold = max * ratio;
        let mut counter: u64 = seed;
        for entry in &self.annotations {
            let val = entry.value().load();
            if val >= threshold {
                // Simple deterministic hash-based noise — no external RNG dependency.
                // Produces a value in [-1.0, 1.0] from a counter.
                counter = counter
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let noise_unit = ((counter >> 33) as f64 / (u32::MAX as f64)) * 2.0 - 1.0;
                let perturbation = val * noise_pct * noise_unit;
                let new_val = (val + perturbation).max(0.0);
                entry.value().store(new_val);
            }
        }
    }

    /// Drains all entries into a Vec for batch flushing, resetting values to 0.
    ///
    /// Returns `(EdgeId, TrailType, f64)` tuples. The pheromone values are
    /// reset to 0.0 atomically (swap). This is used by the periodic flush
    /// to `EdgeAnnotator`.
    pub fn drain_snapshot(&self) -> Vec<(EdgeId, TrailType, f64)> {
        let mut result = Vec::with_capacity(self.annotations.len());
        for entry in &self.annotations {
            let (edge, trail) = *entry.key();
            // Swap to 0 and capture the old value
            let old_bits = entry
                .value()
                .bits
                .swap(0.0_f64.to_bits(), Ordering::Relaxed);
            let val = f64::from_bits(old_bits);
            if val > 0.0 {
                result.push((edge, trail, val));
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// StigmergicEngine — wraps PheromoneMap + EdgeAnnotator
// ---------------------------------------------------------------------------

/// The stigmergic engine: wraps a [`PheromoneMap`] for fast in-memory deposits
/// and a boxed [`EdgeAnnotator`] for periodic batch persistence.
///
/// The sync to the annotator happens via [`flush()`](StigmergicEngine::flush),
/// which should be called periodically (e.g., every few seconds), NOT on
/// every deposit.
pub struct StigmergicEngine {
    /// In-memory pheromone accumulator (lock-free).
    pheromone_map: PheromoneMap,
    /// Persistent annotation backend.
    annotator: Box<dyn EdgeAnnotator>,
}

impl std::fmt::Debug for StigmergicEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StigmergicEngine")
            .field("pheromone_map", &self.pheromone_map)
            .finish()
    }
}

impl StigmergicEngine {
    /// Creates a new `StigmergicEngine` with the given annotator backend.
    pub fn new(annotator: Box<dyn EdgeAnnotator>) -> Self {
        Self {
            pheromone_map: PheromoneMap::new(),
            annotator,
        }
    }

    /// Deposits pheromone on an edge (in-memory, lock-free).
    ///
    /// The deposit is NOT immediately flushed to the `EdgeAnnotator`.
    /// Call [`flush()`](Self::flush) periodically to persist.
    #[inline]
    pub fn deposit(&self, edge: EdgeId, trail: TrailType, delta: f64) {
        self.pheromone_map.deposit(edge, trail, delta);
    }

    /// Reads the current in-memory pheromone value.
    ///
    /// This reads from the in-memory map, not from the annotator backend.
    /// Values are eventually consistent with the backend after [`flush()`](Self::flush).
    #[inline]
    pub fn read(&self, edge: EdgeId, trail: TrailType) -> f64 {
        self.pheromone_map.read(edge, trail)
    }

    /// Returns the full stigmergic trace snapshot for an edge.
    pub fn trace(&self, edge: EdgeId) -> StigmergicTrace {
        self.pheromone_map.trace(edge)
    }

    /// Batch-flushes accumulated pheromone deltas to the `EdgeAnnotator`.
    ///
    /// This drains the in-memory accumulator and calls `annotate()` for each
    /// (edge, trail, delta) triple. The annotator merges the delta with its
    /// existing value.
    ///
    /// Returns the number of annotations flushed.
    pub fn flush(&self) -> usize {
        let snapshot = self.pheromone_map.drain_snapshot();
        let count = snapshot.len();
        for (edge, trail, delta) in snapshot {
            // Read current value from backend, add delta, write back
            let key = trail.annotation_key();
            let current = self.annotator.get_annotation(edge, key).unwrap_or(0.0);
            self.annotator.annotate(edge, key, current + delta);
        }
        count
    }

    /// Applies evaporation to the in-memory pheromone map.
    ///
    /// `factor` in `(0.0, 1.0)` — each value is multiplied by this factor.
    /// Values below `threshold` are removed.
    pub fn evaporate(&self, factor: f64, threshold: f64) {
        self.pheromone_map.evaporate(factor, threshold);
    }

    /// Applies temporal decay to all pheromones (default decay_rate = 0.95).
    ///
    /// Lock-free batch operation. Entries below 1e-6 are pruned.
    pub fn evaporate_all(&self, decay_rate: f64) {
        self.pheromone_map.evaporate_all(decay_rate);
    }

    /// Anti-lock-in: injects noise into pheromones > 90% of max.
    ///
    /// Breaks dominant paths to encourage exploration.
    pub fn prevent_lock_in(&self, noise_pct: f64, seed: u64) {
        self.pheromone_map.inject_noise(0.9, noise_pct, seed);
    }

    /// Returns a reference to the underlying `PheromoneMap`.
    pub fn pheromone_map(&self) -> &PheromoneMap {
        &self.pheromone_map
    }
}

// ---------------------------------------------------------------------------
// StigmergicQueryListener — QueryObserver adapter
// ---------------------------------------------------------------------------

/// A `QueryObserver` that deposits `pheromone_query` on every edge traversed
/// during query execution.
///
/// Each call to `on_traversal` atomically increments the pheromone value
/// for every edge in the path by a configurable `deposit_delta` (default `1.0`).
///
/// This is lock-free: uses the underlying [`PheromoneMap`]'s `AtomicF64` CAS.
pub struct StigmergicQueryListener {
    engine: Arc<StigmergicEngine>,
    /// Amount of pheromone deposited per edge per traversal.
    deposit_delta: f64,
}

impl StigmergicQueryListener {
    /// Creates a new query listener wrapping the given engine.
    pub fn new(engine: Arc<StigmergicEngine>) -> Self {
        Self {
            engine,
            deposit_delta: 1.0,
        }
    }

    /// Creates a new query listener with a custom deposit delta.
    pub fn with_delta(engine: Arc<StigmergicEngine>, delta: f64) -> Self {
        Self {
            engine,
            deposit_delta: delta,
        }
    }

    /// Returns a reference to the underlying engine.
    pub fn engine(&self) -> &Arc<StigmergicEngine> {
        &self.engine
    }
}

impl crate::engram::traits::QueryObserver for StigmergicQueryListener {
    fn on_query_executed(
        &self,
        _query_text: &str,
        _result_count: usize,
        _duration: std::time::Duration,
    ) {
        // No pheromone deposit for query execution stats — only traversals matter.
    }

    fn on_traversal(&self, path: &[EdgeId]) {
        for &edge in path {
            self.engine
                .deposit(edge, TrailType::Query, self.deposit_delta);
        }
    }
}

// ---------------------------------------------------------------------------
// StigmergicMutationListener — MutationListener adapter
// ---------------------------------------------------------------------------

/// A [`MutationListener`](grafeo_reactive::MutationListener) that deposits
/// `pheromone_mutation` on edges involved in mutation events.
///
/// Reacts to `EdgeCreated`, `EdgeUpdated`, and `EdgeDeleted` events.
/// Node mutations are ignored (no edge to annotate).
///
/// Lock-free deposits via [`PheromoneMap`]'s `AtomicF64`.
pub struct StigmergicMutationListener {
    engine: Arc<StigmergicEngine>,
    /// Amount of pheromone deposited per mutation event.
    deposit_delta: f64,
}

impl StigmergicMutationListener {
    /// Creates a new mutation listener wrapping the given engine.
    pub fn new(engine: Arc<StigmergicEngine>) -> Self {
        Self {
            engine,
            deposit_delta: 1.0,
        }
    }

    /// Creates a new mutation listener with a custom deposit delta.
    pub fn with_delta(engine: Arc<StigmergicEngine>, delta: f64) -> Self {
        Self {
            engine,
            deposit_delta: delta,
        }
    }

    /// Returns a reference to the underlying engine.
    pub fn engine(&self) -> &Arc<StigmergicEngine> {
        &self.engine
    }
}

#[async_trait::async_trait]
impl grafeo_reactive::MutationListener for StigmergicMutationListener {
    fn name(&self) -> &str {
        "stigmergic-mutation-listener"
    }

    async fn on_event(&self, event: &grafeo_reactive::MutationEvent) {
        match event {
            grafeo_reactive::MutationEvent::EdgeCreated { edge } => {
                self.engine
                    .deposit(edge.id, TrailType::Mutation, self.deposit_delta);
            }
            grafeo_reactive::MutationEvent::EdgeUpdated { after, .. } => {
                self.engine
                    .deposit(after.id, TrailType::Mutation, self.deposit_delta);
            }
            grafeo_reactive::MutationEvent::EdgeDeleted { edge } => {
                self.engine
                    .deposit(edge.id, TrailType::Mutation, self.deposit_delta);
            }
            // Node mutations don't directly map to edge pheromones.
            _ => {}
        }
    }

    async fn on_batch(&self, events: &[grafeo_reactive::MutationEvent]) {
        for event in events {
            self.on_event(event).await;
        }
    }
}

// ---------------------------------------------------------------------------
// StigmergicFormationBridge — deposit error/surprise on engram formation
// ---------------------------------------------------------------------------

/// Deposits `pheromone_error` and `pheromone_surprise` on edges belonging to
/// an engram's ensemble when the engram is formed with error or surprise context.
///
/// This is not a trait implementation — it's a utility called by the formation
/// pipeline when an engram is created. The caller provides the edge set and
/// the context (scar/surprise).
pub struct StigmergicFormationBridge {
    engine: Arc<StigmergicEngine>,
    /// Delta deposited per edge for error pheromones.
    error_delta: f64,
    /// Delta deposited per edge for surprise pheromones.
    surprise_delta: f64,
    /// Prediction error threshold above which pheromone_surprise is deposited.
    surprise_threshold: f64,
}

impl StigmergicFormationBridge {
    /// Creates a new formation bridge with default deltas.
    pub fn new(engine: Arc<StigmergicEngine>) -> Self {
        Self {
            engine,
            error_delta: 2.0,
            surprise_delta: 1.5,
            surprise_threshold: 0.5,
        }
    }

    /// Creates a formation bridge with custom parameters.
    pub fn with_params(
        engine: Arc<StigmergicEngine>,
        error_delta: f64,
        surprise_delta: f64,
        surprise_threshold: f64,
    ) -> Self {
        Self {
            engine,
            error_delta,
            surprise_delta,
            surprise_threshold,
        }
    }

    /// Deposits `pheromone_error` on all edges in the ensemble when the engram
    /// is associated with a scar (error context).
    ///
    /// `ensemble_edges` — the edges belonging to the engram's node ensemble.
    pub fn on_error_engram(&self, ensemble_edges: &[EdgeId]) {
        for &edge in ensemble_edges {
            self.engine
                .deposit(edge, TrailType::Error, self.error_delta);
        }
    }

    /// Deposits `pheromone_surprise` on all edges in the ensemble when the
    /// prediction error exceeds the surprise threshold.
    ///
    /// `ensemble_edges` — the edges belonging to the engram's node ensemble.
    /// `prediction_error` — the PE magnitude that triggered formation.
    pub fn on_surprise_engram(&self, ensemble_edges: &[EdgeId], prediction_error: f64) {
        if prediction_error >= self.surprise_threshold {
            for &edge in ensemble_edges {
                self.engine
                    .deposit(edge, TrailType::Surprise, self.surprise_delta);
            }
        }
    }

    /// Combined handler: deposits error and/or surprise pheromones based on context.
    ///
    /// - If `has_scar` is true → deposits `pheromone_error` on all edges.
    /// - If `prediction_error >= surprise_threshold` → deposits `pheromone_surprise`.
    pub fn on_engram_formed(
        &self,
        ensemble_edges: &[EdgeId],
        has_scar: bool,
        prediction_error: f64,
    ) {
        if has_scar {
            self.on_error_engram(ensemble_edges);
        }
        self.on_surprise_engram(ensemble_edges, prediction_error);
    }

    /// Returns a reference to the underlying engine.
    pub fn engine(&self) -> &Arc<StigmergicEngine> {
        &self.engine
    }

    /// Returns the surprise threshold.
    pub fn surprise_threshold(&self) -> f64 {
        self.surprise_threshold
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    // ---- AtomicF64 unit tests ----

    #[test]
    fn atomic_f64_basic_operations() {
        let a = AtomicF64::new(1.0);
        assert_eq!(a.load(), 1.0);

        a.store(42.0);
        assert_eq!(a.load(), 42.0);

        let new = a.fetch_add(8.0);
        assert_eq!(new, 50.0);
        assert_eq!(a.load(), 50.0);
    }

    // ---- PheromoneMap unit tests ----

    #[test]
    fn pheromone_map_deposit_and_read() {
        let map = PheromoneMap::new();
        let edge = EdgeId::new(1);

        assert_eq!(map.read(edge, TrailType::Query), 0.0);

        map.deposit(edge, TrailType::Query, 1.0);
        assert_eq!(map.read(edge, TrailType::Query), 1.0);

        map.deposit(edge, TrailType::Query, 0.5);
        assert_eq!(map.read(edge, TrailType::Query), 1.5);

        // Other trail types unaffected
        assert_eq!(map.read(edge, TrailType::Mutation), 0.0);
    }

    #[test]
    fn pheromone_map_trace() {
        let map = PheromoneMap::new();
        let edge = EdgeId::new(42);

        map.deposit(edge, TrailType::Query, 1.0);
        map.deposit(edge, TrailType::Error, 0.3);

        let trace = map.trace(edge);
        assert_eq!(trace.edge, edge);
        assert_eq!(trace.pheromone_query, 1.0);
        assert_eq!(trace.pheromone_mutation, 0.0);
        assert_eq!(trace.pheromone_error, 0.3);
        assert_eq!(trace.pheromone_surprise, 0.0);
    }

    #[test]
    fn pheromone_map_concurrent_deposit() {
        let map = Arc::new(PheromoneMap::new());
        let edge = EdgeId::new(1);
        let deposits_per_thread = 1000;
        let num_threads = 4;

        let handles: Vec<_> = (0..num_threads)
            .map(|_| {
                let map = Arc::clone(&map);
                std::thread::spawn(move || {
                    for _ in 0..deposits_per_thread {
                        map.deposit(edge, TrailType::Query, 1.0);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        let expected = (num_threads * deposits_per_thread) as f64;
        let actual = map.read(edge, TrailType::Query);
        // Exact match expected since CAS loop guarantees correctness
        assert!(
            (actual - expected).abs() < 1.0,
            "Expected ~{expected}, got {actual}"
        );
    }

    #[test]
    fn pheromone_map_evaporate() {
        let map = PheromoneMap::new();
        let e1 = EdgeId::new(1);
        let e2 = EdgeId::new(2);

        map.deposit(e1, TrailType::Query, 10.0);
        map.deposit(e2, TrailType::Mutation, 0.01);

        map.evaporate(0.5, 0.1);

        assert_eq!(map.read(e1, TrailType::Query), 5.0);
        // e2's value (0.005) is below threshold (0.1), should be removed
        assert_eq!(map.read(e2, TrailType::Mutation), 0.0);
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn pheromone_map_drain_snapshot() {
        let map = PheromoneMap::new();
        let edge = EdgeId::new(1);

        map.deposit(edge, TrailType::Query, 5.0);
        map.deposit(edge, TrailType::Error, 2.0);

        let snapshot = map.drain_snapshot();
        assert_eq!(snapshot.len(), 2);

        // After drain, values should be 0
        assert_eq!(map.read(edge, TrailType::Query), 0.0);
        assert_eq!(map.read(edge, TrailType::Error), 0.0);
    }

    // ---- InMemoryEdgeAnnotator for testing ----

    struct InMemoryAnnotator {
        data: DashMap<(EdgeId, String), f64>,
    }

    impl InMemoryAnnotator {
        fn new() -> Self {
            Self {
                data: DashMap::new(),
            }
        }
    }

    impl EdgeAnnotator for InMemoryAnnotator {
        fn annotate(&self, edge: EdgeId, key: &str, value: f64) {
            self.data.insert((edge, key.to_string()), value);
        }

        fn get_annotation(&self, edge: EdgeId, key: &str) -> Option<f64> {
            self.data.get(&(edge, key.to_string())).map(|v| *v)
        }

        fn remove_annotation(&self, edge: EdgeId, key: &str) {
            self.data.remove(&(edge, key.to_string()));
        }
    }

    // ---- StigmergicEngine tests ----

    #[test]
    fn stigmergic_engine_deposit_read_flush() {
        let annotator = InMemoryAnnotator::new();
        let engine = StigmergicEngine::new(Box::new(annotator));

        let edge = EdgeId::new(10);

        // Deposit in-memory
        engine.deposit(edge, TrailType::Query, 3.0);
        engine.deposit(edge, TrailType::Query, 2.0);
        engine.deposit(edge, TrailType::Mutation, 1.0);

        // Read from in-memory map
        assert_eq!(engine.read(edge, TrailType::Query), 5.0);
        assert_eq!(engine.read(edge, TrailType::Mutation), 1.0);

        // Flush to annotator
        let flushed = engine.flush();
        assert_eq!(flushed, 2); // 2 (edge, trail) pairs

        // After flush, in-memory values are reset
        assert_eq!(engine.read(edge, TrailType::Query), 0.0);
    }

    #[test]
    fn stigmergic_engine_trace() {
        let annotator = InMemoryAnnotator::new();
        let engine = StigmergicEngine::new(Box::new(annotator));

        let edge = EdgeId::new(99);
        engine.deposit(edge, TrailType::Surprise, 7.5);

        let trace = engine.trace(edge);
        assert_eq!(trace.pheromone_surprise, 7.5);
        assert_eq!(trace.pheromone_query, 0.0);
    }

    #[test]
    fn stigmergic_engine_multiple_flushes_accumulate() {
        let annotator = Arc::new(InMemoryAnnotator::new());

        // We need to clone into a Box<dyn EdgeAnnotator>
        // Use a wrapper that delegates to the Arc
        struct ArcAnnotator(Arc<InMemoryAnnotator>);
        impl EdgeAnnotator for ArcAnnotator {
            fn annotate(&self, edge: EdgeId, key: &str, value: f64) {
                self.0.annotate(edge, key, value);
            }
            fn get_annotation(&self, edge: EdgeId, key: &str) -> Option<f64> {
                self.0.get_annotation(edge, key)
            }
            fn remove_annotation(&self, edge: EdgeId, key: &str) {
                self.0.remove_annotation(edge, key);
            }
        }

        let engine = StigmergicEngine::new(Box::new(ArcAnnotator(Arc::clone(&annotator))));
        let edge = EdgeId::new(5);

        // First batch
        engine.deposit(edge, TrailType::Query, 3.0);
        engine.flush();

        // Second batch
        engine.deposit(edge, TrailType::Query, 2.0);
        engine.flush();

        // Annotator should have accumulated value
        let val = annotator
            .get_annotation(edge, TrailType::Query.annotation_key())
            .unwrap();
        assert_eq!(val, 5.0);
    }

    // ---- StigmergicQueryListener tests ----

    fn make_engine() -> Arc<StigmergicEngine> {
        Arc::new(StigmergicEngine::new(Box::new(InMemoryAnnotator::new())))
    }

    #[test]
    fn stigmergic_query_listener_deposits_on_traversal() {
        use crate::engram::traits::QueryObserver;

        let engine = make_engine();
        let listener = StigmergicQueryListener::new(Arc::clone(&engine));

        let edges = vec![EdgeId::new(10), EdgeId::new(20), EdgeId::new(30)];
        listener.on_traversal(&edges);

        assert_eq!(engine.read(EdgeId::new(10), TrailType::Query), 1.0);
        assert_eq!(engine.read(EdgeId::new(20), TrailType::Query), 1.0);
        assert_eq!(engine.read(EdgeId::new(30), TrailType::Query), 1.0);
        // Other trail types untouched
        assert_eq!(engine.read(EdgeId::new(10), TrailType::Mutation), 0.0);
    }

    #[test]
    fn stigmergic_query_listener_custom_delta() {
        use crate::engram::traits::QueryObserver;

        let engine = make_engine();
        let listener = StigmergicQueryListener::with_delta(Arc::clone(&engine), 0.5);

        let edges = vec![EdgeId::new(1)];
        listener.on_traversal(&edges);
        listener.on_traversal(&edges);

        assert!((engine.read(EdgeId::new(1), TrailType::Query) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn stigmergic_query_listener_empty_path_is_noop() {
        use crate::engram::traits::QueryObserver;

        let engine = make_engine();
        let listener = StigmergicQueryListener::new(Arc::clone(&engine));

        listener.on_traversal(&[]);
        assert!(engine.pheromone_map().is_empty());
    }

    #[test]
    fn stigmergic_query_listener_multiple_traversals_accumulate() {
        use crate::engram::traits::QueryObserver;

        let engine = make_engine();
        let listener = StigmergicQueryListener::new(Arc::clone(&engine));

        let path = vec![EdgeId::new(1), EdgeId::new(2)];
        for _ in 0..5 {
            listener.on_traversal(&path);
        }
        assert_eq!(engine.read(EdgeId::new(1), TrailType::Query), 5.0);
        assert_eq!(engine.read(EdgeId::new(2), TrailType::Query), 5.0);
    }

    // ---- StigmergicMutationListener tests ----

    #[tokio::test]
    async fn stigmergic_mutation_listener_on_edge_created() {
        use grafeo_reactive::MutationListener;

        let engine = make_engine();
        let listener = StigmergicMutationListener::new(Arc::clone(&engine));

        let event = grafeo_reactive::MutationEvent::EdgeCreated {
            edge: grafeo_reactive::EdgeSnapshot {
                id: EdgeId::new(42),
                src: grafeo_common::types::NodeId(1),
                dst: grafeo_common::types::NodeId(2),
                edge_type: arcstr::literal!("KNOWS"),
                properties: vec![],
            },
        };

        listener.on_event(&event).await;

        assert_eq!(engine.read(EdgeId::new(42), TrailType::Mutation), 1.0);
        assert_eq!(engine.read(EdgeId::new(42), TrailType::Query), 0.0);
    }

    #[tokio::test]
    async fn stigmergic_mutation_listener_on_edge_updated() {
        use grafeo_reactive::MutationListener;

        let engine = make_engine();
        let listener = StigmergicMutationListener::new(Arc::clone(&engine));

        let snapshot = grafeo_reactive::EdgeSnapshot {
            id: EdgeId::new(7),
            src: grafeo_common::types::NodeId(1),
            dst: grafeo_common::types::NodeId(2),
            edge_type: arcstr::literal!("FOLLOWS"),
            properties: vec![],
        };

        let event = grafeo_reactive::MutationEvent::EdgeUpdated {
            before: snapshot.clone(),
            after: snapshot,
        };

        listener.on_event(&event).await;
        assert_eq!(engine.read(EdgeId::new(7), TrailType::Mutation), 1.0);
    }

    #[tokio::test]
    async fn stigmergic_mutation_listener_ignores_node_events() {
        use grafeo_reactive::MutationListener;

        let engine = make_engine();
        let listener = StigmergicMutationListener::new(Arc::clone(&engine));

        let event = grafeo_reactive::MutationEvent::NodeCreated {
            node: grafeo_reactive::NodeSnapshot {
                id: grafeo_common::types::NodeId(1),
                labels: smallvec::smallvec![],
                properties: vec![],
            },
        };

        listener.on_event(&event).await;
        // No edges deposited
        assert!(engine.pheromone_map().is_empty());
    }

    #[tokio::test]
    async fn stigmergic_mutation_listener_batch_deposits() {
        use grafeo_reactive::MutationListener;

        let engine = make_engine();
        let listener = StigmergicMutationListener::new(Arc::clone(&engine));

        let make_edge_event = |id: u64| grafeo_reactive::MutationEvent::EdgeCreated {
            edge: grafeo_reactive::EdgeSnapshot {
                id: EdgeId::new(id),
                src: grafeo_common::types::NodeId(1),
                dst: grafeo_common::types::NodeId(2),
                edge_type: arcstr::literal!("REL"),
                properties: vec![],
            },
        };

        let events = vec![
            make_edge_event(100),
            make_edge_event(200),
            make_edge_event(300),
        ];

        listener.on_batch(&events).await;

        assert_eq!(engine.read(EdgeId::new(100), TrailType::Mutation), 1.0);
        assert_eq!(engine.read(EdgeId::new(200), TrailType::Mutation), 1.0);
        assert_eq!(engine.read(EdgeId::new(300), TrailType::Mutation), 1.0);
    }

    // ---- StigmergicFormationBridge tests ----

    #[test]
    fn stigmergic_formation_bridge_error_deposits() {
        let engine = make_engine();
        let bridge = StigmergicFormationBridge::new(Arc::clone(&engine));

        let edges = vec![EdgeId::new(1), EdgeId::new(2), EdgeId::new(3)];
        bridge.on_error_engram(&edges);

        assert_eq!(engine.read(EdgeId::new(1), TrailType::Error), 2.0); // default error_delta
        assert_eq!(engine.read(EdgeId::new(2), TrailType::Error), 2.0);
        assert_eq!(engine.read(EdgeId::new(3), TrailType::Error), 2.0);
        // Other trails unaffected
        assert_eq!(engine.read(EdgeId::new(1), TrailType::Query), 0.0);
    }

    #[test]
    fn stigmergic_formation_bridge_surprise_deposits_above_threshold() {
        let engine = make_engine();
        let bridge = StigmergicFormationBridge::new(Arc::clone(&engine));

        let edges = vec![EdgeId::new(10), EdgeId::new(20)];
        bridge.on_surprise_engram(&edges, 0.8); // above default threshold 0.5

        assert_eq!(engine.read(EdgeId::new(10), TrailType::Surprise), 1.5); // default surprise_delta
        assert_eq!(engine.read(EdgeId::new(20), TrailType::Surprise), 1.5);
    }

    #[test]
    fn stigmergic_formation_bridge_surprise_ignored_below_threshold() {
        let engine = make_engine();
        let bridge = StigmergicFormationBridge::new(Arc::clone(&engine));

        let edges = vec![EdgeId::new(10)];
        bridge.on_surprise_engram(&edges, 0.3); // below default threshold 0.5

        assert_eq!(engine.read(EdgeId::new(10), TrailType::Surprise), 0.0);
    }

    #[test]
    fn stigmergic_formation_bridge_combined_error_and_surprise() {
        let engine = make_engine();
        let bridge = StigmergicFormationBridge::new(Arc::clone(&engine));

        let edges = vec![EdgeId::new(5), EdgeId::new(6)];
        bridge.on_engram_formed(&edges, true, 0.9);

        // Both error and surprise should be deposited
        assert_eq!(engine.read(EdgeId::new(5), TrailType::Error), 2.0);
        assert_eq!(engine.read(EdgeId::new(5), TrailType::Surprise), 1.5);
        assert_eq!(engine.read(EdgeId::new(6), TrailType::Error), 2.0);
        assert_eq!(engine.read(EdgeId::new(6), TrailType::Surprise), 1.5);
    }

    #[test]
    fn stigmergic_formation_bridge_scar_only_no_surprise() {
        let engine = make_engine();
        let bridge = StigmergicFormationBridge::new(Arc::clone(&engine));

        let edges = vec![EdgeId::new(1)];
        bridge.on_engram_formed(&edges, true, 0.2); // scar but low PE

        assert_eq!(engine.read(EdgeId::new(1), TrailType::Error), 2.0);
        assert_eq!(engine.read(EdgeId::new(1), TrailType::Surprise), 0.0);
    }

    #[test]
    fn stigmergic_formation_bridge_custom_params() {
        let engine = make_engine();
        let bridge = StigmergicFormationBridge::with_params(
            Arc::clone(&engine),
            5.0, // error_delta
            3.0, // surprise_delta
            0.7, // surprise_threshold
        );

        let edges = vec![EdgeId::new(1)];
        bridge.on_engram_formed(&edges, true, 0.8);

        assert_eq!(engine.read(EdgeId::new(1), TrailType::Error), 5.0);
        assert_eq!(engine.read(EdgeId::new(1), TrailType::Surprise), 3.0);
    }

    // ---- Evaporation tests ----

    #[test]
    fn evaporation_100_cycles_decays_to_near_zero() {
        let map = PheromoneMap::new();
        let edge = EdgeId::new(1);
        map.deposit(edge, TrailType::Query, 1.0);

        // Apply 100 cycles of decay at 0.95
        for _ in 0..100 {
            map.evaporate_all(0.95);
        }

        let val = map.read(edge, TrailType::Query);
        // 0.95^100 ≈ 0.00592
        assert!(val < 0.007, "After 100 cycles, expected < 0.007, got {val}");
        assert!(val > 0.005, "After 100 cycles, expected > 0.005, got {val}");
    }

    #[test]
    fn evaporation_removes_entries_below_threshold() {
        let map = PheromoneMap::new();
        let edge = EdgeId::new(1);
        map.deposit(edge, TrailType::Query, 1e-5);

        // A single evaporate at 0.95 → 9.5e-6 → below 1e-6 threshold? No.
        // Need more cycles.
        // 1e-5 * 0.95^n < 1e-6 → 0.95^n < 0.1 → n > log(0.1)/log(0.95) ≈ 44.9
        for _ in 0..50 {
            map.evaporate_all(0.95);
        }

        assert_eq!(map.len(), 0, "Entry should be pruned after enough decay");
    }

    #[test]
    fn evaporate_all_via_engine() {
        let engine = StigmergicEngine::new(Box::new(InMemoryAnnotator::new()));
        let edge = EdgeId::new(1);
        engine.deposit(edge, TrailType::Query, 1.0);

        for _ in 0..100 {
            engine.evaporate_all(0.95);
        }

        let val = engine.read(edge, TrailType::Query);
        assert!(val < 0.007, "Expected < 0.007, got {val}");
        assert!(val > 0.005, "Expected > 0.005, got {val}");
    }

    // ---- Anti-lock-in tests ----

    #[test]
    fn anti_lock_in_injects_noise_on_strong_pheromones() {
        let map = PheromoneMap::new();

        // Create a dominant pheromone and a weak one
        map.deposit(EdgeId::new(1), TrailType::Query, 100.0); // strong
        map.deposit(EdgeId::new(2), TrailType::Query, 50.0); // below 90% of max (90.0)
        map.deposit(EdgeId::new(3), TrailType::Query, 95.0); // above 90% threshold

        let original_strong = map.read(EdgeId::new(1), TrailType::Query);
        let original_weak = map.read(EdgeId::new(2), TrailType::Query);
        let original_high = map.read(EdgeId::new(3), TrailType::Query);

        map.inject_noise(0.9, 0.05, 42);

        let after_strong = map.read(EdgeId::new(1), TrailType::Query);
        let after_weak = map.read(EdgeId::new(2), TrailType::Query);
        let after_high = map.read(EdgeId::new(3), TrailType::Query);

        // Strong pheromone (100.0) should have changed by ±5%
        assert!(
            (after_strong - original_strong).abs() <= original_strong * 0.05 + f64::EPSILON,
            "Strong pheromone should change by at most ±5%, delta = {}",
            (after_strong - original_strong).abs()
        );
        assert!(
            (after_strong - original_strong).abs() > 0.0,
            "Strong pheromone should have been perturbed"
        );

        // Weak pheromone (50.0 < 90.0) should NOT be affected
        assert_eq!(
            after_weak, original_weak,
            "Weak pheromone should not be affected"
        );

        // High pheromone (95.0 >= 90.0) should also be perturbed
        assert!(
            (after_high - original_high).abs() <= original_high * 0.05 + f64::EPSILON,
            "High pheromone should change by at most ±5%"
        );
    }

    #[test]
    fn anti_lock_in_noop_on_empty_map() {
        let map = PheromoneMap::new();
        map.inject_noise(0.9, 0.05, 0); // should not panic
        assert!(map.is_empty());
    }

    #[test]
    fn anti_lock_in_via_engine() {
        let engine = StigmergicEngine::new(Box::new(InMemoryAnnotator::new()));
        engine.deposit(EdgeId::new(1), TrailType::Query, 100.0);

        let before = engine.read(EdgeId::new(1), TrailType::Query);
        engine.prevent_lock_in(0.05, 123);
        let after = engine.read(EdgeId::new(1), TrailType::Query);

        assert!(
            (after - before).abs() <= before * 0.05 + f64::EPSILON,
            "Perturbation should be within ±5%"
        );
        assert!(
            (after - before).abs() > 0.0,
            "Pheromone max should have been perturbed"
        );
    }
}
