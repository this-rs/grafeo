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
        self.pheromone_query + self.pheromone_mutation + self.pheromone_error + self.pheromone_surprise
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
        for entry in self.annotations.iter() {
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

    /// Drains all entries into a Vec for batch flushing, resetting values to 0.
    ///
    /// Returns `(EdgeId, TrailType, f64)` tuples. The pheromone values are
    /// reset to 0.0 atomically (swap). This is used by the periodic flush
    /// to `EdgeAnnotator`.
    pub fn drain_snapshot(&self) -> Vec<(EdgeId, TrailType, f64)> {
        let mut result = Vec::with_capacity(self.annotations.len());
        for entry in self.annotations.iter() {
            let (edge, trail) = *entry.key();
            // Swap to 0 and capture the old value
            let old_bits = entry.value().bits.swap(0.0_f64.to_bits(), Ordering::Relaxed);
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

    /// Returns a reference to the underlying `PheromoneMap`.
    pub fn pheromone_map(&self) -> &PheromoneMap {
        &self.pheromone_map
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
}
