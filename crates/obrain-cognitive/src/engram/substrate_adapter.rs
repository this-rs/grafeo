//! Bridge between cognitive's `VectorIndex` trait (f64 / String IDs / distance)
//! and substrate's `VectorIndex` trait (f32 / u32 NodeOffset / cosine).
//!
//! ## Why an adapter, not a single trait?
//!
//! The two traits capture genuinely different abstractions:
//!
//! - **Cognitive** talks in *opaque string identifiers* (engram IDs, typically
//!   UUID-ish) and *f64* vectors because the engram layer is tier-agnostic —
//!   it may run against a brute-force HashMap (tests), a future external ANN,
//!   or substrate.
//! - **Substrate** talks in *u32 node offsets* (raw positions in the mmap'd
//!   record file) and *f32* vectors because the L0 / L1 / L2 tiers are built
//!   directly from f32 embeddings and indexed by offset for zero-copy access.
//!
//! Collapsing the two would either force substrate to carry a String→u32 map
//! (defeats the compactness gain) or force cognitive to understand substrate
//! offsets (breaks the Option-C tier boundary). So we keep the two traits and
//! bridge them here, on the cognitive side where the adapter dependency is
//! acceptable.
//!
//! ## Semantic mapping
//!
//! | Cognitive                         | Substrate                         |
//! |-----------------------------------|-----------------------------------|
//! | `upsert(id: &str, v: &[f64])`     | `insert(offset: u32, v: &[f32])`  |
//! | `remove(id: &str)`                | `delete(offset: u32)`             |
//! | `nearest(q, k) -> (String, f64)`  | `search_top_k(q, k) -> (u32, f32)`|
//! | `distance` ∈ [0, 2]               | `cosine` ∈ [-1, 1]                |
//!
//! Distance is derived as `1.0 - cosine`, matching the formula used by the
//! brute-force `InMemoryVectorIndex` so callers see the same semantics.
//!
//! ## ID allocation
//!
//! The adapter maintains a bijective map `engram_id_str ↔ NodeOffset`. The
//! first `upsert` for a given string allocates the next free offset (monotone
//! u32 counter); subsequent upserts reuse the offset so the update fast-path
//! in `SubstrateTieredIndex` kicks in. `remove` releases the binding but
//! does *not* reuse the offset — reissuing the offset after a delete could
//! surface stale search hits if a scan were in flight.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use obrain_cognitive::engram::substrate_adapter::SubstrateVectorIndexAdapter;
//! use obrain_substrate::retrieval::SubstrateTieredIndex;
//! use std::sync::Arc;
//!
//! let inner = Arc::new(SubstrateTieredIndex::new(384));
//! let adapter = SubstrateVectorIndexAdapter::new(inner);
//! // `adapter` is now `impl VectorIndex` — plug into EngramManager.
//! ```

use super::traits::VectorIndex;
use obrain_substrate::retrieval::{NodeOffset, VectorIndex as SubstrateVectorIndex};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

/// Adapter that makes a substrate tiered index usable through the cognitive
/// `VectorIndex` trait.
///
/// Cheap to clone (it's just three `Arc`s). The inner substrate index is
/// shared across clones — so cloning and handing a clone to each subsystem
/// works as expected.
#[derive(Clone)]
pub struct SubstrateVectorIndexAdapter {
    inner: Arc<dyn SubstrateVectorIndex>,
    id_map: Arc<RwLock<IdMap>>,
    next_offset: Arc<AtomicU32>,
}

/// Bijective String ↔ NodeOffset mapping.
#[derive(Default)]
struct IdMap {
    str_to_off: HashMap<String, NodeOffset>,
    off_to_str: HashMap<NodeOffset, String>,
}

impl SubstrateVectorIndexAdapter {
    /// Wrap an existing substrate `VectorIndex` (typically a
    /// `SubstrateTieredIndex`) behind the cognitive trait.
    pub fn new(inner: Arc<dyn SubstrateVectorIndex>) -> Self {
        Self {
            inner,
            id_map: Arc::new(RwLock::new(IdMap::default())),
            next_offset: Arc::new(AtomicU32::new(0)),
        }
    }

    /// Access the raw substrate index (for callers who want to bypass the
    /// string-ID layer — e.g. bulk rebuild from offsets directly).
    pub fn inner(&self) -> &Arc<dyn SubstrateVectorIndex> {
        &self.inner
    }

    /// Allocate an offset for `id` if none exists, or return the existing one.
    fn resolve_or_allocate(&self, id: &str) -> NodeOffset {
        // Fast path: already mapped.
        {
            let guard = self.id_map.read();
            if let Some(&off) = guard.str_to_off.get(id) {
                return off;
            }
        }
        // Slow path: allocate under write lock. Double-check in case of
        // concurrent allocators.
        let mut guard = self.id_map.write();
        if let Some(&off) = guard.str_to_off.get(id) {
            return off;
        }
        let off = self.next_offset.fetch_add(1, Ordering::SeqCst);
        guard.str_to_off.insert(id.to_string(), off);
        guard.off_to_str.insert(off, id.to_string());
        off
    }

    fn resolve(&self, id: &str) -> Option<NodeOffset> {
        self.id_map.read().str_to_off.get(id).copied()
    }

    fn label(&self, offset: NodeOffset) -> Option<String> {
        self.id_map.read().off_to_str.get(&offset).cloned()
    }
}

impl VectorIndex for SubstrateVectorIndexAdapter {
    fn upsert(&self, id: &str, vector: &[f64]) {
        let offset = self.resolve_or_allocate(id);
        // Down-cast f64 → f32. Substrate embeddings live in f32 end-to-end,
        // so the cast is the canonical normalization step.
        let f32_vec: Vec<f32> = vector.iter().map(|&x| x as f32).collect();
        self.inner.insert(offset, &f32_vec);
    }

    fn nearest(&self, query: &[f64], k: usize) -> Vec<(String, f64)> {
        let f32_query: Vec<f32> = query.iter().map(|&x| x as f32).collect();
        let hits = self.inner.search_top_k(&f32_query, k);
        let mut out = Vec::with_capacity(hits.len());
        for (offset, cosine) in hits {
            // An unresolved offset would mean the substrate index and the
            // adapter map drifted. Skip defensively rather than panic — a
            // dangling result is less bad than a crash on a hot path.
            if let Some(label) = self.label(offset) {
                // Cognitive's trait returns distance, not cosine. Use the
                // same `1 - cosine` convention as `InMemoryVectorIndex`.
                let distance = 1.0 - cosine as f64;
                out.push((label, distance));
            }
        }
        out
    }

    fn remove(&self, id: &str) {
        let Some(offset) = self.resolve(id) else {
            return;
        };
        self.inner.delete(offset);
        let mut guard = self.id_map.write();
        guard.str_to_off.remove(id);
        guard.off_to_str.remove(&offset);
        // Note: the offset counter is *not* rewound. Offsets are permanently
        // burned once allocated — this keeps concurrent scans correct.
    }

    fn dimensions(&self) -> usize {
        self.inner.dim()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use obrain_substrate::retrieval::SubstrateTieredIndex;
    use obrain_substrate::tiers::L2_DIM;

    const DIM: usize = L2_DIM; // 384 — substrate's fixed L2 width

    fn emb(seed: u32) -> Vec<f64> {
        (0..DIM)
            .map(|i| (seed as f64 * 0.31 + i as f64 * 0.17).sin())
            .collect()
    }

    fn make_adapter() -> SubstrateVectorIndexAdapter {
        let inner: Arc<dyn SubstrateVectorIndex> = Arc::new(SubstrateTieredIndex::new(DIM));
        SubstrateVectorIndexAdapter::new(inner)
    }

    #[test]
    fn empty_adapter_reports_substrate_dim() {
        let a = make_adapter();
        assert_eq!(a.dimensions(), DIM);
        assert!(a.nearest(&emb(0), 5).is_empty());
    }

    #[test]
    fn upsert_then_self_recall_returns_string_id() {
        let a = make_adapter();
        a.upsert("engram-alpha", &emb(1));
        a.upsert("engram-beta", &emb(2));
        let hits = a.nearest(&emb(1), 2);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].0, "engram-alpha", "self-recall must be top-1");
        // Distance for self-recall should be near 0 (cosine ≈ 1 → dist ≈ 0).
        assert!(hits[0].1 < 0.01, "self-distance = {}", hits[0].1);
    }

    #[test]
    fn upsert_reuses_offset_on_second_call() {
        let a = make_adapter();
        a.upsert("engram-x", &emb(5));
        let off_before = a.resolve("engram-x").unwrap();
        a.upsert("engram-x", &emb(99)); // overwrite
        let off_after = a.resolve("engram-x").unwrap();
        assert_eq!(off_before, off_after, "same id must reuse offset");

        // The update must be visible — querying for emb(99) should now win.
        let hits = a.nearest(&emb(99), 1);
        assert_eq!(hits[0].0, "engram-x");
        assert!(hits[0].1 < 0.01);
    }

    #[test]
    fn remove_evicts_the_id() {
        let a = make_adapter();
        for n in 0..4 {
            a.upsert(&format!("engram-{n}"), &emb(n));
        }
        a.remove("engram-1");
        let hits = a.nearest(&emb(1), 4);
        assert!(
            hits.iter().all(|(id, _)| id != "engram-1"),
            "removed id surfaced: {hits:?}"
        );
        assert!(a.resolve("engram-1").is_none());
    }

    #[test]
    fn remove_missing_id_is_noop() {
        let a = make_adapter();
        a.upsert("engram-only", &emb(7));
        a.remove("engram-nope"); // never inserted
        let hits = a.nearest(&emb(7), 1);
        assert_eq!(hits[0].0, "engram-only");
    }

    #[test]
    fn offsets_are_not_reused_after_remove() {
        let a = make_adapter();
        a.upsert("a", &emb(1));
        a.upsert("b", &emb(2));
        let off_b = a.resolve("b").unwrap();
        a.remove("b");
        a.upsert("c", &emb(3));
        let off_c = a.resolve("c").unwrap();
        assert_ne!(
            off_b, off_c,
            "a new id must not inherit a deleted offset (stale-scan hazard)"
        );
    }

    #[test]
    fn distance_monotonicity_matches_cosine_ordering() {
        // Two near-duplicate embeddings and one distant one. The ordering by
        // distance must match the ordering by cosine.
        let a = make_adapter();
        a.upsert("near-1", &emb(100));
        let mut near_2 = emb(100);
        // Perturb the second embedding very slightly.
        for x in near_2.iter_mut().take(10) {
            *x += 0.001;
        }
        a.upsert("near-2", &near_2);
        a.upsert("far", &emb(9999));

        let hits = a.nearest(&emb(100), 3);
        let ids: Vec<String> = hits.iter().map(|(id, _)| id.clone()).collect();
        // near-1 (exact self) must come before near-2, both before far.
        let pos_near1 = ids.iter().position(|s| s == "near-1").unwrap();
        let pos_near2 = ids.iter().position(|s| s == "near-2").unwrap();
        let pos_far = ids.iter().position(|s| s == "far").unwrap();
        assert!(pos_near1 < pos_near2);
        assert!(pos_near2 < pos_far);
        // And distances must be monotone increasing.
        assert!(hits[0].1 <= hits[1].1);
        assert!(hits[1].1 <= hits[2].1);
    }

    #[test]
    fn trait_object_usable() {
        // Compile-time check: the adapter fits behind `Arc<dyn VectorIndex>`,
        // the shape EngramManager uses.
        let adapter: Arc<dyn VectorIndex> = Arc::new(make_adapter());
        adapter.upsert("foo", &emb(1));
        let hits = adapter.nearest(&emb(1), 1);
        assert_eq!(hits[0].0, "foo");
    }
}
