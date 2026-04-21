//! Retrieval façade — `VectorIndex` trait + substrate-backed implementation.
//!
//! The trait defines the public retrieval contract: search top-K by cosine
//! similarity over a corpus of f32 embeddings, plus incremental insert /
//! delete / rebuild. The implementation ([`SubstrateTieredIndex`]) uses the
//! [`tiers`](crate::tiers) L0 / L1 / L2 cascade and the bucket-capture
//! scanner from [`tiered_scan`](crate::tiered_scan) to meet the T8 latency
//! gate (≤ 1 ms p95 on 10⁶ nodes single-thread on AVX-512 hardware;
//! bandwidth-bound ~850 µs on Apple M2).
//!
//! ## Why abstract behind a trait?
//!
//! Callers (EngramRetriever, search modules, context loader) should bind
//! to the *behavior* — "find the k closest vectors" — rather than the
//! concrete representation. That lets us swap the in-memory-Vec impl for
//! a fully mmap'd one in T11 without touching a single caller. It also
//! kills the residual dependency on HNSW: the `retrieval` crate was
//! previously the only public path to hnswlib; once every caller goes
//! through `VectorIndex`, the HNSW implementation can be deleted outright
//! (T8 Step 6).
//!
//! ## API shape
//!
//! ```rust,ignore
//! use obrain_substrate::retrieval::{VectorIndex, SubstrateTieredIndex};
//!
//! let mut index = SubstrateTieredIndex::new(384);
//! index.insert(0, &embedding_for_node_0);
//! index.insert(1, &embedding_for_node_1);
//!
//! let hits = index.search_top_k(&query_embedding, 10);
//! for (node_offset, cosine) in hits {
//!     // cosine ∈ [-1, 1], higher = closer
//! }
//! ```

use crate::tiered_scan::{ScanConfig, TieredQuery, scan_tiered};
use crate::tiers::{L2_DIM, Tier0, Tier0Builder, Tier1, Tier1Builder, Tier2, Tier2Builder};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

/// Node offset within a substrate store — an opaque u32 identity.
///
/// The index never interprets this value beyond returning it in search
/// results. Callers map it back to an `ObjectId` / `NodeId` using whatever
/// convention the consuming crate defines.
pub type NodeOffset = u32;

/// Contract for an embedding-based nearest-neighbor index.
///
/// Implementations must be thread-safe (`Send + Sync`). Search calls can
/// run concurrently with reads to the same index; insert / delete /
/// rebuild take exclusive access internally (see [`SubstrateTieredIndex`]
/// for the concrete locking scheme).
pub trait VectorIndex: Send + Sync {
    /// Return the embedding dimensionality the index was built for.
    /// All `insert` and `search_top_k` calls must pass slices of this
    /// length.
    fn dim(&self) -> usize;

    /// Current number of indexed nodes.
    fn len(&self) -> usize;

    /// `len() == 0`.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Insert or update the embedding for `node_offset`.
    ///
    /// If `node_offset` was already indexed, its tier records are
    /// overwritten in place — the underlying storage slot is kept stable.
    /// Otherwise the embedding is appended.
    fn insert(&self, node_offset: NodeOffset, embedding: &[f32]);

    /// Remove `node_offset` from the index. No-op if it was not indexed.
    fn delete(&self, node_offset: NodeOffset);

    /// Bulk-rebuild the index from `(node_offset, embedding)` pairs.
    ///
    /// Equivalent to `clear` + `insert` for each pair, but may be faster
    /// because the implementation can batch-project all three tiers in
    /// one pass. Existing contents are discarded.
    fn rebuild(&self, pairs: &[(NodeOffset, Vec<f32>)]);

    /// Search the top-K node offsets closest to `query` by cosine
    /// similarity.
    ///
    /// Returns `(node_offset, cosine)` pairs sorted by descending cosine
    /// (i.e. closest first). Ties are broken by ascending `node_offset`
    /// for determinism.
    ///
    /// If `k == 0` or the index is empty, returns an empty Vec.
    fn search_top_k(&self, query: &[f32], k: usize) -> Vec<(NodeOffset, f32)>;
}

// ---------------------------------------------------------------------------
// SubstrateTieredIndex — in-memory implementation on top of the L0/L1/L2
// cascade. Mmap'd-backed variant comes in T11 (hilbert page ordering).
// ---------------------------------------------------------------------------

/// In-memory tiered vector index.
///
/// Storage layout:
///
/// ```text
///   offset_to_slot: HashMap<NodeOffset, usize>   // sparse → dense
///   slot_to_offset: Vec<NodeOffset>              // dense → sparse
///   l0 / l1 / l2:   Vec<TierN>                   // indexed by slot
/// ```
///
/// `slot` is the position in the tier vectors; `NodeOffset` is the
/// substrate-visible identity of the indexed node. Insert appends a new
/// slot; delete swap-removes and fixes up the mapping for the node that
/// was at the last slot.
///
/// ### Concurrency
///
/// The whole state is behind a single `RwLock` for simplicity. Reads
/// (search) take a shared lock, so concurrent queries fan out. Inserts /
/// deletes / rebuilds take the exclusive lock. In the hot path the
/// caller's Engine/UserBrain already serializes mutations per-db, so
/// contention is rare.
pub struct SubstrateTieredIndex {
    inner: RwLock<Inner>,
    b0: Tier0Builder,
    b1: Tier1Builder,
    b2: Tier2Builder,
    dim: usize,
}

struct Inner {
    offset_to_slot: HashMap<NodeOffset, usize>,
    slot_to_offset: Vec<NodeOffset>,
    l0: Vec<Tier0>,
    l1: Vec<Tier1>,
    l2: Vec<Tier2>,
}

impl SubstrateTieredIndex {
    /// Create a new empty index for embeddings of dimensionality `dim`.
    ///
    /// Uses the canonical default seeds for the L0 / L1 projection
    /// matrices — two indexes built with the same `dim` project to
    /// bit-identical tiers. This matters for the substrate import
    /// pipeline: the builder seed is the substrate equivalent of an
    /// "index schema version".
    ///
    /// # Panics
    ///
    /// Panics if `dim != L2_DIM` (currently 384). The L2 tier uses a
    /// fixed-length f16 representation tied to the canonical embedding
    /// width; supporting other widths would require either dimensionality
    /// reduction (loss of recall) or a ragged L2, both of which break the
    /// `#[repr(C)]` ABI contract. Callers with a different native width
    /// should project to L2_DIM before indexing.
    pub fn new(dim: usize) -> Self {
        assert_eq!(
            dim, L2_DIM,
            "SubstrateTieredIndex currently only supports dim = L2_DIM = {L2_DIM}; got {dim}"
        );
        Self {
            inner: RwLock::new(Inner {
                offset_to_slot: HashMap::new(),
                slot_to_offset: Vec::new(),
                l0: Vec::new(),
                l1: Vec::new(),
                l2: Vec::new(),
            }),
            b0: Tier0Builder::with_default_seed(dim),
            b1: Tier1Builder::with_default_seed(dim),
            b2: Tier2Builder::new(),
            dim,
        }
    }

    /// Share the index across threads. Convenience for wiring into the
    /// trait-object-taking callers (`Arc<dyn VectorIndex>`).
    pub fn shared(self) -> Arc<Self> {
        Arc::new(self)
    }

    /// Pre-reserve capacity for at least `additional` more entries. Does
    /// nothing if the index already has room.
    pub fn reserve(&self, additional: usize) {
        let mut guard = self.inner.write();
        guard.offset_to_slot.reserve(additional);
        guard.slot_to_offset.reserve(additional);
        guard.l0.reserve(additional);
        guard.l1.reserve(additional);
        guard.l2.reserve(additional);
    }

    fn project_query(&self, query: &[f32]) -> TieredQuery {
        assert_eq!(
            query.len(),
            self.dim,
            "query.len() = {} but index dim = {}",
            query.len(),
            self.dim
        );
        TieredQuery {
            l0: self.b0.project(query),
            l1: self.b1.project(query),
            l2: self.b2.project(query),
        }
    }
}

impl VectorIndex for SubstrateTieredIndex {
    fn dim(&self) -> usize {
        self.dim
    }

    fn len(&self) -> usize {
        self.inner.read().slot_to_offset.len()
    }

    fn insert(&self, node_offset: NodeOffset, embedding: &[f32]) {
        assert_eq!(
            embedding.len(),
            self.dim,
            "embedding.len() = {} but index dim = {}",
            embedding.len(),
            self.dim
        );
        let tier0 = self.b0.project(embedding);
        let tier1 = self.b1.project(embedding);
        let tier2 = self.b2.project(embedding);

        let mut guard = self.inner.write();
        if let Some(&slot) = guard.offset_to_slot.get(&node_offset) {
            // Update in place — keep the slot stable so ongoing scans
            // (if any — but we hold the write lock, so none) see the
            // fresh fingerprint.
            guard.l0[slot] = tier0;
            guard.l1[slot] = tier1;
            guard.l2[slot] = tier2;
            return;
        }
        let slot = guard.slot_to_offset.len();
        guard.slot_to_offset.push(node_offset);
        guard.l0.push(tier0);
        guard.l1.push(tier1);
        guard.l2.push(tier2);
        guard.offset_to_slot.insert(node_offset, slot);
    }

    fn delete(&self, node_offset: NodeOffset) {
        let mut guard = self.inner.write();
        let Some(slot) = guard.offset_to_slot.remove(&node_offset) else {
            return;
        };
        let last = guard.slot_to_offset.len() - 1;
        if slot != last {
            // Swap-remove: move the last slot into `slot` and update
            // its offset → slot mapping. This keeps the tier vectors
            // dense without a shift.
            let swapped = guard.slot_to_offset[last];
            guard.slot_to_offset.swap(slot, last);
            guard.l0.swap(slot, last);
            guard.l1.swap(slot, last);
            guard.l2.swap(slot, last);
            guard.offset_to_slot.insert(swapped, slot);
        }
        guard.slot_to_offset.pop();
        guard.l0.pop();
        guard.l1.pop();
        guard.l2.pop();
    }

    fn rebuild(&self, pairs: &[(NodeOffset, Vec<f32>)]) {
        // Pre-project off the write lock — building 3 tiers per pair
        // is the slowest step and shouldn't block readers any longer
        // than necessary.
        let n = pairs.len();
        let mut offsets: Vec<NodeOffset> = Vec::with_capacity(n);
        let mut l0: Vec<Tier0> = Vec::with_capacity(n);
        let mut l1: Vec<Tier1> = Vec::with_capacity(n);
        let mut l2: Vec<Tier2> = Vec::with_capacity(n);
        let mut map: HashMap<NodeOffset, usize> = HashMap::with_capacity(n);
        for (i, (off, emb)) in pairs.iter().enumerate() {
            assert_eq!(
                emb.len(),
                self.dim,
                "pair {i}: embedding.len() = {} but index dim = {}",
                emb.len(),
                self.dim
            );
            offsets.push(*off);
            l0.push(self.b0.project(emb));
            l1.push(self.b1.project(emb));
            l2.push(self.b2.project(emb));
            map.insert(*off, i);
        }

        // Swap in the new state under the write lock. Cheap — just
        // moves of Vec headers.
        let mut guard = self.inner.write();
        guard.slot_to_offset = offsets;
        guard.l0 = l0;
        guard.l1 = l1;
        guard.l2 = l2;
        guard.offset_to_slot = map;
    }

    fn search_top_k(&self, query: &[f32], k: usize) -> Vec<(NodeOffset, f32)> {
        if k == 0 {
            return Vec::new();
        }
        let q = self.project_query(query);
        let guard = self.inner.read();
        if guard.slot_to_offset.is_empty() {
            return Vec::new();
        }
        // Clamp k to the corpus size — the cascade panics otherwise.
        let effective_k = k.min(guard.slot_to_offset.len());
        let hits = scan_tiered(
            &q,
            &guard.l0,
            &guard.l1,
            &guard.l2,
            effective_k,
            ScanConfig::default(),
        );
        // Translate slot indices back into NodeOffsets. The cascade's
        // tie-break is `slot` ascending, which aligns with NodeOffset
        // ascending *iff* insertions happened in offset order. To keep
        // the trait's contract ("ties broken by ascending NodeOffset")
        // we do a final stable sort — the working set is ≤ k ≤ 1 000,
        // so the cost is negligible.
        let mut out: Vec<(NodeOffset, f32)> = hits
            .into_iter()
            .map(|h| (guard.slot_to_offset[h.node_offset as usize], h.cosine))
            .collect();
        out.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(core::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        out
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // The L2 tier is fixed at 384 f16 (see `tiers::L2_DIM`), so the index
    // only supports that width. Use it for all tests to exercise the real
    // code path — a smaller test dim would need a separate mini-tier.
    const DIM: usize = L2_DIM;

    fn emb(seed: u32, dim: usize) -> Vec<f32> {
        // Deterministic "embedding": a rotating sine pattern keyed on seed.
        // Not a real embedding — just enough structure that distinct seeds
        // produce distinct projections with high probability.
        (0..dim)
            .map(|i| (seed as f32 * 0.3 + i as f32 * 0.15).sin())
            .collect()
    }

    #[test]
    fn empty_index_returns_empty_top_k() {
        let idx = SubstrateTieredIndex::new(DIM);
        assert!(idx.is_empty());
        assert_eq!(idx.search_top_k(&emb(0, DIM), 10), Vec::new());
    }

    #[test]
    fn insert_then_self_recall() {
        let idx = SubstrateTieredIndex::new(DIM);
        idx.insert(42, &emb(42, DIM));
        idx.insert(7, &emb(7, DIM));
        idx.insert(1234, &emb(1234, DIM));
        assert_eq!(idx.len(), 3);

        let q = emb(42, DIM);
        let hits = idx.search_top_k(&q, 3);
        assert_eq!(hits.len(), 3);
        // Self-recall: the query is an indexed embedding verbatim — must
        // come back as top-1 with cosine ≈ 1.
        assert_eq!(hits[0].0, 42);
        assert!(hits[0].1 > 0.99, "self-cosine = {}", hits[0].1);
    }

    #[test]
    fn update_in_place_rewrites_fingerprint() {
        let idx = SubstrateTieredIndex::new(DIM);
        idx.insert(42, &emb(42, DIM));
        // Overwrite with a totally different embedding.
        idx.insert(42, &emb(99, DIM));
        assert_eq!(idx.len(), 1, "duplicate insert must not append a new slot");

        // Now self-recall with emb(99) should win — emb(42) should not.
        let hits_99 = idx.search_top_k(&emb(99, DIM), 1);
        assert_eq!(hits_99[0].0, 42);
        assert!(hits_99[0].1 > 0.99, "updated embedding not visible");
    }

    #[test]
    fn delete_removes_offset() {
        let idx = SubstrateTieredIndex::new(DIM);
        for n in [10, 20, 30, 40] {
            idx.insert(n, &emb(n, DIM));
        }
        assert_eq!(idx.len(), 4);

        idx.delete(20);
        assert_eq!(idx.len(), 3);

        // The deleted offset must never surface in results, even on a
        // query crafted from its embedding.
        let hits = idx.search_top_k(&emb(20, DIM), 4);
        assert!(
            hits.iter().all(|(o, _)| *o != 20),
            "deleted offset resurfaced: {hits:?}"
        );
        // The other three must still be reachable.
        for n in [10, 30, 40] {
            let hits = idx.search_top_k(&emb(n, DIM), 1);
            assert_eq!(hits[0].0, n);
        }
    }

    #[test]
    fn delete_of_missing_offset_is_noop() {
        let idx = SubstrateTieredIndex::new(DIM);
        idx.insert(1, &emb(1, DIM));
        idx.delete(9999); // never indexed
        assert_eq!(idx.len(), 1);
        // The live offset is still reachable.
        let hits = idx.search_top_k(&emb(1, DIM), 1);
        assert_eq!(hits[0].0, 1);
    }

    #[test]
    fn rebuild_replaces_contents() {
        let idx = SubstrateTieredIndex::new(DIM);
        for n in 0..10 {
            idx.insert(n, &emb(n, DIM));
        }
        assert_eq!(idx.len(), 10);

        // Rebuild with a totally different set.
        let pairs: Vec<(NodeOffset, Vec<f32>)> =
            (100..105).map(|n| (n, emb(n, DIM))).collect();
        idx.rebuild(&pairs);
        assert_eq!(idx.len(), 5);

        // Old offsets are gone.
        for n in 0..10 {
            let hits = idx.search_top_k(&emb(n, DIM), 1);
            // The cascade may still return something (nearest in the new
            // set) but must never return a pre-rebuild offset.
            assert!(hits[0].0 >= 100);
        }
        // New offsets are reachable.
        for n in 100..105 {
            let hits = idx.search_top_k(&emb(n, DIM), 1);
            assert_eq!(hits[0].0, n);
        }
    }

    #[test]
    fn search_top_k_clamps_to_corpus() {
        let idx = SubstrateTieredIndex::new(DIM);
        for n in 0..3 {
            idx.insert(n, &emb(n, DIM));
        }
        // Asking for more than the corpus holds — cascade must clamp,
        // not panic.
        let hits = idx.search_top_k(&emb(0, DIM), 10);
        assert_eq!(hits.len(), 3);
    }

    #[test]
    fn search_top_k_zero_returns_empty() {
        let idx = SubstrateTieredIndex::new(DIM);
        idx.insert(1, &emb(1, DIM));
        assert!(idx.search_top_k(&emb(1, DIM), 0).is_empty());
    }

    #[test]
    fn trait_object_is_usable() {
        // Compile-time check: `SubstrateTieredIndex` is usable behind a
        // `dyn VectorIndex`. This mirrors how the cognitive layer holds
        // the index (`Arc<dyn VectorIndex>`).
        let idx: Arc<dyn VectorIndex> = Arc::new(SubstrateTieredIndex::new(DIM));
        idx.insert(5, &emb(5, DIM));
        let hits = idx.search_top_k(&emb(5, DIM), 1);
        assert_eq!(hits[0].0, 5);
    }

    #[test]
    fn determinism_of_ties() {
        // Craft tied embeddings: identical at every tier. They should
        // come back in ascending NodeOffset order.
        let idx = SubstrateTieredIndex::new(DIM);
        let v = emb(123, DIM);
        for n in [50, 10, 30, 20, 40] {
            idx.insert(n, &v);
        }
        let hits = idx.search_top_k(&v, 5);
        let offsets: Vec<NodeOffset> = hits.iter().map(|h| h.0).collect();
        assert_eq!(offsets, vec![10, 20, 30, 40, 50]);
    }
}
