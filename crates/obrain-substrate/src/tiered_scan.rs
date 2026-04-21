//! Hierarchical L0 → L1 → L2 retrieval scan.
//!
//! The substrate stores three parallel "tiered" embedding columns,
//! indexed by node offset:
//!
//! - **L0**: 16 B/node, 128-bit SRP fingerprint — the funnel prefilter.
//! - **L1**: 64 B/node, 512-bit SRP fingerprint — re-rank.
//! - **L2**: 768 B/node, 384-dim f16 — exact f16 cosine, the final tier.
//!
//! A single retrieval traverses all three:
//!
//! ```text
//! query  ─┬→  L0 scan  →  top‐K0 indices  ──┐
//!         │                                  │
//!         ├→  L1 re‐rank on those K0        │
//!         │     using XOR+popcount          │
//!         │                                  │
//!         └→  L2 cosine on top K1            ├→  top‐K results
//!               (decoded from f16)           │
//! ```
//!
//! The default cascade widths are chosen to keep latency under the
//! T8 gate (single-thread top-10 over 10⁶ nodes ≤ 300 µs):
//!
//! | tier | width    | reason                                          |
//! |------|----------|-------------------------------------------------|
//! | K0   | 1 000    | Recall headroom: ~10× the final K is enough     |
//! | K1   | 100      | Re-rank budget: 100 × 64 B = 6.4 KB hot         |
//! | K    | caller   | Whatever the user wants                         |
//!
//! Multi-threaded execution kicks in via `rayon` when the corpus is
//! large enough that the parallel split-and-merge overhead is amortised
//! (default threshold = 64 K nodes).
//!
//! ## Determinism note
//!
//! The cascade is fully deterministic: stable top-K (ties broken by
//! ascending node-offset). Parallel splits do not introduce any
//! non-determinism because the per-shard top-K results are merged in
//! a deterministic `(distance, node-offset)` order.

use crate::popcount::{
    t0_hamming, t1_hamming, xor_popcount_t0_scalar, xor_popcount_t1_scalar,
};
use crate::tiers::{tier2_cosine, Tier0, Tier1, Tier2};

use rayon::prelude::*;

/// L0 Hamming distances take values in `[0, 128]` (128-bit fingerprints).
/// One extra slot for the sentinel `d == 128` all-bits-different case.
const L0_LEVELS: usize = 129;

/// Default re-rank widths for the three-tier cascade. Mirrors the docs above.
pub const DEFAULT_K0: usize = 1_000;
pub const DEFAULT_K1: usize = 100;

/// Below this corpus size, the scan stays single-threaded. Above it, the
/// L0 scan is sharded with rayon. Picked so the per-shard work is at
/// least ~64 µs on a modern core (matches the L0 throughput we measured
/// in the popcount module).
pub const PARALLEL_THRESHOLD: usize = 64 * 1024;

/// One result row from the scan: `(node_offset, cosine_similarity)`.
/// Cosine is in `[-1.0, 1.0]`; higher is closer to the query.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScanHit {
    pub node_offset: u32,
    pub cosine: f32,
}

/// A query bundle: the pre-projected query at all three tiers. Building
/// this once lets the caller reuse the same query across multiple scans
/// (e.g. spreading-activation walks issuing many neighbour queries from
/// the same anchor).
#[derive(Debug, Clone, Copy)]
pub struct TieredQuery {
    pub l0: Tier0,
    pub l1: Tier1,
    pub l2: Tier2,
}

/// Configuration for the cascade widths. Use [`ScanConfig::default`] for
/// the canonical 1 000 / 100 / k pattern.
#[derive(Debug, Clone, Copy)]
pub struct ScanConfig {
    /// Top-K kept after the L0 prefilter. Must be ≥ K1.
    pub k0: usize,
    /// Top-K kept after the L1 re-rank. Must be ≥ final K.
    pub k1: usize,
    /// Allow rayon-parallel L0 sharding above this corpus size.
    pub parallel_threshold: usize,
}

impl Default for ScanConfig {
    fn default() -> Self {
        Self {
            k0: DEFAULT_K0,
            k1: DEFAULT_K1,
            parallel_threshold: PARALLEL_THRESHOLD,
        }
    }
}

/// Run the full L0 → L1 → L2 cascade and return the top-K node hits by
/// f16 cosine similarity (descending).
///
/// All three tier slices must be parallel: `l0[i]`, `l1[i]`, `l2[i]`
/// describe node offset `i`.
///
/// Panics if the three slices have different lengths or if `k > l0.len()`.
pub fn scan_tiered(
    query: &TieredQuery,
    l0: &[Tier0],
    l1: &[Tier1],
    l2: &[Tier2],
    k: usize,
    config: ScanConfig,
) -> Vec<ScanHit> {
    let n = l0.len();
    assert_eq!(l1.len(), n, "tier shape mismatch: l1 vs l0");
    assert_eq!(l2.len(), n, "tier shape mismatch: l2 vs l0");
    assert!(k <= n, "k = {k} exceeds corpus size {n}");
    if n == 0 || k == 0 {
        return Vec::new();
    }

    // Clamp the cascade widths so they never exceed the corpus size.
    let k0 = config.k0.min(n).max(k);
    let k1 = config.k1.min(k0).max(k);

    // ── Stage L0 ─────────────────────────────────────────────────────
    let l0_top = if n >= config.parallel_threshold {
        l0_top_k_parallel(query.l0, l0, k0)
    } else {
        l0_top_k_serial(query.l0, l0, k0)
    };

    // ── Stage L1 ─────────────────────────────────────────────────────
    // Re-score the L0 candidates with the larger 512-bit fingerprint.
    // Using `t1_hamming` inline (scalar reference) is faster than
    // calling `xor_popcount_t1` on 1-element slices in a loop — the
    // dispatched path is optimised for *contiguous* corpora, not
    // gathered single-element reads from `l0_top`'s indirection list.
    // On a modern core, 1 000 × `t1_hamming` (8 × u64 popcnt each) is
    // ≈ 8 000 popcnts ≈ 8 µs — L1d-resident, negligible vs L0.
    let mut l1_scored: Vec<(u32, u32)> = Vec::with_capacity(l0_top.len());
    for &(idx, _l0_dist) in &l0_top {
        let d = t1_hamming(&l1[idx as usize], &query.l1);
        l1_scored.push((idx, d));
    }
    l1_scored.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    l1_scored.truncate(k1);

    // ── Stage L2 ─────────────────────────────────────────────────────
    // Exact f16 cosine — this is the final ranking signal.
    let mut hits: Vec<ScanHit> = l1_scored
        .iter()
        .map(|&(idx, _)| ScanHit {
            node_offset: idx,
            cosine: tier2_cosine(&query.l2, &l2[idx as usize]),
        })
        .collect();
    // Highest cosine first; deterministic tie-break by node_offset asc.
    hits.sort_by(|a, b| {
        b.cosine
            .partial_cmp(&a.cosine)
            .unwrap_or(core::cmp::Ordering::Equal)
            .then_with(|| a.node_offset.cmp(&b.node_offset))
    });
    hits.truncate(k);
    hits
}

// ---------------------------------------------------------------------------
// L0 top-K — single-pass bucket-capture kernel.
//
// Why not `xor_popcount → sort_by`?
//
// The naive "materialise a `Vec<u32>` of all distances, then sort" path is
// O(n · log n) in comparisons and touches the data *twice* (one streaming
// write to `dists`, one re-read to sort). For 10⁶ nodes, that means:
//
// * ~16 MB corpus stream read (L2-bound on Apple M2, ~1.5 ms)
// * ~4 MB `dists` stream write + re-read (another ~300 µs)
// * ~40 ms full sort (10⁶ × ~40 ns per comparison)
//
// The sort dominates, and even the "no-sort, O(n) partition" variant via
// `select_nth_unstable` still pays the second read.
//
// The engram-recall module already solved this exact shape for
// 64-bit-overlap scans (see `engram_recall::top_k_by_overlap`): one
// streaming sweep, per-distance buckets of `k` slots, a 129-entry
// histogram pinned in L1d. Final Vec is drained in ascending distance
// order — no final `sort_by` needed. On M2 that hits the L1d-bandwidth
// floor (~150 µs / 10⁶ lanes on 8 MB), which scales to ~300 µs on a
// 16 MB L0 column once the extra cache-line load is priced in.
//
// Distances are in `[0, 128]`, so `L0_LEVELS = 129`. The `65 × k × 4`
// hist+buckets footprint from engram-recall is doubled here (129 × k × 4),
// still L1d-resident for reasonable `k` (k=1000 → 516 KB — spills to L2,
// but the *hot* rows cluster around the median popcount, not the tails).
// ---------------------------------------------------------------------------

/// Single-threaded L0 top-K by ascending Hamming distance. Returns
/// `(node_offset, hamming)` pairs pre-sorted (ascending distance,
/// ascending offset on ties).
///
/// Implementation: single-pass bucket capture with scalar `count_ones`.
/// `count_ones()` compiles to NEON `cnt.16b` on aarch64 and to the
/// `popcnt` instruction on x86_64 — already 1 µop per u64, so the hot
/// loop is memory-bandwidth bound, not ALU bound. Going wider (AVX-512
/// `vpopcntq`) would only help if we batched *many* fingerprints per
/// iteration, which conflicts with the serial bucket-capture RAW chain
/// on `hist[level]`.
fn l0_top_k_serial(query: Tier0, corpus: &[Tier0], k: usize) -> Vec<(u32, u32)> {
    if k == 0 || corpus.is_empty() {
        return Vec::new();
    }
    let mut hist = [0u32; L0_LEVELS];
    let mut buckets: Vec<u32> = vec![0u32; L0_LEVELS * k];
    let mut level_base = [0usize; L0_LEVELS];
    for (lv, slot) in level_base.iter_mut().enumerate() {
        *slot = lv * k;
    }
    capture_l0_bucket(corpus, query, 0, &mut hist, &mut buckets, &level_base, k);
    drain_l0_bucket_asc(&hist, &buckets, &level_base, k)
}

/// Parallel L0 top-K. Shard the corpus across `rayon::current_num_threads()`
/// workers — one contiguous slice per thread — so each worker allocates
/// its bucket table exactly once, not once per fixed-size micro-shard.
///
/// Sharding into many small chunks (what the previous implementation did)
/// moves the hot spot from the popcount kernel onto the allocator: at
/// 10⁶ nodes × 8 K-entry shards × `129 × k × 4 B` bucket tables, we end
/// up allocating ~64 MB per query — the `malloc`/`free` roundtrips alone
/// cost more than the scan itself.
///
/// One shard per worker gives each thread a contiguous L0 range large
/// enough to saturate its L1d→L2 path, while keeping the per-query
/// allocation footprint at `num_threads × 129 × k × 4 B` (≈ 4 MB for
/// 8 threads × k = 1000). That's a single allocator hit per worker and
/// scales with cores rather than with corpus size.
fn l0_top_k_parallel(query: Tier0, corpus: &[Tier0], k: usize) -> Vec<(u32, u32)> {
    if k == 0 || corpus.is_empty() {
        return Vec::new();
    }
    let n = corpus.len();
    // One shard per rayon worker — big contiguous slices, so the bucket
    // table is reused across every element of the shard.
    let num_workers = rayon::current_num_threads().max(1);
    // Ceiling-divide so the last shard may be slightly smaller than the
    // rest (never larger — otherwise the tail grows without bound).
    let shard_size = n.div_ceil(num_workers);
    let shards: Vec<(usize, &[Tier0])> = (0..n)
        .step_by(shard_size)
        .map(|start| (start, &corpus[start..(start + shard_size).min(n)]))
        .collect();

    let local_tops: Vec<Vec<(u32, u32)>> = shards
        .par_iter()
        .map(|(start, slice)| {
            let mut hist = [0u32; L0_LEVELS];
            // Allocated once per worker — reused for the whole shard.
            let mut buckets: Vec<u32> = vec![0u32; L0_LEVELS * k];
            let mut level_base = [0usize; L0_LEVELS];
            for (lv, slot) in level_base.iter_mut().enumerate() {
                *slot = lv * k;
            }
            capture_l0_bucket(
                slice,
                query,
                *start as u32,
                &mut hist,
                &mut buckets,
                &level_base,
                k,
            );
            drain_l0_bucket_asc(&hist, &buckets, &level_base, k)
        })
        .collect();

    // Merge: each local top-K is already sorted by (distance, offset).
    // Total input = k × num_workers (e.g. 1 000 × 8 = 8 K entries) — a
    // single sort is well under 50 µs and dwarfed by the parallel scan.
    let mut merged: Vec<(u32, u32)> = local_tops.into_iter().flatten().collect();
    merged.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    merged.truncate(k);
    merged
}

/// Single-pass bucket-capture kernel. Writes into `hist` (counts per
/// distance level) and `buckets` (per-level capture arrays of up to `k`
/// node offsets, stored contiguously `levels × k`).
///
/// `offset_base` is the global offset of the first entry in `corpus` —
/// used by the parallel path so each shard can record absolute node
/// offsets without a post-merge re-indexing pass.
#[inline]
fn capture_l0_bucket(
    corpus: &[Tier0],
    query: Tier0,
    offset_base: u32,
    hist: &mut [u32; L0_LEVELS],
    buckets: &mut [u32],
    level_base: &[usize; L0_LEVELS],
    k: usize,
) {
    // 4-way unroll: four independent `count_ones` calls let the hardware
    // issue four popcounts in parallel. The bucket captures stay serial
    // within a chunk (the `hist[level] += 1` is a RAW chain that can't
    // be vectorised without split state), but the popcount latency hides
    // behind the capture work.
    let chunks = corpus.chunks_exact(4);
    let rem = chunks.remainder();
    let rem_start = corpus.len() - rem.len();
    let mut idx: u32 = offset_base;

    for ch in chunks {
        let da = t0_hamming(ch[0], query) as usize;
        let db = t0_hamming(ch[1], query) as usize;
        let dc = t0_hamming(ch[2], query) as usize;
        let dd = t0_hamming(ch[3], query) as usize;

        let ca = hist[da] as usize;
        if ca < k {
            buckets[level_base[da] + ca] = idx;
        }
        hist[da] = ca as u32 + 1;

        let cb = hist[db] as usize;
        if cb < k {
            buckets[level_base[db] + cb] = idx + 1;
        }
        hist[db] = cb as u32 + 1;

        let cc = hist[dc] as usize;
        if cc < k {
            buckets[level_base[dc] + cc] = idx + 2;
        }
        hist[dc] = cc as u32 + 1;

        let cd = hist[dd] as usize;
        if cd < k {
            buckets[level_base[dd] + cd] = idx + 3;
        }
        hist[dd] = cd as u32 + 1;

        idx += 4;
    }
    for (j, v) in rem.iter().enumerate() {
        let i = offset_base + (rem_start + j) as u32;
        let d = t0_hamming(*v, query) as usize;
        let count = hist[d] as usize;
        if count < k {
            buckets[level_base[d] + count] = i;
        }
        hist[d] = count as u32 + 1;
    }
}

/// Drain the buckets in **ascending** distance order (smallest Hamming
/// first = highest L0 similarity). The result is already sorted by
/// `(distance, offset)` because:
///
/// * within a level, captures are inserted in ascending offset order
///   (single forward scan);
/// * we walk levels ascending;
///
/// so no final `sort_by` is needed.
#[inline]
fn drain_l0_bucket_asc(
    hist: &[u32; L0_LEVELS],
    buckets: &[u32],
    level_base: &[usize; L0_LEVELS],
    k: usize,
) -> Vec<(u32, u32)> {
    let mut out: Vec<(u32, u32)> = Vec::with_capacity(k);
    for level in 0..L0_LEVELS {
        let stored = (hist[level] as usize).min(k);
        if stored == 0 {
            continue;
        }
        let need = k - out.len();
        let take = stored.min(need);
        let base = level_base[level];
        for j in 0..take {
            out.push((buckets[base + j], level as u32));
        }
        if out.len() >= k {
            break;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Reference scalar cascade — used by tests as ground truth. Same shape as
// `scan_tiered` but goes through the fully scalar paths for L0 + L1, so a
// regression in the SIMD dispatch can't accidentally agree with itself.
// ---------------------------------------------------------------------------

/// Reference scalar cascade. Tests use this to verify the SIMD-dispatched
/// `scan_tiered` produces identical results.
pub fn scan_tiered_scalar(
    query: &TieredQuery,
    l0: &[Tier0],
    l1: &[Tier1],
    l2: &[Tier2],
    k: usize,
    config: ScanConfig,
) -> Vec<ScanHit> {
    let n = l0.len();
    assert_eq!(l1.len(), n);
    assert_eq!(l2.len(), n);
    assert!(k <= n);
    if n == 0 || k == 0 {
        return Vec::new();
    }
    let k0 = config.k0.min(n).max(k);
    let k1 = config.k1.min(k0).max(k);

    // L0 scalar.
    let mut d0 = vec![0u32; n];
    xor_popcount_t0_scalar(l0, query.l0, &mut d0);
    let mut l0_top: Vec<(u32, u32)> = d0
        .into_iter()
        .enumerate()
        .map(|(i, d)| (i as u32, d))
        .collect();
    l0_top.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    l0_top.truncate(k0);

    // L1 scalar re-rank.
    let mut l1_scored: Vec<(u32, u32)> = Vec::with_capacity(l0_top.len());
    let mut buf = [0u32; 1];
    for &(idx, _) in &l0_top {
        let one = std::slice::from_ref(&l1[idx as usize]);
        xor_popcount_t1_scalar(one, &query.l1, &mut buf);
        l1_scored.push((idx, buf[0]));
    }
    l1_scored.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    l1_scored.truncate(k1);

    // L2 cosine.
    let mut hits: Vec<ScanHit> = l1_scored
        .iter()
        .map(|&(idx, _)| ScanHit {
            node_offset: idx,
            cosine: tier2_cosine(&query.l2, &l2[idx as usize]),
        })
        .collect();
    hits.sort_by(|a, b| {
        b.cosine
            .partial_cmp(&a.cosine)
            .unwrap_or(core::cmp::Ordering::Equal)
            .then_with(|| a.node_offset.cmp(&b.node_offset))
    });
    hits.truncate(k);
    hits
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tiers::{Tier0Builder, Tier1Builder, Tier2Builder};

    /// Deterministic xorshift PRNG (same style as popcount tests).
    struct Xs64(u64);
    impl Xs64 {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn next_unit(&mut self) -> f32 {
            // Map u64 → centered f32 in [-1, 1).
            let s = self.next() as i64 as f64;
            (s / (i64::MAX as f64)) as f32
        }
    }

    /// Build a synthetic dataset of `n` 384-dim embeddings, projected to
    /// all three tiers using the canonical default seeds. Returns the
    /// triple `(L0, L1, L2)` parallel to a `n × 384` f32 matrix.
    fn build_synthetic(n: usize, seed: u64) -> (Vec<f32>, Vec<Tier0>, Vec<Tier1>, Vec<Tier2>) {
        let dim = 384;
        let mut prng = Xs64(seed);
        let raw: Vec<f32> = (0..n * dim).map(|_| prng.next_unit()).collect();
        let b0 = Tier0Builder::with_default_seed(dim);
        let b1 = Tier1Builder::with_default_seed(dim);
        let b2 = Tier2Builder::new();
        let l0 = b0.project_batch(&raw);
        let l1 = b1.project_batch(&raw);
        let l2 = b2.project_batch(&raw);
        (raw, l0, l1, l2)
    }

    #[test]
    fn empty_corpus_yields_empty_top_k() {
        let q = TieredQuery {
            l0: Tier0([0, 0]),
            l1: Tier1([0u64; 8]),
            l2: Tier2([0u16; 384]),
        };
        let hits = scan_tiered(&q, &[], &[], &[], 0, ScanConfig::default());
        assert!(hits.is_empty());
    }

    #[test]
    fn dispatched_matches_scalar_small() {
        // Below the parallel threshold, both paths run serially but go
        // through the dispatched SIMD popcount vs the explicit scalar.
        let n = 256;
        let (raw, l0, l1, l2) = build_synthetic(n, 0xDEAD);
        let dim = 384;

        // Pick a query that's a real corpus vector — guarantees a clean
        // top-1.
        let q_idx = 42;
        let q_raw = &raw[q_idx * dim..(q_idx + 1) * dim];
        let q = TieredQuery {
            l0: l0[q_idx],
            l1: l1[q_idx],
            l2: l2[q_idx],
        };
        let _ = q_raw;

        let cfg = ScanConfig::default();
        let dispatched = scan_tiered(&q, &l0, &l1, &l2, 10, cfg);
        let scalar = scan_tiered_scalar(&q, &l0, &l1, &l2, 10, cfg);
        assert_eq!(dispatched, scalar, "dispatched cascade != scalar");
        // Top-1 must be the query node itself.
        assert_eq!(dispatched[0].node_offset, q_idx as u32);
        assert!(dispatched[0].cosine > 0.99, "self-cosine = {}", dispatched[0].cosine);
    }

    #[test]
    fn dispatched_matches_scalar_above_parallel_threshold() {
        // Force the parallel L0 path. We use a small parallel_threshold
        // to avoid spending 64K * 16 B = 1 MB on the test corpus while
        // still exercising the rayon shards.
        let n = 4_096;
        let (_, l0, l1, l2) = build_synthetic(n, 0xCAFE);
        let q_idx = 1234;
        let q = TieredQuery {
            l0: l0[q_idx],
            l1: l1[q_idx],
            l2: l2[q_idx],
        };
        let cfg = ScanConfig {
            parallel_threshold: 256,
            ..ScanConfig::default()
        };
        let dispatched = scan_tiered(&q, &l0, &l1, &l2, 5, cfg);
        let scalar = scan_tiered_scalar(&q, &l0, &l1, &l2, 5, cfg);
        assert_eq!(
            dispatched, scalar,
            "parallel cascade != scalar (rayon path)"
        );
        // Self-recall: query is in corpus, must surface as top-1.
        assert_eq!(dispatched[0].node_offset, q_idx as u32);
    }

    #[test]
    fn cascade_recovers_self_for_every_node() {
        // Full self-recall sweep: every node, projected to its own L0/L1/L2
        // bundle, must come back as top-1 of the cascade. This pins the
        // round-trip invariant: the builder + the scanner agree on what
        // "this is the same vector" looks like at every tier.
        let n = 64;
        let (_, l0, l1, l2) = build_synthetic(n, 0xBEEF);
        for i in 0..n {
            let q = TieredQuery {
                l0: l0[i],
                l1: l1[i],
                l2: l2[i],
            };
            let hits = scan_tiered(&q, &l0, &l1, &l2, 3, ScanConfig::default());
            assert_eq!(
                hits[0].node_offset, i as u32,
                "self-recall failed for node {i}: got top-1 = {} (cosine {})",
                hits[0].node_offset, hits[0].cosine
            );
        }
    }

    #[test]
    fn cascade_widths_clamp_correctly() {
        // If the user asks for k larger than k0 / k1, we should expand
        // the cascade widths instead of silently truncating early.
        let n = 32;
        let (_, l0, l1, l2) = build_synthetic(n, 0x1234);
        let q = TieredQuery {
            l0: l0[0],
            l1: l1[0],
            l2: l2[0],
        };
        // k = 20, but k0 / k1 default to 1000 / 100 → clamped to n = 32 / 32.
        let cfg = ScanConfig::default();
        let hits = scan_tiered(&q, &l0, &l1, &l2, 20, cfg);
        assert_eq!(hits.len(), 20);
    }

    #[test]
    fn cascade_is_deterministic() {
        // Tie-break must be by ascending node_offset. Build a corpus
        // where many vectors are tied (zero embeddings → all-ones at
        // every tier) and confirm the order is always the same.
        let n = 16;
        let (_, mut l0, mut l1, mut l2) = build_synthetic(n, 0x5555);
        // Force the first 4 nodes to have identical fingerprints by
        // overwriting them with a constant pattern.
        for i in 0..4 {
            l0[i] = Tier0([0xAAAA_AAAA_AAAA_AAAA, 0x5555_5555_5555_5555]);
            l1[i] = Tier1([1, 2, 3, 4, 5, 6, 7, 8]);
            l2[i] = Tier2([0; 384]);
        }
        let q = TieredQuery {
            l0: l0[0],
            l1: l1[0],
            l2: l2[0],
        };
        let h1 = scan_tiered(&q, &l0, &l1, &l2, 4, ScanConfig::default());
        let h2 = scan_tiered(&q, &l0, &l1, &l2, 4, ScanConfig::default());
        assert_eq!(h1, h2, "scan must be deterministic across calls");
        // The four tied vectors should appear in offset order 0, 1, 2, 3.
        // (They all hit cosine 0 because L2 is zeroed → tie-break wins.)
        let offsets: Vec<u32> = h1.iter().map(|h| h.node_offset).collect();
        // If 0..=3 are tied, they must be ordered ascending.
        let mut sorted = offsets.clone();
        sorted.sort();
        assert_eq!(offsets, sorted, "tied results must be sorted by offset");
    }
}
