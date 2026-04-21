//! T7 Step 3 — bitset-filtered Hopfield recall kernel.
//!
//! Given a query node's 64-bit folded-Bloom engram signature, scan every
//! node's bitset in the column and compute the bit-overlap
//! `popcount(query & column[i])` as a cheap similarity proxy. This is the
//! first-tier filter of the two-tier Hopfield recall pipeline established
//! in T7 Step 2:
//!
//! 1. **Bitset scan** (this module) — O(N) over the 8-byte-per-slot column
//!    with a fused `AND + popcount` kernel. Returns top-k candidates by
//!    overlap. 10⁶ nodes in ≤ 500 µs on reference hardware.
//! 2. **Members verification** (done in [`crate::writer::Writer::hopfield_recall`])
//!    — for each candidate, read the `engram_members` side-table to
//!    resolve Bloom collisions and confirm actual co-engram-membership.
//!
//! The column layout matches [`crate::engram_bitset::EngramBitsetColumn`]:
//! one packed `u64` per node slot, indexed by node id. An all-zero bitset
//! means "the node belongs to no engram" and contributes overlap = 0 for
//! any query — these entries are naturally skipped by the top-k heap.
//!
//! ## Kernel flavours
//!
//! | target_arch | path                                                  |
//! |-------------|-------------------------------------------------------|
//! | `x86_64`    | `u64::count_ones` → POPCNT instruction (SSE4.2+)      |
//! | `aarch64`   | `u64::count_ones` → CNT + UADDLV (LLVM autovectorizes)|
//! | *other*     | scalar `u64::count_ones` (software popcount)          |
//!
//! Rust's `u64::count_ones()` compiles to POPCNT on x86_64 and to a
//! `CNT + UADDLV` sequence on aarch64; both are ~1 cycle per u64. A
//! bench-measured hand-rolled NEON kernel using `vcntq_u8` + `vaddv_u8`
//! was actually *slower* (219 µs vs 146 µs for 10⁶ lanes) because the
//! horizontal-sum intrinsic stalls the pipeline. We therefore dispatch
//! both arches to the scalar path — the compiler does a better job than
//! hand-written intrinsics here.


/// Scalar reference: for each slot `i`, write
/// `(column[i] & query).count_ones()` into `out[i]`. Panics if lengths differ.
#[inline(always)]
pub fn scan_overlap_scalar(column: &[u64], query: u64, out: &mut [u32]) {
    assert_eq!(
        column.len(),
        out.len(),
        "scan_overlap: column and out must be the same length"
    );
    if query == 0 {
        out.fill(0);
        return;
    }
    for (i, &v) in column.iter().enumerate() {
        // u64::count_ones lowers to POPCNT on x86_64 and a small CNT
        // sequence on aarch64. Both compile without intrinsic plumbing.
        out[i] = (v & query).count_ones();
    }
}

/// Overlap scan — delegates to [`scan_overlap_scalar`]. The `u64::count_ones`
/// intrinsic lowers to hardware POPCNT / CNT on both x86_64 and aarch64;
/// bench-measured hand-rolled SIMD does not improve on the compiler's
/// output here (see the module docs for the measured comparison).
///
/// Kept as a separate function to reserve the dispatch seam if a future
/// AVX-512 VPOPCNTDQ kernel proves worthwhile on server-class x86_64.
#[inline]
pub fn scan_overlap(column: &[u64], query: u64, out: &mut [u32]) {
    scan_overlap_scalar(column, query, out);
}

/// Return the top-`k` nodes by overlap score, sorted descending.
///
/// Nodes with zero overlap are skipped. If fewer than `k` nodes have
/// non-zero overlap, the result is shorter than `k`. Ties are broken by
/// ascending node index (stable under sort).
///
/// ## Algorithm
///
/// **Single-pass bucket capture** — O(N + K) — designed around the
/// memory-hierarchy reality that the column does not fit in L1d:
///
/// 1. **One streaming scan** over the column. For each entry compute
///    `popcount(v & query)`, find its overlap level `o`, and append the
///    index into `bucket[o]` — but cap each bucket at `k` entries
///    (we can never need more than `k` results from any one level).
///    `hist[o]` tracks the per-level count; once it reaches `k` the
///    capture branch becomes constant-not-taken, leaving only the
///    increment in the hot path.
/// 2. **Drain** by walking levels `64..=1` in descending order and
///    pushing the captured indices into the result until `k` are out.
///
/// **No final sort needed** — the result is sorted by construction:
/// * descending overlap, because the outer drain walks levels high → low;
/// * ascending idx within a level, because the single forward scan
///   captures indices in the order they appear, and the per-level cap
///   ensures we keep the `k` smallest indices at any tie level.
///
/// ### Why this beats the previous two-pass variants
///
/// All earlier variants (one-buffer / two-pass serial / two-pass with
/// 4-way parallel histograms) did **two reads** of the column. Pass 1
/// streamed at the L1d bandwidth floor (~143 µs / 52 GiB/s for 8 MB).
/// Pass 3 had to re-read the column out of L2 — the 8 MB column does
/// not fit in the M2 perf-core L1d (192 KB), so by the time pass 3
/// started the data had been evicted. The L2-bound second read alone
/// cost ~700 µs on Apple M2, dwarfing all popcount work.
///
/// A single forward scan keeps the column hot in L1d for exactly one
/// streaming sweep, so total runtime approaches the saturated-bandwidth
/// floor measured by `scan_overlap` alone (~143 µs).
///
/// ### Memory footprint
///
/// `buckets` allocates `65 × k × 4` bytes:
/// * k=32  →  8.1 KB — trivially L1d-resident.
/// * k=128 → 32.5 KB — fits L1d (192 KB on M2 perf cores) with room
///   for the column's working set (one cache line at a time).
///
/// Capture write hot lines: most popcounts of `v & query` cluster
/// around `popcount(query)/2` (binomial), so only a handful of bucket
/// rows take the brunt of writes — those rows stay pinned in L1d.
///
/// Measured on 10⁶ lanes (query = arbitrary non-zero u64) on Apple M2:
/// * min-heap variant (removed):           14.3 ms (k=32) / 22.4 ms (k=128)
/// * one-buffer bucket variant:            1.05 ms (k=32) / 1.06 ms (k=128)
/// * two-pass serial-histogram variant:    1.04 ms (k=32) / 1.04 ms (k=128)
/// * two-pass 4-way pass 1 only:           866 µs (k=32) / 864 µs (k=128)
/// * two-pass 4-way passes 1 + 3:          856 µs (k=32) / 862 µs (k=128)
/// * single-pass bucket capture, scalar:   567 µs (k=32) / 587 µs (k=128)
/// * single-pass + 4-way unroll (current): 505 µs (k=32) / 512 µs (k=128)
///
/// Final form is at the L1d-bandwidth floor (~143 µs scan baseline) plus
/// ~360 µs of unavoidable per-element bucket-capture work (RAW dep on
/// `hist[o]` between consecutive elements of the same level limits
/// further parallelism without splitting state across lanes).
pub fn top_k_by_overlap(column: &[u64], query: u64, k: usize) -> Vec<(u32, u32)> {
    if k == 0 || column.is_empty() || query == 0 {
        return Vec::new();
    }

    // Per-level capture state.
    //
    // `hist[level]` counts how many column entries have `popcount(v & query)
    // == level`. We never read past `min(hist[level], k)` of `buckets[level]`,
    // so the contents past that water-mark are intentionally untouched
    // (initialised to zero, but never observed).
    let mut hist = [0u32; 65];
    // Flat 65 × k buffer; `buckets[level * k + j]` is the j-th captured
    // index at that overlap level. Using a flat Vec rather than a 2D
    // array avoids stack pressure when k is large and keeps the hot
    // rows in a contiguous L1d-friendly layout.
    let mut buckets: Vec<u32> = vec![0u32; 65 * k];

    // Pre-compute `level * k` once per level to drop the multiply from
    // the inner-loop indexing. 65 entries, fits a single L1d line.
    let mut level_base = [0usize; 65];
    for (lv, slot) in level_base.iter_mut().enumerate() {
        *slot = lv * k;
    }

    // Streaming scan, 4-way unrolled. The unroll lets four independent
    // popcounts retire in parallel on the M2 popcount pipeline; the
    // four bucket captures that follow share the same critical path
    // shape but execute serially within a chunk (they may hit the same
    // level, in which case the `hist[o]` load-modify-store creates a
    // RAW dep — but only when popcounts collide, ~25% per pair on
    // random u64s).
    //
    // Per-element critical path (steady state):
    //   load v               ; HW-prefetched stream, ~0 cycle marginal
    //   v & query, popcount  ; 1 cycle issue, 3-cycle latency on M2
    //   load hist[o]         ; 3-4 cycle L1d
    //   compare + branch     ; predictor pinned not-taken after warmup
    //   store buckets[base+c]; 1 cycle issue, hidden in L1d write buffer
    //   store hist[o]+1      ; 1 cycle issue, hidden
    //
    // With 4 popcounts in flight per chunk the popcount latency hides;
    // the only true serial cost is the per-element hist load/store.
    let chunks = column.chunks_exact(4);
    let rem = chunks.remainder();
    let rem_start = column.len() - rem.len();
    let mut idx: u32 = 0;

    for ch in chunks {
        // Four parallel popcounts.
        let oa = (ch[0] & query).count_ones() as usize;
        let ob = (ch[1] & query).count_ones() as usize;
        let oc = (ch[2] & query).count_ones() as usize;
        let od = (ch[3] & query).count_ones() as usize;

        // Four serial captures. Each step:
        //   - load hist[o]
        //   - if < k: store buckets[level_base[o] + count] <- idx
        //   - store hist[o] + 1
        // Within a chunk the four (o,count) pairs may collide, in which
        // case the second store sees the new value of hist[o] thanks to
        // the immediate write-back below. Each branch is statically
        // biased not-taken once the bucket fills.
        let ca = hist[oa] as usize;
        if ca < k {
            buckets[level_base[oa] + ca] = idx;
        }
        hist[oa] = ca as u32 + 1;

        let cb = hist[ob] as usize;
        if cb < k {
            buckets[level_base[ob] + cb] = idx + 1;
        }
        hist[ob] = cb as u32 + 1;

        let cc = hist[oc] as usize;
        if cc < k {
            buckets[level_base[oc] + cc] = idx + 2;
        }
        hist[oc] = cc as u32 + 1;

        let cd = hist[od] as usize;
        if cd < k {
            buckets[level_base[od] + cd] = idx + 3;
        }
        hist[od] = cd as u32 + 1;

        idx += 4;
    }
    // Tail (0..=3 entries).
    for (j, &v) in rem.iter().enumerate() {
        let i = (rem_start + j) as u32;
        let o = (v & query).count_ones() as usize;
        let count = hist[o] as usize;
        if count < k {
            buckets[level_base[o] + count] = i;
        }
        hist[o] = count as u32 + 1;
    }

    // Drain — walk levels descending, take up to `k` total.
    //
    // Within a level the captured indices are already in ascending order
    // (single forward scan), and we descend levels in order, so the
    // resulting Vec is sorted as the contract requires — no final
    // `sort_by` call needed.
    let mut result: Vec<(u32, u32)> = Vec::with_capacity(k);
    for level in (1..=64usize).rev() {
        let stored = (hist[level] as usize).min(k);
        if stored == 0 {
            continue;
        }
        let need = k - result.len();
        let take = stored.min(need);
        let base = level * k;
        for j in 0..take {
            result.push((buckets[base + j], level as u32));
        }
        if result.len() >= k {
            break;
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Tests — correctness + threshold-selection edge cases.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_reference(column: &[u64], query: u64) -> Vec<u32> {
        let mut out = vec![0u32; column.len()];
        scan_overlap_scalar(column, query, &mut out);
        out
    }

    #[test]
    fn zero_query_yields_all_zeros() {
        let col = vec![0xFFFF_FFFF_FFFF_FFFFu64; 16];
        let mut out = vec![0u32; 16];
        scan_overlap(&col, 0, &mut out);
        assert_eq!(out, vec![0u32; 16]);
    }

    #[test]
    fn full_query_counts_each_columns_ones() {
        let col = vec![0u64, 1, 0b1011, 0xFFFF_FFFF_FFFF_FFFF, 0x8000_0000_0000_0001];
        let mut out = vec![0u32; col.len()];
        scan_overlap(&col, u64::MAX, &mut out);
        assert_eq!(out, vec![0, 1, 3, 64, 2]);
    }

    #[test]
    fn scalar_simd_parity_on_handcrafted() {
        let col = vec![
            0x0000_0000_0000_0000u64,
            0xAAAA_AAAA_AAAA_AAAA,
            0x5555_5555_5555_5555,
            0xF0F0_F0F0_F0F0_F0F0,
            0x1234_5678_9ABC_DEF0,
            0xFFFF_FFFF_FFFF_FFFF,
            0x0000_0000_FFFF_FFFF,
            0x8000_0000_0000_0001,
            0x0001_0002_0004_0008,
        ];
        for &q in &[0u64, 1, u64::MAX, 0xAAAA_AAAA_AAAA_AAAAu64, 0x5555_5555_5555_5555, 0xFEED_FACE_CAFE_BABE] {
            let expected = scalar_reference(&col, q);
            let mut got = vec![0u32; col.len()];
            scan_overlap(&col, q, &mut got);
            assert_eq!(got, expected, "parity fail at query=0x{q:016X}");
        }
    }

    #[test]
    fn scalar_simd_parity_randomized() {
        let mut state: u64 = 0xC0FFEE_u64 ^ 0xDEAD_BEEF_A5A5_5A5A;
        let mut next = || -> u64 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for len in [0usize, 1, 2, 3, 7, 8, 16, 1023, 1024, 1025, 4096] {
            let col: Vec<u64> = (0..len).map(|_| next()).collect();
            let query = next();
            let expected = scalar_reference(&col, query);
            let mut got = vec![0u32; len];
            scan_overlap(&col, query, &mut got);
            assert_eq!(got, expected, "parity fail at len={len}, query=0x{query:016X}");
        }
    }

    #[test]
    fn top_k_returns_descending_overlap() {
        let col = vec![
            0x00u64,                 // idx 0 — overlap 0, filtered
            0x01,                    // idx 1 — overlap 1
            0x03,                    // idx 2 — overlap 2
            0x07,                    // idx 3 — overlap 3
            0x0F,                    // idx 4 — overlap 4
            0x1F,                    // idx 5 — overlap 5
        ];
        let query = u64::MAX;
        let got = top_k_by_overlap(&col, query, 3);
        assert_eq!(got, vec![(5, 5), (4, 4), (3, 3)]);
    }

    #[test]
    fn top_k_with_k_greater_than_nonzero_returns_all_hits() {
        let col = vec![0u64, 1, 0, 3, 0];
        let got = top_k_by_overlap(&col, u64::MAX, 10);
        // Only indices 1 and 3 have non-zero overlap.
        assert_eq!(got, vec![(3, 2), (1, 1)]);
    }

    #[test]
    fn top_k_breaks_ties_by_ascending_idx() {
        let col = vec![0b11u64; 5];
        let got = top_k_by_overlap(&col, u64::MAX, 3);
        // All have overlap=2 — expect first 3 indices.
        assert_eq!(got, vec![(0, 2), (1, 2), (2, 2)]);
    }

    #[test]
    fn top_k_k_zero_yields_empty() {
        let col = vec![0xFFu64; 4];
        assert!(top_k_by_overlap(&col, u64::MAX, 0).is_empty());
    }

    #[test]
    fn top_k_empty_column_yields_empty() {
        let col: Vec<u64> = Vec::new();
        assert!(top_k_by_overlap(&col, u64::MAX, 5).is_empty());
    }

    #[test]
    fn top_k_zero_query_yields_empty() {
        let col = vec![0xFFu64; 10];
        assert!(top_k_by_overlap(&col, 0, 5).is_empty());
    }
}
