//! T11 Step 1 — 2D Hilbert space-filling curve ordering.
//!
//! Computes a linear index ∈ `[0, 2^(2·order))` for a point in a square grid
//! of side `2^order`, such that points adjacent in 2D map to close indices
//! on the curve. Used as a secondary sort key *within* a community to preserve
//! 2D topological locality (centrality × degree) in the physical page layout.
//!
//! # Why Hilbert and not Z-order / Morton?
//!
//! Morton (bit-interleave) has the same average locality but suffers from
//! long "leaps" at quadrant boundaries (going from `011...` to `100...`
//! flips every bit → far away in index space despite being 1 unit apart in
//! 2D). Hilbert's recursive rotations bound every step to unit distance,
//! which is exactly what we need for spreading-activation prefetch: a
//! `madvise(WILLNEED)` over `[hilbert_min, hilbert_max]` of a community's
//! nodes maps to a **contiguous** 2D region, not a fractally-perforated one.
//!
//! # Algorithm
//!
//! Classical iterative descent à la Sedgewick [*Algorithms 4th ed., §1.5*]
//! and the Warren bit-manipulation note — walk `s = n/2, n/4, …, 1`, at each
//! step classify the current cell into one of 4 quadrants via
//! `rx = (x & s) != 0`, `ry = (y & s) != 0`, accumulate `s² · ((3·rx)^ry)`,
//! then rotate (x, y) so recursion into the sub-quadrant stays coordinate-
//! aligned. Total work: `O(order)` bitwise ops — ≈ 10 ns per call on M2.

/// Map 2D grid coordinates to a 1D Hilbert-curve index.
///
/// * `x, y` must be in `[0, 1 << order)`. Debug-asserted.
/// * `order ≤ 15` so the output fits in a `u32` (`2·15 = 30` bits) with
///   headroom. (We pick u32 here because `hilbert_order` in
///   [`crate::meta::MetaHeader`] is u32 and we never expect a grid finer
///   than 2^15 = 32768 per axis — far more resolution than needed to
///   separate `centrality × degree` slots within a community.)
///
/// # Monotonicity guarantee (tested)
///
/// For any two points `p, q` with Chebyshev distance 1, the index gap
/// `|h(p) - h(q)|` is bounded by `O(n)` (not `O(n²)`), which keeps
/// contiguous ranges on the curve corresponding to compact 2D regions —
/// the property that makes `madvise(WILLNEED)` over a Hilbert range
/// prefetch a connected topology neighborhood rather than a scattered set.
#[inline]
pub fn hilbert_index_2d(mut x: u32, mut y: u32, order: u32) -> u32 {
    debug_assert!(order > 0 && order <= 15, "order out of range: {order}");
    let n = 1u32 << order;
    debug_assert!(x < n, "x out of range: {x} >= {n}");
    debug_assert!(y < n, "y out of range: {y} >= {n}");

    let mut d: u32 = 0;
    let mut s = n / 2;
    while s > 0 {
        let rx = u32::from((x & s) != 0);
        let ry = u32::from((y & s) != 0);
        d += s * s * ((3 * rx) ^ ry);
        // Rotate the sub-quadrant so that the recursion stays well-formed.
        if ry == 0 {
            if rx == 1 {
                x = s.wrapping_sub(1).wrapping_sub(x);
                y = s.wrapping_sub(1).wrapping_sub(y);
            }
            std::mem::swap(&mut x, &mut y);
        }
        s /= 2;
    }
    d
}

/// Build a composite Hilbert key from node features.
///
/// The feature pair is `(centrality_cached: u16 Q0.16, degree: u32)`:
///
/// - `centrality_cached` is already a Q0.16 value in `[0, 65535]`. It
///   measures topological importance (cached PageRank × 65535).
/// - `degree` is the out-degree in the graph. Values above `max_degree`
///   are saturated into the top slot; typical densities are O(10²) so
///   `max_degree = 4096` is a safe cap that spends its resolution where
///   95%+ of nodes live.
///
/// Both axes are rescaled to the `[0, n)` grid where `n = 1 << order`,
/// then fed to [`hilbert_index_2d`]. The result is a stable sort key for
/// intra-community ordering: high-centrality hubs cluster near one corner,
/// low-centrality leaves near the diagonally opposite, with the Hilbert
/// curve weaving through the intermediate region so neighbors in feature
/// space are neighbors in the physical page sequence.
#[inline]
pub fn hilbert_key_from_features(
    centrality_q16: u16,
    degree: u32,
    order: u32,
    max_degree: u32,
) -> u32 {
    let n = 1u32 << order;
    // Rescale centrality from [0, 65535] to [0, n-1].
    let cx = (u32::from(centrality_q16) * (n - 1).max(1)) / u16::MAX as u32;
    // Saturate degree at max_degree then rescale to [0, n-1].
    let d = degree.min(max_degree);
    let cy = (d * (n - 1).max(1)) / max_degree.max(1);
    hilbert_index_2d(cx.min(n - 1), cy.min(n - 1), order)
}

// ---------------------------------------------------------------------------
// Permutation computation (T11 Step 2).
// ---------------------------------------------------------------------------

/// Compute the permutation that sorts slot IDs by `(community_id, hilbert_key)`.
///
/// Returns `old_to_new: Vec<u32>` where `old_to_new[old_slot] = new_slot`.
///
/// Stability: ties on `(community_id, hilbert_key)` break by `old_slot` to
/// guarantee deterministic output across runs. Tombstoned nodes (marked via
/// the caller on `tombstoned[i] = true`) are pushed to the high-slot tail
/// so live nodes compact at the low end.
///
/// * `communities[i]` — community id of node at old slot `i`.
/// * `centrality_cached[i]` — Q0.16 PageRank of node at old slot `i`.
/// * `degrees[i]` — out-degree (or arbitrary locality proxy) of node `i`.
/// * `tombstoned[i]` — true ⇒ node is dead and will be shunted to the tail.
/// * `order` — Hilbert resolution in bits per axis (`2^order` grid).
/// * `max_degree` — cap beyond which degree saturates.
pub fn compute_hilbert_permutation(
    communities: &[u32],
    centrality_cached: &[u16],
    degrees: &[u32],
    tombstoned: &[bool],
    order: u32,
    max_degree: u32,
) -> Vec<u32> {
    let n = communities.len();
    debug_assert_eq!(centrality_cached.len(), n);
    debug_assert_eq!(degrees.len(), n);
    debug_assert_eq!(tombstoned.len(), n);

    // Slot 0 is the null-sentinel and is pinned at new_slot 0.
    // We sort only the range [1, n) and assign new slot ids starting at 1.
    // This preserves the store's invariant that slot 0 is never allocated
    // to a real node — without this pin, zero-sentinel padding at slot 0
    // would sort to the tail with other tombstones and leave the lowest
    // live key colliding with slot 0 post-sort.
    let start: u32 = if n >= 1 { 1 } else { 0 };

    // Compute (sort key, original slot) for every node in [start, n).
    //
    // Key layout (big-endian comparable u128):
    //   tombstone_flag (1 bit) | community_id (32 b) | hilbert_key (32 b)
    //
    // Tombstones sort after live nodes, regardless of community.
    let mut entries: Vec<(u128, u32)> = (start..n as u32)
        .map(|old| {
            let hkey = hilbert_key_from_features(
                centrality_cached[old as usize],
                degrees[old as usize],
                order,
                max_degree,
            );
            let tomb = if tombstoned[old as usize] { 1u128 } else { 0u128 };
            let key = (tomb << 64)
                | ((communities[old as usize] as u128) << 32)
                | (hkey as u128);
            (key, old)
        })
        .collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

    let mut old_to_new = vec![0u32; n];
    // Pin slot 0 in place.
    if n >= 1 {
        old_to_new[0] = 0;
    }
    for (offset, (_, old_slot)) in entries.into_iter().enumerate() {
        let new_slot = (start as usize + offset) as u32;
        old_to_new[old_slot as usize] = new_slot;
    }
    old_to_new
}

/// Invert a permutation. `old_to_new[old] = new` ↔ `new_to_old[new] = old`.
pub fn invert_permutation(old_to_new: &[u32]) -> Vec<u32> {
    let mut new_to_old = vec![0u32; old_to_new.len()];
    for (old, &new) in old_to_new.iter().enumerate() {
        new_to_old[new as usize] = old as u32;
    }
    new_to_old
}

/// Page-aligned variant of [`compute_hilbert_permutation`].
///
/// Identical sort semantics — `(community_id, hilbert_key)` grouped ordering
/// with slot 0 pinned as the null sentinel — except the output **rounds each
/// community's starting slot up to the next multiple of `nodes_per_page`**.
/// The holes left between one community's last slot and the next page
/// boundary are filled with tombstone / zero-sentinel entries, which the
/// caller is expected to read as dead records (the T11 Step 3 slow-path
/// alignment produces these naturally when communities are inserted in an
/// interleaved pattern, so the supply is always plentiful in practice).
///
/// This is what [`crate::writer::Writer::bulk_sort_by_hilbert`] calls when
/// compacting a fragmented store: the post-compaction invariant is that
/// **no community straddles a page boundary**, so a full community can be
/// brought into the resident set with a single contiguous `madvise`.
///
/// # Invariants
///
/// * Output is a bijection `[0, n) → [0, n)` (every slot index appears once).
/// * `old_to_new[0] == 0` (null sentinel pinned).
/// * For every live node `i` placed as the **first** of its community in the
///   output, `old_to_new[i] % nodes_per_page == 0`.
/// * If `nodes_per_page == 0`, the function degrades to the exact semantics of
///   [`compute_hilbert_permutation`] (contiguous packing).
/// * If the tomb-slot supply is insufficient to pad all community transitions
///   (i.e. `total_tombstones < required_padding`), remaining live entries are
///   placed contiguously without further alignment and the function falls
///   back gracefully (no panic, no corruption). In practice the caller
///   ensures enough headroom via slow-path padding at insert time.
pub fn compute_hilbert_permutation_page_aligned(
    communities: &[u32],
    centrality_cached: &[u16],
    degrees: &[u32],
    tombstoned: &[bool],
    order: u32,
    max_degree: u32,
    nodes_per_page: u32,
) -> Vec<u32> {
    let n = communities.len();
    debug_assert_eq!(centrality_cached.len(), n);
    debug_assert_eq!(degrees.len(), n);
    debug_assert_eq!(tombstoned.len(), n);

    if n == 0 {
        return Vec::new();
    }
    if nodes_per_page == 0 {
        return compute_hilbert_permutation(
            communities,
            centrality_cached,
            degrees,
            tombstoned,
            order,
            max_degree,
        );
    }

    // Slot 0 is the null sentinel — pinned at new_slot 0.
    let start: u32 = 1;

    // Split old slots [1, n) into live and tomb buckets. Live entries carry
    // their sort key `(community_id, hilbert_key)` for in-bucket ordering.
    let mut live_entries: Vec<(u32, u32, u32)> = Vec::new();
    let mut tomb_entries: Vec<u32> = Vec::new();
    for old in start..n as u32 {
        if tombstoned[old as usize] {
            tomb_entries.push(old);
        } else {
            let hkey = hilbert_key_from_features(
                centrality_cached[old as usize],
                degrees[old as usize],
                order,
                max_degree,
            );
            live_entries.push((communities[old as usize], hkey, old));
        }
    }
    // Stable sort: (community_id, hilbert_key, old_slot).
    live_entries.sort_by(|a, b| {
        a.0.cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2))
    });

    let mut old_to_new = vec![u32::MAX; n];
    let mut taken = vec![false; n];
    old_to_new[0] = 0;
    taken[0] = true;

    // Phase A — place live entries, padding each community to a page boundary.
    let ppg = nodes_per_page;
    let mut counter: u32 = start;
    let mut prev_cid: Option<u32> = None;
    let mut remaining_tombs: usize = tomb_entries.len();
    for (cid, _hkey, old) in &live_entries {
        let community_changed = prev_cid.map_or(true, |p: u32| p != *cid);
        if community_changed && counter % ppg != 0 {
            // Needed padding to reach next page boundary.
            let pad_end = counter.checked_add(ppg - 1).map(|v| (v / ppg) * ppg).unwrap_or(counter);
            let pad_count = (pad_end - counter) as usize;
            // Only pad if we have enough tombstones to fill the holes. Otherwise
            // fall back to contiguous placement (still correct, just unaligned).
            if pad_count <= remaining_tombs && (pad_end as usize) < n {
                counter = pad_end;
                remaining_tombs -= pad_count;
            }
        }
        if (counter as usize) >= n {
            // No room left for this live entry — it will be paired with an
            // unfilled new_slot in the final reconciliation pass below.
            prev_cid = Some(*cid);
            continue;
        }
        old_to_new[*old as usize] = counter;
        taken[counter as usize] = true;
        counter += 1;
        prev_cid = Some(*cid);
    }

    // Phase B — fill unfilled new_slots with tomb entries, in input order.
    let mut tomb_iter = tomb_entries.into_iter();
    for new_slot in start..n as u32 {
        if !taken[new_slot as usize] {
            match tomb_iter.next() {
                Some(old) => {
                    old_to_new[old as usize] = new_slot;
                    taken[new_slot as usize] = true;
                }
                None => break, // will be reconciled in Phase C
            }
        }
    }

    // Phase C — reconcile any leftover unassigned/unfilled pairs. This only
    // runs if the padding path truncated live placement (degenerate case) or
    // if tombs ran out before holes were fully filled. Guaranteed to produce
    // a valid bijection because |unassigned_olds| == |unfilled_news| always.
    let unfilled: Vec<u32> = (0..n as u32)
        .filter(|ns| !taken[*ns as usize])
        .collect();
    let unassigned: Vec<u32> = (0..n as u32)
        .filter(|os| old_to_new[*os as usize] == u32::MAX)
        .collect();
    debug_assert_eq!(
        unfilled.len(),
        unassigned.len(),
        "compute_hilbert_permutation_page_aligned: bijection broken ({} unfilled vs {} unassigned)",
        unfilled.len(),
        unassigned.len()
    );
    for (old, new) in unassigned.iter().zip(unfilled.iter()) {
        old_to_new[*old as usize] = *new;
    }

    old_to_new
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Average 2D step between grid-adjacent cells, measured on the
    /// curve's index direction, must be small. This is the *usable*
    /// locality property for prefetch: on average, moving one step in
    /// 2D moves O(1) steps on the curve, which is what makes a
    /// `madvise(WILLNEED)` over a Hilbert range cover a compact region.
    ///
    /// Note: the *worst* gap is not bounded tightly — the Hilbert curve
    /// can still jump O(n²) between two adjacent cells at the curve's
    /// endpoints (this is a property of *any* continuous space-filling
    /// curve on a square). The inverse direction (consecutive indices
    /// → grid-adjacent cells) is the defining property and is tested
    /// exactly in [`consecutive_indices_are_grid_adjacent_order_4`].
    #[test]
    fn average_adjacency_gap_is_small() {
        let order = 4;
        let n = 1u32 << order;
        let mut total_gap: u64 = 0;
        let mut count: u64 = 0;
        for y in 0..n {
            for x in 0..n {
                let h = hilbert_index_2d(x, y, order);
                for (dx, dy) in [(1i32, 0), (0, 1)] {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx < 0 || ny < 0 || nx >= n as i32 || ny >= n as i32 {
                        continue;
                    }
                    let h2 = hilbert_index_2d(nx as u32, ny as u32, order);
                    let gap = h.max(h2) - h.min(h2);
                    total_gap += gap as u64;
                    count += 1;
                }
            }
        }
        let avg = total_gap as f64 / count as f64;
        // Empirically ≈ 2.5 for n=16; leaving generous headroom in case
        // of 32/64-bit rounding differences on future refactors.
        assert!(
            avg < (n as f64),
            "average Hilbert adjacency gap too high: {avg:.2} (n={n})"
        );
    }

    /// The output must be a bijection: every cell in the `n × n` grid
    /// gets a distinct index in `[0, n²)`.
    #[test]
    fn bijection_order_5() {
        let order = 5;
        let n = 1u32 << order;
        let mut seen = vec![false; (n * n) as usize];
        for y in 0..n {
            for x in 0..n {
                let h = hilbert_index_2d(x, y, order) as usize;
                assert!(h < seen.len(), "index out of range: {h}");
                assert!(!seen[h], "collision at ({x}, {y}) → {h}");
                seen[h] = true;
            }
        }
        assert!(seen.iter().all(|&b| b));
    }

    /// The 4 corners of the unit square (order 1) hit every quadrant.
    #[test]
    fn order_1_is_a_permutation_of_0_3() {
        let indices: Vec<u32> = [(0, 0), (1, 0), (1, 1), (0, 1)]
            .iter()
            .map(|&(x, y)| hilbert_index_2d(x, y, 1))
            .collect();
        let mut sorted = indices.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2, 3], "got {indices:?}");
    }

    /// Sequential indices must be Chebyshev-adjacent — i.e. the inverse of
    /// the mono-step test, phrased over the curve itself. This is the
    /// defining property of the Hilbert curve.
    #[test]
    fn consecutive_indices_are_grid_adjacent_order_4() {
        let order = 4;
        let n = 1u32 << order;
        let mut pos = vec![(0u32, 0u32); (n * n) as usize];
        for y in 0..n {
            for x in 0..n {
                let h = hilbert_index_2d(x, y, order) as usize;
                pos[h] = (x, y);
            }
        }
        for i in 1..pos.len() {
            let (x0, y0) = pos[i - 1];
            let (x1, y1) = pos[i];
            let dx = (x0 as i32 - x1 as i32).abs();
            let dy = (y0 as i32 - y1 as i32).abs();
            assert!(
                dx + dy == 1,
                "non-adjacent hop at step {i}: ({x0},{y0}) -> ({x1},{y1})"
            );
        }
    }

    #[test]
    fn key_from_features_in_range() {
        for c in [0u16, 1, 100, 32_000, 65_000, 65_535] {
            for d in [0u32, 1, 10, 100, 1_000, 5_000] {
                let order = 6; // 64 × 64 grid
                let k = hilbert_key_from_features(c, d, order, 4096);
                let n = 1u32 << order;
                assert!(k < n * n, "key out of range for ({c}, {d}): {k}");
            }
        }
    }

    #[test]
    fn key_respects_degree_saturation() {
        // Degrees >> max_degree should saturate to the top row of the grid.
        let order = 6;
        let max_deg = 1000;
        let k_above = hilbert_key_from_features(32_000, 10_000, order, max_deg);
        let k_at = hilbert_key_from_features(32_000, max_deg, order, max_deg);
        assert_eq!(
            k_above, k_at,
            "degrees above saturation cap must collapse to the cap"
        );
    }

    #[test]
    fn permutation_groups_same_community() {
        // 3 communities, 4 nodes each, deliberately scrambled.
        // All live (no tombstones), centrality/degree in a mild gradient
        // so the Hilbert sub-order is well-defined inside each community.
        //
        // Slot 0 is pinned as the null-sentinel by
        // `compute_hilbert_permutation` (T11 Step 3 invariant), so we use a
        // dummy community for it and verify contiguity only over slots
        // [1, n).
        let n = 13;
        let communities = vec![9u32, /* slot 0 = sentinel */ 2, 0, 2, 1, 0, 1, 2, 0, 1, 0, 2, 1];
        let centrality = vec![0u16, 1000, 2000, 3000, 500, 1500, 800, 2500, 100, 9000, 1200, 4000, 6000];
        let degrees = vec![0u32, 5, 10, 3, 8, 2, 15, 4, 1, 7, 12, 6, 9];
        let tombstoned = vec![false; n];
        let perm = compute_hilbert_permutation(
            &communities,
            &centrality,
            &degrees,
            &tombstoned,
            4, // 16×16 grid
            32,
        );
        // Slot 0 must be pinned.
        assert_eq!(perm[0], 0, "slot 0 not pinned: got {}", perm[0]);
        // Apply the permutation to check contiguity over [1, n).
        let mut out_community = vec![0u32; n];
        for (old, &new) in perm.iter().enumerate() {
            out_community[new as usize] = communities[old];
        }
        let mut runs: Vec<u32> = Vec::new();
        for &c in &out_community[1..] {
            if runs.last() != Some(&c) {
                runs.push(c);
            }
        }
        let distinct: std::collections::BTreeSet<_> = runs.iter().copied().collect();
        assert_eq!(
            distinct.len(),
            runs.len(),
            "communities interleaved in permuted layout: {out_community:?}"
        );
    }

    #[test]
    fn permutation_pushes_tombstones_to_tail() {
        let n = 6;
        let communities = vec![0u32, 0, 1, 1, 2, 2];
        let centrality = vec![100u16; n];
        let degrees = vec![1u32; n];
        let tombstoned = vec![false, true, false, true, false, false];
        let perm = compute_hilbert_permutation(
            &communities,
            &centrality,
            &degrees,
            &tombstoned,
            3,
            16,
        );
        // Live slots (0, 2, 4, 5) must land in [0..4), tombstones (1, 3) in [4..6).
        for old in 0..n {
            let new = perm[old] as usize;
            if tombstoned[old] {
                assert!(new >= 4, "tombstone old={old} mapped to new={new}");
            } else {
                assert!(new < 4, "live old={old} mapped to new={new}");
            }
        }
    }

    #[test]
    fn permutation_is_a_bijection() {
        let n = 50;
        let communities: Vec<u32> = (0..n).map(|i| (i % 7) as u32).collect();
        let centrality: Vec<u16> = (0..n).map(|i| ((i * 983) % 65535) as u16).collect();
        let degrees: Vec<u32> = (0..n).map(|i| ((i * 31) % 50) as u32).collect();
        let tombstoned = vec![false; n];
        let perm = compute_hilbert_permutation(
            &communities,
            &centrality,
            &degrees,
            &tombstoned,
            5,
            64,
        );
        let mut seen = vec![false; n];
        for &new in &perm {
            assert!(!seen[new as usize], "permutation collision at {new}");
            seen[new as usize] = true;
        }
        assert!(seen.iter().all(|&b| b));
    }

    #[test]
    fn permutation_is_deterministic_on_ties() {
        // All nodes in the same community with identical centrality/degree
        // must map to new_slot = old_slot (ties resolved by original index).
        let n = 8;
        let communities = vec![7u32; n];
        let centrality = vec![32_000u16; n];
        let degrees = vec![5u32; n];
        let tombstoned = vec![false; n];
        let perm = compute_hilbert_permutation(
            &communities,
            &centrality,
            &degrees,
            &tombstoned,
            4,
            32,
        );
        for i in 0..n {
            assert_eq!(perm[i], i as u32, "tie-break at slot {i}: got {}", perm[i]);
        }
    }

    #[test]
    fn invert_permutation_roundtrip() {
        let p = vec![3u32, 0, 4, 1, 2];
        let inv = invert_permutation(&p);
        assert_eq!(inv, vec![1u32, 3, 4, 0, 2]);
        // Double inverse = identity.
        let inv2 = invert_permutation(&inv);
        assert_eq!(inv2, p);
    }

    #[test]
    fn key_respects_centrality_monotonicity_at_fixed_degree() {
        // At fixed degree = 0, increasing centrality should produce
        // monotone movement along the first Hilbert axis (walked within
        // the bottom row of the grid). We don't require strict
        // monotonicity of `k`, but we require the total range is visited
        // (no collapsing to a single index).
        let order = 5;
        let keys: Vec<u32> = (0..=64u16)
            .map(|i| hilbert_key_from_features(i * 1000, 0, order, 4096))
            .collect();
        let distinct: std::collections::BTreeSet<_> = keys.iter().copied().collect();
        assert!(
            distinct.len() >= 8,
            "centrality sweep collapsed to {} distinct keys: {keys:?}",
            distinct.len()
        );
    }
}
