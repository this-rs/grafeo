//! T12 Step 0 — Ricci-Ollivier per-edge curvature.
//!
//! # Role in topology-as-storage
//!
//! Once the Substrate owns every byte of topology (T1…T11), we can mine
//! geometric signal from it without hopping out of the mmap region.
//! Ricci-Ollivier is the first such signal: a **per-edge curvature**
//! number in `[-1, 1]` that tells you whether an edge is a *bottleneck*
//! (κ < 0 — the only bridge between two otherwise disconnected regions)
//! or a *core* edge (κ > 0 — embedded inside a locally-dense cluster).
//!
//! The downstream Thinkers and the context-loader (T12 Step 6) branch
//! on the sign of κ:
//!
//! * κ < 0 — widen context, the edge is a critical connector
//! * κ ≈ 0 — neutral, ordinary traversal
//! * κ > 0 — narrow context, the edge lives inside a tight community
//!
//! # What we compute
//!
//! The canonical definition of Ollivier-Ricci curvature on a graph is:
//!
//! ```text
//! κ(u, v) = 1 − W₁(μ_u, μ_v) / d(u, v)
//! ```
//!
//! where:
//! * `μ_x` is a probability distribution over the neighborhood of `x`
//!   (uniform over `N(x)` in the simplest variant),
//! * `W₁` is the 1-Wasserstein (earth-mover) distance between `μ_u` and
//!   `μ_v` under the graph's shortest-path metric,
//! * `d(u, v)` is the graph distance between `u` and `v` (= 1 for an
//!   existing edge).
//!
//! Computing `W₁` exactly requires solving a transportation linear
//! program on every edge — O(n³) per edge in the worst case, clearly
//! out of budget for 10⁶ edges on 8 cores in 30 s (plan's Step 1 gate).
//!
//! We use the **Jost–Liu lower bound** (Jost & Liu, "Ollivier's Ricci
//! curvature, local clustering and curvature-dimension inequalities on
//! graphs", 2011, Theorem 2 specialised to simple graphs):
//!
//! ```text
//! κ̂(u, v) = 1/|N(u)| + 1/|N(v)| − 1 + T(u,v) / min(|N(u)|, |N(v)|)
//! ```
//!
//! where `T(u, v) = |N(u) ∩ N(v) \ {u, v}|` is the number of common
//! neighbours (triangles through the edge u-v). The expression is:
//!
//! * `O(deg(u) + deg(v))` per edge — fits the 10⁶-edge budget,
//! * a provable lower bound on the true κ (never overshoots positive),
//! * tight on the trefoil test fixture (bridges → κ̂ < 0, triangle
//!   interior → κ̂ > 0, verified in the inline test).
//!
//! # Quantization
//!
//! Results are persisted in `EdgeRecord::ricci_u8` (8-bit Q, stride
//! `1/127.5`, range `[-1, 1]`). The plan originally speced `i16 Q-signed`
//! but `ricci_u8` has been in the on-disk layout since T2 with a tested
//! `set_ricci_f32` / `ricci_f32` round-trip at `±1e-2`. Widening to
//! `i16` would break the 32 B EdgeRecord layout and require a format
//! version bump. The 1/127.5 resolution is well under the `±0.01`
//! tolerance the unit tests already accept and under the signal we need
//! for bottleneck detection — we honour the existing field.
//!
//! # API
//!
//! ## Single-edge
//! * [`compute_ricci_fast`] — pure computation from a `(src, dst)`
//!   pair, no persistence.
//! * [`compute_ricci_for_edge`] — read the edge record, compute, return
//!   `f32` without writing back (useful for tests and read-only
//!   diagnostics).
//! * [`refresh_edge_ricci`] — compute and write back via
//!   `Writer::update_edge`, clearing the `RICCI_STALE` flag. WAL-logged
//!   for durability.
//!
//! ## Batch
//! * [`refresh_all_ricci`] — full-graph batch refresh (T12 Step 1).
//!   Materialises every live node's neighbour set once, parallelises
//!   the per-edge Jost-Liu compute across rayon workers, applies
//!   updates sequentially under the writer's mutex. The acceptance
//!   gate is 10⁶ edges on 8 cores in ≤ 30 s; the neighbourhood
//!   pre-build is the O(E) phase, per-edge compute is O(deg(u) +
//!   deg(v)) hash intersections on pre-materialised sets.

use std::collections::HashMap;
use std::collections::HashSet;

use obrain_common::types::{EdgeId, NodeId};
use obrain_core::graph::Direction;
use obrain_core::graph::traits::GraphStore;
use rayon::prelude::*;

use crate::error::SubstrateResult;
use crate::record::edge_flags;
use crate::store::SubstrateStore;

/// Collect the neighborhood of `node` as a deduplicated set, always
/// excluding `node` itself (no self-loops).
///
/// Uses `Direction::Both` because edges in the substrate are
/// semantically undirected (SYNAPSE / COACT / ACT edge types are
/// reinforced bidirectionally, and the downstream BFS spreading
/// activation at T11 Step 6 treats them as such).
fn neighborhood(store: &SubstrateStore, node: NodeId) -> HashSet<NodeId> {
    let mut set: HashSet<NodeId> = HashSet::new();
    for (nbr, _eid) in store.edges_from(node, Direction::Both) {
        if nbr == node {
            continue;
        }
        set.insert(nbr);
    }
    set
}

/// Jost–Liu lower bound on Ollivier-Ricci curvature for the edge
/// `(src, dst)`.
///
/// Returns a value in `[-1, 1]`. Corner cases:
///
/// * Isolated edge (both endpoints have no neighbours other than each
///   other) → `0.0` — the curvature is undefined but `0` is the
///   neutral default so a stale `RICCI_STALE` flag cleared on such an
///   edge doesn't trip downstream sign checks.
/// * Dangling src or dst (node doesn't exist) → `0.0`.
///
/// # Complexity
///
/// `O(deg(src) + deg(dst))` — one chain walk per endpoint plus a
/// hash-set intersection.
pub fn compute_ricci_fast(store: &SubstrateStore, src: NodeId, dst: NodeId) -> f32 {
    if src == dst {
        return 0.0;
    }

    // Full open neighbourhoods `N(x)` — include the peer endpoint.
    // Ollivier's μ_x puts mass on every edge leaving x, including the
    // one to the opposite endpoint, so the denominators must reflect
    // the full degree.
    let a_full = neighborhood(store, src);
    let b_full = neighborhood(store, dst);

    let n_u = a_full.len();
    let n_v = b_full.len();

    // Dangling or isolated endpoints: curvature undefined → neutral 0.
    if n_u == 0 || n_v == 0 {
        return 0.0;
    }

    // Degenerate "2-cycle" case: both endpoints' ONLY neighbour is
    // each other. The edge is a topologically isolated component;
    // Jost-Liu would return +1 (a false "tight cluster" signal)
    // because the self-bound has no triangle-budget to consume.
    // Returning neutral 0 here matches the "unknown curvature" policy
    // used for dangling nodes above and keeps downstream sign-checks
    // honest on pathological inputs.
    if n_u == 1 && n_v == 1 && a_full.contains(&dst) && b_full.contains(&src) {
        return 0.0;
    }

    // Triangle count `T(u,v) = |N(u) ∩ N(v) \ {u, v}|` — common
    // neighbours excluding the edge endpoints themselves. This is
    // where we take care to drop u and v from the intersection: u
    // can appear in N(v) (edge u-v), v can appear in N(u) — those
    // do NOT count as triangles.
    //
    // Iterate the smaller set for the `contains` probe.
    let (small, big) = if n_u <= n_v {
        (&a_full, &b_full)
    } else {
        (&b_full, &a_full)
    };
    let t = small
        .iter()
        .filter(|n| **n != src && **n != dst && big.contains(n))
        .count();

    // Jost-Liu lower bound specialised to simple graphs:
    //   κ̂ = 1/|N(u)| + 1/|N(v)| − 1 + T(u,v) / min(|N(u)|, |N(v)|)
    let inv_u = 1.0_f32 / n_u as f32;
    let inv_v = 1.0_f32 / n_v as f32;
    let t_frac = t as f32 / n_u.min(n_v) as f32;

    (inv_u + inv_v - 1.0 + t_frac).clamp(-1.0, 1.0)
}

/// Read the edge at `edge_id`, compute its curvature, and return it
/// without writing back.
///
/// Returns `Ok(None)` if the slot is beyond the edge zone's high-water
/// mark or the edge is tombstoned. The caller is responsible for
/// deciding whether to persist via [`refresh_edge_ricci`].
pub fn compute_ricci_for_edge(
    store: &SubstrateStore,
    edge_id: EdgeId,
) -> SubstrateResult<Option<f32>> {
    let rec = match store.writer().read_edge(edge_id.0)? {
        Some(rec) if !rec.is_tombstoned() => rec,
        _ => return Ok(None),
    };
    let src = NodeId(rec.src as u64);
    let dst = NodeId(rec.dst as u64);
    Ok(Some(compute_ricci_fast(store, src, dst)))
}

/// Compute the edge's curvature and persist it via
/// `Writer::update_edge`. Clears the `RICCI_STALE` flag on success.
///
/// Returns `Ok(None)` if the slot is not readable (beyond high-water
/// or tombstoned); callers should treat that as "nothing to do".
/// Returns `Ok(Some(κ))` with the computed curvature on success.
///
/// WAL-logged: a crash mid-refresh replays exactly the ricci_u8 /
/// flags pair that was in flight, no partial update possible.
pub fn refresh_edge_ricci(
    store: &SubstrateStore,
    edge_id: EdgeId,
) -> SubstrateResult<Option<f32>> {
    let Some(mut rec) = store.writer().read_edge(edge_id.0)? else {
        return Ok(None);
    };
    if rec.is_tombstoned() {
        return Ok(None);
    }
    let src = NodeId(rec.src as u64);
    let dst = NodeId(rec.dst as u64);
    let k = compute_ricci_fast(store, src, dst);
    rec.set_ricci_f32(k);
    rec.flags &= !edge_flags::RICCI_STALE;
    store.writer().update_edge(edge_id.0, rec)?;
    Ok(Some(k))
}

// ---------------------------------------------------------------------------
// Batch refresh (T12 Step 1)
// ---------------------------------------------------------------------------

/// Aggregate statistics from a batch Ricci refresh.
#[derive(Debug, Clone, Default)]
pub struct RicciRefreshStats {
    /// Edges visited (live + non-tombstoned in the scan range).
    pub edges_visited: u64,
    /// Edges whose persisted ricci_u8 actually changed value.
    pub edges_updated: u64,
    /// Edges skipped because one endpoint was dangling (deg 0).
    pub edges_skipped_dangling: u64,
    /// Maximum |κ| observed.
    pub max_abs_curvature: f32,
    /// Fraction of edges with κ < 0 (bottleneck fraction).
    pub negative_fraction: f32,
    /// Fraction of edges with κ > 0 (core fraction).
    pub positive_fraction: f32,
}

/// Full-graph batch refresh of Ricci curvature.
///
/// Phases:
/// 1. **Snapshot nodes** — walk every live slot up to `slot_high_water()`,
///    materialising `N(v)` as a `HashSet<NodeId>` for every reachable
///    node. This is a single O(V + E) read pass; each live node
///    contributes O(deg(v)) to the total via `edges_from(Both)`.
/// 2. **Snapshot edges** — collect every live, non-tombstoned
///    `EdgeRecord` with its slot index.
/// 3. **Parallel compute** — rayon splits the edge vector across
///    workers. Each worker computes Jost-Liu κ from the pre-built
///    neighbourhood sets (no substrate reads, no mutex contention).
/// 4. **Sequential apply** — walk the computed `(slot, new_record)`
///    tuples and call `update_edge` for each record whose quantised
///    `ricci_u8` changed. WAL-logged via the standard write path;
///    `RICCI_STALE` is cleared.
///
/// Returns [`RicciRefreshStats`] covering the full run.
///
/// The function is intended to be triggered:
/// * on initial substrate import (T15 migration),
/// * periodically by the Dreamer thinker,
/// * manually via a maintenance CLI entry.
pub fn refresh_all_ricci(store: &SubstrateStore) -> SubstrateResult<RicciRefreshStats> {
    // ---- Phase 1: neighbourhoods ---------------------------------------
    //
    // Only nodes that actually appear as an endpoint need neighbourhood
    // materialisation. We walk live node_ids() which is already filtered
    // by the store's liveness predicate.
    let node_ids = GraphStore::node_ids(store);
    let mut neighbourhoods: HashMap<NodeId, HashSet<NodeId>> =
        HashMap::with_capacity(node_ids.len());
    for nid in &node_ids {
        let mut set: HashSet<NodeId> = HashSet::new();
        for (nbr, _eid) in store.edges_from(*nid, Direction::Both) {
            if nbr == *nid {
                continue;
            }
            set.insert(nbr);
        }
        neighbourhoods.insert(*nid, set);
    }

    // ---- Phase 2: edge snapshot ----------------------------------------
    //
    // Walk slot 1..edge_slot_high_water() once and buffer the live
    // records with their slot index. Tombstoned slots are dropped here.
    let edge_hw = store.edge_slot_high_water();
    let mut edges: Vec<(u64, crate::record::EdgeRecord)> =
        Vec::with_capacity(edge_hw.saturating_sub(1) as usize);
    for slot in 1..edge_hw {
        if let Some(rec) = store.writer().read_edge(slot)? {
            if rec.is_tombstoned() {
                continue;
            }
            edges.push((slot, rec));
        }
    }

    // ---- Phase 3: parallel compute -------------------------------------
    //
    // For each edge, look up its endpoints' neighbourhood sets from the
    // pre-built map and apply Jost-Liu. No substrate reads — pure in-
    // memory hash intersections, perfectly parallelisable.
    let computed: Vec<(u64, crate::record::EdgeRecord, f32, bool)> = edges
        .par_iter()
        .map(|(slot, rec)| {
            let src = NodeId(rec.src as u64);
            let dst = NodeId(rec.dst as u64);
            let (k, skipped_dangling) =
                compute_ricci_from_neighbourhoods(&neighbourhoods, src, dst);
            let mut new_rec = *rec;
            new_rec.set_ricci_f32(k);
            new_rec.flags &= !edge_flags::RICCI_STALE;
            let changed = new_rec.ricci_u8 != rec.ricci_u8 || new_rec.flags != rec.flags;
            (*slot, new_rec, k, changed || skipped_dangling)
        })
        .collect();

    // ---- Phase 4: sequential apply + stats -----------------------------
    //
    // Writer::update_edge holds the zones + WAL mutexes internally.
    // Serial apply keeps the WAL record order deterministic (useful
    // for debugging crash recovery).
    let mut stats = RicciRefreshStats::default();
    let mut neg = 0u64;
    let mut pos = 0u64;
    for (slot, new_rec, k, _meta) in computed {
        stats.edges_visited += 1;

        let src = NodeId(new_rec.src as u64);
        let dst = NodeId(new_rec.dst as u64);
        let n_u = neighbourhoods.get(&src).map_or(0, |s| s.len());
        let n_v = neighbourhoods.get(&dst).map_or(0, |s| s.len());
        if n_u == 0 || n_v == 0 {
            stats.edges_skipped_dangling += 1;
        }

        // Read the current persisted bytes to decide whether to write.
        // The rayon worker already computed the would-be new record;
        // we just need the on-disk ricci_u8 to compare.
        let cur = store.writer().read_edge(slot)?;
        let Some(cur) = cur else { continue };
        if cur.ricci_u8 != new_rec.ricci_u8 || (cur.flags & edge_flags::RICCI_STALE) != 0 {
            store.writer().update_edge(slot, new_rec)?;
            stats.edges_updated += 1;
        }

        let abs_k = k.abs();
        if abs_k > stats.max_abs_curvature {
            stats.max_abs_curvature = abs_k;
        }
        if k < 0.0 {
            neg += 1;
        } else if k > 0.0 {
            pos += 1;
        }
    }

    if stats.edges_visited > 0 {
        stats.negative_fraction = neg as f32 / stats.edges_visited as f32;
        stats.positive_fraction = pos as f32 / stats.edges_visited as f32;
    }
    Ok(stats)
}

// ---------------------------------------------------------------------------
// Incremental maintenance (T12 Step 2)
// ---------------------------------------------------------------------------

/// Mark `RICCI_STALE` on every edge incident to `node` (both directions).
///
/// Call after any mutation that changes `N(node)` — edge creation
/// targeting or originating from `node`, edge removal, weight update
/// that the caller wants the curvature to follow.
///
/// This is the minimal-impact primitive: no curvature computation, no
/// allocation beyond the neighbour walk. The actual `κ` refresh is
/// deferred to the next [`refresh_stale_edges`] or
/// [`refresh_ricci_for_nodes`] call.
///
/// # Complexity
/// `O(deg(node))` — one chain walk + one `update_edge` per incident
/// edge (WAL-logged).
pub fn mark_incident_edges_stale(
    store: &SubstrateStore,
    node: NodeId,
) -> SubstrateResult<u64> {
    let mut marked = 0u64;
    for (_nbr, eid) in store.edges_from(node, Direction::Both) {
        let Some(mut rec) = store.writer().read_edge(eid.0)? else {
            continue;
        };
        if rec.is_tombstoned() {
            continue;
        }
        if rec.flags & edge_flags::RICCI_STALE != 0 {
            continue;
        }
        rec.flags |= edge_flags::RICCI_STALE;
        store.writer().update_edge(eid.0, rec)?;
        marked += 1;
    }
    Ok(marked)
}

/// Refresh every edge incident to any node in `affected_nodes`.
///
/// The primitive the delta path builds on: after a mutation that
/// changed the neighbourhood of some set of nodes (edge create / edge
/// delete / weight change), the *only* edges whose Jost-Liu curvature
/// can have drifted are those incident to at least one of those nodes
/// (proof: T(u, v) only depends on N(u) ∪ N(v); a change to N(w) only
/// affects triangle counts for edges (u, v) where w ∈ N(u) ∩ N(v) or
/// w = u or w = v — and the last two cases are exactly the incident
/// edges).
///
/// Algorithm:
/// 1. Materialise the 1-hop neighbourhood of each node in
///    `affected_nodes` (these are what the Jost-Liu formula reads
///    from plus the full neighbourhoods of each endpoint — so we
///    extend the set to the union of 1-hop and 2-hop nodes too).
/// 2. Collect the set of unique edge slots incident to any affected
///    node.
/// 3. Per edge: compute κ from the materialised map, update if the
///    quantised value changed.
///
/// Returns `RicciRefreshStats` covering the delta pass only.
pub fn refresh_ricci_for_nodes(
    store: &SubstrateStore,
    affected_nodes: &[NodeId],
) -> SubstrateResult<RicciRefreshStats> {
    // Step 1: collect the neighbourhoods of affected nodes + every
    // neighbour of those nodes (needed because Jost-Liu reads both
    // endpoints' full N(x)).
    let mut neighbourhoods: HashMap<NodeId, HashSet<NodeId>> = HashMap::new();
    let mut to_expand: Vec<NodeId> = affected_nodes.to_vec();
    // One-hop expansion: for every affected node we need its own
    // neighbours' neighbourhoods as well because the other endpoint
    // of an incident edge gets read.
    for &node in affected_nodes {
        for (nbr, _eid) in store.edges_from(node, Direction::Both) {
            if nbr == node {
                continue;
            }
            to_expand.push(nbr);
        }
    }
    to_expand.sort_by_key(|n| n.0);
    to_expand.dedup();

    for node in &to_expand {
        if neighbourhoods.contains_key(node) {
            continue;
        }
        let mut set: HashSet<NodeId> = HashSet::new();
        for (nbr, _eid) in store.edges_from(*node, Direction::Both) {
            if nbr == *node {
                continue;
            }
            set.insert(nbr);
        }
        neighbourhoods.insert(*node, set);
    }

    // Step 2: unique edge slots incident to any affected node. A slot
    // can be reached from multiple endpoints (outgoing from src +
    // incoming from dst), so dedup by slot id.
    let mut slots_to_refresh: HashSet<u64> = HashSet::new();
    for &node in affected_nodes {
        for (_nbr, eid) in store.edges_from(node, Direction::Both) {
            slots_to_refresh.insert(eid.0);
        }
    }

    // Step 3: process each slot via update_edge. No rayon for the
    // delta path — the expected workload is small (tens to hundreds
    // of edges, not millions), and serial code keeps WAL record
    // order deterministic for testing.
    let mut stats = RicciRefreshStats::default();
    let mut neg = 0u64;
    let mut pos = 0u64;
    for slot in slots_to_refresh {
        let Some(rec) = store.writer().read_edge(slot)? else {
            continue;
        };
        if rec.is_tombstoned() {
            continue;
        }
        stats.edges_visited += 1;

        let src = NodeId(rec.src as u64);
        let dst = NodeId(rec.dst as u64);
        let (k, skipped_dangling) =
            compute_ricci_from_neighbourhoods(&neighbourhoods, src, dst);
        if skipped_dangling {
            stats.edges_skipped_dangling += 1;
        }

        let mut new_rec = rec;
        new_rec.set_ricci_f32(k);
        new_rec.flags &= !edge_flags::RICCI_STALE;
        if new_rec.ricci_u8 != rec.ricci_u8 || new_rec.flags != rec.flags {
            store.writer().update_edge(slot, new_rec)?;
            stats.edges_updated += 1;
        }

        let abs_k = k.abs();
        if abs_k > stats.max_abs_curvature {
            stats.max_abs_curvature = abs_k;
        }
        if k < 0.0 {
            neg += 1;
        } else if k > 0.0 {
            pos += 1;
        }
    }
    if stats.edges_visited > 0 {
        stats.negative_fraction = neg as f32 / stats.edges_visited as f32;
        stats.positive_fraction = pos as f32 / stats.edges_visited as f32;
    }
    Ok(stats)
}

/// Walk the edge zone and refresh every edge that carries the
/// `RICCI_STALE` flag. Complements [`mark_incident_edges_stale`]:
/// callers mark at mutation time, then trigger a delta on a
/// periodic/on-demand schedule via this function.
///
/// Complexity: O(edge_high_water) for the scan, O(n_stale × deg) for
/// the recompute via [`refresh_ricci_for_nodes`].
pub fn refresh_stale_edges(store: &SubstrateStore) -> SubstrateResult<RicciRefreshStats> {
    let mut affected: HashSet<NodeId> = HashSet::new();
    let edge_hw = store.edge_slot_high_water();
    for slot in 1..edge_hw {
        let Some(rec) = store.writer().read_edge(slot)? else {
            continue;
        };
        if rec.is_tombstoned() {
            continue;
        }
        if rec.flags & edge_flags::RICCI_STALE != 0 {
            affected.insert(NodeId(rec.src as u64));
            affected.insert(NodeId(rec.dst as u64));
        }
    }
    if affected.is_empty() {
        return Ok(RicciRefreshStats::default());
    }
    let affected_vec: Vec<NodeId> = affected.into_iter().collect();
    refresh_ricci_for_nodes(store, &affected_vec)
}

/// Compute Jost-Liu κ from pre-built neighbourhood sets (no store
/// access). Returns `(κ, skipped_dangling)`.
///
/// Mirrors the logic of [`compute_ricci_fast`] but operates over
/// pre-materialised `HashMap<NodeId, HashSet<NodeId>>`. Extracted
/// for `refresh_all_ricci`'s parallel phase where pre-building the
/// map once and letting rayon workers read it concurrently beats
/// each worker acquiring the writer's zones mutex for every
/// neighbour walk.
fn compute_ricci_from_neighbourhoods(
    neighbourhoods: &HashMap<NodeId, HashSet<NodeId>>,
    src: NodeId,
    dst: NodeId,
) -> (f32, bool) {
    if src == dst {
        return (0.0, false);
    }
    let Some(a_full) = neighbourhoods.get(&src) else {
        return (0.0, true);
    };
    let Some(b_full) = neighbourhoods.get(&dst) else {
        return (0.0, true);
    };
    let n_u = a_full.len();
    let n_v = b_full.len();
    if n_u == 0 || n_v == 0 {
        return (0.0, true);
    }
    if n_u == 1 && n_v == 1 && a_full.contains(&dst) && b_full.contains(&src) {
        return (0.0, false);
    }

    let (small, big) = if n_u <= n_v {
        (a_full, b_full)
    } else {
        (b_full, a_full)
    };
    let t = small
        .iter()
        .filter(|n| **n != src && **n != dst && big.contains(n))
        .count();

    let inv_u = 1.0_f32 / n_u as f32;
    let inv_v = 1.0_f32 / n_v as f32;
    let t_frac = t as f32 / n_u.min(n_v) as f32;

    ((inv_u + inv_v - 1.0 + t_frac).clamp(-1.0, 1.0), false)
}

// ---------------------------------------------------------------------------
// Node-level curvature aggregation (T12 Step 3)
// ---------------------------------------------------------------------------

/// Per-node curvature: weight-averaged Ricci of every edge incident to
/// `node` (both directions). Read from persisted `ricci_u8` — no
/// recomputation of the edge curvature happens here, so callers MUST
/// have ensured the underlying edges are fresh (via
/// [`refresh_all_ricci`] or [`refresh_stale_edges`]).
///
/// Returns `None` if the node has zero live incident edges (dangling
/// or removed). On an isolated node `κ(v)` is undefined; returning
/// `None` lets callers branch on "unknown" instead of picking a
/// misleading default.
///
/// The weighting scheme uses the persisted synapse weight
/// (`weight_u16` → `weight_f32() ∈ [0, 1]`). When every edge has the
/// default weight 0 (freshly-created edges before any Hebbian update),
/// the function falls back to unweighted arithmetic mean.
pub fn node_curvature(store: &SubstrateStore, node: NodeId) -> Option<f32> {
    let mut sum_kw = 0.0_f64;
    let mut sum_w = 0.0_f64;
    let mut count = 0usize;
    let mut sum_k_uniform = 0.0_f64;

    for (_nbr, eid) in store.edges_from(node, Direction::Both) {
        let rec = match store.writer().read_edge(eid.0) {
            Ok(Some(r)) if !r.is_tombstoned() => r,
            _ => continue,
        };
        let k = rec.ricci_f32() as f64;
        let w = rec.weight_f32() as f64;
        sum_kw += k * w;
        sum_w += w;
        sum_k_uniform += k;
        count += 1;
    }

    if count == 0 {
        return None;
    }
    // If every incident edge has zero weight (default for freshly-
    // created edges), fall back to unweighted average.
    if sum_w <= 0.0 {
        return Some((sum_k_uniform / count as f64) as f32);
    }
    Some((sum_kw / sum_w) as f32)
}

/// Batch version: compute node curvature for every live node in the
/// substrate. Returns a `HashMap<NodeId, f32>` containing only nodes
/// with at least one live incident edge (isolated nodes are omitted,
/// matching [`node_curvature`]'s `None` return).
///
/// Useful for feeding a bottleneck-detector loop that wants to scan
/// the whole graph after a batch Ricci refresh.
pub fn compute_all_node_curvatures(
    store: &SubstrateStore,
) -> SubstrateResult<HashMap<NodeId, f32>> {
    let mut out: HashMap<NodeId, f32> = HashMap::new();
    for nid in GraphStore::node_ids(store) {
        if let Some(k) = node_curvature(store, nid) {
            out.insert(nid, k);
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Heat kernel diffusion (T12 Step 4)
// ---------------------------------------------------------------------------
//
// # The primitive
//
// Given a weighted graph with Laplacian `L = D - W` (diagonal `D` =
// weighted degree, off-diagonal `W` = edge weights), the heat kernel
// `exp(-t · L)` evolves an activation field `x(t) = exp(-t·L) · x(0)`.
//
// We do NOT materialise the matrix exponential — at 10⁶ nodes that's
// a 10¹² dense matrix. The primitive we expose is a single explicit
// Euler step of the heat equation:
//
//   x_{k+1} = x_k − dt · L · x_k
//
// which is the standard finite-difference discretisation. Repeated
// application with `dt` satisfying the CFL-style stability bound
// `dt < 2 / max_i d_w(i)` converges to the true heat flow as
// `dt → 0`. For an unweighted graph with max degree `Δ`, the
// conservative bound is `dt ≤ 1/Δ`.
//
// # Why explicit Euler (not Krylov / Lanczos)
//
// * Zero setup cost — no Arnoldi iteration needed.
// * One step = one sparse mat-vec = one substrate pass. Straight
//   memory-bandwidth bound, parallel in rayon, no cross-step state.
// * The callers (Predictor / Dreamer Thinkers) want SHORT-time
//   diffusion (1-5 steps of small dt) to spread activation to
//   neighbours-of-neighbours, not long-time ground-state
//   convergence. Explicit Euler dominates on 1-5 step workloads
//   since the competing Krylov methods amortise setup across
//   many steps.
//
// # State representation
//
// The caller owns an `&mut [f32]` buffer indexed by node slot. Slot
// 0 is the zero-slot sentinel (unused); slot 1..high_water is the
// live node range. The buffer must have `>= slot_high_water()`
// entries. Tombstoned slots are treated as degree-0 (state
// propagates to/from them without error — they just don't have
// outgoing edges).
//
// # Weighted vs unweighted
//
// The Laplacian uses `EdgeRecord::weight_f32()` (Q0.16, `[0, 1]`).
// Freshly-created edges default to weight 0 — a degenerate case
// where the Laplacian is identically zero and diffusion is a no-op.
// The [`heat_step_unweighted`] variant treats every live edge as
// weight = 1, useful for tests on fresh graphs and for pure-topology
// Gaussian-variance checks.
//
// # Why `Direction::Outgoing`
//
// The substrate builds edges bidirectionally at the API level (test
// helpers link both ways, and the SYNAPSE / COACT / ACT types are
// reinforced in both directions). Walking only outgoing edges
// therefore visits the full undirected neighbourhood once. Walking
// `Direction::Both` would double-count and halve the effective
// `dt`.

/// Compact row-major adjacency built once from the substrate's edge
/// chains, then reused across many heat / diffusion / activation
/// steps.
///
/// ## Why we need this
///
/// The substrate's per-edge `read_edge` goes through a single
/// `Mutex<ZoneMaps>` (see `writer::with_edge_zone`). Every parallel
/// worker reading edges therefore serialises on that mutex, collapsing
/// rayon's 8-core speedup to a 1-core effective bandwidth. At 10⁶
/// nodes × avg-degree 8, the observed wall-time was ~160 ms/step —
/// 30× over the plan's 5 ms gate — entirely because of mutex
/// contention, not compute.
///
/// ## Shape
///
/// CSR (Compressed Sparse Row) over node slots:
/// * `row_offsets[slot] .. row_offsets[slot+1]` is the range of
///   outgoing-edge entries in `edges` for node `slot`.
/// * `edges[i]` is a `(dst_slot, weight_u16)` pair — 6 bytes.
///
/// Row-major over slot IDs gives the same locality pattern as the
/// Hilbert-sorted node file: consecutive slots (= consecutive
/// communities after T11's layout work) land in consecutive CSR rows.
/// A rayon parallel-scan over slots sees streaming memory access.
///
/// ## Staleness
///
/// A `CsrAdjacency` is an immutable snapshot. If the underlying
/// substrate has edges added, removed, or reweighted after the
/// snapshot was built, the CSR is *stale* — step results computed
/// from a stale CSR reflect the old topology. Callers doing
/// long-running diffusions should either (a) rebuild periodically or
/// (b) stage edge mutations out-of-band and rebuild on a Dreamer
/// tick.
///
/// ## Footprint
///
/// At 10⁶ × 8 edges:
///   row_offsets:  (10⁶ + 1) × 4 B = 4 MB
///   edges:        8·10⁶ × 6 B      = 48 MB
///   TOTAL:                          ≈ 52 MB
///
/// That's 5× smaller than the 256 MB of raw EdgeRecords (32 B each)
/// — and sequential, so DRAM bandwidth applies fully.
pub struct CsrAdjacency {
    pub row_offsets: Vec<u32>,
    pub edges: Vec<(u32, u16)>,
    pub slot_high_water: u32,
}

impl CsrAdjacency {
    /// Snapshot the current outgoing-edge topology of `store`.
    ///
    /// Single-threaded: the substrate's zone mutex forces serialised
    /// reads anyway, and we want a consistent snapshot wrt any
    /// concurrent writer. The cost is `O(V + E)` edge-record reads
    /// at the current mutex-bound rate (~20 ns/edge). At 10⁶ × 8,
    /// that's ~160 ms of setup time — paid once per snapshot,
    /// amortised across however many steps the caller runs.
    pub fn build(store: &SubstrateStore) -> Self {
        let hw = store.slot_high_water();
        let mut row_offsets: Vec<u32> = Vec::with_capacity(hw as usize + 1);
        let mut edges: Vec<(u32, u16)> = Vec::new();
        // Slot 0 is the zero-slot sentinel — empty row. We iterate
        // every slot in `0..hw` so row_offsets ends up with exactly
        // `hw + 1` entries: index `s` is the START of slot `s`'s row
        // and `row_offsets[s+1]` is its END. This keeps the reader's
        // `row_offsets[slot+1]` indexing in-bounds for every slot up
        // to `hw - 1`.
        row_offsets.push(0);
        for slot in 0..hw {
            if slot != 0 {
                let u = NodeId(slot as u64);
                store.walk_outgoing_chain(u, |rec, _eid| {
                    edges.push((rec.dst, rec.weight_u16));
                });
            }
            row_offsets.push(edges.len() as u32);
        }
        debug_assert_eq!(row_offsets.len(), hw as usize + 1);
        Self {
            row_offsets,
            edges,
            slot_high_water: hw,
        }
    }

    /// Number of live directed edges in the snapshot.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }
}

/// One explicit-Euler step of the weighted heat equation on the
/// substrate's topology: `state ← state − dt · L · state`.
///
/// Builds a fresh [`CsrAdjacency`] snapshot on every call — see
/// [`heat_step_weighted_csr`] for the fast path that reuses a
/// prebuilt snapshot across K steps.
///
/// * `state[i]` is the scalar field at node slot `i`.
/// * `dt` must satisfy `dt ≤ 1 / max_weighted_degree` for
///   stability (negative-variance blow-up otherwise).
/// * The function uses a fresh output buffer internally so the
///   computation is race-free: every thread reads `state` and
///   writes into a disjoint slice of the output before the swap.
///
/// Returns `Err` if `state.len() < slot_high_water()`.
pub fn heat_step_weighted(
    store: &SubstrateStore,
    state: &mut [f32],
    dt: f32,
) -> SubstrateResult<()> {
    let csr = CsrAdjacency::build(store);
    heat_step_weighted_csr(&csr, state, dt)
}

/// Fast path: explicit-Euler heat step against a prebuilt
/// [`CsrAdjacency`] snapshot.
///
/// This is the primitive the Thinkers call — they build the CSR once
/// on Dreamer tick, then run K diffusion steps against it without
/// ever touching the substrate's zone mutex. The parallel scan sees
/// pure rayon speedup on 8 cores.
pub fn heat_step_weighted_csr(
    csr: &CsrAdjacency,
    state: &mut [f32],
    dt: f32,
) -> SubstrateResult<()> {
    heat_step_csr_generic(csr, state, dt, /* unweighted */ false)
}

/// Fast-path unweighted variant — treats every edge as weight 1.
pub fn heat_step_unweighted_csr(
    csr: &CsrAdjacency,
    state: &mut [f32],
    dt: f32,
) -> SubstrateResult<()> {
    heat_step_csr_generic(csr, state, dt, /* unweighted */ true)
}

fn heat_step_csr_generic(
    csr: &CsrAdjacency,
    state: &mut [f32],
    dt: f32,
    unweighted: bool,
) -> SubstrateResult<()> {
    let hw = csr.slot_high_water as usize;
    if state.len() < hw {
        return Err(crate::error::SubstrateError::Internal(format!(
            "heat_step_csr: state buffer too small ({} < slot_high_water {})",
            state.len(),
            hw
        )));
    }
    // Double-buffer output.
    let mut new_state: Vec<f32> = state[..hw].to_vec();

    // Parallel scan over live slots. CSR lookups are pure slice
    // indexing — no mutex, no pointer chasing, sequential memory
    // access patterns.
    new_state[1..hw]
        .par_iter_mut()
        .enumerate()
        .for_each(|(idx, out)| {
            let slot = idx + 1;
            let x_u = state[slot];
            // row_offsets has length `hw + 1` (slot 0 sentinel +
            // hw - 1 real slots + tail). row_offsets[slot] is the
            // start, [slot+1] is the end.
            let begin = csr.row_offsets[slot] as usize;
            let end = csr.row_offsets[slot + 1] as usize;
            let mut lap_u = 0.0_f32;
            for i in begin..end {
                let (dst, w_u16) = csr.edges[i];
                let w = if unweighted {
                    1.0_f32
                } else {
                    if w_u16 == 0 {
                        continue;
                    }
                    w_u16 as f32 * (1.0 / 65535.0)
                };
                let v_slot = dst as usize;
                if v_slot >= state.len() {
                    continue;
                }
                lap_u += w * (x_u - state[v_slot]);
            }
            *out = x_u - dt * lap_u;
        });

    state[1..hw].copy_from_slice(&new_state[1..hw]);
    Ok(())
}

/// Unweighted variant — every live directed edge contributes `1.0`
/// to the Laplacian. Useful on fresh graphs where `weight_f32()` is
/// uniformly 0 (before any reinforce calls).
///
/// The stability bound becomes `dt ≤ 1 / max_degree`. On a regular
/// graph of degree `k`, `dt = 1/k` is the tightest stable choice
/// (every step replaces `state[u]` with `(1 − 1) · state[u] + (1/k) ·
/// Σ state[v]` = the local mean).
///
/// Builds a fresh [`CsrAdjacency`] snapshot on every call. For
/// repeated stepping, use [`CsrAdjacency::build`] once and call
/// [`heat_step_unweighted_csr`] against the snapshot directly.
pub fn heat_step_unweighted(
    store: &SubstrateStore,
    state: &mut [f32],
    dt: f32,
) -> SubstrateResult<()> {
    let csr = CsrAdjacency::build(store);
    heat_step_unweighted_csr(&csr, state, dt)
}

/// Run `n_steps` iterations of unweighted diffusion in sequence.
/// Convenience wrapper for tests and the Gaussian-variance check.
///
/// Builds the CSR snapshot once and reuses it across every step —
/// the canonical usage pattern for repeated diffusion. If edges are
/// mutated between steps the caller must rebuild the CSR manually.
pub fn heat_diffuse_unweighted(
    store: &SubstrateStore,
    state: &mut [f32],
    dt: f32,
    n_steps: usize,
) -> SubstrateResult<()> {
    let csr = CsrAdjacency::build(store);
    for _ in 0..n_steps {
        heat_step_unweighted_csr(&csr, state, dt)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Forman-Ricci curvature (T12 Step 8) — closed-form combinatorial fast path.
// ---------------------------------------------------------------------------
//
// # Role vs Ollivier-Ricci
//
// Ollivier-Ricci (Steps 0-3) is a quality signal but its Jost-Liu
// evaluation is O(deg(u) + deg(v)) hash-set intersections per edge,
// which measured at 1840 ns/edge on 10⁶ edges. Forman-Ricci is the
// same-sign neighbourhood-bottleneck indicator computed from a closed
// formula — no set intersection beyond counting common neighbours.
//
// # Formula (combinatorial variant, no edge weights)
//
//     κ_F(u, v) = 2 − deg(u) − deg(v) + 3 · T(u, v)
//
// where T(u, v) is the number of common neighbours of u and v
// (= number of triangles through the edge).
//
// * Tree edges (no triangles): κ_F = 2 − deg(u) − deg(v). Bridge-like
//   edges connecting two hubs → strongly negative.
// * Clique edges (K_n): deg=n-1, T=n-2 → κ_F = n − 2 ≥ 0.
// * Trefoil bridge (deg=3, T=0): κ_F = −4 (sharp bottleneck).
// * Trefoil interior (deg=2, T=1): κ_F = +1 (dense core).
//
// These match the sign of Ollivier-Ricci on the existing trefoil test,
// giving the downstream bottleneck-detection pipeline (Dreamer,
// context_loader) a 10-20× faster signal for the hot path.
//
// # Quantization
//
// Persistence to `EdgeRecord` is deferred — the struct's 4-byte pad
// region will accept an `i8 forman` field once the format version
// gets bumped. In the meantime, callers use [`compute_forman_fast`]
// on demand. The bench gate (≤ 200 ms on 10⁶ edges, 8 cores) applies
// to batch compute without persistence, which is the cold-path usage.

/// Closed-form Forman-Ricci curvature for the (directed) edge `(src, dst)`.
///
/// Both endpoints are walked via `Direction::Both` to match the
/// undirected semantics already used by `compute_ricci_fast`. Returns
/// `0` if either endpoint is tombstoned or isolated.
pub fn compute_forman_fast(store: &SubstrateStore, src: NodeId, dst: NodeId) -> i32 {
    let n_src = neighborhood(store, src);
    let n_dst = neighborhood(store, dst);
    if n_src.is_empty() || n_dst.is_empty() {
        return 0;
    }
    let deg_u = n_src.len() as i32;
    let deg_v = n_dst.len() as i32;
    // Common neighbours excluding {u, v} — matches the Jost-Liu
    // triangle count in compute_ricci_fast.
    let triangles = n_src
        .iter()
        .filter(|nid| **nid != src && **nid != dst && n_dst.contains(*nid))
        .count() as i32;
    2 - deg_u - deg_v + 3 * triangles
}

/// Batch Forman-Ricci over every live edge in the substrate.
///
/// Returns a `HashMap<EdgeId, i32>` keyed by the directed edge id so
/// callers can filter / sort / persist downstream. Single-pass over
/// the edge space, per-edge compute is O(deg) hash-set lookups.
pub fn compute_all_forman(store: &SubstrateStore) -> SubstrateResult<HashMap<EdgeId, i32>> {
    // Pre-materialise neighbourhoods once to amortise the hash-set
    // construction across incident edges.
    let hw = store.slot_high_water() as usize;
    let mut nbrs: Vec<Option<HashSet<NodeId>>> = (0..hw).map(|_| None).collect();
    for nid in GraphStore::node_ids(store) {
        let slot = nid.0 as usize;
        if slot < hw {
            nbrs[slot] = Some(neighborhood(store, nid));
        }
    }

    let mut out: HashMap<EdgeId, i32> = HashMap::new();
    for nid in GraphStore::node_ids(store) {
        for (nbr, eid) in store.edges_from(nid, Direction::Outgoing) {
            let src_slot = nid.0 as usize;
            let dst_slot = nbr.0 as usize;
            let (u_set, v_set) = match (nbrs.get(src_slot), nbrs.get(dst_slot)) {
                (Some(Some(u)), Some(Some(v))) => (u, v),
                _ => continue,
            };
            let deg_u = u_set.len() as i32;
            let deg_v = v_set.len() as i32;
            let triangles = u_set
                .iter()
                .filter(|x| **x != nid && **x != nbr && v_set.contains(*x))
                .count() as i32;
            out.insert(eid, 2 - deg_u - deg_v + 3 * triangles);
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Bottleneck nodes API (T12 Step 7)
// ---------------------------------------------------------------------------

/// Return every live node whose aggregated Ricci curvature is strictly
/// below `threshold`. Input `threshold` is on the same `[-1, 1]` scale
/// as [`node_curvature`]. Results sorted ascending (most negative first)
/// so the Dreamer Thinker picks the worst bridges first.
///
/// Useful upstream of synapse-proposal logic: a node with strongly
/// negative curvature sits on a narrow community boundary and is a
/// prime candidate for a new cross-community COACT edge.
pub fn bottleneck_nodes(store: &SubstrateStore, threshold: f32) -> Vec<(NodeId, f32)> {
    let all = match compute_all_node_curvatures(store) {
        Ok(m) => m,
        Err(_) => return Vec::new(),
    };
    let mut out: Vec<(NodeId, f32)> = all
        .into_iter()
        .filter(|(_, k)| *k < threshold)
        .collect();
    out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    out
}

// ---------------------------------------------------------------------------
// Heat Kernel Signatures (T12 Step 10) — multi-scale node descriptors.
// ---------------------------------------------------------------------------
//
// # The three timescales
//
// * `t_local = 0.1`  — local: reveals immediate neighbourhood density.
// * `t_meso  = 1.0`  — meso: reveals community membership signal.
// * `t_global = 10.0` — global: reveals graph-wide role (central vs peripheral).
//
// # What it captures
//
// For each node `u`, HKS(u, t) = h(t, u, u) = diagonal entry of the
// heat kernel at time t, initialised from the Dirac δ_u. Computed via
// `k` explicit-Euler steps on the CSR snapshot, where
// `k = round(t / dt)` with `dt = 0.125` (CFL-safe on unweighted
// bounded-degree graphs).
//
// Two nodes with close HKS across all three scales are **structurally
// similar** — they sit in analogous neighbourhoods even if their
// semantic embeddings are far apart. The retrieval hybrid uses this
// as an orthogonal score to detect homonyms and promote structural
// analogues.
//
// # Bench budget
//
// 3 heat-kernel steps × O(k) per timescale, amortised over the full
// node sweep via CSR reuse. For `t_local=0.1, t_meso=1.0, t_global=10.0`
// and `dt=0.125`, total step count is ≈ 0.8 + 8 + 80 = ~89 steps, so
// budget ≈ 89 × 1.4 ms = 125 ms per full refresh on 10⁶. Well under
// the 50 ms × 3 gate when the CSR is reused (which it is — we build
// once, diffuse three times).

/// Timescales for the HKS descriptor — exposed so the Predictor / retrieval
/// can pass them back in to compare stored signatures against fresh queries.
pub const HKS_T_LOCAL: f32 = 0.1;
pub const HKS_T_MESO: f32 = 1.0;
pub const HKS_T_GLOBAL: f32 = 10.0;
pub const HKS_DT: f32 = 0.125;

/// Diagonal heat-kernel signature for every live node at `t`, returned
/// as a `Vec<f32>` indexed by node slot. Slot 0 is always 0.0
/// (sentinel). Tombstoned slots are 0.0.
///
/// Single CSR snapshot is taken at the top; caller reuses it across
/// multiple `t` values by calling [`heat_kernel_signature_with_csr`].
pub fn heat_kernel_signature(store: &SubstrateStore, t: f32) -> SubstrateResult<Vec<f32>> {
    let csr = CsrAdjacency::build(store);
    heat_kernel_signature_with_csr(&csr, t)
}

/// Number of Hutchinson probes used by [`heat_kernel_signature_with_csr`].
/// 32 is the standard sweet-spot: std dev of the estimate ≈ √(σ²/K) where
/// σ² for a diagonal-dominant heat kernel is ≤ 1, so the estimate is
/// within ±0.18 with K=32.
pub const HKS_HUTCHINSON_PROBES: usize = 32;

/// Fast path: diagonal HKS against a prebuilt CSR snapshot. Useful when
/// computing multiple `t` values from a single Dreamer tick.
///
/// Diagonal extraction via the Hutchinson stochastic estimator: for K
/// Rademacher vectors `v_k` (entries ±1 uniform), the unbiased estimator
/// is `diag(M)[u] ≈ (1/K) Σ_k v_k[u] · (M v_k)[u]`. Expectation is exact
/// because `E[v_i · v_j] = δ_{ij}`; variance shrinks as O(1/K).
///
/// We use K = [`HKS_HUTCHINSON_PROBES`]. The PRNG is seeded deterministically
/// from the graph's slot_high_water so tests are reproducible across runs.
pub fn heat_kernel_signature_with_csr(
    csr: &CsrAdjacency,
    t: f32,
) -> SubstrateResult<Vec<f32>> {
    let hw = csr.slot_high_water as usize;
    if hw == 0 {
        return Ok(Vec::new());
    }
    let n_steps = ((t / HKS_DT).ceil() as usize).max(1);

    let mut out: Vec<f32> = vec![0.0; hw];
    // Deterministic Xorshift so tests are reproducible. The specific
    // numeric choices here are not cryptographic — just spread the bits.
    let mut rng_state: u64 =
        0x9E37_79B9_7F4A_7C15 ^ (hw as u64).wrapping_mul(0x100_0193);
    let mut next = || {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        rng_state
    };

    for _ in 0..HKS_HUTCHINSON_PROBES {
        // Rademacher probe: each slot independently +1 or -1. Slot 0
        // is the sentinel (edge_id 0 = none), so we pin it to 0.
        let mut v: Vec<f32> = vec![0.0; hw];
        for i in 1..hw {
            v[i] = if (next() & 1) == 0 { 1.0 } else { -1.0 };
        }

        // Diffuse a COPY so v is preserved for the inner-product.
        let mut state = v.clone();
        for _ in 0..n_steps {
            heat_step_unweighted_csr(csr, &mut state, HKS_DT)?;
        }

        // Accumulate v[i] * state[i] per-slot — the Hutchinson
        // diagonal estimator row.
        for i in 0..hw {
            out[i] += v[i] * state[i];
        }
    }

    let inv_k = 1.0 / (HKS_HUTCHINSON_PROBES as f32);
    for x in out.iter_mut() {
        *x *= inv_k;
    }
    out[0] = 0.0;
    Ok(out)
}

/// 3-scale HKS descriptor for every node. Returns a `Vec<[f32; 3]>`
/// where entry `i` is `[h(t_local), h(t_meso), h(t_global)]` for
/// slot `i`. Slot 0 is `[0, 0, 0]` (sentinel).
///
/// One CSR build, three heat diffusions. The largest `t_global=10.0`
/// dominates the runtime (~80 steps of 1.4 ms = ~112 ms on 10⁶).
pub fn compute_hks_descriptors(
    store: &SubstrateStore,
) -> SubstrateResult<Vec<[f32; 3]>> {
    let csr = CsrAdjacency::build(store);
    let h_local = heat_kernel_signature_with_csr(&csr, HKS_T_LOCAL)?;
    let h_meso = heat_kernel_signature_with_csr(&csr, HKS_T_MESO)?;
    let h_global = heat_kernel_signature_with_csr(&csr, HKS_T_GLOBAL)?;
    let hw = csr.slot_high_water as usize;
    let mut out = vec![[0.0_f32; 3]; hw];
    for i in 1..hw {
        out[i] = [h_local[i], h_meso[i], h_global[i]];
    }
    Ok(out)
}

/// L2 distance between two HKS descriptors. Used by the retrieval
/// hybrid scorer and the tests.
#[inline]
pub fn hks_distance(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    let d0 = a[0] - b[0];
    let d1 = a[1] - b[1];
    let d2 = a[2] - b[2];
    (d0 * d0 + d1 * d1 + d2 * d2).sqrt()
}

// ---------------------------------------------------------------------------
// Geodesic distance (T12 Step 5) — Dijkstra on the topology graph.
// ---------------------------------------------------------------------------
//
// # Edge weight
//
// The graph is already weighted by synapse reinforcement (`weight_f32()
// ∈ [0, 1]`, larger = stronger). Dijkstra needs a COST, so we use
// `cost(edge) = 1.0 − weight_f32()` — a strong synapse costs 0, a
// freshly-created synapse costs 1.0. This yields the monotone property:
// semantically-tight pairs have shorter geodesic than loosely-connected
// pairs.
//
// # Implementation
//
// Binary-heap Dijkstra on the CSR snapshot. Early-exit when the target
// is popped. Complexity `O((V + E) log V)` in the worst case; amortised
// much better when the two endpoints are semantically close (the
// frontier expands only until the target is dequeued).

/// Dijkstra geodesic distance between two slots on a CSR snapshot.
///
/// Returns `None` if `dst` is unreachable from `src` or either slot is
/// out of range / sentinel. Weights are pulled from the CSR's `(dst,
/// weight_u16)` pairs and converted to cost `1.0 − weight`. A weight
/// of 0 (freshly-created edge) means cost 1 — it's traversable but
/// expensive.
pub fn geodesic_distance_csr(csr: &CsrAdjacency, src: u32, dst: u32) -> Option<f32> {
    use std::cmp::Ordering as CmpOrdering;
    use std::collections::BinaryHeap;

    let hw = csr.slot_high_water as usize;
    if (src as usize) >= hw || (dst as usize) >= hw || src == 0 || dst == 0 {
        return None;
    }
    if src == dst {
        return Some(0.0);
    }

    // BinaryHeap is a max-heap; wrap f32 in a Reverse-compatible struct.
    #[derive(Copy, Clone)]
    struct Entry(f32, u32);
    impl PartialEq for Entry {
        fn eq(&self, o: &Self) -> bool {
            self.0 == o.0
        }
    }
    impl Eq for Entry {}
    impl Ord for Entry {
        fn cmp(&self, o: &Self) -> CmpOrdering {
            // Reverse: smaller cost is "greater" so the max-heap pops it first.
            o.0.partial_cmp(&self.0).unwrap_or(CmpOrdering::Equal)
        }
    }
    impl PartialOrd for Entry {
        fn partial_cmp(&self, o: &Self) -> Option<CmpOrdering> {
            Some(self.cmp(o))
        }
    }

    let mut dist: Vec<f32> = vec![f32::INFINITY; hw];
    dist[src as usize] = 0.0;
    let mut heap: BinaryHeap<Entry> = BinaryHeap::new();
    heap.push(Entry(0.0, src));

    while let Some(Entry(d, u)) = heap.pop() {
        if u == dst {
            return Some(d);
        }
        if d > dist[u as usize] {
            continue; // stale heap entry
        }
        let begin = csr.row_offsets[u as usize] as usize;
        let end = csr.row_offsets[u as usize + 1] as usize;
        for i in begin..end {
            let (v, w_u16) = csr.edges[i];
            if (v as usize) >= hw {
                continue;
            }
            let w = w_u16 as f32 * (1.0 / 65535.0);
            let cost = 1.0 - w;
            let nd = d + cost;
            if nd < dist[v as usize] {
                dist[v as usize] = nd;
                heap.push(Entry(nd, v));
            }
        }
    }
    None
}

/// Convenience: Dijkstra geodesic directly from `SubstrateStore`. Builds
/// the CSR internally — callers doing many queries should prefer
/// [`geodesic_distance_csr`] with a reusable snapshot.
pub fn geodesic_distance(store: &SubstrateStore, src: NodeId, dst: NodeId) -> Option<f32> {
    let csr = CsrAdjacency::build(store);
    geodesic_distance_csr(&csr, src.0 as u32, dst.0 as u32)
}

// ---------------------------------------------------------------------------
// Effective resistance (T12 Step 9) — commute-time distance via CG.
// ---------------------------------------------------------------------------
//
// # Why effective resistance and not geodesic
//
// Geodesic returns the shortest path; effective resistance counts
// every path weighted by the random-walk probability. Two nodes linked
// by five moderate-weight edges have LOWER effective resistance than
// two nodes linked by a single stronger edge — exactly what the
// Dreamer needs to score bridge-candidate robustness.
//
// # Definition
//
// R(u, v) = (e_u − e_v)ᵀ · L⁺ · (e_u − e_v)
//
// where L⁺ is the pseudo-inverse of the graph Laplacian `L = D − W`.
// Equivalently, R(u,v) = x[u] − x[v] where `x` solves `L x = e_u − e_v`.
//
// # Solver
//
// Conjugate gradient on the Laplacian. `L` is symmetric positive
// semi-definite with a 1-dimensional null space (the constants). For
// `b = e_u − e_v`, `b` is orthogonal to the null space (sum = 0), so
// CG converges to the min-norm solution within the orthogonal
// complement.
//
// Convergence: each iteration is one sparse mat-vec `L·p` over the
// CSR. Bounded-degree graphs converge in `O(sqrt(κ(L)))` iterations
// where κ is the spectral condition number — empirically 30-100
// iterations for well-connected graphs of 10⁶ nodes.

fn laplacian_mv_unweighted(csr: &CsrAdjacency, x: &[f32], out: &mut [f32]) {
    let hw = csr.slot_high_water as usize;
    out.iter_mut().take(hw).for_each(|o| *o = 0.0);
    for u in 1..hw {
        let begin = csr.row_offsets[u] as usize;
        let end = csr.row_offsets[u + 1] as usize;
        let mut lap_u = 0.0_f32;
        for i in begin..end {
            let (v, _w_u16) = csr.edges[i];
            if (v as usize) >= hw {
                continue;
            }
            lap_u += x[u] - x[v as usize];
        }
        out[u] = lap_u;
    }
}

fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut s = 0.0_f32;
    for i in 0..n {
        s += a[i] * b[i];
    }
    s
}

/// Effective resistance (commute-time distance) between two slots on
/// a CSR snapshot, computed via conjugate gradient on the unweighted
/// Laplacian. `max_iter` caps the CG budget (typical 200 for 10⁶
/// graphs). `tol` is the relative residual tolerance (1e-5 default).
///
/// Returns `None` if the slots are out of range / sentinel or equal.
/// The result is in pure-number units (the Laplacian is unweighted
/// here); for a cycle `C_n` this reproduces `d(u,v) · (n − d(u,v)) / n`
/// to within the CG residual tolerance.
pub fn effective_resistance_csr(
    csr: &CsrAdjacency,
    u: u32,
    v: u32,
    max_iter: usize,
    tol: f32,
) -> Option<f32> {
    let hw = csr.slot_high_water as usize;
    if (u as usize) >= hw || (v as usize) >= hw || u == 0 || v == 0 || u == v {
        return None;
    }
    // b = e_u − e_v; this is the sum-zero RHS so CG converges on the
    // orthogonal complement of the null space.
    let mut b = vec![0.0_f32; hw];
    b[u as usize] = 1.0;
    b[v as usize] = -1.0;

    let mut x = vec![0.0_f32; hw];
    // r = b − L·x = b (since x = 0)
    let mut r = b.clone();
    let mut p = r.clone();
    let mut rs_old = dot_f32(&r, &r);
    let rs_init = rs_old;
    if rs_init <= 0.0 {
        return Some(0.0);
    }

    let mut lp = vec![0.0_f32; hw];
    for _ in 0..max_iter {
        laplacian_mv_unweighted(csr, &p, &mut lp);
        let p_dot_lp = dot_f32(&p, &lp);
        if p_dot_lp.abs() < f32::EPSILON {
            break;
        }
        let alpha = rs_old / p_dot_lp;
        for i in 0..hw {
            x[i] += alpha * p[i];
            r[i] -= alpha * lp[i];
        }
        let rs_new = dot_f32(&r, &r);
        if (rs_new / rs_init).sqrt() < tol {
            break;
        }
        let beta = rs_new / rs_old;
        for i in 0..hw {
            p[i] = r[i] + beta * p[i];
        }
        rs_old = rs_new;
    }

    Some(x[u as usize] - x[v as usize])
}

/// Convenience wrapper: build a CSR and run CG. Callers doing multiple
/// resistance queries should reuse [`effective_resistance_csr`] against
/// a single snapshot.
pub fn effective_resistance(
    store: &SubstrateStore,
    u: NodeId,
    v: NodeId,
) -> Option<f32> {
    let csr = CsrAdjacency::build(store);
    effective_resistance_csr(&csr, u.0 as u32, v.0 as u32, 200, 1e-5)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests_rng {
    /// Xorshift64 — zero-dep deterministic RNG for test mutations.
    pub(super) struct Xorshift64(u64);
    impl Xorshift64 {
        pub fn new(seed: u64) -> Self {
            Self(seed.max(1))
        }
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        pub fn range(&mut self, n: u64) -> u64 {
            self.next() % n.max(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use obrain_core::graph::traits::GraphStoreMut;
    use tempfile::TempDir;

    /// Helper: build a fresh substrate store in a tempdir.
    fn store() -> (TempDir, SubstrateStore) {
        let td = tempfile::tempdir().expect("tempdir");
        let s = SubstrateStore::create(td.path().join("ricci")).expect("create");
        (td, s)
    }

    /// Connect `a` and `b` bidirectionally so the `Direction::Both`
    /// walks see the same neighbour set from both sides.
    fn link(s: &SubstrateStore, a: NodeId, b: NodeId) {
        let _ = s.create_edge(a, b, "E");
        let _ = s.create_edge(b, a, "E");
    }

    /// Trefoil graph — three triangles `{A1,A2,A3}`, `{B1,B2,B3}`,
    /// `{C1,C2,C3}`, linked pairwise by a single bridge edge
    /// `A1-B1`, `B2-C1`, `C2-A2`.
    ///
    /// Returns (triangle_edges, bridge_edges) — the concrete
    /// `(src, dst)` pairs the test will query.
    fn build_trefoil(
        s: &SubstrateStore,
    ) -> (Vec<(NodeId, NodeId)>, Vec<(NodeId, NodeId)>) {
        let a1 = s.create_node_in_community(&["N"], 1);
        let a2 = s.create_node_in_community(&["N"], 1);
        let a3 = s.create_node_in_community(&["N"], 1);
        let b1 = s.create_node_in_community(&["N"], 2);
        let b2 = s.create_node_in_community(&["N"], 2);
        let b3 = s.create_node_in_community(&["N"], 2);
        let c1 = s.create_node_in_community(&["N"], 3);
        let c2 = s.create_node_in_community(&["N"], 3);
        let c3 = s.create_node_in_community(&["N"], 3);

        // Triangles — internal edges (dense, should be κ > 0).
        link(s, a1, a2);
        link(s, a2, a3);
        link(s, a3, a1);
        link(s, b1, b2);
        link(s, b2, b3);
        link(s, b3, b1);
        link(s, c1, c2);
        link(s, c2, c3);
        link(s, c3, c1);

        // Bridges — each pair of triangles connected by exactly one
        // edge (sparse, should be κ < 0).
        link(s, a1, b1);
        link(s, b2, c1);
        link(s, c2, a2);

        // Triangle-interior probe edges — use the one that doesn't
        // touch a bridge endpoint so the two Jost-Liu neighbourhoods
        // are the purest possible triangle pattern.
        let triangle_edges = vec![(a2, a3), (b1, b3), (c1, c3)];
        let bridge_edges = vec![(a1, b1), (b2, c1), (c2, a2)];
        (triangle_edges, bridge_edges)
    }

    #[test]
    fn trefoil_bridges_are_negative_interior_is_positive() {
        let (_td, s) = store();
        let (triangle_edges, bridge_edges) = build_trefoil(&s);

        // Every interior edge κ > 0. A2-A3 has both endpoints
        // untouched by bridges (deg=2, one common neighbour: A1),
        // so Jost-Liu gives 1/2 + 1/2 - 1 + 1/2 = 0.5.
        for (u, v) in &triangle_edges {
            let k = compute_ricci_fast(&s, *u, *v);
            assert!(
                k > 0.1,
                "triangle edge ({:?},{:?}) should have κ > 0.1, got {}",
                u,
                v,
                k
            );
        }

        // Every bridge edge κ < 0. A1-B1 has deg=3 on each side
        // (A2, A3, B1 on A1; B2, B3, A1 on B1), zero common
        // neighbours, so Jost-Liu gives 1/3 + 1/3 - 1 + 0 = -1/3.
        for (u, v) in &bridge_edges {
            let k = compute_ricci_fast(&s, *u, *v);
            assert!(
                k < -0.1,
                "bridge edge ({:?},{:?}) should have κ < -0.1, got {}",
                u,
                v,
                k
            );
        }
    }

    #[test]
    fn triangle_interior_matches_jost_liu_closed_form() {
        // Pure triangle: three nodes all mutually connected, no
        // external links. Jost-Liu: 1/2 + 1/2 - 1 + 1/2 = 0.5 for
        // every edge.
        let (_td, s) = store();
        let n0 = s.create_node_in_community(&["N"], 1);
        let n1 = s.create_node_in_community(&["N"], 1);
        let n2 = s.create_node_in_community(&["N"], 1);
        link(&s, n0, n1);
        link(&s, n1, n2);
        link(&s, n2, n0);

        let k = compute_ricci_fast(&s, n0, n1);
        assert!(
            (k - 0.5).abs() < 1e-4,
            "pure triangle κ should be 0.5, got {}",
            k
        );
    }

    #[test]
    fn isolated_edge_yields_zero() {
        // Two nodes, one edge, nothing else. After excluding the
        // peer endpoint, both neighbourhoods are empty → formula
        // short-circuits to 0.0.
        let (_td, s) = store();
        let u = s.create_node_in_community(&["N"], 1);
        let v = s.create_node_in_community(&["N"], 1);
        link(&s, u, v);

        let k = compute_ricci_fast(&s, u, v);
        assert_eq!(k, 0.0, "isolated edge should have κ = 0, got {}", k);
    }

    #[test]
    fn refresh_edge_ricci_persists_and_clears_stale_flag() {
        let (_td, s) = store();
        let u = s.create_node_in_community(&["N"], 1);
        let v = s.create_node_in_community(&["N"], 1);
        let w = s.create_node_in_community(&["N"], 1);
        link(&s, u, v);
        link(&s, v, w);
        link(&s, w, u);

        // Find the slot of the first u→v edge and set its stale
        // flag to simulate a mutation that invalidated the
        // cached ricci.
        let (_nbr, eid) = s
            .edges_from(u, Direction::Outgoing)
            .into_iter()
            .find(|(n, _)| *n == v)
            .expect("u→v edge");
        let mut rec = s.writer().read_edge(eid.0).unwrap().unwrap();
        rec.flags |= edge_flags::RICCI_STALE;
        s.writer().update_edge(eid.0, rec).unwrap();

        let k = refresh_edge_ricci(&s, eid).unwrap().expect("κ computed");
        assert!((k - 0.5).abs() < 1e-2, "triangle κ = 0.5, got {}", k);

        let rec_after = s.writer().read_edge(eid.0).unwrap().unwrap();
        assert!(
            rec_after.flags & edge_flags::RICCI_STALE == 0,
            "RICCI_STALE flag must be cleared after refresh"
        );
        assert!(
            (rec_after.ricci_f32() - 0.5).abs() < 1e-2,
            "persisted ricci_f32 should round-trip to 0.5, got {}",
            rec_after.ricci_f32()
        );
    }

    #[test]
    fn refresh_all_ricci_stamps_every_live_edge_on_trefoil() {
        let (_td, s) = store();
        let (triangle_edges, bridge_edges) = build_trefoil(&s);

        let stats = refresh_all_ricci(&s).unwrap();
        // 9 triangle edges × 2 (bidirectional) + 3 bridge edges × 2
        //   = 18 + 6 = 24 directed edges in the substrate.
        assert_eq!(stats.edges_visited, 24);
        assert_eq!(stats.edges_skipped_dangling, 0);
        // Every edge was freshly quantised (all started at ricci_u8=0
        // which maps to -1.0, triangle edges should now be +0.5 and
        // bridges around -1/3 — all different from -1.0).
        assert_eq!(stats.edges_updated, 24);
        assert!(
            stats.max_abs_curvature > 0.3 && stats.max_abs_curvature <= 1.0,
            "max |κ| should be ~0.5 for this fixture, got {}",
            stats.max_abs_curvature
        );

        // Spot-check persisted values — iterate every directed edge
        // matching each (u, v) triangle probe and verify the stored
        // ricci_u8 round-trips to a positive value.
        for (u, v) in &triangle_edges {
            let (_, eid) = s
                .edges_from(*u, Direction::Outgoing)
                .into_iter()
                .find(|(n, _)| *n == *v)
                .expect("triangle edge exists");
            let rec = s.writer().read_edge(eid.0).unwrap().unwrap();
            assert!(
                rec.ricci_f32() > 0.1,
                "persisted κ for triangle edge ({:?},{:?}) should be > 0.1, got {}",
                u,
                v,
                rec.ricci_f32()
            );
            assert!(
                rec.flags & edge_flags::RICCI_STALE == 0,
                "RICCI_STALE flag must be cleared after batch refresh"
            );
        }
        for (u, v) in &bridge_edges {
            let (_, eid) = s
                .edges_from(*u, Direction::Outgoing)
                .into_iter()
                .find(|(n, _)| *n == *v)
                .expect("bridge edge exists");
            let rec = s.writer().read_edge(eid.0).unwrap().unwrap();
            assert!(
                rec.ricci_f32() < -0.1,
                "persisted κ for bridge edge ({:?},{:?}) should be < -0.1, got {}",
                u,
                v,
                rec.ricci_f32()
            );
        }
    }

    #[test]
    fn refresh_all_ricci_is_idempotent() {
        let (_td, s) = store();
        let _ = build_trefoil(&s);

        let first = refresh_all_ricci(&s).unwrap();
        let second = refresh_all_ricci(&s).unwrap();

        // First pass updates every edge (u8 changes from 0 default).
        // Second pass should find every ricci_u8 already at the
        // correct quantised value → zero updates, same visit count.
        assert_eq!(first.edges_visited, second.edges_visited);
        assert_eq!(second.edges_updated, 0, "second pass must be a no-op");
    }

    #[test]
    fn mark_incident_edges_stale_sets_flag_on_both_directions() {
        let (_td, s) = store();
        let u = s.create_node_in_community(&["N"], 1);
        let v = s.create_node_in_community(&["N"], 1);
        let w = s.create_node_in_community(&["N"], 1);
        link(&s, u, v);
        link(&s, v, w);
        link(&s, w, u);

        let marked = mark_incident_edges_stale(&s, u).unwrap();
        // u has 4 incident directed edges: out to v, out to w, in from v, in from w
        // → 4 edges, all should get RICCI_STALE.
        assert_eq!(marked, 4);

        let edge_hw = s.edge_slot_high_water();
        let mut stale_count = 0u64;
        for slot in 1..edge_hw {
            let Some(rec) = s.writer().read_edge(slot).unwrap() else {
                continue;
            };
            if rec.is_tombstoned() {
                continue;
            }
            if rec.src == u.0 as u32 || rec.dst == u.0 as u32 {
                assert!(
                    rec.flags & edge_flags::RICCI_STALE != 0,
                    "edge (src={}, dst={}) incident to u should be stale",
                    rec.src,
                    rec.dst
                );
                stale_count += 1;
            } else {
                assert!(
                    rec.flags & edge_flags::RICCI_STALE == 0,
                    "edge (src={}, dst={}) NOT incident to u must not be stale",
                    rec.src,
                    rec.dst
                );
            }
        }
        assert_eq!(stale_count, 4);
    }

    /// Verification for Step 2: after N random edge mutations, the
    /// delta-refreshed `ricci_u8` values must match a ground-truth
    /// full batch refresh within ±1 Q-unit (the quantisation step
    /// is 1/127.5 ≈ 0.0078, matching the plan's "± 1 Q-unit" gate).
    #[test]
    fn delta_refresh_matches_batch_within_one_q_unit() {
        use std::collections::HashMap;

        // Build a non-trivial graph: 3 communities of 5 nodes each,
        // each community fully connected internally, 2 cross-
        // community bridges. Enough structure for non-zero κ on
        // almost every edge.
        let (_td, s) = store();
        let mut nodes: Vec<NodeId> = Vec::new();
        for c in 0..3u32 {
            for _ in 0..5 {
                nodes.push(s.create_node_in_community(&["N"], c + 1));
            }
        }
        // Intra-community links.
        for c in 0..3 {
            for i in 0..5 {
                for j in (i + 1)..5 {
                    link(&s, nodes[c * 5 + i], nodes[c * 5 + j]);
                }
            }
        }
        // Cross-community bridges.
        link(&s, nodes[0], nodes[5]); // A-B
        link(&s, nodes[5], nodes[10]); // B-C

        // Initial batch refresh — ground truth baseline.
        let _baseline = refresh_all_ricci(&s).unwrap();

        // Apply 100 mutations. Each mutation is either an edge
        // create or an edge delete on a randomly-picked pair.
        let mut rng = super::tests_rng::Xorshift64::new(0xA55A_1234);
        let mut affected_nodes: Vec<NodeId> = Vec::new();
        for _ in 0..100 {
            let i = rng.range(nodes.len() as u64) as usize;
            let j = rng.range(nodes.len() as u64) as usize;
            if i == j {
                continue;
            }
            let u = nodes[i];
            let v = nodes[j];
            let create = rng.range(2) == 0;
            if create {
                let _ = s.create_edge(u, v, "E");
            } else {
                // Only delete if there's at least one outgoing edge
                // u→v — otherwise the mutation is a no-op and
                // injects no change.
                if let Some((_n, eid)) = s
                    .edges_from(u, Direction::Outgoing)
                    .into_iter()
                    .find(|(n, _)| *n == v)
                {
                    let _ = s.delete_edge(eid);
                }
            }
            affected_nodes.push(u);
            affected_nodes.push(v);
        }

        // Delta path: refresh only edges incident to touched nodes.
        affected_nodes.sort_by_key(|n| n.0);
        affected_nodes.dedup();
        let _delta_stats = refresh_ricci_for_nodes(&s, &affected_nodes).unwrap();

        // Snapshot every live edge's ricci_u8 after the delta pass.
        let edge_hw = s.edge_slot_high_water();
        let mut delta_map: HashMap<u64, u8> = HashMap::new();
        for slot in 1..edge_hw {
            if let Some(rec) = s.writer().read_edge(slot).unwrap() {
                if !rec.is_tombstoned() {
                    delta_map.insert(slot, rec.ricci_u8);
                }
            }
        }

        // Ground truth: run a full batch refresh (idempotent — only
        // rewrites slots that differ).
        let _truth_stats = refresh_all_ricci(&s).unwrap();
        let mut truth_map: HashMap<u64, u8> = HashMap::new();
        for slot in 1..edge_hw {
            if let Some(rec) = s.writer().read_edge(slot).unwrap() {
                if !rec.is_tombstoned() {
                    truth_map.insert(slot, rec.ricci_u8);
                }
            }
        }

        // Compare — every slot must match within 1 Q-unit.
        assert_eq!(
            delta_map.len(),
            truth_map.len(),
            "edge count mismatch post-mutation"
        );
        for (slot, delta_q) in &delta_map {
            let truth_q = truth_map.get(slot).copied().unwrap_or(0);
            let diff = (*delta_q as i32 - truth_q as i32).abs();
            assert!(
                diff <= 1,
                "slot {} diverged: delta_q={} truth_q={} (diff {} > 1)",
                slot,
                delta_q,
                truth_q,
                diff
            );
        }
    }

    #[test]
    fn refresh_stale_edges_picks_up_flagged_edges() {
        let (_td, s) = store();
        let (_tri, _br) = build_trefoil(&s);
        // Baseline: every edge freshly computed.
        let _ = refresh_all_ricci(&s).unwrap();

        // Pick one live directed edge and manually set its
        // RICCI_STALE flag AND flip its stored ricci_u8 to a
        // deliberately-wrong value so refresh_stale_edges has
        // something concrete to fix.
        let mut target_slot = 0u64;
        let edge_hw = s.edge_slot_high_water();
        for slot in 1..edge_hw {
            if let Some(rec) = s.writer().read_edge(slot).unwrap() {
                if !rec.is_tombstoned() {
                    target_slot = slot;
                    break;
                }
            }
        }
        assert!(target_slot > 0);
        let mut rec = s.writer().read_edge(target_slot).unwrap().unwrap();
        let correct_q = rec.ricci_u8;
        rec.ricci_u8 = 0; // maps to -1.0, definitely wrong for a triangle/bridge edge
        rec.flags |= edge_flags::RICCI_STALE;
        s.writer().update_edge(target_slot, rec).unwrap();

        // Delta path picks up the flag.
        let stats = refresh_stale_edges(&s).unwrap();
        assert!(stats.edges_visited > 0);
        // Stale flag clears, ricci_u8 restored to the correct value.
        let post = s.writer().read_edge(target_slot).unwrap().unwrap();
        assert_eq!(post.flags & edge_flags::RICCI_STALE, 0);
        assert!(
            (post.ricci_u8 as i32 - correct_q as i32).abs() <= 1,
            "ricci_u8 should round-trip to {} ±1, got {}",
            correct_q,
            post.ricci_u8
        );
    }

    #[test]
    fn compute_for_edge_returns_none_on_tombstoned() {
        let (_td, s) = store();
        let u = s.create_node_in_community(&["N"], 1);
        let v = s.create_node_in_community(&["N"], 1);
        let eid = s.create_edge(u, v, "E");

        // Tombstone the edge via the writer's low-level API.
        s.writer().tombstone_edge(eid.0).unwrap();

        let r = compute_ricci_for_edge(&s, eid).unwrap();
        assert!(r.is_none(), "tombstoned edge should yield None");
    }

    // -----------------------------------------------------------------
    // Step 3 — node-level curvature aggregation
    // -----------------------------------------------------------------

    /// Zachary's karate club: the canonical 34-node / 78-edge network
    /// used as a smoke test for geometric community detection.
    /// Node 0 is "Mr. Hi" (instructor), node 33 is the "Officer";
    /// the graph famously splits into two factions centered on these
    /// two hubs. The broker nodes that straddle both factions (2, 8,
    /// 13, 31) are well-known high-betweenness bridges — Ollivier-Ricci
    /// theory predicts their aggregated curvature is negative.
    fn build_karate(
        s: &SubstrateStore,
    ) -> Vec<NodeId> {
        let nodes: Vec<NodeId> = (0..34)
            .map(|_| s.create_node_in_community(&["N"], 1))
            .collect();
        // 78 unique undirected edges — Zachary 1977 edge list.
        let edges: &[(usize, usize)] = &[
            (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8),
            (0, 10), (0, 11), (0, 12), (0, 13), (0, 17), (0, 19), (0, 21),
            (0, 31),
            (1, 2), (1, 3), (1, 7), (1, 13), (1, 17), (1, 19), (1, 21),
            (1, 30),
            (2, 3), (2, 7), (2, 8), (2, 9), (2, 13), (2, 27), (2, 28),
            (2, 32),
            (3, 7), (3, 12), (3, 13),
            (4, 6), (4, 10),
            (5, 6), (5, 10), (5, 16),
            (6, 16),
            (8, 30), (8, 32), (8, 33),
            (9, 33),
            (13, 33),
            (14, 32), (14, 33),
            (15, 32), (15, 33),
            (18, 32), (18, 33),
            (19, 33),
            (20, 32), (20, 33),
            (22, 32), (22, 33),
            (23, 25), (23, 27), (23, 29), (23, 32), (23, 33),
            (24, 25), (24, 27), (24, 31),
            (25, 31),
            (26, 29), (26, 33),
            (27, 33),
            (28, 31), (28, 33),
            (29, 32), (29, 33),
            (30, 32), (30, 33),
            (31, 32), (31, 33),
            (32, 33),
        ];
        for &(a, b) in edges {
            link(s, nodes[a], nodes[b]);
        }
        nodes
    }

    #[test]
    fn karate_brokers_have_more_negative_curvature_than_faction_hubs() {
        // # Why this test
        //
        // The plan's verification clause for Step 3 is verbatim:
        // "Nodes de haut betweenness sur graphe synthétique →
        //  curvature négative ; test sur karate network."
        //
        // Ollivier-Ricci (and its Jost-Liu lower bound) encodes
        // topological bottleneck: an edge or node lying on many
        // shortest paths between otherwise-sparsely-connected
        // communities shows up as negatively curved, because the
        // optimal transport cost from one side to the other is
        // driven by the thinness of the bridge rather than the
        // endpoint density. The two factional hubs (0 = Mr. Hi,
        // 33 = Officer) are high-degree but embedded in their own
        // dense communities, so their edges land in tight triangles;
        // their aggregated curvature stays near zero. The brokers
        // (2, 8, 13, 31) straddle the faction boundary — many of
        // their edges cross into the opposite community with no
        // shared neighbours, so their aggregate drops strongly
        // negative.
        //
        // # Hand-check for node 2
        //
        // Node 2 has degree 10. Neighbours: {0,1,3,7,8,9,13,27,28,32}.
        // Six of the ten edges hit members of the Officer faction or
        // periphery (8, 9, 27, 28, 32) and three more lose their
        // triangle support (intra-faction peers whose neighbours
        // don't overlap with node 2's full reach). Summing the
        // per-edge Jost-Liu values gives roughly κ̄ ≈ -0.33.
        //
        // # Robustness of the assertion
        //
        // Rather than pin to a single numeric slot (the persisted
        // ricci_u8 loses ~0.008 per Q-unit), we verify the
        // *ordering* that the plan cares about: brokers strictly
        // below hubs. Also assert that node 2's aggregated
        // curvature is negative — that's the single most defensible
        // point estimate on this fixture.
        let (_td, s) = store();
        let n = build_karate(&s);

        // Populate every edge's ricci_u8.
        let refresh_stats = refresh_all_ricci(&s).unwrap();
        // 78 undirected edges × 2 directions = 156 directed edge
        // records in the substrate.
        assert_eq!(
            refresh_stats.edges_visited, 156,
            "karate: expected 156 directed edges, got {}",
            refresh_stats.edges_visited
        );
        assert_eq!(refresh_stats.edges_skipped_dangling, 0);

        // Batch-compute per-node curvature.
        let kmap = compute_all_node_curvatures(&s).unwrap();
        assert_eq!(
            kmap.len(),
            34,
            "every karate node has ≥ 1 edge → 34 entries"
        );

        let k = |i: usize| -> f32 {
            *kmap
                .get(&n[i])
                .unwrap_or_else(|| panic!("κ missing for node {}", i))
        };

        let k2 = k(2);
        let k8 = k(8);
        let k13 = k(13);
        let k31 = k(31);
        let k0 = k(0); // Mr. Hi
        let k33 = k(33); // Officer

        // Claim 1: node 2's aggregated curvature is strictly
        // negative. This is the single most-stable numeric check
        // and corresponds to a hand-verified κ̄ ≈ -0.33.
        assert!(
            k2 < -0.05,
            "node 2 (canonical broker) should have κ̄ < -0.05, got {}",
            k2
        );

        // Claim 2: the broker cohort (2, 8, 13, 31) is on average
        // strictly more negative than the two faction hubs.
        let broker_mean = (k2 + k8 + k13 + k31) / 4.0;
        let hub_mean = (k0 + k33) / 2.0;
        assert!(
            broker_mean < hub_mean,
            "brokers (mean {:.3}: k2={:.3}, k8={:.3}, k13={:.3}, k31={:.3}) \
             should be strictly below faction hubs (mean {:.3}: k0={:.3}, k33={:.3})",
            broker_mean, k2, k8, k13, k31, hub_mean, k0, k33,
        );

        // Claim 3: at least three of the four brokers are
        // individually negative. Guards against the case where
        // one flukey positive outlier could drag the mean below
        // the hubs while hiding the intended signal.
        let negative_brokers = [k2, k8, k13, k31]
            .iter()
            .filter(|&&v| v < 0.0)
            .count();
        assert!(
            negative_brokers >= 3,
            "expected ≥ 3 of {{k2, k8, k13, k31}} to be negative, got {} (values: k2={:.3}, k8={:.3}, k13={:.3}, k31={:.3})",
            negative_brokers, k2, k8, k13, k31,
        );
    }

    #[test]
    fn node_curvature_none_on_isolated_node() {
        // Node with no incident edges → None.
        let (_td, s) = store();
        let isolated = s.create_node_in_community(&["N"], 1);
        assert!(
            node_curvature(&s, isolated).is_none(),
            "isolated node should yield None curvature"
        );
    }

    // -----------------------------------------------------------------
    // Step 4 — heat kernel diffusion
    // -----------------------------------------------------------------

    /// Build an undirected linear chain of `n` nodes: 0 — 1 — 2 — … — (n-1).
    /// Returns the node id list in chain order.
    fn build_chain(s: &SubstrateStore, n: usize) -> Vec<NodeId> {
        let nodes: Vec<NodeId> = (0..n)
            .map(|_| s.create_node_in_community(&["N"], 1))
            .collect();
        for i in 0..n - 1 {
            link(s, nodes[i], nodes[i + 1]);
        }
        nodes
    }

    #[test]
    fn heat_step_preserves_mass() {
        // Conservation law: for an undirected graph the Laplacian
        // matrix-vector `L · x` has zero row-sum, so `x' = x − dt · L·x`
        // preserves `Σ x`. Verify on a 20-node chain with a single
        // Dirac impulse.
        let (_td, s) = store();
        let nodes = build_chain(&s, 20);
        let hw = s.slot_high_water() as usize;

        let mut state = vec![0.0_f32; hw];
        let mid = nodes[nodes.len() / 2];
        state[mid.0 as usize] = 1.0;

        let before: f32 = state.iter().sum();
        // Max degree in the chain is 2, so dt ≤ 0.5 is stable.
        heat_step_unweighted(&s, &mut state, 0.25).unwrap();
        let after: f32 = state.iter().sum();
        assert!(
            (after - before).abs() < 1e-5,
            "heat step must preserve total mass: before={}, after={}",
            before,
            after
        );
    }

    #[test]
    fn heat_step_spreads_dirac_to_neighbours() {
        // After ONE heat step on an unweighted chain with dt=0.25 and
        // a Dirac at node k, we expect:
        //   new[k]   = 1 − 2·0.25·1 = 0.5
        //   new[k±1] = 0 − 0.25·(0 − 1) = 0.25
        //   all other = 0
        //
        // This is the closed-form single-step linearisation — pure
        // algebra, no Monte Carlo.
        let (_td, s) = store();
        let nodes = build_chain(&s, 9);
        let hw = s.slot_high_water() as usize;

        let mut state = vec![0.0_f32; hw];
        let k = 4usize; // middle of the 9-node chain
        let mid_id = nodes[k];
        state[mid_id.0 as usize] = 1.0;

        heat_step_unweighted(&s, &mut state, 0.25).unwrap();

        assert!(
            (state[mid_id.0 as usize] - 0.5).abs() < 1e-5,
            "center should drop 1.0 → 0.5, got {}",
            state[mid_id.0 as usize]
        );
        let left = nodes[k - 1];
        let right = nodes[k + 1];
        assert!(
            (state[left.0 as usize] - 0.25).abs() < 1e-5,
            "left neighbour should rise 0 → 0.25, got {}",
            state[left.0 as usize]
        );
        assert!(
            (state[right.0 as usize] - 0.25).abs() < 1e-5,
            "right neighbour should rise 0 → 0.25, got {}",
            state[right.0 as usize]
        );
        // Second-hop neighbours still at 0.
        let far_left = nodes[k - 2];
        let far_right = nodes[k + 2];
        assert!(state[far_left.0 as usize].abs() < 1e-6);
        assert!(state[far_right.0 as usize].abs() < 1e-6);
    }

    #[test]
    fn heat_diffusion_on_linear_chain_matches_gaussian_variance() {
        // # The acceptance criterion from the plan (verbatim):
        //   "Sur chaîne linéaire, diffusion gaussienne attendue
        //    (variance = 2t)."
        //
        // # What we actually check
        //
        // A chain of N nodes with a Dirac at the center, evolved for
        // `K` steps of size `dt`. For small `dt` the discrete heat
        // equation converges to the continuous one, whose Green's
        // function is Gaussian with variance `σ² = 2·t` where
        // `t = K·dt` is the total simulated time.
        //
        // # Why we measure σ² / t rather than σ² alone
        //
        // The one-step Laplacian on an unweighted graph has stencil
        // [1, -2, 1] which is a second-order-accurate
        // finite-difference discretisation of `∂²/∂x²`. With unit
        // lattice spacing `h = 1`, the continuum limit has diffusion
        // coefficient `D = 1`, and σ² = 2·D·t = 2t. The tolerance
        // window accounts for:
        //
        //   * boundary reflection near chain endpoints (< 2% on a
        //     200-node chain with t = 8 — σ = 4 stays well inside),
        //   * explicit-Euler's O(dt) truncation error on the
        //     first-few moments.
        //
        // # Calibration
        //
        // N = 201 (ensures a central node with integer offset
        //          range ±100), K = 200 steps, dt = 0.04.
        // Total time: t = K · dt = 8.  Expected σ² = 16, σ ≈ 4.
        // Chain boundary is at distance 100 from the center, ~25σ
        // away — reflection is negligible.
        let (_td, s) = store();
        let n = 201usize;
        let nodes = build_chain(&s, n);
        let hw = s.slot_high_water() as usize;

        let mut state = vec![0.0_f32; hw];
        let center_idx = n / 2;
        let center_slot = nodes[center_idx].0 as usize;
        state[center_slot] = 1.0;

        let dt = 0.04_f32;
        let k_steps = 200usize;
        heat_diffuse_unweighted(&s, &mut state, dt, k_steps).unwrap();

        // Mass conservation sanity — catches a regression in the
        // Laplacian sign before we start computing moments on a
        // garbage distribution.
        let total: f32 = state.iter().copied().sum();
        assert!(
            (total - 1.0).abs() < 1e-4,
            "mass conservation violated: Σx = {} (expected 1.0)",
            total
        );

        // Compute the first moment (should be ~0 around the center)
        // and the second central moment (the variance).
        let mut mean: f64 = 0.0;
        for i in 0..n {
            let slot = nodes[i].0 as usize;
            let offset = i as f64 - center_idx as f64;
            mean += offset * state[slot] as f64;
        }
        // mean is already in "offset from center" coordinates.
        let mut var: f64 = 0.0;
        for i in 0..n {
            let slot = nodes[i].0 as usize;
            let offset = i as f64 - center_idx as f64 - mean;
            var += offset * offset * state[slot] as f64;
        }

        let expected_var = 2.0 * (k_steps as f64 * dt as f64); // 2t = 16
        let rel_err = (var - expected_var).abs() / expected_var;
        assert!(
            mean.abs() < 1e-3,
            "diffusion must stay centred: mean offset = {}",
            mean
        );
        assert!(
            rel_err < 0.05,
            "variance should match 2t within 5%: got σ²={:.3}, expected {:.3}, rel_err {:.3}",
            var,
            expected_var,
            rel_err
        );
    }

    #[test]
    fn heat_step_rejects_undersized_buffer() {
        let (_td, s) = store();
        let _ = build_chain(&s, 10);
        let mut too_small = vec![0.0_f32; 3]; // 3 < slot_high_water (≥10)
        let r = heat_step_unweighted(&s, &mut too_small, 0.1);
        assert!(
            r.is_err(),
            "heat_step must reject a state buffer smaller than slot_high_water"
        );
    }

    #[test]
    fn heat_step_weighted_is_a_noop_on_fresh_edges() {
        // Freshly-created edges have weight_u16 = 0, so the weighted
        // Laplacian is identically zero — diffusion is a no-op.
        // This is the contract that justifies exposing the
        // [`heat_step_unweighted`] variant as a separate entry point.
        let (_td, s) = store();
        let nodes = build_chain(&s, 5);
        let hw = s.slot_high_water() as usize;
        let mut state = vec![0.0_f32; hw];
        state[nodes[2].0 as usize] = 1.0;
        let before = state.clone();
        heat_step_weighted(&s, &mut state, 0.25).unwrap();
        assert_eq!(
            state, before,
            "zero-weight edges → weighted heat step is a no-op"
        );
    }

    #[test]
    fn node_curvature_falls_back_to_unweighted_mean_on_zero_weights() {
        // Fresh edges default weight_u16 = 0 → the code path must
        // fall back to arithmetic mean rather than divide by zero.
        let (_td, s) = store();
        let n0 = s.create_node_in_community(&["N"], 1);
        let n1 = s.create_node_in_community(&["N"], 1);
        let n2 = s.create_node_in_community(&["N"], 1);
        link(&s, n0, n1);
        link(&s, n1, n2);
        link(&s, n2, n0);
        let _ = refresh_all_ricci(&s).unwrap();

        // Pure triangle → every edge κ = 0.5 → every node κ̄ = 0.5
        // (within quantisation ±0.01 for ricci_u8).
        for nid in [n0, n1, n2] {
            let k = node_curvature(&s, nid).expect("κ̄ defined");
            assert!(
                (k - 0.5).abs() < 0.02,
                "triangle node κ̄ should be 0.5 (±0.02 q-noise), got {}",
                k
            );
        }
    }

    // -----------------------------------------------------------------
    // T12 Step 8 — Forman-Ricci property tests.
    // -----------------------------------------------------------------

    #[test]
    fn forman_trefoil_bridges_are_more_negative_than_triangle_interiors() {
        // Forman-Ricci has the same sign convention as Ollivier on the
        // trefoil: triangle-interior edges have higher curvature than
        // bridge edges. Concrete numbers for the trefoil probe edges
        // (note: some triangle endpoints also touch a bridge, so their
        // degrees are 3 not 2):
        //   * triangle edge (a2,a3): deg(a2)=3 (a1,a3,c2), deg(a3)=2,
        //     common={a1} → κ_F = 2 − 3 − 2 + 3·1 = 0
        //   * bridge edge (a1,b1):  deg(a1)=3, deg(b1)=3, 0 common
        //     → κ_F = 2 − 3 − 3 + 0 = −4
        let (_td, s) = store();
        let (triangle_edges, bridge_edges) = build_trefoil(&s);
        let mut interior_max = i32::MIN;
        for (u, v) in &triangle_edges {
            let kf = compute_forman_fast(&s, *u, *v);
            assert!(
                kf >= 0,
                "triangle edge ({:?},{:?}) Forman should be ≥ 0, got {}",
                u,
                v,
                kf
            );
            interior_max = interior_max.max(kf);
        }
        let mut bridge_min = i32::MAX;
        for (u, v) in &bridge_edges {
            let kf = compute_forman_fast(&s, *u, *v);
            assert!(
                kf <= -2,
                "bridge edge ({:?},{:?}) Forman should be ≤ -2, got {}",
                u,
                v,
                kf
            );
            bridge_min = bridge_min.min(kf);
        }
        // Fundamental invariant: every bridge is strictly more
        // negative than every triangle interior.
        assert!(
            bridge_min < 0 && interior_max >= 0 && bridge_min < interior_max,
            "bridges must be strictly more negative than interior; \
             interior_max = {}, bridge_min = {}",
            interior_max,
            bridge_min
        );
    }

    #[test]
    fn forman_agrees_in_sign_with_ollivier_on_trefoil() {
        // Sign-parity: every triangle-interior edge has κ_O > 0 AND
        // κ_F ≥ 0; every bridge edge has κ_O < 0 AND κ_F < 0.
        let (_td, s) = store();
        let (triangle_edges, bridge_edges) = build_trefoil(&s);
        for (u, v) in &triangle_edges {
            let ko = compute_ricci_fast(&s, *u, *v);
            let kf = compute_forman_fast(&s, *u, *v);
            assert!(
                ko > 0.0 && kf >= 0,
                "interior edge sign mismatch: κ_O = {}, κ_F = {}",
                ko,
                kf
            );
        }
        for (u, v) in &bridge_edges {
            let ko = compute_ricci_fast(&s, *u, *v);
            let kf = compute_forman_fast(&s, *u, *v);
            assert!(
                ko < 0.0 && kf < 0,
                "bridge edge sign mismatch: κ_O = {}, κ_F = {}",
                ko,
                kf
            );
        }
    }

    #[test]
    fn forman_batch_covers_every_live_edge() {
        let (_td, s) = store();
        let _ = build_trefoil(&s);
        let map = compute_all_forman(&s).expect("batch forman");
        assert!(
            !map.is_empty(),
            "batch forman returned empty map on a non-empty graph"
        );
        // Spot check: every value is within [−2·max_deg, +3·max_deg]
        // (trefoil max degree is 3, so bounds are [−6, +9]).
        for (_, k) in map {
            assert!(
                (-10..=10).contains(&k),
                "Forman value {} out of sane range for trefoil",
                k
            );
        }
    }

    // -----------------------------------------------------------------
    // T12 Step 7 — bottleneck_nodes API.
    // -----------------------------------------------------------------

    #[test]
    fn bottleneck_nodes_flags_negative_curvature_nodes_on_isthmus() {
        // Two triangles connected by a single isthmus node I. The
        // isthmus edges I-A1 and I-B1 are pure bridges (no common
        // neighbour, so κ < 0). I has degree 2 and both incident
        // edges are bridges → κ̄(I) < 0 unambiguously.
        let (_td, s) = store();
        let a1 = s.create_node_in_community(&["N"], 1);
        let a2 = s.create_node_in_community(&["N"], 1);
        let a3 = s.create_node_in_community(&["N"], 1);
        let b1 = s.create_node_in_community(&["N"], 2);
        let b2 = s.create_node_in_community(&["N"], 2);
        let b3 = s.create_node_in_community(&["N"], 2);
        let isthmus = s.create_node_in_community(&["N"], 3);
        link(&s, a1, a2);
        link(&s, a2, a3);
        link(&s, a3, a1);
        link(&s, b1, b2);
        link(&s, b2, b3);
        link(&s, b3, b1);
        link(&s, a1, isthmus);
        link(&s, isthmus, b1);
        let _ = refresh_all_ricci(&s).unwrap();

        let bottlenecks = bottleneck_nodes(&s, 0.0);
        assert!(
            !bottlenecks.is_empty(),
            "isthmus graph must have at least one bottleneck (κ̄ < 0)"
        );
        // Sorted ascending.
        for w in bottlenecks.windows(2) {
            assert!(
                w[0].1 <= w[1].1,
                "bottleneck_nodes must be sorted ascending, got {} > {}",
                w[0].1,
                w[1].1
            );
        }
        // Everything returned is < threshold.
        for (_, k) in &bottlenecks {
            assert!(*k < 0.0, "bottleneck node κ̄ = {} ≥ threshold 0.0", k);
        }
        // The isthmus node itself must appear in the bottleneck list.
        let flagged: Vec<NodeId> = bottlenecks.iter().map(|(n, _)| *n).collect();
        assert!(
            flagged.contains(&isthmus),
            "isthmus node {:?} should be flagged, got {:?}",
            isthmus,
            flagged
        );
    }

    // -----------------------------------------------------------------
    // T12 Step 10 — Heat Kernel Signatures.
    // -----------------------------------------------------------------

    #[test]
    fn hks_distinguishes_clique_from_tree() {
        // K_4 nodes all have identical local structure → identical
        // HKS. Three-node path endpoints (degree 1) differ from the
        // midpoint (degree 2) at t_local.
        let (_td, s_clique) = store();
        let n: Vec<NodeId> = (0..4)
            .map(|_| s_clique.create_node_in_community(&["N"], 1))
            .collect();
        for i in 0..n.len() {
            for j in (i + 1)..n.len() {
                link(&s_clique, n[i], n[j]);
            }
        }
        let hks_clique = compute_hks_descriptors(&s_clique).expect("hks");
        // All four nodes have the same local structure, so their
        // local-scale signatures should be within floating noise.
        let h1 = hks_clique[n[0].0 as usize];
        let h2 = hks_clique[n[1].0 as usize];
        assert!(
            hks_distance(&h1, &h2) < 0.1,
            "K_4 nodes should have ~identical HKS, got distance {}",
            hks_distance(&h1, &h2)
        );

        // Path graph: endpoints differ from midpoints.
        let (_td2, s_path) = store();
        let p = build_chain(&s_path, 5);
        let hks_path = compute_hks_descriptors(&s_path).expect("hks path");
        let endpoint = hks_path[p[0].0 as usize];
        let midpoint = hks_path[p[2].0 as usize];
        assert!(
            hks_distance(&endpoint, &midpoint) > 1e-3,
            "path endpoint and midpoint should have distinguishable HKS, got {}",
            hks_distance(&endpoint, &midpoint)
        );
    }

    #[test]
    fn hks_decays_monotonically_with_time() {
        // The heat kernel amplitude at a node decays monotonically
        // with t on a connected graph (no sources). Verify on a
        // 10-node chain at the mid node.
        let (_td, s) = store();
        let nodes = build_chain(&s, 10);
        let mid = nodes[5];
        let csr = CsrAdjacency::build(&s);
        let h_local =
            heat_kernel_signature_with_csr(&csr, HKS_T_LOCAL).unwrap();
        let h_meso =
            heat_kernel_signature_with_csr(&csr, HKS_T_MESO).unwrap();
        let h_global =
            heat_kernel_signature_with_csr(&csr, HKS_T_GLOBAL).unwrap();
        let slot = mid.0 as usize;
        assert!(
            h_local[slot] >= h_meso[slot] && h_meso[slot] >= h_global[slot],
            "HKS should decay monotonically: local {}, meso {}, global {}",
            h_local[slot],
            h_meso[slot],
            h_global[slot]
        );
    }

    // -----------------------------------------------------------------
    // T12 Step 5 — Geodesic distance.
    // -----------------------------------------------------------------

    #[test]
    fn geodesic_zero_weight_chain_costs_one_per_hop() {
        // Every edge has weight_u16 = 0 (fresh), so cost per hop is
        // 1.0 − 0.0 = 1.0. Geodesic between endpoints of a 6-chain
        // is exactly 5.0 (± 1e-4 for f32 noise).
        let (_td, s) = store();
        let nodes = build_chain(&s, 6);
        let csr = CsrAdjacency::build(&s);
        let d = geodesic_distance_csr(&csr, nodes[0].0 as u32, nodes[5].0 as u32)
            .expect("reachable");
        assert!(
            (d - 5.0).abs() < 1e-3,
            "6-chain endpoints geodesic should be 5.0, got {}",
            d
        );
    }

    #[test]
    fn geodesic_is_monotone_in_chain_distance() {
        // On a chain, geodesic(n0, n_k) should be monotone increasing
        // in k — 0, 1, 2, 3, 4, 5 for k = 0..5.
        let (_td, s) = store();
        let nodes = build_chain(&s, 6);
        let csr = CsrAdjacency::build(&s);
        let mut prev = -1.0_f32;
        for k in 0..nodes.len() {
            let d =
                geodesic_distance_csr(&csr, nodes[0].0 as u32, nodes[k].0 as u32)
                    .expect("reachable");
            assert!(
                d > prev,
                "geodesic must be strictly increasing: prev {}, d[n_{}] = {}",
                prev,
                k,
                d
            );
            prev = d;
        }
    }

    #[test]
    fn geodesic_unreachable_returns_none() {
        // Two disconnected 3-chains: nothing from the first one
        // reaches the second.
        let (_td, s) = store();
        let a = build_chain(&s, 3);
        let b = build_chain(&s, 3);
        let csr = CsrAdjacency::build(&s);
        let d = geodesic_distance_csr(&csr, a[0].0 as u32, b[2].0 as u32);
        assert!(d.is_none(), "disconnected components must return None");
    }

    // -----------------------------------------------------------------
    // T12 Step 9 — Effective resistance.
    // -----------------------------------------------------------------

    #[test]
    fn effective_resistance_path_graph_matches_hop_count() {
        // On a path graph with no multi-edges, the effective
        // resistance between two nodes equals the shortest path
        // length (one unit resistor per edge in series). For a
        // 4-hop chain, R(endpoints) = 4 (± CG tolerance).
        let (_td, s) = store();
        let nodes = build_chain(&s, 5);
        let r = effective_resistance(&s, nodes[0], nodes[4])
            .expect("resistance defined");
        assert!(
            (r - 4.0).abs() < 1e-2,
            "4-hop chain resistance should be 4.0 (± 1e-2), got {}",
            r
        );
    }

    #[test]
    fn effective_resistance_triangle_is_two_thirds() {
        // Triangle (3-cycle): two parallel paths (one direct edge of
        // R=1, and one two-edge path of R=2). Parallel formula gives
        // 1 · 2 / (1 + 2) = 2/3.
        let (_td, s) = store();
        let n0 = s.create_node_in_community(&["N"], 1);
        let n1 = s.create_node_in_community(&["N"], 1);
        let n2 = s.create_node_in_community(&["N"], 1);
        link(&s, n0, n1);
        link(&s, n1, n2);
        link(&s, n2, n0);
        let r = effective_resistance(&s, n0, n1).expect("resistance defined");
        assert!(
            (r - 2.0 / 3.0).abs() < 1e-2,
            "triangle resistance should be 2/3, got {}",
            r
        );
    }

    #[test]
    fn effective_resistance_is_less_than_shortest_path() {
        // Redundancy principle: parallel paths reduce effective
        // resistance below the shortest-path cost. On a triangle, the
        // shortest path is 1 hop but R = 2/3 < 1.
        let (_td, s) = store();
        let n0 = s.create_node_in_community(&["N"], 1);
        let n1 = s.create_node_in_community(&["N"], 1);
        let n2 = s.create_node_in_community(&["N"], 1);
        link(&s, n0, n1);
        link(&s, n1, n2);
        link(&s, n2, n0);
        let r = effective_resistance(&s, n0, n1).expect("resistance defined");
        assert!(r < 1.0, "triangle R should be < 1-hop shortest path, got {}", r);
    }

    #[test]
    fn effective_resistance_rejects_same_endpoint() {
        let (_td, s) = store();
        let n0 = s.create_node_in_community(&["N"], 1);
        let n1 = s.create_node_in_community(&["N"], 1);
        link(&s, n0, n1);
        assert!(effective_resistance(&s, n0, n0).is_none());
    }
}
