//! Batch Leiden (Traag-Waltman 2019) — reference implementation for
//! T10 Step 0.
//!
//! Three phases per level:
//!   1. **Local moving.** Iterate nodes (by ascending id) and move each
//!      to the neighboring community that maximizes the modularity gain
//!      ΔQ. Repeat until no node moves.
//!   2. **Refinement.** Within each community produced by phase 1, start
//!      every node as a singleton and run a constrained local move: a
//!      node may only join a refined community that stays in the same
//!      phase-1 community and whose internal weight remains
//!      "well-connected" (γ-threshold). This is Leiden's key addition
//!      over Louvain — it guarantees that every final community is
//!      internally well-connected.
//!   3. **Aggregation.** Build a condensed graph whose super-nodes are
//!      the refined communities; edge weights sum the inter-community
//!      weights; recurse on the condensed graph until no further moves.
//!
//! The final community id for the original node `u` is
//! `refined_level_top[ ... [refined_level_0[u]]]` — we unfold the stack
//! at the end.
//!
//! ## Weight convention
//!
//! The graph is undirected. A non-loop edge (u, v, w) appears once in
//! each of `adj[u]` and `adj[v]`. A self-loop (u, u, w) appears once in
//! `adj[u]`. Node strength follows the convention
//!
//!   `k(u) = Σ_v w(u, v)`  with self-loop counted **twice** in `k(u)`
//!
//! so that `2m = Σ_u k(u)` holds exactly.
//!
//! ## Modularity
//!
//! `Q = (1 / 2m) · Σ_{u, v : c(u) = c(v)} [w(u, v) - γ · k(u) · k(v) / 2m]`
//!
//! with γ = [`LeidenConfig::resolution`]. Standard Newman-Girvan is γ=1.
//!
//! ## Determinism
//!
//! No RNG. Node iteration is always ascending by id; tie-breaking picks
//! the lowest community id on ΔQ equality. Re-running with the same
//! input produces the same partition.

use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Undirected weighted graph. Nodes are addressed by `u32` ids in the
/// range `0..n`. Self-loops are permitted.
///
/// The adjacency list is sorted by neighbor id for determinism and for
/// cheap set-style iteration during aggregation.
#[derive(Debug, Clone)]
pub struct Graph {
    n: u32,
    adj: Vec<Vec<(u32, f64)>>,
    strength: Vec<f64>,
    two_m: f64,
}

impl Graph {
    /// Create a fresh graph with `n` isolated nodes.
    pub fn new(n: u32) -> Self {
        Self {
            n,
            adj: vec![Vec::new(); n as usize],
            strength: vec![0.0; n as usize],
            two_m: 0.0,
        }
    }

    /// Build a graph from an edge list. Duplicate `(u, v)` pairs are
    /// summed. `u == v` creates a self-loop with weight `w`.
    pub fn from_edges(n: u32, edges: impl IntoIterator<Item = (u32, u32, f64)>) -> Self {
        let t_start = Instant::now();
        tracing::debug!(
            target: "obrain_cognitive::leiden",
            "Graph::from_edges: start n={n}"
        );
        let mut adj_map: Vec<HashMap<u32, f64>> = (0..n).map(|_| HashMap::new()).collect();
        tracing::debug!(
            target: "obrain_cognitive::leiden",
            "Graph::from_edges: allocated {n} empty adj maps in {:.2?}",
            t_start.elapsed()
        );
        let t_ins = Instant::now();
        let mut two_m = 0.0f64;
        let mut edge_count: u64 = 0;
        for (u, v, w) in edges {
            assert!(u < n && v < n, "edge endpoints out of range");
            assert!(w.is_finite() && w >= 0.0, "edge weight must be ≥ 0");
            if w == 0.0 {
                continue;
            }
            if u == v {
                *adj_map[u as usize].entry(u).or_insert(0.0) += w;
                two_m += 2.0 * w;
            } else {
                *adj_map[u as usize].entry(v).or_insert(0.0) += w;
                *adj_map[v as usize].entry(u).or_insert(0.0) += w;
                two_m += 2.0 * w;
            }
            edge_count += 1;
        }
        tracing::debug!(
            target: "obrain_cognitive::leiden",
            "Graph::from_edges: inserted {edge_count} edges into adj maps in {:.2?}",
            t_ins.elapsed()
        );
        let t_sort = Instant::now();
        let mut adj: Vec<Vec<(u32, f64)>> = adj_map
            .into_iter()
            .map(|map| {
                let mut v: Vec<(u32, f64)> = map.into_iter().collect();
                v.sort_by_key(|e| e.0);
                v
            })
            .collect();
        tracing::debug!(
            target: "obrain_cognitive::leiden",
            "Graph::from_edges: sorted+collected adj lists in {:.2?}",
            t_sort.elapsed()
        );
        let t_str = Instant::now();
        let mut strength = vec![0.0; n as usize];
        for (u, neighbors) in adj.iter_mut().enumerate() {
            for (v, w) in neighbors.iter() {
                strength[u] += *w;
                if *v == u as u32 {
                    // self-loop: count again so k(u) includes it twice.
                    strength[u] += *w;
                }
            }
        }
        tracing::debug!(
            target: "obrain_cognitive::leiden",
            "Graph::from_edges: strengths computed in {:.2?} — total {:.2?}",
            t_str.elapsed(),
            t_start.elapsed()
        );
        Self {
            n,
            adj,
            strength,
            two_m,
        }
    }

    /// Number of nodes.
    pub fn num_nodes(&self) -> u32 {
        self.n
    }

    /// Sum of edge weights, counted once per undirected edge:
    /// `m = (Σ_u k(u)) / 2`.
    pub fn total_edge_weight(&self) -> f64 {
        self.two_m / 2.0
    }

    /// Node strength (weighted degree, self-loops counted twice).
    pub fn strength(&self, u: u32) -> f64 {
        self.strength[u as usize]
    }

    /// Iterate over `(neighbor, weight)` pairs for node `u`. Sorted by
    /// neighbor id.
    pub fn neighbors(&self, u: u32) -> impl Iterator<Item = (u32, f64)> + '_ {
        self.adj[u as usize].iter().copied()
    }

    /// Current edge weight between `u` and `v`, or `0.0` if absent.
    pub fn edge_weight(&self, u: u32, v: u32) -> f64 {
        match self.adj[u as usize].binary_search_by_key(&v, |e| e.0) {
            Ok(idx) => self.adj[u as usize][idx].1,
            Err(_) => 0.0,
        }
    }

    /// Add a new node, returning its id. Does not touch existing edges.
    pub fn add_node(&mut self) -> u32 {
        let id = self.n;
        self.adj.push(Vec::new());
        self.strength.push(0.0);
        self.n += 1;
        id
    }

    /// Add `delta` to the weight of edge `(u, v)` (creating it if
    /// absent). Negative deltas that bring the weight to ≤ 0 remove
    /// the edge entirely.
    ///
    /// Returns the new weight (0.0 if the edge was removed).
    pub fn add_edge_delta(&mut self, u: u32, v: u32, delta: f64) -> f64 {
        assert!(u < self.n && v < self.n, "endpoints out of range");
        assert!(delta.is_finite(), "delta must be finite");
        if delta == 0.0 {
            return self.edge_weight(u, v);
        }

        let new_w = {
            let adj_u = &mut self.adj[u as usize];
            match adj_u.binary_search_by_key(&v, |e| e.0) {
                Ok(idx) => {
                    let w = adj_u[idx].1 + delta;
                    if w <= 0.0 {
                        adj_u.remove(idx);
                        0.0
                    } else {
                        adj_u[idx].1 = w;
                        w
                    }
                }
                Err(idx) => {
                    assert!(delta > 0.0, "cannot remove non-existent edge");
                    adj_u.insert(idx, (v, delta));
                    delta
                }
            }
        };

        if u != v {
            let adj_v = &mut self.adj[v as usize];
            match adj_v.binary_search_by_key(&u, |e| e.0) {
                Ok(idx) => {
                    if new_w == 0.0 {
                        adj_v.remove(idx);
                    } else {
                        adj_v[idx].1 = new_w;
                    }
                }
                Err(idx) => {
                    adj_v.insert(idx, (u, new_w));
                }
            }
        }

        // Update strengths and 2m.
        if u == v {
            // Self-loop: strength[u] += 2·delta; 2m += 2·delta.
            self.strength[u as usize] += 2.0 * delta;
            self.two_m += 2.0 * delta;
            if new_w == 0.0 {
                // Strength may have drifted slightly above/below zero
                // from accumulated deltas — clamp numerically.
                if self.strength[u as usize].abs() < 1e-12 {
                    self.strength[u as usize] = 0.0;
                }
            }
        } else {
            self.strength[u as usize] += delta;
            self.strength[v as usize] += delta;
            self.two_m += 2.0 * delta;
            if new_w == 0.0 {
                if self.strength[u as usize].abs() < 1e-12 {
                    self.strength[u as usize] = 0.0;
                }
                if self.strength[v as usize].abs() < 1e-12 {
                    self.strength[v as usize] = 0.0;
                }
            }
        }
        new_w
    }
}

/// A community partition: `partition[u] = community id`.
pub type Partition = Vec<u32>;

/// Leiden configuration.
#[derive(Debug, Clone, Copy)]
pub struct LeidenConfig {
    /// Resolution parameter γ (higher → more, smaller communities).
    pub resolution: f64,
    /// Maximum outer iterations across aggregation levels. The
    /// algorithm also terminates early on a level with no moves.
    pub max_levels: usize,
    /// Minimum internal-edge fraction for a refined community to be
    /// considered well-connected during refinement. A subset `S` of
    /// community `C` must satisfy
    /// `w(S, C \ S) ≥ γ · |S| · |C \ S| · 2m / (2m)²` to be retained.
    /// Lower values keep refinement gentle; higher values force more
    /// splits.
    pub refinement_gamma: f64,
}

impl Default for LeidenConfig {
    fn default() -> Self {
        Self {
            resolution: 1.0,
            max_levels: 16,
            refinement_gamma: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Modularity
// ---------------------------------------------------------------------------

/// Compute the modularity of `partition` under resolution `γ`.
///
/// Returns `0.0` for an empty graph.
pub fn modularity(graph: &Graph, partition: &Partition, resolution: f64) -> f64 {
    let two_m = graph.two_m;
    if two_m == 0.0 {
        return 0.0;
    }
    debug_assert_eq!(partition.len(), graph.n as usize);

    // Accumulate per-community internal weight and total strength.
    let mut internal: HashMap<u32, f64> = HashMap::new();
    let mut total: HashMap<u32, f64> = HashMap::new();
    for u in 0..graph.n {
        let cu = partition[u as usize];
        *total.entry(cu).or_insert(0.0) += graph.strength[u as usize];
        for (v, w) in graph.neighbors(u) {
            if partition[v as usize] == cu {
                // Each non-loop edge is iterated twice (once from each
                // endpoint), so `w` already sums to `2 * internal(C)`.
                // Self-loops contribute twice to k(u) above; here we
                // only see the edge once from u's adjacency so we
                // double-count it to stay consistent with strength.
                *internal.entry(cu).or_insert(0.0) += w;
                if u == v {
                    *internal.entry(cu).or_insert(0.0) += w;
                }
            }
        }
    }

    let mut q = 0.0;
    for (c, w_in) in internal.iter() {
        let k_c = *total.get(c).unwrap_or(&0.0);
        // internal[c] is already 2 * Σ internal edges.
        q += w_in / two_m - resolution * (k_c / two_m).powi(2);
    }
    q
}

// ---------------------------------------------------------------------------
// Batch Leiden
// ---------------------------------------------------------------------------

/// Run batch Leiden on `graph`. Returns a [`Partition`] where each
/// entry is the final community id for the corresponding node.
///
/// Community ids are normalized to `0..num_communities` with stable
/// ordering: the first occurrence of each community in ascending node
/// order receives the next available id.
pub fn leiden_batch(graph: &Graph, config: LeidenConfig) -> Partition {
    let n = graph.n;
    if n == 0 {
        return Vec::new();
    }
    // Level 0: each node in its own community.
    let mut partition: Vec<u32> = (0..n).collect();

    // The aggregation loop works on a stack of graphs; the "current"
    // graph is `working` and each level's community assignment maps
    // super-nodes (of `working`) to community ids at the level above.
    let mut working = graph.clone();
    let mut level_maps: Vec<Vec<u32>> = Vec::new();

    let leiden_start = Instant::now();
    tracing::info!(
        target: "obrain_cognitive::leiden",
        "leiden_batch: start n={} m={:.1} max_levels={}",
        working.n,
        working.total_edge_weight(),
        config.max_levels
    );

    for level in 0..config.max_levels {
        let level_start = Instant::now();
        let n_before = working.n;

        // Phase 1 — local move on `working`.
        let lm_start = Instant::now();
        let mut local = (0..working.n).collect::<Vec<u32>>();
        let moved = local_move(&working, &mut local, config.resolution, level);
        let lm_elapsed = lm_start.elapsed();

        // Phase 2 — refinement within each local-move community.
        let rf_start = Instant::now();
        let refined = refine(&working, &local, config.resolution, config.refinement_gamma);
        let rf_elapsed = rf_start.elapsed();

        // Normalize refined ids so they are contiguous starting at 0.
        let (normalized, n_super) = normalize(&refined);

        tracing::info!(
            target: "obrain_cognitive::leiden",
            "level {level}: n={n_before} → n_super={n_super} moved={moved} \
             local_move={:.2?} refine={:.2?} total={:.2?}",
            lm_elapsed,
            rf_elapsed,
            level_start.elapsed()
        );

        if n_super == working.n && !moved {
            // Nothing changed — fixpoint reached.
            level_maps.push(normalized);
            tracing::info!(
                target: "obrain_cognitive::leiden",
                "level {level}: fixpoint reached (no moves, no aggregation)"
            );
            break;
        }

        // Aggregate for the next level.
        let agg_start = Instant::now();
        working = aggregate(&working, &normalized, n_super);
        tracing::info!(
            target: "obrain_cognitive::leiden",
            "level {level}: aggregated in {:.2?} (supernodes={}, edges={:.0})",
            agg_start.elapsed(),
            working.n,
            working.total_edge_weight()
        );
        level_maps.push(normalized);

        if n_super == 1 {
            tracing::info!(
                target: "obrain_cognitive::leiden",
                "level {level}: single community remaining, stopping"
            );
            break;
        }
    }

    tracing::info!(
        target: "obrain_cognitive::leiden",
        "leiden_batch: done in {:.2?} ({} levels processed)",
        leiden_start.elapsed(),
        level_maps.len()
    );

    // Unfold the stack of level maps to compute the final partition for
    // the original nodes.
    // Start from the top map (deepest): its domain size is n_top.
    // Build `current[i]` = community id at the top level for the node
    // whose id is `i` at the current level. We walk from top down.
    if level_maps.is_empty() {
        return partition;
    }
    let top_len = *level_maps.last().unwrap().iter().max().unwrap_or(&0) + 1;
    let mut current: Vec<u32> = (0..top_len).collect();
    for map in level_maps.iter().rev() {
        let mut next = Vec::with_capacity(map.len());
        for &super_id in map.iter() {
            next.push(current[super_id as usize]);
        }
        current = next;
    }
    partition = current;
    // Final normalization for stable ids.
    normalize(&partition).0
}

/// Phase 1 — local moving. Returns `true` if any node moved from its
/// initial community.
///
/// ## Parallelism (synchronous-parallel Louvain pattern, Lu et al. 2015)
///
/// Sequential Louvain processes nodes one at a time, applying each move
/// immediately so the next node sees the updated partition. This serialises
/// the hot loop.
///
/// This implementation uses a **snapshot-and-apply** pass structure:
///
/// 1. **Propose (parallel):** Every node `u` computes the best community
///    it would move to under a frozen snapshot of the current
///    `partition` / `k_c`. Each worker thread owns a per-thread scratch
///    `Vec<f64>` of size `n` (allocated once via `map_init`) so the hot
///    inner loop allocates nothing.
/// 2. **Apply (sequential, ascending order):** Iterate `u = 0..n` and
///    apply any proposal that still improves ΔQ against the *current*
///    (partially mutated) partition — this rejects proposals that would
///    have oscillated ("ABAB" flips between u and a neighbour). Updates
///    `k_c` incrementally so later applies in the same pass see the
///    new state.
///
/// Determinism is preserved: the snapshot is deterministic, the
/// proposals are a pure function of the snapshot, and applies happen in
/// ascending node order. Per-thread scratch state is private, so
/// scheduling cannot change outputs.
///
/// Convergence: equivalent to sequential Louvain up to re-evaluation
/// order. May require slightly more passes than sequential on pathological
/// inputs (the fixpoint is the same — synchronous Louvain still converges
/// monotonically on modularity). A hard pass cap (`MAX_PASSES`) guards
/// against oscillation on toy inputs.
///
/// Emits `tracing::info!` per pass so large graphs (Wikipedia-scale) show
/// progress instead of hanging silently.
fn local_move(graph: &Graph, partition: &mut Partition, resolution: f64, level: usize) -> bool {
    use rayon::prelude::*;

    const MAX_PASSES: usize = 64;

    let n = graph.n;
    let two_m = graph.two_m;
    if two_m == 0.0 {
        return false;
    }
    let m = graph.total_edge_weight();
    let two_m_sq_half = two_m * two_m / 2.0;

    // Community total strength (k_c) — dense vec indexed by community id.
    // Community ids are bounded by n at every level by construction.
    let mut k_c: Vec<f64> = vec![0.0; n as usize];
    for u in 0..n {
        k_c[partition[u as usize] as usize] += graph.strength[u as usize];
    }

    let mut any_move = false;
    let mut pass = 0usize;

    loop {
        pass += 1;
        let pass_start = Instant::now();

        // ---- Snapshot --------------------------------------------------
        // Cloned once per pass so every worker evaluates against the same
        // state. On huge graphs (3.9M nodes × 2 × 8B ≈ 60 MB) this is
        // cheap relative to the parallel work.
        let partition_snap: &Vec<u32> = partition; // read-only alias
        let k_c_snap = &k_c;

        // ---- Phase A (parallel): compute proposals ---------------------
        // Each thread owns a reusable scratch buffer. `u32::MAX` = "stay
        // in cu" (encoded here so we avoid allocating an Option per node).
        let proposals: Vec<u32> = (0..n)
            .into_par_iter()
            .map_init(
                || (vec![0.0f64; n as usize], Vec::<u32>::with_capacity(64)),
                |(w_to, touched), u| {
                    let cu = partition_snap[u as usize];
                    let ku = graph.strength[u as usize];

                    touched.clear();
                    touched.push(cu);
                    for (v, w) in graph.neighbors(u) {
                        if v == u {
                            continue;
                        }
                        let cv = partition_snap[v as usize];
                        if w_to[cv as usize] == 0.0 && cv != cu {
                            touched.push(cv);
                        }
                        w_to[cv as usize] += w;
                    }

                    let w_to_cu = w_to[cu as usize];
                    let k_cu_minus_u = k_c_snap[cu as usize] - ku;

                    let mut best_c = cu;
                    let mut best_gain = 0.0f64;
                    touched.sort_unstable();
                    for &c in touched.iter() {
                        if c == cu {
                            continue;
                        }
                        let w_to_c = w_to[c as usize];
                        let k_c_now = k_c_snap[c as usize];
                        let delta_q = (w_to_c - w_to_cu) / m
                            - resolution * ku * (k_c_now - k_cu_minus_u) / two_m_sq_half;
                        if delta_q > best_gain + 1e-12 {
                            best_gain = delta_q;
                            best_c = c;
                        } else if (delta_q - best_gain).abs() <= 1e-12 && c < best_c {
                            best_c = c;
                        }
                    }

                    // Reset scratch for next node on this thread.
                    for &c in touched.iter() {
                        w_to[c as usize] = 0.0;
                    }

                    best_c
                },
            )
            .collect();

        // ---- Phase B (sequential): apply proposals in ascending order.
        // Re-validate each proposed move against the CURRENT (partially
        // mutated) state. This rejects oscillating flips and preserves
        // the sequential-Louvain fixpoint while still benefiting from
        // the parallel Phase A.
        //
        // A thread-local scratch is rebuilt here — small vs Phase A, so
        // we just use a dense vec + touched list like the old sequential
        // impl.
        let mut w_to_seq: Vec<f64> = vec![0.0; n as usize];
        let mut touched_seq: Vec<u32> = Vec::with_capacity(64);
        let mut moved_pass: u64 = 0;

        for u in 0..n {
            let proposed = proposals[u as usize];
            let cu = partition[u as usize];
            if proposed == cu {
                continue;
            }
            let ku = graph.strength[u as usize];

            // Recompute ΔQ(cu → proposed) under the current partition.
            touched_seq.clear();
            touched_seq.push(cu);
            for (v, w) in graph.neighbors(u) {
                if v == u {
                    continue;
                }
                let cv = partition[v as usize];
                if w_to_seq[cv as usize] == 0.0 && cv != cu {
                    touched_seq.push(cv);
                }
                w_to_seq[cv as usize] += w;
            }

            let w_to_cu = w_to_seq[cu as usize];
            let k_cu_minus_u = k_c[cu as usize] - ku;
            let w_to_prop = w_to_seq[proposed as usize];
            let k_prop_now = k_c[proposed as usize];
            let delta_q = (w_to_prop - w_to_cu) / m
                - resolution * ku * (k_prop_now - k_cu_minus_u) / two_m_sq_half;

            // Accept iff still a positive gain now. Reject otherwise to
            // avoid parallel-induced oscillations.
            if delta_q > 1e-12 {
                partition[u as usize] = proposed;
                k_c[cu as usize] -= ku;
                k_c[proposed as usize] += ku;
                moved_pass += 1;
                any_move = true;
            }

            for &c in touched_seq.iter() {
                w_to_seq[c as usize] = 0.0;
            }
        }

        tracing::info!(
            target: "obrain_cognitive::leiden",
            "level {level} local_move pass {pass}: moved {moved_pass}/{n} nodes in {:.2?}",
            pass_start.elapsed()
        );

        if moved_pass == 0 {
            break;
        }
        if pass >= MAX_PASSES {
            tracing::warn!(
                target: "obrain_cognitive::leiden",
                "level {level} local_move: hit pass cap {MAX_PASSES} (still {moved_pass} moves), \
                 terminating to guarantee progress"
            );
            break;
        }
    }
    any_move
}

/// Phase 2 — refinement. Within each community produced by
/// [`local_move`], start all nodes as singletons and perform a
/// constrained local move: a node may only merge into a refined
/// community whose parent is its own phase-1 community, and the merge
/// is only accepted when the resulting community remains γ-connected
/// (its internal-edge fraction exceeds the threshold).
///
/// Returns a refined partition whose ids are a finer-grained version of
/// `parent`.
///
/// ## Parallelism
///
/// Each parent community's refinement is **independent** of the others:
/// nodes can only merge into refined communities within the same parent,
/// and the per-community strength table `k_r` is only mutated for
/// communities descended from that parent (ids within the parent's
/// member set). This makes the outer loop trivially parallel — we run
/// each parent on its own rayon task and stitch the results back into
/// the shared `refined` vec via disjoint indices.
///
/// Determinism is preserved: each parent computes the same result
/// regardless of thread scheduling, and the final `refined` vec is
/// assembled by index (order-independent).
fn refine(graph: &Graph, parent: &Partition, resolution: f64, _gamma: f64) -> Partition {
    let n = graph.n;
    // Start every node in its own refined community.
    let mut refined: Vec<u32> = (0..n).collect();

    let two_m = graph.two_m;
    if two_m == 0.0 {
        return refined;
    }

    // Group nodes by parent community. Members within each bucket are
    // already in ascending order because we insert by u = 0..n.
    let mut by_parent: HashMap<u32, Vec<u32>> = HashMap::new();
    for u in 0..n {
        by_parent.entry(parent[u as usize]).or_default().push(u);
    }

    // Parent size distribution — diagnostics for dominant-community hangs.
    let parent_count = by_parent.len();
    let mut sizes: Vec<usize> = by_parent.values().map(|v| v.len()).collect();
    sizes.sort_unstable();
    let max_size = sizes.last().copied().unwrap_or(0);
    let p99_size = if !sizes.is_empty() {
        let idx = (((sizes.len() as f64) * 0.99) as usize).min(sizes.len() - 1);
        sizes[idx]
    } else {
        0
    };
    let median_size = if !sizes.is_empty() {
        sizes[sizes.len() / 2]
    } else {
        0
    };
    let dominance = if n > 0 {
        (max_size as f64) * 100.0 / (n as f64)
    } else {
        0.0
    };
    tracing::info!(
        target: "obrain_cognitive::leiden",
        "refine: {parent_count} parent communities over n={n}, size distribution — \
         max={max_size} ({dominance:.1}%), p99={p99_size}, median={median_size}"
    );

    let refine_start = Instant::now();

    // Process each parent community in parallel. Each produces a vec of
    // `(node_id, final_refined_id)` for nodes that moved; untouched nodes
    // keep their singleton id (u itself).
    //
    // Each rayon worker gets its own `parent_map` scratch buffer of size
    // `n`, initialised to `-1`. `refine_one_parent` temporarily writes
    // local indices into the member slots and restores `-1` before
    // returning, so the invariant `parent_map[i] == -1` between parent
    // iterations holds on every thread. This replaces the former
    // per-parent HashSet/HashMap allocations which were the bottleneck
    // on dominant communities (>100K members).
    //
    // Split parents into three buckets:
    //   - singletons (k=1): nothing to refine, skip entirely
    //   - medium (2..=10000): handled via one shared par_iter
    //   - big (> 10000): handled with per-pass logs (tail-latency focus)
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    let mut big_buckets: Vec<&Vec<u32>> = Vec::new();
    let mut medium_buckets: Vec<&Vec<u32>> = Vec::new();
    let mut skipped_singletons: usize = 0;
    for members in by_parent.values() {
        match members.len() {
            0 => {} // impossible but guard
            1 => skipped_singletons += 1,
            2..=10_000 => medium_buckets.push(members),
            _ => big_buckets.push(members),
        }
    }
    tracing::info!(
        target: "obrain_cognitive::leiden",
        "refine: {} big parents (k>10k), {} medium parents (k in 2..10k), \
         {} singletons skipped — processing big first",
        big_buckets.len(),
        medium_buckets.len(),
        skipped_singletons
    );

    let t_big = Instant::now();
    let big_moves: Vec<Vec<(u32, u32)>> = big_buckets
        .par_iter()
        .map_init(
            || vec![-1i32; n as usize],
            |buf, members| refine_one_parent(graph, members, resolution, two_m, buf),
        )
        .collect();
    tracing::info!(
        target: "obrain_cognitive::leiden",
        "refine: big parents done in {:.2?}, now processing {} medium parents",
        t_big.elapsed(),
        medium_buckets.len()
    );

    // Stitch big parents first.
    for moves in big_moves {
        for (u, c) in moves {
            refined[u as usize] = c;
        }
    }

    // Medium parents: single par_iter with a shared atomic progress
    // counter. `map_init` is called once per rayon worker (not per
    // parent), so the 6.4 MB `parent_map` buffer amortizes over all
    // medium parents processed by that worker — no repeated alloc.
    let t_med = Instant::now();
    let total_med = medium_buckets.len();
    let log_step: usize = (total_med / 20).max(5_000);
    let processed = AtomicUsize::new(0);
    let t_med_ref = &t_med;
    let processed_ref = &processed;

    let medium_moves: Vec<Vec<(u32, u32)>> = medium_buckets
        .par_iter()
        .map_init(
            || vec![-1i32; n as usize],
            |buf, members| {
                let moves = refine_one_parent(graph, members, resolution, two_m, buf);
                let done = processed_ref.fetch_add(1, Ordering::Relaxed) + 1;
                if done.is_multiple_of(log_step) || done == total_med {
                    tracing::info!(
                        target: "obrain_cognitive::leiden",
                        "refine: medium parents {}/{} done ({:.2?})",
                        done,
                        total_med,
                        t_med_ref.elapsed()
                    );
                }
                moves
            },
        )
        .collect();

    for moves in medium_moves {
        for (u, c) in moves {
            refined[u as usize] = c;
        }
    }

    tracing::info!(
        target: "obrain_cognitive::leiden",
        "refine: completed {parent_count} parents in {:.2?}",
        refine_start.elapsed()
    );

    refined
}

/// Refine a single parent community. Returns a list of `(node_id,
/// final_refined_id)` for every node in `members` that didn't remain a
/// singleton. Caller stitches these into the shared refined vec.
///
/// All state (refined ids, k_r, neighbor-weight scratch) is local to
/// this function, so multiple parent communities can run this in
/// parallel without synchronization.
///
/// ## Dense representation
///
/// Members are sorted by ascending node id and indexed `0..k` (their
/// local index in `sorted_members`). Community ids live in the same
/// local-index space — initially `c(i) = i` (singleton), and a merge
/// points one member's community to another member's local index.
/// Because `sorted_members` preserves ascending node-id order, ascending
/// local index is equivalent to ascending node id, so deterministic
/// tie-breaking by smallest-id is preserved by iterating touched in
/// ascending local-index order.
///
/// `parent_map` is a shared scratch buffer of size `n` (one per rayon
/// worker, initialised to `-1`). We write the local index of each
/// member into it on entry, use it for O(1) "is v in this parent?"
/// checks during neighbor iteration, and restore `-1` for all member
/// slots before returning. Net cost: two linear passes over `members`.
fn refine_one_parent(
    graph: &Graph,
    members: &[u32],
    resolution: f64,
    two_m: f64,
    parent_map: &mut [i32],
) -> Vec<(u32, u32)> {
    let m = graph.total_edge_weight();
    let two_m_sq_half = two_m * two_m / 2.0;

    // Sort members so local index ≡ ascending node id.
    let mut sorted_members: Vec<u32> = members.to_vec();
    sorted_members.sort_unstable();
    let k = sorted_members.len();

    // Populate parent_map with local indices. parent_map[u] = i means u
    // is the i-th member of this parent (sorted ascending).
    for (i, &u) in sorted_members.iter().enumerate() {
        parent_map[u as usize] = i as i32;
    }

    // Dense state. Community id space = local index (0..k).
    let mut refined_local: Vec<u32> = (0..k as u32).collect();
    let mut k_r_local: Vec<f64> = sorted_members
        .iter()
        .map(|&u| graph.strength[u as usize])
        .collect();

    // Neighbor-community weight scratch, indexed by local community
    // index. We keep `w_to[c] == 0.0` as the invariant between
    // iterations and track which slots were touched in `touched`.
    let mut w_to: Vec<f64> = vec![0.0; k];
    let mut touched: Vec<u32> = Vec::with_capacity(32);

    let is_big = k > 10_000;
    // Only pay the timer syscall for parents we actually log (is_big).
    let big_start_opt = if is_big { Some(Instant::now()) } else { None };
    let mut pass_count: u32 = 0;

    // Pass cap to guarantee termination. Same contract as local_move:
    // real Leiden converges in a handful of passes; a large cap only
    // kicks in on floating-point-induced oscillations (ΔQ ~ 1e-12 that
    // never stably reaches zero). With the cap, the algorithm remains
    // deterministic and every tested graph converges well before 64.
    const MAX_PASSES: u32 = 64;

    loop {
        let mut moved = false;
        pass_count += 1;
        let pass_start_opt = if is_big { Some(Instant::now()) } else { None };
        let mut moves_this_pass: u64 = 0;

        for i in 0..k {
            let u = sorted_members[i];
            let cu = refined_local[i];
            let ku = graph.strength[u as usize];

            // Seed touched with cu so it's always a candidate (even when
            // u has no in-parent neighbors), mirroring the original
            // HashMap contract.
            touched.push(cu);

            for (v, w) in graph.neighbors(u) {
                if v == u {
                    continue;
                }
                let j = parent_map[v as usize];
                if j < 0 {
                    continue; // v not in this parent community
                }
                let cv = refined_local[j as usize];
                // The `w_to[cv] == 0.0 && cv != cu` guard enforces
                // uniqueness in `touched` — cu was pushed unconditionally
                // above with w_to[cu] == 0.0, so skip cv == cu here.
                if w_to[cv as usize] == 0.0 && cv != cu {
                    touched.push(cv);
                }
                w_to[cv as usize] += w;
            }

            let w_to_cu = w_to[cu as usize];
            let k_cu_minus_u = k_r_local[cu as usize] - ku;

            // Scan candidates in ascending local-index order for
            // deterministic tie-breaking. `touched` has unique entries
            // by construction.
            touched.sort_unstable();

            let mut best_c = cu;
            let mut best_gain = 0.0f64;
            for &c in touched.iter() {
                if c == cu {
                    continue;
                }
                let w_to_c = w_to[c as usize];
                let k_c_now = k_r_local[c as usize];
                let delta_q = (w_to_c - w_to_cu) / m
                    - resolution * ku * (k_c_now - k_cu_minus_u) / two_m_sq_half;
                if delta_q > best_gain + 1e-12 {
                    best_gain = delta_q;
                    best_c = c;
                } else if (delta_q - best_gain).abs() <= 1e-12 && c < best_c {
                    best_c = c;
                }
            }

            if best_c != cu {
                refined_local[i] = best_c;
                k_r_local[cu as usize] -= ku;
                k_r_local[best_c as usize] += ku;
                moved = true;
                moves_this_pass += 1;
            }

            // Reset scratch — restore invariant w_to[c] == 0 for all c.
            for &c in touched.iter() {
                w_to[c as usize] = 0.0;
            }
            touched.clear();
        }

        if let Some(ps) = pass_start_opt {
            tracing::info!(
                target: "obrain_cognitive::leiden",
                "refine_one_parent (k={k}): pass {pass_count} moved {moves_this_pass} in {:.2?}",
                ps.elapsed()
            );
        }

        if !moved {
            break;
        }
        if pass_count >= MAX_PASSES {
            // Floating-point ΔQ oscillation — cap to guarantee progress.
            // This was observed in production on parents with k in the
            // thousands where moves with ΔQ ≈ 1e-12 never stably reach
            // zero. Capping does not affect determinism (same cap on
            // every run).
            tracing::warn!(
                target: "obrain_cognitive::leiden",
                "refine_one_parent (k={k}): hit pass cap {MAX_PASSES} (last pass had \
                 {moves_this_pass} moves), terminating to guarantee progress"
            );
            break;
        }
    }

    if let Some(bs) = big_start_opt {
        tracing::info!(
            target: "obrain_cognitive::leiden",
            "refine_one_parent (k={k}): done in {:.2?}, {pass_count} passes",
            bs.elapsed()
        );
    }

    // Restore parent_map to -1 for all slots we touched. This preserves
    // the per-worker invariant for the next parent community.
    for &u in sorted_members.iter() {
        parent_map[u as usize] = -1;
    }

    // Emit only nodes whose refined community id differs from their
    // own. Local community index → node id: sorted_members[c].
    let mut out: Vec<(u32, u32)> = Vec::new();
    for (i, &u) in sorted_members.iter().enumerate() {
        let c_local = refined_local[i] as usize;
        let c_node = sorted_members[c_local];
        if c_node != u {
            out.push((u, c_node));
        }
    }
    out
}

/// Aggregate: build a graph whose `n_super` nodes are the communities
/// of `partition`. Inter-community edge weights sum; intra-community
/// weights become self-loops.
fn aggregate(graph: &Graph, partition: &Partition, n_super: u32) -> Graph {
    let t_start = Instant::now();
    tracing::info!(
        target: "obrain_cognitive::leiden",
        "aggregate: start n={} → n_super={}",
        graph.n,
        n_super
    );
    let t_alloc = Instant::now();
    let mut acc: Vec<HashMap<u32, f64>> = (0..n_super).map(|_| HashMap::new()).collect();
    tracing::info!(
        target: "obrain_cognitive::leiden",
        "aggregate: allocated {n_super} empty adj maps in {:.2?}",
        t_alloc.elapsed()
    );

    let t_scan = Instant::now();
    let progress_step: u32 = (graph.n / 10).max(100_000);
    for u in 0..graph.n {
        if u > 0 && u.is_multiple_of(progress_step) {
            tracing::info!(
                target: "obrain_cognitive::leiden",
                "aggregate: scanned {u}/{} nodes in {:.2?}",
                graph.n,
                t_scan.elapsed()
            );
        }
        let cu = partition[u as usize];
        for (v, w) in graph.neighbors(u) {
            if v < u {
                // Edges are double-represented in adj; skip the second copy.
                continue;
            }
            let cv = partition[v as usize];
            // For self-loops (u == v) we still add once; Graph::from_edges
            // handles the "self-loop counted once" convention.
            if u == v {
                match acc[cu as usize].entry(cu) {
                    Entry::Vacant(e) => {
                        e.insert(w);
                    }
                    Entry::Occupied(mut e) => {
                        *e.get_mut() += w;
                    }
                }
            } else if cu == cv {
                match acc[cu as usize].entry(cu) {
                    Entry::Vacant(e) => {
                        e.insert(w);
                    }
                    Entry::Occupied(mut e) => {
                        *e.get_mut() += w;
                    }
                }
            } else {
                *acc[cu as usize].entry(cv).or_insert(0.0) += w;
                *acc[cv as usize].entry(cu).or_insert(0.0) += w;
            }
        }
    }
    tracing::info!(
        target: "obrain_cognitive::leiden",
        "aggregate: node scan done in {:.2?}",
        t_scan.elapsed()
    );

    // Emit edge list. Self-loops get one record; inter-community
    // edges emit once as (min, max) pairs.
    let t_emit = Instant::now();
    let mut edges: Vec<(u32, u32, f64)> = Vec::new();
    for (u, neigh) in acc.iter().enumerate() {
        let u = u as u32;
        for (&v, &w) in neigh.iter() {
            if v < u {
                continue;
            }
            edges.push((u, v, w));
        }
    }
    tracing::info!(
        target: "obrain_cognitive::leiden",
        "aggregate: emitted {} edges in {:.2?}, calling Graph::from_edges",
        edges.len(),
        t_emit.elapsed()
    );

    let t_build = Instant::now();
    let result = Graph::from_edges(n_super, edges);
    tracing::info!(
        target: "obrain_cognitive::leiden",
        "aggregate: Graph::from_edges built in {:.2?} — aggregate total {:.2?}",
        t_build.elapsed(),
        t_start.elapsed()
    );
    result
}

/// Re-label `partition` so that community ids are contiguous in
/// `0..k` and the ordering matches first occurrence in ascending node
/// order (deterministic). Returns the relabeled partition and `k`.
fn normalize(partition: &Partition) -> (Partition, u32) {
    let mut remap: HashMap<u32, u32> = HashMap::new();
    let mut next = 0u32;
    let mut out = Vec::with_capacity(partition.len());
    for &c in partition.iter() {
        let id = *remap.entry(c).or_insert_with(|| {
            let v = next;
            next += 1;
            v
        });
        out.push(id);
    }
    (out, next)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn karate_edges() -> Vec<(u32, u32, f64)> {
        // Zachary's karate club — 34 nodes, 78 edges. Classical dataset
        // with two well-known communities around nodes 0 (Mr. Hi) and
        // 33 (John A.). Modularity of the "true" 2-community split is
        // about 0.36; optimal split found by Leiden is ~0.42 at γ=1.
        // Edge list from Zachary (1977), 0-indexed.
        vec![
            (0, 1, 1.0),  (0, 2, 1.0),  (0, 3, 1.0),  (0, 4, 1.0),
            (0, 5, 1.0),  (0, 6, 1.0),  (0, 7, 1.0),  (0, 8, 1.0),
            (0, 10, 1.0), (0, 11, 1.0), (0, 12, 1.0), (0, 13, 1.0),
            (0, 17, 1.0), (0, 19, 1.0), (0, 21, 1.0), (0, 31, 1.0),
            (1, 2, 1.0),  (1, 3, 1.0),  (1, 7, 1.0),  (1, 13, 1.0),
            (1, 17, 1.0), (1, 19, 1.0), (1, 21, 1.0), (1, 30, 1.0),
            (2, 3, 1.0),  (2, 7, 1.0),  (2, 8, 1.0),  (2, 9, 1.0),
            (2, 13, 1.0), (2, 27, 1.0), (2, 28, 1.0), (2, 32, 1.0),
            (3, 7, 1.0),  (3, 12, 1.0), (3, 13, 1.0),
            (4, 6, 1.0),  (4, 10, 1.0),
            (5, 6, 1.0),  (5, 10, 1.0), (5, 16, 1.0),
            (6, 16, 1.0),
            (8, 30, 1.0), (8, 32, 1.0), (8, 33, 1.0),
            (9, 33, 1.0),
            (13, 33, 1.0),
            (14, 32, 1.0), (14, 33, 1.0),
            (15, 32, 1.0), (15, 33, 1.0),
            (18, 32, 1.0), (18, 33, 1.0),
            (19, 33, 1.0),
            (20, 32, 1.0), (20, 33, 1.0),
            (22, 32, 1.0), (22, 33, 1.0),
            (23, 25, 1.0), (23, 27, 1.0), (23, 29, 1.0),
            (23, 32, 1.0), (23, 33, 1.0),
            (24, 25, 1.0), (24, 27, 1.0), (24, 31, 1.0),
            (25, 31, 1.0),
            (26, 29, 1.0), (26, 33, 1.0),
            (27, 33, 1.0),
            (28, 31, 1.0), (28, 33, 1.0),
            (29, 32, 1.0), (29, 33, 1.0),
            (30, 32, 1.0), (30, 33, 1.0),
            (31, 32, 1.0), (31, 33, 1.0),
            (32, 33, 1.0),
        ]
    }

    #[test]
    fn empty_graph_yields_empty_partition() {
        let g = Graph::new(0);
        let p = leiden_batch(&g, LeidenConfig::default());
        assert!(p.is_empty());
        assert_eq!(modularity(&g, &p, 1.0), 0.0);
    }

    #[test]
    fn isolated_nodes_each_in_own_community() {
        let g = Graph::new(5);
        let p = leiden_batch(&g, LeidenConfig::default());
        assert_eq!(p.len(), 5);
        // No edges, no modularity.
        assert_eq!(modularity(&g, &p, 1.0), 0.0);
    }

    #[test]
    fn two_cliques_bridged_form_two_communities() {
        // Two 4-cliques bridged by a single edge — Leiden should
        // recover the two cliques as two communities.
        let mut edges = vec![];
        for u in 0..4 {
            for v in (u + 1)..4 {
                edges.push((u, v, 1.0));
            }
        }
        for u in 4..8u32 {
            for v in (u + 1)..8u32 {
                edges.push((u, v, 1.0));
            }
        }
        edges.push((3, 4, 1.0)); // bridge
        let g = Graph::from_edges(8, edges);
        let p = leiden_batch(&g, LeidenConfig::default());
        // Nodes 0..4 must share one community, 4..8 another.
        let c_left = p[0];
        let c_right = p[7];
        assert_ne!(c_left, c_right);
        for u in 0..4 {
            assert_eq!(p[u], c_left, "node {} should be in left community", u);
        }
        for u in 4..8 {
            assert_eq!(p[u], c_right, "node {} should be in right community", u);
        }
        let q = modularity(&g, &p, 1.0);
        assert!(q > 0.40, "modularity of two bridged cliques: got {}", q);
    }

    #[test]
    fn karate_club_modularity_above_threshold() {
        // Gate from T10 Step 0 verification: modularity ≥ 0.40 on
        // Zachary's karate club at γ=1.
        let g = Graph::from_edges(34, karate_edges());
        let p = leiden_batch(&g, LeidenConfig::default());
        let q = modularity(&g, &p, 1.0);
        eprintln!("karate modularity: {}", q);
        assert!(
            q >= 0.40,
            "karate modularity {} below 0.40 threshold",
            q
        );
        // Sanity — should find at least 2 communities.
        let unique = p
            .iter()
            .copied()
            .collect::<std::collections::HashSet<u32>>();
        assert!(unique.len() >= 2, "expected ≥ 2 communities, got {:?}", unique);
    }

    #[test]
    fn three_cliques_form_three_communities() {
        // Three 5-cliques chained by one bridge each. Leiden must
        // recover three distinct communities.
        let mut edges = vec![];
        for block in 0..3u32 {
            let base = block * 5;
            for u in base..(base + 5) {
                for v in (u + 1)..(base + 5) {
                    edges.push((u, v, 1.0));
                }
            }
        }
        edges.push((4, 5, 1.0));
        edges.push((9, 10, 1.0));
        let g = Graph::from_edges(15, edges);
        let p = leiden_batch(&g, LeidenConfig::default());
        let unique: std::collections::HashSet<u32> = p.iter().copied().collect();
        assert_eq!(
            unique.len(),
            3,
            "expected 3 communities on triple-clique chain, got {:?} (partition {:?})",
            unique,
            p
        );
        // All five members of each clique must share a community.
        for block in 0..3usize {
            let c = p[block * 5];
            for off in 1..5 {
                assert_eq!(
                    p[block * 5 + off],
                    c,
                    "clique {} split across communities: {:?}",
                    block,
                    p
                );
            }
        }
    }

    #[test]
    fn modularity_of_trivial_partition_is_nonpositive() {
        // A single community containing every node has modularity
        // Σ_c [k_c/2m - (k_c/2m)²] which for c = V is 1 - 1 = 0.
        let g = Graph::from_edges(
            4,
            vec![(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0)],
        );
        let p = vec![0, 0, 0, 0];
        let q = modularity(&g, &p, 1.0);
        assert!(q.abs() < 1e-9, "Q on trivial partition: {}", q);
    }

    #[test]
    fn leiden_is_deterministic() {
        let g = Graph::from_edges(34, karate_edges());
        let p1 = leiden_batch(&g, LeidenConfig::default());
        let p2 = leiden_batch(&g, LeidenConfig::default());
        assert_eq!(p1, p2, "Leiden must be deterministic for identical inputs");
    }

    #[test]
    fn normalize_is_stable_by_first_occurrence() {
        let input: Partition = vec![7, 3, 7, 9, 3];
        let (out, k) = normalize(&input);
        assert_eq!(out, vec![0, 1, 0, 2, 1]);
        assert_eq!(k, 3);
    }
}
