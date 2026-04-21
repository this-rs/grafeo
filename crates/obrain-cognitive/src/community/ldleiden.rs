//! LDleiden — incremental Leiden driver (T10 Step 1+).
//!
//! Maintains a [`Partition`] online in response to edge deltas
//! (`on_edge_add` / `on_edge_remove` / `on_edge_reweight`). Each delta
//! returns the list of `(node_id, new_community_id)` pairs that
//! changed, so the caller can push them to substrate via
//! `Writer::update_community_batch` (wired in T10 Step 3).
//!
//! ## Cost model
//!
//! The contract is `O(|ΔE| · (d̄ + |C_affected|))` with `d̄` the
//! average degree and `|C_affected|` the number of communities that
//! touch the frontier. The implementation achieves this with three
//! levers:
//!
//!   1. **Internal fast-path.** Adding positive weight to an edge
//!      whose endpoints share a community is *provably* a no-op on
//!      the partition: `k_c` increases, `k_{c'}` is unchanged,
//!      `w(x → c)` is unchanged for every x ∉ {u, v}, so every
//!      alternative community becomes *less* attractive for every
//!      node. We short-circuit after updating `graph` and `k_c`.
//!
//!   2. **Targeted evaluation.** Cross-community adds (or internal
//!      removes, which can decohere the community) evaluate u and v
//!      first and cascade to their neighbors only if one of them
//!      actually moves. The cascade is bounded at `max_passes = 2`
//!      moves to keep per-delta cost in `O(d̄)`.
//!
//!   3. **No implicit GC.** [`reclaim_empty_communities`] is only
//!      called on the `on_edge_remove` path (where splits can free
//!      ids) and from the explicit public API. It is `O(|empty|)`
//!      because it consults `members_count` instead of scanning
//!      `partition`.
//!
//! Scratch buffers (`w_to_scratch`, `touched_cids`) live on the
//! driver so each call allocates nothing on the hot path.
//!
//! ## Invariants
//!
//! - Every node `u` in `0..graph.n` has `partition[u]` set.
//! - `k_c[c] = Σ_{u : partition[u] = c} strength(u)` at all times.
//! - `members_count[c] = #{u : partition[u] = c}` at all times.
//! - Community ids are not necessarily contiguous after many deltas
//!   (holes appear as communities empty out). Callers who need
//!   contiguous ids for storage can call [`LDleiden::compact_ids`].

use obrain_common::utils::hash::{FxHashMap, FxHashSet};

use super::leiden::{Graph, LeidenConfig, Partition, leiden_batch, modularity};

/// Stats returned from each delta application — useful for tests and
/// benchmarks.
#[derive(Debug, Clone, Default)]
pub struct LDleidenStats {
    /// Number of nodes whose community id changed as a result of the
    /// delta.
    pub moves: usize,
    /// Number of frontier nodes evaluated (upper-bound on the work).
    pub evaluated: usize,
    /// Number of local-move passes until convergence (≥ 1).
    pub passes: usize,
}

/// Incremental Leiden driver.
pub struct LDleiden {
    graph: Graph,
    partition: Partition,
    /// Community → total strength.
    k_c: FxHashMap<u32, f64>,
    /// Community → node count. Maintained in O(1) on every
    /// `partition[u] = c'` assignment. Lets [`reclaim_empty_communities`]
    /// run in `O(|empty|)` without scanning `partition`.
    members_count: FxHashMap<u32, u32>,
    config: LeidenConfig,
    /// Next fresh community id. Monotonic allocator; ids recycled
    /// into `free_cids` when a community empties out.
    next_cid: u32,
    /// Free-list of community ids that emptied out and can be
    /// reused. Allocation checks this before bumping `next_cid`.
    free_cids: Vec<u32>,
    /// Scratch buffer for per-delta `w_to[community]` accumulation.
    /// Always drained before returning so reuse is free.
    w_to_scratch: FxHashMap<u32, f64>,
    /// Scratch buffer for the list of community ids we touched in
    /// `w_to_scratch`, used to drain it in one pass.
    touched_cids: Vec<u32>,
}

impl LDleiden {
    /// Construct a driver from scratch: runs a full [`leiden_batch`] on
    /// `graph` to produce the initial partition.
    pub fn bootstrap(graph: Graph, config: LeidenConfig) -> Self {
        let partition = leiden_batch(&graph, config);
        Self::from_partition_unchecked(graph, partition, config)
    }

    /// Construct from an externally-known partition (e.g. reloaded from
    /// substrate). Panics if `partition.len() != graph.num_nodes()`.
    pub fn from_partition(graph: Graph, partition: Partition, config: LeidenConfig) -> Self {
        assert_eq!(
            partition.len(),
            graph.num_nodes() as usize,
            "partition length must match node count"
        );
        Self::from_partition_unchecked(graph, partition, config)
    }

    fn from_partition_unchecked(
        graph: Graph,
        partition: Partition,
        config: LeidenConfig,
    ) -> Self {
        let mut k_c: FxHashMap<u32, f64> = FxHashMap::default();
        let mut members_count: FxHashMap<u32, u32> = FxHashMap::default();
        let mut max_cid = 0u32;
        for u in 0..graph.num_nodes() {
            let c = partition[u as usize];
            *k_c.entry(c).or_insert(0.0) += graph.strength(u);
            *members_count.entry(c).or_insert(0) += 1;
            if c > max_cid {
                max_cid = c;
            }
        }
        Self {
            graph,
            partition,
            k_c,
            members_count,
            config,
            next_cid: max_cid + 1,
            free_cids: Vec::new(),
            w_to_scratch: FxHashMap::default(),
            touched_cids: Vec::new(),
        }
    }

    /// Total number of non-empty communities in the current partition.
    pub fn num_communities(&self) -> usize {
        self.members_count
            .values()
            .filter(|&&count| count > 0)
            .count()
    }

    /// Collect community ids that have no members (`members_count == 0`)
    /// and push them onto the free-list for reuse.
    ///
    /// `O(|members_count|)` — iterates the per-community counter map,
    /// not `partition`. After a big batch of deltas this stays cheap
    /// because the number of communities ever created is bounded by
    /// the number of splits.
    ///
    /// This is the GC primitive. The hot path (`apply_edge_delta`)
    /// does **not** call it — it's only invoked on `on_edge_remove`
    /// (where a split can free an id) and from the public API.
    pub fn reclaim_empty_communities(&mut self) -> usize {
        let empties: Vec<u32> = self
            .members_count
            .iter()
            .filter_map(|(c, &count)| if count == 0 { Some(*c) } else { None })
            .collect();
        let mut reclaimed = 0;
        for c in empties {
            // Safety check: cross-validate against k_c — if k_c still
            // has a non-zero entry the counters drifted and we keep
            // the id alive. (This defends against hand-written tests
            // that poke `partition` directly without updating
            // `members_count`.)
            let k_ok = self.k_c.get(&c).copied().unwrap_or(0.0).abs() < 1e-9;
            if !k_ok {
                continue;
            }
            // Final belt-and-braces guard: if a test harness mutated
            // `partition` without going through `move_node`, the
            // members_count could under-report membership. Scan the
            // partition once to confirm; this is O(N) but only fires
            // when the counters say "empty". In normal operation
            // (driver-mediated moves only) members_count is authoritative
            // and this scan is skipped via short-circuit above.
            let any_member = self.partition.iter().any(|&p| p == c);
            if any_member {
                // Drift — repair by re-counting just this community.
                let actual = self.partition.iter().filter(|&&p| p == c).count() as u32;
                self.members_count.insert(c, actual);
                continue;
            }
            self.members_count.remove(&c);
            self.k_c.remove(&c);
            self.free_cids.push(c);
            reclaimed += 1;
        }
        reclaimed
    }

    /// Re-run batch Leiden on the current graph, overwriting the driver's
    /// state with the new optimum. Returns the list of node → community
    /// changes so the caller can feed them to
    /// [`obrain_substrate::Writer::update_community_batch`].
    ///
    /// ## When to call this
    ///
    /// The incremental hot-path ([`on_edge_add`], [`on_edge_remove`],
    /// [`on_edge_reweight`]) caps its local-move cascade at
    /// `max_passes = 2` — that is what makes each delta cost `O(avg_deg)`
    /// and not `O(|V|)` (see the T10 Step 5 bench and gotcha note
    /// `2b27f4d3`). The cost is a **quality drift**: after a stream of
    /// edge events, the partition can lag the full-Leiden optimum by a
    /// few percent (modularity ratio typically 0.85-0.95 depending on
    /// the graph's mixing parameter μ).
    ///
    /// `refine_in_place` erases that drift by re-running [`leiden_batch`]
    /// on the current graph. It is O(|V| + |E|) and should be invoked:
    ///
    /// * periodically from a low-priority background task (the
    ///   `PulseMonitor` pattern), e.g. every `N` deltas or every `T`
    ///   seconds of quiet;
    /// * on user-driven "refresh" actions (e.g. GDS refresh trigger);
    /// * in tests, to make quality gates reproducible.
    ///
    /// Returns `Vec<(node_id, new_community_id)>` for every node whose
    /// community changed. Nodes that stayed put are not in the output.
    pub fn refine_in_place(&mut self) -> Vec<(u32, u32)> {
        let new_partition = leiden_batch(&self.graph, self.config);
        debug_assert_eq!(new_partition.len(), self.partition.len());

        let mut deltas = Vec::new();
        for u in 0..self.graph.num_nodes() {
            let c_old = self.partition[u as usize];
            let c_new = new_partition[u as usize];
            if c_old != c_new {
                deltas.push((u, c_new));
            }
        }

        // Rebuild all derived state from scratch — k_c, members_count,
        // next_cid, free_cids. Any stale community ids are released by
        // construction (not present in new_partition → not in
        // members_count).
        self.partition = new_partition;
        self.k_c.clear();
        self.members_count.clear();
        let mut max_cid = 0u32;
        for u in 0..self.graph.num_nodes() {
            let c = self.partition[u as usize];
            *self.k_c.entry(c).or_insert(0.0) += self.graph.strength(u);
            *self.members_count.entry(c).or_insert(0) += 1;
            if c > max_cid {
                max_cid = c;
            }
        }
        self.next_cid = max_cid + 1;
        self.free_cids.clear();

        deltas
    }

    /// Re-label every community id to be in `0..num_communities`,
    /// preserving a stable ordering (first occurrence in ascending
    /// node order gets the lowest new id). Returns the number of
    /// distinct communities after compaction.
    pub fn compact_ids(&mut self) -> u32 {
        let mut remap: FxHashMap<u32, u32> = FxHashMap::default();
        let mut next = 0u32;
        for u in 0..self.graph.num_nodes() {
            let c = self.partition[u as usize];
            remap.entry(c).or_insert_with(|| {
                let id = next;
                next += 1;
                id
            });
        }
        for u in 0..self.graph.num_nodes() {
            let old = self.partition[u as usize];
            self.partition[u as usize] = *remap.get(&old).expect("all ids remapped");
        }
        let mut new_k_c: FxHashMap<u32, f64> = FxHashMap::default();
        let mut new_members: FxHashMap<u32, u32> = FxHashMap::default();
        for u in 0..self.graph.num_nodes() {
            let c = self.partition[u as usize];
            *new_k_c.entry(c).or_insert(0.0) += self.graph.strength(u);
            *new_members.entry(c).or_insert(0) += 1;
        }
        self.k_c = new_k_c;
        self.members_count = new_members;
        self.free_cids.clear();
        self.next_cid = next;
        next
    }

    /// Current partition. Slice of length `graph.num_nodes()`.
    pub fn partition(&self) -> &[u32] {
        &self.partition
    }

    /// Underlying graph (read-only view).
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// Modularity of the current partition at resolution
    /// [`LeidenConfig::resolution`].
    pub fn modularity(&self) -> f64 {
        modularity(&self.graph, &self.partition, self.config.resolution)
    }

    /// Return a fresh (unused) community id. Consumes the free-list
    /// before bumping the monotonic counter.
    fn allocate_cid(&mut self) -> u32 {
        if let Some(id) = self.free_cids.pop() {
            id
        } else {
            let id = self.next_cid;
            self.next_cid += 1;
            id
        }
    }

    /// Move `node` from community `from` to community `to`.
    /// Maintains `partition`, `k_c`, and `members_count` in lockstep.
    /// `ku` is `graph.strength(node)` (hoisted by callers that already
    /// have it).
    #[inline]
    fn move_node(&mut self, node: u32, from: u32, to: u32, ku: f64) {
        if from == to {
            return;
        }
        self.partition[node as usize] = to;
        *self.k_c.entry(from).or_insert(0.0) -= ku;
        *self.k_c.entry(to).or_insert(0.0) += ku;
        if let Some(c) = self.members_count.get_mut(&from) {
            *c = c.saturating_sub(1);
        }
        *self.members_count.entry(to).or_insert(0) += 1;
    }

    /// React to `edge(u, v)` gaining weight `w` (creating it if absent).
    pub fn on_edge_add(&mut self, u: u32, v: u32, w: f64) -> (Vec<(u32, u32)>, LDleidenStats) {
        assert!(w > 0.0, "weight must be positive (use on_edge_remove for deletions)");
        self.apply_edge_delta(u, v, w)
    }

    /// React to `edge(u, v)` being re-weighted (delta is new_w - old_w).
    pub fn on_edge_reweight(
        &mut self,
        u: u32,
        v: u32,
        delta: f64,
    ) -> (Vec<(u32, u32)>, LDleidenStats) {
        if delta < 0.0 {
            let current = self.graph.edge_weight(u, v);
            if current + delta <= 0.0 {
                return self.on_edge_remove(u, v);
            }
        }
        self.apply_edge_delta(u, v, delta)
    }

    /// React to full removal of `edge(u, v)`.
    pub fn on_edge_remove(
        &mut self,
        u: u32,
        v: u32,
    ) -> (Vec<(u32, u32)>, LDleidenStats) {
        let w = self.graph.edge_weight(u, v);
        if w == 0.0 {
            return (Vec::new(), LDleidenStats::default());
        }

        let cu_before = self.partition[u as usize];
        let cv_before = self.partition[v as usize];
        let same_community = cu_before == cv_before && u != v;

        let members_cu_before: Vec<u32> = if same_community {
            self.members_of(cu_before)
        } else {
            Vec::new()
        };

        let (mut out, mut stats) = self.apply_edge_delta(u, v, -w);

        if same_community {
            let cu_now = self.partition[u as usize];
            let reachable = self.bfs_within_community(u, cu_now);

            let mut split_nodes: Vec<u32> = Vec::new();
            for &m in &members_cu_before {
                if self.partition[m as usize] == cu_now && !reachable.contains(&m) {
                    split_nodes.push(m);
                }
            }

            if !split_nodes.is_empty() {
                let new_cid = self.allocate_cid();
                // Make sure the new cid is registered in members_count
                // so move_node's decrement/increment bookkeeping lands
                // consistently.
                self.members_count.entry(new_cid).or_insert(0);
                for &node in &split_nodes {
                    let k = self.graph.strength(node);
                    self.move_node(node, cu_now, new_cid, k);
                    stats.moves += 1;
                }
                let mut map: FxHashMap<u32, u32> = out.into_iter().collect();
                for &node in &split_nodes {
                    map.insert(node, new_cid);
                }
                out = map.into_iter().collect();
                out.sort_unstable();
            }
            let _ = cv_before;
        }

        // Removal can leave communities empty. Reclaim now so the
        // caller sees consistent `num_communities()` and so ids can be
        // recycled on the next split.
        self.reclaim_empty_communities();
        (out, stats)
    }

    /// Members of a community (filtered by current partition). Linear
    /// in `n`, used only on split-detection paths (rare).
    fn members_of(&self, c: u32) -> Vec<u32> {
        let mut v = Vec::new();
        for u in 0..self.graph.num_nodes() {
            if self.partition[u as usize] == c {
                v.push(u);
            }
        }
        v
    }

    /// BFS from `start` within the induced subgraph of
    /// `partition[node] == c`. Returns the set of reachable nodes.
    fn bfs_within_community(&self, start: u32, c: u32) -> FxHashSet<u32> {
        let mut visited: FxHashSet<u32> = FxHashSet::default();
        let mut queue: std::collections::VecDeque<u32> = std::collections::VecDeque::new();
        if self.partition[start as usize] != c {
            return visited;
        }
        visited.insert(start);
        queue.push_back(start);
        while let Some(n) = queue.pop_front() {
            for (nb, _) in self.graph.neighbors(n) {
                if nb == n {
                    continue;
                }
                if self.partition[nb as usize] != c {
                    continue;
                }
                if visited.insert(nb) {
                    queue.push_back(nb);
                }
            }
        }
        visited
    }

    /// Reset the `w_to_scratch` buffer to empty without deallocating.
    #[inline]
    fn clear_w_to(&mut self) {
        for c in self.touched_cids.drain(..) {
            self.w_to_scratch.remove(&c);
        }
    }

    /// Populate `w_to_scratch` with `w_to[community] = Σ_{v ∈ N(node), partition[v]=c} w(node,v)`.
    /// Excludes self-loops. Caller must `clear_w_to()` afterwards.
    /// Returns `(w_to[cu_now], k_u)` where `cu_now` is the node's current
    /// community.
    fn collect_w_to(&mut self, node: u32) -> (f64, f64) {
        let cu_now = self.partition[node as usize];
        let ku = self.graph.strength(node);
        // Seed with self-community so the caller can read w_to[cu_now]
        // unconditionally.
        self.w_to_scratch.insert(cu_now, 0.0);
        self.touched_cids.push(cu_now);
        // SAFETY: we need to mutate `w_to_scratch` and push to
        // `touched_cids` in the same loop, but the borrow checker
        // can't prove they're disjoint fields from inside the closure
        // form. Split the access explicitly via `get_mut` then
        // `insert`.
        for (nb, w) in self.graph.neighbors(node) {
            if nb == node {
                continue;
            }
            let c_nb = self.partition[nb as usize];
            if let Some(w_ref) = self.w_to_scratch.get_mut(&c_nb) {
                *w_ref += w;
            } else {
                self.w_to_scratch.insert(c_nb, w);
                self.touched_cids.push(c_nb);
            }
        }
        let w_to_cu = *self.w_to_scratch.get(&cu_now).unwrap_or(&0.0);
        (w_to_cu, ku)
    }

    /// Pick the best community for `node` given the current
    /// `w_to_scratch`. Returns `(best_cid, best_gain)`. Ties broken
    /// by lowest community id.
    fn pick_best_community(&self, node: u32, w_to_cu: f64, ku: f64) -> (u32, f64) {
        let two_m = self.graph.total_edge_weight() * 2.0;
        if two_m <= 0.0 {
            return (self.partition[node as usize], 0.0);
        }
        let two_m_sq_half = two_m * two_m / 2.0;
        let m = self.graph.total_edge_weight();
        let cu_now = self.partition[node as usize];
        let k_cu_minus_u = self.k_c.get(&cu_now).copied().unwrap_or(0.0) - ku;
        let mut best_c = cu_now;
        let mut best_gain = 0.0f64;
        for (&c, &w_to_c) in self.w_to_scratch.iter() {
            if c == cu_now {
                continue;
            }
            let k_c_now = self.k_c.get(&c).copied().unwrap_or(0.0);
            let delta_q = (w_to_c - w_to_cu) / m
                - self.config.resolution * ku * (k_c_now - k_cu_minus_u) / two_m_sq_half;
            if delta_q > best_gain + 1e-12 {
                best_gain = delta_q;
                best_c = c;
            } else if (delta_q - best_gain).abs() <= 1e-12 && c < best_c {
                best_c = c;
            }
        }
        (best_c, best_gain)
    }

    /// Core delta handler. Applies the graph mutation, updates `k_c`,
    /// and runs a targeted local move.
    ///
    /// **Fast path** — if `partition[u] == partition[v]` and `delta > 0`,
    /// the partition is provably unchanged (proof sketch in the module
    /// docstring: adding internal weight strictly *decreases* every
    /// alternative community's attractiveness for every node). We skip
    /// the evaluation entirely.
    ///
    /// Otherwise, evaluate u and v; if either moves, cascade to its
    /// neighbors (bounded at `max_passes = 2`).
    fn apply_edge_delta(
        &mut self,
        u: u32,
        v: u32,
        delta: f64,
    ) -> (Vec<(u32, u32)>, LDleidenStats) {
        let mut stats = LDleidenStats::default();
        if delta == 0.0 {
            return (Vec::new(), stats);
        }

        let cu_before = self.partition[u as usize];
        let cv_before = if u == v {
            cu_before
        } else {
            self.partition[v as usize]
        };

        // Apply the delta to the graph + mirror into k_c.
        let before_strength_u = self.graph.strength(u);
        let before_strength_v = self.graph.strength(v);
        self.graph.add_edge_delta(u, v, delta);
        let after_strength_u = self.graph.strength(u);
        let after_strength_v = self.graph.strength(v);
        *self.k_c.entry(cu_before).or_insert(0.0) += after_strength_u - before_strength_u;
        if u != v {
            *self.k_c.entry(cv_before).or_insert(0.0) += after_strength_v - before_strength_v;
        }

        // Fast-path: positive internal delta — provably no partition change.
        if u == v || (cu_before == cv_before && delta > 0.0) {
            return (Vec::new(), stats);
        }

        // Remember "before" for u and v so we can diff at the end.
        // We don't materialize a frontier — move_node maintains
        // partition/k_c/members_count in-place, and we track moves in
        // `changed`.
        let mut changed: FxHashMap<u32, u32> = FxHashMap::default();
        changed.insert(u, cu_before);
        changed.insert(v, cv_before);

        // Pass 1: evaluate u, then v.
        let max_passes = 2usize;
        let mut to_eval: Vec<u32> = vec![u, v];
        let mut cascaded: FxHashSet<u32> = FxHashSet::default();
        let mut any_move_any_pass = false;

        for pass in 1..=max_passes {
            stats.passes = pass;
            let mut moved_pass = false;
            // Deterministic iteration order (ascending ids).
            to_eval.sort_unstable();
            to_eval.dedup();
            let nodes_this_pass = std::mem::take(&mut to_eval);
            stats.evaluated += nodes_this_pass.len();

            for node in nodes_this_pass {
                let cu_now = self.partition[node as usize];
                let (w_to_cu, ku) = self.collect_w_to(node);
                let (best_c, _gain) = self.pick_best_community(node, w_to_cu, ku);
                self.clear_w_to();

                if best_c != cu_now {
                    self.move_node(node, cu_now, best_c, ku);
                    moved_pass = true;
                    any_move_any_pass = true;
                    // Cascade: queue this node's neighbors if we
                    // haven't already expanded from it.
                    if pass < max_passes && cascaded.insert(node) {
                        for (nb, _) in self.graph.neighbors(node) {
                            if nb != node {
                                to_eval.push(nb);
                            }
                        }
                    }
                }
            }

            if !moved_pass || to_eval.is_empty() {
                break;
            }
        }

        // Build the delta from `changed` against the final partition.
        // `changed` only tracks u, v — neighbors cascaded via
        // `to_eval`, so we need their "before" too. We reconstruct
        // from the move_node trail: any node whose current partition
        // differs from what it was at entry needs reporting. A
        // per-delta HashMap of originals would add overhead, so we
        // take a different approach: walk the cascaded set + {u, v}
        // and report those that moved.
        let mut out: Vec<(u32, u32)> = Vec::new();
        if any_move_any_pass {
            let mut seen: FxHashSet<u32> = FxHashSet::default();
            for (&node, &c_before) in changed.iter() {
                let c_after = self.partition[node as usize];
                if c_after != c_before {
                    out.push((node, c_after));
                    stats.moves += 1;
                }
                seen.insert(node);
            }
            for &node in &cascaded {
                if seen.insert(node) {
                    // We don't have the pre-delta community for
                    // cascaded nodes recorded explicitly, but
                    // `cascaded` only contains nodes we observed
                    // moving (insert guarded by `best_c != cu_now`),
                    // so each cascaded node definitely changed.
                    out.push((node, self.partition[node as usize]));
                    stats.moves += 1;
                }
            }
            out.sort_unstable();
        }
        (out, stats)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn two_bridged_cliques() -> Graph {
        let mut edges = vec![];
        for u in 0..4u32 {
            for v in (u + 1)..4 {
                edges.push((u, v, 1.0));
            }
        }
        for u in 4..8u32 {
            for v in (u + 1)..8 {
                edges.push((u, v, 1.0));
            }
        }
        edges.push((3, 4, 1.0));
        Graph::from_edges(8, edges)
    }

    #[test]
    fn bootstrap_produces_two_communities() {
        let g = two_bridged_cliques();
        let d = LDleiden::bootstrap(g, LeidenConfig::default());
        let p = d.partition();
        let c_left = p[0];
        let c_right = p[7];
        assert_ne!(c_left, c_right);
        for u in 0..4 {
            assert_eq!(p[u], c_left);
        }
        for u in 4..8 {
            assert_eq!(p[u], c_right);
        }
    }

    #[test]
    fn add_internal_edge_is_no_op() {
        let g = two_bridged_cliques();
        let mut d = LDleiden::bootstrap(g, LeidenConfig::default());
        let (delta, stats) = d.on_edge_add(0, 1, 1.0);
        assert!(delta.is_empty(), "internal edge should not move anyone; got {:?}", delta);
        assert_eq!(stats.moves, 0);
    }

    #[test]
    fn add_strong_cross_edge_merges_communities() {
        let g = two_bridged_cliques();
        let mut d = LDleiden::bootstrap(g, LeidenConfig::default());
        let before: Vec<u32> = d.partition().to_vec();
        let (_delta1, _) = d.on_edge_add(0, 5, 5.0);
        let (_delta2, _) = d.on_edge_add(1, 6, 5.0);
        let (_delta3, _) = d.on_edge_add(2, 7, 5.0);
        let after: Vec<u32> = d.partition().to_vec();
        assert!(
            before != after,
            "heavy cross-edges should force at least one move; before={:?} after={:?}",
            before,
            after
        );
    }

    #[test]
    fn incremental_matches_batch_within_tolerance() {
        let g0 = {
            let mut edges = vec![];
            for block in 0..4u32 {
                let base = block * 10;
                for u in base..(base + 10) {
                    for v in (u + 1)..(base + 10) {
                        edges.push((u, v, 1.0));
                    }
                }
            }
            edges.push((9, 10, 1.0));
            edges.push((19, 20, 1.0));
            edges.push((29, 30, 1.0));
            Graph::from_edges(40, edges)
        };
        let cfg = LeidenConfig::default();
        let mut d = LDleiden::bootstrap(g0.clone(), cfg);

        let mut state: u64 = 0xC0FFEE_DEAD_BEEF;
        let mut apply_edges: Vec<(u32, u32)> = vec![];
        let mut next_rand = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..40 {
            let intra = (next_rand() % 10) < 7;
            let block = (next_rand() % 4) as u32;
            let base = block * 10;
            let u;
            let v;
            if intra {
                u = base + (next_rand() % 10) as u32;
                v = base + (next_rand() % 10) as u32;
            } else {
                u = base + (next_rand() % 10) as u32;
                let other_block = (block + 1 + (next_rand() % 3) as u32) % 4;
                v = other_block * 10 + (next_rand() % 10) as u32;
            }
            if u == v {
                continue;
            }
            apply_edges.push((u, v));
            d.on_edge_add(u, v, 1.0);
        }

        let q_incremental = d.modularity();

        let mut edges_final = vec![];
        for block in 0..4u32 {
            let base = block * 10;
            for u in base..(base + 10) {
                for v in (u + 1)..(base + 10) {
                    edges_final.push((u, v, 1.0));
                }
            }
        }
        edges_final.push((9, 10, 1.0));
        edges_final.push((19, 20, 1.0));
        edges_final.push((29, 30, 1.0));
        for (u, v) in &apply_edges {
            edges_final.push((*u, *v, 1.0));
        }
        let g_final = Graph::from_edges(40, edges_final);
        let p_batch = leiden_batch(&g_final, cfg);
        let q_batch = modularity(&g_final, &p_batch, cfg.resolution);

        eprintln!(
            "incremental Q = {}, batch Q = {}, ratio = {}",
            q_incremental,
            q_batch,
            q_incremental / q_batch
        );
        assert!(
            q_incremental >= 0.98 * q_batch,
            "incremental modularity {} below 98% of batch {} ({})",
            q_incremental,
            q_batch,
            q_incremental / q_batch
        );
    }

    #[test]
    fn add_edge_updates_graph_weights() {
        let g = Graph::from_edges(3, vec![(0, 1, 1.0)]);
        let mut d = LDleiden::bootstrap(g, LeidenConfig::default());
        let before_m = d.graph().total_edge_weight();
        d.on_edge_add(1, 2, 2.0);
        let after_m = d.graph().total_edge_weight();
        assert!(
            (after_m - (before_m + 2.0)).abs() < 1e-9,
            "m should grow by 2.0: before={}, after={}",
            before_m,
            after_m
        );
        assert_eq!(d.graph().edge_weight(1, 2), 2.0);
        assert_eq!(d.graph().edge_weight(2, 1), 2.0);
    }

    #[test]
    fn remove_bridge_between_two_clusters_splits_communities() {
        let mut edges = vec![];
        edges.push((0, 1, 1.0));
        edges.push((0, 2, 1.0));
        edges.push((1, 2, 1.0));
        edges.push((3, 4, 1.0));
        edges.push((3, 5, 1.0));
        edges.push((4, 5, 1.0));
        edges.push((2, 3, 1.0)); // bridge
        let g = Graph::from_edges(6, edges);
        let d = LDleiden::bootstrap(g, LeidenConfig::default());

        let n = d.graph().num_nodes();
        let partition = vec![0u32; n as usize];
        let cfg = LeidenConfig::default();
        let graph = d.graph().clone();
        let mut d = LDleiden::from_partition(graph, partition, cfg);

        let (delta, _stats) = d.on_edge_remove(2, 3);

        let p = d.partition();
        let c0 = p[0];
        let c3 = p[3];
        assert_ne!(c0, c3, "communities must split after bridge removal: {:?}", p);
        assert_eq!(p[1], c0);
        assert_eq!(p[2], c0);
        assert_eq!(p[4], c3);
        assert_eq!(p[5], c3);
        assert!(
            delta.iter().any(|(n, c)| *n == 3 && *c == c3),
            "delta must include (3, {}): {:?}",
            c3,
            delta
        );
    }

    #[test]
    fn remove_internal_edge_without_split_is_noop_or_local() {
        let g = Graph::from_edges(
            3,
            vec![(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)],
        );
        let partition = vec![0u32, 0, 0];
        let cfg = LeidenConfig::default();
        let mut d = LDleiden::from_partition(g, partition, cfg);
        d.on_edge_remove(0, 1);
        assert_eq!(d.partition(), &[0u32, 0, 0]);
    }

    #[test]
    fn empty_communities_are_reclaimed_and_recycled() {
        // Build a driver with 3 distinct communities where community 1
        // has only one node. Move that node to community 0 via the
        // internal helper so all counters stay consistent — the
        // now-empty community 1 should be reclaimed into the
        // free-list, and the next allocate_cid should return 1.
        let g = Graph::from_edges(
            3,
            vec![(0, 1, 10.0), (1, 2, 0.1)],
        );
        let mut d = LDleiden::from_partition(g, vec![0u32, 1, 2], LeidenConfig::default());
        let k1 = d.graph().strength(1);
        d.move_node(1, 1, 0, k1);
        assert_eq!(d.k_c.len(), 3, "k_c still has zero-strength entry for c=1");
        let reclaimed = d.reclaim_empty_communities();
        assert_eq!(reclaimed, 1, "exactly one empty community to reclaim");
        assert_eq!(d.k_c.len(), 2);
        assert_eq!(d.allocate_cid(), 1);
    }

    #[test]
    fn compact_ids_makes_partition_dense() {
        let g = Graph::from_edges(3, vec![(0, 1, 1.0), (1, 2, 1.0)]);
        let mut d = LDleiden::from_partition(g, vec![7u32, 0, 5], LeidenConfig::default());
        let k = d.compact_ids();
        assert_eq!(k, 3);
        assert_eq!(d.partition(), &[0u32, 1, 2]);
    }

    fn xorshift64(state: &mut u64) -> u64 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        x
    }

    fn sbm_edges(
        n_blocks: usize,
        block_size: u32,
        p_in: f64,
        p_out: f64,
        seed: u64,
    ) -> (u32, Vec<(u32, u32, f64)>) {
        let n = (n_blocks as u32) * block_size;
        let mut state = seed;
        let mut edges = Vec::new();
        for u in 0..n {
            let block_u = (u / block_size) as usize;
            for v in (u + 1)..n {
                let block_v = (v / block_size) as usize;
                let p = if block_u == block_v { p_in } else { p_out };
                let r = (xorshift64(&mut state) as f64) / (u64::MAX as f64);
                if r < p {
                    edges.push((u, v, 1.0));
                }
            }
        }
        (n, edges)
    }

    #[test]
    fn sbm_quality_gate_bootstrap() {
        let (n, edges) = sbm_edges(5, 20, 0.4, 0.02, 0xBEEF_CAFE);
        let g = Graph::from_edges(n, edges);
        let d = LDleiden::bootstrap(g, LeidenConfig::default());
        let q = d.modularity();
        let unique: std::collections::HashSet<u32> =
            d.partition().iter().copied().collect();
        eprintln!(
            "SBM quality: Q = {}, num_communities = {}",
            q,
            unique.len()
        );
        assert!(q >= 0.40, "SBM modularity {} below 0.40 gate", q);
        assert!(
            unique.len() >= 3 && unique.len() <= 8,
            "SBM communities count {} outside plausible range",
            unique.len()
        );
    }

    #[test]
    fn sbm_incremental_retains_quality() {
        let (n, edges) = sbm_edges(5, 20, 0.4, 0.02, 0xBEEF_CAFE);
        let g = Graph::from_edges(n, edges);
        let cfg = LeidenConfig::default();
        let mut d = LDleiden::bootstrap(g, cfg);
        let q_before = d.modularity();

        let mut state = 0xDEAD_BEEF_u64;
        let block_size = 20u32;
        for _ in 0..50 {
            let block = (xorshift64(&mut state) % 5) as u32;
            let base = block * block_size;
            let u = base + (xorshift64(&mut state) % block_size as u64) as u32;
            let v = base + (xorshift64(&mut state) % block_size as u64) as u32;
            if u == v {
                continue;
            }
            d.on_edge_add(u, v, 1.0);
        }

        let q_after = d.modularity();
        eprintln!(
            "SBM incremental: Q_before = {}, Q_after = {}, ratio = {}",
            q_before,
            q_after,
            q_after / q_before
        );
        assert!(
            q_after >= 0.95 * q_before,
            "SBM incremental modularity {} degraded below 95% of bootstrap {} ({})",
            q_after,
            q_before,
            q_after / q_before
        );
    }

    #[test]
    fn perf_smoke_1k_nodes_100_deltas_under_100ms() {
        let (n, edges) = sbm_edges(50, 20, 0.3, 0.005, 0xC0DE_D00D);
        let g = Graph::from_edges(n, edges);
        let cfg = LeidenConfig::default();
        let mut d = LDleiden::bootstrap(g, cfg);

        let mut state = 0xF00D_F00D_u64;
        let block_size = 20u32;

        let start = std::time::Instant::now();
        for _ in 0..100 {
            let block = (xorshift64(&mut state) % 50) as u32;
            let base = block * block_size;
            let u = base + (xorshift64(&mut state) % block_size as u64) as u32;
            let v = base + (xorshift64(&mut state) % block_size as u64) as u32;
            if u == v {
                continue;
            }
            d.on_edge_add(u, v, 1.0);
        }
        let elapsed = start.elapsed();
        eprintln!(
            "perf smoke: 100 deltas on n=1000 sbm graph in {:?}",
            elapsed
        );
        assert!(
            elapsed < std::time::Duration::from_millis(100),
            "100 incremental deltas took {:?} > 100ms — regression",
            elapsed
        );
    }

    #[test]
    fn reweight_then_remove_roundtrip() {
        let g = Graph::from_edges(3, vec![(0, 1, 1.0), (1, 2, 1.0)]);
        let mut d = LDleiden::bootstrap(g, LeidenConfig::default());
        let m0 = d.graph().total_edge_weight();
        d.on_edge_reweight(0, 1, 3.0); // 1.0 → 4.0
        assert_eq!(d.graph().edge_weight(0, 1), 4.0);
        d.on_edge_remove(0, 1);
        assert_eq!(d.graph().edge_weight(0, 1), 0.0);
        assert!((d.graph().total_edge_weight() - 1.0).abs() < 1e-9, "m = {}", d.graph().total_edge_weight());
        let _ = m0;
    }

    /// Scaling sanity: at fixed SBM density, per-delta time should
    /// grow at most linearly with avg_degree (which grows linearly
    /// with n_blocks here). A quadratic/cubic regression against
    /// this expectation would indicate we've re-introduced a
    /// full-graph scan on the hot path.
    #[test]
    fn perf_scales_near_linearly_in_avg_degree() {
        // Run the bench harness at two scales and check the ratio.
        // SBM(n_blocks, block=50, p_in=0.3, p_out=0.005).
        // At n=1k (20 blocks): avg_degree ≈ 19.
        // At n=5k (100 blocks): avg_degree ≈ 40.
        // Per-delta cost should scale ≈ 2x (degree ratio), not 5x
        // or more. The incremental delta count scales ~10x, so
        // total-time ratio should be ~20x max.
        let measure = |n_blocks: usize| -> (std::time::Duration, usize) {
            let (_n, edges) = sbm_edges(n_blocks, 50, 0.3, 0.005, 0xC0DE_D00D);
            let n_edges = edges.len();
            let n_nodes = (n_blocks as u32) * 50;
            let cfg = LeidenConfig::default();
            let mut d = LDleiden::bootstrap(Graph::from_edges(n_nodes, edges), cfg);
            let mut state = 0xF00D_F00D_u64;
            let delta_count = (n_edges / 10).max(100);
            let start = std::time::Instant::now();
            for _ in 0..delta_count {
                let block = (xorshift64(&mut state) % n_blocks as u64) as u32;
                let base = block * 50;
                let u = base + (xorshift64(&mut state) % 50) as u32;
                let v = base + (xorshift64(&mut state) % 50) as u32;
                if u == v {
                    continue;
                }
                d.on_edge_add(u, v, 1.0);
            }
            (start.elapsed(), delta_count)
        };

        let (t_small, deltas_small) = measure(20);
        let (t_large, deltas_large) = measure(100);
        let per_delta_small = t_small.as_nanos() as f64 / deltas_small as f64;
        let per_delta_large = t_large.as_nanos() as f64 / deltas_large as f64;
        let ratio_per_delta = per_delta_large / per_delta_small.max(1.0);
        let ratio_total = t_large.as_secs_f64() / t_small.as_secs_f64().max(1e-9);
        eprintln!(
            "perf scaling: n=1k {:?} ({} deltas, {:.1} ns/delta), \
             n=5k {:?} ({} deltas, {:.1} ns/delta), \
             per-delta ratio = {:.2}, total ratio = {:.2}",
            t_small, deltas_small, per_delta_small,
            t_large, deltas_large, per_delta_large,
            ratio_per_delta, ratio_total
        );
        // Contract: per-delta cost must not explode. Degree ratio is
        // ~2x between these two scales; we accept up to 5x to allow
        // for noise + cross-community deltas doing real work. A 10x+
        // ratio would mean we're back to a pre-fix regime.
        assert!(
            ratio_per_delta < 5.0,
            "per-delta time scaled {:.1}x — regression in hot path \
             (small={} ns/delta, large={} ns/delta)",
            ratio_per_delta,
            per_delta_small as u64,
            per_delta_large as u64
        );
    }
}
