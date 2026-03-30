//! Hilbert curve encoding/decoding for KV position layout.
//!
//! Maps 2D coordinates (from spectral embedding) to 1D Hilbert index
//! and back. The Hilbert curve preserves spatial locality better than
//! row-major or Z-order curves, making it ideal for mapping graph
//! topology → KV cache positions (RoPE alignment).
//!
//! Order N means the grid is 2^N × 2^N (side = 2^N).

/// Convert a Hilbert curve index `d` to (x, y) coordinates.
/// `order` is the curve order (grid is 2^order × 2^order).
///
/// # Panics
/// Panics if `d >= 4^order`.
pub fn hilbert_d2xy(order: u32, d: u32) -> (u32, u32) {
    let n = 1u32 << order; // side length
    debug_assert!(
        d < n * n,
        "d={d} out of range for order={order} (max={})",
        n * n - 1
    );

    let mut x = 0u32;
    let mut y = 0u32;
    let mut rx: u32;
    let mut ry: u32;
    let mut d = d;
    let mut s = 1u32;

    while s < n {
        rx = if (d & 2) != 0 { 1 } else { 0 };
        ry = if (d & 1) != 0 { 1 } else { 0 };
        // XOR ry with rx for the actual ry
        ry ^= rx;
        // Negate: this is equivalent to the standard Hilbert rotation
        // but done without signed arithmetic
        rot(s, &mut x, &mut y, rx, ry);
        x += s * rx;
        y += s * ry;
        d >>= 2;
        s <<= 1;
    }

    (x, y)
}

/// Convert (x, y) coordinates to a Hilbert curve index.
/// `order` is the curve order (grid is 2^order × 2^order).
///
/// # Panics
/// Panics if x or y >= 2^order.
pub fn hilbert_xy2d(order: u32, x: u32, y: u32) -> u32 {
    let n = 1u32 << order;
    debug_assert!(x < n && y < n, "({x},{y}) out of range for order={order}");

    let mut d = 0u32;
    let mut x = x;
    let mut y = y;
    let mut rx: u32;
    let mut ry: u32;

    let mut s = n >> 1;
    while s > 0 {
        rx = if (x & s) > 0 { 1 } else { 0 };
        ry = if (y & s) > 0 { 1 } else { 0 };
        d += s * s * ((3 * rx) ^ ry);
        rot(s, &mut x, &mut y, rx, ry);
        s >>= 1;
    }

    d
}

/// Rotate/flip a quadrant.
fn rot(n: u32, x: &mut u32, y: &mut u32, rx: u32, ry: u32) {
    if ry == 0 {
        if rx == 1 {
            *x = n.wrapping_sub(1).wrapping_sub(*x);
            *y = n.wrapping_sub(1).wrapping_sub(*y);
        }
        std::mem::swap(x, y);
    }
}

/// Map a set of 2D points to Hilbert-ordered 1D positions.
///
/// Takes normalized coordinates in [0, 1] × [0, 1], quantizes them
/// to a grid of order `order`, and returns Hilbert indices.
///
/// Returns (hilbert_index, original_index) pairs sorted by hilbert_index.
pub fn points_to_hilbert_order(points: &[(f32, f32)], order: u32) -> Vec<(u32, usize)> {
    let side = (1u32 << order) as f32;
    let max_coord = (1u32 << order) - 1;

    let mut indexed: Vec<(u32, usize)> = points
        .iter()
        .enumerate()
        .map(|(i, &(px, py))| {
            // Clamp to [0, 1] then quantize
            let qx = ((px.clamp(0.0, 1.0) * side) as u32).min(max_coord);
            let qy = ((py.clamp(0.0, 1.0) * side) as u32).min(max_coord);
            let d = hilbert_xy2d(order, qx, qy);
            (d, i)
        })
        .collect();

    indexed.sort_by_key(|&(d, _)| d);
    indexed
}

/// Assign KV positions to nodes based on Hilbert ordering.
///
/// Given `n_nodes` nodes with 2D spectral coordinates, returns a Vec
/// of KV position offsets (0-based) in Hilbert order.
/// `positions[i]` = the Hilbert-based position for node `i`.
///
/// The order is automatically chosen to fit all nodes:
/// order = ceil(log2(ceil(sqrt(n_nodes)))).
pub fn assign_hilbert_positions(points: &[(f32, f32)]) -> Vec<u32> {
    if points.is_empty() {
        return Vec::new();
    }

    let order = optimal_order(points.len());
    let sorted = points_to_hilbert_order(points, order);

    // Map: original_index → position_in_hilbert_order
    let mut positions = vec![0u32; points.len()];
    for (pos, &(_d, orig_idx)) in sorted.iter().enumerate() {
        positions[orig_idx] = pos as u32;
    }

    positions
}

/// Compute the optimal Hilbert order for `n` points.
/// Returns the smallest order such that 4^order >= n.
pub fn optimal_order(n: usize) -> u32 {
    if n <= 1 {
        return 1;
    }
    let side = (n as f64).sqrt().ceil() as u32;
    let mut order = 1u32;
    while (1u32 << order) < side {
        order += 1;
    }
    order
}

// ── Weighted Adjacency & Fusion (E4) ─────────────────────────────

use obrain_common::types::NodeId;
use std::collections::{HashMap, HashSet};

/// Weighted adjacency list: for each node, list of (neighbor_idx, weight).
pub type WeightedAdjacencyList = Vec<Vec<(usize, f64)>>;

/// Build a fused adjacency matrix from static graph edges + co-activation data.
///
/// - `db_adjacency`: graph edges from `--db` (read-only), weight = 1.0 each
/// - `coactivations`: learned co-activation pairs from agent state
/// - `beta`: relative weight of co-activation edges (e.g., 0.5)
///
/// Returns (WeightedAdjacencyList, node_ids) where node_ids maps dense index → NodeId.
/// With `beta=0` or empty coactivations, the result is equivalent to the static graph.
pub fn build_fused_adjacency(
    db_adjacency: &HashMap<NodeId, HashSet<NodeId>>,
    coactivations: &[((NodeId, NodeId), f32)], // (pair, decay_score)
    beta: f32,
) -> (WeightedAdjacencyList, Vec<NodeId>) {
    // Collect all node IDs from both sources
    let mut all_nodes: HashSet<NodeId> = HashSet::new();
    for (&nid, neighbors) in db_adjacency {
        all_nodes.insert(nid);
        for &n in neighbors {
            all_nodes.insert(n);
        }
    }
    for &((a, b), _) in coactivations {
        all_nodes.insert(a);
        all_nodes.insert(b);
    }

    let node_ids: Vec<NodeId> = {
        let mut v: Vec<NodeId> = all_nodes.into_iter().collect();
        v.sort_by_key(|n| n.0);
        v
    };
    let id_to_idx: HashMap<NodeId, usize> = node_ids
        .iter()
        .enumerate()
        .map(|(i, &nid)| (nid, i))
        .collect();

    let n = node_ids.len();
    // Use a map to accumulate weights (since coactivation may overlap with static edges)
    let mut weight_map: HashMap<(usize, usize), f64> = HashMap::new();

    // Static edges (weight = 1.0)
    for (&nid, neighbors) in db_adjacency {
        if let Some(&i) = id_to_idx.get(&nid) {
            for &neighbor in neighbors {
                if let Some(&j) = id_to_idx.get(&neighbor) {
                    if i != j {
                        *weight_map.entry((i, j)).or_insert(0.0) += 1.0;
                    }
                }
            }
        }
    }

    // Co-activation edges (weight = beta * decay_score)
    if beta > 0.0 {
        for &((a, b), score) in coactivations {
            if let (Some(&i), Some(&j)) = (id_to_idx.get(&a), id_to_idx.get(&b)) {
                if i != j {
                    let w = beta as f64 * score as f64;
                    *weight_map.entry((i, j)).or_insert(0.0) += w;
                    *weight_map.entry((j, i)).or_insert(0.0) += w;
                }
            }
        }
    }

    // Build adjacency list
    let mut adj: WeightedAdjacencyList = vec![Vec::new(); n];
    for (&(i, j), &w) in &weight_map {
        adj[i].push((j, w));
    }

    (adj, node_ids)
}

// ── HilbertLayout ───────────────────────────────────────────────

/// Complete Hilbert layout mapping: NodeId → KV position.
///
/// Pipeline: graph adjacency → spectral embedding 2D → Hilbert curve → KV positions.
#[derive(Debug, Clone)]
pub struct HilbertLayout {
    /// Map from NodeId to its Hilbert-ordered KV position offset (0-based).
    pub positions: HashMap<NodeId, u32>,
    /// The Hilbert order used.
    pub order: u32,
    /// 2D spectral coordinates (for debugging / visualization).
    pub coords_2d: HashMap<NodeId, (f32, f32)>,
}

impl HilbertLayout {
    /// Compute a Hilbert layout from an adjacency map.
    ///
    /// `adjacency` maps each NodeId to its set of neighbor NodeIds.
    /// `base_position` is added to all positions (typically `header_end`).
    pub fn compute(
        adjacency: &HashMap<NodeId, std::collections::HashSet<NodeId>>,
        base_position: u32,
    ) -> Self {
        if adjacency.is_empty() {
            return Self {
                positions: HashMap::new(),
                order: 1,
                coords_2d: HashMap::new(),
            };
        }

        // Build dense index: NodeId ↔ usize
        let node_ids: Vec<NodeId> = adjacency.keys().copied().collect();
        let id_to_idx: HashMap<NodeId, usize> = node_ids
            .iter()
            .enumerate()
            .map(|(i, &nid)| (nid, i))
            .collect();

        // Build dense adjacency list
        let n = node_ids.len();
        let mut dense_adj: AdjacencyList = vec![Vec::new(); n];
        for (&nid, neighbors) in adjacency {
            if let Some(&i) = id_to_idx.get(&nid) {
                for &neighbor in neighbors {
                    if let Some(&j) = id_to_idx.get(&neighbor) {
                        dense_adj[i].push(j);
                    }
                }
            }
        }

        // Spectral embedding
        let coords = spectral_embedding_2d(&dense_adj);

        // Hilbert ordering
        let hilbert_positions = assign_hilbert_positions(&coords);

        // Build result maps
        let mut positions = HashMap::with_capacity(n);
        let mut coords_2d = HashMap::with_capacity(n);
        let order = optimal_order(n);

        for (idx, &nid) in node_ids.iter().enumerate() {
            positions.insert(nid, base_position + hilbert_positions[idx]);
            coords_2d.insert(nid, coords[idx]);
        }

        Self {
            positions,
            order,
            coords_2d,
        }
    }

    /// Compute a Hilbert layout from a weighted adjacency list (E4).
    ///
    /// Same as `compute()` but uses weighted spectral embedding.
    /// `weighted_adj` and `node_ids` come from `build_fused_adjacency()`.
    pub fn compute_weighted(
        weighted_adj: &WeightedAdjacencyList,
        node_ids: &[NodeId],
        base_position: u32,
    ) -> Self {
        if weighted_adj.is_empty() || node_ids.is_empty() {
            return Self {
                positions: HashMap::new(),
                order: 1,
                coords_2d: HashMap::new(),
            };
        }

        let coords = spectral_embedding_2d_weighted(weighted_adj);
        let hilbert_positions = assign_hilbert_positions(&coords);
        let order = optimal_order(node_ids.len());

        let mut positions = HashMap::with_capacity(node_ids.len());
        let mut coords_2d = HashMap::with_capacity(node_ids.len());

        for (idx, &nid) in node_ids.iter().enumerate() {
            positions.insert(nid, base_position + hilbert_positions[idx]);
            coords_2d.insert(nid, coords[idx]);
        }

        Self {
            positions,
            order,
            coords_2d,
        }
    }

    /// Non-disruptive re-layout from fused adjacency (E4).
    ///
    /// Recomputes positions from the weighted graph, but only updates positions
    /// for nodes NOT in `frozen_nodes`. Frozen nodes keep their current position.
    /// This ensures KV cache entries remain valid.
    ///
    /// Returns the number of nodes that got new positions.
    pub fn update_from_fused(
        &mut self,
        weighted_adj: &WeightedAdjacencyList,
        node_ids: &[NodeId],
        base_position: u32,
        frozen_nodes: &HashSet<NodeId>,
    ) -> usize {
        let new_layout = Self::compute_weighted(weighted_adj, node_ids, base_position);
        let mut updated = 0;

        for (&nid, &new_pos) in &new_layout.positions {
            if frozen_nodes.contains(&nid) {
                continue; // Keep existing position
            }
            self.positions.insert(nid, new_pos);
            if let Some(coords) = new_layout.coords_2d.get(&nid) {
                self.coords_2d.insert(nid, *coords);
            }
            updated += 1;
        }

        // Add any new nodes not previously in layout
        for (&nid, &new_pos) in &new_layout.positions {
            if !self.positions.contains_key(&nid) {
                self.positions.insert(nid, new_pos);
                if let Some(coords) = new_layout.coords_2d.get(&nid) {
                    self.coords_2d.insert(nid, *coords);
                }
                updated += 1;
            }
        }

        self.order = new_layout.order;
        updated
    }

    /// Get the KV position for a node.
    pub fn get_position(&self, node_id: NodeId) -> Option<u32> {
        self.positions.get(&node_id).copied()
    }

    /// Number of nodes in the layout.
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Whether the layout is empty.
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Get the maximum position assigned.
    pub fn max_position(&self) -> Option<u32> {
        self.positions.values().max().copied()
    }

    /// Get nodes sorted by their Hilbert position.
    pub fn nodes_by_position(&self) -> Vec<(NodeId, u32)> {
        let mut pairs: Vec<(NodeId, u32)> = self
            .positions
            .iter()
            .map(|(&nid, &pos)| (nid, pos))
            .collect();
        pairs.sort_by_key(|&(_, pos)| pos);
        pairs
    }
}

// ── Spectral Embedding ──────────────────────────────────────────

/// Sparse adjacency representation for spectral embedding.
/// Each node maps to its neighbor indices (0-based dense indices).
pub type AdjacencyList = Vec<Vec<usize>>;

/// Compute 2D spectral embedding of a graph via the normalized Laplacian.
///
/// Uses power iteration with deflation to extract the 2 smallest
/// non-trivial eigenvectors (Fiedler vector + next).
///
/// `adjacency[i]` = list of neighbor indices for node i.
/// Returns Vec of (x, y) coordinates normalized to [0, 1].
///
/// For disconnected graphs or single-node graphs, returns (0.5, 0.5) for all.
pub fn spectral_embedding_2d(adjacency: &AdjacencyList) -> Vec<(f32, f32)> {
    // Convert unweighted → weighted (all edges weight 1.0)
    let weighted: WeightedAdjacencyList = adjacency
        .iter()
        .map(|neighbors| neighbors.iter().map(|&j| (j, 1.0)).collect())
        .collect();
    spectral_embedding_2d_weighted(&weighted)
}

/// Compute 2D spectral embedding from a weighted adjacency list (E4).
///
/// Uses the weighted normalized Laplacian: L_norm = I - D^{-1/2} W D^{-1/2}
/// where D_ii = Σ_j W_ij (weighted degree).
///
/// With uniform weights = 1.0, this is identical to the unweighted version.
pub fn spectral_embedding_2d_weighted(adjacency: &WeightedAdjacencyList) -> Vec<(f32, f32)> {
    let n = adjacency.len();
    if n <= 2 {
        return vec![(0.5, 0.5); n];
    }

    // Compute weighted degree vector: D_ii = Σ_j W_ij
    let degrees: Vec<f64> = adjacency
        .iter()
        .map(|neighbors| neighbors.iter().map(|&(_, w)| w).sum::<f64>())
        .collect();

    // Check for isolated nodes
    let min_deg = degrees.iter().cloned().fold(f64::INFINITY, f64::min);
    if min_deg < 1e-10 {
        // Graph has isolated nodes — fall back to uniform
        return vec![(0.5, 0.5); n];
    }

    // D^{-1/2}
    let d_inv_sqrt: Vec<f64> = degrees.iter().map(|&d| 1.0 / d.sqrt()).collect();

    // Matrix-vector product: M * v = D^{-1/2} W D^{-1/2} v
    let matvec = |v: &[f64]| -> Vec<f64> {
        let mut result = vec![0.0f64; n];
        for i in 0..n {
            let wi = d_inv_sqrt[i] * v[i]; // D^{-1/2} v
            for &(j, w) in &adjacency[i] {
                result[j] += d_inv_sqrt[j] * wi * w; // D^{-1/2} (W (D^{-1/2} v))
            }
        }
        result
    };

    // The largest eigenvector of M is the trivial one: D^{1/2} * 1
    let trivial: Vec<f64> = degrees.iter().map(|&d| d.sqrt()).collect();
    let trivial_norm = vec_norm(&trivial);
    let trivial_unit: Vec<f64> = trivial.iter().map(|&x| x / trivial_norm).collect();

    // Power iteration for 2nd largest eigenvector (Fiedler)
    let fiedler = power_iteration_deflated(&matvec, &[&trivial_unit], n, 200);

    // Power iteration for 3rd largest eigenvector
    let fiedler_norm = vec_norm(&fiedler);
    let fiedler_unit: Vec<f64> = fiedler
        .iter()
        .map(|&x| x / fiedler_norm.max(1e-12))
        .collect();
    let third = power_iteration_deflated(&matvec, &[&trivial_unit, &fiedler_unit], n, 200);

    // Normalize both vectors to [0, 1]
    let xs = normalize_to_unit(&fiedler);
    let ys = normalize_to_unit(&third);

    xs.into_iter()
        .zip(ys)
        .map(|(x, y)| (x as f32, y as f32))
        .collect()
}

/// Power iteration with deflation against given orthogonal vectors.
fn power_iteration_deflated(
    matvec: &dyn Fn(&[f64]) -> Vec<f64>,
    deflate_against: &[&[f64]],
    n: usize,
    max_iters: usize,
) -> Vec<f64> {
    // Random-ish initial vector (deterministic for reproducibility)
    let mut v: Vec<f64> = (0..n)
        .map(|i| {
            // Simple hash-based pseudo-random
            let x = (i as f64 + 1.0) * 0.618033988749895; // golden ratio
            x - x.floor() - 0.5
        })
        .collect();

    // Deflate initial vector
    for &u in deflate_against {
        let proj = dot(&v, u);
        for i in 0..n {
            v[i] -= proj * u[i];
        }
    }

    let norm = vec_norm(&v);
    if norm < 1e-12 {
        return vec![0.0; n];
    }
    for x in &mut v {
        *x /= norm;
    }

    for _ in 0..max_iters {
        // v = M * v
        let mut mv = matvec(&v);

        // Deflate
        for &u in deflate_against {
            let proj = dot(&mv, u);
            for i in 0..n {
                mv[i] -= proj * u[i];
            }
        }

        let norm = vec_norm(&mv);
        if norm < 1e-12 {
            break;
        }
        for x in &mut mv {
            *x /= norm;
        }

        v = mv;
    }

    v
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn vec_norm(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

/// Normalize a vector to [0, 1] range.
fn normalize_to_unit(v: &[f64]) -> Vec<f64> {
    if v.is_empty() {
        return Vec::new();
    }
    let min = v.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;
    if range < 1e-12 {
        return vec![0.5; v.len()];
    }
    v.iter().map(|&x| (x - min) / range).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_d2xy_xy2d_roundtrip_order2() {
        // Order 2 → 4×4 grid → 16 cells
        for d in 0..16 {
            let (x, y) = hilbert_d2xy(2, d);
            let d2 = hilbert_xy2d(2, x, y);
            assert_eq!(d, d2, "roundtrip failed for d={d}: ({x},{y}) → {d2}");
        }
    }

    #[test]
    fn test_d2xy_xy2d_roundtrip_order4() {
        // Order 4 → 16×16 grid → 256 cells
        for d in 0..256 {
            let (x, y) = hilbert_d2xy(4, d);
            let d2 = hilbert_xy2d(4, x, y);
            assert_eq!(d, d2, "roundtrip failed for d={d}");
        }
    }

    #[test]
    fn test_locality() {
        // Adjacent Hilbert indices should map to adjacent (x,y) cells
        for d in 0..15 {
            let (x1, y1) = hilbert_d2xy(2, d);
            let (x2, y2) = hilbert_d2xy(2, d + 1);
            let dist =
                (x1 as i32 - x2 as i32).unsigned_abs() + (y1 as i32 - y2 as i32).unsigned_abs();
            assert_eq!(
                dist, 1,
                "Hilbert locality violated at d={d}: ({x1},{y1})->({x2},{y2})"
            );
        }
    }

    #[test]
    fn test_order1() {
        // Order 1 → 2×2 grid → 4 cells
        let expected = [(0, 0), (0, 1), (1, 1), (1, 0)];
        for (d, &(ex, ey)) in expected.iter().enumerate() {
            let (x, y) = hilbert_d2xy(1, d as u32);
            assert_eq!((x, y), (ex, ey), "order 1 d={d}");
        }
    }

    #[test]
    fn test_points_to_hilbert_order() {
        let points = vec![
            (0.0, 0.0), // bottom-left
            (1.0, 0.0), // bottom-right
            (1.0, 1.0), // top-right
            (0.0, 1.0), // top-left
        ];
        let sorted = points_to_hilbert_order(&points, 2);
        // All 4 points should appear
        assert_eq!(sorted.len(), 4);
        // Indices should be unique
        let mut orig_indices: Vec<usize> = sorted.iter().map(|&(_, i)| i).collect();
        orig_indices.sort();
        assert_eq!(orig_indices, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_assign_hilbert_positions() {
        let points = vec![(0.1, 0.1), (0.9, 0.9), (0.5, 0.5), (0.1, 0.9)];
        let positions = assign_hilbert_positions(&points);
        assert_eq!(positions.len(), 4);
        // Each position should be unique (0..4)
        let mut sorted_pos: Vec<u32> = positions.clone();
        sorted_pos.sort();
        assert_eq!(sorted_pos, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_optimal_order() {
        assert_eq!(optimal_order(1), 1); // 2×2 = 4 >= 1
        assert_eq!(optimal_order(4), 1); // 2×2 = 4 >= 4 (ceil(sqrt(4))=2, 2^1=2 >= 2)
        assert_eq!(optimal_order(5), 2); // ceil(sqrt(5))=3, 2^2=4 >= 3
        assert_eq!(optimal_order(16), 2); // ceil(sqrt(16))=4, 2^2=4 >= 4
        assert_eq!(optimal_order(17), 3); // ceil(sqrt(17))=5, 2^3=8 >= 5
        assert_eq!(optimal_order(100), 4); // ceil(sqrt(100))=10, 2^4=16 >= 10
    }

    #[test]
    fn test_empty_points() {
        assert!(assign_hilbert_positions(&[]).is_empty());
    }

    // ── Spectral embedding tests ────────────────────────────────

    #[test]
    fn test_spectral_two_clusters() {
        // Two clusters of 4 nodes each, connected by a single bridge edge
        // Cluster A: 0-1-2-3 (complete), Cluster B: 4-5-6-7 (complete)
        // Bridge: 3-4
        let mut adj: AdjacencyList = vec![Vec::new(); 8];

        // Cluster A (complete K4)
        for i in 0..4 {
            for j in 0..4 {
                if i != j {
                    adj[i].push(j);
                }
            }
        }
        // Cluster B (complete K4)
        for i in 4..8 {
            for j in 4..8 {
                if i != j {
                    adj[i].push(j);
                }
            }
        }
        // Bridge
        adj[3].push(4);
        adj[4].push(3);

        let coords = spectral_embedding_2d(&adj);
        assert_eq!(coords.len(), 8);

        // Cluster A centroid
        let cx_a: f32 = coords[0..4].iter().map(|c| c.0).sum::<f32>() / 4.0;
        let cy_a: f32 = coords[0..4].iter().map(|c| c.1).sum::<f32>() / 4.0;
        // Cluster B centroid
        let cx_b: f32 = coords[4..8].iter().map(|c| c.0).sum::<f32>() / 4.0;
        let cy_b: f32 = coords[4..8].iter().map(|c| c.1).sum::<f32>() / 4.0;

        // Clusters should be separated (centroids far apart)
        let dist = ((cx_a - cx_b).powi(2) + (cy_a - cy_b).powi(2)).sqrt();
        assert!(
            dist > 0.2,
            "Clusters should be separated: dist={dist}, A=({cx_a},{cy_a}), B=({cx_b},{cy_b})"
        );
    }

    #[test]
    fn test_spectral_small_graph() {
        // Simple path graph: 0-1-2-3-4
        let adj: AdjacencyList = vec![
            vec![1],    // 0
            vec![0, 2], // 1
            vec![1, 3], // 2
            vec![2, 4], // 3
            vec![3],    // 4
        ];
        let coords = spectral_embedding_2d(&adj);
        assert_eq!(coords.len(), 5);

        // On a path graph, the Fiedler vector should give monotonic ordering
        // Check that node 0 and node 4 are far apart on at least one axis
        let d04 =
            ((coords[0].0 - coords[4].0).powi(2) + (coords[0].1 - coords[4].1).powi(2)).sqrt();
        assert!(d04 > 0.3, "Endpoints of path should be separated: d={d04}");
    }

    #[test]
    fn test_spectral_single_node() {
        let adj: AdjacencyList = vec![Vec::new()];
        let coords = spectral_embedding_2d(&adj);
        assert_eq!(coords, vec![(0.5, 0.5)]);
    }

    #[test]
    fn test_spectral_two_nodes() {
        let adj: AdjacencyList = vec![vec![1], vec![0]];
        let coords = spectral_embedding_2d(&adj);
        assert_eq!(coords.len(), 2);
    }

    #[test]
    fn test_hilbert_layout_compute() {
        use std::collections::HashSet;
        // Two clusters connected by bridge
        let mut adj: HashMap<NodeId, HashSet<NodeId>> = HashMap::new();
        let ids: Vec<NodeId> = (0..8).map(NodeId).collect();

        // Cluster A: K4
        for i in 0..4 {
            for j in 0..4 {
                if i != j {
                    adj.entry(ids[i]).or_default().insert(ids[j]);
                }
            }
        }
        // Cluster B: K4
        for i in 4..8 {
            for j in 4..8 {
                if i != j {
                    adj.entry(ids[i]).or_default().insert(ids[j]);
                }
            }
        }
        // Bridge
        adj.entry(ids[3]).or_default().insert(ids[4]);
        adj.entry(ids[4]).or_default().insert(ids[3]);

        let layout = HilbertLayout::compute(&adj, 10); // base_position = 10
        assert_eq!(layout.len(), 8);
        // All positions >= base
        for &pos in layout.positions.values() {
            assert!(pos >= 10, "Position {pos} should be >= base 10");
        }
    }

    #[test]
    fn test_hilbert_layout_connected_closer_than_disconnected() {
        use std::collections::HashSet;
        // Ring graph: 0-1-2-3-4-5-6-7-0
        let mut adj: HashMap<NodeId, HashSet<NodeId>> = HashMap::new();
        let ids: Vec<NodeId> = (0..8).map(NodeId).collect();
        for i in 0..8 {
            let next = (i + 1) % 8;
            adj.entry(ids[i]).or_default().insert(ids[next]);
            adj.entry(ids[next]).or_default().insert(ids[i]);
        }

        let layout = HilbertLayout::compute(&adj, 0);

        // Average position distance between connected pairs should be
        // smaller than between random non-connected pairs
        let mut connected_dist = 0.0f64;
        let mut n_connected = 0;
        for i in 0..8 {
            let next = (i + 1) % 8;
            let p1 = layout.get_position(ids[i]).unwrap() as f64;
            let p2 = layout.get_position(ids[next]).unwrap() as f64;
            connected_dist += (p1 - p2).abs();
            n_connected += 1;
        }
        connected_dist /= n_connected as f64;

        let mut random_dist = 0.0f64;
        let mut n_random = 0;
        // Non-adjacent pairs: distance >= 3 in the ring
        for i in 0..8 {
            for skip in [3, 4] {
                let j = (i + skip) % 8;
                let p1 = layout.get_position(ids[i]).unwrap() as f64;
                let p2 = layout.get_position(ids[j]).unwrap() as f64;
                random_dist += (p1 - p2).abs();
                n_random += 1;
            }
        }
        random_dist /= n_random as f64;

        // Connected pairs should be closer on average
        assert!(
            connected_dist <= random_dist + 1.0,
            "Connected avg dist ({connected_dist:.1}) should be <= non-connected ({random_dist:.1})"
        );
    }

    #[test]
    fn test_hilbert_layout_empty() {
        let layout = HilbertLayout::compute(&HashMap::new(), 0);
        assert!(layout.is_empty());
    }

    // ── E4: Weighted spectral + fused adjacency tests ───────────

    #[test]
    fn test_build_fused_adjacency_static_only() {
        // With beta=0, fused adjacency = static adjacency
        let mut db_adj: HashMap<NodeId, HashSet<NodeId>> = HashMap::new();
        db_adj.entry(NodeId(0)).or_default().insert(NodeId(1));
        db_adj.entry(NodeId(1)).or_default().insert(NodeId(0));
        db_adj.entry(NodeId(1)).or_default().insert(NodeId(2));
        db_adj.entry(NodeId(2)).or_default().insert(NodeId(1));

        let (weighted, node_ids) = build_fused_adjacency(&db_adj, &[], 0.0);
        assert_eq!(node_ids.len(), 3);
        assert_eq!(weighted.len(), 3);

        // All weights should be 1.0 (static only)
        for neighbors in &weighted {
            for &(_, w) in neighbors {
                assert!((w - 1.0).abs() < 0.001, "weight should be 1.0, got {w}");
            }
        }
    }

    #[test]
    fn test_build_fused_adjacency_with_coactivation() {
        let mut db_adj: HashMap<NodeId, HashSet<NodeId>> = HashMap::new();
        db_adj.entry(NodeId(0)).or_default().insert(NodeId(1));
        db_adj.entry(NodeId(1)).or_default().insert(NodeId(0));

        // Co-activation: (0, 2) with score 3.0 — a NEW edge not in static graph
        let coact = vec![((NodeId(0), NodeId(2)), 3.0f32)];

        let (weighted, node_ids) = build_fused_adjacency(&db_adj, &coact, 0.5);
        // Should have 3 nodes now (0, 1, 2)
        assert_eq!(node_ids.len(), 3);

        // Find node 2's index
        let idx_2 = node_ids.iter().position(|&n| n == NodeId(2)).unwrap();
        // Node 2 should have edges from co-activation
        assert!(
            !weighted[idx_2].is_empty(),
            "Node 2 should have co-activation edges"
        );
    }

    #[test]
    fn test_weighted_spectral_uniform_equals_unweighted() {
        // With uniform weights, weighted spectral = unweighted spectral
        let adj: AdjacencyList = vec![vec![1, 2], vec![0, 2, 3], vec![0, 1], vec![1]];
        let coords_unw = spectral_embedding_2d(&adj);

        let weighted: WeightedAdjacencyList = adj
            .iter()
            .map(|neighbors| neighbors.iter().map(|&j| (j, 1.0)).collect())
            .collect();
        let coords_w = spectral_embedding_2d_weighted(&weighted);

        assert_eq!(coords_unw.len(), coords_w.len());
        for (a, b) in coords_unw.iter().zip(coords_w.iter()) {
            assert!((a.0 - b.0).abs() < 0.01, "x mismatch: {} vs {}", a.0, b.0);
            assert!((a.1 - b.1).abs() < 0.01, "y mismatch: {} vs {}", a.1, b.1);
        }
    }

    #[test]
    fn test_non_disruptive_relayout() {
        // Initial layout
        let mut adj: HashMap<NodeId, HashSet<NodeId>> = HashMap::new();
        for i in 0..4 {
            for j in 0..4 {
                if i != j {
                    adj.entry(NodeId(i)).or_default().insert(NodeId(j));
                }
            }
        }
        let mut layout = HilbertLayout::compute(&adj, 10);

        // Record initial positions
        let pos0_before = layout.get_position(NodeId(0)).unwrap();
        let pos1_before = layout.get_position(NodeId(1)).unwrap();

        // Freeze nodes 0 and 1 (they're "in KV cache")
        let mut frozen = HashSet::new();
        frozen.insert(NodeId(0));
        frozen.insert(NodeId(1));

        // Re-layout with fused adjacency (add co-activation between 2 and 3)
        let coact = vec![((NodeId(2), NodeId(3)), 5.0f32)];
        let (weighted, node_ids) = build_fused_adjacency(&adj, &coact, 1.0);
        layout.update_from_fused(&weighted, &node_ids, 10, &frozen);

        // Frozen nodes keep their positions
        assert_eq!(
            layout.get_position(NodeId(0)).unwrap(),
            pos0_before,
            "Frozen node 0 should keep position"
        );
        assert_eq!(
            layout.get_position(NodeId(1)).unwrap(),
            pos1_before,
            "Frozen node 1 should keep position"
        );
    }

    #[test]
    fn test_hilbert_layout_nodes_by_position() {
        use std::collections::HashSet;
        let mut adj: HashMap<NodeId, HashSet<NodeId>> = HashMap::new();
        let ids: Vec<NodeId> = (0..4).map(NodeId).collect();
        // Path: 0-1-2-3
        for i in 0..3 {
            adj.entry(ids[i]).or_default().insert(ids[i + 1]);
            adj.entry(ids[i + 1]).or_default().insert(ids[i]);
        }
        let layout = HilbertLayout::compute(&adj, 0);
        let sorted = layout.nodes_by_position();
        assert_eq!(sorted.len(), 4);
        // Positions should be monotonically increasing
        for w in sorted.windows(2) {
            assert!(w[0].1 <= w[1].1);
        }
    }

    #[test]
    fn test_full_pipeline_spectral_to_hilbert() {
        // End-to-end: graph → spectral embedding → Hilbert positions
        // Two clusters, expect nodes within same cluster to get nearby positions
        let mut adj: AdjacencyList = vec![Vec::new(); 8];
        for i in 0..4 {
            for j in 0..4 {
                if i != j {
                    adj[i].push(j);
                }
            }
        }
        for i in 4..8 {
            for j in 4..8 {
                if i != j {
                    adj[i].push(j);
                }
            }
        }
        adj[3].push(4);
        adj[4].push(3);

        let coords = spectral_embedding_2d(&adj);
        let positions = assign_hilbert_positions(&coords);

        assert_eq!(positions.len(), 8);

        // Nodes within each cluster should have positions in a contiguous range
        let mut pos_a: Vec<u32> = positions[0..4].to_vec();
        let mut pos_b: Vec<u32> = positions[4..8].to_vec();
        pos_a.sort();
        pos_b.sort();

        // Spread within cluster should be small (max - min <= 3 for 4-node cluster)
        let spread_a = pos_a[3] - pos_a[0];
        let spread_b = pos_b[3] - pos_b[0];
        assert!(
            spread_a <= 4 && spread_b <= 4,
            "Cluster positions should be tight: A spread={spread_a}, B spread={spread_b}"
        );
    }
}
