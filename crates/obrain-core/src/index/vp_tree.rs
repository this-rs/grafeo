//! VP-Tree (Vantage Point Tree) for exact k-NN and range search.
//!
//! A VP-Tree is a metric space index that partitions data recursively around
//! **vantage points**. At each internal node, all points closer than the
//! **median distance** to the vantage point go left; the rest go right.
//!
//! ## Complexity
//!
//! | Operation       | Average   | Worst    |
//! |-----------------|-----------|----------|
//! | Build           | O(V log V)| O(V²)   |
//! | k-NN query      | O(log V)  | O(V)    |
//! | Range search    | O(log V)  | O(V)    |
//! | Space           | O(V)      | O(V)    |
//!
//! ## VP-Tree vs KD-Tree vs HNSW
//!
//! - **KD-Tree**: fast in low dimensions (≤20d), degrades in high dims.
//! - **VP-Tree**: works well in any dimension with any metric (only needs
//!   a distance function). Exact results. Good for 64d Hilbert features.
//! - **HNSW**: approximate, faster on very large datasets (>1M points),
//!   but more complex and memory-heavy.
//!
//! ## Example
//!
//! ```
//! use obrain_core::index::vp_tree::VpTree;
//!
//! let points: Vec<(u32, Vec<f32>)> = vec![
//!     (0, vec![0.0, 0.0]),
//!     (1, vec![1.0, 0.0]),
//!     (2, vec![0.0, 1.0]),
//!     (3, vec![1.0, 1.0]),
//!     (4, vec![0.5, 0.5]),
//! ];
//!
//! let tree = VpTree::build(points, |a, b| {
//!     a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt()
//! });
//!
//! let results = tree.knn(&[0.0, 0.0], 2, |a, b| {
//!     a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt()
//! });
//! assert_eq!(results.len(), 2);
//! assert_eq!(results[0].0, 0); // closest = origin
//! ```

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

// ============================================================================
// Types
// ============================================================================

/// A node in the VP-Tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct VpNode<T> {
    /// The point's feature vector.
    point: Vec<f32>,
    /// Associated data (e.g., NodeId).
    data: T,
    /// Median distance to the vantage point (partitioning radius).
    radius: f32,
    /// Index of the left child (points with d < radius), or `usize::MAX` if leaf.
    left: usize,
    /// Index of the right child (points with d >= radius), or `usize::MAX` if leaf.
    right: usize,
}

/// A k-NN result entry: (data, distance).
#[derive(Debug, Clone)]
pub struct KnnResult<T> {
    /// The associated data.
    pub data: T,
    /// Distance from the query point.
    pub distance: f32,
}

/// Max-heap entry for k-NN search (we want to evict the farthest neighbor).
struct HeapEntry<T> {
    data: T,
    distance: f32,
}

impl<T> PartialEq for HeapEntry<T> {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl<T> Eq for HeapEntry<T> {}

impl<T> PartialOrd for HeapEntry<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for HeapEntry<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap by distance (largest distance on top for eviction)
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
    }
}

const NULL: usize = usize::MAX;

// ============================================================================
// VpTree
// ============================================================================

/// Vantage Point Tree for exact k-NN and range search in metric spaces.
///
/// Generic over the data type `T` (e.g., `NodeId`, `u64`, `String`).
/// The distance function is provided at query time, allowing different metrics
/// (e.g., weighted Hilbert distance) on the same tree.
///
/// # Serialization
///
/// The tree is `Serialize + Deserialize` when `T` is, allowing persistence
/// to disk (e.g., alongside a `.grafeo` database file).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VpTree<T> {
    nodes: Vec<VpNode<T>>,
    root: usize,
}

impl<T: Clone> VpTree<T> {
    /// Build a VP-Tree from a set of points.
    ///
    /// # Arguments
    ///
    /// * `points` - Vec of `(data, feature_vector)` pairs
    /// * `distance_fn` - Metric function `(a, b) -> f32` (must satisfy triangle inequality)
    ///
    /// # Returns
    ///
    /// A `VpTree` ready for k-NN / range queries. Returns an empty tree if
    /// `points` is empty.
    ///
    /// # Complexity
    ///
    /// O(V log V) average, O(V²) worst case.
    pub fn build<F>(points: Vec<(T, Vec<f32>)>, distance_fn: F) -> Self
    where
        F: Fn(&[f32], &[f32]) -> f32,
    {
        if points.is_empty() {
            return Self {
                nodes: Vec::new(),
                root: NULL,
            };
        }

        let mut items: Vec<(T, Vec<f32>)> = points;
        let mut nodes: Vec<VpNode<T>> = Vec::with_capacity(items.len());
        let indices: Vec<usize> = (0..items.len()).collect();

        let root = Self::build_recursive(&mut nodes, &mut items, &indices, &distance_fn);

        Self { nodes, root }
    }

    fn build_recursive<F>(
        nodes: &mut Vec<VpNode<T>>,
        items: &mut Vec<(T, Vec<f32>)>,
        indices: &[usize],
        distance_fn: &F,
    ) -> usize
    where
        F: Fn(&[f32], &[f32]) -> f32,
    {
        if indices.is_empty() {
            return NULL;
        }

        if indices.len() == 1 {
            let idx = indices[0];
            let node_idx = nodes.len();
            nodes.push(VpNode {
                point: items[idx].1.clone(),
                data: items[idx].0.clone(),
                radius: 0.0,
                left: NULL,
                right: NULL,
            });
            return node_idx;
        }

        // Select vantage point: pick the one with maximum spread (variance) of distances.
        // Sample up to 5 candidates for efficiency.
        let vp_idx = Self::select_vantage_point(items, indices, distance_fn);

        // Compute distances from vantage point to all other points
        let vp_point = items[vp_idx].1.clone();
        let mut dists: Vec<(usize, f32)> = indices
            .iter()
            .filter(|&&i| i != vp_idx)
            .map(|&i| (i, distance_fn(&vp_point, &items[i].1)))
            .collect();

        // Sort by distance and find the median
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        let median_pos = dists.len() / 2;
        let radius = if dists.is_empty() {
            0.0
        } else {
            dists[median_pos].1
        };

        // Split into left (< median) and right (>= median)
        let left_indices: Vec<usize> = dists[..median_pos].iter().map(|d| d.0).collect();
        let right_indices: Vec<usize> = dists[median_pos..].iter().map(|d| d.0).collect();

        // Reserve a slot for this node
        let node_idx = nodes.len();
        nodes.push(VpNode {
            point: vp_point,
            data: items[vp_idx].0.clone(),
            radius,
            left: NULL,
            right: NULL,
        });

        // Recursively build subtrees
        let left = Self::build_recursive(nodes, items, &left_indices, distance_fn);
        let right = Self::build_recursive(nodes, items, &right_indices, distance_fn);

        nodes[node_idx].left = left;
        nodes[node_idx].right = right;

        node_idx
    }

    /// Select the best vantage point from up to 5 random candidates.
    fn select_vantage_point<F>(items: &[(T, Vec<f32>)], indices: &[usize], distance_fn: &F) -> usize
    where
        F: Fn(&[f32], &[f32]) -> f32,
    {
        if indices.len() <= 5 {
            // Small set: just pick the first one
            return indices[0];
        }

        // Sample 5 candidates evenly spread
        let step = indices.len() / 5;
        let candidates: Vec<usize> = (0..5).map(|i| indices[i * step]).collect();

        let mut best_idx = candidates[0];
        let mut best_spread = 0.0_f64;

        for &cand in &candidates {
            // Compute spread (variance) of distances from candidate to sample points
            let sample_size = indices.len().min(20);
            let sample_step = indices.len() / sample_size;
            let dists: Vec<f64> = (0..sample_size)
                .map(|i| {
                    let j = indices[i * sample_step];
                    distance_fn(&items[cand].1, &items[j].1) as f64
                })
                .collect();

            let mean = dists.iter().sum::<f64>() / dists.len() as f64;
            let variance =
                dists.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / dists.len() as f64;

            if variance > best_spread {
                best_spread = variance;
                best_idx = cand;
            }
        }

        best_idx
    }

    /// Find the k nearest neighbors of a query point.
    ///
    /// # Arguments
    ///
    /// * `query` - The query feature vector
    /// * `k` - Number of neighbors to return
    /// * `distance_fn` - Distance metric (must be the same metric used at build time,
    ///   or at least compatible — the tree's partitioning relies on triangle inequality)
    ///
    /// # Returns
    ///
    /// Up to `k` results sorted by distance (closest first).
    /// Returns fewer than `k` if the tree has fewer points.
    pub fn knn<F>(&self, query: &[f32], k: usize, distance_fn: F) -> Vec<(T, f32)>
    where
        F: Fn(&[f32], &[f32]) -> f32,
    {
        if self.root == NULL || k == 0 {
            return Vec::new();
        }

        let mut heap: BinaryHeap<HeapEntry<T>> = BinaryHeap::with_capacity(k + 1);
        let mut tau = f32::INFINITY; // Current k-th nearest distance

        self.knn_search(self.root, query, k, &distance_fn, &mut heap, &mut tau);

        // Extract results sorted by distance (closest first)
        let mut results: Vec<(T, f32)> = heap.into_iter().map(|e| (e.data, e.distance)).collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        results
    }

    fn knn_search<F>(
        &self,
        node_idx: usize,
        query: &[f32],
        k: usize,
        distance_fn: &F,
        heap: &mut BinaryHeap<HeapEntry<T>>,
        tau: &mut f32,
    ) where
        F: Fn(&[f32], &[f32]) -> f32,
    {
        if node_idx == NULL {
            return;
        }

        let node = &self.nodes[node_idx];
        let dist = distance_fn(query, &node.point);

        // Consider this node as a candidate
        if heap.len() < k {
            heap.push(HeapEntry {
                data: node.data.clone(),
                distance: dist,
            });
            if heap.len() == k {
                *tau = heap.peek().unwrap().distance;
            }
        } else if dist < *tau {
            heap.pop(); // Remove the farthest
            heap.push(HeapEntry {
                data: node.data.clone(),
                distance: dist,
            });
            *tau = heap.peek().unwrap().distance;
        }

        // Pruning via triangle inequality
        if node.left == NULL && node.right == NULL {
            return; // Leaf
        }

        if dist < node.radius {
            // Query is inside the radius → left subtree is closer, search it first
            self.knn_search(node.left, query, k, distance_fn, heap, tau);
            // Only search right if the right subtree could contain closer points
            // Right points have d(vp, p) >= radius, so d(q, p) >= radius - dist
            // We can prune if radius - dist > tau (all right points are farther than tau)
            if node.radius - dist <= *tau {
                self.knn_search(node.right, query, k, distance_fn, heap, tau);
            }
        } else {
            // Query is outside the radius → right subtree is closer, search it first
            self.knn_search(node.right, query, k, distance_fn, heap, tau);
            // Only search left if left subtree could contain closer points
            // Left points have d(vp, p) < radius, so d(q, p) >= dist - radius
            // We can prune if dist - radius > tau
            if dist - node.radius <= *tau {
                self.knn_search(node.left, query, k, distance_fn, heap, tau);
            }
        }
    }

    /// Find all points within a given distance of a query point.
    ///
    /// # Arguments
    ///
    /// * `query` - The query feature vector
    /// * `radius` - Maximum distance (inclusive)
    /// * `distance_fn` - Distance metric
    ///
    /// # Returns
    ///
    /// All matching points as `(data, distance)` pairs, unsorted.
    pub fn range_search<F>(&self, query: &[f32], radius: f32, distance_fn: F) -> Vec<(T, f32)>
    where
        F: Fn(&[f32], &[f32]) -> f32,
    {
        let mut results = Vec::new();
        if self.root != NULL {
            self.range_search_recursive(self.root, query, radius, &distance_fn, &mut results);
        }
        results
    }

    fn range_search_recursive<F>(
        &self,
        node_idx: usize,
        query: &[f32],
        radius: f32,
        distance_fn: &F,
        results: &mut Vec<(T, f32)>,
    ) where
        F: Fn(&[f32], &[f32]) -> f32,
    {
        if node_idx == NULL {
            return;
        }

        let node = &self.nodes[node_idx];
        let dist = distance_fn(query, &node.point);

        if dist <= radius {
            results.push((node.data.clone(), dist));
        }

        // Prune using triangle inequality
        if dist - radius < node.radius && node.left != NULL {
            self.range_search_recursive(node.left, query, radius, distance_fn, results);
        }
        if dist + radius >= node.radius && node.right != NULL {
            self.range_search_recursive(node.right, query, radius, distance_fn, results);
        }
    }

    /// Returns the number of points in the tree.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns `true` if the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple Euclidean distance.
    fn euclidean(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    #[test]
    fn test_knn_small() {
        let points: Vec<(u32, Vec<f32>)> = vec![
            (0, vec![0.0, 0.0]),
            (1, vec![1.0, 0.0]),
            (2, vec![0.0, 1.0]),
            (3, vec![1.0, 1.0]),
            (4, vec![0.5, 0.5]),
        ];

        let tree = VpTree::build(points, euclidean);
        let results = tree.knn(&[0.0, 0.0], 2, euclidean);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0); // Exact match = distance 0
        assert!(results[0].1 < 1e-6);
        // Second closest is either (1,0), (0,1), or (0.5,0.5)
        // (0.5, 0.5) is at distance ~0.707, (1,0) and (0,1) at 1.0
        assert!(results[1].1 < 1.0 + 1e-6);
    }

    #[test]
    fn test_knn_matches_brute_force() {
        // Generate 200 random-ish 8d points
        let mut points: Vec<(u32, Vec<f32>)> = Vec::new();
        for i in 0..200 {
            let v: Vec<f32> = (0..8)
                .map(|j| ((i * 7 + j * 13) % 100) as f32 / 100.0)
                .collect();
            points.push((i as u32, v));
        }

        let tree = VpTree::build(points.clone(), euclidean);
        let query = &[0.5_f32, 0.3, 0.7, 0.1, 0.9, 0.2, 0.8, 0.4];
        let k = 10;

        // VP-Tree k-NN
        let vp_results = tree.knn(query, k, euclidean);

        // Brute-force k-NN
        let mut brute: Vec<(u32, f32)> = points
            .iter()
            .map(|(id, v)| (*id, euclidean(query, v)))
            .collect();
        brute.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let brute_top_k: Vec<(u32, f32)> = brute.into_iter().take(k).collect();

        assert_eq!(vp_results.len(), k);

        // Compare distances (IDs may differ if distances are equal — tie-breaking)
        for i in 0..k {
            assert!(
                (vp_results[i].1 - brute_top_k[i].1).abs() < 1e-4,
                "Distance mismatch at position {i}: vp={:.6} (id={}) brute={:.6} (id={})",
                vp_results[i].1,
                vp_results[i].0,
                brute_top_k[i].1,
                brute_top_k[i].0
            );
        }
    }

    #[test]
    fn test_range_search_empty() {
        let points: Vec<(u32, Vec<f32>)> = vec![(0, vec![0.0, 0.0]), (1, vec![10.0, 10.0])];

        let tree = VpTree::build(points, euclidean);
        // Very small radius around (5,5) — no points nearby
        let results = tree.range_search(&[5.0, 5.0], 0.1, euclidean);
        assert!(results.is_empty());
    }

    #[test]
    fn test_range_search_all() {
        let points: Vec<(u32, Vec<f32>)> = vec![
            (0, vec![0.0, 0.0]),
            (1, vec![1.0, 0.0]),
            (2, vec![0.0, 1.0]),
        ];

        let tree = VpTree::build(points, euclidean);
        // Infinite radius → all points
        let results = tree.range_search(&[0.0, 0.0], f32::MAX, euclidean);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_range_search_exact() {
        let points: Vec<(u32, Vec<f32>)> = vec![
            (0, vec![0.0, 0.0]),
            (1, vec![1.0, 0.0]),
            (2, vec![2.0, 0.0]),
            (3, vec![3.0, 0.0]),
        ];

        let tree = VpTree::build(points, euclidean);
        // Radius 0.0 → only exact matches
        let results = tree.range_search(&[1.0, 0.0], 0.0, euclidean);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 1);
    }

    #[test]
    fn test_clone_roundtrip() {
        // VpTree derives Clone + Serialize + Deserialize.
        // This test validates that a cloned tree produces identical results.
        let points: Vec<(u32, Vec<f32>)> = vec![
            (0, vec![0.0, 0.0]),
            (1, vec![1.0, 0.0]),
            (2, vec![0.0, 1.0]),
            (3, vec![1.0, 1.0]),
            (4, vec![0.5, 0.5]),
        ];

        let tree = VpTree::build(points, euclidean);
        let tree2 = tree.clone();

        // Same k-NN results
        let r1 = tree.knn(&[0.0, 0.0], 3, euclidean);
        let r2 = tree2.knn(&[0.0, 0.0], 3, euclidean);

        assert_eq!(r1.len(), r2.len());
        for i in 0..r1.len() {
            assert_eq!(r1[i].0, r2[i].0);
            assert!((r1[i].1 - r2[i].1).abs() < 1e-6);
        }
    }

    #[test]
    fn test_build_empty() {
        let tree: VpTree<u32> = VpTree::build(vec![], euclidean);
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);

        let results = tree.knn(&[0.0], 5, euclidean);
        assert!(results.is_empty());

        let range = tree.range_search(&[0.0], 100.0, euclidean);
        assert!(range.is_empty());
    }

    #[test]
    fn test_single_point() {
        let tree = VpTree::build(vec![(42, vec![1.0, 2.0])], euclidean);
        assert_eq!(tree.len(), 1);

        let results = tree.knn(&[1.0, 2.0], 5, euclidean);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 42);
        assert!(results[0].1 < 1e-6);
    }

    #[test]
    fn test_64d_vectors() {
        // Test with 64-dimensional vectors (matching Hilbert features)
        let mut points: Vec<(u32, Vec<f32>)> = Vec::new();
        for i in 0..100 {
            let v: Vec<f32> = (0..64)
                .map(|j| ((i * 17 + j * 31) % 1000) as f32 / 1000.0)
                .collect();
            points.push((i as u32, v));
        }

        let tree = VpTree::build(points.clone(), euclidean);
        let query: Vec<f32> = (0..64).map(|j| (j * 11 % 1000) as f32 / 1000.0).collect();

        let results = tree.knn(&query, 5, euclidean);
        assert_eq!(results.len(), 5);

        // Verify ordering
        for i in 1..results.len() {
            assert!(results[i].1 >= results[i - 1].1);
        }

        // Verify against brute force
        let mut brute: Vec<(u32, f32)> = points
            .iter()
            .map(|(id, v)| (*id, euclidean(&query, v)))
            .collect();
        brute.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        for i in 0..5 {
            assert_eq!(results[i].0, brute[i].0);
        }
    }
}
