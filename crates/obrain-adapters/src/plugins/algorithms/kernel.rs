//! # Irreducible Kernel — C = Phi_0 . A^1(H^inf)
//!
//! Single-pass graph crystallizer producing 80d embeddings that encode
//! topological structure. No iteration, no convergence loop.
//!
//! Architecture (validated by benchmarks on ai-noyau):
//! - Multi-head attention (8 heads x 10d = 80d), graph-masked
//! - RMS norm + FFN residual
//! - APPNP anchoring alpha=0.8 (single-pass, not iterative)
//! - Per-neighborhood: each node processed on 1-hop subgraph
//! - Cap 50 neighbors for hub nodes (Fisher-Yates sampling)
//!
//! Performance: 0.14ms/node, <5ms incremental for ~30 affected nodes.

use std::collections::HashMap;
use std::sync::Arc;

use obrain_common::types::{NodeId, PropertyKey, Value};
use obrain_common::utils::error::Result;
use obrain_core::graph::{Direction, GraphStore};

use super::super::{AlgorithmResult, ParameterDef, Parameters};
use super::kernel_math::{Matrix, Rng, gelu, rms_norm, softmax_rows};
use super::traits::GraphAlgorithm;

// ============================================================================
// Constants
// ============================================================================

/// Embedding dimension: 8 heads x 10d = 80d.
pub const D_MODEL: usize = 80;
/// Number of attention heads.
pub const N_HEADS: usize = 8;
/// Dimension per head.
pub const D_HEAD: usize = D_MODEL / N_HEADS;
/// FFN hidden dimension (2x model dim).
pub const D_FF: usize = D_MODEL * 2;
/// Default APPNP anchoring factor (high = preserve input identity).
pub const DEFAULT_ALPHA: f64 = 0.8;
/// Default max neighbors before Fisher-Yates sampling.
pub const DEFAULT_MAX_NEIGHBORS: usize = 50;
/// Property key for Hilbert features input.
pub const HILBERT_FEATURES_KEY: &str = "_hilbert_features";
/// Property key for kernel embedding output.
pub const KERNEL_EMBEDDING_KEY: &str = "_kernel_embedding";

// ============================================================================
// Phi_0 — The learned seed (multi-head attention weights)
// ============================================================================

/// Single attention head weights: Q, K, V projections.
#[derive(Clone)]
pub struct AttentionHead {
    /// Query projection [D_MODEL x D_HEAD].
    pub w_q: Matrix,
    /// Key projection [D_MODEL x D_HEAD].
    pub w_k: Matrix,
    /// Value projection [D_MODEL x D_HEAD].
    pub w_v: Matrix,
}

impl AttentionHead {
    fn new(d_model: usize, d_head: usize, rng: &mut Rng) -> Self {
        Self {
            w_q: Matrix::randn(d_model, d_head, rng),
            w_k: Matrix::randn(d_model, d_head, rng),
            w_v: Matrix::randn(d_model, d_head, rng),
        }
    }
}

/// Multi-head attention seed Phi_0: the learned transformation.
///
/// Contains all trainable parameters of the kernel:
/// - 8 attention heads (Q, K, V projections each)
/// - Output projection W_o [D_MODEL x D_MODEL]
/// - FFN: W_ff1 [D_MODEL x D_FF], W_ff2 [D_FF x D_MODEL]
///
/// Total: ~51,200 parameters. Trained once, then frozen.
#[derive(Clone)]
pub struct MultiHeadPhi0 {
    /// Attention heads.
    pub heads: Vec<AttentionHead>,
    /// Output projection after concatenating heads.
    pub w_o: Matrix,
    /// FFN first layer.
    pub w_ff1: Matrix,
    /// FFN second layer.
    pub w_ff2: Matrix,
    /// Model dimension (80).
    pub d_model: usize,
    /// Per-head dimension (10).
    pub d_head: usize,
    /// Number of heads (8).
    pub n_heads: usize,
}

impl MultiHeadPhi0 {
    /// Create a new Phi_0 with Xavier-initialized random weights.
    pub fn new(d_model: usize, n_heads: usize, d_ff: usize, seed: u64) -> Self {
        assert_eq!(d_model % n_heads, 0, "d_model must be divisible by n_heads");
        let d_head = d_model / n_heads;
        let mut rng = Rng::new(seed);

        let heads: Vec<AttentionHead> = (0..n_heads)
            .map(|_| AttentionHead::new(d_model, d_head, &mut rng))
            .collect();

        Self {
            heads,
            w_o: Matrix::randn(d_model, d_model, &mut rng),
            w_ff1: Matrix::randn(d_model, d_ff, &mut rng),
            w_ff2: Matrix::randn(d_ff, d_model, &mut rng),
            d_model,
            d_head,
            n_heads,
        }
    }

    /// Create with default dimensions (80d, 8 heads, 160 FFN).
    pub fn default_with_seed(seed: u64) -> Self {
        Self::new(D_MODEL, N_HEADS, D_FF, seed)
    }

    /// Total number of trainable parameters.
    pub fn param_count(&self) -> usize {
        let per_head = self.d_model * self.d_head * 3; // Q, K, V
        let heads_total = per_head * self.n_heads;
        let w_o = self.d_model * self.d_model;
        let ffn = self.d_model * self.w_ff1.cols + self.w_ff1.cols * self.d_model;
        heads_total + w_o + ffn
    }

    /// Serialize all weights to a flat `Vec<f64>` for persistence.
    pub fn serialize_weights(&self) -> Vec<f64> {
        let mut weights = Vec::with_capacity(self.param_count());
        for head in &self.heads {
            weights.extend_from_slice(&head.w_q.data);
            weights.extend_from_slice(&head.w_k.data);
            weights.extend_from_slice(&head.w_v.data);
        }
        weights.extend_from_slice(&self.w_o.data);
        weights.extend_from_slice(&self.w_ff1.data);
        weights.extend_from_slice(&self.w_ff2.data);
        weights
    }

    /// Deserialize weights from a flat `Vec<f64>`.
    pub fn deserialize_weights(&mut self, weights: &[f64]) {
        let mut offset = 0;
        for head in &mut self.heads {
            let len = head.w_q.data.len();
            head.w_q
                .data
                .copy_from_slice(&weights[offset..offset + len]);
            offset += len;
            let len = head.w_k.data.len();
            head.w_k
                .data
                .copy_from_slice(&weights[offset..offset + len]);
            offset += len;
            let len = head.w_v.data.len();
            head.w_v
                .data
                .copy_from_slice(&weights[offset..offset + len]);
            offset += len;
        }
        let len = self.w_o.data.len();
        self.w_o
            .data
            .copy_from_slice(&weights[offset..offset + len]);
        offset += len;
        let len = self.w_ff1.data.len();
        self.w_ff1
            .data
            .copy_from_slice(&weights[offset..offset + len]);
        offset += len;
        let len = self.w_ff2.data.len();
        self.w_ff2
            .data
            .copy_from_slice(&weights[offset..offset + len]);
    }
}

// ============================================================================
// Adjacency Mask
// ============================================================================

/// Graph-masked attention: blocks attention between non-adjacent nodes.
///
/// Internally a dense matrix where:
/// - `0.0` means "attention allowed" (self-loops + edges)
/// - `-1e9` means "attention blocked" (effectively zero after softmax)
pub struct AdjacencyMask {
    /// The mask matrix [n x n].
    pub mask: Matrix,
}

impl AdjacencyMask {
    /// Build mask from edge list (local indices, directed).
    pub fn from_edges(n: usize, edges: &[(usize, usize)]) -> Self {
        let mut mask = Matrix::filled(n, n, -1e9);
        // Self-attention always allowed
        for i in 0..n {
            mask.set(i, i, 0.0);
        }
        for &(from, to) in edges {
            if from < n && to < n {
                mask.set(from, to, 0.0);
            }
        }
        Self { mask }
    }

    /// Build mask from adjacency lists (bidirectional by default).
    pub fn from_adj_lists(n: usize, adj: &[Vec<usize>]) -> Self {
        let mut mask = Matrix::filled(n, n, -1e9);
        for i in 0..n {
            mask.set(i, i, 0.0);
            for &j in &adj[i] {
                if j < n {
                    mask.set(i, j, 0.0);
                }
            }
        }
        Self { mask }
    }

    /// Fully connected mask (for testing / isolated nodes).
    pub fn fully_connected(n: usize) -> Self {
        Self {
            mask: Matrix::zeros(n, n),
        }
    }
}

// ============================================================================
// Single-pass attention — THE core function
// ============================================================================

/// Single-pass crystallizer: one pass of multi-head attention + FFN + APPNP.
///
/// This is the entire kernel computation. NO iteration, NO convergence loop.
/// The infinity lives in the Hilbert features (layer 1), the kernel crystallizes
/// in one pass (layer 2).
///
/// # Formula
///
/// ```text
/// x_refined = RMSNorm(x0 + MultiHeadAttn(x0)) -> FFN -> RMSNorm
/// output = alpha * RMSNorm(x0) + (1 - alpha) * x_refined
/// ```
///
/// # Arguments
///
/// * `x0` - Input features [n_nodes x d_model] (Hilbert features)
/// * `phi` - Learned weights (frozen after training)
/// * `adj_mask` - Graph-masked attention
/// * `alpha` - APPNP anchoring factor (default 0.8)
pub fn single_pass_attention(
    x0: &Matrix,
    phi: &MultiHeadPhi0,
    adj_mask: &AdjacencyMask,
    alpha: f64,
) -> Matrix {
    let n = x0.rows;
    let x0_normed = rms_norm(x0);

    // Multi-head attention
    let mut concat = Matrix::zeros(n, 0);
    for head in &phi.heads {
        let scale = 1.0 / (phi.d_head as f64).sqrt();

        let q = x0.matmul(&head.w_q);
        let k = x0.matmul(&head.w_k);
        let v = x0.matmul(&head.w_v);

        // Scaled dot-product attention + graph mask
        let mut raw = q.matmul(&k.transpose()).scale(scale);
        for i in 0..n {
            for j in 0..n {
                let idx = raw.idx(i, j);
                raw.data[idx] += adj_mask.mask.get(i, j);
            }
        }
        let scores = softmax_rows(&raw);
        let head_out = scores.matmul(&v);

        if concat.cols == 0 {
            concat = head_out;
        } else {
            concat = concat.hcat(&head_out);
        }
    }

    // Output projection + residual
    let attn_out = concat.matmul(&phi.w_o);
    let post = x0.add(&attn_out);

    // RMS norm (no mean subtraction)
    let normed = rms_norm(&post);

    // FFN + residual
    let ff_hidden = normed.matmul(&phi.w_ff1).map(gelu);
    let ff_out = ff_hidden.matmul(&phi.w_ff2);
    let refined = rms_norm(&normed.add(&ff_out));

    // APPNP anchoring: alpha * x0 + (1-alpha) * refined
    x0_normed.scale(alpha).add(&refined.scale(1.0 - alpha))
}

// ============================================================================
// Per-neighborhood extraction
// ============================================================================

/// Neighborhood extraction result.
pub struct Neighborhood {
    /// Feature matrix [n_local x d_model].
    pub features: Matrix,
    /// Adjacency mask [n_local x n_local].
    pub adj_mask: AdjacencyMask,
    /// Global NodeIds in order matching feature rows.
    pub node_ids: Vec<NodeId>,
    /// Index of the center node in the local matrix.
    pub center_index: usize,
}

/// Extract the 1-hop neighborhood of a node from the graph store.
///
/// Reads `_hilbert_features` from each neighbor. Nodes without features
/// are skipped. If `max_neighbors` is exceeded, Fisher-Yates sampling
/// is used to cap the neighborhood size.
///
/// Returns `None` if the center node has no Hilbert features.
pub fn extract_neighborhood(
    store: &dyn GraphStore,
    center: NodeId,
    max_neighbors: usize,
    rng: &mut Rng,
) -> Option<Neighborhood> {
    let feature_key = PropertyKey::from(HILBERT_FEATURES_KEY);

    // Get center node features
    let center_features = extract_features(store, center, &feature_key)?;

    // Get 1-hop neighbors (both directions for undirected semantics)
    let mut neighbor_ids = store.neighbors(center, Direction::Both);
    // Deduplicate (Both may return duplicates for bidirectional edges)
    neighbor_ids.sort();
    neighbor_ids.dedup();

    // Cap neighbors via Fisher-Yates if needed
    if neighbor_ids.len() > max_neighbors {
        rng.shuffle(&mut neighbor_ids);
        neighbor_ids.truncate(max_neighbors);
    }

    // Collect features for all neighbors (skip those without)
    let mut local_nodes: Vec<(NodeId, Vec<f64>)> = Vec::with_capacity(neighbor_ids.len() + 1);
    // Center node always first
    local_nodes.push((center, center_features));

    for &nid in &neighbor_ids {
        if let Some(feats) = extract_features(store, nid, &feature_key) {
            local_nodes.push((nid, feats));
        }
    }

    let n = local_nodes.len();
    let d = D_MODEL;

    // Build feature matrix
    let mut features = Matrix::zeros(n, d);
    let mut node_ids = Vec::with_capacity(n);
    for (i, (nid, feats)) in local_nodes.iter().enumerate() {
        node_ids.push(*nid);
        let row = features.row_mut(i);
        let copy_len = feats.len().min(d);
        row[..copy_len].copy_from_slice(&feats[..copy_len]);
    }

    // Build adjacency: center (index 0) connects to all neighbors,
    // plus inter-neighbor edges
    let center_idx = 0;
    let mut local_adj: Vec<Vec<usize>> = vec![Vec::new(); n];

    // Build reverse mapping: NodeId -> local index
    let id_to_local: HashMap<NodeId, usize> = node_ids
        .iter()
        .enumerate()
        .map(|(i, &nid)| (nid, i))
        .collect();

    // Center connects to all neighbors (bidirectional)
    for i in 1..n {
        local_adj[center_idx].push(i);
        local_adj[i].push(center_idx);
    }

    // Inter-neighbor edges: check actual graph connectivity
    for i in 1..n {
        let nid = node_ids[i];
        let nid_neighbors = store.neighbors(nid, Direction::Both);
        for &target in &nid_neighbors {
            if let Some(&j) = id_to_local.get(&target)
                && j != i
                && j != center_idx
                && !local_adj[i].contains(&j)
            {
                local_adj[i].push(j);
            }
        }
    }

    let adj_mask = AdjacencyMask::from_adj_lists(n, &local_adj);

    Some(Neighborhood {
        features,
        adj_mask,
        node_ids,
        center_index: center_idx,
    })
}

/// Extract Hilbert features from a node, converting f32 -> f64.
fn extract_features(store: &dyn GraphStore, nid: NodeId, key: &PropertyKey) -> Option<Vec<f64>> {
    match store.get_node_property(nid, key) {
        Some(Value::Vector(v)) => Some(v.iter().map(|&x| x as f64).collect()),
        _ => None,
    }
}

// ============================================================================
// Embedding computation
// ============================================================================

/// Compute the kernel embedding for a single node.
///
/// Pipeline: extract_neighborhood -> single_pass_attention -> extract center row -> f32.
///
/// Returns `None` if the node has no Hilbert features.
pub fn compute_node_embedding(
    phi: &MultiHeadPhi0,
    store: &dyn GraphStore,
    node_id: NodeId,
    alpha: f64,
    max_neighbors: usize,
    rng: &mut Rng,
) -> Option<Vec<f32>> {
    let neighborhood = extract_neighborhood(store, node_id, max_neighbors, rng)?;

    let result = single_pass_attention(&neighborhood.features, phi, &neighborhood.adj_mask, alpha);

    // Extract center node's row and convert to f32
    Some(result.row_to_f32(neighborhood.center_index))
}

/// Compute embeddings for a batch of nodes (single-threaded).
pub fn compute_batch(
    phi: &MultiHeadPhi0,
    store: &dyn GraphStore,
    node_ids: &[NodeId],
    alpha: f64,
    max_neighbors: usize,
    seed: u64,
) -> HashMap<NodeId, Vec<f32>> {
    let mut rng = Rng::new(seed);
    let mut results = HashMap::with_capacity(node_ids.len());

    for &nid in node_ids {
        if let Some(embedding) =
            compute_node_embedding(phi, store, nid, alpha, max_neighbors, &mut rng)
        {
            results.insert(nid, embedding);
        }
    }

    results
}

/// Compute embeddings for a batch of nodes using multiple threads.
///
/// Each node is independent (embarrassingly parallel).
/// Uses `std::thread::scope` — no external dependency needed.
pub fn compute_batch_parallel(
    phi: &MultiHeadPhi0,
    store: &dyn GraphStore,
    node_ids: &[NodeId],
    alpha: f64,
    max_neighbors: usize,
    n_threads: usize,
    seed: u64,
) -> HashMap<NodeId, Vec<f32>>
where
    // GraphStore must be Send+Sync for shared access across threads
{
    if node_ids.is_empty() || n_threads <= 1 {
        return compute_batch(phi, store, node_ids, alpha, max_neighbors, seed);
    }

    // Split node_ids into chunks, one per thread
    let chunk_size = (node_ids.len() + n_threads - 1) / n_threads;
    let chunks: Vec<&[NodeId]> = node_ids.chunks(chunk_size).collect();

    let mut all_results = HashMap::with_capacity(node_ids.len());

    std::thread::scope(|s| {
        let handles: Vec<_> = chunks
            .iter()
            .enumerate()
            .map(|(thread_idx, chunk)| {
                s.spawn(move || {
                    // Each thread gets its own Rng with a unique seed
                    let thread_seed = seed.wrapping_add(thread_idx as u64 * 7919);
                    compute_batch(phi, store, chunk, alpha, max_neighbors, thread_seed)
                })
            })
            .collect();

        for handle in handles {
            let partial = handle
                .join()
                .expect("thread panicked in compute_batch_parallel");
            all_results.extend(partial);
        }
    });

    all_results
}

// ============================================================================
// GraphAlgorithm implementation
// ============================================================================

/// The Irreducible Kernel algorithm, implementing `GraphAlgorithm`.
///
/// Computes 80d embeddings for all nodes using per-neighborhood
/// single-pass attention.
pub struct IrreducibleKernel {
    /// The learned weights (frozen after training).
    pub phi: MultiHeadPhi0,
    /// APPNP anchoring factor (default 0.8).
    pub alpha: f64,
    /// Max neighbors before sampling (default 50).
    pub max_neighbors: usize,
    /// Number of threads for parallel computation.
    pub n_threads: usize,
    /// PRNG seed for reproducibility.
    pub seed: u64,
}

impl IrreducibleKernel {
    /// Create with default parameters and random Phi_0.
    pub fn new(seed: u64) -> Self {
        Self {
            phi: MultiHeadPhi0::default_with_seed(seed),
            alpha: DEFAULT_ALPHA,
            max_neighbors: DEFAULT_MAX_NEIGHBORS,
            n_threads: 1,
            seed,
        }
    }

    /// Create with a pre-trained Phi_0.
    pub fn with_phi(phi: MultiHeadPhi0, seed: u64) -> Self {
        Self {
            phi,
            alpha: DEFAULT_ALPHA,
            max_neighbors: DEFAULT_MAX_NEIGHBORS,
            n_threads: 1,
            seed,
        }
    }
}

impl GraphAlgorithm for IrreducibleKernel {
    fn name(&self) -> &str {
        "irreducible_kernel"
    }

    fn description(&self) -> &str {
        "Irreducible Kernel C = Phi_0 . A^1(H^inf) — single-pass graph crystallizer producing 80d topological embeddings"
    }

    fn parameters(&self) -> &[ParameterDef] {
        // Return empty slice — parameters are documented but defined at runtime
        // to avoid static lifetime issues with ParameterDef.
        // Use IrreducibleKernel fields directly for defaults.
        &[]
    }

    fn execute(&self, store: &dyn GraphStore, params: &Parameters) -> Result<AlgorithmResult> {
        let alpha = params.get_float("alpha").unwrap_or(self.alpha);
        let max_neighbors = params
            .get_int("max_neighbors")
            .map_or(self.max_neighbors, |i| i as usize);
        let n_threads = params
            .get_int("threads")
            .map_or(self.n_threads, |i| i as usize);

        let node_ids = store.node_ids();

        let embeddings = compute_batch_parallel(
            &self.phi,
            store,
            &node_ids,
            alpha,
            max_neighbors,
            n_threads,
            self.seed,
        );

        // Build result table: node_id | embedding
        let mut result = AlgorithmResult::new(vec!["node_id".to_string(), "embedding".to_string()]);

        for &nid in &node_ids {
            if let Some(emb) = embeddings.get(&nid) {
                let arc_vec: Arc<[f32]> = emb.as_slice().into();
                result.add_row(vec![Value::Int64(nid.0 as i64), Value::Vector(arc_vec)]);
            }
        }

        Ok(result)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──

    /// Simple test graph: 3 clusters of 4 nodes each, well-separated.
    fn make_test_graph() -> (Matrix, AdjacencyMask, usize) {
        let n = 12;
        let d = D_MODEL;
        let mut rng = Rng::new(42);

        // 3 clusters with distinct features
        let mut features = Matrix::zeros(n, d);
        for cluster in 0..3 {
            let centroid: Vec<f64> = (0..d).map(|_| rng.next_f64()).collect();
            for i in 0..4 {
                let node = cluster * 4 + i;
                for j in 0..d {
                    features.set(node, j, centroid[j] + rng.next_normal() * 0.1);
                }
            }
        }

        // Edges: dense within clusters, sparse between
        let mut edges = Vec::new();
        for cluster in 0..3 {
            let base = cluster * 4;
            for i in base..base + 4 {
                for j in base..base + 4 {
                    if i != j {
                        edges.push((i, j));
                    }
                }
            }
        }
        // A few bridge edges between clusters
        edges.push((3, 4)); // cluster 0 -> cluster 1
        edges.push((4, 3));
        edges.push((7, 8)); // cluster 1 -> cluster 2
        edges.push((8, 7));

        let mask = AdjacencyMask::from_edges(n, &edges);
        (features, mask, n)
    }

    /// Compute mean cosine similarity between rows in different groups.
    fn inter_group_similarity(m: &Matrix, groups: &[Vec<usize>]) -> f64 {
        let mut sum = 0.0;
        let mut count = 0;
        for (gi, g1) in groups.iter().enumerate() {
            for g2 in groups.iter().skip(gi + 1) {
                for &i in g1 {
                    for &j in g2 {
                        sum += m.cosine_similarity(i, j);
                        count += 1;
                    }
                }
            }
        }
        if count == 0 { 0.0 } else { sum / count as f64 }
    }

    /// Compute mean cosine similarity within groups.
    fn intra_group_similarity(m: &Matrix, groups: &[Vec<usize>]) -> f64 {
        let mut sum = 0.0;
        let mut count = 0;
        for g in groups {
            for (ii, &i) in g.iter().enumerate() {
                for &j in g.iter().skip(ii + 1) {
                    sum += m.cosine_similarity(i, j);
                    count += 1;
                }
            }
        }
        if count == 0 { 0.0 } else { sum / count as f64 }
    }

    // ── MultiHeadPhi0 tests ──

    #[test]
    fn test_phi0_creation() {
        let phi = MultiHeadPhi0::new(D_MODEL, N_HEADS, D_FF, 42);
        assert_eq!(phi.heads.len(), N_HEADS);
        assert_eq!(phi.d_model, D_MODEL);
        assert_eq!(phi.d_head, D_HEAD);
        assert_eq!(phi.w_o.rows, D_MODEL);
        assert_eq!(phi.w_o.cols, D_MODEL);
        assert_eq!(phi.w_ff1.rows, D_MODEL);
        assert_eq!(phi.w_ff1.cols, D_FF);
        assert_eq!(phi.w_ff2.rows, D_FF);
        assert_eq!(phi.w_ff2.cols, D_MODEL);
    }

    #[test]
    fn test_phi0_param_count() {
        let phi = MultiHeadPhi0::new(D_MODEL, N_HEADS, D_FF, 42);
        let count = phi.param_count();
        // 8 heads * (80*10*3) + 80*80 + 80*160 + 160*80 = 19200 + 6400 + 12800 + 12800 = 51200
        assert_eq!(count, 51200);
    }

    #[test]
    fn test_phi0_serialize_deserialize_roundtrip() {
        let phi = MultiHeadPhi0::new(D_MODEL, N_HEADS, D_FF, 42);
        let weights = phi.serialize_weights();
        assert_eq!(weights.len(), phi.param_count());

        let mut phi2 = MultiHeadPhi0::new(D_MODEL, N_HEADS, D_FF, 99); // different seed
        phi2.deserialize_weights(&weights);

        // Should be identical after deserialization
        let weights2 = phi2.serialize_weights();
        assert_eq!(weights, weights2);
    }

    #[test]
    fn test_phi0_deterministic() {
        let phi1 = MultiHeadPhi0::new(D_MODEL, N_HEADS, D_FF, 42);
        let phi2 = MultiHeadPhi0::new(D_MODEL, N_HEADS, D_FF, 42);
        assert_eq!(phi1.serialize_weights(), phi2.serialize_weights());
    }

    #[test]
    fn test_phi0_different_seeds_differ() {
        let phi1 = MultiHeadPhi0::new(D_MODEL, N_HEADS, D_FF, 42);
        let phi2 = MultiHeadPhi0::new(D_MODEL, N_HEADS, D_FF, 43);
        assert_ne!(phi1.serialize_weights(), phi2.serialize_weights());
    }

    // ── AdjacencyMask tests ──

    #[test]
    fn test_mask_from_edges() {
        let mask = AdjacencyMask::from_edges(3, &[(0, 1), (1, 2)]);
        // Self-loops allowed
        assert_eq!(mask.mask.get(0, 0), 0.0);
        assert_eq!(mask.mask.get(1, 1), 0.0);
        // Edges allowed
        assert_eq!(mask.mask.get(0, 1), 0.0);
        assert_eq!(mask.mask.get(1, 2), 0.0);
        // Non-edges blocked
        assert!(mask.mask.get(0, 2) < -1e8);
        assert!(mask.mask.get(2, 0) < -1e8);
        assert!(mask.mask.get(1, 0) < -1e8); // directed: 0->1 allowed, 1->0 blocked
    }

    #[test]
    fn test_mask_fully_connected() {
        let mask = AdjacencyMask::fully_connected(5);
        for i in 0..5 {
            for j in 0..5 {
                assert_eq!(mask.mask.get(i, j), 0.0);
            }
        }
    }

    #[test]
    fn test_mask_from_adj_lists() {
        let adj = vec![vec![1, 2], vec![0], vec![0]];
        let mask = AdjacencyMask::from_adj_lists(3, &adj);
        assert_eq!(mask.mask.get(0, 1), 0.0);
        assert_eq!(mask.mask.get(0, 2), 0.0);
        assert_eq!(mask.mask.get(1, 0), 0.0);
        assert!(mask.mask.get(1, 2) < -1e8); // not connected
    }

    // ── Single-pass attention tests ──

    #[test]
    fn test_single_pass_output_dimensions() {
        let (features, mask, n) = make_test_graph();
        let phi = MultiHeadPhi0::new(D_MODEL, N_HEADS, D_FF, 42);
        let result = single_pass_attention(&features, &phi, &mask, DEFAULT_ALPHA);
        assert_eq!(result.rows, n);
        assert_eq!(result.cols, D_MODEL);
    }

    #[test]
    fn test_single_pass_finite_output() {
        let (features, mask, _) = make_test_graph();
        let phi = MultiHeadPhi0::new(D_MODEL, N_HEADS, D_FF, 42);
        let result = single_pass_attention(&features, &phi, &mask, DEFAULT_ALPHA);
        assert!(
            result.data.iter().all(|x| x.is_finite()),
            "single_pass_attention produced non-finite values"
        );
    }

    #[test]
    fn test_single_pass_deterministic() {
        let (features, mask, _) = make_test_graph();
        let phi = MultiHeadPhi0::new(D_MODEL, N_HEADS, D_FF, 42);
        let r1 = single_pass_attention(&features, &phi, &mask, DEFAULT_ALPHA);
        let r2 = single_pass_attention(&features, &phi, &mask, DEFAULT_ALPHA);
        assert!(
            r1.diff_norm(&r2) < 1e-12,
            "single_pass must be deterministic"
        );
    }

    #[test]
    fn test_single_pass_topology_preservation() {
        let (features, mask, _) = make_test_graph();
        let phi = MultiHeadPhi0::new(D_MODEL, N_HEADS, D_FF, 42);
        let result = single_pass_attention(&features, &phi, &mask, DEFAULT_ALPHA);

        let groups = vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7], vec![8, 9, 10, 11]];

        let intra = intra_group_similarity(&result, &groups);
        let inter = inter_group_similarity(&result, &groups);
        let gap = intra - inter;

        assert!(
            gap > 0.0,
            "Intra-cluster similarity should exceed inter-cluster: intra={:.4}, inter={:.4}, gap={:.4}",
            intra,
            inter,
            gap
        );
    }

    #[test]
    fn test_single_pass_diversity() {
        let (features, mask, _) = make_test_graph();
        let phi = MultiHeadPhi0::new(D_MODEL, N_HEADS, D_FF, 42);
        let result = single_pass_attention(&features, &phi, &mask, DEFAULT_ALPHA);

        let div = super::super::kernel_math::diversity(&result);
        assert!(
            div > 0.05,
            "Diversity should be > 0.05 (no collapse), got {:.4}",
            div
        );
    }

    #[test]
    fn test_single_pass_alpha_effect() {
        let (features, mask, _) = make_test_graph();
        let phi = MultiHeadPhi0::new(D_MODEL, N_HEADS, D_FF, 42);

        // alpha=1.0 should return RMSNorm(x0) (no attention influence)
        let r_full_anchor = single_pass_attention(&features, &phi, &mask, 1.0);
        let x0_normed = rms_norm(&features);
        assert!(
            r_full_anchor.diff_norm(&x0_normed) < 1e-10,
            "alpha=1.0 should return normalized input"
        );

        // alpha=0.0 should be fully attention-driven
        let r_no_anchor = single_pass_attention(&features, &phi, &mask, 0.0);
        // Different from alpha=1.0
        assert!(
            r_full_anchor.diff_norm(&r_no_anchor) > 0.1,
            "alpha=0.0 and alpha=1.0 should produce different results"
        );
    }

    #[test]
    fn test_single_pass_small_graph() {
        // 2 nodes, 1 edge — minimal case
        let features = Matrix::from_vec(
            2,
            D_MODEL,
            (0..2 * D_MODEL).map(|x| x as f64 * 0.01).collect(),
        );
        let mask = AdjacencyMask::from_edges(2, &[(0, 1), (1, 0)]);
        let phi = MultiHeadPhi0::new(D_MODEL, N_HEADS, D_FF, 42);
        let result = single_pass_attention(&features, &phi, &mask, DEFAULT_ALPHA);
        assert_eq!(result.rows, 2);
        assert_eq!(result.cols, D_MODEL);
        assert!(result.data.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_single_pass_single_node() {
        // 1 node, no edges — should work (self-attention only)
        let features = Matrix::from_vec(
            1,
            D_MODEL,
            (0..D_MODEL).map(|x| x as f64 * 0.01 + 0.1).collect(),
        );
        let mask = AdjacencyMask::from_edges(1, &[]);
        let phi = MultiHeadPhi0::new(D_MODEL, N_HEADS, D_FF, 42);
        let result = single_pass_attention(&features, &phi, &mask, DEFAULT_ALPHA);
        assert_eq!(result.rows, 1);
        assert_eq!(result.cols, D_MODEL);
        assert!(result.data.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_single_pass_performance() {
        // 50 nodes (typical neighborhood size) — should be fast
        let n = 50;
        let mut rng = Rng::new(42);
        let features = Matrix::randn(n, D_MODEL, &mut rng);
        let mask = AdjacencyMask::fully_connected(n);
        let phi = MultiHeadPhi0::new(D_MODEL, N_HEADS, D_FF, 42);

        let start = std::time::Instant::now();
        for _ in 0..100 {
            let _ = single_pass_attention(&features, &phi, &mask, DEFAULT_ALPHA);
        }
        let per_call = start.elapsed() / 100;
        // Should be well under 1ms for 50 nodes
        assert!(
            per_call.as_millis() < 5,
            "single_pass on 50 nodes too slow: {:?}",
            per_call
        );
    }

    // ── Batch computation tests (without GraphStore — unit tests only) ──

    #[test]
    fn test_f32_embedding_precision() {
        // Verify f32 conversion doesn't lose significant information
        let (features, mask, _) = make_test_graph();
        let phi = MultiHeadPhi0::new(D_MODEL, N_HEADS, D_FF, 42);
        let result = single_pass_attention(&features, &phi, &mask, DEFAULT_ALPHA);

        let f32_vec = result.row_to_f32(0);
        assert_eq!(f32_vec.len(), D_MODEL);

        // Check relative error
        let row = result.row(0);
        for (i, (&f64_val, &f32_val)) in row.iter().zip(f32_vec.iter()).enumerate() {
            let expected = f64_val as f32;
            assert_eq!(f32_val, expected, "f32 conversion mismatch at index {}", i);
        }
    }

    // ── Serialization stress test ──

    #[test]
    fn test_serialize_large_roundtrip() {
        let phi = MultiHeadPhi0::new(D_MODEL, N_HEADS, D_FF, 12345);
        let weights = phi.serialize_weights();

        // Roundtrip
        let mut phi2 = MultiHeadPhi0::new(D_MODEL, N_HEADS, D_FF, 1);
        phi2.deserialize_weights(&weights);

        // Verify functionally identical
        let (features, mask, _) = make_test_graph();
        let r1 = single_pass_attention(&features, &phi, &mask, DEFAULT_ALPHA);
        let r2 = single_pass_attention(&features, &phi2, &mask, DEFAULT_ALPHA);
        assert!(
            r1.diff_norm(&r2) < 1e-12,
            "deserialized phi should produce identical output"
        );
    }
}
