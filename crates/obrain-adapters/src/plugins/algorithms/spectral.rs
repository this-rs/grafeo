//! Spectral embedding algorithms based on the normalized graph Laplacian.
//!
//! Computes low-dimensional embeddings by extracting the smallest non-trivial
//! eigenvectors of the normalized Laplacian matrix:
//!
//! ```text
//! L_norm = I - D^{-1/2} W D^{-1/2}
//! ```
//!
//! where `W` is the (weighted) adjacency matrix and `D` is the diagonal degree
//! matrix. The eigenvectors corresponding to the smallest eigenvalues capture
//! the global connectivity structure of the graph.
//!
//! Uses power iteration with deflation on the *similarity matrix*
//! `M = D^{-1/2} W D^{-1/2}` (whose largest eigenvectors = smallest of `L_norm`).
//!
//! ## Usage
//!
//! ```no_run
//! use obrain_adapters::plugins::algorithms::spectral_embedding;
//! use obrain_substrate::SubstrateStore;
//!
//! let store = SubstrateStore::open_tempfile().unwrap();
//! // ... populate graph ...
//! let embeddings = spectral_embedding(&store, 4, None);
//! // embeddings: HashMap<NodeId, Vec<f64>> with 4 dimensions per node
//! ```

use std::collections::HashMap;
use std::sync::OnceLock;

use obrain_common::types::{NodeId, PropertyKey, Value};
use obrain_common::utils::error::Result;
#[cfg(test)]
use obrain_core::graph::GraphStoreMut;
use obrain_core::graph::{Direction, GraphStore};
#[cfg(test)]
use obrain_substrate::SubstrateStore;

use super::super::{AlgorithmResult, ParameterDef, ParameterType, Parameters};
use super::traits::GraphAlgorithm;

// ============================================================================
// Public API
// ============================================================================

/// Result of spectral embedding computation.
#[derive(Debug, Clone)]
pub struct SpectralEmbeddingResult {
    /// Embedding vectors per node: NodeId → `Vec<f64>` of length `dimensions`.
    pub embeddings: HashMap<NodeId, Vec<f64>>,
    /// Number of dimensions.
    pub dimensions: usize,
}

/// Compute spectral embedding for all nodes in a graph.
///
/// Extracts the `dimensions` smallest non-trivial eigenvectors of the
/// normalized Laplacian via power iteration with deflation.
///
/// # Arguments
///
/// * `store` - The graph store (supports `GraphProjection` for filtered views)
/// * `dimensions` - Number of embedding dimensions (typically 2, 4, 6, or 8)
/// * `weight_property` - Optional edge property name to use as weight (default: 1.0)
///
/// # Returns
///
/// A `SpectralEmbeddingResult` with normalized embeddings in [0, 1] per dimension.
///
/// # Complexity
///
/// O(iterations × (V + E)) per eigenvector, with `iterations` ≤ 200.
pub fn spectral_embedding(
    store: &dyn GraphStore,
    dimensions: usize,
    weight_property: Option<&str>,
) -> SpectralEmbeddingResult {
    let node_ids = store.node_ids();
    let n = node_ids.len();

    if n <= 1 {
        let embeddings = node_ids
            .iter()
            .map(|&nid| (nid, vec![0.5; dimensions]))
            .collect();
        return SpectralEmbeddingResult {
            embeddings,
            dimensions,
        };
    }

    // Build dense index: NodeId ↔ usize
    let id_to_idx: HashMap<NodeId, usize> = node_ids
        .iter()
        .enumerate()
        .map(|(i, &nid)| (nid, i))
        .collect();

    // Build weighted adjacency list
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for (idx, &nid) in node_ids.iter().enumerate() {
        for (neighbor, edge_id) in store.edges_from(nid, Direction::Outgoing) {
            if let Some(&j) = id_to_idx.get(&neighbor) {
                let w = weight_property
                    .and_then(|prop| store.get_edge_property(edge_id, &PropertyKey::from(prop)))
                    .and_then(|v| match v {
                        Value::Float64(f) => Some(f),
                        Value::Int64(i) => Some(i as f64),
                        _ => None,
                    })
                    .unwrap_or(1.0);
                adj[idx].push((j, w));
            }
        }
    }

    // Compute eigenvectors
    let eigenvectors = spectral_eigenvectors(&adj, dimensions);

    // Build result
    let mut embeddings = HashMap::with_capacity(n);
    for (idx, &nid) in node_ids.iter().enumerate() {
        let mut vec = Vec::with_capacity(dimensions);
        for ev in &eigenvectors {
            vec.push(ev[idx]);
        }
        embeddings.insert(nid, vec);
    }

    SpectralEmbeddingResult {
        embeddings,
        dimensions,
    }
}

// ============================================================================
// Core Linear Algebra
// ============================================================================

/// Extract `k` smallest non-trivial eigenvectors of the normalized Laplacian.
///
/// Works on the similarity matrix M = D^{-1/2} W D^{-1/2} and finds its
/// largest eigenvectors (excluding the trivial one), which correspond to
/// the smallest non-trivial eigenvectors of L_norm = I - M.
///
/// Returns `k` vectors, each normalized to [0, 1].
fn spectral_eigenvectors(adj: &[Vec<(usize, f64)>], k: usize) -> Vec<Vec<f64>> {
    let n = adj.len();
    if n <= 1 {
        return vec![vec![0.5; n]; k];
    }

    // Compute weighted degree: D_ii = Σ_j W_ij
    let degrees: Vec<f64> = adj
        .iter()
        .map(|neighbors| neighbors.iter().map(|&(_, w)| w).sum::<f64>())
        .collect();

    // Check for isolated nodes
    let min_deg = degrees.iter().copied().fold(f64::INFINITY, f64::min);
    if min_deg < 1e-10 {
        return vec![vec![0.5; n]; k];
    }

    // D^{-1/2}
    let d_inv_sqrt: Vec<f64> = degrees.iter().map(|&d| 1.0 / d.sqrt()).collect();

    // Matrix-vector product: M * v = D^{-1/2} W D^{-1/2} v
    let matvec = |v: &[f64]| -> Vec<f64> {
        let mut result = vec![0.0f64; n];
        for i in 0..n {
            let wi = d_inv_sqrt[i] * v[i];
            for &(j, w) in &adj[i] {
                result[j] += d_inv_sqrt[j] * wi * w;
            }
        }
        result
    };

    // Trivial eigenvector: D^{1/2} * 1 (normalized)
    let trivial: Vec<f64> = degrees.iter().map(|&d| d.sqrt()).collect();
    let trivial_norm = vec_norm(&trivial);
    let trivial_unit: Vec<f64> = trivial.iter().map(|&x| x / trivial_norm).collect();

    // Extract k eigenvectors via successive deflation
    let mut deflate_vecs: Vec<Vec<f64>> = vec![trivial_unit];
    let mut eigenvectors = Vec::with_capacity(k);

    for _ in 0..k {
        let refs: Vec<&[f64]> = deflate_vecs.iter().map(|v| v.as_slice()).collect();
        let ev = power_iteration_deflated(&matvec, &refs, n, 200);
        let normalized = normalize_to_unit(&ev);
        let ev_norm = vec_norm(&ev);
        let ev_unit: Vec<f64> = ev.iter().map(|&x| x / ev_norm.max(1e-12)).collect();
        deflate_vecs.push(ev_unit);
        eigenvectors.push(normalized);
    }

    eigenvectors
}

/// Power iteration with deflation against given orthogonal vectors.
fn power_iteration_deflated(
    matvec: &dyn Fn(&[f64]) -> Vec<f64>,
    deflate_against: &[&[f64]],
    n: usize,
    max_iters: usize,
) -> Vec<f64> {
    // Deterministic pseudo-random initial vector (golden ratio based)
    let mut v: Vec<f64> = (0..n)
        .map(|i| {
            let x = (i as f64 + 1.0) * 0.618033988749895;
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

        // Convergence check: |v - mv| < epsilon
        let diff: f64 = v
            .iter()
            .zip(mv.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        v = mv;

        if diff < 1e-10 {
            break;
        }
    }

    v
}

/// Dot product of two vectors.
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Euclidean norm of a vector.
#[inline]
fn vec_norm(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

/// Normalize a vector to [0, 1] range.
fn normalize_to_unit(v: &[f64]) -> Vec<f64> {
    if v.is_empty() {
        return Vec::new();
    }
    let min = v.iter().copied().fold(f64::INFINITY, f64::min);
    let max = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;
    if range < 1e-12 {
        return vec![0.5; v.len()];
    }
    v.iter().map(|&x| (x - min) / range).collect()
}

// ============================================================================
// GraphAlgorithm trait implementation
// ============================================================================

/// Spectral embedding algorithm wrapper for registry integration.
pub struct SpectralEmbeddingAlgorithm;

static SPECTRAL_PARAMS: OnceLock<Vec<ParameterDef>> = OnceLock::new();

fn spectral_params() -> &'static [ParameterDef] {
    SPECTRAL_PARAMS.get_or_init(|| {
        vec![
            ParameterDef {
                name: "dimensions".to_string(),
                description: "Number of embedding dimensions".to_string(),
                param_type: ParameterType::Integer,
                required: false,
                default: Some("2".to_string()),
            },
            ParameterDef {
                name: "weight_property".to_string(),
                description: "Edge property name to use as weight".to_string(),
                param_type: ParameterType::String,
                required: false,
                default: None,
            },
        ]
    })
}

impl GraphAlgorithm for SpectralEmbeddingAlgorithm {
    fn name(&self) -> &str {
        "obrain.spectral_embedding"
    }

    fn description(&self) -> &str {
        "Spectral embedding via normalized Laplacian eigenvectors"
    }

    fn parameters(&self) -> &[ParameterDef] {
        spectral_params()
    }

    fn execute(&self, store: &dyn GraphStore, params: &Parameters) -> Result<AlgorithmResult> {
        let dimensions = params.get_int("dimensions").unwrap_or(2) as usize;

        let weight_prop_owned = params.get_string("weight_property").map(|s| s.to_string());
        let weight_property = weight_prop_owned.as_deref();

        let result = spectral_embedding(store, dimensions, weight_property);

        // Build result: node_id, embedding (as list of floats)
        let mut algo_result =
            AlgorithmResult::new(vec!["node_id".to_string(), "embedding".to_string()]);

        for (node_id, embedding) in &result.embeddings {
            let embedding_val = Value::List(embedding.iter().map(|&x| Value::Float64(x)).collect());
            algo_result.add_row(vec![Value::Int64(node_id.0 as i64), embedding_val]);
        }

        Ok(algo_result)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_star_graph(n: usize) -> SubstrateStore {
        let store = SubstrateStore::open_tempfile().unwrap();
        let center = store.create_node(&["Center"]);
        for _ in 1..n {
            let leaf = store.create_node(&["Leaf"]);
            store.create_edge(center, leaf, "CONNECTS");
            store.create_edge(leaf, center, "CONNECTS");
        }
        store
    }

    fn make_ring_graph(n: usize) -> SubstrateStore {
        let store = SubstrateStore::open_tempfile().unwrap();
        let nodes: Vec<NodeId> = (0..n).map(|_| store.create_node(&["Node"])).collect();
        for i in 0..n {
            let j = (i + 1) % n;
            store.create_edge(nodes[i], nodes[j], "CONNECTS");
            store.create_edge(nodes[j], nodes[i], "CONNECTS");
        }
        store
    }

    fn make_grid_graph(rows: usize, cols: usize) -> SubstrateStore {
        let store = SubstrateStore::open_tempfile().unwrap();
        let mut nodes = Vec::new();
        for _ in 0..rows {
            let mut row = Vec::new();
            for _ in 0..cols {
                row.push(store.create_node(&["Node"]));
            }
            nodes.push(row);
        }
        for r in 0..rows {
            for c in 0..cols {
                if c + 1 < cols {
                    store.create_edge(nodes[r][c], nodes[r][c + 1], "CONNECTS");
                    store.create_edge(nodes[r][c + 1], nodes[r][c], "CONNECTS");
                }
                if r + 1 < rows {
                    store.create_edge(nodes[r][c], nodes[r + 1][c], "CONNECTS");
                    store.create_edge(nodes[r + 1][c], nodes[r][c], "CONNECTS");
                }
            }
        }
        store
    }

    #[test]
    fn test_spectral_star_dim2() {
        let store = make_star_graph(6); // center + 5 leaves
        let result = spectral_embedding(&store, 2, None);
        assert_eq!(result.dimensions, 2);
        assert_eq!(result.embeddings.len(), 6);
        for emb in result.embeddings.values() {
            assert_eq!(emb.len(), 2);
            for &v in emb {
                assert!((0.0..=1.0).contains(&v), "value {v} not in [0,1]");
            }
        }
    }

    #[test]
    fn test_spectral_ring_dim4() {
        let store = make_ring_graph(10);
        let result = spectral_embedding(&store, 4, None);
        assert_eq!(result.dimensions, 4);
        assert_eq!(result.embeddings.len(), 10);
        for emb in result.embeddings.values() {
            assert_eq!(emb.len(), 4);
        }
    }

    #[test]
    fn test_spectral_grid_dim2() {
        let store = make_grid_graph(4, 4);
        let result = spectral_embedding(&store, 2, None);
        assert_eq!(result.dimensions, 2);
        assert_eq!(result.embeddings.len(), 16);
    }

    #[test]
    fn test_spectral_single_node() {
        let store = SubstrateStore::open_tempfile().unwrap();
        store.create_node(&["Alone"]);
        let result = spectral_embedding(&store, 2, None);
        assert_eq!(result.embeddings.len(), 1);
        for emb in result.embeddings.values() {
            assert_eq!(emb, &vec![0.5, 0.5]);
        }
    }

    #[test]
    fn test_spectral_two_nodes() {
        let store = SubstrateStore::open_tempfile().unwrap();
        let a = store.create_node(&["A"]);
        let b = store.create_node(&["B"]);
        store.create_edge(a, b, "LINK");
        store.create_edge(b, a, "LINK");
        let result = spectral_embedding(&store, 2, None);
        assert_eq!(result.embeddings.len(), 2);
    }

    #[test]
    fn test_spectral_determinism() {
        let store = make_star_graph(8);
        let r1 = spectral_embedding(&store, 4, None);
        let r2 = spectral_embedding(&store, 4, None);
        for (nid, emb1) in &r1.embeddings {
            let emb2 = r2.embeddings.get(nid).unwrap();
            for (a, b) in emb1.iter().zip(emb2.iter()) {
                assert!((a - b).abs() < 1e-10, "Non-deterministic: {a} vs {b}");
            }
        }
    }

    #[test]
    fn test_spectral_higher_dims() {
        let store = make_ring_graph(20);
        let result = spectral_embedding(&store, 8, None);
        assert_eq!(result.dimensions, 8);
        for emb in result.embeddings.values() {
            assert_eq!(emb.len(), 8);
            for &v in emb {
                assert!((0.0..=1.0).contains(&v));
            }
        }
    }

    #[test]
    fn test_spectral_algorithm_trait() {
        let store = make_star_graph(5);
        let algo = SpectralEmbeddingAlgorithm;
        assert_eq!(algo.name(), "obrain.spectral_embedding");

        let mut params = Parameters::new();
        params.set_int("dimensions", 4);
        let result = algo.execute(&store, &params).unwrap();
        assert_eq!(result.columns, vec!["node_id", "embedding"]);
        assert_eq!(result.rows.len(), 5);
    }
}
