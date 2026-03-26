//! Modern Hopfield Network — content-addressable memory with exponential capacity.
//!
//! Implements the retrieve operation:
//!   `retrieve(query) = softmax(β × P^T × query) × P`
//!
//! Where:
//! - P is the pattern matrix (N_engrams × dim_signature), rows = spectral signatures
//! - β is the per-engram precision (from `Engram::precision`)
//! - The softmax produces attention weights over stored patterns
//!
//! # Capacity
//! Modern Hopfield networks have exponential storage capacity (2^(d/2)) vs 0.14N
//! for classical Hopfield, making them suitable for large engram stores.
//!
//! # Concurrency
//! The retrieve operation is lock-free read — it only reads the pattern matrix.
//! Updates to the matrix happen only during engram formation/reconsolidation.

use super::store::EngramStore;
use super::types::{Engram, EngramId};

// ---------------------------------------------------------------------------
// HopfieldResult
// ---------------------------------------------------------------------------

/// A single result from Hopfield retrieval with attention weight.
#[derive(Debug, Clone)]
pub struct HopfieldResult {
    /// The engram identifier.
    pub engram_id: EngramId,
    /// The attention weight from softmax (∈ [0, 1], sums to 1 across results).
    pub attention_weight: f64,
    /// The retrieved engram.
    pub engram: Engram,
}

// ---------------------------------------------------------------------------
// PatternMatrix
// ---------------------------------------------------------------------------

/// The pattern matrix storing spectral signatures of all engrams.
///
/// Stored as a flat Vec<f64> in row-major order (N_engrams × dim_signature).
/// Each row is the spectral_signature of one engram.
#[derive(Debug, Clone)]
pub struct PatternMatrix {
    /// Row-major pattern data: patterns[i * dim .. (i+1) * dim] = signature of engram i.
    data: Vec<f64>,
    /// Engram IDs corresponding to each row.
    ids: Vec<EngramId>,
    /// Per-engram precision β values.
    betas: Vec<f64>,
    /// Dimension of each signature vector.
    dim: usize,
}

impl PatternMatrix {
    /// Create a new empty pattern matrix with the given signature dimension.
    pub fn new(dim: usize) -> Self {
        Self {
            data: Vec::new(),
            ids: Vec::new(),
            betas: Vec::new(),
            dim,
        }
    }

    /// Build a pattern matrix from the current state of an EngramStore.
    ///
    /// Only includes engrams with non-empty spectral signatures of the correct dimension.
    pub fn from_store(store: &EngramStore, dim: usize) -> Self {
        let engrams = store.list();
        let mut matrix = Self::new(dim);
        for engram in &engrams {
            if engram.spectral_signature.len() == dim {
                matrix.add_pattern(engram.id, &engram.spectral_signature, engram.precision);
            }
        }
        matrix
    }

    /// Add a single pattern (engram) to the matrix.
    pub fn add_pattern(&mut self, id: EngramId, signature: &[f64], beta: f64) {
        if signature.len() != self.dim {
            return;
        }
        // Check if this engram already exists — if so, update in place
        if let Some(pos) = self.ids.iter().position(|eid| *eid == id) {
            let start = pos * self.dim;
            self.data[start..start + self.dim].copy_from_slice(signature);
            self.betas[pos] = beta;
        } else {
            self.data.extend_from_slice(signature);
            self.ids.push(id);
            self.betas.push(beta);
        }
    }

    /// Remove a pattern by engram ID.
    pub fn remove_pattern(&mut self, id: EngramId) {
        if let Some(pos) = self.ids.iter().position(|eid| *eid == id) {
            let start = pos * self.dim;
            self.data.drain(start..start + self.dim);
            self.ids.remove(pos);
            self.betas.remove(pos);
        }
    }

    /// Number of stored patterns.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Whether the matrix is empty.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Dimension of each pattern vector.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get the pattern row for a given index (read-only slice).
    fn row(&self, i: usize) -> &[f64] {
        let start = i * self.dim;
        &self.data[start..start + self.dim]
    }
}

// ---------------------------------------------------------------------------
// hopfield_retrieve — the core Modern Hopfield operation
// ---------------------------------------------------------------------------

/// Retrieve engrams using Modern Hopfield attention.
///
/// `scores = softmax(β × P^T × q)` — attention weights over all stored patterns.
/// Returns the top-k engrams sorted by attention weight (descending).
///
/// # Parameters
/// - `matrix`: The pattern matrix (built from engram spectral signatures).
/// - `query`: The query vector (same dimension as the stored patterns).
/// - `store`: The engram store for retrieving full engram data.
/// - `k`: Maximum number of results to return.
///
/// # Per-engram β
/// Each engram has its own precision β. Higher β → sharper attention (one pattern
/// dominates). Lower β → softer attention (distribution spreads out).
/// This is different from a global β: experienced engrams (high precision) get
/// sharper retrieval while uncertain ones stay fuzzy.
pub fn hopfield_retrieve(
    matrix: &PatternMatrix,
    query: &[f64],
    store: &EngramStore,
    k: usize,
) -> Vec<HopfieldResult> {
    if matrix.is_empty() || query.len() != matrix.dim || k == 0 {
        return Vec::new();
    }

    let n = matrix.len();

    // Step 1: Compute β_i × (pattern_i · query) for each pattern i
    let mut logits = Vec::with_capacity(n);
    for i in 0..n {
        let row = matrix.row(i);
        let dot: f64 = row.iter().zip(query.iter()).map(|(p, q)| p * q).sum();
        logits.push(matrix.betas[i] * dot);
    }

    // Step 2: Softmax with numerical stability (subtract max)
    let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut weights: Vec<f64> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
    let sum: f64 = weights.iter().sum();
    if sum > 0.0 {
        for w in &mut weights {
            *w /= sum;
        }
    }

    // Step 3: Collect results, sort by attention weight, take top-k
    let mut results: Vec<(usize, f64)> = weights.iter().copied().enumerate().collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(k);

    results
        .into_iter()
        .filter_map(|(i, weight)| {
            let engram_id = matrix.ids[i];
            let engram = store.get(engram_id)?;
            Some(HopfieldResult {
                engram_id,
                attention_weight: weight,
                engram,
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engram::types::EngramId;
    use grafeo_common::types::NodeId;

    fn make_test_store_and_matrix() -> (EngramStore, PatternMatrix) {
        let store = EngramStore::new(None);
        let dim = 4;
        let mut matrix = PatternMatrix::new(dim);

        // Engram 1: pattern [1, 0, 0, 0] with high precision
        let id1 = store.next_id();
        let mut e1 = Engram::new(id1, vec![(NodeId(1), 1.0)]);
        e1.spectral_signature = vec![1.0, 0.0, 0.0, 0.0];
        e1.precision = 10.0;
        store.insert(e1);
        matrix.add_pattern(id1, &[1.0, 0.0, 0.0, 0.0], 10.0);

        // Engram 2: pattern [0, 1, 0, 0] with low precision
        let id2 = store.next_id();
        let mut e2 = Engram::new(id2, vec![(NodeId(2), 1.0)]);
        e2.spectral_signature = vec![0.0, 1.0, 0.0, 0.0];
        e2.precision = 0.5;
        store.insert(e2);
        matrix.add_pattern(id2, &[0.0, 1.0, 0.0, 0.0], 0.5);

        // Engram 3: pattern [0, 0, 1, 0] with medium precision
        let id3 = store.next_id();
        let mut e3 = Engram::new(id3, vec![(NodeId(3), 1.0)]);
        e3.spectral_signature = vec![0.0, 0.0, 1.0, 0.0];
        e3.precision = 2.0;
        store.insert(e3);
        matrix.add_pattern(id3, &[0.0, 0.0, 1.0, 0.0], 2.0);

        (store, matrix)
    }

    #[test]
    fn pattern_matrix_add_and_remove() {
        let mut matrix = PatternMatrix::new(3);
        assert!(matrix.is_empty());

        matrix.add_pattern(EngramId(1), &[1.0, 0.0, 0.0], 1.0);
        assert_eq!(matrix.len(), 1);

        matrix.add_pattern(EngramId(2), &[0.0, 1.0, 0.0], 2.0);
        assert_eq!(matrix.len(), 2);

        matrix.remove_pattern(EngramId(1));
        assert_eq!(matrix.len(), 1);
        assert_eq!(matrix.ids[0], EngramId(2));
    }

    #[test]
    fn pattern_matrix_update_existing() {
        let mut matrix = PatternMatrix::new(3);
        matrix.add_pattern(EngramId(1), &[1.0, 0.0, 0.0], 1.0);
        matrix.add_pattern(EngramId(1), &[0.0, 1.0, 0.0], 5.0);

        // Should not duplicate — still 1 pattern
        assert_eq!(matrix.len(), 1);
        assert_eq!(matrix.betas[0], 5.0);
        assert_eq!(matrix.row(0), &[0.0, 1.0, 0.0]);
    }

    #[test]
    fn pattern_matrix_ignores_wrong_dimension() {
        let mut matrix = PatternMatrix::new(3);
        matrix.add_pattern(EngramId(1), &[1.0, 0.0], 1.0); // dim 2 != 3
        assert!(matrix.is_empty());
    }

    #[test]
    fn hopfield_retrieve_high_beta_one_dominates() {
        let (store, matrix) = make_test_store_and_matrix();

        // Query aligned with engram 1 (high β=10)
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = hopfield_retrieve(&matrix, &query, &store, 3);

        assert!(!results.is_empty());
        // Engram 1 should dominate with high attention weight
        assert_eq!(results[0].engram_id, EngramId(1));
        assert!(
            results[0].attention_weight > 0.9,
            "Expected high β engram to dominate, got {}",
            results[0].attention_weight
        );
    }

    #[test]
    fn hopfield_retrieve_low_beta_fuzzy_distribution() {
        let store = EngramStore::new(None);
        let dim = 4;
        let mut matrix = PatternMatrix::new(dim);

        // Three engrams all with very low β
        for i in 1..=3u64 {
            let id = store.next_id();
            let mut sig = vec![0.0; dim];
            sig[(i - 1) as usize] = 1.0;
            let mut e = Engram::new(id, vec![(NodeId(i), 1.0)]);
            e.spectral_signature = sig.clone();
            e.precision = 0.01; // Very low β
            store.insert(e);
            matrix.add_pattern(EngramId(i), &sig, 0.01);
        }

        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = hopfield_retrieve(&matrix, &query, &store, 3);

        assert_eq!(results.len(), 3);
        // With very low β, distribution should be nearly uniform
        // No single engram should dominate
        let max_w = results[0].attention_weight;
        let min_w = results[results.len() - 1].attention_weight;
        assert!(
            max_w - min_w < 0.2,
            "Expected fuzzy distribution with low β, but max-min gap = {}",
            max_w - min_w
        );
    }

    #[test]
    fn hopfield_retrieve_empty_matrix() {
        let store = EngramStore::new(None);
        let matrix = PatternMatrix::new(4);
        let results = hopfield_retrieve(&matrix, &[1.0, 0.0, 0.0, 0.0], &store, 5);
        assert!(results.is_empty());
    }

    #[test]
    fn hopfield_retrieve_wrong_query_dim() {
        let (store, matrix) = make_test_store_and_matrix();
        let results = hopfield_retrieve(&matrix, &[1.0, 0.0], &store, 5); // dim 2 != 4
        assert!(results.is_empty());
    }

    #[test]
    fn hopfield_retrieve_respects_k() {
        let (store, matrix) = make_test_store_and_matrix();
        let results = hopfield_retrieve(&matrix, &[1.0, 0.0, 0.0, 0.0], &store, 1);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn hopfield_retrieve_weights_sum_to_one() {
        let (store, matrix) = make_test_store_and_matrix();
        let results = hopfield_retrieve(&matrix, &[0.5, 0.5, 0.0, 0.0], &store, 10);

        // All 3 engrams should be returned
        let total: f64 = results.iter().map(|r| r.attention_weight).sum();
        assert!(
            (total - 1.0).abs() < 1e-10,
            "Attention weights should sum to 1.0, got {}",
            total
        );
    }

    #[test]
    fn pattern_matrix_from_store() {
        let store = EngramStore::new(None);
        let dim = 4;

        let id1 = store.next_id();
        let mut e1 = Engram::new(id1, vec![(NodeId(1), 1.0)]);
        e1.spectral_signature = vec![1.0, 0.0, 0.0, 0.0];
        e1.precision = 3.0;
        store.insert(e1);

        let id2 = store.next_id();
        let mut e2 = Engram::new(id2, vec![(NodeId(2), 1.0)]);
        e2.spectral_signature = vec![0.0, 1.0, 0.0, 0.0];
        e2.precision = 1.0;
        store.insert(e2);

        // Engram with wrong dim should be skipped
        let id3 = store.next_id();
        let mut e3 = Engram::new(id3, vec![(NodeId(3), 1.0)]);
        e3.spectral_signature = vec![1.0, 0.0]; // dim=2, not 4
        store.insert(e3);

        let matrix = PatternMatrix::from_store(&store, dim);
        assert_eq!(matrix.len(), 2);
    }
}
