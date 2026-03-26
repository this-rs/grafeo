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
// softmax_compete — competitive activation with lateral inhibition
// ---------------------------------------------------------------------------

/// Result of softmax competition: candidates with their normalized activation.
#[derive(Debug, Clone)]
pub struct CompetitionResult {
    /// The engram identifier.
    pub engram_id: EngramId,
    /// Normalized activation after softmax (∈ [0, 1], sums to 1).
    pub activation: f64,
    /// The retrieved engram.
    pub engram: Engram,
    /// Whether this is the dominant engram (highest activation).
    pub is_dominant: bool,
}

/// Apply softmax competition with lateral inhibition to Hopfield retrieval results.
///
/// Normalizes the attention weights via softmax and eliminates candidates below
/// `min_threshold`. The dominant engram (highest activation) stays strong while
/// secondaries remain active in the background.
///
/// # Parameters
/// - `candidates`: Results from `hopfield_retrieve`.
/// - `min_threshold`: Minimum activation to keep (default: 0.15). Candidates
///   below this threshold are eliminated (lateral inhibition).
///
/// # Returns
/// Filtered candidates sorted by activation descending, with `is_dominant` set
/// on the top-scoring engram.
pub fn softmax_compete(candidates: &[HopfieldResult], min_threshold: f64) -> Vec<CompetitionResult> {
    if candidates.is_empty() {
        return Vec::new();
    }

    // Re-normalize attention weights over the candidate subset.
    // The attention weights from hopfield_retrieve were softmax'd over ALL patterns
    // in the matrix, so they may be very small when N_patterns is large. Here we
    // re-normalize over only the top-K candidates so that the competitive dynamics
    // operate on a meaningful scale.
    //
    // We use the attention weights as log-space logits (taking log then softmax)
    // to preserve the relative ordering while re-scaling.
    let log_weights: Vec<f64> = candidates
        .iter()
        .map(|c| {
            if c.attention_weight > 0.0 {
                c.attention_weight.ln()
            } else {
                f64::NEG_INFINITY
            }
        })
        .collect();

    let max_lw = log_weights
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    if max_lw == f64::NEG_INFINITY {
        return Vec::new();
    }

    let exp_weights: Vec<f64> = log_weights.iter().map(|&lw| (lw - max_lw).exp()).collect();

    let sum: f64 = exp_weights.iter().sum();
    if sum == 0.0 {
        return Vec::new();
    }

    let normalized: Vec<f64> = exp_weights.iter().map(|w| w / sum).collect();

    // Find the dominant index (highest activation).
    let dominant_idx = normalized
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    // Filter by threshold (lateral inhibition) and build results.
    let mut results: Vec<CompetitionResult> = normalized
        .iter()
        .enumerate()
        .filter(|&(_, act)| *act >= min_threshold)
        .map(|(i, &act)| CompetitionResult {
            engram_id: candidates[i].engram_id,
            activation: act,
            engram: candidates[i].engram.clone(),
            is_dominant: i == dominant_idx,
        })
        .collect();

    // Sort by activation descending.
    results.sort_by(|a, b| {
        b.activation
            .partial_cmp(&a.activation)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    results
}

// ---------------------------------------------------------------------------
// max_marginal_relevance — diversity-aware selection
// ---------------------------------------------------------------------------

/// Result of MMR selection.
#[derive(Debug, Clone)]
pub struct MmrResult {
    /// The engram identifier.
    pub engram_id: EngramId,
    /// The relevance score (from competition activation).
    pub relevance: f64,
    /// The MMR score at selection time.
    pub mmr_score: f64,
    /// The retrieved engram.
    pub engram: Engram,
    /// Whether this is the dominant engram.
    pub is_dominant: bool,
}

/// Select engrams using Max Marginal Relevance for diversity.
///
/// Iteratively selects the candidate that maximizes:
///   `score = λ × relevance − (1 − λ) × max_similarity_to_selected`
///
/// This prevents N engrams from the same cluster from monopolizing the budget.
///
/// # Parameters
/// - `candidates`: Results from `softmax_compete`.
/// - `lambda`: Balance between relevance and diversity. λ=0.7 → 70% relevance,
///   30% diversity. Default: 0.7.
///
/// # Returns
/// Candidates re-ordered by MMR selection order (most relevant & diverse first).
pub fn max_marginal_relevance(candidates: &[CompetitionResult], lambda: f64) -> Vec<MmrResult> {
    if candidates.is_empty() {
        return Vec::new();
    }

    // Pre-compute signatures for similarity comparison.
    // Use the engram's spectral_signature directly.
    let signatures: Vec<&[f64]> = candidates
        .iter()
        .map(|c| c.engram.spectral_signature.as_slice())
        .collect();

    let mut selected: Vec<usize> = Vec::with_capacity(candidates.len());
    let mut remaining: Vec<usize> = (0..candidates.len()).collect();
    let mut results: Vec<MmrResult> = Vec::with_capacity(candidates.len());

    while !remaining.is_empty() {
        let mut best_idx_in_remaining = 0;
        let mut best_mmr = f64::NEG_INFINITY;

        for (ri, &cand_idx) in remaining.iter().enumerate() {
            let relevance = candidates[cand_idx].activation;

            // Max cosine similarity to any already-selected candidate.
            let max_sim = if selected.is_empty() {
                0.0
            } else {
                selected
                    .iter()
                    .map(|&sel_idx| cosine_sim(signatures[cand_idx], signatures[sel_idx]))
                    .fold(0.0f64, f64::max)
            };

            let mmr = lambda * relevance - (1.0 - lambda) * max_sim;

            if mmr > best_mmr {
                best_mmr = mmr;
                best_idx_in_remaining = ri;
            }
        }

        let chosen = remaining.remove(best_idx_in_remaining);
        selected.push(chosen);

        results.push(MmrResult {
            engram_id: candidates[chosen].engram_id,
            relevance: candidates[chosen].activation,
            mmr_score: best_mmr,
            engram: candidates[chosen].engram.clone(),
            is_dominant: candidates[chosen].is_dominant,
        });
    }

    results
}

/// Cosine similarity between two slices. Returns 0.0 for empty/zero-norm vectors.
fn cosine_sim(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
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

    // -- softmax_compete tests --

    #[test]
    fn softmax_compete_empty_input() {
        let results = softmax_compete(&[], 0.15);
        assert!(results.is_empty());
    }

    #[test]
    fn softmax_compete_filters_below_threshold() {
        let (store, matrix) = make_test_store_and_matrix();
        // Query strongly aligned with engram 1 → it dominates
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let candidates = hopfield_retrieve(&matrix, &query, &store, 3);

        let competed = softmax_compete(&candidates, 0.15);
        // The dominant engram should survive, weak ones should be filtered out
        assert!(!competed.is_empty());
        // All surviving activations >= 0.15
        for c in &competed {
            assert!(
                c.activation >= 0.15,
                "Activation {} below threshold 0.15",
                c.activation
            );
        }
        // Exactly one dominant
        let dominant_count = competed.iter().filter(|c| c.is_dominant).count();
        assert_eq!(dominant_count, 1);
    }

    #[test]
    fn softmax_compete_activations_sum_to_one_before_filter() {
        let (store, matrix) = make_test_store_and_matrix();
        let query = vec![0.5, 0.3, 0.2, 0.0];
        let candidates = hopfield_retrieve(&matrix, &query, &store, 3);

        // With threshold 0.0 (no filtering), activations should sum to ~1.0
        let competed = softmax_compete(&candidates, 0.0);
        let total: f64 = competed.iter().map(|c| c.activation).sum();
        assert!(
            (total - 1.0).abs() < 1e-10,
            "Expected sum ~1.0, got {}",
            total
        );
    }

    #[test]
    fn softmax_compete_dominant_is_first() {
        let (store, matrix) = make_test_store_and_matrix();
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let candidates = hopfield_retrieve(&matrix, &query, &store, 3);

        let competed = softmax_compete(&candidates, 0.0);
        assert!(!competed.is_empty());
        assert!(competed[0].is_dominant);
        assert_eq!(competed[0].engram_id, EngramId(1));
    }

    // -- MMR tests --

    #[test]
    fn mmr_empty_input() {
        let results = max_marginal_relevance(&[], 0.7);
        assert!(results.is_empty());
    }

    #[test]
    fn mmr_similar_engrams_diversified() {
        // Create 3 very similar engrams + 1 different one
        let similar_sig = vec![1.0, 0.0, 0.0, 0.0];
        let different_sig = vec![0.0, 0.0, 1.0, 0.0];

        let candidates = vec![
            CompetitionResult {
                engram_id: EngramId(1),
                activation: 0.4,
                engram: {
                    let mut e = Engram::new(EngramId(1), vec![(NodeId(1), 1.0)]);
                    e.spectral_signature = similar_sig.clone();
                    e
                },
                is_dominant: true,
            },
            CompetitionResult {
                engram_id: EngramId(2),
                activation: 0.35,
                engram: {
                    let mut e = Engram::new(EngramId(2), vec![(NodeId(2), 1.0)]);
                    // Very similar to engram 1
                    e.spectral_signature = vec![0.99, 0.1, 0.0, 0.0];
                    e
                },
                is_dominant: false,
            },
            CompetitionResult {
                engram_id: EngramId(3),
                activation: 0.30,
                engram: {
                    let mut e = Engram::new(EngramId(3), vec![(NodeId(3), 1.0)]);
                    // Very similar to engram 1
                    e.spectral_signature = vec![0.98, 0.05, 0.0, 0.0];
                    e
                },
                is_dominant: false,
            },
            CompetitionResult {
                engram_id: EngramId(4),
                activation: 0.25,
                engram: {
                    let mut e = Engram::new(EngramId(4), vec![(NodeId(4), 1.0)]);
                    // Very different from engrams 1-3
                    e.spectral_signature = different_sig;
                    e
                },
                is_dominant: false,
            },
        ];

        let mmr_results = max_marginal_relevance(&candidates, 0.7);

        // First should be the dominant (most relevant)
        assert_eq!(mmr_results[0].engram_id, EngramId(1));

        // Second should be engram 4 (different sig) rather than engram 2 (similar)
        // because MMR penalizes similarity to already-selected
        assert_eq!(
            mmr_results[1].engram_id,
            EngramId(4),
            "MMR should prefer the diverse engram over similar ones"
        );
    }

    #[test]
    fn mmr_preserves_all_candidates() {
        let candidates = vec![
            CompetitionResult {
                engram_id: EngramId(1),
                activation: 0.5,
                engram: {
                    let mut e = Engram::new(EngramId(1), vec![(NodeId(1), 1.0)]);
                    e.spectral_signature = vec![1.0, 0.0];
                    e
                },
                is_dominant: true,
            },
            CompetitionResult {
                engram_id: EngramId(2),
                activation: 0.3,
                engram: {
                    let mut e = Engram::new(EngramId(2), vec![(NodeId(2), 1.0)]);
                    e.spectral_signature = vec![0.0, 1.0];
                    e
                },
                is_dominant: false,
            },
        ];

        let mmr_results = max_marginal_relevance(&candidates, 0.7);
        assert_eq!(mmr_results.len(), 2);
    }
}
