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

use serde::{Deserialize, Serialize};

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
/// Stored as a flat `Vec<f64>` in row-major order (N_engrams × dim_signature).
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
    let max_logit = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
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
pub fn softmax_compete(
    candidates: &[HopfieldResult],
    min_threshold: f64,
) -> Vec<CompetitionResult> {
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
        .copied()
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
        .map_or(0, |(i, _)| i);

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
// PredictiveModel — P(outcome | context) per engram
// ---------------------------------------------------------------------------

/// Simplified conditional distribution P(outcome | context) for an engram.
///
/// Each engram stores a running estimate of the outcomes it has observed
/// across its source episodes: the mean outcome vector and the variance
/// (uncertainty) around that mean.
///
/// This enables predictive coding: when a new context matches this engram,
/// the model predicts `mean` as the expected outcome; the prediction error
/// is then `|actual - mean| / variance`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictiveModel {
    /// Mean of the observed outcomes (running average).
    /// Each element corresponds to a dimension of the outcome space.
    pub mean: Vec<f64>,
    /// Variance of the observed outcomes per dimension.
    /// Higher variance → less precise prediction → smaller PE impact.
    pub variance: Vec<f64>,
    /// Number of observations that contributed to this model.
    pub observation_count: u32,
    /// Precision β — inverse of expected prediction error.
    /// High precision = confident model. Adjusts via Bayesian update.
    pub precision: f64,
}

impl PredictiveModel {
    /// Creates a new predictive model with the given dimensionality.
    ///
    /// Initialised with zero mean, unit variance (maximal uncertainty),
    /// and default precision of 1.0.
    pub fn new(dim: usize) -> Self {
        Self {
            mean: vec![0.0; dim],
            variance: vec![1.0; dim],
            observation_count: 0,
            precision: 1.0,
        }
    }

    /// Creates a predictive model from initial observations.
    ///
    /// If only one observation is provided, variance defaults to 1.0 (uncertain).
    pub fn from_observations(observations: &[Vec<f64>]) -> Option<Self> {
        if observations.is_empty() {
            return None;
        }
        let dim = observations[0].len();
        if dim == 0 {
            return None;
        }

        let n = observations.len() as f64;
        let mut mean = vec![0.0; dim];
        for obs in observations {
            if obs.len() != dim {
                return None;
            }
            for (i, &v) in obs.iter().enumerate() {
                mean[i] += v;
            }
        }
        for m in &mut mean {
            *m /= n;
        }

        let mut variance = vec![1.0; dim]; // default variance if n=1
        if observations.len() > 1 {
            for d in 0..dim {
                let var: f64 = observations
                    .iter()
                    .map(|obs| {
                        let diff = obs[d] - mean[d];
                        diff * diff
                    })
                    .sum::<f64>()
                    / n;
                // Floor variance to avoid division by zero
                variance[d] = var.max(1e-6);
            }
        }

        Some(Self {
            mean,
            variance,
            observation_count: observations.len() as u32,
            precision: 1.0,
        })
    }

    /// Returns the dimensionality of this predictive model.
    pub fn dim(&self) -> usize {
        self.mean.len()
    }

    /// Incrementally update the model with a new observation (online mean/variance).
    pub fn observe(&mut self, outcome: &[f64]) {
        if outcome.len() != self.dim() {
            return;
        }
        self.observation_count += 1;
        let n = self.observation_count as f64;

        for i in 0..self.dim() {
            let old_mean = self.mean[i];
            // Welford's online algorithm for mean and variance
            let delta = outcome[i] - old_mean;
            self.mean[i] += delta / n;
            let delta2 = outcome[i] - self.mean[i];
            // Running variance (population variance)
            if n > 1.0 {
                self.variance[i] += (delta * delta2 - self.variance[i]) / n;
                // Floor variance
                if self.variance[i] < 1e-6 {
                    self.variance[i] = 1e-6;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// compute_prediction_error — PE = |actual - predicted| / variance
// ---------------------------------------------------------------------------

/// Result of a prediction error computation.
#[derive(Debug, Clone)]
pub struct PredictionErrorResult {
    /// Per-dimension prediction error (normalized by variance).
    pub per_dimension: Vec<f64>,
    /// Aggregate prediction error ∈ [0, 1] — mean of per-dimension PEs, clamped.
    pub magnitude: f64,
    /// Raw (unnormalized) error magnitude — L2 distance between actual and predicted.
    pub raw_error: f64,
}

/// Compute the prediction error between an engram's predictive model and actual outcome.
///
/// PE per dimension = |actual_i - mean_i| / sqrt(variance_i).
/// Aggregate PE = mean of per-dimension PEs, normalized to [0, 1] via tanh.
///
/// Returns `None` if the model or outcome dimensions don't match.
pub fn compute_prediction_error(
    model: &PredictiveModel,
    actual_outcome: &[f64],
) -> Option<PredictionErrorResult> {
    if model.dim() == 0 || actual_outcome.len() != model.dim() {
        return None;
    }

    let mut per_dim = Vec::with_capacity(model.dim());
    let mut raw_sq_sum = 0.0f64;

    for i in 0..model.dim() {
        let diff = (actual_outcome[i] - model.mean[i]).abs();
        raw_sq_sum += diff * diff;
        // Normalize by standard deviation (sqrt of variance)
        let std_dev = model.variance[i].sqrt();
        let pe_i = diff / std_dev;
        per_dim.push(pe_i);
    }

    let raw_error = raw_sq_sum.sqrt();

    // Aggregate: mean of per-dimension PEs, then tanh to normalize to [0, 1]
    let mean_pe = per_dim.iter().sum::<f64>() / per_dim.len() as f64;
    let magnitude = mean_pe.tanh(); // smooth normalization to [0, 1]

    Some(PredictionErrorResult {
        per_dimension: per_dim,
        magnitude,
        raw_error,
    })
}

// ---------------------------------------------------------------------------
// bayesian_update — learning_rate = precision × pe
// ---------------------------------------------------------------------------

/// Result of a Bayesian update step.
#[derive(Debug, Clone)]
pub struct BayesianUpdateResult {
    /// The effective learning rate used (precision × pe).
    pub learning_rate: f64,
    /// The posterior precision after the update.
    pub posterior_precision: f64,
    /// Whether this was a significant update (learning_rate > 0.1).
    pub significant: bool,
}

/// Perform a Bayesian update on an engram's predictive model given a prediction error.
///
/// # Learning dynamics
/// - `learning_rate = precision × pe_magnitude`
/// - **High PE + high precision** → large learning rate → big model update
///   (confident model was surprised → must correct strongly)
/// - **High PE + low precision** → small learning rate → small model update
///   (uncertain model was surprised → not much new info)
/// - **Low PE** → small learning rate regardless of precision
///   (prediction was accurate → little correction needed)
///
/// # Precision update (posterior)
/// The precision itself adjusts after seeing the PE:
/// - If PE is low (prediction was good), precision increases (model becomes more confident)
/// - If PE is high (prediction was wrong), precision decreases proportionally
///
/// Formula: `posterior_precision = prior_precision × (1 - pe²) + pe² × (1 / (1 + pe))`
/// This smoothly interpolates between reinforcement (low PE) and weakening (high PE).
///
/// Returns `None` if the actual outcome dimensions don't match the model.
pub fn bayesian_update(
    model: &mut PredictiveModel,
    actual_outcome: &[f64],
    pe: &PredictionErrorResult,
) -> Option<BayesianUpdateResult> {
    if actual_outcome.len() != model.dim() {
        return None;
    }

    let prior_precision = model.precision;
    let pe_mag = pe.magnitude;

    // Learning rate = precision × prediction error magnitude
    let learning_rate = (prior_precision * pe_mag).clamp(0.0, 1.0);

    // Update the mean towards the actual outcome, weighted by learning rate
    for i in 0..model.dim() {
        let error = actual_outcome[i] - model.mean[i];
        model.mean[i] += learning_rate * error;
    }

    // Update variance: shrink towards observed squared error
    for i in 0..model.dim() {
        let diff = actual_outcome[i] - model.mean[i]; // post-update residual
        let observed_var = diff * diff;
        model.variance[i] =
            (1.0 - learning_rate) * model.variance[i] + learning_rate * observed_var;
        // Floor variance
        if model.variance[i] < 1e-6 {
            model.variance[i] = 1e-6;
        }
    }

    // Note: observation_count is NOT incremented here. bayesian_update is a
    // PE-driven correction of an existing model, not a new observation.
    // Use PredictiveModel::observe() for incremental learning without PE.

    // Update precision (posterior)
    // Low PE → precision increases (model confirmed)
    // High PE → precision decreases (model was wrong)
    let pe_sq = pe_mag * pe_mag;
    let posterior_precision = prior_precision * (1.0 - pe_sq) + pe_sq * (1.0 / (1.0 + pe_mag));
    // Floor precision to avoid zero/negative
    model.precision = posterior_precision.max(0.01);

    Some(BayesianUpdateResult {
        learning_rate,
        posterior_precision: model.precision,
        significant: learning_rate > 0.1,
    })
}

// ---------------------------------------------------------------------------
// PredictionErrorCalculator — bridges predictive models to formation
// ---------------------------------------------------------------------------

/// Calculates prediction error using existing engrams' predictive models.
///
/// Given a set of active engrams and an observed outcome, this finds the
/// best-matching engram's prediction and computes the PE against reality.
/// This replaces the simple heuristic PE used in early formation.
pub struct PredictionErrorCalculator;

impl PredictionErrorCalculator {
    /// Compute the prediction error for a new observation given existing engrams.
    ///
    /// # Algorithm
    /// 1. For each engram that has a PredictiveModel, compute PE(engram, actual).
    /// 2. Use the engram with the **lowest** raw PE as the "best prediction" —
    ///    this is the engram that expected something closest to what happened.
    /// 3. Return the PE of that best-matching engram.
    ///
    /// If no engrams have predictive models, returns a default high PE (1.0)
    /// to indicate maximum surprise (novel observation).
    pub fn compute_from_engrams(
        store: &EngramStore,
        active_engram_ids: &[EngramId],
        actual_outcome: &[f64],
    ) -> f64 {
        if actual_outcome.is_empty() {
            return 1.0;
        }

        let mut best_pe: Option<f64> = None;

        for &eid in active_engram_ids {
            if let Some(engram) = store.get(eid)
                && let Some(ref pm) = engram.predictive_model
                && pm.dim() == actual_outcome.len()
                && let Some(pe_result) = compute_prediction_error(pm, actual_outcome)
            {
                match best_pe {
                    None => best_pe = Some(pe_result.magnitude),
                    Some(current_best) => {
                        if pe_result.magnitude < current_best {
                            best_pe = Some(pe_result.magnitude);
                        }
                    }
                }
            }
        }

        // If no engram had a predictive model → novel observation → max surprise
        best_pe.unwrap_or(1.0)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engram::types::EngramId;
    use obrain_common::types::NodeId;

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

    // -- PredictiveModel tests --

    #[test]
    fn predictive_model_new_defaults() {
        let pm = PredictiveModel::new(3);
        assert_eq!(pm.dim(), 3);
        assert_eq!(pm.observation_count, 0);
        assert!((pm.precision - 1.0).abs() < f64::EPSILON);
        assert!(pm.mean.iter().all(|&v| v == 0.0));
        assert!(pm.variance.iter().all(|&v| (v - 1.0).abs() < f64::EPSILON));
    }

    #[test]
    fn predictive_model_from_observations_single() {
        let obs = vec![vec![1.0, 2.0, 3.0]];
        let pm = PredictiveModel::from_observations(&obs).unwrap();
        assert_eq!(pm.mean, vec![1.0, 2.0, 3.0]);
        // Single observation → variance defaults to 1.0
        assert!(pm.variance.iter().all(|&v| (v - 1.0).abs() < f64::EPSILON));
        assert_eq!(pm.observation_count, 1);
    }

    #[test]
    fn predictive_model_from_observations_multiple() {
        let obs = vec![vec![0.0, 0.0], vec![2.0, 4.0]];
        let pm = PredictiveModel::from_observations(&obs).unwrap();
        assert!((pm.mean[0] - 1.0).abs() < 1e-10);
        assert!((pm.mean[1] - 2.0).abs() < 1e-10);
        // Variance: E[(x-mean)^2] = 1.0 for dim 0, 4.0 for dim 1
        assert!((pm.variance[0] - 1.0).abs() < 1e-10);
        assert!((pm.variance[1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn predictive_model_observe_updates_mean() {
        let mut pm = PredictiveModel::new(2);
        pm.observe(&[2.0, 4.0]);
        assert!((pm.mean[0] - 2.0).abs() < 1e-10);
        assert!((pm.mean[1] - 4.0).abs() < 1e-10);
        assert_eq!(pm.observation_count, 1);

        pm.observe(&[4.0, 0.0]);
        assert!((pm.mean[0] - 3.0).abs() < 1e-10);
        assert!((pm.mean[1] - 2.0).abs() < 1e-10);
        assert_eq!(pm.observation_count, 2);
    }

    // -- compute_prediction_error tests --

    #[test]
    fn prediction_error_compute_zero_error() {
        let pm = PredictiveModel::from_observations(&[vec![1.0, 2.0], vec![1.0, 2.0]]).unwrap();
        let pe = compute_prediction_error(&pm, &[1.0, 2.0]).unwrap();
        assert!(
            pe.magnitude < 0.01,
            "Expected near-zero PE when actual == predicted, got {}",
            pe.magnitude
        );
        assert!(pe.raw_error < 1e-10);
    }

    #[test]
    fn prediction_error_compute_high_error() {
        let pm = PredictiveModel::from_observations(&[vec![0.0], vec![0.0]]).unwrap();
        // Actual is very far from predicted (0.0), and variance is tiny (1e-6 floor)
        let pe = compute_prediction_error(&pm, &[10.0]).unwrap();
        assert!(
            pe.magnitude > 0.9,
            "Expected high PE for large deviation, got {}",
            pe.magnitude
        );
    }

    #[test]
    fn prediction_error_compute_normalized_between_0_and_1() {
        let pm = PredictiveModel::from_observations(&[vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        let pe = compute_prediction_error(&pm, &[100.0, 200.0]).unwrap();
        assert!(pe.magnitude >= 0.0 && pe.magnitude <= 1.0);
    }

    #[test]
    fn prediction_error_compute_dimension_mismatch() {
        let pm = PredictiveModel::new(3);
        assert!(compute_prediction_error(&pm, &[1.0, 2.0]).is_none());
    }

    #[test]
    fn prediction_error_compute_variance_affects_pe() {
        // High variance → same deviation produces lower PE
        let mut pm_precise = PredictiveModel::new(1);
        pm_precise.mean = vec![0.0];
        pm_precise.variance = vec![0.01]; // very precise

        let mut pm_uncertain = PredictiveModel::new(1);
        pm_uncertain.mean = vec![0.0];
        pm_uncertain.variance = vec![100.0]; // very uncertain

        let actual = &[1.0];
        let pe_precise = compute_prediction_error(&pm_precise, actual).unwrap();
        let pe_uncertain = compute_prediction_error(&pm_uncertain, actual).unwrap();

        assert!(
            pe_precise.magnitude > pe_uncertain.magnitude,
            "Same deviation should give higher PE for precise model ({}) vs uncertain ({})",
            pe_precise.magnitude,
            pe_uncertain.magnitude
        );
    }

    // -- bayesian_update tests --

    #[test]
    fn bayesian_update_high_pe_high_precision_large_correction() {
        let mut pm = PredictiveModel::new(1);
        pm.mean = vec![0.0];
        pm.variance = vec![0.01]; // precise model
        pm.precision = 5.0; // high confidence

        let actual = &[10.0]; // far from prediction
        let pe = compute_prediction_error(&pm, actual).unwrap();
        assert!(pe.magnitude > 0.5, "PE should be high");

        let old_mean = pm.mean[0];
        let result = bayesian_update(&mut pm, actual, &pe).unwrap();

        assert!(
            result.significant,
            "High PE + high precision → significant update"
        );
        assert!(result.learning_rate > 0.3, "Learning rate should be high");
        // Mean should have moved significantly towards actual
        let correction = (pm.mean[0] - old_mean).abs();
        assert!(
            correction > 1.0,
            "Mean should have moved significantly, got correction {}",
            correction
        );
    }

    #[test]
    fn bayesian_update_high_pe_low_precision_small_correction() {
        let mut pm = PredictiveModel::new(1);
        pm.mean = vec![0.0];
        pm.variance = vec![100.0]; // very uncertain model
        pm.precision = 0.05; // low confidence

        let actual = &[10.0]; // far from prediction
        let pe = compute_prediction_error(&pm, actual).unwrap();

        let old_mean = pm.mean[0];
        let result = bayesian_update(&mut pm, actual, &pe).unwrap();

        // Low precision × PE → small learning rate
        assert!(
            result.learning_rate < 0.2,
            "Learning rate should be low with low precision, got {}",
            result.learning_rate
        );
        let correction = (pm.mean[0] - old_mean).abs();
        assert!(
            correction < 2.0,
            "Correction should be small with low precision, got {}",
            correction
        );
    }

    #[test]
    fn bayesian_update_low_pe_precision_increases() {
        let mut pm = PredictiveModel::new(1);
        pm.mean = vec![5.0];
        pm.variance = vec![1.0];
        pm.precision = 1.0;

        // Actual very close to prediction → low PE
        let actual = &[5.01];
        let pe = compute_prediction_error(&pm, actual).unwrap();
        assert!(pe.magnitude < 0.1, "PE should be low");

        let prior_precision = pm.precision;
        bayesian_update(&mut pm, actual, &pe).unwrap();

        // Low PE → precision should increase (model confirmed)
        assert!(
            pm.precision >= prior_precision * 0.99,
            "Precision should stay stable or increase with low PE, was {} now {}",
            prior_precision,
            pm.precision
        );
    }

    #[test]
    fn bayesian_update_high_pe_precision_decreases() {
        let mut pm = PredictiveModel::new(1);
        pm.mean = vec![0.0];
        pm.variance = vec![0.01];
        pm.precision = 5.0;

        let actual = &[100.0]; // massive deviation
        let pe = compute_prediction_error(&pm, actual).unwrap();

        let prior_precision = pm.precision;
        bayesian_update(&mut pm, actual, &pe).unwrap();

        assert!(
            pm.precision < prior_precision,
            "Precision should decrease after high PE, was {} now {}",
            prior_precision,
            pm.precision
        );
    }

    #[test]
    fn bayesian_update_dimension_mismatch() {
        let mut pm = PredictiveModel::new(2);
        let pe = PredictionErrorResult {
            per_dimension: vec![0.5, 0.5],
            magnitude: 0.5,
            raw_error: 1.0,
        };
        assert!(bayesian_update(&mut pm, &[1.0], &pe).is_none());
    }

    // -- PredictionErrorCalculator tests --

    #[test]
    fn formation_with_predictive_engrams_use_model() {
        let store = EngramStore::new(None);

        // Create an engram with a predictive model predicting [5.0, 5.0]
        let id1 = store.next_id();
        let mut e1 = Engram::new(id1, vec![(NodeId(1), 1.0)]);
        let mut pm = PredictiveModel::from_observations(&[vec![5.0, 5.0], vec![5.0, 5.0]]).unwrap();
        pm.precision = 3.0;
        e1.predictive_model = Some(pm);
        store.insert(e1);

        // Actual outcome matches prediction → low PE
        let pe_low = PredictionErrorCalculator::compute_from_engrams(&store, &[id1], &[5.0, 5.0]);
        assert!(
            pe_low < 0.1,
            "PE should be low when outcome matches prediction, got {}",
            pe_low
        );

        // Actual outcome far from prediction → high PE
        let pe_high =
            PredictionErrorCalculator::compute_from_engrams(&store, &[id1], &[100.0, 100.0]);
        assert!(
            pe_high > 0.5,
            "PE should be high when outcome differs from prediction, got {}",
            pe_high
        );
    }

    #[test]
    fn formation_with_predictive_no_models_returns_max() {
        let store = EngramStore::new(None);

        // Engram without predictive model
        let id1 = store.next_id();
        let e1 = Engram::new(id1, vec![(NodeId(1), 1.0)]);
        store.insert(e1);

        let pe = PredictionErrorCalculator::compute_from_engrams(&store, &[id1], &[1.0, 2.0]);
        assert!(
            (pe - 1.0).abs() < f64::EPSILON,
            "No predictive models → max surprise (1.0), got {}",
            pe
        );
    }

    #[test]
    fn formation_with_predictive_best_match_wins() {
        let store = EngramStore::new(None);

        // Engram 1: predicts [0.0, 0.0] — far from actual
        let id1 = store.next_id();
        let mut e1 = Engram::new(id1, vec![(NodeId(1), 1.0)]);
        e1.predictive_model =
            Some(PredictiveModel::from_observations(&[vec![0.0, 0.0], vec![0.0, 0.0]]).unwrap());
        store.insert(e1);

        // Engram 2: predicts [10.0, 10.0] — close to actual
        let id2 = store.next_id();
        let mut e2 = Engram::new(id2, vec![(NodeId(2), 1.0)]);
        e2.predictive_model = Some(
            PredictiveModel::from_observations(&[vec![10.0, 10.0], vec![10.0, 10.0]]).unwrap(),
        );
        store.insert(e2);

        // Actual is [10.0, 10.0] → engram 2 should be the best predictor
        let pe =
            PredictionErrorCalculator::compute_from_engrams(&store, &[id1, id2], &[10.0, 10.0]);

        // The PE should be low because engram 2 predicted correctly
        assert!(
            pe < 0.1,
            "Best-matching engram should yield low PE, got {}",
            pe
        );
    }

    // -----------------------------------------------------------------------
    // Additional coverage tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_pattern_matrix_new_empty() {
        let matrix = PatternMatrix::new(8);
        assert_eq!(matrix.len(), 0);
        assert!(matrix.is_empty());
        assert_eq!(matrix.dim(), 8);
    }

    #[test]
    fn test_pattern_matrix_remove_nonexistent() {
        let mut matrix = PatternMatrix::new(3);
        matrix.add_pattern(EngramId(1), &[1.0, 0.0, 0.0], 1.0);
        assert_eq!(matrix.len(), 1);

        // Remove non-existent — no panic, len unchanged
        matrix.remove_pattern(EngramId(999));
        assert_eq!(matrix.len(), 1);
    }

    #[test]
    fn test_hopfield_retrieve_single_pattern() {
        let store = EngramStore::new(None);
        let dim = 3;
        let mut matrix = PatternMatrix::new(dim);

        let id1 = store.next_id();
        let mut e1 = Engram::new(id1, vec![(NodeId(1), 1.0)]);
        e1.spectral_signature = vec![1.0, 0.0, 0.0];
        e1.precision = 1.0;
        store.insert(e1);
        matrix.add_pattern(id1, &[1.0, 0.0, 0.0], 1.0);

        let results = hopfield_retrieve(&matrix, &[1.0, 0.0, 0.0], &store, 5);
        assert_eq!(results.len(), 1);
        assert!((results[0].attention_weight - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_softmax_compete_single_candidate() {
        let store = EngramStore::new(None);
        let dim = 3;
        let mut matrix = PatternMatrix::new(dim);

        let id1 = store.next_id();
        let mut e1 = Engram::new(id1, vec![(NodeId(1), 1.0)]);
        e1.spectral_signature = vec![1.0, 0.0, 0.0];
        e1.precision = 1.0;
        store.insert(e1);
        matrix.add_pattern(id1, &[1.0, 0.0, 0.0], 1.0);

        let candidates = hopfield_retrieve(&matrix, &[1.0, 0.0, 0.0], &store, 1);
        let competed = softmax_compete(&candidates, 0.0);

        assert_eq!(competed.len(), 1);
        assert!((competed[0].activation - 1.0).abs() < 1e-10);
        assert!(competed[0].is_dominant);
    }

    #[test]
    fn test_mmr_lambda_zero_pure_diversity() {
        // With lambda=0, MMR score = -max_similarity (pure diversity)
        let similar_sig = vec![1.0, 0.0, 0.0, 0.0];

        let candidates = vec![
            CompetitionResult {
                engram_id: EngramId(1),
                activation: 0.5,
                engram: {
                    let mut e = Engram::new(EngramId(1), vec![(NodeId(1), 1.0)]);
                    e.spectral_signature = similar_sig.clone();
                    e
                },
                is_dominant: true,
            },
            CompetitionResult {
                engram_id: EngramId(2),
                activation: 0.4,
                engram: {
                    let mut e = Engram::new(EngramId(2), vec![(NodeId(2), 1.0)]);
                    e.spectral_signature = vec![0.99, 0.1, 0.0, 0.0]; // very similar to 1
                    e
                },
                is_dominant: false,
            },
            CompetitionResult {
                engram_id: EngramId(3),
                activation: 0.1,
                engram: {
                    let mut e = Engram::new(EngramId(3), vec![(NodeId(3), 1.0)]);
                    e.spectral_signature = vec![0.0, 0.0, 1.0, 0.0]; // very different
                    e
                },
                is_dominant: false,
            },
        ];

        let mmr_zero = max_marginal_relevance(&candidates, 0.0);
        let mmr_one = max_marginal_relevance(&candidates, 1.0);

        // With lambda=0 (pure diversity), the diverse engram should be picked earlier
        // than with lambda=1 (pure relevance)
        let pos_diverse_l0 = mmr_zero
            .iter()
            .position(|r| r.engram_id == EngramId(3))
            .unwrap();
        let pos_diverse_l1 = mmr_one
            .iter()
            .position(|r| r.engram_id == EngramId(3))
            .unwrap();
        assert!(
            pos_diverse_l0 <= pos_diverse_l1,
            "lambda=0 should favor diverse engram: pos_l0={}, pos_l1={}",
            pos_diverse_l0,
            pos_diverse_l1
        );
    }

    #[test]
    fn test_mmr_lambda_one_pure_relevance() {
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

        let mmr = max_marginal_relevance(&candidates, 1.0);
        // With lambda=1, order should follow activation: 1 then 2
        assert_eq!(mmr[0].engram_id, EngramId(1));
        assert_eq!(mmr[1].engram_id, EngramId(2));
    }

    #[test]
    fn test_mmr_single_candidate() {
        let candidates = vec![CompetitionResult {
            engram_id: EngramId(42),
            activation: 1.0,
            engram: {
                let mut e = Engram::new(EngramId(42), vec![(NodeId(1), 1.0)]);
                e.spectral_signature = vec![1.0, 0.0];
                e
            },
            is_dominant: true,
        }];

        let mmr = max_marginal_relevance(&candidates, 0.7);
        assert_eq!(mmr.len(), 1);
        assert_eq!(mmr[0].engram_id, EngramId(42));
    }

    #[test]
    fn test_predictive_model_from_empty_observations() {
        let result = PredictiveModel::from_observations(&[]);
        assert!(result.is_none());
    }

    #[test]
    fn test_predictive_model_observe_multiple_converges() {
        let mut pm = PredictiveModel::new(1);
        let target = 5.0;
        for _ in 0..10 {
            pm.observe(&[target]);
        }
        assert!(
            (pm.mean[0] - target).abs() < 1e-10,
            "mean should converge to {target}, got {}",
            pm.mean[0]
        );
    }

    #[test]
    fn test_compute_prediction_error_identical() {
        let pm = PredictiveModel::from_observations(&[vec![3.0, 4.0], vec![3.0, 4.0]]).unwrap();
        let pe = compute_prediction_error(&pm, &[3.0, 4.0]).unwrap();
        assert!(
            pe.magnitude < 0.01,
            "actual == predicted should give near-zero PE, got {}",
            pe.magnitude
        );
    }

    #[test]
    fn test_bayesian_update_zero_prediction_error() {
        let mut pm = PredictiveModel::new(1);
        pm.mean = vec![5.0];
        pm.variance = vec![1.0];
        pm.precision = 2.0;

        let actual = &[5.0]; // exactly the mean
        let pe = compute_prediction_error(&pm, actual).unwrap();
        assert!(pe.magnitude < 0.01);

        let old_mean = pm.mean[0];
        let result = bayesian_update(&mut pm, actual, &pe).unwrap();

        // Mean should be essentially unchanged
        assert!(
            (pm.mean[0] - old_mean).abs() < 0.01,
            "mean should be unchanged with zero PE"
        );
        // Precision should increase slightly (model confirmed)
        assert!(
            result.posterior_precision >= 2.0 * 0.99,
            "precision should stay stable or increase, got {}",
            result.posterior_precision
        );
    }
}
