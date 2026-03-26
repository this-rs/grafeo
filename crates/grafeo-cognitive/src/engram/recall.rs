//! Recall engine, warmup selection, and reconsolidation.
//!
//! This module implements the two-path recall mechanism:
//! 1. **Direct recall** — find engrams whose ensemble contains the cue nodes.
//! 2. **Spectral recall** — encode cues as a vector and find nearest neighbors.
//!
//! Results are merged, deduplicated, and ranked by a combined confidence score.
//!
//! The [`WarmupSelector`] adds MMR-based diversity to avoid returning a cluster
//! of near-identical engrams. [`reconsolidate`] handles memory updating when
//! prediction error triggers ensemble modification.

use std::collections::{HashMap, HashSet};

use grafeo_common::types::NodeId;
use serde::{Deserialize, Serialize};

use super::spectral::SpectralEncoder;
use super::store::EngramStore;
use super::traits::VectorIndex;
use super::types::{Engram, EngramId};

// ---------------------------------------------------------------------------
// RecallResult
// ---------------------------------------------------------------------------

/// A single recall result with its confidence score and the retrieved engram.
#[derive(Debug, Clone)]
pub struct RecallResult {
    /// The identifier of the recalled engram.
    pub engram_id: EngramId,
    /// Combined confidence score in [0.0, 1.0].
    pub confidence: f64,
    /// The retrieved engram data.
    pub engram: Engram,
}

// ---------------------------------------------------------------------------
// RecallEngine
// ---------------------------------------------------------------------------

/// The recall engine combines direct and spectral retrieval paths.
#[derive(Debug)]
pub struct RecallEngine {
    /// Weight given to the direct (overlap) path vs. spectral path.
    direct_weight: f64,
}

impl RecallEngine {
    /// Creates a new recall engine with default weights (0.6 direct, 0.4 spectral).
    pub fn new() -> Self {
        Self {
            direct_weight: 0.6,
        }
    }

    /// Recall engrams matching the given cue nodes.
    ///
    /// # Two-path recall
    /// 1. **Direct**: finds engrams whose ensemble contains any of the cue nodes
    ///    and scores them by overlap fraction.
    /// 2. **Spectral**: encodes cues as a vector signature and queries the
    ///    `vector_index` for nearest neighbors, converting distance to similarity.
    ///
    /// Results are merged by engram ID, deduplicated, and sorted by descending
    /// confidence.
    pub fn recall(
        &self,
        store: &EngramStore,
        cues: &[NodeId],
        vector_index: &dyn VectorIndex,
        spectral: &SpectralEncoder,
        k: usize,
    ) -> Vec<RecallResult> {
        if cues.is_empty() || k == 0 {
            return Vec::new();
        }

        let cue_set: HashSet<NodeId> = cues.iter().copied().collect();

        // -- Path 1: Direct recall via node overlap --
        let mut scores: HashMap<EngramId, (f64, f64)> = HashMap::new(); // (direct, spectral)

        for &cue in cues {
            for engram_id in store.find_by_node(cue) {
                if let Some(engram) = store.get(engram_id) {
                    let overlap = engram
                        .ensemble
                        .iter()
                        .filter(|(nid, _)| cue_set.contains(nid))
                        .count();
                    let total = engram.ensemble.len().max(1);
                    let overlap_score = overlap as f64 / total as f64;

                    let entry = scores.entry(engram_id).or_insert((0.0, 0.0));
                    // Take the best overlap score across cue lookups.
                    if overlap_score > entry.0 {
                        entry.0 = overlap_score;
                    }
                }
            }
        }

        // -- Path 2: Spectral recall via vector similarity --
        let cue_ensemble: Vec<(NodeId, f64)> = cues.iter().map(|&nid| (nid, 1.0)).collect();
        let query_vec = spectral.encode(&cue_ensemble);

        // Request more than k to account for overlap with direct results.
        let spectral_results = vector_index.nearest(&query_vec, k * 2);

        for (id_str, distance) in &spectral_results {
            if let Ok(raw_id) = id_str.parse::<u64>() {
                let engram_id = EngramId(raw_id);
                // Convert cosine distance to similarity.
                let similarity = (1.0 - distance).clamp(0.0, 1.0);
                let entry = scores.entry(engram_id).or_insert((0.0, 0.0));
                if similarity > entry.1 {
                    entry.1 = similarity;
                }
            }
        }

        // -- Merge and rank --
        let spectral_weight = 1.0 - self.direct_weight;
        let mut results: Vec<RecallResult> = scores
            .into_iter()
            .filter_map(|(engram_id, (direct_score, spectral_score))| {
                let engram = store.get(engram_id)?;
                let confidence =
                    (self.direct_weight * direct_score + spectral_weight * spectral_score)
                        .clamp(0.0, 1.0);
                Some(RecallResult {
                    engram_id,
                    confidence,
                    engram,
                })
            })
            .collect();

        results.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k);
        results
    }
}

impl Default for RecallEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// WarmupConfig + WarmupSelector
// ---------------------------------------------------------------------------

/// Configuration for warmup engram selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmupConfig {
    /// Maximum number of engrams to return.
    pub max_engrams: usize,
    /// MMR diversity parameter: 1.0 = pure relevance, 0.0 = pure diversity.
    pub diversity_lambda: f64,
    /// Minimum engram strength required for inclusion.
    pub min_strength: f64,
}

impl Default for WarmupConfig {
    fn default() -> Self {
        Self {
            max_engrams: 5,
            diversity_lambda: 0.7,
            min_strength: 0.3,
        }
    }
}

/// Selects diverse, relevant engrams for context warm-up using MMR.
#[derive(Debug)]
pub struct WarmupSelector;

impl WarmupSelector {
    /// Select warmup engrams using Maximal Marginal Relevance for diversity.
    ///
    /// # Algorithm (MMR)
    /// At each step, the next engram is chosen to maximize:
    ///   `score = lambda * relevance - (1 - lambda) * max_similarity_to_already_selected`
    ///
    /// This balances relevance to the cues with diversity among selected engrams.
    pub fn select_warmup_engrams(
        store: &EngramStore,
        cues: &[NodeId],
        vector_index: &dyn VectorIndex,
        spectral: &SpectralEncoder,
        config: &WarmupConfig,
    ) -> Vec<RecallResult> {
        if cues.is_empty() || config.max_engrams == 0 {
            return Vec::new();
        }

        // Start with a broad recall to get candidates.
        let engine = RecallEngine::new();
        let candidates = engine.recall(store, cues, vector_index, spectral, config.max_engrams * 3);

        // Filter by minimum strength.
        let candidates: Vec<RecallResult> = candidates
            .into_iter()
            .filter(|r| r.engram.strength >= config.min_strength)
            .collect();

        if candidates.is_empty() {
            return Vec::new();
        }

        // Pre-compute spectral signatures for all candidates.
        let signatures: Vec<Vec<f64>> = candidates
            .iter()
            .map(|r| spectral.encode(&r.engram.ensemble))
            .collect();

        // Greedy MMR selection.
        let lambda = config.diversity_lambda;
        let mut selected: Vec<usize> = Vec::new();
        let mut remaining: HashSet<usize> = (0..candidates.len()).collect();

        while selected.len() < config.max_engrams && !remaining.is_empty() {
            let mut best_idx = None;
            let mut best_score = f64::NEG_INFINITY;

            for &i in &remaining {
                let relevance = candidates[i].confidence;

                // Compute max similarity to any already-selected engram.
                let max_sim = selected
                    .iter()
                    .map(|&j| cosine_similarity(&signatures[i], &signatures[j]))
                    .fold(0.0f64, f64::max);

                let mmr_score = lambda * relevance - (1.0 - lambda) * max_sim;

                if mmr_score > best_score {
                    best_score = mmr_score;
                    best_idx = Some(i);
                }
            }

            if let Some(idx) = best_idx {
                selected.push(idx);
                remaining.remove(&idx);
            } else {
                break;
            }
        }

        selected
            .into_iter()
            .map(|i| candidates[i].clone())
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Reconsolidation
// ---------------------------------------------------------------------------

/// Reconsolidate an engram by merging new nodes into its ensemble.
///
/// When a prediction error exceeds the given threshold, the engram's ensemble
/// is updated to incorporate the new evidence. This models memory reconsolidation
/// — the process by which recalled memories become labile and are re-stored
/// with modifications.
///
/// # Parameters
/// - `engram`: The engram to reconsolidate (modified in place).
/// - `new_nodes`: New node IDs to potentially merge into the ensemble.
/// - `prediction_error`: Magnitude of the prediction error that triggered reconsolidation.
///
/// # Behavior
/// - If `prediction_error <= RECONSOLIDATION_THRESHOLD`, no changes are made.
/// - New nodes not already in the ensemble are added with a weight proportional
///   to the prediction error.
/// - Existing node weights are slightly decayed to make room for new information.
/// - Strength is boosted (successful recall + reconsolidation = reinforcement).
pub fn reconsolidate(engram: &mut Engram, new_nodes: &[NodeId], prediction_error: f64) {
    const RECONSOLIDATION_THRESHOLD: f64 = 0.1;
    const DECAY_FACTOR: f64 = 0.95;
    const MAX_NEW_WEIGHT: f64 = 0.8;
    const STRENGTH_BOOST: f64 = 0.05;

    if prediction_error <= RECONSOLIDATION_THRESHOLD {
        // Sub-threshold: just boost strength for successful recall.
        engram.strength = (engram.strength + STRENGTH_BOOST).min(1.0);
        return;
    }

    // Collect existing node IDs for fast lookup.
    let existing: HashSet<NodeId> = engram.ensemble.iter().map(|(nid, _)| *nid).collect();

    // Decay existing weights to make room for new information.
    for (_nid, weight) in engram.ensemble.iter_mut() {
        *weight *= DECAY_FACTOR;
    }

    // Add new nodes that are not already in the ensemble.
    // Weight is proportional to prediction error, clamped to MAX_NEW_WEIGHT.
    let new_weight = (prediction_error * 0.5).min(MAX_NEW_WEIGHT);
    for &node_id in new_nodes {
        if !existing.contains(&node_id) {
            engram.ensemble.push((node_id, new_weight));
        }
    }

    // Boost strength: reconsolidation reinforces the memory.
    engram.strength = (engram.strength + STRENGTH_BOOST).min(1.0);

    // Increment recall count (reconsolidation counts as a recall).
    engram.recall_count += 1;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Cosine similarity between two vectors. Returns 0.0 for empty or zero-norm vectors.
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engram::traits::InMemoryVectorIndex;

    fn make_store_and_index() -> (EngramStore, InMemoryVectorIndex, SpectralEncoder) {
        let store = EngramStore::new(None);
        let index = InMemoryVectorIndex::new();
        let spectral = SpectralEncoder::new();

        // Insert a few engrams.
        for i in 1..=5u64 {
            let id = store.next_id();
            let nodes: Vec<(NodeId, f64)> = (i..i + 3).map(|n| (NodeId(n), 1.0 / n as f64)).collect();
            let mut engram = Engram::new(id, nodes.clone());
            engram.strength = 0.5 + (i as f64) * 0.05;

            // Index the spectral signature.
            let sig = spectral.encode(&nodes);
            engram.spectral_signature = sig.clone();
            index.upsert(&id.0.to_string(), &sig);

            store.insert(engram);
        }

        (store, index, spectral)
    }

    #[test]
    fn recall_finds_direct_matches() {
        let (store, index, spectral) = make_store_and_index();
        let engine = RecallEngine::new();

        // Cue with NodeId(2) — should match engrams containing node 2.
        let results = engine.recall(&store, &[NodeId(2)], &index, &spectral, 10);
        assert!(!results.is_empty());

        // All results should have non-negative confidence.
        for r in &results {
            assert!(r.confidence >= 0.0);
        }
    }

    #[test]
    fn recall_returns_sorted_by_confidence() {
        let (store, index, spectral) = make_store_and_index();
        let engine = RecallEngine::new();

        let results = engine.recall(&store, &[NodeId(2), NodeId(3)], &index, &spectral, 10);
        for window in results.windows(2) {
            assert!(window[0].confidence >= window[1].confidence);
        }
    }

    #[test]
    fn recall_respects_k_limit() {
        let (store, index, spectral) = make_store_and_index();
        let engine = RecallEngine::new();

        let results = engine.recall(&store, &[NodeId(2)], &index, &spectral, 2);
        assert!(results.len() <= 2);
    }

    #[test]
    fn recall_empty_cues_returns_empty() {
        let (store, index, spectral) = make_store_and_index();
        let engine = RecallEngine::new();
        let results = engine.recall(&store, &[], &index, &spectral, 10);
        assert!(results.is_empty());
    }

    #[test]
    fn warmup_selector_respects_min_strength() {
        let (store, index, spectral) = make_store_and_index();
        let config = WarmupConfig {
            max_engrams: 10,
            diversity_lambda: 0.7,
            min_strength: 0.7, // Only the strongest engrams.
            ..Default::default()
        };

        let results =
            WarmupSelector::select_warmup_engrams(&store, &[NodeId(3)], &index, &spectral, &config);
        for r in &results {
            assert!(r.engram.strength >= config.min_strength);
        }
    }

    #[test]
    fn warmup_selector_limits_count() {
        let (store, index, spectral) = make_store_and_index();
        let config = WarmupConfig {
            max_engrams: 2,
            diversity_lambda: 0.7,
            min_strength: 0.0,
        };

        let results =
            WarmupSelector::select_warmup_engrams(&store, &[NodeId(3)], &index, &spectral, &config);
        assert!(results.len() <= 2);
    }

    #[test]
    fn reconsolidate_below_threshold_only_boosts_strength() {
        let id = EngramId(1);
        let mut engram = Engram::new(id, vec![(NodeId(1), 1.0), (NodeId(2), 0.5)]);
        let old_strength = engram.strength;
        let old_ensemble_len = engram.ensemble.len();

        reconsolidate(&mut engram, &[NodeId(99)], 0.05);

        assert!(engram.strength > old_strength);
        assert_eq!(engram.ensemble.len(), old_ensemble_len); // No new nodes added.
    }

    #[test]
    fn reconsolidate_above_threshold_adds_nodes() {
        let id = EngramId(1);
        let mut engram = Engram::new(id, vec![(NodeId(1), 1.0), (NodeId(2), 0.5)]);

        reconsolidate(&mut engram, &[NodeId(99), NodeId(100)], 0.5);

        let node_ids: Vec<NodeId> = engram.ensemble.iter().map(|(nid, _)| *nid).collect();
        assert!(node_ids.contains(&NodeId(99)));
        assert!(node_ids.contains(&NodeId(100)));
        assert_eq!(engram.recall_count, 1);
    }

    #[test]
    fn reconsolidate_does_not_duplicate_existing_nodes() {
        let id = EngramId(1);
        let mut engram = Engram::new(id, vec![(NodeId(1), 1.0), (NodeId(2), 0.5)]);

        reconsolidate(&mut engram, &[NodeId(1), NodeId(99)], 0.5);

        let count_node1 = engram
            .ensemble
            .iter()
            .filter(|(nid, _)| *nid == NodeId(1))
            .count();
        assert_eq!(count_node1, 1); // Not duplicated.
    }

    #[test]
    fn cosine_similarity_identical_vectors() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn cosine_similarity_orthogonal_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-10);
    }
}
