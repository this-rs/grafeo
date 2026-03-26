//! Recall engine, warmup selection, and reconsolidation.
//!
//! This module implements a two-path recall mechanism:
//! 1. **Direct recall** — find engrams whose ensemble contains the cue nodes.
//! 2. **Hopfield recall** — Modern Hopfield attention: `softmax(β × P^T × q) × P`
//!    with per-engram precision β. Falls back to VectorIndex if no patterns are
//!    loaded in the Hopfield matrix.
//!
//! Results are merged, deduplicated, and ranked by a combined confidence score.
//!
//! The [`WarmupSelector`] adds MMR-based diversity to avoid returning a cluster
//! of near-identical engrams. [`reconsolidate`] handles memory updating when
//! prediction error triggers ensemble modification.

use std::collections::{HashMap, HashSet};

use grafeo_common::types::NodeId;
use serde::{Deserialize, Serialize};

use super::hopfield::{self, MmrResult, PatternMatrix};
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
        Self { direct_weight: 0.6 }
    }

    /// Recall engrams matching the given cue nodes.
    ///
    /// # Two-path recall
    /// 1. **Direct**: finds engrams whose ensemble contains any of the cue nodes
    ///    and scores them by overlap fraction.
    /// 2. **Hopfield**: encodes cues as a spectral vector and uses Modern Hopfield
    ///    retrieval (`softmax(β × P^T × q)`) with per-engram precision β.
    ///    Falls back to VectorIndex nearest-neighbor if no Hopfield patterns are loaded.
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
        let mut scores: HashMap<EngramId, (f64, f64)> = HashMap::new(); // (direct, hopfield)

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

        // -- Path 2: Modern Hopfield retrieval (replaces basic VectorIndex lookup) --
        let cue_ensemble: Vec<(NodeId, f64)> = cues.iter().map(|&nid| (nid, 1.0)).collect();
        let query_vec = spectral.encode(&cue_ensemble);
        let dim = spectral.dimensions();

        // Build Hopfield pattern matrix from the store's spectral signatures.
        let matrix = PatternMatrix::from_store(store, dim);

        if !matrix.is_empty() {
            // Use Modern Hopfield attention: softmax(β × P^T × q)
            let hopfield_results = hopfield::hopfield_retrieve(&matrix, &query_vec, store, k * 2);
            for hr in &hopfield_results {
                let entry = scores.entry(hr.engram_id).or_insert((0.0, 0.0));
                // Attention weight is already in [0, 1] and sums to 1 — use it
                // as the "spectral" score. Scale by the number of patterns so
                // that a dominant Hopfield match produces a score close to 1.0.
                let hopfield_score = (hr.attention_weight * matrix.len() as f64).min(1.0);
                if hopfield_score > entry.1 {
                    entry.1 = hopfield_score;
                }
            }
        } else {
            // Fallback: no spectral signatures indexed → use VectorIndex (Phase 1 compat)
            let spectral_results = vector_index.nearest(&query_vec, k * 2);
            for (id_str, distance) in &spectral_results {
                if let Ok(raw_id) = id_str.parse::<u64>() {
                    let engram_id = EngramId(raw_id);
                    let similarity = (1.0 - distance).clamp(0.0, 1.0);
                    let entry = scores.entry(engram_id).or_insert((0.0, 0.0));
                    if similarity > entry.1 {
                        entry.1 = similarity;
                    }
                }
            }
        }

        // -- Merge and rank --
        let spectral_weight = 1.0 - self.direct_weight;
        let mut results: Vec<RecallResult> = scores
            .into_iter()
            .filter_map(|(engram_id, (direct_score, hopfield_score))| {
                let engram = store.get(engram_id)?;
                let confidence = (self.direct_weight * direct_score
                    + spectral_weight * hopfield_score)
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

/// An activated engram selected for context warm-up.
///
/// The dominant engram is enriched with full detail, while secondaries
/// carry only a condensed summary of their ensemble.
#[derive(Debug, Clone)]
pub struct ActivatedEngram {
    /// The engram identifier.
    pub engram_id: EngramId,
    /// Relevance score from the selection pipeline.
    pub relevance: f64,
    /// MMR score at selection time.
    pub mmr_score: f64,
    /// The full engram data.
    pub engram: Engram,
    /// Whether this is the dominant (highest-activation) engram.
    pub is_dominant: bool,
    /// Summary: for the dominant this is `DetailLevel::Full`, for secondaries
    /// it is `DetailLevel::Summary` with a condensed representation.
    pub detail: DetailLevel,
}

/// Detail level for an activated engram in the warm-up context.
#[derive(Debug, Clone)]
pub enum DetailLevel {
    /// Full detail — all ensemble nodes with weights.
    Full,
    /// Condensed summary — only the top-N contributing nodes.
    Summary {
        /// The top contributing nodes (sorted by weight descending).
        top_nodes: Vec<(NodeId, f64)>,
    },
}

/// Selects diverse, relevant engrams for context warm-up using the full
/// Hopfield → softmax_compete → MMR pipeline.
#[derive(Debug)]
pub struct WarmupSelector;

impl WarmupSelector {
    /// Select warmup engrams using the full Phase 3 pipeline.
    ///
    /// # Pipeline
    /// 1. **Hopfield retrieve** — top-K candidates via Modern Hopfield attention
    /// 2. **Softmax compete** — lateral inhibition eliminates weak candidates (threshold 0.15)
    /// 3. **MMR** — Max Marginal Relevance ensures diversity (λ=0.7)
    /// 4. **Truncate** — limit to budget
    /// 5. **Enrich** — dominant in detail, secondaries in summary
    ///
    /// Falls back to the basic recall path when no spectral signatures are available
    /// (Phase 1 backward compatibility).
    pub fn select_warmup_engrams(
        store: &EngramStore,
        cues: &[NodeId],
        vector_index: &dyn VectorIndex,
        spectral: &SpectralEncoder,
        config: &WarmupConfig,
    ) -> Vec<ActivatedEngram> {
        if cues.is_empty() || config.max_engrams == 0 {
            return Vec::new();
        }

        let budget = config.max_engrams;
        let dim = spectral.dimensions();

        // Build the Hopfield pattern matrix from stored spectral signatures.
        let matrix = PatternMatrix::from_store(store, dim);

        // Encode query from cues.
        let cue_ensemble: Vec<(NodeId, f64)> = cues.iter().map(|&nid| (nid, 1.0)).collect();
        let query_vec = spectral.encode(&cue_ensemble);

        if !matrix.is_empty() {
            // --- Phase 3 pipeline: Hopfield → softmax → MMR → truncate → enrich ---

            // Step 1: Hopfield retrieve top-K candidates (3× budget for headroom).
            let candidates =
                hopfield::hopfield_retrieve(&matrix, &query_vec, store, budget * 3);

            if candidates.is_empty() {
                return Vec::new();
            }

            // Filter by minimum strength before competition.
            let candidates: Vec<_> = candidates
                .into_iter()
                .filter(|c| c.engram.strength >= config.min_strength)
                .collect();

            if candidates.is_empty() {
                return Vec::new();
            }

            // Step 2: Softmax competitive activation with lateral inhibition.
            let competed = hopfield::softmax_compete(&candidates, 0.15);

            if competed.is_empty() {
                return Vec::new();
            }

            // Step 3: MMR for diversity.
            let mmr_results =
                hopfield::max_marginal_relevance(&competed, config.diversity_lambda);

            // Step 4: Truncate to budget.
            let truncated: Vec<_> = mmr_results.into_iter().take(budget).collect();

            // Step 5: Enrich — dominant gets full detail, secondaries get summary.
            enrich_with_background(truncated)
        } else {
            // --- Fallback: Phase 1 compatible path (no spectral signatures) ---
            let engine = RecallEngine::new();
            let recall_results =
                engine.recall(store, cues, vector_index, spectral, budget * 3);

            let filtered: Vec<_> = recall_results
                .into_iter()
                .filter(|r| r.engram.strength >= config.min_strength)
                .take(budget)
                .collect();

            // Convert to ActivatedEngram with first as dominant.
            filtered
                .into_iter()
                .enumerate()
                .map(|(i, r)| {
                    let is_dominant = i == 0;
                    ActivatedEngram {
                        engram_id: r.engram_id,
                        relevance: r.confidence,
                        mmr_score: r.confidence,
                        engram: r.engram.clone(),
                        is_dominant,
                        detail: if is_dominant {
                            DetailLevel::Full
                        } else {
                            make_summary(&r.engram)
                        },
                    }
                })
                .collect()
        }
    }
}

/// Enrich MMR results: dominant engram gets full detail, secondaries get condensed summary.
fn enrich_with_background(results: Vec<MmrResult>) -> Vec<ActivatedEngram> {
    results
        .into_iter()
        .map(|r| {
            let detail = if r.is_dominant {
                DetailLevel::Full
            } else {
                make_summary(&r.engram)
            };
            ActivatedEngram {
                engram_id: r.engram_id,
                relevance: r.relevance,
                mmr_score: r.mmr_score,
                engram: r.engram,
                is_dominant: r.is_dominant,
                detail,
            }
        })
        .collect()
}

/// Create a summary detail level from an engram — top 3 contributing nodes.
fn make_summary(engram: &Engram) -> DetailLevel {
    const SUMMARY_TOP_N: usize = 3;
    let mut top: Vec<(NodeId, f64)> = engram.ensemble.clone();
    top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    top.truncate(SUMMARY_TOP_N);
    DetailLevel::Summary { top_nodes: top }
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
            let nodes: Vec<(NodeId, f64)> =
                (i..i + 3).map(|n| (NodeId(n), 1.0 / n as f64)).collect();
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

    // -- warmup_pipeline end-to-end test --

    #[test]
    fn warmup_pipeline_end_to_end() {
        use std::time::Instant;

        let store = EngramStore::new(None);
        let index = InMemoryVectorIndex::new();
        let spectral = SpectralEncoder::new();
        let _dim = spectral.dimensions();

        // Create 50 engrams with spectral signatures for the Hopfield path.
        // Group them into clusters to test diversity:
        //   - Cluster A (25 engrams): nodes 1..=3, high overlap
        //   - Cluster B (15 engrams): nodes 50..=52, different region
        //   - Cluster C (10 engrams): nodes 100..=102, another region
        for i in 0..50u64 {
            let id = store.next_id();
            let nodes: Vec<(NodeId, f64)> = if i < 25 {
                // Cluster A: variations around nodes 1-3
                vec![
                    (NodeId(1), 1.0),
                    (NodeId(2), 0.8 + (i as f64) * 0.005),
                    (NodeId(3), 0.5),
                ]
            } else if i < 40 {
                // Cluster B
                vec![
                    (NodeId(50), 1.0),
                    (NodeId(51), 0.7 + (i as f64) * 0.005),
                    (NodeId(52), 0.4),
                ]
            } else {
                // Cluster C
                vec![
                    (NodeId(100), 1.0),
                    (NodeId(101), 0.6 + (i as f64) * 0.005),
                    (NodeId(102), 0.3),
                ]
            };

            let mut engram = Engram::new(id, nodes.clone());
            engram.strength = 0.5 + (i as f64) * 0.008;
            let sig = spectral.encode(&nodes);
            engram.spectral_signature = sig.clone();
            engram.precision = 2.0 + (i as f64) * 0.1;
            index.upsert(&id.0.to_string(), &sig);
            store.insert(engram);
        }

        assert_eq!(store.count(), 50);

        let config = WarmupConfig {
            max_engrams: 3,
            diversity_lambda: 0.7,
            min_strength: 0.3,
        };

        // Query with cues from cluster A.
        let cues = vec![NodeId(1), NodeId(2)];

        let start = Instant::now();
        let results =
            WarmupSelector::select_warmup_engrams(&store, &cues, &index, &spectral, &config);
        let elapsed = start.elapsed();

        // Performance: must be < 50ms.
        assert!(
            elapsed.as_millis() < 50,
            "Warmup pipeline took {}ms, expected < 50ms",
            elapsed.as_millis()
        );

        // Budget: at most 3 results.
        assert!(
            results.len() <= 3,
            "Expected <= 3 results, got {}",
            results.len()
        );
        assert!(
            !results.is_empty(),
            "Expected at least 1 result"
        );

        // Exactly one dominant.
        let dominant_count = results.iter().filter(|r| r.is_dominant).count();
        assert_eq!(dominant_count, 1, "Expected exactly 1 dominant engram");

        // Dominant should be first.
        assert!(results[0].is_dominant, "First result should be dominant");

        // Dominant has Full detail, secondaries have Summary detail.
        for r in &results {
            match (&r.detail, r.is_dominant) {
                (DetailLevel::Full, true) => {} // OK
                (DetailLevel::Summary { top_nodes }, false) => {
                    assert!(!top_nodes.is_empty(), "Summary should have top nodes");
                }
                _ => panic!(
                    "Unexpected detail level for engram {} (dominant={})",
                    r.engram_id, r.is_dominant
                ),
            }
        }

        // Diversity check: if we have 3 results, they shouldn't all be from the
        // same cluster. The MMR should have pulled in at least one from a different
        // cluster. We check via the spectral signatures — they should not all be
        // near-identical.
        if results.len() == 3 {
            let sigs: Vec<&[f64]> = results.iter().map(|r| r.engram.spectral_signature.as_slice()).collect();
            let sim_01 = cosine_similarity(sigs[0], sigs[1]);
            let sim_02 = cosine_similarity(sigs[0], sigs[2]);
            let sim_12 = cosine_similarity(sigs[1], sigs[2]);
            let _max_sim = sim_01.max(sim_02).max(sim_12);
            // At least one pair should have lower similarity (diversity).
            let min_sim = sim_01.min(sim_02).min(sim_12);
            assert!(
                min_sim < 0.99,
                "Expected diversity in results but all pairwise similarities are high: {:.3}, {:.3}, {:.3}",
                sim_01, sim_02, sim_12
            );
        }
    }

    #[test]
    fn warmup_pipeline_empty_cues() {
        let store = EngramStore::new(None);
        let index = InMemoryVectorIndex::new();
        let spectral = SpectralEncoder::new();
        let config = WarmupConfig::default();

        let results =
            WarmupSelector::select_warmup_engrams(&store, &[], &index, &spectral, &config);
        assert!(results.is_empty());
    }

    #[test]
    fn warmup_pipeline_fallback_no_signatures() {
        // Test the Phase 1 fallback path (no spectral signatures in store).
        let (store, index, spectral) = make_store_and_index();

        // Clear spectral signatures to force fallback.
        for engram in store.list() {
            store.update(engram.id, |e| {
                e.spectral_signature.clear();
            });
        }

        let config = WarmupConfig {
            max_engrams: 3,
            diversity_lambda: 0.7,
            min_strength: 0.0,
        };

        let results = WarmupSelector::select_warmup_engrams(
            &store,
            &[NodeId(3)],
            &index,
            &spectral,
            &config,
        );
        // Should still work via fallback.
        // (May be empty if VectorIndex returns no results, but should not panic.)
        assert!(results.len() <= 3);
    }
}
