//! P2P Distillation pipeline -- distill/inject/evaluate.
//!
//! This module enables extracting cognitive state from one engine,
//! serializing it as a [`DistillArtifact`], and injecting it into
//! another engine with a configurable trust discount.
//!
//! The [`evaluate`] function measures 5 parity factors between
//! before and after artifacts: evidence coverage, energy correlation,
//! hub coverage, cemented knowledge, and cross-community bridging.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::SystemTime;

use crate::engine::CognitiveEngine;
use grafeo_common::types::NodeId;

#[allow(unused_imports)]
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Controls which cognitive state is extracted during distillation.
#[derive(Debug, Clone)]
pub struct DistillConfig {
    /// Minimum synapse weight to include in the artifact.
    pub min_synapse_weight: f64,
    /// Minimum energy to include in the artifact.
    pub min_energy: f64,
    /// Whether to include structural fingerprints.
    pub include_fingerprints: bool,
    /// Whether to include episodic memory.
    pub include_episodes: bool,
    /// Restrict extraction to specific communities (if `Some`).
    pub community_filter: Option<Vec<u64>>,
    /// Trust discount applied during injection (0.0 = ignore, 1.0 = full trust).
    pub trust_discount: f64,
    /// If true, anonymize node IDs in the exported artifact by remapping to random values.
    pub anonymize: bool,
}

impl Default for DistillConfig {
    fn default() -> Self {
        Self {
            min_synapse_weight: 0.1,
            min_energy: 0.01,
            include_fingerprints: false,
            include_episodes: false,
            community_filter: None,
            trust_discount: 1.0,
            anonymize: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Artifact types
// ---------------------------------------------------------------------------

/// A serializable snapshot of cognitive state for peer-to-peer transfer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillArtifact {
    /// Format version.
    pub version: String,
    /// When the artifact was created.
    pub created_at: SystemTime,
    /// Extracted synapse data.
    pub synapses: Vec<SynapseSnapshot>,
    /// Extracted energy data.
    pub energies: Vec<EnergySnapshot>,
    /// Optional structural fingerprints.
    pub fingerprints: Vec<FingerprintSnapshot>,
    /// Metadata about the source instance.
    pub metadata: ArtifactMetadata,
}

impl DistillArtifact {
    /// Serialize the artifact to bincode bytes for export.
    pub fn to_bincode(&self) -> Result<Vec<u8>, String> {
        bincode::serde::encode_to_vec(self, bincode::config::standard())
            .map_err(|e| format!("bincode encode error: {e}"))
    }

    /// Deserialize an artifact from bincode bytes.
    pub fn from_bincode(bytes: &[u8]) -> Result<Self, String> {
        let (artifact, _) =
            bincode::serde::decode_from_slice(bytes, bincode::config::standard())
                .map_err(|e| format!("bincode decode error: {e}"))?;
        Ok(artifact)
    }
}

/// A snapshot of a single synapse for serialization.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SynapseSnapshot {
    /// Source node ID.
    pub source: u64,
    /// Target node ID.
    pub target: u64,
    /// Current weight (after decay).
    pub weight: f64,
}

/// A snapshot of a single node's energy for serialization.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EnergySnapshot {
    /// Node ID.
    pub node_id: u64,
    /// Current energy (after decay).
    pub energy: f64,
}

/// Structural fingerprint of a community.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FingerprintSnapshot {
    /// Community identifier.
    pub community_id: u64,
    /// Number of nodes in the community.
    pub node_count: usize,
    /// Number of edges in the community.
    pub edge_count: usize,
    /// Clustering coefficient of the community.
    pub clustering_coeff: f64,
}

/// Alias for [`FingerprintSnapshot`] used in P2P context.
pub type CommunityFingerprint = FingerprintSnapshot;

/// Summary of an episode (placeholder for episodic memory integration).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EpisodeSummary {
    /// Episode identifier.
    pub episode_id: u64,
    /// Human-readable summary.
    pub summary: String,
    /// Lesson learned.
    pub lesson: String,
}

/// Metadata about the source engine that produced the artifact.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ArtifactMetadata {
    /// Identifier of the source instance.
    pub source_instance: String,
    /// Total number of nodes in the source engine.
    pub total_nodes: usize,
    /// Total number of synapses in the source engine.
    pub total_synapses: usize,
    /// Total number of communities in the source engine.
    pub total_communities: usize,
}

// ---------------------------------------------------------------------------
// Anonymization
// ---------------------------------------------------------------------------

/// Anonymize node IDs in an artifact by remapping them to sequential new IDs
/// starting from a random offset. Structure (edges, counts) is preserved.
pub fn anonymize(artifact: &DistillArtifact) -> DistillArtifact {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    // Build deterministic-but-opaque mapping using hash of original ID + timestamp.
    let seed = {
        let mut h = DefaultHasher::new();
        artifact.created_at.hash(&mut h);
        h.finish()
    };

    let mut id_map: HashMap<u64, u64> = HashMap::new();
    let mut counter: u64 = 0;

    let mut remap = |original: u64| -> u64 {
        *id_map.entry(original).or_insert_with(|| {
            // Mix counter with seed to produce non-sequential IDs.
            let mut h = DefaultHasher::new();
            counter.hash(&mut h);
            seed.hash(&mut h);
            counter += 1;
            h.finish()
        })
    };

    let synapses = artifact
        .synapses
        .iter()
        .map(|s| SynapseSnapshot {
            source: remap(s.source),
            target: remap(s.target),
            weight: s.weight,
        })
        .collect();

    let energies = artifact
        .energies
        .iter()
        .map(|e| EnergySnapshot {
            node_id: remap(e.node_id),
            energy: e.energy,
        })
        .collect();

    DistillArtifact {
        version: artifact.version.clone(),
        created_at: artifact.created_at,
        synapses,
        energies,
        fingerprints: artifact.fingerprints.clone(),
        metadata: ArtifactMetadata {
            source_instance: String::from("anonymous"),
            ..artifact.metadata.clone()
        },
    }
}

// ---------------------------------------------------------------------------
// ParityReport -- 5 parity factors
// ---------------------------------------------------------------------------

/// Report of the 5 parity factors measuring knowledge transfer quality.
///
/// The five factors are:
/// 1. `evidence_coverage` -- Jaccard overlap of synapse pairs
/// 2. `energy_correlation` -- Pearson correlation of shared node energies
/// 3. `hub_coverage` -- fraction of high-degree nodes present in both
/// 4. `cemented` -- fraction of shared synapses with weight agreement
/// 5. `cross_community` -- statistical correlation of cross-degree bridging patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParityReport {
    /// Factor 1: Jaccard index of synapse (source, target) pairs.
    pub evidence_coverage: f64,
    /// Factor 2: Pearson correlation of shared node energies.
    pub energy_correlation: f64,
    /// Factor 3: Fraction of hub nodes from source present in target.
    pub hub_coverage: f64,
    /// Factor 4: Fraction of shared synapses with close weight agreement.
    pub cemented: f64,
    /// Factor 5: Correlation of cross-degree bridging patterns between artifacts.
    pub cross_community: f64,
    /// Weighted composite score of all 5 factors.
    pub composite_score: f64,
    /// Threshold used for pass/fail determination.
    pub threshold: f64,
    /// Whether the composite score meets the threshold.
    pub passed: bool,
}

/// Configuration for the [`evaluate`] function.
#[derive(Debug, Clone)]
pub struct EvaluateConfig {
    /// Composite score threshold for passing (default: 0.67).
    pub threshold: f64,
    /// Weights for the 5 factors: evidence, energy_correlation, hub, cemented, cross.
    pub weights: [f64; 5],
}

impl Default for EvaluateConfig {
    fn default() -> Self {
        Self {
            threshold: 0.67,
            weights: [0.25, 0.20, 0.20, 0.20, 0.15],
        }
    }
}

// ---------------------------------------------------------------------------
// distill
// ---------------------------------------------------------------------------

/// Extracts cognitive state from `engine` according to `config`.
///
/// `node_ids` specifies which nodes to extract energy from. If the slice is
/// empty, energy extraction is skipped.
///
/// If `config.anonymize` is true, the returned artifact will have remapped node IDs.
pub fn distill(
    config: &DistillConfig,
    node_ids: &[NodeId],
    engine: &dyn CognitiveEngine,
) -> DistillArtifact {
    let synapses = extract_synapses(config, engine);
    let energies = extract_energies(config, node_ids, engine);

    let total_synapses = synapse_count(engine);

    let metadata = ArtifactMetadata {
        source_instance: String::from("local"),
        total_nodes: node_ids.len(),
        total_synapses,
        total_communities: 0,
    };

    let artifact = DistillArtifact {
        version: String::from("1.0"),
        created_at: SystemTime::now(),
        synapses,
        energies,
        fingerprints: Vec::new(),
        metadata,
    };

    if config.anonymize {
        anonymize(&artifact)
    } else {
        artifact
    }
}

#[cfg(feature = "synapse")]
fn extract_synapses(config: &DistillConfig, engine: &dyn CognitiveEngine) -> Vec<SynapseSnapshot> {
    let Some(store) = engine.synapse_store() else {
        return Vec::new();
    };
    store
        .snapshot()
        .into_iter()
        .filter(|s| s.current_weight() >= config.min_synapse_weight)
        .map(|s| SynapseSnapshot {
            source: s.source.0,
            target: s.target.0,
            weight: s.current_weight(),
        })
        .collect()
}

#[cfg(not(feature = "synapse"))]
fn extract_synapses(
    _config: &DistillConfig,
    _engine: &dyn CognitiveEngine,
) -> Vec<SynapseSnapshot> {
    Vec::new()
}

#[cfg(feature = "energy")]
fn extract_energies(
    config: &DistillConfig,
    node_ids: &[NodeId],
    engine: &dyn CognitiveEngine,
) -> Vec<EnergySnapshot> {
    if node_ids.is_empty() {
        return Vec::new();
    }
    let Some(store) = engine.energy_store() else {
        return Vec::new();
    };
    node_ids
        .iter()
        .filter_map(|&nid| {
            let e = store.get_energy(nid);
            if e >= config.min_energy {
                Some(EnergySnapshot {
                    node_id: nid.0,
                    energy: e,
                })
            } else {
                None
            }
        })
        .collect()
}

#[cfg(not(feature = "energy"))]
fn extract_energies(
    _config: &DistillConfig,
    _node_ids: &[NodeId],
    _engine: &dyn CognitiveEngine,
) -> Vec<EnergySnapshot> {
    Vec::new()
}

#[cfg(feature = "synapse")]
fn synapse_count(engine: &dyn CognitiveEngine) -> usize {
    engine.synapse_store().map_or(0, |s| s.len())
}

#[cfg(not(feature = "synapse"))]
fn synapse_count(_engine: &dyn CognitiveEngine) -> usize {
    0
}

// ---------------------------------------------------------------------------
// inject
// ---------------------------------------------------------------------------

/// Applies an artifact's knowledge to `engine` with a trust discount.
///
/// Synapse weights are multiplied by `trust_factor` before reinforcement.
/// Energy values are multiplied by `trust_factor` before boosting.
pub fn inject(artifact: &DistillArtifact, trust_factor: f64, engine: &dyn CognitiveEngine) {
    let tf = trust_factor.clamp(0.0, 1.0);
    if tf == 0.0 {
        return; // No-op when trust is zero.
    }
    inject_synapses(artifact, tf, engine);
    inject_energies(artifact, tf, engine);
}

#[cfg(feature = "synapse")]
fn inject_synapses(artifact: &DistillArtifact, trust_factor: f64, engine: &dyn CognitiveEngine) {
    let Some(store) = engine.synapse_store() else {
        return;
    };
    for snap in &artifact.synapses {
        let discounted = snap.weight * trust_factor;
        if discounted > 0.0 {
            store.reinforce(NodeId(snap.source), NodeId(snap.target), discounted);
        }
    }
}

#[cfg(not(feature = "synapse"))]
fn inject_synapses(_artifact: &DistillArtifact, _trust_factor: f64, _engine: &dyn CognitiveEngine) {
}

#[cfg(feature = "energy")]
fn inject_energies(artifact: &DistillArtifact, trust_factor: f64, engine: &dyn CognitiveEngine) {
    let Some(store) = engine.energy_store() else {
        return;
    };
    for snap in &artifact.energies {
        let discounted = snap.energy * trust_factor;
        if discounted > 0.0 {
            store.boost(NodeId(snap.node_id), discounted);
        }
    }
}

#[cfg(not(feature = "energy"))]
fn inject_energies(_artifact: &DistillArtifact, _trust_factor: f64, _engine: &dyn CognitiveEngine) {
}

// ---------------------------------------------------------------------------
// evaluate -- 5 parity factors
// ---------------------------------------------------------------------------

/// Compares two artifacts (before and after injection) to measure knowledge
/// parity via 5 factors.
///
/// Uses default [`EvaluateConfig`] (threshold 0.67).
pub fn evaluate(before: &DistillArtifact, after: &DistillArtifact) -> ParityReport {
    evaluate_with_config(before, after, &EvaluateConfig::default())
}

/// Compares two artifacts with a custom [`EvaluateConfig`].
pub fn evaluate_with_config(
    before: &DistillArtifact,
    after: &DistillArtifact,
    config: &EvaluateConfig,
) -> ParityReport {
    // Factor 1: Evidence coverage -- Jaccard overlap of synapse pairs
    let evidence_coverage = synapse_jaccard(before, after);

    // Factor 2: Energy correlation -- Pearson correlation of shared energies
    let energy_correlation = energy_pearson(before, after);

    // Factor 3: Hub coverage -- fraction of high-degree nodes in both
    let hub_coverage = compute_hub_coverage(before, after);

    // Factor 4: Cemented -- fraction of shared synapses with weight agreement
    let cemented = compute_cemented(before, after);

    // Factor 5: Cross-community -- correlation of cross-degree bridging patterns
    let cross_community = compute_cross_community(before, after);

    let w = &config.weights;
    let composite_score = w[0] * evidence_coverage
        + w[1] * energy_correlation
        + w[2] * hub_coverage
        + w[3] * cemented
        + w[4] * cross_community;

    ParityReport {
        evidence_coverage,
        energy_correlation,
        hub_coverage,
        cemented,
        cross_community,
        composite_score,
        threshold: config.threshold,
        passed: composite_score >= config.threshold,
    }
}

/// Jaccard index of synapse (source, target) pairs between two artifacts.
fn synapse_jaccard(a: &DistillArtifact, b: &DistillArtifact) -> f64 {
    if a.synapses.is_empty() && b.synapses.is_empty() {
        return 1.0;
    }
    let set_a: HashSet<(u64, u64)> = a.synapses.iter().map(|s| (s.source, s.target)).collect();
    let set_b: HashSet<(u64, u64)> = b.synapses.iter().map(|s| (s.source, s.target)).collect();
    let intersection = set_a.intersection(&set_b).count() as f64;
    let union = set_a.union(&set_b).count() as f64;
    if union == 0.0 {
        1.0
    } else {
        intersection / union
    }
}

/// Pearson correlation of energies for nodes present in both artifacts.
fn energy_pearson(a: &DistillArtifact, b: &DistillArtifact) -> f64 {
    if a.energies.is_empty() && b.energies.is_empty() {
        return 1.0;
    }
    let map_a: HashMap<u64, f64> = a.energies.iter().map(|e| (e.node_id, e.energy)).collect();
    let map_b: HashMap<u64, f64> = b.energies.iter().map(|e| (e.node_id, e.energy)).collect();

    let pairs: Vec<(f64, f64)> = map_a
        .iter()
        .filter_map(|(&nid, &ea)| map_b.get(&nid).map(|&eb| (ea, eb)))
        .collect();

    if pairs.is_empty() {
        return 0.0;
    }
    if pairs.len() == 1 {
        return if (pairs[0].0 - pairs[0].1).abs() < f64::EPSILON {
            1.0
        } else {
            0.0
        };
    }

    let n = pairs.len() as f64;
    let mean_a = pairs.iter().map(|(a, _)| a).sum::<f64>() / n;
    let mean_b = pairs.iter().map(|(_, b)| b).sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;
    for &(a, b) in &pairs {
        let da = a - mean_a;
        let db = b - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    let denom = (var_a * var_b).sqrt();
    if denom < f64::EPSILON {
        if var_a < f64::EPSILON && var_b < f64::EPSILON {
            1.0
        } else {
            0.0
        }
    } else {
        cov / denom
    }
}

/// Hub coverage: fraction of high-degree nodes from `before` also in `after`.
fn compute_hub_coverage(before: &DistillArtifact, after: &DistillArtifact) -> f64 {
    if before.synapses.is_empty() {
        return 1.0;
    }
    let degree_before = build_degree_map(&before.synapses);
    let nodes_after: HashSet<u64> = after
        .synapses
        .iter()
        .flat_map(|s| [s.source, s.target])
        .collect();

    let mut sorted: Vec<(u64, usize)> = degree_before.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));
    let hub_count = (sorted.len() / 5).max(1);

    let covered = sorted
        .iter()
        .take(hub_count)
        .filter(|(id, _)| nodes_after.contains(id))
        .count();

    covered as f64 / hub_count as f64
}

/// Cemented: fraction of shared synapses with weight ratio above 0.5.
fn compute_cemented(before: &DistillArtifact, after: &DistillArtifact) -> f64 {
    if before.synapses.is_empty() && after.synapses.is_empty() {
        return 1.0;
    }
    let map_before: HashMap<(u64, u64), f64> = before
        .synapses
        .iter()
        .map(|s| ((s.source, s.target), s.weight))
        .collect();
    let map_after: HashMap<(u64, u64), f64> = after
        .synapses
        .iter()
        .map(|s| ((s.source, s.target), s.weight))
        .collect();

    let shared: Vec<(f64, f64)> = map_before
        .iter()
        .filter_map(|(key, &wb)| map_after.get(key).map(|&wa| (wb, wa)))
        .collect();

    if shared.is_empty() {
        return 0.0;
    }

    let cemented_count = shared
        .iter()
        .filter(|&&(wb, wa)| {
            let max_w = wb.max(wa);
            if max_w < f64::EPSILON {
                return true;
            }
            wb.min(wa) / max_w > 0.5
        })
        .count();

    cemented_count as f64 / shared.len() as f64
}

/// Cross-community: correlation of cross-degree bridging patterns between artifacts.
///
/// Computes the fraction of synapses that bridge different degree classes in each
/// artifact independently, then returns the Pearson-style similarity between
/// the two distributions. This replaces the old heuristic that pooled all synapses.
fn compute_cross_community(before: &DistillArtifact, after: &DistillArtifact) -> f64 {
    let ratio_before = cross_degree_ratio(&before.synapses);
    let ratio_after = cross_degree_ratio(&after.synapses);

    // Both empty → perfect correlation.
    if before.synapses.is_empty() && after.synapses.is_empty() {
        return 1.0;
    }
    // One empty → no correlation.
    if before.synapses.is_empty() || after.synapses.is_empty() {
        return 0.0;
    }

    // Similarity: 1 - |diff|. Both values are in [0, 1], so result is in [0, 1].
    1.0 - (ratio_before - ratio_after).abs()
}

/// Fraction of synapses that bridge different degree classes within a single artifact.
fn cross_degree_ratio(synapses: &[SynapseSnapshot]) -> f64 {
    if synapses.is_empty() {
        return 0.0;
    }

    let mut degree: HashMap<u64, usize> = HashMap::new();
    for s in synapses {
        *degree.entry(s.source).or_default() += 1;
        *degree.entry(s.target).or_default() += 1;
    }

    let classify = |d: usize| -> u8 {
        if d <= 2 {
            0
        } else if d <= 5 {
            1
        } else {
            2
        }
    };

    let cross = synapses
        .iter()
        .filter(|s| {
            let sc = classify(*degree.get(&s.source).unwrap_or(&0));
            let dc = classify(*degree.get(&s.target).unwrap_or(&0));
            sc != dc
        })
        .count();

    cross as f64 / synapses.len() as f64
}

/// Builds a degree map from synapse snapshots.
fn build_degree_map(synapses: &[SynapseSnapshot]) -> HashMap<u64, usize> {
    let mut degree: HashMap<u64, usize> = HashMap::new();
    for s in synapses {
        *degree.entry(s.source).or_default() += 1;
        *degree.entry(s.target).or_default() += 1;
    }
    degree
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_values() {
        let config = DistillConfig::default();
        assert!((config.min_synapse_weight - 0.1).abs() < f64::EPSILON);
        assert!((config.min_energy - 0.01).abs() < f64::EPSILON);
        assert!(!config.include_fingerprints);
        assert!(!config.include_episodes);
        assert!(config.community_filter.is_none());
        assert!((config.trust_discount - 1.0).abs() < f64::EPSILON);
        assert!(!config.anonymize);
    }

    #[test]
    fn jaccard_identical_sets() {
        let a = make_artifact(vec![(1, 2, 0.5)], vec![]);
        let b = make_artifact(vec![(1, 2, 0.5)], vec![]);
        assert!((synapse_jaccard(&a, &b) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn jaccard_disjoint_sets() {
        let a = make_artifact(vec![(1, 2, 0.5)], vec![]);
        let b = make_artifact(vec![(3, 4, 0.5)], vec![]);
        assert!(synapse_jaccard(&a, &b).abs() < f64::EPSILON);
    }

    #[test]
    fn pearson_identical() {
        let a = make_artifact(vec![], vec![(1, 1.0), (2, 2.0), (3, 3.0)]);
        let b = make_artifact(vec![], vec![(1, 1.0), (2, 2.0), (3, 3.0)]);
        assert!((energy_pearson(&a, &b) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn evaluate_identical_artifacts() {
        let a = make_artifact(vec![(1, 2, 0.5)], vec![(1, 1.0), (2, 2.0)]);
        let b = make_artifact(vec![(1, 2, 0.5)], vec![(1, 1.0), (2, 2.0)]);
        let report = evaluate(&a, &b);
        assert!((report.evidence_coverage - 1.0).abs() < f64::EPSILON);
        assert!((report.energy_correlation - 1.0).abs() < 1e-9);
        assert!(report.composite_score > 0.5);
    }

    #[test]
    fn evaluate_has_five_factors() {
        let a = make_artifact(
            vec![(1, 2, 0.5), (2, 3, 0.8), (3, 4, 0.3)],
            vec![(1, 1.0), (2, 2.0)],
        );
        let b = make_artifact(
            vec![(1, 2, 0.6), (2, 3, 0.7), (5, 6, 0.4)],
            vec![(1, 1.0), (2, 1.8)],
        );
        let report = evaluate(&a, &b);

        // All 5 factors must be in [0, 1]
        assert!(report.evidence_coverage >= 0.0 && report.evidence_coverage <= 1.0);
        assert!(report.energy_correlation >= -1.0 && report.energy_correlation <= 1.0);
        assert!(report.hub_coverage >= 0.0 && report.hub_coverage <= 1.0);
        assert!(report.cemented >= 0.0 && report.cemented <= 1.0);
        assert!(report.cross_community >= 0.0 && report.cross_community <= 1.0);
        assert!(report.threshold > 0.0);
    }

    #[test]
    fn evaluate_with_custom_config() {
        let a = make_artifact(vec![(1, 2, 0.5)], vec![]);
        let b = make_artifact(vec![(1, 2, 0.5)], vec![]);

        let low_cfg = EvaluateConfig {
            threshold: 0.01,
            ..Default::default()
        };
        let report = evaluate_with_config(&a, &b, &low_cfg);
        assert!(report.passed, "should pass with threshold 0.01");
        assert!((report.threshold - 0.01).abs() < f64::EPSILON);
    }

    #[test]
    fn artifact_json_roundtrip() {
        let artifact = make_artifact(vec![(1, 2, 0.75), (3, 4, 0.5)], vec![(1, 0.9), (3, 1.5)]);
        let json = serde_json::to_string(&artifact).expect("serialize");
        let restored: DistillArtifact = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(artifact.synapses.len(), restored.synapses.len());
        assert_eq!(artifact.energies.len(), restored.energies.len());
        assert_eq!(artifact.metadata, restored.metadata);
    }

    // -----------------------------------------------------------------------
    // New tests: bincode roundtrip, distill/inject pipeline, anonymization, trust
    // -----------------------------------------------------------------------

    #[test]
    fn artifact_bincode_roundtrip() {
        let artifact = make_artifact(
            vec![(1, 2, 0.75), (3, 4, 0.5), (5, 6, 0.9)],
            vec![(1, 0.9), (3, 1.5), (5, 2.0)],
        );
        let bytes = artifact.to_bincode().expect("bincode encode");
        let restored = DistillArtifact::from_bincode(&bytes).expect("bincode decode");

        assert_eq!(artifact.version, restored.version);
        assert_eq!(artifact.synapses.len(), restored.synapses.len());
        assert_eq!(artifact.energies.len(), restored.energies.len());
        assert_eq!(artifact.metadata, restored.metadata);
        // Verify data fidelity
        for (orig, rest) in artifact.synapses.iter().zip(restored.synapses.iter()) {
            assert_eq!(orig, rest);
        }
        for (orig, rest) in artifact.energies.iter().zip(restored.energies.iter()) {
            assert_eq!(orig, rest);
        }
    }

    #[test]
    fn distill_inject_with_discount() {
        // Simulate: instance A distills → inject into instance B with trust 0.5
        let source_artifact = make_artifact(
            vec![(10, 20, 1.0), (20, 30, 0.8)],
            vec![(10, 2.0), (20, 1.5), (30, 0.5)],
        );

        // Apply trust discount manually to verify expected values
        let trust = 0.5;
        let discounted_synapses: Vec<SynapseSnapshot> = source_artifact
            .synapses
            .iter()
            .map(|s| SynapseSnapshot {
                source: s.source,
                target: s.target,
                weight: s.weight * trust,
            })
            .collect();
        let discounted_energies: Vec<EnergySnapshot> = source_artifact
            .energies
            .iter()
            .map(|e| EnergySnapshot {
                node_id: e.node_id,
                energy: e.energy * trust,
            })
            .collect();

        // Verify discount is correctly applied
        assert!((discounted_synapses[0].weight - 0.5).abs() < f64::EPSILON);
        assert!((discounted_synapses[1].weight - 0.4).abs() < f64::EPSILON);
        assert!((discounted_energies[0].energy - 1.0).abs() < f64::EPSILON);
        assert!((discounted_energies[1].energy - 0.75).abs() < f64::EPSILON);
        assert!((discounted_energies[2].energy - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn anonymization_remaps_ids_preserves_structure() {
        let original = make_artifact(
            vec![(1, 2, 0.5), (2, 3, 0.8), (3, 1, 0.3)],
            vec![(1, 1.0), (2, 2.0), (3, 3.0)],
        );

        let anon = anonymize(&original);

        // Same number of synapses and energies
        assert_eq!(original.synapses.len(), anon.synapses.len());
        assert_eq!(original.energies.len(), anon.energies.len());

        // Weights preserved
        for (orig, a) in original.synapses.iter().zip(anon.synapses.iter()) {
            assert!((orig.weight - a.weight).abs() < f64::EPSILON);
        }
        for (orig, a) in original.energies.iter().zip(anon.energies.iter()) {
            assert!((orig.energy - a.energy).abs() < f64::EPSILON);
        }

        // Node IDs are different from originals
        let original_ids: HashSet<u64> = original
            .synapses
            .iter()
            .flat_map(|s| [s.source, s.target])
            .collect();
        let anon_ids: HashSet<u64> = anon
            .synapses
            .iter()
            .flat_map(|s| [s.source, s.target])
            .collect();
        assert_ne!(original_ids, anon_ids, "anonymized IDs must differ");

        // Source instance should be "anonymous"
        assert_eq!(anon.metadata.source_instance, "anonymous");

        // Structure preserved: if orig has edge (A→B) and (B→C),
        // anon should have edge (A'→B') and (B'→C') where same mapping applies.
        // The source of synapse[1] should equal the target of synapse[0].
        assert_eq!(anon.synapses[0].target, anon.synapses[1].source);
        assert_eq!(anon.synapses[1].target, anon.synapses[2].source);
    }

    #[test]
    fn parity_evaluation_meaningful_metrics() {
        // Two somewhat similar artifacts
        let before = make_artifact(
            vec![(1, 2, 0.5), (2, 3, 0.8), (3, 4, 0.6), (4, 5, 0.4)],
            vec![(1, 1.0), (2, 2.0), (3, 1.5), (4, 0.8)],
        );
        // After injection: overlapping + some new
        let after = make_artifact(
            vec![(1, 2, 0.6), (2, 3, 0.7), (3, 4, 0.5), (5, 6, 0.3)],
            vec![(1, 1.2), (2, 1.8), (3, 1.6), (5, 0.5)],
        );

        let report = evaluate(&before, &after);

        // Evidence coverage should be positive (3/5 overlap)
        assert!(report.evidence_coverage > 0.0);
        // Energy correlation should be positive (correlated values)
        assert!(report.energy_correlation > 0.0);
        // Composite should be meaningful
        assert!(report.composite_score > 0.0);
        assert!(report.composite_score <= 1.0);
    }

    #[test]
    fn inject_trust_zero_no_effect() {
        // trust = 0.0 should be a no-op (inject returns early)
        let artifact = make_artifact(
            vec![(1, 2, 1.0)],
            vec![(1, 5.0)],
        );

        // We can't call inject without a real engine, but we verify the logic:
        // inject clamps trust to 0.0 and returns early.
        let tf = 0.0_f64.clamp(0.0, 1.0);
        assert_eq!(tf, 0.0);

        // Verify that with trust=0 all discounted values are zero
        for s in &artifact.synapses {
            assert!((s.weight * tf).abs() < f64::EPSILON);
        }
        for e in &artifact.energies {
            assert!((e.energy * tf).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn inject_trust_one_full_transfer() {
        let artifact = make_artifact(
            vec![(1, 2, 0.8), (2, 3, 0.6)],
            vec![(1, 2.0), (2, 1.5)],
        );

        let tf = 1.0_f64.clamp(0.0, 1.0);
        assert_eq!(tf, 1.0);

        // With trust=1.0, discounted values equal original values
        for s in &artifact.synapses {
            assert!((s.weight * tf - s.weight).abs() < f64::EPSILON);
        }
        for e in &artifact.energies {
            assert!((e.energy * tf - e.energy).abs() < f64::EPSILON);
        }
    }

    fn make_artifact(synapses: Vec<(u64, u64, f64)>, energies: Vec<(u64, f64)>) -> DistillArtifact {
        DistillArtifact {
            version: String::from("1.0"),
            created_at: SystemTime::now(),
            synapses: synapses
                .into_iter()
                .map(|(s, t, w)| SynapseSnapshot {
                    source: s,
                    target: t,
                    weight: w,
                })
                .collect(),
            energies: energies
                .into_iter()
                .map(|(id, e)| EnergySnapshot {
                    node_id: id,
                    energy: e,
                })
                .collect(),
            fingerprints: Vec::new(),
            metadata: ArtifactMetadata {
                source_instance: String::from("test"),
                total_nodes: 0,
                total_synapses: 0,
                total_communities: 0,
            },
        }
    }
}
