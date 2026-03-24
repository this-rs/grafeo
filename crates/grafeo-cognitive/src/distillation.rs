//! Peer-to-peer knowledge distillation between Grafeo instances.
//!
//! This module enables extracting cognitive state from one engine,
//! serializing it as a [`DistillArtifact`], and injecting it into
//! another engine with a configurable trust discount.
//!
//! The [`evaluate`] function compares two artifacts to measure
//! knowledge parity via Jaccard overlap, Pearson correlation, and
//! structural fingerprint similarity.

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
}

impl Default for DistillConfig {
    fn default() -> Self {
        Self {
            min_synapse_weight: 0.1,
            min_energy: 0.01,
            include_fingerprints: false,
            include_episodes: false,
            community_filter: None,
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

/// A snapshot of a single synapse for serialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynapseSnapshot {
    /// Source node ID.
    pub source: u64,
    /// Target node ID.
    pub target: u64,
    /// Current weight (after decay).
    pub weight: f64,
}

/// A snapshot of a single node's energy for serialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
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

/// Alias for [`FingerprintSnapshot`] for backward compatibility.
pub type CommunityFingerprint = FingerprintSnapshot;

/// Metadata about the source engine that produced the artifact.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
// ParityReport
// ---------------------------------------------------------------------------

/// Result of comparing two [`DistillArtifact`]s for knowledge parity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParityReport {
    /// Jaccard index of synapse (source, target) pairs.
    pub synapse_overlap: f64,
    /// Pearson correlation of shared node energies.
    pub energy_correlation: f64,
    /// Average fingerprint similarity (0.0 if no fingerprints).
    pub structural_similarity: f64,
    /// Weighted composite score.
    pub composite_score: f64,
}

/// Configuration for the [`evaluate`] function.
#[derive(Debug, Clone)]
pub struct EvaluateConfig {
    /// Composite score threshold for passing (default: 0.67).
    pub threshold: f64,
    /// Weights for synapse overlap, energy correlation, and structural similarity.
    pub weights: [f64; 3],
}

impl Default for EvaluateConfig {
    fn default() -> Self {
        Self {
            threshold: 0.67,
            weights: [0.4, 0.4, 0.2],
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

    DistillArtifact {
        version: String::from("1.0"),
        created_at: SystemTime::now(),
        synapses,
        energies,
        fingerprints: Vec::new(),
        metadata,
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
    inject_synapses(artifact, trust_factor, engine);
    inject_energies(artifact, trust_factor, engine);
}

#[cfg(feature = "synapse")]
fn inject_synapses(artifact: &DistillArtifact, trust_factor: f64, engine: &dyn CognitiveEngine) {
    let Some(store) = engine.synapse_store() else {
        return;
    };
    for snap in &artifact.synapses {
        store.reinforce(
            NodeId(snap.source),
            NodeId(snap.target),
            snap.weight * trust_factor,
        );
    }
}

#[cfg(not(feature = "synapse"))]
fn inject_synapses(
    _artifact: &DistillArtifact,
    _trust_factor: f64,
    _engine: &dyn CognitiveEngine,
) {
}

#[cfg(feature = "energy")]
fn inject_energies(artifact: &DistillArtifact, trust_factor: f64, engine: &dyn CognitiveEngine) {
    let Some(store) = engine.energy_store() else {
        return;
    };
    for snap in &artifact.energies {
        store.boost(NodeId(snap.node_id), snap.energy * trust_factor);
    }
}

#[cfg(not(feature = "energy"))]
fn inject_energies(
    _artifact: &DistillArtifact,
    _trust_factor: f64,
    _engine: &dyn CognitiveEngine,
) {
}

// ---------------------------------------------------------------------------
// evaluate
// ---------------------------------------------------------------------------

/// Compares two artifacts to measure knowledge parity.
///
/// Returns a [`ParityReport`] with Jaccard overlap of synapse pairs,
/// Pearson correlation of shared node energies, structural fingerprint
/// similarity, and a weighted composite score.
pub fn evaluate(before: &DistillArtifact, after: &DistillArtifact) -> ParityReport {
    let synapse_overlap = synapse_jaccard(before, after);
    let energy_correlation = energy_pearson(before, after);
    let structural_similarity = fingerprint_similarity(before, after);

    // Weighted composite: 40% synapse overlap, 40% energy correlation, 20% structural
    let composite_score =
        0.4 * synapse_overlap + 0.4 * energy_correlation + 0.2 * structural_similarity;

    ParityReport {
        synapse_overlap,
        energy_correlation,
        structural_similarity,
        composite_score,
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

    // Collect pairs of (energy_a, energy_b) for shared nodes
    let pairs: Vec<(f64, f64)> = map_a
        .iter()
        .filter_map(|(&nid, &ea)| map_b.get(&nid).map(|&eb| (ea, eb)))
        .collect();

    if pairs.is_empty() {
        return 0.0;
    }
    if pairs.len() == 1 {
        // Single point: if values are equal, correlation is 1.0
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
        // All values are identical in one or both sets
        if var_a < f64::EPSILON && var_b < f64::EPSILON {
            1.0
        } else {
            0.0
        }
    } else {
        cov / denom
    }
}

/// Average similarity of fingerprints matched by community ID.
fn fingerprint_similarity(a: &DistillArtifact, b: &DistillArtifact) -> f64 {
    if a.fingerprints.is_empty() || b.fingerprints.is_empty() {
        return 0.0;
    }
    let map_b: HashMap<u64, &FingerprintSnapshot> = b
        .fingerprints
        .iter()
        .map(|f| (f.community_id, f))
        .collect();

    let mut total_sim = 0.0;
    let mut matched = 0usize;

    for fa in &a.fingerprints {
        if let Some(fb) = map_b.get(&fa.community_id) {
            // Compare node_count, edge_count, clustering_coeff
            let nc_sim = 1.0 - ratio_diff(fa.node_count as f64, fb.node_count as f64);
            let ec_sim = 1.0 - ratio_diff(fa.edge_count as f64, fb.edge_count as f64);
            let cc_sim = 1.0 - (fa.clustering_coeff - fb.clustering_coeff).abs();
            total_sim += (nc_sim + ec_sim + cc_sim) / 3.0;
            matched += 1;
        }
    }

    if matched == 0 {
        0.0
    } else {
        total_sim / matched as f64
    }
}

/// Ratio-based difference: |a - b| / max(a, b). Returns 0.0 if both are 0.
fn ratio_diff(a: f64, b: f64) -> f64 {
    let max = a.max(b);
    if max < f64::EPSILON {
        0.0
    } else {
        (a - b).abs() / max
    }
}

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
        assert!((report.synapse_overlap - 1.0).abs() < f64::EPSILON);
        assert!((report.energy_correlation - 1.0).abs() < 1e-9);
        assert!(report.composite_score > 0.7);
    }

    fn make_artifact(
        synapses: Vec<(u64, u64, f64)>,
        energies: Vec<(u64, f64)>,
    ) -> DistillArtifact {
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
