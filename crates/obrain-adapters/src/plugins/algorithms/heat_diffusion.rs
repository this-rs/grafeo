//! Heat kernel diffusion on Hilbert feature space.
//!
//! Propagates heat from seed nodes through the graph using Gaussian kernel
//! weights derived from weighted Hilbert distances. This enables
//! similarity-aware signal propagation that respects the multi-facette
//! structure of the feature space.
//!
//! ## Algorithm
//!
//! ```text
//! heat(v, t+1) = α × heat(v, t) + (1-α) × Σ_u∈N(v) w(u,v) × heat(u, t) / Σ_u∈N(v) w(u,v)
//! ```
//!
//! Where `w(u,v) = exp(-d² / σ²)` with `d = weighted_hilbert_distance(u, v, weights, levels)`.

use std::collections::HashMap;
use std::sync::OnceLock;

use obrain_common::types::{NodeId, Value};
use obrain_common::utils::error::Result;
use obrain_core::graph::{Direction, GraphStore};

use super::super::{AlgorithmResult, ParameterDef, ParameterType, Parameters};
use super::hilbert_features::{FacetteWeights, HilbertFeaturesResult, weighted_hilbert_distance};
use super::traits::GraphAlgorithm;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for heat kernel diffusion.
#[derive(Debug, Clone)]
pub struct HeatKernelConfig {
    /// Number of diffusion iterations (default: 4).
    pub iterations: usize,
    /// Retention factor: how much of a node's own heat is kept each step (default: 0.4).
    pub alpha: f32,
    /// Gaussian kernel bandwidth. `None` triggers auto-calibration from edge distances.
    pub sigma_sq: Option<f32>,
}

impl Default for HeatKernelConfig {
    fn default() -> Self {
        Self {
            iterations: 4,
            alpha: 0.4,
            sigma_sq: None,
        }
    }
}

// ============================================================================
// Auto-calibration
// ============================================================================

/// Auto-calibrate σ² by sampling edge distances.
///
/// Samples up to 1000 edges, computes the squared weighted Hilbert distance
/// for each, and returns the median as σ². Falls back to 1.0 if no edges
/// are found or all distances are zero.
pub fn calibrate_sigma(
    store: &dyn GraphStore,
    features: &HilbertFeaturesResult,
    weights: &FacetteWeights,
    levels: usize,
) -> f32 {
    let node_ids = store.node_ids();
    let mut dist_sq_samples: Vec<f32> = Vec::with_capacity(1000);

    'outer: for &nid in &node_ids {
        if let Some(feat_a) = features.features.get(&nid) {
            for (neighbor, _) in store.edges_from(nid, Direction::Outgoing) {
                if let Some(feat_b) = features.features.get(&neighbor) {
                    let d = weighted_hilbert_distance(feat_a, feat_b, weights, levels);
                    dist_sq_samples.push(d * d);
                    if dist_sq_samples.len() >= 1000 {
                        break 'outer;
                    }
                }
            }
        }
    }

    if dist_sq_samples.is_empty() {
        return 1.0;
    }

    // Median of squared distances
    dist_sq_samples.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = dist_sq_samples[dist_sq_samples.len() / 2];

    if median <= 0.0 || !median.is_finite() {
        1.0
    } else {
        median
    }
}

// ============================================================================
// Core diffusion
// ============================================================================

/// Run heat kernel diffusion from seed nodes.
///
/// Seeds start with heat 1.0; all other nodes start at 0.0. At each iteration,
/// heat is mixed with neighbors using Gaussian kernel weights derived from
/// weighted Hilbert distance.
///
/// Returns nodes sorted by heat descending. Nodes with zero heat are excluded.
pub fn heat_kernel_diffusion(
    store: &dyn GraphStore,
    features: &HilbertFeaturesResult,
    seeds: &[NodeId],
    weights: &FacetteWeights,
    config: &HeatKernelConfig,
) -> Vec<(NodeId, f32)> {
    if seeds.is_empty() {
        return Vec::new();
    }

    let node_ids = store.node_ids();
    if node_ids.is_empty() {
        return Vec::new();
    }

    // Determine σ²
    let levels = if weights.is_empty() {
        0
    } else {
        let sample_len = features.features.values().next().map_or(0, |v| v.len());
        if !weights.is_empty() {
            sample_len / weights.len()
        } else {
            0
        }
    };

    let sigma_sq = config
        .sigma_sq
        .unwrap_or_else(|| calibrate_sigma(store, features, weights, levels));

    // Guard: if sigma_sq is zero or non-finite, use 1.0
    let sigma_sq = if sigma_sq <= 0.0 || !sigma_sq.is_finite() {
        1.0
    } else {
        sigma_sq
    };

    // Initialize heat map
    let mut heat: HashMap<NodeId, f32> = HashMap::with_capacity(node_ids.len());
    for &nid in &node_ids {
        heat.insert(nid, 0.0);
    }
    for &seed in seeds {
        if heat.contains_key(&seed) {
            heat.insert(seed, 1.0);
        }
    }

    // Pre-compute neighbor lists + kernel weights
    let mut neighbor_weights: HashMap<NodeId, Vec<(NodeId, f32)>> =
        HashMap::with_capacity(node_ids.len());
    for &nid in &node_ids {
        let edges = store.edges_from(nid, Direction::Outgoing);
        let mut nw: Vec<(NodeId, f32)> = Vec::with_capacity(edges.len());
        if let Some(feat_v) = features.features.get(&nid) {
            for (neighbor, _) in &edges {
                if let Some(feat_u) = features.features.get(neighbor) {
                    let d = weighted_hilbert_distance(feat_v, feat_u, weights, levels);
                    let d_sq = d * d;
                    let w = (-d_sq / sigma_sq).exp();
                    // w is guaranteed in (0, 1] due to exp of non-positive value
                    if w.is_finite() {
                        nw.push((*neighbor, w));
                    }
                }
            }
        }
        neighbor_weights.insert(nid, nw);
    }

    let alpha = config.alpha.clamp(0.0, 1.0);

    // Iterative diffusion
    for _ in 0..config.iterations {
        let mut new_heat: HashMap<NodeId, f32> = HashMap::with_capacity(node_ids.len());

        for &nid in &node_ids {
            let self_heat = *heat.get(&nid).unwrap_or(&0.0);

            let Some(neighbors) = neighbor_weights.get(&nid) else {
                new_heat.insert(nid, alpha * self_heat);
                continue;
            };

            if neighbors.is_empty() {
                new_heat.insert(nid, alpha * self_heat);
                continue;
            }

            let mut weighted_sum = 0.0f32;
            let mut weight_total = 0.0f32;
            for &(neighbor, w) in neighbors {
                let neighbor_heat = *heat.get(&neighbor).unwrap_or(&0.0);
                weighted_sum += w * neighbor_heat;
                weight_total += w;
            }

            let diffused = if weight_total > 0.0 {
                weighted_sum / weight_total
            } else {
                0.0
            };

            let h = alpha * self_heat + (1.0 - alpha) * diffused;
            // Clamp to avoid any floating point drift
            let h = if h.is_finite() { h } else { 0.0 };
            new_heat.insert(nid, h);
        }

        heat = new_heat;
    }

    // Collect non-zero, sort descending
    let mut result: Vec<(NodeId, f32)> = heat
        .into_iter()
        .filter(|(_, h)| *h > 0.0 && h.is_finite())
        .collect();
    result.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    result
}

// ============================================================================
// GraphAlgorithm trait implementation
// ============================================================================

/// Heat kernel diffusion algorithm, registered as a `GraphAlgorithm`.
pub struct HeatKernelAlgorithm;

static HEAT_KERNEL_PARAMS: OnceLock<Vec<ParameterDef>> = OnceLock::new();

fn heat_kernel_params() -> &'static [ParameterDef] {
    HEAT_KERNEL_PARAMS.get_or_init(|| {
        vec![
            ParameterDef {
                name: "seed_nodes".to_string(),
                description: "Comma-separated node IDs to seed with heat".to_string(),
                param_type: ParameterType::String,
                required: true,
                default: None,
            },
            ParameterDef {
                name: "iterations".to_string(),
                description: "Number of diffusion iterations".to_string(),
                param_type: ParameterType::Integer,
                required: false,
                default: Some("4".to_string()),
            },
            ParameterDef {
                name: "alpha".to_string(),
                description: "Retention factor (0.0-1.0)".to_string(),
                param_type: ParameterType::Float,
                required: false,
                default: Some("0.4".to_string()),
            },
            ParameterDef {
                name: "sigma_sq".to_string(),
                description: "Gaussian kernel bandwidth (omit for auto-calibration)".to_string(),
                param_type: ParameterType::Float,
                required: false,
                default: None,
            },
        ]
    })
}

impl GraphAlgorithm for HeatKernelAlgorithm {
    fn name(&self) -> &str {
        "obrain.heat_kernel_diffusion"
    }

    fn description(&self) -> &str {
        "Heat kernel diffusion on Hilbert feature space"
    }

    fn parameters(&self) -> &[ParameterDef] {
        heat_kernel_params()
    }

    fn execute(&self, store: &dyn GraphStore, params: &Parameters) -> Result<AlgorithmResult> {
        use super::hilbert_features::{HilbertFeaturesConfig, hilbert_features};

        // Parse seed nodes from comma-separated string
        let seed_str = params.get_string("seed_nodes").unwrap_or("");
        let seeds: Vec<NodeId> = seed_str
            .split(',')
            .filter_map(|s| {
                let s = s.trim();
                if s.is_empty() {
                    None
                } else {
                    s.parse::<u64>().ok().map(NodeId)
                }
            })
            .collect();

        let iterations = params.get_int("iterations").unwrap_or(4) as usize;
        let alpha = params.get_float("alpha").unwrap_or(0.4) as f32;
        let sigma_sq = params.get_float("sigma_sq").map(|v| v as f32);

        let config = HeatKernelConfig {
            iterations,
            alpha,
            sigma_sq,
        };

        // Compute Hilbert features
        let feat_config = HilbertFeaturesConfig::default();
        let features = hilbert_features(store, &feat_config);
        let weights = FacetteWeights::uniform(8);

        let result = heat_kernel_diffusion(store, &features, &seeds, &weights, &config);

        let mut algo_result = AlgorithmResult::new(vec!["node_id".to_string(), "heat".to_string()]);
        for (node_id, heat) in &result {
            algo_result.add_row(vec![
                Value::Int64(node_id.0 as i64),
                Value::Float64(*heat as f64),
            ]);
        }
        Ok(algo_result)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[allow(clippy::many_single_char_names)]
mod tests {
    use super::super::hilbert_features::{HilbertFeaturesConfig, hilbert_features};
    use super::*;
    use obrain_core::graph::lpg::LpgStore;

    /// Build a 5-node line graph: A - B - C - D - E (bidirectional edges).
    fn make_line_graph() -> (LpgStore, Vec<NodeId>) {
        let store = LpgStore::new().unwrap();
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);
        let d = store.create_node(&["Node"]);
        let e = store.create_node(&["Node"]);

        for &(src, dst) in &[(a, b), (b, c), (c, d), (d, e)] {
            store.create_edge(src, dst, "LINK");
            store.create_edge(dst, src, "LINK");
        }
        (store, vec![a, b, c, d, e])
    }

    fn compute_features(store: &LpgStore) -> (HilbertFeaturesResult, FacetteWeights) {
        let config = HilbertFeaturesConfig::default();
        let features = hilbert_features(store, &config);
        let n_facettes = features
            .features
            .values()
            .next()
            .map_or(8, |v| v.len() / config.levels);
        let weights = FacetteWeights::uniform(n_facettes);
        (features, weights)
    }

    #[test]
    fn test_heat_kernel_basic() {
        let (store, nodes) = make_line_graph();
        let (features, weights) = compute_features(&store);

        let config = HeatKernelConfig {
            iterations: 4,
            alpha: 0.4,
            sigma_sq: None,
        };

        // Seed node A only
        let result = heat_kernel_diffusion(&store, &features, &[nodes[0]], &weights, &config);

        // Result should be non-empty
        assert!(!result.is_empty(), "diffusion should produce results");

        // The seed node should have the highest heat
        let heat_map: HashMap<NodeId, f32> = result.iter().copied().collect();
        let seed_heat = heat_map.get(&nodes[0]).copied().unwrap_or(0.0);
        assert!(seed_heat > 0.0, "seed should have positive heat");

        // Heat should generally decrease with graph distance from seed
        // (nodes[0] is seed, nodes[4] is furthest)
        let heat_a = heat_map.get(&nodes[0]).copied().unwrap_or(0.0);
        let heat_e = heat_map.get(&nodes[4]).copied().unwrap_or(0.0);
        assert!(
            heat_a >= heat_e,
            "heat at seed ({}) should be >= heat at far end ({})",
            heat_a,
            heat_e
        );

        // No NaN or Inf
        for (_, h) in &result {
            assert!(h.is_finite(), "heat values must be finite");
        }
    }

    #[test]
    fn test_heat_kernel_convergence() {
        let (store, nodes) = make_line_graph();
        let (features, weights) = compute_features(&store);

        let config_8 = HeatKernelConfig {
            iterations: 8,
            alpha: 0.4,
            sigma_sq: None,
        };
        let config_10 = HeatKernelConfig {
            iterations: 10,
            alpha: 0.4,
            sigma_sq: None,
        };

        let result_8 = heat_kernel_diffusion(&store, &features, &[nodes[0]], &weights, &config_8);
        let result_10 = heat_kernel_diffusion(&store, &features, &[nodes[0]], &weights, &config_10);

        let heat_8: HashMap<NodeId, f32> = result_8.into_iter().collect();
        let heat_10: HashMap<NodeId, f32> = result_10.into_iter().collect();

        // After many iterations, heat should be relatively stable between iter 8 and 10
        for &nid in &nodes {
            let h8 = heat_8.get(&nid).copied().unwrap_or(0.0);
            let h10 = heat_10.get(&nid).copied().unwrap_or(0.0);
            let diff = (h8 - h10).abs();
            assert!(
                diff < 0.15,
                "heat should stabilize: node {:?} had diff {} (h8={}, h10={})",
                nid,
                diff,
                h8,
                h10
            );
        }
    }

    #[test]
    fn test_heat_kernel_zero_distance() {
        // Create nodes that will have identical features (isolated nodes with same label)
        let store = LpgStore::new().unwrap();
        let a = store.create_node(&["Same"]);
        let b = store.create_node(&["Same"]);
        // Connect them so they are neighbors
        store.create_edge(a, b, "LINK");
        store.create_edge(b, a, "LINK");

        let (features, weights) = compute_features(&store);

        let config = HeatKernelConfig {
            iterations: 4,
            alpha: 0.4,
            sigma_sq: None,
        };

        let result = heat_kernel_diffusion(&store, &features, &[a], &weights, &config);

        // No NaN or Inf — this is the primary assertion
        for (_, h) in &result {
            assert!(h.is_finite(), "heat must be finite even with zero distance");
            assert!(!h.is_nan(), "heat must not be NaN");
        }
    }

    #[test]
    fn test_heat_kernel_no_seeds() {
        let (store, _nodes) = make_line_graph();
        let (features, weights) = compute_features(&store);

        let config = HeatKernelConfig::default();
        let result = heat_kernel_diffusion(&store, &features, &[], &weights, &config);

        assert!(result.is_empty(), "no seeds should produce empty result");
    }

    #[test]
    fn test_calibrate_sigma() {
        let (store, _nodes) = make_line_graph();
        let (features, weights) = compute_features(&store);

        let levels = features
            .features
            .values()
            .next()
            .map_or(8, |v| v.len() / weights.len());

        let sigma = calibrate_sigma(&store, &features, &weights, levels);
        assert!(
            sigma > 0.0,
            "calibrated sigma² must be positive, got {}",
            sigma
        );
        assert!(sigma.is_finite(), "calibrated sigma² must be finite");
    }
}
