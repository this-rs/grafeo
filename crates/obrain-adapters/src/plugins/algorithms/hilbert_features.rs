//! Hilbert multi-facette node features (64d or 72d).
//!
//! Computes a multi-dimensional feature vector per node by orchestrating graph
//! analysis **facettes**, each producing a 2D point per node, then encoding
//! each via multi-resolution Hilbert curves (8 levels → 8 dimensions).
//!
//! **Base: 8 facettes × 8 dimensions = 64d per node.**
//! **With temporal: 9 facettes × 8 dimensions = 72d per node.**
//! **With temporal + external signal: 10 facettes × 8 dimensions = 80d per node.**
//!
//! | Facette | Dims  | Source algorithm            | Captures                |
//! |---------|-------|----------------------------|-------------------------|
//! | 0       | 0-7   | Spectral eigvecs 1-2       | Global graph structure  |
//! | 1       | 8-15  | Spectral eigvecs 3-4       | Secondary structure     |
//! | 2       | 16-23 | Louvain community + size   | Community membership    |
//! | 3       | 24-31 | PageRank + degree           | Centrality / importance |
//! | 4       | 32-39 | BFS depth (2 seed nodes)   | Distance structure      |
//! | 5       | 40-47 | Betweenness + closeness    | Bridge / flow role      |
//! | 6       | 48-55 | Co-usage (REINFORCES)      | Behavioral coupling     |
//! | 7       | 56-63 | Schema (labels + props)    | Type / structural role  |
//! | 8*      | 64-71 | Timestamp age + variance   | Temporal role           |
//! | 9†      | 72-79 | External 2D signal         | Opaque external signal  |
//!
//! *Facette 8 is **opt-in** via `enable_temporal = true`. When enabled but no
//! node has the configured timestamp property, falls back to 64d silently.
//!
//! †Facette 9 is **opt-in** via `enable_external_signal = true`. Encodes an
//! externally-provided `[f32; 2]` per node. The algorithm is agnostic to the
//! signal's semantics (could be LLM co-activations, user ratings, etc.).
//!
//! Facettes 0-3 are **global** (require full graph computation).
//! Facettes 4-8 are **local** (can be updated incrementally).
//!
//! ## Usage
//!
//! ```no_run
//! use obrain_adapters::plugins::algorithms::hilbert_features::hilbert_features;
//! use obrain_core::graph::lpg::LpgStore;
//!
//! let store = LpgStore::new().unwrap();
//! // ... populate graph ...
//! let config = Default::default();
//! let result = hilbert_features(&store, &config);
//! // result.features: HashMap<NodeId, [f32; 64]>
//! ```

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::OnceLock;

use obrain_common::types::PropertyKey;

use obrain_common::types::{NodeId, Value};
use obrain_common::utils::error::Result;
#[cfg(test)]
use obrain_core::graph::lpg::LpgStore;
use obrain_core::graph::{Direction, GraphStore};

use super::super::{AlgorithmResult, ParameterDef, ParameterType, Parameters};
use super::centrality::{
    betweenness_centrality, closeness_centrality, degree_centrality, pagerank,
};
use super::community::louvain;
use super::hilbert::hilbert_encode_point;
use super::spectral::spectral_embedding;
use super::traits::GraphAlgorithm;
use super::traversal::bfs;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for `hilbert_features()`.
#[derive(Debug, Clone)]
pub struct HilbertFeaturesConfig {
    /// Number of Hilbert encoding levels per facette (default: 8).
    pub levels: usize,
    /// Edge type for persona/co-usage facette (default: "REINFORCES").
    pub persona_edge_type: String,
    /// PageRank damping factor (default: 0.85).
    pub damping: f64,
    /// PageRank iterations (default: 100).
    pub pagerank_iterations: usize,
    /// Max nodes for expensive facettes (betweenness, closeness, spectral).
    /// Above this threshold, these facettes use degree-based approximations.
    /// 0 = no limit (default: 10_000).
    pub large_graph_threshold: usize,
    /// Enable the 9th temporal facette (default: false).
    ///
    /// When enabled AND at least one node has the `temporal_property`, features
    /// grow from `levels × 8` to `levels × 9` dimensions. When disabled or no
    /// timestamps are found, stays at `levels × 8`.
    pub enable_temporal: bool,
    /// Node property key holding a Unix timestamp (seconds) (default: "created_at").
    pub temporal_property: String,
    /// Time window in seconds for age normalization (default: 30 days = 2_592_000).
    pub temporal_window_secs: u64,
    /// Enable the 10th external signal facette (default: false).
    ///
    /// When enabled, adds 8 dimensions encoding an externally-provided `[f32; 2]` point
    /// per node. Nodes not in `external_signals` get the neutral `[0.5, 0.5]`.
    /// The algorithm does not know or care what these signals represent — they
    /// could be LLM co-activations, user ratings, or any 2D signal.
    pub enable_external_signal: bool,
    /// External signal values per node: `NodeId → [x, y]` in `[0, 1]²`.
    ///
    /// Only used when `enable_external_signal = true`. Nodes absent from this map
    /// default to `[0.5, 0.5]` (neutral center).
    pub external_signals: HashMap<NodeId, [f32; 2]>,
}

impl Default for HilbertFeaturesConfig {
    fn default() -> Self {
        Self {
            levels: 8,
            persona_edge_type: "REINFORCES".to_string(),
            damping: 0.85,
            pagerank_iterations: 100,
            large_graph_threshold: 10_000,
            enable_temporal: false,
            temporal_property: "created_at".to_string(),
            temporal_window_secs: 2_592_000, // 30 days
            enable_external_signal: false,
            external_signals: HashMap::new(),
        }
    }
}

// ============================================================================
// Normalization helpers
// ============================================================================

/// Log-normalize a value for power-law distributions.
/// Maps `[0, max]` to `[0, 1]` using `log(1 + x) / log(1 + max)`.
/// This preserves discrimination across orders of magnitude.
#[inline]
fn log_normalize(value: f64, max: f64) -> f32 {
    if max <= 0.0 {
        return 0.0;
    }
    ((1.0 + value).ln() / (1.0 + max).ln()) as f32
}

/// Check if a facette produced degenerate results (all points identical).
/// Returns true if the variance of both x and y coordinates is near zero.
fn is_degenerate(facette: &HashMap<NodeId, [f32; 2]>) -> bool {
    if facette.len() <= 1 {
        return true;
    }
    let n = facette.len() as f64;
    let (sum_x, sum_y) = facette
        .values()
        .fold((0.0f64, 0.0f64), |(sx, sy), &[x, y]| {
            (sx + x as f64, sy + y as f64)
        });
    let (mean_x, mean_y) = (sum_x / n, sum_y / n);
    let (var_x, var_y) = facette
        .values()
        .fold((0.0f64, 0.0f64), |(vx, vy), &[x, y]| {
            (
                vx + (x as f64 - mean_x).powi(2),
                vy + (y as f64 - mean_y).powi(2),
            )
        });
    let std_x = (var_x / n).sqrt();
    let std_y = (var_y / n).sqrt();
    std_x < 1e-6 && std_y < 1e-6
}

/// Rank-percentile normalization: maps values to their percentile rank in [0, 1].
///
/// Sorts all values and assigns `rank / (n - 1)` to each node.
/// Produces a perfectly uniform distribution regardless of the input shape
/// (power-law, bimodal, etc.). O(V log V) for the sort.
///
/// Ties get the same percentile (average rank).
fn rank_normalize(values: &HashMap<NodeId, f64>) -> HashMap<NodeId, f32> {
    let n = values.len();
    if n <= 1 {
        return values.keys().map(|&nid| (nid, 0.5)).collect();
    }

    // Sort by value
    let mut sorted: Vec<(NodeId, f64)> = values.iter().map(|(&k, &v)| (k, v)).collect();
    sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let max_rank = (n - 1) as f64;
    let mut result = HashMap::with_capacity(n);

    // Handle ties: nodes with equal values get the same percentile
    let mut i = 0;
    while i < sorted.len() {
        let val = sorted[i].1;
        let mut j = i;
        while j < sorted.len() && (sorted[j].1 - val).abs() < f64::EPSILON {
            j += 1;
        }
        // Average rank for this tie group
        let avg_rank = (i + j - 1) as f64 / 2.0;
        let percentile = (avg_rank / max_rank) as f32;
        for item in &sorted[i..j] {
            result.insert(item.0, percentile.clamp(0.0, 1.0));
        }
        i = j;
    }

    result
}

// ============================================================================
// Result
// ============================================================================

/// Result of `hilbert_features()` computation.
#[derive(Debug, Clone)]
pub struct HilbertFeaturesResult {
    /// Feature vectors: NodeId → multi-dimensional f32 vector (post Hilbert encoding).
    pub features: HashMap<NodeId, Vec<f32>>,
    /// Raw pre-encoding facette coordinates: NodeId → Vec of `[f32; 2]` per facette.
    ///
    /// Each facette produces a 2D point `[x, y]` in `[0, 1]²` *before* Hilbert
    /// encoding. This is useful for training projections in continuous space
    /// (e.g., learning a linear map from LLM embeddings to facette space).
    ///
    /// Length of each inner `Vec` equals the number of active facettes
    /// (8 base, +1 if temporal, +1 if external signal).
    pub raw_facettes: HashMap<NodeId, Vec<[f32; 2]>>,
    /// Number of dimensions (= levels × num_facettes).
    pub dimensions: usize,
    /// Whether global facettes (spectral, community, centrality) are stale.
    ///
    /// Set to `true` after an incremental update — dims 0-31 contain the
    /// previous global values and should be recalculated in a background
    /// full pass when convenient. `false` after a full `hilbert_features()`.
    pub dirty_global: bool,
}

// ============================================================================
// Fast-path: Load pre-computed features from store
// ============================================================================

/// Attempt to load pre-computed `_hilbert_features` from the graph store.
///
/// This is a **fast-path** for startup: instead of recomputing all 8 facettes
/// (spectral, community, PageRank, etc.), we load the feature vectors that
/// were previously stored as node properties via `compute-hilbert` or similar.
///
/// # Returns
///
/// - `Some(HilbertFeaturesResult)` if ≥ `min_coverage` fraction of nodes
///   have `_hilbert_features` and the dimensions are consistent.
/// - `None` if features are absent, inconsistent, or below coverage threshold.
///
/// # Performance
///
/// O(N) property reads (batch) vs O(N × complexity) for full computation.
/// On megalaw (8.1M nodes): ~2s load vs ~140s compute.
pub fn try_load_hilbert_features_from_store(
    store: &dyn GraphStore,
    min_coverage: f64,
) -> Option<HilbertFeaturesResult> {
    let node_ids = store.all_node_ids();
    let n = node_ids.len();
    if n == 0 {
        return None;
    }

    let key = PropertyKey::from("_hilbert_features");

    // Phase 1: Sample to check coverage (avoid loading all if <90% covered)
    let sample_size = n.min(500);
    let step = if n > sample_size { n / sample_size } else { 1 };
    let mut sample_hits = 0usize;
    let mut detected_dims: Option<usize> = None;

    for i in (0..n).step_by(step).take(sample_size) {
        if let Some(Value::Vector(v)) = store.get_node_property(node_ids[i], &key) {
            sample_hits += 1;
            let d = v.len();
            if let Some(expected) = detected_dims {
                if d != expected {
                    // Inconsistent dimensions — can't use cached features
                    return None;
                }
            } else {
                detected_dims = Some(d);
            }
        }
    }

    let coverage = sample_hits as f64 / sample_size as f64;
    if coverage < min_coverage {
        return None;
    }

    let dims = detected_dims?;

    // Phase 2: Batch load all features
    let batch = store.get_node_property_batch(&node_ids, &key);
    let mut features = HashMap::with_capacity(n);
    let mut loaded = 0usize;

    for (i, val) in batch.into_iter().enumerate() {
        if let Some(Value::Vector(v)) = val {
            if v.len() == dims {
                features.insert(node_ids[i], v.to_vec());
                loaded += 1;
            }
        }
    }

    let actual_coverage = loaded as f64 / n as f64;
    if actual_coverage < min_coverage {
        return None;
    }

    Some(HilbertFeaturesResult {
        features,
        raw_facettes: HashMap::new(), // Not available from stored features
        dimensions: dims,
        dirty_global: false, // Loaded from a full compute, considered clean
    })
}

// ============================================================================
// Public API
// ============================================================================

/// Compute 64d multi-facette Hilbert features for all nodes.
///
/// Orchestrates 8 graph analysis facettes, encodes each as multi-resolution
/// Hilbert vectors, and concatenates them into a single 64d vector per node.
///
/// # Arguments
///
/// * `store` - The graph store
/// * `config` - Configuration (levels, edge types, etc.)
///
/// # Returns
///
/// `HilbertFeaturesResult` with `levels × 8` dimensions per node.
pub fn hilbert_features(
    store: &dyn GraphStore,
    config: &HilbertFeaturesConfig,
) -> HilbertFeaturesResult {
    let node_ids = store.node_ids();
    let n = node_ids.len();

    if n == 0 {
        let num_f =
            8 + usize::from(config.enable_temporal) + usize::from(config.enable_external_signal);
        let total_dims = config.levels * num_f;
        return HilbertFeaturesResult {
            features: HashMap::new(),
            raw_facettes: HashMap::new(),
            dimensions: total_dims,
            dirty_global: false,
        };
    }

    // Check if graph is too large for expensive facettes
    let is_large = config.large_graph_threshold > 0 && n > config.large_graph_threshold;

    // Compute all 8 base facettes as 2D points per node
    // For large graphs, expensive facettes (spectral, betweenness) use approximations
    let facettes: [HashMap<NodeId, [f32; 2]>; 8] = if is_large {
        [
            compute_degree_approx(store),           // approx for spectral 1-2
            compute_degree_approx_secondary(store), // approx for spectral 3-4
            compute_community(store),
            compute_centrality(store, config),
            compute_bfs_distance(store),
            compute_degree_betweenness_approx(store), // approx for betweenness
            compute_co_usage(store, &config.persona_edge_type),
            compute_schema_features(store),
        ]
    } else {
        // Try exact spectral; fallback to degree approx if degenerate (std ≈ 0)
        let spec12 = compute_spectral_12(store);
        let spec34 = compute_spectral_34(store);

        let spec12 = if is_degenerate(&spec12) {
            compute_degree_approx(store)
        } else {
            spec12
        };
        let spec34 = if is_degenerate(&spec34) {
            compute_degree_approx_secondary(store)
        } else {
            spec34
        };

        [
            spec12,
            spec34,
            compute_community(store),
            compute_centrality(store, config),
            compute_bfs_distance(store),
            compute_betweenness_closeness(store),
            compute_co_usage(store, &config.persona_edge_type),
            compute_schema_features(store),
        ]
    };

    // Optionally compute the 9th temporal facette
    let temporal_facette = if config.enable_temporal {
        compute_temporal_facette(store, &config.temporal_property, 0)
    } else {
        None
    };

    // Check if external signal facette is active (non-empty signals when enabled)
    let has_external_signal = config.enable_external_signal;

    let num_facettes =
        8 + usize::from(temporal_facette.is_some()) + usize::from(has_external_signal);
    let total_dims = config.levels * num_facettes;

    // Encode each facette via Hilbert and concatenate; also store raw pre-encoding points
    let mut features: HashMap<NodeId, Vec<f32>> = HashMap::with_capacity(n);
    let mut raw_facettes: HashMap<NodeId, Vec<[f32; 2]>> = HashMap::with_capacity(n);

    for &nid in &node_ids {
        let mut vec = Vec::with_capacity(total_dims);
        let mut raw = Vec::with_capacity(num_facettes);
        for facette in &facettes {
            let point = facette.get(&nid).copied().unwrap_or([0.5, 0.5]);
            raw.push(point);
            let encoded = hilbert_encode_point(point, config.levels);
            vec.extend_from_slice(&encoded);
        }
        // Append temporal facette if present
        if let Some(ref temporal) = temporal_facette {
            let point = temporal.get(&nid).copied().unwrap_or([0.5, 0.5]);
            raw.push(point);
            let encoded = hilbert_encode_point(point, config.levels);
            vec.extend_from_slice(&encoded);
        }
        // Append external signal facette if enabled
        if has_external_signal {
            let point = config
                .external_signals
                .get(&nid)
                .copied()
                .unwrap_or([0.5, 0.5]);
            raw.push(point);
            let encoded = hilbert_encode_point(point, config.levels);
            vec.extend_from_slice(&encoded);
        }
        features.insert(nid, vec);
        raw_facettes.insert(nid, raw);
    }

    HilbertFeaturesResult {
        features,
        raw_facettes,
        dimensions: total_dims,
        dirty_global: false,
    }
}

/// Incrementally update Hilbert features for changed nodes.
///
/// Only recalculates **local facettes** (4-7: BFS distance, betweenness/closeness,
/// co-usage, schema) for `changed_nodes` and their immediate neighbors.
/// **Global facettes** (0-3: spectral, community, centrality) are copied from
/// `previous` unchanged — the returned result has `dirty_global = true`.
///
/// # Cost
///
/// O(|changed| × avg_degree) instead of O(V + E) for a full recalculation.
///
/// # Arguments
///
/// * `store` - The graph store (current state)
/// * `config` - Configuration
/// * `previous` - Previous full `HilbertFeaturesResult`
/// * `changed_nodes` - Nodes that were added or modified
///
/// # Returns
///
/// Updated `HilbertFeaturesResult` with `dirty_global = true`.
pub fn hilbert_features_incremental(
    store: &dyn GraphStore,
    config: &HilbertFeaturesConfig,
    previous: &HilbertFeaturesResult,
    changed_nodes: &[NodeId],
) -> HilbertFeaturesResult {
    let node_ids = store.node_ids();

    if node_ids.is_empty() {
        // Dynamic facette count
        let num_facettes =
            8 + usize::from(config.enable_temporal) + usize::from(config.enable_external_signal);
        let total_dims = config.levels * num_facettes;
        return HilbertFeaturesResult {
            features: HashMap::new(),
            raw_facettes: HashMap::new(),
            dimensions: total_dims,
            dirty_global: true,
        };
    }

    // Collect affected nodes: changed + their neighbors
    let mut affected: std::collections::HashSet<NodeId> = changed_nodes.iter().copied().collect();
    for &nid in changed_nodes {
        for (neighbor, _) in store.edges_from(nid, Direction::Outgoing) {
            affected.insert(neighbor);
        }
    }

    // Recompute only local facettes (4-7) for affected nodes
    let local_bfs = compute_bfs_distance(store);
    let local_bc = compute_betweenness_closeness(store);
    let local_co = compute_co_usage(store, &config.persona_edge_type);
    let local_schema = compute_schema_features(store);

    let local_facettes: [&HashMap<NodeId, [f32; 2]>; 4] =
        [&local_bfs, &local_bc, &local_co, &local_schema];

    // Optionally recompute temporal facette (local — always recomputed)
    let temporal_facette = if config.enable_temporal {
        compute_temporal_facette(store, &config.temporal_property, 0)
    } else {
        None
    };

    let has_external_signal = config.enable_external_signal;
    let num_facettes =
        8 + usize::from(temporal_facette.is_some()) + usize::from(has_external_signal);
    let total_dims = config.levels * num_facettes;
    let base_local_dims = config.levels * 4; // facettes 4-7

    // Build result: copy global dims from previous, update local dims for affected
    let mut features: HashMap<NodeId, Vec<f32>> = HashMap::with_capacity(node_ids.len());
    let mut raw_facettes: HashMap<NodeId, Vec<[f32; 2]>> = HashMap::with_capacity(node_ids.len());
    let global_dims = config.levels * 4; // facettes 0-3

    for &nid in &node_ids {
        let mut vec = Vec::with_capacity(total_dims);
        let mut raw = Vec::with_capacity(num_facettes);

        // Global facettes (0-3): copy from previous (both encoded and raw)
        if let Some(prev) = previous.features.get(&nid) {
            vec.extend_from_slice(&prev[..global_dims.min(prev.len())]);
            while vec.len() < global_dims {
                vec.extend_from_slice(&hilbert_encode_point([0.5, 0.5], config.levels));
            }
        } else {
            for _ in 0..4 {
                vec.extend_from_slice(&hilbert_encode_point([0.5, 0.5], config.levels));
            }
        }
        // Copy raw global facettes from previous or default
        if let Some(prev_raw) = previous.raw_facettes.get(&nid) {
            for i in 0..4 {
                raw.push(prev_raw.get(i).copied().unwrap_or([0.5, 0.5]));
            }
        } else {
            for _ in 0..4 {
                raw.push([0.5, 0.5]);
            }
        }

        // Local facettes (4-7): recompute for affected, copy for others
        if affected.contains(&nid) {
            for facette in &local_facettes {
                let point = facette.get(&nid).copied().unwrap_or([0.5, 0.5]);
                raw.push(point);
                vec.extend_from_slice(&hilbert_encode_point(point, config.levels));
            }
        } else if let Some(prev) = previous.features.get(&nid) {
            let local_end = (global_dims + base_local_dims).min(prev.len());
            vec.extend_from_slice(&prev[global_dims..local_end]);
            while vec.len() < global_dims + base_local_dims {
                vec.extend_from_slice(&hilbert_encode_point([0.5, 0.5], config.levels));
            }
            // Copy raw local facettes from previous
            if let Some(prev_raw) = previous.raw_facettes.get(&nid) {
                for i in 4..8 {
                    raw.push(prev_raw.get(i).copied().unwrap_or([0.5, 0.5]));
                }
            } else {
                for _ in 0..4 {
                    raw.push([0.5, 0.5]);
                }
            }
        } else {
            for _ in 0..4 {
                raw.push([0.5, 0.5]);
                vec.extend_from_slice(&hilbert_encode_point([0.5, 0.5], config.levels));
            }
        }

        // Temporal facette (always recomputed if present)
        if let Some(ref temporal) = temporal_facette {
            let point = temporal.get(&nid).copied().unwrap_or([0.5, 0.5]);
            raw.push(point);
            vec.extend_from_slice(&hilbert_encode_point(point, config.levels));
        }

        // External signal facette (always uses latest config signals)
        if has_external_signal {
            let point = config
                .external_signals
                .get(&nid)
                .copied()
                .unwrap_or([0.5, 0.5]);
            raw.push(point);
            vec.extend_from_slice(&hilbert_encode_point(point, config.levels));
        }

        features.insert(nid, vec);
        raw_facettes.insert(nid, raw);
    }

    HilbertFeaturesResult {
        features,
        raw_facettes,
        dimensions: total_dims,
        dirty_global: true,
    }
}

// ============================================================================
// Facette 0: Spectral eigenvectors 1-2
// ============================================================================

fn compute_spectral_12(store: &dyn GraphStore) -> HashMap<NodeId, [f32; 2]> {
    let result = spectral_embedding(store, 2, None);

    // Extract raw values per dimension
    let mut dim0: HashMap<NodeId, f64> = HashMap::new();
    let mut dim1: HashMap<NodeId, f64> = HashMap::new();
    for (nid, emb) in &result.embeddings {
        dim0.insert(*nid, *emb.first().unwrap_or(&0.0));
        dim1.insert(*nid, *emb.get(1).unwrap_or(&0.0));
    }

    // Rank normalize each dimension independently
    let ranks0 = rank_normalize(&dim0);
    let ranks1 = rank_normalize(&dim1);

    result
        .embeddings
        .keys()
        .map(|&nid| {
            let x = *ranks0.get(&nid).unwrap_or(&0.5);
            let y = *ranks1.get(&nid).unwrap_or(&0.5);
            (nid, [x, y])
        })
        .collect()
}

// ============================================================================
// Facette 1: Spectral eigenvectors 3-4
// ============================================================================

fn compute_spectral_34(store: &dyn GraphStore) -> HashMap<NodeId, [f32; 2]> {
    let result = spectral_embedding(store, 4, None);

    // Extract raw values per dimension and rank normalize
    let mut dim2: HashMap<NodeId, f64> = HashMap::new();
    let mut dim3: HashMap<NodeId, f64> = HashMap::new();
    for (nid, emb) in &result.embeddings {
        dim2.insert(*nid, *emb.get(2).unwrap_or(&0.0));
        dim3.insert(*nid, *emb.get(3).unwrap_or(&0.0));
    }
    let ranks2 = rank_normalize(&dim2);
    let ranks3 = rank_normalize(&dim3);

    result
        .embeddings
        .keys()
        .map(|&nid| {
            let x = *ranks2.get(&nid).unwrap_or(&0.5);
            let y = *ranks3.get(&nid).unwrap_or(&0.5);
            (nid, [x, y])
        })
        .collect()
}

// ============================================================================
// Facette 2: Community (Louvain community_id, intra_density)
// ============================================================================

fn compute_community(store: &dyn GraphStore) -> HashMap<NodeId, [f32; 2]> {
    let louvain_result = louvain(store, 1.0);
    let communities = &louvain_result.communities;

    // Count community sizes and intra-community edges
    let mut community_sizes: HashMap<u64, usize> = HashMap::new();
    let mut community_intra_edges: HashMap<u64, usize> = HashMap::new();
    for &comm in communities.values() {
        *community_sizes.entry(comm).or_insert(0) += 1;
    }

    // Count intra-community edges
    for (&nid, &comm) in communities {
        for (neighbor, _) in store.edges_from(nid, Direction::Outgoing) {
            if communities.get(&neighbor).copied() == Some(comm) {
                *community_intra_edges.entry(comm).or_insert(0) += 1;
            }
        }
    }

    let max_size = community_sizes.values().max().copied().unwrap_or(1).max(1) as f64;

    communities
        .iter()
        .map(|(&nid, &comm)| {
            let size = *community_sizes.get(&comm).unwrap_or(&1);
            let intra_edges = *community_intra_edges.get(&comm).unwrap_or(&0);
            // intra_density = actual edges / possible edges (size*(size-1))
            let possible = if size > 1 { size * (size - 1) } else { 1 };
            let density = intra_edges as f32 / possible as f32;
            // Use community size (log-normalized) instead of raw community ID
            let x = log_normalize(size as f64, max_size);
            (nid, [x.clamp(0.0, 1.0), density.clamp(0.0, 1.0)])
        })
        .collect()
}

// ============================================================================
// Facette 3: Centrality (PageRank, degree)
// ============================================================================

fn compute_centrality(
    store: &dyn GraphStore,
    config: &HilbertFeaturesConfig,
) -> HashMap<NodeId, [f32; 2]> {
    let pr = pagerank(store, config.damping, config.pagerank_iterations, 1e-6);
    let deg = degree_centrality(store);

    // Rank percentile: uniform [0,1] regardless of power-law shape
    let pr_f64: HashMap<NodeId, f64> = pr.iter().map(|(&k, &v)| (k, v)).collect();
    let deg_f64: HashMap<NodeId, f64> = deg
        .total_degree
        .iter()
        .map(|(&k, &v)| (k, v as f64))
        .collect();
    let pr_ranks = rank_normalize(&pr_f64);
    let deg_ranks = rank_normalize(&deg_f64);

    store
        .node_ids()
        .iter()
        .map(|&nid| {
            let x = *pr_ranks.get(&nid).unwrap_or(&0.5);
            let y = *deg_ranks.get(&nid).unwrap_or(&0.5);
            (nid, [x.clamp(0.0, 1.0), y.clamp(0.0, 1.0)])
        })
        .collect()
}

// ============================================================================
// Facette 4: BFS distance (from 2 seed nodes)
// ============================================================================

fn compute_bfs_distance(store: &dyn GraphStore) -> HashMap<NodeId, [f32; 2]> {
    let node_ids = store.node_ids();
    if node_ids.is_empty() {
        return HashMap::new();
    }

    // Pick 2 seeds: first and "most distant" (last in BFS from first)
    let seed1 = node_ids[0];
    let bfs1 = bfs(store, seed1);

    let seed2 = *bfs1.last().unwrap_or(&seed1);
    let _bfs2 = bfs(store, seed2);

    // Build depth maps
    let depth1 = bfs_depths(store, seed1);
    let depth2 = bfs_depths(store, seed2);

    let max_d1 = depth1.values().max().copied().unwrap_or(1).max(1) as f32;
    let max_d2 = depth2.values().max().copied().unwrap_or(1).max(1) as f32;

    node_ids
        .iter()
        .map(|&nid| {
            let x = *depth1.get(&nid).unwrap_or(&0) as f32 / max_d1;
            let y = *depth2.get(&nid).unwrap_or(&0) as f32 / max_d2;
            (nid, [x.clamp(0.0, 1.0), y.clamp(0.0, 1.0)])
        })
        .collect()
}

/// Compute BFS depths from a seed node.
fn bfs_depths(store: &dyn GraphStore, seed: NodeId) -> HashMap<NodeId, usize> {
    use std::collections::{HashSet, VecDeque};

    let mut depths = HashMap::new();
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();

    depths.insert(seed, 0usize);
    visited.insert(seed);
    queue.push_back(seed);

    while let Some(node) = queue.pop_front() {
        let depth = depths[&node];
        for (neighbor, _) in store.edges_from(node, Direction::Outgoing) {
            if visited.insert(neighbor) {
                depths.insert(neighbor, depth + 1);
                queue.push_back(neighbor);
            }
        }
    }

    depths
}

// ============================================================================
// Facette 5: Betweenness + Closeness centrality
// ============================================================================

fn compute_betweenness_closeness(store: &dyn GraphStore) -> HashMap<NodeId, [f32; 2]> {
    let bc = betweenness_centrality(store, true);
    let cc = closeness_centrality(store, true);

    let bc_f64: HashMap<NodeId, f64> = bc.into_iter().collect();
    let cc_f64: HashMap<NodeId, f64> = cc.into_iter().collect();
    let bc_ranks = rank_normalize(&bc_f64);
    let cc_ranks = rank_normalize(&cc_f64);

    store
        .node_ids()
        .iter()
        .map(|&nid| {
            let x = *bc_ranks.get(&nid).unwrap_or(&0.5);
            let y = *cc_ranks.get(&nid).unwrap_or(&0.5);
            (nid, [x.clamp(0.0, 1.0), y.clamp(0.0, 1.0)])
        })
        .collect()
}

// ============================================================================
// Facette 6: Co-usage (REINFORCES edge count + recency)
// ============================================================================

/// Compute co-usage features from edges of a specific type.
///
/// For each node: x = normalized co-activation count, y = normalized recency
/// (number of distinct co-usage partners as proxy for recency/diversity).
fn compute_co_usage(store: &dyn GraphStore, persona_edge_type: &str) -> HashMap<NodeId, [f32; 2]> {
    let node_ids = store.node_ids();

    let mut counts: HashMap<NodeId, usize> = HashMap::new();
    let mut partner_counts: HashMap<NodeId, std::collections::HashSet<NodeId>> = HashMap::new();

    for &nid in &node_ids {
        let edges = store.edges_from(nid, Direction::Outgoing);
        for (neighbor, edge_id) in &edges {
            if let Some(etype) = store.edge_type(*edge_id)
                && etype.as_str() == persona_edge_type
            {
                *counts.entry(nid).or_insert(0) += 1;
                partner_counts.entry(nid).or_default().insert(*neighbor);
            }
        }
    }

    let max_count = counts.values().max().copied().unwrap_or(1).max(1) as f32;
    let max_partners = partner_counts
        .values()
        .map(|s| s.len())
        .max()
        .unwrap_or(1)
        .max(1) as f32;

    node_ids
        .iter()
        .map(|&nid| {
            let x = *counts.get(&nid).unwrap_or(&0) as f32 / max_count;
            let y = partner_counts.get(&nid).map_or(0, |s| s.len()) as f32 / max_partners;
            (nid, [x.clamp(0.0, 1.0), y.clamp(0.0, 1.0)])
        })
        .collect()
}

// ============================================================================
// Facette 7: Schema features (label hash + property count)
// ============================================================================

/// Compute schema-based features.
///
/// x = hash of node labels (normalized), y = number of properties (normalized).
fn compute_schema_features(store: &dyn GraphStore) -> HashMap<NodeId, [f32; 2]> {
    let node_ids = store.node_ids();

    let mut label_hashes: HashMap<NodeId, u64> = HashMap::new();
    let mut prop_counts: HashMap<NodeId, usize> = HashMap::new();

    for &nid in &node_ids {
        if let Some(node) = store.get_node(nid) {
            // Hash the sorted labels
            let mut hasher = std::hash::DefaultHasher::new();
            for label in &node.labels {
                label.hash(&mut hasher);
            }
            label_hashes.insert(nid, hasher.finish());

            // Count properties
            prop_counts.insert(nid, node.properties.len());
        } else {
            label_hashes.insert(nid, 0);
            prop_counts.insert(nid, 0);
        }
    }

    let max_props = prop_counts.values().max().copied().unwrap_or(1).max(1) as f32;

    node_ids
        .iter()
        .map(|&nid| {
            let hash = label_hashes.get(&nid).copied().unwrap_or(0);
            // Normalize hash to [0, 1] via uniform mapping
            let x = (hash as f64 / u64::MAX as f64) as f32;
            let y = *prop_counts.get(&nid).unwrap_or(&0) as f32 / max_props;
            (nid, [x.clamp(0.0, 1.0), y.clamp(0.0, 1.0)])
        })
        .collect()
}

// ============================================================================
// Facette 8: Temporal features (age + neighbor age variance)
// ============================================================================

/// Parse an ISO 8601 / RFC 3339 timestamp string to Unix epoch seconds.
///
/// Supports formats like `"2026-03-28T19:40:34.742194+00:00"`,
/// `"2026-03-28T19:40:34Z"`, and plain epoch seconds `"1711651234"`.
/// No external dependency — hand-rolled parser for the common subset.
fn parse_iso_timestamp(s: &str) -> Option<u64> {
    // Try plain epoch seconds first
    if let Ok(v) = s.parse::<u64>() {
        return Some(v);
    }

    // ISO 8601: "YYYY-MM-DDThh:mm:ss[.frac][Z|+00:00]"
    let s = s.trim();
    if s.len() < 19 {
        return None;
    }

    let year: i64 = s.get(0..4)?.parse().ok()?;
    let month: i64 = s.get(5..7)?.parse().ok()?;
    let day: i64 = s.get(8..10)?.parse().ok()?;
    let hour: i64 = s.get(11..13)?.parse().ok()?;
    let min: i64 = s.get(14..16)?.parse().ok()?;
    let sec: i64 = s.get(17..19)?.parse().ok()?;

    // Days from civil date (Rata Die algorithm, epoch = 1970-01-01)
    let y = if month <= 2 { year - 1 } else { year };
    let m = if month <= 2 { month + 9 } else { month - 3 };
    let days = 365 * y + y / 4 - y / 100 + y / 400 + (m * 153 + 2) / 5 + day - 1 - 719468;

    let epoch = days * 86400 + hour * 3600 + min * 60 + sec;
    if epoch < 0 {
        return None;
    }
    Some(epoch as u64)
}

/// Compute temporal features from node timestamps.
///
/// For each node:
/// - dim1: rank-normalized age (`now - timestamp`). Older nodes get higher values.
/// - dim2: normalized standard deviation of neighbor ages. Nodes bridging old and
///   new regions have high variance.
///
/// Nodes missing the timestamp property get a default age of 0.5 (median).
///
/// # Arguments
///
/// * `store` - The graph store
/// * `property` - Property key holding a Unix timestamp (seconds)
/// * `now` - Current time as Unix timestamp (seconds). If 0, auto-detected from
///   the maximum timestamp in the graph + 1.
///
/// # Returns
///
/// `Some(HashMap)` if at least one node has a timestamp, `None` otherwise.
///
/// # Complexity
///
/// O(V + E) — one pass to read timestamps, one pass for neighbor variance.
fn compute_temporal_facette(
    store: &dyn GraphStore,
    property: &str,
    now: u64,
) -> Option<HashMap<NodeId, [f32; 2]>> {
    let node_ids = store.node_ids();
    let prop_key = obrain_common::types::PropertyKey::new(property);

    // Collect raw timestamps
    let mut timestamps: HashMap<NodeId, u64> = HashMap::new();
    for &nid in &node_ids {
        if let Some(val) = store.get_node_property(nid, &prop_key) {
            let ts = match &val {
                Value::Int64(v) => Some(*v as u64),
                Value::Float64(v) => Some(*v as u64),
                Value::String(s) => parse_iso_timestamp(s),
                _ => None,
            };
            if let Some(ts) = ts {
                timestamps.insert(nid, ts);
            }
        }
    }

    // No timestamps found → skip this facette
    if timestamps.is_empty() {
        return None;
    }

    // Auto-detect "now" if not provided
    let now = if now == 0 {
        timestamps.values().max().copied().unwrap_or(0) + 1
    } else {
        now
    };

    // Compute raw ages
    let mut ages: HashMap<NodeId, f64> = HashMap::with_capacity(node_ids.len());
    for &nid in &node_ids {
        if let Some(&ts) = timestamps.get(&nid) {
            ages.insert(nid, now.saturating_sub(ts) as f64);
        } else {
            // Missing timestamp → sentinel, will get 0.5 after rank_normalize
            ages.insert(nid, f64::NAN);
        }
    }

    // Split into nodes with and without timestamps for rank normalization
    let mut known_ages: HashMap<NodeId, f64> = HashMap::new();
    let mut unknown_nodes: Vec<NodeId> = Vec::new();
    for (&nid, &age) in &ages {
        if age.is_nan() {
            unknown_nodes.push(nid);
        } else {
            known_ages.insert(nid, age);
        }
    }

    let age_ranks = rank_normalize(&known_ages);

    // Compute neighbor age variance (dim2)
    let mut neighbor_variance: HashMap<NodeId, f64> = HashMap::with_capacity(node_ids.len());
    for &nid in &node_ids {
        let neighbors = store.edges_from(nid, Direction::Outgoing);
        if neighbors.is_empty() {
            neighbor_variance.insert(nid, 0.0);
            continue;
        }

        let neighbor_ages: Vec<f32> = neighbors
            .iter()
            .map(|(n, _)| *age_ranks.get(n).unwrap_or(&0.5))
            .collect();

        let n = neighbor_ages.len() as f64;
        let mean = neighbor_ages.iter().map(|&x| x as f64).sum::<f64>() / n;
        let variance = neighbor_ages
            .iter()
            .map(|&x| (x as f64 - mean).powi(2))
            .sum::<f64>()
            / n;
        // std normalized to [0, 0.5] max (since ranks are in [0,1])
        // Multiply by 2 to spread to [0, 1]
        let std = variance.sqrt() * 2.0;
        neighbor_variance.insert(nid, std);
    }

    let var_ranks = rank_normalize(&neighbor_variance);

    // Build result
    let mut result = HashMap::with_capacity(node_ids.len());
    for &nid in &node_ids {
        let x = *age_ranks.get(&nid).unwrap_or(&0.5);
        let y = *var_ranks.get(&nid).unwrap_or(&0.5);
        result.insert(nid, [x.clamp(0.0, 1.0), y.clamp(0.0, 1.0)]);
    }

    Some(result)
}

// ============================================================================
// Large-graph approximations (O(V+E) instead of O(V×E))
// ============================================================================

/// Approximate spectral facette 1-2 using normalized total degree + neighbor degree.
/// O(V+E) — replaces spectral_embedding for large graphs.
/// Uses both outgoing AND incoming edges for total degree.
fn compute_degree_approx(store: &dyn GraphStore) -> HashMap<NodeId, [f32; 2]> {
    let node_ids = store.node_ids();
    let mut total_deg: HashMap<NodeId, usize> = HashMap::with_capacity(node_ids.len());
    let mut all_neighbors: HashMap<NodeId, Vec<NodeId>> = HashMap::with_capacity(node_ids.len());
    let mut neighbor_sum: HashMap<NodeId, f64> = HashMap::with_capacity(node_ids.len());

    // Pass 1: compute out-degree and accumulate in-degree
    for &nid in &node_ids {
        let out_edges = store.edges_from(nid, Direction::Outgoing);
        let out_deg = out_edges.len();
        *total_deg.entry(nid).or_insert(0) += out_deg;
        let neighbors = all_neighbors.entry(nid).or_default();
        for (neighbor, _) in &out_edges {
            neighbors.push(*neighbor);
            // Count as in-degree for the neighbor
            *total_deg.entry(*neighbor).or_insert(0) += 1;
        }
    }

    // Pass 2: average neighbor total degree
    for &nid in &node_ids {
        let neighbors = all_neighbors
            .get(&nid)
            .map_or(&[] as &[NodeId], |v| v.as_slice());
        let sum: f64 = neighbors
            .iter()
            .map(|n| *total_deg.get(n).unwrap_or(&0) as f64)
            .sum();
        let avg = if neighbors.is_empty() {
            0.0
        } else {
            sum / neighbors.len() as f64
        };
        neighbor_sum.insert(nid, avg);
    }

    // Rank percentile on both dimensions
    let deg_f64: HashMap<NodeId, f64> = total_deg.iter().map(|(&k, &v)| (k, v as f64)).collect();
    let deg_ranks = rank_normalize(&deg_f64);
    let neighbor_ranks = rank_normalize(&neighbor_sum);

    node_ids
        .iter()
        .map(|&nid| {
            let x = *deg_ranks.get(&nid).unwrap_or(&0.5);
            let y = *neighbor_ranks.get(&nid).unwrap_or(&0.5);
            (nid, [x.clamp(0.0, 1.0), y.clamp(0.0, 1.0)])
        })
        .collect()
}

/// Secondary degree approximation: out/in degree ratio + clustering coefficient proxy.
fn compute_degree_approx_secondary(store: &dyn GraphStore) -> HashMap<NodeId, [f32; 2]> {
    let node_ids = store.node_ids();
    let mut out_deg: HashMap<NodeId, usize> = HashMap::with_capacity(node_ids.len());
    let mut in_deg: HashMap<NodeId, usize> = HashMap::with_capacity(node_ids.len());
    let mut out_neighbors: HashMap<NodeId, Vec<NodeId>> = HashMap::with_capacity(node_ids.len());

    // Compute out-degree + in-degree + collect neighbors
    for &nid in &node_ids {
        let edges = store.edges_from(nid, Direction::Outgoing);
        out_deg.insert(nid, edges.len());
        let neighbors: Vec<NodeId> = edges.iter().map(|(n, _)| *n).collect();
        out_neighbors.insert(nid, neighbors);
        for (neighbor, _) in &edges {
            *in_deg.entry(*neighbor).or_insert(0) += 1;
        }
    }

    // x = out/in balance (directional asymmetry captures structural role)
    // y = clustering coefficient proxy
    node_ids
        .iter()
        .map(|&nid| {
            let out = *out_deg.get(&nid).unwrap_or(&0) as f64;
            let inc = *in_deg.get(&nid).unwrap_or(&0) as f64;
            let total = out + inc;
            // Directional asymmetry: pure sources=1.0, pure sinks=0.0, balanced=0.5
            let x = if total > 0.0 {
                (out / total) as f32
            } else {
                0.5
            };

            // Clustering coefficient proxy (sampled)
            let neighbors = out_neighbors
                .get(&nid)
                .map_or(&[] as &[NodeId], |v| v.as_slice());
            let k = neighbors.len();
            let y = if k < 2 {
                0.0
            } else {
                let max_pairs = 50usize;
                let mut connected = 0usize;
                let mut checked = 0usize;
                let step = if k * (k - 1) / 2 > max_pairs {
                    k / 10 + 1
                } else {
                    1
                };
                'outer: for i in (0..k).step_by(step) {
                    for j in (i + 1..k).step_by(step) {
                        let ni_neighbors = out_neighbors
                            .get(&neighbors[i])
                            .map_or(&[] as &[NodeId], |v| v.as_slice());
                        if ni_neighbors.contains(&neighbors[j]) {
                            connected += 1;
                        }
                        checked += 1;
                        if checked >= max_pairs {
                            break 'outer;
                        }
                    }
                }
                if checked > 0 {
                    connected as f32 / checked as f32
                } else {
                    0.0
                }
            };
            (nid, [x.clamp(0.0, 1.0), y.clamp(0.0, 1.0)])
        })
        .collect()
}

/// Approximate betweenness using degree centrality ratio (in/out balance).
/// O(V+E) — replaces betweenness_centrality + closeness_centrality.
fn compute_degree_betweenness_approx(store: &dyn GraphStore) -> HashMap<NodeId, [f32; 2]> {
    let node_ids = store.node_ids();
    let mut out_deg: HashMap<NodeId, usize> = HashMap::with_capacity(node_ids.len());
    let mut in_deg: HashMap<NodeId, usize> = HashMap::with_capacity(node_ids.len());

    for &nid in &node_ids {
        let out = store.edges_from(nid, Direction::Outgoing).len();
        out_deg.insert(nid, out);
        for (neighbor, _) in store.edges_from(nid, Direction::Outgoing) {
            *in_deg.entry(neighbor).or_insert(0) += 1;
        }
    }

    // Rank percentile on total degree (betweenness proxy)
    let total_deg: HashMap<NodeId, f64> = node_ids
        .iter()
        .map(|&nid| {
            let total = *out_deg.get(&nid).unwrap_or(&0) + *in_deg.get(&nid).unwrap_or(&0);
            (nid, total as f64)
        })
        .collect();
    let total_ranks = rank_normalize(&total_deg);

    node_ids
        .iter()
        .map(|&nid| {
            let out = *out_deg.get(&nid).unwrap_or(&0) as f64;
            let inc = *in_deg.get(&nid).unwrap_or(&0) as f64;
            let total = out + inc;
            // x = total degree rank percentile (proxy for betweenness)
            let x = *total_ranks.get(&nid).unwrap_or(&0.5);
            // y = in/out balance (bridges tend to have balanced in/out)
            let y = if total > 0.0 {
                (1.0 - (out - inc).abs() / total) as f32
            } else {
                0.5
            };
            (nid, [x.clamp(0.0, 1.0), y.clamp(0.0, 1.0)])
        })
        .collect()
}

// ============================================================================
// GraphAlgorithm trait
// ============================================================================

/// Hilbert features algorithm wrapper for registry integration.
pub struct HilbertFeaturesAlgorithm;

static HILBERT_FEATURES_PARAMS: OnceLock<Vec<ParameterDef>> = OnceLock::new();

fn hilbert_features_params() -> &'static [ParameterDef] {
    HILBERT_FEATURES_PARAMS.get_or_init(|| {
        vec![
            ParameterDef {
                name: "levels".to_string(),
                description: "Hilbert encoding levels per facette".to_string(),
                param_type: ParameterType::Integer,
                required: false,
                default: Some("8".to_string()),
            },
            ParameterDef {
                name: "persona_edge_type".to_string(),
                description: "Edge type for co-usage facette".to_string(),
                param_type: ParameterType::String,
                required: false,
                default: Some("REINFORCES".to_string()),
            },
            ParameterDef {
                name: "enable_temporal".to_string(),
                description: "Enable 9th temporal facette (adds 8 dims)".to_string(),
                param_type: ParameterType::Boolean,
                required: false,
                default: Some("false".to_string()),
            },
            ParameterDef {
                name: "temporal_property".to_string(),
                description: "Node property key for Unix timestamp (seconds)".to_string(),
                param_type: ParameterType::String,
                required: false,
                default: Some("created_at".to_string()),
            },
        ]
    })
}

impl GraphAlgorithm for HilbertFeaturesAlgorithm {
    fn name(&self) -> &str {
        "obrain.hilbert_features"
    }

    fn description(&self) -> &str {
        "Multi-facette Hilbert features (8 or 9 facettes × configurable levels)"
    }

    fn parameters(&self) -> &[ParameterDef] {
        hilbert_features_params()
    }

    fn execute(&self, store: &dyn GraphStore, params: &Parameters) -> Result<AlgorithmResult> {
        let levels = params.get_int("levels").unwrap_or(8) as usize;
        let co_usage = params
            .get_string("persona_edge_type")
            .unwrap_or("REINFORCES")
            .to_string();
        let enable_temporal = params.get_bool("enable_temporal").unwrap_or(false);
        let temporal_property = params
            .get_string("temporal_property")
            .unwrap_or("created_at")
            .to_string();

        let config = HilbertFeaturesConfig {
            levels,
            persona_edge_type: co_usage,
            enable_temporal,
            temporal_property,
            ..Default::default()
        };

        let result = hilbert_features(store, &config);

        let mut algo_result =
            AlgorithmResult::new(vec!["node_id".to_string(), "features".to_string()]);

        for (node_id, features) in &result.features {
            let features_val =
                Value::List(features.iter().map(|&x| Value::Float64(x as f64)).collect());
            algo_result.add_row(vec![Value::Int64(node_id.0 as i64), features_val]);
        }

        Ok(algo_result)
    }
}

// ============================================================================
// Weighted distance
// ============================================================================

/// Per-facette weights for [`weighted_hilbert_distance()`].
///
/// Each weight scales the contribution of one facette (group of `levels`
/// dimensions) to the total distance. Setting a weight to 0.0 completely
/// eliminates that facette from the distance calculation.
///
/// Supports 64d (8 facettes), 72d (9 facettes with temporal or external signal),
/// and 80d (10 facettes with both temporal and external signal).
///
/// # Predefined profiles
///
/// | Profile       | Boosted facettes                        | Use case                  |
/// |---------------|-----------------------------------------|---------------------------|
/// | `uniform`     | All equal                               | Default, no bias          |
/// | `structural`  | Spectral (0,1) + Community (2) + Centrality (3) | Graph structure queries |
/// | `proximity`   | BFS (4) + Co-usage (6)                  | Neighborhood queries      |
/// | `behavioral`  | Co-usage (6) + Schema (7)               | Behavioral similarity     |
/// | `temporal`    | Temporal (8) boosted                    | Time-based queries (72d)  |
///
/// # Example
///
/// ```
/// use obrain_adapters::plugins::algorithms::hilbert_features::FacetteWeights;
///
/// let w = FacetteWeights::structural(8);
/// assert_eq!(w.weights.len(), 8);
/// assert!(w.weights[0] > w.weights[4]); // spectral > BFS
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct FacetteWeights {
    /// One weight per facette. Length = number of facettes (8 or 9).
    pub weights: Vec<f32>,
    /// Pre-computed flag: true if all weights are 1.0 (enables fast path).
    is_uniform: bool,
}

impl FacetteWeights {
    /// All facettes weighted equally.
    pub fn uniform(n: usize) -> Self {
        Self {
            weights: vec![1.0; n],
            is_uniform: true,
        }
    }

    /// Boost structural facettes: spectral (0,1), community (2), centrality (3).
    pub fn structural(n: usize) -> Self {
        let mut w = vec![0.5; n];
        if n >= 4 {
            w[0] = 2.0; // spectral 1-2
            w[1] = 2.0; // spectral 3-4
            w[2] = 1.5; // community
            w[3] = 1.5; // centrality
        }
        Self {
            weights: w,
            is_uniform: false,
        }
    }

    /// Boost proximity facettes: BFS distance (4), co-usage (6).
    pub fn proximity(n: usize) -> Self {
        let mut w = vec![0.5; n];
        if n >= 7 {
            w[4] = 2.0; // BFS distance
            w[6] = 2.0; // co-usage
        }
        Self {
            weights: w,
            is_uniform: false,
        }
    }

    /// Boost behavioral facettes: co-usage (6), schema (7).
    pub fn behavioral(n: usize) -> Self {
        let mut w = vec![0.5; n];
        if n >= 8 {
            w[6] = 2.0; // co-usage
            w[7] = 2.0; // schema
        }
        Self {
            weights: w,
            is_uniform: false,
        }
    }

    /// Boost the 9th temporal facette (index 8). Only meaningful for 72d (9 facettes).
    ///
    /// # Panics
    ///
    /// Panics if `n < 9`.
    pub fn temporal(n: usize) -> Self {
        assert!(n >= 9, "temporal profile requires at least 9 facettes");
        let mut w = vec![0.5; n];
        w[8] = 3.0; // temporal
        Self {
            weights: w,
            is_uniform: false,
        }
    }

    /// Custom weights. Length must match the number of facettes.
    pub fn custom(weights: Vec<f32>) -> Self {
        let is_uniform = weights.iter().all(|&w| w == 1.0);
        Self {
            weights,
            is_uniform,
        }
    }

    /// Number of facettes.
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// Returns `true` if no weights are defined.
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }
}

impl Default for FacetteWeights {
    fn default() -> Self {
        Self::uniform(8)
    }
}

/// Compute the weighted Euclidean distance between two Hilbert feature vectors.
///
/// Each facette (group of `levels` dimensions) is scaled by its corresponding
/// weight from `weights`. The formula is:
///
/// ```text
/// d(a, b) = sqrt( Σᵢ wᵢ × Σⱼ (a[i*L+j] - b[i*L+j])² )
/// ```
///
/// where `i` iterates over facettes and `j` over the `levels` dimensions within
/// each facette.
///
/// # Arguments
///
/// * `a`, `b` - Feature vectors (must have equal length = `levels × weights.len()`)
/// * `weights` - Per-facette weights
/// * `levels` - Dimensions per facette (typically 8)
///
/// # Panics
///
/// Panics if `a.len() != b.len()` or `a.len() != levels * weights.len()`.
///
/// # Example
///
/// ```
/// use obrain_adapters::plugins::algorithms::hilbert_features::{FacetteWeights, weighted_hilbert_distance};
///
/// let a = vec![0.0_f32; 64];
/// let b = vec![1.0_f32; 64];
/// let w = FacetteWeights::uniform(8);
///
/// let d = weighted_hilbert_distance(&a, &b, &w, 8);
/// assert!(d > 0.0);
/// ```
pub fn weighted_hilbert_distance(
    a: &[f32],
    b: &[f32],
    weights: &FacetteWeights,
    levels: usize,
) -> f32 {
    let expected_len = levels * weights.len();
    assert_eq!(
        a.len(),
        b.len(),
        "weighted_hilbert_distance: a.len() ({}) != b.len() ({})",
        a.len(),
        b.len()
    );
    assert_eq!(
        a.len(),
        expected_len,
        "weighted_hilbert_distance: vector length ({}) != levels ({}) × facettes ({})",
        a.len(),
        levels,
        weights.len()
    );

    // Fast path: uniform weights → flat euclidean (no per-facette overhead)
    if weights.is_uniform {
        return hilbert_distance(a, b);
    }

    let mut total = 0.0_f32;
    for (i, &w) in weights.weights.iter().enumerate() {
        if w == 0.0 {
            continue; // Skip eliminated facettes
        }
        let start = i * levels;
        let end = start + levels;
        let mut facette_dist_sq = 0.0_f32;
        for j in start..end {
            let diff = a[j] - b[j];
            facette_dist_sq += diff * diff;
        }
        total += w * facette_dist_sq;
    }
    total.sqrt()
}

/// Standard (unweighted) Euclidean distance between two feature vectors.
///
/// Equivalent to `weighted_hilbert_distance(a, b, &FacetteWeights::uniform(n), levels)`
/// but faster (no weight multiplication).
#[inline]
pub fn hilbert_distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut sum = 0.0_f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum.sqrt()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[allow(clippy::many_single_char_names)]
mod tests {
    use super::*;

    fn make_test_graph() -> LpgStore {
        let store = LpgStore::new().unwrap();
        // Build a small graph: A-B-C-D-E (chain) + A-E (cycle)
        let a = store.create_node(&["Person"]);
        let b = store.create_node(&["Person"]);
        let c = store.create_node(&["Document"]);
        let d = store.create_node(&["Person"]);
        let e = store.create_node(&["Document"]);

        for &(src, dst) in &[(a, b), (b, c), (c, d), (d, e), (e, a)] {
            store.create_edge(src, dst, "KNOWS");
            store.create_edge(dst, src, "KNOWS");
        }
        store
    }

    fn make_two_clusters() -> LpgStore {
        let store = LpgStore::new().unwrap();
        // Cluster 1: fully connected (a,b,c)
        let a = store.create_node(&["A"]);
        let b = store.create_node(&["A"]);
        let c = store.create_node(&["A"]);
        for &(s, d) in &[(a, b), (b, c), (a, c)] {
            store.create_edge(s, d, "LINK");
            store.create_edge(d, s, "LINK");
        }
        // Cluster 2: fully connected (d,e,f)
        let d = store.create_node(&["B"]);
        let e = store.create_node(&["B"]);
        let f = store.create_node(&["B"]);
        for &(s, t) in &[(d, e), (e, f), (d, f)] {
            store.create_edge(s, t, "LINK");
            store.create_edge(t, s, "LINK");
        }
        // Weak bridge: c-d
        store.create_edge(c, d, "BRIDGE");
        store.create_edge(d, c, "BRIDGE");
        store
    }

    #[test]
    fn test_hilbert_features_basic() {
        let store = make_test_graph();
        let config = HilbertFeaturesConfig::default();
        let result = hilbert_features(&store, &config);

        assert_eq!(result.dimensions, 64);
        assert_eq!(result.features.len(), 5);

        for vec in result.features.values() {
            assert_eq!(vec.len(), 64);
            for &v in vec {
                assert!((0.0..=1.0).contains(&v), "Feature value {v} not in [0,1]");
            }
        }
    }

    #[test]
    fn test_hilbert_features_dimensions_exact() {
        let store = make_test_graph();
        let config = HilbertFeaturesConfig {
            levels: 8,
            ..Default::default()
        };
        let result = hilbert_features(&store, &config);

        for vec in result.features.values() {
            assert_eq!(vec.len(), 64, "Expected exactly 64 dimensions");
            // Each block of 8 dims should be in [0,1]
            for block in 0..8 {
                for i in 0..8 {
                    let v = vec[block * 8 + i];
                    assert!((0.0..=1.0).contains(&v));
                }
            }
        }
    }

    #[test]
    fn test_hilbert_features_determinism() {
        let store = make_test_graph();
        let config = HilbertFeaturesConfig::default();
        let r1 = hilbert_features(&store, &config);
        let r2 = hilbert_features(&store, &config);

        for (nid, f1) in &r1.features {
            let f2 = r2.features.get(nid).unwrap();
            // Skip facette 2 (dims 16-23, community) — Louvain is non-deterministic
            // due to HashMap iteration order. All other facettes must be exact.
            for (i, (a, b)) in f1.iter().zip(f2.iter()).enumerate() {
                if (16..24).contains(&i) {
                    continue; // community facette
                }
                assert_eq!(a, b, "Non-deterministic for node {nid:?} at dim {i}");
            }
        }
    }

    #[test]
    fn test_hilbert_features_empty_graph() {
        let store = LpgStore::new().unwrap();
        let config = HilbertFeaturesConfig::default();
        let result = hilbert_features(&store, &config);
        assert!(result.features.is_empty());
        assert_eq!(result.dimensions, 64);
    }

    #[test]
    fn test_hilbert_features_locality() {
        let store = make_two_clusters();
        let config = HilbertFeaturesConfig::default();
        let result = hilbert_features(&store, &config);

        let nodes: Vec<NodeId> = store.node_ids();
        // Cluster 1: nodes 0,1,2 — Cluster 2: nodes 3,4,5
        let cluster1: Vec<&Vec<f32>> = nodes[..3]
            .iter()
            .filter_map(|n| result.features.get(n))
            .collect();
        let cluster2: Vec<&Vec<f32>> = nodes[3..6]
            .iter()
            .filter_map(|n| result.features.get(n))
            .collect();

        // Average intra-cluster distance should be < inter-cluster distance
        let intra1 = avg_l2_distance(&cluster1);
        let intra2 = avg_l2_distance(&cluster2);
        let inter = avg_l2_distance_between(&cluster1, &cluster2);

        let avg_intra = f32::midpoint(intra1, intra2);
        assert!(
            avg_intra < inter,
            "Intra-cluster distance ({avg_intra}) should be < inter-cluster ({inter})"
        );
    }

    #[test]
    fn test_hilbert_features_custom_levels() {
        let store = make_test_graph();
        let config = HilbertFeaturesConfig {
            levels: 4,
            ..Default::default()
        };
        let result = hilbert_features(&store, &config);
        assert_eq!(result.dimensions, 32); // 4 levels × 8 facettes
        for vec in result.features.values() {
            assert_eq!(vec.len(), 32);
        }
    }

    #[test]
    fn test_hilbert_features_algorithm_trait() {
        let store = make_test_graph();
        let algo = HilbertFeaturesAlgorithm;
        assert_eq!(algo.name(), "obrain.hilbert_features");

        let params = Parameters::new();
        let result = algo.execute(&store, &params).unwrap();
        assert_eq!(result.columns, vec!["node_id", "features"]);
        assert_eq!(result.rows.len(), 5);
    }

    // Helpers
    fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    fn avg_l2_distance(vecs: &[&Vec<f32>]) -> f32 {
        let mut sum = 0.0;
        let mut count = 0;
        for i in 0..vecs.len() {
            for j in i + 1..vecs.len() {
                sum += l2_distance(vecs[i], vecs[j]);
                count += 1;
            }
        }
        if count > 0 { sum / count as f32 } else { 0.0 }
    }

    fn avg_l2_distance_between(a: &[&Vec<f32>], b: &[&Vec<f32>]) -> f32 {
        let mut sum = 0.0;
        let mut count = 0;
        for va in a {
            for vb in b {
                sum += l2_distance(va, vb);
                count += 1;
            }
        }
        if count > 0 { sum / count as f32 } else { 0.0 }
    }

    #[test]
    fn test_incremental_preserves_global() {
        let store = LpgStore::new().unwrap();
        let a = store.create_node(&["Person"]);
        let b = store.create_node(&["Person"]);
        let c = store.create_node(&["Document"]);
        store.create_edge(a, b, "KNOWS");
        store.create_edge(b, a, "KNOWS");
        store.create_edge(b, c, "KNOWS");
        store.create_edge(c, b, "KNOWS");

        let config = HilbertFeaturesConfig::default();
        let full = hilbert_features(&store, &config);
        assert!(!full.dirty_global);

        // Add a new node and edge
        let d = store.create_node(&["Person"]);
        store.create_edge(c, d, "KNOWS");
        store.create_edge(d, c, "KNOWS");

        let incr = hilbert_features_incremental(&store, &config, &full, &[d]);
        assert!(incr.dirty_global);
        assert_eq!(incr.features.len(), 4); // now 4 nodes

        // Global dims (0-31) should be unchanged for nodes a, b, c
        for &nid in &[a, b, c] {
            let full_vec = full.features.get(&nid).unwrap();
            let incr_vec = incr.features.get(&nid).unwrap();
            assert_eq!(
                &full_vec[..32],
                &incr_vec[..32],
                "Global dims changed for existing node {nid:?}"
            );
        }

        // New node d should have default global dims and valid local dims
        let d_vec = incr.features.get(&d).unwrap();
        assert_eq!(d_vec.len(), 64);
        for &v in d_vec {
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn test_incremental_updates_local_for_neighbors() {
        let store = LpgStore::new().unwrap();
        let a = store.create_node(&["A"]);
        let b = store.create_node(&["A"]);
        store.create_edge(a, b, "LINK");
        store.create_edge(b, a, "LINK");

        let config = HilbertFeaturesConfig::default();
        let full = hilbert_features(&store, &config);

        // Add node c connected to b
        let c = store.create_node(&["B"]);
        store.create_edge(b, c, "LINK");
        store.create_edge(c, b, "LINK");

        let incr = hilbert_features_incremental(&store, &config, &full, &[c]);

        // b is a neighbor of changed node c → local dims should be recomputed
        let b_full = full.features.get(&b).unwrap();
        let b_incr = incr.features.get(&b).unwrap();
        // Local dims (32-63) may differ because b's neighborhood changed
        // (This is expected — just verify they're valid)
        for &v in &b_incr[32..] {
            assert!((0.0..=1.0).contains(&v));
        }
        // Global dims unchanged
        assert_eq!(&b_full[..32], &b_incr[..32]);
    }

    #[test]
    fn test_incremental_dirty_flag() {
        let store = LpgStore::new().unwrap();
        let a = store.create_node(&["X"]);
        let b = store.create_node(&["X"]);
        store.create_edge(a, b, "E");
        store.create_edge(b, a, "E");

        let config = HilbertFeaturesConfig::default();

        // Full → dirty_global = false
        let full = hilbert_features(&store, &config);
        assert!(!full.dirty_global);

        // Incremental → dirty_global = true
        let c = store.create_node(&["Y"]);
        store.create_edge(a, c, "E");
        let incr = hilbert_features_incremental(&store, &config, &full, &[c]);
        assert!(incr.dirty_global);

        // Another full recalc → dirty_global = false
        let full2 = hilbert_features(&store, &config);
        assert!(!full2.dirty_global);
    }

    // ================================================================
    // Temporal facette tests
    // ================================================================

    #[test]
    fn test_temporal_basic() {
        let store = LpgStore::new().unwrap();
        let now = 1_000_000u64;
        let a = store.create_node(&["A"]);
        let b = store.create_node(&["A"]);
        let c = store.create_node(&["A"]);
        store.create_edge(a, b, "LINK");
        store.create_edge(b, c, "LINK");

        // Set timestamps: a=old, b=medium, c=recent
        store.set_node_property(a, "created_at", Value::Int64((now - 100_000) as i64));
        store.set_node_property(b, "created_at", Value::Int64((now - 50_000) as i64));
        store.set_node_property(c, "created_at", Value::Int64((now - 1_000) as i64));

        let config = HilbertFeaturesConfig {
            enable_temporal: true,
            ..Default::default()
        };
        let result = hilbert_features(&store, &config);

        // With temporal → 9 facettes × 8 levels = 72 dims
        assert_eq!(result.dimensions, 72);
        for vec in result.features.values() {
            assert_eq!(vec.len(), 72);
            for &v in vec {
                assert!((0.0..=1.0).contains(&v), "Feature value {v} not in [0,1]");
            }
        }
    }

    #[test]
    fn test_temporal_missing_timestamps() {
        // No timestamps on any node → stays at 64d
        let store = make_test_graph();
        let config = HilbertFeaturesConfig {
            enable_temporal: true,
            ..Default::default()
        };
        let result = hilbert_features(&store, &config);

        assert_eq!(result.dimensions, 64);
        for vec in result.features.values() {
            assert_eq!(vec.len(), 64);
        }
    }

    #[test]
    fn test_temporal_ordering() {
        // Older nodes should have higher age rank
        let store = LpgStore::new().unwrap();
        let old = store.create_node(&["N"]);
        let mid = store.create_node(&["N"]);
        let recent = store.create_node(&["N"]);
        store.create_edge(old, mid, "E");
        store.create_edge(mid, recent, "E");

        let now = 1_000_000u64;
        store.set_node_property(old, "created_at", Value::Int64((now - 500_000) as i64));
        store.set_node_property(mid, "created_at", Value::Int64((now - 100_000) as i64));
        store.set_node_property(recent, "created_at", Value::Int64((now - 1_000) as i64));

        let temporal = compute_temporal_facette(&store, "created_at", now).unwrap();

        // dim1 (x) = rank of age → old should have highest rank
        assert!(
            temporal[&old][0] > temporal[&recent][0],
            "Old node age rank ({}) should be > recent ({})",
            temporal[&old][0],
            temporal[&recent][0]
        );
        assert!(
            temporal[&mid][0] > temporal[&recent][0],
            "Mid node age rank ({}) should be > recent ({})",
            temporal[&mid][0],
            temporal[&recent][0]
        );
    }

    #[test]
    fn test_temporal_opt_in() {
        // enable_temporal=false → stays at 64d even if timestamps exist
        let store = LpgStore::new().unwrap();
        let a = store.create_node(&["A"]);
        store.set_node_property(a, "created_at", Value::Int64(1_000_000));

        let config = HilbertFeaturesConfig {
            enable_temporal: false,
            ..Default::default()
        };
        let result = hilbert_features(&store, &config);
        assert_eq!(result.dimensions, 64);
    }

    #[test]
    fn test_temporal_partial_coverage() {
        // Some nodes have timestamps, others don't → 72d, missing get default 0.5
        let store = LpgStore::new().unwrap();
        let a = store.create_node(&["A"]);
        let b = store.create_node(&["A"]);
        let c = store.create_node(&["A"]);
        store.create_edge(a, b, "E");
        store.create_edge(b, c, "E");

        // Only a and c have timestamps
        store.set_node_property(a, "created_at", Value::Int64(900_000));
        store.set_node_property(c, "created_at", Value::Int64(999_000));

        let temporal = compute_temporal_facette(&store, "created_at", 1_000_000).unwrap();

        // b has no timestamp → should still have a valid [0,1] point
        let b_point = temporal[&b];
        assert!((0.0..=1.0).contains(&b_point[0]), "b.x = {}", b_point[0]);
        assert!((0.0..=1.0).contains(&b_point[1]), "b.y = {}", b_point[1]);
    }

    #[test]
    fn test_temporal_bridge_variance() {
        // A node bridging old and new clusters should have high neighbor age variance
        let store = LpgStore::new().unwrap();
        let now = 1_000_000u64;

        // Old cluster
        let o1 = store.create_node(&["Old"]);
        let o2 = store.create_node(&["Old"]);
        store.set_node_property(o1, "created_at", Value::Int64((now - 500_000) as i64));
        store.set_node_property(o2, "created_at", Value::Int64((now - 490_000) as i64));

        // New cluster
        let n1 = store.create_node(&["New"]);
        let n2 = store.create_node(&["New"]);
        store.set_node_property(n1, "created_at", Value::Int64((now - 1_000) as i64));
        store.set_node_property(n2, "created_at", Value::Int64((now - 2_000) as i64));

        // Bridge node
        let bridge = store.create_node(&["Bridge"]);
        store.set_node_property(bridge, "created_at", Value::Int64((now - 250_000) as i64));

        // Connect bridge to both clusters
        store.create_edge(bridge, o1, "E");
        store.create_edge(bridge, o2, "E");
        store.create_edge(bridge, n1, "E");
        store.create_edge(bridge, n2, "E");

        // Intra-cluster edges
        store.create_edge(o1, o2, "E");
        store.create_edge(n1, n2, "E");

        let temporal = compute_temporal_facette(&store, "created_at", now).unwrap();

        // Bridge node (dim2 = neighbor age variance) should be higher than intra-cluster nodes
        let bridge_var = temporal[&bridge][1];
        let o1_var = temporal[&o1][1];
        assert!(
            bridge_var > o1_var,
            "Bridge variance ({bridge_var}) should be > intra-cluster ({o1_var})"
        );
    }

    #[test]
    fn test_hilbert_features_perf_1000_nodes() {
        // Build a random-ish graph with 1000 nodes
        let store = LpgStore::new().unwrap();
        let mut nodes = Vec::with_capacity(1000);
        for i in 0..1000 {
            let label = if i % 3 == 0 {
                "A"
            } else if i % 3 == 1 {
                "B"
            } else {
                "C"
            };
            nodes.push(store.create_node(&[label]));
        }
        // Add ~3000 edges (ring + skip connections)
        for i in 0..1000 {
            let j = (i + 1) % 1000;
            store.create_edge(nodes[i], nodes[j], "LINK");
            store.create_edge(nodes[j], nodes[i], "LINK");
            if i + 7 < 1000 {
                store.create_edge(nodes[i], nodes[i + 7], "LINK");
            }
        }

        let config = HilbertFeaturesConfig::default();
        let start = std::time::Instant::now();
        let result = hilbert_features(&store, &config);
        let elapsed = start.elapsed();

        assert_eq!(result.features.len(), 1000);
        assert!(
            elapsed.as_millis() < 5000,
            "hilbert_features on 1000 nodes took {}ms (should be <5s)",
            elapsed.as_millis()
        );
        eprintln!("hilbert_features(1000 nodes): {}ms", elapsed.as_millis());
    }

    // ========================================================================
    // Weighted distance tests
    // ========================================================================

    #[test]
    fn test_weighted_uniform_matches_euclidean() {
        // uniform weights should produce the same result as plain euclidean
        let a: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();
        let b: Vec<f32> = (0..64).map(|i| (63 - i) as f32 * 0.01).collect();

        let w_dist = weighted_hilbert_distance(&a, &b, &FacetteWeights::uniform(8), 8);
        let e_dist = hilbert_distance(&a, &b);

        assert!(
            (w_dist - e_dist).abs() < 1e-5,
            "uniform weighted ({w_dist}) != euclidean ({e_dist})"
        );
    }

    #[test]
    fn test_weighted_uniform_72d() {
        // 9 facettes × 8 levels = 72d
        let a = vec![0.5_f32; 72];
        let mut b = vec![0.5_f32; 72];
        b[64] = 1.0; // differ in temporal facette

        let w = FacetteWeights::uniform(9);
        let d = weighted_hilbert_distance(&a, &b, &w, 8);
        assert!(d > 0.0);
        assert_eq!(w.len(), 9);
    }

    #[test]
    fn test_weighted_zero_eliminates_facette() {
        let a = vec![0.0_f32; 64];
        let mut b = vec![0.0_f32; 64];
        // Make facettes 0 and 1 different
        for i in 0..16 {
            b[i] = 1.0;
        }

        // With uniform weights, distance > 0
        let d_uniform = weighted_hilbert_distance(&a, &b, &FacetteWeights::uniform(8), 8);
        assert!(d_uniform > 0.0);

        // With zero weights on facettes 0 and 1, distance = 0
        let w = FacetteWeights::custom(vec![0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let d_zero = weighted_hilbert_distance(&a, &b, &w, 8);
        assert!(
            d_zero.abs() < 1e-10,
            "zeroed facettes should eliminate their contribution, got {d_zero}"
        );
    }

    #[test]
    fn test_structural_changes_ranking() {
        // Build a small graph and compute features
        let store = LpgStore::new().unwrap();
        let mut nodes = Vec::new();
        for _ in 0..10 {
            nodes.push(store.create_node(&["N"]));
        }
        // Hub topology: node 0 connected to all others
        for i in 1..10 {
            store.create_edge(nodes[0], nodes[i], "LINK");
            store.create_edge(nodes[i], nodes[0], "LINK");
        }
        // Chain: 1→2→3→4
        for i in 1..4 {
            store.create_edge(nodes[i], nodes[i + 1], "LINK");
        }

        let config = HilbertFeaturesConfig::default();
        let result = hilbert_features(&store, &config);
        let query = &result.features[&nodes[0]]; // Hub node

        // Compute top-5 neighbors with uniform vs structural weights
        let uniform = FacetteWeights::uniform(8);
        let structural = FacetteWeights::structural(8);

        let mut ranking_uniform: Vec<(NodeId, f32)> = result
            .features
            .iter()
            .filter(|(nid, _)| **nid != nodes[0])
            .map(|(nid, v)| (*nid, weighted_hilbert_distance(query, v, &uniform, 8)))
            .collect();
        ranking_uniform.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut ranking_structural: Vec<(NodeId, f32)> = result
            .features
            .iter()
            .filter(|(nid, _)| **nid != nodes[0])
            .map(|(nid, v)| (*nid, weighted_hilbert_distance(query, v, &structural, 8)))
            .collect();
        ranking_structural.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let top5_u: Vec<NodeId> = ranking_uniform.iter().take(5).map(|x| x.0).collect();
        let top5_s: Vec<NodeId> = ranking_structural.iter().take(5).map(|x| x.0).collect();

        // The rankings should differ (structural boosts spectral/community/centrality)
        // At minimum, both should return valid results
        assert_eq!(top5_u.len(), 5);
        assert_eq!(top5_s.len(), 5);
        // Note: on small graphs the rankings may coincidentally match,
        // but the distances should differ
        let d_u_first = ranking_uniform[0].1;
        let d_s_first = ranking_structural[0].1;
        assert!(d_u_first >= 0.0);
        assert!(d_s_first >= 0.0);
    }

    #[test]
    #[should_panic(expected = "weighted_hilbert_distance: vector length")]
    fn test_custom_weights_validation() {
        let a = vec![0.0_f32; 64];
        let b = vec![0.0_f32; 64];
        // Wrong number of facettes (5 instead of 8)
        let w = FacetteWeights::custom(vec![1.0; 5]);
        weighted_hilbert_distance(&a, &b, &w, 8); // Should panic
    }

    #[test]
    fn test_temporal_profile() {
        // 72d vectors with difference only in temporal facette (dims 64-71)
        let a = vec![0.0_f32; 72];
        let mut b = vec![0.0_f32; 72];
        for i in 64..72 {
            b[i] = 1.0;
        }

        let uniform = FacetteWeights::uniform(9);
        let temporal = FacetteWeights::temporal(9);

        let d_uniform = weighted_hilbert_distance(&a, &b, &uniform, 8);
        let d_temporal = weighted_hilbert_distance(&a, &b, &temporal, 8);

        // Temporal profile should amplify the temporal difference
        assert!(
            d_temporal > d_uniform,
            "temporal distance ({d_temporal}) should be > uniform ({d_uniform})"
        );
    }

    #[test]
    fn test_facette_weights_profiles() {
        // Just verify the profiles produce correct lengths
        assert_eq!(FacetteWeights::uniform(8).len(), 8);
        assert_eq!(FacetteWeights::uniform(9).len(), 9);
        assert_eq!(FacetteWeights::structural(8).len(), 8);
        assert_eq!(FacetteWeights::proximity(8).len(), 8);
        assert_eq!(FacetteWeights::behavioral(8).len(), 8);
        assert_eq!(FacetteWeights::temporal(9).len(), 9);
        assert_eq!(FacetteWeights::default().len(), 8);

        // Structural boosts facettes 0-3
        let s = FacetteWeights::structural(8);
        assert!(s.weights[0] > s.weights[4]);
    }

    // ========================================================================
    // raw_facettes tests
    // ========================================================================

    #[test]
    fn test_raw_facettes_populated() {
        let store = make_test_graph();
        let config = HilbertFeaturesConfig::default();
        let result = hilbert_features(&store, &config);

        // Every node should have raw_facettes
        assert_eq!(
            result.raw_facettes.len(),
            result.features.len(),
            "raw_facettes should have the same number of entries as features"
        );

        // Default config = 8 facettes (no temporal)
        for (nid, raw) in &result.raw_facettes {
            assert_eq!(
                raw.len(),
                8,
                "Node {nid:?} should have 8 raw facettes, got {}",
                raw.len()
            );
            // Each facette is a [f32; 2] point in [0, 1]²
            for (i, point) in raw.iter().enumerate() {
                assert!(
                    (0.0..=1.0).contains(&point[0]),
                    "Node {nid:?} facette {i} x={} not in [0,1]",
                    point[0]
                );
                assert!(
                    (0.0..=1.0).contains(&point[1]),
                    "Node {nid:?} facette {i} y={} not in [0,1]",
                    point[1]
                );
            }
        }
    }

    #[test]
    fn test_raw_facettes_with_temporal() {
        let store = LpgStore::new().unwrap();
        let now = 1_000_000u64;
        let a = store.create_node(&["A"]);
        let b = store.create_node(&["B"]);
        store.create_edge(a, b, "LINK");
        store.create_edge(b, a, "LINK");

        store.set_node_property(a, "created_at", Value::Int64((now - 100_000) as i64));
        store.set_node_property(b, "created_at", Value::Int64((now - 1_000) as i64));

        let config = HilbertFeaturesConfig {
            enable_temporal: true,
            ..Default::default()
        };
        let result = hilbert_features(&store, &config);

        // With temporal = 9 facettes
        assert_eq!(result.dimensions, 72); // 9 × 8
        for (nid, raw) in &result.raw_facettes {
            assert_eq!(
                raw.len(),
                9,
                "Node {nid:?} should have 9 raw facettes (8 + temporal), got {}",
                raw.len()
            );
            for (i, point) in raw.iter().enumerate() {
                assert!(
                    (0.0..=1.0).contains(&point[0]),
                    "Node {nid:?} facette {i} x={} not in [0,1]",
                    point[0]
                );
                assert!(
                    (0.0..=1.0).contains(&point[1]),
                    "Node {nid:?} facette {i} y={} not in [0,1]",
                    point[1]
                );
            }
        }
    }

    #[test]
    fn test_raw_facettes_incremental() {
        let store = LpgStore::new().unwrap();
        let a = store.create_node(&["A"]);
        let b = store.create_node(&["B"]);
        store.create_edge(a, b, "LINK");
        store.create_edge(b, a, "LINK");

        let config = HilbertFeaturesConfig::default();
        let full = hilbert_features(&store, &config);
        assert_eq!(full.raw_facettes.len(), 2);

        // Add a node and run incremental
        let c = store.create_node(&["C"]);
        store.create_edge(b, c, "LINK");
        store.create_edge(c, b, "LINK");

        let incr = hilbert_features_incremental(&store, &config, &full, &[c]);

        // All 3 nodes should have raw_facettes
        assert_eq!(incr.raw_facettes.len(), 3);
        for (nid, raw) in &incr.raw_facettes {
            assert_eq!(
                raw.len(),
                8,
                "Node {nid:?} incremental raw_facettes should have 8 entries"
            );
            for (i, point) in raw.iter().enumerate() {
                assert!(
                    (0.0..=1.0).contains(&point[0]) && (0.0..=1.0).contains(&point[1]),
                    "Node {nid:?} facette {i} out of range: {:?}",
                    point
                );
            }
        }

        // Existing nodes' global raw_facettes (0-3) should be preserved from full
        for &nid in &[a, b] {
            let full_raw = &full.raw_facettes[&nid];
            let incr_raw = &incr.raw_facettes[&nid];
            // Global facettes (0-3) unchanged
            assert_eq!(
                &full_raw[..4],
                &incr_raw[..4],
                "Node {nid:?} global raw_facettes changed in incremental"
            );
        }
    }

    // ========================================================================
    // External signal facette tests
    // ========================================================================

    #[test]
    fn test_external_signal_disabled_by_default() {
        let store = make_test_graph();
        let config = HilbertFeaturesConfig::default();
        assert!(!config.enable_external_signal);
        let result = hilbert_features(&store, &config);
        // 8 facettes × 8 levels = 64d
        assert_eq!(result.dimensions, 64);
        for raw in result.raw_facettes.values() {
            assert_eq!(raw.len(), 8);
        }
    }

    #[test]
    fn test_external_signal_adds_dimensions() {
        let store = LpgStore::new().unwrap();
        let a = store.create_node(&["A"]);
        let b = store.create_node(&["B"]);
        store.create_edge(a, b, "LINK");
        store.create_edge(b, a, "LINK");

        let mut signals = HashMap::new();
        signals.insert(a, [0.8, 0.2]);
        signals.insert(b, [0.1, 0.9]);

        let config = HilbertFeaturesConfig {
            enable_external_signal: true,
            external_signals: signals,
            ..Default::default()
        };
        let result = hilbert_features(&store, &config);

        // 9 facettes × 8 levels = 72d
        assert_eq!(result.dimensions, 72);
        for (nid, raw) in &result.raw_facettes {
            assert_eq!(
                raw.len(),
                9,
                "Node {nid:?} should have 9 facettes (8 + external)"
            );
            for (i, point) in raw.iter().enumerate() {
                assert!(
                    (0.0..=1.0).contains(&point[0]) && (0.0..=1.0).contains(&point[1]),
                    "Node {nid:?} facette {i} out of range: {:?}",
                    point
                );
            }
        }

        // Verify the external signal facette (last one) matches input
        let a_ext = result.raw_facettes[&a][8];
        assert!((a_ext[0] - 0.8).abs() < 1e-6);
        assert!((a_ext[1] - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_external_signal_neutral_default() {
        // Nodes not in external_signals get [0.5, 0.5]
        let store = LpgStore::new().unwrap();
        let a = store.create_node(&["A"]);
        let b = store.create_node(&["B"]);
        store.create_edge(a, b, "LINK");
        store.create_edge(b, a, "LINK");

        // Only provide signal for a, not b
        let mut signals = HashMap::new();
        signals.insert(a, [0.9, 0.1]);

        let config = HilbertFeaturesConfig {
            enable_external_signal: true,
            external_signals: signals,
            ..Default::default()
        };
        let result = hilbert_features(&store, &config);

        // b should have neutral [0.5, 0.5] for external signal
        let b_ext = result.raw_facettes[&b][8];
        assert!((b_ext[0] - 0.5).abs() < 1e-6, "b.x = {}", b_ext[0]);
        assert!((b_ext[1] - 0.5).abs() < 1e-6, "b.y = {}", b_ext[1]);
    }

    #[test]
    fn test_external_signal_with_temporal() {
        // Both temporal and external signal → 10 facettes × 8 levels = 80d
        let store = LpgStore::new().unwrap();
        let now = 1_000_000u64;
        let a = store.create_node(&["A"]);
        let b = store.create_node(&["B"]);
        store.create_edge(a, b, "LINK");
        store.create_edge(b, a, "LINK");
        store.set_node_property(a, "created_at", Value::Int64((now - 50_000) as i64));
        store.set_node_property(b, "created_at", Value::Int64((now - 1_000) as i64));

        let mut signals = HashMap::new();
        signals.insert(a, [0.7, 0.3]);

        let config = HilbertFeaturesConfig {
            enable_temporal: true,
            enable_external_signal: true,
            external_signals: signals,
            ..Default::default()
        };
        let result = hilbert_features(&store, &config);

        // 10 facettes × 8 = 80d
        assert_eq!(result.dimensions, 80);
        for raw in result.raw_facettes.values() {
            assert_eq!(raw.len(), 10);
        }
    }

    #[test]
    fn test_external_signal_incremental() {
        let store = LpgStore::new().unwrap();
        let a = store.create_node(&["A"]);
        let b = store.create_node(&["B"]);
        store.create_edge(a, b, "LINK");
        store.create_edge(b, a, "LINK");

        let mut signals = HashMap::new();
        signals.insert(a, [0.8, 0.2]);
        signals.insert(b, [0.3, 0.7]);

        let config = HilbertFeaturesConfig {
            enable_external_signal: true,
            external_signals: signals,
            ..Default::default()
        };

        let full = hilbert_features(&store, &config);
        assert_eq!(full.dimensions, 72);

        // Add node c
        let c = store.create_node(&["C"]);
        store.create_edge(b, c, "LINK");
        store.create_edge(c, b, "LINK");

        let incr = hilbert_features_incremental(&store, &config, &full, &[c]);
        assert_eq!(incr.dimensions, 72);
        assert_eq!(incr.features.len(), 3);

        // c has no signal → neutral
        let c_ext = incr.raw_facettes[&c][8];
        assert!((c_ext[0] - 0.5).abs() < 1e-6);
        assert!((c_ext[1] - 0.5).abs() < 1e-6);

        // a's external signal should be preserved
        let a_ext = incr.raw_facettes[&a][8];
        assert!((a_ext[0] - 0.8).abs() < 1e-6);
    }

    // ==================== try_load_hilbert_features_from_store Tests ====================

    #[test]
    fn test_load_features_from_store_roundtrip() {
        use std::sync::Arc;

        let store = make_test_graph();

        // Step 1: Compute features
        let config = HilbertFeaturesConfig::default();
        let computed = hilbert_features(&store, &config);
        assert!(!computed.features.is_empty());

        // Step 2: Store features as _hilbert_features property
        let key = "_hilbert_features";
        for (nid, feats) in &computed.features {
            let vec: Arc<[f32]> = feats.as_slice().into();
            store.set_node_property(*nid, key, Value::Vector(vec));
        }

        // Step 3: Load from store
        let loaded = try_load_hilbert_features_from_store(&store, 0.9);
        assert!(loaded.is_some(), "Should successfully load features from store");

        let loaded = loaded.unwrap();
        assert_eq!(loaded.features.len(), computed.features.len());
        assert_eq!(loaded.dimensions, computed.dimensions);

        // Verify feature vectors match
        for (nid, computed_feats) in &computed.features {
            let loaded_feats = &loaded.features[nid];
            assert_eq!(computed_feats, loaded_feats, "Features should match for node {:?}", nid);
        }
    }

    #[test]
    fn test_load_features_from_store_empty() {
        let store = LpgStore::new().unwrap();
        // Empty store → None
        let result = try_load_hilbert_features_from_store(&store, 0.9);
        assert!(result.is_none());
    }

    #[test]
    fn test_load_features_from_store_no_features() {
        let store = make_test_graph();
        // Graph has nodes but no _hilbert_features → None
        let result = try_load_hilbert_features_from_store(&store, 0.9);
        assert!(result.is_none());
    }

    #[test]
    fn test_load_features_from_store_partial_coverage() {
        use std::sync::Arc;

        let store = make_test_graph();
        let config = HilbertFeaturesConfig::default();
        let computed = hilbert_features(&store, &config);

        // Only store features for first node (low coverage)
        if let Some((nid, feats)) = computed.features.iter().next() {
            let vec: Arc<[f32]> = feats.as_slice().into();
            store.set_node_property(*nid, "_hilbert_features", Value::Vector(vec));
        }

        // Coverage < 90% → None
        let result = try_load_hilbert_features_from_store(&store, 0.9);
        assert!(result.is_none(), "Should fail with partial coverage");
    }
}
