//! Stagnation Detection — identifies dead zones in the graph.
//!
//! Detects communities or subgraphs with no recent activity,
//! enabling proactive maintenance and knowledge refresh.
//!
//! ## Stagnation Score
//!
//! Each community receives a composite score in `[0, 1]`:
//!
//! ```text
//! stagnation = (1 - avg_energy_norm) * 0.4
//!            + last_mutation_age_norm  * 0.3
//!            + (1 - synapse_activity_norm) * 0.3
//! ```
//!
//! Higher values indicate more stagnant communities.

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;

#[cfg(feature = "energy")]
use crate::energy::EnergyStore;
#[cfg(feature = "synapse")]
use crate::synapse::SynapseStore;
use grafeo_common::types::NodeId;

// ---------------------------------------------------------------------------
// Trend
// ---------------------------------------------------------------------------

/// Direction a community's stagnation is heading.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Trend {
    /// Stagnation score decreased by more than the threshold.
    Improving,
    /// Stagnation score increased by more than the threshold.
    Degrading,
    /// Stagnation score is within the threshold of the previous value.
    Stable,
}

// ---------------------------------------------------------------------------
// StagnationScore
// ---------------------------------------------------------------------------

/// Snapshot of stagnation metrics for a single community.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StagnationScore {
    /// Community identifier.
    pub community_id: u64,
    /// Average energy of nodes in the community.
    pub avg_energy: f64,
    /// Time since the most recent mutation in this community.
    pub last_mutation_age: Duration,
    /// Average synapse weight between community nodes.
    pub synapse_activity: f64,
    /// Composite stagnation score in `[0, 1]`.
    pub stagnation_score: f64,
    /// Trend relative to the previous snapshot.
    pub trend: Trend,
}

// ---------------------------------------------------------------------------
// StagnationConfig
// ---------------------------------------------------------------------------

/// Configuration for the stagnation detection subsystem.
#[derive(Debug, Clone)]
pub struct StagnationConfig {
    /// Weight for the energy component in the stagnation formula.
    pub weight_energy: f64,
    /// Weight for the mutation age component.
    pub weight_mutation_age: f64,
    /// Weight for the synapse activity component.
    pub weight_synapse_activity: f64,
    /// Maximum mutation age used for normalization (ages >= this map to 1.0).
    pub max_mutation_age: Duration,
    /// Score above which a community is considered stagnant.
    pub stagnation_threshold: f64,
    /// Duration for "recently reinforced" synapse window.
    pub synapse_recent_window: Duration,
    /// Number of historical snapshots for trend detection.
    pub trend_window_size: usize,
    /// Minimum delta between snapshots to count as Improving/Degrading.
    pub trend_tolerance: f64,
    /// How often communities should be scanned.
    pub scan_interval: Duration,
    /// Reference energy for normalizing avg_energy via `energy_score(e, ref_energy)`.
    /// Uses smooth `1 - exp(-e/ref)` instead of binary clamp.
    pub ref_energy: f64,
    /// Reference synapse weight for normalizing synapse activity via `synapse_score(w, ref_weight)`.
    /// Uses smooth `tanh(w/ref)` instead of binary clamp.
    pub ref_synapse_weight: f64,
}

impl Default for StagnationConfig {
    fn default() -> Self {
        Self {
            weight_energy: 0.4,
            weight_mutation_age: 0.35,
            weight_synapse_activity: 0.25,
            max_mutation_age: Duration::from_secs(30 * 24 * 3600), // 30 days
            stagnation_threshold: 0.7,
            synapse_recent_window: Duration::from_secs(7 * 24 * 3600), // 7 days
            trend_window_size: 5,
            trend_tolerance: 0.05,
            scan_interval: Duration::from_secs(3600),
            ref_energy: 1.0,
            ref_synapse_weight: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// StagnationStore
// ---------------------------------------------------------------------------

/// Thread-safe store holding per-community stagnation scores and history.
pub struct StagnationStore {
    /// Latest score per community.
    scores: DashMap<u64, StagnationScore>,
    /// Previous score value per community (for trend detection).
    previous: DashMap<u64, f64>,
    /// Configuration.
    config: StagnationConfig,
}

impl StagnationStore {
    /// Creates a new, empty stagnation store.
    pub fn new(config: StagnationConfig) -> Self {
        Self {
            scores: DashMap::new(),
            previous: DashMap::new(),
            config,
        }
    }

    /// Stores a new score for a community and computes its trend.
    ///
    /// The trend is determined by comparing the new stagnation score
    /// against the previously stored value.
    pub fn update(&self, community_id: u64, mut score: StagnationScore) {
        // Compute trend from previous snapshot.
        let trend = if let Some(prev) = self.previous.get(&community_id) {
            let delta = score.stagnation_score - *prev;
            if delta < -self.config.trend_tolerance {
                Trend::Improving
            } else if delta > self.config.trend_tolerance {
                Trend::Degrading
            } else {
                Trend::Stable
            }
        } else {
            Trend::Stable
        };
        score.trend = trend;

        // Rotate: current becomes previous.
        self.previous.insert(community_id, score.stagnation_score);
        self.scores.insert(community_id, score);
    }

    /// Returns communities whose stagnation score exceeds `threshold`,
    /// sorted by stagnation score descending.
    pub fn get_stagnant_zones(&self, threshold: f64) -> Vec<StagnationScore> {
        let mut results: Vec<StagnationScore> = self
            .scores
            .iter()
            .filter(|entry| entry.value().stagnation_score > threshold)
            .map(|entry| entry.value().clone())
            .collect();
        results.sort_by(|a, b| {
            b.stagnation_score
                .partial_cmp(&a.stagnation_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }

    /// Returns the latest stagnation score for a community.
    pub fn get_community_vitality(&self, community_id: u64) -> Option<StagnationScore> {
        self.scores.get(&community_id).map(|e| e.value().clone())
    }

    /// Returns the number of tracked communities.
    pub fn len(&self) -> usize {
        self.scores.len()
    }

    /// Returns `true` if no communities are tracked.
    pub fn is_empty(&self) -> bool {
        self.scores.is_empty()
    }

    /// Returns a reference to the configuration.
    pub fn config(&self) -> &StagnationConfig {
        &self.config
    }
}

impl std::fmt::Debug for StagnationStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StagnationStore")
            .field("tracked_communities", &self.scores.len())
            .field("config", &self.config)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// StagnationDetector
// ---------------------------------------------------------------------------

/// Analyzes communities and produces [`StagnationScore`]s.
///
/// The detector holds references to the energy and synapse stores
/// (when their respective features are enabled) and computes a
/// composite stagnation score for each community.
pub struct StagnationDetector {
    /// Energy store reference (feature-gated).
    #[cfg(feature = "energy")]
    energy_store: Arc<EnergyStore>,
    /// Synapse store reference (feature-gated).
    #[cfg(feature = "synapse")]
    synapse_store: Arc<SynapseStore>,
    /// Stagnation store for persisting results.
    store: Arc<StagnationStore>,
    /// Configuration.
    config: StagnationConfig,
}

impl StagnationDetector {
    /// Creates a new detector.
    ///
    /// Supply references to the energy and synapse stores based on active features.
    pub fn new(
        #[cfg(feature = "energy")] energy_store: Arc<EnergyStore>,
        #[cfg(feature = "synapse")] synapse_store: Arc<SynapseStore>,
        store: Arc<StagnationStore>,
        config: StagnationConfig,
    ) -> Self {
        Self {
            #[cfg(feature = "energy")]
            energy_store,
            #[cfg(feature = "synapse")]
            synapse_store,
            store,
            config,
        }
    }

    /// Returns a reference to the backing store.
    pub fn store(&self) -> &Arc<StagnationStore> {
        &self.store
    }

    /// Analyzes a single community and returns a [`StagnationScore`].
    ///
    /// `node_ids` is the set of nodes belonging to the community.
    /// `last_mutation_age` must be supplied by the caller (e.g. from
    /// mutation timestamps in the reactive layer).
    pub fn analyze_community(
        &self,
        community_id: u64,
        node_ids: &[NodeId],
        last_mutation_age: Duration,
    ) -> StagnationScore {
        let avg_energy = self.compute_avg_energy(node_ids);
        let synapse_activity = self.compute_synapse_activity(node_ids);

        // Normalize components to [0, 1] using smooth bounded functions
        // instead of binary clamp (which loses information for values > 1.0).
        let avg_energy_normalized = energy_score_fn(avg_energy, self.config.ref_energy);
        let last_mutation_age_normalized = (last_mutation_age.as_secs_f64()
            / self.config.max_mutation_age.as_secs_f64())
        .clamp(0.0, 1.0);
        let synapse_activity_normalized =
            synapse_score_fn(synapse_activity, self.config.ref_synapse_weight);

        let raw = (1.0 - avg_energy_normalized) * self.config.weight_energy
            + last_mutation_age_normalized * self.config.weight_mutation_age
            + (1.0 - synapse_activity_normalized) * self.config.weight_synapse_activity;
        let stagnation_score = raw.clamp(0.0, 1.0);

        StagnationScore {
            community_id,
            avg_energy,
            last_mutation_age,
            synapse_activity,
            stagnation_score,
            trend: Trend::Stable, // Will be updated by store.update()
        }
    }

    /// Records a score snapshot, updating trend in the store.
    pub fn record_snapshot(&self, community_id: u64, score: StagnationScore) {
        self.store.update(community_id, score);
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Computes average energy across the given nodes.
    ///
    /// Returns `0.0` when the energy feature is disabled or the slice is empty.
    #[cfg(feature = "energy")]
    fn compute_avg_energy(&self, node_ids: &[NodeId]) -> f64 {
        if node_ids.is_empty() {
            return 0.0;
        }
        let total: f64 = node_ids
            .iter()
            .map(|id| self.energy_store.get_energy(*id))
            .sum();
        total / node_ids.len() as f64
    }

    #[cfg(not(feature = "energy"))]
    fn compute_avg_energy(&self, _node_ids: &[NodeId]) -> f64 {
        0.0
    }

    /// Computes average synapse weight between all pairs of community nodes.
    ///
    /// Returns `0.0` when the synapse feature is disabled or the slice is empty.
    #[cfg(feature = "synapse")]
    fn compute_synapse_activity(&self, node_ids: &[NodeId]) -> f64 {
        if node_ids.len() < 2 {
            return 0.0;
        }
        let mut total_weight = 0.0;
        let mut pair_count = 0u64;
        for i in 0..node_ids.len() {
            for j in (i + 1)..node_ids.len() {
                if let Some(syn) = self.synapse_store.get_synapse(node_ids[i], node_ids[j]) {
                    total_weight += syn.current_weight();
                }
                pair_count += 1;
            }
        }
        if pair_count == 0 {
            0.0
        } else {
            total_weight / pair_count as f64
        }
    }

    #[cfg(not(feature = "synapse"))]
    fn compute_synapse_activity(&self, _node_ids: &[NodeId]) -> f64 {
        0.0
    }
}

impl std::fmt::Debug for StagnationDetector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StagnationDetector")
            .field("config", &self.config)
            .finish()
    }
}
