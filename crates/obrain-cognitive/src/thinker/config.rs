//! # Thinker config wiring (T13 Step 7)
//!
//! Serde-friendly configuration for the thinker fleet. Parsed from
//! `HubConfig.substrate.thinkers` or an equivalent environment-driven
//! source, then converted to the richer internal `*Config` structs the
//! thinkers consume.
//!
//! ## TOML shape
//!
//! ```toml
//! [substrate.thinkers]
//! max_total_cpu_fraction = 1.0
//! shutdown_timeout_secs = 5
//! min_tick_interval_secs = 1
//!
//! [substrate.thinkers.consolidator]
//! enabled = true
//! interval_secs = 60
//! cpu_fraction = 0.15
//! decay_factor = 0.99
//! synapse_prune_threshold_u16 = 64
//! engram_reinforce_delta_u16 = 512
//!
//! [substrate.thinkers.warden]
//! enabled = true
//! interval_secs = 300
//! cpu_fraction = 0.25
//! fragmentation_trigger = 1.30
//!
//! [substrate.thinkers.predictor]
//! enabled = true
//! interval_secs = 30
//! cpu_fraction = 0.05
//! topic_ring_capacity = 64
//! top_k_communities = 3
//! min_breadcrumbs = 4
//!
//! [substrate.thinkers.dreamer]
//! enabled = true
//! interval_secs = 600
//! cpu_fraction = 0.10
//! max_proposals_per_tick = 4
//! proposal_queue_size = 256
//! bottleneck_ricci_max_q = -32
//! ```

use std::time::Duration;

use serde::{Deserialize, Serialize};

use super::{
    ConsolidatorConfig, DreamerConfig, PredictorConfig, ThinkerBudget, ThinkerRuntimeConfig,
    WardenConfig,
};

const fn default_max_total_cpu_fraction() -> f32 {
    1.0
}
const fn default_shutdown_timeout_secs() -> u64 {
    5
}
const fn default_min_tick_interval_secs() -> u64 {
    1
}
const fn default_enabled() -> bool {
    true
}
const fn default_topic_ring_capacity() -> usize {
    64
}

/// Top-level thinker fleet config, embedded under
/// `[substrate.thinkers]` in `HubConfig.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkersConfig {
    #[serde(default = "default_max_total_cpu_fraction")]
    pub max_total_cpu_fraction: f32,
    #[serde(default = "default_shutdown_timeout_secs")]
    pub shutdown_timeout_secs: u64,
    #[serde(default = "default_min_tick_interval_secs")]
    pub min_tick_interval_secs: u64,
    #[serde(default)]
    pub consolidator: ThinkerFleetConfig<ConsolidatorTomlCfg>,
    #[serde(default)]
    pub warden: ThinkerFleetConfig<WardenTomlCfg>,
    #[serde(default)]
    pub predictor: ThinkerFleetConfig<PredictorTomlCfg>,
    #[serde(default)]
    pub dreamer: ThinkerFleetConfig<DreamerTomlCfg>,
    /// T17l — canonical `_hilbert_features` enricher. Only spawned when
    /// the `enrichment` feature is enabled in this crate.
    #[serde(default)]
    pub hilbert_enricher: ThinkerFleetConfig<HilbertEnricherTomlCfg>,
    /// Auxiliary field exposing the shared ring capacity for Predictor
    /// (can also be set per-thinker via [`PredictorTomlCfg`]).
    #[serde(default = "default_topic_ring_capacity")]
    pub predictor_topic_ring_capacity: usize,
    /// Derived: assembled lazily by [`ThinkersConfig::runtime`].
    #[serde(skip, default)]
    pub runtime: ThinkerRuntimeConfig,
}

impl Default for ThinkersConfig {
    fn default() -> Self {
        let mut s = Self {
            max_total_cpu_fraction: default_max_total_cpu_fraction(),
            shutdown_timeout_secs: default_shutdown_timeout_secs(),
            min_tick_interval_secs: default_min_tick_interval_secs(),
            consolidator: ThinkerFleetConfig::default(),
            warden: ThinkerFleetConfig::default(),
            predictor: ThinkerFleetConfig::default(),
            dreamer: ThinkerFleetConfig::default(),
            hilbert_enricher: ThinkerFleetConfig::default(),
            predictor_topic_ring_capacity: default_topic_ring_capacity(),
            runtime: ThinkerRuntimeConfig::default(),
        };
        s.recompute_runtime();
        s
    }
}

impl ThinkersConfig {
    /// Populate `runtime` from the top-level scalar fields. Call once
    /// after loading from TOML.
    pub fn recompute_runtime(&mut self) {
        self.runtime = ThinkerRuntimeConfig {
            max_total_cpu_fraction: self.max_total_cpu_fraction,
            shutdown_timeout: Duration::from_secs(self.shutdown_timeout_secs),
            min_tick_interval: Duration::from_secs(self.min_tick_interval_secs.max(1)),
        };
    }
}

/// Generic wrapper adding the `enabled` flag and per-thinker specifics
/// on top of the concrete TOML config.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkerFleetConfig<T>
where
    T: Clone + Default,
{
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    #[serde(flatten)]
    pub inner_toml: T,
    /// Materialised inner config (computed lazily on first access).
    #[serde(skip, default)]
    inner_cached: Option<ConfigCache>,
    /// Topic ring capacity override for Predictor. Ignored for other
    /// thinkers; kept here so `ThinkerFleetConfig` stays generic.
    #[serde(default = "default_topic_ring_capacity")]
    pub topic_ring_capacity: usize,
}

impl<T: Clone + Default> Default for ThinkerFleetConfig<T> {
    fn default() -> Self {
        Self {
            enabled: true,
            inner_toml: T::default(),
            inner_cached: None,
            topic_ring_capacity: default_topic_ring_capacity(),
        }
    }
}

#[derive(Debug, Clone, Default)]
struct ConfigCache;

/// TOML payload for the Consolidator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidatorTomlCfg {
    #[serde(default)]
    pub interval_secs: Option<u64>,
    #[serde(default)]
    pub cpu_fraction: Option<f32>,
    #[serde(default)]
    pub max_tick_ms: Option<u64>,
    #[serde(default)]
    pub decay_factor: Option<f32>,
    #[serde(default)]
    pub synapse_prune_threshold_u16: Option<u16>,
    #[serde(default)]
    pub engram_reinforce_delta_u16: Option<u16>,
}
impl Default for ConsolidatorTomlCfg {
    fn default() -> Self {
        Self {
            interval_secs: None,
            cpu_fraction: None,
            max_tick_ms: None,
            decay_factor: None,
            synapse_prune_threshold_u16: None,
            engram_reinforce_delta_u16: None,
        }
    }
}

/// TOML payload for the CommunityWarden.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WardenTomlCfg {
    #[serde(default)]
    pub interval_secs: Option<u64>,
    #[serde(default)]
    pub cpu_fraction: Option<f32>,
    #[serde(default)]
    pub max_tick_ms: Option<u64>,
    #[serde(default)]
    pub fragmentation_trigger: Option<f32>,
}
impl Default for WardenTomlCfg {
    fn default() -> Self {
        Self {
            interval_secs: None,
            cpu_fraction: None,
            max_tick_ms: None,
            fragmentation_trigger: None,
        }
    }
}

/// TOML payload for the Predictor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictorTomlCfg {
    #[serde(default)]
    pub interval_secs: Option<u64>,
    #[serde(default)]
    pub cpu_fraction: Option<f32>,
    #[serde(default)]
    pub max_tick_ms: Option<u64>,
    #[serde(default)]
    pub top_k_communities: Option<usize>,
    #[serde(default)]
    pub min_breadcrumbs: Option<usize>,
}
impl Default for PredictorTomlCfg {
    fn default() -> Self {
        Self {
            interval_secs: None,
            cpu_fraction: None,
            max_tick_ms: None,
            top_k_communities: None,
            min_breadcrumbs: None,
        }
    }
}

/// TOML payload for the [`HilbertEnricher`](super::hilbert_enricher::HilbertEnricher)
/// (T17l canonical feature enrichment — `_hilbert_features` 64-72d).
///
/// Gated behind the `enrichment` feature : the struct exists
/// unconditionally (to keep `ThinkersConfig` field layout stable
/// across features) but the thinker itself is only spawned when
/// `enrichment` is active — see `spawn_standard_fleet`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HilbertEnricherTomlCfg {
    #[serde(default)]
    pub interval_secs: Option<u64>,
    #[serde(default)]
    pub cpu_fraction: Option<f32>,
    #[serde(default)]
    pub max_tick_ms: Option<u64>,
    /// Threshold below which a tick triggers the full Hilbert recompute.
    /// `Some(0.10)` = skip tick when coverage ≥ 10%.
    #[serde(default)]
    pub coverage_threshold: Option<f64>,
    /// Override for the levels parameter of `HilbertFeaturesConfig`.
    /// Default 8 → 64 dims (8 levels × 8 base facettes).
    #[serde(default)]
    pub levels: Option<usize>,
}
impl Default for HilbertEnricherTomlCfg {
    fn default() -> Self {
        Self {
            interval_secs: None,
            cpu_fraction: None,
            max_tick_ms: None,
            coverage_threshold: None,
            levels: None,
        }
    }
}

/// TOML payload for the Dreamer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamerTomlCfg {
    #[serde(default)]
    pub interval_secs: Option<u64>,
    #[serde(default)]
    pub cpu_fraction: Option<f32>,
    #[serde(default)]
    pub max_tick_ms: Option<u64>,
    #[serde(default)]
    pub max_proposals_per_tick: Option<usize>,
    #[serde(default)]
    pub proposal_queue_size: Option<usize>,
    #[serde(default)]
    pub bottleneck_ricci_max_q: Option<i16>,
}
impl Default for DreamerTomlCfg {
    fn default() -> Self {
        Self {
            interval_secs: None,
            cpu_fraction: None,
            max_tick_ms: None,
            max_proposals_per_tick: None,
            proposal_queue_size: None,
            bottleneck_ricci_max_q: None,
        }
    }
}

// ---------------------------------------------------------------------
// Conversion helpers: TOML override → materialised *Config.
//
// Each *FleetConfig exposes `.inner` which returns the materialised
// inner config resolved against the Default. Overrides are applied
// field-by-field; fields left as `None` in the TOML keep the default.
// ---------------------------------------------------------------------

impl ThinkerFleetConfig<ConsolidatorTomlCfg> {
    /// Resolve the TOML overrides against the default [`ConsolidatorConfig`].
    pub fn inner(&self) -> ConsolidatorConfig {
        let defaults = ConsolidatorConfig::default();
        let t = &self.inner_toml;
        ConsolidatorConfig {
            decay_factor: t.decay_factor.unwrap_or(defaults.decay_factor),
            synapse_prune_threshold_u16: t
                .synapse_prune_threshold_u16
                .unwrap_or(defaults.synapse_prune_threshold_u16),
            engram_reinforce_delta_u16: t
                .engram_reinforce_delta_u16
                .unwrap_or(defaults.engram_reinforce_delta_u16),
            interval: t
                .interval_secs
                .map(Duration::from_secs)
                .unwrap_or(defaults.interval),
            budget: resolve_budget(t.cpu_fraction, t.max_tick_ms, defaults.budget),
        }
    }
}

impl ThinkerFleetConfig<WardenTomlCfg> {
    /// Resolve the TOML overrides against the default [`WardenConfig`].
    pub fn inner(&self) -> WardenConfig {
        let defaults = WardenConfig::default();
        let t = &self.inner_toml;
        WardenConfig {
            fragmentation_trigger: t
                .fragmentation_trigger
                .unwrap_or(defaults.fragmentation_trigger),
            interval: t
                .interval_secs
                .map(Duration::from_secs)
                .unwrap_or(defaults.interval),
            budget: resolve_budget(t.cpu_fraction, t.max_tick_ms, defaults.budget),
        }
    }
}

impl ThinkerFleetConfig<PredictorTomlCfg> {
    /// Resolve the TOML overrides against the default [`PredictorConfig`].
    pub fn inner(&self) -> PredictorConfig {
        let defaults = PredictorConfig::default();
        let t = &self.inner_toml;
        PredictorConfig {
            top_k_communities: t.top_k_communities.unwrap_or(defaults.top_k_communities),
            min_breadcrumbs: t.min_breadcrumbs.unwrap_or(defaults.min_breadcrumbs),
            interval: t
                .interval_secs
                .map(Duration::from_secs)
                .unwrap_or(defaults.interval),
            budget: resolve_budget(t.cpu_fraction, t.max_tick_ms, defaults.budget),
        }
    }
}

impl ThinkerFleetConfig<DreamerTomlCfg> {
    /// Resolve the TOML overrides against the default [`DreamerConfig`].
    pub fn inner(&self) -> DreamerConfig {
        let defaults = DreamerConfig::default();
        let t = &self.inner_toml;
        DreamerConfig {
            max_proposals_per_tick: t
                .max_proposals_per_tick
                .unwrap_or(defaults.max_proposals_per_tick),
            proposal_queue_size: t
                .proposal_queue_size
                .unwrap_or(defaults.proposal_queue_size),
            bottleneck_ricci_max_q: t
                .bottleneck_ricci_max_q
                .unwrap_or(defaults.bottleneck_ricci_max_q),
            interval: t
                .interval_secs
                .map(Duration::from_secs)
                .unwrap_or(defaults.interval),
            budget: resolve_budget(t.cpu_fraction, t.max_tick_ms, defaults.budget),
        }
    }
}

#[cfg(feature = "enrichment")]
impl ThinkerFleetConfig<HilbertEnricherTomlCfg> {
    /// Resolve TOML overrides against the default
    /// [`super::hilbert_enricher::HilbertEnricherConfig`]. Any field
    /// left as `None` inherits the default. The `levels` override
    /// populates the inner `HilbertFeaturesConfig.levels`.
    pub fn inner(&self) -> super::hilbert_enricher::HilbertEnricherConfig {
        let defaults = super::hilbert_enricher::HilbertEnricherConfig::default();
        let t = &self.inner_toml;
        let mut hilbert = defaults.hilbert.clone();
        if let Some(l) = t.levels {
            hilbert.levels = l;
        }
        super::hilbert_enricher::HilbertEnricherConfig {
            budget: resolve_budget(t.cpu_fraction, t.max_tick_ms, defaults.budget),
            interval: t
                .interval_secs
                .map(Duration::from_secs)
                .unwrap_or(defaults.interval),
            coverage_threshold: t.coverage_threshold.unwrap_or(defaults.coverage_threshold),
            hilbert,
        }
    }
}

fn resolve_budget(
    cpu_fraction: Option<f32>,
    max_tick_ms: Option<u64>,
    defaults: ThinkerBudget,
) -> ThinkerBudget {
    ThinkerBudget {
        cpu_fraction: cpu_fraction.unwrap_or(defaults.cpu_fraction),
        max_tick_duration: max_tick_ms
            .map(Duration::from_millis)
            .unwrap_or(defaults.max_tick_duration),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_round_trip() {
        let cfg = ThinkersConfig::default();
        let toml = toml::to_string(&cfg).expect("serialise");
        let back: ThinkersConfig = toml::from_str(&toml).expect("deserialise");
        assert_eq!(back.max_total_cpu_fraction, cfg.max_total_cpu_fraction);
        assert_eq!(back.consolidator.enabled, cfg.consolidator.enabled);
    }

    #[test]
    fn toml_overrides_apply() {
        let text = r#"
            max_total_cpu_fraction = 0.5
            shutdown_timeout_secs = 10

            [consolidator]
            enabled = false
            interval_secs = 120
            decay_factor = 0.95

            [warden]
            enabled = true
            fragmentation_trigger = 1.5

            [predictor]
            enabled = true
            top_k_communities = 7

            [dreamer]
            enabled = false
        "#;
        let mut cfg: ThinkersConfig = toml::from_str(text).expect("parse");
        cfg.recompute_runtime();
        assert!(!cfg.consolidator.enabled);
        let c = cfg.consolidator.inner();
        assert_eq!(c.interval, Duration::from_secs(120));
        assert!((c.decay_factor - 0.95).abs() < 1e-6);

        let w = cfg.warden.inner();
        assert!((w.fragmentation_trigger - 1.5).abs() < 1e-6);

        let p = cfg.predictor.inner();
        assert_eq!(p.top_k_communities, 7);

        assert!(!cfg.dreamer.enabled);
        assert_eq!(cfg.runtime.max_total_cpu_fraction, 0.5);
        assert_eq!(cfg.runtime.shutdown_timeout, Duration::from_secs(10));
    }

    #[test]
    fn empty_toml_uses_defaults() {
        let mut cfg: ThinkersConfig = toml::from_str("").expect("empty parse");
        cfg.recompute_runtime();
        let c = cfg.consolidator.inner();
        assert_eq!(c.interval, ConsolidatorConfig::default().interval);
        assert!(cfg.consolidator.enabled);
    }
}
