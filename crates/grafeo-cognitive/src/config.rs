//! Configuration for the cognitive engine.
//!
//! [`CognitiveConfig`] provides a unified configuration for all cognitive subsystems.
//! Each subsystem can be independently enabled/disabled with its own parameters.
//! Supports TOML deserialization via `serde`.

use serde::Deserialize;
#[cfg(any(
    feature = "energy",
    feature = "synapse",
    feature = "co-change",
    feature = "memory",
    feature = "stagnation"
))]
use std::time::Duration;

// ---------------------------------------------------------------------------
// Per-subsystem config structs (serde-compatible)
// ---------------------------------------------------------------------------

/// Serializable configuration for the energy subsystem.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct EnergyConfigToml {
    /// Whether the energy subsystem is enabled.
    pub enabled: bool,
    /// Energy boost applied to nodes touched by a mutation.
    pub boost_on_mutation: f64,
    /// Default energy for newly tracked nodes.
    pub default_energy: f64,
    /// Default half-life for energy decay, in seconds.
    pub half_life_secs: u64,
    /// Minimum energy threshold.
    pub min_energy: f64,
}

impl Default for EnergyConfigToml {
    fn default() -> Self {
        Self {
            enabled: true,
            boost_on_mutation: 1.0,
            default_energy: 1.0,
            half_life_secs: 24 * 3600,
            min_energy: 0.01,
        }
    }
}

impl EnergyConfigToml {
    /// Converts to the runtime `EnergyConfig` used by the energy subsystem.
    #[cfg(feature = "energy")]
    pub fn to_runtime(&self) -> crate::energy::EnergyConfig {
        crate::energy::EnergyConfig {
            boost_on_mutation: self.boost_on_mutation,
            default_energy: self.default_energy,
            default_half_life: Duration::from_secs(self.half_life_secs),
            min_energy: self.min_energy,
        }
    }
}

/// Serializable configuration for the synapse subsystem.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct SynapseConfigToml {
    /// Whether the synapse subsystem is enabled.
    pub enabled: bool,
    /// Default initial weight for a newly created synapse.
    pub initial_weight: f64,
    /// Reinforcement amount when nodes are co-activated.
    pub reinforce_amount: f64,
    /// Default half-life for weight decay, in seconds.
    pub half_life_secs: u64,
    /// Minimum weight threshold for pruning.
    pub min_weight: f64,
}

impl Default for SynapseConfigToml {
    fn default() -> Self {
        Self {
            enabled: true,
            initial_weight: 0.1,
            reinforce_amount: 0.2,
            half_life_secs: 7 * 24 * 3600,
            min_weight: 0.01,
        }
    }
}

impl SynapseConfigToml {
    /// Converts to the runtime `SynapseConfig` used by the synapse subsystem.
    #[cfg(feature = "synapse")]
    pub fn to_runtime(&self) -> crate::synapse::SynapseConfig {
        crate::synapse::SynapseConfig {
            initial_weight: self.initial_weight,
            reinforce_amount: self.reinforce_amount,
            default_half_life: Duration::from_secs(self.half_life_secs),
            min_weight: self.min_weight,
        }
    }
}

/// Serializable configuration for the fabric subsystem.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct FabricConfigToml {
    /// Whether the fabric subsystem is enabled.
    pub enabled: bool,
}

impl Default for FabricConfigToml {
    fn default() -> Self {
        Self { enabled: false }
    }
}

/// Serializable configuration for the co-change detection subsystem.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct CoChangeConfigToml {
    /// Whether the co-change subsystem is enabled.
    pub enabled: bool,
    /// Window duration in seconds for grouping mutations.
    pub window_duration_secs: u64,
    /// Decay half-life for strength, in seconds.
    pub strength_half_life_secs: u64,
    /// Maximum batch nodes before skipping combinatorial pairs.
    pub max_batch_nodes: usize,
}

impl Default for CoChangeConfigToml {
    fn default() -> Self {
        Self {
            enabled: false,
            window_duration_secs: 0,
            strength_half_life_secs: 30 * 24 * 3600,
            max_batch_nodes: 100,
        }
    }
}

impl CoChangeConfigToml {
    /// Converts to the runtime `CoChangeConfig`.
    #[cfg(feature = "co-change")]
    pub fn to_runtime(&self) -> crate::co_change::CoChangeConfig {
        crate::co_change::CoChangeConfig {
            window_duration: Duration::from_secs(self.window_duration_secs),
            strength_half_life: Duration::from_secs(self.strength_half_life_secs),
            max_batch_nodes: self.max_batch_nodes,
        }
    }
}

/// Serializable configuration for the scar subsystem.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ScarConfigToml {
    /// Whether the scar subsystem is enabled.
    pub enabled: bool,
}

impl Default for ScarConfigToml {
    fn default() -> Self {
        Self { enabled: false }
    }
}

/// Serializable configuration for the memory horizons subsystem.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct MemoryConfigToml {
    /// Whether the memory horizons subsystem is enabled.
    pub enabled: bool,
    /// Energy threshold for promotion (Operational → Consolidated).
    pub promotion_energy_threshold: f64,
    /// Minimum age (seconds) before promotion is allowed.
    pub promotion_min_age_secs: u64,
    /// Energy threshold below which demotion occurs.
    pub demotion_energy_threshold: f64,
    /// Maximum idle time (seconds) before demotion.
    pub demotion_max_idle_secs: u64,
    /// Sweep interval in seconds (how often the manager checks horizons).
    pub sweep_interval_secs: u64,
}

impl Default for MemoryConfigToml {
    fn default() -> Self {
        Self {
            enabled: false,
            promotion_energy_threshold: 2.0,
            promotion_min_age_secs: 3600,
            demotion_energy_threshold: 0.1,
            demotion_max_idle_secs: 7 * 24 * 3600,
            sweep_interval_secs: 3600,
        }
    }
}

impl MemoryConfigToml {
    /// Converts to the runtime [`MemoryConfig`] used by the memory subsystem.
    #[cfg(feature = "memory")]
    pub fn to_runtime(&self) -> crate::memory::MemoryConfig {
        crate::memory::MemoryConfig {
            promotion_energy_threshold: self.promotion_energy_threshold,
            promotion_min_age: Duration::from_secs(self.promotion_min_age_secs),
            demotion_energy_threshold: self.demotion_energy_threshold,
            demotion_max_idle: Duration::from_secs(self.demotion_max_idle_secs),
            sweep_interval: Duration::from_secs(self.sweep_interval_secs),
        }
    }
}

/// Serializable configuration for the stagnation detection subsystem.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct StagnationConfigToml {
    /// Whether the stagnation detection subsystem is enabled.
    pub enabled: bool,
    /// Weight for the energy component in the stagnation formula.
    pub weight_energy: f64,
    /// Weight for the mutation age component.
    pub weight_mutation_age: f64,
    /// Weight for the synapse activity component.
    pub weight_synapse_activity: f64,
    /// Duration (seconds) used to normalize `last_mutation_age` to `[0, 1]`.
    pub max_mutation_age_secs: u64,
    /// Threshold above which a community is considered stagnant.
    pub stagnation_threshold: f64,
    /// Duration (seconds) for "recently reinforced" synapse window.
    pub synapse_recent_window_secs: u64,
    /// Number of historical snapshots for trend detection.
    pub trend_window_size: usize,
    /// Minimum delta to classify as improving/degrading.
    pub trend_tolerance: f64,
    /// Scan interval in seconds.
    pub scan_interval_secs: u64,
}

impl Default for StagnationConfigToml {
    fn default() -> Self {
        Self {
            enabled: false,
            weight_energy: 0.4,
            weight_mutation_age: 0.35,
            weight_synapse_activity: 0.25,
            max_mutation_age_secs: 30 * 24 * 3600,
            stagnation_threshold: 0.7,
            synapse_recent_window_secs: 7 * 24 * 3600,
            trend_window_size: 5,
            trend_tolerance: 0.05,
            scan_interval_secs: 3600,
        }
    }
}

impl StagnationConfigToml {
    /// Converts to the runtime `StagnationConfig` used by the stagnation subsystem.
    #[cfg(feature = "stagnation")]
    pub fn to_runtime(&self) -> crate::stagnation::StagnationConfig {
        crate::stagnation::StagnationConfig {
            weight_energy: self.weight_energy,
            weight_mutation_age: self.weight_mutation_age,
            weight_synapse_activity: self.weight_synapse_activity,
            max_mutation_age: Duration::from_secs(self.max_mutation_age_secs),
            stagnation_threshold: self.stagnation_threshold,
            synapse_recent_window: Duration::from_secs(self.synapse_recent_window_secs),
            trend_window_size: self.trend_window_size,
            trend_tolerance: self.trend_tolerance,
            scan_interval: Duration::from_secs(self.scan_interval_secs),
        }
    }
}

/// Serializable configuration for the GDS refresh scheduler.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct GdsRefreshConfigToml {
    /// Whether GDS refresh is enabled.
    pub enabled: bool,
    /// Refresh interval in seconds.
    pub refresh_interval_secs: u64,
    /// Number of mutations that triggers an immediate refresh.
    pub mutation_threshold: u64,
}

impl Default for GdsRefreshConfigToml {
    fn default() -> Self {
        Self {
            enabled: false,
            refresh_interval_secs: 5 * 60,
            mutation_threshold: 1000,
        }
    }
}

// ---------------------------------------------------------------------------
// CognitiveConfig — unified top-level configuration
// ---------------------------------------------------------------------------

/// Unified configuration for all cognitive subsystems.
///
/// Supports TOML deserialization. By default, only `energy` and `synapse` are
/// enabled (the "cognitive" feature flag minimum).
///
/// # Example TOML
///
/// ```toml
/// [cognitive]
/// [cognitive.energy]
/// enabled = true
/// boost_on_mutation = 1.0
/// half_life_secs = 86400
///
/// [cognitive.synapse]
/// enabled = true
/// reinforce_amount = 0.2
///
/// [cognitive.fabric]
/// enabled = true
///
/// [cognitive.co_change]
/// enabled = true
/// ```
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct CognitiveConfig {
    /// Energy subsystem configuration.
    pub energy: EnergyConfigToml,
    /// Synapse subsystem configuration.
    pub synapse: SynapseConfigToml,
    /// Knowledge fabric subsystem configuration.
    pub fabric: FabricConfigToml,
    /// Co-change detection configuration.
    pub co_change: CoChangeConfigToml,
    /// GDS refresh scheduler configuration.
    pub gds_refresh: GdsRefreshConfigToml,
    /// Scar memory configuration.
    pub scar: ScarConfigToml,
    /// Memory horizons configuration.
    pub memory: MemoryConfigToml,
    /// Stagnation detection configuration.
    pub stagnation: StagnationConfigToml,
}

impl Default for CognitiveConfig {
    fn default() -> Self {
        Self {
            energy: EnergyConfigToml::default(),
            synapse: SynapseConfigToml::default(),
            fabric: FabricConfigToml::default(),
            co_change: CoChangeConfigToml::default(),
            gds_refresh: GdsRefreshConfigToml::default(),
            scar: ScarConfigToml::default(),
            memory: MemoryConfigToml::default(),
            stagnation: StagnationConfigToml::default(),
        }
    }
}

impl CognitiveConfig {
    /// Creates a default config (energy + synapse enabled).
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a config with all subsystems enabled.
    pub fn full() -> Self {
        Self {
            energy: EnergyConfigToml {
                enabled: true,
                ..Default::default()
            },
            synapse: SynapseConfigToml {
                enabled: true,
                ..Default::default()
            },
            fabric: FabricConfigToml { enabled: true },
            co_change: CoChangeConfigToml {
                enabled: true,
                ..Default::default()
            },
            gds_refresh: GdsRefreshConfigToml {
                enabled: true,
                ..Default::default()
            },
            scar: ScarConfigToml { enabled: true },
            memory: MemoryConfigToml {
                enabled: true,
                ..Default::default()
            },
            stagnation: StagnationConfigToml {
                enabled: true,
                ..Default::default()
            },
        }
    }

    /// Parses a `CognitiveConfig` from a TOML string.
    pub fn from_toml(toml_str: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(toml_str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_enables_energy_and_synapse() {
        let config = CognitiveConfig::default();
        assert!(config.energy.enabled);
        assert!(config.synapse.enabled);
        assert!(!config.fabric.enabled);
        assert!(!config.co_change.enabled);
        assert!(!config.scar.enabled);
        assert!(!config.memory.enabled);
        assert!(!config.stagnation.enabled);
    }

    #[test]
    fn full_config_enables_all() {
        let config = CognitiveConfig::full();
        assert!(config.energy.enabled);
        assert!(config.synapse.enabled);
        assert!(config.fabric.enabled);
        assert!(config.co_change.enabled);
        assert!(config.scar.enabled);
        assert!(config.memory.enabled);
        assert!(config.stagnation.enabled);
    }

    #[test]
    fn deserialize_from_toml() {
        let toml_str = r#"
[energy]
enabled = true
boost_on_mutation = 2.0
half_life_secs = 3600

[synapse]
enabled = true
reinforce_amount = 0.5

[fabric]
enabled = true

[co_change]
enabled = true
max_batch_nodes = 50
"#;
        let config: CognitiveConfig = toml::from_str(toml_str).unwrap();
        assert!(config.energy.enabled);
        assert_eq!(config.energy.boost_on_mutation, 2.0);
        assert_eq!(config.energy.half_life_secs, 3600);
        assert!(config.synapse.enabled);
        assert_eq!(config.synapse.reinforce_amount, 0.5);
        assert!(config.fabric.enabled);
        assert!(config.co_change.enabled);
        assert_eq!(config.co_change.max_batch_nodes, 50);
        // defaults preserved for unspecified
        assert!(!config.scar.enabled);
        assert!(!config.memory.enabled);
    }

    #[test]
    #[cfg(feature = "energy")]
    fn energy_config_to_runtime() {
        let toml_config = EnergyConfigToml {
            half_life_secs: 7200,
            boost_on_mutation: 2.5,
            ..Default::default()
        };
        let runtime = toml_config.to_runtime();
        assert_eq!(runtime.boost_on_mutation, 2.5);
        assert_eq!(runtime.default_half_life, Duration::from_secs(7200));
    }

    #[test]
    #[cfg(feature = "synapse")]
    fn synapse_config_to_runtime() {
        let toml_config = SynapseConfigToml {
            reinforce_amount: 0.5,
            ..Default::default()
        };
        let runtime = toml_config.to_runtime();
        assert_eq!(runtime.reinforce_amount, 0.5);
    }
}
