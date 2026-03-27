//! Extra tests for config module — edge cases and TOML parsing.

use obrain_cognitive::config::CognitiveConfig;

#[test]
fn from_toml_empty_string_uses_defaults() {
    let config = CognitiveConfig::from_toml("").unwrap();
    assert!(config.energy.enabled);
    assert!(config.synapse.enabled);
    assert!(!config.fabric.enabled);
}

#[test]
fn from_toml_invalid_syntax_returns_error() {
    let result = CognitiveConfig::from_toml("[[[bad toml");
    assert!(result.is_err());
}

#[test]
fn from_toml_partial_override() {
    let toml_str = r#"
[energy]
boost_on_mutation = 5.0
"#;
    let config = CognitiveConfig::from_toml(toml_str).unwrap();
    assert_eq!(config.energy.boost_on_mutation, 5.0);
    // rest are defaults
    assert_eq!(config.energy.half_life_secs, 24 * 3600);
    assert!(config.energy.enabled);
}

#[test]
fn cognitive_config_new_equals_default() {
    let a = CognitiveConfig::new();
    let b = CognitiveConfig::default();
    assert_eq!(a.energy.enabled, b.energy.enabled);
    assert_eq!(a.synapse.enabled, b.synapse.enabled);
    assert_eq!(a.fabric.enabled, b.fabric.enabled);
}

#[test]
fn full_config_gds_refresh_enabled() {
    let config = CognitiveConfig::full();
    assert!(config.gds_refresh.enabled);
    assert_eq!(config.gds_refresh.refresh_interval_secs, 5 * 60);
    assert_eq!(config.gds_refresh.mutation_threshold, 1000);
}

#[test]
fn default_gds_refresh_config_toml_values() {
    let config = CognitiveConfig::default();
    assert!(!config.gds_refresh.enabled);
    assert_eq!(config.gds_refresh.refresh_interval_secs, 300);
    assert_eq!(config.gds_refresh.mutation_threshold, 1000);
}

#[test]
fn from_toml_all_subsystem_sections() {
    let toml_str = r#"
[energy]
enabled = false

[synapse]
enabled = false
initial_weight = 0.5

[fabric]
enabled = true

[co_change]
enabled = true
window_duration_secs = 30
strength_half_life_secs = 86400

[scar]
enabled = true

[memory]
enabled = true
promotion_energy_threshold = 5.0
demotion_max_idle_secs = 172800

[stagnation]
enabled = true
weight_energy = 0.5
trend_tolerance = 0.1
scan_interval_secs = 1800

[gds_refresh]
enabled = true
refresh_interval_secs = 600
mutation_threshold = 500
"#;
    let config = CognitiveConfig::from_toml(toml_str).unwrap();
    assert!(!config.energy.enabled);
    assert!(!config.synapse.enabled);
    assert_eq!(config.synapse.initial_weight, 0.5);
    assert!(config.fabric.enabled);
    assert!(config.co_change.enabled);
    assert_eq!(config.co_change.window_duration_secs, 30);
    assert_eq!(config.co_change.strength_half_life_secs, 86400);
    assert!(config.scar.enabled);
    assert!(config.memory.enabled);
    assert_eq!(config.memory.promotion_energy_threshold, 5.0);
    assert_eq!(config.memory.demotion_max_idle_secs, 172800);
    assert!(config.stagnation.enabled);
    assert_eq!(config.stagnation.weight_energy, 0.5);
    assert_eq!(config.stagnation.trend_tolerance, 0.1);
    assert_eq!(config.stagnation.scan_interval_secs, 1800);
    assert!(config.gds_refresh.enabled);
    assert_eq!(config.gds_refresh.refresh_interval_secs, 600);
    assert_eq!(config.gds_refresh.mutation_threshold, 500);
}

#[test]
fn config_debug_formatting() {
    let config = CognitiveConfig::default();
    let dbg = format!("{:?}", config);
    assert!(dbg.contains("CognitiveConfig"), "got: {dbg}");
    assert!(dbg.contains("energy"), "got: {dbg}");
    assert!(dbg.contains("synapse"), "got: {dbg}");
}

#[test]
fn config_clone() {
    let config = CognitiveConfig::full();
    let cloned = config.clone();
    assert_eq!(cloned.energy.enabled, config.energy.enabled);
    assert_eq!(cloned.fabric.enabled, config.fabric.enabled);
}
