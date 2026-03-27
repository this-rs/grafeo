//! Configuration for the RAG pipeline.

use crate::error::{RagError, RagResult};
use serde::{Deserialize, Serialize};

/// Configuration for the RAG pipeline.
///
/// Controls engram recall, spreading activation, context building,
/// and feedback reinforcement parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagConfig {
    // --- Engram Recall ---
    /// Maximum number of engrams to retrieve via Hopfield recall.
    pub max_engrams: usize,

    /// Minimum confidence threshold for engram recall results.
    pub min_recall_confidence: f64,

    /// Weight for direct recall path (vs spectral/Hopfield).
    /// The spectral weight is `1.0 - direct_recall_weight`.
    pub direct_recall_weight: f64,

    // --- Spreading Activation ---
    /// Maximum BFS hops from activated engram nodes.
    pub activation_depth: u32,

    /// Energy decay factor per hop during spreading activation.
    pub activation_decay: f64,

    /// Minimum propagated energy to keep a node in the activation map.
    pub min_activation_energy: f64,

    /// Maximum nodes to activate (circuit breaker).
    pub max_activated_nodes: usize,

    // --- Context Building ---
    /// Maximum tokens in the generated context.
    pub token_budget: usize,

    /// Approximate characters per token (for estimation).
    pub chars_per_token: f64,

    /// Maximum number of nodes to include in the context.
    pub max_context_nodes: usize,

    /// Include relation information in the context output.
    pub include_relations: bool,

    /// Include node labels in the context output.
    pub include_labels: bool,

    /// Property names to exclude from context output (noise filtering).
    /// Properties whose name matches any of these strings (case-insensitive)
    /// are hidden from the formatted context.
    #[serde(default = "default_noise_properties")]
    pub noise_properties: Vec<String>,

    /// Maximum relations to display per direction (outgoing/incoming).
    /// Excess relations show a "... and N more" indicator.
    #[serde(default = "default_max_relations_display")]
    pub max_relations_display: usize,

    // --- Feedback ---
    /// Hebbian reinforcement amount for synapses between co-activated concepts.
    pub feedback_reinforce_amount: f64,

    /// Energy boost amount for engrams whose content was used in the response.
    pub feedback_energy_boost: f64,
}

fn default_noise_properties() -> Vec<String> {
    [
        "id",
        "uuid",
        "created_at",
        "updated_at",
        "modified_at",
        "_id",
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect()
}

fn default_max_relations_display() -> usize {
    10
}

impl RagConfig {
    /// Fast preset — minimal recall, shallow activation, small budget.
    /// Best for low-latency queries or resource-constrained environments.
    pub fn fast() -> Self {
        Self {
            max_engrams: 5,
            min_recall_confidence: 0.2,
            activation_depth: 1,
            activation_decay: 0.3,
            max_activated_nodes: 100,
            token_budget: 1000,
            max_context_nodes: 10,
            ..Self::default()
        }
    }

    /// Balanced preset — the default configuration.
    pub fn balanced() -> Self {
        Self::default()
    }

    /// Thorough preset — deep recall, wider activation, large budget.
    /// Best for complex queries where comprehensiveness matters more than speed.
    pub fn thorough() -> Self {
        Self {
            max_engrams: 30,
            min_recall_confidence: 0.05,
            activation_depth: 3,
            activation_decay: 0.6,
            max_activated_nodes: 1000,
            token_budget: 4000,
            max_context_nodes: 50,
            ..Self::default()
        }
    }

    /// Validate configuration parameters.
    ///
    /// Returns an error if any parameter has an invalid value.
    pub fn validate(&self) -> RagResult<()> {
        if self.activation_decay <= 0.0 || self.activation_decay >= 1.0 {
            return Err(RagError::Config(
                "activation_decay must be in (0.0, 1.0)".into(),
            ));
        }
        if self.token_budget == 0 {
            return Err(RagError::Config("token_budget must be > 0".into()));
        }
        if self.max_context_nodes == 0 {
            return Err(RagError::Config("max_context_nodes must be > 0".into()));
        }
        if self.max_engrams == 0 {
            return Err(RagError::Config("max_engrams must be > 0".into()));
        }
        if self.chars_per_token <= 0.0 {
            return Err(RagError::Config("chars_per_token must be > 0.0".into()));
        }
        Ok(())
    }

    /// Return the preset name if this config matches a known preset.
    pub fn preset_name(&self) -> &'static str {
        let fast = Self::fast();
        let thorough = Self::thorough();

        if self.max_engrams == fast.max_engrams
            && self.activation_depth == fast.activation_depth
            && self.token_budget == fast.token_budget
        {
            "fast"
        } else if self.max_engrams == thorough.max_engrams
            && self.activation_depth == thorough.activation_depth
            && self.token_budget == thorough.token_budget
        {
            "thorough"
        } else {
            "balanced"
        }
    }
}

impl Default for RagConfig {
    fn default() -> Self {
        Self {
            // Recall
            max_engrams: 10,
            min_recall_confidence: 0.1,
            direct_recall_weight: 0.6,

            // Activation
            activation_depth: 2,
            activation_decay: 0.5,
            min_activation_energy: 0.01,
            max_activated_nodes: 500,

            // Context
            token_budget: 2000,
            chars_per_token: 4.0,
            max_context_nodes: 30,
            include_relations: true,
            include_labels: true,
            noise_properties: default_noise_properties(),
            max_relations_display: 10,

            // Feedback
            feedback_reinforce_amount: 0.3,
            feedback_energy_boost: 0.5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid() {
        let config = RagConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn fast_preset_values() {
        let fast = RagConfig::fast();
        assert_eq!(fast.max_engrams, 5);
        assert_eq!(fast.activation_depth, 1);
        assert_eq!(fast.token_budget, 1000);
        assert_eq!(fast.max_context_nodes, 10);
        assert!(fast.validate().is_ok());
    }

    #[test]
    fn balanced_is_default() {
        let balanced = RagConfig::balanced();
        let default = RagConfig::default();
        assert_eq!(balanced.max_engrams, default.max_engrams);
        assert_eq!(balanced.token_budget, default.token_budget);
    }

    #[test]
    fn thorough_preset_values() {
        let thorough = RagConfig::thorough();
        assert_eq!(thorough.max_engrams, 30);
        assert_eq!(thorough.activation_depth, 3);
        assert_eq!(thorough.token_budget, 4000);
        assert_eq!(thorough.max_context_nodes, 50);
        assert!(thorough.validate().is_ok());
    }

    #[test]
    fn validate_rejects_bad_decay() {
        let config = RagConfig {
            activation_decay: 0.0,
            ..RagConfig::default()
        };
        assert!(config.validate().is_err());

        let config = RagConfig {
            activation_decay: 1.0,
            ..RagConfig::default()
        };
        assert!(config.validate().is_err());

        let config = RagConfig {
            activation_decay: -0.5,
            ..RagConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn validate_rejects_zero_budget() {
        let config = RagConfig {
            token_budget: 0,
            ..RagConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn validate_rejects_zero_context_nodes() {
        let config = RagConfig {
            max_context_nodes: 0,
            ..RagConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn validate_rejects_zero_engrams() {
        let config = RagConfig {
            max_engrams: 0,
            ..RagConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn validate_rejects_zero_chars_per_token() {
        let config = RagConfig {
            chars_per_token: 0.0,
            ..RagConfig::default()
        };
        assert!(config.validate().is_err());

        let config = RagConfig {
            chars_per_token: -1.0,
            ..RagConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn preset_name_detects_fast() {
        let config = RagConfig::fast();
        assert_eq!(config.preset_name(), "fast");
    }

    #[test]
    fn preset_name_detects_thorough() {
        let config = RagConfig::thorough();
        assert_eq!(config.preset_name(), "thorough");
    }

    #[test]
    fn preset_name_detects_balanced() {
        let config = RagConfig::default();
        assert_eq!(config.preset_name(), "balanced");
    }

    #[test]
    fn preset_name_custom_is_balanced() {
        // Custom config that doesn't match any preset → falls through to "balanced"
        let config = RagConfig {
            max_engrams: 42,
            activation_depth: 7,
            token_budget: 9999,
            ..RagConfig::default()
        };
        assert_eq!(config.preset_name(), "balanced");
    }

    #[test]
    fn default_noise_properties_include_common_ids() {
        let noise = default_noise_properties();
        assert!(noise.contains(&"id".to_string()));
        assert!(noise.contains(&"uuid".to_string()));
        assert!(noise.contains(&"created_at".to_string()));
        assert!(noise.contains(&"_id".to_string()));
    }

    #[test]
    fn serde_roundtrip() {
        let config = RagConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: RagConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.max_engrams, config.max_engrams);
        assert_eq!(deserialized.token_budget, config.token_budget);
        assert_eq!(
            deserialized.noise_properties.len(),
            config.noise_properties.len()
        );
    }

    #[test]
    fn serde_defaults_applied_when_missing() {
        // JSON without noise_properties and max_relations_display
        let json = r#"{
            "max_engrams": 10,
            "min_recall_confidence": 0.1,
            "direct_recall_weight": 0.6,
            "activation_depth": 2,
            "activation_decay": 0.5,
            "min_activation_energy": 0.01,
            "max_activated_nodes": 500,
            "token_budget": 2000,
            "chars_per_token": 4.0,
            "max_context_nodes": 30,
            "include_relations": true,
            "include_labels": true,
            "feedback_reinforce_amount": 0.3,
            "feedback_energy_boost": 0.5
        }"#;
        let config: RagConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.noise_properties.len(), 6);
        assert_eq!(config.max_relations_display, 10);
    }
}
