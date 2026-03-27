//! Configuration for the RAG pipeline.

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
    ["id", "uuid", "created_at", "updated_at", "modified_at", "_id"]
        .iter()
        .map(|s| (*s).to_string())
        .collect()
}

fn default_max_relations_display() -> usize {
    10
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
