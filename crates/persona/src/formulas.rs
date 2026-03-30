//! Attention formula management: seed formulas, CRUD, and evolution support.
//!
//! Zero-seed architecture: only F0-Identity is seeded. All other formulas
//! emerge through mutation + selection by structural reward.
//! Formulas are stored as `:AttnFormula` nodes in PersonaDB with their DSL
//! serialized as JSON.

use obrain_common::types::NodeId;

/// Metadata for a formula node in PersonaDB.
#[derive(Debug, Clone)]
pub struct AttnFormulaNode {
    pub id: NodeId,
    /// JSON-serialized AttnOp DSL.
    pub dsl_json: String,
    /// Short human-readable name (e.g., "F0-Identity").
    pub name: String,
    /// Ξ(t) energy: decays on bad reward, grows on good reward.
    pub energy: f64,
    /// Running average of rewards when this formula was active.
    pub avg_reward: f64,
    /// Number of times this formula was selected for generation.
    pub activation_count: i64,
    /// Evolution generation (0 = seed, 1+ = mutated).
    pub generation: i64,
    /// Context affinity tags (e.g., "math", "creative").
    pub context_affinity: Vec<String>,
    /// Whether this formula is active (alive in the population).
    pub active: bool,
    /// Optional parent formula (for MUTATED_FROM lineage).
    pub parent_id: Option<NodeId>,
}

/// Seed formula: Identity only.
///
/// Zero-seed principle: the system starts with no attention overlay.
/// The formula evolution mechanism (mutation + structural reward selection)
/// discovers useful formulas autonomously. This works in any language
/// and for any model architecture.
pub fn seed_formulas() -> Vec<(&'static str, &'static str, Vec<&'static str>)> {
    vec![
        // F0: Identity — no modification to attention (baseline)
        // All other formulas emerge through evolution.
        (
            "F0-Identity",
            r#""Identity""#,
            vec![],
        ),
    ]
}

/// Number of seed formulas.
pub const SEED_COUNT: usize = 1;

/// Maximum population size before GC triggers.
pub const MAX_POPULATION: usize = 50;
