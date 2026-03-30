//! Attention formula management: seed formulas, CRUD, and evolution support.
//!
//! Formulas are stored as `:AttnFormula` nodes in PersonaDB with their DSL
//! serialized as JSON. This module defines the seed formulas and provides
//! helpers for the formula lifecycle.

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

/// The 6 seed formulas as JSON strings.
///
/// These are pre-serialized to avoid a dependency on `retrieval` from `persona`.
/// The AttnOp enum lives in `retrieval::attn_dsl` — we serialize the seeds
/// as known-good JSON that round-trips correctly.
pub fn seed_formulas() -> Vec<(&'static str, &'static str, Vec<&'static str>)> {
    vec![
        // F0: Identity — no modification to attention (baseline)
        (
            "F0-Identity",
            r#""Identity""#,
            vec![],
        ),
        // F1: GravityLinear — closer nodes in the graph get more attention
        (
            "F1-GravityLinear",
            r#"{"BiasAdd":{"source":{"GraphDistance":{"max_hops":4}},"weight":1.0}}"#,
            vec!["factual", "multi_hop"],
        ),
        // F2: RepulsionCompeting — mask far nodes + attenuate weak synapses
        (
            "F2-RepulsionCompeting",
            r#"{"Sequence":[{"Mask":{"condition":{"GraphDistanceAbove":3}}},{"BiasAdd":{"source":"SynapseEnergy","weight":-0.5}}]}"#,
            vec!["reasoning", "negation"],
        ),
        // F3: WarpGNN — use GNN hidden states to warp queries (placeholder until GNN)
        (
            "F3-WarpGNN",
            r#"{"WarpQ":{"delta_source":"GnnDelta","alpha":0.5}}"#,
            vec!["complex", "multi_hop"],
        ),
        // F4: PerHeadTopo — heads 0-15 see all, heads 16-31 only close nodes
        (
            "F4-PerHeadTopo",
            r#"{"PerHead":[[[0,16],"Identity"],[[16,32],{"Mask":{"condition":{"GraphDistanceAbove":2}}}]]}"#,
            vec!["structured", "factual"],
        ),
        // F5: QueryDelegateCompute — delegate to graph when uncertain
        (
            "F5-QueryDelegateCompute",
            r#"{"QueryDelegate":{"entropy_threshold":1.5,"query_type":"Compute","max_inject_tokens":32}}"#,
            vec!["math", "computation"],
        ),
    ]
}

/// Number of seed formulas.
pub const SEED_COUNT: usize = 6;

/// Maximum population size before GC triggers.
pub const MAX_POPULATION: usize = 50;
