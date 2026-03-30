//! DSL for evolutionary attention formulas (Phase 4 — AFE).
//!
//! This module defines the `AttnOp` enum and supporting types that represent
//! attention formula operations. Formulas are stored as `:AttnFormula` nodes
//! in PersonaDB and compiled down to `kq_mask` / `kq_b` tensors for llama.cpp.
//!
//! # Safety constraints
//! - Max nesting depth: 8 levels (`validate_depth`)
//! - No recursion (tree, not graph)
//! - All weights clamped to `[-2.0, 2.0]` (`clamp_weights`)
//! - Serde JSON for persistence in PersonaDB

use serde::{Deserialize, Serialize};

// ─── Core DSL ───────────────────────────────────────────────────────────────

/// Top-level attention operation.
///
/// Each variant maps to a specific compilation target in llama.cpp:
/// - `Mask`, `BiasAdd(≤0)` → `kq_mask` (flash-attention compatible)
/// - `BiasAdd(>0)`, `BiasScale`, `WarpQ`, `WarpK` → `kq_b` (requires Phase 2 fork)
/// - `PerHead` → `kq_mask` with `ne[2]` broadcasting
/// - `QueryDelegate` → token injection into next batch
/// - `Conditional`, `Sequence` → evaluated Rust-side before compilation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AttnOp {
    // === Post dot-product score operations ===
    /// Additive bias: score[i][j] += weight * source(i, j)
    BiasAdd { source: BiasSource, weight: f32 },

    /// Multiplicative bias (approx): score[i][j] += weight * ln(source(i, j))
    BiasScale { source: BiasSource, weight: f32 },

    /// Conditional masking: score[i][j] = -∞ if condition(i, j)
    Mask { condition: MaskCondition },

    // === Pre dot-product representation operations ===
    // Compiled via dot product decomposition → kq_b
    /// Warp Q: simulated via additive bias dot(ΔQ, K) injected as kq_b
    WarpQ { delta_source: DeltaSource, alpha: f32 },

    /// Warp K: simulated via additive bias dot(Q, ΔK) injected as kq_b
    WarpK { delta_source: DeltaSource, alpha: f32 },

    // === Computational delegation (REV.3) ===
    /// When entropy exceeds threshold, delegate to Obrain graph.
    /// Result is injected as tokens in the context.
    QueryDelegate {
        entropy_threshold: f32,
        query_type: QueryType,
        max_inject_tokens: u32,
    },

    // === Composition ===
    /// Sequential application of operations (left to right).
    Sequence(Vec<AttnOp>),

    /// Per-head specialization: different ops for different head ranges.
    PerHead(Vec<(HeadRange, AttnOp)>),

    /// Runtime conditional: evaluated before compilation.
    Conditional {
        condition: RuntimeCondition,
        then_op: Box<AttnOp>,
        else_op: Box<AttnOp>,
    },

    /// No-op: pass-through (identity element for composition).
    Identity,
}

// ─── Supporting types ───────────────────────────────────────────────────────

/// Sources of bias values for BiasAdd / BiasScale.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BiasSource {
    /// Energy of the synapse between nodes i and j in Ξ(t).
    SynapseEnergy,
    /// Shortest path distance in the graph (capped at max_hops).
    GraphDistance { max_hops: u8 },
    /// Co-activation frequency between nodes.
    Coactivation,
    /// Temporal decay: exp(-t / half_life) where t = age in turns.
    TemporalDecay { half_life: f32 },
    /// Score from GNN layer (0-indexed).
    GnnPairScore { layer: u8 },
    /// Constant value (useful for uniform bias).
    Constant(f32),
    /// Product of two sources: source_a(i,j) * source_b(i,j).
    Product(Box<BiasSource>, Box<BiasSource>),
}

/// Delta sources for WarpQ / WarpK representation warping.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DeltaSource {
    /// Delta from GNN hidden states.
    GnnDelta,
    /// Mean of neighbor embeddings (optionally filtered by edge type).
    NeighborMean {
        edge_type: Option<String>,
        max_neighbors: u8,
    },
    /// Causal chain aggregation with decay per hop.
    CausalChain { depth: u8, decay: f32 },
}

/// Conditions for masking (setting scores to -∞).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MaskCondition {
    /// Mask if graph distance > threshold.
    GraphDistanceAbove(u8),
    /// Mask if synapse energy < threshold.
    EnergyBelow(f32),
    /// Mask if no path exists between nodes.
    NoPath,
}

/// Runtime conditions evaluated before formula compilation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RuntimeCondition {
    /// Context type string match (e.g., "math", "creative").
    ContextType(String),
    /// Token count in current generation within [min, max].
    TokenCount { min: u32, max: u32 },
    /// Graph density above threshold (edges / possible_edges).
    GraphDensity { threshold: f32 },
    /// Model uncertainty (entropy) above threshold.
    Uncertainty { threshold: f32 },
}

/// Types of queries delegated to the Obrain graph (REV.3).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum QueryType {
    /// Direct lookup: MATCH (n {name: $name}) RETURN n.value
    Lookup,
    /// 1-hop neighbors, optionally filtered by edge type.
    Neighbors { edge_type: Option<String> },
    /// Follow CAUSES edges up to max_depth hops.
    CausalTrace { max_depth: u8 },
    /// Evaluate arithmetic expression on graph properties.
    Compute,
    /// Free-form GQL query — model emits ⟨GQL:...⟩ pattern.
    FreeformGQL,
}

/// Range of attention heads (inclusive start, exclusive end).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HeadRange {
    pub start: u8,
    pub end: u8,
}

// ─── Validation ─────────────────────────────────────────────────────────────

/// Maximum allowed nesting depth for AttnOp trees.
pub const MAX_DEPTH: u32 = 8;

/// Weight bounds: all f32 weights are clamped to this range.
pub const WEIGHT_MIN: f32 = -2.0;
pub const WEIGHT_MAX: f32 = 2.0;

/// Error returned when an AttnOp tree exceeds the maximum nesting depth.
#[derive(Debug, Clone, PartialEq)]
pub struct DepthError {
    pub found: u32,
    pub max: u32,
}

impl std::fmt::Display for DepthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "AttnOp depth {} exceeds maximum {}",
            self.found, self.max
        )
    }
}

impl std::error::Error for DepthError {}

/// Validate that an AttnOp tree does not exceed `max_depth` levels of nesting.
///
/// Returns `Ok(())` if valid, or `Err(DepthError)` with the actual depth found.
pub fn validate_depth(op: &AttnOp, max_depth: u32) -> Result<(), DepthError> {
    let depth = compute_depth(op);
    if depth > max_depth {
        Err(DepthError {
            found: depth,
            max: max_depth,
        })
    } else {
        Ok(())
    }
}

/// Compute the nesting depth of an AttnOp tree.
fn compute_depth(op: &AttnOp) -> u32 {
    match op {
        AttnOp::Identity
        | AttnOp::BiasAdd { .. }
        | AttnOp::BiasScale { .. }
        | AttnOp::Mask { .. }
        | AttnOp::WarpQ { .. }
        | AttnOp::WarpK { .. }
        | AttnOp::QueryDelegate { .. } => 1,

        AttnOp::Sequence(ops) => {
            1 + ops.iter().map(compute_depth).max().unwrap_or(0)
        }

        AttnOp::PerHead(pairs) => {
            1 + pairs
                .iter()
                .map(|(_, op)| compute_depth(op))
                .max()
                .unwrap_or(0)
        }

        AttnOp::Conditional {
            then_op, else_op, ..
        } => 1 + compute_depth(then_op).max(compute_depth(else_op)),
    }
}

// ─── Weight clamping ────────────────────────────────────────────────────────

/// Clamp all f32 weights in an AttnOp tree to [`WEIGHT_MIN`, `WEIGHT_MAX`].
///
/// This is a safety measure to prevent runaway attention scores from
/// evolved formulas. Applied after mutation and before compilation.
pub fn clamp_weights(op: &mut AttnOp) {
    fn clamp(v: &mut f32) {
        *v = v.clamp(WEIGHT_MIN, WEIGHT_MAX);
    }

    fn clamp_bias_source(src: &mut BiasSource) {
        match src {
            BiasSource::TemporalDecay { half_life } => clamp(half_life),
            BiasSource::Constant(v) => clamp(v),
            BiasSource::Product(a, b) => {
                clamp_bias_source(a);
                clamp_bias_source(b);
            }
            _ => {}
        }
    }

    fn clamp_delta_source(src: &mut DeltaSource) {
        if let DeltaSource::CausalChain { decay, .. } = src {
            clamp(decay);
        }
    }

    match op {
        AttnOp::BiasAdd { source, weight } | AttnOp::BiasScale { source, weight } => {
            clamp(weight);
            clamp_bias_source(source);
        }

        AttnOp::WarpQ {
            delta_source,
            alpha,
        }
        | AttnOp::WarpK {
            delta_source,
            alpha,
        } => {
            clamp(alpha);
            clamp_delta_source(delta_source);
        }

        AttnOp::QueryDelegate {
            entropy_threshold, ..
        } => {
            // entropy_threshold is always positive, but clamp to [0, 2.0]
            *entropy_threshold = entropy_threshold.clamp(0.0, WEIGHT_MAX);
        }

        AttnOp::Mask { .. } | AttnOp::Identity => {}

        AttnOp::Sequence(ops) => {
            for o in ops.iter_mut() {
                clamp_weights(o);
            }
        }

        AttnOp::PerHead(pairs) => {
            for (_, o) in pairs.iter_mut() {
                clamp_weights(o);
            }
        }

        AttnOp::Conditional {
            then_op, else_op, ..
        } => {
            clamp_weights(then_op);
            clamp_weights(else_op);
        }
    }
}

// ─── Cost estimation ────────────────────────────────────────────────────────

/// Estimate relative compute cost of an AttnOp formula.
///
/// Used by the FormulaSelector to penalize expensive formulas.
/// Costs are relative: Identity=0, BiasAdd=1, WarpQ/K=3, QueryDelegate=5.
pub fn estimated_cost(op: &AttnOp) -> f32 {
    match op {
        AttnOp::Identity => 0.0,

        AttnOp::BiasAdd { source, .. } | AttnOp::BiasScale { source, .. } => {
            1.0 + bias_source_cost(source)
        }

        AttnOp::Mask { .. } => 0.5,

        AttnOp::WarpQ { .. } | AttnOp::WarpK { .. } => 3.0,

        AttnOp::QueryDelegate { .. } => 5.0,

        AttnOp::Sequence(ops) => ops.iter().map(estimated_cost).sum(),

        AttnOp::PerHead(pairs) => {
            pairs
                .iter()
                .map(|(_, o)| estimated_cost(o))
                .fold(0.0_f32, f32::max)
        }

        AttnOp::Conditional {
            then_op, else_op, ..
        } => {
            // Expected cost: max of both branches (conservative)
            0.5 + estimated_cost(then_op).max(estimated_cost(else_op))
        }
    }
}

fn bias_source_cost(source: &BiasSource) -> f32 {
    match source {
        BiasSource::Constant(_) => 0.0,
        BiasSource::SynapseEnergy | BiasSource::Coactivation => 0.1,
        BiasSource::GraphDistance { .. } => 0.5,
        BiasSource::TemporalDecay { .. } => 0.1,
        BiasSource::GnnPairScore { .. } => 1.0,
        BiasSource::Product(a, b) => bias_source_cost(a) + bias_source_cost(b),
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // -- Depth validation --

    #[test]
    fn test_identity_depth() {
        assert_eq!(validate_depth(&AttnOp::Identity, MAX_DEPTH), Ok(()));
        assert_eq!(compute_depth(&AttnOp::Identity), 1);
    }

    #[test]
    fn test_flat_sequence_depth() {
        let op = AttnOp::Sequence(vec![
            AttnOp::Identity,
            AttnOp::BiasAdd {
                source: BiasSource::Constant(0.5),
                weight: 1.0,
            },
        ]);
        assert_eq!(compute_depth(&op), 2);
        assert!(validate_depth(&op, MAX_DEPTH).is_ok());
    }

    #[test]
    fn test_depth_8_ok() {
        // Build a chain of 8 levels: Seq(Seq(Seq(...(Identity)...)))
        let mut op = AttnOp::Identity;
        for _ in 0..7 {
            op = AttnOp::Sequence(vec![op]);
        }
        assert_eq!(compute_depth(&op), 8);
        assert!(validate_depth(&op, 8).is_ok());
    }

    #[test]
    fn test_depth_9_rejected() {
        let mut op = AttnOp::Identity;
        for _ in 0..8 {
            op = AttnOp::Sequence(vec![op]);
        }
        assert_eq!(compute_depth(&op), 9);
        let err = validate_depth(&op, 8).unwrap_err();
        assert_eq!(err.found, 9);
        assert_eq!(err.max, 8);
    }

    #[test]
    fn test_conditional_depth() {
        let op = AttnOp::Conditional {
            condition: RuntimeCondition::Uncertainty { threshold: 0.5 },
            then_op: Box::new(AttnOp::Sequence(vec![AttnOp::Identity])),
            else_op: Box::new(AttnOp::Identity),
        };
        // Conditional(1) + Sequence(1) + Identity(1) = 3
        assert_eq!(compute_depth(&op), 3);
    }

    #[test]
    fn test_perhead_depth() {
        let op = AttnOp::PerHead(vec![
            (
                HeadRange { start: 0, end: 16 },
                AttnOp::BiasAdd {
                    source: BiasSource::SynapseEnergy,
                    weight: 0.5,
                },
            ),
            (
                HeadRange {
                    start: 16,
                    end: 32,
                },
                AttnOp::Sequence(vec![AttnOp::Identity, AttnOp::Identity]),
            ),
        ]);
        // PerHead(1) + max(BiasAdd=1, Seq(1)+Identity(1)=2) = 3
        assert_eq!(compute_depth(&op), 3);
    }

    // -- Weight clamping --

    #[test]
    fn test_clamp_bias_add() {
        let mut op = AttnOp::BiasAdd {
            source: BiasSource::Constant(5.0),
            weight: 5.0,
        };
        clamp_weights(&mut op);
        match &op {
            AttnOp::BiasAdd { source, weight } => {
                assert_eq!(*weight, 2.0);
                match source {
                    BiasSource::Constant(v) => assert_eq!(*v, 2.0),
                    _ => panic!("expected Constant"),
                }
            }
            _ => panic!("expected BiasAdd"),
        }
    }

    #[test]
    fn test_clamp_negative() {
        let mut op = AttnOp::WarpQ {
            delta_source: DeltaSource::CausalChain {
                depth: 3,
                decay: -5.0,
            },
            alpha: -3.0,
        };
        clamp_weights(&mut op);
        match &op {
            AttnOp::WarpQ {
                delta_source,
                alpha,
            } => {
                assert_eq!(*alpha, -2.0);
                match delta_source {
                    DeltaSource::CausalChain { decay, .. } => assert_eq!(*decay, -2.0),
                    _ => panic!("expected CausalChain"),
                }
            }
            _ => panic!("expected WarpQ"),
        }
    }

    #[test]
    fn test_clamp_recursive_sequence() {
        let mut op = AttnOp::Sequence(vec![
            AttnOp::BiasAdd {
                source: BiasSource::Constant(10.0),
                weight: -10.0,
            },
            AttnOp::PerHead(vec![(
                HeadRange { start: 0, end: 8 },
                AttnOp::BiasScale {
                    source: BiasSource::TemporalDecay { half_life: 99.0 },
                    weight: 0.5,
                },
            )]),
        ]);
        clamp_weights(&mut op);

        if let AttnOp::Sequence(ops) = &op {
            // First: BiasAdd weight clamped
            if let AttnOp::BiasAdd { weight, source } = &ops[0] {
                assert_eq!(*weight, -2.0);
                if let BiasSource::Constant(v) = source {
                    assert_eq!(*v, 2.0);
                }
            }
            // Second: PerHead > BiasScale > TemporalDecay half_life clamped
            if let AttnOp::PerHead(pairs) = &ops[1] {
                if let AttnOp::BiasScale { source, .. } = &pairs[0].1 {
                    if let BiasSource::TemporalDecay { half_life } = source {
                        assert_eq!(*half_life, 2.0);
                    }
                }
            }
        }
    }

    #[test]
    fn test_clamp_query_delegate() {
        let mut op = AttnOp::QueryDelegate {
            entropy_threshold: -1.0,
            query_type: QueryType::Lookup,
            max_inject_tokens: 32,
        };
        clamp_weights(&mut op);
        if let AttnOp::QueryDelegate {
            entropy_threshold, ..
        } = &op
        {
            assert_eq!(*entropy_threshold, 0.0);
        }
    }

    // -- Serde round-trip --

    #[test]
    fn test_serde_identity() {
        let op = AttnOp::Identity;
        let json = serde_json::to_string(&op).unwrap();
        let back: AttnOp = serde_json::from_str(&json).unwrap();
        assert_eq!(op, back);
    }

    #[test]
    fn test_serde_bias_add() {
        let op = AttnOp::BiasAdd {
            source: BiasSource::Product(
                Box::new(BiasSource::SynapseEnergy),
                Box::new(BiasSource::TemporalDecay { half_life: 1.5 }),
            ),
            weight: -0.3,
        };
        let json = serde_json::to_string_pretty(&op).unwrap();
        let back: AttnOp = serde_json::from_str(&json).unwrap();
        assert_eq!(op, back);
    }

    #[test]
    fn test_serde_complex_formula() {
        let formula = AttnOp::Conditional {
            condition: RuntimeCondition::Uncertainty { threshold: 0.7 },
            then_op: Box::new(AttnOp::PerHead(vec![
                (
                    HeadRange { start: 0, end: 16 },
                    AttnOp::Sequence(vec![
                        AttnOp::BiasAdd {
                            source: BiasSource::GnnPairScore { layer: 2 },
                            weight: 1.0,
                        },
                        AttnOp::Mask {
                            condition: MaskCondition::GraphDistanceAbove(3),
                        },
                    ]),
                ),
                (
                    HeadRange {
                        start: 16,
                        end: 32,
                    },
                    AttnOp::WarpQ {
                        delta_source: DeltaSource::GnnDelta,
                        alpha: 0.5,
                    },
                ),
            ])),
            else_op: Box::new(AttnOp::Identity),
        };
        let json = serde_json::to_string(&formula).unwrap();
        let back: AttnOp = serde_json::from_str(&json).unwrap();
        assert_eq!(formula, back);
    }

    #[test]
    fn test_serde_query_delegate() {
        let op = AttnOp::QueryDelegate {
            entropy_threshold: 1.5,
            query_type: QueryType::CausalTrace { max_depth: 4 },
            max_inject_tokens: 64,
        };
        let json = serde_json::to_string(&op).unwrap();
        let back: AttnOp = serde_json::from_str(&json).unwrap();
        assert_eq!(op, back);
    }

    #[test]
    fn test_serde_all_query_types() {
        let types = vec![
            QueryType::Lookup,
            QueryType::Neighbors {
                edge_type: Some("CAUSES".to_string()),
            },
            QueryType::Neighbors { edge_type: None },
            QueryType::CausalTrace { max_depth: 5 },
            QueryType::Compute,
            QueryType::FreeformGQL,
        ];
        for qt in types {
            let json = serde_json::to_string(&qt).unwrap();
            let back: QueryType = serde_json::from_str(&json).unwrap();
            assert_eq!(qt, back);
        }
    }

    #[test]
    fn test_serde_all_mask_conditions() {
        let conds = vec![
            MaskCondition::GraphDistanceAbove(5),
            MaskCondition::EnergyBelow(0.1),
            MaskCondition::NoPath,
        ];
        for c in conds {
            let json = serde_json::to_string(&c).unwrap();
            let back: MaskCondition = serde_json::from_str(&json).unwrap();
            assert_eq!(c, back);
        }
    }

    // -- Cost estimation --

    #[test]
    fn test_cost_identity() {
        assert_eq!(estimated_cost(&AttnOp::Identity), 0.0);
    }

    #[test]
    fn test_cost_bias_add() {
        let op = AttnOp::BiasAdd {
            source: BiasSource::Constant(1.0),
            weight: 0.5,
        };
        assert_eq!(estimated_cost(&op), 1.0); // 1.0 + 0.0 (Constant)
    }

    #[test]
    fn test_cost_warp() {
        assert_eq!(
            estimated_cost(&AttnOp::WarpQ {
                delta_source: DeltaSource::GnnDelta,
                alpha: 0.5,
            }),
            3.0
        );
    }

    #[test]
    fn test_cost_query_delegate() {
        assert_eq!(
            estimated_cost(&AttnOp::QueryDelegate {
                entropy_threshold: 1.0,
                query_type: QueryType::Lookup,
                max_inject_tokens: 32,
            }),
            5.0
        );
    }

    #[test]
    fn test_cost_sequence_sums() {
        let op = AttnOp::Sequence(vec![
            AttnOp::BiasAdd {
                source: BiasSource::Constant(1.0),
                weight: 0.5,
            }, // 1.0
            AttnOp::Mask {
                condition: MaskCondition::NoPath,
            }, // 0.5
            AttnOp::WarpQ {
                delta_source: DeltaSource::GnnDelta,
                alpha: 0.1,
            }, // 3.0
        ]);
        assert_eq!(estimated_cost(&op), 4.5);
    }

    #[test]
    fn test_cost_perhead_max() {
        let op = AttnOp::PerHead(vec![
            (
                HeadRange { start: 0, end: 16 },
                AttnOp::Identity,
            ), // 0.0
            (
                HeadRange {
                    start: 16,
                    end: 32,
                },
                AttnOp::WarpK {
                    delta_source: DeltaSource::GnnDelta,
                    alpha: 0.5,
                },
            ), // 3.0
        ]);
        assert_eq!(estimated_cost(&op), 3.0);
    }

    #[test]
    fn test_cost_gnn_pair_score() {
        let op = AttnOp::BiasAdd {
            source: BiasSource::GnnPairScore { layer: 0 },
            weight: 0.5,
        };
        assert_eq!(estimated_cost(&op), 2.0); // 1.0 + 1.0 (GnnPairScore)
    }

    #[test]
    fn test_cost_product_source() {
        let op = AttnOp::BiasAdd {
            source: BiasSource::Product(
                Box::new(BiasSource::SynapseEnergy),  // 0.1
                Box::new(BiasSource::GraphDistance { max_hops: 3 }), // 0.5
            ),
            weight: 0.5,
        };
        assert_eq!(estimated_cost(&op), 1.6); // 1.0 + 0.1 + 0.5
    }

    // -- Seed formula JSON validation --

    #[test]
    fn test_seed_formula_json_roundtrip() {
        // These JSON strings are used by persona::formulas::seed_formulas()
        // They MUST deserialize to valid AttnOp and round-trip correctly.
        let seeds: Vec<(&str, &str)> = vec![
            ("F0-Identity", r#""Identity""#),
            ("F1-GravityLinear", r#"{"BiasAdd":{"source":{"GraphDistance":{"max_hops":4}},"weight":1.0}}"#),
            ("F2-RepulsionCompeting", r#"{"Sequence":[{"Mask":{"condition":{"GraphDistanceAbove":3}}},{"BiasAdd":{"source":"SynapseEnergy","weight":-0.5}}]}"#),
            ("F3-WarpGNN", r#"{"WarpQ":{"delta_source":"GnnDelta","alpha":0.5}}"#),
            ("F4-PerHeadTopo", r#"{"PerHead":[[[0,16],"Identity"],[[16,32],{"Mask":{"condition":{"GraphDistanceAbove":2}}}]]}"#),
            ("F5-QueryDelegateCompute", r#"{"QueryDelegate":{"entropy_threshold":1.5,"query_type":"Compute","max_inject_tokens":32}}"#),
        ];

        for (name, json) in &seeds {
            let op: AttnOp = serde_json::from_str(json)
                .unwrap_or_else(|e| panic!("{name}: failed to deserialize: {e}"));

            // Validate depth
            validate_depth(&op, MAX_DEPTH)
                .unwrap_or_else(|e| panic!("{name}: depth validation failed: {e}"));

            // Round-trip: serialize back and re-deserialize
            let json2 = serde_json::to_string(&op).unwrap();
            let op2: AttnOp = serde_json::from_str(&json2)
                .unwrap_or_else(|e| panic!("{name}: round-trip failed: {e}"));
            assert_eq!(op, op2, "{name}: round-trip mismatch");
        }
    }
}
