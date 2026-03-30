//! Compiler: AttnOp DSL → kq_mask overlay / kq_b tensors for llama.cpp.
//!
//! The compiler takes a formula ([`AttnOp`]) and a [`CompileContext`] describing
//! the current graph state, and produces a [`CompiledFormula`] with float tensors
//! ready to be applied on top of the existing attention mask.
//!
//! # Compilation targets (from RFC §2.4)
//!
//! | AttnOp           | Target      | FA-compatible |
//! |------------------|-------------|:-------------:|
//! | Mask             | kq_mask     | ✅            |
//! | BiasAdd (≤0)     | kq_mask     | ✅            |
//! | BiasAdd (>0)     | kq_b        | ❌            |
//! | BiasScale        | kq_mask/kq_b| depends       |
//! | WarpQ/WarpK      | kq_b        | ❌            |
//! | PerHead          | per-head    | ✅            |
//! | QueryDelegate    | token inject| ✅            |
//! | Conditional      | Rust-side   | ��            |
//! | Identity         | no-op       | ✅            |

use crate::attn_dsl::*;

// ─── Compile context ───────────────��────────────────────────────────────────

/// Runtime context needed to compile an AttnOp into tensors.
///
/// All matrices are flat row-major `[n_nodes × n_nodes]` where `n_nodes`
/// is the number of graph nodes in the current KV cache (excluding header).
#[derive(Debug, Clone)]
pub struct CompileContext {
    /// Number of graph nodes (positions in the sparse mask, excluding header).
    pub n_nodes: usize,
    /// Number of attention heads.
    pub n_head: usize,
    /// Total sparse positions (header + nodes + query/gen).
    pub n_pos: usize,
    /// Number of header token positions (always visible, not formula-controlled).
    pub header_tokens: usize,

    // ── Graph data (flat row-major [n_nodes × n_nodes]) ──

    /// Shortest-path distance between nodes. `u8::MAX` = no path.
    pub graph_distances: Vec<u8>,
    /// Synapse energy Ξ(t) between node pairs. 0.0 = no synapse.
    pub synapse_energies: Vec<f32>,
    /// Co-activation frequency between node pairs.
    pub coactivations: Vec<f32>,
    /// Age of each node in conversation turns (for temporal decay).
    pub node_ages: Vec<f32>,

    // ── Runtime signals ──

    /// Current generation entropy (for RuntimeCondition::Uncertainty).
    pub current_entropy: f32,
    /// Current token count in this generation.
    pub current_token_count: u32,
    /// Detected context type (e.g., "math", "creative").
    pub context_type: String,
    /// Graph density: edges / possible_edges.
    pub graph_density: f32,
}

impl CompileContext {
    /// Index into a flat `[n_nodes × n_nodes]` matrix.
    #[inline]
    fn idx(&self, row: usize, col: usize) -> usize {
        row * self.n_nodes + col
    }

    /// Get graph distance between two nodes, defaulting to `u8::MAX` if out of bounds.
    fn distance(&self, i: usize, j: usize) -> u8 {
        let idx = self.idx(i, j);
        self.graph_distances.get(idx).copied().unwrap_or(u8::MAX)
    }

    /// Get synapse energy between two nodes.
    fn energy(&self, i: usize, j: usize) -> f32 {
        let idx = self.idx(i, j);
        self.synapse_energies.get(idx).copied().unwrap_or(0.0)
    }

    /// Get co-activation frequency between two nodes.
    fn coactivation(&self, i: usize, j: usize) -> f32 {
        let idx = self.idx(i, j);
        self.coactivations.get(idx).copied().unwrap_or(0.0)
    }

    /// Get age of a node.
    fn age(&self, i: usize) -> f32 {
        self.node_ages.get(i).copied().unwrap_or(0.0)
    }
}

// ─── Compiled output ──────────────────��─────────────────────────────────────

/// Result of compiling an AttnOp formula.
///
/// Contains optional mask overlay, bias tensor, and/or tokens to inject.
/// These are applied **on top of** the existing per-head mask from `build_perhead_mask`.
#[derive(Debug, Clone)]
pub struct CompiledFormula {
    /// Mask overlay: values to ADD to the existing kq_mask.
    /// - `f32::NEG_INFINITY` = force block
    /// - Negative values = attenuation (FA-compatible)
    /// - `0.0` = no change
    ///
    /// Shape: `[n_head × n_nodes × n_nodes]` if `per_head`, else `[n_nodes × n_nodes]`.
    /// Only covers the node×node region; header and query positions are untouched.
    pub kq_mask_overlay: Option<Vec<f32>>,

    /// Bias tensor for kq_b (requires Phase 2 fork, NOT FA-compatible).
    /// Positive values amplify attention. Same shape rules as `kq_mask_overlay`.
    pub kq_b: Option<Vec<f32>>,

    /// Tokens to inject into the next batch (for QueryDelegate).
    /// These are token IDs to be added to the context before generation.
    pub inject_tokens: Option<Vec<u32>>,

    /// Whether the tensors have a per-head dimension.
    pub per_head: bool,
}

impl CompiledFormula {
    /// Empty formula (identity / no-op).
    pub fn identity() -> Self {
        Self {
            kq_mask_overlay: None,
            kq_b: None,
            inject_tokens: None,
            per_head: false,
        }
    }

    /// Check if this formula has any effect.
    pub fn is_noop(&self) -> bool {
        self.kq_mask_overlay.is_none() && self.kq_b.is_none() && self.inject_tokens.is_none()
    }
}

// ─── Compiler ──────────────────────────────────────────────��────────────────

/// Compile error.
#[derive(Debug, Clone)]
pub enum CompileError {
    /// The formula exceeds maximum depth.
    DepthExceeded(DepthError),
    /// A WarpQ/WarpK requires GNN data not available in context.
    MissingGnnData,
    /// PerHead range out of bounds for model's n_head.
    HeadRangeOutOfBounds { range: HeadRange, n_head: usize },
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DepthExceeded(e) => write!(f, "depth exceeded: {e}"),
            Self::MissingGnnData => write!(f, "WarpQ/WarpK requires GNN data not available"),
            Self::HeadRangeOutOfBounds { range, n_head } => {
                write!(
                    f,
                    "head range [{}, {}) out of bounds for n_head={}",
                    range.start, range.end, n_head
                )
            }
        }
    }
}

impl std::error::Error for CompileError {}

/// Compile an AttnOp into tensors ready for llama.cpp.
///
/// Validates depth first, then dispatches to the appropriate sub-compiler.
pub fn compile(op: &AttnOp, ctx: &CompileContext) -> Result<CompiledFormula, CompileError> {
    // Safety: validate depth before compilation
    validate_depth(op, MAX_DEPTH).map_err(CompileError::DepthExceeded)?;

    compile_inner(op, ctx)
}

fn compile_inner(op: &AttnOp, ctx: &CompileContext) -> Result<CompiledFormula, CompileError> {
    match op {
        AttnOp::Identity => Ok(CompiledFormula::identity()),

        AttnOp::Mask { condition } => compile_mask(condition, ctx),

        AttnOp::BiasAdd { source, weight } => compile_bias_add(source, *weight, ctx),

        AttnOp::BiasScale { source, weight } => compile_bias_scale(source, *weight, ctx),

        AttnOp::WarpQ { .. } | AttnOp::WarpK { .. } => compile_warp(op, ctx),

        AttnOp::QueryDelegate {
            entropy_threshold,
            query_type,
            max_inject_tokens,
        } => compile_query_delegate(*entropy_threshold, query_type, *max_inject_tokens, ctx),

        AttnOp::Sequence(ops) => compile_sequence(ops, ctx),

        AttnOp::PerHead(pairs) => compile_per_head(pairs, ctx),

        AttnOp::Conditional {
            condition,
            then_op,
            else_op,
        } => compile_conditional(condition, then_op, else_op, ctx),
    }
}

// ─── Sub-compilers ─────��─────────���──────────────────────────────────────────

/// Mask: score[i][j] = -∞ if condition(i, j).
fn compile_mask(
    condition: &MaskCondition,
    ctx: &CompileContext,
) -> Result<CompiledFormula, CompileError> {
    let n = ctx.n_nodes;
    let mut overlay = vec![0.0f32; n * n];

    for i in 0..n {
        for j in 0..n {
            let should_mask = match condition {
                MaskCondition::GraphDistanceAbove(max_hops) => {
                    ctx.distance(i, j) > *max_hops
                }
                MaskCondition::EnergyBelow(threshold) => {
                    ctx.energy(i, j) < *threshold
                }
                MaskCondition::NoPath => {
                    ctx.distance(i, j) == u8::MAX
                }
            };
            if should_mask {
                overlay[i * n + j] = f32::NEG_INFINITY;
            }
        }
    }

    Ok(CompiledFormula {
        kq_mask_overlay: Some(overlay),
        kq_b: None,
        inject_tokens: None,
        per_head: false,
    })
}

/// BiasAdd: score[i][j] += weight * source(i, j).
///
/// If weight ≤ 0 → kq_mask (FA-compatible attenuation).
/// If weight > 0 → kq_b (requires Phase 2 fork).
fn compile_bias_add(
    source: &BiasSource,
    weight: f32,
    ctx: &CompileContext,
) -> Result<CompiledFormula, CompileError> {
    let n = ctx.n_nodes;
    let mut values = vec![0.0f32; n * n];

    for i in 0..n {
        for j in 0..n {
            values[i * n + j] = weight * eval_bias_source(source, i, j, ctx);
        }
    }

    if weight <= 0.0 {
        // Attenuation → kq_mask overlay (FA-compatible)
        Ok(CompiledFormula {
            kq_mask_overlay: Some(values),
            kq_b: None,
            inject_tokens: None,
            per_head: false,
        })
    } else {
        // Amplification → kq_b (requires Phase 2 fork)
        Ok(CompiledFormula {
            kq_mask_overlay: None,
            kq_b: Some(values),
            inject_tokens: None,
            per_head: false,
        })
    }
}

/// BiasScale: score[i][j] += weight * ln(source(i, j)).
/// Compiled as BiasAdd with ln(source).
fn compile_bias_scale(
    source: &BiasSource,
    weight: f32,
    ctx: &CompileContext,
) -> Result<CompiledFormula, CompileError> {
    let n = ctx.n_nodes;
    let mut values = vec![0.0f32; n * n];

    for i in 0..n {
        for j in 0..n {
            let raw = eval_bias_source(source, i, j, ctx);
            // ln(0) = -inf, ln(negative) = NaN → clamp to small positive
            let log_val = if raw > 1e-10 { raw.ln() } else { -23.0 }; // ln(1e-10) ≈ -23
            values[i * n + j] = weight * log_val;
        }
    }

    // BiasScale with positive weight can produce negative values (ln < 0) ��� kq_mask
    // BiasScale with negative weight can produce positive values → kq_b
    let all_non_positive = values.iter().all(|&v| v <= 0.0);

    if all_non_positive {
        Ok(CompiledFormula {
            kq_mask_overlay: Some(values),
            kq_b: None,
            inject_tokens: None,
            per_head: false,
        })
    } else {
        Ok(CompiledFormula {
            kq_mask_overlay: None,
            kq_b: Some(values),
            inject_tokens: None,
            per_head: false,
        })
    }
}

/// WarpQ/WarpK: compiled as kq_b via dot product decomposition.
///
/// dot(Q + ΔQ, K) = dot(Q, K) + dot(ΔQ, K)
/// The second term becomes a bias: kq_b[i][j] = alpha * delta_signal(i, j).
///
/// NOTE: Full implementation requires GNN hidden states. For now, we produce
/// a placeholder kq_b based on graph structure. Real GNN integration in T0.4.
fn compile_warp(
    _op: &AttnOp,
    ctx: &CompileContext,
) -> Result<CompiledFormula, CompileError> {
    // Placeholder: produce a kq_b proportional to synapse energy.
    // Real implementation will use GNN deltas once FormulaSelector (T0.4) is ready.
    let n = ctx.n_nodes;
    let values = vec![0.0f32; n * n];

    // TODO(T0.4): Implement real dot product decomposition with GNN deltas.
    // For now, WarpQ/WarpK compile to zero-bias (no effect) until GNN is wired.

    Ok(CompiledFormula {
        kq_mask_overlay: None,
        kq_b: Some(values),
        inject_tokens: None,
        per_head: false,
    })
}

/// QueryDelegate: if current entropy > threshold, signal token injection.
///
/// Returns `inject_tokens = Some(vec![])` as a signal that the caller should
/// execute the query. Actual token IDs are filled by the query executor at
/// runtime (not at compile time).
fn compile_query_delegate(
    entropy_threshold: f32,
    _query_type: &QueryType,
    max_inject_tokens: u32,
    ctx: &CompileContext,
) -> Result<CompiledFormula, CompileError> {
    if ctx.current_entropy > entropy_threshold {
        // Signal: delegation should happen. The caller fills in actual tokens.
        Ok(CompiledFormula {
            kq_mask_overlay: None,
            kq_b: None,
            inject_tokens: Some(Vec::with_capacity(max_inject_tokens as usize)),
            per_head: false,
        })
    } else {
        // Entropy below threshold → no delegation, identity
        Ok(CompiledFormula::identity())
    }
}

/// Sequence: apply all ops left-to-right, merging their outputs.
fn compile_sequence(
    ops: &[AttnOp],
    ctx: &CompileContext,
) -> Result<CompiledFormula, CompileError> {
    let mut merged = CompiledFormula::identity();

    for op in ops {
        let compiled = compile_inner(op, ctx)?;
        merged = merge_formulas(merged, compiled, ctx);
    }

    Ok(merged)
}

/// PerHead: different ops for different head ranges.
/// Produces per-head tensors: `[n_head × n_nodes × n_nodes]`.
fn compile_per_head(
    pairs: &[(HeadRange, AttnOp)],
    ctx: &CompileContext,
) -> Result<CompiledFormula, CompileError> {
    let n = ctx.n_nodes;
    let n_h = ctx.n_head;

    // Validate ranges
    for (range, _) in pairs {
        if range.end as usize > n_h {
            return Err(CompileError::HeadRangeOutOfBounds {
                range: range.clone(),
                n_head: n_h,
            });
        }
    }

    // Initialize per-head tensors with zeros (no effect)
    let mut mask_overlay = vec![0.0f32; n_h * n * n];
    let mut bias = vec![0.0f32; n_h * n * n];
    let mut has_mask = false;
    let mut has_bias = false;
    let mut inject = None;

    for (range, op) in pairs {
        let compiled = compile_inner(op, ctx)?;

        // Apply this sub-formula to heads in [start, end)
        for h in range.start as usize..range.end as usize {
            let head_offset = h * n * n;

            if let Some(ref overlay) = compiled.kq_mask_overlay {
                has_mask = true;
                for idx in 0..(n * n) {
                    mask_overlay[head_offset + idx] += overlay[idx % overlay.len()];
                }
            }

            if let Some(ref b) = compiled.kq_b {
                has_bias = true;
                for idx in 0..(n * n) {
                    bias[head_offset + idx] += b[idx % b.len()];
                }
            }
        }

        // Collect inject tokens from any sub-op
        if compiled.inject_tokens.is_some() {
            inject = compiled.inject_tokens;
        }
    }

    Ok(CompiledFormula {
        kq_mask_overlay: if has_mask { Some(mask_overlay) } else { None },
        kq_b: if has_bias { Some(bias) } else { None },
        inject_tokens: inject,
        per_head: true,
    })
}

/// Conditional: evaluate condition, then compile the matching branch.
fn compile_conditional(
    condition: &RuntimeCondition,
    then_op: &AttnOp,
    else_op: &AttnOp,
    ctx: &CompileContext,
) -> Result<CompiledFormula, CompileError> {
    let cond_met = eval_runtime_condition(condition, ctx);

    if cond_met {
        compile_inner(then_op, ctx)
    } else {
        compile_inner(else_op, ctx)
    }
}

// ─── Helpers ───────────────────��───────────────────────���────────────────────

/// Evaluate a BiasSource for a node pair (i, j).
fn eval_bias_source(source: &BiasSource, i: usize, j: usize, ctx: &CompileContext) -> f32 {
    match source {
        BiasSource::SynapseEnergy => ctx.energy(i, j),

        BiasSource::GraphDistance { max_hops } => {
            let d = ctx.distance(i, j);
            if d == u8::MAX {
                0.0
            } else {
                // Normalize: 1.0 at distance 0, 0.0 at max_hops
                1.0 - (d as f32 / *max_hops as f32).min(1.0)
            }
        }

        BiasSource::Coactivation => ctx.coactivation(i, j),

        BiasSource::TemporalDecay { half_life } => {
            // Use the max age of the two nodes
            let age = ctx.age(i).max(ctx.age(j));
            if *half_life <= 0.0 {
                0.0
            } else {
                (-age / half_life).exp()
            }
        }

        BiasSource::GnnPairScore { .. } => {
            // TODO(T0.4): Wire to actual GNN pair scoring.
            // Placeholder: use synapse energy as proxy.
            ctx.energy(i, j)
        }

        BiasSource::Constant(v) => *v,

        BiasSource::Product(a, b) => {
            eval_bias_source(a, i, j, ctx) * eval_bias_source(b, i, j, ctx)
        }
    }
}

/// Evaluate a RuntimeCondition.
fn eval_runtime_condition(condition: &RuntimeCondition, ctx: &CompileContext) -> bool {
    match condition {
        RuntimeCondition::ContextType(expected) => ctx.context_type == *expected,
        RuntimeCondition::TokenCount { min, max } => {
            ctx.current_token_count >= *min && ctx.current_token_count <= *max
        }
        RuntimeCondition::GraphDensity { threshold } => ctx.graph_density > *threshold,
        RuntimeCondition::Uncertainty { threshold } => ctx.current_entropy > *threshold,
    }
}

/// Merge two compiled formulas by element-wise addition.
fn merge_formulas(
    a: CompiledFormula,
    b: CompiledFormula,
    _ctx: &CompileContext,
) -> CompiledFormula {
    let kq_mask_overlay = match (a.kq_mask_overlay, b.kq_mask_overlay) {
        (None, None) => None,
        (Some(v), None) | (None, Some(v)) => Some(v),
        (Some(va), Some(vb)) => {
            let merged: Vec<f32> = va
                .iter()
                .zip(vb.iter())
                .map(|(&x, &y)| {
                    // If either is -inf, result is -inf (blocking wins)
                    if x == f32::NEG_INFINITY || y == f32::NEG_INFINITY {
                        f32::NEG_INFINITY
                    } else {
                        x + y
                    }
                })
                .collect();
            Some(merged)
        }
    };

    let kq_b = match (a.kq_b, b.kq_b) {
        (None, None) => None,
        (Some(v), None) | (None, Some(v)) => Some(v),
        (Some(va), Some(vb)) => {
            Some(va.iter().zip(vb.iter()).map(|(&x, &y)| x + y).collect())
        }
    };

    let inject_tokens = match (a.inject_tokens, b.inject_tokens) {
        (None, None) => None,
        (Some(v), None) | (None, Some(v)) => Some(v),
        (Some(mut va), Some(vb)) => {
            va.extend(vb);
            Some(va)
        }
    };

    CompiledFormula {
        kq_mask_overlay,
        kq_b,
        inject_tokens,
        per_head: a.per_head || b.per_head,
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a simple test context with 4 nodes.
    fn test_ctx() -> CompileContext {
        let n = 4;
        // Distance matrix: diagonal=0, adjacent=1, far=3, no path=MAX
        #[rustfmt::skip]
        let distances: Vec<u8> = vec![
            0, 1, 2, 3,
            1, 0, 1, 2,
            2, 1, 0, 1,
            3, 2, 1, 0,
        ];
        // Energy matrix: higher for closer nodes
        #[rustfmt::skip]
        let energies: Vec<f32> = vec![
            1.0, 0.8, 0.3, 0.1,
            0.8, 1.0, 0.7, 0.2,
            0.3, 0.7, 1.0, 0.6,
            0.1, 0.2, 0.6, 1.0,
        ];
        #[rustfmt::skip]
        let coactivations: Vec<f32> = vec![
            1.0, 0.5, 0.2, 0.0,
            0.5, 1.0, 0.4, 0.1,
            0.2, 0.4, 1.0, 0.3,
            0.0, 0.1, 0.3, 1.0,
        ];

        CompileContext {
            n_nodes: n,
            n_head: 8,
            n_pos: 20,
            header_tokens: 4,
            graph_distances: distances,
            synapse_energies: energies,
            coactivations,
            node_ages: vec![0.0, 1.0, 3.0, 10.0],
            current_entropy: 1.5,
            current_token_count: 50,
            context_type: "math".to_string(),
            graph_density: 0.6,
        }
    }

    // ── Identity ──

    #[test]
    fn test_compile_identity() {
        let ctx = test_ctx();
        let result = compile(&AttnOp::Identity, &ctx).unwrap();
        assert!(result.is_noop());
    }

    // ── Mask ──

    #[test]
    fn test_compile_mask_distance_above() {
        let ctx = test_ctx();
        let result = compile(
            &AttnOp::Mask {
                condition: MaskCondition::GraphDistanceAbove(2),
            },
            &ctx,
        )
        .unwrap();

        let overlay = result.kq_mask_overlay.unwrap();
        assert_eq!(overlay.len(), 16); // 4×4

        // distance(0,3)=3 > 2 → should be -inf
        assert_eq!(overlay[0 * 4 + 3], f32::NEG_INFINITY);
        // distance(0,1)=1 ≤ 2 ��� should be 0
        assert_eq!(overlay[0 * 4 + 1], 0.0);
        // distance(0,2)=2 ≤ 2 → should be 0
        assert_eq!(overlay[0 * 4 + 2], 0.0);
        // Symmetric: distance(3,0)=3 > 2 → -inf
        assert_eq!(overlay[3 * 4 + 0], f32::NEG_INFINITY);
    }

    #[test]
    fn test_compile_mask_energy_below() {
        let ctx = test_ctx();
        let result = compile(
            &AttnOp::Mask {
                condition: MaskCondition::EnergyBelow(0.5),
            },
            &ctx,
        )
        .unwrap();

        let overlay = result.kq_mask_overlay.unwrap();
        // energy(0,2)=0.3 < 0.5 → masked
        assert_eq!(overlay[0 * 4 + 2], f32::NEG_INFINITY);
        // energy(0,1)=0.8 ≥ 0.5 → not masked
        assert_eq!(overlay[0 * 4 + 1], 0.0);
    }

    #[test]
    fn test_compile_mask_no_path() {
        // Override distances to have a u8::MAX entry
        let mut ctx = test_ctx();
        ctx.graph_distances[0 * 4 + 3] = u8::MAX;
        ctx.graph_distances[3 * 4 + 0] = u8::MAX;

        let result = compile(
            &AttnOp::Mask {
                condition: MaskCondition::NoPath,
            },
            &ctx,
        )
        .unwrap();

        let overlay = result.kq_mask_overlay.unwrap();
        assert_eq!(overlay[0 * 4 + 3], f32::NEG_INFINITY);
        assert_eq!(overlay[3 * 4 + 0], f32::NEG_INFINITY);
        assert_eq!(overlay[0 * 4 + 1], 0.0); // has path
    }

    // ── BiasAdd ──

    #[test]
    fn test_compile_bias_add_negative_goes_to_mask() {
        let ctx = test_ctx();
        let result = compile(
            &AttnOp::BiasAdd {
                source: BiasSource::SynapseEnergy,
                weight: -0.5,
            },
            &ctx,
        )
        .unwrap();

        // Negative weight → kq_mask overlay
        assert!(result.kq_mask_overlay.is_some());
        assert!(result.kq_b.is_none());

        let overlay = result.kq_mask_overlay.unwrap();
        // value = -0.5 * energy(0,1) = -0.5 * 0.8 = -0.4
        assert!((overlay[0 * 4 + 1] - (-0.4)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_bias_add_positive_goes_to_kq_b() {
        let ctx = test_ctx();
        let result = compile(
            &AttnOp::BiasAdd {
                source: BiasSource::SynapseEnergy,
                weight: 0.5,
            },
            &ctx,
        )
        .unwrap();

        // Positive weight → kq_b
        assert!(result.kq_mask_overlay.is_none());
        assert!(result.kq_b.is_some());

        let bias = result.kq_b.unwrap();
        // value = 0.5 * energy(0,1) = 0.5 * 0.8 = 0.4
        assert!((bias[0 * 4 + 1] - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_compile_bias_add_constant() {
        let ctx = test_ctx();
        let result = compile(
            &AttnOp::BiasAdd {
                source: BiasSource::Constant(1.0),
                weight: 0.3,
            },
            &ctx,
        )
        .unwrap();

        let bias = result.kq_b.unwrap();
        // All values should be 0.3 * 1.0 = 0.3
        for v in &bias {
            assert!((*v - 0.3).abs() < 1e-6);
        }
    }

    #[test]
    fn test_compile_bias_add_graph_distance() {
        let ctx = test_ctx();
        let result = compile(
            &AttnOp::BiasAdd {
                source: BiasSource::GraphDistance { max_hops: 4 },
                weight: 1.0,
            },
            &ctx,
        )
        .unwrap();

        let bias = result.kq_b.unwrap();
        // distance(0,0)=0 → bias = 1.0 * (1 - 0/4) = 1.0
        assert!((bias[0 * 4 + 0] - 1.0).abs() < 1e-6);
        // distance(0,1)=1 �� bias = 1.0 * (1 - 1/4) = 0.75
        assert!((bias[0 * 4 + 1] - 0.75).abs() < 1e-6);
        // distance(0,3)=3 → bias = 1.0 * (1 - 3/4) = 0.25
        assert!((bias[0 * 4 + 3] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_compile_bias_add_temporal_decay() {
        let ctx = test_ctx();
        let result = compile(
            &AttnOp::BiasAdd {
                source: BiasSource::TemporalDecay { half_life: 1.0 },
                weight: 1.0,
            },
            &ctx,
        )
        .unwrap();

        let bias = result.kq_b.unwrap();
        // age(0)=0, age(1)=1 → max_age = 1.0 → exp(-1/1) ≈ 0.368
        assert!((bias[0 * 4 + 1] - (-1.0f32).exp()).abs() < 1e-3);
        // age(0)=0, age(3)=10 → max_age = 10.0 → exp(-10/1) ≈ 0.0000454
        assert!(bias[0 * 4 + 3] < 0.001);
    }

    #[test]
    fn test_compile_bias_add_product() {
        let ctx = test_ctx();
        let result = compile(
            &AttnOp::BiasAdd {
                source: BiasSource::Product(
                    Box::new(BiasSource::SynapseEnergy),
                    Box::new(BiasSource::Coactivation),
                ),
                weight: 1.0,
            },
            &ctx,
        )
        .unwrap();

        let bias = result.kq_b.unwrap();
        // product(0,1) = energy(0,1) * coact(0,1) = 0.8 * 0.5 = 0.4
        assert!((bias[0 * 4 + 1] - 0.4).abs() < 1e-6);
    }

    // ── BiasScale ──

    #[test]
    fn test_compile_bias_scale() {
        let ctx = test_ctx();
        let result = compile(
            &AttnOp::BiasScale {
                source: BiasSource::SynapseEnergy,
                weight: -1.0,
            },
            &ctx,
        )
        .unwrap();

        // weight=-1.0, ln(energy) is negative for energy<1 → -1 * negative = positive
        // So some values may be positive → kq_b
        // Let's just verify it compiles and has the right shape
        let tensor = result.kq_b.as_ref().or(result.kq_mask_overlay.as_ref()).unwrap();
        assert_eq!(tensor.len(), 16);
    }

    // ── WarpQ/WarpK ─��

    #[test]
    fn test_compile_warp_placeholder() {
        let ctx = test_ctx();
        let result = compile(
            &AttnOp::WarpQ {
                delta_source: DeltaSource::GnnDelta,
                alpha: 0.5,
            },
            &ctx,
        )
        .unwrap();

        // Placeholder: all zeros in kq_b
        assert!(result.kq_b.is_some());
        let bias = result.kq_b.unwrap();
        assert!(bias.iter().all(|&v| v == 0.0));
    }

    // ─��� QueryDelegate ──

    #[test]
    fn test_compile_query_delegate_triggered() {
        let ctx = test_ctx(); // entropy = 1.5
        let result = compile(
            &AttnOp::QueryDelegate {
                entropy_threshold: 1.0, // 1.5 > 1.0 → triggered
                query_type: QueryType::Lookup,
                max_inject_tokens: 32,
            },
            &ctx,
        )
        .unwrap();

        assert!(result.inject_tokens.is_some());
    }

    #[test]
    fn test_compile_query_delegate_not_triggered() {
        let ctx = test_ctx(); // entropy = 1.5
        let result = compile(
            &AttnOp::QueryDelegate {
                entropy_threshold: 2.0, // 1.5 ≤ 2.0 → NOT triggered
                query_type: QueryType::Compute,
                max_inject_tokens: 64,
            },
            &ctx,
        )
        .unwrap();

        assert!(result.is_noop());
    }

    // ── Sequence ──

    #[test]
    fn test_compile_sequence_merges() {
        let ctx = test_ctx();
        let result = compile(
            &AttnOp::Sequence(vec![
                AttnOp::Mask {
                    condition: MaskCondition::GraphDistanceAbove(2),
                },
                AttnOp::BiasAdd {
                    source: BiasSource::SynapseEnergy,
                    weight: -0.5,
                },
            ]),
            &ctx,
        )
        .unwrap();

        // Both produce kq_mask → merged by addition
        let overlay = result.kq_mask_overlay.unwrap();
        assert_eq!(overlay.len(), 16);

        // (0,3): mask=-inf + bias=-0.05 → -inf (blocking wins)
        assert_eq!(overlay[0 * 4 + 3], f32::NEG_INFINITY);

        // (0,1): mask=0 + bias=-0.5*0.8=-0.4 → -0.4
        assert!((overlay[0 * 4 + 1] - (-0.4)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_sequence_empty() {
        let ctx = test_ctx();
        let result = compile(&AttnOp::Sequence(vec![]), &ctx).unwrap();
        assert!(result.is_noop());
    }

    // ── PerHead ──

    #[test]
    fn test_compile_per_head() {
        let ctx = test_ctx();
        let result = compile(
            &AttnOp::PerHead(vec![
                (
                    HeadRange { start: 0, end: 4 },
                    AttnOp::Mask {
                        condition: MaskCondition::GraphDistanceAbove(1),
                    },
                ),
                (
                    HeadRange { start: 4, end: 8 },
                    AttnOp::Identity,
                ),
            ]),
            &ctx,
        )
        .unwrap();

        assert!(result.per_head);
        let overlay = result.kq_mask_overlay.unwrap();
        // Shape: 8 heads × 4 × 4 = 128
        assert_eq!(overlay.len(), 8 * 4 * 4);

        // Head 0: distance(0,2)=2 > 1 → masked
        assert_eq!(overlay[0 * 16 + 0 * 4 + 2], f32::NEG_INFINITY);
        // Head 0: distance(0,1)=1 ≤ 1 → not masked
        assert_eq!(overlay[0 * 16 + 0 * 4 + 1], 0.0);
        // Head 4: Identity → no masking → 0
        assert_eq!(overlay[4 * 16 + 0 * 4 + 2], 0.0);
    }

    #[test]
    fn test_compile_per_head_out_of_bounds() {
        let ctx = test_ctx(); // n_head = 8
        let result = compile(
            &AttnOp::PerHead(vec![(
                HeadRange { start: 0, end: 16 }, // > 8
                AttnOp::Identity,
            )]),
            &ctx,
        );

        assert!(matches!(result, Err(CompileError::HeadRangeOutOfBounds { .. })));
    }

    // ── Conditional ──

    #[test]
    fn test_compile_conditional_true_branch() {
        let ctx = test_ctx(); // context_type = "math"
        let result = compile(
            &AttnOp::Conditional {
                condition: RuntimeCondition::ContextType("math".to_string()),
                then_op: Box::new(AttnOp::BiasAdd {
                    source: BiasSource::Constant(1.0),
                    weight: 0.5,
                }),
                else_op: Box::new(AttnOp::Identity),
            },
            &ctx,
        )
        .unwrap();

        // "math" == "math" → then branch → kq_b with 0.5
        assert!(result.kq_b.is_some());
    }

    #[test]
    fn test_compile_conditional_false_branch() {
        let ctx = test_ctx(); // context_type = "math"
        let result = compile(
            &AttnOp::Conditional {
                condition: RuntimeCondition::ContextType("creative".to_string()),
                then_op: Box::new(AttnOp::BiasAdd {
                    source: BiasSource::Constant(1.0),
                    weight: 0.5,
                }),
                else_op: Box::new(AttnOp::Identity),
            },
            &ctx,
        )
        .unwrap();

        // "math" != "creative" → else branch → Identity
        assert!(result.is_noop());
    }

    #[test]
    fn test_compile_conditional_uncertainty() {
        let ctx = test_ctx(); // entropy = 1.5
        let result = compile(
            &AttnOp::Conditional {
                condition: RuntimeCondition::Uncertainty { threshold: 1.0 },
                then_op: Box::new(AttnOp::Mask {
                    condition: MaskCondition::EnergyBelow(0.5),
                }),
                else_op: Box::new(AttnOp::Identity),
            },
            &ctx,
        )
        .unwrap();

        // 1.5 > 1.0 → then branch → mask
        assert!(result.kq_mask_overlay.is_some());
    }

    #[test]
    fn test_compile_conditional_token_count() {
        let ctx = test_ctx(); // current_token_count = 50
        let result = compile(
            &AttnOp::Conditional {
                condition: RuntimeCondition::TokenCount { min: 100, max: 500 },
                then_op: Box::new(AttnOp::Mask {
                    condition: MaskCondition::NoPath,
                }),
                else_op: Box::new(AttnOp::Identity),
            },
            &ctx,
        )
        .unwrap();

        // 50 < 100 → false → else (Identity)
        assert!(result.is_noop());
    }

    // ── Depth validation ──

    #[test]
    fn test_compile_rejects_deep_tree() {
        let ctx = test_ctx();
        let mut op = AttnOp::Identity;
        for _ in 0..8 {
            op = AttnOp::Sequence(vec![op]);
        }
        // depth = 9 > MAX_DEPTH=8
        let result = compile(&op, &ctx);
        assert!(matches!(result, Err(CompileError::DepthExceeded(_))));
    }

    // ── Round-trip: DSL → compile → verify tensor ���─

    #[test]
    fn test_roundtrip_complex_formula() {
        let ctx = test_ctx();

        // A realistic formula: if uncertain, mask far nodes; else boost close ones
        let formula = AttnOp::Conditional {
            condition: RuntimeCondition::Uncertainty { threshold: 1.0 },
            then_op: Box::new(AttnOp::Sequence(vec![
                AttnOp::Mask {
                    condition: MaskCondition::GraphDistanceAbove(2),
                },
                AttnOp::BiasAdd {
                    source: BiasSource::SynapseEnergy,
                    weight: -0.3,
                },
            ])),
            else_op: Box::new(AttnOp::BiasAdd {
                source: BiasSource::Constant(0.1),
                weight: 1.0,
            }),
        };

        // Serialize → deserialize → compile → verify
        let json = serde_json::to_string(&formula).unwrap();
        let deserialized: AttnOp = serde_json::from_str(&json).unwrap();
        let result = compile(&deserialized, &ctx).unwrap();

        // entropy=1.5 > 1.0 → then branch: mask + bias
        let overlay = result.kq_mask_overlay.unwrap();
        assert_eq!(overlay.len(), 16);

        // (0,3) distance=3 > 2 → -inf from mask, -inf + bias = -inf
        assert_eq!(overlay[0 * 4 + 3], f32::NEG_INFINITY);
        // (0,1) distance=1 ≤ 2 → 0 from mask + (-0.3 * 0.8) = -0.24
        assert!((overlay[0 * 4 + 1] - (-0.24)).abs() < 1e-6);
    }
}
