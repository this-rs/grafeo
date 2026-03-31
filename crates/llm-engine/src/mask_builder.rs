//! Per-head attention mask builder for topological graph masking.
//!
//! Different attention heads can see different subsets of graph nodes,
//! allowing the model to allocate head capacity based on node importance (bank).

use crate::head_router::HeadRouter;

/// Configuration for a single bank's head visibility.
pub struct BankConfig {
    pub bank_id: u32,
    /// Heads from this ratio onwards see this bank's nodes (0.0 = all heads, 0.75 = last 25%)
    pub visibility_start_ratio: f32,
}

/// Returns the default bank configuration:
/// - Bank 0 (core): visible to all heads
/// - Bank 1 (relations): visible to 75% of heads
/// - Bank 2 (2-hop): visible to 50% of heads
/// - Bank 3 (background): visible to 25% of heads
pub fn default_bank_config() -> Vec<BankConfig> {
    vec![
        BankConfig {
            bank_id: 0,
            visibility_start_ratio: 0.0,
        }, // core: all heads
        BankConfig {
            bank_id: 1,
            visibility_start_ratio: 0.25,
        }, // relations: 75% of heads
        BankConfig {
            bank_id: 2,
            visibility_start_ratio: 0.5,
        }, // 2-hop: 50% of heads
        BankConfig {
            bank_id: 3,
            visibility_start_ratio: 0.75,
        }, // background: 25% of heads
    ]
}

/// A node's position range and bank assignment in the KV cache.
pub struct NodePosition {
    pub pos_start: i32, // KV position start (inclusive)
    pub pos_end: i32,   // KV position end (exclusive)
    pub bank: u32,      // which bank this node belongs to
}

/// A per-head attention mask ready to pass to llama_set_attn_mask.
pub struct PerHeadMask {
    /// Flat mask: [n_head * n_pos * n_pos], row-major per head.
    /// 0.0 = visible (allow attention), f32::NEG_INFINITY = blocked.
    pub mask: Vec<f32>,
    /// Position IDs the mask rows/columns correspond to.
    pub positions: Vec<i32>,
    /// Number of head groups (= n_head for full per-head masking).
    pub n_head_groups: i32,
}

/// Build a per-head attention mask for topological graph masking.
///
/// # Arguments
/// - `nodes`: the graph nodes with their KV position ranges and bank assignments
/// - `header_tokens`: number of header/system positions (always visible to all heads)
/// - `total_positions`: total number of sparse positions in the mask
/// - `n_head`: number of attention heads in the model
/// - `config`: bank visibility configuration (used when `router` is None)
/// - `router`: optional HeadRouter with learned α-weights. When provided,
///   visibility is determined by `router.is_visible(h, bank)` instead of
///   the fixed BankConfig ratios.
///
/// # Mask layout
/// `mask[head * n_pos * n_pos + row * n_pos + col]` where:
/// - `0.0` = visible (allow attention)
/// - `f32::NEG_INFINITY` = blocked
pub fn build_perhead_mask(
    nodes: &[NodePosition],
    header_tokens: i32,
    total_positions: i32,
    n_head: i32,
    config: &[BankConfig],
    router: Option<&HeadRouter>,
    sparse_positions: Option<&[i32]>,
) -> PerHeadMask {
    let n_pos = total_positions as usize;
    let n_h = n_head as usize;
    let mut mask = vec![f32::NEG_INFINITY; n_h * n_pos * n_pos];

    // Build a lookup from bank_id -> visibility_start_ratio (used when no router)
    let max_bank = config.iter().map(|c| c.bank_id).max().unwrap_or(0) as usize;
    let mut bank_ratio = vec![1.0f32; max_bank + 1]; // default: invisible
    for c in config {
        bank_ratio[c.bank_id as usize] = c.visibility_start_ratio;
    }

    // Determine where query/gen positions start (after all nodes)
    // With compact indices, this is simply the max compact end of any node.
    let query_start = nodes
        .iter()
        .map(|n| n.pos_end)
        .max()
        .unwrap_or(header_tokens) as usize;

    // Detect position gaps between header, nodes, and query (evicted KV positions).
    // These gaps are correctly blocked (-INF) but logged for diagnostics.
    {
        let header_end = header_tokens as usize;
        let mut covered = vec![false; n_pos];
        for i in 0..header_end.min(n_pos) { covered[i] = true; }
        for node in nodes {
            for p in (node.pos_start as usize)..(node.pos_end as usize).min(n_pos) {
                covered[p] = true;
            }
        }
        for i in query_start..n_pos { covered[i] = true; }
        let n_gaps = covered.iter().filter(|&&c| !c).count();
        if n_gaps > 0 {
            kv_registry::kv_debug!(
                "  [mask-gap] header=0..{}, query_start={}, n_pos={}, GAPS={} positions (evicted KV slots)",
                header_end, query_start, n_pos, n_gaps
            );
        }
    }

    for h in 0..n_h {
        let _head_ratio = h as f32 / n_head as f32;
        let head_offset = h * n_pos * n_pos;

        // 1. Header positions (0..header_tokens): visible to ALL downstream rows (causal).
        //    The system prompt must always be attended to by nodes, query, and gen tokens.
        for col in 0..header_tokens as usize {
            for row in col..n_pos {
                mask[head_offset + row * n_pos + col] = 0.0;
            }
        }

        // 2. Node positions: visible based on router α-weights or fixed BankConfig ratios
        for node in nodes {
            let bank = node.bank as usize;

            let visible = if let Some(r) = router {
                if r.n_updates == 0 {
                    // Not yet trained → all banks visible (safe neutral default)
                    true
                } else {
                    // Phase B: learnable routing via HeadRouter
                    r.is_visible(h, bank)
                }
            } else {
                // No router → all banks visible (broadcast mask).
                // The restrictive Phase A ratios are only meaningful after
                // learning; without a trained router, hiding banks from
                // heads just destroys context for no benefit.
                true
            };

            if visible {
                // Mark node positions as visible from any row at or after them (causal)
                for pos in node.pos_start..node.pos_end {
                    let col = pos as usize;
                    if col >= n_pos {
                        continue;
                    }
                    for row in col..n_pos {
                        mask[head_offset + row * n_pos + col] = 0.0;
                    }
                }
            }
        }

        // 3. Query/gen positions: causal self-attention (always see previous query/gen tokens).
        //    These start after the last node and must always attend to each other causally,
        //    regardless of bank routing. Without this, autoregressive generation breaks.
        for col in query_start..n_pos {
            for row in col..n_pos {
                mask[head_offset + row * n_pos + col] = 0.0;
            }
        }
    }

    // Use sparse positions if provided (compact indices mapped to real KV positions),
    // otherwise fall back to 0..total_positions for dense/legacy callers.
    let positions: Vec<i32> = if let Some(sp) = sparse_positions {
        sp.to_vec()
    } else {
        (0..total_positions).collect()
    };

    PerHeadMask {
        mask,
        positions,
        n_head_groups: n_head,
    }
}

// ── T4.3: Sparse mask analysis for deterministic dequant skip ────

/// A contiguous range of active (non-dead) KV positions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActiveRange {
    /// Start position (inclusive).
    pub start: usize,
    /// End position (exclusive).
    pub end: usize,
}

impl ActiveRange {
    /// Number of positions in this range.
    pub fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }
}

/// Per-head active ranges (positions that are NOT fully masked).
#[derive(Debug, Clone)]
pub struct PerHeadActiveRanges {
    /// Active ranges per head. `ranges[h]` = list of contiguous active ranges for head h.
    pub ranges: Vec<Vec<ActiveRange>>,
    /// Total active positions per head.
    pub active_count: Vec<usize>,
    /// Positions that are dead across ALL heads (globally dead).
    pub globally_dead: Vec<usize>,
    /// Total KV positions in the mask.
    pub total_positions: usize,
}

impl PerHeadActiveRanges {
    /// Global sparsity ratio (fraction of globally dead positions).
    pub fn global_sparsity(&self) -> f32 {
        if self.total_positions == 0 {
            return 0.0;
        }
        self.globally_dead.len() as f32 / self.total_positions as f32
    }

    /// Per-head sparsity ratio for a given head.
    pub fn head_sparsity(&self, h: usize) -> f32 {
        if self.total_positions == 0 || h >= self.active_count.len() {
            return 0.0;
        }
        1.0 - self.active_count[h] as f32 / self.total_positions as f32
    }

    /// Compact position map: original position → compact index.
    /// Only includes globally active positions.
    /// Used to build a compact K/V subtensor.
    pub fn compact_position_map(&self) -> Vec<(usize, usize)> {
        let dead_set: std::collections::HashSet<usize> =
            self.globally_dead.iter().copied().collect();
        let mut mapping = Vec::new();
        let mut compact_idx = 0;
        for pos in 0..self.total_positions {
            if !dead_set.contains(&pos) {
                mapping.push((pos, compact_idx));
                compact_idx += 1;
            }
        }
        mapping
    }

    /// Number of globally active positions (total - globally dead).
    pub fn globally_active_count(&self) -> usize {
        self.total_positions - self.globally_dead.len()
    }
}

/// Analyze a per-head mask to identify dead columns (fully masked KV positions).
///
/// A column `col` is "dead" for head `h` if `mask[h][row][col] == -∞` for ALL `row >= col`.
/// This means no query token ever attends to that KV position for that head.
///
/// Returns `PerHeadActiveRanges` with:
/// - Per-head contiguous ranges of active positions
/// - Globally dead positions (dead across ALL heads)
///
/// # Use case
/// Positions that are globally dead can be excluded from K/V dequantization entirely,
/// saving bandwidth proportional to sparsity. For per-head dead positions, sparse
/// attention kernels can skip those K/V rows in the matmul.
pub fn analyze_mask_sparsity(mask: &PerHeadMask) -> PerHeadActiveRanges {
    let n_pos = mask.positions.len();
    let n_head = mask.n_head_groups as usize;

    if n_pos == 0 || n_head == 0 {
        return PerHeadActiveRanges {
            ranges: vec![vec![]; n_head],
            active_count: vec![0; n_head],
            globally_dead: vec![],
            total_positions: n_pos,
        };
    }

    // Per-head: determine which columns are dead
    // dead_per_head[h][col] = true if col is dead for head h
    let mut dead_per_head = vec![vec![true; n_pos]; n_head];

    for h in 0..n_head {
        let head_offset = h * n_pos * n_pos;
        for col in 0..n_pos {
            // Check all rows >= col (causal: only rows at or after col matter)
            for row in col..n_pos {
                let val = mask.mask[head_offset + row * n_pos + col];
                if val > f32::NEG_INFINITY {
                    // This column is active for this head (at least one row can see it)
                    dead_per_head[h][col] = false;
                    break;
                }
            }
        }
    }

    // Build per-head active ranges (contiguous runs of non-dead columns)
    let mut ranges = Vec::with_capacity(n_head);
    let mut active_count = Vec::with_capacity(n_head);

    for h in 0..n_head {
        let mut head_ranges = Vec::new();
        let mut count = 0usize;
        let mut range_start: Option<usize> = None;

        for col in 0..n_pos {
            if !dead_per_head[h][col] {
                // Active column
                if range_start.is_none() {
                    range_start = Some(col);
                }
                count += 1;
            } else {
                // Dead column — close current range if open
                if let Some(start) = range_start.take() {
                    head_ranges.push(ActiveRange {
                        start,
                        end: col,
                    });
                }
            }
        }
        // Close final range
        if let Some(start) = range_start {
            head_ranges.push(ActiveRange {
                start,
                end: n_pos,
            });
        }

        ranges.push(head_ranges);
        active_count.push(count);
    }

    // Globally dead: columns dead across ALL heads
    let mut globally_dead = Vec::new();
    for col in 0..n_pos {
        if dead_per_head.iter().all(|h_dead| h_dead[col]) {
            globally_dead.push(col);
        }
    }

    PerHeadActiveRanges {
        ranges,
        active_count,
        globally_dead,
        total_positions: n_pos,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a HeadRouter that replicates Phase A fixed-ratio visibility.
    /// Used by sparsity tests that need blocked positions to validate analysis.
    fn phase_a_router(n_head: usize, config: &[BankConfig]) -> HeadRouter {
        let n_bank = config.iter().map(|c| c.bank_id as usize + 1).max().unwrap_or(1);
        let mut router = HeadRouter::new(n_head, n_bank, 0.01, 0);
        router.n_updates = 1; // mark as trained so learned routing is used
        for h in 0..n_head {
            let head_ratio = h as f32 / n_head as f32;
            for c in config {
                let bank = c.bank_id as usize;
                let visible = head_ratio >= c.visibility_start_ratio;
                router.alpha[h * n_bank + bank] = if visible { 5.0 } else { -5.0 };
            }
        }
        router
    }

    /// Test 1: 4 nodes, 4 banks, 8 heads -- verify shape and specific values
    #[test]
    fn test_4_nodes_4_banks_8_heads() {
        let nodes = vec![
            NodePosition {
                pos_start: 2,
                pos_end: 4,
                bank: 0,
            },
            NodePosition {
                pos_start: 4,
                pos_end: 6,
                bank: 1,
            },
            NodePosition {
                pos_start: 6,
                pos_end: 8,
                bank: 2,
            },
            NodePosition {
                pos_start: 8,
                pos_end: 10,
                bank: 3,
            },
        ];
        let config = default_bank_config();
        let header_tokens = 2;
        let total_positions = 12; // 2 header + 4*2 node + 2 query
        let n_head = 8;

        let result = build_perhead_mask(
            &nodes,
            header_tokens,
            total_positions,
            n_head,
            &config,
            None,
            None,
        );

        // Shape: n_head * n_pos * n_pos
        assert_eq!(result.mask.len(), 8 * 12 * 12);
        assert_eq!(result.n_head_groups, 8);
        assert_eq!(result.positions.len(), 12);

        let n_pos = total_positions as usize;

        // No router → all banks visible to all heads (broadcast mask)
        let h0 = 0 * n_pos * n_pos;
        // Bank 0 node at pos 2: visible
        assert_eq!(result.mask[h0 + 3 * n_pos + 2], 0.0); // row 3, col 2
        // Bank 3 node at pos 8: visible (no router = all visible)
        assert_eq!(result.mask[h0 + 9 * n_pos + 8], 0.0);

        // Head 6: all banks visible too
        let h6 = 6 * n_pos * n_pos;
        assert_eq!(result.mask[h6 + 9 * n_pos + 8], 0.0);

        // Head 1: all banks visible (no router)
        let h1 = 1 * n_pos * n_pos;
        // Bank 0 pos 2: visible
        assert_eq!(result.mask[h1 + 3 * n_pos + 2], 0.0);
        // Bank 1 pos 4: visible (no router = broadcast)
        assert_eq!(result.mask[h1 + 5 * n_pos + 4], 0.0);

        // Header positions always visible to all heads
        for h in 0..n_head as usize {
            let ho = h * n_pos * n_pos;
            assert_eq!(result.mask[ho + 0 * n_pos + 0], 0.0); // pos 0 sees itself
            assert_eq!(result.mask[ho + 1 * n_pos + 0], 0.0); // pos 1 sees pos 0
            assert_eq!(result.mask[ho + 1 * n_pos + 1], 0.0); // pos 1 sees itself
        }
    }

    /// Test 2: All nodes bank 0 -- all 0.0 (equivalent to broadcast, all visible)
    #[test]
    fn test_all_bank_0_all_visible() {
        let nodes = vec![
            NodePosition {
                pos_start: 1,
                pos_end: 3,
                bank: 0,
            },
            NodePosition {
                pos_start: 3,
                pos_end: 5,
                bank: 0,
            },
        ];
        let config = default_bank_config();
        let header_tokens = 1;
        let total_positions = 5;
        let n_head = 4;

        let result = build_perhead_mask(
            &nodes,
            header_tokens,
            total_positions,
            n_head,
            &config,
            None,
            None,
        );

        let n_pos = total_positions as usize;
        // All nodes are bank 0 (ratio 0.0), so all heads should see them
        for h in 0..n_head as usize {
            let ho = h * n_pos * n_pos;
            // Node positions 1..5 should be visible causally
            for col in 1..5usize {
                for row in col..n_pos {
                    assert_eq!(
                        result.mask[ho + row * n_pos + col],
                        0.0,
                        "head={h} row={row} col={col} should be visible"
                    );
                }
            }
        }
    }

    /// Test 3: n_head not divisible by n_banks -- proportional mapping still works
    #[test]
    fn test_non_divisible_heads() {
        let nodes = vec![
            NodePosition {
                pos_start: 1,
                pos_end: 2,
                bank: 0,
            },
            NodePosition {
                pos_start: 2,
                pos_end: 3,
                bank: 1,
            },
            NodePosition {
                pos_start: 3,
                pos_end: 4,
                bank: 2,
            },
        ];
        let config = vec![
            BankConfig {
                bank_id: 0,
                visibility_start_ratio: 0.0,
            },
            BankConfig {
                bank_id: 1,
                visibility_start_ratio: 0.33,
            },
            BankConfig {
                bank_id: 2,
                visibility_start_ratio: 0.66,
            },
        ];
        let header_tokens = 1;
        let total_positions = 5;
        let n_head = 7; // not divisible by 3

        let result = build_perhead_mask(
            &nodes,
            header_tokens,
            total_positions,
            n_head,
            &config,
            None,
            None,
        );

        assert_eq!(result.mask.len(), 7 * 5 * 5);
        assert_eq!(result.n_head_groups, 7);

        let n_pos = total_positions as usize;

        // No router → all banks visible to all heads
        let h0 = 0 * n_pos * n_pos;
        assert_eq!(result.mask[h0 + 2 * n_pos + 1], 0.0); // bank 0 pos 1 visible
        assert_eq!(result.mask[h0 + 3 * n_pos + 2], 0.0); // bank 1 pos 2 visible (no router)

        // Head 5: all visible too
        let h5 = 5 * n_pos * n_pos;
        assert_eq!(result.mask[h5 + 4 * n_pos + 3], 0.0); // bank 2 pos 3 visible
    }

    /// Test 4: HeadRouter overrides BankConfig ratios
    #[test]
    fn test_head_router_overrides_bank_config() {
        let nodes = vec![
            NodePosition {
                pos_start: 2,
                pos_end: 4,
                bank: 0,
            }, // core
            NodePosition {
                pos_start: 4,
                pos_end: 6,
                bank: 2,
            }, // 2-hop
        ];
        let config = default_bank_config();
        let header_tokens = 2;
        let total_positions = 8;
        let n_head = 4;
        let n_pos = total_positions as usize;

        // Create a HeadRouter where head 0 explicitly blocks bank 2
        let mut router = HeadRouter::new(4, 4, 0.01, 0);
        router.n_updates = 1; // Mark as trained so learned routing is used
        router.alpha[0 * 4 + 0] = 5.0; // head 0 sees bank 0 (α=5 → sigmoid≈1)
        router.alpha[0 * 4 + 2] = -5.0; // head 0 blocks bank 2 (α=-5 → sigmoid≈0)

        // Head 3 sees bank 2
        router.alpha[3 * 4 + 0] = 5.0;
        router.alpha[3 * 4 + 2] = 5.0;

        let result = build_perhead_mask(
            &nodes,
            header_tokens,
            total_positions,
            n_head,
            &config,
            Some(&router),
            None,
        );

        // Head 0: bank 0 (pos 2-3) visible, bank 2 (pos 4-5) blocked
        let h0 = 0 * n_pos * n_pos;
        assert_eq!(
            result.mask[h0 + 3 * n_pos + 2],
            0.0,
            "head 0 should see bank 0"
        );
        assert_eq!(
            result.mask[h0 + 5 * n_pos + 4],
            f32::NEG_INFINITY,
            "head 0 should block bank 2"
        );

        // Head 3: both visible
        let h3 = 3 * n_pos * n_pos;
        assert_eq!(
            result.mask[h3 + 3 * n_pos + 2],
            0.0,
            "head 3 should see bank 0"
        );
        assert_eq!(
            result.mask[h3 + 5 * n_pos + 4],
            0.0,
            "head 3 should see bank 2"
        );

        // Without router: all banks visible (broadcast mask)
        let result_no_router = build_perhead_mask(
            &nodes,
            header_tokens,
            total_positions,
            n_head,
            &config,
            None,
            None,
        );
        let h0_nr = 0 * n_pos * n_pos;
        assert_eq!(
            result_no_router.mask[h0_nr + 3 * n_pos + 2],
            0.0,
            "no router: head 0 sees bank 0"
        );
        assert_eq!(
            result_no_router.mask[h0_nr + 5 * n_pos + 4],
            0.0,
            "no router: all banks visible (broadcast)"
        );
    }

    // ─────────────────────────────────────────────────────────────────
    // REGRESSION tests for documented bugs
    // ─────────────────────────────────────────────────────────────────

    /// REGRESSION: Header must be visible to ALL downstream rows, including query/gen.
    /// Bug: header visibility loop only covered `i < header_tokens`, so query tokens
    /// at positions > header_tokens could NOT see the system prompt → garbage output.
    /// Fix: header visible from `col..n_pos` (all rows after the header position).
    #[test]
    fn test_regression_header_visible_to_all_rows() {
        let nodes = vec![
            NodePosition { pos_start: 5, pos_end: 8, bank: 0 },
        ];
        let config = default_bank_config();
        let header_tokens = 5;
        let total_positions = 12; // 5 header + 3 node + 4 query/gen
        let n_head = 4;
        let n_pos = total_positions as usize;

        let result = build_perhead_mask(&nodes, header_tokens, total_positions, n_head, &config, None, None);

        for h in 0..n_head as usize {
            let ho = h * n_pos * n_pos;
            // Every header col must be visible from EVERY subsequent row
            for col in 0..header_tokens as usize {
                for row in col..n_pos {
                    assert_eq!(
                        result.mask[ho + row * n_pos + col], 0.0,
                        "head={h} row={row} col={col}: header must be visible to ALL downstream rows"
                    );
                }
            }
            // Specifically: query positions (8..12) must see header positions (0..5)
            for query_row in 8..12 {
                for header_col in 0..5 {
                    assert_eq!(
                        result.mask[ho + query_row * n_pos + header_col], 0.0,
                        "head={h}: query row {query_row} must see header col {header_col}"
                    );
                }
            }
        }
    }

    /// REGRESSION: Query/gen positions must have causal self-attention.
    /// Bug: query positions were not in the `nodes` list, so they couldn't attend
    /// to each other → autoregressive generation broke (model repeated or stopped).
    /// Fix: explicit causal self-attention block for query_start..n_pos.
    #[test]
    fn test_regression_query_self_attention_causal() {
        let nodes = vec![
            NodePosition { pos_start: 2, pos_end: 4, bank: 0 },
        ];
        let config = default_bank_config();
        let header_tokens = 2;
        let total_positions = 8; // 2 header + 2 node + 4 query/gen
        let n_head = 2;
        let n_pos = total_positions as usize;
        let query_start = 4; // after last node

        let result = build_perhead_mask(&nodes, header_tokens, total_positions, n_head, &config, None, None);

        for h in 0..n_head as usize {
            let ho = h * n_pos * n_pos;
            // Query positions must see themselves and all prior query positions (causal)
            for col in query_start..n_pos {
                for row in col..n_pos {
                    assert_eq!(
                        result.mask[ho + row * n_pos + col], 0.0,
                        "head={h}: query row {row} must see query col {col} (causal self-attention)"
                    );
                }
            }
            // But future positions must NOT be visible (causal = no look-ahead)
            for col in (query_start + 1)..n_pos {
                for row in query_start..col {
                    assert_eq!(
                        result.mask[ho + row * n_pos + col], f32::NEG_INFINITY,
                        "head={h}: query row {row} must NOT see future col {col} (causal)"
                    );
                }
            }
        }
    }

    /// REGRESSION: Mask must use f32::NEG_INFINITY, not -1e30 or other values.
    /// Bug: JSON serialization converts -Infinity to -1e30, but the post-pass
    /// comparison used `== -INFINITY` (exact). Threshold comparison is needed.
    /// This test verifies the mask builder uses the correct constant.
    #[test]
    fn test_regression_mask_uses_neg_infinity_not_finite() {
        let nodes = vec![
            NodePosition { pos_start: 1, pos_end: 2, bank: 3 }, // high bank
        ];
        let config = default_bank_config();
        let header_tokens = 1;
        let total_positions = 3;
        let n_head = 2;
        let n_pos = total_positions as usize;

        // Use Phase A router so head 0 blocks bank 3
        let router = phase_a_router(n_head as usize, &config);
        let result = build_perhead_mask(&nodes, header_tokens, total_positions, n_head, &config, Some(&router), None);

        // Head 0 (ratio 0.0) should block bank 3 (ratio 0.75)
        let h0 = 0 * n_pos * n_pos;
        let blocked_val = result.mask[h0 + 2 * n_pos + 1]; // row 2, col 1 (bank 3 node)
        assert!(
            blocked_val.is_infinite() && blocked_val.is_sign_negative(),
            "blocked positions must use f32::NEG_INFINITY, got {}",
            blocked_val
        );
        assert_eq!(blocked_val, f32::NEG_INFINITY);
    }

    /// REGRESSION: Empty nodes list must not panic and must produce valid causal mask.
    #[test]
    fn test_regression_empty_nodes_no_panic() {
        let config = default_bank_config();
        let header_tokens = 3;
        let total_positions = 6;
        let n_head = 2;
        let n_pos = total_positions as usize;

        let result = build_perhead_mask(&[], header_tokens, total_positions, n_head, &config, None, None);

        assert_eq!(result.mask.len(), 2 * 6 * 6);
        // Header should still be visible
        for h in 0..2 {
            let ho = h * n_pos * n_pos;
            for col in 0..3 {
                for row in col..n_pos {
                    assert_eq!(result.mask[ho + row * n_pos + col], 0.0);
                }
            }
        }
        // Query positions (3..6) should have causal self-attention
        for h in 0..2 {
            let ho = h * n_pos * n_pos;
            for col in 3..n_pos {
                for row in col..n_pos {
                    assert_eq!(result.mask[ho + row * n_pos + col], 0.0);
                }
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────
    // T4.3: Sparse mask analysis tests
    // ─────────────────────────────────────────────────────────────────

    /// T4.3: 4 banks, 8 heads — head 0 sees only bank 0 + header + query,
    /// so ~75% of node columns are dead for head 0.
    #[test]
    fn test_sparsity_4_banks_8_heads() {
        let nodes = vec![
            NodePosition { pos_start: 2, pos_end: 4, bank: 0 },
            NodePosition { pos_start: 4, pos_end: 6, bank: 1 },
            NodePosition { pos_start: 6, pos_end: 8, bank: 2 },
            NodePosition { pos_start: 8, pos_end: 10, bank: 3 },
        ];
        let config = default_bank_config();
        let header_tokens = 2;
        let total_positions = 12; // 2 header + 4*2 node + 2 query
        let n_head = 8;

        // Use Phase A router to get meaningful sparsity (no-router = all visible)
        let router = phase_a_router(n_head as usize, &config);
        let mask = build_perhead_mask(&nodes, header_tokens, total_positions, n_head, &config, Some(&router), None);
        let sparsity = analyze_mask_sparsity(&mask);

        assert_eq!(sparsity.total_positions, 12);

        // Head 0 (ratio=0.0): sees bank 0 (pos 2-3) + header (0-1) + query (10-11).
        // Dead: positions 4-9 (banks 1,2,3) = 6 dead / 12 total = 50% sparsity
        assert!(sparsity.head_sparsity(0) >= 0.4, "head 0 should have ~50% sparsity, got {}", sparsity.head_sparsity(0));

        // Head 7 (ratio=7/8=0.875): sees all banks → 0% dead for node positions
        assert!(sparsity.head_sparsity(7) < 0.05, "head 7 should have ~0% sparsity, got {}", sparsity.head_sparsity(7));

        // No position is globally dead because head 7 (sees all banks) makes all positions active
        assert!(sparsity.globally_dead.is_empty(), "no globally dead positions when at least one head sees all banks");
    }

    /// T4.3: All nodes bank 3 — head 0 blocks all node positions.
    /// Only header + query are active for head 0.
    #[test]
    fn test_sparsity_all_bank_3() {
        let nodes = vec![
            NodePosition { pos_start: 2, pos_end: 4, bank: 3 },
            NodePosition { pos_start: 4, pos_end: 6, bank: 3 },
        ];
        let config = default_bank_config();
        let header_tokens = 2;
        let total_positions = 8;
        let n_head = 4;

        // Use Phase A router so head 0 blocks bank 3
        let router = phase_a_router(n_head as usize, &config);
        let mask = build_perhead_mask(&nodes, header_tokens, total_positions, n_head, &config, Some(&router), None);
        let sparsity = analyze_mask_sparsity(&mask);

        // Head 0 (ratio=0.0): blocks bank 3 (ratio 0.75) → positions 2-5 are dead
        // Active: header (0,1) + query (6,7) = 4. Dead: 4. Sparsity = 50%
        assert_eq!(sparsity.active_count[0], 4, "head 0: 4 active positions (header+query)");
        assert!((sparsity.head_sparsity(0) - 0.5).abs() < 0.01);

        // Head 3 (ratio=0.75): sees bank 3 → all positions active
        assert_eq!(sparsity.active_count[3], 8);

        // Verify active ranges for head 0: [0..2] (header), [6..8] (query)
        assert_eq!(sparsity.ranges[0].len(), 2);
        assert_eq!(sparsity.ranges[0][0], ActiveRange { start: 0, end: 2 });
        assert_eq!(sparsity.ranges[0][1], ActiveRange { start: 6, end: 8 });
    }

    /// T4.3: Globally dead positions — all nodes are bank 3 with only 1 head,
    /// and that head can't see bank 3.
    #[test]
    fn test_sparsity_globally_dead() {
        // Single head with ratio 0.0 — can only see bank 0 (ratio 0.0)
        let nodes = vec![
            NodePosition { pos_start: 1, pos_end: 3, bank: 3 },
        ];
        let config = default_bank_config();
        let header_tokens = 1;
        let total_positions = 5; // 1 header + 2 node + 2 query
        let n_head = 1; // ratio 0.0: can't see bank 3

        // Use Phase A router so the single head blocks bank 3
        let router = phase_a_router(n_head as usize, &config);
        let mask = build_perhead_mask(&nodes, header_tokens, total_positions, n_head, &config, Some(&router), None);
        let sparsity = analyze_mask_sparsity(&mask);

        // Positions 1,2 are bank 3 — dead for the only head → globally dead
        assert_eq!(sparsity.globally_dead, vec![1, 2]);
        assert!((sparsity.global_sparsity() - 0.4).abs() < 0.01);
    }

    /// T4.3: Compact position map excludes globally dead positions.
    #[test]
    fn test_compact_position_map() {
        let nodes = vec![
            NodePosition { pos_start: 1, pos_end: 3, bank: 3 },
        ];
        let config = default_bank_config();
        let header_tokens = 1;
        let total_positions = 5;
        let n_head = 1;

        // Use Phase A router so bank 3 is blocked → positions 1,2 become globally dead
        let router = phase_a_router(n_head as usize, &config);
        let mask = build_perhead_mask(&nodes, header_tokens, total_positions, n_head, &config, Some(&router), None);
        let sparsity = analyze_mask_sparsity(&mask);

        let map = sparsity.compact_position_map();
        // Positions 0, 3, 4 are active (header=0, query=3,4). Positions 1,2 are dead.
        assert_eq!(map.len(), 3);
        assert_eq!(map[0], (0, 0)); // original pos 0 → compact 0
        assert_eq!(map[1], (3, 1)); // original pos 3 → compact 1
        assert_eq!(map[2], (4, 2)); // original pos 4 → compact 2
    }

    /// T4.3: Empty mask produces no dead positions.
    #[test]
    fn test_sparsity_empty_mask() {
        let sparsity = analyze_mask_sparsity(&PerHeadMask {
            mask: vec![],
            positions: vec![],
            n_head_groups: 0,
        });
        assert_eq!(sparsity.total_positions, 0);
        assert!(sparsity.globally_dead.is_empty());
        assert_eq!(sparsity.global_sparsity(), 0.0);
    }

    /// T4.3: ActiveRange::len() computes correctly.
    #[test]
    fn test_active_range_len() {
        assert_eq!(ActiveRange { start: 0, end: 5 }.len(), 5);
        assert_eq!(ActiveRange { start: 3, end: 3 }.len(), 0);
        assert_eq!(ActiveRange { start: 10, end: 15 }.len(), 5);
    }
}
