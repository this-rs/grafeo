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
        BankConfig { bank_id: 0, visibility_start_ratio: 0.0 },   // core: all heads
        BankConfig { bank_id: 1, visibility_start_ratio: 0.25 },  // relations: 75% of heads
        BankConfig { bank_id: 2, visibility_start_ratio: 0.5 },   // 2-hop: 50% of heads
        BankConfig { bank_id: 3, visibility_start_ratio: 0.75 },  // background: 25% of heads
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
    let query_start = nodes.iter()
        .map(|n| n.pos_end)
        .max()
        .unwrap_or(header_tokens) as usize;

    for h in 0..n_h {
        let head_ratio = h as f32 / n_head as f32;
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
                // Phase B: learnable routing via HeadRouter
                r.is_visible(h, bank)
            } else {
                // Phase A fallback: fixed ratio-based visibility
                let ratio = if bank < bank_ratio.len() { bank_ratio[bank] } else { 1.0 };
                head_ratio >= ratio
            };

            if visible {
                // Mark node positions as visible from any row at or after them (causal)
                for pos in node.pos_start..node.pos_end {
                    let col = pos as usize;
                    if col >= n_pos { continue; }
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

    // Collect position IDs (0..total_positions for sparse mask)
    let positions: Vec<i32> = (0..total_positions).collect();

    PerHeadMask {
        mask,
        positions,
        n_head_groups: n_head,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test 1: 4 nodes, 4 banks, 8 heads -- verify shape and specific values
    #[test]
    fn test_4_nodes_4_banks_8_heads() {
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

        let result = build_perhead_mask(&nodes, header_tokens, total_positions, n_head, &config, None);

        // Shape: n_head * n_pos * n_pos
        assert_eq!(result.mask.len(), 8 * 12 * 12);
        assert_eq!(result.n_head_groups, 8);
        assert_eq!(result.positions.len(), 12);

        let n_pos = total_positions as usize;

        // Head 0 (ratio=0.0): should see all banks (all ratios <= 0.0)
        let h0 = 0 * n_pos * n_pos;
        // Bank 0 node at pos 2: should be visible from row 2 onwards
        assert_eq!(result.mask[h0 + 3 * n_pos + 2], 0.0); // row 3, col 2
        // Bank 3 node at pos 8: head 0 ratio=0.0, bank 3 ratio=0.75 → 0.0 >= 0.75 is false → blocked
        assert_eq!(result.mask[h0 + 9 * n_pos + 8], f32::NEG_INFINITY);

        // Head 6 (ratio=0.75): should see banks 0,1,2 (ratios 0.0, 0.25, 0.5) but also bank 3 (0.75 <= 0.75)
        let h6 = 6 * n_pos * n_pos;
        // Bank 3 node at pos 8: visible
        assert_eq!(result.mask[h6 + 9 * n_pos + 8], 0.0);

        // Head 1 (ratio=0.125): should see bank 0 (ratio 0.0 <= 0.125) but NOT bank 1 (0.25 > 0.125)
        let h1 = 1 * n_pos * n_pos;
        // Bank 0 pos 2: visible
        assert_eq!(result.mask[h1 + 3 * n_pos + 2], 0.0);
        // Bank 1 pos 4: blocked
        assert_eq!(result.mask[h1 + 5 * n_pos + 4], f32::NEG_INFINITY);

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
            NodePosition { pos_start: 1, pos_end: 3, bank: 0 },
            NodePosition { pos_start: 3, pos_end: 5, bank: 0 },
        ];
        let config = default_bank_config();
        let header_tokens = 1;
        let total_positions = 5;
        let n_head = 4;

        let result = build_perhead_mask(&nodes, header_tokens, total_positions, n_head, &config, None);

        let n_pos = total_positions as usize;
        // All nodes are bank 0 (ratio 0.0), so all heads should see them
        for h in 0..n_head as usize {
            let ho = h * n_pos * n_pos;
            // Node positions 1..5 should be visible causally
            for col in 1..5usize {
                for row in col..n_pos {
                    assert_eq!(
                        result.mask[ho + row * n_pos + col], 0.0,
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
            NodePosition { pos_start: 1, pos_end: 2, bank: 0 },
            NodePosition { pos_start: 2, pos_end: 3, bank: 1 },
            NodePosition { pos_start: 3, pos_end: 4, bank: 2 },
        ];
        let config = vec![
            BankConfig { bank_id: 0, visibility_start_ratio: 0.0 },
            BankConfig { bank_id: 1, visibility_start_ratio: 0.33 },
            BankConfig { bank_id: 2, visibility_start_ratio: 0.66 },
        ];
        let header_tokens = 1;
        let total_positions = 5;
        let n_head = 7; // not divisible by 3

        let result = build_perhead_mask(&nodes, header_tokens, total_positions, n_head, &config, None);

        assert_eq!(result.mask.len(), 7 * 5 * 5);
        assert_eq!(result.n_head_groups, 7);

        let n_pos = total_positions as usize;

        // Head 0 (ratio=0.0): sees bank 0 only (0.0 >= 0.0 yes, 0.0 >= 0.33 no)
        let h0 = 0 * n_pos * n_pos;
        assert_eq!(result.mask[h0 + 2 * n_pos + 1], 0.0);  // bank 0 pos 1 visible
        assert_eq!(result.mask[h0 + 3 * n_pos + 2], f32::NEG_INFINITY); // bank 1 pos 2 blocked

        // Head 5 (ratio=5/7=0.714): sees banks 0,1 (0.714 >= 0.33) and 2 (0.714 >= 0.66)
        let h5 = 5 * n_pos * n_pos;
        assert_eq!(result.mask[h5 + 4 * n_pos + 3], 0.0); // bank 2 pos 3 visible
    }

    /// Test 4: HeadRouter overrides BankConfig ratios
    #[test]
    fn test_head_router_overrides_bank_config() {
        let nodes = vec![
            NodePosition { pos_start: 2, pos_end: 4, bank: 0 },  // core
            NodePosition { pos_start: 4, pos_end: 6, bank: 2 },  // 2-hop
        ];
        let config = default_bank_config();
        let header_tokens = 2;
        let total_positions = 8;
        let n_head = 4;
        let n_pos = total_positions as usize;

        // Create a HeadRouter where head 0 explicitly blocks bank 2
        let mut router = HeadRouter::new(4, 4, 0.01, 0);
        router.alpha[0 * 4 + 0] = 5.0;  // head 0 sees bank 0 (α=5 → sigmoid≈1)
        router.alpha[0 * 4 + 2] = -5.0; // head 0 blocks bank 2 (α=-5 → sigmoid≈0)

        // Head 3 sees bank 2
        router.alpha[3 * 4 + 0] = 5.0;
        router.alpha[3 * 4 + 2] = 5.0;

        let result = build_perhead_mask(&nodes, header_tokens, total_positions, n_head, &config, Some(&router));

        // Head 0: bank 0 (pos 2-3) visible, bank 2 (pos 4-5) blocked
        let h0 = 0 * n_pos * n_pos;
        assert_eq!(result.mask[h0 + 3 * n_pos + 2], 0.0, "head 0 should see bank 0");
        assert_eq!(result.mask[h0 + 5 * n_pos + 4], f32::NEG_INFINITY, "head 0 should block bank 2");

        // Head 3: both visible
        let h3 = 3 * n_pos * n_pos;
        assert_eq!(result.mask[h3 + 3 * n_pos + 2], 0.0, "head 3 should see bank 0");
        assert_eq!(result.mask[h3 + 5 * n_pos + 4], 0.0, "head 3 should see bank 2");

        // Without router: verify Phase A behavior (head 0 ratio=0.0, sees bank 0 only)
        let result_no_router = build_perhead_mask(&nodes, header_tokens, total_positions, n_head, &config, None);
        let h0_nr = 0 * n_pos * n_pos;
        assert_eq!(result_no_router.mask[h0_nr + 3 * n_pos + 2], 0.0, "no router: head 0 sees bank 0");
        assert_eq!(result_no_router.mask[h0_nr + 5 * n_pos + 4], f32::NEG_INFINITY, "no router: head 0 blocks bank 2");
    }
}
