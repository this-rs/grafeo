//! State kq_b — Attention bias from SelfMetrics.
//!
//! Computes a per-head, per-KV-position bias tensor that shifts the model's
//! attention based on internal state (reward, confidence). This replaces the
//! self-embed proprioceptive channel which required fine-tuning.
//!
//! The bias is computed ONCE before generation (static pattern, confirmed
//! superior to dynamic per-token by Elun benchmarks).
//!
//! Layout: `[n_head × n_kv]` flattened row-major (head-major).

/// Classification of KV cache positions for bias computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionType {
    /// System header (instructions, persona). Not biased.
    Header,
    /// Graph node injected via embedding or text. Bias modulated by reward.
    GraphNode,
    /// Conversation context fragment. Bias modulated by confidence.
    ConvContext,
    /// Self-embed proprioceptive position (deprecated). Not biased.
    SelfEmbed,
    /// Query tokens. Not biased (they're the source of attention, not target).
    Query,
}

/// Simplified metrics interface for state bias computation.
/// Decoupled from PersonaDB::SelfMetrics to avoid circular dependencies.
#[derive(Debug, Clone)]
pub struct StateMetrics {
    /// Running average reward [-1.0, 1.0]. Higher = graph nodes are useful.
    pub reward_avg: f32,
    /// Confidence [0.0, 1.0]. Lower = spread attention more on context.
    pub confidence: f32,
}

impl Default for StateMetrics {
    fn default() -> Self {
        Self {
            reward_avg: 0.0,
            confidence: 0.5,
        }
    }
}

/// Configuration for state bias computation.
#[derive(Debug, Clone)]
pub struct StateBiasConfig {
    /// Bias strength for graph nodes: bias = reward_avg × graph_weight.
    pub graph_weight: f32,
    /// Bias strength for conv context: bias = (1 - confidence) × conv_weight.
    pub conv_weight: f32,
    /// Maximum absolute bias value (clamp).
    pub max_bias: f32,
}

impl Default for StateBiasConfig {
    fn default() -> Self {
        Self {
            graph_weight: 0.5,
            conv_weight: 0.3,
            max_bias: 1.0,
        }
    }
}

/// Computes the state_kq_b attention bias tensor.
pub struct StateBiasComputer {
    config: StateBiasConfig,
}

impl StateBiasComputer {
    pub fn new(config: StateBiasConfig) -> Self {
        Self { config }
    }

    /// Build a position type map from KV registry state.
    ///
    /// - `header_end`: token position where the header ends
    /// - `graph_node_ranges`: (start, end) position ranges for each graph node
    /// - `conv_ranges`: (start, end) position ranges for each conv fragment
    /// - `n_kv`: total KV cache positions to map
    ///
    /// Positions not covered by any range default to Query.
    pub fn build_position_map(
        header_end: i32,
        graph_node_ranges: &[(i32, i32)],
        conv_ranges: &[(i32, i32)],
        n_kv: i32,
    ) -> Vec<PositionType> {
        let n = n_kv as usize;
        let mut map = vec![PositionType::Query; n];

        // Header positions
        for i in 0..std::cmp::min(header_end as usize, n) {
            map[i] = PositionType::Header;
        }

        // Graph node positions
        for &(start, end) in graph_node_ranges {
            for i in (start as usize)..std::cmp::min(end as usize, n) {
                map[i] = PositionType::GraphNode;
            }
        }

        // Conversation context positions
        for &(start, end) in conv_ranges {
            for i in (start as usize)..std::cmp::min(end as usize, n) {
                map[i] = PositionType::ConvContext;
            }
        }

        map
    }

    /// Compute the bias tensor: `[n_head × n_kv]` flattened row-major.
    ///
    /// For each (head, position):
    /// - GraphNode: bias = reward_avg × graph_weight
    /// - ConvContext: bias = (1 - confidence) × conv_weight
    /// - Header/SelfEmbed/Query: bias = 0.0
    ///
    /// The bias is uniform across heads (same value for all heads at a given position).
    /// Future: per-head routing via HeadRouter top-5.
    pub fn compute(
        &self,
        metrics: &StateMetrics,
        position_map: &[PositionType],
        n_head: usize,
    ) -> Vec<f32> {
        let n_kv = position_map.len();
        let mut bias = vec![0.0f32; n_head * n_kv];

        // Pre-compute per-position-type biases
        let graph_bias = (metrics.reward_avg * self.config.graph_weight)
            .clamp(-self.config.max_bias, self.config.max_bias);
        let conv_bias = ((1.0 - metrics.confidence) * self.config.conv_weight)
            .clamp(-self.config.max_bias, self.config.max_bias);

        for (pos, &ptype) in position_map.iter().enumerate() {
            let b = match ptype {
                PositionType::GraphNode => graph_bias,
                PositionType::ConvContext => conv_bias,
                PositionType::Header | PositionType::SelfEmbed | PositionType::Query => 0.0,
            };

            if b != 0.0 {
                // Broadcast same bias across all heads for this position
                for h in 0..n_head {
                    bias[h * n_kv + pos] = b;
                }
            }
        }

        bias
    }

    /// Compute bias magnitude for logging.
    pub fn bias_magnitude(bias: &[f32]) -> f32 {
        if bias.is_empty() {
            return 0.0;
        }
        let sum_sq: f32 = bias.iter().map(|b| b * b).sum();
        (sum_sq / bias.len() as f32).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_map_basic() {
        // Header: 0..10, Graph node: 10..15, Conv: 15..20, Rest: Query
        let map = StateBiasComputer::build_position_map(
            10,
            &[(10, 15)],
            &[(15, 20)],
            25,
        );
        assert_eq!(map.len(), 25);
        assert_eq!(map[0], PositionType::Header);
        assert_eq!(map[9], PositionType::Header);
        assert_eq!(map[10], PositionType::GraphNode);
        assert_eq!(map[14], PositionType::GraphNode);
        assert_eq!(map[15], PositionType::ConvContext);
        assert_eq!(map[19], PositionType::ConvContext);
        assert_eq!(map[20], PositionType::Query);
    }

    #[test]
    fn test_positive_reward_boosts_graph_nodes() {
        let computer = StateBiasComputer::new(StateBiasConfig::default());
        let metrics = StateMetrics {
            reward_avg: 1.0,
            confidence: 0.5,
        };
        // 2 heads, 5 positions: [Header, GraphNode, GraphNode, Conv, Query]
        let position_map = vec![
            PositionType::Header,
            PositionType::GraphNode,
            PositionType::GraphNode,
            PositionType::ConvContext,
            PositionType::Query,
        ];
        let bias = computer.compute(&metrics, &position_map, 2);

        assert_eq!(bias.len(), 10); // 2 heads × 5 positions

        // Head 0: [0, 0.5, 0.5, 0.15, 0]
        assert_eq!(bias[0], 0.0); // Header
        assert!((bias[1] - 0.5).abs() < 0.01); // GraphNode: 1.0 × 0.5
        assert!((bias[2] - 0.5).abs() < 0.01); // GraphNode
        assert!((bias[3] - 0.15).abs() < 0.01); // Conv: (1-0.5) × 0.3
        assert_eq!(bias[4], 0.0); // Query

        // Head 1 same values (uniform across heads)
        assert!((bias[6] - 0.5).abs() < 0.01); // GraphNode
    }

    #[test]
    fn test_negative_reward_suppresses_graph_nodes() {
        let computer = StateBiasComputer::new(StateBiasConfig::default());
        let metrics = StateMetrics {
            reward_avg: -1.0,
            confidence: 0.8,
        };
        let position_map = vec![PositionType::GraphNode, PositionType::ConvContext];
        let bias = computer.compute(&metrics, &position_map, 1);

        // GraphNode: -1.0 × 0.5 = -0.5
        assert!((bias[0] - (-0.5)).abs() < 0.01);
        // Conv: (1-0.8) × 0.3 = 0.06
        assert!((bias[1] - 0.06).abs() < 0.01);
    }

    #[test]
    fn test_bias_clamp() {
        let computer = StateBiasComputer::new(StateBiasConfig {
            graph_weight: 5.0, // Intentionally large
            conv_weight: 5.0,
            max_bias: 1.0,
        });
        let metrics = StateMetrics {
            reward_avg: 1.0,
            confidence: 0.0,
        };
        let position_map = vec![PositionType::GraphNode, PositionType::ConvContext];
        let bias = computer.compute(&metrics, &position_map, 1);

        // Should be clamped to 1.0 (not 5.0)
        assert!((bias[0] - 1.0).abs() < 0.01);
        assert!((bias[1] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_empty_position_map() {
        let computer = StateBiasComputer::new(StateBiasConfig::default());
        let metrics = StateMetrics::default();
        let bias = computer.compute(&metrics, &[], 32);
        assert!(bias.is_empty());
    }

    #[test]
    fn test_bias_magnitude() {
        let bias = vec![0.5, -0.5, 0.5, -0.5];
        let mag = StateBiasComputer::bias_magnitude(&bias);
        assert!((mag - 0.5).abs() < 0.01);

        let zero_bias = vec![0.0; 100];
        assert_eq!(StateBiasComputer::bias_magnitude(&zero_bias), 0.0);
    }

    #[test]
    fn test_performance_4096_positions_32_heads() {
        let computer = StateBiasComputer::new(StateBiasConfig::default());
        let metrics = StateMetrics {
            reward_avg: 0.7,
            confidence: 0.3,
        };
        // Simulate realistic KV: 100 header, 500 graph nodes, 200 conv, rest query
        let mut position_map = Vec::with_capacity(4096);
        position_map.extend(std::iter::repeat(PositionType::Header).take(100));
        position_map.extend(std::iter::repeat(PositionType::GraphNode).take(500));
        position_map.extend(std::iter::repeat(PositionType::ConvContext).take(200));
        position_map.extend(std::iter::repeat(PositionType::Query).take(3296));

        let start = std::time::Instant::now();
        let bias = computer.compute(&metrics, &position_map, 32);
        let elapsed = start.elapsed();

        assert_eq!(bias.len(), 32 * 4096);
        // Performance: should be < 1ms
        assert!(
            elapsed.as_millis() < 10, // generous margin for CI
            "compute took {}ms (budget: <1ms)",
            elapsed.as_millis()
        );
    }

    // ─────────────────────────────────────────────────────────────────
    // REGRESSION tests for documented bugs
    // ─────────────────────────────────────────────────────────────────

    /// REGRESSION: State bias must produce zero-bias when reward is 0 and confidence is 0.5.
    /// Ensures no silent drift when the system starts up (default metrics).
    #[test]
    fn test_regression_default_metrics_no_drift() {
        let computer = StateBiasComputer::new(StateBiasConfig::default());
        let metrics = StateMetrics::default(); // reward=0, confidence=0.5
        let position_map = vec![
            PositionType::Header,
            PositionType::GraphNode,
            PositionType::ConvContext,
            PositionType::Query,
        ];
        let bias = computer.compute(&metrics, &position_map, 2);

        // Header and Query must ALWAYS be 0.0
        assert_eq!(bias[0], 0.0, "Header bias must be 0");
        assert_eq!(bias[3], 0.0, "Query bias must be 0");
        assert_eq!(bias[4], 0.0, "Header bias (head 1) must be 0");
        assert_eq!(bias[7], 0.0, "Query bias (head 1) must be 0");

        // GraphNode: reward=0 × graph_weight → 0.0
        assert_eq!(bias[1], 0.0, "GraphNode with reward=0 should have 0 bias");

        // ConvContext: (1-confidence) × conv_weight = (1-0.5) × 0.3 = 0.15
        assert!((bias[2] - 0.15).abs() < 0.01, "ConvContext default: got {}", bias[2]);
    }

    /// REGRESSION: Query positions must NEVER have bias applied.
    /// If query tokens get biased, the model's attention source is corrupted.
    #[test]
    fn test_regression_query_never_biased() {
        let computer = StateBiasComputer::new(StateBiasConfig {
            graph_weight: 10.0,
            conv_weight: 10.0,
            max_bias: 10.0,
        });
        let metrics = StateMetrics {
            reward_avg: 1.0,
            confidence: 0.0, // max conv bias
        };
        // All query positions
        let position_map = vec![PositionType::Query; 100];
        let bias = computer.compute(&metrics, &position_map, 32);

        for (i, &b) in bias.iter().enumerate() {
            assert_eq!(b, 0.0, "Query position bias[{i}] must be 0, got {b}");
        }
    }

    /// REGRESSION: Header positions must NEVER have bias applied.
    /// System prompt attention must be stable regardless of reward/confidence.
    #[test]
    fn test_regression_header_never_biased() {
        let computer = StateBiasComputer::new(StateBiasConfig {
            graph_weight: 10.0,
            conv_weight: 10.0,
            max_bias: 10.0,
        });
        let metrics = StateMetrics {
            reward_avg: 1.0,
            confidence: 0.0,
        };
        let position_map = vec![PositionType::Header; 50];
        let bias = computer.compute(&metrics, &position_map, 8);

        for (i, &b) in bias.iter().enumerate() {
            assert_eq!(b, 0.0, "Header position bias[{i}] must be 0, got {b}");
        }
    }

    /// REGRESSION: Negative reward must SUPPRESS graph node attention (negative bias).
    /// This is the key feedback mechanism: bad graph data → negative bias → model ignores nodes.
    #[test]
    fn test_regression_negative_reward_suppresses_graph() {
        let computer = StateBiasComputer::new(StateBiasConfig::default());
        let metrics = StateMetrics {
            reward_avg: -0.8,
            confidence: 0.5,
        };
        let position_map = vec![PositionType::GraphNode];
        let bias = computer.compute(&metrics, &position_map, 1);

        assert!(bias[0] < 0.0, "Negative reward must produce negative graph bias, got {}", bias[0]);
    }
}
