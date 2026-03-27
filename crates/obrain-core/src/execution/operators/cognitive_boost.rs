//! CognitiveBoost operator — re-ranks results using an activation map.
//!
//! Given a child operator producing rows that include a node-ID column,
//! `CognitiveBoostOperator` appends (or replaces) a score column whose
//! value is derived from a pre-computed `ActivationMap`.
//!
//! Two modes are supported:
//! - **Additive**: `final_score = original_score + activation`
//! - **Multiplicative**: `final_score = original_score × activation`
//!
//! When no score column exists yet, the activation value itself is used.

use std::collections::HashMap;

use obrain_common::types::{LogicalType, Value};

use super::{Operator, OperatorResult};
use crate::execution::DataChunk;
use crate::execution::vector::ValueVector;

// ---------------------------------------------------------------------------
// ActivationMap — re-exported from obrain-cognitive when available,
// but defined locally to avoid obrain-core depending on obrain-cognitive.
// ---------------------------------------------------------------------------

/// Node activation map: node ID (u64) → activation energy (f64).
///
/// Mirrors `obrain_cognitive::ActivationMap` but avoids a circular dependency.
pub type ActivationMap = HashMap<u64, f64>;

/// How the activation score is combined with the existing row.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoostMode {
    /// `final = original + activation`
    Additive,
    /// `final = original × activation`
    Multiplicative,
    /// Activation value replaces the score (or is appended as new column).
    Replace,
}

// ---------------------------------------------------------------------------
// CognitiveBoostOperator
// ---------------------------------------------------------------------------

/// Operator that boosts query results based on an activation map.
///
/// For each input row, the operator looks up the node ID from column
/// `node_id_column` in the activation map. The resulting activation
/// value is combined with an optional existing score column according
/// to `mode`. The output schema is the input schema plus an appended
/// `_cognitive_score` column of type `Float64`.
pub struct CognitiveBoostOperator {
    /// Child operator providing input rows.
    child: Box<dyn Operator>,
    /// Pre-computed activation energies keyed by node ID (as u64).
    activations: ActivationMap,
    /// Column index containing the node ID to look up.
    node_id_column: usize,
    /// Optional column index of an existing score to combine with.
    score_column: Option<usize>,
    /// Combination mode.
    mode: BoostMode,
    /// Default activation when a node is not in the map.
    default_activation: f64,
}

impl CognitiveBoostOperator {
    /// Creates a new cognitive boost operator.
    pub fn new(
        child: Box<dyn Operator>,
        activations: ActivationMap,
        node_id_column: usize,
        score_column: Option<usize>,
        mode: BoostMode,
    ) -> Self {
        Self {
            child,
            activations,
            node_id_column,
            score_column,
            mode,
            default_activation: 0.0,
        }
    }

    /// Sets the default activation for nodes not found in the map.
    #[must_use]
    pub fn with_default_activation(mut self, default: f64) -> Self {
        self.default_activation = default;
        self
    }

    /// Computes the boosted score for a single row.
    fn compute_score(&self, activation: f64, existing_score: Option<f64>) -> f64 {
        match self.mode {
            BoostMode::Additive => existing_score.unwrap_or(0.0) + activation,
            BoostMode::Multiplicative => existing_score.unwrap_or(1.0) * activation,
            BoostMode::Replace => activation,
        }
    }
}

impl Operator for CognitiveBoostOperator {
    fn next(&mut self) -> OperatorResult {
        let Some(chunk) = self.child.next()? else {
            return Ok(None);
        };

        let row_count = chunk.row_count();
        if row_count == 0 {
            return Ok(Some(chunk));
        }

        // Build new score column
        let mut score_vec = ValueVector::with_type(LogicalType::Float64);

        for row_idx in 0..row_count {
            // Extract node ID from the designated column
            let node_id_val = chunk
                .column(self.node_id_column)
                .and_then(|c| c.get_value(row_idx));

            let node_id: u64 = match node_id_val {
                Some(Value::Int64(id)) => id as u64,
                Some(Value::Null) => 0,
                _ => 0,
            };

            let activation = self
                .activations
                .get(&node_id)
                .copied()
                .unwrap_or(self.default_activation);

            // Extract existing score if configured
            let existing_score = self.score_column.and_then(|col| {
                chunk.column(col).and_then(|c| match c.get_value(row_idx) {
                    Some(Value::Float64(v)) => Some(v),
                    Some(Value::Int64(v)) => Some(v as f64),
                    _ => None,
                })
            });

            let final_score = self.compute_score(activation, existing_score);
            score_vec.push_float64(final_score);
        }

        // Build output: all original columns + score column
        let col_count = chunk.column_count();
        let mut columns = Vec::with_capacity(col_count + 1);
        for col_idx in 0..col_count {
            if let Some(col) = chunk.column(col_idx) {
                columns.push(col.clone());
            }
        }
        columns.push(score_vec);

        Ok(Some(DataChunk::new(columns)))
    }

    fn reset(&mut self) {
        self.child.reset();
    }

    fn name(&self) -> &'static str {
        "CognitiveBoost"
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    /// Helper: creates a simple operator producing one chunk with node IDs.
    fn make_test_input(node_ids: &[i64]) -> Box<dyn Operator> {
        let mut col = ValueVector::with_type(LogicalType::Int64);
        for &id in node_ids {
            col.push_int64(id);
        }
        let chunk = DataChunk::new(vec![col]);
        Box::new(SingleChunkOperator { chunk: Some(chunk) })
    }

    struct SingleChunkOperator {
        chunk: Option<DataChunk>,
    }

    impl Operator for SingleChunkOperator {
        fn next(&mut self) -> OperatorResult {
            Ok(self.chunk.take())
        }
        fn reset(&mut self) {}
        fn name(&self) -> &'static str {
            "SingleChunk"
        }
    }

    #[test]
    fn test_additive_boost() {
        let mut activations = ActivationMap::new();
        activations.insert(1, 0.8);
        activations.insert(2, 0.2);
        // Node 3 not in map → default 0.0

        let input = make_test_input(&[1, 2, 3]);
        let mut op = CognitiveBoostOperator::new(
            input,
            activations,
            0,    // node_id_column
            None, // no existing score
            BoostMode::Additive,
        );

        let chunk = op.next().unwrap().unwrap();
        // Should have 2 columns: original node IDs + cognitive score
        assert_eq!(chunk.column_count(), 2);

        // Check scores
        let score_col = chunk.column(1).unwrap();
        assert_eq!(score_col.get_float64(0), Some(0.8)); // node 1
        assert_eq!(score_col.get_float64(1), Some(0.2)); // node 2
        assert_eq!(score_col.get_float64(2), Some(0.0)); // node 3

        // Next call should return None
        assert!(op.next().unwrap().is_none());
    }

    #[test]
    fn test_multiplicative_boost() {
        let mut activations = ActivationMap::new();
        activations.insert(1, 0.8);
        activations.insert(2, 0.2);
        activations.insert(3, 0.0);

        // Input with node IDs and an existing score column
        let mut id_col = ValueVector::with_type(LogicalType::Int64);
        id_col.push_int64(1);
        id_col.push_int64(2);
        id_col.push_int64(3);

        let mut score_col = ValueVector::with_type(LogicalType::Float64);
        score_col.push_float64(10.0);
        score_col.push_float64(10.0);
        score_col.push_float64(10.0);

        let chunk = DataChunk::new(vec![id_col, score_col]);
        let input = Box::new(SingleChunkOperator { chunk: Some(chunk) });

        let mut op = CognitiveBoostOperator::new(
            input,
            activations,
            0,       // node_id_column
            Some(1), // existing score column
            BoostMode::Multiplicative,
        );

        let result = op.next().unwrap().unwrap();
        // 3 columns: node ID, original score, boosted score
        assert_eq!(result.column_count(), 3);

        let boosted = result.column(2).unwrap();
        assert_eq!(boosted.get_float64(0), Some(8.0)); // 10 * 0.8
        assert_eq!(boosted.get_float64(1), Some(2.0)); // 10 * 0.2
        assert_eq!(boosted.get_float64(2), Some(0.0)); // 10 * 0.0
    }

    #[test]
    fn test_replace_mode() {
        let mut activations = ActivationMap::new();
        activations.insert(1, 5.5);

        let input = make_test_input(&[1]);
        let mut op = CognitiveBoostOperator::new(input, activations, 0, None, BoostMode::Replace);

        let chunk = op.next().unwrap().unwrap();
        let score = chunk.column(1).unwrap();
        assert_eq!(score.get_float64(0), Some(5.5));
    }

    #[test]
    fn test_empty_input() {
        let activations = ActivationMap::new();
        let input = Box::new(SingleChunkOperator { chunk: None });
        let mut op = CognitiveBoostOperator::new(input, activations, 0, None, BoostMode::Additive);

        assert!(op.next().unwrap().is_none());
    }

    #[test]
    fn test_default_activation() {
        let activations = ActivationMap::new(); // empty
        let input = make_test_input(&[99]);
        let mut op = CognitiveBoostOperator::new(input, activations, 0, None, BoostMode::Replace)
            .with_default_activation(0.42);

        let chunk = op.next().unwrap().unwrap();
        let score = chunk.column(1).unwrap();
        assert_eq!(score.get_float64(0), Some(0.42));
    }
}
