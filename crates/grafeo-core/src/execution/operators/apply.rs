//! Apply operator (lateral join / correlated subquery).
//!
//! For each row from the outer input, the inner subplan is reset and executed.
//! Results are the concatenation of outer columns with inner columns.
//!
//! This operator is the backend for:
//! - Cypher: `CALL { subquery }`
//! - GQL: `VALUE { subquery }`
//! - Pattern comprehensions (with a Collect aggregate wrapper)

use std::sync::Arc;

use grafeo_common::types::{LogicalType, Value};

use super::parameter_scan::ParameterState;
use super::{DataChunk, Operator, OperatorResult};
use crate::execution::vector::ValueVector;

/// Apply (lateral join) operator.
///
/// Evaluates `inner` once for each row of `outer`. The result schema is
/// `outer_columns ++ inner_columns`. If the inner plan produces zero rows
/// for a given outer row, that outer row is omitted (inner join semantics).
///
/// When `param_state` is set, outer row values for the specified column indices
/// are injected into the shared [`ParameterState`] before each inner execution,
/// allowing the inner plan's [`ParameterScanOperator`](super::ParameterScanOperator) to read them.
pub struct ApplyOperator {
    outer: Box<dyn Operator>,
    inner: Box<dyn Operator>,
    /// Shared parameter state for correlated subqueries.
    param_state: Option<Arc<ParameterState>>,
    /// Indices of outer columns to inject into the inner plan.
    param_col_indices: Vec<usize>,
    /// Buffered outer rows waiting to be combined with inner results.
    state: ApplyState,
}

enum ApplyState {
    /// Pull next outer chunk, process row-by-row.
    Init,
    /// Processing a chunk of outer rows. `outer_chunk` is the current batch,
    /// `outer_row` is the next row index to process.
    Processing {
        outer_chunk: DataChunk,
        outer_row: usize,
        /// Accumulated output rows (combined outer + inner).
        output: Vec<Vec<Value>>,
    },
    /// All outer input exhausted.
    Done,
}

impl ApplyOperator {
    /// Creates a new Apply operator (uncorrelated: no parameter injection).
    pub fn new(outer: Box<dyn Operator>, inner: Box<dyn Operator>) -> Self {
        Self {
            outer,
            inner,
            param_state: None,
            param_col_indices: Vec::new(),
            state: ApplyState::Init,
        }
    }

    /// Creates a correlated Apply operator that injects outer row values.
    ///
    /// `param_state` is shared with a [`ParameterScanOperator`](super::ParameterScanOperator) in the inner plan.
    /// `param_col_indices` specifies which outer columns to inject (by index).
    pub fn new_correlated(
        outer: Box<dyn Operator>,
        inner: Box<dyn Operator>,
        param_state: Arc<ParameterState>,
        param_col_indices: Vec<usize>,
    ) -> Self {
        Self {
            outer,
            inner,
            param_state: Some(param_state),
            param_col_indices,
            state: ApplyState::Init,
        }
    }

    /// Extracts all values from a single row of a DataChunk.
    fn extract_row(chunk: &DataChunk, row: usize) -> Vec<Value> {
        let mut values = Vec::with_capacity(chunk.num_columns());
        for col_idx in 0..chunk.num_columns() {
            let val = chunk
                .column(col_idx)
                .and_then(|col| col.get_value(row))
                .unwrap_or(Value::Null);
            values.push(val);
        }
        values
    }

    /// Builds a DataChunk from accumulated rows.
    fn build_chunk(rows: &[Vec<Value>]) -> DataChunk {
        if rows.is_empty() {
            return DataChunk::empty();
        }
        let num_cols = rows[0].len();
        let mut columns: Vec<ValueVector> = (0..num_cols)
            .map(|_| ValueVector::with_capacity(LogicalType::Any, rows.len()))
            .collect();

        for row in rows {
            for (col_idx, val) in row.iter().enumerate() {
                if col_idx < columns.len() {
                    columns[col_idx].push_value(val.clone());
                }
            }
        }
        DataChunk::new(columns)
    }
}

impl Operator for ApplyOperator {
    fn next(&mut self) -> OperatorResult {
        loop {
            match &mut self.state {
                ApplyState::Init => match self.outer.next()? {
                    Some(chunk) => {
                        self.state = ApplyState::Processing {
                            outer_chunk: chunk,
                            outer_row: 0,
                            output: Vec::new(),
                        };
                    }
                    None => {
                        self.state = ApplyState::Done;
                        return Ok(None);
                    }
                },
                ApplyState::Processing {
                    outer_chunk,
                    outer_row,
                    output,
                } => {
                    let selected: Vec<usize> = outer_chunk.selected_indices().collect();
                    while *outer_row < selected.len() {
                        let row = selected[*outer_row];
                        let outer_values = Self::extract_row(outer_chunk, row);

                        // Inject outer values into the inner plan's parameter state
                        if let Some(ref param_state) = self.param_state {
                            let injected: Vec<Value> = self
                                .param_col_indices
                                .iter()
                                .map(|&idx| outer_values.get(idx).cloned().unwrap_or(Value::Null))
                                .collect();
                            param_state.set_values(injected);
                        }

                        // Reset and run inner plan for this outer row
                        self.inner.reset();
                        while let Some(inner_chunk) = self.inner.next()? {
                            for inner_row in inner_chunk.selected_indices() {
                                let inner_values = Self::extract_row(&inner_chunk, inner_row);
                                let mut combined = outer_values.clone();
                                combined.extend(inner_values);
                                output.push(combined);
                            }
                        }

                        *outer_row += 1;

                        // Flush when we have enough rows
                        if output.len() >= 1024 {
                            let chunk = Self::build_chunk(output);
                            output.clear();
                            return Ok(Some(chunk));
                        }
                    }

                    // Finished this outer chunk; flush any remaining output
                    if !output.is_empty() {
                        let chunk = Self::build_chunk(output);
                        output.clear();
                        self.state = ApplyState::Init;
                        return Ok(Some(chunk));
                    }

                    // Move to next outer chunk
                    self.state = ApplyState::Init;
                }
                ApplyState::Done => return Ok(None),
            }
        }
    }

    fn reset(&mut self) {
        self.outer.reset();
        self.inner.reset();
        self.state = ApplyState::Init;
    }

    fn name(&self) -> &'static str {
        "Apply"
    }
}
