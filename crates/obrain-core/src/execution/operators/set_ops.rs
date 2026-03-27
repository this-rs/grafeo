//! Set operations: EXCEPT, INTERSECT, and OTHERWISE.
//!
//! These operators implement the GQL composite query operations for
//! combining result sets with set semantics.

use std::collections::HashSet;

use grafeo_common::types::{HashableValue, LogicalType, Value};

use super::{DataChunk, Operator, OperatorError, OperatorResult};
use crate::execution::chunk::DataChunkBuilder;

/// A hashable row key: one `HashableValue` per column.
type RowKey = Vec<HashableValue>;

/// Extracts a hashable row key from a `DataChunk`.
fn row_key(chunk: &DataChunk, row: usize) -> RowKey {
    let mut key = Vec::with_capacity(chunk.num_columns());
    for col_idx in 0..chunk.num_columns() {
        let val = chunk
            .column(col_idx)
            .and_then(|col| col.get_value(row))
            .unwrap_or(Value::Null);
        key.push(HashableValue(val));
    }
    key
}

/// Extracts the plain Values from a row key (for chunk reconstruction).
fn row_values(key: &RowKey) -> Vec<Value> {
    key.iter().map(|hv| hv.0.clone()).collect()
}

/// Materializes all rows from an operator into a vector of row keys.
fn materialize(op: &mut dyn Operator) -> Result<Vec<RowKey>, OperatorError> {
    let mut rows = Vec::new();
    while let Some(chunk) = op.next()? {
        for row in chunk.selected_indices() {
            rows.push(row_key(&chunk, row));
        }
    }
    Ok(rows)
}

/// Rebuilds a `DataChunk` from a set of row keys.
fn rows_to_chunk(rows: &[RowKey], schema: &[LogicalType]) -> DataChunk {
    if rows.is_empty() {
        return DataChunk::empty();
    }
    let mut builder = DataChunkBuilder::new(schema);
    for row in rows {
        let values = row_values(row);
        for (col_idx, val) in values.into_iter().enumerate() {
            if let Some(col) = builder.column_mut(col_idx) {
                col.push_value(val);
            }
        }
        builder.advance_row();
    }
    builder.finish()
}

/// EXCEPT operator: rows in left that are not in right.
pub struct ExceptOperator {
    left: Box<dyn Operator>,
    right: Box<dyn Operator>,
    all: bool,
    output_schema: Vec<LogicalType>,
    result: Option<Vec<RowKey>>,
    position: usize,
}

impl ExceptOperator {
    /// Creates a new EXCEPT operator.
    pub fn new(
        left: Box<dyn Operator>,
        right: Box<dyn Operator>,
        all: bool,
        output_schema: Vec<LogicalType>,
    ) -> Self {
        Self {
            left,
            right,
            all,
            output_schema,
            result: None,
            position: 0,
        }
    }

    fn compute(&mut self) -> Result<(), OperatorError> {
        let left_rows = materialize(self.left.as_mut())?;
        let right_rows = materialize(self.right.as_mut())?;

        if self.all {
            // EXCEPT ALL: for each right row, remove one matching left row
            let mut result = left_rows;
            for right_row in &right_rows {
                if let Some(pos) = result.iter().position(|r| r == right_row) {
                    result.remove(pos);
                }
            }
            self.result = Some(result);
        } else {
            // EXCEPT DISTINCT: remove all matching rows
            let right_set: HashSet<RowKey> = right_rows.into_iter().collect();
            let mut seen = HashSet::new();
            let result: Vec<RowKey> = left_rows
                .into_iter()
                .filter(|row| !right_set.contains(row) && seen.insert(row.clone()))
                .collect();
            self.result = Some(result);
        }
        Ok(())
    }
}

impl Operator for ExceptOperator {
    fn next(&mut self) -> OperatorResult {
        if self.result.is_none() {
            self.compute()?;
        }
        let rows = self
            .result
            .as_ref()
            .expect("result is Some: compute() called above");
        if self.position >= rows.len() {
            return Ok(None);
        }
        // Emit up to 1024 rows per chunk
        let end = (self.position + 1024).min(rows.len());
        let batch = &rows[self.position..end];
        self.position = end;
        if batch.is_empty() {
            Ok(None)
        } else {
            Ok(Some(rows_to_chunk(batch, &self.output_schema)))
        }
    }

    fn reset(&mut self) {
        self.left.reset();
        self.right.reset();
        self.result = None;
        self.position = 0;
    }

    fn name(&self) -> &'static str {
        "Except"
    }
}

/// INTERSECT operator: rows common to both inputs.
pub struct IntersectOperator {
    left: Box<dyn Operator>,
    right: Box<dyn Operator>,
    all: bool,
    output_schema: Vec<LogicalType>,
    result: Option<Vec<RowKey>>,
    position: usize,
}

impl IntersectOperator {
    /// Creates a new INTERSECT operator.
    pub fn new(
        left: Box<dyn Operator>,
        right: Box<dyn Operator>,
        all: bool,
        output_schema: Vec<LogicalType>,
    ) -> Self {
        Self {
            left,
            right,
            all,
            output_schema,
            result: None,
            position: 0,
        }
    }

    fn compute(&mut self) -> Result<(), OperatorError> {
        let left_rows = materialize(self.left.as_mut())?;
        let right_rows = materialize(self.right.as_mut())?;

        if self.all {
            // INTERSECT ALL: each right row matches at most one left row
            let mut remaining_right = right_rows;
            let mut result = Vec::new();
            for left_row in &left_rows {
                if let Some(pos) = remaining_right.iter().position(|r| r == left_row) {
                    result.push(left_row.clone());
                    remaining_right.remove(pos);
                }
            }
            self.result = Some(result);
        } else {
            // INTERSECT DISTINCT: rows present in both, deduplicated
            let right_set: HashSet<RowKey> = right_rows.into_iter().collect();
            let mut seen = HashSet::new();
            let result: Vec<RowKey> = left_rows
                .into_iter()
                .filter(|row| right_set.contains(row) && seen.insert(row.clone()))
                .collect();
            self.result = Some(result);
        }
        Ok(())
    }
}

impl Operator for IntersectOperator {
    fn next(&mut self) -> OperatorResult {
        if self.result.is_none() {
            self.compute()?;
        }
        let rows = self
            .result
            .as_ref()
            .expect("result is Some: compute() called above");
        if self.position >= rows.len() {
            return Ok(None);
        }
        let end = (self.position + 1024).min(rows.len());
        let batch = &rows[self.position..end];
        self.position = end;
        if batch.is_empty() {
            Ok(None)
        } else {
            Ok(Some(rows_to_chunk(batch, &self.output_schema)))
        }
    }

    fn reset(&mut self) {
        self.left.reset();
        self.right.reset();
        self.result = None;
        self.position = 0;
    }

    fn name(&self) -> &'static str {
        "Intersect"
    }
}

/// OTHERWISE operator: use left result if non-empty, otherwise use right.
pub struct OtherwiseOperator {
    left: Box<dyn Operator>,
    right: Box<dyn Operator>,
    /// Which input we are currently streaming from.
    state: OtherwiseState,
}

enum OtherwiseState {
    /// Haven't started yet, need to probe left.
    Init,
    /// Left produced rows: buffer first chunk, then stream rest of left.
    StreamingLeft(Option<DataChunk>),
    /// Left was empty: stream right.
    StreamingRight,
    /// Done.
    Done,
}

impl OtherwiseOperator {
    /// Creates a new OTHERWISE operator.
    pub fn new(left: Box<dyn Operator>, right: Box<dyn Operator>) -> Self {
        Self {
            left,
            right,
            state: OtherwiseState::Init,
        }
    }
}

impl Operator for OtherwiseOperator {
    fn next(&mut self) -> OperatorResult {
        loop {
            match &mut self.state {
                OtherwiseState::Init => {
                    // Probe left for first chunk
                    if let Some(chunk) = self.left.next()? {
                        self.state = OtherwiseState::StreamingLeft(Some(chunk));
                    } else {
                        // Left is empty, switch to right
                        self.state = OtherwiseState::StreamingRight;
                    }
                }
                OtherwiseState::StreamingLeft(buffered) => {
                    if let Some(chunk) = buffered.take() {
                        return Ok(Some(chunk));
                    }
                    // Continue streaming from left
                    match self.left.next()? {
                        Some(chunk) => return Ok(Some(chunk)),
                        None => {
                            self.state = OtherwiseState::Done;
                            return Ok(None);
                        }
                    }
                }
                OtherwiseState::StreamingRight => match self.right.next()? {
                    Some(chunk) => return Ok(Some(chunk)),
                    None => {
                        self.state = OtherwiseState::Done;
                        return Ok(None);
                    }
                },
                OtherwiseState::Done => return Ok(None),
            }
        }
    }

    fn reset(&mut self) {
        self.left.reset();
        self.right.reset();
        self.state = OtherwiseState::Init;
    }

    fn name(&self) -> &'static str {
        "Otherwise"
    }
}
