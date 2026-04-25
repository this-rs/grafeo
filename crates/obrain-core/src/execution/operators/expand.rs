//! Expand operator for relationship traversal.

use super::{Operator, OperatorError, OperatorResult};
use crate::execution::DataChunk;
use crate::graph::Direction;
use crate::graph::GraphStore;
use obrain_common::types::{EdgeId, EpochId, LogicalType, NodeId, TransactionId};
use std::sync::Arc;

/// An expand operator that traverses edges from source nodes.
///
/// For each input row containing a source node, this operator produces
/// output rows for each neighbor connected via matching edges.
pub struct ExpandOperator {
    /// The store to traverse.
    store: Arc<dyn GraphStore>,
    /// Input operator providing source nodes.
    input: Box<dyn Operator>,
    /// Index of the source node column in input.
    source_column: usize,
    /// Direction of edge traversal.
    direction: Direction,
    /// Edge type filter (empty = match all types, multiple = match any).
    edge_types: Vec<String>,
    /// Chunk capacity.
    chunk_capacity: usize,
    /// Current input chunk being processed.
    current_input: Option<DataChunk>,
    /// Current row index in the input chunk.
    current_row: usize,
    /// Current edge iterator for the current row.
    current_edges: Vec<(NodeId, EdgeId)>,
    /// Current edge index.
    current_edge_idx: usize,
    /// Whether the operator is exhausted.
    exhausted: bool,
    /// Transaction ID for MVCC visibility (None = use current epoch).
    transaction_id: Option<TransactionId>,
    /// Epoch for version visibility.
    viewing_epoch: Option<EpochId>,
    /// When true, skip versioned (MVCC) lookups even if a transaction ID is
    /// present.  Safe for read-only queries where the transaction has no
    /// pending writes, avoiding the cost of walking version chains.
    read_only: bool,
}

impl ExpandOperator {
    /// Creates a new expand operator.
    pub fn new(
        store: Arc<dyn GraphStore>,
        input: Box<dyn Operator>,
        source_column: usize,
        direction: Direction,
        edge_types: Vec<String>,
    ) -> Self {
        Self {
            store,
            input,
            source_column,
            direction,
            edge_types,
            chunk_capacity: 2048,
            current_input: None,
            current_row: 0,
            current_edges: Vec::with_capacity(16), // typical node degree
            current_edge_idx: 0,
            exhausted: false,
            transaction_id: None,
            viewing_epoch: None,
            read_only: false,
        }
    }

    /// Sets the chunk capacity.
    pub fn with_chunk_capacity(mut self, capacity: usize) -> Self {
        self.chunk_capacity = capacity;
        self
    }

    /// Sets the transaction context for MVCC visibility.
    ///
    /// When set, the expand will only traverse visible edges and nodes.
    pub fn with_transaction_context(
        mut self,
        epoch: EpochId,
        transaction_id: Option<TransactionId>,
    ) -> Self {
        self.viewing_epoch = Some(epoch);
        self.transaction_id = transaction_id;
        self
    }

    /// Marks this expand as read-only, enabling fast-path lookups.
    ///
    /// When the query has no mutations, versioned MVCC lookups (which walk
    /// version chains to find PENDING writes) can be skipped in favour of
    /// cheaper epoch-only visibility checks.
    pub fn with_read_only(mut self, read_only: bool) -> Self {
        self.read_only = read_only;
        self
    }

    /// Loads the next input chunk.
    fn load_next_input(&mut self) -> Result<bool, OperatorError> {
        match self.input.next() {
            Ok(Some(mut chunk)) => {
                // Flatten the chunk if it has a selection vector so we can use direct indexing
                chunk.flatten();
                self.current_input = Some(chunk);
                self.current_row = 0;
                self.current_edges.clear();
                self.current_edge_idx = 0;
                Ok(true)
            }
            Ok(None) => {
                self.exhausted = true;
                Ok(false)
            }
            Err(e) => Err(e),
        }
    }

    /// Loads edges for the current row.
    fn load_edges_for_current_row(&mut self) -> Result<bool, OperatorError> {
        let Some(chunk) = &self.current_input else {
            return Ok(false);
        };

        if self.current_row >= chunk.row_count() {
            return Ok(false);
        }

        let col = chunk.column(self.source_column).ok_or_else(|| {
            OperatorError::ColumnNotFound(format!("Column {} not found", self.source_column))
        })?;

        let source_id = col
            .get_node_id(self.current_row)
            .ok_or_else(|| OperatorError::Execution("Expected node ID in source column".into()))?;

        // Get visibility context.  When `read_only` is true we can skip the
        // more expensive versioned lookups because the transaction has no
        // pending writes: epoch-only visibility is sufficient and avoids
        // walking MVCC version chains.
        let epoch = self.viewing_epoch;
        let transaction_id = self.transaction_id;
        let use_versioned = !self.read_only;

        // Get edges from this node
        let edges: Vec<(NodeId, EdgeId)> = self
            .store
            .edges_from(source_id, self.direction)
            .into_iter()
            .filter(|(target_id, edge_id)| {
                // Filter by edge type if specified
                let type_matches = if self.edge_types.is_empty() {
                    true
                } else {
                    // Use versioned type lookup only when we need to see
                    // PENDING (uncommitted) edges created by this transaction.
                    let actual_type =
                        if use_versioned && let (Some(ep), Some(tx)) = (epoch, transaction_id) {
                            self.store.edge_type_versioned(*edge_id, ep, tx)
                        } else {
                            self.store.edge_type(*edge_id)
                        };
                    actual_type.is_some_and(|t| {
                        self.edge_types
                            .iter()
                            .any(|et| t.as_str().eq_ignore_ascii_case(et.as_str()))
                    })
                };

                if !type_matches {
                    return false;
                }

                // Filter by visibility if we have epoch context
                if let Some(epoch) = epoch {
                    if use_versioned && let Some(tx) = transaction_id {
                        self.store.is_edge_visible_versioned(*edge_id, epoch, tx)
                            && self.store.is_node_visible_versioned(*target_id, epoch, tx)
                    } else {
                        self.store.is_edge_visible_at_epoch(*edge_id, epoch)
                            && self.store.is_node_visible_at_epoch(*target_id, epoch)
                    }
                } else {
                    true
                }
            })
            .collect();

        self.current_edges = edges;
        self.current_edge_idx = 0;
        Ok(true)
    }
}

impl Operator for ExpandOperator {
    fn next(&mut self) -> OperatorResult {
        if self.exhausted {
            return Ok(None);
        }

        // Build output schema: preserve all input columns + edge + target
        // We need to build this dynamically based on input schema
        if self.current_input.is_none() {
            if !self.load_next_input()? {
                return Ok(None);
            }
            self.load_edges_for_current_row()?;
        }
        let input_chunk = self.current_input.as_ref().expect("input loaded above");

        // Build schema: [input_columns..., edge, target]
        let input_col_count = input_chunk.column_count();
        let mut schema: Vec<LogicalType> = (0..input_col_count)
            .map(|i| {
                input_chunk
                    .column(i)
                    .map_or(LogicalType::Any, |c| c.data_type().clone())
            })
            .collect();
        schema.push(LogicalType::Edge);
        schema.push(LogicalType::Node);

        let mut chunk = DataChunk::with_capacity(&schema, self.chunk_capacity);
        let mut count = 0;

        while count < self.chunk_capacity {
            // If we need a new input chunk
            if self.current_input.is_none() {
                if !self.load_next_input()? {
                    break;
                }
                self.load_edges_for_current_row()?;
            }

            // If we've exhausted edges for current row, move to next row
            while self.current_edge_idx >= self.current_edges.len() {
                self.current_row += 1;

                // If we've exhausted the current input chunk, get next one
                if self.current_row >= self.current_input.as_ref().map_or(0, |c| c.row_count()) {
                    self.current_input = None;
                    if !self.load_next_input()? {
                        // No more input chunks
                        if count > 0 {
                            chunk.set_count(count);
                            return Ok(Some(chunk));
                        }
                        return Ok(None);
                    }
                }

                self.load_edges_for_current_row()?;
            }

            // Get the current edge
            let (target_id, edge_id) = self.current_edges[self.current_edge_idx];

            // Copy all input columns to output
            let input = self.current_input.as_ref().expect("input loaded above");
            for col_idx in 0..input_col_count {
                if let Some(input_col) = input.column(col_idx)
                    && let Some(output_col) = chunk.column_mut(col_idx)
                {
                    // Use copy_row_to which preserves NodeId/EdgeId types
                    input_col.copy_row_to(self.current_row, output_col);
                }
            }

            // Add edge column
            if let Some(col) = chunk.column_mut(input_col_count) {
                col.push_edge_id(edge_id);
            }

            // Add target node column
            if let Some(col) = chunk.column_mut(input_col_count + 1) {
                col.push_node_id(target_id);
            }

            count += 1;
            self.current_edge_idx += 1;
        }

        if count > 0 {
            chunk.set_count(count);
            Ok(Some(chunk))
        } else {
            Ok(None)
        }
    }

    fn reset(&mut self) {
        self.input.reset();
        self.current_input = None;
        self.current_row = 0;
        self.current_edges.clear();
        self.current_edge_idx = 0;
        self.exhausted = false;
    }

    fn name(&self) -> &'static str {
        "Expand"
    }
}

// Tests relocated to `crates/obrain-substrate/tests/operators_expand.rs`
// (T17 Step 3 W2 Class-2 migration — decision `b1dfe229`). obrain-core cannot
// take obrain-substrate as a dev-dep (dev-dep cycle, gotcha `598dda40`), so
// the LPG-era fixtures are rebuilt against SubstrateStore in an integration
// test of obrain-substrate.
#[cfg(test)]
mod tests {}
