//! Horizontal aggregation operator for per-row aggregation over list-valued columns.
//!
//! Used for GE09 (ISO GQL horizontal aggregation): aggregating over group-list
//! variables from variable-length path patterns. For each input row, reads a list
//! of entity IDs, accesses a property on each entity, computes the aggregate, and
//! appends the scalar result as a new column.

use std::sync::Arc;

use obrain_common::types::{EdgeId, LogicalType, NodeId, PropertyKey, Value};

use super::accumulator::AggregateFunction;
use super::aggregate::AggregateState;
use super::{Operator, OperatorResult};
use crate::execution::DataChunk;
use crate::execution::vector::ValueVector;
use crate::graph::traits::GraphStore;

/// Whether the horizontal aggregate operates on edges or nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntityKind {
    /// Aggregate over edges in a path.
    Edge,
    /// Aggregate over nodes in a path.
    Node,
}

/// Per-row aggregation over a list-valued column.
///
/// For each input row:
/// 1. Reads a `Value::List` from `list_column_idx` (entity IDs from a path)
/// 2. For each entity ID, looks up `property` via the graph store
/// 3. Feeds property values through an `AggregateState`
/// 4. Finalizes and appends the result as a new column
pub struct HorizontalAggregateOperator {
    /// Child operator.
    child: Box<dyn Operator>,
    /// Column index containing the list of entity IDs.
    list_column_idx: usize,
    /// Whether entities are edges or nodes.
    entity_kind: EntityKind,
    /// The aggregate function to compute per row.
    function: AggregateFunction,
    /// Property name to access on each entity.
    property: String,
    /// Graph store for property lookups.
    store: Arc<dyn GraphStore>,
    /// Number of input columns (to know where to append the result).
    input_column_count: usize,
}

impl HorizontalAggregateOperator {
    /// Creates a new horizontal aggregate operator.
    pub fn new(
        child: Box<dyn Operator>,
        list_column_idx: usize,
        entity_kind: EntityKind,
        function: AggregateFunction,
        property: String,
        store: Arc<dyn GraphStore>,
        input_column_count: usize,
    ) -> Self {
        Self {
            child,
            list_column_idx,
            entity_kind,
            function,
            property,
            store,
            input_column_count,
        }
    }

    /// Looks up a property value for an entity ID.
    fn get_property_value(&self, entity_value: &Value) -> Option<Value> {
        let prop_key = PropertyKey::new(&self.property);
        match self.entity_kind {
            EntityKind::Edge => {
                let id = match entity_value {
                    Value::Int64(i) => EdgeId(*i as u64),
                    _ => return None,
                };
                self.store.get_edge_property(id, &prop_key)
            }
            EntityKind::Node => {
                let id = match entity_value {
                    Value::Int64(i) => NodeId(*i as u64),
                    _ => return None,
                };
                self.store.get_node_property(id, &prop_key)
            }
        }
    }
}

impl Operator for HorizontalAggregateOperator {
    fn next(&mut self) -> OperatorResult {
        let Some(input) = self.child.next()? else {
            return Ok(None);
        };

        // Build output columns: copy input columns + one new aggregate result column
        let mut output_columns: Vec<ValueVector> = (0..self.input_column_count)
            .map(|_| ValueVector::with_capacity(LogicalType::Any, input.row_count()))
            .collect();
        let mut result_column = ValueVector::with_capacity(LogicalType::Float64, input.row_count());

        // Collect selected indices since the iterator can only be consumed once
        let rows: Vec<usize> = input.selected_indices().collect();

        for row in rows {
            // Copy all input columns to output
            for col_idx in 0..self.input_column_count {
                let value = input
                    .column(col_idx)
                    .and_then(|c| c.get_value(row))
                    .unwrap_or(Value::Null);
                output_columns[col_idx].push_value(value);
            }

            // Read the list column and compute horizontal aggregate
            let agg_result = if let Some(Value::List(list)) = input
                .column(self.list_column_idx)
                .and_then(|c| c.get_value(row))
            {
                let mut state = AggregateState::new(self.function, false, None, None);
                for entity_val in list.iter() {
                    let prop_val = self.get_property_value(entity_val);
                    if prop_val.is_some() && !matches!(prop_val, Some(Value::Null)) {
                        state.update(prop_val);
                    }
                }
                state.finalize()
            } else {
                Value::Null
            };

            result_column.push_value(agg_result);
        }

        output_columns.push(result_column);
        Ok(Some(DataChunk::new(output_columns)))
    }

    fn reset(&mut self) {
        self.child.reset();
    }

    fn name(&self) -> &'static str {
        "HorizontalAggregate"
    }
}

// Tests relocated to `crates/obrain-substrate/tests/operators_horizontal_aggregate.rs`
// (T17 Step 3 W2 Class-2 migration — decision `b1dfe229`). The substrate
// backend cannot be used as a dev-dependency of obrain-core without producing
// two distinct compilation units (gotcha `598dda40`); the forward direction
// (substrate → core) is cycle-free, so the 10 integration tests for this
// operator live there now. Run them with:
//
// ```bash
// cargo test -p obrain-substrate --test operators_horizontal_aggregate
// ```
