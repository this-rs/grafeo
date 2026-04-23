//! Integration tests for `obrain_core::execution::operators::filter::*`
//! (`FilterOperator`, `ExpressionPredicate`, `FilterExpression`, etc.) against
//! the substrate backend.
//!
//! Relocated from `crates/obrain-core/src/execution/operators/filter.rs`'s
//! in-crate `#[cfg(test)] mod tests` block as part of T17 W4.p4. Same rationale
//! as the `operators_*` migration (see note tagged `t17 w4.p4 migration-pattern`):
//! a substrate-backed fixture cannot live inside `obrain-core` because the
//! dev-dep cycle (gotcha `598dda40-a186-4be3-97f3-c75053af4e6e`) produces
//! two distinct compilation units of `obrain-core`, breaking the
//! `Arc<SubstrateStore> as Arc<dyn GraphStore>` trait cast.
//!
//! The `CompareOp` enum and `ComparisonPredicate` struct below were previously
//! defined as `#[cfg(test)] pub(crate)` helpers inside `filter.rs`. They move
//! here as file-local test helpers (they were never part of the public or
//! crate-internal API — only test scaffolding).
//!
//! Run with:
//!
//! ```bash
//! cargo test -p obrain-substrate --test operators_filter
//! ```

#![allow(clippy::too_many_lines)]
#![allow(clippy::cognitive_complexity)]

use std::collections::HashMap;
use std::sync::Arc;

use obrain_common::types::{LogicalType, PropertyKey, Value};
use obrain_core::execution::chunk::DataChunkBuilder;
use obrain_core::execution::operators::{
    BinaryFilterOp, ExpressionPredicate, FilterExpression, FilterOperator, ListPredicateKind,
    Operator, OperatorResult, Predicate, UnaryFilterOp,
};
use obrain_core::execution::{ChunkZoneHints, DataChunk};
use obrain_core::graph::GraphStore;
use obrain_substrate::SubstrateStore;

// ── Test-local helpers (relocated from filter.rs cfg(test)) ─────────────

/// A comparison operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CompareOp {
    /// Equal.
    Eq,
    /// Not equal.
    Ne,
    /// Less than.
    Lt,
    /// Less than or equal.
    Le,
    /// Greater than.
    Gt,
    /// Greater than or equal.
    Ge,
}

/// A simple comparison predicate.
struct ComparisonPredicate {
    /// Column index to compare.
    column: usize,
    /// Comparison operator.
    op: CompareOp,
    /// Value to compare against.
    value: Value,
}

impl ComparisonPredicate {
    /// Creates a new comparison predicate.
    fn new(column: usize, op: CompareOp, value: Value) -> Self {
        Self { column, op, value }
    }
}

impl Predicate for ComparisonPredicate {
    fn evaluate(&self, chunk: &DataChunk, row: usize) -> bool {
        let Some(col) = chunk.column(self.column) else {
            return false;
        };

        let Some(cell_value) = col.get_value(row) else {
            return false;
        };

        match (&cell_value, &self.value) {
            (Value::Int64(a), Value::Int64(b)) => match self.op {
                CompareOp::Eq => a == b,
                CompareOp::Ne => a != b,
                CompareOp::Lt => a < b,
                CompareOp::Le => a <= b,
                CompareOp::Gt => a > b,
                CompareOp::Ge => a >= b,
            },
            (Value::Float64(a), Value::Float64(b)) => match self.op {
                CompareOp::Eq => (a - b).abs() < f64::EPSILON,
                CompareOp::Ne => (a - b).abs() >= f64::EPSILON,
                CompareOp::Lt => a < b,
                CompareOp::Le => a <= b,
                CompareOp::Gt => a > b,
                CompareOp::Ge => a >= b,
            },
            (Value::String(a), Value::String(b)) => match self.op {
                CompareOp::Eq => a == b,
                CompareOp::Ne => a != b,
                CompareOp::Lt => a < b,
                CompareOp::Le => a <= b,
                CompareOp::Gt => a > b,
                CompareOp::Ge => a >= b,
            },
            // Cross-type Int64/Float64 coercion
            (Value::Int64(a), Value::Float64(b)) => {
                let a = *a as f64;
                match self.op {
                    CompareOp::Eq => (a - b).abs() < f64::EPSILON,
                    CompareOp::Ne => (a - b).abs() >= f64::EPSILON,
                    CompareOp::Lt => a < *b,
                    CompareOp::Le => a <= *b,
                    CompareOp::Gt => a > *b,
                    CompareOp::Ge => a >= *b,
                }
            }
            (Value::Float64(a), Value::Int64(b)) => {
                let b = *b as f64;
                match self.op {
                    CompareOp::Eq => (a - b).abs() < f64::EPSILON,
                    CompareOp::Ne => (a - b).abs() >= f64::EPSILON,
                    CompareOp::Lt => *a < b,
                    CompareOp::Le => *a <= b,
                    CompareOp::Gt => *a > b,
                    CompareOp::Ge => *a >= b,
                }
            }
            (Value::Bool(a), Value::Bool(b)) => match self.op {
                CompareOp::Eq => a == b,
                CompareOp::Ne => a != b,
                _ => false, // Ordering on booleans doesn't make sense
            },
            _ => false, // Type mismatch
        }
    }

    fn might_match_chunk(&self, hints: &ChunkZoneHints) -> bool {
        let Some(zone_map) = hints.column_hints.get(&self.column) else {
            return true; // No zone map for this column = conservative
        };

        match self.op {
            CompareOp::Eq => zone_map.might_contain_equal(&self.value),
            CompareOp::Ne => true, // Ne is always conservative (might have non-matching values)
            CompareOp::Lt => zone_map.might_contain_less_than(&self.value, false),
            CompareOp::Le => zone_map.might_contain_less_than(&self.value, true),
            CompareOp::Gt => zone_map.might_contain_greater_than(&self.value, false),
            CompareOp::Ge => zone_map.might_contain_greater_than(&self.value, true),
        }
    }
}

// ── Tests ─────────────────────────────────────────────────

struct MockScanOperator {
    chunks: Vec<DataChunk>,
    position: usize,
}

impl Operator for MockScanOperator {
    fn next(&mut self) -> OperatorResult {
        if self.position < self.chunks.len() {
            let chunk = std::mem::replace(&mut self.chunks[self.position], DataChunk::empty());
            self.position += 1;
            Ok(Some(chunk))
        } else {
            Ok(None)
        }
    }

    fn reset(&mut self) {
        self.position = 0;
    }

    fn name(&self) -> &'static str {
        "MockScan"
    }
}

#[test]
fn test_filter_comparison() {
    // Create a chunk with values [10, 20, 30, 40, 50]
    let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
    for i in 1..=5 {
        builder.column_mut(0).unwrap().push_int64(i * 10);
        builder.advance_row();
    }
    let chunk = builder.finish();

    let mock_scan = MockScanOperator {
        chunks: vec![chunk],
        position: 0,
    };

    // Filter for values > 25
    let predicate = ComparisonPredicate::new(0, CompareOp::Gt, Value::Int64(25));
    let mut filter = FilterOperator::new(Box::new(mock_scan), Box::new(predicate));

    let result = filter.next().unwrap().unwrap();
    // Should have 30, 40, 50 (3 values)
    assert_eq!(result.row_count(), 3);
}

#[test]
fn test_regex_operator() {
    
    // Create a store and expression predicate to test regex
    let store: Arc<dyn GraphStore> = Arc::new(SubstrateStore::open_tempfile().unwrap());
    let variable_columns = HashMap::new();

    // Create predicate to test "Smith" =~ ".*Smith$" (should match)
    let predicate = ExpressionPredicate::new(
        FilterExpression::Binary {
            left: Box::new(FilterExpression::Literal(Value::String(
                "John Smith".into(),
            ))),
            op: BinaryFilterOp::Regex,
            right: Box::new(FilterExpression::Literal(Value::String(".*Smith$".into()))),
        },
        variable_columns.clone(),
        Arc::clone(&store),
    );

    // Create a minimal chunk for evaluation
    let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
    let chunk = builder.finish();

    // Should match
    assert!(predicate.evaluate(&chunk, 0));

    // Test non-matching pattern
    let predicate_no_match = ExpressionPredicate::new(
        FilterExpression::Binary {
            left: Box::new(FilterExpression::Literal(Value::String("John Doe".into()))),
            op: BinaryFilterOp::Regex,
            right: Box::new(FilterExpression::Literal(Value::String(".*Smith$".into()))),
        },
        variable_columns,
        store,
    );

    // Should not match
    assert!(!predicate_no_match.evaluate(&chunk, 0));
}

#[test]
fn test_pow_operator() {
    
    let store: Arc<dyn GraphStore> = Arc::new(SubstrateStore::open_tempfile().unwrap());
    let variable_columns = HashMap::new();

    // Create a minimal chunk for evaluation
    let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
    let chunk = builder.finish();

    // Create predicate to test 2^3 = 8.0
    let predicate = ExpressionPredicate::new(
        FilterExpression::Binary {
            left: Box::new(FilterExpression::Binary {
                left: Box::new(FilterExpression::Literal(Value::Int64(2))),
                op: BinaryFilterOp::Pow,
                right: Box::new(FilterExpression::Literal(Value::Int64(3))),
            }),
            op: BinaryFilterOp::Eq,
            right: Box::new(FilterExpression::Literal(Value::Float64(8.0))),
        },
        variable_columns.clone(),
        Arc::clone(&store),
    );

    // 2^3 should equal 8.0
    assert!(predicate.evaluate(&chunk, 0));

    // Test with floats: 2.5^2.0 = 6.25
    let predicate_float = ExpressionPredicate::new(
        FilterExpression::Binary {
            left: Box::new(FilterExpression::Binary {
                left: Box::new(FilterExpression::Literal(Value::Float64(2.5))),
                op: BinaryFilterOp::Pow,
                right: Box::new(FilterExpression::Literal(Value::Float64(2.0))),
            }),
            op: BinaryFilterOp::Eq,
            right: Box::new(FilterExpression::Literal(Value::Float64(6.25))),
        },
        variable_columns,
        store,
    );

    assert!(predicate_float.evaluate(&chunk, 0));
}

#[test]
fn test_map_expression() {
    
    let store: Arc<dyn GraphStore> = Arc::new(SubstrateStore::open_tempfile().unwrap());
    let variable_columns = HashMap::new();

    // Create a minimal chunk for evaluation
    let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
    let chunk = builder.finish();

    // Create map {name: 'Alix', age: 30}
    let predicate = ExpressionPredicate::new(
        FilterExpression::Map(vec![
            (
                "name".to_string(),
                FilterExpression::Literal(Value::String("Alix".into())),
            ),
            (
                "age".to_string(),
                FilterExpression::Literal(Value::Int64(30)),
            ),
        ]),
        variable_columns,
        store,
    );

    // Evaluate the map expression
    let result = predicate.eval_at(&chunk, 0);
    assert!(result.is_some());

    if let Some(Value::Map(m)) = result {
        assert_eq!(
            m.get(&PropertyKey::new("name")),
            Some(&Value::String("Alix".into()))
        );
        assert_eq!(m.get(&PropertyKey::new("age")), Some(&Value::Int64(30)));
    } else {
        panic!("Expected Map value");
    }
}

#[test]
fn test_index_access_list() {
    
    let store: Arc<dyn GraphStore> = Arc::new(SubstrateStore::open_tempfile().unwrap());
    let variable_columns = HashMap::new();

    // Create a minimal chunk for evaluation
    let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
    let chunk = builder.finish();

    // Test [1, 2, 3][1] = 2
    let predicate = ExpressionPredicate::new(
        FilterExpression::Binary {
            left: Box::new(FilterExpression::IndexAccess {
                base: Box::new(FilterExpression::List(vec![
                    FilterExpression::Literal(Value::Int64(1)),
                    FilterExpression::Literal(Value::Int64(2)),
                    FilterExpression::Literal(Value::Int64(3)),
                ])),
                index: Box::new(FilterExpression::Literal(Value::Int64(1))),
            }),
            op: BinaryFilterOp::Eq,
            right: Box::new(FilterExpression::Literal(Value::Int64(2))),
        },
        variable_columns.clone(),
        Arc::clone(&store),
    );

    assert!(predicate.evaluate(&chunk, 0));

    // Test negative indexing: [1, 2, 3][-1] = 3
    let predicate_neg = ExpressionPredicate::new(
        FilterExpression::Binary {
            left: Box::new(FilterExpression::IndexAccess {
                base: Box::new(FilterExpression::List(vec![
                    FilterExpression::Literal(Value::Int64(1)),
                    FilterExpression::Literal(Value::Int64(2)),
                    FilterExpression::Literal(Value::Int64(3)),
                ])),
                index: Box::new(FilterExpression::Literal(Value::Int64(-1))),
            }),
            op: BinaryFilterOp::Eq,
            right: Box::new(FilterExpression::Literal(Value::Int64(3))),
        },
        variable_columns,
        store,
    );

    assert!(predicate_neg.evaluate(&chunk, 0));
}

#[test]
fn test_slice_access() {
    
    let store: Arc<dyn GraphStore> = Arc::new(SubstrateStore::open_tempfile().unwrap());
    let variable_columns = HashMap::new();

    // Create a minimal chunk for evaluation
    let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
    let chunk = builder.finish();

    // Test [1, 2, 3, 4, 5][1..3] should return [2, 3]
    let predicate = ExpressionPredicate::new(
        FilterExpression::SliceAccess {
            base: Box::new(FilterExpression::List(vec![
                FilterExpression::Literal(Value::Int64(1)),
                FilterExpression::Literal(Value::Int64(2)),
                FilterExpression::Literal(Value::Int64(3)),
                FilterExpression::Literal(Value::Int64(4)),
                FilterExpression::Literal(Value::Int64(5)),
            ])),
            start: Some(Box::new(FilterExpression::Literal(Value::Int64(1)))),
            end: Some(Box::new(FilterExpression::Literal(Value::Int64(3)))),
        },
        variable_columns,
        store,
    );

    let result = predicate.eval_at(&chunk, 0);
    assert!(result.is_some());

    if let Some(Value::List(items)) = result {
        assert_eq!(items.len(), 2);
        assert_eq!(items[0], Value::Int64(2));
        assert_eq!(items[1], Value::Int64(3));
    } else {
        panic!("Expected List value");
    }
}

#[test]
fn test_might_match_chunk_no_hints() {
    let predicate = ComparisonPredicate::new(0, CompareOp::Eq, Value::Int64(50));
    let hints = ChunkZoneHints::default();

    // With no zone map for the column, should return true (conservative)
    assert!(predicate.might_match_chunk(&hints));
}

#[test]
fn test_might_match_chunk_equality_match() {
    let predicate = ComparisonPredicate::new(0, CompareOp::Eq, Value::Int64(50));

    let mut hints = ChunkZoneHints::default();
    hints.column_hints.insert(
        0,
        obrain_core::index::ZoneMapEntry::with_min_max(Value::Int64(10), Value::Int64(100), 0, 10),
    );

    // 50 is within [10, 100], should return true
    assert!(predicate.might_match_chunk(&hints));
}

#[test]
fn test_might_match_chunk_equality_no_match() {
    let predicate = ComparisonPredicate::new(0, CompareOp::Eq, Value::Int64(200));

    let mut hints = ChunkZoneHints::default();
    hints.column_hints.insert(
        0,
        obrain_core::index::ZoneMapEntry::with_min_max(Value::Int64(10), Value::Int64(100), 0, 10),
    );

    // 200 is outside [10, 100], should return false
    assert!(!predicate.might_match_chunk(&hints));
}

#[test]
fn test_might_match_chunk_greater_than_match() {
    let predicate = ComparisonPredicate::new(0, CompareOp::Gt, Value::Int64(50));

    let mut hints = ChunkZoneHints::default();
    hints.column_hints.insert(
        0,
        obrain_core::index::ZoneMapEntry::with_min_max(Value::Int64(10), Value::Int64(100), 0, 10),
    );

    // max=100 > 50, so some values might be > 50
    assert!(predicate.might_match_chunk(&hints));
}

#[test]
fn test_might_match_chunk_greater_than_no_match() {
    let predicate = ComparisonPredicate::new(0, CompareOp::Gt, Value::Int64(200));

    let mut hints = ChunkZoneHints::default();
    hints.column_hints.insert(
        0,
        obrain_core::index::ZoneMapEntry::with_min_max(Value::Int64(10), Value::Int64(100), 0, 10),
    );

    // max=100 < 200, so no values can be > 200
    assert!(!predicate.might_match_chunk(&hints));
}

#[test]
fn test_might_match_chunk_less_than_match() {
    let predicate = ComparisonPredicate::new(0, CompareOp::Lt, Value::Int64(50));

    let mut hints = ChunkZoneHints::default();
    hints.column_hints.insert(
        0,
        obrain_core::index::ZoneMapEntry::with_min_max(Value::Int64(10), Value::Int64(100), 0, 10),
    );

    // min=10 < 50, so some values might be < 50
    assert!(predicate.might_match_chunk(&hints));
}

#[test]
fn test_might_match_chunk_less_than_no_match() {
    let predicate = ComparisonPredicate::new(0, CompareOp::Lt, Value::Int64(5));

    let mut hints = ChunkZoneHints::default();
    hints.column_hints.insert(
        0,
        obrain_core::index::ZoneMapEntry::with_min_max(Value::Int64(10), Value::Int64(100), 0, 10),
    );

    // min=10 > 5, so no values can be < 5
    assert!(!predicate.might_match_chunk(&hints));
}

#[test]
fn test_might_match_chunk_not_equal_always_conservative() {
    let predicate = ComparisonPredicate::new(0, CompareOp::Ne, Value::Int64(50));

    let mut hints = ChunkZoneHints::default();
    hints.column_hints.insert(
        0,
        obrain_core::index::ZoneMapEntry::with_min_max(Value::Int64(50), Value::Int64(50), 0, 10),
    );

    // Even if min=max=50, Ne is conservative and returns true
    assert!(predicate.might_match_chunk(&hints));
}

#[test]
fn test_comparison_string() {
    let mut builder = DataChunkBuilder::new(&[LogicalType::String]);
    builder.column_mut(0).unwrap().push_string("banana");
    builder.advance_row();
    let chunk = builder.finish();

    // Test string equality
    let pred_eq = ComparisonPredicate::new(0, CompareOp::Eq, Value::String("banana".into()));
    assert!(pred_eq.evaluate(&chunk, 0));

    let pred_ne = ComparisonPredicate::new(0, CompareOp::Ne, Value::String("apple".into()));
    assert!(pred_ne.evaluate(&chunk, 0));

    // Test string ordering
    let pred_lt = ComparisonPredicate::new(0, CompareOp::Lt, Value::String("cherry".into()));
    assert!(pred_lt.evaluate(&chunk, 0)); // "banana" < "cherry"

    let pred_gt = ComparisonPredicate::new(0, CompareOp::Gt, Value::String("apple".into()));
    assert!(pred_gt.evaluate(&chunk, 0)); // "banana" > "apple"
}

#[test]
fn test_comparison_float64() {
    let mut builder = DataChunkBuilder::new(&[LogicalType::Float64]);
    builder
        .column_mut(0)
        .unwrap()
        .push_float64(std::f64::consts::PI);
    builder.advance_row();
    let chunk = builder.finish();

    // Test float equality (within epsilon)
    let pred_eq =
        ComparisonPredicate::new(0, CompareOp::Eq, Value::Float64(std::f64::consts::PI));
    assert!(pred_eq.evaluate(&chunk, 0));

    let pred_ne = ComparisonPredicate::new(0, CompareOp::Ne, Value::Float64(2.71));
    assert!(pred_ne.evaluate(&chunk, 0));

    let pred_lt = ComparisonPredicate::new(0, CompareOp::Lt, Value::Float64(4.0));
    assert!(pred_lt.evaluate(&chunk, 0));

    let pred_ge =
        ComparisonPredicate::new(0, CompareOp::Ge, Value::Float64(std::f64::consts::PI));
    assert!(pred_ge.evaluate(&chunk, 0));
}

#[test]
fn test_comparison_bool() {
    let mut builder = DataChunkBuilder::new(&[LogicalType::Bool]);
    builder.column_mut(0).unwrap().push_bool(true);
    builder.advance_row();
    let chunk = builder.finish();

    let pred_eq = ComparisonPredicate::new(0, CompareOp::Eq, Value::Bool(true));
    assert!(pred_eq.evaluate(&chunk, 0));

    let pred_ne = ComparisonPredicate::new(0, CompareOp::Ne, Value::Bool(false));
    assert!(pred_ne.evaluate(&chunk, 0));

    // Ordering on booleans returns false
    let pred_lt = ComparisonPredicate::new(0, CompareOp::Lt, Value::Bool(false));
    assert!(!pred_lt.evaluate(&chunk, 0));
}

#[test]
fn test_unary_operators() {
    let store: Arc<dyn GraphStore> = Arc::new(SubstrateStore::open_tempfile().unwrap());
    let variable_columns = HashMap::new();
    let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
    let chunk = builder.finish();

    // Test NOT
    let pred_not = ExpressionPredicate::new(
        FilterExpression::Unary {
            op: UnaryFilterOp::Not,
            operand: Box::new(FilterExpression::Literal(Value::Bool(false))),
        },
        variable_columns.clone(),
        Arc::clone(&store),
    );
    assert!(pred_not.evaluate(&chunk, 0));

    // Test IS NULL
    let pred_is_null = ExpressionPredicate::new(
        FilterExpression::Unary {
            op: UnaryFilterOp::IsNull,
            operand: Box::new(FilterExpression::Literal(Value::Null)),
        },
        variable_columns.clone(),
        Arc::clone(&store),
    );
    assert!(pred_is_null.evaluate(&chunk, 0));

    // Test IS NOT NULL
    let pred_is_not_null = ExpressionPredicate::new(
        FilterExpression::Unary {
            op: UnaryFilterOp::IsNotNull,
            operand: Box::new(FilterExpression::Literal(Value::Int64(42))),
        },
        variable_columns.clone(),
        Arc::clone(&store),
    );
    assert!(pred_is_not_null.evaluate(&chunk, 0));

    // Test negation
    let pred_neg = ExpressionPredicate::new(
        FilterExpression::Binary {
            left: Box::new(FilterExpression::Unary {
                op: UnaryFilterOp::Neg,
                operand: Box::new(FilterExpression::Literal(Value::Int64(5))),
            }),
            op: BinaryFilterOp::Eq,
            right: Box::new(FilterExpression::Literal(Value::Int64(-5))),
        },
        variable_columns,
        store,
    );
    assert!(pred_neg.evaluate(&chunk, 0));
}

#[test]
fn test_arithmetic_operators() {
    let store: Arc<dyn GraphStore> = Arc::new(SubstrateStore::open_tempfile().unwrap());
    let variable_columns = HashMap::new();
    let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
    let chunk = builder.finish();

    // Test Add: 2 + 3 = 5
    let pred_add = ExpressionPredicate::new(
        FilterExpression::Binary {
            left: Box::new(FilterExpression::Binary {
                left: Box::new(FilterExpression::Literal(Value::Int64(2))),
                op: BinaryFilterOp::Add,
                right: Box::new(FilterExpression::Literal(Value::Int64(3))),
            }),
            op: BinaryFilterOp::Eq,
            right: Box::new(FilterExpression::Literal(Value::Int64(5))),
        },
        variable_columns.clone(),
        Arc::clone(&store),
    );
    assert!(pred_add.evaluate(&chunk, 0));

    // Test Sub: 10 - 4 = 6
    let pred_sub = ExpressionPredicate::new(
        FilterExpression::Binary {
            left: Box::new(FilterExpression::Binary {
                left: Box::new(FilterExpression::Literal(Value::Int64(10))),
                op: BinaryFilterOp::Sub,
                right: Box::new(FilterExpression::Literal(Value::Int64(4))),
            }),
            op: BinaryFilterOp::Eq,
            right: Box::new(FilterExpression::Literal(Value::Int64(6))),
        },
        variable_columns.clone(),
        Arc::clone(&store),
    );
    assert!(pred_sub.evaluate(&chunk, 0));

    // Test Mul: 3 * 4 = 12
    let pred_mul = ExpressionPredicate::new(
        FilterExpression::Binary {
            left: Box::new(FilterExpression::Binary {
                left: Box::new(FilterExpression::Literal(Value::Int64(3))),
                op: BinaryFilterOp::Mul,
                right: Box::new(FilterExpression::Literal(Value::Int64(4))),
            }),
            op: BinaryFilterOp::Eq,
            right: Box::new(FilterExpression::Literal(Value::Int64(12))),
        },
        variable_columns.clone(),
        Arc::clone(&store),
    );
    assert!(pred_mul.evaluate(&chunk, 0));

    // Test Div: 20 / 4 = 5
    let pred_div = ExpressionPredicate::new(
        FilterExpression::Binary {
            left: Box::new(FilterExpression::Binary {
                left: Box::new(FilterExpression::Literal(Value::Int64(20))),
                op: BinaryFilterOp::Div,
                right: Box::new(FilterExpression::Literal(Value::Int64(4))),
            }),
            op: BinaryFilterOp::Eq,
            right: Box::new(FilterExpression::Literal(Value::Int64(5))),
        },
        variable_columns.clone(),
        Arc::clone(&store),
    );
    assert!(pred_div.evaluate(&chunk, 0));

    // Test Mod: 17 % 5 = 2
    let pred_mod = ExpressionPredicate::new(
        FilterExpression::Binary {
            left: Box::new(FilterExpression::Binary {
                left: Box::new(FilterExpression::Literal(Value::Int64(17))),
                op: BinaryFilterOp::Mod,
                right: Box::new(FilterExpression::Literal(Value::Int64(5))),
            }),
            op: BinaryFilterOp::Eq,
            right: Box::new(FilterExpression::Literal(Value::Int64(2))),
        },
        variable_columns,
        store,
    );
    assert!(pred_mod.evaluate(&chunk, 0));
}

#[test]
fn test_string_operators() {
    let store: Arc<dyn GraphStore> = Arc::new(SubstrateStore::open_tempfile().unwrap());
    let variable_columns = HashMap::new();
    let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
    let chunk = builder.finish();

    // Test STARTS WITH
    let pred_starts = ExpressionPredicate::new(
        FilterExpression::Binary {
            left: Box::new(FilterExpression::Literal(Value::String(
                "hello world".into(),
            ))),
            op: BinaryFilterOp::StartsWith,
            right: Box::new(FilterExpression::Literal(Value::String("hello".into()))),
        },
        variable_columns.clone(),
        Arc::clone(&store),
    );
    assert!(pred_starts.evaluate(&chunk, 0));

    // Test ENDS WITH
    let pred_ends = ExpressionPredicate::new(
        FilterExpression::Binary {
            left: Box::new(FilterExpression::Literal(Value::String(
                "hello world".into(),
            ))),
            op: BinaryFilterOp::EndsWith,
            right: Box::new(FilterExpression::Literal(Value::String("world".into()))),
        },
        variable_columns.clone(),
        Arc::clone(&store),
    );
    assert!(pred_ends.evaluate(&chunk, 0));

    // Test CONTAINS
    let pred_contains = ExpressionPredicate::new(
        FilterExpression::Binary {
            left: Box::new(FilterExpression::Literal(Value::String(
                "hello world".into(),
            ))),
            op: BinaryFilterOp::Contains,
            right: Box::new(FilterExpression::Literal(Value::String("lo wo".into()))),
        },
        variable_columns,
        store,
    );
    assert!(pred_contains.evaluate(&chunk, 0));
}

#[test]
fn test_in_operator() {
    let store: Arc<dyn GraphStore> = Arc::new(SubstrateStore::open_tempfile().unwrap());
    let variable_columns = HashMap::new();
    let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
    let chunk = builder.finish();

    // Test 3 IN [1, 2, 3, 4, 5]
    let pred_in = ExpressionPredicate::new(
        FilterExpression::Binary {
            left: Box::new(FilterExpression::Literal(Value::Int64(3))),
            op: BinaryFilterOp::In,
            right: Box::new(FilterExpression::List(vec![
                FilterExpression::Literal(Value::Int64(1)),
                FilterExpression::Literal(Value::Int64(2)),
                FilterExpression::Literal(Value::Int64(3)),
                FilterExpression::Literal(Value::Int64(4)),
                FilterExpression::Literal(Value::Int64(5)),
            ])),
        },
        variable_columns.clone(),
        Arc::clone(&store),
    );
    assert!(pred_in.evaluate(&chunk, 0));

    // Test 10 NOT IN [1, 2, 3]
    let pred_not_in = ExpressionPredicate::new(
        FilterExpression::Unary {
            op: UnaryFilterOp::Not,
            operand: Box::new(FilterExpression::Binary {
                left: Box::new(FilterExpression::Literal(Value::Int64(10))),
                op: BinaryFilterOp::In,
                right: Box::new(FilterExpression::List(vec![
                    FilterExpression::Literal(Value::Int64(1)),
                    FilterExpression::Literal(Value::Int64(2)),
                    FilterExpression::Literal(Value::Int64(3)),
                ])),
            }),
        },
        variable_columns,
        store,
    );
    assert!(pred_not_in.evaluate(&chunk, 0));
}

#[test]
fn test_logical_operators() {
    let store: Arc<dyn GraphStore> = Arc::new(SubstrateStore::open_tempfile().unwrap());
    let variable_columns = HashMap::new();
    let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
    let chunk = builder.finish();

    // Test AND: true AND true = true
    let pred_and = ExpressionPredicate::new(
        FilterExpression::Binary {
            left: Box::new(FilterExpression::Literal(Value::Bool(true))),
            op: BinaryFilterOp::And,
            right: Box::new(FilterExpression::Literal(Value::Bool(true))),
        },
        variable_columns.clone(),
        Arc::clone(&store),
    );
    assert!(pred_and.evaluate(&chunk, 0));

    // Test OR: false OR true = true
    let pred_or = ExpressionPredicate::new(
        FilterExpression::Binary {
            left: Box::new(FilterExpression::Literal(Value::Bool(false))),
            op: BinaryFilterOp::Or,
            right: Box::new(FilterExpression::Literal(Value::Bool(true))),
        },
        variable_columns.clone(),
        Arc::clone(&store),
    );
    assert!(pred_or.evaluate(&chunk, 0));

    // Test XOR: true XOR false = true
    let pred_xor = ExpressionPredicate::new(
        FilterExpression::Binary {
            left: Box::new(FilterExpression::Literal(Value::Bool(true))),
            op: BinaryFilterOp::Xor,
            right: Box::new(FilterExpression::Literal(Value::Bool(false))),
        },
        variable_columns,
        store,
    );
    assert!(pred_xor.evaluate(&chunk, 0));
}

#[test]
fn test_case_expression_simple() {
    let store: Arc<dyn GraphStore> = Arc::new(SubstrateStore::open_tempfile().unwrap());
    let variable_columns = HashMap::new();
    let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
    let chunk = builder.finish();

    // Test simple CASE: CASE 2 WHEN 1 THEN 'one' WHEN 2 THEN 'two' ELSE 'other' END = 'two'
    let pred_case = ExpressionPredicate::new(
        FilterExpression::Binary {
            left: Box::new(FilterExpression::Case {
                operand: Some(Box::new(FilterExpression::Literal(Value::Int64(2)))),
                when_clauses: vec![
                    (
                        FilterExpression::Literal(Value::Int64(1)),
                        FilterExpression::Literal(Value::String("one".into())),
                    ),
                    (
                        FilterExpression::Literal(Value::Int64(2)),
                        FilterExpression::Literal(Value::String("two".into())),
                    ),
                ],
                else_clause: Some(Box::new(FilterExpression::Literal(Value::String(
                    "other".into(),
                )))),
            }),
            op: BinaryFilterOp::Eq,
            right: Box::new(FilterExpression::Literal(Value::String("two".into()))),
        },
        variable_columns,
        store,
    );
    assert!(pred_case.evaluate(&chunk, 0));
}

#[test]
fn test_case_expression_searched() {
    let store: Arc<dyn GraphStore> = Arc::new(SubstrateStore::open_tempfile().unwrap());
    let variable_columns = HashMap::new();
    let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
    let chunk = builder.finish();

    // Test searched CASE: CASE WHEN 5 > 3 THEN 'yes' ELSE 'no' END = 'yes'
    let pred_case = ExpressionPredicate::new(
        FilterExpression::Binary {
            left: Box::new(FilterExpression::Case {
                operand: None,
                when_clauses: vec![(
                    FilterExpression::Binary {
                        left: Box::new(FilterExpression::Literal(Value::Int64(5))),
                        op: BinaryFilterOp::Gt,
                        right: Box::new(FilterExpression::Literal(Value::Int64(3))),
                    },
                    FilterExpression::Literal(Value::String("yes".into())),
                )],
                else_clause: Some(Box::new(FilterExpression::Literal(Value::String(
                    "no".into(),
                )))),
            }),
            op: BinaryFilterOp::Eq,
            right: Box::new(FilterExpression::Literal(Value::String("yes".into()))),
        },
        variable_columns,
        store,
    );
    assert!(pred_case.evaluate(&chunk, 0));
}

#[test]
fn test_list_functions() {
    let store: Arc<dyn GraphStore> = Arc::new(SubstrateStore::open_tempfile().unwrap());
    let variable_columns = HashMap::new();
    let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
    let chunk = builder.finish();

    // Test head([1, 2, 3]) = 1
    let pred_head = ExpressionPredicate::new(
        FilterExpression::Binary {
            left: Box::new(FilterExpression::FunctionCall {
                name: "head".to_string(),
                args: vec![FilterExpression::List(vec![
                    FilterExpression::Literal(Value::Int64(1)),
                    FilterExpression::Literal(Value::Int64(2)),
                    FilterExpression::Literal(Value::Int64(3)),
                ])],
            }),
            op: BinaryFilterOp::Eq,
            right: Box::new(FilterExpression::Literal(Value::Int64(1))),
        },
        variable_columns.clone(),
        Arc::clone(&store),
    );
    assert!(pred_head.evaluate(&chunk, 0));

    // Test last([1, 2, 3]) = 3
    let pred_last = ExpressionPredicate::new(
        FilterExpression::Binary {
            left: Box::new(FilterExpression::FunctionCall {
                name: "last".to_string(),
                args: vec![FilterExpression::List(vec![
                    FilterExpression::Literal(Value::Int64(1)),
                    FilterExpression::Literal(Value::Int64(2)),
                    FilterExpression::Literal(Value::Int64(3)),
                ])],
            }),
            op: BinaryFilterOp::Eq,
            right: Box::new(FilterExpression::Literal(Value::Int64(3))),
        },
        variable_columns.clone(),
        Arc::clone(&store),
    );
    assert!(pred_last.evaluate(&chunk, 0));

    // Test size([1, 2, 3]) = 3
    let pred_size = ExpressionPredicate::new(
        FilterExpression::Binary {
            left: Box::new(FilterExpression::FunctionCall {
                name: "size".to_string(),
                args: vec![FilterExpression::List(vec![
                    FilterExpression::Literal(Value::Int64(1)),
                    FilterExpression::Literal(Value::Int64(2)),
                    FilterExpression::Literal(Value::Int64(3)),
                ])],
            }),
            op: BinaryFilterOp::Eq,
            right: Box::new(FilterExpression::Literal(Value::Int64(3))),
        },
        variable_columns,
        store,
    );
    assert!(pred_size.evaluate(&chunk, 0));
}

#[test]
fn test_type_conversion_functions() {
    let store: Arc<dyn GraphStore> = Arc::new(SubstrateStore::open_tempfile().unwrap());
    let variable_columns = HashMap::new();
    let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
    let chunk = builder.finish();

    // Test toInteger("42") = 42
    let pred_to_int = ExpressionPredicate::new(
        FilterExpression::Binary {
            left: Box::new(FilterExpression::FunctionCall {
                name: "toInteger".to_string(),
                args: vec![FilterExpression::Literal(Value::String("42".into()))],
            }),
            op: BinaryFilterOp::Eq,
            right: Box::new(FilterExpression::Literal(Value::Int64(42))),
        },
        variable_columns.clone(),
        Arc::clone(&store),
    );
    assert!(pred_to_int.evaluate(&chunk, 0));

    // Test toFloat(42) = 42.0
    let pred_to_float = ExpressionPredicate::new(
        FilterExpression::Binary {
            left: Box::new(FilterExpression::FunctionCall {
                name: "toFloat".to_string(),
                args: vec![FilterExpression::Literal(Value::Int64(42))],
            }),
            op: BinaryFilterOp::Eq,
            right: Box::new(FilterExpression::Literal(Value::Float64(42.0))),
        },
        variable_columns.clone(),
        Arc::clone(&store),
    );
    assert!(pred_to_float.evaluate(&chunk, 0));

    // Test toBoolean("true") = true
    let pred_to_bool = ExpressionPredicate::new(
        FilterExpression::Binary {
            left: Box::new(FilterExpression::FunctionCall {
                name: "toBoolean".to_string(),
                args: vec![FilterExpression::Literal(Value::String("true".into()))],
            }),
            op: BinaryFilterOp::Eq,
            right: Box::new(FilterExpression::Literal(Value::Bool(true))),
        },
        variable_columns,
        store,
    );
    assert!(pred_to_bool.evaluate(&chunk, 0));
}

#[test]
fn test_coalesce_function() {
    let store: Arc<dyn GraphStore> = Arc::new(SubstrateStore::open_tempfile().unwrap());
    let variable_columns = HashMap::new();
    let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
    let chunk = builder.finish();

    // Test coalesce(null, null, 'default') = 'default'
    let pred_coalesce = ExpressionPredicate::new(
        FilterExpression::Binary {
            left: Box::new(FilterExpression::FunctionCall {
                name: "coalesce".to_string(),
                args: vec![
                    FilterExpression::Literal(Value::Null),
                    FilterExpression::Literal(Value::Null),
                    FilterExpression::Literal(Value::String("default".into())),
                ],
            }),
            op: BinaryFilterOp::Eq,
            right: Box::new(FilterExpression::Literal(Value::String("default".into()))),
        },
        variable_columns,
        store,
    );
    assert!(pred_coalesce.evaluate(&chunk, 0));
}

#[test]
fn test_filter_empty_result() {
    // Create a chunk with values that won't match the predicate
    let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
    for i in 1..=5 {
        builder.column_mut(0).unwrap().push_int64(i);
        builder.advance_row();
    }
    let chunk = builder.finish();

    let mock_scan = MockScanOperator {
        chunks: vec![chunk],
        position: 0,
    };

    // Filter for values > 100 (none will match)
    let predicate = ComparisonPredicate::new(0, CompareOp::Gt, Value::Int64(100));
    let mut filter = FilterOperator::new(Box::new(mock_scan), Box::new(predicate));

    // Should return None since nothing matches
    let result = filter.next().unwrap();
    assert!(result.is_none());
}

#[test]
fn test_filter_operator_reset() {
    // Test that reset() calls child.reset()
    // Since MockScanOperator doesn't preserve chunks after reading,
    // we test that reset is called by checking position resets
    let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
    builder.column_mut(0).unwrap().push_int64(50);
    builder.advance_row();
    let chunk = builder.finish();

    let mock_scan = MockScanOperator {
        chunks: vec![chunk],
        position: 0,
    };

    let predicate = ComparisonPredicate::new(0, CompareOp::Eq, Value::Int64(50));
    let mut filter = FilterOperator::new(Box::new(mock_scan), Box::new(predicate));

    // First iteration
    let result = filter.next().unwrap();
    assert!(result.is_some());
    let result = filter.next().unwrap();
    assert!(result.is_none());

    // Note: MockScanOperator replaces chunks with empty ones when read,
    // so reset doesn't restore the data. This test verifies reset() is called.
    filter.reset();
    // After reset, position is 0 but chunk is empty
    let result = filter.next().unwrap();
    // Empty chunk produces no matches, returns None
    assert!(result.is_none());
}

#[test]
fn test_mixed_type_comparison_int_float() {
    let store: Arc<dyn GraphStore> = Arc::new(SubstrateStore::open_tempfile().unwrap());
    let variable_columns = HashMap::new();
    let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
    let chunk = builder.finish();

    // Test 5 == 5.0 (mixed int/float comparison)
    let pred_mixed = ExpressionPredicate::new(
        FilterExpression::Binary {
            left: Box::new(FilterExpression::Literal(Value::Int64(5))),
            op: BinaryFilterOp::Eq,
            right: Box::new(FilterExpression::Literal(Value::Float64(5.0))),
        },
        variable_columns,
        store,
    );
    assert!(pred_mixed.evaluate(&chunk, 0));
}

#[test]
fn test_zone_map_allows_matching_chunk() {
    // Test that a chunk with zone hints indicating potential matches is evaluated
    let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
    for i in 10..=20 {
        builder.column_mut(0).unwrap().push_int64(i);
        builder.advance_row();
    }
    let mut chunk = builder.finish();

    // Set zone hints: min=10, max=20
    let mut hints = obrain_core::execution::chunk::ChunkZoneHints::default();
    hints.column_hints.insert(
        0,
        obrain_core::index::ZoneMapEntry::with_min_max(Value::Int64(10), Value::Int64(20), 0, 11),
    );
    chunk.set_zone_hints(hints);

    let mock_scan = MockScanOperator {
        chunks: vec![chunk],
        position: 0,
    };

    // Filter for values > 15 (some will match)
    let predicate = ComparisonPredicate::new(0, CompareOp::Gt, Value::Int64(15));
    let mut filter = FilterOperator::new(Box::new(mock_scan), Box::new(predicate));

    // Should return matching rows
    let result = filter.next().unwrap();
    assert!(result.is_some());
    let chunk = result.unwrap();

    // Should have rows 16, 17, 18, 19, 20 (5 rows)
    assert_eq!(chunk.row_count(), 5);
}

#[test]
fn test_filter_with_all_rows_matching() {
    // All values in chunk match the predicate
    let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
    for i in 100..=110 {
        builder.column_mut(0).unwrap().push_int64(i);
        builder.advance_row();
    }
    let chunk = builder.finish();

    let mock_scan = MockScanOperator {
        chunks: vec![chunk],
        position: 0,
    };

    // Filter for values > 50 (all will match)
    let predicate = ComparisonPredicate::new(0, CompareOp::Gt, Value::Int64(50));
    let mut filter = FilterOperator::new(Box::new(mock_scan), Box::new(predicate));

    let result = filter.next().unwrap();
    assert!(result.is_some());
    let chunk = result.unwrap();

    // All 11 rows should be returned
    assert_eq!(chunk.row_count(), 11);
}

#[test]
fn test_filter_with_sparse_data() {
    // Test filtering with sparse matching data
    let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
    // Create values where only some match: 1, 10, 2, 20, 3, 30
    for &v in &[1i64, 10, 2, 20, 3, 30] {
        builder.column_mut(0).unwrap().push_int64(v);
        builder.advance_row();
    }
    let chunk = builder.finish();

    let mock_scan = MockScanOperator {
        chunks: vec![chunk],
        position: 0,
    };

    // Filter for values > 5 (only 10, 20, 30 should match)
    let predicate = ComparisonPredicate::new(0, CompareOp::Gt, Value::Int64(5));
    let mut filter = FilterOperator::new(Box::new(mock_scan), Box::new(predicate));

    let result = filter.next().unwrap();
    assert!(result.is_some());
    let chunk = result.unwrap();

    // Only 10, 20, 30 should match (3 rows)
    assert_eq!(chunk.row_count(), 3);
}

#[test]
fn test_predicate_on_wrong_column_returns_empty() {
    // When the predicate references a column index that's out of bounds
    // or the column type is incompatible
    let mut builder = DataChunkBuilder::new(&[LogicalType::String]);
    builder.column_mut(0).unwrap().push_string("hello");
    builder.advance_row();
    let chunk = builder.finish();

    let mock_scan = MockScanOperator {
        chunks: vec![chunk],
        position: 0,
    };

    // Predicate on column 5 (doesn't exist)
    let predicate = ComparisonPredicate::new(5, CompareOp::Eq, Value::Int64(42));
    let mut filter = FilterOperator::new(Box::new(mock_scan), Box::new(predicate));

    // Should handle gracefully (either error or empty result)
    let result = filter.next();
    // The behavior depends on implementation - just verify no panic
    let _ = result;
}

#[test]
fn test_expression_predicate_with_labels_function() {
    use obrain_core::graph::GraphStoreMut;

    // Test the labels() function in predicates
    let store: Arc<dyn GraphStoreMut> = Arc::new(SubstrateStore::open_tempfile().unwrap());

    // Create a node with a label
    let node_id = store.create_node(&["Person", "Employee"]);

    // Build a chunk with the node
    let mut builder = DataChunkBuilder::new(&[LogicalType::Node]);
    builder.column_mut(0).unwrap().push_node_id(node_id);
    builder.advance_row();
    let chunk = builder.finish();

    // Map column 0 to variable "n"
    let mut variable_columns = HashMap::new();
    variable_columns.insert("n".to_string(), 0);

    // Test: 'Person' IN labels(n)
    let pred = ExpressionPredicate::new(
        FilterExpression::Binary {
            left: Box::new(FilterExpression::Literal(Value::String("Person".into()))),
            op: BinaryFilterOp::In,
            right: Box::new(FilterExpression::FunctionCall {
                name: "labels".to_string(),
                args: vec![FilterExpression::Variable("n".to_string())],
            }),
        },
        variable_columns,
        store.clone() as Arc<dyn GraphStore>,
    );

    assert!(pred.evaluate(&chunk, 0));
}

#[test]
fn test_comparison_with_boundary_values() {
    // Test comparisons at exact boundary values
    let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
    builder.column_mut(0).unwrap().push_int64(i64::MAX);
    builder.advance_row();
    builder.column_mut(0).unwrap().push_int64(i64::MIN);
    builder.advance_row();
    builder.column_mut(0).unwrap().push_int64(0);
    builder.advance_row();
    let chunk = builder.finish();

    // Test >= 0
    let pred_ge = ComparisonPredicate::new(0, CompareOp::Ge, Value::Int64(0));
    assert!(pred_ge.evaluate(&chunk, 0)); // i64::MAX >= 0
    assert!(!pred_ge.evaluate(&chunk, 1)); // i64::MIN >= 0 is false
    assert!(pred_ge.evaluate(&chunk, 2)); // 0 >= 0

    // Test <= 0
    let pred_le = ComparisonPredicate::new(0, CompareOp::Le, Value::Int64(0));
    assert!(!pred_le.evaluate(&chunk, 0)); // i64::MAX <= 0 is false
    assert!(pred_le.evaluate(&chunk, 1)); // i64::MIN <= 0
    assert!(pred_le.evaluate(&chunk, 2)); // 0 <= 0
}

// ── Cross-type equality (String ↔ numeric) ──────────────────────────

/// Regression test: RDF stores numeric literals as strings, so filters
/// like `FILTER(?age = 30)` compare `Value::String("30")` with
/// `Value::Int64(30)`.  The `values_equal` path must coerce.
#[test]
fn test_cross_type_string_int_equality() {
    
    let store: Arc<dyn GraphStore> = Arc::new(SubstrateStore::open_tempfile().unwrap());
    let vc = HashMap::new();
    let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
    let chunk = builder.finish();

    // String "42" == Int64(42)
    let pred = ExpressionPredicate::new(
        FilterExpression::Binary {
            left: Box::new(FilterExpression::Literal(Value::String("42".into()))),
            op: BinaryFilterOp::Eq,
            right: Box::new(FilterExpression::Literal(Value::Int64(42))),
        },
        vc.clone(),
        Arc::clone(&store),
    );
    assert!(pred.evaluate(&chunk, 0));

    // String "42" != Int64(99)
    let pred_ne = ExpressionPredicate::new(
        FilterExpression::Binary {
            left: Box::new(FilterExpression::Literal(Value::String("42".into()))),
            op: BinaryFilterOp::Ne,
            right: Box::new(FilterExpression::Literal(Value::Int64(99))),
        },
        vc.clone(),
        Arc::clone(&store),
    );
    assert!(pred_ne.evaluate(&chunk, 0));

    // Non-numeric string should NOT equal any integer
    let pred_bad = ExpressionPredicate::new(
        FilterExpression::Binary {
            left: Box::new(FilterExpression::Literal(Value::String("hello".into()))),
            op: BinaryFilterOp::Eq,
            right: Box::new(FilterExpression::Literal(Value::Int64(42))),
        },
        vc,
        store,
    );
    assert!(!pred_bad.evaluate(&chunk, 0));
}

/// String ↔ Float64 equality: "7.25" == Float64(7.25)
#[test]
fn test_cross_type_string_float_equality() {
    
    let store: Arc<dyn GraphStore> = Arc::new(SubstrateStore::open_tempfile().unwrap());
    let vc = HashMap::new();
    let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
    let chunk = builder.finish();

    let pred = ExpressionPredicate::new(
        FilterExpression::Binary {
            left: Box::new(FilterExpression::Literal(Value::String("7.25".into()))),
            op: BinaryFilterOp::Eq,
            right: Box::new(FilterExpression::Literal(Value::Float64(7.25))),
        },
        vc.clone(),
        Arc::clone(&store),
    );
    assert!(pred.evaluate(&chunk, 0));

    // "7.25" != 2.5
    let pred_ne = ExpressionPredicate::new(
        FilterExpression::Binary {
            left: Box::new(FilterExpression::Literal(Value::Float64(2.5))),
            op: BinaryFilterOp::Ne,
            right: Box::new(FilterExpression::Literal(Value::String("7.25".into()))),
        },
        vc,
        store,
    );
    assert!(pred_ne.evaluate(&chunk, 0));
}

// ── Cross-type ordering (String ↔ numeric) ──────────────────────────

/// Regression test: String-encoded numbers must support range comparisons
/// so that `FILTER(?age > 25)` works when `?age` is stored as "30".
#[test]
fn test_cross_type_string_numeric_ordering() {
    
    let store: Arc<dyn GraphStore> = Arc::new(SubstrateStore::open_tempfile().unwrap());
    let vc = HashMap::new();
    let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
    let chunk = builder.finish();

    // "30" > Int64(25)
    let pred_gt = ExpressionPredicate::new(
        FilterExpression::Binary {
            left: Box::new(FilterExpression::Literal(Value::String("30".into()))),
            op: BinaryFilterOp::Gt,
            right: Box::new(FilterExpression::Literal(Value::Int64(25))),
        },
        vc.clone(),
        Arc::clone(&store),
    );
    assert!(pred_gt.evaluate(&chunk, 0));

    // Int64(10) < "20.5" (cross Float64 path)
    let pred_lt = ExpressionPredicate::new(
        FilterExpression::Binary {
            left: Box::new(FilterExpression::Literal(Value::Int64(10))),
            op: BinaryFilterOp::Lt,
            right: Box::new(FilterExpression::Literal(Value::String("20.5".into()))),
        },
        vc.clone(),
        Arc::clone(&store),
    );
    assert!(pred_lt.evaluate(&chunk, 0));

    // "2.5" <= Float64(2.5)
    let pred_le = ExpressionPredicate::new(
        FilterExpression::Binary {
            left: Box::new(FilterExpression::Literal(Value::String("2.5".into()))),
            op: BinaryFilterOp::Le,
            right: Box::new(FilterExpression::Literal(Value::Float64(2.5))),
        },
        vc.clone(),
        Arc::clone(&store),
    );
    assert!(pred_le.evaluate(&chunk, 0));

    // Float64(100.0) >= "99.9"
    let pred_ge = ExpressionPredicate::new(
        FilterExpression::Binary {
            left: Box::new(FilterExpression::Literal(Value::Float64(100.0))),
            op: BinaryFilterOp::Ge,
            right: Box::new(FilterExpression::Literal(Value::String("99.9".into()))),
        },
        vc,
        store,
    );
    assert!(pred_ge.evaluate(&chunk, 0));
}

// ── Stacked filter (selection vector preservation) ───────────────────

/// Regression test: when two FilterOperators are stacked (child filter →
/// parent filter), the parent must respect the child's selection vector
/// instead of re-evaluating all physical rows.
#[test]
fn test_stacked_filters_respect_selection_vector() {
    // Chunk: ages = [20, 35, 45, 25, 50]
    let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
    for age in [20, 35, 45, 25, 50] {
        builder.column_mut(0).unwrap().push_int64(age);
        builder.advance_row();
    }
    let chunk = builder.finish();

    let scan = MockScanOperator {
        chunks: vec![chunk],
        position: 0,
    };

    // First filter: age > 25 → rows 1(35), 2(45), 4(50)
    let pred1 = ComparisonPredicate::new(0, CompareOp::Gt, Value::Int64(25));
    let filter1 = FilterOperator::new(Box::new(scan), Box::new(pred1));

    // Second (stacked) filter: age < 50 → should intersect → rows 1(35), 2(45)
    let pred2 = ComparisonPredicate::new(0, CompareOp::Lt, Value::Int64(50));
    let mut filter2 = FilterOperator::new(Box::new(filter1), Box::new(pred2));

    let result = filter2.next().unwrap().unwrap();
    assert_eq!(
        result.row_count(),
        2,
        "stacked filter should yield 2 rows (35, 45)"
    );

    // Verify it's exhausted
    assert!(filter2.next().unwrap().is_none());
}

// === eval_binary_op: Arithmetic Tests ===

/// Helper: creates an `ExpressionPredicate` wrapping a literal expression,
/// evaluates it against an empty chunk, and returns the result `Value`.
fn eval_literal_expr(expr: FilterExpression) -> Option<Value> {
    
    let store: Arc<dyn GraphStore> = Arc::new(SubstrateStore::open_tempfile().unwrap());
    let pred = ExpressionPredicate::new(expr, HashMap::new(), store);
    let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
    let chunk = builder.finish();
    pred.eval_at(&chunk, 0)
}

fn binary(left: Value, op: BinaryFilterOp, right: Value) -> FilterExpression {
    FilterExpression::Binary {
        left: Box::new(FilterExpression::Literal(left)),
        op,
        right: Box::new(FilterExpression::Literal(right)),
    }
}

fn unary(op: UnaryFilterOp, operand: FilterExpression) -> FilterExpression {
    FilterExpression::Unary {
        op,
        operand: Box::new(operand),
    }
}

#[test]
fn test_eval_binary_addition_int() {
    let result = eval_literal_expr(binary(
        Value::Int64(10),
        BinaryFilterOp::Add,
        Value::Int64(20),
    ));
    assert_eq!(result, Some(Value::Int64(30)));
}

#[test]
fn test_eval_binary_subtraction_int() {
    let result = eval_literal_expr(binary(
        Value::Int64(50),
        BinaryFilterOp::Sub,
        Value::Int64(18),
    ));
    assert_eq!(result, Some(Value::Int64(32)));
}

#[test]
fn test_eval_binary_multiplication_int() {
    let result = eval_literal_expr(binary(
        Value::Int64(7),
        BinaryFilterOp::Mul,
        Value::Int64(6),
    ));
    assert_eq!(result, Some(Value::Int64(42)));
}

#[test]
fn test_eval_binary_division_int() {
    let result = eval_literal_expr(binary(
        Value::Int64(100),
        BinaryFilterOp::Div,
        Value::Int64(4),
    ));
    assert_eq!(result, Some(Value::Int64(25)));
}

#[test]
fn test_eval_binary_modulo_int() {
    let result = eval_literal_expr(binary(
        Value::Int64(17),
        BinaryFilterOp::Mod,
        Value::Int64(5),
    ));
    assert_eq!(result, Some(Value::Int64(2)));
}

// === eval_binary_op: Comparisons ===

#[test]
fn test_eval_comparison_lt() {
    let result =
        eval_literal_expr(binary(Value::Int64(3), BinaryFilterOp::Lt, Value::Int64(5)));
    assert_eq!(result, Some(Value::Bool(true)));

    let result =
        eval_literal_expr(binary(Value::Int64(5), BinaryFilterOp::Lt, Value::Int64(3)));
    assert_eq!(result, Some(Value::Bool(false)));
}

#[test]
fn test_eval_comparison_gt() {
    let result = eval_literal_expr(binary(
        Value::Int64(10),
        BinaryFilterOp::Gt,
        Value::Int64(5),
    ));
    assert_eq!(result, Some(Value::Bool(true)));
}

#[test]
fn test_eval_comparison_eq() {
    let result = eval_literal_expr(binary(
        Value::Int64(42),
        BinaryFilterOp::Eq,
        Value::Int64(42),
    ));
    assert_eq!(result, Some(Value::Bool(true)));

    let result = eval_literal_expr(binary(
        Value::Int64(42),
        BinaryFilterOp::Eq,
        Value::Int64(43),
    ));
    assert_eq!(result, Some(Value::Bool(false)));
}

#[test]
fn test_eval_comparison_ne() {
    let result = eval_literal_expr(binary(
        Value::String("hello".into()),
        BinaryFilterOp::Ne,
        Value::String("world".into()),
    ));
    assert_eq!(result, Some(Value::Bool(true)));

    let result = eval_literal_expr(binary(
        Value::String("same".into()),
        BinaryFilterOp::Ne,
        Value::String("same".into()),
    ));
    assert_eq!(result, Some(Value::Bool(false)));
}

#[test]
fn test_eval_comparison_le_ge() {
    // <=
    let result =
        eval_literal_expr(binary(Value::Int64(5), BinaryFilterOp::Le, Value::Int64(5)));
    assert_eq!(result, Some(Value::Bool(true)));

    let result =
        eval_literal_expr(binary(Value::Int64(6), BinaryFilterOp::Le, Value::Int64(5)));
    assert_eq!(result, Some(Value::Bool(false)));

    // >=
    let result =
        eval_literal_expr(binary(Value::Int64(5), BinaryFilterOp::Ge, Value::Int64(5)));
    assert_eq!(result, Some(Value::Bool(true)));

    let result =
        eval_literal_expr(binary(Value::Int64(4), BinaryFilterOp::Ge, Value::Int64(5)));
    assert_eq!(result, Some(Value::Bool(false)));
}

// === eval_binary_op: Logical Operators ===

#[test]
fn test_eval_logical_and() {
    let result = eval_literal_expr(binary(
        Value::Bool(true),
        BinaryFilterOp::And,
        Value::Bool(true),
    ));
    assert_eq!(result, Some(Value::Bool(true)));

    let result = eval_literal_expr(binary(
        Value::Bool(true),
        BinaryFilterOp::And,
        Value::Bool(false),
    ));
    assert_eq!(result, Some(Value::Bool(false)));
}

#[test]
fn test_eval_logical_or() {
    let result = eval_literal_expr(binary(
        Value::Bool(false),
        BinaryFilterOp::Or,
        Value::Bool(true),
    ));
    assert_eq!(result, Some(Value::Bool(true)));

    let result = eval_literal_expr(binary(
        Value::Bool(false),
        BinaryFilterOp::Or,
        Value::Bool(false),
    ));
    assert_eq!(result, Some(Value::Bool(false)));
}

#[test]
fn test_eval_logical_xor() {
    let result = eval_literal_expr(binary(
        Value::Bool(true),
        BinaryFilterOp::Xor,
        Value::Bool(false),
    ));
    assert_eq!(result, Some(Value::Bool(true)));

    let result = eval_literal_expr(binary(
        Value::Bool(true),
        BinaryFilterOp::Xor,
        Value::Bool(true),
    ));
    assert_eq!(result, Some(Value::Bool(false)));
}

// === Type Coercion: Int + Float Arithmetic ===

#[test]
fn test_eval_type_coercion_int_plus_float() {
    let result = eval_literal_expr(binary(
        Value::Int64(10),
        BinaryFilterOp::Add,
        Value::Float64(2.5),
    ));
    assert_eq!(result, Some(Value::Float64(12.5)));
}

#[test]
fn test_eval_type_coercion_float_minus_int() {
    let result = eval_literal_expr(binary(
        Value::Float64(10.0),
        BinaryFilterOp::Sub,
        Value::Int64(3),
    ));
    assert_eq!(result, Some(Value::Float64(7.0)));
}

#[test]
fn test_eval_type_coercion_int_mul_float() {
    let result = eval_literal_expr(binary(
        Value::Int64(4),
        BinaryFilterOp::Mul,
        Value::Float64(2.5),
    ));
    assert_eq!(result, Some(Value::Float64(10.0)));
}

#[test]
fn test_eval_type_coercion_int_eq_float() {
    // Int 42 should equal Float 42.0
    let result = eval_literal_expr(binary(
        Value::Int64(42),
        BinaryFilterOp::Eq,
        Value::Float64(42.0),
    ));
    assert_eq!(result, Some(Value::Bool(true)));
}

#[test]
fn test_eval_type_coercion_int_lt_float() {
    let result = eval_literal_expr(binary(
        Value::Int64(3),
        BinaryFilterOp::Lt,
        Value::Float64(3.5),
    ));
    assert_eq!(result, Some(Value::Bool(true)));
}

// === String Comparison ===

#[test]
fn test_eval_string_comparison() {
    let result = eval_literal_expr(binary(
        Value::String("apple".into()),
        BinaryFilterOp::Lt,
        Value::String("banana".into()),
    ));
    assert_eq!(result, Some(Value::Bool(true)));

    let result = eval_literal_expr(binary(
        Value::String("zebra".into()),
        BinaryFilterOp::Gt,
        Value::String("apple".into()),
    ));
    assert_eq!(result, Some(Value::Bool(true)));
}

#[test]
fn test_eval_string_concatenation() {
    let result = eval_literal_expr(binary(
        Value::String("Hello".into()),
        BinaryFilterOp::Add,
        Value::String(" World".into()),
    ));
    assert_eq!(result, Some(Value::String("Hello World".into())));
}

// === IS NULL / IS NOT NULL ===

#[test]
fn test_eval_is_null() {
    let result = eval_literal_expr(unary(
        UnaryFilterOp::IsNull,
        FilterExpression::Literal(Value::Null),
    ));
    assert_eq!(result, Some(Value::Bool(true)));

    let result = eval_literal_expr(unary(
        UnaryFilterOp::IsNull,
        FilterExpression::Literal(Value::Int64(42)),
    ));
    assert_eq!(result, Some(Value::Bool(false)));
}

#[test]
fn test_eval_is_not_null() {
    let result = eval_literal_expr(unary(
        UnaryFilterOp::IsNotNull,
        FilterExpression::Literal(Value::Int64(42)),
    ));
    assert_eq!(result, Some(Value::Bool(true)));

    let result = eval_literal_expr(unary(
        UnaryFilterOp::IsNotNull,
        FilterExpression::Literal(Value::Null),
    ));
    assert_eq!(result, Some(Value::Bool(false)));
}

#[test]
fn test_eval_is_null_on_missing_variable() {
    // Accessing a non-existent variable should produce None,
    // which IS NULL treats as true
    
    let store: Arc<dyn GraphStore> = Arc::new(SubstrateStore::open_tempfile().unwrap());
    let expr = FilterExpression::Unary {
        op: UnaryFilterOp::IsNull,
        operand: Box::new(FilterExpression::Variable("missing_var".to_string())),
    };
    let pred = ExpressionPredicate::new(expr, HashMap::new(), store);
    let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
    let chunk = builder.finish();
    let result = pred.eval_at(&chunk, 0);
    assert_eq!(result, Some(Value::Bool(true)));
}

// === STARTS WITH / ENDS WITH / CONTAINS ===

#[test]
fn test_eval_starts_with() {
    let result = eval_literal_expr(binary(
        Value::String("hello world".into()),
        BinaryFilterOp::StartsWith,
        Value::String("hello".into()),
    ));
    assert_eq!(result, Some(Value::Bool(true)));

    let result = eval_literal_expr(binary(
        Value::String("hello world".into()),
        BinaryFilterOp::StartsWith,
        Value::String("world".into()),
    ));
    assert_eq!(result, Some(Value::Bool(false)));
}

#[test]
fn test_eval_ends_with() {
    let result = eval_literal_expr(binary(
        Value::String("hello world".into()),
        BinaryFilterOp::EndsWith,
        Value::String("world".into()),
    ));
    assert_eq!(result, Some(Value::Bool(true)));

    let result = eval_literal_expr(binary(
        Value::String("hello world".into()),
        BinaryFilterOp::EndsWith,
        Value::String("hello".into()),
    ));
    assert_eq!(result, Some(Value::Bool(false)));
}

#[test]
fn test_eval_contains() {
    let result = eval_literal_expr(binary(
        Value::String("hello world".into()),
        BinaryFilterOp::Contains,
        Value::String("lo wo".into()),
    ));
    assert_eq!(result, Some(Value::Bool(true)));

    let result = eval_literal_expr(binary(
        Value::String("hello world".into()),
        BinaryFilterOp::Contains,
        Value::String("xyz".into()),
    ));
    assert_eq!(result, Some(Value::Bool(false)));
}

// === List Operations: IN Operator ===

#[test]
fn test_eval_in_operator() {
    
    let store: Arc<dyn GraphStore> = Arc::new(SubstrateStore::open_tempfile().unwrap());
    let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
    let chunk = builder.finish();

    // 2 IN [1, 2, 3] should be true
    let expr = FilterExpression::Binary {
        left: Box::new(FilterExpression::Literal(Value::Int64(2))),
        op: BinaryFilterOp::In,
        right: Box::new(FilterExpression::List(vec![
            FilterExpression::Literal(Value::Int64(1)),
            FilterExpression::Literal(Value::Int64(2)),
            FilterExpression::Literal(Value::Int64(3)),
        ])),
    };
    let pred = ExpressionPredicate::new(expr, HashMap::new(), Arc::clone(&store));
    let result = pred.eval_at(&chunk, 0);
    assert_eq!(result, Some(Value::Bool(true)));

    // 5 IN [1, 2, 3] should be false
    let expr = FilterExpression::Binary {
        left: Box::new(FilterExpression::Literal(Value::Int64(5))),
        op: BinaryFilterOp::In,
        right: Box::new(FilterExpression::List(vec![
            FilterExpression::Literal(Value::Int64(1)),
            FilterExpression::Literal(Value::Int64(2)),
            FilterExpression::Literal(Value::Int64(3)),
        ])),
    };
    let pred = ExpressionPredicate::new(expr, HashMap::new(), Arc::clone(&store));
    let result = pred.eval_at(&chunk, 0);
    assert_eq!(result, Some(Value::Bool(false)));
}

#[test]
fn test_eval_in_operator_strings() {
    
    let store: Arc<dyn GraphStore> = Arc::new(SubstrateStore::open_tempfile().unwrap());
    let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
    let chunk = builder.finish();

    // "banana" IN ["apple", "banana", "cherry"]
    let expr = FilterExpression::Binary {
        left: Box::new(FilterExpression::Literal(Value::String("banana".into()))),
        op: BinaryFilterOp::In,
        right: Box::new(FilterExpression::List(vec![
            FilterExpression::Literal(Value::String("apple".into())),
            FilterExpression::Literal(Value::String("banana".into())),
            FilterExpression::Literal(Value::String("cherry".into())),
        ])),
    };
    let pred = ExpressionPredicate::new(expr, HashMap::new(), store);
    let result = pred.eval_at(&chunk, 0);
    assert_eq!(result, Some(Value::Bool(true)));
}

// === List Index Access ===

#[test]
fn test_eval_list_index_access() {
    // [10, 20, 30][2] = 30
    let result = eval_literal_expr(FilterExpression::IndexAccess {
        base: Box::new(FilterExpression::List(vec![
            FilterExpression::Literal(Value::Int64(10)),
            FilterExpression::Literal(Value::Int64(20)),
            FilterExpression::Literal(Value::Int64(30)),
        ])),
        index: Box::new(FilterExpression::Literal(Value::Int64(2))),
    });
    assert_eq!(result, Some(Value::Int64(30)));
}

#[test]
fn test_eval_list_negative_index() {
    // [10, 20, 30][-2] = 20
    let result = eval_literal_expr(FilterExpression::IndexAccess {
        base: Box::new(FilterExpression::List(vec![
            FilterExpression::Literal(Value::Int64(10)),
            FilterExpression::Literal(Value::Int64(20)),
            FilterExpression::Literal(Value::Int64(30)),
        ])),
        index: Box::new(FilterExpression::Literal(Value::Int64(-2))),
    });
    assert_eq!(result, Some(Value::Int64(20)));
}

// === CASE / NULLIF Pattern ===

#[test]
fn test_eval_case_simple() {
    // CASE WHEN true THEN 'yes' ELSE 'no' END
    let result = eval_literal_expr(FilterExpression::Case {
        operand: None,
        when_clauses: vec![(
            FilterExpression::Literal(Value::Bool(true)),
            FilterExpression::Literal(Value::String("yes".into())),
        )],
        else_clause: Some(Box::new(FilterExpression::Literal(Value::String(
            "no".into(),
        )))),
    });
    assert_eq!(result, Some(Value::String("yes".into())));
}

#[test]
fn test_eval_case_falls_to_else() {
    // CASE WHEN false THEN 'yes' ELSE 'no' END
    let result = eval_literal_expr(FilterExpression::Case {
        operand: None,
        when_clauses: vec![(
            FilterExpression::Literal(Value::Bool(false)),
            FilterExpression::Literal(Value::String("yes".into())),
        )],
        else_clause: Some(Box::new(FilterExpression::Literal(Value::String(
            "no".into(),
        )))),
    });
    assert_eq!(result, Some(Value::String("no".into())));
}

#[test]
fn test_eval_case_no_else_returns_null() {
    // CASE WHEN false THEN 'yes' END (no ELSE, so NULL)
    let result = eval_literal_expr(FilterExpression::Case {
        operand: None,
        when_clauses: vec![(
            FilterExpression::Literal(Value::Bool(false)),
            FilterExpression::Literal(Value::String("yes".into())),
        )],
        else_clause: None,
    });
    assert_eq!(result, Some(Value::Null));
}

#[test]
fn test_eval_nullif_via_case() {
    // NULLIF(a, b) is equivalent to: CASE WHEN a = b THEN NULL ELSE a END
    // Test NULLIF(5, 5) => NULL
    let result = eval_literal_expr(FilterExpression::Case {
        operand: None,
        when_clauses: vec![(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Literal(Value::Int64(5))),
                op: BinaryFilterOp::Eq,
                right: Box::new(FilterExpression::Literal(Value::Int64(5))),
            },
            FilterExpression::Literal(Value::Null),
        )],
        else_clause: Some(Box::new(FilterExpression::Literal(Value::Int64(5)))),
    });
    assert_eq!(result, Some(Value::Null));

    // NULLIF(5, 3) => 5
    let result = eval_literal_expr(FilterExpression::Case {
        operand: None,
        when_clauses: vec![(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Literal(Value::Int64(5))),
                op: BinaryFilterOp::Eq,
                right: Box::new(FilterExpression::Literal(Value::Int64(3))),
            },
            FilterExpression::Literal(Value::Null),
        )],
        else_clause: Some(Box::new(FilterExpression::Literal(Value::Int64(5)))),
    });
    assert_eq!(result, Some(Value::Int64(5)));
}

#[test]
fn test_eval_simple_case_with_operand() {
    // CASE 2 WHEN 1 THEN 'one' WHEN 2 THEN 'two' ELSE 'other' END
    let result = eval_literal_expr(FilterExpression::Case {
        operand: Some(Box::new(FilterExpression::Literal(Value::Int64(2)))),
        when_clauses: vec![
            (
                FilterExpression::Literal(Value::Int64(1)),
                FilterExpression::Literal(Value::String("one".into())),
            ),
            (
                FilterExpression::Literal(Value::Int64(2)),
                FilterExpression::Literal(Value::String("two".into())),
            ),
        ],
        else_clause: Some(Box::new(FilterExpression::Literal(Value::String(
            "other".into(),
        )))),
    });
    assert_eq!(result, Some(Value::String("two".into())));
}

// === Unary Operators ===

#[test]
fn test_eval_unary_not() {
    let result = eval_literal_expr(unary(
        UnaryFilterOp::Not,
        FilterExpression::Literal(Value::Bool(true)),
    ));
    assert_eq!(result, Some(Value::Bool(false)));

    let result = eval_literal_expr(unary(
        UnaryFilterOp::Not,
        FilterExpression::Literal(Value::Bool(false)),
    ));
    assert_eq!(result, Some(Value::Bool(true)));
}

#[test]
fn test_eval_unary_neg() {
    let result = eval_literal_expr(unary(
        UnaryFilterOp::Neg,
        FilterExpression::Literal(Value::Int64(42)),
    ));
    assert_eq!(result, Some(Value::Int64(-42)));

    let result = eval_literal_expr(unary(
        UnaryFilterOp::Neg,
        FilterExpression::Literal(Value::Float64(7.25)),
    ));
    assert_eq!(result, Some(Value::Float64(-7.25)));
}

// === Reduce Expression Evaluation ===

#[test]
fn test_eval_reduce_sum() {
    // reduce(acc = 0, x IN [1, 2, 3] | acc + x) = 6
    let result = eval_literal_expr(FilterExpression::Reduce {
        accumulator: "acc".to_string(),
        initial: Box::new(FilterExpression::Literal(Value::Int64(0))),
        variable: "x".to_string(),
        list: Box::new(FilterExpression::List(vec![
            FilterExpression::Literal(Value::Int64(1)),
            FilterExpression::Literal(Value::Int64(2)),
            FilterExpression::Literal(Value::Int64(3)),
        ])),
        expression: Box::new(FilterExpression::Binary {
            left: Box::new(FilterExpression::Variable("acc".to_string())),
            op: BinaryFilterOp::Add,
            right: Box::new(FilterExpression::Variable("x".to_string())),
        }),
    });
    assert_eq!(result, Some(Value::Int64(6)));
}

#[test]
fn test_eval_reduce_product() {
    // reduce(acc = 1, x IN [2, 3, 4] | acc * x) = 24
    let result = eval_literal_expr(FilterExpression::Reduce {
        accumulator: "acc".to_string(),
        initial: Box::new(FilterExpression::Literal(Value::Int64(1))),
        variable: "x".to_string(),
        list: Box::new(FilterExpression::List(vec![
            FilterExpression::Literal(Value::Int64(2)),
            FilterExpression::Literal(Value::Int64(3)),
            FilterExpression::Literal(Value::Int64(4)),
        ])),
        expression: Box::new(FilterExpression::Binary {
            left: Box::new(FilterExpression::Variable("acc".to_string())),
            op: BinaryFilterOp::Mul,
            right: Box::new(FilterExpression::Variable("x".to_string())),
        }),
    });
    assert_eq!(result, Some(Value::Int64(24)));
}

// === List Comprehension ===

#[test]
fn test_eval_list_comprehension_with_filter() {
    // [x IN [1, 2, 3, 4, 5] WHERE x > 2 | x * 10]
    // Should produce [30, 40, 50]
    let result = eval_literal_expr(FilterExpression::ListComprehension {
        variable: "x".to_string(),
        list_expr: Box::new(FilterExpression::List(vec![
            FilterExpression::Literal(Value::Int64(1)),
            FilterExpression::Literal(Value::Int64(2)),
            FilterExpression::Literal(Value::Int64(3)),
            FilterExpression::Literal(Value::Int64(4)),
            FilterExpression::Literal(Value::Int64(5)),
        ])),
        filter_expr: Some(Box::new(FilterExpression::Binary {
            left: Box::new(FilterExpression::Variable("x".to_string())),
            op: BinaryFilterOp::Gt,
            right: Box::new(FilterExpression::Literal(Value::Int64(2))),
        })),
        map_expr: Box::new(FilterExpression::Binary {
            left: Box::new(FilterExpression::Variable("x".to_string())),
            op: BinaryFilterOp::Mul,
            right: Box::new(FilterExpression::Literal(Value::Int64(10))),
        }),
    });

    if let Some(Value::List(items)) = result {
        assert_eq!(items.len(), 3);
        assert_eq!(items[0], Value::Int64(30));
        assert_eq!(items[1], Value::Int64(40));
        assert_eq!(items[2], Value::Int64(50));
    } else {
        panic!("Expected List, got {:?}", result);
    }
}

// === List Predicate (any/all/none/single) ===

#[test]
fn test_eval_list_predicate_any() {
    let result = eval_literal_expr(FilterExpression::ListPredicate {
        kind: ListPredicateKind::Any,
        variable: "x".to_string(),
        list_expr: Box::new(FilterExpression::List(vec![
            FilterExpression::Literal(Value::Int64(1)),
            FilterExpression::Literal(Value::Int64(5)),
            FilterExpression::Literal(Value::Int64(3)),
        ])),
        predicate: Box::new(FilterExpression::Binary {
            left: Box::new(FilterExpression::Variable("x".to_string())),
            op: BinaryFilterOp::Gt,
            right: Box::new(FilterExpression::Literal(Value::Int64(4))),
        }),
    });
    assert_eq!(result, Some(Value::Bool(true)));
}

#[test]
fn test_eval_list_predicate_all() {
    let result = eval_literal_expr(FilterExpression::ListPredicate {
        kind: ListPredicateKind::All,
        variable: "x".to_string(),
        list_expr: Box::new(FilterExpression::List(vec![
            FilterExpression::Literal(Value::Int64(10)),
            FilterExpression::Literal(Value::Int64(20)),
            FilterExpression::Literal(Value::Int64(30)),
        ])),
        predicate: Box::new(FilterExpression::Binary {
            left: Box::new(FilterExpression::Variable("x".to_string())),
            op: BinaryFilterOp::Gt,
            right: Box::new(FilterExpression::Literal(Value::Int64(5))),
        }),
    });
    assert_eq!(result, Some(Value::Bool(true)));
}

#[test]
fn test_eval_list_predicate_none() {
    let result = eval_literal_expr(FilterExpression::ListPredicate {
        kind: ListPredicateKind::None,
        variable: "x".to_string(),
        list_expr: Box::new(FilterExpression::List(vec![
            FilterExpression::Literal(Value::Int64(1)),
            FilterExpression::Literal(Value::Int64(2)),
            FilterExpression::Literal(Value::Int64(3)),
        ])),
        predicate: Box::new(FilterExpression::Binary {
            left: Box::new(FilterExpression::Variable("x".to_string())),
            op: BinaryFilterOp::Gt,
            right: Box::new(FilterExpression::Literal(Value::Int64(10))),
        }),
    });
    assert_eq!(result, Some(Value::Bool(true)));
}

#[test]
fn test_eval_list_predicate_single() {
    let result = eval_literal_expr(FilterExpression::ListPredicate {
        kind: ListPredicateKind::Single,
        variable: "x".to_string(),
        list_expr: Box::new(FilterExpression::List(vec![
            FilterExpression::Literal(Value::Int64(1)),
            FilterExpression::Literal(Value::Int64(5)),
            FilterExpression::Literal(Value::Int64(3)),
        ])),
        predicate: Box::new(FilterExpression::Binary {
            left: Box::new(FilterExpression::Variable("x".to_string())),
            op: BinaryFilterOp::Gt,
            right: Box::new(FilterExpression::Literal(Value::Int64(4))),
        }),
    });
    // Only x=5 satisfies x > 4, so exactly one
    assert_eq!(result, Some(Value::Bool(true)));
}

// === Map key access via index ===

#[test]
fn test_eval_map_key_access() {
    // {name: 'Alix'}['name'] = 'Alix'
    let result = eval_literal_expr(FilterExpression::IndexAccess {
        base: Box::new(FilterExpression::Map(vec![(
            "name".to_string(),
            FilterExpression::Literal(Value::String("Alix".into())),
        )])),
        index: Box::new(FilterExpression::Literal(Value::String("name".into()))),
    });
    assert_eq!(result, Some(Value::String("Alix".into())));
}

// === LIKE operator tests ===

#[test]
fn test_eval_like_wildcard() {
    // 'hello world' LIKE 'hello%'
    let result = eval_literal_expr(binary(
        Value::String("hello world".into()),
        BinaryFilterOp::Like,
        Value::String("hello%".into()),
    ));
    assert_eq!(result, Some(Value::Bool(true)));

    // 'hello world' LIKE '%world'
    let result = eval_literal_expr(binary(
        Value::String("hello world".into()),
        BinaryFilterOp::Like,
        Value::String("%world".into()),
    ));
    assert_eq!(result, Some(Value::Bool(true)));

    // 'hello world' LIKE '%llo%'
    let result = eval_literal_expr(binary(
        Value::String("hello world".into()),
        BinaryFilterOp::Like,
        Value::String("%llo%".into()),
    ));
    assert_eq!(result, Some(Value::Bool(true)));

    // 'hello' LIKE 'world%'
    let result = eval_literal_expr(binary(
        Value::String("hello".into()),
        BinaryFilterOp::Like,
        Value::String("world%".into()),
    ));
    assert_eq!(result, Some(Value::Bool(false)));
}

#[test]
fn test_eval_like_single_char() {
    // 'cat' LIKE 'c_t'
    let result = eval_literal_expr(binary(
        Value::String("cat".into()),
        BinaryFilterOp::Like,
        Value::String("c_t".into()),
    ));
    assert_eq!(result, Some(Value::Bool(true)));

    // 'cart' LIKE 'c_t'
    let result = eval_literal_expr(binary(
        Value::String("cart".into()),
        BinaryFilterOp::Like,
        Value::String("c_t".into()),
    ));
    assert_eq!(result, Some(Value::Bool(false)));
}

#[test]
fn test_eval_like_null() {
    // NULL LIKE '%' -> NULL
    let result = eval_literal_expr(binary(
        Value::Null,
        BinaryFilterOp::Like,
        Value::String("%".into()),
    ));
    assert_eq!(result, Some(Value::Null));
}

// === Concat operator (||) tests ===

#[test]
fn test_eval_concat_strings() {
    let result = eval_literal_expr(binary(
        Value::String("hello".into()),
        BinaryFilterOp::Concat,
        Value::String(" world".into()),
    ));
    assert_eq!(result, Some(Value::String("hello world".into())));
}

#[test]
fn test_eval_concat_string_with_int() {
    let result = eval_literal_expr(binary(
        Value::String("count: ".into()),
        BinaryFilterOp::Concat,
        Value::Int64(42),
    ));
    assert_eq!(result, Some(Value::String("count: 42".into())));
}

#[test]
fn test_eval_concat_int_with_string() {
    let result = eval_literal_expr(binary(
        Value::Int64(42),
        BinaryFilterOp::Concat,
        Value::String(" items".into()),
    ));
    assert_eq!(result, Some(Value::String("42 items".into())));
}

#[test]
fn test_eval_concat_null() {
    // Null || Null -> Null (hits the null arm)
    let result = eval_literal_expr(binary(Value::Null, BinaryFilterOp::Concat, Value::Null));
    assert_eq!(result, Some(Value::Null));
}

// === Modulo operator tests ===

#[test]
fn test_eval_modulo_float() {
    let result = eval_literal_expr(binary(
        Value::Float64(10.5),
        BinaryFilterOp::Mod,
        Value::Float64(3.0),
    ));
    if let Some(Value::Float64(v)) = result {
        assert!((v - 1.5).abs() < 0.001);
    } else {
        panic!("Expected Float64");
    }
}

#[test]
fn test_eval_modulo_mixed() {
    // int % float
    let result = eval_literal_expr(binary(
        Value::Int64(10),
        BinaryFilterOp::Mod,
        Value::Float64(3.0),
    ));
    if let Some(Value::Float64(v)) = result {
        assert!((v - 1.0).abs() < 0.001);
    } else {
        panic!("Expected Float64");
    }

    // float % int
    let result = eval_literal_expr(binary(
        Value::Float64(10.0),
        BinaryFilterOp::Mod,
        Value::Int64(3),
    ));
    if let Some(Value::Float64(v)) = result {
        assert!((v - 1.0).abs() < 0.001);
    } else {
        panic!("Expected Float64");
    }
}

#[test]
fn test_eval_modulo_by_zero() {
    let result = eval_literal_expr(binary(
        Value::Int64(10),
        BinaryFilterOp::Mod,
        Value::Int64(0),
    ));
    assert_eq!(result, None);

    let result = eval_literal_expr(binary(
        Value::Float64(10.0),
        BinaryFilterOp::Mod,
        Value::Float64(0.0),
    ));
    assert_eq!(result, None);
}

// === String addition with type coercion ===

#[test]
fn test_eval_string_add_int() {
    let result = eval_literal_expr(binary(
        Value::String("val:".into()),
        BinaryFilterOp::Add,
        Value::Int64(42),
    ));
    assert_eq!(result, Some(Value::String("val:42".into())));
}

#[test]
fn test_eval_string_add_bool() {
    let result = eval_literal_expr(binary(
        Value::String("is:".into()),
        BinaryFilterOp::Add,
        Value::Bool(true),
    ));
    assert_eq!(result, Some(Value::String("is:true".into())));
}

#[test]
fn test_eval_string_add_null() {
    let result = eval_literal_expr(binary(
        Value::String("val:".into()),
        BinaryFilterOp::Add,
        Value::Null,
    ));
    assert_eq!(result, Some(Value::Null));
}

// === Slice access tests ===

#[test]
fn test_eval_string_slice() {
    // "hello"[1..3] = "el"
    let result = eval_literal_expr(FilterExpression::SliceAccess {
        base: Box::new(FilterExpression::Literal(Value::String("hello".into()))),
        start: Some(Box::new(FilterExpression::Literal(Value::Int64(1)))),
        end: Some(Box::new(FilterExpression::Literal(Value::Int64(3)))),
    });
    assert_eq!(result, Some(Value::String("el".into())));
}

#[test]
fn test_eval_string_index_access() {
    // "hello"[1] = "e"
    let result = eval_literal_expr(FilterExpression::IndexAccess {
        base: Box::new(FilterExpression::Literal(Value::String("hello".into()))),
        index: Box::new(FilterExpression::Literal(Value::Int64(1))),
    });
    assert_eq!(result, Some(Value::String("e".into())));
}

#[test]
fn test_eval_string_negative_index() {
    // "hello"[-1] = "o"
    let result = eval_literal_expr(FilterExpression::IndexAccess {
        base: Box::new(FilterExpression::Literal(Value::String("hello".into()))),
        index: Box::new(FilterExpression::Literal(Value::Int64(-1))),
    });
    assert_eq!(result, Some(Value::String("o".into())));
}

// === Function tests for uncovered branches ===

#[test]
fn test_eval_tostring_types() {
    
    let store: Arc<dyn GraphStore> = Arc::new(SubstrateStore::open_tempfile().unwrap());
    let vc = HashMap::new();
    let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
    let chunk = builder.finish();

    // Bool -> String
    let pred = ExpressionPredicate::new(
        FilterExpression::FunctionCall {
            name: "toString".to_string(),
            args: vec![FilterExpression::Literal(Value::Bool(true))],
        },
        vc.clone(),
        Arc::clone(&store),
    );
    assert_eq!(pred.eval_at(&chunk, 0), Some(Value::String("true".into())));

    // Float -> String
    let pred = ExpressionPredicate::new(
        FilterExpression::FunctionCall {
            name: "toString".to_string(),
            args: vec![FilterExpression::Literal(Value::Float64(2.72))],
        },
        vc.clone(),
        Arc::clone(&store),
    );
    assert_eq!(pred.eval_at(&chunk, 0), Some(Value::String("2.72".into())));

    // Null -> Null
    let pred = ExpressionPredicate::new(
        FilterExpression::FunctionCall {
            name: "toString".to_string(),
            args: vec![FilterExpression::Literal(Value::Null)],
        },
        vc,
        store,
    );
    assert_eq!(pred.eval_at(&chunk, 0), Some(Value::Null));
}

#[test]
fn test_eval_toboolean() {
    
    let store: Arc<dyn GraphStore> = Arc::new(SubstrateStore::open_tempfile().unwrap());
    let vc = HashMap::new();
    let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
    let chunk = builder.finish();

    let pred = ExpressionPredicate::new(
        FilterExpression::FunctionCall {
            name: "toBoolean".to_string(),
            args: vec![FilterExpression::Literal(Value::String("true".into()))],
        },
        vc.clone(),
        Arc::clone(&store),
    );
    assert_eq!(pred.eval_at(&chunk, 0), Some(Value::Bool(true)));

    let pred = ExpressionPredicate::new(
        FilterExpression::FunctionCall {
            name: "toBoolean".to_string(),
            args: vec![FilterExpression::Literal(Value::String("false".into()))],
        },
        vc.clone(),
        Arc::clone(&store),
    );
    assert_eq!(pred.eval_at(&chunk, 0), Some(Value::Bool(false)));

    let pred = ExpressionPredicate::new(
        FilterExpression::FunctionCall {
            name: "toBoolean".to_string(),
            args: vec![FilterExpression::Literal(Value::Bool(true))],
        },
        vc,
        store,
    );
    assert_eq!(pred.eval_at(&chunk, 0), Some(Value::Bool(true)));
}

#[test]
fn test_eval_tofloat() {
    
    let store: Arc<dyn GraphStore> = Arc::new(SubstrateStore::open_tempfile().unwrap());
    let vc = HashMap::new();
    let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
    let chunk = builder.finish();

    let pred = ExpressionPredicate::new(
        FilterExpression::FunctionCall {
            name: "toFloat".to_string(),
            args: vec![FilterExpression::Literal(Value::String("2.72".into()))],
        },
        vc.clone(),
        Arc::clone(&store),
    );
    if let Some(Value::Float64(v)) = pred.eval_at(&chunk, 0) {
        assert!((v - 2.72).abs() < 0.001);
    } else {
        panic!("Expected Float64");
    }

    let pred = ExpressionPredicate::new(
        FilterExpression::FunctionCall {
            name: "toFloat".to_string(),
            args: vec![FilterExpression::Literal(Value::Int64(42))],
        },
        vc,
        store,
    );
    assert_eq!(pred.eval_at(&chunk, 0), Some(Value::Float64(42.0)));
}

#[test]
fn test_eval_tointeger_from_float() {
    
    let store: Arc<dyn GraphStore> = Arc::new(SubstrateStore::open_tempfile().unwrap());
    let vc = HashMap::new();
    let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
    let chunk = builder.finish();

    let pred = ExpressionPredicate::new(
        FilterExpression::FunctionCall {
            name: "toInteger".to_string(),
            args: vec![FilterExpression::Literal(Value::Float64(3.7))],
        },
        vc,
        store,
    );
    assert_eq!(pred.eval_at(&chunk, 0), Some(Value::Int64(3)));
}

#[test]
fn test_eval_reverse_list() {
    let result = eval_literal_expr(FilterExpression::FunctionCall {
        name: "reverse".to_string(),
        args: vec![FilterExpression::List(vec![
            FilterExpression::Literal(Value::Int64(1)),
            FilterExpression::Literal(Value::Int64(2)),
            FilterExpression::Literal(Value::Int64(3)),
        ])],
    });
    assert_eq!(
        result,
        Some(Value::List(
            vec![Value::Int64(3), Value::Int64(2), Value::Int64(1)].into()
        ))
    );
}

#[test]
fn test_eval_reverse_string() {
    let result = eval_literal_expr(FilterExpression::FunctionCall {
        name: "reverse".to_string(),
        args: vec![FilterExpression::Literal(Value::String("abc".into()))],
    });
    assert_eq!(result, Some(Value::String("cba".into())));
}

#[test]
fn test_eval_exists_function() {
    let result = eval_literal_expr(FilterExpression::FunctionCall {
        name: "exists".to_string(),
        args: vec![FilterExpression::Literal(Value::Int64(42))],
    });
    assert_eq!(result, Some(Value::Bool(true)));

    let result = eval_literal_expr(FilterExpression::FunctionCall {
        name: "exists".to_string(),
        args: vec![FilterExpression::Literal(Value::Null)],
    });
    assert_eq!(result, Some(Value::Bool(false)));
}
