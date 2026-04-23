//! Integration tests for
//! `obrain_core::execution::operators::factorized_filter::PropertyPredicate`
//! against the substrate backend.
//!
//! ## Why this lives in obrain-substrate and not in obrain-core
//!
//! Relocated as part of T17 Step 3 W2/Class-2 follow-up (decision
//! `b1dfe229`). `obrain-core` cannot take `obrain-substrate` as a
//! dev-dependency without creating two distinct compilation units of
//! `obrain-core` (gotcha `598dda40`). The forward direction
//! (`obrain-substrate → obrain-core`) has no cycle, so moving the
//! `property_predicate_tests` sub-module into an integration test of
//! `obrain-substrate` is the post-T17-cutover home for the LPG-era
//! fixtures.
//!
//! Only the `property_predicate_tests` sub-module was LpgStore-coupled;
//! the outer `tests` module in obrain-core stays put because it only
//! exercises FactorizedChunk-level filter predicates that do not touch
//! any GraphStore.
//!
//! Run with:
//!
//! ```bash
//! cargo test -p obrain-substrate --test operators_factorized_filter
//! ```

use obrain_common::types::{LogicalType, Value};
use obrain_core::execution::factorized_chunk::{FactorizationLevel, FactorizedChunk};
use obrain_core::execution::factorized_vector::FactorizedVector;
use obrain_core::execution::operators::{
    FactorizedCompareOp as CompareOp, FactorizedPredicate, PropertyPredicate,
};
use obrain_core::execution::vector::ValueVector;
use obrain_core::graph::traits::GraphStoreMut;
use obrain_substrate::SubstrateStore;
use std::sync::Arc;

fn create_test_store() -> Arc<SubstrateStore> {
    Arc::new(SubstrateStore::open_tempfile().unwrap())
}

fn create_chunk_with_node_ids(store: &Arc<SubstrateStore>) -> FactorizedChunk {
    // Create some nodes with properties
    let node1 = store.create_node(&["Person"]);
    let node2 = store.create_node(&["Person"]);
    let node3 = store.create_node(&["Person"]);

    store.set_node_property(node1, "age", Value::Int64(25));
    store.set_node_property(node2, "age", Value::Int64(35));
    store.set_node_property(node3, "age", Value::Int64(45));

    store.set_node_property(node1, "name", Value::String("Alix".into()));
    store.set_node_property(node2, "name", Value::String("Gus".into()));
    store.set_node_property(node3, "name", Value::String("Harm".into()));

    // Create a chunk with node IDs — capture the returned ids so we don't
    // hardcode NodeId::new(N) across backends (gotcha `f95990d2`).
    let mut node_data = ValueVector::with_type(LogicalType::Node);
    node_data.push_node_id(node1);
    node_data.push_node_id(node2);
    node_data.push_node_id(node3);

    let level0 = FactorizationLevel::flat(
        vec![FactorizedVector::flat(node_data)],
        vec!["n".to_string()],
    );

    let mut chunk = FactorizedChunk::empty();
    chunk.add_factorized_level(level0);
    chunk
}

#[test]
fn test_property_predicate_eq_int() {
    let store = create_test_store();
    let chunk = create_chunk_with_node_ids(&store);

    // Predicate: age = 35
    let pred = PropertyPredicate::eq(0, 0, "age", Value::Int64(35), store.clone());

    assert!(!pred.evaluate(&chunk, 0, 0)); // Alix age=25
    assert!(pred.evaluate(&chunk, 0, 1)); // Gus age=35
    assert!(!pred.evaluate(&chunk, 0, 2)); // Harm age=45
}

#[test]
fn test_property_predicate_gt_int() {
    let store = create_test_store();
    let chunk = create_chunk_with_node_ids(&store);

    // Predicate: age > 30
    let pred =
        PropertyPredicate::new(0, 0, "age", CompareOp::Gt, Value::Int64(30), store.clone());

    assert!(!pred.evaluate(&chunk, 0, 0)); // 25 > 30 = false
    assert!(pred.evaluate(&chunk, 0, 1)); // 35 > 30 = true
    assert!(pred.evaluate(&chunk, 0, 2)); // 45 > 30 = true
}

#[test]
fn test_property_predicate_lt_le_ge() {
    let store = create_test_store();
    let chunk = create_chunk_with_node_ids(&store);

    // age < 35
    let pred_lt =
        PropertyPredicate::new(0, 0, "age", CompareOp::Lt, Value::Int64(35), store.clone());
    assert!(pred_lt.evaluate(&chunk, 0, 0)); // 25 < 35
    assert!(!pred_lt.evaluate(&chunk, 0, 1)); // 35 < 35 = false

    // age <= 35
    let pred_le =
        PropertyPredicate::new(0, 0, "age", CompareOp::Le, Value::Int64(35), store.clone());
    assert!(pred_le.evaluate(&chunk, 0, 0)); // 25 <= 35
    assert!(pred_le.evaluate(&chunk, 0, 1)); // 35 <= 35
    assert!(!pred_le.evaluate(&chunk, 0, 2)); // 45 <= 35 = false

    // age >= 35
    let pred_ge =
        PropertyPredicate::new(0, 0, "age", CompareOp::Ge, Value::Int64(35), store.clone());
    assert!(!pred_ge.evaluate(&chunk, 0, 0)); // 25 >= 35 = false
    assert!(pred_ge.evaluate(&chunk, 0, 1)); // 35 >= 35
    assert!(pred_ge.evaluate(&chunk, 0, 2)); // 45 >= 35
}

#[test]
fn test_property_predicate_ne() {
    let store = create_test_store();
    let chunk = create_chunk_with_node_ids(&store);

    let pred =
        PropertyPredicate::new(0, 0, "age", CompareOp::Ne, Value::Int64(35), store.clone());

    assert!(pred.evaluate(&chunk, 0, 0)); // 25 != 35
    assert!(!pred.evaluate(&chunk, 0, 1)); // 35 != 35 = false
    assert!(pred.evaluate(&chunk, 0, 2)); // 45 != 35
}

#[test]
fn test_property_predicate_string() {
    let store = create_test_store();
    let chunk = create_chunk_with_node_ids(&store);

    // name = "Gus"
    let pred =
        PropertyPredicate::eq(0, 0, "name", Value::String("Gus".into()), store.clone());

    assert!(!pred.evaluate(&chunk, 0, 0)); // Alix
    assert!(pred.evaluate(&chunk, 0, 1)); // Gus
    assert!(!pred.evaluate(&chunk, 0, 2)); // Harm

    // name < "Gus"
    let pred_lt = PropertyPredicate::new(
        0,
        0,
        "name",
        CompareOp::Lt,
        Value::String("Gus".into()),
        store.clone(),
    );
    assert!(pred_lt.evaluate(&chunk, 0, 0)); // "Alix" < "Gus"
    assert!(!pred_lt.evaluate(&chunk, 0, 1)); // "Gus" < "Gus" = false

    // name > "Gus"
    let pred_gt = PropertyPredicate::new(
        0,
        0,
        "name",
        CompareOp::Gt,
        Value::String("Gus".into()),
        store.clone(),
    );
    assert!(pred_gt.evaluate(&chunk, 0, 2)); // "Harm" > "Gus"
    assert!(!pred_gt.evaluate(&chunk, 0, 0)); // "Alix" > "Gus" = false
}

#[test]
fn test_property_predicate_float() {
    let store = create_test_store();

    // Add float properties
    let node1 = store.create_node(&["Thing"]);
    let node2 = store.create_node(&["Thing"]);
    store.set_node_property(node1, "score", Value::Float64(1.5));
    store.set_node_property(node2, "score", Value::Float64(2.5));

    let mut node_data = ValueVector::with_type(LogicalType::Node);
    node_data.push_node_id(node1);
    node_data.push_node_id(node2);

    let level0 = FactorizationLevel::flat(
        vec![FactorizedVector::flat(node_data)],
        vec!["n".to_string()],
    );
    let mut chunk = FactorizedChunk::empty();
    chunk.add_factorized_level(level0);

    // score = 2.5
    let pred_eq = PropertyPredicate::new(
        0,
        0,
        "score",
        CompareOp::Eq,
        Value::Float64(2.5),
        store.clone(),
    );
    assert!(!pred_eq.evaluate(&chunk, 0, 0));
    assert!(pred_eq.evaluate(&chunk, 0, 1));

    // score != 2.5
    let pred_ne = PropertyPredicate::new(
        0,
        0,
        "score",
        CompareOp::Ne,
        Value::Float64(2.5),
        store.clone(),
    );
    assert!(pred_ne.evaluate(&chunk, 0, 0));
    assert!(!pred_ne.evaluate(&chunk, 0, 1));

    // score > 2.0
    let pred_gt = PropertyPredicate::new(
        0,
        0,
        "score",
        CompareOp::Gt,
        Value::Float64(2.0),
        store.clone(),
    );
    assert!(!pred_gt.evaluate(&chunk, 0, 0)); // 1.5 > 2.0 = false
    assert!(pred_gt.evaluate(&chunk, 0, 1)); // 2.5 > 2.0

    // score < 2.0
    let pred_lt = PropertyPredicate::new(
        0,
        0,
        "score",
        CompareOp::Lt,
        Value::Float64(2.0),
        store.clone(),
    );
    assert!(pred_lt.evaluate(&chunk, 0, 0)); // 1.5 < 2.0

    // score <= 1.5
    let pred_le = PropertyPredicate::new(
        0,
        0,
        "score",
        CompareOp::Le,
        Value::Float64(1.5),
        store.clone(),
    );
    assert!(pred_le.evaluate(&chunk, 0, 0)); // 1.5 <= 1.5

    // score >= 2.5
    let pred_ge = PropertyPredicate::new(
        0,
        0,
        "score",
        CompareOp::Ge,
        Value::Float64(2.5),
        store.clone(),
    );
    assert!(pred_ge.evaluate(&chunk, 0, 1)); // 2.5 >= 2.5
}

#[test]
fn test_property_predicate_bool() {
    let store = create_test_store();

    let node1 = store.create_node(&["Flag"]);
    let node2 = store.create_node(&["Flag"]);
    store.set_node_property(node1, "active", Value::Bool(true));
    store.set_node_property(node2, "active", Value::Bool(false));

    let mut node_data = ValueVector::with_type(LogicalType::Node);
    node_data.push_node_id(node1);
    node_data.push_node_id(node2);

    let level0 = FactorizationLevel::flat(
        vec![FactorizedVector::flat(node_data)],
        vec!["n".to_string()],
    );
    let mut chunk = FactorizedChunk::empty();
    chunk.add_factorized_level(level0);

    // active = true
    let pred = PropertyPredicate::eq(0, 0, "active", Value::Bool(true), store.clone());
    assert!(pred.evaluate(&chunk, 0, 0));
    assert!(!pred.evaluate(&chunk, 0, 1));

    // active != true
    let pred_ne = PropertyPredicate::new(
        0,
        0,
        "active",
        CompareOp::Ne,
        Value::Bool(true),
        store.clone(),
    );
    assert!(!pred_ne.evaluate(&chunk, 0, 0));
    assert!(pred_ne.evaluate(&chunk, 0, 1));
}

#[test]
fn test_property_predicate_missing_property() {
    let store = create_test_store();
    let chunk = create_chunk_with_node_ids(&store);

    // Property "nonexistent" doesn't exist
    let pred = PropertyPredicate::eq(0, 0, "nonexistent", Value::Int64(1), store.clone());

    // Should return false for missing property
    assert!(!pred.evaluate(&chunk, 0, 0));
    assert!(!pred.evaluate(&chunk, 0, 1));
}

#[test]
fn test_property_predicate_wrong_level() {
    let store = create_test_store();
    let chunk = create_chunk_with_node_ids(&store);

    // Predicate targets level 1, but chunk only has level 0
    let pred = PropertyPredicate::eq(1, 0, "age", Value::Int64(35), store.clone());

    // Should return true when evaluated at wrong level
    assert!(pred.evaluate(&chunk, 0, 0));
}

#[test]
fn test_property_predicate_invalid_column() {
    let store = create_test_store();
    let chunk = create_chunk_with_node_ids(&store);

    // Column 5 doesn't exist
    let pred = PropertyPredicate::eq(0, 5, "age", Value::Int64(35), store.clone());

    assert!(!pred.evaluate(&chunk, 0, 0));
}

#[test]
fn test_property_predicate_target_level() {
    let store = create_test_store();
    let pred = PropertyPredicate::eq(2, 0, "age", Value::Int64(35), store);
    assert_eq!(pred.target_level(), Some(2));
}

#[test]
fn test_property_predicate_batch() {
    let store = create_test_store();
    let chunk = create_chunk_with_node_ids(&store);

    // Predicate: age > 30
    let pred =
        PropertyPredicate::new(0, 0, "age", CompareOp::Gt, Value::Int64(30), store.clone());

    let selection = pred.evaluate_batch(&chunk, 0);

    // Should select indices 1 and 2 (Gus=35, Harm=45)
    assert_eq!(selection.selected_count(), 2);
    assert!(!selection.is_selected(0)); // Alix=25
    assert!(selection.is_selected(1)); // Gus=35
    assert!(selection.is_selected(2)); // Harm=45
}

#[test]
fn test_property_predicate_batch_wrong_level() {
    let store = create_test_store();
    let chunk = create_chunk_with_node_ids(&store);

    // Predicate targets level 1
    let pred =
        PropertyPredicate::new(1, 0, "age", CompareOp::Gt, Value::Int64(30), store.clone());

    // Batch evaluate at level 0 - should return all selected
    let selection = pred.evaluate_batch(&chunk, 0);
    assert_eq!(selection.selected_count(), 3);
}

#[test]
fn test_property_predicate_batch_invalid_level() {
    let store = create_test_store();
    let chunk = create_chunk_with_node_ids(&store);

    // Predicate targets level 5 which doesn't exist
    let pred =
        PropertyPredicate::new(5, 0, "age", CompareOp::Gt, Value::Int64(30), store.clone());

    let selection = pred.evaluate_batch(&chunk, 5);
    assert_eq!(selection.selected_count(), 0);
}

#[test]
fn test_property_predicate_batch_invalid_column() {
    let store = create_test_store();
    let chunk = create_chunk_with_node_ids(&store);

    // Column 5 doesn't exist
    let pred =
        PropertyPredicate::new(0, 5, "age", CompareOp::Gt, Value::Int64(30), store.clone());

    let selection = pred.evaluate_batch(&chunk, 0);
    // Should return all false (no matches)
    assert_eq!(selection.selected_count(), 0);
}

#[test]
fn test_property_predicate_type_mismatch() {
    let store = create_test_store();
    let chunk = create_chunk_with_node_ids(&store);

    // age is Int64, but we compare with String
    let pred =
        PropertyPredicate::eq(0, 0, "age", Value::String("35".into()), store.clone());

    // Type mismatch should return false
    assert!(!pred.evaluate(&chunk, 0, 1));
}
