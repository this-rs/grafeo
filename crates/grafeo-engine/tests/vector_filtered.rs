//! Integration tests for property-filtered vector search.
//!
//! Tests that `vector_search`, `batch_vector_search`, and `mmr_search`
//! correctly restrict results when property equality filters are provided.

#![cfg(feature = "vector-index")]

use std::collections::HashMap;

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// Helper: create a 3D vector value.
fn vec3(x: f32, y: f32, z: f32) -> Value {
    Value::Vector(vec![x, y, z].into())
}

/// Sets up a database with 6 Doc nodes, each with a vector and a `user_id` property.
fn setup_db() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();

    // user_id=1: nodes near [1, 0, 0]
    let n1 = db.create_node(&["Doc"]);
    db.set_node_property(n1, "emb", vec3(1.0, 0.0, 0.0));
    db.set_node_property(n1, "user_id", Value::Int64(1));

    let n2 = db.create_node(&["Doc"]);
    db.set_node_property(n2, "emb", vec3(0.95, 0.05, 0.0));
    db.set_node_property(n2, "user_id", Value::Int64(1));

    // user_id=2: nodes near [0, 1, 0]
    let n3 = db.create_node(&["Doc"]);
    db.set_node_property(n3, "emb", vec3(0.0, 1.0, 0.0));
    db.set_node_property(n3, "user_id", Value::Int64(2));

    let n4 = db.create_node(&["Doc"]);
    db.set_node_property(n4, "emb", vec3(0.05, 0.95, 0.0));
    db.set_node_property(n4, "user_id", Value::Int64(2));

    // user_id=3: node near [0, 0, 1]
    let n5 = db.create_node(&["Doc"]);
    db.set_node_property(n5, "emb", vec3(0.0, 0.0, 1.0));
    db.set_node_property(n5, "user_id", Value::Int64(3));

    // No user_id property
    let n6 = db.create_node(&["Doc"]);
    db.set_node_property(n6, "emb", vec3(0.5, 0.5, 0.0));

    // Create property index for fast lookups
    db.create_property_index("user_id");

    db.create_vector_index("Doc", "emb", Some(3), Some("cosine"), None, None)
        .expect("create index");

    let _ = (n1, n2, n3, n4, n5, n6);
    db
}

#[test]
fn test_filtered_vector_search_by_user_id() {
    let db = setup_db();

    // Search for vectors near [1, 0, 0] but only among user_id=2 nodes
    let filters: HashMap<String, Value> = [("user_id".to_string(), Value::Int64(2))]
        .into_iter()
        .collect();

    let results = db
        .vector_search("Doc", "emb", &[1.0, 0.0, 0.0], 5, None, Some(&filters))
        .expect("filtered search");

    // Should only return user_id=2 nodes (n3, n4)
    assert!(!results.is_empty());
    assert!(results.len() <= 2);

    // Verify all results have user_id=2
    for (id, _) in &results {
        let node = db.get_node(*id).expect("node exists");
        let uid = node
            .properties
            .get(&grafeo_common::types::PropertyKey::new("user_id"))
            .expect("has user_id");
        assert_eq!(uid, &Value::Int64(2), "result should be user_id=2");
    }
}

#[test]
fn test_filtered_search_without_filters_returns_all() {
    let db = setup_db();

    // No filters — should return all matching nodes
    let results = db
        .vector_search("Doc", "emb", &[0.5, 0.5, 0.0], 10, None, None)
        .expect("unfiltered search");

    assert_eq!(results.len(), 6, "should find all 6 Doc nodes");
}

#[test]
fn test_filtered_search_empty_filters_returns_all() {
    let db = setup_db();

    // Empty filter map — should behave like no filters
    let filters: HashMap<String, Value> = HashMap::new();
    let results = db
        .vector_search("Doc", "emb", &[0.5, 0.5, 0.0], 10, None, Some(&filters))
        .expect("empty filter search");

    assert_eq!(results.len(), 6, "empty filters should return all nodes");
}

#[test]
fn test_filtered_search_no_matches() {
    let db = setup_db();

    // user_id=999 doesn't exist
    let filters: HashMap<String, Value> = [("user_id".to_string(), Value::Int64(999))]
        .into_iter()
        .collect();

    let results = db
        .vector_search("Doc", "emb", &[1.0, 0.0, 0.0], 5, None, Some(&filters))
        .expect("filtered search");

    assert!(results.is_empty(), "no matching nodes should return empty");
}

#[test]
fn test_batch_vector_search_with_filters() {
    let db = setup_db();

    let filters: HashMap<String, Value> = [("user_id".to_string(), Value::Int64(1))]
        .into_iter()
        .collect();

    let queries = vec![vec![1.0f32, 0.0, 0.0], vec![0.0, 1.0, 0.0]];

    let results = db
        .batch_vector_search("Doc", "emb", &queries, 5, None, Some(&filters))
        .expect("batch filtered search");

    assert_eq!(results.len(), 2);

    // All results in both queries should be user_id=1
    for query_results in &results {
        for (id, _) in query_results {
            let node = db.get_node(*id).expect("node exists");
            let uid = node
                .properties
                .get(&grafeo_common::types::PropertyKey::new("user_id"))
                .expect("has user_id");
            assert_eq!(uid, &Value::Int64(1));
        }
    }
}

#[test]
fn test_mmr_search_with_filters() {
    let db = setup_db();

    let filters: HashMap<String, Value> = [("user_id".to_string(), Value::Int64(2))]
        .into_iter()
        .collect();

    let results = db
        .mmr_search(
            "Doc",
            "emb",
            &[0.0, 1.0, 0.0],
            2,
            None,
            None,
            None,
            Some(&filters),
        )
        .expect("mmr filtered search");

    assert!(!results.is_empty());
    assert!(results.len() <= 2);

    for (id, _) in &results {
        let node = db.get_node(*id).expect("node exists");
        let uid = node
            .properties
            .get(&grafeo_common::types::PropertyKey::new("user_id"))
            .expect("has user_id");
        assert_eq!(uid, &Value::Int64(2));
    }
}

#[test]
fn test_filtered_search_non_indexed_property() {
    let db = setup_db();

    // Filter on a property that is NOT indexed (no property index for "user_id=3"
    // actually it IS indexed, but let's use a different property)
    // Add a "category" property without creating a property index
    // Get the first node and add a category
    let results_all = db
        .vector_search("Doc", "emb", &[1.0, 0.0, 0.0], 6, None, None)
        .expect("find all");

    // Set "category" on first 2 nodes only
    for (id, _) in results_all.iter().take(2) {
        db.set_node_property(*id, "category", Value::String("science".into()));
    }

    // No property index for "category" — should still work (scan fallback)
    let filters: HashMap<String, Value> =
        [("category".to_string(), Value::String("science".into()))]
            .into_iter()
            .collect();

    let results = db
        .vector_search("Doc", "emb", &[1.0, 0.0, 0.0], 10, None, Some(&filters))
        .expect("filtered search on non-indexed property");

    assert!(results.len() <= 2, "at most 2 nodes have category=science");
}
