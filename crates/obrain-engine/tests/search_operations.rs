//! Integration tests for search operations covering uncovered paths.
//!
//! Targets low-coverage areas in `database/search.rs`:
//! - hybrid_search (previously untested)
//! - text_search error paths
//! - batch_vector_search with filters
//! - mmr_search with filters
//!
//! ```bash
//! cargo test -p grafeo-engine --features full --test search_operations
//! ```

// ============================================================================
// Vector search tests
// ============================================================================

#[cfg(feature = "vector-index")]
mod vector {
    use grafeo_common::types::Value;
    use grafeo_engine::GrafeoDB;
    use std::collections::HashMap;

    fn vec3(x: f32, y: f32, z: f32) -> Value {
        Value::Vector(vec![x, y, z].into())
    }

    fn setup_vector_db() -> GrafeoDB {
        let db = GrafeoDB::new_in_memory();

        let n1 = db.create_node(&["Doc"]);
        db.set_node_property(n1, "emb", vec3(1.0, 0.0, 0.0));
        db.set_node_property(n1, "category", Value::String("science".into()));

        let n2 = db.create_node(&["Doc"]);
        db.set_node_property(n2, "emb", vec3(0.0, 1.0, 0.0));
        db.set_node_property(n2, "category", Value::String("science".into()));

        let n3 = db.create_node(&["Doc"]);
        db.set_node_property(n3, "emb", vec3(0.0, 0.0, 1.0));
        db.set_node_property(n3, "category", Value::String("art".into()));

        let n4 = db.create_node(&["Doc"]);
        db.set_node_property(n4, "emb", vec3(0.9, 0.1, 0.0));
        db.set_node_property(n4, "category", Value::String("science".into()));

        db.create_property_index("category");
        db.create_vector_index("Doc", "emb", Some(3), Some("cosine"), None, None)
            .expect("create vector index");

        db
    }

    #[test]
    fn test_vector_search_no_index_error() {
        let db = GrafeoDB::new_in_memory();
        let n = db.create_node(&["Doc"]);
        db.set_node_property(n, "emb", vec3(1.0, 0.0, 0.0));

        // No vector index created: search should fail
        let result = db.vector_search("Doc", "emb", &[1.0, 0.0, 0.0], 5, None, None);
        assert!(result.is_err(), "search without index should error");
    }

    #[test]
    fn test_vector_search_with_ef_parameter() {
        let db = setup_vector_db();

        // ef parameter controls search quality (higher = better, slower)
        let results = db
            .vector_search("Doc", "emb", &[1.0, 0.0, 0.0], 2, Some(50), None)
            .expect("search with ef");

        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_batch_vector_search_multiple_queries() {
        let db = setup_vector_db();

        let queries = vec![vec![1.0_f32, 0.0, 0.0], vec![0.0, 1.0, 0.0]];

        let results = db
            .batch_vector_search("Doc", "emb", &queries, 2, None, None)
            .expect("batch search");

        assert_eq!(results.len(), 2);
        // Each query should return up to 2 results
        for result_set in &results {
            assert!(result_set.len() <= 2);
            assert!(!result_set.is_empty());
        }
    }

    #[test]
    fn test_batch_vector_search_with_filter() {
        let db = setup_vector_db();

        let mut filters = HashMap::new();
        filters.insert("category".to_string(), Value::String("science".into()));

        let queries = vec![vec![1.0_f32, 0.0, 0.0]];

        let results = db
            .batch_vector_search("Doc", "emb", &queries, 10, None, Some(&filters))
            .expect("batch search with filter");

        assert_eq!(results.len(), 1);
        // Only science docs should be returned (3 out of 4)
        assert_eq!(results[0].len(), 3);
    }

    #[test]
    fn test_mmr_search_with_filter() {
        let db = setup_vector_db();

        let mut filters = HashMap::new();
        filters.insert("category".to_string(), Value::String("science".into()));

        let results = db
            .mmr_search(
                "Doc",
                "emb",
                &[1.0, 0.0, 0.0],
                2,
                None,
                None,
                None,
                Some(&filters),
            )
            .expect("mmr with filter");

        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_vector_search_k_larger_than_dataset() {
        let db = setup_vector_db();

        // Request more results than nodes exist
        let results = db
            .vector_search("Doc", "emb", &[1.0, 0.0, 0.0], 100, None, None)
            .expect("k > dataset");

        // Should return all 4 nodes, not error
        assert_eq!(results.len(), 4);
    }

    #[test]
    fn test_drop_and_recreate_vector_index() {
        let db = setup_vector_db();

        // Search works
        let r1 = db
            .vector_search("Doc", "emb", &[1.0, 0.0, 0.0], 2, None, None)
            .expect("search before drop");
        assert_eq!(r1.len(), 2);

        // Drop index
        assert!(db.drop_vector_index("Doc", "emb"));

        // Search should fail
        let err = db.vector_search("Doc", "emb", &[1.0, 0.0, 0.0], 2, None, None);
        assert!(err.is_err(), "search after drop should error");

        // Recreate index
        db.create_vector_index("Doc", "emb", Some(3), Some("cosine"), None, None)
            .expect("recreate index");

        // Search works again
        let r2 = db
            .vector_search("Doc", "emb", &[1.0, 0.0, 0.0], 2, None, None)
            .expect("search after recreate");
        assert_eq!(r2.len(), 2);
    }
}

// ============================================================================
// Text search tests
// ============================================================================

#[cfg(feature = "text-index")]
mod text {
    use grafeo_common::types::Value;
    use grafeo_engine::GrafeoDB;

    fn setup_text_db() -> GrafeoDB {
        let db = GrafeoDB::new_in_memory();

        let n1 = db.create_node(&["Article"]);
        db.set_node_property(n1, "title", Value::String("Rust graph database".into()));

        let n2 = db.create_node(&["Article"]);
        db.set_node_property(n2, "title", Value::String("Python machine learning".into()));

        let n3 = db.create_node(&["Article"]);
        db.set_node_property(
            n3,
            "title",
            Value::String("Rust systems programming".into()),
        );

        db.create_text_index("Article", "title")
            .expect("create text index");

        db
    }

    #[test]
    fn test_text_search_basic() {
        let db = setup_text_db();

        let results = db.text_search("Article", "title", "Rust", 10).unwrap();

        // Should match articles with "Rust"
        assert!(results.len() >= 2, "expected at least 2 Rust articles");
    }

    #[test]
    fn test_text_search_no_index_error() {
        let db = GrafeoDB::new_in_memory();
        let n = db.create_node(&["Article"]);
        db.set_node_property(n, "title", Value::String("test".into()));

        // No text index: should error
        let result = db.text_search("Article", "title", "test", 10);
        assert!(result.is_err(), "text search without index should error");
    }

    #[test]
    fn test_text_search_no_matches() {
        let db = setup_text_db();

        let results = db
            .text_search("Article", "title", "nonexistentxyz", 10)
            .unwrap();

        assert!(results.is_empty(), "no matches expected for nonsense query");
    }

    #[test]
    fn test_text_search_after_mutation() {
        let db = setup_text_db();

        // Add a new article
        let n = db.create_node(&["Article"]);
        db.set_node_property(n, "title", Value::String("Rust web framework".into()));

        let results = db.text_search("Article", "title", "Rust", 10).unwrap();

        // Should now include the new article
        assert!(
            results.len() >= 3,
            "expected at least 3 Rust articles after mutation"
        );
    }

    #[test]
    fn test_drop_and_rebuild_text_index() {
        let db = setup_text_db();

        // Search works
        let r1 = db.text_search("Article", "title", "Rust", 10).unwrap();
        assert!(!r1.is_empty());

        // Drop index
        assert!(db.drop_text_index("Article", "title"));

        // Search should fail
        let err = db.text_search("Article", "title", "Rust", 10);
        assert!(err.is_err());

        // Rebuild index
        db.rebuild_text_index("Article", "title").unwrap();

        // Search works again
        let r2 = db.text_search("Article", "title", "Rust", 10).unwrap();
        assert!(!r2.is_empty());
    }
}

// ============================================================================
// Hybrid search tests
// ============================================================================

#[cfg(feature = "hybrid-search")]
mod hybrid {
    use grafeo_common::types::Value;
    use grafeo_engine::GrafeoDB;

    fn vec3(x: f32, y: f32, z: f32) -> Value {
        Value::Vector(vec![x, y, z].into())
    }

    fn setup_hybrid_db() -> GrafeoDB {
        let db = GrafeoDB::new_in_memory();

        let n1 = db.create_node(&["Doc"]);
        db.set_node_property(
            n1,
            "content",
            Value::String("Rust graph database engine".into()),
        );
        db.set_node_property(n1, "emb", vec3(1.0, 0.0, 0.0));

        let n2 = db.create_node(&["Doc"]);
        db.set_node_property(
            n2,
            "content",
            Value::String("Python machine learning framework".into()),
        );
        db.set_node_property(n2, "emb", vec3(0.0, 1.0, 0.0));

        let n3 = db.create_node(&["Doc"]);
        db.set_node_property(
            n3,
            "content",
            Value::String("Rust systems programming language".into()),
        );
        db.set_node_property(n3, "emb", vec3(0.9, 0.1, 0.0));

        let n4 = db.create_node(&["Doc"]);
        db.set_node_property(
            n4,
            "content",
            Value::String("Graph neural network research".into()),
        );
        db.set_node_property(n4, "emb", vec3(0.5, 0.5, 0.0));

        // Create both indexes
        db.create_text_index("Doc", "content")
            .expect("create text index");
        db.create_vector_index("Doc", "emb", Some(3), Some("cosine"), None, None)
            .expect("create vector index");

        db
    }

    #[test]
    fn test_hybrid_search_basic() {
        let db = setup_hybrid_db();

        let results = db
            .hybrid_search(
                "Doc",
                "content",
                "emb",
                "Rust graph",
                Some(&[1.0, 0.0, 0.0]),
                4,
                None,
            )
            .expect("hybrid search");

        assert!(!results.is_empty(), "hybrid search should return results");

        // "Rust graph database engine" should rank highest: matches both
        // text ("Rust graph") and vector (closest to [1,0,0])
        let top_node = results[0].0;
        let top_props = db.get_node(top_node).expect("top node exists");
        let content = top_props
            .properties
            .get(&grafeo_common::types::PropertyKey::new("content"))
            .expect("has content");
        if let Value::String(s) = content {
            assert!(
                s.contains("Rust") || s.contains("graph"),
                "top result should match query terms, got: {s}"
            );
        }
    }

    #[test]
    fn test_hybrid_search_text_only() {
        let db = setup_hybrid_db();

        // No vector query: only text search contributes
        let results = db
            .hybrid_search("Doc", "content", "emb", "Rust", None, 4, None)
            .expect("text-only hybrid");

        assert!(
            !results.is_empty(),
            "text-only hybrid should return results"
        );
    }

    #[test]
    fn test_hybrid_search_no_matches() {
        let db = setup_hybrid_db();

        let results = db
            .hybrid_search(
                "Doc",
                "content",
                "emb",
                "nonexistentxyzquery",
                Some(&[0.0, 0.0, 0.0]),
                4,
                None,
            )
            .expect("hybrid no matches");

        // Even with no text matches, vector search may return results
        // Just verify it doesn't error
        let _ = results;
    }
}

// ============================================================================
// Concurrent index access (T3-05)
// ============================================================================

#[cfg(feature = "vector-index")]
mod concurrent_vector {
    use grafeo_common::types::Value;
    use grafeo_engine::GrafeoDB;

    fn vec3(x: f32, y: f32, z: f32) -> Value {
        Value::Vector(vec![x, y, z].into())
    }

    #[test]
    fn test_concurrent_vector_read_during_write() {
        let db = std::sync::Arc::new(GrafeoDB::new_in_memory());

        // Seed initial data
        let n1 = db.create_node(&["Doc"]);
        db.set_node_property(n1, "emb", vec3(1.0, 0.0, 0.0));
        db.create_vector_index("Doc", "emb", Some(3), Some("cosine"), None, None)
            .unwrap();

        let db_read = std::sync::Arc::clone(&db);
        let db_write = std::sync::Arc::clone(&db);

        // Writer thread: add more nodes
        let writer = std::thread::spawn(move || {
            for i in 0..10 {
                let n = db_write.create_node(&["Doc"]);
                let x = (i as f32) / 10.0;
                db_write.set_node_property(n, "emb", vec3(x, 1.0 - x, 0.0));
            }
        });

        // Reader thread: search concurrently
        let reader = std::thread::spawn(move || {
            for _ in 0..10 {
                let results = db_read.vector_search("Doc", "emb", &[1.0, 0.0, 0.0], 5, None, None);
                // Should not panic or error
                assert!(results.is_ok(), "concurrent read should not error");
            }
        });

        writer.join().expect("writer thread should not panic");
        reader.join().expect("reader thread should not panic");
    }
}

#[cfg(feature = "text-index")]
mod concurrent_text {
    use grafeo_common::types::Value;
    use grafeo_engine::GrafeoDB;

    #[test]
    fn test_concurrent_text_read_during_write() {
        let db = std::sync::Arc::new(GrafeoDB::new_in_memory());

        let n1 = db.create_node(&["Doc"]);
        db.set_node_property(
            n1,
            "content",
            Value::String("initial document about graphs".into()),
        );
        db.create_text_index("Doc", "content").unwrap();

        let db_read = std::sync::Arc::clone(&db);
        let db_write = std::sync::Arc::clone(&db);

        let writer = std::thread::spawn(move || {
            for i in 0..10 {
                let n = db_write.create_node(&["Doc"]);
                db_write.set_node_property(
                    n,
                    "content",
                    Value::String(format!("document number {i} about databases").into()),
                );
            }
        });

        let reader = std::thread::spawn(move || {
            for _ in 0..10 {
                let results = db_read.text_search("Doc", "content", "database", 5);
                assert!(results.is_ok(), "concurrent text read should not error");
            }
        });

        writer.join().expect("writer should not panic");
        reader.join().expect("reader should not panic");
    }
}
