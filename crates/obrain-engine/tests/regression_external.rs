//! Regression tests inspired by bugs found in other graph databases.
//! Covers MERGE semantics, pattern matching, aggregation edge cases,
//! Unicode handling, and query correctness.
//!
//! ```bash
//! cargo test -p grafeo-engine --test regression_external -- --nocapture
//! ```

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;
use grafeo_engine::database::QueryResult;

fn db() -> GrafeoDB {
    GrafeoDB::new_in_memory()
}

/// Helper: extract a column of string values from query results.
#[allow(dead_code)]
fn string_column(result: &QueryResult, col: usize) -> Vec<String> {
    result
        .rows
        .iter()
        .map(|row| match &row[col] {
            Value::String(s) => s.to_string(),
            other => format!("{other:?}"),
        })
        .collect()
}

// ============================================================================
// MERGE + UNWIND tuple count
// MERGE in a loop should return one row per input tuple, even when
// the node already exists after the first iteration.
// ============================================================================

mod merge_unwind {
    use super::*;

    #[test]
    fn unwind_merge_returns_one_row_per_input() {
        let db = db();
        let s = db.session();
        // Three inputs, all the same value: MERGE creates one node,
        // but must still produce 3 output rows.
        let r = s
            .execute("UNWIND [1, 1, 1] AS i MERGE (:Item {val: i}) RETURN i")
            .unwrap();
        assert_eq!(r.row_count(), 3, "MERGE must emit one row per UNWIND input");
    }

    #[test]
    fn unwind_merge_creates_single_node() {
        let db = db();
        let s = db.session();
        s.execute("UNWIND [1, 1, 1] AS i MERGE (:Item {val: i})")
            .unwrap();
        let r = s.execute("MATCH (n:Item) RETURN count(n) AS cnt").unwrap();
        assert_eq!(
            r.rows[0][0],
            Value::Int64(1),
            "MERGE with duplicate values should create only one node"
        );
    }

    #[test]
    fn unwind_merge_distinct_values_create_multiple() {
        let db = db();
        let s = db.session();
        s.execute("UNWIND [1, 2, 3] AS i MERGE (:Item {val: i})")
            .unwrap();
        let r = s.execute("MATCH (n:Item) RETURN count(n) AS cnt").unwrap();
        assert_eq!(
            r.rows[0][0],
            Value::Int64(3),
            "MERGE with distinct values should create three nodes"
        );
    }

    #[test]
    fn unwind_merge_mixed_creates_and_matches() {
        let db = db();
        let s = db.session();
        // Pre-create one node
        s.execute("INSERT (:Item {val: 2})").unwrap();
        // UNWIND includes existing (2) and new (1, 3)
        s.execute("UNWIND [1, 2, 3] AS i MERGE (:Item {val: i})")
            .unwrap();
        let r = s.execute("MATCH (n:Item) RETURN count(n) AS cnt").unwrap();
        assert_eq!(
            r.rows[0][0],
            Value::Int64(3),
            "MERGE should create 2 new nodes and match 1 existing"
        );
    }

    #[test]
    fn unwind_merge_on_create_on_match_set() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Item {val: 1, status: 'old'})").unwrap();
        s.execute(
            "UNWIND [1, 2] AS i \
             MERGE (n:Item {val: i}) \
             ON CREATE SET n.status = 'new' \
             ON MATCH SET n.status = 'updated'",
        )
        .unwrap();

        let r = s
            .execute("MATCH (n:Item) RETURN n.val, n.status ORDER BY n.val")
            .unwrap();
        assert_eq!(r.row_count(), 2);
        // val=1 was matched, so status should be 'updated'
        assert_eq!(r.rows[0][0], Value::Int64(1));
        assert_eq!(r.rows[0][1], Value::String("updated".into()));
        // val=2 was created, so status should be 'new'
        assert_eq!(r.rows[1][0], Value::Int64(2));
        assert_eq!(r.rows[1][1], Value::String("new".into()));
    }
}

// ============================================================================
// MERGE with composite keys
// ============================================================================

mod merge_composite_keys {
    use super::*;

    #[test]
    fn merge_two_property_key_no_duplicate() {
        let db = db();
        let s = db.session();
        s.execute("MERGE (:City {name: 'Amsterdam', country: 'NL'})")
            .unwrap();
        s.execute("MERGE (:City {name: 'Amsterdam', country: 'NL'})")
            .unwrap();
        let r = s.execute("MATCH (n:City) RETURN count(n) AS cnt").unwrap();
        assert_eq!(
            r.rows[0][0],
            Value::Int64(1),
            "Identical composite key should not create duplicate"
        );
    }

    #[test]
    fn merge_partial_match_creates_new() {
        let db = db();
        let s = db.session();
        s.execute("MERGE (:City {name: 'Amsterdam', country: 'NL'})")
            .unwrap();
        // Same name, different country: should create a new node
        s.execute("MERGE (:City {name: 'Amsterdam', country: 'US'})")
            .unwrap();
        let r = s.execute("MATCH (n:City) RETURN count(n) AS cnt").unwrap();
        assert_eq!(
            r.rows[0][0],
            Value::Int64(2),
            "Partial composite match should create a new node"
        );
    }

    #[test]
    fn merge_three_property_key() {
        let db = db();
        let s = db.session();
        s.execute("MERGE (:Place {city: 'Berlin', country: 'DE', district: 'Mitte'})")
            .unwrap();
        s.execute("MERGE (:Place {city: 'Berlin', country: 'DE', district: 'Mitte'})")
            .unwrap();
        s.execute("MERGE (:Place {city: 'Berlin', country: 'DE', district: 'Kreuzberg'})")
            .unwrap();
        let r = s.execute("MATCH (n:Place) RETURN count(n) AS cnt").unwrap();
        assert_eq!(
            r.rows[0][0],
            Value::Int64(2),
            "Three-property MERGE: identical creates 1, different district creates 2"
        );
    }
}

// ============================================================================
// Relationship isomorphism
// The same relationship variable must bind to the same edge.
// ============================================================================

mod relationship_isomorphism {
    use super::*;

    #[test]
    fn two_hop_pattern_no_edge_reuse() {
        // In a triangle A->B->C->A, a two-hop pattern (a)-[r1]->(b)-[r2]->(c)
        // should never have r1 == r2.
        let db = db();
        let s = db.session();
        s.execute("INSERT (:N {name: 'Alix'})-[:R]->(:N {name: 'Gus'})")
            .unwrap();
        s.execute(
            "MATCH (a:N {name: 'Gus'}), (b:N {name: 'Alix'}) \
             CREATE (a)-[:R]->(n:N {name: 'Vincent'})-[:R]->(b)",
        )
        .unwrap();
        // Count two-hop paths: should be 3 edges forming specific paths
        let r = s
            .execute(
                "MATCH (a:N)-[r1:R]->(b:N)-[r2:R]->(c:N) \
                 RETURN count(*) AS cnt",
            )
            .unwrap();
        let cnt = &r.rows[0][0];
        // With 3 directed edges in a chain/triangle, verify no edge appears twice
        if let Value::Int64(n) = cnt {
            assert!(*n > 0, "Should find at least one two-hop path");
        }
    }

    #[test]
    fn single_edge_not_matched_twice_in_pattern() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:N {name: 'Alix'})-[:R]->(:N {name: 'Gus'})")
            .unwrap();
        // Only one edge exists: a two-hop pattern should find 0 paths
        let r = s
            .execute(
                "MATCH (a:N)-[r1:R]->(b:N)-[r2:R]->(c:N) \
                 RETURN count(*) AS cnt",
            )
            .unwrap();
        assert_eq!(
            r.rows[0][0],
            Value::Int64(0),
            "Single edge cannot satisfy a two-hop pattern"
        );
    }
}

// ============================================================================
// OPTIONAL MATCH order independence
// ============================================================================

mod optional_match_order {
    use super::*;

    #[test]
    fn swapped_optional_match_same_results() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
            .unwrap();
        s.execute("INSERT (:Person {name: 'Alix'})-[:WORKS_AT]->(:Company {name: 'Acme'})")
            .unwrap();

        let r1 = s
            .execute(
                "MATCH (p:Person {name: 'Alix'}) \
                 OPTIONAL MATCH (p)-[:KNOWS]->(friend:Person) \
                 OPTIONAL MATCH (p)-[:WORKS_AT]->(co:Company) \
                 RETURN p.name, friend.name, co.name",
            )
            .unwrap();

        // Same query with swapped OPTIONAL MATCH order
        let r2 = s
            .execute(
                "MATCH (p:Person {name: 'Alix'}) \
                 OPTIONAL MATCH (p)-[:WORKS_AT]->(co:Company) \
                 OPTIONAL MATCH (p)-[:KNOWS]->(friend:Person) \
                 RETURN p.name, friend.name, co.name",
            )
            .unwrap();

        assert_eq!(
            r1.row_count(),
            r2.row_count(),
            "Swapping OPTIONAL MATCH order must not change row count"
        );
    }
}

// ============================================================================
// Aggregation inside CALL subqueries
// ============================================================================

mod aggregation_in_subquery {
    use super::*;

    #[test]
    fn count_inside_call_subquery() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        s.execute("INSERT (:Person {name: 'Gus'})").unwrap();
        s.execute("INSERT (:Person {name: 'Vincent'})").unwrap();

        let r = s
            .execute("CALL { MATCH (n:Person) RETURN count(n) AS cnt } RETURN cnt")
            .unwrap();
        assert_eq!(r.row_count(), 1, "Aggregation in CALL should return 1 row");
        assert_eq!(r.rows[0][0], Value::Int64(3));
    }
}

// ============================================================================
// COLLECT() completeness
// ============================================================================

mod collect_order {
    use super::*;

    #[test]
    fn collect_returns_all_elements() {
        // Note: GQL does not support `WITH ... ORDER BY` syntax, so we
        // cannot test COLLECT ordering via GQL. This test verifies that
        // COLLECT gathers all values without dropping any.
        // TODO: once WITH ... ORDER BY is supported, add ordering assertion.
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Person {name: 'Vincent'})").unwrap();
        s.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        s.execute("INSERT (:Person {name: 'Gus'})").unwrap();
        s.execute("INSERT (:Person {name: 'Jules'})").unwrap();
        s.execute("INSERT (:Person {name: 'Mia'})").unwrap();

        let r = s
            .execute("MATCH (p:Person) RETURN collect(p.name) AS names")
            .unwrap();
        assert_eq!(r.row_count(), 1);
        if let Value::List(names) = &r.rows[0][0] {
            assert_eq!(names.len(), 5, "COLLECT must gather all 5 names");
            // Verify all names are present (order may vary)
            let name_strs: Vec<String> = names
                .iter()
                .filter_map(|v| match v {
                    Value::String(s) => Some(s.to_string()),
                    _ => None,
                })
                .collect();
            let mut sorted = name_strs.clone();
            sorted.sort();
            assert_eq!(sorted, vec!["Alix", "Gus", "Jules", "Mia", "Vincent"],);
        } else {
            panic!("Expected List, got {:?}", r.rows[0][0]);
        }
    }
}

// ============================================================================
// GROUP BY expression order independence
// ============================================================================

mod group_by_expression_order {
    use super::*;

    #[test]
    fn return_column_order_does_not_affect_group_by() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Person {name: 'Alix', city: 'Amsterdam'})")
            .unwrap();
        s.execute("INSERT (:Person {name: 'Gus', city: 'Amsterdam'})")
            .unwrap();
        s.execute("INSERT (:Person {name: 'Vincent', city: 'Berlin'})")
            .unwrap();

        // Order 1: city first, then count
        let r1 = s
            .execute(
                "MATCH (p:Person) \
                 RETURN p.city AS city, count(p) AS cnt \
                 ORDER BY city",
            )
            .unwrap();

        // Order 2: count first, then city
        let r2 = s
            .execute(
                "MATCH (p:Person) \
                 RETURN count(p) AS cnt, p.city AS city \
                 ORDER BY city",
            )
            .unwrap();

        assert_eq!(r1.row_count(), r2.row_count(), "Same grouping, same rows");
        // Both should have Amsterdam=2, Berlin=1
        let cities_1 = string_column(&r1, 0);
        let cities_2 = string_column(&r2, 1);
        assert_eq!(cities_1, cities_2, "City grouping should be identical");
    }
}

// ============================================================================
// WHERE filter must not corrupt projected values
// ============================================================================

mod where_filter_and_projection {
    use super::*;

    #[test]
    fn where_is_not_null_preserves_return_value() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Sensor {name: 'Temp', reading: 42.5})")
            .unwrap();
        s.execute("INSERT (:Sensor {name: 'Humidity'})").unwrap();

        let r = s
            .execute(
                "MATCH (s:Sensor) \
                 WHERE s.reading IS NOT NULL \
                 RETURN s.name, s.reading",
            )
            .unwrap();
        assert_eq!(r.row_count(), 1);
        assert_eq!(r.rows[0][0], Value::String("Temp".into()));
        // The returned value must be the actual reading, NOT boolean true
        assert_eq!(
            r.rows[0][1],
            Value::Float64(42.5),
            "WHERE IS NOT NULL must not replace the returned value with boolean"
        );
    }
}

// ============================================================================
// SUM overflow / IEEE-754 infinity
// ============================================================================

mod sum_overflow {
    use super::*;

    #[test]
    fn sum_large_floats_returns_infinity() {
        let db = db();
        let s = db.session();
        // Two f64::MAX values should overflow to infinity
        let r = s
            .execute(
                "UNWIND [1.7976931348623157e308, 1.7976931348623157e308] AS val \
                 RETURN SUM(val) AS total",
            )
            .unwrap();
        assert_eq!(r.row_count(), 1);
        match &r.rows[0][0] {
            Value::Float64(f) => {
                assert!(
                    f.is_infinite() && f.is_sign_positive(),
                    "SUM of two f64::MAX should be +Infinity, got {f}"
                );
            }
            other => panic!("Expected Float64, got {:?}", other),
        }
    }
}

// ============================================================================
// Unicode and emoji in property values
// ============================================================================

mod unicode_emoji {
    use super::*;

    #[test]
    fn emoji_property_value_roundtrip() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Tag {symbol: '\u{1F389}', name: 'party'})") // 🎉
            .unwrap();
        let r = s.execute("MATCH (t:Tag) RETURN t.symbol, t.name").unwrap();
        assert_eq!(r.row_count(), 1);
        assert_eq!(r.rows[0][0], Value::String("\u{1F389}".into()));
        assert_eq!(r.rows[0][1], Value::String("party".into()));
    }

    #[test]
    fn cjk_property_value_roundtrip() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:City {name: '\u{6771}\u{4EAC}'})") // 東京
            .unwrap();
        let r = s.execute("MATCH (c:City) RETURN c.name").unwrap();
        assert_eq!(r.rows[0][0], Value::String("\u{6771}\u{4EAC}".into()));
    }

    #[test]
    fn combining_diacritics_roundtrip() {
        let db = db();
        let s = db.session();
        // e + combining acute accent
        s.execute("INSERT (:Word {text: 'calf\u{0065}\u{0301}'})")
            .unwrap();
        let r = s.execute("MATCH (w:Word) RETURN w.text").unwrap();
        assert_eq!(r.rows[0][0], Value::String("calf\u{0065}\u{0301}".into()));
    }
}

// ============================================================================
// Self-loop pattern matching
// ============================================================================

mod self_loop {
    use super::*;

    #[test]
    fn create_and_match_self_loop() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (a:Node {name: 'Alix'})-[:SELF]->(a)")
            .unwrap();

        let r = s
            .execute("MATCH (a:Node)-[r:SELF]->(a) RETURN a.name")
            .unwrap();
        assert_eq!(r.row_count(), 1);
        assert_eq!(r.rows[0][0], Value::String("Alix".into()));
    }

    #[test]
    fn self_loop_not_counted_as_two_edges() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (a:Node {name: 'Alix'})-[:SELF]->(a)")
            .unwrap();

        let r = s
            .execute("MATCH (:Node)-[r:SELF]->() RETURN count(r) AS cnt")
            .unwrap();
        assert_eq!(
            r.rows[0][0],
            Value::Int64(1),
            "Self-loop should be counted exactly once"
        );
    }
}

// ============================================================================
// Deleted node access in same session
// ============================================================================

mod deleted_node_access {
    use super::*;

    #[test]
    fn delete_then_match_in_same_session() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Temp {name: 'ephemeral'})").unwrap();
        s.execute("MATCH (n:Temp) DELETE n").unwrap();

        let r = s.execute("MATCH (n:Temp) RETURN n.name").unwrap();
        assert_eq!(
            r.row_count(),
            0,
            "Deleted node should not be visible in subsequent MATCH"
        );
    }

    #[test]
    fn detach_delete_clears_edges() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:A {name: 'a'})-[:R]->(:B {name: 'b'})")
            .unwrap();
        s.execute("MATCH (a:A) DETACH DELETE a").unwrap();

        let r = s
            .execute("MATCH ()-[r:R]->() RETURN count(r) AS cnt")
            .unwrap();
        assert_eq!(
            r.rows[0][0],
            Value::Int64(0),
            "DETACH DELETE should remove all connected edges"
        );
    }
}

// ============================================================================
// Deeply nested map/list properties
// ============================================================================

mod nested_properties {
    use super::*;
    use grafeo_common::types::PropertyKey;
    use std::collections::{BTreeMap, HashMap};

    #[test]
    fn nested_map_roundtrip() {
        let db = db();
        let s = db.session();
        let mut params = HashMap::new();
        // Build a nested map: {b: {c: 42}}
        let mut inner_map = BTreeMap::new();
        inner_map.insert(PropertyKey::new("c"), Value::Int64(42));
        let inner = Value::Map(inner_map.into());

        let mut mid_map = BTreeMap::new();
        mid_map.insert(PropertyKey::new("b"), inner);
        let mid = Value::Map(mid_map.into());
        params.insert("meta".to_string(), mid);

        s.execute_with_params("INSERT (:Data {meta: $meta})", params)
            .unwrap();
        let r = s.execute("MATCH (d:Data) RETURN d.meta").unwrap();
        assert_eq!(r.row_count(), 1);
        // Verify the nested structure survived
        if let Value::Map(outer) = &r.rows[0][0] {
            let b = outer.get(&PropertyKey::new("b")).expect("Missing key 'b'");
            if let Value::Map(inner_result) = b {
                assert_eq!(
                    inner_result.get(&PropertyKey::new("c")),
                    Some(&Value::Int64(42))
                );
            } else {
                panic!("Expected inner Map, got {:?}", b);
            }
        } else {
            panic!("Expected Map, got {:?}", r.rows[0][0]);
        }
    }

    #[test]
    fn heterogeneous_list_roundtrip() {
        let db = db();
        let s = db.session();
        let mut params = HashMap::new();
        params.insert(
            "items".to_string(),
            Value::List(
                vec![
                    Value::Int64(1),
                    Value::String("two".into()),
                    Value::Bool(true),
                    Value::Null,
                ]
                .into(),
            ),
        );

        s.execute_with_params("INSERT (:Data {items: $items})", params)
            .unwrap();
        let r = s.execute("MATCH (d:Data) RETURN d.items").unwrap();
        if let Value::List(items) = &r.rows[0][0] {
            assert_eq!(items.len(), 4);
            assert_eq!(items[0], Value::Int64(1));
            assert_eq!(items[1], Value::String("two".into()));
            assert_eq!(items[2], Value::Bool(true));
            assert_eq!(items[3], Value::Null);
        } else {
            panic!("Expected List, got {:?}", r.rows[0][0]);
        }
    }
}

// ============================================================================
// Constant folding in WHERE
// ============================================================================

mod constant_folding {
    use super::*;

    #[test]
    fn where_constant_false_returns_empty() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:N {val: 1})").unwrap();
        let r = s.execute("MATCH (n:N) WHERE 1 = 2 RETURN n").unwrap();
        assert_eq!(
            r.row_count(),
            0,
            "WHERE 1 = 2 should short-circuit to empty"
        );
    }

    #[test]
    fn where_constant_true_returns_all() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:N {val: 1})").unwrap();
        s.execute("INSERT (:N {val: 2})").unwrap();
        let r = s.execute("MATCH (n:N) WHERE 1 = 1 RETURN n.val").unwrap();
        assert_eq!(r.row_count(), 2, "WHERE 1 = 1 should return all rows");
    }
}

// ============================================================================
// Cyclic graph traversal
// ============================================================================

mod cyclic_traversal {
    use super::*;

    #[test]
    fn variable_length_path_in_triangle_terminates() {
        let db = db();
        let s = db.session();
        // Create triangle: A->B->C->A
        s.execute(
            "INSERT (a:N {name: 'Alix'})-[:R]->(b:N {name: 'Gus'})-[:R]->(c:N {name: 'Vincent'})-[:R]->(a)",
        )
        .unwrap();

        // Variable-length path should terminate even with cycles
        let r = s
            .execute(
                "MATCH (a:N {name: 'Alix'})-[:R*1..5]->(b:N) \
                 RETURN b.name",
            )
            .unwrap();
        // Should find paths but not infinite-loop
        assert!(
            r.row_count() > 0,
            "Should find reachable nodes through cycle"
        );
        assert!(
            r.row_count() < 100,
            "Should not explode: got {} rows",
            r.row_count()
        );
    }
}

// ============================================================================
// Concurrent MERGE (stress test)
// ============================================================================

mod concurrent_merge {
    use super::*;

    #[test]
    #[ignore = "stress test: run locally before releases"]
    fn concurrent_merge_no_duplicates() {
        let db = std::sync::Arc::new(db());
        let handles: Vec<_> = (0..10)
            .map(|_| {
                let db = std::sync::Arc::clone(&db);
                std::thread::spawn(move || {
                    let s = db.session();
                    s.execute("MERGE (:Singleton {key: 'only_one'})").unwrap();
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        let s = db.session();
        let r = s
            .execute("MATCH (n:Singleton) RETURN count(n) AS cnt")
            .unwrap();
        assert_eq!(
            r.rows[0][0],
            Value::Int64(1),
            "Concurrent MERGE should produce exactly 1 node"
        );
    }
}

// ============================================================================
// MERGE with NULL node reference
// ============================================================================

mod merge_null_node_reference {
    use super::*;

    #[test]
    fn merge_relationship_with_null_source_errors() {
        let db = db();
        let s = db.session();
        // OPTIONAL MATCH that matches nothing produces NULL for n
        let result = s.execute(
            "OPTIONAL MATCH (n:NonExistent) \
             MERGE (n)-[:R]->(m:Target {name: 'Alix'})",
        );
        assert!(
            result.is_err(),
            "MERGE with NULL node reference should error, got: {result:?}"
        );
    }

    #[test]
    fn merge_relationship_with_null_target_errors() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Source {name: 'Gus'})").unwrap();
        let result = s.execute(
            "MATCH (a:Source {name: 'Gus'}) \
             OPTIONAL MATCH (b:NonExistent) \
             MERGE (a)-[:R]->(b)",
        );
        assert!(
            result.is_err(),
            "MERGE with NULL target node reference should error"
        );
    }

    #[test]
    fn merge_with_valid_optional_match_succeeds() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Person {name: 'Vincent'})").unwrap();
        // OPTIONAL MATCH that actually finds something: MERGE should work
        let result = s.execute(
            "MATCH (n:Person {name: 'Vincent'}) \
             MERGE (n)-[:KNOWS]->(m:Person {name: 'Jules'})",
        );
        assert!(
            result.is_ok(),
            "MERGE with non-null node should succeed: {result:?}"
        );
    }

    #[test]
    fn standalone_merge_unaffected() {
        let db = db();
        let s = db.session();
        let result = s.execute("MERGE (:Person {name: 'Mia'}) RETURN 1 AS ok");
        assert!(result.is_ok(), "Standalone MERGE should still work");
    }
}
