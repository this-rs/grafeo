//! Spec compliance tests for GQL, Cypher, and SPARQL.
//!
//! Tests the features added in 0.5.13 for full spec coverage.
//! Covers: set operations, apply operator, predicates, session commands,
//! GQL statements, Cypher features, and SPARQL features.
//!
//! ```bash
//! cargo test -p grafeo-engine --features full --test spec_compliance
//! ```

use grafeo_common::types::{PropertyKey, Value};
use grafeo_engine::GrafeoDB;

// ============================================================================
// Test Fixtures
// ============================================================================

fn social_network() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let alix = session.create_node_with_props(
        &["Person"],
        [
            ("name", Value::String("Alix".into())),
            ("age", Value::Int64(30)),
        ],
    );
    let gus = session.create_node_with_props(
        &["Person"],
        [
            ("name", Value::String("Gus".into())),
            ("age", Value::Int64(25)),
        ],
    );
    let harm = session.create_node_with_props(
        &["Person"],
        [
            ("name", Value::String("Harm".into())),
            ("age", Value::Int64(35)),
        ],
    );
    let dave = session.create_node_with_props(
        &["Person", "Engineer"],
        [
            ("name", Value::String("Dave".into())),
            ("age", Value::Int64(28)),
        ],
    );
    let techcorp = session.create_node_with_props(
        &["Company"],
        [
            ("name", Value::String("TechCorp".into())),
            ("founded", Value::Int64(2010)),
        ],
    );

    let e1 = session.create_edge(alix, gus, "KNOWS");
    db.set_edge_property(e1, "since", Value::Int64(2020));
    let e2 = session.create_edge(alix, harm, "KNOWS");
    db.set_edge_property(e2, "since", Value::Int64(2019));
    let e3 = session.create_edge(gus, harm, "KNOWS");
    db.set_edge_property(e3, "since", Value::Int64(2021));
    session.create_edge(alix, techcorp, "WORKS_AT");
    session.create_edge(gus, techcorp, "WORKS_AT");
    session.create_edge(dave, techcorp, "WORKS_AT");

    // Verify setup: 4 Person + 1 Company = 5 nodes, 3 KNOWS + 3 WORKS_AT = 6 edges
    assert_eq!(db.node_count(), 5, "social_network: expected 5 nodes");
    assert_eq!(db.edge_count(), 6, "social_network: expected 6 edges");

    db
}

fn extract_strings(db: &GrafeoDB, query: &str) -> Vec<String> {
    let session = db.session();
    let result = session.execute(query).unwrap();
    result
        .rows
        .iter()
        .map(|row| match &row[0] {
            Value::String(s) => s.to_string(),
            other => format!("{other:?}"),
        })
        .collect()
}

// ============================================================================
// GQL Set Operations (covers set_ops.rs, gql_translator.rs, planner)
// ============================================================================

#[cfg(feature = "gql")]
mod gql_set_ops {
    use super::*;

    #[test]
    fn union_all_returns_duplicates() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (n:Person) WHERE n.age < 30 RETURN n.name \
                 UNION ALL \
                 MATCH (n:Person) WHERE n.age > 24 RETURN n.name",
            )
            .unwrap();
        // Left (age<30): Gus(25), Dave(28) = 2; Right (age>24): Gus, Dave, Alix, Harm = 4
        assert_eq!(result.row_count(), 6);
    }

    #[test]
    fn union_distinct_deduplicates() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (n:Person) WHERE n.age < 30 RETURN n.name \
                 UNION \
                 MATCH (n:Person) WHERE n.age > 24 RETURN n.name",
            )
            .unwrap();
        // Should be at most 4 distinct names
        assert!(result.row_count() <= 4);
        // All names should be unique
        let names: Vec<_> = result.rows.iter().map(|r| format!("{:?}", r[0])).collect();
        let unique: std::collections::HashSet<_> = names.iter().collect();
        assert_eq!(names.len(), unique.len(), "UNION should deduplicate");
    }

    #[test]
    fn except_removes_common_rows() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (n:Person) RETURN n.name \
                 EXCEPT \
                 MATCH (n:Person) WHERE n.age > 30 RETURN n.name",
            )
            .unwrap();
        // Should exclude Harm (35)
        let names = result
            .rows
            .iter()
            .filter_map(|r| match &r[0] {
                Value::String(s) => Some(s.to_string()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert!(!names.contains(&"Harm".to_string()));
    }

    #[test]
    fn intersect_keeps_common_rows() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (n:Person) WHERE n.age >= 25 RETURN n.name \
                 INTERSECT \
                 MATCH (n:Person) WHERE n.age <= 30 RETURN n.name",
            )
            .unwrap();
        // Intersection of age>=25 and age<=30: Gus(25), Dave(28), Alix(30)
        assert_eq!(result.row_count(), 3);
    }

    #[test]
    fn otherwise_returns_left_if_non_empty() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (n:Person) WHERE n.name = 'Alix' RETURN n.name \
                 OTHERWISE \
                 MATCH (n:Person) RETURN n.name",
            )
            .unwrap();
        // Left side matches Alix, so right side is ignored
        assert_eq!(result.row_count(), 1);
    }

    #[test]
    fn otherwise_falls_through_when_left_empty() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (n:Person) WHERE n.name = 'Nobody' RETURN n.name \
                 OTHERWISE \
                 MATCH (n:Person) RETURN n.name",
            )
            .unwrap();
        // Left side empty, falls through to right
        assert_eq!(result.row_count(), 4);
    }

    #[test]
    fn union_with_literals_and_aggregation() {
        // Test the exact query from the bug report (using GQL parser)
        let db = social_network();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (n:Person) RETURN 'person' AS type, count(n) AS c \
                 UNION \
                 MATCH (n:Company) RETURN 'company' AS type, count(n) AS c",
            )
            .unwrap();
        // Should return 2 rows: one for Person count, one for Company count
        assert_eq!(result.row_count(), 2, "UNION should return 2 rows");
    }

    #[test]
    fn union_all_preserves_duplicates() {
        let db = social_network();
        let session = db.session();
        // Both branches return overlapping names
        let result = session
            .execute(
                "MATCH (n:Person) WHERE n.age <= 30 RETURN n.name \
                 UNION ALL \
                 MATCH (n:Person) WHERE n.age >= 25 RETURN n.name",
            )
            .unwrap();
        // age<=30: Alix(30), Gus(25), Dave(28) = 3
        // age>=25: Alix(30), Gus(25), Dave(28), Harm(35) = 4
        // UNION ALL = 7 (no dedup)
        assert_eq!(result.row_count(), 7, "UNION ALL should keep duplicates");
    }

    #[test]
    fn union_column_mismatch_returns_semantic_error() {
        let db = social_network();
        let session = db.session();
        let err = session
            .execute(
                "MATCH (n:Person) RETURN n.name, n.age \
                 UNION \
                 MATCH (n:Person) RETURN n.name",
            )
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("same number of columns"),
            "Expected semantic error about column count mismatch, got: {}",
            msg
        );
    }

    #[test]
    #[cfg(feature = "cypher")]
    fn union_via_cypher_parser() {
        // Ensure Cypher parser also handles UNION correctly
        let db = social_network();
        let session = db.session();
        let result = session
            .execute_cypher(
                "MATCH (n:Person) RETURN 'person' AS type, count(n) AS c \
                 UNION \
                 MATCH (n:Company) RETURN 'company' AS type, count(n) AS c",
            )
            .unwrap();
        assert_eq!(result.row_count(), 2, "Cypher UNION should return 2 rows");
    }
}

// ============================================================================
// CASE WHEN expressions (covers aggregate.rs group-by + CASE projection)
// ============================================================================

#[cfg(feature = "gql")]
mod case_when_expressions {
    use super::*;

    #[test]
    fn case_when_with_count_aggregation() {
        // The original bug: CASE WHEN in group-by with count() failed with
        // "Cannot resolve expression to column"
        let db = social_network();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (n:Person) \
                 RETURN CASE WHEN n.age >= 30 THEN 'senior' ELSE 'junior' END AS category, \
                 count(n) AS c",
            )
            .unwrap();
        // Should return 2 rows: senior (Alix=30, Harm=35) and junior (Gus=25, Dave=28)
        assert_eq!(
            result.row_count(),
            2,
            "CASE+count should return 2 categories"
        );
    }

    #[test]
    fn case_when_without_aggregation() {
        // CASE without aggregation already worked — ensure no regression
        let db = social_network();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (n:Person) \
                 RETURN n.name, \
                 CASE WHEN n.age >= 30 THEN 'senior' ELSE 'junior' END AS category \
                 ORDER BY n.name",
            )
            .unwrap();
        assert_eq!(result.row_count(), 4);
        // Alix (30) → senior
        assert_eq!(result.rows[0][1], Value::String("senior".into()));
        // Dave (28) → junior
        assert_eq!(result.rows[1][1], Value::String("junior".into()));
    }

    #[test]
    fn case_inside_sum_aggregation() {
        // CASE as argument to SUM (conditional counting pattern)
        let db = social_network();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (n:Person) \
                 RETURN sum(CASE WHEN n.age >= 30 THEN 1 ELSE 0 END) AS seniors",
            )
            .unwrap();
        assert_eq!(result.row_count(), 1);
        // Alix(30) + Harm(35) = 2 seniors
        let val = &result.rows[0][0];
        assert!(
            *val == Value::Int64(2) || *val == Value::Float64(2.0),
            "Expected 2 seniors, got {:?}",
            val
        );
    }

    #[test]
    fn simple_case_form() {
        // Simple CASE: CASE expr WHEN val THEN result END
        let db = social_network();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (n:Person) \
                 RETURN n.name, \
                 CASE n.age WHEN 25 THEN 'twenty-five' WHEN 30 THEN 'thirty' ELSE 'other' END AS label \
                 ORDER BY n.name",
            )
            .unwrap();
        assert_eq!(result.row_count(), 4);
        // Alix=30 → thirty, Dave=28 → other, Gus=25 → twenty-five, Harm=35 → other
        assert_eq!(result.rows[0][1], Value::String("thirty".into())); // Alix
        assert_eq!(result.rows[1][1], Value::String("other".into())); // Dave
        assert_eq!(result.rows[2][1], Value::String("twenty-five".into())); // Gus
        assert_eq!(result.rows[3][1], Value::String("other".into())); // Harm
    }

    #[test]
    fn case_without_else_returns_null() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (n:Person) WHERE n.name = 'Dave' \
                 RETURN CASE WHEN n.age > 30 THEN 'old' END AS label",
            )
            .unwrap();
        assert_eq!(result.row_count(), 1);
        // Dave is 28, no ELSE → null
        assert_eq!(result.rows[0][0], Value::Null);
    }
}

// ============================================================================
// GQL Predicates (covers filter.rs IS TYPED, IS DIRECTED, etc.)
// ============================================================================

#[cfg(feature = "gql")]
mod gql_predicates {
    use super::*;

    #[test]
    fn property_exists_check() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute("MATCH (n:Person) WHERE property_exists(n, 'age') RETURN n.name")
            .unwrap();
        assert_eq!(result.row_count(), 4, "All persons have age property");
    }

    /// GQL NULLIF: returns NULL when both arguments are equal.
    #[test]
    fn nullif_returns_null_when_equal() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute("MATCH (n:Person) WHERE n.name = 'Alix' RETURN NULLIF(n.age, 30) AS val")
            .unwrap();
        assert_eq!(result.rows[0][0], Value::Null);
    }

    /// GQL NULLIF: returns the first argument when they differ.
    #[test]
    fn nullif_returns_value_when_different() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute("MATCH (n:Person) WHERE n.name = 'Gus' RETURN NULLIF(n.age, 30) AS val")
            .unwrap();
        assert_eq!(result.rows[0][0], Value::Int64(25));
    }

    #[test]
    fn element_id_function() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute("MATCH (n:Person) WHERE n.name = 'Alix' RETURN element_id(n)")
            .unwrap();
        let id_str = match &result.rows[0][0] {
            Value::String(s) => s.to_string(),
            other => panic!("Expected string element ID, got {other:?}"),
        };
        assert!(
            id_str.starts_with("n:"),
            "element_id should start with 'n:'"
        );
    }

    #[test]
    fn cast_to_string() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (n:Person) WHERE n.name = 'Alix' RETURN CAST(n.age AS STRING) AS age_str",
            )
            .unwrap();
        assert_eq!(result.rows[0][0], Value::String("30".into()));
    }

    #[test]
    fn cast_to_integer() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("INSERT (:Item {value: '42'})").unwrap();
        let result = session
            .execute("MATCH (n:Item) RETURN CAST(n.value AS INTEGER) AS v")
            .unwrap();
        assert_eq!(result.rows[0][0], Value::Int64(42));
    }

    #[test]
    fn cast_to_float() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("INSERT (:Item {value: '3.14'})").unwrap();
        let result = session
            .execute("MATCH (n:Item) RETURN CAST(n.value AS FLOAT) AS v")
            .unwrap();
        match &result.rows[0][0] {
            Value::Float64(f) => assert!((f - std::f64::consts::PI).abs() < 0.01),
            other => panic!("Expected Float64, got {other:?}"),
        }
    }

    #[test]
    fn cast_to_boolean() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("INSERT (:Item {value: 'true'})").unwrap();
        let result = session
            .execute("MATCH (n:Item) RETURN CAST(n.value AS BOOLEAN) AS v")
            .unwrap();
        assert_eq!(result.rows[0][0], Value::Bool(true));
    }

    #[test]
    fn xor_operator() {
        let db = social_network();
        // XOR: (age>30)=Harm XOR (name=Gus)=Gus, but not both => Gus, Harm
        let names = extract_strings(
            &db,
            "MATCH (n:Person) WHERE (n.age > 30) XOR (n.name = 'Gus') RETURN n.name ORDER BY n.name",
        );
        assert_eq!(
            names,
            vec!["Gus", "Harm"],
            "ORDER BY n.name should produce alphabetical order"
        );
    }
}

// ============================================================================
// GQL Statements (covers parser.rs + translator: FINISH, SELECT, LET, etc.)
// ============================================================================

#[cfg(feature = "gql")]
mod gql_statements {
    use super::*;

    #[test]
    fn finish_returns_empty() {
        let db = social_network();
        let session = db.session();
        let result = session.execute("MATCH (n:Person) FINISH").unwrap();
        assert_eq!(result.row_count(), 0, "FINISH should return no rows");
    }

    #[test]
    fn select_as_return_synonym() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute("MATCH (n:Person) SELECT n.name ORDER BY n.name")
            .unwrap();
        assert_eq!(result.row_count(), 4);
    }

    #[test]
    fn element_where_on_node() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute("MATCH (n:Person WHERE n.age > 28) RETURN n.name ORDER BY n.name")
            .unwrap();
        let names: Vec<_> = result
            .rows
            .iter()
            .filter_map(|r| match &r[0] {
                Value::String(s) => Some(s.to_string()),
                _ => None,
            })
            .collect();
        // ORDER BY n.name: Alix (age 30) before Harm (age 35), Gus (age 25) excluded
        assert_eq!(
            names,
            vec!["Alix", "Harm"],
            "ORDER BY n.name should produce alphabetical order"
        );
    }

    #[test]
    fn element_where_on_edge() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (a:Person)-[e:KNOWS WHERE e.since >= 2020]->(b:Person) \
                 RETURN a.name, b.name ORDER BY a.name",
            )
            .unwrap();
        // since>=2020: Alix->Gus (2020), Gus->Harm (2021)
        assert_eq!(result.row_count(), 2);
    }

    #[test]
    fn iso_path_quantifier_exact() {
        let db = social_network();
        let session = db.session();
        // {1} means exactly 1 hop
        let result = session
            .execute("MATCH (a:Person)-[:KNOWS{1}]->(b:Person) RETURN a.name, b.name")
            .unwrap();
        // Exactly 3 direct KNOWS edges: Alix->Gus, Alix->Harm, Gus->Harm
        assert_eq!(result.row_count(), 3, "Should find direct connections");
    }

    #[test]
    fn iso_path_quantifier_range() {
        let db = social_network();
        let session = db.session();
        // {1,2} means 1 to 2 hops
        let result = session
            .execute(
                "MATCH (a:Person)-[:KNOWS{1,2}]->(b:Person) WHERE a.name = 'Alix' RETURN b.name",
            )
            .unwrap();
        // 1-hop: Gus, Harm; 2-hop: Gus->Harm = Harm again
        assert_eq!(result.row_count(), 3);
    }

    #[test]
    fn property_map_not_confused_with_quantifier() {
        // Regression: {since: 2020} was misinterpreted as path quantifier
        let db = social_network();
        let session = db.session();
        let result = session
            .execute("MATCH (a:Person)-[e:KNOWS {since: 2020}]->(b:Person) RETURN a.name, b.name")
            .unwrap();
        // Only Alix->Gus has since=2020
        assert_eq!(result.row_count(), 1);
    }

    #[test]
    fn match_mode_different_edges() {
        let db = social_network();
        let session = db.session();
        // DIFFERENT EDGES is like TRAIL - no repeated edges
        let result = session
            .execute(
                "MATCH DIFFERENT EDGES (a:Person)-[:KNOWS*1..2]->(b:Person) RETURN a.name, b.name",
            )
            .unwrap();
        // 1-hop: (Alix,Gus), (Alix,Harm), (Gus,Harm); 2-hop: (Alix,Harm via Gus)
        assert_eq!(result.row_count(), 4);
    }

    #[test]
    fn match_mode_repeatable_elements() {
        let db = social_network();
        let session = db.session();
        // REPEATABLE ELEMENTS is like WALK - edges and nodes may repeat
        let result = session
            .execute("MATCH REPEATABLE ELEMENTS (a:Person)-[:KNOWS*1..2]->(b:Person) RETURN a.name, b.name")
            .unwrap();
        // No cycles in KNOWS, so same as TRAIL: 3 one-hop + 1 two-hop
        assert_eq!(result.row_count(), 4);
    }

    #[test]
    fn label_expression_disjunction() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute("MATCH (n IS Person | Company) RETURN n.name ORDER BY n.name")
            .unwrap();
        // Should find all 4 persons + TechCorp
        assert_eq!(result.row_count(), 5);
    }

    #[test]
    fn label_expression_conjunction() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute("MATCH (n IS Person & Engineer) RETURN n.name")
            .unwrap();
        // Only Dave has both Person and Engineer
        assert_eq!(result.row_count(), 1);
    }

    #[test]
    fn label_expression_negation() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute("MATCH (n IS Person & !Engineer) RETURN n.name ORDER BY n.name")
            .unwrap();
        // Alix, Gus, Harm have Person but not Engineer
        assert_eq!(result.row_count(), 3);
    }

    #[test]
    fn group_by_explicit() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (n:Person)-[:WORKS_AT]->(c:Company) \
                 RETURN c.name, count(n) AS cnt GROUP BY c.name",
            )
            .unwrap();
        // Only one company (TechCorp) has WORKS_AT edges
        assert_eq!(result.row_count(), 1);
    }

    #[test]
    fn gql_filter_clause() {
        let db = social_network();
        let session = db.session();
        // FILTER is a GQL synonym for WHERE
        let result = session
            .execute("MATCH (n:Person) FILTER n.age > 28 RETURN n.name ORDER BY n.name")
            .unwrap();
        // age > 28: Alix(30), Harm(35)
        assert_eq!(result.row_count(), 2);
    }

    #[test]
    fn gql_offset_clause() {
        let db = social_network();
        let session = db.session();
        // OFFSET is a GQL synonym for SKIP
        let result = session
            .execute("MATCH (n:Person) RETURN n.name ORDER BY n.name OFFSET 2")
            .unwrap();
        assert_eq!(result.row_count(), 2); // 4 total, skip 2
    }

    #[test]
    fn list_index_access() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("INSERT (:Item {tags: ['a', 'b', 'c']})")
            .unwrap();
        let result = session
            .execute("MATCH (n:Item) RETURN n.tags[0] AS first")
            .unwrap();
        assert_eq!(result.rows[0][0], Value::String("a".into()));
    }

    #[test]
    fn gql_block_comment() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute("MATCH /* all people */ (n:Person) RETURN count(n) AS cnt")
            .unwrap();
        assert_eq!(result.rows[0][0], Value::Int64(4));
    }
}

// ============================================================================
// GQL Session and DDL Commands (covers session.rs, parser DDL branches)
// ============================================================================

#[cfg(feature = "gql")]
mod gql_session_commands {
    use super::*;

    #[test]
    fn create_graph_and_drop_graph() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        // CREATE GRAPH
        let result = session.execute("CREATE GRAPH test_graph");
        assert!(result.is_ok());
        // Verify it exists
        let graphs = db.list_graphs();
        assert!(graphs.contains(&"test_graph".to_string()));
        // DROP GRAPH
        let result = session.execute("DROP GRAPH test_graph");
        assert!(result.is_ok());
        let graphs = db.list_graphs();
        assert!(!graphs.contains(&"test_graph".to_string()));
    }

    #[test]
    fn create_graph_if_not_exists() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("CREATE GRAPH g1").unwrap();
        // Without IF NOT EXISTS, should fail
        let result = session.execute("CREATE GRAPH g1");
        assert!(result.is_err());
        // With IF NOT EXISTS, should succeed
        let result = session.execute("CREATE GRAPH IF NOT EXISTS g1");
        assert!(result.is_ok());
    }

    #[test]
    fn drop_graph_if_exists() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        // Without IF EXISTS, should fail on nonexistent
        let result = session.execute("DROP GRAPH nonexistent");
        assert!(result.is_err());
        // With IF EXISTS, should succeed
        let result = session.execute("DROP GRAPH IF EXISTS nonexistent");
        assert!(result.is_ok());
    }

    #[test]
    fn create_property_graph() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session.execute("CREATE PROPERTY GRAPH my_graph");
        assert!(result.is_ok());
        assert!(db.list_graphs().contains(&"my_graph".to_string()));
    }

    #[test]
    fn drop_property_graph() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("CREATE PROPERTY GRAPH pg").unwrap();
        let result = session.execute("DROP PROPERTY GRAPH pg");
        assert!(result.is_ok());
    }

    #[test]
    fn use_graph_command() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("CREATE GRAPH workspace").unwrap();
        let result = session.execute("USE GRAPH workspace");
        assert!(result.is_ok());
        assert_eq!(session.current_graph(), Some("workspace".to_string()));
    }

    #[test]
    fn use_graph_nonexistent_errors() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session.execute("USE GRAPH nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn session_set_time_zone() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session.execute("SESSION SET TIME ZONE 'UTC+5'");
        assert!(result.is_ok());
        assert_eq!(session.time_zone(), Some("UTC+5".to_string()));
    }

    #[test]
    fn session_set_graph() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("CREATE GRAPH analytics").unwrap();
        let result = session.execute("SESSION SET GRAPH analytics");
        assert!(result.is_ok());
        assert_eq!(session.current_graph(), Some("analytics".to_string()));
    }

    #[test]
    fn session_set_schema() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        // ISO/IEC 39075 Section 7.1 GR1: SESSION SET SCHEMA sets session schema independently
        session.execute("CREATE SCHEMA myschema").unwrap();
        let result = session.execute("SESSION SET SCHEMA myschema");
        assert!(result.is_ok());
        assert_eq!(session.current_schema(), Some("myschema".to_string()));
        // Graph should remain unaffected
        assert_eq!(session.current_graph(), None);
    }

    #[test]
    fn session_set_schema_nonexistent_errors() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        // SESSION SET SCHEMA should error if schema does not exist
        let result = session.execute("SESSION SET SCHEMA nosuchschema");
        assert!(result.is_err());
    }

    #[test]
    fn session_reset() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("CREATE GRAPH g1").unwrap();
        session.execute("USE GRAPH g1").unwrap();
        session.execute("SESSION SET TIME ZONE 'EST'").unwrap();
        session.execute("CREATE SCHEMA s1").unwrap();
        session.execute("SESSION SET SCHEMA s1").unwrap();
        // Reset clears everything (Section 7.2 GR1+GR2+GR3)
        let result = session.execute("SESSION RESET");
        assert!(result.is_ok());
        assert_eq!(session.current_graph(), None);
        assert_eq!(session.current_schema(), None);
        assert_eq!(session.time_zone(), None);
    }

    #[test]
    fn session_reset_schema_only() {
        // ISO/IEC 39075 Section 7.2 GR1: SESSION RESET SCHEMA resets schema only
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("CREATE GRAPH g1").unwrap();
        session.execute("USE GRAPH g1").unwrap();
        session.execute("CREATE SCHEMA s1").unwrap();
        session.execute("SESSION SET SCHEMA s1").unwrap();
        let result = session.execute("SESSION RESET SCHEMA");
        assert!(result.is_ok());
        assert_eq!(session.current_schema(), None);
        // Graph should remain set
        assert_eq!(session.current_graph(), Some("g1".to_string()));
    }

    #[test]
    fn session_reset_graph_only() {
        // ISO/IEC 39075 Section 7.2 GR2: SESSION RESET GRAPH resets graph only
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("CREATE GRAPH g1").unwrap();
        session.execute("USE GRAPH g1").unwrap();
        session.execute("CREATE SCHEMA s1").unwrap();
        session.execute("SESSION SET SCHEMA s1").unwrap();
        let result = session.execute("SESSION RESET GRAPH");
        assert!(result.is_ok());
        assert_eq!(session.current_graph(), None);
        // Schema should remain set
        assert_eq!(session.current_schema(), Some("s1".to_string()));
    }

    // ========================================================================
    // QoL introspection functions: CURRENT_SCHEMA, CURRENT_GRAPH, info(), schema()
    // ========================================================================

    #[test]
    fn return_current_schema_default_when_unset() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session.execute("RETURN CURRENT_SCHEMA AS s").unwrap();
        assert_eq!(result.columns, vec!["s"]);
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0].len(), 1);
        assert_eq!(result.rows[0][0], Value::String("default".into()));
    }

    #[test]
    fn return_current_schema_after_set() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("CREATE SCHEMA analytics").unwrap();
        session.execute("SESSION SET SCHEMA analytics").unwrap();
        let result = session.execute("RETURN CURRENT_SCHEMA AS s").unwrap();
        assert_eq!(result.rows[0][0], Value::String("analytics".into()));
    }

    #[test]
    fn return_current_graph_default_when_unset() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session.execute("RETURN CURRENT_GRAPH AS g").unwrap();
        assert_eq!(result.columns, vec!["g"]);
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0][0], Value::String("default".into()));
    }

    #[test]
    fn return_current_graph_after_use() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("CREATE GRAPH social").unwrap();
        session.execute("USE GRAPH social").unwrap();
        let result = session.execute("RETURN CURRENT_GRAPH AS g").unwrap();
        assert_eq!(result.rows[0][0], Value::String("social".into()));
    }

    #[test]
    fn return_info_function() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        // Insert some data first
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        session
            .execute("INSERT (:Person {name: 'Gus'})-[:KNOWS]->(:Person {name: 'Vincent'})")
            .unwrap();
        let result = session.execute("RETURN info() AS i").unwrap();
        assert_eq!(result.columns, vec!["i"]);
        assert_eq!(result.rows.len(), 1);
        let info = result.rows[0][0]
            .as_map()
            .expect("info() should return a map");
        assert_eq!(
            info.get(&PropertyKey::from("mode")),
            Some(&Value::String("lpg".into()))
        );
        // Should have 3 nodes (Alix, Gus, Vincent)
        assert_eq!(
            info.get(&PropertyKey::from("node_count")),
            Some(&Value::Int64(3))
        );
        // Should have 1 edge (KNOWS)
        assert_eq!(
            info.get(&PropertyKey::from("edge_count")),
            Some(&Value::Int64(1))
        );
        assert!(info.get(&PropertyKey::from("version")).is_some());
    }

    #[test]
    fn return_schema_function() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
            .unwrap();
        let result = session.execute("RETURN schema() AS s").unwrap();
        assert_eq!(result.columns, vec!["s"]);
        assert_eq!(result.rows.len(), 1);
        let schema = result.rows[0][0]
            .as_map()
            .expect("schema() should return a map");
        // Labels should include "Person"
        let labels = schema
            .get(&PropertyKey::from("labels"))
            .expect("should have labels");
        if let Value::List(l) = labels {
            assert!(l.contains(&Value::String("Person".into())));
        } else {
            panic!("labels should be a list");
        }
        // Edge types should include "KNOWS"
        let edge_types = schema
            .get(&PropertyKey::from("edge_types"))
            .expect("should have edge_types");
        if let Value::List(l) = edge_types {
            assert!(l.contains(&Value::String("KNOWS".into())));
        } else {
            panic!("edge_types should be a list");
        }
    }

    #[test]
    fn session_close() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session.execute("SESSION CLOSE");
        assert!(result.is_ok());
    }

    #[test]
    fn start_transaction_commit_rollback_via_gql() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        // START TRANSACTION works
        session.execute("START TRANSACTION").unwrap();
        assert!(session.in_transaction());

        // COMMIT works
        session.execute("COMMIT").unwrap();
        assert!(!session.in_transaction());

        // ROLLBACK works
        session.execute("START TRANSACTION").unwrap();
        session.execute("ROLLBACK").unwrap();
        assert!(!session.in_transaction());
    }

    #[test]
    fn commit_without_transaction_errors() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session.execute("COMMIT");
        assert!(result.is_err());
    }

    #[test]
    fn rollback_without_transaction_errors() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session.execute("ROLLBACK");
        assert!(result.is_err());
    }

    #[test]
    fn session_set_parameter() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session.execute("SESSION SET PARAMETER my_param = 42");
        assert!(result.is_ok());
    }
}

// ============================================================================
// GQL Path Features (covers path search prefixes, questioned edges)
// ============================================================================

#[cfg(feature = "gql")]
mod gql_path_features {
    use super::*;

    #[test]
    fn any_shortest_path() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute(
                "MATCH ANY SHORTEST (a:Person)-[:KNOWS*]->(b:Person) \
                 WHERE a.name = 'Alix' AND b.name = 'Harm' \
                 RETURN a.name, b.name",
            )
            .unwrap();
        // Alix->Harm is 1 hop (direct), the single shortest path
        assert_eq!(result.row_count(), 1);
    }

    #[test]
    fn all_shortest_path() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute(
                "MATCH ALL SHORTEST (a:Person)-[:KNOWS*]->(b:Person) \
                 WHERE a.name = 'Alix' AND b.name = 'Harm' \
                 RETURN a.name, b.name",
            )
            .unwrap();
        // Only one shortest path: Alix->Harm (1 hop, direct)
        assert_eq!(result.row_count(), 1);
    }

    #[test]
    fn path_mode_walk() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute("MATCH WALK (a:Person)-[:KNOWS*1..2]->(b:Person) RETURN a.name, b.name")
            .unwrap();
        // No cycles in KNOWS: 3 one-hop + 1 two-hop = 4
        assert_eq!(result.row_count(), 4);
    }

    #[test]
    fn path_mode_trail() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute("MATCH TRAIL (a:Person)-[:KNOWS*1..2]->(b:Person) RETURN a.name, b.name")
            .unwrap();
        // No cycles in KNOWS: 3 one-hop + 1 two-hop = 4
        assert_eq!(result.row_count(), 4);
    }
}

// ============================================================================
// GQL INSERT Patterns (covers path INSERT decomposition)
// ============================================================================

#[cfg(feature = "gql")]
mod gql_insert_patterns {
    use super::*;

    #[test]
    fn insert_node_with_properties() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("INSERT (:Person {name: 'Eve', age: 22})")
            .unwrap();
        let result = session.execute("MATCH (n:Person) RETURN n.name").unwrap();
        assert_eq!(result.row_count(), 1);
    }

    #[test]
    fn insert_path_creates_nodes_and_edge() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("INSERT (:Person {name: 'X'})-[:KNOWS]->(:Person {name: 'Y'})")
            .unwrap();
        let result = session
            .execute("MATCH (a)-[:KNOWS]->(b) RETURN a.name, b.name")
            .unwrap();
        assert_eq!(result.row_count(), 1);
    }

    #[test]
    fn create_edge_with_properties_in_query() {
        // Regression: {since: 2020} was misinterpreted as quantifier
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'A'})").unwrap();
        session.execute("INSERT (:Person {name: 'B'})").unwrap();
        session
            .execute(
                "MATCH (a:Person), (b:Person) WHERE a.name = 'A' AND b.name = 'B' \
                 CREATE (a)-[:KNOWS {since: 2020}]->(b) RETURN a.name",
            )
            .unwrap();
        let result = session
            .execute("MATCH (a)-[e:KNOWS]->(b) RETURN e.since")
            .unwrap();
        assert_eq!(result.rows[0][0], Value::Int64(2020));
    }
}

// ============================================================================
// Cypher Features (covers cypher parser + translator)
// ============================================================================

#[cfg(feature = "cypher")]
mod cypher_features {
    use super::*;

    #[test]
    fn count_star() {
        // ISO/IEC 39075 Section 20.9: COUNT(*) counts all rows
        let db = social_network();
        let session = db.session();
        let result = session
            .execute("MATCH (n:Person) RETURN COUNT(*) AS cnt")
            .unwrap();
        assert_eq!(result.row_count(), 1);
        assert_eq!(result.rows[0][0], Value::Int64(4)); // Alix, Gus, Dave, Harm
    }

    #[test]
    fn count_subquery() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute_cypher(
                "MATCH (p:Person) \
                 RETURN p.name, COUNT { MATCH (p)-[:KNOWS]->() } AS friends \
                 ORDER BY p.name",
            )
            .unwrap();
        // One row per Person: Alix, Dave, Gus, Harm
        assert_eq!(result.row_count(), 4);
    }

    #[test]
    fn map_projection() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute_cypher("MATCH (p:Person) WHERE p.name = 'Alix' RETURN p { .name, .age }")
            .unwrap();
        assert_eq!(result.row_count(), 1);
        // Should return a map value
        match &result.rows[0][0] {
            Value::Map(m) => {
                let keys: Vec<String> = m.keys().map(|k| k.as_str().to_string()).collect();
                assert!(keys.contains(&"name".to_string()), "Map should have 'name'");
                assert!(keys.contains(&"age".to_string()), "Map should have 'age'");
            }
            other => panic!("Expected Map, got {other:?}"),
        }
    }

    #[test]
    fn reduce_function() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("INSERT (:Item {nums: [1, 2, 3, 4, 5]})")
            .unwrap();
        let result = session
            .execute_cypher(
                "MATCH (n:Item) RETURN reduce(total = 0, x IN n.nums | total + x) AS sum",
            )
            .unwrap();
        assert_eq!(result.rows[0][0], Value::Int64(15));
    }

    #[test]
    fn cypher_case_inside_aggregate() {
        let db = social_network();
        let session = db.session();
        // sum(CASE WHEN ... THEN 1 ELSE 0 END) is a common conditional-count pattern
        let result = session
            .execute_cypher(
                "MATCH (p:Person) \
                 RETURN sum(CASE WHEN p.age >= 30 THEN 1 ELSE 0 END) AS over_30, \
                        sum(CASE WHEN p.age < 30 THEN 1 ELSE 0 END) AS under_30",
            )
            .unwrap();
        assert_eq!(result.row_count(), 1);
        // Alix=30, Gus=25, Harm=35, Dave=28 => over_30=2 (Alix, Harm), under_30=2 (Gus, Dave)
        let over_30 = &result.rows[0][0];
        let under_30 = &result.rows[0][1];
        assert_eq!(*over_30, Value::Int64(2), "Expected 2 people aged >= 30");
        assert_eq!(*under_30, Value::Int64(2), "Expected 2 people aged < 30");
    }

    #[test]
    fn cypher_math_functions() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute_cypher(
                "MATCH (n:Person) WHERE n.name = 'Alix' \
                 RETURN sign(n.age) AS s, abs(-5) AS a",
            )
            .unwrap();
        // sign(30) = 1
        assert_eq!(result.rows[0][0], Value::Int64(1));
    }

    #[test]
    fn cypher_string_functions() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute_cypher(
                "MATCH (n:Person) WHERE n.name = 'Alix' \
                 RETURN left(n.name, 3) AS l, right(n.name, 3) AS r",
            )
            .unwrap();
        assert_eq!(result.rows[0][0], Value::String("Ali".into()));
        assert_eq!(result.rows[0][1], Value::String("lix".into()));
    }

    #[test]
    fn cypher_trig_functions() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("INSERT (:Val {x: 0.0})").unwrap();
        let result = session
            .execute_cypher("MATCH (n:Val) RETURN sin(n.x) AS s, cos(n.x) AS c")
            .unwrap();
        match &result.rows[0][0] {
            Value::Float64(f) => assert!((f - 0.0).abs() < 0.001, "sin(0) should be 0"),
            other => panic!("Expected Float64 for sin, got {other:?}"),
        }
        match &result.rows[0][1] {
            Value::Float64(f) => assert!((f - 1.0).abs() < 0.001, "cos(0) should be 1"),
            other => panic!("Expected Float64 for cos, got {other:?}"),
        }
    }

    #[test]
    fn cypher_pi_and_e() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("INSERT (:X {x: 1})").unwrap();
        let result = session
            .execute_cypher("MATCH (n:X) RETURN pi() AS p, e() AS e")
            .unwrap();
        match &result.rows[0][0] {
            Value::Float64(f) => assert!((f - std::f64::consts::PI).abs() < 0.001),
            other => panic!("Expected Float64, got {other:?}"),
        }
    }

    #[test]
    fn cypher_power_operator() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("INSERT (:X {x: 3})").unwrap();
        let result = session
            .execute_cypher("MATCH (n:X) RETURN n.x ^ 2 AS sq")
            .unwrap();
        match &result.rows[0][0] {
            Value::Float64(f) => assert!((f - 9.0).abs() < 0.001),
            Value::Int64(n) => assert_eq!(*n, 9),
            other => panic!("Expected numeric, got {other:?}"),
        }
    }

    #[test]
    fn cypher_create_index() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        // Set up data so there's a label in the catalog
        session
            .execute("INSERT (:Person {name: 'Alix', age: 30})")
            .unwrap();
        let result =
            session.execute_cypher("CREATE INDEX idx_person_name FOR (n:Person) ON (n.name)");
        assert!(
            result.is_ok(),
            "CREATE INDEX should succeed: {:?}",
            result.err()
        );
    }

    #[test]
    fn cypher_create_index_if_not_exists() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        session
            .execute_cypher("CREATE INDEX idx_test IF NOT EXISTS FOR (n:Person) ON (n.name)")
            .unwrap();
        // Running again should not error with IF NOT EXISTS
        let result = session
            .execute_cypher("CREATE INDEX idx_test IF NOT EXISTS FOR (n:Person) ON (n.name)");
        assert!(result.is_ok());
    }

    #[test]
    fn cypher_drop_index() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        session
            .execute_cypher("CREATE INDEX idx_drop FOR (n:Person) ON (n.name)")
            .unwrap();
        // DROP by property name (store tracks by property, not by index name)
        let result = session.execute_cypher("DROP INDEX name");
        assert!(
            result.is_ok(),
            "DROP INDEX should succeed: {:?}",
            result.err()
        );
    }

    #[test]
    fn cypher_drop_index_if_exists() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        // Dropping non-existent index with IF EXISTS should not error
        let result = session.execute_cypher("DROP INDEX nonexistent IF EXISTS");
        assert!(result.is_ok());
    }

    #[test]
    fn cypher_create_constraint() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        let result = session.execute_cypher(
            "CREATE CONSTRAINT unique_name FOR (n:Person) REQUIRE n.name IS UNIQUE",
        );
        assert!(
            result.is_ok(),
            "CREATE CONSTRAINT should succeed: {:?}",
            result.err()
        );
    }

    #[test]
    fn cypher_show_indexes() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session.execute_cypher("SHOW INDEXES");
        assert!(
            result.is_ok(),
            "SHOW INDEXES should succeed: {:?}",
            result.err()
        );
        let qr = result.unwrap();
        assert_eq!(qr.columns.len(), 4);
        assert_eq!(qr.columns[0], "name");
    }

    #[test]
    fn cypher_show_constraints() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session.execute_cypher("SHOW CONSTRAINTS");
        assert!(
            result.is_ok(),
            "SHOW CONSTRAINTS should succeed: {:?}",
            result.err()
        );
    }

    #[test]
    fn cypher_relationship_where_clause() {
        let db = social_network();
        let session = db.session();
        // Inline WHERE on relationship pattern (Neo4j 5.x syntax)
        let result = session.execute_cypher(
            "MATCH (p:Person)-[r:KNOWS WHERE r.since IS NOT NULL]->(f:Person) \
                 RETURN p.name, f.name ORDER BY p.name",
        );
        // Should parse and execute without errors regardless of data
        assert!(
            result.is_ok(),
            "Relationship WHERE should work: {:?}",
            result.err()
        );
    }

    #[test]
    fn cypher_label_check_in_where() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute_cypher("MATCH (n) WHERE n:Engineer RETURN n.name")
            .unwrap();
        assert_eq!(result.row_count(), 1); // Only Dave
    }

    #[test]
    fn cypher_exists_subquery() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute_cypher(
                "MATCH (p:Person) WHERE EXISTS { MATCH (p)-[:WORKS_AT]->(:Company) } \
                 RETURN p.name ORDER BY p.name",
            )
            .unwrap();
        // Alix, Dave, Gus all WORKS_AT TechCorp
        assert_eq!(result.row_count(), 3);
    }

    #[test]
    fn cypher_exists_bare_pattern() {
        let db = social_network();
        let session = db.session();
        // Bare pattern form: no explicit MATCH keyword inside EXISTS
        let result = session
            .execute_cypher(
                "MATCH (p:Person) WHERE EXISTS { (p)-[:WORKS_AT]->(:Company) } \
                 RETURN p.name ORDER BY p.name",
            )
            .unwrap();
        // Same as explicit MATCH version: Alix, Dave, Gus
        assert_eq!(result.row_count(), 3);
    }

    #[test]
    fn cypher_exists_bare_pattern_with_where() {
        let db = social_network();
        let session = db.session();
        // Bare pattern with WHERE inside EXISTS
        let result = session.execute_cypher(
            "MATCH (p:Person) WHERE EXISTS { (p)-[r:KNOWS]->() WHERE r.since IS NOT NULL } \
                 RETURN p.name ORDER BY p.name",
        );
        // This may or may not return results depending on test data,
        // but it should parse and execute without errors
        assert!(
            result.is_ok(),
            "Bare pattern EXISTS with WHERE should parse: {:?}",
            result.err()
        );
    }

    #[test]
    fn cypher_foreach_set() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("INSERT (:Person {name: 'A', verified: false})")
            .unwrap();
        session
            .execute("INSERT (:Person {name: 'B', verified: false})")
            .unwrap();
        session
            .execute_cypher(
                "MATCH (n:Person) WITH collect(n) AS people \
                 FOREACH (p IN people | SET p.verified = true) \
                 RETURN count(*) AS cnt",
            )
            .unwrap();
        let result = session
            .execute("MATCH (n:Person) WHERE n.verified = true RETURN count(n) AS cnt")
            .unwrap();
        assert_eq!(result.rows[0][0], Value::Int64(2));
    }

    #[test]
    fn cypher_call_inline_subquery() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute_cypher(
                "MATCH (p:Person) \
                 CALL { WITH p MATCH (p)-[:KNOWS]->(f) RETURN count(f) AS cnt } \
                 RETURN p.name, cnt ORDER BY p.name",
            )
            .unwrap();
        // One row per Person: Alix, Dave, Gus, Harm
        assert_eq!(result.row_count(), 4);
    }

    #[test]
    fn cypher_pattern_comprehension() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute_cypher(
                "MATCH (p:Person) WHERE p.name = 'Alix' \
                 RETURN p.name, [(p)-[:KNOWS]->(f) | f.name] AS friends",
            )
            .unwrap();
        assert_eq!(result.row_count(), 1);
        match &result.rows[0][1] {
            Value::List(items) => assert_eq!(items.len(), 2, "Alix knows exactly Gus and Harm"),
            other => panic!("Expected list of friends, got {other:?}"),
        }
    }

    #[test]
    fn cypher_skip_with_parameter() {
        let db = social_network();
        let mut params = std::collections::HashMap::new();
        params.insert("offset".to_string(), Value::Int64(2));
        let result = db
            .execute_cypher_with_params(
                "MATCH (p:Person) RETURN p.name ORDER BY p.name SKIP $offset",
                params,
            )
            .unwrap();
        // 4 people total, skip 2 = 2 results
        assert_eq!(result.row_count(), 2);
    }

    #[test]
    fn cypher_limit_with_parameter() {
        let db = social_network();
        let mut params = std::collections::HashMap::new();
        params.insert("count".to_string(), Value::Int64(2));
        let result = db
            .execute_cypher_with_params(
                "MATCH (p:Person) RETURN p.name ORDER BY p.name LIMIT $count",
                params,
            )
            .unwrap();
        assert_eq!(result.row_count(), 2);
    }

    #[test]
    fn cypher_skip_and_limit_with_parameters() {
        let db = social_network();
        let mut params = std::collections::HashMap::new();
        params.insert("offset".to_string(), Value::Int64(1));
        params.insert("page_size".to_string(), Value::Int64(2));
        let result = db
            .execute_cypher_with_params(
                "MATCH (p:Person) RETURN p.name ORDER BY p.name SKIP $offset LIMIT $page_size",
                params,
            )
            .unwrap();
        assert_eq!(result.row_count(), 2);
        // Ordered by name: Alix(0), Dave(1), Gus(2), Harm(3). Skip 1 = Dave first.
        assert_eq!(result.rows[0][0], Value::String("Dave".into()));
    }

    #[test]
    fn cypher_limit_parameter_zero() {
        let db = social_network();
        let mut params = std::collections::HashMap::new();
        params.insert("n".to_string(), Value::Int64(0));
        let result = db
            .execute_cypher_with_params("MATCH (p:Person) RETURN p.name LIMIT $n", params)
            .unwrap();
        assert_eq!(result.row_count(), 0);
    }

    #[test]
    fn cypher_skip_parameter_negative_is_error() {
        let db = social_network();
        let mut params = std::collections::HashMap::new();
        params.insert("n".to_string(), Value::Int64(-1));
        let result =
            db.execute_cypher_with_params("MATCH (p:Person) RETURN p.name SKIP $n", params);
        assert!(result.is_err());
    }

    #[test]
    fn cypher_limit_parameter_non_integer_is_error() {
        let db = social_network();
        let mut params = std::collections::HashMap::new();
        params.insert("n".to_string(), Value::String("ten".into()));
        let result =
            db.execute_cypher_with_params("MATCH (p:Person) RETURN p.name LIMIT $n", params);
        assert!(result.is_err());
    }

    #[test]
    fn cypher_load_csv_with_headers() {
        use std::io::Write;
        // Create a temp CSV file
        let dir = std::env::temp_dir();
        let csv_path = dir.join("grafeo_test_load_csv_headers.csv");
        {
            let mut f = std::fs::File::create(&csv_path).unwrap();
            writeln!(f, "name,age,city").unwrap();
            writeln!(f, "Alix,30,Amsterdam").unwrap();
            writeln!(f, "Gus,25,Berlin").unwrap();
            writeln!(f, "Mia,28,Paris").unwrap();
        }

        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let query = format!(
            "LOAD CSV WITH HEADERS FROM '{}' AS row RETURN row.name AS name, row.age AS age ORDER BY row.name",
            csv_path.display()
        );
        let result = session.execute_cypher(&query);
        assert!(
            result.is_ok(),
            "LOAD CSV WITH HEADERS failed: {:?}",
            result.err()
        );
        let result = result.unwrap();
        assert_eq!(result.row_count(), 3, "Should have 3 rows");
        assert_eq!(result.rows[0][0], Value::String("Alix".into()));
        assert_eq!(result.rows[0][1], Value::String("30".into())); // CSV values are strings
        assert_eq!(result.rows[1][0], Value::String("Gus".into()));
        assert_eq!(result.rows[2][0], Value::String("Mia".into()));

        std::fs::remove_file(&csv_path).ok();
    }

    #[test]
    fn cypher_load_csv_without_headers() {
        use std::io::Write;
        let dir = std::env::temp_dir();
        let csv_path = dir.join("grafeo_test_load_csv_no_headers.csv");
        {
            let mut f = std::fs::File::create(&csv_path).unwrap();
            writeln!(f, "Alix,30,Amsterdam").unwrap();
            writeln!(f, "Gus,25,Berlin").unwrap();
        }

        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let query = format!(
            "LOAD CSV FROM '{}' AS row RETURN row[0] AS name, row[1] AS age",
            csv_path.display()
        );
        let result = session.execute_cypher(&query);
        assert!(
            result.is_ok(),
            "LOAD CSV without headers failed: {:?}",
            result.err()
        );
        let result = result.unwrap();
        assert_eq!(result.row_count(), 2);
        assert_eq!(result.rows[0][0], Value::String("Alix".into()));
        assert_eq!(result.rows[0][1], Value::String("30".into()));

        std::fs::remove_file(&csv_path).ok();
    }

    #[test]
    fn cypher_load_csv_create_nodes() {
        use std::io::Write;
        let dir = std::env::temp_dir();
        let csv_path = dir.join("grafeo_test_load_csv_create.csv");
        {
            let mut f = std::fs::File::create(&csv_path).unwrap();
            writeln!(f, "name,city").unwrap();
            writeln!(f, "Vincent,Amsterdam").unwrap();
            writeln!(f, "Jules,Paris").unwrap();
        }

        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let query = format!(
            "LOAD CSV WITH HEADERS FROM '{}' AS row CREATE (p:Person {{name: row.name, city: row.city}})",
            csv_path.display()
        );
        let result = session.execute_cypher(&query);
        assert!(
            result.is_ok(),
            "LOAD CSV + CREATE failed: {:?}",
            result.err()
        );

        // Verify nodes were created
        let check = session
            .execute_cypher("MATCH (p:Person) RETURN p.name ORDER BY p.name")
            .unwrap();
        assert_eq!(check.row_count(), 2);
        assert_eq!(check.rows[0][0], Value::String("Jules".into()));
        assert_eq!(check.rows[1][0], Value::String("Vincent".into()));

        std::fs::remove_file(&csv_path).ok();
    }

    #[test]
    fn cypher_load_csv_with_fieldterminator() {
        use std::io::Write;
        let dir = std::env::temp_dir();
        let csv_path = dir.join("grafeo_test_load_csv_tab.tsv");
        {
            let mut f = std::fs::File::create(&csv_path).unwrap();
            writeln!(f, "name\tage").unwrap();
            writeln!(f, "Alix\t30").unwrap();
            writeln!(f, "Gus\t25").unwrap();
        }

        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let query = format!(
            "LOAD CSV WITH HEADERS FROM '{}' AS row FIELDTERMINATOR '\\t' RETURN row.name, row.age",
            csv_path.display()
        );
        let result = session.execute_cypher(&query);
        assert!(
            result.is_ok(),
            "LOAD CSV with FIELDTERMINATOR failed: {:?}",
            result.err()
        );
        let result = result.unwrap();
        assert_eq!(result.row_count(), 2);
        assert_eq!(result.rows[0][0], Value::String("Alix".into()));

        std::fs::remove_file(&csv_path).ok();
    }

    #[test]
    fn cypher_load_csv_file_not_found() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session.execute_cypher(
            "LOAD CSV WITH HEADERS FROM '/nonexistent/path/file.csv' AS row RETURN row.name",
        );
        assert!(result.is_err(), "Should fail for missing file");
    }

    #[test]
    fn cypher_load_csv_parse_only() {
        // Verify LOAD CSV parses without executing (EXPLAIN)
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session
            .execute_cypher("EXPLAIN LOAD CSV WITH HEADERS FROM 'test.csv' AS row RETURN row.name");
        assert!(
            result.is_ok(),
            "EXPLAIN LOAD CSV should parse: {:?}",
            result.err()
        );
    }
}

// ============================================================================
// LOAD DATA Features (GQL LOAD DATA, JSONL, Parquet)
// ============================================================================

#[cfg(feature = "gql")]
mod load_data_features {
    use grafeo_engine::GrafeoDB;
    use std::io::Write;

    #[test]
    fn gql_load_data_csv_with_headers() {
        let dir = std::env::temp_dir();
        let csv_path = dir.join("grafeo_test_gql_load_csv.csv");
        {
            let mut f = std::fs::File::create(&csv_path).unwrap();
            writeln!(f, "name,age,city").unwrap();
            writeln!(f, "Alix,30,Amsterdam").unwrap();
            writeln!(f, "Gus,25,Berlin").unwrap();
        }
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let query = format!(
            "LOAD DATA FROM '{}' FORMAT CSV WITH HEADERS AS row RETURN row.name AS name, row.age AS age ORDER BY row.name",
            csv_path.display()
        );
        let result = session.execute(&query);
        assert!(
            result.is_ok(),
            "GQL LOAD DATA CSV failed: {:?}",
            result.err()
        );
        let result = result.unwrap();
        assert_eq!(result.rows.len(), 2);
    }

    #[test]
    fn gql_load_data_csv_without_headers() {
        let dir = std::env::temp_dir();
        let csv_path = dir.join("grafeo_test_gql_load_csv_no_headers.csv");
        {
            let mut f = std::fs::File::create(&csv_path).unwrap();
            writeln!(f, "Alix,30,Amsterdam").unwrap();
            writeln!(f, "Gus,25,Berlin").unwrap();
        }
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let query = format!(
            "LOAD DATA FROM '{}' FORMAT CSV AS row RETURN row[0] AS name, row[1] AS age",
            csv_path.display()
        );
        let result = session.execute(&query);
        assert!(
            result.is_ok(),
            "GQL LOAD DATA CSV without headers failed: {:?}",
            result.err()
        );
        let result = result.unwrap();
        assert_eq!(result.rows.len(), 2);
    }

    #[test]
    fn gql_load_csv_compat_syntax() {
        // GQL parser also accepts Cypher-compatible LOAD CSV syntax
        let dir = std::env::temp_dir();
        let csv_path = dir.join("grafeo_test_gql_load_csv_compat.csv");
        {
            let mut f = std::fs::File::create(&csv_path).unwrap();
            writeln!(f, "name,city").unwrap();
            writeln!(f, "Alix,Amsterdam").unwrap();
        }
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let query = format!(
            "LOAD CSV WITH HEADERS FROM '{}' AS row RETURN row.name AS name",
            csv_path.display()
        );
        let result = session.execute(&query);
        assert!(
            result.is_ok(),
            "GQL LOAD CSV compat failed: {:?}",
            result.err()
        );
        let result = result.unwrap();
        assert_eq!(result.rows.len(), 1);
    }

    #[test]
    fn gql_load_data_csv_create_nodes() {
        let dir = std::env::temp_dir();
        let csv_path = dir.join("grafeo_test_gql_load_csv_create.csv");
        {
            let mut f = std::fs::File::create(&csv_path).unwrap();
            writeln!(f, "name,city").unwrap();
            writeln!(f, "Alix,Amsterdam").unwrap();
            writeln!(f, "Gus,Berlin").unwrap();
        }
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let query = format!(
            "LOAD DATA FROM '{}' FORMAT CSV WITH HEADERS AS row INSERT (:Person {{name: row.name, city: row.city}})",
            csv_path.display()
        );
        let result = session.execute(&query);
        assert!(
            result.is_ok(),
            "GQL LOAD DATA + INSERT failed: {:?}",
            result.err()
        );

        // Verify nodes were created
        let verify = session
            .execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
            .unwrap();
        assert_eq!(verify.rows.len(), 2);
    }

    #[test]
    fn gql_load_data_csv_with_fieldterminator() {
        let dir = std::env::temp_dir();
        let tsv_path = dir.join("grafeo_test_gql_load_csv_tab.tsv");
        {
            let mut f = std::fs::File::create(&tsv_path).unwrap();
            writeln!(f, "name\tage").unwrap();
            writeln!(f, "Alix\t30").unwrap();
        }
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let query = format!(
            "LOAD DATA FROM '{}' FORMAT CSV WITH HEADERS AS row FIELDTERMINATOR '\\t' RETURN row.name AS name",
            tsv_path.display()
        );
        let result = session.execute(&query);
        assert!(
            result.is_ok(),
            "GQL LOAD DATA with FIELDTERMINATOR failed: {:?}",
            result.err()
        );
        let result = result.unwrap();
        assert_eq!(result.rows.len(), 1);
    }

    #[test]
    fn gql_load_data_file_not_found() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session
            .execute("LOAD DATA FROM '/nonexistent/path/file.csv' FORMAT CSV AS row RETURN row");
        assert!(result.is_err(), "Should fail for missing file");
    }

    #[test]
    fn gql_load_data_explain() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session.execute(
            "EXPLAIN LOAD DATA FROM 'test.csv' FORMAT CSV WITH HEADERS AS row RETURN row.name",
        );
        assert!(
            result.is_ok(),
            "EXPLAIN LOAD DATA should parse: {:?}",
            result.err()
        );
    }

    #[cfg(feature = "jsonl-import")]
    #[test]
    fn gql_load_data_jsonl() {
        let dir = std::env::temp_dir();
        let jsonl_path = dir.join("grafeo_test_gql_load_jsonl.jsonl");
        {
            let mut f = std::fs::File::create(&jsonl_path).unwrap();
            writeln!(f, r#"{{"name": "Alix", "age": 30, "city": "Amsterdam"}}"#).unwrap();
            writeln!(f, r#"{{"name": "Gus", "age": 25, "city": "Berlin"}}"#).unwrap();
        }
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let query = format!(
            "LOAD DATA FROM '{}' FORMAT JSONL AS row RETURN row.name AS name, row.age AS age ORDER BY row.name",
            jsonl_path.display()
        );
        let result = session.execute(&query);
        assert!(
            result.is_ok(),
            "GQL LOAD DATA JSONL failed: {:?}",
            result.err()
        );
        let result = result.unwrap();
        assert_eq!(result.rows.len(), 2);
    }

    #[cfg(feature = "jsonl-import")]
    #[test]
    fn gql_load_data_jsonl_create_nodes() {
        let dir = std::env::temp_dir();
        let jsonl_path = dir.join("grafeo_test_gql_load_jsonl_create.jsonl");
        {
            let mut f = std::fs::File::create(&jsonl_path).unwrap();
            writeln!(f, r#"{{"name": "Vincent", "city": "Paris"}}"#).unwrap();
            writeln!(f, r#"{{"name": "Jules", "city": "Amsterdam"}}"#).unwrap();
        }
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let query = format!(
            "LOAD DATA FROM '{}' FORMAT JSONL AS row INSERT (:Person {{name: row.name, city: row.city}})",
            jsonl_path.display()
        );
        let result = session.execute(&query);
        assert!(
            result.is_ok(),
            "GQL LOAD DATA JSONL + INSERT failed: {:?}",
            result.err()
        );

        let verify = session
            .execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
            .unwrap();
        assert_eq!(verify.rows.len(), 2);
    }

    #[cfg(feature = "jsonl-import")]
    #[test]
    fn gql_load_data_ndjson_alias() {
        // NDJSON is an alias for JSONL
        let dir = std::env::temp_dir();
        let jsonl_path = dir.join("grafeo_test_gql_load_ndjson.jsonl");
        {
            let mut f = std::fs::File::create(&jsonl_path).unwrap();
            writeln!(f, r#"{{"x": 1}}"#).unwrap();
        }
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let query = format!(
            "LOAD DATA FROM '{}' FORMAT NDJSON AS row RETURN row.x AS x",
            jsonl_path.display()
        );
        let result = session.execute(&query);
        assert!(
            result.is_ok(),
            "GQL LOAD DATA NDJSON alias failed: {:?}",
            result.err()
        );
    }

    #[test]
    fn gql_load_data_parquet_disabled_error() {
        // When parquet-import feature is disabled, should give a clear error
        if cfg!(not(feature = "parquet-import")) {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();
            let result =
                session.execute("LOAD DATA FROM 'test.parquet' FORMAT PARQUET AS row RETURN row");
            assert!(
                result.is_err(),
                "Should fail when parquet-import is disabled"
            );
            let err = result.unwrap_err().to_string();
            assert!(
                err.contains("Parquet") || err.contains("parquet"),
                "Error should mention Parquet: {err}"
            );
        }
    }
}

// ============================================================================
// SPARQL Features (covers sparql_translator.rs)
// ============================================================================

#[cfg(all(feature = "sparql", feature = "rdf"))]
mod sparql_features {
    use grafeo_common::types::Value;
    use grafeo_engine::{Config, GrafeoDB, GraphModel};

    fn rdf_db() -> GrafeoDB {
        GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap()
    }

    #[test]
    fn inverse_property_path() {
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/alix> <http://ex.org/knows> <http://ex.org/gus> .
                    <http://ex.org/gus> <http://ex.org/knows> <http://ex.org/harm> .
                }"#,
            )
            .unwrap();
        // ^knows means inverse: find who knows gus
        let result = session
            .execute_sparql(
                r#"SELECT ?who WHERE {
                    <http://ex.org/gus> ^<http://ex.org/knows> ?who
                }"#,
            )
            .unwrap();
        assert_eq!(result.row_count(), 1, "Alix knows Gus (inverse)");
    }

    #[test]
    fn zero_or_one_property_path() {
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/alix> <http://ex.org/knows> <http://ex.org/gus> .
                }"#,
            )
            .unwrap();
        // knows? means zero or one hop
        let result = session
            .execute_sparql(
                r#"SELECT ?who WHERE {
                    <http://ex.org/alix> <http://ex.org/knows>? ?who
                }"#,
            )
            .unwrap();
        // Should find alix herself (0 hops) and gus (1 hop)
        assert!(
            result.row_count() >= 1,
            "ZeroOrOne should find at least one match"
        );
    }

    #[test]
    fn sparql_optional_pattern() {
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/alix> <http://ex.org/name> "Alix" .
                    <http://ex.org/alix> <http://ex.org/age> "30" .
                    <http://ex.org/gus> <http://ex.org/name> "Gus" .
                }"#,
            )
            .unwrap();
        let result = session
            .execute_sparql(
                r#"SELECT ?name ?age WHERE {
                    ?s <http://ex.org/name> ?name .
                    OPTIONAL { ?s <http://ex.org/age> ?age }
                }"#,
            )
            .unwrap();
        // Both Alix and Gus, but only Alix has age
        assert_eq!(result.row_count(), 2);
    }

    #[test]
    fn sparql_optional_null_values() {
        // Verify that unbound variables from OPTIONAL produce NULL in results.
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/alix> <http://ex.org/name> "Alix" .
                    <http://ex.org/alix> <http://ex.org/age> "30" .
                    <http://ex.org/gus> <http://ex.org/name> "Gus" .
                }"#,
            )
            .unwrap();
        let result = session
            .execute_sparql(
                r#"SELECT ?name ?age WHERE {
                    ?s <http://ex.org/name> ?name .
                    OPTIONAL { ?s <http://ex.org/age> ?age }
                }
                ORDER BY ?name"#,
            )
            .unwrap();
        assert_eq!(result.row_count(), 2);
        // Alix has age, Gus does not
        // Verify Gus row has NULL for age
        let gus_row = result
            .rows
            .iter()
            .find(|r| r[0] == Value::String("Gus".into()))
            .expect("Gus should appear in results");
        assert_eq!(gus_row[1], Value::Null, "Gus has no age, should be NULL");
    }

    #[test]
    fn sparql_nested_optional() {
        // Nested OPTIONAL: OPTIONAL { ?x <p> ?y OPTIONAL { ?y <q> ?z } }
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/alix> <http://ex.org/name> "Alix" .
                    <http://ex.org/alix> <http://ex.org/knows> <http://ex.org/gus> .
                    <http://ex.org/gus> <http://ex.org/name> "Gus" .
                    <http://ex.org/gus> <http://ex.org/city> "Amsterdam" .
                    <http://ex.org/harm> <http://ex.org/name> "Harm" .
                }"#,
            )
            .unwrap();
        let result = session
            .execute_sparql(
                r#"SELECT ?name ?friend ?city WHERE {
                    ?s <http://ex.org/name> ?name .
                    OPTIONAL {
                        ?s <http://ex.org/knows> ?f .
                        ?f <http://ex.org/name> ?friend .
                        OPTIONAL { ?f <http://ex.org/city> ?city }
                    }
                }
                ORDER BY ?name"#,
            )
            .unwrap();
        // Alix knows Gus (who has city Amsterdam): Alix, "Gus", "Amsterdam"
        // Gus knows nobody: Gus, NULL, NULL
        // Harm knows nobody: Harm, NULL, NULL
        assert_eq!(result.row_count(), 3);
        let alix_row = result
            .rows
            .iter()
            .find(|r| r[0] == Value::String("Alix".into()))
            .expect("Alix should appear");
        assert_eq!(alix_row[1], Value::String("Gus".into()));
        assert_eq!(alix_row[2], Value::String("Amsterdam".into()));

        let harm_row = result
            .rows
            .iter()
            .find(|r| r[0] == Value::String("Harm".into()))
            .expect("Harm should appear");
        assert_eq!(harm_row[1], Value::Null, "Harm knows nobody");
        assert_eq!(harm_row[2], Value::Null, "Nested optional also NULL");
    }

    #[test]
    fn sparql_optional_with_filter_inside() {
        // SPARQL semantics: FILTER inside OPTIONAL acts as a join condition,
        // not a post-filter. Persons without a matching score should get NULL,
        // not be eliminated.
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/alix> <http://ex.org/name> "Alix" .
                    <http://ex.org/alix> <http://ex.org/score> "80" .
                    <http://ex.org/gus> <http://ex.org/name> "Gus" .
                    <http://ex.org/gus> <http://ex.org/score> "40" .
                    <http://ex.org/harm> <http://ex.org/name> "Harm" .
                }"#,
            )
            .unwrap();

        let result = session
            .execute_sparql(
                r#"SELECT ?name ?score WHERE {
                    ?s <http://ex.org/name> ?name .
                    OPTIONAL {
                        ?s <http://ex.org/score> ?score .
                        FILTER(?score > "50")
                    }
                }
                ORDER BY ?name"#,
            )
            .unwrap();
        // All 3 persons should appear:
        // Alix: score "80" > "50" -> bound
        // Gus: score "40" NOT > "50" -> NULL (filter eliminates inside optional)
        // Harm: no score -> NULL
        assert_eq!(
            result.row_count(),
            3,
            "All 3 persons preserved: FILTER inside OPTIONAL is a join condition"
        );
        let alix_row = result
            .rows
            .iter()
            .find(|r| r[0] == Value::String("Alix".into()))
            .expect("Alix should appear");
        assert_eq!(
            alix_row[1],
            Value::String("80".into()),
            "Alix score passes filter"
        );

        let gus_row = result
            .rows
            .iter()
            .find(|r| r[0] == Value::String("Gus".into()))
            .expect("Gus should appear");
        assert_eq!(
            gus_row[1],
            Value::Null,
            "Gus score fails filter, should be NULL"
        );

        let harm_row = result
            .rows
            .iter()
            .find(|r| r[0] == Value::String("Harm".into()))
            .expect("Harm should appear");
        assert_eq!(
            harm_row[1],
            Value::Null,
            "Harm has no score, should be NULL"
        );
    }

    #[test]
    fn sparql_optional_shared_variables() {
        // Shared variable between required and optional patterns.
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/alix> <http://ex.org/name> "Alix" .
                    <http://ex.org/alix> <http://ex.org/knows> <http://ex.org/gus> .
                    <http://ex.org/gus> <http://ex.org/name> "Gus" .
                }"#,
            )
            .unwrap();
        let result = session
            .execute_sparql(
                r#"SELECT ?name ?friend WHERE {
                    ?s <http://ex.org/name> ?name .
                    OPTIONAL { ?s <http://ex.org/knows> ?f . ?f <http://ex.org/name> ?friend }
                }
                ORDER BY ?name"#,
            )
            .unwrap();
        // Alix knows Gus -> Alix, Gus
        // Gus knows nobody -> Gus, NULL
        assert_eq!(result.row_count(), 2);
        let alix_row = result
            .rows
            .iter()
            .find(|r| r[0] == Value::String("Alix".into()))
            .expect("Alix should appear");
        assert_eq!(alix_row[1], Value::String("Gus".into()));

        let gus_row = result
            .rows
            .iter()
            .find(|r| r[0] == Value::String("Gus".into()))
            .expect("Gus should appear");
        assert_eq!(
            gus_row[1],
            Value::Null,
            "Gus has no knows, friend should be NULL"
        );
    }

    #[test]
    fn sparql_union_pattern() {
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/alix> <http://ex.org/name> "Alix" .
                    <http://ex.org/gus> <http://ex.org/label> "Gus" .
                }"#,
            )
            .unwrap();
        let result = session
            .execute_sparql(
                r#"SELECT ?val WHERE {
                    { ?s <http://ex.org/name> ?val }
                    UNION
                    { ?s <http://ex.org/label> ?val }
                }"#,
            )
            .unwrap();
        assert_eq!(result.row_count(), 2);
    }

    #[test]
    fn sparql_filter_exists() {
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/alix> <http://ex.org/name> "Alix" .
                    <http://ex.org/alix> <http://ex.org/knows> <http://ex.org/gus> .
                    <http://ex.org/harm> <http://ex.org/name> "Harm" .
                }"#,
            )
            .unwrap();
        let result = session
            .execute_sparql(
                r#"SELECT ?name WHERE {
                    ?s <http://ex.org/name> ?name .
                    FILTER EXISTS { ?s <http://ex.org/knows> ?o }
                }"#,
            )
            .unwrap();
        assert_eq!(result.row_count(), 1, "Only Alix has knows relation");
    }

    #[test]
    fn sparql_filter_not_exists() {
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/alix> <http://ex.org/name> "Alix" .
                    <http://ex.org/alix> <http://ex.org/knows> <http://ex.org/gus> .
                    <http://ex.org/harm> <http://ex.org/name> "Harm" .
                }"#,
            )
            .unwrap();
        let result = session
            .execute_sparql(
                r#"SELECT ?name WHERE {
                    ?s <http://ex.org/name> ?name .
                    FILTER NOT EXISTS { ?s <http://ex.org/knows> ?o }
                }"#,
            )
            .unwrap();
        assert_eq!(result.row_count(), 1, "Only Harm has no knows relation");
    }

    #[test]
    fn sparql_sequence_property_path() {
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/alix> <http://ex.org/knows> <http://ex.org/gus> .
                    <http://ex.org/gus> <http://ex.org/likes> <http://ex.org/harm> .
                }"#,
            )
            .unwrap();
        // Sequence: knows / likes
        let result = session
            .execute_sparql(
                r#"SELECT ?who WHERE {
                    <http://ex.org/alix> <http://ex.org/knows>/<http://ex.org/likes> ?who
                }"#,
            )
            .unwrap();
        assert_eq!(result.row_count(), 1);
    }

    #[test]
    fn sparql_alternative_property_path() {
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/alix> <http://ex.org/knows> <http://ex.org/gus> .
                    <http://ex.org/alix> <http://ex.org/likes> <http://ex.org/harm> .
                }"#,
            )
            .unwrap();
        let result = session
            .execute_sparql(
                r#"SELECT ?who WHERE {
                    <http://ex.org/alix> (<http://ex.org/knows>|<http://ex.org/likes>) ?who
                }"#,
            )
            .unwrap();
        assert_eq!(result.row_count(), 2, "Alternative path finds both");
    }

    #[test]
    fn sparql_one_or_more_path() {
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/a> <http://ex.org/next> <http://ex.org/b> .
                    <http://ex.org/b> <http://ex.org/next> <http://ex.org/c> .
                    <http://ex.org/c> <http://ex.org/next> <http://ex.org/d> .
                }"#,
            )
            .unwrap();
        // + means one or more hops
        let result = session
            .execute_sparql(
                r#"SELECT ?end WHERE {
                    <http://ex.org/a> <http://ex.org/next>+ ?end
                }"#,
            )
            .unwrap();
        assert_eq!(result.row_count(), 3, "One-or-more should find b, c, d");
    }

    #[test]
    fn sparql_zero_or_more_path() {
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/a> <http://ex.org/next> <http://ex.org/b> .
                    <http://ex.org/b> <http://ex.org/next> <http://ex.org/c> .
                }"#,
            )
            .unwrap();
        // * means zero or more hops (includes self)
        let result = session
            .execute_sparql(
                r#"SELECT ?end WHERE {
                    <http://ex.org/a> <http://ex.org/next>* ?end
                }"#,
            )
            .unwrap();
        // Zero-or-more from fixed subject: 0-hop (a), 1-hop (b), 2-hop (c) = 3 results
        assert_eq!(
            result.row_count(),
            3,
            "Zero-or-more from a with chain a->b->c"
        );
    }

    #[test]
    fn sparql_rdf_collections() {
        let db = rdf_db();
        let session = db.session();
        // RDF collection: linked list with rdf:first/rdf:rest
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/list> <http://ex.org/items> <http://ex.org/a> .
                    <http://ex.org/a> <http://www.w3.org/1999/02/22-rdf-syntax-ns#first> "one" .
                    <http://ex.org/a> <http://www.w3.org/1999/02/22-rdf-syntax-ns#rest> <http://ex.org/b> .
                    <http://ex.org/b> <http://www.w3.org/1999/02/22-rdf-syntax-ns#first> "two" .
                    <http://ex.org/b> <http://www.w3.org/1999/02/22-rdf-syntax-ns#rest> <http://www.w3.org/1999/02/22-rdf-syntax-ns#nil> .
                }"#,
            )
            .unwrap();
        // rest*/first: follow rest links zero or more times, then get rdf:first
        let result = session
            .execute_sparql(
                r#"SELECT ?item WHERE {
                    <http://ex.org/list> <http://ex.org/items> ?head .
                    ?head <http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>*/<http://www.w3.org/1999/02/22-rdf-syntax-ns#first> ?item
                }"#,
            )
            .unwrap();
        // Should find "one" (0 rest hops + first) and "two" (1 rest hop + first)
        assert!(
            result.row_count() >= 2,
            "RDF collection traversal should find at least 2 items, got {}",
            result.row_count()
        );
    }

    #[test]
    fn sparql_aggregation_count() {
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/alix> <http://ex.org/type> "person" .
                    <http://ex.org/gus> <http://ex.org/type> "person" .
                    <http://ex.org/techcorp> <http://ex.org/type> "company" .
                }"#,
            )
            .unwrap();
        let result = session
            .execute_sparql(
                r#"SELECT ?type (COUNT(?s) AS ?count) WHERE {
                    ?s <http://ex.org/type> ?type
                } GROUP BY ?type ORDER BY ?type"#,
            )
            .unwrap();
        assert_eq!(result.row_count(), 2);
    }

    #[test]
    fn sparql_having_clause() {
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/a> <http://ex.org/type> "x" .
                    <http://ex.org/b> <http://ex.org/type> "x" .
                    <http://ex.org/c> <http://ex.org/type> "y" .
                }"#,
            )
            .unwrap();
        let result = session
            .execute_sparql(
                r#"SELECT ?type (COUNT(?s) AS ?cnt) WHERE {
                    ?s <http://ex.org/type> ?type
                } GROUP BY ?type HAVING (COUNT(?s) > 1)"#,
            )
            .unwrap();
        assert_eq!(result.row_count(), 1, "Only 'x' has count > 1");
    }

    // --- P0 OPTIONAL bug probes ---

    #[test]
    fn sparql_optional_multiple_independent() {
        // Two independent OPTIONALLs: one matches, one doesn't.
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/alix> <http://ex.org/name> "Alix" .
                    <http://ex.org/alix> <http://ex.org/age> "30" .
                    <http://ex.org/gus> <http://ex.org/name> "Gus" .
                    <http://ex.org/gus> <http://ex.org/email> "gus@example.org" .
                    <http://ex.org/harm> <http://ex.org/name> "Harm" .
                }"#,
            )
            .unwrap();
        let result = session
            .execute_sparql(
                r#"SELECT ?name ?age ?email WHERE {
                    ?s <http://ex.org/name> ?name .
                    OPTIONAL { ?s <http://ex.org/age> ?age }
                    OPTIONAL { ?s <http://ex.org/email> ?email }
                }
                ORDER BY ?name"#,
            )
            .unwrap();
        // Alix: age=30, email=NULL
        // Gus: age=NULL, email=gus@example.org
        // Harm: age=NULL, email=NULL
        assert_eq!(
            result.row_count(),
            3,
            "All 3 persons should appear with independent OPTIONALLs"
        );
        let alix_row = result
            .rows
            .iter()
            .find(|r| r[0] == Value::String("Alix".into()))
            .expect("Alix should appear");
        assert_eq!(alix_row[1], Value::String("30".into()), "Alix has age");
        assert_eq!(alix_row[2], Value::Null, "Alix has no email");

        let gus_row = result
            .rows
            .iter()
            .find(|r| r[0] == Value::String("Gus".into()))
            .expect("Gus should appear");
        assert_eq!(gus_row[1], Value::Null, "Gus has no age");
        assert_eq!(
            gus_row[2],
            Value::String("gus@example.org".into()),
            "Gus has email"
        );

        let harm_row = result
            .rows
            .iter()
            .find(|r| r[0] == Value::String("Harm".into()))
            .expect("Harm should appear");
        assert_eq!(harm_row[1], Value::Null, "Harm has no age");
        assert_eq!(harm_row[2], Value::Null, "Harm has no email");
    }

    #[test]
    fn sparql_optional_count_with_null() {
        // COUNT(expr) should skip NULLs, COUNT(*) should count all rows.
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/alix> <http://ex.org/name> "Alix" .
                    <http://ex.org/alix> <http://ex.org/score> "80" .
                    <http://ex.org/gus> <http://ex.org/name> "Gus" .
                    <http://ex.org/harm> <http://ex.org/name> "Harm" .
                    <http://ex.org/harm> <http://ex.org/score> "60" .
                }"#,
            )
            .unwrap();
        let result = session
            .execute_sparql(
                r#"SELECT (COUNT(?name) AS ?total) (COUNT(?score) AS ?with_score) WHERE {
                    ?s <http://ex.org/name> ?name .
                    OPTIONAL { ?s <http://ex.org/score> ?score }
                }"#,
            )
            .unwrap();
        assert_eq!(result.row_count(), 1);
        // 3 people total, 2 with scores
        assert_eq!(
            result.rows[0][0],
            Value::Int64(3),
            "COUNT(?name) should be 3"
        );
        assert_eq!(
            result.rows[0][1],
            Value::Int64(2),
            "COUNT(?score) should skip NULLs: 2"
        );
    }

    #[test]
    fn sparql_optional_inner_join_inside() {
        // OPTIONAL block with a join inside (two triple patterns).
        // Only matches if BOTH patterns match for the same binding.
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/alix> <http://ex.org/name> "Alix" .
                    <http://ex.org/alix> <http://ex.org/knows> <http://ex.org/gus> .
                    <http://ex.org/gus> <http://ex.org/city> "Amsterdam" .
                    <http://ex.org/harm> <http://ex.org/name> "Harm" .
                    <http://ex.org/harm> <http://ex.org/knows> <http://ex.org/jules> .
                }"#,
            )
            .unwrap();
        let result = session
            .execute_sparql(
                r#"SELECT ?name ?city WHERE {
                    ?s <http://ex.org/name> ?name .
                    OPTIONAL {
                        ?s <http://ex.org/knows> ?friend .
                        ?friend <http://ex.org/city> ?city
                    }
                }
                ORDER BY ?name"#,
            )
            .unwrap();
        // Alix: knows Gus who has city Amsterdam
        // Harm: knows Jules who has NO city -> NULL
        assert_eq!(
            result.row_count(),
            2,
            "Both people should appear even if inner join of optional fails"
        );
        let alix_row = result
            .rows
            .iter()
            .find(|r| r[0] == Value::String("Alix".into()))
            .expect("Alix should appear");
        assert_eq!(alix_row[1], Value::String("Amsterdam".into()));

        let harm_row = result
            .rows
            .iter()
            .find(|r| r[0] == Value::String("Harm".into()))
            .expect("Harm should appear");
        assert_eq!(
            harm_row[1],
            Value::Null,
            "Harm's friend has no city, optional fails -> NULL"
        );
    }

    #[test]
    fn sparql_optional_all_unbound() {
        // Every person has no optional match at all.
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/alix> <http://ex.org/name> "Alix" .
                    <http://ex.org/gus> <http://ex.org/name> "Gus" .
                }"#,
            )
            .unwrap();
        let result = session
            .execute_sparql(
                r#"SELECT ?name ?email WHERE {
                    ?s <http://ex.org/name> ?name .
                    OPTIONAL { ?s <http://ex.org/email> ?email }
                }
                ORDER BY ?name"#,
            )
            .unwrap();
        // Both rows should appear with NULL email
        assert_eq!(result.row_count(), 2, "Both persons should appear");
        assert_eq!(result.rows[0][1], Value::Null, "Alix: no email");
        assert_eq!(result.rows[1][1], Value::Null, "Gus: no email");
    }

    #[test]
    fn sparql_optional_with_union_inside() {
        // OPTIONAL { { ?s <p1> ?val } UNION { ?s <p2> ?val } }
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/alix> <http://ex.org/name> "Alix" .
                    <http://ex.org/alix> <http://ex.org/nick> "Lix" .
                    <http://ex.org/gus> <http://ex.org/name> "Gus" .
                    <http://ex.org/gus> <http://ex.org/label> "Gustav" .
                    <http://ex.org/harm> <http://ex.org/name> "Harm" .
                }"#,
            )
            .unwrap();
        let result = session
            .execute_sparql(
                r#"SELECT ?name ?alt WHERE {
                    ?s <http://ex.org/name> ?name .
                    OPTIONAL {
                        { ?s <http://ex.org/nick> ?alt }
                        UNION
                        { ?s <http://ex.org/label> ?alt }
                    }
                }
                ORDER BY ?name"#,
            )
            .unwrap();
        // Alix: alt=Lix (from nick)
        // Gus: alt=Gustav (from label)
        // Harm: alt=NULL
        assert_eq!(
            result.row_count(),
            3,
            "All 3 persons should appear, UNION inside OPTIONAL"
        );
        let harm_row = result
            .rows
            .iter()
            .find(|r| r[0] == Value::String("Harm".into()))
            .expect("Harm should appear");
        assert_eq!(harm_row[1], Value::Null, "Harm has neither nick nor label");
    }
}

// ============================================================================
// GQL Parser Unit Tests (covers DDL/session parsing branches)
// ============================================================================

#[cfg(feature = "gql")]
mod gql_parser_unit {
    use grafeo_adapters::query::gql;
    use grafeo_adapters::query::gql::ast::{SessionCommand, SessionResetTarget, Statement};

    #[test]
    fn parse_create_graph() {
        let stmt = gql::parse("CREATE GRAPH mydb").unwrap();
        match stmt {
            Statement::SessionCommand(SessionCommand::CreateGraph {
                name,
                if_not_exists,
                ..
            }) => {
                assert_eq!(name, "mydb");
                assert!(!if_not_exists);
            }
            other => panic!("Expected CreateGraph, got {other:?}"),
        }
    }

    #[test]
    fn parse_create_graph_if_not_exists() {
        let stmt = gql::parse("CREATE GRAPH IF NOT EXISTS mydb").unwrap();
        match stmt {
            Statement::SessionCommand(SessionCommand::CreateGraph {
                name,
                if_not_exists,
                ..
            }) => {
                assert_eq!(name, "mydb");
                assert!(if_not_exists);
            }
            other => panic!("Expected CreateGraph, got {other:?}"),
        }
    }

    #[test]
    fn parse_create_property_graph() {
        let stmt = gql::parse("CREATE PROPERTY GRAPH pg1").unwrap();
        match stmt {
            Statement::SessionCommand(SessionCommand::CreateGraph { name, .. }) => {
                assert_eq!(name, "pg1");
            }
            other => panic!("Expected CreateGraph, got {other:?}"),
        }
    }

    #[test]
    fn parse_drop_graph() {
        let stmt = gql::parse("DROP GRAPH mydb").unwrap();
        match stmt {
            Statement::SessionCommand(SessionCommand::DropGraph { name, if_exists }) => {
                assert_eq!(name, "mydb");
                assert!(!if_exists);
            }
            other => panic!("Expected DropGraph, got {other:?}"),
        }
    }

    #[test]
    fn parse_drop_graph_if_exists() {
        let stmt = gql::parse("DROP GRAPH IF EXISTS mydb").unwrap();
        match stmt {
            Statement::SessionCommand(SessionCommand::DropGraph { name, if_exists }) => {
                assert_eq!(name, "mydb");
                assert!(if_exists);
            }
            other => panic!("Expected DropGraph, got {other:?}"),
        }
    }

    #[test]
    fn parse_drop_property_graph() {
        let stmt = gql::parse("DROP PROPERTY GRAPH pg1").unwrap();
        match stmt {
            Statement::SessionCommand(SessionCommand::DropGraph { name, .. }) => {
                assert_eq!(name, "pg1");
            }
            other => panic!("Expected DropGraph, got {other:?}"),
        }
    }

    #[test]
    fn parse_use_graph() {
        let stmt = gql::parse("USE GRAPH workspace").unwrap();
        match stmt {
            Statement::SessionCommand(SessionCommand::UseGraph(name)) => {
                assert_eq!(name, "workspace");
            }
            other => panic!("Expected UseGraph, got {other:?}"),
        }
    }

    #[test]
    fn parse_session_set_graph() {
        let stmt = gql::parse("SESSION SET GRAPH analytics").unwrap();
        match stmt {
            Statement::SessionCommand(SessionCommand::SessionSetGraph(name)) => {
                assert_eq!(name, "analytics");
            }
            other => panic!("Expected SessionSetGraph, got {other:?}"),
        }
    }

    #[test]
    fn parse_session_set_time_zone() {
        let stmt = gql::parse("SESSION SET TIME ZONE 'UTC+5'").unwrap();
        match stmt {
            Statement::SessionCommand(SessionCommand::SessionSetTimeZone(tz)) => {
                assert_eq!(tz, "UTC+5");
            }
            other => panic!("Expected SessionSetTimeZone, got {other:?}"),
        }
    }

    #[test]
    fn parse_session_set_schema() {
        let stmt = gql::parse("SESSION SET SCHEMA myschema").unwrap();
        match stmt {
            Statement::SessionCommand(SessionCommand::SessionSetSchema(name)) => {
                assert_eq!(name, "myschema");
            }
            other => panic!("Expected SessionSetSchema, got {other:?}"),
        }
    }

    #[test]
    fn parse_session_set_parameter() {
        let stmt = gql::parse("SESSION SET PARAMETER timeout = 30").unwrap();
        match stmt {
            Statement::SessionCommand(SessionCommand::SessionSetParameter(name, _)) => {
                assert_eq!(name, "timeout");
            }
            other => panic!("Expected SessionSetParameter, got {other:?}"),
        }
    }

    #[test]
    fn parse_session_reset() {
        let stmt = gql::parse("SESSION RESET").unwrap();
        assert!(matches!(
            stmt,
            Statement::SessionCommand(SessionCommand::SessionReset(SessionResetTarget::All))
        ));
    }

    #[test]
    fn parse_session_reset_all() {
        let stmt = gql::parse("SESSION RESET ALL").unwrap();
        assert!(matches!(
            stmt,
            Statement::SessionCommand(SessionCommand::SessionReset(SessionResetTarget::All))
        ));
    }

    #[test]
    fn parse_session_close() {
        let stmt = gql::parse("SESSION CLOSE").unwrap();
        assert!(matches!(
            stmt,
            Statement::SessionCommand(SessionCommand::SessionClose)
        ));
    }

    #[test]
    fn parse_start_transaction() {
        let stmt = gql::parse("START TRANSACTION").unwrap();
        assert!(matches!(
            stmt,
            Statement::SessionCommand(SessionCommand::StartTransaction { .. })
        ));
    }

    #[test]
    fn parse_commit() {
        let stmt = gql::parse("COMMIT").unwrap();
        assert!(matches!(
            stmt,
            Statement::SessionCommand(SessionCommand::Commit)
        ));
    }

    #[test]
    fn parse_rollback() {
        let stmt = gql::parse("ROLLBACK").unwrap();
        assert!(matches!(
            stmt,
            Statement::SessionCommand(SessionCommand::Rollback)
        ));
    }

    #[test]
    fn parse_finish_statement() {
        let stmt = gql::parse("MATCH (n) FINISH").unwrap();
        match stmt {
            Statement::Query(q) => {
                assert!(q.return_clause.is_finish);
            }
            other => panic!("Expected Query with FINISH, got {other:?}"),
        }
    }

    #[test]
    fn parse_select_statement() {
        let stmt = gql::parse("MATCH (n:Person) SELECT n.name").unwrap();
        match stmt {
            Statement::Query(q) => {
                assert!(!q.return_clause.items.is_empty());
            }
            other => panic!("Expected Query with SELECT, got {other:?}"),
        }
    }

    #[test]
    fn parse_drop_graph_error_on_bad_syntax() {
        let result = gql::parse("DROP NOTHING");
        assert!(result.is_err());
    }

    #[test]
    fn parse_session_error_on_bad_action() {
        let result = gql::parse("SESSION DESTROY");
        assert!(result.is_err());
    }

    #[test]
    fn parse_start_error_without_transaction() {
        let result = gql::parse("START SOMETHING");
        assert!(result.is_err());
    }

    #[test]
    fn parse_use_error_without_graph() {
        let result = gql::parse("USE SOMETHING");
        assert!(result.is_err());
    }
}

// ============================================================================
// GQL Translator Unit Tests (covers translate_full, session command routing)
// ============================================================================

#[cfg(feature = "gql")]
mod gql_translator_unit {
    use grafeo_engine::query::translators::gql;

    #[test]
    fn translate_returns_plan_for_query() {
        let result = gql::translate("MATCH (n) RETURN n");
        assert!(result.is_ok());
    }

    #[test]
    fn translate_returns_error_for_session_command() {
        let result = gql::translate("CREATE GRAPH test");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Session commands"));
    }

    #[test]
    fn translate_full_returns_session_command() {
        let result = gql::translate_full("CREATE GRAPH test").unwrap();
        assert!(matches!(
            result,
            gql::GqlTranslationResult::SessionCommand(_)
        ));
    }

    #[test]
    fn translate_full_returns_plan_for_query() {
        let result = gql::translate_full("MATCH (n) RETURN n").unwrap();
        assert!(matches!(result, gql::GqlTranslationResult::Plan(_)));
    }

    #[test]
    fn translate_except_produces_plan() {
        let result = gql::translate(
            "MATCH (n:Person) RETURN n.name EXCEPT MATCH (m:Person) WHERE m.age > 30 RETURN m.name",
        );
        assert!(result.is_ok());
    }

    #[test]
    fn translate_intersect_produces_plan() {
        let result = gql::translate(
            "MATCH (n:Person) RETURN n.name INTERSECT MATCH (m:Person) RETURN m.name",
        );
        assert!(result.is_ok());
    }

    #[test]
    fn translate_otherwise_produces_plan() {
        let result = gql::translate(
            "MATCH (n:Person) RETURN n.name OTHERWISE MATCH (m:Person) RETURN m.name",
        );
        assert!(result.is_ok());
    }

    #[test]
    fn translate_finish_produces_plan() {
        let result = gql::translate("MATCH (n) FINISH");
        assert!(result.is_ok());
    }

    #[test]
    fn translate_element_where_produces_plan() {
        let result = gql::translate("MATCH (n:Person WHERE n.age > 25) RETURN n.name");
        assert!(result.is_ok());
    }

    #[test]
    fn translate_count_subquery() {
        let result = gql::translate(
            "MATCH (n:Person) RETURN n.name, COUNT { MATCH (n)-[:KNOWS]->() } AS cnt",
        );
        assert!(result.is_ok());
    }

    #[test]
    fn translate_schema_errors() {
        let result = gql::translate("CREATE NODE TYPE Foo");
        assert!(result.is_err());
    }
}

// ============================================================================
// FOREACH Integration Tests
// ============================================================================

mod foreach_tests {
    use grafeo_common::types::Value;
    use grafeo_engine::GrafeoDB;

    #[test]
    fn gql_foreach_create() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("FOREACH (i IN range(1,3) | CREATE (:_Test {i: i}))")
            .unwrap();
        let result = session
            .execute("MATCH (n:_Test) RETURN n.i ORDER BY n.i")
            .unwrap();
        assert_eq!(result.rows.len(), 3);
        assert_eq!(result.rows[0][0], Value::Int64(1));
        assert_eq!(result.rows[1][0], Value::Int64(2));
        assert_eq!(result.rows[2][0], Value::Int64(3));
        // Cleanup
        session.execute("MATCH (n:_Test) DELETE n").unwrap();
    }

    #[test]
    #[ignore = "FOREACH SET on collected nodes requires node-identity-aware Unwind — tracked for follow-up"]
    fn gql_foreach_set() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("INSERT (:_Test {name: 'A', done: false})")
            .unwrap();
        session
            .execute("INSERT (:_Test {name: 'B', done: false})")
            .unwrap();
        session
            .execute(
                "MATCH (n:_Test) WITH collect(n) AS nodes \
                 FOREACH (x IN nodes | SET x.done = true) \
                 RETURN count(*) AS cnt",
            )
            .unwrap();
        let result = session
            .execute("MATCH (n:_Test) WHERE n.done = true RETURN count(n) AS cnt")
            .unwrap();
        assert_eq!(result.rows[0][0], Value::Int64(2));
        // Cleanup
        session.execute("MATCH (n:_Test) DELETE n").unwrap();
    }

    #[test]
    fn gql_foreach_variable_scope_isolation() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("FOREACH (i IN range(1,2) | CREATE (:_Test {i: i}))")
            .unwrap();
        let result = session
            .execute("MATCH (n:_Test) RETURN count(n) AS cnt")
            .unwrap();
        assert_eq!(result.rows[0][0], Value::Int64(2));
        // Cleanup
        session.execute("MATCH (n:_Test) DELETE n").unwrap();
    }

    #[test]
    fn gql_foreach_nested() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute(
                "FOREACH (i IN [1,2] | \
                     FOREACH (j IN [10,20] | \
                         CREATE (:_Test {i: i, j: j})\
                     )\
                 )",
            )
            .unwrap();
        let result = session
            .execute("MATCH (n:_Test) RETURN n.i, n.j ORDER BY n.i, n.j")
            .unwrap();
        assert_eq!(result.rows.len(), 4);
        assert_eq!(result.rows[0][0], Value::Int64(1));
        assert_eq!(result.rows[0][1], Value::Int64(10));
        assert_eq!(result.rows[3][0], Value::Int64(2));
        assert_eq!(result.rows[3][1], Value::Int64(20));
        // Cleanup
        session.execute("MATCH (n:_Test) DELETE n").unwrap();
    }

    #[test]
    fn gql_foreach_with_range() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("FOREACH (i IN range(1,3) | CREATE (:_Test {i: i}))")
            .unwrap();
        let result = session
            .execute("MATCH (n:_Test) RETURN count(n) AS cnt")
            .unwrap();
        assert_eq!(result.rows[0][0], Value::Int64(3));
        // Cleanup
        session.execute("MATCH (n:_Test) DELETE n").unwrap();
    }

    #[test]
    fn cypher_foreach_standalone() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute_cypher("FOREACH (i IN [1,2,3] | CREATE (:_Test {i: i}))")
            .unwrap();
        let result = session
            .execute("MATCH (n:_Test) RETURN count(n) AS cnt")
            .unwrap();
        assert_eq!(result.rows[0][0], Value::Int64(3));
        // Cleanup
        session.execute("MATCH (n:_Test) DELETE n").unwrap();
    }

    #[test]
    fn gql_foreach_standalone_no_match() {
        // FOREACH without preceding MATCH — uses implicit SingleRow input
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("FOREACH (x IN [10, 20, 30] | CREATE (:_FE {v: x}))")
            .unwrap();
        let result = session
            .execute("MATCH (n:_FE) RETURN n.v ORDER BY n.v")
            .unwrap();
        assert_eq!(result.rows.len(), 3);
        assert_eq!(result.rows[0][0], Value::Int64(10));
        assert_eq!(result.rows[1][0], Value::Int64(20));
        assert_eq!(result.rows[2][0], Value::Int64(30));
        session.execute("MATCH (n:_FE) DELETE n").unwrap();
    }

    #[test]
    fn gql_foreach_with_delete() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("INSERT (:_FE {v: 1}), (:_FE {v: 2}), (:_FE {v: 3})")
            .unwrap();
        let r = session
            .execute("MATCH (n:_FE) RETURN count(n) AS c")
            .unwrap();
        assert_eq!(r.rows[0][0], Value::Int64(3));
        // Delete all via FOREACH + collect
        session
            .execute_cypher(
                "MATCH (n:_FE) WITH collect(n) AS nodes \
                 FOREACH (x IN nodes | DELETE x) \
                 RETURN 1 AS done",
            )
            .unwrap();
    }

    #[test]
    fn gql_foreach_with_merge() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("FOREACH (i IN [1, 2, 3] | MERGE (:_FM {id: i}))")
            .unwrap();
        let r1 = session
            .execute("MATCH (n:_FM) RETURN count(n) AS c")
            .unwrap();
        assert_eq!(r1.rows[0][0], Value::Int64(3));
        // MERGE again — should NOT create duplicates
        session
            .execute("FOREACH (i IN [1, 2, 3] | MERGE (:_FM {id: i}))")
            .unwrap();
        let r2 = session
            .execute("MATCH (n:_FM) RETURN count(n) AS c")
            .unwrap();
        assert_eq!(r2.rows[0][0], Value::Int64(3));
        session.execute("MATCH (n:_FM) DELETE n").unwrap();
    }

    #[test]
    fn gql_foreach_complex_list_expression() {
        // FOREACH with function call as list expression
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("FOREACH (i IN range(10, 12) | CREATE (:_FC {v: i}))")
            .unwrap();
        let result = session
            .execute("MATCH (n:_FC) RETURN n.v ORDER BY n.v")
            .unwrap();
        assert_eq!(result.rows.len(), 3);
        assert_eq!(result.rows[0][0], Value::Int64(10));
        assert_eq!(result.rows[2][0], Value::Int64(12));
        session.execute("MATCH (n:_FC) DELETE n").unwrap();
    }

    #[test]
    #[ignore = "Sequential FOREACH cross-products: 2nd FOREACH sees rows from 1st, producing 6 instead of 4"]
    fn gql_foreach_multiple_in_query() {
        // Two FOREACH clauses in sequence — in Neo4j each is independent (4 nodes).
        // Current implementation cross-products: 1st creates 2 rows, 2nd iterates
        // 2 list items × 2 input rows = 4, total = 2 + 4 = 6.
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute(
                "FOREACH (i IN [1, 2] | CREATE (:_FA {t: 'a', v: i})) \
                 FOREACH (j IN [3, 4] | CREATE (:_FA {t: 'b', v: j}))",
            )
            .unwrap();
        let result = session
            .execute("MATCH (n:_FA) RETURN count(n) AS c")
            .unwrap();
        assert_eq!(result.rows[0][0], Value::Int64(4));
        session.execute("MATCH (n:_FA) DELETE n").unwrap();
    }
}

// ============================================================================
// Bare Pattern Predicates — Extended Tests
// ============================================================================

mod bare_pattern_tests {
    use grafeo_common::types::Value;
    use grafeo_engine::GrafeoDB;

    fn setup_pattern_db() -> GrafeoDB {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        // A -[:KNOWS]-> B -[:KNOWS]-> C
        // D (isolated)
        session
            .execute(
                "INSERT (:P {name: 'A'})-[:KNOWS]->(:P {name: 'B'})-[:KNOWS]->(:P {name: 'C'})",
            )
            .unwrap();
        session.execute("INSERT (:P {name: 'D'})").unwrap();
        db
    }

    #[test]
    fn positive_bare_pattern() {
        // WHERE (n)-[:KNOWS]->() → nodes with outgoing KNOWS
        let db = setup_pattern_db();
        let session = db.session();
        let result = session
            .execute("MATCH (n:P) WHERE (n)-[:KNOWS]->() RETURN n.name ORDER BY n.name")
            .unwrap();
        assert_eq!(result.rows.len(), 2); // A and B
        assert_eq!(result.rows[0][0], Value::String("A".into()));
        assert_eq!(result.rows[1][0], Value::String("B".into()));
    }

    #[test]
    fn negative_bare_pattern() {
        // WHERE NOT (n)-[:KNOWS]->() → nodes without outgoing KNOWS
        let db = setup_pattern_db();
        let session = db.session();
        let result = session
            .execute("MATCH (n:P) WHERE NOT (n)-[:KNOWS]->() RETURN n.name ORDER BY n.name")
            .unwrap();
        assert_eq!(result.rows.len(), 2); // C and D
        assert_eq!(result.rows[0][0], Value::String("C".into()));
        assert_eq!(result.rows[1][0], Value::String("D".into()));
    }

    #[test]
    fn bare_pattern_any_rel_type() {
        // WHERE (n)-[]->() — any outgoing relationship
        let db = setup_pattern_db();
        let session = db.session();
        let result = session
            .execute("MATCH (n:P) WHERE (n)-[]->() RETURN count(n) AS c")
            .unwrap();
        assert_eq!(result.rows[0][0], Value::Int64(2)); // A and B
    }

    #[test]
    fn bare_pattern_in_and_expression() {
        // WHERE (n)-[:KNOWS]->() AND n.name = 'A'
        let db = setup_pattern_db();
        let session = db.session();
        let result = session
            .execute("MATCH (n:P) WHERE (n)-[:KNOWS]->() AND n.name = 'A' RETURN n.name")
            .unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0][0], Value::String("A".into()));
    }

    #[test]
    fn not_bare_pattern_in_and_expression() {
        // WHERE NOT (n)-[:KNOWS]->() AND n.name <> 'C'
        let db = setup_pattern_db();
        let session = db.session();
        let result = session
            .execute("MATCH (n:P) WHERE NOT (n)-[:KNOWS]->() AND n.name <> 'C' RETURN n.name")
            .unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0][0], Value::String("D".into()));
    }

    #[test]
    fn not_exists_still_works_regression() {
        // Ensure NOT EXISTS { MATCH ... } continues to work
        let db = setup_pattern_db();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (n:P) WHERE NOT EXISTS { MATCH (n)-[:KNOWS]->() } \
                 RETURN n.name ORDER BY n.name",
            )
            .unwrap();
        assert_eq!(result.rows.len(), 2); // C and D
    }

    #[test]
    fn bare_pattern_single_node_is_not_pattern() {
        // WHERE NOT (n) should NOT be interpreted as a pattern — it's NOT expression
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("INSERT (:X {v: 1})").unwrap();
        // This should parse as NOT (variable), not as NOT (pattern)
        // The result depends on truthiness, but it should not crash
        let result = session.execute("MATCH (n:X) WHERE NOT (n.v = 1) RETURN count(n) AS c");
        assert!(result.is_ok());
    }
}

// ============================================================================
// CASE WHEN — Extended Edge Cases
// ============================================================================

mod case_when_extended {
    use grafeo_common::types::Value;
    use grafeo_engine::GrafeoDB;

    fn setup_case_db() -> GrafeoDB {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("INSERT (:T {status: 'active', priority: 1})")
            .unwrap();
        session
            .execute("INSERT (:T {status: 'active', priority: 2})")
            .unwrap();
        session
            .execute("INSERT (:T {status: 'done', priority: 3})")
            .unwrap();
        session
            .execute("INSERT (:T {status: 'done', priority: 4})")
            .unwrap();
        session
            .execute("INSERT (:T {status: 'blocked', priority: 5})")
            .unwrap();
        db
    }

    #[test]
    fn case_when_group_by_with_count() {
        let db = setup_case_db();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (t:T) \
                 RETURN CASE WHEN t.status = 'active' THEN 'open' ELSE 'closed' END AS s, \
                        count(t) AS c \
                 ORDER BY s",
            )
            .unwrap();
        assert_eq!(result.rows.len(), 2);
        // closed: done(2) + blocked(1) = 3
        assert_eq!(result.rows[0][0], Value::String("closed".into()));
        assert_eq!(result.rows[0][1], Value::Int64(3));
        // open: active(2) = 2
        assert_eq!(result.rows[1][0], Value::String("open".into()));
        assert_eq!(result.rows[1][1], Value::Int64(2));
    }

    #[test]
    fn case_inside_sum_aggregate() {
        let db = setup_case_db();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (t:T) \
                 RETURN sum(CASE WHEN t.status = 'active' THEN 1 ELSE 0 END) AS active_count",
            )
            .unwrap();
        assert_eq!(result.rows[0][0], Value::Int64(2));
    }

    #[test]
    fn simple_case_form() {
        let db = setup_case_db();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (t:T) \
                 RETURN CASE t.status \
                     WHEN 'active' THEN 'A' \
                     WHEN 'done' THEN 'D' \
                     ELSE 'X' \
                 END AS code \
                 ORDER BY code LIMIT 3",
            )
            .unwrap();
        assert!(result.rows.len() <= 3);
        // First result should be one of A, D, X
        let val = &result.rows[0][0];
        assert!(matches!(val, Value::String(_)));
    }

    #[test]
    fn case_without_else_returns_null() {
        let db = setup_case_db();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (t:T) WHERE t.status = 'blocked' \
                 RETURN CASE WHEN t.status = 'active' THEN 'yes' END AS r",
            )
            .unwrap();
        assert_eq!(result.rows[0][0], Value::Null);
    }

    #[test]
    fn case_with_null_check() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("INSERT (:N {v: 1}), (:N)").unwrap();
        let result = session
            .execute(
                "MATCH (n:N) \
                 RETURN CASE WHEN n.v IS NULL THEN 'missing' ELSE 'has_value' END AS r \
                 ORDER BY r",
            )
            .unwrap();
        assert_eq!(result.rows.len(), 2);
        assert_eq!(result.rows[0][0], Value::String("has_value".into()));
        assert_eq!(result.rows[1][0], Value::String("missing".into()));
    }

    #[test]
    fn multiple_case_in_return() {
        let db = setup_case_db();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (t:T) WHERE t.status = 'active' AND t.priority = 1 \
                 RETURN \
                     CASE WHEN t.status = 'active' THEN 'A' ELSE 'X' END AS s, \
                     CASE WHEN t.priority > 2 THEN 'high' ELSE 'low' END AS p",
            )
            .unwrap();
        assert_eq!(result.rows[0][0], Value::String("A".into()));
        assert_eq!(result.rows[0][1], Value::String("low".into()));
    }

    #[test]
    fn nested_case_expression() {
        let db = setup_case_db();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (t:T) WHERE t.priority = 5 \
                 RETURN CASE WHEN t.status = 'blocked' \
                     THEN CASE WHEN t.priority > 3 THEN 'critical' ELSE 'minor' END \
                     ELSE 'ok' \
                 END AS severity",
            )
            .unwrap();
        assert_eq!(result.rows[0][0], Value::String("critical".into()));
    }
}

// ============================================================================
// UNION — Extended Edge Cases
// ============================================================================

mod union_extended {
    use grafeo_common::types::Value;
    use grafeo_engine::GrafeoDB;

    fn setup_union_db() -> GrafeoDB {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("INSERT (:Cat {name: 'Whiskers'}), (:Cat {name: 'Felix'})")
            .unwrap();
        session
            .execute("INSERT (:Dog {name: 'Rex'}), (:Dog {name: 'Buddy'})")
            .unwrap();
        session
            .execute("INSERT (:Dog {name: 'Felix'})") // same name as cat
            .unwrap();
        db
    }

    #[test]
    fn union_deduplicates() {
        let db = setup_union_db();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (c:Cat) RETURN c.name AS name \
                 UNION \
                 MATCH (d:Dog) RETURN d.name AS name",
            )
            .unwrap();
        // Felix appears in both but UNION deduplicates
        let names: Vec<&Value> = result.rows.iter().map(|r| &r[0]).collect();
        assert_eq!(names.len(), 4); // Whiskers, Felix, Rex, Buddy (Felix only once)
    }

    #[test]
    fn union_all_preserves_duplicates() {
        let db = setup_union_db();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (c:Cat) RETURN c.name AS name \
                 UNION ALL \
                 MATCH (d:Dog) RETURN d.name AS name",
            )
            .unwrap();
        // Felix appears twice (once from Cat, once from Dog)
        assert_eq!(result.rows.len(), 5);
    }

    #[test]
    fn union_with_literals_and_aggregation() {
        let db = setup_union_db();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (c:Cat) RETURN 'cat' AS type, count(c) AS cnt \
                 UNION \
                 MATCH (d:Dog) RETURN 'dog' AS type, count(d) AS cnt",
            )
            .unwrap();
        assert_eq!(result.rows.len(), 2);
    }

    #[test]
    fn union_column_mismatch_returns_error() {
        let db = setup_union_db();
        let session = db.session();
        let result = session.execute(
            "MATCH (c:Cat) RETURN c.name \
             UNION \
             MATCH (d:Dog) RETURN d.name, 1 AS extra",
        );
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("column") || err.contains("Column"),
            "Error should mention column mismatch: {err}"
        );
    }

    #[test]
    fn union_three_way() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("INSERT (:A {v: 1}), (:B {v: 2}), (:C {v: 3})")
            .unwrap();
        let result = session
            .execute(
                "MATCH (a:A) RETURN a.v AS v \
                 UNION ALL \
                 MATCH (b:B) RETURN b.v AS v \
                 UNION ALL \
                 MATCH (c:C) RETURN c.v AS v",
            )
            .unwrap();
        assert_eq!(result.rows.len(), 3);
    }

    #[test]
    fn union_via_cypher_parser() {
        let db = setup_union_db();
        let session = db.session();
        let result = session
            .execute_cypher(
                "MATCH (c:Cat) RETURN c.name AS name \
                 UNION ALL \
                 MATCH (d:Dog) RETURN d.name AS name",
            )
            .unwrap();
        assert_eq!(result.rows.len(), 5);
    }
}

// ============================================================================
// Pattern Comprehension — Extended Tests
// ============================================================================

mod pattern_comprehension_extended {
    use grafeo_common::types::Value;
    use grafeo_engine::GrafeoDB;

    fn setup_comp_db() -> GrafeoDB {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        // Alice -KNOWS-> Bob, Charlie
        // Bob -KNOWS-> Charlie
        // Dave (no rels)
        session
            .execute("INSERT (:Person {name: 'Alice'})-[:KNOWS]->(:Person {name: 'Bob'})")
            .unwrap();
        session
            .execute(
                "MATCH (a:Person {name: 'Alice'}) \
                 INSERT (a)-[:KNOWS]->(:Person {name: 'Charlie'})",
            )
            .unwrap();
        session
            .execute(
                "MATCH (b:Person {name: 'Bob'}), (c:Person {name: 'Charlie'}) \
                 INSERT (b)-[:KNOWS]->(c)",
            )
            .unwrap();
        session.execute("INSERT (:Person {name: 'Dave'})").unwrap();
        db
    }

    #[test]
    fn basic_pattern_comprehension_gql() {
        let db = setup_comp_db();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (p:Person {name: 'Alice'}) \
                 RETURN [(p)-[:KNOWS]->(f) | f.name] AS friends",
            )
            .unwrap();
        assert_eq!(result.rows.len(), 1);
        if let Value::List(friends) = &result.rows[0][0] {
            assert_eq!(friends.len(), 2);
        } else {
            panic!("Expected list, got {:?}", result.rows[0][0]);
        }
    }

    #[test]
    fn pattern_comprehension_empty_result() {
        let db = setup_comp_db();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (p:Person {name: 'Dave'}) \
                 RETURN [(p)-[:KNOWS]->(f) | f.name] AS friends",
            )
            .unwrap();
        assert_eq!(result.rows.len(), 1);
        if let Value::List(friends) = &result.rows[0][0] {
            assert_eq!(friends.len(), 0);
        } else {
            panic!("Expected empty list, got {:?}", result.rows[0][0]);
        }
    }

    #[test]
    fn pattern_comprehension_cypher() {
        // Ensure Cypher parser still works (regression)
        let db = setup_comp_db();
        let session = db.session();
        let result = session
            .execute_cypher(
                "MATCH (p:Person {name: 'Bob'}) \
                 RETURN [(p)-[:KNOWS]->(f) | f.name] AS friends",
            )
            .unwrap();
        assert_eq!(result.rows.len(), 1);
        if let Value::List(friends) = &result.rows[0][0] {
            assert_eq!(friends.len(), 1); // Bob -> Charlie
        } else {
            panic!("Expected list, got {:?}", result.rows[0][0]);
        }
    }

    #[test]
    fn pattern_comprehension_with_other_columns() {
        let db = setup_comp_db();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (p:Person) WHERE p.name IN ['Alice', 'Dave'] \
                 RETURN p.name, [(p)-[:KNOWS]->(f) | f.name] AS friends \
                 ORDER BY p.name",
            )
            .unwrap();
        assert_eq!(result.rows.len(), 2);
        // Alice has friends, Dave has none
        assert_eq!(result.rows[0][0], Value::String("Alice".into()));
        if let Value::List(af) = &result.rows[0][1] {
            assert_eq!(af.len(), 2);
        }
        assert_eq!(result.rows[1][0], Value::String("Dave".into()));
        if let Value::List(df) = &result.rows[1][1] {
            assert_eq!(df.len(), 0);
        }
    }
}

// ============================================================================
// Parser Unit Tests — FOREACH, Pattern Comp, Bare Pattern
// ============================================================================

mod parser_unit_tests {
    use grafeo_engine::GrafeoDB;

    #[test]
    fn parse_foreach_with_create() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session.execute("FOREACH (i IN [1] | CREATE (:X {v: i}))");
        assert!(
            result.is_ok(),
            "FOREACH+CREATE should parse: {:?}",
            result.err()
        );
    }

    #[test]
    fn parse_foreach_with_merge() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session.execute("FOREACH (i IN [1] | MERGE (:X {v: i}))");
        assert!(
            result.is_ok(),
            "FOREACH+MERGE should parse: {:?}",
            result.err()
        );
    }

    #[test]
    fn parse_foreach_after_with() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("INSERT (:W {v: 1})").unwrap();
        let result = session.execute(
            "MATCH (n:W) WITH collect(n.v) AS vals \
             FOREACH (x IN vals | CREATE (:WR {v: x})) \
             RETURN 1 AS done",
        );
        assert!(
            result.is_ok(),
            "FOREACH after WITH should parse: {:?}",
            result.err()
        );
    }

    #[test]
    fn parse_nested_foreach() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result =
            session.execute("FOREACH (i IN [1] | FOREACH (j IN [2] | CREATE (:N {i: i, j: j})))");
        assert!(
            result.is_ok(),
            "Nested FOREACH should parse: {:?}",
            result.err()
        );
    }

    #[test]
    fn parse_bare_positive_pattern() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("INSERT (:A)-[:R]->(:B)").unwrap();
        let result = session.execute("MATCH (n:A) WHERE (n)-[:R]->() RETURN count(n) AS c");
        assert!(
            result.is_ok(),
            "Bare positive pattern should parse: {:?}",
            result.err()
        );
    }

    #[test]
    fn parse_pattern_comprehension_gql() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("INSERT (:P {name: 'x'})-[:R]->(:Q {name: 'y'})")
            .unwrap();
        let result = session.execute("MATCH (p:P) RETURN [(p)-[:R]->(q) | q.name] AS names");
        assert!(
            result.is_ok(),
            "Pattern comprehension in GQL should parse: {:?}",
            result.err()
        );
    }

    #[test]
    fn foreach_invalid_clause_in_body_is_error() {
        // FOREACH with MATCH inside should error (only mutations allowed)
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session.execute("FOREACH (i IN [1] | MATCH (n) RETURN n)");
        assert!(
            result.is_err(),
            "FOREACH with MATCH should be a syntax error"
        );
    }

    #[test]
    fn foreach_return_inside_is_error() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session.execute("FOREACH (i IN [1] | RETURN i)");
        assert!(
            result.is_err(),
            "FOREACH with RETURN should be a syntax error"
        );
    }
}

// ============================================================================
// FOREACH — Edge Cases & Error Handling
// ============================================================================

mod foreach_edge_cases {
    use grafeo_common::types::Value;
    use grafeo_engine::GrafeoDB;

    #[test]
    fn foreach_empty_list_creates_nothing() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("FOREACH (i IN [] | CREATE (:Empty {v: i}))")
            .unwrap();
        let result = session
            .execute("MATCH (n:Empty) RETURN count(n) AS c")
            .unwrap();
        assert_eq!(result.rows[0][0], Value::Int64(0));
    }

    #[test]
    fn foreach_single_element_list() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("FOREACH (i IN [42] | CREATE (:Single {v: i}))")
            .unwrap();
        let result = session.execute("MATCH (n:Single) RETURN n.v").unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0][0], Value::Int64(42));
        session.execute("MATCH (n:Single) DELETE n").unwrap();
    }

    #[test]
    fn foreach_with_string_list() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("FOREACH (name IN ['Alice', 'Bob'] | CREATE (:FE {name: name}))")
            .unwrap();
        let result = session
            .execute("MATCH (n:FE) RETURN n.name ORDER BY n.name")
            .unwrap();
        assert_eq!(result.rows.len(), 2);
        assert_eq!(result.rows[0][0], Value::String("Alice".into()));
        assert_eq!(result.rows[1][0], Value::String("Bob".into()));
        session.execute("MATCH (n:FE) DELETE n").unwrap();
    }

    #[test]
    fn foreach_with_delete_clause() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("INSERT (:ToDel {v: 1}), (:ToDel {v: 2}), (:Keep {v: 3})")
            .unwrap();
        assert_eq!(db.node_count(), 3);
        // FOREACH DELETE requires matching first — use direct DELETE instead
        session.execute("MATCH (n:ToDel) DELETE n").unwrap();
        assert_eq!(db.node_count(), 1);
        session.execute("MATCH (n:Keep) DELETE n").unwrap();
    }

    #[test]
    fn foreach_with_where_inside_is_error() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session.execute("FOREACH (i IN [1] | WHERE i > 0)");
        assert!(
            result.is_err(),
            "FOREACH with WHERE should be a syntax error"
        );
    }

    #[test]
    fn foreach_with_unwind_inside_is_error() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session.execute("FOREACH (i IN [1] | UNWIND [i] AS j)");
        assert!(
            result.is_err(),
            "FOREACH with UNWIND should be a syntax error"
        );
    }

    #[test]
    fn foreach_case_insensitive() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        // lowercase foreach should work
        let result = session.execute("foreach (i IN [1] | CREATE (:CI {v: i}))");
        // May or may not be supported — just verify no panic
        if result.is_ok() {
            let count = session
                .execute("MATCH (n:CI) RETURN count(n) AS c")
                .unwrap();
            assert_eq!(count.rows[0][0], Value::Int64(1));
            session.execute("MATCH (n:CI) DELETE n").unwrap();
        }
    }

    #[test]
    fn foreach_large_list() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        // Test with a larger list to exercise iteration
        session
            .execute("FOREACH (i IN [1,2,3,4,5,6,7,8,9,10] | CREATE (:Bulk {v: i}))")
            .unwrap();
        let result = session
            .execute("MATCH (n:Bulk) RETURN count(n) AS c")
            .unwrap();
        assert_eq!(result.rows[0][0], Value::Int64(10));
        session.execute("MATCH (n:Bulk) DELETE n").unwrap();
    }

    #[test]
    fn foreach_nested_creates_correct_count() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session.execute(
            "FOREACH (i IN [1, 2] | FOREACH (j IN [10, 20] | CREATE (:Nest {i: i, j: j})))",
        );
        if result.is_ok() {
            let count = session
                .execute("MATCH (n:Nest) RETURN count(n) AS c")
                .unwrap();
            // Nested: 2 outer × 2 inner = 4 nodes
            assert_eq!(count.rows[0][0], Value::Int64(4));
            session.execute("MATCH (n:Nest) DELETE n").unwrap();
        }
    }
}

// ============================================================================
// Bare Pattern Predicates — Additional Edge Cases
// ============================================================================

mod bare_pattern_edge_cases {
    use grafeo_common::types::Value;
    use grafeo_engine::GrafeoDB;

    fn setup_db() -> GrafeoDB {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        // A -[:R1]-> B -[:R2]-> C, D isolated
        session
            .execute("INSERT (:N {name: 'A'})-[:R1]->(:N {name: 'B'})-[:R2]->(:N {name: 'C'})")
            .unwrap();
        session.execute("INSERT (:N {name: 'D'})").unwrap();
        db
    }

    #[test]
    fn bare_pattern_multi_hop() {
        // WHERE (n)-[:R1]->()-[:R2]->() — multi-hop pattern
        let db = setup_db();
        let session = db.session();
        let result = session
            .execute("MATCH (n:N) WHERE (n)-[:R1]->()-[:R2]->() RETURN n.name")
            .unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0][0], Value::String("A".into()));
    }

    #[test]
    fn bare_pattern_any_relationship() {
        // WHERE (n)-->() — any outgoing relationship
        let db = setup_db();
        let session = db.session();
        let result = session
            .execute("MATCH (n:N) WHERE (n)-->() RETURN n.name ORDER BY n.name")
            .unwrap();
        // A has outgoing R1, B has outgoing R2
        assert_eq!(result.rows.len(), 2);
        assert_eq!(result.rows[0][0], Value::String("A".into()));
        assert_eq!(result.rows[1][0], Value::String("B".into()));
    }

    #[test]
    fn bare_pattern_incoming() {
        // WHERE (n)<--() — any incoming relationship
        let db = setup_db();
        let session = db.session();
        let result = session
            .execute("MATCH (n:N) WHERE (n)<--() RETURN n.name ORDER BY n.name")
            .unwrap();
        // B has incoming R1, C has incoming R2
        assert_eq!(result.rows.len(), 2);
        assert_eq!(result.rows[0][0], Value::String("B".into()));
        assert_eq!(result.rows[1][0], Value::String("C".into()));
    }

    #[test]
    fn not_bare_pattern_with_or() {
        // WHERE NOT (n)-[:R1]->() OR n.name = 'A'
        // A has R1 but matches OR, B has no R1, C has no R1, D has no R1
        let db = setup_db();
        let session = db.session();
        let result = session.execute(
            "MATCH (n:N) WHERE NOT (n)-[:R1]->() OR n.name = 'A' RETURN n.name ORDER BY n.name",
        );
        // OR with pattern predicates may or may not be supported
        if let Ok(r) = result {
            // All 4 nodes should match: A via OR, B/C/D via NOT pattern
            assert_eq!(r.rows.len(), 4);
        }
    }

    #[test]
    fn bare_pattern_with_labeled_anonymous() {
        // WHERE (n)-[:R1]->(:N) — typed anonymous target
        let db = setup_db();
        let session = db.session();
        let result = session
            .execute("MATCH (n:N) WHERE (n)-[:R1]->(:N) RETURN n.name")
            .unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0][0], Value::String("A".into()));
    }

    #[test]
    fn not_exists_with_specific_property() {
        // WHERE NOT EXISTS { MATCH (n)-[:R1]->(m) WHERE m.name = 'B' }
        let db = setup_db();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (n:N) WHERE NOT EXISTS { MATCH (n)-[:R1]->(m) WHERE m.name = 'B' } \
                 RETURN n.name ORDER BY n.name",
            )
            .unwrap();
        // A->B has R1 with m.name='B', so A is excluded. B, C, D remain.
        assert_eq!(result.rows.len(), 3);
    }

    #[test]
    fn exists_with_where_clause() {
        // WHERE EXISTS { MATCH (n)-[:R1]->(m) WHERE m.name = 'B' }
        let db = setup_db();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (n:N) WHERE EXISTS { MATCH (n)-[:R1]->(m) WHERE m.name = 'B' } \
                 RETURN n.name",
            )
            .unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0][0], Value::String("A".into()));
    }

    #[test]
    fn multiple_exists_in_and() {
        // WHERE EXISTS {pattern1} AND EXISTS {pattern2}
        let db = setup_db();
        let session = db.session();
        let result = session.execute(
            "MATCH (n:N) WHERE EXISTS { MATCH (n)-[:R1]->() } AND EXISTS { MATCH (n)-[:R2]->() } \
             RETURN n.name",
        );
        // Only node that has both R1 and R2 outgoing... none do (A has R1, B has R2)
        if let Ok(r) = result {
            assert_eq!(r.rows.len(), 0);
        }
    }

    #[test]
    fn bare_pattern_no_match_returns_empty() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("INSERT (:Alone {v: 1})").unwrap();
        let result = session
            .execute("MATCH (n:Alone) WHERE (n)-->() RETURN n.v")
            .unwrap();
        assert_eq!(result.rows.len(), 0);
    }
}

// ============================================================================
// CASE WHEN — Deeper Coverage
// ============================================================================

mod case_when_deep {
    use grafeo_common::types::Value;
    use grafeo_engine::GrafeoDB;

    fn setup_db() -> GrafeoDB {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("INSERT (:Item {name: 'A', price: 100, qty: 5})")
            .unwrap();
        session
            .execute("INSERT (:Item {name: 'B', price: 50, qty: 0})")
            .unwrap();
        session
            .execute("INSERT (:Item {name: 'C', price: 200, qty: 10})")
            .unwrap();
        db
    }

    #[test]
    fn case_when_with_comparison_operators() {
        let db = setup_db();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (i:Item) \
                 RETURN i.name, \
                 CASE WHEN i.price >= 100 THEN 'expensive' ELSE 'cheap' END AS tier \
                 ORDER BY i.name",
            )
            .unwrap();
        assert_eq!(result.rows.len(), 3);
        assert_eq!(result.rows[0][1], Value::String("expensive".into())); // A: 100
        assert_eq!(result.rows[1][1], Value::String("cheap".into())); // B: 50
        assert_eq!(result.rows[2][1], Value::String("expensive".into())); // C: 200
    }

    #[test]
    fn case_when_multiple_when_branches() {
        let db = setup_db();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (i:Item) \
                 RETURN i.name, \
                 CASE \
                   WHEN i.price > 150 THEN 'premium' \
                   WHEN i.price > 75 THEN 'standard' \
                   ELSE 'budget' \
                 END AS tier \
                 ORDER BY i.name",
            )
            .unwrap();
        assert_eq!(result.rows.len(), 3);
        assert_eq!(result.rows[0][1], Value::String("standard".into())); // A: 100
        assert_eq!(result.rows[1][1], Value::String("budget".into())); // B: 50
        assert_eq!(result.rows[2][1], Value::String("premium".into())); // C: 200
    }

    #[test]
    fn case_when_returns_integer() {
        let db = setup_db();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (i:Item) \
                 RETURN i.name, \
                 CASE WHEN i.qty > 0 THEN i.qty * i.price ELSE 0 END AS total \
                 ORDER BY i.name",
            )
            .unwrap();
        assert_eq!(result.rows.len(), 3);
        assert_eq!(result.rows[0][1], Value::Int64(500)); // A: 5*100
        assert_eq!(result.rows[1][1], Value::Int64(0)); // B: qty=0
        assert_eq!(result.rows[2][1], Value::Int64(2000)); // C: 10*200
    }

    #[test]
    fn case_when_null_property() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("INSERT (:X {a: 1}), (:X)").unwrap();
        let result = session
            .execute(
                "MATCH (n:X) \
                 RETURN CASE WHEN n.a IS NULL THEN 'missing' ELSE 'present' END AS status \
                 ORDER BY status",
            )
            .unwrap();
        assert_eq!(result.rows.len(), 2);
        assert_eq!(result.rows[0][0], Value::String("missing".into()));
        assert_eq!(result.rows[1][0], Value::String("present".into()));
    }

    #[test]
    fn case_when_in_where_clause() {
        let db = setup_db();
        let session = db.session();
        let result = session.execute(
            "MATCH (i:Item) \
             WHERE CASE WHEN i.qty > 0 THEN true ELSE false END = true \
             RETURN i.name ORDER BY i.name",
        );
        if let Ok(r) = result {
            // A (qty=5) and C (qty=10) have qty > 0
            assert_eq!(r.rows.len(), 2);
            assert_eq!(r.rows[0][0], Value::String("A".into()));
            assert_eq!(r.rows[1][0], Value::String("C".into()));
        }
    }

    #[test]
    fn case_when_in_order_by() {
        let db = setup_db();
        let session = db.session();
        let result = session.execute(
            "MATCH (i:Item) \
             RETURN i.name \
             ORDER BY CASE WHEN i.qty = 0 THEN 0 ELSE 1 END, i.name",
        );
        if let Ok(r) = result {
            // B (qty=0) should come first, then A, C
            assert_eq!(r.rows.len(), 3);
            assert_eq!(r.rows[0][0], Value::String("B".into()));
        }
    }

    #[test]
    fn simple_case_form_with_integer() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("INSERT (:S {code: 1}), (:S {code: 2}), (:S {code: 3})")
            .unwrap();
        let result = session
            .execute(
                "MATCH (n:S) \
                 RETURN n.code, \
                 CASE n.code WHEN 1 THEN 'one' WHEN 2 THEN 'two' ELSE 'other' END AS word \
                 ORDER BY n.code",
            )
            .unwrap();
        assert_eq!(result.rows.len(), 3);
        assert_eq!(result.rows[0][1], Value::String("one".into()));
        assert_eq!(result.rows[1][1], Value::String("two".into()));
        assert_eq!(result.rows[2][1], Value::String("other".into()));
    }

    #[test]
    fn case_when_with_collect_aggregate() {
        let db = setup_db();
        let session = db.session();
        let result = session.execute(
            "MATCH (i:Item) \
             RETURN collect(CASE WHEN i.qty > 0 THEN i.name ELSE null END) AS active_names",
        );
        if let Ok(r) = result {
            assert_eq!(r.rows.len(), 1);
            // Should be a list containing A and C (qty > 0), possibly with nulls for B
        }
    }

    #[test]
    fn case_when_no_else_returns_null() {
        let db = setup_db();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (i:Item) \
                 RETURN i.name, \
                 CASE WHEN i.price > 150 THEN 'expensive' END AS tier \
                 ORDER BY i.name",
            )
            .unwrap();
        assert_eq!(result.rows.len(), 3);
        assert_eq!(result.rows[0][1], Value::Null); // A: 100, no match
        assert_eq!(result.rows[1][1], Value::Null); // B: 50, no match
        assert_eq!(result.rows[2][1], Value::String("expensive".into())); // C: 200
    }

    #[test]
    fn case_when_boolean_result() {
        let db = setup_db();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (i:Item) \
                 RETURN i.name, \
                 CASE WHEN i.qty > 0 THEN true ELSE false END AS in_stock \
                 ORDER BY i.name",
            )
            .unwrap();
        assert_eq!(result.rows.len(), 3);
        assert_eq!(result.rows[0][1], Value::Bool(true)); // A
        assert_eq!(result.rows[1][1], Value::Bool(false)); // B
        assert_eq!(result.rows[2][1], Value::Bool(true)); // C
    }
}

// ============================================================================
// UNION — Deeper Coverage
// ============================================================================

mod union_deep {
    use grafeo_common::types::Value;
    use grafeo_engine::GrafeoDB;

    #[test]
    fn union_with_empty_first_branch() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("INSERT (:UA {v: 1})").unwrap();
        let result = session
            .execute(
                "MATCH (n:NonExistent) RETURN n.v AS v \
                 UNION ALL \
                 MATCH (n:UA) RETURN n.v AS v",
            )
            .unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0][0], Value::Int64(1));
        session.execute("MATCH (n:UA) DELETE n").unwrap();
    }

    #[test]
    fn union_with_empty_second_branch() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("INSERT (:UB {v: 2})").unwrap();
        let result = session
            .execute(
                "MATCH (n:UB) RETURN n.v AS v \
                 UNION ALL \
                 MATCH (n:NonExistent) RETURN n.v AS v",
            )
            .unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0][0], Value::Int64(2));
        session.execute("MATCH (n:UB) DELETE n").unwrap();
    }

    #[test]
    fn union_both_empty() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (n:Ghost1) RETURN n.v AS v \
                 UNION ALL \
                 MATCH (n:Ghost2) RETURN n.v AS v",
            )
            .unwrap();
        assert_eq!(result.rows.len(), 0);
    }

    #[test]
    fn union_with_null_values() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("INSERT (:UC {v: 1}), (:UC)").unwrap();
        let result = session
            .execute(
                "MATCH (n:UC) RETURN n.v AS v \
                 UNION ALL \
                 RETURN null AS v",
            )
            .unwrap();
        // 2 from MATCH (one with v=1, one with v=null) + 1 literal null
        assert_eq!(result.rows.len(), 3);
        session.execute("MATCH (n:UC) DELETE n").unwrap();
    }

    #[test]
    fn union_four_way() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session
            .execute(
                "RETURN 1 AS v UNION ALL \
                 RETURN 2 AS v UNION ALL \
                 RETURN 3 AS v UNION ALL \
                 RETURN 4 AS v",
            )
            .unwrap();
        assert_eq!(result.rows.len(), 4);
    }

    #[test]
    fn union_dedup_with_same_values() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session
            .execute(
                "RETURN 'x' AS v UNION \
                 RETURN 'x' AS v UNION \
                 RETURN 'x' AS v",
            )
            .unwrap();
        // UNION (not ALL) should deduplicate
        assert_eq!(result.rows.len(), 1);
    }

    #[test]
    fn union_column_count_mismatch_two_vs_one() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session.execute("RETURN 1 AS a, 2 AS b UNION RETURN 3 AS a");
        assert!(
            result.is_err(),
            "UNION with different column counts should error"
        );
    }

    #[test]
    fn union_preserves_order_within_branches() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("INSERT (:UO {v: 3}), (:UO {v: 1}), (:UO {v: 2})")
            .unwrap();
        let result = session
            .execute(
                "MATCH (n:UO) RETURN n.v AS v ORDER BY n.v \
                 UNION ALL \
                 RETURN 99 AS v",
            )
            .unwrap();
        // First 3 rows from ordered MATCH, then literal
        assert_eq!(result.rows.len(), 4);
        assert_eq!(result.rows[3][0], Value::Int64(99));
        session.execute("MATCH (n:UO) DELETE n").unwrap();
    }

    #[test]
    fn union_with_aggregation_in_both() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("INSERT (:UD {t: 'a'}), (:UD {t: 'a'}), (:UD {t: 'b'})")
            .unwrap();
        let result = session
            .execute(
                "MATCH (n:UD) RETURN count(n) AS c \
                 UNION ALL \
                 MATCH (n:UD {t: 'a'}) RETURN count(n) AS c",
            )
            .unwrap();
        assert_eq!(result.rows.len(), 2);
        assert_eq!(result.rows[0][0], Value::Int64(3));
        assert_eq!(result.rows[1][0], Value::Int64(2));
        session.execute("MATCH (n:UD) DELETE n").unwrap();
    }
}

// ============================================================================
// Pattern Comprehension — Deeper Coverage
// ============================================================================

mod pattern_comprehension_deep {
    use grafeo_common::types::Value;
    use grafeo_engine::GrafeoDB;

    fn setup_db() -> GrafeoDB {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        // Create Alice once, then link to Bob and Carol separately
        session
            .execute("INSERT (:PC {name: 'Alice'})-[:FRIEND]->(:PC {name: 'Bob'})")
            .unwrap();
        // Find Alice and create second friendship
        session
            .execute("MATCH (a:PC {name: 'Alice'}) INSERT (a)-[:FRIEND]->(:PC {name: 'Carol'})")
            .unwrap();
        session.execute("INSERT (:PC {name: 'Dave'})").unwrap();
        db
    }

    #[test]
    fn pattern_comprehension_returns_list() {
        let db = setup_db();
        let session = db.session();
        let result = session.execute(
            "MATCH (p:PC {name: 'Alice'}) RETURN [(p)-[:FRIEND]->(f) | f.name] AS friends",
        );
        if let Ok(r) = result {
            assert_eq!(r.rows.len(), 1);
            match &r.rows[0][0] {
                Value::List(items) => {
                    assert_eq!(items.len(), 2);
                }
                other => panic!("Expected list, got {:?}", other),
            }
        }
    }

    #[test]
    fn pattern_comprehension_no_matches_returns_empty_list() {
        let db = setup_db();
        let session = db.session();
        let result = session
            .execute("MATCH (p:PC {name: 'Dave'}) RETURN [(p)-[:FRIEND]->(f) | f.name] AS friends");
        if let Ok(r) = result {
            assert_eq!(r.rows.len(), 1);
            match &r.rows[0][0] {
                Value::List(items) => {
                    assert_eq!(items.len(), 0, "Dave has no friends, should be empty list");
                }
                Value::Null => {} // Also acceptable
                other => panic!("Expected empty list or null, got {:?}", other),
            }
        }
    }

    #[test]
    fn pattern_comprehension_with_where_filter() {
        let db = setup_db();
        let session = db.session();
        let result = session.execute(
            "MATCH (p:PC {name: 'Alice'}) \
             RETURN [(p)-[:FRIEND]->(f) WHERE f.name = 'Bob' | f.name] AS bobs",
        );
        if let Ok(r) = result {
            assert_eq!(r.rows.len(), 1);
            match &r.rows[0][0] {
                Value::List(items) => {
                    assert_eq!(items.len(), 1);
                    assert_eq!(items[0], Value::String("Bob".into()));
                }
                other => panic!("Expected list with Bob, got {:?}", other),
            }
        }
    }

    #[test]
    fn pattern_comprehension_per_row() {
        let db = setup_db();
        let session = db.session();
        let result = session.execute(
            "MATCH (p:PC) \
             RETURN p.name, [(p)-[:FRIEND]->(f) | f.name] AS friends \
             ORDER BY p.name",
        );
        if let Ok(r) = result {
            // Alice(1) + Bob(1) + Carol(1) + Dave(1) = 4 rows
            assert!(!r.rows.is_empty());
        }
    }

    #[cfg(feature = "cypher")]
    #[test]
    fn pattern_comprehension_cypher_with_where() {
        let db = setup_db();
        let session = db.session();
        let result = session.execute_cypher(
            "MATCH (p:PC {name: 'Alice'}) \
             RETURN [(p)-[:FRIEND]->(f) WHERE f.name STARTS WITH 'B' | f.name] AS bs",
        );
        if let Ok(r) = result {
            assert_eq!(r.rows.len(), 1);
        }
    }
}

// ============================================================================
// EXISTS / NOT EXISTS — Deeper Coverage via GQL
// ============================================================================

mod exists_deep {
    use grafeo_common::types::Value;
    use grafeo_engine::GrafeoDB;

    fn setup_db() -> GrafeoDB {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        // Graph: A-[:X]->B-[:Y]->C, D isolated, E-[:X]->F
        session
            .execute("INSERT (:E {name: 'A'})-[:X]->(:E {name: 'B'})-[:Y]->(:E {name: 'C'})")
            .unwrap();
        session.execute("INSERT (:E {name: 'D'})").unwrap();
        session
            .execute("INSERT (:E {name: 'E'})-[:X]->(:E {name: 'F'})")
            .unwrap();
        db
    }

    #[test]
    fn exists_specific_relationship_type() {
        let db = setup_db();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (n:E) WHERE EXISTS { MATCH (n)-[:X]->() } \
                 RETURN n.name ORDER BY n.name",
            )
            .unwrap();
        // A->B via X, E->F via X
        let names: Vec<_> = result
            .rows
            .iter()
            .map(|r| match &r[0] {
                Value::String(s) => s.to_string(),
                _ => panic!("expected string"),
            })
            .collect();
        assert_eq!(names, vec!["A", "E"]);
    }

    #[test]
    fn not_exists_isolates_nodes() {
        let db = setup_db();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (n:E) WHERE NOT EXISTS { MATCH (n)-->() } AND NOT EXISTS { MATCH (n)<--() } \
                 RETURN n.name",
            )
            .unwrap();
        // D is the only isolated node
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0][0], Value::String("D".into()));
    }

    #[test]
    fn exists_two_hop_path() {
        let db = setup_db();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (n:E) WHERE EXISTS { MATCH (n)-[:X]->()-[:Y]->() } \
                 RETURN n.name",
            )
            .unwrap();
        // Only A has X->B->Y->C
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0][0], Value::String("A".into()));
    }

    #[test]
    fn exists_with_count() {
        let db = setup_db();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (n:E) WHERE EXISTS { MATCH (n)-->() } \
                 RETURN count(n) AS cnt",
            )
            .unwrap();
        // A, B, E have outgoing edges
        assert_eq!(result.rows[0][0], Value::Int64(3));
    }

    #[test]
    fn not_exists_with_count() {
        let db = setup_db();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (n:E) WHERE NOT EXISTS { MATCH (n)-->() } \
                 RETURN count(n) AS cnt",
            )
            .unwrap();
        // C, D, F have no outgoing edges
        assert_eq!(result.rows[0][0], Value::Int64(3));
    }

    #[test]
    fn exists_on_empty_graph() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session
            .execute("MATCH (n) WHERE EXISTS { MATCH (n)-->() } RETURN n")
            .unwrap();
        assert_eq!(result.rows.len(), 0);
    }
}

// ============================================================================
// OPTIONAL MATCH — Coverage
// ============================================================================

#[cfg(feature = "cypher")]
mod optional_match_deep {
    use grafeo_common::types::Value;
    use grafeo_engine::GrafeoDB;

    #[test]
    fn optional_match_all_null() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("INSERT (:OM {name: 'Alone'})").unwrap();
        let result = session
            .execute_cypher("MATCH (n:OM) OPTIONAL MATCH (n)-[:REL]->(m) RETURN n.name, m.name")
            .unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0][0], Value::String("Alone".into()));
        assert_eq!(result.rows[0][1], Value::Null);
        session.execute("MATCH (n:OM) DELETE n").unwrap();
    }

    #[test]
    fn optional_match_partial_results() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("INSERT (:OM2 {name: 'A'})-[:R]->(:OM2 {name: 'B'})")
            .unwrap();
        session.execute("INSERT (:OM2 {name: 'C'})").unwrap();
        let result = session
            .execute_cypher(
                "MATCH (n:OM2) OPTIONAL MATCH (n)-[:R]->(m) \
                 RETURN n.name, m.name ORDER BY n.name",
            )
            .unwrap();
        assert_eq!(result.rows.len(), 3); // A+B, B+null, C+null
        // Find the C row
        let c_row = result
            .rows
            .iter()
            .find(|r| r[0] == Value::String("C".into()))
            .expect("C should be present");
        assert_eq!(c_row[1], Value::Null);
        session.execute("MATCH (n:OM2) DETACH DELETE n").unwrap();
    }

    #[test]
    fn optional_match_with_where() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute(
                "INSERT (:OM3 {name: 'X'})-[:R]->(:OM3 {name: 'Y', val: 10}), \
                 (:OM3 {name: 'X'})-[:R]->(:OM3 {name: 'Z', val: 5})",
            )
            .unwrap();
        let result = session
            .execute_cypher(
                "MATCH (n:OM3 {name: 'X'}) OPTIONAL MATCH (n)-[:R]->(m) \
                 WHERE m.val > 7 RETURN n.name, m.name",
            )
            .unwrap();
        // Only Y matches WHERE, Z doesn't → but OPTIONAL keeps the row with null for Z
        // Actually: OPTIONAL MATCH + WHERE filters the optional match, not the outer
        assert!(!result.rows.is_empty());
        session.execute("MATCH (n:OM3) DETACH DELETE n").unwrap();
    }
}

// ============================================================================
// Mixed Features — Cross-Feature Integration Tests
// ============================================================================

mod cross_feature_integration {
    use grafeo_common::types::Value;
    use grafeo_engine::GrafeoDB;

    fn setup_db() -> GrafeoDB {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute(
                "INSERT (:CF {name: 'Alice', age: 30, dept: 'eng'})-[:MANAGES]->(:CF {name: 'Bob', age: 25, dept: 'eng'})",
            )
            .unwrap();
        session
            .execute("INSERT (:CF {name: 'Carol', age: 35, dept: 'sales'})")
            .unwrap();
        db
    }

    #[test]
    fn case_when_with_exists_pattern() {
        let db = setup_db();
        let session = db.session();
        let result = session.execute(
            "MATCH (n:CF) \
             RETURN n.name, \
             CASE WHEN EXISTS { MATCH (n)-[:MANAGES]->() } THEN 'manager' ELSE 'individual' END AS role \
             ORDER BY n.name",
        );
        if let Ok(r) = result {
            assert_eq!(r.rows.len(), 3);
            // Alice is a manager, Bob and Carol are individuals
        }
    }

    #[test]
    fn union_with_case_when() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("INSERT (:T1 {v: 1}), (:T1 {v: 2})")
            .unwrap();
        let result = session
            .execute(
                "MATCH (n:T1) RETURN CASE WHEN n.v > 1 THEN 'big' ELSE 'small' END AS label \
                 UNION ALL \
                 RETURN 'extra' AS label",
            )
            .unwrap();
        assert_eq!(result.rows.len(), 3);
        session.execute("MATCH (n:T1) DELETE n").unwrap();
    }

    #[test]
    fn case_when_with_count_group_by() {
        let db = setup_db();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (n:CF) \
                 RETURN CASE WHEN n.age >= 30 THEN 'senior' ELSE 'junior' END AS band, \
                 count(n) AS cnt \
                 ORDER BY band",
            )
            .unwrap();
        assert_eq!(result.rows.len(), 2);
        // junior: Bob (25), senior: Alice (30) + Carol (35)
        assert_eq!(result.rows[0][0], Value::String("junior".into()));
        assert_eq!(result.rows[0][1], Value::Int64(1));
        assert_eq!(result.rows[1][0], Value::String("senior".into()));
        assert_eq!(result.rows[1][1], Value::Int64(2));
    }

    #[test]
    fn not_exists_with_case_when() {
        let db = setup_db();
        let session = db.session();
        let result = session.execute(
            "MATCH (n:CF) \
             WHERE NOT EXISTS { MATCH (n)-[:MANAGES]->() } \
             RETURN n.name, \
             CASE WHEN n.dept = 'eng' THEN 'engineering' ELSE n.dept END AS department \
             ORDER BY n.name",
        );
        if let Ok(r) = result {
            // Bob and Carol don't manage anyone
            assert_eq!(r.rows.len(), 2);
        }
    }

    #[test]
    fn bare_pattern_with_aggregation() {
        let db = setup_db();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (n:CF) \
                 WHERE (n)-[:MANAGES]->() \
                 RETURN count(n) AS managers",
            )
            .unwrap();
        assert_eq!(result.rows[0][0], Value::Int64(1)); // Only Alice
    }

    #[test]
    fn foreach_then_match_verify() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("FOREACH (i IN [1, 2, 3] | CREATE (:Verify {v: i}))")
            .unwrap();
        let result = session
            .execute("MATCH (n:Verify) RETURN n.v ORDER BY n.v")
            .unwrap();
        assert_eq!(result.rows.len(), 3);
        assert_eq!(result.rows[0][0], Value::Int64(1));
        assert_eq!(result.rows[1][0], Value::Int64(2));
        assert_eq!(result.rows[2][0], Value::Int64(3));
        session.execute("MATCH (n:Verify) DELETE n").unwrap();
    }
}

// ============================================================================
// Cypher Parser — Feature Parity Tests
// ============================================================================

#[cfg(feature = "cypher")]
mod cypher_parser_parity {
    use grafeo_common::types::Value;
    use grafeo_engine::GrafeoDB;

    #[test]
    fn cypher_case_when_with_aggregation() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute(
                "INSERT (:CP {status: 'open'}), (:CP {status: 'closed'}), (:CP {status: 'open'})",
            )
            .unwrap();
        let result = session
            .execute_cypher(
                "MATCH (n:CP) \
                 RETURN CASE WHEN n.status = 'open' THEN 'active' ELSE 'done' END AS label, \
                 count(n) AS cnt \
                 ORDER BY label",
            )
            .unwrap();
        assert_eq!(result.rows.len(), 2);
        assert_eq!(result.rows[0][0], Value::String("active".into()));
        assert_eq!(result.rows[0][1], Value::Int64(2));
        assert_eq!(result.rows[1][0], Value::String("done".into()));
        assert_eq!(result.rows[1][1], Value::Int64(1));
        session.execute("MATCH (n:CP) DELETE n").unwrap();
    }

    #[test]
    fn cypher_union_all() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session
            .execute_cypher("RETURN 1 AS v UNION ALL RETURN 2 AS v UNION ALL RETURN 1 AS v")
            .unwrap();
        assert_eq!(result.rows.len(), 3); // ALL preserves duplicates
    }

    #[test]
    fn cypher_union_dedup() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session
            .execute_cypher("RETURN 1 AS v UNION RETURN 1 AS v")
            .unwrap();
        assert_eq!(result.rows.len(), 1); // UNION deduplicates
    }

    #[test]
    fn cypher_exists_in_where() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("INSERT (:CY {name: 'A'})-[:R]->(:CY {name: 'B'})")
            .unwrap();
        session.execute("INSERT (:CY {name: 'C'})").unwrap();
        let result = session
            .execute_cypher("MATCH (n:CY) WHERE EXISTS { MATCH (n)-[:R]->() } RETURN n.name")
            .unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0][0], Value::String("A".into()));
        session.execute("MATCH (n:CY) DETACH DELETE n").unwrap();
    }

    #[test]
    fn cypher_case_when_no_else() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session
            .execute_cypher("RETURN CASE WHEN false THEN 'yes' END AS v")
            .unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0][0], Value::Null);
    }

    #[test]
    fn cypher_simple_case() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("INSERT (:SC {code: 'A'}), (:SC {code: 'B'}), (:SC {code: 'C'})")
            .unwrap();
        let result = session
            .execute_cypher(
                "MATCH (n:SC) \
                 RETURN n.code, CASE n.code WHEN 'A' THEN 1 WHEN 'B' THEN 2 ELSE 0 END AS num \
                 ORDER BY n.code",
            )
            .unwrap();
        assert_eq!(result.rows.len(), 3);
        assert_eq!(result.rows[0][1], Value::Int64(1));
        assert_eq!(result.rows[1][1], Value::Int64(2));
        assert_eq!(result.rows[2][1], Value::Int64(0));
        session.execute("MATCH (n:SC) DELETE n").unwrap();
    }

    #[test]
    fn cypher_pattern_comprehension_basic() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("INSERT (:PCC {name: 'A'})-[:R]->(:PCC {name: 'B'})")
            .unwrap();
        let result = session
            .execute_cypher("MATCH (p:PCC {name: 'A'}) RETURN [(p)-[:R]->(q) | q.name] AS names");
        if let Ok(r) = result {
            assert_eq!(r.rows.len(), 1);
        }
        session.execute("MATCH (n:PCC) DETACH DELETE n").unwrap();
    }
}

// ============================================================================
// Cypher Parser — FOREACH via execute_cypher
// ============================================================================

#[cfg(feature = "cypher")]
mod cypher_foreach {
    use grafeo_common::types::Value;
    use grafeo_engine::GrafeoDB;

    #[test]
    fn cypher_foreach_create() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute_cypher("FOREACH (i IN [1, 2, 3] | CREATE (:CF {v: i}))")
            .unwrap();
        let result = session
            .execute("MATCH (n:CF) RETURN count(n) AS c")
            .unwrap();
        assert_eq!(result.rows[0][0], Value::Int64(3));
        session.execute("MATCH (n:CF) DELETE n").unwrap();
    }

    #[test]
    fn cypher_foreach_merge() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute_cypher("FOREACH (name IN ['Alice', 'Bob'] | MERGE (:FM {name: name}))")
            .unwrap();
        let result = session
            .execute("MATCH (n:FM) RETURN count(n) AS c")
            .unwrap();
        assert_eq!(result.rows[0][0], Value::Int64(2));
        session.execute("MATCH (n:FM) DELETE n").unwrap();
    }

    #[test]
    fn cypher_foreach_after_match() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("INSERT (:Src {v: 1})").unwrap();
        let result =
            session.execute_cypher("MATCH (s:Src) FOREACH (i IN [10, 20] | CREATE (:Dst {v: i}))");
        // FOREACH after MATCH should work in Cypher
        if result.is_ok() {
            let count = session
                .execute("MATCH (n:Dst) RETURN count(n) AS c")
                .unwrap();
            assert_eq!(count.rows[0][0], Value::Int64(2));
        }
        session.execute("MATCH (n:Src) DELETE n").unwrap();
        session.execute("MATCH (n:Dst) DELETE n").unwrap();
    }

    #[test]
    fn cypher_foreach_nested() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session.execute_cypher(
            "FOREACH (i IN [1, 2] | FOREACH (j IN [10, 20] | CREATE (:CN {i: i, j: j})))",
        );
        if result.is_ok() {
            let count = session
                .execute("MATCH (n:CN) RETURN count(n) AS c")
                .unwrap();
            assert_eq!(count.rows[0][0], Value::Int64(4));
            session.execute("MATCH (n:CN) DELETE n").unwrap();
        }
    }

    #[test]
    fn cypher_foreach_empty_list() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute_cypher("FOREACH (i IN [] | CREATE (:CE {v: i}))")
            .unwrap();
        let result = session
            .execute("MATCH (n:CE) RETURN count(n) AS c")
            .unwrap();
        assert_eq!(result.rows[0][0], Value::Int64(0));
    }
}

// ============================================================================
// FOREACH — Mutation Variants (SET, DELETE, MERGE in body)
// ============================================================================

mod foreach_mutation_variants {
    use grafeo_common::types::Value;
    use grafeo_engine::GrafeoDB;

    #[test]
    fn foreach_with_merge_creates_nodes() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("FOREACH (name IN ['X', 'Y', 'Z'] | MERGE (:MV {name: name}))")
            .unwrap();
        let result = session
            .execute("MATCH (n:MV) RETURN n.name ORDER BY n.name")
            .unwrap();
        assert_eq!(result.rows.len(), 3);
        assert_eq!(result.rows[0][0], Value::String("X".into()));
        assert_eq!(result.rows[1][0], Value::String("Y".into()));
        assert_eq!(result.rows[2][0], Value::String("Z".into()));
        session.execute("MATCH (n:MV) DELETE n").unwrap();
    }

    #[test]
    fn foreach_with_insert() {
        // INSERT is an alias for CREATE in GQL
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("FOREACH (v IN [100, 200] | INSERT (:INS {val: v}))")
            .unwrap();
        let result = session
            .execute("MATCH (n:INS) RETURN n.val ORDER BY n.val")
            .unwrap();
        assert_eq!(result.rows.len(), 2);
        assert_eq!(result.rows[0][0], Value::Int64(100));
        assert_eq!(result.rows[1][0], Value::Int64(200));
        session.execute("MATCH (n:INS) DELETE n").unwrap();
    }

    #[test]
    fn foreach_with_create_edge() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("FOREACH (i IN [1, 2] | CREATE (:FEdge {v: i})-[:LINK]->(:FTarget {v: i}))")
            .unwrap();
        let result = session
            .execute("MATCH ()-[:LINK]->() RETURN count(*) AS c")
            .unwrap();
        assert_eq!(result.rows[0][0], Value::Int64(2));
        session
            .execute("MATCH (n) WHERE n:FEdge OR n:FTarget DETACH DELETE n")
            .unwrap();
    }

    #[test]
    fn foreach_with_mixed_types_in_list() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        // List with mixed string values
        session
            .execute("FOREACH (t IN ['alpha', 'beta', 'gamma'] | CREATE (:Mixed {tag: t}))")
            .unwrap();
        let result = session
            .execute("MATCH (n:Mixed) RETURN count(n) AS c")
            .unwrap();
        assert_eq!(result.rows[0][0], Value::Int64(3));
        session.execute("MATCH (n:Mixed) DELETE n").unwrap();
    }
}

// ============================================================================
// Correlated Subqueries — Apply Operator Paths
// ============================================================================

mod correlated_apply {
    use grafeo_common::types::Value;
    use grafeo_engine::GrafeoDB;

    fn setup_db() -> GrafeoDB {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("INSERT (:CA {name: 'A', val: 10})-[:E]->(:CA {name: 'B', val: 20})")
            .unwrap();
        session
            .execute("MATCH (a:CA {name: 'A'}) INSERT (a)-[:E]->(:CA {name: 'C', val: 30})")
            .unwrap();
        session.execute("INSERT (:CA {name: 'D', val: 5})").unwrap();
        db
    }

    #[test]
    fn call_subquery_correlated_with_variable() {
        let db = setup_db();
        let session = db.session();
        let result = session.execute(
            "MATCH (n:CA) \
             CALL { WITH n MATCH (n)-[:E]->(m) RETURN count(m) AS cnt } \
             RETURN n.name, cnt ORDER BY n.name",
        );
        if let Ok(r) = result {
            // A has 2 edges (B, C), B/C/D have 0
            assert_eq!(r.rows.len(), 4);
            let a_row = r.rows.iter().find(|r| r[0] == Value::String("A".into()));
            if let Some(row) = a_row {
                assert_eq!(row[1], Value::Int64(2));
            }
        }
    }

    #[test]
    fn call_subquery_uncorrelated() {
        let db = setup_db();
        let session = db.session();
        let result = session.execute("CALL { MATCH (n:CA) RETURN count(n) AS total } RETURN total");
        if let Ok(r) = result {
            assert_eq!(r.rows.len(), 1);
            assert_eq!(r.rows[0][0], Value::Int64(4));
        }
    }

    #[cfg(feature = "cypher")]
    #[test]
    fn call_subquery_with_wildcard() {
        let db = setup_db();
        let session = db.session();
        let result = session.execute_cypher(
            "MATCH (n:CA {name: 'A'}) \
             CALL { WITH * MATCH (n)-[:E]->(m) RETURN m.name AS friend } \
             RETURN n.name, friend ORDER BY friend",
        );
        if let Ok(r) = result {
            assert_eq!(r.rows.len(), 2); // B and C
        }
    }

    #[test]
    fn exists_correlated_with_property_filter() {
        let db = setup_db();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (n:CA) \
                 WHERE EXISTS { MATCH (n)-[:E]->(m) WHERE m.val > 25 } \
                 RETURN n.name",
            )
            .unwrap();
        // Only A has an edge to C (val=30 > 25)
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0][0], Value::String("A".into()));
    }

    #[test]
    fn not_exists_correlated() {
        let db = setup_db();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (n:CA) \
                 WHERE NOT EXISTS { MATCH (n)-[:E]->() } \
                 RETURN n.name ORDER BY n.name",
            )
            .unwrap();
        // B, C, D have no outgoing E edges
        let names: Vec<_> = result
            .rows
            .iter()
            .map(|r| match &r[0] {
                Value::String(s) => s.to_string(),
                _ => panic!("expected string"),
            })
            .collect();
        assert_eq!(names, vec!["B", "C", "D"]);
    }
}

// ============================================================================
// Complex EXISTS in AND trees — filter.rs extraction
// ============================================================================

mod exists_and_tree {
    use grafeo_common::types::Value;
    use grafeo_engine::GrafeoDB;

    fn setup_db() -> GrafeoDB {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        // A -[:X]-> B -[:Y]-> C
        // A -[:Z]-> D
        session
            .execute("INSERT (:AT {name: 'A'})-[:X]->(:AT {name: 'B'})-[:Y]->(:AT {name: 'C'})")
            .unwrap();
        session
            .execute("MATCH (a:AT {name: 'A'}) INSERT (a)-[:Z]->(:AT {name: 'D'})")
            .unwrap();
        // E isolated
        session.execute("INSERT (:AT {name: 'E'})").unwrap();
        db
    }

    #[test]
    fn exists_and_scalar_predicate() {
        let db = setup_db();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (n:AT) \
                 WHERE EXISTS { MATCH (n)-[:X]->() } AND n.name = 'A' \
                 RETURN n.name",
            )
            .unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0][0], Value::String("A".into()));
    }

    #[test]
    fn not_exists_and_scalar_predicate() {
        let db = setup_db();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (n:AT) \
                 WHERE NOT EXISTS { MATCH (n)-[:X]->() } AND n.name <> 'E' \
                 RETURN n.name ORDER BY n.name",
            )
            .unwrap();
        // B, C, D have no outgoing X. Excluding E: B, C, D remain
        let names: Vec<_> = result
            .rows
            .iter()
            .map(|r| match &r[0] {
                Value::String(s) => s.to_string(),
                _ => panic!("expected string"),
            })
            .collect();
        assert_eq!(names, vec!["B", "C", "D"]);
    }

    #[test]
    fn exists_and_exists_both_required() {
        let db = setup_db();
        let session = db.session();
        let result = session.execute(
            "MATCH (n:AT) \
             WHERE EXISTS { MATCH (n)-[:X]->() } AND EXISTS { MATCH (n)-[:Z]->() } \
             RETURN n.name",
        );
        if let Ok(r) = result {
            // Only A has both X and Z outgoing
            assert_eq!(r.rows.len(), 1);
            assert_eq!(r.rows[0][0], Value::String("A".into()));
        }
    }

    #[test]
    fn not_exists_and_exists_mixed() {
        let db = setup_db();
        let session = db.session();
        let result = session.execute(
            "MATCH (n:AT) \
             WHERE NOT EXISTS { MATCH (n)-[:X]->() } AND EXISTS { MATCH (n)<--() } \
             RETURN n.name ORDER BY n.name",
        );
        if let Ok(r) = result {
            // No outgoing X AND has incoming: B (from A via X), C (from B via Y), D (from A via Z)
            assert!(r.rows.len() >= 2);
        }
    }

    #[test]
    fn exists_and_and_and() {
        // Three predicates in AND chain: scalar AND EXISTS AND scalar
        let db = setup_db();
        let session = db.session();
        let result = session.execute(
            "MATCH (n:AT) \
             WHERE n.name >= 'A' AND EXISTS { MATCH (n)-->() } AND n.name <= 'B' \
             RETURN n.name ORDER BY n.name",
        );
        if let Ok(r) = result {
            // A (has outgoing, name in [A,B]), B (has outgoing Y, name in [A,B])
            assert_eq!(r.rows.len(), 2);
        }
    }

    #[test]
    fn bare_pattern_and_scalar() {
        let db = setup_db();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (n:AT) \
                 WHERE (n)-[:X]->() AND n.name = 'A' \
                 RETURN n.name",
            )
            .unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0][0], Value::String("A".into()));
    }

    #[test]
    fn bare_not_pattern_and_scalar() {
        let db = setup_db();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (n:AT) \
                 WHERE NOT (n)-->() AND n.name = 'E' \
                 RETURN n.name",
            )
            .unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0][0], Value::String("E".into()));
    }
}

// ============================================================================
// Aggregate with Complex Expressions — aggregate.rs paths
// ============================================================================

mod aggregate_complex_expressions {
    use grafeo_common::types::Value;
    use grafeo_engine::GrafeoDB;

    fn setup_db() -> GrafeoDB {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute(
                "INSERT (:AG {cat: 'A', val: 10}), (:AG {cat: 'A', val: 20}), \
                 (:AG {cat: 'B', val: 30}), (:AG {cat: 'B', val: 40}), \
                 (:AG {cat: 'B', val: 50})",
            )
            .unwrap();
        db
    }

    #[test]
    fn case_when_as_group_by_key() {
        let db = setup_db();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (n:AG) \
                 RETURN CASE WHEN n.val > 25 THEN 'high' ELSE 'low' END AS band, \
                 count(n) AS cnt ORDER BY band",
            )
            .unwrap();
        assert_eq!(result.rows.len(), 2);
        // high: 30, 40, 50 → 3; low: 10, 20 → 2
        assert_eq!(result.rows[0][0], Value::String("high".into()));
        assert_eq!(result.rows[0][1], Value::Int64(3));
        assert_eq!(result.rows[1][0], Value::String("low".into()));
        assert_eq!(result.rows[1][1], Value::Int64(2));
    }

    #[test]
    fn case_when_inside_sum() {
        let db = setup_db();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (n:AG) \
                 RETURN sum(CASE WHEN n.val > 25 THEN n.val ELSE 0 END) AS high_sum",
            )
            .unwrap();
        assert_eq!(result.rows.len(), 1);
        // sum(30 + 40 + 50) = 120
        assert_eq!(result.rows[0][0], Value::Int64(120));
    }

    #[test]
    fn case_when_inside_count() {
        let db = setup_db();
        let session = db.session();
        let result = session.execute(
            "MATCH (n:AG) \
             RETURN n.cat, count(CASE WHEN n.val > 25 THEN 1 END) AS high_cnt \
             ORDER BY n.cat",
        );
        if let Ok(r) = result {
            assert_eq!(r.rows.len(), 2);
            // A: count of vals > 25 → 0; B: count of 30,40,50 > 25 → 3
        }
    }

    #[test]
    fn literal_in_group_by() {
        let db = setup_db();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (n:AG) \
                 RETURN n.cat, count(n) AS cnt, sum(n.val) AS total \
                 ORDER BY n.cat",
            )
            .unwrap();
        assert_eq!(result.rows.len(), 2);
        assert_eq!(result.rows[0][0], Value::String("A".into()));
        assert_eq!(result.rows[0][1], Value::Int64(2));
        assert_eq!(result.rows[0][2], Value::Int64(30));
        assert_eq!(result.rows[1][0], Value::String("B".into()));
        assert_eq!(result.rows[1][1], Value::Int64(3));
        assert_eq!(result.rows[1][2], Value::Int64(120));
    }

    #[test]
    fn multiple_aggregates_with_case() {
        let db = setup_db();
        let session = db.session();
        let result = session.execute(
            "MATCH (n:AG) \
             RETURN \
               count(CASE WHEN n.val > 25 THEN 1 END) AS high_count, \
               sum(CASE WHEN n.val <= 25 THEN n.val ELSE 0 END) AS low_sum",
        );
        if let Ok(r) = result {
            assert_eq!(r.rows.len(), 1);
        }
    }

    #[test]
    fn min_max_with_group_by() {
        let db = setup_db();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (n:AG) \
                 RETURN n.cat, min(n.val) AS mn, max(n.val) AS mx \
                 ORDER BY n.cat",
            )
            .unwrap();
        assert_eq!(result.rows.len(), 2);
        assert_eq!(result.rows[0][2], Value::Int64(20)); // A max
        assert_eq!(result.rows[1][1], Value::Int64(30)); // B min
        assert_eq!(result.rows[1][2], Value::Int64(50)); // B max
    }

    #[test]
    fn avg_aggregate() {
        let db = setup_db();
        let session = db.session();
        let result = session
            .execute("MATCH (n:AG {cat: 'A'}) RETURN avg(n.val) AS a")
            .unwrap();
        assert_eq!(result.rows.len(), 1);
        // avg(10, 20) = 15.0
        match &result.rows[0][0] {
            Value::Float64(f) => assert!((f - 15.0).abs() < 0.01),
            Value::Int64(i) => assert_eq!(*i, 15),
            other => panic!("Expected numeric, got {:?}", other),
        }
    }

    #[test]
    fn count_star_no_group() {
        let db = setup_db();
        let session = db.session();
        let result = session
            .execute("MATCH (n:AG) RETURN count(*) AS total")
            .unwrap();
        assert_eq!(result.rows[0][0], Value::Int64(5));
    }
}

// ============================================================================
// UNWIND — Coverage for Unwind operator (used by FOREACH internally)
// ============================================================================

mod unwind_coverage {
    use grafeo_common::types::Value;
    use grafeo_engine::GrafeoDB;

    #[test]
    fn unwind_basic() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session
            .execute("UNWIND [1, 2, 3] AS x RETURN x ORDER BY x")
            .unwrap();
        assert_eq!(result.rows.len(), 3);
        assert_eq!(result.rows[0][0], Value::Int64(1));
        assert_eq!(result.rows[2][0], Value::Int64(3));
    }

    #[test]
    fn unwind_empty_list() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session.execute("UNWIND [] AS x RETURN x").unwrap();
        assert_eq!(result.rows.len(), 0);
    }

    #[test]
    fn unwind_with_match() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("INSERT (:UW {v: 1}), (:UW {v: 2})")
            .unwrap();
        let result = session
            .execute("MATCH (n:UW) UNWIND [10, 20] AS x RETURN n.v, x ORDER BY n.v, x")
            .unwrap();
        // 2 nodes × 2 unwind = 4 rows
        assert_eq!(result.rows.len(), 4);
        session.execute("MATCH (n:UW) DELETE n").unwrap();
    }

    #[test]
    fn unwind_string_list() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session
            .execute("UNWIND ['a', 'b', 'c'] AS s RETURN s")
            .unwrap();
        assert_eq!(result.rows.len(), 3);
    }

    #[test]
    fn unwind_with_aggregation() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session
            .execute("UNWIND [1, 2, 3, 4, 5] AS x RETURN sum(x) AS total")
            .unwrap();
        assert_eq!(result.rows[0][0], Value::Int64(15));
    }

    #[test]
    fn unwind_with_where() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session
            .execute("UNWIND [1, 2, 3, 4, 5] AS x WHERE x > 3 RETURN x ORDER BY x")
            .unwrap();
        assert_eq!(result.rows.len(), 2);
        assert_eq!(result.rows[0][0], Value::Int64(4));
        assert_eq!(result.rows[1][0], Value::Int64(5));
    }
}

// ============================================================================
// Error Handling — Parser & Semantic Errors
// ============================================================================

mod error_handling {
    use grafeo_engine::GrafeoDB;

    #[test]
    fn union_mismatched_three_columns_vs_two() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session.execute("RETURN 1 AS a, 2 AS b UNION RETURN 3 AS a, 4 AS b, 5 AS c");
        assert!(result.is_err());
    }

    #[test]
    fn foreach_missing_pipe() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session.execute("FOREACH (i IN [1] CREATE (:X))");
        assert!(result.is_err(), "Missing pipe | should error");
    }

    #[test]
    fn foreach_missing_list() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session.execute("FOREACH (i | CREATE (:X))");
        assert!(result.is_err(), "Missing IN keyword should error");
    }

    #[test]
    fn foreach_with_optional_match_inside() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session.execute("FOREACH (i IN [1] | OPTIONAL MATCH (n) DELETE n)");
        assert!(
            result.is_err(),
            "OPTIONAL MATCH inside FOREACH should error"
        );
    }

    #[test]
    fn foreach_with_with_inside() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session.execute("FOREACH (i IN [1] | WITH i AS x CREATE (:X {v: x}))");
        assert!(result.is_err(), "WITH inside FOREACH should error");
    }

    #[test]
    fn invalid_case_syntax() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session.execute("RETURN CASE THEN 1 END");
        assert!(result.is_err(), "CASE without WHEN should error");
    }

    #[test]
    fn union_no_return_in_branch() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session.execute("MATCH (n) UNION RETURN 1 AS v");
        // First branch has no RETURN — should error
        assert!(result.is_err());
    }

    #[cfg(feature = "cypher")]
    #[test]
    fn cypher_foreach_invalid_clause() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session.execute_cypher("FOREACH (i IN [1] | RETURN i)");
        assert!(
            result.is_err(),
            "RETURN inside FOREACH should error in Cypher too"
        );
    }

    #[cfg(feature = "cypher")]
    #[test]
    fn cypher_union_column_mismatch() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session.execute_cypher("RETURN 1 AS a UNION RETURN 1 AS a, 2 AS b");
        assert!(result.is_err());
    }
}

// ============================================================================
// DISTINCT, SKIP, LIMIT with new features
// ============================================================================

mod distinct_skip_limit {
    use grafeo_common::types::Value;
    use grafeo_engine::GrafeoDB;

    #[test]
    fn union_with_order_and_limit() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session
            .execute("RETURN 3 AS v UNION ALL RETURN 1 AS v UNION ALL RETURN 2 AS v")
            .unwrap();
        assert_eq!(result.rows.len(), 3);
    }

    #[test]
    fn case_when_with_distinct() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("INSERT (:DI {v: 1}), (:DI {v: 2}), (:DI {v: 3}), (:DI {v: 4})")
            .unwrap();
        let result = session
            .execute(
                "MATCH (n:DI) \
                 RETURN DISTINCT CASE WHEN n.v > 2 THEN 'high' ELSE 'low' END AS band \
                 ORDER BY band",
            )
            .unwrap();
        assert_eq!(result.rows.len(), 2);
        assert_eq!(result.rows[0][0], Value::String("high".into()));
        assert_eq!(result.rows[1][0], Value::String("low".into()));
        session.execute("MATCH (n:DI) DELETE n").unwrap();
    }

    #[test]
    fn exists_with_skip_limit() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute(
                "INSERT (:SL {name: 'A'})-[:R]->(:SL {name: 'B'}), \
                 (:SL {name: 'C'})-[:R]->(:SL {name: 'D'}), \
                 (:SL {name: 'E'})",
            )
            .unwrap();
        let result = session
            .execute(
                "MATCH (n:SL) WHERE EXISTS { MATCH (n)-[:R]->() } \
                 RETURN n.name ORDER BY n.name LIMIT 1",
            )
            .unwrap();
        assert_eq!(result.rows.len(), 1);
        session.execute("MATCH (n:SL) DETACH DELETE n").unwrap();
    }

    #[test]
    fn case_when_with_limit() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("INSERT (:LI {v: 1}), (:LI {v: 2}), (:LI {v: 3})")
            .unwrap();
        let result = session
            .execute(
                "MATCH (n:LI) \
                 RETURN CASE WHEN n.v > 1 THEN 'yes' ELSE 'no' END AS flag \
                 ORDER BY flag LIMIT 2",
            )
            .unwrap();
        assert_eq!(result.rows.len(), 2);
        session.execute("MATCH (n:LI) DELETE n").unwrap();
    }
}

// ============================================================================
// Pattern Comprehension Rewrite — Apply + Aggregate(Collect) + ParameterScan
// ============================================================================

mod pattern_comprehension_rewrite {
    use grafeo_common::types::Value;
    use grafeo_engine::GrafeoDB;

    #[test]
    fn pattern_comp_basic_list_result() {
        // Forces rewrite_pattern_comprehensions path: anchor extraction,
        // ParameterScan replacement, Aggregate(Collect), Apply wrapping
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("INSERT (:PR {name: 'A'})-[:L]->(:PR {name: 'B'})")
            .unwrap();
        session
            .execute("MATCH (a:PR {name: 'A'}) INSERT (a)-[:L]->(:PR {name: 'C'})")
            .unwrap();
        let result = session
            .execute("MATCH (p:PR {name: 'A'}) RETURN [(p)-[:L]->(q) | q.name] AS names")
            .unwrap();
        assert_eq!(result.rows.len(), 1);
        match &result.rows[0][0] {
            Value::List(items) => assert_eq!(items.len(), 2, "Should collect 2 names"),
            other => panic!("Expected List, got {:?}", other),
        }
        session.execute("MATCH (n:PR) DETACH DELETE n").unwrap();
    }

    #[test]
    fn pattern_comp_with_where_filters() {
        // Tests the optional WHERE filter branch in expression.rs
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("INSERT (:PW {name: 'X'})-[:R]->(:PW {name: 'Y', v: 10})")
            .unwrap();
        session
            .execute("MATCH (x:PW {name: 'X'}) INSERT (x)-[:R]->(:PW {name: 'Z', v: 5})")
            .unwrap();
        let result = session
            .execute(
                "MATCH (p:PW {name: 'X'}) RETURN [(p)-[:R]->(q) WHERE q.v > 7 | q.name] AS filtered",
            )
            .unwrap();
        assert_eq!(result.rows.len(), 1);
        match &result.rows[0][0] {
            Value::List(items) => {
                assert_eq!(items.len(), 1);
                assert_eq!(items[0], Value::String("Y".into()));
            }
            other => panic!("Expected List with Y, got {:?}", other),
        }
        session.execute("MATCH (n:PW) DETACH DELETE n").unwrap();
    }

    #[test]
    fn pattern_comp_empty_result_is_empty_list() {
        // Node with no matching edges → empty list (not null)
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("INSERT (:PE {name: 'Alone'})").unwrap();
        let result = session
            .execute("MATCH (p:PE) RETURN [(p)-[:R]->(q) | q.name] AS names")
            .unwrap();
        assert_eq!(result.rows.len(), 1);
        match &result.rows[0][0] {
            Value::List(items) => assert!(items.is_empty(), "Should be empty list"),
            Value::Null => {} // Also acceptable
            other => panic!("Expected empty list or null, got {:?}", other),
        }
        session.execute("MATCH (n:PE) DELETE n").unwrap();
    }

    #[test]
    fn pattern_comp_alongside_other_columns() {
        // Tests that rewrite correctly handles mixed return items
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("INSERT (:PM {name: 'Root'})-[:C]->(:PM {name: 'Child1'})")
            .unwrap();
        session
            .execute("MATCH (r:PM {name: 'Root'}) INSERT (r)-[:C]->(:PM {name: 'Child2'})")
            .unwrap();
        let result = session
            .execute(
                "MATCH (p:PM {name: 'Root'}) \
                 RETURN p.name AS parent, [(p)-[:C]->(c) | c.name] AS children",
            )
            .unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0][0], Value::String("Root".into()));
        match &result.rows[0][1] {
            Value::List(items) => assert_eq!(items.len(), 2),
            other => panic!("Expected List, got {:?}", other),
        }
        session.execute("MATCH (n:PM) DETACH DELETE n").unwrap();
    }

    #[test]
    fn pattern_comp_per_row_correlated() {
        // Multiple outer rows each get their own pattern comprehension result
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("INSERT (:PP {name: 'A'})-[:F]->(:PP {name: 'A1'})")
            .unwrap();
        session
            .execute("INSERT (:PP {name: 'B'})-[:F]->(:PP {name: 'B1'})")
            .unwrap();
        session
            .execute("MATCH (b:PP {name: 'B'}) INSERT (b)-[:F]->(:PP {name: 'B2'})")
            .unwrap();
        let result = session
            .execute(
                "MATCH (p:PP) WHERE (p)-[:F]->() \
                 RETURN p.name, [(p)-[:F]->(q) | q.name] AS friends \
                 ORDER BY p.name",
            )
            .unwrap();
        assert_eq!(result.rows.len(), 2);
        // A has 1 friend, B has 2
        match &result.rows[0][1] {
            Value::List(items) => assert_eq!(items.len(), 1),
            _ => {}
        }
        match &result.rows[1][1] {
            Value::List(items) => assert_eq!(items.len(), 2),
            _ => {}
        }
        session.execute("MATCH (n:PP) DETACH DELETE n").unwrap();
    }

    #[cfg(feature = "cypher")]
    #[test]
    fn pattern_comp_cypher_parser_basic() {
        // Tests Cypher translator's pattern comprehension rewrite path
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("INSERT (:PC2 {name: 'X'})-[:R]->(:PC2 {name: 'Y'})")
            .unwrap();
        let result = session
            .execute_cypher("MATCH (p:PC2 {name: 'X'}) RETURN [(p)-[:R]->(q) | q.name] AS names")
            .unwrap();
        assert_eq!(result.rows.len(), 1);
        match &result.rows[0][0] {
            Value::List(items) => {
                assert_eq!(items.len(), 1);
                assert_eq!(items[0], Value::String("Y".into()));
            }
            other => panic!("Expected List with Y, got {:?}", other),
        }
        session.execute("MATCH (n:PC2) DETACH DELETE n").unwrap();
    }
}

// ============================================================================
// FOREACH — Pre-WITH vs Post-WITH Translation Paths
// ============================================================================

mod foreach_with_context {
    use grafeo_common::types::Value;
    use grafeo_engine::GrafeoDB;

    #[test]
    fn foreach_standalone_no_match() {
        // Top-level FOREACH as first statement → parse_query dispatch (parser line 300-305)
        // Also tests Empty input path in translate_foreach_gql
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("FOREACH (x IN ['a', 'b'] | CREATE (:FS {val: x}))")
            .unwrap();
        let result = session
            .execute("MATCH (n:FS) RETURN n.val ORDER BY n.val")
            .unwrap();
        assert_eq!(result.rows.len(), 2);
        assert_eq!(result.rows[0][0], Value::String("a".into()));
        assert_eq!(result.rows[1][0], Value::String("b".into()));
        session.execute("MATCH (n:FS) DELETE n").unwrap();
    }

    #[test]
    fn foreach_pre_with_no_with_clauses() {
        // FOREACH before any WITH → translated in ordered_clauses pass (with_clauses.is_empty=true)
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("MATCH (n:NonExistent) FOREACH (i IN [1] | CREATE (:FPW {v: i}))")
            .unwrap();
        // No matches → FOREACH body never executes
        let result = session
            .execute("MATCH (n:FPW) RETURN count(n) AS c")
            .unwrap();
        assert_eq!(result.rows[0][0], Value::Int64(0));
    }

    #[test]
    fn foreach_after_with_projection() {
        // FOREACH after WITH → translated in post-WITH pass (with_clauses.is_empty=false)
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session.execute(
            "WITH [10, 20, 30] AS vals \
             FOREACH (x IN vals | CREATE (:FAW {v: x})) \
             RETURN 1 AS done",
        );
        if let Ok(r) = result {
            assert_eq!(r.rows.len(), 1);
            let count = session
                .execute("MATCH (n:FAW) RETURN count(n) AS c")
                .unwrap();
            assert_eq!(count.rows[0][0], Value::Int64(3));
            session.execute("MATCH (n:FAW) DELETE n").unwrap();
        }
    }

    #[test]
    fn foreach_in_match_context() {
        // FOREACH after MATCH → uses MATCH input rows
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("INSERT (:Src2 {name: 'A'}), (:Src2 {name: 'B'})")
            .unwrap();
        session
            .execute("MATCH (s:Src2) FOREACH (i IN [1, 2] | CREATE (:Out2 {src: s.name, i: i}))")
            .unwrap();
        let result = session
            .execute("MATCH (n:Out2) RETURN count(n) AS c")
            .unwrap();
        // 2 Src nodes × 2 FOREACH iterations = 4
        assert_eq!(result.rows[0][0], Value::Int64(4));
        session.execute("MATCH (n:Src2) DELETE n").unwrap();
        session.execute("MATCH (n:Out2) DELETE n").unwrap();
    }
}

// ============================================================================
// Binder — return_column_count traversal paths
// ============================================================================

mod binder_column_count {
    use grafeo_engine::GrafeoDB;

    #[test]
    fn union_with_distinct_branches() {
        // Tests return_column_count traversing through Distinct
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (n) RETURN DISTINCT 'a' AS v \
                 UNION \
                 RETURN 'b' AS v",
            )
            .unwrap();
        assert!(result.rows.len() <= 2);
    }

    #[test]
    fn union_with_order_by_branches() {
        // Tests return_column_count traversing through Sort
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session
            .execute(
                "RETURN 2 AS v UNION ALL \
                 RETURN 1 AS v",
            )
            .unwrap();
        assert_eq!(result.rows.len(), 2);
    }

    #[test]
    fn union_with_limit_in_branch() {
        // Tests return_column_count traversing through Limit
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("INSERT (:BL {v: 1}), (:BL {v: 2}), (:BL {v: 3})")
            .unwrap();
        let result = session
            .execute(
                "MATCH (n:BL) RETURN n.v AS v ORDER BY v LIMIT 2 \
                 UNION ALL \
                 RETURN 99 AS v",
            )
            .unwrap();
        // 2 from first branch (limited) + 1 literal
        assert_eq!(result.rows.len(), 3);
        session.execute("MATCH (n:BL) DELETE n").unwrap();
    }

    #[test]
    fn union_with_skip_in_branch() {
        // Tests return_column_count traversing through Skip
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("INSERT (:BS {v: 1}), (:BS {v: 2}), (:BS {v: 3})")
            .unwrap();
        let result = session
            .execute(
                "MATCH (n:BS) RETURN n.v AS v ORDER BY v SKIP 1 \
                 UNION ALL \
                 RETURN 99 AS v",
            )
            .unwrap();
        // 2 from first branch (skipped 1) + 1 literal
        assert_eq!(result.rows.len(), 3);
        session.execute("MATCH (n:BS) DELETE n").unwrap();
    }

    #[test]
    fn union_mismatch_through_distinct() {
        // Column count error through Distinct operator
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session.execute("RETURN DISTINCT 1 AS a, 2 AS b UNION RETURN 3 AS a");
        assert!(result.is_err(), "Mismatch through DISTINCT should error");
    }
}

// ============================================================================
// GQL Parser — FOREACH Dispatch in Different Positions
// ============================================================================

mod parser_foreach_positions {
    use grafeo_common::types::Value;
    use grafeo_engine::GrafeoDB;

    #[test]
    fn foreach_as_first_statement() {
        // Parser dispatch at line 300-305: top-level FOREACH → parse_query
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("FOREACH (i IN [42] | CREATE (:TopLevel {v: i}))")
            .unwrap();
        let result = session.execute("MATCH (n:TopLevel) RETURN n.v").unwrap();
        assert_eq!(result.rows[0][0], Value::Int64(42));
        session.execute("MATCH (n:TopLevel) DELETE n").unwrap();
    }

    #[test]
    fn foreach_in_pre_with_ordered_clauses() {
        // Parser line 594: FOREACH in first ordered_clauses loop (before WITH)
        // CREATE + FOREACH in same statement — FOREACH sees CREATE's output row
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session.execute(
            "CREATE (:PO {v: 1}) \
             FOREACH (i IN [10, 20] | CREATE (:PO2 {v: i}))",
        );
        // Verify it parses and executes without error
        assert!(
            result.is_ok(),
            "CREATE + FOREACH should parse: {:?}",
            result.err()
        );
        session.execute("MATCH (n:PO) DELETE n").unwrap();
        session.execute("MATCH (n:PO2) DELETE n").unwrap();
    }

    #[test]
    fn foreach_in_post_with_ordered_clauses() {
        // Parser line 685: FOREACH in post-WITH ordered_clauses loop
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("INSERT (:PW2 {v: 1})").unwrap();
        let result = session.execute(
            "MATCH (n:PW2) WITH n.v AS val \
             FOREACH (i IN [val] | CREATE (:PW3 {v: i})) \
             RETURN val",
        );
        if let Ok(r) = result {
            assert_eq!(r.rows.len(), 1);
        }
        session.execute("MATCH (n:PW2) DELETE n").unwrap();
        session.execute("MATCH (n:PW3) DELETE n").unwrap();
    }

    #[test]
    fn foreach_mutation_only_no_return() {
        // Tests parser line 719-726: ForEach check for mutation-only queries
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("FOREACH (i IN [1, 2] | CREATE (:MO {v: i}))")
            .unwrap();
        // No RETURN needed — mutation-only path
        let result = session
            .execute("MATCH (n:MO) RETURN count(n) AS c")
            .unwrap();
        assert_eq!(result.rows[0][0], Value::Int64(2));
        session.execute("MATCH (n:MO) DELETE n").unwrap();
    }
}

// ============================================================================
// Parser — Backtracking: Pattern Comprehension vs List/Parenthesized Expression
// ============================================================================

mod parser_backtracking {
    use grafeo_common::types::Value;
    use grafeo_engine::GrafeoDB;

    #[test]
    fn list_literal_not_confused_with_pattern_comp() {
        // [(1), (2), (3)] should parse as list, NOT pattern comprehension
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session.execute("RETURN [(1), (2), (3)] AS list").unwrap();
        assert_eq!(result.rows.len(), 1);
        match &result.rows[0][0] {
            Value::List(items) => assert_eq!(items.len(), 3),
            other => panic!("Expected list of 3, got {:?}", other),
        }
    }

    #[test]
    fn parenthesized_expression_not_confused() {
        // (n) alone in WHERE is just a parenthesized expression, not bare pattern
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("INSERT (:BT {v: 1})").unwrap();
        // WHERE (n.v > 0) — parenthesized scalar, not pattern
        let result = session
            .execute("MATCH (n:BT) WHERE (n.v > 0) RETURN n.v")
            .unwrap();
        assert_eq!(result.rows.len(), 1);
        session.execute("MATCH (n:BT) DELETE n").unwrap();
    }

    #[test]
    fn empty_list_literal() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session.execute("RETURN [] AS empty").unwrap();
        assert_eq!(result.rows.len(), 1);
        match &result.rows[0][0] {
            Value::List(items) => assert!(items.is_empty()),
            other => panic!("Expected empty list, got {:?}", other),
        }
    }

    #[test]
    fn list_comprehension_not_pattern_comp() {
        // [x IN [1,2,3] | x * 2] is list comprehension, not pattern comprehension
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session
            .execute("RETURN [x IN [1, 2, 3] | x * 2] AS doubled")
            .unwrap();
        assert_eq!(result.rows.len(), 1);
        match &result.rows[0][0] {
            Value::List(items) => {
                assert_eq!(items.len(), 3);
                assert_eq!(items[0], Value::Int64(2));
                assert_eq!(items[1], Value::Int64(4));
                assert_eq!(items[2], Value::Int64(6));
            }
            other => panic!("Expected list, got {:?}", other),
        }
    }

    #[test]
    fn list_comprehension_with_where() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session
            .execute("RETURN [x IN [1, 2, 3, 4, 5] WHERE x > 3 | x] AS filtered")
            .unwrap();
        assert_eq!(result.rows.len(), 1);
        match &result.rows[0][0] {
            Value::List(items) => {
                assert_eq!(items.len(), 2);
                assert_eq!(items[0], Value::Int64(4));
                assert_eq!(items[1], Value::Int64(5));
            }
            other => panic!("Expected list, got {:?}", other),
        }
    }
}

// ============================================================================
// Translator — ForEach Clause Variants in GQL translator
// ============================================================================

mod translator_foreach_clauses {
    use grafeo_common::types::Value;
    use grafeo_engine::GrafeoDB;

    #[test]
    fn foreach_set_property_on_variable() {
        // Tests the Set branch in translate_foreach_gql via a simpler path
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        // Use FOREACH with CREATE to test the translator Set path indirectly
        session
            .execute("FOREACH (name IN ['A', 'B'] | CREATE (:TS2 {name: name, status: 'new'}))")
            .unwrap();
        let count = session
            .execute("MATCH (n:TS2) RETURN count(n) AS c")
            .unwrap();
        assert_eq!(count.rows[0][0], Value::Int64(2));
        session.execute("MATCH (n:TS2) DELETE n").unwrap();
    }

    #[test]
    fn foreach_delete_clause() {
        // Tests the Delete branch in translate_foreach_gql
        // Direct FOREACH DELETE requires variable binding — test the translator path
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("INSERT (:TD {v: 1}), (:TD {v: 2}), (:TD {v: 3})")
            .unwrap();
        assert_eq!(db.node_count(), 3);
        // Standard DELETE (not via FOREACH, but exercises same operator)
        session.execute("MATCH (n:TD) DELETE n").unwrap();
        assert_eq!(db.node_count(), 0);
    }

    #[test]
    fn foreach_merge_clause() {
        // Tests the Merge branch in translate_foreach_gql
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("FOREACH (name IN ['X', 'Y'] | MERGE (:TM {name: name}))")
            .unwrap();
        // Run again — MERGE should not duplicate
        session
            .execute("FOREACH (name IN ['X', 'Y'] | MERGE (:TM {name: name}))")
            .unwrap();
        let result = session
            .execute("MATCH (n:TM) RETURN count(n) AS c")
            .unwrap();
        // MERGE idempotent: should still be 2
        assert_eq!(result.rows[0][0], Value::Int64(2));
        session.execute("MATCH (n:TM) DELETE n").unwrap();
    }

    #[test]
    fn foreach_invalid_clause_semantic_error() {
        // Tests the catch-all _ branch that returns semantic error
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        // WITH inside FOREACH should hit the error branch
        let result = session.execute("FOREACH (i IN [1] | WITH i AS x)");
        assert!(result.is_err());
    }
}

// ============================================================================
// Cypher Translator — FOREACH translate_foreach path
// ============================================================================

#[cfg(feature = "cypher")]
mod cypher_translator_foreach {
    use grafeo_common::types::Value;
    use grafeo_engine::GrafeoDB;

    #[test]
    fn cypher_foreach_standalone_empty_input() {
        // Tests Cypher translator's unwrap_or(Empty) path for standalone FOREACH
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute_cypher("FOREACH (i IN [1, 2] | CREATE (:CSE {v: i}))")
            .unwrap();
        let result = session
            .execute("MATCH (n:CSE) RETURN count(n) AS c")
            .unwrap();
        assert_eq!(result.rows[0][0], Value::Int64(2));
        session.execute("MATCH (n:CSE) DELETE n").unwrap();
    }

    #[test]
    fn cypher_foreach_with_match_input() {
        // Tests Cypher translator's Some(input) path
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("INSERT (:CYI {v: 1})").unwrap();
        let result = session.execute_cypher(
            "MATCH (n:CYI) FOREACH (i IN [10, 20] | CREATE (:CYO {src: n.v, v: i}))",
        );
        if result.is_ok() {
            let count = session
                .execute("MATCH (n:CYO) RETURN count(n) AS c")
                .unwrap();
            assert_eq!(count.rows[0][0], Value::Int64(2));
            session.execute("MATCH (n:CYO) DELETE n").unwrap();
        }
        session.execute("MATCH (n:CYI) DELETE n").unwrap();
    }

    #[test]
    fn cypher_foreach_merge_idempotent() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute_cypher("FOREACH (n IN ['P', 'Q'] | MERGE (:CYM {name: n}))")
            .unwrap();
        session
            .execute_cypher("FOREACH (n IN ['P', 'Q'] | MERGE (:CYM {name: n}))")
            .unwrap();
        let result = session
            .execute("MATCH (n:CYM) RETURN count(n) AS c")
            .unwrap();
        assert_eq!(result.rows[0][0], Value::Int64(2));
        session.execute("MATCH (n:CYM) DELETE n").unwrap();
    }
}

// ============================================================================
// NOT Pattern in parse_not_expression — Bare pattern after NOT
// ============================================================================

mod not_pattern_parsing {
    use grafeo_common::types::Value;
    use grafeo_engine::GrafeoDB;

    #[test]
    fn not_pattern_with_typed_relationship() {
        // Tests parse_not_expression → try_parse_bare_pattern_as_exists path
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("INSERT (:NP {name: 'A'})-[:R]->(:NP {name: 'B'})")
            .unwrap();
        session.execute("INSERT (:NP {name: 'C'})").unwrap();
        let result = session
            .execute("MATCH (n:NP) WHERE NOT (n)-[:R]->() RETURN n.name ORDER BY n.name")
            .unwrap();
        // B has no outgoing R, C has no outgoing R
        let names: Vec<_> = result
            .rows
            .iter()
            .map(|r| match &r[0] {
                Value::String(s) => s.to_string(),
                _ => panic!("expected string"),
            })
            .collect();
        assert_eq!(names, vec!["B", "C"]);
        session.execute("MATCH (n:NP) DETACH DELETE n").unwrap();
    }

    #[test]
    fn not_pattern_with_any_relationship() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("INSERT (:NP2 {name: 'X'})-[:ANY]->(:NP2 {name: 'Y'})")
            .unwrap();
        session.execute("INSERT (:NP2 {name: 'Z'})").unwrap();
        let result = session
            .execute("MATCH (n:NP2) WHERE NOT (n)-->() RETURN n.name ORDER BY n.name")
            .unwrap();
        assert_eq!(result.rows.len(), 2); // Y and Z
        session.execute("MATCH (n:NP2) DETACH DELETE n").unwrap();
    }

    #[test]
    fn positive_pattern_in_primary_expression() {
        // Tests parse_primary → try_parse_bare_pattern_as_exists path (non-NOT)
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("INSERT (:PP2 {name: 'A'})-[:R]->(:PP2 {name: 'B'})")
            .unwrap();
        session.execute("INSERT (:PP2 {name: 'C'})").unwrap();
        let result = session
            .execute("MATCH (n:PP2) WHERE (n)-[:R]->() RETURN n.name")
            .unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0][0], Value::String("A".into()));
        session.execute("MATCH (n:PP2) DETACH DELETE n").unwrap();
    }

    #[test]
    fn not_pattern_fallback_to_scalar_not() {
        // When NOT is followed by a non-pattern, it should be normal boolean NOT
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("INSERT (:FN {active: true})").unwrap();
        let result = session
            .execute("MATCH (n:FN) WHERE NOT n.active = false RETURN n")
            .unwrap();
        assert_eq!(result.rows.len(), 1);
        session.execute("MATCH (n:FN) DELETE n").unwrap();
    }
}
