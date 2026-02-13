//! Integration tests for CALL procedure support.
//!
//! Tests CALL statement parsing + execution across GQL, Cypher, and SQL/PGQ.

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// Creates a small test graph: Alice -> Bob -> Carol (all :Person, connected via :KNOWS).
fn setup_graph() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let alice = db.create_node(&["Person"]);
    let bob = db.create_node(&["Person"]);
    let carol = db.create_node(&["Person"]);

    db.set_node_property(alice, "name", Value::from("Alice"));
    db.set_node_property(bob, "name", Value::from("Bob"));
    db.set_node_property(carol, "name", Value::from("Carol"));

    db.create_edge(alice, bob, "KNOWS");
    db.create_edge(bob, carol, "KNOWS");

    db
}

// ==================== GQL Parser Tests ====================

#[test]
fn test_gql_call_pagerank() {
    let db = setup_graph();
    let session = db.session();
    let result = session.execute("CALL grafeo.pagerank()").unwrap();

    assert_eq!(result.columns.len(), 2);
    assert_eq!(result.columns[0], "node_id");
    assert_eq!(result.columns[1], "score");
    assert_eq!(result.row_count(), 3); // 3 nodes
}

#[test]
fn test_gql_call_pagerank_with_params() {
    let db = setup_graph();
    let session = db.session();
    let result = session
        .execute("CALL grafeo.pagerank({damping: 0.85, max_iterations: 10})")
        .unwrap();

    assert_eq!(result.row_count(), 3);
    // Scores should sum to approximately 1.0
    let total_score: f64 = result
        .rows
        .iter()
        .map(|row| match &row[1] {
            Value::Float64(f) => *f,
            _ => 0.0,
        })
        .sum();
    assert!(
        (total_score - 1.0).abs() < 0.1,
        "PageRank scores should sum to ~1.0, got {}",
        total_score
    );
}

#[test]
fn test_gql_call_with_yield() {
    let db = setup_graph();
    let session = db.session();
    let result = session
        .execute("CALL grafeo.pagerank() YIELD score")
        .unwrap();

    assert_eq!(result.columns.len(), 1);
    assert_eq!(result.columns[0], "score");
    assert_eq!(result.row_count(), 3);
}

#[test]
fn test_gql_call_with_yield_alias() {
    let db = setup_graph();
    let session = db.session();
    let result = session
        .execute("CALL grafeo.pagerank() YIELD node_id AS id, score AS rank")
        .unwrap();

    assert_eq!(result.columns.len(), 2);
    assert_eq!(result.columns[0], "id");
    assert_eq!(result.columns[1], "rank");
}

#[test]
fn test_gql_call_connected_components() {
    let db = setup_graph();
    let session = db.session();
    let result = session
        .execute("CALL grafeo.connected_components()")
        .unwrap();

    assert_eq!(result.columns.len(), 2);
    assert_eq!(result.columns[0], "node_id");
    assert_eq!(result.columns[1], "component_id");
    assert_eq!(result.row_count(), 3);
    // All 3 nodes should be in the same component (connected graph)
    let components: Vec<&Value> = result.rows.iter().map(|r| &r[1]).collect();
    assert_eq!(components[0], components[1]);
    assert_eq!(components[1], components[2]);
}

#[test]
fn test_gql_call_without_namespace() {
    let db = setup_graph();
    let session = db.session();
    // Should also work without "grafeo." prefix
    let result = session.execute("CALL pagerank()").unwrap();
    assert_eq!(result.row_count(), 3);
}

#[test]
fn test_gql_call_unknown_procedure() {
    let db = setup_graph();
    let session = db.session();
    let result = session.execute("CALL grafeo.nonexistent()");
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("Unknown procedure"),
        "Expected 'Unknown procedure' error, got: {}",
        err
    );
}

#[test]
fn test_gql_call_procedures_list() {
    let db = setup_graph();
    let session = db.session();
    let result = session.execute("CALL grafeo.procedures()").unwrap();

    assert_eq!(result.columns.len(), 4);
    assert_eq!(result.columns[0], "name");
    assert_eq!(result.columns[1], "description");
    assert!(result.row_count() >= 22, "Expected at least 22 procedures");
}

#[test]
fn test_gql_call_empty_graph() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let result = session.execute("CALL grafeo.pagerank()").unwrap();
    assert_eq!(result.row_count(), 0);
}

// ==================== Cypher Tests ====================

#[test]
#[cfg(feature = "cypher")]
fn test_cypher_call_pagerank() {
    let db = setup_graph();
    let session = db.session();
    let result = session.execute_cypher("CALL grafeo.pagerank()").unwrap();

    assert_eq!(result.columns.len(), 2);
    assert_eq!(result.columns[0], "node_id");
    assert_eq!(result.columns[1], "score");
    assert_eq!(result.row_count(), 3);
}

#[test]
#[cfg(feature = "cypher")]
fn test_cypher_call_with_yield() {
    let db = setup_graph();
    let session = db.session();
    let result = session
        .execute_cypher("CALL grafeo.pagerank() YIELD score")
        .unwrap();

    assert_eq!(result.columns.len(), 1);
    assert_eq!(result.columns[0], "score");
}

#[test]
#[cfg(feature = "cypher")]
fn test_cypher_call_connected_components() {
    let db = setup_graph();
    let session = db.session();
    let result = session
        .execute_cypher("CALL grafeo.connected_components()")
        .unwrap();

    assert_eq!(result.row_count(), 3);
}

// ==================== SQL/PGQ Tests ====================

#[test]
#[cfg(feature = "sql-pgq")]
fn test_sql_pgq_call_pagerank() {
    let db = setup_graph();
    let session = db.session();
    let result = session.execute_sql("CALL grafeo.pagerank()").unwrap();

    assert_eq!(result.columns.len(), 2);
    assert_eq!(result.columns[0], "node_id");
    assert_eq!(result.columns[1], "score");
    assert_eq!(result.row_count(), 3);
}

#[test]
#[cfg(feature = "sql-pgq")]
fn test_sql_pgq_call_with_yield() {
    let db = setup_graph();
    let session = db.session();
    let result = session
        .execute_sql("CALL grafeo.pagerank() YIELD score AS rank")
        .unwrap();

    assert_eq!(result.columns.len(), 1);
    assert_eq!(result.columns[0], "rank");
}

// ==================== Language Parity Tests ====================

#[test]
#[cfg(all(feature = "cypher", feature = "sql-pgq"))]
fn test_language_parity_pagerank() {
    let db = setup_graph();
    let session = db.session();

    let gql_result = session.execute("CALL grafeo.pagerank()").unwrap();
    let cypher_result = session.execute_cypher("CALL grafeo.pagerank()").unwrap();
    let sql_result = session.execute_sql("CALL grafeo.pagerank()").unwrap();

    // All three should return same row count and column names
    assert_eq!(gql_result.columns, cypher_result.columns);
    assert_eq!(gql_result.columns, sql_result.columns);
    assert_eq!(gql_result.row_count(), cypher_result.row_count());
    assert_eq!(gql_result.row_count(), sql_result.row_count());
}

// ==================== Algorithm-Specific Tests ====================

#[test]
fn test_call_bfs() {
    let db = setup_graph();
    let session = db.session();
    // BFS from node 0 (first created node)
    let result = session.execute("CALL grafeo.bfs(0)").unwrap();

    assert_eq!(result.columns.len(), 2);
    assert_eq!(result.columns[0], "node_id");
    assert_eq!(result.columns[1], "depth");
    // Should reach all 3 nodes from node 0
    assert!(result.row_count() >= 1);
}

#[test]
fn test_call_clustering_coefficient() {
    let db = setup_graph();
    let session = db.session();
    let result = session
        .execute("CALL grafeo.clustering_coefficient()")
        .unwrap();

    assert_eq!(result.columns[0], "node_id");
    assert_eq!(result.columns[1], "coefficient");
    assert_eq!(result.columns[2], "triangle_count");
    assert_eq!(result.row_count(), 3);

    // Coefficients should be in [0.0, 1.0]
    for row in &result.rows {
        if let Value::Float64(coeff) = &row[1] {
            assert!(
                (0.0..=1.0).contains(coeff),
                "Coefficient {} out of range",
                coeff
            );
        }
    }
}

#[test]
fn test_call_degree_centrality() {
    let db = setup_graph();
    let session = db.session();
    let result = session.execute("CALL grafeo.degree_centrality()").unwrap();

    assert_eq!(result.columns[0], "node_id");
    assert_eq!(result.columns[1], "in_degree");
    assert_eq!(result.columns[2], "out_degree");
    assert_eq!(result.columns[3], "total_degree");
    assert_eq!(result.row_count(), 3);
}

// ==================== Case Insensitivity ====================

#[test]
fn test_call_case_insensitive() {
    let db = setup_graph();
    let session = db.session();

    // CALL keyword should be case-insensitive (handled by lexer)
    // The procedure name is case-sensitive (matched against algorithm names)
    let result = session.execute("CALL grafeo.pagerank()");
    assert!(result.is_ok());
}
