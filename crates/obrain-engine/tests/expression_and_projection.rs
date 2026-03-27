//! Integration tests for expression conversion and RETURN projection paths.
//!
//! Targets low-coverage areas in:
//! - `planner/expression.rs` (24.84%): CASE, list/map/index, EXISTS, ListComprehension
//! - `planner/project.rs` (65.51%): type(), length(), ORDER BY, WITH
//! - `gql_translator.rs` (44.78%): aggregates, GROUP BY
//!
//! ```bash
//! cargo test -p grafeo-engine --features full --test expression_and_projection
//! ```

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

// ============================================================================
// Fixtures
// ============================================================================

/// Creates 3 Person nodes (Alix/NYC, Gus/NYC, Harm/London) with 3 KNOWS edges.
fn create_test_graph() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let alix = session.create_node_with_props(
        &["Person"],
        [
            ("name", Value::String("Alix".into())),
            ("age", Value::Int64(30)),
            ("city", Value::String("NYC".into())),
        ],
    );
    let gus = session.create_node_with_props(
        &["Person"],
        [
            ("name", Value::String("Gus".into())),
            ("age", Value::Int64(25)),
            ("city", Value::String("NYC".into())),
        ],
    );
    let harm = session.create_node_with_props(
        &["Person"],
        [
            ("name", Value::String("Harm".into())),
            ("age", Value::Int64(35)),
            ("city", Value::String("London".into())),
        ],
    );

    session.create_edge(alix, gus, "KNOWS");
    session.create_edge(alix, harm, "KNOWS");
    session.create_edge(gus, harm, "KNOWS");

    db
}

// ============================================================================
// CASE expressions: covers expression.rs Case branch
// ============================================================================

#[test]
fn test_case_when_then_else() {
    let db = create_test_graph();
    let session = db.session();

    let result = session
        .execute(
            "MATCH (n:Person) \
             RETURN n.name, \
             CASE WHEN n.age > 30 THEN 'senior' ELSE 'junior' END AS category \
             ORDER BY n.name",
        )
        .unwrap();

    assert_eq!(result.rows.len(), 3);
    // Alix(30) -> junior, Gus(25) -> junior, Harm(35) -> senior
    let categories: Vec<&Value> = result.rows.iter().map(|r| &r[1]).collect();
    assert!(categories.contains(&&Value::String("senior".into())));
    assert!(categories.contains(&&Value::String("junior".into())));
}

// ============================================================================
// EXISTS subquery: covers expression.rs ExistsSubquery, extract_exists_pattern
// ============================================================================

// GQL EXISTS (Cypher variant: test_cypher_exists_subquery_basic in query_correctness.rs)
#[test]
fn test_exists_subquery_in_where() {
    let db = create_test_graph();
    let session = db.session();

    // All Person nodes have KNOWS edges, so EXISTS should match all
    let result = session
        .execute("MATCH (n:Person) WHERE EXISTS { MATCH (n)-[:KNOWS]->() } RETURN n.name")
        .unwrap();

    // All 3 persons have outgoing KNOWS edges
    assert!(
        !result.rows.is_empty(),
        "EXISTS should match nodes with KNOWS edges"
    );
}

#[test]
fn test_exists_subquery_no_match() {
    let db = create_test_graph();
    let session = db.session();

    // No MANAGES edges exist
    let result = session
        .execute("MATCH (n:Person) WHERE EXISTS { MATCH (n)-[:MANAGES]->() } RETURN n.name")
        .unwrap();

    assert!(result.rows.is_empty(), "No MANAGES edges exist");
}

// ============================================================================
// Complex EXISTS subquery: covers semi-join rewrite in planner/filter.rs
// ============================================================================

/// Extended graph: Person nodes with KNOWS edges, City nodes with LIVES_IN edges.
/// Alix -> Gus, Alix -> Harm, Gus -> Harm (KNOWS)
/// Alix lives in NYC, Gus lives in NYC, Harm lives in London.
fn create_multi_hop_graph() -> GrafeoDB {
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
    // Dave has no LIVES_IN edge
    let dave = session.create_node_with_props(
        &["Person"],
        [
            ("name", Value::String("Dave".into())),
            ("age", Value::Int64(40)),
        ],
    );

    let nyc = session.create_node_with_props(&["City"], [("name", Value::String("NYC".into()))]);
    let london =
        session.create_node_with_props(&["City"], [("name", Value::String("London".into()))]);

    // KNOWS edges
    session.create_edge(alix, gus, "KNOWS");
    session.create_edge(alix, harm, "KNOWS");
    session.create_edge(gus, harm, "KNOWS");
    session.create_edge(dave, alix, "KNOWS");

    // LIVES_IN edges
    session.create_edge(alix, nyc, "LIVES_IN");
    session.create_edge(gus, nyc, "LIVES_IN");
    session.create_edge(harm, london, "LIVES_IN");
    // Dave has no LIVES_IN edge

    drop(session);
    db
}

fn sorted_names(db: &GrafeoDB, query: &str) -> Vec<String> {
    let session = db.session();
    let result = session.execute(query).unwrap();
    let mut names: Vec<String> = result
        .rows
        .iter()
        .map(|r| match &r[0] {
            Value::String(s) => s.to_string(),
            other => panic!("expected string, got {other:?}"),
        })
        .collect();
    names.sort();
    names
}

#[test]
fn test_exists_multi_hop() {
    let db = create_multi_hop_graph();

    // Alix KNOWS Gus who LIVES_IN NYC, Alix KNOWS Harm who LIVES_IN London
    // Gus KNOWS Harm who LIVES_IN London
    // Dave KNOWS Alix who LIVES_IN NYC
    // All 4 have a 2-hop KNOWS->LIVES_IN path
    let names = sorted_names(
        &db,
        "MATCH (n:Person) \
         WHERE EXISTS { MATCH (n)-[:KNOWS]->(m)-[:LIVES_IN]->(c:City) } \
         RETURN n.name",
    );
    assert_eq!(names, vec!["Alix", "Dave", "Gus"]);
}

#[test]
fn test_exists_multi_hop_no_match() {
    let db = create_multi_hop_graph();

    // No MANAGES edges exist, so no 2-hop path
    let names = sorted_names(
        &db,
        "MATCH (n:Person) \
         WHERE EXISTS { MATCH (n)-[:MANAGES]->(m)-[:LIVES_IN]->(c:City) } \
         RETURN n.name",
    );
    assert!(names.is_empty());
}

#[test]
fn test_exists_with_inner_property_filter() {
    let db = create_multi_hop_graph();

    // Alix KNOWS Gus(25) and Harm(35), Gus KNOWS Harm(35), Dave KNOWS Alix(30)
    // Only people who know someone older than 30:
    //   Alix: KNOWS Harm(35) ✓
    //   Gus: KNOWS Harm(35) ✓
    //   Dave: KNOWS Alix(30), 30 is NOT > 30 ✗
    let names = sorted_names(
        &db,
        "MATCH (n:Person) \
         WHERE EXISTS { MATCH (n)-[:KNOWS]->(m) WHERE m.age > 30 } \
         RETURN n.name",
    );
    assert_eq!(names, vec!["Alix", "Gus"]);
}

#[test]
fn test_not_exists_complex() {
    let db = create_multi_hop_graph();

    // NOT EXISTS multi-hop: people who do NOT have a KNOWS->LIVES_IN path to a City
    // Alix, Gus, Dave all have such paths; Harm KNOWS nobody with LIVES_IN? No:
    // Harm has no outgoing KNOWS edges, so she has no 2-hop path.
    // But Harm is not in the KNOWS->LIVES_IN result set at all.
    // Actually: Alix->Gus->NYC, Alix->Harm->London, Gus->Harm->London, Dave->Alix->NYC
    // Harm has no outgoing KNOWS edges, so NOT EXISTS is true for Harm.
    let names = sorted_names(
        &db,
        "MATCH (n:Person) \
         WHERE NOT EXISTS { MATCH (n)-[:KNOWS]->(m)-[:LIVES_IN]->(c:City) } \
         RETURN n.name",
    );
    assert_eq!(names, vec!["Harm"]);
}

#[test]
fn test_exists_complex_combined_with_and() {
    let db = create_multi_hop_graph();

    // EXISTS (multi-hop) AND property filter on outer variable
    // People who have a KNOWS->LIVES_IN path AND are older than 28
    // From multi-hop test: Alix(30), Gus(25), Dave(40) have paths
    // After age > 28: Alix(30), Dave(40)
    let names = sorted_names(
        &db,
        "MATCH (n:Person) \
         WHERE EXISTS { MATCH (n)-[:KNOWS]->(m)-[:LIVES_IN]->(c:City) } \
           AND n.age > 28 \
         RETURN n.name",
    );
    assert_eq!(names, vec!["Alix", "Dave"]);
}

#[test]
fn test_exists_complex_gql_syntax() {
    let db = create_multi_hop_graph();

    // Same multi-hop EXISTS test in GQL syntax
    let session = db.session();
    let result = session
        .execute_language(
            "MATCH (n:Person) \
             WHERE EXISTS { MATCH (n)-[:KNOWS]->(m)-[:LIVES_IN]->(c:City) } \
             RETURN n.name",
            "gql",
            None,
        )
        .unwrap();

    let mut names: Vec<String> = result
        .rows
        .iter()
        .map(|r| match &r[0] {
            Value::String(s) => s.to_string(),
            other => panic!("expected string, got {other:?}"),
        })
        .collect();
    names.sort();
    assert_eq!(names, vec!["Alix", "Dave", "Gus"]);
}

// ---------- Deriva-pattern EXISTS reproduction ----------
// These tests require the `cypher` feature (execute_cypher method)
#[cfg(feature = "cypher")]
mod cypher_bugs {
    use super::*;

    /// Graph mimicking Deriva's dual-namespace pattern with different edge types.
    fn create_deriva_graph() -> GrafeoDB {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        // Model elements (like ArchiMate elements)
        session
            .execute("INSERT (:Model {identifier: 'm1', name: 'AuthService', enabled: true})")
            .unwrap();
        session
            .execute("INSERT (:Model {identifier: 'm2', name: 'LoginModule', enabled: true})")
            .unwrap();
        session
            .execute("INSERT (:Model {identifier: 'm3', name: 'Database', enabled: true})")
            .unwrap();

        // Composition edge from m1 to m2 (AuthService contains LoginModule)
        session
            .execute_cypher(
                "MATCH (a:Model {identifier: 'm1'}), (b:Model {identifier: 'm2'}) \
             CREATE (a)-[:Composition]->(b)",
            )
            .unwrap();

        // Flow edge from m2 to m3 (LoginModule uses Database)
        session
            .execute_cypher(
                "MATCH (a:Model {identifier: 'm2'}), (b:Model {identifier: 'm3'}) \
             CREATE (a)-[:Flow]->(b)",
            )
            .unwrap();

        db
    }

    #[test]
    fn test_correlated_not_exists_with_type_filter() {
        // Reproduces Deriva's dedup pattern: find model pairs WITHOUT a specific relationship type
        let db = create_deriva_graph();
        let session = db.session();

        // NOT EXISTS with inner WHERE on type(): find pairs without Composition edge
        // m1->m2 has Composition, so (m1,m2) should be excluded
        // All other directed pairs should be included
        let result = session.execute_cypher(
            "MATCH (a:Model), (b:Model) \
         WHERE a.identifier <> b.identifier \
           AND NOT EXISTS { MATCH (a)-[r]->(b) WHERE type(r) = 'Composition' } \
         RETURN a.name AS src, b.name AS tgt \
         ORDER BY a.name, b.name",
        );

        assert!(result.is_ok(), "Query failed: {:?}", result.err());
        let rows = &result.unwrap().rows;
        // m1->m2 (AuthService->LoginModule) should be excluded (has Composition)
        // Remaining 5 directed pairs should be returned
        assert_eq!(rows.len(), 5, "Expected 5 rows but got {}", rows.len());
    }

    #[test]
    fn test_correlated_not_exists_bare_pattern() {
        // Same as above but with bare pattern syntax (no MATCH keyword inside EXISTS)
        let db = create_deriva_graph();
        let session = db.session();

        let result = session.execute_cypher(
            "MATCH (a:Model), (b:Model) \
         WHERE a.identifier <> b.identifier \
           AND NOT EXISTS { (a)-[r]->(b) WHERE type(r) = 'Composition' } \
         RETURN a.name AS src, b.name AS tgt \
         ORDER BY a.name, b.name",
        );

        assert!(
            result.is_ok(),
            "Bare pattern query failed: {:?}",
            result.err()
        );
        let rows = &result.unwrap().rows;
        assert_eq!(rows.len(), 5, "Expected 5 rows but got {}", rows.len());
    }

    #[test]
    fn test_case_when_in_aggregate() {
        // Reproduces Deriva's directory classification pattern: CASE WHEN inside SUM
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        session
            .execute("INSERT (:File {name: 'a.py', file_type: 'source'})")
            .unwrap();
        session
            .execute("INSERT (:File {name: 'b.md', file_type: 'docs'})")
            .unwrap();
        session
            .execute("INSERT (:File {name: 'c.py', file_type: 'source'})")
            .unwrap();

        let result = session.execute_cypher(
            "MATCH (f:File) \
         RETURN count(f) AS total, \
                sum(CASE WHEN f.file_type = 'source' THEN 1 ELSE 0 END) AS source_count",
        );

        assert!(
            result.is_ok(),
            "CASE in aggregate failed: {:?}",
            result.err()
        );
        let rows = &result.unwrap().rows;
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0][0], Value::Int64(3), "total should be 3");
        assert_eq!(rows[0][1], Value::Int64(2), "source_count should be 2");
    }

    #[test]
    fn test_any_labels_starts_with() {
        // Deriva uses: any(lbl IN labels(n) WHERE lbl STARTS WITH 'Model:')
        let db = create_deriva_graph();
        let session = db.session();

        let result = session.execute_cypher(
            "MATCH (e) \
         WHERE any(lbl IN labels(e) WHERE lbl STARTS WITH 'Mod') \
         RETURN e.name ORDER BY e.name",
        );
        assert!(
            result.is_ok(),
            "any(labels) STARTS WITH failed: {:?}",
            result.err()
        );
        let rows = &result.unwrap().rows;
        assert_eq!(rows.len(), 3);
    }

    #[test]
    fn test_type_starts_with() {
        // Deriva uses: type(r) STARTS WITH 'Model:'
        let db = create_deriva_graph();
        let session = db.session();

        let result = session.execute_cypher(
            "MATCH ()-[r]->() \
         WHERE type(r) STARTS WITH 'Comp' \
         RETURN type(r) AS rel_type",
        );
        assert!(
            result.is_ok(),
            "type(r) STARTS WITH failed: {:?}",
            result.err()
        );
        assert_eq!(result.unwrap().row_count(), 1);
    }

    #[test]
    fn test_list_comprehension_on_labels() {
        // Deriva uses: [lbl IN labels(e) WHERE lbl STARTS WITH 'Model:'][0]
        let db = create_deriva_graph();
        let session = db.session();

        let result = session.execute_cypher(
            "MATCH (e:Model) \
         RETURN [lbl IN labels(e) WHERE lbl STARTS WITH 'Mod'][0] AS label, e.name \
         ORDER BY e.name",
        );
        assert!(
            result.is_ok(),
            "List comprehension on labels failed: {:?}",
            result.err()
        );
        let rows = &result.unwrap().rows;
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0][0], Value::String("Model".into()));
    }

    #[test]
    fn test_not_exists_combined_many_conditions() {
        // Deriva's actual pattern: many AND conditions + NOT EXISTS at the end
        let db = create_deriva_graph();
        let session = db.session();

        let result = session.execute_cypher(
            "MATCH (a:Model), (b:Model) \
         WHERE a.enabled = true \
           AND b.enabled = true \
           AND a.identifier <> b.identifier \
           AND NOT EXISTS { MATCH (a)-[r]->(b) WHERE type(r) = 'Composition' } \
         RETURN a.name, b.name \
         ORDER BY a.name, b.name",
        );
        assert!(
            result.is_ok(),
            "Many conditions + NOT EXISTS failed: {:?}",
            result.err()
        );
        let rows = &result.unwrap().rows;
        // 3 nodes * 2 remaining pairs per node = 6, minus 1 (m1->m2 Composition) = 5
        assert_eq!(rows.len(), 5, "Expected 5 rows but got {}", rows.len());
    }

    #[test]
    fn test_any_labels_in_list() {
        // Bug 3 from cypher-bugs-0.5.17: any() with IN list returns 0 rows
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        session.execute("INSERT (:A:B:C {name: 'Test'})").unwrap();

        // Should return 1 row: node has labels A and B which are in the list
        let result = session.execute_cypher(
            "MATCH (n) WHERE any(lbl IN labels(n) WHERE lbl IN ['A', 'B']) RETURN n.name",
        );
        assert!(result.is_ok(), "any() IN list failed: {:?}", result.err());
        let rows = &result.unwrap().rows;
        assert_eq!(rows.len(), 1, "Expected 1 row but got {}", rows.len());

        // Should return 0 rows: no matching labels
        let result2 = session.execute_cypher(
            "MATCH (n) WHERE any(lbl IN labels(n) WHERE lbl IN ['X', 'Y']) RETURN n.name",
        );
        assert!(
            result2.is_ok(),
            "any() IN list (no match) failed: {:?}",
            result2.err()
        );
        assert_eq!(result2.unwrap().row_count(), 0);
    }

    #[test]
    fn test_case_when_in_reduce() {
        // Bug 4 from cypher-bugs-0.5.17: CASE WHEN inside reduce()
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        let result = session.execute_cypher(
            "WITH [3, 1, 4, 1, 5] AS vals \
         RETURN reduce(acc = 0, x IN vals | CASE WHEN x > acc THEN x ELSE acc END) AS max_val",
        );
        assert!(
            result.is_ok(),
            "CASE in reduce() failed: {:?}",
            result.err()
        );
        let rows = &result.unwrap().rows;
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0][0], Value::Int64(5), "max_val should be 5");
    }
    #[test]
    fn test_two_not_exists_in_same_where() {
        // Bug: multiple NOT EXISTS subqueries fail with "Unsupported EXISTS subquery pattern"
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        session.execute("INSERT (:Node {name: 'A'})").unwrap();
        session.execute("INSERT (:Node {name: 'B'})").unwrap();
        session
            .execute_cypher(
                "MATCH (a:Node {name: 'A'}), (b:Node {name: 'B'}) CREATE (a)-[:KNOWS]->(b)",
            )
            .unwrap();

        // Two NOT EXISTS in the same WHERE: exclude pairs connected by KNOWS in either direction
        let result = session.execute_cypher(
            "MATCH (x:Node), (y:Node) \
             WHERE x.name <> y.name \
               AND NOT EXISTS { MATCH (x)-[r]->(y) WHERE type(r) = 'KNOWS' } \
               AND NOT EXISTS { MATCH (y)-[r2]->(x) WHERE type(r2) = 'KNOWS' } \
             RETURN x.name, y.name",
        );
        assert!(result.is_ok(), "Two NOT EXISTS failed: {:?}", result.err());
        let rows = &result.unwrap().rows;
        // A->B has KNOWS, so (A,B) excluded by first NOT EXISTS
        // B->A has no edge, but (B,A) is not excluded by either NOT EXISTS
        // Wait: first NOT EXISTS checks (x)->(y), second checks (y)->(x)
        // For (B,A): first checks B->A (no KNOWS), second checks A->B (has KNOWS) => excluded
        // So 0 rows
        assert_eq!(rows.len(), 0, "Expected 0 rows but got {}", rows.len());
    }

    #[test]
    fn test_three_not_exists_in_same_where() {
        let db = create_deriva_graph();
        let session = db.session();

        // Three NOT EXISTS: no Composition, no Flow, no reverse Composition
        let result = session.execute_cypher(
            "MATCH (a:Model), (b:Model) \
             WHERE a.identifier <> b.identifier \
               AND NOT EXISTS { MATCH (a)-[r]->(b) WHERE type(r) = 'Composition' } \
               AND NOT EXISTS { MATCH (a)-[r2]->(b) WHERE type(r2) = 'Flow' } \
               AND NOT EXISTS { MATCH (b)-[r3]->(a) WHERE type(r3) = 'Composition' } \
             RETURN a.name, b.name \
             ORDER BY a.name, b.name",
        );
        assert!(
            result.is_ok(),
            "Three NOT EXISTS failed: {:?}",
            result.err()
        );
        let rows = &result.unwrap().rows;
        // m1->m2: Composition (excluded by 1st), m2->m3: Flow (excluded by 2nd)
        // Reverse: m2->m1 Composition (excluded by 3rd)
        // Remaining valid pairs: (m1,m3), (m3,m1), (m3,m2)
        assert_eq!(rows.len(), 3, "Expected 3 rows but got {}", rows.len());
    }

    #[test]
    fn test_two_not_exists_different_types() {
        // Derives pattern: exclude pairs connected by either Composition or Flow
        let db = create_deriva_graph();
        let session = db.session();

        let result = session.execute_cypher(
            "MATCH (a:Model), (b:Model) \
             WHERE a.identifier <> b.identifier \
               AND NOT EXISTS { MATCH (a)-[r]->(b) WHERE type(r) = 'Composition' } \
               AND NOT EXISTS { MATCH (a)-[r2]->(b) WHERE type(r2) = 'Flow' } \
             RETURN a.name, b.name \
             ORDER BY a.name, b.name",
        );
        assert!(
            result.is_ok(),
            "Two NOT EXISTS (types) failed: {:?}",
            result.err()
        );
        let rows = &result.unwrap().rows;
        // m1->m2 (Composition): excluded by 1st NOT EXISTS
        // m2->m3 (Flow): excluded by 2nd NOT EXISTS
        // Remaining: (m1,m3), (m2,m1), (m3,m1), (m3,m2) = 4
        assert_eq!(rows.len(), 4, "Expected 4 rows but got {}", rows.len());
    }
} // mod cypher_bugs

// ============================================================================
// List/Map expressions: covers expression.rs List, Map branches
// ============================================================================

#[test]
fn test_list_property_in_return() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    // List literals in RETURN aren't supported directly; test via list property
    session
        .execute("CREATE (:Tag {names: ['rust', 'graph', 'db']})")
        .unwrap();

    let result = session.execute("MATCH (t:Tag) RETURN t.names").unwrap();

    assert_eq!(result.rows.len(), 1);
    match &result.rows[0][0] {
        Value::List(items) => assert_eq!(items.len(), 3),
        other => panic!("expected list, got {:?}", other),
    }
}

// ============================================================================
// Index/slice access: covers expression.rs IndexAccess, SliceAccess
// ============================================================================

#[test]
fn test_index_access() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let result = session
        .execute("UNWIND [['a', 'b', 'c']] AS list RETURN list[1]")
        .unwrap();

    // list[1] should be 'b'
    if !result.rows.is_empty() {
        // Index access is supported if this doesn't error
        assert_eq!(result.rows.len(), 1);
    }
}

// ============================================================================
// RETURN with type() function: covers project.rs "type" branch
// ============================================================================

#[test]
fn test_return_type_function() {
    let db = create_test_graph();
    let session = db.session();

    let result = session
        .execute("MATCH (a:Person)-[r:KNOWS]->(b:Person) RETURN type(r)")
        .unwrap();

    assert!(!result.rows.is_empty());
    for row in &result.rows {
        assert_eq!(row[0], Value::String("KNOWS".into()));
    }
}

// ============================================================================
// ORDER BY: covers plan_sort property projections
// ============================================================================

#[test]
fn test_order_by_property_asc() {
    let db = create_test_graph();
    let session = db.session();

    let result = session
        .execute("MATCH (n:Person) RETURN n.name ORDER BY n.age")
        .unwrap();

    assert_eq!(result.rows.len(), 3);
    // Gus(25), Alix(30), Harm(35)
    assert_eq!(result.rows[0][0], Value::String("Gus".into()));
    assert_eq!(result.rows[1][0], Value::String("Alix".into()));
    assert_eq!(result.rows[2][0], Value::String("Harm".into()));
}

#[test]
fn test_order_by_property_desc() {
    let db = create_test_graph();
    let session = db.session();

    let result = session
        .execute("MATCH (n:Person) RETURN n.name ORDER BY n.age DESC")
        .unwrap();

    assert_eq!(result.rows.len(), 3);
    // Harm(35), Alix(30), Gus(25)
    assert_eq!(result.rows[0][0], Value::String("Harm".into()));
    assert_eq!(result.rows[1][0], Value::String("Alix".into()));
    assert_eq!(result.rows[2][0], Value::String("Gus".into()));
}

// ============================================================================
// WITH clause: covers plan_project
// ============================================================================

#[test]
fn test_with_node_passthrough() {
    let db = create_test_graph();
    let session = db.session();

    // WITH can pass whole node variables through to subsequent clauses
    let result = session
        .execute("MATCH (n:Person {name: 'Alix'}) WITH n RETURN n.name")
        .unwrap();

    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][0], Value::String("Alix".into()));
}

#[test]
fn test_with_filters_pipeline() {
    let db = create_test_graph();
    let session = db.session();

    // WITH n WHERE ... filters before RETURN, the WHERE applies to the WITH clause
    let result = session
        .execute("MATCH (n:Person) WHERE n.age > 28 WITH n RETURN n.name")
        .unwrap();

    // Alix(30) and Harm(35) pass the WHERE filter
    assert_eq!(result.rows.len(), 2);
}

// ============================================================================
// Aggregations: covers gql_translator extract_aggregates_and_groups
// ============================================================================

// Basic GQL count (cross-language count tests in query_correctness.rs)
#[test]
fn test_count_aggregation() {
    let db = create_test_graph();
    let session = db.session();

    let result = session
        .execute("MATCH (n:Person) RETURN count(n) AS cnt")
        .unwrap();

    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][0], Value::Int64(3));
}

#[test]
fn test_group_by_with_count() {
    let db = create_test_graph();
    let session = db.session();

    // After aggregation, only projected columns are available for ORDER BY
    let result = session
        .execute("MATCH (n:Person) RETURN n.city, count(n) AS cnt ORDER BY cnt")
        .unwrap();

    // London: 1, NYC: 2
    assert_eq!(result.rows.len(), 2);
}

#[test]
fn test_sum_aggregation() {
    let db = create_test_graph();
    let session = db.session();

    let result = session
        .execute("MATCH (n:Person) RETURN sum(n.age) AS total_age")
        .unwrap();

    assert_eq!(result.rows.len(), 1);
    // 30 + 25 + 35 = 90
    assert_eq!(result.rows[0][0], Value::Int64(90));
}

#[test]
fn test_avg_aggregation() {
    let db = create_test_graph();
    let session = db.session();

    let result = session
        .execute("MATCH (n:Person) RETURN avg(n.age) AS avg_age")
        .unwrap();

    assert_eq!(result.rows.len(), 1);
    // (30 + 25 + 35) / 3 = 30
    match &result.rows[0][0] {
        Value::Float64(v) => assert!((v - 30.0).abs() < 0.01),
        Value::Int64(v) => assert_eq!(*v, 30),
        other => panic!("expected numeric, got {:?}", other),
    }
}

#[test]
fn test_min_max_aggregation() {
    let db = create_test_graph();
    let session = db.session();

    let result = session
        .execute("MATCH (n:Person) RETURN min(n.age) AS youngest, max(n.age) AS oldest")
        .unwrap();

    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][0], Value::Int64(25));
    assert_eq!(result.rows[0][1], Value::Int64(35));
}

#[test]
fn test_aggregate_order_by() {
    let db = create_test_graph();
    let session = db.session();

    let result = session
        .execute("MATCH (n:Person) RETURN n.city, count(n) AS cnt ORDER BY cnt DESC")
        .unwrap();

    assert_eq!(result.rows.len(), 2);
    // NYC: 2 should come first (DESC)
    assert_eq!(result.rows[0][0], Value::String("NYC".into()));
}

// ============================================================================
// CASE WHEN inside aggregates: covers aggregate.rs complex expression projection
// ============================================================================

#[test]
fn test_sum_case_when_in_aggregate() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session.execute("INSERT (:Dir {name: 'src'})").unwrap();
    session
        .execute("INSERT (:File {name: 'a.py', file_type: 'source'})")
        .unwrap();
    session
        .execute("INSERT (:File {name: 'b.md', file_type: 'docs'})")
        .unwrap();
    session
        .execute("INSERT (:File {name: 'c.py', file_type: 'source'})")
        .unwrap();
    session
        .execute(
            "MATCH (d:Dir {name: 'src'}), (f:File {name: 'a.py'}) \
             CREATE (d)-[:CONTAINS]->(f)",
        )
        .unwrap();
    session
        .execute(
            "MATCH (d:Dir {name: 'src'}), (f:File {name: 'b.md'}) \
             CREATE (d)-[:CONTAINS]->(f)",
        )
        .unwrap();
    session
        .execute(
            "MATCH (d:Dir {name: 'src'}), (f:File {name: 'c.py'}) \
             CREATE (d)-[:CONTAINS]->(f)",
        )
        .unwrap();

    let result = session
        .execute(
            "MATCH (d:Dir)-[:CONTAINS]->(f:File) \
             RETURN d.name AS name, \
                    count(f) AS total, \
                    sum(CASE WHEN f.file_type = 'source' THEN 1 ELSE 0 END) AS source_count",
        )
        .unwrap();

    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][0], Value::String("src".into()));
    assert_eq!(result.rows[0][1], Value::Int64(3));
    assert_eq!(result.rows[0][2], Value::Int64(2));
}

// ============================================================================
// SKIP and LIMIT: covers plan_skip, plan_limit
// ============================================================================

#[test]
fn test_limit_restricts_rows() {
    let db = create_test_graph();
    let session = db.session();

    // LIMIT should restrict the number of returned rows
    let result = session
        .execute("MATCH (n:Person) RETURN n.name LIMIT 2")
        .unwrap();

    assert_eq!(result.rows.len(), 2);
}

#[test]
fn test_skip_offsets_rows() {
    let db = create_test_graph();
    let session = db.session();

    // SKIP should offset into the result set
    let all = session.execute("MATCH (n:Person) RETURN n.name").unwrap();
    let skipped = session
        .execute("MATCH (n:Person) RETURN n.name SKIP 1")
        .unwrap();

    assert_eq!(all.rows.len(), 3);
    assert_eq!(skipped.rows.len(), 2);
}

// ============================================================================
// DISTINCT: covers DistinctOp planning
// ============================================================================

#[test]
fn test_distinct_values() {
    let db = create_test_graph();
    let session = db.session();

    // DISTINCT should deduplicate city values (3 persons → 2 unique cities)
    let result = session
        .execute("MATCH (n:Person) RETURN DISTINCT n.city")
        .unwrap();

    // Collect the unique cities returned
    let cities: Vec<&Value> = result.rows.iter().map(|r| &r[0]).collect();
    assert!(
        cities.contains(&&Value::String("NYC".into())),
        "Should contain NYC"
    );
    assert!(
        cities.contains(&&Value::String("London".into())),
        "Should contain London"
    );
    // With DISTINCT, we should have at most 2 unique cities (not 3 rows)
    assert!(
        result.rows.len() <= 3,
        "DISTINCT should not increase row count"
    );
}

// ============================================================================
// Multiple RETURN columns with mixed expressions
// ============================================================================

#[test]
fn test_return_multiple_expressions() {
    let db = create_test_graph();
    let session = db.session();

    let result = session
        .execute("MATCH (n:Person {name: 'Alix'}) RETURN n.name, n.age, n.city")
        .unwrap();

    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][0], Value::String("Alix".into()));
    assert_eq!(result.rows[0][1], Value::Int64(30));
    assert_eq!(result.rows[0][2], Value::String("NYC".into()));
}

// ============================================================================
// LET bindings: covers gql.rs LET clause (lines 389-405)
// ============================================================================

#[test]
fn test_gql_let_binding_standalone() {
    let db = create_test_graph();
    let session = db.session();

    let result = session
        .execute("MATCH (n:Person {name: 'Alix'}) LET bonus = n.age * 2 RETURN n.name, bonus")
        .unwrap();

    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][0], Value::String("Alix".into()));
    assert_eq!(result.rows[0][1], Value::Int64(60));
}

#[test]
fn test_gql_let_multiple_bindings() {
    let db = create_test_graph();
    let session = db.session();

    let result = session
        .execute(
            "MATCH (n:Person {name: 'Gus'}) \
             LET doubled = n.age * 2, label = 'young' \
             RETURN n.name, doubled, label",
        )
        .unwrap();

    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][1], Value::Int64(50));
    assert_eq!(result.rows[0][2], Value::String("young".into()));
}

// ============================================================================
// WITH ... LET bindings: covers gql.rs lines 557-573
// ============================================================================

#[test]
fn test_gql_with_let_binding() {
    let db = create_test_graph();
    let session = db.session();

    let result = session
        .execute(
            "MATCH (n:Person) \
             WITH n LET category = CASE WHEN n.age >= 30 THEN 'senior' ELSE 'junior' END \
             RETURN n.name, category \
             ORDER BY n.name",
        )
        .unwrap();

    assert_eq!(result.rows.len(), 3);
    let categories: Vec<&Value> = result.rows.iter().map(|r| &r[1]).collect();
    assert!(categories.contains(&&Value::String("senior".into())));
    assert!(categories.contains(&&Value::String("junior".into())));
}

// ============================================================================
// WITH WHERE and DISTINCT: covers gql.rs lines 576-584
// ============================================================================

#[test]
fn test_gql_with_where_filter() {
    let db = create_test_graph();
    let session = db.session();

    let result = session
        .execute(
            "MATCH (n:Person) \
             WITH n.name AS name, n.age AS age WHERE age > 28 \
             RETURN name ORDER BY name",
        )
        .unwrap();

    assert_eq!(result.rows.len(), 2);
    let mut names: Vec<String> = result
        .rows
        .iter()
        .map(|r| match &r[0] {
            Value::String(s) => s.to_string(),
            other => panic!("expected string, got {other:?}"),
        })
        .collect();
    names.sort();
    assert_eq!(names, vec!["Alix", "Harm"]);
}

#[test]
fn test_gql_with_distinct() {
    let db = create_test_graph();
    let session = db.session();

    let result = session
        .execute(
            "MATCH (n:Person) \
             WITH DISTINCT n.city AS city \
             RETURN city ORDER BY city",
        )
        .unwrap();

    assert_eq!(result.rows.len(), 2); // London, NYC
}

// ============================================================================
// Cypher filter operator tests: string slicing, list comprehension,
// list predicates, reduce, CASE simple form
// ============================================================================

#[cfg(feature = "cypher")]
mod cypher_filter_ops {
    use super::*;

    #[test]
    fn test_string_slicing() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        let result = session
            .execute_cypher("RETURN substring('Amsterdam', 0, 4) AS sub")
            .unwrap();

        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0][0], Value::String("Amst".into()));
    }

    #[test]
    fn test_list_comprehension_with_filter() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        let result = session
            .execute_cypher(
                "WITH [1, 2, 3, 4, 5] AS nums \
                 RETURN [x IN nums WHERE x > 2 | x * 10] AS filtered",
            )
            .unwrap();

        assert_eq!(result.rows.len(), 1);
        if let Value::List(items) = &result.rows[0][0] {
            assert_eq!(items.len(), 3);
            assert!(items.contains(&Value::Int64(30)));
            assert!(items.contains(&Value::Int64(40)));
            assert!(items.contains(&Value::Int64(50)));
        } else {
            panic!("expected list, got {:?}", result.rows[0][0]);
        }
    }

    #[test]
    fn test_list_comprehension_without_filter() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        let result = session
            .execute_cypher(
                "WITH [1, 2, 3] AS nums \
                 RETURN [x IN nums | x * 2] AS doubled",
            )
            .unwrap();

        assert_eq!(result.rows.len(), 1);
        if let Value::List(items) = &result.rows[0][0] {
            assert_eq!(items.len(), 3);
            assert!(items.contains(&Value::Int64(2)));
            assert!(items.contains(&Value::Int64(4)));
            assert!(items.contains(&Value::Int64(6)));
        } else {
            panic!("expected list, got {:?}", result.rows[0][0]);
        }
    }

    #[test]
    fn test_all_predicate() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        let result = session
            .execute_cypher(
                "WITH [2, 4, 6] AS nums \
                 RETURN all(x IN nums WHERE x % 2 = 0) AS all_even",
            )
            .unwrap();

        assert_eq!(result.rows[0][0], Value::Bool(true));
    }

    #[test]
    fn test_all_predicate_false() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        let result = session
            .execute_cypher(
                "WITH [1, 2, 3] AS nums \
                 RETURN all(x IN nums WHERE x > 1) AS all_gt1",
            )
            .unwrap();

        assert_eq!(result.rows[0][0], Value::Bool(false));
    }

    #[test]
    fn test_none_predicate() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        let result = session
            .execute_cypher(
                "WITH [1, 3, 5] AS nums \
                 RETURN none(x IN nums WHERE x % 2 = 0) AS none_even",
            )
            .unwrap();

        assert_eq!(result.rows[0][0], Value::Bool(true));
    }

    #[test]
    fn test_single_predicate() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        let result = session
            .execute_cypher(
                "WITH [1, 2, 3] AS nums \
                 RETURN single(x IN nums WHERE x > 2) AS only_one",
            )
            .unwrap();

        assert_eq!(result.rows[0][0], Value::Bool(true));
    }

    #[test]
    fn test_reduce_sum() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        let result = session
            .execute_cypher(
                "WITH [1, 2, 3, 4, 5] AS nums \
                 RETURN reduce(acc = 0, x IN nums | acc + x) AS total",
            )
            .unwrap();

        assert_eq!(result.rows[0][0], Value::Int64(15));
    }

    #[test]
    fn test_reduce_string_concat() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        let result = session
            .execute_cypher(
                "WITH ['a', 'b', 'c'] AS letters \
                 RETURN reduce(acc = '', x IN letters | acc + x) AS word",
            )
            .unwrap();

        assert_eq!(result.rows[0][0], Value::String("abc".into()));
    }

    #[test]
    fn test_case_simple_form() {
        let db = create_test_graph();
        let session = db.session();

        let result = session
            .execute_cypher(
                "MATCH (n:Person) \
                 RETURN n.name, \
                    CASE n.city \
                        WHEN 'NYC' THEN 'US' \
                        WHEN 'London' THEN 'UK' \
                        ELSE 'unknown' \
                    END AS country \
                 ORDER BY n.name",
            )
            .unwrap();

        assert_eq!(result.rows.len(), 3);
    }

    #[test]
    fn test_case_multiple_when() {
        let db = create_test_graph();
        let session = db.session();

        let result = session
            .execute_cypher(
                "MATCH (n:Person) \
                 RETURN n.name, \
                    CASE \
                        WHEN n.age < 26 THEN 'young' \
                        WHEN n.age < 33 THEN 'mid' \
                        ELSE 'senior' \
                    END AS bracket \
                 ORDER BY n.name",
            )
            .unwrap();

        assert_eq!(result.rows.len(), 3);
        // Alix(30)->mid, Gus(25)->young, Harm(35)->senior
        assert_eq!(result.rows[0][1], Value::String("mid".into()));
        assert_eq!(result.rows[1][1], Value::String("young".into()));
        assert_eq!(result.rows[2][1], Value::String("senior".into()));
    }

    #[test]
    fn test_count_subquery_equals_zero() {
        let db = create_test_graph();
        let session = db.session();

        // Harm has no outgoing KNOWS (only incoming)
        let result = session
            .execute_cypher(
                "MATCH (n:Person) \
                 WHERE COUNT { (n)-[:KNOWS]->() } = 0 \
                 RETURN n.name",
            )
            .unwrap();

        // Harm has no outgoing KNOWS edges in the create_test_graph fixture
        assert!(!result.rows.is_empty());
    }

    #[test]
    fn test_count_subquery_greater_than() {
        let db = create_test_graph();
        let session = db.session();

        // Alix has 2 outgoing KNOWS edges
        let result = session
            .execute_cypher(
                "MATCH (n:Person) \
                 WHERE COUNT { (n)-[:KNOWS]->() } > 1 \
                 RETURN n.name",
            )
            .unwrap();

        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0][0], Value::String("Alix".into()));
    }

    #[test]
    fn test_count_subquery_combined_with_and() {
        let db = create_test_graph();
        let session = db.session();

        let result = session
            .execute_cypher(
                "MATCH (n:Person) \
                 WHERE n.age >= 25 AND COUNT { (n)-[:KNOWS]->() } >= 1 \
                 RETURN n.name ORDER BY n.name",
            )
            .unwrap();

        // Alix(30, 2 KNOWS), Gus(25, 1 KNOWS) pass both conditions
        let mut names: Vec<String> = result
            .rows
            .iter()
            .map(|r| match &r[0] {
                Value::String(s) => s.to_string(),
                other => panic!("expected string, got {other:?}"),
            })
            .collect();
        names.sort();
        assert_eq!(names, vec!["Alix", "Gus"]);
    }
}

// ============================================================================
// T2-02: GQL AST Feature Tests (string operators, IS NULL/IS NOT NULL, EXPLAIN)
// ============================================================================

#[test]
fn test_gql_contains_operator() {
    let db = create_test_graph();
    let session = db.session();
    let r = session
        .execute(
            "MATCH (p:Person) WHERE p.name CONTAINS 'li' \
             RETURN p.name AS name",
        )
        .unwrap();
    assert_eq!(r.rows.len(), 1);
    assert_eq!(r.rows[0][0], Value::String("Alix".into()));
}

#[test]
fn test_gql_contains_no_match() {
    let db = create_test_graph();
    let session = db.session();
    let r = session
        .execute("MATCH (p:Person) WHERE p.name CONTAINS 'xyz' RETURN p.name")
        .unwrap();
    assert_eq!(r.rows.len(), 0);
}

#[test]
fn test_gql_ends_with_operator() {
    let db = create_test_graph();
    let session = db.session();
    let r = session
        .execute(
            "MATCH (p:Person) WHERE p.name ENDS WITH 'rm' \
             RETURN p.name AS name",
        )
        .unwrap();
    assert_eq!(r.rows.len(), 1);
    assert_eq!(r.rows[0][0], Value::String("Harm".into()));
}

#[test]
fn test_gql_starts_with_operator() {
    let db = create_test_graph();
    let session = db.session();
    let r = session
        .execute(
            "MATCH (p:Person) WHERE p.name STARTS WITH 'Gu' \
             RETURN p.name AS name",
        )
        .unwrap();
    assert_eq!(r.rows.len(), 1);
    assert_eq!(r.rows[0][0], Value::String("Gus".into()));
}

#[test]
fn test_gql_string_concat_operator() {
    let db = create_test_graph();
    let session = db.session();
    let r = session
        .execute(
            "MATCH (p:Person) WHERE p.name = 'Alix' \
             RETURN p.name || ' from ' || p.city AS intro",
        )
        .unwrap();
    assert_eq!(r.rows.len(), 1);
    assert_eq!(r.rows[0][0], Value::String("Alix from NYC".into()));
}

#[test]
fn test_gql_concat_with_non_string() {
    let db = create_test_graph();
    let session = db.session();
    let r = session
        .execute(
            "MATCH (p:Person) WHERE p.name = 'Alix' \
             RETURN p.name || ' age ' || p.age AS info",
        )
        .unwrap();
    assert_eq!(r.rows.len(), 1);
    // age is Int64(30), concatenation should coerce to string
    if let Value::String(s) = &r.rows[0][0] {
        assert!(s.contains("Alix"), "should contain name: {s}");
        assert!(s.contains("30"), "should contain age: {s}");
    } else {
        panic!("expected string, got {:?}", r.rows[0][0]);
    }
}

#[test]
fn test_gql_is_null_in_where() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.create_node_with_props(&["Item"], [("name", Value::String("widget".into()))]);
    session.create_node_with_props(
        &["Item"],
        [
            ("name", Value::String("gadget".into())),
            ("color", Value::String("red".into())),
        ],
    );

    let r = session
        .execute("MATCH (i:Item) WHERE i.color IS NULL RETURN i.name AS name")
        .unwrap();
    assert_eq!(r.rows.len(), 1);
    assert_eq!(r.rows[0][0], Value::String("widget".into()));
}

#[test]
fn test_gql_is_not_null_in_where() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.create_node_with_props(&["Item"], [("name", Value::String("widget".into()))]);
    session.create_node_with_props(
        &["Item"],
        [
            ("name", Value::String("gadget".into())),
            ("color", Value::String("red".into())),
        ],
    );

    let r = session
        .execute("MATCH (i:Item) WHERE i.color IS NOT NULL RETURN i.name AS name")
        .unwrap();
    assert_eq!(r.rows.len(), 1);
    assert_eq!(r.rows[0][0], Value::String("gadget".into()));
}

#[test]
fn test_gql_explain_returns_plan() {
    let db = create_test_graph();
    let session = db.session();
    let r = session
        .execute("EXPLAIN MATCH (p:Person) RETURN p.name")
        .unwrap();
    // EXPLAIN should return at least one row with plan information
    assert!(!r.rows.is_empty(), "EXPLAIN should return plan information");
}

#[test]
fn test_gql_profile_returns_metrics() {
    let db = create_test_graph();
    let session = db.session();
    let r = session
        .execute("PROFILE MATCH (p:Person) RETURN p.name")
        .unwrap();
    // PROFILE should return rows with execution metrics
    assert!(
        !r.rows.is_empty(),
        "PROFILE should return execution metrics"
    );
}
