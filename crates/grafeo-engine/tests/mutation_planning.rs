//! Integration tests for mutation-related planning paths.
//!
//! Targets low-coverage areas in `planner/mutation.rs`:
//! UNWIND, MERGE, CREATE edge/path, DELETE, SET property, label ops,
//! OPTIONAL MATCH (left join), CALL PROCEDURE.
//!
//! ```bash
//! cargo test -p grafeo-engine --features full --test mutation_planning
//! ```

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

// ============================================================================
// Fixtures
// ============================================================================

/// Social network: Alice(30,NYC), Bob(25,NYC), Carol(35,London)
/// Alice-KNOWS->Bob, Alice-KNOWS->Carol, Bob-KNOWS->Carol
/// Alice-WORKS_AT->TechCorp, Bob-WORKS_AT->TechCorp
fn create_social_network() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let alice = session.create_node_with_props(
        &["Person"],
        [
            ("name", Value::String("Alice".into())),
            ("age", Value::Int64(30)),
            ("city", Value::String("NYC".into())),
        ],
    );
    let bob = session.create_node_with_props(
        &["Person"],
        [
            ("name", Value::String("Bob".into())),
            ("age", Value::Int64(25)),
            ("city", Value::String("NYC".into())),
        ],
    );
    let carol = session.create_node_with_props(
        &["Person"],
        [
            ("name", Value::String("Carol".into())),
            ("age", Value::Int64(35)),
            ("city", Value::String("London".into())),
        ],
    );

    let techcorp =
        session.create_node_with_props(&["Company"], [("name", Value::String("TechCorp".into()))]);

    session.create_edge(alice, bob, "KNOWS");
    session.create_edge(alice, carol, "KNOWS");
    session.create_edge(bob, carol, "KNOWS");
    session.create_edge(alice, techcorp, "WORKS_AT");
    session.create_edge(bob, techcorp, "WORKS_AT");

    db
}

// ============================================================================
// UNWIND — covers plan_unwind() with Empty input, prior input, property refs
// ============================================================================

#[test]
fn test_unwind_literal_list() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let result = session.execute("UNWIND [1, 2, 3] AS x RETURN x").unwrap();

    assert_eq!(result.rows.len(), 3);
    let values: Vec<&Value> = result.rows.iter().map(|r| &r[0]).collect();
    assert!(values.contains(&&Value::Int64(1)));
    assert!(values.contains(&&Value::Int64(2)));
    assert!(values.contains(&&Value::Int64(3)));
}

#[test]
fn test_unwind_after_match() {
    let db = create_social_network();
    let session = db.session();

    // GQL: MATCH and UNWIND are both pre-WHERE clauses, so UNWIND goes before WHERE
    let result = session
        .execute("MATCH (n:Person {name: 'Alice'}) UNWIND [10, 20] AS x RETURN n.name, x")
        .unwrap();

    // Alice x 2 unwind elements = 2 rows
    assert_eq!(result.rows.len(), 2);
    for row in &result.rows {
        assert_eq!(row[0], Value::String("Alice".into()));
    }
}

#[test]
fn test_unwind_with_strings() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let result = session
        .execute("UNWIND ['a', 'b', 'c'] AS letter RETURN letter")
        .unwrap();

    assert_eq!(result.rows.len(), 3);
    let values: Vec<&Value> = result.rows.iter().map(|r| &r[0]).collect();
    assert!(values.contains(&&Value::String("a".into())));
    assert!(values.contains(&&Value::String("b".into())));
    assert!(values.contains(&&Value::String("c".into())));
}

#[cfg(feature = "cypher")]
#[test]
fn test_unwind_create() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    // Cypher syntax for UNWIND + CREATE
    session
        .execute_cypher("UNWIND [1, 2, 3] AS x CREATE (:Number {val: x})")
        .unwrap();

    let result = session
        .execute("MATCH (n:Number) RETURN n.val ORDER BY n.val")
        .unwrap();

    assert_eq!(result.rows.len(), 3);
    // Verify actual property values, not just row count
    assert_eq!(result.rows[0][0], Value::Int64(1));
    assert_eq!(result.rows[1][0], Value::Int64(2));
    assert_eq!(result.rows[2][0], Value::Int64(3));
}

#[test]
fn test_unwind_create_map_property_access() {
    // Regression test: UNWIND with map list + property access in CREATE
    // Previously all properties resolved to NULL (bug in plan_create_node)
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session
        .execute(
            "UNWIND [{id: 'u1', name: 'Bob'}, {id: 'u2', name: 'Carol'}] AS props \
             CREATE (:Test {id: props.id, name: props.name})",
        )
        .unwrap();

    let result = session
        .execute("MATCH (n:Test) RETURN n.id AS id, n.name AS name ORDER BY n.id")
        .unwrap();

    assert_eq!(result.rows.len(), 2);
    assert_eq!(result.rows[0][0], Value::String("u1".into()));
    assert_eq!(result.rows[0][1], Value::String("Bob".into()));
    assert_eq!(result.rows[1][0], Value::String("u2".into()));
    assert_eq!(result.rows[1][1], Value::String("Carol".into()));
}

#[test]
fn test_unwind_param_create_map_property_access() {
    // Same as above but with parameter substitution ($nodes instead of literal list)
    use std::collections::BTreeMap;
    use std::collections::HashMap;
    use std::sync::Arc;

    use grafeo_common::types::PropertyKey;

    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let nodes = Value::List(Arc::from(vec![
        Value::Map(Arc::new(BTreeMap::from([
            (PropertyKey::new("id"), Value::String("u1".into())),
            (PropertyKey::new("name"), Value::String("Bob".into())),
        ]))),
        Value::Map(Arc::new(BTreeMap::from([
            (PropertyKey::new("id"), Value::String("u2".into())),
            (PropertyKey::new("name"), Value::String("Carol".into())),
        ]))),
    ]));

    let params = HashMap::from([("nodes".to_string(), nodes)]);

    session
        .execute_with_params(
            "UNWIND $nodes AS props CREATE (:Test {id: props.id, name: props.name})",
            params,
        )
        .unwrap();

    let result = session
        .execute("MATCH (n:Test) RETURN n.id AS id, n.name AS name ORDER BY n.id")
        .unwrap();

    assert_eq!(result.rows.len(), 2);
    assert_eq!(result.rows[0][0], Value::String("u1".into()));
    assert_eq!(result.rows[0][1], Value::String("Bob".into()));
    assert_eq!(result.rows[1][0], Value::String("u2".into()));
    assert_eq!(result.rows[1][1], Value::String("Carol".into()));
}

#[test]
fn test_for_with_ordinality() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let result = session
        .execute("FOR x IN [10, 20, 30] WITH ORDINALITY i RETURN x, i")
        .unwrap();

    assert_eq!(result.rows.len(), 3);
    // ORDINALITY is 1-based
    let mut pairs: Vec<(i64, i64)> = result
        .rows
        .iter()
        .map(|r| {
            let x = match &r[0] {
                Value::Int64(v) => *v,
                _ => panic!("Expected Int64 for x"),
            };
            let i = match &r[1] {
                Value::Int64(v) => *v,
                _ => panic!("Expected Int64 for i"),
            };
            (x, i)
        })
        .collect();
    pairs.sort_unstable();
    assert_eq!(pairs, vec![(10, 1), (20, 2), (30, 3)]);
}

#[test]
fn test_for_with_offset() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let result = session
        .execute("FOR x IN [10, 20, 30] WITH OFFSET idx RETURN x, idx")
        .unwrap();

    assert_eq!(result.rows.len(), 3);
    // OFFSET is 0-based
    let mut pairs: Vec<(i64, i64)> = result
        .rows
        .iter()
        .map(|r| {
            let x = match &r[0] {
                Value::Int64(v) => *v,
                _ => panic!("Expected Int64 for x"),
            };
            let idx = match &r[1] {
                Value::Int64(v) => *v,
                _ => panic!("Expected Int64 for idx"),
            };
            (x, idx)
        })
        .collect();
    pairs.sort_unstable();
    assert_eq!(pairs, vec![(10, 0), (20, 1), (30, 2)]);
}

#[test]
fn test_for_without_ordinality_or_offset() {
    // Ensure basic FOR without WITH still works
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let result = session.execute("FOR x IN [1, 2, 3] RETURN x").unwrap();

    assert_eq!(result.rows.len(), 3);
}

// ============================================================================
// MERGE — covers plan_merge() Empty/non-Empty input, on_create, on_match
// ============================================================================

#[test]
fn test_merge_creates_when_not_exists() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let result = session
        .execute("MERGE (n:Animal {species: 'Cat'}) RETURN n.species")
        .unwrap();

    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][0], Value::String("Cat".into()));
    assert_eq!(db.node_count(), 1);
}

#[test]
fn test_merge_matches_existing() {
    let db = create_social_network();
    let session = db.session();

    let before = db.node_count();

    let result = session
        .execute("MERGE (n:Person {name: 'Alice'}) RETURN n.name")
        .unwrap();

    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][0], Value::String("Alice".into()));
    // No new node created
    assert_eq!(db.node_count(), before);
}

#[test]
fn test_merge_on_create_set() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let result = session
        .execute(
            "MERGE (n:Person {name: 'NewGuy'}) ON CREATE SET n.created = true RETURN n.name, n.created",
        )
        .unwrap();

    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][0], Value::String("NewGuy".into()));
    assert_eq!(result.rows[0][1], Value::Bool(true));
}

#[test]
fn test_merge_on_match_set() {
    let db = create_social_network();
    let session = db.session();

    let result = session
        .execute(
            "MERGE (n:Person {name: 'Alice'}) ON MATCH SET n.found = true RETURN n.name, n.found",
        )
        .unwrap();

    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][0], Value::String("Alice".into()));
    assert_eq!(result.rows[0][1], Value::Bool(true));
}

// ============================================================================
// CREATE — covers plan_create_node, plan_create_edge, try_fold_expression
// ============================================================================

#[test]
fn test_create_node_with_list_property() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

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

#[cfg(feature = "cypher")]
#[test]
fn test_create_edge_named_variable() {
    let db = create_social_network();
    let session = db.session();

    // Cypher: MATCH + CREATE edge with named variable
    let result = session
        .execute_cypher(
            "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) \
             CREATE (a)-[r:LIKES]->(b) RETURN type(r)",
        )
        .unwrap();

    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][0], Value::String("LIKES".into()));
}

#[cfg(feature = "cypher")]
#[test]
fn test_create_edge_anonymous() {
    let db = create_social_network();
    let session = db.session();

    let edges_before = db.edge_count();

    session
        .execute_cypher(
            "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) \
             CREATE (a)-[:LIKES]->(b)",
        )
        .unwrap();

    assert_eq!(db.edge_count(), edges_before + 1);
}

#[test]
fn test_create_path_with_new_nodes() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    // Create two nodes and an edge using the programmatic API
    // (exercises CreateNodeOp and CreateEdgeOp through a different code path)
    let paris =
        session.create_node_with_props(&["City"], [("name", Value::String("Paris".into()))]);
    let france =
        session.create_node_with_props(&["Country"], [("name", Value::String("France".into()))]);
    session.create_edge(paris, france, "IN");

    assert_eq!(db.node_count(), 2);
    assert_eq!(db.edge_count(), 1);

    // Verify the path can be queried via GQL
    let result = session
        .execute("MATCH (c:City)-[:IN]->(co:Country) RETURN c.name, co.name")
        .unwrap();

    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][0], Value::String("Paris".into()));
    assert_eq!(result.rows[0][1], Value::String("France".into()));
}

// ============================================================================
// DELETE — covers plan_delete_node, plan_delete_edge, detach path
// ============================================================================

#[test]
fn test_delete_node_by_match() {
    let db = create_social_network();
    let session = db.session();

    let before = db.node_count();

    // Delete Carol (who has KNOWS edges)
    session
        .execute("MATCH (n:Person) WHERE n.name = 'Carol' DETACH DELETE n")
        .unwrap();

    assert!(db.node_count() < before);
}

#[test]
fn test_delete_node_reduces_count() {
    let db = create_social_network();

    let before = db.node_count();

    // DETACH DELETE a single node removes it and its edges
    let session = db.session();
    session
        .execute("MATCH (n:Company {name: 'TechCorp'}) DETACH DELETE n")
        .unwrap();

    assert_eq!(db.node_count(), before - 1);
}

#[test]
fn test_detach_delete_all_nodes() {
    let db = create_social_network();
    let session = db.session();

    session.execute("MATCH (n) DETACH DELETE n").unwrap();

    assert_eq!(db.node_count(), 0);
    assert_eq!(db.edge_count(), 0);
}

// ============================================================================
// SET property — covers plan_set_property, expression_to_property_source
// ============================================================================

#[test]
fn test_set_property_literal() {
    let db = create_social_network();
    let session = db.session();

    // GQL: SET is processed before WHERE, so use pattern property match
    session
        .execute("MATCH (n:Person {name: 'Alice'}) SET n.age = 31")
        .unwrap();

    let result = session
        .execute("MATCH (n:Person {name: 'Alice'}) RETURN n.age")
        .unwrap();

    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][0], Value::Int64(31));
}

#[test]
fn test_set_property_string() {
    let db = create_social_network();
    let session = db.session();

    session
        .execute("MATCH (n:Person {name: 'Bob'}) SET n.city = 'Berlin'")
        .unwrap();

    let result = session
        .execute("MATCH (n:Person {name: 'Bob'}) RETURN n.city")
        .unwrap();

    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][0], Value::String("Berlin".into()));
}

// ============================================================================
// Label operations — covers plan_add_label, plan_remove_label
// ============================================================================

#[test]
fn test_add_label_via_set() {
    let db = create_social_network();
    let session = db.session();

    // GQL: SET is processed before WHERE, so use pattern property match
    session
        .execute("MATCH (n:Person {name: 'Alice'}) SET n:Employee")
        .unwrap();

    // Alice should now have both Person and Employee labels
    let result = session.execute("MATCH (n:Employee) RETURN n.name").unwrap();

    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][0], Value::String("Alice".into()));
}

#[test]
fn test_remove_label() {
    let db = create_social_network();
    let session = db.session();

    // First add a label, then remove it (use pattern match, not WHERE)
    session
        .execute("MATCH (n:Person {name: 'Alice'}) SET n:Temp")
        .unwrap();

    let result = session.execute("MATCH (n:Temp) RETURN n.name").unwrap();
    assert_eq!(result.rows.len(), 1);

    session
        .execute("MATCH (n:Person {name: 'Alice'}) REMOVE n:Temp")
        .unwrap();

    let result = session.execute("MATCH (n:Temp) RETURN n.name").unwrap();
    assert!(result.rows.is_empty());
}

// ============================================================================
// OPTIONAL MATCH — covers plan_left_join
// ============================================================================

#[test]
fn test_optional_match_with_results() {
    let db = create_social_network();
    let session = db.session();

    // Alice works at TechCorp — use pattern property match (WHERE comes after OPTIONAL MATCH)
    let result = session
        .execute(
            "MATCH (a:Person {name: 'Alice'}) \
             OPTIONAL MATCH (a)-[:WORKS_AT]->(c:Company) \
             RETURN a.name, c.name",
        )
        .unwrap();

    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][0], Value::String("Alice".into()));
    assert_eq!(result.rows[0][1], Value::String("TechCorp".into()));
}

#[test]
fn test_optional_match_null_when_missing() {
    let db = create_social_network();
    let session = db.session();

    // Carol doesn't manage anyone — OPTIONAL MATCH should still produce a row
    // (LEFT JOIN preserves the left side even when right side has no matches)
    let result = session
        .execute(
            "MATCH (a:Person {name: 'Carol'}) \
             OPTIONAL MATCH (a)-[:MANAGES]->(c:Company) \
             RETURN a, c",
        )
        .unwrap();

    // Should produce exactly 1 row — LEFT JOIN keeps Carol even without a MANAGES match
    assert_eq!(
        result.rows.len(),
        1,
        "OPTIONAL MATCH should return 1 row even when right side has no matches"
    );
}

// ============================================================================
// CALL PROCEDURE — covers plan_call_procedure, plan_static_result
// ============================================================================

#[test]
fn test_call_list_procedures() {
    let db = create_social_network();
    let session = db.session();

    let result = session.execute("CALL grafeo.procedures()").unwrap();

    // Should return a list of available procedures
    assert!(!result.rows.is_empty());
}

#[test]
fn test_call_degree_centrality() {
    let db = create_social_network();
    let session = db.session();

    let result = session.execute("CALL grafeo.degree_centrality()").unwrap();

    // Should return results for each node
    assert!(!result.rows.is_empty());
}

#[test]
fn test_call_procedure_with_yield() {
    let db = create_social_network();
    let session = db.session();

    let result = session
        .execute("CALL grafeo.pagerank() YIELD node_id, score RETURN node_id, score")
        .unwrap();

    assert!(!result.rows.is_empty());
    // Each row should have node_id and score
    assert!(result.columns.len() >= 2);
}
