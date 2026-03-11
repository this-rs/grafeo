//! Integration tests for LPG named graph data isolation (issue #133).
//!
//! Verifies that `USE GRAPH`, `SESSION SET SCHEMA`, and `SESSION SET GRAPH`
//! correctly route queries and mutations to the selected named graph,
//! not the default store.

use grafeo_engine::GrafeoDB;

fn db() -> GrafeoDB {
    GrafeoDB::new_in_memory()
}

// ── Basic data isolation ─────────────────────────────────────────

#[test]
fn use_graph_isolates_inserts() {
    let db = db();
    let session = db.session();

    // Insert into default graph
    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

    // Create and switch to named graph
    session.execute("CREATE GRAPH analytics").unwrap();
    session.execute("USE GRAPH analytics").unwrap();

    // Insert into named graph
    session.execute("INSERT (:Event {type: 'click'})").unwrap();

    // MATCH in named graph should only see Event, not Person
    let result = session.execute("MATCH (n) RETURN n").unwrap();
    assert_eq!(
        result.row_count(),
        1,
        "Named graph should only have 1 node (Event)"
    );
}

#[test]
fn default_graph_unchanged_after_named_graph_insert() {
    let db = db();
    let session = db.session();

    session.execute("CREATE GRAPH analytics").unwrap();
    session.execute("USE GRAPH analytics").unwrap();
    session.execute("INSERT (:Event {type: 'click'})").unwrap();

    // Switch back to default
    session.execute("USE GRAPH default").unwrap();

    // Default should have no data
    let result = session.execute("MATCH (n) RETURN n").unwrap();
    assert_eq!(
        result.row_count(),
        0,
        "Default graph should be empty after inserting into named graph"
    );
}

#[test]
fn session_set_schema_isolates_data() {
    let db = db();
    let session = db.session();

    session.execute("CREATE GRAPH reports").unwrap();
    session.execute("SESSION SET SCHEMA reports").unwrap();

    session.execute("INSERT (:Report {title: 'Q1'})").unwrap();

    let result = session.execute("MATCH (n:Report) RETURN n").unwrap();
    assert_eq!(result.row_count(), 1, "Should see the Report node");

    // Reset session back to default
    session.execute("SESSION RESET").unwrap();

    let result = session.execute("MATCH (n) RETURN n").unwrap();
    assert_eq!(
        result.row_count(),
        0,
        "Default graph should have no data after SESSION RESET"
    );
}

#[test]
fn session_set_graph_isolates_data() {
    let db = db();
    let session = db.session();

    session.execute("CREATE GRAPH mydb").unwrap();
    session.execute("SESSION SET GRAPH mydb").unwrap();

    session.execute("INSERT (:Person {name: 'Gus'})").unwrap();

    let result = session.execute("MATCH (n) RETURN n").unwrap();
    assert_eq!(result.row_count(), 1);

    // Switch back to default
    session.execute("SESSION SET GRAPH default").unwrap();

    let result = session.execute("MATCH (n) RETURN n").unwrap();
    assert_eq!(result.row_count(), 0, "Default graph should be empty");
}

// ── Cross-graph isolation ────────────────────────────────────────

#[test]
fn two_named_graphs_are_independent() {
    let db = db();
    let session = db.session();

    session.execute("CREATE GRAPH alpha").unwrap();
    session.execute("CREATE GRAPH beta").unwrap();

    // Insert into alpha
    session.execute("USE GRAPH alpha").unwrap();
    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    session.execute("INSERT (:Person {name: 'Gus'})").unwrap();

    // Insert into beta
    session.execute("USE GRAPH beta").unwrap();
    session
        .execute("INSERT (:Animal {species: 'Cat'})")
        .unwrap();

    // beta should have 1 node
    let result = session.execute("MATCH (n) RETURN n").unwrap();
    assert_eq!(result.row_count(), 1, "beta should have 1 node");

    // alpha should have 2 nodes
    session.execute("USE GRAPH alpha").unwrap();
    let result = session.execute("MATCH (n) RETURN n").unwrap();
    assert_eq!(result.row_count(), 2, "alpha should have 2 nodes");

    // default should have 0 nodes
    session.execute("USE GRAPH default").unwrap();
    let result = session.execute("MATCH (n) RETURN n").unwrap();
    assert_eq!(result.row_count(), 0, "default should have 0 nodes");
}

// ── Transaction guards ───────────────────────────────────────────

#[test]
fn cannot_switch_graph_in_active_transaction() {
    let db = db();
    let session = db.session();

    session.execute("CREATE GRAPH analytics").unwrap();
    session.execute("START TRANSACTION").unwrap();

    let result = session.execute("USE GRAPH analytics");
    assert!(
        result.is_err(),
        "Should not be able to switch graphs within an active transaction"
    );

    session.execute("ROLLBACK").unwrap();
}

#[test]
fn cannot_session_set_graph_in_active_transaction() {
    let db = db();
    let session = db.session();

    session.execute("CREATE GRAPH analytics").unwrap();
    session.execute("START TRANSACTION").unwrap();

    let result = session.execute("SESSION SET GRAPH analytics");
    assert!(
        result.is_err(),
        "SESSION SET GRAPH should fail within an active transaction"
    );

    session.execute("ROLLBACK").unwrap();
}

// ── Drop active graph ────────────────────────────────────────────

#[test]
fn drop_active_graph_resets_to_default() {
    let db = db();
    let session = db.session();

    // Create graph, switch to it, insert data
    session.execute("CREATE GRAPH temp").unwrap();
    session.execute("USE GRAPH temp").unwrap();
    session
        .execute("INSERT (:Temp {val: 'ephemeral'})")
        .unwrap();

    // Drop the graph we're currently on
    session.execute("DROP GRAPH temp").unwrap();

    // Should now be on default graph (which is empty)
    let result = session.execute("MATCH (n) RETURN n").unwrap();
    assert_eq!(
        result.row_count(),
        0,
        "After dropping active graph, should fall back to empty default"
    );
}

// ── Direct CRUD isolation ────────────────────────────────────────

#[test]
fn session_create_node_respects_active_graph() {
    let db = db();
    let session = db.session();

    session.execute("CREATE GRAPH mydb").unwrap();
    session.execute("USE GRAPH mydb").unwrap();

    // Direct CRUD via session
    session.create_node(&["Widget"]);

    let result = session.execute("MATCH (n:Widget) RETURN n").unwrap();
    assert_eq!(
        result.row_count(),
        1,
        "Direct create_node should write to active graph"
    );

    // Default should be empty
    session.execute("USE GRAPH default").unwrap();
    let result = session.execute("MATCH (n) RETURN n").unwrap();
    assert_eq!(
        result.row_count(),
        0,
        "Default graph should not have the Widget node"
    );
}

// ── Query cache isolation ────────────────────────────────────────

#[test]
fn same_query_different_graph_returns_correct_results() {
    let db = db();
    let session = db.session();

    // Insert data into default
    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

    // Execute query on default graph (warms cache)
    let result = session.execute("MATCH (n) RETURN n").unwrap();
    assert_eq!(result.row_count(), 1, "Default graph has 1 node");

    // Create and switch to empty named graph
    session.execute("CREATE GRAPH empty").unwrap();
    session.execute("USE GRAPH empty").unwrap();

    // Same query text, different graph, should not return cached default result
    let result = session.execute("MATCH (n) RETURN n").unwrap();
    assert_eq!(
        result.row_count(),
        0,
        "Empty named graph should return 0 rows, not cached default result"
    );
}

// ── SESSION RESET restores default ───────────────────────────────

#[test]
fn session_reset_returns_to_default_graph() {
    let db = db();
    let session = db.session();

    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

    session.execute("CREATE GRAPH other").unwrap();
    session.execute("USE GRAPH other").unwrap();

    // SESSION RESET should go back to default
    session.execute("SESSION RESET").unwrap();

    let result = session.execute("MATCH (n) RETURN n").unwrap();
    assert_eq!(
        result.row_count(),
        1,
        "After SESSION RESET, should see default graph data"
    );
}

// ── Edge isolation ───────────────────────────────────────────────

#[test]
fn edges_are_isolated_between_graphs() {
    let db = db();
    let session = db.session();

    // Create edge in default
    session
        .execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
        .unwrap();

    session.execute("CREATE GRAPH social").unwrap();
    session.execute("USE GRAPH social").unwrap();

    // Insert different edge in named graph
    session
        .execute("INSERT (:Person {name: 'Vincent'})-[:WORKS_WITH]->(:Person {name: 'Jules'})")
        .unwrap();

    // Named graph should only see WORKS_WITH
    let result = session.execute("MATCH ()-[r]->() RETURN type(r)").unwrap();
    assert_eq!(result.row_count(), 1, "social graph should have 1 edge");

    // Default graph should only see KNOWS
    session.execute("USE GRAPH default").unwrap();
    let result = session.execute("MATCH ()-[r]->() RETURN type(r)").unwrap();
    assert_eq!(result.row_count(), 1, "default graph should have 1 edge");
}
