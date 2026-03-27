//! Tests for aggregate translator and execution coverage gaps.
//!
//! Targets: aggregate.rs (45.45%), common.rs (64.48%), expression.rs (82.32%)
//!
//! ```bash
//! cargo test -p grafeo-engine --test coverage_aggregates
//! ```

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// Creates 5 Data nodes (Alix, Gus, Vincent, Jules, Mia) with x/y/score properties.
fn stats_graph() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for (name, x, y) in [
        ("Alix", 1.0, 2.0),
        ("Gus", 2.0, 4.0),
        ("Vincent", 3.0, 6.0),
        ("Jules", 4.0, 8.0),
        ("Mia", 5.0, 10.0),
    ] {
        session.create_node_with_props(
            &["Data"],
            [
                ("name", Value::String(name.into())),
                ("x", Value::Float64(x)),
                ("y", Value::Float64(y)),
                ("score", Value::Int64(x as i64 * 10)),
            ],
        );
    }
    db
}

// ---------------------------------------------------------------------------
// Wrapped aggregates: exercises extract_wrapped_aggregate (Binary branch)
// ---------------------------------------------------------------------------

#[test]
fn test_wrapped_aggregate_count_gt_zero() {
    let db = stats_graph();
    let s = db.session();
    let r = s
        .execute("MATCH (d:Data) RETURN count(d) > 0 AS has_data")
        .unwrap();
    assert_eq!(r.rows.len(), 1);
    assert_eq!(r.rows[0][0], Value::Bool(true));
}

#[test]
fn test_wrapped_aggregate_sum_minus_literal() {
    let db = stats_graph();
    let s = db.session();
    let r = s
        .execute("MATCH (d:Data) RETURN sum(d.score) - 10 AS adjusted")
        .unwrap();
    assert_eq!(r.rows.len(), 1);
    // sum(10+20+30+40+50) - 10 = 140
    assert_eq!(r.rows[0][0], Value::Int64(140));
}

#[test]
fn test_wrapped_aggregate_not_count() {
    let db = stats_graph();
    let s = db.session();
    let r = s
        .execute("MATCH (d:Data) RETURN NOT (count(d) > 100) AS few")
        .unwrap();
    assert_eq!(r.rows.len(), 1);
    assert_eq!(r.rows[0][0], Value::Bool(true));
}

// ---------------------------------------------------------------------------
// GROUP_CONCAT / LISTAGG with separator
// ---------------------------------------------------------------------------

#[test]
fn test_group_concat_default_separator() {
    let db = stats_graph();
    let s = db.session();
    let r = s
        .execute("MATCH (d:Data) RETURN group_concat(d.name) AS names")
        .unwrap();
    assert_eq!(r.rows.len(), 1);
    if let Value::String(names) = &r.rows[0][0] {
        assert_eq!(names.split(' ').count(), 5);
    } else {
        panic!("expected string, got {:?}", r.rows[0][0]);
    }
}

#[test]
fn test_group_concat_custom_separator() {
    let db = stats_graph();
    let s = db.session();
    let r = s
        .execute("MATCH (d:Data) RETURN group_concat(d.name, ';') AS names")
        .unwrap();
    assert_eq!(r.rows.len(), 1);
    if let Value::String(names) = &r.rows[0][0] {
        assert!(names.contains(';'), "expected semicolons: {names}");
        assert_eq!(names.split(';').count(), 5);
    } else {
        panic!("expected string, got {:?}", r.rows[0][0]);
    }
}

#[test]
fn test_listagg_default_comma() {
    let db = stats_graph();
    let s = db.session();
    let r = s
        .execute("MATCH (d:Data) RETURN listagg(d.name) AS names")
        .unwrap();
    assert_eq!(r.rows.len(), 1);
    if let Value::String(names) = &r.rows[0][0] {
        assert!(names.contains(','), "expected commas: {names}");
    } else {
        panic!("expected string, got {:?}", r.rows[0][0]);
    }
}

// ---------------------------------------------------------------------------
// SAMPLE aggregate
// ---------------------------------------------------------------------------

#[test]
fn test_sample_returns_one_value() {
    let db = stats_graph();
    let s = db.session();
    let r = s
        .execute("MATCH (d:Data) RETURN sample(d.name) AS picked")
        .unwrap();
    assert_eq!(r.rows.len(), 1);
    if let Value::String(name) = &r.rows[0][0] {
        assert!(
            ["Alix", "Gus", "Vincent", "Jules", "Mia"].contains(&name.as_str()),
            "unexpected: {name}"
        );
    } else {
        panic!("expected string, got {:?}", r.rows[0][0]);
    }
}

// ---------------------------------------------------------------------------
// COLLECT aggregate
// ---------------------------------------------------------------------------

#[test]
fn test_collect_to_list() {
    let db = stats_graph();
    let s = db.session();
    let r = s
        .execute("MATCH (d:Data) RETURN collect(d.score) AS scores")
        .unwrap();
    assert_eq!(r.rows.len(), 1);
    if let Value::List(items) = &r.rows[0][0] {
        assert_eq!(items.len(), 5);
    } else {
        panic!("expected list, got {:?}", r.rows[0][0]);
    }
}

// ---------------------------------------------------------------------------
// PERCENTILE with integer parameter (exercises int-to-float branch)
// ---------------------------------------------------------------------------

#[test]
fn test_percentile_disc_with_integer_param() {
    let db = stats_graph();
    let s = db.session();
    // percentile_disc(score, 1) should return max value
    let r = s
        .execute("MATCH (d:Data) RETURN percentile_disc(d.score, 1) AS p100")
        .unwrap();
    assert_eq!(r.rows.len(), 1);
    // May return Float64 or Int64 depending on implementation
    match &r.rows[0][0] {
        Value::Int64(v) => assert_eq!(*v, 50),
        Value::Float64(v) => assert!((*v - 50.0).abs() < 0.01),
        other => panic!("expected numeric, got {other:?}"),
    }
}

#[test]
fn test_percentile_cont_zero() {
    let db = stats_graph();
    let s = db.session();
    let r = s
        .execute("MATCH (d:Data) RETURN percentile_cont(d.score, 0) AS p0")
        .unwrap();
    assert_eq!(r.rows.len(), 1);
    match &r.rows[0][0] {
        Value::Float64(v) => assert!((*v - 10.0).abs() < 0.01),
        Value::Int64(v) => assert_eq!(*v, 10),
        other => panic!("expected numeric, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// STDEV / VARIANCE (sample and population)
// ---------------------------------------------------------------------------

#[test]
fn test_stdev_and_stdevp() {
    let db = stats_graph();
    let s = db.session();
    let r = s
        .execute("MATCH (d:Data) RETURN stdev(d.score) AS s, stdevp(d.score) AS sp")
        .unwrap();
    assert_eq!(r.rows.len(), 1);
    if let Value::Float64(s) = r.rows[0][0] {
        assert!(s > 0.0);
    }
    if let Value::Float64(sp) = r.rows[0][1] {
        assert!(sp > 0.0);
    }
}

#[test]
fn test_variance_and_variance_pop() {
    let db = stats_graph();
    let s = db.session();
    let r = s
        .execute("MATCH (d:Data) RETURN variance(d.score) AS v, var_pop(d.score) AS vp")
        .unwrap();
    assert_eq!(r.rows.len(), 1);
    if let Value::Float64(v) = r.rows[0][0] {
        assert!(v > 0.0);
    }
}

// ---------------------------------------------------------------------------
// Multiple aggregates + GROUP BY in one query (uses WITH for grouping)
// ---------------------------------------------------------------------------

#[test]
fn test_mixed_aggregates_with_group_by() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for (name, city, score) in [
        ("Alix", "Amsterdam", 80),
        ("Gus", "Amsterdam", 90),
        ("Vincent", "Berlin", 70),
        ("Jules", "Berlin", 85),
    ] {
        session.create_node_with_props(
            &["Person"],
            [
                ("name", Value::String(name.into())),
                ("city", Value::String(city.into())),
                ("score", Value::Int64(score)),
            ],
        );
    }
    // Use RETURN with aggregates directly: non-aggregated p.city acts as grouping key
    let r = session
        .execute(
            "MATCH (p:Person) \
             RETURN p.city AS city, count(p) AS cnt, min(p.score) AS lo, max(p.score) AS hi",
        )
        .unwrap();
    assert_eq!(r.rows.len(), 2);
    // Both groups should have count=2
    assert_eq!(r.rows[0][1], Value::Int64(2));
    assert_eq!(r.rows[1][1], Value::Int64(2));
}

// ---------------------------------------------------------------------------
// ORDER BY alias after aggregation (was: "Undefined variable 'city'")
// ---------------------------------------------------------------------------

#[test]
fn test_order_by_alias_after_aggregation() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for (name, city) in [
        ("Alix", "Berlin"),
        ("Gus", "Amsterdam"),
        ("Vincent", "Berlin"),
        ("Jules", "Amsterdam"),
        ("Mia", "Paris"),
    ] {
        session.create_node_with_props(
            &["Person"],
            [
                ("name", Value::String(name.into())),
                ("city", Value::String(city.into())),
            ],
        );
    }
    let r = session
        .execute("MATCH (p:Person) RETURN p.city AS city, count(p) AS cnt ORDER BY city")
        .unwrap();
    assert_eq!(r.rows.len(), 3);
    // Sorted ascending by city: Amsterdam, Berlin, Paris
    assert_eq!(r.rows[0][0], Value::String("Amsterdam".into()));
    assert_eq!(r.rows[1][0], Value::String("Berlin".into()));
    assert_eq!(r.rows[2][0], Value::String("Paris".into()));
}

#[test]
fn test_order_by_aggregate_alias_desc() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for (name, city) in [
        ("Alix", "Berlin"),
        ("Gus", "Amsterdam"),
        ("Vincent", "Berlin"),
        ("Jules", "Amsterdam"),
        ("Mia", "Paris"),
    ] {
        session.create_node_with_props(
            &["Person"],
            [
                ("name", Value::String(name.into())),
                ("city", Value::String(city.into())),
            ],
        );
    }
    let r = session
        .execute("MATCH (p:Person) RETURN p.city AS city, count(p) AS cnt ORDER BY cnt DESC")
        .unwrap();
    assert_eq!(r.rows.len(), 3);
    // DESC by cnt: Amsterdam(2) and Berlin(2) first, Paris(1) last
    assert_eq!(r.rows[0][1], Value::Int64(2));
    assert_eq!(r.rows[1][1], Value::Int64(2));
    assert_eq!(r.rows[2][1], Value::Int64(1));
    // Paris must be last
    assert_eq!(r.rows[2][0], Value::String("Paris".into()));
}

// ---------------------------------------------------------------------------
// Expression coverage: NULLIF
// ---------------------------------------------------------------------------

#[test]
fn test_nullif_expression() {
    let db = stats_graph();
    let s = db.session();
    let r = s
        .execute("MATCH (d:Data) RETURN nullif(d.score, 10) AS v ORDER BY d.x")
        .unwrap();
    // First person (score=10) should get NULL
    assert_eq!(r.rows[0][0], Value::Null);
    // Second person (score=20) should keep 20
    assert_eq!(r.rows[1][0], Value::Int64(20));
}

// ---------------------------------------------------------------------------
// List predicates via Cypher-compatible syntax
// ---------------------------------------------------------------------------

// Tests all()/any() in WHERE clause (RETURN variant in gql_spec_compliance.rs)

#[test]
fn test_list_predicate_all() {
    let db = GrafeoDB::new_in_memory();
    let s = db.session();
    s.create_node_with_props(&["Flag"], [("v", Value::Int64(1))]);
    let r = s
        .execute("MATCH (f:Flag) WHERE all(x IN [2, 4, 6] WHERE x % 2 = 0) RETURN f.v AS v")
        .unwrap();
    assert_eq!(r.rows.len(), 1);
}

#[test]
fn test_list_predicate_any() {
    let db = GrafeoDB::new_in_memory();
    let s = db.session();
    s.create_node_with_props(&["Flag"], [("v", Value::Int64(1))]);
    let r = s
        .execute("MATCH (f:Flag) WHERE any(x IN [1, 2, 3] WHERE x > 2) RETURN f.v AS v")
        .unwrap();
    assert_eq!(r.rows.len(), 1);
}

// ---------------------------------------------------------------------------
// Wrapped aggregate: sum(x) * 2 (right-side aggregate in binary)
// ---------------------------------------------------------------------------

#[test]
fn test_wrapped_aggregate_literal_plus_count() {
    let db = stats_graph();
    let s = db.session();
    // 100 + count(d): aggregate on the right side of binary
    let r = s
        .execute("MATCH (d:Data) RETURN 100 + count(d) AS total")
        .unwrap();
    assert_eq!(r.rows.len(), 1);
    assert_eq!(r.rows[0][0], Value::Int64(105));
}

// ---------------------------------------------------------------------------
// COUNT(DISTINCT x) and COUNT(x) non-null
// ---------------------------------------------------------------------------

#[test]
fn test_count_distinct() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for city in ["Amsterdam", "Amsterdam", "Berlin", "Berlin", "Paris"] {
        session.create_node_with_props(&["City"], [("name", Value::String(city.into()))]);
    }
    let r = session
        .execute("MATCH (c:City) RETURN count(DISTINCT c.name) AS unique_cities")
        .unwrap();
    assert_eq!(r.rows[0][0], Value::Int64(3));
}

#[test]
fn test_count_expression_non_null() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.create_node_with_props(&["Item"], [("val", Value::Int64(1))]);
    session.create_node_with_props(&["Item"], [("val", Value::Null)]);
    session.create_node_with_props(&["Item"], [("val", Value::Int64(3))]);

    let r = session
        .execute("MATCH (i:Item) RETURN count(i.val) AS cnt")
        .unwrap();
    assert_eq!(r.rows[0][0], Value::Int64(2));
}

// ===========================================================================
// T2-01: HAVING clause tests
// ===========================================================================

#[test]
fn test_having_filters_groups() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for (name, city) in [
        ("Alix", "Amsterdam"),
        ("Gus", "Amsterdam"),
        ("Vincent", "Berlin"),
        ("Jules", "Berlin"),
        ("Mia", "Paris"),
    ] {
        session.create_node_with_props(
            &["Person"],
            [
                ("name", Value::String(name.into())),
                ("city", Value::String(city.into())),
            ],
        );
    }
    let r = session
        .execute(
            "MATCH (p:Person) \
             RETURN p.city AS city, count(p) AS cnt \
             ORDER BY city \
             HAVING cnt > 1",
        )
        .unwrap();
    // Only Amsterdam (2) and Berlin (2) qualify; Paris (1) is filtered out
    assert_eq!(r.rows.len(), 2, "HAVING should filter groups with cnt <= 1");
    assert_eq!(r.rows[0][0], Value::String("Amsterdam".into()));
    assert_eq!(r.rows[0][1], Value::Int64(2));
    assert_eq!(r.rows[1][0], Value::String("Berlin".into()));
    assert_eq!(r.rows[1][1], Value::Int64(2));
}

#[test]
fn test_having_no_matching_groups() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for city in ["Amsterdam", "Berlin", "Paris"] {
        session.create_node_with_props(&["City"], [("name", Value::String(city.into()))]);
    }
    let r = session
        .execute(
            "MATCH (c:City) \
             RETURN c.name AS name, count(c) AS cnt \
             HAVING cnt > 10",
        )
        .unwrap();
    // All groups have cnt=1, none pass HAVING cnt > 10
    assert_eq!(r.rows.len(), 0, "No groups should pass HAVING cnt > 10");
}

#[test]
fn test_having_with_sum_aggregate() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for (name, dept, salary) in [
        ("Alix", "Engineering", 90),
        ("Gus", "Engineering", 80),
        ("Vincent", "Sales", 50),
        ("Jules", "Sales", 60),
        ("Mia", "Marketing", 70),
    ] {
        session.create_node_with_props(
            &["Employee"],
            [
                ("name", Value::String(name.into())),
                ("dept", Value::String(dept.into())),
                ("salary", Value::Int64(salary)),
            ],
        );
    }
    let r = session
        .execute(
            "MATCH (e:Employee) \
             RETURN e.dept AS dept, sum(e.salary) AS total \
             ORDER BY dept \
             HAVING total > 100",
        )
        .unwrap();
    // Engineering: 170, Sales: 110, Marketing: 70
    // Only Engineering and Sales qualify
    assert_eq!(r.rows.len(), 2, "Only depts with total > 100 should appear");
    assert_eq!(r.rows[0][0], Value::String("Engineering".into()));
    assert_eq!(r.rows[1][0], Value::String("Sales".into()));
}
