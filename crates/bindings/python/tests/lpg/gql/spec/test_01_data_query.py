"""GQL spec: Data Query Language (ISO sec 14).

Covers: MATCH, OPTIONAL MATCH, WHERE, FILTER, RETURN, SELECT, FINISH,
ORDER BY, LIMIT, SKIP, OFFSET, FETCH FIRST, WITH, UNWIND, LET, FOR,
NEXT, GROUP BY, HAVING, CALL.
"""

import pytest

# =============================================================================
# MATCH (sec 14.4)
# =============================================================================


class TestMatch:
    """MATCH clause variants."""

    def test_match_single_node(self, db):
        """MATCH (n) returns all nodes."""
        db.create_node(["Person"], {"name": "Alix"})
        db.create_node(["Person"], {"name": "Gus"})
        result = list(db.execute("MATCH (n) RETURN count(n) AS cnt"))
        assert result[0]["cnt"] == 2

    def test_match_by_label(self, db):
        """MATCH (n:Label) filters by label."""
        db.create_node(["Person"], {"name": "Alix"})
        db.create_node(["Animal"], {"name": "Rex"})
        result = list(db.execute("MATCH (n:Person) RETURN n.name"))
        assert len(result) == 1
        assert result[0]["n.name"] == "Alix"

    def test_match_by_property(self, db):
        """MATCH (n {prop: val}) inline property filter."""
        db.create_node(["Person"], {"name": "Alix", "age": 30})
        db.create_node(["Person"], {"name": "Gus", "age": 25})
        result = list(db.execute("MATCH (n:Person {name: 'Alix'}) RETURN n.age"))
        assert result[0]["n.age"] == 30

    def test_match_multi_label_colon(self, db):
        """MATCH (n IS L1 & L2) multi-label conjunction syntax."""
        db.create_node(["Person", "Developer"], {"name": "Alix"})
        db.create_node(["Person"], {"name": "Gus"})
        result = list(db.execute("MATCH (n IS Person & Developer) RETURN n.name"))
        assert len(result) == 1
        assert result[0]["n.name"] == "Alix"

    def test_match_comma_separated_patterns(self, db):
        """MATCH (a), (b) cross-product of two independent patterns."""
        db.create_node(["Person"], {"name": "Alix"})
        db.create_node(["Company"], {"name": "Acme"})
        result = list(db.execute("MATCH (p:Person), (c:Company) RETURN p.name, c.name"))
        assert len(result) == 1
        assert result[0]["p.name"] == "Alix"

    def test_match_multiple_match_clauses(self, db):
        """Two sequential MATCH clauses."""
        db.create_node(["Person"], {"name": "Alix"})
        db.create_node(["City"], {"name": "Amsterdam"})
        result = list(db.execute("MATCH (p:Person) MATCH (c:City) RETURN p.name, c.name"))
        assert len(result) == 1
        assert result[0]["c.name"] == "Amsterdam"

    def test_match_edge_pattern(self, db):
        """MATCH (a)-[:TYPE]->(b) directed edge."""
        a = db.create_node(["Person"], {"name": "Alix"})
        b = db.create_node(["Person"], {"name": "Gus"})
        db.create_edge(a.id, b.id, "KNOWS")
        result = list(db.execute("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name, b.name"))
        assert len(result) == 1
        assert result[0]["a.name"] == "Alix"
        assert result[0]["b.name"] == "Gus"


# =============================================================================
# OPTIONAL MATCH (sec 14.4)
# =============================================================================


class TestOptionalMatch:
    """OPTIONAL MATCH returns nulls when no match."""

    def test_optional_match_with_result(self, db):
        """OPTIONAL MATCH returns rows when relationship exists."""
        a = db.create_node(["Person"], {"name": "Alix"})
        b = db.create_node(["Person"], {"name": "Gus"})
        db.create_edge(a.id, b.id, "KNOWS")
        result = list(
            db.execute(
                "MATCH (a:Person {name: 'Alix'}) "
                "OPTIONAL MATCH (a)-[:KNOWS]->(b:Person) "
                "RETURN b.name"
            )
        )
        assert len(result) >= 1
        assert any(r["b.name"] == "Gus" for r in result)

    def test_optional_match_null(self, db):
        """OPTIONAL MATCH returns null when no relationship exists."""
        db.create_node(["Person"], {"name": "Alix"})
        result = list(
            db.execute(
                "MATCH (a:Person {name: 'Alix'}) "
                "OPTIONAL MATCH (a)-[:MANAGES]->(b) "
                "RETURN a.name, b"
            )
        )
        assert len(result) == 1
        assert result[0]["b"] is None


# =============================================================================
# WHERE / FILTER (sec 14.6)
# =============================================================================


class TestWhere:
    """WHERE and FILTER clauses."""

    def test_where_comparison(self, db):
        """WHERE with comparison operator."""
        db.create_node(["Person"], {"name": "Alix", "age": 30})
        db.create_node(["Person"], {"name": "Gus", "age": 25})
        result = list(db.execute("MATCH (n:Person) WHERE n.age > 28 RETURN n.name"))
        assert len(result) == 1
        assert result[0]["n.name"] == "Alix"

    def test_where_and(self, db):
        """WHERE with AND."""
        db.create_node(["Person"], {"name": "Alix", "age": 30, "city": "Amsterdam"})
        db.create_node(["Person"], {"name": "Gus", "age": 25, "city": "Berlin"})
        result = list(
            db.execute("MATCH (n:Person) WHERE n.age > 20 AND n.city = 'Amsterdam' RETURN n.name")
        )
        assert len(result) == 1
        assert result[0]["n.name"] == "Alix"

    def test_where_or(self, db):
        """WHERE with OR."""
        db.create_node(["Person"], {"name": "Alix"})
        db.create_node(["Person"], {"name": "Gus"})
        db.create_node(["Person"], {"name": "Vincent"})
        result = list(
            db.execute("MATCH (n:Person) WHERE n.name = 'Alix' OR n.name = 'Gus' RETURN n.name")
        )
        assert len(result) == 2

    def test_where_not(self, db):
        """WHERE with NOT."""
        db.create_node(["Person"], {"name": "Alix", "city": "Amsterdam"})
        db.create_node(["Person"], {"name": "Gus", "city": "Berlin"})
        result = list(db.execute("MATCH (n:Person) WHERE NOT n.city = 'Amsterdam' RETURN n.name"))
        assert len(result) == 1
        assert result[0]["n.name"] == "Gus"

    def test_where_xor(self, db):
        """WHERE with XOR."""
        db.create_node(["Person"], {"name": "Alix", "active": True, "admin": False})
        db.create_node(["Person"], {"name": "Gus", "active": True, "admin": True})
        db.create_node(["Person"], {"name": "Vincent", "active": False, "admin": False})
        result = list(db.execute("MATCH (n:Person) WHERE n.active XOR n.admin RETURN n.name"))
        assert len(result) == 1
        assert result[0]["n.name"] == "Alix"

    def test_filter_synonym(self, db):
        """FILTER is ISO GQL synonym for WHERE."""
        db.create_node(["Person"], {"name": "Alix", "age": 30})
        db.create_node(["Person"], {"name": "Gus", "age": 25})
        result = list(db.execute("MATCH (n:Person) FILTER n.age > 28 RETURN n.name"))
        assert len(result) == 1
        assert result[0]["n.name"] == "Alix"


# =============================================================================
# RETURN / SELECT / FINISH (sec 14.10, 14.12)
# =============================================================================


class TestReturn:
    """RETURN, SELECT, and FINISH clauses."""

    def test_return_expression(self, db):
        """RETURN with expression."""
        db.create_node(["Person"], {"name": "Alix", "age": 30})
        result = list(db.execute("MATCH (n:Person) RETURN n.name, n.age"))
        assert result[0]["n.name"] == "Alix"
        assert result[0]["n.age"] == 30

    def test_return_alias(self, db):
        """RETURN expr AS alias."""
        db.create_node(["Person"], {"name": "Alix"})
        result = list(db.execute("MATCH (n:Person) RETURN n.name AS person_name"))
        assert result[0]["person_name"] == "Alix"

    @pytest.mark.xfail(reason="RETURN DISTINCT does not deduplicate results")
    def test_return_distinct(self, db):
        """RETURN DISTINCT deduplicates."""
        db.create_node(["Person"], {"name": "Alix", "city": "Amsterdam"})
        db.create_node(["Person"], {"name": "Gus", "city": "Amsterdam"})
        result = list(db.execute("MATCH (n:Person) RETURN DISTINCT n.city"))
        assert len(result) == 1

    def test_return_star(self, db):
        """RETURN * returns all bound variables."""
        db.create_node(["Person"], {"name": "Alix"})
        result = list(db.execute("MATCH (n:Person) RETURN *"))
        assert len(result) == 1

    def test_select_synonym(self, db):
        """SELECT is ISO alternative to RETURN."""
        db.create_node(["Person"], {"name": "Alix"})
        result = list(db.execute("MATCH (n:Person) SELECT n.name"))
        assert result[0]["n.name"] == "Alix"

    def test_finish(self, db):
        """FINISH consumes input and returns empty."""
        db.create_node(["Person"], {"name": "Alix"})
        result = list(db.execute("MATCH (n:Person) FINISH"))
        assert len(result) == 0


# =============================================================================
# ORDER BY / LIMIT / SKIP / OFFSET / FETCH FIRST (sec 14.11)
# =============================================================================


class TestOrdering:
    """ORDER BY, LIMIT, SKIP, OFFSET, FETCH FIRST."""

    def test_order_by_asc(self, db):
        """ORDER BY ascending."""
        db.create_node(["Person"], {"name": "Gus", "age": 25})
        db.create_node(["Person"], {"name": "Alix", "age": 30})
        result = list(db.execute("MATCH (n:Person) RETURN n.name ORDER BY n.age ASC"))
        assert result[0]["n.name"] == "Gus"
        assert result[1]["n.name"] == "Alix"

    def test_order_by_desc(self, db):
        """ORDER BY descending."""
        db.create_node(["Person"], {"name": "Gus", "age": 25})
        db.create_node(["Person"], {"name": "Alix", "age": 30})
        result = list(db.execute("MATCH (n:Person) RETURN n.name ORDER BY n.age DESC"))
        assert result[0]["n.name"] == "Alix"

    def test_order_by_multiple_keys(self, db):
        """ORDER BY with multiple sort keys."""
        db.create_node(["Person"], {"name": "Alix", "city": "Amsterdam", "age": 30})
        db.create_node(["Person"], {"name": "Gus", "city": "Amsterdam", "age": 25})
        db.create_node(["Person"], {"name": "Vincent", "city": "Berlin", "age": 35})
        result = list(db.execute("MATCH (n:Person) RETURN n.name ORDER BY n.city ASC, n.age DESC"))
        assert result[0]["n.name"] == "Alix"
        assert result[1]["n.name"] == "Gus"
        assert result[2]["n.name"] == "Vincent"

    def test_limit(self, db):
        """LIMIT restricts row count."""
        for i in range(5):
            db.create_node(["Item"], {"val": i})
        result = list(db.execute("MATCH (n:Item) RETURN n.val LIMIT 3"))
        assert len(result) == 3

    def test_skip(self, db):
        """SKIP skips initial rows."""
        for i in range(5):
            db.create_node(["Item"], {"val": i})
        result = list(db.execute("MATCH (n:Item) RETURN n.val ORDER BY n.val SKIP 3"))
        assert len(result) == 2

    def test_offset_synonym(self, db):
        """OFFSET is a synonym for SKIP."""
        for i in range(5):
            db.create_node(["Item"], {"val": i})
        result = list(db.execute("MATCH (n:Item) RETURN n.val ORDER BY n.val OFFSET 4"))
        assert len(result) == 1

    def test_skip_and_limit(self, db):
        """SKIP + LIMIT for pagination."""
        for i in range(5):
            db.create_node(["Item"], {"val": i})
        result = list(db.execute("MATCH (n:Item) RETURN n.val ORDER BY n.val SKIP 1 LIMIT 2"))
        assert len(result) == 2
        assert result[0]["n.val"] == 1
        assert result[1]["n.val"] == 2

    def test_fetch_first_n_rows(self, db):
        """FETCH FIRST n ROWS is SQL-style LIMIT."""
        for i in range(5):
            db.create_node(["Item"], {"val": i})
        result = list(db.execute("MATCH (n:Item) RETURN n.val FETCH FIRST 2 ROWS"))
        assert len(result) == 2

    def test_fetch_next_n_rows(self, db):
        """FETCH NEXT n ROWS is equivalent to FETCH FIRST."""
        for i in range(5):
            db.create_node(["Item"], {"val": i})
        result = list(db.execute("MATCH (n:Item) RETURN n.val FETCH NEXT 3 ROWS"))
        assert len(result) == 3


# =============================================================================
# WITH (projection chaining)
# =============================================================================


class TestWith:
    """WITH clause for pipelining."""

    def test_with_projection(self, db):
        """WITH projects intermediate results."""
        db.create_node(["Person"], {"name": "Alix", "age": 30})
        result = list(
            db.execute("MATCH (n:Person) WITH n.name AS name, n.age AS age RETURN name, age")
        )
        assert result[0]["name"] == "Alix"

    def test_with_distinct(self, db):
        """WITH DISTINCT deduplicates intermediate rows."""
        db.create_node(["Person"], {"name": "Alix", "city": "Amsterdam"})
        db.create_node(["Person"], {"name": "Gus", "city": "Amsterdam"})
        result = list(db.execute("MATCH (n:Person) WITH DISTINCT n.city AS city RETURN city"))
        assert len(result) == 1

    def test_with_where(self, db):
        """WITH ... WHERE filters intermediate results."""
        db.create_node(["Person"], {"name": "Alix", "age": 30})
        db.create_node(["Person"], {"name": "Gus", "age": 25})
        result = list(db.execute("MATCH (n:Person) WITH n WHERE n.age > 28 RETURN n.name"))
        assert len(result) == 1
        assert result[0]["n.name"] == "Alix"

    def test_with_star(self, db):
        """WITH * passes all variables through."""
        db.create_node(["Person"], {"name": "Alix", "age": 30})
        result = list(db.execute("MATCH (n:Person) WITH * RETURN n.name"))
        assert result[0]["n.name"] == "Alix"


# =============================================================================
# UNWIND
# =============================================================================


class TestUnwind:
    """UNWIND list expansion."""

    def test_unwind_list(self, db):
        """UNWIND list AS x."""
        db.create_node(["Anchor"], {"v": 1})
        result = list(db.execute("MATCH (n:Anchor) UNWIND [1, 2, 3] AS x RETURN x"))
        vals = [r["x"] for r in result]
        assert vals == [1, 2, 3]

    @pytest.mark.xfail(reason="List properties may not roundtrip as lists through GQL UNWIND")
    def test_unwind_property_list(self, db):
        """UNWIND a property that is a list."""
        db.create_node(["Item"], {"tags": ["a", "b", "c"]})
        result = list(db.execute("MATCH (n:Item) UNWIND n.tags AS tag RETURN tag"))
        assert len(result) == 3


# =============================================================================
# LET (sec 14.7)
# =============================================================================


class TestLet:
    """LET variable binding."""

    @pytest.mark.xfail(
        reason="LET clause does not resolve variables from preceding WITH/MATCH scope"
    )
    def test_let_simple(self, db):
        """LET var = expr parallel assignment."""
        db.create_node(["Person"], {"name": "Alix", "age": 30})
        result = list(
            db.execute("MATCH (n:Person) WITH n LET doubled = n.age * 2 RETURN n.name, doubled")
        )
        assert result[0]["doubled"] == 60

    @pytest.mark.xfail(
        reason="LET clause does not resolve variables from preceding WITH/MATCH scope"
    )
    def test_let_multiple(self, db):
        """LET with multiple bindings."""
        db.create_node(["Person"], {"name": "Alix", "age": 30})
        result = list(
            db.execute(
                "MATCH (n:Person) WITH n "
                "LET upper_name = toUpper(n.name), next_age = n.age + 1 "
                "RETURN upper_name, next_age"
            )
        )
        assert result[0]["upper_name"] == "ALIX"
        assert result[0]["next_age"] == 31


# =============================================================================
# FOR (sec 14.8)
# =============================================================================


class TestFor:
    """FOR list iteration."""

    def test_for_in_list(self, db):
        """FOR x IN list basic iteration."""
        db.create_node(["Anchor"], {"v": 1})
        result = list(db.execute("MATCH (n:Anchor) FOR x IN [10, 20, 30] RETURN x"))
        vals = sorted(r["x"] for r in result)
        assert vals == [10, 20, 30]

    def test_for_with_ordinality(self, db):
        """FOR x IN list WITH ORDINALITY i (1-based index)."""
        db.create_node(["Anchor"], {"v": 1})
        result = list(
            db.execute(
                "MATCH (n:Anchor) FOR x IN ['a', 'b', 'c'] WITH ORDINALITY i RETURN x, i ORDER BY i"
            )
        )
        assert len(result) == 3
        assert result[0]["i"] == 1
        assert result[2]["i"] == 3

    def test_for_with_offset(self, db):
        """FOR x IN list WITH OFFSET idx (0-based index)."""
        db.create_node(["Anchor"], {"v": 1})
        result = list(
            db.execute(
                "MATCH (n:Anchor) "
                "FOR x IN ['a', 'b', 'c'] WITH OFFSET idx "
                "RETURN x, idx ORDER BY idx"
            )
        )
        assert len(result) == 3
        assert result[0]["idx"] == 0
        assert result[2]["idx"] == 2


# =============================================================================
# NEXT (sec 9.2, linear composition)
# =============================================================================


class TestNext:
    """NEXT for linear composition."""

    def test_next_statement(self, db):
        """NEXT chains two query statements via Apply."""
        db.create_node(["Person"], {"name": "Alix", "age": 30})
        db.create_node(["Person"], {"name": "Gus", "age": 25})
        result = list(
            db.execute(
                "MATCH (n:Person) RETURN n.name AS name, n.age AS age "
                "NEXT "
                "MATCH (m:Person) WHERE m.age > 28 RETURN m.name AS senior"
            )
        )
        assert any(r["senior"] == "Alix" for r in result)


# =============================================================================
# GROUP BY / HAVING (sec 16.15)
# =============================================================================


class TestGroupBy:
    """GROUP BY and HAVING."""

    def test_group_by_explicit(self, db):
        """Explicit GROUP BY."""
        db.create_node(["Person"], {"name": "Alix", "city": "Amsterdam"})
        db.create_node(["Person"], {"name": "Gus", "city": "Amsterdam"})
        db.create_node(["Person"], {"name": "Vincent", "city": "Berlin"})
        result = list(
            db.execute(
                "MATCH (n:Person) RETURN n.city, count(n) AS cnt GROUP BY n.city ORDER BY cnt DESC"
            )
        )
        assert result[0]["n.city"] == "Amsterdam"
        assert result[0]["cnt"] == 2

    def test_having_filter(self, db):
        """HAVING filters on aggregated results."""
        db.create_node(["Person"], {"name": "Alix", "city": "Amsterdam"})
        db.create_node(["Person"], {"name": "Gus", "city": "Amsterdam"})
        db.create_node(["Person"], {"name": "Vincent", "city": "Berlin"})
        result = list(db.execute("MATCH (n:Person) RETURN n.city, count(n) AS cnt HAVING cnt > 1"))
        assert len(result) == 1
        assert result[0]["n.city"] == "Amsterdam"


# =============================================================================
# CALL / Procedures (sec 15)
# =============================================================================


class TestCall:
    """Procedure calls: CALL ... YIELD, CALL { subquery }, OPTIONAL CALL."""

    def test_call_yield(self, db):
        """CALL named procedure with YIELD."""
        db.create_node(["Person"], {"name": "Alix"})
        result = list(db.execute("CALL db.labels() YIELD label"))
        labels = [r["label"] for r in result]
        assert "Person" in labels

    @pytest.mark.xfail(reason="Inline CALL { subquery } syntax not supported in GQL parser")
    def test_call_subquery(self, db):
        """CALL { subquery } inline block."""
        db.create_node(["Person"], {"name": "Alix"})
        db.create_node(["Person"], {"name": "Gus"})
        result = list(
            db.execute(
                "MATCH (n:Person) "
                "CALL { WITH n RETURN n.name AS upper_name } "
                "RETURN n.name, upper_name"
            )
        )
        assert len(result) == 2

    @pytest.mark.xfail(reason="Inline CALL { subquery } syntax not supported in GQL parser")
    def test_optional_call(self, db):
        """OPTIONAL CALL returns null when subquery yields no rows."""
        db.create_node(["Person"], {"name": "Alix"})
        result = list(
            db.execute(
                "MATCH (n:Person) "
                "OPTIONAL CALL { "
                "  WITH n MATCH (n)-[:MISSING]->(x) RETURN x.name AS found "
                "} "
                "RETURN n.name, found"
            )
        )
        assert len(result) >= 1
        assert result[0]["found"] is None
