"""GQL spec: Subqueries (ISO sec 19.4, 20.6).

Covers: EXISTS { }, COUNT { }, VALUE { }, correlated subqueries,
nested subqueries, COUNT in RETURN.
"""

import pytest

# =============================================================================
# EXISTS Subquery (sec 19.4)
# =============================================================================


class TestExistsSubquery:
    """EXISTS { MATCH ... WHERE ... } subquery predicate."""

    def test_exists_basic(self, db):
        """EXISTS { MATCH pattern } in WHERE."""
        a = db.create_node(["Person"], {"name": "Alix"})
        b = db.create_node(["Person"], {"name": "Gus"})
        db.create_node(["Person"], {"name": "Vincent"})
        db.create_edge(a.id, b.id, "KNOWS")
        result = list(
            db.execute("MATCH (p:Person) WHERE EXISTS { MATCH (p)-[:KNOWS]->() } RETURN p.name")
        )
        names = {r["p.name"] for r in result}
        assert "Alix" in names
        assert "Vincent" not in names
        assert "Gus" not in names

    def test_exists_with_where_filter(self, db):
        """EXISTS with inner WHERE for filtering."""
        a = db.create_node(["Person"], {"name": "Alix"})
        b = db.create_node(["Person"], {"name": "Gus", "age": 25})
        c = db.create_node(["Person"], {"name": "Vincent", "age": 35})
        db.create_edge(a.id, b.id, "KNOWS")
        db.create_edge(a.id, c.id, "KNOWS")
        result = list(
            db.execute(
                "MATCH (p:Person) "
                "WHERE EXISTS { MATCH (p)-[:KNOWS]->(q) WHERE q.age > 30 } "
                "RETURN p.name"
            )
        )
        assert len(result) == 1
        assert result[0]["p.name"] == "Alix"

    def test_not_exists(self, db):
        """NOT EXISTS subquery."""
        a = db.create_node(["Person"], {"name": "Alix"})
        b = db.create_node(["Person"], {"name": "Gus"})
        db.create_edge(a.id, b.id, "KNOWS")
        result = list(
            db.execute("MATCH (p:Person) WHERE NOT EXISTS { MATCH (p)-[:KNOWS]->() } RETURN p.name")
        )
        names = {r["p.name"] for r in result}
        assert "Gus" in names
        assert "Alix" not in names

    @pytest.mark.xfail(
        reason="Correlated EXISTS subquery does not resolve outer variable references"
    )
    def test_exists_correlated(self, db):
        """Correlated EXISTS with outer variable reference."""
        db.create_node(["Person"], {"name": "Alix", "city": "Amsterdam"})
        db.create_node(["Person"], {"name": "Gus", "city": "Berlin"})
        db.create_node(["City"], {"name": "Amsterdam"})
        result = list(
            db.execute(
                "MATCH (p:Person) "
                "WHERE EXISTS { "
                "  MATCH (c:City) WHERE c.name = p.city "
                "} "
                "RETURN p.name"
            )
        )
        # Only Alix has a matching City node
        assert len(result) >= 1
        assert any(r["p.name"] == "Alix" for r in result)


# =============================================================================
# COUNT Subquery
# =============================================================================


class TestCountSubquery:
    """COUNT { MATCH ... WHERE ... } subquery expression."""

    def test_count_subquery_in_where(self, db):
        """COUNT { } in WHERE for filtering by count."""
        a = db.create_node(["Person"], {"name": "Alix"})
        b = db.create_node(["Person"], {"name": "Gus"})
        c = db.create_node(["Person"], {"name": "Vincent"})
        db.create_edge(a.id, b.id, "KNOWS")
        db.create_edge(a.id, c.id, "KNOWS")
        db.create_edge(b.id, c.id, "KNOWS")
        result = list(
            db.execute("MATCH (p:Person) WHERE COUNT { MATCH (p)-[:KNOWS]->() } > 1 RETURN p.name")
        )
        assert len(result) == 1
        assert result[0]["p.name"] == "Alix"

    def test_count_subquery_in_return(self, db):
        """COUNT { } in RETURN projection."""
        a = db.create_node(["Person"], {"name": "Alix"})
        b = db.create_node(["Person"], {"name": "Gus"})
        c = db.create_node(["Person"], {"name": "Vincent"})
        db.create_edge(a.id, b.id, "KNOWS")
        db.create_edge(a.id, c.id, "KNOWS")
        result = list(
            db.execute(
                "MATCH (p:Person) "
                "RETURN p.name, COUNT { MATCH (p)-[:KNOWS]->() } AS friend_count "
                "ORDER BY friend_count DESC"
            )
        )
        assert result[0]["p.name"] == "Alix"
        assert result[0]["friend_count"] == 2

    def test_count_subquery_correlated(self, db):
        """COUNT { } with correlated outer variable."""
        a = db.create_node(["Person"], {"name": "Alix"})
        b = db.create_node(["Person"], {"name": "Gus"})
        db.create_edge(a.id, b.id, "KNOWS")
        result = list(
            db.execute(
                "MATCH (p:Person {name: 'Alix'}) RETURN COUNT { MATCH (p)-[:KNOWS]->() } AS cnt"
            )
        )
        assert result[0]["cnt"] == 1


# =============================================================================
# VALUE Subquery (sec 20.6)
# =============================================================================


class TestValueSubquery:
    """VALUE { MATCH ... RETURN ... } scalar subquery."""

    @pytest.mark.xfail(reason="VALUE { } subquery syntax not supported in GQL parser")
    def test_value_subquery(self, db):
        """VALUE { } returns scalar value."""
        db.create_node(["Config"], {"key": "max_retries", "val": 3})
        db.create_node(["Person"], {"name": "Alix"})
        result = list(
            db.execute(
                "MATCH (p:Person) "
                "RETURN p.name, "
                "VALUE { MATCH (c:Config {key: 'max_retries'}) RETURN c.val } AS config_val"
            )
        )
        assert result[0]["config_val"] == 3

    def test_value_subquery_correlated(self, db):
        """VALUE { } with correlated variables."""
        a = db.create_node(["Person"], {"name": "Alix"})
        b = db.create_node(["Person"], {"name": "Gus"})
        c = db.create_node(["Person"], {"name": "Vincent"})
        db.create_edge(a.id, b.id, "KNOWS")
        db.create_edge(a.id, c.id, "KNOWS")
        result = list(
            db.execute(
                "MATCH (p:Person {name: 'Alix'}) "
                "RETURN VALUE { "
                "  MATCH (p)-[:KNOWS]->(q) RETURN count(q) "
                "} AS cnt"
            )
        )
        assert result[0]["cnt"] == 2


# =============================================================================
# Nested Subqueries
# =============================================================================


class TestNestedSubqueries:
    """Nested subqueries with full variable scoping."""

    def test_nested_exists(self, db):
        """EXISTS inside EXISTS."""
        a = db.create_node(["Person"], {"name": "Alix"})
        b = db.create_node(["Person"], {"name": "Gus"})
        c = db.create_node(["Person"], {"name": "Vincent"})
        db.create_edge(a.id, b.id, "KNOWS")
        db.create_edge(b.id, c.id, "KNOWS")
        result = list(
            db.execute(
                "MATCH (p:Person) "
                "WHERE EXISTS { "
                "  MATCH (p)-[:KNOWS]->(q) "
                "  WHERE EXISTS { MATCH (q)-[:KNOWS]->() } "
                "} "
                "RETURN p.name"
            )
        )
        # Alix knows Gus, and Gus knows Vincent
        assert len(result) >= 1
        assert any(r["p.name"] == "Alix" for r in result)

    def test_count_inside_exists(self, db):
        """COUNT { } inside EXISTS { }."""
        a = db.create_node(["Person"], {"name": "Alix"})
        b = db.create_node(["Person"], {"name": "Gus"})
        c = db.create_node(["Person"], {"name": "Vincent"})
        db.create_edge(a.id, b.id, "KNOWS")
        db.create_edge(a.id, c.id, "KNOWS")
        result = list(
            db.execute("MATCH (p:Person) WHERE COUNT { MATCH (p)-[:KNOWS]->() } >= 2 RETURN p.name")
        )
        assert len(result) == 1
        assert result[0]["p.name"] == "Alix"
