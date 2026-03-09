"""Cypher spec: Reading Clauses (openCypher 9 sec 1).

Covers: MATCH, OPTIONAL MATCH, WHERE, WITH, UNWIND, UNION, UNION ALL,
CALL procedure YIELD, CALL { subquery }.
"""

import pytest

# =============================================================================
# MATCH
# =============================================================================


class TestMatch:
    """MATCH clause variants."""

    def test_match_single_node(self, db):
        """MATCH (n) returns all nodes."""
        db.create_node(["Person"], {"name": "Alix"})
        db.create_node(["Person"], {"name": "Gus"})
        result = list(db.execute_cypher("MATCH (n) RETURN count(n) AS cnt"))
        assert result[0]["cnt"] == 2

    def test_match_by_label(self, db):
        """MATCH (n:Label) filters by label."""
        db.create_node(["Person"], {"name": "Alix"})
        db.create_node(["Animal"], {"name": "Rex"})
        result = list(db.execute_cypher("MATCH (n:Person) RETURN n.name"))
        assert len(result) == 1
        assert result[0]["n.name"] == "Alix"

    def test_match_by_property(self, db):
        """MATCH (n {prop: val}) inline property filter."""
        db.create_node(["Person"], {"name": "Alix", "age": 30})
        db.create_node(["Person"], {"name": "Gus", "age": 25})
        result = list(db.execute_cypher("MATCH (n:Person {name: 'Alix'}) RETURN n.age"))
        assert result[0]["n.age"] == 30

    def test_match_multi_label(self, db):
        """MATCH (n:L1:L2) multi-label filter."""
        db.create_node(["Person", "Developer"], {"name": "Alix"})
        db.create_node(["Person"], {"name": "Gus"})
        result = list(db.execute_cypher("MATCH (n:Person:Developer) RETURN n.name"))
        assert len(result) >= 1
        assert any(r["n.name"] == "Alix" for r in result)

    def test_match_comma_patterns(self, db):
        """MATCH (a), (b) cross-product."""
        db.create_node(["Person"], {"name": "Alix"})
        db.create_node(["Company"], {"name": "Acme"})
        result = list(db.execute_cypher("MATCH (p:Person), (c:Company) RETURN p.name, c.name"))
        assert len(result) == 1
        assert result[0]["p.name"] == "Alix"

    def test_match_multiple_clauses(self, db):
        """Two sequential MATCH clauses."""
        db.create_node(["Person"], {"name": "Alix"})
        db.create_node(["City"], {"name": "Amsterdam"})
        result = list(db.execute_cypher("MATCH (p:Person) MATCH (c:City) RETURN p.name, c.name"))
        assert len(result) == 1
        assert result[0]["c.name"] == "Amsterdam"

    def test_match_edge_outgoing(self, db):
        """MATCH (a)-[:TYPE]->(b) directed edge."""
        a = db.create_node(["Person"], {"name": "Alix"})
        b = db.create_node(["Person"], {"name": "Gus"})
        db.create_edge(a.id, b.id, "KNOWS")
        result = list(
            db.execute_cypher("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name, b.name")
        )
        assert len(result) == 1
        assert result[0]["a.name"] == "Alix"
        assert result[0]["b.name"] == "Gus"

    def test_match_edge_incoming(self, db):
        """MATCH (a)<-[:TYPE]-(b) incoming edge."""
        a = db.create_node(["Person"], {"name": "Alix"})
        b = db.create_node(["Person"], {"name": "Gus"})
        db.create_edge(a.id, b.id, "KNOWS")
        result = list(
            db.execute_cypher("MATCH (b:Person)<-[:KNOWS]-(a:Person) RETURN a.name, b.name")
        )
        assert len(result) == 1
        assert result[0]["a.name"] == "Alix"

    def test_match_edge_undirected(self, db):
        """MATCH (a)-[:TYPE]-(b) undirected edge."""
        a = db.create_node(["Person"], {"name": "Alix"})
        b = db.create_node(["Person"], {"name": "Gus"})
        db.create_edge(a.id, b.id, "KNOWS")
        result = list(
            db.execute_cypher("MATCH (a:Person {name: 'Alix'})-[:KNOWS]-(b) RETURN b.name")
        )
        assert len(result) >= 1
        assert any(r["b.name"] == "Gus" for r in result)


# =============================================================================
# OPTIONAL MATCH
# =============================================================================


class TestOptionalMatch:
    """OPTIONAL MATCH returns nulls when no match."""

    def test_optional_match_with_result(self, db):
        """OPTIONAL MATCH returns rows when relationship exists."""
        a = db.create_node(["Person"], {"name": "Alix"})
        b = db.create_node(["Person"], {"name": "Gus"})
        db.create_edge(a.id, b.id, "KNOWS")
        result = list(
            db.execute_cypher(
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
            db.execute_cypher(
                "MATCH (a:Person {name: 'Alix'}) "
                "OPTIONAL MATCH (a)-[:MANAGES]->(b) "
                "RETURN a.name, b"
            )
        )
        assert len(result) == 1
        assert result[0]["b"] is None


# =============================================================================
# WHERE
# =============================================================================


class TestWhere:
    """WHERE clause filtering."""

    def test_where_comparison(self, db):
        db.create_node(["Person"], {"name": "Alix", "age": 30})
        db.create_node(["Person"], {"name": "Gus", "age": 25})
        result = list(db.execute_cypher("MATCH (n:Person) WHERE n.age > 28 RETURN n.name"))
        assert len(result) == 1
        assert result[0]["n.name"] == "Alix"

    def test_where_and(self, db):
        db.create_node(["Person"], {"name": "Alix", "age": 30, "city": "Amsterdam"})
        db.create_node(["Person"], {"name": "Gus", "age": 25, "city": "Berlin"})
        result = list(
            db.execute_cypher(
                "MATCH (n:Person) WHERE n.age > 20 AND n.city = 'Amsterdam' RETURN n.name"
            )
        )
        assert len(result) == 1
        assert result[0]["n.name"] == "Alix"

    def test_where_or(self, db):
        db.create_node(["Person"], {"name": "Alix"})
        db.create_node(["Person"], {"name": "Gus"})
        db.create_node(["Person"], {"name": "Vincent"})
        result = list(
            db.execute_cypher(
                "MATCH (n:Person) WHERE n.name = 'Alix' OR n.name = 'Gus' RETURN n.name"
            )
        )
        assert len(result) == 2

    def test_where_not(self, db):
        db.create_node(["Person"], {"name": "Alix", "city": "Amsterdam"})
        db.create_node(["Person"], {"name": "Gus", "city": "Berlin"})
        result = list(
            db.execute_cypher("MATCH (n:Person) WHERE NOT n.city = 'Amsterdam' RETURN n.name")
        )
        assert len(result) == 1
        assert result[0]["n.name"] == "Gus"

    def test_where_xor(self, db):
        db.create_node(["Person"], {"name": "Alix", "active": True, "admin": False})
        db.create_node(["Person"], {"name": "Gus", "active": True, "admin": True})
        db.create_node(["Person"], {"name": "Vincent", "active": False, "admin": False})
        result = list(
            db.execute_cypher("MATCH (n:Person) WHERE n.active XOR n.admin RETURN n.name")
        )
        assert len(result) == 1
        assert result[0]["n.name"] == "Alix"


# =============================================================================
# WITH
# =============================================================================


class TestWith:
    """WITH clause for pipelining."""

    def test_with_projection(self, db):
        db.create_node(["Person"], {"name": "Alix", "age": 30})
        result = list(
            db.execute_cypher("MATCH (n:Person) WITH n.name AS name, n.age AS age RETURN name, age")
        )
        assert result[0]["name"] == "Alix"

    def test_with_distinct(self, db):
        db.create_node(["Person"], {"name": "Alix", "city": "Amsterdam"})
        db.create_node(["Person"], {"name": "Gus", "city": "Amsterdam"})
        result = list(
            db.execute_cypher("MATCH (n:Person) WITH DISTINCT n.city AS city RETURN city")
        )
        assert len(result) == 1

    def test_with_where(self, db):
        db.create_node(["Person"], {"name": "Alix", "age": 30})
        db.create_node(["Person"], {"name": "Gus", "age": 25})
        result = list(db.execute_cypher("MATCH (n:Person) WITH n WHERE n.age > 28 RETURN n.name"))
        assert len(result) == 1
        assert result[0]["n.name"] == "Alix"

    @pytest.mark.xfail(reason="WITH * not supported in Cypher parser")
    def test_with_star(self, db):
        db.create_node(["Person"], {"name": "Alix"})
        result = list(db.execute_cypher("MATCH (n:Person) WITH * RETURN n.name"))
        assert result[0]["n.name"] == "Alix"


# =============================================================================
# UNWIND
# =============================================================================


class TestUnwind:
    """UNWIND list expansion."""

    def test_unwind_list(self, db):
        result = list(db.execute_cypher("UNWIND [1, 2, 3] AS x RETURN x"))
        vals = [r["x"] for r in result]
        assert vals == [1, 2, 3]

    def test_unwind_with_match(self, db):
        db.create_node(["Person"], {"name": "Alix"})
        db.create_node(["Person"], {"name": "Gus"})
        result = list(
            db.execute_cypher(
                "UNWIND ['Alix', 'Gus'] AS name MATCH (n:Person {name: name}) RETURN n.name"
            )
        )
        assert len(result) == 2


# =============================================================================
# UNION
# =============================================================================


class TestUnion:
    """UNION and UNION ALL."""

    def test_union(self, db):
        """UNION deduplicates results."""
        db.create_node(["Person"], {"name": "Alix"})
        db.create_node(["City"], {"name": "Amsterdam"})
        result = list(
            db.execute_cypher(
                "MATCH (n:Person) RETURN n.name AS name UNION MATCH (c:City) RETURN c.name AS name"
            )
        )
        names = {r["name"] for r in result}
        assert "Alix" in names
        assert "Amsterdam" in names

    def test_union_all(self, db):
        """UNION ALL keeps duplicates."""
        db.create_node(["Person"], {"name": "Alix"})
        result = list(
            db.execute_cypher(
                "MATCH (n:Person) RETURN n.name AS name "
                "UNION ALL "
                "MATCH (n:Person) RETURN n.name AS name"
            )
        )
        assert len(result) == 2


# =============================================================================
# CALL procedure
# =============================================================================


class TestCallProcedure:
    """CALL procedure(args) YIELD ..."""

    def test_call_db_labels(self, db):
        db.create_node(["Person"], {"name": "Alix"})
        db.create_node(["City"], {"name": "Amsterdam"})
        result = list(db.execute_cypher("CALL db.labels() YIELD label"))
        labels = {r["label"] for r in result}
        assert "Person" in labels
        assert "City" in labels

    def test_call_db_relationship_types(self, db):
        a = db.create_node(["Person"], {"name": "Alix"})
        b = db.create_node(["Person"], {"name": "Gus"})
        db.create_edge(a.id, b.id, "KNOWS")
        result = list(db.execute_cypher("CALL db.relationshipTypes() YIELD relationshipType"))
        types = {r["relationshipType"] for r in result}
        assert "KNOWS" in types

    def test_call_db_property_keys(self, db):
        db.create_node(["Person"], {"name": "Alix", "age": 30})
        result = list(db.execute_cypher("CALL db.propertyKeys() YIELD propertyKey"))
        keys = {r["propertyKey"] for r in result}
        assert "name" in keys
        assert "age" in keys

    def test_call_with_alias(self, db):
        db.create_node(["Person"], {"name": "Alix"})
        result = list(db.execute_cypher("CALL db.labels() YIELD label AS l"))
        assert any(r["l"] == "Person" for r in result)


# =============================================================================
# CALL { subquery }
# =============================================================================


class TestCallSubquery:
    """CALL { ... } inline subquery block."""

    def test_call_subquery_basic(self, db):
        """CALL { subquery } with aggregation."""
        db.create_node(["Person"], {"name": "Alix"})
        db.create_node(["Person"], {"name": "Gus"})
        result = list(
            db.execute_cypher("CALL { MATCH (n:Person) RETURN count(n) AS total } RETURN total")
        )
        assert result[0]["total"] == 2

    @pytest.mark.xfail(reason="CALL { WITH outer_var ... } does not resolve outer scope variables")
    def test_call_subquery_with_outer_scope(self, db):
        """CALL { } with outer scope variable propagation."""
        a = db.create_node(["Person"], {"name": "Alix"})
        b = db.create_node(["Person"], {"name": "Gus"})
        db.create_edge(a.id, b.id, "KNOWS")
        result = list(
            db.execute_cypher(
                "MATCH (p:Person {name: 'Alix'}) "
                "CALL { WITH p MATCH (p)-[:KNOWS]->(q) RETURN q.name AS friend } "
                "RETURN p.name, friend"
            )
        )
        assert len(result) == 1
        assert result[0]["friend"] == "Gus"
