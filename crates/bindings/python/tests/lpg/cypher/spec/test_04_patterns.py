"""Cypher spec: Pattern Matching (openCypher 9 sec 4).

Covers: Node patterns, relationship patterns, variable-length paths,
named paths, shortestPath, allShortestPaths, pattern comprehensions,
EXISTS/COUNT subquery patterns.
"""

import pytest

# =============================================================================
# Node Patterns (sec 4.1)
# =============================================================================


class TestNodePatterns:
    """Node pattern syntax variants."""

    def test_anonymous_node(self, db):
        """() anonymous node."""
        db.create_node(["Person"], {"name": "Alix"})
        result = list(db.execute_cypher("MATCH () RETURN count(*) AS cnt"))
        assert result[0]["cnt"] == 1

    def test_variable_binding(self, db):
        """(n) binds variable."""
        db.create_node(["Person"], {"name": "Alix"})
        result = list(db.execute_cypher("MATCH (n) RETURN n.name"))
        assert result[0]["n.name"] == "Alix"

    def test_single_label(self, db):
        db.create_node(["Person"], {"name": "Alix"})
        db.create_node(["Animal"], {"name": "Rex"})
        result = list(db.execute_cypher("MATCH (n:Person) RETURN n.name"))
        assert len(result) == 1

    def test_multiple_labels(self, db):
        db.create_node(["Person", "Developer"], {"name": "Alix"})
        db.create_node(["Person"], {"name": "Gus"})
        result = list(db.execute_cypher("MATCH (n:Person:Developer) RETURN n.name"))
        assert len(result) >= 1
        assert any(r["n.name"] == "Alix" for r in result)

    def test_property_filter(self, db):
        db.create_node(["Person"], {"name": "Alix", "age": 30})
        db.create_node(["Person"], {"name": "Gus", "age": 25})
        result = list(db.execute_cypher("MATCH (n:Person {age: 30}) RETURN n.name"))
        assert result[0]["n.name"] == "Alix"


# =============================================================================
# Relationship Patterns (sec 4.2)
# =============================================================================


class TestRelationshipPatterns:
    """Relationship pattern syntax variants."""

    def test_outgoing(self, db):
        a = db.create_node(["Person"], {"name": "Alix"})
        b = db.create_node(["Person"], {"name": "Gus"})
        db.create_edge(a.id, b.id, "KNOWS")
        result = list(db.execute_cypher("MATCH (a)-[r:KNOWS]->(b) RETURN a.name, b.name"))
        assert result[0]["a.name"] == "Alix"

    def test_incoming(self, db):
        a = db.create_node(["Person"], {"name": "Alix"})
        b = db.create_node(["Person"], {"name": "Gus"})
        db.create_edge(a.id, b.id, "KNOWS")
        result = list(db.execute_cypher("MATCH (b)<-[r:KNOWS]-(a) RETURN a.name"))
        assert result[0]["a.name"] == "Alix"

    def test_undirected(self, db):
        a = db.create_node(["Person"], {"name": "Alix"})
        b = db.create_node(["Person"], {"name": "Gus"})
        db.create_edge(a.id, b.id, "KNOWS")
        result = list(
            db.execute_cypher("MATCH (a:Person {name: 'Gus'})-[r:KNOWS]-(b) RETURN b.name")
        )
        assert any(r["b.name"] == "Alix" for r in result)

    def test_multiple_types(self, db):
        """-[r:T1|T2]-> matches multiple relationship types."""
        a = db.create_node(["Person"], {"name": "Alix"})
        b = db.create_node(["Person"], {"name": "Gus"})
        c = db.create_node(["Person"], {"name": "Vincent"})
        db.create_edge(a.id, b.id, "KNOWS")
        db.create_edge(a.id, c.id, "FOLLOWS")
        result = list(
            db.execute_cypher("MATCH (a:Person {name: 'Alix'})-[:KNOWS|FOLLOWS]->(b) RETURN b.name")
        )
        names = {r["b.name"] for r in result}
        assert "Gus" in names
        assert "Vincent" in names

    def test_relationship_properties(self, db):
        a = db.create_node(["Person"], {"name": "Alix"})
        b = db.create_node(["Person"], {"name": "Gus"})
        db.create_edge(a.id, b.id, "KNOWS", {"since": 2020})
        result = list(
            db.execute_cypher("MATCH (a)-[r:KNOWS {since: 2020}]->(b) RETURN a.name, r.since")
        )
        assert result[0]["r.since"] == 2020

    def test_untyped_relationship(self, db):
        a = db.create_node(["Person"], {"name": "Alix"})
        b = db.create_node(["Person"], {"name": "Gus"})
        db.create_edge(a.id, b.id, "KNOWS")
        result = list(
            db.execute_cypher("MATCH (a:Person {name: 'Alix'})-[r]->(b) RETURN type(r) AS t")
        )
        assert result[0]["t"] == "KNOWS"

    @pytest.mark.xfail(reason="--> anonymous relationship shorthand not parsed")
    def test_anonymous_relationship(self, db):
        a = db.create_node(["Person"], {"name": "Alix"})
        b = db.create_node(["Person"], {"name": "Gus"})
        db.create_edge(a.id, b.id, "KNOWS")
        result = list(db.execute_cypher("MATCH (a:Person {name: 'Alix'})-->(b) RETURN b.name"))
        assert result[0]["b.name"] == "Gus"


# =============================================================================
# Variable-Length Paths (sec 4.3)
# =============================================================================


class TestVariableLengthPaths:
    """Variable-length path patterns."""

    def test_unbounded(self, db):
        """-[*]-> unbounded path."""
        a = db.create_node(["Person"], {"name": "Alix"})
        b = db.create_node(["Person"], {"name": "Gus"})
        c = db.create_node(["Person"], {"name": "Vincent"})
        db.create_edge(a.id, b.id, "KNOWS")
        db.create_edge(b.id, c.id, "KNOWS")
        result = list(
            db.execute_cypher("MATCH (a:Person {name: 'Alix'})-[:KNOWS*]->(b) RETURN b.name")
        )
        names = {r["b.name"] for r in result}
        assert "Gus" in names
        assert "Vincent" in names

    def test_exact_length(self, db):
        """-[*3]-> exact length path."""
        a = db.create_node(["N"], {"v": 1})
        b = db.create_node(["N"], {"v": 2})
        c = db.create_node(["N"], {"v": 3})
        d = db.create_node(["N"], {"v": 4})
        db.create_edge(a.id, b.id, "NEXT")
        db.create_edge(b.id, c.id, "NEXT")
        db.create_edge(c.id, d.id, "NEXT")
        result = list(db.execute_cypher("MATCH (a:N {v: 1})-[:NEXT*3]->(b) RETURN b.v"))
        assert len(result) == 1
        assert result[0]["b.v"] == 4

    def test_range(self, db):
        """-[*1..3]-> range path."""
        a = db.create_node(["N"], {"v": 1})
        b = db.create_node(["N"], {"v": 2})
        c = db.create_node(["N"], {"v": 3})
        db.create_edge(a.id, b.id, "NEXT")
        db.create_edge(b.id, c.id, "NEXT")
        result = list(db.execute_cypher("MATCH (a:N {v: 1})-[:NEXT*1..2]->(b) RETURN b.v"))
        vals = {r["b.v"] for r in result}
        assert 2 in vals
        assert 3 in vals

    def test_max_only(self, db):
        """-[*..3]-> max only."""
        a = db.create_node(["N"], {"v": 1})
        b = db.create_node(["N"], {"v": 2})
        c = db.create_node(["N"], {"v": 3})
        db.create_edge(a.id, b.id, "NEXT")
        db.create_edge(b.id, c.id, "NEXT")
        result = list(db.execute_cypher("MATCH (a:N {v: 1})-[:NEXT*..2]->(b) RETURN b.v"))
        vals = {r["b.v"] for r in result}
        assert 2 in vals
        assert 3 in vals

    def test_min_only(self, db):
        """-[*2..]-> min only."""
        a = db.create_node(["N"], {"v": 1})
        b = db.create_node(["N"], {"v": 2})
        c = db.create_node(["N"], {"v": 3})
        db.create_edge(a.id, b.id, "NEXT")
        db.create_edge(b.id, c.id, "NEXT")
        result = list(db.execute_cypher("MATCH (a:N {v: 1})-[:NEXT*2..]->(b) RETURN b.v"))
        assert len(result) == 1
        assert result[0]["b.v"] == 3


# =============================================================================
# Named Paths (sec 4.4)
# =============================================================================


class TestNamedPaths:
    """Named path patterns and path functions."""

    def test_path_alias(self, db):
        """p = (a)-[*]->(b) path alias."""
        a = db.create_node(["Person"], {"name": "Alix"})
        b = db.create_node(["Person"], {"name": "Gus"})
        db.create_edge(a.id, b.id, "KNOWS")
        result = list(
            db.execute_cypher(
                "MATCH p = (a:Person {name: 'Alix'})-[:KNOWS]->(b) RETURN length(p) AS len"
            )
        )
        assert result[0]["len"] == 1

    def test_shortest_path(self, db):
        """shortestPath((a)-[*]->(b))."""
        a = db.create_node(["N"], {"v": 1})
        b = db.create_node(["N"], {"v": 2})
        c = db.create_node(["N"], {"v": 3})
        db.create_edge(a.id, b.id, "NEXT")
        db.create_edge(b.id, c.id, "NEXT")
        db.create_edge(a.id, c.id, "NEXT")  # direct shortcut
        result = list(
            db.execute_cypher(
                "MATCH p = shortestPath("
                "(a:N {v: 1})-[:NEXT*]->(b:N {v: 3})"
                ") RETURN length(p) AS len"
            )
        )
        assert result[0]["len"] == 1  # direct path

    def test_all_shortest_paths(self, db):
        """allShortestPaths((a)-[*]->(b))."""
        a = db.create_node(["N"], {"v": 1})
        b = db.create_node(["N"], {"v": 2})
        c = db.create_node(["N"], {"v": 3})
        db.create_edge(a.id, b.id, "NEXT")
        db.create_edge(b.id, c.id, "NEXT")
        db.create_edge(a.id, c.id, "NEXT")
        result = list(
            db.execute_cypher(
                "MATCH p = allShortestPaths("
                "(a:N {v: 1})-[:NEXT*]->(b:N {v: 3})"
                ") RETURN length(p) AS len"
            )
        )
        assert all(r["len"] == 1 for r in result)


# =============================================================================
# Advanced Patterns (sec 4.5)
# =============================================================================


class TestAdvancedPatterns:
    """Pattern comprehensions, EXISTS, COUNT subqueries."""

    def test_pattern_comprehension(self, db):
        """[(a)-->(b) | b.prop] pattern comprehension."""
        a = db.create_node(["Person"], {"name": "Alix"})
        b = db.create_node(["Person"], {"name": "Gus"})
        c = db.create_node(["Person"], {"name": "Vincent"})
        db.create_edge(a.id, b.id, "KNOWS")
        db.create_edge(a.id, c.id, "KNOWS")
        result = list(
            db.execute_cypher(
                "MATCH (a:Person {name: 'Alix'}) RETURN [(a)-[:KNOWS]->(b) | b.name] AS friends"
            )
        )
        friends = sorted(result[0]["friends"])
        assert friends == ["Gus", "Vincent"]

    def test_exists_subquery(self, db):
        """EXISTS { MATCH ... } subquery pattern."""
        a = db.create_node(["Person"], {"name": "Alix"})
        b = db.create_node(["Person"], {"name": "Gus"})
        db.create_node(["Person"], {"name": "Vincent"})
        db.create_edge(a.id, b.id, "KNOWS")
        result = list(
            db.execute_cypher(
                "MATCH (p:Person) WHERE EXISTS { MATCH (p)-[:KNOWS]->() } RETURN p.name"
            )
        )
        names = {r["p.name"] for r in result}
        assert "Alix" in names
        assert "Vincent" not in names

    def test_not_exists(self, db):
        """NOT EXISTS { MATCH ... }."""
        a = db.create_node(["Person"], {"name": "Alix"})
        b = db.create_node(["Person"], {"name": "Gus"})
        db.create_edge(a.id, b.id, "KNOWS")
        result = list(
            db.execute_cypher(
                "MATCH (p:Person) WHERE NOT EXISTS { MATCH (p)-[:KNOWS]->() } RETURN p.name"
            )
        )
        names = {r["p.name"] for r in result}
        assert "Gus" in names
        assert "Alix" not in names

    def test_count_subquery(self, db):
        """COUNT { MATCH ... } subquery."""
        a = db.create_node(["Person"], {"name": "Alix"})
        b = db.create_node(["Person"], {"name": "Gus"})
        c = db.create_node(["Person"], {"name": "Vincent"})
        db.create_edge(a.id, b.id, "KNOWS")
        db.create_edge(a.id, c.id, "KNOWS")
        result = list(
            db.execute_cypher(
                "MATCH (p:Person) WHERE COUNT { MATCH (p)-[:KNOWS]->() } > 1 RETURN p.name"
            )
        )
        assert len(result) == 1
        assert result[0]["p.name"] == "Alix"
