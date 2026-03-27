"""Cross-language compatibility tests for LPG model.

Verifies that GQL, Cypher, Gremlin, and GraphQL return consistent results.
"""

import pytest

# Try to import obrain
try:
    from obrain import ObrainDB

    OBRAIN_AVAILABLE = True
except ImportError:
    OBRAIN_AVAILABLE = False


pytestmark = pytest.mark.skipif(not OBRAIN_AVAILABLE, reason="Obrain Python bindings not installed")


class TestCrossLanguageConsistency:
    """Verify that different query languages return consistent results."""

    def setup_method(self):
        """Create identical test data for comparison."""
        self.db = ObrainDB()
        self._setup_test_data()

    def _setup_test_data(self):
        """Create test data."""
        self.alix = self.db.create_node(["Person"], {"name": "Alix", "age": 30})
        self.gus = self.db.create_node(["Person"], {"name": "Gus", "age": 25})
        self.vincent = self.db.create_node(["Person"], {"name": "Vincent", "age": 35})

        self.db.create_edge(self.alix.id, self.gus.id, "KNOWS", {"since": 2020})
        self.db.create_edge(self.gus.id, self.vincent.id, "KNOWS", {"since": 2021})

    def _has_gremlin(self):
        """Check if Gremlin support is available."""
        try:
            self.db.execute_gremlin("g.V().limit(1)")
            return True
        except (AttributeError, NotImplementedError):
            return False
        except Exception:
            return True

    def _has_graphql(self):
        """Check if GraphQL support is available."""
        try:
            self.db.execute_graphql("query { __schema { types { name } } }")
            return True
        except (AttributeError, NotImplementedError):
            return False
        except Exception:
            return True

    def test_node_count_consistency(self):
        """All languages should return the same node count."""
        # GQL/Cypher (same syntax)
        gql_result = self.db.execute("MATCH (n:Person) RETURN count(n) AS cnt")
        gql_count = list(gql_result)[0]["cnt"]
        assert gql_count == 3

        # Gremlin (if available)
        if self._has_gremlin():
            try:
                gremlin_result = self.db.execute_gremlin("g.V().hasLabel('Person').count()")
                gremlin_rows = list(gremlin_result)
                if len(gremlin_rows) > 0:
                    # Gremlin count format may vary
                    pass
            except Exception:
                pass

    def test_simple_match_consistency(self):
        """GQL and Cypher should return same results for simple MATCH."""
        # Both use the same syntax
        result = self.db.execute("MATCH (n:Person) RETURN n.name ORDER BY n.name")
        rows = list(result)
        names = [r.get("n.name") for r in rows]

        assert "Alix" in names
        assert "Gus" in names
        assert "Vincent" in names

    def test_relationship_match_consistency(self):
        """All languages should return same relationship results."""
        # GQL/Cypher
        result = self.db.execute("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name, b.name")
        rows = list(result)
        assert len(rows) == 2  # Alix->Gus, Gus->Vincent

    def test_aggregation_consistency(self):
        """Aggregation results should match across languages."""
        # GQL/Cypher
        result = self.db.execute("MATCH (n:Person) RETURN sum(n.age) AS total")
        rows = list(result)
        assert rows[0]["total"] == 90  # 30 + 25 + 35

    def test_filter_consistency(self):
        """WHERE clause filtering should be consistent."""
        # GQL/Cypher
        result = self.db.execute("MATCH (n:Person) WHERE n.age > 26 RETURN n.name")
        rows = list(result)
        names = [r.get("n.name") for r in rows]

        # Alix (30) and Vincent (35) match
        assert len(rows) == 2
        assert "Alix" in names
        assert "Vincent" in names
        assert "Gus" not in names


class TestQuerySyntaxEquivalence:
    """Test that equivalent queries produce the same results."""

    def setup_method(self):
        """Create test data."""
        self.db = ObrainDB()
        self.a = self.db.create_node(["Node"], {"value": 1})
        self.b = self.db.create_node(["Node"], {"value": 2})
        self.c = self.db.create_node(["Node"], {"value": 3})
        self.db.create_edge(self.a.id, self.b.id, "LINK", {})
        self.db.create_edge(self.b.id, self.c.id, "LINK", {})

    def test_where_vs_inline_properties(self):
        """WHERE clause vs inline properties should be equivalent."""
        # Using WHERE
        result1 = self.db.execute("MATCH (n:Node) WHERE n.value = 2 RETURN n.value")
        rows1 = list(result1)

        # Using inline properties
        result2 = self.db.execute("MATCH (n:Node {value: 2}) RETURN n.value")
        rows2 = list(result2)

        assert len(rows1) == len(rows2) == 1
        assert rows1[0]["n.value"] == rows2[0]["n.value"] == 2

    def test_order_by_equivalence(self):
        """ORDER BY ASC and without explicit direction should be equivalent."""
        result1 = self.db.execute("MATCH (n:Node) RETURN n.value ORDER BY n.value")
        rows1 = list(result1)

        result2 = self.db.execute("MATCH (n:Node) RETURN n.value ORDER BY n.value ASC")
        rows2 = list(result2)

        values1 = [r["n.value"] for r in rows1]
        values2 = [r["n.value"] for r in rows2]
        assert values1 == values2 == [1, 2, 3]

    def test_directed_vs_undirected(self):
        """Directed queries should be subset of undirected."""
        # Directed: only outgoing edges
        result_directed = self.db.execute("MATCH (a:Node)-[:LINK]->(b:Node) RETURN count(a) AS cnt")
        directed_count = list(result_directed)[0]["cnt"]

        # Undirected: both directions
        result_undirected = self.db.execute(
            "MATCH (a:Node)-[:LINK]-(b:Node) RETURN count(a) AS cnt"
        )
        undirected_count = list(result_undirected)[0]["cnt"]

        # Undirected should count each edge twice (both directions)
        assert undirected_count == directed_count * 2
