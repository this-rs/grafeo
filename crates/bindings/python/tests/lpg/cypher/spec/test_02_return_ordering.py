"""Cypher spec: Return and Ordering (openCypher 9 sec 2).

Covers: RETURN, RETURN DISTINCT, RETURN *, ORDER BY, SKIP, LIMIT,
RETURN expressions, aliases, aggregation in RETURN.
"""

import pytest

# =============================================================================
# RETURN
# =============================================================================


class TestReturn:
    """RETURN clause variants."""

    def test_return_expression(self, db):
        db.create_node(["Person"], {"name": "Alix", "age": 30})
        result = list(db.execute_cypher("MATCH (n:Person) RETURN n.name, n.age"))
        assert result[0]["n.name"] == "Alix"
        assert result[0]["n.age"] == 30

    def test_return_alias(self, db):
        db.create_node(["Person"], {"name": "Alix"})
        result = list(db.execute_cypher("MATCH (n:Person) RETURN n.name AS person_name"))
        assert result[0]["person_name"] == "Alix"

    @pytest.mark.xfail(reason="RETURN DISTINCT does not deduplicate results")
    def test_return_distinct(self, db):
        db.create_node(["Person"], {"name": "Alix", "city": "Amsterdam"})
        db.create_node(["Person"], {"name": "Gus", "city": "Amsterdam"})
        result = list(db.execute_cypher("MATCH (n:Person) RETURN DISTINCT n.city"))
        assert len(result) == 1

    def test_return_star(self, db):
        db.create_node(["Person"], {"name": "Alix"})
        result = list(db.execute_cypher("MATCH (n:Person) RETURN *"))
        assert len(result) == 1

    def test_return_count_star(self, db):
        """count(*) works in Cypher (unlike GQL)."""
        db.create_node(["Person"], {"name": "Alix"})
        db.create_node(["Person"], {"name": "Gus"})
        result = list(db.execute_cypher("MATCH (n:Person) RETURN count(*) AS cnt"))
        assert result[0]["cnt"] == 2

    def test_return_arithmetic(self, db):
        db.create_node(["N"], {"v": 10})
        result = list(db.execute_cypher("MATCH (n:N) RETURN n.v + 5 AS r"))
        assert result[0]["r"] == 15

    def test_return_boolean_expression(self, db):
        db.create_node(["N"], {"v": 10})
        result = list(db.execute_cypher("MATCH (n:N) RETURN n.v > 5 AS r"))
        assert result[0]["r"] is True


# =============================================================================
# ORDER BY
# =============================================================================


class TestOrderBy:
    """ORDER BY clause."""

    def test_order_by_asc(self, db):
        db.create_node(["Person"], {"name": "Gus", "age": 25})
        db.create_node(["Person"], {"name": "Alix", "age": 30})
        result = list(db.execute_cypher("MATCH (n:Person) RETURN n.name ORDER BY n.age ASC"))
        assert result[0]["n.name"] == "Gus"
        assert result[1]["n.name"] == "Alix"

    def test_order_by_desc(self, db):
        db.create_node(["Person"], {"name": "Gus", "age": 25})
        db.create_node(["Person"], {"name": "Alix", "age": 30})
        result = list(db.execute_cypher("MATCH (n:Person) RETURN n.name ORDER BY n.age DESC"))
        assert result[0]["n.name"] == "Alix"

    def test_order_by_multiple_keys(self, db):
        db.create_node(["Person"], {"name": "Alix", "city": "Amsterdam", "age": 30})
        db.create_node(["Person"], {"name": "Gus", "city": "Amsterdam", "age": 25})
        db.create_node(["Person"], {"name": "Vincent", "city": "Berlin", "age": 35})
        result = list(
            db.execute_cypher("MATCH (n:Person) RETURN n.name ORDER BY n.city ASC, n.age DESC")
        )
        assert result[0]["n.name"] == "Alix"
        assert result[1]["n.name"] == "Gus"
        assert result[2]["n.name"] == "Vincent"


# =============================================================================
# SKIP / LIMIT
# =============================================================================


class TestSkipLimit:
    """SKIP and LIMIT pagination."""

    def test_limit(self, db):
        for i in range(5):
            db.create_node(["Item"], {"val": i})
        result = list(db.execute_cypher("MATCH (n:Item) RETURN n.val LIMIT 3"))
        assert len(result) == 3

    def test_skip(self, db):
        for i in range(5):
            db.create_node(["Item"], {"val": i})
        result = list(db.execute_cypher("MATCH (n:Item) RETURN n.val ORDER BY n.val SKIP 3"))
        assert len(result) == 2

    def test_skip_and_limit(self, db):
        for i in range(5):
            db.create_node(["Item"], {"val": i})
        result = list(
            db.execute_cypher("MATCH (n:Item) RETURN n.val ORDER BY n.val SKIP 1 LIMIT 2")
        )
        assert len(result) == 2
        assert result[0]["n.val"] == 1
        assert result[1]["n.val"] == 2

    def test_skip_with_parameter(self, db):
        """SKIP with parameter expression."""
        for i in range(5):
            db.create_node(["Item"], {"val": i})
        result = list(
            db.execute_cypher(
                "MATCH (n:Item) RETURN n.val ORDER BY n.val SKIP $s LIMIT $l",
                {"s": 2, "l": 2},
            )
        )
        assert len(result) == 2
        assert result[0]["n.val"] == 2
