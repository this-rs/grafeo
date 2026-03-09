"""GQL spec: Procedures (ISO sec 9, 10, 15).

Covers: Named procedure CALL with YIELD, inline CALL { subquery },
OPTIONAL CALL, CREATE PROCEDURE, DROP PROCEDURE, built-in catalog procedures.
"""

import pytest

# =============================================================================
# Named Procedure CALL (sec 15)
# =============================================================================


class TestNamedProcedureCall:
    """CALL procedure(args) YIELD ..."""

    def test_call_db_labels(self, db):
        """CALL db.labels() YIELD label."""
        db.create_node(["Person"], {"name": "Alix"})
        db.create_node(["City"], {"name": "Amsterdam"})
        result = list(db.execute("CALL db.labels() YIELD label"))
        labels = {r["label"] for r in result}
        assert "Person" in labels
        assert "City" in labels

    def test_call_db_relationship_types(self, db):
        """CALL db.relationshipTypes() YIELD relationshipType."""
        a = db.create_node(["Person"], {"name": "Alix"})
        b = db.create_node(["Person"], {"name": "Gus"})
        db.create_edge(a.id, b.id, "KNOWS")
        result = list(db.execute("CALL db.relationshipTypes() YIELD relationshipType"))
        types = {r["relationshipType"] for r in result}
        assert "KNOWS" in types

    def test_call_db_property_keys(self, db):
        """CALL db.propertyKeys() YIELD propertyKey."""
        db.create_node(["Person"], {"name": "Alix", "age": 30})
        result = list(db.execute("CALL db.propertyKeys() YIELD propertyKey"))
        keys = {r["propertyKey"] for r in result}
        assert "name" in keys
        assert "age" in keys

    @pytest.mark.xfail(reason="WHERE filtering on CALL YIELD may not work in GQL parser")
    def test_call_with_where(self, db):
        """CALL ... YIELD ... WHERE filters yielded rows."""
        db.create_node(["Person"], {"name": "Alix"})
        db.create_node(["City"], {"name": "Amsterdam"})
        result = list(
            db.execute("CALL db.labels() YIELD label WHERE label = 'Person' RETURN label")
        )
        assert len(result) == 1
        assert result[0]["label"] == "Person"

    def test_call_with_alias(self, db):
        """CALL ... YIELD field AS alias."""
        db.create_node(["Person"], {"name": "Alix"})
        result = list(db.execute("CALL db.labels() YIELD label AS l"))
        assert any(r["l"] == "Person" for r in result)


# =============================================================================
# Inline CALL { subquery } (sec 15)
# =============================================================================


class TestInlineCall:
    """CALL { ... } inline subquery block."""

    @pytest.mark.xfail(reason="Inline CALL { subquery } syntax not supported in GQL parser")
    def test_call_subquery_basic(self, db):
        """CALL { subquery } with WITH propagation."""
        db.create_node(["Person"], {"name": "Alix", "age": 30})
        db.create_node(["Person"], {"name": "Gus", "age": 25})
        result = list(
            db.execute(
                "MATCH (n:Person) "
                "CALL { WITH n RETURN n.name AS upper_name } "
                "RETURN n.name, upper_name"
            )
        )
        assert len(result) == 2

    @pytest.mark.xfail(reason="Inline CALL { subquery } syntax not supported in GQL parser")
    def test_call_subquery_aggregation(self, db):
        """CALL { } with aggregation inside."""
        db.create_node(["Person"], {"name": "Alix", "age": 30})
        db.create_node(["Person"], {"name": "Gus", "age": 25})
        result = list(db.execute("CALL { MATCH (n:Person) RETURN count(n) AS total } RETURN total"))
        assert result[0]["total"] == 2


# =============================================================================
# OPTIONAL CALL (sec 15)
# =============================================================================


class TestOptionalCall:
    """OPTIONAL CALL returns null when subquery yields no rows."""

    @pytest.mark.xfail(reason="Inline CALL { subquery } syntax not supported in GQL parser")
    def test_optional_call_no_match(self, db):
        """OPTIONAL CALL with empty subquery result."""
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
        assert len(result) == 1
        assert result[0]["found"] is None

    @pytest.mark.xfail(reason="Inline CALL { subquery } syntax not supported in GQL parser")
    def test_optional_call_with_match(self, db):
        """OPTIONAL CALL with matching subquery result."""
        a = db.create_node(["Person"], {"name": "Alix"})
        b = db.create_node(["Person"], {"name": "Gus"})
        db.create_edge(a.id, b.id, "KNOWS")
        result = list(
            db.execute(
                "MATCH (n:Person {name: 'Alix'}) "
                "OPTIONAL CALL { "
                "  WITH n MATCH (n)-[:KNOWS]->(x) RETURN x.name AS found "
                "} "
                "RETURN n.name, found"
            )
        )
        assert result[0]["found"] == "Gus"


# =============================================================================
# CREATE / DROP PROCEDURE
# =============================================================================


class TestCreateProcedure:
    """User-defined procedures."""

    @pytest.mark.xfail(reason="CREATE PROCEDURE not supported in GQL parser")
    def test_create_procedure(self, db):
        """CREATE PROCEDURE with body."""
        db.execute(
            "CREATE PROCEDURE my_proc() "
            "RETURNS (greeting STRING) "
            "BEGIN "
            "  RETURN 'hello' AS greeting "
            "END"
        )
        result = list(db.execute("CALL my_proc() YIELD greeting RETURN greeting"))
        assert result[0]["greeting"] == "hello"

    @pytest.mark.xfail(reason="CREATE PROCEDURE not supported in GQL parser")
    def test_create_procedure_if_not_exists(self, db):
        """CREATE PROCEDURE IF NOT EXISTS."""
        db.execute(
            "CREATE PROCEDURE IF NOT EXISTS my_proc2() RETURNS (v INT64) BEGIN   RETURN 42 AS v END"
        )
        db.execute(
            "CREATE PROCEDURE IF NOT EXISTS my_proc2() RETURNS (v INT64) BEGIN   RETURN 42 AS v END"
        )

    @pytest.mark.xfail(reason="CREATE PROCEDURE not supported in GQL parser")
    def test_create_or_replace_procedure(self, db):
        """CREATE OR REPLACE PROCEDURE."""
        db.execute("CREATE PROCEDURE replaceable() RETURNS (v INT64) BEGIN   RETURN 1 AS v END")
        db.execute(
            "CREATE OR REPLACE PROCEDURE replaceable() RETURNS (v INT64) BEGIN   RETURN 2 AS v END"
        )
        result = list(db.execute("CALL replaceable() YIELD v RETURN v"))
        assert result[0]["v"] == 2

    @pytest.mark.xfail(reason="DROP PROCEDURE not supported in GQL parser")
    def test_drop_procedure(self, db):
        """DROP PROCEDURE."""
        db.execute("CREATE PROCEDURE to_drop() RETURNS (v INT64) BEGIN   RETURN 1 AS v END")
        db.execute("DROP PROCEDURE to_drop")

    def test_drop_procedure_if_exists(self, db):
        """DROP PROCEDURE IF EXISTS."""
        db.execute("DROP PROCEDURE IF EXISTS nonexistent_proc")


# =============================================================================
# Variable Definitions (sec 10) - MISSING
# =============================================================================


class TestVariableDefinitions:
    """VALUE, binding table, and graph variable definitions (ISO sec 10)."""

    @pytest.mark.xfail(reason="Variable definitions not implemented (sec 10)")
    def test_value_variable_definition(self, db):
        """VALUE var = expr variable binding."""
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("VALUE x = 42 MATCH (n:N) RETURN x"))
        assert result[0]["x"] == 42
