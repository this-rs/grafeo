"""GQL spec: Type System (ISO sec 18, 20, 24).

Covers: IS TYPED, IS NOT TYPED, CAST type checks, LIST<T>,
record types, graph reference types.
"""

import pytest

# =============================================================================
# IS TYPED / IS NOT TYPED predicates (sec 19.6)
# =============================================================================


class TestIsTyped:
    """Runtime type checking with IS TYPED."""

    def test_is_typed_int(self, db):
        db.create_node(["N"], {"v": 42})
        result = list(db.execute("MATCH (n:N) WHERE n.v IS TYPED INT64 RETURN n.v"))
        assert len(result) == 1

    def test_is_typed_float(self, db):
        db.create_node(["N"], {"v": 3.14})
        result = list(db.execute("MATCH (n:N) WHERE n.v IS TYPED FLOAT64 RETURN n.v"))
        assert len(result) == 1

    def test_is_typed_string(self, db):
        db.create_node(["N"], {"v": "hello"})
        result = list(db.execute("MATCH (n:N) WHERE n.v IS TYPED STRING RETURN n.v"))
        assert len(result) == 1

    def test_is_typed_boolean(self, db):
        db.create_node(["N"], {"v": True})
        result = list(db.execute("MATCH (n:N) WHERE n.v IS TYPED BOOLEAN RETURN n.v"))
        assert len(result) == 1

    def test_is_not_typed(self, db):
        db.create_node(["N"], {"v": 42})
        result = list(db.execute("MATCH (n:N) WHERE n.v IS NOT TYPED STRING RETURN n.v"))
        assert len(result) == 1

    @pytest.mark.xfail(reason="SET with date() function may not store typed DATE value")
    def test_is_typed_date(self, db):
        db.create_node(["N"], {"v": 1})
        db.execute("MATCH (n:N) SET n.d = date('2024-01-15')")
        result = list(db.execute("MATCH (n:N) WHERE n.d IS TYPED DATE RETURN n.d"))
        assert len(result) == 1

    def test_is_typed_list(self, db):
        db.create_node(["N"], {"v": [1, 2, 3]})
        result = list(db.execute("MATCH (n:N) WHERE n.v IS TYPED LIST RETURN n.v"))
        assert len(result) == 1

    def test_is_typed_list_of_int(self, db):
        """IS TYPED LIST<INT> parameterized list type."""
        db.create_node(["N"], {"v": [1, 2, 3]})
        result = list(db.execute("MATCH (n:N) WHERE n.v IS TYPED LIST<INT64> RETURN n.v"))
        assert len(result) == 1

    def test_is_typed_record(self, db):
        """IS TYPED RECORD for map values."""
        db.create_node(["N"], {"v": 1})
        result = list(
            db.execute("MATCH (n:N) WITH {a: 1, b: 'x'} AS m WHERE m IS TYPED RECORD RETURN m")
        )
        assert len(result) == 1


# =============================================================================
# Basic Type Values
# =============================================================================


class TestTypeValues:
    """Verify that each value type works correctly end-to-end."""

    def test_int64_roundtrip(self, db):
        db.create_node(["N"], {"v": 9223372036854775807})  # i64 max
        result = list(db.execute("MATCH (n:N) RETURN n.v"))
        assert result[0]["n.v"] == 9223372036854775807

    def test_float64_roundtrip(self, db):
        db.create_node(["N"], {"v": 1.7976931348623157e308})
        result = list(db.execute("MATCH (n:N) RETURN n.v"))
        assert result[0]["n.v"] > 1e307

    def test_string_roundtrip(self, db):
        db.create_node(["N"], {"v": "hello world"})
        result = list(db.execute("MATCH (n:N) RETURN n.v"))
        assert result[0]["n.v"] == "hello world"

    def test_boolean_true(self, db):
        db.create_node(["N"], {"v": True})
        result = list(db.execute("MATCH (n:N) RETURN n.v"))
        assert result[0]["n.v"] is True

    def test_boolean_false(self, db):
        db.create_node(["N"], {"v": False})
        result = list(db.execute("MATCH (n:N) RETURN n.v"))
        assert result[0]["n.v"] is False

    def test_null_value(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN n.missing"))
        assert result[0]["n.missing"] is None

    def test_list_heterogeneous(self, db):
        db.create_node(["N"], {"v": [1, "two", True]})
        result = list(db.execute("MATCH (n:N) RETURN n.v"))
        assert result[0]["n.v"] == [1, "two", True]

    def test_path_type(self, db):
        """PATH is a first-class value type."""
        a = db.create_node(["Node"], {"name": "a"})
        b = db.create_node(["Node"], {"name": "b"})
        db.create_edge(a.id, b.id, "NEXT")
        result = list(
            db.execute("MATCH p = (a:Node {name: 'a'})-[:NEXT]->(b:Node) RETURN length(p) AS len")
        )
        assert result[0]["len"] == 1


# =============================================================================
# CAST type conversions
# =============================================================================


class TestCastTypes:
    """CAST between types."""

    def test_cast_string_to_int(self, db):
        db.create_node(["N"], {"v": "42"})
        result = list(db.execute("MATCH (n:N) RETURN CAST(n.v AS INT) AS r"))
        assert result[0]["r"] == 42

    def test_cast_int_to_string(self, db):
        db.create_node(["N"], {"v": 42})
        result = list(db.execute("MATCH (n:N) RETURN CAST(n.v AS STRING) AS r"))
        assert result[0]["r"] == "42"

    def test_cast_string_to_float(self, db):
        db.create_node(["N"], {"v": "3.14"})
        result = list(db.execute("MATCH (n:N) RETURN CAST(n.v AS FLOAT) AS r"))
        assert abs(result[0]["r"] - 3.14) < 0.001

    def test_cast_string_to_bool(self, db):
        db.create_node(["N"], {"v": "true"})
        result = list(db.execute("MATCH (n:N) RETURN CAST(n.v AS BOOLEAN) AS r"))
        assert result[0]["r"] is True

    def test_cast_to_date(self, db):
        db.create_node(["N"], {"v": "2024-06-15"})
        result = list(db.execute("MATCH (n:N) RETURN CAST(n.v AS DATE) AS r"))
        assert "2024-06-15" in str(result[0]["r"])

    def test_cast_to_time(self, db):
        db.create_node(["N"], {"v": "14:30:00"})
        result = list(db.execute("MATCH (n:N) RETURN CAST(n.v AS TIME) AS r"))
        assert "14:30" in str(result[0]["r"])

    def test_cast_to_datetime(self, db):
        db.create_node(["N"], {"v": "2024-06-15T14:30:00"})
        result = list(db.execute("MATCH (n:N) RETURN CAST(n.v AS DATETIME) AS r"))
        assert "2024" in str(result[0]["r"])

    def test_cast_to_duration(self, db):
        db.create_node(["N"], {"v": "P1Y2M"})
        result = list(db.execute("MATCH (n:N) RETURN CAST(n.v AS DURATION) AS r"))
        assert result[0]["r"] is not None

    def test_cast_to_list(self, db):
        """CAST scalar to LIST wraps in array."""
        db.create_node(["N"], {"v": 42})
        result = list(db.execute("MATCH (n:N) RETURN CAST(n.v AS LIST) AS r"))
        assert result[0]["r"] == [42]

    def test_cast_to_zoned_datetime(self, db):
        db.create_node(["N"], {"v": "2024-06-15T14:30:00+01:00"})
        result = list(db.execute("MATCH (n:N) RETURN CAST(n.v AS ZONED DATETIME) AS r"))
        assert result[0]["r"] is not None

    def test_cast_to_zoned_time(self, db):
        db.create_node(["N"], {"v": "14:30:00+01:00"})
        result = list(db.execute("MATCH (n:N) RETURN CAST(n.v AS ZONED TIME) AS r"))
        assert result[0]["r"] is not None
