"""Cypher spec: Expressions (openCypher 9 sec 5).

Covers: Arithmetic, comparison, logical, string predicates, CASE,
list comprehension, list slice, map literal, map projection,
parameters, property/index access, list predicates (all/any/none/single),
regex match.
"""

import pytest

# =============================================================================
# Operators (sec 5.1)
# =============================================================================


class TestArithmeticOperators:
    """Arithmetic: +, -, *, /, %, ^."""

    def test_addition(self, db):
        db.create_node(["N"], {"v": 10})
        result = list(db.execute_cypher("MATCH (n:N) RETURN n.v + 5 AS r"))
        assert result[0]["r"] == 15

    def test_subtraction(self, db):
        db.create_node(["N"], {"v": 10})
        result = list(db.execute_cypher("MATCH (n:N) RETURN n.v - 3 AS r"))
        assert result[0]["r"] == 7

    def test_multiplication(self, db):
        db.create_node(["N"], {"v": 6})
        result = list(db.execute_cypher("MATCH (n:N) RETURN n.v * 7 AS r"))
        assert result[0]["r"] == 42

    def test_division(self, db):
        db.create_node(["N"], {"v": 10})
        result = list(db.execute_cypher("MATCH (n:N) RETURN n.v / 2 AS r"))
        assert result[0]["r"] == 5

    def test_modulo(self, db):
        db.create_node(["N"], {"v": 10})
        result = list(db.execute_cypher("MATCH (n:N) RETURN n.v % 3 AS r"))
        assert result[0]["r"] == 1

    def test_power(self, db):
        """^ power operator (Cypher-specific)."""
        db.create_node(["N"], {"v": 3})
        result = list(db.execute_cypher("MATCH (n:N) RETURN n.v ^ 2 AS r"))
        assert result[0]["r"] == 9

    def test_unary_minus(self, db):
        db.create_node(["N"], {"v": 5})
        result = list(db.execute_cypher("MATCH (n:N) RETURN -n.v AS r"))
        assert result[0]["r"] == -5

    def test_unary_plus(self, db):
        db.create_node(["N"], {"v": 5})
        result = list(db.execute_cypher("MATCH (n:N) RETURN +n.v AS r"))
        assert result[0]["r"] == 5

    def test_string_concat(self, db):
        """+ overloaded for string concatenation."""
        db.create_node(["N"], {"a": "hello", "b": "world"})
        result = list(db.execute_cypher("MATCH (n:N) RETURN n.a + ' ' + n.b AS r"))
        assert result[0]["r"] == "hello world"


class TestComparisonOperators:
    """Comparison: =, <>, <, <=, >, >=."""

    def test_equals(self, db):
        db.create_node(["N"], {"v": 10})
        result = list(db.execute_cypher("MATCH (n:N) WHERE n.v = 10 RETURN n.v"))
        assert len(result) == 1

    def test_not_equals(self, db):
        db.create_node(["N"], {"v": 10})
        db.create_node(["N"], {"v": 20})
        result = list(db.execute_cypher("MATCH (n:N) WHERE n.v <> 10 RETURN n.v"))
        assert result[0]["n.v"] == 20

    def test_less_than(self, db):
        db.create_node(["N"], {"v": 5})
        db.create_node(["N"], {"v": 15})
        result = list(db.execute_cypher("MATCH (n:N) WHERE n.v < 10 RETURN n.v"))
        assert result[0]["n.v"] == 5

    def test_greater_equal(self, db):
        db.create_node(["N"], {"v": 10})
        db.create_node(["N"], {"v": 5})
        result = list(db.execute_cypher("MATCH (n:N) WHERE n.v >= 10 RETURN n.v"))
        assert len(result) == 1


class TestStringPredicates:
    """STARTS WITH, ENDS WITH, CONTAINS, IN, =~ (regex)."""

    def test_starts_with(self, db):
        db.create_node(["N"], {"v": "Amsterdam"})
        db.create_node(["N"], {"v": "Berlin"})
        result = list(db.execute_cypher("MATCH (n:N) WHERE n.v STARTS WITH 'Am' RETURN n.v"))
        assert result[0]["n.v"] == "Amsterdam"

    def test_ends_with(self, db):
        db.create_node(["N"], {"v": "Amsterdam"})
        db.create_node(["N"], {"v": "Berlin"})
        result = list(db.execute_cypher("MATCH (n:N) WHERE n.v ENDS WITH 'in' RETURN n.v"))
        assert result[0]["n.v"] == "Berlin"

    def test_contains(self, db):
        db.create_node(["N"], {"v": "Amsterdam"})
        db.create_node(["N"], {"v": "Berlin"})
        result = list(db.execute_cypher("MATCH (n:N) WHERE n.v CONTAINS 'ster' RETURN n.v"))
        assert result[0]["n.v"] == "Amsterdam"

    def test_in_list(self, db):
        db.create_node(["N"], {"v": "a"})
        db.create_node(["N"], {"v": "b"})
        db.create_node(["N"], {"v": "c"})
        result = list(db.execute_cypher("MATCH (n:N) WHERE n.v IN ['a', 'c'] RETURN n.v"))
        assert len(result) == 2

    def test_regex_match(self, db):
        """=~ regex match operator."""
        db.create_node(["N"], {"v": "Amsterdam"})
        db.create_node(["N"], {"v": "Berlin"})
        result = list(db.execute_cypher("MATCH (n:N) WHERE n.v =~ 'Am.*' RETURN n.v"))
        assert result[0]["n.v"] == "Amsterdam"

    def test_is_null(self, db):
        db.create_node(["N"], {"v": 1})
        db.create_node(["N"], {"w": 2})
        result = list(db.execute_cypher("MATCH (n:N) WHERE n.v IS NULL RETURN count(n) AS cnt"))
        assert result[0]["cnt"] == 1

    def test_is_not_null(self, db):
        db.create_node(["N"], {"v": 1})
        db.create_node(["N"], {"w": 2})
        result = list(db.execute_cypher("MATCH (n:N) WHERE n.v IS NOT NULL RETURN n.v"))
        assert result[0]["n.v"] == 1


# =============================================================================
# Expression Forms (sec 5.2)
# =============================================================================


class TestExpressionForms:
    """CASE, list comprehension, list slice, map, parameters."""

    def test_case_simple(self, db):
        db.create_node(["N"], {"v": "Amsterdam"})
        result = list(
            db.execute_cypher(
                "MATCH (n:N) "
                "RETURN CASE n.v "
                "WHEN 'Amsterdam' THEN 'NL' "
                "WHEN 'Berlin' THEN 'DE' END AS r"
            )
        )
        assert result[0]["r"] == "NL"

    def test_case_searched(self, db):
        db.create_node(["N"], {"v": 30})
        result = list(
            db.execute_cypher(
                "MATCH (n:N) RETURN CASE WHEN n.v > 25 THEN 'high' ELSE 'low' END AS r"
            )
        )
        assert result[0]["r"] == "high"

    def test_list_literal(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN [1, 2, 3] AS r"))
        assert result[0]["r"] == [1, 2, 3]

    def test_list_comprehension(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(
            db.execute_cypher("MATCH (n:N) RETURN [x IN [1,2,3,4,5] WHERE x > 2 | x * 10] AS r")
        )
        assert result[0]["r"] == [30, 40, 50]

    def test_list_comprehension_filter_only(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN [x IN [1,2,3,4] WHERE x > 2] AS r"))
        assert result[0]["r"] == [3, 4]

    @pytest.mark.xfail(reason="SliceAccess expression not executed at runtime")
    def test_list_slice(self, db):
        """list[start..end] slice syntax."""
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN [1,2,3,4,5][1..3] AS r"))
        assert result[0]["r"] == [2, 3]

    def test_index_access(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN [10, 20, 30][1] AS r"))
        assert result[0]["r"] == 20

    def test_map_literal(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN {a: 1, b: 'x'} AS r"))
        assert result[0]["r"]["a"] == 1
        assert result[0]["r"]["b"] == "x"

    def test_map_projection(self, db):
        """node { .prop1, .prop2 } map projection."""
        db.create_node(["Person"], {"name": "Alix", "age": 30, "city": "Amsterdam"})
        result = list(db.execute_cypher("MATCH (n:Person) RETURN n { .name, .age } AS r"))
        assert result[0]["r"]["name"] == "Alix"
        assert result[0]["r"]["age"] == 30

    @pytest.mark.xfail(reason="n { .* } wraps properties under nested '*' key instead of expanding")
    def test_map_projection_all(self, db):
        """node { .* } all-properties map projection."""
        db.create_node(["Person"], {"name": "Alix", "age": 30})
        result = list(db.execute_cypher("MATCH (n:Person) RETURN n { .* } AS r"))
        assert result[0]["r"]["name"] == "Alix"
        assert result[0]["r"]["age"] == 30

    @pytest.mark.xfail(reason="n['prop'] bracket access returns None")
    def test_bracket_property_access(self, db):
        """n['prop'] bracket access."""
        db.create_node(["N"], {"name": "Alix"})
        result = list(db.execute_cypher("MATCH (n:N) RETURN n['name'] AS r"))
        assert result[0]["r"] == "Alix"

    def test_coalesce(self, db):
        db.create_node(["N"], {"v": 10})
        result = list(db.execute_cypher("MATCH (n:N) RETURN coalesce(n.missing, n.v, 99) AS r"))
        assert result[0]["r"] == 10

    def test_reduce(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(
            db.execute_cypher("MATCH (n:N) RETURN reduce(acc = 0, x IN [1,2,3,4] | acc + x) AS r")
        )
        assert result[0]["r"] == 10


# =============================================================================
# Parameters (sec 5.2)
# =============================================================================


class TestParameters:
    """Dynamic parameters $param."""

    def test_param_in_where(self, db):
        db.create_node(["Person"], {"name": "Alix", "age": 30})
        result = list(
            db.execute_cypher(
                "MATCH (n:Person) WHERE n.name = $name RETURN n.age",
                {"name": "Alix"},
            )
        )
        assert result[0]["n.age"] == 30

    def test_param_in_return(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN $val AS r", {"val": 42}))
        assert result[0]["r"] == 42

    def test_param_list(self, db):
        db.create_node(["N"], {"v": "a"})
        db.create_node(["N"], {"v": "b"})
        db.create_node(["N"], {"v": "c"})
        result = list(
            db.execute_cypher(
                "MATCH (n:N) WHERE n.v IN $vals RETURN n.v",
                {"vals": ["a", "c"]},
            )
        )
        assert len(result) == 2


# =============================================================================
# List Predicates (sec 5.3)
# =============================================================================


class TestListPredicates:
    """all(), any(), none(), single() list predicates."""

    def test_all(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(
            db.execute_cypher("MATCH (n:N) RETURN all(x IN [2, 4, 6] WHERE x % 2 = 0) AS r")
        )
        assert result[0]["r"] is True

    def test_any(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN any(x IN [1, 2, 3] WHERE x > 2) AS r"))
        assert result[0]["r"] is True

    def test_none(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN none(x IN [1, 2, 3] WHERE x > 5) AS r"))
        assert result[0]["r"] is True

    def test_single(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(
            db.execute_cypher("MATCH (n:N) RETURN single(x IN [1, 2, 3] WHERE x > 2) AS r")
        )
        assert result[0]["r"] is True
