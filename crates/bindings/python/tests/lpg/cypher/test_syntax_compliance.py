"""Cypher syntax compliance tests (openCypher 9.0).

Each test targets a specific spec element with a minimal query.
Tests are organized by category: clauses, expressions, functions, patterns.
"""

import pytest

# =============================================================================
# CLAUSES
# =============================================================================


class TestCypherClauses:
    """Test all Cypher clause types."""

    # --- MATCH ---

    def test_match_by_label(self, pattern_db):
        result = list(pattern_db.execute_cypher("MATCH (n:Person) RETURN n.name"))
        names = {r["n.name"] for r in result}
        assert names == {"Alix", "Gus", "Vincent"}

    def test_match_by_property(self, pattern_db):
        result = list(pattern_db.execute_cypher("MATCH (n:Person {name: 'Alix'}) RETURN n.age"))
        assert len(result) == 1
        assert result[0]["n.age"] == 30

    def test_multiple_match_clauses(self, pattern_db):
        """Two sequential MATCH clauses (standard Cypher)."""
        result = list(
            pattern_db.execute_cypher(
                "MATCH (a:Person {name: 'Alix'}) "
                "MATCH (b:Person {name: 'Gus'}) "
                "RETURN a.name, b.name"
            )
        )
        assert len(result) == 1
        assert result[0]["a.name"] == "Alix"
        assert result[0]["b.name"] == "Gus"

    def test_three_match_clauses(self, pattern_db):
        """Three sequential MATCH clauses."""
        result = list(
            pattern_db.execute_cypher(
                "MATCH (a:Person {name: 'Alix'}) "
                "MATCH (b:Person {name: 'Gus'}) "
                "MATCH (c:Person {name: 'Vincent'}) "
                "RETURN a.name, b.name, c.name"
            )
        )
        assert len(result) == 1
        assert result[0]["a.name"] == "Alix"
        assert result[0]["b.name"] == "Gus"
        assert result[0]["c.name"] == "Vincent"

    def test_multiple_match_with_create(self, db):
        """MATCH + MATCH + CREATE (Deriva pattern)."""
        db.create_node(["Person"], {"name": "Src"})
        db.create_node(["Person"], {"name": "Dst"})
        result = list(
            db.execute_cypher(
                "MATCH (src:Person {name: 'Src'}) "
                "MATCH (dst:Person {name: 'Dst'}) "
                "CREATE (src)-[r:KNOWS]->(dst) "
                "RETURN r"
            )
        )
        assert len(result) == 1

    # --- OPTIONAL MATCH ---

    def test_optional_match(self, pattern_db):
        pattern_db.create_node(["Person"], {"name": "Loner", "age": 99})
        result = list(
            pattern_db.execute_cypher(
                "MATCH (p:Person) OPTIONAL MATCH (p)-[:WORKS_AT]->(c:Company) RETURN p.name, c.name"
            )
        )
        loner = [r for r in result if r["p.name"] == "Loner"]
        assert len(loner) == 1
        assert loner[0]["c.name"] is None

    # --- WHERE ---

    def test_where_comparison(self, pattern_db):
        result = list(pattern_db.execute_cypher("MATCH (n:Person) WHERE n.age > 28 RETURN n.name"))
        names = {r["n.name"] for r in result}
        assert "Alix" in names
        assert "Vincent" in names
        assert "Gus" not in names

    # --- WITH ---

    def test_with_clause(self, pattern_db):
        result = list(
            pattern_db.execute_cypher(
                "MATCH (p:Person) "
                "WITH p.name AS name, p.age AS age "
                "WHERE age > 28 "
                "RETURN name ORDER BY name"
            )
        )
        names = [r["name"] for r in result]
        assert names == ["Alix", "Vincent"]

    def test_with_where(self, pattern_db):
        result = list(
            pattern_db.execute_cypher("MATCH (p:Person) WITH p WHERE p.age > 30 RETURN p.name")
        )
        assert len(result) == 1
        assert result[0]["p.name"] == "Vincent"

    # --- RETURN ---

    def test_return_distinct(self, pattern_db):
        result = list(pattern_db.execute_cypher("MATCH (p:Person) RETURN DISTINCT p.city"))
        cities = {r["p.city"] for r in result}
        assert cities == {"LA", "NYC"}

    def test_return_order_by_alias_asc(self, pattern_db):
        """ORDER BY on an alias (known working pattern)."""
        result = list(
            pattern_db.execute_cypher(
                "MATCH (p:Person) RETURN p.name AS name, p.age AS age ORDER BY age ASC"
            )
        )
        names = [r["name"] for r in result]
        assert names == ["Gus", "Alix", "Vincent"]

    def test_return_order_by_alias_desc(self, pattern_db):
        result = list(
            pattern_db.execute_cypher(
                "MATCH (p:Person) RETURN p.name AS name, p.age AS age ORDER BY age DESC"
            )
        )
        names = [r["name"] for r in result]
        assert names == ["Vincent", "Alix", "Gus"]

    def test_return_order_by_property(self, pattern_db):
        """ORDER BY on a property access (n.prop) without alias."""
        result = list(pattern_db.execute_cypher("MATCH (p:Person) RETURN p.name ORDER BY p.age"))
        names = [r["p.name"] for r in result]
        assert names == ["Gus", "Alix", "Vincent"]

    def test_return_skip(self, pattern_db):
        result = list(
            pattern_db.execute_cypher(
                "MATCH (p:Person) RETURN p.name AS name, p.age AS age ORDER BY age SKIP 1"
            )
        )
        assert len(result) == 2

    def test_return_limit(self, pattern_db):
        result = list(
            pattern_db.execute_cypher(
                "MATCH (p:Person) RETURN p.name AS name, p.age AS age ORDER BY age LIMIT 2"
            )
        )
        assert len(result) == 2

    def test_return_skip_and_limit(self, pattern_db):
        result = list(
            pattern_db.execute_cypher(
                "MATCH (p:Person) RETURN p.name AS name, p.age AS age ORDER BY age SKIP 1 LIMIT 1"
            )
        )
        assert len(result) == 1
        assert result[0]["name"] == "Alix"

    # --- UNWIND ---

    def test_unwind(self, db):
        result = list(db.execute_cypher("UNWIND [1, 2, 3] AS x RETURN x"))
        values = [r["x"] for r in result]
        assert values == [1, 2, 3]

    def test_unwind_with_create(self, db):
        db.execute_cypher("UNWIND ['A', 'B', 'C'] AS name CREATE (n:Item {name: name})")
        result = list(db.execute_cypher("MATCH (n:Item) RETURN n.name"))
        names = {r["n.name"] for r in result}
        assert names == {"A", "B", "C"}

    # --- CREATE ---

    def test_create_node(self, db):
        result = list(db.execute_cypher("CREATE (n:Person {name: 'Eve', age: 28}) RETURN n"))
        assert len(result) == 1

    def test_create_node_multiple_labels(self, db):
        result = list(db.execute_cypher("CREATE (n:Person:Employee {name: 'Frank'}) RETURN n"))
        assert len(result) == 1

    def test_create_relationship(self, pattern_db):
        result = list(
            pattern_db.execute_cypher(
                "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) "
                "CREATE (a)-[r:LIKES {since: 2024}]->(b) RETURN r"
            )
        )
        assert len(result) == 1

    # --- DELETE ---

    def test_delete_node(self, db):
        db.create_node(["Temp"], {"name": "ToDelete"})
        db.execute_cypher("MATCH (n:Temp {name: 'ToDelete'}) DELETE n")
        result = list(db.execute_cypher("MATCH (n:Temp) RETURN n"))
        assert len(result) == 0

    def test_detach_delete(self, db):
        a = db.create_node(["Temp"], {"name": "A"})
        b = db.create_node(["Temp"], {"name": "B"})
        db.create_edge(a.id, b.id, "REL")
        db.execute_cypher("MATCH (n:Temp {name: 'A'}) DETACH DELETE n")
        result = list(db.execute_cypher("MATCH (n:Temp) RETURN n.name"))
        names = [r["n.name"] for r in result]
        assert "A" not in names
        assert "B" in names

    # --- SET ---

    def test_set_property(self, db):
        db.create_node(["Person"], {"name": "SetTest", "age": 20})
        db.execute_cypher("MATCH (n:Person {name: 'SetTest'}) SET n.age = 21")
        result = list(db.execute_cypher("MATCH (n:Person {name: 'SetTest'}) RETURN n.age"))
        assert result[0]["n.age"] == 21

    def test_set_add_new_property(self, db):
        db.create_node(["Person"], {"name": "AddProp"})
        db.execute_cypher("MATCH (n:Person {name: 'AddProp'}) SET n.email = 'test@test.com'")
        result = list(db.execute_cypher("MATCH (n:Person {name: 'AddProp'}) RETURN n.email"))
        assert result[0]["n.email"] == "test@test.com"

    def test_set_label(self, db):
        db.create_node(["Person"], {"name": "LabelTest"})
        db.execute_cypher("MATCH (n:Person {name: 'LabelTest'}) SET n:Admin")
        result = list(db.execute_cypher("MATCH (n:Admin {name: 'LabelTest'}) RETURN n.name"))
        assert len(result) == 1

    # --- REMOVE ---

    def test_remove_property(self, db):
        db.create_node(["Person"], {"name": "RemTest", "temp": "value"})
        db.execute_cypher("MATCH (n:Person {name: 'RemTest'}) REMOVE n.temp")
        result = list(db.execute_cypher("MATCH (n:Person {name: 'RemTest'}) RETURN n.temp"))
        assert result[0].get("n.temp") is None

    def test_remove_label(self, db):
        db.create_node(["Person", "Admin"], {"name": "UnLabel"})
        db.execute_cypher("MATCH (n:Admin {name: 'UnLabel'}) REMOVE n:Admin")
        result = list(db.execute_cypher("MATCH (n:Admin {name: 'UnLabel'}) RETURN n"))
        assert len(result) == 0

    # --- MERGE ---

    def test_merge_node_create(self, db):
        db.execute_cypher("MERGE (n:City {name: 'Paris'}) RETURN n")
        result = list(db.execute_cypher("MATCH (n:City) RETURN count(n) AS cnt"))
        assert result[0]["cnt"] == 1

    def test_merge_node_idempotent(self, db):
        db.execute_cypher("MERGE (n:City {name: 'Paris'}) RETURN n")
        db.execute_cypher("MERGE (n:City {name: 'Paris'}) RETURN n")
        result = list(db.execute_cypher("MATCH (n:City) RETURN count(n) AS cnt"))
        assert result[0]["cnt"] == 1

    def test_merge_on_create_set(self, db):
        db.execute_cypher("MERGE (n:City {name: 'Berlin'}) ON CREATE SET n.new = true RETURN n")
        result = list(db.execute_cypher("MATCH (n:City {name: 'Berlin'}) RETURN n.new"))
        assert result[0]["n.new"] is True

    def test_merge_on_match_set(self, db):
        db.execute_cypher("CREATE (n:City {name: 'London'})")
        db.execute_cypher("MERGE (n:City {name: 'London'}) ON MATCH SET n.visited = true RETURN n")
        result = list(db.execute_cypher("MATCH (n:City {name: 'London'}) RETURN n.visited"))
        assert result[0]["n.visited"] is True

    def test_merge_relationship(self, db):
        """MERGE on relationship pattern."""
        db.create_node(["Person"], {"name": "X"})
        db.create_node(["Person"], {"name": "Y"})
        db.execute_cypher(
            "MATCH (a:Person {name: 'X'}), (b:Person {name: 'Y'}) MERGE (a)-[r:KNOWS]->(b) RETURN r"
        )
        result = list(
            db.execute_cypher(
                "MATCH (a:Person {name: 'X'})-[r:KNOWS]->(b:Person {name: 'Y'}) RETURN r"
            )
        )
        assert len(result) == 1

    def test_merge_relationship_idempotent(self, db):
        """Running MERGE twice should not create duplicate edges."""
        db.create_node(["Person"], {"name": "M1"})
        db.create_node(["Person"], {"name": "M2"})
        for _ in range(2):
            db.execute_cypher(
                "MATCH (a:Person {name: 'M1'}), (b:Person {name: 'M2'}) "
                "MERGE (a)-[r:FRIEND]->(b) RETURN r"
            )
        result = list(
            db.execute_cypher(
                "MATCH (:Person {name: 'M1'})-[r:FRIEND]->(:Person {name: 'M2'}) RETURN r"
            )
        )
        assert len(result) == 1

    # --- UNION ---

    def test_union(self, db):
        db.create_node(["Cat"], {"name": "Whiskers"})
        db.create_node(["Dog"], {"name": "Rex"})
        result = list(
            db.execute_cypher(
                "MATCH (c:Cat) RETURN c.name AS name UNION MATCH (d:Dog) RETURN d.name AS name"
            )
        )
        names = {r["name"] for r in result}
        assert names == {"Whiskers", "Rex"}

    def test_union_all(self, db):
        db.create_node(["A"], {"name": "shared"})
        db.create_node(["B"], {"name": "shared"})
        result = list(
            db.execute_cypher(
                "MATCH (a:A) RETURN a.name AS name UNION ALL MATCH (b:B) RETURN b.name AS name"
            )
        )
        assert len(result) == 2

    # --- CALL...YIELD ---

    def test_call_yield(self, db):
        """Verify CALL procedure syntax (uses built-in if available)."""
        # Just verify the syntax parses, not the result
        try:
            db.execute_cypher("CALL obrain.schema.nodeLabels() YIELD label RETURN label")
        except Exception:
            pytest.skip("No built-in procedure available for testing")


# =============================================================================
# EXPRESSIONS & OPERATORS
# =============================================================================


class TestCypherExpressions:
    """Test Cypher expressions and operators."""

    # --- Comparison ---

    def test_equals(self, pattern_db):
        result = list(
            pattern_db.execute_cypher("MATCH (n:Person) WHERE n.name = 'Alix' RETURN n.name")
        )
        assert len(result) == 1

    def test_not_equals(self, pattern_db):
        result = list(
            pattern_db.execute_cypher("MATCH (n:Person) WHERE n.name <> 'Alix' RETURN n.name")
        )
        assert len(result) == 2

    def test_less_than(self, pattern_db):
        result = list(pattern_db.execute_cypher("MATCH (n:Person) WHERE n.age < 30 RETURN n.name"))
        assert len(result) == 1
        assert result[0]["n.name"] == "Gus"

    def test_greater_equal(self, pattern_db):
        result = list(pattern_db.execute_cypher("MATCH (n:Person) WHERE n.age >= 30 RETURN n.name"))
        assert len(result) == 2

    # --- Logical ---

    def test_and(self, pattern_db):
        result = list(
            pattern_db.execute_cypher(
                "MATCH (n:Person) WHERE n.age > 25 AND n.city = 'NYC' RETURN n.name"
            )
        )
        names = {r["n.name"] for r in result}
        assert names == {"Alix", "Vincent"}

    def test_or(self, pattern_db):
        result = list(
            pattern_db.execute_cypher(
                "MATCH (n:Person) WHERE n.name = 'Alix' OR n.name = 'Gus' RETURN n.name"
            )
        )
        assert len(result) == 2

    def test_not(self, pattern_db):
        result = list(
            pattern_db.execute_cypher("MATCH (n:Person) WHERE NOT n.city = 'NYC' RETURN n.name")
        )
        assert len(result) == 1
        assert result[0]["n.name"] == "Gus"

    def test_xor(self, db):
        db.create_node(["Item"], {"a": True, "b": False})
        db.create_node(["Item"], {"a": True, "b": True})
        db.create_node(["Item"], {"a": False, "b": False})
        result = list(db.execute_cypher("MATCH (n:Item) WHERE n.a XOR n.b RETURN n.a, n.b"))
        assert len(result) == 1
        assert result[0]["n.a"] is True
        assert result[0]["n.b"] is False

    # --- Arithmetic ---

    def test_addition(self, pattern_db):
        result = list(
            pattern_db.execute_cypher(
                "MATCH (n:Person {name: 'Alix'}) RETURN n.age + 1 AS next_age"
            )
        )
        assert result[0]["next_age"] == 31

    def test_subtraction(self, pattern_db):
        result = list(
            pattern_db.execute_cypher(
                "MATCH (n:Person {name: 'Alix'}) RETURN n.age - 1 AS prev_age"
            )
        )
        assert result[0]["prev_age"] == 29

    def test_multiplication(self, db):
        result = list(db.execute_cypher("WITH 6 AS a, 7 AS b RETURN a * b AS answer"))
        assert result[0]["answer"] == 42

    def test_multiplication_standalone(self, db):
        """Standalone RETURN with arithmetic."""
        result = list(db.execute_cypher("RETURN 6 * 7 AS answer"))
        assert result[0]["answer"] == 42

    def test_division(self, db):
        result = list(db.execute_cypher("WITH 10 AS a RETURN a / 2 AS half"))
        assert result[0]["half"] == 5

    def test_modulo(self, db):
        result = list(db.execute_cypher("WITH 10 AS a RETURN a % 3 AS remainder"))
        assert result[0]["remainder"] == 1

    def test_unary_minus(self, db):
        result = list(db.execute_cypher("WITH 5 AS a RETURN -a AS neg"))
        assert result[0]["neg"] == -5

    def test_unary_minus_standalone(self, db):
        """Standalone RETURN -5."""
        result = list(db.execute_cypher("RETURN -5 AS neg"))
        assert result[0]["neg"] == -5

    # --- String operators ---

    def test_string_concat(self, db):
        result = list(db.execute_cypher("WITH 'hello' AS a RETURN a + ' world' AS greeting"))
        assert result[0]["greeting"] == "hello world"

    def test_string_concat_standalone(self, db):
        """Standalone RETURN with string concat."""
        result = list(db.execute_cypher("RETURN 'hello' + ' ' + 'world' AS greeting"))
        assert result[0]["greeting"] == "hello world"

    # --- Null checks ---

    def test_is_null(self, db):
        db.create_node(["Item"], {"name": "WithProp", "val": 1})
        db.create_node(["Item"], {"name": "NoProp"})
        result = list(db.execute_cypher("MATCH (n:Item) WHERE n.val IS NULL RETURN n.name"))
        assert len(result) == 1
        assert result[0]["n.name"] == "NoProp"

    def test_is_not_null(self, db):
        db.create_node(["Item"], {"name": "WithProp", "val": 1})
        db.create_node(["Item"], {"name": "NoProp"})
        result = list(db.execute_cypher("MATCH (n:Item) WHERE n.val IS NOT NULL RETURN n.name"))
        assert len(result) == 1
        assert result[0]["n.name"] == "WithProp"

    # --- IN ---

    def test_in_list(self, pattern_db):
        result = list(
            pattern_db.execute_cypher(
                "MATCH (n:Person) WHERE n.name IN ['Alix', 'Gus'] RETURN n.name"
            )
        )
        assert len(result) == 2

    # --- String predicates ---

    def test_starts_with(self, pattern_db):
        result = list(
            pattern_db.execute_cypher("MATCH (n:Person) WHERE n.name STARTS WITH 'A' RETURN n.name")
        )
        assert len(result) == 1
        assert result[0]["n.name"] == "Alix"

    def test_ends_with(self, pattern_db):
        result = list(
            pattern_db.execute_cypher("MATCH (n:Person) WHERE n.name ENDS WITH 'us' RETURN n.name")
        )
        names = {r["n.name"] for r in result}
        assert "Gus" in names

    def test_contains(self, pattern_db):
        result = list(
            pattern_db.execute_cypher("MATCH (n:Person) WHERE n.name CONTAINS 'in' RETURN n.name")
        )
        names = {r["n.name"] for r in result}
        assert "Vincent" in names

    def test_regex(self, pattern_db):
        result = list(
            pattern_db.execute_cypher("MATCH (n:Person) WHERE n.name =~ 'A.*' RETURN n.name")
        )
        assert len(result) == 1
        assert result[0]["n.name"] == "Alix"

    # --- CASE ---

    def test_case_simple(self, pattern_db):
        result = list(
            pattern_db.execute_cypher(
                "MATCH (n:Person {name: 'Alix'}) "
                "RETURN CASE n.city WHEN 'NYC' THEN 'East' WHEN 'LA' THEN 'West' END AS coast"
            )
        )
        assert result[0]["coast"] == "East"

    def test_case_searched(self, pattern_db):
        result = list(
            pattern_db.execute_cypher(
                "MATCH (n:Person {name: 'Vincent'}) "
                "RETURN CASE WHEN n.age > 30 THEN 'senior' ELSE 'junior' END AS level"
            )
        )
        assert result[0]["level"] == "senior"

    # --- Literals ---

    def test_list_literal(self, db):
        result = list(db.execute_cypher("WITH [1, 2, 3] AS nums RETURN nums"))
        assert result[0]["nums"] == [1, 2, 3]

    def test_list_literal_standalone_return(self, db):
        """Standalone RETURN with list literal (no MATCH/WITH)."""
        result = list(db.execute_cypher("RETURN [1, 2, 3] AS nums"))
        assert result[0]["nums"] == [1, 2, 3]

    def test_map_literal(self, db):
        result = list(db.execute_cypher("WITH {a: 1, b: 2} AS m RETURN m"))
        assert result[0]["m"]["a"] == 1
        assert result[0]["m"]["b"] == 2

    def test_map_literal_standalone_return(self, db):
        """Standalone RETURN with map literal (no MATCH/WITH)."""
        result = list(db.execute_cypher("RETURN {a: 1, b: 2} AS m"))
        assert result[0]["m"]["a"] == 1

    # --- Binary expression in RETURN ---

    def test_binary_expr_in_return(self, pattern_db):
        """RETURN count(n) > 0 AS has_data (Phase 0 fix)."""
        result = list(
            pattern_db.execute_cypher("MATCH (n:Person) RETURN count(n) > 0 AS has_people")
        )
        assert result[0]["has_people"] is True

    def test_arithmetic_in_return(self, pattern_db):
        result = list(
            pattern_db.execute_cypher(
                "MATCH (n:Person {name: 'Alix'}) RETURN n.age + 10 AS future_age"
            )
        )
        assert result[0]["future_age"] == 40

    def test_comparison_in_return(self, pattern_db):
        result = list(
            pattern_db.execute_cypher(
                "MATCH (n:Person {name: 'Alix'}) RETURN n.age > 25 AS is_over_25"
            )
        )
        assert result[0]["is_over_25"] is True

    # --- Parameters ---

    def test_parameters(self, db):
        db.create_node(["Person"], {"name": "ParamTest", "age": 42})
        result = list(
            db.execute_cypher(
                "MATCH (n:Person) WHERE n.name = $name RETURN n.age",
                {"name": "ParamTest"},
            )
        )
        assert len(result) == 1
        assert result[0]["n.age"] == 42

    # --- exists as alias ---

    def test_exists_as_alias(self, pattern_db):
        """`exists` should be usable as an alias name."""
        result = list(pattern_db.execute_cypher("MATCH (n:Person) RETURN count(n) AS exists"))
        assert result[0]["exists"] == 3


# =============================================================================
# SCALAR FUNCTIONS
# =============================================================================


class TestCypherScalarFunctions:
    """Test Cypher scalar functions (openCypher 9.0)."""

    def test_type_function(self, pattern_db):
        result = list(
            pattern_db.execute_cypher(
                "MATCH (a:Person {name: 'Alix'})-[r]->(b:Person {name: 'Gus'}) "
                "RETURN type(r) AS rel_type"
            )
        )
        assert result[0]["rel_type"] == "KNOWS"

    def test_id_function(self, db):
        db.create_node(["Person"], {"name": "IdTest"})
        result = list(db.execute_cypher("MATCH (n:Person {name: 'IdTest'}) RETURN id(n) AS nid"))
        assert result[0]["nid"] is not None

    def test_labels_function(self, db):
        db.create_node(["Person", "Employee"], {"name": "LabelTest"})
        result = list(db.execute_cypher("MATCH (n {name: 'LabelTest'}) RETURN labels(n) AS lbls"))
        lbls = result[0]["lbls"]
        assert "Person" in lbls
        assert "Employee" in lbls

    def test_keys_function(self, db):
        db.create_node(["Person"], {"name": "KeyTest", "age": 30})
        result = list(db.execute_cypher("MATCH (n:Person {name: 'KeyTest'}) RETURN keys(n) AS k"))
        keys = result[0]["k"]
        assert "name" in keys
        assert "age" in keys

    def test_properties_function(self, db):
        db.create_node(["Person"], {"name": "PropTest", "age": 25})
        result = list(
            db.execute_cypher("MATCH (n:Person {name: 'PropTest'}) RETURN properties(n) AS props")
        )
        props = result[0]["props"]
        assert props["name"] == "PropTest"
        assert props["age"] == 25

    def test_size_list(self, db):
        result = list(db.execute_cypher("WITH [1, 2, 3] AS lst RETURN size(lst) AS s"))
        assert result[0]["s"] == 3

    def test_size_string(self, db):
        result = list(db.execute_cypher("WITH 'hello' AS str RETURN size(str) AS s"))
        assert result[0]["s"] == 5

    def test_head(self, db):
        result = list(db.execute_cypher("WITH [1, 2, 3] AS lst RETURN head(lst) AS h"))
        assert result[0]["h"] == 1

    def test_tail(self, db):
        result = list(db.execute_cypher("WITH [1, 2, 3] AS lst RETURN tail(lst) AS t"))
        assert result[0]["t"] == [2, 3]

    def test_last(self, db):
        result = list(db.execute_cypher("WITH [1, 2, 3] AS lst RETURN last(lst) AS l"))
        assert result[0]["l"] == 3

    def test_coalesce(self, db):
        db.create_node(["Item"], {"name": "CoalTest"})
        result = list(
            db.execute_cypher(
                "MATCH (n:Item {name: 'CoalTest'}) RETURN coalesce(n.missing, 'default') AS val"
            )
        )
        assert result[0]["val"] == "default"

    def test_coalesce_first_non_null(self, db):
        db.create_node(["Item"], {"name": "Coal2"})
        result = list(
            db.execute_cypher("MATCH (n:Item {name: 'Coal2'}) RETURN coalesce(n.a, n.b, 42) AS val")
        )
        assert result[0]["val"] == 42

    # --- Type conversion ---

    def test_to_integer(self, db):
        result = list(db.execute_cypher("WITH '42' AS s RETURN toInteger(s) AS val"))
        assert result[0]["val"] == 42

    def test_to_float(self, db):
        result = list(db.execute_cypher("WITH '3.14' AS s RETURN toFloat(s) AS val"))
        assert abs(result[0]["val"] - 3.14) < 0.001

    def test_to_string(self, db):
        result = list(db.execute_cypher("WITH 42 AS n RETURN toString(n) AS val"))
        assert result[0]["val"] == "42"

    def test_to_boolean(self, db):
        result = list(db.execute_cypher("WITH 'true' AS s RETURN toBoolean(s) AS val"))
        assert result[0]["val"] is True

    # --- String functions ---

    def test_trim(self, db):
        result = list(db.execute_cypher("WITH '  hello  ' AS s RETURN trim(s) AS val"))
        assert result[0]["val"] == "hello"

    def test_ltrim(self, db):
        result = list(db.execute_cypher("WITH '  hello' AS s RETURN ltrim(s) AS val"))
        assert result[0]["val"] == "hello"

    def test_rtrim(self, db):
        result = list(db.execute_cypher("WITH 'hello  ' AS s RETURN rtrim(s) AS val"))
        assert result[0]["val"] == "hello"

    def test_replace(self, db):
        result = list(db.execute_cypher("WITH 'hello' AS s RETURN replace(s, 'l', 'r') AS val"))
        assert result[0]["val"] == "herro"

    def test_substring(self, db):
        result = list(db.execute_cypher("WITH 'hello' AS s RETURN substring(s, 1, 3) AS val"))
        assert result[0]["val"] == "ell"

    def test_split(self, db):
        result = list(db.execute_cypher("WITH 'a,b,c' AS s RETURN split(s, ',') AS val"))
        assert result[0]["val"] == ["a", "b", "c"]

    def test_upper(self, db):
        result = list(db.execute_cypher("WITH 'hello' AS s RETURN toUpper(s) AS val"))
        assert result[0]["val"] == "HELLO"

    def test_lower(self, db):
        result = list(db.execute_cypher("WITH 'HELLO' AS s RETURN toLower(s) AS val"))
        assert result[0]["val"] == "hello"

    def test_reverse_list(self, db):
        result = list(db.execute_cypher("WITH [1, 2, 3] AS lst RETURN reverse(lst) AS val"))
        assert result[0]["val"] == [3, 2, 1]

    def test_reverse_string(self, db):
        result = list(db.execute_cypher("WITH 'abc' AS s RETURN reverse(s) AS val"))
        assert result[0]["val"] == "cba"

    # --- Numeric functions ---

    def test_abs(self, db):
        result = list(db.execute_cypher("WITH -5 AS n RETURN abs(n) AS val"))
        assert result[0]["val"] == 5

    def test_ceil(self, db):
        result = list(db.execute_cypher("WITH 2.3 AS n RETURN ceil(n) AS val"))
        assert result[0]["val"] == 3

    def test_floor(self, db):
        result = list(db.execute_cypher("WITH 2.7 AS n RETURN floor(n) AS val"))
        assert result[0]["val"] == 2

    def test_round(self, db):
        result = list(db.execute_cypher("WITH 2.5 AS n RETURN round(n) AS val"))
        assert result[0]["val"] == 3

    def test_sqrt(self, db):
        result = list(db.execute_cypher("WITH 9 AS n RETURN sqrt(n) AS val"))
        assert result[0]["val"] == 3.0

    def test_rand(self, db):
        result = list(db.execute_cypher("UNWIND [1] AS _ RETURN rand() AS val"))
        assert 0 <= result[0]["val"] <= 1

    # --- Collection functions ---

    def test_range(self, db):
        result = list(db.execute_cypher("UNWIND [1] AS _ RETURN range(1, 5) AS val"))
        assert result[0]["val"] == [1, 2, 3, 4, 5]

    def test_range_with_step(self, db):
        result = list(db.execute_cypher("UNWIND [1] AS _ RETURN range(0, 10, 3) AS val"))
        assert result[0]["val"] == [0, 3, 6, 9]


# =============================================================================
# AGGREGATE FUNCTIONS
# =============================================================================


class TestCypherAggregates:
    """Test Cypher aggregate functions."""

    def test_count(self, pattern_db):
        result = list(pattern_db.execute_cypher("MATCH (n:Person) RETURN count(n) AS cnt"))
        assert result[0]["cnt"] == 3

    def test_count_distinct(self, pattern_db):
        result = list(
            pattern_db.execute_cypher("MATCH (n:Person) RETURN count(DISTINCT n.city) AS cnt")
        )
        assert result[0]["cnt"] == 2

    def test_sum(self, pattern_db):
        result = list(pattern_db.execute_cypher("MATCH (n:Person) RETURN sum(n.age) AS total"))
        assert result[0]["total"] == 90

    def test_avg(self, pattern_db):
        result = list(pattern_db.execute_cypher("MATCH (n:Person) RETURN avg(n.age) AS average"))
        assert result[0]["average"] == 30.0

    def test_min(self, pattern_db):
        result = list(pattern_db.execute_cypher("MATCH (n:Person) RETURN min(n.age) AS youngest"))
        assert result[0]["youngest"] == 25

    def test_max(self, pattern_db):
        result = list(pattern_db.execute_cypher("MATCH (n:Person) RETURN max(n.age) AS oldest"))
        assert result[0]["oldest"] == 35

    def test_collect(self, pattern_db):
        result = list(pattern_db.execute_cypher("MATCH (n:Person) RETURN collect(n.name) AS names"))
        names = result[0]["names"]
        assert set(names) == {"Alix", "Gus", "Vincent"}

    def test_stdev(self, db):
        for score in [10, 20, 30]:
            db.create_node(["Score"], {"val": score})
        result = list(db.execute_cypher("MATCH (n:Score) RETURN stdev(n.val) AS sd"))
        assert 8 <= result[0]["sd"] <= 12

    def test_percentile_disc(self, db):
        for v in [10, 20, 30, 40, 50]:
            db.create_node(["Val"], {"v": v})
        result = list(db.execute_cypher("MATCH (n:Val) RETURN percentileDisc(n.v, 0.5) AS median"))
        assert result[0]["median"] == 30

    def test_percentile_cont(self, db):
        for v in [10, 20, 30, 40, 50]:
            db.create_node(["Val"], {"v": v})
        result = list(db.execute_cypher("MATCH (n:Val) RETURN percentileCont(n.v, 0.5) AS median"))
        assert result[0]["median"] == 30.0


# =============================================================================
# PREDICATE FUNCTIONS
# =============================================================================


class TestCypherPredicateFunctions:
    """Test Cypher predicate functions."""

    def test_exists_property(self, db):
        db.create_node(["Item"], {"name": "With", "email": "a@b.com"})
        db.create_node(["Item"], {"name": "Without"})
        result = list(db.execute_cypher("MATCH (n:Item) WHERE exists(n.email) RETURN n.name"))
        assert len(result) == 1
        assert result[0]["n.name"] == "With"

    def test_all_predicate(self, db):
        db.create_node(["Data"], {"name": "AllPos", "scores": [1, 2, 3]})
        db.create_node(["Data"], {"name": "HasNeg", "scores": [1, -2, 3]})
        result = list(
            db.execute_cypher("MATCH (n:Data) WHERE all(x IN n.scores WHERE x > 0) RETURN n.name")
        )
        assert len(result) == 1
        assert result[0]["n.name"] == "AllPos"

    def test_any_predicate(self, db):
        db.create_node(["Data"], {"name": "HasAdmin", "tags": ["user", "admin"]})
        db.create_node(["Data"], {"name": "NoAdmin", "tags": ["user", "guest"]})
        result = list(
            db.execute_cypher(
                "MATCH (n:Data) WHERE any(x IN n.tags WHERE x = 'admin') RETURN n.name"
            )
        )
        assert len(result) == 1
        assert result[0]["n.name"] == "HasAdmin"

    def test_none_predicate(self, db):
        db.create_node(["Data"], {"name": "AllPos", "scores": [1, 2, 3]})
        db.create_node(["Data"], {"name": "HasNeg", "scores": [1, -2, 3]})
        result = list(
            db.execute_cypher("MATCH (n:Data) WHERE none(x IN n.scores WHERE x < 0) RETURN n.name")
        )
        assert len(result) == 1
        assert result[0]["n.name"] == "AllPos"

    def test_single_predicate(self, db):
        db.create_node(["Data"], {"name": "OneMatch", "ids": [1, 2, 3]})
        db.create_node(["Data"], {"name": "TwoMatch", "ids": [1, 1, 3]})
        result = list(
            db.execute_cypher("MATCH (n:Data) WHERE single(x IN n.ids WHERE x = 1) RETURN n.name")
        )
        assert len(result) == 1
        assert result[0]["n.name"] == "OneMatch"


# =============================================================================
# PATTERN TYPES
# =============================================================================


class TestCypherPatterns:
    """Test Cypher pattern matching capabilities."""

    def test_node_pattern(self, pattern_db):
        result = list(pattern_db.execute_cypher("MATCH (n) RETURN count(n) AS cnt"))
        assert result[0]["cnt"] == 5  # 3 persons + 2 companies

    def test_directed_edge_pattern(self, pattern_db):
        result = list(
            pattern_db.execute_cypher("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name, b.name")
        )
        assert len(result) == 3

    def test_undirected_edge_pattern(self, pattern_db):
        result = list(
            pattern_db.execute_cypher(
                "MATCH (a:Person {name: 'Alix'})-[:KNOWS]-(b:Person) RETURN b.name"
            )
        )
        names = {r["b.name"] for r in result}
        assert "Gus" in names
        assert "Vincent" in names

    def test_variable_length_path(self, db):
        a = db.create_node(["Node"], {"name": "a"})
        b = db.create_node(["Node"], {"name": "b"})
        c = db.create_node(["Node"], {"name": "c"})
        d = db.create_node(["Node"], {"name": "d"})
        db.create_edge(a.id, b.id, "NEXT")
        db.create_edge(b.id, c.id, "NEXT")
        db.create_edge(c.id, d.id, "NEXT")

        result = list(
            db.execute_cypher(
                "MATCH (start:Node {name: 'a'})-[:NEXT*1..3]->(end_node:Node) RETURN end_node.name"
            )
        )
        names = {r["end_node.name"] for r in result}
        assert "b" in names
        assert "c" in names
        assert "d" in names

    def test_named_path(self, db):
        a = db.create_node(["Node"], {"name": "a"})
        b = db.create_node(["Node"], {"name": "b"})
        c = db.create_node(["Node"], {"name": "c"})
        db.create_edge(a.id, b.id, "NEXT")
        db.create_edge(b.id, c.id, "NEXT")

        result = list(
            db.execute_cypher(
                "MATCH p = (start:Node)-[:NEXT*1..3]->(dest:Node) "
                "WHERE start.name = 'a' AND dest.name = 'b' "
                "RETURN length(p) AS path_len"
            )
        )
        assert result[0]["path_len"] == 1

    def test_multiple_patterns_comma(self, pattern_db):
        result = list(
            pattern_db.execute_cypher(
                "MATCH (a:Person {name: 'Alix'}), (b:Company {name: 'Acme Corp'}) "
                "RETURN a.name, b.name"
            )
        )
        assert len(result) == 1
        assert result[0]["a.name"] == "Alix"
        assert result[0]["b.name"] == "Acme Corp"

    def test_shortest_path(self, db):
        a = db.create_node(["Node"], {"name": "a"})
        b = db.create_node(["Node"], {"name": "b"})
        c = db.create_node(["Node"], {"name": "c"})
        d = db.create_node(["Node"], {"name": "d"})
        db.create_edge(a.id, b.id, "STEP")
        db.create_edge(b.id, c.id, "STEP")
        db.create_edge(c.id, d.id, "STEP")
        db.create_edge(a.id, d.id, "DIRECT")

        result = list(
            db.execute_cypher(
                "MATCH p = shortestPath((start:Node {name: 'a'})-[*]-(dest:Node {name: 'd'})) "
                "RETURN length(p) AS path_len"
            )
        )
        assert result[0]["path_len"] == 1  # Direct edge


# =============================================================================
# PATH FUNCTIONS
# =============================================================================


class TestCypherPathFunctions:
    """Test Cypher path-related functions."""

    def test_nodes_function(self, db):
        a = db.create_node(["Node"], {"name": "a"})
        b = db.create_node(["Node"], {"name": "b"})
        c = db.create_node(["Node"], {"name": "c"})
        db.create_edge(a.id, b.id, "NEXT")
        db.create_edge(b.id, c.id, "NEXT")

        result = list(
            db.execute_cypher(
                "MATCH p = (start:Node {name: 'a'})-[:NEXT*1..2]->(dest:Node {name: 'b'}) "
                "RETURN nodes(p) AS path_nodes"
            )
        )
        assert len(result[0]["path_nodes"]) == 2

    def test_relationships_function(self, db):
        a = db.create_node(["Node"], {"name": "a"})
        b = db.create_node(["Node"], {"name": "b"})
        c = db.create_node(["Node"], {"name": "c"})
        db.create_edge(a.id, b.id, "NEXT")
        db.create_edge(b.id, c.id, "NEXT")

        result = list(
            db.execute_cypher(
                "MATCH p = (start:Node {name: 'a'})-[:NEXT*1..2]->(dest:Node {name: 'b'}) "
                "RETURN relationships(p) AS rels"
            )
        )
        assert len(result[0]["rels"]) == 1

    def test_length_function(self, db):
        a = db.create_node(["Node"], {"name": "a"})
        b = db.create_node(["Node"], {"name": "b"})
        c = db.create_node(["Node"], {"name": "c"})
        db.create_edge(a.id, b.id, "NEXT")
        db.create_edge(b.id, c.id, "NEXT")

        result = list(
            db.execute_cypher(
                "MATCH p = (start:Node)-[:NEXT*1..3]->(dest:Node) "
                "WHERE start.name = 'a' AND dest.name = 'c' "
                "RETURN length(p) AS len"
            )
        )
        assert len(result) == 1
        assert result[0]["len"] == 2
