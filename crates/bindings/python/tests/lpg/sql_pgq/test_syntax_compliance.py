"""SQL/PGQ (SQL:2023 GRAPH_TABLE) syntax compliance tests.

Tests SQL/PGQ queries against the Labeled Property Graph model,
verifying compliance with the SQL:2023 GRAPH_TABLE syntax.

Run with:
    pytest tests/python/lpg/sql_pgq/test_syntax_compliance.py -v
"""

import pytest

# Try to import obrain
try:
    from obrain import ObrainDB

    OBRAIN_AVAILABLE = True
except ImportError:
    OBRAIN_AVAILABLE = False


pytestmark = pytest.mark.skipif(not OBRAIN_AVAILABLE, reason="Obrain Python bindings not installed")


class TestBasicNodeQueries:
    """Test basic GRAPH_TABLE node pattern queries."""

    def setup_method(self):
        """Create a database with test data."""
        self.db = ObrainDB()
        self._setup_test_data()

    def _setup_test_data(self):
        """Create a social network graph."""
        self.alix = self.db.create_node(["Person"], {"name": "Alix", "age": 30, "city": "NYC"})
        self.gus = self.db.create_node(["Person"], {"name": "Gus", "age": 25, "city": "LA"})
        self.vincent = self.db.create_node(
            ["Person"], {"name": "Vincent", "age": 35, "city": "NYC"}
        )
        self.acme = self.db.create_node(["Company"], {"name": "Acme Corp", "founded": 2010})
        self.globex = self.db.create_node(["Company"], {"name": "Globex Inc", "founded": 2015})

        self.db.create_edge(self.alix.id, self.gus.id, "KNOWS", {"since": 2020})
        self.db.create_edge(self.gus.id, self.vincent.id, "KNOWS", {"since": 2021})
        self.db.create_edge(self.alix.id, self.vincent.id, "KNOWS", {"since": 2019})

        self.db.create_edge(self.alix.id, self.acme.id, "WORKS_AT", {"role": "Engineer"})
        self.db.create_edge(self.gus.id, self.globex.id, "WORKS_AT", {"role": "Manager"})
        self.db.create_edge(self.vincent.id, self.acme.id, "WORKS_AT", {"role": "Director"})

    def _execute_sql(self, query: str):
        """Execute SQL/PGQ query, skip if not supported."""
        try:
            return self.db.execute_sql(query)
        except AttributeError:
            pytest.skip("SQL/PGQ support not available")
            return None
        except NotImplementedError:
            pytest.skip("SQL/PGQ not implemented")
            return None

    # =========================================================================
    # Basic SELECT with GRAPH_TABLE
    # =========================================================================

    def test_select_star_single_column(self):
        """SQL/PGQ: SELECT * with one COLUMNS entry."""
        result = self._execute_sql(
            "SELECT * FROM GRAPH_TABLE (  MATCH (n:Person)  COLUMNS (n.name AS name))"
        )
        rows = list(result)
        assert len(rows) == 3, "Should find 3 Person nodes"

    def test_select_star_multiple_columns(self):
        """SQL/PGQ: SELECT * with multiple COLUMNS entries."""
        result = self._execute_sql(
            "SELECT * FROM GRAPH_TABLE ("
            "  MATCH (n:Person)"
            "  COLUMNS (n.name AS name, n.age AS age, n.city AS city)"
            ")"
        )
        rows = list(result)
        assert len(rows) == 3, "Should find 3 Person nodes"

    def test_select_specific_column_from_graph_table(self):
        """SQL/PGQ: SELECT specific column with table alias."""
        result = self._execute_sql(
            "SELECT g.name FROM GRAPH_TABLE ("
            "  MATCH (n:Person)"
            "  COLUMNS (n.name AS name, n.age AS age)"
            ") AS g"
        )
        rows = list(result)
        assert len(rows) == 3, "Should find 3 Person nodes"

    def test_node_label_filter(self):
        """SQL/PGQ: MATCH filters by node label."""
        result = self._execute_sql(
            "SELECT * FROM GRAPH_TABLE (  MATCH (c:Company)  COLUMNS (c.name AS company_name))"
        )
        rows = list(result)
        assert len(rows) == 2, "Should find 2 Company nodes"

    # =========================================================================
    # Edge Patterns
    # =========================================================================

    def test_directed_edge_pattern(self):
        """SQL/PGQ: Directed edge pattern with typed edge."""
        result = self._execute_sql(
            "SELECT * FROM GRAPH_TABLE ("
            "  MATCH (a:Person)-[e:KNOWS]->(b:Person)"
            "  COLUMNS (a.name AS person, b.name AS friend)"
            ")"
        )
        rows = list(result)
        assert len(rows) == 3, "Should find 3 KNOWS edges"

    def test_edge_with_property_in_columns(self):
        """SQL/PGQ: Edge property exposed in COLUMNS clause."""
        result = self._execute_sql(
            "SELECT * FROM GRAPH_TABLE ("
            "  MATCH (a:Person)-[e:KNOWS]->(b:Person)"
            "  COLUMNS (a.name AS person, b.name AS friend, e.since AS since)"
            ")"
        )
        rows = list(result)
        assert len(rows) == 3, "Should find 3 KNOWS edges with since property"

    def test_mixed_node_types_in_edge(self):
        """SQL/PGQ: Edge connecting different node types."""
        result = self._execute_sql(
            "SELECT * FROM GRAPH_TABLE ("
            "  MATCH (p:Person)-[w:WORKS_AT]->(c:Company)"
            "  COLUMNS (p.name AS employee, c.name AS company, w.role AS role)"
            ")"
        )
        rows = list(result)
        assert len(rows) == 3, "Should find 3 WORKS_AT edges"

    # =========================================================================
    # WHERE Clause
    # =========================================================================

    def test_where_with_table_alias(self):
        """SQL/PGQ: SQL WHERE clause filters GRAPH_TABLE results."""
        result = self._execute_sql(
            "SELECT g.name FROM GRAPH_TABLE ("
            "  MATCH (n:Person)"
            "  COLUMNS (n.name AS name, n.age AS age)"
            ") AS g"
            " WHERE g.age > 28"
        )
        rows = list(result)
        # Alix (30) and Vincent (35)
        assert len(rows) == 2, "Should find 2 people with age > 28"

    def test_where_string_equality(self):
        """SQL/PGQ: WHERE with string equality check."""
        result = self._execute_sql(
            "SELECT * FROM GRAPH_TABLE ("
            "  MATCH (n:Person)"
            "  COLUMNS (n.name AS name, n.city AS city)"
            ") AS g"
            " WHERE g.city = 'NYC'"
        )
        rows = list(result)
        # Alix and Vincent are in NYC
        assert len(rows) == 2, "Should find 2 people in NYC"

    def test_where_compound_condition(self):
        """SQL/PGQ: WHERE with AND combining multiple conditions."""
        result = self._execute_sql(
            "SELECT * FROM GRAPH_TABLE ("
            "  MATCH (n:Person)"
            "  COLUMNS (n.name AS name, n.age AS age, n.city AS city)"
            ") AS g"
            " WHERE g.city = 'NYC' AND g.age > 31"
        )
        rows = list(result)
        # Only Vincent (35, NYC)
        assert len(rows) == 1, "Should find 1 person in NYC older than 31"

    # =========================================================================
    # ORDER BY
    # =========================================================================

    def test_order_by_ascending(self):
        """SQL/PGQ: ORDER BY ASC sorts results."""
        result = self._execute_sql(
            "SELECT * FROM GRAPH_TABLE ("
            "  MATCH (n:Person)"
            "  COLUMNS (n.name AS name, n.age AS age)"
            ") AS g"
            " ORDER BY g.age ASC"
        )
        rows = list(result)
        assert len(rows) == 3
        ages = [row.get("age") for row in rows]
        assert ages == sorted(ages), "Ages should be in ascending order"

    def test_order_by_descending(self):
        """SQL/PGQ: ORDER BY DESC sorts results in reverse."""
        result = self._execute_sql(
            "SELECT * FROM GRAPH_TABLE ("
            "  MATCH (n:Person)"
            "  COLUMNS (n.name AS name, n.age AS age)"
            ") AS g"
            " ORDER BY g.age DESC"
        )
        rows = list(result)
        assert len(rows) == 3
        ages = [row.get("age") for row in rows]
        assert ages == sorted(ages, reverse=True), "Ages should be in descending order"

    # =========================================================================
    # LIMIT and OFFSET
    # =========================================================================

    def test_limit(self):
        """SQL/PGQ: LIMIT restricts result count."""
        result = self._execute_sql(
            "SELECT * FROM GRAPH_TABLE ("
            "  MATCH (n:Person)"
            "  COLUMNS (n.name AS name, n.age AS age)"
            ") AS g"
            " LIMIT 2"
        )
        rows = list(result)
        assert len(rows) == 2, "LIMIT 2 should return exactly 2 rows"

    def test_limit_with_order_by(self):
        """SQL/PGQ: LIMIT combined with ORDER BY."""
        result = self._execute_sql(
            "SELECT * FROM GRAPH_TABLE ("
            "  MATCH (n:Person)"
            "  COLUMNS (n.name AS name, n.age AS age)"
            ") AS g"
            " ORDER BY g.age DESC"
            " LIMIT 1"
        )
        rows = list(result)
        assert len(rows) == 1, "LIMIT 1 should return exactly 1 row"
        # Oldest person (Vincent, 35) should be first
        assert rows[0].get("name") == "Vincent"

    def test_offset_with_limit(self):
        """SQL/PGQ: OFFSET skips rows before LIMIT applies."""
        result = self._execute_sql(
            "SELECT * FROM GRAPH_TABLE ("
            "  MATCH (n:Person)"
            "  COLUMNS (n.name AS name, n.age AS age)"
            ") AS g"
            " ORDER BY g.age ASC"
            " LIMIT 10"
            " OFFSET 1"
        )
        rows = list(result)
        # 3 total, skip 1 -> 2 remaining
        assert len(rows) == 2, "OFFSET 1 from 3 rows should give 2 rows"

    # =========================================================================
    # Column Aliases
    # =========================================================================

    def test_column_alias_in_columns_clause(self):
        """SQL/PGQ: COLUMNS clause defines output column aliases."""
        result = self._execute_sql(
            "SELECT * FROM GRAPH_TABLE ("
            "  MATCH (n:Person)"
            "  COLUMNS (n.name AS person_name, n.age AS person_age)"
            ")"
        )
        rows = list(result)
        assert len(rows) == 3
        # Verify aliased column names are accessible
        for row in rows:
            assert "person_name" in row, "Should have aliased column person_name"
            assert "person_age" in row, "Should have aliased column person_age"

    def test_edge_and_node_columns_together(self):
        """SQL/PGQ: COLUMNS from both node and edge variables."""
        result = self._execute_sql(
            "SELECT * FROM GRAPH_TABLE ("
            "  MATCH (a:Person)-[e:KNOWS]->(b:Person)"
            "  COLUMNS ("
            "    a.name AS source_name,"
            "    e.since AS relationship_year,"
            "    b.name AS target_name"
            "  )"
            ")"
        )
        rows = list(result)
        assert len(rows) == 3
        for row in rows:
            assert "source_name" in row
            assert "relationship_year" in row
            assert "target_name" in row


class TestVariableLengthPaths:
    """Test variable-length path patterns in SQL/PGQ."""

    def setup_method(self):
        """Create a chain graph: A -> B -> C -> D."""
        self.db = ObrainDB()
        a = self.db.create_node(["Person"], {"name": "A"})
        b = self.db.create_node(["Person"], {"name": "B"})
        c = self.db.create_node(["Person"], {"name": "C"})
        d = self.db.create_node(["Person"], {"name": "D"})

        self.db.create_edge(a.id, b.id, "KNOWS", {})
        self.db.create_edge(b.id, c.id, "KNOWS", {})
        self.db.create_edge(c.id, d.id, "KNOWS", {})

    def _execute_sql(self, query: str):
        """Execute SQL/PGQ query, skip if not supported."""
        try:
            return self.db.execute_sql(query)
        except AttributeError:
            pytest.skip("SQL/PGQ support not available")
            return None
        except NotImplementedError:
            pytest.skip("SQL/PGQ not implemented")
            return None

    def test_variable_length_path(self):
        """SQL/PGQ: Variable-length path with range quantifier."""
        result = self._execute_sql(
            "SELECT * FROM GRAPH_TABLE ("
            "  MATCH (src:Person)-[p:KNOWS*1..3]->(dst:Person)"
            "  COLUMNS (src.name AS source, dst.name AS target)"
            ")"
        )
        rows = list(result)
        # A->B (1), A->C (2), A->D (3), B->C (1), B->D (2), C->D (1)
        assert len(rows) == 6, "Should find 6 variable-length paths"

    def test_length_path_function(self):
        """SQL/PGQ: LENGTH() function on path variable."""
        result = self._execute_sql(
            "SELECT * FROM GRAPH_TABLE ("
            "  MATCH (src:Person)-[p:KNOWS*1..3]->(dst:Person)"
            "  COLUMNS (src.name AS source, LENGTH(p) AS distance, dst.name AS target)"
            ")"
        )
        rows = list(result)
        assert len(rows) == 6
        distances = sorted(row.get("distance") for row in rows)
        # Three 1-hop, two 2-hop, one 3-hop
        assert distances == [1, 1, 1, 2, 2, 3]


class TestAggregatesInColumns:
    """Test aggregate functions within the COLUMNS clause or SQL layer."""

    def setup_method(self):
        """Create a database with test data."""
        self.db = ObrainDB()
        self.db.create_node(["Person"], {"name": "Alix", "age": 30, "city": "NYC"})
        self.db.create_node(["Person"], {"name": "Gus", "age": 25, "city": "LA"})
        self.db.create_node(["Person"], {"name": "Vincent", "age": 35, "city": "NYC"})

    def _execute_sql(self, query: str):
        """Execute SQL/PGQ query, skip if not supported."""
        try:
            return self.db.execute_sql(query)
        except AttributeError:
            pytest.skip("SQL/PGQ support not available")
            return None
        except NotImplementedError:
            pytest.skip("SQL/PGQ not implemented")
            return None

    def test_count_in_outer_select(self):
        """SQL/PGQ: COUNT(*) in outer SQL SELECT on GRAPH_TABLE."""
        result = self._execute_sql(
            "SELECT COUNT(*) AS total FROM GRAPH_TABLE ("
            "  MATCH (n:Person)"
            "  COLUMNS (n.name AS name)"
            ") AS g"
        )
        rows = list(result)
        assert len(rows) == 1, "COUNT should return a single row"
        assert rows[0].get("total") == 3


class TestErrorHandling:
    """Test SQL/PGQ error conditions."""

    def setup_method(self):
        """Create a database with test data."""
        self.db = ObrainDB()
        self.db.create_node(["Person"], {"name": "Alix", "age": 30})

    def _execute_sql(self, query: str):
        """Execute SQL/PGQ query, return result or raise."""
        try:
            return self.db.execute_sql(query)
        except AttributeError:
            pytest.skip("SQL/PGQ support not available")
            return None
        except NotImplementedError:
            pytest.skip("SQL/PGQ not implemented")
            return None

    def test_syntax_error_raises(self):
        """SQL/PGQ: Malformed SQL should raise an error."""
        with pytest.raises(Exception, match=r".+"):
            self.db.execute_sql("SELECT FROM")

    def test_missing_columns_clause_raises(self):
        """SQL/PGQ: GRAPH_TABLE without COLUMNS clause should raise an error."""
        with pytest.raises(Exception, match=r".+"):
            self.db.execute_sql("SELECT * FROM GRAPH_TABLE (  MATCH (n:Person))")
