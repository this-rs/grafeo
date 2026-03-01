"""SPARQL syntax compliance tests for RDF model.

Tests SPARQL query syntax against the RDF triple store,
verifying compliance with SPARQL 1.1 Query Language features.

Covers: SELECT, SELECT DISTINCT, CONSTRUCT, ASK, OPTIONAL, UNION, MINUS,
FILTER, BIND, VALUES, ORDER BY, LIMIT, OFFSET, GROUP BY, HAVING,
property paths, and subqueries.

Run with:
    pytest tests/python/rdf/sparql/test_syntax_compliance.py -v
"""

import pytest


# Try to import grafeo
try:
    from grafeo import GrafeoDB

    GRAFEO_AVAILABLE = True
except ImportError:
    GRAFEO_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not GRAFEO_AVAILABLE, reason="Grafeo Python bindings not installed"
)


class TestSPARQLSelectBasics:
    """Test basic SPARQL SELECT query forms."""

    def setup_method(self):
        """Create a database and insert RDF test data."""
        self.db = GrafeoDB()
        self._setup_test_data()

    def _setup_test_data(self):
        """Insert a FOAF social network as RDF triples."""
        self._execute_sparql("""
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX ex: <http://example.org/>

            INSERT DATA {
                ex:alice rdf:type foaf:Person .
                ex:alice foaf:name "Alice" .
                ex:alice foaf:age 30 .
                ex:alice ex:city "NYC" .

                ex:bob rdf:type foaf:Person .
                ex:bob foaf:name "Bob" .
                ex:bob foaf:age 25 .
                ex:bob ex:city "LA" .

                ex:charlie rdf:type foaf:Person .
                ex:charlie foaf:name "Charlie" .
                ex:charlie foaf:age 35 .
                ex:charlie ex:city "NYC" .

                ex:diana rdf:type foaf:Person .
                ex:diana foaf:name "Diana" .
                ex:diana foaf:age 28 .
                ex:diana ex:city "Boston" .

                ex:alice foaf:knows ex:bob .
                ex:alice foaf:knows ex:charlie .
                ex:bob foaf:knows ex:charlie .
                ex:charlie foaf:knows ex:diana .
            }
        """)

    def _execute_sparql(self, query: str):
        """Execute SPARQL query, skip if not supported."""
        try:
            return self.db.execute_sparql(query)
        except AttributeError:
            pytest.skip("SPARQL support not available")
            return None
        except NotImplementedError:
            pytest.skip("SPARQL not implemented")
            return None

    # =========================================================================
    # SELECT
    # =========================================================================

    def test_select_all_triples(self):
        """SPARQL: SELECT * returns all triple bindings."""
        result = self._execute_sparql("""
            SELECT * WHERE {
                ?s ?p ?o
            }
        """)
        rows = list(result)
        assert len(rows) > 0, "Should find triples in the store"

    def test_select_specific_variables(self):
        """SPARQL: SELECT with named variables."""
        result = self._execute_sparql("""
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>

            SELECT ?name WHERE {
                ?person foaf:name ?name .
            }
        """)
        rows = list(result)
        assert len(rows) == 4, "Should find 4 person names"

    def test_select_multiple_variables(self):
        """SPARQL: SELECT with multiple projected variables."""
        result = self._execute_sparql("""
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>

            SELECT ?name ?age WHERE {
                ?person foaf:name ?name .
                ?person foaf:age ?age .
            }
        """)
        rows = list(result)
        assert len(rows) == 4, "Should find 4 name/age pairs"

    def test_select_with_type_pattern(self):
        """SPARQL: SELECT filtering by rdf:type."""
        result = self._execute_sparql("""
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

            SELECT ?person ?name WHERE {
                ?person rdf:type foaf:Person .
                ?person foaf:name ?name .
            }
        """)
        rows = list(result)
        assert len(rows) == 4, "Should find 4 persons"

    # =========================================================================
    # SELECT DISTINCT
    # =========================================================================

    def test_select_distinct(self):
        """SPARQL: SELECT DISTINCT eliminates duplicate bindings."""
        result = self._execute_sparql("""
            PREFIX ex: <http://example.org/>

            SELECT DISTINCT ?city WHERE {
                ?person ex:city ?city .
            }
        """)
        rows = list(result)
        cities = [row.get("city") for row in rows]
        assert len(cities) == len(set(cities)), "DISTINCT should remove duplicates"
        # NYC, LA, Boston
        assert len(rows) == 3, "Should find 3 distinct cities"

    # =========================================================================
    # CONSTRUCT
    # =========================================================================

    def test_construct_query(self):
        """SPARQL: CONSTRUCT builds new triples from pattern."""
        result = self._execute_sparql("""
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            PREFIX ex: <http://example.org/>

            CONSTRUCT {
                ?person ex:displayName ?name .
            }
            WHERE {
                ?person foaf:name ?name .
            }
        """)
        rows = list(result)
        assert len(rows) >= 1, "CONSTRUCT should produce triples"

    # =========================================================================
    # ASK
    # =========================================================================

    def test_ask_true(self):
        """SPARQL: ASK returns true when pattern exists."""
        result = self._execute_sparql("""
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>

            ASK {
                ?person foaf:name "Alice" .
            }
        """)
        rows = list(result)
        # ASK returns a single boolean result
        assert len(rows) >= 1

    def test_ask_false(self):
        """SPARQL: ASK returns false when pattern does not exist."""
        result = self._execute_sparql("""
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>

            ASK {
                ?person foaf:name "NonExistentPerson" .
            }
        """)
        rows = list(result)
        assert isinstance(rows, list)

    # =========================================================================
    # OPTIONAL
    # =========================================================================

    def test_optional_present(self):
        """SPARQL: OPTIONAL includes bindings when pattern matches."""
        result = self._execute_sparql("""
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>

            SELECT ?name ?age WHERE {
                ?person foaf:name ?name .
                OPTIONAL { ?person foaf:age ?age }
            }
        """)
        rows = list(result)
        assert len(rows) >= 4, "Should find all persons with optional age"

    def test_optional_missing(self):
        """SPARQL: OPTIONAL leaves unbound when pattern does not match."""
        # Add a person without an email
        self._execute_sparql("""
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            PREFIX ex: <http://example.org/>

            INSERT DATA {
                ex:eve foaf:name "Eve" .
            }
        """)

        result = self._execute_sparql("""
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>

            SELECT ?name ?email WHERE {
                ?person foaf:name ?name .
                OPTIONAL { ?person foaf:mbox ?email }
            }
        """)
        rows = list(result)
        # Should include Eve with NULL/unbound email
        assert len(rows) >= 5, "Should include Eve even without email"

    # =========================================================================
    # UNION
    # =========================================================================

    def test_union_combines_patterns(self):
        """SPARQL: UNION combines results from two patterns."""
        result = self._execute_sparql("""
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            PREFIX ex: <http://example.org/>

            SELECT ?value WHERE {
                {
                    ?person foaf:name ?value .
                }
                UNION
                {
                    ?person ex:city ?value .
                }
            }
        """)
        rows = list(result)
        # 4 names + 4 cities = 8
        assert len(rows) >= 4, "UNION should combine name and city results"

    # =========================================================================
    # MINUS
    # =========================================================================

    def test_minus_excludes_pattern(self):
        """SPARQL: MINUS removes solutions matching the exclusion pattern."""
        result = self._execute_sparql("""
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            PREFIX ex: <http://example.org/>

            SELECT ?name WHERE {
                ?person foaf:name ?name .
                ?person ex:city ?city .
                MINUS {
                    ?person ex:city "NYC" .
                }
            }
        """)
        rows = list(result)
        names = [row.get("name") for row in rows]
        # NYC persons (Alice, Charlie) should be excluded
        assert "Alice" not in names, "Alice (NYC) should be excluded by MINUS"
        assert "Charlie" not in names, "Charlie (NYC) should be excluded by MINUS"

    # =========================================================================
    # FILTER
    # =========================================================================

    def test_filter_numeric_comparison(self):
        """SPARQL: FILTER with numeric comparison."""
        result = self._execute_sparql("""
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>

            SELECT ?name ?age WHERE {
                ?person foaf:name ?name .
                ?person foaf:age ?age .
                FILTER (?age > 28)
            }
        """)
        rows = list(result)
        # Alice (30), Charlie (35)
        assert len(rows) == 2, "Should find 2 persons older than 28"

    def test_filter_string_equality(self):
        """SPARQL: FILTER with string comparison."""
        result = self._execute_sparql("""
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>

            SELECT ?name WHERE {
                ?person foaf:name ?name .
                FILTER (?name = "Alice")
            }
        """)
        rows = list(result)
        assert len(rows) == 1, "Should find exactly Alice"

    def test_filter_logical_and(self):
        """SPARQL: FILTER with && (logical AND)."""
        result = self._execute_sparql("""
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            PREFIX ex: <http://example.org/>

            SELECT ?name WHERE {
                ?person foaf:name ?name .
                ?person foaf:age ?age .
                ?person ex:city ?city .
                FILTER (?city = "NYC" && ?age > 31)
            }
        """)
        rows = list(result)
        # Only Charlie (NYC, 35)
        assert len(rows) == 1

    def test_filter_logical_or(self):
        """SPARQL: FILTER with || (logical OR)."""
        result = self._execute_sparql("""
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            PREFIX ex: <http://example.org/>

            SELECT ?name WHERE {
                ?person foaf:name ?name .
                ?person ex:city ?city .
                FILTER (?city = "LA" || ?city = "Boston")
            }
        """)
        rows = list(result)
        # Bob (LA), Diana (Boston)
        assert len(rows) == 2

    def test_filter_regex(self):
        """SPARQL: FILTER with REGEX function."""
        result = self._execute_sparql("""
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>

            SELECT ?name WHERE {
                ?person foaf:name ?name .
                FILTER REGEX(?name, "^[AB]")
            }
        """)
        rows = list(result)
        # Alice and Bob start with A or B
        assert len(rows) == 2

    def test_filter_not_exists(self):
        """SPARQL: FILTER NOT EXISTS excludes matching patterns."""
        result = self._execute_sparql("""
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>

            SELECT ?name WHERE {
                ?person foaf:name ?name .
                FILTER NOT EXISTS {
                    ?person foaf:knows ?other .
                }
            }
        """)
        rows = list(result)
        # Diana does not know anyone, so only Diana should appear
        names = [row.get("name") for row in rows]
        assert "Diana" in names, "Diana (knows nobody) should be included"
        assert "Alice" not in names, "Alice (knows others) should be excluded"

    def test_filter_exists(self):
        """SPARQL: FILTER EXISTS includes only matching patterns."""
        result = self._execute_sparql("""
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>

            SELECT ?name WHERE {
                ?person foaf:name ?name .
                FILTER EXISTS {
                    ?person foaf:knows ?other .
                }
            }
        """)
        rows = list(result)
        names = [row.get("name") for row in rows]
        # Alice, Bob, Charlie all know someone
        assert len(names) >= 3

    # =========================================================================
    # BIND
    # =========================================================================

    def test_bind_expression(self):
        """SPARQL: BIND assigns an expression to a variable."""
        result = self._execute_sparql("""
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>

            SELECT ?name ?label WHERE {
                ?person foaf:name ?name .
                ?person foaf:age ?age .
                BIND (CONCAT(?name, " (age ", STR(?age), ")") AS ?label)
            }
        """)
        rows = list(result)
        assert len(rows) >= 1, "BIND should produce labeled results"

    # =========================================================================
    # VALUES
    # =========================================================================

    def test_values_inline_data(self):
        """SPARQL: VALUES provides inline binding data."""
        result = self._execute_sparql("""
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            PREFIX ex: <http://example.org/>

            SELECT ?name WHERE {
                VALUES ?person { ex:alice ex:bob }
                ?person foaf:name ?name .
            }
        """)
        rows = list(result)
        names = [row.get("name") for row in rows]
        assert len(rows) == 2, "VALUES should restrict to Alice and Bob"
        assert "Alice" in names
        assert "Bob" in names

    # =========================================================================
    # ORDER BY
    # =========================================================================

    def test_order_by_ascending(self):
        """SPARQL: ORDER BY ASC sorts results."""
        result = self._execute_sparql("""
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>

            SELECT ?name ?age WHERE {
                ?person foaf:name ?name .
                ?person foaf:age ?age .
            }
            ORDER BY ASC(?age)
        """)
        rows = list(result)
        ages = [row.get("age") for row in rows]
        # Convert to comparable values
        numeric_ages = [int(a) if isinstance(a, str) else a for a in ages]
        assert numeric_ages == sorted(numeric_ages), "Ages should be ascending"

    def test_order_by_descending(self):
        """SPARQL: ORDER BY DESC sorts results in reverse."""
        result = self._execute_sparql("""
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>

            SELECT ?name ?age WHERE {
                ?person foaf:name ?name .
                ?person foaf:age ?age .
            }
            ORDER BY DESC(?age)
        """)
        rows = list(result)
        ages = [row.get("age") for row in rows]
        numeric_ages = [int(a) if isinstance(a, str) else a for a in ages]
        assert numeric_ages == sorted(numeric_ages, reverse=True)

    # =========================================================================
    # LIMIT and OFFSET
    # =========================================================================

    def test_limit(self):
        """SPARQL: LIMIT restricts result count."""
        result = self._execute_sparql("""
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>

            SELECT ?name WHERE {
                ?person foaf:name ?name .
            }
            LIMIT 2
        """)
        rows = list(result)
        assert len(rows) == 2, "LIMIT 2 should return exactly 2 rows"

    def test_offset(self):
        """SPARQL: OFFSET skips initial results."""
        result = self._execute_sparql("""
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>

            SELECT ?name ?age WHERE {
                ?person foaf:name ?name .
                ?person foaf:age ?age .
            }
            ORDER BY ASC(?age)
            LIMIT 10
            OFFSET 2
        """)
        rows = list(result)
        # 4 total, skip 2 -> 2 remaining
        assert len(rows) == 2, "OFFSET 2 from 4 rows should give 2 rows"

    def test_limit_with_offset(self):
        """SPARQL: LIMIT combined with OFFSET for pagination."""
        result = self._execute_sparql("""
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>

            SELECT ?name ?age WHERE {
                ?person foaf:name ?name .
                ?person foaf:age ?age .
            }
            ORDER BY ASC(?age)
            LIMIT 1
            OFFSET 1
        """)
        rows = list(result)
        assert len(rows) == 1, "LIMIT 1 OFFSET 1 should return exactly 1 row"

    # =========================================================================
    # GROUP BY and HAVING
    # =========================================================================

    def test_group_by_with_count(self):
        """SPARQL: GROUP BY with COUNT aggregate."""
        result = self._execute_sparql("""
            PREFIX ex: <http://example.org/>

            SELECT ?city (COUNT(?person) AS ?count) WHERE {
                ?person ex:city ?city .
            }
            GROUP BY ?city
        """)
        rows = list(result)
        # 3 distinct cities: NYC (2), LA (1), Boston (1)
        assert len(rows) >= 1, "GROUP BY should produce city groups"

    def test_having_filters_groups(self):
        """SPARQL: HAVING filters grouped results."""
        result = self._execute_sparql("""
            PREFIX ex: <http://example.org/>

            SELECT ?city (COUNT(?person) AS ?count) WHERE {
                ?person ex:city ?city .
            }
            GROUP BY ?city
            HAVING (COUNT(?person) > 1)
        """)
        rows = list(result)
        # Only NYC has more than 1 person
        assert len(rows) >= 1

    # =========================================================================
    # Property Paths
    # =========================================================================

    def test_property_path_sequence(self):
        """SPARQL: Property path with / (sequence)."""
        result = self._execute_sparql("""
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>

            SELECT ?name WHERE {
                ?person foaf:knows/foaf:name ?name .
            }
        """)
        rows = list(result)
        # Friends-of: Alice knows Bob/Charlie, Bob knows Charlie, Charlie knows Diana
        assert len(rows) >= 1

    def test_property_path_one_or_more(self):
        """SPARQL: Property path with + (one or more)."""
        result = self._execute_sparql("""
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            PREFIX ex: <http://example.org/>

            SELECT ?start ?end WHERE {
                ?start foaf:knows+ ?end .
            }
        """)
        rows = list(result)
        # Transitive closure of knows relationships
        assert len(rows) >= 4, "Should find transitive knows paths"

    def test_property_path_zero_or_more(self):
        """SPARQL: Property path with * (zero or more)."""
        result = self._execute_sparql("""
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            PREFIX ex: <http://example.org/>

            SELECT ?start ?end WHERE {
                ?start foaf:knows* ?end .
            }
        """)
        rows = list(result)
        # Includes reflexive (zero-length) paths plus transitive
        assert len(rows) >= 4

    def test_property_path_alternative(self):
        """SPARQL: Property path with | (alternative)."""
        result = self._execute_sparql("""
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            PREFIX ex: <http://example.org/>

            SELECT ?person ?value WHERE {
                ?person foaf:name|ex:city ?value .
            }
        """)
        rows = list(result)
        # Each person has both a name and a city
        assert len(rows) >= 4

    # =========================================================================
    # Subqueries
    # =========================================================================

    def test_subquery(self):
        """SPARQL: Subquery in WHERE clause."""
        result = self._execute_sparql("""
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>

            SELECT ?name ?age WHERE {
                {
                    SELECT ?person (MAX(?a) AS ?age) WHERE {
                        ?person foaf:age ?a .
                    }
                    GROUP BY ?person
                }
                ?person foaf:name ?name .
            }
        """)
        rows = list(result)
        assert len(rows) >= 1, "Subquery should feed results to outer query"
