"""Regression tests for Cypher bugs fixed in 0.5.17.

Bug 1: Correlated EXISTS subquery fails in planner (CRITICAL)
Bug 2: CASE WHEN inside aggregate functions (HIGH)
Bug 3: any() with IN list returns 0 rows (HIGH)
Bug 4: CASE WHEN inside reduce() (MEDIUM)

See .claude/todo/5_beta/cypher-bugs-0.5.17.md for details.
"""


# =============================================================================
# Bug 1: Correlated EXISTS subquery
# =============================================================================


class TestCorrelatedExistsSubquery:
    """Correlated EXISTS subqueries where inner pattern references outer MATCH variables."""

    def test_not_exists_with_type_filter(self, db):
        """NOT EXISTS with WHERE type(r) filter should exclude matched pairs."""
        db.execute_cypher("CREATE (a {name: 'Alix'}), (b {name: 'Gus'}), (c {name: 'Vincent'})")
        db.execute_cypher("MATCH (a {name: 'Alix'}), (b {name: 'Gus'}) CREATE (a)-[:KNOWS]->(b)")
        db.execute_cypher(
            "MATCH (a {name: 'Alix'}), (c {name: 'Vincent'}) CREATE (a)-[:WORKS_WITH]->(c)"
        )

        result = list(
            db.execute_cypher(
                "MATCH (x), (y) WHERE x <> y "
                "AND NOT EXISTS { MATCH (x)-[r]->(y) WHERE type(r) = 'KNOWS' } "
                "RETURN x.name, y.name"
            )
        )
        pairs = {(r["x.name"], r["y.name"]) for r in result}
        # (Alix, Gus) should be excluded (KNOWS edge exists)
        assert ("Alix", "Gus") not in pairs
        # (Alix, Vincent) should be included (edge is WORKS_WITH, not KNOWS)
        assert ("Alix", "Vincent") in pairs

    def test_not_exists_bare_pattern(self, db):
        """NOT EXISTS with bare pattern (no MATCH keyword) should work identically."""
        db.execute_cypher("CREATE (a {name: 'Alix'}), (b {name: 'Gus'}), (c {name: 'Vincent'})")
        db.execute_cypher("MATCH (a {name: 'Alix'}), (b {name: 'Gus'}) CREATE (a)-[:KNOWS]->(b)")
        db.execute_cypher(
            "MATCH (a {name: 'Alix'}), (c {name: 'Vincent'}) CREATE (a)-[:WORKS_WITH]->(c)"
        )

        result = list(
            db.execute_cypher(
                "MATCH (x), (y) WHERE x <> y "
                "AND NOT EXISTS { (x)-[r]->(y) WHERE type(r) = 'KNOWS' } "
                "RETURN x.name, y.name"
            )
        )
        pairs = {(r["x.name"], r["y.name"]) for r in result}
        assert ("Alix", "Gus") not in pairs
        assert ("Alix", "Vincent") in pairs


# =============================================================================
# Bug 2: CASE WHEN inside aggregate functions
# =============================================================================


class TestCaseWhenInAggregate:
    """CASE WHEN expressions used inside aggregate functions like sum()."""

    def test_sum_case_when(self, db):
        """sum(CASE WHEN ... THEN 1 ELSE 0 END) should count matching rows."""
        db.execute_cypher("CREATE (d:Dir {name: 'src'})")
        db.execute_cypher("CREATE (f1:File {name: 'a.py', file_type: 'source'})")
        db.execute_cypher("CREATE (f2:File {name: 'b.md', file_type: 'docs'})")
        db.execute_cypher("CREATE (f3:File {name: 'c.py', file_type: 'source'})")
        db.execute_cypher(
            "MATCH (d:Dir {name: 'src'}), (f:File {name: 'a.py'}) CREATE (d)-[:CONTAINS]->(f)"
        )
        db.execute_cypher(
            "MATCH (d:Dir {name: 'src'}), (f:File {name: 'b.md'}) CREATE (d)-[:CONTAINS]->(f)"
        )
        db.execute_cypher(
            "MATCH (d:Dir {name: 'src'}), (f:File {name: 'c.py'}) CREATE (d)-[:CONTAINS]->(f)"
        )

        result = list(
            db.execute_cypher(
                "MATCH (d:Dir)-[:CONTAINS]->(f:File) "
                "WITH d, count(f) AS total, "
                "     sum(CASE WHEN f.file_type = 'source' THEN 1 ELSE 0 END) AS source_count "
                "RETURN d.name AS name, total, source_count"
            )
        )
        assert len(result) == 1
        assert result[0]["name"] == "src"
        assert result[0]["total"] == 3
        assert result[0]["source_count"] == 2


# =============================================================================
# Bug 3: any() with IN list
# =============================================================================


class TestAnyWithInList:
    """any() list predicate with IN list comparison inside WHERE clause."""

    def test_any_labels_in_list_matches(self, db):
        """any(lbl IN labels(n) WHERE lbl IN ['A', 'B']) should match node with label A."""
        db.execute_cypher("CREATE (:A:B:C {name: 'Test'})")

        result = list(
            db.execute_cypher(
                "MATCH (n) WHERE any(lbl IN labels(n) WHERE lbl IN ['A', 'B']) RETURN n.name"
            )
        )
        assert len(result) == 1
        assert result[0]["n.name"] == "Test"

    def test_any_labels_in_list_no_match(self, db):
        """any() should return 0 rows when no labels match the IN list."""
        db.execute_cypher("CREATE (:A:B:C {name: 'Test'})")

        result = list(
            db.execute_cypher(
                "MATCH (n) WHERE any(lbl IN labels(n) WHERE lbl IN ['X', 'Y']) RETURN n.name"
            )
        )
        assert len(result) == 0

    def test_any_with_single_match(self, db):
        """any() should work when only one label matches from the IN list."""
        db.execute_cypher("CREATE (:Model:Component {name: 'Widget'})")

        result = list(
            db.execute_cypher(
                "MATCH (n) WHERE any(lbl IN labels(n) WHERE lbl IN ['Model', 'Service']) "
                "RETURN n.name"
            )
        )
        assert len(result) == 1
        assert result[0]["n.name"] == "Widget"


# =============================================================================
# Bug 4: CASE WHEN inside reduce()
# =============================================================================


class TestCaseWhenInReduce:
    """CASE WHEN expressions used inside reduce() accumulator body."""

    def test_reduce_with_case_max(self, db):
        """reduce() with CASE WHEN should compute max value."""
        result = list(
            db.execute_cypher(
                "WITH [3, 1, 4, 1, 5] AS vals "
                "RETURN reduce(acc = 0, x IN vals | "
                "  CASE WHEN x > acc THEN x ELSE acc END) AS max_val"
            )
        )
        assert len(result) == 1
        assert result[0]["max_val"] == 5

    def test_reduce_with_case_min(self, db):
        """reduce() with CASE WHEN should compute min value."""
        result = list(
            db.execute_cypher(
                "WITH [3, 1, 4, 1, 5] AS vals "
                "RETURN reduce(acc = 999, x IN vals | "
                "  CASE WHEN x < acc THEN x ELSE acc END) AS min_val"
            )
        )
        assert len(result) == 1
        assert result[0]["min_val"] == 1

    def test_reduce_with_case_conditional_sum(self, db):
        """reduce() with CASE WHEN should conditionally sum values."""
        result = list(
            db.execute_cypher(
                "WITH [1, 2, 3, 4, 5] AS vals "
                "RETURN reduce(acc = 0, x IN vals | "
                "  CASE WHEN x > 2 THEN acc + x ELSE acc END) AS big_sum"
            )
        )
        assert len(result) == 1
        # Only 3+4+5 = 12
        assert result[0]["big_sum"] == 12
