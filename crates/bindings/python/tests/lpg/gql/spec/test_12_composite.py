"""GQL spec: Composite Query Operations (ISO sec 14.2).

Covers: UNION ALL, UNION DISTINCT, EXCEPT, EXCEPT ALL, EXCEPT DISTINCT,
INTERSECT, INTERSECT ALL, INTERSECT DISTINCT, OTHERWISE.
"""


# =============================================================================
# UNION (sec 14.2)
# =============================================================================


class TestUnion:
    """UNION ALL and UNION DISTINCT."""

    def test_union_all(self, db):
        """UNION ALL keeps duplicates."""
        db.create_node(["Person"], {"name": "Alix"})
        db.create_node(["City"], {"name": "Alix"})  # same name
        result = list(
            db.execute("MATCH (n:Person) RETURN n.name UNION ALL MATCH (n:City) RETURN n.name")
        )
        names = [r["n.name"] for r in result]
        assert names.count("Alix") == 2

    def test_union_distinct(self, db):
        """UNION DISTINCT deduplicates."""
        db.create_node(["Person"], {"name": "Alix"})
        db.create_node(["City"], {"name": "Alix"})
        result = list(
            db.execute("MATCH (n:Person) RETURN n.name UNION DISTINCT MATCH (n:City) RETURN n.name")
        )
        names = [r["n.name"] for r in result]
        assert names.count("Alix") == 1

    def test_union_default_distinct(self, db):
        """UNION (without qualifier) defaults to DISTINCT."""
        db.create_node(["Person"], {"name": "Alix"})
        db.create_node(["City"], {"name": "Alix"})
        result = list(
            db.execute("MATCH (n:Person) RETURN n.name UNION MATCH (n:City) RETURN n.name")
        )
        names = [r["n.name"] for r in result]
        # UNION without qualifier is DISTINCT
        assert names.count("Alix") == 1

    def test_union_different_labels(self, db):
        """UNION combines results from different label queries."""
        db.create_node(["Person"], {"name": "Alix"})
        db.create_node(["City"], {"name": "Amsterdam"})
        result = list(
            db.execute("MATCH (n:Person) RETURN n.name UNION ALL MATCH (n:City) RETURN n.name")
        )
        names = {r["n.name"] for r in result}
        assert names == {"Alix", "Amsterdam"}


# =============================================================================
# EXCEPT (sec 14.2)
# =============================================================================


class TestExcept:
    """EXCEPT, EXCEPT ALL, EXCEPT DISTINCT."""

    def test_except(self, db):
        """EXCEPT set difference."""
        db.create_node(["N"], {"v": 1})
        db.create_node(["N"], {"v": 2})
        db.create_node(["N"], {"v": 3})
        db.create_node(["M"], {"v": 2})
        result = list(db.execute("MATCH (n:N) RETURN n.v EXCEPT MATCH (m:M) RETURN m.v"))
        vals = {r["n.v"] for r in result}
        assert 2 not in vals
        assert 1 in vals
        assert 3 in vals

    def test_except_all(self, db):
        """EXCEPT ALL multiset difference."""
        db.create_node(["N"], {"v": 1})
        db.create_node(["N"], {"v": 2})
        db.create_node(["N"], {"v": 2})
        db.create_node(["N"], {"v": 3})
        db.create_node(["M"], {"v": 2})
        result = list(db.execute("MATCH (n:N) RETURN n.v EXCEPT ALL MATCH (m:M) RETURN m.v"))
        vals = [r["n.v"] for r in result]
        # One copy of 2 removed, one remains
        assert vals.count(2) == 1
        assert 1 in vals
        assert 3 in vals

    def test_except_distinct(self, db):
        """EXCEPT DISTINCT removes all instances."""
        db.create_node(["N"], {"v": 1})
        db.create_node(["N"], {"v": 2})
        db.create_node(["N"], {"v": 2})
        db.create_node(["M"], {"v": 2})
        result = list(db.execute("MATCH (n:N) RETURN n.v EXCEPT DISTINCT MATCH (m:M) RETURN m.v"))
        vals = [r["n.v"] for r in result]
        assert 2 not in vals


# =============================================================================
# INTERSECT (sec 14.2)
# =============================================================================


class TestIntersect:
    """INTERSECT, INTERSECT ALL, INTERSECT DISTINCT."""

    def test_intersect(self, db):
        """INTERSECT set intersection."""
        db.create_node(["N"], {"v": 1})
        db.create_node(["N"], {"v": 2})
        db.create_node(["N"], {"v": 3})
        db.create_node(["M"], {"v": 2})
        db.create_node(["M"], {"v": 3})
        db.create_node(["M"], {"v": 4})
        result = list(db.execute("MATCH (n:N) RETURN n.v INTERSECT MATCH (m:M) RETURN m.v"))
        vals = {r["n.v"] for r in result}
        assert vals == {2, 3}

    def test_intersect_all(self, db):
        """INTERSECT ALL multiset intersection."""
        db.create_node(["N"], {"v": 2})
        db.create_node(["N"], {"v": 2})
        db.create_node(["N"], {"v": 3})
        db.create_node(["M"], {"v": 2})
        db.create_node(["M"], {"v": 3})
        db.create_node(["M"], {"v": 3})
        result = list(db.execute("MATCH (n:N) RETURN n.v INTERSECT ALL MATCH (m:M) RETURN m.v"))
        vals = sorted([r["n.v"] for r in result])
        # min(2,1)=1 copy of 2, min(1,2)=1 copy of 3
        assert vals.count(2) == 1
        assert vals.count(3) == 1

    def test_intersect_distinct(self, db):
        """INTERSECT DISTINCT deduplicates intersection."""
        db.create_node(["N"], {"v": 2})
        db.create_node(["N"], {"v": 2})
        db.create_node(["M"], {"v": 2})
        db.create_node(["M"], {"v": 2})
        result = list(
            db.execute("MATCH (n:N) RETURN n.v INTERSECT DISTINCT MATCH (m:M) RETURN m.v")
        )
        assert len(result) == 1
        assert result[0]["n.v"] == 2


# =============================================================================
# OTHERWISE (sec 14.2)
# =============================================================================


class TestOtherwise:
    """OTHERWISE fallback when left side is empty."""

    def test_otherwise_uses_right(self, db):
        """OTHERWISE returns right side when left is empty."""
        db.create_node(["Fallback"], {"v": "default"})
        result = list(
            db.execute("MATCH (n:NonExistent) RETURN n.v OTHERWISE MATCH (n:Fallback) RETURN n.v")
        )
        assert len(result) == 1
        assert result[0]["n.v"] == "default"

    def test_otherwise_uses_left(self, db):
        """OTHERWISE returns left side when it has results."""
        db.create_node(["Primary"], {"v": "main"})
        db.create_node(["Fallback"], {"v": "default"})
        result = list(
            db.execute("MATCH (n:Primary) RETURN n.v OTHERWISE MATCH (n:Fallback) RETURN n.v")
        )
        assert len(result) == 1
        assert result[0]["n.v"] == "main"

    def test_otherwise_empty_both(self, db):
        """OTHERWISE with both sides empty returns empty."""
        result = list(
            db.execute("MATCH (n:Nothing1) RETURN n.v OTHERWISE MATCH (n:Nothing2) RETURN n.v")
        )
        assert len(result) == 0
