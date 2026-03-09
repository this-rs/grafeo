"""GQL spec: Predicates (ISO sec 19).

Covers: comparison, EXISTS, IS NULL, IS TYPED, IS DIRECTED, IS LABELED,
IS SOURCE OF, IS DESTINATION OF, ALL_DIFFERENT, SAME, PROPERTY_EXISTS,
IS NORMALIZED.
"""


# =============================================================================
# Comparison Predicate (sec 19.3)
# =============================================================================


class TestComparisonPredicate:
    """Basic comparison predicates."""

    def test_equals(self, db):
        db.create_node(["N"], {"v": 10})
        result = list(db.execute("MATCH (n:N) WHERE n.v = 10 RETURN n.v"))
        assert len(result) == 1

    def test_not_equals(self, db):
        db.create_node(["N"], {"v": 10})
        db.create_node(["N"], {"v": 20})
        result = list(db.execute("MATCH (n:N) WHERE n.v <> 10 RETURN n.v"))
        assert result[0]["n.v"] == 20

    def test_less_than(self, db):
        db.create_node(["N"], {"v": 5})
        db.create_node(["N"], {"v": 15})
        result = list(db.execute("MATCH (n:N) WHERE n.v < 10 RETURN n.v"))
        assert result[0]["n.v"] == 5

    def test_greater_equal(self, db):
        db.create_node(["N"], {"v": 10})
        db.create_node(["N"], {"v": 5})
        result = list(db.execute("MATCH (n:N) WHERE n.v >= 10 RETURN n.v"))
        assert result[0]["n.v"] == 10


# =============================================================================
# EXISTS Predicate (sec 19.4)
# =============================================================================


class TestExistsPredicate:
    """EXISTS { pattern } and EXISTS { subquery }."""

    def test_exists_pattern(self, db):
        """EXISTS { MATCH pattern } subquery."""
        a = db.create_node(["Person"], {"name": "Alix"})
        b = db.create_node(["Person"], {"name": "Gus"})
        db.create_node(["Person"], {"name": "Vincent"})
        db.create_edge(a.id, b.id, "KNOWS")
        result = list(
            db.execute("MATCH (p:Person) WHERE EXISTS { MATCH (p)-[:KNOWS]->() } RETURN p.name")
        )
        names = {r["p.name"] for r in result}
        assert "Alix" in names
        assert "Vincent" not in names

    def test_exists_with_where(self, db):
        """EXISTS { MATCH ... WHERE ... } correlated subquery."""
        a = db.create_node(["Person"], {"name": "Alix", "age": 30})
        b = db.create_node(["Person"], {"name": "Gus", "age": 25})
        db.create_edge(a.id, b.id, "KNOWS")
        result = list(
            db.execute(
                "MATCH (p:Person) "
                "WHERE EXISTS { MATCH (p)-[:KNOWS]->(q) WHERE q.age < 30 } "
                "RETURN p.name"
            )
        )
        assert len(result) >= 1
        assert result[0]["p.name"] == "Alix"

    def test_not_exists(self, db):
        """NOT EXISTS subquery."""
        a = db.create_node(["Person"], {"name": "Alix"})
        b = db.create_node(["Person"], {"name": "Gus"})
        db.create_edge(a.id, b.id, "KNOWS")
        result = list(
            db.execute("MATCH (p:Person) WHERE NOT EXISTS { MATCH (p)-[:KNOWS]->() } RETURN p.name")
        )
        names = {r["p.name"] for r in result}
        assert "Gus" in names
        assert "Alix" not in names


# =============================================================================
# IS NULL / IS NOT NULL (sec 19.5)
# =============================================================================


class TestNullPredicates:
    """IS NULL and IS NOT NULL."""

    def test_is_null(self, db):
        db.create_node(["N"], {"name": "WithProp", "v": 1})
        db.create_node(["N"], {"name": "NoProp"})
        result = list(db.execute("MATCH (n:N) WHERE n.v IS NULL RETURN n.name"))
        assert result[0]["n.name"] == "NoProp"

    def test_is_not_null(self, db):
        db.create_node(["N"], {"name": "WithProp", "v": 1})
        db.create_node(["N"], {"name": "NoProp"})
        result = list(db.execute("MATCH (n:N) WHERE n.v IS NOT NULL RETURN n.name"))
        assert result[0]["n.name"] == "WithProp"


# =============================================================================
# IS TYPED / IS NOT TYPED (sec 19.6)
# =============================================================================


class TestTypedPredicates:
    """IS TYPED and IS NOT TYPED."""

    def test_is_typed_int(self, db):
        db.create_node(["N"], {"v": 42})
        db.create_node(["N"], {"v": "text"})
        result = list(db.execute("MATCH (n:N) WHERE n.v IS TYPED INT64 RETURN n.v"))
        assert len(result) == 1
        assert result[0]["n.v"] == 42

    def test_is_not_typed_string(self, db):
        db.create_node(["N"], {"v": 42})
        db.create_node(["N"], {"v": "text"})
        result = list(db.execute("MATCH (n:N) WHERE n.v IS NOT TYPED STRING RETURN n.v"))
        assert len(result) == 1
        assert result[0]["n.v"] == 42


# =============================================================================
# IS DIRECTED / IS NOT DIRECTED (sec 19.8)
# =============================================================================


class TestDirectedPredicates:
    """IS DIRECTED and IS NOT DIRECTED on edges."""

    def test_is_directed(self, db):
        a = db.create_node(["N"], {"name": "a"})
        b = db.create_node(["N"], {"name": "b"})
        db.create_edge(a.id, b.id, "REL")
        result = list(db.execute("MATCH (a)-[e]-(b) WHERE e IS DIRECTED RETURN type(e) AS t"))
        assert len(result) >= 1

    def test_is_not_directed(self, db):
        a = db.create_node(["N"], {"name": "a"})
        b = db.create_node(["N"], {"name": "b"})
        db.create_edge(a.id, b.id, "REL")
        result = list(db.execute("MATCH (a)-[e]-(b) WHERE e IS NOT DIRECTED RETURN type(e) AS t"))
        # All edges in Grafeo are directed, so this should be empty
        assert len(result) == 0


# =============================================================================
# IS LABELED / IS NOT LABELED (sec 19.9)
# =============================================================================


class TestLabeledPredicates:
    """IS LABELED and IS NOT LABELED."""

    def test_is_labeled(self, db):
        db.create_node(["Person"], {"name": "Alix"})
        db.create_node(["City"], {"name": "Amsterdam"})
        result = list(db.execute("MATCH (n) WHERE n IS LABELED Person RETURN n.name"))
        assert len(result) == 1
        assert result[0]["n.name"] == "Alix"

    def test_is_not_labeled(self, db):
        db.create_node(["Person"], {"name": "Alix"})
        db.create_node(["City"], {"name": "Amsterdam"})
        result = list(db.execute("MATCH (n) WHERE n IS NOT LABELED Person RETURN n.name"))
        assert result[0]["n.name"] == "Amsterdam"


# =============================================================================
# IS SOURCE OF / IS DESTINATION OF (sec 19.10)
# =============================================================================


class TestEndpointPredicates:
    """IS SOURCE OF and IS DESTINATION OF."""

    def test_is_source_of(self, db):
        a = db.create_node(["Person"], {"name": "Alix"})
        b = db.create_node(["Person"], {"name": "Gus"})
        db.create_edge(a.id, b.id, "KNOWS")
        result = list(db.execute("MATCH (a)-[e:KNOWS]->(b) WHERE a IS SOURCE OF e RETURN a.name"))
        assert result[0]["a.name"] == "Alix"

    def test_is_destination_of(self, db):
        a = db.create_node(["Person"], {"name": "Alix"})
        b = db.create_node(["Person"], {"name": "Gus"})
        db.create_edge(a.id, b.id, "KNOWS")
        result = list(
            db.execute("MATCH (a)-[e:KNOWS]->(b) WHERE b IS DESTINATION OF e RETURN b.name")
        )
        assert result[0]["b.name"] == "Gus"


# =============================================================================
# ALL_DIFFERENT / SAME (sec 19.11, 19.12)
# =============================================================================


class TestElementPredicates:
    """ALL_DIFFERENT and SAME predicates."""

    def test_all_different(self, db):
        a = db.create_node(["N"], {"name": "a"})
        b = db.create_node(["N"], {"name": "b"})
        c = db.create_node(["N"], {"name": "c"})
        db.create_edge(a.id, b.id, "REL")
        db.create_edge(b.id, c.id, "REL")
        result = list(
            db.execute(
                "MATCH (x)-[]->(y)-[]->(z) "
                "WHERE ALL_DIFFERENT(x, y, z) "
                "RETURN x.name, y.name, z.name"
            )
        )
        assert len(result) >= 1
        for r in result:
            assert len({r["x.name"], r["y.name"], r["z.name"]}) == 3

    def test_same(self, db):
        """SAME(a, b) checks element identity."""
        db.create_node(["N"], {"name": "a"})
        result = list(
            db.execute("MATCH (x:N {name: 'a'}), (y:N {name: 'a'}) WHERE SAME(x, y) RETURN x.name")
        )
        assert len(result) == 1


# =============================================================================
# PROPERTY_EXISTS (sec 19.13)
# =============================================================================


class TestPropertyExists:
    """PROPERTY_EXISTS predicate."""

    def test_property_exists_true(self, db):
        db.create_node(["N"], {"name": "Alix", "age": 30})
        result = list(db.execute("MATCH (n:N) WHERE PROPERTY_EXISTS(n, 'age') RETURN n.name"))
        assert result[0]["n.name"] == "Alix"

    def test_property_exists_false(self, db):
        db.create_node(["N"], {"name": "Alix"})
        result = list(db.execute("MATCH (n:N) WHERE PROPERTY_EXISTS(n, 'age') RETURN n.name"))
        assert len(result) == 0


# =============================================================================
# IS NORMALIZED (sec 19.7)
# =============================================================================


class TestNormalized:
    """IS NORMALIZED predicate for Unicode normalization."""

    def test_is_normalized_nfc(self, db):
        """IS NFC NORMALIZED for standard text."""
        db.create_node(["N"], {"v": "hello"})
        result = list(db.execute("MATCH (n:N) WHERE n.v IS NFC NORMALIZED RETURN n.v"))
        assert len(result) == 1

    def test_is_normalized_default(self, db):
        """IS NORMALIZED (default NFC)."""
        db.create_node(["N"], {"v": "hello"})
        result = list(db.execute("MATCH (n:N) WHERE n.v IS NORMALIZED RETURN n.v"))
        assert len(result) == 1
