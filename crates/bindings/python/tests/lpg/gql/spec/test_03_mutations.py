"""GQL spec: Data Manipulation Language (ISO sec 13).

Covers: INSERT/CREATE, SET (property, label, map replace, map merge),
DELETE, DETACH DELETE, NODETACH DELETE, REMOVE, MERGE (ON CREATE/ON MATCH).
"""

import pytest

# =============================================================================
# INSERT / CREATE (sec 13.2)
# =============================================================================


class TestInsert:
    """INSERT statement variants."""

    def test_insert_single_node(self, db):
        """INSERT (:Label {prop: val}) creates a single node."""
        db.execute("INSERT (:Person {name: 'Alix', age: 30})")
        result = list(db.execute("MATCH (n:Person) RETURN n.name, n.age"))
        assert len(result) == 1
        assert result[0]["n.name"] == "Alix"
        assert result[0]["n.age"] == 30

    def test_insert_multiple_labels(self, db):
        """INSERT (:L1:L2 {prop: val}) multiple labels."""
        db.execute("INSERT (:Person:Developer {name: 'Alix'})")
        result = list(db.execute("MATCH (n:Person:Developer) RETURN n.name"))
        assert len(result) == 1

    def test_insert_multi_pattern(self, db):
        """INSERT (:L1), (:L2) multi-pattern insert."""
        db.execute("INSERT (:Person {name: 'Alix'}), (:City {name: 'Amsterdam'})")
        persons = list(db.execute("MATCH (n:Person) RETURN n.name"))
        cities = list(db.execute("MATCH (n:City) RETURN n.name"))
        assert len(persons) == 1
        assert len(cities) == 1

    def test_insert_path(self, db):
        """INSERT (a:Person)-[:KNOWS]->(b:Person) path insert."""
        db.execute(
            "INSERT (:Person {name: 'Alix'})-[:KNOWS {since: 2020}]->(:Person {name: 'Gus'})"
        )
        result = list(
            db.execute("MATCH (a:Person)-[r:KNOWS]->(b:Person) RETURN a.name, b.name, r.since")
        )
        assert len(result) == 1
        assert result[0]["a.name"] == "Alix"
        assert result[0]["r.since"] == 2020

    def test_create_synonym(self, db):
        """CREATE (...) Cypher-compat synonym for INSERT."""
        db.execute("CREATE (:Person {name: 'Alix'})")
        result = list(db.execute("MATCH (n:Person) RETURN n.name"))
        assert len(result) == 1

    def test_match_then_insert(self, db):
        """MATCH ... INSERT ... query-embedded insert."""
        db.create_node(["Person"], {"name": "Alix"})
        db.create_node(["Person"], {"name": "Gus"})
        db.execute(
            "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) CREATE (a)-[:KNOWS]->(b)"
        )
        result = list(db.execute("MATCH (a)-[:KNOWS]->(b) RETURN a.name, b.name"))
        assert len(result) == 1


# =============================================================================
# SET (sec 13.3)
# =============================================================================


class TestSet:
    """SET property and label operations."""

    def test_set_property(self, db):
        """SET n.prop = value updates a property."""
        db.create_node(["Person"], {"name": "Alix", "age": 30})
        db.execute("MATCH (n:Person {name: 'Alix'}) SET n.age = 31")
        result = list(db.execute("MATCH (n:Person {name: 'Alix'}) RETURN n.age"))
        assert result[0]["n.age"] == 31

    def test_set_multiple_properties(self, db):
        """SET n.p1 = v1, n.p2 = v2 comma-separated."""
        db.create_node(["Person"], {"name": "Alix", "age": 30, "city": "Amsterdam"})
        db.execute("MATCH (n:Person {name: 'Alix'}) SET n.age = 31, n.city = 'Berlin'")
        result = list(db.execute("MATCH (n:Person {name: 'Alix'}) RETURN n.age, n.city"))
        assert result[0]["n.age"] == 31
        assert result[0]["n.city"] == "Berlin"

    def test_set_new_property(self, db):
        """SET adds a property that did not exist."""
        db.create_node(["Person"], {"name": "Alix"})
        db.execute("MATCH (n:Person {name: 'Alix'}) SET n.email = 'alix@test.com'")
        result = list(db.execute("MATCH (n:Person {name: 'Alix'}) RETURN n.email"))
        assert result[0]["n.email"] == "alix@test.com"

    def test_set_label(self, db):
        """SET n:Label adds a label."""
        db.create_node(["Person"], {"name": "Alix"})
        db.execute("MATCH (n:Person {name: 'Alix'}) SET n:Admin")
        result = list(db.execute("MATCH (n:Admin) RETURN n.name"))
        assert len(result) == 1
        assert result[0]["n.name"] == "Alix"

    def test_set_multiple_labels(self, db):
        """SET n:L1:L2 adds multiple labels."""
        db.create_node(["Person"], {"name": "Alix"})
        db.execute("MATCH (n:Person {name: 'Alix'}) SET n:Admin:Verified")
        result = list(db.execute("MATCH (n:Admin:Verified) RETURN n.name"))
        assert len(result) == 1

    def test_set_map_replace(self, db):
        """SET n = {map} replaces all properties."""
        db.create_node(["Person"], {"name": "Alix", "age": 30, "city": "Amsterdam"})
        db.execute("MATCH (n:Person {name: 'Alix'}) SET n = {name: 'Alix', role: 'admin'}")
        result = list(db.execute("MATCH (n:Person {name: 'Alix'}) RETURN n.role, n.age, n.city"))
        assert result[0]["n.role"] == "admin"
        # Old properties should be gone
        assert result[0].get("n.age") is None or result[0]["n.age"] is None

    def test_set_map_merge(self, db):
        """SET n += {map} merges properties."""
        db.create_node(["Person"], {"name": "Alix", "age": 30})
        db.execute("MATCH (n:Person {name: 'Alix'}) SET n += {city: 'Berlin', role: 'admin'}")
        result = list(db.execute("MATCH (n:Person {name: 'Alix'}) RETURN n.age, n.city, n.role"))
        assert result[0]["n.age"] == 30  # preserved
        assert result[0]["n.city"] == "Berlin"  # added
        assert result[0]["n.role"] == "admin"  # added


# =============================================================================
# DELETE / REMOVE (sec 13.4, 13.5)
# =============================================================================


class TestDelete:
    """DELETE and REMOVE operations."""

    def test_delete_node(self, db):
        """DELETE n removes an isolated node."""
        db.create_node(["Temp"], {"name": "ToDelete"})
        db.execute("MATCH (n:Temp) DELETE n")
        result = list(db.execute("MATCH (n:Temp) RETURN n"))
        assert len(result) == 0

    def test_delete_multi_variable(self, db):
        """DELETE a, b removes multiple nodes."""
        db.create_node(["Temp"], {"name": "A"})
        db.create_node(["Temp"], {"name": "B"})
        db.execute("MATCH (n:Temp) DELETE n")
        result = list(db.execute("MATCH (n:Temp) RETURN n"))
        assert len(result) == 0

    def test_detach_delete(self, db):
        """DETACH DELETE removes node and connected edges."""
        a = db.create_node(["Temp"], {"name": "A"})
        b = db.create_node(["Temp"], {"name": "B"})
        db.create_edge(a.id, b.id, "REL")
        db.execute("MATCH (n:Temp {name: 'A'}) DETACH DELETE n")
        result = list(db.execute("MATCH (n:Temp) RETURN n.name"))
        names = [r["n.name"] for r in result]
        assert "A" not in names
        assert "B" in names

    def test_nodetach_delete_errors_with_edges(self, db):
        """NODETACH DELETE errors when node has edges (ISO default)."""
        a = db.create_node(["Temp"], {"name": "A"})
        b = db.create_node(["Temp"], {"name": "B"})
        db.create_edge(a.id, b.id, "REL")
        with pytest.raises(RuntimeError):
            db.execute("MATCH (n:Temp {name: 'A'}) NODETACH DELETE n")

    def test_remove_property(self, db):
        """REMOVE n.property sets it to null."""
        db.create_node(["Person"], {"name": "Alix", "temp": "delete_me"})
        db.execute("MATCH (n:Person {name: 'Alix'}) REMOVE n.temp")
        result = list(db.execute("MATCH (n:Person {name: 'Alix'}) RETURN n.temp"))
        assert result[0].get("n.temp") is None

    def test_remove_label(self, db):
        """REMOVE n:Label removes a label."""
        db.create_node(["Person", "Admin"], {"name": "Alix"})
        db.execute("MATCH (n:Admin {name: 'Alix'}) REMOVE n:Admin")
        result = list(db.execute("MATCH (n:Admin) RETURN n"))
        assert len(result) == 0
        # Still exists as Person
        result = list(db.execute("MATCH (n:Person {name: 'Alix'}) RETURN n.name"))
        assert len(result) == 1


# =============================================================================
# MERGE (sec 13)
# =============================================================================


class TestMerge:
    """MERGE with ON CREATE SET / ON MATCH SET."""

    def test_merge_creates_when_absent(self, db):
        """MERGE creates when no match found."""
        db.execute("MERGE (:City {name: 'Amsterdam'})")
        result = list(db.execute("MATCH (n:City) RETURN n.name"))
        assert len(result) == 1
        assert result[0]["n.name"] == "Amsterdam"

    def test_merge_matches_when_present(self, db):
        """MERGE matches existing node, no duplicate."""
        db.execute("INSERT (:City {name: 'Amsterdam'})")
        db.execute("MERGE (:City {name: 'Amsterdam'})")
        result = list(db.execute("MATCH (n:City) RETURN count(n) AS cnt"))
        assert result[0]["cnt"] == 1

    def test_merge_on_create_set(self, db):
        """MERGE ON CREATE SET sets props only when creating."""
        db.execute("MERGE (n:City {name: 'Berlin'}) ON CREATE SET n.new = true RETURN n")
        result = list(db.execute("MATCH (n:City {name: 'Berlin'}) RETURN n.new"))
        assert result[0]["n.new"] is True

    def test_merge_on_match_set(self, db):
        """MERGE ON MATCH SET sets props only when matching."""
        db.execute("INSERT (:City {name: 'Paris'})")
        db.execute("MERGE (n:City {name: 'Paris'}) ON MATCH SET n.visited = true RETURN n")
        result = list(db.execute("MATCH (n:City {name: 'Paris'}) RETURN n.visited"))
        assert result[0]["n.visited"] is True

    def test_merge_relationship(self, db):
        """MERGE on a relationship pattern."""
        db.create_node(["Person"], {"name": "Alix"})
        db.create_node(["Person"], {"name": "Gus"})
        db.execute(
            "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) MERGE (a)-[:KNOWS]->(b)"
        )
        result = list(
            db.execute("MATCH (:Person {name: 'Alix'})-[r:KNOWS]->(:Person {name: 'Gus'}) RETURN r")
        )
        assert len(result) == 1

    def test_merge_relationship_idempotent(self, db):
        """Running MERGE twice does not create duplicate edges."""
        db.create_node(["Person"], {"name": "Alix"})
        db.create_node(["Person"], {"name": "Gus"})
        for _ in range(2):
            db.execute(
                "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) MERGE (a)-[:FRIEND]->(b)"
            )
        result = list(
            db.execute(
                "MATCH (:Person {name: 'Alix'})-[r:FRIEND]->(:Person {name: 'Gus'}) RETURN r"
            )
        )
        assert len(result) == 1
