"""Comprehensive Cypher tests covering all query patterns used by Deriva.

Deriva uses Obrain as an embedded graph database for ArchiMate model generation.
It relies on a dual-namespace pattern (Graph: for source extraction, Model: for
ArchiMate elements) and exercises a wide range of Cypher features.

Each test class corresponds to a category of Cypher patterns from Deriva.
"""


# =============================================================================
# Helpers
# =============================================================================


def setup_dual_namespace_graph(db):
    """Create a graph mimicking Deriva's dual-namespace pattern.

    Graph namespace: source code elements (active/inactive).
    Model namespace: ArchiMate elements (enabled/disabled) with source_identifier links.
    """
    # Graph-namespace nodes (source extraction)
    db.execute_cypher("CREATE (:GraphNode {id: 'g1', name: 'AuthService', active: true})")
    db.execute_cypher("CREATE (:GraphNode {id: 'g2', name: 'LoginModule', active: true})")
    db.execute_cypher("CREATE (:GraphNode {id: 'g3', name: 'Database', active: true})")
    db.execute_cypher("CREATE (:GraphNode {id: 'g4', name: 'OldService', active: false})")

    # Graph-namespace edges (source relationships)
    db.execute_cypher(
        "MATCH (a:GraphNode {id: 'g1'}), (b:GraphNode {id: 'g2'}) CREATE (a)-[:CONTAINS]->(b)"
    )
    db.execute_cypher(
        "MATCH (a:GraphNode {id: 'g2'}), (b:GraphNode {id: 'g3'}) CREATE (a)-[:CALLS]->(b)"
    )
    db.execute_cypher(
        "MATCH (a:GraphNode {id: 'g1'}), (b:GraphNode {id: 'g3'}) CREATE (a)-[:USES]->(b)"
    )

    # Model-namespace nodes (ArchiMate elements)
    db.execute_cypher(
        "CREATE (:Model:ApplicationComponent {identifier: 'm1', name: 'AuthService', "
        "enabled: true, source_identifier: 'g1', element_type: 'ApplicationComponent', "
        "confidence: 0.9})"
    )
    db.execute_cypher(
        "CREATE (:Model:ApplicationComponent {identifier: 'm2', name: 'LoginModule', "
        "enabled: true, source_identifier: 'g2', element_type: 'ApplicationComponent', "
        "confidence: 0.8})"
    )
    db.execute_cypher(
        "CREATE (:Model:DataObject {identifier: 'm3', name: 'Database', "
        "enabled: true, source_identifier: 'g3', element_type: 'DataObject', "
        "confidence: 0.7})"
    )
    db.execute_cypher(
        "CREATE (:Model:ApplicationComponent {identifier: 'm4', name: 'OrphanElement', "
        "enabled: true, source_identifier: 'g_none', element_type: 'ApplicationComponent', "
        "confidence: 0.5})"
    )
    db.execute_cypher(
        "CREATE (:Model:ApplicationComponent {identifier: 'm5', name: 'DisabledElement', "
        "enabled: false, source_identifier: 'g1', element_type: 'ApplicationComponent', "
        "confidence: 0.3})"
    )

    # Model-namespace relationships
    db.execute_cypher(
        "MATCH (a:Model {identifier: 'm1'}), (b:Model {identifier: 'm2'}) "
        "CREATE (a)-[:Composition {identifier: 'r1', confidence: 0.85}]->(b)"
    )
    db.execute_cypher(
        "MATCH (a:Model {identifier: 'm2'}), (b:Model {identifier: 'm3'}) "
        "CREATE (a)-[:Flow {identifier: 'r2', confidence: 0.7}]->(b)"
    )


# =============================================================================
# 1. Basic Node Operations
# =============================================================================


class TestBasicNodeOperations:
    """MATCH, CREATE, MERGE, SET, DELETE patterns."""

    def test_create_node_with_labels_and_properties(self, db):
        """CREATE node with multiple labels and properties."""
        db.execute_cypher("CREATE (:Person:Employee {name: 'Alix', age: 30, city: 'Amsterdam'})")
        result = list(db.execute_cypher("MATCH (n:Person:Employee) RETURN n.name, n.age, n.city"))
        assert len(result) == 1
        assert result[0]["n.name"] == "Alix"
        assert result[0]["n.age"] == 30
        assert result[0]["n.city"] == "Amsterdam"

    def test_merge_creates_if_not_exists(self, db):
        """MERGE creates node when it doesn't exist."""
        db.execute_cypher("MERGE (n:Person {name: 'Gus'}) RETURN n.name")
        result = list(db.execute_cypher("MATCH (n:Person) RETURN n.name"))
        assert len(result) == 1
        assert result[0]["n.name"] == "Gus"

    def test_merge_idempotent(self, db):
        """MERGE does not duplicate when node already exists."""
        db.execute_cypher("MERGE (n:Person {name: 'Gus'})")
        db.execute_cypher("MERGE (n:Person {name: 'Gus'})")
        result = list(db.execute_cypher("MATCH (n:Person {name: 'Gus'}) RETURN n.name"))
        assert len(result) == 1

    def test_set_single_property(self, db):
        """SET updates a single property."""
        db.execute_cypher("CREATE (:Person {name: 'Alix', age: 30})")
        db.execute_cypher("MATCH (n:Person {name: 'Alix'}) SET n.age = 31")
        result = list(db.execute_cypher("MATCH (n:Person {name: 'Alix'}) RETURN n.age"))
        assert result[0]["n.age"] == 31

    def test_set_multiple_properties(self, db):
        """SET updates multiple properties in one clause."""
        db.execute_cypher("CREATE (:Person {name: 'Alix', age: 30, city: 'Amsterdam'})")
        db.execute_cypher("MATCH (n:Person {name: 'Alix'}) SET n.age = 31, n.city = 'Berlin'")
        result = list(db.execute_cypher("MATCH (n:Person {name: 'Alix'}) RETURN n.age, n.city"))
        assert result[0]["n.age"] == 31
        assert result[0]["n.city"] == "Berlin"

    def test_detach_delete(self, db):
        """DETACH DELETE removes node and its relationships."""
        db.execute_cypher("CREATE (a:Person {name: 'Alix'})")
        db.execute_cypher("CREATE (b:Person {name: 'Gus'})")
        db.execute_cypher(
            "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) CREATE (a)-[:KNOWS]->(b)"
        )
        db.execute_cypher("MATCH (n:Person {name: 'Alix'}) DETACH DELETE n")
        result = list(db.execute_cypher("MATCH (n:Person) RETURN n.name"))
        names = [r["n.name"] for r in result]
        assert "Alix" not in names
        assert "Gus" in names

    def test_node_existence_check_with_count(self, db):
        """count(n) > 0 as exists pattern."""
        db.execute_cypher("CREATE (:Person {name: 'Alix'})")
        result = list(
            db.execute_cypher("MATCH (n:Person {name: 'Alix'}) RETURN count(n) > 0 AS exists")
        )
        assert len(result) == 1
        assert result[0]["exists"] is True

        result2 = list(
            db.execute_cypher("MATCH (n:Person {name: 'Nobody'}) RETURN count(n) > 0 AS exists")
        )
        # count() on no rows returns 0 > 0 = false, but MATCH with no results returns 0 rows
        # unless we use OPTIONAL MATCH or aggregate globally
        assert len(result2) == 0 or result2[0]["exists"] is False


# =============================================================================
# 2. Edge/Relationship Operations
# =============================================================================


class TestRelationshipOperations:
    """Relationship creation, matching, type filtering, and deletion."""

    def test_create_relationship(self, db):
        """CREATE relationship between two matched nodes."""
        db.execute_cypher("CREATE (:Person {name: 'Alix'})")
        db.execute_cypher("CREATE (:Person {name: 'Gus'})")
        db.execute_cypher(
            "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) "
            "CREATE (a)-[:KNOWS {since: 2020}]->(b)"
        )
        result = list(
            db.execute_cypher(
                "MATCH (a:Person)-[r:KNOWS]->(b:Person) RETURN a.name, b.name, r.since"
            )
        )
        assert len(result) == 1
        assert result[0]["a.name"] == "Alix"
        assert result[0]["b.name"] == "Gus"
        assert result[0]["r.since"] == 2020

    def test_match_relationship_with_type_function(self, db):
        """type(r) returns the relationship type string."""
        db.execute_cypher("CREATE (:A {name: 'src'})-[:CONTAINS]->(:B {name: 'tgt'})")
        result = list(db.execute_cypher("MATCH ()-[r]->() RETURN type(r) AS rel_type"))
        assert len(result) == 1
        assert result[0]["rel_type"] == "CONTAINS"

    def test_delete_relationship_only(self, db):
        """DELETE r removes only the relationship, not the nodes."""
        db.execute_cypher("CREATE (:A {name: 'src'})-[:LINK]->(:B {name: 'tgt'})")
        db.execute_cypher("MATCH ()-[r:LINK]->() DELETE r")
        # Relationship deleted
        result = list(db.execute_cypher("MATCH ()-[r:LINK]->() RETURN r"))
        assert len(result) == 0
        # Nodes still exist
        result = list(db.execute_cypher("MATCH (n) RETURN n.name"))
        assert len(result) == 2

    def test_self_loop_pattern(self, db):
        """Self-referential relationship (a)-[r]->(a)."""
        db.execute_cypher("CREATE (a:Node {name: 'self'})-[:LOOP]->(a)")
        result = list(db.execute_cypher("MATCH (a:Node)-[r:LOOP]->(a) RETURN a.name, type(r) AS t"))
        assert len(result) == 1
        assert result[0]["a.name"] == "self"
        assert result[0]["t"] == "LOOP"

    def test_multiple_match_clauses_for_edge_creation(self, db):
        """Multiple MATCH clauses followed by CREATE (Deriva pattern)."""
        db.execute_cypher("CREATE (:Person {name: 'Alix'})")
        db.execute_cypher("CREATE (:Person {name: 'Gus'})")
        db.execute_cypher("CREATE (:Person {name: 'Vincent'})")
        db.execute_cypher(
            "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) CREATE (a)-[:KNOWS]->(b)"
        )
        db.execute_cypher(
            "MATCH (a:Person {name: 'Gus'}), (b:Person {name: 'Vincent'}) CREATE (a)-[:KNOWS]->(b)"
        )
        result = list(
            db.execute_cypher(
                "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name, b.name "
                "ORDER BY a.name, b.name"
            )
        )
        assert len(result) == 2


# =============================================================================
# 3. Property Operations
# =============================================================================


class TestPropertyOperations:
    """Property updates, IN filtering, parameterized queries."""

    def test_where_in_list(self, db):
        """WHERE n.prop IN [...] filters by list membership."""
        db.execute_cypher("CREATE (:City {name: 'Amsterdam'})")
        db.execute_cypher("CREATE (:City {name: 'Berlin'})")
        db.execute_cypher("CREATE (:City {name: 'Paris'})")
        db.execute_cypher("CREATE (:City {name: 'Prague'})")
        result = list(
            db.execute_cypher(
                "MATCH (c:City) WHERE c.name IN ['Amsterdam', 'Berlin', 'Prague'] "
                "RETURN c.name ORDER BY c.name"
            )
        )
        assert len(result) == 3
        assert [r["c.name"] for r in result] == ["Amsterdam", "Berlin", "Prague"]

    def test_contains_operator(self, db):
        """CONTAINS string operator for substring matching."""
        db.execute_cypher("CREATE (:Doc {text: 'The quick brown fox jumps over the lazy dog'})")
        db.execute_cypher("CREATE (:Doc {text: 'Hello world'})")
        result = list(db.execute_cypher("MATCH (d:Doc) WHERE d.text CONTAINS 'fox' RETURN d.text"))
        assert len(result) == 1
        assert "fox" in result[0]["d.text"]

    def test_starts_with_operator(self, db):
        """STARTS WITH string operator."""
        db.execute_cypher("CREATE (:File {name: 'auth_service.py'})")
        db.execute_cypher("CREATE (:File {name: 'auth_controller.py'})")
        db.execute_cypher("CREATE (:File {name: 'user_service.py'})")
        result = list(
            db.execute_cypher(
                "MATCH (f:File) WHERE f.name STARTS WITH 'auth' RETURN f.name ORDER BY f.name"
            )
        )
        assert len(result) == 2

    def test_not_equals_operator(self, db):
        """<> (not equals) operator."""
        db.execute_cypher("CREATE (:Person {name: 'Alix'})")
        db.execute_cypher("CREATE (:Person {name: 'Gus'})")
        db.execute_cypher("CREATE (:Person {name: 'Vincent'})")
        result = list(
            db.execute_cypher(
                "MATCH (n:Person) WHERE n.name <> 'Gus' RETURN n.name ORDER BY n.name"
            )
        )
        assert len(result) == 2
        assert [r["n.name"] for r in result] == ["Alix", "Vincent"]


# =============================================================================
# 4. Label Functions
# =============================================================================


class TestLabelFunctions:
    """labels(), any() with labels, label filtering."""

    def test_labels_function(self, db):
        """labels(n) returns list of node labels."""
        db.execute_cypher("CREATE (:Person:Employee {name: 'Alix'})")
        result = list(db.execute_cypher("MATCH (n:Person) RETURN labels(n) AS lbls"))
        assert len(result) == 1
        lbls = result[0]["lbls"]
        assert "Person" in lbls
        assert "Employee" in lbls

    def test_any_labels_starts_with(self, db):
        """any(lbl IN labels(n) WHERE lbl STARTS WITH ...) pattern."""
        db.execute_cypher("CREATE (:Model:Component {name: 'Widget'})")
        db.execute_cypher("CREATE (:Graph:Node {name: 'Source'})")
        db.execute_cypher("CREATE (:Other {name: 'Unrelated'})")

        result = list(
            db.execute_cypher(
                "MATCH (n) WHERE any(lbl IN labels(n) WHERE lbl STARTS WITH 'Model') RETURN n.name"
            )
        )
        assert len(result) == 1
        assert result[0]["n.name"] == "Widget"

    def test_any_labels_in_list(self, db):
        """any(lbl IN labels(n) WHERE lbl IN [...]) pattern."""
        db.execute_cypher("CREATE (:ApplicationComponent {name: 'Svc'})")
        db.execute_cypher("CREATE (:DataObject {name: 'DB'})")
        db.execute_cypher("CREATE (:Other {name: 'Skip'})")

        result = list(
            db.execute_cypher(
                "MATCH (n) "
                "WHERE any(lbl IN labels(n) WHERE lbl IN ['ApplicationComponent', 'DataObject']) "
                "RETURN n.name ORDER BY n.name"
            )
        )
        assert len(result) == 2
        assert [r["n.name"] for r in result] == ["DB", "Svc"]

    def test_multiple_label_match(self, db):
        """MATCH with multiple colon-separated labels."""
        db.execute_cypher("CREATE (:Model:ApplicationComponent {name: 'Auth'})")
        db.execute_cypher("CREATE (:Model:DataObject {name: 'DB'})")
        db.execute_cypher("CREATE (:Other {name: 'Skip'})")

        result = list(db.execute_cypher("MATCH (n:Model:ApplicationComponent) RETURN n.name"))
        assert len(result) == 1
        assert result[0]["n.name"] == "Auth"


# =============================================================================
# 5. Type Functions
# =============================================================================


class TestTypeFunctions:
    """type(r), type comparison, type filtering."""

    def test_type_starts_with(self, db):
        """type(r) STARTS WITH filters relationship types."""
        db.execute_cypher("CREATE (:A {name: 'a'})-[:Composition]->(:B {name: 'b'})")
        db.execute_cypher("CREATE (:C {name: 'c'})-[:Flow]->(:D {name: 'd'})")
        db.execute_cypher("CREATE (:E {name: 'e'})-[:OTHER]->(:F {name: 'f'})")

        result = list(
            db.execute_cypher(
                "MATCH ()-[r]->() WHERE type(r) STARTS WITH 'Comp' RETURN type(r) AS t"
            )
        )
        assert len(result) == 1
        assert result[0]["t"] == "Composition"

    def test_type_equals(self, db):
        """type(r) = 'X' exact match."""
        db.execute_cypher("CREATE (:A {name: 'a'})-[:KNOWS]->(:B {name: 'b'})")
        db.execute_cypher("CREATE (:C {name: 'c'})-[:HATES]->(:D {name: 'd'})")
        result = list(
            db.execute_cypher("MATCH ()-[r]->() WHERE type(r) = 'KNOWS' RETURN type(r) AS t")
        )
        assert len(result) == 1
        assert result[0]["t"] == "KNOWS"

    def test_type_in_list(self, db):
        """type(r) IN [...] checks membership."""
        db.execute_cypher("CREATE (:A {name: 'a'})-[:Flow]->(:B {name: 'b'})")
        db.execute_cypher("CREATE (:C {name: 'c'})-[:Serving]->(:D {name: 'd'})")
        db.execute_cypher("CREATE (:E {name: 'e'})-[:Composition]->(:F {name: 'f'})")

        result = list(
            db.execute_cypher(
                "MATCH ()-[r]->() WHERE type(r) IN ['Flow', 'Serving'] "
                "RETURN type(r) AS t ORDER BY t"
            )
        )
        assert len(result) == 2
        assert [r["t"] for r in result] == ["Flow", "Serving"]


# =============================================================================
# 6. NOT EXISTS Subqueries
# =============================================================================


class TestNotExistsSubqueries:
    """NOT EXISTS patterns used extensively by Deriva for dedup prevention."""

    def test_single_not_exists(self, db):
        """NOT EXISTS with inner WHERE type(r) filter."""
        setup_dual_namespace_graph(db)
        result = list(
            db.execute_cypher(
                "MATCH (a:Model), (b:Model) "
                "WHERE a.identifier <> b.identifier "
                "AND a.enabled = true AND b.enabled = true "
                "AND NOT EXISTS { "
                "    MATCH (a)-[r]->(b) WHERE type(r) = 'Composition' "
                "} "
                "RETURN a.name, b.name ORDER BY a.name, b.name"
            )
        )
        pairs = {(r["a.name"], r["b.name"]) for r in result}
        # m1->m2 has Composition, so (AuthService, LoginModule) excluded
        assert ("AuthService", "LoginModule") not in pairs
        # (LoginModule, Database) has Flow, not Composition, so included
        assert ("LoginModule", "Database") in pairs

    def test_two_not_exists_same_where(self, db):
        """Two NOT EXISTS in same WHERE clause (the 0.5.17 bug)."""
        setup_dual_namespace_graph(db)
        result = list(
            db.execute_cypher(
                "MATCH (a:Model), (b:Model) "
                "WHERE a.identifier <> b.identifier "
                "AND a.enabled = true AND b.enabled = true "
                "AND NOT EXISTS { "
                "    MATCH (a)-[r]->(b) WHERE type(r) = 'Composition' "
                "} "
                "AND NOT EXISTS { "
                "    MATCH (a)-[r2]->(b) WHERE type(r2) = 'Flow' "
                "} "
                "RETURN a.name, b.name ORDER BY a.name, b.name"
            )
        )
        pairs = {(r["a.name"], r["b.name"]) for r in result}
        # m1->m2 Composition excluded, m2->m3 Flow excluded
        assert ("AuthService", "LoginModule") not in pairs
        assert ("LoginModule", "Database") not in pairs
        # Other pairs should be present
        assert ("AuthService", "Database") in pairs or ("Database", "AuthService") in pairs

    def test_three_not_exists(self, db):
        """Three NOT EXISTS subqueries in same WHERE."""
        setup_dual_namespace_graph(db)
        result = list(
            db.execute_cypher(
                "MATCH (a:Model), (b:Model) "
                "WHERE a.identifier <> b.identifier "
                "AND a.enabled = true AND b.enabled = true "
                "AND NOT EXISTS { MATCH (a)-[r]->(b) WHERE type(r) = 'Composition' } "
                "AND NOT EXISTS { MATCH (a)-[r2]->(b) WHERE type(r2) = 'Flow' } "
                "AND NOT EXISTS { MATCH (b)-[r3]->(a) WHERE type(r3) = 'Composition' } "
                "RETURN a.name, b.name ORDER BY a.name, b.name"
            )
        )
        # Should not error (was failing with "Unsupported EXISTS subquery pattern")
        pairs = {(r["a.name"], r["b.name"]) for r in result}
        # All forward edges excluded: m1->m2 (Composition), m2->m3 (Flow)
        # Reverse Composition check: m2->m1 would be excluded by 3rd NOT EXISTS
        assert ("AuthService", "LoginModule") not in pairs
        assert ("LoginModule", "Database") not in pairs

    def test_not_exists_bare_pattern(self, db):
        """NOT EXISTS with bare pattern (no MATCH keyword inside EXISTS)."""
        setup_dual_namespace_graph(db)
        result = list(
            db.execute_cypher(
                "MATCH (a:Model), (b:Model) "
                "WHERE a.identifier <> b.identifier "
                "AND a.enabled = true AND b.enabled = true "
                "AND NOT EXISTS { (a)-[r]->(b) WHERE type(r) = 'Composition' } "
                "RETURN a.name, b.name ORDER BY a.name, b.name"
            )
        )
        pairs = {(r["a.name"], r["b.name"]) for r in result}
        assert ("AuthService", "LoginModule") not in pairs

    def test_not_exists_with_with_clause(self, db):
        """NOT EXISTS after WITH clause (Deriva's orphan detection)."""
        setup_dual_namespace_graph(db)
        result = list(
            db.execute_cypher(
                "MATCH (e:Model) "
                "WHERE e.enabled = true "
                "WITH e "
                "WHERE NOT EXISTS { "
                "    MATCH (e)-[r]-() "
                "} "
                "RETURN e.name ORDER BY e.name"
            )
        )
        # OrphanElement (m4) has no relationships
        names = [r["e.name"] for r in result]
        assert "OrphanElement" in names
        # AuthService, LoginModule, Database all have relationships
        assert "AuthService" not in names
        assert "LoginModule" not in names
        assert "Database" not in names


# =============================================================================
# 7. List Comprehensions
# =============================================================================


class TestListComprehensions:
    """List comprehension patterns for label/type extraction."""

    def test_list_comprehension_filter_labels(self, db):
        """[lbl IN labels(e) WHERE lbl <> 'Model'][0] extracts non-Model label."""
        db.execute_cypher("CREATE (:Model:ApplicationComponent {name: 'Auth'})")
        result = list(
            db.execute_cypher(
                "MATCH (e:Model) "
                "RETURN [lbl IN labels(e) WHERE lbl <> 'Model'][0] AS element_type, "
                "e.name"
            )
        )
        assert len(result) == 1
        assert result[0]["element_type"] == "ApplicationComponent"

    def test_list_comprehension_starts_with(self, db):
        """[lbl IN labels(e) WHERE lbl STARTS WITH 'Mod'][0] pattern."""
        db.execute_cypher("CREATE (:Model:Component {name: 'Widget'})")
        result = list(
            db.execute_cypher(
                "MATCH (e) RETURN [lbl IN labels(e) WHERE lbl STARTS WITH 'Mod'][0] AS matched"
            )
        )
        assert len(result) == 1
        assert result[0]["matched"] == "Model"


# =============================================================================
# 8. Aggregations
# =============================================================================


class TestAggregations:
    """count(), collect(), size(), sum() with various patterns."""

    def test_count_basic(self, db):
        """count() returns row count."""
        db.execute_cypher("CREATE (:Person {name: 'Alix'})")
        db.execute_cypher("CREATE (:Person {name: 'Gus'})")
        db.execute_cypher("CREATE (:Person {name: 'Vincent'})")
        result = list(db.execute_cypher("MATCH (n:Person) RETURN count(n) AS cnt"))
        assert result[0]["cnt"] == 3

    def test_count_star(self, db):
        """count(*) counts all matched rows."""
        db.execute_cypher("CREATE (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
        result = list(
            db.execute_cypher("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN count(*) AS cnt")
        )
        assert result[0]["cnt"] == 1

    def test_collect_aggregation(self, db):
        """collect() gathers values into a list."""
        db.execute_cypher("CREATE (:Team {name: 'Alpha'})")
        db.execute_cypher("CREATE (:Member {name: 'Alix', team: 'Alpha'})")
        db.execute_cypher("CREATE (:Member {name: 'Gus', team: 'Alpha'})")
        db.execute_cypher("CREATE (:Member {name: 'Vincent', team: 'Alpha'})")
        result = list(
            db.execute_cypher(
                "MATCH (m:Member) WHERE m.team = 'Alpha' RETURN collect(m.name) AS members"
            )
        )
        assert len(result) == 1
        members = result[0]["members"]
        assert set(members) == {"Alix", "Gus", "Vincent"}

    def test_collect_with_map(self, db):
        """collect({key: val}) creates a list of maps (Deriva duplicate detection)."""
        db.execute_cypher("CREATE (:A {name: 'src'})-[:Flow {id: 'r1'}]->(:B {name: 'tgt'})")
        db.execute_cypher(
            "MATCH (a:A {name: 'src'}), (b:B {name: 'tgt'}) "
            "CREATE (a)-[:Composition {id: 'r2'}]->(b)"
        )
        result = list(
            db.execute_cypher(
                "MATCH (a)-[r]->(b) "
                "WITH a, b, collect({id: r.id, rel_type: type(r)}) AS rels "
                "WHERE size(rels) > 1 "
                "RETURN a.name AS source, b.name AS target, rels"
            )
        )
        assert len(result) == 1
        assert result[0]["source"] == "src"
        assert len(result[0]["rels"]) == 2

    def test_size_function(self, db):
        """size() returns list length."""
        result = list(db.execute_cypher("RETURN size(['a', 'b', 'c']) AS sz"))
        assert result[0]["sz"] == 3

    def test_sum_case_when_in_aggregate(self, db):
        """sum(CASE WHEN ... THEN 1 ELSE 0 END) conditional counting."""
        db.execute_cypher("CREATE (:File {name: 'a.py', file_type: 'source'})")
        db.execute_cypher("CREATE (:File {name: 'b.md', file_type: 'docs'})")
        db.execute_cypher("CREATE (:File {name: 'c.py', file_type: 'source'})")
        result = list(
            db.execute_cypher(
                "MATCH (f:File) "
                "RETURN count(f) AS total, "
                "sum(CASE WHEN f.file_type = 'source' THEN 1 ELSE 0 END) AS source_count"
            )
        )
        assert result[0]["total"] == 3
        assert result[0]["source_count"] == 2

    def test_group_by_with_count(self, db):
        """Implicit GROUP BY with aggregation."""
        db.execute_cypher("CREATE (:Person {name: 'Alix', city: 'Amsterdam'})")
        db.execute_cypher("CREATE (:Person {name: 'Gus', city: 'Amsterdam'})")
        db.execute_cypher("CREATE (:Person {name: 'Vincent', city: 'Berlin'})")
        result = list(
            db.execute_cypher(
                "MATCH (n:Person) RETURN n.city AS city, count(n) AS cnt ORDER BY city"
            )
        )
        assert len(result) == 2
        assert result[0]["city"] == "Amsterdam"
        assert result[0]["cnt"] == 2
        assert result[1]["city"] == "Berlin"
        assert result[1]["cnt"] == 1


# =============================================================================
# 9. CASE WHEN and reduce()
# =============================================================================


class TestCaseWhenAndReduce:
    """CASE WHEN expressions and reduce() accumulator."""

    def test_case_when_basic(self, db):
        """CASE WHEN ... THEN ... ELSE ... END in RETURN."""
        db.execute_cypher("CREATE (:Person {name: 'Alix', age: 30})")
        db.execute_cypher("CREATE (:Person {name: 'Gus', age: 15})")
        result = list(
            db.execute_cypher(
                "MATCH (n:Person) "
                "RETURN n.name, "
                "CASE WHEN n.age >= 18 THEN 'adult' ELSE 'minor' END AS status "
                "ORDER BY n.name"
            )
        )
        assert result[0]["status"] == "adult"
        assert result[1]["status"] == "minor"

    def test_reduce_sum(self, db):
        """reduce() for simple summation."""
        result = list(
            db.execute_cypher(
                "WITH [1, 2, 3, 4, 5] AS vals RETURN reduce(acc = 0, x IN vals | acc + x) AS total"
            )
        )
        assert result[0]["total"] == 15

    def test_reduce_with_case_max(self, db):
        """reduce() with CASE WHEN to compute max."""
        result = list(
            db.execute_cypher(
                "WITH [3, 1, 4, 1, 5] AS vals "
                "RETURN reduce(acc = 0, x IN vals | "
                "CASE WHEN x > acc THEN x ELSE acc END) AS max_val"
            )
        )
        assert result[0]["max_val"] == 5

    def test_reduce_with_case_min_index(self, db):
        """reduce() to find index of minimum value (Deriva cycle detection)."""
        result = list(
            db.execute_cypher(
                "WITH [0.9, 0.3, 0.7, 0.5] AS confidences "
                "RETURN reduce(minIdx = 0, i IN range(0, size(confidences)-1) | "
                "  CASE WHEN confidences[i] < confidences[minIdx] THEN i ELSE minIdx END"
                ") AS weakest_idx"
            )
        )
        assert result[0]["weakest_idx"] == 1


# =============================================================================
# 10. WITH Clause
# =============================================================================


class TestWithClause:
    """WITH for variable binding, filtering, and pipeline."""

    def test_with_variable_passthrough(self, db):
        """WITH passes variables to next clause."""
        db.execute_cypher("CREATE (:Person {name: 'Alix', age: 30})")
        db.execute_cypher("CREATE (:Person {name: 'Gus', age: 25})")
        result = list(
            db.execute_cypher(
                "MATCH (n:Person) WITH n.name AS name, n.age AS age WHERE age > 27 RETURN name"
            )
        )
        assert len(result) == 1
        assert result[0]["name"] == "Alix"

    def test_with_aggregation_then_filter(self, db):
        """WITH + aggregation + WHERE (Deriva duplicate detection)."""
        db.execute_cypher("CREATE (:A {name: 'src'})-[:X]->(:B {name: 'tgt'})")
        db.execute_cypher("MATCH (a:A {name: 'src'}), (b:B {name: 'tgt'}) CREATE (a)-[:Y]->(b)")
        result = list(
            db.execute_cypher(
                "MATCH (a)-[r]->(b) "
                "WITH a, b, count(r) AS rel_count "
                "WHERE rel_count > 1 "
                "RETURN a.name AS src, b.name AS tgt, rel_count"
            )
        )
        assert len(result) == 1
        assert result[0]["rel_count"] == 2


# =============================================================================
# 11. UNWIND
# =============================================================================


class TestUnwind:
    """UNWIND for list unpacking."""

    def test_unwind_list(self, db):
        """UNWIND produces one row per list element."""
        result = list(db.execute_cypher("UNWIND [1, 2, 3] AS x RETURN x ORDER BY x"))
        assert len(result) == 3
        assert [r["x"] for r in result] == [1, 2, 3]

    def test_unwind_with_match(self, db):
        """UNWIND + MATCH for batch lookup."""
        db.execute_cypher("CREATE (:Person {name: 'Alix'})")
        db.execute_cypher("CREATE (:Person {name: 'Gus'})")
        db.execute_cypher("CREATE (:Person {name: 'Vincent'})")
        result = list(
            db.execute_cypher(
                "UNWIND ['Alix', 'Vincent'] AS target_name "
                "MATCH (n:Person {name: target_name}) "
                "RETURN n.name ORDER BY n.name"
            )
        )
        assert len(result) == 2
        assert [r["n.name"] for r in result] == ["Alix", "Vincent"]


# =============================================================================
# 12. OPTIONAL MATCH
# =============================================================================


class TestOptionalMatch:
    """OPTIONAL MATCH for left-join semantics."""

    def test_optional_match_with_results(self, db):
        """OPTIONAL MATCH returns data when pattern matches."""
        db.execute_cypher("CREATE (:A {name: 'src'})-[:LINK]->(:B {name: 'tgt'})")
        result = list(
            db.execute_cypher(
                "MATCH (a:A) OPTIONAL MATCH (a)-[r:LINK]->(b) "
                "RETURN a.name, b.name, r IS NOT NULL AS has_link"
            )
        )
        assert len(result) == 1
        assert result[0]["b.name"] == "tgt"
        assert result[0]["has_link"] is True

    def test_optional_match_null_when_no_match(self, db):
        """OPTIONAL MATCH returns null when no match."""
        db.execute_cypher("CREATE (:A {name: 'lonely'})")
        result = list(
            db.execute_cypher(
                "MATCH (a:A) OPTIONAL MATCH (a)-[r:LINK]->(b) RETURN a.name, b IS NULL AS no_link"
            )
        )
        assert len(result) == 1
        assert result[0]["no_link"] is True


# =============================================================================
# 13. DISTINCT, LIMIT, ORDER BY
# =============================================================================


class TestResultModifiers:
    """DISTINCT, LIMIT, ORDER BY clauses."""

    def test_distinct(self, db):
        """DISTINCT removes duplicate rows."""
        db.execute_cypher("CREATE (:Person {name: 'Alix', city: 'Amsterdam'})")
        db.execute_cypher("CREATE (:Person {name: 'Gus', city: 'Amsterdam'})")
        db.execute_cypher("CREATE (:Person {name: 'Vincent', city: 'Berlin'})")
        result = list(db.execute_cypher("MATCH (n:Person) RETURN DISTINCT n.city ORDER BY n.city"))
        assert len(result) == 2
        assert [r["n.city"] for r in result] == ["Amsterdam", "Berlin"]

    def test_limit(self, db):
        """LIMIT restricts result count."""
        for i in range(10):
            db.execute_cypher(f"CREATE (:Item {{idx: {i}}})")
        result = list(db.execute_cypher("MATCH (n:Item) RETURN n.idx ORDER BY n.idx LIMIT 3"))
        assert len(result) == 3
        assert [r["n.idx"] for r in result] == [0, 1, 2]

    def test_order_by_asc_desc(self, db):
        """ORDER BY with ASC and DESC."""
        db.execute_cypher("CREATE (:Person {name: 'Alix', age: 30})")
        db.execute_cypher("CREATE (:Person {name: 'Gus', age: 25})")
        db.execute_cypher("CREATE (:Person {name: 'Vincent', age: 35})")
        result_asc = list(db.execute_cypher("MATCH (n:Person) RETURN n.name ORDER BY n.age"))
        assert [r["n.name"] for r in result_asc] == ["Gus", "Alix", "Vincent"]

        result_desc = list(db.execute_cypher("MATCH (n:Person) RETURN n.name ORDER BY n.age DESC"))
        assert [r["n.name"] for r in result_desc] == ["Vincent", "Alix", "Gus"]


# =============================================================================
# 14. Coalesce and IS NOT NULL / IS NULL
# =============================================================================


class TestNullHandling:
    """coalesce(), IS NULL, IS NOT NULL patterns."""

    def test_coalesce(self, db):
        """coalesce() returns first non-null value."""
        db.execute_cypher("CREATE (:Item {name: 'with_val', score: 42})")
        db.execute_cypher("CREATE (:Item {name: 'no_val'})")
        result = list(
            db.execute_cypher(
                "MATCH (n:Item) RETURN n.name, coalesce(n.score, 0) AS score ORDER BY n.name"
            )
        )
        assert len(result) == 2
        assert result[0]["score"] == 0  # no_val, null => 0
        assert result[1]["score"] == 42  # with_val => 42

    def test_is_not_null(self, db):
        """IS NOT NULL filter."""
        db.execute_cypher("CREATE (:Item {name: 'A', score: 42})")
        db.execute_cypher("CREATE (:Item {name: 'B'})")
        result = list(db.execute_cypher("MATCH (n:Item) WHERE n.score IS NOT NULL RETURN n.name"))
        assert len(result) == 1
        assert result[0]["n.name"] == "A"

    def test_is_null(self, db):
        """IS NULL filter."""
        db.execute_cypher("CREATE (:Item {name: 'A', score: 42})")
        db.execute_cypher("CREATE (:Item {name: 'B'})")
        result = list(db.execute_cypher("MATCH (n:Item) WHERE n.score IS NULL RETURN n.name"))
        assert len(result) == 1
        assert result[0]["n.name"] == "B"


# =============================================================================
# 15. elementId() Function
# =============================================================================


class TestElementIdFunction:
    """elementId() for relationship comparison (Deriva dedup)."""

    def test_element_id_comparison(self, db):
        """elementId(r1) < elementId(r2) for dedup ordering."""
        db.execute_cypher("CREATE (:A {name: 'src'})-[:LINK]->(:B {name: 'tgt'})")
        db.execute_cypher("MATCH (a:A {name: 'src'}), (b:B {name: 'tgt'}) CREATE (a)-[:LINK]->(b)")
        # Two LINK relationships between same nodes
        result = list(
            db.execute_cypher(
                "MATCH (a)-[r1:LINK]->(b), (a)-[r2:LINK]->(b) "
                "WHERE elementId(r1) < elementId(r2) "
                "RETURN a.name AS src, b.name AS tgt"
            )
        )
        # Should find exactly 1 pair (deduped by elementId ordering)
        assert len(result) == 1
        assert result[0]["src"] == "src"


# =============================================================================
# 16. Complex Derive Integration Patterns
# =============================================================================


class TestDerivaIntegrationPatterns:
    """End-to-end patterns matching actual Deriva query flows."""

    def test_relationship_derivation_with_not_exists(self, db):
        """Deriva's primary relationship derivation query.

        Find graph edges, match to model elements by source_identifier,
        exclude pairs that already have the target relationship type.
        """
        setup_dual_namespace_graph(db)
        result = list(
            db.execute_cypher(
                "MATCH (graph_src:GraphNode)-[:CONTAINS]->(graph_tgt:GraphNode) "
                "WHERE graph_src.active = true AND graph_tgt.active = true "
                "MATCH (model_src:Model), (model_tgt:Model) "
                "WHERE model_src.enabled = true AND model_tgt.enabled = true "
                "AND model_src.source_identifier = graph_src.id "
                "AND model_tgt.source_identifier = graph_tgt.id "
                "AND model_src.identifier <> model_tgt.identifier "
                "AND NOT EXISTS { "
                "    (model_src)-[existing]->(model_tgt) "
                "    WHERE type(existing) = 'Composition' "
                "} "
                "RETURN DISTINCT "
                "model_src.identifier AS source_id, model_src.name AS source_name, "
                "model_tgt.identifier AS target_id, model_tgt.name AS target_name"
            )
        )
        # g1 CONTAINS g2 maps to m1->m2, but m1->m2 already has Composition
        # so no candidates should be returned
        assert len(result) == 0

    def test_relationship_derivation_finds_candidates(self, db):
        """Derivation query should find candidates when no existing relationship."""
        setup_dual_namespace_graph(db)
        result = list(
            db.execute_cypher(
                "MATCH (graph_src:GraphNode)-[:CALLS]->(graph_tgt:GraphNode) "
                "WHERE graph_src.active = true AND graph_tgt.active = true "
                "MATCH (model_src:Model), (model_tgt:Model) "
                "WHERE model_src.enabled = true AND model_tgt.enabled = true "
                "AND model_src.source_identifier = graph_src.id "
                "AND model_tgt.source_identifier = graph_tgt.id "
                "AND model_src.identifier <> model_tgt.identifier "
                "AND NOT EXISTS { "
                "    (model_src)-[existing]->(model_tgt) "
                "    WHERE type(existing) = 'Serving' "
                "} "
                "RETURN DISTINCT "
                "model_src.name AS source_name, model_tgt.name AS target_name "
                "LIMIT 10"
            )
        )
        # g2 CALLS g3 maps to m2->m3; m2->m3 has Flow (not Serving), so it's a candidate
        assert len(result) == 1
        assert result[0]["source_name"] == "LoginModule"
        assert result[0]["target_name"] == "Database"

    def test_duplicate_relationship_detection(self, db):
        """Detect duplicate relationships between same source and target."""
        db.execute_cypher("CREATE (:Node {name: 'A'})")
        db.execute_cypher("CREATE (:Node {name: 'B'})")
        db.execute_cypher(
            "MATCH (a:Node {name: 'A'}), (b:Node {name: 'B'}) CREATE (a)-[:Flow {id: 'r1'}]->(b)"
        )
        db.execute_cypher(
            "MATCH (a:Node {name: 'A'}), (b:Node {name: 'B'}) CREATE (a)-[:Flow {id: 'r2'}]->(b)"
        )
        result = list(
            db.execute_cypher(
                "MATCH (a:Node)-[r1]->(b:Node), (a)-[r2]->(b) "
                "WHERE type(r1) = type(r2) "
                "AND elementId(r1) < elementId(r2) "
                "RETURN a.name AS source, b.name AS target, "
                "r1.id AS r1_id, r2.id AS r2_id, type(r1) AS rel_type"
            )
        )
        assert len(result) == 1
        assert result[0]["rel_type"] == "Flow"
        assert {result[0]["r1_id"], result[0]["r2_id"]} == {"r1", "r2"}

    def test_orphan_detection(self, db):
        """Find elements with no relationships (Deriva orphan detection)."""
        setup_dual_namespace_graph(db)
        result = list(
            db.execute_cypher(
                "MATCH (e:Model) "
                "WHERE e.enabled = true "
                "WITH e "
                "WHERE NOT EXISTS { MATCH (e)-[r]-() } "
                "RETURN e.identifier AS id, e.name ORDER BY e.name"
            )
        )
        names = [r["e.name"] for r in result]
        assert "OrphanElement" in names

    def test_bidirectional_relationship_check(self, db):
        """Check for bidirectional relationships (cycle detection)."""
        db.execute_cypher("CREATE (:Node {name: 'A', enabled: true})")
        db.execute_cypher("CREATE (:Node {name: 'B', enabled: true})")
        db.execute_cypher(
            "MATCH (a:Node {name: 'A'}), (b:Node {name: 'B'}) "
            "CREATE (a)-[:Composition {id: 'r1', confidence: 0.9}]->(b)"
        )
        db.execute_cypher(
            "MATCH (a:Node {name: 'B'}), (b:Node {name: 'A'}) "
            "CREATE (a)-[:Composition {id: 'r2', confidence: 0.3}]->(b)"
        )
        result = list(
            db.execute_cypher(
                "MATCH (a)-[r1:Composition]->(b)-[r2:Composition]->(a) "
                "WHERE a.enabled = true AND b.enabled = true "
                "AND a.name < b.name "
                "RETURN a.name AS a_name, b.name AS b_name, "
                "r1.confidence AS r1_conf, r2.confidence AS r2_conf"
            )
        )
        assert len(result) == 1
        assert result[0]["a_name"] == "A"
        assert result[0]["b_name"] == "B"

    def test_namespace_clearing_with_any_labels(self, db):
        """DETACH DELETE with any(label IN labels(n) WHERE ...) (Deriva clear)."""
        db.execute_cypher("CREATE (:Model:Component {name: 'keep_me'})")
        db.execute_cypher("CREATE (:Graph:Node {name: 'delete_me'})")
        db.execute_cypher("CREATE (:Other {name: 'also_keep'})")

        db.execute_cypher(
            "MATCH (n) "
            "WHERE any(label IN labels(n) WHERE label STARTS WITH 'Graph') "
            "DETACH DELETE n"
        )

        result = list(db.execute_cypher("MATCH (n) RETURN n.name ORDER BY n.name"))
        names = [r["n.name"] for r in result]
        assert "keep_me" in names
        assert "also_keep" in names
        assert "delete_me" not in names


# =============================================================================
# EXISTS Subquery Tests (Tests 5-11 from exists-subquery-bug.md)
# =============================================================================


def setup_exists_subquery_graph(db):
    """Create a graph for EXISTS subquery tests.

    Graph layer: Repository > Directory > Files, with IMPORTS edges.
    Model layer: ArchiMate elements with source_identifier links back to Graph.
    One existing Model:Composition relationship for dedup testing.

    Nodes use dual labels: a base label (e.g. Model) for simple MATCH patterns,
    and a backtick-escaped namespaced label (e.g. `Model:ApplicationComponent`)
    for STARTS WITH filtering on labels(). This mirrors Deriva's convention.
    """
    # Graph layer nodes (base label + namespaced label)
    db.execute_cypher(
        "CREATE (:Graph:`Graph:Repository` {id: 'repo-1', name: 'TestRepo', active: true})"
    )
    db.execute_cypher("CREATE (:Graph:`Graph:Directory` {id: 'dir-1', name: 'src', active: true})")
    db.execute_cypher("CREATE (:Graph:`Graph:File` {id: 'file-1', name: 'app.py', active: true})")
    db.execute_cypher(
        "CREATE (:Graph:`Graph:File` {id: 'file-2', name: 'models.py', active: true})"
    )
    db.execute_cypher("CREATE (:Graph:`Graph:File` {id: 'file-3', name: 'utils.py', active: true})")

    # Graph layer edges
    db.execute_cypher(
        "MATCH (a:`Graph:Repository` {id: 'repo-1'}), (b:`Graph:Directory` {id: 'dir-1'}) "
        "CREATE (a)-[:`Graph:CONTAINS`]->(b)"
    )
    db.execute_cypher(
        "MATCH (a:`Graph:Directory` {id: 'dir-1'}), (b:`Graph:File` {id: 'file-1'}) "
        "CREATE (a)-[:`Graph:CONTAINS`]->(b)"
    )
    db.execute_cypher(
        "MATCH (a:`Graph:Directory` {id: 'dir-1'}), (b:`Graph:File` {id: 'file-2'}) "
        "CREATE (a)-[:`Graph:CONTAINS`]->(b)"
    )
    db.execute_cypher(
        "MATCH (a:`Graph:File` {id: 'file-1'}), (b:`Graph:File` {id: 'file-2'}) "
        "CREATE (a)-[:`Graph:IMPORTS`]->(b)"
    )

    # Model layer nodes (base label Model + namespaced label)
    db.execute_cypher(
        "CREATE (:Model:`Model:ApplicationComponent` {"
        "identifier: 'ac-1', name: 'App', enabled: true, "
        "source_identifier: 'file-1', "
        'properties_json: \'{"source": "file-1"}\''
        "})"
    )
    db.execute_cypher(
        "CREATE (:Model:`Model:ApplicationComponent` {"
        "identifier: 'ac-2', name: 'Models', enabled: true, "
        "source_identifier: 'file-2', "
        'properties_json: \'{"source": "file-2"}\''
        "})"
    )
    db.execute_cypher(
        "CREATE (:Model:`Model:ApplicationComponent` {"
        "identifier: 'ac-3', name: 'Utils', enabled: true, "
        "source_identifier: 'file-3', "
        'properties_json: \'{"source": "file-3"}\''
        "})"
    )
    db.execute_cypher(
        "CREATE (:Model:`Model:DataObject` {"
        "identifier: 'do-1', name: 'UserData', enabled: true, "
        "source_identifier: 'file-2', "
        'properties_json: \'{"source": "file-2"}\''
        "})"
    )

    # One existing Model relationship (for dedup testing)
    db.execute_cypher(
        "MATCH (a:`Model:ApplicationComponent` {identifier: 'ac-1'}), "
        "(b:`Model:ApplicationComponent` {identifier: 'ac-2'}) "
        "CREATE (a)-[:`Model:Composition`]->(b)"
    )


class TestExistsSubqueryPatterns:
    """Tests for EXISTS/NOT EXISTS subquery patterns (Tests 5-11).

    These test patterns from Deriva's relationship derivation, orphan
    detection, and cross-layer connection checking.
    """

    def test_relationship_dedup_with_circular_prevention(self, db):
        """Test 5: Two NOT EXISTS in same WHERE clause.

        First NOT EXISTS prevents duplicate relationships.
        Second NOT EXISTS prevents circular Composition.
        Should NOT return (App, Models) because ac1 -[:Model:Composition]-> ac2 exists.
        """
        setup_exists_subquery_graph(db)

        result = list(
            db.execute_cypher(
                "MATCH (graph_src)-[edge:`Graph:CONTAINS`]->(graph_tgt) "
                "WHERE graph_src.active = true AND graph_tgt.active = true "
                "MATCH (model_src), (model_tgt) "
                "WHERE any(lbl IN labels(model_src) WHERE lbl STARTS WITH 'Model:') "
                "  AND any(lbl IN labels(model_tgt) WHERE lbl STARTS WITH 'Model:') "
                "  AND model_src.enabled = true AND model_tgt.enabled = true "
                "  AND model_src.source_identifier = graph_src.id "
                "  AND model_tgt.source_identifier = graph_tgt.id "
                "  AND model_src.identifier <> model_tgt.identifier "
                "  AND NOT EXISTS { "
                "      (model_src)-[existing]->(model_tgt) "
                "      WHERE type(existing) = 'Model:Composition' "
                "  } "
                "  AND NOT EXISTS { "
                "      (model_tgt)-[reverse:`Model:Composition`]->(model_src) "
                "  } "
                "RETURN DISTINCT "
                "    model_src.identifier AS source_id, "
                "    model_src.name AS source_name, "
                "    model_tgt.identifier AS target_id, "
                "    model_tgt.name AS target_name"
            )
        )

        # ac1 -> ac2 already has Model:Composition, so should be filtered out.
        # No other CONTAINS edges connect to model layer nodes, so expect 0 rows.
        source_pairs = {(r["source_id"], r["target_id"]) for r in result}
        assert ("ac-1", "ac-2") not in source_pairs, (
            "Should filter out (App, Models): Model:Composition already exists"
        )

    def test_single_not_exists_with_complex_where(self, db):
        """Test 6: Single NOT EXISTS with complex inner WHERE.

        Dedup only (no circular check). Uses Graph:IMPORTS edge type.
        Should return (App, Models) because no Model:ServingRelationship exists.
        """
        setup_exists_subquery_graph(db)

        result = list(
            db.execute_cypher(
                "MATCH (graph_src)-[edge:`Graph:IMPORTS`]->(graph_tgt) "
                "WHERE graph_src.active = true AND graph_tgt.active = true "
                "MATCH (model_src), (model_tgt) "
                "WHERE any(lbl IN labels(model_src) WHERE lbl STARTS WITH 'Model:') "
                "  AND any(lbl IN labels(model_tgt) WHERE lbl STARTS WITH 'Model:') "
                "  AND model_src.enabled = true AND model_tgt.enabled = true "
                "  AND model_src.source_identifier = graph_src.id "
                "  AND model_tgt.source_identifier = graph_tgt.id "
                "  AND model_src.identifier <> model_tgt.identifier "
                "  AND NOT EXISTS { "
                "      (model_src)-[existing]->(model_tgt) "
                "      WHERE type(existing) = 'Model:ServingRelationship' "
                "  } "
                "RETURN DISTINCT "
                "    model_src.identifier AS source_id, "
                "    model_src.name AS source_name, "
                "    model_tgt.identifier AS target_id, "
                "    model_tgt.name AS target_name"
            )
        )

        # Graph:IMPORTS from file-1 to file-2.
        # ac-1 (source_id=file-1) and ac-2 (source_id=file-2) should match.
        # No Model:ServingRelationship exists, so NOT EXISTS passes.
        source_pairs = {(r["source_id"], r["target_id"]) for r in result}
        assert ("ac-1", "ac-2") in source_pairs, (
            "Should return (App, Models): IMPORTS edge exists, "
            "no ServingRelationship to filter it out"
        )

    def test_fallback_query_with_two_not_exists(self, db):
        """Test 7: Two NOT EXISTS using CONTAINS on properties_json.

        Fallback candidate query using string CONTAINS instead of
        source_identifier equality. Should NOT return (App, Models)
        because an existing Model: relationship exists.
        """
        setup_exists_subquery_graph(db)

        result = list(
            db.execute_cypher(
                "MATCH (graph_src)-[edge:`Graph:CONTAINS`]->(graph_tgt) "
                "WHERE graph_src.active = true AND graph_tgt.active = true "
                "WITH graph_src.id as src_id, graph_tgt.id as tgt_id "
                "MATCH (model_src), (model_tgt) "
                "WHERE any(lbl IN labels(model_src) WHERE lbl STARTS WITH 'Model:') "
                "  AND any(lbl IN labels(model_tgt) WHERE lbl STARTS WITH 'Model:') "
                "  AND model_src.enabled = true AND model_tgt.enabled = true "
                "  AND model_src.properties_json CONTAINS src_id "
                "  AND model_tgt.properties_json CONTAINS tgt_id "
                "  AND model_src.identifier <> model_tgt.identifier "
                "  AND NOT EXISTS { "
                "      (model_src)-[existing]->(model_tgt) "
                "      WHERE type(existing) STARTS WITH 'Model:' "
                "  } "
                "  AND NOT EXISTS { "
                "      (model_tgt)-[reverse:`Model:Composition`]->(model_src) "
                "  } "
                "RETURN DISTINCT "
                "    model_src.identifier AS source_id, "
                "    model_src.name AS source_name, "
                "    model_tgt.identifier AS target_id, "
                "    model_tgt.name AS target_name"
            )
        )

        # ac1 -> ac2 has a Model:Composition, so first NOT EXISTS filters it.
        source_pairs = {(r["source_id"], r["target_id"]) for r in result}
        assert ("ac-1", "ac-2") not in source_pairs, (
            "Should filter out (App, Models): Model:Composition already exists"
        )

    def test_orphan_element_detection(self, db):
        """Test 8: Single NOT EXISTS with undirected relationship pattern.

        Finds model elements with no Model: relationships (orphans).
        Should return ac3 (Utils) and do1 (UserData).
        """
        setup_exists_subquery_graph(db)

        result = list(
            db.execute_cypher(
                "MATCH (e) "
                "WHERE any(lbl IN labels(e) WHERE lbl STARTS WITH 'Model:') "
                "  AND e.enabled = true "
                "WITH e "
                "WHERE NOT EXISTS { "
                "    MATCH (e)-[r]-() "
                "    WHERE type(r) STARTS WITH 'Model:' "
                "} "
                "RETURN e.identifier as identifier, "
                "       e.name as name, "
                "       [lbl IN labels(e) WHERE lbl STARTS WITH 'Model:'][0] as label "
                "ORDER BY e.name"
            )
        )

        identifiers = {r["identifier"] for r in result}

        # ac3 (Utils) and do1 (UserData) have no Model: edges
        assert "ac-3" in identifiers, "Utils should be an orphan"
        assert "do-1" in identifiers, "UserData should be an orphan"
        # ac1 and ac2 have Model:Composition between them
        assert "ac-1" not in identifiers, "App has a Model:Composition edge"
        assert "ac-2" not in identifiers, "Models has a Model:Composition edge"

    def test_cross_layer_connection_check(self, db):
        """Test 9: Single NOT EXISTS with undirected relationship and label filter.

        Finds ApplicationComponent/ApplicationService elements with no
        connection to DataObject/TechnologyNode via Model: edges.
        """
        setup_exists_subquery_graph(db)

        result = list(
            db.execute_cypher(
                "MATCH (source) "
                "WHERE any(lbl IN labels(source) WHERE lbl IN "
                "    ['Model:ApplicationComponent', 'Model:ApplicationService']) "
                "  AND source.enabled = true "
                "WITH source "
                "WHERE NOT EXISTS { "
                "    MATCH (source)-[r]-(target) "
                "    WHERE any(lbl IN labels(target) WHERE lbl IN "
                "        ['Model:DataObject', 'Model:TechnologyNode']) "
                "      AND type(r) STARTS WITH 'Model:' "
                "} "
                "RETURN source.identifier as identifier, "
                "       source.name as name, "
                "       [lbl IN labels(source) WHERE lbl STARTS WITH 'Model:'][0] as label "
                "ORDER BY source.name"
            )
        )

        identifiers = {r["identifier"] for r in result}

        # All three ApplicationComponents have no Model: connection to DataObject
        assert "ac-1" in identifiers, "App has no Model: edge to DataObject/TechnologyNode"
        assert "ac-2" in identifiers, "Models has no Model: edge to DataObject/TechnologyNode"
        assert "ac-3" in identifiers, "Utils has no Model: edge to DataObject/TechnologyNode"

    def test_floating_element_check(self, db):
        """Test 10: Single NOT EXISTS, finds DataObjects not connected to Application layer."""
        setup_exists_subquery_graph(db)

        result = list(
            db.execute_cypher(
                "MATCH (elem) "
                "WHERE any(lbl IN labels(elem) WHERE lbl IN ['Model:DataObject']) "
                "  AND elem.enabled = true "
                "WITH elem "
                "WHERE NOT EXISTS { "
                "    MATCH (elem)-[r]-(connected) "
                "    WHERE any(lbl IN labels(connected) WHERE lbl IN "
                "        ['Model:ApplicationComponent', 'Model:ApplicationService']) "
                "      AND type(r) STARTS WITH 'Model:' "
                "} "
                "RETURN elem.identifier as identifier, "
                "       elem.name as name, "
                "       [lbl IN labels(elem) WHERE lbl STARTS WITH 'Model:'][0] as label"
            )
        )

        identifiers = {r["identifier"] for r in result}

        # do1 (UserData) has no Model: edge to any ApplicationComponent
        assert "do-1" in identifiers, "UserData should be floating"

    def test_not_exists_with_backtick_escaped_type(self, db):
        """Test 11: Two NOT EXISTS with backtick-escaped relationship type.

        One NOT EXISTS uses type() = 'Model:Composition', the other uses
        backtick syntax `Model:Composition` in the relationship pattern.
        Both forms must work identically.
        """
        setup_exists_subquery_graph(db)

        result = list(
            db.execute_cypher(
                "MATCH (a:Model), (b:Model) "
                "WHERE a.identifier <> b.identifier "
                "  AND a.enabled = true AND b.enabled = true "
                "  AND NOT EXISTS { "
                "      (a)-[existing]->(b) "
                "      WHERE type(existing) = 'Model:Composition' "
                "  } "
                "  AND NOT EXISTS { "
                "      (b)-[reverse:`Model:Composition`]->(a) "
                "  } "
                "RETURN a.identifier AS a_id, b.identifier AS b_id "
                "ORDER BY a.identifier, b.identifier"
            )
        )

        pairs = {(r["a_id"], r["b_id"]) for r in result}

        # ac1 -> ac2 has Model:Composition, so (ac-1, ac-2) should be filtered
        assert ("ac-1", "ac-2") not in pairs, (
            "Should filter: Model:Composition exists from ac-1 to ac-2"
        )
        # Reverse: ac2 -> ac1 has no Model:Composition, but (ac-2, ac-1) should
        # be filtered by the second NOT EXISTS (reverse check)
        assert ("ac-2", "ac-1") not in pairs, (
            "Should filter: reverse Model:Composition check catches ac-2 to ac-1"
        )
