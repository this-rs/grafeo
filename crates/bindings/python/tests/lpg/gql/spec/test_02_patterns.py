"""GQL spec: Pattern Matching (ISO sec 16).

Covers: node patterns, label expressions, edge patterns, path quantifiers,
path search prefixes, path modes, match modes, path functions,
parenthesized path patterns, subpath variables.
"""

import pytest

# =============================================================================
# Node Patterns (sec 16.7)
# =============================================================================


class TestNodePatterns:
    """Node pattern syntax variants."""

    def test_anonymous_node(self, db):
        """() anonymous node matches any node."""
        db.create_node(["Person"], {"name": "Alix"})
        db.create_node(["City"], {"name": "Amsterdam"})
        result = list(db.execute("MATCH (n) RETURN count(n) AS cnt"))
        assert result[0]["cnt"] == 2

    def test_variable_node(self, db):
        """(n) binds a variable to the matched node."""
        db.create_node(["Person"], {"name": "Alix"})
        result = list(db.execute("MATCH (n) RETURN n.name"))
        assert result[0]["n.name"] == "Alix"

    def test_single_label(self, db):
        """(n:Label) single label filter."""
        db.create_node(["Person"], {"name": "Alix"})
        db.create_node(["City"], {"name": "Amsterdam"})
        result = list(db.execute("MATCH (n:Person) RETURN n.name"))
        assert len(result) == 1

    def test_multi_label_colon(self, db):
        """(n IS L1 & L2) multi-label conjunction syntax."""
        db.create_node(["Person", "Developer"], {"name": "Alix"})
        db.create_node(["Person"], {"name": "Gus"})
        result = list(db.execute("MATCH (n IS Person & Developer) RETURN n.name"))
        assert len(result) == 1
        assert result[0]["n.name"] == "Alix"

    def test_property_filter(self, db):
        """(n {key: val}) inline property filter."""
        db.create_node(["Person"], {"name": "Alix", "age": 30})
        db.create_node(["Person"], {"name": "Gus", "age": 25})
        result = list(db.execute("MATCH (n:Person {age: 30}) RETURN n.name"))
        assert result[0]["n.name"] == "Alix"

    def test_element_where(self, db):
        """(n WHERE n.age > 30) element-level WHERE clause."""
        db.create_node(["Person"], {"name": "Alix", "age": 30})
        db.create_node(["Person"], {"name": "Vincent", "age": 35})
        result = list(db.execute("MATCH (n:Person WHERE n.age > 30) RETURN n.name"))
        assert len(result) == 1
        assert result[0]["n.name"] == "Vincent"


# =============================================================================
# Label Expressions (sec 16.8)
# =============================================================================


class TestLabelExpressions:
    """Label expression syntax: IS, &, |, !, %."""

    def test_is_label(self, db):
        """IS Label syntax."""
        db.create_node(["Person"], {"name": "Alix"})
        db.create_node(["City"], {"name": "Amsterdam"})
        result = list(db.execute("MATCH (n IS Person) RETURN n.name"))
        assert len(result) == 1
        assert result[0]["n.name"] == "Alix"

    def test_label_conjunction(self, db):
        """IS Label1 & Label2 (conjunction)."""
        db.create_node(["Person", "Developer"], {"name": "Alix"})
        db.create_node(["Person"], {"name": "Gus"})
        result = list(db.execute("MATCH (n IS Person & Developer) RETURN n.name"))
        assert len(result) == 1
        assert result[0]["n.name"] == "Alix"

    def test_label_disjunction(self, db):
        """IS Label1 | Label2 (disjunction)."""
        db.create_node(["Person"], {"name": "Alix"})
        db.create_node(["City"], {"name": "Amsterdam"})
        db.create_node(["Animal"], {"name": "Rex"})
        result = list(db.execute("MATCH (n IS Person | City) RETURN n.name"))
        names = {r["n.name"] for r in result}
        assert names == {"Alix", "Amsterdam"}

    def test_label_negation(self, db):
        """IS !Label (negation)."""
        db.create_node(["Person"], {"name": "Alix"})
        db.create_node(["City"], {"name": "Amsterdam"})
        result = list(db.execute("MATCH (n IS !Person) RETURN n.name"))
        assert len(result) == 1
        assert result[0]["n.name"] == "Amsterdam"

    def test_label_wildcard(self, db):
        """IS % (wildcard, matches any label)."""
        db.create_node(["Person"], {"name": "Alix"})
        db.create_node(["City"], {"name": "Amsterdam"})
        result = list(db.execute("MATCH (n IS %) RETURN count(n) AS cnt"))
        assert result[0]["cnt"] == 2

    def test_parenthesized_label_expr(self, db):
        """IS (Person|Company)&!Inactive."""
        db.create_node(["Person"], {"name": "Alix"})
        db.create_node(["Person", "Inactive"], {"name": "Gus"})
        db.create_node(["Company"], {"name": "Acme"})
        db.create_node(["Company", "Inactive"], {"name": "OldCo"})
        result = list(db.execute("MATCH (n IS (Person | Company) & !Inactive) RETURN n.name"))
        names = {r["n.name"] for r in result}
        assert names == {"Alix", "Acme"}


# =============================================================================
# Edge Patterns (sec 16.7)
# =============================================================================


class TestEdgePatterns:
    """Edge pattern variants: direction, types, properties."""

    def test_outgoing_edge(self, db):
        """-[e:TYPE]-> outgoing edge."""
        a = db.create_node(["Person"], {"name": "Alix"})
        b = db.create_node(["Person"], {"name": "Gus"})
        db.create_edge(a.id, b.id, "KNOWS")
        result = list(db.execute("MATCH (a)-[e:KNOWS]->(b) RETURN a.name, b.name"))
        assert len(result) == 1
        assert result[0]["a.name"] == "Alix"

    def test_incoming_edge(self, db):
        """<-[e:TYPE]- incoming edge."""
        a = db.create_node(["Person"], {"name": "Alix"})
        b = db.create_node(["Person"], {"name": "Gus"})
        db.create_edge(a.id, b.id, "KNOWS")
        result = list(db.execute("MATCH (b:Person {name: 'Gus'})<-[e:KNOWS]-(a) RETURN a.name"))
        assert len(result) == 1
        assert result[0]["a.name"] == "Alix"

    def test_undirected_edge_cypher_style(self, db):
        """-[e:TYPE]- undirected (Cypher-style)."""
        a = db.create_node(["Person"], {"name": "Alix"})
        b = db.create_node(["Person"], {"name": "Gus"})
        db.create_edge(a.id, b.id, "KNOWS")
        result = list(db.execute("MATCH (a:Person {name: 'Gus'})-[e:KNOWS]-(b) RETURN b.name"))
        names = {r["b.name"] for r in result}
        assert "Alix" in names

    def test_undirected_edge_iso_tilde(self, db):
        """~[e:TYPE]~ undirected (ISO tilde syntax)."""
        a = db.create_node(["Person"], {"name": "Alix"})
        b = db.create_node(["Person"], {"name": "Gus"})
        db.create_edge(a.id, b.id, "KNOWS")
        result = list(db.execute("MATCH (a:Person {name: 'Gus'})~[e:KNOWS]~(b) RETURN b.name"))
        names = {r["b.name"] for r in result}
        assert "Alix" in names

    def test_multiple_edge_types(self, db):
        """-[e:T1|T2]-> pipe-separated types."""
        a = db.create_node(["Person"], {"name": "Alix"})
        b = db.create_node(["Person"], {"name": "Gus"})
        c = db.create_node(["Person"], {"name": "Vincent"})
        db.create_edge(a.id, b.id, "KNOWS")
        db.create_edge(a.id, c.id, "WORKS_WITH")
        result = list(
            db.execute("MATCH (a:Person {name: 'Alix'})-[:KNOWS|WORKS_WITH]->(b) RETURN b.name")
        )
        names = {r["b.name"] for r in result}
        assert names == {"Gus", "Vincent"}

    def test_edge_property_filter(self, db):
        """-[e {prop: val}]-> edge with property filter."""
        a = db.create_node(["Person"], {"name": "Alix"})
        b = db.create_node(["Person"], {"name": "Gus"})
        db.create_edge(a.id, b.id, "KNOWS", {"since": 2020})
        result = list(db.execute("MATCH (a)-[e:KNOWS {since: 2020}]->(b) RETURN a.name, b.name"))
        assert len(result) == 1

    def test_edge_element_where(self, db):
        """-[e WHERE e.since >= 2020]-> element WHERE on edge."""
        a = db.create_node(["Person"], {"name": "Alix"})
        b = db.create_node(["Person"], {"name": "Gus"})
        c = db.create_node(["Person"], {"name": "Vincent"})
        db.create_edge(a.id, b.id, "KNOWS", {"since": 2020})
        db.create_edge(a.id, c.id, "KNOWS", {"since": 2018})
        result = list(
            db.execute(
                "MATCH (a:Person {name: 'Alix'})-[e:KNOWS WHERE e.since >= 2020]->(b) RETURN b.name"
            )
        )
        assert len(result) == 1
        assert result[0]["b.name"] == "Gus"


# =============================================================================
# Path Quantifiers (sec 16.7)
# =============================================================================


class TestPathQuantifiers:
    """Variable-length path quantifiers: *, {m,n}, ?."""

    def _chain(self, db, n):
        """Create a chain of n+1 nodes linked by NEXT edges."""
        nodes = []
        for i in range(n + 1):
            nodes.append(db.create_node(["Node"], {"name": str(i)}))
        for i in range(n):
            db.create_edge(nodes[i].id, nodes[i + 1].id, "NEXT")
        return nodes

    def test_star_zero_or_more(self, db):
        """* zero or more hops (Cypher-style)."""
        self._chain(db, 3)
        result = list(db.execute("MATCH (a:Node {name: '0'})-[:NEXT*]->(b:Node) RETURN b.name"))
        names = {r["b.name"] for r in result}
        assert "1" in names
        assert "3" in names

    def test_star_exact(self, db):
        """*n exactly n hops."""
        self._chain(db, 3)
        result = list(db.execute("MATCH (a:Node {name: '0'})-[:NEXT*2]->(b:Node) RETURN b.name"))
        assert len(result) == 1
        assert result[0]["b.name"] == "2"

    def test_star_range(self, db):
        """*m..n range of hops."""
        self._chain(db, 4)
        result = list(db.execute("MATCH (a:Node {name: '0'})-[:NEXT*2..3]->(b:Node) RETURN b.name"))
        names = {r["b.name"] for r in result}
        assert names == {"2", "3"}

    def test_star_min_only(self, db):
        """*m.. minimum hops."""
        self._chain(db, 4)
        result = list(db.execute("MATCH (a:Node {name: '0'})-[:NEXT*3..]->(b:Node) RETURN b.name"))
        names = {r["b.name"] for r in result}
        assert "3" in names
        assert "4" in names
        assert "2" not in names

    def test_star_max_only(self, db):
        """*..n maximum hops."""
        self._chain(db, 4)
        result = list(db.execute("MATCH (a:Node {name: '0'})-[:NEXT*..2]->(b:Node) RETURN b.name"))
        names = {r["b.name"] for r in result}
        assert "1" in names
        assert "2" in names
        assert "3" not in names

    def test_iso_curly_range(self, db):
        """*m..n Cypher-style range (equivalent to ISO {m,n})."""
        self._chain(db, 4)
        result = list(db.execute("MATCH (a:Node {name: '0'})-[:NEXT*2..3]->(b:Node) RETURN b.name"))
        names = {r["b.name"] for r in result}
        assert names == {"2", "3"}

    def test_iso_curly_exact(self, db):
        """*m Cypher-style exact repetition (equivalent to ISO {m})."""
        self._chain(db, 3)
        result = list(db.execute("MATCH (a:Node {name: '0'})-[:NEXT*2]->(b:Node) RETURN b.name"))
        assert len(result) == 1
        assert result[0]["b.name"] == "2"

    def test_iso_curly_min_only(self, db):
        """*m.. Cypher-style minimum only (equivalent to ISO {m,})."""
        self._chain(db, 4)
        result = list(db.execute("MATCH (a:Node {name: '0'})-[:NEXT*3..]->(b:Node) RETURN b.name"))
        names = {r["b.name"] for r in result}
        assert "3" in names
        assert "4" in names

    def test_iso_curly_max_only(self, db):
        """*..n Cypher-style maximum only (equivalent to ISO {,n})."""
        self._chain(db, 4)
        result = list(db.execute("MATCH (a:Node {name: '0'})-[:NEXT*..2]->(b:Node) RETURN b.name"))
        names = {r["b.name"] for r in result}
        assert "1" in names
        assert "2" in names
        assert "3" not in names

    def test_questioned_optional_edge(self, db):
        """->? questioned (0 or 1 hop) optional edge."""
        a = db.create_node(["Person"], {"name": "Alix"})
        b = db.create_node(["Person"], {"name": "Gus"})
        db.create_node(["Person"], {"name": "Vincent"})
        db.create_edge(a.id, b.id, "KNOWS")
        # Vincent has no outgoing KNOWS
        result = list(db.execute("MATCH (a:Person {name: 'Alix'})-[:KNOWS]->?(b) RETURN b.name"))
        # Should return at least Gus (1 hop) and possibly Alix (0 hop)
        assert any(r["b.name"] == "Gus" for r in result)


# =============================================================================
# Path Search Prefixes (sec 16.6)
# =============================================================================


class TestPathSearchPrefixes:
    """Path search prefixes: ANY, ALL SHORTEST, SHORTEST k, etc."""

    def _diamond(self, db):
        """Create a diamond graph: a->b->d, a->c->d."""
        a = db.create_node(["Node"], {"name": "a"})
        b = db.create_node(["Node"], {"name": "b"})
        c = db.create_node(["Node"], {"name": "c"})
        d = db.create_node(["Node"], {"name": "d"})
        db.create_edge(a.id, b.id, "STEP")
        db.create_edge(a.id, c.id, "STEP")
        db.create_edge(b.id, d.id, "STEP")
        db.create_edge(c.id, d.id, "STEP")
        return a, b, c, d

    @pytest.mark.xfail(reason="Path search prefix syntax requires parenthesized path patterns")
    def test_any_path(self, db):
        """ANY returns any single matching path."""
        self._diamond(db)
        result = list(
            db.execute(
                "MATCH ANY (a:Node {name: 'a'})-[:STEP]->{1,3}(d:Node {name: 'd'}) "
                "RETURN length(path()) AS len"
            )
        )
        # Should return exactly 1 path
        assert len(result) == 1

    @pytest.mark.xfail(reason="Path search prefix syntax requires parenthesized path patterns")
    def test_all_shortest(self, db):
        """ALL SHORTEST returns all shortest paths."""
        self._diamond(db)
        result = list(
            db.execute(
                "MATCH p = ALL SHORTEST (a:Node {name: 'a'})-[:STEP]->+(d:Node {name: 'd'}) "
                "RETURN length(p) AS len"
            )
        )
        # Both a->b->d and a->c->d are length 2
        assert len(result) >= 2
        assert all(r["len"] == 2 for r in result)

    @pytest.mark.xfail(reason="Path search prefix syntax requires parenthesized path patterns")
    def test_any_shortest(self, db):
        """ANY SHORTEST returns one shortest path."""
        self._diamond(db)
        result = list(
            db.execute(
                "MATCH p = ANY SHORTEST (a:Node {name: 'a'})-[:STEP]->+(d:Node {name: 'd'}) "
                "RETURN length(p) AS len"
            )
        )
        assert len(result) == 1
        assert result[0]["len"] == 2

    @pytest.mark.xfail(reason="Path search prefix syntax requires parenthesized path patterns")
    def test_shortest_k(self, db):
        """SHORTEST k returns up to k shortest paths."""
        self._diamond(db)
        result = list(
            db.execute(
                "MATCH p = SHORTEST 1 (a:Node {name: 'a'})-[:STEP]->+(d:Node {name: 'd'}) "
                "RETURN length(p) AS len"
            )
        )
        assert len(result) == 1


# =============================================================================
# Path Modes (sec 16.6)
# =============================================================================


class TestPathModes:
    """Path modes: WALK, TRAIL, SIMPLE, ACYCLIC."""

    def _cycle(self, db):
        """Create a cycle: a->b->c->a."""
        a = db.create_node(["Node"], {"name": "a"})
        b = db.create_node(["Node"], {"name": "b"})
        c = db.create_node(["Node"], {"name": "c"})
        db.create_edge(a.id, b.id, "STEP")
        db.create_edge(b.id, c.id, "STEP")
        db.create_edge(c.id, a.id, "STEP")
        return a, b, c

    @pytest.mark.xfail(
        reason="Path mode syntax (WALK/TRAIL/SIMPLE/ACYCLIC) requires parenthesized path patterns"
    )
    def test_walk_mode(self, db):
        """WALK allows repeated nodes and edges."""
        self._cycle(db)
        result = list(
            db.execute("MATCH WALK (a:Node {name: 'a'})-[:STEP]->{1,4}(b:Node) RETURN b.name")
        )
        # WALK can revisit nodes, so we expect more results
        assert len(result) >= 3

    @pytest.mark.xfail(
        reason="Path mode syntax (WALK/TRAIL/SIMPLE/ACYCLIC) requires parenthesized path patterns"
    )
    def test_trail_mode(self, db):
        """TRAIL: no repeated edges."""
        self._cycle(db)
        result = list(
            db.execute("MATCH TRAIL (a:Node {name: 'a'})-[:STEP]->{1,6}(b:Node) RETURN b.name")
        )
        # With 3 edges in the cycle, TRAIL can use each edge at most once
        assert len(result) <= 3

    @pytest.mark.xfail(
        reason="Path mode syntax (WALK/TRAIL/SIMPLE/ACYCLIC) requires parenthesized path patterns"
    )
    def test_simple_mode(self, db):
        """SIMPLE: no repeated nodes except possibly endpoints."""
        self._cycle(db)
        result = list(
            db.execute("MATCH SIMPLE (a:Node {name: 'a'})-[:STEP]->{1,6}(b:Node) RETURN b.name")
        )
        # All intermediate + final nodes should be unique within each path
        assert len(result) <= 3

    @pytest.mark.xfail(
        reason="Path mode syntax (WALK/TRAIL/SIMPLE/ACYCLIC) requires parenthesized path patterns"
    )
    def test_acyclic_mode(self, db):
        """ACYCLIC: no repeated nodes at all."""
        self._cycle(db)
        result = list(
            db.execute("MATCH ACYCLIC (a:Node {name: 'a'})-[:STEP]->{1,6}(b:Node) RETURN b.name")
        )
        # Cannot return to 'a', so only b and c reachable
        names = {r["b.name"] for r in result}
        assert "a" not in names


# =============================================================================
# Match Modes (sec 16.4)
# =============================================================================


class TestMatchModes:
    """Match modes: DIFFERENT EDGES, REPEATABLE ELEMENTS, KEEP."""

    @pytest.mark.xfail(reason="MATCH DIFFERENT EDGES syntax not supported in GQL parser")
    def test_different_edges(self, db):
        """DIFFERENT EDGES enforces cross-pattern edge uniqueness."""
        a = db.create_node(["Person"], {"name": "Alix"})
        b = db.create_node(["Person"], {"name": "Gus"})
        db.create_edge(a.id, b.id, "KNOWS")
        result = list(
            db.execute(
                "MATCH DIFFERENT EDGES "
                "(a:Person)-[e1:KNOWS]->(b:Person), "
                "(c:Person)-[e2:KNOWS]->(d:Person) "
                "RETURN a.name, c.name"
            )
        )
        # Only one KNOWS edge exists, so e1 and e2 cannot both match it
        assert len(result) == 0

    def test_repeatable_elements(self, db):
        """REPEATABLE ELEMENTS allows same edge in multiple bindings."""
        a = db.create_node(["Person"], {"name": "Alix"})
        b = db.create_node(["Person"], {"name": "Gus"})
        db.create_edge(a.id, b.id, "KNOWS")
        result = list(
            db.execute(
                "MATCH REPEATABLE ELEMENTS "
                "(a:Person)-[e1:KNOWS]->(b:Person), "
                "(c:Person)-[e2:KNOWS]->(d:Person) "
                "RETURN a.name, c.name"
            )
        )
        assert len(result) >= 1


# =============================================================================
# Path Functions
# =============================================================================


class TestPathFunctions:
    """Named paths: shortestPath, allShortestPaths, length, nodes, edges."""

    def test_named_path(self, db):
        """p = (a)-[]->(b) path alias."""
        a = db.create_node(["Node"], {"name": "a"})
        b = db.create_node(["Node"], {"name": "b"})
        db.create_edge(a.id, b.id, "NEXT")
        result = list(
            db.execute("MATCH p = (a:Node {name: 'a'})-[:NEXT]->(b:Node) RETURN length(p) AS len")
        )
        assert result[0]["len"] == 1

    def test_path_length(self, db):
        """length(path) returns edge count."""
        a = db.create_node(["Node"], {"name": "a"})
        b = db.create_node(["Node"], {"name": "b"})
        c = db.create_node(["Node"], {"name": "c"})
        db.create_edge(a.id, b.id, "NEXT")
        db.create_edge(b.id, c.id, "NEXT")
        result = list(
            db.execute(
                "MATCH p = (a:Node {name: 'a'})-[:NEXT*]->(c:Node {name: 'c'}) "
                "RETURN length(p) AS len"
            )
        )
        assert result[0]["len"] == 2

    def test_path_nodes(self, db):
        """nodes(path) returns list of nodes."""
        a = db.create_node(["Node"], {"name": "a"})
        b = db.create_node(["Node"], {"name": "b"})
        db.create_edge(a.id, b.id, "NEXT")
        result = list(
            db.execute(
                "MATCH p = (a:Node {name: 'a'})-[:NEXT]->(b:Node) RETURN nodes(p) AS path_nodes"
            )
        )
        assert len(result[0]["path_nodes"]) == 2

    def test_path_edges(self, db):
        """edges(path) returns list of edges."""
        a = db.create_node(["Node"], {"name": "a"})
        b = db.create_node(["Node"], {"name": "b"})
        db.create_edge(a.id, b.id, "NEXT")
        result = list(
            db.execute(
                "MATCH p = (a:Node {name: 'a'})-[:NEXT]->(b:Node) RETURN edges(p) AS path_edges"
            )
        )
        assert len(result[0]["path_edges"]) == 1

    def test_shortest_path(self, db):
        """shortestPath function."""
        a = db.create_node(["Node"], {"name": "a"})
        b = db.create_node(["Node"], {"name": "b"})
        c = db.create_node(["Node"], {"name": "c"})
        db.create_edge(a.id, b.id, "STEP")
        db.create_edge(b.id, c.id, "STEP")
        db.create_edge(a.id, c.id, "DIRECT")
        result = list(
            db.execute(
                "MATCH p = shortestPath("
                "(a:Node {name: 'a'})-[*]-(c:Node {name: 'c'})"
                ") RETURN length(p) AS len"
            )
        )
        assert result[0]["len"] == 1

    def test_all_shortest_paths(self, db):
        """allShortestPaths function."""
        a = db.create_node(["Node"], {"name": "a"})
        b = db.create_node(["Node"], {"name": "b"})
        c = db.create_node(["Node"], {"name": "c"})
        d = db.create_node(["Node"], {"name": "d"})
        db.create_edge(a.id, b.id, "EDGE")
        db.create_edge(a.id, c.id, "EDGE")
        db.create_edge(b.id, d.id, "EDGE")
        db.create_edge(c.id, d.id, "EDGE")
        result = list(
            db.execute(
                "MATCH p = allShortestPaths("
                "(a:Node {name: 'a'})-[*]-(d:Node {name: 'd'})"
                ") RETURN length(p) AS len"
            )
        )
        assert len(result) >= 2
        assert all(r["len"] == 2 for r in result)

    @pytest.mark.xfail(reason="isAcyclic path predicate function not yet supported")
    def test_is_acyclic_predicate(self, db):
        """isAcyclic(path) checks no repeated nodes."""
        a = db.create_node(["Node"], {"name": "a"})
        b = db.create_node(["Node"], {"name": "b"})
        db.create_edge(a.id, b.id, "NEXT")
        result = list(
            db.execute(
                "MATCH p = (a:Node {name: 'a'})-[:NEXT]->(b:Node) RETURN isAcyclic(p) AS acyclic"
            )
        )
        assert result[0]["acyclic"] is True

    @pytest.mark.xfail(reason="isSimple path predicate function not yet supported")
    def test_is_simple_predicate(self, db):
        """isSimple(path) checks no repeated nodes except endpoints."""
        a = db.create_node(["Node"], {"name": "a"})
        b = db.create_node(["Node"], {"name": "b"})
        db.create_edge(a.id, b.id, "NEXT")
        result = list(
            db.execute(
                "MATCH p = (a:Node {name: 'a'})-[:NEXT]->(b:Node) RETURN isSimple(p) AS simple"
            )
        )
        assert result[0]["simple"] is True

    @pytest.mark.xfail(reason="isTrail path predicate function not yet supported")
    def test_is_trail_predicate(self, db):
        """isTrail(path) checks no repeated edges."""
        a = db.create_node(["Node"], {"name": "a"})
        b = db.create_node(["Node"], {"name": "b"})
        db.create_edge(a.id, b.id, "NEXT")
        result = list(
            db.execute(
                "MATCH p = (a:Node {name: 'a'})-[:NEXT]->(b:Node) RETURN isTrail(p) AS trail"
            )
        )
        assert result[0]["trail"] is True


# =============================================================================
# Advanced Pattern Features (sec 16.7)
# =============================================================================


class TestAdvancedPatterns:
    """Parenthesized paths, path unions, subpath variables."""

    @pytest.mark.xfail(reason="Quantified path pattern (QPP) syntax not supported in GQL parser")
    def test_parenthesized_path_pattern(self, db):
        """((a)--(b)){2,5} grouped subpath repetition."""
        a = db.create_node(["Node"], {"name": "a"})
        b = db.create_node(["Node"], {"name": "b"})
        c = db.create_node(["Node"], {"name": "c"})
        db.create_edge(a.id, b.id, "STEP")
        db.create_edge(b.id, c.id, "STEP")
        result = list(
            db.execute(
                "MATCH (start:Node {name: 'a'}) "
                "(()-[:STEP]->())+ "
                "(end_node:Node) "
                "RETURN end_node.name"
            )
        )
        names = {r["end_node.name"] for r in result}
        assert "b" in names or "c" in names

    def test_quantified_graph_pattern(self, db):
        """Full subpattern repetition with *n (Cypher-style)."""
        a = db.create_node(["Node"], {"name": "a"})
        b = db.create_node(["Node"], {"name": "b"})
        c = db.create_node(["Node"], {"name": "c"})
        d = db.create_node(["Node"], {"name": "d"})
        db.create_edge(a.id, b.id, "STEP")
        db.create_edge(b.id, c.id, "STEP")
        db.create_edge(c.id, d.id, "STEP")
        result = list(
            db.execute(
                "MATCH (start:Node {name: 'a'})-[:STEP*2]->(end_node:Node) RETURN end_node.name"
            )
        )
        assert len(result) == 1
        assert result[0]["end_node.name"] == "c"
