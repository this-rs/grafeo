"""Gremlin TinkerPop 3.x syntax compliance tests.

Comprehensive coverage of Gremlin traversal language elements:
source steps, navigation, filters, map steps, side effects, and predicates.
"""

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _exec(db, query: str):
    """Execute a Gremlin query, skipping when support is absent."""
    try:
        return db.execute_gremlin(query)
    except AttributeError:
        pytest.skip("Gremlin support not available")
        return None
    except NotImplementedError:
        pytest.skip("Gremlin not implemented")
        return None


def _rows(db, query: str) -> list:
    """Execute and materialise the result list."""
    return list(_exec(db, query))


def _scalar(db, query: str):
    """Return the single scalar produced by an aggregation step."""
    rows = _rows(db, query)
    assert len(rows) >= 1, f"Expected at least one result from: {query}"
    val = rows[0]
    # The engine may return a bare int/float or a single-key dict.
    if isinstance(val, dict):
        return next(iter(val.values()))
    return val


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def social_graph(db):
    """Build a small social network and return (db, ids).

    Topology:
        Alix -knows(since:2018)-> Gus
        Alix -knows(since:2020)-> Vincent
        Gus   -knows(since:2019)-> Vincent
        Gus   -knows(since:2021)-> Jules
        Alix -works_at(role:'engineer')-> Acme
        Gus   -works_at(role:'manager')-> Acme

    Node properties:
        Alix:   Person  age=30, city="NYC",     score=4.5
        Gus:     Person  age=25, city="LA",      score=3.8
        Vincent: Person  age=35, city="NYC",     score=4.9
        Jules:   Person  age=28, city="Chicago", score=4.1
        Acme:    Company revenue=1000000
    """
    alix = db.create_node(["Person"], {"name": "Alix", "age": 30, "city": "NYC", "score": 4.5})
    gus = db.create_node(["Person"], {"name": "Gus", "age": 25, "city": "LA", "score": 3.8})
    vincent = db.create_node(
        ["Person"], {"name": "Vincent", "age": 35, "city": "NYC", "score": 4.9}
    )
    jules = db.create_node(
        ["Person"], {"name": "Jules", "age": 28, "city": "Chicago", "score": 4.1}
    )
    acme = db.create_node(["Company"], {"name": "Acme", "revenue": 1000000})

    e1 = db.create_edge(alix.id, gus.id, "knows", {"since": 2018})
    e2 = db.create_edge(alix.id, vincent.id, "knows", {"since": 2020})
    e3 = db.create_edge(gus.id, vincent.id, "knows", {"since": 2019})
    e4 = db.create_edge(gus.id, jules.id, "knows", {"since": 2021})
    e5 = db.create_edge(alix.id, acme.id, "works_at", {"role": "engineer"})
    e6 = db.create_edge(gus.id, acme.id, "works_at", {"role": "manager"})

    ids = {
        "alix": alix.id,
        "gus": gus.id,
        "vincent": vincent.id,
        "jules": jules.id,
        "acme": acme.id,
        "e1": e1.id,
        "e2": e2.id,
        "e3": e3.id,
        "e4": e4.id,
        "e5": e5.id,
        "e6": e6.id,
    }
    return db, ids


# ===================================================================
# Source Steps
# ===================================================================


class TestGremlinSourceSteps:
    """g.V(), g.V(id), g.E(), g.E(id), g.addV(), g.addE()."""

    def test_g_v_all(self, social_graph):
        """g.V() returns every vertex."""
        db, _ = social_graph
        rows = _rows(db, "g.V()")
        assert len(rows) == 5

    def test_g_v_by_id(self, social_graph):
        """g.V(id) returns the single vertex with that id."""
        db, ids = social_graph
        rows = _rows(db, f"g.V({ids['alix']})")
        assert len(rows) == 1

    def test_g_e_all(self, social_graph):
        """g.E() returns every edge."""
        db, _ = social_graph
        rows = _rows(db, "g.E()")
        assert len(rows) == 6

    def test_g_e_by_id(self, social_graph):
        """g.E(id) returns the single edge with that id."""
        db, ids = social_graph
        rows = _rows(db, f"g.E({ids['e1']})")
        assert len(rows) == 1

    def test_add_v_basic(self, db):
        """g.addV('Label') creates a vertex."""
        _exec(db, "g.addV('Robot')")
        assert len(_rows(db, "g.V().hasLabel('Robot')")) == 1

    def test_add_v_with_properties(self, db):
        """g.addV('L').property(k,v) attaches properties."""
        _exec(db, "g.addV('Robot').property('name', 'R2D2').property('height', 96)")
        rows = _rows(db, "g.V().has('name', 'R2D2')")
        assert len(rows) == 1

    def test_add_e_basic(self, social_graph):
        """g.V().addE('label').to(g.V()) creates an edge."""
        db, _ = social_graph
        _exec(
            db,
            "g.V().has('name', 'Vincent').addE('knows').to(g.V().has('name', 'Jules'))",
        )
        rows = _rows(db, "g.V().has('name', 'Vincent').out('knows')")
        assert len(rows) >= 1

    def test_add_e_with_property(self, social_graph):
        """g.addE with .property(k,v) stores edge properties."""
        db, _ = social_graph
        _exec(
            db,
            "g.V().has('name', 'Jules').addE('knows').to(g.V().has('name', 'Alix'))"
            ".property('since', 2025)",
        )
        rows = _rows(db, "g.V().has('name', 'Jules').outE('knows').has('since', 2025)")
        assert len(rows) == 1


# ===================================================================
# Navigation Steps
# ===================================================================


class TestGremlinNavigation:
    """out, in, both, outE, inE, bothE, outV, inV, bothV, otherV."""

    # -- vertex-to-vertex navigation ----------------------------------

    def test_out_labeled(self, social_graph):
        """out('knows') follows outgoing 'knows' edges."""
        db, _ = social_graph
        rows = _rows(db, "g.V().has('name', 'Alix').out('knows')")
        assert len(rows) == 2  # Gus, Vincent

    def test_out_unlabeled(self, social_graph):
        """out() follows all outgoing edges regardless of label."""
        db, _ = social_graph
        # Alix has knows->Gus, knows->Vincent, works_at->Acme
        rows = _rows(db, "g.V().has('name', 'Alix').out()")
        assert len(rows) == 3

    def test_in_labeled(self, social_graph):
        """in('knows') follows incoming 'knows' edges."""
        db, _ = social_graph
        # Vincent receives knows from Alix and Gus
        rows = _rows(db, "g.V().has('name', 'Vincent').in('knows')")
        assert len(rows) == 2

    def test_in_unlabeled(self, social_graph):
        """in() follows all incoming edges."""
        db, _ = social_graph
        # Acme receives works_at from Alix and Gus
        rows = _rows(db, "g.V().has('name', 'Acme').in()")
        assert len(rows) == 2

    def test_both(self, social_graph):
        """both('knows') returns neighbours in either direction."""
        db, _ = social_graph
        # Gus: in(knows) from Alix, out(knows) to Vincent and Jules
        rows = _rows(db, "g.V().has('name', 'Gus').both('knows')")
        assert len(rows) == 3

    # -- vertex-to-edge navigation ------------------------------------

    def test_out_e(self, social_graph):
        """outE('knows') returns outgoing edge objects."""
        db, _ = social_graph
        rows = _rows(db, "g.V().has('name', 'Alix').outE('knows')")
        assert len(rows) == 2

    def test_in_e(self, social_graph):
        """inE('knows') returns incoming edge objects."""
        db, _ = social_graph
        rows = _rows(db, "g.V().has('name', 'Vincent').inE('knows')")
        assert len(rows) == 2

    def test_both_e(self, social_graph):
        """bothE('knows') returns edges in either direction."""
        db, _ = social_graph
        # Gus: inE(knows)=1 (from Alix), outE(knows)=2 (to Vincent, Jules)
        rows = _rows(db, "g.V().has('name', 'Gus').bothE('knows')")
        assert len(rows) == 3

    # -- edge-to-vertex navigation ------------------------------------

    def test_out_v(self, social_graph):
        """outV() returns the source vertex of an edge."""
        db, _ = social_graph
        rows = _rows(db, "g.V().has('name', 'Alix').outE('knows').outV()")
        # Each edge's source is Alix, so 2 results all pointing to Alix
        assert len(rows) == 2

    def test_in_v(self, social_graph):
        """inV() returns the target vertex of an edge."""
        db, _ = social_graph
        rows = _rows(db, "g.V().has('name', 'Alix').outE('knows').inV()")
        assert len(rows) == 2  # Gus and Vincent

    def test_both_v(self, social_graph):
        """bothV() returns both endpoints of an edge."""
        db, ids = social_graph
        rows = _rows(db, f"g.E({ids['e1']}).bothV()")
        assert len(rows) == 2  # Alix and Gus

    def test_other_v(self, social_graph):
        """otherV() returns the vertex at the other end relative to the traverser."""
        db, _ = social_graph
        rows = _rows(db, "g.V().has('name', 'Alix').outE('knows').otherV()")
        # From Alix's perspective, otherV is the far end: Gus, Vincent
        assert len(rows) == 2


# ===================================================================
# Filter Steps
# ===================================================================


class TestGremlinFilter:
    """has, hasLabel, hasId, hasNot, filter, where, and, or, not, dedup,
    limit, skip, range."""

    # -- has variants --------------------------------------------------

    def test_has_key_value(self, social_graph):
        """has(key, value) filters by exact property match."""
        db, _ = social_graph
        rows = _rows(db, "g.V().has('name', 'Alix')")
        assert len(rows) == 1

    def test_has_label_key_value(self, social_graph):
        """has(label, key, value) filters by label and property."""
        db, _ = social_graph
        rows = _rows(db, "g.V().has('Person', 'city', 'NYC')")
        assert len(rows) == 2  # Alix and Vincent

    def test_has_key_predicate(self, social_graph):
        """has(key, predicate) filters using a predicate."""
        db, _ = social_graph
        rows = _rows(db, "g.V().has('age', gt(29))")
        assert len(rows) == 2  # Alix(30), Vincent(35)

    def test_has_label(self, social_graph):
        """hasLabel('Person') filters by vertex label."""
        db, _ = social_graph
        rows = _rows(db, "g.V().hasLabel('Person')")
        assert len(rows) == 4

    def test_has_label_company(self, social_graph):
        """hasLabel('Company') filters to Company vertices only."""
        db, _ = social_graph
        rows = _rows(db, "g.V().hasLabel('Company')")
        assert len(rows) == 1

    def test_has_id(self, social_graph):
        """hasId(id) filters to the vertex with a specific id."""
        db, ids = social_graph
        rows = _rows(db, f"g.V().hasId({ids['gus']})")
        assert len(rows) == 1

    def test_has_not(self, social_graph):
        """hasNot('key') keeps traversers without the given property."""
        db, _ = social_graph
        # Only Company nodes lack the 'age' property
        rows = _rows(db, "g.V().hasNot('age')")
        assert len(rows) == 1

    # -- boolean combinators -------------------------------------------

    def test_and_step(self, social_graph):
        """and() combines two filter conditions with logical AND."""
        db, _ = social_graph
        rows = _rows(
            db,
            "g.V().and(has('age', gt(24)), has('city', 'NYC'))",
        )
        # Alix(30, NYC) and Vincent(35, NYC)
        assert len(rows) == 2

    def test_or_step(self, social_graph):
        """or() combines two filter conditions with logical OR."""
        db, _ = social_graph
        rows = _rows(
            db,
            "g.V().or(has('city', 'LA'), has('city', 'Chicago'))",
        )
        # Gus(LA) and Jules(Chicago)
        assert len(rows) == 2

    def test_not_step(self, social_graph):
        """not() negates a filter condition."""
        db, _ = social_graph
        rows = _rows(db, "g.V().hasLabel('Person').not(has('city', 'NYC'))")
        # Gus(LA), Jules(Chicago)
        assert len(rows) == 2

    # -- where / filter ------------------------------------------------

    def test_where_step(self, social_graph):
        """where() applies an inline filter to the traversal."""
        db, _ = social_graph
        rows = _rows(
            db,
            "g.V().hasLabel('Person').where(out('knows'))",
        )
        # Alix and Gus have outgoing 'knows' edges
        assert len(rows) == 2

    def test_filter_step(self, social_graph):
        """filter() applies a predicate to each traverser."""
        db, _ = social_graph
        rows = _rows(
            db,
            "g.V().hasLabel('Person').filter(has('age', gt(27)))",
        )
        # Alix(30), Vincent(35), Jules(28)
        assert len(rows) == 3

    # -- dedup / limit / skip / range ----------------------------------

    def test_dedup(self, social_graph):
        """dedup() removes duplicate traversers."""
        db, _ = social_graph
        # Vincent is reachable from both Alix and Gus, but dedup keeps one
        rows = _rows(db, "g.V().hasLabel('Person').out('knows').dedup()")
        # Alix->Gus,Vincent; Gus->Vincent,Jules; dedup => Gus, Vincent, Jules
        assert len(rows) == 3

    def test_limit(self, social_graph):
        """limit(n) caps the traversal to n results."""
        db, _ = social_graph
        rows = _rows(db, "g.V().hasLabel('Person').limit(2)")
        assert len(rows) == 2

    def test_skip(self, social_graph):
        """skip(n) drops the first n traversers."""
        db, _ = social_graph
        all_rows = _rows(db, "g.V().hasLabel('Person')")
        skipped = _rows(db, "g.V().hasLabel('Person').skip(1)")
        assert len(skipped) == len(all_rows) - 1

    def test_range(self, social_graph):
        """range(low, high) keeps traversers in the [low, high) window."""
        db, _ = social_graph
        rows = _rows(db, "g.V().hasLabel('Person').range(1, 3)")
        assert len(rows) == 2


# ===================================================================
# Map Steps
# ===================================================================


class TestGremlinMap:
    """id, label, values, valueMap, elementMap, properties, constant,
    count, sum, min, max, mean, fold, unfold, group, groupCount,
    path, select, project, choose, optional, union, coalesce, order."""

    # -- element access ------------------------------------------------

    def test_id_step(self, social_graph):
        """id() extracts vertex identifiers."""
        db, ids = social_graph
        rows = _rows(db, "g.V().has('name', 'Alix').id()")
        assert len(rows) == 1
        val = rows[0] if not isinstance(rows[0], dict) else next(iter(rows[0].values()))
        assert val == ids["alix"]

    def test_label_step(self, social_graph):
        """label() extracts vertex labels."""
        db, _ = social_graph
        rows = _rows(db, "g.V().has('name', 'Alix').label()")
        assert len(rows) == 1
        val = rows[0] if isinstance(rows[0], str) else next(iter(rows[0].values()))
        assert val == "Person"

    def test_values_single_key(self, social_graph):
        """values('key') projects a single property."""
        db, _ = social_graph
        rows = _rows(db, "g.V().has('name', 'Alix').values('age')")
        assert len(rows) == 1
        val = rows[0] if not isinstance(rows[0], dict) else next(iter(rows[0].values()))
        assert val == 30

    def test_values_multiple_keys(self, social_graph):
        """values('k1','k2') projects several properties as separate traversers."""
        db, _ = social_graph
        rows = _rows(db, "g.V().has('name', 'Alix').values('name', 'age')")
        assert len(rows) == 2

    def test_value_map(self, social_graph):
        """valueMap() returns property maps for each vertex."""
        db, _ = social_graph
        rows = _rows(db, "g.V().has('name', 'Alix').valueMap()")
        assert len(rows) >= 1
        # The result should be a dict-like structure
        row = rows[0]
        assert isinstance(row, dict)

    def test_element_map(self, social_graph):
        """elementMap() returns id, label, and all properties."""
        db, _ = social_graph
        rows = _rows(db, "g.V().has('name', 'Alix').elementMap()")
        assert len(rows) == 1
        row = rows[0]
        assert isinstance(row, dict)

    def test_properties_step(self, social_graph):
        """properties() returns property objects."""
        db, _ = social_graph
        rows = _rows(db, "g.V().has('name', 'Alix').properties('name')")
        assert len(rows) >= 1

    def test_constant_step(self, social_graph):
        """constant(val) injects a fixed value for each traverser."""
        db, _ = social_graph
        rows = _rows(db, "g.V().hasLabel('Person').constant(42)")
        assert len(rows) == 4
        for r in rows:
            val = r if not isinstance(r, dict) else next(iter(r.values()))
            assert val == 42

    # -- aggregation ---------------------------------------------------

    def test_count(self, social_graph):
        """count() returns total traverser count."""
        db, _ = social_graph
        val = _scalar(db, "g.V().hasLabel('Person').count()")
        assert val == 4

    def test_sum(self, social_graph):
        """sum() adds numeric traverser values."""
        db, _ = social_graph
        val = _scalar(db, "g.V().hasLabel('Person').values('age').sum()")
        # 30 + 25 + 35 + 28 = 118
        assert val == 118

    def test_min(self, social_graph):
        """min() returns the smallest value."""
        db, _ = social_graph
        val = _scalar(db, "g.V().hasLabel('Person').values('age').min()")
        assert val == 25

    def test_max(self, social_graph):
        """max() returns the largest value."""
        db, _ = social_graph
        val = _scalar(db, "g.V().hasLabel('Person').values('age').max()")
        assert val == 35

    def test_mean(self, social_graph):
        """mean() returns the arithmetic average."""
        db, _ = social_graph
        val = _scalar(db, "g.V().hasLabel('Person').values('age').mean()")
        # 118 / 4 = 29.5
        assert val == pytest.approx(29.5)

    # -- collection manipulation ---------------------------------------

    def test_fold(self, social_graph):
        """fold() collapses all traversers into a single list."""
        db, _ = social_graph
        rows = _rows(db, "g.V().hasLabel('Person').values('name').fold()")
        assert len(rows) == 1
        folded = rows[0]
        # Result may be a bare list or a single-key dict wrapping a list
        if isinstance(folded, dict):
            folded = next(iter(folded.values()))
        assert isinstance(folded, list)
        assert len(folded) == 4

    def test_unfold(self, social_graph):
        """unfold() expands a collection back into individual traversers."""
        db, _ = social_graph
        rows = _rows(db, "g.V().hasLabel('Person').values('name').fold().unfold()")
        assert len(rows) == 4

    def test_group(self, social_graph):
        """group().by() groups traversers by key."""
        db, _ = social_graph
        rows = _rows(
            db,
            "g.V().hasLabel('Person').group().by('city').by('name')",
        )
        assert len(rows) >= 1
        grouped = rows[0]
        assert isinstance(grouped, dict)

    def test_group_count(self, social_graph):
        """groupCount().by() counts by key."""
        db, _ = social_graph
        rows = _rows(
            db,
            "g.V().hasLabel('Person').groupCount().by('city')",
        )
        assert len(rows) >= 1
        raw = rows[0]
        assert isinstance(raw, dict)
        # The engine returns a single-column row; unwrap the inner map.
        counts = next(iter(raw.values())) if len(raw) == 1 else raw
        assert isinstance(counts, dict)
        # NYC has Alix and Vincent
        assert counts.get("NYC", 0) == 2

    # -- path and projection -------------------------------------------

    def test_path(self, social_graph):
        """path() captures the full traversal route."""
        db, _ = social_graph
        rows = _rows(
            db,
            "g.V().has('name', 'Alix').out('knows').out('knows').path()",
        )
        # Alix->Gus->Vincent, Alix->Gus->Jules
        assert len(rows) >= 1

    def test_select_after_as(self, social_graph):
        """select('a') retrieves a labeled traverser."""
        db, _ = social_graph
        rows = _rows(
            db,
            "g.V().has('name', 'Alix').as('a').out('knows').select('a')",
        )
        # Two outgoing 'knows' edges, but select('a') always points to Alix
        assert len(rows) == 2

    def test_project(self, social_graph):
        """project('k1','k2').by().by() creates named projections."""
        db, _ = social_graph
        rows = _rows(
            db,
            "g.V().has('name', 'Alix').project('n', 'a').by('name').by('age')",
        )
        assert len(rows) == 1
        proj = rows[0]
        assert isinstance(proj, dict)
        assert proj.get("n") == "Alix"
        assert proj.get("a") == 30

    # -- branching / conditional ---------------------------------------

    def test_choose(self, social_graph):
        """choose(pred, true_branch, false_branch) routes traversers."""
        db, _ = social_graph
        rows = _rows(
            db,
            "g.V().hasLabel('Person')"
            ".choose(has('age', gt(29)), constant('senior'), constant('junior'))",
        )
        assert len(rows) == 4
        values = [r if not isinstance(r, dict) else next(iter(r.values())) for r in rows]
        assert values.count("senior") == 2  # Alix(30), Vincent(35)
        assert values.count("junior") == 2  # Gus(25), Jules(28)

    def test_optional(self, social_graph):
        """optional() keeps the current traverser when the inner traversal is empty."""
        db, _ = social_graph
        # Jules has no outgoing 'knows', so optional keeps her
        rows = _rows(
            db,
            "g.V().has('name', 'Jules').optional(out('knows'))",
        )
        assert len(rows) >= 1

    def test_union(self, social_graph):
        """union() merges results from multiple sub-traversals."""
        db, _ = social_graph
        rows = _rows(
            db,
            "g.V().has('name', 'Alix').union(out('knows'), out('works_at'))",
        )
        # 2 knows + 1 works_at = 3
        assert len(rows) == 3

    def test_coalesce(self, social_graph):
        """coalesce() returns the first non-empty traversal."""
        db, _ = social_graph
        # Jules has no 'works_at' but does have incoming 'knows' (from Gus)
        rows = _rows(
            db,
            "g.V().has('name', 'Jules').coalesce(out('works_at'), in('knows'))",
        )
        assert len(rows) >= 1

    # -- ordering ------------------------------------------------------

    def test_order_asc(self, social_graph):
        """order().by('key', asc) sorts ascending."""
        db, _ = social_graph
        rows = _rows(
            db,
            "g.V().hasLabel('Person').order().by('age', asc).values('name')",
        )
        assert len(rows) == 4
        first = rows[0] if isinstance(rows[0], str) else rows[0].get("name")
        assert first == "Gus"  # youngest at 25

    def test_order_desc(self, social_graph):
        """order().by('key', desc) sorts descending."""
        db, _ = social_graph
        rows = _rows(
            db,
            "g.V().hasLabel('Person').order().by('age', desc).values('name')",
        )
        assert len(rows) == 4
        first = rows[0] if isinstance(rows[0], str) else rows[0].get("name")
        assert first == "Vincent"  # oldest at 35


# ===================================================================
# Side Effect Steps
# ===================================================================


class TestGremlinSideEffects:
    """as, property, drop, sideEffect, aggregate, store."""

    def test_as_and_select(self, social_graph):
        """as('label') stores a reference retrievable by select()."""
        db, _ = social_graph
        rows = _rows(
            db,
            "g.V().has('name', 'Alix').as('person')"
            ".out('knows').as('friend')"
            ".select('person', 'friend')",
        )
        assert len(rows) == 2
        for row in rows:
            assert isinstance(row, dict)
            assert "person" in row
            assert "friend" in row

    def test_property_set(self, db):
        """property(key, val) sets a property on an existing vertex."""
        db.create_node(["Gadget"], {"name": "Widget"})
        _exec(db, "g.V().has('name', 'Widget').property('color', 'red')")
        rows = _rows(db, "g.V().has('name', 'Widget').values('color')")
        assert len(rows) == 1
        val = rows[0] if isinstance(rows[0], str) else next(iter(rows[0].values()))
        assert val == "red"

    def test_property_overwrite(self, db):
        """property() overwrites an existing property value."""
        db.create_node(["Gadget"], {"name": "Widget", "color": "blue"})
        _exec(db, "g.V().has('name', 'Widget').property('color', 'green')")
        rows = _rows(db, "g.V().has('name', 'Widget').values('color')")
        assert len(rows) == 1
        val = rows[0] if isinstance(rows[0], str) else next(iter(rows[0].values()))
        assert val == "green"

    def test_drop_vertex(self, db):
        """drop() removes vertices from the graph."""
        db.create_node(["Temp"], {"name": "Gone"})
        assert len(_rows(db, "g.V().hasLabel('Temp')")) == 1
        _exec(db, "g.V().hasLabel('Temp').drop()")
        assert len(_rows(db, "g.V().hasLabel('Temp')")) == 0

    def test_drop_edge(self, social_graph):
        """drop() removes edges from the graph."""
        db, _ = social_graph
        before = len(_rows(db, "g.V().has('name', 'Alix').outE('knows')"))
        assert before == 2
        _exec(db, "g.V().has('name', 'Alix').outE('knows').limit(1).drop()")
        after = len(_rows(db, "g.V().has('name', 'Alix').outE('knows')"))
        assert after == before - 1

    def test_side_effect(self, social_graph):
        """sideEffect() performs an action without altering the traversal."""
        db, _ = social_graph
        # sideEffect should pass traversers through unchanged
        rows = _rows(
            db,
            "g.V().hasLabel('Person').sideEffect(count()).values('name')",
        )
        assert len(rows) == 4

    def test_aggregate(self, social_graph):
        """aggregate('x') collects traversers into a side-effect list."""
        db, _ = social_graph
        rows = _rows(
            db,
            "g.V().hasLabel('Person').aggregate('x').out('knows').aggregate('y').select('x')",
        )
        # The select('x') should return the collected person vertices
        assert len(rows) >= 1

    def test_store(self, social_graph):
        """store('x') lazily collects traversers into a side-effect list."""
        db, _ = social_graph
        rows = _rows(
            db,
            "g.V().hasLabel('Person').store('x').values('name')",
        )
        assert len(rows) == 4


# ===================================================================
# Predicates
# ===================================================================


class TestGremlinPredicates:
    """eq, neq, lt, lte, gt, gte, between, inside, outside,
    within, without, containing, startingWith, endingWith."""

    # -- comparison predicates -----------------------------------------

    def test_eq(self, social_graph):
        """eq(val) matches equal values."""
        db, _ = social_graph
        rows = _rows(db, "g.V().has('age', eq(30))")
        assert len(rows) == 1  # Alix

    def test_neq(self, social_graph):
        """neq(val) excludes equal values."""
        db, _ = social_graph
        rows = _rows(db, "g.V().hasLabel('Person').has('age', neq(30))")
        assert len(rows) == 3  # Gus, Vincent, Jules

    def test_lt(self, social_graph):
        """lt(val) matches strictly less than."""
        db, _ = social_graph
        rows = _rows(db, "g.V().has('age', lt(28))")
        assert len(rows) == 1  # Gus(25)

    def test_lte(self, social_graph):
        """lte(val) matches less than or equal."""
        db, _ = social_graph
        rows = _rows(db, "g.V().has('age', lte(28))")
        assert len(rows) == 2  # Gus(25), Jules(28)

    def test_gt(self, social_graph):
        """gt(val) matches strictly greater than."""
        db, _ = social_graph
        rows = _rows(db, "g.V().has('age', gt(30))")
        assert len(rows) == 1  # Vincent(35)

    def test_gte(self, social_graph):
        """gte(val) matches greater than or equal."""
        db, _ = social_graph
        rows = _rows(db, "g.V().has('age', gte(30))")
        assert len(rows) == 2  # Alix(30), Vincent(35)

    # -- range predicates ----------------------------------------------

    def test_between(self, social_graph):
        """between(low, high) matches low <= x < high."""
        db, _ = social_graph
        rows = _rows(db, "g.V().has('age', between(25, 31))")
        # 25 <= age < 31: Gus(25), Jules(28), Alix(30)
        assert len(rows) == 3

    def test_inside(self, social_graph):
        """inside(low, high) matches low < x < high (exclusive)."""
        db, _ = social_graph
        rows = _rows(db, "g.V().has('age', inside(25, 35))")
        # 25 < age < 35: Jules(28), Alix(30)
        assert len(rows) == 2

    def test_outside(self, social_graph):
        """outside(low, high) matches x < low or x > high."""
        db, _ = social_graph
        rows = _rows(db, "g.V().has('age', outside(26, 34))")
        # age < 26 or age > 34: Gus(25), Vincent(35)
        assert len(rows) == 2

    # -- membership predicates -----------------------------------------

    def test_within(self, social_graph):
        """within(v1, v2, ...) checks set membership."""
        db, _ = social_graph
        rows = _rows(db, "g.V().has('city', within('NYC', 'LA'))")
        # Alix(NYC), Gus(LA), Vincent(NYC)
        assert len(rows) == 3

    def test_without(self, social_graph):
        """without(v1, v2, ...) excludes set members."""
        db, _ = social_graph
        rows = _rows(db, "g.V().hasLabel('Person').has('city', without('NYC', 'LA'))")
        # Jules(Chicago)
        assert len(rows) == 1

    # -- string predicates ---------------------------------------------

    def test_containing(self, social_graph):
        """containing('sub') matches strings containing a substring."""
        db, _ = social_graph
        rows = _rows(db, "g.V().has('name', containing('li'))")
        # Only Alix contains 'li'
        assert len(rows) == 1

    def test_starting_with(self, social_graph):
        """startingWith('prefix') matches strings beginning with prefix."""
        db, _ = social_graph
        rows = _rows(db, "g.V().has('name', startingWith('Al'))")
        assert len(rows) == 1  # Alix

    def test_ending_with(self, social_graph):
        """endingWith('suffix') matches strings ending with suffix."""
        db, _ = social_graph
        rows = _rows(db, "g.V().has('name', endingWith('ent'))")
        assert len(rows) == 1  # Vincent

    # -- predicate combination -----------------------------------------

    def test_predicate_chain_on_edges(self, social_graph):
        """Predicates work on edge properties too."""
        db, _ = social_graph
        rows = _rows(db, "g.E().has('since', gte(2020))")
        # since=2020 (Alix->Vincent), since=2021 (Gus->Jules)
        assert len(rows) == 2

    def test_predicate_with_navigation(self, social_graph):
        """Predicates compose naturally with navigation steps."""
        db, _ = social_graph
        rows = _rows(
            db,
            "g.V().has('name', 'Alix').outE('knows').has('since', lt(2020)).inV().values('name')",
        )
        # Alix->Gus has since=2018 (< 2020)
        assert len(rows) == 1
        val = rows[0] if isinstance(rows[0], str) else next(iter(rows[0].values()))
        assert val == "Gus"

    def test_within_numeric(self, social_graph):
        """within() works with numeric values, not just strings."""
        db, _ = social_graph
        rows = _rows(db, "g.V().has('age', within(25, 35))")
        # Gus(25), Vincent(35)
        assert len(rows) == 2
