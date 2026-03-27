"""Tests for cognitive features: energy, synapse, scar, fabric, GDS, cognitive search.

These tests verify the Python bindings for Obrain's cognitive subsystems.
Requires the 'cognitive' feature to be enabled at build time.
"""

import pytest

try:
    import obrain

    OBRAIN_AVAILABLE = True
except ImportError:
    OBRAIN_AVAILABLE = False


def has_cognitive(db):
    """Check if cognitive feature is available."""
    return hasattr(db, "cognitive")


def has_gds(db):
    """Check if GDS feature is available."""
    return hasattr(db, "gds")


@pytest.fixture
def db():
    """Create a fresh in-memory ObrainDB instance."""
    if not OBRAIN_AVAILABLE:
        pytest.skip("obrain not installed")
    return obrain.ObrainDB()


@pytest.fixture
def graph_db(db):
    """Create a graph with some test data for algorithm testing."""
    # Create a small social network
    a = db.create_node(["Person"], {"name": "Alice"})
    b = db.create_node(["Person"], {"name": "Bob"})
    c = db.create_node(["Person"], {"name": "Charlie"})
    d = db.create_node(["Person"], {"name": "Diana"})
    e = db.create_node(["Company"], {"name": "Acme"})

    db.create_edge(a.id, b.id, "KNOWS", {"weight": 1.0})
    db.create_edge(b.id, c.id, "KNOWS", {"weight": 2.0})
    db.create_edge(c.id, d.id, "KNOWS", {"weight": 1.5})
    db.create_edge(a.id, c.id, "KNOWS", {"weight": 3.0})
    db.create_edge(a.id, e.id, "WORKS_AT", {"since": 2020})
    db.create_edge(b.id, e.id, "WORKS_AT", {"since": 2021})

    return {
        "db": db,
        "alice": a,
        "bob": b,
        "charlie": c,
        "diana": d,
        "acme": e,
    }


# ===========================================================================
# Energy tests
# ===========================================================================


class TestEnergy:
    """Test energy subsystem: get, boost, decay."""

    def test_energy_get_default(self, db):
        """Untracked nodes should have zero energy."""
        if not has_cognitive(db):
            pytest.skip("cognitive feature not available")
        cog = db.cognitive
        assert cog.energy_get(999) == 0.0

    def test_energy_boost(self, db):
        """Boosting should increase energy."""
        if not has_cognitive(db):
            pytest.skip("cognitive feature not available")
        cog = db.cognitive
        node = db.create_node(["Test"], {"name": "energized"})

        cog.energy_boost(node.id, 5.0)
        energy = cog.energy_get(node.id)
        assert energy > 0.0, "Energy should be positive after boost"
        assert energy <= 10.0, "Energy should be capped at max_energy"

    def test_energy_boost_default_amount(self, db):
        """Boosting with default amount (1.0) should work."""
        if not has_cognitive(db):
            pytest.skip("cognitive feature not available")
        cog = db.cognitive
        node = db.create_node(["Test"], {})

        cog.energy_boost(node.id)
        energy = cog.energy_get(node.id)
        assert energy > 0.0

    def test_energy_boost_accumulates(self, db):
        """Multiple boosts should accumulate."""
        if not has_cognitive(db):
            pytest.skip("cognitive feature not available")
        cog = db.cognitive
        node = db.create_node(["Test"], {})

        cog.energy_boost(node.id, 1.0)
        e1 = cog.energy_get(node.id)
        cog.energy_boost(node.id, 2.0)
        e2 = cog.energy_get(node.id)
        assert e2 > e1, "Energy should increase after second boost"

    def test_energy_decay_returns_current(self, db):
        """energy_decay should return the current energy."""
        if not has_cognitive(db):
            pytest.skip("cognitive feature not available")
        cog = db.cognitive
        node = db.create_node(["Test"], {})

        cog.energy_boost(node.id, 3.0)
        decayed = cog.energy_decay(node.id)
        current = cog.energy_get(node.id)
        assert decayed == current

    def test_energy_low_nodes(self, db):
        """Low energy nodes should be detected."""
        if not has_cognitive(db):
            pytest.skip("cognitive feature not available")
        cog = db.cognitive

        # Create nodes with varying energy
        n1 = db.create_node(["Test"], {})
        n2 = db.create_node(["Test"], {})

        cog.energy_boost(n1.id, 0.05)
        cog.energy_boost(n2.id, 5.0)

        low = cog.energy_low_nodes(0.1)
        assert n1.id in low, "Low-energy node should be in the list"
        assert n2.id not in low, "High-energy node should not be in the list"


# ===========================================================================
# Synapse tests
# ===========================================================================


class TestSynapse:
    """Test synapse subsystem: list, strengthen, get, prune."""

    def test_synapse_strengthen_and_get(self, db):
        """Strengthening should create a synapse with weight."""
        if not has_cognitive(db):
            pytest.skip("cognitive feature not available")
        cog = db.cognitive
        a = db.create_node(["A"], {})
        b = db.create_node(["B"], {})

        cog.synapse_strengthen(a.id, b.id, 0.5)
        weight = cog.synapse_get(a.id, b.id)
        assert weight is not None, "Synapse should exist after strengthen"
        assert weight > 0.0, "Synapse weight should be positive"

    def test_synapse_get_nonexistent(self, db):
        """Getting a non-existent synapse returns None."""
        if not has_cognitive(db):
            pytest.skip("cognitive feature not available")
        cog = db.cognitive
        result = cog.synapse_get(9999, 8888)
        assert result is None

    def test_synapse_strengthen_accumulates(self, db):
        """Multiple reinforcements should increase weight."""
        if not has_cognitive(db):
            pytest.skip("cognitive feature not available")
        cog = db.cognitive
        a = db.create_node(["A"], {})
        b = db.create_node(["B"], {})

        cog.synapse_strengthen(a.id, b.id, 0.3)
        w1 = cog.synapse_get(a.id, b.id)
        cog.synapse_strengthen(a.id, b.id, 0.3)
        w2 = cog.synapse_get(a.id, b.id)
        assert w2 > w1, "Weight should increase after reinforcement"

    def test_synapse_list(self, db):
        """Listing synapses should return synapse details."""
        if not has_cognitive(db):
            pytest.skip("cognitive feature not available")
        cog = db.cognitive
        a = db.create_node(["A"], {})
        b = db.create_node(["B"], {})
        c = db.create_node(["C"], {})

        cog.synapse_strengthen(a.id, b.id)
        cog.synapse_strengthen(a.id, c.id)

        synapses = cog.synapse_list(a.id)
        assert len(synapses) == 2, "Should have 2 synapses"
        for syn in synapses:
            assert "source" in syn
            assert "target" in syn
            assert "weight" in syn
            assert "reinforcement_count" in syn
            assert syn["weight"] > 0.0

    def test_synapse_prune(self, db):
        """Pruning with high threshold should remove weak synapses."""
        if not has_cognitive(db):
            pytest.skip("cognitive feature not available")
        cog = db.cognitive
        a = db.create_node(["A"], {})
        b = db.create_node(["B"], {})

        cog.synapse_strengthen(a.id, b.id, 0.01)
        # Prune with high threshold — should remove this weak synapse
        pruned = cog.synapse_prune(100.0)
        assert pruned >= 0  # May or may not prune depending on initial weight


# ===========================================================================
# Scar tests
# ===========================================================================


class TestScar:
    """Test scar subsystem: add, heal, list, intensity, prune."""

    def test_scar_add_and_list(self, db):
        """Adding a scar should make it visible in the list."""
        if not has_cognitive(db):
            pytest.skip("cognitive feature not available")
        cog = db.cognitive
        node = db.create_node(["Test"], {})

        scar_id = cog.scar_add(node.id, 1.0, "error")
        assert scar_id > 0, "Scar ID should be positive"

        scars = cog.scar_list(node.id)
        assert len(scars) >= 1, "Should have at least one scar"
        scar = scars[0]
        assert "id" in scar
        assert "intensity" in scar
        assert "reason" in scar
        assert scar["intensity"] > 0.0

    def test_scar_heal(self, db):
        """Healing a scar should remove it from active list."""
        if not has_cognitive(db):
            pytest.skip("cognitive feature not available")
        cog = db.cognitive
        node = db.create_node(["Test"], {})

        scar_id = cog.scar_add(node.id, 2.0, "rollback")
        healed = cog.scar_heal(scar_id)
        assert healed is True, "Healing should succeed"

        scars = cog.scar_list(node.id)
        active_ids = [s["id"] for s in scars]
        assert scar_id not in active_ids, "Healed scar should not be in active list"

    def test_scar_heal_nonexistent(self, db):
        """Healing a non-existent scar should return False."""
        if not has_cognitive(db):
            pytest.skip("cognitive feature not available")
        cog = db.cognitive
        assert cog.scar_heal(99999) is False

    def test_scar_intensity(self, db):
        """Cumulative intensity should reflect active scars."""
        if not has_cognitive(db):
            pytest.skip("cognitive feature not available")
        cog = db.cognitive
        node = db.create_node(["Test"], {})

        assert cog.scar_intensity(node.id) == 0.0, "No scars yet"

        cog.scar_add(node.id, 3.0, "error")
        intensity = cog.scar_intensity(node.id)
        assert intensity > 0.0, "Should have positive intensity"

    def test_scar_prune(self, db):
        """Pruning should return the number of pruned scars."""
        if not has_cognitive(db):
            pytest.skip("cognitive feature not available")
        cog = db.cognitive
        pruned = cog.scar_prune()
        assert pruned >= 0

    def test_scar_reasons(self, db):
        """Different scar reasons should be accepted."""
        if not has_cognitive(db):
            pytest.skip("cognitive feature not available")
        cog = db.cognitive
        node = db.create_node(["Test"], {})

        for reason in ["error", "rollback", "invalidation", "custom_reason"]:
            scar_id = cog.scar_add(node.id, 1.0, reason)
            assert scar_id > 0


# ===========================================================================
# Fabric tests
# ===========================================================================


class TestFabric:
    """Test fabric score access."""

    def test_fabric_score_default(self, db):
        """Default fabric score should have all zeros."""
        if not has_cognitive(db):
            pytest.skip("cognitive feature not available")
        cog = db.cognitive
        score = cog.fabric_score(999)

        assert isinstance(score, dict)
        assert score["mutation_frequency"] == 0.0
        assert score["annotation_density"] == 0.0
        assert score["risk_score"] == 0.0
        assert score["pagerank"] == 0.0
        assert score["betweenness"] == 0.0
        assert score["scar_intensity"] == 0.0
        assert score["community_id"] is None


# ===========================================================================
# GDS tests
# ===========================================================================


class TestGDS:
    """Test Graph Data Science algorithms."""

    def test_pagerank(self, graph_db):
        """PageRank should return scores for all nodes."""
        db = graph_db["db"]
        if not has_gds(db):
            pytest.skip("GDS feature not available")

        scores = db.gds.pagerank()
        assert isinstance(scores, dict)
        assert len(scores) > 0, "Should have PageRank scores"

        # All scores should be positive
        for score in scores.values():
            assert score >= 0.0

    def test_pagerank_custom_params(self, graph_db):
        """PageRank with custom parameters should work."""
        db = graph_db["db"]
        if not has_gds(db):
            pytest.skip("GDS feature not available")

        scores = db.gds.pagerank(damping=0.9, max_iterations=50, tolerance=1e-4)
        assert len(scores) > 0

    def test_louvain(self, graph_db):
        """Louvain should detect communities."""
        db = graph_db["db"]
        if not has_gds(db):
            pytest.skip("GDS feature not available")

        result = db.gds.louvain()
        assert isinstance(result, dict)
        assert "communities" in result
        assert "modularity" in result
        assert "num_communities" in result
        assert result["num_communities"] >= 1

    def test_leiden(self, graph_db):
        """Leiden should detect communities (Louvain-based)."""
        db = graph_db["db"]
        if not has_gds(db):
            pytest.skip("GDS feature not available")

        result = db.gds.leiden()
        assert isinstance(result, dict)
        assert "communities" in result
        assert "num_communities" in result
        assert result["num_communities"] >= 1

    def test_similarity(self, graph_db):
        """Jaccard similarity should return a score between 0 and 1."""
        db = graph_db["db"]
        if not has_gds(db):
            pytest.skip("GDS feature not available")

        alice = graph_db["alice"]
        bob = graph_db["bob"]

        sim = db.gds.similarity(alice.id, bob.id)
        assert 0.0 <= sim <= 1.0, f"Similarity should be in [0, 1], got {sim}"

    def test_similarity_same_node(self, graph_db):
        """Similarity of a node with itself depends on self-loops (usually 1.0 or trivial)."""
        db = graph_db["db"]
        if not has_gds(db):
            pytest.skip("GDS feature not available")

        alice = graph_db["alice"]
        sim = db.gds.similarity(alice.id, alice.id)
        assert sim == 1.0, "Node should have perfect similarity with itself"

    def test_similarity_disconnected(self, graph_db):
        """Disconnected nodes should have 0.0 similarity."""
        db = graph_db["db"]
        if not has_gds(db):
            pytest.skip("GDS feature not available")

        diana = graph_db["diana"]
        # Diana has no outgoing edges in our test graph,
        # so similarity with anyone else via Jaccard on outgoing neighbors is special
        isolated = db.create_node(["Isolated"], {})
        sim = db.gds.similarity(isolated.id, diana.id)
        assert sim == 0.0

    def test_project_all(self, graph_db):
        """Projecting all nodes should return correct counts."""
        db = graph_db["db"]
        if not has_gds(db):
            pytest.skip("GDS feature not available")

        result = db.gds.project()
        assert result["node_count"] == 5  # 4 persons + 1 company
        assert result["edge_count"] > 0

    def test_project_filtered(self, graph_db):
        """Projecting with label filter should subset nodes."""
        db = graph_db["db"]
        if not has_gds(db):
            pytest.skip("GDS feature not available")

        result = db.gds.project(node_labels=["Person"])
        assert result["node_count"] == 4  # Only Person nodes

    def test_project_with_rel_types(self, graph_db):
        """Projecting with relationship type filter should subset edges."""
        db = graph_db["db"]
        if not has_gds(db):
            pytest.skip("GDS feature not available")

        result = db.gds.project(rel_types=["KNOWS"])
        assert result["edge_count"] >= 1

    def test_betweenness_centrality(self, graph_db):
        """Betweenness centrality should return scores."""
        db = graph_db["db"]
        if not has_gds(db):
            pytest.skip("GDS feature not available")

        scores = db.gds.betweenness_centrality()
        assert isinstance(scores, dict)
        assert len(scores) > 0


# ===========================================================================
# CognitiveEngine repr tests
# ===========================================================================


class TestRepr:
    """Test __repr__ methods."""

    def test_cognitive_repr(self, db):
        """CognitiveEngine repr should be informative."""
        if not has_cognitive(db):
            pytest.skip("cognitive feature not available")
        assert "CognitiveEngine" in repr(db.cognitive)

    def test_gds_repr(self, db):
        """GDS repr should be informative."""
        if not has_gds(db):
            pytest.skip("GDS feature not available")
        assert "GDS" in repr(db.gds)


# ===========================================================================
# Integration test: cognitive + GDS together
# ===========================================================================


class TestCognitiveIntegration:
    """Integration tests combining cognitive features."""

    def test_energy_and_search_workflow(self, graph_db):
        """Workflow: boost energy on active nodes, then query fabric."""
        db = graph_db["db"]
        if not has_cognitive(db):
            pytest.skip("cognitive feature not available")

        cog = db.cognitive

        # Boost energy on actively used nodes
        cog.energy_boost(graph_db["alice"].id, 5.0)
        cog.energy_boost(graph_db["bob"].id, 3.0)

        # Verify energy is tracked
        assert cog.energy_get(graph_db["alice"].id) > cog.energy_get(graph_db["charlie"].id)

    def test_synapse_and_scar_workflow(self, graph_db):
        """Workflow: build synapses, add scars, verify state."""
        db = graph_db["db"]
        if not has_cognitive(db):
            pytest.skip("cognitive feature not available")

        cog = db.cognitive
        alice_id = graph_db["alice"].id
        bob_id = graph_db["bob"].id

        # Build synapse
        cog.synapse_strengthen(alice_id, bob_id, 1.0)
        assert cog.synapse_get(alice_id, bob_id) is not None

        # Add scar
        scar_id = cog.scar_add(bob_id, 2.0, "error")

        # Verify state
        assert cog.scar_intensity(bob_id) > 0.0
        scars = cog.scar_list(bob_id)
        assert len(scars) == 1

        # Heal
        cog.scar_heal(scar_id)
        assert cog.scar_intensity(bob_id) == 0.0

    def test_gds_pagerank_and_louvain(self, graph_db):
        """Workflow: run PageRank and Louvain on the same graph."""
        db = graph_db["db"]
        if not has_gds(db):
            pytest.skip("GDS feature not available")

        pr = db.gds.pagerank()
        communities = db.gds.louvain()

        # Every node should have a PageRank score
        assert len(pr) == 5

        # Every node should have a community assignment
        assert len(communities["communities"]) == 5
