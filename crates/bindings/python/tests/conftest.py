"""Root pytest fixtures for Obrain Python tests.

This module provides common fixtures available to all test modules.
"""

import sys
from pathlib import Path

# Add the python binding root to sys.path so imports like 'tests.bases' work
python_binding_root = Path(__file__).parent.parent
if str(python_binding_root) not in sys.path:
    sys.path.insert(0, str(python_binding_root))

import random  # noqa: E402

import pytest  # noqa: E402

# Try to import obrain
try:
    import obrain

    OBRAIN_AVAILABLE = True
except ImportError:
    OBRAIN_AVAILABLE = False


# Import fixtures from fixtures module
try:
    from tests.fixtures.datasets import (  # noqa: E402
        create_social_graph,
    )
    from tests.fixtures.generators import (  # noqa: E402
        SocialNetworkGenerator,
        load_data_into_db,
    )
except ImportError:
    # Fallback if relative imports don't work
    pass


@pytest.fixture
def db():
    """Create a fresh in-memory ObrainDB instance."""
    if not OBRAIN_AVAILABLE:
        pytest.skip("obrain not installed")
    return obrain.ObrainDB()


@pytest.fixture
def node_ids(db):
    """Create a test graph and return node IDs.

    Creates a random graph with 100 nodes and 300 edges for algorithm testing.
    """
    n_nodes = 100
    n_edges = 300

    node_ids = []
    for i in range(n_nodes):
        node = db.create_node(["Node"], {"index": i})
        node_ids.append(node.id)

    edges = set()
    while len(edges) < n_edges:
        src = random.choice(node_ids)
        dst = random.choice(node_ids)
        if src != dst and (src, dst) not in edges:
            db.create_edge(src, dst, "EDGE", {"weight": random.uniform(0.1, 10.0)})
            edges.add((src, dst))

    return node_ids


@pytest.fixture
def social_graph(db):
    """Create a social network graph.

    Creates 50 Person nodes with KNOWS edges.
    """
    if not OBRAIN_AVAILABLE:
        pytest.skip("obrain not installed")

    try:
        return create_social_graph(db, size=50)
    except NameError:
        # Fallback implementation
        gen = SocialNetworkGenerator(num_nodes=50, avg_edges_per_node=5, seed=42)
        node_count, edge_count = load_data_into_db(db, gen)
        return {"node_count": node_count, "edge_count": edge_count}


@pytest.fixture
def pattern_graph(db):
    """Create a graph for pattern matching tests.

    Creates Person and Company nodes with various relationships.
    """
    if not OBRAIN_AVAILABLE:
        pytest.skip("obrain not installed")

    # Create Person nodes
    alix = db.create_node(["Person"], {"name": "Alix", "age": 30, "city": "NYC"})
    gus = db.create_node(["Person"], {"name": "Gus", "age": 25, "city": "LA"})
    vincent = db.create_node(["Person"], {"name": "Vincent", "age": 35, "city": "NYC"})

    # Create Company nodes
    acme = db.create_node(["Company"], {"name": "Acme Corp", "founded": 2010})
    globex = db.create_node(["Company"], {"name": "Globex Inc", "founded": 2015})

    # Create KNOWS edges
    db.create_edge(alix.id, gus.id, "KNOWS", {"since": 2020})
    db.create_edge(gus.id, vincent.id, "KNOWS", {"since": 2021})
    db.create_edge(alix.id, vincent.id, "KNOWS", {"since": 2019})

    # Create WORKS_AT edges
    db.create_edge(alix.id, acme.id, "WORKS_AT", {"role": "Engineer"})
    db.create_edge(gus.id, globex.id, "WORKS_AT", {"role": "Manager"})
    db.create_edge(vincent.id, acme.id, "WORKS_AT", {"role": "Director"})

    return {
        "alix": alix,
        "gus": gus,
        "vincent": vincent,
        "acme": acme,
        "globex": globex,
    }


# Register markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "benchmark: marks tests as benchmarks")
    config.addinivalue_line("markers", "gql: marks tests requiring GQL support")
    config.addinivalue_line("markers", "cypher: marks tests requiring Cypher support")
    config.addinivalue_line("markers", "gremlin: marks tests requiring Gremlin support")
    config.addinivalue_line("markers", "graphql: marks tests requiring GraphQL support")
    config.addinivalue_line("markers", "sparql: marks tests requiring SPARQL support")
