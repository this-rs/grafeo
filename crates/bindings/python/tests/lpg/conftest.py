"""LPG model pytest fixtures and configuration."""

import pytest

# Try to import obrain
try:
    from obrain import ObrainDB

    OBRAIN_AVAILABLE = True
except ImportError:
    OBRAIN_AVAILABLE = False


@pytest.fixture
def db():
    """Create a fresh in-memory ObrainDB instance."""
    if not OBRAIN_AVAILABLE:
        pytest.skip("obrain not installed")
    return ObrainDB()


@pytest.fixture
def lpg_social_graph(db):
    """Create a social network graph for LPG tests."""
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


@pytest.fixture
def pattern_db(db):
    """Create a database with pattern test data.

    Creates Person and Company nodes with KNOWS and WORKS_AT edges.
    Returns the database instance (not the nodes).
    """
    alix = db.create_node(["Person"], {"name": "Alix", "age": 30, "city": "NYC"})
    gus = db.create_node(["Person"], {"name": "Gus", "age": 25, "city": "LA"})
    vincent = db.create_node(["Person"], {"name": "Vincent", "age": 35, "city": "NYC"})

    acme = db.create_node(["Company"], {"name": "Acme Corp", "founded": 2010})
    globex = db.create_node(["Company"], {"name": "Globex Inc", "founded": 2015})

    db.create_edge(alix.id, gus.id, "KNOWS", {"since": 2020})
    db.create_edge(gus.id, vincent.id, "KNOWS", {"since": 2021})
    db.create_edge(alix.id, vincent.id, "KNOWS", {"since": 2019})

    db.create_edge(alix.id, acme.id, "WORKS_AT", {"role": "Engineer"})
    db.create_edge(gus.id, globex.id, "WORKS_AT", {"role": "Manager"})
    db.create_edge(vincent.id, acme.id, "WORKS_AT", {"role": "Director"})

    return db
