"""GraphQL on RDF pytest fixtures and configuration."""

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
def rdf_graphql_db(db):
    """Create a database with RDF data for GraphQL queries."""
    # Create resources with URIs
    alix = db.create_node(
        ["Resource", "Person"],
        {"uri": "http://example.org/person/alix", "name": "Alix", "age": 30},
    )

    gus = db.create_node(
        ["Resource", "Person"],
        {"uri": "http://example.org/person/gus", "name": "Gus", "age": 25},
    )

    # Create knows relationship
    db.create_edge(alix.id, gus.id, "knows", {})

    return db
