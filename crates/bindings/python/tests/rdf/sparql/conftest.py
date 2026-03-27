"""SPARQL-specific pytest fixtures and configuration."""

import pytest

# Try to import obrain
try:
    from obrain import ObrainDB

    OBRAIN_AVAILABLE = True
except ImportError:
    OBRAIN_AVAILABLE = False


def has_sparql_support(db):
    """Check if SPARQL support is available."""
    try:
        db.execute_sparql("SELECT * WHERE { ?s ?p ?o } LIMIT 1")
        return True
    except (AttributeError, NotImplementedError):
        return False
    except Exception:
        return True  # Has method but might fail for other reasons


@pytest.fixture
def db():
    """Create a fresh in-memory ObrainDB instance."""
    if not OBRAIN_AVAILABLE:
        pytest.skip("obrain not installed")
    db = ObrainDB()
    if not has_sparql_support(db):
        pytest.skip("SPARQL support not available in this build")
    return db


@pytest.fixture
def db_api():
    """Create a fresh in-memory ObrainDB instance for Python API tests.
    This fixture does NOT require SPARQL support."""
    if not OBRAIN_AVAILABLE:
        pytest.skip("obrain not installed")
    return ObrainDB()


@pytest.fixture
def sparql_db(db):
    """Create a database with RDF test data for SPARQL queries."""
    # Create resources representing triples
    alix = db.create_node(
        ["Resource"],
        {
            "uri": "http://example.org/person/alix",
            "rdf:type": "http://xmlns.com/foaf/0.1/Person",
            "foaf:name": "Alix",
            "foaf:age": 30,
        },
    )

    gus = db.create_node(
        ["Resource"],
        {
            "uri": "http://example.org/person/gus",
            "rdf:type": "http://xmlns.com/foaf/0.1/Person",
            "foaf:name": "Gus",
            "foaf:age": 25,
        },
    )

    vincent = db.create_node(
        ["Resource"],
        {
            "uri": "http://example.org/person/vincent",
            "rdf:type": "http://xmlns.com/foaf/0.1/Person",
            "foaf:name": "Vincent",
            "foaf:age": 35,
        },
    )

    # Create foaf:knows relationships
    db.create_edge(alix.id, gus.id, "foaf:knows", {})
    db.create_edge(gus.id, vincent.id, "foaf:knows", {})

    return db
