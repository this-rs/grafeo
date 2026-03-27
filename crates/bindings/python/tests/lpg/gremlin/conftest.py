"""Gremlin-specific pytest fixtures and configuration."""

import pytest

# Try to import obrain
try:
    from obrain import ObrainDB

    OBRAIN_AVAILABLE = True
except ImportError:
    OBRAIN_AVAILABLE = False


def has_gremlin_support(db):
    """Check if Gremlin support is available."""
    try:
        db.execute_gremlin("g.V().limit(1)")
        return True
    except (AttributeError, NotImplementedError):
        return False
    except Exception:
        return True  # Has method but might fail for other reasons


@pytest.fixture
def db():
    """Create a fresh in-memory ObrainDB instance.

    This fixture requires Gremlin support.
    For tests using only Python API, use db_api fixture instead.
    """
    if not OBRAIN_AVAILABLE:
        pytest.skip("obrain not installed")
    database = ObrainDB()
    if not has_gremlin_support(database):
        pytest.skip("Gremlin support not available in this build")
    return database


@pytest.fixture
def db_api():
    """Create a fresh in-memory ObrainDB instance for Python API tests.

    This fixture does NOT require Gremlin support - use for tests
    that only use db.create_node(), db.algorithms, etc.
    """
    if not OBRAIN_AVAILABLE:
        pytest.skip("obrain not installed")
    return ObrainDB()


@pytest.fixture
def traversal_db(db):
    """Create a database with traversal test data."""
    # Create Person nodes
    alix = db.create_node(["Person"], {"name": "Alix", "age": 30})
    gus = db.create_node(["Person"], {"name": "Gus", "age": 25})
    vincent = db.create_node(["Person"], {"name": "Vincent", "age": 35})

    # Create knows edges (lowercase for Gremlin convention)
    db.create_edge(alix.id, gus.id, "knows", {"since": 2020})
    db.create_edge(gus.id, vincent.id, "knows", {"since": 2021})

    return db
