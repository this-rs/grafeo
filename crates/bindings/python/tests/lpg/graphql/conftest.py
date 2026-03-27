"""GraphQL-specific pytest fixtures and configuration."""

import pytest

# Try to import obrain
try:
    from obrain import ObrainDB

    OBRAIN_AVAILABLE = True
except ImportError:
    OBRAIN_AVAILABLE = False


def has_graphql_support(db):
    """Check if GraphQL support is available."""
    try:
        db.execute_graphql("query { __schema { types { name } } }")
        return True
    except (AttributeError, NotImplementedError):
        return False
    except Exception:
        return True  # Has method but might fail for other reasons


@pytest.fixture
def db():
    """Create a fresh in-memory ObrainDB instance.

    This fixture requires GraphQL support.
    For tests using only Python API, use db_api fixture instead.
    """
    if not OBRAIN_AVAILABLE:
        pytest.skip("obrain not installed")
    database = ObrainDB()
    if not has_graphql_support(database):
        pytest.skip("GraphQL support not available in this build")
    return database


@pytest.fixture
def db_api():
    """Create a fresh in-memory ObrainDB instance for Python API tests.

    This fixture does NOT require GraphQL support - use for tests
    that only use db.create_node(), db.algorithms, etc.
    """
    if not OBRAIN_AVAILABLE:
        pytest.skip("obrain not installed")
    return ObrainDB()


@pytest.fixture
def graphql_db(db):
    """Create a database with GraphQL test data."""
    # Create User nodes
    alix = db.create_node(["User"], {"name": "Alix", "email": "alix@example.com", "age": 30})
    gus = db.create_node(["User"], {"name": "Gus", "email": "gus@example.com", "age": 25})

    # Create Post nodes
    post1 = db.create_node(["Post"], {"title": "Hello World", "content": "My first post"})

    # Create relationships
    db.create_edge(alix.id, gus.id, "friends", {})
    db.create_edge(alix.id, post1.id, "posts", {})

    return db
