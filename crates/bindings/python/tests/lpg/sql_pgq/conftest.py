"""SQL/PGQ-specific pytest fixtures and configuration."""

import pytest

# Try to import obrain
try:
    from obrain import ObrainDB

    OBRAIN_AVAILABLE = True
except ImportError:
    OBRAIN_AVAILABLE = False


def has_sql_pgq_support(db):
    """Check if SQL/PGQ support is available."""
    try:
        db.execute_sql("SELECT * FROM GRAPH_TABLE (MATCH (n) COLUMNS (n.name AS name))")
        return True
    except (AttributeError, NotImplementedError):
        return False
    except Exception:
        return True  # Has method but might fail for other reasons


@pytest.fixture
def db():
    """Create a fresh in-memory ObrainDB instance.

    This fixture requires SQL/PGQ support.
    """
    if not OBRAIN_AVAILABLE:
        pytest.skip("obrain not installed")
    database = ObrainDB()
    if not has_sql_pgq_support(database):
        pytest.skip("SQL/PGQ support not available in this build")
    return database


@pytest.fixture
def social_db(db):
    """Create a database with a social network for SQL/PGQ tests.

    Structure:
    - Alix (Person, age 30, city NYC) -KNOWS-> Gus (since 2020)
    - Gus (Person, age 25, city LA) -KNOWS-> Vincent (since 2021)
    - Alix -KNOWS-> Vincent (since 2019)
    - Vincent (Person, age 35, city NYC)
    - Acme Corp (Company, founded 2010)
    - Globex Inc (Company, founded 2015)
    - Alix -WORKS_AT-> Acme Corp (role Engineer)
    - Gus -WORKS_AT-> Globex Inc (role Manager)
    - Vincent -WORKS_AT-> Acme Corp (role Director)
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


@pytest.fixture
def chain_db(db):
    """Create a chain graph for path tests: A -> B -> C -> D.

    All nodes are Person type with KNOWS edges.
    """
    a = db.create_node(["Person"], {"name": "A"})
    b = db.create_node(["Person"], {"name": "B"})
    c = db.create_node(["Person"], {"name": "C"})
    d = db.create_node(["Person"], {"name": "D"})

    db.create_edge(a.id, b.id, "KNOWS", {})
    db.create_edge(b.id, c.id, "KNOWS", {})
    db.create_edge(c.id, d.id, "KNOWS", {})

    return db
