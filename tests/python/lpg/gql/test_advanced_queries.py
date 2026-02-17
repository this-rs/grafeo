"""GQL implementation of advanced query tests."""

import pytest
from tests.python.bases.test_advanced_queries import BaseAdvancedQueriesTest

# Try to import grafeo
try:
    from grafeo import GrafeoDB

    GRAFEO_AVAILABLE = True
except ImportError:
    GRAFEO_AVAILABLE = False


@pytest.fixture
def db():
    if not GRAFEO_AVAILABLE:
        pytest.skip("grafeo not installed")
    return GrafeoDB()


class TestGQLAdvancedQueries(BaseAdvancedQueriesTest):
    def setup_social_graph(self, db):
        alice = db.create_node(["Person"], {"name": "Alice", "age": 30, "city": "NYC"})
        bob = db.create_node(["Person"], {"name": "Bob", "age": 25, "city": "LA"})
        carol = db.create_node(
            ["Person"], {"name": "Carol", "age": 35, "city": "London"}
        )
        db.create_edge(alice.id, bob.id, "KNOWS")
        db.create_edge(alice.id, carol.id, "KNOWS")
        db.create_edge(bob.id, carol.id, "KNOWS")
