"""GQL implementation of advanced query tests."""

import pytest

from tests.bases.test_advanced_queries import BaseAdvancedQueriesTest

# Try to import obrain
try:
    from obrain import ObrainDB

    OBRAIN_AVAILABLE = True
except ImportError:
    OBRAIN_AVAILABLE = False


@pytest.fixture
def db():
    if not OBRAIN_AVAILABLE:
        pytest.skip("obrain not installed")
    return ObrainDB()


class TestGQLAdvancedQueries(BaseAdvancedQueriesTest):
    def setup_social_graph(self, db):
        alix = db.create_node(["Person"], {"name": "Alix", "age": 30, "city": "NYC"})
        gus = db.create_node(["Person"], {"name": "Gus", "age": 25, "city": "LA"})
        harm = db.create_node(["Person"], {"name": "Harm", "age": 35, "city": "London"})
        db.create_edge(alix.id, gus.id, "KNOWS")
        db.create_edge(alix.id, harm.id, "KNOWS")
        db.create_edge(gus.id, harm.id, "KNOWS")
