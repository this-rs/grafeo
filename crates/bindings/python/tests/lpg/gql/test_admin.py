"""GQL-specific admin API tests."""

import pytest

from tests.bases.test_admin import BaseAdminTest

# Try to import obrain
try:
    from obrain import ObrainDB

    OBRAIN_AVAILABLE = True
except ImportError:
    OBRAIN_AVAILABLE = False


class TestGqlAdmin(BaseAdminTest):
    """GQL admin API tests."""

    @pytest.fixture
    def db(self):
        """Create a fresh in-memory ObrainDB instance."""
        if not OBRAIN_AVAILABLE:
            pytest.skip("obrain not installed")
        return ObrainDB()

    def setup_test_graph(self, db):
        """Set up test data for admin tests."""
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
