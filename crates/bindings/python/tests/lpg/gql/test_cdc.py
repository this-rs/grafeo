"""GQL change data capture (CDC) integration tests."""

import pytest

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


class TestCDC:
    def test_node_history_after_create(self, db):
        node = db.create_node(["Person"], {"name": "Alix"})
        history = db.node_history(node.id)
        assert len(history) >= 1

    def test_node_history_after_update(self, db):
        node = db.create_node(["Person"], {"name": "Alix"})
        db.set_node_property(node.id, "age", 30)
        history = db.node_history(node.id)
        assert len(history) >= 2

    def test_edge_history_after_create(self, db):
        a = db.create_node(["N"])
        b = db.create_node(["N"])
        edge = db.create_edge(a.id, b.id, "R")
        history = db.edge_history(edge.id)
        assert len(history) >= 1

    def test_changes_between_epochs(self, db):
        db.create_node(["Person"], {"name": "Alix"})
        db.create_node(["Person"], {"name": "Gus"})
        # Get all changes from epoch 0 to a large number
        changes = db.changes_between(0, 1000)
        assert len(changes) >= 2

    def test_empty_history(self, db):
        # Node ID that doesn't exist
        history = db.node_history(9999)
        assert len(history) == 0
