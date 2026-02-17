"""GQL change data capture (CDC) integration tests."""

import pytest

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


class TestCDC:
    def test_node_history_after_create(self, db):
        node = db.create_node(["Person"], {"name": "Alice"})
        history = db.node_history(node.id)
        assert len(history) >= 1

    def test_node_history_after_update(self, db):
        node = db.create_node(["Person"], {"name": "Alice"})
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
        db.create_node(["Person"], {"name": "Alice"})
        db.create_node(["Person"], {"name": "Bob"})
        # Get all changes from epoch 0 to a large number
        changes = db.changes_between(0, 1000)
        assert len(changes) >= 2

    def test_empty_history(self, db):
        # Node ID that doesn't exist
        history = db.node_history(9999)
        assert len(history) == 0
