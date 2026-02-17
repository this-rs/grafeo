"""GQL text search integration tests."""

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


@pytest.fixture
def text_db(db):
    """Database with text-indexed articles."""
    db.create_node(["Article"], {"title": "Rust graph database engine"})
    db.create_node(["Article"], {"title": "Python machine learning"})
    db.create_node(["Article"], {"title": "Rust systems programming"})
    db.create_text_index("Article", "title")
    return db


class TestTextSearch:
    def test_text_search_basic(self, text_db):
        results = text_db.text_search("Article", "title", "Rust", k=10)
        assert len(results) >= 2

    def test_text_search_no_matches(self, text_db):
        results = text_db.text_search("Article", "title", "nonexistentxyz", k=10)
        assert len(results) == 0

    def test_text_search_after_mutation(self, text_db):
        text_db.create_node(["Article"], {"title": "Rust web framework"})
        results = text_db.text_search("Article", "title", "Rust", k=10)
        assert len(results) >= 3

    def test_drop_and_rebuild_text_index(self, text_db):
        # Search works
        r1 = text_db.text_search("Article", "title", "Rust", k=10)
        assert len(r1) > 0

        # Drop
        text_db.drop_text_index("Article", "title")

        # Rebuild
        text_db.rebuild_text_index("Article", "title")

        # Search works again
        r2 = text_db.text_search("Article", "title", "Rust", k=10)
        assert len(r2) > 0

    def test_text_search_no_index_error(self, db):
        db.create_node(["Article"], {"title": "test"})
        with pytest.raises(Exception):
            db.text_search("Article", "title", "test", k=10)
