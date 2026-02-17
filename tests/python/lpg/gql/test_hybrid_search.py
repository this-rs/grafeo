"""GQL hybrid search integration tests."""

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
def hybrid_db(db):
    """Database with both text and vector indexes."""
    db.create_node(
        ["Doc"],
        {"content": "Rust graph database engine", "emb": [1.0, 0.0, 0.0]},
    )
    db.create_node(
        ["Doc"],
        {"content": "Python machine learning", "emb": [0.0, 1.0, 0.0]},
    )
    db.create_node(
        ["Doc"],
        {"content": "Rust systems programming", "emb": [0.9, 0.1, 0.0]},
    )
    db.create_node(
        ["Doc"],
        {"content": "Graph neural network", "emb": [0.5, 0.5, 0.0]},
    )

    db.create_text_index("Doc", "content")
    db.create_vector_index("Doc", "emb", dimensions=3, metric="cosine")
    return db


class TestHybridSearch:
    def test_hybrid_search_basic(self, hybrid_db):
        results = hybrid_db.hybrid_search(
            "Doc",
            text_property="content",
            vector_property="emb",
            query_text="Rust graph",
            k=4,
            query_vector=[1.0, 0.0, 0.0],
        )
        assert len(results) > 0

    def test_hybrid_search_text_only(self, hybrid_db):
        results = hybrid_db.hybrid_search(
            "Doc",
            text_property="content",
            vector_property="emb",
            query_text="Rust",
            k=4,
        )
        assert len(results) > 0

    def test_hybrid_search_no_text_matches(self, hybrid_db):
        # Even with no text matches, vector search may contribute
        results = hybrid_db.hybrid_search(
            "Doc",
            text_property="content",
            vector_property="emb",
            query_text="nonexistentxyzquery",
            k=4,
            query_vector=[0.0, 0.0, 0.0],
        )
        # Should not error
        assert isinstance(results, list)
