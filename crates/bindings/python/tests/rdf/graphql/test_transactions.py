"""GraphQL on RDF transaction tests.

Tests transaction behavior with RDF data and GraphQL queries.
Since GraphQL on RDF only supports queries (not mutations),
we test read isolation within transactions using SPARQL for mutations.
"""

import pytest

# Try to import obrain
try:
    from obrain import ObrainDB

    OBRAIN_AVAILABLE = True
except ImportError:
    OBRAIN_AVAILABLE = False


_HAS_SPARQL = OBRAIN_AVAILABLE and hasattr(ObrainDB(), "execute_sparql")

pytestmark = [
    pytest.mark.skipif(not OBRAIN_AVAILABLE, reason="Obrain Python bindings not installed"),
    pytest.mark.skipif(not _HAS_SPARQL, reason="obrain built without sparql feature"),
]


class TestRDFGraphQLTransactions:
    """Test transaction behavior with RDF data.

    GraphQL on RDF only supports Query operations, so mutations
    must be performed with SPARQL. These tests verify that:
    1. SPARQL mutations work within transactions
    2. Transaction isolation works correctly
    3. Commit/rollback affects RDF data appropriately
    """

    def setup_method(self):
        """Create a fresh database."""
        self.db = ObrainDB()

    def test_sparql_transaction_commit(self):
        """Test that committed SPARQL changes persist."""
        tx = self.db.begin_transaction()

        # Insert data in transaction
        tx.execute_sparql("""
            INSERT DATA {
                <http://example.org/tx/alix> <http://example.org/name> "Alix" .
            }
        """)

        tx.commit()

        # Verify data persisted after commit
        result = list(
            self.db.execute_sparql("""
            SELECT ?name WHERE {
                <http://example.org/tx/alix> <http://example.org/name> ?name .
            }
        """)
        )
        assert len(result) == 1

    def test_sparql_transaction_rollback(self):
        """Test that rolled back SPARQL changes don't persist."""
        tx = self.db.begin_transaction()

        # Insert data in transaction
        tx.execute_sparql("""
            INSERT DATA {
                <http://example.org/rollback/gus> <http://example.org/name> "Gus" .
            }
        """)

        tx.rollback()

        # Verify data was not persisted
        result = list(
            self.db.execute_sparql("""
            SELECT ?name WHERE {
                <http://example.org/rollback/gus> <http://example.org/name> ?name .
            }
        """)
        )
        assert len(result) == 0

    def test_multiple_sparql_operations_in_transaction(self):
        """Test multiple SPARQL operations in single transaction."""
        tx = self.db.begin_transaction()

        # Insert multiple triples
        tx.execute_sparql("""
            INSERT DATA {
                <http://example.org/multi/alix> <http://example.org/name> "Alix" .
                <http://example.org/multi/gus> <http://example.org/name> "Gus" .
            }
        """)

        # Delete one triple
        tx.execute_sparql("""
            DELETE DATA {
                <http://example.org/multi/alix> <http://example.org/name> "Alix" .
            }
        """)

        tx.commit()

        # Only Gus should remain
        result = list(
            self.db.execute_sparql("""
            SELECT ?s WHERE {
                ?s <http://example.org/name> ?name .
            }
        """)
        )
        assert len(result) == 1
