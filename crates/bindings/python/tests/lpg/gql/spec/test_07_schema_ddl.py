"""GQL spec: Schema / DDL (ISO sec 12).

Covers: CREATE/DROP GRAPH, CREATE/DROP GRAPH TYPE, CREATE/DROP NODE TYPE,
CREATE/DROP EDGE TYPE, CREATE/DROP INDEX, CREATE/DROP CONSTRAINT,
ALTER types, LIKE, AS COPY OF, OR REPLACE, IF NOT EXISTS, IF EXISTS,
USE GRAPH, CREATE SCHEMA, typed graphs, open graphs.
"""

import pytest

# =============================================================================
# Graph Management (sec 12.4, 12.5)
# =============================================================================


class TestGraphManagement:
    """CREATE GRAPH, DROP GRAPH, USE GRAPH."""

    def test_create_graph(self, db):
        """CREATE GRAPH name."""
        db.execute("CREATE GRAPH test_graph")
        # Should not raise

    def test_create_property_graph(self, db):
        """CREATE PROPERTY GRAPH name."""
        db.execute("CREATE PROPERTY GRAPH test_pg")

    def test_create_graph_if_not_exists(self, db):
        """CREATE GRAPH IF NOT EXISTS name."""
        db.execute("CREATE GRAPH IF NOT EXISTS g1")
        db.execute("CREATE GRAPH IF NOT EXISTS g1")  # No error on second call

    def test_drop_graph(self, db):
        """DROP GRAPH name."""
        db.execute("CREATE GRAPH to_drop")
        db.execute("DROP GRAPH to_drop")

    def test_drop_graph_if_exists(self, db):
        """DROP GRAPH IF EXISTS name (no error if missing)."""
        db.execute("DROP GRAPH IF EXISTS nonexistent_graph")

    def test_use_graph(self, db):
        """USE GRAPH name switches working graph."""
        db.execute("CREATE GRAPH my_graph")
        db.execute("USE GRAPH my_graph")

    def test_create_graph_or_replace(self, db):
        """OR REPLACE atomically replaces existing graph type."""
        db.execute("CREATE GRAPH TYPE gt1 AS { }")
        db.execute("CREATE OR REPLACE GRAPH TYPE gt1 AS { }")

    def test_like_graph(self, db):
        """CREATE GRAPH ... LIKE copies type from existing."""
        db.execute("CREATE GRAPH source_g")
        db.execute("CREATE GRAPH copy_g LIKE source_g")

    def test_open_graph(self, db):
        """CREATE GRAPH g ANY for open (untyped) graph."""
        db.execute("CREATE GRAPH open_g ANY")


# =============================================================================
# Graph Type (sec 12.6, 12.7)
# =============================================================================


class TestGraphType:
    """CREATE/DROP GRAPH TYPE."""

    def test_create_graph_type(self, db):
        """CREATE GRAPH TYPE with body."""
        db.execute("CREATE GRAPH TYPE social AS { }")

    def test_drop_graph_type(self, db):
        """DROP GRAPH TYPE."""
        db.execute("CREATE GRAPH TYPE temp_type AS { }")
        db.execute("DROP GRAPH TYPE temp_type")

    def test_drop_graph_type_if_exists(self, db):
        """DROP GRAPH TYPE IF EXISTS."""
        db.execute("DROP GRAPH TYPE IF EXISTS nonexistent_type")


# =============================================================================
# Schema (sec 12.2, 12.3)
# =============================================================================


class TestSchema:
    """CREATE/DROP SCHEMA."""

    def test_create_schema(self, db):
        """CREATE SCHEMA namespace registration."""
        db.execute("CREATE SCHEMA my_schema")

    def test_drop_schema(self, db):
        """DROP SCHEMA."""
        db.execute("CREATE SCHEMA temp_schema")
        db.execute("DROP SCHEMA temp_schema")

    def test_drop_schema_if_exists(self, db):
        """DROP SCHEMA IF EXISTS."""
        db.execute("DROP SCHEMA IF EXISTS nonexistent_schema")


# =============================================================================
# Node / Edge Types (sec 12)
# =============================================================================


class TestNodeEdgeTypes:
    """CREATE/DROP/ALTER NODE TYPE, EDGE TYPE."""

    def test_create_node_type(self, db):
        """CREATE NODE TYPE with properties."""
        db.execute("CREATE NODE TYPE Person (name STRING NOT NULL, age INT64)")

    def test_create_node_type_if_not_exists(self, db):
        """CREATE NODE TYPE IF NOT EXISTS."""
        db.execute("CREATE NODE TYPE IF NOT EXISTS Worker (name STRING)")
        db.execute("CREATE NODE TYPE IF NOT EXISTS Worker (name STRING)")

    def test_create_node_type_or_replace(self, db):
        """CREATE OR REPLACE NODE TYPE."""
        db.execute("CREATE NODE TYPE Temp1 (name STRING)")
        db.execute("CREATE OR REPLACE NODE TYPE Temp1 (name STRING, age INT64)")

    def test_drop_node_type(self, db):
        """DROP NODE TYPE."""
        db.execute("CREATE NODE TYPE ToDrop (name STRING)")
        db.execute("DROP NODE TYPE ToDrop")

    def test_drop_node_type_if_exists(self, db):
        """DROP NODE TYPE IF EXISTS."""
        db.execute("DROP NODE TYPE IF EXISTS NonExistent")

    def test_create_edge_type(self, db):
        """CREATE EDGE TYPE with properties."""
        db.execute("CREATE EDGE TYPE KNOWS (since INT64)")

    def test_create_edge_type_if_not_exists(self, db):
        """CREATE EDGE TYPE IF NOT EXISTS."""
        db.execute("CREATE EDGE TYPE IF NOT EXISTS FOLLOWS (since INT64)")
        db.execute("CREATE EDGE TYPE IF NOT EXISTS FOLLOWS (since INT64)")

    def test_drop_edge_type(self, db):
        """DROP EDGE TYPE."""
        db.execute("CREATE EDGE TYPE TempEdge (weight FLOAT64)")
        db.execute("DROP EDGE TYPE TempEdge")

    def test_drop_edge_type_if_exists(self, db):
        """DROP EDGE TYPE IF EXISTS."""
        db.execute("DROP EDGE TYPE IF EXISTS NonExistent")

    def test_alter_node_type_add_property(self, db):
        """ALTER NODE TYPE ADD property."""
        db.execute("CREATE NODE TYPE Mutable (name STRING)")
        db.execute("ALTER NODE TYPE Mutable ADD city STRING")

    def test_alter_node_type_drop_property(self, db):
        """ALTER NODE TYPE DROP property."""
        db.execute("CREATE NODE TYPE WithExtra (name STRING, temp STRING)")
        db.execute("ALTER NODE TYPE WithExtra DROP temp")

    def test_alter_edge_type_add_property(self, db):
        """ALTER EDGE TYPE ADD property."""
        db.execute("CREATE EDGE TYPE Evolving (since INT64)")
        db.execute("ALTER EDGE TYPE Evolving ADD weight FLOAT64")

    def test_alter_graph_type(self, db):
        """ALTER GRAPH TYPE ADD node/edge types."""
        db.execute("CREATE GRAPH TYPE Flexible AS { }")
        db.execute("ALTER GRAPH TYPE Flexible ADD NODE TYPE FlexNode (v INT64)")


# =============================================================================
# Index (sec 12)
# =============================================================================


class TestIndex:
    """CREATE/DROP INDEX."""

    @pytest.mark.xfail(reason="Index DDL uses Cypher-style syntax, not available in GQL parser")
    def test_create_index(self, db):
        """CREATE INDEX on property."""
        db.create_node(["Person"], {"name": "Alix"})
        db.execute("CREATE INDEX ON :Person(name)")

    @pytest.mark.xfail(reason="Index DDL uses Cypher-style syntax, not available in GQL parser")
    def test_create_index_if_not_exists(self, db):
        """CREATE INDEX IF NOT EXISTS."""
        db.create_node(["Person"], {"name": "Alix"})
        db.execute("CREATE INDEX IF NOT EXISTS ON :Person(name)")
        db.execute("CREATE INDEX IF NOT EXISTS ON :Person(name)")

    @pytest.mark.xfail(reason="Index DDL uses Cypher-style syntax, not available in GQL parser")
    def test_drop_index(self, db):
        """DROP INDEX."""
        db.create_node(["Person"], {"name": "Alix"})
        db.execute("CREATE INDEX idx_person_name ON :Person(name)")
        db.execute("DROP INDEX idx_person_name")

    def test_drop_index_if_exists(self, db):
        """DROP INDEX IF EXISTS."""
        db.execute("DROP INDEX IF EXISTS nonexistent_index")

    @pytest.mark.xfail(reason="Index DDL uses Cypher-style syntax, not available in GQL parser")
    def test_create_vector_index(self, db):
        """CREATE VECTOR INDEX with dimensions and metric."""
        db.create_node(["Doc"], {"embedding": [0.1, 0.2, 0.3]})
        db.execute(
            "CREATE VECTOR INDEX ON :Doc(embedding) OPTIONS {dimensions: 3, metric: 'cosine'}"
        )


# =============================================================================
# Constraint (sec 12)
# =============================================================================


class TestConstraint:
    """CREATE/DROP CONSTRAINT."""

    @pytest.mark.xfail(
        reason="Constraint DDL uses Cypher-style syntax, not available in GQL parser"
    )
    def test_create_unique_constraint(self, db):
        """CREATE CONSTRAINT UNIQUE."""
        db.execute("CREATE CONSTRAINT ON (n:Person) ASSERT n.email IS UNIQUE")

    @pytest.mark.xfail(
        reason="Constraint DDL uses Cypher-style syntax, not available in GQL parser"
    )
    def test_drop_constraint(self, db):
        """DROP CONSTRAINT."""
        db.execute("CREATE CONSTRAINT unique_email ON (n:Person) ASSERT n.email IS UNIQUE")
        db.execute("DROP CONSTRAINT unique_email")

    def test_drop_constraint_if_exists(self, db):
        """DROP CONSTRAINT IF EXISTS."""
        db.execute("DROP CONSTRAINT IF EXISTS nonexistent_constraint")
