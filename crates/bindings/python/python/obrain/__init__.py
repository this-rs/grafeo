"""
Obrain - A cognitive graph database.

This module provides Python bindings for the Obrain graph database,
offering a Pythonic interface for graph operations and GQL queries.

Example:
    >>> from obrain import ObrainDB
    >>> db = ObrainDB()
    >>> node = db.create_node(["Person"], {"name": "Alix", "age": 30})
    >>> result = db.execute("MATCH (n:Person) RETURN n")
    >>> for row in result:
    ...     print(row)
"""

from obrain.obrain import (
    Edge,
    ObrainDB,
    Node,
    QueryResult,
    Value,
    __version__,
    simd_support,
    vector,
)

__all__ = [
    "ObrainDB",
    "Node",
    "Edge",
    "QueryResult",
    "Value",
    "__version__",
    "simd_support",
    "vector",
]
