---
title: Database Operations
description: Creating and managing databases in Python.
tags:
  - python
  - database
---

# Database Operations

Learn how to create and configure Obrain databases in Python.

## Creating a Database

```python
import obrain

# In-memory database
db = obrain.ObrainDB()

# Persistent database
db = obrain.ObrainDB("my_graph.db")
```

## Executing Queries

```python
# Execute a query directly on the database
db.execute("INSERT (:Person {name: 'Alix', age: 30})")

# Query and iterate results
result = db.execute("MATCH (p:Person) RETURN p.name, p.age")
for row in result:
    print(row)
```

## Query Languages

Obrain supports multiple query languages:

```python
# GQL (default)
db.execute("MATCH (p:Person) RETURN p")

# Cypher
db.execute_cypher("MATCH (p:Person) RETURN p")

# Gremlin
db.execute_gremlin("g.V().hasLabel('Person')")

# GraphQL
db.execute_graphql("{ Person { name age } }")

# SPARQL (for RDF data)
db.execute_sparql("SELECT ?s ?p ?o WHERE { ?s ?p ?o }")
```

## Transactions

Use transactions for atomic operations:

```python
# Begin a transaction
with db.begin_transaction() as tx:
    tx.execute("INSERT (:Person {name: 'Alix'})")
    tx.execute("INSERT (:Person {name: 'Gus'})")
    tx.commit()  # Both inserts committed atomically

# Rollback on error
with db.begin_transaction() as tx:
    tx.execute("INSERT (:Person {name: 'Vincent'})")
    tx.rollback()  # Changes discarded
```

## Direct Node/Edge API

Create nodes and edges programmatically:

```python
# Create a node
node = db.create_node(["Person"], {"name": "Alix", "age": 30})
print(f"Created node with ID: {node.id}")

# Create an edge
db.execute("""
    MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'})
    INSERT (a)-[:KNOWS {since: 2024}]->(b)
""")
```

## Label Management

Manage node labels directly:

```python
# Add a label to a node
db.add_node_label(node.id, "Employee")

# Remove a label
db.remove_node_label(node.id, "Contractor")

# Get all labels for a node
labels = db.get_node_labels(node.id)
print(labels)  # ['Person', 'Employee']
```

## Admin APIs

Inspect and manage the database:

```python
# Database info
info = db.info()
print(f"Nodes: {info['node_count']}, Edges: {info['edge_count']}")

# Detailed statistics
stats = db.detailed_stats()

# Schema information
schema = db.schema()

# Validate database integrity
db.validate()
```

## Persistence

```python
# Save database to disk
db.save("backup.db")

# Create an in-memory copy
memory_db = db.to_memory()

# Load database as in-memory
db = obrain.ObrainDB.open_in_memory("my_graph.db")
```
