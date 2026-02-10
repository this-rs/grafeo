# grafeo

Python bindings for [Grafeo](https://grafeo.dev), a high-performance, embeddable graph database with a Rust core.

## Installation

```bash
pip install grafeo
# or
uv add grafeo
```

## Quick Start

```python
from grafeo import GrafeoDB

# In-memory database
db = GrafeoDB()

# Or persistent
# db = GrafeoDB("./my-graph")

# Create nodes
db.execute("INSERT (:Person {name: 'Alice', age: 30})")
db.execute("INSERT (:Person {name: 'Bob', age: 25})")
db.execute("INSERT (:Person {name: 'Alice'})-[:KNOWS]->(:Person {name: 'Bob'})")

# Query the graph
result = db.execute("MATCH (p:Person) WHERE p.age > 20 RETURN p.name, p.age")
for row in result:
    print(row)
```

## API Overview

### Database

```python
db = GrafeoDB()              # in-memory
db = GrafeoDB("./path")      # persistent
db = GrafeoDB.open("./path") # open existing

db.node_count   # number of nodes
db.edge_count   # number of edges
```

### Query Languages

```python
result = db.execute(gql)                        # GQL (ISO standard)
result = db.execute_with_params(gql, params)    # GQL with parameters
result = db.execute_cypher(query)               # Cypher
result = db.execute_sparql(query)               # SPARQL
result = db.execute_gremlin(query)              # Gremlin
result = db.execute_graphql(query)              # GraphQL
```

### Node & Edge CRUD

```python
node = db.create_node(["Person"], {"name": "Alice", "age": 30})
edge = db.create_edge(source_id, target_id, "KNOWS", {"since": 2024})

n = db.get_node(node_id)   # Node or None
e = db.get_edge(edge_id)   # Edge or None

db.set_node_property(node_id, "key", "value")
db.set_edge_property(edge_id, "key", "value")

db.delete_node(node_id)
db.delete_edge(edge_id)
```

### Transactions

```python
# Context manager (auto-rollback on exception)
with db.begin_transaction() as tx:
    tx.execute("INSERT (:Person {name: 'Carol'})")
    tx.commit()

# With isolation levels
from grafeo import IsolationLevel
with db.begin_transaction(IsolationLevel.SERIALIZABLE) as tx:
    tx.execute("MATCH (n:Person) SET n.verified = true")
    tx.commit()
```

### QueryResult

```python
result = db.execute("MATCH (n:Person) RETURN n.name, n.age")

result.columns          # column names
len(result)             # row count
result.execution_time   # execution time (seconds)

for row in result:      # iterate rows
    print(row)

result[0]               # access by index
result.scalar()         # first column of first row
```

### Vector Search

```python
# Create an HNSW index
db.create_vector_index("Document", "embedding", dimensions=384)

# Insert vectors
node = db.create_node(["Document"], {"embedding": [0.1, 0.2, ...]})

# Search
results = db.vector_search("Document", "embedding", query_vector, k=10)
```

## Features

- GQL, Cypher, SPARQL, Gremlin, and GraphQL query languages
- Full node/edge CRUD with native Python types
- ACID transactions with configurable isolation levels
- HNSW vector similarity search
- Property indexes for fast lookups
- Async support via `asyncio`
- Type stubs included

## Links

- [Documentation](https://grafeo.dev)
- [GitHub](https://github.com/GrafeoDB/grafeo)
- [npm Package](https://www.npmjs.com/package/@grafeo-db/js)
- [WASM Package](https://www.npmjs.com/package/@grafeo-db/wasm)

## License

Apache-2.0
