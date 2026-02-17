---
title: Mutations
description: Creating, updating and deleting graph data in GQL.
tags:
  - gql
  - mutations
---

# Mutations

GQL supports mutations for creating, updating and deleting graph data.

## Creating Nodes

```sql
-- Create a node
INSERT (:Person {name: 'Alice', age: 30})

-- Create multiple nodes
INSERT (:Person {name: 'Alice'})
INSERT (:Person {name: 'Bob'})

-- Create with multiple labels
INSERT (:Person:Employee {name: 'Carol'})
```

## Creating Edges

```sql
-- Create an edge between existing nodes
MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})
INSERT (a)-[:KNOWS]->(b)

-- Create edge with properties
MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})
INSERT (a)-[:KNOWS {since: 2020, strength: 'close'}]->(b)
```

## Updating Properties

```sql
-- Set a property
MATCH (p:Person {name: 'Alice'})
SET p.age = 31

-- Set multiple properties
MATCH (p:Person {name: 'Alice'})
SET p.age = 31, p.city = 'New York'

-- Set from another property
MATCH (p:Person)
SET p.displayName = p.firstName + ' ' + p.lastName
```

## Removing Properties

```sql
-- Remove a property
MATCH (p:Person {name: 'Alice'})
REMOVE p.temporaryField

-- Set to null (equivalent)
MATCH (p:Person {name: 'Alice'})
SET p.temporaryField = null
```

## Deleting Nodes

```sql
-- Delete a node (must have no edges)
MATCH (p:Person {name: 'Alice'})
DELETE p

-- Delete node and all its edges
MATCH (p:Person {name: 'Alice'})
DETACH DELETE p
```

## Deleting Edges

```sql
-- Delete specific edge
MATCH (a:Person {name: 'Alice'})-[r:KNOWS]->(b:Person {name: 'Bob'})
DELETE r

-- Delete all edges of a type from a node
MATCH (p:Person {name: 'Alice'})-[r:KNOWS]->()
DELETE r
```

## UNWIND (List Expansion)

Expand a list into individual rows. Useful for batch operations.

```sql
-- Unwind a literal list
UNWIND [1, 2, 3] AS x
RETURN x

-- Unwind with parameters (Python: db.execute(query, {'names': ['Alice', 'Bob']}))
UNWIND $names AS name
RETURN name

-- Batch create edges from a parameter list
UNWIND $edges AS e
MATCH (a:Person {name: e.from}), (b:Person {name: e.to})
INSERT (a)-[:KNOWS]->(b)
```

## FOR (GQL Standard List Iteration)

The GQL standard equivalent of UNWIND (ISO/IEC 39075 section 14.8).

```sql
-- Iterate over a list
FOR x IN [4, 5, 6]
RETURN x

-- Batch insert nodes
FOR person IN $people
INSERT (:Person {name: person.name, age: person.age})
```

## Merge (Upsert)

```sql
-- Create if not exists, match if exists
MERGE (p:Person {email: 'alice@example.com'})
SET p.lastSeen = timestamp()
RETURN p

-- Merge with different actions
MERGE (p:Person {email: 'alice@example.com'})
ON CREATE SET p.created = timestamp()
ON MATCH SET p.lastSeen = timestamp()
RETURN p
```
