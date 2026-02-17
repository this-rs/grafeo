---
title: Mutations
description: Creating, updating and deleting graph data in Cypher.
tags:
  - cypher
  - mutations
---

# Mutations

Cypher supports mutations for creating, updating and deleting graph data.

## Creating Nodes

```cypher
-- Create a node
CREATE (p:Person {name: 'Alice', age: 30})
RETURN p

-- Create multiple nodes
CREATE (a:Person {name: 'Alice'})
CREATE (b:Person {name: 'Bob'})

-- Create with multiple labels
CREATE (e:Person:Employee {name: 'Carol'})
```

## Creating Relationships

```cypher
-- Create a relationship between existing nodes
MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})
CREATE (a)-[:KNOWS]->(b)

-- Create relationship with properties
MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})
CREATE (a)-[:KNOWS {since: 2020, strength: 'close'}]->(b)

-- Create nodes and relationships together
CREATE (a:Person {name: 'Alice'})-[:KNOWS]->(b:Person {name: 'Bob'})
RETURN a, b
```

## Updating Properties

```cypher
-- Set a property
MATCH (p:Person {name: 'Alice'})
SET p.age = 31

-- Set multiple properties
MATCH (p:Person {name: 'Alice'})
SET p.age = 31, p.city = 'New York'

-- Set from another property
MATCH (p:Person)
SET p.displayName = p.firstName + ' ' + p.lastName

-- Replace all properties
MATCH (p:Person {name: 'Alice'})
SET p = {name: 'Alice', age: 31, city: 'NYC'}

-- Add to existing properties
MATCH (p:Person {name: 'Alice'})
SET p += {city: 'NYC', active: true}
```

## Removing Properties

```cypher
-- Remove a property
MATCH (p:Person {name: 'Alice'})
REMOVE p.temporaryField

-- Set to null (equivalent)
MATCH (p:Person {name: 'Alice'})
SET p.temporaryField = null
```

## Deleting Nodes

```cypher
-- Delete a node (must have no relationships)
MATCH (p:Person {name: 'Alice'})
DELETE p

-- Delete node and all its relationships
MATCH (p:Person {name: 'Alice'})
DETACH DELETE p
```

## Deleting Relationships

```cypher
-- Delete specific relationship
MATCH (a:Person {name: 'Alice'})-[r:KNOWS]->(b:Person {name: 'Bob'})
DELETE r

-- Delete all relationships of a type from a node
MATCH (p:Person {name: 'Alice'})-[r:KNOWS]->()
DELETE r
```

## UNWIND (List Expansion)

Expand a list into individual rows. Useful for batch operations.

```cypher
-- Unwind a literal list
UNWIND [1, 2, 3] AS x
RETURN x

-- Unwind with parameters (Python: db.execute_cypher(query, {'names': ['Alice', 'Bob']}))
UNWIND $names AS name
RETURN name

-- Batch create relationships from a parameter list
UNWIND $edges AS e
MATCH (a:Person {name: e.from}), (b:Person {name: e.to})
CREATE (a)-[:KNOWS]->(b)
```

## Merge (Upsert)

```cypher
-- Create if not exists, match if exists
MERGE (p:Person {email: 'alice@example.com'})
SET p.lastSeen = timestamp()
RETURN p

-- Merge with different actions
MERGE (p:Person {email: 'alice@example.com'})
ON CREATE SET p.created = timestamp()
ON MATCH SET p.lastSeen = timestamp()
RETURN p

-- Merge relationships
MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})
MERGE (a)-[:KNOWS]->(b)
```
