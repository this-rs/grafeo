---
title: SQL/PGQ Query Language
description: Query graphs using standard SQL:2023 GRAPH_TABLE syntax.
---

# SQL/PGQ Query Language

SQL/PGQ (SQL:2023, ISO/IEC 9075-16) brings graph pattern matching to standard SQL. Write `SELECT ... FROM GRAPH_TABLE (MATCH ...)` and query your graph without leaving SQL.

## Overview

SQL/PGQ lets SQL developers query graphs using familiar syntax. The inner `MATCH` clause uses GQL pattern syntax, and `COLUMNS` maps graph results to SQL columns.

```sql
SELECT *
FROM GRAPH_TABLE (
    MATCH (a:Person)-[e:KNOWS]->(b:Person)
    COLUMNS (a.name AS person, e.since AS year, b.name AS friend)
) result
WHERE result.person = 'Alice'
ORDER BY result.year DESC
LIMIT 10;
```

## Basic Syntax

### GRAPH_TABLE

The `GRAPH_TABLE` function wraps a graph pattern match inside a SQL `FROM` clause:

```sql
SELECT columns
FROM GRAPH_TABLE (
    MATCH pattern
    COLUMNS (column_list)
) alias
```

### Pattern Matching

The `MATCH` clause uses GQL-style patterns:

```sql
-- Node pattern
MATCH (p:Person)

-- Edge pattern
MATCH (a:Person)-[:KNOWS]->(b:Person)

-- Multi-hop
MATCH (a:Person)-[:KNOWS]->(b)-[:KNOWS]->(c)
```

### COLUMNS Clause

Map graph properties to SQL columns:

```sql
COLUMNS (
    a.name AS person_name,
    b.name AS friend_name,
    e.since AS year
)
```

## Examples

### Find friends of a person

```sql
SELECT *
FROM GRAPH_TABLE (
    MATCH (p:Person {name: 'Alice'})-[:KNOWS]->(f:Person)
    COLUMNS (f.name AS friend, f.age AS age)
);
```

### Friends of friends

```sql
SELECT DISTINCT result.fof_name
FROM GRAPH_TABLE (
    MATCH (me:Person {name: 'Alice'})-[:KNOWS]->()-[:KNOWS]->(fof:Person)
    COLUMNS (fof.name AS fof_name)
) result
WHERE result.fof_name <> 'Alice';
```

### Path functions

```sql
SELECT *
FROM GRAPH_TABLE (
    MATCH path = (a:Person)-[:KNOWS*1..3]->(b:Person)
    COLUMNS (
        a.name AS source,
        b.name AS target,
        LENGTH(path) AS hops
    )
);
```

## Using SQL/PGQ

=== "Python"

    ```python
    result = db.execute_sql("""
        SELECT * FROM GRAPH_TABLE (
            MATCH (p:Person)-[:KNOWS]->(f:Person)
            COLUMNS (p.name AS person, f.name AS friend)
        )
    """)
    ```

=== "Node.js"

    ```javascript
    const result = await db.executeSql(`
        SELECT * FROM GRAPH_TABLE (
            MATCH (p:Person)-[:KNOWS]->(f:Person)
            COLUMNS (p.name AS person, f.name AS friend)
        )
    `);
    ```

=== "Rust"

    ```rust
    let result = session.execute_sql(r#"
        SELECT * FROM GRAPH_TABLE (
            MATCH (p:Person)-[:KNOWS]->(f:Person)
            COLUMNS (p.name AS person, f.name AS friend)
        )
    "#)?;
    ```

=== "Go"

    ```go
    result, err := db.ExecuteSQL(`
        SELECT * FROM GRAPH_TABLE (
            MATCH (p:Person)-[:KNOWS]->(f:Person)
            COLUMNS (p.name AS person, f.name AS friend)
        )
    `)
    ```

## When to Use SQL/PGQ

**Use SQL/PGQ when:**

- Your team already knows SQL
- You want to integrate graph queries into existing SQL workflows
- You need standard SQL features (WHERE, ORDER BY, LIMIT, GROUP BY) around graph patterns

**Use GQL directly when:**

- You want the full power of GQL (INSERT, SET, DELETE, REMOVE)
- You're doing graph-only work without SQL wrapping
