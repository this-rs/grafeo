---
title: GQL Query Language
description: Learn the GQL query language for Grafeo.
---

# GQL Query Language

GQL (Graph Query Language) is the ISO standard for querying property graphs (ISO/IEC 39075). Grafeo implements GQL as its primary query language.

## Overview

GQL uses pattern matching to query and manipulate graph data. The syntax is very similar to Cypher.

## Quick Reference

| Operation | Syntax |
|-----------|--------|
| Match nodes | `MATCH (n:Label)` |
| Match edges | `MATCH (a)-[:TYPE]->(b)` |
| Filter | `WHERE n.property > value` |
| Return | `RETURN n.property` |
| Create | `INSERT (:Label {prop: value})` |
| Update | `SET n.property = value` |
| Delete | `DELETE n` |
| Unwind list | `UNWIND [1, 2, 3] AS x` |
| For loop | `FOR x IN $items` |
| Call procedure | `CALL grafeo.pagerank() YIELD score` |

## Learn More

<div class="grid cards" markdown>

-   **[Basic Queries](basic-queries.md)**

    ---

    MATCH, RETURN and basic pattern matching.

-   **[Pattern Matching](patterns.md)**

    ---

    Node and edge patterns in detail.

-   **[Filtering](filtering.md)**

    ---

    WHERE clauses and conditions.

-   **[Aggregations](aggregations.md)**

    ---

    COUNT, SUM, AVG and grouping.

-   **[Path Queries](paths.md)**

    ---

    Variable-length paths and shortest paths.

-   **[Mutations](mutations.md)**

    ---

    INSERT, SET, DELETE operations.

</div>
