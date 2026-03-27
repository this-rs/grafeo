---
title: SPARQL Query Language
description: Learn the SPARQL query language for RDF data in Obrain.
---

# SPARQL Query Language

SPARQL (SPARQL Protocol and RDF Query Language) is the W3C standard query language for RDF (Resource Description Framework) data. Obrain implements SPARQL 1.1 for querying RDF graphs.

## Overview

SPARQL uses triple patterns to match RDF data. It's designed for querying semantic web data and knowledge graphs.

## Quick Reference

| Operation | Syntax |
|-----------|--------|
| Select variables | `SELECT ?x ?y` |
| Match triples | `?s ?p ?o` |
| Filter results | `FILTER(?x > value)` |
| Optional patterns | `OPTIONAL { ?s ?p ?o }` |
| Union patterns | `{ ... } UNION { ... }` |
| Aggregate | `COUNT(?x)`, `SUM(?x)` |
| Order results | `ORDER BY ?x` |
| Limit results | `LIMIT 10` |

## RDF Data Model

Unlike property graphs (LPG), RDF uses triples:

```
Subject --Predicate--> Object
```

Example triples:
```
<http://example.org/alix> <http://xmlns.com/foaf/0.1/name> "Alix" .
<http://example.org/alix> <http://xmlns.com/foaf/0.1/knows> <http://example.org/gus> .
```

## Using SPARQL

SPARQL is enabled by default in Obrain:

=== "Python"

    ```python
    import obrain

    db = obrain.ObrainDB()

    # Insert RDF triples
    db.execute_sparql("""
        INSERT DATA {
            <http://example.org/alix> <http://xmlns.com/foaf/0.1/name> "Alix" .
            <http://example.org/alix> <http://xmlns.com/foaf/0.1/knows> <http://example.org/gus> .
        }
    """)

    # Query triples
    result = db.execute_sparql("""
        SELECT ?name WHERE {
            <http://example.org/alix> <http://xmlns.com/foaf/0.1/name> ?name .
        }
    """)
    for row in result:
        print(row)
    ```

=== "Rust"

    ```rust
    use obrain::ObrainDB;

    let db = ObrainDB::new_in_memory();
    let mut session = db.session();

    session.execute_sparql(r#"
        SELECT ?s ?p ?o WHERE { ?s ?p ?o }
    "#)?;
    ```

## Learn More

<div class="grid cards" markdown>

-   **[Basic Queries](basic-queries.md)**

    ---

    SELECT, WHERE and basic triple patterns.

-   **[Triple Patterns](patterns.md)**

    ---

    Matching subjects, predicates and objects.

-   **[Filtering](filtering.md)**

    ---

    FILTER expressions and conditions.

-   **[Aggregations](aggregations.md)**

    ---

    COUNT, SUM, AVG, GROUP BY and HAVING.

-   **[Property Paths](paths.md)**

    ---

    Path expressions for traversing relationships.

-   **[Built-in Functions](functions.md)**

    ---

    String, numeric and date/time functions.

</div>
