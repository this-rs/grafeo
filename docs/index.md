---
title: Obrain - The Graph Database That Learns
description: A cognitive graph database with engrams, Hebbian synapses, spreading activation, and 7 biomimetic layers. High-performance Rust core with Python, Node.js, Go, C, C#, Dart and WebAssembly bindings.
hide:
  - navigation
  - toc
---

<style>
.md-typeset h1 {
  display: none;
}
</style>

<div class="hero" markdown>

# **Obrain**

### The graph database that learns

**Cognitive graph engine** — your data forms engrams, builds synapses, and surfaces insights through spreading activation. Built in Rust, embeddable everywhere.

[Get Started](getting-started/index.md){ .md-button .md-button--primary }
[View on GitHub](https://github.com/this-rs/obrain){ .md-button }

</div>

[![Obrain Playground](assets/playground.png)](https://obrain.ai)

---

## Why Obrain?

Traditional graph databases store nodes and edges, then wait for queries. Obrain goes further — it **thinks** about its own topology, continuously learning what matters.

<div class="grid cards" markdown>

-   :material-brain:{ .lg .middle } **Cognitive Engrams**

    ---

    Every node carries **energy** that decays over time. Mutations boost activation; unused data fades naturally. Your graph remembers what matters and forgets what doesn't — like biological memory.

-   :material-transit-connection-variant:{ .lg .middle } **Hebbian Synapses**

    ---

    Edges strengthen when both endpoints are co-activated — "neurons that fire together wire together." Synapses compete, prune, and self-organize based on real usage patterns.

-   :material-scatter-plot:{ .lg .middle } **Spreading Activation**

    ---

    Query one node and energy propagates through synapses via BFS, revealing related hot zones. Discover implicit connections your queries never explicitly asked for.

-   :material-layers-triple:{ .lg .middle } **7 Biomimetic Layers**

    ---

    Query languages → Query engine → Graph storage → Reactive pipeline → Cognitive layer → Index layer → Bindings. Each layer mirrors a biological information-processing stage.

-   :material-shield-alert:{ .lg .middle } **Scars & Risk Fabric**

    ---

    Failed transactions leave **scars** that decay but persist — the graph remembers past problems. The **knowledge fabric** scores every node for risk, staleness, and structural importance.

-   :material-robot:{ .lg .middle } **RAG-Native**

    ---

    HNSW vector search, BM25 full-text, hybrid search with RRF fusion. Combined with cognitive scoring, Obrain is purpose-built for Retrieval-Augmented Generation pipelines.

</div>

---

## Cognitive vs Traditional

| Traditional Graph DB | Obrain (Cognitive Graph DB) |
|---------------------|----------------------------|
| Static storage | **Reactive** — mutations trigger cognitive updates |
| Edges are inert | **Synapses** — edges strengthen with use, decay without it |
| Nodes are passive | **Engrams** — nodes have activation energy that decays exponentially |
| Manual analysis | **Fabric** — continuous risk, staleness, and centrality scoring |
| No memory of errors | **Scars** — persistent memory of failures that influence risk |
| Uniform importance | **Spreading activation** — query one node, discover related hot zones |

---

## Quick Start

=== "Python"

    ```bash
    uv add obrain
    ```

    ```python
    import obrain

    # Create an in-memory cognitive database
    db = obrain.ObrainDB()

    # Create nodes and edges
    db.execute("""
        INSERT (:Person {name: 'Alix', age: 30})
        INSERT (:Person {name: 'Gus', age: 25})
    """)

    db.execute("""
        MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'})
        INSERT (a)-[:KNOWS {since: 2024}]->(b)
    """)

    # Query the graph
    result = db.execute("""
        MATCH (p:Person)-[:KNOWS]->(friend)
        RETURN p.name, friend.name
    """)

    for row in result:
        print(f"{row['p.name']} knows {row['friend.name']}")
    ```

=== "Rust"

    ```bash
    cargo add obrain
    ```

    ```rust
    use obrain::ObrainDB;

    fn main() -> Result<(), obrain_common::utils::error::Error> {
        // Create an in-memory cognitive database
        let db = ObrainDB::new_in_memory();

        // Create a session and execute queries
        let mut session = db.session();

        session.execute(r#"
            INSERT (:Person {name: 'Alix', age: 30})
            INSERT (:Person {name: 'Gus', age: 25})
        "#)?;

        session.execute(r#"
            MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'})
            INSERT (a)-[:KNOWS {since: 2024}]->(b)
        "#)?;

        // Query the graph
        let result = session.execute(r#"
            MATCH (p:Person)-[:KNOWS]->(friend)
            RETURN p.name, friend.name
        "#)?;

        for row in result.rows {
            println!("{:?}", row);
        }

        Ok(())
    }
    ```

---

## Graph Foundation

Obrain's cognitive intelligence runs on top of a battle-tested graph engine:

<div class="grid cards" markdown>

-   :material-lightning-bolt:{ .lg .middle } **High Performance**

    ---

    Fastest graph database tested on the [LDBC Social Network Benchmark](ecosystem/graph-bench.md), both embedded and as a server, with a lower memory footprint than other in-memory databases.

-   :material-database-search:{ .lg .middle } **Multi-Language Queries**

    ---

    GQL, Cypher, Gremlin, GraphQL, SPARQL and SQL/PGQ. Choose the query language that fits the project.

-   :material-graph:{ .lg .middle } **LPG & RDF Support**

    ---

    Dual data model support for both Labeled Property Graphs and RDF triples. Choose the model that fits the domain.

-   :material-vector-line:{ .lg .middle } **Vector Search**

    ---

    HNSW-based similarity search with quantization (Scalar, Binary, Product). Combine graph traversal with semantic similarity.

-   :material-memory:{ .lg .middle } **Embedded or Standalone**

    ---

    Embed directly into applications with zero external dependencies, or run as a standalone server with REST API and web UI.

-   :fontawesome-brands-rust:{ .lg .middle } **Rust Core**

    ---

    Core engine written in Rust with no required C dependencies. Memory-safe by design with fearless concurrency.

-   :material-shield-check:{ .lg .middle } **ACID Transactions**

    ---

    Full ACID compliance with MVCC-based snapshot isolation. Reliable transactions for production workloads.

-   :material-language-python:{ .lg .middle } **Multi-Language Bindings**

    ---

    Python (PyO3), Node.js/TypeScript (napi-rs), Go (CGO), C (FFI), C# (.NET 8 P/Invoke), Dart (dart:ffi) and WebAssembly (wasm-bindgen).

</div>

---

## Features

### Query Languages

Choose the query language that fits the project:

| Language | Data Model | Style |
|----------|------------|-------|
| **GQL** (default) | LPG | ISO standard, declarative pattern matching |
| **Cypher** | LPG | Neo4j-compatible, ASCII-art patterns |
| **Gremlin** | LPG | Apache TinkerPop, traversal-based |
| **GraphQL** | LPG, RDF | Schema-driven, familiar to web developers |
| **SPARQL** | RDF | W3C standard for RDF queries |
| **SQL/PGQ** | LPG | SQL:2023 GRAPH_TABLE for SQL-native graph queries |

### Architecture Highlights

- **7 biomimetic layers**: Query → Engine → Storage → Reactive → Cognitive → Index → Bindings
- **Push-based execution engine** with morsel-driven parallelism
- **Columnar storage** with type-specific compression
- **Cost-based query optimizer** with cardinality estimation
- **MVCC transactions** with snapshot isolation
- **Zone maps** for intelligent data skipping

---

## Installation

=== "Python"

    ```bash
    uv add obrain
    ```

=== "Node.js"

    ```bash
    npm install @obrain-db/js
    ```

=== "Go"

    ```bash
    go get github.com/this-rs/obrain/crates/bindings/go
    ```

=== "Rust"

    ```bash
    cargo add obrain
    ```

=== "C#"

    ```bash
    dotnet add package ObrainDB
    ```

=== "Dart"

    ```yaml
    # pubspec.yaml
    dependencies:
      obrain: ^0.5.21
    ```

=== "WASM"

    ```bash
    npm install @obrain-db/wasm
    ```

---

## License

Obrain is licensed under the [Apache-2.0 License](https://github.com/this-rs/obrain/blob/main/LICENSE).
