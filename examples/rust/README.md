# Rust Examples

Runnable examples demonstrating Obrain's Rust API. Each example is self-contained and produces meaningful output.

## Running

All examples are binary targets in the `obrain-examples` crate:

```bash
cargo run -p obrain-examples --bin <name>
```

## Examples

| Example | Features | Description |
|---------|----------|-------------|
| `basic` | (default) | Create nodes and edges, query with GQL, iterate results |
| `transactions` | (default) | Begin, commit, rollback, savepoints |
| `parameterized` | (default) | Safe parameterized queries with `$name` syntax |
| `vector_search` | (default) | HNSW vector index, k-NN similarity search |
| `algorithms` | (default) | PageRank, connected components, Louvain, degree centrality |
| `persistence` | `storage` | WAL-backed storage, snapshot export/import |
| `multi_language` | `full` | Same data queried with GQL, Cypher, SQL/PGQ |

Examples requiring extra features:

```bash
cargo run -p obrain-examples --bin persistence --features storage
cargo run -p obrain-examples --bin multi_language --features full
```

## Build all

```bash
# Default feature examples
cargo build -p obrain-examples

# All examples (including feature-gated ones)
cargo build -p obrain-examples --bins --features full
```
