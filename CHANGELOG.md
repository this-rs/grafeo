# Obrain Changelog

All notable changes to Obrain will be documented in this file.

This project was forked from [this-rs/obrain](https://github.com/this-rs/obrain) and rebranded as **Obrain** ([obrain.dev](https://obrain.dev)). The upstream Obrain changelog is preserved in `CHANGELOG-grafeo.md` for historical reference.

## [0.0.1] - 2026-03-27

### 🎉 Initial Release — Clean Break from Obrain

This is the first release of **Obrain**, a complete rebrand of the Obrain graph database.

### Summary

- Full rebrand: all references to "obrain" replaced with "obrain" across 15 crates, 7 language bindings, CI/CD, and documentation
- New file extension: `.obrain` (replaces `.obrain`)
- New npm scope: `@obrain`
- Version reset to `0.0.1` — clean break, no backward compatibility with previous Obrain releases

### Inherited Features

- **Multi-model graph database**: LPG (Labeled Property Graph) + RDF dual-model storage
- **6 query languages**: GQL (ISO), Cypher, Gremlin, GraphQL, SPARQL, SQL/PGQ
- **ACID transactions** with MVCC epoch-based concurrency
- **Vector search**: HNSW index for approximate nearest-neighbor queries
- **Full-text search**: BM25-based text indexing
- **Temporal properties**: opt-in append-only versioning with point-in-time queries
- **7 language bindings**: Python, Node.js, C, WASM, Go, Dart, C#
- **Snapshot persistence**: binary snapshot format with WAL (Write-Ahead Log)
- **Observability**: Prometheus metrics export, structured tracing spans
- **Graph Data Science**: community detection, centrality, pathfinding algorithms
- **Read-only mode**: shared file lock for concurrent readers

### Breaking

- No backward compatibility with previous Obrain releases
- File format uses `.obrain` extension exclusively
- All crate names changed from `obrain-*` to `obrain-*`
- Python package: `obrain` (was `obrain`)
- npm package: `@obrain/obrain` (was `@obrain/obrain`)
