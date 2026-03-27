# Roadmap

Obrain is a high-performance, embeddable graph database written in Rust. This roadmap shows where the project has been, where it is now and where it's going. Priorities may shift based on community feedback and real-world usage.

For detailed release notes, see the [CHANGELOG](changelog.md).

---

## Completed Releases

### 0.1: Foundation

Established the core graph engine: labeled property graph (LPG) storage with MVCC transactions, WAL persistence and multiple index types (hash, B-tree, trie, adjacency). Shipped the GQL (ISO standard) parser as the primary query language, with experimental support for Cypher, SPARQL, Gremlin and GraphQL. Python bindings via PyO3 from day one.

### 0.2: Performance

Made the engine competitive on query throughput. Factorized query processing eliminates Cartesian products in multi-hop traversals. Worst-case optimal joins (Leapfrog TrieJoin) handle cyclic patterns efficiently. Lock-free concurrent reads, query plan caching and direct lookup APIs brought large speedups on common access patterns. First graph algorithms (community detection, clustering coefficient, BFS shortest path). Added RDF Ring Index, Block-STM parallel transactions, tiered storage and succinct data structures as opt-in features.

### 0.3: AI Compatibility

Added first-class vector support: `Value::Vector` type, HNSW approximate nearest neighbor index, four distance metrics (cosine, euclidean, dot product, manhattan) with SIMD acceleration (AVX2, SSE, NEON). Vector quantization (scalar, binary, product) for memory-constrained deployments. Hybrid graph + vector queries across all supported query languages. Serializable snapshot isolation for write-heavy workloads.

### 0.4: Developer Accessibility

Expanded the binding surface: Node.js/TypeScript (napi-rs), Go (C FFI), WebAssembly (wasm-bindgen, 660 KB gzipped), SQL/PGQ (SQL:2023 GRAPH_TABLE). Shipped obrain-cli with interactive shell, filtered/MMR vector search with incremental indexing. All public API items documented.

---

## Current: 0.5, Beta

*Preparing for production readiness.*

The beta series focuses on correctness, completeness and real-world durability. Key areas:

**Search and retrieval**: BM25 text search, hybrid search (BM25 + vector via RRF/weighted fusion), optional in-process ONNX embeddings, MMR for diverse RAG results.

**Graph algorithms**: CALL procedure interface exposing all 22 algorithms (PageRank, Dijkstra, Louvain, SSSP, etc.) from GQL, Cypher and SQL/PGQ. Algorithms themselves were introduced in 0.2, the query-callable interface is new in 0.5.

**Data management**: Temporal types (Date, Time, Duration, DateTime), graph type enforcement with schema validation and constraints, LOAD DATA for CSV/JSONL/Parquet, named graph persistence, cross-graph transactions.

**Transaction correctness**: MVCC dirty-read prevention, DELETE rollback with full undo log, write-write conflict detection, session auto-rollback, savepoints.

**Persistence**: Single-file `.obrain` format with dual-header crash safety, index metadata persistence (snapshot v4), read-only open mode with shared file lock for concurrent reader processes.

**Bindings**: C#/.NET 8, Dart/Flutter (community contribution), C FFI layer shared by Go, and C#.

**Ecosystem**: [obrain-server](https://github.com/ObrainDB/obrain-server), [obrain-web](https://github.com/ObrainDB/obrain-web), [obrain-mcp](https://github.com/ObrainDB/obrain-mcp), [obrain-memory](https://github.com/ObrainDB/obrain-memory), [obrain-langchain](https://github.com/ObrainDB/obrain-langchain), [obrain-llamaindex](https://github.com/ObrainDB/obrain-llamaindex).

### What's left in 0.5

- Query language spec compliance (GQL ~95%, Cypher ~90%, SPARQL ~98%, SQL/PGQ ~75%)
- C FFI parity with Python/Node.js surface (so C#, Go and Dart get full coverage)
- API stability review before freezing for 1.0

---

## Next: 0.6, Release Candidate

*Stability and production hardening. No new major features.*

The scope is intentionally narrow:

- **Bug fixes** from real-world 0.5 usage
- **Performance tuning** informed by actual workloads, not synthetic benchmarks
- **API ergonomics** and documentation polish
- **Binary size and compile time** optimization
- **C FFI parity refactor**: expand obrain-c to match Python/Node.js API surface, update downstream bindings

The goal is confidence: if something works in 0.6, it works in 1.0.

---

## 1.0: Stable

Semantic versioning commitment. Public API frozen. No breaking changes without a major version bump.

---

## Future Considerations

Not scheduled, but on the radar:

- Distributed/clustered deployment
- Additional language bindings (Java/Kotlin, Swift)
- Cloud-native integrations

---

## Contributing

Interested in contributing? Check the [GitHub Issues](https://github.com/ObrainDB/obrain/issues) or join the [Discussions](https://github.com/orgs/ObrainDB/discussions).

---

Last updated: March 2026
