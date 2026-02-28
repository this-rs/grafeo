# Changelog

All notable changes to Grafeo, for future reference (and enjoyment).

## [0.5.10] - 2026-02-29

### Added

- **Skip index for adjacency chunks**: compressed cold chunks now maintain a zone-map skip index with per-chunk `(min, max)` destination metadata. New `contains_edge(src, dst)` provides O(log n) point lookups by only decompressing chunks whose range overlaps the target. `edges_in_range(src, min, max)` supports efficient range queries with the same pruning
- **Bidirectional BFS shortest path**: `ShortestPathOperator` now uses meet-in-the-middle BFS, expanding forward and backward frontiers alternately (smaller frontier first). Reduces search space from O(b^d) to O(b^(d/2)). Falls back to unidirectional BFS when backward adjacency is unavailable

### Improved

- **Shared accumulator module**: extracted `AggregateFunction`, `AggregateExpr`, and `HashableValue` from duplicate definitions in pull-based and push-based aggregate operators into a single `accumulator` module, eliminating ~200 lines of duplicated type definitions
- **Thread-local crash injection**: crash injection mechanism now uses thread-local storage instead of global atomics, preventing interference between concurrent tests when `testing-crash-injection` is enabled
- **Crash recovery tests**: 7 deterministic crash injection tests verifying WAL recovery invariants at every crash point - partial writes, uncommitted data, aborted transactions, checkpoint crashes, log rotation, and an exhaustive sweep across all crash points
- **Concurrent stress tests**: 6 multi-threaded stress tests covering concurrent writers (8 threads × 50 writes), mixed read/write workloads, transaction conflict patterns with interleaved commit/rollback, epoch pressure under load, rapid session lifecycle, and concurrent node+edge creation
- **Hardened panic messages**: converted ~50 bare `unwrap()` calls to `expect()` with invariant descriptions across graph algorithms (centrality, community detection, components, flow, MST, structure, clustering) and execution operators (expand, join, unwind, project, merge, shortest path, variable-length expand, factorized chunks). No behavioral change - improves diagnostic quality if an internal invariant is ever violated

## [0.5.9] - 2026-02-28

### Added

- **Snapshot validation**: `import_snapshot()` now pre-validates the entire snapshot before inserting any data: rejects duplicate node/edge IDs and edges referencing non-existent source or destination nodes
- **Crash injection framework**: feature-gated `testing-crash-injection` module in grafeo-core with `maybe_crash()` / `with_crash_at()` for deterministic recovery testing. Three crash points added to WAL write path (`wal_before_write`, `wal_after_write`, `wal_before_flush`). Zero overhead when disabled
- **Backward compatibility tests**: pinned v1 snapshot fixture with 8 regression tests verifying format stability across future code changes
- **RDF batch insert lock batching**: `RdfStore::batch_insert()` acquires the write lock once for the entire batch instead of per-triple, reducing lock overhead for bulk RDF imports

### Fixed

- **WASM build with `getrandom` 0.4**: added `wasm_js` crate feature for `getrandom` 0.4.x on `wasm32-unknown-unknown` targets. The previous `getrandom_backend` cfg flag only worked for 0.3.x
- **WASM binary size regression**: `grafeo-bindings-common` was pulling in all engine default features (`wal`, `parallel`, `spill`, `mmap`) into the WASM build via transitive dependencies. Disabled default features on the common crate's engine dependency, reducing WASM gzip size from 974 KB to 744 KB (under the 800 KB target)

### Improved

- **Compact property storage**: Node and Edge property maps switched from `BTreeMap<PropertyKey, Value>` to a `SmallVec<[(PropertyKey, Value); 4]>` representation. Nodes with 4 or fewer properties avoid heap allocation entirely
- **Cost model per-type fanout**: the query optimizer now tracks per-edge-type average degree statistics, replacing the single global fanout estimate. Improves plan selection for graphs with heterogeneous edge type distributions
- **Translator consolidation**: extracted `capitalize_first()`, `VarGen` (variable name generator), and `combine_with_and()` from individual translators into `translator_common`, removing duplication across 6 query language translators
- **LpgStore API surface reduction**: reduced public API visibility with `pub(crate)` for internal-only methods (`get_node_at_epoch`, `get_edge_at_epoch`, `delete_node_at_epoch`, `delete_edge_at_epoch`, `compute_statistics`) and `#[doc(hidden)]` for test-only methods
- **Release workflow**: replaced deprecated `actions/create-release@v1` and `actions/upload-release-asset@v1` with `softprops/action-gh-release@v2`, eliminating `set-output` deprecation warnings

## [0.5.8] - 2026-02-22

### Added

- **`grafeo-bindings-common` crate**: shared library for all language bindings, eliminating duplicated entity extraction, error classification, and JSON-to-Value conversion. All four bindings (Python, Node.js, C, WASM) now delegate to the common crate
- **Unified query dispatch**: `GrafeoDB::execute_language()` and `Session::execute_language()` accept a language name string (`"gql"`, `"cypher"`, `"gremlin"`, `"graphql"`, `"sparql"`, `"sql"`) with optional parameters, replacing 18 near-identical per-language functions across bindings
- **Node.js API parity**: added `removeNodeProperty()`, `removeEdgeProperty()`, `addNodeLabel()`, `removeNodeLabel()`, `getNodeLabels()`, `info()`, `schema()`, and `version()` methods to match the Python binding's API surface
- **Node.js transaction isolation**: `beginTransaction()` now accepts an optional `isolationLevel` parameter (`"read_committed"`, `"snapshot"`, `"serializable"`), matching Python
- **WASM parameterized queries**: `executeWithParams()`, `executeWithLanguageAndParams()`, and `executeRawWithLanguage()` for parameterized and multi-language queries in the browser
- **WASM per-language convenience methods**: `executeCypher()`, `executeGremlin()`, `executeGraphql()`, `executeSparql()`, `executeSql()` with proper feature gating
- **Batch edge creation**: `LpgStore::batch_create_edges()` and `ChunkedAdjacency::batch_add_edges()` for efficient bulk graph import with single lock acquisition
- **grafeo-adapters documentation**: all previously undocumented public items across Gremlin/GraphQL lexers, keywords, and AST types now have doc comments. `#![allow(missing_docs)]` replaced with `#![warn(missing_docs)]`
- **Doc-test overhaul**: converted all remaining `ignore` doc-tests to compilable `no_run` or fully runnable examples. Zero `ignore` annotations remain in the codebase

### Improved

- **Incremental statistics tracking**: `compute_statistics()` now reads from atomic delta counters instead of scanning all nodes and edges. Reduces statistics refresh from O(n+m) to O(|labels|+|edge_types|). Full resync only triggers after transaction rollback
- **Cost model uses real fanout**: the query optimizer's cost model now derives average edge fanout from actual graph statistics instead of a hardcoded value of 10.0, improving plan selection for sparse and dense graphs
- **Consistent ID sentinels**: `TxId::INVALID` now uses `u64::MAX` (matching `NodeId`, `EdgeId`, `LabelId`, `PropertyKeyId`, `EdgeTypeId`, `IndexId`) instead of `0`
- **Binding consistency**: all four bindings now share entity extraction, error classification, and JSON conversion via `grafeo-bindings-common`, ensuring identical behavior and reducing maintenance surface

## [0.5.7] - 2026-02-19

### Fixed

- **UNWIND mutation property access**: `UNWIND $edges AS e MATCH (a {id: e.src}), (b {id: e.tgt}) CREATE (a)-[:REL {w: e.weight}]->(b)` now correctly resolves map property access (`e.src`, `e.weight`) in CREATE/SET property lists. Previously only column references and constants were supported, causing map properties to resolve as NULL

### Added

- **`algos` feature flag**: graph algorithms (7,500+ LOC) are now gated behind an `algos` feature flag across grafeo-adapters, grafeo-engine, and Python bindings. Included in the `full` feature group so existing users are unaffected. Reduces compile time and binary size when algorithms are not needed

### Improved

- **LpgStore submodule split**: split the monolithic `store.rs` (4,600+ lines) into 10 focused submodules (node_ops, edge_ops, property_ops, traversal, schema, index, search, statistics, versioning). No public API changes. Same pattern as the earlier `database/` module split
- **Translator consolidation**: extracted shared `is_aggregate_function` and `to_aggregate_function` helpers from GQL and Cypher translators into a common `translator_common` module, removing code duplication
- **Doc-test fixes**: converted 9 ignored doc-tests to compilable `no_run` examples across fold.rs, cardinality.rs, and embedding config. Fixed incorrect function signatures in parallel_partition doc-test

## [0.5.6] - 2026-02-18

Safety and performance improvements. Zero unsafe code remaining in property storage. Named graph management operators fully integrated. UNWIND and FOR list expansion for batch operations.

### Added

- **UNWIND clause**: expand lists into individual rows for batch processing. `UNWIND [1, 2, 3] AS x RETURN x` produces 3 rows. Works with literal lists, parameter-substituted lists (`UNWIND $items AS x`), and vectors
- **FOR statement** (GQL standard, ISO/IEC 39075 section 14.8): `FOR x IN [1, 2, 3] RETURN x`: the GQL-native equivalent of Cypher's UNWIND, desugars to the same execution plan
- **FOR WITH ORDINALITY/OFFSET**: `FOR x IN list WITH ORDINALITY i` emits a 1-based index; `FOR x IN list WITH OFFSET i` emits a 0-based index. Threaded through all 7 pipeline layers (lexer, AST, parser, translator, plan, planner, operator)
- **Text index mutation sync**: text indexes are now automatically updated when node properties are set, removed, or when nodes are deleted, no manual `rebuildTextIndex()` needed
- **WASM text/hybrid search bindings**: `createTextIndex`, `dropTextIndex`, `rebuildTextIndex`, `textSearch`, and `hybridSearch` methods added to the WASM `Database` class
- **UNWIND + MATCH + INSERT**: batch mutation pattern: `UNWIND $edges AS e MATCH (a {name: e.from}), (b {name: e.to}) INSERT (a)-[:KNOWS]->(b)` creates edges in bulk from parameter lists
- **Ordered clause processing**: GQL translator now processes MATCH, UNWIND, FOR, INSERT, and RETURN clauses in source order, preserving variable scoping across clause boundaries
- **SPARQL COPY/MOVE/ADD graph operators**: `COPY <src> TO <dst>`, `MOVE <src> TO <dst>`, and `ADD <src> TO <dst>` now execute end-to-end with source-existence validation and SILENT support
- **Embedding model config + auto-download**: `EmbeddingModelConfig` enum with 3 presets (MiniLM-L6-v2, MiniLM-L12-v2, BGE-small-en-v1.5) and HuggingFace Hub auto-download via `load_embedding_model()`. Configurable batch size and ONNX thread count via `EmbeddingOptions`. Exposed in Python and Node.js bindings
- **Native SSSP procedure**: `CALL grafeo.sssp('node_name', 'weight') YIELD node_id, distance`: single-source shortest paths with string-based node name resolution for LDBC Graphanalytics compatibility. Also exposed as `db.algorithms.sssp(source, weight_attr)` in Python

### Fixed

- **UNWIND scalar values returning NULL**: `plan_return` now tracks UNWIND/FOR variables as scalars, preventing `resolve_entities` from incorrectly treating them as node IDs
- **UNWIND with Value::Vector**: `UnwindOperator` now handles both `Value::List` and `Value::Vector`, fixing 0-row results when parameters contain all-numeric Python lists (which PyO3 converts to vectors)
- **UNWIND + MATCH variable scoping**: MATCH clauses following UNWIND now receive the UNWIND output as input (via nested loop join), so property filters like `MATCH (n {id: x})` can reference UNWIND variables. Previously used a cross join that isolated the MATCH from UNWIND scope
- **Map property access in UNWIND**: `e.src` / `e.tgt` property access on map values from UNWIND (e.g., `UNWIND $edges AS e MATCH (a {id: e.src})`) now works in both ProjectOperator and FilterExpression. Previously only node/edge IDs supported property access, returning NULL for maps
- **`RETURN n` now returns full node/edge maps**: queries like `MATCH (n) RETURN n` now return `{_id, _labels, ...properties}` instead of a bare integer ID, matching Neo4j/Memgraph behavior. Resolution now happens inside the ProjectOperator pipeline (single-pass) instead of post-processing
- **GQL lexer UTF-8 panic**: multi-byte characters (e.g., `ç`, `ã`, CJK) in queries no longer cause `byte index is not a char boundary` panics. The lexer now advances by `char.len_utf8()` instead of a fixed 1 byte
- **Scalar column tracking in Return**: `plan_return` now correctly passes through scalar values from Project/Aggregate operators instead of applying NodeResolve. Fixes Gremlin `.values()`, `.count()`, and GQL `WITH expr AS alias` returning NULL
- **Vector index rebuild after drop**: `rebuild_vector_index()` now works even when the index was previously dropped, inferring dimensions from existing data instead of requiring the old index to still exist

### Improved

- **Safe ID conversions**: replaced final 3 `unsafe transmute_copy` calls in property column compression/decompression with safe `EntityId::as_u64()` / `from_u64()` methods. Removed 4 stale `#[allow(unsafe_code)]` attributes, property.rs is now fully safe
- **Statistics access**: `LpgStore::statistics()` now returns `Arc<Statistics>` (cheap pointer clone) instead of deep-cloning the entire statistics struct on every query planner invocation
- **HashableValue hashing/comparison**: recursive `Hash` and `PartialEq` impls for nested `List`/`Map`/`Vector` values now operate by reference instead of cloning each element, eliminating O(n) allocations per hash/compare operation
- **Entity resolution in ProjectOperator**: `RETURN n` entity resolution moved from 6-site post-processing (`resolve_entities()`) into `ProjectExpr::NodeResolve`/`EdgeResolve` variants inside the ProjectOperator pipeline: single-pass, automatic, no call sites to maintain
- **Test coverage**: ~100+ new tests across Rust integration tests (mutation planning, expressions, filter pushdown, search, WAL recovery), C/WASM/Go binding tests, Python integration tests (advanced queries, text/hybrid search, CDC), and Node.js tests

## [0.5.5] - 2026-02-16

Small bugfixes and code quality improvements. 0.5.x will continue to be like this until bugs become extremely rare.

### Added

- **Filter pushdown optimization**: equality predicates on labeled node scans are now pushed down to the store level, bypassing DataChunk materialization and expression evaluation. Compound predicates (e.g., `WHERE n.name = 'Alice' AND n.age > 30`) correctly split: equality part pushed down, remainder kept as post-filter
- **Query error positions**: all six parsers (GQL, Cypher, SPARQL, Gremlin, GraphQL, SQL/PGQ) now produce errors with line/column positions and source-caret display

### Fixed

- **Transaction edge type visibility**: edges created within a transaction are now visible to subsequent queries in the same transaction
- **INSERT DATA with GRAPH clause**: SPARQL `INSERT DATA { GRAPH <uri> { ... } }` and `DELETE DATA { GRAPH <uri> { ... } }` now correctly route triples to the named graph instead of the default graph
- **Compound predicate correctness**: filter pushdown no longer silently drops non-equality parts of compound predicates (e.g., range conditions after equality pushdown)

### Improved

- **Hot-path clone elimination**: replaced unnecessary `.clone()` calls with moves/borrows in sort keys (3 sites), aggregate finalize (`take()` instead of `clone()`), aggregate state transitions (`mem::take` instead of HashSet clone), and filter PropertyKey construction

## [0.5.4] - 2026-02-15

### Fixed

- **Multi-pattern CREATE**: `CREATE (:A {id: 'x'}), (:B {id: 'y'})` now creates all nodes instead of only the first. The GQL translator's `translate_insert()` was only processing `patterns[0]`; it now loops through all patterns and chains `CreateNode` operators

## [0.5.3] - 2026-02-13

### Improved

- **Query error quality**: translator errors (GQL, Cypher, SQL/PGQ, GraphQL, SPARQL, Gremlin) now produce `QueryError` with semantic error codes (`GRAFEO-Q002`) instead of generic internal errors (`GRAFEO-X001`). Error messages are more actionable
- **GraphQL range filters on direct arguments**: operator suffixes (`_gt`, `_lt`, `_gte`, `_lte`, `_ne`, `_contains`, `_starts_with`, `_ends_with`) now work on direct query arguments (`person(age_gt: 30)`) in addition to `where` clauses

### Fixed

- **SPARQL `FILTER NOT EXISTS`**: parser now recognizes `NOT EXISTS` and `EXISTS` as built-in calls in FILTER constraints, correctly producing anti-join / semi-join plans
- **SPARQL `FILTER REGEX`**: added REGEX function evaluation in the RDF query planner (parser and translator already supported it, but the planner silently returned no matches)
- Re-enabled 7 previously-skipped Python tests - all pass: GraphQL range/nested filters (fixed by direct argument operator parsing), SPARQL REGEX and NOT EXISTS (fixed by parser and planner additions)

## [0.5.2] - 2026-02-13

### Added

- **CALL procedure support**: invoke any of the 22 built-in graph algorithms directly from query strings using `CALL grafeo.<algorithm>() [YIELD columns]`. Supported in GQL, Cypher, and SQL/PGQ
- **YIELD clause**: select and alias specific output columns from procedure results (`CALL grafeo.pagerank() YIELD node_id, score AS rank`)
- **Procedure listing**: `CALL grafeo.procedures()` returns all available procedures with descriptions, parameters, and output columns
- **Map literal arguments**: pass named parameters to algorithms (`CALL grafeo.pagerank({damping: 0.85, max_iterations: 20})`)

## [0.5.1] - 2026-02-12

All new functionality for the 0.5.x series: hybrid search (BM25 + vector), built-in ONNX embeddings, change data capture, fully supporting the grafeo-memory AI memory package.

### Added

- **BM25 text search** (`text-index` feature): create inverted indexes on string properties with `create_text_index()` and search with BM25 scoring via `text_search()`. Includes a built-in tokenizer with Unicode word boundaries, lowercasing, and English stop word removal
- **Hybrid search** (`hybrid-search` feature): combine BM25 text scores with HNSW vector similarity via Reciprocal Rank Fusion (RRF) or weighted fusion. Single `hybrid_search()` call across Python and Node.js bindings
- **Built-in embeddings** (`embed` feature, opt-in): generate embeddings in-process via ONNX Runtime. Load any `.onnx` model + tokenizer, call `embed_text()` or `vector_search_text()` at the Rust API level. Binding exposure coming in a future release. Adds ~17MB to binary size, off by default
- **Change data capture** (`cdc` feature): track all node and edge mutations (create, update, delete) with before/after property snapshots. Query history via `history()`, `history_since()`, and `changes_between()`. Available in Python and Node.js bindings

## [0.5.0] - 2026-02-11

Internal engine improvements: ~50% memory savings for vector workloads, safer internals, production-grade error reporting, query timeouts, and automatic MVCC garbage collection.

### Added

- **Standardized error codes**: all errors now carry machine-readable `GRAFEO-XXXX` codes (Q = query, T = transaction, S = storage, V = validation, X = internal) with `error_code()` accessor and `is_retryable()` hint
- **Query timeout**: configurable `query_timeout` in `Config` stops long-running queries cleanly with `GRAFEO-Q003` error. Set via `Config::default().with_query_timeout(Duration::from_secs(30))`
- **MVCC auto-GC**: version chains are garbage-collected automatically every N commits (default: 100, configurable via `with_gc_interval()`). Also exposes `db.gc()` for manual control

### Improved

- **Topology-only HNSW**: vectors are no longer duplicated inside the HNSW index - the index stores only graph topology and reads vectors on-demand through a `VectorAccessor` trait. ~50% memory reduction for vector workloads
- **Safe ID conversions**: replaced 6 `unsafe transmute_copy` calls with safe `EntityId::as_u64()` / `EntityId::from_u64()` methods

## [0.4.4] - 2026-02-11

Adds SQL/PGQ (SQL:2023) graph queries, MMR search for RAG, auto-syncing vector indexes, and a fully rebuilt CLI with an interactive shell.

### Added

- **SQL/PGQ support**: you can now query your graph using standard SQL:2023 syntax, `SELECT ... FROM GRAPH_TABLE (MATCH ... COLUMNS ...)`. This also brings path functions (`LENGTH`, `NODES`, `EDGES`), DDL (`CREATE PROPERTY GRAPH`), and bindings across Python, Node.js, WASM, Go, and C
- **MMR search**: find diverse, relevant results for RAG pipelines with a single `mmr_search()` call. Tune the `lambda` parameter to balance relevance vs. diversity. Available in all bindings
- **Filtered vector search**: `vector_search()`, `batch_vector_search()`, and `mmr_search()` accept optional property equality filters to restrict results without post-filtering. Uses pre-computed allowlists from property indexes for efficient HNSW traversal. Available in Python, Node.js, C, and Go bindings
- **Incremental vector indexing**: vector indexes now stay in sync automatically as nodes change, no manual rebuilds needed. Also adds `drop_vector_index()` and `rebuild_vector_index()` for explicit control
- **CLI overhaul**: new `query`, `init`, `shell`, `version`, and `completions` commands. The interactive shell supports transactions, meta-commands (`:schema`, `:info`, `:stats`, `:format`, `:timing`), persistent history, CSV output, and `NO_COLOR` support
- **CLI distribution**: install `grafeo-cli` via `cargo install`, `pip install`, or `npm install -g` on Linux, macOS, and Windows (x64 + ARM64)
- **Configurable cardinality estimation**: tune 9 selectivity parameters via `SelectivityConfig` and compare estimates vs. actuals with `EstimationLog`
- **AdminService trait**: one interface for all database introspection and maintenance, `info()`, `detailed_stats()`, `schema()`, `validate()`, `wal_status()`, `wal_checkpoint()`
- **GQL `IN` operator**: `WHERE n.name IN ['Alice', 'Bob']` now works in GQL
- **String escape sequences**: `\'`, `\"`, `\\`, `\n`, `\r`, `\t` now work correctly in GQL, Cypher, and SQL/PGQ strings
- Comprehensive binding READMEs for Node.js, C, and Python
- All public API items are now documented (`missing_docs` lint enabled workspace-wide)
- WASM build verification in CI with gzip size check

### Fixed

- **Node.js ID validation**: rejects negative, NaN, Infinity, and values above `MAX_SAFE_INTEGER`
- **Error chain**: `Error::source()` now returns inner errors for `Transaction`, `Storage`, and `Query` variants
- **Value::serialize()**: returns `Result` instead of panicking
- **OrderableValue**: uses standard `TryFrom<&Value>` instead of a custom method

### Changed

- **Python CLI removed**: replaced by the unified `grafeo-cli` Rust binary (install via cargo, pip, or npm)
- Consolidated duplicated `format_bytes()` into shared `output::formatter` module
- Dead code cleanup, removed unused methods and struct fields
- Clippy Phase 3 clean (4 performance lints resolved)

## [0.4.3] - 2026-02-08

Per-database graph model selection, snapshot export/import for backups and browser persistence, and expanded WASM APIs.

### Added

- **Database creation options**: choose between LPG and RDF graph models per database, configure durability mode, toggle schema constraints, and validate configs with clear error messages
- **Query routing**: calling `execute()` on an RDF database now tells you what went wrong instead of silently running GQL. Use `execute_sparql()` and friends for cross-model queries
- **Inspection API**: check your database's graph model and memory limit at runtime
- **Snapshot export/import**: serialize your entire database to a binary snapshot for backups or WASM persistence via IndexedDB
- **WASM API expansion**: `executeWithLanguage()` for multi-language queries, `exportSnapshot()` / `importSnapshot()` for browser persistence, `schema()` for introspection

### Changed

- Re-exported `GraphModel`, `DurabilityMode`, `ConfigError` from umbrella `grafeo` crate

### Fixed

- Go badge on README now shows correctly
- Release workflow pings Go module proxy so pkg.go.dev indexes new versions

## [0.4.2] - 2026-02-08

Grafeo now runs in the browser. WebAssembly bindings with TypeScript definitions, shipped at 660 KB gzipped.

### Added

- **WebAssembly bindings** (`@grafeo-db/wasm`): run Grafeo in the browser with `execute()`, `executeRaw()`, `nodeCount()`, `edgeCount()`, and full TypeScript definitions
- **Feature-gated platform subsystems**: `parallel`, `spill`, `mmap`, `wal` are now opt-in features, making `wasm32` compilation straightforward
- WASM binary comes in at 660 KB gzipped (target was <800 KB)

### Fixed

- Go module versioning on pkg.go.dev now works correctly for monorepo subdirectories

### Changed

- Internal crate dependencies use `default-features = false` for per-crate feature control
- Stricter Clippy linting, removed 6 lint allows

## [0.4.1] - 2026-02-08

Go and C bindings, so you can embed Grafeo in pretty much any language now.

### Added

- **Go bindings** (`github.com/GrafeoDB/grafeo`): full node/edge CRUD, property and label management, multi-language queries, ACID transactions, HNSW vector search, batch operations, and admin APIs
- **C FFI layer** (`grafeo-c`): C-compatible ABI for embedding Grafeo in any language
- Per-crate coverage tracking in Codecov

### Fixed

- npm publish workflow and removed old JS stub package
- Node.js test version now dynamically read from Cargo.toml
- Coverage report now excludes bindings and test code for accurate metrics

## [0.4.0] - 2026-02-07

Node.js/TypeScript bindings with full async support, Python vector search and transaction isolation.

### Added

- **Node.js/TypeScript bindings** (`@grafeo-db/js`): full CRUD, async queries across all 5 languages, transactions, native type mapping, and TypeScript definitions
- **Python vector support**: pass `list[float]` directly as vectors, use `grafeo.vector()`, query with distance functions in GQL, create HNSW indexes, and run k-NN search
- **Python transaction isolation**: choose `"read_committed"`, `"snapshot"`, or `"serializable"` when starting a transaction
- **Batch vector APIs**: `batch_create_nodes()` and `batch_vector_search()` for Python and Node.js
- Node.js CI testing across 3 OS x 3 Node.js versions (18, 20, 22)
- `cargo-deny` integration for dependency auditing

### Fixed

- GQL INSERT with list or `vector()` properties no longer silently drops values
- Multi-hop MATCH queries (3+ hops) no longer return duplicate rows
- GQL multi-hop patterns now correctly filter intermediate nodes by label
- GraphQL filter queries accept `filter` as alias for `where`
- GraphQL nested relationships match edge types case-insensitively
- Transaction `execute()` rejects queries after commit/rollback

### Improved

- **HNSW recall**: Vamana-style diversity pruning with configurable alpha
- **HNSW speed**: pre-normalized cosine vectors, pre-allocated structures
- Query optimizer uses actual store statistics instead of hardcoded defaults

## [0.3.4] - 2026-02-06

Query timing, "did you mean?" error suggestions, and Python pagination.

### Added

- **Query performance metrics**: every result now includes `execution_time_ms` and `rows_scanned` so you can see exactly what happened
- **"Did you mean?" suggestions**: typo in a variable or label name? Grafeo will suggest the closest match
- **Python pagination**: `get_nodes_by_label()` now supports `offset` for efficient paging

### Documentation

- Troubleshooting guide, glossary, migration guide (from Neo4j, NetworkX, etc.), security guide, performance baselines, and example notebooks

## [0.3.3] - Unreleased

### Added

- **VectorJoin operator**: combine graph traversal with vector similarity in a single query, works with both static query vectors and entity-to-entity embedding comparisons
- **Vector zone maps**: automatically skips irrelevant data blocks during vector search
- **Vector cost estimation**: the query optimizer now understands vector scan costs and picks better plans
- **Product quantization**: 8-32x memory compression for large vector datasets with ~90% recall retention
- **Memory-mapped vector storage**: disk-backed storage with LRU caching for datasets that don't fit in RAM
- **Python quantization API**: `ScalarQuantizer`, `ProductQuantizer`, and `BinaryQuantizer` accessible from Python

## [0.3.2] - Unreleased

### Added

- **Selective property loading**: fetch only the properties you need instead of all columns, much faster for wide nodes
- **Parallel node scan**: 3-8x speedup on large scans (10K+ nodes) by distributing work across CPU cores

### Improved

- MVCC hot path inlined for faster full table scans
- Batch property reads pre-allocate for less allocation overhead

## [0.3.1] - Unreleased

### Added

- **Vector quantization**: compress vectors from f32 to u8 (scalar) or 1-bit (binary) for memory-efficient similarity search
- **Quantized HNSW**: approximate quantized search followed by exact rescoring, best of both worlds
- **SIMD acceleration**: 4-8x faster distance computations, automatically uses AVX2/FMA, SSE, or NEON depending on your CPU
- **Vector batch operations**: `batch_insert()` and `batch_search()` for bulk loading and multi-query search
- **VectorScan operators**: vector similarity search integrated into the query execution engine
- **Adaptive WAL flusher**: self-tuning background flush that adjusts timing based on actual disk speed
- **DurabilityMode::Adaptive**: new WAL mode for variable disk latency workloads
- **Fingerprinted hash index**: sharded index with 48-bit fingerprints for near-instant miss detection

## [0.3.0] - Unreleased

Vectors are now a first-class type. Graph + vector hybrid queries let you do things no pure vector database can.

### Added

- **Vector type**: store, hash, and serialize vectors natively with dimension-aware schema validation
- **Distance functions**: cosine, euclidean, dot product, and manhattan distance
- **Brute-force k-NN**: exact nearest neighbor search with optional predicate filtering
- **HNSW index**: O(log n) approximate nearest neighbor search with tunable parameters and presets (`high_recall()`, `fast()`)
- **GQL vector syntax**: `vector([...])` literals, distance functions, and `CREATE VECTOR INDEX` statements
- **SPARQL vector functions**: `VECTOR()`, `COSINE_SIMILARITY()`, `EUCLIDEAN_DISTANCE()`, `DOT_PRODUCT()`, `MANHATTAN_DISTANCE()`
- **Serializable snapshot isolation**: choose between `ReadCommitted`, `SnapshotIsolation`, or `Serializable` per transaction

### Fixed

- RDF queries now correctly exclude pending deletes within a transaction

---

## [0.2.7] - 2026-02-05

New parallel execution primitives and a second-chance LRU cache for concurrent workloads.

### Added

- **Second-chance LRU cache**: lock-free access marking for better concurrent cache performance
- **Parallel fold-reduce utilities**: `parallel_count`, `parallel_sum`, `parallel_stats`, `parallel_partition`
- **Generic collector trait**: composable parallel aggregation with built-in count, materialize, limit, and stats collectors

---

## [0.2.6] - 2026-02-04

Zone map filtering at the chunk level and faster batch property reads.

### Added

- **Local clustering coefficient**: triangle counting with parallel execution
- **Chunk-level zone map filtering**: skip entire data chunks when predicates can't match

### Improved

- Batch property retrieval now acquires a single lock instead of one per entity
- Filter operator checks zone maps before row-by-row evaluation

### Documentation

- Added CONTRIBUTORS.md

---

## [0.2.5] - 2026-02-03

Full SPARQL function coverage, platform allocators for faster memory allocation, and batch property APIs.

### Added

- **Full SPARQL function coverage**: string functions (CONCAT, REPLACE, STRLEN, etc.), type functions (COALESCE, IF, BOUND, etc.), math functions (ABS, CEIL, FLOOR, ROUND), and REGEX
- **EXISTS/NOT EXISTS**: proper subquery support with semi-joins and anti-joins
- **Platform allocators**: optional jemalloc (Linux/macOS) or mimalloc (Windows) for 10-20% faster allocations
- **Batch property APIs**: bulk reads for node properties
- **Compound predicate optimization**: `n.a = 1 AND n.b = 2` now pushes down efficiently
- **Range queries**: `find_nodes_in_range()` with zone map pruning
- **Python batch APIs**: `get_nodes_by_label(label, limit)` and `get_property_batch(ids, prop)`

### Improved

- Community detection (label propagation) is now O(E) instead of O(V^2 E), roughly 100-500x faster on large graphs
- Zone maps integrated into filter planning for automatic predicate pushdown

---

## [0.2.4b] - 2026-02-02

Fixed release workflow `--exclude` flag (requires `--workspace`)

## [0.2.4] - 2026-02-02

Benchmark-driven optimizations: lock-free reads, direct lookup APIs, and much faster filters.

### Improved

- **Lock-free concurrent reads**: hash indexes now use DashMap, 4-6x improvement under concurrent workloads
- **Direct lookup APIs**: O(1) point reads (`get_node()`, `get_node_property()`, `get_neighbors()`) that bypass query planning, 10-20x faster than equivalent MATCH queries
- **Filter performance**: single-property lookups and batch evaluation, 20-50x improvement for equality and range filters
- Better cache locality from expanded hot buffer and adjacency delta buffer sizes

---

## [0.2.3] - Unreleased

### Added

- **Succinct data structures** (feature: `succinct-indexes`): O(1) rank/select bitvectors, Elias-Fano encoding, and wavelet trees
- **Block-STM parallel execution** (feature: `block-stm`): optimistic parallel transactions with conflict detection, 3-4x speedup on batch workloads
- **Ring index for RDF** (feature: `ring-index`): compact triple storage using wavelet trees (~3x space reduction)

### Improved

- **Query plan caching**: repeated queries skip parsing and optimization entirely, 5-10x speedup for read-heavy workloads

---

## [0.2.2] - Unreleased

### Added

- **Bidirectional edge indexing**: efficient incoming edge queries via `edges_to()`, `in_degree()`, `out_degree()`
- **NUMA-aware scheduling**: work-stealing prefers same-node stealing to minimize cross-node memory access
- **Leapfrog TrieJoin**: worst-case optimal joins for cyclic patterns like triangles, O(N^1.5) vs O(N^2)

---

## [0.2.1] - Unreleased

### Added

- **Tiered version index**: hot/cold version separation for memory-efficient MVCC
- **Compressed epoch store**: zone maps for predicate pushdown on archived data
- **Epoch freeze**: compress and archive old epochs to reclaim memory

---

## [0.2.0] - 2026-02-01

Performance foundation: factorized execution to avoid Cartesian products in multi-hop queries.

### Added

- **Factorized execution**: avoids Cartesian product materialization for multi-hop queries, inspired by [Kuzu](https://kuzudb.com/)
- **Benchmarks**: multi-hop traversal and fan-out pattern benchmarks

### Changed

- Version bump to 0.2.0, focusing on performance for 0.2.x
- Switched from Python-based pre-commit to [prek](https://github.com/j178/prek) (Rust-native, faster)

---

## [0.1.4] - 2026-01-31

Label removal, direct label APIs for Python, and all query languages enabled by default.

### Added

- **REMOVE clause**: `REMOVE n:Label` and `REMOVE n.property` in GQL
- **Label APIs**: `add_node_label()`, `remove_node_label()`, `get_node_labels()` in Python
- **WAL support**: label operations now logged for durability
- **RDF transactions**: SPARQL operations now support proper commit/rollback

### Changed

- All query languages enabled by default, no feature flags needed

### Fixed

- RDF transaction rollback properly discards uncommitted changes
- npm and Go module publishing paths corrected

## [0.1.3] - 2026-01-30

CLI for database administration, Python admin APIs, adaptive query execution, and property compression.

### Added

- **CLI** (`grafeo-cli`): inspect, backup, export, manage WAL, and compact databases from the command line
- **Admin APIs**: Python bindings for `info()`, `detailed_stats()`, `schema()`, `validate()`, and WAL management
- **Adaptive execution**: runtime re-optimization when cardinality estimates deviate 3x+ from actuals
- **Property compression**: type-specific codecs (dictionary, delta, RLE) with hot buffer pattern

### Improved

- Query optimizer: projection pushdown and better join reordering
- Cardinality estimation: histogram-based with adaptive feedback
- Parsers: better edge patterns, traversal steps, and pattern matching across languages

## [0.1.2] - 2026-01-29

Comprehensive Python test suite and documentation pass across all crates.

### Added

- Comprehensive Python test suite covering LPG, RDF, all 5 query languages, and plugin integrations
- Docstring pass across all crates with tables, examples, and practical guidance

### Changed

- Core database functionality now fully operational end-to-end

## [0.1.1] - Unreleased

### Added

- **GQL parser**: full ISO/IEC 39075 standard support
- **Multi-language support**: Cypher, Gremlin, GraphQL, and SPARQL translators
- **MVCC transactions**: snapshot isolation with multi-version concurrency control
- **Index types**: hash, B-tree, trie, and adjacency indexes
- **Storage backends**: in-memory and write-ahead log
- **Python bindings**: PyO3-based API exposing full database functionality

### Changed

- Renamed project from Graphos to Grafeo

## [0.1.0] - Unreleased

### Added

- **Core architecture**: modular crate structure (grafeo-common, grafeo-core, grafeo-adapters, grafeo-engine, grafeo-python)
- **Graph models**: Labeled Property Graph (LPG) and RDF triple store
- **In-memory storage**: fast graph operations without persistence overhead

---

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
