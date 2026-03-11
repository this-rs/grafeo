# Changelog

All notable changes to Grafeo, for future reference (and enjoyment).

## [0.5.20] - 2026-03-11

### Added

- **WASM `memoryUsage()` and `importRows()`**: memory introspection and bulk row import (the DataFrame equivalent) now available in WebAssembly bindings

### Fixed

- **`SESSION SET GRAPH` / `SESSION SET SCHEMA` validation**: now errors when the target graph does not exist, matching the behavior of `USE GRAPH`; previously it silently accepted any name and fell back to the default store

## [0.5.19] - 2026-03-11

### Added

- **Graph type enforcement**: full write-path schema enforcement with node type inheritance, edge endpoint validation, UNIQUE/NOT NULL/CHECK constraints, default value injection, closed graph type guards, MERGE validator support, pattern-form syntax, SHOW commands, and Cypher `ALTER CURRENT GRAPH TYPE`
- **LOAD DATA (multi-format import)**: generalized `LOAD DATA FROM 'path' FORMAT CSV|JSONL|PARQUET [WITH HEADERS] AS variable` in GQL, with Cypher-compatible `LOAD CSV` syntax preserved; JSONL behind `jsonl-import` feature, Parquet behind `parquet-import` feature
- **Python `import_df()`**: bulk-import nodes or edges from a pandas or polars DataFrame via `db.import_df(df, 'nodes', label='Person')` or `db.import_df(df, 'edges', edge_type='KNOWS')`
- **Memory introspection**: `db.memory_usage()` returns a hierarchical breakdown of heap usage across store, indexes, MVCC chains, query caches, string pools, and buffer manager regions
- **Named graph WAL persistence**: `CREATE GRAPH` / `DROP GRAPH` and all data mutations within named graphs are now WAL-logged and recovered on restart via `SwitchGraph` context records; concurrent sessions writing to different named graphs are safely interleaved
- **Named graph snapshot persistence**: snapshot v2 format includes named graph data in `export_snapshot`, `import_snapshot`, `restore_snapshot`, `save`, and `to_memory`; v1 snapshots remain backward-compatible
- **SHOW GRAPHS**: `SHOW GRAPHS` lists all named graphs in the database, complementing existing `SHOW NODE TYPES` / `SHOW EDGE TYPES`
- **RDF persistence**: SPARQL INSERT/DELETE/CLEAR/CREATE/DROP operations are now WAL-logged and recovered on restart; snapshot export/import includes RDF triples and RDF named graphs
- **Cross-graph transactions**: `USE GRAPH` and `SESSION SET GRAPH` now work within active transactions; commit/rollback/savepoint operations apply atomically across all touched graphs
- **GrafeoDB graph context**: one-shot `db.execute()` calls now persist `USE GRAPH` context across calls; `current_graph()` and `set_current_graph()` public API for programmatic access
- **WASM batch import**: `importLpg()` and `importRdf()` methods for bulk-loading structured LPG nodes/edges and RDF triples in a single call, with index-relative edge references and typed literal support

### Fixed

- **Named graph data isolation** ([#133](https://github.com/GrafeoDB/grafeo/issues/133)): `USE GRAPH`, `SESSION SET SCHEMA`, and `SESSION SET GRAPH` now correctly route all queries and mutations to the selected named graph instead of always using the default store; query cache keys include the active graph name to prevent cross-graph cache hits; dropping the active graph resets the session to default
- **OPTIONAL MATCH WHERE pushdown**: right-side predicates are now correctly pushed into the join instead of filtering out NULL rows, with dedicated cost/cardinality estimation for LeftJoin
- **Cypher COUNT(expr) NULL skipping**: `COUNT(expr)` now correctly skips NULLs (using `CountNonNull`), matching `COUNT(*)` which counts all rows
- **Vector validity bitmap fix**: consecutive NULL pushes to the same column no longer silently drop null bits, fixing incorrect empty-string results in SPARQL OPTIONAL and RDF left joins

### Improved

- **GQL translator submodules**: split `gql.rs` into `gql/mod.rs`, `expression.rs`, `pattern.rs`, `aggregate.rs` for maintainability
- **Wildcard imports lint**: re-enabled `clippy::wildcard_imports` as warning; replaced `use super::*` in LPG planner submodules with explicit imports
- **Unwrap reduction**: replaced production `.expect()` calls with `Result`/`?` propagation in session initialization, persistence, and WAL recovery paths

## [0.5.18] - 2026-03-09

Query language compliance improvements, expanded test coverage, and Deriva compatibility fixes.

### Added

- **Extensive spec test suites**: 8 Cypher spec modules (reading clauses, return/ordering, writing clauses, patterns, expressions, functions, types, admin/schema) and 12 GQL spec modules (data query, patterns, mutations, expressions, functions, types, schema DDL, sessions, procedures, predicates, subqueries, composite) covering 1,300+ test cases
- **Cypher exotic integration tests**: 67 end-to-end Cypher tests covering exotic query patterns (NOT EXISTS subqueries, any() predicates, reduce, list comprehensions, collect with maps, OPTIONAL MATCH, CASE WHEN, elementId, multi-label MATCH, etc.)

### Fixed (Cypher)

- **CALL subquery variable scope**: `CALL { WITH p MATCH (p)-[:KNOWS]->(q) RETURN q.name AS friend }` now correctly resolves inner RETURN columns in the outer query instead of returning NULL
- **RETURN after DELETE**: `DETACH DELETE n RETURN count(n)` no longer fails with "Variable not found"; delete operators pass through input rows for downstream aggregation
- **Inline MERGE with relationship SET**: `MERGE (a:L {id:1})-[r:REL]->(b:L {id:2}) SET r.weight = 0.5` decomposes inline node patterns into chained MERGE operations
- **WITH \* wildcard**: `WITH *` now correctly passes all bound variables through instead of failing to parse
- **DoubleDash edge patterns**: undirected relationship patterns using `--` are now parsed alongside `-[]-` syntax

### Fixed (GQL)

- **CALL { subquery }**: `CALL { ... } RETURN ...` is now recognized as a query-level clause instead of a procedure call
- **WITH + LET bindings**: LET clauses immediately after WITH are now parsed and attached correctly
- **String concatenation operator**: `||` (CONCAT) is now supported in arithmetic expressions
- **Inline MERGE with relationship SET**: same decomposition fix applied to the GQL translator

### Fixed

- **Multiple NOT EXISTS subqueries**: queries with two or more `NOT EXISTS { ... }` predicates no longer fail with variable-not-found errors
- **SET property transaction rollback**: `SET n.prop = value` changes within a transaction are now correctly undone on `ROLLBACK`
- **Label mutation rollback**: `SET n:Label` and `REMOVE n:Label` changes are correctly undone on `ROLLBACK`
- **MERGE ON MATCH SET rollback**: properties updated via `MERGE ... ON MATCH SET` are correctly restored on `ROLLBACK`
- **Savepoint partial rollback**: `ROLLBACK TO SAVEPOINT` now undoes property and label mutations made after the savepoint while preserving earlier changes
- **NPM package missing native binaries** ([#128](https://github.com/GrafeoDB/grafeo/issues/128)): `@grafeo-db/js` now publishes per-platform packages (`@grafeo-db/js-darwin-arm64`, `@grafeo-db/js-linux-x64-gnu`, etc.) as `optionalDependencies`, so `npm install` and `bun install` pull the correct native binary automatically

## [0.5.17] - 2026-03-09

Cypher query execution bug fixes for Deriva compatibility.

### Fixed

- **Correlated EXISTS subqueries**: `NOT EXISTS { MATCH (a)-[r]->(b) WHERE type(r) = 'X' }` now correctly plans via semi-join instead of failing with "Unsupported EXISTS subquery pattern"
- **CASE WHEN in aggregates**: `sum(CASE WHEN ... THEN 1 ELSE 0 END)` resolves correctly inside aggregate functions
- **any()/all()/none()/single() with IN list**: `any(lbl IN labels(n) WHERE lbl IN ['A', 'B'])` now evaluates the IN operator correctly in list predicate contexts
- **CASE WHEN in reduce()**: `reduce(acc = 0, x IN vals | CASE WHEN x > acc THEN x ELSE acc END)` evaluates CASE expressions with both accumulator and item variable bindings

## [0.5.16] - 2026-03-08

Performance enhancements, bug fixes and Rust examples

### Added

- **LOAD CSV**: `LOAD CSV [WITH HEADERS] FROM 'path' AS row [FIELDTERMINATOR '\t']` in Cypher, with inline CSV parser supporting quoted fields, `file:///` URIs, and custom delimiters
- **Cypher schema DDL**: `CREATE/DROP INDEX`, `CREATE/DROP CONSTRAINT`, `SHOW INDEXES`, `SHOW CONSTRAINTS`
- **Relationship WHERE**: inline predicates on relationship patterns (`-[r WHERE r.since > 2020]->`)
- **Temporal map constructors**: `date({year:2024, month:3})`, `time({hour:14})`, `datetime(...)`, `duration({years:1, months:2, days:3})`
- **PROFILE statement**: `PROFILE MATCH ... RETURN ...` executes the query and returns per-operator metrics (rows, self-time, call counts) for GQL and Cypher
- **Rust examples**: 7 runnable examples in `examples/rust/` covering the core API (basic queries, transactions, parameterized queries, vector search, graph algorithms, WAL persistence, multi-language dispatch)
- **Plan cache invalidation**: query plan cache is automatically cleared after DDL operations (CREATE/DROP INDEX, TYPE, CONSTRAINT, etc.), with manual `clear_plan_cache()` API on `GrafeoDB` and `Session`
- **Cache invalidation counter**: `CacheStats.invalidations` tracks how often DDL clears the plan cache

### Improved

- **Cost model calibration**: recursive plan costing, statistics-aware IO estimation, actual child cardinalities for joins, multi-edge-type expand costing
- **Supply chain audit**: replaced `cargo audit` CI job with `cargo-deny` (licenses, advisories, bans, source verification)
- **Benchmark regression detection**: PRs now run all three criterion suites (arena, index, query) and fail on >10% regression via `benchmark-action`
- **Examples CI**: added `cargo build -p grafeo-examples` to CI checks

### Fixed

- **GQL `-->` shorthand**: parser recognizes `-->` as a directed outgoing edge instead of splitting into `--` and `>`
- **EXISTS bare patterns**: `EXISTS { (a)-[r]->(b) }` without explicit MATCH keyword now works in GQL and Cypher
- **CASE WHEN in aggregates**: expressions like `sum(CASE WHEN ... THEN 1 ELSE 0 END)` resolve correctly in the LPG planner
- **SPARQL parameters**: `execute_sparql_with_params()` now substitutes `$param` values instead of ignoring them

## [0.5.15] - 2026-03-07

Full ecosystem feature profiles reworks and several graph database nice to haves

### Added

- **Ecosystem feature profiles**: `embedded`, `browser`, `server` named profiles across all crates. `storage` convenience group (`wal` + `spill` + `mmap`)
- **WASM multi-variant builds**: AI variant (531 KB gzip) and lite variant (513 KB gzip) via `build-wasm-all.sh`, with `regex-lite` for smaller binaries
- **Savepoints and nested transactions**: `SAVEPOINT`/`ROLLBACK TO`/`RELEASE`, inner `START TRANSACTION` auto-creates savepoints
- **Correlated subqueries**: `EXISTS { ... }`, `COUNT { ... }`, `VALUE { ... }` in WHERE/RETURN
- **Subpath variable binding**: `(p = (a)-[e]->(b)){2,5}` with `length(p)`, `nodes(p)`, `edges(p)`
- **Type system extensions**: `LIST<T>` typed lists with coercion, `IS TYPED RECORD/PATH/GRAPH` predicates, `path()` constructor
- **Graph DDL**: `CREATE GRAPH g2 LIKE g1`, `AS COPY OF`, `CREATE GRAPH g ANY/OPEN`
- **GQLSTATUS diagnostics**: ISO sec 23 status codes and diagnostic records on all query results
- **Catalog procedures**: `CALL db.labels()`, `db.relationshipTypes()`, `db.propertyKeys()` with YIELD
- **Python DataFrame bridge**: `result.to_pandas()`, `result.to_polars()`, `db.nodes_df()`, `db.edges_df()` for zero-friction data science integration

### Fixed

- **Temporal functions**: `local_time()`, `local_datetime()`, `zoned_datetime()` constructors, `date_trunc()` truncation
- **Aggregate separators**: `LISTAGG` and `GROUP_CONCAT` with custom separators and per-language defaults

### Changed

- **Default profile**: facade crate default changed from `full` to `embedded`. All binding crates follow
- **WASM**: default changed to `browser` profile, binary size reduced from 1,001 KB to 513 KB gzipped (49%)

## [0.5.14] - 2026-03-06

Moving crates and lots of small improvements & fixes

### Added

- **EXPLAIN statement**: `EXPLAIN <query>` in GQL and Cypher returns the optimized logical plan tree with pushdown hints (`[index: prop]`, `[range: prop]`, `[label-first]`)
- **WASM size optimization**: `wasm-opt -Oz` applied during release builds
- **NetworkX bridge**: `adj` property and `subgraph(nodes)` method
- **SPARQL built-in functions**: date/time (NOW, YEAR, MONTH, ...), hash (MD5, SHA1, SHA256, SHA384, SHA512), RDF term (LANG, DATATYPE, IRI, BNODE, ...) and RAND
- **GROUP_CONCAT / SAMPLE aggregates**: proper implementations replacing the previous Collect stub

### Fixed

- **Auto-commit for mutations**: single-shot `execute()` calls with INSERT/DELETE/SET now auto-commit instead of silently discarding changes
- **WAL persistence for queries**: mutations via GQL/Cypher now persist to WAL (previously only the CRUD API did)
- **WAL property removal**: `remove_node_property` and `remove_edge_property` now log to WAL
- **Cypher count(\*)**: parses correctly when `count` is tokenized as a keyword
- **SPARQL unary plus**: treated as identity instead of `NOT`
- **CLI fixes**: `data dump`/`data load` now work (JSON Lines), `compact` performs real compaction, `index list` shows per-index details, nonexistent databases error instead of being silently created
- **WASM test suite**: fixed compilation and runtime panics on wasm32

### Changed

- **Node.js `nodeCount`/`edgeCount`**: changed from getter properties to methods (`db.nodeCount()`)
- **Arena allocator**: returns `Result<T, AllocError>` instead of panicking on allocation failure
- **Planner refactor**: split into `planner/lpg/` and `planner/rdf/` with shared operator builders
- **Translator refactor**: shared plan-builder functions extracted into `translators/common.rs`, all 7 translators moved into `query/translators/`
- **Dependency cleanup**: removed unused deps, replaced ahash with foldhash, narrowed tokio features

## [0.5.13] - 2026-03-04

Big language compliance push, schema DDL, time-travel and named graphs.

### Improved

- **GQL**: full compliance with ISO/IEC 39075:2024, covering all features practical for a graph database
- **Cypher**: improved openCypher v9 compliance, plus pattern comprehensions, CALL subqueries, FOREACH
- **SPARQL**: improved W3C SPARQL 1.1 compliance (no 1.2/SPARQL Star yet)

#### Infrastructure

- **LPG named graphs**: multi-graph support with per-graph storage, labels, indexes and MVCC versioning. Public API: `create_graph()`, `drop_graph()`, `list_graphs()`
- **Apply operator**: correlated subquery execution for CALL, VALUE, NEXT and pattern comprehensions
- **Temporal types**: `Date`, `Time`, `Duration` with ISO 8601 parsing, arithmetic and component extraction. JSON encoding as `{"$date": "..."}` etc. Python round-trips via `datetime.date`/`datetime.time`

#### Schema / DDL System

- **Full schema DDL via GQL**: `CREATE`/`DROP`/`ALTER` for NODE TYPE, EDGE TYPE, GRAPH TYPE, INDEX, CONSTRAINT and SCHEMA, with `OR REPLACE`, `IF NOT EXISTS`/`IF EXISTS` and WAL persistence
- **Type definitions**: `CREATE NODE TYPE Person (name STRING NOT NULL, age INT64)` with property types and nullability
- **Index DDL**: `CREATE INDEX ... FOR (n:Label) ON (n.property) [USING TEXT|VECTOR|BTREE]`
- **Constraint enforcement**: UNIQUE, NOT NULL, NODE KEY, EXISTS constraints validated on writes

#### Time-Travel

- **Epoch-based time-travel**: `execute_at_epoch(query, epoch)` runs any query against a historical snapshot. Also available as a persistent session mode via `set_viewing_epoch()` or `SESSION SET PARAMETER viewing_epoch = <n>`
- **Version history**: `get_node_history(id)` and `get_edge_history(id)` return all versions with creation/deletion epochs

#### GQL Spec Compliance (78% to ~97%)

- **New syntax**: LIKE operator, CAST to temporal types, SET map operations (`= {map}`, `+= {map}`), NODETACH DELETE, RETURN \*/WITH \*, list comprehensions and predicates in RETURN, transaction characteristics, zoned temporal types, ALTER DDL, CREATE GRAPH TYPED, stored procedures
- **List property storage**: `reduce()` and list operations now work correctly after INSERT with list-valued properties

### Fixed

- **Time-travel scans**: now use pure epoch-based visibility instead of transaction-aware checks that bypassed epoch filtering
- **LIKE parser**: token existed but was never consumed as an infix operator
- **RETURN * binder**: was incorrectly rejected as an undefined variable
- **List comprehensions in projections**: planner now handles these in RETURN clauses
- **Cypher fixes**: standalone DELETE/SET/REMOVE error messages, `^` power operator, anonymous variable name collisions
- **Temporal comparison**: 10 compare_values paths now handle Date/Time/Timestamp (previously returned false silently)

### Improved

- **Test coverage**: 80+ GQL parser tests (was 44), 137 Python GQL compliance tests (was 100), new SPARQL and Cypher compliance suites

## [0.5.12] - 2026-03-02

Two-phase commit, snapshot restore, EXISTS subqueries.

### Added

- **PreparedCommit**: two-phase commit via `session.prepare_commit()`, inspect pending mutations and attach metadata before finalizing
- **Atomic snapshot restore**: `db.restore_snapshot(data)` replaces the database in place, with full pre-validation (store unchanged on error)
- **EXISTS subqueries** (GQL, Cypher): complex inner patterns with multi-hop traversals, property filters and label constraints via semi-join rewrite

### Fixed

- **SET on edge variables**: Cypher translator now correctly handles SET when targeting an edge variable

### Improved

- **Variable-length path traversal**: BFS path tracking uses shared-prefix `Rc` segments instead of cloning full vectors, reducing per-edge cost from O(depth) to O(1)

## [0.5.11] - 2026-03-02

Pluggable storage traits, query language compliance, UNION support.

### Added

- **Pluggable storage**: `GraphStore`/`GraphStoreMut` traits decouple all query operators and algorithms from `LpgStore`. Use `GrafeoDB::with_store(Arc<dyn GraphStoreMut>, Config)` to plug in any backend
- **Type-safe WAL**: `WalEntry` trait and `TypedWal<R>` wrapper constrain WAL record types at compile time, preventing cross-model logging
- **Query language compliance tests**: spec-level integration tests for all 6 query languages
- **Cypher UNION / UNION ALL**: combining query results with duplicate elimination or preservation
- **GQL MERGE on relationships**: `MERGE (a)-[r:TYPE]->(b)` with idempotent edge creation
- **Gremlin traversal steps**: `and()`, `or()`, `not()`, `where()`, `filter()`, `choose()`, `optional()`, `union()`, `coalesce()` and more
- **SPARQL improvements**: DISTINCT, HAVING, FILTER NOT EXISTS / EXISTS

## [0.5.10] - 2026-02-29

Robustness: bidirectional shortest path, crash recovery tests, stress tests.

### Added

- **Skip index for adjacency chunks**: compressed cold chunks maintain a zone-map skip index. `contains_edge(src, dst)` provides O(log n) point lookups; `edges_in_range(src, min, max)` supports efficient range queries
- **Bidirectional BFS shortest path**: meet-in-the-middle BFS expanding smaller frontier first, reducing search space from O(b^d) to O(b^(d/2))

### Improved

- **Crash recovery tests**: 7 deterministic crash injection tests verifying WAL recovery at every crash point
- **Concurrent stress tests**: 6 multi-threaded tests covering concurrent writers, mixed read/write, transaction conflicts, epoch pressure and rapid session lifecycle
- **Hardened panic messages**: ~50 bare `unwrap()` calls converted to `expect()` with invariant descriptions; no behavioral change

## [0.5.9] - 2026-02-28

Compact property storage, snapshot validation, crash injection framework.

### Added

- **Snapshot validation**: `import_snapshot()` pre-validates everything before inserting: rejects duplicate IDs and dangling edge references
- **Crash injection framework**: feature-gated `maybe_crash()` / `with_crash_at()` for deterministic recovery testing, with three WAL crash points. Zero overhead when disabled
- **Backward compatibility tests**: pinned v1 snapshot fixture with 8 regression tests for format stability

### Fixed

- **WASM build with `getrandom` 0.4**: added `wasm_js` crate feature for 0.4.x on wasm32 targets
- **WASM binary size regression**: disabled transitive engine features in bindings-common, reducing WASM gzip from 974 KB to 744 KB

### Improved

- **Compact property storage**: property maps switched from `BTreeMap` to `SmallVec<4>`, so nodes with 4 or fewer properties avoid heap allocation
- **Cost model per-type fanout**: the optimizer now tracks per-edge-type average degree instead of a single global estimate

## [0.5.8] - 2026-02-22

Shared bindings crate, unified query dispatch, Node.js/WASM API expansion.

### Added

- **`grafeo-bindings-common` crate**: shared entity extraction, error classification and JSON conversion for all four bindings (Python, Node.js, C, WASM)
- **Unified query dispatch**: `execute_language(query, "gql"|"cypher"|"sparql"|...)` replaces 18 per-language functions
- **Node.js API parity**: property removal, label management, `info()`, `schema()`, `version()` and transaction isolation levels now match the Python binding
- **WASM expansion**: parameterized queries, per-language convenience methods, proper feature gating
- **Batch edge creation**: `batch_create_edges()` with single lock acquisition for bulk imports

### Improved

- **Incremental statistics**: `compute_statistics()` reads atomic delta counters instead of scanning all entities, reducing refresh from O(n+m) to O(|labels|+|edge_types|)
- **Cost model uses real fanout**: optimizer derives average edge fanout from actual graph statistics instead of a hardcoded 10.0

## [0.5.7] - 2026-02-19

UNWIND property access fix, `algos` feature flag.

### Fixed

- **UNWIND mutation property access**: map property access like `e.src`, `e.weight` in CREATE/SET now resolves correctly. Previously only column references and constants worked, so map properties came back as NULL

### Added

- **`algos` feature flag**: graph algorithms gated behind `algos` (included in `full`). Reduces compile time and binary size when algorithms are not needed

## [0.5.6] - 2026-02-18

UNWIND/FOR list expansion, embedding model config, zero unsafe in property storage.

### Added

- **UNWIND clause**: expand lists into rows for batch processing. Works with literals, parameters (`UNWIND $items AS x`) and vectors. Combine with MATCH + INSERT for bulk edge creation
- **FOR statement** (GQL standard): `FOR x IN [1, 2, 3] RETURN x`, with `WITH ORDINALITY` (1-based) and `WITH OFFSET` (0-based) index tracking
- **Text index auto-sync**: text indexes update automatically on property changes, no manual rebuild needed. WASM bindings added too
- **SPARQL COPY/MOVE/ADD**: graph management operators with source-existence validation and SILENT support
- **Embedding model config**: 3 presets (MiniLM-L6-v2, MiniLM-L12-v2, BGE-small-en-v1.5) with HuggingFace auto-download. Exposed in Python and Node.js
- **Native SSSP procedure**: `CALL grafeo.sssp('node_name', 'weight')` for LDBC Graphanalytics compatibility

### Fixed

- **UNWIND scoping**: MATCH clauses after UNWIND now correctly receive UNWIND variables, scalar values no longer resolve as node IDs and `Value::Vector` is handled alongside `Value::List`
- **`RETURN n` returns full entities**: `MATCH (n) RETURN n` now returns `{_id, _labels, ...properties}` instead of a bare integer ID
- **GQL lexer UTF-8 panic**: multi-byte characters no longer cause boundary panics
- **Scalar column tracking**: Gremlin `.values()`, `.count()` and GQL `WITH expr AS alias` no longer return NULL
- **Vector index rebuild after drop**: works without the old index, infers dimensions from data

### Improved

- **Zero unsafe in property storage**: replaced final `transmute_copy` calls with safe `EntityId` conversions
- **Statistics access**: `statistics()` returns `Arc<Statistics>` instead of deep-cloning on every planner invocation
- **Entity resolution**: moved from 6-site post-processing into the ProjectOperator pipeline for single-pass resolution

## [0.5.5] - 2026-02-16

Filter pushdown, query error positions, transaction fixes.

### Added

- **Filter pushdown**: equality predicates on labeled scans are pushed to the store level. Compound predicates like `WHERE n.name = 'Alix' AND n.age > 30` correctly split: equality pushed down, range kept as post-filter
- **Query error positions**: all six parsers now produce errors with line/column positions and source-caret display

### Fixed

- **Transaction edge type visibility**: edges created within a transaction are now visible to subsequent queries in the same transaction
- **SPARQL INSERT/DELETE DATA with GRAPH clause**: triples now route to the named graph instead of the default graph
- **Compound predicate correctness**: filter pushdown no longer drops non-equality parts of compound predicates

## [0.5.4] - 2026-02-15

### Fixed

- **Multi-pattern CREATE**: `CREATE (:A {id: 'x'}), (:B {id: 'y'})` now creates all nodes instead of only the first

## [0.5.3] - 2026-02-13

### Improved

- **Query error quality**: translator errors now produce `QueryError` with semantic error codes (`GRAFEO-Q002`) instead of generic internal errors. More actionable messages
- **GraphQL range filters**: operator suffixes (`_gt`, `_lt`, etc.) now work on direct query arguments, not just `where` clauses

### Fixed

- **SPARQL `FILTER NOT EXISTS`**: parser now recognizes NOT EXISTS/EXISTS, producing correct anti-join/semi-join plans
- **SPARQL `FILTER REGEX`**: REGEX evaluation was missing from the RDF planner (parser/translator already supported it)

## [0.5.2] - 2026-02-13

### Added

- **CALL procedure support**: invoke any of the 22 built-in graph algorithms from query strings: `CALL grafeo.<algorithm>() [YIELD columns]`. Supported in GQL, Cypher and SQL/PGQ
- **Map literal arguments**: `CALL grafeo.pagerank({damping: 0.85, max_iterations: 20})`
- **Procedure listing**: `CALL grafeo.procedures()` returns all available procedures

## [0.5.1] - 2026-02-12

Hybrid search, built-in embeddings, change data capture. The features that make grafeo-memory work.

### Added

- **BM25 text search** (`text-index`): inverted indexes on string properties with BM25 scoring. Built-in tokenizer with Unicode word boundaries, lowercasing and stop word removal
- **Hybrid search** (`hybrid-search`): combine BM25 text + HNSW vector similarity via RRF or weighted fusion. Single `hybrid_search()` call in Python and Node.js
- **Built-in embeddings** (`embed`, opt-in): in-process embedding generation via ONNX Runtime. Load any `.onnx` model, call `embed_text()`. Adds ~17MB, off by default
- **Change data capture** (`cdc`): track all mutations with before/after property snapshots. Query via `history()`, `history_since()`, `changes_between()`. Available in Python and Node.js

## [0.5.0] - 2026-02-11

Error codes, query timeouts, auto-GC, ~50% memory savings for vector workloads.

### Added

- **Standardized error codes**: all errors carry `GRAFEO-XXXX` codes (Q = query, T = transaction, S = storage, V = validation, X = internal) with `error_code()` and `is_retryable()`
- **Query timeout**: `Config::default().with_query_timeout(Duration::from_secs(30))` stops long-running queries cleanly
- **MVCC auto-GC**: version chains garbage-collected every N commits (default 100, configurable). Also `db.gc()` for manual control

### Improved

- **Topology-only HNSW**: vectors no longer duplicated inside the index; reads on-demand via `VectorAccessor` trait. ~50% memory reduction for vector workloads

## [0.4.4] - 2026-02-11

SQL/PGQ queries, MMR search for RAG, auto-syncing vector indexes, CLI overhaul.

### Added

- **SQL/PGQ support**: query with SQL:2023 syntax, `SELECT ... FROM GRAPH_TABLE (MATCH ... COLUMNS ...)`. Includes path functions, DDL and all bindings
- **MMR search**: diverse, relevant results for RAG pipelines via `mmr_search()` with tunable relevance/diversity balance
- **Filtered vector search**: property equality filters on `vector_search()`, `batch_vector_search()` and `mmr_search()` using pre-computed allowlists for efficient HNSW traversal
- **Incremental vector indexing**: indexes stay in sync automatically as nodes change
- **CLI overhaul**: interactive shell with transactions, meta-commands (`:schema`, `:info`, `:stats`), persistent history, CSV output. Install via `cargo install`, `pip install` or `npm install -g`
- **Configurable cardinality estimation**: tune 9 selectivity parameters via `SelectivityConfig`
- **AdminService trait**: unified introspection and maintenance: `info()`, `detailed_stats()`, `schema()`, `validate()`, `wal_status()`
- **GQL `IN` operator**: `WHERE n.name IN ['Alix', 'Gus']`
- **String escape sequences**: `\'`, `\"`, `\\`, `\n`, `\r`, `\t` in GQL, Cypher, SQL/PGQ

### Fixed

- **Node.js ID validation**: rejects negative, NaN, Infinity and values above `MAX_SAFE_INTEGER`

### Changed

- **Python CLI removed**: replaced by the unified `grafeo-cli` Rust binary

## [0.4.3] - 2026-02-08

Per-database graph model selection, snapshot export/import, expanded WASM APIs.

### Added

- **Database creation options**: choose LPG or RDF per database, configure durability mode, toggle schema constraints
- **Snapshot export/import**: serialize to binary snapshots for backups or WASM persistence via IndexedDB
- **WASM expansion**: `executeWithLanguage()`, `exportSnapshot()`/`importSnapshot()`, `schema()`

## [0.4.2] - 2026-02-08

Grafeo now runs in the browser. WebAssembly bindings with TypeScript definitions at 660 KB gzipped.

### Added

- **WebAssembly bindings** (`@grafeo-db/wasm`): `execute()`, `executeRaw()`, `nodeCount()`, `edgeCount()`, full TypeScript definitions. 660 KB gzipped (target was <800 KB)
- **Feature-gated platform subsystems**: `parallel`, `spill`, `mmap`, `wal` are opt-in, making wasm32 compilation straightforward

## [0.4.1] - 2026-02-08

Go and C bindings. Grafeo now embeds in pretty much any language.

### Added

- **Go bindings** (`github.com/GrafeoDB/grafeo`): full CRUD, multi-language queries, ACID transactions, vector search, batch operations, admin APIs
- **C FFI layer** (`grafeo-c`): C-compatible ABI for embedding Grafeo in any language

## [0.4.0] - 2026-02-07

Node.js/TypeScript bindings, Python vector search and transaction isolation.

### Added

- **Node.js/TypeScript bindings** (`@grafeo-db/js`): full CRUD, async queries across all 5 languages, transactions, native type mapping, TypeScript definitions
- **Python vector support**: pass `list[float]` directly, `grafeo.vector()`, distance functions in GQL, HNSW indexes, k-NN search
- **Python transaction isolation**: `"read_committed"`, `"snapshot"` or `"serializable"` per transaction
- **Batch vector APIs**: `batch_create_nodes()` and `batch_vector_search()` for Python and Node.js

### Fixed

- GQL INSERT with list or `vector()` properties no longer silently drops values
- Multi-hop MATCH queries (3+ hops) no longer return duplicate rows
- GQL multi-hop patterns now correctly filter intermediate nodes by label
- Transaction `execute()` rejects queries after commit/rollback

### Improved

- **HNSW recall and speed**: Vamana-style diversity pruning, pre-normalized cosine vectors, pre-allocated structures
- Query optimizer uses actual store statistics instead of hardcoded defaults

## [0.3.4] - 2026-02-06

Query timing, "did you mean?" suggestions, Python pagination.

### Added

- **Query performance metrics**: every result includes `execution_time_ms` and `rows_scanned`
- **"Did you mean?" suggestions**: typo in a variable or label? Grafeo suggests the closest match
- **Python pagination**: `get_nodes_by_label()` supports `offset` for paging

## [0.3.3] - Unreleased

### Added

- **VectorJoin operator**: graph traversal + vector similarity in a single query
- **Vector zone maps**: skips irrelevant data blocks during vector search
- **Product quantization**: 8-32x memory compression with ~90% recall retention
- **Memory-mapped vector storage**: disk-backed with LRU caching for large datasets
- **Python quantization API**: `ScalarQuantizer`, `ProductQuantizer`, `BinaryQuantizer`

## [0.3.2] - Unreleased

### Added

- **Selective property loading**: fetch only the properties you need, much faster for wide nodes
- **Parallel node scan**: 3-8x speedup on large scans (10K+ nodes) across CPU cores

## [0.3.1] - Unreleased

### Added

- **Vector quantization**: f32 to u8 (scalar) or 1-bit (binary) compression with quantized HNSW search + exact rescoring
- **SIMD acceleration**: 4-8x faster distance computations; auto-selects AVX2/FMA, SSE or NEON
- **Vector batch operations**: `batch_insert()` and `batch_search()` for bulk loading
- **VectorScan operators**: vector similarity integrated into the query execution engine
- **Adaptive WAL flusher**: self-tuning background flush based on actual disk speed
- **Fingerprinted hash index**: sharded with 48-bit fingerprints for near-instant miss detection

## [0.3.0] - Unreleased

Vectors are a first-class type. Graph + vector hybrid queries let you do things no pure vector database can.

### Added

- **Vector type**: native storage with dimension-aware schema validation
- **Distance functions**: cosine, euclidean, dot product, manhattan
- **HNSW index**: O(log n) approximate nearest neighbor with tunable presets (`high_recall()`, `fast()`). Also brute-force k-NN with optional predicate filtering
- **GQL vector syntax**: `vector([...])` literals, distance functions, `CREATE VECTOR INDEX`
- **SPARQL vector functions**: `COSINE_SIMILARITY()`, `EUCLIDEAN_DISTANCE()`, `DOT_PRODUCT()`, `MANHATTAN_DISTANCE()`
- **Serializable snapshot isolation**: `ReadCommitted`, `SnapshotIsolation` or `Serializable` per transaction

---

## [0.2.7] - 2026-02-05

Parallel execution primitives, second-chance LRU cache.

### Added

- **Second-chance LRU cache**: lock-free access marking for concurrent workloads
- **Parallel fold-reduce**: `parallel_count`, `parallel_sum`, `parallel_stats`, `parallel_partition` and a composable collector trait

---

## [0.2.6] - 2026-02-04

Zone map filtering, clustering coefficient, faster batch reads.

### Added

- **Local clustering coefficient**: triangle counting with parallel execution
- **Chunk-level zone map filtering**: skip entire data chunks when predicates can't match

### Improved

- Batch property retrieval acquires a single lock instead of one per entity

---

## [0.2.5] - 2026-02-03

Full SPARQL functions, platform allocators, batch property APIs.

### Added

- **Full SPARQL function coverage**: string, type, math functions and REGEX
- **EXISTS/NOT EXISTS**: semi-join and anti-join subqueries
- **Platform allocators**: optional jemalloc (Linux/macOS) or mimalloc (Windows) for 10-20% faster allocations
- **Batch property APIs**, compound predicate pushdown, range queries with zone map pruning

### Improved

- Community detection now O(E) instead of O(V^2 E), roughly 100-500x faster on large graphs

---

## [0.2.4b] - 2026-02-02

Fixed release workflow `--exclude` flag (requires `--workspace`).

## [0.2.4] - 2026-02-02

Benchmark-driven optimizations: lock-free reads, direct lookups, faster filters.

### Improved

- **Lock-free concurrent reads**: hash indexes use DashMap, 4-6x improvement under concurrency
- **Direct lookup APIs**: O(1) point reads bypassing query planning, 10-20x faster than MATCH
- **Filter performance**: 20-50x improvement for equality and range filters

---

## [0.2.3] - Unreleased

### Added

- **Succinct data structures** (`succinct-indexes`): O(1) rank/select bitvectors, Elias-Fano, wavelet trees
- **Block-STM parallel execution** (`block-stm`): optimistic parallel transactions, 3-4x batch speedup
- **Ring index for RDF** (`ring-index`): compact triple storage via wavelet trees (~3x space reduction)
- **Query plan caching**: repeated queries skip parsing and optimization, 5-10x speedup

---

## [0.2.2] - Unreleased

### Added

- **Bidirectional edge indexing**: `edges_to()`, `in_degree()`, `out_degree()`
- **NUMA-aware scheduling**: work-stealing prefers same-node to minimize cross-node memory access
- **Leapfrog TrieJoin**: worst-case optimal joins for cyclic patterns, O(N^1.5) vs O(N^2)

---

## [0.2.1] - Unreleased

### Added

- **Tiered version index**: hot/cold separation for memory-efficient MVCC
- **Compressed epoch store**: zone maps for predicate pushdown on archived data
- **Epoch freeze**: compress and archive old epochs to reclaim memory

---

## [0.2.0] - 2026-02-01

Performance foundation: factorized execution to avoid Cartesian products in multi-hop queries.

### Added

- **Factorized execution**: avoids Cartesian product materialization, inspired by [Kuzu](https://kuzudb.com/)

### Changed

- Switched from Python-based pre-commit to [prek](https://github.com/j178/prek) (Rust-native, faster)

---

## [0.1.4] - 2026-01-31

Label removal, Python label APIs, all languages on by default.

### Added

- **REMOVE clause**: `REMOVE n:Label` and `REMOVE n.property` in GQL
- **Label APIs**: `add_node_label()`, `remove_node_label()`, `get_node_labels()` in Python
- **RDF transactions**: SPARQL now supports proper commit/rollback

### Changed

- All query languages enabled by default, no feature flags needed

## [0.1.3] - 2026-01-30

CLI, Python admin APIs, adaptive execution, property compression.

### Added

- **CLI** (`grafeo-cli`): inspect, backup, export, manage WAL, compact databases
- **Admin APIs**: Python bindings for `info()`, `detailed_stats()`, `schema()`, `validate()`
- **Adaptive execution**: runtime re-optimization when cardinality estimates deviate 3x+ from actuals
- **Property compression**: dictionary, delta, RLE codecs with hot buffer pattern

### Improved

- Query optimizer: projection pushdown, better join reordering, histogram-based cardinality estimation

## [0.1.2] - 2026-01-29

Python test suite, documentation pass.

### Added

- Comprehensive Python test suite covering LPG, RDF, all 5 query languages and plugins
- Docstring pass across all crates

## [0.1.1] - Unreleased

### Added

- **GQL parser**: full ISO/IEC 39075 support
- **Multi-language**: Cypher, Gremlin, GraphQL, SPARQL translators
- **MVCC transactions**: snapshot isolation
- **Indexes**: hash, B-tree, trie, adjacency
- **Storage**: in-memory and write-ahead log
- **Python bindings**: PyO3-based API

### Changed

- Renamed from Graphos to Grafeo, reset version to 0.1.0

## [0.1.0] - Unreleased

### Added

- **Core architecture**: modular crate structure (common, core, adapters, engine, python)
- **Graph models**: LPG and RDF triple store
- **In-memory storage**: fast graph operations without persistence overhead

---

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
