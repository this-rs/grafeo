# Obrain Changelog

All notable changes to Obrain will be documented in this file.

This project was forked from [GrafeoDB/grafeo](https://github.com/GrafeoDB/grafeo) and rebranded as **Obrain** ([obrain.dev](https://obrain.dev)). The upstream Grafeo changelog is preserved in `CHANGELOG-grafeo.md` for historical reference.

## [0.2.0] - 2026-04-22

### 🌋 Substrate cutover — topology-as-storage

Obrain's primary storage backend is now **`SubstrateStore`** — a single
mmap'd directory where the topology *is* the storage. `LpgStore` (the
legacy in-memory graph + bincode snapshots) is deprecated and will be
removed in a subsequent release.

See [RFC-SUBSTRATE](docs/rfc/substrate/format-spec.md) and the
[Migration guide](docs/MIGRATION.md).

### Breaking

- **File format**: new `substrate.obrain/` directory layout (32 B node
  records, 30 B edge records, 4 KiB property pages, string heap with
  `(offset, len)`). Existing `.obrain` files must be migrated via
  `obrain-migrate --upgrade`. A 7-day rollback window is supported
  using the `legacy-v0.1` release binary.
- **Cognitive stores** (`Energy`, `Scar`, `Utility`, `Affinity`,
  `Synapse`, `Engram`, `Coactivation`) are now thin views over
  substrate columns; direct writers that targeted the old monolithic
  stores must move to `SubstrateStore::*` column-view APIs.
- **HNSW index replaced** by the L0 (128-bit) / L1 (512-bit) / L2 (f16
  cosine) tier cascade. The public retrieval API (`vector_search`)
  keeps the same signature, but the implementation no longer builds
  an in-RAM HNSW graph.

### Added

- `obrain-substrate` crate — single mmap'd store with WAL-native
  durability, index-free adjacency, column-inline cognitive state.
- `obrain-migrate` CLI — one-shot offline migrator from `.obrain`
  legacy files to the substrate directory layout (dry-run,
  upgrade, force modes).
- **Self-maintaining Thinkers** — auto-started on `open()`:
  `Consolidator` (energy decay), `CommunityWarden` (LDleiden + Hilbert
  compaction), `Predictor` (retrieval warm-up), `Dreamer`
  (cross-community synapse proposals).
- **Geometric inference columns** — `ricci` (Ricci-Ollivier curvature
  per edge), `curvature` (per node), heat-kernel diffusion,
  geodesics over the f16 embedding manifold.
- **LDleiden incremental community detection** — online, `O(|ΔE|)`
  on edge mutations; drives `community_id` and Hilbert page
  ordering.
- `neo4j2obrain` rewrite — writes substrate directly with cognitive
  by-construction (embeddings, tiers, communities, centrality,
  Ricci, COACT edges) in one import.

### Performance (measured on 10⁶-node synthetic + megalaw)

| Metric | 0.1.x (LpgStore) | 0.2.0 (Substrate) | Gain |
|---|---|---|---|
| RSS total (10 users + 5 KBs) | ~110 GiB | ≤ 18 GiB | ÷ 6-10 |
| RSS anonymous incompressible | 5–10 GiB | ≤ 1 GiB | ÷ 10 |
| Startup (10⁶ nodes) | 5–15 s | ≤ 100 ms | ÷ 50-150 |
| Retrieval p95 (10⁶ nodes) | 30–100 ms | ≤ 1 ms | ÷ 30-100 |
| Recall@10 vs HNSW f32 baseline | 100 % | ≥ 99 % | preserved |

### Fixed (resolved by construction)

- Cognitive stores bypass-WAL gotcha — every reinforce / energy
  update / scar insert is now WAL-logged synchronously before the
  column write.
- `AssetRegistry` / `RoomStore` / `SpaceStore` / `IntegrationStore`
  direct-mutation paths — all routed through `HubWalStore` + substrate
  WAL-native columns.
- Dichotomy `ObrainDB::open` (full RAM) vs `ObrainDB::open_overlay`
  (mmap + delta LpgStore) — single `open` path.
- Cognitive state WAL vs full-snapshot inconsistency — single source
  of truth (substrate).

## [0.0.1] - 2026-03-27

### 🎉 Initial Release — Clean Break from Grafeo

This is the first release of **Obrain**, a complete rebrand of the Grafeo graph database.

### Summary

- Full rebrand: all references to "grafeo" replaced with "obrain" across 15 crates, 7 language bindings, CI/CD, and documentation
- New file extension: `.obrain` (replaces `.grafeo`)
- New npm scope: `@obrain`
- Version reset to `0.0.1` — clean break, no backward compatibility with previous Grafeo releases

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
