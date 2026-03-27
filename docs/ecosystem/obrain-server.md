# obrain-server

Standalone HTTP server and web UI for the Obrain graph database.

[:octicons-mark-github-16: GitHub](https://github.com/this-rs/obrain-server){ .md-button }
[:material-docker: Docker Hub](https://hub.docker.com/r/obrain/obrain-server){ .md-button }

## Overview

obrain-server wraps the Obrain engine in a REST API, GQL Wire Protocol (gRPC) and Bolt v5.x wire protocol, turning it from an embeddable library into a standalone database server. Pure Rust, single binary. Available in four tiers to match different deployment needs.

- **REST API** with auto-commit and explicit transaction modes
- **GQL Wire Protocol** (gRPC) on port 7688 for binary wire-protocol clients
- **Bolt v5.x** on port 7687 for Neo4j driver compatibility
- **Multi-language queries**: GQL, Cypher, GraphQL, Gremlin, SPARQL, SQL/PGQ
- **Admin API**: database stats, WAL management, integrity validation, index management
- **Search API**: vector (KNN/HNSW), text (BM25) and hybrid search
- **Batch queries** with atomic rollback
- **WebSocket streaming** for interactive query execution
- **Web UI** (Studio) for interactive query exploration
- **ACID transactions** with session-based lifecycle
- **In-memory or persistent**: omit data directory for ephemeral, set it for durable storage

## Docker Image Tiers

Four tiers are published to Docker Hub on every release:

| Tier | Tag | Transport | Languages | AI/Search | Web UI | Binary |
|------|-----|-----------|-----------|-----------|--------|--------|
| **gwp** | `obrain-server:gwp` | GWP (gRPC :7688) | GQL | No | No | ~7 MB |
| **bolt** | `obrain-server:bolt` | Bolt v5 (:7687) | Cypher | No | No | ~8 MB |
| **standard** | `obrain-server:latest` | HTTP (:7474) | All 6 | No | Studio | ~21 MB |
| **full** | `obrain-server:full` | HTTP + GWP + Bolt | All 6 | Yes + embed | Studio | ~25 MB |

Versioned tags follow the pattern: `0.4.6`, `0.4.6-gwp`, `0.4.6-bolt`, `0.4.6-full`.

### GWP

GQL-only gRPC wire protocol. Minimal footprint for machine-to-machine communication. Ideal for:

- Sidecar containers
- CI/CD test environments
- Edge deployments

```bash
docker run -p 7688:7688 obrain/obrain-server:gwp --data-dir /data
```

### Bolt

Cypher-only Bolt v5 wire protocol. Compatible with existing Neo4j drivers (Python `neo4j`, JavaScript `neo4j-driver`, etc.).

```bash
docker run -p 7687:7687 obrain/obrain-server:bolt --data-dir /data
```

### Standard (default)

HTTP REST API with all query languages, graph algorithms and the Studio web UI. This is the default `obrain-server:latest` image.

```bash
docker run -p 7474:7474 obrain/obrain-server
```

### Full

Everything in standard plus GWP, Bolt, AI/search features, ONNX embedding generation, authentication (bearer token, HTTP Basic), TLS and JSON Schema validation. Production-ready with all features and security built in.

```bash
docker run -p 7474:7474 -p 7687:7687 -p 7688:7688 obrain/obrain-server:full
```

## Quick Start

### Docker

```bash
# In-memory (ephemeral)
docker run -p 7474:7474 obrain/obrain-server

# Persistent storage
docker run -p 7474:7474 -v obrain-data:/data obrain/obrain-server --data-dir /data
```

### Docker Compose

```bash
docker compose up -d
```

Server at `http://localhost:7474`. Web UI at `http://localhost:7474/studio/`.

### Binary

```bash
obrain-server --data-dir ./mydata    # persistent
obrain-server                        # in-memory
```

## API Endpoints

### Query (auto-commit)

| Endpoint | Language | Example |
|----------|----------|---------|
| `POST /query` | GQL (default) | `{"query": "MATCH (p:Person) RETURN p.name"}` |
| `POST /cypher` | Cypher | `{"query": "MATCH (n) RETURN count(n)"}` |
| `POST /graphql` | GraphQL | `{"query": "{ Person { name age } }"}` |
| `POST /gremlin` | Gremlin | `{"query": "g.V().hasLabel('Person').values('name')"}` |
| `POST /sparql` | SPARQL | `{"query": "SELECT ?s WHERE { ?s a foaf:Person }"}` |
| `POST /sql` | SQL/PGQ | `{"query": "CALL obrain.procedures() YIELD name"}` |
| `POST /batch` | Mixed | Multiple queries in one atomic transaction |

```bash
curl -X POST http://localhost:7474/query \
  -H "Content-Type: application/json" \
  -d '{"query": "MATCH (p:Person) RETURN p.name, p.age"}'
```

### Transactions

```bash
# Begin
SESSION=$(curl -s -X POST http://localhost:7474/tx/begin | jq -r .session_id)

# Execute
curl -X POST http://localhost:7474/tx/query \
  -H "Content-Type: application/json" \
  -H "X-Session-Id: $SESSION" \
  -d '{"query": "INSERT (:Person {name: '\''Alix'\''})"}'

# Commit (or POST /tx/rollback)
curl -X POST http://localhost:7474/tx/commit \
  -H "X-Session-Id: $SESSION"
```

### WebSocket

Connect to `ws://localhost:7474/ws` for interactive query execution:

```json
{"type": "query", "id": "q1", "query": "MATCH (n) RETURN n", "language": "gql"}
```

### Admin

Database introspection, maintenance and index management:

| Endpoint | Description |
|----------|-------------|
| `GET /admin/{db}/stats` | Node/edge/label/property counts, memory, disk usage |
| `GET /admin/{db}/wal` | WAL status (enabled, path, size, record count) |
| `POST /admin/{db}/wal/checkpoint` | Force WAL checkpoint |
| `GET /admin/{db}/validate` | Database integrity validation |
| `POST /admin/{db}/index` | Create property, vector or text index |
| `DELETE /admin/{db}/index` | Drop an index |

### Search

Vector, text and hybrid search (requires `vector-index`, `text-index`, `hybrid-search` features):

| Endpoint | Description |
|----------|-------------|
| `POST /search/vector` | KNN vector similarity search via HNSW index |
| `POST /search/text` | Full-text BM25 search |
| `POST /search/hybrid` | Combined vector + text search with rank fusion |

### GQL Wire Protocol (GWP)

The gwp and full builds include a gRPC-based binary wire protocol on port 7688. All query, transaction, database, admin and search operations are available over gRPC. Configure with `--gwp-port` or `OBRAIN_GWP_PORT`.

### Bolt v5.x (BOLTR)

The bolt and full builds include a Bolt v5.x wire protocol on port 7687, compatible with Neo4j drivers. Configure with `--bolt-port` or `OBRAIN_BOLT_PORT`.

### Health & Feature Discovery

```bash
curl http://localhost:7474/health
```

The health endpoint reports which features are compiled into the running server:

```json
{
  "status": "ok",
  "features": {
    "languages": ["gql", "cypher", "sparql", "gremlin", "graphql", "sql-pgq"],
    "engine": ["parallel", "wal", "spill", "mmap"],
    "server": ["gwp"]
  }
}
```

## Configuration

Environment variables (prefix `OBRAIN_`), overridden by CLI flags:

| Variable | Default | Description |
|----------|---------|-------------|
| `OBRAIN_HOST` | `0.0.0.0` | Bind address |
| `OBRAIN_PORT` | `7474` | Bind port |
| `OBRAIN_DATA_DIR` | _(none)_ | Persistence directory (omit for in-memory) |
| `OBRAIN_SESSION_TTL` | `300` | Transaction session timeout (seconds) |
| `OBRAIN_QUERY_TIMEOUT` | `30` | Query execution timeout in seconds (0 = disabled) |
| `OBRAIN_CORS_ORIGINS` | _(none)_ | Comma-separated allowed origins (`*` for all) |
| `OBRAIN_LOG_LEVEL` | `info` | Tracing log level |
| `OBRAIN_LOG_FORMAT` | `pretty` | Log format: `pretty` or `json` |
| `OBRAIN_GWP_PORT` | `7688` | GQL Wire Protocol (gRPC) port |
| `OBRAIN_GWP_MAX_SESSIONS` | `0` | Max concurrent GWP sessions (0 = unlimited) |
| `OBRAIN_BOLT_PORT` | `7687` | Bolt v5.x wire protocol port |
| `OBRAIN_BOLT_MAX_SESSIONS` | `0` | Max concurrent Bolt sessions (0 = unlimited) |
| `OBRAIN_RATE_LIMIT` | `0` | Max requests per window per IP (0 = disabled) |

### Authentication (feature: `auth`)

| Variable | Description |
|----------|-------------|
| `OBRAIN_AUTH_TOKEN` | Bearer token / API key |
| `OBRAIN_AUTH_USER` | HTTP Basic username |
| `OBRAIN_AUTH_PASSWORD` | HTTP Basic password |

### TLS (feature: `tls`)

| Variable | Description |
|----------|-------------|
| `OBRAIN_TLS_CERT` | Path to TLS certificate (PEM) |
| `OBRAIN_TLS_KEY` | Path to TLS private key (PEM) |

## Feature Flags (building from source)

When building from source, Cargo feature flags control which capabilities are compiled in:

| Tier | Cargo Command | Matches Docker |
|------|---------------|----------------|
| GWP | `cargo build --release --no-default-features --features gwp` | `gwp` |
| Bolt | `cargo build --release --no-default-features --features bolt` | `bolt` |
| Standard | `cargo build --release` | `standard` |
| Full | `cargo build --release --features full` | `full` |

Individual features can also be mixed:

```bash
# GWP + Bolt (both wire protocols, no HTTP)
cargo build --release --no-default-features --features "gwp,bolt,gql,cypher,storage"

# HTTP API with auth
cargo build --release --features auth
```

See the [obrain-server README](https://github.com/this-rs/obrain-server#feature-flags) for the complete feature flag reference.

## Wire Protocols

obrain-server supports two binary wire protocols for high-performance client-server communication. Both are standalone Rust crates that any database can adopt via backend traits.

### GWP (GQL Wire Protocol)

[:octicons-mark-github-16: GitHub](https://github.com/this-rs/gql-wire-protocol){ .md-button }
[:material-package-variant: crates.io](https://crates.io/crates/gwp){ .md-button }

A pure Rust gRPC wire protocol for [GQL (ISO/IEC 39075)](https://www.iso.org/standard/76120.html), the international standard query language for property graphs. GWP is the primary wire protocol for obrain-server, available on port 7688 by default.

**Key features:**

- Full GQL type system including extended numerics (BigInteger, BigFloat, Decimal)
- Six gRPC services: Session, GQL, Catalog, Admin, Search, Health
- Server-side streaming for large result sets
- GQLSTATUS codes for structured error reporting per the ISO standard
- Pluggable backend via `GqlBackend` trait
- Optional TLS via rustls
- Idle session reaping and graceful shutdown

**Client bindings:**

| Language | Package | Install |
|----------|---------|---------|
| Rust | [gwp](https://crates.io/crates/gwp) | `cargo add gwp` |
| Python | [gwp-py](https://pypi.org/project/gwp-py/) | `uv add gwp-py` |
| JavaScript | [gwp-js](https://www.npmjs.com/package/gwp-js) | `npm install gwp-js` |
| Go | [gwp/go](https://github.com/this-rs/gql-wire-protocol) | `go get github.com/this-rs/gwp/go` |
| Java | [dev.obrain:gwp](https://central.sonatype.com/) | Maven Central |

**Status:** v0.1.6, active development. The type system and service architecture are stable. Recent work has focused on aligning the catalog hierarchy (catalog > schema > graph) with the GQL specification.

### BOLTR (Bolt Wire Protocol)

[:octicons-mark-github-16: GitHub](https://github.com/this-rs/boltr){ .md-button }
[:material-package-variant: crates.io](https://crates.io/crates/boltr){ .md-button }

A pure Rust implementation of the [Bolt v5.x wire protocol](https://neo4j.com/docs/bolt/current/), the binary protocol used by Neo4j for client-server communication. BOLTR enables compatibility with existing Neo4j drivers and tooling.

**Key features:**

- Full Bolt v5.1-5.4 protocol support with PackStream binary encoding
- Complete Bolt type system: scalars, graph elements, temporal and spatial types
- TCP chunked message framing
- Both server (`BoltBackend` trait) and client (`BoltConnection`, `BoltSession`) APIs
- Pluggable authentication via `AuthValidator` trait
- Optional TLS via tokio-rustls
- ROUTE message support for cluster-aware drivers
- Graceful connection draining on shutdown

**Status:** v0.1.1, active development. Spec-complete for Bolt 5.1-5.4 including ROUTE and TELEMETRY messages. Rust-only at this time (existing Neo4j drivers in other languages work out of the box).

### Protocol Comparison

| Aspect | GWP | BOLTR |
|--------|-----|-------|
| **Standard** | GQL (ISO/IEC 39075) | Bolt v5.x (Neo4j) |
| **Transport** | gRPC + Protocol Buffers | TCP + PackStream |
| **Streaming** | Server-side gRPC streaming | Pull-based (PULL/DISCARD) |
| **Error model** | GQLSTATUS codes (ISO) | Neo4j error codes |
| **Client bindings** | Rust, Python, JS, Go, Java | Rust (Neo4j drivers compatible) |
| **Default port** | 7688 | 7687 |
| **Use case** | Standards-based GQL clients | Neo4j driver compatibility |

## When to Use

| Use Case | Recommendation |
|----------|----------------|
| Multi-client access over HTTP | obrain-server |
| Embedded in an application | [obrain](https://github.com/this-rs/obrain) (library) |
| Browser-only, no backend | [obrain-web](obrain-web.md) (WASM) |
| Lightweight sidecar / CI | obrain-server **gwp** or **bolt** tier |
| Production with security | obrain-server **full** variant |

## Requirements

- Docker (recommended) or Rust toolchain for building from source

## License

Apache-2.0
