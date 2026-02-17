# grafeo-server

Standalone HTTP server and web UI for the Grafeo graph database.

[:octicons-mark-github-16: GitHub](https://github.com/GrafeoDB/grafeo-server){ .md-button }
[:material-docker: Docker Hub](https://hub.docker.com/r/grafeo/grafeo-server){ .md-button }

## Overview

grafeo-server wraps the Grafeo engine in a REST API, turning it from an embeddable library into a standalone database server. Pure Rust, single binary.

- **REST API** with auto-commit and explicit transaction modes
- **Multi-language queries**: GQL, Cypher, GraphQL, Gremlin, SPARQL, SQL/PGQ
- **Batch queries** with atomic rollback
- **WebSocket streaming** for interactive query execution
- **Web UI** (Studio) for interactive query exploration
- **ACID transactions** with session-based lifecycle
- **In-memory or persistent**: omit data directory for ephemeral, set it for durable storage
- **Multiple Docker image variants** for different deployment needs

## Docker Image Variants

Three variants are published to Docker Hub on every release:

| Variant | Tag | Languages | Engine Features | Web UI | Auth/TLS |
|---------|-----|-----------|-----------------|--------|----------|
| **lite** | `grafeo-server:lite` | GQL only | Core storage | No | No |
| **standard** | `grafeo-server:latest` | All 6 | All + AI/search | Yes | No |
| **full** | `grafeo-server:full` | All 6 | All + AI + ONNX embed | Yes | Yes |

Versioned tags follow the pattern: `0.3.0`, `0.3.0-lite`, `0.3.0-full`.

### Lite

Minimal footprint. GQL query language with core storage features (parallel execution, WAL, spill-to-disk, mmap). No web UI, no schema parsing, no auth/TLS. Ideal for:

- Sidecar containers
- CI/CD test environments
- Embedded deployments
- Development and prototyping

```bash
docker run -p 7474:7474 grafeo/grafeo-server:lite
```

### Standard (default)

All query languages, AI/search features (vector index, text index, hybrid search, CDC), RDF support and the Studio web UI. This is the default `grafeo-server:latest` image.

```bash
docker run -p 7474:7474 grafeo/grafeo-server
```

### Full

Everything in standard plus authentication (bearer token, HTTP Basic), TLS, JSON Schema validation and ONNX embedding generation. Production-ready with security features built in.

```bash
docker run -p 7474:7474 grafeo/grafeo-server:full \
  --auth-token my-secret --tls-cert /certs/cert.pem --tls-key /certs/key.pem
```

## Quick Start

### Docker

```bash
# In-memory (ephemeral)
docker run -p 7474:7474 grafeo/grafeo-server

# Persistent storage
docker run -p 7474:7474 -v grafeo-data:/data grafeo/grafeo-server --data-dir /data
```

### Docker Compose

```bash
docker compose up -d
```

Server at `http://localhost:7474`. Web UI at `http://localhost:7474/studio/`.

### Binary

```bash
grafeo-server --data-dir ./mydata    # persistent
grafeo-server                        # in-memory
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
  -d '{"query": "INSERT (:Person {name: '\''Alice'\''})"}'

# Commit (or POST /tx/rollback)
curl -X POST http://localhost:7474/tx/commit \
  -H "X-Session-Id: $SESSION"
```

### WebSocket

Connect to `ws://localhost:7474/ws` for interactive query execution:

```json
{"type": "query", "id": "q1", "query": "MATCH (n) RETURN n", "language": "gql"}
```

### Health & Feature Discovery

```bash
curl http://localhost:7474/health
```

The health endpoint reports which features are compiled into the running server:

```json
{
  "status": "ok",
  "version": "0.3.0",
  "features": {
    "languages": ["gql", "cypher", "sparql", "gremlin", "graphql", "sql-pgq"],
    "engine": ["parallel", "wal", "spill", "mmap", "rdf", "vector-index", "text-index", "hybrid-search", "cdc"],
    "server": ["owl-schema", "rdfs-schema"]
  }
}
```

## Configuration

Environment variables (prefix `GRAFEO_`), overridden by CLI flags:

| Variable | Default | Description |
|----------|---------|-------------|
| `GRAFEO_HOST` | `0.0.0.0` | Bind address |
| `GRAFEO_PORT` | `7474` | Bind port |
| `GRAFEO_DATA_DIR` | _(none)_ | Persistence directory (omit for in-memory) |
| `GRAFEO_SESSION_TTL` | `300` | Transaction session timeout (seconds) |
| `GRAFEO_QUERY_TIMEOUT` | `30` | Query execution timeout in seconds (0 = disabled) |
| `GRAFEO_CORS_ORIGINS` | _(none)_ | Comma-separated allowed origins (`*` for all) |
| `GRAFEO_LOG_LEVEL` | `info` | Tracing log level |
| `GRAFEO_LOG_FORMAT` | `pretty` | Log format: `pretty` or `json` |
| `GRAFEO_RATE_LIMIT` | `0` | Max requests per window per IP (0 = disabled) |

### Authentication (full variant)

| Variable | Description |
|----------|-------------|
| `GRAFEO_AUTH_TOKEN` | Bearer token / API key |
| `GRAFEO_AUTH_USER` | HTTP Basic username |
| `GRAFEO_AUTH_PASSWORD` | HTTP Basic password |

### TLS (full variant)

| Variable | Description |
|----------|-------------|
| `GRAFEO_TLS_CERT` | Path to TLS certificate (PEM) |
| `GRAFEO_TLS_KEY` | Path to TLS private key (PEM) |

## Feature Flags (building from source)

When building from source, Cargo feature flags control which capabilities are compiled in:

| Preset | Cargo Command | Matches Docker |
|--------|---------------|----------------|
| Lite | `cargo build --release --no-default-features --features "gql,storage"` | `lite` |
| Standard | `cargo build --release` | `standard` |
| Full | `cargo build --release --features full` | `full` |

Individual features can also be mixed:

```bash
# GQL + Cypher only, with auth
cargo build --release --no-default-features --features "gql,cypher,storage,auth"
```

See the [grafeo-server README](https://github.com/GrafeoDB/grafeo-server#feature-flags) for the complete feature flag reference.

## When to Use

| Use Case | Recommendation |
|----------|----------------|
| Multi-client access over HTTP | grafeo-server |
| Embedded in an application | [grafeo](https://github.com/GrafeoDB/grafeo) (library) |
| Browser-only, no backend | [grafeo-web](https://github.com/GrafeoDB/grafeo-web) (WASM) |
| Lightweight sidecar / CI | grafeo-server **lite** variant |
| Production with security | grafeo-server **full** variant |

## Requirements

- Docker (recommended) or Rust toolchain for building from source

## License

Apache-2.0
