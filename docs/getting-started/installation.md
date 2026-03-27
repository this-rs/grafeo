---
title: Installation
description: Install Obrain for Python, Node.js, Go, C#, Dart, Rust or WebAssembly.
---

# Installation

Obrain supports Python, Node.js/TypeScript, Go, C#, Dart, Rust and WebAssembly. Choose the installation method for the preferred language.

## Python

### Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer:

```bash
uv add obrain
```

### Using pip (alternative)

```bash
pip install obrain  # If uv is not available
```

### Verify Installation

```python
import obrain

# Print version
print(obrain.__version__)

# Create a test database
db = obrain.ObrainDB()
print("Obrain installed successfully!")
```

### Platform Support

| Platform | Architecture | Support |
|----------|--------------|---------|
| Linux    | x86_64       | :material-check: Full |
| Linux    | aarch64      | :material-check: Full |
| macOS    | x86_64       | :material-check: Full |
| macOS    | arm64 (M1/M2)| :material-check: Full |
| Windows  | x86_64       | :material-check: Full |

## Node.js / TypeScript

```bash
npm install @obrain-db/js
```

### Verify Installation

```js
const { ObrainDB } = require('@obrain-db/js');

const db = await ObrainDB.create();
console.log('Obrain installed successfully!');
await db.close();
```

## Go

```bash
go get github.com/this-rs/obrain/crates/bindings/go
```

### Verify Installation

```go
package main

import (
    "fmt"
    obrain "github.com/this-rs/obrain/crates/bindings/go"
)

func main() {
    db, _ := obrain.OpenInMemory()
    defer db.Close()
    fmt.Println("Obrain installed successfully!")
}
```

## WebAssembly

```bash
npm install @obrain-db/wasm
```

### Verify Installation

```js
import init, { Database } from '@obrain-db/wasm';

await init();
const db = new Database();
console.log('Obrain WASM installed successfully!');
```

## C# / .NET

```bash
dotnet add package ObrainDB
```

### Verify Installation

```csharp
using Obrain;

var db = new ObrainDB();
Console.WriteLine("Obrain installed successfully!");
```

## Dart

Add to `pubspec.yaml`:

```yaml
dependencies:
  obrain: ^0.5.21
```

### Verify Installation

```dart
import 'package:obrain/obrain.dart';

void main() {
  final db = ObrainDB.memory();
  print('Obrain installed successfully!');
  db.close();
}
```

## Rust

### Using Cargo

Add Obrain to the project:

```bash
cargo add obrain
```

Or add it manually to `Cargo.toml`:

```toml
[dependencies]
obrain = "0.5"
```

### Feature Flags

The `embedded` profile is enabled by default: GQL, AI features (vector/text/hybrid search, CDC), graph algorithms and parallel execution. Use feature groups or individual flags to customize:

```toml
[dependencies]
# Default (embedded profile): GQL + AI + algorithms + parallel
obrain = "0.5"

# All languages + AI + storage + RDF
obrain = { version = "0.5", default-features = false, features = ["full"] }

# Only query languages, no AI features
obrain = { version = "0.5", default-features = false, features = ["languages"] }

# GQL with AI features
obrain = { version = "0.5", default-features = false, features = ["gql", "ai"] }

# Minimal: GQL only
obrain = { version = "0.5", default-features = false, features = ["gql"] }

# With ONNX embedding generation (opt-in, not in full)
obrain = { version = "0.5", features = ["embed"] }
```

#### Feature Groups

| Profile / Group | Contents | Description |
|-----------------|----------|-------------|
| `embedded` | gql, ai, algos, parallel, regex | Default for libraries and bindings |
| `browser` | gql, regex-lite | Default for WASM |
| `server` / `full` | embedded + languages + storage + rdf + cdc | Everything except embed |
| `languages` | gql, cypher, sparql, gremlin, graphql, sql-pgq | All query language parsers |
| `ai` | vector-index, text-index, hybrid-search, cdc | AI/RAG search + change tracking |
| `storage` | wal, spill, mmap, obrain-file | Persistence backends |
| `algos` | graph algorithms | SSSP, PageRank, centrality, community detection |
| `embed` | ort, tokenizers | ONNX embedding generation (opt-in, ~17MB) |

#### Individual Language Flags

| Feature | Description |
|---------|-------------|
| `gql` | GQL (ISO/IEC 39075): default query language |
| `cypher` | Cypher (openCypher 9.0) |
| `sparql` | SPARQL (W3C 1.1) + RDF support |
| `gremlin` | Gremlin (Apache TinkerPop) |
| `graphql` | GraphQL |
| `sql-pgq` | SQL/PGQ (SQL:2023 GRAPH_TABLE) |

#### Individual AI Flags

| Feature | Description |
|---------|-------------|
| `vector-index` | HNSW approximate nearest neighbor index |
| `text-index` | BM25 inverted index for full-text search |
| `hybrid-search` | Combined text + vector search with score fusion |
| `cdc` | Change data capture (before/after property snapshots) |

### Verify Installation

```rust
use obrain::ObrainDB;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let db = ObrainDB::new_in_memory();
    println!("Obrain installed successfully!");
    Ok(())
}
```

## Obrain Server (Docker)

For a standalone database server accessible via REST API, use [obrain-server](../ecosystem/obrain-server.md):

```bash
# Standard: all query languages, AI/search, web UI
docker run -p 7474:7474 obrain/obrain-server
```

Three image variants are available:

| Variant | Tag | Description |
|---------|-----|-------------|
| **lite** | `obrain-server:lite` | GQL only, no UI, smallest footprint |
| **standard** | `obrain-server:latest` | All languages + AI/search + web UI |
| **full** | `obrain-server:full` | Everything + auth + TLS + ONNX embed |

```bash
# Lite: minimal, GQL only
docker run -p 7474:7474 obrain/obrain-server:lite

# Full: production with auth and TLS
docker run -p 7474:7474 obrain/obrain-server:full \
  --auth-token my-secret --data-dir /data
```

Server at `http://localhost:7474`. Web UI (standard/full) at `http://localhost:7474/studio/`.

See the [obrain-server documentation](../ecosystem/obrain-server.md) for full API reference and configuration.

## Building from Source

### Clone the Repository

```bash
git clone https://github.com/this-rs/obrain.git
cd obrain
```

### Build Rust Crates

```bash
cargo build --workspace --release
```

### Build Python Package

```bash
cd crates/bindings/python
uv add maturin
maturin develop --release
```

### Build Node.js Package

```bash
cd crates/bindings/node
npm install
npm run build
```

### Build WASM Package

```bash
wasm-pack build crates/bindings/wasm --target web --release
```

## Next Steps

With Obrain installed, continue to the [Quick Start](quickstart.md) guide.
