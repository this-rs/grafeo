# obrain

Go bindings for [Obrain](https://obrain.dev), a high-performance, embeddable graph database with a Rust core and no required C dependencies.

## Requirements

- Go 1.22+
- CGO enabled (`CGO_ENABLED=1`)
- The `obrain-c` shared library (`libobrain_c.so` / `libobrain_c.dylib` / `obrain_c.dll`)

## Installation

```bash
go get github.com/this-rs/obrain/crates/bindings/go
```

## Quick Start

```go
package main

import (
    "fmt"
    "log"

    obrain "github.com/this-rs/obrain/crates/bindings/go"
)

func main() {
    db, err := obrain.OpenInMemory()
    if err != nil {
        log.Fatal(err)
    }
    defer db.Close()

    // Create nodes
    db.CreateNode([]string{"Person"}, map[string]any{"name": "Alix", "age": 30})
    db.CreateNode([]string{"Person"}, map[string]any{"name": "Gus", "age": 25})

    // Query with GQL
    result, err := db.Execute("MATCH (p:Person) WHERE p.age > 20 RETURN p.name, p.age")
    if err != nil {
        log.Fatal(err)
    }
    for _, row := range result.Rows {
        fmt.Printf("Name: %v, Age: %v\n", row["p.name"], row["p.age"])
    }
}
```

## Features

- GQL, Cypher, SPARQL, Gremlin and GraphQL query languages
- Full node/edge CRUD with property management
- ACID transactions with configurable isolation levels
- HNSW vector similarity search
- Property indexes for fast lookups
- Thread-safe for concurrent use

## Building the Shared Library

```bash
# From the Obrain repository root:
cargo build --release -p obrain-c --features full

# The library is at:
#   target/release/libobrain_c.so      (Linux)
#   target/release/libobrain_c.dylib   (macOS)
#   target/release/obrain_c.dll        (Windows)
```

## Links

- [Documentation](https://obrain.dev)
- [GitHub](https://github.com/this-rs/obrain)
- [Python Package](https://pypi.org/project/obrain/)
- [npm Package](https://www.npmjs.com/package/@obrain-db/js)
