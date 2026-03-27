---
title: API Reference
description: Complete API reference for Obrain.
---

# API Reference

Complete API documentation for all supported languages.

## Python API

The Python API provides a Pythonic interface to Obrain.

- [Python Overview](python/index.md)
- [obrain.ObrainDB](python/database.md)
- [obrain.Node](python/node.md)
- [obrain.Edge](python/edge.md)
- [obrain.QueryResult](python/result.md)
- [obrain.Transaction](python/transaction.md)

## Node.js / TypeScript API

Native bindings via napi-rs with full TypeScript definitions.

- [Node.js Overview](node/index.md)
- [ObrainDB](node/database.md)
- [QueryResult](node/query.md)
- [Transaction](node/transaction.md)
- [JsNode](node/node.md)
- [JsEdge](node/edge.md)

## Go API

CGO bindings for cloud-native applications.

- **Package**: [`github.com/this-rs/obrain/crates/bindings/go`](https://pkg.go.dev/github.com/this-rs/obrain/crates/bindings/go)
- Node/edge CRUD, property management, label operations
- ACID transactions, vector search, batch operations

## C# / .NET API

Full-featured .NET 8 bindings via source-generated P/Invoke.

- **Package**: `ObrainDB` (NuGet)
- GQL + multi-language queries (sync and async), ACID transactions
- Typed node/edge CRUD, vector search (k-NN + MMR), temporal type support
- `SafeHandle`-based resource management

## Dart API

Dart FFI bindings wrapping the C layer.

- **Package**: [`obrain`](https://pub.dev/packages/obrain)
- GQL query execution with parameterized queries, ACID transactions
- Typed node/edge CRUD, vector search, `NativeFinalizer` resource management

## WebAssembly API

Run Obrain in the browser, Deno or Cloudflare Workers.

- **Package**: [`@obrain-db/wasm`](https://www.npmjs.com/package/@obrain-db/wasm)
- In-memory only, all query languages supported
- 660 KB gzipped binary

## Rust API

The Rust API provides direct access to Obrain internals.

- [Rust Overview](rust/index.md)
- [obrain-common](rust/common.md)
- [obrain-core](rust/core.md)
- [obrain-adapters](rust/adapters.md)
- [obrain-engine](rust/engine.md)

## API Stability

| API | Stability |
|-----|-----------|
| Python | Stable |
| Node.js / TypeScript | Stable |
| Go | Stable |
| C# / .NET | Stable |
| Dart | Stable |
| WebAssembly | Stable |
| Rust (obrain-engine) | Stable |
| Rust (internal crates) | Unstable |
