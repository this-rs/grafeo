---
title: obrain-adapters
description: Adapters crate.
tags:
  - api
  - rust
---

# obrain-adapters

Parsers, storage backends and plugins.

## GQL Parser

```rust
use obrain_adapters::query::gql;

let ast = gql::parse("MATCH (n:Person) RETURN n")?;
```

## Storage

```rust
use obrain_adapters::storage::MemoryBackend;
use obrain_adapters::storage::wal::WalManager;

let backend = MemoryBackend::new();
let wal = WalManager::open("path/to/wal")?;
```

## Plugins

```rust
use obrain_adapters::plugins::{Plugin, PluginRegistry};

let registry = PluginRegistry::new();
registry.register(MyPlugin::new())?;
```

## Note

This is an internal crate. The API may change between minor versions.
