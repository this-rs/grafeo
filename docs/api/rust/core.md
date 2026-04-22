---
title: obrain-core
description: Core data structures crate.
tags:
  - api
  - rust
---

# obrain-core

Core graph storage and execution engine.

## Graph Storage

Since the T17 substrate cutover, the canonical backend is `SubstrateStore`
(mmap + WAL-native) from the `obrain-substrate` crate. It implements the
`GraphStore` / `GraphStoreMut` traits defined in `obrain-core`.

```rust
use obrain_substrate::SubstrateStore;
use obrain_core::graph::traits::GraphStoreMut;

let store = SubstrateStore::open_tempfile().unwrap();
let node_id = store.create_node(&["Person"]);
```

## Indexes

```rust
use obrain_core::index::HashIndex;

let index: HashIndex<String, NodeId> = HashIndex::new();
index.insert("Alix".into(), node_id);
```

## Execution

```rust
use obrain_core::execution::{DataChunk, ValueVector, SelectionVector};

let chunk = DataChunk::empty();
```

## Note

This is an internal crate. The API may change between minor versions.
