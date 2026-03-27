---
title: Rust API
description: Rust API reference.
---

# Rust API Reference

Obrain is written in Rust and provides a native Rust API.

## Crates

| Crate | docs.rs |
|-------|---------|
| obrain | [docs.rs/obrain](https://docs.rs/obrain) |
| obrain-engine | [docs.rs/obrain-engine](https://docs.rs/obrain-engine) |

## Quick Start

```rust
use obrain::ObrainDB;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let db = ObrainDB::new_in_memory()?;
    let session = db.session()?;

    session.execute("INSERT (:Person {name: 'Alix'})")?;

    Ok(())
}
```

## Crate Documentation

- [obrain-common](common.md) - Foundation types
- [obrain-core](core.md) - Core data structures
- [obrain-adapters](adapters.md) - Parsers and storage
- [obrain-engine](engine.md) - Database facade

## API Stability

The public API (`obrain` and `obrain-engine`) follows semver.

Internal crates may have breaking changes in minor versions.
