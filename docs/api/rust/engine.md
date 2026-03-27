---
title: obrain-engine
description: Database engine crate.
tags:
  - api
  - rust
---

# obrain-engine

Main database facade and coordination.

## ObrainDB

```rust
use obrain_engine::{ObrainDB, Config};

// In-memory
let db = ObrainDB::new_in_memory();

// Persistent
let db = ObrainDB::open("path/to/db")?;

// With config
let config = Config::builder()
    .memory_limit(4 * 1024 * 1024 * 1024)
    .threads(8)
    .build()?;
let db = ObrainDB::with_config(config);
```

## Session

```rust
let mut session = db.session();

session.execute("INSERT (:Person {name: 'Alix'})")?;

let result = session.execute("MATCH (p:Person) RETURN p.name")?;
for row in result.rows {
    println!("{:?}", row);
}
```

## Transactions

```rust
let mut session = db.session();
session.begin_tx()?;
session.execute("...")?;
session.commit()?;
// or
session.rollback()?;
```
