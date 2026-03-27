---
title: Database Setup
description: Creating and configuring databases in Rust.
tags:
  - rust
  - database
---

# Database Setup

Learn how to create and configure Obrain databases in Rust.

## Creating a Database

```rust
use obrain::ObrainDB;

// In-memory database
let db = ObrainDB::new_in_memory();

// Persistent database
let db = ObrainDB::open("my_graph.db")?;
```

## Configuration

```rust
use obrain::{ObrainDB, Config};

let config = Config::builder()
    .memory_limit(4 * 1024 * 1024 * 1024)  // 4 GB
    .threads(8)
    .build()?;

let db = ObrainDB::with_config(config);
```

## Database Lifecycle

```rust
use obrain::ObrainDB;

fn main() -> Result<(), obrain_common::utils::error::Error> {
    // Create database
    let db = ObrainDB::open("my_graph.db")?;

    // Use the database
    let mut session = db.session();
    session.execute("INSERT (:Person {name: 'Alix'})")?;

    // Database is dropped and closed when it goes out of scope
    Ok(())
}
```

## Thread Safety

`ObrainDB` is `Send` and `Sync`, so it can be shared across threads:

```rust
use obrain::ObrainDB;
use std::sync::Arc;
use std::thread;

let db = Arc::new(ObrainDB::new_in_memory());

let handles: Vec<_> = (0..4).map(|i| {
    let db = Arc::clone(&db);
    thread::spawn(move || {
        let mut session = db.session();
        session.execute(&format!(
            "INSERT (:Person {{id: {}}})", i
        )).unwrap();
    })
}).collect();

for handle in handles {
    handle.join().unwrap();
}
```
