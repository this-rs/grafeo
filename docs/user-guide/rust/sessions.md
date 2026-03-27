---
title: Sessions
description: Session management in Rust.
tags:
  - rust
  - sessions
---

# Sessions

Sessions provide transactional access to the database.

## Creating Sessions

```rust
let db = ObrainDB::new_in_memory()?;

// Create a session
let session = db.session()?;

// Use the session
session.execute("INSERT (:Person {name: 'Alix'})")?;
```

## Transactions

```rust
let session = db.session()?;

// Begin explicit transaction
session.begin_tx()?;

session.execute("INSERT (:Person {name: 'Alix'})")?;
session.execute("INSERT (:Person {name: 'Gus'})")?;

// Commit
session.commit()?;
```

## Rollback

```rust
let session = db.session()?;

session.begin_tx()?;
session.execute("INSERT (:Person {name: 'Alix'})")?;

// Something went wrong, rollback
session.rollback()?;
```

## Transaction Closure

```rust
let session = db.session()?;

// Execute in transaction with automatic commit/rollback
session.transaction(|tx| {
    tx.execute("INSERT (:Person {name: 'Alix'})")?;
    tx.execute("INSERT (:Person {name: 'Gus'})")?;
    Ok(())
})?;
```

## Multiple Sessions

```rust
let db = ObrainDB::new_in_memory()?;

// Each session has isolated transactions
let session1 = db.session()?;
let session2 = db.session()?;

session1.begin_tx()?;
session1.execute("INSERT (:Person {name: 'Alix'})")?;
// session2 won't see Alix until session1 commits

session1.commit()?;
// Now session2 can see Alix
```
