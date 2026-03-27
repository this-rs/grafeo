---
title: Persistent Storage
description: Using Obrain with durable storage.
tags:
  - persistence
  - storage
---

# Persistent Storage

Persistent mode stores data durably on disk.

## Creating a Persistent Database

=== "Python"

    ```python
    import obrain

    db = obrain.ObrainDB(path="my_graph.db")
    ```

=== "Rust"

    ```rust
    use obrain::ObrainDB;

    let db = ObrainDB::new("my_graph.db")?;
    ```

## File Structure

```
my_graph.db/
├── data/           # Main data files
├── wal/            # Write-ahead log
└── metadata        # Database metadata
```

## Durability Guarantees

- **Write-Ahead Logging (WAL)** - All changes logged before applying
- **Checkpointing** - Periodic consolidation of WAL into data files
- **Crash Recovery** - Automatic recovery from WAL on startup

## Configuration

```python
db = obrain.ObrainDB(
    path="my_graph.db",
    # Sync mode: 'full' (default), 'normal', 'off'
    sync_mode='full'
)
```

| Sync Mode | Durability | Performance |
|-----------|------------|-------------|
| `full` | Highest | Slower |
| `normal` | Good | Faster |
| `off` | None | Fastest |

## Reopening a Database

```python
# First session
db = obrain.ObrainDB(path="my_graph.db")
db.execute("INSERT (:Person {name: 'Alix'})")

# Later session - data persists
db = obrain.ObrainDB(path="my_graph.db")
result = db.execute("MATCH (p:Person) RETURN p.name")
# Returns 'Alix'
```
