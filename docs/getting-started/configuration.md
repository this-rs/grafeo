---
title: Configuration
description: Configure Obrain for different use cases.
---

# Configuration

Obrain can be configured for different use cases, from small embedded applications to high-performance server deployments.

## Database Modes

### In-Memory Mode

For temporary data or maximum performance:

=== "Python"

    ```python
    import obrain

    # In-memory database (default)
    db = obrain.ObrainDB()
    ```

=== "Rust"

    ```rust
    use obrain::ObrainDB;

    let db = ObrainDB::new_in_memory()?;
    ```

!!! note "Data Persistence"
    In-memory databases do not persist data. All data is lost when the database is closed.

### Persistent Mode

For durable storage:

=== "Python"

    ```python
    import obrain

    # Persistent database
    db = obrain.ObrainDB(path="my_graph.db")
    ```

=== "Rust"

    ```rust
    use obrain::ObrainDB;

    let db = ObrainDB::new("my_graph.db")?;
    ```

## Configuration Options

### Memory Limit

Control the maximum memory usage:

=== "Python"

    ```python
    db = obrain.ObrainDB(
        path="my_graph.db",
        memory_limit=4 * 1024 * 1024 * 1024  # 4 GB
    )
    ```

=== "Rust"

    ```rust
    use obrain::{ObrainDB, Config};

    let config = Config::builder()
        .memory_limit(4 * 1024 * 1024 * 1024)  // 4 GB
        .build()?;

    let db = ObrainDB::with_config(config)?;
    ```

### Thread Pool Size

Configure parallelism:

=== "Python"

    ```python
    db = obrain.ObrainDB(
        path="my_graph.db",
        threads=8
    )
    ```

=== "Rust"

    ```rust
    use obrain::{ObrainDB, Config};

    let config = Config::builder()
        .threads(8)
        .build()?;

    let db = ObrainDB::with_config(config)?;
    ```

!!! tip "Default Thread Count"
    By default, Obrain uses the number of available CPU cores.

## Configuration Reference

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `path` | `string` | `None` | Database file path (None for in-memory) |
| `memory_limit` | `int` | System RAM | Maximum memory usage in bytes |
| `threads` | `int` | CPU cores | Number of worker threads |
| `read_only` | `bool` | `false` | Open database in read-only mode |

## Environment Variables

Obrain can also be configured via environment variables:

| Variable | Description |
|----------|-------------|
| `OBRAIN_MEMORY_LIMIT` | Maximum memory in bytes |
| `OBRAIN_THREADS` | Number of worker threads |
| `OBRAIN_LOG_LEVEL` | Logging level (error, warn, info, debug, trace) |

## Performance Tuning

### For High-Throughput Workloads

```python
db = obrain.ObrainDB(
    path="high_throughput.db",
    memory_limit=8 * 1024 * 1024 * 1024,  # 8 GB
    threads=16
)
```

### For Low-Memory Environments

```python
db = obrain.ObrainDB(
    path="embedded.db",
    memory_limit=256 * 1024 * 1024,  # 256 MB
    threads=2
)
```

### For Read-Heavy Workloads

```python
# Multiple read replicas can be opened read-only
db = obrain.ObrainDB(
    path="replica.db",
    read_only=True
)
```

## Next Steps

- [User Guide](../user-guide/index.md) - Learn more about using Obrain
- [Architecture](../architecture/index.md) - Understand how Obrain works
