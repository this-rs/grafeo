# obrain-c

C FFI bindings for [Obrain](https://obrain.dev), a high-performance, embeddable graph database with a Rust core.

## Building

```bash
# From the Obrain repository root:
cargo build --release -p obrain-c --features full

# Output:
#   target/release/libobrain_c.so      (Linux)
#   target/release/libobrain_c.dylib   (macOS)
#   target/release/obrain_c.dll        (Windows)
```

The header file is at `crates/bindings/c/obrain.h`.

## Quick Start

```c
#include "obrain.h"
#include <stdio.h>

int main(void) {
    // Open an in-memory database
    ObrainDatabase *db = NULL;
    if (obrain_open_memory(&db) != OBRAIN_OK) {
        fprintf(stderr, "Error: %s\n", obrain_last_error());
        return 1;
    }

    // Create nodes
    uint64_t alix_id, gus_id;
    const char *labels[] = {"Person"};
    obrain_create_node(db, labels, 1, "{\"name\":\"Alix\",\"age\":30}", &alix_id);
    obrain_create_node(db, labels, 1, "{\"name\":\"Gus\",\"age\":25}", &gus_id);

    // Query with GQL
    ObrainResult *result = NULL;
    obrain_execute(db, "MATCH (p:Person) RETURN p.name, p.age", &result);

    char *json = NULL;
    obrain_result_json(result, &json);
    printf("%s\n", json);

    // Cleanup
    obrain_free_string(json);
    obrain_free_result(result);
    obrain_free_database(db);
    return 0;
}
```

Compile with:

```bash
gcc -o example example.c -lobrain_c -L/path/to/target/release
```

## API Overview

### Lifecycle

```c
obrain_open_memory(&db);      // in-memory database
obrain_open(path, &db);       // persistent database
obrain_close(db);             // flush and close
obrain_free_database(db);     // free handle
```

### Query Execution

```c
obrain_execute(db, gql, &result);                       // GQL
obrain_execute_with_params(db, gql, params_json, &result); // GQL + params
obrain_execute_cypher(db, query, &result);               // Cypher
obrain_execute_gremlin(db, query, &result);              // Gremlin
obrain_execute_graphql(db, query, &result);              // GraphQL
obrain_execute_sparql(db, query, &result);               // SPARQL
```

### Results

```c
obrain_result_json(result, &json);            // full result as JSON
obrain_result_row_count(result, &count);      // number of rows
obrain_result_execution_time_ms(result, &ms); // execution time
obrain_free_result(result);
```

### Node & Edge CRUD

```c
obrain_create_node(db, labels, label_count, props_json, &id);
obrain_create_edge(db, source, target, type, props_json, &id);
obrain_get_node(db, id, &node);
obrain_get_edge(db, id, &edge);
obrain_delete_node(db, id);
obrain_delete_edge(db, id);
obrain_set_node_property(db, id, key, value_json);
obrain_set_edge_property(db, id, key, value_json);
```

### Transactions

```c
ObrainTransaction *tx = NULL;
obrain_begin_transaction(db, &tx);
obrain_transaction_execute(tx, "INSERT (:Person {name: 'Harm'})", &result);
obrain_commit(tx);   // or obrain_rollback(tx)
```

### Vector Search

```c
obrain_create_vector_index(db, "Document", "embedding", 384, "cosine", 16, 200);
obrain_vector_search(db, "Document", "embedding", query_vec, dims, k, ef, &result);
obrain_batch_create_nodes(db, "Document", "embedding", vectors, count, dims, &ids);
```

### Error Handling

All functions return `ObrainStatus`. On error, call `obrain_last_error()`:

```c
if (obrain_execute(db, query, &result) != OBRAIN_OK) {
    fprintf(stderr, "Error: %s\n", obrain_last_error());
    obrain_clear_error();
}
```

### Memory Management

- Opaque pointers (`ObrainDatabase*`, `ObrainResult*`, etc.) must be freed with their `obrain_free_*` function
- Strings returned via `char**` out-params are caller-owned: free with `obrain_free_string()`
- Pointers accessed via getters (e.g. `obrain_edge_type()`) are valid until the parent is freed

## Features

- GQL, Cypher, SPARQL, Gremlin and GraphQL query languages
- Full node/edge CRUD with JSON property serialization
- ACID transactions with configurable isolation levels
- HNSW vector similarity search with batch operations
- Property indexes for fast lookups
- Thread-safe for concurrent use

## Links

- [Documentation](https://obrain.dev)
- [GitHub](https://github.com/this-rs/obrain)
- [Go Bindings](https://github.com/this-rs/obrain/tree/main/crates/bindings/go) (uses this library via CGO)
- [C# Bindings](https://github.com/this-rs/obrain/tree/main/crates/bindings/csharp) (uses this library via P/Invoke)
- [Dart Bindings](https://github.com/this-rs/obrain/tree/main/crates/bindings/dart) (uses this library via dart:ffi)

## License

Apache-2.0
