# grafeo-c

C FFI bindings for [Grafeo](https://grafeo.dev), a high-performance, embeddable graph database with a Rust core.

## Building

```bash
# From the Grafeo repository root:
cargo build --release -p grafeo-c --features full

# Output:
#   target/release/libgrafeo_c.so      (Linux)
#   target/release/libgrafeo_c.dylib   (macOS)
#   target/release/grafeo_c.dll        (Windows)
```

The header file is at `crates/bindings/c/grafeo.h`.

## Quick Start

```c
#include "grafeo.h"
#include <stdio.h>

int main(void) {
    // Open an in-memory database
    GrafeoDatabase *db = NULL;
    if (grafeo_open_memory(&db) != GRAFEO_OK) {
        fprintf(stderr, "Error: %s\n", grafeo_last_error());
        return 1;
    }

    // Create nodes
    uint64_t alice_id, bob_id;
    const char *labels[] = {"Person"};
    grafeo_create_node(db, labels, 1, "{\"name\":\"Alice\",\"age\":30}", &alice_id);
    grafeo_create_node(db, labels, 1, "{\"name\":\"Bob\",\"age\":25}", &bob_id);

    // Query with GQL
    GrafeoResult *result = NULL;
    grafeo_execute(db, "MATCH (p:Person) RETURN p.name, p.age", &result);

    char *json = NULL;
    grafeo_result_json(result, &json);
    printf("%s\n", json);

    // Cleanup
    grafeo_free_string(json);
    grafeo_free_result(result);
    grafeo_free_database(db);
    return 0;
}
```

Compile with:

```bash
gcc -o example example.c -lgrafeo_c -L/path/to/target/release
```

## API Overview

### Lifecycle

```c
grafeo_open_memory(&db);      // in-memory database
grafeo_open(path, &db);       // persistent database
grafeo_close(db);             // flush and close
grafeo_free_database(db);     // free handle
```

### Query Execution

```c
grafeo_execute(db, gql, &result);                       // GQL
grafeo_execute_with_params(db, gql, params_json, &result); // GQL + params
grafeo_execute_cypher(db, query, &result);               // Cypher
grafeo_execute_gremlin(db, query, &result);              // Gremlin
grafeo_execute_graphql(db, query, &result);              // GraphQL
grafeo_execute_sparql(db, query, &result);               // SPARQL
```

### Results

```c
grafeo_result_json(result, &json);            // full result as JSON
grafeo_result_row_count(result, &count);      // number of rows
grafeo_result_execution_time_ms(result, &ms); // execution time
grafeo_free_result(result);
```

### Node & Edge CRUD

```c
grafeo_create_node(db, labels, label_count, props_json, &id);
grafeo_create_edge(db, source, target, type, props_json, &id);
grafeo_get_node(db, id, &node);
grafeo_get_edge(db, id, &edge);
grafeo_delete_node(db, id);
grafeo_delete_edge(db, id);
grafeo_set_node_property(db, id, key, value_json);
grafeo_set_edge_property(db, id, key, value_json);
```

### Transactions

```c
GrafeoTransaction *tx = NULL;
grafeo_begin_tx(db, &tx);
grafeo_tx_execute(tx, "INSERT (:Person {name: 'Carol'})", &result);
grafeo_commit(tx);   // or grafeo_rollback(tx)
```

### Vector Search

```c
grafeo_create_vector_index(db, "Document", "embedding", 384, "cosine", 16, 200);
grafeo_vector_search(db, "Document", "embedding", query_vec, dims, k, ef, &result);
grafeo_batch_create_nodes(db, "Document", "embedding", vectors, count, dims, &ids);
```

### Error Handling

All functions return `GrafeoStatus`. On error, call `grafeo_last_error()`:

```c
if (grafeo_execute(db, query, &result) != GRAFEO_OK) {
    fprintf(stderr, "Error: %s\n", grafeo_last_error());
    grafeo_clear_error();
}
```

### Memory Management

- Opaque pointers (`GrafeoDatabase*`, `GrafeoResult*`, etc.) must be freed with their `grafeo_free_*` function
- Strings returned via `char**` out-params are caller-owned — free with `grafeo_free_string()`
- Pointers accessed via getters (e.g. `grafeo_edge_type()`) are valid until the parent is freed

## Features

- GQL, Cypher, SPARQL, Gremlin, and GraphQL query languages
- Full node/edge CRUD with JSON property serialization
- ACID transactions with configurable isolation levels
- HNSW vector similarity search with batch operations
- Property indexes for fast lookups
- Thread-safe for concurrent use

## Links

- [Documentation](https://grafeo.dev)
- [GitHub](https://github.com/GrafeoDB/grafeo)
- [Go Bindings](https://github.com/GrafeoDB/grafeo/tree/main/crates/bindings/go) (uses this library via CGO)

## License

Apache-2.0
