---
title: C API
description: API reference for the grafeo-c FFI bindings.
---

# C API

C-compatible FFI layer for embedding Grafeo in any language. Used by the Go bindings via CGO.

## Building

```bash
cargo build --release -p grafeo-c --features full
```

Output:

- `target/release/libgrafeo_c.so` (Linux)
- `target/release/libgrafeo_c.dylib` (macOS)
- `target/release/grafeo_c.dll` (Windows)

Header: `crates/bindings/c/grafeo.h`

## Quick Start

```c
#include "grafeo.h"
#include <stdio.h>

int main(void) {
    GrafeoDatabase *db = grafeo_open_memory();
    if (!db) {
        fprintf(stderr, "Error: %s\n", grafeo_last_error());
        return 1;
    }

    GrafeoResult *r = grafeo_execute(db, "MATCH (p:Person) RETURN p.name");
    if (r) {
        printf("Rows: %zu\n", grafeo_result_row_count(r));
        printf("JSON: %s\n", grafeo_result_json(r));
        grafeo_free_result(r);
    }

    grafeo_free_database(db);
    return 0;
}
```

Compile:

```bash
gcc -o example example.c -lgrafeo_c -L/path/to/target/release
```

## Lifecycle

```c
GrafeoDatabase* grafeo_open_memory();           // in-memory
GrafeoDatabase* grafeo_open(const char* path);  // persistent
void grafeo_free_database(GrafeoDatabase* db);  // free handle
```

## Query Execution

```c
GrafeoResult* grafeo_execute(db, query);
GrafeoResult* grafeo_execute_with_params(db, query, params_json);
GrafeoResult* grafeo_execute_cypher(db, query);
GrafeoResult* grafeo_execute_gremlin(db, query);
GrafeoResult* grafeo_execute_graphql(db, query);
GrafeoResult* grafeo_execute_sparql(db, query);
GrafeoResult* grafeo_execute_sql(db, query);
```

## Result Access

```c
const char* grafeo_result_json(const GrafeoResult* r);
size_t      grafeo_result_row_count(const GrafeoResult* r);
double      grafeo_result_execution_time_ms(const GrafeoResult* r);
void        grafeo_free_result(GrafeoResult* r);
```

## Node & Edge CRUD

```c
uint64_t grafeo_create_node(db, labels, label_count);
uint64_t grafeo_create_edge(db, source, target, edge_type);
void     grafeo_set_node_property(db, id, key, value_json);
void     grafeo_set_edge_property(db, id, key, value_json);
bool     grafeo_delete_node(db, id);
bool     grafeo_delete_edge(db, id);
```

## Transactions

```c
GrafeoTransaction* grafeo_begin_tx(db);
GrafeoResult*      grafeo_tx_execute(tx, query);
int                grafeo_commit(tx);
int                grafeo_rollback(tx);
```

## Vector Search

```c
int grafeo_create_vector_index(db, label, property, dims, metric, m, ef);
GrafeoResult* grafeo_vector_search(db, label, property, query, dims, k, ef);
GrafeoResult* grafeo_mmr_search(db, label, property, query, dims, k, fetch_k, lambda, ef);
```

## Error Handling

Functions return `NULL` on error. Check with `grafeo_last_error()`:

```c
GrafeoResult *r = grafeo_execute(db, query);
if (!r) {
    fprintf(stderr, "Error: %s\n", grafeo_last_error());
}
```

## Memory Management

- All `Grafeo*` pointers must be freed with their `grafeo_free_*` function
- String pointers from result accessors are valid until the parent is freed

## Links

- [GitHub](https://github.com/GrafeoDB/grafeo/tree/main/crates/bindings/c)
- [Go bindings](https://github.com/GrafeoDB/grafeo/tree/main/crates/bindings/go) (built on this library)
