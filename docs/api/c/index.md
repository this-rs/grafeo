---
title: C API
description: API reference for the obrain-c FFI bindings.
---

# C API

C-compatible FFI layer for embedding Obrain in any language. Used by the Go bindings via CGO.

## Building

```bash
cargo build --release -p obrain-c --features full
```

Output:

- `target/release/libobrain_c.so` (Linux)
- `target/release/libobrain_c.dylib` (macOS)
- `target/release/obrain_c.dll` (Windows)

Header: `crates/bindings/c/obrain.h`

## Quick Start

```c
#include "obrain.h"
#include <stdio.h>

int main(void) {
    ObrainDatabase *db = obrain_open_memory();
    if (!db) {
        fprintf(stderr, "Error: %s\n", obrain_last_error());
        return 1;
    }

    ObrainResult *r = obrain_execute(db, "MATCH (p:Person) RETURN p.name");
    if (r) {
        printf("Rows: %zu\n", obrain_result_row_count(r));
        printf("JSON: %s\n", obrain_result_json(r));
        obrain_free_result(r);
    }

    obrain_free_database(db);
    return 0;
}
```

Compile:

```bash
gcc -o example example.c -lobrain_c -L/path/to/target/release
```

## Lifecycle

```c
ObrainDatabase* obrain_open_memory();           // in-memory
ObrainDatabase* obrain_open(const char* path);  // persistent
void obrain_free_database(ObrainDatabase* db);  // free handle
```

## Query Execution

```c
ObrainResult* obrain_execute(db, query);
ObrainResult* obrain_execute_with_params(db, query, params_json);
ObrainResult* obrain_execute_cypher(db, query);
ObrainResult* obrain_execute_gremlin(db, query);
ObrainResult* obrain_execute_graphql(db, query);
ObrainResult* obrain_execute_sparql(db, query);
ObrainResult* obrain_execute_sql(db, query);
```

## Result Access

```c
const char* obrain_result_json(const ObrainResult* r);
size_t      obrain_result_row_count(const ObrainResult* r);
double      obrain_result_execution_time_ms(const ObrainResult* r);
void        obrain_free_result(ObrainResult* r);
```

## Node & Edge CRUD

```c
uint64_t obrain_create_node(db, labels, label_count);
uint64_t obrain_create_edge(db, source, target, edge_type);
void     obrain_set_node_property(db, id, key, value_json);
void     obrain_set_edge_property(db, id, key, value_json);
bool     obrain_delete_node(db, id);
bool     obrain_delete_edge(db, id);
```

## Transactions

```c
ObrainTransaction* obrain_begin_tx(db);
ObrainResult*      obrain_tx_execute(tx, query);
int                obrain_commit(tx);
int                obrain_rollback(tx);
```

## Vector Search

```c
int obrain_create_vector_index(db, label, property, dims, metric, m, ef);
ObrainResult* obrain_vector_search(db, label, property, query, dims, k, ef);
ObrainResult* obrain_mmr_search(db, label, property, query, dims, k, fetch_k, lambda, ef);
```

## Error Handling

Functions return `NULL` on error. Check with `obrain_last_error()`:

```c
ObrainResult *r = obrain_execute(db, query);
if (!r) {
    fprintf(stderr, "Error: %s\n", obrain_last_error());
}
```

## Memory Management

- All `Obrain*` pointers must be freed with their `obrain_free_*` function
- String pointers from result accessors are valid until the parent is freed

## Links

- [GitHub](https://github.com/this-rs/obrain/tree/main/crates/bindings/c)
- [Go bindings](https://github.com/this-rs/obrain/tree/main/crates/bindings/go) (built on this library)
