---
title: WebAssembly API
description: API reference for the @grafeo-db/wasm package.
---

# WebAssembly API

Run Grafeo in the browser, Deno, or Cloudflare Workers. ~660 KB gzipped.

```bash
npm install @grafeo-db/wasm
```

## Quick Start

```javascript
import init, { Database } from '@grafeo-db/wasm';

await init();
const db = new Database();

db.execute("INSERT (:Person {name: 'Alice', age: 30})");
db.execute("INSERT (:Person {name: 'Bob', age: 25})");

const results = db.execute("MATCH (p:Person) RETURN p.name, p.age");
console.log(results); // [{name: "Alice", age: 30}, {name: "Bob", age: 25}]
```

## Database

```javascript
const db = new Database();   // in-memory (all WASM databases are in-memory)
```

## Query Methods

```javascript
db.execute(gql);                 // GQL — returns array of row objects
db.executeRaw(gql);             // GQL — returns {columns, rows, executionTimeMs}
db.executeWithLanguage(query, language);  // "gql", "cypher", "graphql", etc.
```

## Properties

```javascript
db.nodeCount();   // number of nodes
db.edgeCount();   // number of edges
db.schema();      // database schema as JSON
```

## Snapshots (Persistence)

Export/import the entire database as a binary snapshot for IndexedDB persistence:

```javascript
// Export
const snapshot = db.exportSnapshot();
// Store in IndexedDB...

// Import
const db2 = Database.importSnapshot(snapshot);
```

## Supported Query Languages

The WASM build supports query languages based on compile-time features:

| Feature | Language | Default |
|---------|----------|---------|
| `gql` | GQL | Yes |
| `cypher` | Cypher | No |
| `sparql` | SPARQL | No |
| `gremlin` | Gremlin | No |
| `graphql` | GraphQL | No |
| `sql-pgq` | SQL/PGQ | No |

The `full` feature enables all languages. The default npm package includes only GQL to minimize bundle size.

## Bundle Size

| Build | Size |
|-------|------|
| Default (GQL only) | ~660 KB gzipped |
| Full (all languages) | ~800 KB gzipped |

## Links

- [npm package](https://www.npmjs.com/package/@grafeo-db/wasm)
- [GitHub](https://github.com/GrafeoDB/grafeo/tree/main/crates/bindings/wasm)
