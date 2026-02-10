---
title: Node.js / TypeScript API
description: API reference for the @grafeo-db/js package.
---

# Node.js / TypeScript API

Native bindings for Grafeo via [napi-rs](https://napi.rs). Install from npm:

```bash
npm install @grafeo-db/js
```

## Quick Start

```typescript
import { GrafeoDB } from '@grafeo-db/js';

const db = GrafeoDB.create();

db.createNode(['Person'], { name: 'Alice', age: 30 });
db.createNode(['Person'], { name: 'Bob', age: 25 });
db.createEdge(0, 1, 'KNOWS', { since: 2024 });

const result = await db.execute('MATCH (p:Person) RETURN p.name, p.age');
console.log(result.toArray());

db.close();
```

## Database

```typescript
// Create / open
const db = GrafeoDB.create();           // in-memory
const db = GrafeoDB.create('./path');    // persistent
const db = GrafeoDB.open('./path');      // open existing

// Properties
db.nodeCount;   // number of nodes
db.edgeCount;   // number of edges
```

## Query Languages

All query methods return `Promise<QueryResult>`:

```typescript
await db.execute(gql, params?);         // GQL (ISO standard)
await db.executeCypher(query, params?);  // Cypher
await db.executeGremlin(query, params?); // Gremlin
await db.executeGraphql(query, params?); // GraphQL
await db.executeSparql(query);           // SPARQL
await db.executeSql(query);              // SQL/PGQ
```

## Node & Edge CRUD

```typescript
const id = db.createNode(['Label'], { key: 'value' });
const eid = db.createEdge(sourceId, targetId, 'TYPE', { key: 'value' });

const node = db.getNode(id);     // JsNode | null
const edge = db.getEdge(id);     // JsEdge | null

db.setNodeProperty(id, 'key', 'value');
db.setEdgeProperty(id, 'key', 'value');

db.deleteNode(id);  // returns boolean
db.deleteEdge(id);  // returns boolean
```

## Transactions

```typescript
const tx = db.beginTransaction();
try {
  await tx.execute("INSERT (:Person {name: 'Carol'})");
  tx.commit();
} catch (e) {
  tx.rollback();
}
```

## QueryResult

```typescript
result.columns;          // column names
result.length;           // row count
result.executionTimeMs;  // execution time (ms)
result.get(0);           // single row as object
result.toArray();        // all rows as objects
result.scalar();         // first column of first row
```

## Vector Search

```typescript
// Create HNSW index
await db.createVectorIndex('Document', 'embedding', 384);

// Bulk insert
const ids = await db.batchCreateNodes('Document', 'embedding', vectors);

// k-NN search
const results = await db.vectorSearch('Document', 'embedding', queryVec, 10);

// Filtered search
const results = await db.vectorSearch(
  'Document', 'embedding', queryVec, 10, undefined, { user_id: 1 }
);

// MMR search (diverse results)
const results = await db.mmrSearch(
  'Document', 'embedding', queryVec, 5, undefined, 0.5
);
```

## Type Mapping

| JavaScript | Grafeo |
|-----------|--------|
| `number` (integer) | Int64 |
| `number` (float) | Float64 |
| `string` | String |
| `boolean` | Boolean |
| `BigInt` | Int64 |
| `null` | Null |
| `Array<number>` | Vector |
| `Date` | String (ISO 8601) |
| `Buffer` | Bytes |

## Links

- [npm package](https://www.npmjs.com/package/@grafeo-db/js)
- [GitHub](https://github.com/GrafeoDB/grafeo/tree/main/crates/bindings/node)
