---
title: ObrainDB
description: ObrainDB class reference for Node.js.
tags:
  - api
  - nodejs
---

# ObrainDB

The main database class. All query methods return `Promise<QueryResult>`.

## Constructor

```typescript
// In-memory database
const db = ObrainDB.create();

// Persistent database
const db = ObrainDB.create('./my_graph.db');

// Open existing database
const db = ObrainDB.open('./my_graph.db');
```

### Parameters

| Method | Parameters | Description |
|--------|-----------|-------------|
| `create(path?)` | `path: string \| undefined` | Create a database (in-memory if no path) |
| `open(path)` | `path: string` | Open an existing database |

## Query Methods

All query methods accept an optional `params` object for parameterized queries.

### execute()

Execute a GQL (ISO standard) query.

```typescript
async execute(query: string, params?: object): Promise<QueryResult>
```

```typescript
const result = await db.execute(
  'MATCH (p:Person) WHERE p.age > $minAge RETURN p.name',
  { minAge: 25 }
);
```

### executeCypher()

Execute a Cypher query. Requires the `cypher` feature.

```typescript
async executeCypher(query: string, params?: object): Promise<QueryResult>
```

### executeGremlin()

Execute a Gremlin query. Requires the `gremlin` feature.

```typescript
async executeGremlin(query: string, params?: object): Promise<QueryResult>
```

### executeGraphql()

Execute a GraphQL query. Requires the `graphql` feature.

```typescript
async executeGraphql(query: string, params?: object): Promise<QueryResult>
```

### executeSparql()

Execute a SPARQL query against the RDF triple store. Requires the `sparql` feature.

```typescript
async executeSparql(query: string, params?: object): Promise<QueryResult>
```

### executeSql()

Execute a SQL/PGQ query (SQL:2023 GRAPH_TABLE). Requires the `sql-pgq` feature.

```typescript
async executeSql(query: string, params?: object): Promise<QueryResult>
```

```typescript
const result = await db.executeSql(
  'SELECT * FROM GRAPH_TABLE (MATCH (p:Person) COLUMNS (p.name AS name))'
);
```

## Node Operations

### createNode()

Create a node with labels and optional properties.

```typescript
createNode(labels: string[], properties?: object): JsNode
```

```typescript
const node = db.createNode(['Person'], { name: 'Alix', age: 30 });
console.log(node.id);     // 0
console.log(node.labels);  // ['Person']
```

### getNode()

Get a node by ID. Returns `null` if not found.

```typescript
getNode(id: number): JsNode | null
```

### deleteNode()

Delete a node by ID. Returns `true` if the node existed.

```typescript
deleteNode(id: number): boolean
```

### setNodeProperty()

Set a property on a node.

```typescript
setNodeProperty(id: number, key: string, value: any): void
```

### removeNodeProperty()

Remove a property from a node. Returns `true` if the property existed.

```typescript
removeNodeProperty(id: number, key: string): boolean
```

### addNodeLabel()

Add a label to an existing node. Returns `true` if the label was added.

```typescript
addNodeLabel(id: number, label: string): boolean
```

### removeNodeLabel()

Remove a label from a node. Returns `true` if the label was removed.

```typescript
removeNodeLabel(id: number, label: string): boolean
```

### getNodeLabels()

Get all labels for a node. Returns `null` if the node doesn't exist.

```typescript
getNodeLabels(id: number): string[] | null
```

## Edge Operations

### createEdge()

Create an edge between two nodes with a type and optional properties.

```typescript
createEdge(sourceId: number, targetId: number, edgeType: string, properties?: object): JsEdge
```

```typescript
const edge = db.createEdge(0, 1, 'KNOWS', { since: 2024 });
console.log(edge.edgeType);  // 'KNOWS'
console.log(edge.sourceId);  // 0
console.log(edge.targetId);  // 1
```

### getEdge()

Get an edge by ID. Returns `null` if not found.

```typescript
getEdge(id: number): JsEdge | null
```

### deleteEdge()

Delete an edge by ID. Returns `true` if the edge existed.

```typescript
deleteEdge(id: number): boolean
```

### setEdgeProperty()

Set a property on an edge.

```typescript
setEdgeProperty(id: number, key: string, value: any): void
```

### removeEdgeProperty()

Remove a property from an edge. Returns `true` if the property existed.

```typescript
removeEdgeProperty(id: number, key: string): boolean
```

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `nodeCount` | `number` | Number of nodes in the database |
| `edgeCount` | `number` | Number of edges in the database |

## Transaction Methods

### beginTransaction()

Start a new transaction with an optional isolation level.

```typescript
beginTransaction(isolationLevel?: string): Transaction
```

Isolation levels: `"read_committed"`, `"snapshot"` (default), `"serializable"`.

```typescript
const tx = db.beginTransaction();
const tx = db.beginTransaction('serializable');
```

## Vector Search

### createVectorIndex()

Create an HNSW vector similarity index on a node property.

```typescript
async createVectorIndex(
  label: string,
  property: string,
  dimensions?: number,
  metric?: string,     // 'cosine' (default), 'euclidean', 'dot'
  m?: number,          // connections per node (default: 16)
  efConstruction?: number  // build quality (default: 128)
): Promise<void>
```

### dropVectorIndex()

Drop a vector index. Returns `true` if the index existed.

```typescript
async dropVectorIndex(label: string, property: string): Promise<boolean>
```

### rebuildVectorIndex()

Rebuild a vector index by rescanning all matching nodes.

```typescript
async rebuildVectorIndex(label: string, property: string): Promise<void>
```

### vectorSearch()

Search for the k nearest neighbors of a query vector.
Returns `[[nodeId, distance], ...]` sorted by distance.

```typescript
async vectorSearch(
  label: string,
  property: string,
  query: number[],
  k: number,
  ef?: number,
  filters?: Record<string, any>
): Promise<number[][]>
```

```typescript
const results = await db.vectorSearch('Document', 'embedding', queryVec, 10);
for (const [nodeId, distance] of results) {
  console.log(`Node ${nodeId}: distance ${distance}`);
}

// With metadata filter
const filtered = await db.vectorSearch(
  'Document', 'embedding', queryVec, 10, undefined, { user_id: 1 }
);
```

### batchCreateNodes()

Bulk-insert nodes with vector properties. Returns an array of node IDs.

```typescript
async batchCreateNodes(
  label: string,
  property: string,
  vectors: number[][]
): Promise<number[]>
```

### batchVectorSearch()

Batch search for nearest neighbors of multiple query vectors.

```typescript
async batchVectorSearch(
  label: string,
  property: string,
  queries: number[][],
  k: number,
  ef?: number,
  filters?: Record<string, any>
): Promise<number[][][]>
```

### mmrSearch()

Search for diverse nearest neighbors using Maximal Marginal Relevance.

```typescript
async mmrSearch(
  label: string,
  property: string,
  query: number[],
  k: number,
  fetchK?: number,
  lambdaMult?: number,  // diversity vs relevance (0 = max diversity, 1 = max relevance)
  ef?: number,
  filters?: Record<string, any>
): Promise<number[][]>
```

## Text Search

### createTextIndex()

Create a BM25 text index on a node property for full-text search.

```typescript
async createTextIndex(label: string, property: string): Promise<void>
```

### dropTextIndex()

Drop a text index. Returns `true` if the index existed.

```typescript
async dropTextIndex(label: string, property: string): Promise<boolean>
```

### rebuildTextIndex()

Rebuild a text index by rescanning all matching nodes.

```typescript
async rebuildTextIndex(label: string, property: string): Promise<void>
```

### textSearch()

Search a text index using BM25 scoring. Returns `[[nodeId, score], ...]`.

```typescript
async textSearch(
  label: string,
  property: string,
  query: string,
  k: number
): Promise<number[][]>
```

### hybridSearch()

Combine text (BM25) and vector similarity search. Returns `[[nodeId, score], ...]`.

```typescript
async hybridSearch(
  label: string,
  textProperty: string,
  vectorProperty: string,
  queryText: string,
  k: number,
  queryVector?: number[],
  fusion?: string,         // 'weighted' for weighted fusion
  weights?: number[]       // [textWeight, vectorWeight], default [0.5, 0.5]
): Promise<number[][]>
```

## Embedding (opt-in)

These methods require the `embed` feature flag.

### registerEmbeddingModel()

Register an ONNX embedding model for text-to-vector conversion.

```typescript
async registerEmbeddingModel(
  name: string,
  modelPath: string,
  tokenizerPath: string,
  batchSize?: number
): Promise<void>
```

### embedText()

Generate embeddings for a list of texts. Returns one float array per input text.

```typescript
async embedText(modelName: string, texts: string[]): Promise<number[][]>
```

### vectorSearchText()

Search a vector index using a text query, generating the embedding on-the-fly.

```typescript
async vectorSearchText(
  label: string,
  property: string,
  modelName: string,
  queryText: string,
  k: number,
  ef?: number
): Promise<number[][]>
```

## Change Data Capture

These methods require the `cdc` feature flag.

### nodeHistory()

Returns the full change history for a node.

```typescript
async nodeHistory(nodeId: number): Promise<ChangeEvent[]>
```

### edgeHistory()

Returns the full change history for an edge.

```typescript
async edgeHistory(edgeId: number): Promise<ChangeEvent[]>
```

### nodeHistorySince()

Returns change events for a node since a given epoch.

```typescript
async nodeHistorySince(nodeId: number, sinceEpoch: number): Promise<ChangeEvent[]>
```

### changesBetween()

Returns all change events across entities in an epoch range.

```typescript
async changesBetween(startEpoch: number, endEpoch: number): Promise<ChangeEvent[]>
```

Each `ChangeEvent` is a JSON object:

```typescript
{
  entity_id: number;
  entity_type: 'node' | 'edge';
  kind: 'create' | 'update' | 'delete';
  epoch: number;
  timestamp: number;
  before: Record<string, any> | null;
  after: Record<string, any> | null;
}
```

## Admin Methods

### info()

Returns high-level database information as a JSON object.

```typescript
info(): object
```

### schema()

Returns schema information (labels, edge types, property keys).

```typescript
schema(): object
```

### version()

Returns the Obrain engine version string.

```typescript
version(): string
```

### close()

Close the database and release resources.

```typescript
close(): void
```
