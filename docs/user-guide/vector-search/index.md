---
title: Vector Search
description: Semantic similarity search with vector embeddings in Obrain.
tags:
  - vector-search
  - embeddings
  - similarity
---

# Vector Search

Obrain provides first-class support for vector similarity search, enabling semantic search, recommendation systems and AI-powered applications.

## Overview

Vector search finds nodes based on the semantic similarity of their embeddings rather than exact property matches. This is essential for:

- **Semantic search** - Find documents by meaning, not keywords
- **Recommendations** - Suggest similar items based on embeddings
- **RAG applications** - Retrieve relevant context for LLMs
- **Hybrid queries** - Combine graph traversal with vector similarity

## Key Features

| Feature | Description |
| ------- | ----------- |
| **HNSW Index** | O(log n) approximate nearest neighbor search |
| **Distance Metrics** | Cosine, Euclidean, Dot Product, Manhattan |
| **Quantization** | Scalar (4x), Binary (32x), Product (8-192x) compression |
| **Filtered Search** | Property equality filters via pre-computed ID allowlists |
| **MMR Search** | Maximal Marginal Relevance for diverse RAG retrieval |
| **Incremental Indexing** | Indexes stay in sync automatically as nodes change |
| **Batch Operations** | `batch_create_nodes()` and `batch_vector_search()` |
| **Hybrid Queries** | Combine graph patterns with vector similarity |
| **BM25 Text Search** | Full-text keyword search with inverted indexes |
| **Hybrid Search** | Combined text + vector search with RRF or weighted fusion |
| **Built-in Embeddings** | In-process ONNX embedding generation (opt-in `embed` feature) |
| **SIMD Acceleration** | AVX2, SSE, NEON optimized distance computation |

## Quick Example

```python
import obrain

db = obrain.ObrainDB()

# Create nodes with embeddings
db.execute("""
    INSERT (:Document {
        title: 'Introduction to Graphs',
        embedding: [0.1, 0.2, 0.3, 0.4, 0.5]
    })
""")

# Find similar documents
query_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
result = db.execute("""
    MATCH (d:Document)
    RETURN d.title, cosine_similarity(d.embedding, $query) AS similarity
    ORDER BY similarity DESC
    LIMIT 10
""", {"query": query_embedding})

for row in result:
    print(f"{row['d.title']}: {row['similarity']:.3f}")
```

## Text Search (BM25)

Create inverted indexes for full-text keyword search with BM25 scoring:

```python
db.create_text_index("Document", "content")
results = db.text_search("Document", "content", "graph database", k=10)
for r in results:
    print(f"Node {r['node_id']}: score {r['score']:.3f}")
```

Text indexes stay in sync automatically as nodes are created, updated or deleted.

## Hybrid Search

Combine BM25 text scores with HNSW vector similarity via Reciprocal Rank Fusion:

```python
results = db.hybrid_search(
    label="Document",
    text_property="content", text_query="graph database",
    vector_property="embedding", vector_query=query_vec,
    k=10,
)
```

## Built-in Embeddings

Generate embeddings in-process with ONNX Runtime (requires the `embed` feature):

```python
from obrain import load_embedding_model, EmbeddingModelConfig

model = load_embedding_model(EmbeddingModelConfig.MiniLM_L6_v2)
vectors = model.embed(["graph databases are fast", "hello world"])
```

Three presets are available: MiniLM-L6-v2 (22M params), MiniLM-L12-v2 (33M) and BGE-small-en-v1.5 (33M). Models are auto-downloaded from HuggingFace Hub on first use.

## Documentation

- [**Getting Started**](basics.md) - Store and query vector embeddings
- [**HNSW Index**](hnsw-index.md) - Configure the approximate nearest neighbor index
- [**Quantization**](quantization.md) - Compress vectors for memory efficiency
- [**Python API**](python-api.md) - Python bindings for vector operations

## Performance

Benchmark results on 384-dimensional vectors:

| Operation | Performance |
| --------- | ----------- |
| Distance computation (cosine) | 38 ns |
| Brute-force k-NN (10k vectors) | 308 µs |
| HNSW search (5k vectors, k=10) | 108 µs |
| PQ distance with table | 4.5 ns |

## When to Use Vector Search

**Use vector search when:**

- Semantic/meaning-based retrieval is needed
- Working with embeddings from ML models (OpenAI, Sentence Transformers, etc.)
- Building recommendation or similarity features
- Implementing RAG (Retrieval-Augmented Generation)

**Use traditional queries when:**

- Exact property matches are needed
- Working with structured data (names, IDs, dates)
- Relationships matter more than similarity
