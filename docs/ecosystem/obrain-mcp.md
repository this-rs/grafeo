---
title: obrain-mcp
description: Model Context Protocol server for exposing ObrainDB to AI agents.
---

# obrain-mcp

A [Model Context Protocol](https://modelcontextprotocol.io/) (MCP) server that exposes ObrainDB to AI agents. Zero infrastructure required - the server embeds the database directly.

[:octicons-mark-github-16: GitHub](https://github.com/this-rs/obrain-mcp){ .md-button }
[:material-package-variant: PyPI](https://pypi.org/project/obrain-mcp/){ .md-button }

## Overview

obrain-mcp lets AI agents (Claude, Copilot, etc.) query and manipulate a Obrain graph database through the MCP standard. It runs as a standalone executable with no separate database server needed.

## Installation

```bash
uv tool install obrain-mcp
# or
pip install obrain-mcp
```

Requires Python 3.12+ and obrain >= 0.4.4.

## Quick Start

### Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "obrain": {
      "command": "obrain-mcp"
    }
  }
}
```

### With persistent storage

```bash
OBRAIN_DB_PATH=./my-graph.db obrain-mcp
```

## MCP Tools

obrain-mcp exposes 16 tools organized in 4 categories:

### Query

- **`query`** - Execute GQL queries with auto-normalization of Cypher syntax

### Graph CRUD & Traversal

- **`create_node`** - Create a node with labels and properties
- **`create_edge`** - Create a relationship between nodes
- **`get_node`** - Retrieve a node by ID with properties and connections
- **`search_nodes`** - Find nodes by label, with optional property filters
- **`get_neighbors`** - Explore a node's neighborhood (in/out/both)
- **`get_schema`** - Discover labels, edge types and property keys
- **`get_stats`** - Database statistics (counts, memory, configuration)

### Vector Search

- **`vector_search`** - k-NN similarity search using HNSW indexes
- **`mmr_search`** - MMR-diversified search for RAG pipelines
- **`create_vector_index`** - Create an HNSW index on a property
- **`hybrid_search`** - Combine vector similarity with graph traversal

### Graph Algorithms

- **`pagerank`** - Compute PageRank centrality scores
- **`shortest_path`** - Dijkstra shortest path between nodes
- **`community_detection`** - Louvain community detection
- **`centrality`** - Betweenness centrality and connected components

## MCP Resources

- `graph://schema` - Rich schema with labels, properties and edge types
- `graph://stats` - Database statistics (counts, memory, disk, config)
- `graph://nodes/{id}` - Individual node details and connection summary

## Configuration

| Environment Variable | Default | Description |
|----------------------|---------|-------------|
| `OBRAIN_DB_PATH` | *(in-memory)* | Path to persistent database file |

## Transport

Supports stdio (default), SSE and streamable HTTP transports.

## Requirements

- Python 3.12+
- obrain >= 0.4.4
- mcp >= 1.20

## See Also

- **[obrain-memory](obrain-memory.md)** includes a built-in MCP server (`obrain-memory-mcp`) that wraps the high-level memory API - extract, reconcile, search, summarize. If you need AI memory management rather than raw graph access, use that instead.

## License

Apache-2.0
