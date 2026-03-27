---
title: Python API
description: Python API reference.
---

# Python API Reference

Complete reference for the `obrain` Python package.

## Installation

```bash
uv add obrain
```

## Quick Start

```python
import obrain

db = obrain.ObrainDB()
db.execute("INSERT (:Person {name: 'Alix'})")
```

## Classes

| Class | Description |
|-------|-------------|
| [Database](database.md) | Database connection and management |
| [Node](node.md) | Graph node representation |
| [Edge](edge.md) | Graph edge representation |
| [QueryResult](result.md) | Query result iteration |
| [Transaction](transaction.md) | Transaction management |
