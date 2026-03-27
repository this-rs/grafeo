---
title: Development Setup
description: Setting up the development environment.
tags:
  - contributing
---

# Development Setup

## Prerequisites

- Rust 1.91.1+
- Python 3.12+ (for Python bindings)
- Node.js 20+ (for Node.js bindings)
- Git

## Clone Repository

```bash
git clone https://github.com/ObrainDB/obrain.git
cd obrain
```

## Build

```bash
# Build all crates
cargo build --workspace

# Build in release mode
cargo build --workspace --release
```

## Run Tests

```bash
cargo test --workspace
```

## Build Python Package

```bash
cd crates/bindings/python
uv add maturin
maturin develop
```

## Build Node.js Package

```bash
cd crates/bindings/node
npm install
npm run build
npm test
```

## IDE Setup

### VS Code

Recommended extensions:

- rust-analyzer
- Python
- TOML

### IntelliJ/CLion

- Install Rust plugin
- Open as Cargo project
