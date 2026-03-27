# Contributing to Obrain

Thanks for wanting to help out! Here's what you need to know.

## Setup

```bash
git clone https://github.com/ObrainDB/obrain.git
cd obrain
cargo build --workspace
```

You'll need **Rust 1.91.1+** and optionally **Python 3.12+** / **Node.js 20+** for the bindings.

## Branching

We use feature branches off `main`:

- `feature/<description>` for new functionality
- `fix/<description>` for bug fixes
- `release/<version>` for release stabilization

Create your branch from `main`, open a PR back to `main` when ready.

## Making Changes

1. Create a branch: `git checkout -b feature/my-thing`
2. Write code and tests
3. Run checks: `./scripts/ci-local.sh` (or `.\scripts\ci-local.ps1` on Windows)
4. Push and open a PR

You can also run checks individually:

```bash
cargo fmt --all              # Format
cargo clippy --all-targets --all-features -- -D warnings  # Lint
cargo test --all-features --workspace     # Test
```

### Commit Messages

We use conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `perf:`, `ci:`.

## Architecture

| Crate | What it does |
| ----- | ------------ |
| `obrain` | Top-level facade, re-exports public API |
| `obrain-common` | Foundation types, memory, utilities |
| `obrain-core` | Graph storage, indexes, execution |
| `obrain-adapters` | Query parsers (GQL, Cypher, Gremlin, GraphQL, SPARQL, SQL/PGQ) |
| `obrain-engine` | Database facade, sessions, transactions |
| `obrain-cli` | CLI with interactive shell |
| `obrain-bindings-common` | Shared library for all language bindings |
| `obrain-python` | Python bindings (PyO3) |
| `obrain-node` | Node.js/TypeScript bindings (napi-rs) |
| `obrain-c` | C FFI layer (also used by Go via CGO) |
| `obrain-wasm` | WebAssembly bindings (wasm-bindgen) |
| `obrain-csharp` | C# / .NET 8 bindings (P/Invoke, wraps obrain-c) |
| `obrain-dart` | Dart bindings (dart:ffi, wraps obrain-c) |

## Code Style

- Standard Rust conventions: `rustfmt` and `clippy` are enforced in CI
- Use `thiserror` for error types
- Tests go in the same file under `#[cfg(test)]`
- Descriptive test names: `test_<function>_<scenario>`

## Python Bindings

```bash
cd crates/bindings/python
maturin develop
pytest tests/ -v --ignore=tests/benchmark_phases.py
```

## Node.js Bindings

```bash
cd crates/bindings/node
npm install
npm run build
npm test
```

## Ecosystem Projects

These companion projects live in separate repositories under the [ObrainDB](https://github.com/ObrainDB) organization:

| Project | Description |
| ------- | ----------- |
| [obrain-server](https://github.com/ObrainDB/obrain-server) | HTTP server & web UI |
| [obrain-web](https://github.com/ObrainDB/obrain-web) | Browser-based Obrain (WASM) |
| [gwp](https://github.com/ObrainDB/gql-wire-protocol) | GQL Wire Protocol (gRPC) |
| [boltr](https://github.com/ObrainDB/boltr) | Bolt v5.x Wire Protocol |
| [obrain-memory](https://github.com/ObrainDB/obrain-memory) | AI memory layer for LLM applications |
| [obrain-langchain](https://github.com/ObrainDB/obrain-langchain) | LangChain graph + vector store |
| [obrain-llamaindex](https://github.com/ObrainDB/obrain-llamaindex) | LlamaIndex PropertyGraphStore |
| [obrain-mcp](https://github.com/ObrainDB/obrain-mcp) | MCP server for LLM agents |
| [anywidget-graph](https://github.com/ObrainDB/anywidget-graph) | Graph visualization widget |
| [anywidget-vector](https://github.com/ObrainDB/anywidget-vector) | Vector visualization widget |
| [graph-bench](https://github.com/ObrainDB/graph-bench) | Benchmark suite |
| [ann-benchmarks](https://github.com/ObrainDB/ann-benchmarks) | Vector search benchmarking |

## Pre-commit Hooks (Optional)

```bash
cargo install prek
prek install
```

This runs format, lint and license checks automatically before each commit.

## Links

- [Repository](https://github.com/ObrainDB/obrain)
- [Issues](https://github.com/ObrainDB/obrain/issues)
- [Documentation](https://obrain.dev)

## License

By contributing, you agree that your contributions will be licensed under Apache-2.0.
