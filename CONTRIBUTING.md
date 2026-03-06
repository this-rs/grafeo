# Contributing to Grafeo

Thanks for wanting to help out! Here's what you need to know.

## Setup

```bash
git clone https://github.com/GrafeoDB/grafeo.git
cd grafeo
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

See [ARCHITECTURE.md](.claude/ARCHITECTURE.md) for the full picture. The short version:

| Crate | What it does |
| ----- | ------------ |
| `grafeo` | Top-level facade, re-exports public API |
| `grafeo-common` | Foundation types, memory, utilities |
| `grafeo-core` | Graph storage, indexes, execution |
| `grafeo-adapters` | Query parsers (GQL, Cypher, Gremlin, GraphQL, SPARQL, SQL/PGQ) |
| `grafeo-engine` | Database facade, sessions, transactions |
| `grafeo-cli` | CLI with interactive shell |
| `grafeo-bindings-common` | Shared library for all language bindings |
| `grafeo-python` | Python bindings (PyO3) |
| `grafeo-node` | Node.js/TypeScript bindings (napi-rs) |
| `grafeo-c` | C FFI layer (also used by Go via CGO) |
| `grafeo-wasm` | WebAssembly bindings (wasm-bindgen) |

## Code Style

- Standard Rust conventions: `rustfmt` and `clippy` are enforced in CI
- Use `thiserror` for error types
- Tests go in the same file under `#[cfg(test)]`
- Descriptive test names: `test_<function>_<scenario>`

See [CODE_STYLE.md](.claude/CODE_STYLE.md) for the full guide.

## Python Bindings

```bash
cd crates/bindings/python
maturin develop
pytest crates/bindings/python/tests/ -v --ignore=crates/bindings/python/tests/benchmark_phases.py
```

## Node.js Bindings

```bash
cd crates/bindings/node
npm install
npm run build
npm test
```

## Ecosystem Projects

These companion projects live in separate repositories under the [GrafeoDB](https://github.com/GrafeoDB) organization:

| Project | Description |
| ------- | ----------- |
| [grafeo-server](https://github.com/GrafeoDB/grafeo-server) | HTTP server & web UI |
| [grafeo-web](https://github.com/GrafeoDB/grafeo-web) | Browser-based Grafeo (WASM) |
| [gwp](https://github.com/GrafeoDB/gql-wire-protocol) | GQL Wire Protocol (gRPC) |
| [boltr](https://github.com/GrafeoDB/boltr) | Bolt v5.x Wire Protocol |
| [grafeo-memory](https://github.com/GrafeoDB/grafeo-memory) | AI memory layer for LLM applications |
| [grafeo-langchain](https://github.com/GrafeoDB/grafeo-langchain) | LangChain graph + vector store |
| [grafeo-llamaindex](https://github.com/GrafeoDB/grafeo-llamaindex) | LlamaIndex PropertyGraphStore |
| [grafeo-mcp](https://github.com/GrafeoDB/grafeo-mcp) | MCP server for LLM agents |
| [anywidget-graph](https://github.com/GrafeoDB/anywidget-graph) | Graph visualization widget |
| [anywidget-vector](https://github.com/GrafeoDB/anywidget-vector) | Vector visualization widget |
| [graph-bench](https://github.com/GrafeoDB/graph-bench) | Benchmark suite |
| [ann-benchmarks](https://github.com/GrafeoDB/ann-benchmarks) | Vector search benchmarking |

## Pre-commit Hooks (Optional)

```bash
cargo install prek
prek install
```

This runs format, lint, and license checks automatically before each commit.

## Links

- [Repository](https://github.com/GrafeoDB/grafeo)
- [Issues](https://github.com/GrafeoDB/grafeo/issues)
- [Documentation](https://grafeo.dev)

## License

By contributing, you agree that your contributions will be licensed under Apache-2.0.
