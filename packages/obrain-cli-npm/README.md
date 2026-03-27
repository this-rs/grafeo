# @obrain-db/cli

Command-line interface for [Obrain](https://obrain.dev) graph database, distributed via npm.

## Installation

```bash
npm install -g @obrain-db/cli
# or
npx @obrain-db/cli --help
```

The correct platform-specific binary is installed automatically via optional dependencies.

## Supported Platforms

| Platform | Package |
|----------|---------|
| Linux x64 | `@obrain-db/cli-linux-x64` |
| Linux ARM64 | `@obrain-db/cli-linux-arm64` |
| macOS x64 | `@obrain-db/cli-darwin-x64` |
| macOS ARM64 | `@obrain-db/cli-darwin-arm64` |
| Windows x64 | `@obrain-db/cli-win32-x64` |

## Usage

```bash
# Database management
obrain info ./my-db
obrain stats ./my-db
obrain validate ./my-db

# Query execution
obrain query ./my-db "MATCH (n:Person) RETURN n.name"

# Interactive shell
obrain shell ./my-db

# Create a new database
obrain init ./new-db
```

## Alternative Installation

```bash
# Via Rust
cargo install obrain-cli

# Via pip/uv
uv add obrain-cli

# Direct download
# https://github.com/this-rs/obrain/releases
```

## Links

- [Documentation](https://obrain.dev)
- [GitHub](https://github.com/this-rs/obrain)

## License

Apache-2.0
