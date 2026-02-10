# @grafeo-db/cli

Command-line interface for [Grafeo](https://grafeo.dev) graph database, distributed via npm.

## Installation

```bash
npm install -g @grafeo-db/cli
# or
npx @grafeo-db/cli --help
```

The correct platform-specific binary is installed automatically via optional dependencies.

## Supported Platforms

| Platform | Package |
|----------|---------|
| Linux x64 | `@grafeo-db/cli-linux-x64` |
| Linux ARM64 | `@grafeo-db/cli-linux-arm64` |
| macOS x64 | `@grafeo-db/cli-darwin-x64` |
| macOS ARM64 | `@grafeo-db/cli-darwin-arm64` |
| Windows x64 | `@grafeo-db/cli-win32-x64` |

## Usage

```bash
# Database management
grafeo info ./my-db
grafeo stats ./my-db
grafeo validate ./my-db

# Query execution
grafeo query ./my-db "MATCH (n:Person) RETURN n.name"

# Interactive shell
grafeo shell ./my-db

# Create a new database
grafeo init ./new-db
```

## Alternative Installation

```bash
# Via Rust
cargo install grafeo-cli

# Via pip
pip install grafeo-cli

# Direct download
# https://github.com/GrafeoDB/grafeo/releases
```

## Links

- [Documentation](https://grafeo.dev)
- [GitHub](https://github.com/GrafeoDB/grafeo)

## License

AGPL-3.0-or-later
