# obrain-cli

Command-line interface for [Obrain](https://obrain.dev) graph database.

This is a thin Python launcher package that runs the pre-built Obrain CLI binary.

## Installation

```bash
uv add obrain-cli
# or: pip install obrain-cli
```

The package looks for the `obrain` binary in this order:
1. Bundled with the wheel (platform-specific wheels)
2. In the virtualenv `bin/` or `Scripts/` directory
3. On your system `PATH`

If no binary is found, install it separately:

```bash
# Via cargo
cargo install obrain-cli

# Or download from GitHub releases
# https://github.com/this-rs/obrain/releases
```

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

# Shell completions
obrain completions bash > ~/.local/share/bash-completion/completions/obrain
```

## Links

- [Documentation](https://obrain.dev)
- [GitHub](https://github.com/this-rs/obrain)
- [Obrain Python Library](https://pypi.org/project/obrain/) (the database engine, not this CLI tool)

## License

Apache-2.0
