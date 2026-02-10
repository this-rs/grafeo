# grafeo-cli

Command-line interface for [Grafeo](https://grafeo.dev) graph database.

This is a thin Python launcher package that runs the pre-built Grafeo CLI binary.

## Installation

```bash
pip install grafeo-cli
```

The package looks for the `grafeo` binary in this order:
1. Bundled with the wheel (platform-specific wheels)
2. In the virtualenv `bin/` or `Scripts/` directory
3. On your system `PATH`

If no binary is found, install it separately:

```bash
# Via cargo
cargo install grafeo-cli

# Or download from GitHub releases
# https://github.com/GrafeoDB/grafeo/releases
```

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

# Shell completions
grafeo completions bash > ~/.local/share/bash-completion/completions/grafeo
```

## Links

- [Documentation](https://grafeo.dev)
- [GitHub](https://github.com/GrafeoDB/grafeo)
- [Grafeo Python Library](https://pypi.org/project/grafeo/) (the database engine, not this CLI tool)

## License

AGPL-3.0-or-later
