---
title: Command-Line Interface
description: Obrain CLI for querying, inspecting and maintaining graph databases.
tags:
  - getting-started
  - cli
  - admin
---

# Command-Line Interface

Obrain ships a single Rust CLI binary (`obrain`) for querying, inspecting and maintaining databases. It includes an interactive REPL, admin commands and multiple output formats.

## Installation

Install via any of these methods: they all provide the same CLI:

=== "Cargo (Recommended)"

    ```bash
    cargo install obrain-cli
    ```

=== "pip / uv"

    ```bash
    uv add obrain-cli
    # or
    pip install obrain-cli
    ```

=== "npm"

    ```bash
    npm install -g @obrain-db/cli
    # or one-shot:
    npx @obrain-db/cli --version
    ```

=== "Download"

    Pre-built binaries for all platforms are attached to every
    [GitHub release](https://github.com/this-rs/obrain/releases).

Verify the installation:

```bash
obrain version
```

## Quick Start

```bash
# Create a new database
obrain init ./mydb

# Run a query
obrain query ./mydb "INSERT (:Person {name: 'Alix', age: 30})"
obrain query ./mydb "MATCH (n:Person) RETURN n.name, n.age"

# Launch the interactive shell
obrain shell ./mydb
```

## Commands

### Query Execution

```bash
# Inline query
obrain query ./mydb "MATCH (n) RETURN n LIMIT 10"

# From a file
obrain query ./mydb --file query.gql

# From stdin
echo "MATCH (n) RETURN count(n)" | obrain query ./mydb --stdin

# With parameters
obrain query ./mydb "MATCH (n {name: \$name}) RETURN n" -p name=Alix

# Choose query language (default: gql)
obrain query ./mydb "MATCH (n) RETURN n" --lang cypher
obrain query ./mydb "SELECT * FROM GRAPH_TABLE ..." --lang sql

# Show execution time
obrain query ./mydb "MATCH (n) RETURN n" --timing

# Truncate wide columns
obrain query ./mydb "MATCH (n) RETURN n" --max-width 40
```

### Interactive Shell (REPL)

```bash
obrain shell ./mydb
```

```
Obrain 0.5.10 - Lpg mode, 42 nodes, 87 edges
Type :help for commands, :quit to exit.

obrain> MATCH (n:Person) RETURN n.name, n.age
┌──────────┬───────┐
│ n.name   │ n.age │
├──────────┼───────┤
│ "Alix"  │ 30    │
│ "Gus"    │ 25    │
└──────────┴───────┘
2 rows (0.8ms)

obrain> :begin
Transaction started.
obrain[tx]> INSERT (:Person {name: 'Harm', age: 45})
obrain[tx]> :commit
Transaction committed.
```

**Meta-commands:**

| Command | Description |
|---------|-------------|
| `:help` | Show available commands |
| `:quit` / Ctrl-D | Exit the shell |
| `:schema` | Show labels, edge types, property keys |
| `:info` | Show database info |
| `:stats` | Show detailed statistics |
| `:format <f>` | Set output format (`table`, `json`, `csv`) |
| `:timing` | Toggle query timing display |
| `:begin` | Start a transaction |
| `:commit` | Commit the current transaction |
| `:rollback` | Roll back the current transaction |

Transaction keywords (`BEGIN`, `COMMIT`, `ROLLBACK`) also work as plain text.

### Database Creation

```bash
# Create an LPG database (default)
obrain init ./mydb

# Create an RDF database
obrain init ./mydb --mode rdf
```

### Inspection

```bash
# Overview: counts, size, mode
obrain info ./mydb

# Detailed statistics
obrain stats ./mydb

# Schema: labels, edge types, property keys
obrain schema ./mydb

# Integrity check (exit code 2 on failure)
obrain validate ./mydb
```

### Index Management

```bash
obrain index list ./mydb
obrain index stats ./mydb
```

### Backup & Restore

```bash
obrain backup create ./mydb -o backup.obrain
obrain backup restore backup.obrain ./restored --force
```

### Data Export & Import

```bash
obrain data dump ./mydb -o ./export/
obrain data load ./export/ ./newdb
```

### WAL Management

```bash
obrain wal status ./mydb
obrain wal checkpoint ./mydb
```

### Compaction

```bash
obrain compact ./mydb
obrain compact ./mydb --dry-run
```

### Shell Completions

```bash
# Generate completions for your shell
obrain completions bash > ~/.local/share/bash-completion/completions/obrain
obrain completions zsh > ~/.zfunc/_obrain
obrain completions fish > ~/.config/fish/completions/obrain.fish
obrain completions powershell >> $PROFILE
```

### Version & Build Info

```bash
$ obrain version
obrain 0.5.10

Build:
  rustc:    1.91.1
  target:   x86_64
  os:       linux
  features: gql, cypher, sparql, sql-pgq

Paths:
  config:   /home/user/.config/obrain
  history:  /home/user/.config/obrain/history
```

## Output Formats

All commands support multiple output formats:

```bash
# Auto-detect: table on TTY, JSON when piped
obrain info ./mydb

# Explicit format
obrain info ./mydb --format table
obrain info ./mydb --format json
obrain info ./mydb --format csv
```

## Global Options

| Option | Description |
|--------|-------------|
| `--format <auto\|table\|json\|csv>` | Output format (default: `auto`) |
| `--quiet`, `-q` | Suppress progress messages |
| `--verbose`, `-v` | Enable debug logging |
| `--no-color` | Disable colored output (also respects `NO_COLOR` env var) |
| `--color` | Force colored output even when piped |
| `--help` | Show help |
| `--version` | Show version |

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error (runtime, I/O, query) |
| 2 | Validation failed (`obrain validate` found errors) |

## Python API Equivalents

The Python API provides the same functionality programmatically:

```python
import obrain

db = obrain.ObrainDB("./mydb")

# Equivalent to: obrain info ./mydb
print(db.info())

# Equivalent to: obrain stats ./mydb
print(db.detailed_stats())

# Equivalent to: obrain schema ./mydb
print(db.schema())

# Equivalent to: obrain validate ./mydb
print(db.validate())

# Equivalent to: obrain query ./mydb "MATCH (n) RETURN n"
result = db.execute("MATCH (n) RETURN n")
```

## Migrating from the Python CLI

!!! note "Python CLI removed in 0.4.4"
    The `obrain[cli]` Python CLI (Click-based) has been removed. Install the Rust binary
    instead via `cargo install obrain-cli`, `pip install obrain-cli`, or
    `npm install -g @obrain-db/cli`. All previous commands are available with the same
    syntax, plus new features: `query`, `shell`, `init`, CSV output and shell completions.
