//! Grafeo CLI - Command-line interface for Grafeo graph databases.
//!
//! Provides admin commands, single-shot query execution, and an interactive REPL.

#![deny(unsafe_code)]

mod commands;
mod output;
mod repl;

use clap::{CommandFactory, Parser, Subcommand, ValueEnum};
use clap_complete::Shell;
use std::path::PathBuf;

/// CLI exit codes for scripting and CI integration.
///
/// | Code | Meaning |
/// |------|---------|
/// | 0 | Success |
/// | 1 | General error (runtime, I/O, query) |
/// | 2 | Validation failed (`grafeo validate`) |
#[repr(u8)]
enum ExitCode {
    /// Command completed normally.
    Success = 0,
    /// Runtime error, I/O failure, or query error.
    GeneralError = 1,
    /// `grafeo validate` found integrity errors.
    ValidationFailed = 2,
}

/// Grafeo graph database CLI.
///
/// Inspect, query, and maintain Grafeo graph databases from the command line.
#[derive(Parser)]
#[command(name = "grafeo")]
#[command(author, version, about, long_about = None)]
#[allow(clippy::struct_excessive_bools)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Output format (auto-detects TTY: table for interactive, JSON for piped)
    #[arg(long, global = true, default_value = "auto")]
    format: OutputFormat,

    /// Suppress progress and info messages
    #[arg(long, short, global = true)]
    quiet: bool,

    /// Enable verbose debug logging
    #[arg(long, short, global = true)]
    verbose: bool,

    /// Disable colored output (also respects NO_COLOR env var)
    #[arg(long, global = true)]
    no_color: bool,

    /// Force colored output even when piped
    #[arg(long, global = true, conflicts_with = "no_color")]
    color: bool,
}

/// Output format options.
#[derive(Clone, Copy, ValueEnum, Default)]
enum OutputFormat {
    /// Auto-detect: table for TTY, JSON when piped
    #[default]
    Auto,
    /// Human-readable table format
    Table,
    /// Machine-readable JSON format
    Json,
    /// CSV format
    Csv,
}

/// Available commands.
#[derive(Subcommand)]
enum Commands {
    /// Display database information (counts, size, mode)
    Info {
        /// Path to the database
        path: PathBuf,
    },

    /// Show detailed statistics
    Stats {
        /// Path to the database
        path: PathBuf,
    },

    /// Display schema information (labels, edge types, property keys)
    Schema {
        /// Path to the database
        path: PathBuf,
    },

    /// Validate database integrity
    Validate {
        /// Path to the database
        path: PathBuf,
    },

    /// Manage indexes
    #[command(subcommand)]
    Index(IndexCommands),

    /// Manage backups
    #[command(subcommand)]
    Backup(BackupCommands),

    /// Export/import data
    #[command(subcommand)]
    Data(DataCommands),

    /// Manage Write-Ahead Log
    #[command(subcommand)]
    Wal(WalCommands),

    /// Compact the database
    Compact {
        /// Path to the database
        path: PathBuf,

        /// Perform a dry-run (show what would be done)
        #[arg(long)]
        dry_run: bool,
    },

    /// Execute a query against a database
    Query {
        /// Path to the database
        path: PathBuf,

        /// GQL query string (or use --file / --stdin)
        query: Option<String>,

        /// Read query from a file
        #[arg(short, long)]
        file: Option<PathBuf>,

        /// Read query from stdin
        #[arg(long)]
        stdin: bool,

        /// Query parameter (key=value), can be repeated
        #[arg(short, long)]
        param: Vec<String>,

        /// Query language
        #[arg(short, long, default_value = "gql")]
        lang: QueryLanguage,

        /// Show query execution time
        #[arg(long)]
        timing: bool,

        /// Truncate cell values wider than N characters
        #[arg(long)]
        max_width: Option<usize>,
    },

    /// Create a new database
    Init {
        /// Path for the new database
        path: PathBuf,

        /// Graph model
        #[arg(long, default_value = "lpg")]
        mode: GraphMode,
    },

    /// Interactive query shell (REPL)
    Shell {
        /// Path to the database
        path: PathBuf,

        /// Query language
        #[arg(short, long, default_value = "gql")]
        lang: QueryLanguage,

        /// Show query execution time (toggle in shell with :timing)
        #[arg(long)]
        timing: bool,
    },

    /// Show version and build info
    Version,

    /// Generate shell completions
    Completions {
        /// Shell to generate completions for
        shell: Shell,
    },
}

/// Index management commands.
#[derive(Subcommand)]
enum IndexCommands {
    /// List all indexes
    List {
        /// Path to the database
        path: PathBuf,
    },

    /// Show index statistics
    Stats {
        /// Path to the database
        path: PathBuf,
    },
}

/// Backup commands.
#[derive(Subcommand)]
enum BackupCommands {
    /// Create a native backup
    Create {
        /// Path to the database
        path: PathBuf,

        /// Output file path
        #[arg(short, long)]
        output: PathBuf,
    },

    /// Restore from a native backup
    Restore {
        /// Path to the backup file
        backup: PathBuf,

        /// Target database path
        path: PathBuf,

        /// Overwrite if exists
        #[arg(long)]
        force: bool,
    },
}

/// Data export/import commands.
#[derive(Subcommand)]
enum DataCommands {
    /// Export data to a portable format
    Dump {
        /// Path to the database
        path: PathBuf,

        /// Output file or directory
        #[arg(short, long)]
        output: PathBuf,

        /// Export format (parquet, turtle, json)
        #[arg(long = "export-format")]
        export_format: Option<String>,
    },

    /// Import data from a dump
    Load {
        /// Path to the dump file/directory
        input: PathBuf,

        /// Target database path
        path: PathBuf,
    },
}

/// WAL management commands.
#[derive(Subcommand)]
enum WalCommands {
    /// Show WAL status
    Status {
        /// Path to the database
        path: PathBuf,
    },

    /// Force a WAL checkpoint
    Checkpoint {
        /// Path to the database
        path: PathBuf,
    },
}

/// Query language selection.
#[derive(Clone, Copy, ValueEnum)]
enum QueryLanguage {
    /// GQL (ISO/IEC 39075:2024)
    Gql,
    /// Cypher (openCypher 9.0)
    Cypher,
    /// SPARQL (W3C 1.1)
    Sparql,
    /// SQL/PGQ (SQL:2023)
    Sql,
}

/// Graph model selection.
#[derive(Clone, Copy, ValueEnum)]
enum GraphMode {
    /// Labeled Property Graph
    Lpg,
    /// RDF Triple Store
    Rdf,
}

fn main() {
    let cli = Cli::parse();

    // Handle color flags.
    // SAFETY: called once at startup before any threads are spawned.
    #[allow(unsafe_code)]
    if cli.no_color {
        unsafe { std::env::set_var("NO_COLOR", "1") };
    } else if cli.color {
        unsafe { std::env::remove_var("NO_COLOR") };
    }

    // Set up logging based on verbosity
    if cli.verbose {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .init();
    } else if !cli.quiet {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::WARN)
            .init();
    }

    let result = match cli.command {
        Commands::Info { path } => commands::info::run(&path, cli.format, cli.quiet),
        Commands::Stats { path } => commands::stats::run(&path, cli.format, cli.quiet),
        Commands::Schema { path } => commands::schema::run(&path, cli.format, cli.quiet),
        Commands::Validate { path } => commands::validate::run(&path, cli.format, cli.quiet),
        Commands::Index(cmd) => commands::index::run(cmd, cli.format, cli.quiet),
        Commands::Backup(cmd) => commands::backup::run(cmd, cli.format, cli.quiet),
        Commands::Data(cmd) => commands::data::run(cmd, cli.format, cli.quiet),
        Commands::Wal(cmd) => commands::wal::run(cmd, cli.format, cli.quiet),
        Commands::Compact { path, dry_run } => {
            commands::compact::run(&path, dry_run, cli.format, cli.quiet)
        }
        Commands::Query {
            path,
            query,
            file,
            stdin,
            param,
            lang,
            timing,
            max_width,
        } => commands::query::run(commands::query::QueryRunOptions {
            path: &path,
            query,
            file,
            stdin,
            params: &param,
            lang,
            format: cli.format,
            quiet: cli.quiet,
            timing,
            max_width,
        }),
        Commands::Init { path, mode } => commands::init::run(&path, mode, cli.format, cli.quiet),
        Commands::Shell { path, lang, timing } => {
            repl::run(&path, lang, cli.format, cli.quiet, timing)
        }
        Commands::Version => {
            commands::version::run(cli.quiet);
            Ok(())
        }
        Commands::Completions { shell } => {
            clap_complete::generate(shell, &mut Cli::command(), "grafeo", &mut std::io::stdout());
            Ok(())
        }
    };

    let code = match result {
        Ok(()) => ExitCode::Success,
        Err(e) => {
            if e.downcast_ref::<commands::validate::ValidationFailed>()
                .is_some()
            {
                // Validation already printed results, just set the exit code.
                ExitCode::ValidationFailed
            } else {
                output::error(&e.to_string());
                ExitCode::GeneralError
            }
        }
    };

    std::process::exit(code as i32);
}
