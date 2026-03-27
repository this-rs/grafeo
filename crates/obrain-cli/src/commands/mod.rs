//! CLI command implementations.

use std::path::Path;

use anyhow::{Context, Result};
use grafeo_engine::GrafeoDB;

pub mod backup;
pub mod compact;
pub mod data;
pub mod index;
pub mod info;
pub mod init;
pub mod query;
pub mod schema;
pub mod stats;
pub mod validate;
pub mod version;
pub mod wal;

/// Open an existing database, returning a clear error if the path does not exist.
///
/// Use this instead of `GrafeoDB::open` for commands that expect a pre-existing database.
/// The `init` command should use `GrafeoDB::open` directly since it intentionally creates.
pub fn open_existing(path: &Path) -> Result<GrafeoDB> {
    if !path.exists() {
        anyhow::bail!(
            "Database not found: {}\n\
             Use `grafeo init {}` to create a new database.",
            path.display(),
            path.display()
        );
    }
    GrafeoDB::open(path).with_context(|| format!("Failed to open database at {}", path.display()))
}
