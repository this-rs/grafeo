//! Backup management commands.
//!
//! T17 final cutover (2026-04-23): `obrain backup create` and
//! `obrain backup restore` used `ObrainDB::save()`, which serialized
//! the in-memory graph as a bincode snapshot into a single `.obrain`
//! file (v1/v2 legacy format). That format and its support code were
//! removed as part of the substrate cutover — substrate persists
//! directly to its directory layout, so a "backup" is just a
//! filesystem copy of the directory.
//!
//! These commands now return a deprecation error and point the caller
//! at `cp -r`, `tar`, `rsync`, or any filesystem-level copy tool. A
//! future release may re-introduce backup tooling as a substrate-
//! native incremental snapshot pipeline (WAL + tier cold copy).

use anyhow::Result;

use crate::{BackupCommands, OutputFormat};

/// Run backup commands.
pub fn run(cmd: BackupCommands, _format: OutputFormat, _quiet: bool) -> Result<()> {
    let (subcmd, detail) = match &cmd {
        BackupCommands::Create { path, output } => (
            "create",
            format!("copy `{}` to `{}`", path.display(), output.display()),
        ),
        BackupCommands::Restore { backup, path, .. } => (
            "restore",
            format!(
                "copy `{}` to `{}`",
                backup.display(),
                path.display()
            ),
        ),
    };

    anyhow::bail!(
        "`obrain backup {subcmd}` is no longer supported since the T17 substrate \
         cutover. Substrate persists directly to its directory layout, so a \
         backup is just a filesystem copy — use `cp -r`, `tar`, or `rsync` \
         to {detail}."
    )
}
