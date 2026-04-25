//! # Resume-safe checkpoint file
//!
//! A tiny JSON file written to `<out>/.migrate-checkpoint` that tracks
//! which phases have completed. On `--resume`, the migrator reads it and
//! short-circuits any phase already marked done.
//!
//! The file is intentionally small and human-readable: this is a
//! one-shot tool, not a hot-path artifact.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Name of the checkpoint file inside the output directory.
pub const CHECKPOINT_FILE: &str = ".migrate-checkpoint.json";

/// Phases in execution order. `PartialOrd` reflects the dependency
/// chain — a phase N assumes every phase M < N has completed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    /// Output store created, dicts materialized, nothing else done.
    StoreInit,
    /// All legacy nodes have been written to substrate; `node_map` is
    /// complete.
    Nodes,
    /// All legacy edges have been written.
    Edges,
    /// Cognitive columns (energy / scar / utility / affinity / synapse
    /// weight) have been transferred.
    Cognitive,
    /// L0 / L1 / L2 retrieval tiers built from surviving `_st_embedding`
    /// properties (only if `--with-tiers`).
    Tiers,
    /// Batch LDleiden + PageRank + Ricci ran (only if
    /// `--with-cognitive-init`).
    CognitiveInit,
    /// Migration finalised, output flushed, checkpoint can be deleted.
    Done,
}

/// On-disk payload. Extend with additional fields as phases grow
/// richer — never remove, rename, or re-order existing ones without a
/// version bump.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Format version — bump on breaking schema changes.
    pub version: u32,
    /// Last phase completed (inclusive).
    pub last_completed: Phase,
    /// Absolute input path at the time the checkpoint was written.
    /// Resuming against a different input is a hard error.
    pub input_canonical: PathBuf,
    /// Running count of nodes written (for diagnostics / progress bars).
    pub nodes_written: u64,
    /// Running count of edges written.
    pub edges_written: u64,
}

impl Checkpoint {
    pub const CURRENT_VERSION: u32 = 1;

    pub fn new(input_canonical: PathBuf) -> Self {
        Self {
            version: Self::CURRENT_VERSION,
            last_completed: Phase::StoreInit,
            input_canonical,
            nodes_written: 0,
            edges_written: 0,
        }
    }

    /// Path to the checkpoint file inside the given output dir.
    pub fn path(out: &Path) -> PathBuf {
        out.join(CHECKPOINT_FILE)
    }

    pub fn load(out: &Path) -> Result<Option<Self>> {
        let p = Self::path(out);
        if !p.exists() {
            return Ok(None);
        }
        let bytes = fs::read(&p).with_context(|| format!("read checkpoint {}", p.display()))?;
        let cp: Self = serde_json::from_slice(&bytes)
            .with_context(|| format!("parse checkpoint {}", p.display()))?;
        if cp.version != Self::CURRENT_VERSION {
            anyhow::bail!(
                "checkpoint version mismatch: found {}, expected {} — delete {} to restart",
                cp.version,
                Self::CURRENT_VERSION,
                p.display()
            );
        }
        Ok(Some(cp))
    }

    pub fn save(&self, out: &Path) -> Result<()> {
        let p = Self::path(out);
        let bytes = serde_json::to_vec_pretty(self).context("serialize checkpoint")?;
        // Atomic write: tmp file + rename.
        let tmp = p.with_extension("json.tmp");
        fs::write(&tmp, bytes).with_context(|| format!("write {}", tmp.display()))?;
        fs::rename(&tmp, &p)
            .with_context(|| format!("rename {} → {}", tmp.display(), p.display()))?;
        Ok(())
    }

    /// Deletes the checkpoint file — called after `Phase::Done` is
    /// committed successfully.
    pub fn finalize(out: &Path) -> Result<()> {
        let p = Self::path(out);
        if p.exists() {
            fs::remove_file(&p).with_context(|| format!("remove {}", p.display()))?;
        }
        Ok(())
    }

    /// Mark a phase complete and persist the checkpoint atomically.
    pub fn mark(&mut self, phase: Phase, out: &Path) -> Result<()> {
        if phase > self.last_completed {
            self.last_completed = phase;
        }
        self.save(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_ordering_is_monotone() {
        assert!(Phase::StoreInit < Phase::Nodes);
        assert!(Phase::Nodes < Phase::Edges);
        assert!(Phase::Edges < Phase::Cognitive);
        assert!(Phase::Cognitive < Phase::Tiers);
        assert!(Phase::Tiers < Phase::CognitiveInit);
        assert!(Phase::CognitiveInit < Phase::Done);
    }

    #[test]
    fn roundtrip_checkpoint() {
        let td = tempfile::TempDir::new().unwrap();
        let mut cp = Checkpoint::new(PathBuf::from("/in"));
        cp.nodes_written = 42;
        cp.mark(Phase::Nodes, td.path()).unwrap();

        let loaded = Checkpoint::load(td.path()).unwrap().unwrap();
        assert_eq!(loaded.last_completed, Phase::Nodes);
        assert_eq!(loaded.nodes_written, 42);
    }

    #[test]
    fn finalize_removes_file() {
        let td = tempfile::TempDir::new().unwrap();
        let cp = Checkpoint::new(PathBuf::from("/in"));
        cp.save(td.path()).unwrap();
        assert!(Checkpoint::path(td.path()).exists());
        Checkpoint::finalize(td.path()).unwrap();
        assert!(!Checkpoint::path(td.path()).exists());
    }
}
