//! WAL recovery.

use super::record::WalEntry;
use super::{CheckpointMetadata, WalManager, WalRecord};
use grafeo_common::utils::error::{Error, Result, StorageError};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

/// Name of the checkpoint metadata file.
const CHECKPOINT_METADATA_FILE: &str = "checkpoint.meta";

/// Handles WAL recovery after a crash.
pub struct WalRecovery {
    /// Directory containing WAL files.
    dir: std::path::PathBuf,
}

impl WalRecovery {
    /// Creates a new recovery handler for the given WAL directory.
    pub fn new(dir: impl AsRef<Path>) -> Self {
        Self {
            dir: dir.as_ref().to_path_buf(),
        }
    }

    /// Creates a recovery handler from a WAL manager.
    #[must_use]
    pub fn from_wal(wal: &WalManager) -> Self {
        Self {
            dir: wal.dir().to_path_buf(),
        }
    }

    /// Reads checkpoint metadata if it exists.
    ///
    /// Returns `None` if no checkpoint metadata is found.
    pub fn read_checkpoint_metadata(&self) -> Result<Option<CheckpointMetadata>> {
        let metadata_path = self.dir.join(CHECKPOINT_METADATA_FILE);

        if !metadata_path.exists() {
            return Ok(None);
        }

        let file = File::open(&metadata_path)?;
        let mut reader = BufReader::new(file);
        let mut data = Vec::new();
        reader.read_to_end(&mut data)?;

        let (metadata, _): (CheckpointMetadata, _) =
            bincode::serde::decode_from_slice(&data, bincode::config::standard())
                .map_err(|e| Error::Serialization(e.to_string()))?;

        Ok(Some(metadata))
    }

    /// Returns the checkpoint metadata, if any.
    ///
    /// This is useful for determining whether to perform a full or
    /// incremental recovery.
    #[must_use]
    pub fn checkpoint(&self) -> Option<CheckpointMetadata> {
        self.read_checkpoint_metadata().ok().flatten()
    }

    /// Recovers committed records from all WAL files.
    ///
    /// Returns only records that were part of committed transactions.
    /// If checkpoint metadata exists, only replays files from the
    /// checkpoint sequence onwards.
    ///
    /// # Errors
    ///
    /// Returns an error if recovery fails.
    pub fn recover(&self) -> Result<Vec<WalRecord>> {
        self.recover_as::<WalRecord>()
    }

    /// Recovers committed records of a specific type from all WAL files.
    ///
    /// This is the generic version of [`recover`](Self::recover). Use it
    /// when recovering a WAL that stores a custom record type.
    ///
    /// # Errors
    ///
    /// Returns an error if recovery fails.
    pub fn recover_as<R: WalEntry>(&self) -> Result<Vec<R>> {
        let checkpoint = self.read_checkpoint_metadata()?;
        self.recover_internal_as::<R>(checkpoint)
    }

    /// Recovers committed records, starting from a specific checkpoint.
    ///
    /// This can be used for incremental recovery when you want to
    /// skip WAL files that precede the checkpoint.
    ///
    /// # Errors
    ///
    /// Returns an error if recovery fails.
    pub fn recover_from_checkpoint(
        &self,
        checkpoint: Option<&CheckpointMetadata>,
    ) -> Result<Vec<WalRecord>> {
        self.recover_internal_as::<WalRecord>(checkpoint.cloned())
    }

    fn recover_internal_as<R: WalEntry>(
        &self,
        checkpoint: Option<CheckpointMetadata>,
    ) -> Result<Vec<R>> {
        let mut current_tx_records = Vec::new();
        let mut committed_records = Vec::new();

        // Get all log files in order
        let log_files = self.get_log_files()?;

        // Determine the minimum sequence number to process
        let min_sequence = checkpoint.as_ref().map_or(0, |cp| cp.log_sequence);

        if checkpoint.is_some() {
            tracing::info!(
                "Recovering from checkpoint at epoch {:?}, starting from log sequence {}",
                checkpoint.as_ref().map(|c| c.epoch),
                min_sequence
            );
        }

        // Read log files in sequence, skipping those before checkpoint
        for log_file in log_files {
            // Extract sequence number from filename
            let sequence = Self::sequence_from_path(&log_file).unwrap_or(0);

            // Skip files that are completely before the checkpoint
            // We include the checkpoint sequence file because it may contain
            // records after the checkpoint record itself
            if sequence < min_sequence {
                tracing::debug!(
                    "Skipping log file {:?} (sequence {} < checkpoint {})",
                    log_file,
                    sequence,
                    min_sequence
                );
                continue;
            }

            let file = match File::open(&log_file) {
                Ok(f) => f,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
                Err(e) => return Err(e.into()),
            };
            let mut reader = BufReader::new(file);

            // Read all records from this file
            loop {
                match self.read_record_as::<R>(&mut reader) {
                    Ok(Some(record)) => {
                        if record.is_commit() {
                            committed_records.append(&mut current_tx_records);
                            committed_records.push(record);
                        } else if record.is_abort() {
                            current_tx_records.clear();
                        } else if record.is_checkpoint() {
                            current_tx_records.clear();
                            committed_records.push(record);
                        } else {
                            current_tx_records.push(record);
                        }
                    }
                    Ok(None) => break, // EOF
                    Err(e) => {
                        // Log corruption - stop reading this file but continue
                        // with remaining files (best-effort recovery)
                        tracing::warn!("WAL corruption detected in {:?}: {}", log_file, e);
                        break;
                    }
                }
            }
        }

        // Uncommitted records in current_tx_records are discarded

        Ok(committed_records)
    }

    /// Extracts the sequence number from a WAL log file path.
    fn sequence_from_path(path: &Path) -> Option<u64> {
        path.file_stem()
            .and_then(|s| s.to_str())
            .and_then(|s| s.strip_prefix("wal_"))
            .and_then(|s| s.parse().ok())
    }

    /// Recovers committed records from a single WAL file.
    ///
    /// # Errors
    ///
    /// Returns an error if recovery fails.
    pub fn recover_file(&self, path: impl AsRef<Path>) -> Result<Vec<WalRecord>> {
        self.recover_file_as::<WalRecord>(path)
    }

    /// Recovers committed records of a specific type from a single WAL file.
    ///
    /// # Errors
    ///
    /// Returns an error if recovery fails.
    pub fn recover_file_as<R: WalEntry>(&self, path: impl AsRef<Path>) -> Result<Vec<R>> {
        let file = File::open(path.as_ref())?;
        let mut reader = BufReader::new(file);

        let mut current_tx_records = Vec::new();
        let mut committed_records = Vec::new();

        loop {
            match self.read_record_as::<R>(&mut reader) {
                Ok(Some(record)) => {
                    if record.is_commit() {
                        committed_records.append(&mut current_tx_records);
                        committed_records.push(record);
                    } else if record.is_abort() {
                        current_tx_records.clear();
                    } else {
                        current_tx_records.push(record);
                    }
                }
                Ok(None) => break,
                Err(e) => {
                    tracing::warn!("WAL corruption detected: {}", e);
                    break;
                }
            }
        }

        Ok(committed_records)
    }

    fn get_log_files(&self) -> Result<Vec<std::path::PathBuf>> {
        let mut files = Vec::new();

        if !self.dir.exists() {
            return Ok(files);
        }

        if let Ok(entries) = std::fs::read_dir(&self.dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().is_some_and(|ext| ext == "log") {
                    files.push(path);
                }
            }
        }

        // Sort by filename (which includes sequence number)
        files.sort();

        Ok(files)
    }

    fn read_record_as<R: WalEntry>(&self, reader: &mut BufReader<File>) -> Result<Option<R>> {
        // Read length prefix
        let mut len_buf = [0u8; 4];
        match reader.read_exact(&mut len_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(e.into()),
        }
        let len = u32::from_le_bytes(len_buf) as usize;

        // Read data
        let mut data = vec![0u8; len];
        reader.read_exact(&mut data)?;

        // Read and verify checksum
        let mut checksum_buf = [0u8; 4];
        reader.read_exact(&mut checksum_buf)?;
        let stored_checksum = u32::from_le_bytes(checksum_buf);
        let computed_checksum = crc32fast::hash(&data);

        if stored_checksum != computed_checksum {
            return Err(Error::Storage(StorageError::Corruption(
                "WAL checksum mismatch".to_string(),
            )));
        }

        // Deserialize
        let (record, _): (R, _) =
            bincode::serde::decode_from_slice(&data, bincode::config::standard())
                .map_err(|e| Error::Serialization(e.to_string()))?;

        Ok(Some(record))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grafeo_common::types::{NodeId, TransactionId};
    use tempfile::tempdir;

    #[test]
    fn test_recovery_committed() {
        let dir = tempdir().unwrap();

        // Write some records
        {
            let wal = WalManager::open(dir.path()).unwrap();

            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(1),
                labels: vec!["Person".to_string()],
            })
            .unwrap();

            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(1),
            })
            .unwrap();

            wal.sync().unwrap();
        }

        // Recover
        let recovery = WalRecovery::new(dir.path());
        let records = recovery.recover().unwrap();

        assert_eq!(records.len(), 2);
    }

    #[test]
    fn test_recovery_uncommitted() {
        let dir = tempdir().unwrap();

        // Write some records without commit
        {
            let wal = WalManager::open(dir.path()).unwrap();

            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(1),
                labels: vec!["Person".to_string()],
            })
            .unwrap();

            // No commit!
            wal.sync().unwrap();
        }

        // Recover
        let recovery = WalRecovery::new(dir.path());
        let records = recovery.recover().unwrap();

        // Uncommitted records should be discarded
        assert_eq!(records.len(), 0);
    }

    #[test]
    fn test_recovery_multiple_files() {
        let dir = tempdir().unwrap();

        // Write records across multiple files
        {
            let config = super::super::WalConfig {
                max_log_size: 100, // Force rotation
                ..Default::default()
            };
            let wal = WalManager::with_config(dir.path(), config).unwrap();

            // First transaction
            for i in 0..5 {
                wal.log(&WalRecord::CreateNode {
                    id: NodeId::new(i),
                    labels: vec!["Test".to_string()],
                })
                .unwrap();
            }
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(1),
            })
            .unwrap();

            // Second transaction
            for i in 5..10 {
                wal.log(&WalRecord::CreateNode {
                    id: NodeId::new(i),
                    labels: vec!["Test".to_string()],
                })
                .unwrap();
            }
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(2),
            })
            .unwrap();

            wal.sync().unwrap();
        }

        // Recover
        let recovery = WalRecovery::new(dir.path());
        let records = recovery.recover().unwrap();

        // Should have 10 CreateNode + 2 TransactionCommit
        assert_eq!(records.len(), 12);
    }

    #[test]
    fn test_checkpoint_metadata() {
        use grafeo_common::types::EpochId;

        let dir = tempdir().unwrap();

        // Write records and create a checkpoint
        {
            let wal = WalManager::open(dir.path()).unwrap();

            // First transaction
            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(1),
                labels: vec!["Test".to_string()],
            })
            .unwrap();
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(1),
            })
            .unwrap();

            // Create checkpoint
            wal.checkpoint(TransactionId::new(1), EpochId::new(10))
                .unwrap();

            // Second transaction after checkpoint
            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(2),
                labels: vec!["Test".to_string()],
            })
            .unwrap();
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(2),
            })
            .unwrap();

            wal.sync().unwrap();
        }

        // Verify checkpoint metadata was written
        let recovery = WalRecovery::new(dir.path());
        let checkpoint = recovery.checkpoint();
        assert!(checkpoint.is_some(), "Checkpoint metadata should exist");

        let cp = checkpoint.unwrap();
        assert_eq!(cp.epoch.as_u64(), 10);
        assert_eq!(cp.transaction_id.as_u64(), 1);
    }

    #[test]
    fn test_recovery_from_checkpoint() {
        use super::super::WalConfig;
        use grafeo_common::types::EpochId;

        let dir = tempdir().unwrap();

        // Write records across multiple log files with checkpoint
        {
            let config = WalConfig {
                max_log_size: 100, // Force rotation
                ..Default::default()
            };
            let wal = WalManager::with_config(dir.path(), config).unwrap();

            // First batch of records (should end up in early log files)
            for i in 0..5 {
                wal.log(&WalRecord::CreateNode {
                    id: NodeId::new(i),
                    labels: vec!["Before".to_string()],
                })
                .unwrap();
            }
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(1),
            })
            .unwrap();

            // Create checkpoint
            wal.checkpoint(TransactionId::new(1), EpochId::new(100))
                .unwrap();

            // Second batch after checkpoint
            for i in 100..103 {
                wal.log(&WalRecord::CreateNode {
                    id: NodeId::new(i),
                    labels: vec!["After".to_string()],
                })
                .unwrap();
            }
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(2),
            })
            .unwrap();

            wal.sync().unwrap();
        }

        // Recovery should use checkpoint metadata to skip old files
        let recovery = WalRecovery::new(dir.path());
        let records = recovery.recover().unwrap();

        // We should get all committed records (checkpoint metadata is used for optimization)
        // The number depends on how many log files were skipped
        assert!(!records.is_empty(), "Should recover some records");
    }

    #[test]
    fn test_recover_as_generic() {
        let dir = tempdir().unwrap();

        // Write records using WalManager
        {
            let wal = WalManager::open(dir.path()).unwrap();

            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(1),
                labels: vec!["Person".to_string()],
            })
            .unwrap();

            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(1),
            })
            .unwrap();

            wal.sync().unwrap();
        }

        // Recover using the generic method
        let recovery = WalRecovery::new(dir.path());
        let records: Vec<WalRecord> = recovery.recover_as().unwrap();

        assert_eq!(records.len(), 2);

        // Verify the records are correct via WalEntry trait methods
        assert!(!records[0].is_commit());
        assert!(records[1].is_commit());
    }

    #[test]
    fn test_recovery_truncated_wal_mid_record() {
        let dir = tempdir().unwrap();

        // Write valid records first
        {
            let wal = WalManager::open(dir.path()).unwrap();
            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(1),
                labels: vec!["Person".to_string()],
            })
            .unwrap();
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(1),
            })
            .unwrap();
            wal.sync().unwrap();
        }

        // Find the WAL file and append a truncated record (length prefix only, no data)
        let wal_files: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| {
                let e = e.ok()?;
                if e.path().extension().is_some_and(|ext| ext == "log") {
                    Some(e.path())
                } else {
                    None
                }
            })
            .collect();
        assert!(!wal_files.is_empty());

        // Append a partial record: just a length prefix, then truncate
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&wal_files[0])
            .unwrap();
        f.write_all(&100u32.to_le_bytes()).unwrap(); // length=100 but no data follows

        // Recovery should still return the committed records (best-effort)
        let recovery = WalRecovery::new(dir.path());
        let records = recovery.recover().unwrap();
        assert_eq!(
            records.len(),
            2,
            "committed records before truncation should be recovered"
        );
    }

    #[test]
    fn test_recovery_corrupted_checksum() {
        let dir = tempdir().unwrap();

        // Write valid records
        {
            let wal = WalManager::open(dir.path()).unwrap();
            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(1),
                labels: vec!["First".to_string()],
            })
            .unwrap();
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(1),
            })
            .unwrap();
            wal.sync().unwrap();
        }

        // Find the WAL file and corrupt a byte in the data section
        let wal_files: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| {
                let e = e.ok()?;
                if e.path().extension().is_some_and(|ext| ext == "log") {
                    Some(e.path())
                } else {
                    None
                }
            })
            .collect();
        assert!(!wal_files.is_empty());

        let mut data = std::fs::read(&wal_files[0]).unwrap();
        // Flip a byte in the middle of the data (after the 4-byte length prefix)
        if data.len() > 8 {
            data[6] ^= 0xFF;
        }
        std::fs::write(&wal_files[0], &data).unwrap();

        // Recovery should handle corruption gracefully (not panic)
        let recovery = WalRecovery::new(dir.path());
        let result = recovery.recover();
        // Should either succeed with fewer records or return an error
        match result {
            Ok(records) => {
                // Best-effort: may have recovered 0 records due to corruption
                assert!(records.len() <= 2);
            }
            Err(_) => {
                // Also acceptable: corruption detection as error
            }
        }
    }

    #[test]
    fn test_recovery_empty_wal_file() {
        let dir = tempdir().unwrap();

        // Create an empty WAL file
        std::fs::write(dir.path().join("wal_00000000.log"), []).unwrap();

        let recovery = WalRecovery::new(dir.path());
        let records = recovery.recover().unwrap();
        assert_eq!(records.len(), 0, "empty WAL should produce no records");
    }
}

/// Crash injection tests for WAL recovery.
///
/// These tests verify that WAL recovery produces a consistent state after
/// simulated crashes at every crash point in the write path. The three crash
/// points are:
/// - `wal_before_write`: before writing length prefix + data + checksum
/// - `wal_after_write`: after writing data but before durability handling
/// - `wal_before_flush`: before fsync on TransactionCommit in Sync mode
///
/// Run with:
/// ```bash
/// cargo test -p grafeo-adapters --features "wal,testing-crash-injection" -- crash
/// ```
#[cfg(all(test, feature = "testing-crash-injection"))]
mod crash_tests {
    use super::*;
    use grafeo_common::types::{EpochId, NodeId, TransactionId, Value};
    use grafeo_core::testing::crash::{CrashResult, with_crash_at};
    use tempfile::tempdir;

    /// Helper: Sync durability config so all three crash points are reachable.
    fn sync_config() -> super::super::WalConfig {
        super::super::WalConfig {
            durability: super::super::DurabilityMode::Sync,
            ..Default::default()
        }
    }

    /// Crash at `wal_before_write`: no record bytes reach disk.
    /// Recovery should only return previously committed data.
    #[test]
    fn test_crash_before_write_discards_record() {
        let dir = tempdir().unwrap();
        let path = dir.path().to_path_buf();

        // Seed one committed transaction
        {
            let wal = WalManager::with_config(&path, sync_config()).unwrap();
            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(1),
                labels: vec!["Committed".into()],
            })
            .unwrap();
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(1),
            })
            .unwrap();
        }

        // Crash at the first crash point (wal_before_write)
        let p = path.clone();
        let result = with_crash_at(1, move || {
            let wal = WalManager::with_config(&p, sync_config()).unwrap();
            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(2),
                labels: vec!["Lost".into()],
            })
            .unwrap();
        });
        assert!(matches!(result, CrashResult::Crashed));

        // Only the first committed tx should survive
        let recovery = WalRecovery::new(&path);
        let records = recovery.recover().unwrap();
        assert_eq!(records.len(), 2, "CreateNode(1) + TransactionCommit(1)");
    }

    /// Crash at `wal_after_write`: data may be in BufWriter but no commit
    /// marker. Recovery should discard the uncommitted record.
    #[test]
    fn test_crash_after_write_uncommitted_discarded() {
        let dir = tempdir().unwrap();
        let path = dir.path().to_path_buf();

        // For a non-commit record the crash points are:
        //   1 = wal_before_write, 2 = wal_after_write
        let p = path.clone();
        let result = with_crash_at(2, move || {
            let wal = WalManager::with_config(&p, sync_config()).unwrap();
            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(1),
                labels: vec!["Partial".into()],
            })
            .unwrap();
        });
        assert!(matches!(result, CrashResult::Crashed));

        // No committed tx ⇒ recovery returns nothing
        let recovery = WalRecovery::new(&path);
        let records = recovery.recover().unwrap();
        assert_eq!(records.len(), 0, "Uncommitted records must be discarded");
    }

    /// Two committed transactions, then crash during the third.
    /// Recovery should preserve exactly the first two.
    #[test]
    fn test_crash_preserves_prior_committed_transactions() {
        let dir = tempdir().unwrap();
        let path = dir.path().to_path_buf();

        // Commit two transactions
        {
            let wal = WalManager::with_config(&path, sync_config()).unwrap();
            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(1),
                labels: vec!["T1".into()],
            })
            .unwrap();
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(1),
            })
            .unwrap();
            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(2),
                labels: vec!["T2".into()],
            })
            .unwrap();
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(2),
            })
            .unwrap();
        }

        // Third transaction crashes immediately
        let p = path.clone();
        let result = with_crash_at(1, move || {
            let wal = WalManager::with_config(&p, sync_config()).unwrap();
            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(3),
                labels: vec!["T3".into()],
            })
            .unwrap();
        });
        assert!(matches!(result, CrashResult::Crashed));

        // Both committed txs intact, third discarded
        let recovery = WalRecovery::new(&path);
        let records = recovery.recover().unwrap();
        assert_eq!(records.len(), 4, "2 CreateNode + 2 TransactionCommit");
    }

    /// Crash during checkpoint: committed data must still be recoverable.
    #[test]
    fn test_crash_during_checkpoint_preserves_data() {
        for crash_at in 1..15 {
            let dir = tempdir().unwrap();
            let path = dir.path().to_path_buf();

            // Seed committed data
            {
                let wal = WalManager::with_config(&path, sync_config()).unwrap();
                wal.log(&WalRecord::CreateNode {
                    id: NodeId::new(1),
                    labels: vec!["A".into()],
                })
                .unwrap();
                wal.log(&WalRecord::TransactionCommit {
                    transaction_id: TransactionId::new(1),
                })
                .unwrap();
            }

            // Crash during checkpoint
            let p = path.clone();
            let _result = with_crash_at(crash_at, move || {
                let wal = WalManager::with_config(&p, sync_config()).unwrap();
                wal.checkpoint(TransactionId::new(1), EpochId::new(10))
                    .unwrap();
            });

            // Committed data must survive regardless of checkpoint outcome
            let recovery = WalRecovery::new(&path);
            let records = recovery.recover().unwrap();
            assert!(
                !records.is_empty(),
                "crash_at={crash_at}: committed data must survive checkpoint crash"
            );
        }
    }

    /// Crash with rotated log files: recovery should span all files.
    #[test]
    fn test_crash_with_log_rotation() {
        let dir = tempdir().unwrap();
        let path = dir.path().to_path_buf();

        // Write enough to trigger rotation
        {
            let config = super::super::WalConfig {
                durability: super::super::DurabilityMode::Sync,
                max_log_size: 100, // force rotation
                ..Default::default()
            };
            let wal = WalManager::with_config(&path, config).unwrap();
            for i in 0..5 {
                wal.log(&WalRecord::CreateNode {
                    id: NodeId::new(i),
                    labels: vec!["Rotated".into()],
                })
                .unwrap();
            }
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(1),
            })
            .unwrap();
        }

        // Crash during additional write
        let p = path.clone();
        let result = with_crash_at(1, move || {
            let config = super::super::WalConfig {
                durability: super::super::DurabilityMode::Sync,
                max_log_size: 100,
                ..Default::default()
            };
            let wal = WalManager::with_config(&p, config).unwrap();
            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(99),
                labels: vec!["Crash".into()],
            })
            .unwrap();
        });
        assert!(matches!(result, CrashResult::Crashed));

        // All committed data across rotated files should survive
        let recovery = WalRecovery::new(&path);
        let records = recovery.recover().unwrap();
        assert_eq!(records.len(), 6, "5 CreateNode + 1 TransactionCommit");
    }

    /// Exhaustive sweep: crash at every possible point during a multi-record
    /// transaction and verify recovery invariants.
    ///
    /// Invariants checked:
    /// 1. Previously committed transactions always survive
    /// 2. Recovery output never contains partial (uncommitted) transactions
    #[test]
    fn test_crash_sweep_all_points() {
        for crash_at in 1..20 {
            let dir = tempdir().unwrap();
            let path = dir.path().to_path_buf();

            // Seed one committed transaction
            {
                let wal = WalManager::with_config(&path, sync_config()).unwrap();
                wal.log(&WalRecord::CreateNode {
                    id: NodeId::new(1),
                    labels: vec!["Base".into()],
                })
                .unwrap();
                wal.log(&WalRecord::TransactionCommit {
                    transaction_id: TransactionId::new(1),
                })
                .unwrap();
            }

            // Attempt a second transaction with crash injection
            let p = path.clone();
            let result = with_crash_at(crash_at, move || {
                let wal = WalManager::with_config(&p, sync_config()).unwrap();
                wal.log(&WalRecord::CreateNode {
                    id: NodeId::new(100),
                    labels: vec!["New".into()],
                })
                .unwrap();
                wal.log(&WalRecord::SetNodeProperty {
                    id: NodeId::new(100),
                    key: "name".into(),
                    value: Value::String("test".into()),
                })
                .unwrap();
                wal.log(&WalRecord::TransactionCommit {
                    transaction_id: TransactionId::new(2),
                })
                .unwrap();
            });

            // Verify recovery invariants
            let recovery = WalRecovery::new(&path);
            let records = recovery.recover().unwrap();

            // Invariant 1: base committed tx always survives
            assert!(
                records.len() >= 2,
                "crash_at={crash_at}: base tx must survive, got {} records",
                records.len()
            );

            // Invariant 2: no partial transactions in output
            let mut pending = 0usize;
            for record in &records {
                match record {
                    WalRecord::TransactionCommit { .. }
                    | WalRecord::TransactionAbort { .. }
                    | WalRecord::Checkpoint { .. } => pending = 0,
                    _ => pending += 1,
                }
            }
            assert_eq!(
                pending, 0,
                "crash_at={crash_at}: recovery must not output partial transactions"
            );

            // If the operation completed, the second tx should also be present
            if matches!(result, CrashResult::Completed(())) {
                assert!(
                    records.len() >= 5,
                    "crash_at={crash_at}: completed run should include second tx"
                );
            }
        }
    }

    /// Aborted transactions are not recovered even without a crash.
    /// Verifies that TransactionAbort correctly discards pending records.
    #[test]
    fn test_abort_then_crash_discards_aborted_tx() {
        let dir = tempdir().unwrap();
        let path = dir.path().to_path_buf();

        {
            let wal = WalManager::with_config(&path, sync_config()).unwrap();
            // Committed tx
            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(1),
                labels: vec!["Keep".into()],
            })
            .unwrap();
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(1),
            })
            .unwrap();
            // Aborted tx
            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(2),
                labels: vec!["Discard".into()],
            })
            .unwrap();
            wal.log(&WalRecord::TransactionAbort {
                transaction_id: TransactionId::new(2),
            })
            .unwrap();
        }

        // Crash during a third transaction
        let p = path.clone();
        let result = with_crash_at(1, move || {
            let wal = WalManager::with_config(&p, sync_config()).unwrap();
            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(3),
                labels: vec!["Also lost".into()],
            })
            .unwrap();
        });
        assert!(matches!(result, CrashResult::Crashed));

        let recovery = WalRecovery::new(&path);
        let records = recovery.recover().unwrap();
        // Only the committed tx (2 records)
        assert_eq!(
            records.len(),
            2,
            "Aborted + crashed records should both be discarded"
        );
    }
}
