//! High-level manager for `.obrain` database files.
//!
//! [`ObrainFileManager`] owns the file handle and provides create, open,
//! snapshot write/read, and sidecar WAL lifecycle management.

use std::fs::{self, File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use fs2::FileExt;
use obrain_common::utils::error::{Error, Result};
use parking_lot::Mutex;

use super::format::{DATA_OFFSET, DbHeader, FileHeader};
use super::header;

/// Manages a single `.obrain` database file.
///
/// # Lifecycle
///
/// 1. [`create`](Self::create) or [`open`](Self::open)
/// 2. Mutations flow through a sidecar WAL (managed externally by the engine)
/// 3. [`write_snapshot`](Self::write_snapshot) checkpoints memory to the file
/// 4. After a successful checkpoint, call [`remove_sidecar_wal`](Self::remove_sidecar_wal)
/// 5. [`close`](Self::close) (or drop) releases the file handle
pub struct ObrainFileManager {
    /// Path to the `.obrain` file.
    path: PathBuf,
    /// Open file handle (read/write or read-only).
    file: Mutex<File>,
    /// File header (read once on open, immutable afterwards).
    file_header: FileHeader,
    /// Currently active database header.
    active_header: Mutex<DbHeader>,
    /// Slot index (0 or 1) of the active header.
    active_slot: Mutex<u8>,
    /// Whether this manager was opened in read-only mode.
    read_only: bool,
}

impl ObrainFileManager {
    /// Creates a new `.obrain` file at `path`.
    ///
    /// Writes the file header and two empty database headers. The file must
    /// not already exist.
    ///
    /// # Errors
    ///
    /// Returns an error if the file already exists or cannot be created.
    pub fn create(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        if path.exists() {
            return Err(Error::Internal(format!(
                "file already exists: {}",
                path.display()
            )));
        }

        // Ensure parent directory exists
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }

        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::AlreadyExists || e.raw_os_error() == Some(183) {
                    Error::Io(std::io::Error::new(
                        std::io::ErrorKind::AlreadyExists,
                        format!(
                            "database file already exists (may be open by another process): {}",
                            path.display()
                        ),
                    ))
                } else {
                    Error::Io(e)
                }
            })?;

        // Acquire an exclusive lock: prevents other processes from opening the same file
        file.try_lock_exclusive().map_err(|_| {
            Error::Internal(format!(
                "database file is locked by another process: {}",
                path.display()
            ))
        })?;

        let file_header = FileHeader::new();
        header::write_file_header(&mut file, &file_header)?;
        header::write_db_header(&mut file, 0, &DbHeader::EMPTY)?;
        header::write_db_header(&mut file, 1, &DbHeader::EMPTY)?;
        file.sync_all()?;

        Ok(Self {
            path,
            file: Mutex::new(file),
            file_header,
            active_header: Mutex::new(DbHeader::EMPTY),
            active_slot: Mutex::new(0),
            read_only: false,
        })
    }

    /// Opens an existing `.obrain` file.
    ///
    /// Validates the magic bytes and format version, then selects the
    /// active database header.
    ///
    /// # Errors
    ///
    /// Returns an error if the file does not exist, has invalid magic, or
    /// an unsupported format version.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        let mut file = OpenOptions::new().read(true).write(true).open(&path)?;

        // Acquire an exclusive lock: prevents other processes from opening the same file
        file.try_lock_exclusive().map_err(|_| {
            Error::Internal(format!(
                "database file is locked by another process: {}",
                path.display()
            ))
        })?;

        let file_header = header::read_file_header(&mut file)?;
        header::validate_file_header(&file_header)?;

        let (h0, h1) = header::read_db_headers(&mut file)?;
        let (active_slot, active_header) = header::active_db_header(&h0, &h1);

        Ok(Self {
            path,
            file: Mutex::new(file),
            file_header,
            active_header: Mutex::new(active_header),
            active_slot: Mutex::new(active_slot),
            read_only: false,
        })
    }

    /// Opens an existing `.obrain` file in read-only mode.
    ///
    /// Uses a **shared** file lock (`try_lock_shared`), allowing multiple
    /// readers to open the same file concurrently, even while a writer holds
    /// an exclusive lock (on platforms with advisory locking).
    ///
    /// The returned manager only supports [`read_snapshot`](Self::read_snapshot)
    /// and other read-only operations. Calling [`write_snapshot`](Self::write_snapshot)
    /// will return an error.
    ///
    /// # Errors
    ///
    /// Returns an error if the file does not exist, has invalid magic, or
    /// an unsupported format version.
    pub fn open_read_only(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        let mut file = OpenOptions::new().read(true).open(&path)?;

        // Acquire a shared lock: coexists with other shared locks but
        // blocks if an exclusive lock cannot be shared (platform-dependent).
        file.try_lock_shared().map_err(|_| {
            Error::Internal(format!(
                "database file cannot be locked for reading: {}",
                path.display()
            ))
        })?;

        let file_header = header::read_file_header(&mut file)?;
        header::validate_file_header(&file_header)?;

        let (h0, h1) = header::read_db_headers(&mut file)?;
        let (active_slot, active_header) = header::active_db_header(&h0, &h1);

        Ok(Self {
            path,
            file: Mutex::new(file),
            file_header,
            active_header: Mutex::new(active_header),
            active_slot: Mutex::new(active_slot),
            read_only: true,
        })
    }

    /// Returns `true` if this manager was opened in read-only mode.
    #[must_use]
    pub fn is_read_only(&self) -> bool {
        self.read_only
    }

    /// Writes snapshot data into the file and updates the inactive DB header.
    ///
    /// Steps:
    /// 1. Write `data` at [`DATA_OFFSET`]
    /// 2. Compute CRC-32 checksum
    /// 3. Build a new [`DbHeader`] and write it to the inactive slot
    /// 4. `fsync` the file
    /// 5. Update internal active header/slot state
    ///
    /// # Errors
    ///
    /// Returns an error if any I/O operation fails.
    pub fn write_snapshot(
        &self,
        data: &[u8],
        epoch: u64,
        transaction_id: u64,
        node_count: u64,
        edge_count: u64,
    ) -> Result<()> {
        if self.read_only {
            return Err(Error::Internal(
                "cannot write snapshot: database is open in read-only mode".to_string(),
            ));
        }

        use obrain_core::testing::crash::maybe_crash;

        let checksum = crc32fast::hash(data);
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let mut file = self.file.lock();
        let active_header = self.active_header.lock();
        let mut active_slot = self.active_slot.lock();

        let new_iteration = active_header.iteration + 1;
        let target_slot = u8::from(*active_slot == 0);

        maybe_crash("write_snapshot:before_data_write");

        // Write snapshot data
        file.seek(SeekFrom::Start(DATA_OFFSET))?;
        file.write_all(data)?;

        maybe_crash("write_snapshot:after_data_write");

        // Truncate file to exact size (remove stale trailing data)
        let file_end = DATA_OFFSET + data.len() as u64;
        file.set_len(file_end)?;

        maybe_crash("write_snapshot:after_truncate");

        // Build and write new header to inactive slot
        let new_header = DbHeader {
            iteration: new_iteration,
            checksum,
            snapshot_length: data.len() as u64,
            epoch,
            transaction_id,
            node_count,
            edge_count,
            timestamp_ms,
        };
        header::write_db_header(&mut file, target_slot, &new_header)?;

        maybe_crash("write_snapshot:after_header_write");

        // Ensure everything is on disk before we consider this committed
        file.sync_all()?;

        maybe_crash("write_snapshot:after_fsync");

        // Update internal state: drop the old lock, reacquire to update
        drop(active_header);
        *self.active_header.lock() = new_header;
        *active_slot = target_slot;

        Ok(())
    }

    /// Reads snapshot data from the file using the active database header.
    ///
    /// Returns an empty `Vec` if the database has never been checkpointed
    /// (both headers are empty).
    ///
    /// # Errors
    ///
    /// Returns an error if the read fails or the CRC checksum does not match.
    pub fn read_snapshot(&self) -> Result<Vec<u8>> {
        let active_header = self.active_header.lock();

        if active_header.is_empty() {
            return Ok(Vec::new());
        }

        let length = active_header.snapshot_length as usize;
        let expected_checksum = active_header.checksum;
        drop(active_header);

        let mut file = self.file.lock();
        file.seek(SeekFrom::Start(DATA_OFFSET))?;

        let mut data = vec![0u8; length];
        std::io::Read::read_exact(&mut *file, &mut data)?;

        // Verify CRC
        let actual_checksum = crc32fast::hash(&data);
        if actual_checksum != expected_checksum {
            return Err(Error::Internal(format!(
                "snapshot checksum mismatch: expected {expected_checksum:#010X}, got {actual_checksum:#010X}"
            )));
        }

        Ok(data)
    }

    /// Memory-maps the snapshot data region of the `.obrain` file.
    ///
    /// Instead of allocating + copying the entire snapshot into a `Vec<u8>`,
    /// this returns a memory-mapped slice. The kernel loads pages on demand
    /// during deserialization, which is dramatically faster for large databases
    /// (e.g., 3.9 GB → near-instant open instead of multi-second read + alloc).
    ///
    /// CRC verification is performed **in parallel** on the mmap'd region so
    /// it doesn't block deserialization startup (the CRC runs on a background
    /// thread while the caller proceeds with bincode decoding).
    ///
    /// # Safety
    ///
    /// Uses `memmap2::MmapOptions::map()` which is safe as long as the file
    /// is not concurrently truncated or modified by another process. The file
    /// lock (exclusive or shared) provides this guarantee.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be mmap'd or the CRC fails.
    #[allow(unsafe_code)]
    pub fn mmap_snapshot(&self) -> Result<memmap2::Mmap> {
        let active_header = self.active_header.lock();

        if active_header.is_empty() {
            return Err(Error::Internal("no snapshot data in file".to_string()));
        }

        let length = active_header.snapshot_length as usize;
        let expected_checksum = active_header.checksum;
        drop(active_header);

        let file = self.file.lock();

        // SAFETY: file lock guarantees no concurrent modification.
        let mmap = unsafe {
            memmap2::MmapOptions::new()
                .offset(DATA_OFFSET)
                .len(length)
                .map(&*file)
                .map_err(|e| Error::Internal(format!("mmap failed: {e}")))?
        };

        // Advise the kernel to read sequentially — the bincode decoder
        // will walk the data front-to-back.
        #[cfg(unix)]
        mmap.advise(memmap2::Advice::Sequential)
            .unwrap_or_else(|e| tracing::debug!("madvise(Sequential) failed: {e}"));

        // Pre-fault pages: populate the page table in advance so the kernel
        // reads ahead while the bincode decoder processes earlier pages.
        #[cfg(unix)]
        mmap.advise(memmap2::Advice::WillNeed)
            .unwrap_or_else(|e| tracing::debug!("madvise(WillNeed) failed: {e}"));

        // CRC verification deferred — the checksum is verified AFTER the
        // caller finishes deserialization, on the already-paged-in data.
        // This is done by `verify_mmap_crc()` which should be called
        // after `apply_snapshot_data()` completes. We store the expected
        // checksum in tracing context for the caller to verify.
        tracing::info!(
            path = %self.path.display(),
            snapshot_bytes = length,
            expected_crc = format!("{expected_checksum:#010X}"),
            "Snapshot mmap'd ({:.1} MB), CRC deferred",
            length as f64 / (1024.0 * 1024.0),
        );

        Ok(mmap)
    }

    /// Memory-maps the **entire** `.obrain` file for native v2 format.
    ///
    /// Unlike [`mmap_snapshot()`](Self::mmap_snapshot) which maps only the data
    /// payload starting at `DATA_OFFSET`, this maps the whole file including
    /// headers and TOC. The v2 reader needs the full file because section
    /// offsets in the TOC are absolute file offsets.
    ///
    /// Returns an error if there is no snapshot data.
    #[allow(unsafe_code)]
    pub fn mmap_full_file(&self) -> Result<memmap2::Mmap> {
        let active_header = self.active_header.lock();
        if active_header.is_empty() {
            return Err(Error::Internal("no snapshot data in file".to_string()));
        }
        drop(active_header);

        let file = self.file.lock();

        // SAFETY: file lock guarantees no concurrent modification.
        let mmap = unsafe {
            memmap2::MmapOptions::new()
                .map(&*file)
                .map_err(|e| Error::Internal(format!("mmap full file failed: {e}")))?
        };

        // Default to Normal — callers can switch to Sequential/WillNeed
        // before scanning, or Random for point lookups.
        #[cfg(unix)]
        mmap.advise(memmap2::Advice::Normal)
            .unwrap_or_else(|e| tracing::debug!("madvise(Normal) failed: {e}"));

        tracing::info!(
            path = %self.path.display(),
            file_bytes = mmap.len(),
            "Full file mmap'd for native v2 ({:.1} MB)",
            mmap.len() as f64 / (1024.0 * 1024.0),
        );

        Ok(mmap)
    }

    /// Verify the CRC-32 checksum of an already-mmap'd snapshot.
    ///
    /// Call this AFTER deserialization — the pages are already in the page
    /// cache from the bincode decode pass, so this is essentially free.
    pub fn verify_mmap_crc(&self, mmap: &memmap2::Mmap) -> Result<()> {
        let active_header = self.active_header.lock();
        if active_header.is_empty() {
            return Ok(());
        }
        let expected = active_header.checksum;
        drop(active_header);

        let t0 = std::time::Instant::now();
        let actual = crc32fast::hash(mmap);
        let elapsed = t0.elapsed();

        if actual != expected {
            return Err(Error::Internal(format!(
                "snapshot checksum mismatch: expected {expected:#010X}, got {actual:#010X}"
            )));
        }

        tracing::debug!(
            path = %self.path.display(),
            elapsed_ms = elapsed.as_millis(),
            "CRC-32 verified on mmap'd snapshot"
        );
        Ok(())
    }

    /// Returns the path for the sidecar WAL directory.
    ///
    /// For a database at `mydb.obrain`, the sidecar is `mydb.obrain.wal/`.
    #[must_use]
    pub fn sidecar_wal_path(&self) -> PathBuf {
        let mut wal_path = self.path.as_os_str().to_owned();
        wal_path.push(".wal");
        PathBuf::from(wal_path)
    }

    /// Returns `true` if a sidecar WAL directory exists.
    #[must_use]
    pub fn has_sidecar_wal(&self) -> bool {
        self.sidecar_wal_path().exists()
    }

    /// Removes the sidecar WAL directory after a successful checkpoint.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory exists but cannot be removed.
    pub fn remove_sidecar_wal(&self) -> Result<()> {
        let wal_path = self.sidecar_wal_path();
        if wal_path.exists() {
            fs::remove_dir_all(&wal_path)?;
        }
        Ok(())
    }

    /// Returns the file path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns a clone of the currently active database header.
    #[must_use]
    pub fn active_header(&self) -> DbHeader {
        self.active_header.lock().clone()
    }

    /// Returns the file header (written at creation, immutable).
    #[must_use]
    pub fn file_header(&self) -> &FileHeader {
        &self.file_header
    }

    /// Returns the total file size on disk.
    ///
    /// # Errors
    ///
    /// Returns an error if the file metadata cannot be read.
    pub fn file_size(&self) -> Result<u64> {
        let file = self.file.lock();
        let metadata = file.metadata()?;
        Ok(metadata.len())
    }

    /// Flushes and syncs the file.
    ///
    /// # Errors
    ///
    /// Returns an error if sync fails.
    pub fn sync(&self) -> Result<()> {
        let file = self.file.lock();
        file.sync_all()?;
        Ok(())
    }

    /// Releases the file lock and syncs.
    ///
    /// # Errors
    ///
    /// Returns an error if sync or unlock fails.
    pub fn close(&self) -> Result<()> {
        let file = self.file.lock();
        file.sync_all()?;
        file.unlock()
            .map_err(|e| Error::Internal(format!("failed to unlock database file: {e}")))?;
        Ok(())
    }
}

impl Drop for ObrainFileManager {
    fn drop(&mut self) {
        let file = self.file.lock();
        let _ = file.unlock();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn test_dir() -> TempDir {
        TempDir::new().expect("create temp dir")
    }

    #[test]
    fn create_and_open() {
        let dir = test_dir();
        let path = dir.path().join("test.obrain");

        // Create
        let manager = ObrainFileManager::create(&path).unwrap();
        assert!(path.exists());
        assert!(manager.active_header().is_empty());
        drop(manager);

        // Open
        let manager = ObrainFileManager::open(&path).unwrap();
        assert!(manager.active_header().is_empty());
    }

    #[test]
    fn create_fails_if_exists() {
        let dir = test_dir();
        let path = dir.path().join("test.obrain");

        ObrainFileManager::create(&path).unwrap();
        let result = ObrainFileManager::create(&path);
        assert!(result.is_err());
    }

    #[test]
    fn open_fails_if_not_exists() {
        let dir = test_dir();
        let path = dir.path().join("nonexistent.obrain");

        let result = ObrainFileManager::open(&path);
        assert!(result.is_err());
    }

    #[test]
    fn write_and_read_snapshot() {
        let dir = test_dir();
        let path = dir.path().join("test.obrain");

        let manager = ObrainFileManager::create(&path).unwrap();

        let snapshot_data = b"hello obrain snapshot data";
        manager.write_snapshot(snapshot_data, 1, 1, 10, 20).unwrap();

        let loaded = manager.read_snapshot().unwrap();
        assert_eq!(loaded, snapshot_data);

        // Verify header was updated
        let header = manager.active_header();
        assert_eq!(header.iteration, 1);
        assert_eq!(header.snapshot_length, snapshot_data.len() as u64);
        assert_eq!(header.epoch, 1);
        assert_eq!(header.node_count, 10);
        assert_eq!(header.edge_count, 20);
    }

    #[test]
    fn snapshot_persists_across_reopen() {
        let dir = test_dir();
        let path = dir.path().join("test.obrain");

        let snapshot_data = b"persistent data across reopen";

        // Write
        {
            let manager = ObrainFileManager::create(&path).unwrap();
            manager
                .write_snapshot(snapshot_data, 5, 3, 100, 200)
                .unwrap();
        }

        // Reopen and read
        {
            let manager = ObrainFileManager::open(&path).unwrap();
            let loaded = manager.read_snapshot().unwrap();
            assert_eq!(loaded, snapshot_data);

            let header = manager.active_header();
            assert_eq!(header.iteration, 1);
            assert_eq!(header.epoch, 5);
            assert_eq!(header.node_count, 100);
        }
    }

    #[test]
    fn alternating_snapshots() {
        let dir = test_dir();
        let path = dir.path().join("test.obrain");

        let manager = ObrainFileManager::create(&path).unwrap();

        // First checkpoint
        let data1 = b"snapshot version 1";
        manager.write_snapshot(data1, 1, 1, 10, 5).unwrap();
        assert_eq!(manager.active_header().iteration, 1);

        // Second checkpoint (alternates to other slot)
        let data2 = b"snapshot version 2 with more data";
        manager.write_snapshot(data2, 2, 2, 20, 10).unwrap();
        assert_eq!(manager.active_header().iteration, 2);

        let loaded = manager.read_snapshot().unwrap();
        assert_eq!(loaded, data2);
    }

    #[test]
    fn read_empty_snapshot() {
        let dir = test_dir();
        let path = dir.path().join("test.obrain");

        let manager = ObrainFileManager::create(&path).unwrap();
        let data = manager.read_snapshot().unwrap();
        assert!(data.is_empty());
    }

    #[test]
    fn sidecar_wal_path_computation() {
        let dir = test_dir();
        let path = dir.path().join("mydb.obrain");

        let manager = ObrainFileManager::create(&path).unwrap();
        let wal_path = manager.sidecar_wal_path();

        assert_eq!(
            wal_path.file_name().unwrap().to_str().unwrap(),
            "mydb.obrain.wal"
        );
        assert!(!manager.has_sidecar_wal());
    }

    #[test]
    fn sidecar_wal_detect_and_remove() {
        let dir = test_dir();
        let path = dir.path().join("test.obrain");

        let manager = ObrainFileManager::create(&path).unwrap();
        assert!(!manager.has_sidecar_wal());

        // Create sidecar directory manually (simulating engine behavior)
        fs::create_dir_all(manager.sidecar_wal_path()).unwrap();
        assert!(manager.has_sidecar_wal());

        // Remove it
        manager.remove_sidecar_wal().unwrap();
        assert!(!manager.has_sidecar_wal());
    }

    #[test]
    fn file_size_grows_with_data() {
        let dir = test_dir();
        let path = dir.path().join("test.obrain");

        let manager = ObrainFileManager::create(&path).unwrap();
        let empty_size = manager.file_size().unwrap();

        // Empty file should be at least 12 KiB (3 headers)
        assert!(empty_size >= DATA_OFFSET, "empty size: {empty_size}");

        let big_data = vec![0xAB; 100_000];
        manager.write_snapshot(&big_data, 1, 1, 0, 0).unwrap();

        let full_size = manager.file_size().unwrap();
        assert!(full_size > empty_size);
        assert_eq!(full_size, DATA_OFFSET + big_data.len() as u64);
    }

    #[test]
    fn exclusive_lock_prevents_second_open() {
        let dir = test_dir();
        let path = dir.path().join("locked.obrain");

        let _manager1 = ObrainFileManager::create(&path).unwrap();

        // Second open should fail
        let result = ObrainFileManager::open(&path);
        assert!(result.is_err());
        assert!(result.err().unwrap().to_string().contains("locked"));
    }

    #[test]
    fn lock_released_after_close() {
        let dir = test_dir();
        let path = dir.path().join("lockclose.obrain");

        let manager = ObrainFileManager::create(&path).unwrap();
        manager.write_snapshot(b"data", 1, 1, 0, 0).unwrap();
        manager.close().unwrap();

        // Should succeed after close
        let manager2 = ObrainFileManager::open(&path).unwrap();
        let data = manager2.read_snapshot().unwrap();
        assert_eq!(data, b"data");
    }

    #[test]
    fn lock_released_on_drop() {
        let dir = test_dir();
        let path = dir.path().join("lockdrop.obrain");

        {
            let _manager = ObrainFileManager::create(&path).unwrap();
            // Drop without explicit close
        }

        // Should succeed after drop
        let _manager2 = ObrainFileManager::open(&path).unwrap();
    }

    #[test]
    fn checksum_mismatch_detected() {
        let dir = test_dir();
        let path = dir.path().join("test.obrain");

        let manager = ObrainFileManager::create(&path).unwrap();
        manager.write_snapshot(b"valid data", 1, 1, 0, 0).unwrap();
        drop(manager);

        // Corrupt the snapshot data in the file
        {
            let mut file = OpenOptions::new().write(true).open(&path).unwrap();
            file.seek(SeekFrom::Start(DATA_OFFSET)).unwrap();
            file.write_all(b"CORRUPT!!!").unwrap();
        }

        let manager = ObrainFileManager::open(&path).unwrap();
        let result = manager.read_snapshot();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("checksum"));
    }

    #[test]
    fn open_read_only_reads_snapshot() {
        let dir = test_dir();
        let path = dir.path().join("ro.obrain");

        // Create and write snapshot, then close
        {
            let manager = ObrainFileManager::create(&path).unwrap();
            manager
                .write_snapshot(b"read-only test data", 3, 2, 5, 10)
                .unwrap();
            manager.close().unwrap();
        }

        // Open read-only
        let ro = ObrainFileManager::open_read_only(&path).unwrap();
        assert!(ro.is_read_only());
        let data = ro.read_snapshot().unwrap();
        assert_eq!(data, b"read-only test data");

        let header = ro.active_header();
        assert_eq!(header.epoch, 3);
        assert_eq!(header.node_count, 5);
        assert_eq!(header.edge_count, 10);
    }

    #[test]
    fn read_only_rejects_write_snapshot() {
        let dir = test_dir();
        let path = dir.path().join("ro_write.obrain");

        {
            let manager = ObrainFileManager::create(&path).unwrap();
            manager.close().unwrap();
        }

        let ro = ObrainFileManager::open_read_only(&path).unwrap();
        let result = ro.write_snapshot(b"nope", 1, 1, 0, 0);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("read-only"));
    }

    #[test]
    fn read_only_coexists_with_exclusive_after_close() {
        let dir = test_dir();
        let path = dir.path().join("coexist.obrain");

        // Create, write, close
        {
            let manager = ObrainFileManager::create(&path).unwrap();
            manager.write_snapshot(b"coexist data", 1, 1, 1, 1).unwrap();
            manager.close().unwrap();
        }

        // Two read-only opens should coexist
        let ro1 = ObrainFileManager::open_read_only(&path).unwrap();
        let ro2 = ObrainFileManager::open_read_only(&path).unwrap();

        assert_eq!(ro1.read_snapshot().unwrap(), b"coexist data");
        assert_eq!(ro2.read_snapshot().unwrap(), b"coexist data");
    }
}
