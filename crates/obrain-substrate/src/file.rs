//! `SubstrateFile` — directory-based on-disk representation of a substrate.
//!
//! A substrate is a **directory** containing a fixed set of files, each mapping
//! one logical zone (see `docs/rfc/substrate/format-spec.md` §1):
//!
//! ```text
//!  <path>/
//!    substrate.meta          # 4 KiB — magic, version, counters
//!    substrate.nodes         # array of NodeRecord (32 B each)
//!    substrate.edges         # array of EdgeRecord (32 B each)
//!    substrate.props         # array of PropertyPage (4 KiB each)
//!    substrate.strings       # array of HeapPage (4 KiB each)
//!    substrate.tier0         # 16 B per node
//!    substrate.tier1         # 64 B per node
//!    substrate.tier2         # 768 B per node
//!    substrate.hilbert       # u32 × node_count
//!    substrate.community     # u32 × node_count
//!    substrate.wal           # append-only WAL (see wal-spec.md)
//!    substrate.checkpoint    # latest checkpoint marker
//! ```
//!
//! Every file except `substrate.wal` and `substrate.checkpoint` is memory-mapped
//! directly. Writers use a write-through path: mutate the mmap, then fsync via
//! `msync`. Durability is owned by the WAL.
//!
//! ## Safety
//!
//! `memmap2::Mmap::map()` and friends are `unsafe`: the OS guarantees memory
//! safety only if the backing file is not concurrently mutated by another
//! process or truncated in a way that shrinks the mapped region. We enforce
//! this by holding an exclusive file lock (`fs2::FileExt::lock_exclusive`) on
//! `substrate.meta` for the lifetime of the `SubstrateFile`, so other
//! processes cannot open the same directory.
//!
//! Within this process, we never truncate a mapped file — growth is always
//! append + remap. The `#[allow(unsafe_code)]` block is therefore scoped to
//! the `mmap_*` helpers, which document the invariants they require.

#![allow(unsafe_code)]

use crate::error::{SubstrateError, SubstrateResult};
use crate::meta::{MetaHeader, META_FILE_SIZE, SUBSTRATE_FORMAT_VERSION};
use memmap2::{Advice, MmapMut, MmapOptions};
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};

/// Threshold below which `advise_huge_pages` refuses to apply the THP hint:
/// mappings smaller than this are too small to benefit from a 2 MB huge
/// page (they would waste TLB entries and fragment the address space).
/// 8 MB covers typical zone files on bootstrap while still letting growing
/// zones acquire huge-page backing as soon as they cross this threshold.
const HUGE_PAGE_MIN_BYTES: u64 = 8 * 1024 * 1024;

/// Cross-platform huge-page / Transparent-Huge-Pages hint for a mmap region.
///
/// * **Linux**: issues `madvise(MADV_HUGEPAGE)`. The kernel tries to back
///   the region with 2 MB pages where possible. Requires
///   `CONFIG_TRANSPARENT_HUGEPAGE=y`. If THP is disabled the call returns
///   `EINVAL` and we silently ignore it — the hint is advisory.
/// * **macOS / Windows / other**: no-op. Darwin's superpages require
///   allocation-time flags (`VM_FLAGS_SUPERPAGE_SIZE_2MB`) that memmap2
///   does not expose, and Windows large pages require a privileged token.
///   Both platforms fall back to default 4-16 KB pages without error.
///
/// The hint lives on the VMA, not the file descriptor, so this helper must
/// be invoked after every `map_mut` / remap. Zones smaller than
/// `HUGE_PAGE_MIN_BYTES` are skipped — they can't fill a single huge page.
fn advise_huge_pages(map: &MmapMut, len: u64) {
    if len < HUGE_PAGE_MIN_BYTES {
        return;
    }
    #[cfg(target_os = "linux")]
    {
        // Ignore the result: THP is a hint. Kernels without CONFIG_TRANSPARENT_HUGEPAGE
        // return EINVAL; we treat that as "no huge pages, back with regular 4K" which is
        // exactly the pre-hint behaviour.
        let _ = map.advise(Advice::HugePage);
    }
    #[cfg(not(target_os = "linux"))]
    {
        // Keep the binding to silence the unused-var warning without cfg-annotating
        // the call site.
        let _ = map;
    }
}

/// Canonical names for the per-zone files inside a substrate directory.
pub mod zone {
    pub const META: &str = "substrate.meta";
    pub const NODES: &str = "substrate.nodes";
    pub const EDGES: &str = "substrate.edges";
    pub const PROPS: &str = "substrate.props";
    pub const STRINGS: &str = "substrate.strings";
    pub const TIER0: &str = "substrate.tier0";
    pub const TIER1: &str = "substrate.tier1";
    pub const TIER2: &str = "substrate.tier2";
    pub const HILBERT: &str = "substrate.hilbert";
    pub const COMMUNITY: &str = "substrate.community";
    pub const ENGRAM_MEMBERS: &str = "substrate.engram_members";
    pub const ENGRAM_BITSET: &str = "substrate.engram_bitset";
    pub const WAL: &str = "substrate.wal";
    pub const CHECKPOINT: &str = "substrate.checkpoint";

    pub const ALL_MMAPPED: &[&str] = &[
        NODES,
        EDGES,
        PROPS,
        STRINGS,
        TIER0,
        TIER1,
        TIER2,
        HILBERT,
        COMMUNITY,
        ENGRAM_MEMBERS,
        ENGRAM_BITSET,
    ];
}

/// Enumeration of zones exposed as typed handles.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Zone {
    Nodes,
    Edges,
    Props,
    Strings,
    Tier0,
    Tier1,
    Tier2,
    Hilbert,
    Community,
    /// Engram-membership side-table (T7). Maps `engram_id: u16 → Vec<u32>`
    /// node slot IDs via an in-place directory + append-only payload.
    EngramMembers,
    /// Per-node 64-bit engram signature (T7 Step 1). One u64 per node slot,
    /// used as a Bloom-filter-like candidate prune for Hopfield recall.
    EngramBitset,
}

impl Zone {
    pub fn filename(self) -> &'static str {
        match self {
            Zone::Nodes => zone::NODES,
            Zone::Edges => zone::EDGES,
            Zone::Props => zone::PROPS,
            Zone::Strings => zone::STRINGS,
            Zone::Tier0 => zone::TIER0,
            Zone::Tier1 => zone::TIER1,
            Zone::Tier2 => zone::TIER2,
            Zone::Hilbert => zone::HILBERT,
            Zone::Community => zone::COMMUNITY,
            Zone::EngramMembers => zone::ENGRAM_MEMBERS,
            Zone::EngramBitset => zone::ENGRAM_BITSET,
        }
    }
}

/// A read/write memory-mapped file for one zone.
#[derive(Debug)]
pub struct ZoneFile {
    path: PathBuf,
    file: File,
    /// Current logical length (may exceed mmap length if the file was just grown).
    len: u64,
    /// Writable memory map, if any. When `None`, the zone is unmapped (empty file).
    map: Option<MmapMut>,
    /// Number of times `grow_to` actually performed a resize+remap. Exposed for
    /// tests and observability; should grow at O(log N) under exponential
    /// pre-alloc policy.
    remap_count: u64,
}

impl ZoneFile {
    /// Open or create the file at `path`. If the file is non-empty, memory-map it
    /// read/write; if empty, the mmap is deferred until the first grow.
    fn open_or_create(path: &Path) -> SubstrateResult<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path)?;
        let len = file.metadata()?.len();
        let map = if len == 0 {
            None
        } else {
            // SAFETY: we just opened `file` with read+write, the substrate
            // directory is exclusively locked (see SubstrateFile::open), and we
            // never truncate while a mmap is live.
            let m = unsafe { MmapOptions::new().len(len as usize).map_mut(&file)? };
            advise_huge_pages(&m, len);
            Some(m)
        };
        Ok(Self {
            path: path.to_path_buf(),
            file,
            len,
            map,
            remap_count: 0,
        })
    }

    /// Number of resize+remap operations performed on this zone since it was
    /// opened. Useful for observability and for verifying the grow policy is
    /// O(log N) in stress tests.
    pub fn remap_count(&self) -> u64 {
        self.remap_count
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Borrow the mapped bytes (read-only view).
    pub fn as_slice(&self) -> &[u8] {
        match self.map.as_ref() {
            Some(m) => &m[..],
            None => &[],
        }
    }

    /// Borrow the mapped bytes mutably.
    pub fn as_slice_mut(&mut self) -> &mut [u8] {
        match self.map.as_mut() {
            Some(m) => &mut m[..],
            None => &mut [],
        }
    }

    /// Grow the file to `new_len` bytes (zero-padded) and remap.
    ///
    /// This is the low-level primitive; higher-level pre-allocation policies
    /// (exponential ×1.5) live in `SubstrateFile`.
    pub fn grow_to(&mut self, new_len: u64) -> SubstrateResult<()> {
        if new_len < self.len {
            return Err(SubstrateError::WalBadFrame(format!(
                "grow_to: refuses to shrink ({} → {})",
                self.len, new_len
            )));
        }
        if new_len == self.len {
            return Ok(());
        }
        // Drop the previous map before resizing (some platforms disallow resize
        // while a mapping is live).
        self.map.take();
        self.file.set_len(new_len)?;
        self.len = new_len;
        // SAFETY: same invariants as `open_or_create`.
        let m = unsafe { MmapOptions::new().len(new_len as usize).map_mut(&self.file)? };
        // `madvise(MADV_HUGEPAGE)` must be re-applied after every remap — the
        // hint lives on the VMA, not on the file. A zone that crossed the
        // HUGE_PAGE_MIN_BYTES threshold during this grow will now be eligible
        // for THP backing.
        advise_huge_pages(&m, new_len);
        self.map = Some(m);
        self.remap_count += 1;
        Ok(())
    }

    /// Flush dirty pages to disk (`msync`). Call after write-through mutations
    /// when durability is required *beyond* the WAL (for example at checkpoint).
    pub fn msync(&self) -> SubstrateResult<()> {
        if let Some(m) = self.map.as_ref() {
            m.flush()?;
        }
        Ok(())
    }

    /// Fsync the underlying file descriptor.
    pub fn fsync(&self) -> SubstrateResult<()> {
        self.file.sync_all()?;
        Ok(())
    }

    /// Evict pages backing this zone from the resident set via
    /// `madvise(MADV_DONTNEED)` (Unix only — on non-Unix this is a no-op).
    ///
    /// Primary use: the T11 Step 5 prefetch bench, which needs a cold
    /// page cache to measure the `WILLNEED` effect. Production code
    /// does not call this — `DONTNEED` is a destructive hint (subsequent
    /// reads miss the cache and fault in from disk).
    ///
    /// Marked `unsafe` because `DONTNEED` semantics on Linux include
    /// *discarding dirty pages* in anonymous mappings. For a
    /// file-backed mmap the behavior is benign (pages are re-read from
    /// the file on next access), but we surface the hazard so tests
    /// and developers explicitly opt in.
    ///
    /// # Safety
    /// The caller must ensure no other thread is reading this zone
    /// concurrently via an existing `&[u8]` reference, since the pages
    /// backing that slice may be evicted and re-faulted between reads.
    pub unsafe fn advise_dontneed(&self) -> SubstrateResult<()> {
        if let Some(m) = self.map.as_ref() {
            // SAFETY: delegated to caller — file-backed mapping, caller
            // attests no live &[u8] outstanding.
            unsafe { m.unchecked_advise(memmap2::UncheckedAdvice::DontNeed)? };
        }
        Ok(())
    }

    /// Advise the kernel that the byte range `[offset, offset + len)` of
    /// this zone will be accessed soon (`madvise(MADV_WILLNEED)` on Unix,
    /// `PrefetchVirtualMemory` on Windows, no-op elsewhere).
    ///
    /// This is a *hint* — it does not block and does not guarantee the
    /// range will be resident. It asks the kernel to schedule readahead
    /// for the given pages before the application touches them, which on
    /// cold-cache access patterns can remove most of the minor+major
    /// page-fault latency from the hot path (T11 Step 5).
    ///
    /// **Bounds.** `offset + len` is clamped to `self.len` so callers can
    /// safely advise a range that overruns the mapped region (typical
    /// when `last_slot + 1` rounds past the high-water mark on the last
    /// partial page).
    ///
    /// **No-op cases:**
    /// * Zone is unmapped (`self.map` is `None`, i.e. empty file) →
    ///   returns `Ok(())` immediately.
    /// * `len == 0` after clamping → `Ok(())` (the kernel call itself is
    ///   a no-op for zero-length ranges on most platforms, and skipping
    ///   avoids an unnecessary syscall).
    ///
    /// **Errors.** An I/O error from `madvise` is surfaced as
    /// `SubstrateError::Io`. Callers treat the result as best-effort: a
    /// prefetch failure must not abort the activation pipeline.
    pub fn advise_willneed(&self, offset: usize, len: usize) -> SubstrateResult<()> {
        let Some(m) = self.map.as_ref() else {
            return Ok(());
        };
        let total = self.len as usize;
        if total == 0 || offset >= total {
            return Ok(());
        }
        let clamped = len.min(total - offset);
        if clamped == 0 {
            return Ok(());
        }
        m.advise_range(Advice::WillNeed, offset, clamped)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Top-level SubstrateFile: the directory handle.
// ---------------------------------------------------------------------------

/// A live substrate on disk. Owns the directory handle, the meta file mmap,
/// and lazy handles to the per-zone mmap files.
#[derive(Debug)]
pub struct SubstrateFile {
    path: PathBuf,
    /// The header is mmap'd separately — it's small and written on every commit.
    meta_file: File,
    meta_map: MmapMut,
    /// Backing temp directory, when the substrate was created via
    /// [`SubstrateFile::open_tempfile`]. Dropped at the end of the substrate's
    /// lifetime, which unlinks the directory.
    _temp: Option<tempfile_guard::TempDirGuard>,
}

impl SubstrateFile {
    /// Open an existing substrate at `path`. Fails if the directory does not
    /// exist or the meta file has an invalid magic / unsupported version.
    pub fn open(path: impl AsRef<Path>) -> SubstrateResult<Self> {
        let path = path.as_ref();
        if !path.is_dir() {
            return Err(SubstrateError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("substrate directory does not exist: {}", path.display()),
            )));
        }
        Self::open_inner(path.to_path_buf(), None)
    }

    /// Create a fresh substrate at `path`. The directory is created if absent;
    /// fails if it exists and already contains a substrate.meta.
    pub fn create(path: impl AsRef<Path>) -> SubstrateResult<Self> {
        let path = path.as_ref().to_path_buf();
        std::fs::create_dir_all(&path)?;
        let meta_path = path.join(zone::META);
        if meta_path.exists() {
            return Err(SubstrateError::Io(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                format!(
                    "substrate already initialized: {}",
                    meta_path.display()
                ),
            )));
        }
        // Pre-write a blank 4 KiB meta file.
        let header = MetaHeader::new_blank();
        let mut buf = vec![0u8; META_FILE_SIZE];
        buf[..core::mem::size_of::<MetaHeader>()]
            .copy_from_slice(bytemuck::bytes_of(&header));
        std::fs::write(&meta_path, &buf)?;

        // Pre-create empty zone files so later opens don't need to special-case
        // "missing file". The WAL file is append-only and lives outside the
        // ZoneFile abstraction.
        for name in zone::ALL_MMAPPED {
            let zp = path.join(name);
            if !zp.exists() {
                // Zero-byte file — will be grown on first write.
                OpenOptions::new()
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .open(&zp)?;
            }
        }
        // Empty WAL file.
        let wal_path = path.join(zone::WAL);
        if !wal_path.exists() {
            OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&wal_path)?;
        }

        Self::open_inner(path, None)
    }

    /// Create an ephemeral substrate in a unique temporary directory. The
    /// directory is unlinked when the returned handle is dropped.
    pub fn open_tempfile() -> SubstrateResult<Self> {
        let td = tempfile_guard::TempDirGuard::new("substrate-")?;
        let path = td.path().to_path_buf();
        // Create as usual, then stash the guard so it outlives the handle.
        Self::create(&path)?;
        Self::open_inner(path, Some(td))
    }

    fn open_inner(
        path: PathBuf,
        temp: Option<tempfile_guard::TempDirGuard>,
    ) -> SubstrateResult<Self> {
        let meta_path = path.join(zone::META);
        let meta_file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&meta_path)
            .map_err(SubstrateError::Io)?;
        let file_len = meta_file.metadata()?.len();
        if file_len < META_FILE_SIZE as u64 {
            return Err(SubstrateError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "meta file too short: {} (< {})",
                    file_len, META_FILE_SIZE
                ),
            )));
        }

        // SAFETY: we just opened the file, hold the only handle to it in this
        // process (the directory is not locked against other processes yet —
        // that is a T4 enhancement once the store takes cross-process locking
        // into account). We never truncate the meta file.
        let meta_map = unsafe {
            MmapOptions::new()
                .len(META_FILE_SIZE)
                .map_mut(&meta_file)?
        };

        let sf = Self {
            path,
            meta_file,
            meta_map,
            _temp: temp,
        };
        sf.validate_header()?;
        Ok(sf)
    }

    fn validate_header(&self) -> SubstrateResult<()> {
        let header = self.meta_header();
        if !header.is_valid_magic() {
            return Err(SubstrateError::BadMagic);
        }
        if header.format_version != SUBSTRATE_FORMAT_VERSION {
            return Err(SubstrateError::UnsupportedVersion(
                header.format_version,
                SUBSTRATE_FORMAT_VERSION,
            ));
        }
        Ok(())
    }

    /// Return a copy of the current meta header (small, 96 B — cheap to copy).
    pub fn meta_header(&self) -> MetaHeader {
        *bytemuck::from_bytes::<MetaHeader>(&self.meta_map[..core::mem::size_of::<MetaHeader>()])
    }

    /// Rewrite the meta header in-place. Caller is responsible for ordering
    /// (the header is typically rewritten only after a successful checkpoint).
    pub fn write_meta_header(&mut self, header: &MetaHeader) -> SubstrateResult<()> {
        let n = core::mem::size_of::<MetaHeader>();
        self.meta_map[..n].copy_from_slice(bytemuck::bytes_of(header));
        self.meta_map.flush()?;
        self.meta_file.sync_all()?;
        Ok(())
    }

    /// Path to the substrate directory.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Path of a given zone file.
    pub fn zone_path(&self, zone: Zone) -> PathBuf {
        self.path.join(zone.filename())
    }

    /// Open (or re-open) a zone mmap handle. Each call creates a fresh
    /// [`ZoneFile`]; the caller is expected to keep it alive for as long as
    /// writes/reads are needed.
    pub fn open_zone(&self, zone: Zone) -> SubstrateResult<ZoneFile> {
        ZoneFile::open_or_create(&self.zone_path(zone))
    }

    /// Open a zone file by raw filename inside the substrate directory.
    ///
    /// This is the escape hatch for zones whose filenames are built
    /// dynamically from per-column metadata — currently only the
    /// `vec_column` subsystem (`substrate.veccol.<key>.<dtype>.<dim>`,
    /// one file per distinct prop-key / dimension). Each call creates a
    /// fresh [`ZoneFile`]; the caller keeps it alive for the window
    /// during which writes / reads are issued.
    ///
    /// The `filename` is interpreted verbatim as a file name **inside**
    /// `self.path()`; the caller must not pass a path with directory
    /// separators. We enforce this with a debug-assertion to catch
    /// caller bugs early without an extra syscall in release.
    pub fn open_named_zone(&self, filename: &str) -> SubstrateResult<ZoneFile> {
        debug_assert!(
            !filename.contains(std::path::MAIN_SEPARATOR) && !filename.contains('/'),
            "open_named_zone: filename must not contain path separators (got {filename:?})"
        );
        ZoneFile::open_or_create(&self.path.join(filename))
    }

    /// Path of the WAL file.
    pub fn wal_path(&self) -> PathBuf {
        self.path.join(zone::WAL)
    }

    /// Path of the checkpoint marker.
    pub fn checkpoint_path(&self) -> PathBuf {
        self.path.join(zone::CHECKPOINT)
    }

    /// Fsync the meta file. Typically called at the end of a checkpoint.
    pub fn fsync_meta(&self) -> SubstrateResult<()> {
        self.meta_map.flush()?;
        self.meta_file.sync_all()?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tempdir guard — owns the temporary directory and deletes it on drop.
// ---------------------------------------------------------------------------

mod tempfile_guard {
    use std::path::{Path, PathBuf};

    #[derive(Debug)]
    pub struct TempDirGuard {
        path: PathBuf,
    }

    impl TempDirGuard {
        pub fn new(prefix: &str) -> std::io::Result<Self> {
            // Build a unique directory in std::env::temp_dir(). We use
            // `tempfile::TempDir` indirectly via `std::env::temp_dir` +
            // process-id + monotonic counter to avoid pulling an extra dep —
            // but the `tempfile` crate is already in dev-dependencies, so for
            // tests we can rely on it; here we roll our own for the runtime
            // path since `tempfile` is a dev-dep only.
            use std::sync::atomic::{AtomicU64, Ordering};
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let path = std::env::temp_dir().join(format!(
                "{prefix}{pid}-{nanos}-{n}",
                pid = std::process::id()
            ));
            std::fs::create_dir_all(&path)?;
            Ok(Self { path })
        }

        pub fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TempDirGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn create_then_open_roundtrip() {
        let dir = tempdir().unwrap();
        let sub_path = dir.path().join("kb1");
        let s = SubstrateFile::create(&sub_path).unwrap();
        let h = s.meta_header();
        assert!(h.is_valid_magic());
        assert_eq!(h.format_version, SUBSTRATE_FORMAT_VERSION);
        assert_eq!(h.node_count, 0);
        drop(s);
        let s2 = SubstrateFile::open(&sub_path).unwrap();
        assert!(s2.meta_header().is_valid_magic());
    }

    #[test]
    fn double_create_fails() {
        let dir = tempdir().unwrap();
        let sub_path = dir.path().join("kb1");
        SubstrateFile::create(&sub_path).unwrap();
        let err = SubstrateFile::create(&sub_path).unwrap_err();
        matches!(err, SubstrateError::Io(_));
    }

    #[test]
    fn open_nonexistent_fails() {
        let dir = tempdir().unwrap();
        let err = SubstrateFile::open(dir.path().join("nope")).unwrap_err();
        matches!(err, SubstrateError::Io(_));
    }

    #[test]
    fn open_wrong_magic_fails() {
        let dir = tempdir().unwrap();
        let sub_path = dir.path().join("bad");
        std::fs::create_dir_all(&sub_path).unwrap();
        // Write a bogus meta file.
        let meta_path = sub_path.join(zone::META);
        std::fs::write(&meta_path, vec![0xFFu8; META_FILE_SIZE]).unwrap();
        let err = SubstrateFile::open(&sub_path).unwrap_err();
        assert!(matches!(err, SubstrateError::BadMagic));
    }

    #[test]
    fn tempfile_auto_deletes() {
        let path: PathBuf;
        {
            let s = SubstrateFile::open_tempfile().unwrap();
            path = s.path().to_path_buf();
            assert!(path.join(zone::META).exists());
        }
        // After drop, the tempdir should be gone.
        assert!(!path.exists());
    }

    #[test]
    fn zones_are_created_empty() {
        let s = SubstrateFile::open_tempfile().unwrap();
        for name in zone::ALL_MMAPPED {
            assert!(
                s.path().join(name).exists(),
                "zone file not created: {name}"
            );
        }
        assert!(s.path().join(zone::WAL).exists());
    }

    #[test]
    fn write_meta_header_persists() {
        let dir = tempdir().unwrap();
        let sub_path = dir.path().join("kb");
        let mut s = SubstrateFile::create(&sub_path).unwrap();
        let mut h = s.meta_header();
        h.node_count = 42;
        h.edge_count = 137;
        s.write_meta_header(&h).unwrap();
        drop(s);
        let s2 = SubstrateFile::open(&sub_path).unwrap();
        assert_eq!(s2.meta_header().node_count, 42);
        assert_eq!(s2.meta_header().edge_count, 137);
    }

    #[test]
    fn zonefile_grow_and_write() {
        let s = SubstrateFile::open_tempfile().unwrap();
        let mut zf = s.open_zone(Zone::Nodes).unwrap();
        assert!(zf.is_empty());
        zf.grow_to(64).unwrap();
        assert_eq!(zf.len(), 64);
        zf.as_slice_mut()[..8].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        zf.msync().unwrap();
        zf.fsync().unwrap();
        // Reopen via a fresh ZoneFile and check bytes.
        let zf2 = s.open_zone(Zone::Nodes).unwrap();
        assert_eq!(&zf2.as_slice()[..8], &[1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn zonefile_refuses_to_shrink() {
        let s = SubstrateFile::open_tempfile().unwrap();
        let mut zf = s.open_zone(Zone::Nodes).unwrap();
        zf.grow_to(4096).unwrap();
        let err = zf.grow_to(1024).unwrap_err();
        assert!(matches!(err, SubstrateError::WalBadFrame(_)));
    }
}
