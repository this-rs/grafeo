//! Uncached pread-based reader for the Strings section of a v2 `.obrain` file.
//!
//! This is a companion to [`MmapStore`](crate::database::mmap_store::MmapStore):
//! the mmap is kept for the *small* index/catalog sections (nodes, props, keys),
//! while the *large* Strings section is read via a separate file descriptor
//! with the buffer cache disabled.
//!
//! ## Why?
//!
//! On very large DBs (10+ GB strings section), mmap-based reads progressively
//! fill the OS page cache. Under memory pressure — especially on macOS with a
//! near-full swap — this triggers jetsam kills that are hard to predict.
//!
//! By setting `fcntl(F_NOCACHE, 1)` on a dedicated fd, pread() calls bypass
//! the buffer cache entirely. The resident-set stays bounded at the cost of
//! slightly slower reads (no OS read-ahead coalescing).
//!
//! On Linux we use `posix_fadvise(POSIX_FADV_DONTNEED)` which has a similar
//! effect of keeping the page cache out of our way.

use std::fs::File;
use std::io;
use std::os::unix::fs::FileExt;
use std::os::unix::io::AsRawFd;
use std::path::Path;

/// Reader for the Strings section of a `.obrain` file that bypasses the
/// OS buffer cache to keep resident-set bounded during bulk extraction.
pub struct NocacheStringReader {
    file: File,
    strings_file_offset: u64,
}

impl NocacheStringReader {
    /// Opens `path` with read-only access and configures the fd to bypass
    /// the buffer cache.
    ///
    /// - On macOS: sets `F_NOCACHE` via `fcntl()`.
    /// - On Linux: issues `posix_fadvise(POSIX_FADV_RANDOM)` as a hint;
    ///   the caller may additionally call [`advise_dontneed`] to drop
    ///   already-cached pages.
    ///
    /// `strings_file_offset` is the absolute file offset where the Strings
    /// section starts (obtain it from `MmapStore::strings_file_offset()`).
    pub fn open(path: impl AsRef<Path>, strings_file_offset: u64) -> io::Result<Self> {
        let file = File::open(path)?;

        #[cfg(target_vendor = "apple")]
        unsafe {
            // F_NOCACHE = 48 on Darwin — disables the unified buffer cache
            // for this fd. Reads go straight to the device layer.
            let rc = libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1);
            if rc < 0 {
                return Err(io::Error::last_os_error());
            }
        }

        #[cfg(all(target_os = "linux", not(target_vendor = "apple")))]
        unsafe {
            // posix_fadvise(RANDOM) disables read-ahead which at least
            // prevents unnecessary prefetching. For true uncached reads
            // on Linux, use O_DIRECT (but that requires alignment).
            let _ = libc::posix_fadvise(
                file.as_raw_fd(),
                0,
                0,
                libc::POSIX_FADV_RANDOM,
            );
        }

        Ok(Self {
            file,
            strings_file_offset,
        })
    }

    /// Reads `len` bytes at `offset` within the Strings section and returns
    /// them as a UTF-8 string. Returns `None` on I/O error or invalid UTF-8.
    pub fn read_string(&self, offset: u32, len: u32) -> Option<String> {
        if len == 0 {
            return Some(String::new());
        }
        let file_off = self.strings_file_offset.checked_add(offset as u64)?;
        let mut buf = vec![0u8; len as usize];
        self.file.read_exact_at(&mut buf, file_off).ok()?;
        String::from_utf8(buf).ok()
    }

    /// On Linux, advise the kernel to drop any cached pages of this file
    /// that we've already read — useful to call periodically during long
    /// extraction loops. No-op on macOS (F_NOCACHE already bypasses cache).
    #[allow(dead_code)]
    pub fn advise_dontneed(&self) {
        #[cfg(all(target_os = "linux", not(target_vendor = "apple")))]
        unsafe {
            let _ = libc::posix_fadvise(
                self.file.as_raw_fd(),
                0,
                0,
                libc::POSIX_FADV_DONTNEED,
            );
        }
    }
}
