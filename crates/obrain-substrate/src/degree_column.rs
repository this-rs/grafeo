//! # Degree column sidecar (T17h T5)
//!
//! Per-node in/out degree counters maintained atomically by the writer
//!
//! (`#![allow(unsafe_code)]` mirrors `simd.rs` / `props_zone.rs` — this
//! module uses `AtomicU32::from_ptr` on mmap'd memory for zero-copy
//! atomic counters. Safety invariants are documented at each call site.)
//! and queried in O(1) by graph algorithms and the Cypher planner.
//!
//! ## Why
//!
//! Graph queries like `MATCH (f:File) OPTIONAL MATCH (f)-[:IMPORTS]->()
//! WITH f, count(*) AS d RETURN f ORDER BY d DESC LIMIT 50` need a
//! per-node degree. Without a sidecar this is O(|File| × avg_degree)
//! which on PO (13k Files, 1.88M edges) costs ~256 ms. Neo4j does it in
//! ~31 ms via cached relationship counts per node. This column does the
//! same : O(1) degree lookup per node, bench target < 30 ms for the
//! TOP-50 query.
//!
//! ## On-disk format
//!
//! File : `<substrate_dir>/substrate.degrees.node.u32`
//!
//! ```text
//! [ 64 B DegreeHeader ]
//!   magic        : u32 = b"DEGR" little-endian
//!   version      : u32 = 1
//!   n_slots      : u32 = store.next_node_id at persist time
//!   record_size  : u32 = 8  (out:u32 + in:u32)
//!   crc32        : u32 = CRC32 over payload at flush time
//!   flags        : u32 = 0 (reserved)
//!   _reserved    : [u32; 10] = zero padding
//!
//! [ n_slots × (u32 out, u32 in) contiguous, little-endian ]
//!   slot 0 = sentinel null (out=0, in=0, ignored)
//!   slot 1..n_slots-1 = live or tombstoned nodes' degrees
//! ```
//!
//! Total size = 64 + 8 × n_slots bytes.
//!
//! ## Concurrency
//!
//! Slot increments are in-place atomic via `AtomicU32::from_ptr` on the
//! mmap (stable Rust 1.75+; workspace MSRV is 1.91.1). No global lock —
//! each u32 slot is its own atomic. Writers on `create_edge(src, dst)`
//! increment two disjoint slots (src.out, dst.in) without contention.
//!
//! `Relaxed` ordering : the counter is eventually-consistent with the
//! EdgeRecord mutations; no happens-before dependency.
//!
//! ## Crash safety
//!
//! CRC32 over payload guards the header. A crash before `msync` leaves
//! the header CRC stale → next `open` returns `None` → caller invokes
//! `rebuild_from_scan` (O(edges)) for a clean rebuild. Never fatal.
//!
//! Counter drift post-crash (WAL records committed but mmap not msynced):
//! replay_from reconstructs the EdgeRecord zone, then the caller can
//! invoke `SubstrateStore::rebuild_live_counters_from_zones` which
//! re-derives degrees from the canonical edge zone.
//!
//! ## Grow policy
//!
//! The column sizes itself to `next_node_id` at `create/rebuild` time.
//! If a subsequent `create_node` pushes `next_node_id > n_slots`, the
//! next `incr_out/in` call triggers a doubling grow (amortised O(1) per
//! node). Empty slots default to (0, 0).

#![allow(unsafe_code)]

use std::sync::atomic::{AtomicU32, Ordering};

use bytemuck::{Pod, Zeroable};

use crate::error::{SubstrateError, SubstrateResult};
use crate::file::{SubstrateFile, ZoneFile};
use crate::writer::ensure_room;

/// Filename for the node degree column.
pub const DEGREE_COLUMN_FILENAME: &str = "substrate.degrees.node.u32";

/// Magic bytes — little-endian ASCII "DEGR".
pub const DEGREE_MAGIC: u32 = u32::from_le_bytes(*b"DEGR");

/// Current on-disk format version.
pub const DEGREE_VERSION: u32 = 1;

/// Size of one record: u32 out + u32 in = 8 bytes.
pub const DEGREE_RECORD_SIZE: usize = 8;

/// Fixed header size in bytes.
pub const HEADER_SIZE: usize = 64;

/// On-disk header. `#[repr(C)]` + `Pod + Zeroable` for bytemuck casts.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct DegreeHeader {
    /// `DEGREE_MAGIC`.
    pub magic: u32,
    /// `DEGREE_VERSION` at write time.
    pub version: u32,
    /// Number of node slots covered by this column.
    pub n_slots: u32,
    /// `DEGREE_RECORD_SIZE` — redundant with magic but lets a hexdump
    /// reader sanity-check the payload width.
    pub record_size: u32,
    /// CRC32 (IEEE polynomial) over the payload `[out_u32, in_u32] × n_slots`.
    pub crc32: u32,
    /// Reserved flags. No bit defined yet.
    pub flags: u32,
    /// Zero-fill to bring the header to 64 B. Reserved for future
    /// versions to extend without reshuffling offsets.
    pub _reserved: [u32; 10],
}

// Static check: the header MUST be exactly 64 B.
const _: [(); 1] = [(); (std::mem::size_of::<DegreeHeader>() == HEADER_SIZE) as usize];

/// In-memory degree column handle. Wraps a mmap'd zone file.
pub struct DegreeColumn {
    zf: ZoneFile,
    n_slots: u32,
}

impl DegreeColumn {
    /// Create a fresh degree column sized for `n_slots` nodes using
    /// the default filename (T5 total degree column).
    pub fn create(sub: &SubstrateFile, n_slots: u32) -> SubstrateResult<Self> {
        Self::create_by_filename(sub, DEGREE_COLUMN_FILENAME, n_slots)
    }

    /// Create a fresh degree column at a custom filename (T17h T8 :
    /// per-edge-type columns use `substrate.degrees.node.<type>.u32`).
    ///
    /// Fills the payload with zeros, writes the header with CRC = 0
    /// (caller fills in the real CRC at flush time via `persist`).
    /// Leaves the file open for subsequent mutations.
    pub fn create_by_filename(
        sub: &SubstrateFile,
        filename: &str,
        n_slots: u32,
    ) -> SubstrateResult<Self> {
        let mut zf = sub.open_named_zone(filename)?;
        let payload_bytes = (n_slots as usize) * DEGREE_RECORD_SIZE;
        let total_bytes = HEADER_SIZE + payload_bytes;
        ensure_room(&mut zf, total_bytes, 1 << 20)?;
        // Zero-init payload (the underlying `ensure_room` grow_to already
        // zero-fills new pages).
        // Write header with crc32=0, flags=0 — the payload is all-zero so
        // the canonical crc is the CRC of a zero-filled slice, but we
        // leave it at 0 for simplicity and let `persist` recompute.
        let header = DegreeHeader {
            magic: DEGREE_MAGIC,
            version: DEGREE_VERSION,
            n_slots,
            record_size: DEGREE_RECORD_SIZE as u32,
            crc32: 0,
            flags: 0,
            _reserved: [0u32; 10],
        };
        zf.as_slice_mut()[..HEADER_SIZE]
            .copy_from_slice(bytemuck::bytes_of(&header));
        Ok(Self { zf, n_slots })
    }

    /// Open an existing total-degree column (T5 default filename).
    pub fn open(sub: &SubstrateFile) -> SubstrateResult<Option<Self>> {
        Self::open_by_filename(sub, DEGREE_COLUMN_FILENAME)
    }

    /// Open an existing column at a custom filename and validate
    /// magic + version + CRC. Returns `None` on :
    /// - file absent (caller should rebuild)
    /// - truncated file (header doesn't fit)
    /// - magic/version mismatch
    /// - CRC mismatch (payload tampered or crash-during-flush)
    ///
    /// The caller treats `None` as "no valid sidecar" and rebuilds via
    /// `rebuild_from_scan`. This matches the graceful-degradation
    /// contract of tier zones (never fatal at open).
    pub fn open_by_filename(
        sub: &SubstrateFile,
        filename: &str,
    ) -> SubstrateResult<Option<Self>> {
        let zf = sub.open_named_zone(filename)?;
        if zf.is_empty() {
            return Ok(None);
        }
        let slice = zf.as_slice();
        if slice.len() < HEADER_SIZE {
            return Ok(None);
        }
        let header: &DegreeHeader =
            bytemuck::from_bytes(&slice[..HEADER_SIZE]);
        if header.magic != DEGREE_MAGIC {
            return Ok(None);
        }
        if header.version != DEGREE_VERSION {
            return Ok(None);
        }
        if header.record_size != DEGREE_RECORD_SIZE as u32 {
            return Ok(None);
        }
        let payload_bytes = (header.n_slots as usize) * DEGREE_RECORD_SIZE;
        let expected_end = HEADER_SIZE + payload_bytes;
        if slice.len() < expected_end {
            return Ok(None);
        }
        // CRC validation (IEEE polynomial via crc32fast).
        let payload = &slice[HEADER_SIZE..expected_end];
        let mut h = crc32fast::Hasher::new();
        h.update(payload);
        let computed = h.finalize();
        if computed != header.crc32 {
            return Ok(None);
        }
        // Copy n_slots out before we move zf (borrow on `header` keeps
        // `zf` alive otherwise).
        let n_slots = header.n_slots;
        Ok(Some(Self { zf, n_slots }))
    }

    /// Number of slots currently covered (including slot 0 sentinel).
    pub fn n_slots(&self) -> u32 {
        self.n_slots
    }

    /// Read the out-degree for `slot`. Returns 0 for out-of-bounds slots
    /// (graceful — a node not yet covered has no recorded degree yet,
    /// same as a freshly-created node with no edges).
    pub fn out_degree(&self, slot: u32) -> u32 {
        if slot >= self.n_slots {
            return 0;
        }
        let offset = HEADER_SIZE + (slot as usize) * DEGREE_RECORD_SIZE;
        let ptr = unsafe {
            self.zf.as_slice().as_ptr().add(offset) as *const u32 as *mut u32
        };
        // SAFETY: the mmap region is live for as long as `&self`, the
        // offset is 4-byte aligned (HEADER_SIZE=64 aligned, slot*8
        // aligned), and all writes go through AtomicU32 → no data race.
        unsafe { AtomicU32::from_ptr(ptr) }.load(Ordering::Relaxed)
    }

    /// Read the in-degree for `slot`. Same out-of-bounds behavior.
    pub fn in_degree(&self, slot: u32) -> u32 {
        if slot >= self.n_slots {
            return 0;
        }
        let offset = HEADER_SIZE + (slot as usize) * DEGREE_RECORD_SIZE + 4;
        let ptr = unsafe {
            self.zf.as_slice().as_ptr().add(offset) as *const u32 as *mut u32
        };
        unsafe { AtomicU32::from_ptr(ptr) }.load(Ordering::Relaxed)
    }

    /// Atomically increment out-degree for `slot` by `delta`. Positive
    /// delta = fetch_add, negative = fetch_sub. `slot` must be in bounds
    /// — callers that may grow past `n_slots` should call `ensure_slot`
    /// first.
    pub fn incr_out(&self, slot: u32, delta: i32) {
        debug_assert!(slot < self.n_slots, "incr_out: slot {} >= n_slots {}", slot, self.n_slots);
        if slot >= self.n_slots {
            return;
        }
        let offset = HEADER_SIZE + (slot as usize) * DEGREE_RECORD_SIZE;
        let ptr = unsafe {
            self.zf.as_slice().as_ptr().add(offset) as *const u32 as *mut u32
        };
        let atomic = unsafe { AtomicU32::from_ptr(ptr) };
        if delta >= 0 {
            atomic.fetch_add(delta as u32, Ordering::Relaxed);
        } else {
            atomic.fetch_sub((-delta) as u32, Ordering::Relaxed);
        }
    }

    /// Atomically increment in-degree for `slot` by `delta`. Same as
    /// `incr_out` but for the second u32 of the record.
    pub fn incr_in(&self, slot: u32, delta: i32) {
        debug_assert!(slot < self.n_slots, "incr_in: slot {} >= n_slots {}", slot, self.n_slots);
        if slot >= self.n_slots {
            return;
        }
        let offset = HEADER_SIZE + (slot as usize) * DEGREE_RECORD_SIZE + 4;
        let ptr = unsafe {
            self.zf.as_slice().as_ptr().add(offset) as *const u32 as *mut u32
        };
        let atomic = unsafe { AtomicU32::from_ptr(ptr) };
        if delta >= 0 {
            atomic.fetch_add(delta as u32, Ordering::Relaxed);
        } else {
            atomic.fetch_sub((-delta) as u32, Ordering::Relaxed);
        }
    }

    /// Ensure `slot` is covered by the column. If `slot >= n_slots`,
    /// grow the file (doubling policy) so new slots become available.
    ///
    /// Called by the writer on `create_node` when the allocator pushes
    /// `next_node_id` past the current `n_slots`.
    pub fn ensure_slot(&mut self, slot: u32) -> SubstrateResult<()> {
        if slot < self.n_slots {
            return Ok(());
        }
        // Double the size (amortised O(1) per slot) or at minimum
        // accommodate the requested slot.
        let new_n_slots = std::cmp::max(self.n_slots.saturating_mul(2), slot + 1);
        let payload_bytes = (new_n_slots as usize) * DEGREE_RECORD_SIZE;
        let total_bytes = HEADER_SIZE + payload_bytes;
        ensure_room(&mut self.zf, total_bytes, 1 << 20)?;
        // New payload region is zero-filled by ensure_room (grow_to
        // zeroes new pages). Update header + n_slots.
        self.n_slots = new_n_slots;
        let header = DegreeHeader {
            magic: DEGREE_MAGIC,
            version: DEGREE_VERSION,
            n_slots: new_n_slots,
            record_size: DEGREE_RECORD_SIZE as u32,
            crc32: 0, // caller will recompute at persist
            flags: 0,
            _reserved: [0u32; 10],
        };
        self.zf.as_slice_mut()[..HEADER_SIZE]
            .copy_from_slice(bytemuck::bytes_of(&header));
        Ok(())
    }

    /// Compute CRC over the payload and patch the header. Call before
    /// `msync` / `fsync` for durability.
    pub fn persist_header_crc(&mut self) {
        let payload_bytes = (self.n_slots as usize) * DEGREE_RECORD_SIZE;
        let payload_end = HEADER_SIZE + payload_bytes;
        let mut h = crc32fast::Hasher::new();
        h.update(&self.zf.as_slice()[HEADER_SIZE..payload_end]);
        let crc = h.finalize();
        // Patch crc32 field in-place in the mmap'd header.
        let bytes = self.zf.as_slice_mut();
        bytes[16..20].copy_from_slice(&crc.to_le_bytes());
    }

    /// Flush dirty pages to disk (msync).
    pub fn msync(&self) -> SubstrateResult<()> {
        self.zf.msync().map_err(|e| {
            SubstrateError::Internal(format!("degree_column msync failed: {e}"))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::file::SubstrateFile;
    use std::sync::atomic::Ordering;

    fn fresh_substrate() -> (tempfile::TempDir, SubstrateFile) {
        let td = tempfile::TempDir::new().unwrap();
        let sub = SubstrateFile::create(td.path()).unwrap();
        (td, sub)
    }

    #[test]
    fn create_and_read_empty() {
        let (_td, sub) = fresh_substrate();
        let col = DegreeColumn::create(&sub, 100).unwrap();
        assert_eq!(col.n_slots(), 100);
        assert_eq!(col.out_degree(5), 0);
        assert_eq!(col.in_degree(5), 0);
        assert_eq!(col.out_degree(99), 0);
        assert_eq!(col.out_degree(150), 0); // out of bounds → 0
    }

    #[test]
    fn incr_and_read() {
        let (_td, sub) = fresh_substrate();
        let col = DegreeColumn::create(&sub, 100).unwrap();
        col.incr_out(5, 3);
        col.incr_out(5, 2);
        col.incr_in(5, 7);
        assert_eq!(col.out_degree(5), 5);
        assert_eq!(col.in_degree(5), 7);
        col.incr_out(5, -2);
        assert_eq!(col.out_degree(5), 3);
    }

    #[test]
    fn out_and_in_are_independent() {
        let (_td, sub) = fresh_substrate();
        let col = DegreeColumn::create(&sub, 10).unwrap();
        col.incr_out(3, 5);
        col.incr_in(3, 10);
        assert_eq!(col.out_degree(3), 5);
        assert_eq!(col.in_degree(3), 10);
    }

    #[test]
    fn persist_and_reopen_roundtrip() {
        let td = tempfile::TempDir::new().unwrap();
        let sub_path = td.path();
        {
            let sub = SubstrateFile::create(sub_path).unwrap();
            let mut col = DegreeColumn::create(&sub, 64).unwrap();
            col.incr_out(10, 42);
            col.incr_in(10, 17);
            col.incr_out(20, 3);
            col.persist_header_crc();
            col.msync().unwrap();
        }
        // Reopen
        let sub = SubstrateFile::open(sub_path).unwrap();
        let col = DegreeColumn::open(&sub).unwrap().expect("open Some");
        assert_eq!(col.n_slots(), 64);
        assert_eq!(col.out_degree(10), 42);
        assert_eq!(col.in_degree(10), 17);
        assert_eq!(col.out_degree(20), 3);
    }

    #[test]
    fn open_absent_returns_none() {
        let (_td, sub) = fresh_substrate();
        // Never created → file absent / empty zone.
        let col = DegreeColumn::open(&sub).unwrap();
        assert!(col.is_none());
    }

    #[test]
    fn open_crc_mismatch_returns_none() {
        let td = tempfile::TempDir::new().unwrap();
        let sub_path = td.path();
        {
            let sub = SubstrateFile::create(sub_path).unwrap();
            let mut col = DegreeColumn::create(&sub, 32).unwrap();
            col.incr_out(5, 10);
            col.persist_header_crc();
            col.msync().unwrap();
        }
        // Tamper a payload byte (invalidate CRC).
        let path = sub_path.join(DEGREE_COLUMN_FILENAME);
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[HEADER_SIZE + 10] ^= 0xFF;
        std::fs::write(&path, bytes).unwrap();
        // Reopen returns None.
        let sub = SubstrateFile::open(sub_path).unwrap();
        let col = DegreeColumn::open(&sub).unwrap();
        assert!(col.is_none());
    }

    #[test]
    fn ensure_slot_grows_column() {
        let (_td, sub) = fresh_substrate();
        let mut col = DegreeColumn::create(&sub, 10).unwrap();
        col.incr_out(5, 7);
        col.ensure_slot(100).unwrap();
        assert!(col.n_slots() > 100);
        // Old slot preserved.
        assert_eq!(col.out_degree(5), 7);
        // New slot works.
        col.incr_out(99, 13);
        assert_eq!(col.out_degree(99), 13);
    }
}
