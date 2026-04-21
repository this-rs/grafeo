//! # Blob columns — per-prop-key variable-length mmap storage
//!
//! **T16.7 Step 4 — foundation.** Implements the on-disk format,
//! writer and reader for "blob columns": one pair of mmap'd zones per
//! variable-length property key, generalising the dense
//! [`crate::vec_column`] pattern to payloads that do not all share the
//! same width.
//!
//! ## Why
//!
//! Two families of properties dominated `substrate.props` anon-RSS on
//! PO after T16.7 Step 3 landed vector routing:
//!
//! | Key        | Total  | Count  | Avg B | Kind        |
//! |-----------:|-------:|-------:|------:|-------------|
//! | `data`     | 928 MB | 681 k  | 1361  | `Value::String` (chat payload blobs) |
//! | `file_path`| 29 MB  | 295 k  | 97    | `Value::String` (< threshold, scalar) |
//!
//! Vectors could be routed because every vector for a given key has
//! the same `(dim, dtype)` — a single dense mmap is enough. Blobs are
//! variable-length, so they need an explicit `(offset, len)` per slot.
//!
//! ## On-disk layout (v1)
//!
//! Two files per column:
//!
//! ```text
//! substrate.blobcol.<kind>.<key_id_hex04>.idx
//!   ┌──────────────────────────────────────────────────────────────┐
//!   │ [64 B  BlobColumnHeader]    magic "BLB1", version, spec,     │
//!   │                             n_slots, arena_bytes, both CRCs  │
//!   │ [ n_slots × 16 B SlotEntry ] arena_offset u64, len u32, pad  │
//!   └──────────────────────────────────────────────────────────────┘
//!
//! substrate.blobcol.<kind>.<key_id_hex04>.dat
//!   ┌──────────────────────────────────────────────────────────────┐
//!   │ [ arena_bytes × u8 ]       concatenated raw payloads,        │
//!   │                            not padded, not aligned           │
//!   └──────────────────────────────────────────────────────────────┘
//! ```
//!
//! Two files rather than one because the two regions grow on
//! different triggers: the index grows with `n_slots` (step up in
//! multiples of 16 B per newly-seen slot), the arena grows with
//! payload bytes (variable increments). Keeping them separate means
//! each file is `ensure_room`'d independently and no region-shift
//! dance is needed at grow time.
//!
//! **Absence semantics.** A `SlotEntry` with `len == 0` means
//! "absent" — indistinguishable from the zero-initialised bytes a
//! freshly-grown file carries. Empty blobs are rejected on write, the
//! same way empty vectors are rejected in
//! [`crate::vec_column::VecColumnWriter::write_slot`]. Callers who
//! need to represent "present but empty" should use [`Value::Null`]
//! or remove the key.
//!
//! ## Crash safety
//!
//! Same model as [`crate::vec_column`] and [`crate::tier_persist`]:
//!
//!   1. `ensure_room` grows both files.
//!   2. Payload bytes are memcpy'd into the arena; slot entry is
//!      written into the idx.
//!   3. CRC32 is computed over both live regions.
//!   4. Header is written into the idx file last, with both CRCs +
//!      the final `n_slots` + `arena_bytes`.
//!   5. `msync` + `fsync` make both files durable.
//!
//! A crash between steps 2 and 4 leaves a header with a stale
//! `(n_slots, arena_bytes, CRCs)` — the reader rejects it and the
//! column degrades to "absent", same as the vec-column contract.
//!
//! ## Scope of Step 4a
//!
//! This module delivers the **format** and the **I/O primitives**.
//! It is not yet called from `SubstrateStore`: routing of oversized
//! `Value::String` / `Value::Bytes` is Step 4d, and `obrain-migrate`
//! phase_nodes / phase_edges wiring is Step 4e. The tests here
//! exercise the module end-to-end in isolation via a temporary
//! substrate directory.

use bytemuck::{Pod, Zeroable};

use crate::error::{SubstrateError, SubstrateResult};
use crate::file::{SubstrateFile, ZoneFile};
use crate::vec_column::EntityKind;
use crate::writer::ensure_room;

// ---------------------------------------------------------------------------
// Format constants
// ---------------------------------------------------------------------------

/// Little-endian `u32` of the ASCII bytes `b"BLB1"`. Used as the
/// magic word in [`BlobColumnHeader::magic`]; chosen so `hexdump -C`
/// shows a human-readable identifier as the first 4 bytes of every
/// blob-column idx file.
pub const BLOB_COLUMN_MAGIC: u32 = u32::from_le_bytes(*b"BLB1");

/// Current on-disk format version. Bump together with the header
/// layout, the slot layout, or the arena encoding.
pub const BLOB_COLUMN_VERSION: u32 = 1;

/// Fixed header size in bytes. Same convention as [`crate::vec_column`]:
/// 64 B gives room for future non-mandatory fields without a version
/// bump (via the trailing `_reserved` block).
pub const BLOB_HEADER_SIZE: usize = 64;

/// Wire size of one [`BlobSlotEntry`].
pub const BLOB_SLOT_STRIDE: usize = 16;

// ---------------------------------------------------------------------------
// Spec + filenames
// ---------------------------------------------------------------------------

/// Descriptor of one blob column. The two fields are the primary key
/// of the on-disk zone pair: different prop-keys or different entity
/// kinds all land in **distinct file pairs**.
///
/// `prop_key_id` is the interned id assigned by the store's
/// `PropKeyRegistry` — stable across reopens because the registry
/// itself is persisted in `substrate.dict`. Unlike
/// [`crate::vec_column::VecColSpec`], there is no `dim` / `dtype`
/// field: blobs are variable-length opaque bytes.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct BlobColSpec {
    pub prop_key_id: u16,
    pub entity_kind: EntityKind,
}

impl BlobColSpec {
    /// Filename of the slot-index file for this column.
    pub fn idx_filename(&self) -> String {
        format!(
            "substrate.blobcol.{}.{:04x}.idx",
            self.entity_kind.as_suffix(),
            self.prop_key_id
        )
    }

    /// Filename of the byte-arena file for this column.
    pub fn dat_filename(&self) -> String {
        format!(
            "substrate.blobcol.{}.{:04x}.dat",
            self.entity_kind.as_suffix(),
            self.prop_key_id
        )
    }
}

// ---------------------------------------------------------------------------
// On-disk types
// ---------------------------------------------------------------------------

/// 64 B `#[repr(C)]` header written at the start of every blob-column
/// idx file. All fields are little-endian.
///
/// Field order is frozen by [`BLOB_COLUMN_VERSION`]; the `_reserved`
/// tail gives 32 bytes for future extensions without reshuffling
/// offsets.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct BlobColumnHeader {
    /// [`BLOB_COLUMN_MAGIC`] — b"BLB1" little-endian.
    pub magic: u32,
    /// [`BLOB_COLUMN_VERSION`] at write time.
    pub version: u32,
    /// Matches [`BlobColSpec::prop_key_id`].
    pub prop_key_id: u16,
    /// 0 for node, 1 for edge. See [`EntityKind`].
    pub entity_kind: u8,
    /// Zero-filled padding to align `n_slots` on 4 B.
    pub _pad0: u8,
    /// Number of allocated slots. Slot `i` lives at
    /// `BLOB_HEADER_SIZE + i * BLOB_SLOT_STRIDE` in the idx file.
    pub n_slots: u32,
    /// Total bytes written into the arena file. The live payload
    /// occupies `[0, arena_bytes)` of the dat file.
    pub arena_bytes: u64,
    /// CRC32 (IEEE polynomial) over the idx slot region —
    /// `idx[BLOB_HEADER_SIZE .. BLOB_HEADER_SIZE + n_slots * 16]`.
    pub idx_crc32: u32,
    /// CRC32 (IEEE polynomial) over the live arena region —
    /// `dat[0 .. arena_bytes]`.
    pub arena_crc32: u32,
    /// Zero-filled padding to bring the header to 64 B.
    pub _reserved: [u32; 8],
}

// Compile-time sanity: the header is exactly 64 B.
const _: [(); 1] =
    [(); (core::mem::size_of::<BlobColumnHeader>() == BLOB_HEADER_SIZE) as usize];

impl BlobColumnHeader {
    fn new(spec: &BlobColSpec, n_slots: u32, arena_bytes: u64) -> Self {
        Self {
            magic: BLOB_COLUMN_MAGIC,
            version: BLOB_COLUMN_VERSION,
            prop_key_id: spec.prop_key_id,
            entity_kind: spec.entity_kind as u8,
            _pad0: 0,
            n_slots,
            arena_bytes,
            idx_crc32: 0,
            arena_crc32: 0,
            _reserved: [0u32; 8],
        }
    }

    /// Parse a header out of the first [`BLOB_HEADER_SIZE`] bytes of
    /// a mmap region. Returns `None` if the region is too short or
    /// fails alignment; never panics.
    fn try_read(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < BLOB_HEADER_SIZE {
            return None;
        }
        bytemuck::try_pod_read_unaligned::<BlobColumnHeader>(&bytes[..BLOB_HEADER_SIZE]).ok()
    }

    /// Validate the header matches the expected spec. Any mismatch
    /// is reported as a descriptive `&'static str`; the reader turns
    /// this into `Ok(None)` downstream.
    fn validate_against(&self, spec: &BlobColSpec) -> Result<(), &'static str> {
        if self.magic != BLOB_COLUMN_MAGIC {
            return Err("magic mismatch");
        }
        if self.version != BLOB_COLUMN_VERSION {
            return Err("version mismatch");
        }
        if self.prop_key_id != spec.prop_key_id {
            return Err("prop_key_id mismatch");
        }
        if self.entity_kind != spec.entity_kind as u8 {
            return Err("entity_kind mismatch");
        }
        Ok(())
    }
}

/// 16 B `#[repr(C)]` slot-directory entry written at
/// `BLOB_HEADER_SIZE + slot * 16` in the idx file.
///
/// `len == 0` means "absent" — the slot has never been written. The
/// writer rejects zero-length payloads so this marker is unambiguous.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct BlobSlotEntry {
    /// Offset into the dat file where this slot's payload begins.
    pub arena_offset: u64,
    /// Payload length in bytes. Zero = absent.
    pub len: u32,
    /// Zero-filled padding to bring the entry to 16 B.
    pub _pad: u32,
}

const _: [(); 1] =
    [(); (core::mem::size_of::<BlobSlotEntry>() == BLOB_SLOT_STRIDE) as usize];

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

/// Variable-length writer for one blob column.
///
/// The writer keeps handles to both files (`.idx` and `.dat`). Each
/// `write_slot` appends the payload to the arena, updates the
/// matching slot entry in the idx, and bumps the high-water marks
/// (`n_slots`, `arena_end`). Durability is batched into [`Self::sync`]
/// (called from `SubstrateStore::flush`).
///
/// ## Usage pattern
///
/// ```ignore
/// let spec = BlobColSpec { prop_key_id: 7, entity_kind: EntityKind::Node };
/// let mut w = BlobColumnWriter::create(&sub, spec)?;
/// w.write_slot(42, b"hello world")?;
/// w.sync()?;
/// ```
///
/// Writes are **not** transactional at the entry level — a crash
/// between `write_slot` and the next `sync` leaves the on-disk
/// `n_slots` / `arena_bytes` counters behind the actual written
/// extent, which the reader will either truncate to or reject
/// depending on CRC state. In the worst case the last few writes
/// since the last sync are lost; no corruption of earlier data.
#[derive(Debug)]
pub struct BlobColumnWriter {
    spec: BlobColSpec,
    idx: ZoneFile,
    dat: ZoneFile,
    /// Highest slot index ever written +1. Monotonic.
    n_slots: u32,
    /// Bytes written into the arena so far. Monotonic.
    arena_end: u64,
}

impl BlobColumnWriter {
    /// Create (or re-open) a blob-column pair for the given spec.
    ///
    /// If the column already exists on disk, the writer inherits its
    /// current `n_slots` / `arena_bytes` from the idx header so
    /// subsequent `write_slot`s past that point grow the column. An
    /// unreadable / mismatching idx header is treated as a fresh
    /// start — the old contents are overwritten.
    pub fn create(sub: &SubstrateFile, spec: BlobColSpec) -> SubstrateResult<Self> {
        let idx = sub.open_named_zone(&spec.idx_filename())?;
        let dat = sub.open_named_zone(&spec.dat_filename())?;
        let (n_slots, arena_end) = match BlobColumnHeader::try_read(idx.as_slice()) {
            Some(h) if h.validate_against(&spec).is_ok() => (h.n_slots, h.arena_bytes),
            _ => (0, 0),
        };
        Ok(Self {
            spec,
            idx,
            dat,
            n_slots,
            arena_end,
        })
    }

    /// Append `bytes` as a new payload and record the `(offset, len)`
    /// in slot `slot`. Overwriting an existing slot does NOT reclaim
    /// the old arena bytes — they are left as garbage. This matches
    /// the append-only discipline of log-structured stores.
    ///
    /// Rejects empty payloads: `len == 0` is the absence marker in
    /// the on-disk slot directory.
    pub fn write_slot(&mut self, slot: u32, bytes: &[u8]) -> SubstrateResult<()> {
        if bytes.is_empty() {
            return Err(SubstrateError::WalBadFrame(
                "blob_column: cannot store empty payload (len=0 is the absence marker)"
                    .into(),
            ));
        }
        if bytes.len() > u32::MAX as usize {
            return Err(SubstrateError::WalBadFrame(format!(
                "blob_column: payload length {} exceeds u32::MAX",
                bytes.len()
            )));
        }

        let offset = self.arena_end;
        let len = bytes.len() as u32;

        // (1) Append to arena.
        let arena_needed_end = (offset as usize) + (len as usize);
        ensure_room(&mut self.dat, arena_needed_end, 0)?;
        {
            let region = self.dat.as_slice_mut();
            region[offset as usize..arena_needed_end].copy_from_slice(bytes);
        }
        self.arena_end = offset + (len as u64);

        // (2) Write slot entry into idx.
        let slot_offset = BLOB_HEADER_SIZE + (slot as usize) * BLOB_SLOT_STRIDE;
        let slot_end = slot_offset + BLOB_SLOT_STRIDE;
        ensure_room(&mut self.idx, slot_end, 0)?;
        let entry = BlobSlotEntry {
            arena_offset: offset,
            len,
            _pad: 0,
        };
        {
            let region = self.idx.as_slice_mut();
            region[slot_offset..slot_end].copy_from_slice(bytemuck::bytes_of(&entry));
        }

        // (3) Bump n_slots HWM.
        if slot + 1 > self.n_slots {
            self.n_slots = slot + 1;
        }
        Ok(())
    }

    /// Read slot `slot` through the writer's own mmap. Returns `None`
    /// if the slot is past the current high-water mark or marked
    /// absent (`len == 0`).
    ///
    /// Mirrors [`BlobColumnReader::read_slot`] but serves reads
    /// through the same mmap that writes go through, so the T16.7
    /// routing path does not need to maintain a separate reader
    /// handle per spec. Between `write_slot` and the next `sync` the
    /// on-disk header still carries stale counters + CRCs, so a
    /// fresh [`BlobColumnReader`] would refuse to open — but the
    /// payload bytes are live in the mmap and safe to re-read here.
    pub fn read_slot(&self, slot: u32) -> Option<&[u8]> {
        if slot >= self.n_slots {
            return None;
        }
        let slot_offset = BLOB_HEADER_SIZE + (slot as usize) * BLOB_SLOT_STRIDE;
        let slot_end = slot_offset + BLOB_SLOT_STRIDE;
        let idx_slice = self.idx.as_slice();
        if slot_end > idx_slice.len() {
            return None;
        }
        let entry = bytemuck::try_pod_read_unaligned::<BlobSlotEntry>(
            &idx_slice[slot_offset..slot_end],
        )
        .ok()?;
        if entry.len == 0 {
            return None;
        }
        let begin = entry.arena_offset as usize;
        let end = begin + entry.len as usize;
        let dat_slice = self.dat.as_slice();
        if end > dat_slice.len() {
            return None;
        }
        Some(&dat_slice[begin..end])
    }

    /// Force-finalise current state: compute both CRCs, write header
    /// into the idx file, `msync` + `fsync` both files. The writer
    /// stays alive so subsequent `write_slot`s can extend the column.
    /// Called from `SubstrateStore::flush`.
    pub fn sync(&mut self) -> SubstrateResult<()> {
        // Ensure idx file has room for the header + all slots.
        let idx_end =
            BLOB_HEADER_SIZE + (self.n_slots as usize) * BLOB_SLOT_STRIDE;
        ensure_room(&mut self.idx, idx_end, 0)?;

        // Ensure dat file has room for the live arena.
        ensure_room(&mut self.dat, self.arena_end as usize, 0)?;

        // CRC over slot region.
        let idx_crc = {
            let slice = self.idx.as_slice();
            let payload = &slice[BLOB_HEADER_SIZE..idx_end];
            let mut h = crc32fast::Hasher::new();
            h.update(payload);
            h.finalize()
        };

        // CRC over arena region.
        let arena_crc = {
            let slice = self.dat.as_slice();
            let payload = &slice[..self.arena_end as usize];
            let mut h = crc32fast::Hasher::new();
            h.update(payload);
            h.finalize()
        };

        let mut header =
            BlobColumnHeader::new(&self.spec, self.n_slots, self.arena_end);
        header.idx_crc32 = idx_crc;
        header.arena_crc32 = arena_crc;
        {
            let slice = self.idx.as_slice_mut();
            slice[..BLOB_HEADER_SIZE].copy_from_slice(bytemuck::bytes_of(&header));
        }

        self.idx.msync()?;
        self.idx.fsync()?;
        self.dat.msync()?;
        self.dat.fsync()?;
        Ok(())
    }

    /// Drop-consuming variant of [`Self::sync`]. Included for
    /// symmetry with [`crate::vec_column::VecColumnWriter::finalize`].
    pub fn finalize(mut self) -> SubstrateResult<()> {
        self.sync()
    }

    /// Current high-water mark (number of slots written so far).
    pub fn n_slots(&self) -> u32 {
        self.n_slots
    }

    /// Current high-water mark of the arena (bytes written so far).
    pub fn arena_end(&self) -> u64 {
        self.arena_end
    }

    /// The spec this writer targets.
    pub fn spec(&self) -> &BlobColSpec {
        &self.spec
    }
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

/// Mmap-backed reader for one blob column.
///
/// Opening the reader validates:
///
/// 1. idx header magic / version / spec match,
/// 2. idx file is long enough to cover `n_slots * 16 B` past the header,
/// 3. dat file is long enough to cover `arena_bytes`,
/// 4. CRC32 over both regions matches the stored CRCs.
///
/// On any failure, `open` returns `Ok(None)` and the caller is
/// expected to fall back to "absent column" semantics. No
/// half-readable columns are surfaced.
///
/// [`Self::read_slot`] returns a zero-copy `&[u8]` slice into the
/// dat mmap, valid for the lifetime of the reader handle.
pub struct BlobColumnReader {
    spec: BlobColSpec,
    idx: ZoneFile,
    dat: ZoneFile,
    header: BlobColumnHeader,
}

impl BlobColumnReader {
    /// Open the blob-column pair matching `spec`. Returns:
    ///
    /// * `Ok(Some(reader))` on success,
    /// * `Ok(None)` if the zones do not exist, have a mismatching
    ///   header, or fail any CRC check — the caller is expected to
    ///   rebuild or fall back to an empty column,
    /// * `Err(_)` only on underlying I/O errors from the mmap
    ///   subsystem (i.e. "we couldn't even read the bytes").
    pub fn open(sub: &SubstrateFile, spec: BlobColSpec) -> SubstrateResult<Option<Self>> {
        let idx = sub.open_named_zone(&spec.idx_filename())?;
        let dat = sub.open_named_zone(&spec.dat_filename())?;

        // Empty idx file — fresh column.
        if idx.is_empty() {
            return Ok(None);
        }

        let idx_slice = idx.as_slice();
        let Some(header) = BlobColumnHeader::try_read(idx_slice) else {
            return Ok(None);
        };
        if header.validate_against(&spec).is_err() {
            return Ok(None);
        }

        // Length checks.
        let idx_end =
            BLOB_HEADER_SIZE + (header.n_slots as usize) * BLOB_SLOT_STRIDE;
        if idx_slice.len() < idx_end {
            return Ok(None);
        }
        let dat_slice = dat.as_slice();
        if (dat_slice.len() as u64) < header.arena_bytes {
            return Ok(None);
        }

        // CRC checks.
        let idx_crc = {
            let payload = &idx_slice[BLOB_HEADER_SIZE..idx_end];
            let mut h = crc32fast::Hasher::new();
            h.update(payload);
            h.finalize()
        };
        if idx_crc != header.idx_crc32 {
            return Ok(None);
        }
        let arena_crc = {
            let payload = &dat_slice[..header.arena_bytes as usize];
            let mut h = crc32fast::Hasher::new();
            h.update(payload);
            h.finalize()
        };
        if arena_crc != header.arena_crc32 {
            return Ok(None);
        }

        Ok(Some(Self {
            spec,
            idx,
            dat,
            header,
        }))
    }

    /// Number of slots currently stored.
    pub fn n_slots(&self) -> u32 {
        self.header.n_slots
    }

    /// Bytes written into the arena.
    pub fn arena_bytes(&self) -> u64 {
        self.header.arena_bytes
    }

    /// The spec this reader targets.
    pub fn spec(&self) -> &BlobColSpec {
        &self.spec
    }

    /// Borrow the raw bytes for one slot, or `None` if the slot is
    /// out of range or marked absent.
    ///
    /// The returned slice aliases the mmap directly — cheap in both
    /// latency (no copy) and memory (no anon allocation).
    pub fn read_slot(&self, slot: u32) -> Option<&[u8]> {
        if slot >= self.header.n_slots {
            return None;
        }
        let slot_offset = BLOB_HEADER_SIZE + (slot as usize) * BLOB_SLOT_STRIDE;
        let slot_end = slot_offset + BLOB_SLOT_STRIDE;
        let idx_slice = self.idx.as_slice();
        if slot_end > idx_slice.len() {
            return None;
        }
        let entry = bytemuck::try_pod_read_unaligned::<BlobSlotEntry>(
            &idx_slice[slot_offset..slot_end],
        )
        .ok()?;
        if entry.len == 0 {
            return None;
        }
        let begin = entry.arena_offset as usize;
        let end = begin + entry.len as usize;
        let dat_slice = self.dat.as_slice();
        if end > dat_slice.len() {
            return None;
        }
        Some(&dat_slice[begin..end])
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_spec(prop_key_id: u16) -> BlobColSpec {
        BlobColSpec {
            prop_key_id,
            entity_kind: EntityKind::Node,
        }
    }

    #[test]
    fn header_size_matches_constant() {
        assert_eq!(core::mem::size_of::<BlobColumnHeader>(), BLOB_HEADER_SIZE);
    }

    #[test]
    fn slot_stride_matches_constant() {
        assert_eq!(core::mem::size_of::<BlobSlotEntry>(), BLOB_SLOT_STRIDE);
    }

    #[test]
    fn magic_bytes_are_ascii_blb1() {
        assert_eq!(BLOB_COLUMN_MAGIC.to_le_bytes(), *b"BLB1");
    }

    #[test]
    fn spec_filename_is_deterministic_and_descriptive() {
        let s = BlobColSpec {
            prop_key_id: 0x2A,
            entity_kind: EntityKind::Node,
        };
        assert_eq!(s.idx_filename(), "substrate.blobcol.node.002a.idx");
        assert_eq!(s.dat_filename(), "substrate.blobcol.node.002a.dat");

        let e = BlobColSpec {
            prop_key_id: 7,
            entity_kind: EntityKind::Edge,
        };
        assert_eq!(e.idx_filename(), "substrate.blobcol.edge.0007.idx");
        assert_eq!(e.dat_filename(), "substrate.blobcol.edge.0007.dat");
    }

    #[test]
    fn empty_zones_return_none_on_open() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let spec = sample_spec(1);
        let reader = BlobColumnReader::open(&sub, spec).unwrap();
        assert!(reader.is_none(), "fresh substrate has no blob column");
    }

    #[test]
    fn roundtrip_single_slot() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let spec = sample_spec(1);
        let payload = b"hello, substrate blob column!";

        let mut w = BlobColumnWriter::create(&sub, spec).unwrap();
        w.write_slot(0, payload).unwrap();
        assert_eq!(w.n_slots(), 1);
        assert_eq!(w.arena_end() as usize, payload.len());
        // Writer-served read works before sync.
        assert_eq!(w.read_slot(0).unwrap(), payload);
        assert!(w.read_slot(1).is_none());
        w.sync().unwrap();

        let r = BlobColumnReader::open(&sub, spec).unwrap().unwrap();
        assert_eq!(r.n_slots(), 1);
        assert_eq!(r.arena_bytes() as usize, payload.len());
        assert_eq!(r.read_slot(0).unwrap(), payload);
        assert!(r.read_slot(1).is_none());
    }

    #[test]
    fn roundtrip_many_slots_variable_length() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let spec = sample_spec(7);
        let n = 50u32;

        let mut w = BlobColumnWriter::create(&sub, spec).unwrap();
        let mut expected: Vec<Vec<u8>> = Vec::with_capacity(n as usize);
        for slot in 0..n {
            let payload: Vec<u8> =
                (0..(slot as u8 + 1) * 3).map(|i| i.wrapping_add(slot as u8)).collect();
            expected.push(payload.clone());
            w.write_slot(slot, &payload).unwrap();
        }
        assert_eq!(w.n_slots(), n);
        w.sync().unwrap();

        let r = BlobColumnReader::open(&sub, spec).unwrap().unwrap();
        assert_eq!(r.n_slots(), n);
        for slot in 0..n {
            assert_eq!(r.read_slot(slot).unwrap(), &expected[slot as usize][..]);
        }
    }

    #[test]
    fn sparse_writes_have_absent_gaps() {
        // Write slot 0 and slot 10; slots 1..10 should be reported
        // as absent by the reader (len=0 SlotEntry from zero-init).
        let sub = SubstrateFile::open_tempfile().unwrap();
        let spec = sample_spec(3);
        let p0 = b"first payload";
        let p10 = b"tenth payload, different length";

        let mut w = BlobColumnWriter::create(&sub, spec).unwrap();
        w.write_slot(0, p0).unwrap();
        w.write_slot(10, p10).unwrap();
        assert_eq!(w.n_slots(), 11, "n_slots tracks highest+1");
        w.sync().unwrap();

        let r = BlobColumnReader::open(&sub, spec).unwrap().unwrap();
        assert_eq!(r.n_slots(), 11);
        assert_eq!(r.read_slot(0).unwrap(), p0);
        assert_eq!(r.read_slot(10).unwrap(), p10);
        for mid in 1..10 {
            assert!(
                r.read_slot(mid).is_none(),
                "slot {mid} must read as absent, got {:?}",
                r.read_slot(mid)
            );
        }
    }

    #[test]
    fn empty_payload_is_rejected() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let spec = sample_spec(1);
        let mut w = BlobColumnWriter::create(&sub, spec).unwrap();
        let err = w.write_slot(0, b"").unwrap_err();
        let msg = format!("{err:?}");
        assert!(msg.contains("empty"), "unexpected err: {msg}");
    }

    #[test]
    fn writer_reopen_inherits_counters() {
        // Create, write slots 0..5, sync. Reopen a new writer —
        // it must observe n_slots=5 and the arena_end so appending
        // to slot 5 doesn't clobber the live arena.
        let sub = SubstrateFile::open_tempfile().unwrap();
        let spec = sample_spec(1);

        let mut w = BlobColumnWriter::create(&sub, spec).unwrap();
        for slot in 0..5 {
            let p = format!("payload-{slot}").into_bytes();
            w.write_slot(slot, &p).unwrap();
        }
        let n1 = w.n_slots();
        let arena1 = w.arena_end();
        w.sync().unwrap();
        drop(w);

        let mut w2 = BlobColumnWriter::create(&sub, spec).unwrap();
        assert_eq!(w2.n_slots(), n1, "reopen inherits prior n_slots");
        assert_eq!(w2.arena_end(), arena1, "reopen inherits prior arena_end");

        w2.write_slot(5, b"new entry").unwrap();
        assert_eq!(w2.n_slots(), 6);
        assert_eq!(w2.arena_end(), arena1 + b"new entry".len() as u64);
        w2.sync().unwrap();

        let r = BlobColumnReader::open(&sub, spec).unwrap().unwrap();
        assert_eq!(r.n_slots(), 6);
        for slot in 0..5 {
            let expected = format!("payload-{slot}");
            assert_eq!(r.read_slot(slot).unwrap(), expected.as_bytes());
        }
        assert_eq!(r.read_slot(5).unwrap(), b"new entry");
    }

    #[test]
    fn corrupt_idx_crc_returns_none_on_open() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let spec = sample_spec(1);
        let mut w = BlobColumnWriter::create(&sub, spec).unwrap();
        w.write_slot(0, b"payload").unwrap();
        w.sync().unwrap();

        // Flip a byte inside the slot region of the idx file.
        {
            let mut zf = sub.open_named_zone(&spec.idx_filename()).unwrap();
            zf.as_slice_mut()[BLOB_HEADER_SIZE + 3] ^= 0xFF;
            zf.msync().unwrap();
        }

        let r = BlobColumnReader::open(&sub, spec).unwrap();
        assert!(r.is_none(), "corrupt idx CRC must degrade to None");
    }

    #[test]
    fn corrupt_arena_crc_returns_none_on_open() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let spec = sample_spec(1);
        let mut w = BlobColumnWriter::create(&sub, spec).unwrap();
        w.write_slot(0, b"payload-to-corrupt").unwrap();
        w.sync().unwrap();

        // Flip a byte inside the live arena region of the dat file.
        {
            let mut zf = sub.open_named_zone(&spec.dat_filename()).unwrap();
            zf.as_slice_mut()[3] ^= 0xFF;
            zf.msync().unwrap();
        }

        let r = BlobColumnReader::open(&sub, spec).unwrap();
        assert!(r.is_none(), "corrupt arena CRC must degrade to None");
    }

    #[test]
    fn corrupt_magic_returns_none_on_open() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let spec = sample_spec(1);
        let mut w = BlobColumnWriter::create(&sub, spec).unwrap();
        w.write_slot(0, b"X").unwrap();
        w.sync().unwrap();

        {
            let mut zf = sub.open_named_zone(&spec.idx_filename()).unwrap();
            zf.as_slice_mut()[0] ^= 0xFF;
            zf.msync().unwrap();
        }

        assert!(BlobColumnReader::open(&sub, spec).unwrap().is_none());
    }

    #[test]
    fn bumped_version_returns_none_on_open() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let spec = sample_spec(1);
        let mut w = BlobColumnWriter::create(&sub, spec).unwrap();
        w.write_slot(0, b"X").unwrap();
        w.sync().unwrap();

        // Bump the version field (u32 at offset 4) to a value the
        // reader does not know.
        {
            let mut zf = sub.open_named_zone(&spec.idx_filename()).unwrap();
            let bad: u32 = BLOB_COLUMN_VERSION + 99;
            zf.as_slice_mut()[4..8].copy_from_slice(&bad.to_le_bytes());
            zf.msync().unwrap();
        }

        assert!(BlobColumnReader::open(&sub, spec).unwrap().is_none());
    }

    #[test]
    fn truncated_idx_returns_none_on_open() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let spec = sample_spec(1);
        let n = 10u32;
        let mut w = BlobColumnWriter::create(&sub, spec).unwrap();
        for slot in 0..n {
            w.write_slot(slot, format!("p{slot}").as_bytes()).unwrap();
        }
        w.sync().unwrap();

        // Truncate the idx file below its expected length.
        let expected = BLOB_HEADER_SIZE + (n as usize) * BLOB_SLOT_STRIDE;
        let path = sub.path().join(spec.idx_filename());
        let f = std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap();
        f.set_len(expected as u64 - 4).unwrap();
        drop(f);

        assert!(BlobColumnReader::open(&sub, spec).unwrap().is_none());
    }

    #[test]
    fn truncated_dat_returns_none_on_open() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let spec = sample_spec(1);
        let mut w = BlobColumnWriter::create(&sub, spec).unwrap();
        w.write_slot(0, b"a-fairly-long-payload-to-truncate").unwrap();
        w.sync().unwrap();

        // Truncate the dat file below arena_bytes.
        let path = sub.path().join(spec.dat_filename());
        let f = std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap();
        f.set_len(5).unwrap();
        drop(f);

        assert!(BlobColumnReader::open(&sub, spec).unwrap().is_none());
    }

    #[test]
    fn node_and_edge_same_key_use_distinct_files() {
        // Same prop_key_id but different entity_kind → distinct
        // files, no collision. Verifies the invariant that lets the
        // store route node-properties and edge-properties
        // independently.
        let sub = SubstrateFile::open_tempfile().unwrap();
        let node_spec = BlobColSpec {
            prop_key_id: 5,
            entity_kind: EntityKind::Node,
        };
        let edge_spec = BlobColSpec {
            prop_key_id: 5,
            entity_kind: EntityKind::Edge,
        };
        assert_ne!(node_spec.idx_filename(), edge_spec.idx_filename());
        assert_ne!(node_spec.dat_filename(), edge_spec.dat_filename());

        let p_node = b"node payload";
        let p_edge = b"edge payload";

        let mut wn = BlobColumnWriter::create(&sub, node_spec).unwrap();
        wn.write_slot(0, p_node).unwrap();
        wn.sync().unwrap();
        let mut we = BlobColumnWriter::create(&sub, edge_spec).unwrap();
        we.write_slot(0, p_edge).unwrap();
        we.sync().unwrap();

        let rn = BlobColumnReader::open(&sub, node_spec).unwrap().unwrap();
        let re = BlobColumnReader::open(&sub, edge_spec).unwrap().unwrap();
        assert_eq!(rn.read_slot(0).unwrap(), p_node);
        assert_eq!(re.read_slot(0).unwrap(), p_edge);
    }

    #[test]
    fn overwrite_appends_new_bytes_and_updates_slot() {
        // Blob columns are append-only — overwriting a slot appends
        // the new payload to the arena and rewrites the slot entry
        // to point at it. The old bytes are left as garbage in the
        // arena but are no longer reachable through the slot entry.
        let sub = SubstrateFile::open_tempfile().unwrap();
        let spec = sample_spec(1);
        let mut w = BlobColumnWriter::create(&sub, spec).unwrap();

        let p_old = b"the original payload";
        let p_new = b"a completely different replacement payload of a different length";
        w.write_slot(0, p_old).unwrap();
        let arena_after_old = w.arena_end();
        w.write_slot(0, p_new).unwrap();
        // Arena must have grown by len(new), not replaced.
        assert_eq!(w.arena_end(), arena_after_old + p_new.len() as u64);
        w.sync().unwrap();

        let r = BlobColumnReader::open(&sub, spec).unwrap().unwrap();
        assert_eq!(r.read_slot(0).unwrap(), p_new);
    }

    #[test]
    fn writer_sync_then_read_through_fresh_reader() {
        // After sync(), a fresh reader must see everything written
        // so far. Subsequent writes through the same writer must
        // also become visible after another sync.
        let sub = SubstrateFile::open_tempfile().unwrap();
        let spec = sample_spec(2);
        let mut w = BlobColumnWriter::create(&sub, spec).unwrap();

        w.write_slot(0, b"alpha").unwrap();
        w.sync().unwrap();
        {
            let r = BlobColumnReader::open(&sub, spec).unwrap().unwrap();
            assert_eq!(r.read_slot(0).unwrap(), b"alpha");
        }

        w.write_slot(1, b"beta").unwrap();
        w.sync().unwrap();
        {
            let r = BlobColumnReader::open(&sub, spec).unwrap().unwrap();
            assert_eq!(r.n_slots(), 2);
            assert_eq!(r.read_slot(0).unwrap(), b"alpha");
            assert_eq!(r.read_slot(1).unwrap(), b"beta");
        }
    }

    #[test]
    fn finalize_consumes_writer_and_syncs() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let spec = sample_spec(1);
        let mut w = BlobColumnWriter::create(&sub, spec).unwrap();
        w.write_slot(0, b"durable-on-drop").unwrap();
        w.finalize().unwrap();

        let r = BlobColumnReader::open(&sub, spec).unwrap().unwrap();
        assert_eq!(r.read_slot(0).unwrap(), b"durable-on-drop");
    }

    #[test]
    fn large_payload_roundtrips() {
        // ~1 MiB payload, verifying the ensure_room growth policy
        // handles arenas that blow past the initial allocation.
        let sub = SubstrateFile::open_tempfile().unwrap();
        let spec = sample_spec(9);
        let mut w = BlobColumnWriter::create(&sub, spec).unwrap();

        let payload: Vec<u8> = (0..(1usize << 20)).map(|i| (i & 0xFF) as u8).collect();
        w.write_slot(0, &payload).unwrap();
        w.sync().unwrap();

        let r = BlobColumnReader::open(&sub, spec).unwrap().unwrap();
        assert_eq!(r.arena_bytes() as usize, payload.len());
        assert_eq!(r.read_slot(0).unwrap(), &payload[..]);
    }
}
