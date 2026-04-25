//! # Vector columns — per-prop-key dense mmap storage
//!
//! **T16.7 Step 1 — foundation.** Implements the on-disk format, writer
//! and reader for "vector columns": one mmap'd zone per vector-typed
//! property key, generalising the pattern already used by
//! [`crate::tier_persist`] for the 384-dim `_st_embedding` SRP
//! projection.
//!
//! ## Why
//!
//! Before this module, every `Value::Vector` property in the store
//! landed in the bincode-encoded `substrate.props` sidecar. At open
//! time the sidecar was read with `std::fs::read` + `decode_from_slice`,
//! which inflates the anon heap to **2×** the sidecar size (the file
//! bytes and the deserialised `Vec<PropEntry>` tree are live at the
//! same time). On a 4.5 M-node base with embeddings ≈ 1.5 KB each,
//! that peaks at ≈ 14 GB anon RSS — well past the T16 gate of 1 GB.
//!
//! Vector columns side-step this entirely:
//!
//! 1. Vectors live in dense mmap zones, one file per (prop_key,
//!    dim, dtype, entity_kind) combination.
//! 2. Open is O(1) — just an mmap plus a 64 B header validation.
//! 3. `get_node_property(.., "embedding")` reads directly from the
//!    mmap (zero-copy slice into the page cache).
//!
//! ## On-disk layout (v1)
//!
//! ```text
//! [64 B  VecColumnHeader]   magic "VEC1", version, spec, n_slots, CRC32
//! [ payload: n_slots × dim × sizeof(dtype)   ]
//! ```
//!
//! There is intentionally **no** slot-to-offset index in v1: columns
//! are dense-keyed by entity slot id. Sparse entities (those without
//! this property) still reserve their slot; the convention is
//! "all-zero slot == absent". Callers that need presence semantics
//! either wrap this with a side-bitset or choose a sentinel value.
//!
//! A 64 B Pod header keeps the layout stable; CRC32 covers the
//! payload region only (not the header itself), matching the
//! `tier_persist` contract so the two read paths share the same
//! "header-last write, CRC-validated read" discipline.
//!
//! ## Crash safety
//!
//! Writes are ordered the same way as [`crate::tier_persist`]:
//!
//!   1. `ensure_room` grows the zone to the target size.
//!   2. Slot payloads are memcpy'd into the mmap.
//!   3. CRC32 is computed over the final payload region.
//!   4. The header (with the final CRC) is written last.
//!   5. `msync` + `fsync` make the whole file durable.
//!
//! A crash before step 4 leaves a header with a stale CRC; step 5
//! is ordered after 4, so a crash after 4 but before 5 leaves the
//! kernel to flush on close — durability at that point is controlled
//! by the filesystem journal, same as the other zones.
//!
//! ## Scope of Step 1
//!
//! This module delivers the **format** and the **I/O primitives**.
//! It is not yet called from `SubstrateStore`: routing
//! `set_node_property(Value::Vector)` / `get_node_property` through
//! the vec-column registry is Step 2, and `obrain-migrate` routing
//! is Step 3. The tests here exercise the module end-to-end in
//! isolation via a temporary substrate directory.

use crate::error::{SubstrateError, SubstrateResult};
use crate::file::{SubstrateFile, ZoneFile};
use crate::writer::ensure_room;
use bytemuck::{Pod, Zeroable};

// ---------------------------------------------------------------------------
// Format constants
// ---------------------------------------------------------------------------

/// Little-endian `u32` of the ASCII bytes `b"VEC1"`. Used as the
/// magic word in [`VecColumnHeader::magic`]; chosen so `hexdump -C`
/// shows a human-readable identifier as the first 4 bytes of every
/// vec-column zone.
pub const VEC_COLUMN_MAGIC: u32 = u32::from_le_bytes(*b"VEC1");

/// Current on-disk format version. Bump together with the header
/// layout, the dtype enum or the payload encoding. Older zones with
/// a mismatching version are treated as "missing" by the reader —
/// they are never silently upgraded.
pub const VEC_COLUMN_VERSION: u32 = 1;

/// Fixed header size in bytes. The payload begins at [`HEADER_SIZE`]
/// unless the header's `flags` bit 0 is set (reserved for a future
/// sparse index; unused in v1).
pub const HEADER_SIZE: usize = 64;

// ---------------------------------------------------------------------------
// Entity kind + dtype
// ---------------------------------------------------------------------------

/// Which kind of entity a vector column keys on. Stored as a single
/// byte in the header so a hexdump can disambiguate node-scoped
/// columns from edge-scoped ones without parsing the prop-key table.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum EntityKind {
    Node = 0,
    Edge = 1,
}

impl EntityKind {
    pub fn as_suffix(self) -> &'static str {
        match self {
            EntityKind::Node => "node",
            EntityKind::Edge => "edge",
        }
    }

    pub fn from_u8(b: u8) -> Option<Self> {
        match b {
            0 => Some(EntityKind::Node),
            1 => Some(EntityKind::Edge),
            _ => None,
        }
    }
}

/// On-disk element type. The numeric value is the wire encoding in
/// [`VecColumnHeader::dtype`]; do **not** renumber variants once
/// they have shipped.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum VecDType {
    /// IEEE-754 binary32. The canonical dtype for neural embeddings.
    F32 = 0,
    /// IEEE-754 binary16 (aka half-float). Stored as raw u16 bytes;
    /// interpreting them as `f16` is the caller's responsibility since
    /// stable Rust has no `f16` primitive yet.
    F16 = 1,
    /// Unsigned byte. Useful for quantised embeddings (int8 → u8
    /// sign-shifted), packed bitmasks, and categorical features.
    U8 = 2,
    /// Unsigned 16-bit integer.
    U16 = 3,
    /// Signed byte. Used by most int8-quantised embedding schemes.
    I8 = 4,
    /// Signed 16-bit integer.
    I16 = 5,
    /// Signed 32-bit integer.
    I32 = 6,
}

impl VecDType {
    /// Width of one element in bytes.
    pub fn size_bytes(self) -> usize {
        match self {
            VecDType::F32 | VecDType::I32 => 4,
            VecDType::F16 | VecDType::U16 | VecDType::I16 => 2,
            VecDType::U8 | VecDType::I8 => 1,
        }
    }

    /// Short ASCII suffix used in vec-column filenames so a file
    /// listing is self-describing. Mirrors the numpy dtype short
    /// codes where practical.
    pub fn as_suffix(self) -> &'static str {
        match self {
            VecDType::F32 => "f32",
            VecDType::F16 => "f16",
            VecDType::U8 => "u8",
            VecDType::U16 => "u16",
            VecDType::I8 => "i8",
            VecDType::I16 => "i16",
            VecDType::I32 => "i32",
        }
    }

    /// Inverse of [`Self::as_suffix`]. Returns `None` for unknown
    /// codes so a corrupt filename degrades to "column missing"
    /// rather than panicking.
    pub fn from_suffix(s: &str) -> Option<Self> {
        Some(match s {
            "f32" => VecDType::F32,
            "f16" => VecDType::F16,
            "u8" => VecDType::U8,
            "u16" => VecDType::U16,
            "i8" => VecDType::I8,
            "i16" => VecDType::I16,
            "i32" => VecDType::I32,
            _ => return None,
        })
    }

    pub fn from_u8(b: u8) -> Option<Self> {
        Some(match b {
            0 => VecDType::F32,
            1 => VecDType::F16,
            2 => VecDType::U8,
            3 => VecDType::U16,
            4 => VecDType::I8,
            5 => VecDType::I16,
            6 => VecDType::I32,
            _ => return None,
        })
    }
}

// ---------------------------------------------------------------------------
// Spec + filename
// ---------------------------------------------------------------------------

/// Descriptor of one vector column. The four fields are the primary
/// key of the on-disk zone: different prop-keys, different entity
/// kinds, different dims, or different dtypes all land in **distinct
/// files**.
///
/// `prop_key_id` is the interned id assigned by the store's
/// `PropKeyRegistry` — stable across reopens because the registry
/// itself is persisted in `substrate.dict`. Using the id (not the
/// name) keeps filenames short and avoids filesystem-unfriendly
/// characters that a free-form prop name might contain.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct VecColSpec {
    pub prop_key_id: u16,
    pub entity_kind: EntityKind,
    pub dim: u32,
    pub dtype: VecDType,
}

impl VecColSpec {
    /// Byte stride of one slot in the payload region.
    #[inline]
    pub fn slot_stride(&self) -> usize {
        (self.dim as usize) * self.dtype.size_bytes()
    }

    /// Build the canonical filename for a column. The scheme is
    /// `substrate.veccol.<kind>.<key_id_hex04>.<dtype>.<dim>`; a `ls`
    /// grouped by prop-key is obtained by sorting lexicographically.
    pub fn filename(&self) -> String {
        format!(
            "substrate.veccol.{}.{:04x}.{}.{}",
            self.entity_kind.as_suffix(),
            self.prop_key_id,
            self.dtype.as_suffix(),
            self.dim
        )
    }
}

// ---------------------------------------------------------------------------
// On-disk header
// ---------------------------------------------------------------------------

/// 64 B `#[repr(C)]` header written at the start of every vec-column
/// zone. All fields are little-endian.
///
/// Field order is frozen by [`VEC_COLUMN_VERSION`]; adding a
/// mandatory field requires a version bump. The `_reserved` tail
/// gives 36 bytes for future non-mandatory extensions without
/// reshuffling offsets.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct VecColumnHeader {
    /// [`VEC_COLUMN_MAGIC`] — b"VEC1" little-endian.
    pub magic: u32,
    /// [`VEC_COLUMN_VERSION`] at write time.
    pub version: u32,
    /// Matches [`VecColSpec::prop_key_id`].
    pub prop_key_id: u16,
    /// 0 for node, 1 for edge. See [`EntityKind`].
    pub entity_kind: u8,
    /// [`VecDType`] encoded as a single byte. The numeric value is
    /// part of the on-disk contract — do not renumber.
    pub dtype: u8,
    /// Dimensionality of each vector.
    pub dim: u32,
    /// Number of allocated slots. Slot `i` lives at
    /// `HEADER_SIZE + i * dim * sizeof(dtype)`.
    pub n_slots: u32,
    /// Bit 0 reserved for a future sparse `slot_to_offset` index
    /// (unused in v1, always zero). Bits 1..31 unused.
    pub flags: u32,
    /// CRC32 (IEEE polynomial) over the payload region only —
    /// `[HEADER_SIZE, HEADER_SIZE + n_slots * slot_stride)`. The
    /// header itself is not self-CRC'd; a torn header manifests as
    /// a bad magic / version / record-size and degrades to "missing
    /// column" the same way.
    pub crc32: u32,
    /// Zero-filled padding to bring the header to 64 B. Never read
    /// by the current reader; lets future writers tack on
    /// non-mandatory fields without a version bump.
    pub _reserved: [u32; 9],
}

// Compile-time sanity: the header is exactly 64 B.
const _: [(); 1] = [(); (core::mem::size_of::<VecColumnHeader>() == HEADER_SIZE) as usize];

impl VecColumnHeader {
    fn new(spec: &VecColSpec, n_slots: u32) -> Self {
        Self {
            magic: VEC_COLUMN_MAGIC,
            version: VEC_COLUMN_VERSION,
            prop_key_id: spec.prop_key_id,
            entity_kind: spec.entity_kind as u8,
            dtype: spec.dtype as u8,
            dim: spec.dim,
            n_slots,
            flags: 0,
            crc32: 0,
            _reserved: [0u32; 9],
        }
    }

    /// Parse a header out of the first [`HEADER_SIZE`] bytes of a
    /// mmap region. Returns `None` if the region is too short or
    /// fails alignment; never panics.
    fn try_read(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < HEADER_SIZE {
            return None;
        }
        bytemuck::try_pod_read_unaligned::<VecColumnHeader>(&bytes[..HEADER_SIZE]).ok()
    }

    /// Validate that this header matches the expected spec. Returns
    /// `Ok(())` on match; any mismatch is reported as a descriptive
    /// error string so upstream can log it before falling back to
    /// "column missing". Note: the reader does **not** surface this
    /// error — it turns any validation failure into `Ok(None)`.
    fn validate_against(&self, spec: &VecColSpec) -> Result<(), &'static str> {
        if self.magic != VEC_COLUMN_MAGIC {
            return Err("magic mismatch");
        }
        if self.version != VEC_COLUMN_VERSION {
            return Err("version mismatch");
        }
        if self.prop_key_id != spec.prop_key_id {
            return Err("prop_key_id mismatch");
        }
        if self.entity_kind != spec.entity_kind as u8 {
            return Err("entity_kind mismatch");
        }
        if self.dtype != spec.dtype as u8 {
            return Err("dtype mismatch");
        }
        if self.dim != spec.dim {
            return Err("dim mismatch");
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

/// Dense-keyed writer for one vector column.
///
/// The writer grows the zone on demand (each `write_slot` past the
/// current high-water mark re-sizes the mmap once), memcpy's caller
/// bytes into the payload region, and commits the header last in
/// [`Self::finalize`].
///
/// ## Usage pattern
///
/// ```ignore
/// let spec = VecColSpec { prop_key_id: 7, entity_kind: EntityKind::Node,
///                         dim: 384, dtype: VecDType::F32 };
/// let mut w = VecColumnWriter::create(&sub, spec)?;
/// w.write_slot(42, bytemuck::cast_slice(&vector_f32))?;
/// w.finalize()?;
/// ```
///
/// Writes are **not** transactional within the writer itself — a
/// crash between `write_slot` and `finalize` leaves a zone with a
/// stale CRC, which the reader will reject. Callers that need
/// transactional semantics should keep slots coherent with the WAL
/// (tracked in T17 property-pages).
#[derive(Debug)]
pub struct VecColumnWriter {
    spec: VecColSpec,
    zf: ZoneFile,
    /// Highest slot index ever written (+1), i.e. `n_slots` at
    /// finalise time. Starts at zero and monotonically grows.
    n_slots: u32,
}

impl VecColumnWriter {
    /// Create (or re-open) a vec-column zone for the given spec.
    ///
    /// If the zone already exists on disk, the writer inherits its
    /// current `n_slots` from the header so subsequent `write_slot`s
    /// past that point grow the column. An unreadable / mismatching
    /// header is treated as a fresh start — the old contents are
    /// overwritten.
    pub fn create(sub: &SubstrateFile, spec: VecColSpec) -> SubstrateResult<Self> {
        if spec.dim == 0 {
            return Err(SubstrateError::WalBadFrame(
                "vec_column: dim must be non-zero".into(),
            ));
        }
        let zf = sub.open_named_zone(&spec.filename())?;
        let n_slots = match VecColumnHeader::try_read(zf.as_slice()) {
            Some(h) if h.validate_against(&spec).is_ok() => h.n_slots,
            _ => 0,
        };
        Ok(Self { spec, zf, n_slots })
    }

    /// Write (or overwrite) slot `slot` with the raw byte content
    /// of one vector. The caller guarantees that
    /// `bytes.len() == spec.slot_stride()`; mismatches are rejected.
    pub fn write_slot(&mut self, slot: u32, bytes: &[u8]) -> SubstrateResult<()> {
        let stride = self.spec.slot_stride();
        if bytes.len() != stride {
            return Err(SubstrateError::WalBadFrame(format!(
                "vec_column::write_slot: payload len {} != expected stride {} \
                 (dim={}, dtype={:?})",
                bytes.len(),
                stride,
                self.spec.dim,
                self.spec.dtype
            )));
        }
        let slot_usize = slot as usize;
        let end = HEADER_SIZE + (slot_usize + 1) * stride;
        ensure_room(&mut self.zf, end, 0)?;
        {
            let region = self.zf.as_slice_mut();
            let offset = HEADER_SIZE + slot_usize * stride;
            region[offset..offset + stride].copy_from_slice(bytes);
        }
        if slot + 1 > self.n_slots {
            self.n_slots = slot + 1;
        }
        Ok(())
    }

    /// Finalise the column: compute the CRC over the live payload
    /// region, write the header, `msync`, `fsync`. Consumes the
    /// writer — the file is durable on return.
    ///
    /// Calling `finalize` on a writer that never issued a
    /// `write_slot` is valid and produces a header with `n_slots=0`;
    /// a subsequent reader sees an empty column.
    pub fn finalize(mut self) -> SubstrateResult<()> {
        let stride = self.spec.slot_stride();
        let payload_bytes = (self.n_slots as usize) * stride;
        let total = HEADER_SIZE + payload_bytes;
        ensure_room(&mut self.zf, total, 0)?;

        // CRC over the payload. Zero-slot columns hash an empty
        // slice, which `crc32fast` maps to `0`.
        let crc = {
            let slice = self.zf.as_slice();
            let payload = &slice[HEADER_SIZE..HEADER_SIZE + payload_bytes];
            let mut h = crc32fast::Hasher::new();
            h.update(payload);
            h.finalize()
        };

        let mut header = VecColumnHeader::new(&self.spec, self.n_slots);
        header.crc32 = crc;
        {
            let slice = self.zf.as_slice_mut();
            slice[..HEADER_SIZE].copy_from_slice(bytemuck::bytes_of(&header));
        }

        self.zf.msync()?;
        self.zf.fsync()?;
        Ok(())
    }

    /// Current high-water mark (number of slots written so far).
    pub fn n_slots(&self) -> u32 {
        self.n_slots
    }

    /// The spec this writer targets.
    pub fn spec(&self) -> &VecColSpec {
        &self.spec
    }

    /// Borrow the raw bytes for one slot out of the writer's own
    /// mmap, or `None` if the slot index is past the current
    /// high-water mark.
    ///
    /// This mirrors [`VecColumnReader::read_slot`] but serves reads
    /// through the *same* mmap that writes go through, so the T16.7
    /// routing path does not need to maintain a separate reader
    /// handle per spec. Between `write_slot` and the next
    /// `finalize` the on-disk header still carries a stale CRC, so a
    /// fresh [`VecColumnReader`] would refuse to open — but the
    /// payload bytes are live in the mmap and safe to re-read here.
    ///
    /// The returned slice is `dim * sizeof(dtype)` bytes long and
    /// aliases the writer's mmap directly — zero-copy.
    pub fn read_slot(&self, slot: u32) -> Option<&[u8]> {
        if slot >= self.n_slots {
            return None;
        }
        let stride = self.spec.slot_stride();
        let offset = HEADER_SIZE + (slot as usize) * stride;
        let end = offset + stride;
        let slice = self.zf.as_slice();
        if end > slice.len() {
            // Should never happen — ensure_room already grew the
            // file to accommodate `n_slots`. Defensive only.
            return None;
        }
        Some(&slice[offset..end])
    }

    /// Convenience typed reader for `f32` columns served by the
    /// writer's mmap. Returns `None` if the slot is out of range or
    /// the stored dtype is not F32. See [`Self::read_slot`] for the
    /// through-writer rationale.
    pub fn read_slot_f32(&self, slot: u32) -> Option<&[f32]> {
        if self.spec.dtype != VecDType::F32 {
            return None;
        }
        let bytes = self.read_slot(slot)?;
        bytemuck::try_cast_slice::<u8, f32>(bytes).ok()
    }

    /// Force-finalise the current state: compute the payload CRC,
    /// write the header, `msync` + `fsync`. Unlike
    /// [`Self::finalize`], the writer stays alive so subsequent
    /// `write_slot`s can extend the column after the fact. Used by
    /// `SubstrateStore::flush` to make all open vec columns durable
    /// without dropping the registry.
    pub fn sync(&mut self) -> SubstrateResult<()> {
        let stride = self.spec.slot_stride();
        let payload_bytes = (self.n_slots as usize) * stride;
        let total = HEADER_SIZE + payload_bytes;
        ensure_room(&mut self.zf, total, 0)?;

        let crc = {
            let slice = self.zf.as_slice();
            let payload = &slice[HEADER_SIZE..HEADER_SIZE + payload_bytes];
            let mut h = crc32fast::Hasher::new();
            h.update(payload);
            h.finalize()
        };
        let mut header = VecColumnHeader::new(&self.spec, self.n_slots);
        header.crc32 = crc;
        {
            let slice = self.zf.as_slice_mut();
            slice[..HEADER_SIZE].copy_from_slice(bytemuck::bytes_of(&header));
        }
        self.zf.msync()?;
        self.zf.fsync()?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

/// Mmap-backed reader for one vector column.
///
/// Opening a reader is O(1): a 64 B header read plus a CRC32 pass
/// over the payload (the CRC *is* O(payload_bytes), but runs at
/// ~2 GB/s on modern hardware via `crc32fast` — bounded by L1d
/// bandwidth, dominated by first-use paging cost). We do this once
/// per open and trust the column for the lifetime of the handle.
///
/// [`Self::read_slot`] returns a zero-copy `&[u8]` slice into the
/// mmap. The slice is valid for as long as the [`VecColumnReader`]
/// is alive — callers that need owned data must clone explicitly.
pub struct VecColumnReader {
    spec: VecColSpec,
    zf: ZoneFile,
    header: VecColumnHeader,
    payload_start: usize,
    slot_stride: usize,
}

impl VecColumnReader {
    /// Open the vec-column zone matching `spec`. Returns:
    ///
    /// * `Ok(Some(reader))` on success,
    /// * `Ok(None)` if the zone does not exist, has a mismatching
    ///   header, or fails its CRC check — the caller is expected to
    ///   rebuild or fall back to an empty column,
    /// * `Err(_)` only on underlying I/O errors from the mmap
    ///   subsystem (i.e. "we couldn't even read the bytes").
    pub fn open(sub: &SubstrateFile, spec: VecColSpec) -> SubstrateResult<Option<Self>> {
        if spec.dim == 0 {
            return Ok(None);
        }
        let zf = sub.open_named_zone(&spec.filename())?;
        // Empty zone file — a fresh substrate that has never written
        // this column. This is the hot path for the first open after
        // migration, so stay O(1).
        if zf.is_empty() {
            return Ok(None);
        }
        let slice = zf.as_slice();
        let Some(header) = VecColumnHeader::try_read(slice) else {
            return Ok(None);
        };
        if header.validate_against(&spec).is_err() {
            return Ok(None);
        }
        let stride = spec.slot_stride();
        let payload_bytes = (header.n_slots as usize) * stride;
        let expected_end = HEADER_SIZE + payload_bytes;
        if slice.len() < expected_end {
            // Truncated file — the writer never finalised, or the
            // zone was manually corrupted.
            return Ok(None);
        }

        // CRC check.
        let crc = {
            let payload = &slice[HEADER_SIZE..expected_end];
            let mut h = crc32fast::Hasher::new();
            h.update(payload);
            h.finalize()
        };
        if crc != header.crc32 {
            return Ok(None);
        }

        Ok(Some(Self {
            spec,
            zf,
            header,
            payload_start: HEADER_SIZE,
            slot_stride: stride,
        }))
    }

    /// Number of slots currently stored.
    pub fn n_slots(&self) -> u32 {
        self.header.n_slots
    }

    /// Element count per vector.
    pub fn dim(&self) -> u32 {
        self.header.dim
    }

    /// Stored dtype.
    pub fn dtype(&self) -> VecDType {
        // `unwrap_or` keeps read_slot infallible even if the on-disk
        // dtype byte was corrupted — validation already rejected
        // such headers in `open`, so this is defensive.
        VecDType::from_u8(self.header.dtype).unwrap_or(VecDType::F32)
    }

    /// The spec this reader targets.
    pub fn spec(&self) -> &VecColSpec {
        &self.spec
    }

    /// Borrow the raw bytes for one slot, or `None` if the slot
    /// index is out of range.
    ///
    /// The returned slice is `dim * sizeof(dtype)` bytes long and
    /// aliases the mmap directly — cheap in both latency (no copy)
    /// and memory (no anon allocation).
    pub fn read_slot(&self, slot: u32) -> Option<&[u8]> {
        if slot >= self.header.n_slots {
            return None;
        }
        let offset = self.payload_start + (slot as usize) * self.slot_stride;
        let end = offset + self.slot_stride;
        let slice = self.zf.as_slice();
        if end > slice.len() {
            // Should never happen after validation; defensive.
            return None;
        }
        Some(&slice[offset..end])
    }

    /// Convenience typed reader for `f32` columns. Returns `None`
    /// if the slot is out of range or the stored dtype is not F32.
    pub fn read_slot_f32(&self, slot: u32) -> Option<&[f32]> {
        if self.dtype() != VecDType::F32 {
            return None;
        }
        let bytes = self.read_slot(slot)?;
        bytemuck::try_cast_slice::<u8, f32>(bytes).ok()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_spec_f32(prop_key_id: u16, dim: u32) -> VecColSpec {
        VecColSpec {
            prop_key_id,
            entity_kind: EntityKind::Node,
            dim,
            dtype: VecDType::F32,
        }
    }

    #[test]
    fn header_size_matches_constant() {
        assert_eq!(core::mem::size_of::<VecColumnHeader>(), HEADER_SIZE);
    }

    #[test]
    fn magic_bytes_are_ascii_vec1() {
        assert_eq!(VEC_COLUMN_MAGIC.to_le_bytes(), *b"VEC1");
    }

    #[test]
    fn dtype_roundtrips_through_u8_and_suffix() {
        for dt in [
            VecDType::F32,
            VecDType::F16,
            VecDType::U8,
            VecDType::U16,
            VecDType::I8,
            VecDType::I16,
            VecDType::I32,
        ] {
            assert_eq!(VecDType::from_u8(dt as u8), Some(dt));
            assert_eq!(VecDType::from_suffix(dt.as_suffix()), Some(dt));
        }
        assert_eq!(VecDType::from_u8(42), None);
        assert_eq!(VecDType::from_suffix("bogus"), None);
    }

    #[test]
    fn spec_filename_is_deterministic_and_descriptive() {
        let s = VecColSpec {
            prop_key_id: 0x2A,
            entity_kind: EntityKind::Node,
            dim: 384,
            dtype: VecDType::F32,
        };
        assert_eq!(s.filename(), "substrate.veccol.node.002a.f32.384");

        let e = VecColSpec {
            prop_key_id: 7,
            entity_kind: EntityKind::Edge,
            dim: 16,
            dtype: VecDType::U8,
        };
        assert_eq!(e.filename(), "substrate.veccol.edge.0007.u8.16");
    }

    #[test]
    fn empty_zone_returns_none_on_open() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let spec = sample_spec_f32(1, 8);
        let reader = VecColumnReader::open(&sub, spec).unwrap();
        assert!(reader.is_none(), "fresh substrate has no vec column");
    }

    #[test]
    fn roundtrip_single_slot_f32() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let spec = sample_spec_f32(1, 4);
        let v: [f32; 4] = [1.5, -0.25, 3.14, 42.0];

        let mut w = VecColumnWriter::create(&sub, spec).unwrap();
        w.write_slot(0, bytemuck::cast_slice(&v)).unwrap();
        assert_eq!(w.n_slots(), 1);
        w.finalize().unwrap();

        let r = VecColumnReader::open(&sub, spec).unwrap().unwrap();
        assert_eq!(r.dim(), 4);
        assert_eq!(r.dtype(), VecDType::F32);
        assert_eq!(r.n_slots(), 1);
        let got = r.read_slot_f32(0).unwrap();
        assert_eq!(got, &v);
        // Out-of-range slot is None, never a panic.
        assert!(r.read_slot_f32(1).is_none());
    }

    #[test]
    fn roundtrip_many_slots_dense() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let spec = sample_spec_f32(7, 16);
        let n = 100u32;

        let mut w = VecColumnWriter::create(&sub, spec).unwrap();
        for slot in 0..n {
            let v: Vec<f32> = (0..spec.dim)
                .map(|i| slot as f32 + i as f32 * 0.125)
                .collect();
            w.write_slot(slot, bytemuck::cast_slice(&v)).unwrap();
        }
        assert_eq!(w.n_slots(), n);
        w.finalize().unwrap();

        let r = VecColumnReader::open(&sub, spec).unwrap().unwrap();
        assert_eq!(r.n_slots(), n);
        for slot in 0..n {
            let got = r.read_slot_f32(slot).unwrap();
            let expected: Vec<f32> = (0..spec.dim)
                .map(|i| slot as f32 + i as f32 * 0.125)
                .collect();
            assert_eq!(got, &expected[..]);
        }
    }

    #[test]
    fn sparse_writes_are_zero_filled_between() {
        // Write slot 0 and slot 10; the reader should see zero-filled
        // vectors at slots 1..10 (dense semantics, unused slots are
        // "all-zero == absent" by convention).
        let sub = SubstrateFile::open_tempfile().unwrap();
        let spec = sample_spec_f32(3, 4);

        let v0: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let v10: [f32; 4] = [10.0, 20.0, 30.0, 40.0];

        let mut w = VecColumnWriter::create(&sub, spec).unwrap();
        w.write_slot(0, bytemuck::cast_slice(&v0)).unwrap();
        w.write_slot(10, bytemuck::cast_slice(&v10)).unwrap();
        assert_eq!(w.n_slots(), 11, "n_slots tracks highest+1");
        w.finalize().unwrap();

        let r = VecColumnReader::open(&sub, spec).unwrap().unwrap();
        assert_eq!(r.n_slots(), 11);
        assert_eq!(r.read_slot_f32(0).unwrap(), &v0);
        assert_eq!(r.read_slot_f32(10).unwrap(), &v10);
        for mid in 1..10 {
            let got = r.read_slot_f32(mid).unwrap();
            assert!(
                got.iter().all(|x| *x == 0.0),
                "slot {mid} should be zero-filled, got {got:?}"
            );
        }
    }

    #[test]
    fn wrong_stride_in_write_is_rejected() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let spec = sample_spec_f32(1, 4); // stride = 16 B
        let mut w = VecColumnWriter::create(&sub, spec).unwrap();
        let too_short = [0u8; 8];
        let err = w.write_slot(0, &too_short).unwrap_err();
        assert!(matches!(err, SubstrateError::WalBadFrame(_)));
    }

    #[test]
    fn zero_dim_spec_is_rejected() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let spec = VecColSpec {
            prop_key_id: 0,
            entity_kind: EntityKind::Node,
            dim: 0,
            dtype: VecDType::F32,
        };
        let err = VecColumnWriter::create(&sub, spec).unwrap_err();
        assert!(matches!(err, SubstrateError::WalBadFrame(_)));
        // Reader degrades to None rather than erroring — dim=0 is not
        // a distinguishable "empty" vs "invalid".
        assert!(VecColumnReader::open(&sub, spec).unwrap().is_none());
    }

    #[test]
    fn mismatching_spec_returns_none_on_open() {
        // Write with spec A, try to open with spec B (different dim).
        // Must degrade gracefully to None.
        let sub = SubstrateFile::open_tempfile().unwrap();
        let spec_a = sample_spec_f32(1, 4);
        let v: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let mut w = VecColumnWriter::create(&sub, spec_a).unwrap();
        w.write_slot(0, bytemuck::cast_slice(&v)).unwrap();
        w.finalize().unwrap();

        // Different dim → different filename entirely (disjoint zones),
        // so the "mismatch" is really "missing". That's expected.
        let spec_other_dim = sample_spec_f32(1, 8);
        assert!(
            VecColumnReader::open(&sub, spec_other_dim)
                .unwrap()
                .is_none()
        );

        // Same filename, but caller asks for a different prop_key_id.
        // We need to forge this by targeting the same on-disk file
        // through a distinct spec — which cannot happen in practice
        // because spec is part of the filename. The only realistic
        // mismatch is post-corruption, covered by the CRC test below.
    }

    #[test]
    fn corrupt_payload_crc_returns_none_on_open() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let spec = sample_spec_f32(1, 4);
        let v: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let mut w = VecColumnWriter::create(&sub, spec).unwrap();
        w.write_slot(0, bytemuck::cast_slice(&v)).unwrap();
        w.finalize().unwrap();

        // Flip one payload byte behind the reader's back.
        {
            let mut zf = sub.open_named_zone(&spec.filename()).unwrap();
            zf.as_slice_mut()[HEADER_SIZE + 3] ^= 0xFF;
            zf.msync().unwrap();
        }

        let r = VecColumnReader::open(&sub, spec).unwrap();
        assert!(r.is_none(), "corrupt CRC must degrade to None");
    }

    #[test]
    fn corrupt_magic_returns_none_on_open() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let spec = sample_spec_f32(1, 4);
        let v: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let mut w = VecColumnWriter::create(&sub, spec).unwrap();
        w.write_slot(0, bytemuck::cast_slice(&v)).unwrap();
        w.finalize().unwrap();

        // Flip the first magic byte.
        {
            let mut zf = sub.open_named_zone(&spec.filename()).unwrap();
            zf.as_slice_mut()[0] ^= 0xFF;
            zf.msync().unwrap();
        }

        assert!(VecColumnReader::open(&sub, spec).unwrap().is_none());
    }

    #[test]
    fn bumped_version_returns_none_on_open() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let spec = sample_spec_f32(1, 4);
        let v: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let mut w = VecColumnWriter::create(&sub, spec).unwrap();
        w.write_slot(0, bytemuck::cast_slice(&v)).unwrap();
        w.finalize().unwrap();

        // Bump the version field (u32 at offset 4) to something the
        // reader does not know.
        {
            let mut zf = sub.open_named_zone(&spec.filename()).unwrap();
            let bad: u32 = VEC_COLUMN_VERSION + 99;
            zf.as_slice_mut()[4..8].copy_from_slice(&bad.to_le_bytes());
            zf.msync().unwrap();
        }

        assert!(VecColumnReader::open(&sub, spec).unwrap().is_none());
    }

    #[test]
    fn truncated_file_returns_none_on_open() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let spec = sample_spec_f32(1, 16);
        let n = 10u32;
        let mut w = VecColumnWriter::create(&sub, spec).unwrap();
        for slot in 0..n {
            let v: Vec<f32> = vec![slot as f32; spec.dim as usize];
            w.write_slot(slot, bytemuck::cast_slice(&v)).unwrap();
        }
        w.finalize().unwrap();

        // Truncate the file below `HEADER_SIZE + n * stride`.
        let expected = HEADER_SIZE + (n as usize) * spec.slot_stride();
        let path = sub.path().join(spec.filename());
        let f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        f.set_len(expected as u64 - 4).unwrap();
        drop(f);

        assert!(VecColumnReader::open(&sub, spec).unwrap().is_none());
    }

    #[test]
    fn writer_reopen_inherits_n_slots() {
        // Create, write slots 0..5, finalise. Reopen a new writer —
        // it must observe n_slots=5 and let us append slot 5 rather
        // than overwriting the whole file.
        let sub = SubstrateFile::open_tempfile().unwrap();
        let spec = sample_spec_f32(1, 4);
        let v0: [f32; 4] = [1.0, 2.0, 3.0, 4.0];

        let mut w = VecColumnWriter::create(&sub, spec).unwrap();
        for slot in 0..5 {
            let v: Vec<f32> = vec![slot as f32; spec.dim as usize];
            w.write_slot(slot, bytemuck::cast_slice(&v)).unwrap();
        }
        w.finalize().unwrap();

        let mut w2 = VecColumnWriter::create(&sub, spec).unwrap();
        assert_eq!(w2.n_slots(), 5, "reopen inherits the prior n_slots");
        w2.write_slot(5, bytemuck::cast_slice(&v0)).unwrap();
        assert_eq!(w2.n_slots(), 6);
        w2.finalize().unwrap();

        let r = VecColumnReader::open(&sub, spec).unwrap().unwrap();
        assert_eq!(r.n_slots(), 6);
        for slot in 0..5 {
            let got = r.read_slot_f32(slot).unwrap();
            let expected: Vec<f32> = vec![slot as f32; spec.dim as usize];
            assert_eq!(got, &expected[..]);
        }
        assert_eq!(r.read_slot_f32(5).unwrap(), &v0);
    }

    #[test]
    fn empty_column_finalise_has_zero_crc() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let spec = sample_spec_f32(9, 4);
        let w = VecColumnWriter::create(&sub, spec).unwrap();
        w.finalize().unwrap();

        let r = VecColumnReader::open(&sub, spec).unwrap().unwrap();
        assert_eq!(r.n_slots(), 0);
        assert!(r.read_slot(0).is_none());
    }

    #[test]
    fn u8_dtype_roundtrip() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let spec = VecColSpec {
            prop_key_id: 2,
            entity_kind: EntityKind::Node,
            dim: 8,
            dtype: VecDType::U8,
        };
        let v: [u8; 8] = [1, 2, 3, 4, 5, 6, 7, 8];

        let mut w = VecColumnWriter::create(&sub, spec).unwrap();
        w.write_slot(3, &v).unwrap();
        w.finalize().unwrap();

        let r = VecColumnReader::open(&sub, spec).unwrap().unwrap();
        assert_eq!(r.n_slots(), 4);
        assert_eq!(r.read_slot(3).unwrap(), &v);
        // f32 typed reader rejects non-f32 columns.
        assert!(r.read_slot_f32(3).is_none());
    }

    #[test]
    fn edge_kind_has_distinct_filename_from_node_kind() {
        // Same prop_key_id but different entity_kind → distinct
        // files, no collision. Verifies the invariant that lets the
        // store route node-properties and edge-properties
        // independently.
        let sub = SubstrateFile::open_tempfile().unwrap();
        let node_spec = VecColSpec {
            prop_key_id: 5,
            entity_kind: EntityKind::Node,
            dim: 4,
            dtype: VecDType::F32,
        };
        let edge_spec = VecColSpec {
            entity_kind: EntityKind::Edge,
            ..node_spec
        };

        assert_ne!(node_spec.filename(), edge_spec.filename());

        let v_node: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let v_edge: [f32; 4] = [10.0, 20.0, 30.0, 40.0];

        let mut wn = VecColumnWriter::create(&sub, node_spec).unwrap();
        wn.write_slot(0, bytemuck::cast_slice(&v_node)).unwrap();
        wn.finalize().unwrap();

        let mut we = VecColumnWriter::create(&sub, edge_spec).unwrap();
        we.write_slot(0, bytemuck::cast_slice(&v_edge)).unwrap();
        we.finalize().unwrap();

        let rn = VecColumnReader::open(&sub, node_spec).unwrap().unwrap();
        let re = VecColumnReader::open(&sub, edge_spec).unwrap().unwrap();
        assert_eq!(rn.read_slot_f32(0).unwrap(), &v_node);
        assert_eq!(re.read_slot_f32(0).unwrap(), &v_edge);
    }

    #[test]
    fn writer_read_slot_matches_written_bytes() {
        // The T16.7 Step 2 hot path: the routing layer keeps a
        // writer open and reads from it without a separate reader
        // handle. This exercises the writer-served read path.
        let sub = SubstrateFile::open_tempfile().unwrap();
        let spec = sample_spec_f32(3, 6);
        let mut w = VecColumnWriter::create(&sub, spec).unwrap();

        let v0: [f32; 6] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let v2: [f32; 6] = [2.0, 2.1, 2.2, 2.3, 2.4, 2.5];
        w.write_slot(0, bytemuck::cast_slice(&v0)).unwrap();
        w.write_slot(2, bytemuck::cast_slice(&v2)).unwrap();

        // Slot 0 reads back what we wrote.
        assert_eq!(w.read_slot_f32(0).unwrap(), &v0);
        // Slot 1 is the zero-filled gap.
        assert_eq!(w.read_slot_f32(1).unwrap(), &[0.0f32; 6][..]);
        // Slot 2 reads back what we wrote.
        assert_eq!(w.read_slot_f32(2).unwrap(), &v2);
        // Slot 3 is past the high-water mark.
        assert_eq!(w.read_slot_f32(3), None);
    }

    #[test]
    fn writer_sync_makes_new_reader_valid_without_dropping() {
        // Sync writes the header (with fresh CRC) + msync+fsync,
        // but keeps the writer alive. A fresh Reader opened after
        // sync sees the data. Subsequent writes through the same
        // writer should still be possible and visible after another
        // sync.
        let sub = SubstrateFile::open_tempfile().unwrap();
        let spec = sample_spec_f32(4, 3);
        let mut w = VecColumnWriter::create(&sub, spec).unwrap();

        let v0: [f32; 3] = [1.0, 2.0, 3.0];
        w.write_slot(0, bytemuck::cast_slice(&v0)).unwrap();
        w.sync().unwrap();

        // Read via a fresh reader.
        let r = VecColumnReader::open(&sub, spec).unwrap().unwrap();
        assert_eq!(r.n_slots(), 1);
        assert_eq!(r.read_slot_f32(0).unwrap(), &v0);
        drop(r);

        // Writer still alive — extend the column.
        let v1: [f32; 3] = [10.0, 20.0, 30.0];
        w.write_slot(1, bytemuck::cast_slice(&v1)).unwrap();
        w.sync().unwrap();

        let r = VecColumnReader::open(&sub, spec).unwrap().unwrap();
        assert_eq!(r.n_slots(), 2);
        assert_eq!(r.read_slot_f32(0).unwrap(), &v0);
        assert_eq!(r.read_slot_f32(1).unwrap(), &v1);
    }

    #[test]
    fn writer_read_slot_rejects_wrong_dtype() {
        // A u8 writer should refuse to serve read_slot_f32 even
        // though the payload happens to be 4-byte aligned.
        let sub = SubstrateFile::open_tempfile().unwrap();
        let spec = VecColSpec {
            prop_key_id: 5,
            entity_kind: EntityKind::Node,
            dim: 4,
            dtype: VecDType::U8,
        };
        let mut w = VecColumnWriter::create(&sub, spec).unwrap();
        w.write_slot(0, &[1u8, 2, 3, 4]).unwrap();
        assert!(w.read_slot_f32(0).is_none());
        assert_eq!(w.read_slot(0).unwrap(), &[1u8, 2, 3, 4][..]);
    }
}
