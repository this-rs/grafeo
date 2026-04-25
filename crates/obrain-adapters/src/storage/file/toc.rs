//! Table of Contents (TOC) for the `.obrain` v2 native format.
//!
//! The TOC lives at offset 12 KiB (immediately after FileHeader + 2 DbHeaders)
//! and occupies exactly 4 KiB. It describes the location and size of every
//! data section in the file.
//!
//! Layout within the 4 KiB page:
//!
//! | Offset | Size | Field |
//! |--------|------|-------|
//! | 0 | 4 | `magic` (`b"TOC\0"`) |
//! | 4 | 4 | `section_count` (u32) |
//! | 8 | 4 | `_reserved` |
//! | 12 | 4 | `_reserved2` |
//! | 16 | N×24 | `SectionEntry[N]` |
//!
//! With 24 bytes per entry and 4080 bytes available, the TOC supports up
//! to **170 sections** — far more than needed.

use obrain_common::types::flat::{CastError, Pod, cast_ref, cast_slice};
use std::mem;

/// Magic bytes identifying a valid TOC page.
pub const TOC_MAGIC: [u8; 4] = *b"TOC\0";

/// Size of the TOC page in bytes.
pub const TOC_PAGE_SIZE: usize = 4096;

/// Byte offset where the TOC starts in the file (after FileHeader + 2 DbHeaders).
pub const TOC_OFFSET: u64 = 12288;

/// Maximum number of section entries that fit in the TOC page.
/// (4096 - 16 header bytes) / 24 bytes per entry = 170
pub const MAX_SECTIONS: usize = (TOC_PAGE_SIZE - 16) / mem::size_of::<SectionEntry>();

// ─────────────────────────────────────────────────────────────────────
// SectionType — identifies each section in the file
// ─────────────────────────────────────────────────────────────────────

/// Identifies a data section in the v2 file format.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SectionType {
    /// Concatenated UTF-8 strings (labels, edge types, property keys, string values).
    Strings = 1,
    /// `NodeSlot` array, indexed by `NodeId.0`.
    Nodes = 2,
    /// `EdgeSlot` array, indexed by `EdgeId.0`.
    Edges = 3,
    /// CSR forward adjacency: offsets array (`u64 × (node_count + 1)`).
    CsrForwardOffsets = 4,
    /// CSR forward adjacency: targets array (`u64 × edge_count`).
    CsrForwardTargets = 5,
    /// CSR backward adjacency: offsets array.
    CsrBackwardOffsets = 6,
    /// CSR backward adjacency: targets array.
    CsrBackwardTargets = 7,
    /// Property entries: `PropEntry` array for all nodes and edges.
    Properties = 8,
    /// Property values: raw bytes referenced by `PropEntry.value_offset`.
    PropertyValues = 9,
    /// Label catalog: mapping label_id ↔ label name (StringRef array).
    LabelCatalog = 10,
    /// Edge type catalog: mapping type_id ↔ type name (StringRef array).
    EdgeTypeCatalog = 11,
    /// HNSW topology header + flat neighbor data.
    HnswTopology = 12,
    /// BM25 inverted index posting lists.
    Bm25Data = 13,
    /// Schema / named graphs metadata.
    Schema = 14,
    /// Label → node_id bitset index.
    LabelIndex = 15,
    /// CSR forward: edge_id array (parallel to targets, for edge lookup).
    CsrForwardEdgeIds = 16,
    /// CSR backward: edge_id array.
    CsrBackwardEdgeIds = 17,
}

impl SectionType {
    /// Converts a raw `u32` to a `SectionType`, returning `None` for unknown values.
    #[inline]
    #[must_use]
    pub const fn from_u32(v: u32) -> Option<Self> {
        match v {
            1 => Some(Self::Strings),
            2 => Some(Self::Nodes),
            3 => Some(Self::Edges),
            4 => Some(Self::CsrForwardOffsets),
            5 => Some(Self::CsrForwardTargets),
            6 => Some(Self::CsrBackwardOffsets),
            7 => Some(Self::CsrBackwardTargets),
            8 => Some(Self::Properties),
            9 => Some(Self::PropertyValues),
            10 => Some(Self::LabelCatalog),
            11 => Some(Self::EdgeTypeCatalog),
            12 => Some(Self::HnswTopology),
            13 => Some(Self::Bm25Data),
            14 => Some(Self::Schema),
            15 => Some(Self::LabelIndex),
            16 => Some(Self::CsrForwardEdgeIds),
            17 => Some(Self::CsrBackwardEdgeIds),
            _ => None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// SectionEntry — describes one section's location in the file
// ─────────────────────────────────────────────────────────────────────

/// Describes the location and size of a single section in the file.
///
/// Each entry is exactly 24 bytes, `#[repr(C)]`, and directly castable
/// from mmap'd bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(C)]
pub struct SectionEntry {
    /// Section type discriminant (see [`SectionType`]).
    pub section_type: u32,
    /// CRC-32 checksum of the section data.
    pub checksum: u32,
    /// Byte offset from the start of the file.
    pub offset: u64,
    /// Byte length of the section data.
    pub length: u64,
}

unsafe impl Pod for SectionEntry {}

const _: () = assert!(mem::size_of::<SectionEntry>() == 24);
const _: () = assert!(mem::align_of::<SectionEntry>() == 8);

impl SectionEntry {
    /// Creates a new section entry.
    #[inline]
    #[must_use]
    pub const fn new(section_type: SectionType, offset: u64, length: u64, checksum: u32) -> Self {
        Self {
            section_type: section_type as u32,
            checksum,
            offset,
            length,
        }
    }

    /// Returns the section type, or `None` if the discriminant is unknown.
    #[inline]
    #[must_use]
    pub const fn section_type(&self) -> Option<SectionType> {
        SectionType::from_u32(self.section_type)
    }

    /// Returns `true` if this entry is unused (zero offset and length).
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.offset == 0 && self.length == 0
    }

    /// Returns the byte range `[offset, offset + length)` for slicing into mmap.
    #[inline]
    #[must_use]
    pub const fn byte_range(&self) -> (usize, usize) {
        let start = self.offset as usize;
        (start, start + self.length as usize)
    }
}

// ─────────────────────────────────────────────────────────────────────
// TocHeader — the fixed 16-byte prefix of the TOC page
// ─────────────────────────────────────────────────────────────────────

/// Fixed 16-byte header at the start of the TOC page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct TocHeader {
    /// Magic bytes: `b"TOC\0"`.
    pub magic: [u8; 4],
    /// Number of valid section entries following this header.
    pub section_count: u32,
    /// Reserved for future use.
    pub _reserved: u32,
    /// Reserved for future use.
    pub _reserved2: u32,
}

unsafe impl Pod for TocHeader {}

const _: () = assert!(mem::size_of::<TocHeader>() == 16);

impl TocHeader {
    /// Creates a new TOC header.
    #[inline]
    #[must_use]
    pub const fn new(section_count: u32) -> Self {
        Self {
            magic: TOC_MAGIC,
            section_count,
            _reserved: 0,
            _reserved2: 0,
        }
    }

    /// Validates the magic bytes.
    #[inline]
    #[must_use]
    pub const fn is_valid(&self) -> bool {
        self.magic[0] == TOC_MAGIC[0]
            && self.magic[1] == TOC_MAGIC[1]
            && self.magic[2] == TOC_MAGIC[2]
            && self.magic[3] == TOC_MAGIC[3]
    }
}

// ─────────────────────────────────────────────────────────────────────
// FileToc — parsed Table of Contents (zero-copy from mmap)
// ─────────────────────────────────────────────────────────────────────

/// Parsed Table of Contents providing access to all section entries.
///
/// This is a zero-copy view over the TOC page: the header and entries
/// are borrowed directly from the mmap'd bytes.
#[derive(Debug)]
pub struct FileToc<'a> {
    /// The TOC header.
    pub header: &'a TocHeader,
    /// Slice of section entries (length = `header.section_count`).
    pub entries: &'a [SectionEntry],
}

/// Errors when parsing a TOC from raw bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TocError {
    /// The byte slice is too small to contain even the header.
    TooSmall {
        /// Minimum expected size.
        expected: usize,
        /// Actual size.
        actual: usize,
    },
    /// Invalid magic bytes.
    BadMagic {
        /// Actual magic bytes found.
        actual: [u8; 4],
    },
    /// Section count exceeds maximum.
    TooManySections {
        /// Declared section count.
        count: u32,
    },
    /// The byte slice is too small for the declared number of sections.
    Truncated {
        /// Minimum expected size.
        expected: usize,
        /// Actual size.
        actual: usize,
    },
    /// Cast error from the underlying byte reinterpretation.
    Cast(CastError),
}

impl std::fmt::Display for TocError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TocError::TooSmall { expected, actual } => {
                write!(f, "TOC too small: need {expected} bytes, got {actual}")
            }
            TocError::BadMagic { actual } => {
                write!(f, "TOC bad magic: {:?}", actual)
            }
            TocError::TooManySections { count } => {
                write!(f, "TOC section count {count} exceeds max {MAX_SECTIONS}")
            }
            TocError::Truncated { expected, actual } => {
                write!(f, "TOC truncated: need {expected} bytes, got {actual}")
            }
            TocError::Cast(e) => write!(f, "TOC cast error: {e}"),
        }
    }
}

impl std::error::Error for TocError {}

impl From<CastError> for TocError {
    fn from(e: CastError) -> Self {
        TocError::Cast(e)
    }
}

impl<'a> FileToc<'a> {
    /// Parse a `FileToc` from a raw byte slice (typically the 4 KiB TOC page).
    ///
    /// This is a zero-copy operation: the header and entries are borrowed
    /// from the input bytes.
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self, TocError> {
        let header_size = mem::size_of::<TocHeader>();
        if bytes.len() < header_size {
            return Err(TocError::TooSmall {
                expected: header_size,
                actual: bytes.len(),
            });
        }

        let header = cast_ref::<TocHeader>(bytes)?;

        if !header.is_valid() {
            return Err(TocError::BadMagic {
                actual: header.magic,
            });
        }

        let count = header.section_count as usize;
        if count > MAX_SECTIONS {
            return Err(TocError::TooManySections {
                count: header.section_count,
            });
        }

        let entries_size = count * mem::size_of::<SectionEntry>();
        let total_needed = header_size + entries_size;
        if bytes.len() < total_needed {
            return Err(TocError::Truncated {
                expected: total_needed,
                actual: bytes.len(),
            });
        }

        let entries_bytes = &bytes[header_size..header_size + entries_size];
        let entries = cast_slice::<SectionEntry>(entries_bytes)?;

        Ok(FileToc { header, entries })
    }

    /// Look up a section entry by type.
    ///
    /// Returns the first entry matching the given section type, or `None`.
    #[must_use]
    pub fn find(&self, section_type: SectionType) -> Option<&SectionEntry> {
        let target = section_type as u32;
        self.entries.iter().find(|e| e.section_type == target)
    }

    /// Look up a section entry by type, returning an error if not found.
    pub fn require(&self, section_type: SectionType) -> Result<&SectionEntry, TocError> {
        self.find(section_type).ok_or(TocError::TooSmall {
            expected: 1,
            actual: 0,
        })
    }

    /// Returns the number of section entries.
    #[inline]
    #[must_use]
    pub fn section_count(&self) -> usize {
        self.entries.len()
    }
}

// ─────────────────────────────────────────────────────────────────────
// TocBuilder — for constructing a TOC page in the writer
// ─────────────────────────────────────────────────────────────────────

/// Builder for constructing a TOC page to write into a v2 file.
///
/// Accumulates section entries, then serializes to a 4 KiB page.
#[derive(Debug, Default)]
pub struct TocBuilder {
    entries: Vec<SectionEntry>,
}

impl TocBuilder {
    /// Creates a new empty builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::with_capacity(20),
        }
    }

    /// Adds a section entry.
    ///
    /// # Panics
    ///
    /// Panics if the number of sections exceeds [`MAX_SECTIONS`].
    pub fn add(
        &mut self,
        section_type: SectionType,
        offset: u64,
        length: u64,
        checksum: u32,
    ) -> &mut Self {
        assert!(
            self.entries.len() < MAX_SECTIONS,
            "TOC overflow: max {MAX_SECTIONS} sections"
        );
        self.entries
            .push(SectionEntry::new(section_type, offset, length, checksum));
        self
    }

    /// Serializes the TOC to a 4 KiB page, zero-padded.
    #[must_use]
    pub fn build(&self) -> Vec<u8> {
        use obrain_common::types::flat::{as_bytes, slice_as_bytes};

        let mut page = vec![0u8; TOC_PAGE_SIZE];

        // Write header
        let header = TocHeader::new(self.entries.len() as u32);
        let header_bytes = as_bytes(&header);
        page[..header_bytes.len()].copy_from_slice(header_bytes);

        // Write entries
        let entries_bytes = slice_as_bytes(&self.entries);
        let offset = mem::size_of::<TocHeader>();
        page[offset..offset + entries_bytes.len()].copy_from_slice(entries_bytes);

        page
    }

    /// Returns the number of entries added so far.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if no entries have been added.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_section_entry_size() {
        assert_eq!(mem::size_of::<SectionEntry>(), 24);
    }

    #[test]
    fn test_toc_header_size() {
        assert_eq!(mem::size_of::<TocHeader>(), 16);
    }

    #[test]
    fn test_max_sections() {
        // (4096 - 16) / 24 = 170
        assert_eq!(MAX_SECTIONS, 170);
    }

    #[test]
    fn test_toc_roundtrip() {
        let mut builder = TocBuilder::new();
        builder.add(SectionType::Strings, 16384, 50000, 0xDEAD_BEEF);
        builder.add(SectionType::Nodes, 70000, 240000, 0x1234_5678);
        builder.add(SectionType::Edges, 310000, 480000, 0xABCD_EF01);

        let page = builder.build();
        assert_eq!(page.len(), TOC_PAGE_SIZE);

        let toc = FileToc::from_bytes(&page).expect("parse should succeed");
        assert_eq!(toc.section_count(), 3);

        // Verify Strings section
        let strings = toc.find(SectionType::Strings).expect("strings section");
        assert_eq!(strings.offset, 16384);
        assert_eq!(strings.length, 50000);
        assert_eq!(strings.checksum, 0xDEAD_BEEF);
        assert_eq!(strings.section_type(), Some(SectionType::Strings));

        // Verify Nodes section
        let nodes = toc.find(SectionType::Nodes).expect("nodes section");
        assert_eq!(nodes.offset, 70000);
        assert_eq!(nodes.length, 240000);

        // Verify Edges section
        let edges = toc.find(SectionType::Edges).expect("edges section");
        assert_eq!(edges.offset, 310000);
        assert_eq!(edges.length, 480000);

        // Non-existent section
        assert!(toc.find(SectionType::HnswTopology).is_none());
    }

    #[test]
    fn test_toc_bad_magic() {
        let mut page = vec![0u8; TOC_PAGE_SIZE];
        page[0..4].copy_from_slice(b"NOPE");
        let result = FileToc::from_bytes(&page);
        assert!(matches!(result, Err(TocError::BadMagic { .. })));
    }

    #[test]
    fn test_toc_too_small() {
        let tiny = [0u8; 8];
        let result = FileToc::from_bytes(&tiny);
        assert!(matches!(result, Err(TocError::TooSmall { .. })));
    }

    #[test]
    fn test_section_entry_byte_range() {
        let entry = SectionEntry::new(SectionType::Nodes, 1000, 500, 0);
        let (start, end) = entry.byte_range();
        assert_eq!(start, 1000);
        assert_eq!(end, 1500);
    }

    #[test]
    fn test_section_type_roundtrip() {
        for val in 1..=17u32 {
            let st = SectionType::from_u32(val).expect("valid section type");
            assert_eq!(st as u32, val);
        }
        assert!(SectionType::from_u32(0).is_none());
        assert!(SectionType::from_u32(18).is_none());
        assert!(SectionType::from_u32(255).is_none());
    }

    #[test]
    fn test_empty_toc() {
        let builder = TocBuilder::new();
        assert!(builder.is_empty());

        let page = builder.build();
        let toc = FileToc::from_bytes(&page).expect("parse empty TOC");
        assert_eq!(toc.section_count(), 0);
    }
}
