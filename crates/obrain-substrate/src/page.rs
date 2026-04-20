//! 4 KiB property pages with an inline slot directory.
//!
//! See `docs/rfc/substrate/format-spec.md` §5 for the layout.

use crate::record::U48;
use bytemuck::{Pod, Zeroable};

/// Magic marker for [`PropertyPage`] (FNV-1a of "property").
pub const PROP_PAGE_MAGIC: u32 = 0xF507_5FA6;

/// Size of a property page on disk.
pub const PAGE_SIZE: usize = 4096;

/// Header of a [`PropertyPage`]. 24 B, no implicit padding.
///
/// Field order is chosen so each `u32` lands on a 4-aligned offset and the
/// `U48` lands on a 1-aligned offset after all `u32`s, with the `u16` trio
/// filling the tail exactly.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct PropertyPageHeader {
    pub magic: u32,        // @0..4
    pub node_id: u32,      // @4..8
    pub crc32: u32,        // @8..12
    pub next_page: U48,    // @12..18
    pub entry_count: u16,  // @18..20
    pub free_offset: u16,  // @20..22
    pub tombstones: u16,   // @22..24
}

const _: [(); 1] = [(); (core::mem::size_of::<PropertyPageHeader>() == 24) as usize];

/// Full [`PropertyPage`] including the inline entry region.
///
/// The entry region is opaque to the page type — callers serialize
/// [`PropertyEntry`] values into `payload` using [`PropertyCursor`].
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PropertyPage {
    pub header: PropertyPageHeader,
    pub payload: [u8; PAGE_SIZE - core::mem::size_of::<PropertyPageHeader>()],
}

const _: [(); 1] = [(); (core::mem::size_of::<PropertyPage>() == PAGE_SIZE) as usize];

impl Default for PropertyPage {
    fn default() -> Self {
        Self {
            header: PropertyPageHeader::default(),
            payload: [0; PAGE_SIZE - core::mem::size_of::<PropertyPageHeader>()],
        }
    }
}

impl core::fmt::Debug for PropertyPage {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PropertyPage")
            .field("header", &self.header)
            .field("payload_len", &self.payload.len())
            .finish()
    }
}

impl PropertyPage {
    /// Initialize a fresh page for `node_id`.
    pub fn new(node_id: u32) -> Self {
        let mut page = Self::default();
        page.header.magic = PROP_PAGE_MAGIC;
        page.header.node_id = node_id;
        page.header.free_offset = 0;
        page
    }

    /// Compute CRC32 over the page, treating the `crc32` header field as zero.
    pub fn compute_crc32(&self) -> u32 {
        let mut h = crc32fast::Hasher::new();
        // Header with crc32 field zeroed
        let mut header_copy = self.header;
        header_copy.crc32 = 0;
        h.update(bytemuck::bytes_of(&header_copy));
        h.update(&self.payload);
        h.finalize()
    }

    pub fn verify_crc32(&self) -> bool {
        self.compute_crc32() == self.header.crc32
    }

    pub fn seal_crc32(&mut self) {
        self.header.crc32 = self.compute_crc32();
    }
}

// ---------------------------------------------------------------------------
// Property value tags and entry layout.
// ---------------------------------------------------------------------------

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ValueTag {
    Null = 0,
    Bool = 1,
    I64 = 2,
    F64 = 3,
    StringRef = 4,
    BytesRef = 5,
    ArrI64 = 6,
    ArrF64 = 7,
    ArrStringRef = 8,
}

impl ValueTag {
    pub fn from_u8(x: u8) -> Option<Self> {
        Some(match x {
            0 => Self::Null,
            1 => Self::Bool,
            2 => Self::I64,
            3 => Self::F64,
            4 => Self::StringRef,
            5 => Self::BytesRef,
            6 => Self::ArrI64,
            7 => Self::ArrF64,
            8 => Self::ArrStringRef,
            _ => return None,
        })
    }
}

/// Reference to a heap entry — 8 B.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct HeapRef {
    pub page_id: u32,
    pub offset: u32,
}

const _: [(); 1] = [(); (core::mem::size_of::<HeapRef>() == 8) as usize];

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn property_page_is_4kib() {
        assert_eq!(core::mem::size_of::<PropertyPage>(), PAGE_SIZE);
    }

    #[test]
    fn fresh_page_has_magic() {
        let p = PropertyPage::new(42);
        assert_eq!(p.header.magic, PROP_PAGE_MAGIC);
        assert_eq!(p.header.node_id, 42);
    }

    #[test]
    fn crc32_roundtrip() {
        let mut p = PropertyPage::new(7);
        p.header.entry_count = 3;
        p.header.free_offset = 128;
        p.seal_crc32();
        assert!(p.verify_crc32());
        // Tamper
        p.payload[0] = 0xFF;
        assert!(!p.verify_crc32());
    }

    #[test]
    fn heap_ref_size() {
        assert_eq!(core::mem::size_of::<HeapRef>(), 8);
    }
}
