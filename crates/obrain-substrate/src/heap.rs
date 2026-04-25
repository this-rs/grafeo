//! String/bytes/array heap — 4 KiB append-only pages with tombstones.
//!
//! See `docs/rfc/substrate/format-spec.md` §6 for the layout.

use bytemuck::{Pod, Zeroable};

/// Magic marker for [`HeapPage`] (FNV-1a truncated for "heap").
pub const HEAP_PAGE_MAGIC: u32 = 0xF1EA_95FF;

pub const HEAP_PAGE_SIZE: usize = 4096;

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct HeapPageHeader {
    pub magic: u32,       // @0..4
    pub page_id: u32,     // @4..8
    pub crc32: u32,       // @8..12
    pub entry_count: u16, // @12..14
    pub free_offset: u16, // @14..16
}

const _: [(); 1] = [(); (core::mem::size_of::<HeapPageHeader>() == 16) as usize];

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct HeapPage {
    pub header: HeapPageHeader,
    pub payload: [u8; HEAP_PAGE_SIZE - core::mem::size_of::<HeapPageHeader>()],
}

const _: [(); 1] = [(); (core::mem::size_of::<HeapPage>() == HEAP_PAGE_SIZE) as usize];

impl Default for HeapPage {
    fn default() -> Self {
        Self {
            header: HeapPageHeader::default(),
            payload: [0; HEAP_PAGE_SIZE - core::mem::size_of::<HeapPageHeader>()],
        }
    }
}

impl core::fmt::Debug for HeapPage {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("HeapPage")
            .field("header", &self.header)
            .field("payload_len", &self.payload.len())
            .finish()
    }
}

impl HeapPage {
    pub fn new(page_id: u32) -> Self {
        let mut p = Self::default();
        p.header.magic = HEAP_PAGE_MAGIC;
        p.header.page_id = page_id;
        p.header.free_offset = 0;
        p
    }

    /// Append a `&[u8]` entry. Returns the offset within `payload` where the entry starts,
    /// or `None` if the page is full.
    pub fn append(&mut self, bytes: &[u8]) -> Option<u32> {
        let entry_len = 4 + bytes.len(); // 4 B length prefix + payload
        let cursor = self.header.free_offset as usize;
        if cursor + entry_len > self.payload.len() {
            return None;
        }
        // Write u32 LE length prefix, then the bytes.
        let len_bytes = (bytes.len() as u32).to_le_bytes();
        self.payload[cursor..cursor + 4].copy_from_slice(&len_bytes);
        self.payload[cursor + 4..cursor + 4 + bytes.len()].copy_from_slice(bytes);
        let offset = cursor as u32;
        self.header.free_offset = (cursor + entry_len) as u16;
        self.header.entry_count = self.header.entry_count.saturating_add(1);
        Some(offset)
    }

    /// Read an entry at a given offset. Returns `None` if the offset is past the free region
    /// or the claimed length would overflow the page.
    pub fn read_at(&self, offset: u32) -> Option<&[u8]> {
        let off = offset as usize;
        if off + 4 > self.header.free_offset as usize {
            return None;
        }
        let mut len_bytes = [0u8; 4];
        len_bytes.copy_from_slice(&self.payload[off..off + 4]);
        let len = u32::from_le_bytes(len_bytes) as usize;
        if off + 4 + len > self.payload.len() {
            return None;
        }
        Some(&self.payload[off + 4..off + 4 + len])
    }

    pub fn compute_crc32(&self) -> u32 {
        let mut h = crc32fast::Hasher::new();
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
// StringHeap — in-memory writer + reader over a Vec<HeapPage>.
// This is the "build path" used by migration; the runtime uses mmap'd heap pages.
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct StringHeap {
    pub pages: Vec<HeapPage>,
}

impl StringHeap {
    pub fn new() -> Self {
        Self { pages: vec![] }
    }

    /// Intern the given bytes. Returns `(page_id, offset)` suitable for a [`crate::page::HeapRef`].
    pub fn intern(&mut self, bytes: &[u8]) -> (u32, u32) {
        // Try the last page first.
        if let Some(last) = self.pages.last_mut() {
            if let Some(off) = last.append(bytes) {
                return (last.header.page_id, off);
            }
        }
        // Allocate a new page.
        let page_id = self.pages.len() as u32;
        let mut page = HeapPage::new(page_id);
        // If the payload is too big to fit in a single fresh page, we fail loudly.
        let off = page
            .append(bytes)
            .expect("string-heap: single entry larger than heap page payload");
        self.pages.push(page);
        (page_id, off)
    }

    pub fn get(&self, page_id: u32, offset: u32) -> Option<&[u8]> {
        self.pages
            .get(page_id as usize)
            .and_then(|p| p.read_at(offset))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heap_page_is_4kib() {
        assert_eq!(core::mem::size_of::<HeapPage>(), HEAP_PAGE_SIZE);
    }

    #[test]
    fn append_and_read() {
        let mut p = HeapPage::new(0);
        let off1 = p.append(b"hello").unwrap();
        let off2 = p.append(b"world").unwrap();
        assert_eq!(p.read_at(off1), Some(&b"hello"[..]));
        assert_eq!(p.read_at(off2), Some(&b"world"[..]));
        assert_eq!(p.header.entry_count, 2);
    }

    #[test]
    fn fills_and_rejects() {
        let mut p = HeapPage::new(0);
        let big = vec![0u8; HEAP_PAGE_SIZE - 16 - 4]; // exactly fills
        assert!(p.append(&big).is_some());
        // Next append must fail.
        assert!(p.append(b"x").is_none());
    }

    #[test]
    fn crc32_roundtrip() {
        let mut p = HeapPage::new(0);
        p.append(b"abc").unwrap();
        p.seal_crc32();
        assert!(p.verify_crc32());
        p.payload[0] = 0xFF;
        assert!(!p.verify_crc32());
    }

    #[test]
    fn string_heap_spills_to_new_page() {
        let mut h = StringHeap::new();
        // Force spill by interning many small entries.
        let mut refs = vec![];
        for i in 0..2000 {
            let s = format!("value-{i:05}");
            refs.push((h.intern(s.as_bytes()), s));
        }
        // Should have spilled to > 1 page.
        assert!(h.pages.len() > 1, "expected spill, got {}", h.pages.len());
        for ((page_id, off), s) in refs {
            let got = h.get(page_id, off).unwrap();
            assert_eq!(got, s.as_bytes());
        }
    }

    #[test]
    fn read_at_out_of_bounds_returns_none() {
        let p = HeapPage::new(0);
        assert!(p.read_at(0).is_none());
        assert!(p.read_at(99_999).is_none());
    }
}
