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
// PropertyEntry + PropertyValue + PropertyCursor (T17c Step 1)
// ---------------------------------------------------------------------------
//
// On-disk encoding (little-endian):
//
//   [flags: u8] [prop_key: u16 LE] [tag: u8] [payload: variable]
//
// `flags` bit 0 marks a tombstone entry (logically deleted). Tombstone
// entries still carry `prop_key` + `tag` so that compaction / iteration
// can skip them cheaply, but their payload semantics are tag-defined.
//
// Payload by tag:
//   Null         (0) → 0 bytes
//   Bool         (1) → 1 byte (0x00 / 0x01)
//   I64          (2) → 8 bytes LE
//   F64          (3) → 8 bytes LE
//   StringRef    (4) → 8 bytes HeapRef (page_id u32 LE, offset u32 LE)
//   BytesRef     (5) → 8 bytes HeapRef
//   ArrI64       (6) → 4 bytes u32 LE length + N × 8 bytes LE
//   ArrF64       (7) → 4 bytes u32 LE length + N × 8 bytes LE
//   ArrStringRef (8) → 4 bytes u32 LE length + N × 8 bytes HeapRef

/// Marks a logically deleted entry. The slot still occupies space until
/// page compaction reclaims it.
pub const ENTRY_FLAG_TOMBSTONE: u8 = 0x01;

/// Fixed prefix size before the per-tag payload: flags (1) + key (2) + tag (1).
pub const ENTRY_HEADER_SIZE: usize = 4;

/// Logical representation of a property entry's value, parameterised by
/// the 9 [`ValueTag`] variants the substrate supports on-disk. Heap-typed
/// values (strings, byte arrays, arrays thereof) carry [`HeapRef`] only —
/// payload bytes live in the companion heap pages, not inside the
/// property page.
#[derive(Clone, Debug, PartialEq)]
pub enum PropertyValue {
    Null,
    Bool(bool),
    I64(i64),
    F64(f64),
    StringRef(HeapRef),
    BytesRef(HeapRef),
    ArrI64(Vec<i64>),
    ArrF64(Vec<f64>),
    ArrStringRef(Vec<HeapRef>),
}

impl PropertyValue {
    /// Return the on-disk tag for this value.
    pub fn tag(&self) -> ValueTag {
        match self {
            Self::Null => ValueTag::Null,
            Self::Bool(_) => ValueTag::Bool,
            Self::I64(_) => ValueTag::I64,
            Self::F64(_) => ValueTag::F64,
            Self::StringRef(_) => ValueTag::StringRef,
            Self::BytesRef(_) => ValueTag::BytesRef,
            Self::ArrI64(_) => ValueTag::ArrI64,
            Self::ArrF64(_) => ValueTag::ArrF64,
            Self::ArrStringRef(_) => ValueTag::ArrStringRef,
        }
    }

    /// Number of bytes this value will occupy in the payload region.
    pub fn encoded_len(&self) -> usize {
        match self {
            Self::Null => 0,
            Self::Bool(_) => 1,
            Self::I64(_) | Self::F64(_) => 8,
            Self::StringRef(_) | Self::BytesRef(_) => 8,
            Self::ArrI64(v) => 4 + v.len() * 8,
            Self::ArrF64(v) => 4 + v.len() * 8,
            Self::ArrStringRef(v) => 4 + v.len() * 8,
        }
    }
}

/// A fully decoded property entry as it appears on-disk.
#[derive(Clone, Debug, PartialEq)]
pub struct PropertyEntry {
    pub flags: u8,
    pub prop_key: u16,
    pub value: PropertyValue,
}

impl PropertyEntry {
    pub fn new(prop_key: u16, value: PropertyValue) -> Self {
        Self {
            flags: 0,
            prop_key,
            value,
        }
    }

    pub fn tombstone(prop_key: u16, tag: ValueTag) -> Self {
        // Tombstones carry a Null payload marker regardless of original tag;
        // the original tag is preserved only for semantic replay of the
        // deletion (the reader inspects `flags` first).
        Self {
            flags: ENTRY_FLAG_TOMBSTONE,
            prop_key,
            value: match tag {
                ValueTag::Null => PropertyValue::Null,
                _ => PropertyValue::Null,
            },
        }
    }

    pub fn is_tombstone(&self) -> bool {
        self.flags & ENTRY_FLAG_TOMBSTONE != 0
    }

    /// Total bytes this entry will occupy (header + value payload).
    pub fn encoded_len(&self) -> usize {
        ENTRY_HEADER_SIZE + self.value.encoded_len()
    }
}

/// Error returned when a property entry cannot be encoded or decoded.
#[derive(Debug, PartialEq, Eq)]
pub enum PropertyCodecError {
    /// Not enough bytes left in the slice to decode a full entry.
    Truncated,
    /// The tag byte didn't correspond to any known [`ValueTag`].
    UnknownTag(u8),
    /// Not enough free space in the page to encode the entry.
    PageFull {
        needed: usize,
        available: usize,
    },
    /// An array length would overflow the remaining payload.
    InvalidArrayLen(u32),
}

impl core::fmt::Display for PropertyCodecError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Truncated => write!(f, "property entry truncated"),
            Self::UnknownTag(b) => write!(f, "unknown property value tag {:#x}", b),
            Self::PageFull { needed, available } => write!(
                f,
                "property page full: need {} B, have {} B",
                needed, available
            ),
            Self::InvalidArrayLen(n) => {
                write!(f, "property array length {} overflows payload", n)
            }
        }
    }
}

impl std::error::Error for PropertyCodecError {}

/// Encode a single [`PropertyEntry`] into `dst`, returning the number of
/// bytes written. The caller is responsible for slicing `dst` at the
/// desired offset before the call.
pub fn encode_entry(entry: &PropertyEntry, dst: &mut [u8]) -> Result<usize, PropertyCodecError> {
    let need = entry.encoded_len();
    if dst.len() < need {
        return Err(PropertyCodecError::PageFull {
            needed: need,
            available: dst.len(),
        });
    }
    dst[0] = entry.flags;
    dst[1..3].copy_from_slice(&entry.prop_key.to_le_bytes());
    dst[3] = entry.value.tag() as u8;
    let body = &mut dst[ENTRY_HEADER_SIZE..need];
    match &entry.value {
        PropertyValue::Null => {}
        PropertyValue::Bool(b) => body[0] = *b as u8,
        PropertyValue::I64(v) => body[0..8].copy_from_slice(&v.to_le_bytes()),
        PropertyValue::F64(v) => body[0..8].copy_from_slice(&v.to_le_bytes()),
        PropertyValue::StringRef(r) | PropertyValue::BytesRef(r) => {
            body[0..4].copy_from_slice(&r.page_id.to_le_bytes());
            body[4..8].copy_from_slice(&r.offset.to_le_bytes());
        }
        PropertyValue::ArrI64(v) => {
            body[0..4].copy_from_slice(&(v.len() as u32).to_le_bytes());
            for (i, x) in v.iter().enumerate() {
                let off = 4 + i * 8;
                body[off..off + 8].copy_from_slice(&x.to_le_bytes());
            }
        }
        PropertyValue::ArrF64(v) => {
            body[0..4].copy_from_slice(&(v.len() as u32).to_le_bytes());
            for (i, x) in v.iter().enumerate() {
                let off = 4 + i * 8;
                body[off..off + 8].copy_from_slice(&x.to_le_bytes());
            }
        }
        PropertyValue::ArrStringRef(v) => {
            body[0..4].copy_from_slice(&(v.len() as u32).to_le_bytes());
            for (i, r) in v.iter().enumerate() {
                let off = 4 + i * 8;
                body[off..off + 4].copy_from_slice(&r.page_id.to_le_bytes());
                body[off + 4..off + 8].copy_from_slice(&r.offset.to_le_bytes());
            }
        }
    }
    Ok(need)
}

/// Decode a single entry from `src`, returning the decoded entry and the
/// number of bytes consumed.
pub fn decode_entry(src: &[u8]) -> Result<(PropertyEntry, usize), PropertyCodecError> {
    if src.len() < ENTRY_HEADER_SIZE {
        return Err(PropertyCodecError::Truncated);
    }
    let flags = src[0];
    let prop_key = u16::from_le_bytes([src[1], src[2]]);
    let tag = ValueTag::from_u8(src[3]).ok_or(PropertyCodecError::UnknownTag(src[3]))?;
    let body = &src[ENTRY_HEADER_SIZE..];
    let (value, body_len) = match tag {
        ValueTag::Null => (PropertyValue::Null, 0usize),
        ValueTag::Bool => {
            if body.is_empty() {
                return Err(PropertyCodecError::Truncated);
            }
            (PropertyValue::Bool(body[0] != 0), 1)
        }
        ValueTag::I64 => {
            if body.len() < 8 {
                return Err(PropertyCodecError::Truncated);
            }
            let v = i64::from_le_bytes(body[0..8].try_into().unwrap());
            (PropertyValue::I64(v), 8)
        }
        ValueTag::F64 => {
            if body.len() < 8 {
                return Err(PropertyCodecError::Truncated);
            }
            let v = f64::from_le_bytes(body[0..8].try_into().unwrap());
            (PropertyValue::F64(v), 8)
        }
        ValueTag::StringRef | ValueTag::BytesRef => {
            if body.len() < 8 {
                return Err(PropertyCodecError::Truncated);
            }
            let page_id = u32::from_le_bytes(body[0..4].try_into().unwrap());
            let offset = u32::from_le_bytes(body[4..8].try_into().unwrap());
            let r = HeapRef { page_id, offset };
            let v = if tag == ValueTag::StringRef {
                PropertyValue::StringRef(r)
            } else {
                PropertyValue::BytesRef(r)
            };
            (v, 8)
        }
        ValueTag::ArrI64 | ValueTag::ArrF64 | ValueTag::ArrStringRef => {
            if body.len() < 4 {
                return Err(PropertyCodecError::Truncated);
            }
            let n = u32::from_le_bytes(body[0..4].try_into().unwrap());
            let need = 4usize.checked_add((n as usize).checked_mul(8).unwrap_or(usize::MAX))
                .unwrap_or(usize::MAX);
            if body.len() < need {
                return Err(PropertyCodecError::InvalidArrayLen(n));
            }
            let start = 4;
            let value = match tag {
                ValueTag::ArrI64 => {
                    let mut v = Vec::with_capacity(n as usize);
                    for i in 0..n as usize {
                        let off = start + i * 8;
                        v.push(i64::from_le_bytes(
                            body[off..off + 8].try_into().unwrap(),
                        ));
                    }
                    PropertyValue::ArrI64(v)
                }
                ValueTag::ArrF64 => {
                    let mut v = Vec::with_capacity(n as usize);
                    for i in 0..n as usize {
                        let off = start + i * 8;
                        v.push(f64::from_le_bytes(
                            body[off..off + 8].try_into().unwrap(),
                        ));
                    }
                    PropertyValue::ArrF64(v)
                }
                ValueTag::ArrStringRef => {
                    let mut v = Vec::with_capacity(n as usize);
                    for i in 0..n as usize {
                        let off = start + i * 8;
                        let page_id =
                            u32::from_le_bytes(body[off..off + 4].try_into().unwrap());
                        let offset =
                            u32::from_le_bytes(body[off + 4..off + 8].try_into().unwrap());
                        v.push(HeapRef { page_id, offset });
                    }
                    PropertyValue::ArrStringRef(v)
                }
                _ => unreachable!(),
            };
            (value, need)
        }
    };
    Ok((
        PropertyEntry {
            flags,
            prop_key,
            value,
        },
        ENTRY_HEADER_SIZE + body_len,
    ))
}

impl PropertyPage {
    /// Append a new entry to the page, advancing `free_offset` and
    /// incrementing the appropriate counter. Returns the byte offset at
    /// which the entry was written (relative to `payload`).
    pub fn append_entry(
        &mut self,
        entry: &PropertyEntry,
    ) -> Result<u16, PropertyCodecError> {
        let free = self.header.free_offset as usize;
        let room = self.payload.len().saturating_sub(free);
        let need = entry.encoded_len();
        if need > room {
            return Err(PropertyCodecError::PageFull {
                needed: need,
                available: room,
            });
        }
        encode_entry(entry, &mut self.payload[free..free + need])?;
        self.header.free_offset = (free + need) as u16;
        self.header.entry_count = self.header.entry_count.saturating_add(1);
        if entry.is_tombstone() {
            self.header.tombstones = self.header.tombstones.saturating_add(1);
        }
        Ok(free as u16)
    }

    /// Borrowing iterator over all entries (including tombstones) in the
    /// page, in write order.
    pub fn cursor(&self) -> PropertyCursor<'_> {
        PropertyCursor {
            buf: &self.payload[..self.header.free_offset as usize],
            pos: 0,
        }
    }
}

/// Iterator over the raw entries in a page payload. Surfaces decode
/// errors via `Result` items instead of aborting the iteration early, so
/// a single corrupt entry doesn't shadow the rest of the page.
pub struct PropertyCursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> PropertyCursor<'a> {
    /// Number of payload bytes this cursor hasn't yet consumed.
    pub fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }
}

impl<'a> Iterator for PropertyCursor<'a> {
    type Item = Result<PropertyEntry, PropertyCodecError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.buf.len() {
            return None;
        }
        match decode_entry(&self.buf[self.pos..]) {
            Ok((entry, consumed)) => {
                self.pos += consumed;
                Some(Ok(entry))
            }
            Err(e) => {
                // Ensure a single bad entry doesn't lead to infinite
                // iteration — seek to end so subsequent calls return None.
                self.pos = self.buf.len();
                Some(Err(e))
            }
        }
    }
}

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

    // -------- PropertyCursor / codec round-trips -------------------------

    fn roundtrip(entry: PropertyEntry) {
        let mut buf = vec![0u8; entry.encoded_len()];
        let n = encode_entry(&entry, &mut buf).expect("encode");
        assert_eq!(n, entry.encoded_len());
        let (decoded, consumed) = decode_entry(&buf).expect("decode");
        assert_eq!(consumed, n);
        assert_eq!(decoded, entry);
    }

    #[test]
    fn codec_null() {
        roundtrip(PropertyEntry::new(1, PropertyValue::Null));
    }

    #[test]
    fn codec_bool() {
        roundtrip(PropertyEntry::new(2, PropertyValue::Bool(true)));
        roundtrip(PropertyEntry::new(2, PropertyValue::Bool(false)));
    }

    #[test]
    fn codec_i64() {
        roundtrip(PropertyEntry::new(3, PropertyValue::I64(0)));
        roundtrip(PropertyEntry::new(3, PropertyValue::I64(i64::MIN)));
        roundtrip(PropertyEntry::new(3, PropertyValue::I64(i64::MAX)));
    }

    #[test]
    fn codec_f64() {
        roundtrip(PropertyEntry::new(4, PropertyValue::F64(0.0)));
        roundtrip(PropertyEntry::new(4, PropertyValue::F64(core::f64::consts::PI)));
        roundtrip(PropertyEntry::new(4, PropertyValue::F64(f64::NEG_INFINITY)));
    }

    #[test]
    fn codec_string_ref() {
        roundtrip(PropertyEntry::new(
            5,
            PropertyValue::StringRef(HeapRef {
                page_id: 17,
                offset: 0xABCD,
            }),
        ));
    }

    #[test]
    fn codec_bytes_ref() {
        roundtrip(PropertyEntry::new(
            6,
            PropertyValue::BytesRef(HeapRef {
                page_id: 0,
                offset: 0,
            }),
        ));
        roundtrip(PropertyEntry::new(
            6,
            PropertyValue::BytesRef(HeapRef {
                page_id: u32::MAX,
                offset: u32::MAX,
            }),
        ));
    }

    #[test]
    fn codec_arr_i64() {
        roundtrip(PropertyEntry::new(7, PropertyValue::ArrI64(vec![])));
        roundtrip(PropertyEntry::new(
            7,
            PropertyValue::ArrI64(vec![1, -1, 0, i64::MAX, i64::MIN]),
        ));
    }

    #[test]
    fn codec_arr_f64() {
        roundtrip(PropertyEntry::new(8, PropertyValue::ArrF64(vec![])));
        roundtrip(PropertyEntry::new(
            8,
            PropertyValue::ArrF64(vec![0.0, 1.5, -2.5, f64::INFINITY]),
        ));
    }

    #[test]
    fn codec_arr_string_ref() {
        roundtrip(PropertyEntry::new(
            9,
            PropertyValue::ArrStringRef(vec![
                HeapRef { page_id: 1, offset: 2 },
                HeapRef { page_id: 3, offset: 4 },
                HeapRef { page_id: 5, offset: 6 },
            ]),
        ));
    }

    #[test]
    fn tombstone_roundtrip_and_flag() {
        let mut e = PropertyEntry::new(42, PropertyValue::I64(1234));
        e.flags = ENTRY_FLAG_TOMBSTONE;
        assert!(e.is_tombstone());
        roundtrip(e);
    }

    #[test]
    fn decode_truncated_header() {
        assert_eq!(
            decode_entry(&[0u8; 3]).unwrap_err(),
            PropertyCodecError::Truncated
        );
    }

    #[test]
    fn decode_unknown_tag() {
        let buf = [0u8, 0, 0, 0xFE];
        match decode_entry(&buf).unwrap_err() {
            PropertyCodecError::UnknownTag(0xFE) => {}
            e => panic!("wrong error: {:?}", e),
        }
    }

    #[test]
    fn decode_truncated_payload() {
        // I64 entry header but only 4 body bytes.
        let buf = [0u8, 0x01, 0x00, ValueTag::I64 as u8, 1, 2, 3, 4];
        assert_eq!(
            decode_entry(&buf).unwrap_err(),
            PropertyCodecError::Truncated
        );
    }

    #[test]
    fn decode_invalid_array_len() {
        // ArrI64 with declared length 10⁸ but empty body.
        let mut buf = vec![0u8, 0x00, 0x00, ValueTag::ArrI64 as u8];
        buf.extend_from_slice(&100_000_000u32.to_le_bytes());
        match decode_entry(&buf).unwrap_err() {
            PropertyCodecError::InvalidArrayLen(n) => assert_eq!(n, 100_000_000),
            e => panic!("wrong error: {:?}", e),
        }
    }

    #[test]
    fn page_append_and_cursor() {
        let mut p = PropertyPage::new(100);
        let entries = vec![
            PropertyEntry::new(1, PropertyValue::I64(42)),
            PropertyEntry::new(
                2,
                PropertyValue::StringRef(HeapRef {
                    page_id: 1,
                    offset: 16,
                }),
            ),
            {
                let mut t = PropertyEntry::new(1, PropertyValue::Null);
                t.flags = ENTRY_FLAG_TOMBSTONE;
                t
            },
            PropertyEntry::new(3, PropertyValue::Bool(true)),
        ];
        for e in &entries {
            p.append_entry(e).expect("append");
        }
        assert_eq!(p.header.entry_count as usize, entries.len());
        assert_eq!(p.header.tombstones, 1);
        let collected: Vec<_> = p
            .cursor()
            .map(|r| r.expect("decode"))
            .collect();
        assert_eq!(collected, entries);
    }

    #[test]
    fn page_append_full_returns_err() {
        let mut p = PropertyPage::new(0);
        // Enough small entries to nearly fill the payload (~4 kB).
        let mut pushed = 0usize;
        loop {
            let e = PropertyEntry::new(0, PropertyValue::I64(pushed as i64));
            if p.append_entry(&e).is_err() {
                break;
            }
            pushed += 1;
        }
        // Next append must fail with PageFull
        let e = PropertyEntry::new(
            0,
            PropertyValue::ArrI64(vec![0; 10_000]), // huge
        );
        match p.append_entry(&e).unwrap_err() {
            PropertyCodecError::PageFull { .. } => {}
            other => panic!("unexpected err {:?}", other),
        }
        assert!(pushed > 0);
    }

    #[test]
    fn crc_detects_entry_tampering() {
        let mut p = PropertyPage::new(123);
        p.append_entry(&PropertyEntry::new(1, PropertyValue::I64(7)))
            .unwrap();
        p.append_entry(&PropertyEntry::new(
            2,
            PropertyValue::StringRef(HeapRef {
                page_id: 1,
                offset: 1,
            }),
        ))
        .unwrap();
        p.seal_crc32();
        assert!(p.verify_crc32());
        // Tamper within a written entry (flip a payload byte).
        p.payload[5] ^= 0xFF;
        assert!(!p.verify_crc32());
    }

    #[test]
    fn cursor_stops_on_decode_error() {
        // First entry OK (I64 = 42), then garbage (unknown tag 0xFF).
        let ok = PropertyEntry::new(1, PropertyValue::I64(42));
        let mut p = PropertyPage::new(0);
        p.append_entry(&ok).unwrap();
        // Write a bad tag into the payload after the first entry.
        let pos = p.header.free_offset as usize;
        p.payload[pos] = 0;
        p.payload[pos + 1] = 0;
        p.payload[pos + 2] = 0;
        p.payload[pos + 3] = 0xFF; // invalid tag
        p.header.free_offset += 4;
        let mut cur = p.cursor();
        let first = cur.next().unwrap().unwrap();
        assert_eq!(first, ok);
        let second = cur.next().unwrap();
        assert!(matches!(second, Err(PropertyCodecError::UnknownTag(0xFF))));
        assert!(cur.next().is_none());
    }

    #[test]
    fn all_value_tags_covered_by_cursor() {
        // Regression guard: every ValueTag variant round-trips via cursor.
        let all: Vec<PropertyEntry> = vec![
            PropertyEntry::new(0, PropertyValue::Null),
            PropertyEntry::new(1, PropertyValue::Bool(true)),
            PropertyEntry::new(2, PropertyValue::I64(-1)),
            PropertyEntry::new(3, PropertyValue::F64(1.25)),
            PropertyEntry::new(
                4,
                PropertyValue::StringRef(HeapRef { page_id: 1, offset: 2 }),
            ),
            PropertyEntry::new(
                5,
                PropertyValue::BytesRef(HeapRef { page_id: 3, offset: 4 }),
            ),
            PropertyEntry::new(6, PropertyValue::ArrI64(vec![1, 2, 3])),
            PropertyEntry::new(7, PropertyValue::ArrF64(vec![1.0, 2.0])),
            PropertyEntry::new(
                8,
                PropertyValue::ArrStringRef(vec![HeapRef {
                    page_id: 9,
                    offset: 10,
                }]),
            ),
        ];
        let mut p = PropertyPage::new(0);
        for e in &all {
            p.append_entry(e).unwrap();
        }
        let got: Vec<_> = p.cursor().map(|r| r.unwrap()).collect();
        assert_eq!(got, all);
        // Tag coverage sanity.
        let mut tags: Vec<u8> = got.iter().map(|e| e.value.tag() as u8).collect();
        tags.sort();
        assert_eq!(tags, (0u8..=8).collect::<Vec<_>>());
    }
}
