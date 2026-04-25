//! Flat, mmap-safe types for the native `.obrain` v2 format.
//!
//! All types in this module are `#[repr(C)]` with fixed sizes and no pointers.
//! They can be directly cast from a memory-mapped `&[u8]` slice with zero
//! deserialization cost.
//!
//! # Safety
//!
//! These types implement the [`Pod`] marker trait, indicating they are safe to
//! interpret from arbitrary bytes (no padding-dependent invariants, no pointers).
//! Use [`cast_slice`] for safe casting from raw bytes.

use std::mem;

// ─────────────────────────────────────────────────────────────────────
// Pod trait — marker for types safe to reinterpret from raw bytes
// ─────────────────────────────────────────────────────────────────────

/// Marker trait for plain-old-data types that can be safely cast from `&[u8]`.
///
/// # Safety
///
/// Implementors must:
/// - Be `#[repr(C)]` or `#[repr(transparent)]`
/// - Have no padding bytes that could cause UB when read
/// - Contain no pointers, references, or non-Pod fields
/// - Be valid for any bit pattern (no discriminant invariants)
pub unsafe trait Pod: Copy + 'static {}

// Primitives
unsafe impl Pod for u8 {}
unsafe impl Pod for u16 {}
unsafe impl Pod for u32 {}
unsafe impl Pod for u64 {}
unsafe impl Pod for i64 {}
unsafe impl Pod for f32 {}
unsafe impl Pod for f64 {}

// ─────────────────────────────────────────────────────────────────────
// cast_slice — safe zero-copy cast from &[u8] to &[T]
// ─────────────────────────────────────────────────────────────────────

/// Error type for [`cast_slice`] failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CastError {
    /// The byte slice is not properly aligned for type `T`.
    Alignment {
        /// Required alignment.
        required: usize,
        /// Actual alignment of the pointer.
        actual: usize,
    },
    /// The byte slice length is not a multiple of `size_of::<T>()`.
    Size {
        /// Size of one element.
        type_size: usize,
        /// Actual byte slice length.
        slice_len: usize,
    },
}

impl std::fmt::Display for CastError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CastError::Alignment { required, actual } => {
                write!(f, "alignment error: required {required}, got {actual}")
            }
            CastError::Size {
                type_size,
                slice_len,
            } => {
                write!(
                    f,
                    "size error: type size {type_size}, slice len {slice_len} (not a multiple)"
                )
            }
        }
    }
}

impl std::error::Error for CastError {}

/// Safely reinterpret a byte slice as a slice of `T: Pod`.
///
/// Returns an error if the pointer is not aligned or the length is not a
/// multiple of `size_of::<T>()`.
///
/// # Examples
///
/// ```
/// use obrain_common::types::flat::{cast_slice, NodeSlot};
///
/// // In practice, bytes come from mmap — here we use a Vec for demo
/// let slot = NodeSlot::new(42, 0x01, 100, 5);
/// let bytes: &[u8] = unsafe {
///     std::slice::from_raw_parts(
///         &slot as *const NodeSlot as *const u8,
///         std::mem::size_of::<NodeSlot>(),
///     )
/// };
/// let slots = cast_slice::<NodeSlot>(bytes).unwrap();
/// assert_eq!(slots.len(), 1);
/// assert_eq!(slots[0].node_id(), 42);
/// ```
pub fn cast_slice<T: Pod>(bytes: &[u8]) -> Result<&[T], CastError> {
    let type_size = mem::size_of::<T>();
    let align = mem::align_of::<T>();

    // Check alignment
    let ptr = bytes.as_ptr() as usize;
    if ptr % align != 0 {
        return Err(CastError::Alignment {
            required: align,
            actual: ptr % align,
        });
    }

    // Check size
    if type_size == 0 {
        // ZSTs: any slice works, return empty
        return Ok(&[]);
    }
    if bytes.len() % type_size != 0 {
        return Err(CastError::Size {
            type_size,
            slice_len: bytes.len(),
        });
    }

    let count = bytes.len() / type_size;
    // SAFETY: We've verified alignment and size. T: Pod guarantees no
    // invalid bit patterns.
    Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const T, count) })
}

/// Like [`cast_slice`] but returns a single `&T` from the start of the slice.
pub fn cast_ref<T: Pod>(bytes: &[u8]) -> Result<&T, CastError> {
    let type_size = mem::size_of::<T>();
    let align = mem::align_of::<T>();

    let ptr = bytes.as_ptr() as usize;
    if ptr % align != 0 {
        return Err(CastError::Alignment {
            required: align,
            actual: ptr % align,
        });
    }
    if bytes.len() < type_size {
        return Err(CastError::Size {
            type_size,
            slice_len: bytes.len(),
        });
    }

    // SAFETY: alignment and size verified, T: Pod.
    Ok(unsafe { &*(bytes.as_ptr() as *const T) })
}

/// Convert a `Pod` value to its raw byte representation.
pub fn as_bytes<T: Pod>(value: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts(value as *const T as *const u8, mem::size_of::<T>()) }
}

/// Convert a slice of `Pod` values to raw bytes.
pub fn slice_as_bytes<T: Pod>(values: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            values.as_ptr() as *const u8,
            values.len() * mem::size_of::<T>(),
        )
    }
}

// ─────────────────────────────────────────────────────────────────────
// StringRef — reference into the StringTable
// ─────────────────────────────────────────────────────────────────────

/// A reference to a string in the StringTable section.
///
/// Strings are stored contiguously in a single byte buffer. A `StringRef`
/// points to a substring via `(offset, len)`. The maximum string length
/// is 4 GiB, which is more than sufficient.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub struct StringRef {
    /// Byte offset into the StringTable.
    pub offset: u32,
    /// Byte length of the string.
    pub len: u32,
}

unsafe impl Pod for StringRef {}

const _: () = assert!(mem::size_of::<StringRef>() == 8);

impl StringRef {
    /// A null/empty string reference.
    pub const EMPTY: Self = Self { offset: 0, len: 0 };

    /// Creates a new string reference.
    #[inline]
    #[must_use]
    pub const fn new(offset: u32, len: u32) -> Self {
        Self { offset, len }
    }

    /// Returns `true` if this is an empty string reference.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Resolves this reference against a StringTable byte buffer.
    ///
    /// Returns `None` if the reference is out of bounds or not valid UTF-8.
    #[inline]
    pub fn resolve<'a>(&self, string_table: &'a [u8]) -> Option<&'a str> {
        let start = self.offset as usize;
        let end = start + self.len as usize;
        string_table
            .get(start..end)
            .and_then(|bytes| std::str::from_utf8(bytes).ok())
    }
}

// ─────────────────────────────────────────────────────────────────────
// NodeSlot — fixed-size node record, indexed by NodeId.0
// ─────────────────────────────────────────────────────────────────────

/// Fixed-size on-disk node record. Indexed directly by `NodeId.0`.
///
/// The `label_mask` is a bitmap supporting up to 64 distinct labels per
/// database (sufficient for virtually all use cases). Each bit position
/// corresponds to a label ID from the label catalog.
///
/// Properties are stored separately in the Properties section.
/// `prop_offset` and `prop_count` point into the per-node PropEntry array.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(C)]
pub struct NodeSlot {
    /// Node ID (redundant with array index, used for validation).
    pub id: u64,
    /// Bitmap of label IDs (bit N = has label N). Supports up to 64 labels.
    pub label_mask: u64,
    /// Byte offset into the Properties section for this node's PropEntries.
    pub prop_offset: u32,
    /// Number of properties on this node.
    pub prop_count: u16,
    /// Flags: bit 0 = deleted.
    pub flags: u16,
}

unsafe impl Pod for NodeSlot {}

const _: () = assert!(mem::size_of::<NodeSlot>() == 24);

impl NodeSlot {
    /// Flag: this node has been deleted.
    pub const FLAG_DELETED: u16 = 0x01;

    /// Creates a new node slot.
    #[inline]
    #[must_use]
    pub const fn new(id: u64, label_mask: u64, prop_offset: u32, prop_count: u16) -> Self {
        Self {
            id,
            label_mask,
            prop_offset,
            prop_count,
            flags: 0,
        }
    }

    /// Returns the node ID.
    #[inline]
    #[must_use]
    pub const fn node_id(&self) -> u64 {
        self.id
    }

    /// Returns `true` if this slot is marked deleted.
    #[inline]
    #[must_use]
    pub const fn is_deleted(&self) -> bool {
        self.flags & Self::FLAG_DELETED != 0
    }

    /// Checks if this node has a given label ID.
    #[inline]
    #[must_use]
    pub const fn has_label(&self, label_id: u32) -> bool {
        if label_id >= 64 {
            return false;
        }
        self.label_mask & (1u64 << label_id) != 0
    }

    /// Returns an iterator over the label IDs set in the mask.
    pub fn label_ids(&self) -> impl Iterator<Item = u32> + '_ {
        (0u32..64).filter(|&bit| self.label_mask & (1u64 << bit) != 0)
    }
}

// ─────────────────────────────────────────────────────────────────────
// EdgeSlot — fixed-size edge record, indexed by EdgeId.0
// ─────────────────────────────────────────────────────────────────────

/// Fixed-size on-disk edge record. Indexed directly by `EdgeId.0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(C)]
pub struct EdgeSlot {
    /// Edge ID (redundant with array index, used for validation).
    pub id: u64,
    /// Source node ID.
    pub src: u64,
    /// Destination node ID.
    pub dst: u64,
    /// Edge type as a reference into the StringTable.
    pub type_ref: StringRef,
    /// Byte offset into the Properties section for this edge's PropEntries.
    pub prop_offset: u32,
    /// Number of properties on this edge.
    pub prop_count: u16,
    /// Flags: bit 0 = deleted.
    pub flags: u16,
    /// Reserved padding to reach 48 bytes (cache-friendly size).
    pub _pad: [u8; 8],
}

unsafe impl Pod for EdgeSlot {}

const _: () = assert!(mem::size_of::<EdgeSlot>() == 48);

impl EdgeSlot {
    /// Flag: this edge has been deleted.
    pub const FLAG_DELETED: u16 = 0x01;

    /// Creates a new edge slot.
    #[inline]
    #[must_use]
    pub const fn new(
        id: u64,
        src: u64,
        dst: u64,
        type_ref: StringRef,
        prop_offset: u32,
        prop_count: u16,
    ) -> Self {
        Self {
            id,
            src,
            dst,
            type_ref,
            prop_offset,
            prop_count,
            flags: 0,
            _pad: [0; 8],
        }
    }

    /// Returns the edge ID.
    #[inline]
    #[must_use]
    pub const fn edge_id(&self) -> u64 {
        self.id
    }

    /// Returns the source node ID.
    #[inline]
    #[must_use]
    pub const fn src_id(&self) -> u64 {
        self.src
    }

    /// Returns the destination node ID.
    #[inline]
    #[must_use]
    pub const fn dst_id(&self) -> u64 {
        self.dst
    }

    /// Returns `true` if this slot is marked deleted.
    #[inline]
    #[must_use]
    pub const fn is_deleted(&self) -> bool {
        self.flags & Self::FLAG_DELETED != 0
    }
}

// ─────────────────────────────────────────────────────────────────────
// PropEntry — property descriptor in the Properties section
// ─────────────────────────────────────────────────────────────────────

/// Describes a single property on a node or edge.
///
/// The actual value bytes live in the Values section; this entry stores
/// the key reference and a pointer to the value data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(C)]
pub struct PropEntry {
    /// Property key as a reference into the StringTable.
    pub key_ref: StringRef,
    /// Value type discriminant (see [`ValueType`]).
    pub value_type: u8,
    /// Padding for alignment.
    pub _pad: [u8; 3],
    /// Byte offset into the Values section where the value data starts.
    pub value_offset: u32,
    /// Byte length of the value data.
    pub value_len: u32,
    /// Reserved padding to reach 24 bytes.
    pub _pad2: [u8; 4],
}

unsafe impl Pod for PropEntry {}

const _: () = assert!(mem::size_of::<PropEntry>() == 24);

impl PropEntry {
    /// Creates a new property entry.
    #[inline]
    #[must_use]
    pub const fn new(
        key_ref: StringRef,
        value_type: u8,
        value_offset: u32,
        value_len: u32,
    ) -> Self {
        Self {
            key_ref,
            value_type,
            _pad: [0; 3],
            value_offset,
            value_len,
            _pad2: [0; 4],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// ValueType — discriminant for property values
// ─────────────────────────────────────────────────────────────────────

/// Discriminant for property value types stored in [`PropEntry::value_type`].
///
/// These match the `Value` enum variants in obrain-common.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueType {
    /// No value.
    Null = 0,
    /// Boolean.
    Bool = 1,
    /// Signed 64-bit integer.
    Int64 = 2,
    /// 64-bit IEEE 754 float.
    Float64 = 3,
    /// UTF-8 string (stored as StringRef in values section).
    String = 4,
    /// Raw byte array.
    Bytes = 5,
    /// Nanosecond-precision timestamp.
    Timestamp = 6,
    /// Calendar date.
    Date = 7,
    /// Time of day.
    Time = 8,
    /// Duration / interval.
    Duration = 9,
    /// Timezone-aware datetime.
    ZonedDatetime = 10,
    /// Heterogeneous list.
    List = 11,
    /// Key-value map.
    Map = 12,
    /// Float vector (embeddings).
    Vector = 13,
    /// Graph path.
    Path = 14,
}

impl ValueType {
    /// Converts a raw `u8` to a `ValueType`, returning `None` for unknown values.
    #[inline]
    #[must_use]
    pub const fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Null),
            1 => Some(Self::Bool),
            2 => Some(Self::Int64),
            3 => Some(Self::Float64),
            4 => Some(Self::String),
            5 => Some(Self::Bytes),
            6 => Some(Self::Timestamp),
            7 => Some(Self::Date),
            8 => Some(Self::Time),
            9 => Some(Self::Duration),
            10 => Some(Self::ZonedDatetime),
            11 => Some(Self::List),
            12 => Some(Self::Map),
            13 => Some(Self::Vector),
            14 => Some(Self::Path),
            _ => None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size_assertions() {
        assert_eq!(mem::size_of::<StringRef>(), 8);
        assert_eq!(mem::size_of::<NodeSlot>(), 24);
        assert_eq!(mem::size_of::<EdgeSlot>(), 48);
        assert_eq!(mem::size_of::<PropEntry>(), 24);
    }

    #[test]
    fn test_alignment() {
        assert_eq!(mem::align_of::<StringRef>(), 4);
        assert_eq!(mem::align_of::<NodeSlot>(), 8);
        assert_eq!(mem::align_of::<EdgeSlot>(), 8);
        assert_eq!(mem::align_of::<PropEntry>(), 4);
    }

    #[test]
    fn test_node_slot_roundtrip() {
        let slot = NodeSlot::new(42, 0b101, 1024, 3);
        let bytes = as_bytes(&slot);
        assert_eq!(bytes.len(), 24);

        let recovered = cast_ref::<NodeSlot>(bytes).unwrap();
        assert_eq!(recovered.node_id(), 42);
        assert_eq!(recovered.label_mask, 0b101);
        assert!(recovered.has_label(0));
        assert!(!recovered.has_label(1));
        assert!(recovered.has_label(2));
        assert!(!recovered.is_deleted());
    }

    #[test]
    fn test_edge_slot_roundtrip() {
        let type_ref = StringRef::new(100, 5);
        let slot = EdgeSlot::new(7, 10, 20, type_ref, 2048, 2);
        let bytes = as_bytes(&slot);
        assert_eq!(bytes.len(), 48);

        let recovered = cast_ref::<EdgeSlot>(bytes).unwrap();
        assert_eq!(recovered.edge_id(), 7);
        assert_eq!(recovered.src_id(), 10);
        assert_eq!(recovered.dst_id(), 20);
        assert_eq!(recovered.type_ref, type_ref);
        assert!(!recovered.is_deleted());
    }

    #[test]
    fn test_string_ref_resolve() {
        let table = b"hello world!";
        let sref = StringRef::new(6, 6); // "world!"
        assert_eq!(sref.resolve(table), Some("world!"));

        let empty = StringRef::EMPTY;
        assert_eq!(empty.resolve(table), Some(""));
        assert!(empty.is_empty());
    }

    #[test]
    fn test_cast_slice_multiple() {
        let slots = [
            NodeSlot::new(0, 0x01, 0, 1),
            NodeSlot::new(1, 0x02, 24, 2),
            NodeSlot::new(2, 0x03, 72, 3),
        ];
        let bytes = slice_as_bytes(&slots);
        assert_eq!(bytes.len(), 72); // 3 × 24

        let recovered = cast_slice::<NodeSlot>(bytes).unwrap();
        assert_eq!(recovered.len(), 3);
        assert_eq!(recovered[0].node_id(), 0);
        assert_eq!(recovered[1].node_id(), 1);
        assert_eq!(recovered[2].node_id(), 2);
    }

    #[test]
    fn test_cast_slice_alignment_error() {
        let data = [0u8; 32];
        // Offset by 1 byte → misaligned for NodeSlot (align=8)
        let misaligned = &data[1..25];
        let result = cast_slice::<NodeSlot>(misaligned);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), CastError::Alignment { .. }));
    }

    #[test]
    fn test_cast_slice_size_error() {
        let data = [0u8; 25]; // 25 is not a multiple of 24
        // Use aligned memory
        let aligned: Vec<u64> = vec![0; 4]; // 32 bytes, aligned to 8
        let bytes = &unsafe { std::slice::from_raw_parts(aligned.as_ptr() as *const u8, 32) }[..25];
        let result = cast_slice::<NodeSlot>(bytes);
        let _ = data; // suppress unused
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), CastError::Size { .. }));
    }

    #[test]
    fn test_node_slot_label_ids() {
        let slot = NodeSlot::new(0, 0b1010_0101, 0, 0);
        let ids: Vec<u32> = slot.label_ids().collect();
        assert_eq!(ids, vec![0, 2, 5, 7]);
    }

    #[test]
    fn test_prop_entry_roundtrip() {
        let entry = PropEntry::new(StringRef::new(50, 4), ValueType::Vector as u8, 8192, 1536);
        let bytes = as_bytes(&entry);
        assert_eq!(bytes.len(), 24);

        let recovered = cast_ref::<PropEntry>(bytes).unwrap();
        assert_eq!(recovered.key_ref, StringRef::new(50, 4));
        assert_eq!(
            ValueType::from_u8(recovered.value_type),
            Some(ValueType::Vector)
        );
        assert_eq!(recovered.value_offset, 8192);
        assert_eq!(recovered.value_len, 1536);
    }

    #[test]
    fn test_value_type_from_u8() {
        assert_eq!(ValueType::from_u8(0), Some(ValueType::Null));
        assert_eq!(ValueType::from_u8(13), Some(ValueType::Vector));
        assert_eq!(ValueType::from_u8(14), Some(ValueType::Path));
        assert_eq!(ValueType::from_u8(15), None);
        assert_eq!(ValueType::from_u8(255), None);
    }
}
