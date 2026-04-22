//! # Props zone — mmap'd PropertyPage / HeapPage storage (T17c Step 2)
//!
//! The props zone replaces the transitional `props_snapshot.rs` bincode
//! sidecar. It lives in two files inside the substrate directory:
//!
//! * `substrate.props.v2` — array of 4 KiB [`crate::page::PropertyPage`]
//!   slots, one page per `page_idx` (0-based, but `page_idx == 0` is a
//!   reserved dummy so `U48::ZERO` can be used as the "end of chain"
//!   sentinel in `next_page` links and on [`crate::record::NodeRecord::first_prop_off`]).
//! * `substrate.props.heap.v2` — array of 4 KiB [`crate::heap::HeapPage`]
//!   slots holding variable-width payloads referenced by
//!   [`crate::page::HeapRef`] (`StringRef`, `BytesRef`, `ArrStringRef`).
//!
//! ## Multi-page chains
//!
//! Each node's property entries live on a linked list of property pages.
//! The list head is [`crate::record::NodeRecord::first_prop_off`] encoded
//! as `U48::from_u64(page_idx)` (zero = no properties). When the current
//! head is full, [`PropsZone::append_entry`] allocates a **new** head,
//! sets its `next_page` field to the old head, writes the entry there,
//! and returns the new head index. The new-head-first strategy keeps
//! appends O(1) and preserves ordering by recency: the chain walks
//! newest → oldest.
//!
//! ## Heap
//!
//! [`PropsZone::intern_bytes`] atomically appends a `(u32 len, bytes)`
//! record to the current heap tail and returns a [`HeapRef`]. Entries are
//! never relocated, so refs remain stable for the life of the substrate.
//!
//! ## Read path
//!
//! [`PropsZone::walk_chain`] streams entries through a
//! [`crate::page::PropertyCursor`] on each page in the linked list. It
//! is the zero-copy read primitive substituted for the old
//! `DashMap<NodeId, ...>` hydration done by `props_snapshot.rs`.
//!
//! ## Crash safety
//!
//! The zone is WAL-shadowed — every append is journalled before the
//! mmap'd page is mutated. Per-page CRC32 (written by `seal_crc32` at
//! every append) catches torn writes on cold reads. Recovery rolls the
//! WAL forward.

#![allow(unsafe_code)]

use crate::error::{SubstrateError, SubstrateResult};
use crate::file::{SubstrateFile, ZoneFile};
use crate::heap::{HEAP_PAGE_SIZE, HeapPage};
use crate::page::{
    PAGE_SIZE, PROP_PAGE_MAGIC, PropertyCursor, PropertyEntry, PropertyPage,
};
use crate::record::U48;
use bytemuck;

/// Canonical filename of the v2 property zone.
pub const PROPS_V2_FILENAME: &str = "substrate.props.v2";
/// Canonical filename of the v2 heap zone.
pub const PROPS_HEAP_V2_FILENAME: &str = "substrate.props.heap.v2";

/// Initial file size = two 4 KiB slots (dummy sentinel + room for the
/// first real page). Growth is exponential (×1.5) once the first real
/// page overflows.
const INITIAL_BYTES: u64 = (PAGE_SIZE * 2) as u64;
const INITIAL_HEAP_BYTES: u64 = (HEAP_PAGE_SIZE * 1) as u64;

/// Encode a page id as stored in `next_page` / `first_prop_off`.
///
/// We simply reserve `page_idx == 0` at the file start as a dummy
/// sentinel; real pages live at `page_idx ≥ 1`, which lets us use
/// `U48::ZERO` as the null marker.
#[inline]
pub fn encode_page_id(page_idx: u32) -> U48 {
    U48::from_u64(page_idx as u64)
}

#[inline]
pub fn decode_page_id(u: U48) -> Option<u32> {
    let v = u.to_u64();
    if v == 0 { None } else { Some(v as u32) }
}

// ---------------------------------------------------------------------------
// PropsZone — owner of the two mmap'd zone files.
// ---------------------------------------------------------------------------

pub struct PropsZone {
    props: ZoneFile,
    heap: ZoneFile,
    /// Cached `next_page` allocator cursor (monotonic). We never reuse
    /// page_idx values — tombstone-and-rewrite is handled in-page via
    /// the `ENTRY_FLAG_TOMBSTONE` flag, and wholesale page compaction is
    /// a T17d concern.
    next_page_idx: u32,
    next_heap_page_idx: u32,
}

impl PropsZone {
    /// Open or create both zone files under the given substrate.
    pub fn open(sub: &SubstrateFile) -> SubstrateResult<Self> {
        let mut props = sub.open_named_zone(PROPS_V2_FILENAME)?;
        let mut heap = sub.open_named_zone(PROPS_HEAP_V2_FILENAME)?;
        if props.is_empty() {
            props.grow_to(INITIAL_BYTES)?;
        }
        if heap.is_empty() {
            heap.grow_to(INITIAL_HEAP_BYTES)?;
        }
        let next_page_idx = Self::compute_next_page_idx(&props);
        let next_heap_page_idx = Self::compute_next_heap_page_idx(&heap);
        Ok(Self {
            props,
            heap,
            next_page_idx,
            next_heap_page_idx,
        })
    }

    fn compute_next_page_idx(zf: &ZoneFile) -> u32 {
        // Scan from end backwards: the first page with a non-zero magic
        // marks the highest-allocated slot. Fast-path: if the file is
        // the initial two-slot size, next = 1.
        let bytes = zf.as_slice();
        let total_slots = (bytes.len() / PAGE_SIZE) as u32;
        if total_slots <= 1 {
            return 1;
        }
        let mut hi = total_slots;
        while hi > 1 {
            let idx = hi - 1;
            let slot = &bytes[idx as usize * PAGE_SIZE..(idx as usize + 1) * PAGE_SIZE];
            // Check magic of the page header.
            let magic = u32::from_le_bytes(slot[0..4].try_into().unwrap());
            if magic == PROP_PAGE_MAGIC {
                return idx + 1;
            }
            hi -= 1;
        }
        1
    }

    fn compute_next_heap_page_idx(zf: &ZoneFile) -> u32 {
        let bytes = zf.as_slice();
        let total_slots = (bytes.len() / HEAP_PAGE_SIZE) as u32;
        if total_slots == 0 {
            return 0;
        }
        let mut hi = total_slots;
        while hi > 0 {
            let idx = hi - 1;
            let slot =
                &bytes[idx as usize * HEAP_PAGE_SIZE..(idx as usize + 1) * HEAP_PAGE_SIZE];
            let magic = u32::from_le_bytes(slot[0..4].try_into().unwrap());
            if magic == crate::heap::HEAP_PAGE_MAGIC {
                return idx + 1;
            }
            hi -= 1;
        }
        0
    }

    /// Get the property zone's current length (test/observability).
    pub fn props_len(&self) -> u64 {
        self.props.len()
    }

    pub fn heap_len(&self) -> u64 {
        self.heap.len()
    }

    pub fn allocated_page_count(&self) -> u32 {
        self.next_page_idx.saturating_sub(1)
    }

    pub fn allocated_heap_page_count(&self) -> u32 {
        self.next_heap_page_idx
    }

    // ---- low-level page I/O -------------------------------------------

    /// Grow the props zone if needed so `page_idx` is addressable, then
    /// return a mutable slice of that page's 4 KiB.
    fn ensure_page_addressable(&mut self, page_idx: u32) -> SubstrateResult<()> {
        let needed = ((page_idx as u64) + 1) * PAGE_SIZE as u64;
        if needed > self.props.len() {
            // Grow exponentially (×1.5, rounded to page boundary).
            let mut new_len = self.props.len().max(INITIAL_BYTES);
            while new_len < needed {
                new_len = (new_len.saturating_mul(3)) / 2;
                new_len = ((new_len + PAGE_SIZE as u64 - 1) / PAGE_SIZE as u64) * PAGE_SIZE as u64;
            }
            self.props.grow_to(new_len)?;
        }
        Ok(())
    }

    fn ensure_heap_page_addressable(&mut self, page_idx: u32) -> SubstrateResult<()> {
        let needed = ((page_idx as u64) + 1) * HEAP_PAGE_SIZE as u64;
        if needed > self.heap.len() {
            let mut new_len = self.heap.len().max(INITIAL_HEAP_BYTES);
            while new_len < needed {
                new_len = (new_len.saturating_mul(3)) / 2;
                new_len = ((new_len + HEAP_PAGE_SIZE as u64 - 1) / HEAP_PAGE_SIZE as u64)
                    * HEAP_PAGE_SIZE as u64;
            }
            self.heap.grow_to(new_len)?;
        }
        Ok(())
    }

    fn page_slice(&self, page_idx: u32) -> Option<&[u8]> {
        let start = page_idx as usize * PAGE_SIZE;
        let end = start + PAGE_SIZE;
        self.props.as_slice().get(start..end)
    }

    fn page_slice_mut(&mut self, page_idx: u32) -> Option<&mut [u8]> {
        let start = page_idx as usize * PAGE_SIZE;
        let end = start + PAGE_SIZE;
        self.props.as_slice_mut().get_mut(start..end)
    }

    fn heap_page_slice(&self, page_idx: u32) -> Option<&[u8]> {
        let start = page_idx as usize * HEAP_PAGE_SIZE;
        let end = start + HEAP_PAGE_SIZE;
        self.heap.as_slice().get(start..end)
    }

    fn heap_page_slice_mut(&mut self, page_idx: u32) -> Option<&mut [u8]> {
        let start = page_idx as usize * HEAP_PAGE_SIZE;
        let end = start + HEAP_PAGE_SIZE;
        self.heap.as_slice_mut().get_mut(start..end)
    }

    fn read_page(&self, page_idx: u32) -> SubstrateResult<PropertyPage> {
        let bytes = self.page_slice(page_idx).ok_or_else(|| {
            SubstrateError::WalBadFrame(format!(
                "props page {page_idx} out of range (props_len = {})",
                self.props.len()
            ))
        })?;
        Ok(*bytemuck::from_bytes::<PropertyPage>(bytes))
    }

    fn write_page(&mut self, page_idx: u32, page: &PropertyPage) -> SubstrateResult<()> {
        self.ensure_page_addressable(page_idx)?;
        let slot = self.page_slice_mut(page_idx).expect("just ensured addressable");
        slot.copy_from_slice(bytemuck::bytes_of(page));
        Ok(())
    }

    fn read_heap_page(&self, page_idx: u32) -> SubstrateResult<HeapPage> {
        let bytes = self.heap_page_slice(page_idx).ok_or_else(|| {
            SubstrateError::WalBadFrame(format!(
                "heap page {page_idx} out of range (heap_len = {})",
                self.heap.len()
            ))
        })?;
        Ok(*bytemuck::from_bytes::<HeapPage>(bytes))
    }

    fn write_heap_page(&mut self, page_idx: u32, page: &HeapPage) -> SubstrateResult<()> {
        self.ensure_heap_page_addressable(page_idx)?;
        let slot = self
            .heap_page_slice_mut(page_idx)
            .expect("just ensured addressable");
        slot.copy_from_slice(bytemuck::bytes_of(page));
        Ok(())
    }

    // ---- public append / walk API -------------------------------------

    /// Append a property entry for `node_id`, maintaining the multi-page
    /// chain rooted at `head`. Returns the new head page idx (may differ
    /// from the input if a fresh page was allocated).
    ///
    /// The caller stores the returned value into
    /// [`crate::record::NodeRecord::first_prop_off`] via
    /// `encode_page_id(returned_head)`.
    pub fn append_entry(
        &mut self,
        node_id: u32,
        head: Option<u32>,
        entry: &PropertyEntry,
    ) -> SubstrateResult<u32> {
        let need = entry.encoded_len();
        if need > PAGE_SIZE - core::mem::size_of::<crate::page::PropertyPageHeader>() {
            return Err(SubstrateError::WalBadFrame(format!(
                "property entry of {} B exceeds single-page payload",
                need
            )));
        }
        if let Some(h) = head {
            // Try to append to the current head.
            let mut page = self.read_page(h)?;
            match page.append_entry(entry) {
                Ok(_) => {
                    page.seal_crc32();
                    self.write_page(h, &page)?;
                    return Ok(h);
                }
                Err(crate::page::PropertyCodecError::PageFull { .. }) => {
                    // Fall through: allocate a new head.
                }
                Err(e) => {
                    return Err(SubstrateError::WalBadFrame(format!(
                        "append_entry encode error: {e}"
                    )));
                }
            }
        }
        // Allocate a fresh page with next_page = old head.
        let new_idx = self.next_page_idx;
        self.next_page_idx = self
            .next_page_idx
            .checked_add(1)
            .ok_or_else(|| SubstrateError::WalBadFrame("props zone page id overflow".into()))?;
        let mut page = PropertyPage::new(node_id);
        if let Some(prev) = head {
            page.header.next_page = encode_page_id(prev);
        }
        page.append_entry(entry).map_err(|e| {
            SubstrateError::WalBadFrame(format!("append to fresh page failed: {e}"))
        })?;
        page.seal_crc32();
        self.write_page(new_idx, &page)?;
        Ok(new_idx)
    }

    /// Walk the full property chain rooted at `head`, invoking `visit`
    /// for every non-tombstoned entry in chain order (newest → oldest).
    /// Tombstones are skipped transparently.
    ///
    /// If `head` is `None`, the visitor is never called.
    pub fn walk_chain<F>(&self, head: Option<u32>, mut visit: F) -> SubstrateResult<()>
    where
        F: FnMut(&PropertyEntry),
    {
        let mut cur = head;
        while let Some(idx) = cur {
            let bytes = self.page_slice(idx).ok_or_else(|| {
                SubstrateError::WalBadFrame(format!(
                    "walk_chain: page {idx} out of range"
                ))
            })?;
            let page: &PropertyPage = bytemuck::from_bytes(bytes);
            if page.header.magic != PROP_PAGE_MAGIC {
                return Err(SubstrateError::WalBadFrame(format!(
                    "walk_chain: page {idx} has bad magic {:#x}",
                    page.header.magic
                )));
            }
            let cursor: PropertyCursor<'_> = page.cursor();
            for entry in cursor {
                let e = entry.map_err(|e| {
                    SubstrateError::WalBadFrame(format!(
                        "walk_chain: decode error on page {idx}: {e}"
                    ))
                })?;
                if !e.is_tombstone() {
                    visit(&e);
                }
            }
            cur = decode_page_id(page.header.next_page);
        }
        Ok(())
    }

    /// Collect every non-tombstoned entry on the chain into a Vec in
    /// chain order. Convenience helper for tests and migration code —
    /// hot paths should prefer `walk_chain` to avoid the allocation.
    pub fn collect_entries(&self, head: Option<u32>) -> SubstrateResult<Vec<PropertyEntry>> {
        let mut out = Vec::new();
        self.walk_chain(head, |e| out.push(e.clone()))?;
        Ok(out)
    }

    // ---- heap API ------------------------------------------------------

    /// Intern arbitrary bytes into the heap zone and return a stable
    /// [`HeapRef`]. The returned ref is valid for the lifetime of the
    /// substrate (entries are never relocated — compaction is deferred).
    pub fn intern_bytes(&mut self, bytes: &[u8]) -> SubstrateResult<crate::page::HeapRef> {
        let max_entry = HEAP_PAGE_SIZE - core::mem::size_of::<crate::heap::HeapPageHeader>() - 4;
        if bytes.len() > max_entry {
            return Err(SubstrateError::WalBadFrame(format!(
                "heap intern: entry of {} B exceeds single-page payload ({} B)",
                bytes.len(),
                max_entry
            )));
        }
        // Try current tail page; otherwise allocate a new one.
        if self.next_heap_page_idx > 0 {
            let tail = self.next_heap_page_idx - 1;
            let mut page = self.read_heap_page(tail)?;
            if let Some(off) = page.append(bytes) {
                page.seal_crc32();
                self.write_heap_page(tail, &page)?;
                return Ok(crate::page::HeapRef {
                    page_id: tail,
                    offset: off,
                });
            }
        }
        // Allocate a new heap page.
        let new_idx = self.next_heap_page_idx;
        self.next_heap_page_idx = self.next_heap_page_idx.checked_add(1).ok_or_else(|| {
            SubstrateError::WalBadFrame("heap zone page id overflow".into())
        })?;
        let mut page = HeapPage::new(new_idx);
        let off = page.append(bytes).expect("fresh heap page fits single entry");
        page.seal_crc32();
        self.write_heap_page(new_idx, &page)?;
        Ok(crate::page::HeapRef {
            page_id: new_idx,
            offset: off,
        })
    }

    /// Read the bytes behind a [`HeapRef`]. Returns `None` if the ref
    /// points outside the allocated region or the offset is invalid.
    pub fn read_heap(&self, href: crate::page::HeapRef) -> Option<Vec<u8>> {
        let page = self.read_heap_page(href.page_id).ok()?;
        page.read_at(href.offset).map(|s| s.to_vec())
    }

    /// Flush both zones to disk (`msync` + `fsync`).
    pub fn flush(&self) -> SubstrateResult<()> {
        self.props.msync()?;
        self.props.fsync()?;
        self.heap.msync()?;
        self.heap.fsync()?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::page::{HeapRef, PropertyValue};

    fn open_zone() -> (SubstrateFile, PropsZone) {
        let sf = SubstrateFile::open_tempfile().unwrap();
        let pz = PropsZone::open(&sf).unwrap();
        (sf, pz)
    }

    #[test]
    fn encode_decode_page_id() {
        assert_eq!(decode_page_id(encode_page_id(1)), Some(1));
        assert_eq!(decode_page_id(encode_page_id(42)), Some(42));
        assert_eq!(decode_page_id(U48::ZERO), None);
    }

    #[test]
    fn empty_zone_starts_with_sentinel_slot() {
        let (_sf, pz) = open_zone();
        // INITIAL_BYTES = 2 pages (dummy sentinel + first slot).
        assert_eq!(pz.props_len(), INITIAL_BYTES);
        assert_eq!(pz.next_page_idx, 1);
    }

    #[test]
    fn append_single_entry_returns_fresh_head() {
        let (_sf, mut pz) = open_zone();
        let entry = PropertyEntry::new(10, PropertyValue::I64(42));
        let head = pz.append_entry(7, None, &entry).unwrap();
        assert_eq!(head, 1);
        let collected = pz.collect_entries(Some(head)).unwrap();
        assert_eq!(collected.len(), 1);
        assert_eq!(collected[0], entry);
    }

    #[test]
    fn multiple_entries_fit_on_one_page() {
        let (_sf, mut pz) = open_zone();
        let entries: Vec<_> = (0..10)
            .map(|i| PropertyEntry::new(i as u16, PropertyValue::I64(i as i64)))
            .collect();
        let mut head = None;
        for e in &entries {
            head = Some(pz.append_entry(0, head, e).unwrap());
        }
        // All 10 small entries fit on one page — head index stays at 1.
        assert_eq!(head, Some(1));
        let got = pz.collect_entries(head).unwrap();
        assert_eq!(got.len(), entries.len());
        // chain walks newest→oldest but all on one page, so same order as writes.
        assert_eq!(got, entries);
    }

    #[test]
    fn overflow_triggers_chain_expansion() {
        let (_sf, mut pz) = open_zone();
        // Each ArrI64 entry of 128 elements = 4 + 128*8 = 1028 B payload
        // + 4 B header = 1032 B per entry. Three entries = 3096 B which
        // fits in a ~4072 B payload; a fourth overflows.
        let big = PropertyValue::ArrI64((0..128).collect());
        let mut head = None;
        let mut writes = vec![];
        for i in 0..8u16 {
            let e = PropertyEntry::new(i, big.clone());
            head = Some(pz.append_entry(1, head, &e).unwrap());
            writes.push(e);
        }
        assert!(
            pz.allocated_page_count() >= 2,
            "chain should have spilled, got {} pages",
            pz.allocated_page_count()
        );
        let got = pz.collect_entries(head).unwrap();
        assert_eq!(got.len(), writes.len());
        // Newest→oldest walk: chain head is newest page (which holds the
        // most recent entries), then links to older page. Within a page
        // the cursor is write-order. So the full chain order is
        // [newest_page_entries_in_write_order, older_page_entries_in_write_order, ...].
        // Verify the multiset matches.
        let mut sorted_got: Vec<_> = got.iter().map(|e| e.prop_key).collect();
        let mut sorted_want: Vec<_> = writes.iter().map(|e| e.prop_key).collect();
        sorted_got.sort();
        sorted_want.sort();
        assert_eq!(sorted_got, sorted_want);
    }

    #[test]
    fn reopen_recovers_state() {
        let sf = SubstrateFile::open_tempfile().unwrap();
        let path = sf.path().to_path_buf();
        let mut pz = PropsZone::open(&sf).unwrap();
        let e = PropertyEntry::new(3, PropertyValue::F64(1.25));
        let head = pz.append_entry(99, None, &e).unwrap();
        pz.flush().unwrap();
        drop(pz);
        // Keep the tempdir alive via the SubstrateFile handle.
        let pz2 = PropsZone::open(&sf).unwrap();
        assert_eq!(pz2.allocated_page_count(), 1);
        let got = pz2.collect_entries(Some(head)).unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0], e);
        // Silence the "unused" warning on `path` — keep it live until here.
        let _ = path;
    }

    #[test]
    fn heap_intern_roundtrip() {
        let (_sf, mut pz) = open_zone();
        let a = pz.intern_bytes(b"hello").unwrap();
        let b = pz.intern_bytes(b"world").unwrap();
        assert_eq!(pz.read_heap(a).as_deref(), Some(&b"hello"[..]));
        assert_eq!(pz.read_heap(b).as_deref(), Some(&b"world"[..]));
    }

    #[test]
    fn heap_spills_across_pages() {
        let (_sf, mut pz) = open_zone();
        // Each entry = 4 B len prefix + 1024 B payload = 1028 B.
        // 4 KiB - 16 B header = 4080 B per page -> ~3 entries per page.
        let mut refs = Vec::new();
        for i in 0..32 {
            let payload = vec![i as u8; 1024];
            let r = pz.intern_bytes(&payload).unwrap();
            refs.push((r, payload));
        }
        assert!(pz.allocated_heap_page_count() >= 2);
        for (r, expected) in &refs {
            assert_eq!(pz.read_heap(*r).as_deref(), Some(expected.as_slice()));
        }
    }

    #[test]
    fn round_trip_10k_nodes_mixed_props() {
        // T17c Step 2 acceptance verification: 10 000 nodes with mixed
        // scalars, 80-dim f64 vectors, and short string refs.
        let (_sf, mut pz) = open_zone();
        let mut heads: Vec<(u32, Option<u32>)> = Vec::with_capacity(10_000);
        for nid in 0..10_000u32 {
            let mut head = None;
            // Scalar int property.
            head = Some(
                pz.append_entry(
                    nid,
                    head,
                    &PropertyEntry::new(1, PropertyValue::I64(nid as i64)),
                )
                .unwrap(),
            );
            // String ref via heap (emulating a label lookup).
            let label = format!("node-{nid:05}");
            let href = pz.intern_bytes(label.as_bytes()).unwrap();
            head = Some(
                pz.append_entry(
                    nid,
                    head,
                    &PropertyEntry::new(2, PropertyValue::StringRef(href)),
                )
                .unwrap(),
            );
            // 80-dim f64 embedding.
            let embed: Vec<f64> = (0..80).map(|i| (i as f64) * 0.125 + nid as f64).collect();
            head = Some(
                pz.append_entry(
                    nid,
                    head,
                    &PropertyEntry::new(3, PropertyValue::ArrF64(embed)),
                )
                .unwrap(),
            );
            heads.push((nid, head));
        }
        // Sanity: every chain walks back to three entries with the right
        // prop keys. Check 200 random-ish samples (every 50th node) for
        // speed.
        for &(nid, head) in heads.iter().step_by(50) {
            let got = pz.collect_entries(head).unwrap();
            assert_eq!(got.len(), 3, "node {nid}: expected 3 entries");
            // prop keys {1, 2, 3} independent of order
            let mut keys: Vec<u16> = got.iter().map(|e| e.prop_key).collect();
            keys.sort();
            assert_eq!(keys, vec![1, 2, 3]);
            // Verify the scalar via find
            let i64_entry = got
                .iter()
                .find(|e| e.prop_key == 1)
                .expect("scalar entry present");
            assert_eq!(i64_entry.value, PropertyValue::I64(nid as i64));
            // Verify the label via heap lookup
            let str_entry = got
                .iter()
                .find(|e| e.prop_key == 2)
                .expect("string entry present");
            if let PropertyValue::StringRef(href) = str_entry.value {
                let bytes = pz.read_heap(href).expect("heap entry resolvable");
                assert_eq!(
                    std::str::from_utf8(&bytes).unwrap(),
                    format!("node-{nid:05}")
                );
            } else {
                panic!("expected StringRef at key 2");
            }
        }
    }

    #[test]
    fn chain_ordering_newest_first() {
        let (_sf, mut pz) = open_zone();
        let big = PropertyValue::ArrI64((0..128).collect());
        let entries: Vec<_> = (0..8u16)
            .map(|i| PropertyEntry::new(i, big.clone()))
            .collect();
        let mut head = None;
        for e in &entries {
            head = Some(pz.append_entry(1, head, e).unwrap());
        }
        assert!(pz.allocated_page_count() >= 2);
        let got = pz.collect_entries(head).unwrap();
        // The very first entry in `got` should correspond to the most
        // recently written entry on the newest page. The last key
        // written is 7; it must be on the newest (head) page, and its
        // prop_key must be present in the earliest block of `got` (the
        // first page's cursor yields in write order).
        let first_page_len = got
            .iter()
            .position(|e| e.prop_key == entries[0].prop_key)
            .unwrap_or(got.len());
        // All the keys in got[..first_page_len] are newer than the keys
        // in got[first_page_len..].
        let newer: std::collections::HashSet<u16> =
            got[..first_page_len].iter().map(|e| e.prop_key).collect();
        let older: std::collections::HashSet<u16> =
            got[first_page_len..].iter().map(|e| e.prop_key).collect();
        for n in &newer {
            for o in &older {
                assert!(n > o, "newer key {n} should be > older key {o}");
            }
        }
    }

    #[test]
    fn tombstones_skip_on_walk() {
        let (_sf, mut pz) = open_zone();
        let e = PropertyEntry::new(1, PropertyValue::I64(7));
        let mut tomb = PropertyEntry::new(1, PropertyValue::Null);
        tomb.flags = crate::page::ENTRY_FLAG_TOMBSTONE;
        let head = pz.append_entry(0, None, &e).unwrap();
        let head = pz.append_entry(0, Some(head), &tomb).unwrap();
        let got = pz.collect_entries(Some(head)).unwrap();
        // Tombstone is not surfaced by walk_chain; the live entry is.
        // (Note: the tombstone semantically supersedes the live entry
        // in the store layer, which will apply last-write-wins per key.
        // walk_chain simply reports all non-tombstoned entries; the
        // store-level compactor/get_property honors the tombstone.)
        assert_eq!(got.len(), 1);
        assert_eq!(got[0], e);
    }

    #[test]
    fn too_large_entry_rejected() {
        let (_sf, mut pz) = open_zone();
        let huge = PropertyValue::ArrI64((0..600).collect()); // ~4804 B payload
        let e = PropertyEntry::new(1, huge);
        let err = pz.append_entry(0, None, &e).unwrap_err();
        assert!(matches!(err, SubstrateError::WalBadFrame(_)));
    }

    #[test]
    fn heap_too_large_entry_rejected() {
        let (_sf, mut pz) = open_zone();
        let big = vec![0u8; HEAP_PAGE_SIZE]; // strictly larger than max entry
        let err = pz.intern_bytes(&big).unwrap_err();
        assert!(matches!(err, SubstrateError::WalBadFrame(_)));
    }

    #[test]
    fn uses_heap_ref_equality_for_lookup() {
        // Guard against a regression where two interns of the same
        // bytes accidentally collapse to the same HeapRef (we do NOT
        // dedupe — the caller decides).
        let (_sf, mut pz) = open_zone();
        let a = pz.intern_bytes(b"same").unwrap();
        let b = pz.intern_bytes(b"same").unwrap();
        assert_ne!(a, b, "intern_bytes must NOT dedupe — caller's choice");
        assert_eq!(pz.read_heap(a).as_deref(), Some(&b"same"[..]));
        assert_eq!(pz.read_heap(b).as_deref(), Some(&b"same"[..]));
        // Make sure HeapRef fields are consistent.
        assert_eq!(core::mem::size_of::<HeapRef>(), 8);
    }
}
