//! Read-only graph store backed by a memory-mapped `.obrain` v2 file.
//!
//! `MmapStore` implements [`GraphStore`] by reading directly from mmap'd
//! flat arrays. No deserialization, no HashMap — open is instant.
//!
//! All data access is zero-copy: `get_node()`, `neighbors()`, `get_node_property()`
//! read from the mmap'd region without allocation (except for constructing the
//! returned `Node`/`Edge` objects).

use std::sync::Arc;

use arcstr::ArcStr;
use memmap2::Mmap;
use obrain_common::types::flat::{
    cast_slice, EdgeSlot, NodeSlot, PropEntry, StringRef, ValueType,
};
use obrain_common::types::{
    Date, Duration, EdgeId, EpochId, NodeId, PropertyKey, PropertyMap, Time, Timestamp,
    TransactionId, Value, ZonedDatetime,
};
use obrain_common::utils::hash::FxHashMap;

use obrain_adapters::storage::file::toc::{FileToc, SectionType, TocError};

use obrain_core::graph::lpg::{CompareOp, Edge, Node};
use obrain_core::graph::traits::GraphStore;
use obrain_core::graph::Direction;
use obrain_core::statistics::Statistics;

use super::nocache_reader::NocacheStringReader;

// ─────────────────────────────────────────────────────────────────────
// MmapStore
// ─────────────────────────────────────────────────────────────────────

/// A read-only graph store backed by a memory-mapped v2 file.
///
/// All reads are zero-copy from the mmap. Mutations are not supported;
/// to mutate, the caller must materialize into an `LpgStore` first.
pub struct MmapStore {
    /// The memory-mapped file. Must be kept alive as long as slices reference it.
    _mmap: Mmap,

    // ── Parsed sections (zero-copy borrows into _mmap) ──

    /// StringTable raw bytes.
    strings: *const [u8],
    /// Node slots, indexed by NodeId.0.
    nodes: *const [NodeSlot],
    /// Edge slots, indexed by EdgeId.0.
    edges: *const [EdgeSlot],
    /// Property entries for all nodes and edges.
    props: *const [PropEntry],
    /// Property values raw bytes.
    prop_values: *const [u8],
    /// CSR forward adjacency offsets (len = node_count + 1).
    csr_fwd_offsets: *const [u64],
    /// CSR forward adjacency targets.
    csr_fwd_targets: *const [u64],
    /// CSR forward edge IDs (parallel to targets).
    csr_fwd_edge_ids: *const [u64],
    /// CSR backward adjacency offsets.
    csr_bwd_offsets: *const [u64],
    /// CSR backward adjacency targets.
    csr_bwd_targets: *const [u64],
    /// CSR backward edge IDs.
    csr_bwd_edge_ids: *const [u64],
    /// Label catalog: label_id → label name.
    label_names: Vec<ArcStr>,
    /// Edge type catalog: type_id → type name.
    edge_type_names: Vec<ArcStr>,

    /// Raw HNSW topology section bytes (if present in the file).
    hnsw_bytes: Option<*const [u8]>,

    /// Absolute file offset where the Strings section starts.
    /// Used by `NocacheStringReader` for uncached pread-based string fetches,
    /// bypassing the buffer cache on large-file workloads.
    strings_file_offset: u64,

    /// Epoch stored in the DbHeader.
    epoch: EpochId,
    /// Empty statistics (no cost estimation for mmap store yet).
    statistics: Arc<Statistics>,
}

// SAFETY: MmapStore is Send+Sync because:
// - The raw pointers point into the Mmap which is alive as long as MmapStore lives
// - All access is read-only (no mutation of pointed-to data)
// - The Mmap itself is Send+Sync
unsafe impl Send for MmapStore {}
unsafe impl Sync for MmapStore {}

/// Errors during MmapStore construction.
#[derive(Debug)]
pub enum MmapStoreError {
    /// TOC parsing failed.
    Toc(TocError),
    /// A required section is missing from the file.
    MissingSection(SectionType),
    /// A section could not be cast to the expected type.
    Cast(obrain_common::types::flat::CastError),
}

impl std::fmt::Display for MmapStoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MmapStoreError::Toc(e) => write!(f, "TOC error: {e}"),
            MmapStoreError::MissingSection(s) => write!(f, "missing section: {s:?}"),
            MmapStoreError::Cast(e) => write!(f, "cast error: {e}"),
        }
    }
}

impl std::error::Error for MmapStoreError {}

impl From<TocError> for MmapStoreError {
    fn from(e: TocError) -> Self {
        MmapStoreError::Toc(e)
    }
}

impl From<obrain_common::types::flat::CastError> for MmapStoreError {
    fn from(e: obrain_common::types::flat::CastError) -> Self {
        MmapStoreError::Cast(e)
    }
}

/// Layout of the file sections needed for mmap-free extraction.
/// Obtained from [`MmapStore::section_layout`].
#[derive(Debug, Clone)]
pub struct SectionLayout {
    pub strings_offset: u64,
    pub strings_len: u64,
    pub nodes_offset: u64,
    pub nodes_len: u64,
    pub props_offset: u64,
    pub props_len: u64,
    pub prop_values_offset: u64,
    pub prop_values_len: u64,
    pub node_count: usize,
    pub prop_count: usize,
}

impl MmapStore {
    /// Opens a v2 `.obrain` file as a read-only MmapStore.
    ///
    /// The `mmap` must be the entire file. The `toc_offset` is where the
    /// TOC page starts (typically 12 KiB).
    pub fn from_mmap(mmap: Mmap, toc_offset: usize, epoch: EpochId) -> Result<Self, MmapStoreError> {
        // SAFETY: We use raw pointers to reference slices inside the mmap.
        // The mmap is moved into the struct and kept alive for the lifetime
        // of MmapStore, so all pointers remain valid.
        let bytes: *const [u8] = &mmap[..] as *const [u8];
        let bytes_ref = unsafe { &*bytes };

        // Parse TOC
        let toc_end = (toc_offset + 4096).min(bytes_ref.len());
        let toc = FileToc::from_bytes(&bytes_ref[toc_offset..toc_end])?;

        // Helper: get section bytes or error
        let section_bytes = |st: SectionType| -> Result<&[u8], MmapStoreError> {
            let entry = toc.find(st).ok_or(MmapStoreError::MissingSection(st))?;
            let (start, end) = entry.byte_range();
            if end > bytes_ref.len() {
                return Err(MmapStoreError::MissingSection(st));
            }
            Ok(&bytes_ref[start..end])
        };

        // Parse required sections
        let strings_bytes = section_bytes(SectionType::Strings)?;
        // Capture the absolute file offset of the strings section (for uncached pread).
        let strings_file_offset = toc
            .find(SectionType::Strings)
            .map(|e| e.byte_range().0 as u64)
            .unwrap_or(0);
        let nodes_slice = cast_slice::<NodeSlot>(section_bytes(SectionType::Nodes)?)?;
        let edges_slice = cast_slice::<EdgeSlot>(section_bytes(SectionType::Edges)?)?;
        let props_slice = cast_slice::<PropEntry>(section_bytes(SectionType::Properties)?)?;
        let values_bytes = section_bytes(SectionType::PropertyValues)?;

        let fwd_offsets = cast_slice::<u64>(section_bytes(SectionType::CsrForwardOffsets)?)?;
        let fwd_targets = cast_slice::<u64>(section_bytes(SectionType::CsrForwardTargets)?)?;
        let fwd_edge_ids = cast_slice::<u64>(section_bytes(SectionType::CsrForwardEdgeIds)?)?;

        let bwd_offsets = cast_slice::<u64>(section_bytes(SectionType::CsrBackwardOffsets)?)?;
        let bwd_targets = cast_slice::<u64>(section_bytes(SectionType::CsrBackwardTargets)?)?;
        let bwd_edge_ids = cast_slice::<u64>(section_bytes(SectionType::CsrBackwardEdgeIds)?)?;

        // Parse label catalog — resolve to owned ArcStr
        let label_cat = cast_slice::<StringRef>(section_bytes(SectionType::LabelCatalog)?)?;
        let label_names: Vec<ArcStr> = label_cat
            .iter()
            .map(|sref| {
                let s = sref.resolve(strings_bytes).unwrap_or("");
                ArcStr::from(s)
            })
            .collect();

        // Parse edge type catalog
        let et_cat = cast_slice::<StringRef>(section_bytes(SectionType::EdgeTypeCatalog)?)?;
        let edge_type_names: Vec<ArcStr> = et_cat
            .iter()
            .map(|sref| {
                let s = sref.resolve(strings_bytes).unwrap_or("");
                ArcStr::from(s)
            })
            .collect();

        // Optional: HNSW topology section
        let hnsw_bytes = section_bytes(SectionType::HnswTopology)
            .ok()
            .filter(|b| !b.is_empty())
            .map(|b| b as *const [u8]);

        Ok(MmapStore {
            _mmap: mmap,
            strings: strings_bytes as *const [u8],
            nodes: nodes_slice as *const [NodeSlot],
            edges: edges_slice as *const [EdgeSlot],
            props: props_slice as *const [PropEntry],
            prop_values: values_bytes as *const [u8],
            csr_fwd_offsets: fwd_offsets as *const [u64],
            csr_fwd_targets: fwd_targets as *const [u64],
            csr_fwd_edge_ids: fwd_edge_ids as *const [u64],
            csr_bwd_offsets: bwd_offsets as *const [u64],
            csr_bwd_targets: bwd_targets as *const [u64],
            csr_bwd_edge_ids: bwd_edge_ids as *const [u64],
            label_names,
            edge_type_names,
            hnsw_bytes,
            strings_file_offset,
            epoch,
            statistics: Arc::new(Statistics::default()),
        })
    }

    /// Returns the absolute file offset where the Strings section starts.
    /// Callers pair this with `NocacheStringReader::open()` to do uncached
    /// pread-based fetches of long string values, bypassing the buffer cache.
    pub fn strings_file_offset(&self) -> u64 {
        self.strings_file_offset
    }

    /// Returns the length (in bytes) of the Strings section.
    pub fn strings_len(&self) -> usize {
        self.strings().len()
    }

    /// Returns absolute file offsets and lengths for the catalog sections
    /// needed by a fully-uncached extractor (no mmap).
    ///
    /// This lets callers build a pread-only extraction pipeline that never
    /// touches the mmap, keeping RSS bounded on very large files.
    pub fn section_layout(&self) -> SectionLayout {
        // Compute offsets by comparing each pointer to the mmap base.
        let base = self._mmap.as_ptr() as usize;
        let nodes_ptr = self.nodes().as_ptr() as *const u8 as usize;
        let props_ptr = self.props().as_ptr() as *const u8 as usize;
        let pv_ptr = self.prop_values().as_ptr() as usize;
        SectionLayout {
            strings_offset: self.strings_file_offset,
            strings_len: self.strings().len() as u64,
            nodes_offset: (nodes_ptr - base) as u64,
            nodes_len: (self.nodes().len() * std::mem::size_of::<NodeSlot>()) as u64,
            props_offset: (props_ptr - base) as u64,
            props_len: (self.props().len() * std::mem::size_of::<PropEntry>()) as u64,
            prop_values_offset: (pv_ptr - base) as u64,
            prop_values_len: self.prop_values().len() as u64,
            node_count: self.nodes().len(),
            prop_count: self.props().len(),
        }
    }

    /// Pre-resolve the `StringRef` for each key name in `keys` by scanning the
    /// property entries. Returns `None` for keys that don't appear in the DB.
    ///
    /// This is a one-time setup for bulk scans: once resolved, callers can
    /// compare `entry.key_ref == target_ref` directly (8-byte equality, no
    /// string-table lookup) which avoids faulting in scattered pages of the
    /// (possibly multi-GB) Strings section during the main extraction loop.
    ///
    /// Implementation: iterates the first N prop entries, collecting distinct
    /// `key_ref` values and resolving each only once. With ~100 unique keys
    /// in a typical DB this converges within a few thousand entries and
    /// touches only a handful of Strings pages.
    pub fn resolve_key_refs(&self, keys: &[&str]) -> Vec<Option<StringRef>> {
        let mut result: Vec<Option<StringRef>> = vec![None; keys.len()];
        let mut remaining = keys.len();
        if remaining == 0 {
            return result;
        }

        let props = self.props();
        let strings = self.strings();
        let mut seen: FxHashMap<(u32, u32), ()> = FxHashMap::default();
        // Scan up to 2M prop entries — plenty to cover all distinct keys in
        // any sane DB without faulting in too many Strings pages.
        let scan_n = props.len().min(2_000_000);
        for entry in &props[..scan_n] {
            let kr = entry.key_ref;
            let key_pair = (kr.offset, kr.len);
            if seen.insert(key_pair, ()).is_some() {
                continue; // already seen this key_ref
            }
            if let Some(k) = kr.resolve(strings) {
                for (i, &wanted) in keys.iter().enumerate() {
                    if result[i].is_none() && k == wanted {
                        result[i] = Some(kr);
                        remaining -= 1;
                        break;
                    }
                }
                if remaining == 0 {
                    break;
                }
            }
        }
        result
    }

    /// Returns the number of node slots (including potentially deleted ones).
    pub fn node_slot_count(&self) -> usize {
        self.nodes().len()
    }

    /// Returns the number of edge slots (including potentially deleted ones).
    pub fn edge_slot_count(&self) -> usize {
        self.edges().len()
    }

    // ── Materialization helpers (used by materialize_mmap_to_lpg) ──

    /// Iterates over all live nodes, yielding `(NodeId, labels, properties)`.
    ///
    /// Used to bulk-populate an LpgStore when opening a v2 file in read-write mode.
    pub fn iter_nodes(&self) -> impl Iterator<Item = (NodeId, smallvec::SmallVec<[ArcStr; 4]>, PropertyMap)> + '_ {
        self.nodes().iter().enumerate().filter_map(move |(i, slot)| {
            if slot.is_deleted() {
                return None;
            }
            let id = NodeId::new(i as u64);
            let mut labels = smallvec::SmallVec::new();
            for label_id in slot.label_ids() {
                if let Some(name) = self.label_names.get(label_id as usize) {
                    labels.push(name.clone());
                }
            }
            let properties = self.load_properties(slot.prop_offset, slot.prop_count);
            Some((id, labels, properties))
        })
    }

    /// Iterates over all live edges, yielding `(EdgeId, src, dst, edge_type, properties)`.
    pub fn iter_edges(&self) -> impl Iterator<Item = (EdgeId, NodeId, NodeId, ArcStr, PropertyMap)> + '_ {
        self.edges().iter().enumerate().filter_map(move |(i, slot)| {
            if slot.is_deleted() {
                return None;
            }
            let id = EdgeId::new(i as u64);
            let src = NodeId::new(slot.src);
            let dst = NodeId::new(slot.dst);
            let edge_type = slot
                .type_ref
                .resolve(self.strings())
                .map(ArcStr::from)
                .unwrap_or_default();
            let properties = self.load_properties(slot.prop_offset, slot.prop_count);
            Some((id, src, dst, edge_type, properties))
        })
    }

    /// Returns the epoch stored in the file header.
    pub fn epoch(&self) -> EpochId {
        self.epoch
    }

    /// Prepare the mmap for a sequential full scan.
    ///
    /// Sets `madvise(Sequential)` which enables kernel read-ahead
    /// (typically 2× the normal window). Does NOT use `WillNeed` to
    /// avoid forcing the entire file into the page cache at once
    /// (which can trigger OOM on large files).
    #[cfg(unix)]
    pub fn advise_sequential_scan(&self) {
        self._mmap
            .advise(memmap2::Advice::Sequential)
            .unwrap_or_else(|e| tracing::debug!("madvise(Sequential) failed: {e}"));
    }

    /// Prepare the mmap for random point lookups (disables read-ahead).
    #[cfg(unix)]
    pub fn advise_random_access(&self) {
        self._mmap
            .advise(memmap2::Advice::Random)
            .unwrap_or_else(|e| tracing::debug!("madvise(Random) failed: {e}"));
    }

    /// Tell the kernel we no longer need these pages — release them from the
    /// page cache back to the free list. Use this between super-chunks on
    /// large mmaps to prevent runaway resident-set growth.
    ///
    /// Safety: the mapping is read-only, so MADV_DONTNEED just causes the
    /// kernel to drop cached pages. Subsequent reads will page-fault them
    /// back from disk. This cannot cause UB for read-only shared mappings.
    #[cfg(unix)]
    pub fn advise_dontneed(&self) {
        let slice: &[u8] = &self._mmap;
        let addr = slice.as_ptr() as *mut libc::c_void;
        let len = slice.len();
        // Safety: read-only file-backed mapping; DONTNEED only evicts pages.
        let rc = unsafe { libc::madvise(addr, len, libc::MADV_DONTNEED) };
        if rc != 0 {
            let e = std::io::Error::last_os_error();
            tracing::debug!("madvise(DONTNEED) failed: {e}");
        }
    }

    /// Returns the label catalog (label_id → label name).
    pub fn label_catalog(&self) -> &[ArcStr] {
        &self.label_names
    }

    /// Returns the edge type catalog (type_id → type name).
    pub fn edge_type_catalog(&self) -> &[ArcStr] {
        &self.edge_type_names
    }

    /// Returns the raw HNSW topology section bytes, if present in the file.
    ///
    /// The bytes use the multi-index envelope format:
    /// `[count: u32] [key_len: u32, key_bytes, topo_len: u32, topo_bytes]*`
    #[must_use]
    pub fn hnsw_topology_bytes(&self) -> Option<&[u8]> {
        self.hnsw_bytes.map(|ptr| unsafe { &*ptr })
    }

    // ── Internal accessors (safe wrappers around raw pointers) ──

    #[inline]
    fn strings(&self) -> &[u8] {
        unsafe { &*self.strings }
    }

    #[inline]
    fn nodes(&self) -> &[NodeSlot] {
        unsafe { &*self.nodes }
    }

    #[inline]
    fn edges(&self) -> &[EdgeSlot] {
        unsafe { &*self.edges }
    }

    #[inline]
    fn props(&self) -> &[PropEntry] {
        unsafe { &*self.props }
    }

    #[inline]
    fn prop_values(&self) -> &[u8] {
        unsafe { &*self.prop_values }
    }

    #[inline]
    fn fwd_offsets(&self) -> &[u64] {
        unsafe { &*self.csr_fwd_offsets }
    }

    #[inline]
    fn fwd_targets(&self) -> &[u64] {
        unsafe { &*self.csr_fwd_targets }
    }

    #[inline]
    fn fwd_edge_ids(&self) -> &[u64] {
        unsafe { &*self.csr_fwd_edge_ids }
    }

    #[inline]
    fn bwd_offsets(&self) -> &[u64] {
        unsafe { &*self.csr_bwd_offsets }
    }

    #[inline]
    fn bwd_targets(&self) -> &[u64] {
        unsafe { &*self.csr_bwd_targets }
    }

    #[inline]
    fn bwd_edge_ids(&self) -> &[u64] {
        unsafe { &*self.csr_bwd_edge_ids }
    }

    // ── Property resolution ──

    /// Reads properties for a node/edge from the props section.
    fn load_properties(&self, prop_offset: u32, prop_count: u16) -> PropertyMap {
        let mut map = PropertyMap::with_capacity(prop_count as usize);
        let start = prop_offset as usize;
        let end = start + prop_count as usize;
        let props = self.props();

        if end > props.len() {
            return map;
        }

        for entry in &props[start..end] {
            let key_str = entry.key_ref.resolve(self.strings()).unwrap_or("");
            let key = PropertyKey::from(key_str);
            if let Some(value) = self.decode_value(entry) {
                map.insert(key, value);
            }
        }
        map
    }

    /// Decodes a single property value from the mmap.
    fn decode_value(&self, entry: &PropEntry) -> Option<Value> {
        let vt = ValueType::from_u8(entry.value_type)?;
        let start = entry.value_offset as usize;
        let end = start + entry.value_len as usize;
        let data = self.prop_values();

        if end > data.len() {
            return None;
        }
        let bytes = &data[start..end];

        Some(match vt {
            ValueType::Null => Value::Null,
            ValueType::Bool => Value::Bool(bytes.first().copied().unwrap_or(0) != 0),
            ValueType::Int64 => {
                let v = i64::from_le_bytes(bytes[..8].try_into().ok()?);
                Value::Int64(v)
            }
            ValueType::Float64 => {
                let v = f64::from_le_bytes(bytes[..8].try_into().ok()?);
                Value::Float64(v)
            }
            ValueType::String => {
                // Stored as StringRef (offset:u32 + len:u32)
                let off = u32::from_le_bytes(bytes[..4].try_into().ok()?);
                let len = u32::from_le_bytes(bytes[4..8].try_into().ok()?);
                let sref = StringRef::new(off, len);
                let s = sref.resolve(self.strings())?;
                Value::String(ArcStr::from(s))
            }
            ValueType::Bytes => Value::Bytes(Arc::from(bytes)),
            ValueType::Timestamp => {
                let micros = i64::from_le_bytes(bytes[..8].try_into().ok()?);
                Value::Timestamp(Timestamp::from_micros(micros))
            }
            ValueType::Date => {
                let days = i32::from_le_bytes(bytes[..4].try_into().ok()?);
                Value::Date(Date::from_days(days))
            }
            ValueType::Time => {
                let nanos = u64::from_le_bytes(bytes[..8].try_into().ok()?);
                let offset_secs = i32::from_le_bytes(bytes[8..12].try_into().ok()?);
                let t = Time::from_nanos(nanos)?;
                Value::Time(if offset_secs != 0 {
                    t.with_offset(offset_secs)
                } else {
                    t
                })
            }
            ValueType::Duration => {
                let months = i64::from_le_bytes(bytes[..8].try_into().ok()?);
                let days = i64::from_le_bytes(bytes[8..16].try_into().ok()?);
                let nanos = i64::from_le_bytes(bytes[16..24].try_into().ok()?);
                Value::Duration(Duration::new(months, days, nanos))
            }
            ValueType::ZonedDatetime => {
                let micros = i64::from_le_bytes(bytes[..8].try_into().ok()?);
                let offset_secs = i32::from_le_bytes(bytes[8..12].try_into().ok()?);
                let ts = Timestamp::from_micros(micros);
                Value::ZonedDatetime(ZonedDatetime::from_timestamp_offset(ts, offset_secs))
            }
            ValueType::Vector => {
                let floats: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                Value::Vector(Arc::from(floats))
            }
            // List, Map, Path: complex recursive types — skip for now
            ValueType::List | ValueType::Map | ValueType::Path => Value::Null,
        })
    }

    /// Build a Node object from a NodeSlot.
    fn build_node(&self, slot: &NodeSlot) -> Node {
        let id = NodeId::new(slot.id);
        let mut labels = smallvec::SmallVec::new();
        for label_id in slot.label_ids() {
            if let Some(name) = self.label_names.get(label_id as usize) {
                labels.push(name.clone());
            }
        }
        let properties = self.load_properties(slot.prop_offset, slot.prop_count);
        Node {
            id,
            labels,
            properties,
        }
    }

    /// Build an Edge object from an EdgeSlot.
    fn build_edge(&self, slot: &EdgeSlot) -> Edge {
        let id = EdgeId::new(slot.id);
        let src = NodeId::new(slot.src);
        let dst = NodeId::new(slot.dst);
        let edge_type = slot
            .type_ref
            .resolve(self.strings())
            .map(ArcStr::from)
            .unwrap_or_default();
        let properties = self.load_properties(slot.prop_offset, slot.prop_count);
        Edge {
            id,
            src,
            dst,
            edge_type,
            properties,
        }
    }

    /// Returns the CSR range [start, end) for a given node index.
    fn csr_range(offsets: &[u64], node_idx: usize) -> Option<(usize, usize)> {
        if node_idx + 1 >= offsets.len() {
            return None;
        }
        let start = offsets[node_idx] as usize;
        let end = offsets[node_idx + 1] as usize;
        Some((start, end))
    }

    /// Check if a node has a specific property WITHOUT decoding the value.
    /// Much faster than `get_node_property` when you only need existence check.
    pub fn node_has_property(&self, id: NodeId, key: &str) -> bool {
        let idx = id.0 as usize;
        let slot = match self.nodes().get(idx) {
            Some(s) if !s.is_deleted() => s,
            _ => return false,
        };

        let start = slot.prop_offset as usize;
        let end = start + slot.prop_count as usize;
        let props = self.props();
        if end > props.len() {
            return false;
        }

        for entry in &props[start..end] {
            if let Some(k) = entry.key_ref.resolve(self.strings()) {
                if k == key {
                    return true;
                }
            }
        }
        false
    }

    /// Extract and concatenate the String-valued text properties of a node
    /// using an uncached file reader (F_NOCACHE) for the actual string bytes.
    ///
    /// This is critical for enrichment workloads on very large databases: the
    /// strings section can be tens of GB and would otherwise saturate the
    /// buffer cache, triggering OOM/jetsam under pressure. By reading the
    /// bulk of the text via direct I/O we keep the resident-set bounded.
    ///
    /// The mmap is still used for the (small) prop catalog and key-string
    /// resolution — those pages are tiny and OK to cache.
    ///
    /// Returns `None` if the node has no text. Order of concatenation follows
    /// the order of `keys`; found values are joined with a single space.
    pub fn get_node_text_nocache(
        &self,
        id: NodeId,
        keys: &[&str],
        reader: &NocacheStringReader,
    ) -> Option<String> {
        let idx = id.0 as usize;
        let slot = match self.nodes().get(idx) {
            Some(s) if !s.is_deleted() => s,
            _ => return None,
        };

        let start = slot.prop_offset as usize;
        let end = start + slot.prop_count as usize;
        let props = self.props();
        if end > props.len() {
            return None;
        }

        // Collect (key_index, offset, len) for matching String-type props.
        // We defer I/O so all pread calls happen in one tight loop.
        let prop_values = self.prop_values();
        let mut hits: Vec<(usize, u32, u32)> = Vec::with_capacity(keys.len());
        let mut found_mask: u32 = 0;

        for entry in &props[start..end] {
            if found_mask.count_ones() as usize == keys.len() { break; }
            // Only interested in Strings — cheap typecheck before resolving key.
            if ValueType::from_u8(entry.value_type) != Some(ValueType::String) {
                continue;
            }
            let k = match entry.key_ref.resolve(self.strings()) {
                Some(k) => k,
                None => continue,
            };
            for (i, &wanted) in keys.iter().enumerate() {
                let bit = 1u32 << i;
                if (found_mask & bit) == 0 && k == wanted {
                    // Parse the StringRef (offset:u32 + len:u32) from prop_values.
                    let vstart = entry.value_offset as usize;
                    let vend = vstart + entry.value_len as usize;
                    if vend <= prop_values.len() && entry.value_len as usize >= 8 {
                        let bytes = &prop_values[vstart..vstart + 8];
                        let off = u32::from_le_bytes(bytes[..4].try_into().unwrap());
                        let len = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
                        hits.push((i, off, len));
                        found_mask |= bit;
                    }
                    break;
                }
            }
        }

        if hits.is_empty() {
            return None;
        }

        // Sort by key-order to preserve deterministic output.
        hits.sort_by_key(|(i, _, _)| *i);

        // Uncached reads — kernel bypasses the buffer cache (macOS F_NOCACHE)
        // or posix_fadvise(DONTNEED) on Linux.
        let mut text = String::new();
        for (_, off, len) in hits {
            if len == 0 { continue; }
            if let Some(s) = reader.read_string(off, len) {
                if !s.is_empty() {
                    if !text.is_empty() { text.push(' '); }
                    text.push_str(&s);
                }
            }
        }

        if text.is_empty() { None } else { Some(text) }
    }

    /// Same as [`get_node_text_nocache`] but takes pre-resolved `StringRef`s
    /// for each key, avoiding any Strings-section access in the hot loop.
    ///
    /// Use [`resolve_key_refs`] once at startup to get the refs, then call
    /// this per node. Keys with `None` refs are silently ignored.
    ///
    /// On very large DBs (20+ GB Strings section), this is the difference
    /// between a bounded ~100 MB RSS and an unbounded multi-GB RSS balloon:
    /// the `key_ref.resolve()` call in the string-keyed version reads pages
    /// scattered across the entire Strings section, which the kernel faults
    /// in eagerly.
    pub fn get_node_text_nocache_by_ref(
        &self,
        id: NodeId,
        key_refs: &[Option<StringRef>],
        reader: &NocacheStringReader,
    ) -> Option<String> {
        let idx = id.0 as usize;
        let slot = match self.nodes().get(idx) {
            Some(s) if !s.is_deleted() => s,
            _ => return None,
        };

        let start = slot.prop_offset as usize;
        let end = start + slot.prop_count as usize;
        let props = self.props();
        if end > props.len() {
            return None;
        }

        let prop_values = self.prop_values();
        let mut hits: Vec<(usize, u32, u32)> = Vec::with_capacity(key_refs.len());
        let mut found_mask: u32 = 0;

        for entry in &props[start..end] {
            if found_mask.count_ones() as usize == key_refs.len() { break; }
            if ValueType::from_u8(entry.value_type) != Some(ValueType::String) {
                continue;
            }
            let kr = entry.key_ref;
            for (i, wanted) in key_refs.iter().enumerate() {
                let bit = 1u32 << i;
                if (found_mask & bit) != 0 { continue; }
                let Some(target) = wanted else { continue };
                // Direct 8-byte equality — NO strings-section access.
                if kr.offset == target.offset && kr.len == target.len {
                    let vstart = entry.value_offset as usize;
                    let vend = vstart + entry.value_len as usize;
                    if vend <= prop_values.len() && entry.value_len as usize >= 8 {
                        let bytes = &prop_values[vstart..vstart + 8];
                        let off = u32::from_le_bytes(bytes[..4].try_into().unwrap());
                        let len = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
                        hits.push((i, off, len));
                        found_mask |= bit;
                    }
                    break;
                }
            }
        }

        if hits.is_empty() {
            return None;
        }

        hits.sort_by_key(|(i, _, _)| *i);

        let mut text = String::new();
        for (_, off, len) in hits {
            if len == 0 { continue; }
            if let Some(s) = reader.read_string(off, len) {
                if !s.is_empty() {
                    if !text.is_empty() { text.push(' '); }
                    text.push_str(&s);
                }
            }
        }

        if text.is_empty() { None } else { Some(text) }
    }

    /// Pre-resolved variant of `node_has_property` — compares `key_ref` by
    /// 8-byte equality to avoid Strings-section reads in the hot loop.
    pub fn node_has_property_by_ref(&self, id: NodeId, key_ref: StringRef) -> bool {
        let idx = id.0 as usize;
        let slot = match self.nodes().get(idx) {
            Some(s) if !s.is_deleted() => s,
            _ => return false,
        };

        let start = slot.prop_offset as usize;
        let end = start + slot.prop_count as usize;
        let props = self.props();
        if end > props.len() {
            return false;
        }

        for entry in &props[start..end] {
            if entry.key_ref.offset == key_ref.offset
                && entry.key_ref.len == key_ref.len
            {
                return true;
            }
        }
        false
    }

    /// Get multiple properties in a single pass over the node's prop entries.
    /// Much faster than calling `get_node_property` N times (1 pass vs N passes,
    /// 1 set of page faults vs N sets).
    pub fn get_node_properties_multi(&self, id: NodeId, keys: &[&str]) -> Vec<Option<Value>> {
        let mut result = vec![None; keys.len()];
        let idx = id.0 as usize;
        let slot = match self.nodes().get(idx) {
            Some(s) if !s.is_deleted() => s,
            _ => return result,
        };

        let start = slot.prop_offset as usize;
        let end = start + slot.prop_count as usize;
        let props = self.props();
        if end > props.len() {
            return result;
        }

        let mut found = 0usize;
        for entry in &props[start..end] {
            if found == keys.len() { break; }
            if let Some(k) = entry.key_ref.resolve(self.strings()) {
                for (i, &wanted) in keys.iter().enumerate() {
                    if result[i].is_none() && k == wanted {
                        result[i] = self.decode_value(entry);
                        found += 1;
                        break;
                    }
                }
            }
        }
        result
    }
}

// ─────────────────────────────────────────────────────────────────────
// GraphStore implementation
// ─────────────────────────────────────────────────────────────────────

impl GraphStore for MmapStore {
    fn get_node(&self, id: NodeId) -> Option<Node> {
        let idx = id.0 as usize;
        let slot = self.nodes().get(idx)?;
        if slot.is_deleted() || slot.id != id.0 {
            return None;
        }
        Some(self.build_node(slot))
    }

    fn get_edge(&self, id: EdgeId) -> Option<Edge> {
        let idx = id.0 as usize;
        let slot = self.edges().get(idx)?;
        if slot.is_deleted() || slot.id != id.0 {
            return None;
        }
        Some(self.build_edge(slot))
    }

    fn get_node_versioned(
        &self,
        id: NodeId,
        _epoch: EpochId,
        _transaction_id: TransactionId,
    ) -> Option<Node> {
        self.get_node(id)
    }

    fn get_edge_versioned(
        &self,
        id: EdgeId,
        _epoch: EpochId,
        _transaction_id: TransactionId,
    ) -> Option<Edge> {
        self.get_edge(id)
    }

    fn get_node_at_epoch(&self, id: NodeId, _epoch: EpochId) -> Option<Node> {
        self.get_node(id)
    }

    fn get_edge_at_epoch(&self, id: EdgeId, _epoch: EpochId) -> Option<Edge> {
        self.get_edge(id)
    }

    fn get_node_property(&self, id: NodeId, key: &PropertyKey) -> Option<Value> {
        let idx = id.0 as usize;
        let slot = self.nodes().get(idx)?;
        if slot.is_deleted() {
            return None;
        }

        let start = slot.prop_offset as usize;
        let end = start + slot.prop_count as usize;
        let props = self.props();
        if end > props.len() {
            return None;
        }

        let key_str = key.as_str();
        for entry in &props[start..end] {
            if let Some(k) = entry.key_ref.resolve(self.strings()) {
                if k == key_str {
                    return self.decode_value(entry);
                }
            }
        }
        None
    }

    fn get_edge_property(&self, id: EdgeId, key: &PropertyKey) -> Option<Value> {
        let idx = id.0 as usize;
        let slot = self.edges().get(idx)?;
        if slot.is_deleted() {
            return None;
        }

        let start = slot.prop_offset as usize;
        let end = start + slot.prop_count as usize;
        let props = self.props();
        if end > props.len() {
            return None;
        }

        let key_str = key.as_str();
        for entry in &props[start..end] {
            if let Some(k) = entry.key_ref.resolve(self.strings()) {
                if k == key_str {
                    return self.decode_value(entry);
                }
            }
        }
        None
    }

    fn get_node_property_batch(&self, ids: &[NodeId], key: &PropertyKey) -> Vec<Option<Value>> {
        ids.iter().map(|id| self.get_node_property(*id, key)).collect()
    }

    fn get_nodes_properties_batch(&self, ids: &[NodeId]) -> Vec<FxHashMap<PropertyKey, Value>> {
        ids.iter()
            .map(|id| {
                let idx = id.0 as usize;
                if let Some(slot) = self.nodes().get(idx) {
                    if !slot.is_deleted() {
                        let pmap = self.load_properties(slot.prop_offset, slot.prop_count);
                        return pmap.into_iter().collect();
                    }
                }
                FxHashMap::default()
            })
            .collect()
    }

    fn get_nodes_properties_selective_batch(
        &self,
        ids: &[NodeId],
        keys: &[PropertyKey],
    ) -> Vec<FxHashMap<PropertyKey, Value>> {
        ids.iter()
            .map(|id| {
                let mut map = FxHashMap::default();
                for key in keys {
                    if let Some(val) = self.get_node_property(*id, key) {
                        map.insert(key.clone(), val);
                    }
                }
                map
            })
            .collect()
    }

    fn get_edges_properties_selective_batch(
        &self,
        ids: &[EdgeId],
        keys: &[PropertyKey],
    ) -> Vec<FxHashMap<PropertyKey, Value>> {
        ids.iter()
            .map(|id| {
                let mut map = FxHashMap::default();
                for key in keys {
                    if let Some(val) = self.get_edge_property(*id, key) {
                        map.insert(key.clone(), val);
                    }
                }
                map
            })
            .collect()
    }

    // ── Traversal (CSR) ──

    fn neighbors(&self, node: NodeId, direction: Direction) -> Vec<NodeId> {
        let idx = node.0 as usize;
        let mut result = Vec::new();

        if matches!(direction, Direction::Outgoing | Direction::Both) {
            if let Some((start, end)) = Self::csr_range(self.fwd_offsets(), idx) {
                let targets = self.fwd_targets();
                if end <= targets.len() {
                    result.extend(targets[start..end].iter().map(|&t| NodeId::new(t)));
                }
            }
        }

        if matches!(direction, Direction::Incoming | Direction::Both) {
            if let Some((start, end)) = Self::csr_range(self.bwd_offsets(), idx) {
                let targets = self.bwd_targets();
                if end <= targets.len() {
                    result.extend(targets[start..end].iter().map(|&t| NodeId::new(t)));
                }
            }
        }

        result
    }

    fn edges_from(&self, node: NodeId, direction: Direction) -> Vec<(NodeId, EdgeId)> {
        let idx = node.0 as usize;
        let mut result = Vec::new();

        if matches!(direction, Direction::Outgoing | Direction::Both) {
            if let Some((start, end)) = Self::csr_range(self.fwd_offsets(), idx) {
                let targets = self.fwd_targets();
                let eids = self.fwd_edge_ids();
                if end <= targets.len() && end <= eids.len() {
                    for i in start..end {
                        result.push((NodeId::new(targets[i]), EdgeId::new(eids[i])));
                    }
                }
            }
        }

        if matches!(direction, Direction::Incoming | Direction::Both) {
            if let Some((start, end)) = Self::csr_range(self.bwd_offsets(), idx) {
                let targets = self.bwd_targets();
                let eids = self.bwd_edge_ids();
                if end <= targets.len() && end <= eids.len() {
                    for i in start..end {
                        result.push((NodeId::new(targets[i]), EdgeId::new(eids[i])));
                    }
                }
            }
        }

        result
    }

    fn out_degree(&self, node: NodeId) -> usize {
        Self::csr_range(self.fwd_offsets(), node.0 as usize)
            .map(|(s, e)| e - s)
            .unwrap_or(0)
    }

    fn in_degree(&self, node: NodeId) -> usize {
        Self::csr_range(self.bwd_offsets(), node.0 as usize)
            .map(|(s, e)| e - s)
            .unwrap_or(0)
    }

    fn has_backward_adjacency(&self) -> bool {
        !self.bwd_offsets().is_empty()
    }

    // ── Scans ──

    fn node_ids(&self) -> Vec<NodeId> {
        self.nodes()
            .iter()
            .enumerate()
            .filter(|(_, slot)| !slot.is_deleted())
            .map(|(i, _)| NodeId::new(i as u64))
            .collect()
    }

    fn nodes_by_label(&self, label: &str) -> Vec<NodeId> {
        let label_id = self.label_names.iter().position(|l| l.as_str() == label);
        let Some(lid) = label_id else {
            return Vec::new();
        };
        let lid = lid as u32;

        self.nodes()
            .iter()
            .enumerate()
            .filter(|(_, slot)| !slot.is_deleted() && slot.has_label(lid))
            .map(|(i, _)| NodeId::new(i as u64))
            .collect()
    }

    fn node_count(&self) -> usize {
        self.nodes().iter().filter(|s| !s.is_deleted()).count()
    }

    fn edge_count(&self) -> usize {
        self.edges().iter().filter(|s| !s.is_deleted()).count()
    }

    fn edge_type(&self, id: EdgeId) -> Option<ArcStr> {
        let slot = self.edges().get(id.0 as usize)?;
        if slot.is_deleted() {
            return None;
        }
        slot.type_ref.resolve(self.strings()).map(ArcStr::from)
    }

    fn find_nodes_by_property(&self, property: &str, value: &Value) -> Vec<NodeId> {
        let key = PropertyKey::from(property);
        self.nodes()
            .iter()
            .enumerate()
            .filter(|(_, slot)| !slot.is_deleted())
            .filter_map(|(i, _)| {
                let id = NodeId::new(i as u64);
                let v = self.get_node_property(id, &key)?;
                if &v == value { Some(id) } else { None }
            })
            .collect()
    }

    fn find_nodes_by_properties(&self, conditions: &[(&str, Value)]) -> Vec<NodeId> {
        let keys: Vec<PropertyKey> = conditions.iter().map(|(k, _)| PropertyKey::from(*k)).collect();
        self.nodes()
            .iter()
            .enumerate()
            .filter(|(_, slot)| !slot.is_deleted())
            .filter_map(|(i, _)| {
                let id = NodeId::new(i as u64);
                for (idx, (_, expected)) in conditions.iter().enumerate() {
                    let actual = self.get_node_property(id, &keys[idx])?;
                    if &actual != expected {
                        return None;
                    }
                }
                Some(id)
            })
            .collect()
    }

    fn find_nodes_in_range(
        &self,
        property: &str,
        _min: Option<&Value>,
        _max: Option<&Value>,
        _min_inclusive: bool,
        _max_inclusive: bool,
    ) -> Vec<NodeId> {
        // Mmap store doesn't have property indexes — return all nodes with
        // the property and let the query engine filter. This is a fallback;
        // for production use, convert to LpgStore first.
        let key = PropertyKey::from(property);
        self.nodes()
            .iter()
            .enumerate()
            .filter(|(_, slot)| !slot.is_deleted())
            .filter_map(|(i, _)| {
                let id = NodeId::new(i as u64);
                self.get_node_property(id, &key).map(|_| id)
            })
            .collect()
    }

    fn node_property_might_match(
        &self,
        _property: &PropertyKey,
        _op: CompareOp,
        _value: &Value,
    ) -> bool {
        true // No zone maps — conservatively return true
    }

    fn edge_property_might_match(
        &self,
        _property: &PropertyKey,
        _op: CompareOp,
        _value: &Value,
    ) -> bool {
        true
    }

    fn statistics(&self) -> Arc<Statistics> {
        self.statistics.clone()
    }

    fn estimate_label_cardinality(&self, label: &str) -> f64 {
        self.nodes_by_label(label).len() as f64
    }

    fn estimate_avg_degree(&self, _edge_type: &str, _outgoing: bool) -> f64 {
        let nc = self.node_count();
        if nc == 0 { 0.0 } else { self.edge_count() as f64 / nc as f64 }
    }

    fn current_epoch(&self) -> EpochId {
        self.epoch
    }

    fn all_labels(&self) -> Vec<String> {
        self.label_names.iter().map(|s| s.to_string()).collect()
    }

    fn all_edge_types(&self) -> Vec<String> {
        self.edge_type_names.iter().map(|s| s.to_string()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use obrain_adapters::storage::file::format::DATA_OFFSET;
    use obrain_core::graph::lpg::LpgStore;
    use obrain_core::graph::traits::GraphStore;

    /// Creates a populated LpgStore, writes it to v2 native format,
    /// opens it as MmapStore, and verifies all queries return identical results.
    #[test]
    fn test_mmap_store_equivalence() {
        let store = LpgStore::new().unwrap();

        // ── Populate store with varied data ──

        let labels_person = vec!["Person"];
        let labels_city = vec!["City", "Place"];
        let labels_company = vec!["Company", "Organization"];

        let mut node_ids = Vec::new();
        let names = [
            "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Hank",
        ];
        for (i, name) in names.iter().enumerate() {
            let nid = NodeId::new(i as u64);
            store.create_node_with_id(nid, &labels_person).unwrap();
            store.set_node_property_bulk(nid, "name", Value::String(ArcStr::from(*name)));
            store.set_node_property_bulk(nid, "age", Value::Int64(20 + i as i64));
            store.set_node_property_bulk(nid, "score", Value::Float64(0.5 + i as f64 * 0.1));
            store.set_node_property_bulk(nid, "active", Value::Bool(i % 2 == 0));
            node_ids.push(nid);
        }

        // City nodes
        let cities = ["Paris", "London", "Berlin"];
        for (i, city) in cities.iter().enumerate() {
            let nid = NodeId::new((names.len() + i) as u64);
            store.create_node_with_id(nid, &labels_city).unwrap();
            store.set_node_property_bulk(nid, "name", Value::String(ArcStr::from(*city)));
            store.set_node_property_bulk(nid, "population", Value::Int64((1_000_000 * (i + 1)) as i64));
            node_ids.push(nid);
        }

        // Company node
        let company_id = NodeId::new((names.len() + cities.len()) as u64);
        store.create_node_with_id(company_id, &labels_company).unwrap();
        store.set_node_property_bulk(company_id, "name", Value::String(ArcStr::from("Acme Corp")));
        node_ids.push(company_id);

        // ── Edges ──
        let mut edge_ids = Vec::new();

        // KNOWS edges (Person → Person)
        let knows_pairs = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7)];
        for (i, (src, dst)) in knows_pairs.iter().enumerate() {
            let eid = EdgeId::new(i as u64);
            store.create_edge_with_id(eid, NodeId::new(*src as u64), NodeId::new(*dst as u64), "KNOWS").unwrap();
            store.set_edge_property_bulk(eid, "since", Value::Int64(2015 + i as i64));
            edge_ids.push(eid);
        }

        // LIVES_IN edges (Person → City)
        for (i, city_offset) in [0usize, 1, 2, 0, 1, 2, 0, 1].iter().enumerate() {
            let eid = EdgeId::new((knows_pairs.len() + i) as u64);
            store.create_edge_with_id(
                eid,
                NodeId::new(i as u64),
                NodeId::new((names.len() + city_offset) as u64),
                "LIVES_IN",
            ).unwrap();
            edge_ids.push(eid);
        }

        // WORKS_AT edge
        let works_eid = EdgeId::new((knows_pairs.len() + names.len()) as u64);
        store.create_edge_with_id(works_eid, NodeId::new(0), company_id, "WORKS_AT").unwrap();
        store.set_edge_property_bulk(works_eid, "role", Value::String(ArcStr::from("Engineer")));
        edge_ids.push(works_eid);

        let total_nodes = store.node_count();
        let total_edges = store.edge_count();

        // ── Write to v2 native format ──
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("equivalence.obrain");

        super::super::native_writer::write_native_v2(&store, &path, 42, 100).unwrap();

        // ── Open as MmapStore ──
        let file = std::fs::File::open(&path).unwrap();
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file).unwrap() };
        let mmap_store = MmapStore::from_mmap(mmap, DATA_OFFSET as usize, EpochId(42)).unwrap();

        // ── Verify counts ──
        assert_eq!(mmap_store.node_count(), total_nodes, "node_count mismatch");
        assert_eq!(mmap_store.edge_count(), total_edges, "edge_count mismatch");

        // ── Verify all nodes ──
        for &nid in &node_ids {
            let lpg_node = store.get_node(nid).expect("LpgStore node missing");
            let mmap_node = mmap_store.get_node(nid).expect(&format!("MmapStore node {nid:?} missing"));

            assert_eq!(lpg_node.id, mmap_node.id, "node id mismatch");

            // Labels (order may differ — compare as sets)
            let mut lpg_labels: Vec<String> = lpg_node.labels.iter().map(|l| l.to_string()).collect();
            let mut mmap_labels: Vec<String> = mmap_node.labels.iter().map(|l| l.to_string()).collect();
            lpg_labels.sort();
            mmap_labels.sort();
            assert_eq!(lpg_labels, mmap_labels, "labels mismatch for node {nid:?}");

            // Properties
            for (key, lpg_val) in &lpg_node.properties {
                let mmap_val = mmap_store.get_node_property(nid, key)
                    .expect(&format!("MmapStore missing property {key} on node {nid:?}"));
                assert_eq!(*lpg_val, mmap_val, "property {key} mismatch on node {nid:?}");
            }
        }

        // ── Verify all edges ──
        for &eid in &edge_ids {
            let lpg_edge = store.get_edge(eid).expect("LpgStore edge missing");
            let mmap_edge = mmap_store.get_edge(eid).expect(&format!("MmapStore edge {eid:?} missing"));

            assert_eq!(lpg_edge.id, mmap_edge.id, "edge id mismatch");
            assert_eq!(lpg_edge.src, mmap_edge.src, "edge src mismatch for {eid:?}");
            assert_eq!(lpg_edge.dst, mmap_edge.dst, "edge dst mismatch for {eid:?}");
            assert_eq!(lpg_edge.edge_type, mmap_edge.edge_type, "edge type mismatch for {eid:?}");

            // Edge properties
            for (key, lpg_val) in &lpg_edge.properties {
                let mmap_val = mmap_store.get_edge_property(eid, key)
                    .expect(&format!("MmapStore missing property {key} on edge {eid:?}"));
                assert_eq!(*lpg_val, mmap_val, "edge property {key} mismatch on {eid:?}");
            }
        }

        // ── Verify CSR traversal ──
        // Check outgoing neighbors
        for i in 0..names.len() {
            let nid = NodeId::new(i as u64);
            let mut lpg_out: Vec<NodeId> = store.neighbors(nid, Direction::Outgoing).collect();
            let mut mmap_out = mmap_store.neighbors(nid, Direction::Outgoing);
            lpg_out.sort();
            mmap_out.sort();
            assert_eq!(lpg_out, mmap_out, "outgoing neighbors mismatch for node {i}");

            let mut lpg_in: Vec<NodeId> = store.neighbors(nid, Direction::Incoming).collect();
            let mut mmap_in = mmap_store.neighbors(nid, Direction::Incoming);
            lpg_in.sort();
            mmap_in.sort();
            assert_eq!(lpg_in, mmap_in, "incoming neighbors mismatch for node {i}");
        }

        // ── Verify edges_from ──
        let nid0 = NodeId::new(0);
        let mut lpg_edges_out: Vec<(NodeId, EdgeId)> = store.edges_from(nid0, Direction::Outgoing).collect();
        let mut mmap_edges_out: Vec<(NodeId, EdgeId)> = mmap_store.edges_from(nid0, Direction::Outgoing);
        lpg_edges_out.sort_by_key(|(n, _)| *n);
        mmap_edges_out.sort_by_key(|(n, _)| *n);
        assert_eq!(lpg_edges_out.len(), mmap_edges_out.len(), "edges_from count mismatch");

        // ── Verify label scan ──
        let mut lpg_persons = store.nodes_by_label("Person");
        let mut mmap_persons = mmap_store.nodes_by_label("Person");
        lpg_persons.sort();
        mmap_persons.sort();
        assert_eq!(lpg_persons, mmap_persons, "Person label scan mismatch");

        let mut lpg_cities = store.nodes_by_label("City");
        let mut mmap_cities = mmap_store.nodes_by_label("City");
        lpg_cities.sort();
        mmap_cities.sort();
        assert_eq!(lpg_cities, mmap_cities, "City label scan mismatch");

        // ── Verify degree ──
        assert_eq!(
            store.out_degree(nid0),
            mmap_store.out_degree(nid0),
            "out_degree mismatch for node 0"
        );
        assert_eq!(
            store.in_degree(NodeId::new(1)),
            mmap_store.in_degree(NodeId::new(1)),
            "in_degree mismatch for node 1"
        );

        // ── Verify catalogs ──
        let mut lpg_labels = store.all_labels();
        let mut mmap_labels = mmap_store.all_labels();
        lpg_labels.sort();
        mmap_labels.sort();
        assert_eq!(lpg_labels, mmap_labels, "all_labels mismatch");

        let mut lpg_etypes = store.all_edge_types();
        let mut mmap_etypes = mmap_store.all_edge_types();
        lpg_etypes.sort();
        mmap_etypes.sort();
        assert_eq!(lpg_etypes, mmap_etypes, "all_edge_types mismatch");

        // ── Verify epoch ──
        assert_eq!(mmap_store.current_epoch(), EpochId(42), "epoch mismatch");

        eprintln!(
            "Equivalence verified: {} nodes, {} edges, all properties, CSR traversal, label scans ✓",
            total_nodes, total_edges
        );
    }
}
