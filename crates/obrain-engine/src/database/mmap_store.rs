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
            epoch,
            statistics: Arc::new(Statistics::default()),
        })
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

    /// Returns the label catalog (label_id → label name).
    pub fn label_catalog(&self) -> &[ArcStr] {
        &self.label_names
    }

    /// Returns the edge type catalog (type_id → type name).
    pub fn edge_type_catalog(&self) -> &[ArcStr] {
        &self.edge_type_names
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
