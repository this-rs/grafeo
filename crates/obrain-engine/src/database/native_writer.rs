//! Native v2 writer: converts an in-memory LpgStore into the mmap-friendly
//! `.obrain` v2 file format.
//!
//! The writer produces a single file with this layout:
//!
//! | Offset   | Size     | Contents |
//! |----------|----------|----------|
//! | 0        | 4 KiB    | FileHeader (magic=GRAF, version=2) |
//! | 4 KiB    | 4 KiB    | DbHeader slot 0 |
//! | 8 KiB    | 4 KiB    | DbHeader slot 1 |
//! | 12 KiB   | 4 KiB    | TOC (section table) |
//! | 16 KiB+  | variable | Data sections (strings, nodes, edges, CSR, props...) |

use std::collections::{HashMap, HashSet};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use obrain_adapters::storage::file::format::{
    DbHeader, FileHeader, DATA_OFFSET, DB_HEADER_SIZE, FILE_HEADER_SIZE,
};
use obrain_adapters::storage::file::toc::{SectionType, TocBuilder, TOC_PAGE_SIZE};
use obrain_common::types::flat::{
    EdgeSlot, NodeSlot, PropEntry, StringRef, ValueType,
};
use obrain_common::types::Value;
use obrain_common::utils::error::{Error, Result};
use obrain_core::graph::lpg::LpgStore;

/// Alignment for data sections (8 bytes for u64 casting).
const SECTION_ALIGN: u64 = 8;

/// Byte offset where data sections begin (after FileHeader + 2 DbHeaders + TOC).
const SECTIONS_START: u64 = DATA_OFFSET + TOC_PAGE_SIZE as u64;

// ─────────────────────────────────────────────────────────────────────
// StringTable builder
// ───────────────────────────────────────────────────��─────────────────

/// Collects unique strings and produces a compact StringTable.
struct StringTableBuilder {
    /// Buffer holding concatenated UTF-8 strings.
    buffer: Vec<u8>,
    /// Map from string content to its StringRef.
    lookup: HashMap<String, StringRef>,
}

impl StringTableBuilder {
    fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(1024 * 1024), // 1 MB initial
            lookup: HashMap::with_capacity(4096),
        }
    }

    /// Interns a string, returning its StringRef. Deduplicates.
    fn intern(&mut self, s: &str) -> StringRef {
        if let Some(&sref) = self.lookup.get(s) {
            return sref;
        }
        let offset = self.buffer.len() as u32;
        let len = s.len() as u32;
        self.buffer.extend_from_slice(s.as_bytes());
        let sref = StringRef::new(offset, len);
        self.lookup.insert(s.to_string(), sref);
        sref
    }

    /// Returns the raw bytes of the string table.
    fn into_bytes(self) -> Vec<u8> {
        self.buffer
    }
}

// ─────────────────────────────────────────────────────────────────────
// Values writer — spills to temp file instead of buffering in RAM
// ─────────────────────────────────────────────────────────────────────

/// Writes serialized property values to a temporary file on disk.
///
/// This replaces the in-memory `Vec<u8>` values buffer which caused OOM
/// for large databases (8M+ nodes with 384d+ embedding vectors = 20+ GB
/// of property values that don't fit in RAM alongside the store itself).
struct ValuesWriter {
    writer: BufWriter<std::fs::File>,
    offset: u32,
    path: std::path::PathBuf,
}

impl ValuesWriter {
    fn new(parent_dir: &Path) -> std::io::Result<Self> {
        let path = parent_dir.join(format!(
            ".obrain-values-{}.tmp",
            std::process::id()
        ));
        let file = std::fs::File::create(&path)?;
        Ok(Self {
            writer: BufWriter::with_capacity(8 * 1024 * 1024, file),
            offset: 0,
            path,
        })
    }

    #[inline]
    fn offset(&self) -> u32 {
        self.offset
    }

    #[inline]
    fn write(&mut self, data: &[u8]) -> std::io::Result<()> {
        self.writer.write_all(data)?;
        self.offset += data.len() as u32;
        Ok(())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.writer.flush()
    }

    /// Total bytes written.
    fn len(&self) -> u64 {
        self.offset as u64
    }

    /// Copy all values to the target writer, then delete the temp file.
    fn copy_to_and_cleanup<W: Write>(&mut self, target: &mut W) -> std::io::Result<()> {
        self.writer.flush()?;
        // Re-open the same file for reading (writer still holds the write handle
        // but we've flushed, so the read will see all data)
        let file = std::fs::File::open(&self.path)?;
        let mut reader = BufReader::with_capacity(8 * 1024 * 1024, file);
        std::io::copy(&mut reader, target)?;
        Ok(())
        // temp file is cleaned up by Drop
    }
}

impl Drop for ValuesWriter {
    fn drop(&mut self) {
        // Best-effort cleanup of temp file
        std::fs::remove_file(&self.path).ok();
    }
}

// ─────────────────────────────────────────────────────────────────────
// Value serializer
// ─────────────────────────────────────────────────────────────────────

/// Serializes a `Value` into the values buffer, returning (value_type, offset, len).
fn serialize_value(
    value: &Value,
    values: &mut ValuesWriter,
    strings: &mut StringTableBuilder,
) -> std::result::Result<(u8, u32, u32), Error> {
    let offset = values.offset();
    let r = |e: std::io::Error| Error::Internal(format!("values write: {e}"));
    match value {
        Value::Null => Ok((ValueType::Null as u8, offset, 0)),
        Value::Bool(b) => {
            values.write(&[if *b { 1 } else { 0 }]).map_err(r)?;
            Ok((ValueType::Bool as u8, offset, 1))
        }
        Value::Int64(v) => {
            values.write(&v.to_le_bytes()).map_err(r)?;
            Ok((ValueType::Int64 as u8, offset, 8))
        }
        Value::Float64(v) => {
            values.write(&v.to_le_bytes()).map_err(r)?;
            Ok((ValueType::Float64 as u8, offset, 8))
        }
        Value::String(s) => {
            let sref = strings.intern(s.as_str());
            values.write(&sref.offset.to_le_bytes()).map_err(r)?;
            values.write(&sref.len.to_le_bytes()).map_err(r)?;
            Ok((ValueType::String as u8, offset, 8))
        }
        Value::Bytes(b) => {
            let len = b.len() as u32;
            values.write(b).map_err(r)?;
            Ok((ValueType::Bytes as u8, offset, len))
        }
        Value::Timestamp(ts) => {
            values.write(&ts.as_micros().to_le_bytes()).map_err(r)?;
            Ok((ValueType::Timestamp as u8, offset, 8))
        }
        Value::Date(d) => {
            values.write(&d.as_days().to_le_bytes()).map_err(r)?;
            Ok((ValueType::Date as u8, offset, 4))
        }
        Value::Time(t) => {
            values.write(&t.as_nanos().to_le_bytes()).map_err(r)?;
            values.write(&t.offset_seconds().unwrap_or(0).to_le_bytes()).map_err(r)?;
            Ok((ValueType::Time as u8, offset, 12))
        }
        Value::Duration(d) => {
            values.write(&d.months().to_le_bytes()).map_err(r)?;
            values.write(&d.days().to_le_bytes()).map_err(r)?;
            values.write(&d.nanos().to_le_bytes()).map_err(r)?;
            Ok((ValueType::Duration as u8, offset, 24))
        }
        Value::ZonedDatetime(zdt) => {
            values.write(&zdt.as_timestamp().as_micros().to_le_bytes()).map_err(r)?;
            values.write(&zdt.offset_seconds().to_le_bytes()).map_err(r)?;
            Ok((ValueType::ZonedDatetime as u8, offset, 12))
        }
        Value::Vector(v) => {
            let byte_len = (v.len() * 4) as u32;
            for &f in v.iter() {
                values.write(&f.to_le_bytes()).map_err(r)?;
            }
            Ok((ValueType::Vector as u8, offset, byte_len))
        }
        Value::List(items) => {
            let start = values.offset();
            values.write(&(items.len() as u32).to_le_bytes()).map_err(r)?;
            for item in items.iter() {
                let (vt, _off, _len) = serialize_value(item, values, strings)?;
                values.write(&[vt]).map_err(r)?;
            }
            let total_len = values.offset() - start;
            Ok((ValueType::List as u8, start, total_len))
        }
        Value::Map(map) => {
            let start = values.offset();
            values.write(&(map.len() as u32).to_le_bytes()).map_err(r)?;
            for (k, v) in map.iter() {
                let key_ref = strings.intern(k.as_str());
                values.write(&key_ref.offset.to_le_bytes()).map_err(r)?;
                values.write(&key_ref.len.to_le_bytes()).map_err(r)?;
                let (vt, _off, _len) = serialize_value(v, values, strings)?;
                values.write(&[vt]).map_err(r)?;
            }
            let total_len = values.offset() - start;
            Ok((ValueType::Map as u8, start, total_len))
        }
        Value::Path { nodes, edges } => {
            let start = values.offset();
            values.write(&(nodes.len() as u32).to_le_bytes()).map_err(r)?;
            values.write(&(edges.len() as u32).to_le_bytes()).map_err(r)?;
            for n in nodes.iter() {
                let (vt, _, _) = serialize_value(n, values, strings)?;
                values.write(&[vt]).map_err(r)?;
            }
            for e in edges.iter() {
                let (vt, _, _) = serialize_value(e, values, strings)?;
                values.write(&[vt]).map_err(r)?;
            }
            let total_len = values.offset() - start;
            Ok((ValueType::Path as u8, start, total_len))
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────

/// Writes the store contents to a native v2 `.obrain` file.
///
/// This produces a file that can be mmap'd and read with zero deserialization.
/// If `hnsw_section` is provided, it's written as the HnswTopology section.
pub fn write_native_v2(
    store: &LpgStore,
    path: &Path,
    epoch: u64,
    transaction_id: u64,
) -> Result<()> {
    write_native_v2_inner(store, path, epoch, transaction_id, None)
}

/// Writes the store contents plus optional HNSW topology to a native v2 file.
pub fn write_native_v2_with_hnsw(
    store: &LpgStore,
    path: &Path,
    epoch: u64,
    transaction_id: u64,
    hnsw_section: &[u8],
) -> Result<()> {
    write_native_v2_inner(store, path, epoch, transaction_id, Some(hnsw_section))
}

fn write_native_v2_inner(
    store: &LpgStore,
    path: &Path,
    epoch: u64,
    transaction_id: u64,
    hnsw_section: Option<&[u8]>,
) -> Result<()> {
    let file = std::fs::File::create(path)
        .map_err(|e| Error::Internal(format!("create file: {e}")))?;
    let mut w = BufWriter::with_capacity(8 * 1024 * 1024, file); // 8MB buffer

    // ── Phase 1: collect structural data in memory, spill values to disk ──

    let mut strings = StringTableBuilder::new();
    let values_dir = path.parent().unwrap_or(Path::new("."));
    let mut values_writer = ValuesWriter::new(values_dir)
        .map_err(|e| Error::Internal(format!("create values temp file: {e}")))?;

    // 1a. Build label catalog: label_name → label_id (u32)
    let all_labels = store.all_labels();
    let mut label_name_to_id: HashMap<String, u32> = HashMap::new();
    let mut label_refs: Vec<StringRef> = Vec::with_capacity(all_labels.len());
    for (i, label) in all_labels.iter().enumerate() {
        label_name_to_id.insert(label.clone(), i as u32);
        label_refs.push(strings.intern(label));
    }

    // 1b. Build edge type catalog: edge_type → type_id (stored as StringRef)
    let all_edge_types = store.all_edge_types();
    let mut edge_type_refs: Vec<StringRef> = Vec::with_capacity(all_edge_types.len());
    for et in &all_edge_types {
        edge_type_refs.push(strings.intern(et));
    }

    // 1c. Collect nodes → NodeSlots + PropEntries
    let node_count = store.node_count();
    let edge_count = store.edge_count();
    // Initialize with FLAG_DELETED so gaps in the ID space are marked as tombstones
    let tombstone_node = {
        let mut s = NodeSlot::default();
        s.flags = NodeSlot::FLAG_DELETED;
        s
    };
    let mut node_slots: Vec<NodeSlot> = vec![tombstone_node; node_count];
    let mut prop_entries: Vec<PropEntry> = Vec::with_capacity(node_count * 5);

    for node in store.all_nodes() {
        let idx = node.id.0 as usize;
        if idx >= node_slots.len() {
            node_slots.resize(idx + 1, tombstone_node);
        }

        // Build label_mask bitmap
        let mut label_mask: u64 = 0;
        for label in &node.labels {
            if let Some(&lid) = label_name_to_id.get(label.as_str()) {
                if lid < 64 {
                    label_mask |= 1u64 << lid;
                }
            }
        }

        // Collect properties (values spilled to temp file on disk)
        let prop_offset = prop_entries.len() as u32;
        let mut prop_count = 0u16;
        for (key, value) in &node.properties {
            let key_ref = strings.intern(key.as_str());
            let (vt, voff, vlen) = serialize_value(value, &mut values_writer, &mut strings)?;
            prop_entries.push(PropEntry::new(key_ref, vt, voff, vlen));
            prop_count += 1;
        }

        node_slots[idx] = NodeSlot::new(node.id.0, label_mask, prop_offset, prop_count);
    }

    // Build a set of live node IDs for orphan-edge filtering
    let live_node_ids: HashSet<u64> = node_slots
        .iter()
        .enumerate()
        .filter(|(_, s)| !s.is_deleted())
        .map(|(i, _)| i as u64)
        .collect();

    // 1d. Collect edges → EdgeSlots + PropEntries
    let tombstone_edge = {
        let mut s = EdgeSlot::default();
        s.flags = EdgeSlot::FLAG_DELETED;
        s
    };
    let mut edge_slots: Vec<EdgeSlot> = vec![tombstone_edge; edge_count];
    for edge in store.all_edges() {
        // Skip orphan edges (src or dst points to a deleted/non-existent node)
        if !live_node_ids.contains(&edge.src.0) || !live_node_ids.contains(&edge.dst.0) {
            continue;
        }

        let idx = edge.id.0 as usize;
        if idx >= edge_slots.len() {
            edge_slots.resize(idx + 1, tombstone_edge);
        }

        let type_ref = strings.intern(edge.edge_type.as_str());

        // Collect properties (values spilled to temp file on disk)
        let prop_offset = prop_entries.len() as u32;
        let mut prop_count = 0u16;
        for (key, value) in &edge.properties {
            let key_ref = strings.intern(key.as_str());
            let (vt, voff, vlen) = serialize_value(value, &mut values_writer, &mut strings)?;
            prop_entries.push(PropEntry::new(key_ref, vt, voff, vlen));
            prop_count += 1;
        }

        edge_slots[idx] = EdgeSlot::new(
            edge.id.0,
            edge.src.0,
            edge.dst.0,
            type_ref,
            prop_offset,
            prop_count,
        );
    }

    // 1e. Build CSR forward + backward adjacency (using live nodes/edges only)
    let actual_node_count = node_slots.len();
    let (csr_fwd_offsets, csr_fwd_targets, csr_fwd_edge_ids) =
        build_csr_forward(store, actual_node_count, &live_node_ids);
    let (csr_bwd_offsets, csr_bwd_targets, csr_bwd_edge_ids) =
        build_csr_backward(store, actual_node_count, &live_node_ids);

    // 1f. Finalize string table
    let string_table = strings.into_bytes();

    // ── Phase 2: write file ──

    // Reserve space for headers + TOC (will be written at the end)
    let header_reserve = SECTIONS_START;
    w.seek(SeekFrom::Start(header_reserve))
        .map_err(|e| Error::Internal(format!("seek: {e}")))?;

    let mut toc = TocBuilder::new();
    let mut cursor = header_reserve;

    // Helper: write a section, record in TOC
    macro_rules! write_section {
        ($section_type:expr, $data:expr) => {{
            // Align to SECTION_ALIGN
            let padding = align_padding(cursor, SECTION_ALIGN);
            if padding > 0 {
                w.write_all(&vec![0u8; padding as usize])
                    .map_err(|e| Error::Internal(format!("write padding: {e}")))?;
                cursor += padding;
            }

            let data: &[u8] = $data;
            let offset = cursor;
            let length = data.len() as u64;
            w.write_all(data)
                .map_err(|e| Error::Internal(format!("write section: {e}")))?;
            cursor += length;

            // CRC32 placeholder (0 for now, can be computed later)
            toc.add($section_type, offset, length, 0);
        }};
    }

    // Write sections in order
    write_section!(SectionType::Strings, &string_table);

    let nodes_bytes = obrain_common::types::flat::slice_as_bytes(&node_slots);
    write_section!(SectionType::Nodes, nodes_bytes);

    let edges_bytes = obrain_common::types::flat::slice_as_bytes(&edge_slots);
    write_section!(SectionType::Edges, edges_bytes);

    let props_bytes = obrain_common::types::flat::slice_as_bytes(&prop_entries);
    write_section!(SectionType::Properties, props_bytes);

    // PropertyValues: stream from temp file on disk (not held in RAM)
    {
        values_writer.flush()
            .map_err(|e| Error::Internal(format!("flush values: {e}")))?;
        let values_len = values_writer.len();

        // Align
        let padding = align_padding(cursor, SECTION_ALIGN);
        if padding > 0 {
            w.write_all(&vec![0u8; padding as usize])
                .map_err(|e| Error::Internal(format!("write padding: {e}")))?;
            cursor += padding;
        }

        let offset = cursor;
        values_writer.copy_to_and_cleanup(&mut w)
            .map_err(|e| Error::Internal(format!("copy values: {e}")))?;
        cursor += values_len;

        toc.add(SectionType::PropertyValues, offset, values_len, 0);
    }

    // CSR Forward
    let fwd_off_bytes = u64_slice_as_bytes(&csr_fwd_offsets);
    write_section!(SectionType::CsrForwardOffsets, fwd_off_bytes);

    let fwd_tgt_bytes = u64_slice_as_bytes(&csr_fwd_targets);
    write_section!(SectionType::CsrForwardTargets, fwd_tgt_bytes);

    let fwd_eid_bytes = u64_slice_as_bytes(&csr_fwd_edge_ids);
    write_section!(SectionType::CsrForwardEdgeIds, fwd_eid_bytes);

    // CSR Backward
    let bwd_off_bytes = u64_slice_as_bytes(&csr_bwd_offsets);
    write_section!(SectionType::CsrBackwardOffsets, bwd_off_bytes);

    let bwd_tgt_bytes = u64_slice_as_bytes(&csr_bwd_targets);
    write_section!(SectionType::CsrBackwardTargets, bwd_tgt_bytes);

    let bwd_eid_bytes = u64_slice_as_bytes(&csr_bwd_edge_ids);
    write_section!(SectionType::CsrBackwardEdgeIds, bwd_eid_bytes);

    // Label catalog (array of StringRef)
    let label_cat_bytes = obrain_common::types::flat::slice_as_bytes(&label_refs);
    write_section!(SectionType::LabelCatalog, label_cat_bytes);

    // Edge type catalog
    let et_cat_bytes = obrain_common::types::flat::slice_as_bytes(&edge_type_refs);
    write_section!(SectionType::EdgeTypeCatalog, et_cat_bytes);

    // HNSW topology (optional)
    if let Some(hnsw_data) = hnsw_section {
        write_section!(SectionType::HnswTopology, hnsw_data);
    }

    // ── Phase 3: write headers at the start ──

    // Build TOC page
    let toc_page = toc.build();

    // Seek back and write FileHeader
    w.seek(SeekFrom::Start(0))
        .map_err(|e| Error::Internal(format!("seek header: {e}")))?;

    let file_header = FileHeader::new_native();
    let fh_bytes = bincode::serde::encode_to_vec(&file_header, bincode::config::standard())
        .map_err(|e| Error::Internal(format!("encode file header: {e}")))?;
    w.write_all(&fh_bytes)
        .map_err(|e| Error::Internal(format!("write file header: {e}")))?;

    // Pad to FILE_HEADER_SIZE
    let fh_pad = FILE_HEADER_SIZE as usize - fh_bytes.len();
    w.write_all(&vec![0u8; fh_pad])
        .map_err(|e| Error::Internal(format!("write fh padding: {e}")))?;

    // Write DbHeader slot 0
    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;

    let db_header = DbHeader {
        iteration: 1,
        checksum: 0, // TODO: compute CRC32 over all sections
        snapshot_length: cursor - SECTIONS_START,
        epoch,
        transaction_id,
        node_count: node_slots.len() as u64,
        edge_count: edge_slots.len() as u64,
        timestamp_ms: now_ms,
    };
    let dbh_bytes = bincode::serde::encode_to_vec(&db_header, bincode::config::standard())
        .map_err(|e| Error::Internal(format!("encode db header: {e}")))?;
    w.write_all(&dbh_bytes)
        .map_err(|e| Error::Internal(format!("write db header 0: {e}")))?;
    let dbh_pad = DB_HEADER_SIZE as usize - dbh_bytes.len();
    w.write_all(&vec![0u8; dbh_pad])
        .map_err(|e| Error::Internal(format!("write dbh0 padding: {e}")))?;

    // DbHeader slot 1 (empty)
    let empty_dbh =
        bincode::serde::encode_to_vec(&DbHeader::EMPTY, bincode::config::standard())
            .map_err(|e| Error::Internal(format!("encode empty db header: {e}")))?;
    w.write_all(&empty_dbh)
        .map_err(|e| Error::Internal(format!("write db header 1: {e}")))?;
    let dbh1_pad = DB_HEADER_SIZE as usize - empty_dbh.len();
    w.write_all(&vec![0u8; dbh1_pad])
        .map_err(|e| Error::Internal(format!("write dbh1 padding: {e}")))?;

    // Write TOC at offset 12 KiB
    w.write_all(&toc_page)
        .map_err(|e| Error::Internal(format!("write toc: {e}")))?;

    w.flush()
        .map_err(|e| Error::Internal(format!("flush: {e}")))?;

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// CSR builders
// ─────────────────────────────────────────────────────────────────────

/// Build CSR forward adjacency from the store.
/// Returns (offsets, targets, edge_ids).
fn build_csr_forward(
    store: &LpgStore,
    node_count: usize,
    live_node_ids: &HashSet<u64>,
) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
    // Collect all outgoing edges per node (skip orphan edges)
    let mut adj: Vec<Vec<(u64, u64)>> = vec![Vec::new(); node_count]; // (target, edge_id)

    for edge in store.all_edges() {
        if !live_node_ids.contains(&edge.src.0) || !live_node_ids.contains(&edge.dst.0) {
            continue;
        }
        let src = edge.src.0 as usize;
        if src < node_count {
            adj[src].push((edge.dst.0, edge.id.0));
        }
    }

    // Build CSR arrays
    let mut offsets = Vec::with_capacity(node_count + 1);
    let total_edges: usize = adj.iter().map(|v| v.len()).sum();
    let mut targets = Vec::with_capacity(total_edges);
    let mut edge_ids = Vec::with_capacity(total_edges);

    let mut offset = 0u64;
    for neighbors in &adj {
        offsets.push(offset);
        for &(target, eid) in neighbors {
            targets.push(target);
            edge_ids.push(eid);
        }
        offset += neighbors.len() as u64;
    }
    offsets.push(offset);

    (offsets, targets, edge_ids)
}

/// Build CSR backward adjacency from the store.
/// Returns (offsets, targets, edge_ids).
fn build_csr_backward(
    store: &LpgStore,
    node_count: usize,
    live_node_ids: &HashSet<u64>,
) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
    // Collect all incoming edges per node (skip orphan edges)
    let mut adj: Vec<Vec<(u64, u64)>> = vec![Vec::new(); node_count]; // (source, edge_id)

    for edge in store.all_edges() {
        if !live_node_ids.contains(&edge.src.0) || !live_node_ids.contains(&edge.dst.0) {
            continue;
        }
        let dst = edge.dst.0 as usize;
        if dst < node_count {
            adj[dst].push((edge.src.0, edge.id.0));
        }
    }

    // Build CSR arrays
    let mut offsets = Vec::with_capacity(node_count + 1);
    let total_edges: usize = adj.iter().map(|v| v.len()).sum();
    let mut targets = Vec::with_capacity(total_edges);
    let mut edge_ids = Vec::with_capacity(total_edges);

    let mut offset = 0u64;
    for neighbors in &adj {
        offsets.push(offset);
        for &(target, eid) in neighbors {
            targets.push(target);
            edge_ids.push(eid);
        }
        offset += neighbors.len() as u64;
    }
    offsets.push(offset);

    (offsets, targets, edge_ids)
}

// ──────────────────────────────────────────────────────────────���──────
// Helpers
// ───────────────────────────────────────────────────────���─────────────

/// Compute padding needed to reach the next aligned offset.
#[inline]
fn align_padding(offset: u64, align: u64) -> u64 {
    let rem = offset % align;
    if rem == 0 {
        0
    } else {
        align - rem
    }
}

/// View a `&[u64]` as `&[u8]`.
#[inline]
fn u64_slice_as_bytes(slice: &[u64]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len() * 8)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Multi-index HNSW envelope
// ─────────────────────────────────────────────────────────────────────
//
// Wire format (all little-endian):
//   [count: u32]
//   for each index:
//     [key_len: u32] [key_bytes: key_len] [topo_len: u32] [topo_bytes: topo_len]
//
// `key` is the `"label:property"` string identifying the vector index.
// `topo_bytes` is the output of `HnswFlatTopology::to_bytes()`.

/// Serializes multiple HNSW indexes into a single envelope blob.
///
/// Each entry is a `(key, topology_bytes)` pair where `key` is the
/// `"label:property"` identifier and `topology_bytes` comes from
/// `HnswFlatTopology::to_bytes()`.
pub fn pack_hnsw_indexes(entries: &[(&str, Vec<u8>)]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(4 + entries.len() * 64);
    buf.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for (key, topo) in entries {
        let key_bytes = key.as_bytes();
        buf.extend_from_slice(&(key_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(key_bytes);
        buf.extend_from_slice(&(topo.len() as u32).to_le_bytes());
        buf.extend_from_slice(topo);
    }
    buf
}

/// Deserializes a multi-index HNSW envelope into `(key, topology_bytes)` pairs.
///
/// Returns `None` if the data is malformed.
pub fn unpack_hnsw_indexes(data: &[u8]) -> Option<Vec<(String, &[u8])>> {
    if data.len() < 4 {
        return None;
    }
    let count = u32::from_le_bytes(data[..4].try_into().ok()?) as usize;
    let mut pos = 4;
    let mut result = Vec::with_capacity(count);

    for _ in 0..count {
        if pos + 4 > data.len() {
            return None;
        }
        let key_len = u32::from_le_bytes(data[pos..pos + 4].try_into().ok()?) as usize;
        pos += 4;
        if pos + key_len > data.len() {
            return None;
        }
        let key = std::str::from_utf8(&data[pos..pos + key_len]).ok()?;
        pos += key_len;

        if pos + 4 > data.len() {
            return None;
        }
        let topo_len = u32::from_le_bytes(data[pos..pos + 4].try_into().ok()?) as usize;
        pos += 4;
        if pos + topo_len > data.len() {
            return None;
        }
        let topo = &data[pos..pos + topo_len];
        pos += topo_len;

        result.push((key.to_string(), topo));
    }
    Some(result)
}

/// Exports all HNSW indexes from the store as a packed envelope blob.
///
/// Returns `None` if there are no vector indexes.
#[cfg(feature = "vector-index")]
pub fn export_hnsw_section(store: &LpgStore) -> Option<Vec<u8>> {
    let entries = store.vector_index_entries();
    if entries.is_empty() {
        return None;
    }

    let packed: Vec<(&str, Vec<u8>)> = entries
        .iter()
        .map(|(key, index)| {
            let flat = index.export_flat();
            (key.as_str(), flat.to_bytes())
        })
        .collect();

    Some(pack_hnsw_indexes(&packed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hnsw_envelope_roundtrip() {
        let entries = vec![
            ("Person:embedding", vec![1u8, 2, 3, 4, 5]),
            ("Document:vector", vec![10, 20, 30]),
        ];
        let packed = pack_hnsw_indexes(&entries);
        let unpacked = unpack_hnsw_indexes(&packed).expect("unpack should succeed");

        assert_eq!(unpacked.len(), 2);
        assert_eq!(unpacked[0].0, "Person:embedding");
        assert_eq!(unpacked[0].1, &[1, 2, 3, 4, 5]);
        assert_eq!(unpacked[1].0, "Document:vector");
        assert_eq!(unpacked[1].1, &[10, 20, 30]);
    }

    #[test]
    fn test_hnsw_envelope_empty() {
        let entries: Vec<(&str, Vec<u8>)> = vec![];
        let packed = pack_hnsw_indexes(&entries);
        let unpacked = unpack_hnsw_indexes(&packed).expect("unpack should succeed");
        assert!(unpacked.is_empty());
    }

    #[test]
    fn test_hnsw_envelope_malformed() {
        assert!(unpack_hnsw_indexes(&[]).is_none());
        assert!(unpack_hnsw_indexes(&[1, 0, 0, 0]).is_none()); // count=1 but no data
    }
}
