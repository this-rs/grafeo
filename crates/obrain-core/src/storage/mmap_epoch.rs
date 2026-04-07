//! Memory-mapped epoch file format for zero-copy cold storage.
//!
//! This module implements the on-disk format for frozen epoch data,
//! enabling near-instant database startup by memory-mapping epoch files
//! instead of replaying the full WAL.
//!
//! # File Layout
//!
//! ```text
//! ┌───────────────────────────────────────────────────────────────┐
//! │  EpochFileHeader (256 bytes, fixed)                          │
//! ├───────────────────────────────────────────────────────────────┤
//! │  Node Index: [IndexEntry; node_count]  (16 bytes each)       │
//! ├───────────────────────────────────────────────────────────────┤
//! │  Node Data: bincode-serialized NodeRecords                   │
//! ├─────────────── (8-byte aligned padding) ─────────────────────┤
//! │  Edge Index: [IndexEntry; edge_count]  (16 bytes each)       │
//! ├───────────────────────────────────────────────────────────────┤
//! │  Edge Data: bincode-serialized EdgeRecords                   │
//! ├─────────────── (8-byte aligned padding) ─────────────────────┤
//! │  Property Section (serialized properties per entity)         │
//! ├─────────────── (8-byte aligned padding) ─────────────────────┤
//! │  Label Section (serialized label mappings)                   │
//! ├─────────────── (8-byte aligned padding) ─────────────────────┤
//! │  Adjacency Section (serialized adjacency data)               │
//! └───────────────────────────────────────────────────────────────┘
//! ```
//!
//! The index entries are `#[repr(C)]` + `Pod`, allowing zero-copy casting
//! from the mmap'd region. Record data uses bincode serialization and is
//! deserialized on demand.
//!
//! # Usage
//!
//! ```no_run
//! use std::path::Path;
//! use obrain_core::storage::mmap_epoch::{MmapEpochBlock, write_epoch_file, EpochFileData};
//! use obrain_common::types::EpochId;
//!
//! // Write an epoch file
//! // write_epoch_file(Path::new("epoch_0001.oeb"), &data)?;
//!
//! // Open and read via mmap (zero-copy index, lazy record deserialization)
//! // let block = MmapEpochBlock::open(Path::new("epoch_0001.oeb"))?;
//! // let node = block.get_node_by_id(42);
//! ```

use std::fs::File;
use std::io::{self, Write};
use std::path::Path;

use bytemuck::{Pod, Zeroable};
use memmap2::Mmap;
use obrain_common::types::EpochId;

use super::epoch_store::{IndexEntry, ZoneMap};
use crate::graph::lpg::{EdgeRecord, NodeRecord};

/// Index entry for property/adjacency sections, supporting >4GB offsets.
///
/// Used in the v2 indexed property and adjacency format to enable
/// per-entity binary search without deserializing the entire section.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[allow(clippy::pub_underscore_fields)]
pub struct WideIndexEntry {
    /// The entity ID (node or edge) this entry refers to.
    pub entity_id: u64,
    /// Byte offset of the serialized data within the data region.
    pub data_offset: u64,
    /// Size of the serialized data in bytes.
    pub data_size: u32,
    /// Padding for 8-byte alignment.
    pub _pad: u32,
}

/// Magic bytes identifying an indexed property section (v2 format).
pub const INDEXED_PROPERTY_MAGIC: [u8; 8] = *b"PROPIDX\0";
/// Magic bytes identifying an indexed adjacency section (v2 format).
pub const INDEXED_ADJACENCY_MAGIC: [u8; 8] = *b"ADJIDX\0\0";

// =============================================================================
// Constants
// =============================================================================

/// Magic bytes identifying an Obrain epoch file.
pub const EPOCH_FILE_MAGIC: [u8; 8] = *b"OBRAIN01";

/// Current epoch file format version.
pub const EPOCH_FILE_VERSION: u32 = 1;

/// Size of the file header in bytes.
pub const HEADER_SIZE: usize = 256;

/// File extension for epoch files.
pub const EPOCH_FILE_EXTENSION: &str = "oeb";

// =============================================================================
// EpochFileHeader — fixed 256-byte header at offset 0
// =============================================================================

/// Fixed-size header at the start of every epoch file.
///
/// All offsets are absolute byte positions from the start of the file.
/// The header is exactly 256 bytes, padded with reserved space for
/// future extensions without breaking the format.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[allow(clippy::pub_underscore_fields)]
pub struct EpochFileHeader {
    /// Magic bytes: `OBRAIN01`.
    pub magic: [u8; 8],
    /// Format version (currently 1).
    pub version: u32,
    /// Padding for alignment.
    pub _pad0: u32,

    /// Epoch ID this file represents.
    pub epoch: u64,

    /// Number of node records.
    pub node_count: u32,
    /// Number of edge records.
    pub edge_count: u32,

    // --- Node section ---
    /// Absolute offset of the node index (array of IndexEntry).
    pub node_index_offset: u64,
    /// Absolute offset of the node data blob.
    pub node_data_offset: u64,
    /// Size of the node data blob in bytes.
    pub node_data_size: u64,

    // --- Edge section ---
    /// Absolute offset of the edge index (array of IndexEntry).
    pub edge_index_offset: u64,
    /// Absolute offset of the edge data blob.
    pub edge_data_offset: u64,
    /// Size of the edge data blob in bytes.
    pub edge_data_size: u64,

    // --- Extended sections (populated by T2: persist_epoch) ---
    /// Absolute offset of the property section.
    pub property_section_offset: u64,
    /// Size of the property section in bytes.
    pub property_section_size: u64,
    /// Absolute offset of the label section.
    pub label_section_offset: u64,
    /// Size of the label section in bytes.
    pub label_section_size: u64,
    /// Absolute offset of the adjacency section.
    pub adjacency_section_offset: u64,
    /// Size of the adjacency section in bytes.
    pub adjacency_section_size: u64,

    // --- Inline zone map (avoids a separate allocation) ---
    /// Minimum node ID in this epoch.
    pub zone_min_node_id: u64,
    /// Maximum node ID in this epoch.
    pub zone_max_node_id: u64,
    /// Minimum edge ID in this epoch.
    pub zone_min_edge_id: u64,
    /// Maximum edge ID in this epoch.
    pub zone_max_edge_id: u64,
    /// Node count for zone map filtering.
    pub zone_node_count: u32,
    /// Edge count for zone map filtering.
    pub zone_edge_count: u32,
    /// Minimum epoch (same as epoch for single-epoch files).
    pub zone_min_epoch: u64,
    /// Maximum epoch (same as epoch for single-epoch files).
    pub zone_max_epoch: u64,

    /// Reserved for future use. Must be zeroed.
    pub _reserved: [u64; 9],
}

// Compile-time assertion: header must be exactly 256 bytes.
const _: () = assert!(
    std::mem::size_of::<EpochFileHeader>() == HEADER_SIZE,
    "EpochFileHeader must be exactly 256 bytes"
);

impl EpochFileHeader {
    /// Validates the magic bytes and version.
    pub fn validate(&self) -> io::Result<()> {
        if self.magic != EPOCH_FILE_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "invalid epoch file magic: expected {:?}, got {:?}",
                    EPOCH_FILE_MAGIC, self.magic
                ),
            ));
        }
        if self.version != EPOCH_FILE_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "unsupported epoch file version: expected {}, got {}",
                    EPOCH_FILE_VERSION, self.version
                ),
            ));
        }
        Ok(())
    }

    /// Returns the epoch ID.
    #[must_use]
    pub fn epoch_id(&self) -> EpochId {
        EpochId::new(self.epoch)
    }

    /// Returns whether this file has a property section.
    #[must_use]
    pub fn has_properties(&self) -> bool {
        self.property_section_size > 0
    }

    /// Returns whether this file has a label section.
    #[must_use]
    pub fn has_labels(&self) -> bool {
        self.label_section_size > 0
    }

    /// Returns whether this file has an adjacency section.
    #[must_use]
    pub fn has_adjacency(&self) -> bool {
        self.adjacency_section_size > 0
    }
}

// =============================================================================
// EpochFileData — input data for writing an epoch file
// =============================================================================

/// All data needed to write an epoch file.
///
/// The node/edge index entries must be sorted by `entity_id` for binary search.
#[derive(Clone, Copy)]
pub struct EpochFileData<'a> {
    /// Epoch this file represents.
    pub epoch: EpochId,
    /// Node index entries (sorted by entity_id).
    pub node_index: &'a [IndexEntry],
    /// Serialized node data blob.
    pub node_data: &'a [u8],
    /// Edge index entries (sorted by entity_id).
    pub edge_index: &'a [IndexEntry],
    /// Serialized edge data blob.
    pub edge_data: &'a [u8],
    /// Zone map for predicate pushdown.
    pub zone_map: &'a ZoneMap,
    /// Optional serialized property data (populated by T2).
    pub property_data: Option<&'a [u8]>,
    /// Optional serialized label data (populated by T2).
    pub label_data: Option<&'a [u8]>,
    /// Optional serialized adjacency data (populated by T2).
    pub adjacency_data: Option<&'a [u8]>,
}

// =============================================================================
// write_epoch_file — serialize to disk
// =============================================================================

/// Writes padding bytes to align to 8-byte boundary.
fn write_alignment_padding(file: &mut File, current_offset: u64) -> io::Result<u64> {
    let remainder = current_offset % 8;
    if remainder != 0 {
        let padding = 8 - remainder;
        let zeros = [0u8; 8];
        file.write_all(&zeros[..padding as usize])?;
        Ok(current_offset + padding)
    } else {
        Ok(current_offset)
    }
}

/// Writes an epoch file to disk.
///
/// The file format allows zero-copy reads of the index entries via mmap,
/// with on-demand deserialization of the record data.
///
/// # Errors
///
/// Returns an error if the file cannot be created or written to.
pub fn write_epoch_file(path: &Path, data: &EpochFileData<'_>) -> io::Result<()> {
    let mut file = File::create(path)?;
    let index_entry_size = std::mem::size_of::<IndexEntry>() as u64;

    // Calculate section offsets
    let node_index_offset = HEADER_SIZE as u64;
    let node_index_size = data.node_index.len() as u64 * index_entry_size;
    let node_data_offset = node_index_offset + node_index_size;
    let node_data_size = data.node_data.len() as u64;

    // Edge section (8-byte aligned after node data)
    let edge_section_start = align_to_8(node_data_offset + node_data_size);
    let edge_index_offset = edge_section_start;
    let edge_index_size = data.edge_index.len() as u64 * index_entry_size;
    let edge_data_offset = edge_index_offset + edge_index_size;
    let edge_data_size = data.edge_data.len() as u64;

    // Extended sections (8-byte aligned)
    let mut next_offset = align_to_8(edge_data_offset + edge_data_size);

    let property_section_offset;
    let property_section_size;
    if let Some(prop_data) = data.property_data {
        property_section_offset = next_offset;
        property_section_size = prop_data.len() as u64;
        next_offset = align_to_8(next_offset + property_section_size);
    } else {
        property_section_offset = 0;
        property_section_size = 0;
    }

    let label_section_offset;
    let label_section_size;
    if let Some(lbl_data) = data.label_data {
        label_section_offset = next_offset;
        label_section_size = lbl_data.len() as u64;
        next_offset = align_to_8(next_offset + label_section_size);
    } else {
        label_section_offset = 0;
        label_section_size = 0;
    }

    let adjacency_section_offset;
    let adjacency_section_size;
    if let Some(adj_data) = data.adjacency_data {
        adjacency_section_offset = next_offset;
        adjacency_section_size = adj_data.len() as u64;
    } else {
        adjacency_section_offset = 0;
        adjacency_section_size = 0;
    }

    // Build header
    let header = EpochFileHeader {
        magic: EPOCH_FILE_MAGIC,
        version: EPOCH_FILE_VERSION,
        _pad0: 0,
        epoch: data.epoch.as_u64(),
        node_count: data.node_index.len() as u32,
        edge_count: data.edge_index.len() as u32,
        node_index_offset,
        node_data_offset,
        node_data_size,
        edge_index_offset,
        edge_data_offset,
        edge_data_size,
        property_section_offset,
        property_section_size,
        label_section_offset,
        label_section_size,
        adjacency_section_offset,
        adjacency_section_size,
        zone_min_node_id: data.zone_map.min_node_id,
        zone_max_node_id: data.zone_map.max_node_id,
        zone_min_edge_id: data.zone_map.min_edge_id,
        zone_max_edge_id: data.zone_map.max_edge_id,
        zone_node_count: data.zone_map.node_count,
        zone_edge_count: data.zone_map.edge_count,
        zone_min_epoch: data.zone_map.min_epoch,
        zone_max_epoch: data.zone_map.max_epoch,
        _reserved: [0u64; 9],
    };

    // Write header
    file.write_all(bytemuck::bytes_of(&header))?;

    // Write node index (already at correct offset = HEADER_SIZE)
    file.write_all(bytemuck::cast_slice(data.node_index))?;

    // Write node data
    file.write_all(data.node_data)?;

    // Align to 8 bytes before edge section
    let current = node_data_offset + node_data_size;
    let _ = write_alignment_padding(&mut file, current)?;

    // Write edge index
    file.write_all(bytemuck::cast_slice(data.edge_index))?;

    // Write edge data
    file.write_all(data.edge_data)?;

    // Extended sections
    let mut current = edge_data_offset + edge_data_size;

    if let Some(prop_data) = data.property_data {
        current = write_alignment_padding(&mut file, current)?;
        file.write_all(prop_data)?;
        current += prop_data.len() as u64;
    }

    if let Some(lbl_data) = data.label_data {
        current = write_alignment_padding(&mut file, current)?;
        file.write_all(lbl_data)?;
        current += lbl_data.len() as u64;
    }

    if let Some(adj_data) = data.adjacency_data {
        let _ = write_alignment_padding(&mut file, current)?;
        file.write_all(adj_data)?;
    }

    let _ = current; // consumed by extended sections above

    file.sync_all()?;
    Ok(())
}

/// Aligns a value up to the nearest 8-byte boundary.
const fn align_to_8(offset: u64) -> u64 {
    (offset + 7) & !7
}

// =============================================================================
// MmapEpochBlock — zero-copy reader over a memory-mapped epoch file
// =============================================================================

/// A memory-mapped epoch file providing zero-copy index access and
/// lazy deserialization of node/edge records.
///
/// The index entries are cast directly from the mmap'd region (zero allocation),
/// while record data is deserialized on demand via bincode.
///
/// # Thread Safety
///
/// `MmapEpochBlock` is `Send + Sync` — the underlying `Mmap` is immutable
/// and can be shared across threads safely.
pub struct MmapEpochBlock {
    /// The memory-mapped file.
    mmap: Mmap,
}

impl std::fmt::Debug for MmapEpochBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let header = self.header();
        f.debug_struct("MmapEpochBlock")
            .field("epoch", &header.epoch)
            .field("node_count", &header.node_count)
            .field("edge_count", &header.edge_count)
            .field("file_size", &self.mmap.len())
            .finish()
    }
}

impl MmapEpochBlock {
    /// Opens an epoch file and memory-maps it.
    ///
    /// The file is validated (magic + version check) but no data is
    /// deserialized — everything stays as mmap'd pages until accessed.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file cannot be opened
    /// - The file is too small to contain a valid header
    /// - The magic bytes or version don't match
    #[allow(unsafe_code)]
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let file_len = file.metadata()?.len() as usize;

        if file_len < HEADER_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "epoch file too small: {} bytes (minimum {})",
                    file_len, HEADER_SIZE
                ),
            ));
        }

        // SAFETY: We only create a read-only mmap, and we validate the
        // header before exposing any data. The file is not modified.
        let mmap = unsafe { Mmap::map(&file)? };

        let block = Self { mmap };
        block.header().validate()?;

        // Validate that section offsets are within file bounds
        block.validate_bounds(file_len)?;

        Ok(block)
    }

    /// Creates an `MmapEpochBlock` from an existing `Mmap` (for testing or
    /// when the caller already has the mapping).
    ///
    /// # Errors
    ///
    /// Returns an error if the header is invalid.
    pub fn from_mmap(mmap: Mmap) -> io::Result<Self> {
        if mmap.len() < HEADER_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "mmap too small for epoch file header",
            ));
        }
        let block = Self { mmap };
        block.header().validate()?;
        block.validate_bounds(block.mmap.len())?;
        Ok(block)
    }

    /// Validates that all section offsets and sizes fit within the file.
    fn validate_bounds(&self, file_len: usize) -> io::Result<()> {
        let h = self.header();
        let index_entry_size = std::mem::size_of::<IndexEntry>();

        // Node index bounds
        let node_idx_end = h.node_index_offset as usize + h.node_count as usize * index_entry_size;
        if node_idx_end > file_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "node index extends beyond file",
            ));
        }

        // Node data bounds
        let node_data_end = h.node_data_offset as usize + h.node_data_size as usize;
        if node_data_end > file_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "node data extends beyond file",
            ));
        }

        // Edge index bounds
        let edge_idx_end = h.edge_index_offset as usize + h.edge_count as usize * index_entry_size;
        if edge_idx_end > file_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "edge index extends beyond file",
            ));
        }

        // Edge data bounds
        let edge_data_end = h.edge_data_offset as usize + h.edge_data_size as usize;
        if edge_data_end > file_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "edge data extends beyond file",
            ));
        }

        // Extended section bounds
        if h.property_section_size > 0 {
            let end = h.property_section_offset as usize + h.property_section_size as usize;
            if end > file_len {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "property section extends beyond file",
                ));
            }
        }
        if h.label_section_size > 0 {
            let end = h.label_section_offset as usize + h.label_section_size as usize;
            if end > file_len {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "label section extends beyond file",
                ));
            }
        }
        if h.adjacency_section_size > 0 {
            let end = h.adjacency_section_offset as usize + h.adjacency_section_size as usize;
            if end > file_len {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "adjacency section extends beyond file",
                ));
            }
        }

        Ok(())
    }

    // =========================================================================
    // Header & metadata
    // =========================================================================

    /// Returns a reference to the file header (zero-copy from mmap).
    #[must_use]
    pub fn header(&self) -> &EpochFileHeader {
        bytemuck::from_bytes(&self.mmap[..HEADER_SIZE])
    }

    /// Returns the epoch ID.
    #[must_use]
    pub fn epoch(&self) -> EpochId {
        self.header().epoch_id()
    }

    /// Returns the number of nodes in this epoch.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.header().node_count as usize
    }

    /// Returns the number of edges in this epoch.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.header().edge_count as usize
    }

    /// Returns the total size of the memory-mapped file.
    #[must_use]
    pub fn file_size(&self) -> usize {
        self.mmap.len()
    }

    /// Reconstructs the `ZoneMap` from the header fields.
    #[must_use]
    pub fn zone_map(&self) -> ZoneMap {
        let h = self.header();
        ZoneMap {
            min_node_id: h.zone_min_node_id,
            max_node_id: h.zone_max_node_id,
            min_edge_id: h.zone_min_edge_id,
            max_edge_id: h.zone_max_edge_id,
            node_count: h.zone_node_count,
            edge_count: h.zone_edge_count,
            min_epoch: h.zone_min_epoch,
            max_epoch: h.zone_max_epoch,
        }
    }

    // =========================================================================
    // Zero-copy index access
    // =========================================================================

    /// Returns the node index as a slice (zero-copy cast from mmap).
    #[must_use]
    pub fn node_index(&self) -> &[IndexEntry] {
        let h = self.header();
        if h.node_count == 0 {
            return &[];
        }
        let start = h.node_index_offset as usize;
        let byte_len = h.node_count as usize * std::mem::size_of::<IndexEntry>();
        bytemuck::cast_slice(&self.mmap[start..start + byte_len])
    }

    /// Returns the edge index as a slice (zero-copy cast from mmap).
    #[must_use]
    pub fn edge_index(&self) -> &[IndexEntry] {
        let h = self.header();
        if h.edge_count == 0 {
            return &[];
        }
        let start = h.edge_index_offset as usize;
        let byte_len = h.edge_count as usize * std::mem::size_of::<IndexEntry>();
        bytemuck::cast_slice(&self.mmap[start..start + byte_len])
    }

    // =========================================================================
    // Raw data section access
    // =========================================================================

    /// Returns the raw node data blob.
    #[must_use]
    pub fn node_data(&self) -> &[u8] {
        let h = self.header();
        let start = h.node_data_offset as usize;
        let end = start + h.node_data_size as usize;
        &self.mmap[start..end]
    }

    /// Returns the raw edge data blob.
    #[must_use]
    pub fn edge_data(&self) -> &[u8] {
        let h = self.header();
        let start = h.edge_data_offset as usize;
        let end = start + h.edge_data_size as usize;
        &self.mmap[start..end]
    }

    /// Returns the raw property section data, if present.
    #[must_use]
    pub fn property_data(&self) -> Option<&[u8]> {
        let h = self.header();
        if h.property_section_size == 0 {
            return None;
        }
        let start = h.property_section_offset as usize;
        let end = start + h.property_section_size as usize;
        Some(&self.mmap[start..end])
    }

    /// Returns the raw label section data, if present.
    #[must_use]
    pub fn label_data_section(&self) -> Option<&[u8]> {
        let h = self.header();
        if h.label_section_size == 0 {
            return None;
        }
        let start = h.label_section_offset as usize;
        let end = start + h.label_section_size as usize;
        Some(&self.mmap[start..end])
    }

    /// Returns the raw adjacency section data, if present.
    #[must_use]
    pub fn adjacency_data(&self) -> Option<&[u8]> {
        let h = self.header();
        if h.adjacency_section_size == 0 {
            return None;
        }
        let start = h.adjacency_section_offset as usize;
        let end = start + h.adjacency_section_size as usize;
        Some(&self.mmap[start..end])
    }

    // =========================================================================
    // Lookup methods
    // =========================================================================

    /// Binary-searches the node index for the given entity ID.
    ///
    /// Returns the `IndexEntry` if found, enabling offset-based record access.
    #[must_use]
    pub fn find_node_entry(&self, entity_id: u64) -> Option<&IndexEntry> {
        let index = self.node_index();
        index
            .binary_search_by_key(&entity_id, |e| e.entity_id)
            .ok()
            .map(|i| &index[i])
    }

    /// Binary-searches the edge index for the given entity ID.
    #[must_use]
    pub fn find_edge_entry(&self, entity_id: u64) -> Option<&IndexEntry> {
        let index = self.edge_index();
        index
            .binary_search_by_key(&entity_id, |e| e.entity_id)
            .ok()
            .map(|i| &index[i])
    }

    /// Gets a node record by entity ID.
    ///
    /// Performs a binary search on the mmap'd index, then deserializes
    /// the record from the data section.
    #[must_use]
    pub fn get_node_by_id(&self, entity_id: u64) -> Option<NodeRecord> {
        let entry = self.find_node_entry(entity_id)?;
        self.get_node(entry.offset, entry.length)
    }

    /// Gets a node record by offset and length within the data section.
    #[must_use]
    pub fn get_node(&self, offset: u32, length: u16) -> Option<NodeRecord> {
        let data = self.node_data();
        let start = offset as usize;
        let end = start + length as usize;
        if end > data.len() {
            return None;
        }
        let config = bincode::config::standard();
        bincode::serde::decode_from_slice(&data[start..end], config)
            .ok()
            .map(|(record, _)| record)
    }

    /// Gets an edge record by entity ID.
    #[must_use]
    pub fn get_edge_by_id(&self, entity_id: u64) -> Option<EdgeRecord> {
        let entry = self.find_edge_entry(entity_id)?;
        self.get_edge(entry.offset, entry.length)
    }

    /// Gets an edge record by offset and length within the data section.
    #[must_use]
    pub fn get_edge(&self, offset: u32, length: u16) -> Option<EdgeRecord> {
        let data = self.edge_data();
        let start = offset as usize;
        let end = start + length as usize;
        if end > data.len() {
            return None;
        }
        let config = bincode::config::standard();
        bincode::serde::decode_from_slice(&data[start..end], config)
            .ok()
            .map(|(record, _)| record)
    }

    /// Checks if this epoch file might contain the given node ID
    /// (zone map predicate pushdown).
    #[must_use]
    pub fn might_contain_node(&self, node_id: u64) -> bool {
        let h = self.header();
        h.node_count > 0 && node_id >= h.zone_min_node_id && node_id <= h.zone_max_node_id
    }

    /// Checks if this epoch file might contain the given edge ID.
    #[must_use]
    pub fn might_contain_edge(&self, edge_id: u64) -> bool {
        let h = self.header();
        h.edge_count > 0 && edge_id >= h.zone_min_edge_id && edge_id <= h.zone_max_edge_id
    }

    // =========================================================================
    // Indexed property/adjacency access (v2 format)
    // =========================================================================

    /// Checks if the property section uses the indexed format (v2).
    pub fn has_indexed_properties(&self) -> bool {
        self.property_data()
            .is_some_and(|d| d.len() >= 32 && d[..8] == INDEXED_PROPERTY_MAGIC)
    }

    /// Returns the node property index entries (zero-copy from mmap).
    pub fn node_property_index(&self) -> Option<&[WideIndexEntry]> {
        let data = self.property_data()?;
        if data.len() < 32 || data[..8] != INDEXED_PROPERTY_MAGIC {
            return None;
        }
        let node_count = u64::from_le_bytes(data[8..16].try_into().ok()?) as usize;
        if node_count == 0 {
            return Some(&[]);
        }
        let start = 32; // after magic(8) + node_count(8) + edge_count(8) + data_region_offset(8)
        let byte_len = node_count * std::mem::size_of::<WideIndexEntry>();
        if start + byte_len > data.len() {
            return None;
        }
        Some(bytemuck::cast_slice(&data[start..start + byte_len]))
    }

    /// Returns the edge property index entries (zero-copy from mmap).
    pub fn edge_property_index(&self) -> Option<&[WideIndexEntry]> {
        let data = self.property_data()?;
        if data.len() < 32 || data[..8] != INDEXED_PROPERTY_MAGIC {
            return None;
        }
        let node_count = u64::from_le_bytes(data[8..16].try_into().ok()?) as usize;
        let edge_count = u64::from_le_bytes(data[16..24].try_into().ok()?) as usize;
        if edge_count == 0 {
            return Some(&[]);
        }
        let entry_size = std::mem::size_of::<WideIndexEntry>();
        let start = 32 + node_count * entry_size;
        let byte_len = edge_count * entry_size;
        if start + byte_len > data.len() {
            return None;
        }
        Some(bytemuck::cast_slice(&data[start..start + byte_len]))
    }

    /// Returns the data region offset within the property section.
    fn property_data_region_offset(&self) -> Option<usize> {
        let data = self.property_data()?;
        if data.len() < 32 || data[..8] != INDEXED_PROPERTY_MAGIC {
            return None;
        }
        Some(u64::from_le_bytes(data[24..32].try_into().ok()?) as usize)
    }

    /// Gets all properties for a single entity from the indexed property section.
    /// Returns None if not found. is_edge=false for nodes, true for edges.
    pub fn get_entity_properties(
        &self,
        entity_id: u64,
        is_edge: bool,
    ) -> Option<Vec<(String, obrain_common::types::Value)>> {
        let section = self.property_data()?;
        let data_region_offset = self.property_data_region_offset()?;
        let index = if is_edge {
            self.edge_property_index()?
        } else {
            self.node_property_index()?
        };
        let pos = index
            .binary_search_by_key(&entity_id, |e| e.entity_id)
            .ok()?;
        let entry = &index[pos];
        let start = data_region_offset + entry.data_offset as usize;
        let end = start + entry.data_size as usize;
        if end > section.len() {
            return None;
        }
        let config = bincode::config::standard();
        bincode::serde::decode_from_slice::<Vec<(String, obrain_common::types::Value)>, _>(
            &section[start..end],
            config,
        )
        .ok()
        .map(|(v, _)| v)
    }

    /// Checks if the adjacency section uses the indexed format.
    pub fn has_indexed_adjacency(&self) -> bool {
        self.adjacency_data()
            .is_some_and(|d| d.len() >= 32 && d[..8] == INDEXED_ADJACENCY_MAGIC)
    }

    /// Gets forward adjacency for a single node from the indexed adjacency section.
    pub fn get_forward_adj(&self, node_id: u64) -> Option<Vec<(u64, u64)>> {
        let section = self.adjacency_data()?;
        if section.len() < 32 || section[..8] != INDEXED_ADJACENCY_MAGIC {
            return None;
        }
        let forward_count = u64::from_le_bytes(section[8..16].try_into().ok()?) as usize;
        let _backward_count = u64::from_le_bytes(section[16..24].try_into().ok()?) as usize;
        let data_region_offset = u64::from_le_bytes(section[24..32].try_into().ok()?) as usize;
        if forward_count == 0 {
            return None;
        }
        let entry_size = std::mem::size_of::<WideIndexEntry>();
        let start = 32;
        let byte_len = forward_count * entry_size;
        if start + byte_len > section.len() {
            return None;
        }
        let index: &[WideIndexEntry] = bytemuck::cast_slice(&section[start..start + byte_len]);
        let pos = index.binary_search_by_key(&node_id, |e| e.entity_id).ok()?;
        let entry = &index[pos];
        let dstart = data_region_offset + entry.data_offset as usize;
        let dend = dstart + entry.data_size as usize;
        if dend > section.len() {
            return None;
        }
        let config = bincode::config::standard();
        bincode::serde::decode_from_slice::<Vec<(u64, u64)>, _>(&section[dstart..dend], config)
            .ok()
            .map(|(v, _)| v)
    }

    /// Gets backward adjacency for a single node from the indexed adjacency section.
    pub fn get_backward_adj(&self, node_id: u64) -> Option<Vec<(u64, u64)>> {
        let section = self.adjacency_data()?;
        if section.len() < 32 || section[..8] != INDEXED_ADJACENCY_MAGIC {
            return None;
        }
        let forward_count = u64::from_le_bytes(section[8..16].try_into().ok()?) as usize;
        let backward_count = u64::from_le_bytes(section[16..24].try_into().ok()?) as usize;
        let data_region_offset = u64::from_le_bytes(section[24..32].try_into().ok()?) as usize;
        if backward_count == 0 {
            return None;
        }
        let entry_size = std::mem::size_of::<WideIndexEntry>();
        let bwd_start = 32 + forward_count * entry_size;
        let byte_len = backward_count * entry_size;
        if bwd_start + byte_len > section.len() {
            return None;
        }
        let index: &[WideIndexEntry] =
            bytemuck::cast_slice(&section[bwd_start..bwd_start + byte_len]);
        let pos = index.binary_search_by_key(&node_id, |e| e.entity_id).ok()?;
        let entry = &index[pos];
        let dstart = data_region_offset + entry.data_offset as usize;
        let dend = dstart + entry.data_size as usize;
        if dend > section.len() {
            return None;
        }
        let config = bincode::config::standard();
        bincode::serde::decode_from_slice::<Vec<(u64, u64)>, _>(&section[dstart..dend], config)
            .ok()
            .map(|(v, _)| v)
    }

    // =========================================================================
    // Iteration
    // =========================================================================

    /// Iterates over all node entries and their deserialized records.
    ///
    /// This is useful for bulk loading into the in-memory store.
    pub fn iter_nodes(&self) -> impl Iterator<Item = (u64, NodeRecord)> + '_ {
        self.node_index().iter().filter_map(move |entry| {
            self.get_node(entry.offset, entry.length)
                .map(|record| (entry.entity_id, record))
        })
    }

    /// Iterates over all edge entries and their deserialized records.
    pub fn iter_edges(&self) -> impl Iterator<Item = (u64, EdgeRecord)> + '_ {
        self.edge_index().iter().filter_map(move |entry| {
            self.get_edge(entry.offset, entry.length)
                .map(|record| (entry.entity_id, record))
        })
    }
}

// =============================================================================
// Epoch directory management
// =============================================================================

/// Returns the canonical filename for an epoch file.
///
/// Format: `epoch_{epoch_id:08}.oeb`
#[must_use]
pub fn epoch_filename(epoch: EpochId) -> String {
    format!("epoch_{:08}.{}", epoch.as_u64(), EPOCH_FILE_EXTENSION)
}

/// Scans a directory for epoch files and returns them sorted by epoch ID.
///
/// # Errors
///
/// Returns an error if the directory cannot be read.
pub fn scan_epoch_files(dir: &Path) -> io::Result<Vec<(EpochId, std::path::PathBuf)>> {
    let mut epochs = Vec::new();

    if !dir.exists() {
        return Ok(epochs);
    }

    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path
            .extension()
            .is_some_and(|ext| ext == EPOCH_FILE_EXTENSION)
        {
            // Try to extract epoch ID from filename
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str())
                && let Some(epoch_str) = stem.strip_prefix("epoch_")
                && let Ok(epoch_id) = epoch_str.parse::<u64>()
            {
                epochs.push((EpochId::new(epoch_id), path));
            }
        }
    }

    epochs.sort_by_key(|(epoch, _)| epoch.as_u64());
    Ok(epochs)
}

/// Reads the epoch checkpoint file to determine the last persisted epoch
/// and corresponding WAL sequence.
///
/// The checkpoint file (`epoch_checkpoint.json`) is written atomically
/// after each epoch file is successfully flushed. It tells the database
/// which WAL entries can be skipped on recovery.
///
/// # Errors
///
/// Returns an error if the file exists but cannot be read or parsed.
pub fn read_epoch_checkpoint(dir: &Path) -> io::Result<Option<EpochCheckpoint>> {
    let path = dir.join("epoch_checkpoint.json");
    if !path.exists() {
        return Ok(None);
    }
    let content = std::fs::read_to_string(&path)?;
    let checkpoint: EpochCheckpoint = serde_json::from_str(&content)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    Ok(Some(checkpoint))
}

/// Writes the epoch checkpoint file atomically (write-to-temp + rename).
///
/// # Errors
///
/// Returns an error if the file cannot be written.
pub fn write_epoch_checkpoint(dir: &Path, checkpoint: &EpochCheckpoint) -> io::Result<()> {
    let path = dir.join("epoch_checkpoint.json");
    let tmp_path = dir.join("epoch_checkpoint.json.tmp");
    let content = serde_json::to_string_pretty(checkpoint).map_err(io::Error::other)?;
    std::fs::write(&tmp_path, content)?;
    std::fs::rename(&tmp_path, &path)?;
    Ok(())
}

/// Checkpoint metadata linking epoch files to WAL position.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EpochCheckpoint {
    /// The last epoch that has been fully persisted to an epoch file.
    pub last_persisted_epoch: u64,
    /// The WAL log sequence number up to which all data is covered
    /// by epoch files. On recovery, replay starts AFTER this sequence.
    pub wal_sequence: u64,
    /// Number of epoch files that exist.
    pub epoch_file_count: u64,
    /// Timestamp of last checkpoint (ISO 8601).
    pub timestamp: String,
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::lpg::{EdgeFlags, EdgeRecord, NodeFlags, NodeRecord};
    use obrain_common::types::{EdgeId, EpochId, NodeId};

    fn make_node(id: u64, epoch: u64) -> NodeRecord {
        NodeRecord {
            id: NodeId::new(id),
            epoch: EpochId::new(epoch),
            props_offset: 0,
            label_count: 0,
            _reserved: 0,
            props_count: 0,
            flags: NodeFlags(0),
            _padding: 0,
        }
    }

    fn make_edge(id: u64, src: u64, dst: u64, epoch: u64) -> EdgeRecord {
        EdgeRecord {
            id: EdgeId::new(id),
            src: NodeId::new(src),
            dst: NodeId::new(dst),
            type_id: 0,
            props_offset: 0,
            props_count: 0,
            flags: EdgeFlags(0),
            epoch: EpochId::new(epoch),
        }
    }

    fn build_test_data(
        epoch: u64,
        nodes: &[(u64, NodeRecord)],
        edges: &[(u64, EdgeRecord)],
    ) -> (Vec<IndexEntry>, Vec<u8>, Vec<IndexEntry>, Vec<u8>, ZoneMap) {
        let config = bincode::config::standard();

        let mut node_data = Vec::new();
        let mut node_index = Vec::new();
        for (id, record) in nodes {
            let offset = node_data.len() as u32;
            let serialized = bincode::serde::encode_to_vec(record, config).expect("serialize");
            let length = serialized.len() as u16;
            node_index.push(IndexEntry {
                entity_id: *id,
                offset,
                length,
                _pad: 0,
            });
            node_data.extend_from_slice(&serialized);
        }

        let mut edge_data = Vec::new();
        let mut edge_index = Vec::new();
        for (id, record) in edges {
            let offset = edge_data.len() as u32;
            let serialized = bincode::serde::encode_to_vec(record, config).expect("serialize");
            let length = serialized.len() as u16;
            edge_index.push(IndexEntry {
                entity_id: *id,
                offset,
                length,
                _pad: 0,
            });
            edge_data.extend_from_slice(&serialized);
        }

        let zone_map = ZoneMap {
            min_node_id: nodes.first().map_or(0, |(id, _)| *id),
            max_node_id: nodes.last().map_or(0, |(id, _)| *id),
            min_edge_id: edges.first().map_or(0, |(id, _)| *id),
            max_edge_id: edges.last().map_or(0, |(id, _)| *id),
            node_count: nodes.len() as u32,
            edge_count: edges.len() as u32,
            min_epoch: epoch,
            max_epoch: epoch,
        };

        (node_index, node_data, edge_index, edge_data, zone_map)
    }

    #[test]
    fn test_header_size() {
        assert_eq!(std::mem::size_of::<EpochFileHeader>(), HEADER_SIZE);
    }

    #[test]
    fn test_index_entry_size() {
        assert_eq!(std::mem::size_of::<IndexEntry>(), 16);
    }

    #[test]
    fn test_roundtrip_write_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("epoch_00000001.oeb");

        let epoch = 1u64;
        let nodes = vec![
            (1, make_node(1, epoch)),
            (5, make_node(5, epoch)),
            (10, make_node(10, epoch)),
        ];
        let edges = vec![
            (100, make_edge(100, 1, 5, epoch)),
            (200, make_edge(200, 5, 10, epoch)),
        ];

        let (node_index, node_data, edge_index, edge_data, zone_map) =
            build_test_data(epoch, &nodes, &edges);

        let data = EpochFileData {
            epoch: EpochId::new(epoch),
            node_index: &node_index,
            node_data: &node_data,
            edge_index: &edge_index,
            edge_data: &edge_data,
            zone_map: &zone_map,
            property_data: None,
            label_data: None,
            adjacency_data: None,
        };

        write_epoch_file(&path, &data).unwrap();

        // Open and verify
        let block = MmapEpochBlock::open(&path).unwrap();
        assert_eq!(block.epoch(), EpochId::new(1));
        assert_eq!(block.node_count(), 3);
        assert_eq!(block.edge_count(), 2);

        // Lookup by ID
        let n1 = block.get_node_by_id(1).unwrap();
        assert_eq!(n1.id, NodeId::new(1));

        let n5 = block.get_node_by_id(5).unwrap();
        assert_eq!(n5.id, NodeId::new(5));

        let n10 = block.get_node_by_id(10).unwrap();
        assert_eq!(n10.id, NodeId::new(10));

        // Non-existent node
        assert!(block.get_node_by_id(42).is_none());

        // Edge lookup
        let e100 = block.get_edge_by_id(100).unwrap();
        assert_eq!(e100.id, EdgeId::new(100));
        assert_eq!(e100.src, NodeId::new(1));
        assert_eq!(e100.dst, NodeId::new(5));

        let e200 = block.get_edge_by_id(200).unwrap();
        assert_eq!(e200.src, NodeId::new(5));
        assert_eq!(e200.dst, NodeId::new(10));

        assert!(block.get_edge_by_id(999).is_none());
    }

    #[test]
    fn test_zone_map_filtering() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("epoch_00000002.oeb");

        let epoch = 2u64;
        let nodes = vec![(100, make_node(100, epoch)), (200, make_node(200, epoch))];
        let edges: Vec<(u64, EdgeRecord)> = vec![];

        let (node_index, node_data, edge_index, edge_data, zone_map) =
            build_test_data(epoch, &nodes, &edges);

        let data = EpochFileData {
            epoch: EpochId::new(epoch),
            node_index: &node_index,
            node_data: &node_data,
            edge_index: &edge_index,
            edge_data: &edge_data,
            zone_map: &zone_map,
            property_data: None,
            label_data: None,
            adjacency_data: None,
        };
        write_epoch_file(&path, &data).unwrap();

        let block = MmapEpochBlock::open(&path).unwrap();
        assert!(block.might_contain_node(100));
        assert!(block.might_contain_node(150));
        assert!(block.might_contain_node(200));
        assert!(!block.might_contain_node(50));
        assert!(!block.might_contain_node(300));
        assert!(!block.might_contain_edge(1));
    }

    #[test]
    fn test_iteration() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("epoch_00000003.oeb");

        let epoch = 3u64;
        let nodes = vec![
            (1, make_node(1, epoch)),
            (2, make_node(2, epoch)),
            (3, make_node(3, epoch)),
        ];
        let edges = vec![(10, make_edge(10, 1, 2, epoch))];

        let (node_index, node_data, edge_index, edge_data, zone_map) =
            build_test_data(epoch, &nodes, &edges);

        let data = EpochFileData {
            epoch: EpochId::new(epoch),
            node_index: &node_index,
            node_data: &node_data,
            edge_index: &edge_index,
            edge_data: &edge_data,
            zone_map: &zone_map,
            property_data: None,
            label_data: None,
            adjacency_data: None,
        };
        write_epoch_file(&path, &data).unwrap();

        let block = MmapEpochBlock::open(&path).unwrap();

        let all_nodes: Vec<_> = block.iter_nodes().collect();
        assert_eq!(all_nodes.len(), 3);
        assert_eq!(all_nodes[0].0, 1);
        assert_eq!(all_nodes[1].0, 2);
        assert_eq!(all_nodes[2].0, 3);

        let all_edges: Vec<_> = block.iter_edges().collect();
        assert_eq!(all_edges.len(), 1);
        assert_eq!(all_edges[0].0, 10);
    }

    #[test]
    fn test_empty_epoch() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("epoch_empty.oeb");

        let zone_map = ZoneMap {
            min_node_id: 0,
            max_node_id: 0,
            min_edge_id: 0,
            max_edge_id: 0,
            node_count: 0,
            edge_count: 0,
            min_epoch: 1,
            max_epoch: 1,
        };

        let data = EpochFileData {
            epoch: EpochId::new(1),
            node_index: &[],
            node_data: &[],
            edge_index: &[],
            edge_data: &[],
            zone_map: &zone_map,
            property_data: None,
            label_data: None,
            adjacency_data: None,
        };
        write_epoch_file(&path, &data).unwrap();

        let block = MmapEpochBlock::open(&path).unwrap();
        assert_eq!(block.node_count(), 0);
        assert_eq!(block.edge_count(), 0);
        assert!(block.get_node_by_id(1).is_none());
    }

    #[test]
    fn test_scan_epoch_files() {
        let dir = tempfile::tempdir().unwrap();

        // Create some epoch files
        let zone_map = ZoneMap {
            min_node_id: 0,
            max_node_id: 0,
            min_edge_id: 0,
            max_edge_id: 0,
            node_count: 0,
            edge_count: 0,
            min_epoch: 0,
            max_epoch: 0,
        };
        let empty_data = EpochFileData {
            epoch: EpochId::new(0),
            node_index: &[],
            node_data: &[],
            edge_index: &[],
            edge_data: &[],
            zone_map: &zone_map,
            property_data: None,
            label_data: None,
            adjacency_data: None,
        };

        for epoch in [1, 3, 5] {
            let mut data = empty_data;
            data.epoch = EpochId::new(epoch);
            let path = dir.path().join(epoch_filename(EpochId::new(epoch)));
            write_epoch_file(&path, &data).unwrap();
        }

        // Also create a non-epoch file to ensure it's ignored
        std::fs::write(dir.path().join("other.txt"), "ignored").unwrap();

        let files = scan_epoch_files(dir.path()).unwrap();
        assert_eq!(files.len(), 3);
        assert_eq!(files[0].0, EpochId::new(1));
        assert_eq!(files[1].0, EpochId::new(3));
        assert_eq!(files[2].0, EpochId::new(5));
    }

    #[test]
    fn test_epoch_checkpoint_roundtrip() {
        let dir = tempfile::tempdir().unwrap();

        let checkpoint = EpochCheckpoint {
            last_persisted_epoch: 5,
            wal_sequence: 12345,
            epoch_file_count: 5,
            timestamp: "2026-04-07T00:00:00Z".to_string(),
        };

        write_epoch_checkpoint(dir.path(), &checkpoint).unwrap();
        let loaded = read_epoch_checkpoint(dir.path()).unwrap().unwrap();
        assert_eq!(loaded.last_persisted_epoch, 5);
        assert_eq!(loaded.wal_sequence, 12345);
        assert_eq!(loaded.epoch_file_count, 5);
    }

    #[test]
    fn test_epoch_checkpoint_missing() {
        let dir = tempfile::tempdir().unwrap();
        let result = read_epoch_checkpoint(dir.path()).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_invalid_magic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.oeb");
        let mut header = EpochFileHeader::zeroed();
        header.magic = *b"NOTVALID";
        std::fs::write(&path, bytemuck::bytes_of(&header)).unwrap();

        let result = MmapEpochBlock::open(&path);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("magic"));
    }

    #[test]
    fn test_file_too_small() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tiny.oeb");
        std::fs::write(&path, b"too small").unwrap();

        let result = MmapEpochBlock::open(&path);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("too small"));
    }

    /// T2 round-trip: property, label, and adjacency sections survive write → mmap open.
    #[test]
    fn test_extended_sections_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("epoch_extended.oeb");
        let config = bincode::config::standard();

        let epoch = 42u64;
        let nodes = vec![
            (1, make_node(1, epoch)),
            (2, make_node(2, epoch)),
            (3, make_node(3, epoch)),
        ];
        let edges = vec![
            (100, make_edge(100, 1, 2, epoch)),
            (200, make_edge(200, 2, 3, epoch)),
        ];

        let (node_index, node_data, edge_index, edge_data, zone_map) =
            build_test_data(epoch, &nodes, &edges);

        // Build property data (same format as collect_epoch_properties)
        // Format: (node_props, edge_props) where each is Vec<(u64, Vec<(String, Value)>)>
        use obrain_common::types::Value;
        let node_props: Vec<(u64, Vec<(String, Value)>)> = vec![
            (
                1,
                vec![
                    ("name".to_string(), Value::from("Alice")),
                    ("age".to_string(), Value::from(42i64)),
                ],
            ),
            (2, vec![("name".to_string(), Value::from("Bob"))]),
        ];
        let edge_props: Vec<(u64, Vec<(String, Value)>)> =
            vec![(100, vec![("weight".to_string(), Value::from(1.5f64))])];
        let property_data =
            bincode::serde::encode_to_vec((&node_props, &edge_props), config).unwrap();

        // Build label data: (label_entries, edge_types)
        let label_entries: Vec<(u64, Vec<String>)> = vec![
            (1, vec!["Person".to_string(), "Employee".to_string()]),
            (2, vec!["Person".to_string()]),
        ];
        let edge_types: Vec<String> = vec!["KNOWS".to_string()];
        let label_data =
            bincode::serde::encode_to_vec((&label_entries, &edge_types), config).unwrap();

        // Build adjacency data
        let forward: Vec<(u64, Vec<(u64, u64)>)> = vec![(1, vec![(2, 100)]), (2, vec![(3, 200)])];
        let backward: Vec<(u64, Vec<(u64, u64)>)> = vec![(2, vec![(1, 100)]), (3, vec![(2, 200)])];
        let adjacency_data = bincode::serde::encode_to_vec(&(forward, backward), config).unwrap();

        let data = EpochFileData {
            epoch: EpochId::new(epoch),
            node_index: &node_index,
            node_data: &node_data,
            edge_index: &edge_index,
            edge_data: &edge_data,
            zone_map: &zone_map,
            property_data: Some(&property_data),
            label_data: Some(&label_data),
            adjacency_data: Some(&adjacency_data),
        };

        write_epoch_file(&path, &data).unwrap();

        // Open and verify structure
        let block = MmapEpochBlock::open(&path).unwrap();
        assert_eq!(block.node_count(), 3);
        assert_eq!(block.edge_count(), 2);
        assert!(block.header().has_properties());
        assert!(block.header().has_labels());
        assert!(block.header().has_adjacency());

        // Verify property section is readable
        let prop_bytes = block
            .property_data()
            .expect("property section should exist");
        let (decoded_props, _): (
            (
                Vec<(u64, Vec<(String, Value)>)>,
                Vec<(u64, Vec<(String, Value)>)>,
            ),
            _,
        ) = bincode::serde::decode_from_slice(prop_bytes, config).unwrap();
        let (node_section, edge_section) = decoded_props;
        // Node 1 has 2 props, node 2 has 1 prop
        assert_eq!(node_section.len(), 2);
        assert_eq!(node_section[0].0, 1); // node_id=1
        assert_eq!(node_section[0].1.len(), 2); // 2 properties
        assert_eq!(edge_section.len(), 1); // 1 edge with properties

        // Verify label section
        let lbl_bytes = block
            .label_data_section()
            .expect("label section should exist");
        let (decoded_labels, _): ((Vec<(u64, Vec<String>)>, Vec<String>), _) =
            bincode::serde::decode_from_slice(lbl_bytes, config).unwrap();
        let (label_entries_decoded, edge_types_decoded) = decoded_labels;
        assert_eq!(label_entries_decoded.len(), 2);
        assert_eq!(label_entries_decoded[0].0, 1); // node_id=1
        assert!(label_entries_decoded[0].1.contains(&"Person".to_string()));
        assert!(label_entries_decoded[0].1.contains(&"Employee".to_string()));
        assert_eq!(label_entries_decoded[1].0, 2);
        assert_eq!(label_entries_decoded[1].1, vec!["Person".to_string()]);
        assert_eq!(edge_types_decoded, vec!["KNOWS".to_string()]);

        // Verify adjacency section
        let adj_bytes = block
            .adjacency_data()
            .expect("adjacency section should exist");
        let (decoded_adj, _): (
            (Vec<(u64, Vec<(u64, u64)>)>, Vec<(u64, Vec<(u64, u64)>)>),
            _,
        ) = bincode::serde::decode_from_slice(adj_bytes, config).unwrap();
        let (fwd, bwd) = decoded_adj;
        assert_eq!(fwd.len(), 2);
        assert_eq!(fwd[0], (1, vec![(2, 100)]));
        assert_eq!(fwd[1], (2, vec![(3, 200)]));
        assert_eq!(bwd.len(), 2);
        assert_eq!(bwd[0], (2, vec![(1, 100)]));
        assert_eq!(bwd[1], (3, vec![(2, 200)]));
    }
}
