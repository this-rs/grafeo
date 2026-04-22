//! Multi-index HNSW envelope codec — all that remains of the v2 `.obrain`
//! native writer after T17 W3c slice 3.
//!
//! ## Context
//!
//! The v2 native writer used to serialize an `LpgStore` into a mmap-friendly
//! `.obrain` file. Slice 2 of T17 W3c (see commit `beacd400`) stubbed every
//! caller — `checkpoint_to_file` (v2 branch) and `save_as_obrain_file` now
//! return `Error::Internal` because those paths pulled data from the legacy
//! `LpgStore` field which holds a dummy empty store in substrate mode.
//!
//! Slice 3 (this file) deletes the producer side: ~700 LOC covering
//! `write_native_v2`, `write_native_v2_inner`, `StringTableBuilder`,
//! `ValuesWriter`, `serialize_value`, CSR builders, and the helpers
//! (`align_padding`, `u64_slice_as_bytes`, `export_hnsw_section`).
//!
//! ## What stays
//!
//! The two envelope-format primitives `pack_hnsw_indexes` /
//! `unpack_hnsw_indexes` are still live — `materialize_mmap_to_lpg`
//! (`mod.rs:1270`, `mod.rs:1399`) uses `unpack_hnsw_indexes` to rehydrate
//! legacy v2 files into an `LpgStore` during migration. Packing is kept to
//! keep the codec pair symmetric and to let the unit tests exercise it.
//!
//! ## Restore plan
//!
//! T17b will reintroduce a real snapshot producer — `SubstrateStore::
//! snapshot_to_path()` — writing directly from the substrate zones (mmap +
//! WAL tail), with no `LpgStore`-shaped intermediary.

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
