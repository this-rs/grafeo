//! `substrate.meta` — 4 KiB file header + dictionary section.
//!
//! See `docs/rfc/substrate/format-spec.md` §2 and §9.

use bytemuck::{Pod, Zeroable};

pub const SUBSTRATE_MAGIC: [u8; 8] = *b"SUBSTRT\0";
pub const SUBSTRATE_FORMAT_VERSION: u32 = 1;
pub const META_HEADER_SIZE: usize = 96;
pub const META_FILE_SIZE: usize = 4096;

pub mod meta_flags {
    pub const COGNITIVE_ACTIVE: u32 = 1 << 0;
    pub const HILBERT_SORTED: u32 = 1 << 1;
}

/// Fixed-size header portion of `substrate.meta` (96 B).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct MetaHeader {
    pub magic: [u8; 8],
    pub format_version: u32,
    pub flags: u32,
    pub node_count: u64,
    pub edge_count: u64,
    pub property_page_count: u64,
    pub string_heap_size: u64,
    pub created_at: i64,
    pub last_checkpoint: i64,
    pub last_wal_offset: u64,
    pub tier0_dim_bits: u32,
    pub tier1_dim_bits: u32,
    pub tier2_dim: u32,
    pub hilbert_order: u32,
    pub schema_crc32: u32,
    pub embedding_matrix_seed: u32,
}

const _: [(); 1] = [(); (core::mem::size_of::<MetaHeader>() == META_HEADER_SIZE) as usize];

impl MetaHeader {
    pub fn new_blank() -> Self {
        Self {
            magic: SUBSTRATE_MAGIC,
            format_version: SUBSTRATE_FORMAT_VERSION,
            flags: 0,
            node_count: 0,
            edge_count: 0,
            property_page_count: 0,
            string_heap_size: 0,
            created_at: now_unix_seconds(),
            last_checkpoint: 0,
            last_wal_offset: 0,
            tier0_dim_bits: 128,
            tier1_dim_bits: 512,
            tier2_dim: 384,
            hilbert_order: 0,
            schema_crc32: 0,
            embedding_matrix_seed: 0,
        }
    }

    pub fn is_valid_magic(&self) -> bool {
        self.magic == SUBSTRATE_MAGIC
    }
}

fn now_unix_seconds() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meta_header_layout() {
        assert_eq!(core::mem::size_of::<MetaHeader>(), 96);
    }

    #[test]
    fn meta_header_blank() {
        let m = MetaHeader::new_blank();
        assert!(m.is_valid_magic());
        assert_eq!(m.format_version, SUBSTRATE_FORMAT_VERSION);
        assert_eq!(m.tier0_dim_bits, 128);
        assert_eq!(m.tier1_dim_bits, 512);
        assert_eq!(m.tier2_dim, 384);
    }

    #[test]
    fn meta_header_pod_roundtrip() {
        let m = MetaHeader {
            magic: SUBSTRATE_MAGIC,
            format_version: SUBSTRATE_FORMAT_VERSION,
            flags: meta_flags::COGNITIVE_ACTIVE,
            node_count: 12_345,
            edge_count: 67_890,
            property_page_count: 111,
            string_heap_size: 2048,
            created_at: 1_700_000_000,
            last_checkpoint: 1_700_000_042,
            last_wal_offset: 123_456,
            tier0_dim_bits: 128,
            tier1_dim_bits: 512,
            tier2_dim: 384,
            hilbert_order: 1,
            schema_crc32: 0xDEAD_BEEF,
            embedding_matrix_seed: 42,
        };
        let bytes: &[u8] = bytemuck::bytes_of(&m);
        assert_eq!(bytes.len(), 96);
        let m2: MetaHeader = *bytemuck::from_bytes::<MetaHeader>(bytes);
        assert_eq!(m, m2);
    }
}
