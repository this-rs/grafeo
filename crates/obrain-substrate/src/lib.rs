//! # obrain-substrate
//!
//! Topology-as-storage substrate: fixed-size `#[repr(C)]` Pod records, 4 KiB pages,
//! a native WAL for durability, and tiered (binary + f16) embedding indexes.
//!
//! This crate implements the on-disk format specified in
//! [`docs/rfc/substrate/format-spec.md`](../../docs/rfc/substrate/format-spec.md)
//! and the WAL protocol in
//! [`docs/rfc/substrate/wal-spec.md`](../../docs/rfc/substrate/wal-spec.md).
//!
//! ## Module map
//!
//! | module   | content                                                           |
//! |----------|-------------------------------------------------------------------|
//! | `record` | `NodeRecord` (32 B) and `EdgeRecord` (32 B) Pod types + accessors |
//! | `page`   | 4 KiB `PropertyPage` with CRC32 and slot directory                |
//! | `heap`   | 4 KiB `HeapPage` + in-memory `StringHeap` writer/reader           |
//! | `meta`   | `substrate.meta` header (magic, version, counters)                |
//! | `wal`    | `WalRecord` framing, `WalPayload` enum, encode/decode             |
//! | `tiers`  | `Tier0` / `Tier1` / `Tier2` embeddings + Hamming / cosine kernels |
//! | `error`  | `SubstrateError`, `SubstrateResult<T>`                            |
//!
//! ## Stability
//!
//! The on-disk types are **ABI-sensitive**. Any change to a `#[repr(C)]` record
//! MUST bump `meta::SUBSTRATE_FORMAT_VERSION` and provide a migration via the
//! `obrain-migrate` binary.

#![deny(unsafe_code)]

pub mod blob_column;
pub mod blob_column_registry;
pub mod checkpoint;
pub mod dict;
pub mod engram;
pub mod engram_bitset;
pub mod engram_recall;
pub mod error;
pub mod file;
pub mod geometric;
pub mod heap;
pub mod hilbert;
pub mod meta;
pub mod page;
pub mod popcount;
pub mod props_snapshot;
pub mod record;
pub mod replay;
pub mod retrieval;
pub mod simd;
pub mod store;
pub mod tier_persist;
pub mod tiered_scan;
pub mod tiers;
pub mod vec_column;
pub mod vec_column_registry;
pub mod wal;
pub mod wal_io;
pub mod warden;
pub mod writer;

pub use error::{SubstrateError, SubstrateResult};
pub use file::{SubstrateFile, Zone, ZoneFile, zone};
pub use geometric::{
    compute_all_forman, compute_all_node_curvatures, compute_hks_descriptors,
    compute_ricci_fast, compute_ricci_for_edge, effective_resistance_csr,
    geodesic_distance_csr, heat_diffuse_unweighted,
    heat_kernel_signature_with_csr, heat_step_unweighted, heat_step_unweighted_csr,
    heat_step_weighted, heat_step_weighted_csr, mark_incident_edges_stale,
    node_curvature, refresh_all_ricci, refresh_edge_ricci, refresh_ricci_for_nodes,
    refresh_stale_edges, CsrAdjacency, RicciRefreshStats, HKS_DT, HKS_HUTCHINSON_PROBES,
    HKS_T_GLOBAL, HKS_T_LOCAL, HKS_T_MESO,
};
pub use heap::{HEAP_PAGE_MAGIC, HEAP_PAGE_SIZE, HeapPage, HeapPageHeader, StringHeap};
pub use hilbert::{
    compute_hilbert_permutation, compute_hilbert_permutation_page_aligned, hilbert_index_2d,
    hilbert_key_from_features, invert_permutation,
};
pub use meta::{
    META_FILE_SIZE, META_HEADER_SIZE, MetaHeader, SUBSTRATE_FORMAT_VERSION, SUBSTRATE_MAGIC,
    meta_flags,
};
pub use page::{HeapRef, PAGE_SIZE, PROP_PAGE_MAGIC, PropertyPage, PropertyPageHeader, ValueTag};
pub use record::{
    COACT_EDGE_TYPE_NAME, EdgeRecord, NodeRecord, NODES_PER_PAGE, PackedScarUtilAff,
    SCAR_MAX_INTENSITY_Q5, U48, UTILITY_MAX_SCORE_Q5, affinity_to_q5, edge_flags, f32_to_q1_15,
    node_flags, q1_15_to_f32, q5_to_affinity, q5_to_scar, q5_to_utility, scar_to_q5, utility_to_q5,
};
pub use tiers::{
    L0_BITS, L1_BITS, L2_DIM, Tier0, Tier1, Tier2, f16_to_f32, f32_to_f16, tier0_hamming,
    tier0_topk, tier1_hamming, tier1_topk, tier2_cosine,
};
pub use wal::{PropValue, WAL_HEADER_SIZE, WalKind, WalPayload, WalRecord};
pub use wal_io::{SyncMode, WalIter, WalReader, WalWriter};
pub use blob_column::{
    BlobColSpec, BlobColumnHeader, BlobColumnReader, BlobColumnWriter, BlobSlotEntry,
    BLOB_COLUMN_MAGIC, BLOB_COLUMN_VERSION, BLOB_HEADER_SIZE, BLOB_SLOT_STRIDE,
};
pub use checkpoint::{checkpoint, CheckpointStats};
pub use dict::DictSnapshot;
pub use engram::{EngramZone, ENGRAM_HEADER_SIZE, ENGRAM_MAGIC, ENGRAM_ZONE_VERSION, MAX_ENGRAM_ID};
pub use engram_bitset::{engram_bit_mask, EngramBitsetColumn, BITSET_ENTRY_SIZE};
pub use engram_recall::{scan_overlap, scan_overlap_scalar, top_k_by_overlap};
pub use popcount::{
    backend as popcount_backend, t0_hamming, t1_hamming, xor_popcount_t0, xor_popcount_t0_scalar,
    xor_popcount_t1, xor_popcount_t1_scalar, Backend as PopcountBackend,
};
pub use replay::{replay_from, ReplayStats};
pub use retrieval::{NodeOffset, SubstrateTieredIndex, VectorIndex};
pub use props_snapshot::{PropertiesSnapshotV1, PropertiesStreamingWriter, PROPS_FILENAME};
pub use store::{SubstrateStore, SKIP_ON_LOAD_PROP_KEYS};
pub use warden::{
    CommunityFragmentation, CommunityWarden, FragmentationReport, DEFAULT_FRAGMENTATION_TRIGGER,
};
pub use writer::{apply_coact_decay_to_zone, CompactionResult, Writer};
