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

pub mod checkpoint;
pub mod dict;
pub mod error;
pub mod file;
pub mod heap;
pub mod meta;
pub mod page;
pub mod record;
pub mod replay;
pub mod store;
pub mod tiers;
pub mod wal;
pub mod wal_io;
pub mod writer;

pub use error::{SubstrateError, SubstrateResult};
pub use file::{SubstrateFile, Zone, ZoneFile, zone};
pub use heap::{HEAP_PAGE_MAGIC, HEAP_PAGE_SIZE, HeapPage, HeapPageHeader, StringHeap};
pub use meta::{
    META_FILE_SIZE, META_HEADER_SIZE, MetaHeader, SUBSTRATE_FORMAT_VERSION, SUBSTRATE_MAGIC,
    meta_flags,
};
pub use page::{HeapRef, PAGE_SIZE, PROP_PAGE_MAGIC, PropertyPage, PropertyPageHeader, ValueTag};
pub use record::{
    EdgeRecord, NodeRecord, PackedScarUtilAff, SCAR_MAX_INTENSITY_Q5, U48, UTILITY_MAX_SCORE_Q5,
    affinity_to_q5, edge_flags, f32_to_q1_15, node_flags, q1_15_to_f32, q5_to_affinity, q5_to_scar,
    q5_to_utility, scar_to_q5, utility_to_q5,
};
pub use tiers::{
    L0_BITS, L1_BITS, L2_DIM, Tier0, Tier1, Tier2, f16_to_f32, f32_to_f16, tier0_hamming,
    tier0_topk, tier1_hamming, tier1_topk, tier2_cosine,
};
pub use wal::{PropValue, WAL_HEADER_SIZE, WalKind, WalPayload, WalRecord};
pub use wal_io::{SyncMode, WalIter, WalReader, WalWriter};
pub use checkpoint::{checkpoint, CheckpointStats};
pub use dict::DictSnapshot;
pub use replay::{replay_from, ReplayStats};
pub use store::SubstrateStore;
pub use writer::Writer;
