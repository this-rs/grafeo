//! # Community detection (T10)
//!
//! Batch Leiden (Traag-Waltman 2019) + incremental LDleiden. Drives the
//! `NodeRecord.community_id u32` column on substrate via
//! `Writer::update_community_batch` (T10 Step 3, wired separately).
//!
//! The module carries its own minimal adjacency representation
//! ([`Graph`]) so the batch reference and the incremental variant can
//! share state without pulling in `obrain-adapters`. The mapping from a
//! substrate zone to [`Graph`] belongs to the caller (typically the
//! LDleiden driver in [`ldleiden`] or the bootstrap path in
//! `neo4j2obrain`/`obrain-migrate`).
//!
//! ## Model
//!
//! Undirected weighted graph with `n` nodes (0..n) and edge weights
//! `w(u, v) = w(v, u) ≥ 0`. Self-loops are permitted and count toward
//! `2m = Σ_u Σ_v w(u,v)` exactly as in the canonical modularity formula.
//!
//! ## Modularity (CPM-less, Newman-Girvan formulation)
//!
//! `Q = (1 / 2m) * Σ_{u,v} [w(u,v) - γ · k(u) · k(v) / 2m] · δ(c(u), c(v))`
//!
//! with `γ = 1.0` by default (standard Newman-Girvan). CPM (Constant
//! Potts Model) quality is reachable by tuning γ but is not the target
//! for T10 — we calibrate against Newman-Girvan Q on karate/football.
//!
//! ## Determinism
//!
//! Node iteration order is ascending by `u32` id. Tie-breaking within
//! `local_move` picks the lowest community id. No RNG is used, so the
//! algorithm is byte-for-byte reproducible.

pub mod leiden;
pub mod ldleiden;

pub use ldleiden::{LDleiden, LDleidenStats};
pub use leiden::{Graph, LeidenConfig, Partition, leiden_batch, modularity};
