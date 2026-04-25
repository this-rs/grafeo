//! # obrain-core
//!
//! The core data structures behind Obrain. You'll find graph storage, indexes,
//! and the execution engine here.
//!
//! Most users don't need this crate directly - use `obrain` or `obrain-engine`
//! instead. But if you're building algorithms or need low-level access, this
//! is where the action is.
//!
//! ## Modules
//!
//! - [`cache`] - Caching utilities: second-chance LRU
//! - [`graph`] - Graph storage: LPG (labeled property graph) and RDF triple stores
//! - [`index`] - Fast lookups: hash, B-tree, adjacency lists, tries
//! - [`execution`] - Query execution: data chunks, vectors, operators
//! - [`statistics`] - Cardinality estimates for the query optimizer
//! - [`storage`] - Compression: dictionary encoding, bit-packing, delta encoding

#![deny(unsafe_code)]

pub mod cache;
pub mod change_tracker;
pub mod execution;
pub mod graph;
pub mod index;
pub mod statistics;
pub mod storage;
pub mod subscription;
pub mod testing;

// Re-export the types you'll use most often
pub use change_tracker::{ChangeTracker, EntityRef, GraphDiff, GraphEvent};
// T17 Step 15 slice: drop `LpgStore` from the root re-export. Consumers
// still needing the legacy in-memory LpgStore must import it via the
// fully qualified `obrain_core::graph::lpg::LpgStore` path (no caller
// uses the root re-export — audited 2026-04-23, 0 hits for
// `use obrain_core::LpgStore`). `Edge` and `Node` stay re-exported as
// structural data types are stable across backends.
pub use graph::lpg::{Edge, Node};
pub use index::adjacency::ChunkedAdjacency;
pub use statistics::{ColumnStatistics, Histogram, LabelStatistics, Statistics};
pub use storage::{DictionaryBuilder, DictionaryEncoding};
pub use subscription::{EventFilter, EventType, SubscriptionId, SubscriptionManager};
