//! # obrain-adapters
//!
//! The integration layer - parsers, storage backends, and plugins.
//!
//! This is where external formats meet Obrain's internal representation.
//! You probably don't need this crate directly unless you're extending Obrain.
//!
//! ## Modules
//!
//! - [`query`] - Parsers for GQL, Cypher, SPARQL, Gremlin, GraphQL
//! - [`storage`] - Persistence: write-ahead log, memory-mapped files
//! - [`plugins`] - Extension points for custom functions and algorithms

// deny (not forbid) so individual methods (e.g., mmap_snapshot) can
// #[allow(unsafe_code)] with a documented safety justification.
#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod plugins;
pub mod query;
pub mod storage;
