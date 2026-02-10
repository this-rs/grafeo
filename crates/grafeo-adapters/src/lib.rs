//! # grafeo-adapters
//!
//! The integration layer - parsers, storage backends, and plugins.
//!
//! This is where external formats meet Grafeo's internal representation.
//! You probably don't need this crate directly unless you're extending Grafeo.
//!
//! ## Modules
//!
//! - [`query`] - Parsers for GQL, Cypher, SPARQL, Gremlin, GraphQL
//! - [`storage`] - Persistence: write-ahead log, memory-mapped files
//! - [`plugins`] - Extension points for custom functions and algorithms

// TODO(0.5.x): Document all public parser types and remove this allow
#![allow(missing_docs)]

pub mod plugins;
pub mod query;
pub mod storage;
