//! Node.js/TypeScript bindings for Obrain graph database.
//!
//! Use Obrain from JavaScript with native Rust performance.
//!
//! ```js
//! const { ObrainDB } = require('@obrain-db/js');
//!
//! const db = ObrainDB.create();
//! await db.execute("INSERT (:Person {name: 'Alix'})");
//! const result = await db.execute("MATCH (p:Person) RETURN p.name");
//! console.log(result.toArray());
//! ```

// napi FFI requires unsafe casts between JS value types
#![allow(unsafe_code)]

#[macro_use]
extern crate napi_derive;

mod database;
mod error;
mod graph;
mod query;
mod transaction;
mod types;

/// Returns the active SIMD instruction set for vector operations.
#[napi]
pub fn simd_support() -> String {
    obrain_core::index::vector::simd_support().to_string()
}

/// Returns the Obrain version.
#[napi]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
