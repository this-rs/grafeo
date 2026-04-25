//! Graph entity types shared by the query engine and all graph backends.
//!
//! T17 final cutover (2026-04-23): the in-memory `LpgStore` backend
//! (~13 000 LOC under `store/`) was retired — substrate is the single
//! production backend. Only value types (`Node`, `Edge`, records,
//! `PropertyStorage`, `CompareOp`) remain.

mod edge;
mod node;
mod property;

pub use edge::{Edge, EdgeFlags, EdgeRecord};
pub use node::{Node, NodeFlags, NodeRecord};
pub use property::{CompareOp, PropertyStorage};
