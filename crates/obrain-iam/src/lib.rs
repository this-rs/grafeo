//! # obrain-iam
//!
//! Identity and Access Management for Obrain.
//!
//! Provides the **ORN** (Obrain Resource Name) system for fine-grained resource
//! identification within the graph, policy evaluation, and tenant-scoped sessions.
//!
//! ## ORN Format
//!
//! ```text
//! orn:obrain:{service}:{account_id}:{resource_type}/{resource_path}
//! ```
//!
//! Examples:
//! - `orn:obrain:graph:alice:node/12345`
//! - `orn:obrain:graph:alice:tenant/chess-kb`
//! - `orn:obrain:iam:*:user/*` (wildcard matching)
//!
//! ## Modules
//!
//! - [`orn`] — ORN parsing, formatting, pattern matching, wildcard, variable substitution
//! - [`error`] — IAM error types

#![deny(unsafe_code)]

pub mod error;
pub mod orn;

pub use error::IamError;
pub use orn::Orn;
