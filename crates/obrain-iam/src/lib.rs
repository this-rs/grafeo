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
//! - [`model`] — Domain types (User, Role, Policy, Credential, AuditEvent)
//! - [`store`] — Graph-backed IAM store (CRUD operations on `__system` graph)
//! - [`policy`] — Policy evaluation engine (allow/deny/implicit-deny)
//! - [`error`] — IAM error types

#![deny(unsafe_code)]

pub mod error;
pub mod model;
pub mod orn;
pub mod policy;
pub mod store;

pub use error::{IamError, IamResult};
pub use model::{
    AuditEvent, Credential, CredentialType, EntityStatus, Policy, PolicyDecision, PolicyEffect,
    Role, User,
};
pub use orn::Orn;
pub use store::IamStore;
