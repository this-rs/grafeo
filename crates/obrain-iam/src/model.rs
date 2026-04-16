//! Domain models for IAM entities stored in the graph.
//!
//! Each model corresponds to a label in the `__system` named graph:
//! - `:IamUser` — user accounts
//! - `:IamRole` — roles (collections of policies)
//! - `:IamPolicy` — authorization policies (allow/deny statements)
//! - `:IamCredential` — session credentials with TTL
//! - `:IamAuditEvent` — immutable audit trail entries

use std::fmt;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::orn::Orn;

// ---------------------------------------------------------------------------
// Labels (constants for graph node labels)
// ---------------------------------------------------------------------------

/// Label for user nodes in the `__system` graph.
pub const LABEL_USER: &str = "IamUser";
/// Label for role nodes.
pub const LABEL_ROLE: &str = "IamRole";
/// Label for policy nodes.
pub const LABEL_POLICY: &str = "IamPolicy";
/// Label for credential nodes.
pub const LABEL_CREDENTIAL: &str = "IamCredential";
/// Label for audit event nodes.
pub const LABEL_AUDIT_EVENT: &str = "IamAuditEvent";

// ---------------------------------------------------------------------------
// Edge types
// ---------------------------------------------------------------------------

/// Edge: User -[:HAS_ROLE]-> Role
pub const EDGE_HAS_ROLE: &str = "HAS_ROLE";
/// Edge: Role -[:HAS_POLICY]-> Policy
pub const EDGE_HAS_POLICY: &str = "HAS_POLICY";
/// Edge: User -[:HAS_CREDENTIAL]-> Credential
pub const EDGE_HAS_CREDENTIAL: &str = "HAS_CREDENTIAL";
/// Edge: User -[:PERFORMED]-> AuditEvent
pub const EDGE_PERFORMED: &str = "PERFORMED";

// ---------------------------------------------------------------------------
// Property keys (constants to avoid typos)
// ---------------------------------------------------------------------------

/// Common property keys used across IAM entities.
pub mod props {
    /// Unique identifier (UUID string).
    pub const ID: &str = "iam_id";
    /// Human-readable name / username.
    pub const NAME: &str = "iam_name";
    /// Email address (users).
    pub const EMAIL: &str = "iam_email";
    /// Entity status (active/inactive/suspended).
    pub const STATUS: &str = "iam_status";
    /// Description text.
    pub const DESCRIPTION: &str = "iam_description";
    /// Creation timestamp (ISO 8601).
    pub const CREATED_AT: &str = "iam_created_at";
    /// Last update timestamp.
    pub const UPDATED_AT: &str = "iam_updated_at";

    // Policy-specific
    /// Policy effect: "allow" or "deny".
    pub const EFFECT: &str = "iam_effect";
    /// Comma-separated list of actions.
    pub const ACTIONS: &str = "iam_actions";
    /// Comma-separated list of resource ORNs.
    pub const RESOURCES: &str = "iam_resources";

    // Credential-specific
    /// Credential type: "session" or "api_key".
    pub const CRED_TYPE: &str = "iam_cred_type";
    /// Token hash (bcrypt or similar — never store raw tokens).
    pub const TOKEN_HASH: &str = "iam_token_hash";
    /// Expiration timestamp (ISO 8601).
    pub const EXPIRES_AT: &str = "iam_expires_at";

    // Password-specific
    /// Whether the user must change their password on next login.
    pub const MUST_CHANGE_PASSWORD: &str = "iam_must_change_password";
    /// Bcrypt hash of the user's password.
    pub const PASSWORD_HASH: &str = "iam_password_hash";

    // Audit-specific
    /// The action that was attempted.
    pub const ACTION: &str = "iam_action";
    /// The target resource ORN.
    pub const RESOURCE: &str = "iam_resource";
    /// The result: "allow" or "deny".
    pub const RESULT: &str = "iam_result";
    /// The principal (user ORN) who performed the action.
    pub const PRINCIPAL: &str = "iam_principal";
}

// ---------------------------------------------------------------------------
// User
// ---------------------------------------------------------------------------

/// An IAM user account.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    /// Unique identifier (UUID).
    pub id: String,
    /// Username (unique within account).
    pub username: String,
    /// Email address (optional).
    pub email: Option<String>,
    /// Account status.
    pub status: EntityStatus,
    /// Whether the user must change their password on next login.
    #[serde(default)]
    pub must_change_password: bool,
    /// Creation timestamp.
    pub created_at: String,
}

// ---------------------------------------------------------------------------
// Role
// ---------------------------------------------------------------------------

/// An IAM role — a named collection of policies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Role {
    /// Unique identifier (UUID).
    pub id: String,
    /// Role name (unique within account).
    pub name: String,
    /// Human-readable description.
    pub description: Option<String>,
    /// Creation timestamp.
    pub created_at: String,
}

// ---------------------------------------------------------------------------
// Policy
// ---------------------------------------------------------------------------

/// An IAM policy — a single allow/deny statement.
///
/// Policies are evaluated in order:
/// 1. Any explicit `Deny` → access denied
/// 2. Any explicit `Allow` → access granted
/// 3. No match → implicit deny
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Policy {
    /// Unique identifier (UUID).
    pub id: String,
    /// Policy name.
    pub name: String,
    /// Allow or Deny.
    pub effect: PolicyEffect,
    /// Actions this policy applies to (e.g. `["graph:read", "graph:write"]`).
    pub actions: Vec<String>,
    /// Resource ORNs this policy applies to (supports wildcards).
    pub resources: Vec<Orn>,
    /// Creation timestamp.
    pub created_at: String,
}

/// The effect of a policy statement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PolicyEffect {
    /// Explicitly allow the action.
    Allow,
    /// Explicitly deny the action.
    Deny,
}

impl fmt::Display for PolicyEffect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Allow => write!(f, "allow"),
            Self::Deny => write!(f, "deny"),
        }
    }
}

impl PolicyEffect {
    /// Parse from a string (case-insensitive).
    pub fn from_str_loose(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "allow" => Some(Self::Allow),
            "deny" => Some(Self::Deny),
            _ => None,
        }
    }
}

/// The result of evaluating a set of policies against an action+resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PolicyDecision {
    /// Explicitly allowed by at least one policy (no deny).
    Allow,
    /// Explicitly denied by at least one policy.
    Deny,
    /// No policy matched — implicit deny.
    ImplicitDeny,
}

impl PolicyDecision {
    /// Returns `true` if access is granted.
    pub fn is_allowed(self) -> bool {
        self == Self::Allow
    }
}

// ---------------------------------------------------------------------------
// Credential
// ---------------------------------------------------------------------------

/// A session credential with TTL.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Credential {
    /// Unique identifier.
    pub id: String,
    /// Credential type.
    pub cred_type: CredentialType,
    /// Hash of the token (never store raw tokens).
    pub token_hash: String,
    /// Time-to-live.
    pub ttl: Duration,
    /// Creation timestamp.
    pub created_at: String,
    /// Expiration timestamp.
    pub expires_at: String,
}

/// Type of credential.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CredentialType {
    /// Session token (temporary).
    Session,
    /// API key (long-lived).
    ApiKey,
    /// Password credential (bcrypt hash).
    Password,
}

impl fmt::Display for CredentialType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Session => write!(f, "session"),
            Self::ApiKey => write!(f, "api_key"),
            Self::Password => write!(f, "password"),
        }
    }
}

// ---------------------------------------------------------------------------
// AuditEvent
// ---------------------------------------------------------------------------

/// An immutable audit trail entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    /// Unique identifier.
    pub id: String,
    /// The principal who performed the action (user ORN).
    pub principal: Orn,
    /// The action that was attempted.
    pub action: String,
    /// The target resource (ORN).
    pub resource: Orn,
    /// The policy decision result.
    pub result: PolicyDecision,
    /// Timestamp.
    pub timestamp: String,
}

// ---------------------------------------------------------------------------
// EntityStatus
// ---------------------------------------------------------------------------

/// Status of an IAM entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EntityStatus {
    /// Active and usable.
    Active,
    /// Temporarily suspended.
    Inactive,
    /// Suspended by admin action.
    Suspended,
}

impl fmt::Display for EntityStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Active => write!(f, "active"),
            Self::Inactive => write!(f, "inactive"),
            Self::Suspended => write!(f, "suspended"),
        }
    }
}

impl EntityStatus {
    /// Parse from a string (case-insensitive).
    pub fn from_str_loose(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "active" => Some(Self::Active),
            "inactive" => Some(Self::Inactive),
            "suspended" => Some(Self::Suspended),
            _ => None,
        }
    }
}
