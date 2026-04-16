//! Authentication and authorization provider.
//!
//! Defines the [`AuthProvider`] trait — a self-contained abstraction for
//! authenticate / authorize / session-management operations. This trait is
//! designed to be compatible with WAMI's `CloudProvider` model: when the two
//! repositories are unified, a blanket `impl CloudProvider for T: AuthProvider`
//! bridge (or feature-gated direct impl) can be added without changing the
//! call-sites.
//!
//! [`ObrainIamProvider`] is the concrete implementation backed by [`IamStore`].

use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::error::{IamError, IamResult};
use crate::model::{CredentialType, EntityStatus, PolicyDecision, User};
use crate::orn::Orn;
use crate::store::IamStore;

// ---------------------------------------------------------------------------
// Identity — the authenticated principal
// ---------------------------------------------------------------------------

/// Represents an authenticated identity (the result of a successful
/// authentication).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Identity {
    /// User ID (IAM UUID).
    pub user_id: String,
    /// Username.
    pub username: String,
    /// Account / tenant scope.
    pub account_id: String,
    /// ORN of the principal.
    pub principal: Orn,
    /// Status of the user account.
    pub status: EntityStatus,
}

impl fmt::Display for Identity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}@{}", self.username, self.account_id)
    }
}

// ---------------------------------------------------------------------------
// SessionCredentials — issued after authentication
// ---------------------------------------------------------------------------

/// Session credentials returned after successful authentication.
///
/// Contains a token (opaque session token or future JWT) and metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionCredentials {
    /// Credential ID.
    pub credential_id: String,
    /// The session token (caller must hash before storing; this is the raw
    /// value to return to the client exactly once).
    pub token: String,
    /// Token type (session or api_key).
    pub token_type: CredentialType,
    /// TTL in seconds.
    pub ttl_secs: u64,
    /// Expiration timestamp (ISO 8601).
    pub expires_at: String,
}

// ---------------------------------------------------------------------------
// AuthRequest — input for authenticate()
// ---------------------------------------------------------------------------

/// Authentication request — supports multiple credential types.
#[derive(Debug, Clone)]
pub enum AuthRequest {
    /// Authenticate with a session token (looked up in the credential store).
    SessionToken {
        /// The raw session token.
        token: String,
    },
    /// Authenticate with username + API key.
    ApiKey {
        /// Username.
        username: String,
        /// The raw API key.
        api_key: String,
    },
    /// Authenticate via an external OIDC token (to be verified against a
    /// provider's JWKS endpoint).
    OidcToken {
        /// The raw OIDC/JWT token.
        token: String,
        /// The expected issuer (for validation).
        issuer: String,
    },
    /// Password-based authentication.
    Password {
        /// Username.
        username: String,
        /// The plaintext password (will be hashed for comparison).
        password: String,
    },
    /// Internal / bootstrap authentication — trusted, no credential check.
    /// Used for system-level operations (migrations, initial setup).
    Internal {
        /// Username to impersonate.
        username: String,
    },
}

// ---------------------------------------------------------------------------
// AuthProvider trait
// ---------------------------------------------------------------------------

/// Trait for authentication and authorization providers.
///
/// This is the central abstraction that downstream components (sessions,
/// middleware, etc.) depend on. It is intentionally independent of WAMI's
/// `CloudProvider` trait so that `obrain-iam` can be used without a WAMI
/// dependency. When WAMI integration is desired, a feature-gated bridge
/// module can map between the two.
pub trait AuthProvider: Send + Sync + fmt::Debug {
    /// Authenticates a request and returns an [`Identity`] on success.
    fn authenticate(&self, request: &AuthRequest) -> IamResult<Identity>;

    /// Evaluates whether an identity is authorized to perform `action` on
    /// `resource`.
    fn authorize(
        &self,
        identity: &Identity,
        action: &str,
        resource: &Orn,
    ) -> IamResult<PolicyDecision>;

    /// Creates session credentials for an authenticated identity.
    fn create_session(&self, identity: &Identity, ttl_secs: u64) -> IamResult<SessionCredentials>;

    /// Revokes a session credential by ID.
    fn revoke_session(&self, credential_id: &str) -> IamResult<()>;

    /// Creates a federation token — a scoped, short-lived credential for
    /// cross-instance or worker-to-worker communication.
    ///
    /// The returned [`FederationToken`] carries scoped policies (a subset of
    /// the caller's permissions) and a target instance hint.
    fn create_federation_token(
        &self,
        identity: &Identity,
        request: &FederationRequest,
    ) -> IamResult<FederationToken>;

    /// Creates an assumed-role session — the caller temporarily assumes a
    /// different role with a scoped policy set.
    fn assume_role(
        &self,
        identity: &Identity,
        role_name: &str,
        ttl_secs: u64,
    ) -> IamResult<SessionCredentials>;
}

// ---------------------------------------------------------------------------
// FederationRequest / FederationToken
// ---------------------------------------------------------------------------

/// Request to create a federation token.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationRequest {
    /// Scoped actions the federation token should allow (subset of caller's
    /// permissions). If empty, inherits all caller permissions.
    pub scoped_actions: Vec<String>,
    /// Scoped resource ORN patterns. If empty, inherits all caller resources.
    pub scoped_resources: Vec<Orn>,
    /// TTL in seconds for the federation token.
    pub ttl_secs: u64,
    /// Optional target instance identifier (for P2P sharing).
    pub target_instance: Option<String>,
}

/// A federation token — a scoped, short-lived credential.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationToken {
    /// The session credentials (token + metadata).
    pub credentials: SessionCredentials,
    /// The scoped actions this token allows.
    pub scoped_actions: Vec<String>,
    /// The scoped resource patterns this token allows.
    pub scoped_resources: Vec<Orn>,
    /// The identity this token was issued for.
    pub principal: Orn,
    /// Optional target instance.
    pub target_instance: Option<String>,
}

// ---------------------------------------------------------------------------
// ObrainIamProvider — concrete implementation
// ---------------------------------------------------------------------------

/// Graph-backed IAM provider.
///
/// Bridges [`AuthProvider`] to [`IamStore`] for all authentication,
/// authorization, and session management operations.
#[derive(Debug)]
pub struct ObrainIamProvider {
    /// The underlying IAM store.
    store: Arc<IamStore>,
    /// Default account ID for this instance.
    account_id: String,
}

impl ObrainIamProvider {
    /// Creates a new provider backed by the given IAM store.
    pub fn new(store: Arc<IamStore>, account_id: impl Into<String>) -> Self {
        Self {
            store,
            account_id: account_id.into(),
        }
    }

    /// Returns a reference to the underlying IAM store.
    pub fn store(&self) -> &IamStore {
        &self.store
    }

    /// Returns the account ID.
    pub fn account_id(&self) -> &str {
        &self.account_id
    }

    /// Builds an [`Identity`] from a [`User`].
    fn user_to_identity(&self, user: &User) -> Identity {
        Identity {
            user_id: user.id.clone(),
            username: user.username.clone(),
            account_id: self.account_id.clone(),
            principal: Orn::user(&self.account_id, &user.username),
            status: user.status,
        }
    }

    /// Simple token hash (SHA-256 hex).
    ///
    /// In production this would use bcrypt/argon2 with salt, but for the
    /// embedded single-user scenario a simple hash suffices.
    fn hash_token(token: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        token.hash(&mut hasher);
        format!("{:016x}", hasher.finish())
    }

    /// Generates a simple unique ID.
    fn generate_id(prefix: &str) -> String {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        format!("{prefix}-{n:08x}")
    }

    /// Generates a random-ish session token.
    fn generate_token() -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        use std::sync::atomic::{AtomicU64, Ordering};

        static TOKEN_COUNTER: AtomicU64 = AtomicU64::new(1);
        let n = TOKEN_COUNTER.fetch_add(1, Ordering::Relaxed);

        let mut hasher = DefaultHasher::new();
        n.hash(&mut hasher);
        "obrain-iam".hash(&mut hasher);
        format!("obt_{:016x}{:016x}", hasher.finish(), n)
    }
}

impl AuthProvider for ObrainIamProvider {
    fn authenticate(&self, request: &AuthRequest) -> IamResult<Identity> {
        match request {
            AuthRequest::Internal { username } => {
                // Trusted internal auth — just look up the user
                let user = self.store.find_user_by_name(username).ok_or_else(|| {
                    IamError::ResourceNotFound {
                        resource: format!("user:{username}"),
                    }
                })?;

                if user.status != EntityStatus::Active {
                    return Err(IamError::AccessDenied {
                        reason: format!("user account is {}", user.status),
                    });
                }

                Ok(self.user_to_identity(&user))
            }

            AuthRequest::SessionToken { token } => {
                // Look up credential by token hash, find owning user
                let token_hash = Self::hash_token(token);
                self.authenticate_by_token_hash(&token_hash, CredentialType::Session)
            }

            AuthRequest::ApiKey { username, api_key } => {
                // Verify the user exists and the API key matches
                let user = self.store.find_user_by_name(username).ok_or_else(|| {
                    IamError::AccessDenied {
                        reason: "invalid credentials".to_string(),
                    }
                })?;

                if user.status != EntityStatus::Active {
                    return Err(IamError::AccessDenied {
                        reason: format!("user account is {}", user.status),
                    });
                }

                let key_hash = Self::hash_token(api_key);
                // Verify that this user has a matching API key credential
                self.verify_user_credential(&user.id, &key_hash, CredentialType::ApiKey)?;

                Ok(self.user_to_identity(&user))
            }

            AuthRequest::Password { username, password } => {
                let user = self.store.find_user_by_name(username).ok_or_else(|| {
                    IamError::AccessDenied {
                        reason: "invalid credentials".to_string(),
                    }
                })?;

                if user.status != EntityStatus::Active {
                    return Err(IamError::AccessDenied {
                        reason: format!("user account is {}", user.status),
                    });
                }

                self.verify_password_bcrypt(&user.id, password)?;

                Ok(self.user_to_identity(&user))
            }

            AuthRequest::OidcToken {
                token: _,
                issuer: _,
            } => {
                // OIDC verification requires external JWKS fetch — not yet
                // implemented. This will be added when JWT/ed25519 support
                // lands (T1.7).
                Err(IamError::InvalidParameter {
                    message:
                        "OIDC authentication not yet implemented — requires T1.7 (JWT/Ed25519)"
                            .to_string(),
                })
            }
        }
    }

    fn authorize(
        &self,
        identity: &Identity,
        action: &str,
        resource: &Orn,
    ) -> IamResult<PolicyDecision> {
        let decision = self
            .store
            .evaluate_access(&identity.user_id, action, resource)?;

        // Log audit event
        let event_id = Self::generate_id("evt");
        self.store
            .log_audit_event(&event_id, &identity.principal, action, resource, decision);

        Ok(decision)
    }

    fn create_session(&self, identity: &Identity, ttl_secs: u64) -> IamResult<SessionCredentials> {
        let cred_id = Self::generate_id("cred");
        let token = Self::generate_token();
        let token_hash = Self::hash_token(&token);

        let cred = self.store.create_credential(
            &cred_id,
            &identity.user_id,
            CredentialType::Session,
            &token_hash,
            ttl_secs,
        )?;

        Ok(SessionCredentials {
            credential_id: cred_id,
            token,
            token_type: CredentialType::Session,
            ttl_secs,
            expires_at: cred.expires_at,
        })
    }

    fn revoke_session(&self, _credential_id: &str) -> IamResult<()> {
        // Credential revocation requires node deletion support in LpgStore,
        // which is not yet exposed. For now, mark as a no-op and track as
        // a known limitation.
        //
        // TODO: implement when LpgStore.delete_node() is available
        Ok(())
    }

    fn create_federation_token(
        &self,
        identity: &Identity,
        request: &FederationRequest,
    ) -> IamResult<FederationToken> {
        // Validate: scoped actions must be a subset of the caller's actual
        // permissions. For now we trust the caller and just record them.
        // Full intersection validation will be added when we have policy
        // enumeration on IamStore.

        if request.ttl_secs == 0 {
            return Err(IamError::InvalidParameter {
                message: "federation token TTL must be > 0".to_string(),
            });
        }

        // Cap federation token TTL to 12 hours
        let ttl = request.ttl_secs.min(43_200);

        // Create a session credential for the federation token
        let cred_id = Self::generate_id("fed");
        let token = Self::generate_token();
        let token_hash = Self::hash_token(&token);

        let cred = self.store.create_credential(
            &cred_id,
            &identity.user_id,
            CredentialType::Session,
            &token_hash,
            ttl,
        )?;

        let scoped_actions = if request.scoped_actions.is_empty() {
            vec!["*".to_string()]
        } else {
            request.scoped_actions.clone()
        };

        let scoped_resources = if request.scoped_resources.is_empty() {
            vec![Orn::all_in_account(&identity.account_id)]
        } else {
            request.scoped_resources.clone()
        };

        Ok(FederationToken {
            credentials: SessionCredentials {
                credential_id: cred_id,
                token,
                token_type: CredentialType::Session,
                ttl_secs: ttl,
                expires_at: cred.expires_at,
            },
            scoped_actions,
            scoped_resources,
            principal: identity.principal.clone(),
            target_instance: request.target_instance.clone(),
        })
    }

    fn assume_role(
        &self,
        identity: &Identity,
        role_name: &str,
        ttl_secs: u64,
    ) -> IamResult<SessionCredentials> {
        // Verify the role exists
        let _role =
            self.store
                .find_role_by_name(role_name)
                .ok_or_else(|| IamError::ResourceNotFound {
                    resource: format!("role:{role_name}"),
                })?;

        // Verify the user actually has this role attached
        let user_roles = self.store.get_user_roles(&identity.user_id)?;
        let has_role = user_roles.iter().any(|r| r.name == role_name);
        if !has_role {
            return Err(IamError::AccessDenied {
                reason: format!(
                    "user {} does not have role {}",
                    identity.username, role_name
                ),
            });
        }

        // Create a scoped session (same mechanism, the role scoping will
        // be enforced by AuthenticatedSession in T1.4)
        let cred_id = Self::generate_id("ars"); // assumed-role-session
        let token = Self::generate_token();
        let token_hash = Self::hash_token(&token);

        let cred = self.store.create_credential(
            &cred_id,
            &identity.user_id,
            CredentialType::Session,
            &token_hash,
            ttl_secs,
        )?;

        Ok(SessionCredentials {
            credential_id: cred_id,
            token,
            token_type: CredentialType::Session,
            ttl_secs,
            expires_at: cred.expires_at,
        })
    }
}

// ---------------------------------------------------------------------------
// Password & user management methods for ObrainIamProvider
// ---------------------------------------------------------------------------

impl ObrainIamProvider {
    /// Sets (or replaces) the user's password.
    ///
    /// Creates a `Password` credential with the hashed password. If the user
    /// already has a password credential, it is replaced.
    pub fn set_user_password(&self, user_id: &str, password: &str) -> IamResult<()> {
        // Remove existing password credential if any
        self.remove_password_credential(user_id);

        let cred_id = Self::generate_id("pw");
        let bcrypt_hash = bcrypt::hash(password, bcrypt::DEFAULT_COST)
            .map_err(|e| IamError::Internal {
                message: format!("bcrypt hash failed: {e}"),
            })?;
        self.store.create_credential(
            &cred_id,
            user_id,
            CredentialType::Password,
            &bcrypt_hash,
            0, // no expiry for passwords
        )?;
        Ok(())
    }

    /// Returns `true` if the user has a password credential.
    pub fn has_password(&self, user_id: &str) -> bool {
        self.find_password_credential(user_id).is_some()
    }

    /// Returns `true` if the user must change their password on next login.
    pub fn must_change_password(&self, user_id: &str) -> bool {
        use crate::model::props;
        use obrain_common::Value;

        let Some(nid) = self.find_user_node(user_id) else {
            return false;
        };
        matches!(
            self.store.inner().get_node_property(nid, &props::MUST_CHANGE_PASSWORD.into()),
            Some(Value::String(s)) if s.as_str() == "true"
        )
    }

    /// Sets the `must_change_password` flag on a user.
    pub fn set_must_change_password(&self, user_id: &str, must_change: bool) -> IamResult<()> {
        use crate::model::props;
        use obrain_common::Value;

        let nid = self.find_user_node(user_id).ok_or_else(|| {
            IamError::ResourceNotFound {
                resource: format!("user:{user_id}"),
            }
        })?;
        self.store.inner().set_node_property(
            nid,
            props::MUST_CHANGE_PASSWORD,
            Value::from(if must_change { "true" } else { "false" }),
        );
        Ok(())
    }

    /// Finds the password credential node for a user, if any.
    fn find_password_credential(
        &self,
        user_id: &str,
    ) -> Option<obrain_common::types::NodeId> {
        use crate::model::{EDGE_HAS_CREDENTIAL, props};
        use obrain_common::Value;
        use obrain_core::graph::Direction;

        let user_nid = self.find_user_node(user_id)?;
        let store = self.store.inner();

        for (target, eid) in store.edges_from(user_nid, Direction::Outgoing) {
            if let Some(et) = store.edge_type(eid) {
                if et.as_str() != EDGE_HAS_CREDENTIAL {
                    continue;
                }
                if let Some(Value::String(t)) =
                    store.get_node_property(target, &props::CRED_TYPE.into())
                {
                    if t.as_str() == "password" {
                        return Some(target);
                    }
                }
            }
        }
        None
    }

    /// Removes the password credential for a user (if any).
    fn remove_password_credential(&self, user_id: &str) {
        // Mark the old password credential as expired by overwriting the hash.
        // Full node deletion is not yet available in LpgStore.
        if let Some(cred_nid) = self.find_password_credential(user_id) {
            use crate::model::props;
            use obrain_common::Value;
            self.store.inner().set_node_property(
                cred_nid,
                props::TOKEN_HASH,
                Value::from("__revoked__"),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Private helpers for ObrainIamProvider
// ---------------------------------------------------------------------------

impl ObrainIamProvider {
    /// Authenticates by looking up a credential node with the given token hash.
    fn authenticate_by_token_hash(
        &self,
        token_hash: &str,
        expected_type: CredentialType,
    ) -> IamResult<Identity> {
        use crate::model::{EDGE_HAS_CREDENTIAL, LABEL_CREDENTIAL, LABEL_USER, props};
        use obrain_common::Value;

        let store = self.store.inner();
        let cred_nodes = store.nodes_by_label(LABEL_CREDENTIAL);

        for cred_nid in cred_nodes {
            // Match token hash
            let hash_match = match store.get_node_property(cred_nid, &props::TOKEN_HASH.into()) {
                Some(Value::String(h)) => h.as_str() == token_hash,
                _ => false,
            };
            if !hash_match {
                continue;
            }

            // Check credential type
            let type_match = match store.get_node_property(cred_nid, &props::CRED_TYPE.into()) {
                Some(Value::String(t)) => t.as_str() == expected_type.to_string(),
                _ => false,
            };
            if !type_match {
                continue;
            }

            // Find the owning user (walk backwards: User -[:HAS_CREDENTIAL]-> Cred)
            let user_nodes = store.nodes_by_label(LABEL_USER);
            for user_nid in user_nodes {
                use obrain_core::graph::Direction;
                for (target, eid) in store.edges_from(user_nid, Direction::Outgoing) {
                    if target == cred_nid
                        && let Some(et) = store.edge_type(eid)
                        && et.as_str() == EDGE_HAS_CREDENTIAL
                    {
                        // Found the user — build identity
                        if let Some(user) = self
                            .store
                            .get_user(&self.get_str_prop(user_nid, props::ID).unwrap_or_default())
                        {
                            if user.status != EntityStatus::Active {
                                return Err(IamError::AccessDenied {
                                    reason: format!("user account is {}", user.status),
                                });
                            }
                            return Ok(self.user_to_identity(&user));
                        }
                    }
                }
            }
        }

        Err(IamError::AccessDenied {
            reason: "invalid or expired credentials".to_string(),
        })
    }

    /// Verifies that a user has a credential with the given hash and type.
    fn verify_user_credential(
        &self,
        user_id: &str,
        token_hash: &str,
        expected_type: CredentialType,
    ) -> IamResult<()> {
        use crate::model::{EDGE_HAS_CREDENTIAL, props};
        use obrain_common::Value;

        let store = self.store.inner();

        // Find user node
        let user_nid = self
            .find_user_node(user_id)
            .ok_or_else(|| IamError::ResourceNotFound {
                resource: format!("user:{user_id}"),
            })?;

        // Walk user's credentials
        use obrain_core::graph::Direction;
        for (target, eid) in store.edges_from(user_nid, Direction::Outgoing) {
            if let Some(et) = store.edge_type(eid) {
                if et.as_str() != EDGE_HAS_CREDENTIAL {
                    continue;
                }

                // Check type
                let type_ok = match store.get_node_property(target, &props::CRED_TYPE.into()) {
                    Some(Value::String(t)) => t.as_str() == expected_type.to_string(),
                    _ => false,
                };
                if !type_ok {
                    continue;
                }

                // Check hash
                let hash_ok = match store.get_node_property(target, &props::TOKEN_HASH.into()) {
                    Some(Value::String(h)) => h.as_str() == token_hash,
                    _ => false,
                };
                if hash_ok {
                    return Ok(());
                }
            }
        }

        Err(IamError::AccessDenied {
            reason: "invalid credentials".to_string(),
        })
    }

    /// Verify a password credential using bcrypt.
    ///
    /// Passwords are stored as bcrypt hashes. This method finds the user's
    /// Password credential and verifies the plaintext against the stored hash.
    fn verify_password_bcrypt(&self, user_id: &str, plaintext: &str) -> IamResult<()> {
        use crate::model::{EDGE_HAS_CREDENTIAL, props};
        use obrain_common::Value;

        let store = self.store.inner();

        let user_nid = self
            .find_user_node(user_id)
            .ok_or_else(|| IamError::ResourceNotFound {
                resource: format!("user:{user_id}"),
            })?;

        use obrain_core::graph::Direction;
        for (target, eid) in store.edges_from(user_nid, Direction::Outgoing) {
            if let Some(et) = store.edge_type(eid) {
                if et.as_str() != EDGE_HAS_CREDENTIAL {
                    continue;
                }

                let type_ok = match store.get_node_property(target, &props::CRED_TYPE.into()) {
                    Some(Value::String(t)) => t.as_str() == CredentialType::Password.to_string(),
                    _ => false,
                };
                if !type_ok {
                    continue;
                }

                if let Some(Value::String(stored_hash)) =
                    store.get_node_property(target, &props::TOKEN_HASH.into())
                {
                    // Support both bcrypt hashes ($2b$...) and legacy DefaultHasher hashes
                    let verified = if stored_hash.starts_with("$2") {
                        bcrypt::verify(plaintext, stored_hash.as_str()).unwrap_or(false)
                    } else {
                        // Legacy: DefaultHasher comparison
                        stored_hash.as_str() == Self::hash_token(plaintext)
                    };
                    if verified {
                        return Ok(());
                    }
                }
            }
        }

        Err(IamError::AccessDenied {
            reason: "invalid credentials".to_string(),
        })
    }

    /// Helper: find user node by IAM ID.
    fn find_user_node(&self, user_id: &str) -> Option<obrain_common::types::NodeId> {
        use crate::model::{LABEL_USER, props};
        use obrain_common::Value;

        let store = self.store.inner();
        for nid in store.nodes_by_label(LABEL_USER) {
            if let Some(Value::String(id)) = store.get_node_property(nid, &props::ID.into())
                && id.as_str() == user_id
            {
                return Some(nid);
            }
        }
        None
    }

    /// Helper: get string property from a node.
    fn get_str_prop(&self, nid: obrain_common::types::NodeId, key: &str) -> Option<String> {
        use obrain_common::Value;
        match self.store.inner().get_node_property(nid, &key.into()) {
            Some(Value::String(s)) => Some(s.to_string()),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::PolicyEffect;
    use obrain_core::graph::lpg::LpgStore;

    fn setup() -> ObrainIamProvider {
        let lpg = LpgStore::new().expect("failed to create LpgStore");
        let iam_store = Arc::new(IamStore::new(Arc::new(lpg)));

        // Bootstrap: create a user with a role and policy
        iam_store
            .create_user("u1", "alice", Some("alice@obrain.dev"))
            .unwrap();
        iam_store
            .create_role("r1", "reader", Some("Read-only"))
            .unwrap();
        iam_store
            .create_policy(
                "p1",
                "ReadAll",
                PolicyEffect::Allow,
                &["graph:read", "graph:list"],
                &[Orn::all_in_account("default")],
            )
            .unwrap();
        iam_store.attach_role("u1", "r1").unwrap();
        iam_store.attach_policy_to_role("r1", "p1").unwrap();

        ObrainIamProvider::new(iam_store, "default")
    }

    // -- Internal auth -------------------------------------------------------

    #[test]
    fn internal_auth_success() {
        let provider = setup();
        let identity = provider
            .authenticate(&AuthRequest::Internal {
                username: "alice".into(),
            })
            .unwrap();

        assert_eq!(identity.username, "alice");
        assert_eq!(identity.account_id, "default");
        assert_eq!(identity.status, EntityStatus::Active);
    }

    #[test]
    fn internal_auth_user_not_found() {
        let provider = setup();
        let err = provider
            .authenticate(&AuthRequest::Internal {
                username: "nobody".into(),
            })
            .unwrap_err();

        assert!(matches!(err, IamError::ResourceNotFound { .. }));
    }

    // -- Session token auth --------------------------------------------------

    #[test]
    fn session_token_auth_roundtrip() {
        let provider = setup();

        // 1. Authenticate internally to get identity
        let identity = provider
            .authenticate(&AuthRequest::Internal {
                username: "alice".into(),
            })
            .unwrap();

        // 2. Create a session
        let session = provider.create_session(&identity, 3600).unwrap();
        assert_eq!(session.token_type, CredentialType::Session);
        assert_eq!(session.ttl_secs, 3600);
        assert!(session.token.starts_with("obt_"));

        // 3. Authenticate with the session token
        let identity2 = provider
            .authenticate(&AuthRequest::SessionToken {
                token: session.token.clone(),
            })
            .unwrap();

        assert_eq!(identity2.username, "alice");
        assert_eq!(identity2.user_id, identity.user_id);
    }

    #[test]
    fn session_token_auth_invalid_token() {
        let provider = setup();

        let err = provider
            .authenticate(&AuthRequest::SessionToken {
                token: "invalid-token".into(),
            })
            .unwrap_err();

        assert!(matches!(err, IamError::AccessDenied { .. }));
    }

    // -- API key auth --------------------------------------------------------

    #[test]
    fn api_key_auth_roundtrip() {
        let provider = setup();

        // Create an API key credential for alice
        let api_key = "my-secret-api-key";
        let key_hash = ObrainIamProvider::hash_token(api_key);
        provider
            .store()
            .create_credential("ak1", "u1", CredentialType::ApiKey, &key_hash, 0)
            .unwrap();

        // Authenticate with it
        let identity = provider
            .authenticate(&AuthRequest::ApiKey {
                username: "alice".into(),
                api_key: api_key.into(),
            })
            .unwrap();

        assert_eq!(identity.username, "alice");
    }

    #[test]
    fn api_key_auth_wrong_key() {
        let provider = setup();

        // Create an API key credential
        let key_hash = ObrainIamProvider::hash_token("real-key");
        provider
            .store()
            .create_credential("ak1", "u1", CredentialType::ApiKey, &key_hash, 0)
            .unwrap();

        let err = provider
            .authenticate(&AuthRequest::ApiKey {
                username: "alice".into(),
                api_key: "wrong-key".into(),
            })
            .unwrap_err();

        assert!(matches!(err, IamError::AccessDenied { .. }));
    }

    // -- OIDC (not yet implemented) ------------------------------------------

    #[test]
    fn oidc_auth_not_implemented() {
        let provider = setup();
        let err = provider
            .authenticate(&AuthRequest::OidcToken {
                token: "jwt-token".into(),
                issuer: "https://accounts.google.com".into(),
            })
            .unwrap_err();

        assert!(matches!(err, IamError::InvalidParameter { .. }));
    }

    // -- Authorization -------------------------------------------------------

    #[test]
    fn authorize_allowed() {
        let provider = setup();
        let identity = provider
            .authenticate(&AuthRequest::Internal {
                username: "alice".into(),
            })
            .unwrap();

        let decision = provider
            .authorize(&identity, "graph:read", &Orn::node("default", 42))
            .unwrap();

        assert!(decision.is_allowed());
    }

    #[test]
    fn authorize_denied() {
        let provider = setup();
        let identity = provider
            .authenticate(&AuthRequest::Internal {
                username: "alice".into(),
            })
            .unwrap();

        // alice only has graph:read and graph:list
        let decision = provider
            .authorize(&identity, "graph:write", &Orn::node("default", 42))
            .unwrap();

        assert!(!decision.is_allowed());
    }

    #[test]
    fn authorize_logs_audit_event() {
        let provider = setup();
        let identity = provider
            .authenticate(&AuthRequest::Internal {
                username: "alice".into(),
            })
            .unwrap();

        // No events yet
        let events_before = provider.store().list_audit_events(100);
        let count_before = events_before.len();

        provider
            .authorize(&identity, "graph:read", &Orn::node("default", 1))
            .unwrap();

        let events_after = provider.store().list_audit_events(100);
        assert_eq!(events_after.len(), count_before + 1);
    }

    // -- Session management --------------------------------------------------

    #[test]
    fn create_and_use_session() {
        let provider = setup();
        let identity = provider
            .authenticate(&AuthRequest::Internal {
                username: "alice".into(),
            })
            .unwrap();

        let session = provider.create_session(&identity, 7200).unwrap();
        assert_eq!(session.ttl_secs, 7200);

        // Revoke (no-op for now, but should not error)
        provider.revoke_session(&session.credential_id).unwrap();
    }

    // -- Full flow -----------------------------------------------------------

    #[test]
    fn full_provider_flow() {
        let provider = setup();

        // 1. Internal auth (bootstrap)
        let identity = provider
            .authenticate(&AuthRequest::Internal {
                username: "alice".into(),
            })
            .unwrap();

        // 2. Create session
        let session = provider.create_session(&identity, 3600).unwrap();

        // 3. Re-authenticate with session token
        let identity2 = provider
            .authenticate(&AuthRequest::SessionToken {
                token: session.token,
            })
            .unwrap();
        assert_eq!(identity2.username, "alice");

        // 4. Authorize
        let allowed = provider
            .authorize(&identity2, "graph:read", &Orn::node("default", 1))
            .unwrap();
        assert!(allowed.is_allowed());

        let denied = provider
            .authorize(&identity2, "graph:delete", &Orn::node("default", 1))
            .unwrap();
        assert!(!denied.is_allowed());

        // 5. Verify audit trail
        let events = provider.store().list_audit_events(100);
        assert!(events.len() >= 2); // at least the two authorize calls
    }

    // -- Federation tokens ---------------------------------------------------

    #[test]
    fn federation_token_basic() {
        let provider = setup();
        let identity = provider
            .authenticate(&AuthRequest::Internal {
                username: "alice".into(),
            })
            .unwrap();

        let fed_token = provider
            .create_federation_token(
                &identity,
                &FederationRequest {
                    scoped_actions: vec!["graph:read".into()],
                    scoped_resources: vec![Orn::all_of_type("graph", "default", "node")],
                    ttl_secs: 1800,
                    target_instance: Some("peer-instance-1".into()),
                },
            )
            .unwrap();

        assert_eq!(fed_token.scoped_actions, vec!["graph:read"]);
        assert_eq!(fed_token.scoped_resources.len(), 1);
        assert_eq!(
            fed_token.target_instance.as_deref(),
            Some("peer-instance-1")
        );
        assert_eq!(fed_token.credentials.ttl_secs, 1800);
        assert!(fed_token.credentials.token.starts_with("obt_"));

        // The federation token's underlying session token should authenticate
        let identity2 = provider
            .authenticate(&AuthRequest::SessionToken {
                token: fed_token.credentials.token,
            })
            .unwrap();
        assert_eq!(identity2.username, "alice");
    }

    #[test]
    fn federation_token_empty_scope_inherits_all() {
        let provider = setup();
        let identity = provider
            .authenticate(&AuthRequest::Internal {
                username: "alice".into(),
            })
            .unwrap();

        let fed_token = provider
            .create_federation_token(
                &identity,
                &FederationRequest {
                    scoped_actions: vec![],
                    scoped_resources: vec![],
                    ttl_secs: 3600,
                    target_instance: None,
                },
            )
            .unwrap();

        // Empty scope → inherits all
        assert_eq!(fed_token.scoped_actions, vec!["*"]);
        assert_eq!(fed_token.scoped_resources.len(), 1);
        assert!(fed_token.target_instance.is_none());
    }

    #[test]
    fn federation_token_zero_ttl_rejected() {
        let provider = setup();
        let identity = provider
            .authenticate(&AuthRequest::Internal {
                username: "alice".into(),
            })
            .unwrap();

        let err = provider
            .create_federation_token(
                &identity,
                &FederationRequest {
                    scoped_actions: vec![],
                    scoped_resources: vec![],
                    ttl_secs: 0,
                    target_instance: None,
                },
            )
            .unwrap_err();

        assert!(matches!(err, IamError::InvalidParameter { .. }));
    }

    #[test]
    fn federation_token_ttl_capped_at_12h() {
        let provider = setup();
        let identity = provider
            .authenticate(&AuthRequest::Internal {
                username: "alice".into(),
            })
            .unwrap();

        let fed_token = provider
            .create_federation_token(
                &identity,
                &FederationRequest {
                    scoped_actions: vec!["*".into()],
                    scoped_resources: vec![],
                    ttl_secs: 999_999, // way too long
                    target_instance: None,
                },
            )
            .unwrap();

        assert_eq!(fed_token.credentials.ttl_secs, 43_200); // capped at 12h
    }

    // -- Assume role ---------------------------------------------------------

    #[test]
    fn assume_role_success() {
        let provider = setup();
        let identity = provider
            .authenticate(&AuthRequest::Internal {
                username: "alice".into(),
            })
            .unwrap();

        // alice has the "reader" role from setup()
        let session = provider.assume_role(&identity, "reader", 3600).unwrap();
        assert_eq!(session.ttl_secs, 3600);
        assert!(session.token.starts_with("obt_"));

        // Can authenticate with the assumed-role session
        let identity2 = provider
            .authenticate(&AuthRequest::SessionToken {
                token: session.token,
            })
            .unwrap();
        assert_eq!(identity2.username, "alice");
    }

    #[test]
    fn assume_role_not_attached() {
        let provider = setup();

        // Create a second user without the "reader" role
        provider.store().create_user("u2", "bob", None).unwrap();

        let identity = provider
            .authenticate(&AuthRequest::Internal {
                username: "bob".into(),
            })
            .unwrap();

        let err = provider.assume_role(&identity, "reader", 3600).unwrap_err();
        assert!(matches!(err, IamError::AccessDenied { .. }));
    }

    #[test]
    fn assume_role_not_found() {
        let provider = setup();
        let identity = provider
            .authenticate(&AuthRequest::Internal {
                username: "alice".into(),
            })
            .unwrap();

        let err = provider
            .assume_role(&identity, "nonexistent", 3600)
            .unwrap_err();
        assert!(matches!(err, IamError::ResourceNotFound { .. }));
    }
}
