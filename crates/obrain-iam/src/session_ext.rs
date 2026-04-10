//! Authenticated session — IAM context for query execution.
//!
//! [`AuthenticatedSession`] is a companion object that carries the
//! authenticated identity and provides policy-checking methods. It does
//! **not** wrap `Session` directly (to avoid circular dependencies between
//! `obrain-iam` and `obrain-engine`). Instead, the integration layer
//! (middleware, server) creates both a `Session` and an
//! `AuthenticatedSession`, and synchronizes them:
//!
//! ```text
//! // Middleware pseudocode:
//! let session = db.session();
//! let auth_session = AuthenticatedSession::new(identity, provider);
//!
//! // Synchronize graph scope
//! session.use_graph(auth_session.tenant());
//! tenant_manager.switch_tenant(auth_session.tenant());
//!
//! // Before query execution, check policies:
//! auth_session.check_access("graph:read", &resource_orn)?;
//! session.execute(query)?;
//! ```

use std::sync::Arc;

use crate::error::{IamError, IamResult};
use crate::model::PolicyDecision;
use crate::orn::Orn;
use crate::provider::{AuthProvider, Identity};

// ---------------------------------------------------------------------------
// AuthenticatedSession
// ---------------------------------------------------------------------------

/// An authenticated session context.
///
/// Carries the caller's [`Identity`], an [`AuthProvider`] reference for
/// policy evaluation, and the tenant scope. Provides `check_access()` to
/// enforce policies and automatic audit logging.
pub struct AuthenticatedSession {
    /// The authenticated identity.
    identity: Identity,
    /// The auth provider for policy evaluation.
    provider: Arc<dyn AuthProvider>,
    /// The tenant (named graph) this session is scoped to.
    tenant: Option<String>,
    /// Access counter (for metrics / rate limiting).
    access_count: std::sync::atomic::AtomicU64,
}

impl AuthenticatedSession {
    /// Creates a new authenticated session.
    ///
    /// # Arguments
    ///
    /// * `identity` — the authenticated caller
    /// * `provider` — the auth provider for policy checks
    pub fn new(identity: Identity, provider: Arc<dyn AuthProvider>) -> Self {
        Self {
            identity,
            provider,
            tenant: None,
            access_count: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Creates a new authenticated session scoped to a tenant.
    pub fn with_tenant(
        identity: Identity,
        provider: Arc<dyn AuthProvider>,
        tenant: impl Into<String>,
    ) -> Self {
        Self {
            identity,
            provider,
            tenant: Some(tenant.into()),
            access_count: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Returns the authenticated identity.
    pub fn identity(&self) -> &Identity {
        &self.identity
    }

    /// Returns the user ID.
    pub fn user_id(&self) -> &str {
        &self.identity.user_id
    }

    /// Returns the username.
    pub fn username(&self) -> &str {
        &self.identity.username
    }

    /// Returns the principal ORN.
    pub fn principal(&self) -> &Orn {
        &self.identity.principal
    }

    /// Returns the account ID.
    pub fn account_id(&self) -> &str {
        &self.identity.account_id
    }

    /// Returns the tenant this session is scoped to, if any.
    pub fn tenant(&self) -> Option<&str> {
        self.tenant.as_deref()
    }

    /// Returns the number of access checks performed in this session.
    pub fn access_count(&self) -> u64 {
        self.access_count.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Returns a reference to the auth provider.
    pub fn provider(&self) -> &dyn AuthProvider {
        self.provider.as_ref()
    }

    // -----------------------------------------------------------------------
    // Policy enforcement
    // -----------------------------------------------------------------------

    /// Checks whether the authenticated identity is allowed to perform
    /// `action` on `resource`. Logs an audit event regardless of the result.
    ///
    /// # Errors
    ///
    /// Returns `IamError::PermissionDenied` if the policy evaluation results
    /// in `Deny` or `ImplicitDeny`.
    pub fn check_access(&self, action: &str, resource: &Orn) -> IamResult<()> {
        self.access_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let decision = self.provider.authorize(&self.identity, action, resource)?;

        if decision.is_allowed() {
            Ok(())
        } else {
            Err(IamError::PermissionDenied {
                action: action.to_string(),
                resource: resource.to_string(),
            })
        }
    }

    /// Like `check_access`, but returns the raw [`PolicyDecision`] without
    /// converting to an error. Still logs the audit event.
    pub fn evaluate(&self, action: &str, resource: &Orn) -> IamResult<PolicyDecision> {
        self.access_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.provider.authorize(&self.identity, action, resource)
    }

    /// Checks access for a graph operation, automatically building the
    /// resource ORN from the tenant context.
    ///
    /// # Arguments
    ///
    /// * `action` — e.g. `"graph:read"`, `"graph:write"`, `"graph:delete"`
    /// * `resource_type` — e.g. `"node"`, `"edge"`, `"tenant"`
    /// * `resource_id` — the specific resource identifier (e.g. node ID)
    pub fn check_graph_access(
        &self,
        action: &str,
        resource_type: &str,
        resource_id: &str,
    ) -> IamResult<()> {
        let account = self.tenant().unwrap_or(self.account_id());
        let resource = Orn::new("graph", account, resource_type, resource_id)?;
        self.check_access(action, &resource)
    }

    /// Checks read access to a graph resource.
    pub fn check_read(&self, resource_type: &str, resource_id: &str) -> IamResult<()> {
        self.check_graph_access("graph:read", resource_type, resource_id)
    }

    /// Checks write access to a graph resource.
    pub fn check_write(&self, resource_type: &str, resource_id: &str) -> IamResult<()> {
        self.check_graph_access("graph:write", resource_type, resource_id)
    }

    /// Checks delete access to a graph resource.
    pub fn check_delete(&self, resource_type: &str, resource_id: &str) -> IamResult<()> {
        self.check_graph_access("graph:delete", resource_type, resource_id)
    }

    // -----------------------------------------------------------------------
    // Convenience: tenant-scoped ORN builders
    // -----------------------------------------------------------------------

    /// Builds a node ORN scoped to this session's tenant/account.
    pub fn node_orn(&self, node_id: u64) -> Orn {
        let account = self.tenant().unwrap_or(self.account_id());
        Orn::node(account, node_id)
    }

    /// Builds an edge ORN scoped to this session's tenant/account.
    pub fn edge_orn(&self, edge_id: u64) -> Orn {
        let account = self.tenant().unwrap_or(self.account_id());
        Orn::edge(account, edge_id)
    }

    /// Builds a tenant ORN.
    pub fn tenant_orn(&self) -> Option<Orn> {
        self.tenant().map(|t| Orn::tenant(self.account_id(), t))
    }
}

impl std::fmt::Debug for AuthenticatedSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AuthenticatedSession")
            .field("identity", &self.identity)
            .field("tenant", &self.tenant)
            .field("access_count", &self.access_count())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::PolicyEffect;
    use crate::provider::{AuthRequest, ObrainIamProvider};
    use crate::store::IamStore;
    use obrain_core::graph::lpg::LpgStore;

    fn setup() -> (Arc<dyn AuthProvider>, Identity) {
        let lpg = LpgStore::new().expect("LpgStore");
        let iam_store = Arc::new(IamStore::new(Arc::new(lpg)));

        // Bootstrap
        iam_store.create_user("u1", "alice", None).unwrap();
        iam_store.create_role("r1", "reader", None).unwrap();
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

        let provider: Arc<dyn AuthProvider> =
            Arc::new(ObrainIamProvider::new(iam_store, "default"));

        let identity = provider
            .authenticate(&AuthRequest::Internal {
                username: "alice".into(),
            })
            .unwrap();

        (provider, identity)
    }

    #[test]
    fn check_access_allowed() {
        let (provider, identity) = setup();
        let session = AuthenticatedSession::new(identity, provider);

        session
            .check_access("graph:read", &Orn::node("default", 42))
            .unwrap();

        assert_eq!(session.access_count(), 1);
    }

    #[test]
    fn check_access_denied() {
        let (provider, identity) = setup();
        let session = AuthenticatedSession::new(identity, provider);

        let err = session
            .check_access("graph:write", &Orn::node("default", 42))
            .unwrap_err();

        assert!(matches!(err, IamError::PermissionDenied { .. }));
        assert_eq!(session.access_count(), 1);
    }

    #[test]
    fn evaluate_returns_decision() {
        let (provider, identity) = setup();
        let session = AuthenticatedSession::new(identity, provider);

        let decision = session
            .evaluate("graph:read", &Orn::node("default", 1))
            .unwrap();
        assert!(decision.is_allowed());

        let decision = session
            .evaluate("graph:write", &Orn::node("default", 1))
            .unwrap();
        assert!(!decision.is_allowed());
    }

    #[test]
    fn check_graph_access_convenience() {
        let (provider, identity) = setup();
        let session = AuthenticatedSession::new(identity, provider);

        // Read → allowed
        session.check_read("node", "42").unwrap();

        // Write → denied
        let err = session.check_write("node", "42").unwrap_err();
        assert!(matches!(err, IamError::PermissionDenied { .. }));
    }

    #[test]
    fn tenant_scoped_session() {
        let (provider, identity) = setup();
        let session = AuthenticatedSession::with_tenant(identity, provider, "chess-kb");

        assert_eq!(session.tenant(), Some("chess-kb"));
        assert_eq!(session.username(), "alice");

        // Tenant ORN
        let orn = session.tenant_orn().unwrap();
        assert_eq!(orn.to_string(), "orn:obrain:graph:default:tenant/chess-kb");

        // Node ORN uses tenant as account scope
        let node = session.node_orn(42);
        assert_eq!(node.to_string(), "orn:obrain:graph:chess-kb:node/42");
    }

    #[test]
    fn no_tenant_session() {
        let (provider, identity) = setup();
        let session = AuthenticatedSession::new(identity, provider);

        assert!(session.tenant().is_none());
        assert!(session.tenant_orn().is_none());

        // Node ORN falls back to account_id
        let node = session.node_orn(7);
        assert_eq!(node.to_string(), "orn:obrain:graph:default:node/7");
    }

    #[test]
    fn identity_accessors() {
        let (provider, identity) = setup();
        let session = AuthenticatedSession::new(identity, provider);

        assert_eq!(session.user_id(), "u1");
        assert_eq!(session.username(), "alice");
        assert_eq!(session.account_id(), "default");
        assert_eq!(
            session.principal().to_string(),
            "orn:obrain:iam:default:user/alice"
        );
    }

    #[test]
    fn multiple_access_checks_counted() {
        let (provider, identity) = setup();
        let session = AuthenticatedSession::new(identity, provider);

        for _ in 0..5 {
            let _ = session.check_access("graph:read", &Orn::node("default", 1));
        }

        assert_eq!(session.access_count(), 5);
    }

    #[test]
    fn debug_format() {
        let (provider, identity) = setup();
        let session = AuthenticatedSession::new(identity, provider);
        let debug = format!("{session:?}");
        assert!(debug.contains("AuthenticatedSession"));
        assert!(debug.contains("alice"));
    }
}
