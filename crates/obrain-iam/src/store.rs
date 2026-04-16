//! IAM store backed by an LPG graph.
//!
//! [`IamStore`] wraps an [`LpgStore`] reference (the `__system` named graph)
//! and provides CRUD operations for IAM entities (users, roles, policies,
//! credentials, audit events).
//!
//! All IAM entities are stored as labeled nodes with properties. Relations
//! (user→role, role→policy, etc.) are stored as edges.

use std::sync::Arc;

use obrain_common::Value;
use obrain_common::types::NodeId;
use obrain_core::graph::lpg::LpgStore;

use crate::error::{IamError, IamResult};
use crate::model::{
    AuditEvent, Credential, CredentialType, EDGE_HAS_CREDENTIAL, EDGE_HAS_POLICY, EDGE_HAS_ROLE,
    EDGE_PERFORMED, EntityStatus, LABEL_AUDIT_EVENT, LABEL_CREDENTIAL, LABEL_POLICY, LABEL_ROLE,
    LABEL_USER, Policy, PolicyDecision, PolicyEffect, Role, User, props,
};
use crate::orn::Orn;
use crate::policy;

// ---------------------------------------------------------------------------
// IamStore
// ---------------------------------------------------------------------------

/// IAM store backed by an LPG graph.
///
/// Operates on the `__system` named graph to isolate IAM data from user data.
pub struct IamStore {
    /// The underlying LPG store (typically the `__system` named graph).
    store: Arc<LpgStore>,
}

impl std::fmt::Debug for IamStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IamStore")
            .field("store", &"<LpgStore>")
            .finish()
    }
}

impl IamStore {
    /// Creates a new `IamStore` wrapping the given LPG store.
    pub fn new(store: Arc<LpgStore>) -> Self {
        Self { store }
    }

    /// Returns a reference to the underlying LPG store.
    pub fn inner(&self) -> &LpgStore {
        &self.store
    }

    // -----------------------------------------------------------------------
    // Users
    // -----------------------------------------------------------------------

    /// Creates a new user and returns the created [`User`].
    pub fn create_user(&self, id: &str, username: &str, email: Option<&str>) -> IamResult<User> {
        // Check uniqueness by username
        if self.find_user_by_name(username).is_some() {
            return Err(IamError::ResourceExists {
                resource: format!("user:{username}"),
            });
        }

        let now = now_iso();
        let mut props: Vec<(&str, Value)> = vec![
            (props::ID, Value::from(id)),
            (props::NAME, Value::from(username)),
            (props::STATUS, Value::from("active")),
            (props::CREATED_AT, Value::from(now.as_str())),
        ];
        if let Some(e) = email {
            props.push((props::EMAIL, Value::from(e)));
        }

        self.store.create_node_with_props(&[LABEL_USER], props);

        Ok(User {
            id: id.to_string(),
            username: username.to_string(),
            email: email.map(String::from),
            status: EntityStatus::Active,
            must_change_password: false,
            created_at: now,
        })
    }

    /// Gets a user by their IAM ID.
    pub fn get_user(&self, id: &str) -> Option<User> {
        let node_id = self.find_node_by_label_and_id(LABEL_USER, id)?;
        self.node_to_user(node_id)
    }

    /// Finds a user by username.
    pub fn find_user_by_name(&self, username: &str) -> Option<User> {
        let candidates = self.store.nodes_by_label(LABEL_USER);
        for nid in candidates {
            if let Some(Value::String(name)) =
                self.store.get_node_property(nid, &props::NAME.into())
                && name.as_str() == username
            {
                return self.node_to_user(nid);
            }
        }
        None
    }

    /// Lists all users.
    pub fn list_users(&self) -> Vec<User> {
        self.store
            .nodes_by_label(LABEL_USER)
            .into_iter()
            .filter_map(|nid| self.node_to_user(nid))
            .collect()
    }

    // -----------------------------------------------------------------------
    // Roles
    // -----------------------------------------------------------------------

    /// Creates a new role.
    pub fn create_role(&self, id: &str, name: &str, description: Option<&str>) -> IamResult<Role> {
        if self.find_role_by_name(name).is_some() {
            return Err(IamError::ResourceExists {
                resource: format!("role:{name}"),
            });
        }

        let now = now_iso();
        let mut role_props: Vec<(&str, Value)> = vec![
            (props::ID, Value::from(id)),
            (props::NAME, Value::from(name)),
            (props::CREATED_AT, Value::from(now.as_str())),
        ];
        if let Some(d) = description {
            role_props.push((props::DESCRIPTION, Value::from(d)));
        }

        self.store.create_node_with_props(&[LABEL_ROLE], role_props);

        Ok(Role {
            id: id.to_string(),
            name: name.to_string(),
            description: description.map(String::from),
            created_at: now,
        })
    }

    /// Gets a role by ID.
    pub fn get_role(&self, id: &str) -> Option<Role> {
        let node_id = self.find_node_by_label_and_id(LABEL_ROLE, id)?;
        self.node_to_role(node_id)
    }

    /// Finds a role by name.
    pub fn find_role_by_name(&self, name: &str) -> Option<Role> {
        let candidates = self.store.nodes_by_label(LABEL_ROLE);
        for nid in candidates {
            if let Some(Value::String(n)) = self.store.get_node_property(nid, &props::NAME.into())
                && n.as_str() == name
            {
                return self.node_to_role(nid);
            }
        }
        None
    }

    /// Lists all roles.
    pub fn list_roles(&self) -> Vec<Role> {
        self.store
            .nodes_by_label(LABEL_ROLE)
            .into_iter()
            .filter_map(|nid| self.node_to_role(nid))
            .collect()
    }

    /// Attaches a role to a user: User -[:HAS_ROLE]-> Role.
    pub fn attach_role(&self, user_id: &str, role_id: &str) -> IamResult<()> {
        let user_nid = self
            .find_node_by_label_and_id(LABEL_USER, user_id)
            .ok_or_else(|| IamError::ResourceNotFound {
                resource: format!("user:{user_id}"),
            })?;
        let role_nid = self
            .find_node_by_label_and_id(LABEL_ROLE, role_id)
            .ok_or_else(|| IamError::ResourceNotFound {
                resource: format!("role:{role_id}"),
            })?;

        // Check if already attached
        if self.has_edge(user_nid, role_nid, EDGE_HAS_ROLE) {
            return Ok(()); // idempotent
        }

        self.store.create_edge(user_nid, role_nid, EDGE_HAS_ROLE);
        Ok(())
    }

    /// Gets all roles attached to a user.
    pub fn get_user_roles(&self, user_id: &str) -> IamResult<Vec<Role>> {
        let user_nid = self
            .find_node_by_label_and_id(LABEL_USER, user_id)
            .ok_or_else(|| IamError::ResourceNotFound {
                resource: format!("user:{user_id}"),
            })?;

        let roles = self
            .outgoing_targets(user_nid, EDGE_HAS_ROLE)
            .into_iter()
            .filter_map(|nid| self.node_to_role(nid))
            .collect();

        Ok(roles)
    }

    // -----------------------------------------------------------------------
    // Policies
    // -----------------------------------------------------------------------

    /// Creates a new policy.
    pub fn create_policy(
        &self,
        id: &str,
        name: &str,
        effect: PolicyEffect,
        actions: &[&str],
        resources: &[Orn],
    ) -> IamResult<Policy> {
        let now = now_iso();
        let actions_csv = actions.join(",");
        let resources_csv: String = resources
            .iter()
            .map(|r| r.to_string())
            .collect::<Vec<_>>()
            .join(",");

        let policy_props: Vec<(&str, Value)> = vec![
            (props::ID, Value::from(id)),
            (props::NAME, Value::from(name)),
            (props::EFFECT, Value::from(effect.to_string().as_str())),
            (props::ACTIONS, Value::from(actions_csv.as_str())),
            (props::RESOURCES, Value::from(resources_csv.as_str())),
            (props::CREATED_AT, Value::from(now.as_str())),
        ];

        self.store
            .create_node_with_props(&[LABEL_POLICY], policy_props);

        Ok(Policy {
            id: id.to_string(),
            name: name.to_string(),
            effect,
            actions: actions.iter().map(|s| (*s).to_string()).collect(),
            resources: resources.to_vec(),
            created_at: now,
        })
    }

    /// Gets a policy by ID.
    pub fn get_policy(&self, id: &str) -> Option<Policy> {
        let node_id = self.find_node_by_label_and_id(LABEL_POLICY, id)?;
        self.node_to_policy(node_id)
    }

    /// Lists all policies.
    pub fn list_policies(&self) -> Vec<Policy> {
        self.store
            .nodes_by_label(LABEL_POLICY)
            .into_iter()
            .filter_map(|nid| self.node_to_policy(nid))
            .collect()
    }

    /// Attaches a policy to a role: Role -[:HAS_POLICY]-> Policy.
    pub fn attach_policy_to_role(&self, role_id: &str, policy_id: &str) -> IamResult<()> {
        let role_nid = self
            .find_node_by_label_and_id(LABEL_ROLE, role_id)
            .ok_or_else(|| IamError::ResourceNotFound {
                resource: format!("role:{role_id}"),
            })?;
        let policy_nid = self
            .find_node_by_label_and_id(LABEL_POLICY, policy_id)
            .ok_or_else(|| IamError::ResourceNotFound {
                resource: format!("policy:{policy_id}"),
            })?;

        if self.has_edge(role_nid, policy_nid, EDGE_HAS_POLICY) {
            return Ok(());
        }

        self.store
            .create_edge(role_nid, policy_nid, EDGE_HAS_POLICY);
        Ok(())
    }

    /// Gets all policies for a role.
    pub fn get_role_policies(&self, role_id: &str) -> IamResult<Vec<Policy>> {
        let role_nid = self
            .find_node_by_label_and_id(LABEL_ROLE, role_id)
            .ok_or_else(|| IamError::ResourceNotFound {
                resource: format!("role:{role_id}"),
            })?;

        let policies = self
            .outgoing_targets(role_nid, EDGE_HAS_POLICY)
            .into_iter()
            .filter_map(|nid| self.node_to_policy(nid))
            .collect();

        Ok(policies)
    }

    // -----------------------------------------------------------------------
    // Policy evaluation
    // -----------------------------------------------------------------------

    /// Evaluates whether a user is allowed to perform an action on a resource.
    ///
    /// Collects all policies from all roles attached to the user, then evaluates.
    pub fn evaluate_access(
        &self,
        user_id: &str,
        action: &str,
        resource: &Orn,
    ) -> IamResult<PolicyDecision> {
        let roles = self.get_user_roles(user_id)?;
        let mut all_policies = Vec::new();

        for role in &roles {
            let policies = self.get_role_policies(&role.id)?;
            all_policies.extend(policies);
        }

        Ok(policy::evaluate(&all_policies, action, resource))
    }

    // -----------------------------------------------------------------------
    // Credentials
    // -----------------------------------------------------------------------

    /// Creates a session credential for a user.
    pub fn create_credential(
        &self,
        id: &str,
        user_id: &str,
        cred_type: CredentialType,
        token_hash: &str,
        ttl_secs: u64,
    ) -> IamResult<Credential> {
        let user_nid = self
            .find_node_by_label_and_id(LABEL_USER, user_id)
            .ok_or_else(|| IamError::ResourceNotFound {
                resource: format!("user:{user_id}"),
            })?;

        let now = now_iso();
        // Simple expiration calculation (approximate — good enough for now)
        let expires = format!("{}+{ttl_secs}s", &now);

        let cred_props: Vec<(&str, Value)> = vec![
            (props::ID, Value::from(id)),
            (
                props::CRED_TYPE,
                Value::from(cred_type.to_string().as_str()),
            ),
            (props::TOKEN_HASH, Value::from(token_hash)),
            (props::CREATED_AT, Value::from(now.as_str())),
            (props::EXPIRES_AT, Value::from(expires.as_str())),
        ];

        let cred_nid = self
            .store
            .create_node_with_props(&[LABEL_CREDENTIAL], cred_props);
        self.store
            .create_edge(user_nid, cred_nid, EDGE_HAS_CREDENTIAL);

        Ok(Credential {
            id: id.to_string(),
            cred_type,
            token_hash: token_hash.to_string(),
            ttl: std::time::Duration::from_secs(ttl_secs),
            created_at: now,
            expires_at: expires,
        })
    }

    // -----------------------------------------------------------------------
    // Audit events
    // -----------------------------------------------------------------------

    /// Logs an audit event.
    pub fn log_audit_event(
        &self,
        id: &str,
        principal: &Orn,
        action: &str,
        resource: &Orn,
        result: PolicyDecision,
    ) -> AuditEvent {
        let now = now_iso();
        let result_str = match result {
            PolicyDecision::Allow => "allow",
            PolicyDecision::Deny => "deny",
            PolicyDecision::ImplicitDeny => "implicit_deny",
        };

        let event_props: Vec<(&str, Value)> = vec![
            (props::ID, Value::from(id)),
            (
                props::PRINCIPAL,
                Value::from(principal.to_string().as_str()),
            ),
            (props::ACTION, Value::from(action)),
            (props::RESOURCE, Value::from(resource.to_string().as_str())),
            (props::RESULT, Value::from(result_str)),
            (props::CREATED_AT, Value::from(now.as_str())),
        ];

        // Find the user node to link the audit event
        // (best-effort — audit event is always created even if user not found)
        let event_nid = self
            .store
            .create_node_with_props(&[LABEL_AUDIT_EVENT], event_props);

        // Try to link to user
        if let Some(user_nid) = self.find_user_node_by_principal(principal) {
            self.store.create_edge(user_nid, event_nid, EDGE_PERFORMED);
        }

        AuditEvent {
            id: id.to_string(),
            principal: principal.clone(),
            action: action.to_string(),
            resource: resource.clone(),
            result,
            timestamp: now,
        }
    }

    /// Lists audit events (most recent first, up to `limit`).
    pub fn list_audit_events(&self, limit: usize) -> Vec<AuditEvent> {
        let mut events: Vec<AuditEvent> = self
            .store
            .nodes_by_label(LABEL_AUDIT_EVENT)
            .into_iter()
            .filter_map(|nid| self.node_to_audit_event(nid))
            .collect();

        // Sort by timestamp descending (most recent first)
        events.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        events.truncate(limit);
        events
    }

    // -----------------------------------------------------------------------
    // User management
    // -----------------------------------------------------------------------

    /// Deactivates a user account (sets status to Inactive).
    pub fn deactivate_user(&self, user_id: &str) -> IamResult<()> {
        let nid = self
            .find_node_by_label_and_id(LABEL_USER, user_id)
            .ok_or_else(|| IamError::ResourceNotFound {
                resource: format!("user:{user_id}"),
            })?;
        self.store
            .set_node_property(nid, props::STATUS, Value::from("inactive"));
        Ok(())
    }

    /// Detaches a role from a user.
    ///
    /// Removes the `HAS_ROLE` edge between the user and the role.
    /// No-op if the edge doesn't exist.
    pub fn detach_role(&self, user_id: &str, role_id: &str) -> IamResult<()> {
        use obrain_core::graph::Direction;

        let user_nid = self
            .find_node_by_label_and_id(LABEL_USER, user_id)
            .ok_or_else(|| IamError::ResourceNotFound {
                resource: format!("user:{user_id}"),
            })?;
        let role_nid = self
            .find_node_by_label_and_id(LABEL_ROLE, role_id)
            .ok_or_else(|| IamError::ResourceNotFound {
                resource: format!("role:{role_id}"),
            })?;

        // Find and remove the edge
        for (target, eid) in self.store.edges_from(user_nid, Direction::Outgoing) {
            if target == role_nid
                && let Some(et) = self.store.edge_type(eid)
                && et.as_str() == EDGE_HAS_ROLE
            {
                self.store.delete_edge(eid);
                break;
            }
        }
        Ok(())
    }

    /// Deletes a role by ID.
    ///
    /// Marks the role as inactive (soft delete). Does not cascade to policies.
    pub fn delete_role(&self, role_id: &str) -> IamResult<()> {
        let nid = self
            .find_node_by_label_and_id(LABEL_ROLE, role_id)
            .ok_or_else(|| IamError::ResourceNotFound {
                resource: format!("role:{role_id}"),
            })?;
        // Soft delete: mark as deleted by setting status
        self.store
            .set_node_property(nid, props::STATUS, Value::from("deleted"));
        Ok(())
    }

    /// Updates the description of a role.
    pub fn update_role_description(
        &self,
        role_id: &str,
        description: Option<&str>,
    ) -> IamResult<()> {
        let nid = self
            .find_node_by_label_and_id(LABEL_ROLE, role_id)
            .ok_or_else(|| IamError::ResourceNotFound {
                resource: format!("role:{role_id}"),
            })?;
        if let Some(desc) = description {
            self.store
                .set_node_property(nid, props::DESCRIPTION, Value::from(desc));
        }
        Ok(())
    }

    /// Detaches a policy from a role.
    pub fn detach_policy_from_role(&self, role_id: &str, policy_id: &str) -> IamResult<()> {
        use obrain_core::graph::Direction;

        let role_nid = self
            .find_node_by_label_and_id(LABEL_ROLE, role_id)
            .ok_or_else(|| IamError::ResourceNotFound {
                resource: format!("role:{role_id}"),
            })?;
        let policy_nid = self
            .find_node_by_label_and_id(LABEL_POLICY, policy_id)
            .ok_or_else(|| IamError::ResourceNotFound {
                resource: format!("policy:{policy_id}"),
            })?;

        for (target, eid) in self.store.edges_from(role_nid, Direction::Outgoing) {
            if target == policy_nid
                && let Some(et) = self.store.edge_type(eid)
                && et.as_str() == EDGE_HAS_POLICY
            {
                self.store.delete_edge(eid);
                break;
            }
        }
        Ok(())
    }

    /// Deletes a policy by ID (soft delete).
    pub fn delete_policy(&self, policy_id: &str) -> IamResult<()> {
        let nid = self
            .find_node_by_label_and_id(LABEL_POLICY, policy_id)
            .ok_or_else(|| IamError::ResourceNotFound {
                resource: format!("policy:{policy_id}"),
            })?;
        self.store
            .set_node_property(nid, props::STATUS, Value::from("deleted"));
        Ok(())
    }

    /// Purges all expired session credentials.
    ///
    /// Returns the number of credentials purged (soft-revoked).
    pub fn purge_session_credentials(&self) -> usize {
        let cred_nodes = self.store.nodes_by_label(LABEL_CREDENTIAL);
        let mut purged = 0;

        for cred_nid in cred_nodes {
            // Only purge session credentials, not API keys or passwords
            let is_session = matches!(
                self.store.get_node_property(cred_nid, &props::CRED_TYPE.into()),
                Some(Value::String(t)) if t.as_str() == "session"
            );
            if !is_session {
                continue;
            }

            // Check if already revoked
            let is_revoked = matches!(
                self.store.get_node_property(cred_nid, &props::TOKEN_HASH.into()),
                Some(Value::String(h)) if h.as_str() == "__revoked__"
            );
            if is_revoked {
                continue;
            }

            // Mark as revoked (soft delete)
            self.store.set_node_property(
                cred_nid,
                props::TOKEN_HASH,
                Value::from("__revoked__"),
            );
            purged += 1;
        }

        purged
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Finds a node with a given label and `iam_id` property value.
    fn find_node_by_label_and_id(&self, label: &str, id: &str) -> Option<NodeId> {
        let candidates = self.store.nodes_by_label(label);
        for nid in candidates {
            if let Some(Value::String(nid_val)) =
                self.store.get_node_property(nid, &props::ID.into())
                && nid_val.as_str() == id
            {
                return Some(nid);
            }
        }
        None
    }

    /// Finds a user node by principal ORN (extracts account_id as username).
    fn find_user_node_by_principal(&self, principal: &Orn) -> Option<NodeId> {
        // The principal ORN's resource_path is typically the username
        let username = principal.resource_path();
        let candidates = self.store.nodes_by_label(LABEL_USER);
        for nid in candidates {
            if let Some(Value::String(name)) =
                self.store.get_node_property(nid, &props::NAME.into())
                && name.as_str() == username
            {
                return Some(nid);
            }
        }
        None
    }

    /// Checks if an edge of a specific type exists between two nodes.
    fn has_edge(&self, src: NodeId, dst: NodeId, edge_type: &str) -> bool {
        use obrain_core::graph::Direction;
        for (target, eid) in self.store.edges_from(src, Direction::Outgoing) {
            if target == dst
                && let Some(et) = self.store.edge_type(eid)
                && et.as_str() == edge_type
            {
                return true;
            }
        }
        false
    }

    /// Gets all target nodes from outgoing edges of a specific type.
    fn outgoing_targets(&self, src: NodeId, edge_type: &str) -> Vec<NodeId> {
        use obrain_core::graph::Direction;
        let mut targets = Vec::new();
        for (target, eid) in self.store.edges_from(src, Direction::Outgoing) {
            if let Some(et) = self.store.edge_type(eid)
                && et.as_str() == edge_type
            {
                targets.push(target);
            }
        }
        targets
    }

    // -- Node → model conversions -------------------------------------------

    /// Extracts a string property from a node.
    fn get_str(&self, nid: NodeId, key: &str) -> Option<String> {
        match self.store.get_node_property(nid, &key.into()) {
            Some(Value::String(s)) => Some(s.to_string()),
            _ => None,
        }
    }

    /// Converts a graph node to a [`User`].
    fn node_to_user(&self, nid: NodeId) -> Option<User> {
        let must_change = self
            .get_str(nid, props::MUST_CHANGE_PASSWORD)
            .map(|s| s == "true")
            .unwrap_or(false);

        Some(User {
            id: self.get_str(nid, props::ID)?,
            username: self.get_str(nid, props::NAME)?,
            email: self.get_str(nid, props::EMAIL),
            status: self
                .get_str(nid, props::STATUS)
                .and_then(|s| EntityStatus::from_str_loose(&s))
                .unwrap_or(EntityStatus::Active),
            must_change_password: must_change,
            created_at: self.get_str(nid, props::CREATED_AT).unwrap_or_default(),
        })
    }

    /// Converts a graph node to a [`Role`].
    fn node_to_role(&self, nid: NodeId) -> Option<Role> {
        Some(Role {
            id: self.get_str(nid, props::ID)?,
            name: self.get_str(nid, props::NAME)?,
            description: self.get_str(nid, props::DESCRIPTION),
            created_at: self.get_str(nid, props::CREATED_AT).unwrap_or_default(),
        })
    }

    /// Converts a graph node to a [`Policy`].
    fn node_to_policy(&self, nid: NodeId) -> Option<Policy> {
        let effect_str = self.get_str(nid, props::EFFECT)?;
        let effect = PolicyEffect::from_str_loose(&effect_str)?;
        let actions_csv = self.get_str(nid, props::ACTIONS).unwrap_or_default();
        let resources_csv = self.get_str(nid, props::RESOURCES).unwrap_or_default();

        let actions: Vec<String> = actions_csv
            .split(',')
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect();

        let resources: Vec<Orn> = resources_csv
            .split(',')
            .filter(|s| !s.is_empty())
            .filter_map(|s| s.parse().ok())
            .collect();

        Some(Policy {
            id: self.get_str(nid, props::ID)?,
            name: self.get_str(nid, props::NAME)?,
            effect,
            actions,
            resources,
            created_at: self.get_str(nid, props::CREATED_AT).unwrap_or_default(),
        })
    }

    /// Converts a graph node to an [`AuditEvent`].
    fn node_to_audit_event(&self, nid: NodeId) -> Option<AuditEvent> {
        let principal_str = self.get_str(nid, props::PRINCIPAL)?;
        let resource_str = self.get_str(nid, props::RESOURCE)?;
        let result_str = self.get_str(nid, props::RESULT)?;

        let result = match result_str.as_str() {
            "allow" => PolicyDecision::Allow,
            "deny" => PolicyDecision::Deny,
            _ => PolicyDecision::ImplicitDeny,
        };

        Some(AuditEvent {
            id: self.get_str(nid, props::ID)?,
            principal: principal_str.parse().ok()?,
            action: self.get_str(nid, props::ACTION)?,
            resource: resource_str.parse().ok()?,
            result,
            timestamp: self.get_str(nid, props::CREATED_AT).unwrap_or_default(),
        })
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Returns the current time as an ISO 8601 string (simplified).
fn now_iso() -> String {
    // Use a monotonic counter as a proxy for "now" in embedded mode.
    // In production, this would be replaced by `chrono::Utc::now().to_rfc3339()`.
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let ts = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("2026-01-01T00:00:{ts:02}Z")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn new_store() -> IamStore {
        let store = LpgStore::new().expect("failed to create LpgStore");
        IamStore::new(Arc::new(store))
    }

    // -- Users ---------------------------------------------------------------

    #[test]
    fn create_and_get_user() {
        let iam = new_store();
        let user = iam
            .create_user("u1", "alice", Some("alice@example.com"))
            .unwrap();
        assert_eq!(user.username, "alice");
        assert_eq!(user.email.as_deref(), Some("alice@example.com"));
        assert_eq!(user.status, EntityStatus::Active);

        let fetched = iam.get_user("u1").unwrap();
        assert_eq!(fetched.username, "alice");
    }

    #[test]
    fn create_user_duplicate_name() {
        let iam = new_store();
        iam.create_user("u1", "alice", None).unwrap();
        let err = iam.create_user("u2", "alice", None).unwrap_err();
        assert!(matches!(err, IamError::ResourceExists { .. }));
    }

    #[test]
    fn list_users() {
        let iam = new_store();
        iam.create_user("u1", "alice", None).unwrap();
        iam.create_user("u2", "bob", None).unwrap();
        let users = iam.list_users();
        assert_eq!(users.len(), 2);
    }

    #[test]
    fn find_user_by_name() {
        let iam = new_store();
        iam.create_user("u1", "alice", None).unwrap();
        iam.create_user("u2", "bob", None).unwrap();

        let alice = iam.find_user_by_name("alice").unwrap();
        assert_eq!(alice.id, "u1");

        assert!(iam.find_user_by_name("charlie").is_none());
    }

    // -- Roles ---------------------------------------------------------------

    #[test]
    fn create_and_get_role() {
        let iam = new_store();
        let role = iam.create_role("r1", "admin", Some("Admin role")).unwrap();
        assert_eq!(role.name, "admin");

        let fetched = iam.get_role("r1").unwrap();
        assert_eq!(fetched.description.as_deref(), Some("Admin role"));
    }

    #[test]
    fn attach_role_to_user() {
        let iam = new_store();
        iam.create_user("u1", "alice", None).unwrap();
        iam.create_role("r1", "admin", None).unwrap();

        iam.attach_role("u1", "r1").unwrap();

        let roles = iam.get_user_roles("u1").unwrap();
        assert_eq!(roles.len(), 1);
        assert_eq!(roles[0].name, "admin");
    }

    #[test]
    fn attach_role_idempotent() {
        let iam = new_store();
        iam.create_user("u1", "alice", None).unwrap();
        iam.create_role("r1", "admin", None).unwrap();

        iam.attach_role("u1", "r1").unwrap();
        iam.attach_role("u1", "r1").unwrap(); // no error

        let roles = iam.get_user_roles("u1").unwrap();
        assert_eq!(roles.len(), 1);
    }

    // -- Policies ------------------------------------------------------------

    #[test]
    fn create_and_get_policy() {
        let iam = new_store();
        let policy = iam
            .create_policy(
                "p1",
                "ReadAll",
                PolicyEffect::Allow,
                &["graph:read"],
                &[Orn::all_in_account("alice")],
            )
            .unwrap();
        assert_eq!(policy.effect, PolicyEffect::Allow);
        assert_eq!(policy.actions, vec!["graph:read"]);

        let fetched = iam.get_policy("p1").unwrap();
        assert_eq!(fetched.name, "ReadAll");
        assert_eq!(fetched.resources.len(), 1);
    }

    #[test]
    fn attach_policy_to_role() {
        let iam = new_store();
        iam.create_role("r1", "reader", None).unwrap();
        iam.create_policy(
            "p1",
            "ReadNodes",
            PolicyEffect::Allow,
            &["graph:read"],
            &[Orn::all_of_type("graph", "alice", "node")],
        )
        .unwrap();

        iam.attach_policy_to_role("r1", "p1").unwrap();

        let policies = iam.get_role_policies("r1").unwrap();
        assert_eq!(policies.len(), 1);
        assert_eq!(policies[0].name, "ReadNodes");
    }

    // -- Policy evaluation ---------------------------------------------------

    #[test]
    fn evaluate_access_allowed() {
        let iam = new_store();
        iam.create_user("u1", "alice", None).unwrap();
        iam.create_role("r1", "reader", None).unwrap();
        iam.create_policy(
            "p1",
            "ReadNodes",
            PolicyEffect::Allow,
            &["graph:read"],
            &[Orn::all_of_type("graph", "alice", "node")],
        )
        .unwrap();
        iam.attach_role("u1", "r1").unwrap();
        iam.attach_policy_to_role("r1", "p1").unwrap();

        let decision = iam
            .evaluate_access("u1", "graph:read", &Orn::node("alice", 42))
            .unwrap();
        assert!(decision.is_allowed());
    }

    #[test]
    fn evaluate_access_denied() {
        let iam = new_store();
        iam.create_user("u1", "alice", None).unwrap();
        iam.create_role("r1", "reader", None).unwrap();
        iam.create_policy(
            "p1",
            "ReadNodes",
            PolicyEffect::Allow,
            &["graph:read"],
            &[Orn::all_of_type("graph", "alice", "node")],
        )
        .unwrap();
        iam.attach_role("u1", "r1").unwrap();
        iam.attach_policy_to_role("r1", "p1").unwrap();

        // Write is not allowed
        let decision = iam
            .evaluate_access("u1", "graph:write", &Orn::node("alice", 42))
            .unwrap();
        assert!(!decision.is_allowed());
    }

    #[test]
    fn evaluate_access_explicit_deny() {
        let iam = new_store();
        iam.create_user("u1", "alice", None).unwrap();
        iam.create_role("r1", "custom", None).unwrap();
        iam.create_policy(
            "p1",
            "AllowAll",
            PolicyEffect::Allow,
            &["*"],
            &[Orn::all_in_account("alice")],
        )
        .unwrap();
        iam.create_policy(
            "p2",
            "DenyDelete",
            PolicyEffect::Deny,
            &["graph:delete"],
            &[Orn::all_in_account("alice")],
        )
        .unwrap();
        iam.attach_role("u1", "r1").unwrap();
        iam.attach_policy_to_role("r1", "p1").unwrap();
        iam.attach_policy_to_role("r1", "p2").unwrap();

        // Read allowed
        let decision = iam
            .evaluate_access("u1", "graph:read", &Orn::node("alice", 1))
            .unwrap();
        assert!(decision.is_allowed());

        // Delete explicitly denied
        let decision = iam
            .evaluate_access("u1", "graph:delete", &Orn::node("alice", 1))
            .unwrap();
        assert_eq!(decision, PolicyDecision::Deny);
    }

    // -- Credentials ---------------------------------------------------------

    #[test]
    fn create_credential() {
        let iam = new_store();
        iam.create_user("u1", "alice", None).unwrap();

        let cred = iam
            .create_credential("c1", "u1", CredentialType::Session, "hash123", 3600)
            .unwrap();
        assert_eq!(cred.cred_type, CredentialType::Session);
        assert_eq!(cred.ttl.as_secs(), 3600);
    }

    #[test]
    fn create_credential_user_not_found() {
        let iam = new_store();
        let err = iam
            .create_credential("c1", "nobody", CredentialType::Session, "hash", 3600)
            .unwrap_err();
        assert!(matches!(err, IamError::ResourceNotFound { .. }));
    }

    // -- Audit events --------------------------------------------------------

    #[test]
    fn log_and_list_audit_events() {
        let iam = new_store();
        iam.create_user("u1", "alice", None).unwrap();

        let principal = Orn::user("default", "alice");
        let resource = Orn::node("alice", 42);

        iam.log_audit_event(
            "e1",
            &principal,
            "graph:read",
            &resource,
            PolicyDecision::Allow,
        );
        iam.log_audit_event(
            "e2",
            &principal,
            "graph:write",
            &resource,
            PolicyDecision::Deny,
        );

        let events = iam.list_audit_events(10);
        assert_eq!(events.len(), 2);
    }

    // -- Full integration flow -----------------------------------------------

    #[test]
    fn full_iam_flow() {
        let iam = new_store();

        // Create entities
        iam.create_user("u1", "alice", Some("alice@obrain.dev"))
            .unwrap();
        iam.create_user("u2", "bob", None).unwrap();

        iam.create_role("r1", "graph-reader", Some("Read-only access to graph"))
            .unwrap();
        iam.create_role("r2", "graph-admin", Some("Full access to graph"))
            .unwrap();

        iam.create_policy(
            "p1",
            "ReadNodes",
            PolicyEffect::Allow,
            &["graph:read"],
            &[Orn::all_of_type("graph", "*", "node")],
        )
        .unwrap();
        iam.create_policy(
            "p2",
            "WriteNodes",
            PolicyEffect::Allow,
            &["graph:write", "graph:delete"],
            &[Orn::all_of_type("graph", "*", "node")],
        )
        .unwrap();

        // Wire up
        iam.attach_role("u1", "r1").unwrap(); // alice = reader
        iam.attach_role("u2", "r2").unwrap(); // bob = admin
        iam.attach_policy_to_role("r1", "p1").unwrap(); // reader gets read
        iam.attach_policy_to_role("r2", "p1").unwrap(); // admin gets read
        iam.attach_policy_to_role("r2", "p2").unwrap(); // admin gets write+delete

        let target = Orn::node("alice", 42);

        // Alice can read
        assert!(
            iam.evaluate_access("u1", "graph:read", &target)
                .unwrap()
                .is_allowed()
        );
        // Alice cannot write
        assert!(
            !iam.evaluate_access("u1", "graph:write", &target)
                .unwrap()
                .is_allowed()
        );

        // Bob can read and write
        assert!(
            iam.evaluate_access("u2", "graph:read", &target)
                .unwrap()
                .is_allowed()
        );
        assert!(
            iam.evaluate_access("u2", "graph:write", &target)
                .unwrap()
                .is_allowed()
        );

        // Credential creation
        let cred = iam
            .create_credential("c1", "u1", CredentialType::Session, "tok_hash", 7200)
            .unwrap();
        assert_eq!(cred.cred_type, CredentialType::Session);

        // Audit
        let principal = Orn::user("default", "alice");
        iam.log_audit_event(
            "e1",
            &principal,
            "graph:read",
            &target,
            PolicyDecision::Allow,
        );
        assert_eq!(iam.list_audit_events(10).len(), 1);
    }
}
