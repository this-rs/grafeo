//! Policy evaluation engine.
//!
//! Evaluates a set of [`Policy`] statements against an action + resource ORN.
//!
//! Evaluation follows the standard IAM model:
//! 1. Any explicit **Deny** → access denied (deny wins over allow)
//! 2. Any explicit **Allow** → access granted
//! 3. No match → **implicit deny**

use crate::model::{Policy, PolicyDecision, PolicyEffect};
use crate::orn::Orn;

/// Evaluates a collection of policies against an action and resource.
///
/// # Arguments
///
/// * `policies` — all policies attached to the caller (via roles)
/// * `action` — the action being attempted (e.g. `"graph:read"`, `"iam:create_user"`)
/// * `resource` — the target resource ORN
///
/// # Returns
///
/// A [`PolicyDecision`]: `Allow`, `Deny`, or `ImplicitDeny`.
///
/// # Examples
///
/// ```
/// use obrain_iam::policy::evaluate;
/// use obrain_iam::model::{Policy, PolicyEffect, PolicyDecision};
/// use obrain_iam::orn::Orn;
///
/// let allow_all = Policy {
///     id: "p1".into(),
///     name: "AllowAll".into(),
///     effect: PolicyEffect::Allow,
///     actions: vec!["*".into()],
///     resources: vec![Orn::all_in_account("alice")],
///     created_at: "2026-01-01T00:00:00Z".into(),
/// };
///
/// let resource = Orn::node("alice", 42);
/// let decision = evaluate(&[allow_all], "graph:read", &resource);
/// assert!(decision.is_allowed());
/// ```
pub fn evaluate(policies: &[Policy], action: &str, resource: &Orn) -> PolicyDecision {
    let mut has_allow = false;

    for policy in policies {
        if !action_matches(&policy.actions, action) {
            continue;
        }
        if !resource_matches(&policy.resources, resource) {
            continue;
        }

        match policy.effect {
            PolicyEffect::Deny => return PolicyDecision::Deny,
            PolicyEffect::Allow => has_allow = true,
        }
    }

    if has_allow {
        PolicyDecision::Allow
    } else {
        PolicyDecision::ImplicitDeny
    }
}

/// Checks if the action matches any of the policy's action patterns.
///
/// Supports:
/// - `"*"` matches any action
/// - `"graph:*"` matches any action starting with `"graph:"`
/// - Exact match
fn action_matches(patterns: &[String], action: &str) -> bool {
    patterns.iter().any(|pattern| {
        if pattern == "*" {
            return true;
        }
        if let Some(prefix) = pattern.strip_suffix('*') {
            return action.starts_with(prefix);
        }
        pattern == action
    })
}

/// Checks if the resource matches any of the policy's resource ORN patterns.
fn resource_matches(patterns: &[Orn], resource: &Orn) -> bool {
    patterns.iter().any(|pattern| pattern.matches(resource))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{PolicyDecision, PolicyEffect};

    fn make_policy(
        id: &str,
        effect: PolicyEffect,
        actions: &[&str],
        resources: Vec<Orn>,
    ) -> Policy {
        Policy {
            id: id.into(),
            name: id.into(),
            effect,
            actions: actions.iter().map(|s| (*s).to_string()).collect(),
            resources,
            created_at: "2026-01-01T00:00:00Z".into(),
        }
    }

    #[test]
    fn allow_all() {
        let policy = make_policy(
            "allow-all",
            PolicyEffect::Allow,
            &["*"],
            vec![Orn::all_in_account("alice")],
        );
        let resource = Orn::node("alice", 42);
        assert_eq!(evaluate(&[policy], "graph:read", &resource), PolicyDecision::Allow);
    }

    #[test]
    fn implicit_deny_no_policies() {
        let resource = Orn::node("alice", 42);
        assert_eq!(evaluate(&[], "graph:read", &resource), PolicyDecision::ImplicitDeny);
    }

    #[test]
    fn explicit_deny_overrides_allow() {
        let allow = make_policy(
            "allow",
            PolicyEffect::Allow,
            &["graph:*"],
            vec![Orn::all_of_type("graph", "alice", "node")],
        );
        let deny = make_policy(
            "deny",
            PolicyEffect::Deny,
            &["graph:write"],
            vec![Orn::all_of_type("graph", "alice", "node")],
        );
        let resource = Orn::node("alice", 42);

        // Deny should override allow for graph:write
        assert_eq!(
            evaluate(&[allow.clone(), deny], "graph:write", &resource),
            PolicyDecision::Deny,
        );

        // graph:read should still be allowed
        assert_eq!(
            evaluate(&[allow], "graph:read", &resource),
            PolicyDecision::Allow,
        );
    }

    #[test]
    fn action_prefix_matching() {
        let policy = make_policy(
            "graph-read",
            PolicyEffect::Allow,
            &["graph:*"],
            vec![Orn::all_in_account("alice")],
        );
        let resource = Orn::node("alice", 1);

        assert_eq!(evaluate(&[policy.clone()], "graph:read", &resource), PolicyDecision::Allow);
        assert_eq!(evaluate(&[policy.clone()], "graph:write", &resource), PolicyDecision::Allow);
        assert_eq!(evaluate(&[policy], "iam:create_user", &resource), PolicyDecision::ImplicitDeny);
    }

    #[test]
    fn resource_wildcard_matching() {
        let policy = make_policy(
            "all-nodes",
            PolicyEffect::Allow,
            &["graph:read"],
            vec![Orn::all_of_type("graph", "alice", "node")],
        );
        let node = Orn::node("alice", 42);
        let edge = Orn::edge("alice", 7);

        assert_eq!(evaluate(&[policy.clone()], "graph:read", &node), PolicyDecision::Allow);
        assert_eq!(evaluate(&[policy], "graph:read", &edge), PolicyDecision::ImplicitDeny);
    }

    #[test]
    fn wrong_account_denied() {
        let policy = make_policy(
            "alice-only",
            PolicyEffect::Allow,
            &["*"],
            vec![Orn::all_in_account("alice")],
        );
        let bob_resource = Orn::node("bob", 1);
        assert_eq!(evaluate(&[policy], "graph:read", &bob_resource), PolicyDecision::ImplicitDeny);
    }

    #[test]
    fn multiple_policies_combined() {
        let read_nodes = make_policy(
            "read-nodes",
            PolicyEffect::Allow,
            &["graph:read"],
            vec![Orn::all_of_type("graph", "alice", "node")],
        );
        let write_edges = make_policy(
            "write-edges",
            PolicyEffect::Allow,
            &["graph:write"],
            vec![Orn::all_of_type("graph", "alice", "edge")],
        );
        let policies = vec![read_nodes, write_edges];

        assert_eq!(
            evaluate(&policies, "graph:read", &Orn::node("alice", 1)),
            PolicyDecision::Allow,
        );
        assert_eq!(
            evaluate(&policies, "graph:write", &Orn::edge("alice", 1)),
            PolicyDecision::Allow,
        );
        // Write on nodes → not allowed
        assert_eq!(
            evaluate(&policies, "graph:write", &Orn::node("alice", 1)),
            PolicyDecision::ImplicitDeny,
        );
    }

    #[test]
    fn exact_action_match() {
        let policy = make_policy(
            "exact",
            PolicyEffect::Allow,
            &["graph:read"],
            vec![Orn::all_in_account("alice")],
        );

        assert_eq!(
            evaluate(&[policy.clone()], "graph:read", &Orn::node("alice", 1)),
            PolicyDecision::Allow,
        );
        assert_eq!(
            evaluate(&[policy], "graph:read_write", &Orn::node("alice", 1)),
            PolicyDecision::ImplicitDeny,
        );
    }

    #[test]
    fn tenant_scoped_policy() {
        let policy = make_policy(
            "tenant-scope",
            PolicyEffect::Allow,
            &["*"],
            vec![
                "orn:obrain:graph:alice:tenant/chess-kb".parse().unwrap(),
            ],
        );

        let chess = Orn::tenant("alice", "chess-kb");
        let other = Orn::tenant("alice", "other-kb");

        assert_eq!(evaluate(&[policy.clone()], "graph:read", &chess), PolicyDecision::Allow);
        assert_eq!(evaluate(&[policy], "graph:read", &other), PolicyDecision::ImplicitDeny);
    }
}
