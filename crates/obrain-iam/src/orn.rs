//! # Obrain Resource Names (ORN)
//!
//! ORN is the resource identification system for Obrain, inspired by AWS ARN
//! and WAMI ARN but designed specifically for graph database resources.
//!
//! ## Format
//!
//! ```text
//! orn:obrain:{service}:{account_id}:{resource_type}/{resource_path}
//! ```
//!
//! ## Components
//!
//! | Component       | Description                        | Examples                  |
//! |-----------------|------------------------------------|---------------------------|
//! | `service`       | Obrain subsystem                   | `graph`, `iam`, `cognitive` |
//! | `account_id`    | Tenant / account identifier        | `alice`, `org-42`, `*`    |
//! | `resource_type` | Type of resource                   | `node`, `edge`, `tenant`, `user`, `role` |
//! | `resource_path` | Resource path (may contain `/`)    | `12345`, `chess-kb`, `admin/*` |
//!
//! ## Wildcards
//!
//! - `*` matches any single segment (no `/`)
//! - `**` matches zero or more segments (including `/`)
//!
//! ## Variable Substitution
//!
//! - `${tenant}` is replaced with the current tenant ID at evaluation time
//! - `${principal}` is replaced with the current principal ID
//!
//! ## Examples
//!
//! ```
//! use obrain_iam::Orn;
//!
//! // Parse
//! let orn: Orn = "orn:obrain:graph:alice:node/12345".parse().unwrap();
//! assert_eq!(orn.service(), "graph");
//! assert_eq!(orn.account_id(), "alice");
//! assert_eq!(orn.resource_type(), "node");
//! assert_eq!(orn.resource_path(), "12345");
//!
//! // Wildcard matching
//! let pattern: Orn = "orn:obrain:graph:alice:node/*".parse().unwrap();
//! assert!(pattern.matches(&orn));
//!
//! // Account wildcard
//! let any_account: Orn = "orn:obrain:graph:*:node/*".parse().unwrap();
//! assert!(any_account.matches(&orn));
//!
//! // Variable substitution
//! let template: Orn = "orn:obrain:graph:${tenant}:node/*".parse().unwrap();
//! let mut vars = std::collections::HashMap::new();
//! vars.insert("tenant".to_string(), "alice".to_string());
//! let resolved = template.substitute(&vars);
//! assert!(resolved.matches(&orn));
//! ```

use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::error::IamError;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// The fixed prefix for all ORNs.
const ORN_PREFIX: &str = "orn";
/// The fixed partition for Obrain ORNs.
const ORN_PARTITION: &str = "obrain";
/// Minimum number of colon-separated segments in a valid ORN.
const ORN_MIN_SEGMENTS: usize = 5;

// ---------------------------------------------------------------------------
// Orn
// ---------------------------------------------------------------------------

/// An Obrain Resource Name — a structured identifier for any resource in the
/// graph database.
///
/// Format: `orn:obrain:{service}:{account_id}:{resource_type}/{resource_path}`
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Orn {
    service: String,
    account_id: String,
    resource_type: String,
    resource_path: String,
}

impl Orn {
    /// Creates a new ORN from individual components.
    ///
    /// # Errors
    ///
    /// Returns [`IamError::InvalidOrn`] if any component is empty.
    pub fn new(
        service: impl Into<String>,
        account_id: impl Into<String>,
        resource_type: impl Into<String>,
        resource_path: impl Into<String>,
    ) -> Result<Self, IamError> {
        let service = service.into();
        let account_id = account_id.into();
        let resource_type = resource_type.into();
        let resource_path = resource_path.into();

        if service.is_empty() {
            return Err(IamError::InvalidOrn {
                reason: "service cannot be empty".into(),
            });
        }
        if account_id.is_empty() {
            return Err(IamError::InvalidOrn {
                reason: "account_id cannot be empty".into(),
            });
        }
        if resource_type.is_empty() {
            return Err(IamError::InvalidOrn {
                reason: "resource_type cannot be empty".into(),
            });
        }
        if resource_path.is_empty() {
            return Err(IamError::InvalidOrn {
                reason: "resource_path cannot be empty".into(),
            });
        }

        Ok(Self {
            service,
            account_id,
            resource_type,
            resource_path,
        })
    }

    // -- Accessors -----------------------------------------------------------

    /// The service component (e.g. `graph`, `iam`, `cognitive`).
    #[inline]
    pub fn service(&self) -> &str {
        &self.service
    }

    /// The account / tenant identifier.
    #[inline]
    pub fn account_id(&self) -> &str {
        &self.account_id
    }

    /// The resource type (e.g. `node`, `edge`, `user`).
    #[inline]
    pub fn resource_type(&self) -> &str {
        &self.resource_type
    }

    /// The resource path (may contain `/` for hierarchical resources).
    #[inline]
    pub fn resource_path(&self) -> &str {
        &self.resource_path
    }

    // -- Matching ------------------------------------------------------------

    /// Tests whether this ORN (as a pattern) matches a concrete ORN.
    ///
    /// Supports:
    /// - `*` matches any single segment (no `/`)
    /// - `**` matches zero or more segments (including `/`)
    /// - `${variable}` patterns match literally (use [`substitute`] first)
    ///
    /// [`substitute`]: Orn::substitute
    pub fn matches(&self, other: &Orn) -> bool {
        segment_matches(&self.service, &other.service)
            && segment_matches(&self.account_id, &other.account_id)
            && segment_matches(&self.resource_type, &other.resource_type)
            && path_matches(&self.resource_path, &other.resource_path)
    }

    // -- Variable Substitution -----------------------------------------------

    /// Replaces `${variable}` placeholders with values from the provided map.
    ///
    /// Unknown variables are left as-is.
    pub fn substitute(&self, vars: &HashMap<String, String>) -> Orn {
        Orn {
            service: substitute_vars(&self.service, vars),
            account_id: substitute_vars(&self.account_id, vars),
            resource_type: substitute_vars(&self.resource_type, vars),
            resource_path: substitute_vars(&self.resource_path, vars),
        }
    }

    /// Returns `true` if any component contains a `${…}` placeholder.
    pub fn has_variables(&self) -> bool {
        has_var(&self.service)
            || has_var(&self.account_id)
            || has_var(&self.resource_type)
            || has_var(&self.resource_path)
    }

    // -- Conversion ----------------------------------------------------------

    /// Converts this ORN to its canonical string representation.
    pub fn to_string_canonical(&self) -> String {
        format!(
            "{}:{}:{}:{}:{}/{}",
            ORN_PREFIX,
            ORN_PARTITION,
            self.service,
            self.account_id,
            self.resource_type,
            self.resource_path,
        )
    }

    // -- Builder convenience -------------------------------------------------

    /// Creates an ORN for a graph node.
    pub fn node(account_id: impl Into<String>, node_id: impl fmt::Display) -> Self {
        Self {
            service: "graph".into(),
            account_id: account_id.into(),
            resource_type: "node".into(),
            resource_path: node_id.to_string(),
        }
    }

    /// Creates an ORN for a graph edge.
    pub fn edge(account_id: impl Into<String>, edge_id: impl fmt::Display) -> Self {
        Self {
            service: "graph".into(),
            account_id: account_id.into(),
            resource_type: "edge".into(),
            resource_path: edge_id.to_string(),
        }
    }

    /// Creates an ORN for a named graph (tenant).
    pub fn tenant(account_id: impl Into<String>, graph_name: impl Into<String>) -> Self {
        Self {
            service: "graph".into(),
            account_id: account_id.into(),
            resource_type: "tenant".into(),
            resource_path: graph_name.into(),
        }
    }

    /// Creates an ORN for an IAM user.
    pub fn user(account_id: impl Into<String>, username: impl Into<String>) -> Self {
        Self {
            service: "iam".into(),
            account_id: account_id.into(),
            resource_type: "user".into(),
            resource_path: username.into(),
        }
    }

    /// Creates an ORN for an IAM role.
    pub fn role(account_id: impl Into<String>, role_name: impl Into<String>) -> Self {
        Self {
            service: "iam".into(),
            account_id: account_id.into(),
            resource_type: "role".into(),
            resource_path: role_name.into(),
        }
    }

    /// Creates an ORN for an IAM policy.
    pub fn policy(account_id: impl Into<String>, policy_name: impl Into<String>) -> Self {
        Self {
            service: "iam".into(),
            account_id: account_id.into(),
            resource_type: "policy".into(),
            resource_path: policy_name.into(),
        }
    }

    /// Creates an ORN for a cognitive resource (engram, synapse, etc.).
    pub fn cognitive(
        account_id: impl Into<String>,
        resource_type: impl Into<String>,
        resource_path: impl Into<String>,
    ) -> Self {
        Self {
            service: "cognitive".into(),
            account_id: account_id.into(),
            resource_type: resource_type.into(),
            resource_path: resource_path.into(),
        }
    }

    /// Creates a wildcard ORN pattern matching all resources of a given type
    /// for a specific account.
    pub fn all_of_type(
        service: impl Into<String>,
        account_id: impl Into<String>,
        resource_type: impl Into<String>,
    ) -> Self {
        Self {
            service: service.into(),
            account_id: account_id.into(),
            resource_type: resource_type.into(),
            resource_path: "*".into(),
        }
    }

    /// Creates a wildcard ORN matching everything in an account.
    pub fn all_in_account(account_id: impl Into<String>) -> Self {
        Self {
            service: "*".into(),
            account_id: account_id.into(),
            resource_type: "*".into(),
            resource_path: "**".into(),
        }
    }
}

// ---------------------------------------------------------------------------
// Display / FromStr
// ---------------------------------------------------------------------------

impl fmt::Display for Orn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}:{}:{}:{}:{}/{}",
            ORN_PREFIX,
            ORN_PARTITION,
            self.service,
            self.account_id,
            self.resource_type,
            self.resource_path,
        )
    }
}

impl FromStr for Orn {
    type Err = IamError;

    /// Parses an ORN string.
    ///
    /// Expected format: `orn:obrain:{service}:{account_id}:{resource_type}/{resource_path}`
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Split on `:` — we expect at least 5 segments:
        //   [0] = "orn"
        //   [1] = "obrain"
        //   [2] = service
        //   [3] = account_id
        //   [4] = resource_type/resource_path
        let segments: Vec<&str> = s.splitn(ORN_MIN_SEGMENTS, ':').collect();

        if segments.len() < ORN_MIN_SEGMENTS {
            return Err(IamError::InvalidOrn {
                reason: format!(
                    "expected at least {} colon-separated segments, got {}",
                    ORN_MIN_SEGMENTS,
                    segments.len()
                ),
            });
        }

        if segments[0] != ORN_PREFIX {
            return Err(IamError::InvalidOrn {
                reason: format!("expected prefix '{}', got '{}'", ORN_PREFIX, segments[0]),
            });
        }

        if segments[1] != ORN_PARTITION {
            return Err(IamError::InvalidOrn {
                reason: format!(
                    "expected partition '{}', got '{}'",
                    ORN_PARTITION, segments[1]
                ),
            });
        }

        let service = segments[2];
        let account_id = segments[3];
        let resource_part = segments[4]; // "resource_type/resource_path"

        // Split resource_part on first `/`
        let (resource_type, resource_path) =
            resource_part
                .split_once('/')
                .ok_or_else(|| IamError::InvalidOrn {
                    reason: format!(
                        "resource segment must contain '/' separator: '{resource_part}'"
                    ),
                })?;

        Orn::new(service, account_id, resource_type, resource_path)
    }
}

// ---------------------------------------------------------------------------
// Matching helpers
// ---------------------------------------------------------------------------

/// Matches a single segment (no path separators). `*` matches anything.
fn segment_matches(pattern: &str, value: &str) -> bool {
    pattern == "*" || pattern == value
}

/// Matches a resource path supporting `*` (single segment) and `**` (any depth).
fn path_matches(pattern: &str, value: &str) -> bool {
    if pattern == "**" {
        return true;
    }
    if pattern == "*" {
        // `*` matches a single segment — no `/` in value
        return !value.contains('/');
    }
    if !pattern.contains('*') {
        return pattern == value;
    }

    // Split into segments and match recursively
    let pattern_parts: Vec<&str> = pattern.split('/').collect();
    let value_parts: Vec<&str> = value.split('/').collect();

    path_parts_match(&pattern_parts, &value_parts)
}

/// Recursive segment-by-segment matching.
fn path_parts_match(pattern: &[&str], value: &[&str]) -> bool {
    match (pattern.first(), value.first()) {
        // Both exhausted → match
        (None, None) => true,

        // Pattern has `**` → try consuming 0..N segments of value
        (Some(&"**"), _) => {
            let rest_pattern = &pattern[1..];
            // Try matching `**` against 0, 1, 2, … segments
            for skip in 0..=value.len() {
                if path_parts_match(rest_pattern, &value[skip..]) {
                    return true;
                }
            }
            false
        }

        // Pattern has `*` → matches one segment
        (Some(&"*"), Some(_)) => path_parts_match(&pattern[1..], &value[1..]),

        // Literal match
        (Some(p), Some(v)) => {
            if *p == *v {
                path_parts_match(&pattern[1..], &value[1..])
            } else {
                false
            }
        }

        // One side exhausted, other not → no match
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Variable substitution helpers
// ---------------------------------------------------------------------------

/// Replaces `${var}` patterns in a string.
fn substitute_vars(s: &str, vars: &HashMap<String, String>) -> String {
    let mut result = s.to_string();
    for (key, value) in vars {
        let pattern = format!("${{{key}}}");
        result = result.replace(&pattern, value);
    }
    result
}

/// Returns `true` if the string contains a `${…}` placeholder.
fn has_var(s: &str) -> bool {
    s.contains("${")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_roundtrip() {
        let input = "orn:obrain:graph:alice:node/12345";
        let orn: Orn = input.parse().unwrap();
        assert_eq!(orn.service(), "graph");
        assert_eq!(orn.account_id(), "alice");
        assert_eq!(orn.resource_type(), "node");
        assert_eq!(orn.resource_path(), "12345");
        assert_eq!(orn.to_string(), input);
    }

    #[test]
    fn parse_with_path_segments() {
        let input = "orn:obrain:graph:alice:tenant/chess-kb";
        let orn: Orn = input.parse().unwrap();
        assert_eq!(orn.resource_type(), "tenant");
        assert_eq!(orn.resource_path(), "chess-kb");
    }

    #[test]
    fn parse_with_deep_path() {
        let input = "orn:obrain:cognitive:org-42:engram/layer/3/12345";
        let orn: Orn = input.parse().unwrap();
        assert_eq!(orn.service(), "cognitive");
        assert_eq!(orn.account_id(), "org-42");
        assert_eq!(orn.resource_type(), "engram");
        assert_eq!(orn.resource_path(), "layer/3/12345");
    }

    #[test]
    fn parse_error_missing_segments() {
        let result = "orn:obrain:graph".parse::<Orn>();
        assert!(result.is_err());
    }

    #[test]
    fn parse_error_wrong_prefix() {
        let result = "arn:obrain:graph:alice:node/1".parse::<Orn>();
        assert!(result.is_err());
    }

    #[test]
    fn parse_error_wrong_partition() {
        let result = "orn:aws:graph:alice:node/1".parse::<Orn>();
        assert!(result.is_err());
    }

    #[test]
    fn parse_error_no_slash_in_resource() {
        let result = "orn:obrain:graph:alice:node".parse::<Orn>();
        assert!(result.is_err());
    }

    #[test]
    fn display_roundtrip() {
        let orn = Orn::node("alice", 42);
        let s = orn.to_string();
        let parsed: Orn = s.parse().unwrap();
        assert_eq!(orn, parsed);
    }

    // -- Wildcard matching ---------------------------------------------------

    #[test]
    fn wildcard_star_matches_single_segment() {
        let pattern: Orn = "orn:obrain:graph:alice:node/*".parse().unwrap();
        let target: Orn = "orn:obrain:graph:alice:node/12345".parse().unwrap();
        assert!(pattern.matches(&target));
    }

    #[test]
    fn wildcard_star_does_not_match_deep_path() {
        let pattern: Orn = "orn:obrain:graph:alice:node/*".parse().unwrap();
        let target: Orn = "orn:obrain:graph:alice:node/a/b".parse().unwrap();
        assert!(!pattern.matches(&target));
    }

    #[test]
    fn wildcard_double_star_matches_any_depth() {
        let pattern: Orn = "orn:obrain:graph:alice:node/**".parse().unwrap();
        let target: Orn = "orn:obrain:graph:alice:node/a/b/c".parse().unwrap();
        assert!(pattern.matches(&target));
    }

    #[test]
    fn wildcard_double_star_matches_single() {
        let pattern: Orn = "orn:obrain:graph:alice:node/**".parse().unwrap();
        let target: Orn = "orn:obrain:graph:alice:node/12345".parse().unwrap();
        assert!(pattern.matches(&target));
    }

    #[test]
    fn wildcard_account_star() {
        let pattern: Orn = "orn:obrain:graph:*:node/12345".parse().unwrap();
        let target: Orn = "orn:obrain:graph:alice:node/12345".parse().unwrap();
        assert!(pattern.matches(&target));
    }

    #[test]
    fn wildcard_service_star() {
        let pattern: Orn = "orn:obrain:*:alice:node/12345".parse().unwrap();
        let target: Orn = "orn:obrain:graph:alice:node/12345".parse().unwrap();
        assert!(pattern.matches(&target));
    }

    #[test]
    fn exact_mismatch() {
        let pattern: Orn = "orn:obrain:graph:alice:node/12345".parse().unwrap();
        let target: Orn = "orn:obrain:graph:alice:node/99999".parse().unwrap();
        assert!(!pattern.matches(&target));
    }

    #[test]
    fn all_in_account_matches_everything() {
        let pattern = Orn::all_in_account("alice");
        let n = Orn::node("alice", 1);
        let e = Orn::edge("alice", 2);
        let u = Orn::user("alice", "admin");
        assert!(pattern.matches(&n));
        assert!(pattern.matches(&e));
        assert!(pattern.matches(&u));
    }

    #[test]
    fn all_in_account_does_not_match_other_account() {
        let pattern = Orn::all_in_account("alice");
        let target = Orn::node("bob", 1);
        assert!(!pattern.matches(&target));
    }

    #[test]
    fn all_of_type_matches() {
        let pattern = Orn::all_of_type("graph", "alice", "node");
        let target = Orn::node("alice", 42);
        assert!(pattern.matches(&target));
    }

    #[test]
    fn mixed_path_wildcard() {
        let pattern: Orn = "orn:obrain:cognitive:alice:engram/layer/*/data"
            .parse()
            .unwrap();
        let target: Orn = "orn:obrain:cognitive:alice:engram/layer/3/data"
            .parse()
            .unwrap();
        assert!(pattern.matches(&target));

        let no_match: Orn = "orn:obrain:cognitive:alice:engram/layer/3/other"
            .parse()
            .unwrap();
        assert!(!pattern.matches(&no_match));
    }

    // -- Variable substitution -----------------------------------------------

    #[test]
    fn substitute_tenant() {
        let template: Orn = "orn:obrain:graph:${tenant}:node/*".parse().unwrap();
        assert!(template.has_variables());

        let mut vars = HashMap::new();
        vars.insert("tenant".to_string(), "alice".to_string());
        let resolved = template.substitute(&vars);

        assert!(!resolved.has_variables());
        assert_eq!(resolved.account_id(), "alice");

        let target = Orn::node("alice", 42);
        assert!(resolved.matches(&target));
    }

    #[test]
    fn substitute_multiple_vars() {
        let template: Orn = "orn:obrain:${service}:${tenant}:${type}/*"
            .parse()
            .unwrap();

        let mut vars = HashMap::new();
        vars.insert("service".to_string(), "graph".to_string());
        vars.insert("tenant".to_string(), "bob".to_string());
        vars.insert("type".to_string(), "node".to_string());
        let resolved = template.substitute(&vars);

        assert_eq!(resolved.service(), "graph");
        assert_eq!(resolved.account_id(), "bob");
        assert_eq!(resolved.resource_type(), "node");
    }

    #[test]
    fn substitute_unknown_var_kept() {
        let template: Orn = "orn:obrain:graph:${unknown}:node/*".parse().unwrap();
        let resolved = template.substitute(&HashMap::new());
        assert_eq!(resolved.account_id(), "${unknown}");
        assert!(resolved.has_variables());
    }

    // -- Builder convenience -------------------------------------------------

    #[test]
    fn builder_node() {
        let orn = Orn::node("alice", 42);
        assert_eq!(orn.to_string(), "orn:obrain:graph:alice:node/42");
    }

    #[test]
    fn builder_edge() {
        let orn = Orn::edge("alice", 7);
        assert_eq!(orn.to_string(), "orn:obrain:graph:alice:edge/7");
    }

    #[test]
    fn builder_tenant() {
        let orn = Orn::tenant("alice", "chess-kb");
        assert_eq!(orn.to_string(), "orn:obrain:graph:alice:tenant/chess-kb");
    }

    #[test]
    fn builder_user() {
        let orn = Orn::user("org-42", "admin");
        assert_eq!(orn.to_string(), "orn:obrain:iam:org-42:user/admin");
    }

    #[test]
    fn builder_role() {
        let orn = Orn::role("org-42", "db-admin");
        assert_eq!(orn.to_string(), "orn:obrain:iam:org-42:role/db-admin");
    }

    #[test]
    fn builder_policy() {
        let orn = Orn::policy("org-42", "read-only");
        assert_eq!(orn.to_string(), "orn:obrain:iam:org-42:policy/read-only");
    }

    #[test]
    fn builder_cognitive() {
        let orn = Orn::cognitive("alice", "engram", "42");
        assert_eq!(
            orn.to_string(),
            "orn:obrain:cognitive:alice:engram/42"
        );
    }

    // -- Serde roundtrip -----------------------------------------------------

    #[test]
    fn serde_roundtrip() {
        let orn = Orn::node("alice", 42);
        let json = serde_json::to_string(&orn).unwrap();
        let parsed: Orn = serde_json::from_str(&json).unwrap();
        assert_eq!(orn, parsed);
    }
}
