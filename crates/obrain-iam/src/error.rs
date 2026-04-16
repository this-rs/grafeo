//! IAM error types.

/// Errors that can occur in the IAM subsystem.
#[derive(Debug, thiserror::Error)]
pub enum IamError {
    /// Invalid ORN format.
    #[error("invalid ORN: {reason}")]
    InvalidOrn {
        /// Explanation of why the ORN is invalid.
        reason: String,
    },

    /// A resource was not found.
    #[error("resource not found: {resource}")]
    ResourceNotFound {
        /// The resource identifier that was not found.
        resource: String,
    },

    /// Access denied by policy evaluation.
    #[error("access denied: {reason}")]
    AccessDenied {
        /// Why access was denied.
        reason: String,
    },

    /// Permission denied — the caller lacks the required permission.
    #[error("permission denied: action={action} resource={resource}")]
    PermissionDenied {
        /// The action that was attempted.
        action: String,
        /// The resource the action was attempted on.
        resource: String,
    },

    /// A resource already exists.
    #[error("resource already exists: {resource}")]
    ResourceExists {
        /// The conflicting resource identifier.
        resource: String,
    },

    /// Invalid parameter.
    #[error("invalid parameter: {message}")]
    InvalidParameter {
        /// Description of the invalid parameter.
        message: String,
    },

    /// Serialization/deserialization error.
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Internal error — wraps an unexpected failure.
    #[error("internal error: {message}")]
    Internal {
        /// Description of what went wrong.
        message: String,
    },
}

/// Convenience type alias.
pub type IamResult<T> = Result<T, IamError>;
