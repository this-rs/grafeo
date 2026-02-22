//! Language-agnostic error classification for bindings.
//!
//! Each binding maps [`ErrorCategory`] to its language-specific exception type
//! (Python `PyErr`, Node.js `napi::Error`, C `GrafeoStatus`, etc.) using a
//! single small match expression.

use grafeo_common::utils::error::Error;

/// Categories that all bindings map errors into.
///
/// These mirror the natural groupings in [`grafeo_common::utils::error::Error`]
/// and match what every binding was already doing independently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    /// Query parsing, semantic, or execution error.
    Query,
    /// Transaction conflict, timeout, or invalid state.
    Transaction,
    /// Storage-layer error (disk, memory limit).
    Storage,
    /// I/O error (file, network).
    Io,
    /// Serialization/deserialization failure.
    Serialization,
    /// Internal error (should not happen in normal operation).
    Internal,
    /// Catch-all for other database errors (not found, type mismatch, etc.).
    Database,
}

/// Classifies a Grafeo error into a binding-agnostic category.
#[must_use]
pub fn classify_error(err: &Error) -> ErrorCategory {
    match err {
        Error::Query(_) => ErrorCategory::Query,
        Error::Transaction(_) => ErrorCategory::Transaction,
        Error::Storage(_) => ErrorCategory::Storage,
        Error::Io(_) => ErrorCategory::Io,
        Error::Serialization(_) => ErrorCategory::Serialization,
        Error::Internal(_) => ErrorCategory::Internal,
        _ => ErrorCategory::Database,
    }
}

/// Returns the human-readable message for a Grafeo error.
#[must_use]
pub fn error_message(err: &Error) -> String {
    err.to_string()
}

#[cfg(test)]
mod tests {
    use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind};

    use super::*;

    #[test]
    fn classifies_query_error() {
        let err = Error::Query(QueryError::new(QueryErrorKind::Syntax, "bad syntax"));
        assert_eq!(classify_error(&err), ErrorCategory::Query);
    }

    #[test]
    fn classifies_not_found_as_database() {
        let err = Error::NodeNotFound(grafeo_common::types::NodeId(42));
        assert_eq!(classify_error(&err), ErrorCategory::Database);
    }

    #[test]
    fn classifies_internal() {
        let err = Error::Internal("oops".into());
        assert_eq!(classify_error(&err), ErrorCategory::Internal);
    }
}
