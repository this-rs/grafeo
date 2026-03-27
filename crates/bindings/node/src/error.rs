//! Converts Rust errors to JavaScript exceptions.
//!
//! Type errors and invalid arguments become `InvalidArg` status errors,
//! while database, query, and transaction errors become `GenericFailure`.

use napi::Status;
use thiserror::Error;

/// Obrain errors that translate to JavaScript Error instances.
#[derive(Error, Debug)]
pub enum NodeObrainError {
    #[error("Database error: {0}")]
    Database(String),

    #[error("Query error: {0}")]
    Query(String),

    #[error("Type error: {0}")]
    Type(String),

    #[error("Transaction error: {0}")]
    Transaction(String),

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),
}

impl From<NodeObrainError> for napi::Error {
    fn from(err: NodeObrainError) -> Self {
        match &err {
            NodeObrainError::InvalidArgument(_) | NodeObrainError::Type(_) => {
                napi::Error::new(Status::InvalidArg, err.to_string())
            }
            NodeObrainError::Database(_)
            | NodeObrainError::Query(_)
            | NodeObrainError::Transaction(_) => {
                napi::Error::new(Status::GenericFailure, err.to_string())
            }
        }
    }
}

impl From<obrain_common::utils::error::Error> for NodeObrainError {
    fn from(err: obrain_common::utils::error::Error) -> Self {
        use obrain_bindings_common::error::{ErrorCategory, classify_error};
        let msg = err.to_string();
        match classify_error(&err) {
            ErrorCategory::Query => NodeObrainError::Query(msg),
            ErrorCategory::Transaction => NodeObrainError::Transaction(msg),
            _ => NodeObrainError::Database(msg),
        }
    }
}
