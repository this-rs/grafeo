//! Converts Rust errors to Python exceptions.
//!
//! Type errors and invalid arguments become `ValueError`, while database,
//! query, and transaction errors become `RuntimeError`.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use thiserror::Error;

/// Obrain errors that translate to Python exceptions.
#[derive(Error, Debug)]
pub enum PyObrainError {
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

impl From<PyObrainError> for PyErr {
    fn from(err: PyObrainError) -> Self {
        match err {
            PyObrainError::InvalidArgument(msg) | PyObrainError::Type(msg) => {
                PyValueError::new_err(msg)
            }
            PyObrainError::Database(msg)
            | PyObrainError::Query(msg)
            | PyObrainError::Transaction(msg) => PyRuntimeError::new_err(msg),
        }
    }
}

impl From<obrain_common::utils::error::Error> for PyObrainError {
    fn from(err: obrain_common::utils::error::Error) -> Self {
        use obrain_bindings_common::error::{ErrorCategory, classify_error};
        let msg = err.to_string();
        match classify_error(&err) {
            ErrorCategory::Query => PyObrainError::Query(msg),
            ErrorCategory::Transaction => PyObrainError::Transaction(msg),
            _ => PyObrainError::Database(msg),
        }
    }
}

/// Convenience type for functions that may fail with a Python-compatible error.
pub type PyObrainResult<T> = Result<T, PyObrainError>;
