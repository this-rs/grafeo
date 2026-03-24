//! Error types for the cognitive subsystem.

use thiserror::Error;

/// Errors that can occur in the cognitive subsystem.
#[derive(Debug, Error)]
pub enum CognitiveError {
    /// Error in the energy subsystem.
    #[error("energy error: {0}")]
    Energy(String),

    /// Error in the synapse subsystem.
    #[error("synapse error: {0}")]
    Synapse(String),

    /// Error in the fabric/knowledge scoring subsystem.
    #[error("fabric error: {0}")]
    Fabric(String),

    /// Configuration error.
    #[error("config error: {0}")]
    Config(String),

    /// Store-level error (node not found, etc.).
    #[error("store error: {0}")]
    Store(String),

    /// Error propagated from the reactive substrate.
    #[error("reactive error: {0}")]
    Reactive(#[from] grafeo_reactive::ReactiveError),
}

/// Result type alias for cognitive operations.
pub type CognitiveResult<T> = Result<T, CognitiveError>;
