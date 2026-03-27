//! Error types for grafeo-rag.

/// Errors that can occur during RAG operations.
#[derive(Debug, thiserror::Error)]
pub enum RagError {
    /// No engrams found for the given query.
    #[error("no engrams found for query: {0}")]
    NoEngramsFound(String),

    /// The cognitive engine is not available or not configured.
    #[error("cognitive engine unavailable: {0}")]
    CognitiveUnavailable(String),

    /// Graph store error during node/property extraction.
    #[error("graph store error: {0}")]
    GraphStore(String),

    /// Token budget exhausted — no content fits within the budget.
    #[error("token budget exhausted (budget={budget}, min_required={min_required})")]
    BudgetExhausted {
        /// The configured token budget.
        budget: usize,
        /// The minimum tokens required for the smallest node.
        min_required: usize,
    },

    /// Invalid configuration parameter.
    #[error("config error: {0}")]
    Config(String),

    /// Internal error.
    #[error("internal error: {0}")]
    Internal(String),
}

/// Result type alias for RAG operations.
pub type RagResult<T> = Result<T, RagError>;
