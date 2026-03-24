//! Error types for the reactive substrate.

use tokio::sync::broadcast;

use crate::event::MutationBatch;

/// Errors that can occur in the reactive substrate.
#[derive(Debug, thiserror::Error)]
pub enum ReactiveError {
    /// The broadcast channel is full and the oldest message was dropped.
    /// This happens when subscribers are too slow to process events.
    #[error("bus capacity exceeded: {0} messages lagged")]
    BusCapacityExceeded(u64),

    /// A mutation listener returned an error during event processing.
    #[error("listener '{listener}' failed: {source}")]
    ListenerError {
        /// Name of the listener that failed.
        listener: String,
        /// The underlying error.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// Failed to send events on the broadcast channel.
    #[error("failed to send mutation batch: no active receivers")]
    SendError,
}

impl From<broadcast::error::SendError<MutationBatch>> for ReactiveError {
    fn from(_: broadcast::error::SendError<MutationBatch>) -> Self {
        Self::SendError
    }
}

impl From<broadcast::error::RecvError> for ReactiveError {
    fn from(_: broadcast::error::RecvError) -> Self {
        // RecvError means the channel was closed (all senders dropped)
        Self::SendError
    }
}

/// Result type for reactive operations.
pub type Result<T> = std::result::Result<T, ReactiveError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display() {
        let err = ReactiveError::BusCapacityExceeded(42);
        assert_eq!(err.to_string(), "bus capacity exceeded: 42 messages lagged");

        let err = ReactiveError::SendError;
        assert_eq!(
            err.to_string(),
            "failed to send mutation batch: no active receivers"
        );

        let err = ReactiveError::ListenerError {
            listener: "energy".to_string(),
            source: Box::new(std::io::Error::other("boom")),
        };
        assert!(err.to_string().contains("energy"));
    }

    #[test]
    fn error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ReactiveError>();
    }

    #[test]
    fn error_implements_std_error() {
        fn assert_error<T: std::error::Error>() {}
        assert_error::<ReactiveError>();
    }

    #[test]
    fn from_broadcast_send_error() {
        let batch = MutationBatch::new(vec![]);
        let send_err = broadcast::error::SendError(batch);
        let reactive_err: ReactiveError = send_err.into();
        assert!(matches!(reactive_err, ReactiveError::SendError));
    }

    #[test]
    fn result_type_works_with_question_mark() {
        fn fallible() -> super::Result<()> {
            let err = ReactiveError::BusCapacityExceeded(10);
            Err(err)?;
            Ok(())
        }
        assert!(fallible().is_err());
    }
}
