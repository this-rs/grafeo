//! The [`MutationListener`] trait for consuming mutation events asynchronously.

use crate::event::MutationEvent;
use async_trait::async_trait;

/// A listener that reacts to graph mutations asynchronously.
///
/// Implementations should be lightweight — heavy processing should be spawned
/// into background tasks. The `on_batch` / `on_event` methods are called from
/// a dedicated scheduler task, never from the commit hot path.
///
/// # Example
///
/// ```rust,no_run
/// use grafeo_reactive::{MutationListener, MutationEvent};
/// use async_trait::async_trait;
///
/// struct EnergyListener;
///
/// #[async_trait]
/// impl MutationListener for EnergyListener {
///     fn name(&self) -> &str { "energy" }
///
///     async fn on_event(&self, event: &MutationEvent) {
///         // Boost energy for touched nodes/edges
///     }
/// }
/// ```
#[async_trait]
pub trait MutationListener: Send + Sync {
    /// Human-readable name for logging and metrics.
    fn name(&self) -> &str;

    /// Called for each individual event.
    ///
    /// Override this for event-level processing.
    async fn on_event(&self, event: &MutationEvent);

    /// Called with a batch of events accumulated by the scheduler.
    ///
    /// Default implementation delegates to [`on_event`](Self::on_event) for each
    /// event in the batch. Override for batch-level optimizations (e.g., deduplication).
    async fn on_batch(&self, events: &[MutationEvent]) {
        for event in events {
            self.on_event(event).await;
        }
    }

    /// Whether this listener is interested in a given event.
    ///
    /// Return `false` to skip processing. Default accepts all events.
    fn accepts(&self, _event: &MutationEvent) -> bool {
        true
    }
}
