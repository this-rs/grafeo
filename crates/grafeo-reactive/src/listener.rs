//! The [`MutationListener`] trait for consuming mutation events.

use crate::event::{MutationBatch, MutationEvent};

/// A listener that reacts to graph mutations.
///
/// Implementations should be lightweight — heavy processing should be spawned
/// into background tasks. The `on_batch` / `on_event` methods are called from
/// a dedicated scheduler task, never from the commit hot path.
///
/// # Example
///
/// ```rust,no_run
/// use grafeo_reactive::{MutationListener, MutationEvent, MutationBatch};
///
/// struct EnergyListener;
///
/// impl MutationListener for EnergyListener {
///     fn name(&self) -> &str { "energy" }
///
///     fn on_event(&self, event: &MutationEvent) {
///         // Boost energy for touched nodes/edges
///     }
/// }
/// ```
pub trait MutationListener: Send + Sync {
    /// Human-readable name for logging and metrics.
    fn name(&self) -> &str;

    /// Called for each individual event.
    ///
    /// Default implementation is a no-op. Override this for event-level processing.
    fn on_event(&self, _event: &MutationEvent) {}

    /// Called with a full batch from a single commit.
    ///
    /// Default implementation delegates to [`on_event`](Self::on_event) for each
    /// event in the batch. Override for batch-level optimizations (e.g., deduplication).
    fn on_batch(&self, batch: &MutationBatch) {
        for event in &batch.events {
            self.on_event(event);
        }
    }

    /// Whether this listener is interested in a given event.
    ///
    /// Return `false` to skip processing. Default accepts all events.
    fn accepts(&self, _event: &MutationEvent) -> bool {
        true
    }
}
