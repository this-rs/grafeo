//! The [`MutationListener`] trait for consuming mutation events asynchronously.

use crate::event::{EventContext, MutationEvent};
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
/// use obrain_reactive::{MutationListener, MutationEvent};
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

    /// Returns whether this listener accepts events with the given context.
    ///
    /// Default implementation accepts all events regardless of context.
    /// Override to implement tenant-level filtering.
    fn accepts_context(&self, _context: Option<&EventContext>) -> bool {
        true
    }
}

/// A wrapper that filters mutations by tenant.
///
/// Only passes through mutations whose [`EventContext::tenant_id`]
/// matches the configured tenant. Events without context are passed
/// through (bootstrap mode).
pub struct TenantFilteredListener<L: MutationListener> {
    inner: L,
    tenant_id: String,
}

impl<L: MutationListener> TenantFilteredListener<L> {
    /// Creates a new `TenantFilteredListener` that only passes events
    /// matching the given tenant ID to the inner listener.
    pub fn new(inner: L, tenant_id: impl Into<String>) -> Self {
        Self {
            inner,
            tenant_id: tenant_id.into(),
        }
    }
}

#[async_trait]
impl<L: MutationListener> MutationListener for TenantFilteredListener<L> {
    fn name(&self) -> &str {
        self.inner.name()
    }

    async fn on_event(&self, event: &MutationEvent) {
        self.inner.on_event(event).await;
    }

    async fn on_batch(&self, events: &[MutationEvent]) {
        self.inner.on_batch(events).await;
    }

    fn accepts(&self, event: &MutationEvent) -> bool {
        self.inner.accepts(event)
    }

    fn accepts_context(&self, context: Option<&EventContext>) -> bool {
        match context {
            // No context (bootstrap mode) — pass through
            None => true,
            Some(ctx) => match &ctx.tenant_id {
                // No tenant in context — pass through
                None => true,
                // Tenant must match
                Some(tid) => *tid == self.tenant_id,
            },
        }
    }
}
