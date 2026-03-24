//! [`Scheduler`] — dispatches mutation events from the [`MutationBus`](crate::MutationBus)
//! to registered [`MutationListener`](crate::MutationListener)s with configurable batching.
//!
//! The scheduler runs in a dedicated `tokio::task` and accumulates events into batches
//! based on [`BatchConfig`]. This prevents overwhelming listeners during bulk imports
//! while still providing low-latency delivery for single mutations.

use crate::bus::MutationBus;
use crate::event::{MutationBatch, MutationEvent};
use crate::listener::MutationListener;
use parking_lot::RwLock;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::broadcast;
use tokio::task::JoinHandle;

/// Configuration for event batching in the [`Scheduler`].
///
/// The scheduler accumulates events into batches and flushes when either:
/// - The batch reaches `max_batch_size` events, or
/// - `max_delay` has elapsed since the first event in the current batch.
///
/// This provides a good balance between throughput (batching reduces per-event overhead)
/// and latency (the timeout ensures events are delivered promptly when traffic is low).
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum number of events per batch before forced flush.
    /// Default: 100
    pub max_batch_size: usize,

    /// Maximum delay before flushing an incomplete batch.
    /// Default: 50ms
    pub max_delay: Duration,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 100,
            max_delay: Duration::from_millis(50),
        }
    }
}

impl BatchConfig {
    /// Creates a new `BatchConfig` with the given parameters.
    pub fn new(max_batch_size: usize, max_delay: Duration) -> Self {
        Self {
            max_batch_size,
            max_delay,
        }
    }
}

/// Shared state for listeners, accessible from both the scheduler task and the
/// public API for dynamic registration.
type ListenerList = Arc<RwLock<Vec<Arc<dyn MutationListener>>>>;

/// The reactive scheduler that dispatches mutation events to registered listeners.
///
/// The scheduler subscribes to a [`MutationBus`], accumulates individual
/// [`MutationEvent`]s into batches (controlled by [`BatchConfig`]), and
/// dispatches each batch to all registered [`MutationListener`]s.
///
/// # Lifecycle
///
/// 1. Create with [`Scheduler::new`] — this spawns the background task.
/// 2. Register listeners with [`register_listener`](Scheduler::register_listener).
/// 3. Events flow automatically: `MutationBus` → `Scheduler` → `MutationListener`s.
/// 4. Call [`shutdown`](Scheduler::shutdown) to gracefully stop (drains pending events).
pub struct Scheduler {
    /// Registered listeners (shared with the background task).
    listeners: ListenerList,
    /// Handle to the background dispatch task.
    task_handle: Option<JoinHandle<()>>,
    /// Shutdown signal sender.
    shutdown_tx: tokio::sync::watch::Sender<bool>,
    /// Batch configuration (for introspection).
    config: BatchConfig,
}

impl Scheduler {
    /// Creates a new scheduler that subscribes to the given bus.
    ///
    /// This immediately spawns a background `tokio::task` that begins
    /// receiving events from the bus.
    pub fn new(bus: &MutationBus, config: BatchConfig) -> Self {
        let listeners: ListenerList = Arc::new(RwLock::new(Vec::new()));
        let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
        let rx = bus.subscribe();

        let task_listeners = Arc::clone(&listeners);
        let task_config = config.clone();

        // Only spawn the background task if a tokio runtime is available.
        // When running inside sync-only tests (e.g. grafeo-c), there is no
        // reactor and tokio::spawn would panic.
        let task_handle = tokio::runtime::Handle::try_current().ok().map(|handle| {
            handle.spawn(scheduler_loop(rx, task_listeners, task_config, shutdown_rx))
        });

        Self {
            listeners,
            task_handle,
            shutdown_tx,
            config,
        }
    }

    /// Registers a new listener that will receive future event batches.
    ///
    /// Listeners are called in registration order. This can be called at any
    /// time, even while the scheduler is running — the new listener will
    /// receive events starting from the next batch.
    pub fn register_listener(&self, listener: Arc<dyn MutationListener>) {
        tracing::info!(listener = listener.name(), "registering mutation listener");
        self.listeners.write().push(listener);
    }

    /// Returns the current number of registered listeners.
    pub fn listener_count(&self) -> usize {
        self.listeners.read().len()
    }

    /// Returns a reference to the batch configuration.
    pub fn config(&self) -> &BatchConfig {
        &self.config
    }

    /// Gracefully shuts down the scheduler.
    ///
    /// Signals the background task to stop, waits for it to drain any
    /// pending events, and then returns. After shutdown, the scheduler
    /// will no longer dispatch events.
    pub async fn shutdown(mut self) {
        tracing::info!("shutting down reactive scheduler");
        // Signal the task to stop
        let _ = self.shutdown_tx.send(true);
        // Wait for the task to finish draining
        if let Some(handle) = self.task_handle.take() {
            let _ = handle.await;
        }
    }
}

impl std::fmt::Debug for Scheduler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Scheduler")
            .field("config", &self.config)
            .field("listener_count", &self.listener_count())
            .field("running", &self.task_handle.is_some())
            .finish()
    }
}

/// The main loop that runs inside the spawned tokio task.
///
/// Receives `MutationBatch`es from the broadcast channel, flattens them into
/// individual events, accumulates into batches per `BatchConfig`, and dispatches
/// to all registered listeners.
async fn scheduler_loop(
    mut rx: broadcast::Receiver<MutationBatch>,
    listeners: ListenerList,
    config: BatchConfig,
    mut shutdown_rx: tokio::sync::watch::Receiver<bool>,
) {
    let mut buffer: Vec<MutationEvent> = Vec::with_capacity(config.max_batch_size);
    let mut flush_deadline: Option<tokio::time::Instant> = None;

    tracing::info!(
        max_batch_size = config.max_batch_size,
        max_delay_ms = config.max_delay.as_millis(),
        "scheduler loop started"
    );

    loop {
        let timeout = flush_deadline.map_or(Duration::from_secs(3600), |d| {
            d.saturating_duration_since(tokio::time::Instant::now())
        }); // effectively infinite when no pending events

        tokio::select! {
            biased;

            // Check for shutdown signal.
            // `changed()` returns Err when the Sender is dropped (Scheduler dropped
            // without explicit shutdown). In both cases we drain and exit.
            result = shutdown_rx.changed() => {
                if result.is_err() || *shutdown_rx.borrow() {
                    // Drain remaining events from the channel
                    while let Ok(batch) = rx.try_recv() {
                        for event in batch.events {
                            buffer.push(event);
                        }
                    }
                    // Flush any remaining buffer
                    if !buffer.is_empty() {
                        dispatch_batch(&buffer, &listeners).await;
                        buffer.clear();
                    }
                    tracing::info!("scheduler loop shutting down");
                    break;
                }
            }

            // Receive a batch from the bus
            result = rx.recv() => {
                match result {
                    Ok(batch) => {
                        for event in batch.events {
                            buffer.push(event);
                        }

                        // Set flush deadline on first event in buffer
                        if flush_deadline.is_none() && !buffer.is_empty() {
                            flush_deadline = Some(tokio::time::Instant::now() + config.max_delay);
                        }

                        // Flush if buffer is full
                        while buffer.len() >= config.max_batch_size {
                            let drain_end = config.max_batch_size.min(buffer.len());
                            let batch: Vec<MutationEvent> = buffer.drain(..drain_end).collect();
                            dispatch_batch(&batch, &listeners).await;
                        }

                        // Reset deadline if buffer is now empty
                        if buffer.is_empty() {
                            flush_deadline = None;
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        tracing::warn!(
                            lagged = n,
                            "scheduler lagged behind bus, some events were dropped"
                        );
                        // Continue receiving — we'll get the next available message
                    }
                    Err(broadcast::error::RecvError::Closed) => {
                        tracing::info!("bus closed, flushing remaining events");
                        if !buffer.is_empty() {
                            dispatch_batch(&buffer, &listeners).await;
                            buffer.clear();
                        }
                        break;
                    }
                }
            }

            // Timeout: flush incomplete batch
            () = tokio::time::sleep(timeout), if flush_deadline.is_some() => {
                if !buffer.is_empty() {
                    dispatch_batch(&buffer, &listeners).await;
                    buffer.clear();
                }
                flush_deadline = None;
            }
        }
    }
}

/// Dispatches a batch of events to all registered listeners.
async fn dispatch_batch(events: &[MutationEvent], listeners: &ListenerList) {
    if events.is_empty() {
        return;
    }

    // Snapshot the listener list to avoid holding the lock during async dispatch
    let listener_snapshot: Vec<Arc<dyn MutationListener>> = listeners.read().clone();

    if listener_snapshot.is_empty() {
        return;
    }

    tracing::trace!(
        event_count = events.len(),
        listener_count = listener_snapshot.len(),
        "dispatching batch to listeners"
    );

    for listener in &listener_snapshot {
        // Filter events this listener accepts
        let accepted: Vec<&MutationEvent> = events.iter().filter(|e| listener.accepts(e)).collect();

        if accepted.is_empty() {
            continue;
        }

        // Build a contiguous slice for on_batch — we need owned references
        // Since on_batch takes &[MutationEvent], pass the full events if all are accepted
        if accepted.len() == events.len() {
            // All accepted — pass the original slice directly
            listener.on_batch(events).await;
        } else {
            // Partial acceptance — collect accepted events
            let filtered: Vec<MutationEvent> = accepted.into_iter().cloned().collect();
            listener.on_batch(&filtered).await;
        }
    }
}
