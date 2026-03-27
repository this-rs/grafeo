//! [`Scheduler`] — dispatches mutation events from the [`MutationBus`](crate::MutationBus)
//! to registered [`MutationListener`](crate::MutationListener)s with configurable batching.
//!
//! On native targets, the scheduler runs in a dedicated `tokio::task` and accumulates
//! events into batches based on [`BatchConfig`]. On WASM targets, the scheduler uses
//! a synchronous dispatch model (no background task, events are dispatched inline).
//!
//! The background task is **lazily spawned** — it only starts when the first listener
//! is registered via [`Scheduler::register_listener`]. This makes the scheduler
//! zero-cost when no listeners are present (e.g. in benchmarks or tests that don't
//! need reactive features).

use crate::bus::MutationBus;
#[cfg(not(target_arch = "wasm32"))]
use crate::event::MutationEvent;
use crate::listener::MutationListener;
use parking_lot::RwLock;
use std::sync::Arc;
use std::time::Duration;

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

// =============================================================================
// Native (non-WASM) scheduler — uses tokio tasks
// =============================================================================
#[cfg(not(target_arch = "wasm32"))]
mod native {
    use super::{
        Arc, BatchConfig, Duration, ListenerList, MutationBus, MutationEvent, MutationListener,
        RwLock,
    };
    use crate::event::MutationBatch;
    use parking_lot::Mutex;
    use tokio::sync::broadcast;
    use tokio::task::JoinHandle;

    /// The reactive scheduler that dispatches mutation events to registered listeners.
    ///
    /// The scheduler subscribes to a [`MutationBus`], accumulates individual
    /// [`MutationEvent`]s into batches (controlled by [`BatchConfig`]), and
    /// dispatches each batch to all registered [`MutationListener`]s.
    ///
    /// # Lifecycle
    ///
    /// 1. Create with [`Scheduler::new`] — lightweight, no background task yet.
    /// 2. Register listeners with [`register_listener`](Scheduler::register_listener)
    ///    — the background task is lazily spawned on the first registration.
    /// 3. Events flow automatically: `MutationBus` → `Scheduler` → `MutationListener`s.
    /// 4. Call [`shutdown`](Scheduler::shutdown) to gracefully stop (drains pending events).
    ///
    /// If no listener is ever registered, the scheduler is truly zero-cost: no tokio task
    /// is spawned, no bus subscription is created, and no events are buffered.
    pub struct Scheduler {
        /// Registered listeners (shared with the background task).
        listeners: ListenerList,
        /// Handle to the background dispatch task (lazily spawned).
        task_handle: Mutex<Option<JoinHandle<()>>>,
        /// Shutdown signal sender.
        shutdown_tx: tokio::sync::watch::Sender<bool>,
        /// Batch configuration (for introspection).
        config: BatchConfig,
        /// Reference to the bus for deferred subscription.
        bus: MutationBus,
    }

    impl Scheduler {
        /// Creates a new scheduler associated with the given bus.
        ///
        /// This is lightweight — no background task is spawned until the first
        /// listener is registered via [`register_listener`](Self::register_listener).
        pub fn new(bus: &MutationBus, config: BatchConfig) -> Self {
            let listeners: ListenerList = Arc::new(RwLock::new(Vec::new()));
            let (shutdown_tx, _shutdown_rx) = tokio::sync::watch::channel(false);

            Self {
                listeners,
                task_handle: Mutex::new(None),
                shutdown_tx,
                config,
                bus: bus.clone(),
            }
        }

        /// Ensures the background task is running. Called on first listener registration.
        ///
        /// Subscribes to the bus and spawns the scheduler loop. If no tokio runtime
        /// is available (e.g. sync-only tests), the task is not spawned and listeners
        /// will not receive events.
        fn ensure_running(&self) {
            let mut handle = self.task_handle.lock();
            if handle.is_some() {
                return; // Already running
            }

            let rx = self.bus.subscribe();
            let shutdown_rx = self.shutdown_tx.subscribe();
            let task_listeners = Arc::clone(&self.listeners);
            let task_config = self.config.clone();

            // Only spawn the background task if a tokio runtime is available.
            // When running inside sync-only tests (e.g. obrain-c), there is no
            // reactor and tokio::spawn would panic.
            if let Ok(rt_handle) = tokio::runtime::Handle::try_current() {
                *handle = Some(rt_handle.spawn(scheduler_loop(
                    rx,
                    task_listeners,
                    task_config,
                    shutdown_rx,
                )));
            }
        }

        /// Registers a new listener that will receive future event batches.
        ///
        /// Listeners are called in registration order. This can be called at any
        /// time — the new listener will receive events starting from the next batch.
        ///
        /// On the first call, this lazily spawns the background dispatch task.
        pub fn register_listener(&self, listener: Arc<dyn MutationListener>) {
            tracing::info!(listener = listener.name(), "registering mutation listener");
            self.listeners.write().push(listener);
            self.ensure_running();
        }

        /// Returns the current number of registered listeners.
        pub fn listener_count(&self) -> usize {
            self.listeners.read().len()
        }

        /// Returns a reference to the batch configuration.
        pub fn config(&self) -> &BatchConfig {
            &self.config
        }

        /// Returns `true` if the background task has been spawned.
        pub fn is_running(&self) -> bool {
            self.task_handle.lock().is_some()
        }

        /// Gracefully shuts down the scheduler.
        ///
        /// Signals the background task to stop, waits for it to drain any
        /// pending events, and then returns. After shutdown, the scheduler
        /// will no longer dispatch events.
        ///
        /// If the background task was never started (no listeners registered),
        /// this is a no-op.
        pub async fn shutdown(self) {
            // Signal the task to stop
            let _ = self.shutdown_tx.send(true);
            // Take the handle out of the mutex
            let handle = self.task_handle.lock().take();
            // Wait for the task to finish draining
            if let Some(handle) = handle {
                let _ = handle.await;
            }
        }
    }

    impl std::fmt::Debug for Scheduler {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("Scheduler")
                .field("config", &self.config)
                .field("listener_count", &self.listener_count())
                .field("running", &self.is_running())
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

    /// Dispatches a batch of events to all registered listeners **concurrently**.
    ///
    /// Each listener receives its filtered event slice in a separate spawned task
    /// via `tokio::task::JoinSet`. This ensures a slow listener does not block
    /// faster ones (required by T1.5 acceptance criteria).
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
            "dispatching batch to listeners concurrently"
        );

        // Share events across all spawned tasks via Arc
        let shared_events: Arc<Vec<MutationEvent>> = Arc::new(events.to_vec());
        let mut join_set = tokio::task::JoinSet::new();

        for listener in listener_snapshot {
            let events_ref = Arc::clone(&shared_events);
            join_set.spawn(async move {
                // Filter events this listener accepts
                let accepted: Vec<&MutationEvent> =
                    events_ref.iter().filter(|e| listener.accepts(e)).collect();

                if accepted.is_empty() {
                    return;
                }

                if accepted.len() == events_ref.len() {
                    // All accepted — pass the full slice
                    listener.on_batch(&events_ref).await;
                } else {
                    // Partial acceptance — collect accepted events
                    let filtered: Vec<MutationEvent> = accepted.into_iter().cloned().collect();
                    listener.on_batch(&filtered).await;
                }
            });
        }

        // Wait for all listeners to finish processing
        while join_set.join_next().await.is_some() {}
    }
}

// =============================================================================
// WASM scheduler — synchronous, no background task
// =============================================================================
#[cfg(target_arch = "wasm32")]
mod wasm {
    use super::*;

    /// A synchronous scheduler for WASM targets.
    ///
    /// Since WASM (wasm32-unknown-unknown) has no native threading or tokio runtime,
    /// this scheduler simply stores listeners and provides a manual `dispatch` method.
    /// Events are dispatched synchronously when `flush` is called.
    pub struct Scheduler {
        /// Registered listeners.
        listeners: ListenerList,
        /// Batch configuration (for introspection).
        config: BatchConfig,
        /// Reference to the bus.
        _bus: MutationBus,
    }

    impl Scheduler {
        /// Creates a new synchronous scheduler associated with the given bus.
        pub fn new(bus: &MutationBus, config: BatchConfig) -> Self {
            Self {
                listeners: Arc::new(RwLock::new(Vec::new())),
                config,
                _bus: bus.clone(),
            }
        }

        /// Registers a new listener.
        pub fn register_listener(&self, listener: Arc<dyn MutationListener>) {
            tracing::info!(
                listener = listener.name(),
                "registering mutation listener (wasm sync)"
            );
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

        /// Returns `true` if any listeners are registered.
        ///
        /// On WASM there is no background task — this returns `true` when listeners
        /// are present (i.e. the scheduler is "active").
        pub fn is_running(&self) -> bool {
            self.listener_count() > 0
        }

        /// No-op on WASM — there is no background task to shut down.
        pub async fn shutdown(self) {
            // Nothing to do
        }
    }

    impl std::fmt::Debug for Scheduler {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("Scheduler")
                .field("config", &self.config)
                .field("listener_count", &self.listener_count())
                .field("running", &self.is_running())
                .finish()
        }
    }
}

// Re-export the platform-appropriate Scheduler
#[cfg(not(target_arch = "wasm32"))]
pub use native::Scheduler;
#[cfg(target_arch = "wasm32")]
pub use wasm::Scheduler;
