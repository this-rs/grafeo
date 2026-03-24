//! The [`MutationBus`] — a thin wrapper around `tokio::broadcast` for publishing
//! mutation events with minimal overhead.

use crate::event::{MutationBatch, MutationEvent};
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::broadcast;

/// Default capacity of the broadcast channel.
///
/// With 16384 slots and ~256 bytes per event, this is ~4 MB of buffer.
/// Slow consumers that fall behind will receive a `Lagged` error and can
/// skip ahead.
const DEFAULT_CAPACITY: usize = 16_384;

/// The mutation event bus.
///
/// Backed by a `tokio::broadcast` channel. Publishing is lock-free and
/// costs < 5us when there are no subscribers (the message is simply dropped).
///
/// # Usage
///
/// ```rust,no_run
/// use grafeo_reactive::{MutationBus, MutationEvent, MutationBatch};
///
/// let bus = MutationBus::new();
/// let mut rx = bus.subscribe();
///
/// // Publisher side (in commit hook):
/// let batch = MutationBatch::new(vec![]);
/// bus.publish_batch(batch);
///
/// // Subscriber side (in async listener):
/// // while let Ok(batch) = rx.recv().await { ... }
/// ```
#[derive(Debug)]
pub struct MutationBus {
    tx: broadcast::Sender<MutationBatch>,
    /// Total events published (for metrics).
    events_published: AtomicU64,
    /// Total batches published (for metrics).
    batches_published: AtomicU64,
}

impl MutationBus {
    /// Creates a new bus with default capacity (16384 events).
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CAPACITY)
    }

    /// Creates a new bus with the given channel capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        let (tx, _) = broadcast::channel(capacity);
        Self {
            tx,
            events_published: AtomicU64::new(0),
            batches_published: AtomicU64::new(0),
        }
    }

    /// Publishes a single event as a batch of one.
    ///
    /// Returns `true` if at least one subscriber received the event.
    pub fn publish(&self, event: MutationEvent) -> bool {
        self.publish_batch(MutationBatch::new(vec![event]))
    }

    /// Publishes a batch of events from a single commit.
    ///
    /// Returns `true` if at least one subscriber received the batch.
    /// When there are no subscribers, the batch is dropped with minimal overhead.
    pub fn publish_batch(&self, batch: MutationBatch) -> bool {
        if batch.is_empty() {
            return false;
        }

        let event_count = batch.len() as u64;
        let result = self.tx.send(batch);

        match result {
            Ok(receiver_count) => {
                self.events_published
                    .fetch_add(event_count, Ordering::Relaxed);
                self.batches_published.fetch_add(1, Ordering::Relaxed);
                tracing::trace!(
                    events = event_count,
                    receivers = receiver_count,
                    "mutation batch published"
                );
                true
            }
            Err(_) => {
                // No subscribers — this is the zero-cost path.
                // The batch is dropped here.
                tracing::trace!(
                    events = event_count,
                    "mutation batch dropped (no subscribers)"
                );
                false
            }
        }
    }

    /// Creates a new subscriber to this bus.
    ///
    /// The subscriber will receive all batches published after this call.
    /// If the subscriber falls behind, it will receive a `Lagged` error
    /// indicating how many messages were missed.
    pub fn subscribe(&self) -> broadcast::Receiver<MutationBatch> {
        self.tx.subscribe()
    }

    /// Returns the total number of events published since bus creation.
    pub fn total_events_published(&self) -> u64 {
        self.events_published.load(Ordering::Relaxed)
    }

    /// Returns the total number of batches published since bus creation.
    pub fn total_batches_published(&self) -> u64 {
        self.batches_published.load(Ordering::Relaxed)
    }

    /// Returns the current number of active subscribers.
    pub fn subscriber_count(&self) -> usize {
        self.tx.receiver_count()
    }
}

impl Default for MutationBus {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for MutationBus {
    fn clone(&self) -> Self {
        Self {
            tx: self.tx.clone(),
            events_published: AtomicU64::new(self.events_published.load(Ordering::Relaxed)),
            batches_published: AtomicU64::new(self.batches_published.load(Ordering::Relaxed)),
        }
    }
}
