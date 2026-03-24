//! # grafeo-reactive
//!
//! Reactive event substrate for Grafeo. Provides a [`MutationBus`] that publishes
//! [`MutationEvent`]s whenever graph mutations occur, enabling downstream cognitive
//! features to react in real-time.
//!
//! ## Architecture
//!
//! ```text
//! GraphStoreMut calls
//!        |
//!   InstrumentedStore<S>  (wrapper, intercepts mutations)
//!        |
//!   MutationBus           (tokio::broadcast channel)
//!        |
//!   MutationListener(s)   (async consumers: energy, synapses, fabric, ...)
//! ```
//!
//! ## Zero-cost when unused
//!
//! The [`MutationBus`] uses `tokio::broadcast` which has < 5us overhead per
//! publish when there are no subscribers (just an atomic check + discard).

#![deny(unsafe_code)]

mod bus;
pub mod error;
mod event;
mod listener;
mod store;

pub use bus::MutationBus;
pub use error::ReactiveError;
pub use event::{MutationEvent, MutationBatch, NodeSnapshot, EdgeSnapshot};
pub use listener::MutationListener;
pub use store::InstrumentedStore;

#[cfg(test)]
mod tests;
