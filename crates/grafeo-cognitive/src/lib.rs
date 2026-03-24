//! # grafeo-cognitive
//!
//! Cognitive features for Grafeo — energy scoring, Hebbian synapses,
//! spreading activation, knowledge fabric, and more.
//!
//! Each cognitive subsystem implements [`grafeo_reactive::MutationListener`]
//! and reacts asynchronously to graph mutations via the reactive substrate.
//!
//! ## Architecture
//!
//! ```text
//! MutationBus (grafeo-reactive)
//!       |
//!   Scheduler → dispatches batches to:
//!       |
//!       ├── EnergyListener   — tracks node activation energy
//!       ├── SynapseListener  — Hebbian co-activation learning
//!       ├── FabricListener   — knowledge density & risk scoring
//!       └── ...              — more cognitive subsystems
//! ```

#![deny(unsafe_code)]

pub mod activation;
pub mod co_change;
pub mod energy;
pub mod error;
pub mod synapse;

pub use activation::{
    ActivationMap, ActivationSource, SpreadConfig, SynapseActivationSource, spread, spread_single,
};
pub use co_change::{CoChangeConfig, CoChangeDetector, CoChangeRelation, CoChangeStore};
pub use energy::{EnergyConfig, EnergyListener, EnergyStore, NodeEnergy};
pub use error::CognitiveError;
pub use synapse::{Synapse, SynapseConfig, SynapseListener, SynapseStore};
