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
//!       ├── CoChangeDetector — temporal coupling detection
//!       └── ...              — more cognitive subsystems
//! ```
//!
//! ## Feature flags
//!
//! - `energy` — Energy subsystem (exponential decay + boost)
//! - `synapse` — Hebbian synapse learning
//! - `fabric` — Knowledge fabric metrics (churn, density, risk)
//! - `co-change` — Co-change detection (temporal coupling)
//! - `gds-refresh` — GDS refresh scheduler (PageRank, Louvain, Betweenness)
//! - `cognitive` — Convenience: energy + synapse
//! - `cognitive-fabric` — Convenience: cognitive + fabric + co-change + gds-refresh
//! - `cognitive-full` — Convenience: all subsystems

#![deny(unsafe_code)]

// ---------------------------------------------------------------------------
// Feature-gated subsystem modules
// ---------------------------------------------------------------------------

#[cfg(feature = "energy")]
pub mod energy;

#[cfg(feature = "synapse")]
pub mod synapse;

#[cfg(feature = "synapse")]
pub mod activation;

#[cfg(feature = "fabric")]
pub mod fabric;

#[cfg(feature = "co-change")]
pub mod co_change;

#[cfg(feature = "gds-refresh")]
pub mod gds_refresh;

#[cfg(feature = "scar")]
pub mod scar;

#[cfg(feature = "memory")]
pub mod memory;

#[cfg(feature = "stagnation")]
pub mod stagnation;

#[cfg(feature = "fingerprint")]
pub mod fingerprint;

#[cfg(feature = "distillation")]
pub mod distillation;

#[cfg(feature = "episodic")]
pub mod episodic;

// Always-available modules
pub mod config;
pub mod engine;
pub mod error;
pub mod store_trait;

// ---------------------------------------------------------------------------
// Re-exports
// ---------------------------------------------------------------------------

#[cfg(feature = "synapse")]
pub use activation::{
    ActivationMap, ActivationSource, SpreadConfig, SynapseActivationSource, spread, spread_single,
};

#[cfg(feature = "co-change")]
pub use co_change::{CoChangeConfig, CoChangeDetector, CoChangeRelation, CoChangeStore};

pub use config::CognitiveConfig;

#[cfg(feature = "energy")]
pub use energy::{EnergyConfig, EnergyListener, EnergyStore, NodeEnergy, energy_score};

pub use engine::{CognitiveEngine, CognitiveEngineBuilder, DefaultCognitiveEngine};
pub use error::CognitiveError;

#[cfg(feature = "fabric")]
pub use fabric::{FabricListener, FabricScore, FabricStore};

#[cfg(feature = "gds-refresh")]
pub use gds_refresh::{GdsRefreshConfig, GdsRefreshScheduler};

#[cfg(feature = "synapse")]
pub use synapse::{Synapse, SynapseConfig, SynapseListener, SynapseStore, synapse_score};

#[cfg(feature = "scar")]
pub use scar::{Scar, ScarConfig, ScarStore};

#[cfg(feature = "memory")]
pub use memory::{
    ArchiveBackend, FileArchiveBackend, InMemoryArchiveBackend, MemoryConfig, MemoryHorizon,
    MemoryManager, MemoryStore, NodeMemoryState, SweepResult,
};

#[cfg(feature = "stagnation")]
pub use stagnation::{
    StagnationConfig, StagnationDetector, StagnationScore, StagnationStore, Trend,
};

#[cfg(feature = "fingerprint")]
pub use fingerprint::{MotifType, StructuralFingerprint, compare, detect_twins, fingerprint};

#[cfg(feature = "distillation")]
pub use distillation::{
    ArtifactMetadata, CommunityFingerprint, DistillArtifact, DistillConfig, EnergySnapshot,
    EvaluateConfig, ParityReport, SynapseSnapshot, distill, evaluate, evaluate_with_config, inject,
};

#[cfg(feature = "episodic")]
pub use episodic::{
    ActivationStep, Episode, EpisodeConfig, EpisodeHorizon, EpisodeMemoryManager, EpisodeRecorder,
    EpisodeStore, EpisodeSweepResult, Outcome, Stimulus, ValidationResult,
};
