//! # Engram System — Mémoire Vivante pour Grafeo
//!
//! Engrams are emergent memory traces formed from repeated co-activation patterns.
//! Unlike raw synapses or energy scores, engrams represent *consolidated knowledge*
//! — groups of nodes that consistently appear together across episodes.
//!
//! ## Architecture (6 Biomimetic Layers + Homeostasis)
//!
//! - **Layer 0**: Hebbian + FSRS — engram formation via co-activation with surprise,
//!   decay via Free Spaced Repetition Scheduler
//! - **Layer 1**: Immuno-memory — anomaly detection via shape space + affinity maturation
//! - **Layer 2**: Stigmergy — diffuse memory via pheromones on edges
//! - **Layer 3+4**: Modern Hopfield + Predictive Coding — content-addressable retrieval
//!   with prediction error as main learning signal
//! - **Layer 5**: Epigenetics — cross-instance memory marks with expression conditions
//! - **Layer ⊥**: Homeostasis — synaptic scaling, T-reg regulation, anti-lock-in
//!
//! ## Feature flags
//!
//! - `engram` — Core engram formation, recall, decay (Layer 0)
//! - `engram-immune` — Immuno-memory subsystem (Layer 1)
//! - `engram-stigmergy` — Stigmergic pheromones (Layer 2)
//! - `engram-hopfield` — Modern Hopfield + predictive coding (Layers 3+4)
//! - `engram-epigenetic` — Cross-instance epigenetic marks (Layer 5)
//! - `engram-full` — All engram layers

pub mod traits;

mod types;
pub use types::*;

#[cfg(feature = "engram")]
mod store;
#[cfg(feature = "engram")]
pub use store::EngramStore;

#[cfg(feature = "engram")]
mod formation;
#[cfg(feature = "engram")]
pub use formation::{
    CoActivationDetector, EngramFormationTrigger, FormationConfig, HebbianWithSurprise,
};

#[cfg(feature = "engram")]
mod decay;
#[cfg(feature = "engram")]
pub use decay::{FsrsConfig, FsrsScheduler, ReviewRating};

#[cfg(feature = "engram")]
mod spectral;
#[cfg(feature = "engram")]
pub use spectral::SpectralEncoder;

#[cfg(feature = "engram")]
mod recall;
#[cfg(feature = "engram")]
pub use recall::{RecallEngine, RecallResult, WarmupConfig, WarmupSelector};

#[cfg(feature = "engram")]
mod hopfield;
#[cfg(feature = "engram")]
pub use hopfield::{HopfieldResult, PatternMatrix, hopfield_retrieve};

#[cfg(feature = "engram")]
mod homeostasis;
#[cfg(feature = "engram")]
pub use homeostasis::{HomeostasisConfig, HomeostasisEngine};

#[cfg(feature = "engram")]
mod observe;
#[cfg(feature = "engram")]
pub use observe::{
    CognitiveMetrics, CognitiveMetricsSnapshot, EngramMetricsCollector,
    compute_max_pheromone_ratio, compute_pheromone_entropy,
};

#[cfg(feature = "engram")]
mod manager;
#[cfg(feature = "engram")]
pub use manager::EngramManager;
