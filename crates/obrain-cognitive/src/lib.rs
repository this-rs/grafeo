//! # obrain-cognitive
//!
//! Cognitive features for Obrain — energy scoring, Hebbian synapses,
//! spreading activation, knowledge fabric, and more.
//!
//! Each cognitive subsystem implements [`obrain_reactive::MutationListener`]
//! and reacts asynchronously to graph mutations via the reactive substrate.
//!
//! ## Architecture
//!
//! ```text
//! MutationBus (obrain-reactive)
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
//! - `fabric` — Knowledge fabric metrics (mutation frequency, annotation density, risk)
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

#[cfg(feature = "consolidation")]
pub mod consolidation;

// Kernel embeddings — irreducible kernel C = Φ₀ ∘ A¹(H^∞)
#[cfg(feature = "kernel")]
pub mod kernel;

// Stigmergy — diffuse pheromone memory on edges (Layer 2)
#[cfg(feature = "stigmergy")]
pub mod stigmergy;

// Immune system — anomaly detection via shape space (Layer 1)
#[cfg(feature = "immune")]
pub mod immune;

// Epigenetic layer — cross-instance transgenerational memory (Layer 5)
#[cfg(feature = "epigenetic")]
pub mod epigenetic;

// Cognitive Persona — database-native knowledge lens
#[cfg(feature = "persona")]
pub mod persona;

// Engram system — biomimetic memory traces (Layer 0+)
pub mod engram;

// Level 2 introspection procedures (engrams.list, engrams.inspect, engrams.forget, cognitive.metrics)
#[cfg(feature = "engram")]
pub mod procedures;

// Provenance — automatic cognitive event tracking
pub mod provenance;

// Per-tenant isolation via named graphs
pub mod tenant;

// Search pipeline (requires at least cognitive features for full functionality)
pub mod search;

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

pub use config::{CognitiveConfig, KernelConfigToml};

#[cfg(feature = "energy")]
pub use energy::{
    EnergyConfig, EnergyListener, EnergyStore, NodeEnergy, effective_half_life, energy_score,
};

pub use engine::{CognitiveEngine, CognitiveEngineBuilder, DefaultCognitiveEngine};
pub use error::CognitiveError;

#[cfg(feature = "fabric")]
pub use fabric::{FabricListener, FabricScore, FabricStore, RiskWeights};

#[cfg(feature = "gds-refresh")]
pub use gds_refresh::{GdsRefreshConfig, GdsRefreshScheduler};

#[cfg(feature = "synapse")]
pub use synapse::{
    Synapse, SynapseConfig, SynapseListener, SynapseStore, mutation_frequency_score, synapse_score,
};

#[cfg(feature = "scar")]
pub use scar::{Scar, ScarConfig, ScarStore};

#[cfg(all(feature = "memory", not(target_arch = "wasm32")))]
pub use memory::FileArchiveBackend;
#[cfg(feature = "memory")]
pub use memory::{
    ArchiveBackend, InMemoryArchiveBackend, MemoryConfig, MemoryHorizon, MemoryManager,
    MemoryStore, NodeMemoryState, SweepResult,
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

#[cfg(feature = "consolidation")]
pub use consolidation::{
    ConsolidationConfig, ConsolidationEngine, ConsolidationResult,
    EDGE_DERIVED_FROM as EDGE_CONSOLIDATION_DERIVED_FROM,
};

pub use search::{
    NoopReranker, Reranker, SearchConfig, SearchPipeline, SearchResult, SearchWeights,
};

pub use provenance::{
    CognitiveEvent, CognitiveEventId, CognitiveEventType, DerivedFromRecord, EDGE_DERIVED_FROM,
    EDGE_HAS_COGNITIVE_EVENT, ProvenanceRecorder,
};

pub use tenant::{TenantError, TenantGraph, TenantInfo, TenantManager};

// Engram re-exports (always available: traits + types; feature-gated: subsystems)
pub use engram::traits::{
    CognitiveEdge, CognitiveFilter, CognitiveNode, CognitiveObservability, CognitiveStorage,
    EdgeAnnotator, InMemoryVectorIndex, NoopQueryObserver, QueryObserver, VectorIndex,
};
pub use engram::{
    Engram, EngramHorizon, EngramId, EpisodeId, FsrsState, PredictionError, RecallEvent,
    RecallFeedback,
};

#[cfg(feature = "kernel")]
pub use kernel::KernelListener;

#[cfg(feature = "stigmergy")]
pub use stigmergy::{
    AtomicF64, PheromoneMap, StigmergicEngine, StigmergicFormationBridge,
    StigmergicMutationListener, StigmergicQueryListener, StigmergicTrace, TrailType,
};

#[cfg(feature = "engram")]
pub use engram::{
    ActivatedEngram, CoActivationDetector, CognitiveMetrics, CompetitionResult,
    CrystallizationConfig, CrystallizationDetector, CrystallizationProposal, CrystallizationResult,
    DetailLevel, EngramFormationTrigger, EngramManager, EngramMetricsCollector, EngramStore,
    FormationConfig, FsrsConfig, FsrsScheduler, HebbianWithSurprise, HomeostasisConfig,
    HomeostasisEngine, HomeostasisSignal, IMMUNE_FP_RATE_THRESHOLD, LABEL_CRYSTALLIZED_NOTE,
    LOW_PRECISION_BETA_THRESHOLD, MmrResult, REL_CRYSTALLIZED_IN, RecallEngine, RecallResult,
    ReviewRating, SpectralEncoder, WarmupConfig, WarmupSelector, crystallize, generate_summary,
    hopfield_retrieve, max_marginal_relevance, softmax_compete,
};

#[cfg(feature = "immune")]
pub use immune::{
    DEFAULT_AFFINITY_RADIUS, Detection, DetectorId, ImmuneDetector, ImmuneSystem, ShapeDescriptor,
};

#[cfg(feature = "epigenetic")]
pub use epigenetic::{
    EngramTemplate, EpigeneticBridge, EpigeneticMark, EpigeneticMarkId, ExpressionCondition,
    LABEL_EPIGENETIC_MARK, ProjectContext, SerializedMark, TRANSGENERATIONAL_DECAY,
};

#[cfg(feature = "persona")]
pub use persona::{
    CognitivePersona, PersonaConfig, PersonaFeedbackResult, PersonaId, PersonaRecallEngine,
    PersonaRecallResult, PersonaStats, PersonaStore,
};
