//! Core engram types — data model for the engram system.

use obrain_common::types::NodeId;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::time::SystemTime;

// ---------------------------------------------------------------------------
// EngramId
// ---------------------------------------------------------------------------

/// Unique identifier for an engram.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EngramId(pub u64);

impl fmt::Display for EngramId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "engram:{}", self.0)
    }
}

/// Unique identifier for an episode (links to episodic memory).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EpisodeId(pub u64);

// ---------------------------------------------------------------------------
// Engram — the core data structure
// ---------------------------------------------------------------------------

/// An engram is a consolidated memory trace formed from repeated co-activation.
///
/// It represents a group of nodes that consistently appear together across
/// multiple episodes, with associated strength, valence, and spectral signature.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Engram {
    /// Unique engram identifier.
    pub id: EngramId,

    /// The node ensemble: (NodeId, contribution_weight) ordered temporally.
    /// The contribution weight reflects how central each node is to this engram.
    pub ensemble: Vec<(NodeId, f64)>,

    /// Spectral signature — a compact vector representation of this engram's
    /// "shape" in the graph. Used for similarity search and Hopfield retrieval.
    pub spectral_signature: Vec<f64>,

    /// Strength ∈ [0.0, 1.0] — how consolidated this engram is.
    /// Increases with recalls, decays via FSRS when not accessed.
    pub strength: f64,

    /// Valence ∈ [-1.0, 1.0] — emotional/outcome coloring.
    /// Negative = error/bug pattern, positive = successful pattern.
    pub valence: f64,

    /// Precision β for Modern Hopfield retrieval.
    /// Higher precision = sharper pattern matching, lower = fuzzier.
    pub precision: f64,

    /// History of recalls (when and in what context this engram was retrieved).
    pub recall_history: Vec<RecallEvent>,

    /// Source episodes that contributed to this engram's formation.
    pub source_episodes: Vec<EpisodeId>,

    /// Current memory horizon (operational, consolidated, archived).
    pub horizon: EngramHorizon,

    /// When this engram was first formed.
    pub created_at: SystemTime,

    /// When this engram was last recalled or reinforced.
    pub last_activated: SystemTime,

    /// FSRS scheduling state for spaced repetition decay.
    pub fsrs_state: FsrsState,

    /// Number of times this engram has been recalled.
    pub recall_count: u32,

    /// Predictive model — P(outcome | context) for this engram.
    /// Stores mean + variance of observed outcomes. Used by the predictive
    /// coding layer (Layer 3+4) to compute prediction errors.
    #[cfg(feature = "engram")]
    #[serde(default)]
    pub predictive_model: Option<super::hopfield::PredictiveModel>,
}

impl Engram {
    /// Create a new engram with default values.
    pub fn new(id: EngramId, ensemble: Vec<(NodeId, f64)>) -> Self {
        let now = SystemTime::now();
        Self {
            id,
            ensemble,
            spectral_signature: Vec::new(),
            strength: 0.5,
            valence: 0.0,
            precision: 1.0,
            recall_history: Vec::new(),
            source_episodes: Vec::new(),
            horizon: EngramHorizon::Operational,
            created_at: now,
            last_activated: now,
            fsrs_state: FsrsState::default(),
            recall_count: 0,
            #[cfg(feature = "engram")]
            predictive_model: None,
        }
    }

    /// Returns the nodes in this engram's ensemble.
    pub fn node_ids(&self) -> Vec<NodeId> {
        self.ensemble.iter().map(|(id, _)| *id).collect()
    }

    /// Returns true if the engram's strength is above the consolidation threshold.
    pub fn is_consolidated(&self) -> bool {
        self.strength >= 0.6
    }

    /// Returns true if the engram is a candidate for crystallization (→ note).
    pub fn is_crystallization_candidate(&self) -> bool {
        self.strength > 0.85 && self.recall_count >= 5
    }
}

// ---------------------------------------------------------------------------
// Supporting types
// ---------------------------------------------------------------------------

/// Memory horizon for engrams (similar to node MemoryHorizon but engram-specific).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EngramHorizon {
    /// Hot: recently formed or actively recalled.
    Operational,
    /// Warm: stable, confirmed pattern.
    Consolidated,
    /// Cold: decaying, rarely recalled.
    Archived,
}

impl fmt::Display for EngramHorizon {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Operational => write!(f, "operational"),
            Self::Consolidated => write!(f, "consolidated"),
            Self::Archived => write!(f, "archived"),
        }
    }
}

/// A record of when and in what context an engram was recalled.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallEvent {
    /// When the recall happened.
    pub timestamp: SystemTime,
    /// The cues that triggered this recall (e.g., file paths, node IDs).
    pub cues: Vec<String>,
    /// How useful the recall was (feedback from the consumer).
    pub feedback: Option<RecallFeedback>,
}

/// Feedback on a recall event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecallFeedback {
    /// The recalled engram was useful.
    Used,
    /// The recalled engram was not useful.
    Rejected,
}

/// FSRS scheduling state for an engram.
///
/// Implements the Free Spaced Repetition Scheduler algorithm for
/// memory decay and optimal review scheduling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FsrsState {
    /// Stability — how long until the memory decays to 90% recall probability.
    pub stability: f64,
    /// Difficulty ∈ [0.0, 1.0] — how hard this pattern is to remember.
    pub difficulty: f64,
    /// Number of successful recalls (reviews).
    pub reps: u32,
    /// Number of consecutive lapses (forgetting events).
    pub lapses: u32,
    /// Last review timestamp.
    pub last_review: Option<SystemTime>,
    /// Scheduled next review timestamp.
    pub next_review: Option<SystemTime>,
}

impl Default for FsrsState {
    fn default() -> Self {
        Self {
            stability: 1.0,
            difficulty: 0.3,
            reps: 0,
            lapses: 0,
            last_review: None,
            next_review: None,
        }
    }
}

/// Prediction error from the predictive coding layer.
///
/// When the system predicts what should happen next and reality differs,
/// the prediction error is the main learning trigger.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionError {
    /// The predicted outcome (expected node activations).
    pub predicted: Vec<NodeId>,
    /// The actual outcome (observed node activations).
    pub actual: Vec<NodeId>,
    /// Magnitude of the error ∈ [0.0, ∞).
    pub magnitude: f64,
    /// Surprise factor — how unexpected was this? (normalized 0-1)
    pub surprise: f64,
}
