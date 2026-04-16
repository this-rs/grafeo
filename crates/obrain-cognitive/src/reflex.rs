//! # Reflex Engine
//!
//! Fast-path pattern matching that bypasses LLM inference for known situations.
//! Each reflex is a `:ReflexPattern` node in the graph with energy and synapses,
//! allowing reflexes to strengthen with use and dissolve when inactive.
//!
//! ## Trigger Types
//!
//! - `Regex` — pattern match on input text
//! - `GraphPattern` — condition on graph state (node type, label, neighbor count)
//! - `EnergyThreshold` — fires when a node's energy crosses a threshold
//! - `EventType` — matches mutation bus events by type
//!
//! ## Action Types
//!
//! - `Query` — execute a graph query and return results
//! - `InjectContext` — inject specific nodes into the context bundle
//! - `EmitEvent` — emit an event on the mutation bus
//! - `SkipLlm` — return a pre-computed response without LLM
//!
//! ## Learning
//!
//! - Successful reflex (no user correction) → Hebbian boost to energy
//! - Failed reflex (user overrides) → energy penalty
//! - Energy reaches 0 → reflex dissolves

use obrain_common::types::NodeId;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Label for reflex pattern nodes in the graph.
pub const LABEL_REFLEX_PATTERN: &str = "ReflexPattern";

// ---------------------------------------------------------------------------
// Trigger types
// ---------------------------------------------------------------------------

/// What condition activates a reflex.
#[derive(Debug, Clone, PartialEq)]
pub enum TriggerType {
    /// Regex pattern match on input text.
    Regex(String),
    /// Condition on graph state.
    GraphPattern(GraphCondition),
    /// Fires when a node's energy exceeds/drops below threshold.
    EnergyThreshold {
        node_id: NodeId,
        threshold: f64,
        above: bool,
    },
    /// Matches events from the mutation bus.
    EventType(String),
}

/// A condition evaluated on the graph.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphCondition {
    /// Label the node must have.
    pub label: Option<String>,
    /// Minimum neighbor count.
    pub min_neighbors: Option<u32>,
    /// Property key that must exist.
    pub has_property: Option<String>,
    /// Property key=value match.
    pub property_equals: Option<(String, String)>,
}

// ---------------------------------------------------------------------------
// Action types
// ---------------------------------------------------------------------------

/// What the reflex does when triggered.
#[derive(Debug, Clone, PartialEq)]
pub enum ActionType {
    /// Execute a pre-defined query pattern.
    Query(String),
    /// Inject specific nodes into the agent context.
    InjectContext(Vec<NodeId>),
    /// Emit an event on the mutation bus.
    EmitEvent(String),
    /// Return a pre-computed response (skip LLM entirely).
    SkipLlm(String),
}

// ---------------------------------------------------------------------------
// ReflexPattern
// ---------------------------------------------------------------------------

/// A reflex pattern stored as a graph node.
#[derive(Debug, Clone)]
pub struct ReflexPattern {
    /// Node id in the graph (set after persistence).
    pub node_id: Option<NodeId>,
    /// What activates this reflex.
    pub trigger: TriggerType,
    /// What the reflex does.
    pub action: ActionType,
    /// Number of times this reflex has been activated.
    pub hit_count: u64,
    /// Success rate (successful activations / total activations).
    pub confidence: f64,
    /// Energy — participates in engram lifecycle.
    pub energy: f64,
    /// Whether this reflex was created automatically (from crystallization).
    pub auto_created: bool,
}

impl ReflexPattern {
    /// Create a new reflex pattern.
    pub fn new(trigger: TriggerType, action: ActionType) -> Self {
        Self {
            node_id: None,
            trigger,
            action,
            hit_count: 0,
            confidence: 0.5, // neutral confidence until proven
            energy: 1.0,
            auto_created: false,
        }
    }

    /// Record a successful activation (user didn't correct).
    pub fn record_success(&mut self) {
        self.hit_count += 1;
        // EMA update toward 1.0
        self.confidence = self.confidence * 0.9 + 0.1;
        // Hebbian boost
        self.energy = (self.energy + 0.1).min(1.0);
    }

    /// Record a failed activation (user corrected/overrode).
    pub fn record_failure(&mut self) {
        self.hit_count += 1;
        // EMA update toward 0.0
        self.confidence = self.confidence * 0.9;
        // Energy penalty
        self.energy = (self.energy - 0.15).max(0.0);
    }

    /// Check if this reflex should dissolve (energy depleted).
    pub fn should_dissolve(&self) -> bool {
        self.energy <= 0.01
    }

    /// Check if this reflex is active (enough energy to fire).
    pub fn is_active(&self, min_energy: f64) -> bool {
        self.energy >= min_energy
    }
}

// ---------------------------------------------------------------------------
// Reflex Engine
// ---------------------------------------------------------------------------

/// Result of checking reflexes against an input.
#[derive(Debug, Clone)]
pub struct ReflexMatch {
    /// Index of the matched pattern in the engine's store.
    pub pattern_idx: usize,
    /// The action to execute.
    pub action: ActionType,
    /// Confidence of the match.
    pub confidence: f64,
}

/// The reflex engine — evaluates patterns against inputs.
#[derive(Debug)]
pub struct ReflexEngine {
    /// All registered patterns.
    patterns: Vec<ReflexPattern>,
    /// Minimum energy for a pattern to be considered.
    min_energy: f64,
    /// Minimum confidence for a match to be returned.
    min_confidence: f64,
}

impl ReflexEngine {
    /// Create a new reflex engine.
    pub fn new(min_energy: f64, min_confidence: f64) -> Self {
        Self {
            patterns: Vec::new(),
            min_energy,
            min_confidence,
        }
    }

    /// Register a new reflex pattern.
    pub fn register(&mut self, pattern: ReflexPattern) -> usize {
        let idx = self.patterns.len();
        self.patterns.push(pattern);
        idx
    }

    /// Get a mutable reference to a pattern by index.
    pub fn get_mut(&mut self, idx: usize) -> Option<&mut ReflexPattern> {
        self.patterns.get_mut(idx)
    }

    /// Check all active patterns against a text input.
    /// Returns the best match (highest confidence) if any.
    pub fn check_text(&self, input: &str) -> Option<ReflexMatch> {
        let mut best: Option<ReflexMatch> = None;

        for (idx, pattern) in self.patterns.iter().enumerate() {
            if !pattern.is_active(self.min_energy) {
                continue;
            }
            if pattern.confidence < self.min_confidence {
                continue;
            }

            let matches = match &pattern.trigger {
                TriggerType::Regex(re) => {
                    // Simple substring match (in production, use regex crate)
                    input.contains(re.as_str())
                }
                _ => false, // text check only matches regex triggers
            };

            if matches {
                if best.as_ref().map_or(true, |b| pattern.confidence > b.confidence) {
                    best = Some(ReflexMatch {
                        pattern_idx: idx,
                        action: pattern.action.clone(),
                        confidence: pattern.confidence,
                    });
                }
            }
        }

        best
    }

    /// Check all active patterns against an event type.
    pub fn check_event(&self, event_type: &str) -> Option<ReflexMatch> {
        let mut best: Option<ReflexMatch> = None;

        for (idx, pattern) in self.patterns.iter().enumerate() {
            if !pattern.is_active(self.min_energy) {
                continue;
            }
            if pattern.confidence < self.min_confidence {
                continue;
            }

            let matches = match &pattern.trigger {
                TriggerType::EventType(et) => et == event_type,
                _ => false,
            };

            if matches {
                if best.as_ref().map_or(true, |b| pattern.confidence > b.confidence) {
                    best = Some(ReflexMatch {
                        pattern_idx: idx,
                        action: pattern.action.clone(),
                        confidence: pattern.confidence,
                    });
                }
            }
        }

        best
    }

    /// Remove dissolved patterns (energy <= 0).
    pub fn sweep_dissolved(&mut self) -> usize {
        let before = self.patterns.len();
        self.patterns.retain(|p| !p.should_dissolve());
        before - self.patterns.len()
    }

    /// Get all active pattern count.
    pub fn active_count(&self) -> usize {
        self.patterns
            .iter()
            .filter(|p| p.is_active(self.min_energy))
            .count()
    }

    /// Total pattern count (including inactive).
    pub fn total_count(&self) -> usize {
        self.patterns.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_and_check() {
        let mut engine = ReflexEngine::new(0.1, 0.3);
        engine.register(ReflexPattern::new(
            TriggerType::Regex("compilation error".into()),
            ActionType::Query("MATCH (n:Gotcha) WHERE n.content CONTAINS 'compile'".into()),
        ));

        let result = engine.check_text("I got a compilation error on main.rs");
        assert!(result.is_some());
        assert_eq!(result.unwrap().pattern_idx, 0);
    }

    #[test]
    fn test_no_match() {
        let mut engine = ReflexEngine::new(0.1, 0.3);
        engine.register(ReflexPattern::new(
            TriggerType::Regex("compilation error".into()),
            ActionType::SkipLlm("Check the compiler output".into()),
        ));

        let result = engine.check_text("everything is working fine");
        assert!(result.is_none());
    }

    #[test]
    fn test_confidence_filter() {
        let mut engine = ReflexEngine::new(0.1, 0.8);
        let mut pattern = ReflexPattern::new(
            TriggerType::Regex("test".into()),
            ActionType::SkipLlm("response".into()),
        );
        pattern.confidence = 0.5; // below threshold
        engine.register(pattern);

        let result = engine.check_text("test input");
        assert!(result.is_none()); // filtered by confidence
    }

    #[test]
    fn test_energy_depletion() {
        let mut pattern = ReflexPattern::new(
            TriggerType::Regex("x".into()),
            ActionType::SkipLlm("y".into()),
        );
        assert!(!pattern.should_dissolve());

        // Simulate repeated failures
        for _ in 0..20 {
            pattern.record_failure();
        }
        assert!(pattern.should_dissolve());
    }

    #[test]
    fn test_success_builds_confidence() {
        let mut pattern = ReflexPattern::new(
            TriggerType::Regex("x".into()),
            ActionType::SkipLlm("y".into()),
        );
        let initial = pattern.confidence;

        for _ in 0..10 {
            pattern.record_success();
        }
        assert!(pattern.confidence > initial);
        assert!(pattern.energy > 0.9);
    }

    #[test]
    fn test_event_matching() {
        let mut engine = ReflexEngine::new(0.1, 0.3);
        engine.register(ReflexPattern::new(
            TriggerType::EventType("NodeCreated".into()),
            ActionType::EmitEvent("RefreshContext".into()),
        ));

        assert!(engine.check_event("NodeCreated").is_some());
        assert!(engine.check_event("NodeDeleted").is_none());
    }

    #[test]
    fn test_sweep_dissolved() {
        let mut engine = ReflexEngine::new(0.1, 0.3);
        let mut dead = ReflexPattern::new(
            TriggerType::Regex("dead".into()),
            ActionType::SkipLlm("x".into()),
        );
        dead.energy = 0.0;
        engine.register(dead);

        engine.register(ReflexPattern::new(
            TriggerType::Regex("alive".into()),
            ActionType::SkipLlm("y".into()),
        ));

        assert_eq!(engine.total_count(), 2);
        let swept = engine.sweep_dissolved();
        assert_eq!(swept, 1);
        assert_eq!(engine.total_count(), 1);
    }

    #[test]
    fn test_best_match_by_confidence() {
        let mut engine = ReflexEngine::new(0.1, 0.3);

        let mut p1 = ReflexPattern::new(
            TriggerType::Regex("error".into()),
            ActionType::SkipLlm("response_1".into()),
        );
        p1.confidence = 0.6;

        let mut p2 = ReflexPattern::new(
            TriggerType::Regex("error".into()),
            ActionType::SkipLlm("response_2".into()),
        );
        p2.confidence = 0.9;

        engine.register(p1);
        engine.register(p2);

        let result = engine.check_text("got an error");
        assert!(result.is_some());
        assert_eq!(result.unwrap().pattern_idx, 1); // higher confidence wins
    }
}
