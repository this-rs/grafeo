//! # Skill Trigger System
//!
//! Evaluates trigger patterns against events to determine which skills
//! should activate. Supports regex, file globs, semantic similarity,
//! event matching, and graph mutation patterns.
//!
//! ## Trigger Types
//!
//! - `Regex` — regex match on text (file path, content, input)
//! - `FileGlob` — glob pattern on file paths (e.g. "src/**/*.rs")
//! - `Semantic` — cosine similarity on embeddings (uses HNSW index)
//! - `Event` — matches mutation bus events by type/label
//! - `GraphMutation` — matches specific graph mutations (node created, edge added)
//!
//! ## Lifecycle
//!
//! - Triggers are created automatically during crystallization (T6)
//! - Triggers that never fire lose energy and dissolve
//! - Triggers that fire frequently gain energy
//! - Each trigger is a `:TriggerPattern` node linked to its Skill

use obrain_common::types::NodeId;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Label for trigger pattern nodes.
pub const LABEL_TRIGGER_PATTERN: &str = "TriggerPattern";

/// Edge linking a skill to its trigger.
pub const EDGE_TRIGGERED_BY: &str = "TRIGGERED_BY";

/// Edge linking a trigger to what it watches.
pub const EDGE_WATCHES: &str = "WATCHES";

// ---------------------------------------------------------------------------
// Trigger Types
// ---------------------------------------------------------------------------

/// The type of pattern a trigger uses.
#[derive(Debug, Clone, PartialEq)]
pub enum TriggerPatternType {
    /// Regex match on text input.
    Regex(String),
    /// Glob match on file paths.
    FileGlob(String),
    /// Semantic similarity (embedding cosine > threshold).
    Semantic {
        embedding: Vec<f32>,
        threshold: f32,
    },
    /// Event type match on mutation bus.
    Event {
        event_type: String,
        label_filter: Option<String>,
    },
    /// Graph mutation pattern (new node of type, edge created, etc.)
    GraphMutation {
        mutation_type: MutationType,
        label_filter: Option<String>,
    },
}

/// Types of graph mutations to watch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MutationType {
    NodeCreated,
    NodeUpdated,
    NodeDeleted,
    EdgeCreated,
    EdgeDeleted,
}

impl MutationType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::NodeCreated => "node_created",
            Self::NodeUpdated => "node_updated",
            Self::NodeDeleted => "node_deleted",
            Self::EdgeCreated => "edge_created",
            Self::EdgeDeleted => "edge_deleted",
        }
    }
}

// ---------------------------------------------------------------------------
// TriggerPattern
// ---------------------------------------------------------------------------

/// A trigger pattern node.
#[derive(Debug, Clone)]
pub struct TriggerPattern {
    /// Node id in the graph.
    pub node_id: Option<NodeId>,
    /// The pattern to match.
    pub pattern: TriggerPatternType,
    /// Minimum score for activation (for semantic triggers).
    pub threshold: f64,
    /// Energy — participates in engram lifecycle.
    pub energy: f64,
    /// Number of times this trigger has fired.
    pub fire_count: u64,
    /// The skill this trigger belongs to.
    pub skill_id: Option<String>,
}

impl TriggerPattern {
    pub fn new(pattern: TriggerPatternType, skill_id: Option<String>) -> Self {
        Self {
            node_id: None,
            pattern,
            threshold: 0.5,
            energy: 1.0,
            fire_count: 0,
            skill_id,
        }
    }

    /// Record a successful fire.
    pub fn record_fire(&mut self) {
        self.fire_count += 1;
        self.energy = (self.energy + 0.05).min(1.0);
    }

    /// Apply energy decay (called periodically).
    pub fn decay(&mut self, rate: f64) {
        self.energy *= rate;
    }

    /// Check if trigger should dissolve.
    pub fn should_dissolve(&self, min_energy: f64) -> bool {
        self.energy < min_energy
    }
}

// ---------------------------------------------------------------------------
// Event input (what we evaluate triggers against)
// ---------------------------------------------------------------------------

/// An event to evaluate triggers against.
#[derive(Debug, Clone)]
pub struct TriggerEvent {
    /// Text input (file path, user message, etc.)
    pub text: Option<String>,
    /// File path (if file-related event).
    pub file_path: Option<String>,
    /// Event type from mutation bus.
    pub event_type: Option<String>,
    /// Label of the affected node.
    pub label: Option<String>,
    /// Mutation type (if graph mutation).
    pub mutation_type: Option<MutationType>,
    /// Embedding vector (if semantic matching is available).
    pub embedding: Option<Vec<f32>>,
}

impl TriggerEvent {
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            text: Some(text.into()),
            file_path: None,
            event_type: None,
            label: None,
            mutation_type: None,
            embedding: None,
        }
    }

    pub fn file(path: impl Into<String>) -> Self {
        let path = path.into();
        Self {
            text: Some(path.clone()),
            file_path: Some(path),
            event_type: None,
            label: None,
            mutation_type: None,
            embedding: None,
        }
    }

    pub fn mutation(mutation_type: MutationType, label: Option<String>) -> Self {
        Self {
            text: None,
            file_path: None,
            event_type: None,
            label,
            mutation_type: Some(mutation_type),
            embedding: None,
        }
    }

    pub fn event(event_type: impl Into<String>, label: Option<String>) -> Self {
        Self {
            text: None,
            file_path: None,
            event_type: Some(event_type.into()),
            label,
            mutation_type: None,
            embedding: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Trigger match result
// ---------------------------------------------------------------------------

/// Result of evaluating a trigger.
#[derive(Debug, Clone)]
pub struct TriggerMatch {
    /// Index of the matched trigger.
    pub trigger_idx: usize,
    /// Skill id to activate.
    pub skill_id: Option<String>,
    /// Match score [0, 1].
    pub score: f64,
}

// ---------------------------------------------------------------------------
// Trigger Engine
// ---------------------------------------------------------------------------

/// Evaluates triggers against events.
#[derive(Debug)]
pub struct TriggerEngine {
    triggers: Vec<TriggerPattern>,
    /// Minimum energy for a trigger to be evaluated.
    min_energy: f64,
}

impl TriggerEngine {
    pub fn new(min_energy: f64) -> Self {
        Self {
            triggers: Vec::new(),
            min_energy,
        }
    }

    /// Register a trigger pattern.
    pub fn register(&mut self, trigger: TriggerPattern) -> usize {
        let idx = self.triggers.len();
        self.triggers.push(trigger);
        idx
    }

    /// Get mutable reference to a trigger.
    pub fn get_mut(&mut self, idx: usize) -> Option<&mut TriggerPattern> {
        self.triggers.get_mut(idx)
    }

    /// Evaluate all active triggers against an event.
    /// Returns all matches sorted by score (best first).
    pub fn evaluate(&self, event: &TriggerEvent) -> Vec<TriggerMatch> {
        let mut matches = Vec::new();

        for (idx, trigger) in self.triggers.iter().enumerate() {
            if trigger.energy < self.min_energy {
                continue;
            }

            let score = evaluate_trigger(trigger, event);
            if score >= trigger.threshold {
                matches.push(TriggerMatch {
                    trigger_idx: idx,
                    skill_id: trigger.skill_id.clone(),
                    score,
                });
            }
        }

        matches.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        matches
    }

    /// Apply decay to all triggers.
    pub fn decay_all(&mut self, rate: f64) {
        for trigger in &mut self.triggers {
            trigger.decay(rate);
        }
    }

    /// Remove dissolved triggers. Returns count removed.
    pub fn sweep_dissolved(&mut self) -> usize {
        let before = self.triggers.len();
        self.triggers
            .retain(|t| !t.should_dissolve(self.min_energy * 0.1));
        before - self.triggers.len()
    }

    /// Get active trigger count.
    pub fn active_count(&self) -> usize {
        self.triggers
            .iter()
            .filter(|t| t.energy >= self.min_energy)
            .count()
    }

    /// Total count.
    pub fn total_count(&self) -> usize {
        self.triggers.len()
    }
}

// ---------------------------------------------------------------------------
// Pattern evaluation
// ---------------------------------------------------------------------------

fn evaluate_trigger(trigger: &TriggerPattern, event: &TriggerEvent) -> f64 {
    match &trigger.pattern {
        TriggerPatternType::Regex(pattern) => {
            if let Some(text) = &event.text {
                if text.contains(pattern.as_str()) {
                    1.0
                } else {
                    0.0
                }
            } else {
                0.0
            }
        }

        TriggerPatternType::FileGlob(glob) => {
            if let Some(path) = &event.file_path {
                if glob_match(glob, path) {
                    1.0
                } else {
                    0.0
                }
            } else {
                0.0
            }
        }

        TriggerPatternType::Semantic {
            embedding,
            threshold,
        } => {
            if let Some(event_emb) = &event.embedding {
                let sim = cosine_similarity(embedding, event_emb);
                if sim >= *threshold {
                    sim as f64
                } else {
                    0.0
                }
            } else {
                0.0
            }
        }

        TriggerPatternType::Event {
            event_type,
            label_filter,
        } => {
            if let Some(et) = &event.event_type {
                if et == event_type {
                    if let Some(lf) = label_filter {
                        if event.label.as_ref() == Some(lf) {
                            1.0
                        } else {
                            0.0
                        }
                    } else {
                        1.0
                    }
                } else {
                    0.0
                }
            } else {
                0.0
            }
        }

        TriggerPatternType::GraphMutation {
            mutation_type,
            label_filter,
        } => {
            if event.mutation_type.as_ref() == Some(mutation_type) {
                if let Some(lf) = label_filter {
                    if event.label.as_ref() == Some(lf) {
                        1.0
                    } else {
                        0.0
                    }
                } else {
                    1.0
                }
            } else {
                0.0
            }
        }
    }
}

/// Simple glob matching (supports * and **).
fn glob_match(pattern: &str, path: &str) -> bool {
    if pattern == "**" || pattern == "*" {
        return true;
    }

    // Handle **/*.ext pattern (match any path ending with .ext)
    if pattern.starts_with("**/") {
        let suffix = &pattern[3..];
        if suffix.contains('*') {
            // e.g. **/*.rs → match anything ending in .rs
            if let Some(ext) = suffix.strip_prefix('*') {
                return path.ends_with(ext);
            }
        }
        return path.ends_with(suffix) || path.contains(&format!("/{suffix}"));
    }

    // Handle prefix/** pattern
    if let Some(stripped) = pattern.strip_suffix("/**") {
        return path.starts_with(stripped) || path.starts_with(&format!("{stripped}/"));
    }

    // Handle dir/*.ext pattern (e.g. src/*.rs)
    if let Some(star_pos) = pattern.find('*') {
        let prefix = &pattern[..star_pos];
        let suffix = &pattern[star_pos + 1..];
        if suffix.contains('*') {
            // Multiple stars — simplified: just check prefix
            return path.starts_with(prefix);
        }
        return path.starts_with(prefix) && path.ends_with(suffix);
    }

    pattern == path
}

/// Cosine similarity between two f32 vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }

    dot / (mag_a * mag_b)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regex_trigger() {
        let mut engine = TriggerEngine::new(0.1);
        let mut t = TriggerPattern::new(
            TriggerPatternType::Regex("auth".into()),
            Some("skill_auth".into()),
        );
        t.threshold = 0.5;
        engine.register(t);

        let matches = engine.evaluate(&TriggerEvent::text("modifying auth/manager.rs"));
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].skill_id, Some("skill_auth".into()));
    }

    #[test]
    fn test_file_glob_trigger() {
        let mut engine = TriggerEngine::new(0.1);
        let mut t = TriggerPattern::new(
            TriggerPatternType::FileGlob("src/**/*.rs".into()),
            Some("skill_rust".into()),
        );
        t.threshold = 0.5;
        engine.register(t);

        let matches = engine.evaluate(&TriggerEvent::file("src/api/handlers.rs"));
        assert_eq!(matches.len(), 1);
    }

    #[test]
    fn test_event_trigger() {
        let mut engine = TriggerEngine::new(0.1);
        let mut t = TriggerPattern::new(
            TriggerPatternType::Event {
                event_type: "NodeCreated".into(),
                label_filter: Some("Note".into()),
            },
            Some("skill_notes".into()),
        );
        t.threshold = 0.5;
        engine.register(t);

        let matches = engine.evaluate(&TriggerEvent::event("NodeCreated", Some("Note".into())));
        assert_eq!(matches.len(), 1);

        // Wrong label → no match
        let matches = engine.evaluate(&TriggerEvent::event("NodeCreated", Some("Task".into())));
        assert_eq!(matches.len(), 0);
    }

    #[test]
    fn test_graph_mutation_trigger() {
        let mut engine = TriggerEngine::new(0.1);
        let mut t = TriggerPattern::new(
            TriggerPatternType::GraphMutation {
                mutation_type: MutationType::EdgeCreated,
                label_filter: None,
            },
            Some("skill_fabric".into()),
        );
        t.threshold = 0.5;
        engine.register(t);

        let matches = engine.evaluate(&TriggerEvent::mutation(MutationType::EdgeCreated, None));
        assert_eq!(matches.len(), 1);
    }

    #[test]
    fn test_semantic_trigger() {
        let mut engine = TriggerEngine::new(0.1);
        let emb = vec![1.0, 0.0, 0.0];
        let mut t = TriggerPattern::new(
            TriggerPatternType::Semantic {
                embedding: emb.clone(),
                threshold: 0.9,
            },
            Some("skill_sim".into()),
        );
        t.threshold = 0.5;
        engine.register(t);

        // Similar embedding → match
        let mut event = TriggerEvent::text("anything");
        event.embedding = Some(vec![0.99, 0.1, 0.0]);
        let matches = engine.evaluate(&event);
        assert_eq!(matches.len(), 1);

        // Dissimilar → no match
        let mut event2 = TriggerEvent::text("anything");
        event2.embedding = Some(vec![0.0, 1.0, 0.0]);
        let matches2 = engine.evaluate(&event2);
        assert_eq!(matches2.len(), 0);
    }

    #[test]
    fn test_energy_decay_and_dissolution() {
        let mut engine = TriggerEngine::new(0.1);
        engine.register(TriggerPattern::new(
            TriggerPatternType::Regex("x".into()),
            None,
        ));

        // Decay aggressively
        for _ in 0..100 {
            engine.decay_all(0.9);
        }

        let swept = engine.sweep_dissolved();
        assert_eq!(swept, 1);
        assert_eq!(engine.total_count(), 0);
    }

    #[test]
    fn test_fire_increases_energy() {
        let mut trigger = TriggerPattern::new(TriggerPatternType::Regex("x".into()), None);
        trigger.energy = 0.5;
        trigger.record_fire();
        assert!(trigger.energy > 0.5);
        assert_eq!(trigger.fire_count, 1);
    }

    #[test]
    fn test_glob_matching() {
        assert!(glob_match("**/*.rs", "src/main.rs"));
        assert!(glob_match("src/**", "src/deep/nested/file.ts"));
        assert!(glob_match("*.rs", "main.rs"));
        assert!(!glob_match("*.rs", "main.ts"));
        assert!(glob_match("src/*.rs", "src/main.rs"));
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &c)).abs() < 0.001);
    }
}
