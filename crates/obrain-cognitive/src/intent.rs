//! # Intent Detection
//!
//! Graph-based intent classification that categorizes user inputs by evaluating
//! signals directly on the graph state — no embeddings or LLM required.
//!
//! ## How it works
//!
//! Each `:IntentPattern` node defines:
//! - A category (query, action, exploration, creation, debugging, review, planning)
//! - A set of signals (conditions evaluated on the graph)
//! - Weights for each signal
//!
//! The engine evaluates all signals for each pattern and returns the category
//! with the highest weighted score.
//!
//! ## Signal Types
//!
//! - `Keyword` — words present in the input
//! - `EntityMentioned` — a graph entity type is referenced
//! - `RecentActivity` — recent mutations on certain node types
//! - `SessionPhase` — current phase of the session
//! - `CommunityActive` — a community was recently touched
//!
//! ## Adaptation
//!
//! Signal weights are KernelParams — they adjust via the feedback loop
//! when detected intents prove incorrect (agent changes strategy mid-task).

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Label for intent pattern nodes in the graph.
pub const LABEL_INTENT_PATTERN: &str = "IntentPattern";

// ---------------------------------------------------------------------------
// Intent Categories
// ---------------------------------------------------------------------------

/// High-level intent categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntentCategory {
    /// User is asking a question about existing state.
    Query,
    /// User wants to perform a specific action/modification.
    Action,
    /// User is exploring/browsing without a specific goal.
    Exploration,
    /// User wants to create something new.
    Creation,
    /// User is debugging a problem.
    Debugging,
    /// User is reviewing existing work.
    Review,
    /// User is planning future work.
    Planning,
}

impl IntentCategory {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Query => "query",
            Self::Action => "action",
            Self::Exploration => "exploration",
            Self::Creation => "creation",
            Self::Debugging => "debugging",
            Self::Review => "review",
            Self::Planning => "planning",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "query" => Some(Self::Query),
            "action" => Some(Self::Action),
            "exploration" => Some(Self::Exploration),
            "creation" => Some(Self::Creation),
            "debugging" => Some(Self::Debugging),
            "review" => Some(Self::Review),
            "planning" => Some(Self::Planning),
            _ => None,
        }
    }

    /// All categories for iteration.
    pub fn all() -> &'static [Self] {
        &[
            Self::Query,
            Self::Action,
            Self::Exploration,
            Self::Creation,
            Self::Debugging,
            Self::Review,
            Self::Planning,
        ]
    }
}

// ---------------------------------------------------------------------------
// Signals
// ---------------------------------------------------------------------------

/// A signal that contributes to intent detection.
#[derive(Debug, Clone)]
pub enum Signal {
    /// Input contains one of these keywords.
    Keyword(Vec<String>),
    /// An entity type is mentioned in the input.
    EntityMentioned(String),
    /// Recent mutations exist for this node type within window_secs.
    RecentActivity { node_type: String, window_secs: u64 },
    /// Session is in this phase.
    SessionPhase(SessionPhase),
    /// A community was touched recently.
    CommunityActive(u64),
}

/// Session phases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionPhase {
    Start,
    Middle,
    End,
}

// ---------------------------------------------------------------------------
// IntentPattern
// ---------------------------------------------------------------------------

/// A pattern that matches an intent category.
#[derive(Debug, Clone)]
pub struct IntentPattern {
    /// Which category this pattern detects.
    pub category: IntentCategory,
    /// Signals with their weights.
    pub signals: Vec<(Signal, f64)>,
    /// Energy — participates in engram lifecycle.
    pub energy: f64,
}

impl IntentPattern {
    pub fn new(category: IntentCategory, signals: Vec<(Signal, f64)>) -> Self {
        Self {
            category,
            signals,
            energy: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Session Context (provided by the caller)
// ---------------------------------------------------------------------------

/// Context provided for intent detection.
#[derive(Debug, Clone, Default)]
pub struct SessionContext {
    /// Current session phase.
    pub phase: Option<SessionPhase>,
    /// Entity types mentioned in the input (pre-extracted).
    pub mentioned_entities: Vec<String>,
    /// Recently active node types (type → count of recent mutations).
    pub recent_activity: HashMap<String, u64>,
    /// Recently touched community ids.
    pub active_communities: Vec<u64>,
}

// ---------------------------------------------------------------------------
// Intent Result
// ---------------------------------------------------------------------------

/// Result of intent detection.
#[derive(Debug, Clone)]
pub struct DetectedIntent {
    /// The detected category.
    pub category: IntentCategory,
    /// Confidence score [0, 1].
    pub confidence: f64,
    /// Which signals matched.
    pub matched_signals: Vec<String>,
    /// Runner-up category (for ambiguous cases).
    pub runner_up: Option<(IntentCategory, f64)>,
}

// ---------------------------------------------------------------------------
// Intent Engine
// ---------------------------------------------------------------------------

/// The intent detection engine.
#[derive(Debug)]
pub struct IntentEngine {
    patterns: Vec<IntentPattern>,
}

impl IntentEngine {
    /// Create a new engine with default patterns.
    pub fn new() -> Self {
        Self {
            patterns: default_patterns(),
        }
    }

    /// Create an empty engine (for custom pattern loading).
    pub fn empty() -> Self {
        Self {
            patterns: Vec::new(),
        }
    }

    /// Add a pattern.
    pub fn add_pattern(&mut self, pattern: IntentPattern) {
        self.patterns.push(pattern);
    }

    /// Detect intent from input text and session context.
    pub fn detect(&self, input: &str, context: &SessionContext) -> DetectedIntent {
        let input_lower = input.to_lowercase();
        let mut scores: HashMap<IntentCategory, (f64, Vec<String>)> = HashMap::new();

        for pattern in &self.patterns {
            if pattern.energy < 0.05 {
                continue; // skip dead patterns
            }

            let mut pattern_score = 0.0;
            let mut matched = Vec::new();

            for (signal, weight) in &pattern.signals {
                let signal_matches = evaluate_signal(signal, &input_lower, context);
                if signal_matches {
                    pattern_score += weight;
                    matched.push(describe_signal(signal));
                }
            }

            if pattern_score > 0.0 {
                let entry = scores.entry(pattern.category).or_insert((0.0, Vec::new()));
                entry.0 += pattern_score * pattern.energy;
                entry.1.extend(matched);
            }
        }

        // Find top 2 categories
        let mut ranked: Vec<_> = scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1 .0.partial_cmp(&a.1 .0).unwrap_or(std::cmp::Ordering::Equal));

        if ranked.is_empty() {
            return DetectedIntent {
                category: IntentCategory::Query, // default
                confidence: 0.0,
                matched_signals: Vec::new(),
                runner_up: None,
            };
        }

        let top = &ranked[0];
        let total_score: f64 = ranked.iter().map(|(_, (s, _))| s).sum();
        let confidence = if total_score > 0.0 {
            (top.1 .0 / total_score).min(1.0)
        } else {
            0.0
        };

        let runner_up = ranked.get(1).map(|(cat, (score, _))| {
            let rc = if total_score > 0.0 {
                (score / total_score).min(1.0)
            } else {
                0.0
            };
            (*cat, rc)
        });

        DetectedIntent {
            category: top.0,
            confidence,
            matched_signals: top.1 .1.clone(),
            runner_up,
        }
    }
}

impl Default for IntentEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Signal Evaluation
// ---------------------------------------------------------------------------

fn evaluate_signal(signal: &Signal, input_lower: &str, context: &SessionContext) -> bool {
    match signal {
        Signal::Keyword(words) => words.iter().any(|w| input_lower.contains(w.as_str())),
        Signal::EntityMentioned(entity_type) => {
            context.mentioned_entities.iter().any(|e| e == entity_type)
        }
        Signal::RecentActivity {
            node_type,
            window_secs: _,
        } => context
            .recent_activity
            .get(node_type)
            .map_or(false, |&count| count > 0),
        Signal::SessionPhase(phase) => context.phase.as_ref() == Some(phase),
        Signal::CommunityActive(id) => context.active_communities.contains(id),
    }
}

fn describe_signal(signal: &Signal) -> String {
    match signal {
        Signal::Keyword(words) => format!("keyword:{}", words.join("|")),
        Signal::EntityMentioned(t) => format!("entity:{t}"),
        Signal::RecentActivity { node_type, .. } => format!("activity:{node_type}"),
        Signal::SessionPhase(p) => format!("phase:{p:?}"),
        Signal::CommunityActive(id) => format!("community:{id}"),
    }
}

// ---------------------------------------------------------------------------
// Default Patterns
// ---------------------------------------------------------------------------

fn default_patterns() -> Vec<IntentPattern> {
    vec![
        IntentPattern::new(
            IntentCategory::Debugging,
            vec![
                (
                    Signal::Keyword(vec![
                        "error".into(),
                        "bug".into(),
                        "fail".into(),
                        "crash".into(),
                        "panic".into(),
                        "broken".into(),
                        "fix".into(),
                        "why".into(),
                        "wrong".into(),
                    ]),
                    1.0,
                ),
                (
                    Signal::RecentActivity {
                        node_type: "Error".into(),
                        window_secs: 300,
                    },
                    0.5,
                ),
            ],
        ),
        IntentPattern::new(
            IntentCategory::Query,
            vec![
                (
                    Signal::Keyword(vec![
                        "what".into(),
                        "how".into(),
                        "where".into(),
                        "which".into(),
                        "show".into(),
                        "list".into(),
                        "get".into(),
                        "status".into(),
                    ]),
                    0.8,
                ),
                (Signal::SessionPhase(SessionPhase::Start), 0.3),
            ],
        ),
        IntentPattern::new(
            IntentCategory::Creation,
            vec![
                (
                    Signal::Keyword(vec![
                        "create".into(),
                        "add".into(),
                        "new".into(),
                        "implement".into(),
                        "build".into(),
                        "write".into(),
                        "generate".into(),
                    ]),
                    1.0,
                ),
                (Signal::EntityMentioned("Plan".into()), 0.4),
            ],
        ),
        IntentPattern::new(
            IntentCategory::Action,
            vec![
                (
                    Signal::Keyword(vec![
                        "update".into(),
                        "delete".into(),
                        "change".into(),
                        "modify".into(),
                        "move".into(),
                        "rename".into(),
                        "run".into(),
                        "execute".into(),
                        "deploy".into(),
                    ]),
                    1.0,
                ),
            ],
        ),
        IntentPattern::new(
            IntentCategory::Exploration,
            vec![
                (
                    Signal::Keyword(vec![
                        "explore".into(),
                        "browse".into(),
                        "look".into(),
                        "search".into(),
                        "find".into(),
                        "discover".into(),
                    ]),
                    0.8,
                ),
                (Signal::SessionPhase(SessionPhase::Start), 0.2),
            ],
        ),
        IntentPattern::new(
            IntentCategory::Review,
            vec![
                (
                    Signal::Keyword(vec![
                        "review".into(),
                        "check".into(),
                        "audit".into(),
                        "verify".into(),
                        "validate".into(),
                        "compare".into(),
                        "diff".into(),
                    ]),
                    1.0,
                ),
                (Signal::EntityMentioned("Commit".into()), 0.5),
            ],
        ),
        IntentPattern::new(
            IntentCategory::Planning,
            vec![
                (
                    Signal::Keyword(vec![
                        "plan".into(),
                        "schedule".into(),
                        "roadmap".into(),
                        "milestone".into(),
                        "priority".into(),
                        "next".into(),
                        "organize".into(),
                    ]),
                    1.0,
                ),
                (Signal::EntityMentioned("Plan".into()), 0.6),
                (Signal::EntityMentioned("Task".into()), 0.4),
            ],
        ),
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_debugging_intent() {
        let engine = IntentEngine::new();
        let ctx = SessionContext::default();
        let result = engine.detect("why is this test failing with a panic?", &ctx);
        assert_eq!(result.category, IntentCategory::Debugging);
        assert!(result.confidence > 0.3);
    }

    #[test]
    fn test_creation_intent() {
        let engine = IntentEngine::new();
        let ctx = SessionContext::default();
        // Use clear creation keywords without ambiguity
        let result = engine.detect("implement a new authentication module", &ctx);
        assert_eq!(result.category, IntentCategory::Creation);
    }

    #[test]
    fn test_query_intent() {
        let engine = IntentEngine::new();
        let ctx = SessionContext::default();
        // Use clear query keywords without "plan" (which triggers Planning)
        let result = engine.detect("show me the list of active tasks", &ctx);
        assert_eq!(result.category, IntentCategory::Query);
    }

    #[test]
    fn test_planning_with_entity_context() {
        let engine = IntentEngine::new();
        let ctx = SessionContext {
            mentioned_entities: vec!["Plan".into(), "Task".into()],
            ..Default::default()
        };
        let result = engine.detect("let's plan the next milestone", &ctx);
        assert_eq!(result.category, IntentCategory::Planning);
        assert!(result.confidence > 0.4);
    }

    #[test]
    fn test_exploration_intent() {
        let engine = IntentEngine::new();
        let ctx = SessionContext {
            phase: Some(SessionPhase::Start),
            ..Default::default()
        };
        let result = engine.detect("let me explore the codebase", &ctx);
        assert_eq!(result.category, IntentCategory::Exploration);
    }

    #[test]
    fn test_ambiguous_with_runner_up() {
        let engine = IntentEngine::new();
        let ctx = SessionContext::default();
        // "create" = creation, "plan" = planning -> ambiguous
        let result = engine.detect("create a plan", &ctx);
        assert!(result.runner_up.is_some());
    }

    #[test]
    fn test_empty_input() {
        let engine = IntentEngine::new();
        let ctx = SessionContext::default();
        let result = engine.detect("", &ctx);
        // No signals match -> defaults to Query with 0 confidence
        assert_eq!(result.confidence, 0.0);
    }

    #[test]
    fn test_dead_pattern_ignored() {
        let mut engine = IntentEngine::empty();
        let mut pattern = IntentPattern::new(
            IntentCategory::Debugging,
            vec![(Signal::Keyword(vec!["error".into()]), 1.0)],
        );
        pattern.energy = 0.01; // below threshold
        engine.add_pattern(pattern);

        let ctx = SessionContext::default();
        let result = engine.detect("error occurred", &ctx);
        assert_eq!(result.confidence, 0.0); // pattern ignored
    }
}
