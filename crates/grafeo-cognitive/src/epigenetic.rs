//! # Epigenetic Layer — Layer 5: Cross-Instance Memory
//!
//! Transgenerational memory marks that modulate engram expression across
//! projects and instances. Marks are **immutable** (Copy-on-Write): any
//! mutation produces a new mark with `generation + 1`.
//!
//! ## Core Concepts
//!
//! - **EpigeneticMark** — a modulation directive targeting an `EngramTemplate`.
//!   Amplifies or suppresses engram expression depending on context.
//! - **ExpressionCondition** — predicates evaluated against a `ProjectContext`
//!   to decide whether a mark should be expressed (activated) in a given project.
//! - **Copy-on-Write** — marks are write-once. `with_*` methods return new marks
//!   with incremented generation. The original is never modified.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

// ═══════════════════════════════════════════════════════════════════════════════
// EpigeneticMarkId
// ═══════════════════════════════════════════════════════════════════════════════

/// Unique identifier for an epigenetic mark.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EpigeneticMarkId(pub u64);

static NEXT_MARK_ID: AtomicU64 = AtomicU64::new(1);

impl EpigeneticMarkId {
    /// Allocate a fresh, globally unique mark ID.
    pub fn next() -> Self {
        Self(NEXT_MARK_ID.fetch_add(1, Ordering::Relaxed))
    }
}

impl fmt::Display for EpigeneticMarkId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "epigenetic-mark:{}", self.0)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// EngramTemplate — target pattern for a mark
// ═══════════════════════════════════════════════════════════════════════════════

/// A template describing which engrams a mark targets.
///
/// Rather than pointing to a specific `EngramId` (which is instance-local),
/// the template uses structural descriptors that can match engrams across
/// different project instances.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EngramTemplate {
    /// Labels that the engram's ensemble nodes must carry.
    /// Empty means "match any labels".
    pub required_labels: Vec<String>,

    /// Minimum spectral similarity (cosine) to match. 0.0 = match all.
    pub min_spectral_similarity: f64,

    /// Optional text pattern in the engram's source description.
    pub description_pattern: Option<String>,
}

impl EngramTemplate {
    /// Create a template that matches any engram.
    pub fn any() -> Self {
        Self {
            required_labels: Vec::new(),
            min_spectral_similarity: 0.0,
            description_pattern: None,
        }
    }

    /// Create a template targeting engrams with specific labels.
    pub fn with_labels(labels: Vec<String>) -> Self {
        Self {
            required_labels: labels,
            min_spectral_similarity: 0.0,
            description_pattern: None,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ExpressionCondition — predicate for mark activation
// ═══════════════════════════════════════════════════════════════════════════════

/// A condition that must be met in the target project for a mark to be expressed.
///
/// Marks never apply blindly: each mark evaluates its expression conditions
/// against the `ProjectContext`. A non-pertinent mark is silently ignored
/// (log::debug).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ExpressionCondition {
    /// The project must contain a file with this path/name.
    HasFile(String),

    /// The project's graph must have at least one node with this label.
    HasLabel(String),

    /// The project's graph must have at least this many nodes.
    MinNodes(usize),

    /// A custom predicate identified by a key-value pair.
    /// The key is the predicate name, the value is the expected value.
    Custom {
        /// Predicate identifier (e.g., "language", "framework").
        key: String,
        /// Expected value (e.g., "rust", "actix-web").
        value: String,
    },
}

// ═══════════════════════════════════════════════════════════════════════════════
// ProjectContext — the context against which conditions are evaluated
// ═══════════════════════════════════════════════════════════════════════════════

/// Context of the target project, used to evaluate expression conditions.
///
/// This is a lightweight snapshot of the project's state, not a live reference.
#[derive(Debug, Clone, Default)]
pub struct ProjectContext {
    /// Files present in the project (paths relative to project root).
    pub files: HashSet<String>,

    /// Labels present on nodes in the project graph.
    pub labels: HashSet<String>,

    /// Total number of nodes in the project graph.
    pub node_count: usize,

    /// Custom properties of the project (language, framework, etc.).
    pub custom_properties: std::collections::HashMap<String, String>,
}

impl ProjectContext {
    /// Create a new empty project context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: add a file to the context.
    pub fn with_file(mut self, file: impl Into<String>) -> Self {
        self.files.insert(file.into());
        self
    }

    /// Builder: add a label to the context.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.labels.insert(label.into());
        self
    }

    /// Builder: set the node count.
    pub fn with_node_count(mut self, count: usize) -> Self {
        self.node_count = count;
        self
    }

    /// Builder: add a custom property.
    pub fn with_custom(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.custom_properties.insert(key.into(), value.into());
        self
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// EpigeneticMark — the core immutable structure
// ═══════════════════════════════════════════════════════════════════════════════

/// An epigenetic mark that modulates engram expression across instances.
///
/// # Immutability (Copy-on-Write)
///
/// `EpigeneticMark` is **immutable after creation**. All `with_*` methods
/// consume `&self` and return a **new** mark with `generation + 1`. The
/// original mark is never modified.
///
/// This design ensures thread safety (marks are write-once) and preserves
/// the full lineage of modifications via the `generation` counter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EpigeneticMark {
    /// Unique identifier for this mark.
    id: EpigeneticMarkId,

    /// Which engram pattern this mark targets.
    target: EngramTemplate,

    /// Modulation factor ∈ [-1.0, +1.0].
    /// - Negative = suppression (reduce engram strength)
    /// - Positive = amplification (boost engram strength)
    /// - 0.0 = neutral (no effect)
    modulation: f64,

    /// Whether this mark can be transmitted to child/sibling projects.
    transmissible: bool,

    /// Generation counter. Starts at 0. Incremented on each CoW mutation.
    /// Higher generation = weaker inherited signal (transgenerational decay).
    generation: u32,

    /// Conditions that must ALL be met for this mark to be expressed.
    /// Empty = always expressed (unconditional mark).
    expression_conditions: Vec<ExpressionCondition>,
}

impl EpigeneticMark {
    /// Create a new epigenetic mark (generation 0).
    ///
    /// # Arguments
    ///
    /// * `target` — which engram template this mark modulates
    /// * `modulation` — modulation factor, clamped to [-1.0, +1.0]
    /// * `transmissible` — whether this mark propagates to other projects
    /// * `expression_conditions` — conditions for activation in a target project
    pub fn new(
        target: EngramTemplate,
        modulation: f64,
        transmissible: bool,
        expression_conditions: Vec<ExpressionCondition>,
    ) -> Self {
        Self {
            id: EpigeneticMarkId::next(),
            target,
            modulation: modulation.clamp(-1.0, 1.0),
            transmissible,
            generation: 0,
            expression_conditions,
        }
    }

    // ─── Accessors (read-only) ─────────────────────────────────────────────

    /// Returns the mark's unique identifier.
    pub fn id(&self) -> EpigeneticMarkId {
        self.id
    }

    /// Returns the target engram template.
    pub fn target(&self) -> &EngramTemplate {
        &self.target
    }

    /// Returns the modulation factor ∈ [-1.0, +1.0].
    pub fn modulation(&self) -> f64 {
        self.modulation
    }

    /// Returns whether this mark is transmissible across projects.
    pub fn transmissible(&self) -> bool {
        self.transmissible
    }

    /// Returns the generation counter.
    pub fn generation(&self) -> u32 {
        self.generation
    }

    /// Returns the expression conditions.
    pub fn expression_conditions(&self) -> &[ExpressionCondition] {
        &self.expression_conditions
    }

    // ─── Condition evaluation ──────────────────────────────────────────────

    /// Evaluate all expression conditions against a project context.
    ///
    /// Returns `true` if **all** conditions are satisfied (AND semantics).
    /// An empty condition list returns `true` (unconditional mark).
    ///
    /// Non-matching marks are silently ignored with a debug log.
    pub fn evaluate_conditions(&self, ctx: &ProjectContext) -> bool {
        if self.expression_conditions.is_empty() {
            return true;
        }

        for condition in &self.expression_conditions {
            let met = match condition {
                ExpressionCondition::HasFile(file) => ctx.files.contains(file.as_str()),
                ExpressionCondition::HasLabel(label) => ctx.labels.contains(label.as_str()),
                ExpressionCondition::MinNodes(min) => ctx.node_count >= *min,
                ExpressionCondition::Custom { key, value } => {
                    ctx.custom_properties.get(key).is_some_and(|v| v == value)
                }
            };

            if !met {
                tracing::debug!(
                    mark_id = %self.id,
                    condition = ?condition,
                    "epigenetic mark condition not met — mark will not be expressed"
                );
                return false;
            }
        }

        true
    }

    /// Compute the effective modulation, accounting for transgenerational decay.
    ///
    /// Each generation reduces the modulation by a decay factor (default 0.8).
    /// Generation 0 = full modulation, generation 1 = 80%, generation 2 = 64%, etc.
    pub fn effective_modulation(&self) -> f64 {
        self.effective_modulation_with_decay(0.8)
    }

    /// Compute the effective modulation with a custom decay factor per generation.
    pub fn effective_modulation_with_decay(&self, decay_per_generation: f64) -> f64 {
        self.modulation * decay_per_generation.powi(self.generation as i32)
    }

    // ─── Copy-on-Write mutations ───────────────────────────────────────────

    /// Create a new mark with a different modulation (CoW).
    ///
    /// Returns a **new** mark with `generation + 1`. The original is unchanged.
    pub fn with_modulation(&self, new_modulation: f64) -> Self {
        Self {
            id: EpigeneticMarkId::next(),
            target: self.target.clone(),
            modulation: new_modulation.clamp(-1.0, 1.0),
            transmissible: self.transmissible,
            generation: self.generation + 1,
            expression_conditions: self.expression_conditions.clone(),
        }
    }

    /// Create a new mark with additional expression conditions (CoW).
    ///
    /// Returns a **new** mark with `generation + 1`. The original is unchanged.
    pub fn with_additional_conditions(&self, conditions: Vec<ExpressionCondition>) -> Self {
        let mut new_conditions = self.expression_conditions.clone();
        new_conditions.extend(conditions);
        Self {
            id: EpigeneticMarkId::next(),
            target: self.target.clone(),
            modulation: self.modulation,
            transmissible: self.transmissible,
            generation: self.generation + 1,
            expression_conditions: new_conditions,
        }
    }

    /// Create a new mark with transmissibility changed (CoW).
    ///
    /// Returns a **new** mark with `generation + 1`. The original is unchanged.
    pub fn with_transmissible(&self, transmissible: bool) -> Self {
        Self {
            id: EpigeneticMarkId::next(),
            target: self.target.clone(),
            modulation: self.modulation,
            transmissible,
            generation: self.generation + 1,
            expression_conditions: self.expression_conditions.clone(),
        }
    }

    /// Create a transmitted copy for a child/sibling project (CoW).
    ///
    /// Returns `None` if the mark is not transmissible.
    /// Returns a **new** mark with `generation + 1`. The original is unchanged.
    pub fn transmit(&self) -> Option<Self> {
        if !self.transmissible {
            return None;
        }
        Some(Self {
            id: EpigeneticMarkId::next(),
            target: self.target.clone(),
            modulation: self.modulation,
            transmissible: self.transmissible,
            generation: self.generation + 1,
            expression_conditions: self.expression_conditions.clone(),
        })
    }
}

impl fmt::Display for EpigeneticMark {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "EpigeneticMark({}, gen={}, mod={:.2}, tx={})",
            self.id, self.generation, self.modulation, self.transmissible
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CognitiveStorage persistence helpers
// ═══════════════════════════════════════════════════════════════════════════════

/// Label used for EpigeneticMark nodes in CognitiveStorage.
pub const LABEL_EPIGENETIC_MARK: &str = "EpigeneticMark";

impl EpigeneticMark {
    /// Persist this mark to a CognitiveStorage backend.
    ///
    /// Creates a new `:EpigeneticMark` node. Since marks are immutable,
    /// there is no `update` — a CoW mutation creates a new node.
    pub fn persist(&self, storage: &dyn crate::engram::traits::CognitiveStorage) -> grafeo_common::types::NodeId {
        use grafeo_common::types::Value;
        use std::collections::HashMap;

        let mut props = HashMap::new();
        props.insert("mark_id".to_string(), Value::Int64(self.id.0 as i64));
        props.insert("modulation".to_string(), Value::Float64(self.modulation));
        props.insert(
            "transmissible".to_string(),
            Value::Bool(self.transmissible),
        );
        props.insert(
            "generation".to_string(),
            Value::Int64(i64::from(self.generation)),
        );

        // Serialize target and conditions as JSON strings for storage
        if let Ok(target_json) = serde_json::to_string(&self.target) {
            props.insert("target".to_string(), Value::String(target_json.into()));
        }
        if let Ok(conditions_json) = serde_json::to_string(&self.expression_conditions) {
            props.insert(
                "expression_conditions".to_string(),
                Value::String(conditions_json.into()),
            );
        }

        storage.create_node(LABEL_EPIGENETIC_MARK, &props)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_mark() -> EpigeneticMark {
        EpigeneticMark::new(
            EngramTemplate::with_labels(vec!["Function".into(), "File".into()]),
            0.75,
            true,
            vec![
                ExpressionCondition::HasFile("filter.rs".into()),
                ExpressionCondition::MinNodes(10),
            ],
        )
    }

    // ─── Step 1: Struct definitions compile with serde ─────────────────────

    #[test]
    fn test_serde_roundtrip_mark() {
        let mark = sample_mark();
        let json = serde_json::to_string(&mark).unwrap();
        let deserialized: EpigeneticMark = serde_json::from_str(&json).unwrap();
        assert_eq!(mark.modulation(), deserialized.modulation());
        assert_eq!(mark.generation(), deserialized.generation());
        assert_eq!(mark.transmissible(), deserialized.transmissible());
        assert_eq!(mark.target(), deserialized.target());
        assert_eq!(
            mark.expression_conditions(),
            deserialized.expression_conditions()
        );
    }

    #[test]
    fn test_serde_roundtrip_conditions() {
        let conditions = vec![
            ExpressionCondition::HasFile("main.rs".into()),
            ExpressionCondition::HasLabel("Function".into()),
            ExpressionCondition::MinNodes(42),
            ExpressionCondition::Custom {
                key: "language".into(),
                value: "rust".into(),
            },
        ];
        let json = serde_json::to_string(&conditions).unwrap();
        let deserialized: Vec<ExpressionCondition> = serde_json::from_str(&json).unwrap();
        assert_eq!(conditions, deserialized);
    }

    #[test]
    fn test_modulation_clamped() {
        let mark = EpigeneticMark::new(EngramTemplate::any(), 5.0, false, vec![]);
        assert_eq!(mark.modulation(), 1.0);

        let mark2 = EpigeneticMark::new(EngramTemplate::any(), -3.0, false, vec![]);
        assert_eq!(mark2.modulation(), -1.0);
    }

    // ─── Step 2: evaluate_conditions tests ─────────────────────────────────

    #[test]
    fn test_evaluate_conditions_has_file_present() {
        let mark = EpigeneticMark::new(
            EngramTemplate::any(),
            0.5,
            true,
            vec![ExpressionCondition::HasFile("filter.rs".into())],
        );

        let ctx = ProjectContext::new().with_file("filter.rs");
        assert!(mark.evaluate_conditions(&ctx));
    }

    #[test]
    fn test_evaluate_conditions_has_file_absent() {
        let mark = EpigeneticMark::new(
            EngramTemplate::any(),
            0.5,
            true,
            vec![ExpressionCondition::HasFile("filter.rs".into())],
        );

        let ctx = ProjectContext::new().with_file("main.rs");
        assert!(!mark.evaluate_conditions(&ctx));
    }

    #[test]
    fn test_evaluate_conditions_has_label() {
        let mark = EpigeneticMark::new(
            EngramTemplate::any(),
            0.5,
            false,
            vec![ExpressionCondition::HasLabel("Function".into())],
        );

        let ctx_with = ProjectContext::new().with_label("Function");
        assert!(mark.evaluate_conditions(&ctx_with));

        let ctx_without = ProjectContext::new().with_label("Struct");
        assert!(!mark.evaluate_conditions(&ctx_without));
    }

    #[test]
    fn test_evaluate_conditions_min_nodes() {
        let mark = EpigeneticMark::new(
            EngramTemplate::any(),
            0.5,
            false,
            vec![ExpressionCondition::MinNodes(10)],
        );

        let ctx_enough = ProjectContext::new().with_node_count(15);
        assert!(mark.evaluate_conditions(&ctx_enough));

        let ctx_exact = ProjectContext::new().with_node_count(10);
        assert!(mark.evaluate_conditions(&ctx_exact));

        let ctx_insufficient = ProjectContext::new().with_node_count(5);
        assert!(!mark.evaluate_conditions(&ctx_insufficient));
    }

    #[test]
    fn test_evaluate_conditions_custom() {
        let mark = EpigeneticMark::new(
            EngramTemplate::any(),
            0.5,
            false,
            vec![ExpressionCondition::Custom {
                key: "language".into(),
                value: "rust".into(),
            }],
        );

        let ctx_match = ProjectContext::new().with_custom("language", "rust");
        assert!(mark.evaluate_conditions(&ctx_match));

        let ctx_mismatch = ProjectContext::new().with_custom("language", "python");
        assert!(!mark.evaluate_conditions(&ctx_mismatch));

        let ctx_missing = ProjectContext::new();
        assert!(!mark.evaluate_conditions(&ctx_missing));
    }

    #[test]
    fn test_evaluate_conditions_all_must_match() {
        let mark = EpigeneticMark::new(
            EngramTemplate::any(),
            0.5,
            false,
            vec![
                ExpressionCondition::HasFile("filter.rs".into()),
                ExpressionCondition::MinNodes(5),
            ],
        );

        // Both satisfied
        let ctx_both = ProjectContext::new()
            .with_file("filter.rs")
            .with_node_count(10);
        assert!(mark.evaluate_conditions(&ctx_both));

        // Only file
        let ctx_file_only = ProjectContext::new()
            .with_file("filter.rs")
            .with_node_count(2);
        assert!(!mark.evaluate_conditions(&ctx_file_only));

        // Only nodes
        let ctx_nodes_only = ProjectContext::new().with_node_count(10);
        assert!(!mark.evaluate_conditions(&ctx_nodes_only));
    }

    #[test]
    fn test_evaluate_conditions_empty_is_unconditional() {
        let mark = EpigeneticMark::new(EngramTemplate::any(), 0.5, false, vec![]);
        let ctx = ProjectContext::new();
        assert!(mark.evaluate_conditions(&ctx));
    }

    // ─── Step 3: Copy-on-Write tests ───────────────────────────────────────

    #[test]
    fn test_cow_with_modulation() {
        let original = sample_mark();
        let original_id = original.id();
        let original_gen = original.generation();
        let original_mod = original.modulation();

        let mutated = original.with_modulation(0.3);

        // Original unchanged
        assert_eq!(original.id(), original_id);
        assert_eq!(original.generation(), original_gen);
        assert_eq!(original.modulation(), original_mod);

        // Mutated is new
        assert_ne!(mutated.id(), original_id);
        assert_eq!(mutated.generation(), original_gen + 1);
        assert!((mutated.modulation() - 0.3).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cow_with_transmissible() {
        let original = sample_mark();
        assert!(original.transmissible());

        let mutated = original.with_transmissible(false);
        assert!(original.transmissible()); // original unchanged
        assert!(!mutated.transmissible());
        assert_eq!(mutated.generation(), original.generation() + 1);
    }

    #[test]
    fn test_cow_with_additional_conditions() {
        let original = EpigeneticMark::new(
            EngramTemplate::any(),
            0.5,
            true,
            vec![ExpressionCondition::HasFile("main.rs".into())],
        );
        assert_eq!(original.expression_conditions().len(), 1);

        let mutated = original.with_additional_conditions(vec![
            ExpressionCondition::MinNodes(100),
        ]);

        assert_eq!(original.expression_conditions().len(), 1); // original unchanged
        assert_eq!(mutated.expression_conditions().len(), 2);
        assert_eq!(mutated.generation(), original.generation() + 1);
    }

    #[test]
    fn test_cow_transmit() {
        let original = EpigeneticMark::new(EngramTemplate::any(), 0.8, true, vec![]);
        let transmitted = original.transmit().expect("should transmit");
        assert_eq!(transmitted.generation(), original.generation() + 1);
        assert_ne!(transmitted.id(), original.id());

        // Non-transmissible mark
        let non_tx = EpigeneticMark::new(EngramTemplate::any(), 0.5, false, vec![]);
        assert!(non_tx.transmit().is_none());
    }

    #[test]
    fn test_effective_modulation_decay() {
        let mark = EpigeneticMark::new(EngramTemplate::any(), 1.0, true, vec![]);
        assert!((mark.effective_modulation() - 1.0).abs() < f64::EPSILON);

        let gen1 = mark.transmit().unwrap();
        assert!((gen1.effective_modulation() - 0.8).abs() < f64::EPSILON);

        let gen2 = gen1.transmit().unwrap();
        assert!((gen2.effective_modulation() - 0.64).abs() < f64::EPSILON);
    }

    #[test]
    fn test_generation_increments_chain() {
        let g0 = EpigeneticMark::new(EngramTemplate::any(), 0.5, true, vec![]);
        assert_eq!(g0.generation(), 0);

        let g1 = g0.with_modulation(0.6);
        assert_eq!(g1.generation(), 1);

        let g2 = g1.with_transmissible(false);
        assert_eq!(g2.generation(), 2);

        let g3 = g0.with_additional_conditions(vec![ExpressionCondition::MinNodes(5)]);
        assert_eq!(g3.generation(), 1); // branched from g0, not g2
    }

    #[test]
    fn test_display() {
        let mark = EpigeneticMark::new(EngramTemplate::any(), 0.75, true, vec![]);
        let display = format!("{mark}");
        assert!(display.contains("gen=0"));
        assert!(display.contains("mod=0.75"));
        assert!(display.contains("tx=true"));
    }
}
