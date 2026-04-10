//! # Cognitive Persona — Database-native knowledge lens
//!
//! A Persona is a **weighted projection** of the shared graph. It doesn't
//! copy anything — it biases the spreading activation so that the same
//! graph yields different recall results depending on who asks.
//!
//! ## Core idea: KNOWS relations as activation bias
//!
//! ```text
//! Persona "Traffic Law"
//!   KNOWS(vitesse, 0.9)
//!   KNOWS(amende, 0.85)
//!   KNOWS(permis, 0.7)
//!
//! Persona "Labor Law"
//!   KNOWS(licenciement, 0.9)
//!   KNOWS(contrat, 0.8)
//!   KNOWS(prud'hommes, 0.75)
//! ```
//!
//! When a cue like "excès" activates the graph:
//! - Through "Traffic Law": vitesse gets extra energy → traffic cluster lights up
//! - Through "Labor Law": same graph, same synapses, but no boost → traffic
//!   cluster stays dim, labor cluster responds normally
//!
//! ## Multi-client benefit
//!
//! ```text
//! Agent A (CLI) ──→ recall(cues, persona=Traffic) → feedback(Rewarded)
//!                      │                                │
//!                      │  ← shared synapses reinforced   │
//!                      │  ← shared energy boosted        │
//!                      ▼                                ▼
//! Agent B (HTTP) ──→ recall(cues, persona=Traffic) → better results!
//! ```
//!
//! The persona's KNOWS weights are the ONLY per-persona state. Everything
//! else (synapses, energy, stigmergy) is shared and benefits all clients.
//!
//! ## Multi-potential
//!
//! A single persona can evolve its KNOWS over time, discovering new regions
//! of the graph through successful queries. A persona that starts narrow
//! ("only traffic law") can organically expand if rewarded on adjacent
//! topics ("insurance law" → KNOWS weight grows from 0.0 to 0.5).

use dashmap::DashMap;
use obrain_common::types::NodeId;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

#[cfg(feature = "synapse")]
use crate::activation::{ActivationMap, ActivationSource, SpreadConfig, spread};
#[cfg(feature = "energy")]
use crate::energy::EnergyStore;
#[cfg(feature = "synapse")]
use crate::synapse::SynapseStore;

// ═══════════════════════════════════════════════════════════════════════════
// PersonaId
// ═══════════════════════════════════════════════════════════════════════════

/// Unique identifier for a persona within the cognitive engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PersonaId(pub u64);

// ═══════════════════════════════════════════════════════════════════════════
// CognitivePersona — a weighted lens on the shared graph
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for persona behavior.
#[derive(Debug, Clone)]
pub struct PersonaConfig {
    /// Multiplier applied to KNOWS weights during activation seeding.
    /// Higher = persona has stronger influence on recall.
    /// Default: 0.5 (a KNOWS(node, 1.0) adds 0.5 extra energy to that seed).
    pub activation_bias: f64,

    /// How much KNOWS weight increases on rewarded nodes.
    /// Default: 0.1
    pub reward_knows_delta: f64,

    /// How much KNOWS weight decreases on penalized nodes.
    /// Default: 0.05
    pub penalty_knows_delta: f64,

    /// Initial KNOWS weight for newly discovered nodes (via reward on unknown node).
    /// Default: 0.2
    pub discovery_weight: f64,

    /// KNOWS weight below which the relation is pruned.
    /// Default: 0.01
    pub prune_threshold: f64,

    /// Maximum KNOWS weight (clamped).
    /// Default: 1.0
    pub max_knows_weight: f64,

    /// Spreading activation config used for persona-biased recall.
    #[cfg(feature = "synapse")]
    pub spread_config: SpreadConfig,
}

impl Default for PersonaConfig {
    fn default() -> Self {
        Self {
            activation_bias: 0.5,
            reward_knows_delta: 0.1,
            penalty_knows_delta: 0.05,
            discovery_weight: 0.2,
            prune_threshold: 0.01,
            max_knows_weight: 1.0,
            #[cfg(feature = "synapse")]
            spread_config: SpreadConfig::default(),
        }
    }
}

/// A cognitive persona — a named, weighted projection of the shared graph.
///
/// The persona does NOT own graph data. It only holds:
/// - KNOWS: `HashMap<NodeId, f64>` — weighted familiarity with nodes
/// - Skills: named groupings (optional, for routing)
/// - Config: how aggressively it biases recall
///
/// All learning (synapses, energy, stigmergy) happens on the SHARED graph.
/// The persona only adjusts its KNOWS weights based on feedback.
#[derive(Debug, Clone)]
pub struct CognitivePersona {
    /// Unique persona identifier.
    pub id: PersonaId,
    /// Human-readable name (e.g. "Traffic Law Expert").
    pub name: String,
    /// Optional description of the persona's domain.
    pub description: String,

    /// Weighted familiarity: node → weight in [0.0, max_knows_weight].
    /// Higher weight = persona "knows" this concept well, will boost it during recall.
    knows: HashMap<NodeId, f64>,

    /// Named skill clusters (optional). Each skill is a set of KNOWS nodes
    /// that form a coherent domain. Used for routing and explanation.
    skills: HashMap<String, Vec<NodeId>>,

    /// Configuration for this persona.
    pub config: PersonaConfig,

    /// Total queries through this persona (for stats).
    pub query_count: u64,

    /// Total rewards received (for stats).
    pub reward_count: u64,

    /// Total penalties received (for stats).
    pub penalty_count: u64,

    /// When this persona was created.
    pub created_at: Instant,

    /// When this persona was last used for recall.
    pub last_active: Instant,
}

impl CognitivePersona {
    /// Create a new persona with empty KNOWS.
    pub fn new(id: PersonaId, name: impl Into<String>, description: impl Into<String>) -> Self {
        let now = Instant::now();
        Self {
            id,
            name: name.into(),
            description: description.into(),
            knows: HashMap::new(),
            skills: HashMap::new(),
            config: PersonaConfig::default(),
            query_count: 0,
            reward_count: 0,
            penalty_count: 0,
            created_at: now,
            last_active: now,
        }
    }

    /// Create with custom config.
    pub fn with_config(mut self, config: PersonaConfig) -> Self {
        self.config = config;
        self
    }

    // ── KNOWS management ───────────────────────────────────────────

    /// Add or update a KNOWS relation.
    pub fn add_knows(&mut self, node: NodeId, weight: f64) {
        let w = weight.clamp(0.0, self.config.max_knows_weight);
        self.knows.insert(node, w);
    }

    /// Batch add KNOWS relations.
    pub fn add_knows_batch(&mut self, nodes: &[(NodeId, f64)]) {
        for &(node, weight) in nodes {
            self.add_knows(node, weight);
        }
    }

    /// Remove a KNOWS relation.
    pub fn remove_knows(&mut self, node: NodeId) {
        self.knows.remove(&node);
    }

    /// Get KNOWS weight for a node (0.0 if unknown).
    pub fn knows_weight(&self, node: NodeId) -> f64 {
        self.knows.get(&node).copied().unwrap_or(0.0)
    }

    /// All KNOWS relations.
    pub fn knows(&self) -> &HashMap<NodeId, f64> {
        &self.knows
    }

    /// Number of known nodes.
    pub fn knows_count(&self) -> usize {
        self.knows.len()
    }

    /// Top-N known nodes by weight.
    pub fn top_knows(&self, n: usize) -> Vec<(NodeId, f64)> {
        let mut entries: Vec<_> = self.knows.iter().map(|(&k, &v)| (k, v)).collect();
        entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        entries.truncate(n);
        entries
    }

    // ── Skills management ──────────────────────────────────────────

    /// Register a skill (named cluster of KNOWS nodes).
    pub fn add_skill(&mut self, name: impl Into<String>, nodes: Vec<NodeId>) {
        self.skills.insert(name.into(), nodes);
    }

    /// Get skill nodes.
    pub fn skill(&self, name: &str) -> Option<&[NodeId]> {
        self.skills.get(name).map(|v| v.as_slice())
    }

    /// List all skill names.
    pub fn skill_names(&self) -> Vec<&str> {
        self.skills.keys().map(|s| s.as_str()).collect()
    }

    // ── Persona-biased recall ──────────────────────────────────────

    /// Build activation sources with persona bias.
    ///
    /// For each cue node:
    /// - Base energy = 1.0
    /// - If KNOWS(cue) exists: energy += knows_weight × activation_bias
    ///
    /// This means a persona that "knows" a concept well will amplify its
    /// signal during spreading activation, causing related engrams to
    /// score higher.
    pub fn biased_sources(&self, cues: &[NodeId]) -> Vec<(NodeId, f64)> {
        cues.iter()
            .map(|&node| {
                let knows_boost = self.knows_weight(node) * self.config.activation_bias;
                (node, 1.0 + knows_boost)
            })
            .collect()
    }

    /// Perform spreading activation with persona bias.
    ///
    /// This is the core recall mechanism:
    /// 1. Build biased sources from cues + KNOWS
    /// 2. Spread through the SHARED synapse graph
    /// 3. Return activation map (shared by all — persona only influenced the seeds)
    #[cfg(feature = "synapse")]
    pub fn spread_activation(
        &mut self,
        cues: &[NodeId],
        source: &dyn ActivationSource,
    ) -> ActivationMap {
        self.query_count += 1;
        self.last_active = Instant::now();

        let sources = self.biased_sources(cues);
        spread(&sources, source, &self.config.spread_config)
    }

    // ── Feedback (learning) ────────────────────────────────────────

    /// Process reward feedback: strengthen KNOWS for activated nodes.
    ///
    /// Two effects:
    /// 1. SHARED: caller should also call synapse.reinforce() + energy.boost()
    ///    (this method does NOT touch shared state — only persona state)
    /// 2. PERSONA-SPECIFIC: KNOWS weights increase for activated nodes,
    ///    and new nodes are discovered (added with discovery_weight).
    pub fn on_reward(&mut self, activated_nodes: &[NodeId]) {
        self.reward_count += 1;
        let delta = self.config.reward_knows_delta;
        let discovery = self.config.discovery_weight;
        let max_w = self.config.max_knows_weight;

        for &node in activated_nodes {
            self.knows
                .entry(node)
                .and_modify(|w| *w = (*w + delta).min(max_w))
                .or_insert(discovery);
        }
    }

    /// Process penalty feedback: weaken KNOWS for activated nodes.
    ///
    /// Nodes whose KNOWS weight drops below prune_threshold are removed.
    /// This is how a persona "forgets" concepts that lead to bad recalls.
    pub fn on_penalty(&mut self, activated_nodes: &[NodeId]) {
        self.penalty_count += 1;
        let delta = self.config.penalty_knows_delta;
        let threshold = self.config.prune_threshold;

        for &node in activated_nodes {
            if let Some(w) = self.knows.get_mut(&node) {
                *w -= delta;
            }
        }

        // Prune weak KNOWS
        self.knows.retain(|_, w| *w >= threshold);
    }

    /// Prune KNOWS relations below threshold.
    pub fn prune_weak_knows(&mut self) -> usize {
        let threshold = self.config.prune_threshold;
        let before = self.knows.len();
        self.knows.retain(|_, w| *w >= threshold);
        before - self.knows.len()
    }

    // ── Stats ──────────────────────────────────────────────────────

    /// Persona statistics.
    pub fn stats(&self) -> PersonaStats {
        let total_weight: f64 = self.knows.values().sum();
        let avg_weight = if self.knows.is_empty() {
            0.0
        } else {
            total_weight / self.knows.len() as f64
        };

        PersonaStats {
            id: self.id,
            name: self.name.clone(),
            knows_count: self.knows.len(),
            total_knows_weight: total_weight,
            avg_knows_weight: avg_weight,
            skill_count: self.skills.len(),
            query_count: self.query_count,
            reward_count: self.reward_count,
            penalty_count: self.penalty_count,
            reward_rate: if self.query_count > 0 {
                self.reward_count as f64 / self.query_count as f64
            } else {
                0.0
            },
        }
    }
}

/// Statistics for a persona.
#[derive(Debug, Clone)]
pub struct PersonaStats {
    /// Persona identifier.
    pub id: PersonaId,
    /// Persona name.
    pub name: String,
    /// Number of KNOWS relations.
    pub knows_count: usize,
    /// Sum of all KNOWS weights.
    pub total_knows_weight: f64,
    /// Average KNOWS weight.
    pub avg_knows_weight: f64,
    /// Number of named skills.
    pub skill_count: usize,
    /// Total queries through this persona.
    pub query_count: u64,
    /// Total reward feedbacks.
    pub reward_count: u64,
    /// Total penalty feedbacks.
    pub penalty_count: u64,
    /// Reward rate (reward_count / query_count).
    pub reward_rate: f64,
}

// ═══════════════════════════════════════════════════════════════════════════
// PersonaStore — shared registry of all personas
// ═══════════════════════════════════════════════════════════════════════════

/// Thread-safe registry of all personas in the database.
///
/// This is the "memory" of the persona system — it persists persona state
/// across queries and sessions. Multiple agents can share the same persona
/// or use different ones.
pub struct PersonaStore {
    personas: DashMap<PersonaId, CognitivePersona>,
    /// Name → ID index for lookup by name.
    name_index: DashMap<String, PersonaId>,
    /// Next persona ID.
    next_id: std::sync::atomic::AtomicU64,
}

impl PersonaStore {
    /// Create a new empty store.
    pub fn new() -> Self {
        Self {
            personas: DashMap::new(),
            name_index: DashMap::new(),
            next_id: std::sync::atomic::AtomicU64::new(1),
        }
    }

    /// Create a new persona and register it.
    pub fn create(&self, name: impl Into<String>, description: impl Into<String>) -> PersonaId {
        let name = name.into();
        let id = PersonaId(
            self.next_id
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst),
        );
        let persona = CognitivePersona::new(id, name.clone(), description);
        self.name_index.insert(name, id);
        self.personas.insert(id, persona);
        id
    }

    /// Create with config.
    pub fn create_with_config(
        &self,
        name: impl Into<String>,
        description: impl Into<String>,
        config: PersonaConfig,
    ) -> PersonaId {
        let name = name.into();
        let id = PersonaId(
            self.next_id
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst),
        );
        let persona = CognitivePersona::new(id, name.clone(), description).with_config(config);
        self.name_index.insert(name, id);
        self.personas.insert(id, persona);
        id
    }

    /// Get a persona by ID (immutable).
    pub fn get(
        &self,
        id: PersonaId,
    ) -> Option<dashmap::mapref::one::Ref<'_, PersonaId, CognitivePersona>> {
        self.personas.get(&id)
    }

    /// Get a persona by ID (mutable).
    pub fn get_mut(
        &self,
        id: PersonaId,
    ) -> Option<dashmap::mapref::one::RefMut<'_, PersonaId, CognitivePersona>> {
        self.personas.get_mut(&id)
    }

    /// Find persona by name.
    pub fn find_by_name(&self, name: &str) -> Option<PersonaId> {
        self.name_index.get(name).map(|r| *r.value())
    }

    /// Delete a persona.
    pub fn remove(&self, id: PersonaId) -> Option<CognitivePersona> {
        if let Some((_, persona)) = self.personas.remove(&id) {
            self.name_index.remove(&persona.name);
            Some(persona)
        } else {
            None
        }
    }

    /// List all personas.
    pub fn list(&self) -> Vec<PersonaStats> {
        self.personas
            .iter()
            .map(|entry| entry.value().stats())
            .collect()
    }

    /// Number of registered personas.
    pub fn count(&self) -> usize {
        self.personas.len()
    }

    /// Find the best persona for a set of cue nodes.
    ///
    /// Returns the persona whose KNOWS has the highest total weight
    /// overlap with the given cues. This enables automatic persona
    /// routing: "which expert should answer this question?"
    pub fn best_for_cues(&self, cues: &[NodeId]) -> Option<PersonaId> {
        let mut best: Option<(PersonaId, f64)> = None;

        for entry in &self.personas {
            let persona = entry.value();
            let score: f64 = cues.iter().map(|c| persona.knows_weight(*c)).sum();
            if score > 0.0 {
                match &best {
                    Some((_, best_score)) if score > *best_score => {
                        best = Some((persona.id, score));
                    }
                    None => {
                        best = Some((persona.id, score));
                    }
                    _ => {}
                }
            }
        }

        best.map(|(id, _)| id)
    }
}

impl Default for PersonaStore {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for PersonaStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PersonaStore")
            .field("count", &self.personas.len())
            .finish()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PersonaRecallEngine — the unified recall path through a persona
// ═══════════════════════════════════════════════════════════════════════════

/// Outcome of a persona-biased recall.
#[derive(Debug)]
pub struct PersonaRecallResult {
    /// Which persona was used (None = raw, no persona bias).
    pub persona_id: Option<PersonaId>,
    /// Activation map from spreading activation.
    pub activation: HashMap<NodeId, f64>,
    /// Trail ID for feedback.
    pub trail_id: u64,
    /// Nodes activated above threshold, sorted by energy descending.
    pub top_nodes: Vec<(NodeId, f64)>,
}

/// Outcome of a feedback operation.
#[derive(Debug, Clone)]
pub struct PersonaFeedbackResult {
    /// Number of KNOWS relations strengthened.
    pub knows_strengthened: usize,
    /// Number of KNOWS relations weakened.
    pub knows_weakened: usize,
    /// Number of KNOWS relations discovered (new).
    pub knows_discovered: usize,
    /// Number of KNOWS relations pruned (forgotten).
    pub knows_pruned: usize,
    /// Number of synapses reinforced (shared).
    pub synapses_reinforced: usize,
    /// Number of nodes energy-boosted (shared).
    pub nodes_boosted: usize,
}

/// Trail record — what happened during a recall, for feedback tracing.
#[derive(Debug)]
struct Trail {
    persona_id: Option<PersonaId>,
    cues: Vec<NodeId>,
    activated_nodes: Vec<NodeId>,
    created_at: Instant,
}

/// The unified recall+feedback engine that coordinates persona-biased
/// queries with shared cognitive infrastructure.
///
/// This is the API that the database exposes to ALL clients.
/// It is the single point where recall and learning happen.
pub struct PersonaRecallEngine {
    /// All personas.
    pub store: Arc<PersonaStore>,
    /// Synapse store (shared — all personas read/write the same synapses).
    #[cfg(feature = "synapse")]
    pub synapse_store: Arc<SynapseStore>,
    /// Energy store (shared — all personas read/write the same energy).
    #[cfg(feature = "energy")]
    pub energy_store: Arc<EnergyStore>,
    /// Active trails (for feedback).
    trails: DashMap<u64, Trail>,
    /// Trail ID counter.
    trail_counter: std::sync::atomic::AtomicU64,
}

impl PersonaRecallEngine {
    /// Create a new engine with shared cognitive stores.
    #[cfg(all(feature = "synapse", feature = "energy"))]
    pub fn new(
        store: Arc<PersonaStore>,
        synapse_store: Arc<SynapseStore>,
        energy_store: Arc<EnergyStore>,
    ) -> Self {
        Self {
            store,
            synapse_store,
            energy_store,
            trails: DashMap::new(),
            trail_counter: std::sync::atomic::AtomicU64::new(1),
        }
    }

    /// Recall with persona bias.
    ///
    /// 1. Look up persona (or use raw recall if None)
    /// 2. Build biased sources from cues + persona KNOWS
    /// 3. Spread activation through SHARED synapses
    /// 4. Record trail for feedback
    /// 5. Return activation map + trail_id
    #[cfg(feature = "synapse")]
    pub fn recall(&self, cues: &[NodeId], persona_id: Option<PersonaId>) -> PersonaRecallResult {
        let trail_id = self
            .trail_counter
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        // Build sources with persona bias
        let sources: Vec<(NodeId, f64)> = match persona_id {
            Some(pid) => {
                match self.store.get_mut(pid) {
                    Some(mut persona) => {
                        persona.query_count += 1;
                        persona.last_active = Instant::now();
                        persona.biased_sources(cues)
                    }
                    None => {
                        // Persona not found — fallback to raw
                        cues.iter().map(|&n| (n, 1.0)).collect()
                    }
                }
            }
            None => cues.iter().map(|&n| (n, 1.0)).collect(),
        };

        // Spread through shared synapse graph
        let spread_config = match persona_id {
            Some(pid) => self
                .store
                .get(pid)
                .map(|p| p.config.spread_config.clone())
                .unwrap_or_default(),
            None => SpreadConfig::default(),
        };

        let source =
            crate::activation::SynapseActivationSource::new(Arc::clone(&self.synapse_store));
        let activation = spread(&sources, &source, &spread_config);

        // Sort by activation descending
        let mut top_nodes: Vec<(NodeId, f64)> = activation.iter().map(|(&n, &e)| (n, e)).collect();
        top_nodes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let activated_node_ids: Vec<NodeId> = top_nodes.iter().map(|(n, _)| *n).collect();

        // Record trail
        self.trails.insert(
            trail_id,
            Trail {
                persona_id,
                cues: cues.to_vec(),
                activated_nodes: activated_node_ids,
                created_at: Instant::now(),
            },
        );

        PersonaRecallResult {
            persona_id,
            activation,
            trail_id,
            top_nodes,
        }
    }

    /// Provide feedback for a trail.
    ///
    /// Two-layer learning:
    /// 1. SHARED: reinforce/weaken synapses + boost/drain energy
    ///    → benefits ALL future clients
    /// 2. PERSONA: adjust KNOWS weights
    ///    → benefits future queries through this persona
    #[cfg(all(feature = "synapse", feature = "energy"))]
    pub fn feedback(&self, trail_id: u64, rewarded: bool) -> Option<PersonaFeedbackResult> {
        let (_, trail) = self.trails.remove(&trail_id)?;
        let nodes = &trail.activated_nodes;

        let mut result = PersonaFeedbackResult {
            knows_strengthened: 0,
            knows_weakened: 0,
            knows_discovered: 0,
            knows_pruned: 0,
            synapses_reinforced: 0,
            nodes_boosted: 0,
        };

        // ── Layer 1: SHARED learning (benefits everyone) ──────────────

        if rewarded {
            // Reinforce synapses between co-activated nodes
            let n = nodes.len().min(20); // cap to avoid O(n²) explosion
            for i in 0..n {
                for j in (i + 1)..n {
                    self.synapse_store.reinforce(nodes[i], nodes[j], 0.1);
                    result.synapses_reinforced += 1;
                }
            }

            // Boost energy on activated nodes
            for &node in nodes.iter().take(n) {
                self.energy_store.boost(node, 0.2);
                result.nodes_boosted += 1;
            }
        } else {
            // Penalty: lighter touch on shared state
            // Only drain energy (don't weaken synapses — that's destructive for others)
            for &node in nodes.iter().take(10) {
                self.energy_store.boost(node, -0.1); // negative boost = drain
                result.nodes_boosted += 1;
            }
        }

        // ── Layer 2: PERSONA learning (benefits this persona only) ────

        if let Some(persona_id) = trail.persona_id
            && let Some(mut persona) = self.store.get_mut(persona_id)
        {
            let before_count = persona.knows_count();

            if rewarded {
                persona.on_reward(nodes);
                let after_count = persona.knows_count();
                result.knows_discovered = after_count.saturating_sub(before_count);
                result.knows_strengthened = nodes.len().min(before_count);
            } else {
                persona.on_penalty(nodes);
                let after_count = persona.knows_count();
                result.knows_pruned = before_count.saturating_sub(after_count);
                result.knows_weakened = nodes.len();
            }
        }

        Some(result)
    }

    /// Get trail info (for debugging/tracing).
    pub fn trail_info(
        &self,
        trail_id: u64,
    ) -> Option<(Option<PersonaId>, Vec<NodeId>, Vec<NodeId>)> {
        self.trails
            .get(&trail_id)
            .map(|t| (t.persona_id, t.cues.clone(), t.activated_nodes.clone()))
    }

    /// Clean up old trails (older than given duration).
    pub fn cleanup_trails(&self, max_age: std::time::Duration) -> usize {
        let now = Instant::now();
        let before = self.trails.len();
        self.trails
            .retain(|_, trail| now.duration_since(trail.created_at) < max_age);
        before - self.trails.len()
    }
}

impl std::fmt::Debug for PersonaRecallEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PersonaRecallEngine")
            .field("personas", &self.store.count())
            .field("active_trails", &self.trails.len())
            .finish()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persona_knows_management() {
        let mut p = CognitivePersona::new(PersonaId(1), "test", "test persona");

        // Add KNOWS
        p.add_knows(NodeId(10), 0.9);
        p.add_knows(NodeId(20), 0.5);
        p.add_knows(NodeId(30), 0.1);

        assert_eq!(p.knows_count(), 3);
        assert!((p.knows_weight(NodeId(10)) - 0.9).abs() < f64::EPSILON);
        assert!((p.knows_weight(NodeId(99)) - 0.0).abs() < f64::EPSILON); // unknown

        // Top KNOWS
        let top = p.top_knows(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, NodeId(10)); // highest weight first
    }

    #[test]
    fn biased_sources_boost_known_nodes() {
        let mut p = CognitivePersona::new(PersonaId(1), "expert", "");
        p.add_knows(NodeId(10), 1.0); // fully known
        p.add_knows(NodeId(20), 0.5); // partially known

        let sources = p.biased_sources(&[NodeId(10), NodeId(20), NodeId(30)]);

        // NodeId(10): 1.0 + 1.0 * 0.5 = 1.5
        assert!((sources[0].1 - 1.5).abs() < f64::EPSILON);
        // NodeId(20): 1.0 + 0.5 * 0.5 = 1.25
        assert!((sources[1].1 - 1.25).abs() < f64::EPSILON);
        // NodeId(30): 1.0 + 0.0 * 0.5 = 1.0 (unknown node, no boost)
        assert!((sources[2].1 - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn reward_strengthens_and_discovers() {
        let mut p = CognitivePersona::new(PersonaId(1), "learner", "");
        p.add_knows(NodeId(10), 0.5);

        // Reward with known + unknown nodes
        p.on_reward(&[NodeId(10), NodeId(99)]);

        // Known node strengthened
        assert!((p.knows_weight(NodeId(10)) - 0.6).abs() < f64::EPSILON);
        // Unknown node discovered with discovery_weight (0.2)
        assert!((p.knows_weight(NodeId(99)) - 0.2).abs() < f64::EPSILON);
    }

    #[test]
    fn penalty_weakens_and_prunes() {
        let mut p = CognitivePersona::new(PersonaId(1), "forgetter", "");
        p.add_knows(NodeId(10), 0.5);
        p.add_knows(NodeId(20), 0.02); // just above prune threshold

        p.on_penalty(&[NodeId(10), NodeId(20)]);

        // Node 10: 0.5 - 0.05 = 0.45 (still above threshold)
        assert!((p.knows_weight(NodeId(10)) - 0.45).abs() < f64::EPSILON);
        // Node 20: 0.02 - 0.05 = -0.03 → pruned
        assert_eq!(p.knows_weight(NodeId(20)), 0.0);
        assert_eq!(p.knows_count(), 1);
    }

    #[test]
    fn repeated_rewards_cap_at_max() {
        let mut p = CognitivePersona::new(PersonaId(1), "maxer", "");
        p.add_knows(NodeId(10), 0.95);

        for _ in 0..10 {
            p.on_reward(&[NodeId(10)]);
        }

        // Should cap at max_knows_weight (1.0)
        assert!(p.knows_weight(NodeId(10)) <= 1.0);
    }

    #[test]
    fn persona_store_crud() {
        let store = PersonaStore::new();

        let id1 = store.create("Traffic", "Traffic law expert");
        let id2 = store.create("Labor", "Labor law expert");

        assert_eq!(store.count(), 2);

        // Find by name
        assert_eq!(store.find_by_name("Traffic"), Some(id1));
        assert_eq!(store.find_by_name("Unknown"), None);

        // Get
        let p = store.get(id1).unwrap();
        assert_eq!(p.name, "Traffic");
        drop(p);

        // List
        let stats = store.list();
        assert_eq!(stats.len(), 2);

        // Remove
        store.remove(id2);
        assert_eq!(store.count(), 1);
        assert_eq!(store.find_by_name("Labor"), None);
    }

    #[test]
    fn best_for_cues_routes_correctly() {
        let store = PersonaStore::new();

        let traffic_id = store.create("Traffic", "");
        let labor_id = store.create("Labor", "");

        // Traffic persona knows traffic nodes
        {
            let mut p = store.get_mut(traffic_id).unwrap();
            p.add_knows(NodeId(1), 0.9); // vitesse
            p.add_knows(NodeId(2), 0.8); // amende
        }

        // Labor persona knows labor nodes
        {
            let mut p = store.get_mut(labor_id).unwrap();
            p.add_knows(NodeId(10), 0.9); // contrat
            p.add_knows(NodeId(11), 0.85); // licenciement
        }

        // Traffic cues → Traffic persona
        assert_eq!(
            store.best_for_cues(&[NodeId(1), NodeId(2)]),
            Some(traffic_id)
        );

        // Labor cues → Labor persona
        assert_eq!(
            store.best_for_cues(&[NodeId(10), NodeId(11)]),
            Some(labor_id)
        );

        // Unknown cues → None
        assert_eq!(store.best_for_cues(&[NodeId(99)]), None);
    }

    #[test]
    fn skills_management() {
        let mut p = CognitivePersona::new(PersonaId(1), "test", "");

        p.add_skill("sanctions", vec![NodeId(1), NodeId(2), NodeId(3)]);
        p.add_skill("procedure", vec![NodeId(4), NodeId(5)]);

        assert_eq!(p.skill_names().len(), 2);
        assert_eq!(p.skill("sanctions").unwrap().len(), 3);
    }

    #[test]
    fn persona_stats() {
        let mut p = CognitivePersona::new(PersonaId(1), "test", "");
        p.add_knows(NodeId(1), 0.8);
        p.add_knows(NodeId(2), 0.6);
        p.on_reward(&[NodeId(1)]);
        p.on_penalty(&[NodeId(3)]);

        let stats = p.stats();
        assert_eq!(stats.knows_count, 2); // NodeId(1)=0.9, NodeId(2)=0.6
        assert_eq!(stats.reward_count, 1);
        assert_eq!(stats.penalty_count, 1);
        assert!(stats.avg_knows_weight > 0.0);
    }
}
