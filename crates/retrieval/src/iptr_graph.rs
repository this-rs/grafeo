//! GraphLookupTool — In-process graph retrieval for IPTR.
//!
//! When entropy spikes during generation (model hesitation), this tool
//! performs a fast BFS traversal on a snapshot of the persona fact graph
//! to find relevant concepts, then returns logit biases that steer the
//! model toward tokens associated with those concepts.
//!
//! Architecture:
//! - `FactSnapshot` captures the graph state before generation starts
//! - `GraphLookupTool` implements `InProcessTool` (Send + Sync)
//! - Pipeline: seed_selection (text match) → bfs_traverse (2-hop) → token_mapping
//!
//! The snapshot approach avoids holding a reference to PersonaDB (which uses
//! RefCell and is not Send+Sync) during generation.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::iptr::{InProcessTool, ToolContext, ToolResult};

// ─────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────

/// A single fact node captured from PersonaDB.
#[derive(Debug, Clone)]
pub struct FactEntry {
    pub node_id: u64,
    pub key: String,
    pub value: String,
    pub energy: f64,
    pub confidence: f64,
    /// Precomputed lowercase tokens for fast matching.
    tokens: Vec<String>,
    /// Pre-tokenized value via LLM tokenizer (computed at snapshot time).
    /// Each pair is (token_id, weight=1.0). Empty if no tokenizer was provided.
    pub value_token_ids: Vec<(u32, f32)>,
}

impl FactEntry {
    pub fn new(node_id: u64, key: &str, value: &str, energy: f64, confidence: f64) -> Self {
        let text = format!("{} {}", key, value).to_lowercase();
        let tokens: Vec<String> = text
            .split_whitespace()
            .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
            .filter(|s| !s.is_empty())
            .collect();
        Self {
            node_id,
            key: key.to_string(),
            value: value.to_string(),
            energy,
            confidence,
            tokens,
            value_token_ids: Vec::new(),
        }
    }

    /// Create with pre-tokenized value.
    pub fn with_tokens(
        node_id: u64,
        key: &str,
        value: &str,
        energy: f64,
        confidence: f64,
        value_token_ids: Vec<(u32, f32)>,
    ) -> Self {
        let mut entry = Self::new(node_id, key, value, energy, confidence);
        entry.value_token_ids = value_token_ids;
        entry
    }

    /// Simple term-frequency score against query terms.
    fn score(&self, query_terms: &[String]) -> f64 {
        let mut hits = 0;
        for qt in query_terms {
            for ft in &self.tokens {
                if ft.contains(qt.as_str()) {
                    hits += 1;
                    break; // one match per query term
                }
            }
        }
        if query_terms.is_empty() {
            return 0.0;
        }
        (hits as f64 / query_terms.len() as f64) * self.energy * self.confidence
    }
}

/// Frozen snapshot of the persona graph for thread-safe access.
#[derive(Debug, Clone)]
pub struct FactSnapshot {
    /// All active facts.
    pub facts: Vec<FactEntry>,
    /// Adjacency list: node_id → neighbor node_ids (bidirectional edges).
    pub adjacency: HashMap<u64, Vec<u64>>,
}

impl FactSnapshot {
    /// Create an empty snapshot (no facts loaded).
    pub fn empty() -> Self {
        Self {
            facts: Vec::new(),
            adjacency: HashMap::new(),
        }
    }

    /// Build a snapshot from PersonaDB with optional pre-tokenization.
    ///
    /// Captures all active facts and their graph edges in a single pass.
    /// Call this once before generation starts.
    ///
    /// `tokenize_fn`: optional function that converts text to (token_id, weight) pairs.
    /// Pass the LLM engine's tokenizer to enable logit biasing.
    pub fn from_persona_db(
        db: &persona::PersonaDB,
        tokenize_fn: Option<&dyn Fn(&str) -> Vec<(u32, f32)>>,
    ) -> Self {
        use obrain_core::graph::Direction;

        let store = db.db.store();

        // Collect facts with metadata
        let mut facts = Vec::new();
        let mut fact_ids = HashSet::new();

        for entry in db.list_facts() {
            let (nid, key, value, _turn, active, confidence, energy, _fact_type) = entry;
            if !active {
                continue;
            }
            fact_ids.insert(nid.as_u64());
            let token_ids = tokenize_fn
                .map(|f| f(&value))
                .unwrap_or_default();
            facts.push(FactEntry::with_tokens(
                nid.as_u64(),
                &key,
                &value,
                energy,
                confidence,
                token_ids,
            ));
        }

        // Build adjacency from edges between fact nodes
        let mut adjacency: HashMap<u64, Vec<u64>> = HashMap::new();
        for &nid_u64 in &fact_ids {
            let nid = obrain_common::types::NodeId::from(nid_u64);
            // Outgoing edges to other facts
            for (target, _eid) in store.edges_from(nid, Direction::Outgoing).collect::<Vec<_>>() {
                let target_u64 = target.as_u64();
                if fact_ids.contains(&target_u64) {
                    adjacency.entry(nid_u64).or_default().push(target_u64);
                    adjacency.entry(target_u64).or_default().push(nid_u64);
                }
            }
        }

        // Deduplicate adjacency lists
        for neighbors in adjacency.values_mut() {
            neighbors.sort_unstable();
            neighbors.dedup();
        }

        Self { facts, adjacency }
    }

    /// Number of facts in the snapshot.
    pub fn len(&self) -> usize {
        self.facts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.facts.is_empty()
    }
}

/// Configuration for the graph lookup tool.
#[derive(Debug, Clone)]
pub struct GraphLookupConfig {
    /// Maximum BFS hops from seed nodes.
    pub max_hops: usize,
    /// Logit bias for hop-0 (seed) concepts.
    pub bias_hop0: f32,
    /// Logit bias for hop-1 neighbors.
    pub bias_hop1: f32,
    /// Logit bias for hop-2 neighbors.
    pub bias_hop2: f32,
    /// Maximum seed facts to consider.
    pub max_seeds: usize,
    /// Minimum score to qualify as a seed.
    pub min_seed_score: f64,
    /// Execution budget in ms.
    pub budget: u64,
}

impl Default for GraphLookupConfig {
    fn default() -> Self {
        Self {
            max_hops: 2,
            bias_hop0: 4.0,
            bias_hop1: 3.0,
            bias_hop2: 1.5,
            max_seeds: 5,
            min_seed_score: 0.1,
            budget: 5,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// GraphLookupTool
// ─────────────────────────────────────────────────────────────────────

/// In-process tool that performs BFS on the persona fact graph.
///
/// Pipeline:
/// 1. **Seed selection**: text-match recent_text against fact keys/values
/// 2. **BFS traversal**: 2-hop expansion with energy decay
/// 3. **Token mapping**: convert fact values to logit biases via tokenizer
pub struct GraphLookupTool {
    snapshot: FactSnapshot,
    config: GraphLookupConfig,
}

impl GraphLookupTool {
    /// Create a new GraphLookupTool from a pre-tokenized fact snapshot.
    ///
    /// The snapshot should be built with `from_persona_db(db, Some(&tokenize_fn))`
    /// so that `value_token_ids` are populated.
    pub fn new(snapshot: FactSnapshot, config: GraphLookupConfig) -> Self {
        Self { snapshot, config }
    }

    /// Create with default config.
    pub fn with_defaults(snapshot: FactSnapshot) -> Self {
        Self::new(snapshot, GraphLookupConfig::default())
    }

    /// Step 1: Find seed facts that match the recent text.
    fn seed_selection(&self, recent_text: &str, already_retrieved: &[u64]) -> Vec<(usize, f64)> {
        let query_terms: Vec<String> = recent_text
            .to_lowercase()
            .split_whitespace()
            .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
            .filter(|s| s.len() >= 2) // skip very short tokens
            .collect();

        if query_terms.is_empty() {
            return Vec::new();
        }

        let already: HashSet<u64> = already_retrieved.iter().copied().collect();

        let mut scored: Vec<(usize, f64)> = self
            .snapshot
            .facts
            .iter()
            .enumerate()
            .filter(|(_, f)| !already.contains(&f.node_id))
            .map(|(idx, f)| (idx, f.score(&query_terms)))
            .filter(|(_, score)| *score >= self.config.min_seed_score)
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(self.config.max_seeds);
        scored
    }

    /// Step 2: BFS traversal from seed nodes with hop-level tracking.
    ///
    /// Returns: Vec<(node_id, hop_distance)> for all reached nodes.
    fn bfs_traverse(&self, seed_indices: &[(usize, f64)]) -> Vec<(u64, usize)> {
        let mut visited: HashMap<u64, usize> = HashMap::new(); // node_id → min_hop
        let mut queue: VecDeque<(u64, usize)> = VecDeque::new();

        // Seeds are at hop 0
        for &(idx, _score) in seed_indices {
            let nid = self.snapshot.facts[idx].node_id;
            if !visited.contains_key(&nid) {
                visited.insert(nid, 0);
                queue.push_back((nid, 0));
            }
        }

        while let Some((nid, depth)) = queue.pop_front() {
            if depth >= self.config.max_hops {
                continue;
            }
            if let Some(neighbors) = self.snapshot.adjacency.get(&nid) {
                for &neighbor in neighbors {
                    let new_depth = depth + 1;
                    // Only insert if not visited or if we found a shorter path
                    if !visited.contains_key(&neighbor) {
                        visited.insert(neighbor, new_depth);
                        queue.push_back((neighbor, new_depth));
                    }
                }
            }
        }

        visited.into_iter().collect()
    }

    /// Step 3: Convert reached nodes to logit biases via the tokenizer.
    fn token_mapping(
        &self,
        reached: &[(u64, usize)],
        already_retrieved: &[u64],
    ) -> (HashMap<u32, f32>, Vec<String>) {
        let already: HashSet<u64> = already_retrieved.iter().copied().collect();
        let node_to_fact: HashMap<u64, &FactEntry> = self
            .snapshot
            .facts
            .iter()
            .map(|f| (f.node_id, f))
            .collect();

        let mut biases: HashMap<u32, f32> = HashMap::new();
        let mut concepts: Vec<String> = Vec::new();

        for &(nid, hop) in reached {
            if already.contains(&nid) {
                continue;
            }
            let Some(fact) = node_to_fact.get(&nid) else {
                continue;
            };

            // Hop-dependent bias strength with energy decay
            let base_bias = match hop {
                0 => self.config.bias_hop0,
                1 => self.config.bias_hop1,
                _ => self.config.bias_hop2,
            };
            let effective_bias = base_bias * fact.energy as f32;

            // Use pre-tokenized value (computed at snapshot time)
            for (token_id, weight) in &fact.value_token_ids {
                let bias = effective_bias * weight;
                *biases.entry(*token_id).or_insert(0.0) += bias;
            }

            concepts.push(format!("{}={} (hop{})", fact.key, fact.value, hop));
        }

        (biases, concepts)
    }
}

impl InProcessTool for GraphLookupTool {
    fn name(&self) -> &str {
        "graph_lookup"
    }

    fn budget_ms(&self) -> u64 {
        self.config.budget
    }

    fn execute(&self, context: &ToolContext) -> ToolResult {
        if self.snapshot.is_empty() {
            return ToolResult::empty(self.name());
        }

        // Step 1: Seed selection
        let seeds = self.seed_selection(&context.recent_text, &context.already_retrieved);
        if seeds.is_empty() {
            return ToolResult::empty(self.name());
        }

        // Step 2: BFS traversal
        let reached = self.bfs_traverse(&seeds);

        // Step 3: Token mapping
        let (biases, concepts) = self.token_mapping(&reached, &context.already_retrieved);

        if biases.is_empty() {
            return ToolResult::empty(self.name());
        }

        ToolResult {
            biases,
            tool_name: self.name().to_string(),
            latency_us: 0, // filled by dispatcher
            concepts_found: concepts,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple tokenizer: maps each whitespace-separated word to a hash-based token ID.
    fn test_tokenize(text: &str) -> Vec<(u32, f32)> {
        text.split_whitespace()
            .map(|word| {
                let hash = word.bytes().fold(0u32, |acc, b| {
                    acc.wrapping_mul(31).wrapping_add(b as u32)
                });
                (hash % 32000, 1.0) // simulate a 32k vocab
            })
            .collect()
    }

    fn make_snapshot() -> FactSnapshot {
        let facts = vec![
            FactEntry::with_tokens(1, "city", "Lyon", 1.0, 0.9, test_tokenize("Lyon")),
            FactEntry::with_tokens(2, "name", "Thomas", 1.0, 0.95, test_tokenize("Thomas")),
            FactEntry::with_tokens(3, "pet", "cat named Milo", 0.8, 0.7, test_tokenize("cat named Milo")),
            FactEntry::with_tokens(4, "job", "developer", 1.0, 0.85, test_tokenize("developer")),
            FactEntry::with_tokens(5, "hobby", "climbing", 0.6, 0.6, test_tokenize("climbing")),
        ];

        // Graph: 1-2 (Thomas lives in Lyon), 2-3 (Thomas has cat), 4-5 (dev+climbing)
        let mut adjacency: HashMap<u64, Vec<u64>> = HashMap::new();
        adjacency.insert(1, vec![2]);
        adjacency.insert(2, vec![1, 3]);
        adjacency.insert(3, vec![2]);
        adjacency.insert(4, vec![5]);
        adjacency.insert(5, vec![4]);

        FactSnapshot { facts, adjacency }
    }

    fn make_context(text: &str, pos: u32) -> ToolContext {
        ToolContext {
            recent_text: text.to_string(),
            entropy: 4.5,
            already_retrieved: Vec::new(),
            token_position: pos,
        }
    }

    #[test]
    fn test_seed_selection_finds_matching_facts() {
        let snapshot = make_snapshot();
        let tool = GraphLookupTool::with_defaults(snapshot);

        let seeds = tool.seed_selection("Thomas habite à", &[]);
        assert!(!seeds.is_empty(), "should find Thomas");
        // Index 1 = "name: Thomas"
        assert_eq!(tool.snapshot.facts[seeds[0].0].key, "name");
    }

    #[test]
    fn test_seed_selection_respects_already_retrieved() {
        let snapshot = make_snapshot();
        let tool = GraphLookupTool::with_defaults(snapshot);

        // Node 2 = Thomas is already retrieved
        let seeds = tool.seed_selection("Thomas habite à", &[2]);
        // Should not include Thomas (node_id=2)
        for &(idx, _) in &seeds {
            assert_ne!(tool.snapshot.facts[idx].node_id, 2);
        }
    }

    #[test]
    fn test_bfs_reaches_neighbors() {
        let snapshot = make_snapshot();
        let tool = GraphLookupTool::with_defaults(snapshot);

        // Seed on Thomas (index 1, node_id 2)
        let reached = tool.bfs_traverse(&[(1, 1.0)]);
        let reached_ids: HashSet<u64> = reached.iter().map(|(nid, _)| *nid).collect();

        // Should reach: Thomas(2), Lyon(1) at hop1, cat(3) at hop2
        assert!(reached_ids.contains(&2), "seed: Thomas");
        assert!(reached_ids.contains(&1), "hop1: Lyon");
        assert!(reached_ids.contains(&3), "hop2: cat");
        // Should NOT reach: developer(4), climbing(5) — different component
        assert!(!reached_ids.contains(&4));
        assert!(!reached_ids.contains(&5));
    }

    #[test]
    fn test_bfs_hop_distances() {
        let snapshot = make_snapshot();
        let tool = GraphLookupTool::with_defaults(snapshot);

        let reached = tool.bfs_traverse(&[(1, 1.0)]); // seed = Thomas (idx 1, nid 2)
        let hop_map: HashMap<u64, usize> = reached.into_iter().collect();

        assert_eq!(hop_map[&2], 0, "Thomas is seed (hop 0)");
        assert_eq!(hop_map[&1], 1, "Lyon is hop 1");
        assert_eq!(hop_map[&3], 1, "cat is hop 1");
    }

    #[test]
    fn test_execute_returns_biases() {
        let snapshot = make_snapshot();
        let tool = GraphLookupTool::with_defaults(snapshot);

        let ctx = make_context("Thomas habite à", 0);
        let result = tool.execute(&ctx);

        assert!(!result.biases.is_empty(), "should produce biases");
        assert!(!result.concepts_found.is_empty(), "should report concepts");
        assert!(
            result.concepts_found.iter().any(|c| c.contains("Thomas")),
            "should find Thomas concept"
        );
    }

    #[test]
    fn test_execute_empty_snapshot() {
        let snapshot = FactSnapshot::empty();
        let tool = GraphLookupTool::with_defaults(snapshot);

        let ctx = make_context("anything", 0);
        let result = tool.execute(&ctx);
        assert!(result.biases.is_empty());
    }

    #[test]
    fn test_execute_no_match() {
        let snapshot = make_snapshot();
        let tool = GraphLookupTool::with_defaults(snapshot);

        let ctx = make_context("xyzzy quantum singularity", 0);
        let result = tool.execute(&ctx);
        assert!(result.biases.is_empty(), "no facts match this text");
    }

    #[test]
    fn test_energy_decay_affects_bias() {
        let snapshot = make_snapshot();
        let tool = GraphLookupTool::with_defaults(snapshot);

        // Climbing has energy 0.6, developer has energy 1.0
        // Both are seeds if we query "developer climbing"
        let ctx = make_context("developer climbing", 0);
        let result = tool.execute(&ctx);

        // The biases should exist (specific values depend on tokenizer)
        assert!(!result.biases.is_empty());
    }

    #[test]
    fn test_performance_large_graph() {
        // Simulate 500 facts with chain topology
        let mut facts = Vec::new();
        let mut adjacency: HashMap<u64, Vec<u64>> = HashMap::new();
        for i in 0u64..500 {
            let val = format!("value for concept {}", i);
            let toks = test_tokenize(&val);
            facts.push(FactEntry::with_tokens(
                i,
                &format!("fact_{}", i),
                &val,
                1.0,
                0.8,
                toks,
            ));
            if i > 0 {
                adjacency.entry(i).or_default().push(i - 1);
                adjacency.entry(i - 1).or_default().push(i);
            }
        }
        let snapshot = FactSnapshot { facts, adjacency };
        let tool = GraphLookupTool::with_defaults(snapshot);

        let ctx = make_context("concept 250", 0);
        let start = std::time::Instant::now();
        let result = tool.execute(&ctx);
        let elapsed = start.elapsed();

        assert!(!result.biases.is_empty());
        assert!(
            elapsed.as_millis() < 5,
            "should complete in < 5ms, took {}ms",
            elapsed.as_millis()
        );
    }
}
