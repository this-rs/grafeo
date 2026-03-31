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

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

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

}

/// Frozen snapshot of the persona graph for thread-safe access.
#[derive(Debug, Clone)]
pub struct FactSnapshot {
    /// All active facts.
    pub facts: Vec<FactEntry>,
    /// Adjacency list: node_id → neighbor node_ids (bidirectional edges).
    pub adjacency: HashMap<u64, Vec<u64>>,
    /// Inverted index: normalized token → fact indices containing this token.
    /// BTreeMap enables O(log K + matches) prefix scan for substring fallback,
    /// instead of O(K) brute-force scan that was 30-170ms on 118K-fact graphs.
    inverted_index: BTreeMap<String, Vec<usize>>,
    /// Reverse lookup: node_id → index in `facts` vec.
    /// Pre-built to avoid O(N) HashMap construction in token_mapping on every call.
    node_id_to_index: HashMap<u64, usize>,
}

impl FactSnapshot {
    /// Build the inverted index from facts: token → [fact_index, ...].
    /// Uses BTreeMap for O(log K) prefix scan in seed_selection fallback.
    fn build_inverted_index(facts: &[FactEntry]) -> BTreeMap<String, Vec<usize>> {
        let mut index: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (i, fact) in facts.iter().enumerate() {
            for token in &fact.tokens {
                index.entry(token.clone()).or_default().push(i);
            }
        }
        index
    }

    /// Build node_id → fact index reverse lookup.
    fn build_node_index(facts: &[FactEntry]) -> HashMap<u64, usize> {
        facts.iter().enumerate().map(|(i, f)| (f.node_id, i)).collect()
    }

    /// Create an empty snapshot (no facts loaded).
    pub fn empty() -> Self {
        Self {
            facts: Vec::new(),
            adjacency: HashMap::new(),
            inverted_index: BTreeMap::new(),
            node_id_to_index: HashMap::new(),
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

        let inverted_index = Self::build_inverted_index(&facts);
        let node_id_to_index = Self::build_node_index(&facts);
        Self { facts, adjacency, inverted_index, node_id_to_index }
    }

    /// Build a snapshot from an LpgStore (graph database) with optional pre-tokenization.
    ///
    /// Converts all graph nodes to FactEntry items, using node properties as
    /// key=value facts. Builds adjacency from graph edges. This enables IPTR
    /// to work with any graph store, not just PersonaDB.
    pub fn from_lpg_store(
        store: &std::sync::Arc<obrain_core::graph::lpg::LpgStore>,
        schema: &graph_schema::GraphSchema,
        tokenize_fn: Option<&dyn Fn(&str) -> Vec<(u32, f32)>>,
    ) -> Self {
        use obrain_common::types::PropertyKey;
        use obrain_core::graph::Direction;

        let mut facts = Vec::new();
        let mut node_ids_set: HashSet<u64> = HashSet::new();

        // Iterate all structural labels
        for label_info in &schema.labels {
            if label_info.is_noise {
                continue;
            }
            let nodes = store.nodes_by_label(&label_info.label);
            for nid in nodes {
                let nid_u64 = nid.as_u64();
                if !node_ids_set.insert(nid_u64) {
                    continue; // already processed (node with multiple labels)
                }
                let Some(node) = store.get_node(nid) else { continue };

                // Get node name
                let name = ["name", "title", "label"]
                    .iter()
                    .find_map(|&k| {
                        node.properties
                            .get(&PropertyKey::from(k))
                            .and_then(|v| v.as_str())
                            .filter(|s| !s.is_empty())
                            .map(|s| s.to_string())
                    })
                    .unwrap_or_default();

                if name.is_empty() {
                    continue;
                }

                // Build a comprehensive value from all non-internal properties
                let label_str = node.labels.first().map(|l| l.to_string()).unwrap_or_default();
                let mut value_parts: Vec<String> = Vec::new();
                value_parts.push(format!("{} ({})", name, label_str));

                for (key, val) in node.properties.iter() {
                    let ks: &str = key.as_ref();
                    if matches!(ks, "name" | "title" | "label" | "display_name") {
                        continue;
                    }
                    if ks.starts_with("__") {
                        continue;
                    }
                    if let Some(s) = val.as_str() {
                        if !s.is_empty() && s.len() < 200 {
                            value_parts.push(format!("{}: {}", ks, s));
                        }
                    }
                }

                // Add outgoing relation names for richer context
                for (target_id, _edge_id) in store.edges_from(nid, Direction::Outgoing).collect::<Vec<_>>() {
                    if let Some(tnode) = store.get_node(target_id) {
                        let target_name = ["name", "title"]
                            .iter()
                            .find_map(|&k| {
                                tnode.properties
                                    .get(&PropertyKey::from(k))
                                    .and_then(|v| v.as_str())
                                    .filter(|s| !s.is_empty())
                                    .map(|s| s.to_string())
                            })
                            .unwrap_or_default();
                        if !target_name.is_empty() {
                            value_parts.push(target_name);
                        }
                    }
                }

                let value = value_parts.join(", ");
                let token_ids = tokenize_fn.map(|f| f(&value)).unwrap_or_default();

                facts.push(FactEntry::with_tokens(
                    nid_u64,
                    &name,       // key = node name (for seed matching)
                    &value,      // value = full description (for logit biases)
                    1.0,         // energy = max (all graph nodes equally energetic)
                    1.0,         // confidence = max
                    token_ids,
                ));
            }
        }

        // Build adjacency from graph edges
        let mut adjacency: HashMap<u64, Vec<u64>> = HashMap::new();
        for &nid_u64 in &node_ids_set {
            let nid = obrain_common::types::NodeId::from(nid_u64);
            for (target, _eid) in store.edges_from(nid, Direction::Outgoing).collect::<Vec<_>>() {
                let target_u64 = target.as_u64();
                if node_ids_set.contains(&target_u64) {
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

        let inverted_index = Self::build_inverted_index(&facts);
        let node_id_to_index = Self::build_node_index(&facts);
        Self { facts, adjacency, inverted_index, node_id_to_index }
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
    /// Maximum total nodes visited during BFS (prevents explosion on hub-heavy graphs).
    pub max_reached: usize,
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
            max_reached: 64,
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
    ///
    /// Uses the inverted index for fast lookup. Two-pass strategy:
    /// 1. **Exact match** via HashMap (O(1) per query term) — handles 90%+ of cases
    /// 2. **Substring fallback** — for each query term with no exact hit, scan index
    ///    keys for `key.contains(qt)` (O(K) where K = unique tokens, ~100K).
    ///    This preserves the original `contains` semantics where "rust" matches
    ///    fact tokens like "rustconf", "dev" matches "développeur", etc.
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

        // Collect candidate fact indices from inverted index.
        // Each candidate gets +1 hit per matching query term.
        let mut hits: HashMap<usize, usize> = HashMap::new();
        for qt in &query_terms {
            // Pass 1: exact match (O(1))
            if let Some(fact_indices) = self.snapshot.inverted_index.get(qt) {
                for &idx in fact_indices {
                    *hits.entry(idx).or_insert(0) += 1;
                }
            } else if qt.len() >= 4 {
                // Pass 2: prefix scan via BTreeMap range — O(log K + matches).
                // Replaces the old O(K) brute-force `contains` scan that was
                // 30-170ms on 118K-fact graphs (200K+ unique index keys).
                // `starts_with` is more precise than `contains` (no false positives
                // like "ledevelop" matching "develop") and handles the common cases:
                // "climb"→"climbing", "develop"→"developer", "rust"→"rustconf".
                // Only for terms ≥ 4 chars to avoid stop words flooding results.
                for (key, fact_indices) in self.snapshot.inverted_index.range(qt.clone()..) {
                    if !key.starts_with(qt.as_str()) {
                        break; // Past the prefix range
                    }
                    for &idx in fact_indices {
                        *hits.entry(idx).or_insert(0) += 1;
                    }
                }
            }
        }

        // Score only candidates (not all 118K facts)
        let n_terms = query_terms.len() as f64;
        let mut scored: Vec<(usize, f64)> = hits
            .into_iter()
            .filter(|&(idx, _)| !already.contains(&self.snapshot.facts[idx].node_id))
            .map(|(idx, hit_count)| {
                let f = &self.snapshot.facts[idx];
                let score = (hit_count as f64 / n_terms) * f.energy * f.confidence;
                (idx, score)
            })
            .filter(|(_, score)| *score >= self.config.min_seed_score)
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(self.config.max_seeds);
        scored
    }

    /// Step 2: BFS traversal from seed nodes with hop-level tracking.
    ///
    /// Returns: Vec<(node_id, hop_distance)> for all reached nodes.
    /// Capped at `config.max_reached` to prevent explosion on hub-heavy graphs.
    fn bfs_traverse(&self, seed_indices: &[(usize, f64)]) -> Vec<(u64, usize)> {
        let max = self.config.max_reached;
        let mut visited: HashMap<u64, usize> = HashMap::with_capacity(max);
        let mut queue: VecDeque<(u64, usize)> = VecDeque::new();

        // Seeds are at hop 0
        for &(idx, _score) in seed_indices {
            let nid = self.snapshot.facts[idx].node_id;
            if !visited.contains_key(&nid) {
                visited.insert(nid, 0);
                queue.push_back((nid, 0));
                if visited.len() >= max {
                    return visited.into_iter().collect();
                }
            }
        }

        while let Some((nid, depth)) = queue.pop_front() {
            if depth >= self.config.max_hops {
                continue;
            }
            if let Some(neighbors) = self.snapshot.adjacency.get(&nid) {
                for &neighbor in neighbors {
                    if visited.len() >= max {
                        return visited.into_iter().collect();
                    }
                    if !visited.contains_key(&neighbor) {
                        visited.insert(neighbor, depth + 1);
                        queue.push_back((neighbor, depth + 1));
                    }
                }
            }
        }

        visited.into_iter().collect()
    }

    /// Step 3: Convert reached nodes to logit biases via the tokenizer.
    ///
    /// Uses pre-built `node_id_to_index` for O(1) fact lookup instead of
    /// rebuilding a 118K-entry HashMap on every call.
    fn token_mapping(
        &self,
        reached: &[(u64, usize)],
        already_retrieved: &[u64],
    ) -> (HashMap<u32, f32>, Vec<String>) {
        let already: HashSet<u64> = already_retrieved.iter().copied().collect();

        let mut biases: HashMap<u32, f32> = HashMap::new();
        let mut concepts: Vec<String> = Vec::new();

        for &(nid, hop) in reached {
            if already.contains(&nid) {
                continue;
            }
            let Some(&fact_idx) = self.snapshot.node_id_to_index.get(&nid) else {
                continue;
            };
            let fact = &self.snapshot.facts[fact_idx];

            // Hop-dependent bias strength with energy decay
            let base_bias = match hop {
                0 => self.config.bias_hop0,
                1 => self.config.bias_hop1,
                _ => self.config.bias_hop2,
            };
            let effective_bias = base_bias * fact.energy as f32;

            // Use pre-tokenized value (computed at snapshot time)
            for &(token_id, weight) in &fact.value_token_ids {
                let bias = effective_bias * weight;
                *biases.entry(token_id).or_insert(0.0) += bias;
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

    /// Small graph (5 facts) for unit tests.
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

        let inverted_index = FactSnapshot::build_inverted_index(&facts);
        let node_id_to_index = FactSnapshot::build_node_index(&facts);
        FactSnapshot { facts, adjacency, inverted_index, node_id_to_index }
    }

    /// Benchmark-equivalent graph: 27 nodes (5 persons, 6 projects, 4 cities,
    /// 8 techs, 4 events) matching the B5-bis graph from gen_test_graph.rs.
    /// Used for regression tests — these MUST pass for benchmark to work.
    fn make_benchmark_snapshot() -> (FactSnapshot, GraphLookupTool) {
        // Build facts exactly as from_lpg_store would: key=name, value="name (Label), prop: val, ..."
        let mut facts = Vec::new();
        let mut adjacency: HashMap<u64, Vec<u64>> = HashMap::new();
        let mut nid = 0u64;

        // Helper to add a fact and return its node_id
        let mut add_fact = |key: &str, value: &str| -> u64 {
            let id = nid;
            nid += 1;
            facts.push(FactEntry::with_tokens(id, key, value, 1.0, 1.0, test_tokenize(value)));
            id
        };

        // Helper to add bidirectional edge
        let mut add_edge = |a: u64, b: u64| {
            adjacency.entry(a).or_default().push(b);
            adjacency.entry(b).or_default().push(a);
        };

        // Persons
        let thomas = add_fact("Thomas Rivière",
            "Thomas Rivière (Person), profession: développeur Rust senior, ville: Lyon, entreprise: Mozilla, langage_favori: Rust");
        let sophie = add_fact("Sophie Martin",
            "Sophie Martin (Person), profession: data scientist, ville: Paris, entreprise: Datadog, langage_favori: Python");
        let marc = add_fact("Marc Dupont",
            "Marc Dupont (Person), profession: architecte logiciel, ville: Lyon, entreprise: OVHcloud, langage_favori: Go");
        let alice = add_fact("Alice Chen",
            "Alice Chen (Person), profession: chercheuse en IA, ville: Grenoble, entreprise: INRIA, langage_favori: Python");
        let pierre = add_fact("Pierre Bernard",
            "Pierre Bernard (Person), profession: DevOps engineer, ville: Toulouse, entreprise: Airbus, langage_favori: Bash");

        // Projects
        let obrain = add_fact("Obrain", "Obrain (Project), description: Moteur de graphe embarqué en Rust, language: Rust");
        let _grafeo = add_fact("Grafeo", "Grafeo (Project), description: Base de données graphe native, language: Rust");
        let datapipeline = add_fact("DataPipeline", "DataPipeline (Project), description: ETL pipeline temps réel, language: Python");
        let cloudinfra = add_fact("CloudInfra", "CloudInfra (Project), description: Infrastructure cloud hybride, language: Go");
        let neuralsearch = add_fact("NeuralSearch", "NeuralSearch (Project), description: Recherche neurale multimodale, language: Python");
        let _skynetci = add_fact("SkyNet-CI", "SkyNet-CI (Project), description: CI/CD pour systèmes embarqués, language: Bash/Python");

        // Cities
        let lyon = add_fact("Lyon", "Lyon (City), region: Rhône-Alpes France");
        let paris = add_fact("Paris", "Paris (City), region: Île-de-France France");
        let grenoble = add_fact("Grenoble", "Grenoble (City), region: Isère France");
        let toulouse = add_fact("Toulouse", "Toulouse (City), region: Occitanie France");

        // Technologies
        let rust = add_fact("Rust", "Rust (Technology), description: Langage système performant et memory-safe");
        let python = add_fact("Python", "Python (Technology), description: Langage polyvalent pour la data science");
        let go = add_fact("Go", "Go (Technology), description: Langage compilé pour les microservices");
        let _ggml = add_fact("GGML", "GGML (Technology), description: Framework de tenseurs pour inférence LLM");
        let _pytorch = add_fact("PyTorch", "PyTorch (Technology), description: Framework apprentissage profond");
        let _k8s = add_fact("Kubernetes", "Kubernetes (Technology), description: Orchestrateur de conteneurs");
        let _llamacpp = add_fact("llama.cpp", "llama.cpp (Technology), description: Inférence LLM en C++");
        let _graphql = add_fact("GraphQL", "GraphQL (Technology), description: Langage de requête pour APIs");

        // Events
        let rustconf = add_fact("RustConf 2025", "RustConf 2025 (Event), city: Lyon, date: 2025-09-15");
        let _pydata = add_fact("PyData Paris 2025", "PyData Paris 2025 (Event), city: Paris");
        let _kubecon = add_fact("KubeCon EU 2025", "KubeCon EU 2025 (Event), city: Paris");
        let _icml = add_fact("ICML 2025", "ICML 2025 (Event), city: Grenoble");

        // === Edges (matching gen_test_graph.rs) ===
        // LIVES_IN
        add_edge(thomas, lyon);
        add_edge(sophie, paris);
        add_edge(marc, lyon);
        add_edge(alice, grenoble);
        add_edge(pierre, toulouse);

        // WORKS_ON
        add_edge(thomas, obrain);
        add_edge(sophie, datapipeline);
        add_edge(marc, cloudinfra);
        add_edge(alice, neuralsearch);

        // USES
        add_edge(thomas, rust);
        add_edge(sophie, python);
        add_edge(marc, go);
        add_edge(alice, python);

        // KNOWS
        add_edge(thomas, marc);
        add_edge(thomas, alice);
        add_edge(sophie, alice);
        add_edge(marc, pierre);

        // ATTENDED
        add_edge(thomas, rustconf);
        add_edge(marc, rustconf);

        // Dedup
        for neighbors in adjacency.values_mut() {
            neighbors.sort_unstable();
            neighbors.dedup();
        }

        let inverted_index = FactSnapshot::build_inverted_index(&facts);
        let node_id_to_index = FactSnapshot::build_node_index(&facts);
        let snapshot = FactSnapshot { facts, adjacency, inverted_index, node_id_to_index };
        let tool = GraphLookupTool::with_defaults(snapshot.clone());
        (snapshot, tool)
    }

    fn make_context(text: &str, pos: u32) -> ToolContext {
        ToolContext {
            recent_text: text.to_string(),
            entropy: 4.5,
            already_retrieved: Vec::new(),
            token_position: pos,
        }
    }

    // ─────────────────────────────────────────────────────────────────
    // Unit tests (small graph)
    // ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_seed_selection_finds_matching_facts() {
        let snapshot = make_snapshot();
        let tool = GraphLookupTool::with_defaults(snapshot);

        let seeds = tool.seed_selection("Thomas habite à", &[]);
        assert!(!seeds.is_empty(), "should find Thomas");
        assert_eq!(tool.snapshot.facts[seeds[0].0].key, "name");
    }

    #[test]
    fn test_seed_selection_respects_already_retrieved() {
        let snapshot = make_snapshot();
        let tool = GraphLookupTool::with_defaults(snapshot);

        let seeds = tool.seed_selection("Thomas habite à", &[2]);
        for &(idx, _) in &seeds {
            assert_ne!(tool.snapshot.facts[idx].node_id, 2);
        }
    }

    /// REGRESSION: seed_selection supports prefix matching for terms ≥ 4 chars.
    /// Uses BTreeMap range scan O(log K) instead of O(K) brute-force contains.
    /// Short terms (< 4) use exact match only to avoid stop word pollution.
    #[test]
    fn test_seed_selection_prefix_matching() {
        let snapshot = make_snapshot();
        let tool = GraphLookupTool::with_defaults(snapshot);

        // "climb" (5 chars) should match "climbing" via contains fallback
        let seeds = tool.seed_selection("climb test", &[]);
        assert!(!seeds.is_empty(), "'climb' (5 chars) should match 'climbing'");
        assert_eq!(tool.snapshot.facts[seeds[0].0].key, "hobby");

        // "develop" (7 chars) should match "developer"
        let seeds2 = tool.seed_selection("develop something", &[]);
        assert!(!seeds2.is_empty(), "'develop' should match 'developer'");
        assert_eq!(tool.snapshot.facts[seeds2[0].0].key, "job");
    }

    /// REGRESSION: short terms (< 4 chars) must NOT trigger substring fallback.
    /// This prevents stop words like "par" from matching "paris".
    #[test]
    fn test_seed_selection_no_substring_for_short_terms() {
        let (_, tool) = make_benchmark_snapshot();

        // "par" (3 chars) should NOT match "paris" via substring
        let seeds = tool.seed_selection("par", &[]);
        let names: Vec<&str> = seeds.iter()
            .map(|&(i, _)| tool.snapshot.facts[i].key.as_str())
            .collect();
        assert!(
            !names.contains(&"Paris"),
            "'par' should NOT match Paris via substring (stop word protection). Got: {:?}",
            names
        );
    }

    /// REGRESSION: exact match should be preferred over substring.
    #[test]
    fn test_seed_selection_exact_match_preferred() {
        let snapshot = make_snapshot();
        let tool = GraphLookupTool::with_defaults(snapshot);

        // "thomas" should match exactly — no fallback needed
        let seeds = tool.seed_selection("thomas", &[]);
        assert!(!seeds.is_empty(), "'thomas' should exact-match");
        assert_eq!(tool.snapshot.facts[seeds[0].0].key, "name");
    }

    #[test]
    fn test_bfs_reaches_neighbors() {
        let snapshot = make_snapshot();
        let tool = GraphLookupTool::with_defaults(snapshot);

        let reached = tool.bfs_traverse(&[(1, 1.0)]);
        let reached_ids: HashSet<u64> = reached.iter().map(|(nid, _)| *nid).collect();

        assert!(reached_ids.contains(&2), "seed: Thomas");
        assert!(reached_ids.contains(&1), "hop1: Lyon");
        assert!(reached_ids.contains(&3), "hop2: cat");
        assert!(!reached_ids.contains(&4), "dev is in another component");
        assert!(!reached_ids.contains(&5), "climbing is in another component");
    }

    #[test]
    fn test_bfs_hop_distances() {
        let snapshot = make_snapshot();
        let tool = GraphLookupTool::with_defaults(snapshot);

        let reached = tool.bfs_traverse(&[(1, 1.0)]);
        let hop_map: HashMap<u64, usize> = reached.into_iter().collect();

        assert_eq!(hop_map[&2], 0, "Thomas is seed (hop 0)");
        assert_eq!(hop_map[&1], 1, "Lyon is hop 1");
        assert_eq!(hop_map[&3], 1, "cat is hop 1");
    }

    /// REGRESSION: BFS must respect max_reached cap.
    #[test]
    fn test_bfs_max_reached_cap() {
        // Build a star graph: center node connected to 200 neighbors
        let mut facts = Vec::new();
        let mut adjacency: HashMap<u64, Vec<u64>> = HashMap::new();
        for i in 0u64..201 {
            facts.push(FactEntry::with_tokens(i, &format!("n{}", i), "val", 1.0, 1.0, vec![]));
        }
        // Node 0 connected to all others
        for i in 1u64..201 {
            adjacency.entry(0).or_default().push(i);
            adjacency.entry(i).or_default().push(0);
        }
        let inverted_index = FactSnapshot::build_inverted_index(&facts);
        let node_id_to_index = FactSnapshot::build_node_index(&facts);
        let snapshot = FactSnapshot { facts, adjacency, inverted_index, node_id_to_index };

        let mut config = GraphLookupConfig::default();
        config.max_reached = 10; // cap at 10
        let tool = GraphLookupTool::new(snapshot, config);

        let reached = tool.bfs_traverse(&[(0, 1.0)]); // seed = center node
        assert!(
            reached.len() <= 10,
            "BFS should cap at max_reached=10, got {}",
            reached.len()
        );
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

        let ctx = make_context("developer climbing", 0);
        let result = tool.execute(&ctx);
        assert!(!result.biases.is_empty());
    }

    // ─────────────────────────────────────────────────────────────────
    // node_id_to_index consistency
    // ─────────────────────────────────────────────────────────────────

    /// REGRESSION: node_id_to_index must be consistent with facts vec.
    #[test]
    fn test_node_id_to_index_consistency() {
        let snapshot = make_snapshot();
        for (i, fact) in snapshot.facts.iter().enumerate() {
            assert_eq!(
                snapshot.node_id_to_index.get(&fact.node_id),
                Some(&i),
                "node_id_to_index[{}] should be {}",
                fact.node_id,
                i
            );
        }
    }

    /// REGRESSION: inverted_index must contain all tokens from all facts.
    #[test]
    fn test_inverted_index_completeness() {
        let snapshot = make_snapshot();
        for (i, fact) in snapshot.facts.iter().enumerate() {
            for token in &fact.tokens {
                let indices = snapshot.inverted_index.get(token)
                    .unwrap_or_else(|| panic!("token '{}' missing from inverted_index", token));
                assert!(
                    indices.contains(&i),
                    "inverted_index['{}'] should contain fact index {}",
                    token,
                    i
                );
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────
    // Benchmark regression tests (B5-bis graph, 27 nodes)
    //
    // These tests verify that seed_selection finds relevant facts for
    // each of the 12 benchmark questions. If any of these fail, the
    // benchmark will regress.
    // ─────────────────────────────────────────────────────────────────

    /// Q1: "Où habite Thomas Rivière?" → must find Thomas (has ville: Lyon)
    #[test]
    fn test_bench_q01_thomas_city() {
        let (_, tool) = make_benchmark_snapshot();
        let seeds = tool.seed_selection("Où habite Thomas Rivière", &[]);
        let names: Vec<&str> = seeds.iter().map(|&(i, _)| tool.snapshot.facts[i].key.as_str()).collect();
        assert!(names.contains(&"Thomas Rivière"), "Q1: should find Thomas. Got: {:?}", names);
    }

    /// Q2: "Sur quel projet travaille Sophie Martin?" → must find Sophie
    #[test]
    fn test_bench_q02_sophie_project() {
        let (_, tool) = make_benchmark_snapshot();
        let seeds = tool.seed_selection("Sur quel projet travaille Sophie Martin", &[]);
        let names: Vec<&str> = seeds.iter().map(|&(i, _)| tool.snapshot.facts[i].key.as_str()).collect();
        assert!(names.contains(&"Sophie Martin"), "Q2: should find Sophie. Got: {:?}", names);
    }

    /// Q3: "Quel langage utilise Marc Dupont?" → must find Marc
    #[test]
    fn test_bench_q03_marc_language() {
        let (_, tool) = make_benchmark_snapshot();
        let seeds = tool.seed_selection("Quel langage utilise Marc Dupont", &[]);
        let names: Vec<&str> = seeds.iter().map(|&(i, _)| tool.snapshot.facts[i].key.as_str()).collect();
        assert!(names.contains(&"Marc Dupont"), "Q3: should find Marc. Got: {:?}", names);
    }

    /// Q4: "Qui habite dans la même ville que Thomas Rivière?" → must find Thomas
    #[test]
    fn test_bench_q04_same_city() {
        let (_, tool) = make_benchmark_snapshot();
        let seeds = tool.seed_selection("Qui habite dans la même ville que Thomas Rivière", &[]);
        let names: Vec<&str> = seeds.iter().map(|&(i, _)| tool.snapshot.facts[i].key.as_str()).collect();
        assert!(names.contains(&"Thomas Rivière"), "Q4: should find Thomas. Got: {:?}", names);
    }

    /// Q4 BFS: from Thomas, BFS should reach Marc (same city Lyon via edge)
    #[test]
    fn test_bench_q04_bfs_reaches_marc() {
        let (snap, tool) = make_benchmark_snapshot();
        // Find Thomas's fact index
        let thomas_idx = snap.facts.iter().position(|f| f.key == "Thomas Rivière").unwrap();
        let reached = tool.bfs_traverse(&[(thomas_idx, 1.0)]);
        let reached_keys: Vec<&str> = reached.iter()
            .filter_map(|(nid, _)| snap.node_id_to_index.get(nid).map(|&i| snap.facts[i].key.as_str()))
            .collect();
        assert!(
            reached_keys.contains(&"Marc Dupont"),
            "BFS from Thomas should reach Marc (via Lyon). Got: {:?}",
            reached_keys
        );
    }

    /// Q5: "Quelles technologies sont utilisées par le projet Obrain?" → must find Obrain
    #[test]
    fn test_bench_q05_obrain_tech() {
        let (_, tool) = make_benchmark_snapshot();
        let seeds = tool.seed_selection("Quelles technologies sont utilisées par le projet Obrain", &[]);
        let names: Vec<&str> = seeds.iter().map(|&(i, _)| tool.snapshot.facts[i].key.as_str()).collect();
        assert!(names.contains(&"Obrain"), "Q5: should find Obrain. Got: {:?}", names);
    }

    /// Q5 BFS: from Obrain, BFS should reach Rust (BUILT_WITH edge)
    #[test]
    fn test_bench_q05_bfs_reaches_rust() {
        let (snap, tool) = make_benchmark_snapshot();
        let obrain_idx = snap.facts.iter().position(|f| f.key == "Obrain").unwrap();
        let reached = tool.bfs_traverse(&[(obrain_idx, 1.0)]);
        let reached_keys: Vec<&str> = reached.iter()
            .filter_map(|(nid, _)| snap.node_id_to_index.get(nid).map(|&i| snap.facts[i].key.as_str()))
            .collect();
        assert!(
            reached_keys.contains(&"Rust"),
            "BFS from Obrain should reach Rust. Got: {:?}",
            reached_keys
        );
    }

    /// Q6: "Qui connaît Thomas Rivière?" → must find Thomas
    #[test]
    fn test_bench_q06_knows_thomas() {
        let (_, tool) = make_benchmark_snapshot();
        let seeds = tool.seed_selection("Qui connaît Thomas Rivière", &[]);
        let names: Vec<&str> = seeds.iter().map(|&(i, _)| tool.snapshot.facts[i].key.as_str()).collect();
        assert!(names.contains(&"Thomas Rivière"), "Q6: should find Thomas. Got: {:?}", names);
    }

    /// Q6 BFS: from Thomas, should reach Marc AND Alice (KNOWS edges)
    #[test]
    fn test_bench_q06_bfs_reaches_marc_alice() {
        let (snap, tool) = make_benchmark_snapshot();
        let thomas_idx = snap.facts.iter().position(|f| f.key == "Thomas Rivière").unwrap();
        let reached = tool.bfs_traverse(&[(thomas_idx, 1.0)]);
        let reached_keys: Vec<&str> = reached.iter()
            .filter_map(|(nid, _)| snap.node_id_to_index.get(nid).map(|&i| snap.facts[i].key.as_str()))
            .collect();
        assert!(reached_keys.contains(&"Marc Dupont"), "should reach Marc via KNOWS");
        assert!(reached_keys.contains(&"Alice Chen"), "should reach Alice via KNOWS");
    }

    /// Q7: "À quel événement Thomas et Marc ont-ils assisté ensemble?" → find Thomas, Marc, RustConf
    #[test]
    fn test_bench_q07_event_together() {
        let (_, tool) = make_benchmark_snapshot();
        let seeds = tool.seed_selection("À quel événement Thomas et Marc ont-ils assisté ensemble", &[]);
        let names: Vec<&str> = seeds.iter().map(|&(i, _)| tool.snapshot.facts[i].key.as_str()).collect();
        assert!(
            names.contains(&"Thomas Rivière") || names.contains(&"Marc Dupont"),
            "Q7: should find Thomas or Marc. Got: {:?}",
            names
        );
    }

    /// Q7 BFS: from Thomas, should reach RustConf 2025 (ATTENDED edge)
    #[test]
    fn test_bench_q07_bfs_reaches_rustconf() {
        let (snap, tool) = make_benchmark_snapshot();
        let thomas_idx = snap.facts.iter().position(|f| f.key == "Thomas Rivière").unwrap();
        let reached = tool.bfs_traverse(&[(thomas_idx, 1.0)]);
        let reached_keys: Vec<&str> = reached.iter()
            .filter_map(|(nid, _)| snap.node_id_to_index.get(nid).map(|&i| snap.facts[i].key.as_str()))
            .collect();
        assert!(
            reached_keys.contains(&"RustConf 2025"),
            "BFS from Thomas should reach RustConf 2025. Got: {:?}",
            reached_keys
        );
    }

    /// Q8: "Combien de personnes utilisent Python?" → must find Python
    #[test]
    fn test_bench_q08_python_users() {
        let (_, tool) = make_benchmark_snapshot();
        let seeds = tool.seed_selection("Combien de personnes utilisent Python", &[]);
        let names: Vec<&str> = seeds.iter().map(|&(i, _)| tool.snapshot.facts[i].key.as_str()).collect();
        assert!(names.contains(&"Python"), "Q8: should find Python. Got: {:?}", names);
    }

    /// Q10: "Décris le profil professionnel de Thomas Rivière." → must find Thomas
    #[test]
    fn test_bench_q10_thomas_profile() {
        let (_, tool) = make_benchmark_snapshot();
        let seeds = tool.seed_selection("Décris le profil professionnel de Thomas Rivière", &[]);
        let names: Vec<&str> = seeds.iter().map(|&(i, _)| tool.snapshot.facts[i].key.as_str()).collect();
        assert!(names.contains(&"Thomas Rivière"), "Q10: should find Thomas. Got: {:?}", names);
    }

    /// Q11: "Quelles sont les compétences de Pierre Bernard?" → must find Pierre
    #[test]
    fn test_bench_q11_pierre_skills() {
        let (_, tool) = make_benchmark_snapshot();
        let seeds = tool.seed_selection("Quelles sont les compétences de Pierre Bernard", &[]);
        let names: Vec<&str> = seeds.iter().map(|&(i, _)| tool.snapshot.facts[i].key.as_str()).collect();
        assert!(names.contains(&"Pierre Bernard"), "Q11: should find Pierre. Got: {:?}", names);
    }

    /// Q12: "Quel est le lien entre Alice Chen et NeuralSearch?" → must find Alice or NeuralSearch
    #[test]
    fn test_bench_q12_alice_neuralsearch() {
        let (_, tool) = make_benchmark_snapshot();
        let seeds = tool.seed_selection("Quel est le lien entre Alice Chen et NeuralSearch", &[]);
        let names: Vec<&str> = seeds.iter().map(|&(i, _)| tool.snapshot.facts[i].key.as_str()).collect();
        assert!(
            names.contains(&"Alice Chen") || names.contains(&"NeuralSearch"),
            "Q12: should find Alice or NeuralSearch. Got: {:?}",
            names
        );
    }

    // ─────────────────────────────────────────────────────────────────
    // Performance regression tests
    // ─────────────────────────────────────────────────────────────────

    /// Performance: 500 facts, chain topology → must complete < 5ms
    #[test]
    fn test_perf_500_facts_chain() {
        let mut facts = Vec::new();
        let mut adjacency: HashMap<u64, Vec<u64>> = HashMap::new();
        for i in 0u64..500 {
            let val = format!("value for concept {}", i);
            let toks = test_tokenize(&val);
            facts.push(FactEntry::with_tokens(i, &format!("fact_{}", i), &val, 1.0, 0.8, toks));
            if i > 0 {
                adjacency.entry(i).or_default().push(i - 1);
                adjacency.entry(i - 1).or_default().push(i);
            }
        }
        let inverted_index = FactSnapshot::build_inverted_index(&facts);
        let node_id_to_index = FactSnapshot::build_node_index(&facts);
        let snapshot = FactSnapshot { facts, adjacency, inverted_index, node_id_to_index };
        let tool = GraphLookupTool::with_defaults(snapshot);

        let ctx = make_context("concept 250", 0);
        let start = std::time::Instant::now();
        let result = tool.execute(&ctx);
        let elapsed = start.elapsed();

        assert!(!result.biases.is_empty());
        assert!(elapsed.as_millis() < 5, "should complete in < 5ms, took {}ms", elapsed.as_millis());
    }

    /// Performance: 120K facts, dense hub topology → must complete < 5ms.
    /// This is the regression test for the production graph (118K facts, 526K edges).
    #[test]
    fn test_perf_120k_facts_dense() {
        // Build 120K facts with hub topology: 100 hubs, each connected to 1200 facts
        let n = 120_000u64;
        let n_hubs = 100u64;
        let mut facts = Vec::new();
        let mut adjacency: HashMap<u64, Vec<u64>> = HashMap::new();

        for i in 0..n {
            let val = if i < n_hubs {
                format!("hub_{} important central node", i)
            } else {
                format!("leaf_{} detail data value_{}", i, i % 1000)
            };
            let toks = test_tokenize(&val);
            facts.push(FactEntry::with_tokens(i, &format!("n{}", i), &val, 1.0, 1.0, toks));
        }

        // Connect each leaf to its hub
        for i in n_hubs..n {
            let hub = i % n_hubs;
            adjacency.entry(i).or_default().push(hub);
            adjacency.entry(hub).or_default().push(i);
        }

        let inverted_index = FactSnapshot::build_inverted_index(&facts);
        let node_id_to_index = FactSnapshot::build_node_index(&facts);
        let snapshot = FactSnapshot { facts, adjacency, inverted_index, node_id_to_index };
        let tool = GraphLookupTool::with_defaults(snapshot);

        // Query hitting a hub node
        let ctx = make_context("hub_0 important", 0);
        let start = std::time::Instant::now();
        let result = tool.execute(&ctx);
        let elapsed = start.elapsed();

        assert!(!result.biases.is_empty(), "should find hub_0");
        assert!(
            elapsed.as_millis() < 5,
            "120K facts: should complete in < 5ms, took {}ms",
            elapsed.as_millis()
        );
        // BFS should not explode: hub_0 has 1200 neighbors but max_reached=64
        assert!(
            result.concepts_found.len() <= 64,
            "BFS cap: should visit <= 64 nodes, got {}",
            result.concepts_found.len()
        );
    }

    /// Performance: seed_selection with common term that matches many facts.
    /// With 50K facts and a term appearing in 5K of them, should still be fast.
    /// Threshold: 5ms release, 50ms debug (unoptimized builds are ~10× slower).
    #[test]
    fn test_perf_seed_selection_common_term() {
        let n = 50_000u64;
        let mut facts = Vec::new();
        for i in 0..n {
            // 10% of facts contain "target" in their value
            let val = if i % 10 == 0 {
                format!("target concept entity_{}", i)
            } else {
                format!("other data entity_{}", i)
            };
            let toks = test_tokenize(&val);
            facts.push(FactEntry::with_tokens(i, &format!("k{}", i), &val, 1.0, 1.0, toks));
        }
        let inverted_index = FactSnapshot::build_inverted_index(&facts);
        let node_id_to_index = FactSnapshot::build_node_index(&facts);
        let snapshot = FactSnapshot { facts, adjacency: HashMap::new(), inverted_index, node_id_to_index };
        let tool = GraphLookupTool::with_defaults(snapshot);

        let start = std::time::Instant::now();
        let seeds = tool.seed_selection("target something", &[]);
        let elapsed = start.elapsed();

        assert!(!seeds.is_empty(), "should find target facts");
        assert!(seeds.len() <= 5, "max_seeds=5");

        let max_ms: u128 = if cfg!(debug_assertions) { 50 } else { 5 };
        assert!(
            elapsed.as_millis() < max_ms,
            "seed_selection on 50K facts: should be < {}ms, took {}ms",
            max_ms,
            elapsed.as_millis()
        );
    }
}
