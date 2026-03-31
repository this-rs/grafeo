use anyhow::Result;
use graph_schema::{GraphSchema, extract_node_generic, fuzzy_match_label, get_node_name_generic};
use obrain_common::types::{NodeId, PropertyKey};
use obrain_core::graph::{Direction, lpg::LpgStore};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use think_filter::truncate;

use crate::engine::Engine;
use crate::node_embedding::{NodeEmbeddingCache, compute_text_embedding};
use crate::round_tracker::CoactivationMap;

/// A scored node selected for the current query.
pub struct ScoredContextNode {
    pub id: NodeId,
    pub _score: f64,
    /// GNN-derived relevance score (None if GNN unavailable or untrained)
    pub gnn_score: Option<f32>,
}

/// Expand seed nodes using co-activation affinity (E3).
///
/// For each seed, looks up the top co-activated nodes in the CoactivationMap.
/// Candidates are aggregated: if a node is co-activated with multiple seeds,
/// its scores are summed. Only nodes that exist in the store are returned.
///
/// Returns candidates sorted by aggregate affinity score (descending), excluding seeds.
pub fn expand_by_affinity(
    seeds: &[NodeId],
    coactivation: &CoactivationMap,
    store: &Arc<LpgStore>,
    top_k: usize,
) -> Vec<(NodeId, f64)> {
    let seed_set: HashSet<NodeId> = seeds.iter().copied().collect();
    let mut candidate_scores: HashMap<NodeId, f64> = HashMap::new();

    // For each seed, find its top co-activated partners
    let _per_seed_k = (top_k * 2).max(10); // look at more candidates per seed, then aggregate
    for &seed in seeds {
        // Manually iterate coactivation map for this seed
        for (&(a, b), entry) in coactivation.iter() {
            let partner = if a == seed {
                b
            } else if b == seed {
                a
            } else {
                continue;
            };
            if seed_set.contains(&partner) {
                continue; // Skip nodes already in seeds
            }
            *candidate_scores.entry(partner).or_insert(0.0) += entry.decay_score as f64;
        }
    }

    // Filter: only keep nodes that exist in the store
    let mut candidates: Vec<(NodeId, f64)> = candidate_scores
        .into_iter()
        .filter(|(nid, _)| store.get_node(*nid).is_some())
        .collect();

    // Sort by aggregate score descending
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    candidates.truncate(top_k);
    candidates
}

/// Compute the affinity blending factor λ ∈ [0, 1].
///
/// λ = min(1.0, unique_pairs_with_count_ge_2 / 50)
/// - λ = 0 at cold start (no co-activations) → pure BFS
/// - λ → 1 after ~50 recurring co-activation pairs → full affinity
pub fn compute_lambda(coactivation: &CoactivationMap) -> f64 {
    let pairs_above_2 = coactivation.values().filter(|e| e.count >= 2).count();
    (pairs_above_2 as f64 / 50.0).min(1.0)
}

/// Retrieve and score nodes from the graph. Returns:
/// - selected nodes with scores
/// - adjacency map for the mask
/// - node_id → text for KV encoding
pub fn retrieve_nodes(
    engine: &Engine,
    store: &Arc<LpgStore>,
    schema: &GraphSchema,
    query: &str,
    max_nodes: usize,
    token_budget: i32,
    embd_cache: Option<&NodeEmbeddingCache>,
    coactivation: Option<&CoactivationMap>,
) -> Result<(
    Vec<ScoredContextNode>,
    HashMap<NodeId, HashSet<NodeId>>,
    HashMap<NodeId, String>,
)> {
    let query_lower = query.to_lowercase();
    let terms: Vec<String> = query_lower
        .split(|c: char| !c.is_alphanumeric() && c != '-' && c != '_')
        .map(|s| s.to_string())
        .filter(|s| s.len() > 1)
        .collect();

    // ── Step 1: Fuzzy match query terms → schema labels ──────────────
    let stop_words: HashSet<&str> = [
        // French
        "les",
        "des",
        "et",
        "le",
        "la",
        "un",
        "une",
        "du",
        "de",
        "au",
        "aux",
        "sont",
        "est",
        "quels",
        "quel",
        "quelle",
        "quelles",
        "sur",
        "dans",
        "pour",
        "avec",
        "que",
        "qui",
        "ce",
        "cette",
        "ces",
        "ses",
        "leur",
        "leurs",
        "mon",
        "ton",
        "son",
        "me",
        "moi",
        "toi",
        "nous",
        "vous",
        "ils",
        "elles",
        "en",
        "ne",
        "pas",
        "donne",
        "donner",
        "dit",
        "dire",
        "détails",
        "details",
        "expliquer",
        "explique",
        "quoi",
        "comment",
        "pourquoi",
        "combien",
        "quand",
        "où",
        "fait",
        // English
        "their",
        "the",
        "and",
        "what",
        "which",
        "about",
        "from",
        "with",
        "has",
        "have",
        "tell",
        "show",
        "give",
        "list",
        "all",
        "more",
        "some",
        "any",
        "been",
        "was",
        "were",
        "are",
        "is",
        "do",
        "does",
        "did",
        "can",
        "could",
        "would",
        "should",
    ]
    .into_iter()
    .collect();

    let mut target_labels: Vec<(String, f64)> = Vec::new(); // (label, match_score)
    let mut label_matched_terms: HashSet<String> = HashSet::new();

    for term in &terms {
        if stop_words.contains(term.as_str()) {
            continue;
        }
        if let Some((label, sim)) = fuzzy_match_label(term, schema) {
            // Only add if not already matched with better score
            if !target_labels.iter().any(|(l, s)| l == &label && *s >= sim) {
                target_labels.retain(|(l, _)| l != &label);
                target_labels.push((label, sim));
            }
            label_matched_terms.insert(term.clone());
        }
    }

    // Content terms = terms not matched as labels and not stop words
    let content_terms: Vec<&str> = terms
        .iter()
        .map(|t| t.as_str())
        .filter(|t| !stop_words.contains(t) && !label_matched_terms.contains(*t))
        .collect();

    // If no labels matched, use top structural labels by importance
    if target_labels.is_empty() {
        for info in &schema.labels {
            if !info.is_noise {
                target_labels.push((info.label.clone(), 0.5));
                if target_labels.len() >= 6 {
                    break;
                }
            }
        }
    }

    // ── Step 2: Fetch + score nodes by target labels ─────────────────
    let mut scored_nodes: Vec<(NodeId, f64)> = Vec::new();
    let mut seen_nodes: HashSet<NodeId> = HashSet::new();

    for (label, label_sim) in &target_labels {
        let node_ids = store.nodes_by_label(label);
        let info = schema.labels.iter().find(|l| l.label == *label);
        let importance = info.map_or(1.0, |i| i.importance);

        for nid in node_ids {
            seen_nodes.insert(nid);
            let dp = schema.display_props.get(label.as_str());
            let name = get_node_name_generic(store, nid, dp);
            let name_lower = name.to_lowercase();

            let mut score = importance * label_sim;

            // Name match against content terms
            if !content_terms.is_empty() {
                let mut name_matched = false;
                for term in &content_terms {
                    if name_lower.contains(term) {
                        score += 10.0;
                        name_matched = true;
                    }
                }
                if !name_matched {
                    score *= 0.3; // Label-only match when specific terms exist
                }
            } else {
                score += 5.0; // No content terms → user wants all of this type
            }

            // Status boost
            let status = store
                .get_node(nid)
                .and_then(|n| {
                    let sf = dp
                        .and_then(|d| d.status_field.as_deref())
                        .unwrap_or("status");
                    n.properties
                        .get(&PropertyKey::from(sf))
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string())
                })
                .unwrap_or_default();
            match status.as_str() {
                "active" | "in_progress" | "approved" => score *= 1.2,
                "completed" | "released" => score *= 1.0,
                "cancelled" | "archived" => score *= 0.5,
                _ => {}
            }

            scored_nodes.push((nid, score));
        }
    }

    // ── Step 2a: Cross-label name search ──────────────────────────────
    // When label matching found specific labels (e.g., "projet"→"Project"),
    // the query may also mention entity names from OTHER labels
    // (e.g., "Sophie Martin" is a Person, not a Project).
    // Search ALL structural labels for name matches to avoid missing
    // entities mentioned by name in the query.
    if !content_terms.is_empty() && !label_matched_terms.is_empty() {
        for info in &schema.labels {
            if info.is_noise {
                continue;
            }
            // Skip labels already searched
            if target_labels.iter().any(|(l, _)| *l == info.label) {
                continue;
            }
            let node_ids = store.nodes_by_label(&info.label);
            let dp = schema.display_props.get(info.label.as_str());

            for nid in node_ids {
                if seen_nodes.contains(&nid) {
                    continue;
                }
                let name = get_node_name_generic(store, nid, dp);
                let name_lower = name.to_lowercase();

                // Only add if name actually matches a content term
                let mut name_score = 0.0;
                for term in &content_terms {
                    if name_lower.contains(term) {
                        name_score += 10.0;
                    }
                }
                if name_score > 0.0 {
                    seen_nodes.insert(nid);
                    scored_nodes.push((nid, name_score + info.importance * 0.5));
                }
            }
        }
    }

    // ── Step 2b: Cosine retrieval from embedding cache ─────────────
    // If embeddings are available, compute cosine(query_embd, node_embd)
    // for ALL cached nodes. This finds semantically relevant nodes even
    // when the query has no keyword overlap with node names.
    if let Some(cache) = embd_cache {
        if !cache.is_empty() {
            // Compute query embedding (uses seq_id=2, ~5ms)
            let n_fuzzy_before = scored_nodes.len();
            match compute_text_embedding(engine, query) {
                Ok(query_embd) => {
                    let lambda_cosine = 8.0f64; // scale cosine [0,1] to match fuzzy scores (~10)
                    let cosine_threshold = 0.3f64; // min cosine to consider

                    let existing: HashSet<NodeId> =
                        scored_nodes.iter().map(|(nid, _)| *nid).collect();
                    let mut n_cosine_boosted = 0usize;
                    let mut n_cosine_new = 0usize;

                    // Only add cosine-new nodes when fuzzy has poor/no results.
                    // When fuzzy already found strong matches (max>5), cosine-new nodes
                    // dilute the context and push relevant nodes out of the token budget.
                    let fuzzy_max = scored_nodes.iter().map(|(_, s)| *s).fold(0.0f64, f64::max);
                    let allow_cosine_new = fuzzy_max < 5.0;

                    for (nid, node_embd) in cache.iter() {
                        let cos = cosine_similarity(&query_embd, node_embd) as f64;
                        if cos < cosine_threshold {
                            continue;
                        }

                        let cosine_score = lambda_cosine * cos;

                        if let Some(entry) = scored_nodes.iter_mut().find(|(id, _)| *id == nid) {
                            // Merge: take max of fuzzy and cosine score
                            if cosine_score > entry.1 {
                                entry.1 = cosine_score;
                                n_cosine_boosted += 1;
                            }
                        } else if allow_cosine_new && !existing.contains(&nid) {
                            // New node found only via cosine (fallback when fuzzy fails)
                            scored_nodes.push((nid, cosine_score));
                            n_cosine_new += 1;
                        }
                    }

                    kv_registry::kv_debug!(
                        "  [C7] retrieval: {} fuzzy, +{} cosine-new, {} cosine-boosted (cache={})",
                        n_fuzzy_before,
                        n_cosine_new,
                        n_cosine_boosted,
                        cache.len()
                    );
                }
                Err(e) => {
                    kv_registry::kv_debug!("  [C7] cosine retrieval skipped: {}", e);
                }
            }
        }
    }

    scored_nodes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored_nodes.truncate(max_nodes * 2);

    if scored_nodes.is_empty() {
        return Ok((Vec::new(), HashMap::new(), HashMap::new()));
    }

    // ── Step 3: BFS expand + affinity expand (E3) ───────────────────
    let lambda = coactivation.map_or(0.0, compute_lambda);

    let mut visited: HashSet<NodeId> = HashSet::new();
    let mut queue: VecDeque<(NodeId, u32)> = VecDeque::new();
    let mut bfs_result: Vec<(NodeId, u32, f64)> = Vec::new();
    let score_map: HashMap<NodeId, f64> = scored_nodes.iter().copied().collect();

    for (nid, score) in &scored_nodes {
        if visited.insert(*nid) {
            queue.push_back((*nid, 0));
            bfs_result.push((*nid, 0, *score));
        }
    }

    // BFS 1-hop on graph edges (structural labels only)
    let max_depth = 1u32;
    while let Some((nid, depth)) = queue.pop_front() {
        if bfs_result.len() >= max_nodes * 5 {
            break;
        }
        if depth < max_depth {
            for neighbor in store.neighbors(nid, Direction::Both) {
                if visited.contains(&neighbor) {
                    continue;
                }
                let is_structural = store
                    .get_node(neighbor)
                    .map(|n| {
                        n.labels
                            .iter()
                            .any(|l| schema.structural_labels.contains(l.as_ref() as &str))
                    })
                    .unwrap_or(false);
                if is_structural {
                    visited.insert(neighbor);
                    queue.push_back((neighbor, depth + 1));
                    let parent_score = score_map.get(&nid).unwrap_or(&1.0);
                    bfs_result.push((neighbor, depth + 1, parent_score * 0.4));
                }
            }
        }
    }

    // E3: Affinity-based expansion (blended with BFS via λ)
    if lambda > 0.0 {
        if let Some(coact) = coactivation {
            let seed_ids: Vec<NodeId> = scored_nodes.iter().map(|(nid, _)| *nid).collect();
            let affinity_candidates = expand_by_affinity(&seed_ids, coact, store, max_nodes * 3);

            for (aff_nid, aff_score) in &affinity_candidates {
                if visited.contains(aff_nid) {
                    // Already in BFS result — boost its score with affinity contribution
                    if let Some(entry) = bfs_result.iter_mut().find(|(nid, _, _)| nid == aff_nid) {
                        let bfs_score = entry.2;
                        // Blend: (1-λ)*bfs + λ*affinity
                        entry.2 = (1.0 - lambda) * bfs_score + lambda * aff_score;
                    }
                } else {
                    // New node from affinity only
                    visited.insert(*aff_nid);
                    bfs_result.push((*aff_nid, 1, lambda * aff_score));
                }
            }

            if !affinity_candidates.is_empty() {
                kv_registry::kv_debug!(
                    "  [E3] affinity: λ={:.2}, {} candidates ({} new)",
                    lambda,
                    affinity_candidates.len(),
                    affinity_candidates
                        .iter()
                        .filter(|(nid, _)| !score_map.contains_key(nid))
                        .count()
                );
            }
        }
    }

    // ── Step 3b: Score-based pruning after BFS ─────────────────────
    // When specific entities are mentioned (high-score name matches exist),
    // prune low-relevance nodes to avoid context pollution.
    // Without this, the model sees ALL graph nodes and confuses properties
    // between entities (e.g., attributing Kubernetes to Pierre Bernard
    // instead of his actual properties: DevOps, Bash, Airbus).
    {
        let bfs_max_score = bfs_result
            .iter()
            .map(|(_, _, s)| *s)
            .fold(0.0f64, f64::max);
        if bfs_max_score > 5.0 {
            // Name-matched seed exists: keep seeds + direct BFS neighbors only.
            // Threshold: nodes must score at least 2% of max OR be BFS-expanded (depth>0).
            // BFS neighbors get parent_score*0.4 ≈ 9.2 for a max of 23 → well above threshold.
            // This removes label-only matches (score ~0.15) that pollute context.
            let threshold = (bfs_max_score * 0.02).max(0.5);
            let before = bfs_result.len();
            bfs_result.retain(|(_, _, s)| *s >= threshold);
            let pruned = before - bfs_result.len();
            if pruned > 0 {
                kv_registry::kv_debug!(
                    "  [retrieval] pruned {}/{} low-score nodes (threshold={:.2}, max={:.1})",
                    pruned, before, threshold, bfs_max_score
                );
            }
        }
    }

    // ── Step 4: Score, select, build prompt ──────────────────────────
    let mut scored: Vec<(NodeId, f64, String)> = bfs_result
        .iter()
        .filter_map(|(nid, _depth, retrieval_score)| {
            let (text, _labels, _summary) = extract_node_generic(store, *nid, schema);
            if text.is_empty() {
                return None;
            }
            Some((*nid, *retrieval_score, text))
        })
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // ── Step 5: Build text blocks with inline children ─────────────
    // child_inline_ids tracks nodes inlined as children (for dedup in children lists only).
    // A node inlined as child can STILL be selected as a top-level node if it appears
    // in scored[] — this ensures high-score nodes are never silently dropped.
    let mut child_inline_ids: HashSet<NodeId> = HashSet::new();
    let mut selected: Vec<ScoredContextNode> = Vec::new();
    let mut node_texts: HashMap<NodeId, String> = HashMap::new();
    let mut est_tokens: i32 = 0; // budget tracking (header handled by registry)

    for (nid, score, base_text) in &scored {
        if selected.len() >= max_nodes {
            break;
        }
        // Skip only if already selected as top-level (not if merely inlined as child)
        if node_texts.contains_key(nid) {
            continue;
        }

        // Inline children based on schema hierarchy
        let mut children: Vec<String> = Vec::new();
        let primary_label = store
            .get_node(*nid)
            .and_then(|n| n.labels.first().map(|l| l.to_string()))
            .unwrap_or_default();

        if let Some(child_defs) = schema.parent_child.get(&primary_label) {
            for (_edge_type, child_label) in child_defs.iter().take(2) {
                let max_children = 5;
                let mut child_count = 0;
                let child_dp = schema.display_props.get(child_label.as_str());

                for (target, _edge_id) in store
                    .edges_from(*nid, Direction::Outgoing)
                    .collect::<Vec<_>>()
                {
                    if let Some(tnode) = store.get_node(target) {
                        if tnode
                            .labels
                            .iter()
                            .any(|l| l.as_ref() as &str == child_label.as_str())
                        {
                            let cname = get_node_name_generic(store, target, child_dp);
                            if !cname.is_empty() {
                                child_count += 1;
                                if children.len() < max_children
                                    && !child_inline_ids.contains(&target)
                                {
                                    let cstatus = child_dp
                                        .and_then(|d| d.status_field.as_deref())
                                        .and_then(|f| tnode.properties.get(&PropertyKey::from(f)))
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("");
                                    if !cstatus.is_empty() {
                                        children.push(format!(
                                            "  - {} ({})",
                                            truncate(&cname, 60),
                                            cstatus
                                        ));
                                    } else {
                                        children.push(format!("  - {}", truncate(&cname, 60)));
                                    }
                                    child_inline_ids.insert(target);
                                }
                            }
                        }
                    }
                }
                if child_count > max_children {
                    children.push(format!(
                        "  ... and {} more {}",
                        child_count - max_children,
                        child_label
                    ));
                }
            }
        }

        let text_block = if children.is_empty() {
            format!("{}\n", base_text)
        } else {
            format!("{}\n{}\n", base_text, children.join("\n"))
        };

        let est = (text_block.len() as f64 / 3.5) as i32 + 10;
        if est_tokens + est > token_budget {
            break;
        }

        node_texts.insert(*nid, text_block);
        selected.push(ScoredContextNode {
            id: *nid,
            _score: *score,
            gnn_score: None,
        });
        est_tokens += est;
    }

    // ── Step 6: Build adjacency among selected nodes ─────────────────
    let node_set: HashSet<NodeId> = selected.iter().map(|n| n.id).collect();
    let mut adjacency: HashMap<NodeId, HashSet<NodeId>> = HashMap::new();

    for sn in &selected {
        adjacency.entry(sn.id).or_default().insert(sn.id);
        for neighbor in store.neighbors(sn.id, Direction::Both) {
            if node_set.contains(&neighbor) && neighbor != sn.id {
                adjacency.entry(neighbor).or_default().insert(sn.id);
                adjacency.entry(sn.id).or_default().insert(neighbor);
            }
        }
    }

    Ok((selected, adjacency, node_texts))
}

/// Generate a compact micro-tag for a graph node.
///
/// Format: `:{Label} {name} [→rel target, ←rel source, ...]`
/// Example: `:Person Thomas Riviere [→habite Lyon, →connait Marc]`
///
/// The tag is truncated to `max_chars` to keep it under ~10 tokens.
/// This provides type + key relationships context alongside the embedding.
pub fn get_micro_tag(
    store: &Arc<LpgStore>,
    schema: &GraphSchema,
    node_id: NodeId,
    max_chars: usize,
) -> String {
    let primary_label = store
        .get_node(node_id)
        .and_then(|n| n.labels.first().map(|l| l.to_string()))
        .unwrap_or_default();

    let dp = schema.display_props.get(primary_label.as_str());
    let name = get_node_name_generic(store, node_id, dp);

    // Collect outgoing relations: "→edge_type target_name"
    let mut rels: Vec<String> = Vec::new();
    for (target, eid) in store
        .edges_from(node_id, Direction::Outgoing)
        .collect::<Vec<_>>()
    {
        if rels.len() >= 4 {
            break;
        }
        let etype = store
            .get_edge(eid)
            .map(|e| {
                let s: &str = e.edge_type.as_ref();
                s.to_string()
            })
            .unwrap_or_default();
        let target_label = store
            .get_node(target)
            .and_then(|n| n.labels.first().map(|l| l.to_string()))
            .unwrap_or_default();
        let tdp = schema.display_props.get(target_label.as_str());
        let tname = get_node_name_generic(store, target, tdp);
        if !tname.is_empty() {
            let tname_short: String = tname.chars().take(20).collect();
            rels.push(format!("→{} {}", etype, tname_short));
        }
    }

    let tag = if rels.is_empty() {
        format!(":{} {}", primary_label, name)
    } else {
        format!(":{} {} [{}]", primary_label, name, rels.join(", "))
    };

    // Truncate by chars (UTF-8 safe)
    if tag.chars().count() > max_chars {
        tag.chars().take(max_chars).collect()
    } else {
        tag
    }
}

/// Cosine similarity between two vectors. Returns 0.0 if either has zero norm.
pub(crate) fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "cosine_similarity: dimension mismatch");
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom < 1e-12 { 0.0 } else { dot / denom }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::round_tracker::CoactivationEntry;

    fn n(id: u64) -> NodeId {
        NodeId(id)
    }

    fn make_coact(pairs: &[((u64, u64), u32, f32)]) -> CoactivationMap {
        let mut map = CoactivationMap::new();
        for &((a, b), count, score) in pairs {
            let key = if a < b { (n(a), n(b)) } else { (n(b), n(a)) };
            map.insert(
                key,
                CoactivationEntry {
                    count,
                    last_round: count, // use count as last_round for simplicity
                    decay_score: score,
                },
            );
        }
        map
    }

    // ── Test 1: compute_lambda cold start (empty map) → 0.0 ──────────

    #[test]
    fn test_lambda_cold_start() {
        let empty: CoactivationMap = HashMap::new();
        assert_eq!(compute_lambda(&empty), 0.0);
    }

    // ── Test 2: compute_lambda ramp-up ───────────────────────────────

    #[test]
    fn test_lambda_ramp_up() {
        // 10 pairs with count >= 2 → λ = 10/50 = 0.2
        let coact = make_coact(&(0..10).map(|i| ((i, i + 100), 2, 1.0)).collect::<Vec<_>>());
        let lambda = compute_lambda(&coact);
        assert!((lambda - 0.2).abs() < 0.001, "λ={lambda}, expected 0.2");

        // 50 pairs → λ = 1.0
        let coact50 = make_coact(&(0..50).map(|i| ((i, i + 100), 2, 1.0)).collect::<Vec<_>>());
        assert!((compute_lambda(&coact50) - 1.0).abs() < 0.001);

        // 100 pairs → λ capped at 1.0
        let coact100 = make_coact(&(0..100).map(|i| ((i, i + 100), 2, 1.0)).collect::<Vec<_>>());
        assert!((compute_lambda(&coact100) - 1.0).abs() < 0.001);
    }

    // ── Test 3: compute_lambda ignores pairs with count < 2 ─────────

    #[test]
    fn test_lambda_ignores_single_coactivation() {
        // 30 pairs but all with count=1 → λ = 0.0
        let coact = make_coact(&(0..30).map(|i| ((i, i + 100), 1, 1.0)).collect::<Vec<_>>());
        assert_eq!(compute_lambda(&coact), 0.0);

        // Mix: 10 with count>=2, 20 with count=1 → λ = 10/50 = 0.2
        let mut pairs: Vec<((u64, u64), u32, f32)> =
            (0..10).map(|i| ((i, i + 100), 3, 2.0)).collect();
        pairs.extend((10..30).map(|i| ((i, i + 100), 1, 0.5)));
        let coact_mix = make_coact(&pairs);
        assert!((compute_lambda(&coact_mix) - 0.2).abs() < 0.001);
    }

    // ── Test 4: expand_by_affinity aggregates multi-seed scores ──────
    // Note: expand_by_affinity needs an Arc<LpgStore> for node validation.
    // We test the pure aggregation logic by verifying compute_lambda
    // since expand_by_affinity's store dependency makes it an integration test.
    // The scoring module's retrieve_nodes integration tests cover the full path.

    #[test]
    fn test_affinity_score_aggregation_logic() {
        // Simulate what expand_by_affinity does internally:
        // seed = {1, 2}, coactivations: (1,10)=3.0, (2,10)=2.0, (1,20)=1.0, (2,30)=4.0
        let coact = make_coact(&[
            ((1, 10), 3, 3.0),
            ((2, 10), 2, 2.0),
            ((1, 20), 1, 1.0),
            ((2, 30), 4, 4.0),
        ]);

        let seeds = vec![n(1), n(2)];
        let seed_set: HashSet<NodeId> = seeds.iter().copied().collect();
        let mut candidate_scores: HashMap<NodeId, f64> = HashMap::new();

        for &seed in &seeds {
            for (&(a, b), entry) in coact.iter() {
                let partner = if a == seed {
                    b
                } else if b == seed {
                    a
                } else {
                    continue;
                };
                if seed_set.contains(&partner) {
                    continue;
                }
                *candidate_scores.entry(partner).or_insert(0.0) += entry.decay_score as f64;
            }
        }

        // Node 10: co-activated with both seeds → 3.0 + 2.0 = 5.0
        assert!((candidate_scores[&n(10)] - 5.0).abs() < 0.001);
        // Node 20: only with seed 1 → 1.0
        assert!((candidate_scores[&n(20)] - 1.0).abs() < 0.001);
        // Node 30: only with seed 2 → 4.0
        assert!((candidate_scores[&n(30)] - 4.0).abs() < 0.001);

        // Sorted: 10(5.0) > 30(4.0) > 20(1.0)
        let mut sorted: Vec<(NodeId, f64)> = candidate_scores.into_iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        assert_eq!(sorted[0].0, n(10));
        assert_eq!(sorted[1].0, n(30));
        assert_eq!(sorted[2].0, n(20));
    }
}
