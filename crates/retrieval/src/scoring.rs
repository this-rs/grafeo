use anyhow::Result;
use graph_schema::{GraphSchema, get_node_name_generic, extract_node_generic, fuzzy_match_label};
use obrain_common::types::{NodeId, PropertyKey};
use obrain_core::graph::{Direction, lpg::LpgStore};
use think_filter::truncate;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use crate::engine::Engine;

/// A scored node selected for the current query.
pub struct ScoredContextNode {
    pub id: NodeId,
    pub _score: f64,
    /// GNN-derived relevance score (None if GNN unavailable or untrained)
    pub gnn_score: Option<f32>,
}

/// Retrieve and score nodes from the graph. Returns:
/// - selected nodes with scores
/// - adjacency map for the mask
/// - node_id → text for KV encoding
pub fn retrieve_nodes(
    _engine: &Engine,
    store: &Arc<LpgStore>,
    schema: &GraphSchema,
    query: &str,
    max_nodes: usize,
    token_budget: i32,
) -> Result<(Vec<ScoredContextNode>, HashMap<NodeId, HashSet<NodeId>>, HashMap<NodeId, String>)> {
    let query_lower = query.to_lowercase();
    let terms: Vec<String> = query_lower
        .split(|c: char| !c.is_alphanumeric() && c != '-' && c != '_')
        .map(|s| s.to_string())
        .filter(|s| s.len() > 1)
        .collect();

    // ── Step 1: Fuzzy match query terms → schema labels ──────────────
    let stop_words: HashSet<&str> = [
        // French
        "les", "des", "et", "le", "la", "un", "une", "du", "de", "au", "aux",
        "sont", "est", "quels", "quel", "quelle", "quelles", "sur", "dans", "pour", "avec",
        "que", "qui", "ce", "cette", "ces", "ses", "leur", "leurs", "mon", "ton", "son",
        "me", "moi", "toi", "nous", "vous", "ils", "elles", "en", "ne", "pas",
        "donne", "donner", "dit", "dire", "détails", "details", "expliquer", "explique",
        "quoi", "comment", "pourquoi", "combien", "quand", "où", "fait",
        // English
        "their", "the", "and", "what", "which", "about", "from", "with", "has", "have",
        "tell", "show", "give", "list", "all", "more", "some", "any", "been", "was", "were",
        "are", "is", "do", "does", "did", "can", "could", "would", "should",
    ].into_iter().collect();

    let mut target_labels: Vec<(String, f64)> = Vec::new(); // (label, match_score)
    let mut label_matched_terms: HashSet<String> = HashSet::new();

    for term in &terms {
        if stop_words.contains(term.as_str()) { continue; }
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
    let content_terms: Vec<&str> = terms.iter()
        .map(|t| t.as_str())
        .filter(|t| !stop_words.contains(t) && !label_matched_terms.contains(*t))
        .collect();

    // If no labels matched, use top structural labels by importance
    if target_labels.is_empty() {
        for info in &schema.labels {
            if !info.is_noise {
                target_labels.push((info.label.clone(), 0.5));
                if target_labels.len() >= 6 { break; }
            }
        }
    }

    // ── Step 2: Fetch + score nodes by target labels ─────────────────
    let mut scored_nodes: Vec<(NodeId, f64)> = Vec::new();

    for (label, label_sim) in &target_labels {
        let node_ids = store.nodes_by_label(label);
        let info = schema.labels.iter().find(|l| l.label == *label);
        let importance = info.map_or(1.0, |i| i.importance);

        for nid in node_ids {
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
            let status = store.get_node(nid)
                .and_then(|n| {
                    let sf = dp.and_then(|d| d.status_field.as_deref()).unwrap_or("status");
                    n.properties.get(&PropertyKey::from(sf))
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

    scored_nodes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored_nodes.truncate(max_nodes * 2);

    if scored_nodes.is_empty() {
        return Ok((Vec::new(), HashMap::new(), HashMap::new()));
    }

    // ── Step 3: BFS expand from seeds (structural labels only) ───────
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

    let max_depth = 1u32;
    while let Some((nid, depth)) = queue.pop_front() {
        if bfs_result.len() >= max_nodes * 5 { break; }
        if depth < max_depth {
            for neighbor in store.neighbors(nid, Direction::Both) {
                if visited.contains(&neighbor) { continue; }
                let is_structural = store.get_node(neighbor)
                    .map(|n| n.labels.iter().any(|l| schema.structural_labels.contains(l.as_ref() as &str)))
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

    // ── Step 4: Score, select, build prompt ──────────────────────────
    let mut scored: Vec<(NodeId, f64, String)> = bfs_result.iter()
        .filter_map(|(nid, _depth, retrieval_score)| {
            let (text, _labels, _summary) = extract_node_generic(store, *nid, schema);
            if text.is_empty() { return None; }
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
        if selected.len() >= max_nodes { break; }
        // Skip only if already selected as top-level (not if merely inlined as child)
        if node_texts.contains_key(nid) { continue; }

        // Inline children based on schema hierarchy
        let mut children: Vec<String> = Vec::new();
        let primary_label = store.get_node(*nid)
            .and_then(|n| n.labels.first().map(|l| l.to_string()))
            .unwrap_or_default();

        if let Some(child_defs) = schema.parent_child.get(&primary_label) {
            for (_edge_type, child_label) in child_defs.iter().take(2) {
                let max_children = 5;
                let mut child_count = 0;
                let child_dp = schema.display_props.get(child_label.as_str());

                for (target, _edge_id) in store.edges_from(*nid, Direction::Outgoing).collect::<Vec<_>>() {
                    if let Some(tnode) = store.get_node(target) {
                        if tnode.labels.iter().any(|l| l.as_ref() as &str == child_label.as_str()) {
                            let cname = get_node_name_generic(store, target, child_dp);
                            if !cname.is_empty() {
                                child_count += 1;
                                if children.len() < max_children && !child_inline_ids.contains(&target) {
                                    let cstatus = child_dp
                                        .and_then(|d| d.status_field.as_deref())
                                        .and_then(|f| tnode.properties.get(&PropertyKey::from(f)))
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("");
                                    if !cstatus.is_empty() {
                                        children.push(format!("  - {} ({})", truncate(&cname, 60), cstatus));
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
                    children.push(format!("  ... and {} more {}", child_count - max_children, child_label));
                }
            }
        }

        let text_block = if children.is_empty() {
            format!("{}\n", base_text)
        } else {
            format!("{}\n{}\n", base_text, children.join("\n"))
        };

        let est = (text_block.len() as f64 / 3.5) as i32 + 10;
        if est_tokens + est > token_budget { break; }

        node_texts.insert(*nid, text_block);
        selected.push(ScoredContextNode { id: *nid, _score: *score, gnn_score: None });
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
