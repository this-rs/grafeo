//! obrain-chat — Generic Graph-Augmented LLM with Attention Masking
//!
//! Architecture:
//! 1. Startup: open graph DB + auto-discover schema (labels, hierarchy, properties)
//! 2. Per query: structured retrieval driven by schema (fuzzy label match + name search)
//! 3. BFS expand + topological attention mask
//! 4. Send to llama.cpp server with mask
//!
//! Schema-agnostic: works on ANY graph, not just the PO schema.

use anyhow::{Context, Result, bail};
use obrain::ObrainDB;
use obrain_common::types::{NodeId, PropertyKey, Value};
use obrain_core::graph::{Direction, lpg::LpgStore};
use serde_json::json;
use std::collections::{HashMap, HashSet, VecDeque};
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use chrono::Utc;

const DEFAULT_SERVER: &str = "http://localhost:8090";
const SYSTEM_HEADER: &str = "<|im_start|>system\nYou are a knowledge graph assistant. Below is structured data from a graph database. Each entry shows [Type] Name and its relations. Use ONLY this data to answer. Answer in the same language as the question. /no_think\n\n";

// ═══════════════════════════════════════════════════════════════════════════════
// GraphSchema — auto-discovered at startup
// ═══════════════════════════════════════════════════════════════════════════════

/// Display properties discovered for a label.
#[derive(Debug, Clone)]
struct DisplayProps {
    /// Primary name field (e.g. "name", "title")
    name_field: Option<String>,
    /// Description field (e.g. "description", "content", "text")
    desc_field: Option<String>,
    /// Status field (e.g. "status", "state")
    status_field: Option<String>,
}

/// Info about a label discovered in the graph.
#[derive(Debug, Clone)]
struct LabelInfo {
    label: String,
    count: usize,
    avg_degree: f64,
    importance: f64,
    is_noise: bool,
}

/// Auto-discovered graph schema.
struct GraphSchema {
    /// Labels sorted by importance (highest first).
    labels: Vec<LabelInfo>,
    /// Parent → Vec<(edge_type, child_label)>.
    parent_child: HashMap<String, Vec<(String, String)>>,
    /// Display properties per label.
    display_props: HashMap<String, DisplayProps>,
    /// Labels classified as noise (high count, low connectivity, no name field).
    noise_labels: HashSet<String>,
    /// Structural labels (non-noise).
    structural_labels: HashSet<String>,
}

/// Discover the graph schema by introspecting labels, edges, and properties.
fn discover_schema(store: &LpgStore) -> GraphSchema {
    let t0 = Instant::now();
    let all_labels = store.all_labels();
    let total_nodes = store.node_count().max(1);

    // ── S1.1: Label counts ───────────────────────────────────────────
    let mut label_infos: Vec<(String, usize)> = Vec::new();
    for label in &all_labels {
        let count = store.nodes_by_label(label).len();
        if count > 0 {
            label_infos.push((label.clone(), count));
        }
    }

    // ── S1.2: Sample properties per label ────────────────────────────
    let name_candidates = ["name", "title", "label", "display_name"];
    let desc_candidates = ["description", "content", "text", "message", "body", "rationale", "summary"];
    let status_candidates = ["status", "state", "phase"];

    let mut display_props: HashMap<String, DisplayProps> = HashMap::new();

    for (label, _count) in &label_infos {
        let node_ids = store.nodes_by_label(label);
        let sample_size = node_ids.len().min(50);
        let sample = &node_ids[..sample_size];

        let mut field_freq: HashMap<String, usize> = HashMap::new();
        for &nid in sample {
            if let Some(node) = store.get_node(nid) {
                for (key, val) in node.properties.iter() {
                    if val.as_str().map_or(false, |s| !s.is_empty()) {
                        *field_freq.entry(key.to_string()).or_default() += 1;
                    }
                }
            }
        }

        let find_best = |candidates: &[&str]| -> Option<String> {
            candidates.iter()
                .filter_map(|&c| {
                    let freq = field_freq.get(c).copied().unwrap_or(0);
                    if freq > sample_size / 3 { Some((c.to_string(), freq)) } else { None }
                })
                .max_by_key(|(_, f)| *f)
                .map(|(name, _)| name)
        };

        display_props.insert(label.clone(), DisplayProps {
            name_field: find_best(&name_candidates),
            desc_field: find_best(&desc_candidates),
            status_field: find_best(&status_candidates),
        });
    }

    // ── S1.3: Edge types between labels → hierarchy ──────────────────
    // Sample edges from each label to discover parent→child patterns
    let mut edge_patterns: HashMap<(String, String, String), usize> = HashMap::new(); // (src_label, edge_type, dst_label) → count

    for (label, _count) in &label_infos {
        let node_ids = store.nodes_by_label(label);
        let sample_size = node_ids.len().min(100);
        for &nid in &node_ids[..sample_size] {
            for (target, edge_id) in store.edges_from(nid, Direction::Outgoing).collect::<Vec<_>>() {
                if let Some(tnode) = store.get_node(target) {
                    let etype = store.edge_type(edge_id)
                        .map(|s| s.to_string())
                        .unwrap_or_default();
                    let tlabel = tnode.labels.first()
                        .map(|l| l.to_string())
                        .unwrap_or_default();
                    if !etype.is_empty() && !tlabel.is_empty() {
                        *edge_patterns.entry((label.clone(), etype, tlabel)).or_default() += 1;
                    }
                }
            }
        }
    }

    // Build parent_child: keep edges that appear in >20% of sampled parents
    let mut parent_child: HashMap<String, Vec<(String, String)>> = HashMap::new();
    for ((src, etype, dst), count) in &edge_patterns {
        let src_count = label_infos.iter().find(|(l, _)| l == src).map(|(_, c)| *c).unwrap_or(1);
        let sample_size = src_count.min(100);
        if *count > sample_size / 5 && src != dst {
            parent_child.entry(src.clone())
                .or_default()
                .push((etype.clone(), dst.clone()));
        }
    }
    // Deduplicate
    for children in parent_child.values_mut() {
        children.sort();
        children.dedup();
    }

    // ── S1.4: Importance = (1/log(count)) × avg_degree ───────────────
    let mut labels_with_importance: Vec<LabelInfo> = Vec::new();
    for (label, count) in &label_infos {
        let node_ids = store.nodes_by_label(label);
        let sample_size = node_ids.len().min(100);
        let total_degree: usize = node_ids[..sample_size].iter()
            .map(|&nid| store.neighbors(nid, Direction::Both).count())
            .sum();
        let avg_degree = if sample_size > 0 { total_degree as f64 / sample_size as f64 } else { 0.0 };

        let count_f = (*count as f64).max(2.0);
        let importance = (1.0 / count_f.ln()) * (avg_degree + 1.0);

        labels_with_importance.push(LabelInfo {
            label: label.clone(),
            count: *count,
            avg_degree,
            importance,
            is_noise: false, // set in S1.5
        });
    }

    // ── S1.5: Classify noise vs structural ───────────────────────────
    let mut noise_labels: HashSet<String> = HashSet::new();
    let mut structural_labels: HashSet<String> = HashSet::new();

    for info in &mut labels_with_importance {
        let frac = info.count as f64 / total_nodes as f64;
        let has_name = display_props.get(&info.label)
            .map_or(false, |d| d.name_field.is_some());

        // Noise: >5% of total nodes AND (low connectivity OR no name field)
        if frac > 0.05 && (info.avg_degree < 2.0 || !has_name) {
            info.is_noise = true;
            noise_labels.insert(info.label.clone());
        } else {
            structural_labels.insert(info.label.clone());
        }
    }

    labels_with_importance.sort_by(|a, b| b.importance.partial_cmp(&a.importance).unwrap_or(std::cmp::Ordering::Equal));

    eprintln!("  Schema discovered in {:.0}ms:", t0.elapsed().as_millis());
    eprintln!("    {} labels ({} structural, {} noise)",
        labels_with_importance.len(), structural_labels.len(), noise_labels.len());
    for info in labels_with_importance.iter().take(10) {
        let marker = if info.is_noise { " [noise]" } else { "" };
        eprintln!("      {:>6} {:20} imp={:.3} deg={:.1}{}",
            info.count, info.label, info.importance, info.avg_degree, marker);
    }
    if let Some(top_parent) = parent_child.iter().next() {
        eprintln!("    Hierarchy sample: {} → {:?}", top_parent.0,
            top_parent.1.iter().map(|(e, c)| format!("--{}--> {}", e, c)).collect::<Vec<_>>());
    }

    GraphSchema {
        labels: labels_with_importance,
        parent_child,
        display_props,
        noise_labels,
        structural_labels,
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Fuzzy label matching
// ═══════════════════════════════════════════════════════════════════════════════

/// Normalized Levenshtein distance (0.0 = identical, 1.0 = completely different).
fn levenshtein_norm(a: &str, b: &str) -> f64 {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let (la, lb) = (a_chars.len(), b_chars.len());
    if la == 0 && lb == 0 { return 0.0; }
    if la == 0 || lb == 0 { return 1.0; }

    let mut prev: Vec<usize> = (0..=lb).collect();
    let mut curr = vec![0; lb + 1];

    for i in 1..=la {
        curr[0] = i;
        for j in 1..=lb {
            let cost = if a_chars[i - 1] == b_chars[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    let dist = prev[lb];
    1.0 - (dist as f64 / la.max(lb) as f64)
}

/// Try to match a query term to a schema label. Returns (label, similarity).
fn fuzzy_match_label(term: &str, schema: &GraphSchema) -> Option<(String, f64)> {
    let term_lower = term.to_lowercase();
    let mut best: Option<(String, f64)> = None;

    for info in &schema.labels {
        if info.is_noise { continue; }
        let label_lower = info.label.to_lowercase();

        // Exact match
        if term_lower == label_lower {
            return Some((info.label.clone(), 1.0));
        }

        // Check with/without trailing 's' (simple plural)
        let candidates = vec![
            label_lower.clone(),
            format!("{}s", label_lower),
            if label_lower.ends_with('s') { label_lower[..label_lower.len()-1].to_string() } else { String::new() },
        ];

        for candidate in &candidates {
            if candidate.is_empty() { continue; }
            // Prefix match (e.g. "proj" matches "project")
            if candidate.starts_with(&term_lower) || term_lower.starts_with(candidate.as_str()) {
                let sim = term_lower.len().min(candidate.len()) as f64 / term_lower.len().max(candidate.len()) as f64;
                if sim > 0.6 {
                    if best.as_ref().map_or(true, |(_, s)| sim > *s) {
                        best = Some((info.label.clone(), sim));
                    }
                }
            }
            // Levenshtein
            let sim = levenshtein_norm(&term_lower, candidate);
            if sim >= 0.70 {
                if best.as_ref().map_or(true, |(_, s)| sim > *s) {
                    best = Some((info.label.clone(), sim));
                }
            }
        }
    }

    best
}

// ═══════════════════════════════════════════════════════════════════════════════
// Generic node text extraction
// ═══════════════════════════════════════════════════════════════════════════════

fn get_node_name_generic(store: &LpgStore, node_id: NodeId, props: Option<&DisplayProps>) -> String {
    store.get_node(node_id)
        .and_then(|n| {
            if let Some(dp) = props {
                if let Some(ref field) = dp.name_field {
                    return n.properties.get(&PropertyKey::from(field.as_str()))
                        .and_then(|v| v.as_str())
                        .filter(|s| !s.is_empty())
                        .map(|s| s.to_string());
                }
            }
            // Fallback: try common names
            ["name", "title", "label"].iter()
                .filter_map(|&k| n.properties.get(&PropertyKey::from(k))
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty()))
                .next()
                .map(|s| s.to_string())
                .or_else(|| {
                    // For Files: use path (basename)
                    n.properties.get(&PropertyKey::from("path"))
                        .and_then(|v| v.as_str())
                        .filter(|s| !s.is_empty())
                        .map(|s| s.rsplit('/').next().unwrap_or(s).to_string())
                })
                .or_else(|| {
                    // For Notes/text: first 40 chars of content
                    n.properties.get(&PropertyKey::from("content"))
                        .and_then(|v| v.as_str())
                        .filter(|s| !s.is_empty())
                        .map(|s| {
                            let clean: String = s.chars()
                                .filter(|c| !matches!(c, '#' | '*' | '\n' | '\r'))
                                .take(50)
                                .collect();
                            let t = clean.trim();
                            let truncated: String = t.chars().take(40).collect();
                            if truncated.len() < t.len() { format!("{}…", truncated) } else { t.to_string() }
                        })
                })
        })
        .unwrap_or_default()
}

fn extract_node_generic(store: &LpgStore, node_id: NodeId, schema: &GraphSchema) -> (String, String, String) {
    let empty = (String::new(), String::new(), String::new());
    let node = match store.get_node(node_id) {
        Some(n) => n,
        None => return empty,
    };

    let labels = node.labels.iter().map(|l| l.to_string()).collect::<Vec<_>>().join(",");
    let primary_label = node.labels.first().map(|l| l.to_string()).unwrap_or_default();
    let dp = schema.display_props.get(&primary_label);

    // Name
    let name = if let Some(dp) = dp {
        if let Some(ref field) = dp.name_field {
            node.properties.get(&PropertyKey::from(field.as_str()))
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .map(|s| truncate(s, 80))
                .unwrap_or_default()
        } else { String::new() }
    } else {
        ["name", "title"].iter()
            .filter_map(|&k| node.properties.get(&PropertyKey::from(k))
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty()))
            .next()
            .map(|s| truncate(s, 80))
            .unwrap_or_default()
    };

    // Description
    let desc = if let Some(dp) = dp {
        if let Some(ref field) = dp.desc_field {
            node.properties.get(&PropertyKey::from(field.as_str()))
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty() && s != &name)
                .map(|s| {
                    let first_line = s.lines()
                        .map(|l| l.trim())
                        .find(|l| !l.is_empty() && !l.starts_with('#') && !l.starts_with("---"))
                        .unwrap_or(s);
                    truncate(first_line, 150)
                })
                .unwrap_or_default()
        } else { String::new() }
    } else {
        ["description", "content", "text", "message", "body", "rationale"].iter()
            .filter_map(|&k| node.properties.get(&PropertyKey::from(k))
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty() && s != &name))
            .next()
            .map(|s| {
                let first_line = s.lines()
                    .map(|l| l.trim())
                    .find(|l| !l.is_empty() && !l.starts_with('#') && !l.starts_with("---"))
                    .unwrap_or(s);
                truncate(first_line, 150)
            })
            .unwrap_or_default()
    };

    // Status
    let status = if let Some(dp) = dp {
        if let Some(ref field) = dp.status_field {
            node.properties.get(&PropertyKey::from(field.as_str()))
                .and_then(|v| v.as_str())
                .unwrap_or("")
        } else { "" }
    } else {
        node.properties.get(&PropertyKey::from("status"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
    };

    let summary = if !name.is_empty() { name.clone() } else { truncate(&desc, 60) };

    // Fallback for nodes with no name/desc: find any non-trivial string property
    if name.is_empty() && desc.is_empty() {
        for (key, val) in node.properties.iter() {
            if let Some(s) = val.as_str() {
                let ks = key.as_ref() as &str;
                if ks.starts_with("cc_") || ks == "hash" || ks == "embedding_model"
                    || ks.ends_with("_fingerprint") || ks.ends_with("_dna")
                    || ks == "project_id" || ks == "workspace_id" || ks == "id" { continue; }
                if !s.is_empty() && s.len() > 2 && s.len() < 500 {
                    let text = format!("[{}] {}: {}", labels, ks, truncate(s, 150));
                    return (text, labels, truncate(s, 60));
                }
            }
        }
        return empty;
    }

    let text = if !desc.is_empty() {
        if !status.is_empty() {
            format!("[{}] {} — {} ({})", labels, name, desc, status)
        } else {
            format!("[{}] {} — {}", labels, name, desc)
        }
    } else if !status.is_empty() {
        format!("[{}] {} ({})", labels, name, status)
    } else {
        format!("[{}] {}", labels, name)
    };

    (text, labels, summary)
}

// ═══════════════════════════════════════════════════════════════════════════════
// KV Node Registry — tracks what's in the KV cache across queries
// ═══════════════════════════════════════════════════════════════════════════════

/// A slot in the KV cache occupied by one graph node's text.
#[derive(Debug, Clone)]
struct KvSlot {
    /// Token position range [start, end) in the KV cache.
    start: i32,
    end: i32,
    /// Number of tokens.
    n_tokens: i32,
    /// When this slot was last used (query counter).
    last_used: u64,
    /// The text that was encoded for this node (needed to reconstruct prompt prefix).
    text: String,
}

/// Metrics for KV cache usage.
struct KvMetrics {
    total_queries: u64,
    cache_hits: u64,
    cache_misses: u64,
    encode_time_ms: u128,
}

impl KvMetrics {
    fn new() -> Self {
        Self { total_queries: 0, cache_hits: 0, cache_misses: 0, encode_time_ms: 0 }
    }

    fn hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 { 0.0 } else { self.cache_hits as f64 / total as f64 }
    }
}

/// Registry that tracks which graph nodes are encoded in the KV cache.
///
/// With `cache_prompt: true`, llama.cpp reuses the KV cache when the prompt
/// prefix matches. So we keep nodes in a stable order — header first, then
/// nodes in insertion order. New nodes are appended at the end.
struct KvNodeRegistry {
    /// NodeId → KvSlot for nodes currently in KV.
    nodes: HashMap<NodeId, KvSlot>,
    /// Insertion-ordered list of node IDs (matches KV position order).
    order: Vec<NodeId>,
    /// End of the system header in token positions.
    header_end: i32,
    /// Next free token position.
    next_pos: i32,
    /// Query counter for LRU tracking.
    query_counter: u64,
    /// Metrics.
    metrics: KvMetrics,
    /// The system header text (for prompt reconstruction).
    header_text: String,
}

impl KvNodeRegistry {
    fn new(header_text: &str, header_tokens: i32) -> Self {
        Self {
            nodes: HashMap::new(),
            order: Vec::new(),
            header_end: header_tokens,
            next_pos: header_tokens,
            query_counter: 0,
            metrics: KvMetrics::new(),
            header_text: header_text.to_string(),
        }
    }

    /// Check which nodes need to be loaded (not yet in KV).
    fn find_missing(&self, needed: &[NodeId]) -> Vec<NodeId> {
        needed.iter()
            .filter(|id| !self.nodes.contains_key(id))
            .copied()
            .collect()
    }

    /// Touch nodes that are already in KV (update last_used).
    fn touch(&mut self, ids: &[NodeId]) {
        for id in ids {
            if let Some(slot) = self.nodes.get_mut(id) {
                slot.last_used = self.query_counter;
                self.metrics.cache_hits += 1;
            }
        }
    }

    /// Register newly encoded nodes in the KV.
    fn register(&mut self, node_id: NodeId, text: &str, n_tokens: i32) {
        let slot = KvSlot {
            start: self.next_pos,
            end: self.next_pos + n_tokens,
            n_tokens,
            last_used: self.query_counter,
            text: text.to_string(),
        };
        self.next_pos += n_tokens;
        self.nodes.insert(node_id, slot);
        self.order.push(node_id);
        self.metrics.cache_misses += 1;
    }

    /// Get the KV position range for a node (if loaded).
    fn get_slot(&self, node_id: NodeId) -> Option<&KvSlot> {
        self.nodes.get(&node_id)
    }

    /// Reconstruct the full prompt prefix (header + all loaded nodes in order).
    /// This is what llama.cpp needs for `cache_prompt: true` matching.
    fn reconstruct_prompt(&self) -> String {
        let mut prompt = self.header_text.clone();
        for nid in &self.order {
            if let Some(slot) = self.nodes.get(nid) {
                prompt.push_str(&slot.text);
            }
        }
        prompt
    }

    /// Start a new query — increment counter.
    fn begin_query(&mut self) {
        self.query_counter += 1;
        self.metrics.total_queries += 1;
    }

    /// Log KV metrics to stderr.
    fn log_metrics(&self) {
        let occupancy = self.next_pos;
        let n_nodes = self.nodes.len();
        eprintln!("  KV: {} nodes, {} tokens, hit_rate={:.0}% (hits={}, misses={}, queries={})",
            n_nodes, occupancy,
            self.metrics.hit_rate() * 100.0,
            self.metrics.cache_hits, self.metrics.cache_misses,
            self.metrics.total_queries);
    }

    /// Evict least-recently-used nodes to free up token capacity.
    ///
    /// `tokens_needed` = how many tokens we need to free.
    /// `protected` = node IDs that must NOT be evicted (current query's nodes).
    ///
    /// Strategy: sort by last_used ascending, evict oldest first.
    /// After eviction, rebuild the position map (compact) since we use
    /// `cache_prompt: true` which needs a contiguous prefix.
    fn evict_lru(&mut self, tokens_needed: i32, protected: &HashSet<NodeId>) -> Vec<NodeId> {
        // Sort nodes by last_used (ascending = oldest first)
        let mut candidates: Vec<(NodeId, u64, i32)> = self.nodes.iter()
            .filter(|(id, _)| !protected.contains(id))
            .map(|(id, slot)| (*id, slot.last_used, slot.n_tokens))
            .collect();
        candidates.sort_by_key(|(_, last_used, _)| *last_used);

        let mut freed = 0i32;
        let mut evicted: Vec<NodeId> = Vec::new();

        for (nid, last_used, n_tokens) in &candidates {
            if freed >= tokens_needed { break; }
            eprintln!("    evict: node {} (last_q={}, {} tokens)", nid.0, last_used, n_tokens);
            evicted.push(*nid);
            freed += n_tokens;
        }

        // Remove evicted nodes
        for nid in &evicted {
            self.nodes.remove(nid);
        }
        self.order.retain(|id| !evicted.contains(id));

        // Rebuild positions (compact — no gaps)
        self.rebuild_positions();

        eprintln!("  Evicted {} nodes, freed ~{} tokens", evicted.len(), freed);
        evicted
    }

    /// Rebuild all KV positions from scratch after eviction.
    /// Keeps insertion order, reassigns contiguous positions.
    /// The prompt will be reconstructed, and `cache_prompt: true`
    /// will match the longest common prefix automatically.
    fn rebuild_positions(&mut self) {
        let mut pos = self.header_end;
        for nid in &self.order {
            if let Some(slot) = self.nodes.get_mut(nid) {
                let n_tok = slot.n_tokens;
                slot.start = pos;
                slot.end = pos + n_tok;
                pos += n_tok;
            }
        }
        self.next_pos = pos;
    }

    /// Ensure capacity for `needed_tokens` new tokens.
    /// If over `max_kv_tokens`, evict LRU nodes.
    fn ensure_capacity(&mut self, needed_tokens: i32, max_kv_tokens: i32, protected: &HashSet<NodeId>) {
        let available = max_kv_tokens - self.next_pos;
        if available >= needed_tokens { return; }

        let to_free = needed_tokens - available;
        eprintln!("  KV full: {} tokens used, need {} more (capacity {}), evicting...",
            self.next_pos, needed_tokens, max_kv_tokens);
        self.evict_lru(to_free, protected);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// KV Banks — semantic groupings of nodes
// ═══════════════════════════════════════════════════════════════════════════════

/// A semantic bank = a group of related nodes that can be loaded/evicted together.
#[derive(Clone)]
struct KvBank {
    /// Human-readable name (e.g. "Project: Elun").
    name: String,
    /// Root node (the "anchor" of this bank, e.g. the Project node).
    root: NodeId,
    /// All node IDs in this bank (root + children).
    node_ids: Vec<NodeId>,
    /// Pre-built text blocks for each node (node_id → text).
    texts: HashMap<NodeId, String>,
    /// Estimated total tokens.
    est_tokens: i32,
}

/// Try to load banks from an obrain cache DB. Returns None if cache is stale or missing.
fn load_bank_cache(cache_path: &Path, node_count: usize, edge_count: usize) -> Option<Vec<KvBank>> {
    if !cache_path.exists() { return None; }

    // Remove stale checkpoint.meta (same workaround as main DB)
    let ckpt = cache_path.join("wal/checkpoint.meta");
    let _ = std::fs::remove_file(&ckpt);

    let db = ObrainDB::open(cache_path.to_str()?).ok()?;
    let store = db.store();

    // Check fingerprint
    let meta_ids = store.nodes_by_label("Meta");
    let meta_id = meta_ids.first()?;
    let meta = store.get_node(*meta_id)?;
    let cached_nc = meta.properties.get(&PropertyKey::from("node_count"))
        .and_then(|v| if let Value::Int64(n) = v { Some(*n as usize) } else { None })?;
    let cached_ec = meta.properties.get(&PropertyKey::from("edge_count"))
        .and_then(|v| if let Value::Int64(n) = v { Some(*n as usize) } else { None })?;
    if cached_nc != node_count || cached_ec != edge_count {
        return None; // graph changed, cache stale
    }

    // Load banks
    let bank_ids = store.nodes_by_label("Bank");
    let mut banks: Vec<(i64, KvBank)> = Vec::new();

    for &bank_id in &bank_ids {
        let bank_node = store.get_node(bank_id)?;
        let name = bank_node.properties.get(&PropertyKey::from("name"))
            .and_then(|v| v.as_str())?.to_string();
        let root_id = bank_node.properties.get(&PropertyKey::from("root_id"))
            .and_then(|v| if let Value::Int64(n) = v { Some(*n) } else { None })?;
        let est_tokens = bank_node.properties.get(&PropertyKey::from("est_tokens"))
            .and_then(|v| if let Value::Int64(n) = v { Some(*n as i32) } else { None })?;
        let order = bank_node.properties.get(&PropertyKey::from("order"))
            .and_then(|v| if let Value::Int64(n) = v { Some(*n) } else { None })
            .unwrap_or(0);

        let mut node_ids = Vec::new();
        let mut texts: HashMap<NodeId, String> = HashMap::new();

        // Read entries via HAS_ENTRY edges
        for (target, _eid) in store.edges_from(bank_id, Direction::Outgoing).collect::<Vec<_>>() {
            if let Some(entry) = store.get_node(target) {
                let source_id = entry.properties.get(&PropertyKey::from("source_id"))
                    .and_then(|v| if let Value::Int64(n) = v { Some(NodeId(*n as u64)) } else { None });
                let text = entry.properties.get(&PropertyKey::from("text"))
                    .and_then(|v| v.as_str()).map(|s| s.to_string());
                if let (Some(sid), Some(txt)) = (source_id, text) {
                    node_ids.push(sid);
                    texts.insert(sid, txt);
                }
            }
        }

        banks.push((order, KvBank {
            name,
            root: NodeId(root_id as u64),
            node_ids,
            texts,
            est_tokens,
        }));
    }

    // Sort by original order
    banks.sort_by_key(|(order, _)| *order);
    Some(banks.into_iter().map(|(_, b)| b).collect())
}

/// Save banks to an obrain cache DB.
fn save_bank_cache(cache_path: &Path, banks: &[KvBank], node_count: usize, edge_count: usize) {
    // Remove old cache entirely and recreate
    let _ = std::fs::remove_dir_all(cache_path);

    let db = match ObrainDB::open(cache_path.to_str().unwrap_or_default()) {
        Ok(db) => db,
        Err(e) => {
            eprintln!("  Warning: could not create bank cache DB: {e}");
            return;
        }
    };

    // Meta node (fingerprint)
    db.create_node_with_props(&["Meta"], [
        ("node_count", Value::Int64(node_count as i64)),
        ("edge_count", Value::Int64(edge_count as i64)),
    ]);

    // One Bank node per bank, with Entry children
    for (i, bank) in banks.iter().enumerate() {
        let bank_id = db.create_node_with_props(&["Bank"], [
            ("name", Value::String(bank.name.clone().into())),
            ("root_id", Value::Int64(bank.root.0 as i64)),
            ("est_tokens", Value::Int64(bank.est_tokens as i64)),
            ("order", Value::Int64(i as i64)),
        ]);

        for nid in &bank.node_ids {
            let text = bank.texts.get(nid).cloned().unwrap_or_default();
            let entry_id = db.create_node_with_props(&["Entry"], [
                ("source_id", Value::Int64(nid.0 as i64)),
                ("text", Value::String(text.into())),
            ]);
            db.create_edge(bank_id, entry_id, "HAS_ENTRY");
        }
    }
}

/// Discover banks from the graph schema hierarchy.
/// Each top-level node + its hierarchy (BFS up to depth 3) = 1 bank.
/// Token budget per bank prevents explosion on highly connected nodes.
fn discover_banks(
    store: &LpgStore,
    schema: &GraphSchema,
    max_banks: usize,
) -> Vec<KvBank> {
    let t0 = Instant::now();

    // Use the highest importance label as the bank anchor
    let top_label = match schema.labels.iter().find(|l| !l.is_noise) {
        Some(l) => &l.label,
        None => return Vec::new(),
    };

    let roots = store.nodes_by_label(top_label);

    // Priority labels for bank content (skip noise-like high-volume labels)
    let priority_child_labels: Vec<&str> = schema.labels.iter()
        .filter(|l| !l.is_noise && l.label != *top_label)
        .filter(|l| {
            // Skip very high-volume labels that aren't informative as bank content
            let dominated = matches!(l.label.as_str(),
                "ChatSession" | "ChatEvent" | "File" | "Import" |
                "Function" | "Struct" | "Enum" | "Impl" | "Trait" |
                "Commit" | "TouchedNode" | "TrajectoryNode" | "Trajectory" |
                "ProtocolState" | "ProtocolTransition" | "ProtocolRun" |
                "TriggerFiring" | "RefreshToken" | "AgentExecution" | "Alert"
            );
            !dominated
        })
        .map(|l| l.label.as_str())
        .collect();

    const MAX_TOKENS_PER_BANK: i32 = 500; // Token budget per bank
    const MAX_DEPTH: usize = 2;           // BFS depth from root
    const MAX_CHILDREN_PER_EDGE_TYPE: usize = 5; // Max children per edge type per level

    let mut banks: Vec<KvBank> = Vec::new();

    for &root_id in roots.iter().take(max_banks) {
        let root_dp = schema.display_props.get(top_label.as_str());
        let root_name = get_node_name_generic(store, root_id, root_dp);
        if root_name.is_empty() { continue; }

        let mut node_ids = vec![root_id];
        let mut texts: HashMap<NodeId, String> = HashMap::new();
        let mut visited: HashSet<NodeId> = HashSet::new();
        visited.insert(root_id);
        let mut est_tokens: i32 = 0;

        // Add root text
        let (root_text, _, _) = extract_node_generic(store, root_id, schema);
        if !root_text.is_empty() {
            let tok = (root_text.len() as f64 / 3.5) as i32 + 5;
            est_tokens += tok;
            texts.insert(root_id, format!("{}\n", root_text));
        }

        // BFS through hierarchy with token budget
        let mut frontier: Vec<NodeId> = vec![root_id];
        for _depth in 0..MAX_DEPTH {
            if est_tokens >= MAX_TOKENS_PER_BANK { break; }
            let mut next_frontier: Vec<NodeId> = Vec::new();

            for &parent_id in &frontier {
                if est_tokens >= MAX_TOKENS_PER_BANK { break; }

                // Group outgoing edges by target label
                let mut by_label: HashMap<String, Vec<NodeId>> = HashMap::new();
                for (target, _eid) in store.edges_from(parent_id, Direction::Outgoing).collect::<Vec<_>>() {
                    if visited.contains(&target) { continue; }
                    if let Some(tnode) = store.get_node(target) {
                        for label in &tnode.labels {
                            let lstr = label.as_ref() as &str;
                            if priority_child_labels.contains(&lstr) {
                                by_label.entry(lstr.to_string()).or_default().push(target);
                                break;
                            }
                        }
                    }
                }

                // Take top N children per label type, sorted by importance
                for (_label, targets) in &by_label {
                    for &target in targets.iter().take(MAX_CHILDREN_PER_EDGE_TYPE) {
                        if est_tokens >= MAX_TOKENS_PER_BANK { break; }
                        if !visited.insert(target) { continue; }

                        let (ctext, _, _) = extract_node_generic(store, target, schema);
                        if ctext.is_empty() { continue; }

                        let tok = (ctext.len() as f64 / 3.5) as i32 + 5;
                        node_ids.push(target);
                        est_tokens += tok;
                        texts.insert(target, format!("{}\n", ctext));
                        next_frontier.push(target);
                    }
                }
            }

            frontier = next_frontier;
        }

        banks.push(KvBank {
            name: format!("{}: {}", top_label, root_name),
            root: root_id,
            node_ids,
            texts,
            est_tokens,
        });
    }

    // Sort banks by token count descending (richest first)
    banks.sort_by(|a, b| b.est_tokens.cmp(&a.est_tokens));

    eprintln!("  Discovered {} banks in {:.0}ms (top label: {})",
        banks.len(), t0.elapsed().as_millis(), top_label);
    for bank in banks.iter().take(8) {
        eprintln!("    {} ({} nodes, ~{} tokens)",
            bank.name, bank.node_ids.len(), bank.est_tokens);
    }

    banks
}

// ═══════════════════════════════════════════════════════════════════════════════
// Context structures
// ═══════════════════════════════════════════════════════════════════════════════

struct ContextNode {
    id: NodeId,
    token_start: i32,
    token_end: i32,
}

struct QueryContext {
    nodes: Vec<ContextNode>,
    adjacency: HashMap<NodeId, HashSet<NodeId>>,
    prompt: String,
    total_tokens: i32,
    header_tokens: i32,
    edge_count: usize,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Conversation Fragment Registry — conversation turns as KV cache nodes
// ═══════════════════════════════════════════════════════════════════════════════

/// Conversation fragments live in NodeId space 0xFFFF_0000_0000_xxxx
/// to avoid collision with graph NodeIds.
const CONV_NODE_BASE: u64 = 0xFFFF_0000_0000_0000;

/// A conversation fragment — a concise Q&A summary stored as a KV node.
#[derive(Clone, Debug)]
struct ConvFragment {
    /// Virtual NodeId in the KV registry.
    node_id: NodeId,
    /// The question asked by the user.
    question: String,
    /// Key terms from the question (for matching).
    terms: Vec<String>,
    /// Concise summary text registered in the KV.
    kv_text: String,
    /// Which graph NodeIds were relevant to this Q&A (for attention links).
    related_graph_nodes: Vec<NodeId>,
    /// Turn number.
    turn: u32,
}

/// Manages conversation fragments as KV-cache-resident nodes.
struct ConvFragments {
    fragments: Vec<ConvFragment>,
    next_turn: u32,
}

impl ConvFragments {
    fn new() -> Self {
        Self { fragments: Vec::new(), next_turn: 0 }
    }

    /// Create a concise fragment from a Q&A exchange and register it in the KV.
    fn add_turn(
        &mut self,
        question: &str,
        answer: &str,
        related_nodes: &[NodeId],
        registry: &mut KvNodeRegistry,
        client: &reqwest::blocking::Client,
        server: &str,
        kv_capacity: i32,
    ) -> Result<NodeId> {
        let turn = self.next_turn;
        self.next_turn += 1;
        let node_id = NodeId(CONV_NODE_BASE + turn as u64);

        // Extract key terms from the question
        let terms: Vec<String> = question.to_lowercase()
            .split(|c: char| !c.is_alphanumeric() && c != '-' && c != '_')
            .filter(|s| s.len() > 2)
            .map(|s| s.to_string())
            .collect();

        // Build a concise fragment — question + first meaningful lines of answer
        let answer_summary = Self::summarize_answer(answer, 200);
        let kv_text = format!("[Conv Q{}] {}\n→ {}\n", turn + 1, question, answer_summary);

        // Estimate tokens and ensure capacity
        let est_tokens = (kv_text.len() as f64 / 3.5) as i32 + 5;
        let protected: HashSet<NodeId> = related_nodes.iter().copied()
            .chain(self.fragments.iter().map(|f| f.node_id)) // protect existing conv fragments
            .collect();
        registry.ensure_capacity(est_tokens, kv_capacity, &protected);

        // Tokenize and register in the KV cache
        let n_tokens = tokenize(client, server, &kv_text)?;
        registry.register(node_id, &kv_text, n_tokens);

        let fragment = ConvFragment {
            node_id,
            question: question.to_string(),
            terms,
            kv_text,
            related_graph_nodes: related_nodes.to_vec(),
            turn,
        };

        eprintln!("  [Conv] Registered turn {} as KV node {} ({} tokens)",
            turn + 1, node_id.0, n_tokens);

        self.fragments.push(fragment);
        Ok(node_id)
    }

    /// Extract the first meaningful lines of the answer as a summary.
    fn summarize_answer(answer: &str, max_chars: usize) -> String {
        let mut summary = String::new();
        for line in answer.lines() {
            let trimmed = line.trim();
            // Skip empty lines and markdown headers with no content
            if trimmed.is_empty() { continue; }
            // Skip decorative markdown
            if trimmed.starts_with("---") || trimmed.starts_with("===") { continue; }
            // Skip code blocks
            if trimmed.starts_with("```") { continue; }

            if !summary.is_empty() { summary.push(' '); }
            summary.push_str(trimmed);

            if summary.len() >= max_chars { break; }
        }
        // Truncate safely at char boundary
        if summary.len() > max_chars {
            summary = summary.chars().take(max_chars).collect();
            summary.push_str("…");
        }
        summary
    }

    /// Find conversation fragments relevant to a query.
    /// Returns (fragment_node_ids, fragment_adjacency_to_graph_nodes).
    fn find_relevant(&self, query: &str, registry: &mut KvNodeRegistry) -> (Vec<NodeId>, HashMap<NodeId, HashSet<NodeId>>) {
        let query_lower = query.to_lowercase();
        let query_terms: HashSet<String> = query_lower
            .split(|c: char| !c.is_alphanumeric() && c != '-' && c != '_')
            .filter(|s| s.len() > 2)
            .map(|s| s.to_string())
            .collect();

        let stop_words: HashSet<&str> = [
            "les", "des", "est", "sont", "que", "qui", "quels", "quel", "quelle",
            "pour", "dans", "avec", "sur", "par", "une", "the", "and", "what",
            "which", "about", "from", "with", "donne", "détails", "details",
            "plus", "encore", "more", "tell", "show", "comment", "quoi",
        ].into_iter().collect();

        let meaningful_terms: HashSet<&String> = query_terms.iter()
            .filter(|t| !stop_words.contains(t.as_str()))
            .collect();

        let mut relevant_ids = Vec::new();
        let mut conv_adjacency: HashMap<NodeId, HashSet<NodeId>> = HashMap::new();

        for frag in &self.fragments {
            // Check if fragment is still in the KV (might have been evicted)
            if registry.get_slot(frag.node_id).is_none() { continue; }

            // Score by term overlap
            let frag_terms: HashSet<&String> = frag.terms.iter()
                .filter(|t| !stop_words.contains(t.as_str()))
                .collect();

            let overlap = meaningful_terms.intersection(&frag_terms).count();

            // Also check if any query term appears in the fragment's kv_text
            let text_lower = frag.kv_text.to_lowercase();
            let text_matches = meaningful_terms.iter()
                .filter(|t| text_lower.contains(t.as_str()))
                .count();

            let score = overlap * 2 + text_matches;

            if score > 0 {
                relevant_ids.push(frag.node_id);
                registry.touch(&[frag.node_id]);

                // Conv fragment "sees" its related graph nodes (bidirectional attention)
                let related: HashSet<NodeId> = frag.related_graph_nodes.iter().copied().collect();
                conv_adjacency.insert(frag.node_id, related.clone());
                // Graph nodes also see this conv fragment
                for &gn in &frag.related_graph_nodes {
                    conv_adjacency.entry(gn).or_default().insert(frag.node_id);
                }
            }
        }

        // Also include the LAST fragment (most recent turn) even if no term overlap,
        // since it provides immediate context continuity.
        if let Some(last) = self.fragments.last() {
            if !relevant_ids.contains(&last.node_id) {
                if registry.get_slot(last.node_id).is_some() {
                    relevant_ids.push(last.node_id);
                    registry.touch(&[last.node_id]);
                    let related: HashSet<NodeId> = last.related_graph_nodes.iter().copied().collect();
                    conv_adjacency.insert(last.node_id, related.clone());
                    for &gn in &last.related_graph_nodes {
                        conv_adjacency.entry(gn).or_default().insert(last.node_id);
                    }
                }
            }
        }

        if !relevant_ids.is_empty() {
            eprintln!("  [Conv] {} relevant fragments (of {})",
                relevant_ids.len(), self.fragments.len());
        }

        (relevant_ids, conv_adjacency)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Conversation DB — stores Q&A history in obrain graph
// ═══════════════════════════════════════════════════════════════════════════════

struct ConversationDB {
    db: ObrainDB,
    current_conv_id: NodeId,
}

impl ConversationDB {
    /// Open or create conversation DB at given path.
    fn open(path: &str) -> Result<Self> {
        let db = ObrainDB::open(path)
            .context(format!("Failed to open conversation DB at {path}"))?;

        // Find or create the current conversation
        let conv_id = {
            let store = db.store();
            let convs = store.nodes_by_label("Conversation");
            // Use the most recent one, or create a new one
            let latest = convs.iter().filter_map(|&nid| {
                let node = store.get_node(nid)?;
                let ts = node.properties.get(&PropertyKey::from("created_at"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("1970-01-01");
                Some((nid, ts.to_string()))
            })
            .max_by(|a, b| a.1.cmp(&b.1));

            match latest {
                Some((nid, _)) => nid,
                None => Self::create_conv_node(&db, "Conversation 1"),
            }
        };

        Ok(Self { db, current_conv_id: conv_id })
    }

    fn create_conv_node(db: &ObrainDB, title: &str) -> NodeId {
        db.create_node_with_props(&["Conversation"], [
            ("title", Value::String(title.to_string().into())),
            ("created_at", Value::String(Utc::now().to_rfc3339().into())),
        ])
    }

    /// Start a new conversation within the same DB.
    fn new_conversation(&mut self, title: &str) {
        self.current_conv_id = Self::create_conv_node(&self.db, title);
    }

    /// Add a message to the current conversation. Returns the message NodeId.
    fn add_message(&self, role: &str, content: &str) -> NodeId {
        let store = self.db.store();
        let msg_count = store.edges_from(self.current_conv_id, Direction::Outgoing)
            .count();

        let msg_id = self.db.create_node_with_props(&["Message"], [
            ("role", Value::String(role.to_string().into())),
            ("content", Value::String(content.to_string().into())),
            ("timestamp", Value::String(Utc::now().to_rfc3339().into())),
            ("order", Value::Int64(msg_count as i64)),
        ]);
        self.db.create_edge(self.current_conv_id, msg_id, "HAS_MSG");
        msg_id
    }

    /// Link a reply message to the message it's replying to.
    fn link_reply(&self, reply_id: NodeId, parent_id: NodeId) {
        self.db.create_edge(reply_id, parent_id, "REPLIES_TO");
    }

    /// Get recent messages from the current conversation (for context injection).
    fn recent_messages(&self, limit: usize) -> Vec<(String, String)> {
        let store = self.db.store();
        let mut msgs: Vec<(i64, String, String)> = Vec::new();

        for (target, _eid) in store.edges_from(self.current_conv_id, Direction::Outgoing).collect::<Vec<_>>() {
            if let Some(node) = store.get_node(target) {
                let has_msg_label = node.labels.iter().any(|l| {
                    let s: &str = l.as_ref();
                    s == "Message"
                });
                if !has_msg_label { continue; }

                let role = node.properties.get(&PropertyKey::from("role"))
                    .and_then(|v| v.as_str()).unwrap_or("user").to_string();
                let content = node.properties.get(&PropertyKey::from("content"))
                    .and_then(|v| v.as_str()).unwrap_or("").to_string();
                let order = node.properties.get(&PropertyKey::from("order"))
                    .and_then(|v| if let Value::Int64(n) = v { Some(*n) } else { None })
                    .unwrap_or(0);
                msgs.push((order, role, content));
            }
        }

        msgs.sort_by_key(|(order, _, _)| *order);
        msgs.into_iter()
            .rev().take(limit).collect::<Vec<_>>()
            .into_iter().rev()
            .map(|(_, role, content)| (role, content))
            .collect()
    }

    /// List all conversations in the DB.
    fn list_conversations(&self) -> Vec<(NodeId, String, String, usize)> {
        let store = self.db.store();
        let convs = store.nodes_by_label("Conversation");
        let mut result = Vec::new();

        for &nid in &convs {
            if let Some(node) = store.get_node(nid) {
                let title = node.properties.get(&PropertyKey::from("title"))
                    .and_then(|v| v.as_str()).unwrap_or("(untitled)").to_string();
                let created = node.properties.get(&PropertyKey::from("created_at"))
                    .and_then(|v| v.as_str()).unwrap_or("?").to_string();
                let msg_count = store.edges_from(nid, Direction::Outgoing)
                    .count();
                result.push((nid, title, created, msg_count));
            }
        }

        result.sort_by(|a, b| b.2.cmp(&a.2)); // newest first
        result
    }

    /// Switch to an existing conversation by NodeId.
    fn switch_to(&mut self, conv_id: NodeId) -> bool {
        let store = self.db.store();
        if store.get_node(conv_id).is_some() {
            self.current_conv_id = conv_id;
            true
        } else {
            false
        }
    }

    /// Get current conversation title.
    fn current_title(&self) -> String {
        let store = self.db.store();
        store.get_node(self.current_conv_id)
            .and_then(|n| n.properties.get(&PropertyKey::from("title"))
                .and_then(|v| v.as_str()).map(|s| s.to_string()))
            .unwrap_or_else(|| "(untitled)".to_string())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════════

fn main() -> Result<()> {
    let server = std::env::args()
        .position(|a| a == "--server")
        .and_then(|i| std::env::args().nth(i + 1))
        .unwrap_or_else(|| DEFAULT_SERVER.to_string());

    let db_path = std::env::args()
        .position(|a| a == "--db")
        .and_then(|i| std::env::args().nth(i + 1))
        .unwrap_or_else(|| "/tmp/neo4j2grafeo/grafeo.db".to_string());

    let max_nodes: usize = std::env::args()
        .position(|a| a == "--max-nodes")
        .and_then(|i| std::env::args().nth(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(40);

    let token_budget: i32 = std::env::args()
        .position(|a| a == "--budget")
        .and_then(|i| std::env::args().nth(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(1400);

    let kv_capacity: i32 = std::env::args()
        .position(|a| a == "--kv-capacity")
        .and_then(|i| std::env::args().nth(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(4096); // max tokens in KV node pool

    let conv_path: Option<PathBuf> = std::env::args()
        .position(|a| a == "--conv")
        .and_then(|i| std::env::args().nth(i + 1))
        .map(PathBuf::from);

    eprintln!("=== obrain-chat — Generic Graph LLM + Topological Mask ===\n");

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()?;
    check_server(&client, &server)?;

    // ── Open database ────────────────────────────────────────────────
    eprintln!("Opening database: {db_path}");
    let ckpt = format!("{db_path}/wal/checkpoint.meta");
    let _ = std::fs::remove_file(&ckpt);

    let db = ObrainDB::open(&db_path)
        .context(format!("Failed to open DB at {db_path}"))?;
    let store = Arc::clone(db.store());
    eprintln!("  {} nodes, {} edges\n", store.node_count(), store.edge_count());

    // ── Schema introspection ─────────────────────────────────────────
    eprintln!("Discovering schema...");
    let schema = discover_schema(&store);

    eprintln!("\n{}", "=".repeat(60));
    eprintln!("Ready. {} structural labels, {} hierarchy rules.",
        schema.structural_labels.len(),
        schema.parent_child.len());
    // ── Discover or load KV Banks ──────────────────────────────────
    let bank_cache_path = std::path::PathBuf::from(format!("{db_path}.banks"));
    let nc = store.node_count();
    let ec = store.edge_count();
    let banks = match load_bank_cache(&bank_cache_path, nc, ec) {
        Some(cached) => {
            eprintln!("Loaded {} banks from cache ({})", cached.len(), bank_cache_path.display());
            cached
        }
        None => {
            eprintln!("Discovering banks...");
            let banks = discover_banks(&store, &schema, 50);
            save_bank_cache(&bank_cache_path, &banks, nc, ec);
            eprintln!("  Saved bank cache to {}", bank_cache_path.display());
            banks
        }
    };

    // ── Conversation DB ────────────────────────────────────────────
    let mut conv_db = if let Some(ref cp) = conv_path {
        let cp_str = cp.to_str().unwrap_or("conv.db");
        // Remove stale checkpoint if exists (same as main DB)
        let ckpt_conv = format!("{cp_str}/wal/checkpoint.meta");
        let _ = std::fs::remove_file(&ckpt_conv);
        match ConversationDB::open(cp_str) {
            Ok(cdb) => {
                let convs = cdb.list_conversations();
                eprintln!("Conversation DB: {} conversations (current: \"{}\")",
                    convs.len(), cdb.current_title());
                // Show recent context
                let recent = cdb.recent_messages(4);
                if !recent.is_empty() {
                    eprintln!("  Recent context ({} messages):", recent.len());
                    for (role, content) in &recent {
                        let snippet: String = content.chars().take(80).collect();
                        eprintln!("    {}: {}{}",
                            role, snippet,
                            if content.len() > 80 { "..." } else { "" });
                    }
                }
                Some(cdb)
            }
            Err(e) => {
                eprintln!("  Warning: could not open conversation DB: {e}");
                None
            }
        }
    } else {
        None
    };

    eprintln!("\nCommands: /quit, /schema, /kv, /banks, /history, /conversations, /new <title>\n");

    // ── Initialize KV Node Registry ─────────────────────────────
    let header_tokens = tokenize(&client, &server, SYSTEM_HEADER)?;
    let mut registry = KvNodeRegistry::new(SYSTEM_HEADER, header_tokens);
    let mut conv_frags = ConvFragments::new();
    eprintln!("KV Registry initialized: header={} tokens", header_tokens);

    // ── Warmup: pre-load top banks into KV ──────────────────────
    let warmup_count = 3.min(banks.len());
    if warmup_count > 0 {
        let warmup_t0 = Instant::now();
        let mut warmup_loaded = 0;
        for bank in banks.iter().take(warmup_count) {
            let protected: HashSet<NodeId> = bank.node_ids.iter().copied().collect();
            registry.ensure_capacity(bank.est_tokens, kv_capacity, &protected);
            for nid in &bank.node_ids {
                if registry.get_slot(*nid).is_some() { continue; }
                if let Some(text) = bank.texts.get(nid) {
                    let n_tok = tokenize(&client, &server, text)?;
                    registry.register(*nid, text, n_tok);
                    warmup_loaded += 1;
                }
            }
        }
        eprintln!("  Warmup: pre-loaded {} banks ({} nodes) in {:.0}ms",
            warmup_count, warmup_loaded, warmup_t0.elapsed().as_millis());
    }

    // ── Interactive loop ─────────────────────────────────────────────
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();

    loop {
        print!("you> ");
        io::stdout().flush()?;

        let line = match lines.next() {
            Some(Ok(l)) => l,
            _ => break,
        };
        let line = line.trim().to_string();
        if line.is_empty() { continue; }
        if line == "/quit" || line == "/exit" || line == "quit" { break; }
        if line == "/schema" {
            eprintln!("  Structural: {:?}", schema.structural_labels);
            eprintln!("  Noise: {:?}", schema.noise_labels);
            for (parent, children) in &schema.parent_child {
                eprintln!("  {} → {:?}", parent, children);
            }
            continue;
        }
        if line == "/banks" {
            for (i, bank) in banks.iter().enumerate() {
                let loaded = bank.node_ids.iter()
                    .filter(|nid| registry.get_slot(**nid).is_some())
                    .count();
                eprintln!("  [{}] {} ({} nodes, ~{} tok, {}/{} in KV)",
                    i, bank.name, bank.node_ids.len(), bank.est_tokens,
                    loaded, bank.node_ids.len());
            }
            continue;
        }
        if line == "/history" {
            if let Some(ref cdb) = conv_db {
                let msgs = cdb.recent_messages(20);
                if msgs.is_empty() {
                    eprintln!("  (no messages in current conversation)");
                } else {
                    eprintln!("  History ({} messages):", msgs.len());
                    for (role, content) in &msgs {
                        let snippet: String = content.chars().take(120).collect();
                        eprintln!("  {}: {}{}", role, snippet,
                            if content.len() > 120 { "..." } else { "" });
                    }
                }
            } else {
                eprintln!("  No conversation DB (use --conv <path>)");
            }
            continue;
        }
        if line == "/conversations" || line == "/convs" {
            if let Some(ref cdb) = conv_db {
                let convs = cdb.list_conversations();
                if convs.is_empty() {
                    eprintln!("  (no conversations)");
                } else {
                    for (nid, title, created, msg_count) in &convs {
                        let marker = if *nid == cdb.current_conv_id { " ←" } else { "" };
                        eprintln!("  [{}] {} ({} msgs, {}){}", nid.0, title, msg_count, &created[..10.min(created.len())], marker);
                    }
                }
            } else {
                eprintln!("  No conversation DB (use --conv <path>)");
            }
            continue;
        }
        if line.starts_with("/new ") {
            if let Some(ref mut cdb) = conv_db {
                let title = line.strip_prefix("/new ").unwrap().trim();
                cdb.new_conversation(title);
                eprintln!("  Created new conversation: \"{}\"", title);
            } else {
                eprintln!("  No conversation DB (use --conv <path>)");
            }
            continue;
        }
        if line.starts_with("/switch ") {
            if let Some(ref mut cdb) = conv_db {
                let id_str = line.strip_prefix("/switch ").unwrap().trim();
                if let Ok(id_num) = id_str.parse::<u64>() {
                    if cdb.switch_to(NodeId(id_num)) {
                        eprintln!("  Switched to conversation: \"{}\"", cdb.current_title());
                    } else {
                        eprintln!("  Conversation {} not found", id_num);
                    }
                }
            } else {
                eprintln!("  No conversation DB (use --conv <path>)");
            }
            continue;
        }
        if line == "/kv" {
            registry.log_metrics();
            eprintln!("  Loaded nodes ({}):", registry.order.len());
            for nid in registry.order.iter().take(20) {
                // Get label for display props lookup
                let label: Option<String> = store.get_node(*nid)
                    .and_then(|n| n.labels.first().map(|l| {
                        let s: &str = l.as_ref();
                        s.to_string()
                    }));
                let dp = label.as_deref()
                    .and_then(|l| schema.display_props.get(l));
                let name = get_node_name_generic(&store, *nid, dp);
                let label_str = label.as_deref().unwrap_or("?");
                let slot = registry.get_slot(*nid);
                if let Some(s) = slot {
                    eprintln!("    {} [{}-{}] last_q={} [{}] {}",
                        nid.0, s.start, s.end, s.last_used, label_str,
                        if name.is_empty() { "(unnamed)" } else { &name });
                }
            }
            if registry.order.len() > 20 {
                eprintln!("    ... and {} more", registry.order.len() - 20);
            }
            continue;
        }

        // Store user message in conversation DB
        let user_msg_id = conv_db.as_ref().map(|cdb| cdb.add_message("user", &line));

        match query_with_registry(&client, &server, &store, &schema, &mut registry, &mut conv_frags, &banks, &line, max_nodes, token_budget, kv_capacity) {
            Ok((response, relevant_graph_nodes)) => {
                // Response already streamed to stdout by complete_streaming/chat_complete
                let clean = strip_think_tags(&response);
                let trimmed = clean.trim().to_string();

                // Register Q&A as a concise fragment in the KV cache
                if let Err(e) = conv_frags.add_turn(
                    &line, &trimmed, &relevant_graph_nodes,
                    &mut registry, &client, &server, kv_capacity,
                ) {
                    eprintln!("  Warning: could not register conv fragment: {e}");
                }

                // Store full messages in conversation DB (for persistence across sessions)
                if let Some(ref cdb) = conv_db {
                    let asst_id = cdb.add_message("assistant", &trimmed);
                    if let Some(uid) = user_msg_id {
                        cdb.link_reply(asst_id, uid);
                    }
                }
            },
            Err(e) => eprintln!("  Error: {e}\n"),
        }
    }

    eprintln!("Bye!");
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════════
// Registry-aware retrieval pipeline
// ═══════════════════════════════════════════════════════════════════════════════

fn query_with_registry(
    client: &reqwest::blocking::Client,
    server: &str,
    store: &Arc<LpgStore>,
    schema: &GraphSchema,
    registry: &mut KvNodeRegistry,
    conv_frags: &mut ConvFragments,
    banks: &[KvBank],
    query: &str,
    max_nodes: usize,
    token_budget: i32,
    kv_capacity: i32,
) -> Result<(String, Vec<NodeId>)> {
    registry.begin_query();
    let t_start = Instant::now();

    // ── Check if query matches a bank name (load entire bank) ────
    let query_lower = query.to_lowercase();
    let mut bank_loaded = false;
    for bank in banks {
        let bank_name_lower = bank.name.to_lowercase();
        // Check if any content term matches the bank name
        let terms: Vec<&str> = query_lower
            .split(|c: char| !c.is_alphanumeric() && c != '-' && c != '_')
            .filter(|s| s.len() > 2)
            .collect();
        let matches_bank = terms.iter().any(|term| {
            bank_name_lower.contains(term) && *term != "project" && *term != "les" && *term != "des"
        });
        if matches_bank {
            let missing: Vec<NodeId> = bank.node_ids.iter()
                .filter(|nid| !registry.nodes.contains_key(nid))
                .copied()
                .collect();
            if !missing.is_empty() {
                let protected: HashSet<NodeId> = bank.node_ids.iter().copied().collect();
                registry.ensure_capacity(bank.est_tokens, kv_capacity, &protected);

                eprintln!("  Loading bank '{}': {} nodes ({} new)",
                    bank.name, bank.node_ids.len(), missing.len());
                for nid in &missing {
                    if let Some(text) = bank.texts.get(nid) {
                        let n_tok = tokenize(client, server, text)?;
                        registry.register(*nid, text, n_tok);
                    }
                }
                bank_loaded = true;
            } else {
                eprintln!("  Bank '{}' already fully loaded in KV", bank.name);
                registry.touch(&bank.node_ids);
                bank_loaded = true;
            }
        }
    }
    if bank_loaded {
        eprintln!("  Bank loading done in {}ms", t_start.elapsed().as_millis());
    }

    // Run the generic retrieval to get scored nodes
    let (scored_nodes_for_query, adjacency, node_texts) =
        retrieve_nodes(client, server, store, schema, query, max_nodes, token_budget)?;

    if scored_nodes_for_query.is_empty() {
        let resp = chat_complete(client, server,
            "No relevant data found in the knowledge graph for this query.",
            query, 256)?;
        return Ok((resp, Vec::new()));
    }

    // Identify which nodes need to be loaded into KV
    let needed_ids: Vec<NodeId> = scored_nodes_for_query.iter().map(|n| n.id).collect();
    let missing = registry.find_missing(&needed_ids);
    registry.touch(&needed_ids);

    // Estimate tokens needed for missing nodes
    let est_missing_tokens: i32 = missing.iter()
        .filter_map(|nid| node_texts.get(nid))
        .map(|text| (text.len() as f64 / 3.5) as i32 + 10)
        .sum();

    // Evict LRU if needed (protect nodes used in current query)
    let protected: HashSet<NodeId> = needed_ids.iter().copied().collect();
    registry.ensure_capacity(est_missing_tokens, kv_capacity, &protected);

    eprintln!("  KV: {} needed, {} cached, {} to encode (~{} tokens)",
        needed_ids.len(), needed_ids.len() - missing.len(), missing.len(), est_missing_tokens);

    // Encode missing nodes — append their text to the KV
    for &nid in &missing {
        if let Some(text) = node_texts.get(&nid) {
            let n_tok = tokenize(client, server, text)?;
            registry.register(nid, text, n_tok);
        }
    }

    let encode_time = t_start.elapsed().as_millis();
    eprintln!("  KV encode: {}ms", encode_time);

    // ── Find relevant conversation fragments ──────────────────────
    let (conv_node_ids, conv_adjacency) = conv_frags.find_relevant(query, registry);

    // Pull in graph nodes referenced by relevant conv fragments
    // (enables "donne plus de details" to re-surface Elun nodes from the last turn)
    const MAX_CONV_PULL: usize = 10;
    let mut conv_pulled_ids: Vec<NodeId> = Vec::new();
    for frag in &conv_frags.fragments {
        if conv_node_ids.contains(&frag.node_id) {
            let mut pulled_from_frag = 0;
            for &gn in &frag.related_graph_nodes {
                if pulled_from_frag >= MAX_CONV_PULL { break; }
                let already_in_query = scored_nodes_for_query.iter().any(|n| n.id == gn);
                if !already_in_query && registry.get_slot(gn).is_some() {
                    conv_pulled_ids.push(gn);
                    pulled_from_frag += 1;
                }
            }
        }
    }
    if !conv_pulled_ids.is_empty() {
        eprintln!("  [Conv] Pulled {} graph nodes from fragment references", conv_pulled_ids.len());
        registry.touch(&conv_pulled_ids);
    }

    // If the query is vague (all scores equal = no strong match) and conv fragments
    // provide context, prefer fragment-referenced nodes over generic retrieval.
    let vague_query = {
        let scores: HashSet<u64> = scored_nodes_for_query.iter()
            .map(|n| (n._score * 100.0) as u64)
            .collect();
        scores.len() <= 2 // all nodes have same score = nothing specific matched
    };
    let conv_provides_context = !conv_pulled_ids.is_empty();

    // Build QueryContext from registry positions (not from re-tokenization)
    let mut ctx_nodes: Vec<ContextNode> = Vec::new();
    if vague_query && conv_provides_context {
        // Vague query + conv context: only include graph nodes that are also
        // referenced by fragments (or already in fragment pull list)
        let pulled_set: HashSet<NodeId> = conv_pulled_ids.iter().copied().collect();
        eprintln!("  [Conv] Vague query detected — using {} fragment-referenced nodes instead of {} generic",
            conv_pulled_ids.len(), scored_nodes_for_query.len());
        for cn in &scored_nodes_for_query {
            if pulled_set.contains(&cn.id) {
                if let Some(slot) = registry.get_slot(cn.id) {
                    ctx_nodes.push(ContextNode {
                        id: cn.id,
                        token_start: slot.start,
                        token_end: slot.end,
                    });
                }
            }
        }
    } else {
        for cn in &scored_nodes_for_query {
            if let Some(slot) = registry.get_slot(cn.id) {
                ctx_nodes.push(ContextNode {
                    id: cn.id,
                    token_start: slot.start,
                    token_end: slot.end,
                });
            }
        }
    }

    // Add graph nodes pulled from conv fragment references
    for &gn in &conv_pulled_ids {
        if let Some(slot) = registry.get_slot(gn) {
            // Avoid duplicates
            if !ctx_nodes.iter().any(|n| n.id == gn) {
                ctx_nodes.push(ContextNode {
                    id: gn,
                    token_start: slot.start,
                    token_end: slot.end,
                });
            }
        }
    }

    // Add conversation fragment nodes to context
    for &conv_nid in &conv_node_ids {
        if let Some(slot) = registry.get_slot(conv_nid) {
            ctx_nodes.push(ContextNode {
                id: conv_nid,
                token_start: slot.start,
                token_end: slot.end,
            });
        }
    }

    // Merge adjacency: graph adjacency + conv fragment adjacency
    let mut merged_adjacency = adjacency;
    for (nid, neighbors) in conv_adjacency {
        merged_adjacency.entry(nid).or_default().extend(neighbors);
    }
    // Conv-pulled nodes should see each other (they were part of the same context)
    if conv_pulled_ids.len() > 1 {
        for i in 0..conv_pulled_ids.len() {
            for j in 0..conv_pulled_ids.len() {
                if i != j {
                    merged_adjacency.entry(conv_pulled_ids[i]).or_default().insert(conv_pulled_ids[j]);
                }
            }
        }
    }

    let ctx = QueryContext {
        total_tokens: registry.next_pos,
        header_tokens: registry.header_end,
        prompt: registry.reconstruct_prompt(),
        nodes: ctx_nodes,
        adjacency: merged_adjacency,
        edge_count: scored_nodes_for_query.len(),
    };

    registry.log_metrics();

    // Collect relevant graph node IDs for the caller (to link conv fragment)
    let relevant_graph_nodes: Vec<NodeId> = scored_nodes_for_query.iter()
        .map(|n| n.id)
        .collect();

    let response = generate_with_mask(client, server, &ctx, query)?;
    Ok((response, relevant_graph_nodes))
}

/// A scored node selected for the current query.
struct ScoredContextNode {
    id: NodeId,
    _score: f64,
}

/// Retrieve and score nodes from the graph. Returns:
/// - selected nodes with scores
/// - adjacency map for the mask
/// - node_id → text for KV encoding
fn retrieve_nodes(
    _client: &reqwest::blocking::Client,
    _server: &str,
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

    eprintln!("  Terms: {:?}", terms);
    eprintln!("  Matched labels: {:?}", target_labels);
    eprintln!("  Content terms: {:?}", content_terms);

    // ── Step 2: Fetch + score nodes by target labels ─────────────────
    let mut scored_nodes: Vec<(NodeId, f64)> = Vec::new();

    for (label, label_sim) in &target_labels {
        let node_ids = store.nodes_by_label(label);
        let info = schema.labels.iter().find(|l| l.label == *label);
        let importance = info.map_or(1.0, |i| i.importance);

        eprintln!("    {} {} nodes (imp={:.2})", node_ids.len(), label, importance);

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

    eprintln!("  Scored {} seed nodes", scored_nodes.len());

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

    eprintln!("  BFS expanded to {} nodes", bfs_result.len());

    // ── Step 4: Score, select, build prompt ──────────────────────────
    let mut scored: Vec<(NodeId, f64, String)> = bfs_result.iter()
        .filter_map(|(nid, _depth, retrieval_score)| {
            let (text, _labels, _summary) = extract_node_generic(store, *nid, schema);
            if text.is_empty() { return None; }
            Some((*nid, *retrieval_score, text))
        })
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    for (i, (_nid, score, text)) in scored.iter().take(10).enumerate() {
        eprintln!("    [{i}] score={:.2} {}", score, truncate(text, 80));
    }

    // ── Step 5: Build text blocks with inline children ─────────────
    let mut shown_ids: HashSet<NodeId> = HashSet::new();
    let mut selected: Vec<ScoredContextNode> = Vec::new();
    let mut node_texts: HashMap<NodeId, String> = HashMap::new();
    let mut est_tokens: i32 = 0; // budget tracking (header handled by registry)

    for (nid, score, base_text) in &scored {
        if selected.len() >= max_nodes { break; }
        if shown_ids.contains(nid) { continue; }

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
                                if children.len() < max_children {
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
                                    shown_ids.insert(target);
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

        shown_ids.insert(*nid);
        node_texts.insert(*nid, text_block);
        selected.push(ScoredContextNode { id: *nid, _score: *score });
        est_tokens += est;
    }

    eprintln!("  Selected {} nodes for query", selected.len());

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

// ═══════════════════════════════════════════════════════════════════════════════
// Attention mask + generation
// ═══════════════════════════════════════════════════════════════════════════════

fn generate_with_mask(
    client: &reqwest::blocking::Client,
    server: &str,
    ctx: &QueryContext,
    query: &str,
) -> Result<String> {
    // Conversation context is now in the KV cache as fragment nodes — no injection needed
    let query_text = format!(
        "<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
    );
    let query_tokens = tokenize(client, server, &query_text)?;

    let n_predict: i32 = 1024;
    let n_gen_mask: i32 = 32;

    // ── Build sparse position map ──────────────────────────────────
    // Only include positions that matter: header + relevant nodes + query + gen.
    // Positions from non-relevant nodes in the KV are excluded (Rule 3b).
    let mut positions: Vec<i32> = Vec::new();
    let mut pos_remap: HashMap<i32, usize> = HashMap::new(); // real_pos → mask_index

    // Header positions
    for p in 0..ctx.header_tokens {
        let idx = positions.len();
        pos_remap.insert(p, idx);
        positions.push(p);
    }

    // Relevant node positions (from KV registry slots)
    for node in &ctx.nodes {
        for p in node.token_start..node.token_end {
            if !pos_remap.contains_key(&p) {
                let idx = positions.len();
                pos_remap.insert(p, idx);
                positions.push(p);
            }
        }
    }

    // Query + gen positions (appended AFTER all KV content)
    let q_start_real = ctx.total_tokens; // real position after all KV
    for offset in 0..(query_tokens + n_gen_mask) {
        let p = q_start_real + offset;
        let idx = positions.len();
        pos_remap.insert(p, idx);
        positions.push(p);
    }

    let sz = positions.len();
    let max_mask_positions: i32 = 2000;

    eprintln!("  Mask: {} sparse positions (of {} total KV), query={}, gen={}, {:.1}MB",
        sz, ctx.total_tokens, query_tokens, n_gen_mask,
        (sz * sz * 8) as f64 / 1_000_000.0);

    if sz as i32 > max_mask_positions {
        eprintln!("  ⚠ mask_size={} > {} — chat API fallback.", sz, max_mask_positions);
        return chat_complete(client, server, &ctx.prompt, query, n_predict);
    }

    // ── Compile attention mask (sparse) ──────────────────────────────
    let mut mask = vec![-1e30f32; sz * sz];

    let allow = |m: &mut [f32], real_i: i32, real_j: i32, remap: &HashMap<i32, usize>, sz: usize| {
        if let (Some(&mi), Some(&mj)) = (remap.get(&real_i), remap.get(&real_j)) {
            if mi < sz && mj < sz && real_j <= real_i {
                m[mi * sz + mj] = 0.0;
            }
        }
    };

    let h = ctx.header_tokens;

    // Rule 0: System header — causal
    for i in 0..h {
        for j in 0..=i {
            allow(&mut mask, i, j, &pos_remap, sz);
        }
    }

    // Rule 1: Each node sees header + causal self
    for node in &ctx.nodes {
        for i in node.token_start..node.token_end {
            for j in 0..h { allow(&mut mask, i, j, &pos_remap, sz); }
            for j in node.token_start..=i { allow(&mut mask, i, j, &pos_remap, sz); }
        }
    }

    // Rule 2: Connected nodes attend to each other
    for ni in &ctx.nodes {
        if let Some(visible) = ctx.adjacency.get(&ni.id) {
            for nj in &ctx.nodes {
                if nj.id == ni.id { continue; }
                if visible.contains(&nj.id) {
                    for i in ni.token_start..ni.token_end {
                        for j in nj.token_start..nj.token_end {
                            allow(&mut mask, i, j, &pos_remap, sz);
                        }
                    }
                }
            }
        }
    }

    // Rule 3: Query + gen tokens see header + relevant nodes + causal self
    // Rule 3b (implicit): non-relevant KV positions are NOT in pos_remap → invisible
    for offset in 0..(query_tokens + n_gen_mask) {
        let i = q_start_real + offset;
        // See header
        for j in 0..h { allow(&mut mask, i, j, &pos_remap, sz); }
        // See relevant nodes
        for node in &ctx.nodes {
            for j in node.token_start..node.token_end {
                allow(&mut mask, i, j, &pos_remap, sz);
            }
        }
        // Causal within query+gen
        for prev in 0..=offset {
            allow(&mut mask, i, q_start_real + prev, &pos_remap, sz);
        }
    }

    let connected = ctx.nodes.iter()
        .filter(|n| ctx.adjacency.get(&n.id).map_or(0, |s| s.len()) > 1)
        .count();
    eprintln!("  Topology: {} connected, {} isolated",
        connected, ctx.nodes.len() - connected);

    set_attn_mask(client, server, &mask, &positions)?;

    let full_prompt = format!("{}{}", ctx.prompt, query_text);
    let response = complete_streaming(client, server, &full_prompt, n_predict)?;

    clear_attn_mask(client, server)?;

    Ok(response)
}

// ═══════════════════════════════════════════════════════════════════════════════
// HTTP helpers
// ═══════════════════════════════════════════════════════════════════════════════

fn check_server(client: &reqwest::blocking::Client, server: &str) -> Result<()> {
    let resp = client.get(format!("{server}/health")).send()
        .context(format!("Cannot reach {server}"))?;
    if !resp.status().is_success() {
        bail!("Server unhealthy: {}", resp.status());
    }
    eprintln!("Server: OK ({server})");
    Ok(())
}

fn tokenize(client: &reqwest::blocking::Client, server: &str, text: &str) -> Result<i32> {
    let resp = client
        .post(format!("{server}/tokenize"))
        .json(&json!({ "content": text, "special": true }))
        .send()?;
    let body: serde_json::Value = resp.json()?;
    Ok(body["tokens"].as_array().context("no tokens")?.len() as i32)
}

fn set_attn_mask(client: &reqwest::blocking::Client, server: &str, mask: &[f32], positions: &[i32]) -> Result<()> {
    let n_pos = positions.len();
    eprintln!("  Sending mask: {} positions, {:.1}MB payload",
        n_pos, (mask.len() * 8) as f64 / 1_000_000.0);
    let resp = client
        .post(format!("{server}/attn-mask"))
        .json(&json!({ "mask": mask, "positions": positions }))
        .send()
        .context("Failed to send attention mask")?;
    let body: serde_json::Value = resp.json()
        .context("Failed to parse mask response")?;
    if body["success"].as_bool() != Some(true) {
        bail!("set mask failed: {body}");
    }
    Ok(())
}

fn clear_attn_mask(client: &reqwest::blocking::Client, server: &str) -> Result<()> {
    client.post(format!("{server}/attn-mask"))
        .json(&json!({ "mask": null }))
        .send()?;
    Ok(())
}

/// State machine for filtering <think>...</think> blocks during streaming.
/// Handles 3 cases:
///   1. <think>...</think> — standard think tags
///   2. No opening <think>, but </think> appears — implicit thinking (buffer until </think>)
///   3. No think tags at all — passthrough
///
/// Strategy: buffer initial tokens. If </think> appears, discard everything before it.
/// If we accumulate enough tokens without seeing </think>, flush as real content.
struct ThinkFilter {
    state: ThinkState,
    buffer: String,
    printed_any: bool,
}

enum ThinkState {
    /// Buffering initial tokens, waiting to see if </think> or <think> appears.
    Probing,
    /// Inside a <think>...</think> block, discarding content.
    InThink,
    /// Passthrough mode — no think tags, emit everything.
    Passthrough,
}

const PROBE_LIMIT: usize = 500; // max tokens to buffer before deciding it's passthrough

impl ThinkFilter {
    fn new() -> Self {
        Self { state: ThinkState::Probing, buffer: String::new(), printed_any: false }
    }

    fn feed(&mut self, token: &str) -> String {
        self.buffer.push_str(token);
        let mut output = String::new();

        loop {
            match self.state {
                ThinkState::Probing => {
                    // Check for </think> (implicit end of thinking without <think>)
                    if let Some(end) = self.buffer.find("</think>") {
                        // Everything before </think> was thinking — discard it
                        self.buffer = self.buffer[end + 8..].to_string();
                        self.state = ThinkState::Passthrough;
                        continue;
                    }
                    // Check for explicit <think>
                    if let Some(start) = self.buffer.find("<think>") {
                        // Emit anything before <think> (unlikely at start)
                        output.push_str(&self.buffer[..start]);
                        self.buffer = self.buffer[start + 7..].to_string();
                        self.state = ThinkState::InThink;
                        continue;
                    }
                    // Check for partial tags — keep buffering
                    if self.buffer.contains("</thi") || self.buffer.contains("<thin") {
                        break; // wait for more tokens
                    }
                    // Count "tokens" (rough: split by whitespace)
                    let token_count = self.buffer.split_whitespace().count();
                    if token_count > PROBE_LIMIT {
                        // No think tags after 50 tokens — it's real content
                        self.state = ThinkState::Passthrough;
                        continue;
                    }
                    break;
                }
                ThinkState::InThink => {
                    if let Some(end) = self.buffer.find("</think>") {
                        self.buffer = self.buffer[end + 8..].to_string();
                        self.state = ThinkState::Passthrough;
                        continue;
                    }
                    // Partial </think> match — keep buffering
                    if self.buffer.ends_with("</") || self.buffer.contains("</thi") {
                        break;
                    }
                    // Discard accumulated thinking content (keep last 20 chars for partial match)
                    if self.buffer.len() > 100 {
                        let keep = self.buffer.len() - 20;
                        self.buffer = self.buffer[keep..].to_string();
                    }
                    break;
                }
                ThinkState::Passthrough => {
                    // Check for new <think> blocks mid-stream
                    if let Some(start) = self.buffer.find("<think>") {
                        output.push_str(&self.buffer[..start]);
                        self.buffer = self.buffer[start + 7..].to_string();
                        self.state = ThinkState::InThink;
                        continue;
                    }
                    // Hold partial <think match
                    if self.buffer.ends_with('<') || self.buffer.ends_with("<t")
                        || self.buffer.ends_with("<th") || self.buffer.ends_with("<thi")
                        || self.buffer.ends_with("<thin") || self.buffer.ends_with("<think") {
                        let hold_from = self.buffer.rfind('<').unwrap_or(self.buffer.len());
                        output.push_str(&self.buffer[..hold_from]);
                        self.buffer = self.buffer[hold_from..].to_string();
                        break;
                    }
                    output.push_str(&self.buffer);
                    self.buffer.clear();
                    break;
                }
            }
        }

        // Clean role artifacts from beginning of output
        if !self.printed_any && !output.is_empty() {
            let trimmed = output.trim_start();
            for prefix in &["system\n", "assistant\n", "user\n"] {
                if trimmed.starts_with(prefix) {
                    output = trimmed[prefix.len()..].to_string();
                    break;
                }
            }
            if !output.trim().is_empty() {
                self.printed_any = true;
            }
        }

        output
    }

    fn flush(&mut self) -> String {
        match self.state {
            ThinkState::InThink => String::new(), // discard unclosed think
            ThinkState::Probing => {
                // Never saw </think> or enough tokens — emit as-is
                let out = std::mem::take(&mut self.buffer);
                self.clean_role_prefix(out)
            }
            ThinkState::Passthrough => {
                let out = std::mem::take(&mut self.buffer);
                self.clean_role_prefix(out)
            }
        }
    }

    fn clean_role_prefix(&self, s: String) -> String {
        let trimmed = s.trim_start();
        for prefix in &["system\n", "assistant\n"] {
            if trimmed.starts_with(prefix) {
                return trimmed[prefix.len()..].to_string();
            }
        }
        s
    }
}

/// Streaming completion via llama.cpp SSE endpoint.
/// Prints tokens to stdout as they arrive, returns the full text.
fn complete_streaming(client: &reqwest::blocking::Client, server: &str, prompt: &str, n_predict: i32) -> Result<String> {
    let resp = client
        .post(format!("{server}/completion"))
        .json(&json!({
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": 0.1,
            "top_p": 0.9,
            "cache_prompt": true,
            "special": true,
            "stream": true,
            "stop": ["<|im_start|>", "<|im_end|>", "<|endoftext|>"],
        }))
        .send()
        .context("completion request failed")?;

    print!("assistant> ");
    io::stdout().flush()?;

    let mut full_text = String::new();
    let mut filter = ThinkFilter::new();
    let reader = BufReader::new(resp);

    for line in reader.lines() {
        let line = line?;
        if !line.starts_with("data: ") { continue; }
        let data = &line[6..];
        if data == "[DONE]" { break; }

        if let Ok(chunk) = serde_json::from_str::<serde_json::Value>(data) {
            if let Some(token) = chunk["content"].as_str() {
                let visible = filter.feed(token);
                if !visible.is_empty() {
                    print!("{}", visible);
                    io::stdout().flush()?;
                    full_text.push_str(&visible);
                }
            }
            if chunk["stop"].as_bool() == Some(true) { break; }
        }
    }
    // Flush remaining
    let remaining = filter.flush();
    if !remaining.is_empty() {
        print!("{}", remaining);
        full_text.push_str(&remaining);
    }
    println!("\n");

    Ok(full_text)
}

/// Non-streaming completion (kept for internal use like tokenization tests).
fn complete(client: &reqwest::blocking::Client, server: &str, prompt: &str, n_predict: i32) -> Result<String> {
    let resp = client
        .post(format!("{server}/completion"))
        .json(&json!({
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": 0.1,
            "top_p": 0.9,
            "cache_prompt": true,
            "special": true,
            "stop": ["<|im_start|>", "<|im_end|>", "<|endoftext|>"],
        }))
        .send()?;
    let body: serde_json::Value = resp.json()?;
    Ok(body["content"].as_str().unwrap_or("[no content]").to_string())
}

/// Streaming chat completion via /v1/chat/completions (SSE).
fn chat_complete(
    client: &reqwest::blocking::Client,
    server: &str,
    system: &str,
    user_msg: &str,
    n_predict: i32,
) -> Result<String> {
    let mut messages = Vec::new();
    if !system.is_empty() {
        messages.push(json!({"role": "system", "content": format!(
            "You have access to a knowledge graph. Answer based on the following data:\n\n{}\n\nAnswer concisely in the user's language. /no_think",
            system
        )}));
    }
    messages.push(json!({"role": "user", "content": user_msg}));
    let resp = client
        .post(format!("{server}/v1/chat/completions"))
        .json(&json!({
            "messages": messages,
            "max_tokens": n_predict,
            "temperature": 0.1,
            "top_p": 0.9,
            "stream": true,
        }))
        .send()?;

    print!("assistant> ");
    io::stdout().flush()?;

    let mut full_text = String::new();
    let mut filter = ThinkFilter::new();
    let reader = BufReader::new(resp);

    for line in reader.lines() {
        let line = line?;
        if !line.starts_with("data: ") { continue; }
        let data = &line[6..];
        if data == "[DONE]" { break; }

        if let Ok(chunk) = serde_json::from_str::<serde_json::Value>(data) {
            if let Some(delta) = chunk["choices"][0]["delta"]["content"].as_str() {
                let visible = filter.feed(delta);
                if !visible.is_empty() {
                    print!("{}", visible);
                    io::stdout().flush()?;
                    full_text.push_str(&visible);
                }
            }
        }
    }
    let remaining = filter.flush();
    if !remaining.is_empty() {
        print!("{}", remaining);
        full_text.push_str(&remaining);
    }
    println!("\n");

    Ok(full_text)
}

fn strip_think_tags(s: &str) -> String {
    let mut result = s.to_string();
    if let Some(end) = result.find("</think>") {
        if result[..end].find("<think>").is_none() {
            result = result[end + 8..].to_string();
        }
    }
    while let Some(start) = result.find("<think>") {
        if let Some(end) = result[start..].find("</think>") {
            result = format!("{}{}", &result[..start], &result[start + end + 8..]);
        } else {
            result = result[..start].to_string();
            break;
        }
    }
    result = result.replace("<|im_start|>", "").replace("<|im_end|>", "");
    let trimmed = result.trim_start();
    for prefix in &["system\n", "assistant\n", "user\n"] {
        if trimmed.starts_with(prefix) {
            result = trimmed[prefix.len()..].to_string();
            break;
        }
    }
    result
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let t: String = s.chars().take(max).collect();
        format!("{t}...")
    }
}
