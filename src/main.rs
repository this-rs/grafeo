//! obrain-chat — Generic Graph-Augmented LLM with Attention Masking
//!
//! Architecture:
//! 1. Startup: open graph DB + auto-discover schema (labels, hierarchy, properties)
//! 2. Per query: structured retrieval driven by schema (fuzzy label match + name search)
//! 3. BFS expand + topological attention mask
//! 4. Send to llama.cpp server with mask
//!
//! Schema-agnostic: works on ANY graph, not just the PO schema.

mod engine;
mod mask_builder;

use anyhow::{Context, Result};
use engine::LlamaEngine;
use obrain::ObrainDB;
use obrain_common::types::{NodeId, PropertyKey, Value};
use obrain_core::graph::{Direction, lpg::LpgStore};
#[cfg(feature = "http")]
use serde_json::json;
use std::collections::{HashMap, HashSet, VecDeque};
use std::io::{self, BufRead, Write};
#[cfg(feature = "http")]
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;
use chrono::Utc;

use engine::EngineConfig;

/// Global debug flag — set via --debug CLI arg.
static DEBUG: AtomicBool = AtomicBool::new(false);

/// Print to stderr only if --debug is enabled.
macro_rules! debug {
    ($($arg:tt)*) => {
        if DEBUG.load(Ordering::Relaxed) {
            eprintln!($($arg)*);
        }
    };
}

#[cfg(feature = "http")]
const DEFAULT_SERVER: &str = "http://localhost:8090";
// Legacy constant — replaced by build_system_header() which is persona-aware.
#[allow(dead_code)]
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

    debug!("  Schema discovered in {:.0}ms:", t0.elapsed().as_millis());
    debug!("    {} labels ({} structural, {} noise)",
        labels_with_importance.len(), structural_labels.len(), noise_labels.len());
    for info in labels_with_importance.iter().take(10) {
        let marker = if info.is_noise { " [noise]" } else { "" };
        debug!("      {:>6} {:20} imp={:.3} deg={:.1}{}",
            info.count, info.label, info.importance, info.avg_degree, marker);
    }
    if let Some(top_parent) = parent_child.iter().next() {
        debug!("    Hierarchy sample: {} → {:?}", top_parent.0,
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

    // Description — extract meaningful content lines (not just the first line)
    let desc = {
        let raw_desc = if let Some(dp) = dp {
            if let Some(ref field) = dp.desc_field {
                node.properties.get(&PropertyKey::from(field.as_str()))
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty() && s != &name)
                    .unwrap_or("")
            } else { "" }
        } else {
            ["description", "content", "text", "message", "body", "rationale"].iter()
                .filter_map(|&k| node.properties.get(&PropertyKey::from(k))
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty() && s != &name))
                .next()
                .unwrap_or("")
        };
        if raw_desc.is_empty() {
            String::new()
        } else {
            // Take up to 500 chars of meaningful content (skip markdown headers, separators)
            let mut out = String::new();
            for line in raw_desc.lines() {
                let trimmed = line.trim();
                if trimmed.is_empty() || trimmed.starts_with("---") { continue; }
                if !out.is_empty() { out.push(' '); }
                out.push_str(trimmed);
                if out.len() >= 500 { break; }
            }
            truncate(&out, 500)
        }
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
    #[allow(dead_code)]
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

    /// Register and encode a node into the KV cache via FFI.
    fn register(&mut self, node_id: NodeId, text: &str, engine: &LlamaEngine) -> Result<()> {
        let tokens = engine.tokenize(text, false, true)?;
        let n_tokens = tokens.len() as i32;
        let positions: Vec<i32> = (self.next_pos..self.next_pos + n_tokens).collect();
        engine.encode(&tokens, &positions, 0)?; // seq_id=0 = persistent context

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
        Ok(())
    }

    /// Get the KV position range for a node (if loaded).
    fn get_slot(&self, node_id: NodeId) -> Option<&KvSlot> {
        self.nodes.get(&node_id)
    }

    /// Update the system header text (e.g. when facts change mid-session).
    /// In FFI mode, the KV cache positions 0..header_end are already encoded and
    /// cannot be cheaply replaced. The updated text will be used for:
    /// - HTTP mode: reconstruct_prompt() will use the new header
    /// - FFI mode: the new header takes effect on next full session restart
    /// Facts injected via the header ARE visible immediately because they were
    /// also just encoded as conversation context by detect_facts.
    fn update_header(&mut self, new_header: &str) {
        self.header_text = new_header.to_string();
    }

    /// Reconstruct the full prompt prefix (header + all loaded nodes in order).
    /// Used in HTTP mode for `cache_prompt: true` matching.
    #[cfg(feature = "http")]
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
        debug!("  KV: {} nodes, {} tokens, hit_rate={:.0}% (hits={}, misses={}, queries={})",
            n_nodes, occupancy,
            self.metrics.hit_rate() * 100.0,
            self.metrics.cache_hits, self.metrics.cache_misses,
            self.metrics.total_queries);
    }

    /// Evict least-recently-used nodes to free up token capacity.
    ///
    /// FFI mode: positions are monotonic — no rebuild_positions().
    /// Gaps from eviction are accepted. Full-recompact at 90% n_ctx.
    fn evict_lru(&mut self, tokens_needed: i32, protected: &HashSet<NodeId>, engine: &LlamaEngine) -> Vec<NodeId> {
        let mut candidates: Vec<(NodeId, u64, i32)> = self.nodes.iter()
            .filter(|(id, _)| !protected.contains(id))
            .map(|(id, slot)| (*id, slot.last_used, slot.n_tokens))
            .collect();
        candidates.sort_by_key(|(_, last_used, _)| *last_used);

        let mut freed = 0i32;
        let mut evicted: Vec<NodeId> = Vec::new();

        for (nid, last_used, n_tokens) in &candidates {
            if freed >= tokens_needed { break; }
            debug!("    evict: node {} (last_q={}, {} tokens)", nid.0, last_used, n_tokens);
            evicted.push(*nid);
            freed += n_tokens;
        }

        // Remove from KV cache (FFI) + registry
        for nid in &evicted {
            if let Some(slot) = self.nodes.remove(nid) {
                engine.evict(slot.start, slot.end);
            }
        }
        self.order.retain(|id| !evicted.contains(id));

        // FFI: update next_pos to actual KV cache state (llama.cpp requires consecutive positions)
        if !evicted.is_empty() {
            self.next_pos = engine.seq_pos_max(0) + 1;
        }

        debug!("  Evicted {} nodes, freed ~{} tokens", evicted.len(), freed);
        evicted
    }

    /// Ensure capacity for `needed_tokens` new tokens.
    /// If position space is near 90% of n_ctx, do a full recompact.
    fn ensure_capacity(&mut self, needed_tokens: i32, max_kv_tokens: i32, protected: &HashSet<NodeId>, engine: &LlamaEngine) {
        // Check if position space is nearly exhausted (full-recompact trigger)
        let n_ctx = engine.n_ctx() as i32;
        if self.next_pos + needed_tokens > (n_ctx as f64 * 0.9) as i32 {
            eprintln!("  ⚠ Position space near limit ({}/{}), full recompact...", self.next_pos, n_ctx);
            self.full_recompact(engine);
        }

        let total_tokens: i32 = self.nodes.values().map(|s| s.n_tokens).sum();
        let available = max_kv_tokens - total_tokens;
        if available >= needed_tokens { return; }

        let to_free = needed_tokens - available;
        debug!("  KV full: {} tokens in cache, need {} more (budget {}), evicting...",
            total_tokens, needed_tokens, max_kv_tokens);
        self.evict_lru(to_free, protected, engine);
    }

    /// Full recompact: clear everything and re-encode from scratch.
    /// Used when position space is nearly exhausted (gaps from evictions).
    fn full_recompact(&mut self, engine: &LlamaEngine) {
        debug!("  Full recompact: clearing KV and re-encoding {} nodes...", self.nodes.len());
        engine.clear_kv();

        // Re-encode header
        if let Ok(tokens) = engine.tokenize(&self.header_text, false, true) {
            let positions: Vec<i32> = (0..tokens.len() as i32).collect();
            let _ = engine.encode(&tokens, &positions, 0);
        }

        // Re-encode all nodes with fresh contiguous positions
        let mut pos = self.header_end;
        for nid in &self.order {
            if let Some(slot) = self.nodes.get_mut(nid) {
                if let Ok(tokens) = engine.tokenize(&slot.text, false, true) {
                    let n_tok = tokens.len() as i32;
                    let positions: Vec<i32> = (pos..pos + n_tok).collect();
                    let _ = engine.encode(&tokens, &positions, 0);
                    slot.start = pos;
                    slot.end = pos + n_tok;
                    slot.n_tokens = n_tok;
                    pos += n_tok;
                }
            }
        }
        self.next_pos = pos;
        debug!("  Recompact done: {} nodes, next_pos={}", self.nodes.len(), self.next_pos);
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

    debug!("  Discovered {} banks in {:.0}ms (top label: {})",
        banks.len(), t0.elapsed().as_millis(), top_label);
    for bank in banks.iter().take(8) {
        debug!("    {} ({} nodes, ~{} tokens)",
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
    bank: u32, // 0=core, 1=relations, 2=2-hop, 3=background
}

struct QueryContext {
    nodes: Vec<ContextNode>,
    #[allow(dead_code)] // Reserved for future Rule 2 (inter-node attention when re-encode is supported)
    adjacency: HashMap<NodeId, HashSet<NodeId>>,
    total_tokens: i32,
    header_tokens: i32,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Conversation Fragment Registry — conversation turns as KV cache nodes
// ═══════════════════════════════════════════════════════════════════════════════

/// Conversation fragments live in NodeId space 0xFFFF_0000_0000_xxxx
/// to avoid collision with graph NodeIds.
const CONV_NODE_BASE: u64 = 0xFFFF_0000_0000_0000;

/// A conversation fragment — a concise Q&A summary stored as a KV node.
#[derive(Clone, Debug)]
#[allow(dead_code)]
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
        engine: &LlamaEngine,
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
            .chain(self.fragments.iter().map(|f| f.node_id))
            .collect();
        registry.ensure_capacity(est_tokens, kv_capacity, &protected, engine);

        // Encode and register in the KV cache via FFI
        let n_tokens = engine.token_count(&kv_text)? as i32;
        registry.register(node_id, &kv_text, engine)?;

        let fragment = ConvFragment {
            node_id,
            question: question.to_string(),
            terms,
            kv_text,
            related_graph_nodes: related_nodes.to_vec(),
            turn,
        };

        debug!("  [Conv] Registered turn {} as KV node {} ({} tokens)",
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
            debug!("  [Conv] {} relevant fragments (of {})",
                relevant_ids.len(), self.fragments.len());
        }

        (relevant_ids, conv_adjacency)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Persona DB — stores conversations, facts and identity in obrain graph
// ═══════════════════════════════════════════════════════════════════════════════

struct PersonaDB {
    db: ObrainDB,
    current_conv_id: NodeId,
}

impl PersonaDB {
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

    // ── Fact management ──────────────────────────────────────────

    /// Store a persistent fact (key-value pair).
    /// If a fact with the same key already exists, it is deactivated and replaced.
    fn add_fact(&self, key: &str, value: &str, source_turn: u32) -> NodeId {
        // Deactivate existing fact with same key
        let store = self.db.store();
        for &nid in &store.nodes_by_label("Fact") {
            if let Some(node) = store.get_node(nid) {
                let k = node.properties.get(&PropertyKey::from("key"))
                    .and_then(|v| v.as_str()).unwrap_or("");
                let active = node.properties.get(&PropertyKey::from("active"))
                    .and_then(|v| if let Value::Bool(b) = v { Some(*b) } else { None })
                    .unwrap_or(true);
                if k == key && active {
                    self.db.set_node_property(nid, "active", Value::Bool(false));
                }
            }
        }

        let fact_id = self.db.create_node_with_props(&["Fact"], [
            ("key", Value::String(key.to_string().into())),
            ("value", Value::String(value.to_string().into())),
            ("source_turn", Value::Int64(source_turn as i64)),
            ("created_at", Value::String(Utc::now().to_rfc3339().into())),
            ("active", Value::Bool(true)),
        ]);
        fact_id
    }

    /// Get all active facts as (key, value) pairs.
    fn active_facts(&self) -> Vec<(String, String)> {
        let store = self.db.store();
        let mut facts = Vec::new();
        for &nid in &store.nodes_by_label("Fact") {
            if let Some(node) = store.get_node(nid) {
                let active = node.properties.get(&PropertyKey::from("active"))
                    .and_then(|v| if let Value::Bool(b) = v { Some(*b) } else { None })
                    .unwrap_or(true);
                if !active { continue; }
                let key = node.properties.get(&PropertyKey::from("key"))
                    .and_then(|v| v.as_str()).unwrap_or("").to_string();
                let value = node.properties.get(&PropertyKey::from("value"))
                    .and_then(|v| v.as_str()).unwrap_or("").to_string();
                facts.push((key, value));
            }
        }
        facts
    }

    /// List all facts with details for /facts command.
    fn list_facts(&self) -> Vec<(NodeId, String, String, i64, bool)> {
        let store = self.db.store();
        let mut facts = Vec::new();
        for &nid in &store.nodes_by_label("Fact") {
            if let Some(node) = store.get_node(nid) {
                let key = node.properties.get(&PropertyKey::from("key"))
                    .and_then(|v| v.as_str()).unwrap_or("").to_string();
                let value = node.properties.get(&PropertyKey::from("value"))
                    .and_then(|v| v.as_str()).unwrap_or("").to_string();
                let turn = node.properties.get(&PropertyKey::from("source_turn"))
                    .and_then(|v| if let Value::Int64(n) = v { Some(*n) } else { None })
                    .unwrap_or(0);
                let active = node.properties.get(&PropertyKey::from("active"))
                    .and_then(|v| if let Value::Bool(b) = v { Some(*b) } else { None })
                    .unwrap_or(true);
                facts.push((nid, key, value, turn, active));
            }
        }
        facts
    }

    /// Deactivate a fact by key.
    fn forget_fact(&self, key: &str) -> bool {
        let store = self.db.store();
        let mut found = false;
        for &nid in &store.nodes_by_label("Fact") {
            if let Some(node) = store.get_node(nid) {
                let k = node.properties.get(&PropertyKey::from("key"))
                    .and_then(|v| v.as_str()).unwrap_or("");
                let active = node.properties.get(&PropertyKey::from("active"))
                    .and_then(|v| if let Value::Bool(b) = v { Some(*b) } else { None })
                    .unwrap_or(true);
                if k == key && active {
                    self.db.set_node_property(nid, "active", Value::Bool(false));
                    found = true;
                }
            }
        }
        found
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Fact detection — heuristic extraction of persistent facts from user messages
// ═══════════════════════════════════════════════════════════════════════════════

/// Detect persistent facts in user message. Returns Vec<(key, value)>.
fn detect_facts(msg: &str) -> Vec<(String, String)> {
    let lower = msg.to_lowercase();
    let mut facts = Vec::new();

    // Patterns: "ton nom est X", "tu t'appelles X", "your name is X"
    for prefix in &[
        "ton nom est ", "tu t'appelles ", "tu es ", "appelle-toi ",
        "your name is ", "you are ", "call yourself ",
    ] {
        if let Some(rest) = lower.strip_prefix(prefix) {
            let value = rest.trim().trim_end_matches(|c: char| c == '.' || c == '!' || c == '?');
            if !value.is_empty() && value.len() < 100 {
                facts.push(("name".to_string(), value.to_string()));
            }
        }
    }

    // Patterns: "retiens que X", "rappelle-toi que X", "remember that X"
    for prefix in &[
        "retiens que ", "rappelle-toi que ", "rappelle toi que ",
        "n'oublie pas que ", "remember that ", "don't forget that ",
        "mémorise que ", "memorise que ",
    ] {
        if let Some(rest) = lower.strip_prefix(prefix) {
            let value = rest.trim().trim_end_matches(|c: char| c == '.' || c == '!');
            if !value.is_empty() && value.len() < 200 {
                facts.push(("memory".to_string(), value.to_string()));
            }
        }
    }

    // Patterns: "réponds toujours en X", "always respond in X"
    for prefix in &[
        "réponds toujours en ", "reponds toujours en ",
        "parle toujours en ", "always respond in ", "always answer in ",
    ] {
        if let Some(rest) = lower.strip_prefix(prefix) {
            let value = rest.trim().trim_end_matches(|c: char| c == '.' || c == '!');
            if !value.is_empty() && value.len() < 50 {
                facts.push(("language".to_string(), value.to_string()));
            }
        }
    }

    // Infix patterns: "... ton nom est X ..."  "... tu es X ..."
    if facts.is_empty() {
        for (marker, key) in &[
            ("ton nom est ", "name"), ("tu t'appelles ", "name"),
            ("your name is ", "name"), ("tu es ", "identity"),
        ] {
            if let Some(pos) = lower.find(marker) {
                let rest = &lower[pos + marker.len()..];
                // Take until end of sentence or comma
                let value: String = rest.chars()
                    .take_while(|c| *c != '.' && *c != ',' && *c != '!' && *c != '?')
                    .collect();
                let value = value.trim().to_string();
                if !value.is_empty() && value.len() < 100 {
                    facts.push((key.to_string(), value));
                }
            }
        }
    }

    facts
}

/// Build dynamic system header based on graph presence and persistent facts.
fn build_system_header(has_graph: bool, facts: &[(String, String)]) -> String {
    let mut header = String::from("<|im_start|>system\n");

    // Identity from facts
    let name = facts.iter().find(|(k, _)| k == "name").map(|(_, v)| v.as_str());
    if let Some(name) = name {
        header.push_str(&format!("You are {name}. "));
    } else {
        header.push_str("You are a helpful assistant. ");
    }

    if has_graph {
        header.push_str("Below is structured data from a graph database. Each entry shows [Type] Name and its relations. Use ONLY this data to answer. ");
    }

    header.push_str("Answer in the same language as the question. /no_think\n");

    // Inject other persistent facts
    let other_facts: Vec<&(String, String)> = facts.iter()
        .filter(|(k, _)| k != "name")
        .collect();
    if !other_facts.is_empty() {
        header.push_str("\nKnown facts:\n");
        for (key, value) in &other_facts {
            header.push_str(&format!("- {key}: {value}\n"));
        }
    }

    header.push('\n');
    header
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════════

fn parse_arg(flag: &str) -> Option<String> {
    std::env::args()
        .position(|a| a == flag)
        .and_then(|i| std::env::args().nth(i + 1))
}

fn main() -> Result<()> {
    // --debug flag (no value, just presence)
    let is_debug = std::env::args().any(|a| a == "--debug");
    if is_debug {
        DEBUG.store(true, Ordering::Relaxed);
    }
    // Control llama.cpp C-level log verbosity
    engine::set_verbose(is_debug);

    let model_path = parse_arg("--model")
        .unwrap_or_else(|| "/Users/triviere/models/qwen3-8b-q8.gguf".to_string());

    let db_path: Option<String> = parse_arg("--db");

    let max_nodes: usize = parse_arg("--max-nodes")
        .and_then(|s| s.parse().ok())
        .unwrap_or(40);

    let token_budget: i32 = parse_arg("--budget")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1400);

    let kv_capacity: i32 = parse_arg("--kv-capacity")
        .and_then(|s| s.parse().ok())
        .unwrap_or(4096);

    let n_ctx: u32 = parse_arg("--n-ctx")
        .and_then(|s| s.parse().ok())
        .unwrap_or(32768);

    let n_gpu: i32 = parse_arg("--n-gpu")
        .and_then(|s| s.parse().ok())
        .unwrap_or(99);

    let persona_path: Option<PathBuf> = parse_arg("--persona").map(PathBuf::from);

    eprintln!("=== obrain-chat — Generic Graph LLM + Topological Mask (FFI) ===\n");

    // ── Load LLM via FFI ────────────────────────────────────────
    eprintln!("Loading model: {model_path}");
    let engine = LlamaEngine::new(&EngineConfig {
        model_path: model_path.clone(),
        n_ctx,
        n_gpu_layers: n_gpu,
        ..EngineConfig::default()
    })?;
    eprintln!("  Model loaded: n_ctx={}", engine.n_ctx());

    // ── Open database (optional) ──────────────────────────────────────
    let (db_holder, store, schema, banks): (
        Option<ObrainDB>,
        Option<Arc<LpgStore>>,
        Option<GraphSchema>,
        Vec<KvBank>,
    ) = if let Some(ref db_path) = db_path {
        eprintln!("Opening database: {db_path}");
        let ckpt = format!("{db_path}/wal/checkpoint.meta");
        let _ = std::fs::remove_file(&ckpt);

        let db = ObrainDB::open(db_path)
            .context(format!("Failed to open DB at {db_path}"))?;
        let st = Arc::clone(db.store());
        eprintln!("  {} nodes, {} edges\n", st.node_count(), st.edge_count());

        debug!("Discovering schema...");
        let sch = discover_schema(&st);
        debug!("\n{}", "=".repeat(60));
        eprintln!("Ready. {} structural labels, {} hierarchy rules.",
            sch.structural_labels.len(),
            sch.parent_child.len());

        // Discover or load KV Banks
        let bank_cache_path = std::path::PathBuf::from(format!("{db_path}.banks"));
        let nc = st.node_count();
        let ec = st.edge_count();
        let bnks = match load_bank_cache(&bank_cache_path, nc, ec) {
            Some(cached) => {
                debug!("Loaded {} banks from cache ({})", cached.len(), bank_cache_path.display());
                cached
            }
            None => {
                debug!("Discovering banks...");
                let discovered = discover_banks(&st, &sch, 50);
                save_bank_cache(&bank_cache_path, &discovered, nc, ec);
                debug!("  Saved bank cache to {}", bank_cache_path.display());
                discovered
            }
        };

        (Some(db), Some(st), Some(sch), bnks)
    } else {
        eprintln!("No database specified (use --db <path> for graph-augmented mode)\n");
        (None, None, None, Vec::new())
    };
    let _ = &db_holder; // keep DB alive for the store Arc

    // ── Conversation DB (auto-created if not specified) ──────────────
    let persona_resolved_path: String = if let Some(ref cp) = persona_path {
        cp.to_str().unwrap_or("conv.db").to_string()
    } else if let Some(ref db_p) = db_path {
        // Auto-create alongside the graph DB
        format!("{db_p}.persona")
    } else {
        // No graph DB: use ~/.obrain-chat/conv.db
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
        let dir = format!("{home}/.obrain-chat");
        let _ = std::fs::create_dir_all(&dir);
        format!("{dir}/default.persona")
    };
    debug!("Persona DB path: {persona_resolved_path}");
    // Remove stale checkpoint if exists
    let ckpt_conv = format!("{persona_resolved_path}/wal/checkpoint.meta");
    let _ = std::fs::remove_file(&ckpt_conv);
    let mut persona_db = match PersonaDB::open(&persona_resolved_path) {
        Ok(cdb) => {
            let convs = cdb.list_conversations();
            eprintln!("Persona: {} ({} conversations, current: \"{}\")",
                persona_resolved_path, convs.len(), cdb.current_title());
            let recent = cdb.recent_messages(4);
            if !recent.is_empty() {
                debug!("  Recent context ({} messages):", recent.len());
                for (role, content) in &recent {
                    let snippet: String = content.chars().take(80).collect();
                    debug!("    {}: {}{}",
                        role, snippet,
                        if content.len() > 80 { "..." } else { "" });
                }
            }
            Some(cdb)
        }
        Err(e) => {
            eprintln!("  Warning: could not open persona DB: {e}");
            None
        }
    };

    if store.is_some() {
        eprintln!("\nCommands: /quit, /schema, /kv, /banks, /history, /conversations, /new <title>\n");
    } else {
        eprintln!("\nCommands: /quit, /history, /conversations, /new <title>\n");
    }

    // ── Load persistent facts and build dynamic system header ──
    let persona_facts: Vec<(String, String)> = persona_db.as_ref()
        .map(|pdb| pdb.active_facts())
        .unwrap_or_default();
    if !persona_facts.is_empty() {
        eprintln!("  Facts: {}", persona_facts.iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(", "));
    }
    let system_header = build_system_header(store.is_some(), &persona_facts);
    debug!("System header:\n{system_header}");

    // ── Initialize KV Node Registry ─────────────────────────────
    let header_tokens_vec = engine.tokenize(&system_header, false, true)?;
    let header_n = header_tokens_vec.len() as i32;
    let header_positions: Vec<i32> = (0..header_n).collect();
    engine.encode(&header_tokens_vec, &header_positions, 0)?;
    let mut registry = KvNodeRegistry::new(&system_header, header_n);
    let mut conv_frags = ConvFragments::new();
    debug!("KV Registry initialized: header={} tokens (encoded in KV)", header_n);

    // ── Warmup: pre-load top banks into KV ──────────────────────
    let warmup_count = 3.min(banks.len());
    if warmup_count > 0 {
        let warmup_t0 = Instant::now();
        let mut warmup_loaded = 0;
        for bank in banks.iter().take(warmup_count) {
            let protected: HashSet<NodeId> = bank.node_ids.iter().copied().collect();
            registry.ensure_capacity(bank.est_tokens, kv_capacity, &protected, &engine);
            for nid in &bank.node_ids {
                if registry.get_slot(*nid).is_some() { continue; }
                if let Some(text) = bank.texts.get(nid) {
                    registry.register(*nid, text, &engine)?;
                    warmup_loaded += 1;
                }
            }
        }
        debug!("  Warmup: pre-loaded {} banks ({} nodes) in {:.0}ms",
            warmup_count, warmup_loaded, warmup_t0.elapsed().as_millis());
    }

    // ── Restore conversation fragments from persona_db (T3) ──────────
    if let Some(ref cdb) = persona_db {
        let recent = cdb.recent_messages(20); // 20 messages = up to 10 Q/A pairs
        if !recent.is_empty() {
            let restore_t0 = Instant::now();
            let mut restored = 0u32;
            // Pair up consecutive (user, assistant) messages
            let mut i = 0;
            while i + 1 < recent.len() {
                let (ref role_q, ref content_q) = recent[i];
                let (ref role_a, ref content_a) = recent[i + 1];
                if role_q == "user" && role_a == "assistant" {
                    if let Err(e) = conv_frags.add_turn(
                        content_q, content_a, &[],
                        &mut registry, &engine, kv_capacity,
                    ) {
                        debug!("  Warning: could not restore conv fragment: {e}");
                    } else {
                        restored += 1;
                    }
                    i += 2;
                } else {
                    i += 1;
                }
            }
            if restored > 0 {
                debug!("  Restored {} conversation fragments in {:.0}ms",
                    restored, restore_t0.elapsed().as_millis());
            }
        }
    }

    // ── Interactive loop ─────────────────────────────────────────────
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();
    let mut turn_count: u32 = 0;
    #[allow(unused_assignments)]
    let mut current_facts = persona_facts; // live-updated when facts change

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
            if let Some(ref sch) = schema {
                eprintln!("  Structural: {:?}", sch.structural_labels);
                eprintln!("  Noise: {:?}", sch.noise_labels);
                for (parent, children) in &sch.parent_child {
                    eprintln!("  {} → {:?}", parent, children);
                }
            } else {
                eprintln!("  No graph loaded (use --db <path>)");
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
            if let Some(ref cdb) = persona_db {
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
                eprintln!("  No persona DB (use --persona <path>)");
            }
            continue;
        }
        if line == "/conversations" || line == "/convs" {
            if let Some(ref cdb) = persona_db {
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
                eprintln!("  No persona DB (use --persona <path>)");
            }
            continue;
        }
        if line.starts_with("/new ") {
            if let Some(ref mut cdb) = persona_db {
                let title = line.strip_prefix("/new ").unwrap().trim();
                cdb.new_conversation(title);
                eprintln!("  Created new conversation: \"{}\"", title);
            } else {
                eprintln!("  No persona DB (use --persona <path>)");
            }
            continue;
        }
        if line.starts_with("/switch ") {
            if let Some(ref mut cdb) = persona_db {
                let id_str = line.strip_prefix("/switch ").unwrap().trim();
                if let Ok(id_num) = id_str.parse::<u64>() {
                    if cdb.switch_to(NodeId(id_num)) {
                        eprintln!("  Switched to conversation: \"{}\"", cdb.current_title());
                    } else {
                        eprintln!("  Conversation {} not found", id_num);
                    }
                }
            } else {
                eprintln!("  No persona DB (use --persona <path>)");
            }
            continue;
        }
        if line == "/facts" {
            if let Some(ref pdb) = persona_db {
                let facts = pdb.list_facts();
                let active: Vec<_> = facts.iter().filter(|(_, _, _, _, a)| *a).collect();
                if active.is_empty() {
                    eprintln!("  (no facts stored)");
                } else {
                    eprintln!("  Persistent facts ({}):", active.len());
                    for (nid, key, value, turn, _) in &active {
                        eprintln!("    {} = {} (turn {}, id={})", key, value, turn, nid.0);
                    }
                }
            } else {
                eprintln!("  No persona DB (use --persona <path>)");
            }
            continue;
        }
        if line.starts_with("/forget ") {
            if let Some(ref pdb) = persona_db {
                let key = line.strip_prefix("/forget ").unwrap().trim();
                if pdb.forget_fact(key) {
                    eprintln!("  ✓ Forgot fact: {}", key);
                } else {
                    eprintln!("  No active fact with key '{}'", key);
                }
            } else {
                eprintln!("  No persona DB (use --persona <path>)");
            }
            continue;
        }
        if line == "/kv" {
            registry.log_metrics();
            eprintln!("  Loaded nodes ({}):", registry.order.len());
            for nid in registry.order.iter().take(20) {
                let (label, name) = if let (Some(st), Some(sch)) = (&store, &schema) {
                    let label: Option<String> = st.get_node(*nid)
                        .and_then(|n| n.labels.first().map(|l| {
                            let s: &str = l.as_ref();
                            s.to_string()
                        }));
                    let dp = label.as_deref()
                        .and_then(|l| sch.display_props.get(l));
                    let name = get_node_name_generic(st, *nid, dp);
                    (label, name)
                } else {
                    (None, String::new())
                };
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
        let user_msg_id = persona_db.as_ref().map(|cdb| cdb.add_message("user", &line));

        // T6: Meta queries (identity, memory) bypass graph retrieval
        let meta = is_meta_query(&line);
        let (q_store, q_schema) = if meta {
            debug!("  [Meta] Query is about identity/memory — skipping graph retrieval");
            (None, None)
        } else {
            (store.as_ref(), schema.as_ref())
        };

        match query_with_registry(&engine, q_store, q_schema, &mut registry, &mut conv_frags, &banks, &line, max_nodes, token_budget, kv_capacity) {
            Ok((response, relevant_graph_nodes)) => {
                // Response already streamed to stdout by engine.generate
                let clean = strip_think_tags(&response);
                let trimmed = clean.trim().to_string();

                // Register Q&A as a concise fragment in the KV cache
                if let Err(e) = conv_frags.add_turn(
                    &line, &trimmed, &relevant_graph_nodes,
                    &mut registry, &engine, kv_capacity,
                ) {
                    eprintln!("  Warning: could not register conv fragment: {e}");
                }

                // Store full messages in conversation DB (for persistence across sessions)
                if let Some(ref cdb) = persona_db {
                    let asst_id = cdb.add_message("assistant", &trimmed);
                    if let Some(uid) = user_msg_id {
                        cdb.link_reply(asst_id, uid);
                    }
                }

                // Detect and store persistent facts from user input (T4)
                turn_count += 1;
                let detected = detect_facts(&line);
                if !detected.is_empty() {
                    if let Some(ref pdb) = persona_db {
                        for (key, value) in &detected {
                            pdb.add_fact(key, value, turn_count);
                            eprintln!("  💾 Fact stored: {} = {}", key, value);
                        }
                        // Update live facts and rebuild header for next query
                        current_facts = pdb.active_facts();
                        let new_header = build_system_header(store.is_some(), &current_facts);
                        registry.update_header(&new_header);
                        debug!("  Header updated with {} facts", current_facts.len());
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
// Meta-query detection (T6 — Hybrid retrieval)
// ═══════════════════════════════════════════════════════════════════════════════

/// Detect if a query is about the system itself (identity, memory, facts).
/// These queries should be answered from persona facts + conv history,
/// NOT from the knowledge graph.
fn is_meta_query(query: &str) -> bool {
    let lower = query.to_lowercase();

    // Identity questions
    let identity_patterns = [
        "qui es-tu", "qui es tu", "c'est quoi ton nom", "quel est ton nom",
        "comment tu t'appelles", "comment t'appelles-tu",
        "what is your name", "who are you", "what are you",
        "tu es qui", "tu t'appelles comment",
    ];
    if identity_patterns.iter().any(|p| lower.contains(p)) {
        return true;
    }

    // Memory / fact recall questions
    let memory_patterns = [
        "qu'est-ce que tu sais sur moi", "que sais-tu de moi", "que sais-tu sur moi",
        "qu'est-ce que tu retiens", "que retiens-tu",
        "what do you know about me", "what do you remember",
        "tu te souviens", "te rappelles-tu", "te souviens-tu",
        "do you remember",
    ];
    if memory_patterns.iter().any(|p| lower.contains(p)) {
        return true;
    }

    false
}

// ═══════════════════════════════════════════════════════════════════════════════
// Registry-aware retrieval pipeline
// ═══════════════════════════════════════════════════════════════════════════════

fn query_with_registry(
    engine: &LlamaEngine,
    store: Option<&Arc<LpgStore>>,
    schema: Option<&GraphSchema>,
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
                registry.ensure_capacity(bank.est_tokens, kv_capacity, &protected, engine);

                debug!("  Loading bank '{}': {} nodes ({} new)",
                    bank.name, bank.node_ids.len(), missing.len());
                for nid in &missing {
                    if let Some(text) = bank.texts.get(nid) {
                        registry.register(*nid, text, engine)?;
                    }
                }
                bank_loaded = true;
            } else {
                debug!("  Bank '{}' already fully loaded in KV", bank.name);
                registry.touch(&bank.node_ids);
                bank_loaded = true;
            }
        }
    }
    if bank_loaded {
        debug!("  Bank loading done in {}ms", t_start.elapsed().as_millis());
    }

    // Run the generic retrieval to get scored nodes (only if graph is loaded)
    let (scored_nodes_for_query, adjacency, node_texts) =
        if let (Some(st), Some(sch)) = (store, schema) {
            retrieve_nodes(engine, st, sch, query, max_nodes, token_budget)?
        } else {
            (Vec::new(), HashMap::new(), HashMap::new())
        };

    if scored_nodes_for_query.is_empty() {
        // No graph data — generate from conversation context + system prompt
        let fallback_text = format!(
            "<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
        );
        let tokens = engine.tokenize(&fallback_text, false, true)?;
        // seq_cp: let seq_id=1 see the system header on seq_id=0
        engine.seq_cp(0, 1, 0, -1);
        print!("assistant> ");
        io::stdout().flush()?;
        let mut filter = ThinkFilter::new();
        let (resp, _) = engine.generate(&tokens, registry.next_pos, 256, 1, |piece| {
            let visible = filter.feed(piece);
            if !visible.is_empty() {
                print!("{}", visible);
                let _ = io::stdout().flush();
            }
            true
        })?;
        let remaining = filter.flush();
        if !remaining.is_empty() { print!("{}", remaining); }
        println!("\n");
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
    registry.ensure_capacity(est_missing_tokens, kv_capacity, &protected, engine);

    debug!("  KV: {} needed, {} cached, {} to encode (~{} tokens)",
        needed_ids.len(), needed_ids.len() - missing.len(), missing.len(), est_missing_tokens);

    // Encode missing nodes into KV via FFI
    for &nid in &missing {
        if let Some(text) = node_texts.get(&nid) {
            registry.register(nid, text, engine)?;
        }
    }

    let encode_time = t_start.elapsed().as_millis();
    debug!("  KV encode: {}ms", encode_time);

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
        debug!("  [Conv] Pulled {} graph nodes from fragment references", conv_pulled_ids.len());
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
    // Bank assignment: top 25% → bank 0 (core), next 25% → bank 1, etc.
    let total_scored = scored_nodes_for_query.len();
    let bank_for_idx = |idx: usize| -> u32 {
        if total_scored == 0 { return 0; }
        let ratio = idx as f32 / total_scored as f32;
        if ratio < 0.25 { 0 } else if ratio < 0.50 { 1 } else if ratio < 0.75 { 2 } else { 3 }
    };
    let mut ctx_nodes: Vec<ContextNode> = Vec::new();
    if vague_query && conv_provides_context {
        // Vague query + conv context: only include graph nodes that are also
        // referenced by fragments (or already in fragment pull list)
        let pulled_set: HashSet<NodeId> = conv_pulled_ids.iter().copied().collect();
        debug!("  [Conv] Vague query detected — using {} fragment-referenced nodes instead of {} generic",
            conv_pulled_ids.len(), scored_nodes_for_query.len());
        for (idx, cn) in scored_nodes_for_query.iter().enumerate() {
            if pulled_set.contains(&cn.id) {
                if let Some(slot) = registry.get_slot(cn.id) {
                    ctx_nodes.push(ContextNode {
                        id: cn.id,
                        token_start: slot.start,
                        token_end: slot.end,
                        bank: bank_for_idx(idx),
                    });
                }
            }
        }
    } else {
        for (idx, cn) in scored_nodes_for_query.iter().enumerate() {
            if let Some(slot) = registry.get_slot(cn.id) {
                ctx_nodes.push(ContextNode {
                    id: cn.id,
                    token_start: slot.start,
                    token_end: slot.end,
                    bank: bank_for_idx(idx),
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
                    bank: 1, // conv-pulled graph nodes = relations tier
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
                bank: 2, // conversation fragments = 2-hop context tier
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
        nodes: ctx_nodes,
        adjacency: merged_adjacency,
    };

    registry.log_metrics();

    // Collect relevant graph node IDs for the caller (to link conv fragment)
    let relevant_graph_nodes: Vec<NodeId> = scored_nodes_for_query.iter()
        .map(|n| n.id)
        .collect();

    let response = generate_with_mask(engine, &ctx, query)?;
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
    _engine: &LlamaEngine,
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

    debug!("  Terms: {:?}", terms);
    debug!("  Matched labels: {:?}", target_labels);
    debug!("  Content terms: {:?}", content_terms);

    // ── Step 2: Fetch + score nodes by target labels ─────────────────
    let mut scored_nodes: Vec<(NodeId, f64)> = Vec::new();

    for (label, label_sim) in &target_labels {
        let node_ids = store.nodes_by_label(label);
        let info = schema.labels.iter().find(|l| l.label == *label);
        let importance = info.map_or(1.0, |i| i.importance);

        debug!("    {} {} nodes (imp={:.2})", node_ids.len(), label, importance);

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

    debug!("  Scored {} seed nodes", scored_nodes.len());

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

    debug!("  BFS expanded to {} nodes", bfs_result.len());

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
        debug!("    [{i}] score={:.2} {}", score, truncate(text, 80));
    }

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
        selected.push(ScoredContextNode { id: *nid, _score: *score });
        est_tokens += est;
    }

    debug!("  Selected {} nodes for query", selected.len());

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
    engine: &LlamaEngine,
    ctx: &QueryContext,
    query: &str,
) -> Result<String> {
    let query_text = format!(
        "<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
    );
    let query_tokens = engine.tokenize(&query_text, false, true)?;
    let query_n = query_tokens.len() as i32;

    let n_predict: i32 = 1024;

    // ── Build sparse position map ──────────────────────────────────
    // Only include positions that matter: header + relevant nodes + query + gen.
    // Positions from non-relevant nodes in the KV are excluded (Rule 3b).
    // n_gen_mask MUST cover the full generation to prevent fallback to default
    // causal attention (which would let the model see ALL KV including irrelevant nodes).
    let n_gen_mask: i32 = n_predict;

    let mut positions: Vec<i32> = Vec::new();
    let mut pos_remap: HashMap<i32, usize> = HashMap::new();

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
    let q_start_real = ctx.total_tokens;
    for offset in 0..(query_n + n_gen_mask) {
        let p = q_start_real + offset;
        let idx = positions.len();
        pos_remap.insert(p, idx);
        positions.push(p);
    }

    let sz = positions.len();
    let max_mask_positions: i32 = 4000;

    debug!("  Mask: {} sparse positions (of {} total KV), query={}, gen={}, {:.1}MB",
        sz, ctx.total_tokens, query_n, n_gen_mask,
        (sz * sz * 4) as f64 / 1_000_000.0);

    if sz as i32 > max_mask_positions {
        eprintln!("  ⚠ mask_size={} > {} — generating without mask.", sz, max_mask_positions);
        // Fallback: generate without topological mask (but still need seq_cp!)
        engine.seq_cp(0, 1, 0, -1);
        print!("assistant> ");
        io::stdout().flush()?;
        let mut filter = ThinkFilter::new();
        let (resp, _) = engine.generate(&query_tokens, q_start_real, n_predict, 1, |piece| {
            let visible = filter.feed(piece);
            if !visible.is_empty() {
                print!("{}", visible);
                let _ = io::stdout().flush();
            }
            true
        })?;
        let remaining = filter.flush();
        if !remaining.is_empty() { print!("{}", remaining); }
        println!("\n");
        return Ok(resp);
    }

    // ── Compile attention mask (sparse) ──────────────────────────────
    // FFI: use f32::NEG_INFINITY (exact), not -1e30 (JSON workaround eliminated)
    let mut mask = vec![f32::NEG_INFINITY; sz * sz];

    let allow = |m: &mut [f32], real_i: i32, real_j: i32, remap: &HashMap<i32, usize>, sz: usize| {
        if let (Some(&mi), Some(&mj)) = (remap.get(&real_i), remap.get(&real_j)) {
            if mi < sz && mj < sz && real_j <= real_i {
                m[mi * sz + mj] = 0.0;
            }
        }
    };

    let h = ctx.header_tokens;

    // Note: Rules 1 & 2 (inter-node attention) were removed — they set mask entries
    // for positions already in KV cache. The custom mask only affects tokens decoded
    // AFTER set_attn_mask(), i.e. query+gen tokens. Node KV representations are fixed
    // at encode() time with default causal attention.

    // Rule 0: Header positions — causal among themselves
    for i in 0..h {
        for j in 0..=i {
            allow(&mut mask, i, j, &pos_remap, sz);
        }
    }

    // Rule 1: Query + gen tokens see header + ALL relevant nodes + causal self
    // Non-relevant KV positions are NOT in pos_remap → invisible (sparse exclusion)
    for offset in 0..(query_n + n_gen_mask) {
        let i = q_start_real + offset;
        // See header
        for j in 0..h { allow(&mut mask, i, j, &pos_remap, sz); }
        // See all relevant node tokens
        for node in &ctx.nodes {
            for j in node.token_start..node.token_end {
                allow(&mut mask, i, j, &pos_remap, sz);
            }
        }
        // Causal self among query+gen tokens
        for prev in 0..=offset {
            allow(&mut mask, i, q_start_real + prev, &pos_remap, sz);
        }
    }

    debug!("  Mask: {} nodes visible to query", ctx.nodes.len());
    {
        let bank_counts: [usize; 4] = [
            ctx.nodes.iter().filter(|n| n.bank == 0).count(),
            ctx.nodes.iter().filter(|n| n.bank == 1).count(),
            ctx.nodes.iter().filter(|n| n.bank == 2).count(),
            ctx.nodes.iter().filter(|n| n.bank == 3).count(),
        ];
        debug!("  Banks: core={} relations={} 2hop={} background={}", bank_counts[0], bank_counts[1], bank_counts[2], bank_counts[3]);
    }

    // ── Set mask via FFI ──────────────────────────────────────────
    // Per-head masking: when enabled, different heads see different banks.
    // Enable via OBRAIN_PERHEAD=1 env var.
    let use_perhead = std::env::var("OBRAIN_PERHEAD").map(|v| v == "1").unwrap_or(false);
    if use_perhead {
        // Convert ContextNodes to mask_builder::NodePosition with bank from retrieval ranking.
        let mb_nodes: Vec<mask_builder::NodePosition> = ctx.nodes.iter().map(|n| {
            mask_builder::NodePosition {
                pos_start: n.token_start,
                pos_end: n.token_end,
                bank: n.bank,
            }
        }).collect();
        let config = mask_builder::default_bank_config();
        let perhead = mask_builder::build_perhead_mask(
            &mb_nodes,
            ctx.header_tokens,
            sz as i32,
            engine.n_head(),
            &config,
        );
        debug!("  Per-head mask: {} groups, {} positions, mask size {}", perhead.n_head_groups, perhead.positions.len(), perhead.mask.len());
        engine.set_attn_mask(&perhead.mask, &perhead.positions, perhead.n_head_groups)?;
    } else {
        debug!("  Broadcast mask: {} positions", positions.len());
        engine.set_attn_mask(&mask, &positions, 0)?;
    }

    // ── CRITICAL: Make seq_id=0 KV entries visible to seq_id=1 ──
    // Without this, query tokens on seq_id=1 cannot attend to context
    // nodes encoded on seq_id=0. This was the root cause of the model
    // ignoring all graph context (hallucinating instead of using data).
    engine.seq_cp(0, 1, 0, -1);
    debug!("  seq_cp(0→1) for full buffer (0..-1)");

    // ── Generate with streaming + auto-continuation ────────────────
    // Round 0: generate with custom mask (graph-guided attention).
    // If the model hits max_tokens without EOG, clear the mask and continue
    // with default causal attention. The model already has the graph context
    // from the first round. seq_id=1 is kept alive between rounds.
    print!("assistant> ");
    io::stdout().flush()?;

    let mut filter = ThinkFilter::new();
    let mut full_response = String::new();
    let max_continuations = 3;

    // Round 0: with mask
    let (chunk, mut hit_eog, mut next_pos) = engine.generate_ex(
        &query_tokens, q_start_real, n_predict, 1,
        true, // keep_seq=true in case we need continuation
        |piece| {
            let visible = filter.feed(piece);
            if !visible.is_empty() {
                print!("{}", visible);
                let _ = io::stdout().flush();
            }
            true
        }
    )?;
    full_response.push_str(&chunk);

    // Clear mask — either done or switching to causal for continuations
    engine.clear_attn_mask();

    // Continuations: no mask, causal attention only
    if !hit_eog {
        let cont_token = engine.tokenize(" ", false, false)?;

        for cont in 1..=max_continuations {
            debug!("  [continuation {}/{}] hit max_tokens at pos={}, continuing...",
                cont, max_continuations, next_pos);

            let is_last = cont == max_continuations;

            let (chunk, eog, end_pos) = engine.generate_ex(
                &cont_token, next_pos, n_predict, 1,
                !is_last, // keep_seq alive unless last round
                |piece| {
                    let visible = filter.feed(piece);
                    if !visible.is_empty() {
                        print!("{}", visible);
                        let _ = io::stdout().flush();
                    }
                    true
                }
            )?;

            next_pos = end_pos;
            full_response.push_str(&chunk);
            hit_eog = eog;

            if eog {
                // Natural end — clean up if seq was kept alive
                if !is_last { engine.clear_seq(1); }
                break;
            }
        }

        // If we exhausted continuations without EOG, final cleanup
        if !hit_eog {
            debug!("  [continuation] max reached, stopping (total ~{} chars)", full_response.len());
            // seq already cleaned by last generate_ex (keep_seq=false)
        }
    } else {
        // Natural end on round 0 — clean up seq_id=1
        engine.clear_seq(1);
    }

    let remaining = filter.flush();
    if !remaining.is_empty() { print!("{}", remaining); }
    println!("\n");

    Ok(full_response)
}

// ═══════════════════════════════════════════════════════════════════════════════
// HTTP helpers (kept behind feature flag for backwards compatibility)
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "http")]
fn check_server(client: &reqwest::blocking::Client, server: &str) -> Result<()> {
    let resp = client.get(format!("{server}/health")).send()
        .context(format!("Cannot reach {server}"))?;
    if !resp.status().is_success() {
        bail!("Server unhealthy: {}", resp.status());
    }
    debug!("Server: OK ({server})");
    Ok(())
}

#[cfg(feature = "http")]
fn tokenize(client: &reqwest::blocking::Client, server: &str, text: &str) -> Result<i32> {
    let resp = client
        .post(format!("{server}/tokenize"))
        .json(&json!({ "content": text, "special": true }))
        .send()?;
    let body: serde_json::Value = resp.json()?;
    Ok(body["tokens"].as_array().context("no tokens")?.len() as i32)
}

#[cfg(feature = "http")]
fn set_attn_mask(client: &reqwest::blocking::Client, server: &str, mask: &[f32], positions: &[i32]) -> Result<()> {
    let n_pos = positions.len();
    debug!("  Sending mask: {} positions, {:.1}MB payload",
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

#[cfg(feature = "http")]
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
                        // Use char boundary to avoid panic on multi-byte UTF-8 (β, α, ₐ, etc.)
                        let keep: String = self.buffer.chars()
                            .rev().take(20).collect::<Vec<_>>()
                            .into_iter().rev().collect();
                        self.buffer = keep;
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

#[cfg(feature = "http")]
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

#[cfg(feature = "http")]
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

#[cfg(feature = "http")]
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
