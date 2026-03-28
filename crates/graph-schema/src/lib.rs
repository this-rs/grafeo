//! Graph schema auto-discovery — introspects labels, edges, properties.
//!
//! Schema-agnostic: works on ANY LPG graph.

use obrain_common::types::{NodeId, PropertyKey};
use obrain_core::graph::{Direction, lpg::LpgStore};
use std::collections::{HashMap, HashSet};
use think_filter::truncate;

/// Display properties discovered for a label.
#[derive(Debug, Clone)]
pub struct DisplayProps {
    /// Primary name field (e.g. "name", "title")
    pub name_field: Option<String>,
    /// Description field (e.g. "description", "content", "text")
    pub desc_field: Option<String>,
    /// Status field (e.g. "status", "state")
    pub status_field: Option<String>,
}

/// Info about a label discovered in the graph.
#[derive(Debug, Clone)]
pub struct LabelInfo {
    pub label: String,
    pub count: usize,
    pub avg_degree: f64,
    pub importance: f64,
    pub is_noise: bool,
}

/// Auto-discovered graph schema.
pub struct GraphSchema {
    /// Labels sorted by importance (highest first).
    pub labels: Vec<LabelInfo>,
    /// Parent → Vec<(edge_type, child_label)>.
    pub parent_child: HashMap<String, Vec<(String, String)>>,
    /// Display properties per label.
    pub display_props: HashMap<String, DisplayProps>,
    /// Labels classified as noise (high count, low connectivity, no name field).
    pub noise_labels: HashSet<String>,
    /// Structural labels (non-noise).
    pub structural_labels: HashSet<String>,
}

/// Discover the graph schema by introspecting labels, edges, and properties.
pub fn discover_schema(store: &LpgStore) -> GraphSchema {
    let all_labels = store.all_labels();
    let total_nodes = store.node_count().max(1);

    // S1.1: Label counts
    let mut label_infos: Vec<(String, usize)> = Vec::new();
    for label in &all_labels {
        let count = store.nodes_by_label(label).len();
        if count > 0 {
            label_infos.push((label.clone(), count));
        }
    }

    // S1.2: Sample properties per label
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

    // S1.3: Edge types between labels → hierarchy
    let mut edge_patterns: HashMap<(String, String, String), usize> = HashMap::new();

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
    for children in parent_child.values_mut() {
        children.sort();
        children.dedup();
    }

    // S1.4: Importance = (1/log(count)) × avg_degree
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
            is_noise: false,
        });
    }

    // S1.5: Classify noise vs structural
    let mut noise_labels: HashSet<String> = HashSet::new();
    let mut structural_labels: HashSet<String> = HashSet::new();

    for info in &mut labels_with_importance {
        let frac = info.count as f64 / total_nodes as f64;
        let has_name = display_props.get(&info.label)
            .map_or(false, |d| d.name_field.is_some());

        if frac > 0.05 && (info.avg_degree < 2.0 || !has_name) {
            info.is_noise = true;
            noise_labels.insert(info.label.clone());
        } else {
            structural_labels.insert(info.label.clone());
        }
    }

    labels_with_importance.sort_by(|a, b| b.importance.partial_cmp(&a.importance).unwrap_or(std::cmp::Ordering::Equal));

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

/// Normalized Levenshtein similarity (1.0 = identical, 0.0 = completely different).
pub fn levenshtein_norm(a: &str, b: &str) -> f64 {
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
pub fn fuzzy_match_label(term: &str, schema: &GraphSchema) -> Option<(String, f64)> {
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
            // Prefix match
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

pub fn get_node_name_generic(store: &LpgStore, node_id: NodeId, props: Option<&DisplayProps>) -> String {
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

pub fn extract_node_generic(store: &LpgStore, node_id: NodeId, schema: &GraphSchema) -> (String, String, String) {
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

    // Fallback for nodes with no name/desc
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
