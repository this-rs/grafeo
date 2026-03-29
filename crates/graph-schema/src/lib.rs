//! Graph schema auto-discovery — introspects labels, edges, properties.
//!
//! Schema-agnostic: works on ANY LPG graph.

use obrain_common::types::{NodeId, PropertyKey};
use obrain_core::graph::{Direction, lpg::LpgStore};
use std::collections::{HashMap, HashSet};
use think_filter::truncate;

/// Controls how much property detail is rendered for a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextBudget {
    /// All non-internal properties (for scoring and Alpha tier).
    Full,
    /// Name + top 3 properties (for Beta tier).
    Compact,
    /// Name only (for Gamma tier labels).
    Minimal,
}

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

/// Backward-compatible wrapper: extracts node text with [`TextBudget::Full`].
pub fn extract_node_generic(store: &LpgStore, node_id: NodeId, schema: &GraphSchema) -> (String, String, String) {
    extract_node_with_budget(store, node_id, schema, TextBudget::Full)
}

/// Returns true if `key` is an internal/infrastructure property that should be
/// excluded from user-facing rendering.
fn is_internal_key(key: &str) -> bool {
    matches!(key, "id" | "hash" | "project_id" | "workspace_id")
        || key.starts_with("embedding_")
        || key.starts_with("cc_")
        || key.ends_with("_fingerprint")
        || key.ends_with("_dna")
}

/// Collect non-internal, non-name/desc properties, sorted alphabetically.
/// Each value is truncated to 50 chars. Returns at most `limit` entries
/// (0 = unlimited).
fn collect_extra_props(
    node: &obrain_core::graph::lpg::Node,
    name_key: Option<&str>,
    desc_key: Option<&str>,
    limit: usize,
) -> Vec<(String, String)> {
    let mut props: Vec<(String, String)> = Vec::new();
    for (key, val) in node.properties.iter() {
        let ks: &str = key.as_ref();
        if is_internal_key(ks) { continue; }
        // Skip name and description keys (already rendered separately)
        if let Some(nk) = name_key { if ks == nk { continue; } }
        if let Some(dk) = desc_key { if ks == dk { continue; } }
        // Also skip the common fallback keys that might have been used as name/desc
        if matches!(ks, "name" | "title" | "label" | "description" | "content"
                      | "text" | "message" | "body" | "rationale" | "summary"
                      | "status" | "state" | "phase" | "path" | "display_name") {
            continue;
        }
        // Prefer the raw string value (no quotes); fall back to Display for non-strings.
        let raw = match val.as_str() {
            Some(s) => s.to_string(),
            None => {
                let d = val.to_string();
                if d == "NULL" { continue; }
                d
            }
        };
        if raw.is_empty() { continue; }
        let val_str = truncate(&raw, 50);
        props.push((ks.to_string(), val_str));
    }
    props.sort_by(|a, b| a.0.cmp(&b.0));
    if limit > 0 && props.len() > limit {
        props.truncate(limit);
    }
    props
}

/// Collect a compact summary of outgoing relations for a node.
/// Format: " | → KNOWS: Marc Dupont, Alice Chen | → WORKS_ON: Obrain"
/// Grouped by edge type, max `limit` edges total, target identified by name or first property.
fn collect_outgoing_relations(store: &LpgStore, node_id: NodeId, node_name: &str, limit: usize) -> String {
    let mut by_type: HashMap<String, Vec<String>> = HashMap::new();
    let mut count = 0;

    for (target_id, edge_id) in store.edges_from(node_id, Direction::Outgoing).collect::<Vec<_>>() {
        if count >= limit { break; }
        let etype = store.edge_type(edge_id)
            .map(|s| s.to_string())
            .unwrap_or_else(|| "RELATED".to_string());

        // Skip internal/noisy relation types
        if etype.starts_with("__") || etype == "HAS_PROPERTY" { continue; }

        // Get target node name
        let target_name = if let Some(tnode) = store.get_node(target_id) {
            let mut found = String::new();
            for &k in &["name", "title", "label"] {
                if let Some(v) = tnode.properties.get(&PropertyKey::from(k))
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                {
                    found = truncate(v, 40);
                    break;
                }
            }
            if found.is_empty() {
                // Use first label as fallback
                tnode.labels.first().map(|l| format!("({})", l)).unwrap_or_default()
            } else {
                found
            }
        } else {
            continue;
        };

        // Skip self-references
        if target_name == node_name { continue; }

        by_type.entry(etype).or_default().push(target_name);
        count += 1;
    }

    if by_type.is_empty() {
        return String::new();
    }

    let mut parts: Vec<String> = Vec::new();
    let mut types: Vec<String> = by_type.keys().cloned().collect();
    types.sort();
    for etype in types {
        let targets = by_type.get(&etype).unwrap();
        parts.push(format!("→{}: {}", etype, targets.join(", ")));
    }
    format!(" | {}", parts.join(" | "))
}

/// Extract node text with a specific [`TextBudget`].
pub fn extract_node_with_budget(
    store: &LpgStore,
    node_id: NodeId,
    schema: &GraphSchema,
    budget: TextBudget,
) -> (String, String, String) {
    let empty = (String::new(), String::new(), String::new());
    let node = match store.get_node(node_id) {
        Some(n) => n,
        None => return empty,
    };

    let labels = node.labels.iter().map(|l| l.to_string()).collect::<Vec<_>>().join(",");
    let primary_label = node.labels.first().map(|l| l.to_string()).unwrap_or_default();
    let dp = schema.display_props.get(&primary_label);

    // Name
    let (name, name_key_used) = if let Some(dp) = dp {
        if let Some(ref field) = dp.name_field {
            let val = node.properties.get(&PropertyKey::from(field.as_str()))
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .map(|s| truncate(s, 80))
                .unwrap_or_default();
            (val, Some(field.as_str()))
        } else { (String::new(), None) }
    } else {
        let mut found = None;
        for &k in &["name", "title"] {
            if let Some(v) = node.properties.get(&PropertyKey::from(k))
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
            {
                found = Some((truncate(v, 80), k));
                break;
            }
        }
        match found {
            Some((val, key)) => (val, Some(key)),
            None => (String::new(), None),
        }
    };

    // For Minimal budget, return just [Labels] Name
    if budget == TextBudget::Minimal {
        let summary = name.clone();
        if name.is_empty() {
            return empty;
        }
        let text = format!("[{}] {}", labels, name);
        return (text, labels, summary);
    }

    // Description
    let (desc, desc_key_used) = {
        let (raw_desc, dk) = if let Some(dp) = dp {
            if let Some(ref field) = dp.desc_field {
                let v = node.properties.get(&PropertyKey::from(field.as_str()))
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty() && s != &name)
                    .unwrap_or("");
                (v, Some(field.as_str()))
            } else { ("", None) }
        } else {
            let mut found = None;
            for &k in &["description", "content", "text", "message", "body", "rationale"] {
                if let Some(v) = node.properties.get(&PropertyKey::from(k))
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty() && s != &name)
                {
                    found = Some((v, k));
                    break;
                }
            }
            match found {
                Some((v, k)) => (v, Some(k)),
                None => ("", None),
            }
        };
        if raw_desc.is_empty() {
            (String::new(), dk)
        } else {
            let mut out = String::new();
            for line in raw_desc.lines() {
                let trimmed = line.trim();
                if trimmed.is_empty() || trimmed.starts_with("---") { continue; }
                if !out.is_empty() { out.push(' '); }
                out.push_str(trimmed);
                if out.len() >= 500 { break; }
            }
            (truncate(&out, 500), dk)
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
                if is_internal_key(ks) || ks == "embedding_model" { continue; }
                if !s.is_empty() && s.len() > 2 && s.len() < 500 {
                    let text = format!("[{}] {}: {}", labels, ks, truncate(s, 150));
                    return (text, labels, truncate(s, 60));
                }
            }
        }
        return empty;
    }

    // Collect extra properties for Full/Compact budgets
    let prop_limit = match budget {
        TextBudget::Full => 0,     // unlimited
        TextBudget::Compact => 3,
        TextBudget::Minimal => 0,  // unreachable, handled above
    };
    let extra = collect_extra_props(&node, name_key_used, desc_key_used, prop_limit);
    let props_suffix = if extra.is_empty() {
        String::new()
    } else {
        let parts: Vec<String> = extra.iter()
            .map(|(k, v)| format!("{}: {}", k, v))
            .collect();
        format!(" | {}", parts.join(" | "))
    };

    // ── Outgoing relations summary (Full budget only) ──
    let rel_suffix = if budget == TextBudget::Full {
        collect_outgoing_relations(store, node_id, &name, 5)
    } else {
        String::new()
    };

    let text = if !desc.is_empty() {
        if !status.is_empty() {
            format!("[{}] {} — {} ({}){}{}", labels, name, desc, status, props_suffix, rel_suffix)
        } else {
            format!("[{}] {} — {}{}{}", labels, name, desc, props_suffix, rel_suffix)
        }
    } else if !status.is_empty() {
        format!("[{}] {} ({}){}{}", labels, name, status, props_suffix, rel_suffix)
    } else if !props_suffix.is_empty() {
        format!("[{}] {}{}{}", labels, name, props_suffix, rel_suffix)
    } else if !rel_suffix.is_empty() {
        format!("[{}] {}{}", labels, name, rel_suffix)
    } else {
        format!("[{}] {}", labels, name)
    };

    (text, labels, summary)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use obrain_common::types::Value;

    /// Helper: create a store with a single node that has the given properties.
    /// Returns (store, node_id).
    fn make_store_with_node(label: &str, props: &[(&str, &str)]) -> (LpgStore, NodeId) {
        let store = LpgStore::new().expect("store");
        let nid = store.create_node(&[label]);
        for &(k, v) in props {
            store.set_node_property(nid, k, Value::String(v.into()));
        }
        (store, nid)
    }

    #[test]
    fn full_budget_renders_all_non_internal_props() {
        let (store, nid) = make_store_with_node("Task", &[
            ("name", "Fix login bug"),
            ("description", "Users cannot log in"),
            ("priority", "high"),
            ("assigned_to", "alice"),
            ("due_date", "2026-04-01"),
            ("category", "backend"),
            ("severity", "critical"),
            ("sprint", "S42"),
            // internal keys that must be excluded
            ("id", "12345"),
            ("hash", "abc123"),
            ("cc_version", "3"),
        ]);
        let schema = discover_schema(&store);
        let (text, _labels, _summary) = extract_node_with_budget(&store, nid, &schema, TextBudget::Full);

        // Should contain extra props alphabetically
        assert!(text.contains("assigned_to: alice"), "missing assigned_to: {text}");
        assert!(text.contains("category: backend"), "missing category: {text}");
        assert!(text.contains("due_date: 2026-04-01"), "missing due_date: {text}");
        assert!(text.contains("priority: high"), "missing priority: {text}");
        assert!(text.contains("severity: critical"), "missing severity: {text}");
        assert!(text.contains("sprint: S42"), "missing sprint: {text}");

        // Internal keys must NOT appear
        assert!(!text.contains("id: 12345"), "internal id leaked: {text}");
        assert!(!text.contains("hash: abc123"), "internal hash leaked: {text}");
        assert!(!text.contains("cc_version"), "internal cc_ leaked: {text}");
    }

    #[test]
    fn compact_budget_renders_only_3_props() {
        let (store, nid) = make_store_with_node("Task", &[
            ("name", "Fix login bug"),
            ("description", "Users cannot log in"),
            ("priority", "high"),
            ("assigned_to", "alice"),
            ("due_date", "2026-04-01"),
            ("category", "backend"),
            ("severity", "critical"),
            ("sprint", "S42"),
        ]);
        let schema = discover_schema(&store);
        let (text, _, _) = extract_node_with_budget(&store, nid, &schema, TextBudget::Compact);

        // Count pipes: [Labels] Name — Desc | p1 | p2 | p3
        // The description separator uses " — " not " | "
        // Extra props are separated by " | " — there should be exactly 3 extra
        let after_desc = text.splitn(2, " — ").nth(1).unwrap_or("");
        let pipe_count = after_desc.matches(" | ").count();
        assert_eq!(pipe_count, 3, "compact should have exactly 3 extra props, got {pipe_count}: {text}");

        // First 3 alphabetically: assigned_to, category, due_date
        assert!(text.contains("assigned_to: alice"), "missing 1st: {text}");
        assert!(text.contains("category: backend"), "missing 2nd: {text}");
        assert!(text.contains("due_date: 2026-04-01"), "missing 3rd: {text}");
        // 4th+ should not appear
        assert!(!text.contains("priority:"), "4th prop leaked: {text}");
        assert!(!text.contains("severity:"), "5th prop leaked: {text}");
    }

    #[test]
    fn minimal_budget_renders_only_name() {
        let (store, nid) = make_store_with_node("Task", &[
            ("name", "Fix login bug"),
            ("description", "Users cannot log in"),
            ("priority", "high"),
            ("assigned_to", "alice"),
        ]);
        let schema = discover_schema(&store);
        let (text, labels, _) = extract_node_with_budget(&store, nid, &schema, TextBudget::Minimal);

        assert_eq!(text, format!("[{}] Fix login bug", labels));
        assert!(!text.contains("Users cannot log in"), "desc in minimal: {text}");
        assert!(!text.contains("priority"), "props in minimal: {text}");
    }

    #[test]
    fn internal_keys_are_excluded() {
        let (store, nid) = make_store_with_node("Item", &[
            ("name", "Test item"),
            ("id", "id-001"),
            ("hash", "deadbeef"),
            ("embedding_ada", "vector-data"),
            ("cc_score", "0.9"),
            ("content_fingerprint", "fp123"),
            ("graph_dna", "dna-abc"),
            ("project_id", "proj-1"),
            ("workspace_id", "ws-1"),
            ("visible_prop", "yes"),
        ]);
        let schema = discover_schema(&store);
        let (text, _, _) = extract_node_with_budget(&store, nid, &schema, TextBudget::Full);

        assert!(text.contains("visible_prop: yes"), "visible prop missing: {text}");
        assert!(!text.contains("id-001"), "id leaked: {text}");
        assert!(!text.contains("deadbeef"), "hash leaked: {text}");
        assert!(!text.contains("vector-data"), "embedding_ leaked: {text}");
        assert!(!text.contains("cc_score"), "cc_ leaked: {text}");
        assert!(!text.contains("fp123"), "fingerprint leaked: {text}");
        assert!(!text.contains("dna-abc"), "_dna leaked: {text}");
        assert!(!text.contains("proj-1"), "project_id leaked: {text}");
        assert!(!text.contains("ws-1"), "workspace_id leaked: {text}");
    }

    #[test]
    fn long_property_values_are_truncated_at_50_chars() {
        let long_val = "a".repeat(80);
        let (store, nid) = make_store_with_node("Doc", &[
            ("name", "My Document"),
            ("long_field", &long_val),
        ]);
        let schema = discover_schema(&store);
        let (text, _, _) = extract_node_with_budget(&store, nid, &schema, TextBudget::Full);

        // The truncated value should be at most 50 chars + "..."
        if let Some(idx) = text.find("long_field: ") {
            let after = &text[idx + "long_field: ".len()..];
            let val_end = after.find(" | ").unwrap_or(after.len());
            let rendered_val = &after[..val_end];
            assert!(rendered_val.chars().count() <= 53, // 50 + "..." (3 chars)
                "value not truncated: len={}, val='{rendered_val}'", rendered_val.chars().count());
            assert!(rendered_val.contains("..."), "no truncation marker: {rendered_val}");
        } else {
            panic!("long_field not found in output: {text}");
        }
    }

    #[test]
    fn backward_compat_wrapper_uses_full_budget() {
        let (store, nid) = make_store_with_node("Task", &[
            ("name", "Test"),
            ("visible_prop", "yes"),
        ]);
        let schema = discover_schema(&store);
        let via_wrapper = extract_node_generic(&store, nid, &schema);
        let via_explicit = extract_node_with_budget(&store, nid, &schema, TextBudget::Full);
        assert_eq!(via_wrapper, via_explicit);
    }
}
