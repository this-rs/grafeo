use crate::PersonaDB;
use obrain_common::types::{NodeId, PropertyKey, Value};

// ═══════════════════════════════════════════════════════════════════════════════
// Fact detection — heuristic extraction of persistent facts from user messages
// ═══════════════════════════════════════════════════════════════════════════════

/// Ξ(t) T2: Result of a pattern match — includes the Pattern NodeId for tracking.
pub struct PatternMatch {
    pub pattern_nid: NodeId,
    pub key: String,
    pub value: String,
}

/// Ξ(t) T2: Detect facts from :Pattern nodes in the graph.
/// Each :Pattern has a trigger (prefix match on lowercased input).
/// Returns matches with pattern provenance for EXTRACTS edges and hit_count updates.
pub fn detect_facts_from_graph(pdb: &PersonaDB, msg: &str) -> Vec<PatternMatch> {
    let store = pdb.db.store();
    let lower = msg.to_lowercase();
    let mut matches = Vec::new();

    let patterns = store.nodes_by_label("Pattern");
    for &nid in &patterns {
        let node = match store.get_node(nid) {
            Some(n) => n,
            None => continue,
        };

        // Skip inactive patterns
        let active = node.properties.get(&PropertyKey::from("active"))
            .and_then(|v| if let Value::Bool(b) = v { Some(*b) } else { None })
            .unwrap_or(true);
        if !active { continue; }

        let trigger = match node.properties.get(&PropertyKey::from("trigger")) {
            Some(v) => v.as_str().unwrap_or("").to_string(),
            None => continue,
        };
        let key_template = node.properties.get(&PropertyKey::from("key_template"))
            .and_then(|v| v.as_str()).unwrap_or("memory").to_string();

        if trigger.is_empty() { continue; }

        // Prefix match
        if let Some(rest) = lower.strip_prefix(trigger.as_str()) {
            let value = rest.trim().trim_end_matches(|c: char| c == '.' || c == '!' || c == '?');
            if !value.is_empty() && value.len() < 200 {
                matches.push(PatternMatch {
                    pattern_nid: nid,
                    key: key_template.clone(),
                    value: value.to_string(),
                });
            }
        }

        // Infix match (fallback for "... trigger ..." patterns)
        if matches.is_empty() {
            if let Some(pos) = lower.find(trigger.as_str()) {
                let rest = &lower[pos + trigger.len()..];
                let value: String = rest.chars()
                    .take_while(|c| *c != '.' && *c != ',' && *c != '!' && *c != '?')
                    .collect();
                let value = value.trim().to_string();
                if !value.is_empty() && value.len() < 200 {
                    matches.push(PatternMatch {
                        pattern_nid: nid,
                        key: key_template,
                        value,
                    });
                }
            }
        }
    }

    // Update hit_count for matched patterns
    for m in &matches {
        let hit_count = store.get_node(m.pattern_nid)
            .and_then(|n| n.properties.get(&PropertyKey::from("hit_count"))
                .and_then(|v| if let Value::Int64(n) = v { Some(*n) } else { None }))
            .unwrap_or(0);
        pdb.db.set_node_property(m.pattern_nid, "hit_count", Value::Int64(hit_count + 1));
    }

    matches
}

/// Legacy fallback: detect facts via hardcoded patterns (used when no PersonaDB).
pub fn detect_facts(msg: &str) -> Vec<(String, String)> {
    let lower = msg.to_lowercase();
    let mut facts = Vec::new();

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

    for prefix in &[
        "retiens que ", "rappelle-toi que ", "rappelle toi que ",
        "n'oublie pas que ", "remember that ", "don't forget that ",
    ] {
        if let Some(rest) = lower.strip_prefix(prefix) {
            let value = rest.trim().trim_end_matches(|c: char| c == '.' || c == '!');
            if !value.is_empty() && value.len() < 200 {
                facts.push(("memory".to_string(), value.to_string()));
            }
        }
    }

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

    facts
}
