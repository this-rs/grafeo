//! KV Banks — semantic groupings of nodes for batch load/evict.

use obrain::ObrainDB;
use obrain_common::types::{NodeId, PropertyKey, Value};
use obrain_core::graph::{Direction, lpg::LpgStore};
use graph_schema::{GraphSchema, get_node_name_generic, extract_node_generic};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::Instant;

/// A semantic bank = a group of related nodes that can be loaded/evicted together.
#[derive(Clone)]
pub struct KvBank {
    /// Human-readable name (e.g. "Project: Elun").
    pub name: String,
    /// Root node (the "anchor" of this bank, e.g. the Project node).
    pub root: NodeId,
    /// All node IDs in this bank (root + children).
    pub node_ids: Vec<NodeId>,
    /// Pre-built text blocks for each node (node_id → text).
    pub texts: HashMap<NodeId, String>,
    /// Estimated total tokens.
    pub est_tokens: i32,
}

/// Try to load banks from an obrain cache DB. Returns None if cache is stale or missing.
pub fn load_bank_cache(cache_path: &Path, node_count: usize, edge_count: usize) -> Option<Vec<KvBank>> {
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
pub fn save_bank_cache(cache_path: &Path, banks: &[KvBank], node_count: usize, edge_count: usize) {
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
/// Each top-level node + its hierarchy (BFS up to depth 2) = 1 bank.
/// Token budget per bank prevents explosion on highly connected nodes.
pub fn discover_banks(
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

    const MAX_TOKENS_PER_BANK: i32 = 500;
    const MAX_DEPTH: usize = 2;
    const MAX_CHILDREN_PER_EDGE_TYPE: usize = 5;

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

    let _elapsed = t0.elapsed().as_millis();
    banks
}
