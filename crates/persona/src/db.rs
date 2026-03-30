use anyhow::{Context, Result};
use chrono::Utc;
use obrain::ObrainDB;
use obrain_common::types::{NodeId, PropertyKey, Value};
use obrain_core::graph::Direction;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

use crate::bm25::{MessageHit, MessageIndex};
use kv_registry::{ColdHit, ColdSearch};

// ═══════════════════════════════════════════════════════════════════════════════
// Persona DB — stores conversations, facts and identity in obrain graph
// ═══════════════════════════════════════════════════════════════════════════════

pub struct PersonaDB {
    pub db: ObrainDB,
    pub current_conv_id: NodeId,
    /// BM25 index over all :Message nodes (cross-conversation search).
    /// Behind RefCell to allow indexing from &self methods (single-threaded).
    msg_index: RefCell<MessageIndex>,
}

impl PersonaDB {
    /// Open or create conversation DB at given path.
    pub fn open(path: &str) -> Result<Self> {
        let db =
            ObrainDB::open(path).context(format!("Failed to open conversation DB at {path}"))?;

        // Find or create the current conversation
        let conv_id = {
            let store = db.store();
            let convs = store.nodes_by_label("Conversation");
            // Use the most recent one, or create a new one
            let latest = convs
                .iter()
                .filter_map(|&nid| {
                    let node = store.get_node(nid)?;
                    let ts = node
                        .properties
                        .get(&PropertyKey::from("created_at"))
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

        let persona = Self {
            db,
            current_conv_id: conv_id,
            msg_index: RefCell::new(MessageIndex::new()),
        };
        persona.rebuild_message_index();
        Ok(persona)
    }

    /// Build (or rebuild) the BM25 index from all :Message nodes across all conversations.
    fn rebuild_message_index(&self) {
        let store = self.db.store();
        let convs = store.nodes_by_label("Conversation");
        let mut count = 0u32;
        let mut idx = self.msg_index.borrow_mut();

        for &conv_id in &convs {
            for (target, _eid) in store
                .edges_from(conv_id, Direction::Outgoing)
                .collect::<Vec<_>>()
            {
                if let Some(node) = store.get_node(target) {
                    let is_msg = node.labels.iter().any(|l| {
                        let s: &str = l.as_ref();
                        s == "Message"
                    });
                    if !is_msg {
                        continue;
                    }

                    let role = node
                        .properties
                        .get(&PropertyKey::from("role"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("user");
                    let content = node
                        .properties
                        .get(&PropertyKey::from("content"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    if content.is_empty() {
                        continue;
                    }

                    idx.add(target, conv_id, role, content);
                    count += 1;
                }
            }
        }

        // Also index :Self nodes (Phase 4 introspection)
        let self_nodes = store.nodes_by_label("Self");
        for &nid in &self_nodes {
            if let Some(node) = store.get_node(nid) {
                let value = node
                    .properties
                    .get(&PropertyKey::from("value"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                if !value.is_empty() {
                    // Use a synthetic conv_id (the node itself) and role "self"
                    idx.add(nid, nid, "self", value);
                    count += 1;
                }
            }
        }

        if count > 0 {
            kv_registry::kv_debug!(
                "[PersonaDB] BM25 index built: {} messages + {} :Self nodes across {} conversations",
                count - self_nodes.len() as u32,
                self_nodes.len(),
                convs.len()
            );
        }
    }

    pub fn create_conv_node(db: &ObrainDB, title: &str) -> NodeId {
        db.create_node_with_props(
            &["Conversation"],
            [
                ("title", Value::String(title.to_string().into())),
                ("created_at", Value::String(Utc::now().to_rfc3339().into())),
            ],
        )
    }

    /// Start a new conversation within the same DB.
    pub fn new_conversation(&mut self, title: &str) {
        self.current_conv_id = Self::create_conv_node(&self.db, title);
    }

    /// Add a message to the current conversation. Returns the message NodeId.
    /// Also indexes the message content in the BM25 index for cross-conversation search.
    pub fn add_message(&self, role: &str, content: &str) -> NodeId {
        let store = self.db.store();
        let msg_count = store
            .edges_from(self.current_conv_id, Direction::Outgoing)
            .count();

        let msg_id = self.db.create_node_with_props(
            &["Message"],
            [
                ("role", Value::String(role.to_string().into())),
                ("content", Value::String(content.to_string().into())),
                ("timestamp", Value::String(Utc::now().to_rfc3339().into())),
                ("order", Value::Int64(msg_count as i64)),
            ],
        );
        self.db.create_edge(self.current_conv_id, msg_id, "HAS_MSG");

        // Incremental BM25 indexing
        if !content.is_empty() {
            self.msg_index
                .borrow_mut()
                .add(msg_id, self.current_conv_id, role, content);
        }

        msg_id
    }

    /// Search all messages across all conversations using BM25 scoring.
    /// Returns up to `limit` hits sorted by relevance score descending.
    pub fn search_messages(&self, query: &str, limit: usize) -> Vec<MessageHit> {
        self.msg_index.borrow().search(query, limit)
    }

    /// Number of messages in the BM25 index (for diagnostics).
    pub fn indexed_message_count(&self) -> usize {
        self.msg_index.borrow().len()
    }

    /// Find the adjacent message (by order ±1) in the same conversation.
    /// Returns (role, content) if found.
    pub fn adjacent_message(
        &self,
        conv_id: NodeId,
        msg_node_id: NodeId,
        direction: i64,
    ) -> Option<(NodeId, String, String)> {
        let store = self.db.store();
        // Get the order of the source message
        let src_order = store
            .get_node(msg_node_id)?
            .properties
            .get(&PropertyKey::from("order"))
            .and_then(|v| {
                if let Value::Int64(n) = v {
                    Some(*n)
                } else {
                    None
                }
            })?;
        let target_order = src_order + direction;

        // Search through conversation messages for the target order
        for (target, _eid) in store
            .edges_from(conv_id, Direction::Outgoing)
            .collect::<Vec<_>>()
        {
            if let Some(node) = store.get_node(target) {
                let is_msg = node.labels.iter().any(|l| {
                    let s: &str = l.as_ref();
                    s == "Message"
                });
                if !is_msg {
                    continue;
                }
                let order = node
                    .properties
                    .get(&PropertyKey::from("order"))
                    .and_then(|v| {
                        if let Value::Int64(n) = v {
                            Some(*n)
                        } else {
                            None
                        }
                    })
                    .unwrap_or(-1);
                if order == target_order {
                    let role = node
                        .properties
                        .get(&PropertyKey::from("role"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("user")
                        .to_string();
                    let content = node
                        .properties
                        .get(&PropertyKey::from("content"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    return Some((target, role, content));
                }
            }
        }
        None
    }

    /// Link a reply message to the message it's replying to.
    pub fn link_reply(&self, reply_id: NodeId, parent_id: NodeId) {
        self.db.create_edge(reply_id, parent_id, "REPLIES_TO");
    }

    /// Store reward score on a :Message node.
    /// Called when the reward for turn N is computed at turn N+1.
    /// This allows BM25 search to boost validated corrections (reward+) and demote bad answers (reward-).
    pub fn set_message_reward(&self, msg_id: NodeId, reward: f64) {
        self.db
            .set_node_property(msg_id, "reward", Value::Float64(reward));
    }

    /// Read the reward stored on a :Message node (if any).
    pub fn get_message_reward(&self, msg_id: NodeId) -> Option<f64> {
        let store = self.db.store();
        store.get_node(msg_id).and_then(|n| {
            n.properties
                .get(&PropertyKey::from("reward"))
                .and_then(|v| {
                    if let Value::Float64(f) = v {
                        Some(*f)
                    } else {
                        None
                    }
                })
        })
    }

    /// Get recent messages from the current conversation (for context injection).
    pub fn recent_messages(&self, limit: usize) -> Vec<(String, String)> {
        let store = self.db.store();
        let mut msgs: Vec<(i64, String, String)> = Vec::new();

        for (target, _eid) in store
            .edges_from(self.current_conv_id, Direction::Outgoing)
            .collect::<Vec<_>>()
        {
            if let Some(node) = store.get_node(target) {
                let has_msg_label = node.labels.iter().any(|l| {
                    let s: &str = l.as_ref();
                    s == "Message"
                });
                if !has_msg_label {
                    continue;
                }

                let role = node
                    .properties
                    .get(&PropertyKey::from("role"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("user")
                    .to_string();
                let content = node
                    .properties
                    .get(&PropertyKey::from("content"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let order = node
                    .properties
                    .get(&PropertyKey::from("order"))
                    .and_then(|v| {
                        if let Value::Int64(n) = v {
                            Some(*n)
                        } else {
                            None
                        }
                    })
                    .unwrap_or(0);
                msgs.push((order, role, content));
            }
        }

        msgs.sort_by_key(|(order, _, _)| *order);
        msgs.into_iter()
            .rev()
            .take(limit)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .map(|(_, role, content)| (role, content))
            .collect()
    }

    /// Get all user prompts across all conversations, oldest first (for readline history).
    pub fn user_history(&self) -> Vec<String> {
        let store = self.db.store();
        let mut prompts: Vec<(String, i64, String)> = Vec::new(); // (timestamp, order, content)

        for &conv_id in &store.nodes_by_label("Conversation") {
            for (target, _eid) in store
                .edges_from(conv_id, Direction::Outgoing)
                .collect::<Vec<_>>()
            {
                if let Some(node) = store.get_node(target) {
                    let is_msg = node.labels.iter().any(|l| {
                        let s: &str = l.as_ref();
                        s == "Message"
                    });
                    if !is_msg {
                        continue;
                    }
                    let role = node
                        .properties
                        .get(&PropertyKey::from("role"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    if role != "user" {
                        continue;
                    }
                    let content = node
                        .properties
                        .get(&PropertyKey::from("content"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    if content.is_empty() {
                        continue;
                    }
                    let ts = node
                        .properties
                        .get(&PropertyKey::from("timestamp"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let order = node
                        .properties
                        .get(&PropertyKey::from("order"))
                        .and_then(|v| {
                            if let Value::Int64(n) = v {
                                Some(*n)
                            } else {
                                None
                            }
                        })
                        .unwrap_or(0);
                    prompts.push((ts, order, content));
                }
            }
        }

        // Sort by timestamp then order (oldest first → added to history first)
        prompts.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        prompts.into_iter().map(|(_, _, content)| content).collect()
    }

    /// List all conversations in the DB.
    pub fn list_conversations(&self) -> Vec<(NodeId, String, String, usize)> {
        let store = self.db.store();
        let convs = store.nodes_by_label("Conversation");
        let mut result = Vec::new();

        for &nid in &convs {
            if let Some(node) = store.get_node(nid) {
                let title = node
                    .properties
                    .get(&PropertyKey::from("title"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("(untitled)")
                    .to_string();
                let created = node
                    .properties
                    .get(&PropertyKey::from("created_at"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("?")
                    .to_string();
                let msg_count = store.edges_from(nid, Direction::Outgoing).count();
                result.push((nid, title, created, msg_count));
            }
        }

        result.sort_by(|a, b| b.2.cmp(&a.2)); // newest first
        result
    }

    /// Switch to an existing conversation by NodeId.
    pub fn switch_to(&mut self, conv_id: NodeId) -> bool {
        let store = self.db.store();
        if store.get_node(conv_id).is_some() {
            self.current_conv_id = conv_id;
            true
        } else {
            false
        }
    }

    /// Get current conversation title.
    pub fn current_title(&self) -> String {
        let store = self.db.store();
        store
            .get_node(self.current_conv_id)
            .and_then(|n| {
                n.properties
                    .get(&PropertyKey::from("title"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
            })
            .unwrap_or_else(|| "(untitled)".to_string())
    }

    // ── Fact management (Ξ(t) T1 — enriched FactGraph) ────────────

    /// Infer fact_type from the key name.
    pub fn infer_fact_type(key: &str) -> &'static str {
        match key {
            "name" | "identity" | "nickname" | "surname" => "identity",
            "language" | "preference" | "style" | "tone" => "preference",
            "memory" | "remember" | "episodic" => "episodic",
            _ => "rule",
        }
    }

    /// Store a persistent fact with enriched properties (T1).
    /// Deactivates existing fact with same key, creates CONTRADICTS edge.
    /// If conv_turn_id is provided, creates EXTRACTED_FROM edge.
    pub fn add_fact(
        &self,
        key: &str,
        value: &str,
        source_turn: u32,
        conv_turn_id: Option<NodeId>,
    ) -> NodeId {
        let store = self.db.store();
        let mut old_fact_id: Option<NodeId> = None;

        // Deactivate existing fact with same key
        for &nid in &store.nodes_by_label("Fact") {
            if let Some(node) = store.get_node(nid) {
                let k = node
                    .properties
                    .get(&PropertyKey::from("key"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let active = node
                    .properties
                    .get(&PropertyKey::from("active"))
                    .and_then(|v| {
                        if let Value::Bool(b) = v {
                            Some(*b)
                        } else {
                            None
                        }
                    })
                    .unwrap_or(true);
                if k == key && active {
                    self.db.set_node_property(nid, "active", Value::Bool(false));
                    old_fact_id = Some(nid);
                }
            }
        }

        let fact_type = Self::infer_fact_type(key);
        // Ξ(t) T5: Estimate token cost naïvely: "- key : value\n" ≈ (len+6)/4 tokens
        let token_cost = ((key.len() + value.len() + 6) / 4).max(1) as i64;
        let fact_id = self.db.create_node_with_props(
            &["Fact"],
            [
                ("key", Value::String(key.to_string().into())),
                ("value", Value::String(value.to_string().into())),
                ("fact_type", Value::String(fact_type.to_string().into())),
                ("confidence", Value::Float64(0.8)),
                ("energy", Value::Float64(1.0)),
                ("source_turn", Value::Int64(source_turn as i64)),
                ("created_at", Value::String(Utc::now().to_rfc3339().into())),
                ("active", Value::Bool(true)),
                ("token_cost", Value::Int64(token_cost)),
                ("utility", Value::Float64(0.5)),
                ("cost_efficiency", Value::Float64(0.5 / token_cost as f64)),
                ("activation_count", Value::Int64(0)),
            ],
        );

        // CONTRADICTS edge to old fact
        if let Some(old_id) = old_fact_id {
            self.db.create_edge(fact_id, old_id, "CONTRADICTS");
        }

        // EXTRACTED_FROM edge to the conversation turn that produced it
        if let Some(ct_id) = conv_turn_id {
            self.db.create_edge(fact_id, ct_id, "EXTRACTED_FROM");
        }

        fact_id
    }

    // ── :Memory nodes (neural persistence — replaces pattern-based :Fact) ──

    /// Store a user message as a :Memory node (PersistNet decided to persist).
    ///
    /// Returns the NodeId of the created :Memory.
    /// The memory stores the raw message text — no key/value extraction needed.
    pub fn add_memory(
        &self,
        text: &str,
        persist_score: f32,
        source_turn: u32,
        conv_turn_id: Option<NodeId>,
    ) -> NodeId {
        let token_cost = ((text.len() + 4) / 4).max(1) as i64;
        let mem_id = self.db.create_node_with_props(
            &["Memory"],
            [
                ("text", Value::String(text.to_string().into())),
                ("persist_score", Value::Float64(persist_score as f64)),
                ("energy", Value::Float64(1.0)),
                ("source_turn", Value::Int64(source_turn as i64)),
                ("created_at", Value::String(Utc::now().to_rfc3339().into())),
                ("active", Value::Bool(true)),
                ("token_cost", Value::Int64(token_cost)),
                ("utility", Value::Float64(0.5)),
                ("times_used", Value::Int64(0)),
                ("last_useful_turn", Value::Int64(source_turn as i64)),
            ],
        );

        // Link to the conversation turn that produced it
        if let Some(ct_id) = conv_turn_id {
            self.db.create_edge(mem_id, ct_id, "EXTRACTED_FROM");
        }

        mem_id
    }

    /// Get all active :Memory texts, sorted by energy descending.
    pub fn active_memories(&self) -> Vec<(NodeId, String, f64)> {
        let store = self.db.store();
        let mut memories = Vec::new();
        for &nid in &store.nodes_by_label("Memory") {
            if let Some(node) = store.get_node(nid) {
                let active = node
                    .properties
                    .get(&PropertyKey::from("active"))
                    .and_then(|v| {
                        if let Value::Bool(b) = v {
                            Some(*b)
                        } else {
                            None
                        }
                    })
                    .unwrap_or(true);
                if !active {
                    continue;
                }
                let text = node
                    .properties
                    .get(&PropertyKey::from("text"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let energy = node
                    .properties
                    .get(&PropertyKey::from("energy"))
                    .and_then(|v| {
                        if let Value::Float64(f) = v {
                            Some(*f)
                        } else {
                            None
                        }
                    })
                    .unwrap_or(1.0);
                memories.push((nid, text, energy));
            }
        }
        memories.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        memories
    }

    /// Get active :Memory NodeIds (for USED_IN tracking, same as facts).
    pub fn active_memory_ids(&self) -> Vec<NodeId> {
        let store = self.db.store();
        let mut ids = Vec::new();
        for &nid in &store.nodes_by_label("Memory") {
            if let Some(node) = store.get_node(nid) {
                let active = node
                    .properties
                    .get(&PropertyKey::from("active"))
                    .and_then(|v| {
                        if let Value::Bool(b) = v {
                            Some(*b)
                        } else {
                            None
                        }
                    })
                    .unwrap_or(true);
                if active {
                    ids.push(nid);
                }
            }
        }
        ids
    }

    /// Mark a memory as "useful" this turn (bump times_used + last_useful_turn).
    pub fn mark_memory_useful(&self, mem_id: NodeId, current_turn: u32) {
        let store = self.db.store();
        if let Some(node) = store.get_node(mem_id) {
            let times = node
                .properties
                .get(&PropertyKey::from("times_used"))
                .and_then(|v| {
                    if let Value::Int64(n) = v {
                        Some(*n)
                    } else {
                        None
                    }
                })
                .unwrap_or(0);
            self.db
                .set_node_property(mem_id, "times_used", Value::Int64(times + 1));
            self.db.set_node_property(
                mem_id,
                "last_useful_turn",
                Value::Int64(current_turn as i64),
            );
        }
    }

    /// Audit stale memories: returns (NodeId, text, source_turn, last_useful_turn)
    /// for memories that haven't been useful for `stale_threshold` turns.
    pub fn stale_memories(
        &self,
        current_turn: u32,
        stale_threshold: u32,
    ) -> Vec<(NodeId, String, u32, u32)> {
        let store = self.db.store();
        let mut stale = Vec::new();
        for &nid in &store.nodes_by_label("Memory") {
            if let Some(node) = store.get_node(nid) {
                let active = node
                    .properties
                    .get(&PropertyKey::from("active"))
                    .and_then(|v| {
                        if let Value::Bool(b) = v {
                            Some(*b)
                        } else {
                            None
                        }
                    })
                    .unwrap_or(true);
                if !active {
                    continue;
                }

                let source_turn = node
                    .properties
                    .get(&PropertyKey::from("source_turn"))
                    .and_then(|v| {
                        if let Value::Int64(n) = v {
                            Some(*n as u32)
                        } else {
                            None
                        }
                    })
                    .unwrap_or(0);
                let last_useful = node
                    .properties
                    .get(&PropertyKey::from("last_useful_turn"))
                    .and_then(|v| {
                        if let Value::Int64(n) = v {
                            Some(*n as u32)
                        } else {
                            None
                        }
                    })
                    .unwrap_or(source_turn);
                let text = node
                    .properties
                    .get(&PropertyKey::from("text"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                if current_turn > last_useful + stale_threshold {
                    stale.push((nid, text, source_turn, last_useful));
                }
            }
        }
        stale
    }

    /// Deactivate a memory (soft delete).
    pub fn deactivate_memory(&self, mem_id: NodeId) {
        self.db
            .set_node_property(mem_id, "active", Value::Bool(false));
    }

    /// Get all active facts as (key, value) pairs.
    pub fn active_facts(&self) -> Vec<(String, String)> {
        let store = self.db.store();
        let mut facts = Vec::new();
        for &nid in &store.nodes_by_label("Fact") {
            if let Some(node) = store.get_node(nid) {
                let active = node
                    .properties
                    .get(&PropertyKey::from("active"))
                    .and_then(|v| {
                        if let Value::Bool(b) = v {
                            Some(*b)
                        } else {
                            None
                        }
                    })
                    .unwrap_or(true);
                if !active {
                    continue;
                }
                let key = node
                    .properties
                    .get(&PropertyKey::from("key"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let value = node
                    .properties
                    .get(&PropertyKey::from("value"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                facts.push((key, value));
            }
        }
        facts
    }

    /// Get active fact NodeIds (for USED_IN tracking).
    pub fn active_fact_ids(&self) -> Vec<NodeId> {
        let store = self.db.store();
        let mut ids = Vec::new();
        for &nid in &store.nodes_by_label("Fact") {
            if let Some(node) = store.get_node(nid) {
                let active = node
                    .properties
                    .get(&PropertyKey::from("active"))
                    .and_then(|v| {
                        if let Value::Bool(b) = v {
                            Some(*b)
                        } else {
                            None
                        }
                    })
                    .unwrap_or(true);
                if active {
                    ids.push(nid);
                }
            }
        }
        ids
    }

    /// List all facts with details for /facts command.
    pub fn list_facts(&self) -> Vec<(NodeId, String, String, i64, bool, f64, f64, String)> {
        let store = self.db.store();
        let mut facts = Vec::new();
        for &nid in &store.nodes_by_label("Fact") {
            if let Some(node) = store.get_node(nid) {
                let key = node
                    .properties
                    .get(&PropertyKey::from("key"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let value = node
                    .properties
                    .get(&PropertyKey::from("value"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let turn = node
                    .properties
                    .get(&PropertyKey::from("source_turn"))
                    .and_then(|v| {
                        if let Value::Int64(n) = v {
                            Some(*n)
                        } else {
                            None
                        }
                    })
                    .unwrap_or(0);
                let active = node
                    .properties
                    .get(&PropertyKey::from("active"))
                    .and_then(|v| {
                        if let Value::Bool(b) = v {
                            Some(*b)
                        } else {
                            None
                        }
                    })
                    .unwrap_or(true);
                let confidence = node
                    .properties
                    .get(&PropertyKey::from("confidence"))
                    .and_then(|v| {
                        if let Value::Float64(f) = v {
                            Some(*f)
                        } else {
                            None
                        }
                    })
                    .unwrap_or(0.8);
                let energy = node
                    .properties
                    .get(&PropertyKey::from("energy"))
                    .and_then(|v| {
                        if let Value::Float64(f) = v {
                            Some(*f)
                        } else {
                            None
                        }
                    })
                    .unwrap_or(1.0);
                let fact_type = node
                    .properties
                    .get(&PropertyKey::from("fact_type"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("rule")
                    .to_string();
                facts.push((nid, key, value, turn, active, confidence, energy, fact_type));
            }
        }
        facts
    }

    /// Deactivate a fact by key.
    pub fn forget_fact(&self, key: &str) -> bool {
        let store = self.db.store();
        let mut found = false;
        for &nid in &store.nodes_by_label("Fact") {
            if let Some(node) = store.get_node(nid) {
                let k = node
                    .properties
                    .get(&PropertyKey::from("key"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let active = node
                    .properties
                    .get(&PropertyKey::from("active"))
                    .and_then(|v| {
                        if let Value::Bool(b) = v {
                            Some(*b)
                        } else {
                            None
                        }
                    })
                    .unwrap_or(true);
                if k == key && active {
                    self.db.set_node_property(nid, "active", Value::Bool(false));
                    found = true;
                }
            }
        }
        found
    }

    // ── ConvTurn management (Ξ(t) T1.2) ─────────────────────────

    /// Create a :ConvTurn node for the current exchange.
    pub fn create_conv_turn(&self, query: &str, response_text: &str, turn_number: u32) -> NodeId {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut qh = DefaultHasher::new();
        query.hash(&mut qh);
        let query_hash = qh.finish();

        let mut rh = DefaultHasher::new();
        response_text.hash(&mut rh);
        let response_hash = rh.finish();

        // Truncate query for storage (keep first 500 chars for multi-line pastes)
        let query_trunc: String = query.chars().take(500).collect();

        self.db.create_node_with_props(
            &["ConvTurn"],
            [
                ("query_hash", Value::Int64(query_hash as i64)),
                ("response_hash", Value::Int64(response_hash as i64)),
                ("query_text", Value::String(query_trunc.into())),
                ("reward", Value::Float64(0.0)),
                ("timestamp", Value::String(Utc::now().to_rfc3339().into())),
                ("turn_number", Value::Int64(turn_number as i64)),
            ],
        )
    }

    /// Link two consecutive ConvTurns with TEMPORAL_NEXT.
    pub fn link_temporal(&self, from_turn: NodeId, to_turn: NodeId) {
        self.db.create_edge(from_turn, to_turn, "TEMPORAL_NEXT");
    }

    /// Mark facts as USED_IN a ConvTurn (they were in the context sent to LLM).
    pub fn mark_facts_used_in(&self, fact_ids: &[NodeId], conv_turn_id: NodeId) {
        for &fid in fact_ids {
            self.db.create_edge(fid, conv_turn_id, "USED_IN");
        }
    }

    /// Link a ConvTurn to a node from the main graph (MENTIONS).
    pub fn link_mentions(&self, conv_turn_id: NodeId, mentioned_nid: NodeId) {
        self.db.create_edge(conv_turn_id, mentioned_nid, "MENTIONS");
    }

    /// Create REINFORCES edges between facts co-used in a high-reward turn.
    /// Used by T3 (RewardDetector) when reward > 0.5.
    pub fn create_reinforces(&self, fact_ids: &[NodeId], reward: f64) {
        if reward <= 0.5 || fact_ids.len() < 2 {
            return;
        }
        let store = self.db.store();
        for i in 0..fact_ids.len() {
            for j in (i + 1)..fact_ids.len() {
                // Check if edge already exists
                let already = store
                    .edges_from(fact_ids[i], Direction::Outgoing)
                    .any(|(target, _)| target == fact_ids[j]);
                if !already {
                    self.db.create_edge(fact_ids[i], fact_ids[j], "REINFORCES");
                }
            }
        }
    }

    // ── Pattern management (Ξ(t) T1.3) ──────────────────────────

    /// Seed default extraction patterns, adding any missing ones.
    /// Unlike seed_formulas_if_empty, this is additive: it checks each trigger
    /// individually so new patterns get added to existing DBs.
    pub fn seed_default_patterns(&self) {
        let store = self.db.store();
        let existing = store.nodes_by_label("Pattern");

        // Collect existing triggers for dedup
        let existing_triggers: std::collections::HashSet<String> = existing
            .iter()
            .filter_map(|&nid| {
                store.get_node(nid).and_then(|n| {
                    n.properties
                        .get(&PropertyKey::from("trigger"))
                        .and_then(|v| v.as_str().map(|s| s.to_string()))
                })
            })
            .collect();

        let defaults = [
            // Identity patterns — 1st person (most common in conversation!)
            ("je m'appelle ", "name", "identity"),
            ("je m appelle ", "name", "identity"),
            ("mon nom est ", "name", "identity"),
            ("mon nom c'est ", "name", "identity"),
            ("mon prénom est ", "name", "identity"),
            ("mon prénom c'est ", "name", "identity"),
            ("my name is ", "name", "identity"),
            ("i'm ", "identity", "identity"),
            ("i am ", "identity", "identity"),
            ("je suis ", "identity", "identity"),
            // Identity patterns — 2nd person (commands)
            ("ton nom est ", "name", "identity"),
            ("tu t'appelles ", "name", "identity"),
            ("tu es ", "identity", "identity"),
            ("appelle-toi ", "name", "identity"),
            ("your name is ", "name", "identity"),
            ("you are ", "identity", "identity"),
            ("call yourself ", "name", "identity"),
            // Location patterns
            ("j'habite à ", "city", "preference"),
            ("j'habite a ", "city", "preference"),
            ("j'habite en ", "country", "preference"),
            ("je vis à ", "city", "preference"),
            ("je vis a ", "city", "preference"),
            ("je vis en ", "country", "preference"),
            ("i live in ", "city", "preference"),
            // Preference patterns
            ("ma couleur préférée est ", "preference_color", "preference"),
            (
                "ma couleur préférée c'est ",
                "preference_color",
                "preference",
            ),
            ("ma couleur preferee est ", "preference_color", "preference"),
            (
                "ma couleur preferee c'est ",
                "preference_color",
                "preference",
            ),
            ("my favorite color is ", "preference_color", "preference"),
            ("j'aime ", "preference", "preference"),
            ("j'adore ", "preference", "preference"),
            ("je préfère ", "preference", "preference"),
            ("je deteste ", "dislike", "preference"),
            ("je déteste ", "dislike", "preference"),
            ("i like ", "preference", "preference"),
            ("i love ", "preference", "preference"),
            ("i hate ", "dislike", "preference"),
            // Work/occupation
            ("je travaille ", "work", "identity"),
            ("je suis étudiant ", "occupation", "identity"),
            ("je suis étudiante ", "occupation", "identity"),
            ("i work ", "work", "identity"),
            // Memory patterns
            ("retiens que ", "memory", "episodic"),
            ("rappelle-toi que ", "memory", "episodic"),
            ("rappelle toi que ", "memory", "episodic"),
            ("n'oublie pas que ", "memory", "episodic"),
            ("remember that ", "memory", "episodic"),
            ("don't forget that ", "memory", "episodic"),
            // Language/preference patterns
            ("réponds toujours en ", "language", "preference"),
            ("reponds toujours en ", "language", "preference"),
            ("parle toujours en ", "language", "preference"),
            ("always respond in ", "language", "preference"),
            ("always answer in ", "language", "preference"),
        ];

        let mut added = 0u32;
        for (trigger, key_template, fact_type) in defaults {
            if existing_triggers.contains(trigger) {
                continue;
            }
            self.db.create_node_with_props(
                &["Pattern"],
                [
                    ("trigger", Value::String(trigger.to_string().into())),
                    (
                        "key_template",
                        Value::String(key_template.to_string().into()),
                    ),
                    ("fact_type", Value::String(fact_type.to_string().into())),
                    ("hit_count", Value::Int64(0)),
                    ("avg_reward", Value::Float64(0.0)),
                    ("auto_generated", Value::Bool(false)),
                    ("active", Value::Bool(true)),
                ],
            );
            added += 1;
        }
        if added > 0 {
            eprintln!("  [Persona] Seeded {} new patterns (total: {})", added, existing.len() + added as usize);
        }
    }

    // ── Migration (Ξ(t) T1.5) ───────────────────────────────────

    /// Migrate old :Fact nodes to enriched format (fact_type + cost tracking).
    pub fn migrate_facts(&self) {
        let store = self.db.store();
        for &nid in &store.nodes_by_label("Fact") {
            if let Some(node) = store.get_node(nid) {
                let key = node
                    .properties
                    .get(&PropertyKey::from("key"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("");

                // Migration 1: fact_type (T1.5)
                if !node
                    .properties
                    .contains_key(&PropertyKey::from("fact_type"))
                {
                    let fact_type = Self::infer_fact_type(key);
                    self.db.set_node_property(
                        nid,
                        "fact_type",
                        Value::String(fact_type.to_string().into()),
                    );
                    self.db
                        .set_node_property(nid, "confidence", Value::Float64(0.8));
                    self.db
                        .set_node_property(nid, "energy", Value::Float64(1.0));
                }

                // Migration 2: cost tracking (T5)
                if !node
                    .properties
                    .contains_key(&PropertyKey::from("token_cost"))
                {
                    let value = node
                        .properties
                        .get(&PropertyKey::from("value"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let token_cost = ((key.len() + value.len() + 6) / 4).max(1) as i64;
                    self.db
                        .set_node_property(nid, "token_cost", Value::Int64(token_cost));
                    self.db
                        .set_node_property(nid, "utility", Value::Float64(0.5));
                    self.db.set_node_property(
                        nid,
                        "cost_efficiency",
                        Value::Float64(0.5 / token_cost as f64),
                    );
                    self.db
                        .set_node_property(nid, "activation_count", Value::Int64(0));
                }
            }
        }
    }

    // ── Ξ(t) T5: Pattern Auto-Generation ────────────────────────

    /// Extract n-grams (2 and 3 words) from text, lowercased, stopwords removed.
    pub fn extract_ngrams(text: &str, max_n: usize) -> Vec<String> {
        const STOPWORDS: &[&str] = &[
            "le", "la", "les", "de", "du", "des", "un", "une", "est", "a", "et", "en", "au", "aux",
            "que", "qui", "the", "is", "of", "a", "an", "and", "in", "to", "it", "for", "on",
            "with", "at", "by", "i", "je", "tu", "il", "elle", "nous", "vous", "ils", "me", "te",
            "se", "ce", "ça",
        ];
        let lower = text.to_lowercase();
        let words: Vec<String> = lower
            .split_whitespace()
            .filter(|w| !STOPWORDS.contains(w) && w.len() > 1)
            .map(|s| s.to_string())
            .collect();

        let mut ngrams = Vec::new();
        for n in 2..=max_n.min(3) {
            if words.len() >= n {
                for window in words.windows(n) {
                    ngrams.push(window.join(" "));
                }
            }
        }
        ngrams
    }

    /// Try to auto-generate new :Pattern nodes from frequent n-grams in ConvTurn queries.
    /// Call every ~5 turns. Max 2 new patterns per call.
    pub fn try_generate_patterns(&self) -> u32 {
        let store = self.db.store();
        let facts: Vec<NodeId> = store
            .nodes_by_label("Fact")
            .iter()
            .filter(|&&nid| {
                store
                    .get_node(nid)
                    .and_then(|n| {
                        n.properties
                            .get(&PropertyKey::from("confidence"))
                            .and_then(|v| {
                                if let Value::Float64(f) = v {
                                    Some(*f > 0.7)
                                } else {
                                    None
                                }
                            })
                    })
                    .unwrap_or(false)
            })
            .copied()
            .collect();

        if facts.len() < 3 {
            return 0;
        }

        // Collect query texts from ConvTurns that produced high-confidence facts
        let mut query_texts: Vec<String> = Vec::new();
        for &fid in &facts {
            for (target, _) in store
                .edges_from(fid, Direction::Outgoing)
                .collect::<Vec<_>>()
            {
                if let Some(node) = store.get_node(target) {
                    let is_conv = node.labels.iter().any(|l| {
                        let s: &str = l.as_ref();
                        s == "ConvTurn"
                    });
                    if is_conv {
                        if let Some(qt) = node
                            .properties
                            .get(&PropertyKey::from("query_text"))
                            .and_then(|v| v.as_str())
                        {
                            query_texts.push(qt.to_string());
                        }
                    }
                }
            }
        }

        if query_texts.len() < 3 {
            return 0;
        }

        // Count n-grams across all query texts
        let mut ngram_counts: HashMap<String, u32> = HashMap::new();
        for qt in &query_texts {
            let ngrams = Self::extract_ngrams(qt, 3);
            for ng in ngrams {
                *ngram_counts.entry(ng).or_insert(0) += 1;
            }
        }

        // Get existing pattern triggers for dedup
        let existing_triggers: HashSet<String> = store
            .nodes_by_label("Pattern")
            .iter()
            .filter_map(|&nid| {
                store.get_node(nid).and_then(|n| {
                    n.properties
                        .get(&PropertyKey::from("trigger"))
                        .and_then(|v| v.as_str().map(|s| s.to_string()))
                })
            })
            .collect();

        // Find frequent n-grams not already covered
        let mut generated = 0u32;
        let mut candidates: Vec<(String, u32)> = ngram_counts
            .into_iter()
            .filter(|(ng, count)| {
                *count >= 3 && !existing_triggers.iter().any(|t| t.contains(ng.as_str()))
            })
            .collect();
        candidates.sort_by(|a, b| b.1.cmp(&a.1));

        for (ngram, _count) in candidates.iter().take(2) {
            let trigger = format!("{} ", ngram); // trailing space for prefix match
            self.db.create_node_with_props(
                &["Pattern"],
                [
                    ("trigger", Value::String(trigger.clone().into())),
                    ("key_template", Value::String("memory".to_string().into())),
                    ("fact_type", Value::String("episodic".to_string().into())),
                    ("hit_count", Value::Int64(0)),
                    ("avg_reward", Value::Float64(0.0)),
                    ("auto_generated", Value::Bool(true)),
                    ("active", Value::Bool(true)),
                ],
            );
            generated += 1;
        }

        // Promote patterns: auto_generated=true, hit_count >= 5, avg_reward > 0.4
        for &nid in &store.nodes_by_label("Pattern") {
            if let Some(node) = store.get_node(nid) {
                let auto = node
                    .properties
                    .get(&PropertyKey::from("auto_generated"))
                    .and_then(|v| {
                        if let Value::Bool(b) = v {
                            Some(*b)
                        } else {
                            None
                        }
                    })
                    .unwrap_or(false);
                if !auto {
                    continue;
                }
                let hits = node
                    .properties
                    .get(&PropertyKey::from("hit_count"))
                    .and_then(|v| {
                        if let Value::Int64(n) = v {
                            Some(*n)
                        } else {
                            None
                        }
                    })
                    .unwrap_or(0);
                let avg_r = node
                    .properties
                    .get(&PropertyKey::from("avg_reward"))
                    .and_then(|v| {
                        if let Value::Float64(f) = v {
                            Some(*f)
                        } else {
                            None
                        }
                    })
                    .unwrap_or(0.0);
                if hits >= 5 && avg_r > 0.4 {
                    self.db
                        .set_node_property(nid, "auto_generated", Value::Bool(false));
                }
            }
        }

        generated
    }

    // ── Ξ(t) T5: Garbage Collection ─────────────────────────────

    /// Garbage-collect dead patterns, low-energy facts, and old ConvTurns.
    /// Call every ~20 turns.
    pub fn gc_persona_graph(&self, current_turn: u32) {
        let store = self.db.store();

        // (1) Deactivate patterns with hit_count > 5 and avg_reward < 0.2
        for &nid in &store.nodes_by_label("Pattern") {
            if let Some(node) = store.get_node(nid) {
                let active = node
                    .properties
                    .get(&PropertyKey::from("active"))
                    .and_then(|v| {
                        if let Value::Bool(b) = v {
                            Some(*b)
                        } else {
                            None
                        }
                    })
                    .unwrap_or(true);
                if !active {
                    continue;
                }
                let hits = node
                    .properties
                    .get(&PropertyKey::from("hit_count"))
                    .and_then(|v| {
                        if let Value::Int64(n) = v {
                            Some(*n)
                        } else {
                            None
                        }
                    })
                    .unwrap_or(0);
                let avg_r = node
                    .properties
                    .get(&PropertyKey::from("avg_reward"))
                    .and_then(|v| {
                        if let Value::Float64(f) = v {
                            Some(*f)
                        } else {
                            None
                        }
                    })
                    .unwrap_or(0.0);
                if hits > 5 && avg_r < 0.2 {
                    self.db.set_node_property(nid, "active", Value::Bool(false));
                }
            }
        }

        // (2) Deactivate facts with energy < 0.1 and no recent USED_IN
        for &nid in &store.nodes_by_label("Fact") {
            if let Some(node) = store.get_node(nid) {
                let active = node
                    .properties
                    .get(&PropertyKey::from("active"))
                    .and_then(|v| {
                        if let Value::Bool(b) = v {
                            Some(*b)
                        } else {
                            None
                        }
                    })
                    .unwrap_or(true);
                if !active {
                    continue;
                }
                let energy = node
                    .properties
                    .get(&PropertyKey::from("energy"))
                    .and_then(|v| {
                        if let Value::Float64(f) = v {
                            Some(*f)
                        } else {
                            None
                        }
                    })
                    .unwrap_or(1.0);
                if energy >= 0.1 {
                    continue;
                }

                // Check if used recently (within last 50 turns)
                let source_turn = node
                    .properties
                    .get(&PropertyKey::from("source_turn"))
                    .and_then(|v| {
                        if let Value::Int64(n) = v {
                            Some(*n as u32)
                        } else {
                            None
                        }
                    })
                    .unwrap_or(0);
                if current_turn.saturating_sub(source_turn) > 50 {
                    self.db.set_node_property(nid, "active", Value::Bool(false));
                }
            }
        }

        // (3) Clean old ConvTurns (> 1000 turns ago): we don't delete them
        // but they'll naturally be excluded from BFS subgraph by hop limit
        // (Mark as old — GNN will naturally ignore via BFS hop limit)
    }

    // ── Ξ(t) T5: Stats ──────────────────────────────────────────

    /// Get Ξ(t) system metrics for /stats command.
    pub fn xi_stats(&self) -> XiStats {
        let store = self.db.store();

        let all_facts = store.nodes_by_label("Fact");
        let active_facts: Vec<_> = all_facts
            .iter()
            .filter(|&&nid| {
                store
                    .get_node(nid)
                    .and_then(|n| {
                        n.properties
                            .get(&PropertyKey::from("active"))
                            .and_then(|v| {
                                if let Value::Bool(b) = v {
                                    Some(*b)
                                } else {
                                    None
                                }
                            })
                    })
                    .unwrap_or(true)
            })
            .collect();
        let avg_energy: f64 = if active_facts.is_empty() {
            0.0
        } else {
            active_facts
                .iter()
                .map(|&&nid| {
                    store
                        .get_node(nid)
                        .and_then(|n| {
                            n.properties
                                .get(&PropertyKey::from("energy"))
                                .and_then(|v| {
                                    if let Value::Float64(f) = v {
                                        Some(*f)
                                    } else {
                                        None
                                    }
                                })
                        })
                        .unwrap_or(1.0)
                })
                .sum::<f64>()
                / active_facts.len() as f64
        };
        let avg_confidence: f64 = if active_facts.is_empty() {
            0.0
        } else {
            active_facts
                .iter()
                .map(|&&nid| {
                    store
                        .get_node(nid)
                        .and_then(|n| {
                            n.properties
                                .get(&PropertyKey::from("confidence"))
                                .and_then(|v| {
                                    if let Value::Float64(f) = v {
                                        Some(*f)
                                    } else {
                                        None
                                    }
                                })
                        })
                        .unwrap_or(0.8)
                })
                .sum::<f64>()
                / active_facts.len() as f64
        };

        let all_patterns = store.nodes_by_label("Pattern");
        let active_patterns: Vec<_> = all_patterns
            .iter()
            .filter(|&&nid| {
                store
                    .get_node(nid)
                    .and_then(|n| {
                        n.properties
                            .get(&PropertyKey::from("active"))
                            .and_then(|v| {
                                if let Value::Bool(b) = v {
                                    Some(*b)
                                } else {
                                    None
                                }
                            })
                    })
                    .unwrap_or(true)
            })
            .collect();
        let auto_patterns = all_patterns
            .iter()
            .filter(|&&nid| {
                store
                    .get_node(nid)
                    .and_then(|n| {
                        n.properties
                            .get(&PropertyKey::from("auto_generated"))
                            .and_then(|v| {
                                if let Value::Bool(b) = v {
                                    Some(*b)
                                } else {
                                    None
                                }
                            })
                    })
                    .unwrap_or(false)
            })
            .count();

        let conv_turns = store.nodes_by_label("ConvTurn");
        let rewards: Vec<f64> = conv_turns
            .iter()
            .filter_map(|&nid| {
                store.get_node(nid).and_then(|n| {
                    n.properties
                        .get(&PropertyKey::from("reward"))
                        .and_then(|v| {
                            if let Value::Float64(f) = v {
                                Some(*f)
                            } else {
                                None
                            }
                        })
                })
            })
            .collect();
        let avg_reward = if rewards.is_empty() {
            0.0
        } else {
            // Average of last 20 rewards
            let recent: Vec<_> = rewards.iter().rev().take(20).collect();
            recent.iter().copied().sum::<f64>() / recent.len() as f64
        };

        // Mask quality: average of last 20 mask_reward values
        let mask_rewards: Vec<f64> = conv_turns
            .iter()
            .filter_map(|&nid| {
                store.get_node(nid).and_then(|n| {
                    n.properties
                        .get(&PropertyKey::from("mask_reward"))
                        .and_then(|v| {
                            if let Value::Float64(f) = v {
                                Some(*f)
                            } else {
                                None
                            }
                        })
                })
            })
            .collect();
        let avg_mask_reward = if mask_rewards.is_empty() {
            0.0
        } else {
            let recent: Vec<_> = mask_rewards.iter().rev().take(20).collect();
            recent.iter().copied().sum::<f64>() / recent.len() as f64
        };

        XiStats {
            facts_active: active_facts.len(),
            facts_total: all_facts.len(),
            avg_energy,
            avg_confidence,
            patterns_active: active_patterns.len(),
            patterns_total: all_patterns.len(),
            patterns_auto: auto_patterns,
            conv_turns: conv_turns.len(),
            avg_reward_recent: avg_reward,
            reward_tokens: store.nodes_by_label("RewardToken").len(),
            avg_mask_reward,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// XiStats
// ═══════════════════════════════════════════════════════════════════════════════

/// Ξ(t) system metrics.
pub struct XiStats {
    pub facts_active: usize,
    pub facts_total: usize,
    pub avg_energy: f64,
    pub avg_confidence: f64,
    pub patterns_active: usize,
    pub patterns_total: usize,
    pub patterns_auto: usize,
    pub conv_turns: usize,
    pub avg_reward_recent: f64,
    pub reward_tokens: usize,
    /// Average mask_reward over last 20 turns (topology mask quality signal)
    pub avg_mask_reward: f64,
}

// ── ColdSearch trait implementation for PersonaDB ──────────────────────

/// Adjust a BM25 score by the reward stored on the :Message node.
/// Uses exponential scaling: reward=+0.8 → ×2.0, reward=-0.5 → ×0.5
/// This gives enough dynamic range to override BM25 text matching
/// when the quality signal is strong.
fn reward_adjusted_score(base_score: f64, reward: Option<f64>) -> f64 {
    match reward {
        // exp(reward) gives: -0.5→×0.61, -0.3→×0.74, 0→×1.0, +0.5→×1.65, +0.7→×2.01, +0.9→×2.46
        Some(r) => base_score * (r.clamp(-1.0, 1.0)).exp(),
        None => base_score,
    }
}

/// Maximum :Self nodes allowed in search results (anti-narcissisme).
const MAX_SELF_IN_RESULTS: usize = 2;

impl ColdSearch for PersonaDB {
    fn search(&self, query: &str, limit: usize) -> Vec<ColdHit> {
        let store = self.db.store();
        let mut results: Vec<ColdHit> = self
            .search_messages(query, limit * 2) // fetch more, re-rank
            .into_iter()
            .map(|hit| {
                let reward = self.get_message_reward(hit.node_id);
                let adjusted_score = reward_adjusted_score(hit.score, reward);
                let conv_title = store.get_node(hit.conv_id).and_then(|n| {
                    n.properties
                        .get(&PropertyKey::from("title"))
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string())
                });
                ColdHit {
                    role: hit.role,
                    content: hit.content,
                    conv_id: hit.conv_id,
                    conv_title,
                    score: adjusted_score,
                }
            })
            .collect();
        // Re-sort by adjusted score and truncate
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        // Cap :Self nodes to prevent narcissism (Phase 4)
        let mut self_count = 0usize;
        results.retain(|hit| {
            if hit.role == "self" {
                self_count += 1;
                self_count <= MAX_SELF_IN_RESULTS
            } else {
                true
            }
        });
        results.truncate(limit);
        results
    }

    fn search_pairs(&self, query: &str, limit: usize) -> Vec<(ColdHit, Option<ColdHit>)> {
        let store = self.db.store();
        let hits = self.search_messages(query, limit * 2);
        let mut seen_nodes: std::collections::HashSet<NodeId> = std::collections::HashSet::new();

        let mut pairs: Vec<(ColdHit, Option<ColdHit>)> = hits
            .into_iter()
            .filter_map(|hit| {
                if seen_nodes.contains(&hit.node_id) {
                    return None;
                }
                seen_nodes.insert(hit.node_id);

                let reward = self.get_message_reward(hit.node_id);
                let adjusted_score = reward_adjusted_score(hit.score, reward);

                let conv_title = store.get_node(hit.conv_id).and_then(|n| {
                    n.properties
                        .get(&PropertyKey::from("title"))
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string())
                });

                let primary = ColdHit {
                    role: hit.role.clone(),
                    content: hit.content.clone(),
                    conv_id: hit.conv_id,
                    conv_title: conv_title.clone(),
                    score: adjusted_score,
                };

                // Find adjacent message to form a Q/A pair:
                // If hit is "user", look for next (assistant reply)
                // If hit is "assistant", look for prev (user question)
                let direction = if hit.role == "user" { 1 } else { -1 };
                let adjacent = self
                    .adjacent_message(hit.conv_id, hit.node_id, direction)
                    .and_then(|(adj_nid, adj_role, adj_content)| {
                        if seen_nodes.contains(&adj_nid) {
                            return None;
                        }
                        seen_nodes.insert(adj_nid);
                        let adj_reward = self.get_message_reward(adj_nid);
                        let adj_score = reward_adjusted_score(adjusted_score * 0.8, adj_reward);
                        Some(ColdHit {
                            role: adj_role,
                            content: adj_content,
                            conv_id: hit.conv_id,
                            conv_title: conv_title.clone(),
                            score: adj_score,
                        })
                    });

                Some((primary, adjacent))
            })
            .collect();
        // Re-sort by primary adjusted score and truncate
        pairs.sort_by(|a, b| {
            b.0.score
                .partial_cmp(&a.0.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        pairs.truncate(limit);
        pairs
    }
}

// ── Standalone: AttnFormula CRUD on impl PersonaDB ──

impl PersonaDB {
    // ── :AttnFormula nodes (Phase 4 — Evolutionary Attention Formulas) ──

    /// Add a new :AttnFormula node to the graph.
    ///
    /// `dsl_json` is the serde_json serialization of an `AttnOp` (from retrieval::attn_dsl).
    /// `parent_id` creates a `MUTATED_FROM` edge for evolutionary lineage.
    pub fn add_formula(
        &self,
        name: &str,
        dsl_json: &str,
        context_affinity: &[&str],
        generation: i64,
        parent_id: Option<NodeId>,
    ) -> NodeId {
        let affinity_str = context_affinity.join(",");
        let fid = self.db.create_node_with_props(
            &["AttnFormula"],
            [
                ("name", Value::String(name.to_string().into())),
                ("dsl_json", Value::String(dsl_json.to_string().into())),
                ("energy", Value::Float64(1.0)),
                ("avg_reward", Value::Float64(0.0)),
                ("activation_count", Value::Int64(0)),
                ("generation", Value::Int64(generation)),
                ("context_affinity", Value::String(affinity_str.into())),
                ("active", Value::Bool(true)),
                ("created_at", Value::String(Utc::now().to_rfc3339().into())),
            ],
        );

        if let Some(parent) = parent_id {
            self.db.create_edge(fid, parent, "MUTATED_FROM");
        }

        fid
    }

    /// List all :AttnFormula nodes (active and inactive).
    pub fn list_formulas(&self) -> Vec<crate::formulas::AttnFormulaNode> {
        let store = self.db.store();
        let mut formulas = Vec::new();
        for &nid in &store.nodes_by_label("AttnFormula") {
            if let Some(f) = self.read_formula_node(nid) {
                formulas.push(f);
            }
        }
        formulas
    }

    /// Get a single :AttnFormula by NodeId.
    pub fn get_formula(&self, formula_id: NodeId) -> Option<crate::formulas::AttnFormulaNode> {
        self.read_formula_node(formula_id)
    }

    /// Update avg_reward and activation_count for a formula after a generation step.
    pub fn update_formula_reward(&self, formula_id: NodeId, reward: f64) {
        let store = self.db.store();
        if let Some(node) = store.get_node(formula_id) {
            let old_avg = node
                .properties
                .get(&PropertyKey::from("avg_reward"))
                .and_then(|v| {
                    if let Value::Float64(f) = v {
                        Some(*f)
                    } else {
                        None
                    }
                })
                .unwrap_or(0.0);
            let old_count = node
                .properties
                .get(&PropertyKey::from("activation_count"))
                .and_then(|v| {
                    if let Value::Int64(i) = v {
                        Some(*i)
                    } else {
                        None
                    }
                })
                .unwrap_or(0);
            let old_energy = node
                .properties
                .get(&PropertyKey::from("energy"))
                .and_then(|v| {
                    if let Value::Float64(f) = v {
                        Some(*f)
                    } else {
                        None
                    }
                })
                .unwrap_or(1.0);

            let new_count = old_count + 1;
            // Exponential moving average with α=0.3
            let alpha = 0.3;
            let new_avg = old_avg * (1.0 - alpha) + reward * alpha;
            // Energy update: +0.1 on positive reward, -0.2 on negative, clamped [0, 2]
            let energy_delta = if reward > 0.0 { 0.1 } else { -0.2 };
            let new_energy = (old_energy + energy_delta).clamp(0.0, 2.0);

            self.db
                .set_node_property(formula_id, "avg_reward", Value::Float64(new_avg));
            self.db
                .set_node_property(formula_id, "activation_count", Value::Int64(new_count));
            self.db
                .set_node_property(formula_id, "energy", Value::Float64(new_energy));
        }
    }

    /// Deactivate a formula (mark as dead in the population).
    pub fn deactivate_formula(&self, formula_id: NodeId) {
        self.db
            .set_node_property(formula_id, "active", Value::Bool(false));
    }

    /// Seed the 6 initial formulas if no :AttnFormula nodes exist.
    /// Called from open() or explicitly by the user.
    /// Returns the number of formulas created (0 if already seeded).
    pub fn seed_formulas_if_empty(&self) -> usize {
        let store = self.db.store();
        let existing = store.nodes_by_label("AttnFormula").len();
        if existing > 0 {
            return 0;
        }
        let _ = store;

        let seeds = crate::formulas::seed_formulas();
        for (name, dsl_json, affinity) in &seeds {
            self.add_formula(name, dsl_json, affinity, 0, None);
        }
        seeds.len()
    }

    /// Internal: read an AttnFormula node into the struct.
    fn read_formula_node(&self, nid: NodeId) -> Option<crate::formulas::AttnFormulaNode> {
        let store = self.db.store();
        let node = store.get_node(nid)?;

        let name = node
            .properties
            .get(&PropertyKey::from("name"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let dsl_json = node
            .properties
            .get(&PropertyKey::from("dsl_json"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let energy = node
            .properties
            .get(&PropertyKey::from("energy"))
            .and_then(|v| {
                if let Value::Float64(f) = v {
                    Some(*f)
                } else {
                    None
                }
            })
            .unwrap_or(1.0);
        let avg_reward = node
            .properties
            .get(&PropertyKey::from("avg_reward"))
            .and_then(|v| {
                if let Value::Float64(f) = v {
                    Some(*f)
                } else {
                    None
                }
            })
            .unwrap_or(0.0);
        let activation_count = node
            .properties
            .get(&PropertyKey::from("activation_count"))
            .and_then(|v| {
                if let Value::Int64(i) = v {
                    Some(*i)
                } else {
                    None
                }
            })
            .unwrap_or(0);
        let generation = node
            .properties
            .get(&PropertyKey::from("generation"))
            .and_then(|v| {
                if let Value::Int64(i) = v {
                    Some(*i)
                } else {
                    None
                }
            })
            .unwrap_or(0);
        let affinity_str = node
            .properties
            .get(&PropertyKey::from("context_affinity"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let context_affinity: Vec<String> = if affinity_str.is_empty() {
            vec![]
        } else {
            affinity_str.split(',').map(|s| s.to_string()).collect()
        };
        let active = node
            .properties
            .get(&PropertyKey::from("active"))
            .and_then(|v| {
                if let Value::Bool(b) = v {
                    Some(*b)
                } else {
                    None
                }
            })
            .unwrap_or(true);

        // Check for MUTATED_FROM edge
        let parent_id = store
            .edges_from(nid, Direction::Outgoing)
            .find(|(_, eid)| {
                store
                    .get_edge(*eid)
                    .map(|e| {
                        let l: &str = e.edge_type.as_ref();
                        l == "MUTATED_FROM"
                    })
                    .unwrap_or(false)
            })
            .map(|(target, _)| target);

        Some(crate::formulas::AttnFormulaNode {
            id: nid,
            name,
            dsl_json,
            energy,
            avg_reward,
            activation_count,
            generation,
            context_affinity,
            active,
            parent_id,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// :Self nodes — introspective metrics (Phase 4)
// ═══════════════════════════════════════════════════════════════════════════════

/// Aggregated self-metrics for introspection.
/// Updated after each turn's reward loop.
#[derive(Debug, Clone)]
pub struct SelfMetrics {
    /// Running average reward (from RewardDetector).
    pub reward_avg: f64,
    /// Running average mask quality (factual_signal from reward).
    pub mask_reward_avg: f64,
    /// Top-5 HeadRouter heads (head_index, α value) — empty if no router.
    pub head_router_top5: Vec<(usize, f32)>,
    /// Name of the currently active attention formula.
    pub formula_active_name: String,
    /// Human-readable explanation of the active formula.
    pub formula_explanation: String,
    /// Number of active GNN-scored facts.
    pub gnn_facts_active: usize,
    /// Recent reward trend: "improving", "stable", or "declining".
    pub learning_trend: String,
}

impl PersonaDB {
    /// Create or update :Self nodes from aggregated metrics.
    ///
    /// Each metric becomes a :Self node with `key` as unique identifier.
    /// Upsert: if a :Self node with matching key exists, update it; otherwise create.
    pub fn upsert_self_metrics(&self, metrics: &SelfMetrics) {
        let entries: Vec<(&str, String, String)> = vec![
            (
                "reward_avg",
                format!("Mon reward moyen récent est {:.2}", metrics.reward_avg),
                format!("{:.4}", metrics.reward_avg),
            ),
            (
                "mask_reward_avg",
                format!(
                    "La qualité de mon masque d'attention est {:.2}",
                    metrics.mask_reward_avg
                ),
                format!("{:.4}", metrics.mask_reward_avg),
            ),
            (
                "head_router_top5",
                if metrics.head_router_top5.is_empty() {
                    "Pas de routage par tête actif".to_string()
                } else {
                    let top: Vec<String> = metrics
                        .head_router_top5
                        .iter()
                        .map(|(h, a)| format!("head{}={:.2}", h, a))
                        .collect();
                    format!("Mes têtes d'attention les plus actives : {}", top.join(", "))
                },
                metrics
                    .head_router_top5
                    .iter()
                    .map(|(h, a)| format!("{}:{:.3}", h, a))
                    .collect::<Vec<_>>()
                    .join(","),
            ),
            (
                "formula_active",
                format!(
                    "Ma formule d'attention active est '{}' : {}",
                    metrics.formula_active_name, metrics.formula_explanation
                ),
                metrics.formula_active_name.clone(),
            ),
            (
                "gnn_state",
                format!(
                    "Mon GNN surveille {} faits actifs",
                    metrics.gnn_facts_active
                ),
                format!("{}", metrics.gnn_facts_active),
            ),
            (
                "learning_trajectory",
                format!("Ma trajectoire d'apprentissage est : {}", metrics.learning_trend),
                metrics.learning_trend.clone(),
            ),
        ];

        let store = self.db.store();
        let now = Utc::now().to_rfc3339();

        for (key, value, value_raw) in entries {
            // Find existing :Self node with this key
            let existing = store
                .nodes_by_label("Self")
                .into_iter()
                .find(|nid| {
                    store
                        .get_node(*nid)
                        .and_then(|n| {
                            n.properties
                                .get(&PropertyKey::from("key"))
                                .and_then(|v| {
                                    if let Value::String(s) = v {
                                        let s_ref: &str = s.as_ref();
                                    Some(s_ref == key)
                                    } else {
                                        None
                                    }
                                })
                        })
                        .unwrap_or(false)
                });

            if let Some(nid) = existing {
                // Update existing
                self.db
                    .set_node_property(nid, "value", Value::String(value.clone().into()));
                self.db
                    .set_node_property(nid, "value_raw", Value::String(value_raw.into()));
                self.db
                    .set_node_property(nid, "updated_at", Value::String(now.clone().into()));
                // Update BM25 index (re-add with new content)
                self.msg_index
                    .borrow_mut()
                    .add(nid, nid, "self", &value);
            } else {
                // Create new :Self node
                let nid = self.db.create_node_with_props(
                    &["Self"],
                    [
                        ("key", Value::String(key.to_string().into())),
                        ("value", Value::String(value.clone().into())),
                        ("value_raw", Value::String(value_raw.into())),
                        ("updated_at", Value::String(now.clone().into())),
                    ],
                );
                // Add to BM25 index
                self.msg_index
                    .borrow_mut()
                    .add(nid, nid, "self", &value);
            }
        }
    }

    /// Save self-embedding projector weights to a :SelfEmbedWeights node.
    /// Uses the flat f32 vec from SelfEmbeddingProjector::save_weights().
    pub fn save_self_embed_weights(&self, weights: &[f32]) {
        let store = self.db.store();

        // Remove old weights node
        for nid in store.nodes_by_label("SelfEmbedWeights") {
            self.db.delete_node(nid);
        }

        // Encode as little-endian bytes → base64
        let bytes: Vec<u8> = weights.iter().flat_map(|f| f.to_le_bytes()).collect();
        let b64 = base64_encode_simple(&bytes);

        self.db.create_node_with_props(
            &["SelfEmbedWeights"],
            [
                ("data", Value::String(b64.into())),
                ("n_floats", Value::Int64(weights.len() as i64)),
                (
                    "updated_at",
                    Value::String(Utc::now().to_rfc3339().into()),
                ),
            ],
        );
    }

    /// Load self-embedding projector weights from :SelfEmbedWeights node.
    /// Returns the flat f32 vec for SelfEmbeddingProjector::load_weights().
    pub fn load_self_embed_weights(&self) -> Option<Vec<f32>> {
        let store = self.db.store();
        let nodes = store.nodes_by_label("SelfEmbedWeights");
        let nid = *nodes.first()?;
        let node = store.get_node(nid)?;

        let b64 = match node.properties.get(&PropertyKey::from("data"))? {
            Value::String(s) => s.to_string(),
            _ => return None,
        };

        let bytes = base64_decode_simple(&b64)?;
        if bytes.len() % 4 != 0 {
            return None;
        }
        Some(
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        )
    }

    /// List all :Self nodes as (key, value) pairs.
    pub fn list_self_metrics(&self) -> Vec<(String, String)> {
        let store = self.db.store();
        store
            .nodes_by_label("Self")
            .into_iter()
            .filter_map(|nid| {
                let node = store.get_node(nid)?;
                let key = match node.properties.get(&PropertyKey::from("key"))? {
                    Value::String(s) => s.to_string(),
                    _ => return None,
                };
                let value = match node.properties.get(&PropertyKey::from("value"))? {
                    Value::String(s) => s.to_string(),
                    _ => return None,
                };
                Some((key, value))
            })
            .collect()
    }

    /// Save HeadRouter self_alpha weights to a :SelfAlphaWeights node.
    pub fn save_head_router_self_alpha(&self, weights: &[f32]) {
        let store = self.db.store();
        for nid in store.nodes_by_label("SelfAlphaWeights") {
            self.db.delete_node(nid);
        }
        let bytes: Vec<u8> = weights.iter().flat_map(|f| f.to_le_bytes()).collect();
        let b64 = base64_encode_simple(&bytes);
        self.db.create_node_with_props(
            &["SelfAlphaWeights"],
            [
                ("data", Value::String(b64.into())),
                ("n_floats", Value::Int64(weights.len() as i64)),
                (
                    "updated_at",
                    Value::String(Utc::now().to_rfc3339().into()),
                ),
            ],
        );
    }

    /// Load HeadRouter self_alpha weights from :SelfAlphaWeights node.
    pub fn load_head_router_self_alpha(&self) -> Option<Vec<f32>> {
        let store = self.db.store();
        let nodes = store.nodes_by_label("SelfAlphaWeights");
        let nid = *nodes.first()?;
        let node = store.get_node(nid)?;
        let b64 = match node.properties.get(&PropertyKey::from("data"))? {
            Value::String(s) => s.to_string(),
            _ => return None,
        };
        let bytes = base64_decode_simple(&b64)?;
        if bytes.len() % 4 != 0 {
            return None;
        }
        Some(
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        )
    }

    /// Read current :Self nodes back as a SelfMetrics struct.
    ///
    /// Returns a default SelfMetrics if no :Self nodes exist yet.
    pub fn current_self_metrics(&self) -> SelfMetrics {
        let kvs = self.list_self_metrics();
        let get = |key: &str| -> String {
            kvs.iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v.clone())
                .unwrap_or_default()
        };
        SelfMetrics {
            reward_avg: get("reward_avg").parse().unwrap_or(0.0),
            mask_reward_avg: get("mask_reward_avg").parse().unwrap_or(0.0),
            head_router_top5: Vec::new(), // not reconstructed from text
            formula_active_name: get("formula_active"),
            formula_explanation: String::new(),
            gnn_facts_active: get("gnn_state").parse().unwrap_or(0),
            learning_trend: get("learning_trajectory"),
        }
    }
}

// ── base64 helpers (no external dependency) ─────────────────────────────

const B64_CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn base64_encode_simple(data: &[u8]) -> String {
    let mut result = String::with_capacity((data.len() + 2) / 3 * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;

        result.push(B64_CHARS[((triple >> 18) & 0x3F) as usize] as char);
        result.push(B64_CHARS[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            result.push(B64_CHARS[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
        if chunk.len() > 2 {
            result.push(B64_CHARS[(triple & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
    }
    result
}

fn base64_decode_simple(s: &str) -> Option<Vec<u8>> {
    let s = s.trim_end_matches('=');
    let mut result = Vec::with_capacity(s.len() * 3 / 4);
    let mut buf: u32 = 0;
    let mut bits: u32 = 0;

    for c in s.chars() {
        let val = match c {
            'A'..='Z' => c as u32 - 'A' as u32,
            'a'..='z' => c as u32 - 'a' as u32 + 26,
            '0'..='9' => c as u32 - '0' as u32 + 52,
            '+' => 62,
            '/' => 63,
            _ => continue,
        };
        buf = (buf << 6) | val;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            result.push((buf >> bits) as u8);
            buf &= (1 << bits) - 1;
        }
    }

    Some(result)
}
