use tokio::sync::{mpsc, oneshot};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use anyhow::Result;
use graph_schema::GraphSchema;
use kv_registry::{KvNodeRegistry, KvBank, ConvFragments};
use obrain_common::types::NodeId;
use obrain_core::graph::lpg::LpgStore;
use retrieval::{Engine, GenerationControl, OutputMode, is_meta_query, query_with_registry};
use think_filter::strip_think_tags;
use persona::{PersonaDB, detect_facts_from_graph, fact_gnn::FactGNN};

/// Result of a chat completion generation
pub struct GenerateResult {
    /// The full raw response (may contain <think> tags)
    pub raw_response: String,
    /// The visible response (think tags stripped)
    pub visible_response: String,
    /// Token counts (approximate)
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    /// Whether generation was interrupted
    pub interrupted: bool,
    /// Whether the model hit max tokens (length) vs natural stop
    pub hit_max_tokens: bool,
}

/// Conversation info returned by list/get operations
#[derive(Clone)]
pub struct ConversationInfo {
    pub id: NodeId,
    pub title: String,
    pub created_at: String,
    pub message_count: usize,
}

/// A message from conversation history
#[derive(Clone)]
pub struct MessageInfo {
    pub role: String,
    pub content: String,
}

/// Messages sent to the LLM actor thread
pub enum ActorMessage {
    /// Generate a completion from a user query.
    Generate {
        query: String,
        result_tx: oneshot::Sender<Result<GenerateResult>>,
        token_tx: Option<mpsc::UnboundedSender<String>>,
    },
    /// List all conversations
    ListConversations {
        result_tx: oneshot::Sender<Vec<ConversationInfo>>,
    },
    /// Create a new conversation
    CreateConversation {
        title: String,
        result_tx: oneshot::Sender<ConversationInfo>,
    },
    /// Get a conversation by ID
    GetConversation {
        conv_id: NodeId,
        result_tx: oneshot::Sender<Option<ConversationInfo>>,
    },
    /// Delete a conversation
    DeleteConversation {
        conv_id: NodeId,
        result_tx: oneshot::Sender<bool>,
    },
    /// Switch to a conversation (for subsequent generates)
    SwitchConversation {
        conv_id: NodeId,
        result_tx: oneshot::Sender<bool>,
    },
    /// Get messages from a conversation
    GetMessages {
        conv_id: NodeId,
        limit: usize,
        result_tx: oneshot::Sender<Vec<MessageInfo>>,
    },
    /// Add a message to a conversation
    AddMessage {
        conv_id: NodeId,
        role: String,
        content: String,
        result_tx: oneshot::Sender<NodeId>,
    },
    /// Shutdown the actor
    Shutdown,
}

/// Configuration to initialize the actor's LLM resources
pub struct ActorConfig {
    pub engine: Engine,
    pub store: Option<Arc<LpgStore>>,
    pub schema: Option<GraphSchema>,
    pub banks: Vec<KvBank>,
    pub registry: KvNodeRegistry,
    pub conv_frags: ConvFragments,
    pub persona_db: Option<PersonaDB>,
    pub fact_gnn: Option<FactGNN>,
    pub max_nodes: usize,
    pub token_budget: i32,
    pub kv_capacity: i32,
}

/// Handle to communicate with the LLM actor thread
#[derive(Clone)]
pub struct ActorHandle {
    tx: mpsc::UnboundedSender<ActorMessage>,
}

impl ActorHandle {
    /// Spawn the actor on a dedicated std::thread (NOT tokio — FFI is blocking).
    pub fn spawn(config: ActorConfig) -> Self {
        let (tx, mut rx) = mpsc::unbounded_channel();

        std::thread::spawn(move || {
            let ActorConfig {
                engine,
                store,
                schema,
                banks,
                mut registry,
                mut conv_frags,
                mut persona_db,
                fact_gnn: _fact_gnn,
                max_nodes,
                token_budget,
                kv_capacity,
            } = config;

            let gen_ctl = GenerationControl {
                generating: Arc::new(AtomicBool::new(false)),
                sigint_received: Arc::new(AtomicBool::new(false)),
                gen_interrupted: Arc::new(AtomicBool::new(false)),
            };

            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("Failed to create actor runtime");

            rt.block_on(async move {
                while let Some(msg) = rx.recv().await {
                    match msg {
                        ActorMessage::Generate { query, result_tx, token_tx } => {
                            let result = Self::handle_generate(
                                &engine, &store, &schema, &mut registry,
                                &mut conv_frags, &banks, &query,
                                max_nodes, token_budget, kv_capacity,
                                &gen_ctl, &persona_db, token_tx,
                            );
                            let _ = result_tx.send(result);
                        }
                        ActorMessage::ListConversations { result_tx } => {
                            let result = Self::handle_list_conversations(&persona_db);
                            let _ = result_tx.send(result);
                        }
                        ActorMessage::CreateConversation { title, result_tx } => {
                            let result = Self::handle_create_conversation(&mut persona_db, &title);
                            let _ = result_tx.send(result);
                        }
                        ActorMessage::GetConversation { conv_id, result_tx } => {
                            let result = Self::handle_get_conversation(&persona_db, conv_id);
                            let _ = result_tx.send(result);
                        }
                        ActorMessage::DeleteConversation { conv_id: _, result_tx } => {
                            // TODO: PersonaDB doesn't support delete yet
                            let _ = result_tx.send(false);
                        }
                        ActorMessage::SwitchConversation { conv_id, result_tx } => {
                            let ok = if let Some(ref mut pdb) = persona_db {
                                pdb.switch_to(conv_id)
                            } else {
                                false
                            };
                            let _ = result_tx.send(ok);
                        }
                        ActorMessage::GetMessages { conv_id, limit, result_tx } => {
                            let result = Self::handle_get_messages(&mut persona_db, conv_id, limit);
                            let _ = result_tx.send(result);
                        }
                        ActorMessage::AddMessage { conv_id, role, content, result_tx } => {
                            let result = Self::handle_add_message(&mut persona_db, conv_id, &role, &content);
                            let _ = result_tx.send(result);
                        }
                        ActorMessage::Shutdown => break,
                    }
                }
            });
        });

        ActorHandle { tx }
    }

    pub fn send(&self, msg: ActorMessage) -> Result<()> {
        self.tx.send(msg).map_err(|_| anyhow::anyhow!("Actor channel closed"))
    }

    // ── Convenience methods ─────────────────────────────────────

    pub async fn generate(&self, query: String) -> Result<GenerateResult> {
        let (result_tx, result_rx) = oneshot::channel();
        self.send(ActorMessage::Generate { query, result_tx, token_tx: None })?;
        result_rx.await.map_err(|_| anyhow::anyhow!("Actor dropped"))?
    }

    pub async fn generate_streaming(
        &self, query: String, token_tx: mpsc::UnboundedSender<String>,
    ) -> Result<GenerateResult> {
        let (result_tx, result_rx) = oneshot::channel();
        self.send(ActorMessage::Generate { query, result_tx, token_tx: Some(token_tx) })?;
        result_rx.await.map_err(|_| anyhow::anyhow!("Actor dropped"))?
    }

    pub async fn list_conversations(&self) -> Result<Vec<ConversationInfo>> {
        let (result_tx, result_rx) = oneshot::channel();
        self.send(ActorMessage::ListConversations { result_tx })?;
        Ok(result_rx.await.map_err(|_| anyhow::anyhow!("Actor dropped"))?)
    }

    pub async fn create_conversation(&self, title: String) -> Result<ConversationInfo> {
        let (result_tx, result_rx) = oneshot::channel();
        self.send(ActorMessage::CreateConversation { title, result_tx })?;
        Ok(result_rx.await.map_err(|_| anyhow::anyhow!("Actor dropped"))?)
    }

    pub async fn get_conversation(&self, conv_id: NodeId) -> Result<Option<ConversationInfo>> {
        let (result_tx, result_rx) = oneshot::channel();
        self.send(ActorMessage::GetConversation { conv_id, result_tx })?;
        Ok(result_rx.await.map_err(|_| anyhow::anyhow!("Actor dropped"))?)
    }

    pub async fn delete_conversation(&self, conv_id: NodeId) -> Result<bool> {
        let (result_tx, result_rx) = oneshot::channel();
        self.send(ActorMessage::DeleteConversation { conv_id, result_tx })?;
        Ok(result_rx.await.map_err(|_| anyhow::anyhow!("Actor dropped"))?)
    }

    pub async fn switch_conversation(&self, conv_id: NodeId) -> Result<bool> {
        let (result_tx, result_rx) = oneshot::channel();
        self.send(ActorMessage::SwitchConversation { conv_id, result_tx })?;
        Ok(result_rx.await.map_err(|_| anyhow::anyhow!("Actor dropped"))?)
    }

    pub async fn get_messages(&self, conv_id: NodeId, limit: usize) -> Result<Vec<MessageInfo>> {
        let (result_tx, result_rx) = oneshot::channel();
        self.send(ActorMessage::GetMessages { conv_id, limit, result_tx })?;
        Ok(result_rx.await.map_err(|_| anyhow::anyhow!("Actor dropped"))?)
    }

    pub async fn add_message(&self, conv_id: NodeId, role: String, content: String) -> Result<NodeId> {
        let (result_tx, result_rx) = oneshot::channel();
        self.send(ActorMessage::AddMessage { conv_id, role, content, result_tx })?;
        Ok(result_rx.await.map_err(|_| anyhow::anyhow!("Actor dropped"))?)
    }

    // ── Internal handlers ───────────────────────────────────────

    fn handle_generate(
        engine: &Engine,
        store: &Option<Arc<LpgStore>>,
        schema: &Option<GraphSchema>,
        registry: &mut KvNodeRegistry,
        conv_frags: &mut ConvFragments,
        banks: &[KvBank],
        query: &str,
        max_nodes: usize,
        token_budget: i32,
        kv_capacity: i32,
        gen_ctl: &GenerationControl,
        persona_db: &Option<PersonaDB>,
        token_tx: Option<mpsc::UnboundedSender<String>>,
    ) -> Result<GenerateResult> {
        let output = match token_tx {
            Some(tx) => OutputMode::Channel(tx),
            None => OutputMode::Channel({
                let (tx, _rx) = mpsc::unbounded_channel();
                tx
            }),
        };

        let meta = is_meta_query(query);
        let (q_store, q_schema) = if meta {
            (None, None)
        } else {
            (store.as_ref(), schema.as_ref())
        };

        let prompt_tokens = registry.next_pos as u32;

        // TODO: pass real GNN context when fact_gnn is stored in actor state
        let (raw_response, relevant_graph_nodes) = query_with_registry(
            engine, q_store, q_schema, registry, conv_frags, banks,
            query, max_nodes, token_budget, kv_capacity, gen_ctl, &output,
            None, // gnn_ctx — server path, no GNN yet
        )?;

        let clean = strip_think_tags(&raw_response);
        let mut visible = clean.trim().to_string();

        if visible.is_empty() && !raw_response.is_empty() {
            let think_content = raw_response
                .strip_prefix("<think>").unwrap_or(&raw_response)
                .strip_suffix("</think>").unwrap_or(&raw_response)
                .trim();
            if !think_content.is_empty() {
                visible = think_content.to_string();
            }
        }

        let completion_tokens = (raw_response.len() as f64 / 3.5) as u32;
        let interrupted = gen_ctl.gen_interrupted.swap(false, Ordering::SeqCst);

        if let Err(e) = conv_frags.add_turn(
            query, &visible, &relevant_graph_nodes,
            registry, engine, kv_capacity,
        ) {
            eprintln!("  Warning: could not register conv fragment: {e}");
        }

        if let Some(pdb) = persona_db {
            let user_id = pdb.add_message("user", query);
            let asst_id = pdb.add_message("assistant", &visible);
            pdb.link_reply(asst_id, user_id);

            let matches = detect_facts_from_graph(pdb, query);
            for m in &matches {
                pdb.add_fact(&m.key, &m.value, 0, None);
            }
        }

        Ok(GenerateResult {
            raw_response,
            visible_response: visible,
            prompt_tokens,
            completion_tokens,
            interrupted,
            hit_max_tokens: false,
        })
    }

    fn handle_list_conversations(persona_db: &Option<PersonaDB>) -> Vec<ConversationInfo> {
        match persona_db {
            Some(pdb) => {
                pdb.list_conversations().into_iter().map(|(id, title, created_at, msg_count)| {
                    ConversationInfo { id, title, created_at, message_count: msg_count }
                }).collect()
            }
            None => Vec::new(),
        }
    }

    fn handle_create_conversation(persona_db: &mut Option<PersonaDB>, title: &str) -> ConversationInfo {
        if let Some(pdb) = persona_db {
            pdb.new_conversation(title);
            ConversationInfo {
                id: pdb.current_conv_id,
                title: title.to_string(),
                created_at: chrono::Utc::now().to_rfc3339(),
                message_count: 0,
            }
        } else {
            // No PersonaDB — return a dummy
            ConversationInfo {
                id: NodeId(0),
                title: title.to_string(),
                created_at: chrono::Utc::now().to_rfc3339(),
                message_count: 0,
            }
        }
    }

    fn handle_get_conversation(persona_db: &Option<PersonaDB>, conv_id: NodeId) -> Option<ConversationInfo> {
        let pdb = persona_db.as_ref()?;
        let convs = pdb.list_conversations();
        convs.into_iter()
            .find(|(id, _, _, _)| *id == conv_id)
            .map(|(id, title, created_at, msg_count)| {
                ConversationInfo { id, title, created_at, message_count: msg_count }
            })
    }

    fn handle_get_messages(persona_db: &mut Option<PersonaDB>, conv_id: NodeId, limit: usize) -> Vec<MessageInfo> {
        let pdb = match persona_db {
            Some(pdb) => pdb,
            None => return Vec::new(),
        };
        // Temporarily switch to the target conversation to get its messages
        let original_conv = pdb.current_conv_id;
        if !pdb.switch_to(conv_id) {
            return Vec::new();
        }
        let msgs = pdb.recent_messages(limit);
        // Switch back
        pdb.switch_to(original_conv);
        msgs.into_iter().map(|(role, content)| MessageInfo { role, content }).collect()
    }

    fn handle_add_message(persona_db: &mut Option<PersonaDB>, conv_id: NodeId, role: &str, content: &str) -> NodeId {
        let pdb = match persona_db {
            Some(pdb) => pdb,
            None => return NodeId(0),
        };
        let original_conv = pdb.current_conv_id;
        if !pdb.switch_to(conv_id) {
            return NodeId(0);
        }
        let msg_id = pdb.add_message(role, content);
        pdb.switch_to(original_conv);
        msg_id
    }
}
