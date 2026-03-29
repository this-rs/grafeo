//! Types for the OpenAI Responses API (POST /v1/responses)
//! See: https://platform.openai.com/docs/api-reference/responses

use serde::{Deserialize, Serialize};

// ── Request ────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct ResponseRequest {
    pub model: String,
    /// Input can be a simple string or an array of input items
    pub input: ResponseInput,
    /// System instructions (optional, overrides default)
    #[serde(default)]
    pub instructions: Option<String>,
    /// Chain to a previous response for multi-turn
    #[serde(default)]
    pub previous_response_id: Option<String>,
    /// Associate with a conversation
    #[serde(default)]
    pub conversation: Option<ConversationRef>,
    /// Whether to persist in server-side storage (default true)
    #[serde(default = "default_true")]
    pub store: bool,
    /// Enable SSE streaming
    #[serde(default)]
    pub stream: bool,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default)]
    pub max_output_tokens: Option<i32>,
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
}

fn default_true() -> bool { true }
fn default_temperature() -> f32 { 1.0 }

/// Input: either a plain string or structured input items
#[derive(Deserialize)]
#[serde(untagged)]
pub enum ResponseInput {
    Text(String),
    Items(Vec<InputItem>),
}

impl ResponseInput {
    /// Extract the last user message text (or the plain text)
    pub fn last_user_text(&self) -> Option<&str> {
        match self {
            ResponseInput::Text(s) => Some(s.as_str()),
            ResponseInput::Items(items) => {
                items.iter().rev()
                    .find(|i| i.role.as_deref() == Some("user"))
                    .and_then(|i| i.content.first())
                    .and_then(|c| c.text.as_deref())
            }
        }
    }
}

/// Conversation reference: auto-create or use existing
#[derive(Deserialize)]
#[serde(untagged)]
pub enum ConversationRef {
    /// Use an existing conversation by ID
    Id(String),
    /// Auto-create config
    Config(ConversationConfig),
}

#[derive(Deserialize)]
pub struct ConversationConfig {
    /// "auto" or "none"
    #[serde(default)]
    pub mode: Option<String>,
}

#[derive(Deserialize)]
pub struct InputItem {
    #[serde(rename = "type", default)]
    pub item_type: Option<String>,
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub content: Vec<InputContent>,
}

#[derive(Deserialize)]
pub struct InputContent {
    #[serde(rename = "type", default)]
    pub content_type: Option<String>,
    #[serde(default)]
    pub text: Option<String>,
}

// ── Response ───────────────────────────────────────────────

#[derive(Serialize)]
pub struct ResponseObject {
    pub id: String,
    pub object: String, // "response"
    pub created_at: u64,
    pub model: String,
    pub status: String, // "completed", "failed", "in_progress"
    pub output: Vec<OutputItem>,
    pub usage: ResponseUsage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation_id: Option<String>,
}

#[derive(Serialize)]
pub struct OutputItem {
    #[serde(rename = "type")]
    pub item_type: String, // "message"
    pub id: String,
    pub role: String, // "assistant"
    pub content: Vec<ContentPart>,
    pub status: String, // "completed"
}

#[derive(Serialize)]
pub struct ContentPart {
    #[serde(rename = "type")]
    pub content_type: String, // "output_text"
    pub text: String,
}

#[derive(Serialize)]
pub struct ResponseUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
}

// ── SSE Streaming Events ───────────────────────────────────

/// Typed SSE events for the Responses API streaming
#[derive(Serialize)]
pub struct ResponseStreamEvent {
    #[serde(rename = "type")]
    pub event_type: String,
    #[serde(flatten)]
    pub data: serde_json::Value,
}

/// Generate a response ID in OpenAI format
pub fn generate_response_id() -> String {
    format!("resp_{}", uuid::Uuid::new_v4().simple())
}

/// Generate an output item ID
pub fn generate_item_id() -> String {
    format!("msg_{}", uuid::Uuid::new_v4().simple())
}
