//! Types for the OpenAI Conversations API
//! See: https://platform.openai.com/docs/api-reference/conversations

use serde::{Deserialize, Serialize};

// ── Conversation Object ────────────────────────────────────

#[derive(Serialize)]
pub struct Conversation {
    pub id: String,
    pub object: String, // "conversation"
    pub created_at: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

// ── Conversation Items ─────────────────────────────────────

#[derive(Serialize)]
pub struct ConversationItem {
    pub id: String,
    #[serde(rename = "type")]
    pub item_type: String, // "message"
    pub role: String, // "user" | "assistant"
    pub content: Vec<ConversationContent>,
    pub status: String, // "completed"
}

#[derive(Serialize)]
pub struct ConversationContent {
    #[serde(rename = "type")]
    pub content_type: String, // "input_text" | "output_text"
    pub text: String,
}

// ── Requests ───────────────────────────────────────────────

#[derive(Deserialize)]
pub struct CreateConversationRequest {
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
}

#[derive(Deserialize)]
pub struct AddItemsRequest {
    pub items: Vec<AddItem>,
}

#[derive(Deserialize)]
pub struct AddItem {
    #[serde(rename = "type", default = "default_message_type")]
    pub item_type: String,
    pub role: String,
    pub content: Vec<AddItemContent>,
}

fn default_message_type() -> String { "message".to_string() }

#[derive(Deserialize)]
pub struct AddItemContent {
    #[serde(rename = "type", default = "default_input_text_type")]
    pub content_type: String,
    pub text: String,
}

fn default_input_text_type() -> String { "input_text".to_string() }

// ── List Responses ─────────────────────────────────────────

#[derive(Serialize)]
pub struct ConversationList {
    pub object: String, // "list"
    pub data: Vec<Conversation>,
}

#[derive(Serialize)]
pub struct ConversationItemList {
    pub object: String, // "list"
    pub data: Vec<ConversationItem>,
}

// ── Helpers ────────────────────────────────────────────────

/// Generate a conversation ID in a stable format
pub fn generate_conversation_id() -> String {
    format!("conv_{}", uuid::Uuid::new_v4().simple())
}

/// Generate a message item ID
pub fn generate_message_id() -> String {
    format!("msg_{}", uuid::Uuid::new_v4().simple())
}
