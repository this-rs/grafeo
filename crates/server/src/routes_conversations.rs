//! Routes for the Conversations API (CRUD + items)
//! Maps to PersonaDB conversations via the actor.

use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::{
    Json,
    extract::{Path, State},
};
use obrain_common::types::NodeId;
use std::sync::Arc;

use crate::state::AppState;
use crate::types::{OpenAIError, OpenAIErrorResponse};
use crate::types_conversations::*;

/// GET /v1/conversations — list all conversations
pub async fn list_conversations(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    match state.actor.list_conversations().await {
        Ok(convs) => {
            let data = convs
                .into_iter()
                .map(|c| Conversation {
                    id: format!("conv_{}", c.id.0),
                    object: "conversation".to_string(),
                    created_at: parse_timestamp(&c.created_at),
                    metadata: Some(serde_json::json!({ "title": c.title })),
                })
                .collect();

            (
                StatusCode::OK,
                Json(
                    serde_json::to_value(ConversationList {
                        object: "list".to_string(),
                        data,
                    })
                    .unwrap(),
                ),
            )
                .into_response()
        }
        Err(e) => {
            error_response(StatusCode::INTERNAL_SERVER_ERROR, &format!("{e}")).into_response()
        }
    }
}

/// POST /v1/conversations — create a new conversation
pub async fn create_conversation(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateConversationRequest>,
) -> impl IntoResponse {
    let title = req
        .metadata
        .as_ref()
        .and_then(|m| m.get("title"))
        .and_then(|v| v.as_str())
        .unwrap_or("New conversation")
        .to_string();

    match state.actor.create_conversation(title.clone()).await {
        Ok(info) => {
            let conv = Conversation {
                id: format!("conv_{}", info.id.0),
                object: "conversation".to_string(),
                created_at: parse_timestamp(&info.created_at),
                metadata: Some(serde_json::json!({ "title": info.title })),
            };
            (
                StatusCode::CREATED,
                Json(serde_json::to_value(conv).unwrap()),
            )
                .into_response()
        }
        Err(e) => {
            error_response(StatusCode::INTERNAL_SERVER_ERROR, &format!("{e}")).into_response()
        }
    }
}

/// GET /v1/conversations/:id — get a conversation
pub async fn get_conversation(
    State(state): State<Arc<AppState>>,
    Path(conv_id_str): Path<String>,
) -> impl IntoResponse {
    let conv_id = match parse_conv_id(&conv_id_str) {
        Some(id) => id,
        None => {
            return error_response(StatusCode::BAD_REQUEST, "Invalid conversation ID format")
                .into_response();
        }
    };

    match state.actor.get_conversation(conv_id).await {
        Ok(Some(info)) => {
            let conv = Conversation {
                id: conv_id_str,
                object: "conversation".to_string(),
                created_at: parse_timestamp(&info.created_at),
                metadata: Some(serde_json::json!({ "title": info.title })),
            };
            (StatusCode::OK, Json(serde_json::to_value(conv).unwrap())).into_response()
        }
        Ok(None) => error_response(StatusCode::NOT_FOUND, "Conversation not found").into_response(),
        Err(e) => {
            error_response(StatusCode::INTERNAL_SERVER_ERROR, &format!("{e}")).into_response()
        }
    }
}

/// DELETE /v1/conversations/:id — delete a conversation
pub async fn delete_conversation(
    State(state): State<Arc<AppState>>,
    Path(conv_id_str): Path<String>,
) -> impl IntoResponse {
    let conv_id = match parse_conv_id(&conv_id_str) {
        Some(id) => id,
        None => {
            return error_response(StatusCode::BAD_REQUEST, "Invalid conversation ID format")
                .into_response();
        }
    };

    match state.actor.delete_conversation(conv_id).await {
        Ok(true) => StatusCode::NO_CONTENT.into_response(),
        Ok(false) => {
            error_response(StatusCode::NOT_FOUND, "Conversation not found").into_response()
        }
        Err(e) => {
            error_response(StatusCode::INTERNAL_SERVER_ERROR, &format!("{e}")).into_response()
        }
    }
}

/// GET /v1/conversations/:id/items — list messages in a conversation
pub async fn list_items(
    State(state): State<Arc<AppState>>,
    Path(conv_id_str): Path<String>,
) -> impl IntoResponse {
    let conv_id = match parse_conv_id(&conv_id_str) {
        Some(id) => id,
        None => {
            return error_response(StatusCode::BAD_REQUEST, "Invalid conversation ID format")
                .into_response();
        }
    };

    match state.actor.get_messages(conv_id, 100).await {
        Ok(msgs) => {
            let data: Vec<ConversationItem> = msgs
                .into_iter()
                .enumerate()
                .map(|(i, m)| {
                    let content_type = if m.role == "user" {
                        "input_text"
                    } else {
                        "output_text"
                    };
                    ConversationItem {
                        id: format!("msg_{}", i),
                        item_type: "message".to_string(),
                        role: m.role,
                        content: vec![ConversationContent {
                            content_type: content_type.to_string(),
                            text: m.content,
                        }],
                        status: "completed".to_string(),
                    }
                })
                .collect();

            (
                StatusCode::OK,
                Json(
                    serde_json::to_value(ConversationItemList {
                        object: "list".to_string(),
                        data,
                    })
                    .unwrap(),
                ),
            )
                .into_response()
        }
        Err(e) => {
            error_response(StatusCode::INTERNAL_SERVER_ERROR, &format!("{e}")).into_response()
        }
    }
}

/// POST /v1/conversations/:id/items — add messages to a conversation
pub async fn add_items(
    State(state): State<Arc<AppState>>,
    Path(conv_id_str): Path<String>,
    Json(req): Json<AddItemsRequest>,
) -> impl IntoResponse {
    let conv_id = match parse_conv_id(&conv_id_str) {
        Some(id) => id,
        None => {
            return error_response(StatusCode::BAD_REQUEST, "Invalid conversation ID format")
                .into_response();
        }
    };

    for item in &req.items {
        let content_text = item
            .content
            .iter()
            .map(|c| c.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");

        if let Err(e) = state
            .actor
            .add_message(conv_id, item.role.clone(), content_text)
            .await
        {
            return error_response(StatusCode::INTERNAL_SERVER_ERROR, &format!("{e}"))
                .into_response();
        }
    }

    StatusCode::OK.into_response()
}

// ── Helpers ─────────────────────────────────────────────────

/// Parse "conv_123" → NodeId(123)
fn parse_conv_id(s: &str) -> Option<NodeId> {
    let num_str = s.strip_prefix("conv_")?;
    num_str.parse::<u64>().ok().map(NodeId)
}

/// Parse RFC3339 timestamp to unix seconds, or return 0
fn parse_timestamp(ts: &str) -> u64 {
    chrono::DateTime::parse_from_rfc3339(ts)
        .map(|dt| dt.timestamp() as u64)
        .unwrap_or(0)
}

fn error_response(status: StatusCode, message: &str) -> (StatusCode, Json<serde_json::Value>) {
    (
        status,
        Json(
            serde_json::to_value(OpenAIErrorResponse {
                error: OpenAIError {
                    message: message.to_string(),
                    error_type: "server_error".to_string(),
                    param: None,
                    code: None,
                },
            })
            .unwrap(),
        ),
    )
}
