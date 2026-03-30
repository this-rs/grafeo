//! Routes for the OpenAI Responses API (POST /v1/responses)
//! Supports: stateful multi-turn via previous_response_id, conversation linking, SSE streaming.

use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::response::sse::{Event, Sse};
use axum::{Json, extract::State};
use obrain_common::types::NodeId;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::UnboundedReceiverStream;

use crate::state::AppState;
use crate::types::{OpenAIError, OpenAIErrorResponse};
use crate::types_responses::*;

/// In-memory cache for response chaining via previous_response_id.
/// Maps response_id → (conversation_id, last_query, last_response).
pub type ResponseCache = Arc<Mutex<HashMap<String, CachedResponse>>>;

pub struct CachedResponse {
    pub conversation_id: Option<NodeId>,
    pub query: String,
    pub response_text: String,
}

/// Create a new empty response cache
pub fn new_response_cache() -> ResponseCache {
    Arc::new(Mutex::new(HashMap::new()))
}

/// POST /v1/responses — create a response (non-streaming or streaming)
pub async fn create_response(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ResponseRequest>,
) -> impl IntoResponse {
    // Extract user query from input
    let query = match req.input.last_user_text() {
        Some(text) => text.to_string(),
        None => {
            return error_response(StatusCode::BAD_REQUEST, "No user text found in input")
                .into_response();
        }
    };

    // Handle conversation switching if specified
    if let Some(ref conv_ref) = req.conversation {
        match conv_ref {
            ConversationRef::Id(id) => {
                if let Some(conv_id) = parse_conv_id(id) {
                    let _ = state.actor.switch_conversation(conv_id).await;
                }
            }
            ConversationRef::Config(_) => {
                // "auto" mode — use current conversation (default behavior)
            }
        }
    }

    // Handle previous_response_id chaining
    if let Some(ref prev_id) = req.previous_response_id {
        let cache = state.response_cache.lock().await;
        if let Some(cached) = cache.get(prev_id) {
            // Switch to the conversation from the previous response
            if let Some(conv_id) = cached.conversation_id {
                let _ = state.actor.switch_conversation(conv_id).await;
            }
        }
        // If prev_id not found, we continue anyway (the conversation context
        // is maintained by PersonaDB's current conversation)
    }

    if req.stream {
        stream_response(state, req, query).await.into_response()
    } else {
        non_stream_response(state, req, query).await.into_response()
    }
}

/// Non-streaming response
async fn non_stream_response(
    state: Arc<AppState>,
    req: ResponseRequest,
    query: String,
) -> impl IntoResponse {
    match state.actor.generate(query.clone()).await {
        Ok(result) => {
            let response_id = generate_response_id();
            let item_id = generate_item_id();

            let response = ResponseObject {
                id: response_id.clone(),
                object: "response".to_string(),
                created_at: crate::types::unix_timestamp(),
                model: req.model,
                status: "completed".to_string(),
                output: vec![OutputItem {
                    item_type: "message".to_string(),
                    id: item_id,
                    role: "assistant".to_string(),
                    content: vec![ContentPart {
                        content_type: "output_text".to_string(),
                        text: result.visible_response.clone(),
                    }],
                    status: "completed".to_string(),
                }],
                usage: ResponseUsage {
                    input_tokens: result.prompt_tokens,
                    output_tokens: result.completion_tokens,
                    total_tokens: result.prompt_tokens + result.completion_tokens,
                },
                metadata: req.metadata,
                previous_response_id: req.previous_response_id,
                conversation_id: None, // TODO: expose current conv_id from actor
            };

            // Cache for chaining
            if req.store {
                let mut cache = state.response_cache.lock().await;
                cache.insert(
                    response_id.clone(),
                    CachedResponse {
                        conversation_id: None,
                        query,
                        response_text: result.visible_response,
                    },
                );
            }

            (
                StatusCode::OK,
                Json(serde_json::to_value(response).unwrap()),
            )
                .into_response()
        }
        Err(e) => {
            eprintln!("  [server/responses] Generation error: {e}");
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("Generation failed: {e}"),
            )
            .into_response()
        }
    }
}

/// SSE streaming response with typed events
async fn stream_response(
    state: Arc<AppState>,
    req: ResponseRequest,
    query: String,
) -> impl IntoResponse {
    let (token_tx, token_rx) = tokio::sync::mpsc::unbounded_channel::<String>();

    let response_id = generate_response_id();
    let item_id = generate_item_id();
    let created_at = crate::types::unix_timestamp();
    let model = req.model.clone();

    // Spawn generation
    let actor = state.actor.clone();
    let cache = state.response_cache.clone();
    let store_response = req.store;
    let query_clone = query.clone();
    let response_id_clone = response_id.clone();
    tokio::spawn(async move {
        let result = actor
            .generate_streaming(query_clone.clone(), token_tx)
            .await;
        // Cache result for chaining
        if store_response {
            if let Ok(ref r) = result {
                let mut c = cache.lock().await;
                c.insert(
                    response_id_clone,
                    CachedResponse {
                        conversation_id: None,
                        query: query_clone,
                        response_text: r.visible_response.clone(),
                    },
                );
            }
        }
        drop(result);
    });

    let token_stream = UnboundedReceiverStream::new(token_rx);

    // Event 1: response.created
    let created_event = serde_json::json!({
        "type": "response.created",
        "response": {
            "id": response_id,
            "object": "response",
            "created_at": created_at,
            "model": model,
            "status": "in_progress",
            "output": [],
        }
    });

    // Event 2: response.output_item.added
    let item_added_event = serde_json::json!({
        "type": "response.output_item.added",
        "output_index": 0,
        "item": {
            "type": "message",
            "id": item_id,
            "role": "assistant",
            "content": [],
            "status": "in_progress",
        }
    });

    // Event 3: response.content_part.added
    let part_added_event = serde_json::json!({
        "type": "response.content_part.added",
        "output_index": 0,
        "content_index": 0,
        "part": {
            "type": "output_text",
            "text": "",
        }
    });

    // Initial events
    let init_events = futures::stream::iter(vec![
        Ok::<_, std::convert::Infallible>(Event::default().data(created_event.to_string())),
        Ok(Event::default().data(item_added_event.to_string())),
        Ok(Event::default().data(part_added_event.to_string())),
    ]);

    // Delta events for each token
    let delta_events = token_stream.map(move |token| {
        let event = serde_json::json!({
            "type": "response.output_text.delta",
            "output_index": 0,
            "content_index": 0,
            "delta": token,
        });
        Ok::<_, std::convert::Infallible>(Event::default().data(event.to_string()))
    });

    // Final events
    let rid = response_id.clone();
    let iid = item_id.clone();
    let m = model.clone();
    let final_events = futures::stream::iter(vec![
        // response.output_text.done
        Ok::<_, std::convert::Infallible>(
            Event::default().data(
                serde_json::json!({
                    "type": "response.output_text.done",
                    "output_index": 0,
                    "content_index": 0,
                })
                .to_string(),
            ),
        ),
        // response.output_item.done
        Ok(Event::default().data(
            serde_json::json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "message",
                    "id": iid,
                    "role": "assistant",
                    "status": "completed",
                }
            })
            .to_string(),
        )),
        // response.completed
        Ok(Event::default().data(
            serde_json::json!({
                "type": "response.completed",
                "response": {
                    "id": rid,
                    "object": "response",
                    "created_at": created_at,
                    "model": m,
                    "status": "completed",
                }
            })
            .to_string(),
        )),
    ]);

    let full_stream = init_events.chain(delta_events).chain(final_events);

    Sse::new(full_stream)
}

// ── Helpers ─────────────────────────────────────────────────

fn parse_conv_id(s: &str) -> Option<NodeId> {
    let num_str = s.strip_prefix("conv_")?;
    num_str.parse::<u64>().ok().map(NodeId)
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
