//! # obrain-chat HTTP Server — OpenAI-compatible API
//!
//! Embedded HTTP server activated via `--http host:port`. Provides three API surfaces:
//!
//! ## Chat Completions API (Phase 1)
//!
//! ```text
//! GET  /v1/models                     → List available models
//! POST /v1/chat/completions           → Generate a completion (stream: false|true)
//! ```
//!
//! **Non-streaming example:**
//! ```bash
//! curl http://localhost:8080/v1/chat/completions \
//!   -H "Content-Type: application/json" \
//!   -d '{"model":"qwen3-8b","messages":[{"role":"user","content":"Hello"}]}'
//! ```
//!
//! **Streaming (SSE) example:**
//! ```bash
//! curl --no-buffer http://localhost:8080/v1/chat/completions \
//!   -H "Content-Type: application/json" \
//!   -d '{"model":"qwen3-8b","messages":[{"role":"user","content":"Hello"}],"stream":true}'
//! ```
//! Produces: `data: {"id":"chatcmpl-...","choices":[{"delta":{"content":"..."}}]}\n\n`
//! Ends with: `data: [DONE]\n\n`
//!
//! ## Responses API (Phase 2)
//!
//! ```text
//! POST /v1/responses                  → Create a response (stateful, supports chaining)
//! ```
//!
//! **Features:** `previous_response_id` for multi-turn chaining, `conversation` for
//! PersonaDB conversation linking, `stream: true` for typed SSE events
//! (`response.created`, `response.output_text.delta`, `response.completed`).
//!
//! ## Conversations API (Phase 2)
//!
//! ```text
//! GET    /v1/conversations             → List all conversations
//! POST   /v1/conversations             → Create a conversation
//! GET    /v1/conversations/{id}        → Get a conversation
//! DELETE /v1/conversations/{id}        → Delete a conversation
//! GET    /v1/conversations/{id}/items  → List messages in a conversation
//! POST   /v1/conversations/{id}/items  → Add messages to a conversation
//! ```
//!
//! Conversation IDs are in `conv_{node_id}` format (e.g. `conv_42`).
//!
//! ## WebSocket (Phase 3)
//!
//! ```text
//! GET /v1/realtime (Upgrade: websocket) → Persistent WebSocket connection
//! ```
//!
//! Client sends `{"type":"response.create", "model":"...", "input":"..."}`.
//! Server streams typed events. Supports `response.cancel`. 60-min timeout.
//!
//! ## Architecture
//!
//! All LLM inference runs on a **dedicated OS thread** (actor pattern) via
//! `tokio::sync::mpsc`. The Engine + KvRegistry + PersonaDB are owned by the actor
//! and never cross thread boundaries. Axum handlers communicate exclusively through
//! message-passing.
//!
//! ## SDK Compatibility
//!
//! Tested with the Python `openai` SDK:
//! ```python
//! from openai import OpenAI
//! client = OpenAI(base_url="http://localhost:8080/v1", api_key="none")
//! response = client.chat.completions.create(
//!     model="qwen3-8b",
//!     messages=[{"role": "user", "content": "Hello"}],
//! )
//! ```

pub mod actor;
pub mod state;
pub mod types;
pub mod types_responses;
pub mod types_conversations;
pub mod routes_conversations;
pub mod routes_responses;
pub mod ws;

use std::sync::Arc;
use std::net::SocketAddr;
use axum::{Router, Json, extract::State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::response::sse::{Event, Sse};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_stream::StreamExt;
use tower_http::cors::{CorsLayer, Any};

use crate::state::AppState;
use crate::types::*;

/// Start the HTTP server on the given address
pub async fn start_server(addr: SocketAddr, state: Arc<AppState>) -> anyhow::Result<()> {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/v1/models", axum::routing::get(list_models))
        .route("/v1/chat/completions", axum::routing::post(chat_completions))
        // Conversations API
        .route("/v1/conversations", axum::routing::get(routes_conversations::list_conversations))
        .route("/v1/conversations", axum::routing::post(routes_conversations::create_conversation))
        .route("/v1/conversations/{id}", axum::routing::get(routes_conversations::get_conversation))
        .route("/v1/conversations/{id}", axum::routing::delete(routes_conversations::delete_conversation))
        .route("/v1/conversations/{id}/items", axum::routing::get(routes_conversations::list_items))
        .route("/v1/conversations/{id}/items", axum::routing::post(routes_conversations::add_items))
        // Responses API
        .route("/v1/responses", axum::routing::post(routes_responses::create_response))
        // WebSocket endpoint for Responses API
        .route("/v1/realtime", axum::routing::get(ws::ws_handler))
        .layer(cors)
        .with_state(state);

    eprintln!("  HTTP server listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn list_models(State(state): State<Arc<AppState>>) -> Json<ModelList> {
    Json(ModelList {
        object: "list".to_string(),
        data: vec![ModelObject {
            id: state.model_name.clone(),
            object: "model".to_string(),
            created: state.model_created,
            owned_by: "obrain".to_string(),
        }],
    })
}

/// POST /v1/chat/completions — handles both streaming and non-streaming
async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    // Validate request
    if req.messages.is_empty() {
        return error_response(
            StatusCode::BAD_REQUEST,
            "messages must not be empty",
            "invalid_request_error",
            Some("messages"),
        ).into_response();
    }

    // Extract the last user message as the query
    let query = match req.messages.iter().rev().find(|m| m.role == "user") {
        Some(msg) => msg.content.clone(),
        None => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "No user message found in messages array",
                "invalid_request_error",
                Some("messages"),
            ).into_response();
        }
    };

    if req.stream {
        // ── SSE streaming mode ──────────────────────────────────────
        stream_chat_completions(state, query).await.into_response()
    } else {
        // ── Non-streaming mode ──────────────────────────────────────
        non_stream_chat_completions(state, query).await.into_response()
    }
}

/// Non-streaming: generate full response, return as JSON
async fn non_stream_chat_completions(
    state: Arc<AppState>,
    query: String,
) -> impl IntoResponse {
    match state.actor.generate(query).await {
        Ok(result) => {
            let finish_reason = if result.hit_max_tokens { "length" } else { "stop" };

            let response = ChatCompletionResponse {
                id: generate_completion_id(),
                object: "chat.completion".to_string(),
                created: unix_timestamp(),
                model: state.model_name.clone(),
                choices: vec![ChatChoice {
                    index: 0,
                    message: ChatMessage {
                        role: "assistant".to_string(),
                        content: result.visible_response,
                    },
                    finish_reason: Some(finish_reason.to_string()),
                }],
                usage: UsageStats {
                    prompt_tokens: result.prompt_tokens,
                    completion_tokens: result.completion_tokens,
                    total_tokens: result.prompt_tokens + result.completion_tokens,
                },
            };

            (StatusCode::OK, Json(serde_json::to_value(response).unwrap())).into_response()
        }
        Err(e) => {
            eprintln!("  [server] Generation error: {e}");
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("Generation failed: {e}"),
                "server_error",
                None,
            ).into_response()
        }
    }
}

/// SSE streaming: send token-by-token chunks as Server-Sent Events
async fn stream_chat_completions(
    state: Arc<AppState>,
    query: String,
) -> impl IntoResponse {
    let (token_tx, token_rx) = tokio::sync::mpsc::unbounded_channel::<String>();

    let completion_id = generate_completion_id();
    let model_name = state.model_name.clone();
    let created = unix_timestamp();

    // Spawn the generation in background — actor sends tokens via token_tx
    let actor = state.actor.clone();
    tokio::spawn(async move {
        let result = actor.generate_streaming(query, token_tx).await;
        if let Err(ref e) = result {
            eprintln!("  [server/sse] Generation error: {e}");
        }
        // When this task ends, token_tx is dropped → stream closes naturally
        drop(result);
    });

    // Build the SSE stream from the token channel
    let token_stream = UnboundedReceiverStream::new(token_rx);

    // First chunk: role announcement
    let first_chunk = ChatCompletionChunk {
        id: completion_id.clone(),
        object: "chat.completion.chunk".to_string(),
        created,
        model: model_name.clone(),
        choices: vec![ChunkChoice {
            index: 0,
            delta: ChunkDelta {
                role: Some("assistant".to_string()),
                content: None,
            },
            finish_reason: None,
        }],
    };

    let first_event = Event::default()
        .data(serde_json::to_string(&first_chunk).unwrap());

    // Map token fragments to SSE events
    let id = completion_id.clone();
    let model = model_name.clone();
    let content_stream = token_stream.map(move |token| {
        let chunk = ChatCompletionChunk {
            id: id.clone(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: model.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    content: Some(token),
                },
                finish_reason: None,
            }],
        };
        Ok::<_, std::convert::Infallible>(
            Event::default().data(serde_json::to_string(&chunk).unwrap())
        )
    });

    // Final chunk with finish_reason + [DONE] sentinel
    let final_id = completion_id;
    let final_model = model_name;
    let final_stream = futures::stream::once(async move {
        // Finish chunk
        let chunk = ChatCompletionChunk {
            id: final_id,
            object: "chat.completion.chunk".to_string(),
            created,
            model: final_model,
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    content: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
        };
        Ok::<_, std::convert::Infallible>(
            Event::default().data(serde_json::to_string(&chunk).unwrap())
        )
    }).chain(futures::stream::once(async {
        // [DONE] sentinel — standard OpenAI termination
        Ok::<_, std::convert::Infallible>(
            Event::default().data("[DONE]".to_string())
        )
    }));

    // Combine: first_event → content_events → finish + [DONE]
    let first = futures::stream::once(async move {
        Ok::<_, std::convert::Infallible>(first_event)
    });

    let full_stream = first
        .chain(content_stream)
        .chain(final_stream);

    Sse::new(full_stream)
}

/// Helper to build an OpenAI error response
fn error_response(
    status: StatusCode,
    message: &str,
    error_type: &str,
    param: Option<&str>,
) -> (StatusCode, Json<serde_json::Value>) {
    (
        status,
        Json(serde_json::to_value(OpenAIErrorResponse {
            error: OpenAIError {
                message: message.to_string(),
                error_type: error_type.to_string(),
                param: param.map(|s| s.to_string()),
                code: None,
            },
        }).unwrap()),
    )
}
