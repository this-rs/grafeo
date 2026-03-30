//! WebSocket handler for the Responses API.
//! Provides a persistent connection where the client sends response.create
//! messages and receives typed SSE-like events over the WebSocket.

use axum::extract::ws::{Message, WebSocket};
use axum::extract::{State, WebSocketUpgrade};
use axum::response::IntoResponse;
use std::sync::Arc;
use tokio::sync::mpsc;

use crate::routes_responses::CachedResponse;
use crate::state::AppState;
use crate::types_responses::*;

/// WebSocket client message types
#[derive(serde::Deserialize)]
struct WsClientMessage {
    #[serde(rename = "type")]
    msg_type: String,
    #[serde(flatten)]
    data: serde_json::Value,
}

/// GET /v1/realtime — WebSocket upgrade for Responses API
pub async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_ws(socket, state))
}

async fn handle_ws(mut socket: WebSocket, state: Arc<AppState>) {
    // Timeout: 60 minutes max per connection
    let timeout = tokio::time::Duration::from_secs(60 * 60);
    let deadline = tokio::time::Instant::now() + timeout;

    // Track in-flight generation for cancellation
    let mut cancel_tx: Option<tokio::sync::oneshot::Sender<()>> = None;

    loop {
        let msg = tokio::select! {
            msg = socket.recv() => msg,
            _ = tokio::time::sleep_until(deadline) => {
                let _ = socket.send(Message::Close(None)).await;
                break;
            }
        };

        match msg {
            Some(Ok(Message::Text(text))) => {
                let parsed: Result<WsClientMessage, _> = serde_json::from_str(&text);
                match parsed {
                    Ok(client_msg) => {
                        match client_msg.msg_type.as_str() {
                            "response.create" => {
                                // Cancel any in-flight generation
                                if let Some(tx) = cancel_tx.take() {
                                    let _ = tx.send(());
                                }

                                // Parse the response request from the data
                                let req: Result<ResponseRequest, _> =
                                    serde_json::from_value(client_msg.data);
                                match req {
                                    Ok(req) => {
                                        let (ctx, crx) = tokio::sync::oneshot::channel();
                                        cancel_tx = Some(ctx);
                                        handle_response_create(&mut socket, &state, req, crx).await;
                                    }
                                    Err(e) => {
                                        send_error(&mut socket, &format!("Invalid request: {e}"))
                                            .await;
                                    }
                                }
                            }
                            "response.cancel" => {
                                if let Some(tx) = cancel_tx.take() {
                                    let _ = tx.send(());
                                    send_event(
                                        &mut socket,
                                        "response.cancelled",
                                        serde_json::json!({}),
                                    )
                                    .await;
                                }
                            }
                            other => {
                                send_error(&mut socket, &format!("Unknown message type: {other}"))
                                    .await;
                            }
                        }
                    }
                    Err(e) => {
                        send_error(&mut socket, &format!("Invalid JSON: {e}")).await;
                    }
                }
            }
            Some(Ok(Message::Close(_))) | None => break,
            Some(Ok(Message::Ping(data))) => {
                let _ = socket.send(Message::Pong(data)).await;
            }
            _ => {} // ignore binary, pong
        }
    }
}

/// Handle a response.create message: generate and stream events back
async fn handle_response_create(
    socket: &mut WebSocket,
    state: &Arc<AppState>,
    req: ResponseRequest,
    cancel_rx: tokio::sync::oneshot::Receiver<()>,
) {
    let query = match req.input.last_user_text() {
        Some(text) => text.to_string(),
        None => {
            send_error(socket, "No user text found in input").await;
            return;
        }
    };

    let response_id = generate_response_id();
    let item_id = generate_item_id();
    let created_at = crate::types::unix_timestamp();
    let model = req.model.clone();

    // Send response.created
    send_event(
        socket,
        "response.created",
        serde_json::json!({
            "response": {
                "id": &response_id,
                "object": "response",
                "created_at": created_at,
                "model": &model,
                "status": "in_progress",
            }
        }),
    )
    .await;

    // Send output_item.added
    send_event(
        socket,
        "response.output_item.added",
        serde_json::json!({
            "output_index": 0,
            "item": {
                "type": "message",
                "id": &item_id,
                "role": "assistant",
                "status": "in_progress",
            }
        }),
    )
    .await;

    // Send content_part.added
    send_event(
        socket,
        "response.content_part.added",
        serde_json::json!({
            "output_index": 0,
            "content_index": 0,
            "part": { "type": "output_text", "text": "" }
        }),
    )
    .await;

    // Start generation with streaming
    let (token_tx, mut token_rx) = mpsc::unbounded_channel::<String>();
    let actor = state.actor.clone();
    let query_clone = query.clone();
    let gen_handle =
        tokio::spawn(async move { actor.generate_streaming(query_clone, token_tx).await });

    // Stream deltas until generation ends or cancellation
    let mut cancelled = false;
    tokio::pin!(cancel_rx);
    loop {
        tokio::select! {
            token = token_rx.recv() => {
                match token {
                    Some(t) => {
                        send_event(socket, "response.output_text.delta", serde_json::json!({
                            "output_index": 0,
                            "content_index": 0,
                            "delta": t,
                        })).await;
                    }
                    None => break, // channel closed — generation done
                }
            }
            _ = &mut cancel_rx => {
                cancelled = true;
                break;
            }
        }
    }

    // Wait for generation task to finish
    let _ = gen_handle.await;

    let status = if cancelled { "cancelled" } else { "completed" };

    // Send completion events
    send_event(
        socket,
        "response.output_text.done",
        serde_json::json!({
            "output_index": 0,
            "content_index": 0,
        }),
    )
    .await;

    send_event(
        socket,
        "response.output_item.done",
        serde_json::json!({
            "output_index": 0,
            "item": {
                "type": "message",
                "id": &item_id,
                "role": "assistant",
                "status": status,
            }
        }),
    )
    .await;

    send_event(
        socket,
        "response.completed",
        serde_json::json!({
            "response": {
                "id": &response_id,
                "object": "response",
                "created_at": created_at,
                "model": &model,
                "status": status,
            }
        }),
    )
    .await;

    // Cache for chaining
    if req.store {
        let mut cache = state.response_cache.lock().await;
        cache.insert(
            response_id,
            CachedResponse {
                conversation_id: None,
                query,
                response_text: String::new(), // full text not easily available in streaming
            },
        );
    }
}

async fn send_event(socket: &mut WebSocket, event_type: &str, data: serde_json::Value) {
    let mut event = data;
    event
        .as_object_mut()
        .map(|o| o.insert("type".to_string(), serde_json::json!(event_type)));
    let text = serde_json::to_string(&event).unwrap_or_default();
    let _ = socket.send(Message::Text(text.into())).await;
}

async fn send_error(socket: &mut WebSocket, message: &str) {
    send_event(
        socket,
        "error",
        serde_json::json!({
            "message": message,
        }),
    )
    .await;
}
