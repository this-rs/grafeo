use crate::actor::ActorHandle;
use crate::routes_responses::ResponseCache;

/// Shared application state for axum handlers
pub struct AppState {
    pub actor: ActorHandle,
    pub model_name: String,
    pub model_created: u64,
    pub response_cache: ResponseCache,
}
