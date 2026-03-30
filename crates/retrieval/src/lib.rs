pub mod attn_compiler;
pub mod attn_dsl;
pub mod contrastive;
pub mod formula_evolution;
pub mod formula_selector;
mod control;
mod engine;
mod generation;
mod meta;
pub mod node_embedding;
pub mod projection_net;
mod query;
pub mod round_tracker;
mod scoring;
pub mod iptr;
pub mod iptr_graph;
pub mod self_embedding;
pub mod state_bias;
pub mod training;

pub use contrastive::{ContrastiveConfig, ContrastiveSample, GraphTopology};
pub use control::{GenerationControl, OutputMode, Spinner};
pub use engine::Engine;
pub use generation::{AblationReward, generate_with_mask};
pub use meta::is_meta_query;
pub use node_embedding::{
    FusionContext, NodeEmbeddingCache, compute_fused_embedding, compute_node_embeddings,
    compute_node_embeddings_with_fusion, compute_text_embedding,
};
pub use projection_net::{ProjectionNet, alpha_schedule, soft_mix};
pub use query::{GnnContext, maybe_relayout, query_with_registry};
pub use round_tracker::{CoactivationMap, DemotionType, RoundTracker};
pub use scoring::{ScoredContextNode, get_micro_tag, retrieve_nodes};
pub use scoring::{compute_lambda, expand_by_affinity};
pub use training::{TrainingConfig, TrainingManager, weights_path_for_persona};
