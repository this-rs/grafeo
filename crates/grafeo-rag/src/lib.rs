//! # grafeo-rag
//!
//! Graph-Augmented Retrieval via Engrams — schema-agnostic RAG for Grafeo.
//!
//! This crate implements a retrieval-augmented generation pipeline that uses
//! Grafeo's cognitive layer (engrams, synapses, energy) to find relevant
//! graph content for LLM prompts, **without knowing the database schema**.
//!
//! ## Architecture
//!
//! The pipeline operates in three stages:
//!
//! 1. **Engram Recall** — Given a text query, use Hopfield spectral matching
//!    to find the most relevant engrams (consolidated memory traces).
//!
//! 2. **Spreading Activation** — From the nodes in recalled engrams, propagate
//!    activation through synapses (BFS with decay) to discover related content.
//!
//! 3. **Context Building** — Extract text properties from activated nodes
//!    (schema-agnostic), rank by composite score, budget tokens, and format.
//!
//! An optional **Feedback Loop** reinforces synapses between concepts that
//! were useful in the LLM's response, so the system improves with usage.
//!
//! ## Schema Agnosticism
//!
//! The retriever never hardcodes node labels or property names.
//! Instead, it:
//! - Uses **engrams** as the entry point (cognitive abstraction over raw nodes)
//! - Extracts **all string-valued properties** from activated nodes
//! - Includes **labels and relation types** as metadata
//!
//! This means it works on any GrafeoDB regardless of its schema.
//!
//! ## Example
//!
//! ```rust,no_run
//! use grafeo_rag::{RagPipeline, RagConfig};
//!
//! // pipeline = RagPipeline::new(db, cognitive_engine, config);
//! // let context = pipeline.query("What projects use plans?")?;
//! // inject context.text into LLM prompt
//! ```

#![deny(unsafe_code)]

pub mod budget;
pub mod config;
pub mod context;
pub mod error;
pub mod feedback;
pub mod pipeline;
pub mod ranking;
pub mod retriever;
pub mod traits;

// Re-exports
pub use config::RagConfig;
pub use context::GraphContextBuilder;
pub use error::{RagError, RagResult};
pub use feedback::CognitiveFeedback;
pub use pipeline::RagPipeline;
pub use retriever::EngramRetriever;
pub use traits::{
    ContextBuilder, FeedbackSink, FeedbackStats, RagContext, RetrievalResult, RetrievalSource,
    RetrievedNode, Retriever,
};
