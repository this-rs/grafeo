//! RAG Pipeline — the high-level API that ties retrieval, context building,
//! and feedback into a single workflow.
//!
//! ```rust,ignore
//! let pipeline = RagPipeline::new(retriever, context_builder, feedback_sink, config);
//! let context = pipeline.query("What projects use plans?")?;
//! // Inject context.text into LLM prompt...
//! // After LLM response:
//! pipeline.feedback(&context, &llm_response)?;
//! ```

use std::sync::Arc;

use crate::config::RagConfig;
use crate::error::RagResult;
use crate::traits::{
    ContextBuilder, FeedbackSink, FeedbackStats, RagContext, RetrievalResult, Retriever,
};

/// The main RAG pipeline — query, build context, and provide feedback.
pub struct RagPipeline {
    retriever: Arc<dyn Retriever>,
    context_builder: Box<dyn ContextBuilder>,
    feedback_sink: Option<Box<dyn FeedbackSink>>,
    config: RagConfig,
}

impl RagPipeline {
    /// Create a new RAG pipeline with all components.
    ///
    /// The retriever is wrapped in `Arc` so the caller can keep a
    /// reference to the concrete type (e.g. `EngramRetriever`) for
    /// incremental index updates while the pipeline uses it for queries.
    pub fn new(
        retriever: Arc<dyn Retriever>,
        context_builder: impl ContextBuilder + 'static,
        feedback_sink: Option<Box<dyn FeedbackSink>>,
        config: RagConfig,
    ) -> Self {
        Self {
            retriever,
            context_builder: Box::new(context_builder),
            feedback_sink,
            config,
        }
    }

    /// Execute a RAG query: retrieve → rank → budget → format.
    ///
    /// Returns the formatted context ready for LLM prompt injection.
    pub fn query(&self, query: &str) -> RagResult<RagContext> {
        let retrieval = self.retrieve(query)?;
        self.context_builder.build(&retrieval, &self.config)
    }

    /// Execute only the retrieval step (useful for debugging).
    pub fn retrieve(&self, query: &str) -> RagResult<RetrievalResult> {
        self.retriever.retrieve(query, &self.config)
    }

    /// Provide feedback after an LLM response.
    ///
    /// Reinforces synapses between co-activated concepts and boosts
    /// energy of nodes that were included in the context.
    pub fn feedback(&self, context: &RagContext, response: &str) -> RagResult<FeedbackStats> {
        match &self.feedback_sink {
            Some(sink) => sink.feedback(context, response, &self.config),
            None => Ok(FeedbackStats::default()),
        }
    }

    /// Get the current configuration.
    pub fn config(&self) -> &RagConfig {
        &self.config
    }

    /// Update the configuration.
    pub fn set_config(&mut self, config: RagConfig) {
        self.config = config;
    }
}

impl std::fmt::Debug for RagPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RagPipeline")
            .field("config", &self.config)
            .field("has_feedback", &self.feedback_sink.is_some())
            .finish()
    }
}
