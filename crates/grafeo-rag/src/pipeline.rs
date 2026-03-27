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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{
        ContextBuilder, FeedbackSink, FeedbackStats, RagContext, RetrievalResult, RetrievalSource,
        RetrievedNode, Retriever,
    };
    use grafeo_common::types::NodeId;
    use std::collections::HashMap;

    /// Mock retriever that returns a fixed set of nodes.
    struct MockRetriever {
        nodes: Vec<RetrievedNode>,
    }

    impl MockRetriever {
        fn new(count: usize) -> Self {
            let nodes = (0..count)
                .map(|i| {
                    let mut props = HashMap::new();
                    props.insert("name".into(), format!("Node {i}"));
                    RetrievedNode {
                        node_id: NodeId(i as u64),
                        labels: vec!["Test".into()],
                        properties: props,
                        score: 1.0 - (i as f64 * 0.1),
                        source: RetrievalSource::SpreadingActivation {
                            depth: 0,
                            activation: 1.0,
                        },
                        outgoing_relations: vec![],
                        incoming_relations: vec![],
                    }
                })
                .collect();
            Self { nodes }
        }
    }

    impl Retriever for MockRetriever {
        fn retrieve(&self, _query: &str, _config: &RagConfig) -> RagResult<RetrievalResult> {
            Ok(RetrievalResult {
                nodes: self.nodes.clone(),
                engrams_matched: self.nodes.len(),
                nodes_activated: self.nodes.len(),
            })
        }
    }

    /// Mock context builder that produces simple text.
    struct MockContextBuilder;

    impl ContextBuilder for MockContextBuilder {
        fn build(&self, result: &RetrievalResult, _config: &RagConfig) -> RagResult<RagContext> {
            Ok(RagContext {
                text: format!("{} nodes in context", result.nodes.len()),
                estimated_tokens: result.nodes.len() * 10,
                nodes_included: result.nodes.len(),
                node_ids: result.nodes.iter().map(|n| n.node_id).collect(),
                node_texts: vec![],
            })
        }
    }

    /// Mock feedback sink that counts calls.
    struct MockFeedback;

    impl FeedbackSink for MockFeedback {
        fn feedback(
            &self,
            context: &RagContext,
            _response: &str,
            _config: &RagConfig,
        ) -> RagResult<FeedbackStats> {
            Ok(FeedbackStats {
                synapses_reinforced: context.node_ids.len(),
                nodes_boosted: context.node_ids.len(),
            })
        }
    }

    fn make_pipeline(with_feedback: bool) -> RagPipeline {
        let retriever = Arc::new(MockRetriever::new(3));
        let feedback: Option<Box<dyn FeedbackSink>> = if with_feedback {
            Some(Box::new(MockFeedback))
        } else {
            None
        };
        RagPipeline::new(
            retriever,
            MockContextBuilder,
            feedback,
            RagConfig::default(),
        )
    }

    #[test]
    fn query_returns_context() {
        let pipeline = make_pipeline(false);
        let ctx = pipeline.query("test query").unwrap();
        assert_eq!(ctx.nodes_included, 3);
        assert!(ctx.text.contains("3 nodes"));
    }

    #[test]
    fn retrieve_returns_raw_results() {
        let pipeline = make_pipeline(false);
        let result = pipeline.retrieve("test").unwrap();
        assert_eq!(result.nodes.len(), 3);
        assert_eq!(result.engrams_matched, 3);
    }

    #[test]
    fn feedback_with_sink() {
        let pipeline = make_pipeline(true);
        let ctx = pipeline.query("test").unwrap();
        let stats = pipeline
            .feedback(&ctx, "response mentioning Node 0")
            .unwrap();
        assert_eq!(stats.synapses_reinforced, 3);
        assert_eq!(stats.nodes_boosted, 3);
    }

    #[test]
    fn feedback_without_sink_returns_default() {
        let pipeline = make_pipeline(false);
        let ctx = pipeline.query("test").unwrap();
        let stats = pipeline.feedback(&ctx, "some response").unwrap();
        assert_eq!(stats.synapses_reinforced, 0);
        assert_eq!(stats.nodes_boosted, 0);
    }

    #[test]
    fn config_getter() {
        let pipeline = make_pipeline(false);
        assert_eq!(
            pipeline.config().token_budget,
            RagConfig::default().token_budget
        );
    }

    #[test]
    fn set_config_updates() {
        let mut pipeline = make_pipeline(false);
        let new_config = RagConfig::fast();
        pipeline.set_config(new_config);
        assert_eq!(pipeline.config().token_budget, 1000);
        assert_eq!(pipeline.config().max_engrams, 5);
    }

    #[test]
    fn debug_impl() {
        let pipeline = make_pipeline(true);
        let debug = format!("{:?}", pipeline);
        assert!(debug.contains("RagPipeline"));
        assert!(debug.contains("has_feedback"));
    }
}
