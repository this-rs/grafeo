//! # Reasoning Trees
//!
//! Structured multi-step reasoning stored as sub-graphs in obrain.
//! Each reasoning tree decomposes a problem into hypotheses, evaluations,
//! and a final synthesis — all as graph nodes with energy and synapses.
//!
//! ## Node Types
//!
//! - **Root** — the initial question or problem statement
//! - **Hypothesis** — a proposed approach or explanation
//! - **Evaluation** — assessment of a hypothesis with evidence and score
//! - **Synthesis** — final aggregation of best branches into a decision
//!
//! ## Lifecycle
//!
//! 1. Create tree with root question
//! 2. Branch into hypotheses
//! 3. Evaluate each hypothesis
//! 4. Synthesize the best results
//! 5. Over time: frequently-used trees consolidate into patterns
//! 6. Unused trees lose energy and dissolve
//!
//! ## Relations
//!
//! - `:ReasoningNode -[:BRANCH {weight}]-> :ReasoningNode`
//! - `:ReasoningNode -[:EVALUATES]-> :ReasoningNode`
//! - `:ReasoningNode -[:SYNTHESIZES]-> :ReasoningNode`
//! - `:ReasoningNode -[:ABOUT]-> <any entity>`

use std::collections::HashMap;

use obrain_common::types::NodeId;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Label for reasoning nodes in the graph.
pub const LABEL_REASONING_NODE: &str = "ReasoningNode";

/// Edge type for branching (root → hypothesis, hypothesis → sub-hypothesis).
pub const EDGE_BRANCH: &str = "BRANCH";

/// Edge type for evaluation.
pub const EDGE_EVALUATES: &str = "EVALUATES";

/// Edge type for synthesis.
pub const EDGE_SYNTHESIZES: &str = "SYNTHESIZES";

/// Edge type linking a reasoning node to the entity it reasons about.
pub const EDGE_ABOUT: &str = "ABOUT";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// The type of a reasoning node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReasoningNodeType {
    Root,
    Hypothesis,
    Evaluation,
    Synthesis,
}

impl ReasoningNodeType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Root => "root",
            Self::Hypothesis => "hypothesis",
            Self::Evaluation => "evaluation",
            Self::Synthesis => "synthesis",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "root" => Some(Self::Root),
            "hypothesis" => Some(Self::Hypothesis),
            "evaluation" => Some(Self::Evaluation),
            "synthesis" => Some(Self::Synthesis),
            _ => None,
        }
    }
}

/// The type of reasoning being conducted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningType {
    /// Breaking down a problem into components.
    Analytical,
    /// Comparing multiple options.
    Comparative,
    /// Generating new ideas or approaches.
    Creative,
    /// Judging the quality or fitness of something.
    Evaluative,
}

/// A node in a reasoning tree.
#[derive(Debug, Clone)]
pub struct ReasoningNode {
    /// Node id in the graph (set after persistence).
    pub node_id: Option<NodeId>,
    /// Type of this reasoning node.
    pub node_type: ReasoningNodeType,
    /// Content of the reasoning (question, hypothesis, evaluation text).
    pub content: String,
    /// Confidence score [0, 1].
    pub confidence: f64,
    /// Energy (participates in engram lifecycle).
    pub energy: f64,
    /// Evidence supporting this node (for evaluations).
    pub evidence: Vec<String>,
    /// Entities this node reasons about (links to graph nodes).
    pub about: Vec<NodeId>,
}

impl ReasoningNode {
    pub fn root(content: impl Into<String>) -> Self {
        Self {
            node_id: None,
            node_type: ReasoningNodeType::Root,
            content: content.into(),
            confidence: 1.0,
            energy: 1.0,
            evidence: Vec::new(),
            about: Vec::new(),
        }
    }

    pub fn hypothesis(content: impl Into<String>, confidence: f64) -> Self {
        Self {
            node_id: None,
            node_type: ReasoningNodeType::Hypothesis,
            content: content.into(),
            confidence: confidence.clamp(0.0, 1.0),
            energy: 1.0,
            evidence: Vec::new(),
            about: Vec::new(),
        }
    }

    pub fn evaluation(content: impl Into<String>, confidence: f64, evidence: Vec<String>) -> Self {
        Self {
            node_id: None,
            node_type: ReasoningNodeType::Evaluation,
            content: content.into(),
            confidence: confidence.clamp(0.0, 1.0),
            energy: 1.0,
            evidence,
            about: Vec::new(),
        }
    }

    pub fn synthesis(content: impl Into<String>, confidence: f64) -> Self {
        Self {
            node_id: None,
            node_type: ReasoningNodeType::Synthesis,
            content: content.into(),
            confidence: confidence.clamp(0.0, 1.0),
            energy: 1.0,
            evidence: Vec::new(),
            about: Vec::new(),
        }
    }

    pub fn with_about(mut self, entities: Vec<NodeId>) -> Self {
        self.about = entities;
        self
    }
}

/// A complete reasoning tree.
#[derive(Debug, Clone)]
pub struct ReasoningTree {
    /// The root node of the tree.
    pub root: ReasoningNode,
    /// All nodes indexed by their position in the tree.
    pub nodes: Vec<ReasoningNode>,
    /// Edges: (parent_idx, child_idx, edge_type, weight).
    pub edges: Vec<(usize, usize, EdgeType, f64)>,
    /// Type of reasoning.
    pub reasoning_type: ReasoningType,
    /// Number of times this tree (or similar) has been reused.
    pub reuse_count: u32,
}

/// Edge types within a reasoning tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeType {
    Branch,
    Evaluates,
    Synthesizes,
}

impl EdgeType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Branch => EDGE_BRANCH,
            Self::Evaluates => EDGE_EVALUATES,
            Self::Synthesizes => EDGE_SYNTHESIZES,
        }
    }
}

// ---------------------------------------------------------------------------
// Tree Builder
// ---------------------------------------------------------------------------

/// Builder for constructing reasoning trees incrementally.
#[derive(Debug)]
pub struct ReasoningTreeBuilder {
    nodes: Vec<ReasoningNode>,
    edges: Vec<(usize, usize, EdgeType, f64)>,
    reasoning_type: ReasoningType,
}

impl ReasoningTreeBuilder {
    /// Start a new reasoning tree with the root question.
    pub fn new(root: ReasoningNode, reasoning_type: ReasoningType) -> Self {
        assert_eq!(root.node_type, ReasoningNodeType::Root);
        Self {
            nodes: vec![root],
            edges: Vec::new(),
            reasoning_type,
        }
    }

    /// Add a hypothesis branching from a parent node.
    /// Returns the index of the new node.
    pub fn add_hypothesis(&mut self, parent_idx: usize, node: ReasoningNode, weight: f64) -> usize {
        assert_eq!(node.node_type, ReasoningNodeType::Hypothesis);
        assert!(parent_idx < self.nodes.len());
        let idx = self.nodes.len();
        self.nodes.push(node);
        self.edges.push((parent_idx, idx, EdgeType::Branch, weight));
        idx
    }

    /// Add an evaluation of a hypothesis.
    /// Returns the index of the new node.
    pub fn add_evaluation(&mut self, target_idx: usize, node: ReasoningNode) -> usize {
        assert_eq!(node.node_type, ReasoningNodeType::Evaluation);
        assert!(target_idx < self.nodes.len());
        let idx = self.nodes.len();
        self.nodes.push(node);
        self.edges
            .push((idx, target_idx, EdgeType::Evaluates, 1.0));
        idx
    }

    /// Add a synthesis node aggregating multiple branches.
    /// Returns the index of the synthesis node.
    pub fn add_synthesis(&mut self, source_indices: &[usize], node: ReasoningNode) -> usize {
        assert_eq!(node.node_type, ReasoningNodeType::Synthesis);
        let idx = self.nodes.len();
        self.nodes.push(node);
        for &src in source_indices {
            assert!(src < idx);
            self.edges
                .push((idx, src, EdgeType::Synthesizes, 1.0));
        }
        idx
    }

    /// Build the final tree.
    pub fn build(self) -> ReasoningTree {
        let root = self.nodes[0].clone();
        ReasoningTree {
            root,
            nodes: self.nodes,
            edges: self.edges,
            reasoning_type: self.reasoning_type,
            reuse_count: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Pattern Detection
// ---------------------------------------------------------------------------

/// A consolidated reasoning pattern (frequent tree structure).
#[derive(Debug, Clone)]
pub struct ReasoningPattern {
    /// Template structure (generalized from multiple similar trees).
    pub template: ReasoningTree,
    /// How many trees contributed to this pattern.
    pub consolidation_count: u32,
    /// Topics this pattern applies to (node ids from :ABOUT links).
    pub topics: Vec<NodeId>,
    /// Energy (patterns with low reuse lose energy).
    pub energy: f64,
}

/// Check if two reasoning trees are structurally similar (for consolidation).
pub fn structural_similarity(a: &ReasoningTree, b: &ReasoningTree) -> f64 {
    // Compare by: number of nodes per type, depth, edge structure
    let a_types = type_counts(&a.nodes);
    let b_types = type_counts(&b.nodes);

    let mut matches = 0.0;
    let mut total = 0.0;
    for node_type in &[
        ReasoningNodeType::Root,
        ReasoningNodeType::Hypothesis,
        ReasoningNodeType::Evaluation,
        ReasoningNodeType::Synthesis,
    ] {
        let a_count = a_types.get(node_type).copied().unwrap_or(0) as f64;
        let b_count = b_types.get(node_type).copied().unwrap_or(0) as f64;
        let max_count = a_count.max(b_count);
        if max_count > 0.0 {
            matches += a_count.min(b_count) / max_count;
            total += 1.0;
        }
    }

    // Also compare reasoning type
    let type_bonus = if a.reasoning_type == b.reasoning_type {
        0.2
    } else {
        0.0
    };

    if total == 0.0 {
        return type_bonus;
    }

    (matches / total) * 0.8 + type_bonus
}

fn type_counts(nodes: &[ReasoningNode]) -> HashMap<ReasoningNodeType, usize> {
    let mut counts = HashMap::new();
    for node in nodes {
        *counts.entry(node.node_type).or_insert(0) += 1;
    }
    counts
}

/// Check if a tree should be consolidated into a pattern.
/// Returns true if tree has been reused >= threshold times.
pub fn should_consolidate(tree: &ReasoningTree, threshold: u32) -> bool {
    tree.reuse_count >= threshold
}

/// Check if a pattern should dissolve (energy too low).
pub fn should_dissolve_pattern(pattern: &ReasoningPattern, min_energy: f64) -> bool {
    pattern.energy < min_energy
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_simple_tree() {
        let mut builder =
            ReasoningTreeBuilder::new(ReasoningNode::root("Why is this test failing?"), ReasoningType::Analytical);

        let h1 = builder.add_hypothesis(
            0,
            ReasoningNode::hypothesis("Maybe a race condition", 0.7),
            1.0,
        );
        let h2 = builder.add_hypothesis(
            0,
            ReasoningNode::hypothesis("Maybe wrong input data", 0.5),
            0.8,
        );

        builder.add_evaluation(
            h1,
            ReasoningNode::evaluation("Confirmed: async ordering issue", 0.9, vec!["log trace shows interleaving".into()]),
        );
        builder.add_evaluation(
            h2,
            ReasoningNode::evaluation("Ruled out: input is correct", 0.1, vec![]),
        );

        let tree = builder.build();
        assert_eq!(tree.nodes.len(), 5); // root + 2 hypotheses + 2 evaluations
        assert_eq!(tree.edges.len(), 4);
        assert_eq!(tree.reasoning_type, ReasoningType::Analytical);
    }

    #[test]
    fn test_synthesis() {
        let mut builder =
            ReasoningTreeBuilder::new(ReasoningNode::root("Best approach?"), ReasoningType::Comparative);

        let h1 = builder.add_hypothesis(0, ReasoningNode::hypothesis("Approach A", 0.8), 1.0);
        let h2 = builder.add_hypothesis(0, ReasoningNode::hypothesis("Approach B", 0.6), 1.0);

        builder.add_synthesis(
            &[h1, h2],
            ReasoningNode::synthesis("Combine A's core with B's error handling", 0.85),
        );

        let tree = builder.build();
        assert_eq!(tree.nodes.len(), 4);
        // Synthesis node has 2 edges (synthesizes h1 and h2)
        let synth_edges: Vec<_> = tree
            .edges
            .iter()
            .filter(|(_, _, t, _)| *t == EdgeType::Synthesizes)
            .collect();
        assert_eq!(synth_edges.len(), 2);
    }

    #[test]
    fn test_structural_similarity_identical() {
        let builder =
            ReasoningTreeBuilder::new(ReasoningNode::root("Q1"), ReasoningType::Analytical);
        let tree_a = builder.build();

        let builder =
            ReasoningTreeBuilder::new(ReasoningNode::root("Q2"), ReasoningType::Analytical);
        let tree_b = builder.build();

        let sim = structural_similarity(&tree_a, &tree_b);
        assert!(sim > 0.9); // same structure + same type
    }

    #[test]
    fn test_structural_similarity_different() {
        let mut builder =
            ReasoningTreeBuilder::new(ReasoningNode::root("Q1"), ReasoningType::Analytical);
        builder.add_hypothesis(0, ReasoningNode::hypothesis("H1", 0.5), 1.0);
        builder.add_hypothesis(0, ReasoningNode::hypothesis("H2", 0.5), 1.0);
        builder.add_hypothesis(0, ReasoningNode::hypothesis("H3", 0.5), 1.0);
        let tree_a = builder.build();

        let builder =
            ReasoningTreeBuilder::new(ReasoningNode::root("Q2"), ReasoningType::Creative);
        let tree_b = builder.build();

        let sim = structural_similarity(&tree_a, &tree_b);
        assert!(sim < 0.5); // very different structure + different type
    }

    #[test]
    fn test_consolidation_threshold() {
        let builder =
            ReasoningTreeBuilder::new(ReasoningNode::root("Q"), ReasoningType::Analytical);
        let mut tree = builder.build();

        assert!(!should_consolidate(&tree, 5));
        tree.reuse_count = 5;
        assert!(should_consolidate(&tree, 5));
    }

    #[test]
    fn test_confidence_clamping() {
        let node = ReasoningNode::hypothesis("test", 1.5);
        assert_eq!(node.confidence, 1.0);

        let node = ReasoningNode::hypothesis("test", -0.5);
        assert_eq!(node.confidence, 0.0);
    }
}
