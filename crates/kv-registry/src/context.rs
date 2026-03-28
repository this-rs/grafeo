//! Context structures for query pipeline.

use obrain_common::types::NodeId;
use std::collections::{HashMap, HashSet};

pub struct ContextNode {
    pub id: NodeId,
    pub token_start: i32,
    pub token_end: i32,
    pub bank: u32, // 0=core, 1=relations, 2=2-hop, 3=background
}

pub struct QueryContext {
    pub nodes: Vec<ContextNode>,
    #[allow(dead_code)]
    pub adjacency: HashMap<NodeId, HashSet<NodeId>>,
    pub total_tokens: i32,
    pub header_tokens: i32,
}
