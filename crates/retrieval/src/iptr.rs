//! IPTR — In-Process Tool Runtime.
//!
//! Demand-driven inline tool execution during generation, triggered by
//! high entropy (model hesitation). Each tool runs within a strict time budget
//! and returns logit biases to steer the next token.
//!
//! Architecture:
//! - `InProcessTool` trait: any tool that can run inline (< 5ms)
//! - `IptrDispatcher`: manages tools, cooldown, and dispatch
//! - `ToolContext`: input to a tool (recent text, entropy, state)
//! - `ToolResult`: output from a tool (token_id → logit bias)
//!
//! The dispatcher enforces a cooldown between triggers to prevent
//! thrashing on sustained high-entropy sequences.

use std::collections::HashMap;
use std::time::Instant;

/// Context provided to an in-process tool when triggered.
#[derive(Debug, Clone)]
pub struct ToolContext {
    /// Recent generated text (last ~50 tokens decoded to string).
    pub recent_text: String,
    /// Current token entropy that triggered the dispatch.
    pub entropy: f32,
    /// Node IDs already retrieved in this generation (avoid duplicates).
    pub already_retrieved: Vec<u64>,
    /// Current token position in the generation.
    pub token_position: u32,
}

/// Result from an in-process tool execution.
#[derive(Debug, Clone)]
pub struct ToolResult {
    /// Token ID → logit bias to apply at the next sampling step.
    pub biases: HashMap<u32, f32>,
    /// Name of the tool that produced this result.
    pub tool_name: String,
    /// Execution latency in microseconds.
    pub latency_us: u64,
    /// Concepts/entities found (for logging).
    pub concepts_found: Vec<String>,
}

impl ToolResult {
    /// Empty result (no biases to apply).
    pub fn empty(tool_name: &str) -> Self {
        Self {
            biases: HashMap::new(),
            tool_name: tool_name.to_string(),
            latency_us: 0,
            concepts_found: Vec::new(),
        }
    }

    /// Merge another result into this one. On conflict, biases are summed.
    pub fn merge(&mut self, other: &ToolResult) {
        for (&token_id, &bias) in &other.biases {
            *self.biases.entry(token_id).or_insert(0.0) += bias;
        }
        self.concepts_found.extend(other.concepts_found.iter().cloned());
    }

    /// Number of biased tokens.
    pub fn n_biases(&self) -> usize {
        self.biases.len()
    }
}

/// Trait for tools that can execute inline during generation.
///
/// Implementors must be fast (< budget_ms()) and deterministic.
/// They receive a `ToolContext` and return logit biases.
pub trait InProcessTool: Send + Sync {
    /// Tool name (for logging and identification).
    fn name(&self) -> &str;

    /// Maximum execution time in milliseconds. The dispatcher will skip
    /// this tool if a previous call exceeded the budget.
    fn budget_ms(&self) -> u64;

    /// Execute the tool and return logit biases.
    ///
    /// May return an empty result if no relevant information is found.
    fn execute(&self, context: &ToolContext) -> ToolResult;
}

/// Dispatches in-process tools during generation.
///
/// Manages cooldown between triggers and enforces time budgets.
pub struct IptrDispatcher {
    /// Registered tools (executed in order, results merged).
    tools: Vec<Box<dyn InProcessTool>>,
    /// Minimum tokens between two consecutive triggers.
    cooldown_tokens: usize,
    /// Token position of the last trigger.
    last_trigger_pos: Option<u32>,
    /// Total dispatches in this generation.
    pub dispatch_count: u32,
    /// Total tool execution time in microseconds.
    pub total_latency_us: u64,
    /// All concepts found across dispatches (for dedup in ToolContext).
    retrieved_node_ids: Vec<u64>,
}

impl IptrDispatcher {
    /// Create a new dispatcher with the given cooldown.
    pub fn new(cooldown_tokens: usize) -> Self {
        Self {
            tools: Vec::new(),
            cooldown_tokens,
            last_trigger_pos: None,
            dispatch_count: 0,
            total_latency_us: 0,
            retrieved_node_ids: Vec::new(),
        }
    }

    /// Create with default cooldown (16 tokens).
    pub fn with_defaults() -> Self {
        Self::new(16)
    }

    /// Register an in-process tool.
    pub fn register_tool(&mut self, tool: Box<dyn InProcessTool>) {
        self.tools.push(tool);
    }

    /// Check if dispatch is allowed (cooldown respected).
    fn cooldown_ok(&self, token_position: u32) -> bool {
        match self.last_trigger_pos {
            None => true,
            Some(last) => (token_position as i64 - last as i64) >= self.cooldown_tokens as i64,
        }
    }

    /// Dispatch all registered tools if cooldown allows.
    ///
    /// Returns `Some(merged_result)` if any tool produced biases,
    /// `None` if cooldown prevents dispatch or no tools registered.
    pub fn dispatch(&mut self, context: &ToolContext) -> Option<ToolResult> {
        if self.tools.is_empty() {
            return None;
        }

        if !self.cooldown_ok(context.token_position) {
            return None;
        }

        // Build context with accumulated retrieved node IDs
        let mut ctx = context.clone();
        ctx.already_retrieved = self.retrieved_node_ids.clone();

        let mut merged = ToolResult::empty("iptr_merged");
        let mut any_result = false;

        for tool in &self.tools {
            let start = Instant::now();
            let result = tool.execute(&ctx);
            let elapsed_us = start.elapsed().as_micros() as u64;

            if elapsed_us > tool.budget_ms() * 1000 {
                eprintln!(
                    "  [IPTR] ⚠ {} exceeded budget: {}µs > {}ms",
                    tool.name(),
                    elapsed_us,
                    tool.budget_ms()
                );
            }

            if !result.biases.is_empty() {
                any_result = true;
                merged.merge(&result);
                merged.latency_us += elapsed_us;
            }

            self.total_latency_us += elapsed_us;
        }

        if any_result {
            self.last_trigger_pos = Some(context.token_position);
            self.dispatch_count += 1;
            Some(merged)
        } else {
            None
        }
    }

    /// Get dispatch statistics.
    pub fn stats(&self) -> (u32, u64) {
        (self.dispatch_count, self.total_latency_us)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock tool that always returns a fixed bias.
    struct MockTool {
        name: String,
        bias_token: u32,
        bias_value: f32,
    }

    impl InProcessTool for MockTool {
        fn name(&self) -> &str { &self.name }
        fn budget_ms(&self) -> u64 { 5 }
        fn execute(&self, _ctx: &ToolContext) -> ToolResult {
            let mut biases = HashMap::new();
            biases.insert(self.bias_token, self.bias_value);
            ToolResult {
                biases,
                tool_name: self.name.clone(),
                latency_us: 100,
                concepts_found: vec!["test_concept".into()],
            }
        }
    }

    /// Mock tool that returns empty (no relevant info found).
    struct EmptyTool;
    impl InProcessTool for EmptyTool {
        fn name(&self) -> &str { "empty" }
        fn budget_ms(&self) -> u64 { 5 }
        fn execute(&self, _ctx: &ToolContext) -> ToolResult {
            ToolResult::empty("empty")
        }
    }

    fn make_context(pos: u32) -> ToolContext {
        ToolContext {
            recent_text: "Thomas habite à".into(),
            entropy: 4.0,
            already_retrieved: Vec::new(),
            token_position: pos,
        }
    }

    #[test]
    fn test_dispatch_basic() {
        let mut dispatcher = IptrDispatcher::new(16);
        dispatcher.register_tool(Box::new(MockTool {
            name: "graph".into(),
            bias_token: 42,
            bias_value: 3.0,
        }));

        let result = dispatcher.dispatch(&make_context(0));
        assert!(result.is_some());
        let r = result.unwrap();
        assert_eq!(r.biases.get(&42), Some(&3.0));
        assert_eq!(dispatcher.dispatch_count, 1);
    }

    #[test]
    fn test_cooldown_prevents_rapid_dispatch() {
        let mut dispatcher = IptrDispatcher::new(16);
        dispatcher.register_tool(Box::new(MockTool {
            name: "graph".into(),
            bias_token: 42,
            bias_value: 3.0,
        }));

        // First dispatch at pos=0: OK
        assert!(dispatcher.dispatch(&make_context(0)).is_some());

        // Second dispatch at pos=5: blocked by cooldown (< 16)
        assert!(dispatcher.dispatch(&make_context(5)).is_none());

        // Third dispatch at pos=20: OK (>= 16 tokens since last)
        assert!(dispatcher.dispatch(&make_context(20)).is_some());
        assert_eq!(dispatcher.dispatch_count, 2);
    }

    #[test]
    fn test_merge_multiple_tools() {
        let mut dispatcher = IptrDispatcher::new(16);
        dispatcher.register_tool(Box::new(MockTool {
            name: "tool_a".into(),
            bias_token: 10,
            bias_value: 2.0,
        }));
        dispatcher.register_tool(Box::new(MockTool {
            name: "tool_b".into(),
            bias_token: 20,
            bias_value: 1.5,
        }));

        let result = dispatcher.dispatch(&make_context(0)).unwrap();
        assert_eq!(result.biases.len(), 2);
        assert_eq!(result.biases.get(&10), Some(&2.0));
        assert_eq!(result.biases.get(&20), Some(&1.5));
        assert_eq!(result.concepts_found.len(), 2);
    }

    #[test]
    fn test_empty_tool_no_dispatch() {
        let mut dispatcher = IptrDispatcher::new(16);
        dispatcher.register_tool(Box::new(EmptyTool));

        // Tool returns empty → dispatch returns None
        assert!(dispatcher.dispatch(&make_context(0)).is_none());
        assert_eq!(dispatcher.dispatch_count, 0);
    }

    #[test]
    fn test_no_tools_registered() {
        let mut dispatcher = IptrDispatcher::new(16);
        assert!(dispatcher.dispatch(&make_context(0)).is_none());
    }

    #[test]
    fn test_tool_result_merge_sums_overlapping_biases() {
        let mut r1 = ToolResult {
            biases: HashMap::from([(42, 2.0), (43, 1.0)]),
            tool_name: "a".into(),
            latency_us: 100,
            concepts_found: vec!["Lyon".into()],
        };
        let r2 = ToolResult {
            biases: HashMap::from([(42, 1.5), (44, 0.5)]),
            tool_name: "b".into(),
            latency_us: 50,
            concepts_found: vec!["Marc".into()],
        };
        r1.merge(&r2);

        assert_eq!(r1.biases.get(&42), Some(&3.5)); // 2.0 + 1.5
        assert_eq!(r1.biases.get(&43), Some(&1.0));
        assert_eq!(r1.biases.get(&44), Some(&0.5));
        assert_eq!(r1.concepts_found, vec!["Lyon", "Marc"]);
    }
}
