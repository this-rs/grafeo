//! Streaming filter for `<think>...</think>` blocks in LLM output.
//!
//! Handles 3 cases:
//!   1. `<think>...</think>` — standard think tags
//!   2. No opening `<think>`, but `</think>` appears — implicit thinking (buffer until `</think>`)
//!   3. No think tags at all — passthrough
//!
//! Strategy: buffer initial tokens. If `</think>` appears, discard everything before it.
//! If we accumulate enough tokens without seeing `</think>`, flush as real content.

/// Max words to buffer in probing mode before deciding it's passthrough.
const PROBE_LIMIT: usize = 500;

/// State machine for filtering `<think>...</think>` blocks during streaming.
pub struct ThinkFilter {
    state: ThinkState,
    buffer: String,
    printed_any: bool,
}

enum ThinkState {
    /// Buffering initial tokens, waiting to see if `</think>` or `<think>` appears.
    Probing,
    /// Inside a `<think>...</think>` block, discarding content.
    InThink,
    /// Passthrough mode — no think tags, emit everything.
    Passthrough,
}

impl ThinkFilter {
    pub fn new() -> Self {
        Self {
            state: ThinkState::Probing,
            buffer: String::new(),
            printed_any: false,
        }
    }

    pub fn feed(&mut self, token: &str) -> String {
        self.buffer.push_str(token);
        let mut output = String::new();

        loop {
            match self.state {
                ThinkState::Probing => {
                    // Check for </think> (implicit end of thinking without <think>)
                    if let Some(end) = self.buffer.find("</think>") {
                        // Everything before </think> was thinking — discard it
                        self.buffer = self.buffer[end + 8..].to_string();
                        self.state = ThinkState::Passthrough;
                        continue;
                    }
                    // Check for explicit <think>
                    if let Some(start) = self.buffer.find("<think>") {
                        // Emit anything before <think> (unlikely at start)
                        output.push_str(&self.buffer[..start]);
                        self.buffer = self.buffer[start + 7..].to_string();
                        self.state = ThinkState::InThink;
                        continue;
                    }
                    // Check for partial tags — keep buffering
                    if self.buffer.contains("</thi") || self.buffer.contains("<thin") {
                        break; // wait for more tokens
                    }
                    // Count "tokens" (rough: split by whitespace)
                    let token_count = self.buffer.split_whitespace().count();
                    if token_count > PROBE_LIMIT {
                        // No think tags after PROBE_LIMIT tokens — it's real content
                        self.state = ThinkState::Passthrough;
                        continue;
                    }
                    break;
                }
                ThinkState::InThink => {
                    if let Some(end) = self.buffer.find("</think>") {
                        self.buffer = self.buffer[end + 8..].to_string();
                        self.state = ThinkState::Passthrough;
                        continue;
                    }
                    // Partial </think> match — keep buffering
                    if self.buffer.ends_with("</") || self.buffer.contains("</thi") {
                        break;
                    }
                    // Discard accumulated thinking content (keep last 20 chars for partial match)
                    if self.buffer.len() > 100 {
                        // Use char boundary to avoid panic on multi-byte UTF-8 (β, α, ₐ, etc.)
                        let keep: String = self
                            .buffer
                            .chars()
                            .rev()
                            .take(20)
                            .collect::<Vec<_>>()
                            .into_iter()
                            .rev()
                            .collect();
                        self.buffer = keep;
                    }
                    break;
                }
                ThinkState::Passthrough => {
                    // Check for new <think> blocks mid-stream
                    if let Some(start) = self.buffer.find("<think>") {
                        output.push_str(&self.buffer[..start]);
                        self.buffer = self.buffer[start + 7..].to_string();
                        self.state = ThinkState::InThink;
                        continue;
                    }
                    // Hold partial <think match
                    if self.buffer.ends_with('<')
                        || self.buffer.ends_with("<t")
                        || self.buffer.ends_with("<th")
                        || self.buffer.ends_with("<thi")
                        || self.buffer.ends_with("<thin")
                        || self.buffer.ends_with("<think")
                    {
                        let hold_from = self.buffer.rfind('<').unwrap_or(self.buffer.len());
                        output.push_str(&self.buffer[..hold_from]);
                        self.buffer = self.buffer[hold_from..].to_string();
                        break;
                    }
                    output.push_str(&self.buffer);
                    self.buffer.clear();
                    break;
                }
            }
        }

        // Clean role artifacts from beginning of output
        if !self.printed_any && !output.is_empty() {
            let trimmed = output.trim_start();
            for prefix in &["system\n", "assistant\n", "user\n"] {
                if trimmed.starts_with(prefix) {
                    output = trimmed[prefix.len()..].to_string();
                    break;
                }
            }
            if !output.trim().is_empty() {
                self.printed_any = true;
            }
        }

        output
    }

    pub fn flush(&mut self) -> String {
        match self.state {
            ThinkState::InThink => String::new(), // discard unclosed think
            ThinkState::Probing => {
                // Never saw </think> or enough tokens — emit as-is
                let out = std::mem::take(&mut self.buffer);
                self.clean_role_prefix(out)
            }
            ThinkState::Passthrough => {
                let out = std::mem::take(&mut self.buffer);
                self.clean_role_prefix(out)
            }
        }
    }

    fn clean_role_prefix(&self, s: String) -> String {
        let trimmed = s.trim_start();
        for prefix in &["system\n", "assistant\n"] {
            if trimmed.starts_with(prefix) {
                return trimmed[prefix.len()..].to_string();
            }
        }
        s
    }
}

impl Default for ThinkFilter {
    fn default() -> Self {
        Self::new()
    }
}

/// Remove `<think>...</think>` blocks from a completed string.
/// Also strips `<|im_start|>` and `<|im_end|>` special tokens.
pub fn strip_think_tags(s: &str) -> String {
    let mut result = s.to_string();
    if let Some(end) = result.find("</think>") {
        if result[..end].find("<think>").is_none() {
            result = result[end + 8..].to_string();
        }
    }
    while let Some(start) = result.find("<think>") {
        if let Some(end) = result[start..].find("</think>") {
            result = format!("{}{}", &result[..start], &result[start + end + 8..]);
        } else {
            result = result[..start].to_string();
            break;
        }
    }
    // Strip special tokens from all known chat template families
    result = result
        .replace("<|im_start|>", "").replace("<|im_end|>", "")           // ChatML (Qwen3)
        .replace("<|start_header_id|>", "").replace("<|end_header_id|>", "") // Llama 3
        .replace("<|eot_id|>", "")                                       // Llama 3
        .replace("<|endoftext|>", "");                                   // General
    result.trim().to_string()
}

/// Truncate a string to `max` characters, appending "..." if truncated.
/// Safe for multi-byte UTF-8.
pub fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let t: String = s.chars().take(max).collect();
        format!("{t}...")
    }
}
