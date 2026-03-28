//! HTTP helpers for llama.cpp server mode (behind feature flag).

use anyhow::{bail, Context, Result};
use serde_json::json;
use std::io::{self, BufRead, BufReader, Write};
use think_filter::ThinkFilter;

#[allow(dead_code)]
pub const DEFAULT_SERVER: &str = "http://localhost:8090";

pub fn check_server(client: &reqwest::blocking::Client, server: &str) -> Result<()> {
    let resp = client.get(format!("{server}/health")).send()
        .context(format!("Cannot reach {server}"))?;
    if !resp.status().is_success() {
        bail!("Server unhealthy: {}", resp.status());
    }
    Ok(())
}

pub fn tokenize(client: &reqwest::blocking::Client, server: &str, text: &str) -> Result<i32> {
    let resp = client
        .post(format!("{server}/tokenize"))
        .json(&json!({ "content": text, "special": true }))
        .send()?;
    let body: serde_json::Value = resp.json()?;
    Ok(body["tokens"].as_array().context("no tokens")?.len() as i32)
}

pub fn set_attn_mask(client: &reqwest::blocking::Client, server: &str, mask: &[f32], positions: &[i32]) -> Result<()> {
    let resp = client
        .post(format!("{server}/attn-mask"))
        .json(&json!({ "mask": mask, "positions": positions }))
        .send()
        .context("Failed to send attention mask")?;
    let body: serde_json::Value = resp.json()
        .context("Failed to parse mask response")?;
    if body["success"].as_bool() != Some(true) {
        bail!("set mask failed: {body}");
    }
    Ok(())
}

pub fn clear_attn_mask(client: &reqwest::blocking::Client, server: &str) -> Result<()> {
    client.post(format!("{server}/attn-mask"))
        .json(&json!({ "mask": null }))
        .send()?;
    Ok(())
}

/// Streaming completion via llama.cpp SSE endpoint.
/// Prints tokens to stdout as they arrive, returns the full text.
pub fn complete_streaming(client: &reqwest::blocking::Client, server: &str, prompt: &str, n_predict: i32) -> Result<String> {
    let resp = client
        .post(format!("{server}/completion"))
        .json(&json!({
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": 0.1,
            "top_p": 0.9,
            "cache_prompt": true,
            "special": true,
            "stream": true,
            "stop": ["<|im_start|>", "<|im_end|>", "<|endoftext|>"],
        }))
        .send()
        .context("completion request failed")?;

    print!("assistant> ");
    io::stdout().flush()?;

    let mut full_text = String::new();
    let mut filter = ThinkFilter::new();
    let reader = BufReader::new(resp);

    for line in reader.lines() {
        let line = line?;
        if !line.starts_with("data: ") { continue; }
        let data = &line[6..];
        if data == "[DONE]" { break; }

        if let Ok(chunk) = serde_json::from_str::<serde_json::Value>(data) {
            if let Some(token) = chunk["content"].as_str() {
                let visible = filter.feed(token);
                if !visible.is_empty() {
                    print!("{}", visible);
                    io::stdout().flush()?;
                    full_text.push_str(&visible);
                }
            }
            if chunk["stop"].as_bool() == Some(true) { break; }
        }
    }
    // Flush remaining
    let remaining = filter.flush();
    if !remaining.is_empty() {
        print!("{}", remaining);
        full_text.push_str(&remaining);
    }
    println!("\n");

    Ok(full_text)
}

/// Non-streaming completion (kept for internal use like tokenization tests).
pub fn complete(client: &reqwest::blocking::Client, server: &str, prompt: &str, n_predict: i32) -> Result<String> {
    let resp = client
        .post(format!("{server}/completion"))
        .json(&json!({
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": 0.1,
            "top_p": 0.9,
            "cache_prompt": true,
            "special": true,
            "stop": ["<|im_start|>", "<|im_end|>", "<|endoftext|>"],
        }))
        .send()?;
    let body: serde_json::Value = resp.json()?;
    Ok(body["content"].as_str().unwrap_or("[no content]").to_string())
}

/// Streaming chat completion via /v1/chat/completions (SSE).
pub fn chat_complete(
    client: &reqwest::blocking::Client,
    server: &str,
    system: &str,
    user_msg: &str,
    n_predict: i32,
) -> Result<String> {
    let mut messages = Vec::new();
    if !system.is_empty() {
        messages.push(json!({"role": "system", "content": format!(
            "You have access to a knowledge graph. Answer based on the following data:\n\n{}\n\nAnswer concisely in the user's language. /no_think",
            system
        )}));
    }
    messages.push(json!({"role": "user", "content": user_msg}));
    let resp = client
        .post(format!("{server}/v1/chat/completions"))
        .json(&json!({
            "messages": messages,
            "max_tokens": n_predict,
            "temperature": 0.1,
            "top_p": 0.9,
            "stream": true,
        }))
        .send()?;

    print!("assistant> ");
    io::stdout().flush()?;

    let mut full_text = String::new();
    let mut filter = ThinkFilter::new();
    let reader = BufReader::new(resp);

    for line in reader.lines() {
        let line = line?;
        if !line.starts_with("data: ") { continue; }
        let data = &line[6..];
        if data == "[DONE]" { break; }

        if let Ok(chunk) = serde_json::from_str::<serde_json::Value>(data) {
            if let Some(delta) = chunk["choices"][0]["delta"]["content"].as_str() {
                let visible = filter.feed(delta);
                if !visible.is_empty() {
                    print!("{}", visible);
                    io::stdout().flush()?;
                    full_text.push_str(&visible);
                }
            }
        }
    }
    let remaining = filter.flush();
    if !remaining.is_empty() {
        print!("{}", remaining);
        full_text.push_str(&remaining);
    }
    println!("\n");

    Ok(full_text)
}
