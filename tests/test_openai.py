#!/usr/bin/env python3
"""Test script for obrain-chat OpenAI-compatible HTTP server.

Usage:
    1. Start the server:  obrain-chat --model <path.gguf> --http 127.0.0.1:8080
    2. Run this script:   pip install openai && python tests/test_openai.py

Tests: /v1/models, /v1/chat/completions (non-stream + stream),
       /v1/conversations (CRUD), /v1/responses (chaining).
"""

import sys
import json
import requests

BASE = "http://127.0.0.1:8080"

def test_models():
    print("=== GET /v1/models ===")
    r = requests.get(f"{BASE}/v1/models")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    data = r.json()
    assert data["object"] == "list"
    assert len(data["data"]) > 0
    print(f"  OK — {len(data['data'])} model(s): {data['data'][0]['id']}")
    return data["data"][0]["id"]

def test_chat_non_streaming(model):
    print("\n=== POST /v1/chat/completions (non-streaming) ===")
    r = requests.post(f"{BASE}/v1/chat/completions", json={
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello in one word."},
        ],
    })
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert "choices" in data, f"Missing choices: {data}"
    assert len(data["choices"]) > 0
    assert data["choices"][0]["finish_reason"] in ("stop", "length")
    content = data["choices"][0]["message"]["content"]
    print(f"  OK — response: {content[:80]}...")
    print(f"  Usage: {data['usage']}")

def test_chat_streaming(model):
    print("\n=== POST /v1/chat/completions (streaming) ===")
    r = requests.post(f"{BASE}/v1/chat/completions", json={
        "model": model,
        "messages": [{"role": "user", "content": "Count to 3."}],
        "stream": True,
    }, stream=True)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    assert "text/event-stream" in r.headers.get("content-type", ""), \
        f"Expected SSE content-type, got {r.headers.get('content-type')}"

    chunks = 0
    got_done = False
    got_finish = False
    for line in r.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("data: "):
            payload = line[6:]
            if payload == "[DONE]":
                got_done = True
                break
            chunk = json.loads(payload)
            assert chunk["object"] == "chat.completion.chunk"
            if chunk["choices"][0].get("finish_reason"):
                got_finish = True
            chunks += 1

    assert got_done, "Never received data: [DONE]"
    assert got_finish, "Never received a chunk with finish_reason"
    print(f"  OK — {chunks} chunks, got finish_reason + [DONE]")

def test_chat_validation():
    print("\n=== POST /v1/chat/completions (validation) ===")
    # Empty messages
    r = requests.post(f"{BASE}/v1/chat/completions", json={"model": "x", "messages": []})
    assert r.status_code == 400, f"Expected 400 for empty messages, got {r.status_code}"
    print("  OK — empty messages returns 400")

    # No user message
    r = requests.post(f"{BASE}/v1/chat/completions", json={
        "model": "x", "messages": [{"role": "system", "content": "hi"}]
    })
    assert r.status_code == 400, f"Expected 400 for no user msg, got {r.status_code}"
    print("  OK — no user message returns 400")

def test_conversations():
    print("\n=== Conversations API ===")
    # Create
    r = requests.post(f"{BASE}/v1/conversations", json={
        "metadata": {"title": "Test conversation"}
    })
    assert r.status_code == 201, f"Create failed: {r.status_code} {r.text}"
    conv = r.json()
    conv_id = conv["id"]
    print(f"  Created: {conv_id}")

    # List
    r = requests.get(f"{BASE}/v1/conversations")
    assert r.status_code == 200
    data = r.json()
    assert any(c["id"] == conv_id for c in data["data"]), "Created conv not in list"
    print(f"  Listed: {len(data['data'])} conversations")

    # Get
    r = requests.get(f"{BASE}/v1/conversations/{conv_id}")
    assert r.status_code == 200
    print(f"  Got: {r.json()['id']}")

    # Add items
    r = requests.post(f"{BASE}/v1/conversations/{conv_id}/items", json={
        "items": [
            {"role": "user", "content": [{"type": "input_text", "text": "Hello"}]},
            {"role": "assistant", "content": [{"type": "output_text", "text": "Hi there!"}]},
        ]
    })
    assert r.status_code == 200, f"Add items failed: {r.status_code} {r.text}"
    print("  Added 2 items")

    # List items
    r = requests.get(f"{BASE}/v1/conversations/{conv_id}/items")
    assert r.status_code == 200
    items = r.json()
    assert len(items["data"]) >= 2, f"Expected >= 2 items, got {len(items['data'])}"
    print(f"  Listed: {len(items['data'])} items")

    print("  OK — Conversations CRUD works")

def test_responses(model):
    print("\n=== POST /v1/responses ===")
    # Non-streaming
    r = requests.post(f"{BASE}/v1/responses", json={
        "model": model,
        "input": "Say hi.",
    })
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert data["object"] == "response"
    assert data["status"] == "completed"
    resp_id = data["id"]
    print(f"  OK — response {resp_id}: {data['output'][0]['content'][0]['text'][:60]}...")

    # Chaining via previous_response_id
    r2 = requests.post(f"{BASE}/v1/responses", json={
        "model": model,
        "input": "What did I just ask?",
        "previous_response_id": resp_id,
    })
    assert r2.status_code == 200, f"Chaining failed: {r2.status_code}"
    print(f"  OK — chained response: {r2.json()['id']}")

def test_sdk_python(model):
    """Test with the official openai Python SDK."""
    print("\n=== SDK Python openai ===")
    try:
        from openai import OpenAI
    except ImportError:
        print("  SKIP — openai not installed (pip install openai)")
        return

    client = OpenAI(base_url=f"{BASE}/v1", api_key="none")

    # Non-streaming
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Say hello."}],
    )
    assert resp.choices[0].message.content
    print(f"  Non-stream OK: {resp.choices[0].message.content[:60]}")

    # Streaming
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Count to 3."}],
        stream=True,
    )
    chunks = 0
    for chunk in stream:
        chunks += 1
    print(f"  Stream OK: {chunks} chunks")

if __name__ == "__main__":
    try:
        requests.get(f"{BASE}/v1/models", timeout=2)
    except requests.ConnectionError:
        print(f"Server not running at {BASE}. Start with:")
        print(f"  obrain-chat --model <path.gguf> --http 127.0.0.1:8080")
        sys.exit(1)

    model = test_models()
    test_chat_validation()
    test_chat_non_streaming(model)
    test_chat_streaming(model)
    test_conversations()
    test_responses(model)
    test_sdk_python(model)

    print("\n" + "=" * 50)
    print("ALL TESTS PASSED")
