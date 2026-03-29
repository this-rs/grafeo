#!/usr/bin/env python3
"""
B5 — Benchmark A/B: per-head routing vs masque broadcast vs baseline.

Drives obrain-chat interactively via pexpect across 3 modes:
  1. baseline: no persona/graph → pure LLM
  2. phase_a: --persona with fixed BankConfig (broadcast mask)
  3. phase_b: --persona --head-router (per-head α-routing)
"""

import pexpect
import sys
import os
import re
import time
import json

# ── Config ──────────────────────────────────────────────────────
BINARY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "target", "release", "obrain-chat")
MODEL = os.path.expanduser("~/models/llama3.2-3b.gguf")
PERSONA_BASE = "/tmp/obrain-b5"
N_CTX = "4096"

# ANSI escape code pattern
ANSI_RE = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]|\x1b\[\?[0-9]+[hl]|\[2K|\[5C')

# Test questions: setup + evaluation
QUESTIONS = [
    {"q": "Je m'appelle Thomas et j'habite à Lyon. Je suis développeur Rust.",
     "kw": [], "setup": True},
    {"q": "Comment je m'appelle ?",
     "kw": ["Thomas"], "cat": "direct_recall"},
    {"q": "Où est-ce que j'habite ?",
     "kw": ["Lyon"], "cat": "direct_recall"},
    {"q": "Quel est mon métier ?",
     "kw": ["Rust", "développeur"], "cat": "direct_recall"},
    {"q": "Résume ce que tu sais de moi en une phrase.",
     "kw": ["Thomas"], "cat": "multi_fact"},
    {"q": "Est-ce que j'habite à Paris ?",
     "kw": ["non"], "cat": "negation"},
]


def clean_response(raw: str, query: str) -> str:
    """Extract the model's actual response from pexpect output."""
    lines = raw.split("\n")
    result = []
    for line in lines:
        clean = ANSI_RE.sub('', line).strip()
        # Skip: empty, debug lines, query echo, spinner, ggml internal
        if not clean:
            continue
        if clean.startswith(("[", "  [", "ggml_", "graph_", "init:", "update:")):
            continue
        if "réflexion" in clean or "spinner" in clean.lower():
            continue
        if clean == query.strip():
            continue
        if clean.startswith("you>"):
            continue
        if "kernel_" in clean or "pipeline:" in clean:
            continue
        result.append(clean)
    return " ".join(result).strip()


def run_mode(mode: str, extra_args: list = None) -> list:
    """Run a full interactive session, return per-question results."""
    args = ['--model', MODEL, '--n-gpu', '99', '--n-ctx', N_CTX, '--debug']
    if extra_args:
        args.extend(extra_args)

    print(f"\n{'='*60}")
    print(f"  Mode: {mode}")
    print(f"  Extra: {extra_args or 'none'}")
    print(f"{'='*60}")

    child = pexpect.spawn(BINARY, args, encoding='utf-8', timeout=180, dimensions=(80, 200))

    # Wait for first prompt
    idx = child.expect([r'you>', pexpect.EOF, pexpect.TIMEOUT], timeout=120)
    if idx != 0:
        print(f"  ❌ Failed to start (idx={idx})")
        child.close()
        return [{"error": True}]

    startup = child.before or ''
    print(f"  ✓ Model loaded ({len(startup)} chars startup)")

    # Check: KV cache management (should see "no graph" or graph loading, NOT text injection)
    kv_managed = "fallback" in startup.lower() or "graph" in startup.lower() or "kv" in startup.lower()
    has_text_injection = "injecting text" in startup.lower()
    print(f"  KV management: {kv_managed}, Text injection: {has_text_injection}")

    results = []
    for i, q in enumerate(QUESTIONS):
        query = q["q"]
        print(f"\n  [{i+1}/{len(QUESTIONS)}] {query[:55]}...")

        child.sendline(query)
        t0 = time.time()
        idx = child.expect([r'you>', pexpect.EOF, pexpect.TIMEOUT], timeout=90)
        latency = time.time() - t0

        if idx != 0:
            print(f"    ⚠ No prompt returned (idx={idx})")
            results.append({"query": query, "response": "[TIMEOUT/EOF]", "error": True})
            break

        raw = child.before or ''
        response = clean_response(raw, query)

        # Debug: check for B3 ablation logs
        has_b3 = "[B3]" in raw
        has_entropy = "head_contribution_entropy" in raw

        # Keyword matching
        kw_hit = [k for k in q.get("kw", []) if k.lower() in response.lower()]
        kw_miss = [k for k in q.get("kw", []) if k.lower() not in response.lower()]
        hit_rate = len(kw_hit) / max(len(q.get("kw", [])), 1)

        # Quality checks
        is_empty = len(response) < 10
        has_artifacts = bool(re.search(r'BeNull|<\|endoftext|<\|im_start\|>', response))

        r = {
            "query": query,
            "response": response[:300],
            "setup": q.get("setup", False),
            "category": q.get("cat", "setup"),
            "kw_hit": kw_hit,
            "kw_miss": kw_miss,
            "hit_rate": hit_rate,
            "is_empty": is_empty,
            "has_artifacts": has_artifacts,
            "latency_s": round(latency, 1),
            "has_b3_log": has_b3,
            "has_entropy_log": has_entropy,
        }
        results.append(r)

        status = "✅" if (hit_rate >= 1.0 or q.get("setup")) else ("⚠" if hit_rate > 0 else "❌")
        print(f"    {status} Response ({len(response)}c, {latency:.1f}s): {response[:100]}")
        if kw_miss:
            print(f"    Missing: {kw_miss}")
        if has_b3:
            print(f"    [B3 ablation active]")

    # Quit
    child.sendline("/quit")
    try:
        child.expect(pexpect.EOF, timeout=10)
    except:
        pass
    child.close()

    return results


def summarize(all_results: dict) -> dict:
    summary = {}
    for mode, results in all_results.items():
        evals = [r for r in results if not r.get("setup") and not r.get("error")]
        n = len(evals)
        if n == 0:
            summary[mode] = {"error": True}
            continue
        avg_hit = sum(r["hit_rate"] for r in evals) / n
        n_empty = sum(1 for r in evals if r["is_empty"])
        n_artifacts = sum(1 for r in evals if r["has_artifacts"])
        avg_latency = sum(r.get("latency_s", 0) for r in evals) / n
        has_b3 = any(r.get("has_b3_log") for r in results)

        summary[mode] = {
            "questions": n,
            "avg_hit_rate": round(avg_hit, 3),
            "empty_responses": n_empty,
            "artifact_responses": n_artifacts,
            "avg_latency_s": round(avg_latency, 1),
            "b3_active": has_b3,
        }
    return summary


def main():
    if not os.path.exists(BINARY):
        print(f"ERROR: Binary not found: {BINARY}")
        sys.exit(1)
    if not os.path.exists(MODEL):
        print(f"ERROR: Model not found: {MODEL}")
        sys.exit(1)

    all_results = {}

    # ── Mode 1: Baseline ──
    print("\n\n" + "▓"*70)
    print("  PHASE 1: BASELINE (no persona)")
    print("▓"*70)
    all_results["baseline"] = run_mode("baseline")

    # ── Mode 2: Phase A (persona, broadcast mask) ──
    persona_a = f"{PERSONA_BASE}-a"
    os.makedirs(persona_a, exist_ok=True)
    print("\n\n" + "▓"*70)
    print("  PHASE 2: PHASE A (persona + broadcast mask)")
    print("▓"*70)
    all_results["phase_a"] = run_mode("phase_a", ["--persona", persona_a])

    # ── Mode 3: Phase B (persona, head-router) ──
    persona_b = f"{PERSONA_BASE}-b"
    os.makedirs(persona_b, exist_ok=True)
    print("\n\n" + "▓"*70)
    print("  PHASE 3: PHASE B (persona + head-router)")
    print("▓"*70)
    all_results["phase_b"] = run_mode("phase_b", [
        "--persona", persona_b,
        "--head-router",
        "--head-router-warmup", "0",
    ])

    # ── Summary ──
    summary = summarize(all_results)

    print("\n\n" + "="*70)
    print("  BENCHMARK RESULTS — B5")
    print("="*70)
    print(f"\n{'Mode':<15} {'Hit Rate':>10} {'Empty':>7} {'Artif.':>7} {'Latency':>9} {'B3':>5}")
    print("-" * 55)
    for mode, s in summary.items():
        if s.get("error"):
            print(f"{mode:<15} {'ERROR':>10}")
            continue
        print(f"{mode:<15} {s['avg_hit_rate']:>9.0%} {s['empty_responses']:>7} {s['artifact_responses']:>7} {s['avg_latency_s']:>8.1f}s {'✓' if s.get('b3_active') else '—':>5}")

    print("\n\nDetailed Results:")
    for mode, results in all_results.items():
        print(f"\n--- {mode} ---")
        for r in results:
            if r.get("error"):
                print(f"  ❌ ERROR")
                continue
            status = "📋" if r.get("setup") else ("✅" if r["hit_rate"] >= 1.0 else "❌")
            print(f"  {status} [{r.get('category','?')}] {r['query'][:50]}")
            print(f"     → {r['response'][:120]}")

    # Save
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_b5_results.json")
    with open(output_path, "w") as f:
        json.dump({"summary": summary, "details": all_results}, f, indent=2, ensure_ascii=False)
    print(f"\n\nResults saved to {output_path}")

    return summary


if __name__ == "__main__":
    main()
