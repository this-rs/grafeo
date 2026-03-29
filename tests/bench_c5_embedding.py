#!/usr/bin/env python3
"""
C5 — Benchmark: embedding injection vs text encoding.

Compares 4 modes on the same graph (27 nodes, 50 edges):
  1. Phase A — graph + broadcast mask (text encoding)
  2. Phase B — graph + head-router (text encoding, per-head mask)
  3. Phase C 50% — graph + head-router + 50% embedding injection
  4. Phase C 100% — graph + head-router + 100% embedding injection

Key metrics:
  - Factual recall (keyword hit rate)
  - KV positions used (text ~50 tok/node vs embedding 1 pos/node)
  - Latency per query
  - B3 ablation fires
"""

import pexpect
import sys
import os
import re
import time
import json

BINARY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "target", "release", "obrain-chat")
MODEL = os.path.expanduser("~/models/Qwen3-14B-Claude-4.5-Opus-Distill.q4_k_m.gguf")
GRAPH_DB = "/tmp/obrain-b5bis-graph"
N_CTX = "8192"

# Ensure llama.cpp shared library is findable
_LLAMA_LIB = os.path.expanduser("~/projects/ia/llama.cpp/build/bin")
if os.path.isdir(_LLAMA_LIB):
    os.environ.setdefault("DYLD_LIBRARY_PATH", _LLAMA_LIB)

ANSI_RE = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]|\x1b\[\?[0-9]+[hl]|\[2K|\[5C')

# Same questions as B5-bis for comparison
QUESTIONS = [
    {"q": "Ou habite Thomas Riviere ?",
     "kw": ["Lyon"], "cat": "direct_recall"},
    {"q": "Sur quel projet travaille Sophie Martin ?",
     "kw": ["DataPipeline"], "cat": "direct_recall"},
    {"q": "Quel langage utilise Marc Dupont ?",
     "kw": ["Go"], "cat": "direct_recall"},

    {"q": "Qui habite dans la meme ville que Thomas Riviere ?",
     "kw": ["Marc"], "cat": "multi_hop"},
    {"q": "Quelles technologies sont utilisees par le projet Obrain ?",
     "kw": ["Rust"], "cat": "multi_hop"},

    {"q": "Qui connait Thomas Riviere ?",
     "kw": ["Marc", "Alice"], "cat": "relationship"},
    {"q": "A quel evenement Thomas et Marc ont-ils assiste ensemble ?",
     "kw": ["RustConf"], "cat": "relationship"},

    {"q": "Est-ce que Pierre Bernard habite a Lyon ?",
     "kw": ["non", "Toulouse"], "cat": "negation"},
    {"q": "Est-ce que Alice Chen utilise Rust ?",
     "kw": ["non", "Python"], "cat": "negation"},

    {"q": "Combien de personnes utilisent Python ?",
     "kw": ["Sophie", "Alice"], "cat": "aggregation"},
    {"q": "Liste toutes les villes mentionnees dans le graphe.",
     "kw": ["Lyon", "Paris", "Grenoble", "Toulouse"], "cat": "aggregation"},

    {"q": "Decris le profil professionnel de Thomas Riviere.",
     "kw": ["Rust", "Mozilla", "Obrain"], "cat": "context"},
]


def clean_response(raw: str, query: str) -> str:
    lines = raw.split("\n")
    result = []
    for line in lines:
        clean = ANSI_RE.sub('', line).strip()
        if not clean:
            continue
        if clean.startswith(("[", "  [", "ggml_", "graph_", "init:", "update:", "Pipeline")):
            continue
        if any(x in clean.lower() for x in ["reflexion", "spinner", "kernel_", "phase c", "phase b", "pre-comput"]):
            continue
        if clean == query.strip() or clean.startswith("you>"):
            continue
        result.append(clean)
    return " ".join(result).strip()


def extract_c4_metrics(raw: str) -> dict:
    """Extract Phase C embedding injection metrics from debug output."""
    metrics = {"embd_injected": 0, "text_encoded": 0, "precomputed": 0}

    for line in raw.split("\n"):
        m = re.search(r'\[C4\] nodes: (\d+) via embedding, (\d+) via text', line)
        if m:
            metrics["embd_injected"] += int(m.group(1))
            metrics["text_encoded"] += int(m.group(2))
        m = re.search(r'Pre-computed (\d+) node embeddings', line)
        if m:
            metrics["precomputed"] = int(m.group(1))

    return metrics


def extract_kv_positions(raw: str) -> int:
    """Extract KV position usage from debug output."""
    for line in raw.split("\n"):
        m = re.search(r'next_pos=(\d+)', line)
        if m:
            return int(m.group(1))
        m = re.search(r'total_tokens.*?(\d+)', line)
        if m:
            return int(m.group(1))
    return -1


def run_mode(mode: str, extra_args: list = None) -> dict:
    """Run a full session, return results."""
    args = ['--model', MODEL, '--n-gpu', '99', '--n-ctx', N_CTX, '--debug']
    if extra_args:
        args.extend(extra_args)

    print(f"\n{'='*60}")
    print(f"  Mode: {mode}")
    print(f"  Args: {' '.join(extra_args or [])}")
    print(f"{'='*60}")

    persona_dir = f"/tmp/obrain-c5-persona-{mode}"
    os.makedirs(persona_dir, exist_ok=True)

    child = pexpect.spawn(BINARY, args, encoding='utf-8', timeout=300, dimensions=(80, 200))

    idx = child.expect([r'you>', pexpect.EOF, pexpect.TIMEOUT], timeout=240)
    if idx != 0:
        print(f"  FAIL to start (idx={idx})")
        child.close()
        return {"error": True, "results": []}

    startup = child.before or ''
    n_nodes_match = re.search(r'(\d+) nodes?, (\d+) edges?', startup)
    n_nodes = int(n_nodes_match.group(1)) if n_nodes_match else 0
    n_edges = int(n_nodes_match.group(2)) if n_nodes_match else 0

    # Check embedding pre-computation
    c4_startup = extract_c4_metrics(startup)
    print(f"  Model loaded. Graph: {n_nodes} nodes, {n_edges} edges")
    if c4_startup["precomputed"] > 0:
        print(f"  [C] Pre-computed {c4_startup['precomputed']} node embeddings")

    results = []
    total_embd = 0
    total_text = 0

    for i, q in enumerate(QUESTIONS):
        query = q["q"]
        print(f"\n  [{i+1}/{len(QUESTIONS)}] {query[:55]}...")

        child.sendline(query)
        t0 = time.time()
        idx = child.expect([r'you>', pexpect.EOF, pexpect.TIMEOUT], timeout=120)
        latency = time.time() - t0

        if idx != 0:
            print(f"    No prompt (idx={idx})")
            results.append({"query": query, "error": True})
            break

        raw = child.before or ''
        response = clean_response(raw, query)
        c4 = extract_c4_metrics(raw)
        total_embd += c4["embd_injected"]
        total_text += c4["text_encoded"]

        kw_hit = [k for k in q.get("kw", []) if k.lower() in response.lower()]
        kw_miss = [k for k in q.get("kw", []) if k.lower() not in response.lower()]
        hit_rate = len(kw_hit) / max(len(q.get("kw", [])), 1)

        is_empty = len(response) < 10
        has_artifacts = bool(re.search(r'BeNull|<\|endoftext', response))

        r = {
            "query": query, "response": response[:300], "category": q["cat"],
            "kw_hit": kw_hit, "kw_miss": kw_miss, "hit_rate": hit_rate,
            "is_empty": is_empty, "has_artifacts": has_artifacts,
            "latency_s": round(latency, 1),
            "embd_injected": c4["embd_injected"],
            "text_encoded": c4["text_encoded"],
        }
        results.append(r)

        status = "OK" if hit_rate >= 1.0 else ("~" if hit_rate > 0 else "X")
        c4_tag = f" [E:{c4['embd_injected']}/T:{c4['text_encoded']}]" if c4["embd_injected"] > 0 else ""
        print(f"    {status} ({len(response)}c, {latency:.1f}s){c4_tag}: {response[:90]}")
        if kw_miss:
            print(f"    Missing: {kw_miss}")

    try:
        child.sendline("/quit")
    except:
        pass
    try:
        child.expect(pexpect.EOF, timeout=10)
    except:
        pass
    try:
        child.close()
    except:
        pass

    evals = [r for r in results if not r.get("error")]
    avg_hit = sum(r["hit_rate"] for r in evals) / max(len(evals), 1)
    avg_lat = sum(r.get("latency_s", 0) for r in evals) / max(len(evals), 1)

    return {
        "results": results,
        "avg_hit_rate": round(avg_hit, 3),
        "n_questions": len(evals),
        "avg_latency": round(avg_lat, 1),
        "graph_nodes": n_nodes,
        "graph_edges": n_edges,
        "total_embd_injected": total_embd,
        "total_text_encoded": total_text,
        "precomputed": c4_startup["precomputed"],
    }


def main():
    if not os.path.exists(BINARY):
        print(f"ERROR: Binary not found: {BINARY}")
        print("  Run: cargo build --release")
        sys.exit(1)
    if not os.path.exists(MODEL):
        print(f"ERROR: Model not found: {MODEL}")
        sys.exit(1)
    if not os.path.isdir(GRAPH_DB):
        print(f"ERROR: Graph DB not found: {GRAPH_DB}")
        print("  Run the gen_test_graph example first.")
        sys.exit(1)

    all_modes = {}

    # Mode 1: Phase A — broadcast mask (text only)
    print("\n" + "="*70)
    print("  MODE 1: PHASE A -- broadcast mask (text encoding)")
    print("="*70)
    all_modes["phase_a"] = run_mode("phase_a", [
        "--db", GRAPH_DB, "--persona", "/tmp/obrain-c5-persona-a",
    ])

    # Mode 2: Phase B — head-router (text only)
    print("\n" + "="*70)
    print("  MODE 2: PHASE B -- head-router (text encoding)")
    print("="*70)
    all_modes["phase_b"] = run_mode("phase_b", [
        "--db", GRAPH_DB, "--persona", "/tmp/obrain-c5-persona-b",
        "--head-router", "--head-router-warmup", "0",
    ])

    # Mode 3: Phase C 50% — head-router + 50% embedding injection
    print("\n" + "="*70)
    print("  MODE 3: PHASE C 50% -- head-router + 50% embedding injection")
    print("="*70)
    all_modes["phase_c_50"] = run_mode("phase_c_50", [
        "--db", GRAPH_DB, "--persona", "/tmp/obrain-c5-persona-c50",
        "--head-router", "--head-router-warmup", "0",
        "--embd-injection-ratio", "0.5",
    ])

    # Mode 4: Phase C 100% — head-router + 100% embedding injection
    print("\n" + "="*70)
    print("  MODE 4: PHASE C 100% -- head-router + 100% embedding injection")
    print("="*70)
    all_modes["phase_c_100"] = run_mode("phase_c_100", [
        "--db", GRAPH_DB, "--persona", "/tmp/obrain-c5-persona-c100",
        "--head-router", "--head-router-warmup", "0",
        "--embd-injection-ratio", "1.0",
    ])

    # === Summary ===
    print("\n\n" + "="*70)
    print("  C5 RESULTS -- Embedding Injection Benchmark")
    print("="*70)

    print(f"\n{'Mode':<14} {'Hit Rate':>10} {'Embd/Text':>12} {'Pre-comp':>10} {'Latency':>9}")
    print("-" * 60)
    for mode, data in all_modes.items():
        if data.get("error"):
            print(f"{mode:<14} {'ERROR':>10}")
            continue
        embd_txt = f"{data['total_embd_injected']}/{data['total_text_encoded']}"
        print(f"{mode:<14} {data['avg_hit_rate']:>9.0%} {embd_txt:>12} {data['precomputed']:>10} {data['avg_latency']:>8.1f}s")

    # Comparison analysis
    pa = all_modes.get("phase_a", {})
    pb = all_modes.get("phase_b", {})
    pc50 = all_modes.get("phase_c_50", {})
    pc100 = all_modes.get("phase_c_100", {})

    pa_hr = pa.get("avg_hit_rate", 0)
    pb_hr = pb.get("avg_hit_rate", 0)
    pc50_hr = pc50.get("avg_hit_rate", 0)
    pc100_hr = pc100.get("avg_hit_rate", 0)

    print("\n\n=== Phase C Validation ===")
    print(f"  Phase A (text broadcast):   {pa_hr:.0%}")
    print(f"  Phase B (text per-head):    {pb_hr:.0%}")
    print(f"  Phase C 50% (mixed):        {pc50_hr:.0%}")
    print(f"  Phase C 100% (all embed):   {pc100_hr:.0%}")

    # Quality check
    c100_ok = pc100_hr >= pb_hr * 0.9  # within 10% of Phase B
    c50_ok = pc50_hr >= pb_hr * 0.9

    # Compression check
    compression = "N/A"
    if pc100.get("total_embd_injected", 0) > 0:
        # Estimate: each text node ~50 tokens, each embd node = 1 position
        est_text_tokens = pc100["total_embd_injected"] * 50
        embd_positions = pc100["total_embd_injected"]
        compression = f"{est_text_tokens / max(embd_positions, 1):.0f}x"

    print(f"\n  C100 >= 90% of B recall:    {'YES' if c100_ok else 'NO'}")
    print(f"  C50 >= 90% of B recall:     {'YES' if c50_ok else 'NO'}")
    print(f"  Est. compression (C100):    {compression}")

    go = c100_ok or c50_ok
    print(f"\n  -> {'GO -- Phase C validated' if go else 'INVESTIGATE -- quality regression'}")

    # Per-question detail
    print("\n\nDetailed Results:")
    for mode, data in all_modes.items():
        print(f"\n--- {mode} ---")
        for r in data.get("results", []):
            if r.get("error"):
                print(f"  X ERROR: {r.get('query','?')}")
                continue
            status = "OK" if r["hit_rate"] >= 1.0 else ("~" if r["hit_rate"] > 0 else "X")
            e_tag = f" [E:{r['embd_injected']}]" if r.get("embd_injected", 0) > 0 else ""
            print(f"  {status} [{r['category']}]{e_tag} {r['query'][:50]}")
            print(f"     -> {r['response'][:120]}")

    # Save
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_c5_results.json")
    with open(output_path, "w") as f:
        json.dump(all_modes, f, indent=2, ensure_ascii=False)
    print(f"\n\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
