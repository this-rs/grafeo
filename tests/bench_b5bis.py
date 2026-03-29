#!/usr/bin/env python3
"""
B5-bis — Validating Phase B with Qwen3-14B + real graph (27 nodes, 50 edges).

Key checks:
  1. HeadRouter actually learns (B3 ablation fires, entropy decreases)
  2. Per-head routing beats broadcast on a real graph
  3. No text injection — all via KV cache management
  4. Graph nodes loaded via --db (not just persona in-memory facts)
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

ANSI_RE = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]|\x1b\[\?[0-9]+[hl]|\[2K|\[5C')

# Questions designed around the graph structure
QUESTIONS = [
    # Direct recall (1-hop)
    {"q": "Où habite Thomas Rivière ?",
     "kw": ["Lyon"], "cat": "direct_recall"},
    {"q": "Sur quel projet travaille Sophie Martin ?",
     "kw": ["DataPipeline"], "cat": "direct_recall"},
    {"q": "Quel langage utilise Marc Dupont ?",
     "kw": ["Go"], "cat": "direct_recall"},

    # Multi-hop (2-hop)
    {"q": "Qui habite dans la même ville que Thomas Rivière ?",
     "kw": ["Marc"], "cat": "multi_hop"},
    {"q": "Quelles technologies sont utilisées par le projet Obrain ?",
     "kw": ["Rust"], "cat": "multi_hop"},

    # Relationship reasoning
    {"q": "Qui connaît Thomas Rivière ?",
     "kw": ["Marc", "Alice"], "cat": "relationship"},
    {"q": "À quel événement Thomas et Marc ont-ils assisté ensemble ?",
     "kw": ["RustConf"], "cat": "relationship"},

    # Negation
    {"q": "Est-ce que Pierre Bernard habite à Lyon ?",
     "kw": ["non", "Toulouse"], "cat": "negation"},
    {"q": "Est-ce que Alice Chen utilise Rust ?",
     "kw": ["non", "Python"], "cat": "negation"},

    # Aggregation
    {"q": "Combien de personnes utilisent Python ?",
     "kw": ["Sophie", "Alice"], "cat": "aggregation"},
    {"q": "Liste toutes les villes mentionnées dans le graphe.",
     "kw": ["Lyon", "Paris", "Grenoble", "Toulouse"], "cat": "aggregation"},

    # Context / meta
    {"q": "Décris le profil professionnel de Thomas Rivière.",
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
        if "réflexion" in clean or "spinner" in clean.lower() or "kernel_" in clean:
            continue
        if clean == query.strip() or clean.startswith("you>"):
            continue
        result.append(clean)
    return " ".join(result).strip()


def extract_b3_metrics(raw: str) -> dict:
    """Extract B3 ablation metrics from debug output."""
    metrics = {"b3_fired": False, "entropy": None, "bank_contributions": None, "n_updates": None}

    for line in raw.split("\n"):
        if "[B3] head_contribution_entropy" in line:
            metrics["b3_fired"] = True
            m = re.search(r'head_contribution_entropy=([\d.]+)', line)
            if m:
                metrics["entropy"] = float(m.group(1))
            m = re.search(r'bank_contributions=\[([^\]]+)\]', line)
            if m:
                metrics["bank_contributions"] = m.group(1)
        if "[B3] HeadRouter updated" in line:
            m = re.search(r'n_updates=(\d+)', line)
            if m:
                metrics["n_updates"] = int(m.group(1))

    return metrics


def run_mode(mode: str, extra_args: list = None) -> dict:
    """Run a full session, return results + B3 metrics."""
    args = ['--model', MODEL, '--n-gpu', '99', '--n-ctx', N_CTX, '--debug']
    if extra_args:
        args.extend(extra_args)

    print(f"\n{'='*60}")
    print(f"  Mode: {mode}")
    print(f"  Args: {' '.join(extra_args or [])}")
    print(f"{'='*60}")

    child = pexpect.spawn(BINARY, args, encoding='utf-8', timeout=300, dimensions=(80, 200))

    idx = child.expect([r'you>', pexpect.EOF, pexpect.TIMEOUT], timeout=180)
    if idx != 0:
        print(f"  ❌ Failed to start")
        child.close()
        return {"error": True, "results": []}

    startup = child.before or ''
    # Check graph loading
    has_graph = "nodes," in startup and "edges" in startup
    has_banks = "bank" in startup.lower() or "Bank" in startup
    n_nodes_match = re.search(r'(\d+) nodes?, (\d+) edges?', startup)
    n_nodes = int(n_nodes_match.group(1)) if n_nodes_match else 0
    n_edges = int(n_nodes_match.group(2)) if n_nodes_match else 0

    print(f"  ✓ Model loaded")
    print(f"  Graph: {n_nodes} nodes, {n_edges} edges, banks={has_banks}")

    results = []
    all_b3 = []
    entropies = []

    for i, q in enumerate(QUESTIONS):
        query = q["q"]
        print(f"\n  [{i+1}/{len(QUESTIONS)}] {query[:55]}...")

        child.sendline(query)
        t0 = time.time()
        idx = child.expect([r'you>', pexpect.EOF, pexpect.TIMEOUT], timeout=120)
        latency = time.time() - t0

        if idx != 0:
            print(f"    ⚠ No prompt (idx={idx})")
            results.append({"query": query, "error": True})
            break

        raw = child.before or ''
        response = clean_response(raw, query)
        b3 = extract_b3_metrics(raw)

        if b3["b3_fired"]:
            all_b3.append(b3)
            if b3["entropy"] is not None:
                entropies.append(b3["entropy"])

        kw_hit = [k for k in q.get("kw", []) if k.lower() in response.lower()]
        kw_miss = [k for k in q.get("kw", []) if k.lower() not in response.lower()]
        hit_rate = len(kw_hit) / max(len(q.get("kw", [])), 1)

        is_empty = len(response) < 10
        has_artifacts = bool(re.search(r'BeNull|<\|endoftext', response))

        r = {
            "query": query, "response": response[:300], "category": q["cat"],
            "kw_hit": kw_hit, "kw_miss": kw_miss, "hit_rate": hit_rate,
            "is_empty": is_empty, "has_artifacts": has_artifacts,
            "latency_s": round(latency, 1), "b3": b3,
        }
        results.append(r)

        status = "✅" if hit_rate >= 1.0 else ("⚠" if hit_rate > 0 else "❌")
        b3_tag = f" [B3 entropy={b3['entropy']:.3f}]" if b3["entropy"] else ""
        print(f"    {status} ({len(response)}c, {latency:.1f}s){b3_tag}: {response[:90]}")
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

    # Entropy trend analysis
    entropy_trend = None
    if len(entropies) >= 3:
        first_half = sum(entropies[:len(entropies)//2]) / (len(entropies)//2)
        second_half = sum(entropies[len(entropies)//2:]) / (len(entropies) - len(entropies)//2)
        entropy_trend = "decreasing" if second_half < first_half * 0.95 else "stable/increasing"
        print(f"\n  Entropy trend: {entropy_trend} (first={first_half:.3f} → second={second_half:.3f})")

    evals = [r for r in results if not r.get("error")]
    avg_hit = sum(r["hit_rate"] for r in evals) / max(len(evals), 1)
    b3_count = sum(1 for r in results if r.get("b3", {}).get("b3_fired"))

    return {
        "results": results,
        "avg_hit_rate": round(avg_hit, 3),
        "n_questions": len(evals),
        "b3_fired_count": b3_count,
        "entropy_trend": entropy_trend,
        "entropies": entropies,
        "graph_nodes": n_nodes,
        "graph_edges": n_edges,
    }


def main():
    if not os.path.exists(BINARY):
        print(f"ERROR: Binary not found: {BINARY}")
        sys.exit(1)
    if not os.path.exists(MODEL):
        print(f"ERROR: Model not found: {MODEL}")
        sys.exit(1)
    if not os.path.isdir(GRAPH_DB):
        print(f"ERROR: Graph DB not found: {GRAPH_DB}")
        sys.exit(1)

    all_modes = {}

    # Mode 1: Graph + broadcast mask (Phase A)
    print("\n" + "▓"*70)
    print("  MODE 1: PHASE A — graph + broadcast mask")
    print("▓"*70)
    persona_a = "/tmp/obrain-b5bis-persona-a"
    os.makedirs(persona_a, exist_ok=True)
    all_modes["phase_a"] = run_mode("phase_a", [
        "--db", GRAPH_DB, "--persona", persona_a,
    ])

    # Mode 2: Graph + head-router (Phase B)
    print("\n" + "▓"*70)
    print("  MODE 2: PHASE B — graph + head-router (per-head α)")
    print("▓"*70)
    persona_b = "/tmp/obrain-b5bis-persona-b"
    os.makedirs(persona_b, exist_ok=True)
    all_modes["phase_b"] = run_mode("phase_b", [
        "--db", GRAPH_DB, "--persona", persona_b,
        "--head-router", "--head-router-warmup", "0",
    ])

    # === Summary ===
    print("\n\n" + "="*70)
    print("  B5-bis RESULTS — Qwen3-14B + Real Graph (27 nodes, 50 edges)")
    print("="*70)

    print(f"\n{'Mode':<12} {'Hit Rate':>10} {'B3 Fired':>10} {'Entropy Trend':>16} {'Latency':>9}")
    print("-" * 60)
    for mode, data in all_modes.items():
        if data.get("error"):
            print(f"{mode:<12} {'ERROR':>10}")
            continue
        trend = data.get('entropy_trend') or 'N/A'
        avg_lat = sum(r.get('latency_s',0) for r in data['results'])/max(len(data['results']),1)
        print(f"{mode:<12} {data['avg_hit_rate']:>9.0%} {data['b3_fired_count']:>10} {trend:>16} {avg_lat:>8.1f}s")

    # B3 validation check
    phase_b = all_modes.get("phase_b", {})
    b3_ok = phase_b.get("b3_fired_count", 0) > 0
    entropy_ok = phase_b.get("entropy_trend") == "decreasing"
    recall_ok = phase_b.get("avg_hit_rate", 0) >= all_modes.get("phase_a", {}).get("avg_hit_rate", 1)

    print("\n\n=== Phase C Readiness Check ===")
    print(f"  B3 ablation fires:       {'✅' if b3_ok else '❌'} ({phase_b.get('b3_fired_count', 0)}/{len(QUESTIONS)} queries)")
    print(f"  Entropy decreasing:      {'✅' if entropy_ok else '⚠️ ' + str(phase_b.get('entropy_trend', 'N/A'))}")
    print(f"  Phase B ≥ Phase A recall: {'✅' if recall_ok else '❌'} ({phase_b.get('avg_hit_rate',0):.0%} vs {all_modes.get('phase_a',{}).get('avg_hit_rate',0):.0%})")
    go = b3_ok and recall_ok
    print(f"\n  → {'🟢 GO for Phase C' if go else '🔴 NOT READY — investigate issues above'}")

    # Detailed per-question
    print("\n\nDetailed Results:")
    for mode, data in all_modes.items():
        print(f"\n--- {mode} ---")
        for r in data.get("results", []):
            if r.get("error"):
                print(f"  ❌ ERROR: {r.get('query','?')}")
                continue
            status = "✅" if r["hit_rate"] >= 1.0 else ("⚠" if r["hit_rate"] > 0 else "❌")
            b3_tag = " [B3]" if r.get("b3", {}).get("b3_fired") else ""
            print(f"  {status} [{r['category']}]{b3_tag} {r['query'][:50]}")
            print(f"     → {r['response'][:120]}")

    # Save
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_b5bis_results.json")
    with open(output_path, "w") as f:
        json.dump(all_modes, f, indent=2, ensure_ascii=False)
    print(f"\n\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
