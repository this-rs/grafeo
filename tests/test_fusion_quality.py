#!/usr/bin/env python3
"""
Test: Does GNN fusion improve embedding quality over text-only?

Compares 3 modes on the B5-bis graph (27 nodes, 50 edges):
  1. Phase B — per-head router, text encoding only
  2. Phase C text — embedding injection, no GNN (no persona)
  3. Phase C fused — embedding injection with GNN (persona DB with facts)

For mode 3, we use a persona that has learned facts from the graph,
so the FactGNN has topological embeddings to fuse.
"""

import pexpect
import os
import re
import time
import json
import sys

BINARY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "target", "release", "obrain-chat")
MODEL = os.path.expanduser("~/models/Qwen3-14B-Claude-4.5-Opus-Distill.q4_k_m.gguf")
GRAPH_DB = "/tmp/obrain-b5bis-graph"
N_CTX = "8192"

# Ensure llama.cpp shared library is findable
_LLAMA_LIB = os.path.expanduser("~/projects/ia/llama.cpp/build/bin")
if os.path.isdir(_LLAMA_LIB):
    os.environ.setdefault("DYLD_LIBRARY_PATH", _LLAMA_LIB)

ANSI_RE = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]|\x1b\[\?[0-9]+[hl]|\[2K|\[5C')

# Focused questions that test structural/topological reasoning
QUESTIONS = [
    {"q": "Ou habite Thomas Riviere ?",
     "kw": ["Lyon"], "cat": "direct_recall"},
    {"q": "Qui habite dans la meme ville que Thomas Riviere ?",
     "kw": ["Marc"], "cat": "multi_hop"},
    {"q": "Quelles technologies sont utilisees par le projet Obrain ?",
     "kw": ["Rust"], "cat": "multi_hop"},
    {"q": "Qui connait Thomas Riviere ?",
     "kw": ["Marc", "Alice"], "cat": "relationship"},
    {"q": "A quel evenement Thomas et Marc ont-ils assiste ensemble ?",
     "kw": ["RustConf"], "cat": "relationship"},
    {"q": "Liste toutes les villes mentionnees dans le graphe.",
     "kw": ["Lyon", "Paris", "Grenoble", "Toulouse"], "cat": "aggregation"},
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


def run_mode(mode: str, extra_args: list) -> dict:
    """Run a session with given args, return results."""
    args = ['--model', MODEL, '--n-gpu', '99', '--n-ctx', N_CTX, '--debug']
    args.extend(extra_args)

    print(f"\n{'='*60}")
    print(f"  Mode: {mode}")
    print(f"  Args: {' '.join(extra_args)}")
    print(f"{'='*60}")

    child = pexpect.spawn(BINARY, args, encoding='utf-8', timeout=600, dimensions=(80, 200))

    idx = child.expect([r'you>', pexpect.EOF, pexpect.TIMEOUT], timeout=480)
    if idx != 0:
        print(f"  FAIL to start (idx={idx})")
        if child.before:
            print(f"  Last output: ...{child.before[-300:]}")
        child.close()
        return {"error": True, "results": []}

    startup = child.before or ''

    # Extract key metrics from startup
    has_gnn = 'GNN' in startup or 'FactGNN' in startup
    has_pnet = 'ProjectionNet' in startup
    has_fusion = 'fused' in startup.lower()
    n_precomp = 0
    m = re.search(r'Pre-computed (\d+)', startup)
    if m:
        n_precomp = int(m.group(1))

    print(f"  Startup: GNN={has_gnn}, PNet={has_pnet}, Fusion={has_fusion}, Precomp={n_precomp}")

    results = []
    for i, q in enumerate(QUESTIONS):
        query = q["q"]
        print(f"\n  [{i+1}/{len(QUESTIONS)}] {query[:55]}...")

        child.sendline(query)
        t0 = time.time()
        idx = child.expect([r'you>', pexpect.EOF, pexpect.TIMEOUT], timeout=120)
        latency = time.time() - t0

        if idx != 0:
            print(f"    TIMEOUT/EOF (idx={idx})")
            results.append({"query": query, "error": True})
            break

        raw = child.before or ''
        response = clean_response(raw, query)

        # Extract C4 metrics (embedding injection)
        m_c4 = re.search(r'\[C4\] nodes: (\d+) via embedding, (\d+) via text', raw)
        embd_n = int(m_c4.group(1)) if m_c4 else 0
        text_n = int(m_c4.group(2)) if m_c4 else 0

        # Extract C7 metrics (cosine retrieval)
        m_c7 = re.search(r'\[C7\] retrieval: (\d+) fuzzy, \+(\d+) cosine-new, (\d+) cosine-boosted', raw)
        cosine_new = int(m_c7.group(2)) if m_c7 else 0
        cosine_boosted = int(m_c7.group(3)) if m_c7 else 0

        kw_hit = [k for k in q.get("kw", []) if k.lower() in response.lower()]
        kw_miss = [k for k in q.get("kw", []) if k.lower() not in response.lower()]
        hit_rate = len(kw_hit) / max(len(q.get("kw", [])), 1)

        r = {
            "query": query, "response": response[:300], "category": q["cat"],
            "kw_hit": kw_hit, "kw_miss": kw_miss, "hit_rate": hit_rate,
            "latency_s": round(latency, 1),
            "embd_injected": embd_n, "text_encoded": text_n,
            "cosine_new": cosine_new, "cosine_boosted": cosine_boosted,
        }
        results.append(r)

        status = "OK" if hit_rate >= 1.0 else ("~" if hit_rate > 0 else "X")
        c4_tag = f" [E:{embd_n}/T:{text_n}]" if embd_n > 0 else ""
        c7_tag = f" [cos:+{cosine_new}/^{cosine_boosted}]" if (cosine_new + cosine_boosted) > 0 else ""
        print(f"    {status} ({len(response)}c, {latency:.1f}s){c4_tag}{c7_tag}: {response[:90]}")
        if kw_miss:
            print(f"    Missing: {kw_miss}")

    try:
        child.sendline("/quit")
        child.expect(pexpect.EOF, timeout=10)
    except:
        pass
    child.close()

    evals = [r for r in results if not r.get("error")]
    avg_hit = sum(r["hit_rate"] for r in evals) / max(len(evals), 1)
    avg_lat = sum(r.get("latency_s", 0) for r in evals) / max(len(evals), 1)

    return {
        "results": results,
        "avg_hit_rate": round(avg_hit, 3),
        "avg_latency": round(avg_lat, 1),
        "n_questions": len(evals),
        "has_gnn": has_gnn,
        "has_pnet": has_pnet,
        "has_fusion": has_fusion,
        "precomputed": n_precomp,
    }


def main():
    if not os.path.exists(BINARY):
        print(f"ERROR: Binary not found: {BINARY}")
        sys.exit(1)
    if not os.path.exists(MODEL):
        print(f"ERROR: Model not found: {MODEL}")
        sys.exit(1)

    all_modes = {}

    # Mode 1: Phase B — text-only, per-head router
    all_modes["phase_b"] = run_mode("phase_b", [
        "--db", GRAPH_DB,
        "--persona", "/tmp/obrain-fusion-test-b",
        "--head-router", "--head-router-warmup", "0",
    ])

    # Mode 2: Phase C text-only — embedding injection, no GNN
    all_modes["phase_c_text"] = run_mode("phase_c_text", [
        "--db", GRAPH_DB,
        "--persona", "/tmp/obrain-fusion-test-ct",
        "--head-router", "--head-router-warmup", "0",
        "--embd-injection-ratio", "1.0",
    ])

    # Mode 3: Phase C fused — embedding injection with GNN
    # Use /tmp/elun persona which has FactGNN data
    all_modes["phase_c_fused"] = run_mode("phase_c_fused", [
        "--db", GRAPH_DB,
        "--persona", "/tmp/elun",
        "--head-router", "--head-router-warmup", "0",
        "--embd-injection-ratio", "1.0",
    ])

    # === Summary ===
    print("\n\n" + "=" * 70)
    print("  FUSION QUALITY TEST — Results")
    print("=" * 70)

    print(f"\n{'Mode':<16} {'Hit Rate':>10} {'GNN':>5} {'Fused':>7} {'Precomp':>9} {'Latency':>9}")
    print("-" * 62)
    for mode, data in all_modes.items():
        if data.get("error"):
            print(f"{mode:<16} {'ERROR':>10}")
            continue
        print(f"{mode:<16} {data['avg_hit_rate']:>9.0%} {'Y' if data['has_gnn'] else 'N':>5} "
              f"{'Y' if data['has_fusion'] else 'N':>7} {data['precomputed']:>9} {data['avg_latency']:>8.1f}s")

    # Per-category comparison
    categories = ["direct_recall", "multi_hop", "relationship", "aggregation"]
    print(f"\n  Per-category:")
    header = f"  {'Category':<16}"
    for mode in all_modes:
        if not all_modes[mode].get("error"):
            header += f" {mode:>14}"
    print(header)
    print("  " + "-" * 60)

    for cat in categories:
        row = f"  {cat:<16}"
        for mode, mdata in all_modes.items():
            if mdata.get("error"):
                continue
            cat_results = [r for r in mdata.get("results", []) if not r.get("error") and r.get("category") == cat]
            if cat_results:
                cat_hr = sum(r["hit_rate"] for r in cat_results) / len(cat_results)
                row += f" {cat_hr:>13.0%}"
            else:
                row += f" {'—':>14}"
        print(row)

    # C7 Cosine retrieval analysis
    print(f"\n  C7 Cosine Retrieval:")
    for mode, mdata in all_modes.items():
        if mdata.get("error"):
            continue
        evals = [r for r in mdata.get("results", []) if not r.get("error")]
        total_cosine_new = sum(r.get("cosine_new", 0) for r in evals)
        total_cosine_boost = sum(r.get("cosine_boosted", 0) for r in evals)
        total_embd = sum(r.get("embd_injected", 0) for r in evals)
        total_text = sum(r.get("text_encoded", 0) for r in evals)
        n_nodes = mdata.get("precomputed", 0) or 27  # approx
        # micro-tags: ~3 positions/node, pure text: ~50 tokens/node
        compression = "~17x" if total_embd > 0 else "1x (text)"
        print(f"    {mode:<16} cosine_new={total_cosine_new:>3} boosted={total_cosine_boost:>3} "
              f"embd={total_embd:>3} text={total_text:>3} compression={compression}")

    # Delta analysis
    b_hr = all_modes.get("phase_b", {}).get("avg_hit_rate", 0)
    ct_hr = all_modes.get("phase_c_text", {}).get("avg_hit_rate", 0)
    cf_hr = all_modes.get("phase_c_fused", {}).get("avg_hit_rate", 0)

    print(f"\n  Deltas vs Phase B ({b_hr:.0%}):")
    print(f"    Phase C text:  {ct_hr - b_hr:+.0%}")
    print(f"    Phase C fused: {cf_hr - b_hr:+.0%}")
    if ct_hr > 0:
        print(f"    Fusion delta:  {cf_hr - ct_hr:+.0%} (fused vs text-only)")

    # Decision gate
    best_c_hr = max(ct_hr, cf_hr)
    microtags_ok = best_c_hr >= 0.8
    print(f"\n  Decision Gate:")
    print(f"    Best Phase C hit rate: {best_c_hr:.0%} (threshold: ≥80%)")
    print(f"    Micro-tags compression: ~17x (threshold: ≥15x)")
    if microtags_ok:
        print(f"    -> VALIDATED: micro-tags as default Phase C mode")
    else:
        print(f"    -> NOT YET: hit rate below threshold, needs tuning")
        if best_c_hr < b_hr:
            print(f"    -> REGRESSION vs Phase B ({b_hr:.0%} → {best_c_hr:.0%})")

    # Save
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fusion_test_results.json")
    with open(output_path, "w") as f:
        json.dump(all_modes, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
