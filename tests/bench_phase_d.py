#!/usr/bin/env python3
"""
Phase D Benchmark — Validates the Hilbert + 3-Tier KV architecture.

Compares 3 modes:
  1. Phase B (text only, no embeddings)
  2. Phase C (embeddings, no tiers)
  3. Phase D (embeddings + tiers + Hilbert)

Metrics:
  - Hit rate per category (direct_recall, multi_hop, relationship, aggregation)
  - Latency per query (seconds)
  - Tier distribution at round N (Α/Β/Γ counts)
  - Promotion/demotion counts per round
  - KV compression (positions used / nodes)

Decision gate: Phase D ≥85% hit rate AND <3s avg latency.
"""

import pexpect
import sys
import os
import re
import time
import json
from collections import defaultdict

BINARY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "target", "release", "obrain-chat")
MODEL = os.path.expanduser("~/models/Qwen3-14B-Claude-4.5-Opus-Distill.q4_k_m.gguf")
GRAPH_DB = "/tmp/obrain-d-bench-graph"
N_CTX = "8192"

ANSI_RE = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]|\x1b\[\?[0-9]+[hl]|\[2K|\[5C')

# 12 questions covering 5 categories
QUESTIONS = [
    # Direct recall (1-hop)
    {"q": "Où habite Thomas Rivière ?", "kw": ["Lyon"], "cat": "direct_recall"},
    {"q": "Sur quel projet travaille Sophie Martin ?", "kw": ["DataPipeline"], "cat": "direct_recall"},
    {"q": "Quel langage utilise Marc Dupont ?", "kw": ["Go"], "cat": "direct_recall"},

    # Multi-hop (2-hop)
    {"q": "Qui habite dans la même ville que Thomas Rivière ?", "kw": ["Marc"], "cat": "multi_hop"},
    {"q": "Quelles technologies sont utilisées par le projet Obrain ?", "kw": ["Rust"], "cat": "multi_hop"},

    # Relationship
    {"q": "Qui connaît Thomas Rivière ?", "kw": ["Marc", "Alice"], "cat": "relationship"},
    {"q": "À quel événement Thomas et Marc ont-ils assisté ensemble ?", "kw": ["RustConf"], "cat": "relationship"},

    # Aggregation
    {"q": "Combien de personnes utilisent Python ?", "kw": ["Sophie", "Alice"], "cat": "aggregation"},
    {"q": "Liste toutes les villes mentionnées dans le graphe.", "kw": ["Lyon", "Paris", "Grenoble", "Toulouse"], "cat": "aggregation"},

    # Context / meta
    {"q": "Décris le profil professionnel de Thomas Rivière.", "kw": ["Rust", "Mozilla", "Obrain"], "cat": "context"},
    {"q": "Quelles sont les compétences de Pierre Bernard ?", "kw": ["DevOps", "Bash", "Airbus"], "cat": "context"},
    {"q": "Quel est le lien entre Alice Chen et NeuralSearch ?", "kw": ["travaille", "INRIA"], "cat": "relationship"},
]


def clean_response(raw: str, query: str) -> str:
    """Extract the model's response from the raw pexpect output.

    The raw output contains everything between the last 'you>' and the next 'you>',
    which includes the query echo and the model's response.
    We skip everything up to and including the query echo, then collect the response.
    """
    lines = raw.split("\n")
    result = []
    found_query = False

    for line in lines:
        clean = ANSI_RE.sub('', line).strip()
        if not clean:
            continue

        # Skip until we find the query we sent (it gets echoed back)
        if not found_query:
            if query.strip()[:30] in clean:
                found_query = True
            continue

        # Skip technical/debug lines
        if clean.startswith(("[", "  [", "ggml_", "graph_", "init:", "update:", "Pipeline")):
            continue
        if "réflexion" in clean or "spinner" in clean.lower() or "kernel_" in clean:
            continue
        if clean.startswith("you>") or clean.startswith("/quit"):
            continue
        if "get_embeddings_ith" in clean or "the last position stored" in clean:
            continue
        if "Memory persisted" in clean:
            # Keep the line but strip the memory marker
            clean = re.sub(r'🧠.*$', '', clean).strip()
            if not clean:
                continue
        result.append(clean)

    # Fallback: if we never found the query echo, use the old approach
    if not found_query:
        for line in lines:
            clean = ANSI_RE.sub('', line).strip()
            if not clean:
                continue
            if clean.startswith(("[", "  [", "ggml_", "graph_", "init:", "update:", "Pipeline", "you>", "/quit")):
                continue
            if clean == query.strip():
                continue
            result.append(clean)

    return " ".join(result).strip()


def extract_d2_metrics(raw: str) -> dict:
    """Extract [D2] tier metrics from output."""
    m = re.search(r'\[D2\] round (\d+): \+(\d+) promoted, -(\d+) demoted \(Α=(\d+) Β=(\d+) Γ=(\d+)\)', raw)
    if m:
        return {
            "round": int(m.group(1)),
            "promoted": int(m.group(2)),
            "demoted": int(m.group(3)),
            "alpha": int(m.group(4)),
            "beta": int(m.group(5)),
            "gamma": int(m.group(6)),
        }
    return {}


def extract_d3_metrics(raw: str) -> dict:
    """Extract [D3] reward feedback metrics."""
    m = re.search(r'\[D3\] reward feedback: (\d+) accelerated demotes, (\d+) preemptive promotes', raw)
    if m:
        return {"d3_demotes": int(m.group(1)), "d3_promotes": int(m.group(2))}
    return {}


def extract_d6_metrics(raw: str) -> dict:
    """Extract [D6] rescore metrics."""
    m = re.search(r'\[D6\] (\d+) promoted nodes pending rescore', raw)
    if m:
        return {"d6_rescored": int(m.group(1))}
    return {}


def run_benchmark(mode_name: str, extra_args: list) -> dict:
    """Run one benchmark configuration and return results."""
    cmd_args = [
        BINARY,
        "--model", MODEL,
        "--db", GRAPH_DB,
        "--n-ctx", N_CTX,
        "--n-gpu", "99",
        "--max-nodes", "20",
    ] + extra_args

    print(f"\n{'='*60}")
    print(f"  Mode: {mode_name}")
    print(f"  Args: {' '.join(extra_args) if extra_args else '(default)'}")
    print(f"{'='*60}")

    try:
        child = pexpect.spawn(" ".join(cmd_args), timeout=120, encoding='utf-8', maxread=200000)
        # Wait for the first "you>" prompt — this means the app is ready.
        # For Phase C/D, ProjectionNet training can take 2+ minutes.
        child.expect("you>", timeout=240)
        # Small pause to let any trailing output settle
        time.sleep(0.5)
        # Drain any remaining buffered output
        try:
            while True:
                child.read_nonblocking(size=4096, timeout=0.5)
        except (pexpect.TIMEOUT, pexpect.EOF):
            pass
    except Exception as e:
        print(f"  ❌ Failed to start: {e}")
        return {"mode": mode_name, "error": str(e)}

    results = {
        "mode": mode_name,
        "questions": [],
        "tier_metrics": [],
        "d3_metrics": [],
        "d6_metrics": [],
    }

    for i, q in enumerate(QUESTIONS):
        print(f"  [{i+1}/{len(QUESTIONS)}] {q['cat']}: {q['q'][:50]}...", end=" ", flush=True)
        t0 = time.time()

        try:
            child.sendline(q["q"])
            child.expect("you>", timeout=90)
            raw = child.before
            latency = time.time() - t0
        except Exception as e:
            print(f"TIMEOUT ({e})")
            results["questions"].append({
                "category": q["cat"],
                "query": q["q"],
                "hit": False,
                "latency": 60.0,
                "error": str(e),
            })
            continue

        response = clean_response(raw, q["q"])
        response_lower = response.lower()

        # Check keywords
        hits = sum(1 for kw in q["kw"] if kw.lower() in response_lower)
        hit = hits >= 1  # At least one keyword found
        hit_ratio = hits / len(q["kw"])

        # Extract Phase D metrics
        d2 = extract_d2_metrics(raw)
        d3 = extract_d3_metrics(raw)
        d6 = extract_d6_metrics(raw)

        if d2:
            results["tier_metrics"].append(d2)
        if d3:
            results["d3_metrics"].append(d3)
        if d6:
            results["d6_metrics"].append(d6)

        status = "✅" if hit else "❌"
        print(f"{status} ({latency:.1f}s, {hits}/{len(q['kw'])} kw)")

        if not hit:
            snippet = response[:200] if response else "(empty)"
            print(f"       Expected: {q['kw']}")
            print(f"       Got: {snippet}")

        results["questions"].append({
            "category": q["cat"],
            "query": q["q"],
            "hit": hit,
            "hit_ratio": hit_ratio,
            "latency": latency,
            "keywords_found": hits,
            "keywords_total": len(q["kw"]),
            "d2": d2,
        })

    # Cleanup
    try:
        child.sendline("/quit")
        child.expect(pexpect.EOF, timeout=10)
    except:
        child.close()

    return results


def analyze_results(all_results: list) -> dict:
    """Compute summary statistics."""
    summary = {}

    for result in all_results:
        mode = result["mode"]
        qs = result.get("questions", [])
        if not qs:
            summary[mode] = {"error": result.get("error", "unknown")}
            continue

        # Overall hit rate
        hits = sum(1 for q in qs if q["hit"])
        total = len(qs)
        hit_rate = hits / total if total > 0 else 0

        # Per-category hit rate
        cats = defaultdict(lambda: {"hits": 0, "total": 0})
        for q in qs:
            cats[q["category"]]["total"] += 1
            if q["hit"]:
                cats[q["category"]]["hits"] += 1
        cat_rates = {c: v["hits"]/v["total"] for c, v in cats.items()}

        # Latency stats
        latencies = [q["latency"] for q in qs if "error" not in q]
        avg_lat = sum(latencies) / len(latencies) if latencies else 0
        p95_lat = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0

        # Tier evolution
        tier_metrics = result.get("tier_metrics", [])
        last_tier = tier_metrics[-1] if tier_metrics else {}
        total_promotes = sum(t.get("promoted", 0) for t in tier_metrics)
        total_demotes = sum(t.get("demoted", 0) for t in tier_metrics)

        # D3 reward feedback
        d3 = result.get("d3_metrics", [])
        total_d3_demotes = sum(m.get("d3_demotes", 0) for m in d3)
        total_d3_promotes = sum(m.get("d3_promotes", 0) for m in d3)

        summary[mode] = {
            "hit_rate": hit_rate,
            "hit_rate_pct": f"{hit_rate*100:.0f}%",
            "hits": hits,
            "total": total,
            "per_category": cat_rates,
            "avg_latency_s": round(avg_lat, 2),
            "p95_latency_s": round(p95_lat, 2),
            "last_tier": last_tier,
            "total_promotes": total_promotes,
            "total_demotes": total_demotes,
            "d3_accelerated_demotes": total_d3_demotes,
            "d3_preemptive_promotes": total_d3_promotes,
        }

    return summary


def print_comparison(summary: dict):
    """Pretty-print the comparison table."""
    print(f"\n{'='*70}")
    print("  PHASE D BENCHMARK RESULTS")
    print(f"{'='*70}\n")

    modes = list(summary.keys())
    header = f"{'Metric':<30}" + "".join(f"{m:<20}" for m in modes)
    print(header)
    print("-" * len(header))

    # Hit rate
    row = f"{'Hit Rate':<30}"
    for m in modes:
        s = summary[m]
        if "error" in s:
            row += f"{'ERROR':<20}"
        else:
            row += f"{s['hit_rate_pct']} ({s['hits']}/{s['total']})"
            row += " " * max(0, 20 - len(f"{s['hit_rate_pct']} ({s['hits']}/{s['total']})"))
    print(row)

    # Avg latency
    row = f"{'Avg Latency':<30}"
    for m in modes:
        s = summary[m]
        if "error" in s:
            row += f"{'—':<20}"
        else:
            row += f"{s['avg_latency_s']:.1f}s{'':<15}"
    print(row)

    # P95 latency
    row = f"{'P95 Latency':<30}"
    for m in modes:
        s = summary[m]
        if "error" in s:
            row += f"{'—':<20}"
        else:
            row += f"{s['p95_latency_s']:.1f}s{'':<15}"
    print(row)

    # Per-category breakdown
    all_cats = set()
    for s in summary.values():
        if "per_category" in s:
            all_cats.update(s["per_category"].keys())

    for cat in sorted(all_cats):
        row = f"  {cat:<28}"
        for m in modes:
            s = summary[m]
            if "per_category" in s and cat in s["per_category"]:
                rate = s["per_category"][cat]
                row += f"{rate*100:.0f}%{'':<17}"
            else:
                row += f"{'—':<20}"
        print(row)

    # Tier metrics (Phase D only)
    print()
    for m in modes:
        s = summary[m]
        if s.get("total_promotes", 0) > 0 or s.get("total_demotes", 0) > 0:
            lt = s.get("last_tier", {})
            print(f"  {m} tier dynamics:")
            print(f"    Promotions: {s['total_promotes']}, Demotions: {s['total_demotes']}")
            if s.get("d3_accelerated_demotes", 0) > 0 or s.get("d3_preemptive_promotes", 0) > 0:
                print(f"    D3 feedback: {s['d3_accelerated_demotes']} accel demotes, {s['d3_preemptive_promotes']} preemptive promotes")
            if lt:
                print(f"    Final distribution: Α={lt.get('alpha',0)} Β={lt.get('beta',0)} Γ={lt.get('gamma',0)}")

    # Decision gate
    print(f"\n{'='*70}")
    print("  DECISION GATE")
    print(f"{'='*70}")

    for m in modes:
        s = summary[m]
        if "error" in s:
            print(f"  {m}: ❌ ERROR")
            continue
        hr_ok = s["hit_rate"] >= 0.85
        lat_ok = s["avg_latency_s"] < 3.0
        verdict = "✅ PASS" if (hr_ok and lat_ok) else "❌ FAIL"
        hr_str = f"{'✅' if hr_ok else '❌'} hit_rate={s['hit_rate_pct']} (≥85%)"
        lat_str = f"{'✅' if lat_ok else '❌'} latency={s['avg_latency_s']:.1f}s (<3s)"
        print(f"  {m}: {verdict}")
        print(f"    {hr_str}")
        print(f"    {lat_str}")


def main():
    if not os.path.exists(BINARY):
        print(f"Binary not found: {BINARY}")
        print("Run: cargo build --release")
        sys.exit(1)

    if not os.path.exists(MODEL):
        print(f"Model not found: {MODEL}")
        sys.exit(1)

    if not os.path.isdir(GRAPH_DB):
        print(f"Graph DB not found: {GRAPH_DB}")
        print("Run: cargo run --release --example gen_test_graph -- /tmp/obrain-d-bench-graph")
        sys.exit(1)

    all_results = []

    # Mode 1: Phase B — text only (no embeddings)
    r = run_benchmark("Phase_B_text", [])
    all_results.append(r)

    # Mode 2: Phase C — embeddings, no tiers
    r = run_benchmark("Phase_C_embd", [
        "--embd-injection-ratio", "1.0",
    ])
    all_results.append(r)

    # Mode 3: Phase D — embeddings + tiers + Hilbert
    r = run_benchmark("Phase_D_hilbert", [
        "--embd-injection-ratio", "1.0",
        "--hilbert",
    ])
    all_results.append(r)

    # Analyze
    summary = analyze_results(all_results)
    print_comparison(summary)

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), "phase_d_results.json")
    with open(out_path, "w") as f:
        json.dump({"results": all_results, "summary": {k: {kk: vv for kk, vv in v.items() if kk != "per_category" or True} for k, v in summary.items()}}, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
