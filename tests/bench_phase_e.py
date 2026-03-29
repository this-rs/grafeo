#!/usr/bin/env python3
"""
Phase E Benchmark — Validates self-structuring retrieval (co-activation + affinity).

Single session, 2 passes of the same 12 questions:
  - Pass 1: cold start (no co-activations) — should be ≈ Phase D
  - Pass 2: same questions (co-activations accumulated from pass 1) — should improve

Metrics:
  - Hit rate per pass
  - Latency per query
  - E3 affinity expansion (λ value, candidates)
  - E4 re-layout trigger (if round_count hits threshold)
  - Tier distribution

Decision gate:
  - Pass 1 ≥ Phase D baseline (83% ± 5%)
  - Pass 2 ≥ Pass 1 hit rate
  - Pierre Bernard question passes (E1 property rendering)
  - Latency overhead < 500ms vs Phase D
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

# Same 12 questions as Phase D benchmark
QUESTIONS = [
    {"q": "Où habite Thomas Rivière ?", "kw": ["Lyon"], "cat": "direct_recall"},
    {"q": "Sur quel projet travaille Sophie Martin ?", "kw": ["DataPipeline"], "cat": "direct_recall"},
    {"q": "Quel langage utilise Marc Dupont ?", "kw": ["Go"], "cat": "direct_recall"},
    {"q": "Qui habite dans la même ville que Thomas Rivière ?", "kw": ["Marc"], "cat": "multi_hop"},
    {"q": "Quelles technologies sont utilisées par le projet Obrain ?", "kw": ["Rust"], "cat": "multi_hop"},
    {"q": "Qui connaît Thomas Rivière ?", "kw": ["Marc", "Alice"], "cat": "relationship"},
    {"q": "À quel événement Thomas et Marc ont-ils assisté ensemble ?", "kw": ["RustConf"], "cat": "relationship"},
    {"q": "Combien de personnes utilisent Python ?", "kw": ["Sophie", "Alice"], "cat": "aggregation"},
    {"q": "Liste toutes les villes mentionnées dans le graphe.", "kw": ["Lyon", "Paris", "Grenoble", "Toulouse"], "cat": "aggregation"},
    {"q": "Décris le profil professionnel de Thomas Rivière.", "kw": ["Rust", "Mozilla", "Obrain"], "cat": "context"},
    {"q": "Quelles sont les compétences de Pierre Bernard ?", "kw": ["DevOps", "Bash", "Airbus"], "cat": "context"},
    {"q": "Quel est le lien entre Alice Chen et NeuralSearch ?", "kw": ["travaille", "INRIA"], "cat": "relationship"},
]


def clean_response(raw: str, query: str) -> str:
    """Extract the model's response from raw pexpect output."""
    lines = raw.split("\n")
    result = []
    found_query = False

    for line in lines:
        clean = ANSI_RE.sub('', line).strip()
        if not clean:
            continue
        if not found_query:
            if query.strip()[:30] in clean:
                found_query = True
            continue
        if clean.startswith(("[", "  [", "ggml_", "graph_", "init:", "update:", "Pipeline")):
            continue
        if "réflexion" in clean or "spinner" in clean.lower() or "kernel_" in clean:
            continue
        if clean.startswith("you>") or clean.startswith("/quit"):
            continue
        if "get_embeddings_ith" in clean or "the last position stored" in clean:
            continue
        if "Memory persisted" in clean:
            clean = re.sub(r'🧠.*$', '', clean).strip()
            if not clean:
                continue
        result.append(clean)

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


def extract_metrics(raw: str) -> dict:
    """Extract all E3/E4/D2/D3 metrics from raw output."""
    metrics = {}

    # E3 affinity
    m = re.search(r'\[E3\] affinity: λ=([\d.]+), (\d+) candidates \((\d+) new\)', raw)
    if m:
        metrics["e3_lambda"] = float(m.group(1))
        metrics["e3_candidates"] = int(m.group(2))
        metrics["e3_new"] = int(m.group(3))

    # E4 re-layout
    m = re.search(r'\[E4\] Hilbert re-layout: (\d+) nodes, (\d+) co-activation edges', raw)
    if m:
        metrics["e4_nodes"] = int(m.group(1))
        metrics["e4_coact_edges"] = int(m.group(2))

    # E5 bank resegment
    m = re.search(r'\[E5\] bank resegment: (\d+) nodes migrated', raw)
    if m:
        metrics["e5_migrated"] = int(m.group(1))

    # D2 tier distribution
    m = re.search(r'\[D2\] round (\d+): \+(\d+) promoted, -(\d+) demoted \(Α=(\d+) Β=(\d+) Γ=(\d+)\)', raw)
    if m:
        metrics["d2_round"] = int(m.group(1))
        metrics["d2_promoted"] = int(m.group(2))
        metrics["d2_demoted"] = int(m.group(3))
        metrics["d2_alpha"] = int(m.group(4))
        metrics["d2_beta"] = int(m.group(5))
        metrics["d2_gamma"] = int(m.group(6))

    # D3 reward
    m = re.search(r'\[D3\] reward feedback: (\d+) accelerated demotes, (\d+) preemptive promotes', raw)
    if m:
        metrics["d3_demotes"] = int(m.group(1))
        metrics["d3_promotes"] = int(m.group(2))

    return metrics


def run_pass(child, pass_name: str) -> dict:
    """Run one pass of all questions on an existing session."""
    results = {"pass": pass_name, "questions": []}

    for i, q in enumerate(QUESTIONS):
        print(f"  [{pass_name}][{i+1}/{len(QUESTIONS)}] {q['cat']}: {q['q'][:50]}...", end=" ", flush=True)
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

        hits = sum(1 for kw in q["kw"] if kw.lower() in response_lower)
        hit = hits >= 1
        hit_ratio = hits / len(q["kw"])

        metrics = extract_metrics(raw)

        status = "✅" if hit else "❌"
        extra = ""
        if metrics.get("e3_lambda", 0) > 0:
            extra = f" λ={metrics['e3_lambda']:.2f}"
        print(f"{status} ({latency:.1f}s, {hits}/{len(q['kw'])} kw{extra})")

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
            "metrics": metrics,
        })

    return results


def compute_stats(pass_result: dict) -> dict:
    """Compute summary stats for one pass."""
    qs = pass_result.get("questions", [])
    if not qs:
        return {"error": "no questions"}

    hits = sum(1 for q in qs if q["hit"])
    total = len(qs)
    hit_rate = hits / total if total > 0 else 0

    cats = defaultdict(lambda: {"hits": 0, "total": 0})
    for q in qs:
        cats[q["category"]]["total"] += 1
        if q["hit"]:
            cats[q["category"]]["hits"] += 1
    cat_rates = {c: v["hits"] / v["total"] for c, v in cats.items()}

    latencies = [q["latency"] for q in qs if "error" not in q]
    avg_lat = sum(latencies) / len(latencies) if latencies else 0

    # Pierre Bernard check
    pierre_q = next((q for q in qs if "Pierre" in q["query"]), None)
    pierre_hit = pierre_q["hit"] if pierre_q else None

    # E3 stats
    e3_lambdas = [q["metrics"].get("e3_lambda", 0) for q in qs if "metrics" in q]
    avg_lambda = sum(e3_lambdas) / len(e3_lambdas) if e3_lambdas else 0

    return {
        "hit_rate": hit_rate,
        "hit_rate_pct": f"{hit_rate * 100:.0f}%",
        "hits": hits,
        "total": total,
        "per_category": cat_rates,
        "avg_latency_s": round(avg_lat, 2),
        "pierre_bernard_hit": pierre_hit,
        "avg_e3_lambda": round(avg_lambda, 3),
    }


def print_comparison(stats_pass1: dict, stats_pass2: dict):
    """Pretty-print the comparison."""
    print(f"\n{'=' * 70}")
    print("  PHASE E BENCHMARK RESULTS")
    print(f"{'=' * 70}\n")

    header = f"{'Metric':<30}{'Pass 1 (cold)':<20}{'Pass 2 (warm)':<20}"
    print(header)
    print("-" * 70)

    print(f"{'Hit Rate':<30}{stats_pass1['hit_rate_pct']} ({stats_pass1['hits']}/{stats_pass1['total']}){'':<8}"
          f"{stats_pass2['hit_rate_pct']} ({stats_pass2['hits']}/{stats_pass2['total']})")

    print(f"{'Avg Latency':<30}{stats_pass1['avg_latency_s']:.1f}s{'':<15}"
          f"{stats_pass2['avg_latency_s']:.1f}s")

    print(f"{'Avg λ (affinity)':<30}{stats_pass1['avg_e3_lambda']:.3f}{'':<15}"
          f"{stats_pass2['avg_e3_lambda']:.3f}")

    pb1 = "✅" if stats_pass1.get("pierre_bernard_hit") else "❌"
    pb2 = "✅" if stats_pass2.get("pierre_bernard_hit") else "❌"
    print(f"{'Pierre Bernard':<30}{pb1:<20}{pb2}")

    print()
    all_cats = set(list(stats_pass1.get("per_category", {}).keys()) +
                   list(stats_pass2.get("per_category", {}).keys()))
    for cat in sorted(all_cats):
        r1 = stats_pass1.get("per_category", {}).get(cat, 0)
        r2 = stats_pass2.get("per_category", {}).get(cat, 0)
        delta = r2 - r1
        delta_str = f" ({'+' if delta > 0 else ''}{delta * 100:.0f}%)" if delta != 0 else ""
        print(f"  {cat:<28}{r1 * 100:.0f}%{'':<17}{r2 * 100:.0f}%{delta_str}")

    # Decision gate
    print(f"\n{'=' * 70}")
    print("  DECISION GATE")
    print(f"{'=' * 70}")

    phase_d_baseline = 0.83  # Phase D hit rate from previous benchmark
    pass1_ok = stats_pass1["hit_rate"] >= phase_d_baseline - 0.05
    pass2_ge_pass1 = stats_pass2["hit_rate"] >= stats_pass1["hit_rate"]
    pierre_ok = stats_pass2.get("pierre_bernard_hit", False)
    latency_ok = stats_pass2["avg_latency_s"] < phase_d_baseline + 0.5  # not used, use absolute

    # More meaningful latency check: pass2 < pass1 + 500ms overhead
    latency_overhead = stats_pass2["avg_latency_s"] - stats_pass1["avg_latency_s"]
    lat_ok = latency_overhead < 0.5

    checks = [
        (pass1_ok, f"Pass 1 ≥ Phase D - 5% ({stats_pass1['hit_rate_pct']} vs {phase_d_baseline * 100:.0f}%)"),
        (pass2_ge_pass1, f"Pass 2 ≥ Pass 1 ({stats_pass2['hit_rate_pct']} vs {stats_pass1['hit_rate_pct']})"),
        (pierre_ok, f"Pierre Bernard question passes"),
        (lat_ok, f"Latency overhead < 500ms ({latency_overhead * 1000:.0f}ms)"),
    ]

    all_pass = True
    for ok, desc in checks:
        print(f"  {'✅' if ok else '❌'} {desc}")
        if not ok:
            all_pass = False

    verdict = "✅ PHASE E VALIDATED" if all_pass else "❌ PHASE E NEEDS WORK"
    print(f"\n  {verdict}")


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

    # Launch a SINGLE session with Phase D+E features enabled
    cmd_args = [
        BINARY,
        "--model", MODEL,
        "--db", GRAPH_DB,
        "--n-ctx", N_CTX,
        "--n-gpu", "99",
        "--max-nodes", "20",
        "--embd-injection-ratio", "1.0",
        "--hilbert",
    ]

    print(f"\n{'=' * 60}")
    print(f"  Phase E Benchmark — Single session, 2 passes")
    print(f"  Binary: {BINARY}")
    print(f"  Model: {os.path.basename(MODEL)}")
    print(f"{'=' * 60}")

    try:
        child = pexpect.spawn(" ".join(cmd_args), timeout=120, encoding='utf-8', maxread=200000)
        child.expect("you>", timeout=240)
        time.sleep(0.5)
        try:
            while True:
                child.read_nonblocking(size=4096, timeout=0.5)
        except (pexpect.TIMEOUT, pexpect.EOF):
            pass
    except Exception as e:
        print(f"  ❌ Failed to start: {e}")
        sys.exit(1)

    # Pass 1: Cold start (no co-activations yet)
    print(f"\n{'—' * 60}")
    print("  PASS 1 — Cold start (co-activations empty, λ=0)")
    print(f"{'—' * 60}")
    pass1 = run_pass(child, "Pass1_cold")

    # Pass 2: Same questions (co-activations from pass 1)
    print(f"\n{'—' * 60}")
    print("  PASS 2 — Warm (co-activations from pass 1)")
    print(f"{'—' * 60}")
    pass2 = run_pass(child, "Pass2_warm")

    # Cleanup
    try:
        child.sendline("/quit")
        child.expect(pexpect.EOF, timeout=10)
    except:
        child.close()

    # Analyze
    stats1 = compute_stats(pass1)
    stats2 = compute_stats(pass2)
    print_comparison(stats1, stats2)

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), "phase_e_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "pass1": pass1,
            "pass2": pass2,
            "stats_pass1": stats1,
            "stats_pass2": stats2,
        }, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
