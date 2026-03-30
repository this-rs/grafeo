#!/usr/bin/env python3
"""
Phase 5 IPTR + State kq_b Benchmark — 3-Mode A/B/C Comparison.

Compares three modes on the 12 Phase D questions:
  Mode A — Baseline:     No IPTR, no state kq_b (both disabled)
  Mode B — IPTR only:    GraphLookupTool active, no state kq_b
  Mode C — IPTR + kq_b:  Both axes active

Decision gate: IPTR+kq_b ≥ Baseline on ≥ 4/5 categories.

Expected improvements:
  - multi_hop:     0% → ~100% (BFS 2-hops finds indirect relations)
  - relationship: 33% → ~67-100% (BFS traverses relationship edges)
  - global:       42% → ~75-85%

Environment variables:
  OBRAIN_IPTR_DISABLE=1     → disable IPTR dispatcher
  OBRAIN_STATE_BIAS_DISABLE=1 → disable state kq_b attention bias
  OBRAIN_IPTR_COOLDOWN=N    → tokens between IPTR triggers (default: 16)
"""

import pexpect
import sys
import os
import re
import time
import json
from collections import defaultdict

BINARY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dist", "obrain-chat-macos-arm64")
MODEL = os.path.expanduser("~/models/Qwen3-14B-Claude-4.5-Opus-Distill.q4_k_m.gguf")
GRAPH_DB = "/tmp/obrain-d-bench-graph"
N_CTX = "4096"

ANSI_RE = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]|\x1b\[\?[0-9]+[hl]|\[2K|\[5C')

# 12 questions covering 5 categories (same as Phase D benchmark)
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
    {"q": "Quel est le lien entre Alice Chen et NeuralSearch ?", "kw": ["travaille", "INRIA"], "cat": "relationship"},

    # Aggregation
    {"q": "Combien de personnes utilisent Python ?", "kw": ["Sophie", "Alice"], "cat": "aggregation"},
    {"q": "Liste toutes les villes mentionnées dans le graphe.", "kw": ["Lyon", "Paris", "Grenoble", "Toulouse"], "cat": "aggregation"},

    # Context / meta
    {"q": "Décris le profil professionnel de Thomas Rivière.", "kw": ["Rust", "Mozilla", "Obrain"], "cat": "context"},
    {"q": "Quelles sont les compétences de Pierre Bernard ?", "kw": ["DevOps", "Bash", "Airbus"], "cat": "context"},
]


def clean_response(raw: str, query: str) -> str:
    """Extract model response from raw pexpect output."""
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
        # Skip log lines
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


def extract_iptr_log(raw: str) -> dict:
    """Extract [IPTR] and [StateBias] logs from output."""
    info = {"iptr_triggers": 0, "iptr_total_biases": 0, "iptr_concepts": []}

    # Count IPTR triggers
    triggers = re.findall(r'\[IPTR\] graph_lookup triggered: (\d+) biases, (\d+) concepts', raw)
    info["iptr_triggers"] = len(triggers)
    for m in triggers:
        info["iptr_total_biases"] += int(m[0])

    # IPTR latency
    latencies = re.findall(r'\[IPTR\].*latency=(\d+)µs', raw)
    if latencies:
        info["iptr_latency_us"] = sum(int(l) for l in latencies)

    # StateBias info
    m = re.search(r'\[StateBias\] reward=([0-9.-]+), confidence=([0-9.-]+), magnitude=([0-9.]+)', raw)
    if m:
        info["state_bias_reward"] = float(m.group(1))
        info["state_bias_confidence"] = float(m.group(2))
        info["state_bias_magnitude"] = float(m.group(3))

    # Dispatcher ready
    m = re.search(r'\[IPTR\] dispatcher ready: (\d+) facts', raw)
    if m:
        info["iptr_facts_loaded"] = int(m.group(1))

    return info


def run_benchmark(mode_name: str, extra_env: dict = None, use_persona: bool = True) -> dict:
    """Run one benchmark pass."""
    cmd_args = [
        BINARY,
        "--model", MODEL,
        "--db", GRAPH_DB,
        "--n-ctx", N_CTX,
        "--n-gpu", "99",
        "--max-nodes", "20",
    ]

    # Use a fresh persona for each run to avoid cross-contamination
    persona_dir = f"/tmp/obrain-iptr-bench-{mode_name.lower().replace(' ', '_')}"
    if use_persona:
        cmd_args += ["--persona", persona_dir]

    # Set up environment
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    print(f"\n{'='*60}")
    print(f"  Mode: {mode_name}")
    if extra_env:
        for k, v in extra_env.items():
            if k.startswith("OBRAIN_"):
                print(f"    {k}={v}")
    print(f"{'='*60}")

    try:
        child = pexpect.spawn(" ".join(cmd_args), timeout=120, encoding='utf-8', maxread=200000, env=env)
        child.expect("you>", timeout=240)
        time.sleep(0.5)
        try:
            while True:
                child.read_nonblocking(size=4096, timeout=0.5)
        except (pexpect.TIMEOUT, pexpect.EOF):
            pass
    except Exception as e:
        print(f"  Failed to start: {e}")
        return {"mode": mode_name, "error": str(e)}

    results = {
        "mode": mode_name,
        "questions": [],
        "iptr_logs": [],
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
        hit = hits >= 1
        hit_ratio = hits / len(q["kw"])

        # Extract IPTR/StateBias logs
        iptr = extract_iptr_log(raw)
        results["iptr_logs"].append(iptr)

        status = "✅" if hit else "❌"
        iptr_info = ""
        if iptr["iptr_triggers"] > 0:
            iptr_info = f", iptr={iptr['iptr_triggers']}×{iptr['iptr_total_biases']}biases"
        bias_info = ""
        if "state_bias_magnitude" in iptr and iptr["state_bias_magnitude"] > 0:
            bias_info = f", kqb={iptr['state_bias_magnitude']:.3f}"
        print(f"{status} ({latency:.1f}s, {hits}/{len(q['kw'])} kw{iptr_info}{bias_info})")

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
            "iptr": iptr,
        })

    # Cleanup
    try:
        child.sendline("/quit")
        child.expect(pexpect.EOF, timeout=10)
    except:
        child.close()

    return results


def analyze_and_compare(all_results: list) -> dict:
    """Compare modes per category."""
    summary = {}

    for result in all_results:
        mode = result["mode"]
        qs = result.get("questions", [])
        if not qs:
            summary[mode] = {"error": result.get("error", "unknown")}
            continue

        hits = sum(1 for q in qs if q["hit"])
        total = len(qs)

        cats = defaultdict(lambda: {"hits": 0, "total": 0, "latencies": []})
        for q in qs:
            cats[q["category"]]["total"] += 1
            if q["hit"]:
                cats[q["category"]]["hits"] += 1
            cats[q["category"]]["latencies"].append(q["latency"])

        cat_rates = {}
        for c, v in cats.items():
            cat_rates[c] = {
                "hit_rate": v["hits"] / v["total"] if v["total"] > 0 else 0,
                "hits": v["hits"],
                "total": v["total"],
                "avg_latency": sum(v["latencies"]) / len(v["latencies"]) if v["latencies"] else 0,
            }

        latencies = [q["latency"] for q in qs if "error" not in q]
        avg_lat = sum(latencies) / len(latencies) if latencies else 0

        # IPTR stats
        iptr_total_triggers = sum(q.get("iptr", {}).get("iptr_triggers", 0) for q in qs)
        iptr_total_latency = sum(q.get("iptr", {}).get("iptr_latency_us", 0) for q in qs)

        summary[mode] = {
            "hit_rate": hits / total if total > 0 else 0,
            "hit_rate_pct": f"{hits/total*100:.0f}%" if total > 0 else "0%",
            "hits": hits,
            "total": total,
            "per_category": cat_rates,
            "avg_latency_s": round(avg_lat, 2),
            "iptr_triggers": iptr_total_triggers,
            "iptr_latency_us": iptr_total_latency,
        }

    return summary


def print_comparison(summary: dict):
    """Pretty-print comparison."""
    modes = list(summary.keys())

    print(f"\n{'='*80}")
    print("  PHASE 5 BENCHMARK — IPTR + State kq_b vs Baseline")
    print(f"{'='*80}\n")

    header = f"{'Metric':<25}" + "".join(f"{m:<20}" for m in modes)
    print(header)
    print("-" * len(header))

    # Overall hit rate
    row = f"{'Hit Rate':<25}"
    for m in modes:
        s = summary[m]
        if "error" in s:
            row += f"{'ERROR':<20}"
        else:
            cell = f"{s['hit_rate_pct']} ({s['hits']}/{s['total']})"
            row += f"{cell:<20}"
    print(row)

    # Latency
    row = f"{'Avg Latency':<25}"
    for m in modes:
        s = summary[m]
        row += f"{s.get('avg_latency_s', 0):.1f}s{'':<17}" if "error" not in s else f"{'—':<20}"
    print(row)

    # IPTR triggers
    row = f"{'IPTR Triggers':<25}"
    for m in modes:
        s = summary[m]
        row += f"{s.get('iptr_triggers', 0):<20}" if "error" not in s else f"{'—':<20}"
    print(row)

    # Per-category breakdown
    all_cats = sorted(set(c for s in summary.values() if "per_category" in s for c in s["per_category"]))
    print()
    print(f"{'Category':<25}" + "".join(f"{m:<20}" for m in modes))
    print("-" * (25 + 20 * len(modes)))

    wins = {m: 0 for m in modes}
    ties = 0
    for cat in all_cats:
        row = f"  {cat:<23}"
        rates = {}
        for m in modes:
            s = summary[m]
            if "per_category" in s and cat in s["per_category"]:
                cr = s["per_category"][cat]
                rate = cr["hit_rate"]
                rates[m] = rate
                cell = f"{rate*100:.0f}% ({cr['hits']}/{cr['total']})"
                row += f"{cell:<20}"
            else:
                row += f"{'—':<20}"
        print(row)
        if rates:
            best_rate = max(rates.values())
            if best_rate > 0:
                best_modes = [m for m, r in rates.items() if r == best_rate]
                if len(best_modes) == 1:
                    wins[best_modes[0]] += 1
                else:
                    ties += 1

    # Decision gate
    print(f"\n{'='*80}")
    print("  DECISION GATE: IPTR+kq_b ≥ Baseline on ≥ 4/5 categories")
    print(f"{'='*80}")

    for m in modes:
        print(f"  {m}: wins {wins.get(m, 0)}/5 categories")
    if ties > 0:
        print(f"  Ties: {ties}/5 categories")

    # Compare IPTR+kq_b vs Baseline
    best_mode = [m for m in modes if "iptr_kqb" in m.lower() or "iptr+kqb" in m.lower() or "full" in m.lower()]
    base_mode = [m for m in modes if "baseline" in m.lower()]

    if best_mode and base_mode:
        b_wins = wins[best_mode[0]]
        base_wins = wins[base_mode[0]]
        effective = b_wins + ties  # Ties = no regression
        if effective >= 4:
            print(f"\n  ✅ GO — IPTR+kq_b wins {b_wins}/5 categories + {ties} ties = {effective}/5 (≥4 required)")
        elif b_wins > base_wins:
            print(f"\n  ⚠️  PARTIAL — IPTR+kq_b wins {b_wins}/5 (needs 4, but leads)")
        else:
            print(f"\n  ❌ NO-GO — IPTR+kq_b wins only {b_wins}/5 categories (<4)")

    # Per-mode improvement deltas
    if len(modes) >= 2:
        print(f"\n  Improvement deltas vs {modes[0]}:")
        base_s = summary[modes[0]]
        for m in modes[1:]:
            s = summary[m]
            if "error" not in s and "error" not in base_s:
                delta = (s["hit_rate"] - base_s["hit_rate"]) * 100
                sign = "+" if delta >= 0 else ""
                print(f"    {m}: {sign}{delta:.0f}pp overall hit rate")
                # Per-cat deltas
                for cat in all_cats:
                    if cat in s.get("per_category", {}) and cat in base_s.get("per_category", {}):
                        d = (s["per_category"][cat]["hit_rate"] - base_s["per_category"][cat]["hit_rate"]) * 100
                        if abs(d) > 0.5:
                            sign = "+" if d >= 0 else ""
                            print(f"      {cat}: {sign}{d:.0f}pp")


def main():
    global BINARY
    if not os.path.exists(BINARY):
        alt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "target", "release", "obrain-chat")
        if os.path.exists(alt):
            BINARY = alt
        else:
            print(f"Binary not found: {BINARY}")
            print("Run: make dist")
            sys.exit(1)
    print(f"Using binary: {BINARY}")

    if not os.path.exists(MODEL):
        print(f"Model not found: {MODEL}")
        sys.exit(1)

    if not os.path.isdir(GRAPH_DB):
        print(f"Graph DB not found: {GRAPH_DB}")
        print("Run: cargo run --release --example gen_test_graph -- /tmp/obrain-d-bench-graph")
        sys.exit(1)

    all_results = []

    base_env = {"OBRAIN_MASK_BUDGET_MB": "128"}

    # Mode A: Baseline (both disabled)
    r = run_benchmark("A_baseline", extra_env={
        **base_env,
        "OBRAIN_IPTR_DISABLE": "1",
        "OBRAIN_STATE_BIAS_DISABLE": "1",
    })
    all_results.append(r)

    # Mode B: IPTR only (kq_b disabled)
    r = run_benchmark("B_iptr_only", extra_env={
        **base_env,
        "OBRAIN_STATE_BIAS_DISABLE": "1",
    })
    all_results.append(r)

    # Mode C: IPTR + kq_b (full Phase 5)
    r = run_benchmark("C_iptr_kqb_full", extra_env=base_env)
    all_results.append(r)

    # Analyze
    summary = analyze_and_compare(all_results)
    print_comparison(summary)

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), "iptr_results.json")
    with open(out_path, "w") as f:
        json.dump({"results": all_results, "summary": summary}, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
