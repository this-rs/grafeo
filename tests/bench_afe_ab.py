#!/usr/bin/env python3
"""
Phase 4 AFE A/B Benchmark — Adaptive Formula vs Identity.

Step T0.6.4: Compare the adaptive formula selection engine against the
Identity baseline (no formula overlay) on the 12 Phase D questions.

Decision gate: Adaptive ≥ Identity on ≥ 3/5 categories.

Categories: direct_recall, multi_hop, relationship, aggregation, context
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


def extract_afe_log(raw: str) -> dict:
    """Extract [AFE] logs from output."""
    info = {}
    m = re.search(r'\[AFE\] selected formula: (\S+) \(score=([0-9.]+)\)', raw)
    if m:
        info["formula"] = m.group(1)
        info["score"] = float(m.group(2))
    m = re.search(r'\[AFE\] formula overlay applied: (\d+) cells modified', raw)
    if m:
        info["cells_modified"] = int(m.group(1))
    m = re.search(r'\[AFE\] compiled formula.*density=([0-9.]+), n_nodes=(\d+)', raw)
    if m:
        info["density"] = float(m.group(1))
        info["n_nodes"] = int(m.group(2))
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
    persona_dir = f"/tmp/obrain-afe-bench-{mode_name.lower().replace(' ', '_')}"
    if use_persona:
        cmd_args += ["--persona", persona_dir]

    # Set up environment
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    print(f"\n{'='*60}")
    print(f"  Mode: {mode_name}")
    if extra_env:
        print(f"  Env: {extra_env}")
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
        "afe_logs": [],
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

        # Extract AFE logs
        afe = extract_afe_log(raw)
        results["afe_logs"].append(afe)

        status = "✅" if hit else "❌"
        formula_info = f", formula={afe.get('formula', '?')}" if afe.get('formula') else ""
        cells_info = f", cells={afe.get('cells_modified', 0)}" if afe.get('cells_modified') else ""
        print(f"{status} ({latency:.1f}s, {hits}/{len(q['kw'])} kw{formula_info}{cells_info})")

        if not hit:
            snippet = response[:200] if response else "(empty)"
            print(f"       Expected: {q['kw']}")
            print(f"       Got: {snippet}")
            # Debug: show raw output for diagnosis
            raw_clean = ANSI_RE.sub('', raw).strip()
            if "Error:" in raw_clean or "llama_decode" in raw_clean or "position" in raw_clean.lower():
                for line in raw_clean.split('\n'):
                    line = line.strip()
                    if any(k in line for k in ["Error:", "llama_decode", "position", "init:", "M-RoPE", "seq_pos"]):
                        print(f"       [DEBUG] {line[:200]}")

        results["questions"].append({
            "category": q["cat"],
            "query": q["q"],
            "hit": hit,
            "hit_ratio": hit_ratio,
            "latency": latency,
            "keywords_found": hits,
            "keywords_total": len(q["kw"]),
            "afe": afe,
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

        # AFE formula distribution
        formulas_used = defaultdict(int)
        for q in qs:
            f = q.get("afe", {}).get("formula", "none")
            formulas_used[f] += 1

        summary[mode] = {
            "hit_rate": hits / total if total > 0 else 0,
            "hit_rate_pct": f"{hits/total*100:.0f}%" if total > 0 else "0%",
            "hits": hits,
            "total": total,
            "per_category": cat_rates,
            "avg_latency_s": round(avg_lat, 2),
            "formulas_used": dict(formulas_used),
        }

    return summary


def print_comparison(summary: dict):
    """Pretty-print comparison."""
    modes = list(summary.keys())

    print(f"\n{'='*70}")
    print("  AFE A/B BENCHMARK — Adaptive Formula vs Identity")
    print(f"{'='*70}\n")

    header = f"{'Metric':<25}" + "".join(f"{m:<25}" for m in modes)
    print(header)
    print("-" * len(header))

    # Overall
    row = f"{'Hit Rate':<25}"
    for m in modes:
        s = summary[m]
        if "error" in s:
            row += f"{'ERROR':<25}"
        else:
            row += f"{s['hit_rate_pct']} ({s['hits']}/{s['total']})"
            row += " " * max(0, 25 - len(f"{s['hit_rate_pct']} ({s['hits']}/{s['total']})"))
    print(row)

    row = f"{'Avg Latency':<25}"
    for m in modes:
        s = summary[m]
        row += f"{s.get('avg_latency_s', 0):.1f}s{'':<22}" if "error" not in s else f"{'—':<25}"
    print(row)

    # Per-category
    all_cats = sorted(set(c for s in summary.values() if "per_category" in s for c in s["per_category"]))
    print()
    print(f"{'Category':<25}" + "".join(f"{m:<25}" for m in modes))
    print("-" * (25 + 25 * len(modes)))

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
                row += f"{rate*100:.0f}% ({cr['hits']}/{cr['total']}){'':<14}"
            else:
                row += f"{'—':<25}"
        print(row)
        if rates:
            best_rate = max(rates.values())
            if best_rate > 0:
                best_modes = [m for m, r in rates.items() if r == best_rate]
                if len(best_modes) == 1:
                    wins[best_modes[0]] += 1
                else:
                    ties += 1  # Tie — both get credit

    # Formula distribution
    print()
    for m in modes:
        s = summary[m]
        if "formulas_used" in s and s["formulas_used"]:
            print(f"  {m} formulas: {s['formulas_used']}")

    # Decision gate
    print(f"\n{'='*70}")
    print("  DECISION GATE: Adaptive ≥ Identity on ≥ 3/5 categories")
    print(f"{'='*70}")

    for m in modes:
        print(f"  {m}: wins {wins.get(m, 0)}/5 categories")
    if ties > 0:
        print(f"  Ties: {ties}/5 categories")

    # Compare adaptive vs identity
    adaptive_mode = [m for m in modes if "adaptive" in m.lower() or "afe" in m.lower()]
    identity_mode = [m for m in modes if "identity" in m.lower() or "baseline" in m.lower()]

    if adaptive_mode and identity_mode:
        a_wins = wins[adaptive_mode[0]]
        i_wins = wins[identity_mode[0]]
        a_effective = a_wins + ties  # Ties count as non-regression
        if a_effective >= 3:
            if a_wins > i_wins:
                print(f"\n  ✅ GO — Adaptive wins {a_wins}/5 categories (≥3 required)")
            elif ties > 0:
                print(f"\n  ✅ GO (TIE) — Adaptive wins {a_wins} + ties {ties} = {a_effective}/5 (no regression)")
            else:
                print(f"\n  ✅ GO — Adaptive wins {a_wins}/5 categories (≥3 required)")
        elif a_wins == i_wins and a_wins == 0:
            print(f"\n  ⚠️  TIE — Both win {a_wins}/5 categories, {ties} ties")
        else:
            print(f"\n  ❌ NO-GO — Adaptive wins only {a_wins}/5 categories (<3)")


def main():
    global BINARY
    if not os.path.exists(BINARY):
        # Try target/release as fallback (dynamic build)
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

    # Common env: increase mask budget to accommodate self-embed position
    base_env = {"OBRAIN_MASK_BUDGET_MB": "128"}

    # Mode 1: Identity baseline (no AFE formulas — disable via env)
    r = run_benchmark("Identity_baseline", extra_env={**base_env, "OBRAIN_AFE_DISABLE": "1"})
    all_results.append(r)

    # Mode 2: Adaptive formula selection (AFE enabled, default)
    r = run_benchmark("AFE_adaptive", extra_env=base_env)
    all_results.append(r)

    # Analyze
    summary = analyze_and_compare(all_results)
    print_comparison(summary)

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), "afe_ab_results.json")
    with open(out_path, "w") as f:
        json.dump({"results": all_results, "summary": summary}, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
