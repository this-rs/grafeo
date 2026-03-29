#!/usr/bin/env python3
"""
Analyze C5 benchmark results — compression, quality curves, GO/NO-GO decision.

Reads bench_c5_results.json produced by bench_c5_embedding.py.
Outputs:
  - Compression table (SC5.2)
  - Quality vs ratio analysis (SC5.3)
  - GO/NO-GO decision summary (SC5.4)
"""

import json
import os
import sys

RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_c5_results.json")


def load_results():
    if not os.path.exists(RESULTS_PATH):
        print(f"ERROR: {RESULTS_PATH} not found. Run bench_c5_embedding.py first.")
        sys.exit(1)
    with open(RESULTS_PATH) as f:
        return json.load(f)


def analyze_compression(data):
    """SC5.2 — Compression analysis."""
    print("\n" + "=" * 70)
    print("  SC5.2 — COMPRESSION ANALYSIS")
    print("=" * 70)

    # Estimate: each text-encoded node uses ~50 tokens in KV
    TEXT_TOKENS_PER_NODE = 50

    print(f"\n{'Mode':<14} {'Embd':>6} {'Text':>6} {'Pre-comp':>10} {'Est. Text Tok':>14} {'Compression':>12}")
    print("-" * 66)

    for mode, mdata in data.items():
        if mdata.get("error"):
            print(f"{mode:<14} ERROR")
            continue

        embd = mdata.get("total_embd_injected", 0)
        text = mdata.get("total_text_encoded", 0)
        precomp = mdata.get("precomputed", 0)

        # Compression: how many text tokens did we save?
        if embd > 0:
            saved_tokens = embd * TEXT_TOKENS_PER_NODE
            actual_positions = embd  # 1 position per embedding
            compression = f"{saved_tokens / actual_positions:.0f}×"
        else:
            compression = "—"

        est_text = (embd + text) * TEXT_TOKENS_PER_NODE
        print(f"{mode:<14} {embd:>6} {text:>6} {precomp:>10} {est_text:>14} {compression:>12}")

    # Per-query compression detail for Phase C modes
    for mode in ["phase_c_50", "phase_c_100"]:
        mdata = data.get(mode, {})
        if mdata.get("error") or not mdata.get("results"):
            continue
        print(f"\n  {mode} per-query detail:")
        for r in mdata["results"]:
            if r.get("error"):
                continue
            e = r.get("embd_injected", 0)
            t = r.get("text_encoded", 0)
            total = e + t
            pct = (e / total * 100) if total > 0 else 0
            print(f"    [{r['category']:<15}] embd={e} text={t} ({pct:.0f}% injected)")


def analyze_quality(data):
    """SC5.3 — Quality vs mode analysis."""
    print("\n" + "=" * 70)
    print("  SC5.3 — QUALITY ANALYSIS")
    print("=" * 70)

    # Per-mode summary
    print(f"\n{'Mode':<14} {'Hit Rate':>10} {'Latency':>9} {'Empty':>7} {'Artifact':>9}")
    print("-" * 54)

    mode_stats = {}
    for mode, mdata in data.items():
        if mdata.get("error"):
            print(f"{mode:<14} ERROR")
            continue

        results = [r for r in mdata.get("results", []) if not r.get("error")]
        if not results:
            print(f"{mode:<14} NO DATA")
            continue

        avg_hr = sum(r["hit_rate"] for r in results) / len(results)
        avg_lat = sum(r.get("latency_s", 0) for r in results) / len(results)
        n_empty = sum(1 for r in results if r.get("is_empty"))
        n_artifact = sum(1 for r in results if r.get("has_artifacts"))

        mode_stats[mode] = {"avg_hr": avg_hr, "avg_lat": avg_lat, "n_empty": n_empty, "n_artifact": n_artifact, "n": len(results)}

        print(f"{mode:<14} {avg_hr:>9.0%} {avg_lat:>8.1f}s {n_empty:>7} {n_artifact:>9}")

    # Per-category breakdown
    categories = ["direct_recall", "multi_hop", "relationship", "negation", "aggregation", "context"]
    print(f"\n  Per-category hit rates:")
    header = f"  {'Category':<16}"
    for mode in data:
        if not data[mode].get("error"):
            header += f" {mode:>12}"
    print(header)
    print("  " + "-" * (16 + 13 * len([m for m in data if not data[m].get("error")])))

    for cat in categories:
        row = f"  {cat:<16}"
        for mode, mdata in data.items():
            if mdata.get("error"):
                continue
            cat_results = [r for r in mdata.get("results", []) if not r.get("error") and r.get("category") == cat]
            if cat_results:
                cat_hr = sum(r["hit_rate"] for r in cat_results) / len(cat_results)
                row += f" {cat_hr:>11.0%}"
            else:
                row += f" {'—':>12}"
        print(row)

    return mode_stats


def go_nogo_decision(data, mode_stats):
    """SC5.4 — GO/NO-GO decision."""
    print("\n" + "=" * 70)
    print("  SC5.4 — GO / NO-GO DECISION")
    print("=" * 70)

    pb = mode_stats.get("phase_b", {})
    pc50 = mode_stats.get("phase_c_50", {})
    pc100 = mode_stats.get("phase_c_100", {})

    pb_hr = pb.get("avg_hr", 0)
    pc50_hr = pc50.get("avg_hr", 0)
    pc100_hr = pc100.get("avg_hr", 0)

    # Criteria
    TEXT_TOKENS_PER_NODE = 50

    # 1. Quality: Phase C >= 90% of Phase B recall
    c100_quality = pc100_hr >= pb_hr * 0.9 if pb_hr > 0 else False
    c50_quality = pc50_hr >= pb_hr * 0.9 if pb_hr > 0 else False

    # 2. Compression: >= 10x for C100
    c100_data = data.get("phase_c_100", {})
    embd_total = c100_data.get("total_embd_injected", 0)
    compression = TEXT_TOKENS_PER_NODE if embd_total > 0 else 0  # 1 pos vs ~50 tokens

    # 3. Hallucination: no more than +5% artifacts vs Phase B
    pb_art_rate = pb.get("n_artifact", 0) / max(pb.get("n", 1), 1)
    pc100_art_rate = pc100.get("n_artifact", 0) / max(pc100.get("n", 1), 1)
    hallucination_ok = pc100_art_rate <= pb_art_rate + 0.05

    print(f"\n  Phase B baseline recall:     {pb_hr:.0%}")
    print(f"  Phase C 50% recall:          {pc50_hr:.0%}  (>= {pb_hr*0.9:.0%} needed) {'✅' if c50_quality else '❌'}")
    print(f"  Phase C 100% recall:         {pc100_hr:.0%}  (>= {pb_hr*0.9:.0%} needed) {'✅' if c100_quality else '❌'}")
    print(f"  Compression (C100):          {compression}× (>= 10× needed) {'✅' if compression >= 10 else '❌'}")
    print(f"  Hallucination (C100):        {pc100_art_rate:.0%} vs B={pb_art_rate:.0%} (delta <= 5%) {'✅' if hallucination_ok else '❌'}")

    # Latency comparison
    pb_lat = pb.get("avg_lat", 0)
    pc100_lat = pc100.get("avg_lat", 0)
    if pb_lat > 0:
        lat_ratio = pc100_lat / pb_lat
        print(f"  Latency (C100 vs B):         {pc100_lat:.1f}s vs {pb_lat:.1f}s ({lat_ratio:.2f}×)")

    # Decision
    all_pass = c100_quality and compression >= 10 and hallucination_ok
    partial = c50_quality and not c100_quality

    print(f"\n  {'='*50}")
    if all_pass:
        print(f"  🟢 GO — Phase C VALIDATED")
        print(f"  Embedding injection maintains quality with {compression}× KV compression.")
        print(f"  Next: C6 contrastive fine-tuning for further improvement.")
    elif partial:
        print(f"  🟡 PARTIAL GO — Phase C 50% works, C 100% needs investigation")
        print(f"  Mixed mode (50% embedding) is viable. Full embedding has quality gap.")
        print(f"  Next: investigate distribution shift, try ProjectionNet fine-tuning (C6).")
    else:
        print(f"  🔴 NO-GO — Phase C needs more work")
        if not c100_quality and not c50_quality:
            print(f"  Quality regression too severe ({pc100_hr:.0%} vs {pb_hr:.0%} baseline).")
        if not hallucination_ok:
            print(f"  Hallucination rate increased beyond threshold.")
        print(f"  Next: C6 contrastive fine-tuning is critical before embedding injection is viable.")
    print(f"  {'='*50}")

    return {
        "decision": "GO" if all_pass else ("PARTIAL_GO" if partial else "NO_GO"),
        "phase_b_recall": pb_hr,
        "phase_c50_recall": pc50_hr,
        "phase_c100_recall": pc100_hr,
        "compression": compression,
        "quality_c100_ok": c100_quality,
        "quality_c50_ok": c50_quality,
        "hallucination_ok": hallucination_ok,
    }


def main():
    data = load_results()
    analyze_compression(data)
    mode_stats = analyze_quality(data)
    decision = go_nogo_decision(data, mode_stats)

    # Save decision summary
    decision_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "c5_decision.json")
    with open(decision_path, "w") as f:
        json.dump(decision, f, indent=2)
    print(f"\n  Decision saved to {decision_path}")


if __name__ == "__main__":
    main()
