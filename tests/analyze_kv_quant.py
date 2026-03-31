#!/usr/bin/env python3
"""
Analyze KV quantization benchmark results.
Compares f16 vs q8_0 vs q4_0 hit rates and latencies.
Produces GO/NO-GO verdict for P3 benchmark matrix.

Usage: python3 tests/analyze_kv_quant.py [results_dir]
"""

import json
import os
import sys

RESULTS_DIR = sys.argv[1] if len(sys.argv) > 1 else "tests/results/kv_quant"

# GO/NO-GO thresholds
Q8_HIT_RATE_THRESHOLD = 0.90  # q8_0 must retain >= 90% of f16 hit rate
Q4_HIT_RATE_THRESHOLD = 0.70  # q4_0 must retain >= 70% of f16 hit rate


def load_report(path):
    with open(path) as f:
        return json.load(f)


def print_table(configs):
    """Print comparison table."""
    header = f"{'Config':>10} | {'Hit Rate':>12} | {'Avg (ms)':>10} | {'P95 (ms)':>10} | {'Hits':>6}"
    sep = "-" * len(header)

    print(sep)
    print(header)
    print(sep)

    for name, report in sorted(configs.items()):
        s = report["summary"]
        print(
            f"{name:>10} | {s['hit_rate']*100:>10.1f}% | {s['latency_avg_ms']:>10.0f} | {s['latency_p95_ms']:>10.0f} | {s['hits']:>3}/{s['total']}"
        )

    print(sep)


def print_category_breakdown(configs):
    """Print per-category comparison."""
    # Collect all categories
    all_cats = set()
    for report in configs.values():
        all_cats.update(report["summary"]["by_category"].keys())

    print(f"\n{'Category':>15}", end="")
    for name in sorted(configs.keys()):
        print(f" | {name:>12}", end="")
    print()
    print("-" * (16 + 15 * len(configs)))

    for cat in sorted(all_cats):
        print(f"{cat:>15}", end="")
        for name in sorted(configs.keys()):
            stats = configs[name]["summary"]["by_category"].get(cat, {})
            hits = stats.get("hits", 0)
            total = stats.get("total", 0)
            rate = stats.get("hit_rate", 0) * 100
            print(f" | {hits:>2}/{total:<2} ({rate:>4.0f}%)", end="")
        print()


def verdict(configs):
    """Compute GO/NO-GO verdict."""
    if "f16" not in configs:
        print("\n⚠️  No f16 baseline found — cannot compute verdict.")
        return None

    f16_rate = configs["f16"]["summary"]["hit_rate"]

    if f16_rate == 0:
        print("\n⚠️  f16 baseline has 0% hit rate — benchmark ran without graph DB?")
        print("     Re-run with --db <graph_path> for meaningful results.")
        return None

    results = {}
    print(f"\n{'Config':>10} | {'Hit Rate':>10} | {'vs f16':>8} | {'Threshold':>10} | {'Verdict':>8}")
    print("-" * 65)

    for name in ["q8_0", "q4_0"]:
        if name not in configs:
            print(f"{name:>10} | {'N/A':>10} | {'N/A':>8} | {'N/A':>10} | SKIP")
            continue

        rate = configs[name]["summary"]["hit_rate"]
        ratio = rate / f16_rate if f16_rate > 0 else 0
        threshold = Q8_HIT_RATE_THRESHOLD if name == "q8_0" else Q4_HIT_RATE_THRESHOLD
        passed = ratio >= threshold

        results[name] = passed
        mark = "✓ GO" if passed else "✗ NO-GO"
        print(
            f"{name:>10} | {rate*100:>8.1f}% | {ratio*100:>6.1f}% | {threshold*100:>8.0f}% | {mark}"
        )

    print(f"{'f16':>10} | {f16_rate*100:>8.1f}% | {'(base)':>8} | {'(base)':>10} | (baseline)")

    # Overall verdict
    all_pass = all(results.values()) if results else False
    print()
    if all_pass:
        print("🟢 OVERALL VERDICT: GO — Proceed to P3 benchmark matrix")
    elif any(results.values()):
        passing = [k for k, v in results.items() if v]
        failing = [k for k, v in results.items() if not v]
        print(f"🟡 PARTIAL: {', '.join(passing)} OK, {', '.join(failing)} below threshold")
    else:
        print("🔴 OVERALL VERDICT: NO-GO — KV quantization degrades quality too much")

    return all_pass


def main():
    configs = {}
    for name in ["f16", "q8_0", "q4_0"]:
        path = os.path.join(RESULTS_DIR, f"benchmark_{name}.json")
        if os.path.exists(path):
            configs[name] = load_report(path)
            print(f"  Loaded: {path}")
        else:
            print(f"  Missing: {path}")

    if not configs:
        print("\nNo benchmark results found. Run tests/bench_kv_quant.sh first.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("  KV Quantization Benchmark — Comparison Report")
    print(f"{'='*60}\n")

    print_table(configs)
    print_category_breakdown(configs)
    result = verdict(configs)

    # Return exit code for CI
    if result is None:
        sys.exit(2)  # inconclusive
    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()
