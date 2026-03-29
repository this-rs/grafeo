#!/usr/bin/env python3
"""A/B Benchmark: Topology Mask vs Baseline (no mask).

Runs the same set of queries in two modes:
  - BASELINE: OBRAIN_NO_MASK=1 (default causal attention, model sees all KV)
  - MASKED:   default (topology mask restricts attention to scored nodes)

Metrics:
  (1) Factual Recall — does the response contain the expected fact?
      Formula: count(expected_facts ∩ response) / count(expected_facts)
  (2) Hallucination — does the response invent facts NOT in the persona?
      Formula: count(invented_names | invented_facts) — manual review column
  (3) Latency — time from query send to full response (seconds)
  (4) Token throughput — approximate tokens/sec from debug logs

Usage:
  python3 tests/bench_attn_bias.py [--model PATH] [--persona PATH] [--runs N]
"""
import subprocess, os, pty, select, time, re, csv, sys, json

BIN = "/tmp/obrain-chat/target/release/obrain-chat"
MODEL = os.environ.get("OBRAIN_MODEL",
    "/Users/triviere/models/Qwen3-14B-Claude-4.5-Opus-Distill.q4_k_m.gguf")
PERSONA = os.environ.get("OBRAIN_PERSONA", "/tmp/bench_attn_persona")
LOG = "/tmp/bench_attn_{mode}.log"
CSV_OUT = "/tmp/bench_attn_results.csv"
REPORT = "/tmp/bench_attn_report.md"

# ── Test queries with expected facts ──────────────────────────────
# Each tuple: (query, expected_facts_in_response, setup_query)
# setup_query is sent first to populate the persona (only in first run)
SETUP = [
    "Je m'appelle Thomas et j'habite à Lyon",
    "Ma couleur préférée c'est le bleu",
    "Je suis développeur Rust et j'adore le café",
    "Mon chat s'appelle Pixel et il a 3 ans",
    "Je travaille chez Anthropic depuis 2024",
]

QUERIES = [
    ("Comment je m'appelle ?", ["thomas"]),
    ("Où est-ce que j'habite ?", ["lyon"]),
    ("Quelle est ma couleur préférée ?", ["bleu"]),
    ("Qu'est-ce que tu sais sur moi ?", ["thomas", "lyon", "bleu", "rust", "café", "pixel", "anthropic"]),
    ("Comment s'appelle mon chat ?", ["pixel"]),
    ("Quel est mon métier ?", ["développeur", "rust"]),
    ("Depuis quand je travaille chez Anthropic ?", ["2024"]),
]

def read_stdout(proc, timeout=0.5):
    output = b""
    while True:
        r, _, _ = select.select([proc.stdout], [], [], timeout)
        if r:
            chunk = os.read(proc.stdout.fileno(), 8192)
            if chunk:
                output += chunk
                timeout = 0.2
            else:
                break
        else:
            break
    return output.decode("utf-8", errors="replace")

def clean(text):
    text = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', text)
    return re.sub(r'\[2K', '', text).strip()

def send(master_fd, proc, msg, wait_secs=25):
    os.write(master_fd, (msg + "\n").encode())
    time.sleep(wait_secs)
    text = clean(read_stdout(proc))
    resp = ""
    if "assistant>" in text:
        resp = text.split("assistant>")[-1].split("you>")[0].strip()
    return resp, text

def run_session(mode, do_setup=False):
    """Run a full benchmark session in the given mode."""
    env = os.environ.copy()
    # Ensure dylibs are found (macOS dynamic linking)
    llama_lib = os.environ.get("DYLD_LIBRARY_PATH",
        "/Users/triviere/projects/ia/llama.cpp/build/bin")
    env["DYLD_LIBRARY_PATH"] = llama_lib
    if mode == "baseline":
        env["OBRAIN_NO_MASK"] = "1"
    else:
        env.pop("OBRAIN_NO_MASK", None)

    log_path = LOG.format(mode=mode)
    master_fd, slave_fd = pty.openpty()
    proc = subprocess.Popen(
        [BIN, "--model", MODEL, "--n-gpu", "99", "--persona", PERSONA, "--debug"],
        stdin=slave_fd, stdout=subprocess.PIPE, stderr=open(log_path, "w"),
        bufsize=0, env=env)
    os.close(slave_fd)

    print(f"\n{'='*60}")
    print(f"  Mode: {mode.upper()}")
    print(f"{'='*60}")
    print("  Loading model...")
    time.sleep(14)
    read_stdout(proc, 2)

    # Setup: populate persona (only first run)
    if do_setup:
        for setup_msg in SETUP:
            print(f"  [setup] {setup_msg[:50]}...")
            send(master_fd, proc, setup_msg, 20)

    # Benchmark queries
    results = []
    for query, expected_facts in QUERIES:
        print(f"\n  Q: {query}")
        t0 = time.time()
        resp, raw = send(master_fd, proc, query, 25)
        elapsed = time.time() - t0

        # Factual recall
        resp_lower = resp.lower()
        hits = sum(1 for f in expected_facts if f.lower() in resp_lower)
        recall = hits / len(expected_facts) if expected_facts else 0

        print(f"  A: {resp[:200]}")
        print(f"  Recall: {hits}/{len(expected_facts)} = {recall:.0%} | Time: {elapsed:.1f}s")

        results.append({
            "mode": mode,
            "query": query,
            "response": resp[:500],
            "expected_facts": ",".join(expected_facts),
            "hits": hits,
            "total_facts": len(expected_facts),
            "recall": recall,
            "latency_s": round(elapsed, 2),
        })

    # Quit
    os.write(master_fd, b"/quit\n")
    time.sleep(3)
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except:
        proc.kill()
    os.close(master_fd)

    # Extract token throughput from logs
    with open(log_path) as f:
        log = f.read()
    tps_matches = re.findall(r'(\d+\.?\d*)\s*t(?:ok(?:en)?)?s?/s', log)
    if tps_matches:
        avg_tps = sum(float(t) for t in tps_matches) / len(tps_matches)
        for r in results:
            r["tokens_per_sec"] = round(avg_tps, 1)
    else:
        for r in results:
            r["tokens_per_sec"] = None

    return results

def generate_report(all_results):
    """Generate markdown report comparing baseline vs masked."""
    baseline = [r for r in all_results if r["mode"] == "baseline"]
    masked = [r for r in all_results if r["mode"] == "masked"]

    def avg(lst, key):
        vals = [r[key] for r in lst if r[key] is not None]
        return sum(vals) / len(vals) if vals else 0

    b_recall = avg(baseline, "recall")
    m_recall = avg(masked, "recall")
    b_latency = avg(baseline, "latency_s")
    m_latency = avg(masked, "latency_s")
    b_tps = avg(baseline, "tokens_per_sec")
    m_tps = avg(masked, "tokens_per_sec")

    recall_delta = (m_recall - b_recall) * 100
    latency_delta = ((m_latency - b_latency) / b_latency * 100) if b_latency > 0 else 0

    report = f"""# A/B Benchmark: Topology Mask vs Baseline

**Date**: {time.strftime("%Y-%m-%d %H:%M")}
**Model**: {os.path.basename(MODEL)}
**Queries**: {len(QUERIES)}

## Summary

| Metric | Baseline (no mask) | Masked (topology) | Delta |
|--------|-------------------|-------------------|-------|
| Factual Recall | {b_recall:.0%} | {m_recall:.0%} | {recall_delta:+.1f}pp |
| Avg Latency | {b_latency:.1f}s | {m_latency:.1f}s | {latency_delta:+.1f}% |
| Tokens/sec | {b_tps:.1f} | {m_tps:.1f} | — |

## Decision Gate

- Factual recall improvement ≥ 10%: **{"✅ PASS" if recall_delta >= 10 else "❌ FAIL"}** ({recall_delta:+.1f}pp)
- Latency regression < 10%: **{"✅ PASS" if latency_delta < 10 else "❌ FAIL"}** ({latency_delta:+.1f}%)
- **Overall: {"✅ VALIDATED — proceed to Phase B" if recall_delta >= 10 and latency_delta < 10 else "⚠️ NEEDS INVESTIGATION"}**

## Per-Query Results

| Query | Baseline Recall | Masked Recall | Δ |
|-------|----------------|---------------|---|
"""
    for q, _ in QUERIES:
        b = next((r for r in baseline if r["query"] == q), None)
        m = next((r for r in masked if r["query"] == q), None)
        br = f"{b['recall']:.0%}" if b else "—"
        mr = f"{m['recall']:.0%}" if m else "—"
        d = ""
        if b and m:
            d = f"{(m['recall'] - b['recall'])*100:+.0f}pp"
        report += f"| {q} | {br} | {mr} | {d} |\n"

    report += f"\n## Raw Data\n\nSee CSV: `{CSV_OUT}`\n"
    return report, {
        "recall_delta_pp": round(recall_delta, 1),
        "latency_delta_pct": round(latency_delta, 1),
        "validated": recall_delta >= 10 and latency_delta < 10,
    }

if __name__ == "__main__":
    # Parse args
    runs = 1
    for i, arg in enumerate(sys.argv):
        if arg == "--model" and i + 1 < len(sys.argv):
            MODEL = sys.argv[i + 1]
        elif arg == "--persona" and i + 1 < len(sys.argv):
            PERSONA = sys.argv[i + 1]
        elif arg == "--runs" and i + 1 < len(sys.argv):
            runs = int(sys.argv[i + 1])

    print(f"A/B Benchmark: Topology Mask vs Baseline")
    print(f"Model: {MODEL}")
    print(f"Persona: {PERSONA}")
    print(f"Runs: {runs}")

    all_results = []

    for run_idx in range(runs):
        # First run sets up persona, subsequent runs reuse it
        do_setup = (run_idx == 0)

        # Run masked FIRST (to populate persona with setup facts)
        masked_results = run_session("masked", do_setup=do_setup)
        all_results.extend(masked_results)

        # Then baseline (persona already has facts)
        baseline_results = run_session("baseline", do_setup=False)
        all_results.extend(baseline_results)

    # Write CSV
    with open(CSV_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "mode", "query", "response", "expected_facts",
            "hits", "total_facts", "recall", "latency_s", "tokens_per_sec"
        ])
        writer.writeheader()
        writer.writerows(all_results)

    # Generate report
    report, metrics = generate_report(all_results)
    with open(REPORT, "w") as f:
        f.write(report)

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(report)
    print(f"\nCSV: {CSV_OUT}")
    print(f"Report: {REPORT}")
    print(f"Metrics: {json.dumps(metrics, indent=2)}")
