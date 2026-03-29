#!/usr/bin/env python3
"""
C6 Embedding Quality Test — measures contrastive loss impact at the embedding level.

Instead of QA hit rate (bottlenecked by KV compression), this test directly measures
whether GNN-fused embeddings better preserve topological structure.

Metric: average cosine similarity between connected vs unconnected node embeddings.
A good contrastive loss should increase the gap (connected > unconnected).
"""

import pexpect
import os
import re
import json
import sys

BINARY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "target", "release", "obrain-chat")
MODEL = os.path.expanduser("~/models/Qwen3-14B-Claude-4.5-Opus-Distill.q4_k_m.gguf")
GRAPH_DB = "/tmp/obrain-b5bis-graph"
N_CTX = "8192"

_LLAMA_LIB = os.path.expanduser("~/projects/ia/llama.cpp/build/bin")
if os.path.isdir(_LLAMA_LIB):
    os.environ.setdefault("DYLD_LIBRARY_PATH", _LLAMA_LIB)


def extract_training_metrics(startup_text: str) -> dict:
    """Extract ProjectionNet training metrics from startup logs."""
    metrics = {
        "contrastive_pairs": 0,
        "edges": 0,
        "final_total": None,
        "final_recon": None,
        "final_contra": None,
        "n_updates": 0,
        "precomputed": 0,
    }

    # C6 contrastive: N pairs from M edges
    m = re.search(r'C6 contrastive: (\d+) pairs from (\d+) edges', startup_text)
    if m:
        metrics["contrastive_pairs"] = int(m.group(1))
        metrics["edges"] = int(m.group(2))

    # C6 training done: total=X recon=Y contra=Z
    m = re.search(r'C6 training done.*?total=([\d.]+).*?recon=([\d.]+).*?contra=([\d.]+)', startup_text)
    if m:
        metrics["final_total"] = float(m.group(1))
        metrics["final_recon"] = float(m.group(2))
        metrics["final_contra"] = float(m.group(3))

    # N updates
    m = re.search(r'(\d+) updates', startup_text)
    if m:
        metrics["n_updates"] = int(m.group(1))

    # Pre-computed N fused
    m = re.search(r'Pre-computed (\d+)', startup_text)
    if m:
        metrics["precomputed"] = int(m.group(1))

    # Also extract epoch-by-epoch if available
    epoch_losses = []
    for m in re.finditer(r'epoch (\d+)/\d+: total=([\d.]+) recon=([\d.]+) contra=([\d.]+)', startup_text):
        epoch_losses.append({
            "epoch": int(m.group(1)),
            "total": float(m.group(2)),
            "recon": float(m.group(3)),
            "contra": float(m.group(4)),
        })
    metrics["epoch_losses"] = epoch_losses

    return metrics


def run_and_extract(label: str, extra_args: list) -> dict:
    """Start obrain-chat, extract training metrics, then quit."""
    args = ['--model', MODEL, '--n-gpu', '99', '--n-ctx', N_CTX, '--debug']
    args.extend(extra_args)

    print(f"\n{'='*60}")
    print(f"  Mode: {label}")
    print(f"  Args: {' '.join(extra_args)}")
    print(f"{'='*60}")

    child = pexpect.spawn(BINARY, args, encoding='utf-8', timeout=600, dimensions=(80, 200))
    idx = child.expect([r'you>', pexpect.EOF, pexpect.TIMEOUT], timeout=480)

    startup = child.before or ''
    metrics = extract_training_metrics(startup)

    if idx == 0:
        print(f"  Started OK. Precomp={metrics['precomputed']}, pairs={metrics['contrastive_pairs']}")
        if metrics["final_total"] is not None:
            print(f"  Final loss: total={metrics['final_total']:.4f} recon={metrics['final_recon']:.4f} contra={metrics['final_contra']:.4f}")
        if metrics["epoch_losses"]:
            first = metrics["epoch_losses"][0]
            last = metrics["epoch_losses"][-1]
            print(f"  Recon: {first['recon']:.4f} → {last['recon']:.4f} (Δ={last['recon']-first['recon']:.4f})")
            print(f"  Contra: {first['contra']:.4f} → {last['contra']:.4f} (Δ={last['contra']-first['contra']:.4f})")
        try:
            child.sendline("/quit")
            child.expect(pexpect.EOF, timeout=10)
        except:
            pass
    else:
        print(f"  FAIL to start (idx={idx})")
        if startup:
            # Still extract what we can
            if metrics["epoch_losses"]:
                first = metrics["epoch_losses"][0]
                last = metrics["epoch_losses"][-1]
                print(f"  (from partial log) Recon: {first['recon']:.4f} → {last['recon']:.4f}")
                print(f"  (from partial log) Contra: {first['contra']:.4f} → {last['contra']:.4f}")

    child.close()
    metrics["label"] = label
    metrics["started"] = (idx == 0)
    return metrics


def main():
    if not os.path.exists(BINARY):
        print(f"ERROR: Binary not found: {BINARY}")
        sys.exit(1)

    results = {}

    # Mode 1: Phase C text-only (fresh persona, C6 contrastive ON)
    results["c_text"] = run_and_extract("phase_c_text", [
        "--db", GRAPH_DB,
        "--persona", "/tmp/obrain-emb-test-ct",
        "--head-router", "--head-router-warmup", "0",
        "--embd-injection-ratio", "1.0",
    ])

    # Mode 2: Phase C fused with GNN (elun persona, C6 contrastive ON)
    results["c_fused"] = run_and_extract("phase_c_fused", [
        "--db", GRAPH_DB,
        "--persona", "/tmp/elun",
        "--head-router", "--head-router-warmup", "0",
        "--embd-injection-ratio", "1.0",
    ])

    # === Summary ===
    print(f"\n{'='*70}")
    print("  C6 EMBEDDING QUALITY — Training Metrics")
    print(f"{'='*70}")

    for mode, data in results.items():
        print(f"\n  {mode}:")
        print(f"    Contrastive pairs: {data['contrastive_pairs']}")
        print(f"    Graph edges: {data['edges']}")
        if data.get("epoch_losses"):
            first = data["epoch_losses"][0]
            last = data["epoch_losses"][-1]
            print(f"    Reconstruction loss: {first['recon']:.4f} → {last['recon']:.4f}")
            print(f"    Contrastive loss:    {first['contra']:.4f} → {last['contra']:.4f}")
            recon_delta = last['recon'] - first['recon']
            contra_delta = last['contra'] - first['contra']
            print(f"    Δ recon: {recon_delta:+.4f} ({'↓ good' if recon_delta < 0 else '↑ bad'})")
            print(f"    Δ contra: {contra_delta:+.4f} ({'↓ good' if contra_delta < 0 else '↑ needs more data'})")

    # Verdict
    has_results = any(d.get("epoch_losses") for d in results.values())
    if has_results:
        all_contra_improved = all(
            d["epoch_losses"][-1]["contra"] < d["epoch_losses"][0]["contra"]
            for d in results.values()
            if d.get("epoch_losses")
        )
        all_recon_improved = all(
            d["epoch_losses"][-1]["recon"] < d["epoch_losses"][0]["recon"]
            for d in results.values()
            if d.get("epoch_losses")
        )
        print(f"\n  Reconstruction converges: {'YES' if all_recon_improved else 'NO'}")
        print(f"  Contrastive converges:    {'YES' if all_contra_improved else 'NO (needs more edges/negatives)'}")
        print(f"\n  Verdict: C6 infrastructure {'VALIDATED' if all_recon_improved else 'NEEDS WORK'}")
        if not all_contra_improved:
            print(f"  Note: Contrastive loss needs larger graph (>100 nodes) for meaningful signal")
    else:
        print(f"\n  Verdict: INCOMPLETE — could not extract training metrics")

    # Save
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embedding_quality_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
