#!/bin/bash
# bench_kv_quant.sh — Run benchmark with 3 KV cache configurations
# Usage: ./tests/bench_kv_quant.sh [model_path] [db_path]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
BINARY="$ROOT_DIR/dist/obrain-chat-macos-arm64"
MODEL="${1:-$HOME/models/llama3.2-3b.gguf}"
DB="${2:-/tmp/obrain-d-bench-graph}"
N_CTX="${N_CTX:-8192}"
MAX_NODES="${MAX_NODES:-10}"
RESULTS_DIR="$ROOT_DIR/tests/results/kv_quant"

mkdir -p "$RESULTS_DIR"

if [ ! -f "$BINARY" ]; then
    echo "ERROR: Binary not found at $BINARY"
    echo "Run 'make macos' first."
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    echo "ERROR: Model not found at $MODEL"
    exit 1
fi

if [ ! -d "$DB" ]; then
    echo "ERROR: Database not found at $DB"
    exit 1
fi

echo "╔══════════════════════════════════════════════╗"
echo "║  KV Quantization Benchmark (3 configs)      ║"
echo "╠══════════════════════════════════════════════╣"
echo "║  Model:  $(basename "$MODEL")"
echo "║  DB:     $DB"
echo "║  n_ctx:  $N_CTX"
echo "║  nodes:  $MAX_NODES"
echo "║  Output: $RESULTS_DIR/"
echo "╚══════════════════════════════════════════════╝"
echo ""

CONFIGS=("f16:1" "q8_0:q8_0" "q4_0:q4_0")

for cfg in "${CONFIGS[@]}"; do
    NAME="${cfg%%:*}"
    KV_TYPE="${cfg##*:}"

    OUTPUT="$RESULTS_DIR/benchmark_${NAME}.json"
    echo "━━━ Config: $NAME ━━━"

    if [ "$NAME" = "f16" ]; then
        CTK_FLAG=""
        CTV_FLAG=""
    else
        CTK_FLAG="--ctk $KV_TYPE"
        CTV_FLAG="--ctv $KV_TYPE"
    fi

    "$BINARY" \
        --model "$MODEL" \
        --db "$DB" \
        --n-ctx "$N_CTX" \
        --max-nodes "$MAX_NODES" \
        $CTK_FLAG $CTV_FLAG \
        --benchmark \
        --benchmark-output "$OUTPUT" \
        2>&1 | grep -E '^\s*(===|\[|Hit rate|Latency|KV type|By category|  )' || true

    echo ""
    if [ -f "$OUTPUT" ]; then
        echo "  ✓ Results: $OUTPUT"
    else
        echo "  ✗ FAILED: no output file"
    fi
    echo ""
done

echo "━━━ All configs complete ━━━"
echo "Results in: $RESULTS_DIR/"
ls -la "$RESULTS_DIR/"*.json 2>/dev/null || echo "  (no JSON files)"
echo ""
echo "Run: python3 tests/analyze_kv_quant.py to compare results"
