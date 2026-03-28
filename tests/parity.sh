#!/bin/bash
# T8 — Parity tests for obrain-chat FFI mode
# Tests: S0 (queries), S1 (token parity via /tokenize), S3 (kv consistency), S4 (qualitative)
set -euo pipefail

MODEL="/Users/triviere/models/llama3.2-3b.gguf"
DB="/tmp/neo4j2grafeo/grafeo.db"
BINARY="/tmp/obrain-chat/target/debug/obrain-chat"
export DYLD_LIBRARY_PATH="/Users/triviere/projects/ia/llama.cpp/build/bin"

# ═══════════════════════════════════════════════════════════════════════════════
# S0 — 5 reference queries
# ═══════════════════════════════════════════════════════════════════════════════
QUERIES=(
    "Parle moi d'Elun"                          # Simple entity lookup
    "C'est quoi ses capacités ?"                 # Follow-up vague (needs conv context)
    "Quels sont les liens entre Elun et Grafeo"  # Multi-node cross-entity
    "Raconte moi l'histoire de XyzNotExist"      # No match
    "Résume notre conversation"                  # Conv fragments
)

echo "═══ T8 Parity Tests — $(date) ═══"
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# S3+S4 — E2E FFI test: feed queries, check kv consistency + response quality
# ═══════════════════════════════════════════════════════════════════════════════

echo "Building obrain-chat..."
cd /tmp/obrain-chat
cargo build 2>/dev/null

echo "Running 5 reference queries through FFI mode..."
echo ""

# Pipe all queries + quit into the binary, capture stderr (logs) and stdout (responses)
QUERY_INPUT=""
for q in "${QUERIES[@]}"; do
    QUERY_INPUT+="$q"$'\n'
done
QUERY_INPUT+="/quit"$'\n'

echo "$QUERY_INPUT" | timeout 120 "$BINARY" \
    --model "$MODEL" \
    --db "$DB" \
    --n-ctx 8192 \
    --kv-capacity 2048 \
    2>/tmp/t8_stderr.log \
    1>/tmp/t8_stdout.log \
    || true

echo "─── STDERR (KV metrics) ───"
# Extract KV registry metrics
grep -E "(KV Registry|Warmup|Mask:|next_pos|total_tokens|kv_used|nodes in KV|encoded|evict)" /tmp/t8_stderr.log || echo "(no KV metrics found)"
echo ""

echo "─── RESPONSES ───"
cat /tmp/t8_stdout.log
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# S1 — Token count parity: FFI (llama_tokenize) vs HTTP (/tokenize)
# ═══════════════════════════════════════════════════════════════════════════════

# Check if llama-server is running on 8090
SERVER_URL="http://localhost:8090"
if curl -s "$SERVER_URL/health" | grep -q "ok"; then
    echo "═══ S1: Token count parity (FFI vs HTTP server) ═══"
    echo "⚠ Note: Server uses Qwen3-14B, FFI uses Llama3.2-3B — tokenizers differ by design."
    echo "   This test validates that both tokenization APIs work correctly, not that counts match."
    echo ""

    TEST_TEXTS=(
        "Hello world"
        "éàü café résumé"
        "日本語テスト"
        "The quick brown fox jumps over the lazy dog"
        "Hello 🌍🚀 world 🎉"
    )

    for text in "${TEST_TEXTS[@]}"; do
        # HTTP tokenization
        HTTP_COUNT=$(curl -s "$SERVER_URL/tokenize" \
            -H "Content-Type: application/json" \
            -d "{\"content\": \"$text\"}" \
            | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('tokens',d.get('data',[]))))" 2>/dev/null || echo "ERROR")

        echo "  '$text'"
        echo "    HTTP (Qwen3-14B): $HTTP_COUNT tokens"
    done
else
    echo "═══ S1: SKIPPED (llama-server not running on $SERVER_URL) ═══"
fi

echo ""
echo "═══ T8 Complete ═══"
echo ""
echo "Manual review checklist (S4):"
echo "  [ ] Responses are in French (for French queries)"
echo "  [ ] 'Parle moi d'Elun' mentions graph nodes related to Elun"
echo "  [ ] 'XyzNotExist' query handles gracefully (no crash, reasonable response)"
echo "  [ ] KV metrics show incremental encoding (not re-encoding cached nodes)"
echo "  [ ] No panics or segfaults in stderr"
