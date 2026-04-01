#!/bin/bash
# TP.3 — Benchmark E2E : formules + self + proprioception
# Usage: ./benchmark_tp3.sh [model_path]
#
# Creates a virgin persona, runs benchmark questions, captures output.
# Requires: obrain-chat binary, model GGUF file.

set -euo pipefail

MODEL="${1:-$HOME/models/UD-IQ4_XS/Qwen3.5-122B-A10B-UD-IQ4_XS-00001-of-00003.gguf}"
BINARY="/tmp/obrain-chat/dist/obrain-chat-macos-arm64"
PERSONA_DIR="/tmp/tp3_benchmark_persona"
LOG_DIR="/tmp/tp3_benchmark_logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$LOG_DIR"

echo "═══════════════════════════════════════════════════════════════"
echo " TP.3 — Benchmark E2E"
echo " Model: $MODEL"
echo " Persona: $PERSONA_DIR (virgin)"
echo " Logs: $LOG_DIR"
echo "═══════════════════════════════════════════════════════════════"

# ─── Phase 1: Benchmark questions (12 questions Phase D style) ───────────

run_session() {
    local session_name="$1"
    local input_file="$2"
    local log_file="$LOG_DIR/${session_name}_${TIMESTAMP}.log"

    echo ""
    echo "▸ Running session: $session_name"
    echo "  Input: $input_file"
    echo "  Log: $log_file"

    # Clean persona for each session (cold start)
    rm -rf "$PERSONA_DIR"
    mkdir -p "$PERSONA_DIR"

    # Use 'script' to fake a TTY for rustyline
    # The input is fed via a FIFO with delays between questions
    local fifo="/tmp/tp3_fifo_$$"
    mkfifo "$fifo"

    # Feed questions with delays (model needs time to respond)
    (
        sleep 10  # wait for model load
        while IFS= read -r line; do
            echo "$line"
            sleep 45  # wait for generation (122B is slow)
        done < "$input_file"
        echo "/quit"
        sleep 5
    ) > "$fifo" &
    local feeder_pid=$!

    # Run obrain-chat with faked TTY
    script -q "$log_file" "$BINARY" \
        --model "$MODEL" \
        --n-ctx 8192 \
        --kv-capacity 8192 \
        --persona "$PERSONA_DIR" \
        < "$fifo" 2>&1 || true

    kill "$feeder_pid" 2>/dev/null || true
    rm -f "$fifo"

    echo "  ✓ Session complete: $(wc -l < "$log_file") lines"
}

# ─── Create input files ──────────────────────────────────────────────────

# Session 1: Knowledge & reasoning (6 questions)
cat > /tmp/tp3_input_knowledge.txt << 'QUESTIONS'
Explique-moi la fonction zeta de Riemann et pourquoi les zeros non-triviaux sont importants
Quelle est la relation entre la transformee de Fourier et le principe d'incertitude de Heisenberg ?
En quoi le theoreme de Godel impacte-t-il les fondements des mathematiques ?
Decris l'algorithme de Dijkstra et sa complexite
Comment fonctionne l'attention multi-tete dans un transformer ?
Quelle est la difference entre un B-tree et un B+tree pour les bases de donnees ?
QUESTIONS

# Session 2: Self-awareness (3 questions)
cat > /tmp/tp3_input_self.txt << 'QUESTIONS'
Parle-moi de toi, comment tu fonctionnes ?
Quel est ton niveau de confiance actuel et quelle formule d'attention utilises-tu ?
Decris ton etat interne : reward moyen, nombre de memoires, formules actives
QUESTIONS

# Session 3: Progressive feedback (cold start convergence)
cat > /tmp/tp3_input_convergence.txt << 'QUESTIONS'
Bonjour, je m'appelle Thomas
Je suis developpeur Rust et je travaille sur des bases de donnees graphe
Mon projet principal s'appelle Obrain, c'est une base de donnees cognitive
Merci, c'est tres clair
Peux-tu me rappeler ce que je t'ai dit sur mon projet ?
Quel est ton reward moyen et combien de memoires as-tu retenues ?
Quelles formules d'attention as-tu decouvertes ?
QUESTIONS

# Session 4: Reformulation stress test (should trigger negative reward)
cat > /tmp/tp3_input_reformulation.txt << 'QUESTIONS'
Explique-moi ce qu'est un KV cache dans un LLM
Redis-moi ce qu'est un KV cache dans un LLM
Explique le KV cache des LLM
Qu'est-ce que le KV cache ?
Merci, maintenant parle-moi de la quantization des modeles
QUESTIONS

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " Starting benchmark sessions..."
echo " NOTE: Each session loads the model fresh. Be patient."
echo "═══════════════════════════════════════════════════════════════"

# ─── Run sessions ────────────────────────────────────────────────────────

echo ""
echo "=== SESSION 1/4: Knowledge & Reasoning ==="
run_session "01_knowledge" "/tmp/tp3_input_knowledge.txt"

echo ""
echo "=== SESSION 2/4: Self-Awareness ==="
run_session "02_self" "/tmp/tp3_input_self.txt"

echo ""
echo "=== SESSION 3/4: Convergence (Progressive Feedback) ==="
run_session "03_convergence" "/tmp/tp3_input_convergence.txt"

echo ""
echo "=== SESSION 4/4: Reformulation Stress Test ==="
run_session "04_reformulation" "/tmp/tp3_input_reformulation.txt"

# ─── Extract metrics ─────────────────────────────────────────────────────

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " Extracting metrics..."
echo "═══════════════════════════════════════════════════════════════"

REPORT="$LOG_DIR/tp3_report_${TIMESTAMP}.txt"

cat > "$REPORT" << HEADER
# TP.3 — Benchmark E2E Report
# Date: $(date)
# Model: $MODEL
# Binary: $BINARY

## Raw Metrics
HEADER

for log in "$LOG_DIR"/*_${TIMESTAMP}.log; do
    session=$(basename "$log" | sed "s/_${TIMESTAMP}.log//")
    echo "" >> "$REPORT"
    echo "### $session" >> "$REPORT"

    # Extract reward signals
    echo "Rewards:" >> "$REPORT"
    grep -o "\[Reward\].*" "$log" 2>/dev/null >> "$REPORT" || echo "  (no rewards)" >> "$REPORT"

    # Extract formula events
    echo "Formulas:" >> "$REPORT"
    grep -o "\[AFE\].*" "$log" 2>/dev/null >> "$REPORT" || echo "  (no AFE events)" >> "$REPORT"

    # Extract T4 token polarity
    echo "Token Polarity:" >> "$REPORT"
    grep -o "\[T4\].*" "$log" 2>/dev/null >> "$REPORT" || echo "  (no T4 events)" >> "$REPORT"

    # Extract PersistNet
    echo "PersistNet:" >> "$REPORT"
    grep -o "\[PersistNet\].*" "$log" 2>/dev/null >> "$REPORT" || echo "  (no PersistNet events)" >> "$REPORT"

    # Extract GNN
    echo "GNN:" >> "$REPORT"
    grep -o "\[GNN\].*" "$log" 2>/dev/null >> "$REPORT" || echo "  (no GNN events)" >> "$REPORT"
done

echo "" >> "$REPORT"
echo "## Summary" >> "$REPORT"
echo "Total reward lines: $(grep -c '\[Reward\]' "$LOG_DIR"/*_${TIMESTAMP}.log 2>/dev/null || echo 0)" >> "$REPORT"
echo "Total AFE events: $(grep -c '\[AFE\]' "$LOG_DIR"/*_${TIMESTAMP}.log 2>/dev/null || echo 0)" >> "$REPORT"
echo "Total T4 events: $(grep -c '\[T4\]' "$LOG_DIR"/*_${TIMESTAMP}.log 2>/dev/null || echo 0)" >> "$REPORT"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " ✅ Benchmark complete!"
echo " Report: $REPORT"
echo " Logs: $LOG_DIR/*_${TIMESTAMP}.log"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Next: read the report and evaluate Go/No-Go criteria"
