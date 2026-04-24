#!/usr/bin/env bash
# Full regression harness — lancé avant/après chaque step de T17j+ pour
# détecter toute dérive, pas seulement le périmètre du changement en
# cours.
#
# Couvre 4 axes :
#   1. Lib tests (substrate + engine) — no behavioural regression
#   2. DB open bench (PO/Wiki/Megalaw) — no startup regression
#   3. RAM bench (3 corpora) — no memory regression
#   4. Full PO query bench (26 queries) — no perf regression
#
# Usage :
#   scripts/bench-regression.sh               # run all
#   scripts/bench-regression.sh open-only     # fast pre-commit check
#   scripts/bench-regression.sh --no-write    # skip writes on existing DBs
#
# Output : sauvegarde chaque log dans /tmp/bench-regression-$(date +%s)/
# + affiche un récap final avec verdicts PASS/FAIL/DIFF.
#
# Gates par défaut (configurable via env) :
#   T17J_GATE_PO_OPEN_MS=20
#   T17J_GATE_WIKI_OPEN_MS=30
#   T17J_GATE_MEGALAW_OPEN_MS=45
#   T17J_GATE_MOST_CONNECTED_MS=30
#   T17J_GATE_COUNT_N_MS=1
#   T17J_GATE_RAM_PO_MB=150
#   T17J_GATE_RAM_MEGALAW_MB=1200
#   T17J_GATE_RAM_WIKI_MB=5000

set -euo pipefail

MODE="${1:-full}"
TS="$(date +%s)"
OUTDIR="/tmp/bench-regression-$TS"
mkdir -p "$OUTDIR"
echo "═══ T17j regression harness — logs : $OUTDIR ═══"

PASS=()
FAIL=()

run_or_fail() {
    local label="$1"
    shift
    local logfile="$OUTDIR/${label// /_}.log"
    echo "→ $label"
    if "$@" > "$logfile" 2>&1; then
        PASS+=("$label")
        tail -1 "$logfile" | sed 's/^/    /'
    else
        FAIL+=("$label")
        echo "  ✗ FAIL — see $logfile"
    fi
}

# ── 1. Lib tests ────────────────────────────────────────────
echo
echo "── Phase 1 : lib tests"
run_or_fail "obrain-substrate lib (523 tests)" \
    cargo test -p obrain-substrate --lib --release
run_or_fail "obrain-engine lib (663 tests)" \
    cargo test -p obrain-engine --lib --release

# ── 2. DB open bench ────────────────────────────────────────
echo
echo "── Phase 2 : DB open (3 corpora)"
run_or_fail "t17i_db_open_bench" \
    cargo test -p obrain-engine --release --features cypher \
    --test t17i_db_open_bench -- --nocapture

if [[ "$MODE" == "open-only" ]]; then
    echo
    echo "── open-only mode — stopping"
    echo "PASS: ${#PASS[@]} / FAIL: ${#FAIL[@]}"
    exit $(( ${#FAIL[@]} > 0 ))
fi

# ── 3. RAM bench ────────────────────────────────────────────
echo
echo "── Phase 3 : RAM bench"
run_or_fail "t17i_ram_bench" \
    cargo test -p obrain-engine --release --features cypher \
    --test t17i_ram_bench -- --nocapture

# ── 4. Full PO bench ────────────────────────────────────────
echo
echo "── Phase 4 : 26-query PO bench vs Neo4j"
run_or_fail "po_vs_neo4j_bench" \
    cargo test -p obrain-engine --release --features cypher \
    --test po_vs_neo4j_bench -- bench_po_vs_neo4j --nocapture --exact

# ── 5. T9b planner suite ────────────────────────────────────
echo
echo "── Phase 5 : T9b planner (snapshot + variance + bench gate)"
run_or_fail "t9_planner_rewrite_snapshot" \
    cargo test -p obrain-engine --release --features cypher \
    --test t9_planner_rewrite_snapshot -- --nocapture
run_or_fail "t9_planner_rewrite_variance" \
    cargo test -p obrain-engine --release --features cypher \
    --test t9_planner_rewrite_variance -- --nocapture
run_or_fail "t9_planner_rewrite_correctness" \
    cargo test -p obrain-engine --release --features cypher \
    --test t9_planner_rewrite_correctness bench_gate_most_connected_under_30ms \
    -- --nocapture

# ── Recap ──────────────────────────────────────────────────
echo
echo "═══ RECAP ═══"
echo "PASS : ${#PASS[@]} / ${#PASS[@]}+${#FAIL[@]}"
for p in "${PASS[@]}"; do echo "  ✓ $p"; done
if [[ ${#FAIL[@]} -gt 0 ]]; then
    echo "FAIL :"
    for f in "${FAIL[@]}"; do echo "  ✗ $f"; done
    echo
    echo "Logs dans $OUTDIR/"
    exit 1
fi
echo
echo "Logs dans $OUTDIR/ pour diff avec baseline."
