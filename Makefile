# lab/grafeo — top-level Makefile
#
# Benchmark + substrate targets. For general development, prefer per-crate
# `cargo` commands. This Makefile orchestrates multi-step baselines and the
# Substrate RFC validation gates (see docs/rfc/substrate/format-spec.md §10).

# ---------------------------------------------------------------------------
# Config — override on the command line, e.g.:
#     make bench-baseline PO_DB=/path/to/po.obrain MEGALAW_DB=/path/to/megalaw.obrain
# ---------------------------------------------------------------------------
PO_DB      ?= $(HOME)/.obrain/hub.db
MEGALAW_DB ?= $(HOME)/.obrain/megalaw.obrain
BENCH_DIR  := bench/substrate
BENCH_OUT  := target/bench
CARGO      ?= cargo

.PHONY: help
help:
	@echo "Grafeo top-level targets:"
	@echo "  make bench-baseline          — Run LpgStore baseline on PO + megalaw"
	@echo "  make bench-baseline-po       — Baseline on PO only"
	@echo "  make bench-baseline-megalaw  — Baseline on megalaw only"
	@echo "  make bench-substrate         — Run SubstrateStore on PO + megalaw (after T3)"
	@echo "  make bench-compare           — Compare baseline vs substrate, emit gates report"
	@echo "  make bench-smoke             — Quick smoke test (no real databases needed)"
	@echo "  make bench-clean             — Remove target/bench/"
	@echo "  make check-substrate         — cargo check the bench harness"
	@echo ""
	@echo "Variables:"
	@echo "  PO_DB       = $(PO_DB)"
	@echo "  MEGALAW_DB  = $(MEGALAW_DB)"
	@echo "  BENCH_OUT   = $(BENCH_OUT)"

# ---------------------------------------------------------------------------
# Bench — baseline (LpgStore)
# ---------------------------------------------------------------------------

$(BENCH_OUT):
	mkdir -p $(BENCH_OUT)

.PHONY: bench-baseline
bench-baseline: bench-baseline-po bench-baseline-megalaw
	@echo ""
	@echo "Baseline complete. Outputs:"
	@echo "  $(BENCH_OUT)/baseline-po.json"
	@echo "  $(BENCH_OUT)/baseline-megalaw.json"
	@echo "Run 'make bench-compare' after Substrate is available (T9)."

.PHONY: bench-baseline-po
bench-baseline-po: $(BENCH_OUT)
	@test -e "$(PO_DB)" || (echo "ERROR: PO_DB not found: $(PO_DB)"; echo "Set with: make bench-baseline-po PO_DB=<path>"; exit 1)
	cd $(BENCH_DIR) && $(CARGO) run --release --bin bench-baseline -- \
		--target po --db "$(PO_DB)" --out ../../$(BENCH_OUT)/baseline-po.json

.PHONY: bench-baseline-megalaw
bench-baseline-megalaw: $(BENCH_OUT)
	@test -e "$(MEGALAW_DB)" || (echo "ERROR: MEGALAW_DB not found: $(MEGALAW_DB)"; echo "Set with: make bench-baseline-megalaw MEGALAW_DB=<path>"; exit 1)
	cd $(BENCH_DIR) && $(CARGO) run --release --bin bench-baseline -- \
		--target megalaw --db "$(MEGALAW_DB)" --out ../../$(BENCH_OUT)/baseline-megalaw.json

# ---------------------------------------------------------------------------
# Bench — substrate (wired in T3; runs as stub until then)
# ---------------------------------------------------------------------------

.PHONY: bench-substrate
bench-substrate: bench-substrate-po bench-substrate-megalaw

.PHONY: bench-substrate-po
bench-substrate-po: $(BENCH_OUT)
	cd $(BENCH_DIR) && $(CARGO) run --release --bin bench-substrate -- \
		--target po --db "$(PO_DB).substrate" --out ../../$(BENCH_OUT)/substrate-po.json

.PHONY: bench-substrate-megalaw
bench-substrate-megalaw: $(BENCH_OUT)
	cd $(BENCH_DIR) && $(CARGO) run --release --bin bench-substrate -- \
		--target megalaw --db "$(MEGALAW_DB).substrate" --out ../../$(BENCH_OUT)/substrate-megalaw.json

# ---------------------------------------------------------------------------
# Bench — compare (T9 — gates)
# ---------------------------------------------------------------------------
#
# Validation gates (from docs/rfc/substrate/format-spec.md §10 and
# bench/substrate/README.md):
#
#   | metric                          | target                              |
#   |---------------------------------|-------------------------------------|
#   | rss_anon_peak megalaw           | ≤ 1 GiB (vs 5–10 GiB baseline)      |
#   | rss_total_peak megalaw          | ≤ 18 GiB (vs 110 GiB baseline)      |
#   | open_duration megalaw           | ≤ 100 ms (vs 5–15 s baseline)       |
#   | retrieval_latency@p95           | ≤ 1 ms (vs 3–10 ms baseline)        |
#   | activation_latency@p95          | ≤ 1 ms (vs 5–20 ms baseline)        |
#   | recall@10 vs baseline (megalaw) | ≥ 99 %                               |
#
# The compare step is implemented in T9 (bench/substrate/src/bin/compare.rs).

.PHONY: bench-compare
bench-compare:
	@echo "bench-compare: T9 — run after both baseline and substrate JSONs exist."
	@ls -la $(BENCH_OUT)/baseline-*.json $(BENCH_OUT)/substrate-*.json 2>/dev/null || true
	@echo "(compare binary TBD — emits gates report and non-zero exit on gate violation)"

# ---------------------------------------------------------------------------
# Smoke — no real databases; exercises the bench-substrate stub path
# ---------------------------------------------------------------------------

.PHONY: bench-smoke
bench-smoke: $(BENCH_OUT)
	cd $(BENCH_DIR) && $(CARGO) run --release --bin bench-substrate -- \
		--target custom --db /tmp/nonexistent --out ../../$(BENCH_OUT)/smoke.json --smoke
	@echo ""
	@echo "Smoke JSON:"
	@cat $(BENCH_OUT)/smoke.json | head -30

# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------

.PHONY: check-substrate
check-substrate:
	cd $(BENCH_DIR) && $(CARGO) check --bins --benches

.PHONY: bench-clean
bench-clean:
	rm -rf $(BENCH_OUT)
