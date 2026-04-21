# bench-substrate

Baseline + Substrate comparison benchmarks for the Substrate storage RFC.

This is a **standalone Cargo workspace** (not a member of `lab/grafeo`) to keep bench-only
dependencies (criterion, sysinfo, clap) out of release builds.

## Layout

- `src/lib.rs`                  — shared harness: `BenchResult`, `RssSampler`, `HostInfo`, `Measurement`
- `src/bench_baseline.rs`       — LpgStore baseline runner (wired in T1-step3)
- `src/bench_substrate.rs`      — SubstrateStore runner (wired in T3 + T8 + T10)
- `benches/retrieval.rs`        — criterion micro-bench for retrieval (T8)
- `benches/activation.rs`       — criterion micro-bench for activation spreading (T11/T12)
- `benches/crud.rs`             — criterion micro-bench for CRUD (T3/T6)

## Usage

```bash
# Baseline
cargo run --release --bin bench-baseline -- \
  --target po --db /path/to/po.obrain --out ../../target/bench/baseline-po.json
cargo run --release --bin bench-baseline -- \
  --target megalaw --db /path/to/megalaw.obrain --out ../../target/bench/baseline-megalaw.json

# Substrate (after T3)
cargo run --release --bin bench-substrate -- \
  --target po --db /tmp/po-substrate --out ../../target/bench/substrate-po.json

# Criterion micro-benches
cargo bench --bench retrieval
cargo bench --bench activation
cargo bench --bench crud
```

## JSON output schema

Each runner writes a `BenchResult` — see [`src/lib.rs`](src/lib.rs) for the struct definition.

- `schema_version: u32` (currently 1) — bump on breaking schema change
- `backend: Backend` — `"Lpg"` or `"Substrate"`
- `target: Target` — `"Po"`, `"Megalaw"`, `"Custom"`
- `host: HostInfo` — OS, arch, CPU model, core count, total RAM, rustc version
- `measurements: Vec<Measurement>` — each with `name`, `unit`, `value`, optional `stat`
  (`mean`/`p50`/`p95`/`p99`/`max`)

## Gates (T9)

These numbers are the acceptance targets; T9 compares baseline and substrate JSONs.

| Metric | Baseline (current) | Target (substrate) |
|--------|-------------------|-------------------|
| `rss_anon_peak` (megalaw)      | 5–10 GiB | ≤ 1 GiB |
| `rss_total_peak` (megalaw)     | 110 GiB  | ≤ 18 GiB |
| `open_duration` (megalaw)      | 5–15 s   | ≤ 100 ms |
| `retrieval_latency@p95`        | 3–10 ms  | ≤ 1 ms |
| `activation_latency@p95`       | 5–20 ms  | ≤ 1 ms |
| recall@10 vs baseline (megalaw)| —        | ≥ 99 % |
