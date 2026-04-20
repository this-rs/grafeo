# T5 Step 0 — Migration Audit

**Generated**: 2026-04-20
**Scope**: Substrate — plan SUBSTRATE, task T5
**Goal**: exhaustive inventory of every call site that touches
`ObrainDB::open`, `ObrainDB::open_overlay`, or `PersistentStore`, tagged with
a target for T5 (`migrate` / `keep` / `delete`).

## Tag legend

| Tag       | Meaning                                                                             |
|-----------|-------------------------------------------------------------------------------------|
| `migrate` | Behaviour changes as part of T5 — call site must be updated, reviewed, or rewritten |
| `keep`    | Public API surface is unchanged — site continues to work verbatim after T5          |
| `delete`  | Code is removed entirely in T5                                                      |

A `migrate` tag on a *test* or *bench* means the test must still pass
end-to-end against the new substrate-backed `ObrainDB::open`; no code change
is expected, but these are the sites that validate T5 behavioural parity.

---

## 1. Repo: `lab/grafeo`

### 1.1 `open_overlay` — the overlay path (to delete)

| File                                                 | Line  | Snippet                                           | Target   |
|------------------------------------------------------|-------|---------------------------------------------------|----------|
| `crates/obrain-engine/src/database/mod.rs`           | 1376  | `pub fn open_overlay(path) -> Result<(Self, OverlayStore)>` | **delete** |

`open_overlay` is the only parallel open path. Its whole reason for existing
(avoid the full materialization of `ObrainDB::open`) disappears once the
storage layer is mmap-native. The single call site in `obrain-hub` (see
§2.1) is rewritten to use `ObrainDB::open` directly.

### 1.2 `ObrainDB::open` in production code

| File                                                 | Line  | Snippet                                           | Target   |
|------------------------------------------------------|-------|---------------------------------------------------|----------|
| `crates/obrain-engine/src/database/mod.rs`           | 196+  | `ObrainDB::open` method definition                | **migrate** |
| `crates/obrain-cli/src/commands/data.rs`             | 94    | `obrain_engine::ObrainDB::open(&path)`            | keep (auto via open refactor) |
| `crates/obrain-cli/src/commands/mod.rs`              | 34    | `ObrainDB::open(path)` helper                     | keep     |
| `scripts/bench_boot.rs`                              | 31    | `obrain_engine::ObrainDB::open(path)`             | keep     |
| `bench/substrate/src/bench_baseline.rs`              | 63,65 | `TODO(T1-step3): wire ObrainDB::open`             | **migrate** — fill stub |

`ObrainDB::open` itself is the single seam where T5 swaps `LpgStore`
materialization for a `SubstrateStore` constructor (gated by
`substrate-backend` feature). Once this is done, every caller above
transparently moves to the substrate path.

### 1.3 `ObrainDB::open` in tests (validate T5 parity)

All tagged **keep** — the public API is unchanged, but these suites are the
T5 regression surface. They must pass 100% after the backend swap.

| File                                                 | Lines           | Target |
|------------------------------------------------------|-----------------|--------|
| `crates/obrain-cli/tests/commands.rs`                | 8, 30, 34, 192  | keep   |
| `crates/obrain-cli/src/commands/data.rs`             | 190, 240, 356   | keep (inline tests) |
| `crates/obrain-cli/src/commands/index.rs`            | 99              | keep   |
| `crates/obrain-cli/src/commands/compact.rs`          | 145             | keep   |
| `crates/obrain-engine/src/database/mod.rs`           | 2240…2394       | keep (inline tests — 9 sites) |
| `crates/obrain-engine/src/database/persistence.rs`   | 1847, 1897, 1977 | keep  |
| `crates/obrain-engine/tests/wal_recovery.rs`         | 30 sites        | keep (backend-agnostic via `ObrainDB::open`) |
| `crates/obrain-engine/tests/wal_directory.rs`        | 26              | keep   |
| `crates/obrain-engine/tests/obrain_file.rs`          | 758, 764, 781, 813, 854, 898, 960, 990, 1008 | keep |
| `crates/obrain-engine/tests/crash_injection_single_file.rs` | 84, 145, 189, 220, 240 | keep |
| `crates/obrain-engine/tests/agent_memory_migration.rs` | 116, 157, 396, 405, 440, 460, 682 | keep |
| `crates/obrain-engine/tests/megalaw_bench.rs`        | 29              | keep   |
| `crates/obrain-engine/tests/po_vs_neo4j_bench.rs`    | 698             | keep   |
| `crates/obrain-engine/tests/po_profiling_bench.rs`   | 40              | keep   |
| `crates/obrain-engine/tests/property_index_po_bench.rs` | 324, 352, 489 | keep   |

### 1.4 Examples and docs

| File                                                 | Lines    | Target |
|------------------------------------------------------|----------|--------|
| `examples/rust/migrate_v2.rs`                        | 76, 96   | keep   |
| `examples/rust/persistence.rs`                       | 21, 42   | keep   |
| `examples/rust/bench_load.rs`                        | 38       | keep   |
| `crates/obrain-rag/examples/rag_demo.rs`             | 63       | keep   |
| `docs/getting-started/quickstart.md`                 | 33       | keep (prose) |
| `docs/user-guide/rust/database.md`                   | 22, 45   | keep (prose) |
| `docs/api/rust/engine.md`                            | 22       | keep (prose) |
| `CHANGELOG-grafeo.md`                                | 12, 70   | keep (already historical) |
| `.github/ISSUE_TEMPLATE/bug_report.yml`              | 23       | keep   |

### 1.5 Python bindings

| File                                                 | Line  | Snippet                                  | Target |
|------------------------------------------------------|-------|------------------------------------------|--------|
| `crates/bindings/python/src/database.rs`             | 1887  | `obrain_engine::ObrainDB::open_in_memory` | keep — `open_in_memory` out of T5 scope |

---

## 2. Repo: `obrain/obrain-hub`

### 2.1 `ObrainDB::open_overlay` — callers to rewrite

| File                          | Line  | Snippet                                   | Target   |
|-------------------------------|-------|-------------------------------------------|----------|
| `src/knowledge_store.rs`      | 318   | `let (db, _overlay) = ObrainDB::open_overlay(&path)?` | **migrate** — replace with `ObrainDB::open(&path)` |
| `src/knowledge_store.rs`      | 317   | comment "avoids the 2.8GB full materialization that ObrainDB::open() would do" | **migrate** — rewrite comment: substrate open = mmap-native, no full materialization |

### 2.2 `PersistentStore` — the WAL-wrapper to delete

`PersistentStore` exists solely to work around `LpgStore`'s in-memory-only
nature by double-logging mutations to a WAL. With substrate, every mutation
goes through `WalGraphStore` at the engine level, and the wrapper becomes
redundant. All sites are **migrate** (rewrite to `Arc<ObrainDB>` or
`Arc<dyn GraphStoreMut>`) or **delete**.

| File                                  | Lines                       | Snippet / role                                    | Target    |
|---------------------------------------|-----------------------------|---------------------------------------------------|-----------|
| `src/persistent_store.rs`             | (whole file, 17-200+)       | Module definition                                 | **delete** |
| `src/lib.rs`                          | 25                          | `pub mod persistent_store;`                       | **delete** |
| `src/asset_registry.rs`               | 7, 18, 101, 104, 111        | `AssetRegistry { store: PersistentStore }`        | **migrate** — switch to `Arc<ObrainDB>`; fixes WAL bypass gotcha |
| `src/user_brain.rs`                   | 26, 87, 124, 127, 191, 235  | `UserBrain { store: PersistentStore }`            | **migrate** |
| `src/state.rs`                        | 6, 122, 135, 139, 361       | Hub state wiring                                  | **migrate** |
| `src/persona.rs`                      | 27, 107, 111, 116           | `PersonaStore { store: PersistentStore }`         | **migrate** |
| `src/integration/store.rs`            | 4, 34, 38                   | `IntegrationStore { lpg: PersistentStore }`       | **migrate** |
| `src/space/store.rs`                  | 6, 59, 63, 611              | `SpaceStore { lpg: PersistentStore }`             | **migrate** |
| `src/room.rs`                         | 13, 182, 196                | `RoomStore { lpg: Option<PersistentStore> }`      | **migrate** |
| `src/cognitive_brain.rs`              | 443, 525, 537, 542, 1145    | `pstore: PersistentStore` — cognitive stores path | **migrate** — critical, closes the WAL-bypass regression reported in project gotchas |

### 2.3 `ObrainDB::open` in obrain-hub

| File                          | Line  | Snippet                                  | Target   |
|-------------------------------|-------|------------------------------------------|----------|
| `src/user_brain.rs`           | 124   | `ObrainDB::open(&db_path)` — user brain open | keep (goes through refactored open) |
| `src/user_brain.rs`           | 191   | `ObrainDB::open(legacy_path)` — legacy v1 loader | keep |
| `src/state.rs`                | 122   | `ObrainDB::open(&v2_path)` — hub state open | keep |
| `src/state.rs`                | 361   | `ObrainDB::open(&hk_path)` — hub_knowledge_db open | keep |
| `src/routes/spaces.rs`        | 829   | `obrain_engine::ObrainDB::open(&path_clone)` — space mount | keep |
| `examples/debug_delete.rs`    | 24    | example                                  | keep     |
| `examples/dump_brain.rs`      | 20    | example                                  | keep     |

---

## 3. Migration strategy — ordering

Ordering derived from call-site dependencies (see T5 steps 1-7):

1. **Step 1** — Refactor `ObrainDB::open` (grafeo §1.2) behind
   `substrate-backend` feature flag. All grafeo call sites (§1.3, §1.4)
   become the regression surface: cargo test must stay green.
2. **Step 2** — Remove `open_overlay` definition (§1.1) + rewrite its single
   caller (§2.1).
3. **Step 3** — Remove `PersistentStore` module (§2.2 top row) after migrating
   all consumers (§2.2 body rows). The consumers are migrated top-down in the
   obrain-hub source tree; `cognitive_brain.rs` is last because it is the
   biggest and touches the write paths that closed the WAL-bypass regression.
4. **Step 4–5** — Migrate `AssetRegistry` (§2.2 `asset_registry.rs`) with
   explicit `db.commit_wal()` no longer needed — ObrainDB routes every write
   through the WAL by construction. This closes the AssetRegistry WAL-bypass
   gotcha (project global guideline).
5. **Step 6** — Enable `substrate-backend` as the *default* feature for
   obrain-engine. grafeo §1.3 suites now exercise the substrate path
   exclusively.
6. **Step 7** — Full integration test pass across both repos.

Nothing in §1.3 / §1.4 / §1.5 should require source changes. If a test fails
post-migration, it is diagnosing a substrate regression — not an audit miss.

## 4. Out-of-scope (explicitly NOT touched in T5)

- `ObrainDB::open_read_only` (CHANGELOG-grafeo.md:12) — separate path, not
  impacted by backend swap
- `ObrainDB::open_in_memory` (wal_recovery.rs:138, python bindings:1887) —
  in-memory remains in-memory; substrate-backend flag does not apply
- `overlay_store::OverlayStore` — still available as an API type, but only
  constructed internally by obrain-engine; hub callers stop using it

## 5. Call-site totals

| Bucket                                       | Count |
|----------------------------------------------|------:|
| `ObrainDB::open` in grafeo production        | 5     |
| `ObrainDB::open` in grafeo tests/benches     | 70    |
| `ObrainDB::open` in grafeo examples/docs     | 10    |
| `ObrainDB::open_overlay` call sites          | 1     |
| `ObrainDB::open_overlay` definition          | 1     |
| `PersistentStore` import/usage in obrain-hub | ~35 (across 10 files) |
| `ObrainDB::open` in obrain-hub               | 5     |
| **Total tagged sites**                       | ~127  |

## 6. Risk summary

| Risk                                                                   | Severity | Mitigation                                                               |
|------------------------------------------------------------------------|----------|--------------------------------------------------------------------------|
| `PersistentStore::Deref<Target=LpgStore>` — removing it breaks any `&*store.method()` style ambiguity | medium | Each caller already holds a `store: PersistentStore` field; replace the field type, errors surface at compile time |
| `open_overlay` consumers depend on `_overlay: OverlayStore` staying in scope (mmap lifetime) | low | Single caller (`knowledge_store.rs:318`); switching to `ObrainDB::open` removes the mmap/delta split — ObrainDB owns the mmap |
| Cognitive stores (§2.2 `cognitive_brain.rs`) currently write WAL through `PersistentStore` only; removing that middleware must re-route through ObrainDB sessions | **high** | Dedicated T5 steps validate this via the 5-min checkpoint regression scenario documented in `Gotchas`. E2E tests in T17 replay the "Atlas identity" cross-session scenario. |

---

## 7. Verification — this document satisfies T5 Step 0

- ✅ Every file/line touched is listed.
- ✅ Each site has an explicit tag (`migrate` / `keep` / `delete`).
- ✅ Ordering of T5 steps 1-7 is derivable from the audit.
- ✅ Risks are catalogued with severities.

Step 0 ready to close; Step 1 can begin immediately.
