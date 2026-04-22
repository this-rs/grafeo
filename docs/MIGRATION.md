---
title: Migration Guide — LpgStore → SubstrateStore
description: Step-by-step migration of legacy `.obrain` databases to the Substrate format.
tags:
  - migration
  - substrate
  - operations
---

# Migration Guide — LpgStore → SubstrateStore

**Applies to:** Obrain ≥ 0.2.0 (T17 Substrate cutover).

As of this release, Obrain's primary storage backend is
[`SubstrateStore`](./architecture/overview.md#storage) — a single mmap'd
file where the topology is the storage. The legacy `LpgStore`
(in-memory graph + bincode snapshots + separate HNSW/BM25 indexes) is
deprecated and will be removed in a subsequent release.

This guide walks you through migrating existing `.obrain` databases
without data loss.

## TL;DR

```bash
# 1. Stop the hub / any writer
obrain-hub stop

# 2. Snapshot the legacy file as insurance
cp -a /path/to/db.obrain /path/to/db.obrain.pre-substrate.bak

# 3. Run the migrator (dry-run first — ALWAYS)
obrain-migrate --db /path/to/db.obrain --dry-run

# 4. Run it for real
obrain-migrate --db /path/to/db.obrain --upgrade

# 5. Reopen with the new binary — substrate is auto-detected
obrain-hub start
```

## What changes

| Aspect | Legacy (LpgStore) | Substrate |
|---|---|---|
| On-disk layout | `.obrain` bincode snapshot + sidecar WAL | `substrate.obrain/` directory (mmap zones + WAL) |
| Open time (10⁶ nodes) | 5–15 s (full RAM materialization) | ≤ 100 ms (mmap lazy-load) |
| RSS anonymous | 5–10 GiB | ≤ 1 GiB |
| RSS total (10 users + 5 KBs) | ~110 GiB | ≤ 18 GiB |
| Retrieval p95 (10⁶ nodes) | 30–100 ms (HNSW) | ≤ 1 ms (L0→L1→L2 cascade) |
| Recall@10 vs HNSW f32 | 100 % | ≥ 99 % |
| Cognitive state | 7 separate stores, rehydrated on open, WAL-bypass gotchas | Inline columns on nodes/edges, WAL-logged by construction |

## Snapshot and rollback plan

**Always snapshot before migrating.** The migrator is one-way in the
sense that the substrate directory is a new layout; the `.obrain`
file is not modified in place, but any downstream tooling that only
knows the legacy format will not understand the new directory.

- Keep the `.obrain.pre-substrate.bak` file for **≥ 7 days** on hot
  storage. If a regression surfaces during that window, you can roll
  back to a binary ≤ 0.1.x (tagged on crates.io and GitHub releases
  under `legacy-v0.1`) that still reads the legacy format.
- After the 7-day window, archive the `.bak` to cold storage; you can
  re-run the migrator at any time later.

## Step-by-step

### 1. Quiesce writers

Stop any process that holds the database open for writes. This
includes `obrain-hub`, any CLI with `--read-write`, and any in-process
embedders.

```bash
systemctl stop obrain-hub
# or if you run it in a terminal:
kill -TERM $(pgrep -f obrain-hub)
```

### 2. Back up the legacy file

```bash
cp -a /var/lib/obrain/db.obrain /var/lib/obrain/db.obrain.pre-substrate.bak
```

The migrator does not modify the source file, but the defence-in-depth
cost is negligible.

### 3. Dry-run the migrator

```bash
obrain-migrate \
  --db /var/lib/obrain/db.obrain \
  --dry-run \
  --log-level info
```

The dry-run will:

- Validate the `.obrain` file header and checksums.
- Enumerate nodes, edges, properties, and indexes.
- Report the expected substrate layout (node zone size, edge zone size,
  property page count, string heap size).
- **Not write anything.**

Inspect the output. If the dry-run reports errors, do not proceed —
file an issue with the dry-run log attached.

### 4. Run the upgrade

```bash
obrain-migrate \
  --db /var/lib/obrain/db.obrain \
  --upgrade \
  --out /var/lib/obrain/substrate.obrain
```

The migrator writes to `substrate.obrain/` (a directory). The original
`.obrain` file is untouched.

Under the hood:

1. Read the legacy snapshot into memory (one-shot).
2. Allocate the substrate zones (nodes, edges, props, string heap).
3. Write records with the new 32 B / 30 B layout.
4. Rebuild embedding tiers L0 / L1 / L2 from the existing `embedding`
   property.
5. Run initial `LDleiden` to populate `community_id`.
6. `msync` all zones + fsync the WAL checkpoint.

The migrator is idempotent with respect to the substrate output: if
`--out` already exists, it fails unless `--force` is passed.

### 5. Point the hub at the new database

Update `hub.toml`:

```toml
[storage]
# Old:
# path = "/var/lib/obrain/db.obrain"

# New:
path = "/var/lib/obrain/substrate.obrain"
```

Then restart:

```bash
systemctl start obrain-hub
```

The hub auto-detects the substrate format by directory layout — no
feature flag is needed.

### 6. Verify

```bash
# Startup time (expect ≤ 100 ms on 10⁶ nodes):
time obrain-cli --db /var/lib/obrain/substrate.obrain ping

# Counts match:
obrain-cli --db /var/lib/obrain/substrate.obrain stats
# node_count: <same as before>
# edge_count: <same as before>

# Retrieval sanity check:
obrain-cli --db /var/lib/obrain/substrate.obrain \
  vector-search --label Doc --property embedding --k 10 --query "<any text>"
```

Run your normal smoke tests against the hub before decommissioning
the backup.

## Rollback

If anything is wrong within the 7-day window:

```bash
systemctl stop obrain-hub
mv /var/lib/obrain/substrate.obrain /var/lib/obrain/substrate.obrain.broken
cp -a /var/lib/obrain/db.obrain.pre-substrate.bak /var/lib/obrain/db.obrain
# Point hub.toml back at the .obrain path, downgrade to obrain ≤ 0.1.x
systemctl start obrain-hub
```

Then file an issue with:

- The dry-run log
- The upgrade log
- The `substrate.obrain.broken` directory (tar it up) if size permits

## FAQ

**Q: Do I need to re-import from Neo4j?**
No. `obrain-migrate` reads the existing `.obrain` file directly. A
re-import is only needed if you want to pick up new `neo4j2obrain`
features (auto-computed embeddings, initial Ricci curvature, etc.).

**Q: Can I run substrate and legacy side by side?**
Yes, on different paths. A single process opens a single database.

**Q: What about custom cognitive extensions that wrote to LpgStore
directly?**
They must be ported to `SubstrateStore` or to the `GraphStoreMut`
trait. See the [architecture overview](./architecture/overview.md) and
the T6 cognitive stores migration (commit `84b0f5d1`).

**Q: What happens to my HNSW and BM25 indexes?**
HNSW is replaced by the L0/L1/L2 tier cascade (rebuilt during
migration). BM25 stays on the same trait — the inverted index is
still an `obrain-core::index::text::InvertedIndex`, rebuilt from the
substrate node stream.

## References

- [RFC-SUBSTRATE](./rfc/substrate/format-spec.md) — on-disk format
  specification, bit-precise.
- [WAL spec](./rfc/substrate/wal-spec.md) — durability semantics.
- [T9 validation gates](../crates/obrain-substrate/README.md) — recall,
  p95, RSS, startup.
- [Cognitive quality reports](./rfc/substrate/cognitive-quality.md) —
  end-to-end quality measurements on PO / megalaw / wikipedia.
