---
title: Transactions
description: Transaction management and MVCC.
tags:
  - architecture
  - transactions
---

# Transactions

Grafeo provides ACID transactions with MVCC (Multi-Version Concurrency Control).

## Why MVCC with Snapshot Isolation?

Graph traversals can touch many nodes and edges in a single query. Locking all of them would cause severe contention. Snapshot isolation lets readers proceed without blocking writers, and vice versa. Write-write conflicts are detected at commit time. This is the right fit for workloads where reads vastly outnumber writes.

## How MVCC Works

Each row has version metadata:

```text
Row:
├── created_txn: 100
├── deleted_txn: NULL (or txn that deleted)
└── data: {...}
```

A row is visible to transaction T if:

1. `created_txn < T` (created before T started)
2. `deleted_txn IS NULL OR deleted_txn > T` (not deleted before T)

Updates create new versions linked in a chain:

```text
Row v3 (current) <- Row v2 <- Row v1 (oldest)
```

## Garbage Collection

Old versions are cleaned up when no transaction needs them:

```text
Active transactions: [txn 100, txn 105]
Oldest active: 100
Safe to remove: versions with deleted_txn < 100
```

## Snapshot Isolation

- Each transaction sees a consistent snapshot
- Reads never block writes
- Writes never block reads
- Write conflicts detected at commit

### Phenomena Prevented

| Phenomenon | Prevented? |
| ---------- | ---------- |
| Dirty Read | Yes |
| Non-Repeatable Read | Yes |
| Phantom Read | Yes |
| Write Skew | Partially |

## Conflict Detection

Write-write conflicts are detected at commit time. If two concurrent transactions modify the same entity, the second to commit fails with a conflict error.
