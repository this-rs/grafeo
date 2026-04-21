# Cognitive quality — `/Users/triviere/.obrain/db/po`

*Generated epoch_day=20564 16:21 (UTC) by `obrain-substrate/examples/cognitive_quality.rs`*

## 1. Structural counts

| Metric | Value |
|---|---:|
| Total node slots (incl. null sentinel) | 1687552 |
| Live nodes (non-tombstoned) | 1687551 |
| Tombstoned nodes | 0 |
| Total edge slots (incl. null sentinel) | 2531328 |
| Live edges (non-tombstoned) | 2531327 |
| Tombstoned edges | 0 |

## 2. Community structure

| Metric | Value |
|---|---:|
| Distinct communities | 188060 |
| Unassigned nodes (`community_id = u32::MAX`) | 0 (0.00%) |
| Intra-community edges | 1946802 (96.94%) |
| Inter-community edges | 61485 |
| Edges with unknown endpoint community | 523040 |

> Intra-community fraction acts as a modularity proxy. Gate = ≥ 0.4 (stage 4 LDleiden baseline). Values near 0.5+ indicate crisp community boundaries.

### Size distribution

| Size class | # communities | # members | % of live nodes |
|---|---:|---:|---:|
| size = 1 (singletons) | 187320 | 187320 | 11.10% |
| 2 — 5 | 13 | 39 | 0.00% |
| 6 — 20 | 30 | 449 | 0.03% |
| 21 — 100 | 242 | 12064 | 0.71% |
| 101 — 1 000 | 262 | 85541 | 5.07% |
| 1 001 — 10 000 | 167 | 501617 | 29.72% |
| > 10 000 | 26 | 900521 | 53.36% |

### Top 20 largest communities

| Rank | community_id | size | % of live nodes |
|---:|---:|---:|---:|
| 1 | 101 | 119238 | 7.07% |
| 2 | 102 | 116838 | 6.92% |
| 3 | 10 | 68281 | 4.05% |
| 4 | 0 | 62478 | 3.70% |
| 5 | 103 | 54391 | 3.22% |
| 6 | 2 | 42701 | 2.53% |
| 7 | 206 | 39377 | 2.33% |
| 8 | 30 | 37248 | 2.21% |
| 9 | 929 | 36212 | 2.15% |
| 10 | 2760 | 29481 | 1.75% |
| 11 | 5053 | 29302 | 1.74% |
| 12 | 12388 | 27568 | 1.63% |
| 13 | 130 | 25952 | 1.54% |
| 14 | 1 | 23261 | 1.38% |
| 15 | 355 | 23150 | 1.37% |
| 16 | 816 | 21636 | 1.28% |
| 17 | 246 | 19671 | 1.17% |
| 18 | 12387 | 18638 | 1.10% |
| 19 | 100 | 15128 | 0.90% |
| 20 | 2892 | 14669 | 0.87% |

## 3. Ricci-Ollivier curvature

| Statistic | Value |
|---|---:|
| Live edges | 2531327 |
| Edges flagged `RICCI_STALE` | 0 (0.00%) |
| Edges still at quantised sentinel (`ricci_u8 = 0` ≡ -1.0) | 527652 (20.84%) |
| Mean ricci | -0.3696 |
| p05 | -1.0000 |
| p25 | -0.8902 |
| p50 (median) | 0.0039 |
| p75 | 0.0039 |
| p95 | 0.0196 |

### Distribution

| Range | count | share |
|---|---:|---:|
| strong negative (bottleneck) [-1, -0.5) | 965058 | 38.12% |
| negative [-0.5, -0.1) | 229654 | 9.07% |
| near-zero [-0.1, 0.1) | 1308662 | 51.70% |
| positive [0.1, 0.5) | 27953 | 1.10% |
| strong positive (dense interior) [0.5, 1] | 0 | 0.00% |

> Strongly negative Ricci (`< -0.5`) marks bottleneck edges — candidates for the Dreamer thinker to propose shortcut synapses. Strongly positive Ricci (`> 0.5`) marks interior of dense communities — the Consolidator thinker reinforces those.

## 4. Engrams (Hopfield recall clusters)

| Metric | Value |
|---|---:|
| next_engram_id high-water | 734 |
| Seeded engrams with ≥ 1 live member | 733 |
| Nodes flagged `ENGRAM_SEED` | 0 (0.00%) |

### Size distribution

| Size class | # engrams |
|---|---:|
| 2 — 3 | 3 |
| 4 — 5 | 730 |
| 6 — 10 | 0 |
| 11 — 20 | 0 |
| 21 — 50 | 0 |
| 51+ | 0 |

### Top 20 engrams by member count

| Rank | engram_id | members | sample label mix |
|---:|---:|---:|---|
| 1 | 1 | 5 | 5 members |
| 2 | 2 | 5 | 5 members |
| 3 | 3 | 5 | 5 members |
| 4 | 4 | 5 | 5 members |
| 5 | 5 | 5 | 5 members |
| 6 | 6 | 5 | 5 members |
| 7 | 7 | 5 | 5 members |
| 8 | 8 | 5 | 5 members |
| 9 | 9 | 5 | 5 members |
| 10 | 10 | 5 | 5 members |
| 11 | 11 | 5 | 5 members |
| 12 | 12 | 5 | 5 members |
| 13 | 13 | 5 | 5 members |
| 14 | 14 | 5 | 5 members |
| 15 | 15 | 5 | 5 members |
| 16 | 16 | 5 | 5 members |
| 17 | 17 | 5 | 5 members |
| 18 | 18 | 5 | 5 members |
| 19 | 19 | 5 | 5 members |
| 20 | 20 | 5 | 5 members |

### Member detail for top 5 engrams

#### Engram 1 (5 members)

| NodeId | community_id | labels | centrality (Q0.16) |
|---:|---:|---|---:|
| 8921 | 101 | Alert | 0.0000 |
| 8923 | 101 | Alert | 0.0000 |
| 9918 | 101 | Alert | 0.0000 |
| 9920 | 101 | Alert | 0.0000 |
| 16164 | 101 | Alert | 0.0000 |

#### Engram 2 (5 members)

| NodeId | community_id | labels | centrality (Q0.16) |
|---:|---:|---|---:|
| 8922 | 102 | Alert | 0.0000 |
| 8924 | 102 | Alert | 0.0000 |
| 9919 | 102 | Alert | 0.0000 |
| 16223 | 102 | Alert | 0.0000 |
| 16231 | 102 | Alert | 0.0000 |

#### Engram 3 (5 members)

| NodeId | community_id | labels | centrality (Q0.16) |
|---:|---:|---|---:|
| 47102 | 10 | Trait | 0.0005 |
| 18975 | 10 | Function | 0.0004 |
| 275251 | 10 | Function | 0.0004 |
| 18985 | 10 | Function | 0.0004 |
| 407672 | 10 | Function | 0.0004 |

#### Engram 4 (5 members)

| NodeId | community_id | labels | centrality (Q0.16) |
|---:|---:|---|---:|
| 8934 | 103 | Alert | 0.0000 |
| 9917 | 103 | Alert | 0.0000 |
| 9921 | 103 | Alert | 0.0000 |
| 16224 | 103 | Alert | 0.0000 |
| 16232 | 103 | Alert | 0.0000 |

#### Engram 5 (5 members)

| NodeId | community_id | labels | centrality (Q0.16) |
|---:|---:|---|---:|
| 3 | 2 | Project | 0.0008 |
| 72403 | 2 | Protocol | 0.0000 |
| 4188 | 2 | Commit | 0.0000 |
| 5882 | 2 | Skill | 0.0000 |
| 5883 | 2 | Skill | 0.0000 |

## 5. Node-level signals

| Metric | Value |
|---|---:|
| Mean energy (Q1.15 → f32) | 0.0000 |
| Mean centrality (Q0.16 → f32) | 0.000000 |
| Max centrality | 0.002441 |
| Nodes with non-zero centrality | 1698 (0.10%) |
| Nodes flagged `EMBEDDING_STALE` | 0 (0.00%) |
| Nodes flagged `CENTRALITY_STALE` | 0 (0.00%) |
| Nodes flagged `HILBERT_DIRTY` | 0 (0.00%) |
| Nodes flagged `IDENTITY` | 0 (0.00%) |

