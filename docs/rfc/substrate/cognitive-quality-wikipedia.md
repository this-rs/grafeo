# Cognitive quality — `/Users/triviere/.obrain/db/wikipedia/substrate.obrain`

*Generated epoch_day=20564 16:22 (UTC) by `obrain-substrate/examples/cognitive_quality.rs`*

## 1. Structural counts

| Metric | Value |
|---|---:|
| Total node slots (incl. null sentinel) | 5695488 |
| Live nodes (non-tombstoned) | 5695487 |
| Tombstoned nodes | 0 |
| Total edge slots (incl. null sentinel) | 145969408 |
| Live edges (non-tombstoned) | 145969407 |
| Tombstoned edges | 0 |

## 2. Community structure

| Metric | Value |
|---|---:|
| Distinct communities | 30063 |
| Unassigned nodes (`community_id = u32::MAX`) | 0 (0.00%) |
| Intra-community edges | 85936091 (72.12%) |
| Inter-community edges | 33218233 |
| Edges with unknown endpoint community | 26815083 |

> Intra-community fraction acts as a modularity proxy. Gate = ≥ 0.4 (stage 4 LDleiden baseline). Values near 0.5+ indicate crisp community boundaries.

### Size distribution

| Size class | # communities | # members | % of live nodes |
|---|---:|---:|---:|
| size = 1 (singletons) | 30007 | 30007 | 0.53% |
| 2 — 5 | 14 | 46 | 0.00% |
| 6 — 20 | 8 | 62 | 0.00% |
| 21 — 100 | 1 | 26 | 0.00% |
| 101 — 1 000 | 0 | 0 | 0.00% |
| 1 001 — 10 000 | 7 | 18533 | 0.33% |
| > 10 000 | 26 | 5646813 | 99.15% |

### Top 20 largest communities

| Rank | community_id | size | % of live nodes |
|---:|---:|---:|---:|
| 1 | 0 | 1350797 | 23.72% |
| 2 | 6 | 770078 | 13.52% |
| 3 | 3 | 491979 | 8.64% |
| 4 | 2 | 469308 | 8.24% |
| 5 | 9 | 443176 | 7.78% |
| 6 | 4 | 407888 | 7.16% |
| 7 | 1 | 246046 | 4.32% |
| 8 | 16 | 232862 | 4.09% |
| 9 | 8 | 203341 | 3.57% |
| 10 | 7 | 199082 | 3.50% |
| 11 | 20 | 176064 | 3.09% |
| 12 | 5 | 101436 | 1.78% |
| 13 | 17 | 87386 | 1.53% |
| 14 | 26 | 69693 | 1.22% |
| 15 | 25 | 52889 | 0.93% |
| 16 | 14 | 49308 | 0.87% |
| 17 | 23 | 42343 | 0.74% |
| 18 | 10 | 40771 | 0.72% |
| 19 | 12 | 38822 | 0.68% |
| 20 | 22 | 37597 | 0.66% |

## 3. Ricci-Ollivier curvature

| Statistic | Value |
|---|---:|
| Live edges | 145969407 |
| Edges flagged `RICCI_STALE` | 0 (0.00%) |
| Edges still at quantised sentinel (`ricci_u8 = 0` ≡ -1.0) | 26851906 (18.40%) |
| Mean ricci | -0.1149 |
| p05 | -1.0000 |
| p25 | -0.9608 |
| p50 (median) | -0.8745 |
| p75 | -0.7333 |
| p95 | -0.3020 |

### Distribution

| Range | count | share |
|---|---:|---:|
| strong negative (bottleneck) [-1, -0.5) | 130075103 | 89.11% |
| negative [-0.5, -0.1) | 12619205 | 8.65% |
| near-zero [-0.1, 0.1) | 3249321 | 2.23% |
| positive [0.1, 0.5) | 25778 | 0.02% |
| strong positive (dense interior) [0.5, 1] | 0 | 0.00% |

> Strongly negative Ricci (`< -0.5`) marks bottleneck edges — candidates for the Dreamer thinker to propose shortcut synapses. Strongly positive Ricci (`> 0.5`) marks interior of dense communities — the Consolidator thinker reinforces those.

## 4. Engrams (Hopfield recall clusters)

| Metric | Value |
|---|---:|
| next_engram_id high-water | 51 |
| Seeded engrams with ≥ 1 live member | 50 |
| Nodes flagged `ENGRAM_SEED` | 0 (0.00%) |

### Size distribution

| Size class | # engrams |
|---|---:|
| 2 — 3 | 2 |
| 4 — 5 | 48 |
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
| 2190024 | 6 | Article | 0.0012 |
| 2631934 | 6 | Article | 0.0012 |
| 3608035 | 6 | Article | 0.0008 |
| 4495074 | 6 | Article | 0.0007 |
| 4560474 | 6 | Portal | 0.0004 |

#### Engram 2 (5 members)

| NodeId | community_id | labels | centrality (Q0.16) |
|---:|---:|---|---:|
| 1862693 | 3 | Article | 0.0008 |
| 4560432 | 3 | Portal | 0.0004 |
| 4560299 | 3 | Portal | 0.0003 |
| 4175630 | 3 | Article | 0.0003 |
| 4560356 | 3 | Portal | 0.0003 |

#### Engram 3 (5 members)

| NodeId | community_id | labels | centrality (Q0.16) |
|---:|---:|---|---:|
| 2054425 | 2 | Article | 0.0005 |
| 2434147 | 2 | Article | 0.0005 |
| 2683335 | 2 | Article | 0.0005 |
| 3506606 | 2 | Article | 0.0005 |
| 2311292 | 2 | Article | 0.0005 |

#### Engram 4 (5 members)

| NodeId | community_id | labels | centrality (Q0.16) |
|---:|---:|---|---:|
| 3550126 | 9 | Article | 0.0008 |
| 4209695 | 9 | Article | 0.0007 |
| 3894350 | 9 | Article | 0.0004 |
| 2617510 | 9 | Article | 0.0003 |
| 2628693 | 9 | Article | 0.0002 |

#### Engram 5 (5 members)

| NodeId | community_id | labels | centrality (Q0.16) |
|---:|---:|---|---:|
| 2315575 | 4 | Article | 0.0025 |
| 2807262 | 4 | Article,List | 0.0013 |
| 2055725 | 4 | Article | 0.0011 |
| 2256041 | 4 | Article | 0.0010 |
| 4246986 | 4 | Article | 0.0008 |

## 5. Node-level signals

| Metric | Value |
|---|---:|
| Mean energy (Q1.15 → f32) | 0.4005 |
| Mean centrality (Q0.16 → f32) | 0.000000 |
| Max centrality | 0.002487 |
| Nodes with non-zero centrality | 12212 (0.21%) |
| Nodes flagged `EMBEDDING_STALE` | 0 (0.00%) |
| Nodes flagged `CENTRALITY_STALE` | 0 (0.00%) |
| Nodes flagged `HILBERT_DIRTY` | 0 (0.00%) |
| Nodes flagged `IDENTITY` | 0 (0.00%) |

