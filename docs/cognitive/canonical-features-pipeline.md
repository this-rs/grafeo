---
title: Canonical Cognitive Features Pipeline
description: How `_hilbert_features` / `_kernel_embedding` / `_st_embedding` are produced, stored, and consumed across the Obrain stack — the single source-of-truth pipeline from Neo4j import to hub retrieval.
tags:
  - cognitive
  - substrate
  - retrieval
  - enrichment
  - pipeline
---

# Canonical Cognitive Features Pipeline

**Status** : v1 (T17l, 2026-04-24) — single canonical pipeline, no parallel paths.

This document describes **the** one system that produces the three canonical cognitive features required by the retrieval pipeline, persists them in the substrate, and consumes them from the hub. All other enrichment entry points converge on these same keys + zones.

## The three canonical features

| Key | Dim | Type | Produced by | Depends on |
|-----|----:|------|-------------|------------|
| `_hilbert_features` | 64 | `Vector` | `obrain-adapters::plugins::algorithms::hilbert_features` | graph topology only |
| `_kernel_embedding` | 80 | `Vector` | `obrain-adapters::plugins::algorithms::kernel_manager::KernelManager` | `_hilbert_features` (per-node + neighborhood) |
| `_st_embedding` | 384 | `Vector` | ONNX MiniLM sentence-transformer (via `obrain-engine::embedding::onnx` or `obrain-chat::retrieval::sentence_transformer`) | node text properties (`title` / `name` / `content` / `abstract` / …) |

All three are stored as `Value::Vector` properties on nodes, persisted via the substrate **vec_columns** zone (`substrate.veccol.node.NNNN.f32.DIM`).

## System overview

```mermaid
graph TB
    subgraph ImportTime[Import Time — neo4j2obrain CLI]
        N4J[Neo4j source]
        N4J -->|"bolt stream"| S1[Stage 1 — structural]
        S1 -->|_st_embedding| S2[Stage 2 — ONNX MiniLM 384d]
        S2 -->|tiers L0/L1/L2| S3[Stage 3 — SubstrateTieredIndex]
        S3 --> S4to8[Stages 4-8 — Leiden (batch) / PageRank / Ricci / COACT / engrams]
        S4to8 -->|"_hilbert_features"| S9[Stage 9 — Hilbert TOPO 64d]
        S9 -->|"_kernel_embedding"| S10[Stage 10 — Kernel Φ₀ 80d]
    end

    subgraph Substrate[Substrate — on-disk zones at ~/.obrain/db/&lt;name&gt;/]
        Veccol["substrate.veccol.node.NNNN.f32.DIM<br/>— _hilbert_features (f32.64)<br/>— _kernel_embedding  (f32.80)<br/>— _st_embedding       (f32.384)"]
        Blobcol["substrate.blobcol.node.NNNN.*<br/>— String/Bytes props > 256 B"]
        Propsv2["substrate.props.v2.*<br/>— Scalar props chained on NodeRecord"]
        Tiers["substrate.tier0 / tier1 / tier2<br/>— built from _st_embedding"]
        Syn["substrate.synapse.*<br/>— SYNAPSE edges (cognitive L2)"]
        Engrams["substrate.engram.*<br/>— Engrams (cognitive L0)"]
    end

    S10 --> Veccol
    S3 --> Tiers
    S4to8 --> Syn
    S4to8 --> Engrams

    subgraph RuntimeEnrich[Runtime Enrichment — wardens on live bases]
        HE["HilbertEnricher thinker<br/>(obrain-cognitive, feature enrichment)"]
        KE["KernelEnricher thinker<br/>(future — T17l Phase 2)"]
        SE["StEnricher thinker<br/>(future — T17l Phase 3)"]
        Cons["Consolidator · CommunityWarden<br/>· Predictor · Dreamer"]
    end

    HE -.->|"incremental fallback<br/>when coverage &lt; 10%"| Veccol
    KE -.->|"incremental fallback"| Veccol
    SE -.->|"incremental fallback"| Veccol
    Cons -.->|"maintenance (energy decay,<br/>community compaction, …)"| Substrate

    subgraph Hub[obrain-hub runtime]
        KSM["KnowledgeStoreManager::get_or_open"]
        Fleet["spawn_standard_fleet<br/>(5 threads per KB)"]
        CachedStore["CachedStore<br/>(LRU cache)"]
        KSM --> CachedStore
        KSM --> Fleet
        Fleet --> HE
        Fleet --> KE
        Fleet --> SE
        Fleet --> Cons
    end

    subgraph Retrieval[Retrieval — obrain-chat ContextEngine]
        PQM["prepare_context_multi_store"]
        QEmb["Query → ST 384d"]
        BM25["BM25 pre-filter / substring"]
        ONNX["ONNX rerank"]
        CompE["CompositeEmbedding 532d<br/>= _kernel_embedding (80) ⊕<br/>  _hilbert_features (64) ⊕<br/>  _st_embedding     (384) ⊕<br/>  degree features   (4)"]
        Top["Top-K nodes → PreparedContext"]
        PQM --> QEmb
        QEmb --> BM25
        BM25 --> ONNX
        ONNX --> CompE
        CompE --> Top
    end

    CachedStore -->|"Arc&lt;dyn GraphStore&gt;"| PQM
    Substrate -.->|"read via get_node_property"| CompE

    subgraph UserBrain[UserBrain — user's own growing brain.db]
        UB["UserBrain::open"]
        UBFleet["spawn_standard_fleet<br/>(on brain.db)"]
        UB --> UBFleet
        UBFleet --> HE
        UBFleet --> KE
        UBFleet --> SE
        UBFleet --> Cons
    end

    style Veccol fill:#1a73e8,color:#fff
    style Tiers fill:#34a853,color:#fff
    style CompE fill:#9c27b0,color:#fff
    style HE fill:#ff6d00,color:#fff
    style ImportTime fill:#fff3e0,color:#000
    style Hub fill:#e8f5e9,color:#000
    style Retrieval fill:#f3e5f5,color:#000
    style UserBrain fill:#fce4ec,color:#000
```

## Separation of concerns

### 1. Filesystem (substrate zones) — source of truth

Features live on disk as typed mmap zones. There is **exactly one key per canonical feature** and **one zone per (key, dim, dtype)** :

```text
~/.obrain/db/<name>/
├── substrate.veccol.node.0000.f32.64        ← _hilbert_features
├── substrate.veccol.node.0001.f32.80        ← _kernel_embedding
├── substrate.veccol.node.0002.f32.384       ← _st_embedding
├── substrate.tier0 / tier1 / tier2           ← built from _st_embedding
├── substrate.blobcol.node.NNNN.*             ← String/Bytes > 256B
├── substrate.props.v2.*                      ← scalar + small String
└── substrate.synapse.* / engram.* / …         ← cognitive layers
```

No other file or zone carries these vectors. No legacy `embedding` key. If a base has a mismatched key (e.g. legacy `embedding` at 80d), it is **not canonical** and must be re-enriched.

### 2. Enrichment (producers) — one tool, multiple stages

**`neo4j2obrain`** is the canonical enrichment tool. Both for fresh imports and for `--enrich-only` re-runs on existing substrates. All other "enrich" points (the `HilbertEnricher` / `KernelEnricher` thinkers) are **runtime fallbacks** for bases opened in the hub that were not pre-enriched — they use the same algorithms and write the same keys, converging on the same end state.

Stage matrix (implemented ✓ / to do ⏳) :

| Stage | Key / Output | Status | Algorithm |
|-------|--------------|:------:|-----------|
| 1 | Structural nodes + edges | ✓ | bolt stream |
| 2 | `_st_embedding` 384d | ✓ | ONNX MiniLM |
| 3 | `substrate.tier0/1/2` | ✓ | `SubstrateTieredIndex` build from Stage 2 |
| 4 | `community_id` | ✓ | Leiden (batch) batch |
| 5 | `centrality_cached` | ✓ | PageRank batch |
| 6 | `edge.ricci` | ✓ | Ricci-Ollivier batch |
| 7 | COACT edges | ✓ | co-activation seeding |
| 8 | Engrams | ✓ | initial seeding |
| **9** | **`_hilbert_features` 64d** | **⏳ T17l** | `obrain-adapters::hilbert_features` |
| **10** | **`_kernel_embedding` 80d** | **⏳ T17l** | `KernelManager::compute_all` (depends on Hilbert) |

`neo4j2obrain --enrich-only <path>` is idempotent per stage : coverage probe at the start of each stage skips it if already populated (unless `--force-rerun-cognitive`).

### 3. Hub (consumer) — opens ready bases, runs maintenance

The hub's `KnowledgeStoreManager` opens bases that were already enriched externally. The `spawn_standard_fleet` that fires on open is **maintenance** (Consolidator / CommunityWarden / Predictor / Dreamer) plus the **Enricher thinkers** which serve as a safety net — they probe coverage and only run the missing phases.

If a base was fully enriched via `neo4j2obrain`, the enricher thinkers skip on every tick (coverage probe short-circuits in < 1 ms). If a user connects a legacy base that was not enriched, the enricher thinkers progressively populate the missing keys in background without blocking the hub.

### 4. Retrieval (consumer) — CompositeEmbedding fuses the three canonical features

The retrieval pipeline (`obrain-chat::context-engine::prepare_context_multi_store`) performs two-stage retrieval : BM25 pre-filter → ONNX rerank. The rerank uses `CompositeEmbedding` which reads the **three canonical keys** and concatenates + L2-normalises them into a 532d vector for cosine similarity :

```text
composite(node) = concat(
    L2(_kernel_embedding),   // 80d · w_phi0    = 0.25
    L2(_hilbert_features),    // 64d · w_hilbert = 0.15
    L2(_st_embedding),         // 384d · w_text   = 0.55
    L2(degree_features),       //  4d · w_degree = 0.05
) -> L2-normalised 532d
```

The query is `ST(query_text) → 384d`, zero-padded on the structural dims. Cosine similarity against the composite vectors ranks candidates.

## User Brain vs Knowledge Base

| Aspect | UserBrain (`brain.db`) | Knowledge Base (`~/.obrain/db/<name>/`) |
|--------|------------------------|------------------------------------------|
| **Scope** | Per-user, grows continuously (CogMessages, Identity, …) | Imported corpus (PO / Wikipedia / Megalaw / …) |
| **Enrichment** | Runtime — thinkers run continuously | Import-time — `neo4j2obrain` runs all stages once, then bases are "ready" |
| **Spawned by** | `UserBrain::open` (hub) | `KnowledgeStoreManager::get_or_open` (hub) |
| **Fleet** | 4 standard + Enrichers | 4 standard + Enrichers (mostly idle on ready bases) |
| **Source of truth** | Live substrate mutations | On-disk substrate (read-mostly) |

## Non-goals (explicit)

- **No "separate tool per feature"** : Hilbert and Kernel enrichment go into `neo4j2obrain` as Stages 9-10, not a new binary.
- **No legacy `embedding` key support** : the canonical keys are `_hilbert_features` / `_kernel_embedding` / `_st_embedding`. Bases with the legacy `embedding` key must be re-enriched (the retrieval layer does not read legacy keys).
- **No in-hub heavy batch computation** : the hub stays responsive ; the enricher thinkers run in background threads and are safe no-op when coverage is already high.

## Operational workflow

1. **Import a new base** :
   ```bash
   neo4j2obrain \
       --neo4j-url bolt://localhost:7687 --user neo4j --password "$PASS" \
       --out ~/.obrain/db/myproject
   # → runs Stages 1-10 in sequence, base is fully canonical when done
   ```

2. **Re-enrich an existing base** (e.g. PO imported before Stages 9-10 existed) :
   ```bash
   neo4j2obrain --enrich-only --out ~/.obrain/db/po
   # → probes each stage, runs only what's missing
   # → Stage 9 (Hilbert) + Stage 10 (Kernel) added in T17l
   ```

3. **Hub opens a ready base** :
   ```rust
   let store = manager.get_or_open("~/.obrain/db/po").await?;
   // → fleet spawned, enricher thinkers skip (coverage 100%)
   // → retrieval works via CompositeEmbedding on canonical features
   ```

4. **Hub opens a not-fully-enriched base** (safety net) :
   ```rust
   let store = manager.get_or_open("~/.obrain/db/new_kb").await?;
   // → fleet spawned
   // → enricher thinkers progressively populate missing keys over minutes
   // → retrieval quality ramps up as each canonical key becomes populated
   ```

## Related docs

- [`cognitive/overview.md`](overview.md) — the 7 cognitive layers (energy / synapses / engrams / stigmergy / hopfield / epigenetics / homeostasis)
- [`cognitive/rag.md`](rag.md) — RAG pipeline (engram recall + spreading activation)
- [`rfc/substrate/format-spec.md`](../rfc/substrate/format-spec.md) — substrate on-disk format
- [`architecture/crates.md`](../architecture/crates.md) — crate-by-crate responsibilities
