---
title: Knowledge Fabric & Risk Scoring
description: Auto-computed metrics for every graph node — mutation frequency, annotation density, centrality, and composite risk scoring with community detection.
tags:
  - cognitive
  - fabric
  - risk
---

# Knowledge Fabric & Risk Scoring

The **Knowledge Fabric** automatically computes and maintains quality metrics for every node in the graph. It answers the question: *which parts of my knowledge graph need attention?*

## FabricScore — Per-Node Metrics

Every tracked node has a `FabricScore` with these metrics:

| Metric | Description | Range |
|--------|-------------|-------|
| `mutation_frequency` | How often the node has been mutated | 0 → ∞ |
| `annotation_density` | Richness of linked context (notes, decisions) | [0.0, 1.0] |
| `staleness` | Seconds since last mutation (freshness) | 0 → ∞ |
| `pagerank` | Link-structure importance (PageRank) | [0.0, 1.0] |
| `betweenness` | Path-involvement centrality | [0.0, 1.0] |
| `scar_intensity` | Cumulative past failures (rollbacks, errors) | 0 → ∞ |
| `community_id` | Louvain community assignment | `Option<u64>` |
| **`risk_score`** | **Composite weighted sum** | **[0.0, 1.0]** |

## Risk Score Formula

$$
\text{risk} = w_{\text{pr}} \cdot \hat{P} + w_{\text{mf}} \cdot \hat{M} + w_{\text{ag}} \cdot (1 - D) + w_{\text{bt}} \cdot \hat{B} + w_{\text{sc}} \cdot \hat{S}
$$

Where:

- $\hat{P} = \text{normalize}(\text{pagerank}, \max_{\text{pagerank}})$ — importance
- $\hat{M} = \text{normalize}(\text{mutation\_freq}, \max_{\text{mutation\_freq}})$ — volatility (churn)
- $(1 - D)$ — annotation gap (inverse of `annotation_density`) — missing context
- $\hat{B} = \text{normalize}(\text{betweenness}, \max_{\text{betweenness}})$ — bridge criticality
- $\hat{S} = \text{normalize}(\text{scar\_intensity}, \max_{\text{scar}})$ — past failures

Normalization is min-max: $\hat{x} = x / x_{\max}$ (with $\hat{x} = 0$ when $x_{\max} = 0$).

## Default Risk Weights

```rust
RiskWeights {
    pagerank: 0.25,          // structural importance
    mutation_frequency: 0.25, // volatility
    annotation_gap: 0.20,     // missing documentation
    betweenness: 0.15,        // bridge criticality
    scar: 0.15,               // past failures
}
```

### Interpreting Risk

| Risk Score | Interpretation |
|------------|----------------|
| 0.0–0.2 | Low risk — well-documented, stable node |
| 0.2–0.5 | Moderate — some attention needed |
| 0.5–0.8 | High — volatile, poorly documented, or structurally critical |
| 0.8–1.0 | Critical — high churn + important + scarred |

### Custom Weights

Weights are automatically normalized to sum to 1.0:

```rust
let weights = RiskWeights::new(
    0.4,  // emphasize structural importance
    0.3,  // and volatility
    0.1,  // less weight on annotation gaps
    0.1,  // and centrality
    0.1,  // and scars
);
let store = FabricStore::with_risk_weights(weights);
```

## Community Detection

The `GdsRefreshScheduler` periodically runs graph algorithms to update fabric metrics:

| Algorithm | Updates | Schedule |
|-----------|---------|----------|
| **PageRank** | `pagerank` on all nodes | Configurable interval |
| **Louvain** | `community_id` on all nodes | Configurable interval |
| **Betweenness Centrality** | `betweenness` on all nodes | Configurable interval |

After a GDS refresh, all risk scores are **recalculated** with the new global maximums.

## Incremental Updates

The `FabricListener` reacts to every mutation batch in real-time:

```text
MutationBus → FabricListener → FabricStore (DashMap<NodeId, FabricScore>)
                                    │
                                    ├── increment mutation_frequency
                                    ├── reset staleness to 0
                                    ├── recalculate risk_score
                                    └── write-through: risk, mutation_freq, annotation_density
```

No manual refresh needed — fabric metrics stay current with every mutation.

## Querying the Fabric

### High-Risk Nodes

```rust
// Get all nodes with risk_score ≥ 0.7
let risky = fabric_store.high_risk_nodes(0.7);
// Returns Vec<(NodeId, FabricScore)>
```

### Staleness Detection

```rust
let score = fabric_store.get_fabric_score(node_id);
if score.staleness > 86400.0 * 30.0 {
    // Node hasn't been touched in 30+ days
}
```

## Write-Through Persistence

When a backing `GraphStoreMut` is provided, three key metrics are persisted as node properties:

| Property Key | Metric |
|-------------|--------|
| `_cog_risk` | `risk_score` |
| `_cog_mutation_freq` | `mutation_frequency` |
| `_cog_annotation_density` | `annotation_density` |

This enables querying fabric metrics directly via GQL/Cypher without going through the cognitive API.
