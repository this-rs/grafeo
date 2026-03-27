---
title: Synapses & Spreading Activation
description: Hebbian synapses learn from co-activation patterns, and spreading activation propagates relevance through the graph.
tags:
  - cognitive
  - synapses
  - spreading-activation
---

# Synapses & Spreading Activation

## Hebbian Synapses — "Nodes That Fire Together, Wire Together"

A **synapse** is a weighted connection between two nodes, learned automatically via co-activation. When two nodes are mutated or queried in the same batch, the synapse between them is **reinforced**. Unused synapses decay over time.

### How Synapses Work

1. **Co-activation** — Two nodes appear in the same mutation batch.
2. **Synapse creation/reinforcement** — If no synapse exists, one is created with `initial_weight`. Otherwise, `reinforce_amount` is added.
3. **Exponential decay** — Synapse weight decays with the same model as energy:

$$
W(t) = W_0 \times 2^{-\Delta t / t_{1/2}}
$$

Default half-life is 7 days — synapses between rarely co-used nodes fade naturally.

### Competitive Hebbian Learning

To prevent runaway weight accumulation, Obrain applies **competitive normalization**: when the total outgoing weight from a node exceeds `max_total_outgoing_weight`, all outgoing weights are normalized proportionally.

```rust
SynapseConfig {
    initial_weight: 0.1,          // weight for new synapses
    reinforce_amount: 0.2,        // added on each co-activation
    default_half_life: Duration::from_secs(604800), // 7 days
    min_weight: 0.01,             // pruning threshold
    max_synapse_weight: 10.0,     // individual cap
    max_total_outgoing_weight: 100.0, // competitive normalization
}
```

### Synapse Properties

Each synapse tracks:

| Property | Description |
|----------|-------------|
| `source` / `target` | Connected node pair |
| `weight` | Current weight (decays over time) |
| `reinforcement_count` | How many times co-activated |
| `last_reinforced` | Timestamp of last reinforcement |
| `created_at` | When the synapse first formed |
| `half_life` | Decay half-life |

### Mutation Frequency Score

For integration with the knowledge fabric, synapse activity is normalized:

$$
\text{mf\_score} = 1 - e^{-\text{frequency} / \text{ref}}
$$

## Spreading Activation — BFS Energy Propagation

**Spreading activation** propagates energy from source nodes through synapses using BFS. It discovers contextually relevant subgraphs by following synapse connections, attenuating energy at each hop.

### Algorithm

```text
1. Initialize queue with source nodes at initial_energy = 1.0
2. For each node in the queue:
   a. For each outgoing synapse (weight W):
      propagated = current_energy × W × decay_factor
   b. If propagated > min_propagated_energy AND hops < max_hops:
      add neighbor to queue with propagated energy
   c. Accumulate energy per node (multiple paths add up)
3. Return ActivationMap: node → total accumulated energy
```

### Configuration

```rust
SpreadConfig {
    max_hops: 3,                   // BFS depth limit
    min_propagated_energy: 0.01,   // cutoff threshold
    decay_factor: 0.5,             // energy retained per hop
    activation_threshold: 0.0,     // minimum energy for inclusion
    max_activated_nodes: 1000,     // circuit breaker
}
```

### Example

```text
Source: NodeA (energy=1.0)
  │
  ├─── synapse(w=0.8) → NodeB: 1.0 × 0.8 × 0.5 = 0.40
  │       │
  │       └─── synapse(w=0.6) → NodeD: 0.40 × 0.6 × 0.5 = 0.12
  │
  └─── synapse(w=0.3) → NodeC: 1.0 × 0.3 × 0.5 = 0.15

ActivationMap = { A: 1.0, B: 0.40, C: 0.15, D: 0.12 }
```

### Circuit Breaker

The `max_activated_nodes` parameter prevents runaway activation in dense graphs. Once the limit is reached, no new nodes are enqueued — only existing nodes continue processing.

### Multiple Sources

Spreading activation supports multiple source nodes simultaneously. Energy from different sources accumulates at shared neighbors, naturally surfacing nodes that bridge multiple active contexts.

## Architecture

```text
MutationBus → SynapseListener → SynapseStore (DashMap<(NodeId, NodeId), Synapse>)
                                     │
                                     ├── detect co-activated node pairs
                                     ├── create/reinforce synapses
                                     ├── competitive normalization
                                     └── write-through to graph store

                            spread(sources, synapse_store, config)
                                     │
                                     └── BFS → ActivationMap
```

The `SynapseStore` uses `DashMap` for lock-free concurrent access with optional write-through persistence.
