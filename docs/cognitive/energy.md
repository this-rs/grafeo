---
title: Energy & Decay
description: Node energy system with exponential decay, activation boosting, and structural reinforcement.
tags:
  - cognitive
  - energy
  - decay
---

# Energy & Decay

Every node in Obrain's cognitive layer has an **energy level** that reflects its recent importance. Energy decays exponentially over time — nodes that aren't accessed gradually fade from active consideration, while frequently used nodes stay "hot".

## The Decay Model

$$
E(t) = E_0 \times 2^{-\Delta t / t_{1/2}}
$$

Where:

- $E_0$ — energy at last activation
- $\Delta t$ — elapsed time since last activation
- $t_{1/2}$ — half-life (default: 24 hours)

After one half-life, energy drops to 50%. After two, 25%. This creates natural forgetting without hard cutoffs.

### Example

```text
t=0h    E=1.0   ████████████████████
t=12h   E=0.71  ██████████████
t=24h   E=0.50  ██████████
t=48h   E=0.25  █████
t=72h   E=0.125 ██
```

## Boosting — Reactivation

When a node is mutated, queried, or otherwise accessed, it receives an **energy boost**:

```text
E_new = E_current(t) + boost_amount
```

The boost is applied *after* computing the current decayed energy, then the activation timestamp resets. This means frequently accessed nodes accumulate energy and resist decay.

## Structural Reinforcement

Highly-connected nodes retain energy longer via **structural reinforcement**:

$$
t_{1/2}^{\text{eff}} = t_{1/2}^{\text{base}} \times (1 + \alpha \cdot \ln(1 + \text{degree}))
$$

Where:

- $\alpha$ — structural reinforcement coefficient (default: 0.0 = disabled)
- $\text{degree}$ — node degree (number of connections)

Hub nodes with many connections get a longer effective half-life, reflecting their structural importance.

### Example (α = 0.5)

| Degree | Effective Half-Life |
|--------|-------------------|
| 0 | 24h (base) |
| 5 | 24h × 1.90 = 45.5h |
| 20 | 24h × 2.52 = 60.5h |
| 100 | 24h × 3.31 = 79.4h |

## Normalized Energy Score

Raw energy values are unbounded. For cross-metric comparison, use the normalized score:

$$
\text{score}(E) = 1 - e^{-E / E_{\text{ref}}}
$$

- $E = 0$ → score = 0
- $E \to \infty$ → score → 1
- $E_{\text{ref}}$ controls the curve spread (default: 1.0)

## Configuration

```rust
EnergyConfig {
    boost_on_mutation: 1.0,            // energy added per mutation
    default_energy: 1.0,               // initial energy for new nodes
    default_half_life: Duration::from_secs(86400), // 24 hours
    min_energy: 0.01,                  // below this → "low energy"
    max_energy: 10.0,                  // energy cap
    ref_energy: 1.0,                   // normalization reference
    structural_reinforcement_alpha: 0.0, // set > 0 to enable
}
```

## Low-Energy Detection

Nodes whose energy falls below `min_energy` are candidates for archival or eviction:

```rust
let cold_nodes = energy_store.list_low_energy(0.01);
// Returns all NodeIds with current energy < 0.01
```

This enables automatic memory management — cold nodes can be archived to disk or pruned from working memory.

## Architecture

The `EnergyListener` implements `MutationListener` and reacts to every mutation batch:

```text
MutationBus → EnergyListener → EnergyStore (DashMap<NodeId, NodeEnergy>)
                                    │
                                    ├── boost affected nodes
                                    ├── clamp to [0, max_energy]
                                    └── write-through to graph store (optional)
```

The `EnergyStore` uses `DashMap` for lock-free concurrent access with optional LRU eviction for memory-constrained environments.
