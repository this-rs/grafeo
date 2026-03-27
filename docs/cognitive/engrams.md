---
title: Engrams — Living Memory Traces
description: Engrams are consolidated memory traces formed from repeated co-activation patterns. They support Hopfield recall, FSRS decay scheduling, and crystallization.
tags:
  - cognitive
  - engrams
  - memory
---

# Engrams — Living Memory Traces

Engrams are the core memory unit of Obrain's cognitive layer. Unlike raw nodes or edges, an engram represents **consolidated knowledge** — a group of nodes that consistently appear together across episodes, weighted by prediction error (surprise).

## Formation — How Engrams Are Born

Engrams form through **Hebbian co-activation with surprise**:

1. **Episode recording** — Each query or mutation batch is an "episode" that activates a set of nodes.
2. **Co-activation detection** — A `CoActivationDetector` tracks which node pairs appear together across episodes.
3. **Threshold crossing** — When a pair exceeds `min_episodes` co-occurrences with sufficient overlap (`min_overlap`) and cumulative prediction error (`min_prediction_error`), an engram candidate is created.
4. **Ensemble formation** — Connected co-activated pairs are merged into ensembles (up to `max_ensemble_size` nodes).

```rust
FormationConfig {
    min_episodes: 3,          // minimum co-occurrence episodes
    min_overlap: 0.5,         // Jaccard-like overlap ratio
    min_prediction_error: 0.3, // surprise threshold
    max_ensemble_size: 20,     // max nodes per engram
}
```

### Formation Triggers

Three trigger modes control when an engram forms:

| Trigger | Description |
|---------|-------------|
| `CoActivation` | Pure frequency — nodes that co-occur often enough |
| `PredictionError` | Surprise-driven — prediction error exceeds threshold |
| `HebbianWithSurprise` | Combined — both co-occurrence *and* surprise required |

## Hopfield Recall — Content-Addressable Memory

Engrams are stored as **spectral signatures** in a Modern Hopfield Network, enabling content-addressable retrieval with exponential capacity.

### The Retrieve Operation

$$
\text{retrieve}(\mathbf{q}) = \text{softmax}(\beta \cdot \mathbf{P}^\top \cdot \mathbf{q}) \cdot \mathbf{P}
$$

Where:

- $\mathbf{P}$ is the pattern matrix ($N_{\text{engrams}} \times d_{\text{signature}}$), each row is a spectral signature
- $\beta$ is per-engram precision (from `engram.precision`)
- The softmax produces attention weights over all stored patterns

### Capacity

Modern Hopfield networks have **exponential** storage capacity ($2^{d/2}$) compared to classical Hopfield's $0.14N$, making them suitable for large engram stores.

### Softmax Competition

When multiple engrams match a query, **softmax competition** selects the winners:

```rust
let results: Vec<HopfieldResult> = hopfield_retrieve(&query, &pattern_matrix);
// Each result has:
//   .engram_id       — the matching engram
//   .attention_weight — softmax attention ∈ [0, 1], sums to 1
//   .engram          — the full Engram struct
```

### Maximal Marginal Relevance (MMR)

To avoid redundancy, recall results are diversified via MMR:

$$
\text{MMR} = \arg\max_{d_i \in R \setminus S} \left[ \lambda \cdot \text{sim}(d_i, q) - (1 - \lambda) \cdot \max_{d_j \in S} \text{sim}(d_i, d_j) \right]
$$

## FSRS Decay — Spaced Repetition for Graphs

Each engram carries an `FsrsState` governed by the **FSRS-5** algorithm (Free Spaced Repetition Scheduler):

```rust
FsrsState {
    stability: f64,      // days until recall probability = 90%
    difficulty: f64,      // item difficulty ∈ [1, 10]
    reps: u32,            // successful recall count
    lapses: u32,          // failure count
    last_review: DateTime, // last review timestamp
}
```

### The FSRS-5 Model

FSRS uses **19 parameters** that control:

| Parameters | What they control |
|------------|-------------------|
| `w[0..4]` | Initial stability for Again/Hard/Good/Easy on first review |
| `w[4..6]` | Difficulty initialization |
| `w[6]` | Difficulty update rate |
| `w[7]` | Grade factor for stability increase |
| `w[8..11]` | Stability after successful recall |
| `w[11..14]` | Stability after lapse (failure) |
| `w[14..17]` | Short-term stability parameters |
| `w[17..19]` | Calibration parameters |

### Review Ratings

When an engram is recalled (successfully or not), it receives a rating:

| Rating | Meaning |
|--------|---------|
| **Again** | Complete failure — memory has lapsed |
| **Hard** | Recalled with significant difficulty |
| **Good** | Recalled with moderate effort (standard pass) |
| **Easy** | Recalled effortlessly |

The rating updates `stability` and `difficulty`, determining the next review interval.

## Crystallization — From Engram to Persistent Knowledge

When an engram becomes **highly stable** (high retention, many successful recalls), it can be **crystallized** — converted into a permanent node in the graph:

```rust
CrystallizationConfig {
    min_stability: 30.0,   // days
    min_reps: 5,
    min_recall_rate: 0.9,
}
```

Crystallized engrams become `:CrystallizedNote` nodes connected via `:CRYSTALLIZED_IN` edges to the original ensemble members.

## Procedures — Inspecting and Managing Engrams

Obrain exposes engram operations as callable procedures:

### `obrain.engrams.list()`

Lists all engrams with key metrics:

| Column | Description |
|--------|-------------|
| `id` | Engram identifier |
| `strength` | Current strength (combined energy + stability) |
| `valence` | Emotional valence / importance marker |
| `precision` | Hopfield β — higher = sharper recall |
| `recall_count` | Number of successful recalls |
| `horizon` | Memory horizon (working / short-term / long-term) |
| `ensemble_size` | Number of nodes in the engram ensemble |

### `obrain.engrams.inspect(id)`

Returns detailed information about a single engram: ensemble members, spectral signature, FSRS state, formation trigger, and recall history.

### `obrain.engrams.forget(id)`

Removes an engram — implements the **right to erasure** (RGPD *droit à l'oubli*). Deletes the engram, its spectral signature, and all associated recall events.

### `obrain.cognitive.metrics()`

Returns a full snapshot of the cognitive system: total engrams, active energy levels, synapse count, homeostasis signals, and subsystem health.

## Engram Horizons

Engrams are classified by their memory horizon, analogous to human memory systems:

| Horizon | Description | Typical Stability |
|---------|-------------|-------------------|
| **Working** | Very recent, volatile | < 1 day |
| **Short-term** | Recent, decaying | 1–7 days |
| **Long-term** | Consolidated, stable | > 7 days |
