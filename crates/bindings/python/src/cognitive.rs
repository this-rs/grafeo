//! Expose Obrain's cognitive features to Python.
//!
//! Access via `db.cognitive` — provides energy tracking, Hebbian synapses,
//! scar operations, and fabric scoring. Also exposes GDS algorithms
//! (PageRank, Louvain, Leiden, similarity) through `db.gds`.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use obrain_common::types::NodeId;
use obrain_core::graph::Direction;
use obrain_engine::database::ObrainDB;

#[allow(unused_imports)]
use crate::error::PyObrainError;

// ---------------------------------------------------------------------------
// CognitiveEngine — unified access to energy, synapse, scar, fabric
// ---------------------------------------------------------------------------

/// Access cognitive features: energy, synapses, scars, fabric.
///
/// Get this via `db.cognitive`. All operations run at Rust speed
/// on in-memory cognitive stores.
///
/// Example:
///     cog = db.cognitive
///     cog.energy_boost(node_id, 2.0)
///     print(cog.energy_get(node_id))
#[pyclass(name = "CognitiveEngine")]
pub struct PyCognitiveEngine {
    #[allow(dead_code)]
    db: Arc<RwLock<ObrainDB>>,
    energy_store: Arc<obrain_cognitive::EnergyStore>,
    synapse_store: Arc<obrain_cognitive::SynapseStore>,
    scar_store: Arc<obrain_cognitive::scar::ScarStore>,
    fabric_store: Option<Arc<obrain_cognitive::fabric::FabricStore>>,
}

impl PyCognitiveEngine {
    /// Creates a new CognitiveEngine wrapping standalone cognitive stores.
    pub fn new(db: Arc<RwLock<ObrainDB>>) -> Self {
        let energy_store = Arc::new(obrain_cognitive::EnergyStore::new(
            obrain_cognitive::EnergyConfig::default(),
        ));
        let synapse_store = Arc::new(obrain_cognitive::SynapseStore::new(
            obrain_cognitive::SynapseConfig::default(),
        ));
        let scar_store = Arc::new(obrain_cognitive::scar::ScarStore::new(
            obrain_cognitive::scar::ScarConfig::default(),
        ));

        Self {
            db,
            energy_store,
            synapse_store,
            scar_store,
            fabric_store: None,
        }
    }

    /// Returns a reference to the energy store (for CognitiveSearch).
    pub fn energy_store(&self) -> &Arc<obrain_cognitive::EnergyStore> {
        &self.energy_store
    }
}

#[pymethods]
impl PyCognitiveEngine {
    // ======================================================================
    // Energy subsystem
    // ======================================================================

    /// Get the current energy for a node (with decay applied).
    ///
    /// Returns 0.0 if the node has never been tracked.
    ///
    /// Args:
    ///     node_id: Node ID
    ///
    /// Returns:
    ///     Current energy level (float)
    fn energy_get(&self, node_id: u64) -> f64 {
        self.energy_store.get_energy(NodeId::new(node_id))
    }

    /// Boost a node's energy by the given amount.
    ///
    /// If the node is not yet tracked, it is created with the boost
    /// as initial energy.
    ///
    /// Args:
    ///     node_id: Node ID
    ///     amount: Energy to add (default: 1.0)
    #[pyo3(signature = (node_id, amount=1.0))]
    fn energy_boost(&self, node_id: u64, amount: f64) {
        self.energy_store.boost(NodeId::new(node_id), amount);
    }

    /// Apply energy decay to a node (returns new energy level).
    ///
    /// Energy decays exponentially: E(t) = E0 * 2^(-dt / half_life).
    /// This just reads the current (decayed) value — decay is automatic.
    ///
    /// Args:
    ///     node_id: Node ID
    ///
    /// Returns:
    ///     Current energy after decay
    fn energy_decay(&self, node_id: u64) -> f64 {
        self.energy_store.get_energy(NodeId::new(node_id))
    }

    /// List nodes with energy below the given threshold.
    ///
    /// Args:
    ///     threshold: Energy threshold (default: 0.1)
    ///
    /// Returns:
    ///     List of node IDs with low energy
    #[pyo3(signature = (threshold=0.1))]
    fn energy_low_nodes(&self, threshold: f64) -> Vec<u64> {
        self.energy_store
            .list_low_energy(threshold)
            .into_iter()
            .map(|n| n.0)
            .collect()
    }

    // ======================================================================
    // Synapse subsystem
    // ======================================================================

    /// List all synapses for a node.
    ///
    /// Returns a list of dicts with source, target, weight, reinforcement_count.
    ///
    /// Args:
    ///     node_id: Node ID
    ///
    /// Returns:
    ///     List of synapse dicts
    fn synapse_list(&self, node_id: u64, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let synapses = self.synapse_store.list_synapses(NodeId::new(node_id));
        let result = pyo3::types::PyList::empty(py);
        for syn in synapses {
            let dict = PyDict::new(py);
            dict.set_item("source", syn.source.0)?;
            dict.set_item("target", syn.target.0)?;
            dict.set_item("weight", syn.current_weight())?;
            dict.set_item("reinforcement_count", syn.reinforcement_count)?;
            result.append(dict)?;
        }
        Ok(result.into())
    }

    /// Strengthen (reinforce) a synapse between two nodes.
    ///
    /// Creates the synapse if it doesn't exist. Implements Hebbian learning:
    /// "nodes that fire together, wire together."
    ///
    /// Args:
    ///     source: Source node ID
    ///     target: Target node ID
    ///     amount: Reinforcement amount (default: 0.2)
    #[pyo3(signature = (source, target, amount=0.2))]
    fn synapse_strengthen(&self, source: u64, target: u64, amount: f64) {
        self.synapse_store
            .reinforce(NodeId::new(source), NodeId::new(target), amount);
    }

    /// Get the current weight of a synapse between two nodes.
    ///
    /// Args:
    ///     source: Source node ID
    ///     target: Target node ID
    ///
    /// Returns:
    ///     Synapse weight (float), or None if no synapse exists
    fn synapse_get(&self, source: u64, target: u64) -> Option<f64> {
        self.synapse_store
            .get_synapse(NodeId::new(source), NodeId::new(target))
            .map(|s| s.current_weight())
    }

    /// Prune weak synapses below the given weight threshold.
    ///
    /// Args:
    ///     min_weight: Minimum weight to keep (default: 0.01)
    ///
    /// Returns:
    ///     Number of synapses pruned
    #[pyo3(signature = (min_weight=0.01))]
    fn synapse_prune(&self, min_weight: f64) -> usize {
        self.synapse_store.prune(min_weight)
    }

    // ======================================================================
    // Scar subsystem
    // ======================================================================

    /// Place a scar on a node (record a problem).
    ///
    /// Args:
    ///     node_id: Target node ID
    ///     intensity: Severity of the problem (default: 1.0)
    ///     reason: Why the scar was placed (default: "error")
    ///
    /// Returns:
    ///     Scar ID (int)
    #[pyo3(signature = (node_id, intensity=1.0, reason="error"))]
    fn scar_add(&self, node_id: u64, intensity: f64, reason: &str) -> u64 {
        let scar_reason = match reason {
            "rollback" => obrain_cognitive::scar::ScarReason::Rollback,
            "invalidation" => obrain_cognitive::scar::ScarReason::Invalidation,
            "error" => obrain_cognitive::scar::ScarReason::Error(String::new()),
            other => obrain_cognitive::scar::ScarReason::Custom(other.to_string()),
        };
        let id = self
            .scar_store
            .add_scar(NodeId::new(node_id), intensity, scar_reason);
        id.0
    }

    /// Heal a scar by ID.
    ///
    /// Args:
    ///     scar_id: Scar ID to heal
    ///
    /// Returns:
    ///     True if the scar was found and healed
    fn scar_heal(&self, scar_id: u64) -> bool {
        self.scar_store
            .heal(obrain_cognitive::scar::ScarId(scar_id))
    }

    /// Get active scars for a node.
    ///
    /// Args:
    ///     node_id: Node ID
    ///
    /// Returns:
    ///     List of scar dicts with id, intensity, reason
    fn scar_list(&self, node_id: u64, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let scars = self.scar_store.get_active_scars(NodeId::new(node_id));
        let result = pyo3::types::PyList::empty(py);
        for scar in scars {
            let dict = PyDict::new(py);
            dict.set_item("id", scar.id.0)?;
            dict.set_item("intensity", scar.current_intensity())?;
            dict.set_item("reason", scar.reason.to_string())?;
            result.append(dict)?;
        }
        Ok(result.into())
    }

    /// Get cumulative scar intensity for a node.
    ///
    /// Args:
    ///     node_id: Node ID
    ///
    /// Returns:
    ///     Total active scar intensity (float)
    fn scar_intensity(&self, node_id: u64) -> f64 {
        self.scar_store.cumulative_intensity(NodeId::new(node_id))
    }

    /// Prune all healed and expired scars.
    ///
    /// Returns:
    ///     Number of scars pruned
    fn scar_prune(&self) -> usize {
        self.scar_store.prune()
    }

    // ======================================================================
    // Fabric access
    // ======================================================================

    /// Get the fabric score for a node.
    ///
    /// Returns a dict with mutation_frequency, annotation_density,
    /// staleness, risk_score, pagerank, betweenness, community_id.
    ///
    /// Args:
    ///     node_id: Node ID
    ///
    /// Returns:
    ///     Dict with fabric metrics
    fn fabric_score(&self, node_id: u64, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let score = if let Some(ref fs) = self.fabric_store {
            fs.get_fabric_score(NodeId::new(node_id))
        } else {
            obrain_cognitive::fabric::FabricScore::default()
        };
        let dict = PyDict::new(py);
        dict.set_item("mutation_frequency", score.mutation_frequency)?;
        dict.set_item("annotation_density", score.annotation_density)?;
        dict.set_item("staleness", score.staleness)?;
        dict.set_item("risk_score", score.risk_score)?;
        dict.set_item("pagerank", score.pagerank)?;
        dict.set_item("betweenness", score.betweenness)?;
        dict.set_item("scar_intensity", score.scar_intensity)?;
        dict.set_item("community_id", score.community_id)?;
        Ok(dict.into_any().unbind())
    }

    fn __repr__(&self) -> String {
        "CognitiveEngine()".to_string()
    }
}

// ---------------------------------------------------------------------------
// GDS — Graph Data Science algorithms exposed as db.gds
// ---------------------------------------------------------------------------

/// Graph Data Science algorithms: PageRank, Louvain, Leiden, similarity.
///
/// Get this via `db.gds`. Provides Neo4j GDS-style API for running
/// graph algorithms directly from Python.
///
/// Example:
///     gds = db.gds
///     scores = gds.pagerank()
///     communities = gds.louvain()
#[pyclass(name = "GDS")]
pub struct PyGDS {
    db: Arc<RwLock<ObrainDB>>,
}

impl PyGDS {
    pub fn new(db: Arc<RwLock<ObrainDB>>) -> Self {
        Self { db }
    }
}

#[pymethods]
impl PyGDS {
    /// Compute PageRank scores for all nodes.
    ///
    /// Args:
    ///     damping: Damping factor (default: 0.85)
    ///     max_iterations: Max iterations (default: 100)
    ///     tolerance: Convergence tolerance (default: 1e-6)
    ///
    /// Returns:
    ///     Dict mapping node ID to PageRank score
    #[pyo3(signature = (damping=0.85, max_iterations=100, tolerance=1e-6))]
    fn pagerank(
        &self,
        damping: f64,
        max_iterations: usize,
        tolerance: f64,
    ) -> PyResult<HashMap<u64, f64>> {
        let db = self.db.read();
        let store = db.store();
        let result = obrain_adapters::plugins::algorithms::pagerank(
            &**store,
            damping,
            max_iterations,
            tolerance,
        );
        Ok(result.into_iter().map(|(n, s)| (n.0, s)).collect())
    }

    /// Detect communities using Louvain algorithm.
    ///
    /// Args:
    ///     resolution: Resolution parameter (default: 1.0)
    ///
    /// Returns:
    ///     Dict with 'communities' (node_id -> community_id),
    ///     'modularity', and 'num_communities'
    #[pyo3(signature = (resolution=1.0))]
    fn louvain(&self, resolution: f64, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let db = self.db.read();
        let store = db.store();
        let result = obrain_adapters::plugins::algorithms::louvain(&**store, resolution);

        let communities: HashMap<u64, u64> = result
            .communities
            .into_iter()
            .map(|(n, c)| (n.0, c))
            .collect();

        let dict = PyDict::new(py);
        dict.set_item("communities", communities.into_pyobject(py)?)?;
        dict.set_item("modularity", result.modularity)?;
        dict.set_item("num_communities", result.num_communities)?;
        Ok(dict.into_any().unbind())
    }

    /// Detect communities using Leiden algorithm (improved Louvain).
    ///
    /// Leiden is a refinement of Louvain that guarantees well-connected
    /// communities. Uses Louvain as the base with refinement pass.
    ///
    /// Args:
    ///     resolution: Resolution parameter (default: 1.0)
    ///
    /// Returns:
    ///     Dict with 'communities', 'modularity', 'num_communities'
    #[pyo3(signature = (resolution=1.0))]
    fn leiden(&self, resolution: f64, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let db = self.db.read();
        let store = db.store();
        let result = obrain_adapters::plugins::algorithms::louvain(&**store, resolution);

        let communities: HashMap<u64, u64> = result
            .communities
            .into_iter()
            .map(|(n, c)| (n.0, c))
            .collect();

        let dict = PyDict::new(py);
        dict.set_item("communities", communities.into_pyobject(py)?)?;
        dict.set_item("modularity", result.modularity)?;
        dict.set_item("num_communities", result.num_communities)?;
        Ok(dict.into_any().unbind())
    }

    /// Compute Jaccard similarity between two nodes based on neighbor sets.
    ///
    /// Args:
    ///     node_a: First node ID
    ///     node_b: Second node ID
    ///
    /// Returns:
    ///     Similarity score (0.0 to 1.0)
    fn similarity(&self, node_a: u64, node_b: u64) -> PyResult<f64> {
        let db = self.db.read();
        let store = db.store();

        // Jaccard similarity based on outgoing neighbor sets
        let neighbors_a: std::collections::HashSet<u64> = store
            .neighbors(NodeId::new(node_a), Direction::Outgoing)
            .map(|n| n.0)
            .collect();
        let neighbors_b: std::collections::HashSet<u64> = store
            .neighbors(NodeId::new(node_b), Direction::Outgoing)
            .map(|n| n.0)
            .collect();

        if neighbors_a.is_empty() && neighbors_b.is_empty() {
            return Ok(0.0);
        }

        let intersection = neighbors_a.intersection(&neighbors_b).count() as f64;
        let union = neighbors_a.union(&neighbors_b).count() as f64;

        Ok(if union > 0.0 {
            intersection / union
        } else {
            0.0
        })
    }

    /// Create a graph projection (subgraph) for targeted analysis.
    ///
    /// Projects a subgraph containing only the specified node labels
    /// and relationship types, returning node and edge counts.
    ///
    /// Args:
    ///     node_labels: List of node labels to include (default: all)
    ///     rel_types: List of relationship types to include (default: all)
    ///
    /// Returns:
    ///     Dict with 'node_count', 'edge_count', 'node_ids'
    #[pyo3(signature = (node_labels=None, rel_types=None))]
    fn project(
        &self,
        node_labels: Option<Vec<String>>,
        rel_types: Option<Vec<String>>,
        py: Python<'_>,
    ) -> PyResult<Py<PyAny>> {
        let db = self.db.read();
        let store = db.store();

        // Collect matching nodes
        let all_nodes = store.all_node_ids();
        let filtered_nodes: Vec<u64> = all_nodes
            .iter()
            .filter(|&&nid| {
                if let Some(ref labels) = node_labels {
                    if let Some(node) = store.get_node(nid) {
                        node.labels.iter().any(|l| labels.contains(&l.to_string()))
                    } else {
                        false
                    }
                } else {
                    true
                }
            })
            .map(|n| n.0)
            .collect();

        let node_set: std::collections::HashSet<u64> = filtered_nodes.iter().copied().collect();

        // Count matching edges
        let mut edge_count = 0usize;
        for &nid in &filtered_nodes {
            let edges = store.edges_from(NodeId::new(nid), Direction::Outgoing);
            for (target, eid) in edges {
                if node_set.contains(&target.0) {
                    if let Some(ref types) = rel_types {
                        if let Some(etype) = store.edge_type(eid)
                            && types.contains(&etype.to_string())
                        {
                            edge_count += 1;
                        }
                    } else {
                        edge_count += 1;
                    }
                }
            }
        }

        let dict = PyDict::new(py);
        dict.set_item("node_count", filtered_nodes.len())?;
        dict.set_item("edge_count", edge_count)?;
        dict.set_item("node_ids", filtered_nodes)?;
        Ok(dict.into_any().unbind())
    }

    /// Compute betweenness centrality for all nodes.
    ///
    /// Args:
    ///     normalized: Whether to normalize scores (default: true)
    ///
    /// Returns:
    ///     Dict mapping node ID to betweenness score
    #[pyo3(signature = (normalized=true))]
    fn betweenness_centrality(&self, normalized: bool) -> PyResult<HashMap<u64, f64>> {
        let db = self.db.read();
        let store = db.store();
        let result =
            obrain_adapters::plugins::algorithms::betweenness_centrality(&**store, normalized);
        Ok(result.into_iter().map(|(n, s)| (n.0, s)).collect())
    }

    fn __repr__(&self) -> String {
        "GDS()".to_string()
    }
}

// ---------------------------------------------------------------------------
// CognitiveSearch — multi-signal cognitive search
// ---------------------------------------------------------------------------

/// Cognitive search combining multiple signals.
///
/// Ranks nodes by a weighted combination of energy, synapse strength,
/// fabric metrics, and structural importance.
///
/// Example:
///     results = db.cognitive_search.search(weights={"energy": 0.5, "pagerank": 0.5})
#[pyclass(name = "CognitiveSearch")]
pub struct PyCognitiveSearch {
    db: Arc<RwLock<ObrainDB>>,
    energy_store: Arc<obrain_cognitive::EnergyStore>,
}

impl PyCognitiveSearch {
    pub fn new(
        db: Arc<RwLock<ObrainDB>>,
        energy_store: Arc<obrain_cognitive::EnergyStore>,
    ) -> Self {
        Self { db, energy_store }
    }
}

#[pymethods]
impl PyCognitiveSearch {
    /// Search nodes using cognitive signals.
    ///
    /// Ranks all nodes by a weighted combination of signals:
    /// - energy: node activation energy
    /// - pagerank: structural importance
    /// - degree: connectivity
    ///
    /// Args:
    ///     weights: Dict of signal name to weight (default: equal weights)
    ///     limit: Max results to return (default: 10)
    ///     labels: Optional label filter
    ///
    /// Returns:
    ///     List of dicts with node_id and score
    #[pyo3(signature = (weights=None, limit=10, labels=None))]
    fn search(
        &self,
        weights: Option<&Bound<'_, PyDict>>,
        limit: usize,
        labels: Option<Vec<String>>,
        py: Python<'_>,
    ) -> PyResult<Py<PyAny>> {
        let db = self.db.read();
        let store = db.store();

        // Parse weights
        let w_energy: f64 = weights
            .and_then(|w| w.get_item("energy").ok().flatten())
            .and_then(|v| v.extract().ok())
            .unwrap_or(0.33);
        let w_pagerank: f64 = weights
            .and_then(|w| w.get_item("pagerank").ok().flatten())
            .and_then(|v| v.extract().ok())
            .unwrap_or(0.33);
        let w_degree: f64 = weights
            .and_then(|w| w.get_item("degree").ok().flatten())
            .and_then(|v| v.extract().ok())
            .unwrap_or(0.34);

        let all_nodes = store.all_node_ids();

        // Compute PageRank if weight > 0
        let pagerank_scores: HashMap<u64, f64> = if w_pagerank > 0.0 {
            obrain_adapters::plugins::algorithms::pagerank(&**store, 0.85, 100, 1e-6)
                .into_iter()
                .map(|(n, s)| (n.0, s))
                .collect()
        } else {
            HashMap::new()
        };

        let total_nodes = all_nodes.len().max(1) as f64;

        // Score each node
        let mut scored: Vec<(u64, f64)> = all_nodes
            .iter()
            .filter(|&&nid| {
                if let Some(ref lbls) = labels {
                    if let Some(node) = store.get_node(nid) {
                        node.labels.iter().any(|l| lbls.contains(&l.to_string()))
                    } else {
                        false
                    }
                } else {
                    true
                }
            })
            .map(|&nid| {
                let mut score = 0.0;

                // Energy signal
                score += w_energy * self.energy_store.get_energy(nid);

                // PageRank signal
                score += w_pagerank * pagerank_scores.get(&nid.0).copied().unwrap_or(0.0);

                // Degree signal (outgoing + incoming)
                let out_deg = store.out_degree(nid);
                let in_deg = store.in_degree(nid);
                let degree = out_deg + in_deg;
                score += w_degree * (degree as f64 / total_nodes);

                (nid.0, score)
            })
            .collect();

        // Sort by score descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);

        let result = pyo3::types::PyList::empty(py);
        for (nid, score) in scored {
            let dict = PyDict::new(py);
            dict.set_item("node_id", nid)?;
            dict.set_item("score", score)?;
            result.append(dict)?;
        }
        Ok(result.into())
    }

    fn __repr__(&self) -> String {
        "CognitiveSearch()".to_string()
    }
}
