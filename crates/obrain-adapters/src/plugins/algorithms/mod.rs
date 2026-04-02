//! Classic graph algorithms - traversals, paths, centrality, communities.
//!
//! Everything you'd expect from a graph analytics library, designed to work
//! seamlessly with Obrain's LPG store. All algorithms are available from Python too.
//!
//! | Category | Algorithms |
//! | -------- | ---------- |
//! | Traversal | BFS, DFS with visitor pattern |
//! | Components | Connected, strongly connected, topological sort |
//! | Shortest paths | Dijkstra, A*, Bellman-Ford, Floyd-Warshall |
//! | Centrality | PageRank, betweenness, closeness, degree |
//! | Community | Louvain, Leiden, label propagation |
//! | Structure | K-core, bridges, articulation points |
//! | Embedding | Spectral embedding, Hilbert curve encoding, Hilbert 64d features |
//!
//! ## Usage
//!
//! ```no_run
//! use obrain_adapters::plugins::algorithms::{bfs, connected_components, dijkstra};
//! use obrain_core::graph::lpg::LpgStore;
//! use obrain_common::types::NodeId;
//!
//! let store = LpgStore::new().unwrap();
//! let n0 = store.create_node(&["Node"]);
//! let n1 = store.create_node(&["Node"]);
//! store.create_edge(n0, n1, "CONNECTS");
//!
//! // Run BFS from the first node
//! let visited = bfs(&store, n0);
//!
//! // Find connected components
//! let components = connected_components(&store);
//!
//! // Run Dijkstra's shortest path
//! let result = dijkstra(&store, n0, Some("weight"));
//! ```

mod centrality;
mod clustering;
mod community;
mod components;
pub mod contraction;
mod ego_graph;
mod flow;
pub mod hilbert;
pub mod hilbert_features;
pub mod hilbert_manager;
mod mst;
pub mod projection;
pub mod relevance;
mod shortest_path;
mod similarity;
pub mod spectral;
pub mod stable_communities;
mod structure;
mod traits;
mod traversal;

// Core traits
pub use traits::{
    Control, DistanceMap, GraphAlgorithm, MinScored, ParallelGraphAlgorithm, TraversalEvent,
};

// Traversal algorithms
pub use traversal::{bfs, bfs_layers, bfs_with_visitor, dfs, dfs_all, dfs_with_visitor};

// Component algorithms
pub use components::{
    UnionFind, connected_component_count, connected_components, is_dag,
    strongly_connected_component_count, strongly_connected_components, topological_sort,
};

// Shortest path algorithms
pub use shortest_path::{
    BellmanFordResult, DijkstraResult, FloydWarshallResult, astar, bellman_ford, dijkstra,
    dijkstra_path, floyd_warshall,
};

// Centrality algorithms
pub use centrality::{
    DegreeCentralityResult, HitsResult, betweenness_centrality, closeness_centrality,
    degree_centrality, degree_centrality_normalized, hits, pagerank,
};

// Clustering algorithms
#[cfg(feature = "parallel")]
pub use clustering::clustering_coefficient_parallel;
pub use clustering::{
    ClusteringCoefficientResult, clustering_coefficient, global_clustering_coefficient,
    hilbert_bank_allocation, local_clustering_coefficient, total_triangles, triangle_count,
};

// Community detection algorithms
pub use community::{LouvainResult, community_count, label_propagation, leiden, louvain};

// Minimum Spanning Tree algorithms
pub use mst::{MstResult, kruskal, prim};

// Network Flow algorithms
pub use flow::{MaxFlowResult, MinCostFlowResult, max_flow, min_cost_max_flow};

// Structure analysis algorithms
pub use structure::{KCoreResult, articulation_points, bridges, k_core, kcore_decomposition};

// Algorithm wrappers (for future registry integration)
pub use centrality::{
    BetweennessCentralityAlgorithm, ClosenessCentralityAlgorithm, DegreeCentralityAlgorithm,
    HitsAlgorithm, PageRankAlgorithm,
};
pub use clustering::ClusteringCoefficientAlgorithm;
pub use community::{LabelPropagationAlgorithm, LeidenAlgorithm, LouvainAlgorithm};
pub use components::{
    ConnectedComponentsAlgorithm, StronglyConnectedComponentsAlgorithm, TopologicalSortAlgorithm,
};
pub use ego_graph::{EgoEdge, EgoGraph, KHopAlgorithm, KHopConfig, khop_subgraph};
pub use flow::{MaxFlowAlgorithm, MinCostFlowAlgorithm};
pub use mst::{KruskalAlgorithm, PrimAlgorithm};
pub use shortest_path::{
    BellmanFordAlgorithm, DijkstraAlgorithm, FloydWarshallAlgorithm, SsspAlgorithm,
};
pub use structure::{ArticulationPointsAlgorithm, BridgesAlgorithm, KCoreAlgorithm};
pub use traversal::{BfsAlgorithm, DfsAlgorithm};

// Hilbert curve encoding
pub use hilbert::{hilbert_d2xy, hilbert_encode_point, hilbert_xy2d};

// Spectral embedding algorithms
pub use spectral::{SpectralEmbeddingAlgorithm, SpectralEmbeddingResult, spectral_embedding};

// Hilbert 64d multi-facette features
pub use hilbert_features::{
    FacetteWeights, HilbertFeaturesAlgorithm, HilbertFeaturesConfig, HilbertFeaturesResult,
    hilbert_distance, hilbert_features, hilbert_features_incremental, weighted_hilbert_distance,
};

// Hilbert incremental feature manager
pub use hilbert_manager::HilbertFeatureManager;

// Node similarity algorithms
pub use similarity::{
    NodeSimilarityAlgorithm, SimilarityMetric, SimilarityScore, TopKSimilarAlgorithm, adamic_adar,
    cosine_similarity, jaccard, overlap_coefficient, resource_allocation, top_k_similar,
};

// Personalized PageRank & relevance subgraph
pub use relevance::{
    PersonalizedPageRankAlgorithm, PprConfig, PprResult, extract_subgraph, personalized_pagerank,
};

// Stable community detection
pub use stable_communities::{
    StableCommunitiesAlgorithm, StableCommunityConfig, StableLouvainResult, stabilize_communities,
};

// Subgraph contraction (graph coarsening)
pub use contraction::{
    AggregationStrategy, ContractionConfig, ContractionResult, ContractionSnapshot,
    SubgraphContractionAlgorithm, contract_by_communities, contract_subgraph, expand_supernode,
};

// Graph projections (virtual filtered views)
pub use projection::{
    EdgeFilter, GraphProjection, NodeFilter, ProjectionBuilder, ProjectionConfig,
    ProjectionRegistry, PropertyPredicate,
};
