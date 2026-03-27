"""Base class for solvOR plugin comparison tests.

This module defines tests that compare Obrain's solvOR plugin (db.as_solvor())
against the standalone solvOR library to verify correctness.

Tests cover:
- Shortest Path (Dijkstra)
- Maximum Flow
- Minimum Spanning Tree
"""

from abc import ABC, abstractmethod

import pytest

# Try to import standalone solvOR library
try:
    from solvor import dijkstra as solvor_dijkstra
    from solvor import kruskal as solvor_kruskal
    from solvor import max_flow as solvor_max_flow

    SOLVOR_AVAILABLE = True
except ImportError:
    SOLVOR_AVAILABLE = False


class BaseSolvORComparisonTest(ABC):
    """Abstract base class for solvOR plugin comparison tests.

    Subclasses implement graph construction using their query language,
    then the tests compare Obrain's as_solvor() plugin results against
    the standalone solvOR library.
    """

    @abstractmethod
    def create_db(self):
        """Create a fresh database instance."""
        raise NotImplementedError

    @abstractmethod
    def setup_flow_network(self, db, n_nodes: int, n_edges: int, seed: int = 42) -> dict:
        """Set up a flow network graph using the query language.

        Args:
            db: Database instance
            n_nodes: Number of nodes
            n_edges: Number of edges
            seed: Random seed

        Returns:
            dict with:
                'node_ids': list of node IDs
                'source': source node ID
                'sink': sink node ID
                'edges': list of (src, dst, capacity, cost) tuples
        """
        raise NotImplementedError

    # ===== Shortest Path Tests =====

    @pytest.mark.skipif(not SOLVOR_AVAILABLE, reason="solvOR not installed")
    def test_shortest_path_vs_solvor(self, db):
        """Obrain's as_solvor().shortest_path() should match standalone solvOR."""
        graph_info = self.setup_flow_network(db, 50, 150, seed=42)
        node_ids = graph_info["node_ids"]
        edges = graph_info["edges"]
        source = graph_info["source"]
        sink = graph_info["sink"]

        # Obrain solvOR plugin
        solvor_adapter = db.as_solvor()
        obrain_result = solvor_adapter.shortest_path(source, sink, weight="cost")

        # Standalone solvOR dijkstra
        # Build adjacency list: {node_idx: [(neighbor_idx, cost), ...]}
        node_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        adj: dict[int, list[tuple[int, float]]] = {i: [] for i in range(len(node_ids))}
        for src, dst, _capacity, cost in edges:
            i, j = node_to_idx[src], node_to_idx[dst]
            adj[i].append((j, cost))

        # solvOR dijkstra uses: dijkstra(start, goal, neighbors_func)
        def neighbors(node: int):
            return adj.get(node, [])

        solvor_result = solvor_dijkstra(node_to_idx[source], node_to_idx[sink], neighbors)

        if obrain_result is not None and solvor_result.status.name == "OPTIMAL":
            obrain_dist, obrain_path = obrain_result
            solvor_dist = solvor_result.objective
            assert abs(obrain_dist - solvor_dist) < 1e-6, (
                f"Shortest path mismatch: Obrain={obrain_dist}, solvOR={solvor_dist}"
            )

    # ===== Max Flow Tests =====

    @pytest.mark.skipif(not SOLVOR_AVAILABLE, reason="solvOR not installed")
    def test_max_flow_vs_solvor(self, db):
        """Obrain's as_solvor().max_flow() should match standalone solvOR."""
        graph_info = self.setup_flow_network(db, 20, 60, seed=42)
        node_ids = graph_info["node_ids"]
        edges = graph_info["edges"]
        source = graph_info["source"]
        sink = graph_info["sink"]

        # Obrain solvOR plugin
        solvor_adapter = db.as_solvor()
        obrain_result = solvor_adapter.max_flow(source, sink, capacity="capacity")

        # Standalone solvOR max_flow
        # Build graph: {node: [(neighbor, capacity), ...]}
        node_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        graph: dict[int, list[tuple[int, int]]] = {i: [] for i in range(len(node_ids))}
        for src, dst, capacity, _ in edges:
            i, j = node_to_idx[src], node_to_idx[dst]
            graph[i].append((j, capacity))

        solvor_result = solvor_max_flow(graph, node_to_idx[source], node_to_idx[sink])

        if obrain_result is not None and solvor_result.status.name == "OPTIMAL":
            obrain_flow = obrain_result.get("max_flow")
            solvor_flow = solvor_result.objective
            assert obrain_flow == solvor_flow, (
                f"Max flow mismatch: Obrain={obrain_flow}, solvOR={solvor_flow}"
            )

    # ===== Minimum Spanning Tree Tests =====

    @pytest.mark.skipif(not SOLVOR_AVAILABLE, reason="solvOR not installed")
    def test_mst_vs_solvor(self, db):
        """Obrain's as_solvor().minimum_spanning_tree() should match solvOR."""
        graph_info = self.setup_flow_network(db, 30, 100, seed=42)
        node_ids = graph_info["node_ids"]
        edges = graph_info["edges"]

        # Obrain solvOR plugin
        solvor_adapter = db.as_solvor()
        obrain_result = solvor_adapter.minimum_spanning_tree(weight="cost")

        # Standalone solvOR kruskal
        # Build edge list: [(u, v, weight), ...]
        # MST algorithms work on undirected graphs, so for each node pair
        # keep only the minimum cost edge (there may be edges in both directions)
        node_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        edge_costs: dict[tuple[int, int], int] = {}
        for src, dst, _, cost in edges:
            i, j = node_to_idx[src], node_to_idx[dst]
            edge_key = (min(i, j), max(i, j))
            if edge_key not in edge_costs or cost < edge_costs[edge_key]:
                edge_costs[edge_key] = cost
        edge_list = [(k[0], k[1], v) for k, v in edge_costs.items()]

        solvor_result = solvor_kruskal(len(node_ids), edge_list, allow_forest=True)

        if obrain_result is not None and solvor_result.status.name in (
            "OPTIMAL",
            "FEASIBLE",
        ):
            obrain_weight = obrain_result.get("total_weight", 0)
            solvor_weight = solvor_result.objective
            # MST weights should match (allowing small float error)
            assert abs(obrain_weight - solvor_weight) < 1e-6, (
                f"MST weight mismatch: Obrain={obrain_weight}, solvOR={solvor_weight}"
            )


class BaseSolvORBenchmarkTest(ABC):
    """Abstract base class for solvOR plugin performance comparison.

    Compares Obrain's as_solvor() plugin performance against standalone solvOR.
    """

    @abstractmethod
    def create_db(self):
        """Create a fresh database instance."""
        raise NotImplementedError

    @abstractmethod
    def setup_flow_network(self, db, n_nodes: int, n_edges: int, seed: int = 42) -> dict:
        """Set up a flow network graph using the query language."""
        raise NotImplementedError

    @pytest.mark.skipif(not SOLVOR_AVAILABLE, reason="solvOR not installed")
    def test_max_flow_performance(self, db):
        """Compare max flow performance: Obrain plugin vs standalone solvOR."""
        import time

        graph_info = self.setup_flow_network(db, 100, 500, seed=42)
        node_ids = graph_info["node_ids"]
        edges = graph_info["edges"]
        source = graph_info["source"]
        sink = graph_info["sink"]

        # Obrain plugin timing
        solvor_adapter = db.as_solvor()
        start = time.perf_counter()
        solvor_adapter.max_flow(source, sink, capacity="capacity")
        obrain_time = (time.perf_counter() - start) * 1000

        # Standalone solvOR timing
        node_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        graph: dict[int, list[tuple[int, int]]] = {i: [] for i in range(len(node_ids))}
        for src, dst, capacity, _ in edges:
            i, j = node_to_idx[src], node_to_idx[dst]
            graph[i].append((j, capacity))

        start = time.perf_counter()
        solvor_max_flow(graph, node_to_idx[source], node_to_idx[sink])
        solvor_time = (time.perf_counter() - start) * 1000

        print("\nMax Flow (100 nodes, 500 edges):")
        print(f"  Obrain plugin: {obrain_time:.2f}ms")
        print(f"  Standalone solvOR: {solvor_time:.2f}ms")
        if solvor_time > 0:
            print(f"  Ratio: {obrain_time / solvor_time:.2f}x")

        assert obrain_time >= 0
        assert solvor_time >= 0

    @pytest.mark.skipif(not SOLVOR_AVAILABLE, reason="solvOR not installed")
    def test_shortest_path_performance(self, db):
        """Compare shortest path performance: Obrain plugin vs standalone solvOR."""
        import time

        graph_info = self.setup_flow_network(db, 500, 2000, seed=42)
        node_ids = graph_info["node_ids"]
        edges = graph_info["edges"]
        source = graph_info["source"]
        sink = graph_info["sink"]

        # Obrain plugin timing
        solvor_adapter = db.as_solvor()
        start = time.perf_counter()
        solvor_adapter.shortest_path(source, sink, weight="cost")
        obrain_time = (time.perf_counter() - start) * 1000

        # Standalone solvOR timing
        node_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        adj: dict[int, list[tuple[int, float]]] = {i: [] for i in range(len(node_ids))}
        for src, dst, _capacity, cost in edges:
            i, j = node_to_idx[src], node_to_idx[dst]
            adj[i].append((j, cost))

        def neighbors(node: int):
            return adj.get(node, [])

        start = time.perf_counter()
        solvor_dijkstra(node_to_idx[source], node_to_idx[sink], neighbors)
        solvor_time = (time.perf_counter() - start) * 1000

        print("\nShortest Path (500 nodes, 2000 edges):")
        print(f"  Obrain plugin: {obrain_time:.2f}ms")
        print(f"  Standalone solvOR: {solvor_time:.2f}ms")
        if solvor_time > 0:
            print(f"  Ratio: {obrain_time / solvor_time:.2f}x")

        assert obrain_time >= 0
        assert solvor_time >= 0
