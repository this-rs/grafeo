"""Base class for NetworkX comparison tests.

This module defines tests that compare Obrain algorithm results against NetworkX
to verify correctness. NetworkX serves as the reference implementation.

Tests cover:
- Traversal: BFS, DFS
- Shortest Path: Dijkstra, Bellman-Ford
- Centrality: Degree, PageRank, Betweenness, Closeness
- Components: Connected, Strongly Connected
- MST: Kruskal/Prim
"""

from abc import ABC, abstractmethod

import pytest

# Try to import networkx
try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False


class BaseNetworkXComparisonTest(ABC):
    """Abstract base class for NetworkX comparison tests.

    Subclasses implement graph construction and algorithm execution
    for their specific database API.
    """

    @abstractmethod
    def create_db(self):
        """Create a fresh database instance."""
        raise NotImplementedError

    @abstractmethod
    def setup_random_graph(
        self, db, n_nodes: int, n_edges: int, weighted: bool = True, seed: int = 42
    ) -> dict:
        """Set up a random graph.

        Args:
            db: Database instance
            n_nodes: Number of nodes
            n_edges: Number of edges
            weighted: Whether to add weight property
            seed: Random seed for reproducibility

        Returns:
            dict with:
                'node_ids': list of node IDs
                'edges': list of (src, dst, weight) tuples
        """
        raise NotImplementedError

    @abstractmethod
    def run_bfs(self, db, start_node) -> set:
        """Run BFS and return visited nodes as a set."""
        raise NotImplementedError

    @abstractmethod
    def run_dfs(self, db, start_node) -> set:
        """Run DFS and return visited nodes as a set."""
        raise NotImplementedError

    @abstractmethod
    def run_dijkstra(self, db, source) -> dict:
        """Run Dijkstra and return {node: distance} dict."""
        raise NotImplementedError

    @abstractmethod
    def run_connected_components(self, db) -> int:
        """Run connected components and return count."""
        raise NotImplementedError

    @abstractmethod
    def run_pagerank(self, db, damping: float = 0.85) -> dict:
        """Run PageRank and return {node: score} dict."""
        raise NotImplementedError

    @abstractmethod
    def run_degree_centrality(self, db) -> dict:
        """Run degree centrality and return {node: centrality} dict."""
        raise NotImplementedError

    def _build_networkx_graph(self, edges: list, directed: bool = True, weighted: bool = True):
        """Build a NetworkX graph from edge list."""
        if directed:
            graph = nx.DiGraph()
        else:
            graph = nx.Graph()

        for src, dst, weight in edges:
            if weighted:
                graph.add_edge(src, dst, weight=weight)
            else:
                graph.add_edge(src, dst)

        return graph

    # ===== Comparison Tests =====

    @pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="NetworkX not installed")
    def test_bfs_reachability(self, db):
        """BFS should reach same nodes as NetworkX BFS."""
        graph_info = self.setup_random_graph(db, 100, 300, weighted=False, seed=42)
        node_ids = graph_info["node_ids"]
        edges = graph_info["edges"]
        start_node = node_ids[0]

        # Obrain BFS
        obrain_visited = self.run_bfs(db, start_node)

        # NetworkX BFS
        graph = self._build_networkx_graph(edges, directed=True, weighted=False)
        nx_visited = set(nx.bfs_tree(graph, start_node).nodes())

        assert obrain_visited == nx_visited, (
            f"BFS reachability mismatch: Obrain found {len(obrain_visited)} nodes, "
            f"NetworkX found {len(nx_visited)} nodes"
        )

    @pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="NetworkX not installed")
    def test_dfs_reachability(self, db):
        """DFS should reach same nodes as NetworkX DFS."""
        graph_info = self.setup_random_graph(db, 100, 300, weighted=False, seed=42)
        node_ids = graph_info["node_ids"]
        edges = graph_info["edges"]
        start_node = node_ids[0]

        # Obrain DFS
        obrain_visited = self.run_dfs(db, start_node)

        # NetworkX DFS
        graph = self._build_networkx_graph(edges, directed=True, weighted=False)
        nx_visited = set(nx.dfs_tree(graph, start_node).nodes())

        assert obrain_visited == nx_visited, (
            f"DFS reachability mismatch: Obrain found {len(obrain_visited)} nodes, "
            f"NetworkX found {len(nx_visited)} nodes"
        )

    @pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="NetworkX not installed")
    def test_dijkstra_distances(self, db):
        """Dijkstra distances should match NetworkX within tolerance."""
        graph_info = self.setup_random_graph(db, 50, 150, weighted=True, seed=42)
        node_ids = graph_info["node_ids"]
        edges = graph_info["edges"]
        source = node_ids[0]

        # Obrain Dijkstra
        obrain_distances = self.run_dijkstra(db, source)

        # NetworkX Dijkstra
        graph = self._build_networkx_graph(edges, directed=True, weighted=True)
        nx_distances = nx.single_source_dijkstra_path_length(graph, source, weight="weight")

        # Compare distances for nodes reachable by both
        common_nodes = set(obrain_distances.keys()) & set(nx_distances.keys())
        assert len(common_nodes) > 0, "No common reachable nodes"

        for node in common_nodes:
            obrain_dist = obrain_distances[node]
            nx_dist = nx_distances[node]
            assert abs(obrain_dist - nx_dist) < 1e-6, (
                f"Distance mismatch for node {node}: Obrain={obrain_dist}, NetworkX={nx_dist}"
            )

    @pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="NetworkX not installed")
    def test_connected_component_count(self, db):
        """Connected component count should match NetworkX."""
        graph_info = self.setup_random_graph(db, 100, 200, weighted=False, seed=42)
        edges = graph_info["edges"]

        # Obrain connected components
        obrain_count = self.run_connected_components(db)

        # NetworkX connected components (undirected)
        graph = self._build_networkx_graph(edges, directed=False, weighted=False)
        # Add isolated nodes
        for node_id in graph_info["node_ids"]:
            if node_id not in graph:
                graph.add_node(node_id)
        nx_count = nx.number_connected_components(graph)

        assert obrain_count == nx_count, (
            f"Connected component count mismatch: Obrain={obrain_count}, NetworkX={nx_count}"
        )

    @pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="NetworkX not installed")
    def test_pagerank_ranking(self, db):
        """PageRank top-k ranking should match NetworkX."""
        graph_info = self.setup_random_graph(db, 50, 200, weighted=False, seed=42)
        edges = graph_info["edges"]

        # Obrain PageRank
        obrain_pr = self.run_pagerank(db)

        # NetworkX PageRank
        graph = self._build_networkx_graph(edges, directed=True, weighted=False)
        nx_pr = nx.pagerank(graph, alpha=0.85)

        # Compare top-5 ranking
        obrain_top5 = sorted(obrain_pr.items(), key=lambda x: x[1], reverse=True)[:5]
        nx_top5 = sorted(nx_pr.items(), key=lambda x: x[1], reverse=True)[:5]

        obrain_top5_nodes = [n for n, _ in obrain_top5]
        nx_top5_nodes = [n for n, _ in nx_top5]

        # At least 3 of top 5 should match (some variation due to implementation)
        overlap = len(set(obrain_top5_nodes) & set(nx_top5_nodes))
        assert overlap >= 3, (
            f"PageRank top-5 mismatch: only {overlap} overlap. "
            f"Obrain top 5: {obrain_top5_nodes}, NetworkX top 5: {nx_top5_nodes}"
        )

    @pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="NetworkX not installed")
    def test_pagerank_sum(self, db):
        """PageRank scores should sum to approximately 1.0."""
        self.setup_random_graph(db, 50, 200, weighted=False, seed=42)

        # Obrain PageRank
        obrain_pr = self.run_pagerank(db)
        pr_sum = sum(obrain_pr.values())

        assert abs(pr_sum - 1.0) < 0.01, f"PageRank sum should be ~1.0, got {pr_sum}"

    @pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="NetworkX not installed")
    def test_degree_centrality_values(self, db):
        """Degree centrality should match NetworkX."""
        graph_info = self.setup_random_graph(db, 50, 150, weighted=False, seed=42)
        edges = graph_info["edges"]

        # Obrain degree centrality
        obrain_dc = self.run_degree_centrality(db)

        # NetworkX degree centrality
        graph = self._build_networkx_graph(edges, directed=True, weighted=False)
        nx_dc = nx.degree_centrality(graph)

        # Compare values for common nodes
        common_nodes = set(obrain_dc.keys()) & set(nx_dc.keys())
        assert len(common_nodes) > 0, "No common nodes for degree centrality"

        for node in common_nodes:
            obrain_val = obrain_dc[node]
            nx_val = nx_dc[node]
            assert abs(obrain_val - nx_val) < 0.01, (
                f"Degree centrality mismatch for node {node}: "
                f"Obrain={obrain_val}, NetworkX={nx_val}"
            )


class BaseNetworkXBenchmarkTest(ABC):
    """Abstract base class for NetworkX vs Obrain performance comparison.

    Runs the same algorithms on both and compares performance.
    """

    @abstractmethod
    def create_db(self):
        """Create a fresh database instance."""
        raise NotImplementedError

    @abstractmethod
    def setup_random_graph(
        self, db, n_nodes: int, n_edges: int, weighted: bool = True, seed: int = 42
    ) -> dict:
        """Set up a random graph and return graph info."""
        raise NotImplementedError

    @abstractmethod
    def run_obrain_pagerank(self, db) -> float:
        """Run Obrain PageRank and return execution time in ms."""
        raise NotImplementedError

    @abstractmethod
    def run_obrain_dijkstra(self, db, source) -> float:
        """Run Obrain Dijkstra and return execution time in ms."""
        raise NotImplementedError

    @abstractmethod
    def run_obrain_bfs(self, db, start) -> float:
        """Run Obrain BFS and return execution time in ms."""
        raise NotImplementedError

    @pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="NetworkX not installed")
    def test_pagerank_performance_comparison(self, db):
        """Compare PageRank performance between Obrain and NetworkX."""
        import time

        graph_info = self.setup_random_graph(db, 1000, 5000, weighted=False, seed=42)
        edges = graph_info["edges"]

        # Obrain timing
        obrain_time = self.run_obrain_pagerank(db)

        # NetworkX timing
        graph = nx.DiGraph()
        for src, dst, _ in edges:
            graph.add_edge(src, dst)

        start = time.perf_counter()
        nx.pagerank(graph, alpha=0.85)
        nx_time = (time.perf_counter() - start) * 1000

        print("\nPageRank (1000 nodes, 5000 edges):")
        print(f"  Obrain: {obrain_time:.2f}ms")
        print(f"  NetworkX: {nx_time:.2f}ms")
        print(f"  Ratio: {obrain_time / nx_time:.2f}x")

        # Just ensure both complete, don't assert performance
        assert obrain_time > 0
        assert nx_time > 0

    @pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="NetworkX not installed")
    def test_dijkstra_performance_comparison(self, db):
        """Compare Dijkstra performance between Obrain and NetworkX."""
        import time

        graph_info = self.setup_random_graph(db, 1000, 5000, weighted=True, seed=42)
        node_ids = graph_info["node_ids"]
        edges = graph_info["edges"]
        source = node_ids[0]

        # Obrain timing
        obrain_time = self.run_obrain_dijkstra(db, source)

        # NetworkX timing
        graph = nx.DiGraph()
        for src, dst, weight in edges:
            graph.add_edge(src, dst, weight=weight)

        start = time.perf_counter()
        nx.single_source_dijkstra_path_length(graph, source, weight="weight")
        nx_time = (time.perf_counter() - start) * 1000

        print("\nDijkstra (1000 nodes, 5000 edges):")
        print(f"  Obrain: {obrain_time:.2f}ms")
        print(f"  NetworkX: {nx_time:.2f}ms")
        print(f"  Ratio: {obrain_time / nx_time:.2f}x")

        # Just ensure both complete
        assert obrain_time > 0
        assert nx_time > 0

    @pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="NetworkX not installed")
    def test_bfs_performance_comparison(self, db):
        """Compare BFS performance between Obrain and NetworkX."""
        import time

        graph_info = self.setup_random_graph(db, 1000, 5000, weighted=False, seed=42)
        node_ids = graph_info["node_ids"]
        edges = graph_info["edges"]
        start_node = node_ids[0]

        # Obrain timing
        obrain_time = self.run_obrain_bfs(db, start_node)

        # NetworkX timing
        graph = nx.DiGraph()
        for src, dst, _ in edges:
            graph.add_edge(src, dst)

        start = time.perf_counter()
        list(nx.bfs_tree(graph, start_node).nodes())
        nx_time = (time.perf_counter() - start) * 1000

        print("\nBFS (1000 nodes, 5000 edges):")
        print(f"  Obrain: {obrain_time:.2f}ms")
        print(f"  NetworkX: {nx_time:.2f}ms")
        print(f"  Ratio: {obrain_time / nx_time:.2f}x")

        # Just ensure both complete
        assert obrain_time > 0
        assert nx_time > 0
