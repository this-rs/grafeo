"""Abstract base test classes for Obrain test suite.

This module contains base classes that define WHAT to test. Each language
implementation inherits from these and provides HOW (the query syntax).

Base Test Classes:
- BaseMutationsTest: CRUD operations
- BaseTransactionsTest: Transaction handling
- BaseAlgorithmsTest: Graph algorithms

Base Benchmark Classes:
- BaseBenchStorage: Storage benchmarks (reads + writes)
- BaseBenchAlgorithms: Algorithm benchmarks

Comparison Test Classes:
- BaseNetworkXComparisonTest: Compare against NetworkX reference
- BaseNetworkXBenchmarkTest: Benchmark against NetworkX
- BaseSolvORComparisonTest: Compare against OR-Tools reference
- BaseSolvORBenchmarkTest: Benchmark against OR-Tools
"""

from .bench_algorithms import BaseBenchAlgorithms
from .bench_storage import BaseBenchStorage, BenchmarkResult
from .test_algorithms import BaseAlgorithmsTest
from .test_mutations import BaseMutationsTest
from .test_networkx import BaseNetworkXBenchmarkTest, BaseNetworkXComparisonTest
from .test_solvor import BaseSolvORBenchmarkTest, BaseSolvORComparisonTest
from .test_transactions import BaseTransactionsTest

__all__ = [
    # Test base classes
    "BaseMutationsTest",
    "BaseTransactionsTest",
    "BaseAlgorithmsTest",
    # Benchmark base classes
    "BaseBenchStorage",
    "BaseBenchAlgorithms",
    "BenchmarkResult",
    # Comparison test classes
    "BaseNetworkXComparisonTest",
    "BaseNetworkXBenchmarkTest",
    "BaseSolvORComparisonTest",
    "BaseSolvORBenchmarkTest",
]
