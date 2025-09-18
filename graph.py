"""Graph module.

This provides graphs built on top of NetworkX.

1. BaseGraph: Graph base class.
2. UnDiGraph: Undirected graph.
3. DiGraph: Directed graph.
"""

from abc import ABC
import numpy as np
import networkx as nx

class BaseGraph(ABC):
    """Base graph class."""

    def __init__(self, is_directed: bool, V: tuple[int], E: list[tuple[int, int]], A: np.ndarray):
        """Initialize the graph.

        Args:
            is_directed (bool): Whether the graph is directed.
            V (tuple[int]): Nodes.
            E (list[tuple[int, int]]): Edges.
            A (np.ndarray): Adjacency matrix.
        """
        self.is_directed : bool = is_directed
        self.graph = nx.DiGraph() if is_directed else nx.Graph()

        # Check non-negativity of adjacency matrix
        if np.any(A < 0):
            raise ValueError("Adjacency matrix must have non-negative entries.")
        # Check non-connected edge in adjacency matrix equal to zero
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if i == j and (i, j) in V:
                    raise ValueError(f"Node {i} cannot have a self-loop.")
                if self.is_directed == False and (i, j) not in E and (j, i) not in E and A[i, j] != 0:
                    raise ValueError(f"Adjacency matrix entry A[{i}, {j}] must be zero for non-connected edges.")
                if self.is_directed == True and (i, j) not in E and A[i, j] != 0:
                    raise ValueError(f"Adjacency matrix entry A[{i}, {j}] must be zero for non-connected edges.")

        self.graph.add_nodes_from(V)
        for u, v in E:
            self.graph.add_edge(u, v, weight=A[u, v])
        self.adj = A

class UnDiGraph(BaseGraph):

    def __init__(self, V: tuple[int], E: list[tuple[int, int]], A: np.ndarray):
        # Check symmetry of adjacency matrix
        if not np.array_equal(A, A.T):
            raise ValueError("Adjacency matrix must be symmetric for undirected graphs.")
        super().__init__(is_directed=False, V=V, E=E, A=A)

class DiGraph(BaseGraph):

    def __init__(self, V: tuple[int], E: list[tuple[int, int]], A: np.ndarray):
        super().__init__(is_directed=True, V=V, E=E, A=A)