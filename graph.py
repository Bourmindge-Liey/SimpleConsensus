"""Graph module.

This provides graphs built on top of NetworkX.

1. BaseGraph: Graph base class.
2. UnDiGraph: Undirected graph.
3. DiGraph: Directed graph.
"""

from abc import ABC
import numpy as np
import networkx as nx
from copy import deepcopy

class BaseGraph(ABC):
    """Base graph class."""

    def __init__(self, is_directed: bool, V: tuple[int], E: list[tuple[int, int]], A: np.ndarray | None = None):
        """Initialize the graph.

        Args:
            is_directed (bool): Whether the graph is directed.
            V (tuple[int]): Nodes.
            E (list[tuple[int, int]]): Edges.
            A (np.ndarray): Adjacency matrix. If None, defaults to 1.
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
        self.V = V
        self.E = E
        self.A = A


    def balanced(self) -> bool:
        """Check if the graph is balanced.

        A directed graph is balanced if for every node, the in-degree equals the out-degree.

        Returns:
            bool: True if the graph is balanced, False otherwise.
        """
        # undirected graphs are always balanced``
        if not self.is_directed:
            return True
        for node in self.graph.nodes():
            if self.graph.in_degree(node) != self.graph.out_degree(node):
                return False
        return True


    def strongly_connected(self) -> bool:
        """Check if the directed graph is strongly connected.

        A directed graph is strongly connected if there is a path from any node to every other node.

        Returns:
            bool: True if the graph is strongly connected, False otherwise.
        """
        if not self.is_directed:
            return True
        return nx.is_strongly_connected(self.graph)


class UnDiGraph(BaseGraph):

    def __init__(self, V: tuple[int], E: list[tuple[int, int]], A: np.ndarray | None = None):

        if A is None:
            A = np.zeros((len(V), len(V)))
            for u, v in E:
                A[u, v] = 1
                A[v, u] = 1
        # Check symmetry of adjacency matrix
        if not np.array_equal(A, A.T):
            raise ValueError("A must be symmetric for undirected graphs.")
        for i in range(A.shape[0]):
            if A[i, i] != 0 or (i, i) in E:
                raise ValueError(f"Node {i} cannot have a self-loop.")
            for j in range(A.shape[1]):
                if (i, j) not in E and (j, i) not in E and A[i, j] != 0:
                    raise ValueError(f"A[{i}, {j}] must be zero for non-connected edges.")
                if A[i, j] == 0:
                    if (i, j) in E or (j, i) in E:
                        raise ValueError(f"A[{i}, {j}] must be non-zero for connected edges.")

        super().__init__(is_directed=False, V=V, E=E, A=A)


class DiGraph(BaseGraph):

    def __init__(self, V: tuple[int], E: list[tuple[int, int]], A: np.ndarray | None = None):

        if A is None:
            A = np.zeros((len(V), len(V)))
            for u, v in E:
                A[u, v] = 1
        for i in range(A.shape[0]):
            if A[i, i] != 0 or (i, i) in E:
                raise ValueError(f"Node {i} cannot have a self-loop.")
            for j in range(A.shape[1]):
                if (i, j) not in E and A[i, j] != 0:
                    raise ValueError(f"A[{i}, {j}] must be zero for non-connected edges.")
                if (i, j) in E and A[i, j] == 0:
                    raise ValueError(f"A[{i}, {j}] must be non-zero for connected edges.")        

        super().__init__(is_directed=True, V=V, E=E, A=A)


def get_mirror(G: BaseGraph) -> UnDiGraph:
    """Get the mirror graph of a directed graph.

    The mirror graph is obtained by adding reverse edges to all existing edges.

    Args:
        G (BaseGraph): The original directed graph.

    Returns:
        BaseGraph: The mirror graph.
    """
    if isinstance(G, UnDiGraph):
        return deepcopy(G)
    V_mirror = G.V
    E_mirror = []
    for u, v in G.E:
        if (v, u) not in E_mirror or (u, v) not in E_mirror:
            E_mirror.append((u, v))
    A_mirror = 0.5 * (G.A.T + G.A)
    return UnDiGraph(V=V_mirror, E=E_mirror, A=A_mirror)