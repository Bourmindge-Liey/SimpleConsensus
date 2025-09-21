from abc import ABC
import numpy as np
import casadi as ca
from scipy.linalg import expm
import networkx as nx

from graph import BaseGraph
from protocol import BaseProtocol
from consensus import Consensus


class MultiAgentSystem(ABC):

    def __init__(self, G: BaseGraph, P: BaseProtocol, x0: np.ndarray = None):
        """Initialize the multi-agent system.

        Args:
            G (BaseGraph): The underlying graph.
            P (BaseProtocol): The protocol for agent communication.
            dt (float): Time step for simulation.
            x0 (np.ndarray): Initial states of the agents.
        """
        
        self.G = G
        self.P = P(G)
        self.L = self.P.L
        self.n = G.graph.number_of_nodes()

        self.x = ca.MX.sym('x', self.n)  # State vector
        self.u = ca.Function("u", [self.x], [self.P(self.x)])  # Control input
        self.disagreement = ca.Function("disagree", [self.x], [self.x.T @ self.G.adj @ self.x])  # Disagreement function

        x0 = np.zeros((self.n)) if x0 is None else x0
        self.reset(x0)


    def simulate(self, t_e: float, dt: float = 0.01) -> dict:
        n_steps = int(t_e / dt) + 1
        sim_result = {
            "time": np.linspace(0, t_e, n_steps),
            "x": np.zeros((self.n, n_steps)),
            "u": np.zeros((self.n, n_steps)),
            "disagreement": np.zeros(n_steps),
            "consensus": np.zeros(n_steps),
        }
        Ld = expm(-self.L * dt)

        x = self.x0
        for i in range(len(sim_result["time"])):
            sim_result["x"][:, i] = x
            sim_result["u"][:, i] = self.u(x).full().flatten()
            sim_result["disagreement"][i] = self.disagreement(x)
            sim_result["consensus"][i] = self.get_consensus(Consensus("average"), x)
            x = Ld @ x

        return sim_result

    def switch_graph(self, G: BaseGraph):
        """Switch the underlying graph.

        Graph, protocol, and disagreement function are updated accordingly.

        Args:
            G (BaseGraph): The new graph.
        """
        if G.graph.number_of_nodes() != self.n:
            raise ValueError("The new graph must have the same number of nodes as the current graph.")
        
        self.G = G
        self.P.update_graph(G)
        self.L = self.P.L
        self.u = ca.Function("u", [self.x], [self.P(self.x)])  # Control input
        self.disagreement = ca.Function("disagree", [self.x], [self.x.T @ self.G.adj @ self.x])  # Disagreement function


    def reset(self, x0: np.ndarray):
        """Reset the states of the agents.

        Args:
            x0 (np.ndarray): Initial states of the agents.
        """
        self.x0 = x0
    
        
    def get_consensus(self, C: Consensus, x: np.ndarray) -> float:
        """Get the consensus value.

        Args:
            C (Consensus): The consensus protocol.

        Returns:
            float: The consensus value.
        """
        return C(x)