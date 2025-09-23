from abc import ABC
import numpy as np
import casadi as ca
from scipy.linalg import expm
import networkx as nx

from graph import BaseGraph
from protocol import BaseProtocol
from consensus import Consensus


class MultiAgentSystem(ABC):

    def __init__(self, G: BaseGraph, T_delay: np.array = None):
        """Initialize the multi-agent system.

        Args:
            G: The graph.
            T_delay: n*n matrix, t_ij is delay i->j. Defaults is 0.
        """
        
        # System Proterties
        self.G = G
        self.n = G.graph.number_of_nodes()

        if T_delay is not None:
            if T_delay.shape != (self.n, self.n):
                raise ValueError(f"T_delay must be of shape ({self.n}, {self.n}), but got {T_delay.shape}.")
            if np.any(T_delay < 0):
                raise ValueError("T_delay must be non-negative.")
        else:
            T_delay = np.zeros((self.n, self.n))
        self.T_delay = T_delay

        # Disagreement expression
        x = ca.MX.sym('x', self.n)  # State vector
        self.disagreement = ca.Function("disagree", [x], [x.T @ x]) 

        # reset states
        self.x0 = np.zeros((self.n))
        self.reset(np.zeros((self.n)))


    def simulate_numerical(self, P: BaseProtocol, t_e: float, dt: float = 0.01) -> dict:
        """Simulate the multi-agent system using numerical integration.

        $$ x(t + dt) = x(t) + u(t) \cdot dt $$

        Args:
            P: The protocol.
            t_e: End time of the simulation.
            dt: Time step for the simulation.
        """
        # Initialize result storage
        n_steps = int(t_e / dt) + 1
        delay_steps = (self.T_delay / dt).astype(int) 
        sim_result = {
            "time": np.linspace(0, t_e, n_steps),
            "x": np.zeros((self.n, n_steps)),
            "u": np.zeros((self.n, n_steps)),
            "disagreement": np.zeros(n_steps),
            "consensus": np.zeros(n_steps),
        }

        Pi = P()
        x = self.x0
        for i in range(n_steps):
            sim_result["x"][:, i] = x

            idx = np.maximum(0, i - delay_steps)           # (n,n)
            x_k = sim_result["x"][np.arange(self.n), idx]  # (n,n): delayed neighbor states
            x_j = sim_result["x"][np.arange(self.n).reshape(-1, 1), idx]  # (n,n): delayed self states
            u = np.array([Pi(self.G.A[j, :], x_k[j, :], x_j[j, :]) for j in range(self.n)])
            sim_result["u"][:, i] = u

            sim_result["consensus"][i] = Consensus("average")(x)
            sim_result["disagreement"][i] = self.disagreement(x - sim_result["consensus"][i])
            x += u * dt
        self.x0 = x

        return sim_result


    def simulate_analytical(self, P: BaseProtocol, t_e: float, dt: float = 0.01) -> dict:
        """Simulate the multi-agent system using analytical solution.
        
        $$ x(t) = e^(-Lt)x0 $$

        Args:
            P: The protocol.
            t_e: End time of the simulation.
            dt: Time step for the simulation.
        """
        # initialize simulation result
        n_steps = int(t_e / dt) + 1
        sim_result = {
            "time": np.linspace(0, t_e, n_steps),
            "x": np.zeros((self.n, n_steps)),
            "u": np.zeros((self.n, n_steps)),
            "disagreement": np.zeros(n_steps),
            "consensus": np.zeros(n_steps),
        }

        # precompute matrix exponential
        Pi = P(self.G)
        Ld = expm(-Pi.L * dt)

        x = self.x0
        for i in range(len(sim_result["time"])):
            sim_result["x"][:, i] = x
            sim_result["u"][:, i] = Pi(x)
            sim_result["consensus"][i] = Consensus("average")(x)
            sim_result["disagreement"][i] = self.disagreement(x - sim_result["consensus"][i])
            x = Ld @ x
        self.x0 = x

        return sim_result


    def switch_graph(self, G: BaseGraph):
        """Switch the underlying graph.

        Graph, protocol, and disagreement function are updated accordingly.

        Args:
            G : The new graph.
        """
        if G.graph.number_of_nodes() != self.n:
            raise ValueError("The new graph must have the same number of nodes as the current graph.")
        self.G = G

    
    def switch_delay(self, T_delay: np.array):
        """Update the communication delay matrix."""
        if T_delay is not None:
            if T_delay.shape != (self.n, self.n):
                raise ValueError(f"T_delay must be of shape ({self.n}, {self.n}), but got {T_delay.shape}.")
            if np.any(T_delay < 0):
                raise ValueError("T_delay must be non-negative.")
            self.T_delay = T_delay


    def reset(self, x0: np.ndarray):
        """Reset the states of the agents.
    
        Args:
            x0 (np.ndarray): Initial states of the agents.
        """
        if x0.shape != (self.n,):
            raise ValueError(f"x0 must be of shape ({self.n},), but got {x0.shape}.")
        self.x0 = x0


    def algebraic_connectivity(self) -> float:
        """Get the algebraic connectivity of the underlying graph.
        i.e. Second smallest eigenvalue
        
        Returns:
            float: The algebraic connectivity.
        """
        if not self.G.is_directed:
            raise ValueError("Algebraic connectivity is only defined for directed graphs.")
        return np.real(np.sort(np.linalg.eigvals(nx.laplacian_matrix(self.G.graph)))[1]) 

