from abc import ABC, abstractmethod
import numpy as np
import casadi as ca
from graph import BaseGraph

class BaseProtocol(ABC):
    
    def __init__(self, protocol_type: str):
        self.protocol_type = protocol_type

    @abstractmethod
    def __call__(self, x):
        pass


class NumProtocol(BaseProtocol):
    """Distributed protocol based on adjacency matrix.
    $$ u_i = \sum^N_j=1 a_ij(x_j - x_i) $$
    """
    def __init__(self):
        super().__init__("distributed")

    def __call__(self, Ai: np.array, x_k: np.ndarray, x_i: np.ndarray) -> np.ndarray:
        return Ai @ (x_k - x_i)
    

"""implemented for analytical simulation and matrix calculation"""
class AnaProtocol(BaseProtocol):
    def __init__(self, G: BaseGraph):
        super().__init__("matrix")
        self.L = np.diag(np.sum(G.A, axis=1)) - G.A

    def __call__(self, x: np.ndarray | ca.MX) -> np.ndarray | ca.MX:
        return - self.L @ x
    
    def update_graph(self, G: BaseGraph):
        self.L = np.diag(np.sum(G.A, axis=1)) - G.A