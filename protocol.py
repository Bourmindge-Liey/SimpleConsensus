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

    @abstractmethod
    def update_graph(self, G: BaseGraph):
        pass


class A1Protocol(BaseProtocol):
    def __init__(self, G: BaseGraph):
        super().__init__("A1")
        self.L = np.diag(np.sum(G.adj, axis=1)) - G.adj

    def __call__(self, x: np.ndarray | ca.MX) -> np.ndarray | ca.MX:
        return - self.L @ x
    
    def update_graph(self, G: BaseGraph):
        self.L = np.diag(np.sum(G.adj, axis=1)) - G.adj