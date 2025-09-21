from abc import ABC, abstractmethod
import numpy as np


class Consensus(ABC):
    """consensus computations."""

    def __init__(self, consensus_type: str):
        self.consensus_type = consensus_type
        match consensus_type:
            case "average":
                self.consensus = np.mean
            case "max":
                self.consensus = np.max
            case "min":
                self.consensus = np.min
            case _:
                raise ValueError(f"Unknown consensus type: {consensus_type}")

    def __call__(self, x: np.ndarray) -> float:
        """Compute consensus value from state vector x."""
        return self.consensus(x)