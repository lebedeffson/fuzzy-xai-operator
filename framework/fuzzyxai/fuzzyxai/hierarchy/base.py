from abc import ABC, abstractmethod
from typing import Any, Set

class FuzzyRepresentation(ABC):
    class_name: str = 'base'

    @abstractmethod
    def membership(self, x: float) -> Any:
        raise NotImplementedError

    @abstractmethod
    def distance(self, other: 'FuzzyRepresentation') -> float:
        raise NotImplementedError

    @abstractmethod
    def reduce_to_f0(self):
        raise NotImplementedError

    def coverage(self) -> Set[str]:
        return set()

    def complexity(self) -> float:
        return 0.0

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))
