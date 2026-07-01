from dataclasses import dataclass
from typing import Callable, Set
from .base import FuzzyRepresentation, clamp01

@dataclass
class F0(FuzzyRepresentation):
    mu: Callable[[float], float]
    label: str = 'mu'
    class_name: str = 'F0'

    def membership(self, x: float) -> float:
        return clamp01(self.mu(float(x)))

    def distance(self, other: FuzzyRepresentation) -> float:
        if not isinstance(other, F0):
            return other.distance(self)
        grid = [i / 100 for i in range(101)]
        return max(abs(self.membership(x) - other.membership(x)) for x in grid)

    def reduce_to_f0(self):
        return self, 0.0

    def coverage(self) -> Set[str]:
        return {'u_num', 'u_ling'}

    def complexity(self) -> float:
        return 0.10
