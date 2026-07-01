from dataclasses import dataclass
from typing import Callable, Iterable, List, Set
from .base import FuzzyRepresentation, clamp01
from .f0 import F0

def hausdorff_1d(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 1.0
    return max(max(min(abs(x - y) for y in b) for x in a),
               max(min(abs(y - x) for x in a) for y in b))

@dataclass
class HesitantFS(FuzzyRepresentation):
    values_fn: Callable[[float], Iterable[float]]
    class_name: str = 'FH'

    def membership(self, x: float) -> List[float]:
        return sorted(clamp01(v) for v in self.values_fn(float(x)))

    def distance(self, other: FuzzyRepresentation) -> float:
        if isinstance(other, F0):
            other = HesitantFS(lambda x: [other.membership(x)])
        if not isinstance(other, HesitantFS):
            raise TypeError(f'cannot compare HesitantFS with {type(other).__name__}')
        grid = [i / 100 for i in range(101)]
        return max(hausdorff_1d(self.membership(x), other.membership(x)) for x in grid)

    def reduce_to_f0(self):
        def mu(x: float) -> float:
            vals = self.membership(x)
            return sum(vals) / len(vals) if vals else 0.0
        red = F0(mu, 'red(FH)')
        embedded = HesitantFS(lambda x: [red.membership(x)])
        return red, self.distance(embedded)

    def coverage(self) -> Set[str]:
        return {'u_num', 'u_ling', 'u_exp'}

    def complexity(self) -> float:
        return 0.30
