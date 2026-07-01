from dataclasses import dataclass
from typing import Callable, Literal, Set
from .base import FuzzyRepresentation, clamp01
from .f0 import F0

@dataclass
class IntervalFS(FuzzyRepresentation):
    lower: Callable[[float], float]
    upper: Callable[[float], float]
    policy: Literal['mid', 'upper', 'lower'] = 'mid'
    class_name: str = 'FI'

    def membership(self, x: float):
        lo = clamp01(self.lower(float(x)))
        hi = clamp01(self.upper(float(x)))
        return (min(lo, hi), max(lo, hi))

    def distance(self, other: FuzzyRepresentation) -> float:
        if isinstance(other, F0):
            other = IntervalFS(other.mu, other.mu)
        if not isinstance(other, IntervalFS):
            raise TypeError(f'cannot compare IntervalFS with {type(other).__name__}')
        grid = [i / 100 for i in range(101)]
        return max(max(abs(self.membership(x)[0] - other.membership(x)[0]),
                       abs(self.membership(x)[1] - other.membership(x)[1])) for x in grid)

    def reduce_to_f0(self):
        def mu(x: float) -> float:
            lo, hi = self.membership(x)
            if self.policy == 'upper':
                return hi
            if self.policy == 'lower':
                return lo
            return 0.5 * (lo + hi)
        grid = [i / 100 for i in range(101)]
        delta = 0.5 * max(abs(self.membership(x)[1] - self.membership(x)[0]) for x in grid)
        return F0(mu, 'red(FI)'), float(delta)

    def coverage(self) -> Set[str]:
        return {'u_num', 'u_ling', 'u_int'}

    def complexity(self) -> float:
        return 0.22
