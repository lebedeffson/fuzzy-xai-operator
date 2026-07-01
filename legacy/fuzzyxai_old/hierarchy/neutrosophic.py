from dataclasses import dataclass
from typing import Callable, Set
from .base import FuzzyRepresentation, clamp01
from .f0 import F0

@dataclass
class NeutrosophicFS(FuzzyRepresentation):
    t_fn: Callable[[float], float]
    i_fn: Callable[[float], float]
    f_fn: Callable[[float], float]
    source_t: str = ''
    source_i: str = ''
    source_f: str = ''
    class_name: str = 'FNsrc'

    def membership(self, x: float):
        return (clamp01(self.t_fn(float(x))), clamp01(self.i_fn(float(x))), clamp01(self.f_fn(float(x))))

    def distance(self, other: FuzzyRepresentation) -> float:
        if isinstance(other, F0):
            other = NeutrosophicFS(other.mu, lambda x: 0.0, lambda x: 1.0 - other.membership(x))
        if not isinstance(other, NeutrosophicFS):
            raise TypeError(f'cannot compare NeutrosophicFS with {type(other).__name__}')
        grid = [i / 100 for i in range(101)]
        vdist = max(max(abs(a - b) for a, b in zip(self.membership(x), other.membership(x))) for x in grid)
        s1 = (self.source_t, self.source_i, self.source_f)
        s2 = (other.source_t, other.source_i, other.source_f)
        return min(1.0, vdist + (0.0 if s1 == s2 else 0.05))

    def reduce_to_f0(self):
        def mu(x: float) -> float:
            t, _, f = self.membership(x)
            return clamp01((t + 1.0 - f) / 2.0)
        red = F0(mu, 'red(FNsrc)')
        embedded = NeutrosophicFS(red.mu, lambda x: 0.0, lambda x: 1.0 - red.membership(x))
        return red, self.distance(embedded)

    def coverage(self) -> Set[str]:
        cov = {'u_num', 'u_ling', 'u_if', 'u_conf'}
        if self.source_t or self.source_i or self.source_f:
            cov.add('u_trace')
        return cov

    def complexity(self) -> float:
        return 0.42
