from dataclasses import dataclass, field
from typing import List, Set, Tuple
from .base import FuzzyRepresentation
from .f0 import F0

@dataclass
class MultiLevelFS(FuzzyRepresentation):
    levels: List[FuzzyRepresentation]
    gamma: Set[Tuple[str, str, str]] = field(default_factory=set)
    weights: List[float] = field(default_factory=list)
    class_name: str = 'FML'

    def __post_init__(self):
        if not self.levels:
            raise ValueError('MultiLevelFS requires levels')
        if not self.weights:
            self.weights = [1 / len(self.levels)] * len(self.levels)
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

    def membership(self, x: float):
        return [level.membership(x) for level in self.levels]

    def distance(self, other: FuzzyRepresentation) -> float:
        if not isinstance(other, MultiLevelFS):
            other = MultiLevelFS([other])
        n = min(len(self.levels), len(other.levels))
        dist = 0.0
        for i in range(n):
            w = self.weights[i] if i < len(self.weights) else 1 / n
            try:
                dist += w * self.levels[i].distance(other.levels[i])
            except TypeError:
                dist += w
        level_penalty = abs(len(self.levels) - len(other.levels)) / max(len(self.levels), len(other.levels))
        gamma_penalty = 0.0 if self.gamma == other.gamma else 0.05
        return min(1.0, dist + 0.10 * level_penalty + gamma_penalty)

    def reduce_to_f0(self):
        reduced = []
        deltas = []
        for level in self.levels:
            red, delta = level.reduce_to_f0()
            reduced.append(red)
            deltas.append(delta)
        def mu(x: float) -> float:
            return sum(w * r.membership(x) for w, r in zip(self.weights, reduced))
        delta = min(1.0, sum(w * d for w, d in zip(self.weights, deltas)) + 0.02 * len(self.gamma))
        return F0(mu, 'red(FML)'), delta

    def coverage(self) -> Set[str]:
        cov = set()
        for level in self.levels:
            cov |= level.coverage()
        if self.gamma:
            cov.add('u_multi')
        return cov

    def complexity(self) -> float:
        return min(1.0, sum(l.complexity() for l in self.levels) + 0.05 * len(self.gamma))
