from __future__ import annotations

from dataclasses import dataclass
from typing import Any




@dataclass(frozen=True)
class Candidate:
    """Minimal local Pareto candidate used by the standalone experiment pack."""
    name: str
    coverage: set[str]
    cognitive_complexity: float
    computational_complexity: float
    expected_reduction_loss: float

    def covers(self, profile: set[str]) -> bool:
        return profile <= self.coverage


def _select_minimal_sufficient(profile: set[str], candidates: list[Candidate]) -> Candidate | None:
    admissible = [c for c in candidates if c.covers(profile)]
    if not admissible:
        return None
    return min(admissible, key=lambda c: (c.expected_reduction_loss, c.cognitive_complexity, c.computational_complexity))

@dataclass(frozen=True)
class Fuzzy0:
    """Classical fuzzy value."""
    mu: float


@dataclass(frozen=True)
class FuzzyInt:
    """Interval-valued fuzzy representation."""
    low: float
    high: float

    def to_mid(self) -> float:
        """Reduce interval to midpoint."""
        return (self.low + self.high) / 2

    def distance_to(self, other: 'FuzzyInt') -> float:
        """Return L1 interval endpoint distance."""
        return (abs(self.low - other.low) + abs(self.high - other.high)) / 2


@dataclass(frozen=True)
class FuzzyH:
    """Hesitant fuzzy representation."""
    values: list[float]

    def to_mean(self) -> float:
        """Reduce hesitant set to mean."""
        return sum(self.values) / max(1, len(self.values))

    def hausdorff_distance(self, other: 'FuzzyH') -> float:
        """Compute simple 1D Hausdorff distance."""
        if not self.values or not other.values:
            return 1.0
        return max(
            max(min(abs(a - b) for b in other.values) for a in self.values),
            max(min(abs(b - a) for a in self.values) for b in other.values),
        )


@dataclass(frozen=True)
class FuzzyNsrc:
    """Source-aware neutrosophic representation."""
    T: float
    I: float
    F: float
    sources: list[str]

    def to_T(self) -> float:
        """Reduce to truth component."""
        return self.T

    def distance(self, other: 'FuzzyNsrc') -> float:
        """Return component-wise mean distance."""
        return (abs(self.T - other.T) + abs(self.I - other.I) + abs(self.F - other.F)) / 3


@dataclass(frozen=True)
class FuzzyML:
    """Multi-level fuzzy representation."""
    levels: list[Any]

    def reduce(self, weights: list[float]) -> tuple[float, float]:
        """Return weighted mean and simple reduction loss."""
        vals: list[float] = []
        for x in self.levels:
            if hasattr(x, 'mu'):
                vals.append(float(x.mu))
            elif hasattr(x, 'to_mid'):
                vals.append(float(x.to_mid()))
            elif hasattr(x, 'to_mean'):
                vals.append(float(x.to_mean()))
            else:
                vals.append(float(x))
        total = sum(weights) or 1.0
        value = sum(v * w for v, w in zip(vals, weights)) / total
        delta = max(vals) - min(vals) if vals else 0.0
        return value, delta


def select_class(profile: dict[str, bool], plan: dict[str, Any]) -> str:
    """Select a minimal sufficient representation class."""
    candidates = [
        Candidate('F0', {'scalar'}, 1, 1, 0.25),
        Candidate('F_int', {'scalar', 'interval'}, 2, 2, 0.12),
        Candidate('F_H', {'scalar', 'experts'}, 3, 2, 0.10),
        Candidate('F_N_src', {'scalar', 'conflict', 'sources'}, 4, 3, 0.08),
        Candidate('F_ML', {'scalar', 'interval', 'experts', 'conflict', 'sources', 'trace'}, 5, 4, 0.05),
    ]
    required = {k for k, v in profile.items() if v}
    
    selected = _select_minimal_sufficient(required, candidates)
    return selected.name if selected is not None else 'DiagnosticState'
