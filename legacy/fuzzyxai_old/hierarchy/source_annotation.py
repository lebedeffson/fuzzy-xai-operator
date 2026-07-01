from dataclasses import dataclass
from typing import Any, Set
from .base import FuzzyRepresentation

@dataclass
class SourceAnnotated(FuzzyRepresentation):
    base: FuzzyRepresentation
    source: str
    trace_id: str
    class_name: str = 'Fsrc'

    def membership(self, x: float) -> Any:
        return self.base.membership(x)

    def distance(self, other: FuzzyRepresentation) -> float:
        if isinstance(other, SourceAnnotated):
            return min(1.0, self.base.distance(other.base) + (0 if self.source == other.source else 0.05) + (0 if self.trace_id == other.trace_id else 0.05))
        return min(1.0, self.base.distance(other) + 0.10)

    def reduce_to_f0(self):
        red, delta = self.base.reduce_to_f0()
        return red, min(1.0, delta + 0.05)

    def coverage(self) -> Set[str]:
        return set(self.base.coverage()) | {'u_trace'}

    def complexity(self) -> float:
        return min(1.0, self.base.complexity() + 0.08)
