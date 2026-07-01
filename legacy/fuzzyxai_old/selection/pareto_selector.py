from dataclasses import dataclass
from typing import List, Sequence, Set
from .choice_diagnostic import choice_diagnostic

@dataclass(frozen=True)
class Candidate:
    name: str
    coverage: Set[str]
    cognitive_complexity: float
    computational_complexity: float
    expected_reduction_loss: float
    factory: object = None

    def covers(self, profile: Set[str]) -> bool:
        return profile <= self.coverage

def dominates(a: Candidate, b: Candidate) -> bool:
    return (
        a.cognitive_complexity <= b.cognitive_complexity and
        a.computational_complexity <= b.computational_complexity and
        a.expected_reduction_loss <= b.expected_reduction_loss and
        (a.cognitive_complexity < b.cognitive_complexity or
         a.computational_complexity < b.computational_complexity or
         a.expected_reduction_loss < b.expected_reduction_loss)
    )

def pareto_front(candidates: Sequence[Candidate]) -> List[Candidate]:
    return [c for c in candidates if not any(dominates(o, c) for o in candidates if o is not c)]

def select_minimal_sufficient(profile: Set[str], candidates: Sequence[Candidate], mode: str = 'audit'):
    admissible = [c for c in candidates if c.covers(profile)]
    if not admissible:
        covered = set().union(*(c.coverage for c in candidates)) if candidates else set()
        return choice_diagnostic(profile, covered)
    front = pareto_front(admissible)
    if mode == 'user':
        return min(front, key=lambda c: (c.cognitive_complexity, c.expected_reduction_loss, c.computational_complexity))
    if mode == 'audit':
        return min(front, key=lambda c: (c.expected_reduction_loss, c.cognitive_complexity, c.computational_complexity))
    return front
