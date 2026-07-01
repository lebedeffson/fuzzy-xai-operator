from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Tuple

from .base import FuzzyRepresentation
from .f0 import F0
from .interval import IntervalFS
from .hesitant import HesitantFS
from .neutrosophic import NeutrosophicFS


@dataclass(frozen=True)
class ReductionResult:
    reduced: FuzzyRepresentation
    delta: float
    policy: str


class MetaReducer:
    """Goal-aware reducer for chapter 3.

    It evaluates admissible reduction policies and returns the policy with the
    smallest loss, subject to the chosen explanation goal.
    """

    def __init__(self, goal: str = 'neutral'):
        self.goal = goal

    def reduce(self, obj: FuzzyRepresentation) -> ReductionResult:
        candidates: list[ReductionResult] = []
        if isinstance(obj, IntervalFS):
            policies = ['mid']
            if self.goal in {'audit', 'conservative', 'risk'}:
                policies.append('upper')
            if self.goal in {'optimistic'}:
                policies.append('lower')
            for policy in policies:
                trial = IntervalFS(obj.lower, obj.upper, policy=policy)
                red, delta = trial.reduce_to_f0()
                candidates.append(ReductionResult(red, delta, policy))
        elif isinstance(obj, HesitantFS):
            red, delta = obj.reduce_to_f0()
            candidates.append(ReductionResult(red, delta, 'mean'))
        elif isinstance(obj, NeutrosophicFS):
            red_bal, delta_bal = obj.reduce_to_f0()
            candidates.append(ReductionResult(red_bal, delta_bal, 'balanced_neutrosophic'))
            support = F0(lambda x: obj.membership(x)[0], label='support_only')
            embedded = NeutrosophicFS(support.mu, lambda x: 0.0, lambda x: 1.0 - support.membership(x))
            candidates.append(ReductionResult(support, obj.distance(embedded), 'support_only'))
        else:
            red, delta = obj.reduce_to_f0()
            candidates.append(ReductionResult(red, delta, 'canonical'))
        return min(candidates, key=lambda c: c.delta)
