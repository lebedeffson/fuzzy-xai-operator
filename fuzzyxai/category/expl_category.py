from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from fuzzyxai.core.explanation_object import ExplanationObject
from fuzzyxai.core.trust_evaluator import semantic_disagreement


@dataclass(frozen=True)
class ExplanationCategoryObject:
    """Object of Expl: a valid extended explanation object with a stable key."""

    key: str
    explanation: ExplanationObject


@dataclass(frozen=True)
class ExplanationMorphism:
    """Successful admissible alignment T_ij: E_i -> E_j."""

    source: ExplanationCategoryObject
    target: ExplanationCategoryObject
    name: str
    gamma: float = 0.0
    delta: float = 0.0
    trace: tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def signature(self) -> str:
        return f'{self.source.key}->{self.target.key}:{self.name}'


class ExplanationCategory:
    """Small category of successful explanation alignments for one ExplainPlan.

    Diagnostic states are intentionally outside this category: if alignment
    violates gamma_max or interface rules, no morphism is created.
    """

    def __init__(self, beta: Mapping[str, float] | None = None, gamma_max: float | None = None) -> None:
        self.beta = dict(beta or {})
        if gamma_max is not None:
            self.beta['gamma_max'] = float(gamma_max)
        self.gamma_max = self.beta.get('gamma_max')

    def object(self, key: str, explanation: ExplanationObject) -> ExplanationCategoryObject:
        return ExplanationCategoryObject(key=key, explanation=explanation)

    def identity(self, obj: ExplanationCategoryObject) -> ExplanationMorphism:
        return ExplanationMorphism(
            source=obj,
            target=obj,
            name=f'id_{obj.key}',
            gamma=0.0,
            delta=0.0,
            trace=(f'id:{obj.key}',),
        )

    def is_valid_morphism(self, morphism: ExplanationMorphism) -> bool:
        if morphism.gamma < 0 or morphism.delta < 0:
            return False
        return self.gamma_max is None or morphism.gamma <= float(self.gamma_max)

    def make_morphism(
        self,
        source: ExplanationCategoryObject,
        target: ExplanationCategoryObject,
        *,
        name: str | None = None,
        gamma: float | None = None,
        delta: float | None = None,
        trace: tuple[str, ...] | None = None,
    ) -> ExplanationMorphism:
        if gamma is None:
            gamma = semantic_disagreement(source.explanation, target.explanation, self.beta)
        if delta is None:
            delta = abs(source.explanation.reduction_loss - target.explanation.reduction_loss)
        morphism = ExplanationMorphism(
            source=source,
            target=target,
            name=name or f'T_{source.key}_{target.key}',
            gamma=float(gamma),
            delta=float(delta),
            trace=trace or (source.key, target.key),
        )
        if not self.is_valid_morphism(morphism):
            raise ValueError('morphism violates gamma_max or category constraints')
        return morphism

    def compose(self, first: ExplanationMorphism, second: ExplanationMorphism) -> ExplanationMorphism:
        """Return second o first for A --first--> B --second--> C."""
        if first.target.key != second.source.key:
            raise ValueError('morphism endpoints do not match')
        gamma = min(1.0, first.gamma + second.gamma)
        delta = min(1.0, first.delta + second.delta)
        composed = ExplanationMorphism(
            source=first.source,
            target=second.target,
            name=f'{second.name}_after_{first.name}',
            gamma=gamma,
            delta=delta,
            trace=first.trace + second.trace,
            metadata={'components': (first.signature, second.signature)},
        )
        if not self.is_valid_morphism(composed):
            raise ValueError('composed morphism violates gamma_max or category constraints')
        return composed
