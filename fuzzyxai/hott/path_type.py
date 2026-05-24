from __future__ import annotations

from dataclasses import dataclass, field

from fuzzyxai.category.expl_category import ExplanationMorphism


@dataclass(frozen=True)
class ExplanationPath:
    """Directed HoTT-style path certificate in Expl."""

    source: str
    target: str
    morphisms: tuple[ExplanationMorphism, ...] = field(default_factory=tuple)
    gamma_total: float = 0.0
    delta_total: float = 0.0
    trace: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_morphism(cls, morphism: ExplanationMorphism) -> 'ExplanationPath':
        return cls(
            source=morphism.source.key,
            target=morphism.target.key,
            morphisms=(morphism,),
            gamma_total=morphism.gamma,
            delta_total=morphism.delta,
            trace=morphism.trace,
        )

    def concat(self, other: 'ExplanationPath') -> 'ExplanationPath':
        if self.target != other.source:
            raise ValueError('path endpoints do not match')
        return ExplanationPath(
            source=self.source,
            target=other.target,
            morphisms=self.morphisms + other.morphisms,
            gamma_total=min(1.0, self.gamma_total + other.gamma_total),
            delta_total=min(1.0, self.delta_total + other.delta_total),
            trace=self.trace + other.trace,
        )

    def is_valid(self, gamma_max: float | None = None) -> bool:
        if not self.morphisms:
            return self.source == self.target
        if gamma_max is not None and self.gamma_total > gamma_max:
            return False
        return self.morphisms[0].source.key == self.source and self.morphisms[-1].target.key == self.target
