from __future__ import annotations

from dataclasses import dataclass

from fuzzyxai.core.diagnostics import DiagnosticState

from .expl_category import ExplanationCategory, ExplanationCategoryObject, ExplanationMorphism


@dataclass(frozen=True)
class MorphismResult:
    success: bool
    morphism: ExplanationMorphism | None = None
    diagnostic: DiagnosticState | None = None


def try_make_morphism(
    category: ExplanationCategory,
    source: ExplanationCategoryObject,
    target: ExplanationCategoryObject,
    *,
    name: str | None = None,
    gamma: float | None = None,
) -> MorphismResult:
    try:
        morphism = category.make_morphism(source, target, name=name, gamma=gamma)
        return MorphismResult(success=True, morphism=morphism)
    except ValueError as exc:
        threshold = category.gamma_max
        return MorphismResult(
            success=False,
            diagnostic=DiagnosticState(
                code='D_ij',
                reason=str(exc),
                severity='critical',
                context={
                    'source': source.key,
                    'target': target.key,
                    'gamma': gamma,
                    'gamma_max': threshold,
                },
            ),
        )
