from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from .expl_category import ExplanationCategoryObject, ExplanationMorphism

ContextMap = Callable[[Any], Any]


@dataclass
class ExplanationPresheaf:
    """Set-valued presheaf on Expl, represented by finite context sets."""

    contexts: dict[str, set[Any]] = field(default_factory=dict)
    restrictions: dict[str, ContextMap] = field(default_factory=dict)

    def __call__(self, obj: ExplanationCategoryObject | str) -> set[Any]:
        key = obj if isinstance(obj, str) else obj.key
        return self.contexts.get(key, set())

    def add_contexts(self, obj: ExplanationCategoryObject, values: Iterable[Any]) -> None:
        self.contexts[obj.key] = set(values)

    def set_restriction(self, morphism: ExplanationMorphism, mapping: ContextMap) -> None:
        self.restrictions[morphism.signature] = mapping

    def restrict(self, morphism: ExplanationMorphism, context: Any) -> Any:
        mapping = self.restrictions.get(morphism.signature)
        if mapping is None:
            return context
        return mapping(context)

    def check_identity(self, obj: ExplanationCategoryObject, identity: ExplanationMorphism) -> bool:
        return all(self.restrict(identity, c) == c for c in self.contexts.get(obj.key, set()))

    def check_contravariant_composition(
        self,
        first: ExplanationMorphism,
        second: ExplanationMorphism,
        composed: ExplanationMorphism,
    ) -> bool:
        for context in self.contexts.get(second.target.key, set()):
            left = self.restrict(composed, context)
            right = self.restrict(first, self.restrict(second, context))
            if left != right:
                return False
        return True


class ContextPresheaf(ExplanationPresheaf):
    pass


@dataclass
class RepresentablePresheaf(ExplanationPresheaf):
    represented: ExplanationCategoryObject | None = None
