from __future__ import annotations

from .expl_category import ExplanationCategory, ExplanationCategoryObject, ExplanationMorphism
from .presheaf import ExplanationPresheaf


class RepresentablePresheaf(ExplanationPresheaf):
    """Representable presheaf y(E)=Hom_Expl(-, E)."""

    def __init__(self, category: ExplanationCategory, target: ExplanationCategoryObject) -> None:
        super().__init__()
        self.category = category
        self.represented = target
        self.contexts = {
            obj.key: set(category.hom(obj, target))
            for obj in category.objects()
        }
        for morphism in category.morphisms():
            self.set_restriction(morphism, self._restriction(morphism))

    def _restriction(self, morphism: ExplanationMorphism):
        def apply(target_morphism: ExplanationMorphism) -> ExplanationMorphism:
            return self.category.compose(morphism, target_morphism)

        return apply


def yoneda_element_count(presheaf: ExplanationPresheaf, obj: ExplanationCategoryObject) -> int:
    """Finite demo form of Nat(y(obj), P) ~= P(obj): returns |P(obj)|."""

    return len(presheaf(obj))
