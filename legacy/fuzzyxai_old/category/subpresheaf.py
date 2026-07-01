from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .expl_category import ExplanationCategoryObject
from .presheaf import ExplanationPresheaf


class Subpresheaf(ExplanationPresheaf):
    """Finite subpresheaf selected by a context predicate."""

    def __init__(self, parent: ExplanationPresheaf, condition: Callable[[str, Any], bool]) -> None:
        self.parent = parent
        self.condition = condition
        data = {
            obj_key: {value for value in values if condition(obj_key, value)}
            for obj_key, values in parent.contexts.items()
        }
        super().__init__(contexts=data, restrictions=dict(parent.restrictions))

    def is_subobject_of(self, other: ExplanationPresheaf) -> bool:
        return all(values.issubset(other.contexts.get(obj_key, set())) for obj_key, values in self.contexts.items())


def auto_accept_subpresheaf(parent: ExplanationPresheaf) -> Subpresheaf:
    return Subpresheaf(parent, lambda _obj, action: action in {'accept', 'lower_confidence'})


def has_auto_accept(parent: ExplanationPresheaf, obj: ExplanationCategoryObject | str) -> bool:
    return bool(auto_accept_subpresheaf(parent)(obj))
