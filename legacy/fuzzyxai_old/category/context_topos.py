from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .expl_category import ExplanationCategory, ExplanationCategoryObject
from .presheaf import ExplanationPresheaf

AUTO_ACTIONS = {'accept', 'lower_confidence'}


def _key(obj: ExplanationCategoryObject | str) -> str:
    return obj if isinstance(obj, str) else obj.key


class RiskContext(ExplanationPresheaf):
    """Presheaf of admissible observer actions for each explanation object."""

    def __init__(
        self,
        category: ExplanationCategory,
        actions: Mapping[ExplanationCategoryObject | str, set[str]],
    ) -> None:
        super().__init__()
        self.category = category
        self.contexts = {_key(obj): set(values) for obj, values in actions.items()}
        self._install_identity_restrictions()

    def _install_identity_restrictions(self) -> None:
        for obj in self.category.objects():
            self.set_restriction(self.category.identity(obj), lambda action: action)

    def allowed_actions(self, obj: ExplanationCategoryObject | str) -> set[str]:
        return set(self.contexts.get(_key(obj), set()))

    def allows_auto_accept(self, obj: ExplanationCategoryObject | str) -> bool:
        return bool(self.allowed_actions(obj) & AUTO_ACTIONS)


class AuditContext(ExplanationPresheaf):
    """Presheaf of audit modes/requirements."""

    def __init__(self, category: ExplanationCategory, levels: Mapping[ExplanationCategoryObject | str, set[str]]) -> None:
        super().__init__({_key(obj): set(values) for obj, values in levels.items()})
        for obj in category.objects():
            self.set_restriction(category.identity(obj), lambda level: level)


class UserContext(ExplanationPresheaf):
    """Presheaf of user-facing presentation modes."""

    def __init__(self, category: ExplanationCategory, modes: Mapping[ExplanationCategoryObject | str, set[str]]) -> None:
        super().__init__({_key(obj): set(values) for obj, values in modes.items()})
        for obj in category.objects():
            self.set_restriction(category.identity(obj), lambda mode: mode)


class TraceContext(ExplanationPresheaf):
    """Presheaf of trace disclosure levels."""

    def __init__(self, category: ExplanationCategory, traces: Mapping[ExplanationCategoryObject | str, set[Any]]) -> None:
        super().__init__({_key(obj): set(values) for obj, values in traces.items()})
        for obj in category.objects():
            self.set_restriction(category.identity(obj), lambda trace: trace)


@dataclass(frozen=True)
class PresheafToposDescriptor:
    """Descriptor of the safe topos layer Set^{Expl^op}.

    This is intentionally not a full categorical algebra package. It records
    the standard construction used in the dissertation appendix.
    """

    base_category: str = 'Expl'
    notation: str = 'Set^{Expl^op}'
    role: str = 'topos of explanation contexts'

    def contains(self, presheaf: ExplanationPresheaf) -> bool:
        return isinstance(presheaf, ExplanationPresheaf)
