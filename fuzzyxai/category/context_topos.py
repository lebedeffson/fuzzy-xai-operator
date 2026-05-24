from __future__ import annotations

from dataclasses import dataclass

from .presheaf import ExplanationPresheaf


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
