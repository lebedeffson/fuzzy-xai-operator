from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from fuzzyxai.category.diagnostic_completion import try_make_morphism
from fuzzyxai.category.expl_category import ExplanationCategory, ExplanationCategoryObject

from .path_type import ExplanationPath
from .rupture_type import RuptureType


@dataclass(frozen=True)
class TemporalExplanationPath:
    times: tuple[str, ...]
    objects: tuple[str, ...]
    paths: tuple[ExplanationPath, ...] = field(default_factory=tuple)
    ruptures: tuple[RuptureType, ...] = field(default_factory=tuple)

    @property
    def is_continuous(self) -> bool:
        return not self.ruptures


def build_temporal_drift_path(
    category: ExplanationCategory,
    times: Sequence[str],
    objects: Sequence[ExplanationCategoryObject],
) -> TemporalExplanationPath:
    if len(times) != len(objects):
        raise ValueError('times and objects must have equal length')
    paths: list[ExplanationPath] = []
    ruptures: list[RuptureType] = []
    for left, right in zip(objects, objects[1:]):
        result = try_make_morphism(category, left, right, name=f'drift_{left.key}_{right.key}')
        if result.success and result.morphism is not None:
            paths.append(ExplanationPath.from_morphism(result.morphism))
        elif result.diagnostic is not None:
            ruptures.append(RuptureType.from_diagnostic(
                left.key,
                right.key,
                result.diagnostic,
                gamma=result.diagnostic.context.get('gamma'),
                threshold=result.diagnostic.context.get('gamma_max'),
            ))
    return TemporalExplanationPath(
        times=tuple(str(t) for t in times),
        objects=tuple(obj.key for obj in objects),
        paths=tuple(paths),
        ruptures=tuple(ruptures),
    )
