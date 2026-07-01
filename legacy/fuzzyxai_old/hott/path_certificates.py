from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .path_type import ExplanationPath
from .rupture_type import RuptureType


@dataclass(frozen=True)
class PathCertificate:
    source: str
    target: str
    length: int
    gamma_total: float
    delta_total: float
    trace: tuple[str, ...]
    valid: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuptureCertificate:
    source: str
    target: str
    reason: str
    gamma: float | None
    gamma_max: float | None
    diagnostic_code: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def certify_path(path: ExplanationPath, *, gamma_max: float | None = None) -> PathCertificate:
    return PathCertificate(
        source=path.source,
        target=path.target,
        length=len(path.morphisms),
        gamma_total=path.gamma_total,
        delta_total=path.delta_total,
        trace=path.trace,
        valid=path.is_valid(gamma_max=gamma_max),
    )


def certify_rupture(rupture: RuptureType) -> RuptureCertificate:
    return RuptureCertificate(
        source=rupture.source,
        target=rupture.target,
        reason=rupture.reason,
        gamma=rupture.gamma,
        gamma_max=rupture.threshold,
        diagnostic_code=rupture.diagnostic_state.code,
    )
