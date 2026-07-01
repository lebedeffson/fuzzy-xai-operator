from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from fuzzyxai.category import CertifiedPath
from fuzzyxai.core.diagnostics import DiagnosticState


@dataclass(frozen=True)
class RuptureType:
    """Certificate that a directed explanation path cannot be constructed."""

    source: str
    target: str
    diagnostic_state: DiagnosticState
    reason: str
    gamma: float | None = None
    threshold: float | None = None
    metadata: Mapping[str, Any] | None = None

    @classmethod
    def from_diagnostic(
        cls,
        source: str,
        target: str,
        diagnostic: DiagnosticState,
        *,
        gamma: float | None = None,
        threshold: float | None = None,
    ) -> 'RuptureType':
        return cls(
            source=source,
            target=target,
            diagnostic_state=diagnostic,
            reason=diagnostic.reason,
            gamma=gamma,
            threshold=threshold,
            metadata=diagnostic.context,
        )


def rupture_inhabited(certified_path: CertifiedPath) -> bool:
    if certified_path.rupture:
        return True
    return any('critical' in d.lower() or 'crit' in d.lower() for d in certified_path.diagnostics)
