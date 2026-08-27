from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class AlignmentInput:
    """Typed input for the chapter 2 alignment operator."""

    components: Mapping[str, float]
    weights: Mapping[str, float]
    gamma_max: float
    delta_t: float = 0.0
    delta_max: float = 1.0


@dataclass(frozen=True)
class ReductionInput:
    """Typed input for a representation reduction and its loss."""

    components: Mapping[str, float]
    weights: Mapping[str, float]
    delta_max: float
    kappa_delta: float = 1.0


@dataclass(frozen=True)
class ReductionOperatorResult:
    delta: float
    delta_max: float
    r_delta: float
    allowed: bool


@dataclass(frozen=True)
class RiskInput:
    """Typed input for strict P19 rho.

    Components are exactly ``rho_p``, ``u_M``, ``one_minus_I_pre``, ``Delta``
    and ``chi_R``. Weights use those same keys and must sum to one.
    """

    components: Mapping[str, float]
    weights: Mapping[str, float]
    thresholds: Mapping[str, float]
    chi_r_crit: int = 0


@dataclass(frozen=True)
class DiagnosticRule:
    """Declarative mapping from an observed condition to a diagnostic state."""

    condition: str
    code: str
    reason: str
    severity: str = "warning"
    recommended_action: str = "review"
    context: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionResult:
    action: str
    status: str
    reason: str
