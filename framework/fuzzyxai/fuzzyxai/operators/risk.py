from __future__ import annotations

from fuzzyxai.core.risk_observer import RiskResult
from fuzzyxai.core.risk_observer import observe_risk as _observe_risk

from .contracts import RiskInput


def observe_risk(request: RiskInput) -> RiskResult:
    """Compute rho and action using the canonical core risk observer."""

    return _observe_risk(
        request.components,
        request.weights,
        request.thresholds,
        request.chi_r_crit,
    )
