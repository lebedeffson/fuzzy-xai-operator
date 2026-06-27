from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RiskObserver:
    """Risk observer computing rho, chi_R, chi_Auto and action."""
    weights: dict[str, float]
    theta: list[float]
    last_report: dict[str, Any] = field(default_factory=dict)

    def decide(self, E_model: Any, predicted_risk: float) -> dict[str, Any]:
        """Compute application risk and select action."""
        metadata = getattr(E_model, 'metadata', {}) or {}
        diagnostics = metadata.get('diagnostics', [])
        reduction_loss = float(getattr(E_model, 'reduction_loss', 0.0))
        uncertainty = float(getattr(E_model, 'u', getattr(E_model, 'uncertainty', 0.0)))
        i_pre = float(metadata.get('I_pre', 0.8))
        diagnostic_flag = 1.0 if diagnostics else 0.0
        rho = round(
            self.weights.get('predicted_risk', 0.5) * predicted_risk
            + self.weights.get('uncertainty', 0.2) * uncertainty
            + self.weights.get('interpretability_gap', 0.1) * (1 - i_pre)
            + self.weights.get('reduction_loss', 0.1) * reduction_loss
            + self.weights.get('diagnostic', 0.1) * diagnostic_flag,
            6,
        )
        chi_r = 1 if diagnostics else 0
        chi_auto = chi_r == 0 and rho < self.theta[1]
        if chi_r:
            action = 'block'
        elif rho < self.theta[0]:
            action = 'accept'
        elif rho < self.theta[1]:
            action = 'lower_confidence'
        elif rho < self.theta[2]:
            action = 'request_more_data'
        elif rho < self.theta[3]:
            action = 'defer_to_human'
        else:
            action = 'block'
        self.last_report = {'rho': rho, 'chi_R': chi_r, 'chi_Auto': chi_auto, 'action': action, 'diagnostics': diagnostics}
        return self.last_report

    def to_report(self) -> dict[str, Any]:
        """Return last JSON-compatible observer decision."""
        return dict(self.last_report)
