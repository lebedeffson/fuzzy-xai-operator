from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from fuzzyxai.core.explain_plan import ExplainPlan

from .risk_function import compute_application_risk
from .risk_observer_config import (
    load_calibrated_risk_weights,
    load_thresholds,
    weights_from_explain_plan,
)


@dataclass(frozen=True)
class ObserverDecision:
    rho: float
    action: str
    reason: str
    thresholds: tuple[float, float, float, float]


class RiskAwareObserver:
    """Threshold-based observer using calibrated risk weights."""

    def __init__(self, plan: ExplainPlan | None = None, calibration_path: str = 'reports/chapter5/chapter5_experiments.json') -> None:
        self.plan = plan
        self.weights = weights_from_explain_plan(plan, fallback=load_calibrated_risk_weights(calibration_path)) if plan else load_calibrated_risk_weights(calibration_path)
        self.thresholds = load_thresholds(plan, calibration_path)

    def decide(self, rho: float, has_critical_rupture: bool = False) -> str:
        t1, t2, t3, t4 = self.thresholds
        if has_critical_rupture:
            return 'block'
        if rho < t1:
            return 'accept'
        if rho < t2:
            return 'lower_confidence'
        if rho < t3:
            return 'request_more_data'
        if rho < t4:
            return 'defer_to_human'
        return 'block'

    def evaluate(
        self,
        predicted_risk: float,
        uncertainty: float,
        pre_interpretability: float,
        reduction_loss: float = 0.0,
        diagnostics: Sequence[str] | None = None,
    ) -> ObserverDecision:
        diagnostics = list(diagnostics or [])
        breakdown = compute_application_risk(
            predicted_risk,
            uncertainty,
            pre_interpretability,
            reduction_loss,
            diagnostics,
            self.weights,
        )
        action = self.decide(float(breakdown.rho), bool(diagnostics))
        reason = 'critical rupture' if diagnostics else 'threshold policy'
        return ObserverDecision(rho=float(breakdown.rho), action=action, reason=reason, thresholds=self.thresholds)
