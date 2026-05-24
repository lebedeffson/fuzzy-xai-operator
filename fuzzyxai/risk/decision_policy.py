from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence

from .risk_function import clip01 as _clip01, compute_application_risk


class RiskAction(Enum):
    ACCEPT = 'accept'
    LOWER_CONFIDENCE = 'lower_confidence'
    REQUEST_MORE_DATA = 'request_more_data'
    DEFER_TO_HUMAN = 'defer_to_human'
    BLOCK = 'block'


@dataclass(frozen=True)
class RiskDecision:
    action: RiskAction
    risk_score: float
    corrected_confidence: float
    reason: str
    diagnostics: list[str] = field(default_factory=list)


@dataclass
class RiskPolicy:
    theta_mid: float = 0.35
    theta_high: float = 0.65
    block_on_diagnostic: bool = True
    risk_weights: Mapping[str, float] = field(default_factory=lambda: {
        'predicted_risk': 0.30,
        'uncertainty': 0.25,
        'reduction_loss': 0.15,
        'interpretability_gap': 0.20,
        'diagnostic': 0.10,
    })

    def risk_score(
        self,
        predicted_risk: float,
        uncertainty: float,
        interpretability: float,
        reduction_loss: float,
        diagnostics: Sequence[str] | None = None,
    ) -> float:
        return compute_application_risk(
            predicted_risk,
            uncertainty,
            interpretability,
            reduction_loss,
            diagnostics,
            self.risk_weights,
        ).rho

    def choose(
        self,
        predicted_risk: float,
        uncertainty: float,
        interpretability: float,
        reduction_loss: float,
        diagnostics: Sequence[str] | None = None,
    ) -> RiskDecision:
        diagnostics = list(diagnostics or [])
        if diagnostics and self.block_on_diagnostic:
            return RiskDecision(RiskAction.BLOCK, 1.0, 0.0, 'diagnostic state blocks automatic decision', diagnostics)

        rho = self.risk_score(predicted_risk, uncertainty, interpretability, reduction_loss, diagnostics)
        corrected = _clip01((1.0 - uncertainty) * (1.0 - rho) * interpretability * (1.0 - reduction_loss))
        if rho >= self.theta_high:
            return RiskDecision(RiskAction.DEFER_TO_HUMAN, rho, corrected, 'risk score exceeds theta_high', diagnostics)
        if rho >= self.theta_mid:
            if uncertainty >= 0.45:
                return RiskDecision(RiskAction.REQUEST_MORE_DATA, rho, corrected, 'medium risk with elevated uncertainty', diagnostics)
            return RiskDecision(RiskAction.LOWER_CONFIDENCE, rho, corrected, 'medium risk: accept only with lower confidence', diagnostics)
        return RiskDecision(RiskAction.ACCEPT, rho, corrected, 'risk score below theta_mid', diagnostics)
