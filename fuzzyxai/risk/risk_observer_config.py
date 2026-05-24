from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from fuzzyxai.core.explain_plan import ExplainPlan

from .decision_policy import RiskPolicy
from .risk_function import normalize_risk_weights

DEFAULT_CALIBRATED_WEIGHTS_PATH = Path('reports/chapter5/chapter5_experiments.json')


def load_calibrated_risk_weights(path: str | Path = DEFAULT_CALIBRATED_WEIGHTS_PATH) -> dict[str, float]:
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    return normalize_risk_weights(data['calibration']['weights'])


def weights_from_explain_plan(plan: ExplainPlan, fallback: Mapping[str, float] | None = None) -> dict[str, float]:
    weights = plan.metadata.get('risk_weights') if isinstance(plan.metadata, dict) else None
    if weights is None:
        weights = fallback
    return normalize_risk_weights(weights)


def policy_from_calibration(
    path: str | Path = DEFAULT_CALIBRATED_WEIGHTS_PATH,
    *,
    theta_mid: float = 0.35,
    theta_high: float = 0.65,
    block_on_diagnostic: bool = True,
) -> RiskPolicy:
    return RiskPolicy(
        theta_mid=theta_mid,
        theta_high=theta_high,
        block_on_diagnostic=block_on_diagnostic,
        risk_weights=load_calibrated_risk_weights(path),
    )


def attach_risk_weights_to_plan(plan: ExplainPlan, weights: Mapping[str, float]) -> ExplainPlan:
    metadata: dict[str, Any] = dict(plan.metadata)
    metadata['risk_weights'] = normalize_risk_weights(weights)
    return ExplainPlan(
        beta=dict(plan.beta),
        lambda_=dict(plan.lambda_),
        eta=dict(plan.eta),
        i_min=plan.i_min,
        activation_threshold=plan.activation_threshold,
        epsilon=plan.epsilon,
        metadata=metadata,
    )
