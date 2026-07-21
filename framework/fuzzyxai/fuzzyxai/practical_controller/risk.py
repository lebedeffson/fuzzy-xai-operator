"""Predictive and explanation-route risk estimators."""

from __future__ import annotations

import math

import numpy as np

from .calibration import apply_calibrator
from .contracts import ExplanationArtifact, PracticalPolicy, PredictionArtifact, RouteArtifacts


def predictive_features(artifact: PredictionArtifact) -> tuple[float, ...]:
    return (
        1.0 - artifact.confidence,
        artifact.entropy,
        1.0 - artifact.prediction_margin,
        artifact.calibration_residual,
        1.0 - artifact.boundary_distance,
        artifact.model_disagreement,
        artifact.shift_score,
        float(artifact.rare_group),
    )


def route_features(explanation: ExplanationArtifact, route: RouteArtifacts, *, required_channels: tuple[str, ...]) -> tuple[float, ...]:
    missing_fraction = len(set(required_channels) - set(route.observed_provenance_channels)) / max(1, len(required_channels))
    provenance_incompleteness = max(missing_fraction, 1.0 - len(route.observed_provenance_channels) / max(1, len(explanation.available_channels)))
    return (
        explanation.explainer_disagreement,
        explanation.seed_instability,
        explanation.bootstrap_instability,
        explanation.perturbation_instability,
        float(np.clip(provenance_incompleteness, 0.0, 1.0)),
        float(route.route_fault_type is not None or route.natural_failure is not None),
        explanation.representation_loss,
        explanation.rule_redundancy,
        explanation.conflict_severity,
        float(missing_fraction > 0.0),
    )


def estimate_predictive_risk(policy: PracticalPolicy, artifact: PredictionArtifact) -> float:
    raw = _logistic(policy.predictive_intercept + float(np.dot(policy.predictive_weights, predictive_features(artifact))))
    return float(apply_calibrator(policy.calibration_method, policy.calibration_parameters, raw))


def estimate_route_risk(
    policy: PracticalPolicy,
    explanation: ExplanationArtifact,
    route: RouteArtifacts,
    *,
    required_channels: tuple[str, ...],
) -> float:
    values = route_features(explanation, route, required_channels=required_channels)
    return _logistic(policy.route_intercept + float(np.dot(policy.route_weights, values)))


def combine_operational_risk(predictive_risk: float, route_risk: float) -> float:
    """Probability union keeps either independently high risk visible."""
    return float(np.clip(1.0 - (1.0 - predictive_risk) * (1.0 - route_risk), 0.0, 1.0))


def risk_interval(risk: float, *, effective_sample_size: int = 100) -> tuple[float, float]:
    standard_error = math.sqrt(max(risk * (1.0 - risk), 1e-9) / max(2, effective_sample_size))
    return max(0.0, risk - 1.96 * standard_error), min(1.0, risk + 1.96 * standard_error)


def _logistic(value: float) -> float:
    return float(1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, value)))))
