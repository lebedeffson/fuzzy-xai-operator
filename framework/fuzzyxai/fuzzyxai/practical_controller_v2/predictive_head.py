"""Predictive-error risk head."""

from __future__ import annotations

from fuzzyxai.practical_controller import PredictionArtifact

from .calibration import CalibratedRiskHead

PREDICTIVE_FEATURES = ("calibrated_confidence", "entropy", "prediction_margin", "calibration_residual", "boundary_distance", "model_disagreement")


def predictive_feature_map(artifact: PredictionArtifact) -> dict[str, float]:
    return {
        "calibrated_confidence": 1.0 - artifact.confidence,
        "entropy": artifact.entropy,
        "prediction_margin": 1.0 - artifact.prediction_margin,
        "calibration_residual": artifact.calibration_residual,
        "boundary_distance": 1.0 - artifact.boundary_distance,
        "model_disagreement": artifact.model_disagreement,
    }


def estimate_predictive_risk(head: CalibratedRiskHead, artifact: PredictionArtifact) -> float:
    if head.target_name != "model_error":
        raise ValueError("predictive head target must be model_error")
    return head.predict(predictive_feature_map(artifact))
