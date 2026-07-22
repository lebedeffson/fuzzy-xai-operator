"""Deployment-envelope shift risk head."""

from __future__ import annotations

from fuzzyxai.practical_controller import PredictionArtifact

from .calibration import CalibratedRiskHead

SHIFT_FEATURES = ("shift_score", "reference_population_distance", "artifact_age", "version_distance", "rare_group")


def shift_feature_map(
    artifact: PredictionArtifact,
    *,
    reference_population_distance: float = 0.0,
    artifact_age: float = 0.0,
    version_distance: float = 0.0,
) -> dict[str, float]:
    return {
        "shift_score": artifact.shift_score,
        "reference_population_distance": reference_population_distance,
        "artifact_age": artifact_age,
        "version_distance": version_distance,
        "rare_group": float(artifact.rare_group),
    }


def estimate_shift_risk(
    head: CalibratedRiskHead,
    artifact: PredictionArtifact,
    *,
    reference_population_distance: float = 0.0,
    artifact_age: float = 0.0,
    version_distance: float = 0.0,
) -> float:
    if head.target_name != "outside_deployment_envelope":
        raise ValueError("shift head target must be outside_deployment_envelope")
    return head.predict(shift_feature_map(artifact, reference_population_distance=reference_population_distance, artifact_age=artifact_age, version_distance=version_distance))
