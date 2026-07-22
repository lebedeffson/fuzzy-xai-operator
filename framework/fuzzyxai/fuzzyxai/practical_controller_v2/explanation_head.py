"""Explanation-instability risk head."""

from __future__ import annotations

from fuzzyxai.practical_controller import ExplanationArtifact

from .calibration import CalibratedRiskHead

EXPLANATION_FEATURES = ("explainer_disagreement", "seed_instability", "bootstrap_instability", "perturbation_instability", "representation_loss", "source_conflict_count")


def explanation_feature_map(artifact: ExplanationArtifact, *, source_conflict_count: int = 0) -> dict[str, float]:
    return {
        "explainer_disagreement": artifact.explainer_disagreement,
        "seed_instability": artifact.seed_instability,
        "bootstrap_instability": artifact.bootstrap_instability,
        "perturbation_instability": artifact.perturbation_instability,
        "representation_loss": artifact.representation_loss,
        "source_conflict_count": min(1.0, source_conflict_count / 5.0),
    }


def estimate_explanation_risk(head: CalibratedRiskHead, artifact: ExplanationArtifact, *, source_conflict_count: int = 0) -> float:
    if head.target_name != "explanation_unstable_or_incomplete":
        raise ValueError("explanation head target is invalid")
    return head.predict(explanation_feature_map(artifact, source_conflict_count=source_conflict_count))
