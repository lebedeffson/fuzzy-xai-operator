"""Optional independent score-response audit for local *linear additive* explanations.

This protocol is only valid when replacing a single feature with its fixed
reference value yields an additive target-logit change. It is not a generic
faithfulness certificate for nonlinear, interacting, or model-agnostic XAI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .evidence_identity import (
    LocalEvidenceObservation,
    LocalEvidenceStatus,
    LocalExplanationEvidence,
)

SUPPORTED_PROTOCOL = "linear_logit_single_feature_reference_v1"


@dataclass(frozen=True)
class LocalEffectEvidence:
    """Source-model measurements, collected independently of the explainer.

    effect_on_score[j] is the difference between the target-class score at
    the original input and the score when only feature j is replaced with
    the specified reference. reference_score is the model score at the
    all-reference input. All scores must be in the same target-logit space.
    The trusted model adapter must provide this evidence from actual calls;
    declaring values does not prove that any model was executed.
    """

    protocol: str | None = None
    object_id: str | None = None
    model_version: str | None = None
    target_class: str | None = None
    feature_names: tuple[str, ...] | None = None
    effect_on_score: tuple[float, ...] | None = None
    reference_score: float | None = None
    explained_score: float | None = None


def verify_additive_local_effects(
    explanation: LocalExplanationEvidence | None,
    evidence: LocalEffectEvidence | None,
    *,
    absolute_tolerance: float = 1e-6,
) -> LocalEvidenceObservation:
    """Fail closed on missing, unsupported, inconsistent or nonfinite effects.

    A successful result concerns a linear-additive score decomposition only.
    It neither estimates predictive accuracy nor validates causality.
    """

    if not math.isfinite(absolute_tolerance) or absolute_tolerance < 0.0:
        raise ValueError("absolute_tolerance must be nonnegative and finite")

    if explanation is None:
        return LocalEvidenceObservation(
            status=LocalEvidenceStatus.INSUFFICIENT_EVIDENCE,
            missing_fields=("local_explanation_evidence",),
            contradictions=(),
        )
    if evidence is None:
        return LocalEvidenceObservation(
            status=LocalEvidenceStatus.INSUFFICIENT_EVIDENCE,
            missing_fields=("observed_local_effects",),
            contradictions=(),
        )

    missing: list[str] = []
    contradictions: list[str] = []

    if evidence.protocol is None:
        missing.append("effect_protocol")
    elif evidence.protocol != SUPPORTED_PROTOCOL:
        missing.append("supported_effect_protocol")

    for field in ("object_id", "model_version", "target_class"):
        actual = getattr(evidence, field)
        expected = getattr(explanation, field)
        if not actual or not expected:
            missing.append(field)
        elif actual != expected:
            contradictions.append(f"{field}_mismatch")

    axis = evidence.feature_names
    if axis is None or explanation.feature_names is None:
        missing.append("feature_names")
    elif axis != explanation.feature_names:
        contradictions.append("feature_axis_mismatch")

    def finite_scalar(value: float | None, name: str) -> float | None:
        if value is None:
            missing.append(name)
            return None
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            contradictions.append(f"invalid_{name}")
            return None
        return float(value)

    baseline = finite_scalar(evidence.reference_score, "reference_score")
    actual = finite_scalar(evidence.explained_score, "explained_score")
    observed_effects = evidence.effect_on_score
    declared = explanation.contributions

    if observed_effects is None or declared is None:
        missing.append("effect_or_contribution_vector")
    elif axis is None or explanation.feature_names is None:
        missing.append("effect_axis")
    elif len(observed_effects) != len(axis) or len(declared) != len(axis):
        contradictions.append("effect_width_mismatch")
    elif any(
        isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
        for x in (*observed_effects, *declared)
    ):
        contradictions.append("nonfinite_or_invalid_effect")
    else:
        delta = [float(x) - float(y) for x, y in zip(declared, observed_effects, strict=True)]
        if any(abs(d) > absolute_tolerance for d in delta):
            contradictions.append("feature_effect_mismatch")
        if baseline is not None and actual is not None:
            if abs(sum(declared) - (actual - baseline)) > absolute_tolerance:
                contradictions.append("score_reconstruction_mismatch")

    status = (
        LocalEvidenceStatus.INCOMPATIBLE
        if contradictions
        else LocalEvidenceStatus.INSUFFICIENT_EVIDENCE
        if missing
        else LocalEvidenceStatus.CERTIFIED
    )
    return LocalEvidenceObservation(
        status=status,
        missing_fields=tuple(dict.fromkeys(missing)),
        contradictions=tuple(dict.fromkeys(contradictions)),
    )
