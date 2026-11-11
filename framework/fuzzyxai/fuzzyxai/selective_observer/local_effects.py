"""Optional independent score-response audit for local *linear additive* explanations.

This protocol is only valid when replacing a single feature with its fixed
reference value yields an additive target-logit change. It is not a generic
faithfulness certificate for nonlinear, interacting, or model-agnostic XAI.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum

from .evidence_identity import (
    LocalEvidenceObservation,
    LocalEvidenceStatus,
    LocalExplanationEvidence,
)

SUPPORTED_PROTOCOL = "linear_logit_single_feature_reference_v1"


def hash_numeric_input(feature_names: tuple[str, ...], values: Sequence[float]) -> str:
    """Canonical digest of ordered finite numeric features, not a signature."""

    if not feature_names or len(feature_names) != len(values):
        raise ValueError("input vector must match nonempty feature axis")
    if len(set(feature_names)) != len(feature_names) or any(
        not isinstance(name, str) or not name.strip() for name in feature_names
    ):
        raise ValueError("input feature names must be unique and nonempty")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in values
    ):
        raise ValueError("input values must be finite numeric scalars")
    payload = {
        "schema": "numeric-tabular-input-v1",
        "feature_names": feature_names,
        "values": tuple(float(value) for value in values),
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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
    input_sha256: str | None = None
    reference_sha256: str | None = None


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
        if baseline is not None and actual is not None and abs(sum(declared) - (actual - baseline)) > absolute_tolerance:
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


def collect_linear_reference_effects(
    *,
    object_id: str,
    model_version: str,
    target_class: str,
    feature_names: tuple[str, ...],
    input_values: Sequence[float],
    reference_values: Sequence[float],
    target_logit: Callable[[tuple[float, ...]], float],
) -> LocalEffectEvidence:
    """Probe a caller-supplied trusted target-logit function feature by feature.

    Only reference replacement is performed. The caller must select the
    correct target class and connect this callable to the source model;
    this helper cannot authenticate the callable or source-model bytes.
    """

    width = len(feature_names)
    if not object_id or not model_version or not target_class:
        raise ValueError("source identity and target class are required")
    if not width or any(not isinstance(x, str) or not x.strip() for x in feature_names):
        raise ValueError("feature axis must be nonempty")
    if len(set(feature_names)) != width:
        raise ValueError("feature names must be unique")
    if len(input_values) != width or len(reference_values) != width:
        raise ValueError("reference and input widths must equal feature axis")
    if any(
        isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x)
        for x in (*input_values, *reference_values)
    ):
        raise ValueError("reference and input values must be finite")

    observed = tuple(float(x) for x in input_values)
    reference = tuple(float(x) for x in reference_values)

    def evaluate(values: tuple[float, ...]) -> float:
        result = target_logit(values)
        if isinstance(result, bool) or not isinstance(result, (float, int)) or not math.isfinite(result):
            raise ValueError("target-logit function must return a finite scalar")
        return float(result)

    original_score = evaluate(observed)
    baseline_score = evaluate(reference)
    contributions: list[float] = []
    for index in range(width):
        changed = list(observed)
        changed[index] = reference[index]
        contributions.append(original_score - evaluate(tuple(changed)))
    return LocalEffectEvidence(
        protocol=SUPPORTED_PROTOCOL,
        object_id=object_id,
        model_version=model_version,
        target_class=target_class,
        feature_names=feature_names,
        effect_on_score=tuple(contributions),
        reference_score=baseline_score,
        explained_score=original_score,
        input_sha256=hash_numeric_input(feature_names, observed),
        reference_sha256=hash_numeric_input(feature_names, reference),
    )



def collect_linear_reference_effects_batched(
    *,
    object_id: str,
    model_version: str,
    target_class: str,
    feature_names: tuple[str, ...],
    input_values: Sequence[float],
    reference_values: Sequence[float],
    target_logits: Callable[[tuple[tuple[float, ...], ...]], Sequence[float]],
) -> LocalEffectEvidence:
    """Evaluate all d+2 reference probes in a single source-model call.

    The computational number of score evaluations is still O(d); only the
    Python-level inference-call count is reduced. This protocol is valid
    solely for linear-additive target logits and trusted source adapters.
    """

    if not object_id or not model_version or not target_class:
        raise ValueError("source identity and target class are required")
    input_digest = hash_numeric_input(feature_names, input_values)
    hash_numeric_input(feature_names, reference_values)
    observed = tuple(float(x) for x in input_values)
    reference = tuple(float(x) for x in reference_values)
    probe_rows: list[tuple[float, ...]] = [observed, reference]
    for feature_index in range(len(feature_names)):
        changed = list(observed)
        changed[feature_index] = reference[feature_index]
        probe_rows.append(tuple(changed))

    scores = tuple(target_logits(tuple(probe_rows)))
    if len(scores) != len(probe_rows):
        raise ValueError("batched model must produce exactly one score per probe row")
    if any(
        isinstance(score, bool)
        or not isinstance(score, (float, int))
        or not math.isfinite(score)
        for score in scores
    ):
        raise ValueError("batched target-logit function must return finite numeric scores")
    original_score = float(scores[0])
    reference_score = float(scores[1])
    return LocalEffectEvidence(
        protocol=SUPPORTED_PROTOCOL,
        object_id=object_id,
        model_version=model_version,
        target_class=target_class,
        feature_names=feature_names,
        effect_on_score=tuple(original_score - float(score) for score in scores[2:]),
        reference_score=reference_score,
        explained_score=original_score,
        input_sha256=input_digest,
        reference_sha256=hash_numeric_input(feature_names, reference),
    )


class ReferenceRebaseStatus(str, Enum):
    """A proposal is not a recertified explanation."""

    CANDIDATE = "candidate"
    INCOMPATIBLE = "incompatible"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(frozen=True)
class LinearReferenceRebaseProposal:
    status: ReferenceRebaseStatus
    contributions: tuple[float, ...] | None
    new_reference_sha256: str | None
    original_max_residual: float | None
    missing_fields: tuple[str, ...]
    contradictions: tuple[str, ...]


def propose_linear_reference_rebase(
    evidence: LocalExplanationEvidence | None,
    *,
    feature_names: tuple[str, ...],
    input_values: Sequence[float] | None,
    original_reference: Sequence[float] | None,
    proposed_reference: Sequence[float] | None,
    signed_raw_coefficients: Sequence[float] | None,
    absolute_tolerance: float = 1e-6,
) -> LinearReferenceRebaseProposal:
    """Propose a change of reference for a *known additive target logit*.

    For f_t(x)=b_t+sum_j w_j*x_j, the local contribution at reference r is
    phi_j(x;r)=w_j*(x_j-r_j). Rebase from r0 to r1 using
    phi_j(x;r1)=phi_j(x;r0)+w_j*(r0_j-r1_j).

    The adapter must supply independently verified target-class coefficients
    expressed in the original feature coordinate system. The candidate
    remains UNTRUSTED until source-model effects are collected at the new
    reference, the canonical explanation record is rehashed, and the full
    practical-controller guard certifies it. This helper cannot authenticate
    model bytes or silently repair an already-issued explanation.
    """

    if not math.isfinite(absolute_tolerance) or absolute_tolerance < 0:
        raise ValueError("absolute_tolerance must be finite and nonnegative")

    missing: list[str] = []
    contradictions: list[str] = []
    if evidence is None:
        return LinearReferenceRebaseProposal(
            ReferenceRebaseStatus.INSUFFICIENT_EVIDENCE, None, None, None,
            ("local_explanation_evidence",), (),
        )

    width = len(feature_names)
    if not width or any(not isinstance(name, str) or not name.strip() for name in feature_names) or len(set(feature_names)) != width:
        contradictions.append("invalid_feature_axis")
    if evidence.feature_names is None:
        missing.append("evidence_feature_axis")
    elif evidence.feature_names != feature_names:
        contradictions.append("feature_axis_mismatch")
    if evidence.scope is None:
        missing.append("scope")
    elif evidence.scope != "local":
        contradictions.append("non_local_scope")
    for field in ("object_id", "model_version", "target_class"):
        if not getattr(evidence, field):
            missing.append(field)

    def vector(value: Sequence[float] | None, field: str) -> tuple[float, ...] | None:
        if value is None:
            missing.append(field)
            return None
        if len(value) != width:
            contradictions.append(f"{field}_width_mismatch")
            return None
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value):
            contradictions.append(f"invalid_{field}")
            return None
        return tuple(float(x) for x in value)

    x = vector(input_values, "input_values")
    r0 = vector(original_reference, "original_reference")
    r1 = vector(proposed_reference, "proposed_reference")
    w = vector(signed_raw_coefficients, "signed_raw_coefficients")
    original = vector(evidence.contributions, "contributions")

    if x is not None:
        if not evidence.input_sha256:
            missing.append("input_sha256")
        elif evidence.input_sha256 != hash_numeric_input(feature_names, x):
            contradictions.append("input_sha256_mismatch")
    if r0 is not None:
        if not evidence.reference_sha256:
            missing.append("reference_sha256")
        elif evidence.reference_sha256 != hash_numeric_input(feature_names, r0):
            contradictions.append("reference_sha256_mismatch")

    residual = None
    if all(value is not None for value in (x, r0, w, original)):
        assert x is not None and r0 is not None and w is not None and original is not None
        residual = max(
            abs(actual - coeff * (value - ref))
            for actual, coeff, value, ref in zip(original, w, x, r0, strict=True)
        )
        if residual > absolute_tolerance:
            contradictions.append("original_contribution_mismatch")

    status = (
        ReferenceRebaseStatus.INCOMPATIBLE if contradictions
        else ReferenceRebaseStatus.INSUFFICIENT_EVIDENCE if missing
        else ReferenceRebaseStatus.CANDIDATE
    )
    if status is not ReferenceRebaseStatus.CANDIDATE:
        return LinearReferenceRebaseProposal(
            status, None, None, residual,
            tuple(dict.fromkeys(missing)), tuple(dict.fromkeys(contradictions)),
        )
    assert original is not None and w is not None and r0 is not None and r1 is not None
    rebased = tuple(
        contribution + coeff * (old - new)
        for contribution, coeff, old, new in zip(original, w, r0, r1, strict=True)
    )
    if not all(math.isfinite(item) for item in rebased):
        return LinearReferenceRebaseProposal(
            ReferenceRebaseStatus.INCOMPATIBLE, None, None, residual, (),
            ("nonfinite_rebased_contributions",),
        )
    return LinearReferenceRebaseProposal(
        ReferenceRebaseStatus.CANDIDATE,
        rebased,
        hash_numeric_input(feature_names, r1),
        residual,
        (),
        (),
    )
