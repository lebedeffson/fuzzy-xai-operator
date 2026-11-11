"""Independent exact empirical-interventional Shapley audit for small tabular inputs.

The characteristic function is v(S) = mean_{r in B} f_t(x_S, r_{not S}),
for a FIXED scalar model output and an explicitly supplied, finite,
empirical background B. This is not conditional/causal SHAP or an
arbitrary-explainer fidelity certificate.

The enumerator is exponential and deliberately limited to <=8 features
and <=8192 model evaluation rows. The supplied model callback and declared
identities are not cryptographically authenticated by this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Real

from .evidence_identity import (
    LocalEvidenceObservation,
    LocalEvidenceStatus,
    LocalExplanationEvidence,
)
from .local_effects import hash_numeric_input

EXACT_INTERVENTIONAL_PROTOCOL = "exact_empirical_interventional_shapley_v1"
MAX_EXACT_FEATURES = 8
MAX_EXACT_SCORE_ROWS = 8192
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _finite_vector(
    values: Sequence[float], *, width: int, label: str
) -> tuple[float, ...]:
    if len(values) != width:
        raise ValueError(f"{label} has wrong feature width")
    if any(
        isinstance(item, bool)
        or not isinstance(item, Real)
        or not math.isfinite(item)
        for item in values
    ):
        raise ValueError(f"{label} must have finite numeric values")
    return tuple(float(item) for item in values)


def hash_empirical_background(
    feature_names: tuple[str, ...],
    background_rows: Sequence[Sequence[float]],
) -> str:
    """Digest *all* ordered background rows, not merely their feature mean."""

    if not feature_names or len(feature_names) != len(set(feature_names)):
        raise ValueError("feature axis must be unique and nonempty")
    if any(not isinstance(name, str) or not name.strip() for name in feature_names):
        raise ValueError("feature axis must contain nonblank names")
    if len(background_rows) == 0:
        raise ValueError("background must contain at least one row")
    rows = tuple(
        _finite_vector(row, width=len(feature_names), label="background row")
        for row in background_rows
    )
    payload = {
        "schema": EXACT_INTERVENTIONAL_PROTOCOL,
        "feature_names": feature_names,
        "rows": rows,
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        allow_nan=False, ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ExactShapleyEvidence:
    """Separately probed score-game evidence for a fixed scalar output."""

    protocol: str | None = None
    object_id: str | None = None
    model_version: str | None = None
    target_class: str | None = None
    score_channel: str | None = None
    feature_names: tuple[str, ...] | None = None
    attributions: tuple[float, ...] | None = None
    baseline_score: float | None = None
    explained_score: float | None = None
    input_sha256: str | None = None
    background_sha256: str | None = None
    background_rows: int | None = None
    score_evaluations: int | None = None


def collect_exact_interventional_shapley(
    *,
    object_id: str,
    model_version: str,
    target_class: str,
    feature_names: tuple[str, ...],
    score_channel: str | None = None,
    input_values: Sequence[float],
    background_rows: Sequence[Sequence[float]],
    target_scores: Callable[[tuple[tuple[float, ...], ...]], Sequence[float]],
) -> ExactShapleyEvidence:
    """Exhaustively evaluate one fixed-output, empirical interventional game.

    The callback receives 2**d * len(background_rows) vectors in one batch.
    It must return the *same selected target score*, not an entire
    multiclass matrix or different target per row. Verify its source and
    target-axis binding in the trusted adapter.
    """

    if not all(isinstance(x, str) and x for x in (object_id, model_version, target_class)):
        raise ValueError("object/model/target identity is required")
    width = len(feature_names)
    if width < 1 or width > MAX_EXACT_FEATURES:
        raise ValueError("exact Shapley is supported for 1..8 features only")
    observed = _finite_vector(input_values, width=width, label="input")
    input_digest = hash_numeric_input(feature_names, observed)
    background_digest = hash_empirical_background(feature_names, background_rows)
    background = tuple(
        _finite_vector(row, width=width, label="background row")
        for row in background_rows
    )
    batch_width = len(background)
    n_coalitions = 1 << width
    n_score_rows = n_coalitions * batch_width
    if n_score_rows > MAX_EXACT_SCORE_ROWS:
        raise ValueError("exact Shapley evaluation budget exceeded")

    rows = tuple(
        tuple(
            observed[index] if (mask & (1 << index)) else bg[index]
            for index in range(width)
        )
        for mask in range(n_coalitions)
        for bg in background
    )
    raw_scores = tuple(target_scores(rows))
    if len(raw_scores) != n_score_rows:
        raise ValueError("source model returned the wrong score count")
    if any(
        isinstance(score, bool)
        or not isinstance(score, Real)
        or not math.isfinite(score)
        for score in raw_scores
    ):
        raise ValueError("source model must return finite scalar target scores")
    scores = tuple(float(score) for score in raw_scores)
    full_mask_scores = scores[-batch_width:]
    if max(full_mask_scores) - min(full_mask_scores) > 1e-9:
        raise ValueError("source model is non-deterministic on the repeated input")

    values = tuple(
        math.fsum(scores[mask * batch_width : (mask + 1) * batch_width])
        / batch_width
        for mask in range(n_coalitions)
    )
    shapley_values = []
    for index in range(width):
        bit = 1 << index
        terms = []
        for mask in range(n_coalitions):
            if mask & bit:
                continue
            cardinality = mask.bit_count()
            weight = 1.0 / (width * math.comb(width - 1, cardinality))
            terms.append(weight * (values[mask | bit] - values[mask]))
        shapley_values.append(math.fsum(terms))
    return ExactShapleyEvidence(
        protocol=EXACT_INTERVENTIONAL_PROTOCOL,
        object_id=object_id,
        model_version=model_version,
        target_class=target_class,
        score_channel=score_channel,
        feature_names=feature_names,
        attributions=tuple(shapley_values),
        baseline_score=values[0],
        explained_score=values[-1],
        input_sha256=input_digest,
        background_sha256=background_digest,
        background_rows=batch_width,
        score_evaluations=n_score_rows,
    )


def verify_exact_interventional_shapley(
    explanation: LocalExplanationEvidence | None,
    evidence: ExactShapleyEvidence | None,
    *,
    absolute_tolerance: float = 1e-6,
) -> LocalEvidenceObservation:
    """Reject game disagreements; review missing/unsupported evidence.

    Certification means exact agreement with the declared empirical
    interventional game, not causal correctness, real-world truth or
    correctness of another explanation game.
    """

    if not math.isfinite(absolute_tolerance) or absolute_tolerance < 0:
        raise ValueError("absolute_tolerance must be finite and nonnegative")
    if explanation is None or evidence is None:
        return LocalEvidenceObservation(
            status=LocalEvidenceStatus.INSUFFICIENT_EVIDENCE,
            missing_fields=(
                ("local_explanation_evidence",)
                if explanation is None else ("exact_shapley_evidence",)
            ),
            contradictions=(),
        )
    missing: list[str] = []
    contradictions: list[str] = []
    if not evidence.protocol:
        missing.append("exact_shapley_protocol")
    elif evidence.protocol != EXACT_INTERVENTIONAL_PROTOCOL:
        missing.append("supported_exact_shapley_protocol")
    # An exact interventional oracle cannot treat disagreement with a
    # differently defined, conditional or approximate method as a fault.
    # Unsupported games require REVIEW; identity conflicts still BLOCK.
    compatible_game = evidence.protocol == EXACT_INTERVENTIONAL_PROTOCOL
    if not explanation.attribution_protocol:
        missing.append("explanation_attribution_protocol")
        compatible_game = False
    elif explanation.attribution_protocol != EXACT_INTERVENTIONAL_PROTOCOL:
        missing.append("supported_explanation_attribution_protocol")
        compatible_game = False
    for field in ("object_id", "model_version", "target_class", "score_channel"):
        actual = getattr(evidence, field)
        expected = getattr(explanation, field)
        if not actual or not expected:
            missing.append(field)
        elif actual != expected:
            contradictions.append(f"{field}_mismatch")
    if evidence.feature_names is None or explanation.feature_names is None:
        missing.append("feature_names")
    elif evidence.feature_names != explanation.feature_names:
        contradictions.append("feature_axis_mismatch")

    for name, declared, actual in (
        ("input_sha256", explanation.input_sha256, evidence.input_sha256),
        ("background_sha256", explanation.background_sha256, evidence.background_sha256),
    ):
        if not declared:
            missing.append(f"explanation_{name}")
        elif not _SHA256.fullmatch(declared):
            contradictions.append(f"invalid_explanation_{name}")
        if not actual:
            missing.append(f"model_{name}")
        elif not _SHA256.fullmatch(actual):
            contradictions.append(f"invalid_model_{name}")
        if declared and actual and declared != actual:
            contradictions.append(f"{name}_mismatch")

    def finite_scalar(value: float | None, name: str) -> float | None:
        if value is None:
            missing.append(name)
            return None
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            contradictions.append(f"invalid_{name}")
            return None
        return float(value)

    baseline = finite_scalar(evidence.baseline_score, "baseline_score")
    explained = finite_scalar(evidence.explained_score, "explained_score")
    if evidence.background_rows is None or evidence.score_evaluations is None:
        missing.append("probe_budget_metadata")
    elif (
        isinstance(evidence.background_rows, bool)
        or isinstance(evidence.score_evaluations, bool)
        or not isinstance(evidence.background_rows, int)
        or not isinstance(evidence.score_evaluations, int)
        or evidence.background_rows <= 0
        or evidence.score_evaluations <= 0
    ):
        contradictions.append("invalid_probe_budget_metadata")
    elif evidence.feature_names is not None and (
        len(evidence.feature_names) > MAX_EXACT_FEATURES
        or evidence.score_evaluations != (1 << len(evidence.feature_names)) * evidence.background_rows
        or evidence.score_evaluations > MAX_EXACT_SCORE_ROWS
    ):
        contradictions.append("incompatible_probe_budget_metadata")

    expected_width = (
        len(explanation.feature_names) if explanation.feature_names is not None else None
    )
    claimed, measured = explanation.contributions, evidence.attributions
    if claimed is None or measured is None:
        missing.append("attribution_vector")
    elif expected_width is None:
        missing.append("feature_axis")
    elif len(claimed) != expected_width or len(measured) != expected_width:
        contradictions.append("attribution_width_mismatch")
    elif any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        for value in (*claimed, *measured)
    ):
        contradictions.append("invalid_attribution_values")
    elif compatible_game:
        if any(abs(float(a) - float(b)) > absolute_tolerance for a, b in zip(claimed, measured, strict=True)):
            contradictions.append("shapley_attribution_mismatch")
        if baseline is not None and explained is not None and abs(
            math.fsum(float(v) for v in claimed) - (explained - baseline)
        ) > absolute_tolerance:
            contradictions.append("shapley_efficiency_mismatch")
    status = (
        LocalEvidenceStatus.INCOMPATIBLE if contradictions
        else LocalEvidenceStatus.INSUFFICIENT_EVIDENCE if missing
        else LocalEvidenceStatus.CERTIFIED
    )
    return LocalEvidenceObservation(
        status=status,
        missing_fields=tuple(dict.fromkeys(missing)),
        contradictions=tuple(dict.fromkeys(contradictions)),
    )
