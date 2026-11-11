"""Evidence-first identity checks for *local* explanation artifacts.

This observer validates declared bindings and shapes. It does not establish
fidelity, causality, or the truth of upstream provenance declarations.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class LocalEvidenceStatus(str, Enum):
    CERTIFIED = "certified"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    INCOMPATIBLE = "incompatible"


@dataclass(frozen=True)
class LocalExplanationEvidence:
    """A single target-class local attribution vector with source binding.

    Multi-class explainers must select and name one target-class vector;
    flattened class-by-feature matrices are not local feature vectors.
    A SHA256 value is an identity declaration, not independent verification
    of bytes: canonical source verification belongs to the upstream adapter.
    Missing attributes remain None so that the observer can abstain explicitly.
    """

    object_id: str | None = None
    model_version: str | None = None
    target_class: str | None = None
    scope: str | None = None
    feature_names: tuple[str, ...] | None = None
    contributions: tuple[float, ...] | None = None
    source_sha256: str | None = None
    input_sha256: str | None = None


@dataclass(frozen=True)
class LocalEvidenceObservation:
    status: LocalEvidenceStatus
    missing_fields: tuple[str, ...]
    contradictions: tuple[str, ...]

    @property
    def certified(self) -> bool:
        return self.status is LocalEvidenceStatus.CERTIFIED


def observe_local_explanation(
    evidence: LocalExplanationEvidence | None,
    *,
    expected_object_id: str,
    expected_model_version: str,
    expected_target_class: str,
    expected_feature_names: tuple[str, ...] | None,
    expected_source_sha256: str,
) -> LocalEvidenceObservation:
    """Check artifact identity and attribution shape without consulting labels.

    Conflicting evidence is INCOMPATIBLE (must block). Absent evidence is
    INSUFFICIENT_EVIDENCE (must review/abstain). Only a complete and
    consistent record is CERTIFIED for *structural* use. No model-accuracy
    or explanation-faithfulness claim follows from this certification.
    """

    missing: list[str] = []
    contradictions: list[str] = []
    if evidence is None:
        return LocalEvidenceObservation(
            status=LocalEvidenceStatus.INSUFFICIENT_EVIDENCE,
            missing_fields=("local_explanation_evidence",),
            contradictions=(),
        )

    def compare(field: str, actual: str | None, expected: str) -> None:
        if not actual:
            missing.append(field)
        elif actual != expected:
            contradictions.append(f"{field}_mismatch")

    compare("object_id", evidence.object_id, expected_object_id)
    compare("model_version", evidence.model_version, expected_model_version)
    compare("target_class", evidence.target_class, expected_target_class)

    if not evidence.scope:
        missing.append("scope")
    elif evidence.scope != "local":
        contradictions.append("non_local_scope")

    if evidence.source_sha256 is None or evidence.source_sha256 == "":
        missing.append("source_sha256")
    elif not _SHA256.fullmatch(evidence.source_sha256):
        contradictions.append("invalid_source_sha256")
    elif evidence.source_sha256 != expected_source_sha256:
        contradictions.append("source_sha256_mismatch")

    def valid_axis(names: tuple[str, ...]) -> bool:
        return bool(names) and all(isinstance(name, str) and bool(name.strip()) for name in names) and len(names) == len(set(names))

    expected_valid = expected_feature_names is not None and valid_axis(expected_feature_names)
    if expected_feature_names is None:
        missing.append("expected_feature_names")
    elif not expected_valid:
        contradictions.append("invalid_expected_feature_axis")

    actual_valid = evidence.feature_names is not None and valid_axis(evidence.feature_names)
    if evidence.feature_names is None:
        missing.append("feature_names")
    elif not actual_valid:
        contradictions.append("invalid_feature_axis")

    if expected_valid and actual_valid and evidence.feature_names != expected_feature_names:
        contradictions.append("feature_axis_mismatch")

    values = evidence.contributions
    if values is None or len(values) == 0:
        missing.append("contributions")
    elif any(isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) for value in values):
        contradictions.append("invalid_contribution_values")
    elif actual_valid and len(values) != len(evidence.feature_names):
        contradictions.append("contribution_width_mismatch")

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



def observe_input_binding(
    *,
    prediction_input_sha256: str | None,
    explanation_input_sha256: str | None,
    model_effect_input_sha256: str | None,
    expect_model_effects: bool,
) -> LocalEvidenceObservation:
    """Compare input hashes from different route stages, without using labels.

    The hash covers ordered feature names as well as feature values.
    This structural check does not authenticate the source adapter.
    """

    missing: list[str] = []
    contradictions: list[str] = []

    def checked(name: str, digest: str | None) -> str | None:
        if not digest:
            missing.append(name)
            return None
        if not _SHA256.fullmatch(digest):
            contradictions.append(f"invalid_{name}")
            return None
        return digest

    expected = checked("prediction_input_sha256", prediction_input_sha256)
    local = checked("explanation_input_sha256", explanation_input_sha256)
    observed = (
        checked("model_effect_input_sha256", model_effect_input_sha256)
        if expect_model_effects or model_effect_input_sha256 is not None
        else None
    )
    if expected is not None and local is not None and expected != local:
        contradictions.append("explanation_input_sha256_mismatch")
    if expected is not None and observed is not None and expected != observed:
        contradictions.append("model_effect_input_sha256_mismatch")

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
