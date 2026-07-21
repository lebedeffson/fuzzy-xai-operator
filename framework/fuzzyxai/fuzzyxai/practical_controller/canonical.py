"""Lossless canonical explanation storage and explicitly lossy presentation projections."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class CanonicalReason:
    reason_id: str
    identity: str
    raw_sign: int
    raw_magnitude: float
    raw_rank: int
    kind: str

    def __post_init__(self) -> None:
        if not self.reason_id or not self.identity or self.raw_sign not in {-1, 0, 1}:
            raise ValueError("canonical reason identity and sign are required")
        if self.raw_rank <= 0 or not np.isfinite(self.raw_magnitude):
            raise ValueError("canonical rank and magnitude must be finite")


@dataclass(frozen=True)
class CanonicalExplanation:
    source_payload: bytes
    source_sha256: str
    source_media_type: str
    explainer_parameters_json: str
    background_identity: str
    reasons: tuple[CanonicalReason, ...]
    raw_rule_conditions_json: str | None = None
    raw_mask_sha256: str | None = None

    def __post_init__(self) -> None:
        actual = hashlib.sha256(self.source_payload).hexdigest()
        if actual != self.source_sha256:
            raise ValueError("canonical payload hash differs from source explanation payload")
        if not self.source_media_type or not self.background_identity or not self.reasons:
            raise ValueError("canonical source metadata and reasons are required")
        json.loads(self.explainer_parameters_json)
        if self.raw_rule_conditions_json is not None:
            json.loads(self.raw_rule_conditions_json)
        if self.raw_mask_sha256 is not None and len(self.raw_mask_sha256) != 64:
            raise ValueError("raw mask identity must be a SHA256 digest")

    @classmethod
    def from_source(
        cls,
        source_payload: bytes,
        *,
        source_media_type: str,
        explainer_parameters: dict[str, object],
        background_identity: str,
        reasons: Sequence[CanonicalReason],
        raw_rule_conditions: object | None = None,
        raw_mask: bytes | None = None,
    ) -> "CanonicalExplanation":
        return cls(
            source_payload=bytes(source_payload),
            source_sha256=hashlib.sha256(source_payload).hexdigest(),
            source_media_type=source_media_type,
            explainer_parameters_json=json.dumps(explainer_parameters, sort_keys=True, separators=(",", ":")),
            background_identity=background_identity,
            reasons=tuple(reasons),
            raw_rule_conditions_json=None
            if raw_rule_conditions is None
            else json.dumps(raw_rule_conditions, sort_keys=True, separators=(",", ":")),
            raw_mask_sha256=None if raw_mask is None else hashlib.sha256(raw_mask).hexdigest(),
        )

    def verify_exact_source(self, source_payload: bytes) -> bool:
        return self.source_payload == source_payload and hashlib.sha256(source_payload).hexdigest() == self.source_sha256


@dataclass(frozen=True)
class PresentationReason:
    reason_id: str
    label: str
    normalized_magnitude: float
    direction: str
    source_rank: int


@dataclass(frozen=True)
class PresentationProjection:
    canonical_sha256: str
    reasons: tuple[PresentationReason, ...]
    omitted_reason_ids: tuple[str, ...]
    grouping_policy: str
    normalization_policy: str
    text: str
    limitations: tuple[str, ...]
    projection_sha256: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def project_explanation(
    canonical: CanonicalExplanation,
    *,
    labels: dict[str, str],
    top_k: int = 5,
    grouping_policy: str = "identity",
    normalization_policy: str = "max_abs",
) -> PresentationProjection:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    ordered = sorted(canonical.reasons, key=lambda reason: (reason.raw_rank, -abs(reason.raw_magnitude), reason.reason_id))
    selected = ordered[:top_k]
    scale = max((abs(reason.raw_magnitude) for reason in canonical.reasons), default=1.0) or 1.0
    reasons = tuple(
        PresentationReason(
            reason_id=reason.reason_id,
            label=labels.get(reason.identity, reason.identity),
            normalized_magnitude=abs(reason.raw_magnitude) / scale,
            direction="supports" if reason.raw_sign > 0 else "contradicts" if reason.raw_sign < 0 else "neutral",
            source_rank=reason.raw_rank,
        )
        for reason in selected
    )
    omitted = tuple(reason.reason_id for reason in ordered[top_k:])
    text = "; ".join(f"{reason.label}: {reason.direction}" for reason in reasons)
    limitations = (f"Presentation shows {len(reasons)} of {len(canonical.reasons)} canonical reasons.",)
    payload = {
        "canonical_sha256": canonical.source_sha256,
        "reasons": [asdict(reason) for reason in reasons],
        "omitted_reason_ids": omitted,
        "grouping_policy": grouping_policy,
        "normalization_policy": normalization_policy,
        "text": text,
        "limitations": limitations,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return PresentationProjection(
        canonical_sha256=canonical.source_sha256,
        reasons=reasons,
        omitted_reason_ids=omitted,
        grouping_policy=grouping_policy,
        normalization_policy=normalization_policy,
        text=text,
        limitations=limitations,
        projection_sha256=digest,
    )


def projection_metrics(canonical: CanonicalExplanation, projection: PresentationProjection) -> dict[str, object]:
    source = {reason.reason_id: reason for reason in canonical.reasons}
    selected = [source[reason.reason_id] for reason in projection.reasons]
    retained_mass = sum(abs(reason.raw_magnitude) for reason in selected) / max(
        1e-12, sum(abs(reason.raw_magnitude) for reason in canonical.reasons)
    )
    sign_preserved = all(
        projected.direction == ("supports" if source[projected.reason_id].raw_sign > 0 else "contradicts" if source[projected.reason_id].raw_sign < 0 else "neutral")
        for projected in projection.reasons
    )
    return {
        "canonical_hash_preserved": projection.canonical_sha256 == canonical.source_sha256,
        "selected_reason_identity_preserved": all(reason.reason_id in source for reason in projection.reasons),
        "sign_preserved": sign_preserved,
        "retained_absolute_magnitude": float(retained_mass),
        "sparsity": len(projection.reasons) / len(canonical.reasons),
        "presentation_length_characters": len(projection.text),
    }
