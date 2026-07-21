from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence, cast

from .contracts import ComparisonStatement, DomainFeatureLanguage, DomainLanguageValidation


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_domain_language(
    domain_language: Mapping[str, Any],
    *,
    contribution_directions: Mapping[str, float] | None = None,
    regulated_domain: bool = False,
) -> DomainLanguageValidation:
    """Validate semantic direction and expert-review provenance of a domain dictionary."""

    features = domain_language.get("features", {})
    if not isinstance(features, Mapping):
        return DomainLanguageValidation(
            version=str(domain_language.get("version", "unversioned")),
            language_hash=_canonical_hash(domain_language),
            status="rejected",
            checked_features=0,
            errors=("domain_language.features must be a mapping",),
            warnings=(),
            expert_review_required=regulated_domain,
        )

    errors: list[str] = []
    warnings: list[str] = []
    contributions = dict(contribution_directions or {})
    for feature, raw in features.items():
        if not isinstance(raw, Mapping):
            errors.append(f"feature {feature}: definition must be a mapping")
            continue
        try:
            item = DomainFeatureLanguage(
                label=str(raw.get("label", "")),
                meaning=str(raw.get("meaning", "")),
                unit=None if raw.get("unit") is None else str(raw["unit"]),
                high_text=None if raw.get("high_text") is None else str(raw["high_text"]),
                low_text=None if raw.get("low_text") is None else str(raw["low_text"]),
                positive_effect_text=None if raw.get("positive_effect_text") is None else str(raw["positive_effect_text"]),
                negative_effect_text=None if raw.get("negative_effect_text") is None else str(raw["negative_effect_text"]),
                expected_direction=cast(Any, str(raw.get("expected_direction", "unknown"))),
                expert_review_status=cast(Any, str(raw.get("expert_review_status", "not_reviewed"))),
                reviewer_role=None if raw.get("reviewer_role") is None else str(raw["reviewer_role"]),
                reviewed_at=None if raw.get("reviewed_at") is None else str(raw["reviewed_at"]),
            )
        except (TypeError, ValueError) as exc:
            errors.append(f"feature {feature}: {exc}")
            continue
        if not item.label or not item.meaning:
            errors.append(f"feature {feature}: label and meaning are required")
        contribution = contributions.get(str(feature))
        if contribution is not None:
            if item.expected_direction == "increases_target" and contribution < 0:
                errors.append(f"feature {feature}: negative contribution contradicts increases_target")
            if item.expected_direction == "decreases_target" and contribution > 0:
                errors.append(f"feature {feature}: positive contribution contradicts decreases_target")
        if item.expected_direction == "non_monotonic" and (item.positive_effect_text or item.negative_effect_text):
            warnings.append(f"feature {feature}: non-monotonic effect must be explained without a universal direction")
        if item.expected_direction == "unknown" and (item.positive_effect_text or item.negative_effect_text):
            warnings.append(f"feature {feature}: directional text is hidden because expected direction is unknown")
        if item.expert_review_status == "rejected":
            errors.append(f"feature {feature}: expert review rejected the wording")
        if item.expert_review_status != "reviewed":
            warnings.append(f"feature {feature}: semantic wording has not been externally reviewed")

    if regulated_domain and any(
        not isinstance(raw, Mapping) or raw.get("expert_review_status") != "reviewed" for raw in features.values()
    ):
        warnings.append("regulated domain requires external expert review before categorical domain-user wording")
    status = "rejected" if errors else "pass"
    if not errors and (regulated_domain or warnings):
        status = "insufficient_domain_language"
    return DomainLanguageValidation(
        version=str(domain_language.get("version", "unversioned")),
        language_hash=_canonical_hash(domain_language),
        status=cast(Any, status),
        checked_features=len(features),
        errors=tuple(errors),
        warnings=tuple(dict.fromkeys(warnings)),
        expert_review_required=regulated_domain,
    )


def comparison_statement(
    values: Sequence[float],
    object_value: float,
    *,
    reference_label: str,
    representation: str,
) -> ComparisonStatement:
    """Create sample-size-aware wording without overstating small references."""

    finite = sorted(float(value) for value in values if value is not None)
    size = len(finite)
    if not size:
        return ComparisonStatement(
            0,
            reference_label,
            representation,
            None,
            None,
            "insufficient_evidence",
            "Недостаточно данных для сравнения.",
            ("reference sample is empty",),
        )
    rank = sum(value <= object_value for value in finite)
    percentile = 100.0 * rank / size
    if size < 20:
        text = (
            f"Значение является максимальным среди {size} контрольных объектов ({reference_label})."
            if rank == size
            else f"Значение занимает {size - rank + 1}-е место сверху среди {size} контрольных объектов ({reference_label})."
        )
        policy = "small_sample_rank"
        limitations = ("small reference sample; percentile wording suppressed",)
    elif size < 100:
        text = f"Значение входит в число наиболее высоких в контрольной выборке из {size} объектов ({reference_label})."
        policy = "medium_sample_tail"
        limitations = ("reference sample is too small for a precise population percentile claim",)
    else:
        upper = max(0.0, 100.0 - percentile)
        text = f"Значение находится в верхних {max(1, round(upper))}% распределения среди {size} объектов ({reference_label})."
        policy = "large_sample_percentile"
        limitations = ()
    return ComparisonStatement(
        size,
        reference_label,
        representation,
        round(percentile, 6),
        rank,
        cast(Any, policy),
        text,
        limitations,
    )


def comparison_from_percentile(
    sample_size: int,
    percentile: float,
    *,
    reference_label: str,
    representation: str,
) -> ComparisonStatement:
    """Apply the wording policy when only an audited percentile summary is retained."""

    if sample_size <= 0:
        return comparison_statement((), 0.0, reference_label=reference_label, representation=representation)
    rank = max(1, min(sample_size, round(percentile * sample_size / 100.0)))
    if sample_size < 20:
        text = (
            f"Значение является максимальным среди {sample_size} контрольных объектов ({reference_label})."
            if rank == sample_size
            else f"Значение занимает {sample_size - rank + 1}-е место сверху среди {sample_size} контрольных объектов ({reference_label})."
        )
        policy = "small_sample_rank"
        limitations = ("small reference sample; percentile wording suppressed",)
    elif sample_size < 100:
        text = f"Значение входит в число наиболее высоких в контрольной выборке из {sample_size} объектов ({reference_label})."
        policy = "medium_sample_tail"
        limitations = ("reference sample is too small for a precise population percentile claim",)
    else:
        upper = max(0.0, 100.0 - percentile)
        text = f"Значение находится в верхних {max(1, round(upper))}% распределения среди {sample_size} объектов ({reference_label})."
        policy = "large_sample_percentile"
        limitations = ()
    return ComparisonStatement(
        sample_size,
        reference_label,
        representation,
        round(percentile, 6),
        rank,
        cast(Any, policy),
        text,
        limitations,
    )
