"""Evidence-first critical-rupture construction and association metrics."""

from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

from .contracts import CriticalRupture, CriticalRuptureType


CRITICAL_RUPTURE_DEFINITION = (
    "A critical rupture is a state with no admissible certified path from available "
    "evidence to an automatic action because a required uncertainty type is uncovered, "
    "a conflict is forbidden, provenance is invalid, or an information-loss threshold is exceeded."
)


def detect_critical_ruptures(
    *,
    object_id: str,
    required_evidence_present: bool,
    forbidden_rule_conflict: bool,
    provenance_verified: bool,
    representation_covers_profile: bool,
    reduction_loss: float,
    reduction_loss_threshold: float,
    distribution_shift: float,
    distribution_shift_threshold: float,
    explanation_stability: float,
    stability_threshold: float,
    cross_model_disagreement: float,
    disagreement_threshold: float,
    evidence_refs: Mapping[str, Sequence[str]],
) -> tuple[CriticalRupture, ...]:
    checks = (
        (not required_evidence_present, CriticalRuptureType.MISSING_REQUIRED_EVIDENCE, None, None),
        (forbidden_rule_conflict, CriticalRuptureType.FORBIDDEN_RULE_CONFLICT, None, None),
        (not provenance_verified, CriticalRuptureType.UNVERIFIED_PROVENANCE, None, None),
        (not representation_covers_profile, CriticalRuptureType.REPRESENTATION_UNDERCOVERAGE, None, None),
        (
            reduction_loss > reduction_loss_threshold,
            CriticalRuptureType.REDUCTION_LOSS_EXCEEDED,
            reduction_loss,
            reduction_loss_threshold,
        ),
        (
            distribution_shift > distribution_shift_threshold,
            CriticalRuptureType.DISTRIBUTION_SHIFT,
            distribution_shift,
            distribution_shift_threshold,
        ),
        (
            explanation_stability < stability_threshold,
            CriticalRuptureType.UNSTABLE_EXPLANATION,
            explanation_stability,
            stability_threshold,
        ),
        (
            cross_model_disagreement > disagreement_threshold,
            CriticalRuptureType.CROSS_MODEL_DISAGREEMENT,
            cross_model_disagreement,
            disagreement_threshold,
        ),
    )
    ruptures: list[CriticalRupture] = []
    for active, rupture_type, value, threshold in checks:
        if not active:
            continue
        refs = tuple(str(item) for item in evidence_refs.get(rupture_type.value, ()))
        if not refs:
            raise ValueError(f"missing evidence refs for active rupture {rupture_type.value}")
        ruptures.append(
            CriticalRupture(
                rupture_type=rupture_type,
                object_id=object_id,
                evidence_refs=refs,
                measured_value=value,
                threshold=threshold,
            )
        )
    return tuple(ruptures)


def rupture_error_association(
    rupture_flags: Sequence[bool],
    wrong_automatic_flags: Sequence[bool],
) -> dict[str, float | int | str]:
    if len(rupture_flags) != len(wrong_automatic_flags) or len(rupture_flags) == 0:
        raise ValueError("rupture and error flags must be non-empty and aligned")
    counts = Counter((bool(rupture), bool(error)) for rupture, error in zip(rupture_flags, wrong_automatic_flags))
    with_rupture = counts[(True, True)] + counts[(True, False)]
    without_rupture = counts[(False, True)] + counts[(False, False)]
    p_with = counts[(True, True)] / with_rupture if with_rupture else 0.0
    p_without = counts[(False, True)] / without_rupture if without_rupture else 0.0
    lift = p_with / p_without if p_without else None
    return {
        "n_objects": len(rupture_flags),
        "wrong_auto_with_rupture": counts[(True, True)],
        "wrong_auto_without_rupture": counts[(False, True)],
        "p_wrong_auto_given_rupture": p_with,
        "p_wrong_auto_without_rupture": p_without,
        "risk_ratio": lift if lift is not None else "not_estimable",
        "interpretation": (
            "predictive_association_measured"
            if p_with > p_without and with_rupture and without_rupture
            else "structural_diagnostic_only"
        ),
    }
