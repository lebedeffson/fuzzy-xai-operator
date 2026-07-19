"""Separate structural rupture localization from error prediction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Mapping, Sequence


class StructuralDefect(str, Enum):
    MISSING_REQUIRED_EVIDENCE = "missing_required_evidence"
    INVALID_PROVENANCE = "invalid_provenance"
    FORBIDDEN_CONFLICT = "forbidden_conflict"
    REPRESENTATION_UNDERCOVERAGE = "representation_undercoverage"
    REDUCTION_LOSS_EXCEEDED = "reduction_loss_exceeded"
    UNSTABLE_EXPLANATION = "unstable_explanation"
    DISTRIBUTION_SHIFT = "distribution_shift"
    CROSS_MODEL_DISAGREEMENT = "cross_model_disagreement"


@dataclass(frozen=True)
class StructuralObservation:
    object_id: str
    available_evidence: frozenset[str]
    required_evidence: frozenset[str]
    provenance_valid: bool
    forbidden_conflict: bool
    representation_covered: bool
    reduction_loss: float
    explanation_stability: float
    distribution_shift: float
    cross_model_disagreement: float
    evidence_refs: Mapping[StructuralDefect, tuple[str, ...]]


@dataclass(frozen=True)
class StructuralDiagnosis:
    object_id: str
    defects: tuple[StructuralDefect, ...]
    evidence_refs: tuple[str, ...]
    certified_complete: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["defects"] = [item.value for item in self.defects]
        return payload


def diagnose_structural_ruptures(
    observation: StructuralObservation,
    *,
    reduction_threshold: float = 0.20,
    stability_threshold: float = 0.70,
    shift_threshold: float = 0.20,
    disagreement_threshold: float = 0.20,
) -> StructuralDiagnosis:
    defects: list[StructuralDefect] = []
    if not observation.required_evidence.issubset(observation.available_evidence):
        defects.append(StructuralDefect.MISSING_REQUIRED_EVIDENCE)
    if not observation.provenance_valid:
        defects.append(StructuralDefect.INVALID_PROVENANCE)
    if observation.forbidden_conflict:
        defects.append(StructuralDefect.FORBIDDEN_CONFLICT)
    if not observation.representation_covered:
        defects.append(StructuralDefect.REPRESENTATION_UNDERCOVERAGE)
    if observation.reduction_loss > reduction_threshold:
        defects.append(StructuralDefect.REDUCTION_LOSS_EXCEEDED)
    if observation.explanation_stability < stability_threshold:
        defects.append(StructuralDefect.UNSTABLE_EXPLANATION)
    if observation.distribution_shift > shift_threshold:
        defects.append(StructuralDefect.DISTRIBUTION_SHIFT)
    if observation.cross_model_disagreement > disagreement_threshold:
        defects.append(StructuralDefect.CROSS_MODEL_DISAGREEMENT)
    references: list[str] = []
    for defect in defects:
        refs = observation.evidence_refs.get(defect, ())
        if not refs:
            raise ValueError(f"missing evidence refs for detected defect {defect.value}")
        references.extend(refs)
    return StructuralDiagnosis(
        object_id=observation.object_id,
        defects=tuple(defects),
        evidence_refs=tuple(dict.fromkeys(references)),
        certified_complete=not defects,
    )


def structural_metrics(
    expected: Sequence[Sequence[StructuralDefect]],
    diagnosed: Sequence[StructuralDiagnosis],
) -> dict[str, float | int]:
    if not expected or len(expected) != len(diagnosed):
        raise ValueError("structural metrics require aligned observations")
    tp = fp = fn = exact = false_certification = 0
    for truth, result in zip(expected, diagnosed):
        truth_set = set(truth)
        predicted = set(result.defects)
        tp += len(truth_set & predicted)
        fp += len(predicted - truth_set)
        fn += len(truth_set - predicted)
        exact += int(truth_set == predicted)
        false_certification += int(bool(truth_set) and result.certified_complete)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        "n_objects": len(expected),
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(1e-12, precision + recall),
        "exact_type_accuracy": exact / len(expected),
        "false_certification_rate": false_certification / len(expected),
    }
