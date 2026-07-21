"""Trace completeness and controlled missing-channel diagnosis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class EvidenceClaim:
    claim_id: str
    evidence_refs: tuple[str, ...]
    source_types: tuple[str, ...]
    versions: tuple[str, ...]
    hashes: tuple[str, ...]

    @property
    def has_complete_path(self) -> bool:
        return bool(self.evidence_refs and self.source_types and self.versions and self.hashes)


@dataclass(frozen=True)
class MissingnessPrediction:
    object_id: str
    actual_missing: tuple[str, ...]
    predicted_missing: tuple[str, ...]
    certified_complete: bool


@dataclass(frozen=True)
class MissingnessReport:
    n_objects: int
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    precision: float
    recall: float
    f1: float
    source_localization_accuracy: float
    false_certification_rate: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def traceability_score(claims: Sequence[EvidenceClaim]) -> float:
    if not claims:
        return 0.0
    return sum(item.has_complete_path for item in claims) / len(claims)


def diagnose_missing_channels(
    available: Mapping[str, object | None],
    required: Sequence[str],
) -> tuple[str, ...]:
    return tuple(sorted(channel for channel in required if channel not in available or available[channel] is None))


def evaluate_missingness(rows: Sequence[MissingnessPrediction]) -> MissingnessReport:
    if not rows:
        raise ValueError("missingness evaluation requires observations")
    tp = fp = fn = tn = 0
    localization_correct = localization_total = false_certified = incomplete = 0
    for row in rows:
        actual = set(row.actual_missing)
        predicted = set(row.predicted_missing)
        actual_flag = bool(actual)
        predicted_flag = bool(predicted)
        tp += int(actual_flag and predicted_flag)
        fp += int(not actual_flag and predicted_flag)
        fn += int(actual_flag and not predicted_flag)
        tn += int(not actual_flag and not predicted_flag)
        if actual_flag:
            incomplete += 1
            localization_total += len(actual)
            localization_correct += len(actual & predicted)
            false_certified += int(row.certified_complete)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return MissingnessReport(
        n_objects=len(rows),
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        true_negative=tn,
        precision=precision,
        recall=recall,
        f1=f1,
        source_localization_accuracy=localization_correct / max(1, localization_total),
        false_certification_rate=false_certified / max(1, incomplete),
    )
