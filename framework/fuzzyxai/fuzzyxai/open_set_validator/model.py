"""Distance-calibrated typed open-set validator."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .contracts import OpenSetAssessment, OpenSetOutcome, OpenSetTrainingRow, StructuralObservation


@dataclass(frozen=True)
class OpenSetValidatorSpec:
    feature_names: tuple[str, ...]
    family_names: tuple[str, ...]
    centroids: tuple[tuple[float, ...], ...]
    valid_centroid: tuple[float, ...]
    feature_means: tuple[float, ...]
    feature_scales: tuple[float, ...]
    known_distance_threshold: float
    valid_energy_threshold: float
    valid_distance_threshold: float
    insufficient_channel_limit: int
    development_sha256: str
    selected_without_test: bool = True

    def __post_init__(self) -> None:
        if not self.selected_without_test:
            raise ValueError("open-set thresholds must be selected without test")
        if len(self.family_names) != len(self.centroids) or not self.family_names:
            raise ValueError("family centroids must be non-empty and aligned")
        if len(self.feature_names) != len(self.feature_means) or len(self.feature_names) != len(self.feature_scales):
            raise ValueError("feature normalization is inconsistent")


def fit_open_set_validator(
    rows: Sequence[OpenSetTrainingRow],
    *,
    known_quantile: float = 0.95,
    valid_energy_quantile: float = 0.99,
) -> OpenSetValidatorSpec:
    if len(rows) < 20:
        raise ValueError("at least 20 OOF development rows are required")
    names = tuple(sorted(rows[0].observation.features))
    if any(tuple(sorted(row.observation.features)) != names for row in rows):
        raise ValueError("open-set feature schema changed within development")
    matrix = np.asarray([[row.observation.features[name] for name in names] for row in rows], dtype=float)
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales[scales < 1e-9] = 1.0
    normalized = (matrix - means) / scales
    labels = np.asarray([row.fault_family for row in rows], dtype=object)
    valid_mask = labels == "valid_route"
    families = tuple(sorted(set(labels) - {"valid_route"}))
    if not families or not np.any(valid_mask):
        raise ValueError("development requires valid routes and known fault families")
    centroids = tuple(tuple(float(value) for value in normalized[labels == family].mean(axis=0)) for family in families)
    distances = []
    for family, centroid in zip(families, centroids, strict=True):
        family_rows = normalized[labels == family]
        distances.extend(np.linalg.norm(family_rows - np.asarray(centroid), axis=1).tolist())
    valid_energy = np.linalg.norm(normalized[valid_mask], axis=1)
    valid_centroid = normalized[valid_mask].mean(axis=0)
    valid_distances = np.linalg.norm(normalized[valid_mask] - valid_centroid, axis=1)
    payload = [
        {"id": row.observation.observation_id, "family": row.fault_family, "features": [row.observation.features[name] for name in names]}
        for row in rows
    ]
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return OpenSetValidatorSpec(
        feature_names=names,
        family_names=families,
        centroids=centroids,
        valid_centroid=tuple(float(value) for value in valid_centroid),
        feature_means=tuple(float(value) for value in means),
        feature_scales=tuple(float(value) for value in scales),
        known_distance_threshold=float(np.quantile(distances, known_quantile)),
        valid_energy_threshold=float(np.quantile(valid_energy, valid_energy_quantile)),
        valid_distance_threshold=float(np.quantile(valid_distances, valid_energy_quantile)),
        insufficient_channel_limit=0,
        development_sha256=digest,
    )


def assess_open_set(spec: OpenSetValidatorSpec, observation: StructuralObservation) -> OpenSetAssessment:
    missing = set(spec.feature_names) - set(observation.features)
    if observation.missing_channels or missing:
        return OpenSetAssessment(
            OpenSetOutcome.INSUFFICIENT_EVIDENCE,
            None,
            1.0,
            0.0,
            tuple(sorted(set(observation.missing_channels) | missing)),
            tuple(f"collect:{item}" for item in sorted(set(observation.missing_channels) | missing)),
            ("MISSING_REQUIRED_STRUCTURAL_EVIDENCE",),
        )
    values = np.asarray([observation.features[name] for name in spec.feature_names], dtype=float)
    normalized = (values - np.asarray(spec.feature_means)) / np.asarray(spec.feature_scales)
    valid_distance = float(np.linalg.norm(normalized - np.asarray(spec.valid_centroid)))
    centroids = np.asarray(spec.centroids)
    distances = np.linalg.norm(centroids - normalized, axis=1)
    nearest = int(np.argmin(distances))
    nearest_distance = float(distances[nearest])
    threshold = max(spec.known_distance_threshold, 1e-9)
    unknown_score = float(1.0 - np.exp(-nearest_distance / threshold))
    known_confidence = float(np.exp(-nearest_distance / threshold))
    suspected = _suspected_regions(spec, observation, normalized)
    repairs = tuple(f"inspect_or_restore:{region}" for region in suspected)
    if valid_distance <= spec.valid_distance_threshold:
        return OpenSetAssessment(OpenSetOutcome.VALID_ROUTE, None, unknown_score, known_confidence, (), (), ("WITHIN_VALID_ROUTE_ENVELOPE",))
    if nearest_distance <= spec.known_distance_threshold:
        family = spec.family_names[nearest]
        return OpenSetAssessment(OpenSetOutcome.KNOWN_FAULT_TYPE, family, unknown_score, known_confidence, suspected, repairs, (f"KNOWN_FAULT:{family}",))
    return OpenSetAssessment(OpenSetOutcome.UNKNOWN_STRUCTURAL_FAULT, None, unknown_score, known_confidence, suspected, repairs, ("UNKNOWN_STRUCTURAL_FAULT", "AUTOMATIC_CERTIFICATION_DENIED"))


def _suspected_regions(spec: OpenSetValidatorSpec, observation: StructuralObservation, normalized: np.ndarray) -> tuple[str, ...]:
    order = np.argsort(-np.abs(normalized), kind="stable")
    regions = []
    for index in order:
        region = observation.feature_regions[spec.feature_names[int(index)]]
        if region not in regions:
            regions.append(region)
        if len(regions) == 3:
            break
    return tuple(regions)
