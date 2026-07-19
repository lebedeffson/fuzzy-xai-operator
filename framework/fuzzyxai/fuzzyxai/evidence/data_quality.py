from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import DataEvidence


def collect_data_evidence(
    values: Sequence[Sequence[Any]],
    *,
    object_ids: Sequence[str] | None = None,
    feature_names: Sequence[str] | None = None,
    reference_values: Sequence[Sequence[Any]] | None = None,
    source_trace: Mapping[str, Any] | None = None,
    outlier_threshold: float = 3.5,
) -> list[DataEvidence]:
    """Measure missingness and robust median/MAD deviation for tabular objects.

    The outlier score is a robust z-score. A large score identifies a deviation,
    not an error; callers need domain evidence to distinguish a rare subtype
    from corrupted data.
    """

    raw = np.asarray(values, dtype=float)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    reference = np.asarray(reference_values if reference_values is not None else values, dtype=float)
    if reference.ndim == 1:
        reference = reference.reshape(1, -1)
    if raw.shape[1] != reference.shape[1]:
        raise ValueError("values and reference_values must have the same feature count")

    names = list(feature_names or [f"feature_{index}" for index in range(raw.shape[1])])
    if len(names) != raw.shape[1]:
        raise ValueError("feature_names length must match the input width")
    ids = list(object_ids or [f"object_{index}" for index in range(raw.shape[0])])
    if len(ids) != raw.shape[0]:
        raise ValueError("object_ids length must match the number of rows")

    median = np.nanmedian(reference, axis=0)
    q05 = np.nanpercentile(reference, 5, axis=0)
    q95 = np.nanpercentile(reference, 95, axis=0)
    mad = np.nanmedian(np.abs(reference - median), axis=0)
    fallback = np.nanstd(reference, axis=0)
    scale = np.where(mad > 1e-12, 1.4826 * mad, np.where(fallback > 1e-12, fallback, 1.0))

    results: list[DataEvidence] = []
    for object_id, row in zip(ids, raw):
        missing = np.isnan(row)
        normalized = (row - median) / scale
        scores = np.abs(normalized)
        anomaly_features = [names[index] for index, score in enumerate(scores) if np.isfinite(score) and score >= outlier_threshold]
        missing_features = [names[index] for index, flag in enumerate(missing) if flag]
        missing_fraction = float(np.mean(missing))
        anomaly_fraction = float(len(anomaly_features) / max(len(names), 1))
        quality = max(0.0, min(1.0, 1.0 - missing_fraction - 0.5 * anomaly_fraction))
        warnings: list[str] = []
        reference_profiles = {}
        for index, name in enumerate(names):
            finite = reference[:, index][np.isfinite(reference[:, index])]
            percentile = None
            if finite.size and np.isfinite(row[index]):
                percentile = float(100.0 * np.mean(finite <= row[index]))
            reference_profiles[name] = {
                "q05": None if not np.isfinite(q05[index]) else float(q05[index]),
                "median": None if not np.isfinite(median[index]) else float(median[index]),
                "q95": None if not np.isfinite(q95[index]) else float(q95[index]),
                "percentile": percentile,
            }
        if missing_features:
            warnings.append("missing values: " + ", ".join(missing_features))
        if anomaly_features:
            warnings.append(
                "robust deviation detected; domain validation is required before treating the object as erroneous"
            )
        results.append(
            DataEvidence(
                object_id=str(object_id),
                feature_names=names,
                raw_values=[None if np.isnan(value) else float(value) for value in row],
                normalized_values=[None if not np.isfinite(value) else float(value) for value in normalized],
                missingness={name: bool(flag) for name, flag in zip(names, missing)},
                outlier_scores={name: None if not np.isfinite(score) else float(score) for name, score in zip(names, scores)},
                anomaly_labels=anomaly_features,
                data_quality=round(quality, 6),
                source_trace=dict(source_trace or {}),
                warnings=warnings,
                evidence_refs=["robust_median_mad_profile"],
                reference_profiles=reference_profiles,
            )
        )
    return results
