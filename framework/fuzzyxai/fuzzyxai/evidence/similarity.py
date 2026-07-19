from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .contracts import SimilarCaseEvidence


def find_similar_tabular_cases(
    query: Sequence[float],
    reference_values: Sequence[Sequence[float]],
    *,
    query_object_id: str,
    reference_ids: Sequence[str] | None = None,
    feature_names: Sequence[str] | None = None,
    reference_labels: Sequence[Any] | None = None,
    reference_predictions: Sequence[Any] | None = None,
    reference_outcomes: Sequence[Any] | None = None,
    limit: int = 3,
) -> list[SimilarCaseEvidence]:
    """Find cases using robust standardized Euclidean feature distance."""

    matrix = np.asarray(reference_values, dtype=float)
    vector = np.asarray(query, dtype=float).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[1] != len(vector):
        raise ValueError("query and reference_values must have matching feature width")
    names = list(feature_names or [f"feature_{index}" for index in range(matrix.shape[1])])
    ids = list(reference_ids or [f"reference_{index}" for index in range(len(matrix))])
    median = np.nanmedian(matrix, axis=0)
    mad = np.nanmedian(np.abs(matrix - median), axis=0)
    scale = np.where(mad > 1e-12, 1.4826 * mad, np.where(np.nanstd(matrix, axis=0) > 1e-12, np.nanstd(matrix, axis=0), 1.0))
    deltas = np.abs((matrix - vector) / scale)
    distances = np.sqrt(np.nanmean(deltas**2, axis=1))
    similarity = np.exp(-distances)
    results: list[SimilarCaseEvidence] = []
    selected = [index for index in np.argsort(distances) if str(ids[index]) != str(query_object_id)][:limit]
    for index in selected:
        matched = [name for name, delta in zip(names, deltas[index]) if np.isfinite(delta) and delta <= 0.5]
        different = [name for name, delta in zip(names, deltas[index]) if not np.isfinite(delta) or delta > 0.5]
        results.append(
            SimilarCaseEvidence(
                query_object_id=str(query_object_id),
                reference_object_id=str(ids[index]),
                similarity_score=round(float(similarity[index]), 6),
                similarity_method="robust_standardized_euclidean",
                compared_representation="normalized tabular feature vector",
                matched_features=matched,
                different_features=different,
                matched_regions=[],
                coverage_score=len(matched) / max(len(names), 1),
                reference_label=_at(reference_labels, index),
                reference_prediction=_at(reference_predictions, index),
                reference_outcome=_at(reference_outcomes, index),
                limitations=["feature distance does not establish causal or clinical similarity"],
                trace={"reference_index": int(index), "scale": "median absolute deviation"},
            )
        )
    return results


def compare_region_masks(
    query_mask: Sequence[Sequence[bool]],
    reference_mask: Sequence[Sequence[bool]],
    *,
    query_object_id: str,
    reference_object_id: str,
) -> SimilarCaseEvidence:
    """Measure explicitly labelled binary-mask intersection over union."""

    query = np.asarray(query_mask, dtype=bool)
    reference = np.asarray(reference_mask, dtype=bool)
    if query.shape != reference.shape:
        raise ValueError("query_mask and reference_mask must have the same shape")
    intersection = int(np.logical_and(query, reference).sum())
    union = int(np.logical_or(query, reference).sum())
    score = intersection / union if union else 1.0
    return SimilarCaseEvidence(
        query_object_id=query_object_id,
        reference_object_id=reference_object_id,
        similarity_score=round(score, 6),
        similarity_method="binary_mask_intersection_over_union",
        compared_representation="segmentation masks (pixels inside the selected region)",
        matched_features=[],
        different_features=[],
        matched_regions=["segmentation_mask"],
        coverage_score=round(score, 6),
        reference_label=None,
        reference_prediction=None,
        reference_outcome=None,
        limitations=["mask overlap does not measure semantic or diagnostic equivalence"],
        trace={"intersection_pixels": intersection, "union_pixels": union},
    )


def _at(values: Sequence[Any] | None, index: int) -> Any:
    return None if values is None else values[index]
