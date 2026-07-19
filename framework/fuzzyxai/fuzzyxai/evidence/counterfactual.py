from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from fuzzyxai.adapters.model import ModelAdapter

from .contracts import CounterfactualEvidence


def find_tabular_counterfactuals(
    adapter: ModelAdapter,
    query: Sequence[float],
    reference_values: Sequence[Sequence[float]],
    *,
    feature_names: Sequence[str] | None = None,
    limit: int = 3,
) -> list[CounterfactualEvidence]:
    """Search one-feature, reference-quantile changes that alter the prediction."""

    vector = np.asarray(query, dtype=float).reshape(-1)
    reference = np.asarray(reference_values, dtype=float)
    if reference.ndim != 2 or reference.shape[1] != len(vector):
        raise ValueError("query and reference_values must have matching feature width")
    names = list(feature_names or [f"feature_{index}" for index in range(len(vector))])
    source = adapter.predict([vector.tolist()])
    source_prediction = _first_prediction(source.predictions)
    scale = np.nanstd(reference, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    candidates: list[tuple[float, CounterfactualEvidence]] = []
    for feature_index, name in enumerate(names):
        for value in np.unique(np.nanquantile(reference[:, feature_index], [0.1, 0.25, 0.5, 0.75, 0.9])):
            changed = vector.copy()
            changed[feature_index] = value
            prediction = adapter.predict([changed.tolist()])
            target = _first_prediction(prediction.predictions)
            if target == source_prediction:
                continue
            distance = abs(float(value - vector[feature_index])) / float(scale[feature_index])
            score_before = source.primary_score()
            score_after = prediction.primary_score()
            observed = None if score_before is None or score_after is None else float(score_after - score_before)
            candidates.append(
                (
                    distance,
                    CounterfactualEvidence(
                        source_prediction=source_prediction,
                        target_prediction=target,
                        changed_features={name: {"from": float(vector[feature_index]), "to": float(value)}},
                        changed_regions=[],
                        changed_rules=[],
                        minimality=round(distance, 6),
                        plausibility=1.0,
                        stability=None,
                        expected_effect=None,
                        observed_effect=None if observed is None else round(observed, 6),
                        actionability="requires domain review",
                        limitations=[
                            "quantile search tests association, not causal feasibility",
                            "observed_effect is the change in maximum predicted probability, not a causal effect",
                        ],
                        evidence_refs=["model prediction before and after feature intervention"],
                    ),
                )
            )
    return [item for _, item in sorted(candidates, key=lambda pair: pair[0])[:limit]]


def _first_prediction(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, list) and value:
        value = value[0]
    return value
