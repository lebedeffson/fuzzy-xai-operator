from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping
import math

import numpy as np

from .explain_plan import ExplainPlan


def _as_dataframe(data):
    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError('pandas is required for ExplainPlan.from_data') from exc
    if isinstance(data, pd.DataFrame):
        return data.copy()
    return pd.DataFrame(data)


def _quantiles(values: np.ndarray) -> Dict[str, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return {'min': 0.0, 'q25': 0.25, 'median': 0.5, 'q75': 0.75, 'max': 1.0}
    qs = np.quantile(clean, [0.0, 0.25, 0.5, 0.75, 1.0])
    if qs[0] == qs[-1]:
        qs = np.array([qs[0], qs[0] + 0.25, qs[0] + 0.5, qs[0] + 0.75, qs[0] + 1.0], dtype=float)
    return {'min': float(qs[0]), 'q25': float(qs[1]), 'median': float(qs[2]), 'q75': float(qs[3]), 'max': float(qs[4])}


def build_explain_plan_from_dataframe(data, *, target: str | None = None, n_terms: int = 3, mode: str = 'audit') -> ExplainPlan:
    """Generate an ExplainPlan from tabular data.

    This is a reproducible constructor for the dissertation demo layer. It does
    not claim domain optimality; it generates an auditable starting plan with
    quantile-based linguistic terms.
    """
    df = _as_dataframe(data)
    if target and target in df.columns:
        feature_df = df.drop(columns=[target])
    else:
        feature_df = df

    numeric_features = []
    categorical_features = []
    feature_terms: Dict[str, Any] = {}

    for col in feature_df.columns:
        series = feature_df[col]
        if np.issubdtype(series.dropna().dtype, np.number):
            numeric_features.append(str(col))
            q = _quantiles(series.to_numpy(dtype=float))
            feature_terms[str(col)] = {
                'type': 'numeric',
                'quantiles': q,
                'terms': {
                    'low': {'kind': 'left_shoulder', 'a': q['min'], 'b': q['median']},
                    'medium': {'kind': 'triangle', 'a': q['q25'], 'b': q['median'], 'c': q['q75']},
                    'high': {'kind': 'right_shoulder', 'a': q['median'], 'b': q['max']},
                },
            }
        else:
            categorical_features.append(str(col))
            values = [str(v) for v in series.dropna().unique()[:20]]
            feature_terms[str(col)] = {'type': 'categorical', 'values': values}

    metadata = {
        'generated': True,
        'generation_method': 'quantile_terms',
        'mode': mode,
        'target': target,
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'feature_terms': feature_terms,
        'n_rows': int(len(df)),
        'n_features': int(len(feature_df.columns)),
    }

    plan = ExplainPlan(metadata=metadata)
    plan.validate()
    return plan
