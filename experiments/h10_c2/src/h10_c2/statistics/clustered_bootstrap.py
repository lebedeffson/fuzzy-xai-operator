from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable

import numpy as np


def paired_hierarchical_bootstrap(
    rows: Iterable[dict],
    *,
    metric: str,
    method: str,
    baseline: str,
    repetitions: int,
    seed: int,
) -> dict:
    by_pipeline: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_pipeline[str(row["pipeline"])].append(row)
    pipelines = sorted(by_pipeline)
    if not pipelines:
        raise ValueError("no rows for bootstrap")
    rng = random.Random(seed)
    effects = []
    for _ in range(repetitions):
        selected_pipelines = [rng.choice(pipelines) for _ in pipelines]
        sampled_effects = []
        for pipeline in selected_pipelines:
            grouped: dict[str, dict[str, float]] = defaultdict(dict)
            for row in by_pipeline[pipeline]:
                grouped[str(row["case_id"])][str(row["method"])] = float(row[metric])
            case_ids = [case_id for case_id, values in grouped.items() if method in values and baseline in values]
            if not case_ids:
                continue
            for _ in case_ids:
                case_id = rng.choice(case_ids)
                values = grouped[case_id]
                sampled_effects.append(values[method] - values[baseline])
        if sampled_effects:
            effects.append(float(np.mean(sampled_effects)))
    if not effects:
        raise ValueError("no paired observations")
    values = np.asarray(effects)
    return {
        "effect": float(values.mean()),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "p_raw": float(2.0 * min(np.mean(values <= 0.0), np.mean(values >= 0.0))),
        "repetitions": len(effects),
        "clusters": len(pipelines),
    }

