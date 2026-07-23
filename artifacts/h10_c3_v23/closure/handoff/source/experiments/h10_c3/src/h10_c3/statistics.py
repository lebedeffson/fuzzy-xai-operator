from __future__ import annotations

import math
import random
from collections import defaultdict


def _paired_differences(
    rows: list[dict[str, object]],
    baseline: str,
    metric: str,
) -> dict[str, list[float]]:
    indexed = {(row["case_id"], row["method"]): row for row in rows}
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["method"] != "full_fuzzyxai":
            continue
        peer = indexed[(row["case_id"], baseline)]
        full_value = float(row[metric])
        baseline_value = float(peer[metric])
        difference = (
            baseline_value - full_value
            if metric == "normalized_cost_regret"
            else full_value - baseline_value
        )
        grouped[str(row["pipeline"])].append(difference)
    return grouped


def hierarchical_bootstrap(
    rows: list[dict[str, object]],
    baseline: str,
    metric: str,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    grouped = _paired_differences(rows, baseline, metric)
    pipelines = sorted(grouped)
    observed = sum(sum(values) for values in grouped.values()) / sum(map(len, grouped.values()))
    rng = random.Random(seed)
    draws = []
    for _ in range(repetitions):
        sampled_pipelines = [rng.choice(pipelines) for _ in pipelines]
        values = []
        for pipeline in sampled_pipelines:
            source = grouped[pipeline]
            values.extend(rng.choice(source) for _ in source)
        draws.append(sum(values) / len(values))
    ordered = sorted(draws)
    low = ordered[int(0.025 * repetitions)]
    high = ordered[min(repetitions - 1, int(0.975 * repetitions))]
    p_value = (sum(value <= 0 for value in draws) + 1) / (repetitions + 1)
    return {
        "effect": observed,
        "ci_low": low,
        "ci_high": high,
        "p_raw": p_value,
        "repetitions": repetitions,
    }


def holm(results: list[dict[str, object]]) -> None:
    ordered = sorted(enumerate(results), key=lambda item: float(item[1]["p_raw"]))
    running = 0.0
    total = len(results)
    for rank, (index, result) in enumerate(ordered):
        adjusted = min(1.0, (total - rank) * float(result["p_raw"]))
        running = max(running, adjusted)
        results[index]["p_holm"] = running


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 1.0
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return centre - radius, centre + radius


def pipeline_directions(
    rows: list[dict[str, object]],
    baseline: str,
    metric: str,
) -> dict[str, float]:
    grouped = _paired_differences(rows, baseline, metric)
    return {pipeline: sum(values) / len(values) for pipeline, values in sorted(grouped.items())}

