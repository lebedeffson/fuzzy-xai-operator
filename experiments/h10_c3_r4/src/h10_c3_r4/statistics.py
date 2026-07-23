from __future__ import annotations

import math
import random
from collections import defaultdict


def paired_template_differences(
    rows: list[dict[str, object]],
    baseline: str,
    metric: str,
) -> dict[str, dict[str, list[float]]]:
    indexed = {
        (str(row["case_id"]), str(row["method"])): row
        for row in rows
    }
    grouped: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        if row["method"] != "full_h10":
            continue
        peer = indexed[(str(row["case_id"]), baseline)]
        full_value = float(row[metric])
        baseline_value = float(peer[metric])
        difference = (
            baseline_value - full_value
            if metric == "normalized_cost_regret"
            else full_value - baseline_value
        )
        grouped[str(row["pipeline_family"])][
            str(row["template_hash"])
        ].append(difference)
    return grouped


def hierarchical_bootstrap(
    rows: list[dict[str, object]],
    baseline: str,
    metric: str,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    grouped = paired_template_differences(rows, baseline, metric)
    pipelines = sorted(grouped)
    observed_values = [
        value
        for templates in grouped.values()
        for values in templates.values()
        for value in values
    ]
    observed = sum(observed_values) / len(observed_values)
    rng = random.Random(seed)
    draws = []
    for _ in range(repetitions):
        sampled_pipelines = [
            rng.choice(pipelines) for _ in pipelines
        ]
        values = []
        for pipeline in sampled_pipelines:
            templates = sorted(grouped[pipeline])
            sampled_templates = [
                rng.choice(templates) for _ in templates
            ]
            for template in sampled_templates:
                cases = grouped[pipeline][template]
                values.extend(rng.choice(cases) for _ in cases)
        draws.append(sum(values) / len(values))
    ordered = sorted(draws)
    return {
        "effect": observed,
        "ci_low": ordered[int(0.025 * repetitions)],
        "ci_high": ordered[min(repetitions - 1, int(0.975 * repetitions))],
        "p_raw": (
            sum(value <= 0 for value in draws) + 1
        )
        / (repetitions + 1),
        "repetitions": repetitions,
        "independent_template_units": sum(
            len(templates) for templates in grouped.values()
        ),
    }


def holm(results: list[dict[str, object]]) -> None:
    ordered = sorted(
        enumerate(results),
        key=lambda item: float(item[1]["p_raw"]),
    )
    running = 0.0
    total = len(results)
    for rank, (index, result) in enumerate(ordered):
        adjusted = min(
            1.0,
            (total - rank) * float(result["p_raw"]),
        )
        running = max(running, adjusted)
        results[index]["p_holm"] = running


def pipeline_effects(
    rows: list[dict[str, object]],
    baseline: str,
    metric: str,
) -> dict[str, float]:
    grouped = paired_template_differences(rows, baseline, metric)
    return {
        pipeline: sum(
            value
            for values in templates.values()
            for value in values
        )
        / sum(len(values) for values in templates.values())
        for pipeline, templates in sorted(grouped.items())
    }


def wilson_interval(
    successes: int,
    total: int,
    z: float = 1.96,
) -> tuple[float, float]:
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (
        proportion + z * z / (2 * total)
    ) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total
            + z * z / (4 * total * total)
        )
        / denominator
    )
    return centre - radius, centre + radius
