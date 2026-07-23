from __future__ import annotations


def per_pipeline_effects(rows: list[dict], metric: str, baseline: str) -> dict[str, float]:
    grouped: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows:
        grouped.setdefault((row["pipeline"], row["case_id"]), {})[row["method"]] = float(row[metric])
    effects: dict[str, list[float]] = {}
    for (pipeline, _), values in grouped.items():
        if "fuzzyxai_v21" in values and baseline in values:
            effects.setdefault(pipeline, []).append(values["fuzzyxai_v21"] - values[baseline])
    return {pipeline: sum(values) / len(values) for pipeline, values in effects.items()}

