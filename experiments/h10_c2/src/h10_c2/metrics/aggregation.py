from __future__ import annotations

from collections import defaultdict


def means_by_method(rows: list[dict], metric: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(float(row[metric]))
    return {method: sum(values) / len(values) for method, values in grouped.items()}

