from __future__ import annotations

from collections import defaultdict


def select_baseline(rows: list[dict], metric: str = "optimal_cut_set_membership") -> str:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["method"] != "fuzzyxai_v21":
            values[row["method"]].append(float(row[metric]))
    if not values:
        raise ValueError("no baseline rows")
    return max(values, key=lambda method: (sum(values[method]) / len(values[method]), method))

