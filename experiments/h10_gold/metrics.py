from __future__ import annotations

import json
from typing import Any, Iterable


def set_scores(expected: Iterable[str], predicted: Iterable[str]) -> tuple[float, float, float, float]:
    truth, output = set(expected), set(predicted)
    if not truth and not output:
        return 1.0, 1.0, 1.0, 1.0
    intersection = len(truth & output)
    precision = intersection / len(output) if output else 0.0
    recall = intersection / len(truth) if truth else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    union = len(truth | output)
    return precision, recall, f1, intersection / union if union else 1.0


def best_set_scores(candidates: Iterable[Iterable[str]], predicted: Iterable[str]) -> tuple[float, float, float, float]:
    values = [set_scores(candidate, predicted) for candidate in candidates]
    return max(values, key=lambda item: (item[2], item[3])) if values else set_scores((), predicted)


def repair_action_cost(encoded: str) -> float:
    action: dict[str, Any] = json.loads(encoded)
    operation = action["operation"]
    if operation in {"add_edge", "remove_edge"}:
        return 1.0
    if operation == "restore_attribute":
        node = action["node_id"]
    elif operation == "restore_node":
        node = action["node"]["id"]
    elif operation == "remove_node":
        node = action["node_id"]
    else:
        return 1.0
    return {
        "source": 1.0,
        "preprocessor": 2.0,
        "reference": 2.0,
        "calibration": 1.0,
        "canonical": 2.0,
        "reducer": 2.0,
        "explainer": 4.0,
        "model": 8.0,
        "output": 1.0,
    }.get(node, 20.0)
