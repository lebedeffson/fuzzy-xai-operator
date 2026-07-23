from __future__ import annotations


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    size = len(ordered)
    for index, (name, value) in enumerate(ordered):
        candidate = min(1.0, (size - index) * value)
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted

