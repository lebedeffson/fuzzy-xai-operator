from __future__ import annotations

from typing import Any

from fuzzyxai.core.thresholds import HYBRID_XIRIS_THRESHOLDS


def compute_reduction(values: dict[str, Any]) -> dict[str, Any]:
    components = values.get(
        "reduction_components",
        {"hybrid_delta": 0.106811},
    )
    weights = values.get("reduction_weights", {"hybrid_delta": 1.0})
    delta = round(sum(float(components[key]) * float(weights[key]) for key in components), 6)
    kappa_delta = float(values.get("kappa_delta", HYBRID_XIRIS_THRESHOLDS["kappa_delta"]))
    r_delta = round(min(1.0, kappa_delta * delta), 4)
    delta_max = float(values.get("delta_max", HYBRID_XIRIS_THRESHOLDS["delta_max"]))
    return {
        "delta": delta,
        "r_delta": r_delta,
        "delta_max": delta_max,
        "kappa_delta": kappa_delta,
        "status": "passed" if delta <= delta_max else "blocked",
        "value_source": "computed",
        "components": components,
        "weights": weights,
    }
