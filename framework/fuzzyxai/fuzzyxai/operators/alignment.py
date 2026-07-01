from __future__ import annotations

from typing import Any


def compute_alignment(values: dict[str, Any]) -> dict[str, Any]:
    components = values.get(
        "alignment_components",
        {"d_mu": 0.39, "d_R": 0.40, "d_u": 0.5675, "d_tau": 0.0},
    )
    weights = values.get(
        "alignment_weights",
        {"d_mu": 0.25, "d_R": 0.35, "d_u": 0.20, "d_tau": 0.20},
    )
    gamma = round(sum(float(components[key]) * float(weights[key]) for key in components), 6)
    gamma_max = float(values.get("gamma_max", 0.40))
    return {
        "gamma": gamma,
        "gamma_max": gamma_max,
        "status": "warning" if gamma > gamma_max * 0.75 else "passed",
        "value_source": "computed",
        "components": components,
        "weights": weights,
    }
