from __future__ import annotations

from typing import Any


def observe_risk(values: dict[str, Any], reduction: dict[str, Any]) -> dict[str, Any]:
    components = values.get(
        "risk_components",
        {
            "model_signal": float(values["model_match_signal"]),
            "block_rule": float(values["alpha_block"]),
            "source_conflict": 1.0,
            "reduction_component": float(reduction["r_delta"]),
        },
    )
    weights = values.get(
        "risk_weights",
        {"model_signal": 0.35, "block_rule": 0.25, "source_conflict": 0.20, "reduction_component": 0.20},
    )
    rho = round(sum(float(components[key]) * float(weights[key]) for key in components), 3)
    source_conflict = (
        float(values["model_match_signal"]) >= float(values["alpha_accept"])
        and float(values["image_quality"]) < 0.5
        and float(values["segmentation_quality"]) < 0.5
    )
    chi_crit = 1 if source_conflict else 0
    return {
        "rho": rho,
        "chi_crit": chi_crit,
        "risk_zone": "critical" if chi_crit else "normal",
        "status": "blocked" if chi_crit else "passed",
        "reason_ru": "критический конфликт качества источника и модельного сигнала" if chi_crit else "критическая зона не активна",
        "value_source": "computed",
        "components": components,
        "weights": weights,
    }
