from __future__ import annotations

from fuzzyxai.core.types import AdaptedInput

EXAMPLES = ("hybrid_xiris", "medical_ecg_signal", "gd_anfis_shap", "beacon_xai", "gis_integro")


def list_examples() -> tuple[str, ...]:
    return EXAMPLES


def load_example(scenario_id: str) -> AdaptedInput:
    if scenario_id == "hybrid_xiris":
        from .hybrid_xiris import get_input
    elif scenario_id == "medical_ecg_signal":
        from .medical_ecg_signal import get_input
    elif scenario_id == "gd_anfis_shap":
        from .gd_anfis_shap import get_input
    elif scenario_id == "beacon_xai":
        from .beacon_xai import get_input
    elif scenario_id == "gis_integro":
        from .gis_integro import get_input
    else:
        raise KeyError(f"unknown FuzzyXAI example: {scenario_id}")
    return get_input()


def get_scenario_input(scenario_id: str) -> AdaptedInput:
    return load_example(scenario_id)


__all__ = ["list_examples", "load_example", "get_scenario_input"]
