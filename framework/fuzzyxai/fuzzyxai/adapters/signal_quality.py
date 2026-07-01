from __future__ import annotations

from .base import BaseAdapter


class SignalQualityAdapter(BaseAdapter):
    repo_id = "control/ecg-signal-quality"
    scenario_id = "medical_ecg_signal"
    required_fields = ("sampling_rate", "quality_score", "noise_segments", "missing_fragments", "baseline_instability")


def adapt_medical_ecg_signal(payload: dict) -> object:
    return SignalQualityAdapter().to_adapted_input(payload)
