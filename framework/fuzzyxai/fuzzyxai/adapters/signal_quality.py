from __future__ import annotations

from .base import BaseAdapter


class SignalQualityAdapter(BaseAdapter):
    repo_id = "control/ecg-signal-quality"
    scenario_id = "medical_ecg_signal"
    required_fields = ("sampling_rate", "quality_score", "noise_segments", "missing_fragments", "baseline_instability")

    def validate(self, payload: dict) -> None:
        super().validate(payload)
        from fuzzyxai.ecg_quality import assess_ecg_quality

        assess_ecg_quality(payload)


def adapt_medical_ecg_signal(payload: dict) -> object:
    return SignalQualityAdapter().to_adapted_input(payload)
