from __future__ import annotations

from fuzzyxai.adapters.signal_quality import adapt_medical_ecg_signal


def get_input():
    return adapt_medical_ecg_signal(
        {
            "sampling_rate": 250,
            "quality_score": 0.58,
            "noise_segments": 1,
            "missing_fragments": 2,
            "baseline_instability": 0.21,
        }
    )
