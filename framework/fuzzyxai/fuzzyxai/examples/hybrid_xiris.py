from __future__ import annotations

from fuzzyxai.adapters.iris_biometric import adapt_hybrid_xiris


def get_input():
    payload = {
        "image_quality": 0.31,
        "segmentation_quality": 0.27,
        "model_match_signal": 0.88,
        "alpha_accept": 0.82,
        "alpha_block": 0.91,
    }
    return adapt_hybrid_xiris(payload)
