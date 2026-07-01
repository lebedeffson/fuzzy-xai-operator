from __future__ import annotations

from .base import BaseAdapter


class IrisBiometricAdapter(BaseAdapter):
    repo_id = "lebedeffson/hybrid-xiris-biometric"
    scenario_id = "hybrid_xiris"
    required_fields = (
        "image_quality",
        "segmentation_quality",
        "model_match_signal",
        "alpha_accept",
        "alpha_block",
    )


def adapt_hybrid_xiris(payload: dict) -> object:
    return IrisBiometricAdapter().to_adapted_input(payload)
