from __future__ import annotations

from research_validation.adapters.external_classification_payload import make_payload


def make_regression_payload(experiment: dict) -> dict:
    return make_payload(experiment)
