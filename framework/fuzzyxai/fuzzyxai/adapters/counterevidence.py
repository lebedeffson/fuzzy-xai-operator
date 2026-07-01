from __future__ import annotations

from .base import BaseAdapter


class CounterevidenceAdapter(BaseAdapter):
    repo_id = "fims9000/BeaconXAI"
    scenario_id = "beacon_xai"
    required_fields = (
        "single_object_fragments",
        "support_fragments",
        "counter_fragments",
        "batch_objects",
        "objects_with_counterevidence",
        "audit_reports",
        "checks_full",
        "checks_reduced",
    )


def adapt_beacon_xai(payload: dict) -> object:
    return CounterevidenceAdapter().to_adapted_input(payload)
