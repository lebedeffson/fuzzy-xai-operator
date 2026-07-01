from __future__ import annotations

from fuzzyxai.adapters.counterevidence import adapt_beacon_xai


def get_input():
    return adapt_beacon_xai(
        {
            "single_object_fragments": 100,
            "support_fragments": 70,
            "counter_fragments": 30,
            "batch_objects": 100,
            "objects_with_counterevidence": 83,
            "audit_reports": 12,
            "checks_full": 64,
            "checks_reduced": 11,
        }
    )
