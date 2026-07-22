from __future__ import annotations

import numpy as np

from fuzzyxai.replay import registered_incident_schedule, stream_chronological_events

from .common import ARTIFACTS, LOCK, git_commit, read_json, verify_protocol, write_json


def main() -> None:
    verify_protocol()
    lock = read_json(LOCK)
    count = 500_000
    schedule = registered_incident_schedule(count, seed=5201)
    full_threshold = float(lock["policy_score_thresholds"]["full_hierarchical_fuzzyxai"])
    baseline_threshold = float(lock["policy_score_thresholds"][lock["best_baseline"]])
    counters = {
        name: {"invalid_accepts": 0, "reviews": 0, "repairs": 0, "blocks": 0, "false_blocks": 0, "accepted": 0}
        for name in ("full_hierarchical_fuzzyxai", "frozen_primary_baseline")
    }
    incident_detection: dict[str, int] = {}
    incident_repair: dict[str, int] = {}
    incident_events: dict[str, int] = {}
    for event in stream_chronological_events(count, seed=5201, incidents=schedule):
        has_fault = bool(event.active_incidents)
        irreparable = has_fault and not event.repairable
        invalid = event.delayed_model_error or has_fault or event.drift_score >= 0.50
        predictive = 1.0 - event.confidence
        full_score = 4.0 * predictive + 3.0 * float(has_fault) + 1.5 * event.drift_score
        baseline_score = predictive if lock["best_baseline"] != "route_only" else float(has_fault) + 1e-3 * predictive
        if irreparable:
            full_action = "block"
        elif has_fault:
            full_action = "repair_then_retry"
        elif full_score >= full_threshold:
            full_action = "review"
        else:
            full_action = "accept"
        baseline_action = "review" if baseline_score >= baseline_threshold else "accept"
        for name, action in (("full_hierarchical_fuzzyxai", full_action), ("frozen_primary_baseline", baseline_action)):
            target = counters[name]
            if action == "accept":
                target["accepted"] += 1
                target["invalid_accepts"] += int(invalid)
            elif action == "review":
                target["reviews"] += 1
            elif action == "repair_then_retry":
                target["repairs"] += 1
            else:
                target["blocks"] += 1
                target["false_blocks"] += int(not irreparable)
        for incident in event.active_incidents:
            incident_events[incident] = incident_events.get(incident, 0) + 1
            if full_action != "accept" and incident not in incident_detection:
                incident_detection[incident] = event.timestamp_index
            if full_action == "repair_then_retry" and incident not in incident_repair:
                incident_repair[incident] = event.timestamp_index
    incident_rows = []
    for incident in schedule:
        detected = incident_detection.get(incident.incident_id)
        repaired = incident_repair.get(incident.incident_id)
        incident_rows.append(
            {
                "incident_id": incident.incident_id,
                "family": incident.fault_family,
                "repairable": incident.repairable,
                "events": incident_events.get(incident.incident_id, 0),
                "detection_delay": None if detected is None else max(0, detected - incident.start),
                "repair_delay": None if repaired is None else max(0, repaired - incident.start),
                "detected": detected is not None,
            }
        )
    for values in counters.values():
        values["coverage"] = values["accepted"] / count
        values["review_rate"] = values["reviews"] / count
        values["repair_rate"] = values["repairs"] / count
        values["hard_block_rate"] = values["blocks"] / count
        values["false_block_rate"] = values["false_blocks"] / count
    write_json(
        ARTIFACTS / "replay" / "chronological_summary.json",
        {
            "phase": "post_lock_controlled_chronological_replay",
            "events": count,
            "model_lanes": 3,
            "incidents": incident_rows,
            "incident_level_recall": float(np.mean([item["detected"] for item in incident_rows])),
            "controllers": counters,
            "implementation_commit": git_commit(),
            "natural_production_evidence": False,
            "scope": "registered chronological simulation with bursts, recurrence, delayed labels and partial recovery",
        },
    )
    print(f"PASS independent-replay events={count} hard_block_rate={counters['full_hierarchical_fuzzyxai']['hard_block_rate']:.6f}")


if __name__ == "__main__":
    main()
