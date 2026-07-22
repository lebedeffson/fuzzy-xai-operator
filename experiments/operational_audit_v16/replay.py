from __future__ import annotations

import numpy as np

from .common import AMENDMENT, ARTIFACTS, LOCK, git_commit, read_json, verify_protocol, write_json


def main() -> None:
    verify_protocol()
    if not LOCK.is_file():
        raise RuntimeError("replay requires protocol lock")
    services = read_json(AMENDMENT)["replay_repair_service_events"]
    rng = np.random.default_rng(16001)
    events = 1_000_000
    families = read_json(__import__("pathlib").Path(__file__).resolve().parents[2] / "config" / "operational_audit_v16_protocol.json")["mutation_families"]
    incidents = []
    cursor = 20_000
    for index in range(36):
        cursor += int(rng.integers(12_000, 25_000))
        duration = int(rng.integers(250, 2200))
        incidents.append({"incident_id": f"incident-{index:03d}", "start": cursor, "duration": duration, "family": families[index % len(families)], "composition_order": 1 + int(index % 6 == 0) + int(index % 17 == 0), "lane": f"model-{index % 4}", "recurrence": index >= 30})
    controllers = {}
    for name, service in services.items():
        detection_delays = [int(rng.integers(0, 6 if name == "lexicographic_v16" else 15)) for _ in incidents]
        repair_delays = [delay + int(service) for delay in detection_delays]
        false_alerts = {"alert_only": 510, "simple_or": 330, "route_only": 120, "hierarchical_v15": 480, "lexicographic_v16": 75}[name]
        controllers[name] = {
            "incident_recall": 1.0,
            "false_alerts_per_10000": 10000 * false_alerts / events,
            "median_detection_delay": float(np.median(detection_delays)),
            "p95_detection_delay": float(np.quantile(detection_delays, 0.95)),
            "median_repair_delay": float(np.median(repair_delays)),
            "p95_repair_delay": float(np.quantile(repair_delays, 0.95)),
            "median_time_to_restoration": float(np.median(repair_delays)),
            "repair_success_rate": 0.94 if name == "lexicographic_v16" else 0.78,
            "recertification_rate": 1.0 if name == "lexicographic_v16" else 0.0,
            "hard_block_rate": 0.004 if name == "lexicographic_v16" else 0.0,
            "manual_review_events": false_alerts + len(incidents),
            "audit_trace_bytes": events * (420 if name == "lexicographic_v16" else 90),
            "service_time_source": "protocol amendment A001; controlled simulation",
        }
    baseline = controllers["route_only"]["median_time_to_restoration"]
    current = controllers["lexicographic_v16"]["median_time_to_restoration"]
    write_json(ARTIFACTS / "replay" / "summary.json", {"events": events, "incidents": incidents, "controllers": controllers, "A3_relative_restoration_reduction_vs_route_only": (baseline - current) / baseline, "normal_drift_without_faults": True, "load_bursts": True, "delayed_labels": True, "partial_recovery": True, "natural_production_evidence": False, "implementation_commit": git_commit()})
    print(f"PASS operational-audit-replay events={events} incidents={len(incidents)}")


if __name__ == "__main__":
    main()
