from __future__ import annotations

import argparse
import json
import random
import resource
from dataclasses import replace
from pathlib import Path

import numpy as np

from baselines.h10 import IndependentRulesBaseline, SimpleOrBaseline, TypedRouteBaseline

from .common import ARTIFACT_ROOT, ROOT, load_yaml, write_csv, write_json
from .metrics import best_alternative_scores, normalize
from .mutations import mutate_route
from .run_confirmatory import _load_auditor
from .serialization import route_from_dict


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _plain_recertify(route, repair_nodes: tuple[str, ...]) -> bool:
    observed = dict(route.observed)
    for node in repair_nodes:
        if node in route.expected:
            observed[node] = route.expected[node]
    for field in route.mandatory_fields:
        if observed.get(field) in (None, "", (), []):
            return False
    for field, expected in route.expected.items():
        value = observed.get(field)
        if field == "calibration_age_days":
            if float(value or 0.0) > float(expected or 30.0):
                return False
        elif field == "reduction_loss":
            if float(value or 0.0) > float(expected or 0.1):
                return False
        elif value != expected:
            return False
    return True


def _drift_variant(route, index: int):
    # A legitimate context change updates expected and observed together and
    # therefore must not be treated as a contract fault.
    expected = dict(route.expected)
    observed = dict(route.observed)
    expected["runtime_context_id"] = f"clean-drift-{index % 17}"
    observed["runtime_context_id"] = expected["runtime_context_id"]
    return replace(route, expected=expected, observed=observed, route_id=f"{route.route_id}:drift:{index}")


def run(config_path: Path) -> None:
    config = load_yaml(config_path)
    replay = config["replay"]
    rng = random.Random(int(config["seed"]) + 900)
    clean_rows = _jsonl(ARTIFACT_ROOT / "routes" / "clean_routes.jsonl")
    routes = [route_from_dict(row["route"]) for row in clean_rows if row["split"] == "sealed_test"]
    if not routes:
        raise RuntimeError("no clean sealed routes available for replay")
    auditor = _load_auditor(config)
    methods = {
        "simple_or": SimpleOrBaseline().diagnose,
        "independent_if_else": IndependentRulesBaseline().diagnose,
        "typed_route": TypedRouteBaseline().diagnose,
        "full_h10": auditor.diagnose,
    }
    leaves = tuple(config["known_leaves"]) + tuple(config["held_out_leaves"])
    held_out = set(config["held_out_leaves"])
    incidents = []
    for index in range(int(replay["incidents"])):
        route = routes[(index * 37) % len(routes)]
        leaf = leaves[(index * 7) % len(leaves)]
        second = leaves[(index * 11 + 3) % len(leaves)] if index % 4 == 0 else None
        incident_leaves = (leaf, second) if second and second != leaf else (leaf,)
        phases = [
            mutate_route(route, incident_leaves, severity, unknown=any(item in held_out for item in incident_leaves))
            for severity in ("subtle", "moderate", "severe")
        ]
        start = 10_000 + index * max(1, ((int(replay["events"]) - 20_000) // int(replay["incidents"]))) + rng.randrange(0, 1000)
        for method, diagnose in methods.items():
            detected_phase = None
            result = None
            truth = None
            detected_route = None
            for phase, (mutated, phase_truth) in enumerate(phases):
                candidate = normalize(diagnose(mutated))
                if candidate["route_status"] != "valid":
                    detected_phase, result, truth, detected_route = phase, candidate, phase_truth, mutated
                    break
            if result is None:
                detected_route, truth = phases[-1]
                result = normalize(diagnose(detected_route))
            assert truth is not None and detected_route is not None
            _, _, repair_f1, _, _ = best_alternative_scores(truth.repair_sets, tuple(result["repair_nodes"]))
            recertified = _plain_recertify(detected_route, tuple(result.get("repair_fields", result["repair_nodes"])))
            extra = min(
                len(set(result["repair_nodes"]) - set(alt)) for alt in (truth.repair_sets or ((),))
            )
            incidents.append(
                {
                    "incident_id": f"incident-{index:03d}",
                    "method": method,
                    "model_line": index % int(replay["model_lines"]),
                    "start_event": start,
                    "unknown": any(item in held_out for item in incident_leaves),
                    "composite": len(incident_leaves) > 1,
                    "detected": detected_phase is not None,
                    "detection_delay": 50 * detected_phase if detected_phase is not None else 300,
                    "repair_delay": 10 * len(result["repair_nodes"]) + (0 if recertified else 50),
                    "repair_success": recertified and repair_f1 > 0.0,
                    "recertified": recertified,
                    "manual_review": not recertified or bool(result["abstained"]),
                    "hard_block": result["route_status"] == "invalid" and not result["repair_nodes"],
                    "erroneous_repair_actions": extra,
                    "trace_size": len(result["trace"]),
                }
            )
    normal_routes = [_drift_variant(route, index) for index, route in enumerate(routes[: min(10_000, len(routes))])]
    false_alerts = {
        method: sum(normalize(diagnose(route))["route_status"] != "valid" for route in normal_routes)
        for method, diagnose in methods.items()
    }
    clean_stream_failures = {method: count for method, count in false_alerts.items() if count > 0}
    if clean_stream_failures:
        raise RuntimeError(
            "H10 replay clean-stream sanity check failed; results must not be published: "
            f"{clean_stream_failures}"
        )
    summaries = []
    for method in methods:
        rows = [row for row in incidents if row["method"] == method]
        summaries.append(
            {
                "method": method,
                "incident_recall": float(np.mean([row["detected"] for row in rows])),
                "false_alerts_per_10k": 10_000 * false_alerts[method] / max(len(normal_routes), 1),
                "detection_delay_p95": float(np.quantile([row["detection_delay"] for row in rows], 0.95)),
                "repair_delay_p95": float(np.quantile([row["repair_delay"] for row in rows], 0.95)),
                "repair_success": float(np.mean([row["repair_success"] for row in rows])),
                "recertification_rate": float(np.mean([row["recertified"] for row in rows])),
                "manual_load": float(np.mean([row["manual_review"] for row in rows])),
                "hard_block_rate": float(np.mean([row["hard_block"] for row in rows])),
                "trace_size": float(np.mean([row["trace_size"] for row in rows])),
                "erroneous_repair_actions": int(sum(row["erroneous_repair_actions"] for row in rows)),
            }
        )
    write_csv(ARTIFACT_ROOT / "replay" / "incident_results.csv", incidents)
    write_csv(ARTIFACT_ROOT / "replay" / "method_summary.csv", summaries)
    write_json(
        ARTIFACT_ROOT / "replay" / "summary.json",
        {
            "events": int(replay["events"]),
            "incidents": int(replay["incidents"]),
            "model_lines": int(replay["model_lines"]),
            "normal_stream_source": "clean_routes_with_joint_expected_observed_context_drift",
            "normal_stream_contains_mutated_fault_routes": False,
            "clean_stream_sanity_status": "PASS",
            "clean_stream_false_alert_ceiling": 0,
            "delayed_evidence": True,
            "partial_and_failed_repair": True,
            "service_times_preassigned_to_favor_h10": False,
            "repair_delay_rule": "10 events per proposed repair target plus 50 events when recertification fails",
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "methods": summaries,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "h10_v19_protocol.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
