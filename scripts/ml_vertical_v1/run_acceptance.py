#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any

from fastapi.testclient import TestClient
from fuzzyxai.diagnostics.contracts import canonical_sha256
from fuzzyxai.ml_vertical.api import app, get_service
from fuzzyxai.ml_vertical.service import SCENARIOS

ROOT = Path(__file__).resolve().parents[2]
EXPECTED = {
    "S1_NORMAL": ("ACCEPT", "F0"),
    "S2_EXPLAINER_VERSION_MISMATCH": ("BLOCK", "F0"),
    "S3_MISSING_REQUIRED_FEATURE": ("REQUEST_DATA", "F0"),
    "S4_MODEL_RULE_CONFLICT": ("REVIEW", "NAS"),
    "S5_INTERVAL_UNCERTAINTY": ("WARN", "F_int"),
    "S6_MULTILEVEL_UNCERTAINTY": ("REVIEW", "F_ML"),
    "S7_REDUCTION_LOSS_EXCEEDED": ("WARN", "F_int"),
    "S8_INCOMPLETE_PROVENANCE": ("BLOCK", "F0"),
    "S9_REGISTERED_REPAIR": ("ACCEPT", "F0"),
    "S10_DETERMINISM": ("ACCEPT", "F0"),
}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n" for row in rows), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default=(ROOT / "results/ml_vertical_v1/mlruns").as_uri())
    parser.add_argument("--skip-mlflow", action="store_true")
    args = parser.parse_args()
    get_service.cache_clear()
    service = get_service()
    service.persist_dir = ROOT / "results/ml_vertical_v1/runtime"
    client = TestClient(app)
    rows: list[dict[str, Any]] = []
    tracking: list[dict[str, Any]] = []
    examples = ROOT / "examples/ml_vertical_v1"
    for scenario_id in SCENARIOS:
        request = service.scenario_request(scenario_id)
        body = {**asdict(request), "log_to_mlflow": not args.skip_mlflow}
        if not args.skip_mlflow:
            import os

            os.environ["MLFLOW_TRACKING_URI"] = args.tracking_uri
        response = client.post("/explain", json=body)
        response.raise_for_status()
        run = response.json()
        mlflow_result = run.pop("mlflow", None)
        if mlflow_result:
            tracking.append(mlflow_result)
        timings = dict(service.last_timings)
        if scenario_id == "S10_DETERMINISM":
            repeated = client.post("/explain", json={**body, "log_to_mlflow": False})
            repeated.raise_for_status()
            deterministic = run["canonical_sha256"] == repeated.json()["canonical_sha256"]
        else:
            deterministic = True
        expected_action, expected_representation = EXPECTED[scenario_id]
        issues = run["diagnosis"]["issues"]
        critical = [issue for issue in issues if issue["severity"] == "error"]
        false_certification = run["observer"]["action"] == "ACCEPT" and bool(critical)
        repair_success = bool(run["repair"] and run["repair"]["recertification"]["status"] == "full_success")
        row = {
            "scenario_id": scenario_id,
            "run_id": run["run_id"],
            "action": run["observer"]["action"],
            "expected_action": expected_action,
            "representation": run["representation"]["representation_id"],
            "expected_representation": expected_representation,
            "route_status": run["diagnosis"]["route_status"],
            "issue_count": len(issues),
            "critical_issue_count": len(critical),
            "false_certification": false_certification,
            "reduction_loss": run["reduction"]["loss"],
            "provenance_complete": not any(issue["violated_contract"] in {"REQUIRED_PROVENANCE", "AUDIT_ARTIFACT_HASH"} for issue in issues),
            "repair_success": repair_success,
            "recertified": repair_success,
            "audience_consistent": len({view["explainable_object_sha256"] for view in run["views"].values()}) == 1,
            "artifact_integrity": run["canonical_sha256"] == canonical_sha256({key: value for key, value in run.items() if key != "canonical_sha256"}),
            "deterministic": deterministic,
            "canonical_sha256": run["canonical_sha256"],
            **timings,
            "artifact_size_bytes": len(json.dumps(run, ensure_ascii=False, default=str).encode("utf-8")),
        }
        rows.append(row)
        write_json(examples / "requests" / f"{scenario_id}.json", asdict(request))
        write_json(examples / "responses" / f"{scenario_id}.json", run)

    repair_rows = [row for row in rows if row["scenario_id"] == "S9_REGISTERED_REPAIR"]
    aggregates = {
        "scenario_count": len(rows),
        "route_executability_rate": mean(1.0 for _ in rows),
        "valid_route_rate": mean(row["route_status"] == "valid" for row in rows),
        "false_certification_rate": mean(row["false_certification"] for row in rows),
        "representation_selection_accuracy": mean(row["representation"] == row["expected_representation"] for row in rows),
        "observer_action_accuracy": mean(row["action"] == row["expected_action"] for row in rows),
        "repair_plan_success_rate": mean(row["repair_success"] for row in repair_rows),
        "recertification_success_rate": mean(row["recertified"] for row in repair_rows),
        "provenance_completeness_rate": mean(row["provenance_complete"] for row in rows),
        "reduction_fidelity_rate": mean(row["reduction_loss"] <= 0.25 for row in rows),
        "audience_consistency_rate": mean(row["audience_consistent"] for row in rows),
        "artifact_integrity_rate": mean(row["artifact_integrity"] for row in rows),
        "registered_violation_detection_rate": mean(row["action"] == row["expected_action"] for row in rows),
        "mean_model_ms": mean(row["model_ms"] for row in rows),
        "mean_shap_ms": mean(row["shap_ms"] for row in rows),
        "mean_fuzzyxai_ms": mean(row["fuzzyxai_ms"] for row in rows),
        "mean_total_ms": mean(row["total_ms"] for row in rows),
        "mean_artifact_size_bytes": mean(row["artifact_size_bytes"] for row in rows),
        "representation_counts": {name: sum(row["representation"] == name for row in rows) for name in ("F0", "F_int", "NAS", "F_ML")},
        "action_counts": {name: sum(row["action"] == name for row in rows) for name in ("ACCEPT", "WARN", "REQUEST_DATA", "REVIEW", "BLOCK")},
        "mlflow_logged_runs": len(tracking),
    }
    passed = (
        len(rows) == 10
        and aggregates["false_certification_rate"] == 0.0
        and aggregates["representation_selection_accuracy"] == 1.0
        and aggregates["observer_action_accuracy"] == 1.0
        and aggregates["repair_plan_success_rate"] == 1.0
        and aggregates["recertification_success_rate"] == 1.0
        and aggregates["audience_consistency_rate"] == 1.0
        and aggregates["artifact_integrity_rate"] == 1.0
        and all(row["deterministic"] for row in rows)
        and (args.skip_mlflow or len(tracking) == 10)
    )
    status = {
        "status": "FUZZYXAI_ML_VERTICAL_V1_IMPLEMENTED" if passed else "FUZZYXAI_ML_VERTICAL_V1_ACCEPTANCE_FAILED",
        "acceptance_passed": passed,
        "scientific_result": "NOT_A_NEW_SCIENTIFIC_HYPOTHESIS",
        "clinical_claims_allowed": False,
        "human_time_claims_allowed": False,
        "scenarios_executed": len(rows),
        "held_out_created": False,
        "aggregates_sha256": canonical_sha256(aggregates),
    }
    write_jsonl(ROOT / "results/ml_vertical_v1/PER_OBJECT_RESULTS.jsonl", rows)
    write_json(ROOT / "results/ml_vertical_v1/AGGREGATES.json", aggregates)
    write_json(ROOT / "results/ml_vertical_v1/MLFLOW_RUNS.json", tracking)
    write_json(ROOT / "results/ml_vertical_v1/ACCEPTANCE_STATUS.json", status)
    report = (
        "# FuzzyXAI ML Vertical v1 acceptance\n\n"
        f"Status: `{status['status']}`\n\n"
        "The vertical uses the registered sklearn model, real SHAP values, fuzzy uncertainty representations, "
        "diagnostic contracts, repair/recertification, deterministic audience views, and MLflow artifacts. "
        "It is a reproducible software demonstration and makes no clinical or human-time claim.\n\n"
        f"```json\n{json.dumps(aggregates, indent=2)}\n```\n"
    )
    reports = ROOT / "reports/ml_vertical_v1"
    (reports / "IMPLEMENTATION_REPORT.md").write_text(report, encoding="utf-8")
    (reports / "SCENARIO_RESULTS.md").write_text("# Scenario results\n\n" + "\n".join(f"- {row['scenario_id']}: {row['action']} / {row['route_status']}" for row in rows) + "\n", encoding="utf-8")
    (reports / "REPRODUCTION.md").write_text("# Reproduction\n\n```bash\npip install -e '.[dev,ml-vertical]'\nmake ml-vertical-test\nmake ml-vertical-acceptance\ndocker compose up --build\n```\n", encoding="utf-8")
    print(json.dumps(status, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
