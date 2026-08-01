#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
from fastapi.testclient import TestClient
from fuzzyxai.ml_vertical.api import app, get_pipeline_service
from fuzzyxai.ml_vertical.pipeline import ALL_SCENARIOS, V2_SCENARIOS

ROOT = Path(__file__).resolve().parents[2]
V1_EXPECTED = {
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


def repository_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def legacy_signature(payload: dict[str, Any]) -> tuple[Any, ...]:
    diagnosis = payload["diagnosis"]
    return (
        payload["scenario_id"],
        payload["observer"]["action"],
        payload["representation"]["representation_id"],
        diagnosis["route_status"],
        tuple(sorted(item["violated_contract"] for item in diagnosis["issues"])),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default=(ROOT / "results/ml_pipeline_v2/mlruns").as_uri())
    parser.add_argument("--skip-mlflow", action="store_true")
    parser.add_argument("--git-commit", default=repository_head())
    args = parser.parse_args()
    os.environ["MLFLOW_TRACKING_URI"] = args.tracking_uri
    os.environ["FUZZYXAI_GIT_COMMIT"] = args.git_commit
    get_pipeline_service.cache_clear()
    service = get_pipeline_service()
    service.persist_dir = ROOT / "results/ml_pipeline_v2/runtime"
    client = TestClient(app)
    rows: list[dict[str, Any]] = []
    mlflow_runs: list[dict[str, Any]] = []
    required_node_fields = {"stage", "component_id", "component_type", "version", "input_schema", "output_schema", "configuration", "artifact_sha256", "evidence_refs", "execution_status"}

    for scenario_id in ALL_SCENARIOS:
        response = client.post(
            f"/api/v1/pipeline/scenario/{scenario_id}",
            json={"scenario_id": scenario_id, "log_to_mlflow": not args.skip_mlflow},
        )
        response.raise_for_status()
        run = response.json()
        mlflow_result = run.pop("mlflow", None)
        if mlflow_result:
            mlflow_runs.append(mlflow_result)
        repeated = client.post(f"/api/v1/pipeline/scenario/{scenario_id}", json={"scenario_id": scenario_id})
        repeated.raise_for_status()
        deterministic = run["canonical_sha256"] == repeated.json()["canonical_sha256"]
        action = run["diagnosis"]["recommended_action"]
        representation = run["representation"]["representation_id"]
        if scenario_id in V1_EXPECTED:
            expected_action, expected_representation = V1_EXPECTED[scenario_id]
            baseline = json.loads((ROOT / "examples/ml_vertical_v1/responses" / f"{scenario_id}.json").read_text(encoding="utf-8"))
            expected_stage = expected_contract = None
            legacy_unchanged = legacy_signature(run["legacy_vertical"]) == legacy_signature(baseline)
            localization_correct = True
        else:
            expected = V2_SCENARIOS[scenario_id]
            expected_action = expected["action"]
            expected_representation = "F0"
            expected_stage = expected["stage"]
            expected_contract = expected["contract"]
            legacy_unchanged = True
            localization_correct = (
                run["diagnosis"]["failed_stage"] == expected_stage
                and run["diagnosis"]["violated_contract"] == expected_contract
            )
        violations = run["contract_report"]["violations"]
        critical = [item for item in violations if item["severity"] == "HIGH"]
        recertified = bool(run["recertification"] and run["recertification"].get("full_recertification"))
        false_certification = action == "ACCEPT" and bool(critical) and not recertified
        expected_repair = scenario_id in {"S9_REGISTERED_REPAIR", "S13_PREPROCESSOR_FULL_FIT", "S14_FEATURE_ORDER", "S16_MODEL_ARTIFACT_TAMPER", "S18_MISSING_EXPLANATION_PROVENANCE"}
        repair_success = recertified if expected_repair else True
        evidence_bound = not run["diagnosis"]["violated_contract"] or bool(run["diagnosis"]["evidence_refs"])
        node_schema_valid = all(required_node_fields.issubset(node) for node in run["route_graph"]["nodes"])
        scenario_pass = all(
            (
                action == expected_action,
                representation == expected_representation,
                localization_correct,
                deterministic,
                legacy_unchanged,
                not false_certification,
                repair_success,
                evidence_bound,
                node_schema_valid,
            )
        )
        rows.append(
            {
                "scenario_id": scenario_id,
                "scenario_pass": scenario_pass,
                "pipeline_status": run["pipeline_status"],
                "failed_stage": run["diagnosis"]["failed_stage"],
                "expected_stage": expected_stage,
                "violated_contract": run["diagnosis"]["violated_contract"],
                "expected_contract": expected_contract,
                "action": action,
                "expected_action": expected_action,
                "representation": representation,
                "expected_representation": expected_representation,
                "localization_correct": localization_correct,
                "legacy_v1_unchanged": legacy_unchanged,
                "evidence_bound": evidence_bound,
                "false_certification": false_certification,
                "repair_expected": expected_repair,
                "repair_success": repair_success,
                "recertification_success": recertified if expected_repair else None,
                "new_critical_violations": int(run["recertification"].get("new_critical_violations", 0)) if run["recertification"] else 0,
                "deterministic": deterministic,
                "node_schema_valid": node_schema_valid,
                "runtime_ms": float(run["runtime_ms"]),
                "canonical_sha256": run["canonical_sha256"],
            }
        )

    runtimes = [row["runtime_ms"] for row in rows]
    new_rows = [row for row in rows if row["scenario_id"] in V2_SCENARIOS]
    repair_rows = [row for row in rows if row["repair_expected"]]
    aggregates = {
        "scenario_count": len(rows),
        "scenario_pass_count": sum(row["scenario_pass"] for row in rows),
        "valid_scenarios": sum(row["pipeline_status"] == "VALID" for row in rows),
        "intentionally_invalid_scenarios": sum(row["failed_stage"] is not None for row in rows),
        "detected_v2_violations": sum(row["violated_contract"] is not None for row in new_rows),
        "missed_v2_violations": sum(not row["localization_correct"] for row in new_rows),
        "false_certifications": sum(row["false_certification"] for row in rows),
        "stage_localization_accuracy": float(np.mean([row["localization_correct"] for row in new_rows])),
        "violated_contract_accuracy": float(np.mean([row["violated_contract"] == row["expected_contract"] for row in new_rows])),
        "representation_accuracy": float(np.mean([row["representation"] == row["expected_representation"] for row in rows])),
        "observer_action_accuracy": float(np.mean([row["action"] == row["expected_action"] for row in rows])),
        "registered_repair_success": float(np.mean([row["repair_success"] for row in repair_rows])),
        "full_recertification_success": float(np.mean([row["recertification_success"] for row in repair_rows])),
        "new_critical_violations_after_repair": sum(row["new_critical_violations"] for row in repair_rows),
        "canonical_output_determinism": float(np.mean([row["deterministic"] for row in rows])),
        "v1_semantic_immutability": float(np.mean([row["legacy_v1_unchanged"] for row in rows[:10]])),
        "repairs_attempted": len(repair_rows),
        "repairs_completed": sum(row["repair_success"] for row in repair_rows),
        "recertifications_completed": sum(bool(row["recertification_success"]) for row in repair_rows),
        "runtime_median_ms": median(runtimes),
        "runtime_p95_ms": float(np.percentile(runtimes, 95)),
        "mlflow_runs": len(mlflow_runs),
    }
    passed = (
        aggregates["scenario_pass_count"] == 18
        and aggregates["false_certifications"] == 0
        and aggregates["stage_localization_accuracy"] == 1.0
        and aggregates["violated_contract_accuracy"] == 1.0
        and aggregates["representation_accuracy"] == 1.0
        and aggregates["observer_action_accuracy"] == 1.0
        and aggregates["registered_repair_success"] == 1.0
        and aggregates["full_recertification_success"] == 1.0
        and aggregates["new_critical_violations_after_repair"] == 0
        and aggregates["canonical_output_determinism"] == 1.0
        and aggregates["v1_semantic_immutability"] == 1.0
        and (args.skip_mlflow or aggregates["mlflow_runs"] == 18)
    )
    status = {
        "status": "FUZZYXAI_ML_PIPELINE_V2_CORE_ACCEPTANCE_PASS" if passed else "FUZZYXAI_ML_PIPELINE_V2_ACCEPTANCE_FAILED",
        "acceptance_passed": passed,
        "scenario_count": len(rows),
        "scientific_result": "ENGINEERING_IMPLEMENTATION",
        "clinical_claims_allowed": False,
        "human_time_claims_allowed": False,
        "docx_pdf_modified": False,
        "aggregates_sha256": canonical_hash(aggregates),
    }
    results = ROOT / "results/ml_pipeline_v2"
    write_jsonl(results / "SCENARIO_RESULTS.jsonl", rows)
    write_json(results / "AGGREGATES.json", aggregates)
    write_json(results / "FINAL_STATUS.json", status)
    write_json(results / "MLFLOW_RUNS.json", mlflow_runs)
    reports = ROOT / "reports/ml_pipeline_v2"
    reports.mkdir(parents=True, exist_ok=True)
    table = ["| Scenario | Stage | Contract | Action | Repair | Recertification |", "|---|---|---|---|---|---|"]
    table.extend(
        f"| {row['scenario_id']} | {row['failed_stage'] or '-'} | {row['violated_contract'] or '-'} | {row['action']} | {'PASS' if row['repair_expected'] and row['repair_success'] else '-'} | {'PASS' if row['recertification_success'] else '-'} |"
        for row in rows
    )
    (reports / "SCENARIO_REPORT.md").write_text("# ML Pipeline v2 scenarios\n\n" + "\n".join(table) + "\n", encoding="utf-8")
    (reports / "IMPLEMENTATION_REPORT.md").write_text("# ML Pipeline v2 implementation\n\nThe implementation executes real deterministic data preparation, splitting, StandardScaler fit/transform, LogisticRegression training, model serialization, inference, SHAP, FuzzyXAI diagnosis, registered repair and full recertification.\n", encoding="utf-8")
    (reports / "REPAIR_REPORT.md").write_text(f"# Repair report\n\nRepairs attempted: `{aggregates['repairs_attempted']}`. Repairs completed: `{aggregates['repairs_completed']}`. Full recertifications: `{aggregates['recertifications_completed']}`. New critical violations: `{aggregates['new_critical_violations_after_repair']}`.\n", encoding="utf-8")
    (reports / "FINAL_REPORT.md").write_text(f"# FuzzyXAI ML Pipeline v2\n\nStatus: `{status['status']}`\n\n```json\n{json.dumps(aggregates, indent=2)}\n```\n\nThis is an engineering implementation result, not a clinical or human-time evaluation.\n", encoding="utf-8")
    (reports / "REPRODUCTION.md").write_text("# Reproduction\n\n```bash\npip install -e '.[dev,ml-vertical]'\npython scripts/ml_pipeline_v2/run_acceptance.py\ndocker compose up --build\n```\n", encoding="utf-8")
    print(json.dumps(status, indent=2))
    return 0 if passed else 1


def canonical_hash(payload: Any) -> str:
    return __import__("hashlib").sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
