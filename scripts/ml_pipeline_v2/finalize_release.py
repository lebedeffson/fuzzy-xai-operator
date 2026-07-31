#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/ml_pipeline_v2"


def load(name: str) -> dict[str, Any]:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def main() -> int:
    aggregates = load("AGGREGATES.json")
    parent = load("PARENT_IMMUTABILITY.json")
    rest = load("REST_SMOKE.json")
    ui = load("UI_SMOKE.json")
    docker = load("DOCKER_SMOKE.json")
    mlflow = load("MLFLOW_ARTIFACT_AUDIT.json")
    core_passed = all(
        (
            aggregates["scenario_pass_count"] == 18,
            aggregates["false_certifications"] == 0,
            aggregates["stage_localization_accuracy"] == 1.0,
            aggregates["violated_contract_accuracy"] == 1.0,
            aggregates["representation_accuracy"] == 1.0,
            aggregates["observer_action_accuracy"] == 1.0,
            aggregates["registered_repair_success"] == 1.0,
            aggregates["full_recertification_success"] == 1.0,
            aggregates["new_critical_violations_after_repair"] == 0,
            aggregates["canonical_output_determinism"] == 1.0,
            aggregates["mlflow_runs"] == 18,
        )
    )
    checks = {
        "core_acceptance": core_passed,
        "rest_smoke": rest.get("status") == "PASS",
        "ui_smoke": ui.get("status") == "PASS",
        "docker_smoke": docker.get("status") == "PASS",
        "mlflow_artifact_completeness": mlflow.get("status") == "PASS" and mlflow.get("complete_runs") == 18,
        "parent_immutability": parent.get("status") == "PASS",
    }
    passed = all(checks.values())
    payload = {
        "status": "FUZZYXAI_ML_PIPELINE_V2_IMPLEMENTED" if passed else "FUZZYXAI_ML_PIPELINE_V2_RELEASE_CHECK_FAILED",
        "acceptance_passed": passed,
        "scenario_count": aggregates["scenario_count"],
        "scientific_result": "ENGINEERING_IMPLEMENTATION",
        "clinical_claims_allowed": False,
        "human_time_claims_allowed": False,
        "docx_pdf_modified": False,
        "checks": checks,
        "aggregates_sha256": canonical_hash(aggregates),
    }
    (RESULTS / "FINAL_STATUS.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
