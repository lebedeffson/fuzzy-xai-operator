#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
import sklearn
from fuzzyxai.pipelines.practical import MODE_IDS, MUTATION_FAMILIES, REPAIR_FOR_CONTRACT
from fuzzyxai.pipelines.registry import list_pipeline_registrations

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "protocol/cross_pipeline_v1"
BOOTSTRAP_SEED = 1729
BOOTSTRAP_ITERATIONS = 10_000
LOCKED_SOURCES = (
    "framework/fuzzyxai/fuzzyxai/pipelines/registry.py",
    "framework/fuzzyxai/fuzzyxai/pipelines/practical.py",
    "framework/fuzzyxai/fuzzyxai/pipelines/practical_api.py",
    "framework/fuzzyxai/fuzzyxai/pipelines/practical_tracking.py",
    "apps/cross_pipeline_practical.py",
    "data/cross_pipeline_v1/mixed_features.csv",
    "scripts/cross_pipeline_v1/run_evaluation.py",
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    PROTOCOL.mkdir(parents=True, exist_ok=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    pipelines = []
    for item in list_pipeline_registrations():
        payload = asdict(item)
        payload["registration_sha256"] = item.sha256
        pipelines.append(payload)
    matrix = []
    for pipeline in pipelines:
        for family in MUTATION_FAMILIES.values():
            for level in family.levels:
                matrix.append(
                    {
                        "pipeline_id": pipeline["pipeline_id"],
                        "mutation_family": family.family_id,
                        "category": family.category,
                        "mutation_level": level.level_id,
                        "description": level.description,
                        "expected_stage": level.expected_stage,
                        "expected_contract": level.expected_contract,
                        "expected_action": level.expected_action,
                    }
                )
    write_json(
        PROTOCOL / "PIPELINE_LOCK.json",
        {
            "protocol": "FUZZYXAI_CROSS_PIPELINE_PRACTICAL_V1",
            "implementation_commit": head,
            "pipeline_count": len(pipelines),
            "pipelines": pipelines,
            "libraries": {
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
                "shap": shap.__version__,
            },
        },
    )
    write_json(PROTOCOL / "MUTATION_MATRIX.json", {"case_count": len(matrix), "cases": matrix})
    write_json(
        PROTOCOL / "BASELINE_LOCK.json",
        {
            "modes": list(MODE_IDS),
            "definitions": {
                "B_LOCAL_STRONG": "local component checks only",
                "B_PAIRWISE_RULES": "all registered pairwise checks without graph cut or recertification",
                "B_MLFLOW_QUERY": "deterministic query of already registered MLflow fields",
                "B_GREEDY_CROSS_STAGE": "first violation by stage and contract order",
                "O_FUZZYXAI": "RouteGraph, causal cut, registered repair, rollback and full recertification",
            },
        },
    )
    write_json(
        PROTOCOL / "METRIC_LOCK.json",
        {
            "diagnosis": [
                "violation_recall",
                "local_contract_recall",
                "cross_stage_contract_recall",
                "stage_accuracy",
                "contract_accuracy",
                "component_accuracy",
                "root_cause_accuracy",
                "action_accuracy",
                "false_certification",
                "evidence_completeness",
            ],
            "global_analysis": ["diagnostic_cut_size", "reported_symptoms", "proposed_repairs", "redundant_repairs", "root_cause_rank"],
            "repair": ["repair_success", "target_contract_repaired", "full_recertification", "new_critical_violations", "rollback_success"],
            "performance": ["runtime_ms", "graph_build_ms", "audit_ms", "cut_ms", "recertification_ms", "peak_rss_kb", "artifact_bytes"],
            "unit": "pipeline x mutation_family; mutation levels are repeated controlled measurements",
        },
    )
    write_json(
        PROTOCOL / "STATISTICAL_PLAN.json",
        {
            "bootstrap": "hierarchical: resample pipelines, then mutation families within pipeline",
            "seed": BOOTSTRAP_SEED,
            "iterations": BOOTSTRAP_ITERATIONS,
            "confidence": 0.95,
            "primary_comparisons": [
                ["O_FUZZYXAI", "B_LOCAL_STRONG"],
                ["O_FUZZYXAI", "B_PAIRWISE_RULES"],
                ["O_FUZZYXAI", "B_GREEDY_CROSS_STAGE"],
            ],
            "p_value_release_gate": False,
        },
    )
    write_json(
        PROTOCOL / "ACCEPTANCE_CRITERIA.json",
        {
            "minimum_pipelines": 4,
            "minimum_completed_cases": 160,
            "maximum_false_certification": 0,
            "minimum_cross_stage_contract_recall": 0.95,
            "minimum_stage_accuracy": 0.95,
            "minimum_contract_accuracy": 0.95,
            "minimum_root_cause_accuracy": 0.90,
            "evidence_completeness": 1.0,
            "minimum_repair_success": 0.95,
            "minimum_full_recertification": 0.95,
            "maximum_new_critical_violations": 0,
            "requires_local_advantage": True,
            "requires_graph_advantage": True,
        },
    )
    write_json(PROTOCOL / "REPAIR_REGISTRY.json", {"operations_by_contract": REPAIR_FOR_CONTRACT})
    parent_files = {}
    for parent in (ROOT / "protocol", ROOT / "results", ROOT / "reports"):
        for path in sorted(item for item in parent.rglob("*") if item.is_file()):
            relative = path.relative_to(ROOT).as_posix()
            if relative.startswith(("protocol/cross_pipeline_v1/", "results/cross_pipeline_v1/", "reports/cross_pipeline_v1/")):
                continue
            parent_files[relative] = sha256(path)
    write_json(PROTOCOL / "PARENT_FILES_SHA256.json", {"file_count": len(parent_files), "files": parent_files})
    lock = {
        "protocol": "FUZZYXAI_CROSS_PIPELINE_PRACTICAL_V1",
        "status": "LOCKED_BEFORE_SCORING",
        "implementation_commit": head,
        "locked_source_sha256": {relative: sha256(ROOT / relative) for relative in LOCKED_SOURCES},
        "protocol_file_sha256": {
            path.name: sha256(path) for path in sorted(PROTOCOL.glob("*.json")) if path.name not in {"PIPELINE_PROTOCOL_LOCK.json", "SHA256SUMS.json"}
        },
        "case_count": len(matrix),
        "mode_count": len(MODE_IDS),
        "decision_count": len(matrix) * len(MODE_IDS),
        "gold_available_to_modes": False,
    }
    write_json(PROTOCOL / "PIPELINE_PROTOCOL_LOCK.json", lock)
    checks = {path.name: sha256(path) for path in sorted(PROTOCOL.glob("*.json")) if path.name != "SHA256SUMS.json"}
    write_json(PROTOCOL / "SHA256SUMS.json", checks)
    print(
        json.dumps(
            {"status": "LOCKED", "pipelines": len(pipelines), "cases": len(matrix), "decisions": len(matrix) * len(MODE_IDS), "parent_files": len(parent_files)}
        )
    )


if __name__ == "__main__":
    main()
