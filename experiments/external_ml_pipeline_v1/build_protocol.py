from __future__ import annotations

import inspect
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import shap
import sklearn
from fuzzyxai.external_adapters.manifest import ManifestExternalPipelineAdapter

from .benchmark import FAULTS, MODE_IDS, REPAIR_OPERATIONS
from .external_runners import SEED, SPECS
from .io import sha256, write_json

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "protocol/external_ml_pipeline_v1"
FIXTURES = ROOT / "experiments/external_ml_pipeline_v1/fixtures"
UPSTREAM = ROOT / "experiments/external_ml_pipeline_v1/upstream"
BOOTSTRAP_SEED = 1729
BOOTSTRAP_ITERATIONS = 10_000
CORE_FILES = (
    "framework/fuzzyxai/fuzzyxai/diagnostics/contracts.py",
    "framework/fuzzyxai/fuzzyxai/diagnostics/service.py",
    "framework/fuzzyxai/fuzzyxai/diagnostics/validator.py",
    "framework/fuzzyxai/fuzzyxai/diagnostics/minimal_cut.py",
    "framework/fuzzyxai/fuzzyxai/diagnostics/repair_planner.py",
    "framework/fuzzyxai/fuzzyxai/diagnostics/repair_executor.py",
    "framework/fuzzyxai/fuzzyxai/diagnostics/recertification.py",
    "framework/fuzzyxai/fuzzyxai/pipelines/practical.py",
)
IMPLEMENTATION_FILES = (
    "framework/fuzzyxai/fuzzyxai/external_adapters/base.py",
    "framework/fuzzyxai/fuzzyxai/external_adapters/manifest.py",
    "experiments/external_ml_pipeline_v1/external_runners.py",
    "experiments/external_ml_pipeline_v1/benchmark.py",
    "experiments/external_ml_pipeline_v1/run_evaluation.py",
)


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _parent_manifest() -> dict[str, str]:
    values = {}
    for directory in (ROOT / "protocol", ROOT / "results", ROOT / "reports"):
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            relative = path.relative_to(ROOT).as_posix()
            if "/external_ml_pipeline_v1/" not in f"/{relative}":
                values[relative] = sha256(path)
    return values


def main() -> None:
    PROTOCOL.mkdir(parents=True, exist_ok=True)
    repositories = []
    for index, spec in enumerate(SPECS, 1):
        upstream_dir = UPSTREAM / f"ext{index}_{'sklearn' if index == 1 else 'shap' if index == 2 else 'mlflow' if index == 3 else 'lime'}"
        snapshots = {path.relative_to(ROOT).as_posix(): sha256(path) for path in sorted(upstream_dir.glob("*")) if path.is_file()}
        repositories.append({**asdict(spec), "upstream_snapshot_sha256": snapshots})
    write_json(PROTOCOL / "REPOSITORY_LOCK.json", {"repositories": repositories, "count": len(repositories), "network_required_after_lock": False})
    pipelines = []
    for spec in SPECS:
        variants = {}
        for variant in ("baseline", "retrained"):
            root = FIXTURES / spec.pipeline_id / variant
            variants[variant] = {path.name: sha256(path) for path in sorted(root.glob("*")) if path.is_file() and path.name != "SHA256SUMS.json"}
        pipelines.append({"pipeline_id": spec.pipeline_id, "task_type": spec.task_type, "variants": variants, "split_seed": SEED})
    write_json(
        PROTOCOL / "PIPELINE_LOCK.json",
        {
            "pipelines": pipelines,
            "libraries": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
                "shap": shap.__version__,
                "mlflow": mlflow.__version__,
                "lime": "0.2.0.1",
            },
        },
    )
    adapter_path = ROOT / "framework/fuzzyxai/fuzzyxai/external_adapters/manifest.py"
    adapter_source = inspect.getsource(ManifestExternalPipelineAdapter)
    write_json(
        PROTOCOL / "ADAPTER_LOCK.json",
        {
            "adapter_sha256": sha256(adapter_path),
            "shared_adapter_loc": len([line for line in adapter_source.splitlines() if line.strip()]),
            "adapter_files": 1,
            "diagnostic_logic_in_adapter": False,
            "core_files_changed": 0,
            "pipeline_specific_auditor_conditions": 0,
            "core_sha256": {path: sha256(ROOT / path) for path in CORE_FILES},
        },
    )
    matrix = []
    for spec in SPECS:
        for fault in FAULTS:
            matrix.append(
                {
                    "pipeline_id": spec.pipeline_id,
                    "repository_commit": spec.repository_commit,
                    "case_id": fault.case_id,
                    "variant": fault.variant,
                    "expected_invalid": fault.contract_id is not None,
                    "expected_stage": fault.stage,
                    "expected_contract": fault.contract_id,
                    "expected_root_cause": fault.contract_id,
                    "expected_action": fault.action,
                    "repairable": fault.repair_operation is not None,
                    "dependent_contracts": fault.dependent_contracts,
                }
            )
    write_json(PROTOCOL / "FAULT_MATRIX.json", {"case_count": len(matrix), "cases": matrix})
    write_json(
        PROTOCOL / "BASELINE_LOCK.json",
        {
            "modes": MODE_IDS,
            "definitions": {
                "B_LOCAL_STRONG": "component-only schema, finite, required-field, artifact-hash and local explanation checks",
                "B_PAIRWISE_RULES": "all adjacent-stage checks without global cause, DiagnosticCut or full recertification",
                "B_MLFLOW_QUERY": "registered parameters, tags, versions, paths and hashes only",
                "B_GREEDY_CROSS_STAGE": "first violation by stage and contract order",
                "O_FUZZYXAI": "unchanged RouteGraph auditor, global cut, repair executor, rollback and full recertification",
            },
        },
    )
    write_json(
        PROTOCOL / "METRIC_LOCK.json",
        {
            "diagnosis": [
                "violation_recall",
                "stage_accuracy",
                "contract_accuracy",
                "component_accuracy",
                "root_cause_accuracy",
                "action_accuracy",
                "false_certification",
                "false_blocking",
                "evidence_completeness",
            ],
            "causal": ["root_cause_rank", "dependent_symptom_count", "diagnostic_cut_size", "proposed_repair_count", "redundant_repair_count"],
            "repair": ["repair_success", "full_recertification", "new_critical_violations", "rollback_success"],
            "transfer": ["contract_reuse_rate", "evidence_extraction_coverage", "adapter_loc", "new_contract_count", "core_changed_files"],
            "unit": "external_repository x fault_family",
        },
    )
    write_json(
        PROTOCOL / "STATISTICAL_PLAN.json",
        {
            "unit": "external_repository x fault_family",
            "bootstrap": "hierarchical repository then fault family",
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            "confidence": 0.95,
            "paired_test": "exact sign test",
            "multiple_testing": "Holm",
            "p_value_release_gate": False,
        },
    )
    write_json(
        PROTOCOL / "ACCEPTANCE_CRITERIA.json",
        {
            "external_pipelines": 4,
            "completed_cases": 40,
            "completed_decisions": 200,
            "maximum_false_certification": 0,
            "maximum_false_blocking": 0,
            "minimum_cross_stage_recall": 0.90,
            "minimum_stage_accuracy": 0.90,
            "minimum_contract_accuracy": 0.90,
            "minimum_root_cause_accuracy": 0.90,
            "minimum_evidence_completeness": 0.95,
            "minimum_contract_reuse": 0.75,
            "minimum_full_recertification": 0.90,
            "maximum_new_critical_violations": 0,
            "rollback_success": 1.0,
            "core_changed_files": 0,
            "pipeline_specific_auditor_branches": 0,
            "requires_graph_advantage": True,
        },
    )
    write_json(PROTOCOL / "REPAIR_REGISTRY.json", {"operations": REPAIR_OPERATIONS, "operation_count": len(set(REPAIR_OPERATIONS.values()))})
    parent = _parent_manifest()
    write_json(PROTOCOL / "PARENT_FILES_SHA256.json", {"file_count": len(parent), "files": parent})
    lock_files = tuple(PROTOCOL.glob("*.json"))
    write_json(
        PROTOCOL / "PROTOCOL_LOCK.json",
        {
            "protocol": "FUZZYXAI_EXTERNAL_ML_PIPELINE_VALIDATION_V1",
            "status": "LOCKED_BEFORE_SCORING",
            "implementation_commit": _git_head(),
            "case_count": 40,
            "decision_count": 200,
            "new_contracts": 0,
            "gold_available_to_modes": False,
            "implementation_sha256": {path: sha256(ROOT / path) for path in IMPLEMENTATION_FILES},
            "protocol_sha256": {path.name: sha256(path) for path in lock_files},
        },
    )
    write_json(PROTOCOL / "SHA256SUMS.json", {path.name: sha256(path) for path in sorted(PROTOCOL.glob("*.json")) if path.name != "SHA256SUMS.json"})


if __name__ == "__main__":
    main()
