#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "protocol/cross_pipeline_v1"
RESULTS = ROOT / "results/cross_pipeline_v1"
REPORTS = ROOT / "reports/cross_pipeline_v1"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_hashes() -> None:
    for name, expected in read_json(PROTOCOL / "SHA256SUMS.json").items():
        actual = hashlib.sha256((PROTOCOL / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"protocol hash mismatch: {name}")
    for relative, expected in read_json(RESULTS / "SHA256SUMS").items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"result hash mismatch: {relative}")


def main() -> None:
    verify_hashes()
    status = read_json(RESULTS / "FINAL_STATUS.json")
    if status["status"] != "FUZZYXAI_CROSS_PIPELINE_PRACTICAL_V1_SUPPORTED" or not status["supported"]:
        raise RuntimeError("release status does not satisfy the locked criteria")
    if not all(status["criteria"].values()):
        raise RuntimeError("one or more locked acceptance criteria failed")
    if status["parent_immutability"]["status"] != "PASS":
        raise RuntimeError("parent immutability failed")
    rows = pd.read_parquet(RESULTS / "PER_RUN_RESULTS.parquet")
    if len(rows) != 1000 or rows["pipeline_id"].nunique() != 5 or rows["mode_id"].nunique() != 5:
        raise RuntimeError("scored decision matrix is incomplete")
    registry = read_json(RESULTS / "MLFLOW_RUNS.json")
    if registry["expected"] != 1000 or registry["logged"] != 1000:
        raise RuntimeError("MLflow run registry is incomplete")
    if len({item["run_id"] for item in registry["runs"]}) != 1000:
        raise RuntimeError("MLflow run registry contains duplicate IDs")
    client = MlflowClient(tracking_uri=(RESULTS / "mlruns").resolve().as_uri())
    experiment = client.get_experiment_by_name("fuzzyxai-cross-pipeline-practical-v1")
    if experiment is None:
        raise RuntimeError("MLflow experiment is missing")
    runs = client.search_runs([experiment.experiment_id], max_results=2000)
    if len(runs) != 1000 or any(run.info.status != "FINISHED" for run in runs):
        raise RuntimeError("MLflow export does not contain 1000 completed runs")
    required_metrics = {
        "detected",
        "stage_correct",
        "contract_correct",
        "root_cause_correct",
        "false_certification",
        "evidence_completeness",
        "cut_size",
        "repair_success",
        "recertification_success",
        "new_critical_violations",
        "runtime_ms",
    }
    if any(not required_metrics.issubset(run.data.metrics) for run in runs):
        raise RuntimeError("one or more MLflow runs lacks required metrics")
    required_reports = {
        "PIPELINES.md",
        "MUTATION_BENCHMARK.md",
        "BASELINE_COMPARISON.md",
        "ROOT_CAUSE_ANALYSIS.md",
        "REPAIR_AND_RECERTIFICATION.md",
        "PERFORMANCE.md",
        "LIMITATIONS.md",
        "FINAL_REPORT.md",
        "REPRODUCTION.md",
    }
    if {path.name for path in REPORTS.glob("*.md")} != required_reports:
        raise RuntimeError("report set is incomplete or contains an unregistered report")
    print(json.dumps({"status": "PASS", "decisions": len(rows), "mlflow_runs": len(runs), "parent_files": status["parent_immutability"]["checked_files"]}))


if __name__ == "__main__":
    main()
