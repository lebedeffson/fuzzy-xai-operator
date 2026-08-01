from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/cross_pipeline_v1"
PROTOCOL = ROOT / "protocol/cross_pipeline_v1"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_protocol_was_locked_before_scoring() -> None:
    lock = read_json(PROTOCOL / "PIPELINE_PROTOCOL_LOCK.json")
    assert lock["status"] == "LOCKED_BEFORE_SCORING"
    assert lock["gold_available_to_modes"] is False
    assert lock["case_count"] == 200 and lock["decision_count"] == 1000


def test_official_matrix_is_complete() -> None:
    rows = pd.read_parquet(RESULTS / "PER_RUN_RESULTS.parquet")
    assert len(rows) == 1000
    assert rows["pipeline_id"].nunique() == 5
    assert rows["mutation_family"].nunique() == 8
    assert rows["mutation_level"].nunique() == 5
    assert rows["mode_id"].nunique() == 5


def test_supported_status_is_derived_from_all_locked_criteria() -> None:
    status = read_json(RESULTS / "FINAL_STATUS.json")
    assert all(status["criteria"].values())
    assert status["supported"] is True
    assert status["status"] == "FUZZYXAI_CROSS_PIPELINE_PRACTICAL_V1_SUPPORTED"


def test_parent_release_is_immutable() -> None:
    audit = read_json(RESULTS / "PARENT_IMMUTABILITY.json")
    assert audit == {"checked_files": 2164, "failures": [], "status": "PASS"}


def test_mlflow_registry_has_one_run_per_decision() -> None:
    registry = read_json(RESULTS / "MLFLOW_RUNS.json")
    assert registry["expected"] == registry["logged"] == 1000
    assert len({row["run_id"] for row in registry["runs"]}) == 1000
    assert {row["artifact_count"] for row in registry["runs"]} == {12}


def test_full_fuzzyxai_meets_practical_metrics() -> None:
    status = read_json(RESULTS / "FINAL_STATUS.json")["o_fuzzyxai"]
    assert status["false_certification_count"] == 0
    assert status["cross_stage_contract_recall"] == 1.0
    assert status["stage_accuracy"] == 1.0
    assert status["contract_accuracy"] == 1.0
    assert status["root_cause_accuracy"] == 1.0
    assert status["repair_success"] == 1.0
    assert status["full_recertification"] == 1.0


def test_strong_pairwise_baseline_is_not_artificially_weakened() -> None:
    rows = pd.read_csv(RESULTS / "BASELINE_COMPARISON.csv").set_index("mode_id")
    assert rows.loc["B_PAIRWISE_RULES", "cross_stage_contract_recall"] == 1.0
    assert rows.loc["B_PAIRWISE_RULES", "contract_accuracy"] == 1.0
    assert rows.loc["B_PAIRWISE_RULES", "root_cause_accuracy"] == 0.0
    assert rows.loc["B_PAIRWISE_RULES", "redundant_repairs_mean"] > 0.0


def test_hierarchical_bootstrap_uses_locked_unit() -> None:
    intervals = pd.read_csv(RESULTS / "BOOTSTRAP_INTERVALS.csv")
    local = intervals[(intervals["comparison"] == "O_vs_LOCAL") & (intervals["metric"] == "contract_correct")].iloc[0]
    assert local["ci95_lower"] > 0.0
    assert set(intervals["iterations"]) == {10_000}


def test_no_natural_or_human_claim_is_made() -> None:
    status = read_json(RESULTS / "FINAL_STATUS.json")
    assert status["natural_incidents_evaluated"] is False
    assert status["human_utility_evaluated"] is False
    assert status["docx_pdf_modified"] is False
