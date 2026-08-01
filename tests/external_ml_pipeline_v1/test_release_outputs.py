from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "protocol/external_ml_pipeline_v1"
RESULTS = ROOT / "results/external_ml_pipeline_v1"


def read_json(name: str):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def read_csv(name: str):
    with (RESULTS / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_protocol_was_locked_before_scoring() -> None:
    lock = json.loads((PROTOCOL / "PROTOCOL_LOCK.json").read_text(encoding="utf-8"))
    assert lock["status"] == "LOCKED_BEFORE_SCORING"
    assert lock["case_count"] == 40 and lock["decision_count"] == 200
    assert lock["gold_available_to_modes"] is False and lock["new_contracts"] == 0


def test_official_matrix_has_40_cases_and_200_decisions() -> None:
    rows = read_csv("PER_CASE.csv")
    assert len(rows) == 200
    assert len({(item["pipeline_id"], item["case_id"]) for item in rows}) == 40
    assert len({item["pipeline_id"] for item in rows}) == 4
    assert len({item["mode_id"] for item in rows}) == 5


def test_supported_status_is_derived_from_every_locked_gate() -> None:
    status = read_json("FINAL_STATUS.json")
    assert status["status"] == "FUZZYXAI_EXTERNAL_ML_PIPELINE_VALIDATION_V1_SUPPORTED"
    assert status["supported"] and all(status["criteria"].values())


def test_full_mode_has_no_false_certification_or_blocking() -> None:
    result = read_json("FINAL_STATUS.json")["o_fuzzyxai"]
    assert result["false_certification_count"] == 0
    assert result["false_blocking_count"] == 0


def test_full_mode_meets_localization_and_recertification_gates() -> None:
    result = read_json("FINAL_STATUS.json")["o_fuzzyxai"]
    assert result["cross_stage_contract_recall"] == 1.0
    assert result["stage_accuracy"] == result["contract_accuracy"] == result["root_cause_accuracy"] == 1.0
    assert result["repair_success"] == result["full_recertification"] == result["rollback_success"] == 1.0


def test_pairwise_baseline_is_strong_but_has_no_global_root() -> None:
    rows = {item["mode_id"]: item for item in read_csv("BASELINE_COMPARISON.csv")}
    pairwise = rows["B_PAIRWISE_RULES"]
    assert float(pairwise["cross_stage_contract_recall"]) == 1.0
    assert float(pairwise["contract_accuracy"]) == 1.0
    assert float(pairwise["root_cause_accuracy"]) == 0.0
    assert float(pairwise["redundant_repair_mean"]) > 0.0


def test_contract_transfer_uses_no_new_contract_or_core_branch() -> None:
    rows = read_csv("CONTRACT_REUSE.csv")
    assert len(rows) == 4
    assert {float(item["contract_reuse_rate"]) for item in rows} == {1.0}
    assert {int(item["new_contracts"]) for item in rows} == {0}
    assert {int(item["core_files_changed"]) for item in rows} == {0}
    assert {int(item["auditor_pipeline_specific_conditions"]) for item in rows} == {0}


def test_mlflow_has_one_run_and_twelve_artifacts_per_decision() -> None:
    registry = read_json("MLFLOW_RUNS.json")
    assert registry["expected"] == registry["logged"] == 200
    assert len({item["run_id"] for item in registry["runs"]}) == 200
    assert {item["artifact_count"] for item in registry["runs"]} == {12}


def test_parent_and_external_sources_are_immutable() -> None:
    assert read_json("PARENT_IMMUTABILITY.json")["status"] == "PASS"
    assert read_json("EXTERNAL_SOURCE_IMMUTABILITY.json")["status"] == "PASS"


def test_bootstrap_uses_preregistered_hierarchy() -> None:
    rows = read_csv("BOOTSTRAP_INTERVALS.csv")
    assert {int(item["seed"]) for item in rows} == {1729}
    assert {int(item["iterations"]) for item in rows} == {10_000}
    local = next(item for item in rows if item["comparison"] == "O_vs_LOCAL" and item["metric"] == "contract_correct")
    assert float(local["ci95_lower"]) > 0.0


def test_claim_scope_remains_controlled_pipeline_validation() -> None:
    status = read_json("FINAL_STATUS.json")
    assert status["natural_incidents_evaluated"] is False
    assert status["human_utility_evaluated"] is False
    assert status["source_bug_localization_evaluated"] is False
    assert status["docx_pdf_modified"] is False
