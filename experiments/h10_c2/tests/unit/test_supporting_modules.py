from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import pytest

from h10_c2.adjudication.agreement import categorical_agreement
from h10_c2.adjudication.resolve_disagreements import unresolved
from h10_c2.audit.claim_audit import validate_claim_status
from h10_c2.data import generate_cases
from h10_c2.data.case_validator import validate_case
from h10_c2.data.graph_diff import changed_subjects
from h10_c2.data.split_builder import split_counts
from h10_c2.manifest import build_manifest
from h10_c2.metrics.aggregation import means_by_method
from h10_c2.metrics.safety_metrics import safety_rates
from h10_c2.oracle.transaction_reversal import apply_transaction, inverse_action, reverse_transaction
from h10_c2.random_state import generators
from h10_c2.repair.execution_audit import validate_audit
from h10_c2.reporting.figures import plot_metric
from h10_c2.reporting.tables import write_rows
from h10_c2.statistics.aggregation import select_baseline
from h10_c2.statistics.multiple_testing import holm_adjust
from h10_c2.statistics.sensitivity_analysis import per_pipeline_effects


def test_small_supporting_algorithms(tmp_path: Path) -> None:
    assert categorical_agreement(["a", "b"], ["a", "x"]) == 0.5
    with pytest.raises(ValueError):
        categorical_agreement([], [])
    assert unresolved(["a", "b"], {"a": "resolved"}) == ["b"]
    validate_claim_status("not_evaluated", False)
    with pytest.raises(ValueError):
        validate_claim_status("supported", False)
    assert split_counts(10, {"development": 0.3, "sealed": 0.7}) == {"development": 3, "sealed": 7}
    with pytest.raises(ValueError):
        split_counts(0, {"sealed": 1.0})
    assert safety_rates([]) == {"false_certification": 0.0, "false_block": 0.0}
    assert safety_rates([{"false_certification": True, "false_block": False}])["false_certification"] == 1.0
    rows = [{"method": "a", "score": 1}, {"method": "a", "score": 0}, {"method": "b", "score": 0}]
    assert means_by_method(rows, "score") == {"a": 0.5, "b": 0.0}
    assert select_baseline([{**row, "optimal_cut_set_membership": row["score"]} for row in rows]) == "a"
    assert holm_adjust({"a": 0.01, "b": 0.04}) == {"a": 0.02, "b": 0.04}
    paired = [
        {"pipeline": "p", "case_id": "1", "method": "fuzzyxai_v21", "m": 1},
        {"pipeline": "p", "case_id": "1", "method": "a", "m": 0},
    ]
    assert per_pipeline_effects(paired, "m", "a") == {"p": 1.0}
    py_rng, np_rng = generators(1)
    assert py_rng.random() == generators(1)[0].random()
    assert np_rng.random() == generators(1)[1].random()
    text = tmp_path / "evidence.txt"
    text.write_text("evidence", encoding="utf-8")
    assert build_manifest([text], tmp_path / "manifest.json")["files"][0]["path"] == str(text)
    write_rows(tmp_path / "table.csv", rows)
    assert list(csv.DictReader((tmp_path / "table.csv").open(encoding="utf-8")))
    write_rows(tmp_path / "empty.csv", [])
    plot_metric(tmp_path / "plot.png", ["a"], [1.0], "metric")
    assert (tmp_path / "plot.png").is_file()


def test_case_validation_and_graph_diff() -> None:
    case = generate_cases("development", 20, seed=8)[12]
    validate_case(case)
    assert changed_subjects(case.clean_route, case.observed_route)
    with pytest.raises(ValueError, match="identity"):
        validate_case(replace(case, case_id=""))
    duplicate = {**case.observed_route, "nodes": [case.observed_route["nodes"][0]] * 2}
    with pytest.raises(ValueError, match="duplicate"):
        validate_case(replace(case, observed_route=duplicate))


def test_transaction_round_trip_and_audit() -> None:
    case = next(item for item in generate_cases("development", 30, seed=9) if item.transactions)
    transaction = case.transactions[0]
    applied = apply_transaction(case.clean_route, transaction)
    restored = reverse_transaction(applied, transaction)
    assert restored == case.clean_route
    assert inverse_action(transaction)["target"] == transaction.target_id
    trace = [
        {
            "step": 0,
            "target": transaction.target_id,
            "precondition_passed": True,
            "status": "completed",
            "before_sha256": "a",
            "after_sha256": "b",
        }
    ]
    validate_audit(trace)
    with pytest.raises(ValueError, match="incomplete"):
        validate_audit([{"step": 0}])

