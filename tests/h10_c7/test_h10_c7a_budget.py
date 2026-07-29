from __future__ import annotations

import json
from pathlib import Path

import pytest
from fuzzyxai.experiments.h10_c7a import (
    _append_tail,
    load_budget_inputs,
    select_budget_locks,
)
from fuzzyxai.repository_diagnostics.guided_retrieval import RankedSymbol


def _ranked(node_id: str, score: float) -> RankedSymbol:
    return RankedSymbol(
        node_id,
        f"{node_id}.py",
        node_id,
        score,
        ("test",),
        1,
        (),
    )


def test_extended_budget_never_reorders_frozen_prefix() -> None:
    frozen = (_ranked("first", 1.0), _ranked("second", 0.5))
    extended = (
        _ranked("tail-with-higher-score", 100.0),
        _ranked("first", 200.0),
    )

    result = _append_tail(frozen, extended, limit=3)

    assert [item.node_id for item in result] == [
        "first",
        "second",
        "tail-with-higher-score",
    ]


def test_budget_lock_uses_smallest_budget_reaching_recall() -> None:
    summary = [
        {
            "method": method,
            "budget": budget,
            "incident_count": 40,
            "available_incident_count": 40,
            "recall": recall,
            "mean_search_space_reduction": 1.0 - budget / 1000,
        }
        for method, values in {
            "R5": ((5, 0.70), (10, 0.85), (20, 0.90)),
            "B_GREEDY": ((5, 0.40), (10, 0.60), (20, 0.80)),
        }.items()
        for budget, recall in values
    ]

    result = select_budget_locks(summary)

    assert result["method_budgets"]["R5"]["k_star"] == 10
    assert result["method_budgets"]["B_GREEDY"]["k_star"] == 20
    assert result["selected_baseline"] == "B_GREEDY"


def test_unavailable_dense_methods_do_not_block_structural_budget() -> None:
    summary = [
        {
            "method": "R5",
            "budget": 10,
            "incident_count": 40,
            "available_incident_count": 40,
            "recall": 0.85,
            "mean_search_space_reduction": 0.99,
        },
        {
            "method": "B_DENSE",
            "budget": 10,
            "incident_count": 40,
            "available_incident_count": 0,
            "recall": 0.0,
            "mean_search_space_reduction": 0.0,
        },
    ]

    result = select_budget_locks(summary)

    assert result["method_budgets"]["R5"]["k_star"] == 10
    assert "B_DENSE" in result["unavailable_optional_methods"]


def test_observable_manifest_rejects_gold_fields(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    gold = tmp_path / "gold.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "incident_id": "i",
                "repository": "owner/repo",
                "split": "development",
                "gold_symbol": "secret",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    gold.write_text(
        json.dumps(
            {
                "incident_id": "i",
                "atoms": [
                    {
                        "file_path": "a.py",
                        "symbol": "f",
                        "contract": "CONFIGURATION",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Gold"):
        load_budget_inputs(manifest, gold)


def test_r5v_remains_blocked_audit() -> None:
    root = Path(__file__).resolve().parents[2]
    status = json.loads(
        (root / "results/h10_c7/verification/R5V_STATUS.json").read_text(
            encoding="utf-8"
        )
    )

    assert status["status"] == "H10_C7_R5V_BLOCKED_AUDIT"
    assert status["scientific_result"] == "NOT_EVALUATED"
    assert status["v0_executed"] is False
    assert status["v1_executed"] is False
