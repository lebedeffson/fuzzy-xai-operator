from __future__ import annotations

import json
from pathlib import Path

import pytest
from fuzzyxai.experiments.h10_c7r import (
    final_status,
    load_held_out_inputs,
    repository_cluster_bootstrap,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(value) + "\n" for value in values),
        encoding="utf-8",
    )


def _graph(repository: str) -> dict[str, object]:
    return {
        "repository": repository,
        "revision": "buggy",
        "nodes": [],
        "edges": [],
        "evidence": [],
        "obligations": [],
        "limitations": [],
    }


def _observable(repository: str) -> dict[str, object]:
    return {
        "incident_id": "incident",
        "repository": repository,
        "split": "held_out",
        "runtime_evidence_status": "BUG_REPRODUCED_WITH_TRACE",
        "repository_symbol_count": 100,
        "graph_path": "graph.json",
        "runtime_events_path": "events.jsonl",
        "query": {
            "issue": "failure",
            "failing_tests": ["tests/test_x.py::test_x"],
            "traceback": "tests/test_x.py:1",
            "assertion": "assert x",
        },
    }


def test_contract_metric_is_not_a_support_gate() -> None:
    rows = [
        {
            "repository": f"owner/repo-{index}",
            "r5_hit_at_20": 1.0,
            "baseline_hit_at_160": 1.0,
            "r5_candidate_count": 20,
            "r5_search_space_reduction": 0.98,
            "baseline_search_space_reduction": 0.84,
        }
        for index in range(12)
        for _ in range(4)
    ]
    bootstrap = {
        "ci_lower": 0.10,
        "ci_upper": 0.18,
        "mean_difference": 0.14,
        "iterations": 20000,
        "seed": 7102026,
        "repository_count": 12,
    }

    status = final_status(
        rows,
        bootstrap,
        gold_leakage=0,
        method_signature_passed=True,
        budget_signature_passed=True,
        single_official_scoring=True,
    )

    assert status["status"] == "H10_C7R_SUPPORTED"
    assert status["contract_macro_f1_is_gate"] is False
    assert "contract_macro_f1" not in status["checks"]


def test_repository_bootstrap_is_deterministic_and_clustered() -> None:
    rows = [
        {"repository": "a/repo", "delta_reduction": 0.1},
        {"repository": "a/repo", "delta_reduction": 0.2},
        {"repository": "b/repo", "delta_reduction": 0.3},
    ]

    first = repository_cluster_bootstrap(rows, iterations=1000, seed=7)
    second = repository_cluster_bootstrap(rows, iterations=1000, seed=7)

    assert first == second
    assert first["repository_count"] == 2
    assert first["ci_lower"] > 0


def test_excluded_repository_is_rejected(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    gold = tmp_path / "gold.jsonl"
    exclusions = tmp_path / "exclusions.json"
    _write_json(tmp_path / "graph.json", _graph("owner/repo"))
    (tmp_path / "events.jsonl").write_text("", encoding="utf-8")
    _write_jsonl(manifest, [_observable("owner/repo")])
    _write_jsonl(
        gold,
        [
            {
                "incident_id": "incident",
                "atoms": [{"file_path": "a.py", "symbol": "f"}],
            }
        ],
    )
    _write_json(exclusions, {"excluded_repositories": ["owner/repo"]})

    with pytest.raises(ValueError, match="excluded repository"):
        load_held_out_inputs(
            manifest,
            gold,
            exclusions,
            minimum_incidents=1,
            minimum_repositories=1,
        )


def test_gold_fields_are_rejected_from_observable(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    gold = tmp_path / "gold.jsonl"
    exclusions = tmp_path / "exclusions.json"
    row = _observable("new/repo")
    row["gold_symbol"] = "secret"
    _write_jsonl(manifest, [row])
    _write_jsonl(
        gold,
        [
            {
                "incident_id": "incident",
                "atoms": [{"file_path": "a.py", "symbol": "f"}],
            }
        ],
    )
    _write_json(exclusions, {"excluded_repositories": []})

    with pytest.raises(ValueError, match="Gold"):
        load_held_out_inputs(
            manifest,
            gold,
            exclusions,
            minimum_incidents=1,
            minimum_repositories=1,
        )


def test_locked_h10_c7a_result_is_preserved() -> None:
    status = json.loads(
        Path(
            "results/h10_c7a/development/H10_C7A_DEVELOPMENT_GATES.json"
        ).read_text(encoding="utf-8")
    )

    assert status["status"] == "H10_C7A_BLOCKED_DEVELOPMENT_GATE"
    assert status["scientific_result"] == "NOT_EVALUATED"
    assert status["gate_passed"] is False


def test_h10_c7r_budgets_are_locked() -> None:
    budget = json.loads(
        Path("protocol/h10_c7r/H10_C7R_BUDGET_LOCK.json").read_text(
            encoding="utf-8"
        )
    )

    assert budget["primary"] == {"method": "R5", "budget": 20}
    assert budget["baseline"] == {"method": "B_BM25", "budget": 160}
    assert budget["minimum_recall"] == 0.8
