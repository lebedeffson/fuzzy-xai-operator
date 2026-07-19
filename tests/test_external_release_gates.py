from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("pilot_scorer", ROOT / "scripts/score_comprehension_pilot.py")
assert SPEC and SPEC.loader
SCORER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCORER)


def _rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for participant in range(6):
        for scenario in ("forgetting_case", "rule_ablation", "image_similarity"):
            for mode in ("technical_baseline", "human_explanation"):
                human = mode == "human_explanation"
                rows.append(
                    {
                        "participant_id": f"P{participant + 1:02d}",
                        "role": "domain_specialist" if participant < 3 else "model_integrator",
                        "condition_order": "AB" if participant % 2 == 0 else "BA",
                        "scenario_id": scenario,
                        "mode": mode,
                        "decision_correct": "true",
                        "reasons_correct": "true",
                        "concern_correct": "true" if human else "false",
                        "reliability_correct": "true",
                        "action_correct": "true",
                        "limitation_correct": "true",
                        "provenance_correct": "true",
                        "similarity_correct": "true",
                        "counterfactual_correct": "true",
                        "native_surrogate_correct": "true",
                        "overtrust_error": "false",
                        "iou_misinterpreted_as_probability": "false",
                        "sensitivity_misinterpreted_as_recommendation": "false",
                        "unsupported_inference_count": "0",
                        "completion_time_sec": "90" if human else "100",
                        "subjective_clarity_1_5": "5" if human else "3",
                        "cognitive_load_1_5": "2" if human else "4",
                        "notes": "",
                    }
                )
    return rows


def test_empty_pilot_is_explicitly_not_run() -> None:
    report = SCORER.score_rows([])
    assert report["status"] == "planned_not_run"
    assert report["claim_allowed"] is False


def test_complete_real_shape_can_pass_without_weakening_gates() -> None:
    report = SCORER.score_rows(_rows())
    assert report["status"] == "pass"
    assert report["participant_count"] == 6
    assert report["scenario_count"] == 3


def test_missing_scenario_is_rejected() -> None:
    rows = [row for row in _rows() if row["scenario_id"] != "image_similarity"]
    with pytest.raises(ValueError, match="missing required scenarios"):
        SCORER.score_rows(rows)


def test_response_template_matches_scorer_contract() -> None:
    path = ROOT / "release_evidence/user_study/comprehension_pilot/response_template.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        assert set(next(csv.reader(handle))) >= {
            "participant_id",
            "scenario_id",
            "provenance_correct",
            "overtrust_error",
            "subjective_clarity_1_5",
        }
