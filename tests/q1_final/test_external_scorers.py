from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.q1_final.score_comprehension import REQUIRED_COLUMNS, score as score_comprehension
from scripts.q1_final.score_domain_review import score as score_domain
from scripts.q1_final.score_expert_action_review import score as score_expert


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_comprehension_scorer_accepts_complete_realistic_panel(tmp_path: Path) -> None:
    stimuli = [f"stimulus_{index:03d}" for index in range(12)]
    assignments = tmp_path / "assignments.json"
    assignments.write_text(json.dumps([{"assignment_slot": "slot_001", "stimulus_ids": stimuli}]), encoding="utf-8")
    rows: list[dict[str, object]] = []
    for participant in range(24):
        for stimulus in stimuli:
            for condition in ("A", "B"):
                rows.append(
                    {
                        "participant_hash": f"participant_{participant:03d}",
                        "assignment_slot": "slot_001",
                        "stimulus_id": stimulus,
                        "condition": condition,
                        "consent": "true",
                        "attention_pass": "true",
                        "decision_correct": "true",
                        "reason_correct": "true",
                        "limitation_correct": str(condition == "B").lower(),
                        "action_correct": str(condition == "B").lower(),
                        "unsafe_overtrust": "false",
                        "completion_seconds": "8.0",
                    }
                )
    responses = tmp_path / "responses.csv"
    _write_csv(responses, sorted(REQUIRED_COLUMNS), rows)
    result = score_comprehension(responses, assignments)
    assert result["status"] == "supported"
    assert result["valid_participants"] == 24
    assert result["participant_records_generated_by_scorer"] is False


def test_comprehension_scorer_rejects_duplicate_rows(tmp_path: Path) -> None:
    assignments = tmp_path / "assignments.json"
    assignments.write_text(json.dumps([{"assignment_slot": "slot_001", "stimulus_ids": ["s1"]}]), encoding="utf-8")
    row = {field: "true" for field in REQUIRED_COLUMNS}
    row.update(
        {
            "participant_hash": "p1",
            "assignment_slot": "slot_001",
            "stimulus_id": "s1",
            "condition": "A",
            "completion_seconds": "8.0",
        }
    )
    responses = tmp_path / "responses.csv"
    _write_csv(responses, sorted(REQUIRED_COLUMNS), [row, row])
    with pytest.raises(ValueError, match="duplicate"):
        score_comprehension(responses, assignments)


def test_expert_scorer_uses_shared_panel_and_consensus(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = []
    for reviewer in range(3):
        for object_index in range(100):
            consensus = "accept" if object_index % 3 else "review"
            rows.append(
                {
                    "reviewer_hash": f"reviewer_{reviewer}",
                    "object_id": f"object_{object_index:03d}",
                    "condition": "expert_only",
                    "action": consensus,
                }
            )
    for object_index in range(100):
        consensus = "accept" if object_index % 3 else "review"
        rows.append(
            {
                "reviewer_hash": "reviewer_0",
                "object_id": f"object_{object_index:03d}",
                "condition": "adaptive_fuzzyxai",
                "action": consensus,
            }
        )
        rows.append(
            {
                "reviewer_hash": "reviewer_1",
                "object_id": f"object_{object_index:03d}",
                "condition": "strong_simple_baseline",
                "action": "accept",
            }
        )
    responses = tmp_path / "expert.csv"
    _write_csv(responses, ["reviewer_hash", "object_id", "condition", "action"], rows)
    result = score_expert(responses)
    assert result["status"] == "supported"
    assert result["reviewer_count"] == 3
    assert result["single_expert_is_gold_standard"] is False


def test_domain_scorer_requires_independent_roles_and_closed_findings(tmp_path: Path) -> None:
    reviewers = tmp_path / "reviewers.json"
    reviewers.write_text(
        json.dumps(
            [
                {"role": "domain_specialist", "independent": True, "signed_record": "signed/domain_1.pdf"},
                {"role": "domain_specialist", "independent": True, "signed_record": "signed/domain_2.pdf"},
                {"role": "hci", "independent": True, "signed_record": "signed/hci.pdf"},
            ]
        ),
        encoding="utf-8",
    )
    findings = tmp_path / "findings.csv"
    _write_csv(findings, ["finding_id", "severity", "status"], [{"finding_id": "F1", "severity": "major", "status": "closed"}])
    dictionary = tmp_path / "dictionary.json"
    cards = tmp_path / "cards.json"
    dictionary.write_text("{}\n", encoding="utf-8")
    cards.write_text("[]\n", encoding="utf-8")
    result = score_domain(reviewers, findings, dictionary, cards)
    assert result["status"] == "pass"
    assert result["reviewer_count"] == 3
