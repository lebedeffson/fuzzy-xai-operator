from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from fuzzyxai.ai_pre_review.contracts import StudyBoundaryError, contains_method_identity, read_jsonl, validate_score_map
from fuzzyxai.ai_pre_review.generator import _action
from fuzzyxai.ai_pre_review.review_io import validate_human_review_directory


ROOT = Path(__file__).resolve().parents[2]


def test_blind_master_log_has_required_case_boundary() -> None:
    rows = read_jsonl(ROOT / "study/ai_pre_review/master_explanation_log.jsonl")
    assert len(rows) == 1080
    assert len({row["object_id_hash"] for row in rows}) == 360
    assert Counter(row["split"] for row in rows) == {"formative": 720, "confirmatory": 360}
    assert all(not contains_method_identity({key: value for key, value in row.items() if key != "method_identity_encrypted"}) for row in rows)


def test_all_actions_exist_in_every_modality() -> None:
    rows = read_jsonl(ROOT / "study/ai_pre_review/master_explanation_log.jsonl")
    for modality in ("tabular", "image", "text", "timeseries"):
        actions = {row["action"] for row in rows if row["modality"] == modality}
        assert actions == {"accept", "short_review", "full_review", "block"}


def test_action_contract_distinguishes_baseline_and_route() -> None:
    assert _action("strong_local_baseline", 0.82, True, True, True, True, True) == "accept"
    assert _action("selective_risk_observer", 0.82, True, False, True, False, False) == "block"
    assert _action("full_operator_route", 0.82, False, False, False, True, False) == "block"


def test_score_contract_rejects_out_of_range() -> None:
    scores = {name: 3 for name in json.loads((ROOT / "study/ai_pre_review/ai_review_schema.json").read_text())["properties"]["scores"]["required"]}
    scores["clarity"] = 5
    with pytest.raises(StudyBoundaryError):
        validate_score_map(scores)


def test_human_import_fails_before_ai_commitment(tmp_path: Path) -> None:
    with pytest.raises(StudyBoundaryError, match="protocol lock and AI score commitment"):
        validate_human_review_directory(ROOT, tmp_path)


def test_claim_registry_does_not_claim_external_validation() -> None:
    registry = json.loads((ROOT / "study/ai_pre_review/claim_registry.json").read_text(encoding="utf-8"))
    assert registry["stable_release_allowed"] is False
    assert all(row["status"] not in {"supported", "human_confirmed", "expert_validated"} for row in registry["claims"])
