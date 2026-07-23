from __future__ import annotations

import csv
from pathlib import Path

import pytest

from h10_c2.adjudication.export_blind_cases import FIELDS, export_blind_cases
from h10_c2.adjudication.import_reviewer_results import import_results
from h10_c2.audit import run_leakage_audit
from h10_c2.cli import bootstrap
from h10_c2.paths import ARTIFACT_ROOT
from h10_c2.runner import build_nonconfirmatory_statistics, generate_from_design, run_split
from h10_c2.sealing.scoring_gate import preconfirmatory_gate
from h10_c2.statistics import run_power


def test_power_generation_and_blind_forms_are_empty() -> None:
    bootstrap()
    design = run_power()
    assert design["requires_human_approval"] is True
    generate_from_design("protocol_validation", seed_offset=2)
    manifest = export_blind_cases(12)
    assert manifest["same_cases"] and manifest["different_order"]
    with (ARTIFACT_ROOT / "adjudication" / "blind" / "reviewer_1" / "form.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows and all(not row["optimal_cuts_valid"] for row in rows)


def test_fake_single_reviewer_or_incomplete_forms_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "review.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow({"case_id": "case"})
    with pytest.raises(ValueError, match="incomplete"):
        import_results(path, path)


def test_protocol_validation_run_is_nonconfirmatory() -> None:
    run_split("protocol_validation")
    statistics = build_nonconfirmatory_statistics("protocol_validation")
    assert all(item["confirmatory"] is False for item in statistics)
    assert run_leakage_audit()["old_new_case_hash_intersection"] == 0
    gate = preconfirmatory_gate()
    assert gate["status"] in {"BLOCKED_POWER", "BLOCKED_PROTOCOL"}
    assert gate["sealed_opening_count"] == 0


def test_sealed_generation_is_blocked_when_power_is_insufficient() -> None:
    with pytest.raises(PermissionError, match="BLOCKED_POWER"):
        generate_from_design("sealed", seed_offset=3)
