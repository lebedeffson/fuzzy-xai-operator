from __future__ import annotations

import copy
import json
import subprocess
import zipfile
from pathlib import Path

import pytest

from fuzzyxai.ai_pre_review_final import audit_blind_records
from fuzzyxai.ai_pre_review_final.contracts import FinalStudyError, read_jsonl, sha256_file


ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "study/ai_pre_review_final/public_formative"


def _rows() -> list[dict[str, object]]:
    return read_jsonl(PUBLIC / "reviewer_cases.jsonl")


def test_public_formative_packet_passes_full_blinding_audit() -> None:
    report = audit_blind_records(
        _rows(),
        root=PUBLIC,
        expected_cases=240,
        expected_records=720,
    )
    assert report["status"] == "PASS"
    assert report["claim_evidence_coverage_min"] == 1.0
    assert report["outcome_leakage"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("is_correct", True),
        ("true_label", 1),
        ("stratum", "wrong_prediction"),
        ("expected_action", "block"),
        ("method_identity", "full_operator_explanation"),
    ],
)
def test_outcome_action_and_method_leakage_fail_closed(field: str, value: object) -> None:
    rows = _rows()
    rows[0] = copy.deepcopy(rows[0])
    rows[0][field] = value
    with pytest.raises(FinalStudyError, match="blindness audit failed"):
        audit_blind_records(rows, root=PUBLIC, expected_cases=240, expected_records=720)


def test_all_modalities_have_human_readable_measured_evidence() -> None:
    required = {
        "tabular": {"observed_value_anonymized", "reference_percentile"},
        "image": {"bounding_box", "region_id"},
        "text": {"phrase", "character_position"},
        "timeseries": {"interval_start", "interval_end", "signal_channel"},
    }
    counts = {name: 0 for name in required}
    for row in _rows():
        modality = str(row["modality"])
        counts[modality] += 1
        for item in row["interpretable_evidence"]:
            assert required[modality] <= set(item)
            assert item["display_name"]
            assert item["direction"] in {"supports", "opposes", "neutral"}
            assert 0.0 <= float(item["magnitude_normalized"]) <= 1.0
            assert item["evidence_refs"]
        if modality == "image":
            assert (PUBLIC / row["observable_asset"]["thumbnail_ref"]).is_file()
    assert counts == {name: 180 for name in required}


def test_variants_are_semantically_distinct_without_revealing_identity() -> None:
    rows = _rows()
    by_case: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_case.setdefault(str(row["case_id"]), []).append(row)
    for variants in by_case.values():
        signatures = {
            (
                len(row["interpretable_evidence"]),
                len(row["candidate_explanation"]["provenance_summary"]),
                row["presentation"]["detail"],
                bool(row["candidate_explanation"]["counterfactuals"]),
            )
            for row in variants
        }
        assert signatures == {(3, 0, "short", False), (4, 2, "short", True), (5, 3, "full", True)}
        assert all("semantic_blocks" not in row for row in variants)


def test_public_archive_excludes_private_and_confirmatory_material() -> None:
    commit = subprocess.check_output(["git", "rev-parse", "--short=12", "HEAD"], cwd=ROOT, text=True).strip()
    archive = ROOT / "release_artifacts/ai_pre_review_final" / f"fuzzyxai-ai-pre-review-final-bundles-{commit}.zip"
    assert archive.is_file(), "run the public bundle builder for the current commit before this test"
    with zipfile.ZipFile(archive) as handle:
        names = handle.namelist()
        assert not any("hidden_scoring_key" in name or "/private/" in name for name in names)
        assert not any("/confirmatory/" in name for name in names)
        assert not any("BLINDING_AUDIT" in name or "claim_registry" in name for name in names)
        root_name = names[0].split("/", 1)[0]
        manifest = json.loads(handle.read(f"{root_name}/manifest.json"))
        rows = [json.loads(line) for line in handle.read(f"{root_name}/reviewer_cases.jsonl").decode().splitlines()]
    assert manifest["hidden_scoring_key_included"] is False
    assert manifest["confirmatory_material_included"] is False
    assert len(rows) == 720
    assert len({row["case_id"] for row in rows}) == 240
    checksum = archive.with_suffix(".zip.sha256")
    assert checksum.is_file()
    assert checksum.read_text().split()[0] == sha256_file(archive)


def test_confirmatory_lock_is_absent_before_real_formative_acceptance() -> None:
    assert not (ROOT / "study/ai_pre_review_final/confirmatory_protocol_lock.json").exists()
    assert not (ROOT / "study/ai_pre_review_final/formative_acceptance.json").exists()
