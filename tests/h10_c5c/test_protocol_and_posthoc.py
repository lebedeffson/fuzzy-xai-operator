from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
from fuzzyxai.experiments.h10_c5c import validate_development_manifest

from scripts.ch4_revision.analyze_h10_c5b_errors import analyze


def test_h10_c5b_postopen_lock_is_immutable() -> None:
    lock = json.loads(Path("protocol/h10_c5b_repository_grounded/H10_C5B_POSTOPEN_LOCK.json").read_text(encoding="utf-8"))
    for path, expected in lock["immutable_files"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    assert lock["opening_count"] == 1
    assert lock["status"] == "FINAL_NEGATIVE_RESULT_LOCKED"


def test_posthoc_analysis_reports_shared_candidate_boundary() -> None:
    result = analyze(Path("results/h10_c5b/PER_INCIDENT_RESULTS.csv"))
    assert result["o_route_abstained"] == 19
    assert result["o_route_diagnosed"] == 5
    assert result["o_route_false_localizations_among_diagnosed"] == 5
    assert result["identical_top_k_incidents"] == 24
    assert result["limitation_counts"] == {
        "contract_family_not_supported_by_evidence": 12,
        "no_structural_candidate": 6,
    }


def test_committed_error_table_has_all_five_classes() -> None:
    with Path("reports/h10_c5c/H10_C5B_INCIDENT_ERROR_CLASSIFICATION.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 24
    assert {row["error_class"] for row in rows} == {
        "RETRIEVAL_MISS",
        "CONTRACT_INFERENCE_MISS",
        "GRAPH_CONSTRUCTION_MISS",
        "INSUFFICIENT_RUNTIME_EVIDENCE",
        "REPAIR_NOT_EXPRESSIBLE",
    }
    assert {row["retrieved_top10"] for row in rows} == {"NOT_ESTIMABLE_FROZEN_TOP3_ONLY"}


def test_development_runner_rejects_insufficient_matrix(tmp_path: Path) -> None:
    manifest = tmp_path / "development.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "incident_id": "one",
                "repository": "new/repository",
                "buggy_revision": "buggy",
                "repository_root": str(tmp_path),
                "failing_tests": [],
                "split": "development",
                "patch_path": str(tmp_path / "fix.patch"),
                "before_sources_path": str(tmp_path / "before.json"),
                "after_sources_path": str(tmp_path / "after.json"),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="at least 30"):
        validate_development_manifest(manifest, Path.cwd())


def test_development_runner_requires_runtime_trace(tmp_path: Path) -> None:
    manifest = tmp_path / "development.jsonl"
    rows = [
        {
            "incident_id": f"incident-{index}",
            "repository": f"new/repository-{index % 8}",
            "buggy_revision": "buggy",
            "repository_root": str(tmp_path),
            "failing_tests": [],
            "split": "development",
            "patch_path": str(tmp_path / "fix.patch"),
            "before_sources_path": str(tmp_path / "before.json"),
            "after_sources_path": str(tmp_path / "after.json"),
            "runtime_evidence_status": "FAILING_TEST_IDS_ONLY",
        }
        for index in range(30)
    ]
    manifest.write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="BUG_REPRODUCED_WITH_TRACE"):
        validate_development_manifest(manifest, Path.cwd())
