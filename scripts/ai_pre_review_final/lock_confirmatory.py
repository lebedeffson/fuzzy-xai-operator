#!/usr/bin/env python3
"""Fail-closed confirmatory lock created only from real formative acceptance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fuzzyxai.ai_pre_review_final.contracts import FinalStudyError, read_jsonl, sha256_file


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--acceptance",
        type=Path,
        default=ROOT / "study/ai_pre_review_final/formative_acceptance.json",
    )
    args = parser.parse_args()
    if not args.acceptance.is_file():
        raise FinalStudyError("confirmatory lock blocked: real formative acceptance is missing")
    acceptance = json.loads(args.acceptance.read_text(encoding="utf-8"))
    _validate_acceptance(acceptance)
    full_records = ROOT / "study/ai_pre_review_final/reviewer_cases.jsonl"
    if not full_records.is_file():
        raise FinalStudyError("confirmatory lock blocked: private full reviewer pool is unavailable")
    confirmatory_ids = {
        row["case_id"]
        for row in read_jsonl(ROOT / "study/ai_pre_review/source_case_evidence.jsonl")
        if row["split"] == "confirmatory"
    }
    confirmatory = [row for row in read_jsonl(full_records) if row["case_id"] in confirmatory_ids]
    if len(confirmatory_ids) != 120 or len(confirmatory) != 360:
        raise FinalStudyError("confirmatory lock blocked: expected 120 cases and 360 variants")
    public_manifest = json.loads(
        (ROOT / "study/ai_pre_review_final/public_formative/manifest.json").read_text(encoding="utf-8")
    )
    if acceptance["reviewer_cases_sha256"] != public_manifest["reviewer_cases_sha256"]:
        raise FinalStudyError("formative acceptance refers to a different reviewer packet")
    payload = {
        "schema_version": "1.0",
        "state": "locked_after_real_formative_acceptance",
        "formative_acceptance_sha256": sha256_file(args.acceptance),
        "formative_reviewer_cases_sha256": public_manifest["reviewer_cases_sha256"],
        "confirmatory_case_count": 120,
        "confirmatory_variant_count": 360,
        "confirmatory_records_sha256": _rows_sha256(confirmatory),
        "rubric_sha256": sha256_file(ROOT / "study/ai_pre_review/rubric_v1.yaml"),
        "response_schema_sha256": sha256_file(ROOT / "study/ai_pre_review/ai_review_schema.json"),
        "reviewer_schema_sha256": sha256_file(ROOT / "study/ai_pre_review_final/reviewer_case_schema_v2.json"),
        "changes_after_lock_forbidden": True,
        "human_pack_allowed": False,
        "stable_release_allowed": False,
    }
    output = ROOT / "study/ai_pre_review_final/confirmatory_protocol_lock.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PASS: confirmatory_protocol_lock path={output.relative_to(ROOT)}")


def _validate_acceptance(value: object) -> None:
    if not isinstance(value, dict):
        raise FinalStudyError("formative acceptance must be a JSON object")
    required = {
        "schema_version",
        "stage",
        "status",
        "reviewer_cases_sha256",
        "independent_run_hashes",
        "critical_flags",
        "median_scores",
        "authorized_by",
        "authorized_at",
    }
    if not required.issubset(value):
        raise FinalStudyError("formative acceptance is incomplete")
    if value["stage"] != "formative" or value["status"] != "accepted_after_real_review":
        raise FinalStudyError("formative acceptance has not passed real review")
    if len(value["independent_run_hashes"]) < 2:
        raise FinalStudyError("at least two real formative review runs are required")
    if any(int(count) != 0 for count in value["critical_flags"].values()):
        raise FinalStudyError("critical formative defects remain open")
    required_scores = {"factual_consistency", "clarity", "uncertainty_honesty", "usability"}
    if not required_scores.issubset(value["median_scores"]):
        raise FinalStudyError("formative median scores are incomplete")
    if any(float(value["median_scores"][name]) < 3.0 for name in required_scores):
        raise FinalStudyError("formative acceptance thresholds are not met")


def _rows_sha256(rows: list[dict[str, object]]) -> str:
    import hashlib

    body = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    return hashlib.sha256(body.encode()).hexdigest()


if __name__ == "__main__":
    try:
        main()
    except FinalStudyError as error:
        print(f"BLOCKED: {error}")
        raise SystemExit(2) from None
