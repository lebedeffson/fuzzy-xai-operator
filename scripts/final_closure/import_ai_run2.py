#!/usr/bin/env python3
"""Validate all raw clean-session run-2 reviews and calculate acceptance."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from statistics import median

from fuzzyxai.ai_pre_review.contracts import (
    METHOD_NAMES,
    SCORE_KEYS,
    StudyBoundaryError,
    canonical_json,
    validate_flag_rows,
    validate_score_map,
)

from common import ROOT, STUDY, load, sha256, write


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    args = parser.parse_args()
    source = (ROOT / args.input).resolve()
    reviews_path, metadata_path = _input_paths(source)
    rows = _read_rows(reviews_path)
    metadata = load(metadata_path)
    protocol = load(STUDY / "ai_formative_run2/protocol.json")
    _validate_metadata(metadata)
    expected = _expected_pairs()
    batches = {row["batch_id"]: row for row in protocol["batches"]}
    observed: set[tuple[str, str]] = set()
    for row in rows:
        _validate_row(row, batches, protocol)
        key = (str(row["case_id"]), str(row["variant_id"]))
        if key in observed:
            raise StudyBoundaryError(f"duplicate AI run-2 review: {key}")
        observed.add(key)
    if observed != expected or len(rows) != 720:
        raise StudyBoundaryError(f"run-2 coverage mismatch: expected=720 observed={len(observed)}")
    critical = _critical_counts(rows)
    medians = {name: median(row["scores"][name] for row in rows) for name in SCORE_KEYS}
    blockers = [name for name, value in critical.items() if value]
    for name in ("uncertainty_honesty", "clarity", "limitation_completeness"):
        if medians[name] < 3:
            blockers.append(f"median_{name}")
    status = "pass" if not blockers else "fail"
    payload = {
        "status": status,
        "case_count": 240,
        "variant_count": 720,
        "blind": True,
        "ai_review_is_external_validation": False,
        "session_metadata": metadata,
        **critical,
        **{f"median_{name}": value for name, value in medians.items()},
        "blockers": blockers,
        "source_sha256": sha256(reviews_path),
    }
    evidence_dir = STUDY / "ai_formative_run2/results"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    frozen = evidence_dir / f"reviews-{sha256(reviews_path)}.jsonl"
    if frozen.exists():
        if sha256(frozen) != sha256(reviews_path):
            raise StudyBoundaryError("frozen run-2 review hash mismatch")
    else:
        shutil.copy2(reviews_path, frozen)
    write(STUDY / "ai_formative_run2_acceptance.json", payload)
    _write_report(payload)
    if blockers:
        raise SystemExit(f"BLOCKED: AI run 2 failed acceptance: {blockers}")
    print("PASS: final_ai_run2_import cases=240 variants=720 external_validation=false")


def _input_paths(source: Path) -> tuple[Path, Path]:
    if source.is_file():
        reviews = source
        metadata = source.with_name("session_metadata.json")
    else:
        reviews = source / "reviews.jsonl"
        metadata = source / "session_metadata.json"
    if not reviews.is_file() or not metadata.is_file():
        raise StudyBoundaryError("AI_RUN2_INPUT requires reviews.jsonl and session_metadata.json")
    return reviews, metadata


def _read_rows(path: Path) -> list[dict[str, object]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise StudyBoundaryError(f"invalid JSONL at line {line_number}: {error}") from error
        if not isinstance(row, dict):
            raise StudyBoundaryError(f"review row {line_number} is not an object")
        rows.append(row)
    return rows


def _validate_metadata(metadata: dict[str, object]) -> None:
    required = {
        "session_type": "temporary_clean_chat",
        "prior_context": False,
        "project_context": False,
        "memory_used": False,
        "review_status": "fully_blind",
        "run": "formative_run_2",
        "cases": 240,
        "variants": 720,
    }
    mismatches = [key for key, value in required.items() if metadata.get(key) != value]
    if mismatches or not str(metadata.get("reviewer_model_label", "")).strip():
        raise StudyBoundaryError(f"invalid clean-session metadata: {mismatches}")


def _expected_pairs() -> set[tuple[str, str]]:
    path = STUDY / "ai_formative_run2/reviewer_cases.jsonl"
    return {
        (str(row["case_id"]), str(row["variant_id"]))
        for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    }


def _validate_row(row: dict[str, object], batches: dict[str, dict[str, object]], protocol: dict[str, object]) -> None:
    if row.get("review_schema_version") != "1.0" or row.get("reviewer_type") != "ai_pre_reviewer":
        raise StudyBoundaryError("invalid run-2 reviewer schema or type")
    batch = batches.get(str(row.get("batch_id", "")))
    if batch is None or row.get("input_batch_sha256") != batch["sha256"]:
        raise StudyBoundaryError("unknown or modified run-2 batch")
    if row.get("ai_run_id") != "AI_RUN_2" or row.get("ai_review_commit") != protocol["source_commit"]:
        raise StudyBoundaryError("run-2 ID or source commit mismatch")
    validate_score_map(row.get("scores"))
    validate_flag_rows(row.get("critical_flags"))
    if not 1 <= int(row.get("preferred_variant_rank", 0)) <= 3:
        raise StudyBoundaryError("preferred variant rank is invalid")
    confidence = row.get("confidence_in_review")
    if not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
        raise StudyBoundaryError("confidence_in_review must be in [0, 1]")
    if len(str(row.get("summary", ""))) > 600:
        raise StudyBoundaryError("review summary exceeds 600 characters")
    if len(row.get("required_changes", [])) > 8 or len(row.get("optional_changes", [])) > 5:
        raise StudyBoundaryError("review change list exceeds rubric limits")
    if any(name in canonical_json(row).lower() for name in METHOD_NAMES):
        raise StudyBoundaryError("method identity leaked in run-2 response")


def _critical_counts(rows: list[dict[str, object]]) -> dict[str, int]:
    mapping = {
        "critical_unsupported_claims": "unsupported_claim",
        "critical_contradictions": "contradicts_evidence",
        "critical_unjustified_actions": "unsafe_or_unjustified_action",
    }
    return {
        output: sum(
            flag["flag"] == expected and flag["severity"] == "critical"
            for row in rows
            for flag in row["critical_flags"]
        )
        for output, expected in mapping.items()
    }


def _write_report(payload: dict[str, object]) -> None:
    lines = [
        "# AI formative run 2",
        "",
        f"Status: {payload['status']}",
        "",
        "This is automated formative text review, not external validation.",
        "",
        f"Cases: {payload['case_count']}; variants: {payload['variant_count']}.",
        "",
        f"Critical unsupported claims: {payload['critical_unsupported_claims']}",
        f"Critical contradictions: {payload['critical_contradictions']}",
        f"Critical unjustified actions: {payload['critical_unjustified_actions']}",
        f"Median uncertainty honesty: {payload['median_uncertainty_honesty']}",
        f"Median clarity: {payload['median_clarity']}",
        f"Median limitation completeness: {payload['median_limitation_completeness']}",
    ]
    (STUDY / "ai_formative_run2_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
