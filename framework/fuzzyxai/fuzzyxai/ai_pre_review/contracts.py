"""Shared contracts and fail-closed validation for the pre-review study."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping


SCORE_KEYS = (
    "factual_consistency",
    "traceability",
    "uncertainty_honesty",
    "action_quality",
    "clarity",
    "limitation_completeness",
    "overtrust_control",
    "causal_correctness",
    "audience_fit",
    "overall_usability",
)
CRITICAL_FLAGS = (
    "unsupported_claim",
    "contradicts_evidence",
    "missing_critical_limitation",
    "unsafe_or_unjustified_action",
    "causal_overclaim",
    "method_identity_leak",
    "unexplained_technical_term",
    "confidence_overstatement",
    "clinical_or_domain_overclaim",
    "counterfactual_presented_as_recommendation",
    "provenance_broken",
    "wrong_audience_level",
)
AI_RUN_IDS = ("AI_RUN_1", "AI_RUN_2", "AI_RUN_3")
METHOD_NAMES = ("fuzzyxai", "selective observer", "full fuzzyxai")
CASE_RE = re.compile(r"^case_[0-9]{6}$")
VARIANT_RE = re.compile(r"^X[123]$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class StudyBoundaryError(RuntimeError):
    """Raised when a blind-study or external-validation boundary is crossed."""


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise StudyBoundaryError(f"{path}:{line_number}: JSONL row is not an object")
            rows.append(value)
    except json.JSONDecodeError as exc:
        raise StudyBoundaryError(f"{path}: invalid JSON: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canonical_json(dict(row)) + "\n" for row in rows), encoding="utf-8")


def contains_method_identity(value: object) -> bool:
    text = canonical_json(value).lower()
    return any(name in text for name in METHOD_NAMES)


def validate_score_map(scores: object) -> None:
    if not isinstance(scores, dict) or set(scores) != set(SCORE_KEYS):
        raise StudyBoundaryError("review scores must contain exactly R1-R10 fields")
    if any(type(value) is not int or not 0 <= value <= 4 for value in scores.values()):
        raise StudyBoundaryError("review scores must be integer values from 0 to 4")


def validate_flag_rows(flags: object) -> None:
    if not isinstance(flags, list):
        raise StudyBoundaryError("critical_flags must be a list")
    for row in flags:
        if not isinstance(row, dict) or row.get("flag") not in CRITICAL_FLAGS:
            raise StudyBoundaryError("unknown critical flag")
        if row.get("present") is not True:
            raise StudyBoundaryError("critical_flags must contain present flags only")
        if row.get("severity") not in {"critical", "major", "minor"}:
            raise StudyBoundaryError("critical flag severity is invalid")
        if not str(row.get("evidence", "")).strip() or not str(row.get("recommended_fix", "")).strip():
            raise StudyBoundaryError("critical flag requires evidence and recommended_fix")
