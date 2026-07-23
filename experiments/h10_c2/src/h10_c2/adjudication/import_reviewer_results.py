from __future__ import annotations

import csv
from pathlib import Path

from ..hashing import object_sha256, write_json
from ..paths import ARTIFACT_ROOT
from .export_blind_cases import FIELDS


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(not all(row.get(field, "").strip() for field in FIELDS) for row in rows):
        raise ValueError("review form is incomplete")
    return rows


def import_results(reviewer_1: Path, reviewer_2: Path) -> dict:
    first, second = _read(reviewer_1), _read(reviewer_2)
    signatures = {row["reviewer_signature"] for row in first}, {row["reviewer_signature"] for row in second}
    if len(signatures[0]) != 1 or len(signatures[1]) != 1 or signatures[0] == signatures[1]:
        raise ValueError("two distinct reviewer signatures are required")
    by_first = {row["case_id"]: row for row in first}
    by_second = {row["case_id"]: row for row in second}
    if by_first.keys() != by_second.keys():
        raise ValueError("reviewers evaluated different cases")
    compared = ("mutation_log_consistent", "optimal_cuts_valid", "repair_actions_valid", "ambiguous", "sufficient_evidence")
    disagreements = [
        case_id
        for case_id in by_first
        if any(by_first[case_id][field] != by_second[case_id][field] for field in compared)
    ]
    agreement = 1.0 - len(disagreements) / len(by_first)
    report = {
        "status": "PASS" if not disagreements else "BLOCKED_HUMAN_ADJUDICATION",
        "case_count": len(by_first),
        "agreement": agreement,
        "unresolved_disagreements": disagreements,
        "reviewer_signatures_sha256": object_sha256(sorted((next(iter(signatures[0])), next(iter(signatures[1]))))),
    }
    root = ARTIFACT_ROOT / "adjudication"
    write_json(root / "agreement.json", report)
    write_json(root / "status.json", report)
    return report

