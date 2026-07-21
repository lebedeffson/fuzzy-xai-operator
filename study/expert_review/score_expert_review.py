#!/usr/bin/env python3
"""Score real external-review responses; refuse empty or single-reviewer data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REQUIRED = {
    "reviewer_id",
    "review_item_id",
    "recommended_action",
    "explanation_sufficient",
    "main_reason_understood",
    "concern_understood",
    "confidence_1_to_5",
}


def score(source: Path, output: Path) -> dict[str, object]:
    with source.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not REQUIRED.issubset(reader.fieldnames or ()):
            raise RuntimeError(f"missing response columns: {sorted(REQUIRED - set(reader.fieldnames or ()))}")
        rows = list(reader)
    reviewers = {row["reviewer_id"] for row in rows if row["reviewer_id"]}
    if len(reviewers) < 2 or len(rows) < 100:
        raise RuntimeError("independent review requires at least two reviewers and 100 completed rows")
    def yes_rate(field: str) -> float:
        return sum(row[field].strip().lower() == "yes" for row in rows) / len(rows)

    payload = {
        "status": "completed",
        "reviewers": len(reviewers),
        "responses": len(rows),
        "explanation_sufficient_rate": yes_rate("explanation_sufficient"),
        "main_reason_understood_rate": yes_rate("main_reason_understood"),
        "concern_understood_rate": yes_rate("concern_understood"),
        "source_file": str(source),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("responses", type=Path)
    parser.add_argument("--output", type=Path, default=Path("study/expert_review/scoring_report.json"))
    arguments = parser.parse_args()
    score(arguments.responses, arguments.output)
