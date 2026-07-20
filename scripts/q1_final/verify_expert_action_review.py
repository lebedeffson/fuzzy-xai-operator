#!/usr/bin/env python3
"""Verify that expert-action scoring is backed by independent records."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    result = ROOT / "release_evidence/q1_final/external/expert_action_review/scoring.json"
    raw = ROOT / "study/q1_final/expert_action_review/raw_anonymized"
    signed = ROOT / "study/q1_final/expert_action_review/signed_records"
    if not result.is_file() or not any(raw.glob("*")) or len(list(signed.glob("*"))) < 3:
        raise RuntimeError("expert-action gate requires scoring, raw records and three signed reviewer records")
    payload = json.loads(result.read_text(encoding="utf-8"))
    if int(payload.get("reviewer_count", 0)) < 3 or int(payload.get("object_count", 0)) < 100:
        raise RuntimeError("expert-action gate lacks the preregistered panel")
    if payload.get("records_generated_by_scorer") is not False:
        raise RuntimeError("scorer must not generate expert records")
    print(f"PASS: expert_action_verified status={payload['status']}")


if __name__ == "__main__":
    main()
