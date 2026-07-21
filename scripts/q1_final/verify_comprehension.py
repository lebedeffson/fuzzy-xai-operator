#!/usr/bin/env python3
"""Verify that comprehension scoring is backed by genuine study records."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    result = ROOT / "release_evidence/q1_final/external/comprehension/scoring.json"
    raw = ROOT / "study/q1_final/comprehension/raw_anonymized"
    signed = ROOT / "study/q1_final/comprehension/signed_records"
    if not result.is_file() or not any(raw.glob("*")) or not any(signed.glob("*")):
        raise RuntimeError("comprehension gate requires scoring, raw anonymized records and signed records")
    payload = json.loads(result.read_text(encoding="utf-8"))
    if int(payload.get("valid_participants", 0)) < 24:
        raise RuntimeError("comprehension gate has fewer than 24 valid participants")
    if payload.get("participant_records_generated_by_scorer") is not False:
        raise RuntimeError("scorer must not generate participant records")
    print(f"PASS: comprehension_verified status={payload['status']}")


if __name__ == "__main__":
    main()
