#!/usr/bin/env python3
"""Verify independent domain-language approval of the final hashes."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    result = ROOT / "release_evidence/q1_final/external/domain_language_review/scoring.json"
    raw = ROOT / "study/q1_final/domain_language_review/raw_anonymized"
    signed = ROOT / "study/q1_final/domain_language_review/signed_records"
    if not result.is_file() or not any(raw.glob("*")) or len(list(signed.glob("*"))) < 3:
        raise RuntimeError("domain gate requires scoring, raw findings and three signed records")
    payload = json.loads(result.read_text(encoding="utf-8"))
    if payload.get("status") != "pass" or int(payload.get("open_critical_or_major", 1)):
        raise RuntimeError("domain-language review has unresolved blocking findings")
    if not payload.get("final_dictionary_sha256") or not payload.get("final_cards_sha256"):
        raise RuntimeError("domain gate must approve final dictionary and card hashes")
    print("PASS: domain_review_verified")


if __name__ == "__main__":
    main()
