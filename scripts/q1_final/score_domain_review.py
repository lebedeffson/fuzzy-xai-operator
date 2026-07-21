#!/usr/bin/env python3
"""Validate independent domain-language findings and final approval records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def score(
    reviewers_path: Path,
    findings_path: Path,
    dictionary_path: Path,
    cards_path: Path,
) -> dict[str, object]:
    reviewers = json.loads(reviewers_path.read_text(encoding="utf-8"))
    independent = [row for row in reviewers if row.get("independent") and row.get("signed_record")]
    roles = {row.get("role") for row in independent}
    if len(independent) < 3 or "domain_specialist" not in roles or not roles.intersection({"hci", "scientific_communication"}):
        raise ValueError("domain gate requires two domain specialists and one HCI or communication reviewer")
    if sum(row.get("role") == "domain_specialist" for row in independent) < 2:
        raise ValueError("domain gate requires two independent domain specialists")
    with findings_path.open(newline="", encoding="utf-8") as handle:
        findings = list(csv.DictReader(handle))
    open_blocking = [row for row in findings if row["severity"] in {"critical", "major"} and row["status"] != "closed"]
    return {
        "schema_version": "1.0",
        "status": "pass" if not open_blocking else "fail",
        "reviewer_count": len(independent),
        "finding_count": len(findings),
        "open_critical_or_major": len(open_blocking),
        "final_dictionary_sha256": hashlib.sha256(dictionary_path.read_bytes()).hexdigest(),
        "final_cards_sha256": hashlib.sha256(cards_path.read_bytes()).hexdigest(),
        "reviewers_sha256": hashlib.sha256(reviewers_path.read_bytes()).hexdigest(),
        "findings_sha256": hashlib.sha256(findings_path.read_bytes()).hexdigest(),
        "scorer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "records_generated_by_scorer": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviewers", type=Path, required=True)
    parser.add_argument("--findings", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--cards", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = score(args.reviewers, args.findings, args.dictionary, args.cards)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if payload["status"] != "pass":
        raise RuntimeError("domain-language review has open critical or major findings")
    print(f"PASS: domain_review_scoring reviewers={payload['reviewer_count']}")


if __name__ == "__main__":
    main()
