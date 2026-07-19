"""Validate external-gate artifacts and optionally require their real completion."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "release_evidence/user_study/comprehension_pilot"
DOMAIN = ROOT / "release_evidence/domain_language_review"


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_manifest(directory: Path) -> None:
    manifest = read_json(directory / "manifest_sha256.json")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError(f"empty manifest: {directory}")
    for relative, expected in files.items():
        path = directory / str(relative)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"manifest mismatch: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    verify_manifest(PILOT)
    verify_manifest(DOMAIN)

    pilot = read_json(PILOT / "scoring_report.json")
    domain = read_json(DOMAIN / "review_record.json")
    source = ROOT / str(domain["domain_language_artifact"])
    if not source.is_file() or sha256(source) != domain["domain_language_sha256"]:
        raise ValueError("domain-language review is not bound to the current dictionary")
    with (PILOT / "anonymized_responses.csv").open(encoding="utf-8", newline="") as handle:
        response_count = sum(1 for _ in csv.DictReader(handle))
    if pilot.get("status") == "pass" and response_count == 0:
        raise ValueError("pilot cannot pass without archived anonymized responses")
    domain_pass = (
        domain.get("status") in {"approved", "approved_with_comments"}
        and domain.get("independent_of_project") is True
        and bool(domain.get("reviewer_id"))
        and bool(domain.get("reviewer_role"))
        and domain.get("claim_allowed") is True
    )
    pilot_pass = pilot.get("status") == "pass" and pilot.get("claim_allowed") is True
    print("PASS: external_gate_artifact_integrity")
    print(f"{'PASS' if pilot_pass else 'BLOCKED'}: comprehension_pilot {pilot.get('status')}")
    print(f"{'PASS' if domain_pass else 'BLOCKED'}: domain_language {domain.get('status')}")
    if args.require_pass and not (pilot_pass and domain_pass):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
