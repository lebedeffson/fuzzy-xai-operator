#!/usr/bin/env python3
"""Import a real blinded formative AI-review result; never synthesize scores."""

from __future__ import annotations

import argparse

from common import ROOT, STUDY, load_json, sha256, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="externally produced blinded run-2 summary JSON")
    arguments = parser.parse_args()
    source = (ROOT / arguments.input).resolve()
    payload = load_json(source)
    required = {
        "critical_unsupported_claims": 0,
        "critical_contradictions": 0,
        "critical_unjustified_actions": 0,
    }
    blockers = [name for name, expected in required.items() if payload.get(name) != expected]
    for metric in ("median_uncertainty_honesty", "median_clarity", "median_limitation_completeness"):
        if not isinstance(payload.get(metric), (int, float)) or float(payload[metric]) < 3.0:
            blockers.append(metric)
    if payload.get("blind") is not True or payload.get("ai_review_is_external_validation") is not False:
        blockers.append("methodological_boundary")
    if int(payload.get("case_count", 0)) < 240:
        blockers.append("case_count")
    if blockers:
        raise SystemExit(f"BLOCKED: AI formative run 2 acceptance failed: {', '.join(blockers)}")
    record = {
        **payload,
        "status": "pass",
        "source_path": source.relative_to(ROOT).as_posix(),
        "source_sha256": sha256(source),
        "ai_review_is_external_validation": False,
    }
    write_json(STUDY / "ai_formative_run2_acceptance.json", record)
    print(f"PASS: ai_formative_run2_import cases={record['case_count']} external_validation=false")


if __name__ == "__main__":
    main()

