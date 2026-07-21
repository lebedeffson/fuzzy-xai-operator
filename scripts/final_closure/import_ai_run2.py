#!/usr/bin/env python3
from __future__ import annotations

import argparse

from common import ROOT, STUDY, load, sha256, write


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    args = parser.parse_args()
    source = (ROOT / args.input).resolve()
    payload = load(source)
    blockers = []
    for key in ("critical_unsupported_claims", "critical_contradictions", "critical_unjustified_actions"):
        if payload.get(key) != 0:
            blockers.append(key)
    for key in ("median_uncertainty_honesty", "median_clarity", "median_limitation_completeness"):
        if float(payload.get(key, 0)) < 3:
            blockers.append(key)
    if payload.get("case_count") != 240 or payload.get("blind") is not True:
        blockers.append("blind_240_case_contract")
    if payload.get("ai_review_is_external_validation") is not False:
        blockers.append("AI_review_boundary")
    if blockers:
        raise SystemExit(f"BLOCKED: AI run 2 failed acceptance: {blockers}")
    write(STUDY / "ai_formative_run2_acceptance.json", {**payload, "status": "pass", "source_sha256": sha256(source)})
    print("PASS: final_ai_run2_import cases=240 external_validation=false")


if __name__ == "__main__":
    main()
