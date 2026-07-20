#!/usr/bin/env python3
"""Run Q1 sensitivity, scalability and preregistered power calculations."""

from __future__ import annotations

import argparse
from pathlib import Path

from fuzzyxai.q1_validation.additional_protocols import run_power_analysis, run_scalability, run_sensitivity


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "release_evidence/q1_remediation"


def main(profile: str) -> None:
    run_power_analysis(EVIDENCE / "power/power_analysis.json")
    run_sensitivity(EVIDENCE / "sensitivity/sensitivity.json", n_objects=1_200 if profile == "smoke" else 10_000)
    run_scalability(EVIDENCE / "scalability/scalability.json", include_100k=profile == "full")
    print(f"PASS: q1_additional_protocols profile={profile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    args = parser.parse_args()
    main(args.profile)
