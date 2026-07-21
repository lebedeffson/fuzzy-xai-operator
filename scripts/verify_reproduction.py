#!/usr/bin/env python3
"""Verify full evidence plus automatically generated dissertation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from verify_full_empirical_validation import verify


ROOT = Path(__file__).resolve().parents[1]


def run(profile: str) -> None:
    evidence = ROOT / "release_evidence/full_empirical_validation"
    verify(profile, evidence)
    required = (
        ROOT / "dissertation_artifacts/chapter3/table_3_uncertainty_hierarchy.csv",
        ROOT / "dissertation_artifacts/chapter4/table_4_multimodal_models.csv",
        ROOT / "dissertation_artifacts/chapter4/table_4_policy_comparison.csv",
        ROOT / "dissertation_artifacts/chapter3/fig_3_hierarchy_risk_complexity.png",
        ROOT / "dissertation_artifacts/chapter4/fig_4_risk_coverage.png",
        ROOT / "dissertation_artifacts/chapter4/fig_4_scalability_loglog.png",
        ROOT / "dissertation_artifacts/claims/chapter3_4_claims.json",
        ROOT / "reports/empirical_validation/full_empirical_validation.md",
    )
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"missing generated dissertation artifacts: {missing}")
    claims = json.loads((ROOT / "dissertation_artifacts/claims/chapter3_4_claims.json").read_text(encoding="utf-8"))
    if any(not claim.get("evidence_files") for claim in claims["claims"]):
        raise RuntimeError("a generated dissertation claim lacks evidence files")
    if not (ROOT / "requirements.lock").is_file():
        raise RuntimeError("requirements.lock is missing")
    print("PASS: generated_dissertation_tables")
    print("PASS: generated_dissertation_figures")
    print("PASS: generated_claim_provenance")
    print("PASS: reproduction_bundle_inputs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    args = parser.parse_args()
    run(args.profile)
