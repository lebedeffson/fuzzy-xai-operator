#!/usr/bin/env python3
"""Verify final technical evidence and optionally require the heavy profile."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "release_evidence/q1_final"


def load(relative: str) -> dict[str, object]:
    path = EVIDENCE / relative
    if not path.is_file():
        raise RuntimeError(f"missing final evidence: {relative}")
    return json.loads(path.read_text(encoding="utf-8"))


def main(require_heavy: bool) -> None:
    identity = load("run_identity.json")
    if not re.fullmatch(r"[0-9a-f]{40}", str(identity["final_commit"])):
        raise RuntimeError("run identity lacks a complete final commit")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if identity["final_commit"] != head:
        raise RuntimeError("run identity differs from HEAD")
    external = load("external/status.json")
    if external.get("human_records_generated_by_code") is not False:
        raise RuntimeError("external status must prove that code generated no human records")
    claims = load("claim_registry.json")
    predictive = next(row for row in claims["claims"] if row["claim_id"] == "H5-predictive")
    if predictive["status"] == "supported" and float(predictive["metrics"]["incremental_auprc"]) <= 0:
        raise RuntimeError("predictive rupture claim is unsupported by its metric")
    gate = load("final_gate_matrix.json")
    if any(value == "open" for value in external["gates"].values()) and gate["stable_release_allowed"]:
        raise RuntimeError("stable release cannot pass with open external gates")
    dod = load("dod_185.json")
    if len(dod["items"]) != 185:
        raise RuntimeError("final DoD must contain 185 items")
    if require_heavy:
        real = load("real_benchmarks/combined_status.json")
        if real["status"] != "PASS":
            raise RuntimeError(f"heavy real benchmark status is {real['status']}")
        load("hypotheses/final_results.json")
        load("rule_ablation/final_claim_status.json")
        scaling = load("scalability/end_to_end.json")
        sizes = {row["n_objects"] for row in scaling["measurements"]}
        if not {1000, 5000, 10000, 50000, 100000}.issubset(sizes):
            raise RuntimeError("heavy scalability sizes are incomplete")
    print(f"PASS: q1_final_verification heavy={require_heavy} stable={gate['stable_release_allowed']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-heavy", action="store_true")
    args = parser.parse_args()
    main(args.require_heavy)
