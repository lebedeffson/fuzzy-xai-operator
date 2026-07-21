#!/usr/bin/env python3
"""Fail-closed verification of the Q1 remediation evidence package."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterator


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "release_evidence/q1_remediation"


def load(relative: str) -> dict[str, object]:
    path = EVIDENCE / relative
    if not path.is_file():
        raise RuntimeError(f"missing required Q1 evidence: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def numbers(value: object) -> Iterator[float]:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        yield float(value)
    elif isinstance(value, dict):
        for item in value.values():
            yield from numbers(item)
    elif isinstance(value, list):
        for item in value:
            yield from numbers(item)


def main() -> None:
    baseline = load("baseline_snapshot/manifest.json")
    if baseline["base_commit"] != "cafe403c7d60e36b08f56a5325ba380718a5be35" or baseline["file_count"] < 40:
        raise RuntimeError("baseline snapshot is incomplete")
    h1 = load("fidelity/h1_fidelity_noninferiority.json")["summary"]
    h2 = load("traceability/h2_traceability_missingness.json")
    h3 = load("cascade/h3_adaptive_cascade.json")
    h4 = load("uncertainty/h4_uncertainty_hierarchy.json")
    h5 = load("critical_rupture/h5_critical_rupture.json")
    h6 = load("rule_ablation/h6_rule_ablation.json")
    calibration = load("calibration/q1_calibration.json")
    external = load("external_studies/status.json")
    checks = {
        "h1_paired_noninferiority": bool(h1["n_pairs"] and h1["margin"] == -0.02),
        "h2_traceability": float(h2["fuzzyxai_k_trace"]) > float(h2["baseline_k_trace"]),
        "h2_missingness": float(h2["missingness"]["f1"]) >= 0.95,
        "h3_all_comparators": len(h3["policies"]) == 5,
        "h4_all_modes": {row["mode"] for row in h4["rows"]} == {"always_F0", "always_Fint", "always_NAS", "always_FML", "adaptive"},
        "h5_structural_predictive_split": "structural" in h5 and "predictive" in h5,
        "h5_predictive_wording_blocked_on_no_gain": bool(h5["predictive_claim_allowed"] or h5["allowed_interpretation"] == "critical rupture is a structural diagnostic indicator only"),
        "h6_repetitions": int(h6["summary"]["n_pairs"]) >= 50,
        "calibration_validation_only": calibration["test_partition_used"] is False,
        "external_gates_honest": external["comprehension"]["status"] == "planned_not_run" and not external["comprehension"]["claim_allowed"],
    }
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Q1 verification failures: {failed}")
    registry = load("claim_registry.json")
    for row in registry["claims"]:
        for relative in row["evidence"]:
            if not (EVIDENCE / relative).is_file():
                raise RuntimeError(f"claim {row['claim_id']} has missing evidence: {relative}")
    for relative in (
        "fidelity/h1_fidelity_noninferiority.json",
        "traceability/h2_traceability_missingness.json",
        "cascade/h3_adaptive_cascade.json",
        "uncertainty/h4_uncertainty_hierarchy.json",
        "critical_rupture/h5_critical_rupture.json",
        "rule_ablation/h6_rule_ablation.json",
    ):
        nonfinite = [value for value in numbers(load(relative)) if not math.isfinite(value)]
        if nonfinite:
            raise RuntimeError(f"unexplained non-finite values in {relative}")
    manifest = load("manifest_sha256.json")
    for row in manifest["files"]:
        path = ROOT / row["path"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise RuntimeError(f"checksum mismatch: {row['path']}")
    print("PASS: q1_verify_all")


if __name__ == "__main__":
    main()
