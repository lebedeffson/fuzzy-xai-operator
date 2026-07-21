#!/usr/bin/env python3
"""Build the immutable development protocol for the final practical cycle."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "study/final_practical_closure"
BASE = "63cef7578d28a28dac63654f24642a980b49bc90"


def main() -> None:
    subprocess.run(["git", "merge-base", "--is-ancestor", BASE, "HEAD"], cwd=ROOT, check=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _write("frozen_results.json", _frozen())
    _write("practical_protocol.json", _protocol())
    _write("confirmatory_dataset_requirements.json", _datasets())
    _write("external_claim_gates.json", _external_gates())
    paths = sorted(path for path in OUTPUT.glob("*.json") if path.name != "manifest.json")
    _write(
        "manifest.json",
        {
            "schema_version": "1.0",
            "base_commit": BASE,
            "implementation_commit": _git("rev-parse", "HEAD"),
            "stage": "formative_development",
            "confirmatory_protocol_locked": False,
            "confirmatory_test_opened": False,
            "stable_technical_release_allowed": False,
            "files": [{"path": path.relative_to(ROOT).as_posix(), "sha256": _sha(path)} for path in paths],
        },
    )
    print("PASS: practical_protocol hypotheses=11 budgets=5 cost_profiles=3 confirmatory_opened=false")


def _frozen() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "immutable": {
            "H1": "supported",
            "H2": "supported",
            "H3-original": "not_supported",
            "H4": "supported",
            "H5-S": "supported",
            "H5-P-original": "not_supported",
            "H6-general": "not_supported",
        },
        "new": {
            "H3-P1": "not_run",
            "H3-P2": "not_run",
            "H3-P3": "not_run",
            "H3-P4": "not_run",
            "H5-A": "not_run",
            "H6-A": "not_run",
            "H6-B": "not_run",
            "H7-A": "not_run",
            "H7-B": "not_run",
            "H8": "not_run",
            "H9": "not_run",
        },
    }


def _protocol() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "stage": "formative_development",
        "target": "operationally_invalid_automatic_action",
        "invalid_action_definition": [
            "wrong prediction accepted",
            "route contract violation accepted",
            "mandatory provenance absent during accept",
            "out-of-applicability input accepted",
            "critical explanation instability accepted",
            "hard guard bypassed",
            "action contradicts frozen operational contract",
        ],
        "review_budgets": [0.05, 0.10, 0.20, 0.30, 1.0],
        "primary_cost_profile": "unsafe_accept_sensitive",
        "cost_profiles": {
            "balanced": {"unsafe_accept": 8.0, "short_review": 0.4, "full_review": 1.0, "false_block": 3.0},
            "unsafe_accept_sensitive": {"unsafe_accept": 20.0, "short_review": 0.5, "full_review": 1.2, "false_block": 4.0},
            "review_expensive": {"unsafe_accept": 10.0, "short_review": 1.5, "full_review": 4.0, "false_block": 3.0},
        },
        "frozen_strata": [
            "high_confidence_disagreement",
            "low_confidence_object",
            "unstable_explanation",
            "incomplete_provenance",
            "detected_shift",
            "rare_group",
            "boundary_object",
            "route_fault",
        ],
        "false_block_ceiling": 0.01,
        "hard_fault_recall_minimum": 0.95,
        "H3_P1_minimum_relative_reduction": 0.15,
        "H3_P2_minimum_coverage_gain": 0.05,
        "statistics": ["paired bootstrap", "cluster bootstrap", "McNemar", "Wilcoxon", "permutation test", "Holm correction"],
        "forbidden_after_lock": [
            "feature changes",
            "threshold changes",
            "cost-profile changes",
            "dataset changes",
            "case removal based on outcomes",
            "new primary metrics",
            "new subgroups",
        ],
        "confirmatory_test_opened": False,
    }


def _datasets() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "status": "not_sealed",
        "minimum": {"tabular": 2, "image": 1, "text": 1, "timeseries": 1},
        "requirements": [
            "not used in formative tuning",
            "group split for repeated entities",
            "temporal split for time series",
            "test identities frozen by SHA256",
            "labels unavailable to tuning runner",
            "controller features out-of-fold",
        ],
        "datasets": [],
    }


def _external_gates() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "domain_language": "open_external_not_in_scope",
        "comprehension": "open_external_not_in_scope",
        "expert_action": "open_external_not_in_scope",
        "claims_enabled": [],
        "claims_removed": [
            "understandable to users",
            "confirmed by experts",
            "improves domain safety",
            "better than specialist decisions",
        ],
        "technical_release_blocked_by_external_gates": False,
        "scientific_claim_rule": "enabled claim with absent evidence is blocked; removed out-of-scope claim is not a technical-release blocker",
    }


def _write(name: str, payload: object) -> None:
    (OUTPUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


if __name__ == "__main__":
    main()
