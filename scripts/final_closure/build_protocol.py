#!/usr/bin/env python3
"""Freeze final-cycle endpoints and finite formative iteration history."""

from __future__ import annotations

import subprocess

from common import BASE, ROOT, STUDY, sha256, write


def main() -> None:
    subprocess.run(["git", "merge-base", "--is-ancestor", BASE, "HEAD"], cwd=ROOT, check=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    protocol = {
        "schema_version": "1.0",
        "identifier": "FXAI-FINAL-PRACTICAL-CONFIRMATORY-CLOSURE",
        "base_commit": BASE,
        "implementation_commit": head,
        "phase": "formative_iteration_2_prelock",
        "primary_endpoint": "operationally_invalid_automatic_actions_at_20_percent_review_budget",
        "primary_review_budget": 0.20,
        "secondary_review_budgets": [0.05, 0.10, 0.30],
        "primary_cost_profile": "unsafe_accept_sensitive",
        "false_block_ceiling": 0.01,
        "hard_fault_recall_minimum": 0.95,
        "maximum_formative_iterations": 3,
        "current_formative_iteration": 2,
        "confirmatory_test_opened": False,
        "post_lock_changes_forbidden": True,
        "immutable_results": {
            "H1": "supported", "H2": "supported", "H3-original": "not_supported", "H4": "supported",
            "H5-S": "supported", "H5-P-original": "not_supported", "H6-general": "not_supported",
        },
    }
    write(STUDY / "protocol.json", protocol)
    iteration = {
        "iterations": [
            {
                "iteration": 1,
                "commit": BASE,
                "result": "completed",
                "config_sha256": _hash_if_present(ROOT / "study/final_practical_closure/practical_protocol.json"),
                "results_sha256": _hash_if_present(ROOT / "release_evidence/final_practical_closure/formative/summary.json"),
                "reason": "initial practical-controller formative cycle",
            },
            {
                "iteration": 2,
                "commit": head,
                "result": "in_progress",
                "config_sha256": sha256(STUDY / "protocol.json"),
                "results_sha256": None,
                "reason": "predeclared confirmatory contracts, H6 estimand correction and run-2 card preparation",
            },
        ],
        "iteration_3_allowed_only_if_predeclared_before_iteration_2_results": True,
    }
    write(STUDY / "formative_iteration_log.json", iteration)
    write(STUDY / "protocol_manifest.json", {"protocol_sha256": sha256(STUDY / "protocol.json"), "test_opened": False})
    print("PASS: final_protocol iteration=2/3 confirmatory_opened=false")


def _hash_if_present(path):
    return sha256(path) if path.is_file() else None


if __name__ == "__main__":
    main()
