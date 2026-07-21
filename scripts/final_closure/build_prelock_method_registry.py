#!/usr/bin/env python3
"""Register measured formative evidence and confirmatory-only method boundaries."""

from __future__ import annotations

from common import ROOT, STUDY, load, sha256, write


HISTORICAL = ROOT / "release_evidence/final_practical_closure/formative"


def _artifact(path):
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)}


def main() -> None:
    current = {
        "H3": STUDY / "formative_real/summary.json",
        "H7": STUDY / "h7_formative/summary.json",
        "comparators": STUDY / "comparator_formative/summary.json",
    }
    historical = {
        "H5-A": HISTORICAL / "H5_A_route_validity/summary.json",
        "H6-A": HISTORICAL / "H6_A_detectability/summary.json",
        "H8": HISTORICAL / "H8_grid/summary.json",
        "H9": HISTORICAL / "H9_scaling/summary.json",
    }
    missing = [path.relative_to(ROOT).as_posix() for path in (*current.values(), *historical.values()) if not path.is_file()]
    if missing:
        raise SystemExit(f"FAIL: missing formative method evidence: {missing}")
    h7 = load(current["H7"])
    h6 = load(historical["H6-A"])
    payload = {
        "schema_version": "1.0",
        "phase": "prelock_method_readiness",
        "confirmatory_test_opened": False,
        "current_formative": {name: _artifact(path) for name, path in current.items()},
        "frozen_historical_formative": {name: _artifact(path) for name, path in historical.items()},
        "method_status": {
            "H3-P1-P4": "measured_formative_current_not_confirmatory",
            "H5-A": "controlled_formative_ready_for_sealed_replay",
            "H6-A": "formative_target_not_met_method_boundary_preserved",
            "H6-B": "confirmatory_only_requires_two_sealed_tabular_datasets",
            "H7-A": "formative_pass_exact_hash_preservation",
            "H7-B": "blocked_projection_stability_not_measured",
            "H8": "controlled_formative_ready_for_independent_check",
            "H9": "operator_only_formative_ready_end_to_end_pending",
        },
        "method_checks": {
            "H7_A_preservation": h7["H7_A"]["canonical_hash_preservation_rate"],
            "H6_A_formative_target_met": bool(h6.get("formative_target_met")),
            "H6_B_result_available": False,
        },
        "lock_readiness": "ready_with_confirmatory_only_experiments_pending",
        "confirmatory_claim_allowed": False,
    }
    write(STUDY / "prelock_method_registry.json", payload)
    print("PASS: final_prelock_method_registry claims=confirmatory_pending")


if __name__ == "__main__":
    main()
