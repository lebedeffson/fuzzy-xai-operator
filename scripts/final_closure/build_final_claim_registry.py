#!/usr/bin/env python3
"""Enable only claims supported by immutable confirmatory statistics."""

from __future__ import annotations

from common import EVIDENCE, STUDY, load, write


def main() -> None:
    statistics_path = STUDY / "confirmatory/final_statistics.json"
    if not statistics_path.is_file():
        raise SystemExit("BLOCKED: final claim registry requires confirmatory statistics")
    statistics = load(statistics_path)
    h3 = statistics["H3"]["P1_vs_baseline"]
    h5 = next(row for row in statistics["H5-A"]["methods"] if row["method"] == "typed_route_validity")
    h6a = statistics["H6-A"]
    h6b = statistics["H6-B"]
    h7 = statistics["H7-A"]
    h8 = statistics["H8"]
    h9 = statistics["H9"]
    claims = {
        "H3-P1": _status(
            h3["relative_invalid_action_reduction"] >= 0.15
            and h3["confidence_interval_95"][0] > 0
            and h3["holm_adjusted_p"] < 0.05
        ),
        "H3-P2": _fixed_risk_status(statistics["H3"]["fixed_risk"]),
        "H3-P3": _status(statistics["H3"]["P1_vs_P0"]["confidence_interval_95"][0] > 0 and statistics["H3"]["P1_vs_P0"]["holm_adjusted_p"] < 0.05),
        "H3-P4": "dataset_or_stratum_specific_pending_detailed_analysis",
        "H5-A": _status(h5["f1"] >= 0.95 and h5["false_certification"] <= 0.01 and h5["source_localization"] >= 0.90),
        "H6-A": _status(h6a["detection_rate"] >= 0.80 and h6a["false_discovery_rate"] <= 0.10 and h6a["sign_accuracy"] >= 0.90),
        "H6-B": _status(h6b["replicated_positive_direction"]),
        "H7-A": _status(h7["canonical_hash_preservation_rate"] == 1.0 and h7["artifacts"] >= 10_000),
        "H7-B": "not_supported_projection_tradeoff_incomplete",
        "H8": _status(h8["recommended_range_target_met"]),
        "H9": _status(
            h9["maximum_objects"] >= 5_000_000
            and h9["operator_only"]["empirical_scaling_exponent"] <= 1.10
            and h9["operator_only"]["deterministic_repeat"]
        ),
    }
    payload = {
        "schema_version": "3.0",
        "phase": "sealed_confirmatory",
        "frozen_previous": {
            "H1": "supported", "H2": "supported", "H3-original": "not_supported", "H4": "supported",
            "H5-S": "supported", "H5-P-original": "not_supported", "H6-general": "not_supported",
        },
        "new_claims": claims,
        "human_claims": {
            "understandable_to_users": "out_of_scope_disabled",
            "confirmed_by_experts": "out_of_scope_disabled",
            "improves_domain_safety": "out_of_scope_disabled",
            "matches_specialist_decisions": "out_of_scope_disabled",
        },
        "manual_positive_override_allowed": False,
    }
    write(EVIDENCE / "final_claim_registry.json", payload)
    print(f"PASS: final_claim_registry supported={sum(value == 'supported' for value in claims.values())} manual_override=false")


def _status(condition: bool) -> str:
    return "supported" if condition else "not_supported"


def _fixed_risk_status(result: dict[str, object]) -> str:
    if result.get("status") == "not_estimable_no_development_operating_point":
        return "not_supported_no_development_operating_point"
    return _status(
        result["coverage_gain_vs_baseline"] >= 0.05
        and result["confidence_interval_95"][0] > 0
        and result.get("holm_adjusted_p", 1.0) < 0.05
        and result["observed_test_risk"]["full_fuzzyxai_P1"] <= result["risk_ceiling"]
    )


if __name__ == "__main__":
    main()
