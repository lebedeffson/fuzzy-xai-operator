from __future__ import annotations

from collections.abc import Iterable, Mapping


CLAIM_MARGIN = 0.04
P_VALUE_MAX = 0.05
POSITIVE_PIPELINES_MIN = 5
FALSE_CERTIFICATION_MAX = 0.01
NEW_CRITICAL_VIOLATIONS_MAX = 0.01
RUNTIME_P95_MS_MAX = 50.0


def _claim(
    statistics: Mapping[str, object],
    *,
    require_cost_regret: bool,
) -> bool:
    passed = (
        float(statistics["effect"]) >= CLAIM_MARGIN
        and float(statistics["ci_low"]) > 0
        and float(statistics["p_holm"]) < P_VALUE_MAX
        and int(statistics["positive_pipeline_families"])
        >= POSITIVE_PIPELINES_MIN
    )
    if not require_cost_regret:
        return passed
    return (
        passed
        and float(statistics["cost_regret_effect"]) > 0
        and float(statistics["cost_regret_ci_low"]) > 0
        and float(statistics["cost_regret_p_holm"]) < P_VALUE_MAX
    )


def classify_confirmatory_result(
    statistics: Iterable[Mapping[str, object]],
    *,
    false_certification: float,
    new_critical_violations: float,
    runtime_p95_ms: float,
) -> dict[str, object]:
    by_claim = {str(item["claim"]): item for item in statistics}
    required = {"H10-C3a", "H10-C3b"}
    if set(by_claim) != required:
        raise ValueError("confirmatory statistics must contain H10-C3a and H10-C3b")

    safety = {
        "false_certification": (
            false_certification <= FALSE_CERTIFICATION_MAX
        ),
        "new_critical_violations": (
            new_critical_violations <= NEW_CRITICAL_VIOLATIONS_MAX
        ),
        "runtime_p95": runtime_p95_ms <= RUNTIME_P95_MS_MAX,
    }
    claim_pass = {
        "H10-C3a": _claim(
            by_claim["H10-C3a"],
            require_cost_regret=True,
        ),
        "H10-C3b": _claim(
            by_claim["H10-C3b"],
            require_cost_regret=False,
        ),
    }
    safety_pass = all(safety.values())
    return {
        "H10-C3a": (
            "CONFIRMATORY_PASS"
            if claim_pass["H10-C3a"] and safety_pass
            else "CONFIRMATORY_FAIL"
        ),
        "H10-C3b": (
            "CONFIRMATORY_PASS"
            if claim_pass["H10-C3b"] and safety_pass
            else "CONFIRMATORY_FAIL"
        ),
        "safety": {
            "checks": safety,
            "false_certification": false_certification,
            "new_critical_violations": new_critical_violations,
            "runtime_p95_ms": runtime_p95_ms,
            "status": "PASS" if safety_pass else "FAIL",
        },
        "scientific_status": (
            "SCIENTIFIC_PASS"
            if all(claim_pass.values()) and safety_pass
            else "SCIENTIFIC_FAIL"
        ),
    }
