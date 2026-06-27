from copy import deepcopy

import pytest

from fuzzyxai.core.proof_package import build_proof_package, verify_proof_package
from fuzzyxai.core.scenario_engine import hybrid_xiris_engine_payload
from fuzzyxai.studio.operator_scenarios import build_report, load_scenarios


def _package() -> dict:
    scenario = next(s for s in load_scenarios() if s["scenario_id"] == "hybrid_xiris")
    return build_proof_package(scenario, build_report(scenario), hybrid_xiris_engine_payload())


@pytest.mark.parametrize(
    ("node", "field", "value", "expected"),
    [
        ("risk_observer", "rho", 0.74, "inconsistent:risk_observer.rho"),
        ("reduction", "delta", 0.08, "inconsistent:reduction.delta"),
        ("alignment", "gamma_ij", 0.35, "inconsistent:alignment.gamma"),
    ],
)
def test_verifier_rejects_operator_value_tamper(node: str, field: str, value: float, expected: str) -> None:
    package = _package()
    for item in package["operator_values"]:
        if item["node_id"] == node:
            item["computed"][field] = value
    assert any(error.startswith(expected) for error in verify_proof_package(package).errors)


def test_verifier_rejects_accept_when_critical() -> None:
    package = _package()
    package["computed_result"]["action"] = "accept"
    assert "invalid:critical_not_blocked" in verify_proof_package(package).errors


def test_verifier_rejects_f0_for_source_conflict() -> None:
    package = _package()
    package["computed_result"]["selected_class"] = "F0"
    assert "invalid:F0_selected_for_source_conflict" in verify_proof_package(package).errors


def test_verifier_rejects_delta_over_max_but_passed() -> None:
    package = _package()
    for item in package["operator_values"]:
        if item["node_id"] == "reduction":
            item["computed"]["delta"] = 0.99
            item["status"] = "passed"
    errors = verify_proof_package(package).errors
    assert "invalid:reduction_passed_over_delta_max" in errors


def test_verifier_rejects_gamma_over_max_but_passed() -> None:
    package = _package()
    for item in package["operator_values"]:
        if item["node_id"] == "alignment":
            item["computed"]["gamma_ij"] = 0.99
            item["status"] = "passed"
    errors = verify_proof_package(package).errors
    assert "invalid:alignment_passed_over_gamma_max" in errors


def test_verifier_rejects_code_version_when_requested() -> None:
    package = _package()
    package["source_commit"] = "old"
    assert any(error.startswith("invalid:source_commit") for error in verify_proof_package(package, require_current_code_version=True).errors)


def test_verifier_rejects_explain_plan_hash_tamper() -> None:
    package = _package()
    package["explain_plan_hash"] = "bad"
    assert "invalid:explain_plan_hash" in verify_proof_package(package).errors


def test_verifier_rejects_untyped_legacy_diagnostic() -> None:
    package = _package()
    package["diagnostics"][0] = deepcopy(package["diagnostics"][0])
    package["diagnostics"][0]["diagnostic_id"] = "D_source_conflict"
    package["diagnostics"][0].pop("diagnostic_type", None)
    package["diagnostics"][0].pop("legacy_id", None)
    assert "invalid:legacy_diagnostic_without_typed_id" in verify_proof_package(package).errors
