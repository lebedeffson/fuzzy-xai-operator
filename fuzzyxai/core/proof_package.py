from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any


@dataclass(frozen=True)
class VerificationResult:
    valid: bool
    errors: list[str]


def canonicalize_explain_plan(plan: Any) -> str:
    return json.dumps(plan, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def compute_explain_plan_hash(plan: Any) -> str:
    return sha256(canonicalize_explain_plan(plan).encode("utf-8")).hexdigest()


def _stable_hash(payload: Any) -> str:
    return sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _code_version() -> str:
    try:
        head = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = subprocess.run(["git", "diff", "--quiet"], check=False)
        staged = subprocess.run(["git", "diff", "--cached", "--quiet"], check=False)
        return f"{head}-dirty" if dirty.returncode or staged.returncode else head
    except Exception:
        return "unknown-dirty"


def _operator_values_from_engine(engine_payload: dict[str, Any] | None, report: dict[str, Any]) -> list[dict[str, Any]]:
    values = (engine_payload or {}).get("computed_result", {}).get("operator_values")
    if values:
        diagnostics = report.get("diagnostics", [])
        return [{**item, "diagnostics": diagnostics if item.get("node_id") in {"alignment", "risk_observer", "action"} else item.get("diagnostics", [])} for item in values]
    return [
        {
            "node_id": item.get("node_id"),
            "operator_id": item.get("operator_id"),
            "status": item.get("status"),
            "computed": item.get("computed", {}),
            "diagnostics": item.get("diagnostics", []),
        }
        for item in report.get("operators", [])
    ]


def build_proof_package(scenario: dict[str, Any], report: dict[str, Any], engine_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    explain_plan = (engine_payload or {}).get("explain_plan", {})
    package = {
        "package_type": "FuzzyXAIProofPackage",
        "schema_version": "1.0",
        "scenario_id": scenario.get("scenario_id"),
        "run_id": report.get("run_id"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_version": _code_version(),
        "input": (engine_payload or {}).get("input_values", {}),
        "model_version": report.get("trace", {}).get("model_version"),
        "adapter_version": report.get("trace", {}).get("adapter_version"),
        "explain_plan": explain_plan,
        "explain_plan_hash": compute_explain_plan_hash(explain_plan),
        "operator_values": _operator_values_from_engine(engine_payload, report),
        "computed_result": (engine_payload or {}).get("computed_result", {}),
        "diagnostics": report.get("diagnostics", []),
        "final_action": report.get("final_action"),
        "action_reason": report.get("action_reason"),
        "scenario_hash": _stable_hash(scenario),
    }
    package["package_hash"] = _stable_hash({k: v for k, v in package.items() if k != "package_hash"})
    return package


def verify_proof_package(package: dict[str, Any], require_current_code_version: bool = False) -> VerificationResult:
    errors: list[str] = []
    for key in ["package_type", "scenario_id", "run_id", "explain_plan_hash", "operator_values", "diagnostics", "final_action", "package_hash"]:
        if key not in package:
            errors.append(f"missing:{key}")
    if package.get("package_type") != "FuzzyXAIProofPackage":
        errors.append("invalid:package_type")
    expected_hash = _stable_hash({k: v for k, v in package.items() if k != "package_hash"})
    if package.get("package_hash") != expected_hash:
        errors.append("invalid:package_hash")
    if package.get("explain_plan_hash") != compute_explain_plan_hash(package.get("explain_plan", {})):
        errors.append("invalid:explain_plan_hash")
    if require_current_code_version and package.get("code_version") != _code_version():
        errors.append(f"invalid:code_version:{package.get('code_version')}!={_code_version()}")
    if package.get("final_action") == "block" and not package.get("diagnostics"):
        errors.append("invalid:block_without_diagnostics")
    computed = package.get("computed_result", {})
    operators = {item.get("node_id"): item for item in package.get("operator_values", [])}

    def approx_equal(left: Any, right: Any, tol: float = 1e-6) -> bool:
        try:
            return abs(float(left) - float(right)) <= tol
        except Exception:
            return left == right

    checks = [
        ("alignment.gamma", computed.get("gamma"), operators.get("alignment", {}).get("computed", {}).get("gamma_ij")),
        ("reduction.delta", computed.get("delta"), operators.get("reduction", {}).get("computed", {}).get("delta")),
        ("risk_observer.rho", computed.get("rho"), operators.get("risk_observer", {}).get("computed", {}).get("rho")),
        ("action.action", computed.get("action"), operators.get("action", {}).get("computed", {}).get("action")),
    ]
    for label, expected, actual in checks:
        if expected is not None and actual is not None and not approx_equal(expected, actual):
            errors.append(f"inconsistent:{label}:{expected}!={actual}")

    if computed.get("action") == "block" and int(computed.get("chi_r_crit", computed.get("chi_R_crit", 0))) != 1:
        errors.append("invalid:block_without_chi_r_crit")
    if int(computed.get("chi_r_crit", computed.get("chi_R_crit", 0))) == 1 and computed.get("action") != "block":
        errors.append("invalid:critical_not_blocked")
    if computed.get("selected_class") == "F0" and computed.get("diagnostic_type") == "quality_source_conflict":
        errors.append("invalid:F0_selected_for_source_conflict")
    for diagnostic in package.get("diagnostics", []):
        if computed.get("chi_r_crit", computed.get("chi_R_crit")) == 1 and diagnostic.get("criticality") != "high":
            errors.append("invalid:critical_without_high_diagnostic")
        if diagnostic.get("diagnostic_id") == "D_source_conflict" and not diagnostic.get("diagnostic_type") and not diagnostic.get("legacy_id"):
            errors.append("invalid:legacy_diagnostic_without_typed_id")
    alignment = operators.get("alignment", {}).get("computed", {})
    alignment_status = operators.get("alignment", {}).get("status")
    if alignment and alignment_status == "passed" and float(alignment.get("gamma_ij", 0.0)) > float(alignment.get("gamma_max", 1.0)):
        errors.append("invalid:alignment_passed_over_gamma_max")
    reduction = operators.get("reduction", {}).get("computed", {})
    reduction_status = operators.get("reduction", {}).get("status")
    if reduction and reduction_status == "passed" and float(reduction.get("delta", 0.0)) > float(reduction.get("delta_max", 1.0)):
        errors.append("invalid:reduction_passed_over_delta_max")
    return VerificationResult(valid=not errors, errors=errors)
