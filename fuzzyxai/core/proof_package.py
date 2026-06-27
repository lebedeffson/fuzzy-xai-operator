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


def _stable_hash(payload: Any) -> str:
    return sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _code_version() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


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
        "explain_plan_hash": _stable_hash(explain_plan),
        "operator_values": [
            {
                "node_id": item.get("node_id"),
                "operator_id": item.get("operator_id"),
                "status": item.get("status"),
                "computed": item.get("computed", {}),
                "diagnostics": item.get("diagnostics", []),
            }
            for item in report.get("operators", [])
        ],
        "computed_result": (engine_payload or {}).get("computed_result", {}),
        "diagnostics": report.get("diagnostics", []),
        "final_action": report.get("final_action"),
        "action_reason": report.get("action_reason"),
        "scenario_hash": _stable_hash(scenario),
    }
    package["package_hash"] = _stable_hash({k: v for k, v in package.items() if k != "package_hash"})
    return package


def verify_proof_package(package: dict[str, Any]) -> VerificationResult:
    errors: list[str] = []
    for key in ["package_type", "scenario_id", "run_id", "explain_plan_hash", "operator_values", "diagnostics", "final_action", "package_hash"]:
        if key not in package:
            errors.append(f"missing:{key}")
    if package.get("package_type") != "FuzzyXAIProofPackage":
        errors.append("invalid:package_type")
    expected_hash = _stable_hash({k: v for k, v in package.items() if k != "package_hash"})
    if package.get("package_hash") != expected_hash:
        errors.append("invalid:package_hash")
    if package.get("final_action") == "block" and not package.get("diagnostics"):
        errors.append("invalid:block_without_diagnostics")
    return VerificationResult(valid=not errors, errors=errors)
