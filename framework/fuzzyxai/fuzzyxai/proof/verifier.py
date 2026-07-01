from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fuzzyxai.core.types import ProofTrace


@dataclass(frozen=True)
class VerificationResult:
    valid: bool
    errors: list[str]


def _as_dict(trace: ProofTrace | dict[str, Any]) -> dict[str, Any]:
    return trace.to_dict() if isinstance(trace, ProofTrace) else trace


def verify_proof_trace(trace: ProofTrace | dict[str, Any]) -> VerificationResult:
    data = _as_dict(trace)
    errors: list[str] = []
    if data.get("package_type") != "FuzzyXAIProofTrace":
        errors.append("invalid:package_type")
    route = data.get("route", {}) or {}
    computed = data.get("computed_result", {}) or {}
    if route.get("computed_result") != computed:
        errors.append("inconsistent:route.computed_result")
    if route.get("final_action") != data.get("final_action"):
        errors.append("inconsistent:final_action")
    if computed.get("chi_crit") == 1 and computed.get("action") != "block":
        errors.append("invalid:critical_not_blocked")
    diagnostics = data.get("diagnostics", []) or []
    diagnostic_ids = {item.get("diagnostic_id") for item in diagnostics if isinstance(item, dict)}
    if computed.get("diagnostic_id") and computed.get("diagnostic_id") not in diagnostic_ids:
        errors.append("inconsistent:diagnostic_id")
    return VerificationResult(valid=not errors, errors=errors)
