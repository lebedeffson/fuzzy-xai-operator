from __future__ import annotations

import json
from pathlib import Path

from fuzzyxai.audit.common import ROOT, current_commit


OUT = ROOT / "reports/practice_demo"
QA = OUT / "qa"
PROOFS = OUT / "proof_packages"
SCENARIO_INPUTS = OUT / "scenario_inputs"

EXPECTED_ACTIONS = {
    "hybrid_xiris": "block",
    "medical_ecg_signal": "defer_to_human",
    "gd_anfis_shap": "audit",
    "beacon_xai": "audit",
    "gis_integro": "audit_report",
}

EXPECTED_DIAGNOSTICS = {
    "hybrid_xiris": "D_quality_source_conflict",
    "medical_ecg_signal": "D_signal_quality",
    "gd_anfis_shap": "D_rule_attribution_conflict",
    "beacon_xai": "D_counterevidence_conflict",
    "gis_integro": "D_route_context_limit",
}


def run_qc() -> dict[str, object]:
    QA.mkdir(parents=True, exist_ok=True)
    issues: list[str] = []
    rows = []
    for sid, action in EXPECTED_ACTIONS.items():
        path = PROOFS / f"{sid}_proof_package.json"
        if not path.exists():
            issues.append(f"{sid}: missing proof package")
            continue
        proof = json.loads(path.read_text(encoding="utf-8"))
        input_path = SCENARIO_INPUTS / f"{sid}_input.json"
        expected_input_action = None
        if input_path.exists():
            expected_input_action = json.loads(input_path.read_text(encoding="utf-8")).get("expected_outputs", {}).get("action")
        checks = {
            "package_type": proof.get("package_type") == "FuzzyXAIProofPackage",
            "verifier_status": proof.get("verifier_status") == "PASS",
            "source_commit": proof.get("source_commit") == current_commit(),
            "code_version": proof.get("code_version") == current_commit(),
            "artifact_commit": proof.get("artifact_commit") == current_commit(),
            "scenario_input_hash": bool(proof.get("scenario_input_hash")),
            "model_card_hash": bool(proof.get("model_card_hash")),
            "computed_result": bool(proof.get("computed_result")),
            "operator_values": bool(proof.get("operator_values")),
            "diagnostics": bool(proof.get("diagnostics")),
            "final_action": proof.get("final_action") == action == expected_input_action,
            "diagnostic_id": proof.get("diagnostics", [{}])[0].get("diagnostic_id") == EXPECTED_DIAGNOSTICS[sid],
        }
        failed = [name for name, ok in checks.items() if not ok]
        if failed:
            issues.append(f"{sid}: failed {', '.join(failed)}")
        rows.append({"scenario_id": sid, "final_action": proof.get("final_action"), "diagnostic_id": proof.get("diagnostics", [{}])[0].get("diagnostic_id"), "status": "PASS" if not failed else "FAIL"})
    result = {"status": "PASS" if not issues else "FAIL", "issues": issues, "proof_packages": rows}
    (QA / "QA_PROOF_PACKAGE_SCHEMA.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# QA Proof Package Schema", "", f"Status: {result['status']}", "", "| scenario | action | diagnostic | status |", "|---|---|---|---|"]
    for row in rows:
        lines.append(f"| `{row['scenario_id']}` | `{row['final_action']}` | `{row['diagnostic_id']}` | {row['status']} |")
    (QA / "QA_PROOF_PACKAGE_SCHEMA.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    result = run_qc()
    print(QA / "QA_PROOF_PACKAGE_SCHEMA.md")
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

