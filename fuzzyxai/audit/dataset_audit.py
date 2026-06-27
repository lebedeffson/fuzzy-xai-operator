from __future__ import annotations

import json
from pathlib import Path

from fuzzyxai.audit.common import ROOT
from fuzzyxai.data.registry import load_dataset_registry
from fuzzyxai.data.validators import validate_beacon, validate_ecg, validate_gd, validate_gis, validate_hybrid


OUT = ROOT / "reports" / "dataset_audit"


VALIDATORS = {
    "hybrid_xiris": validate_hybrid,
    "medical_ecg_signal": validate_ecg,
    "gd_anfis_shap": validate_gd,
    "beacon_xai": validate_beacon,
    "gis_integro": validate_gis,
}


def run_audit() -> dict[str, object]:
    registry = load_dataset_registry()
    issues: list[dict[str, str]] = []
    for scenario_id, validator in VALIDATORS.items():
        if scenario_id not in registry:
            issues.append({"scenario_id": scenario_id, "severity": "BLOCKER", "message": "missing registry entry"})
            continue
        for message in validator(registry[scenario_id]):
            issues.append({"scenario_id": scenario_id, "severity": "BLOCKER", "message": message})
    return {
        "status": "PASS" if not issues else "FAIL",
        "checked_scenarios": sorted(VALIDATORS),
        "issues": issues,
    }


def write_report(result: dict[str, object]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dataset_audit_report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Dataset audit",
        "",
        f"Status: {result['status']}",
        "",
        "Checked scenarios:",
        *[f"- `{sid}`" for sid in result["checked_scenarios"]],
        "",
        "Issues:",
    ]
    issues = result["issues"]
    if issues:
        lines.extend(f"- {i['severity']} {i['scenario_id']}: {i['message']}" for i in issues)  # type: ignore[index]
    else:
        lines.append("- none")
    (OUT / "dataset_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    result = run_audit()
    write_report(result)
    print(OUT / "dataset_audit_report.md")
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

