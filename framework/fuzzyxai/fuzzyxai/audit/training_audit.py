from __future__ import annotations

import json

from fuzzyxai.audit.common import ROOT
from fuzzyxai.practice.fixtures import SCENARIOS


def main() -> None:
    issues: list[str] = []
    for sid, fixture in SCENARIOS.items():
        training = ROOT / "reports/training" / f"{sid}_training_report.json"
        eval_report = ROOT / "reports/evaluation" / f"{sid}_eval_report.json"
        if not training.exists():
            issues.append(f"missing training report: {sid}")
        if not eval_report.exists():
            issues.append(f"missing evaluation report: {sid}")
        if training.exists():
            data = json.loads(training.read_text(encoding="utf-8"))
            if data.get("model_status") != fixture["model_status"]:
                issues.append(f"{sid}: model_status mismatch")
            if not data.get("not_a_claim"):
                issues.append(f"{sid}: missing not_a_claim")
        if eval_report.exists():
            data = json.loads(eval_report.read_text(encoding="utf-8"))
            if data.get("status") != "PASS":
                issues.append(f"{sid}: eval status not PASS")
            if not data.get("scenario_input_hash"):
                issues.append(f"{sid}: missing scenario_input_hash")
    result = {"status": "PASS" if not issues else "FAIL", "issues": issues}
    out = ROOT / "reports/training/training_audit_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out)
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

