from __future__ import annotations

import csv
import json
from pathlib import Path

from fuzzyxai.audit.common import ROOT, current_commit


REPORTS = ROOT / "reports" / "real_validation"
MANIFEST = REPORTS / "real_artifacts_manifest.json"


def run_validation() -> dict[str, object]:
    issues: list[str] = []
    if not MANIFEST.exists():
        return {"status": "FAIL", "issues": ["missing real_artifacts_manifest.json"]}
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    iris = manifest["artifacts"]["iris"]
    ecg = manifest["artifacts"]["ecg"]
    iris_path = ROOT / iris["path"]
    ecg_csv = ROOT / ecg["csv_path"]
    if not iris_path.exists():
        issues.append("missing real iris image")
    if iris.get("width", 0) < 512 or iris.get("height", 0) < 512:
        issues.append("real iris image too small")
    if not ecg_csv.exists():
        issues.append("missing parsed ECG csv")
    else:
        with ecg_csv.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        if len(rows) < 3000:
            issues.append(f"ECG csv too short: {len(rows)}")
        vals = [float(r["mlII"]) for r in rows[:1000]]
        if max(vals) - min(vals) <= 100:
            issues.append("ECG signal amplitude range too small")
    return {
        "status": "PASS" if not issues else "FAIL",
        "source_commit": current_commit(),
        "issues": issues,
        "real_public_artifacts": ["iris", "ecg"],
    }


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    result = run_validation()
    (REPORTS / "real_validation_report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Real validation report",
        "",
        f"Status: {result['status']}",
        "",
        "Real public artifacts:",
        "- Iris image: Wikimedia Commons",
        "- ECG signal: PhysioNet MIT-BIH record 100",
        "",
        "Issues:",
    ]
    lines += [f"- {issue}" for issue in result["issues"]] if result["issues"] else ["- none"]
    (REPORTS / "real_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORTS / "real_validation_report.md")
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

