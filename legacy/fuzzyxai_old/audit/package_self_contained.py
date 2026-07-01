from __future__ import annotations

import json
import zipfile

import yaml

from fuzzyxai.audit.common import ROOT


PRACTICE_ZIP = ROOT / "reports/practice_demo/FuzzyXAI_practice_demo_package.zip"
QA = ROOT / "reports/practice_demo/qa"


def run_check() -> dict[str, object]:
    QA.mkdir(parents=True, exist_ok=True)
    issues: list[str] = []
    with zipfile.ZipFile(PRACTICE_ZIP) as z:
        names = set(z.namelist())
        registry_name = "practice_demo/dataset_registry/datasets.yaml"
        if registry_name not in names:
            issues.append("missing dataset_registry/datasets.yaml")
            registry = {}
        else:
            registry = yaml.safe_load(z.read(registry_name).decode("utf-8")) or {}
        required = [
            "practice_demo/README_PRACTICE_DEMO.md",
            "practice_demo/practice_manifest.json",
            "practice_demo/dataset_audit/dataset_audit_report.md",
            "practice_demo/dataset_audit/dataset_audit_report.json",
            "practice_demo/render_report/figure_to_chapter_mapping.md",
            "practice_demo/qa/QA_SCREENSHOT_REPORT.md",
            "practice_demo/qa/QA_SCREENSHOT_REPORT.json",
            "practice_demo/qa/QA_PROOF_PACKAGE_SCHEMA.md",
        ]
        required += [f"practice_demo/scenario_inputs/{sid}_input.json" for sid in registry]
        required += [f"practice_demo/proof_packages/{sid}_proof_package.json" for sid in registry]
        for entry in registry.values():
            for file_name in entry.get("required_files", []):
                required.append(f"practice_demo/inputs/{file_name}")
        for path in required:
            if path not in names:
                issues.append(f"missing {path}")
        for folder in ["model_cards", "training_reports", "evaluation_reports", "tables", "screenshots", "chapter_tables"]:
            if not any(name.startswith(f"practice_demo/{folder}/") for name in names):
                issues.append(f"empty practice_demo/{folder}/")
        screenshot_count = sum(1 for name in names if name.startswith("practice_demo/screenshots/") and name.endswith(".png"))
        if screenshot_count != 18:
            issues.append(f"screenshot count {screenshot_count} != 18")
    result = {"status": "PASS" if not issues else "FAIL", "issues": issues}
    (QA / "QA_PACKAGE_CONTENTS.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (QA / "QA_DATASET_REQUIRED_FILES.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# QA Package Contents", "", f"Status: {result['status']}", "", "Issues:"]
    lines += [f"- {issue}" for issue in issues] if issues else ["- none"]
    (QA / "QA_PACKAGE_CONTENTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (QA / "QA_DATASET_REQUIRED_FILES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    result = run_check()
    print(QA / "QA_PACKAGE_CONTENTS.md")
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
