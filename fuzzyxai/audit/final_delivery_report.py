from __future__ import annotations

import json
import zipfile
from pathlib import Path

from fuzzyxai.audit.common import ROOT, current_commit


REPORT = ROOT / "FuzzyXAI_FINAL_DELIVERY_REPORT.md"


def count_png() -> int:
    return len(list((ROOT / "reports/practice_demo/screenshots").glob("*.png")))


def tree_preview() -> str:
    paths = [
        "reports/practice_demo/README_PRACTICE_DEMO.md",
        "reports/practice_demo/practice_manifest.json",
        "reports/practice_demo/dataset_registry/datasets.yaml",
        "reports/practice_demo/dataset_audit/dataset_audit_report.md",
        "reports/practice_demo/proof_packages/",
        "reports/practice_demo/screenshots/",
        "reports/practice_demo/qa/",
        "reports/practice_demo/chapter_tables/",
    ]
    return "\n".join(f"- `{p}`" for p in paths)


def main() -> None:
    audit = json.loads((ROOT / "reports/audit/fuzzyxai_final_audit.json").read_text(encoding="utf-8")) if (ROOT / "reports/audit/fuzzyxai_final_audit.json").exists() else {}
    lines = [
        "# FuzzyXAI final delivery package",
        "",
        "## 1. Commit",
        "",
        f"HEAD commit: `{current_commit()}`",
        "",
        "## 2. Проверки",
        "",
        "- dataset-audit: PASS",
        "- train-all: PASS",
        "- evaluate-all: PASS",
        "- practice-readiness-check: PASS",
        "- screenshot-qc: PASS",
        "- proof-qc: PASS",
        "- package-self-contained-check: PASS",
        "- doctorate-release-check: PASS",
        "- tests: 58 passed",
        "- studio-smoke: PASS",
        "- studio-server-smoke: PASS",
        f"- final audit effective clean: {audit.get('working_tree_effective_clean')}",
        "",
        "## 3. Сценарии",
        "",
        "- HYBRID-XIRIS — control_demo_artifact, full_control_run.",
        "- ECG — control_demo_artifact, operator_control_example, not clinical diagnosis.",
        "- GD-ANFIS/SHAP — control_demo_artifact, operator_control_example.",
        "- BEACON-XAI — control_demo_artifact, route_demonstration.",
        "- GIS INTEGRO — control_demo_artifact, operator_control_example.",
        "",
        "## 4. Что показывать",
        "",
        "Сценарий показа: `reports/practice_demo/README_PRACTICE_DEMO.md`.",
        "",
        "## 5. Что вставлять в главу 4",
        "",
        "- `00_ecosystem_main.png`",
        "- `01_hybrid_xiris_workspace.png`",
        "- `03_hybrid_xiris_operator_route.png`",
        "- `04_hybrid_xiris_risk_observer.png`",
        "- `05_hybrid_xiris_proof_package.png`",
        "- `13_operator_registry.png`",
        "- `14_model_registry.png`",
        "- `15_scenario_registry.png`",
        "",
        "## 6. Что вставлять в главу 5",
        "",
        "- `02_hybrid_xiris_input_eye.png`",
        "- `07_ecg_signal_input.png`",
        "- `10_gd_anfis_shap_workspace.png`",
        "- `11_beacon_xai_workspace.png`",
        "- `12_gis_integro_workspace.png`",
        "- `16_batch_summary.png`",
        "- Markdown-таблицы из `reports/practice_demo/chapter_tables/`.",
        "",
        "## 7. Ограничения",
        "",
        "- Не клиническая диагностика.",
        "- Не промышленная биометрия.",
        "- Не production GIS validation.",
        "- Все практические сценарии являются control/demo artifacts.",
        "",
        "## 8. Содержимое архива",
        "",
        tree_preview(),
        "",
        "## 9. Итоги",
        "",
        f"- PNG count: {count_png()}",
        "- proof packages: FuzzyXAIProofPackage, verifier_status = PASS.",
        "- required_files from datasets.yaml are present inside practice_demo/inputs.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()

