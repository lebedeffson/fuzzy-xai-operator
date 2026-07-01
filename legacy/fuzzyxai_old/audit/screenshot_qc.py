from __future__ import annotations

import json
import struct
from pathlib import Path

from fuzzyxai.audit.common import ROOT


OUT = ROOT / "reports/practice_demo"
QA = OUT / "qa"
SCREENSHOTS = OUT / "screenshots"


EXPECTATIONS = {
    "00_ecosystem_main.png": ("Глава 4, рисунок 4.1", "карта экосистемы, контейнеры, русский пользовательский слой"),
    "01_hybrid_xiris_workspace.png": ("Глава 4", "основной сценарий HYBRID-XIRIS, маршрут, инспектор, действие"),
    "02_hybrid_xiris_input_eye.png": ("Глава 5", "радужка, Q_img, Q_seg, p_match, D_quality_source_conflict, БЛОКИРОВКА"),
    "03_hybrid_xiris_operator_route.png": ("Глава 4", "Вход, адаптер, E_k/D_k, T_ij, F, Δ, risk observer, action"),
    "04_hybrid_xiris_risk_observer.png": ("Глава 4/5", "γ, Δ, r_Δ, ρ, χ_R^crit, action block"),
    "05_hybrid_xiris_proof_package.png": ("Глава 4", "computed_result = operator_values, verifier PASS"),
    "06_ecg_workspace.png": ("Глава 5", "ECG operator_control_example, not clinical diagnosis"),
    "07_ecg_signal_input.png": ("Глава 5", "ECG graph, noise/missing, quality_score, defer_to_human"),
    "08_ecg_operator_route.png": ("Глава 4/app", "ECG route, D_signal_quality, proof package"),
    "09_ecg_diagnostic_action.png": ("Глава 5", "D_signal_quality, reason, action defer_to_human"),
    "10_gd_anfis_shap_workspace.png": ("Глава 5", "ANFIS rule, SHAP, γ_rule-shap, audit"),
    "11_beacon_xai_workspace.png": ("Глава 5", "100 fragments, 70 support, 30 counter, 83 objects, 12 audit reports"),
    "12_gis_integro_workspace.png": ("Глава 5", "geolayer, p, α_mean, s, γ_route, Δ"),
    "13_operator_registry.png": ("Глава 4", "operator registry"),
    "14_model_registry.png": ("Глава 4", "model cards and model status"),
    "15_scenario_registry.png": ("Глава 4", "scenario registry and evidence levels"),
    "16_batch_summary.png": ("Глава 5", "1000, 612, 201, 187, 168→0"),
    "17_exported_tables.png": ("Приложение", "technical exported tables screen"),
}


def png_width(path: Path) -> int:
    with path.open("rb") as fh:
        sig = fh.read(24)
    if sig[:8] != b"\x89PNG\r\n\x1a\n":
        return 0
    return struct.unpack(">I", sig[16:20])[0]


def run_qc() -> dict[str, object]:
    QA.mkdir(parents=True, exist_ok=True)
    rows = []
    issues = []
    for name, (chapter, expectation) in EXPECTATIONS.items():
        path = SCREENSHOTS / name
        exists = path.exists()
        width = png_width(path) if exists else 0
        status = "PASS" if exists and width >= 1600 else "FAIL"
        if status != "PASS":
            issues.append(f"{name}: exists={exists}, width={width}")
        rows.append({
            "file": name,
            "chapter_status": chapter,
            "expected_visible_content": expectation,
            "exists": exists,
            "width_px": width,
            "status": status,
            "limitation": "17_exported_tables.png is technical; use CSV/Markdown tables in chapter text." if name == "17_exported_tables.png" else "",
        })
    result = {"status": "PASS" if not issues else "FAIL", "issues": issues, "screenshots": rows}
    (QA / "QA_SCREENSHOT_REPORT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# QA Screenshot Report", "", f"Status: {result['status']}", ""]
    lines += ["| file | chapter | status | expected content |", "|---|---|---|---|"]
    for row in rows:
        lines.append(f"| `{row['file']}` | {row['chapter_status']} | {row['status']} | {row['expected_visible_content']} |")
    (QA / "QA_SCREENSHOT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    result = run_qc()
    print(QA / "QA_SCREENSHOT_REPORT.md")
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

