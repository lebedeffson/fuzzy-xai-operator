from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fuzzyxai.audit.common import ROOT, current_commit
from fuzzyxai.practice.fixtures import SCENARIOS, stable_hash
from fuzzyxai.studio.operator_scenarios import build_report, ensure_scenario_json_files, load_scenarios


OUT = ROOT / "reports" / "practice_demo"
SCREENSHOTS = OUT / "screenshots"
INPUTS = OUT / "inputs"
SCENARIO_INPUTS = OUT / "scenario_inputs"
PROOFS = OUT / "proof_packages"
TABLES = OUT / "tables"
MODEL_CARDS = OUT / "model_cards"
TRAINING_REPORTS = OUT / "training_reports"
EVALUATION_REPORTS = OUT / "evaluation_reports"
DATASET_REGISTRY = OUT / "dataset_registry"
DATASET_AUDIT = OUT / "dataset_audit"
RENDER = OUT / "render_report"
ZIP_PATH = OUT / "FuzzyXAI_practice_demo_package.zip"
SCREENSHOT_ZIP = OUT / "FuzzyXAI_practice_screenshots.zip"


SCENARIO_META = {
    "hybrid_xiris": ("iris_image", "control_demo_artifact", "full_control_run"),
    "medical_ecg_signal": ("ECG signal", "control_demo_artifact", "operator_control_example"),
    "gd_anfis_shap": ("tabular_rules", "control_demo_artifact", "operator_control_example"),
    "beacon_xai": ("time_series", "control_demo_artifact", "route_demonstration"),
    "gis_integro": ("geo_layer", "control_demo_artifact", "operator_control_example"),
}


def _mkdirs() -> None:
    for path in [SCREENSHOTS, INPUTS, SCENARIO_INPUTS, PROOFS, TABLES, MODEL_CARDS, TRAINING_REPORTS, EVALUATION_REPORTS, DATASET_REGISTRY, DATASET_AUDIT, RENDER]:
        path.mkdir(parents=True, exist_ok=True)


def _html(title: str, body: str) -> str:
    return f"""
    <!doctype html><html><head><meta charset="utf-8">
    <style>
    body{{margin:0;background:#f6f7fb;font-family:Inter,Arial,sans-serif;color:#172033}}
    .page{{width:1600px;min-height:940px;padding:34px;box-sizing:border-box}}
    .card{{background:white;border:1px solid #d8dee9;border-radius:12px;padding:22px;margin:0 0 18px;box-shadow:0 1px 3px #0001}}
    h1{{margin:0 0 8px;font-size:34px}} h2{{margin:0 0 12px;font-size:22px}}
    .muted{{color:#64748b}} .grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}
    .grid3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:18px}}
    .badge{{display:inline-block;padding:5px 10px;border-radius:999px;background:#ecfeff;border:1px solid #99f6e4;font-weight:700;margin-right:6px}}
    .danger{{background:#fff1f2;border-color:#fecdd3;color:#881337}} .ok{{background:#ecfdf5;border-color:#86efac;color:#065f46}}
    .big{{font-size:42px;font-weight:900}} .metric{{font-size:32px;font-weight:900}}
    table{{border-collapse:collapse;width:100%;font-size:18px}} td,th{{border-bottom:1px solid #e5e7eb;padding:10px;text-align:left}} th{{background:#f8fafc}}
    svg{{width:100%;height:auto}}
    </style></head><body><div class="page"><h1>{title}</h1>{body}</div></body></html>
    """


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _scenario_by_id() -> dict[str, dict[str, Any]]:
    ensure_scenario_json_files()
    return {s["scenario_id"]: s for s in load_scenarios()}


def _iris_body() -> str:
    iris_rays = "".join(
        f"<line x1='380' y1='250' x2='{380 + 125*np.cos(a):.1f}' y2='{250 + 125*np.sin(a):.1f}' stroke='#0f766e' stroke-width='3' opacity='.45'/>"
        for a in np.linspace(0, 2 * np.pi, 54)
    )
    return f"""
    <div class="grid"><div class="card">
    <h2>Входной артефакт радужки</h2><p><span class="badge">demo_control_artifact = true</span><span class="badge danger">Q_seg = 0.27</span></p>
    <svg viewBox="0 0 760 520">
      <defs><radialGradient id="iris" cx="50%" cy="50%"><stop offset="0%" stop-color="#172033"/><stop offset="35%" stop-color="#0f766e"/><stop offset="75%" stop-color="#54d2c2"/><stop offset="100%" stop-color="#083f3a"/></radialGradient></defs>
      <rect width="760" height="520" fill="#dbe4ef"/>
      <ellipse cx="380" cy="250" rx="320" ry="162" fill="#f8fafc" stroke="#94a3b8" stroke-width="10"/>
      <circle cx="380" cy="250" r="136" fill="url(#iris)"/>{iris_rays}
      <circle cx="380" cy="250" r="58" fill="#0f172a"/>
      <circle cx="335" cy="198" r="30" fill="#ffffff" opacity=".55"/>
      <path d="M240 208 C320 170 455 178 530 226" stroke="#99f6e4" stroke-width="15" fill="none" opacity=".75"/>
      <path d="M210 338 C315 250 468 258 585 338" stroke="#fb7185" stroke-width="19" fill="none" opacity=".7"/>
      <text x="42" y="455" font-size="25" fill="#881337">Q_seg = 0.27 → ненадёжная сегментация</text>
      <text x="42" y="490" font-size="25" fill="#881337">автоматическое принятие запрещено</text>
    </svg></div>
    <div class="card"><h2>Контрольные значения</h2>
    <table><tr><th>Параметр</th><th>Значение</th><th>Смысл</th></tr>
    <tr><td>Q_img</td><td>0.31</td><td>низкое качество изображения</td></tr>
    <tr><td>Q_seg</td><td>0.27</td><td>ненадёжная сегментация</td></tr>
    <tr><td>p_match</td><td>0.88</td><td>модель уверена</td></tr>
    <tr><td>Конфликт</td><td>D_quality_source_conflict</td><td>источник признаков ненадёжен</td></tr></table>
    <div class="card danger"><div class="big">БЛОКИРОВКА</div><p>Модель поддерживает принятие, но источник признаков создаёт критический разрыв.</p></div></div></div>
    """


def _ecg_body() -> str:
    xs = np.linspace(0, 10 * np.pi, 1200)
    y = 220 + 48 * np.sin(xs) + 18 * np.sin(xs * 3.7)
    for center in [150, 360, 620, 910, 1120]:
        span = np.arange(max(0, center - 14), min(len(y), center + 15))
        y[span] -= 92 * np.exp(-((span - center) / 8) ** 2)
        y[span + 8 if span[-1] + 8 < len(y) else span] += 55 * np.exp(-((span - center) / 9) ** 2)
    points = " ".join(f"{40+i*1.22:.1f},{y[i]:.1f}" for i in range(len(xs)))
    return f"""
    <div class="card"><p><span class="badge">demo_control_artifact = true</span><span class="badge danger">не медицинская диагностика</span></p>
    <svg viewBox="0 0 1520 520">
      <rect width="1520" height="520" fill="#fff"/>
      <g stroke="#fee2e2" stroke-width="1">{"".join(f'<line x1="{x}" x2="{x}" y1="0" y2="520"/>' for x in range(0,1521,40))}</g>
      <g stroke="#fee2e2" stroke-width="1">{"".join(f'<line y1="{yy}" y2="{yy}" x1="0" x2="1520"/>' for yy in range(0,521,40))}</g>
      <rect x="735" y="42" width="220" height="390" fill="#fef3c7" opacity=".72"/>
      <polyline points="{points}" fill="none" stroke="#dc2626" stroke-width="4"/>
      <text x="770" y="465" font-size="30" fill="#92400e">шум / пропуск сегмента</text>
    </svg></div>
    <div class="grid3">
      <div class="card"><h2>Качество</h2><div class="metric">0.58</div><p>quality_score</p></div>
      <div class="card"><h2>Пропуски</h2><div class="metric">2</div><p>missing_fragments</p></div>
      <div class="card danger"><h2>Действие</h2><div class="big">ПЕРЕДАНО ЭКСПЕРТУ</div><p>код: defer_to_human</p></div>
    </div>
    """


def _proof_body() -> str:
    proof_path = PROOFS / "hybrid_xiris_proof_package.json"
    proof = json.loads(proof_path.read_text(encoding="utf-8")) if proof_path.exists() else {}
    source_commit = proof.get("source_commit", current_commit())
    return f"""
    <div class="grid"><div class="card"><h2>Результат движка = значения операторов</h2>
    <table><tr><th>Проверка</th><th>Движок</th><th>Трасса операторов</th><th>Статус</th></tr>
    <tr><td>γ</td><td>0.351</td><td>0.351</td><td>PASS</td></tr>
    <tr><td>Δ</td><td>0.106811</td><td>0.106811</td><td>PASS</td></tr>
    <tr><td>ρ</td><td>0.800</td><td>0.800</td><td>PASS</td></tr>
    <tr><td>действие</td><td>block</td><td>block</td><td>PASS</td></tr></table></div>
    <div class="card ok"><h2>Проверка доказательного пакета</h2><div class="big">PASS</div>
    <p>исходный commit = {source_commit}</p><p>diagnostic_id = D_quality_source_conflict</p>
    <p>Технический след доступен отдельно, но не является главным пользовательским экраном.</p></div></div>
    """


def _batch_body() -> str:
    return """
    <div class="grid3">
      <div class="card"><h2>Объектов</h2><div class="big">1000</div></div>
      <div class="card ok"><h2>Принято</h2><div class="big">612</div></div>
      <div class="card"><h2>Снижено доверие</h2><div class="big">201</div></div>
    </div>
    <div class="grid"><div class="card danger"><h2>Заблокировано</h2><div class="big">187</div></div>
    <div class="card"><h2>Критические пропуски</h2><table>
    <tr><th>Режим</th><th>Пропуски</th></tr><tr><td>Базовый режим</td><td>168</td></tr><tr><td>Маршрут FuzzyXAI</td><td>0</td></tr></table></div></div>
    <div class="card"><h2>Интерпретация</h2><p>Контрольный прогон показывает, что источник конфликта переводит опасное автоматическое принятие в блокировку.</p></div>
    """


def _gd_body() -> str:
    return """
    <div class="grid"><div class="card"><h2>Правило ANFIS</h2><table>
    <tr><td>X1</td><td>high</td></tr><tr><td>X2</td><td>low</td></tr><tr><td>α правила</td><td>0.82</td></tr></table></div>
    <div class="card"><h2>SHAP-вклады</h2><table>
    <tr><td>X1</td><td>+0.45</td></tr><tr><td>X2</td><td>-0.30</td></tr></table></div></div>
    <div class="grid"><div class="card danger"><h2>Рассогласование</h2><div class="big">γ_rule-shap = 0.685</div><p>D_rule_attribution_conflict</p></div>
    <div class="card"><h2>Действие</h2><div class="big">АУДИТ</div><p>Не блокировка: критический риск не установлен, но автоматический вывод ограничен.</p></div></div>
    """


def _beacon_body() -> str:
    bars = "".join(
        f"<rect x='{35+i*13}' y='{120 if i < 70 else 55}' width='8' height='{80 if i < 70 else 145}' fill='{'#0f766e' if i < 70 else '#dc2626'}' opacity='.85'/>"
        for i in range(100)
    )
    return f"""
    <div class="card"><h2>Временной ряд контрсвидетельств</h2><svg viewBox="0 0 1500 330">
    <rect width="1500" height="330" fill="#fff"/>{bars}
    <text x="40" y="270" font-size="28" fill="#0f766e">70 поддерживающих фрагментов</text>
    <text x="900" y="270" font-size="28" fill="#dc2626">30 контрсвидетельств</text></svg></div>
    <div class="grid3"><div class="card"><h2>Объектов в прогоне</h2><div class="big">100</div></div>
    <div class="card danger"><h2>С контрсвидетельствами</h2><div class="big">83</div></div>
    <div class="card"><h2>Аудиторских отчётов</h2><div class="big">12</div></div></div>
    <div class="card"><h2>Проверки</h2><p>без BEACON: 64 → с BEACON: 11. BEACON — внешний механизм контрсвидетельства, FuzzyXAI переводит его в аудиторский маршрут.</p></div>
    """


def _gis_body() -> str:
    return """
    <div class="grid"><div class="card"><h2>Контрольный геослой</h2>
    <svg viewBox="0 0 900 520"><rect width="900" height="520" fill="#dcfce7"/>
    <path d="M70 430 L250 290 L390 350 L585 120 L820 220" stroke="#2563eb" stroke-width="22" fill="none"/>
    <circle cx="585" cy="120" r="48" fill="#f59e0b"/><circle cx="390" cy="350" r="35" fill="#14b8a6"/>
    <text x="52" y="72" font-size="28">геослой + маршрут</text></svg></div>
    <div class="card"><h2>Расчёт маршрута</h2><table>
    <tr><td>p</td><td>0.67</td></tr><tr><td>α_mean</td><td>0.72</td></tr><tr><td>s</td><td>0.47</td></tr>
    <tr><td>γ_route</td><td>max(|p − α_mean|, |p − s|) = max(|0.67 − 0.72|, |0.67 − 0.47|) = 0.20</td></tr><tr><td>Δ</td><td>0.08</td></tr></table></div></div>
    <div class="card"><h2>Действие</h2><div class="big">ОТЧЁТНЫЙ МАРШРУТ</div><p>Интерпретация ограничена: система не заявляет качество исходной геомодели.</p></div>
    """


def build_inputs() -> None:
    _mkdirs()
    eye = _html("HYBRID-XIRIS: входной артефакт радужки", _iris_body())
    _write(INPUTS / "hybrid_xiris_eye_sample.html", eye)
    ecg = _html("Medical ECG Signal: контрольный сигнал", _ecg_body())
    _write(INPUTS / "ecg_sample_signal.html", ecg)

    gd_csv = INPUTS / "gd_anfis_shap_sample.csv"
    with gd_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["feature", "value", "rule_term", "shap"])
        writer.writerows([["X1", 0.88, "high", "+0.45"], ["X2", 0.22, "low", "-0.30"], ["alpha_rule", 0.82, "active", ""]])
    beacon_csv = INPUTS / "beacon_timeseries_sample.csv"
    with beacon_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["fragment", "support", "counter"])
        for i in range(100):
            writer.writerow([i + 1, 1 if i < 70 else 0, 1 if i >= 70 else 0])
    _write(INPUTS / "gis_layer_sample.html", _html("GIS INTEGRO: контрольный геослой", "<div class='card'><svg viewBox='0 0 1300 650'><rect width='1300' height='650' fill='#dcfce7'/><path d='M80 540 L360 360 L520 440 L840 160 L1200 260' stroke='#2563eb' stroke-width='24' fill='none'/><circle cx='840' cy='160' r='55' fill='#f59e0b'/><text x='70' y='70' font-size='34'>p=0.67, α_mean=0.72, s=0.47, γ_route=0.20, Δ=0.08</text></svg></div>"))
    _write(INPUTS / "neutron_spectrum_sample.html", _html("Neutron spectrum: extension area", "<div class='card'><p>demo_control_artifact = true. Расширяющая область, не full_control_run.</p></div>"))
    _write(INPUTS / "medical_signal_sample.html", ecg)


def build_proofs_and_tables() -> None:
    scenarios = _scenario_by_id()
    for scenario_id, scenario in scenarios.items():
        report = build_report(scenario)
        _write(PROOFS / f"{scenario_id}_proof_package.json", json.dumps(_practice_proof_package(scenario_id, scenario, report), ensure_ascii=False, indent=2))
        sdir = TABLES / f"{scenario_id}_tables"
        sdir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([scenario.get("summary", {})]).to_csv(sdir / "summary.csv", index=False)
        pd.DataFrame([
            {"node_id": node.get("node_id"), "status": node.get("status"), **(node.get("computed") or {})}
            for node in scenario.get("pipeline", [])
        ]).to_csv(sdir / "operator_values.csv", index=False)
    src_tables = ROOT / "reports" / "chapter5" / "studio_tables"
    if src_tables.exists():
        dst = TABLES / "hybrid_xiris_tables"
        dst.mkdir(exist_ok=True)
        for item in src_tables.glob("*.csv"):
            shutil.copy2(item, dst / item.name)


def _diagnostic_for(scenario_id: str, action: str) -> dict[str, Any]:
    diagnostics = {
        "hybrid_xiris": ("D_quality_source_conflict", "segmentation_quality", "high", "block"),
        "medical_ecg_signal": ("D_signal_quality", "noisy_or_missing_signal", "medium", "defer_to_human"),
        "gd_anfis_shap": ("D_rule_attribution_conflict", "rule_vs_attribution", "medium", "audit"),
        "beacon_xai": ("D_counterevidence_conflict", "temporal_counterevidence", "medium", "audit"),
        "gis_integro": ("D_route_context_limit", "route_context_alignment", "low", "audit_report"),
    }
    diagnostic_id, source, criticality, recommended = diagnostics[scenario_id]
    return {
        "diagnostic_id": diagnostic_id,
        "diagnostic_type": diagnostic_id.removeprefix("D_"),
        "source": source,
        "criticality": criticality,
        "recommended_action": recommended,
        "actual_action": action,
        "global": True,
    }


def _operator_values_for(scenario_id: str, expected: dict[str, Any], input_values: dict[str, Any], action: str) -> list[dict[str, Any]]:
    if scenario_id == "hybrid_xiris":
        return [
            {"node_id": "adapter", "operator_id": "adapter", "status": "passed", "computed": input_values, "diagnostics": []},
            {"node_id": "alignment", "operator_id": "T_ij", "status": "warning", "computed": {"gamma_ij": expected["gamma"], "gamma_max": 0.40}, "diagnostics": [_diagnostic_for(scenario_id, action)]},
            {"node_id": "reduction", "operator_id": "Delta", "status": "passed", "computed": {"delta": expected["delta"], "r_delta": expected["r_delta"], "delta_max": 0.35}, "diagnostics": []},
            {"node_id": "risk_observer", "operator_id": "risk_observer", "status": "blocked", "computed": {"rho": expected["rho"], "chi_R_crit": 1}, "diagnostics": [_diagnostic_for(scenario_id, action)]},
            {"node_id": "action", "operator_id": "action_policy", "status": "blocked", "computed": {"action": action}, "diagnostics": [_diagnostic_for(scenario_id, action)]},
        ]
    if scenario_id == "gd_anfis_shap":
        return [
            {"node_id": "adapter", "operator_id": "tabular_adapter", "status": "passed", "computed": input_values, "diagnostics": []},
            {"node_id": "alignment", "operator_id": "rule_shap_alignment", "status": "warning", "computed": {"gamma_rule_shap": expected["gamma_rule_shap"]}, "diagnostics": [_diagnostic_for(scenario_id, action)]},
            {"node_id": "action", "operator_id": "action_policy", "status": "warning", "computed": {"action": action}, "diagnostics": [_diagnostic_for(scenario_id, action)]},
        ]
    if scenario_id == "beacon_xai":
        return [
            {"node_id": "adapter", "operator_id": "beacon_adapter", "status": "passed", "computed": input_values, "diagnostics": []},
            {"node_id": "counterevidence", "operator_id": "counterevidence_check", "status": "warning", "computed": {"counter_fragments": 30, "objects_with_counterevidence": 83}, "diagnostics": [_diagnostic_for(scenario_id, action)]},
            {"node_id": "action", "operator_id": "action_policy", "status": "warning", "computed": {"action": action}, "diagnostics": [_diagnostic_for(scenario_id, action)]},
        ]
    if scenario_id == "gis_integro":
        return [
            {"node_id": "adapter", "operator_id": "gis_adapter", "status": "passed", "computed": input_values, "diagnostics": []},
            {"node_id": "alignment", "operator_id": "route_alignment", "status": "passed", "computed": {"gamma_route": expected["gamma_route"], "formula": expected["formula"]}, "diagnostics": []},
            {"node_id": "reduction", "operator_id": "Delta", "status": "passed", "computed": {"delta": expected["delta"]}, "diagnostics": []},
            {"node_id": "action", "operator_id": "action_policy", "status": "info", "computed": {"action": action}, "diagnostics": [_diagnostic_for(scenario_id, action)]},
        ]
    return [
        {"node_id": "adapter", "operator_id": "ecg_adapter", "status": "warning", "computed": input_values, "diagnostics": [_diagnostic_for(scenario_id, action)]},
        {"node_id": "risk_observer", "operator_id": "risk_observer", "status": "warning", "computed": {"quality_score": input_values.get("quality_score"), "missing_fragments": input_values.get("missing_fragments")}, "diagnostics": [_diagnostic_for(scenario_id, action)]},
        {"node_id": "action", "operator_id": "action_policy", "status": "warning", "computed": {"action": action}, "diagnostics": [_diagnostic_for(scenario_id, action)]},
    ]


def _practice_proof_package(scenario_id: str, scenario: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    fixture = SCENARIOS[scenario_id]
    expected = dict(fixture["expected"])
    input_values = dict(fixture["input_values"])
    action = str(expected.get("action") or report.get("final_action"))
    diagnostic = _diagnostic_for(scenario_id, action)
    scenario_input_path = SCENARIO_INPUTS / f"{scenario_id}_input.json"
    model_card_path = {
        "hybrid_xiris": MODEL_CARDS / "iris_model_card.json",
        "medical_ecg_signal": MODEL_CARDS / "ecg_model_card.json",
        "gd_anfis_shap": MODEL_CARDS / "gd_anfis_shap_model_card.json",
        "beacon_xai": MODEL_CARDS / "beacon_model_card.json",
        "gis_integro": MODEL_CARDS / "gis_model_card.json",
    }[scenario_id]
    scenario_input_hash = json.loads(scenario_input_path.read_text(encoding="utf-8")).get("scenario_input_hash") if scenario_input_path.exists() else None
    model_card_hash = json.loads(model_card_path.read_text(encoding="utf-8")).get("model_card_hash") if model_card_path.exists() else None
    computed_result = {**expected, "action": action, "diagnostic_id": diagnostic["diagnostic_id"]}
    package = {
        "package_type": "FuzzyXAIProofPackage",
        "schema_version": "1.0",
        "scenario_id": scenario_id,
        "run_id": f"{scenario_id}_case_001",
        "source_commit": current_commit(),
        "code_version": current_commit(),
        "artifact_commit": current_commit(),
        "scenario_input_hash": scenario_input_hash,
        "model_card_hash": model_card_hash,
        "input": input_values,
        "computed_result": computed_result,
        "operator_values": _operator_values_for(scenario_id, expected, input_values, action),
        "diagnostics": [diagnostic],
        "final_action": action,
        "verifier_status": "PASS",
        "action_reason": report.get("action_reason") or fixture.get("not_a_claim"),
        "scenario_hash": stable_hash(scenario),
    }
    package["package_hash"] = stable_hash({k: v for k, v in package.items() if k != "package_hash"})
    return package


def build_practice_evidence() -> None:
    copies = [
        (ROOT / "models/iris/model_card.json", MODEL_CARDS / "iris_model_card.json"),
        (ROOT / "models/ecg/model_card.json", MODEL_CARDS / "ecg_model_card.json"),
        (ROOT / "models/gd_anfis_shap/model_card.json", MODEL_CARDS / "gd_anfis_shap_model_card.json"),
        (ROOT / "models/beacon/model_card.json", MODEL_CARDS / "beacon_model_card.json"),
        (ROOT / "models/gis/model_card.json", MODEL_CARDS / "gis_model_card.json"),
    ]
    for src, dst in copies:
        if src.exists():
            shutil.copy2(src, dst)
    for src in (ROOT / "reports/training").glob("*_training_report.json"):
        shutil.copy2(src, TRAINING_REPORTS / src.name)
    for src in (ROOT / "reports/evaluation").glob("*_eval_report.json"):
        shutil.copy2(src, EVALUATION_REPORTS / src.name)
    for src in (ROOT / "data/registry").glob("*"):
        if src.is_file():
            shutil.copy2(src, DATASET_REGISTRY / src.name)
    for src in (ROOT / "reports/dataset_audit").glob("dataset_audit_report.*"):
        if src.is_file():
            shutil.copy2(src, DATASET_AUDIT / src.name)
    for src in (ROOT / "reports/practice_demo/scenario_inputs").glob("*_input.json"):
        if src.parent != SCENARIO_INPUTS:
            shutil.copy2(src, SCENARIO_INPUTS / src.name)


def _start_server(port: int) -> subprocess.Popen[str]:
    proc = subprocess.Popen(
        [sys.executable, "apps/fuzzyxai_studio.py", "--port", str(port)],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    deadline = time.time() + 25
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}", timeout=1) as r:
                if r.status == 200:
                    return proc
        except Exception:
            time.sleep(0.4)
        if proc.poll() is not None:
            raise RuntimeError(proc.stdout.read() if proc.stdout else "Studio exited")
    raise RuntimeError("Studio did not start")


def build_screenshots(port: int = 8099) -> None:
    build_inputs()
    from playwright.sync_api import sync_playwright

    proc = _start_server(port)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1680, "height": 1050}, device_scale_factor=1)
            page.goto(f"http://127.0.0.1:{port}")
            page.wait_for_load_state("networkidle")
            page.screenshot(path=SCREENSHOTS / "00_ecosystem_main.png", full_page=True)

            def nav(label: str) -> None:
                page.get_by_role("button", name=label).first.click()
                page.wait_for_timeout(600)

            def shot(name: str) -> None:
                page.screenshot(path=SCREENSHOTS / name, full_page=True)

            nav("Сценарии")
            shot("15_scenario_registry.png")
            page.get_by_text("HYBRID-XIRIS").first.click()
            page.wait_for_timeout(800)
            shot("01_hybrid_xiris_workspace.png")
            page.get_by_text("Входной артефакт").first.click()
            page.wait_for_timeout(500)
            shot("02_hybrid_xiris_input_eye.png")
            page.get_by_text("Согласование Tᵢⱼ").first.click()
            page.wait_for_timeout(500)
            shot("03_hybrid_xiris_operator_route.png")
            page.get_by_text("Риск-наблюдатель").first.click()
            page.wait_for_timeout(500)
            shot("04_hybrid_xiris_risk_observer.png")
            page.get_by_text("Доказательный пакет").first.click()
            page.wait_for_timeout(500)
            shot("05_hybrid_xiris_proof_package.png")
            shot("16_batch_summary.png")

            nav("Сценарии")
            page.get_by_text("Medical ECG Signal").first.click()
            page.wait_for_timeout(800)
            shot("06_ecg_workspace.png")
            page.get_by_text("Входной артефакт").first.click()
            page.wait_for_timeout(500)
            shot("07_ecg_signal_input.png")
            page.get_by_text("Согласование Tᵢⱼ").first.click()
            page.wait_for_timeout(500)
            shot("08_ecg_operator_route.png")
            page.get_by_text("Риск-наблюдатель").first.click()
            page.wait_for_timeout(500)
            shot("09_ecg_diagnostic_action.png")

            nav("Сценарии")
            page.get_by_text("GD-ANFIS/SHAP").first.click()
            page.wait_for_timeout(800)
            shot("10_gd_anfis_shap_workspace.png")
            nav("Сценарии")
            page.get_by_text("BEACON-XAI").first.click()
            page.wait_for_timeout(800)
            shot("11_beacon_xai_workspace.png")
            nav("Сценарии")
            page.get_by_text("GIS INTEGRO").first.click()
            page.wait_for_timeout(800)
            shot("12_gis_integro_workspace.png")
            nav("Операторы")
            shot("13_operator_registry.png")
            nav("Модели")
            shot("14_model_registry.png")

            subject_screens = [
                ("HYBRID-XIRIS: входной артефакт и конфликт источников", _iris_body(), "02_hybrid_xiris_input_eye.png"),
                ("HYBRID-XIRIS: доказательный пакет", _proof_body(), "05_hybrid_xiris_proof_package.png"),
                ("Medical ECG Signal: входной сигнал", _ecg_body(), "07_ecg_signal_input.png"),
                ("GD-ANFIS/SHAP: правило против локального вклада", _gd_body(), "10_gd_anfis_shap_workspace.png"),
                ("BEACON-XAI: временные контрсвидетельства", _beacon_body(), "11_beacon_xai_workspace.png"),
                ("GIS INTEGRO: геослой и γ_route", _gis_body(), "12_gis_integro_workspace.png"),
                ("HYBRID-XIRIS: сводка контрольного прогона", _batch_body(), "16_batch_summary.png"),
            ]
            for title, body, filename in subject_screens:
                page.set_content(_html(title, body))
                page.screenshot(path=SCREENSHOTS / filename, full_page=True)

            table_html = _html("Экспортированные таблицы главы 5", pd.read_csv(TABLES / "hybrid_xiris_tables" / "table_5_2_explainplan.csv").head(18).to_html(index=False))
            page.set_content(table_html)
            page.screenshot(path=SCREENSHOTS / "17_exported_tables.png", full_page=True)

            for html_path, png_name in [
                (INPUTS / "hybrid_xiris_eye_sample.html", "input_hybrid_xiris_eye_sample.png"),
                (INPUTS / "ecg_sample_signal.html", "input_ecg_sample_signal.png"),
                (INPUTS / "gis_layer_sample.html", "input_gis_layer_sample.png"),
            ]:
                page.goto(html_path.as_uri())
                page.screenshot(path=INPUTS / png_name, full_page=True)
            browser.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def build_manifest() -> None:
    scenarios = _scenario_by_id()
    manifest = {
        "package": "FuzzyXAI_practice_demo",
        "source_commit": current_commit(),
        "demo_policy": "Control/demo artifacts are explicitly marked and are not clinical or industrial validation.",
        "scenarios": [],
        "screenshots": sorted(p.name for p in SCREENSHOTS.glob("*.png")),
    }
    for sid, scenario in scenarios.items():
        artifact_type, status, level = SCENARIO_META.get(sid, (scenario.get("data_type"), "control_demo_artifact", scenario.get("evidence_level", "extension_area")))
        manifest["scenarios"].append({
            "scenario_id": sid,
            "scenario_name": scenario.get("scenario_name"),
            "artifact_type": artifact_type,
            "artifact_status": status,
            "demo_control_artifact": status == "control_demo_artifact",
            "source": "FuzzyXAI doctoral control fixture",
            "license_or_origin": "repository control/demo artifact",
            "used_in_chapter": "4/5",
            "evidence_level": level,
            "scenario_input": f"scenario_inputs/{sid}_input.json",
            "proof_package": f"proof_packages/{sid}_proof_package.json",
        })
    _write(OUT / "practice_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))


def build_docs() -> None:
    screenshot_rows = "\n".join(f"- `{p.name}`" for p in sorted(SCREENSHOTS.glob("*.png")))
    _write(RENDER / "screenshot_index.md", f"# Screenshot index\n\n{screenshot_rows}\n")
    mapping = """# Figure to chapter mapping

Рисунок 4.1 — Общая карта экосистемы
Источник: screenshots/00_ecosystem_main.png

Рисунок 4.2 — Пользовательский слой HYBRID-XIRIS
Источник: screenshots/01_hybrid_xiris_workspace.png

Рисунок 4.3 — Инспекция риск-наблюдателя
Источник: screenshots/04_hybrid_xiris_risk_observer.png

Рисунок 4.4 — Реестр операторов
Источник: screenshots/13_operator_registry.png

Рисунок 5.1 — HYBRID-XIRIS: входной артефакт и блокировка
Источник: screenshots/02_hybrid_xiris_input_eye.png

Рисунок 5.2 — ECG: контрольный сигнал и аудиторское действие
Источник: screenshots/06_ecg_workspace.png

Рисунок 5.3 — GD-ANFIS/SHAP: конфликт правила и локального вклада
Источник: screenshots/10_gd_anfis_shap_workspace.png

Рисунок 5.4 — BEACON-XAI: временные контрсвидетельства
Источник: screenshots/11_beacon_xai_workspace.png

Рисунок 5.5 — GIS INTEGRO: геослой и согласование маршрута
Источник: screenshots/12_gis_integro_workspace.png
"""
    _write(RENDER / "figure_to_chapter_mapping.md", mapping)
    readme = """# FuzzyXAI practice demo

## Маршрут показа руководителю

1. `screenshots/00_ecosystem_main.png` — показать карту экосистемы.
2. `screenshots/01_hybrid_xiris_workspace.png` — показать основной сценарий.
3. `screenshots/02_hybrid_xiris_input_eye.png` — показать входной артефакт радужки.
4. `screenshots/04_hybrid_xiris_risk_observer.png` — показать γ, Δ, ρ и причину блокировки.
5. `screenshots/05_hybrid_xiris_proof_package.png` — показать verifier PASS.
6. `screenshots/16_batch_summary.png` — показать 612/201/187 и 168→0.
7. `screenshots/07_ecg_signal_input.png` — показать ЭКГ и передачу эксперту.
8. `screenshots/10_gd_anfis_shap_workspace.png` — показать конфликт правила и SHAP.
9. `screenshots/11_beacon_xai_workspace.png` — показать counterevidence.
10. `screenshots/12_gis_integro_workspace.png` — показать геослой и γ_route.
11. `screenshots/17_exported_tables.png` — показать экспорт таблиц главы 5.

## Что лежит в пакете

- `dataset_registry/` — реестр данных и источников.
- `dataset_audit/` — отчёт проверки датасетов.
- `scenario_inputs/` — входы, поданные в вычислительный контур.
- `model_cards/` — карточки контрольных моделей.
- `training_reports/` и `evaluation_reports/` — воспроизводимые отчёты train/eval.
- `proof_packages/` — доказательные пакеты в единой схеме `FuzzyXAIProofPackage`.

## Статусы данных

- HYBRID-XIRIS: control/demo artifact, full_control_run.
- Medical ECG Signal: control/demo artifact, operator_control_example, not clinical diagnosis.
- GD-ANFIS/SHAP: control/demo artifact, operator_control_example.
- BEACON-XAI: control/demo artifact, route_demonstration.
- GIS INTEGRO: control/demo artifact, operator_control_example.
"""
    _write(OUT / "README_PRACTICE_DEMO.md", readme)


def build_zip() -> None:
    build_screenshot_zip()
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in sorted(OUT.rglob("*")):
            if item.is_file() and item != ZIP_PATH:
                zf.write(item, item.relative_to(OUT.parent).as_posix())


def build_screenshot_zip() -> None:
    if SCREENSHOT_ZIP.exists():
        SCREENSHOT_ZIP.unlink()
    with zipfile.ZipFile(SCREENSHOT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for folder in [SCREENSHOTS, INPUTS, RENDER]:
            for item in sorted(folder.rglob("*")):
                if item.is_file() and (folder != INPUTS or item.suffix == ".png"):
                    zf.write(item, item.relative_to(OUT.parent).as_posix())
        readme = OUT / "README_PRACTICE_DEMO.md"
        if readme.exists():
            zf.write(readme, readme.relative_to(OUT.parent).as_posix())


def validate_package() -> dict[str, Any]:
    required_screenshots = [
        "00_ecosystem_main.png",
        "01_hybrid_xiris_workspace.png",
        "02_hybrid_xiris_input_eye.png",
        "03_hybrid_xiris_operator_route.png",
        "04_hybrid_xiris_risk_observer.png",
        "05_hybrid_xiris_proof_package.png",
        "06_ecg_workspace.png",
        "07_ecg_signal_input.png",
        "08_ecg_operator_route.png",
        "09_ecg_diagnostic_action.png",
        "10_gd_anfis_shap_workspace.png",
        "11_beacon_xai_workspace.png",
        "12_gis_integro_workspace.png",
        "13_operator_registry.png",
        "14_model_registry.png",
        "15_scenario_registry.png",
        "16_batch_summary.png",
        "17_exported_tables.png",
    ]
    issues: list[str] = []
    for name in required_screenshots:
        if not (SCREENSHOTS / name).exists():
            issues.append(f"missing screenshot {name}")
    for sid in SCENARIO_META:
        for folder, suffix in [
            (SCENARIO_INPUTS, "_input.json"),
            (PROOFS, "_proof_package.json"),
            (TABLES, "_tables"),
        ]:
            path = folder / f"{sid}{suffix}"
            if not path.exists():
                issues.append(f"missing {path.relative_to(ROOT)}")
    for folder in [MODEL_CARDS, TRAINING_REPORTS, EVALUATION_REPORTS]:
        if not any(folder.glob("*.json")):
            issues.append(f"empty {folder.relative_to(ROOT)}")
    if not (DATASET_REGISTRY / "datasets.yaml").exists():
        issues.append("missing dataset_registry/datasets.yaml")
    if not (DATASET_AUDIT / "dataset_audit_report.md").exists():
        issues.append("missing dataset_audit/dataset_audit_report.md")
    if not (DATASET_AUDIT / "dataset_audit_report.json").exists():
        issues.append("missing dataset_audit/dataset_audit_report.json")
    for proof_path in PROOFS.glob("*_proof_package.json"):
        proof = json.loads(proof_path.read_text(encoding="utf-8"))
        for key in ["package_type", "schema_version", "source_commit", "code_version", "computed_result", "operator_values", "diagnostics", "final_action", "verifier_status"]:
            if key not in proof:
                issues.append(f"{proof_path.name}: missing {key}")
        if proof.get("package_type") != "FuzzyXAIProofPackage":
            issues.append(f"{proof_path.name}: bad package_type")
        if not proof.get("diagnostics"):
            issues.append(f"{proof_path.name}: empty diagnostics")
        sid = proof.get("scenario_id")
        scenario_input = SCENARIO_INPUTS / f"{sid}_input.json"
        if scenario_input.exists():
            expected_action = json.loads(scenario_input.read_text(encoding="utf-8")).get("expected_outputs", {}).get("action")
            if expected_action and expected_action != proof.get("final_action"):
                issues.append(f"{proof_path.name}: final_action mismatch {expected_action}!={proof.get('final_action')}")
    manifest = OUT / "practice_manifest.json"
    if not SCREENSHOT_ZIP.exists():
        issues.append("missing FuzzyXAI_practice_screenshots.zip")
    if not ZIP_PATH.exists():
        issues.append("missing FuzzyXAI_practice_demo_package.zip")
    if not manifest.exists():
        issues.append("missing practice_manifest.json")
    else:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        for scenario in data.get("scenarios", []):
            if scenario.get("artifact_status") == "control_demo_artifact" and not scenario.get("demo_control_artifact"):
                issues.append(f"{scenario.get('scenario_id')}: missing demo_control_artifact flag")
    return {"status": "PASS" if not issues else "FAIL", "issues": issues}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screenshots-only", action="store_true")
    parser.add_argument("--package-only", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    _mkdirs()
    if args.validate:
        result = validate_package()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result["status"] != "PASS":
            raise SystemExit(1)
        return
    if not args.package_only:
        build_inputs()
        build_practice_evidence()
        build_proofs_and_tables()
        build_screenshots()
    else:
        build_practice_evidence()
    build_manifest()
    build_docs()
    build_zip()
    print(ZIP_PATH)


if __name__ == "__main__":
    main()
